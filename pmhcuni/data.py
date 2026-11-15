"""
Dataset and DataModule for peptide-MHC eluted ligand prediction.

Uses the ESM C tokenizer (EsmSequenceTokenizer) from evolutionaryscale/esm.
"""

from __future__ import annotations

import csv
import math
import random
import itertools
from typing import Any, Callable, List, Optional
from collections import OrderedDict

import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from esm.tokenization import EsmSequenceTokenizer


# ──────────────────────────────────────────────────
# HLA pseudo-sequence definitions (minimal example)
# In practice, import from your own allele sequence file.
# ──────────────────────────────────────────────────
MIN_LEN = 346  # pseudo-sequence truncation length (per chain for class II)


def _resolve_mhc_tokens(
    mhc_name: str,
    allele_sequences: OrderedDict[str, str],
    tokenizer: EsmSequenceTokenizer,
    chain_break_id: int,
    max_mhc_tokens: int | None = None,
    pad_id: int | None = None,
) -> list[int]:
    """
    Resolve MHC name to token list, supporting both single-chain (class I)
    and two-chain (class II) alleles.

    Class II alleles use "/" separator (e.g. "HLA-DPA1*01:03/DPB1*04:01").
    The chain_seqs file stores individual chains; this function looks up
    each chain, truncates to MIN_LEN, and concatenates with a chain_break token.

    If max_mhc_tokens is given, pads or truncates to that length.
    """
    if "/" in mhc_name:
        # Two-chain (class II): "alpha_name/beta_name"
        chain_a, chain_b = [c.strip() for c in mhc_name.split("/", 1)]
        seq_a = allele_sequences.get(chain_a, "")[:MIN_LEN]
        seq_b = allele_sequences.get(chain_b, "")[:MIN_LEN]
        if not seq_a or not seq_b:
            # Try with "HLA-" prefix
            if not seq_a and not chain_a.startswith("HLA-"):
                seq_a = allele_sequences.get(f"HLA-{chain_a}", "")[:MIN_LEN]
            if not seq_b and not chain_b.startswith("HLA-"):
                seq_b = allele_sequences.get(f"HLA-{chain_b}", "")[:MIN_LEN]
        tokens_a = tokenizer.encode(seq_a, add_special_tokens=False) if seq_a else []
        tokens_b = tokenizer.encode(seq_b, add_special_tokens=False) if seq_b else []
        mhc_tokens = tokens_a + [chain_break_id] + tokens_b
    else:
        # Single-chain (class I) or direct lookup
        seq = allele_sequences.get(mhc_name, "")[:MIN_LEN]
        if not seq:
            # Try "<name> alpha" / "<name> beta" pattern
            alpha_key = f"{mhc_name} alpha"
            beta_key = f"{mhc_name} beta"
            if alpha_key in allele_sequences and beta_key in allele_sequences:
                seq_a = allele_sequences[alpha_key][:MIN_LEN]
                seq_b = allele_sequences[beta_key][:MIN_LEN]
                tokens_a = tokenizer.encode(seq_a, add_special_tokens=False)
                tokens_b = tokenizer.encode(seq_b, add_special_tokens=False)
                mhc_tokens = tokens_a + [chain_break_id] + tokens_b
            else:
                mhc_tokens = []
        else:
            mhc_tokens = tokenizer.encode(seq, add_special_tokens=False)

    # Pad or truncate to max_mhc_tokens if specified
    if max_mhc_tokens is not None and pad_id is not None:
        if len(mhc_tokens) > max_mhc_tokens:
            mhc_tokens = mhc_tokens[:max_mhc_tokens]
        elif len(mhc_tokens) < max_mhc_tokens:
            mhc_tokens = mhc_tokens + [pad_id] * (max_mhc_tokens - len(mhc_tokens))

    return mhc_tokens


def _detect_two_chain(allele_sequences: OrderedDict[str, str], mhc_names: list[str] | None = None) -> bool:
    """Detect if we're in two-chain (class II) mode."""
    if mhc_names:
        if any("/" in name for name in mhc_names):
            return True
    # Check if allele_sequences has "alpha"/"beta" entries
    for name in allele_sequences:
        if name.endswith(" alpha") or name.endswith(" beta"):
            return True
    return False


def _compute_max_mhc_tokens(two_chain: bool) -> int:
    """Compute max MHC token length (with chain_break for two-chain)."""
    if two_chain:
        return 2 * MIN_LEN + 1  # alpha + chain_break + beta
    return MIN_LEN


class DataView:
    """Lazy view over a split of the dataset."""

    def __init__(self, data: list[dict], process: Callable, split: str):
        self.data = data
        self.process = process
        self.split = split

    def __getitem__(self, index: int) -> Any:
        return self.process(self.data[index], split=self.split)

    def __len__(self) -> int:
        return len(self.data)


class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    Two-pool balanced batch sampler for imbalanced binary classification.

    Pre-builds positive and negative index pools from the label list, then for
    each batch draws ``n_pos`` positives and ``n_neg`` negatives.  Positives
    are drawn **without replacement** within an epoch (a random permutation is
    consumed in order; when exhausted it is reshuffled and the next pass
    begins, so each positive appears at most once before any is repeated).
    Negatives are drawn **with replacement** to ensure a stable supply.  This
    achieves approximately ``pos_frac_target`` positives per batch and avoids
    the ``torch.multinomial`` 2^24-category limit that causes
    ``WeightedRandomSampler`` to crash on large datasets.

    DDP behaviour: each rank generates the *same* global batch index list
    (identical seed per epoch), then slices out its own shard of size
    ``batch_size // num_ranks``.  This mirrors ``DistributedSampler`` so
    every sample is covered without duplicates across ranks.  The ``rank``
    and ``num_ranks`` values should be updated before each epoch via
    ``set_epoch`` (PyTorch Lightning calls this automatically).  Because the
    sampler already handles distribution, the DataLoader must be created with
    ``shuffle=False`` and no additional ``DistributedSampler``; inform
    Lightning by returning the DataLoader with ``use_distributed_sampler=False``
    or by setting the DataModule's ``_log_hyperparams`` flag (see usage in
    ``PmhcuniDataset.train_dataloader``).

    Args:
        labels: Binary label list/sequence (1 = positive, 0 = negative).
        batch_size: *Global* total samples per batch (across all ranks).
        pos_frac_target: Target fraction of positives per batch (default 0.2).
        num_batches: Number of batches (steps) per epoch.  If ``None``,
            defaults to ``ceil(len(labels) / batch_size)``.
        seed: Base random seed.  The effective seed per epoch is
            ``seed + epoch``.
        rank: Process rank for distributed training (default 0).
        num_ranks: Total number of processes (default 1).
    """

    def __init__(
        self,
        labels: list,
        batch_size: int,
        pos_frac_target: float = 0.2,
        num_batches: Optional[int] = None,
        seed: int = 0,
        rank: int = 0,
        num_ranks: int = 1,
    ):
        self.batch_size = batch_size          # per-rank local batch size
        self.global_batch_size = batch_size * max(num_ranks, 1)
        self.pos_frac_target = pos_frac_target
        self.seed = seed
        self.rank = rank
        self.num_ranks = num_ranks
        self._epoch = 0

        # Build positive and negative index pools
        self._pos_indices = [i for i, lbl in enumerate(labels) if lbl > 0.5]
        self._neg_indices = [i for i, lbl in enumerate(labels) if lbl <= 0.5]

        if not self._pos_indices:
            raise ValueError("BalancedBatchSampler: no positive samples found.")
        if not self._neg_indices:
            raise ValueError("BalancedBatchSampler: no negative samples found.")

        # n_pos / n_neg are defined over the global batch
        self.n_pos = max(1, round(self.global_batch_size * pos_frac_target))
        self.n_neg = self.global_batch_size - self.n_pos

        if num_batches is None:
            num_batches = math.ceil(len(self._pos_indices) / self.n_pos)
        self.num_batches = num_batches

    # ------------------------------------------------------------------
    # Lightning / DistributedSampler compatibility
    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for reproducible per-epoch randomisation."""
        self._epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        # All ranks share the same seed → same global batch order.
        # Each rank then slices its own contiguous shard from each global batch.
        rng = random.Random(self.seed + self._epoch)
        # Auto-advance epoch so successive calls without set_epoch still differ
        self._epoch += 1

        # Positive pool: without replacement within epoch; reshuffle when exhausted
        pos_pool = list(self._pos_indices)
        rng.shuffle(pos_pool)
        pos_cursor = 0

        for _ in range(self.num_batches):
            # Draw n_pos positives without replacement; reshuffle pool when depleted
            pos_batch = []
            while len(pos_batch) < self.n_pos:
                if pos_cursor >= len(pos_pool):
                    rng.shuffle(pos_pool)
                    pos_cursor = 0
                take = min(self.n_pos - len(pos_batch), len(pos_pool) - pos_cursor)
                pos_batch.extend(pos_pool[pos_cursor:pos_cursor + take])
                pos_cursor += take

            # Draw n_neg negatives with replacement (large pool, stable supply)
            neg_batch = rng.choices(self._neg_indices, k=self.n_neg)

            global_batch = pos_batch + neg_batch
            rng.shuffle(global_batch)

            # Slice this rank's shard from the global batch
            start = self.rank * self.batch_size
            end = start + self.batch_size
            yield global_batch[start:end]


class AlleleBalancedBatchSampler(torch.utils.data.Sampler):
    """
    Two-level balanced batch sampler: first selects alleles weighted by
    sqrt(frequency), then samples pos/neg within each allele.

    This ensures low-frequency alleles get more training exposure compared
    to global-level balanced sampling, while not completely starving
    high-frequency alleles (sqrt weighting as compromise).

    Each batch:
      1. Sample an allele from the allele pool (weighted by sqrt(train_count))
      2. From that allele, draw n_pos positives and n_neg negatives
      3. Repeat until batch is full

    Args:
        labels: Binary label list (1=positive, 0=negative).
        mhc_names: MHC allele name per sample (aligned with labels).
        batch_size: Global batch size.
        pos_frac_target: Target positive fraction per batch.
        num_batches: Steps per epoch. None = auto.
        seed: Random seed.
        rank / num_ranks: DDP support.
    """

    def __init__(
        self,
        labels: list,
        mhc_names: list[str],
        batch_size: int,
        pos_frac_target: float = 0.15,
        num_batches: Optional[int] = None,
        seed: int = 0,
        rank: int = 0,
        num_ranks: int = 1,
    ):
        self.batch_size = batch_size
        self.global_batch_size = batch_size * max(num_ranks, 1)
        self.pos_frac_target = pos_frac_target
        self.seed = seed
        self.rank = rank
        self.num_ranks = num_ranks
        self._epoch = 0

        self.n_pos = max(1, round(self.global_batch_size * pos_frac_target))
        self.n_neg = self.global_batch_size - self.n_pos

        # Build per-allele positive/negative index pools
        from collections import defaultdict
        allele_pos: dict[str, list[int]] = defaultdict(list)
        allele_neg: dict[str, list[int]] = defaultdict(list)
        for i, (lbl, mhc) in enumerate(zip(labels, mhc_names)):
            if lbl > 0.5:
                allele_pos[mhc].append(i)
            else:
                allele_neg[mhc].append(i)

        # Only keep alleles that have both positives and negatives
        self._alleles = sorted(
            a for a in allele_pos if a in allele_neg
        )
        self._allele_pos = {a: allele_pos[a] for a in self._alleles}
        self._allele_neg = {a: allele_neg[a] for a in self._alleles}

        # Allele sampling weights: sqrt(total_count) for each allele
        counts = {a: len(allele_pos[a]) + len(allele_neg[a]) for a in self._alleles}
        raw_weights = [math.sqrt(counts[a]) for a in self._alleles]
        total_w = sum(raw_weights)
        self._allele_weights = [w / total_w for w in raw_weights]

        # Total positives for num_batches estimation
        total_pos = sum(len(v) for v in self._allele_pos.values())
        if num_batches is None:
            num_batches = math.ceil(total_pos / self.n_pos)
        self.num_batches = num_batches

        n_alleles = len(self._alleles)
        print(
            f"AlleleBalancedBatchSampler: {n_alleles} alleles, "
            f"{self.num_batches} batches/epoch, "
            f"pos_frac={pos_frac_target:.2f}"
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        # Per-allele positive cursors (without-replacement within epoch)
        pos_pools = {a: list(idxs) for a, idxs in self._allele_pos.items()}
        for pool in pos_pools.values():
            rng.shuffle(pool)
        pos_cursors = {a: 0 for a in self._alleles}

        for _ in range(self.num_batches):
            pos_batch = []
            neg_batch = []

            # Fill positive slots by sampling alleles
            while len(pos_batch) < self.n_pos:
                # Pick an allele (weighted by sqrt freq)
                allele = rng.choices(self._alleles, weights=self._allele_weights, k=1)[0]

                # Draw one positive from that allele (without replacement)
                pool = pos_pools[allele]
                cursor = pos_cursors[allele]
                if cursor >= len(pool):
                    rng.shuffle(pool)
                    cursor = 0
                pos_batch.append(pool[cursor])
                pos_cursors[allele] = cursor + 1

            # Fill negative slots by sampling alleles proportionally
            while len(neg_batch) < self.n_neg:
                allele = rng.choices(self._alleles, weights=self._allele_weights, k=1)[0]
                neg_pool = self._allele_neg[allele]
                if neg_pool:
                    neg_batch.append(rng.choice(neg_pool))

            global_batch = pos_batch + neg_batch
            rng.shuffle(global_batch)

            start = self.rank * self.batch_size
            end = start + self.batch_size
            yield global_batch[start:end]


class BAGroupedBatchSampler(torch.utils.data.Sampler):
    """
    Grouped-allele batch sampler for BA (affinity) specialist training with the
    per-allele pairwise RankNet loss.

    Rank losses need many same-allele pairs *within a batch*: a naively-shuffled
    BA batch has almost no intra-allele pairs, making the ranking signal look
    falsely weak. This sampler builds each batch from
    ``R = max(1, batch_size // k_per_allele)`` alleles, each contributing exactly
    ``k_per_allele`` affinity-labeled rows drawn QUANTILE-STRATIFIED by affinity
    (so every group spans the allele's full affinity range → strong pair signal).

    Only rows with a real (non-None/NaN) affinity are eligible; alleles with
    fewer than ``k_per_allele`` eligible rows are dropped. Alleles are picked
    weighted by ``sqrt(count)``. DDP-sharded exactly like
    :class:`AlleleBalancedBatchSampler` (each rank yields its contiguous slice).

    Args:
        affinities: per-dataset-row affinity (None/NaN if the row has no BA label).
        mhc_names:  MHC allele name per row (aligned with ``affinities``).
        batch_size: per-rank batch size.
        k_per_allele: rows drawn per allele per batch.
        num_batches: steps per epoch. None = ceil(total_eligible / global_batch).
        seed / rank / num_ranks: DDP support.
    """

    def __init__(
        self,
        affinities: list,
        mhc_names: list,
        batch_size: int,
        k_per_allele: int = 8,
        num_batches: Optional[int] = None,
        seed: int = 0,
        rank: int = 0,
        num_ranks: int = 1,
    ):
        self.batch_size = batch_size
        self.global_batch_size = batch_size * max(num_ranks, 1)
        self.k_per_allele = k_per_allele
        self.seed = seed
        self.rank = rank
        self.num_ranks = num_ranks
        self._epoch = 0

        # R alleles per (global) batch; each contributes k rows.
        self.R = max(1, self.global_batch_size // k_per_allele)

        from collections import defaultdict
        allele_rows: dict = defaultdict(list)
        for i, (aff, mhc) in enumerate(zip(affinities, mhc_names)):
            if aff is None:
                continue
            try:
                fa = float(aff)
            except (ValueError, TypeError):
                continue
            if math.isnan(fa):
                continue
            allele_rows[mhc].append((i, fa))

        # Keep only alleles with >= k_per_allele eligible rows.
        self._alleles = sorted(
            a for a, rows in allele_rows.items() if len(rows) >= k_per_allele
        )
        # Per-allele row indices sorted by affinity (for quantile stratification).
        self._allele_rows: dict[str, list[int]] = {}
        for a in self._alleles:
            rows = sorted(allele_rows[a], key=lambda t: t[1])
            self._allele_rows[a] = [idx for idx, _ in rows]

        counts = {a: len(self._allele_rows[a]) for a in self._alleles}
        raw_weights = [math.sqrt(counts[a]) for a in self._alleles]
        total_w = sum(raw_weights) or 1.0
        self._allele_weights = [w / total_w for w in raw_weights]

        total_eligible = sum(counts.values())
        if num_batches is None:
            num_batches = max(1, math.ceil(total_eligible / max(self.global_batch_size, 1)))
        self.num_batches = num_batches

        print(
            f"BAGroupedBatchSampler: {len(self._alleles)} eligible alleles "
            f"({total_eligible} BA rows), R={self.R} alleles × k={self.k_per_allele} "
            f"per global batch, {self.num_batches} batches/epoch"
        )
        if not self._alleles:
            raise ValueError(
                "BAGroupedBatchSampler: no allele has >= k_per_allele affinity "
                f"rows (k_per_allele={k_per_allele}). Lower k_per_allele or check "
                "that the data has an 'affinity' column."
            )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def _draw_group(self, rng: random.Random, allele: str) -> list[int]:
        """Quantile-stratified draw of k rows from an allele (sorted by affinity)."""
        rows = self._allele_rows[allele]
        n = len(rows)
        k = self.k_per_allele
        picks = []
        for b in range(k):
            lo = (b * n) // k
            hi = ((b + 1) * n) // k
            if hi <= lo:
                hi = lo + 1
            picks.append(rows[rng.randrange(lo, min(hi, n))])
        return picks

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        for _ in range(self.num_batches):
            # Pick R distinct alleles (weighted by sqrt count) when possible.
            if len(self._alleles) <= self.R:
                picked = list(self._alleles)
            else:
                picked, seen = [], set()
                while len(picked) < self.R:
                    a = rng.choices(self._alleles, weights=self._allele_weights, k=1)[0]
                    if a not in seen:
                        seen.add(a)
                        picked.append(a)
            global_batch: list[int] = []
            for a in picked:
                global_batch.extend(self._draw_group(rng, a))

            start = self.rank * self.batch_size
            end = start + self.batch_size
            yield global_batch[start:end]


class ELGroupedBatchSampler(torch.utils.data.Sampler):
    """
    Grouped-allele batch sampler for EL (binary) training with the per-allele
    pairwise RankNet loss.

    The prod :class:`AlleleBalancedBatchSampler` draws positive-alleles and
    negative-alleles *independently*, so a batch contains only ~3-4 within-allele
    pos-neg pairs — a per-allele ranking loss is starved. This sampler instead
    builds each global batch from ``R = max(1, global_batch // k_per_allele)``
    alleles, each contributing exactly ``pos_per_allele`` positives (without
    replacement within an epoch) + ``k_per_allele - pos_per_allele`` negatives
    (with replacement). That yields ``R * pos_per_allele * (k - pos_per_allele)``
    within-allele pos-neg pairs per batch (e.g. 8×3×13 = 312 at batch 128).

    Only alleles with ``>= pos_per_allele`` positives AND ``>= k - pos_per_allele``
    negatives are eligible; alleles are picked weighted by ``sqrt(count)``.
    DDP-sharded exactly like the other samplers (each rank yields a contiguous
    slice; allele groups are kept contiguous so a rank's slice holds whole
    groups). The effective per-batch positive fraction is
    ``pos_per_allele / k_per_allele`` — set ``model.pos_frac`` to match so focal
    alpha reflects the sampled prior.

    Args:
        labels: binary label per dataset row (1=positive, 0=negative).
        mhc_names: MHC allele name per row (aligned with ``labels``).
        batch_size: per-rank batch size.
        k_per_allele: rows drawn per allele per batch.
        pos_per_allele: positives per allele per batch (rest are negatives).
        num_batches: steps per epoch. None = ceil(total_pos / (R*pos_per_allele)).
        seed / rank / num_ranks: DDP support.
    """

    def __init__(
        self,
        labels: list,
        mhc_names: list,
        batch_size: int,
        k_per_allele: int = 16,
        pos_per_allele: int = 3,
        num_batches: Optional[int] = None,
        seed: int = 0,
        rank: int = 0,
        num_ranks: int = 1,
    ):
        if not 0 < pos_per_allele < k_per_allele:
            raise ValueError(
                f"pos_per_allele ({pos_per_allele}) must be in "
                f"(0, k_per_allele={k_per_allele})"
            )
        self.batch_size = batch_size
        self.global_batch_size = batch_size * max(num_ranks, 1)
        self.k_per_allele = k_per_allele
        self.pos_per_allele = pos_per_allele
        self.neg_per_allele = k_per_allele - pos_per_allele
        self.seed = seed
        self.rank = rank
        self.num_ranks = num_ranks
        self._epoch = 0

        self.R = max(1, self.global_batch_size // k_per_allele)

        from collections import defaultdict
        allele_pos: dict = defaultdict(list)
        allele_neg: dict = defaultdict(list)
        for i, (lbl, mhc) in enumerate(zip(labels, mhc_names)):
            if lbl > 0.5:
                allele_pos[mhc].append(i)
            else:
                allele_neg[mhc].append(i)

        # Eligible: enough positives AND negatives to fill a per-allele group.
        self._alleles = sorted(
            a for a in allele_pos
            if len(allele_pos[a]) >= pos_per_allele
            and len(allele_neg.get(a, [])) >= self.neg_per_allele
        )
        self._allele_pos = {a: allele_pos[a] for a in self._alleles}
        self._allele_neg = {a: allele_neg[a] for a in self._alleles}

        counts = {a: len(self._allele_pos[a]) + len(self._allele_neg[a])
                  for a in self._alleles}
        raw_weights = [math.sqrt(counts[a]) for a in self._alleles]
        total_w = sum(raw_weights) or 1.0
        self._allele_weights = [w / total_w for w in raw_weights]

        total_pos = sum(len(v) for v in self._allele_pos.values())
        if num_batches is None:
            num_batches = max(1, math.ceil(
                total_pos / max(self.R * pos_per_allele, 1)))
        self.num_batches = num_batches

        print(
            f"ELGroupedBatchSampler: {len(self._alleles)} eligible alleles "
            f"({total_pos} pos), R={self.R} alleles × k={k_per_allele} "
            f"({pos_per_allele} pos + {self.neg_per_allele} neg) per global batch "
            f"→ {self.R * pos_per_allele * self.neg_per_allele} pairs/batch, "
            f"{self.num_batches} batches/epoch, pos_frac={pos_per_allele/k_per_allele:.3f}"
        )
        if not self._alleles:
            raise ValueError(
                "ELGroupedBatchSampler: no allele has both >= pos_per_allele "
                f"positives and >= {self.neg_per_allele} negatives. Lower "
                "k_per_allele/pos_per_allele or check the data."
            )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        # Positive cursors: without-replacement within epoch (reshuffle on wrap).
        pos_pools = {a: list(idxs) for a, idxs in self._allele_pos.items()}
        for pool in pos_pools.values():
            rng.shuffle(pool)
        pos_cursors = {a: 0 for a in self._alleles}

        for _ in range(self.num_batches):
            # Pick R distinct alleles (weighted by sqrt count) when possible.
            if len(self._alleles) <= self.R:
                picked = list(self._alleles)
            else:
                picked, seen = [], set()
                while len(picked) < self.R:
                    a = rng.choices(
                        self._alleles, weights=self._allele_weights, k=1)[0]
                    if a not in seen:
                        seen.add(a)
                        picked.append(a)

            global_batch: list[int] = []
            for a in picked:
                # positives without replacement
                pool = pos_pools[a]
                cur = pos_cursors[a]
                if cur + self.pos_per_allele > len(pool):
                    rng.shuffle(pool)
                    cur = 0
                global_batch.extend(pool[cur:cur + self.pos_per_allele])
                pos_cursors[a] = cur + self.pos_per_allele
                # negatives with replacement
                negs = self._allele_neg[a]
                global_batch.extend(
                    rng.choice(negs) for _ in range(self.neg_per_allele))

            start = self.rank * self.batch_size
            end = start + self.batch_size
            yield global_batch[start:end]


class PmhcuniDataset(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for peptide eluted-ligand data.

    Expects a CSV with at least columns: pep, mhc, label.

    Args:
        path: Path to training CSV.
        allele_sequences: OrderedDict mapping allele names to pseudo-sequences.
        batch_size: Batch size.
        num_workers: Number of DataLoader workers.
        pin_memory: Whether to pin CUDA memory.
        min_epitope_len: Minimum peptide length.
        max_epitope_len: Maximum peptide length.
        balance_sampling: If True, use :class:`BalancedBatchSampler` to
            balance positive/negative samples in the training DataLoader.
            Stage-1 SA data is highly imbalanced (~98% negatives); this
            option resamples to approximately ``pos_frac_target`` positives
            per batch using a two-pool strategy (draws from separate positive
            and negative index pools with replacement).  Has no effect on
            validation.
        pos_frac_target: Target fraction of positives when
            ``balance_sampling=True`` (default 0.2 ≈ 1 positive per 4
            negatives, i.e., neg:pos = 4:1).  Set ``model.pos_frac`` to
            the same value so ``pos_weight`` matches the resampled ratio.
        steps_per_epoch: Number of batches per epoch when
            ``balance_sampling=True``.  Passed as ``num_batches`` to
            :class:`BalancedBatchSampler`.  If ``None``, defaults to
            ``ceil(len(train_split) / batch_size)``.
    """

    def __init__(
        self,
        path: str,
        allele_sequences: OrderedDict[str, str],
        batch_size: int = 64,
        num_workers: int = 2,
        pin_memory: bool = True,
        min_epitope_len: int = 8,
        max_epitope_len: int = 15,
        balance_sampling: bool = False,
        allele_balanced: bool = False,
        ba_grouped: bool = False,
        el_grouped: bool = False,
        k_per_allele: int = 8,
        pos_per_allele: int = 3,
        pos_frac_target: float = 0.2,
        steps_per_epoch: Optional[int] = None,
        backbone_family: str = "esmc",
        esm2_name: Optional[str] = None,
        esm2_sep_concat: bool = False,
        subsample_train_frac: Optional[float] = None,
        val_path: Optional[str] = None,
    ):
        super().__init__()
        self._sep_concat = esm2_sep_concat
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.min_epitope_len = min_epitope_len
        self.max_epitope_len = max_epitope_len
        self.balance_sampling = balance_sampling
        self.allele_balanced = allele_balanced
        self.ba_grouped = ba_grouped
        self.el_grouped = el_grouped
        self.k_per_allele = k_per_allele
        self.pos_per_allele = pos_per_allele
        self.pos_frac_target = pos_frac_target
        self.steps_per_epoch = steps_per_epoch

        # ── Tokenizer: ESM-C (default) or ESM2 (HF) for the backbone ablation ──
        self._family = backbone_family
        if backbone_family == "esm2":
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(esm2_name)
            self.chain_break_id = self.tokenizer.eos_token_id  # ESM2 has no chain_break (unused: mean pooling, class I)
        else:
            self.tokenizer = EsmSequenceTokenizer()
            self.chain_break_id = self.tokenizer.chain_break_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.mask_id = self.tokenizer.mask_token_id

        # ── Allele info ──
        self.allele_sequences = allele_sequences

        # ── Auto-detect MHC class II two-chain mode ──
        # Load a small sample of MHC names to detect "/" separator
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            sample_mhc_names = []
            for i, row in enumerate(reader):
                sample_mhc_names.append(row["mhc"])
                if i >= 100:
                    break
        self._two_chain = _detect_two_chain(allele_sequences, sample_mhc_names)
        self._max_mhc_tokens = _compute_max_mhc_tokens(self._two_chain)
        if self._two_chain:
            print(f"[PmhcuniDataset] Two-chain (class II) mode detected, "
                  f"max_mhc_tokens={self._max_mhc_tokens}")

        # Build group map from MHC names in data (for class II composite names)
        if self._two_chain:
            # For class II, group by composite allele name from data
            mhc_names_unique = list(OrderedDict.fromkeys(sample_mhc_names))
            # We'll rebuild this after loading all data
            self.group_map = OrderedDict()
            self.group_map_inv = OrderedDict()
        else:
            self.group_map = OrderedDict(
                (i, k) for i, k in enumerate(allele_sequences.keys())
            )
            self.group_map_inv = OrderedDict(
                (k, i) for i, k in self.group_map.items()
            )

        # Pre-compute MHC token cache
        self._mhc_token_cache: dict[str, list[int]] = {}

        # ── Load & filter data ──
        with open(path, "r") as f:
            raw = list(csv.DictReader(f))

        raw = [
            s for s in raw
            if min_epitope_len <= len(s["sequence"]) <= max_epitope_len
        ]

        # Build group map for class II from actual data MHC names
        if self._two_chain:
            all_mhc_names = list(OrderedDict.fromkeys(s["mhc"] for s in raw))
            self.group_map = OrderedDict((i, k) for i, k in enumerate(all_mhc_names))
            self.group_map_inv = OrderedDict((k, i) for i, k in self.group_map.items())

        # Detect EL label column: 'lable' (user convention) takes priority, then 'label'
        self._label_col: str = "lable" if raw and "lable" in raw[0] else "label"
        # Detect optional BA affinity column (values are 1-log50k, range [0, 1])
        self._has_affinity: bool = bool(raw and "affinity" in raw[0])
        # Detect optional censoring flag column for the BA head (0=exact, +1=IC50>, -1=IC50<)
        self._has_ineq: bool = bool(raw and "ba_ineq" in raw[0])
        # Detect optional immunogenicity label column (joint EL+immuno multi-task FT;
        # blank/nan for rows without a T-cell-assay immunogenicity readout, same
        # convention as affinity — see model.py's use_immuno_head)
        self._has_immuno: bool = bool(raw and "immuno_label" in raw[0])

        # An explicit val_path keeps ALL of `path` for training and reads validation
        # from a separate CSV. The default random 5% row-split shares its negative
        # pool (and often its source proteins) with the training rows, so a model that
        # memorises the decoy pool scores well on it — which is precisely why it cannot
        # detect decoy overfitting, and why earlier runs never located a real
        # convergence point. Point val_path at a protein-disjoint set to get an honest
        # early-stopping signal.
        if val_path:
            with open(val_path, "r") as f:
                vraw = list(csv.DictReader(f))
            vraw = [
                s for s in vraw
                if min_epitope_len <= len(s["sequence"]) <= max_epitope_len
            ]
            self._train, self._val = raw, vraw
            print(f"[INFO] val_path={val_path} -> {len(vraw)} val rows "
                  f"(train keeps all {len(raw)} rows)")
        else:
            self._train, self._val = train_test_split(
                raw, test_size=0.05, random_state=42
            )
        # Data-scaling study: keep a reproducible fraction of UNIQUE training rows
        # (val unchanged). With fixed max_steps this varies data diversity/repetition.
        if subsample_train_frac is not None and 0 < subsample_train_frac < 1.0:
            import random as _random
            k = max(1, int(len(self._train) * subsample_train_frac))
            self._train = _random.Random(42).sample(self._train, k)
            print(f"[INFO] subsample_train_frac={subsample_train_frac} -> {k} unique train rows")
        self._test: list = []

        # Cache training labels for BalancedBatchSampler (built in setup())
        self._train_labels: list[float] = [
            float(s.get(self._label_col) or s.get("label") or 0.0)
            for s in self._train
        ]
        self._train_mhc_names: list[str] = [s["mhc"] for s in self._train]
        # Per-record affinity aligned with self._train (None when no BA label) —
        # consumed by BAGroupedBatchSampler when ba_grouped=True.
        def _parse_aff(s: dict) -> Optional[float]:
            if not self._has_affinity:
                return None
            raw = s.get("affinity", "")
            if raw is None or str(raw).strip().lower() in ("", "nan"):
                return None
            try:
                return float(raw)
            except (ValueError, TypeError):
                return None
        self._train_affinities: list[Optional[float]] = [_parse_aff(s) for s in self._train]
        n_pos = sum(1 for lbl in self._train_labels if lbl > 0.5)
        n_neg = len(self._train_labels) - n_pos
        pos_frac_real = n_pos / max(len(self._train_labels), 1)
        print(f"Train length: {len(self._train)}")
        print(f"Val length:   {len(self._val)}")
        print(f"Train label distribution: pos={n_pos}, neg={n_neg}, "
              f"pos_frac={pos_frac_real:.4f}")
        if balance_sampling:
            neg_to_pos_ratio = (1.0 - pos_frac_target) / pos_frac_target
            print(f"Balance sampling enabled: target pos_frac={pos_frac_target:.2f} "
                  f"(neg:pos ≈ {neg_to_pos_ratio:.1f}:1) — "
                  f"set model.pos_frac={pos_frac_target} to match pos_weight")

    # ──────────────────────────────────────────────
    # Processing
    # ──────────────────────────────────────────────
    def _tokenize_sequence(self, seq: str, add_special_tokens: bool = True) -> torch.Tensor:
        """Tokenize a single sequence with BOS/EOS using the ESM C tokenizer."""
        encoded = self.tokenizer.encode(seq, add_special_tokens = add_special_tokens)
        return encoded

    def _get_mhc_tokens(self, mhc_name: str) -> list[int]:
        """Get MHC tokens with caching, supporting two-chain mode."""
        if mhc_name not in self._mhc_token_cache:
            self._mhc_token_cache[mhc_name] = _resolve_mhc_tokens(
                mhc_name, self.allele_sequences, self.tokenizer,
                self.chain_break_id,
                max_mhc_tokens=self._max_mhc_tokens,
                pad_id=self.pad_id,
            )
        return self._mhc_token_cache[mhc_name]

    def process(self, sample: dict, split: str) -> dict:
        pep_seq = sample["sequence"]
        mhc_name = sample["mhc"]

        mhc_tokens = self._get_mhc_tokens(mhc_name)

        # EL label: use detected column ('lable' or 'label'), with explicit None fallback
        raw_label = sample.get(self._label_col)
        if raw_label is None:
            raw_label = sample.get("label")
        label = float(raw_label) if raw_label is not None else 0.0

        # Tokenize peptide
        max_len = self.max_epitope_len
        pep_tokens = self._tokenize_sequence(pep_seq, add_special_tokens=False)

        pad_len = max_len - len(pep_seq)

        if getattr(self, "_family", "esmc") == "esm2":
            if getattr(self, "_sep_concat", False):
                # RSAP/sep_concat: insert ESM2 <eos> as the MHC|peptide separator
                # ([CLS] MHC <eos> peptide [EOS]) so sep_concat can split the regions.
                full_seq = (
                    [self.cls_id] + mhc_tokens + [self.eos_id] + pep_tokens
                    + [self.eos_id] + [self.pad_id] * pad_len
                )
            else:
                # ESM2: mean/cls pooling over [CLS] MHC peptide [EOS]
                full_seq = (
                    [self.cls_id] + mhc_tokens + pep_tokens
                    + [self.eos_id] + [self.pad_id] * pad_len
                )
        else:
            full_seq = (
                [self.cls_id]
                + mhc_tokens
                + [self.chain_break_id]
                + pep_tokens
                + [self.eos_id]
                + [self.pad_id] * pad_len
            )
        full_seq = torch.tensor(full_seq, dtype=torch.long)
        # Allele group index for metrics
        group_idx = self.group_map_inv.get(mhc_name, -1)

        result: dict = {
            "sequence_tokens": full_seq,
            "label": torch.tensor(label, dtype=torch.float),
            "groups": torch.tensor(group_idx, dtype=torch.long),
        }

        # Optional BA affinity (1-log50k values in [0, 1]; 0-placeholder when absent)
        if self._has_affinity:
            raw_aff = sample.get("affinity", "")
            if raw_aff and str(raw_aff).strip().lower() not in ("nan",):
                aff_val = float(raw_aff)
                result["affinity"] = torch.tensor(aff_val, dtype=torch.float)
                result["affinity_mask"] = torch.tensor(True)
            else:
                result["affinity"] = torch.tensor(0.0, dtype=torch.float)
                result["affinity_mask"] = torch.tensor(False)
            # censoring flag (0=exact, +1=IC50>, -1=IC50<); default 0 (exact) when absent
            if self._has_ineq:
                try:
                    result["affinity_ineq"] = torch.tensor(int(float(sample.get("ba_ineq", 0) or 0)), dtype=torch.long)
                except (ValueError, TypeError):
                    result["affinity_ineq"] = torch.tensor(0, dtype=torch.long)

        # Optional immunogenicity label (binary; 0-placeholder + mask=False when absent)
        if self._has_immuno:
            raw_im = sample.get("immuno_label", "")
            if raw_im is not None and str(raw_im).strip().lower() not in ("", "nan"):
                im_val = float(raw_im)
                result["immuno_label"] = torch.tensor(im_val, dtype=torch.float)
                result["immuno_mask"] = torch.tensor(True)
            else:
                result["immuno_label"] = torch.tensor(0.0, dtype=torch.float)
                result["immuno_mask"] = torch.tensor(False)

        return result

    # ──────────────────────────────────────────────
    # DataModule interface
    # ──────────────────────────────────────────────
    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        self._train_view = DataView(self._train, self.process, "train")
        self._val_view = DataView(self._val, self.process, "val")

    def _get_ddp_rank_info(self) -> tuple[int, int]:
        """Return (rank, num_ranks) from the attached Trainer or torch.distributed."""
        # Prefer the Trainer's global_rank / world_size (set before train_dataloader is called)
        if self.trainer is not None:
            return self.trainer.global_rank, self.trainer.world_size
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return torch.distributed.get_rank(), torch.distributed.get_world_size()
        except Exception:
            pass
        return 0, 1

    def train_dataloader(self):
        if getattr(self, "el_grouped", False):
            # EL grouped-allele sampling: each batch = R alleles × k rows
            # (pos_per_allele pos + rest neg), for the per-allele EL RankNet
            # loss (needs many intra-allele pos-neg pairs per batch, which the
            # prod AlleleBalancedBatchSampler does not provide).
            rank, num_ranks = self._get_ddp_rank_info()
            batch_sampler = ELGroupedBatchSampler(
                labels=self._train_labels,
                mhc_names=self._train_mhc_names,
                batch_size=self.batch_size,
                k_per_allele=self.k_per_allele,
                pos_per_allele=self.pos_per_allele,
                num_batches=self.steps_per_epoch,
                seed=42,
                rank=rank,
                num_ranks=num_ranks,
            )
            return torch.utils.data.DataLoader(
                self._train_view,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        if getattr(self, "ba_grouped", False):
            # BA-specialist grouped-allele sampling: each batch = R alleles ×
            # k_per_allele quantile-stratified affinity rows, for the pairwise
            # RankNet loss (needs many intra-allele pairs per batch).
            rank, num_ranks = self._get_ddp_rank_info()
            batch_sampler = BAGroupedBatchSampler(
                affinities=self._train_affinities,
                mhc_names=self._train_mhc_names,
                batch_size=self.batch_size,
                k_per_allele=self.k_per_allele,
                num_batches=self.steps_per_epoch,
                seed=42,
                rank=rank,
                num_ranks=num_ranks,
            )
            return torch.utils.data.DataLoader(
                self._train_view,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        if getattr(self, "allele_balanced", False):
            # Stage-1 allele-balanced sampling (E3, CL6): two-level sqrt(freq) over alleles.
            rank, num_ranks = self._get_ddp_rank_info()
            batch_sampler = AlleleBalancedBatchSampler(
                labels=self._train_labels,
                mhc_names=self._train_mhc_names,
                batch_size=self.batch_size,
                pos_frac_target=self.pos_frac_target,
                num_batches=self.steps_per_epoch,
                seed=42,
                rank=rank,
                num_ranks=num_ranks,
            )
            return torch.utils.data.DataLoader(
                self._train_view,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        if self.balance_sampling:
            rank, num_ranks = self._get_ddp_rank_info()

            batch_sampler = BalancedBatchSampler(
                labels=self._train_labels,
                batch_size=self.batch_size,
                pos_frac_target=self.pos_frac_target,
                num_batches=self.steps_per_epoch,
                seed=42,
                rank=rank,
                num_ranks=num_ranks,
            )

            if rank == 0:
                print(
                    f"BalancedBatchSampler: {batch_sampler.num_batches} batches/epoch, "
                    f"local batch={batch_sampler.batch_size}/GPU × {num_ranks} GPUs "
                    f"= global {batch_sampler.global_batch_size}, "
                    f"n_pos={batch_sampler.n_pos}, n_neg={batch_sampler.n_neg} per global batch "
                    f"(target pos_frac={self.pos_frac_target:.2f})"
                )

            # batch_sampler already handles DDP sharding; tell Lightning not to
            # wrap the DataLoader with an extra DistributedSampler.
            return torch.utils.data.DataLoader(
                self._train_view,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        return torch.utils.data.DataLoader(
            self._train_view,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_view,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Stage-2: Multi-Allele (MA) bag datasets + combined DataModule
# ──────────────────────────────────────────────────────────────────────────────

def _encode_pair(
    pep_seq: str,
    mhc_seq: str,
    tokenizer: EsmSequenceTokenizer,
    cls_id: int,
    chain_break_id: int,
    eos_id: int,
    pad_id: int,
    max_epitope_len: int,
    mhc_tokens: list[int] | None = None,
) -> torch.Tensor:
    """Encode a (peptide, MHC) pair into a fixed-length token tensor.

    If ``mhc_tokens`` is provided, uses them directly (for pre-resolved
    two-chain MHC). Otherwise tokenizes ``mhc_seq[:MIN_LEN]`` as before.
    """
    pep_tokens = tokenizer.encode(pep_seq, add_special_tokens=False)
    if mhc_tokens is None:
        mhc_tokens = tokenizer.encode(mhc_seq[:MIN_LEN], add_special_tokens=False)
        # Pad or truncate to fixed length for consistent tensor shapes
        if len(mhc_tokens) > MIN_LEN:
            mhc_tokens = mhc_tokens[:MIN_LEN]
        elif len(mhc_tokens) < MIN_LEN:
            mhc_tokens = mhc_tokens + [pad_id] * (MIN_LEN - len(mhc_tokens))
    pad_len = max_epitope_len - len(pep_seq)
    full_seq = (
        [cls_id]
        + mhc_tokens
        + [chain_break_id]
        + pep_tokens
        + [eos_id]
        + [pad_id] * pad_len
    )
    return torch.tensor(full_seq, dtype=torch.long)


class BagDataset(torch.utils.data.Dataset):
    """
    Dataset that yields bag-shaped samples for MIL training.

    Each sample is a peptide with one or more candidate MHC alleles.
    For SA data (single allele), the bag has K=1.
    For MA data (multiple alleles), the bag has K = min(len(alleles), k_max).

    All bags are zero-padded to ``k_max`` candidates along the K dimension.

    SA records may optionally carry a binding-affinity label (``affinity``
    column).  When present the value is forwarded as ``affinity`` /
    ``affinity_mask`` keys in the returned dict so the BA head can be trained
    jointly.  MA records never carry affinity labels.

    Args:
        records: List of dicts with keys ``sequence``, ``mhc``, ``label``.
            For SA records ``mhc`` is a single allele name.
            For MA records ``mhc`` is a comma-separated list of allele names.
        allele_sequences: Ordered dict mapping allele name → pseudo-sequence.
        tokenizer: ESM C sequence tokenizer.
        k_max: Maximum number of candidates per bag (truncation/padding bound).
        min_epitope_len: Minimum peptide length (inclusive).
        max_epitope_len: Maximum peptide length (inclusive).
        is_sa: If True, treat each record as SA (K=1, single allele in ``mhc``).
    """

    def __init__(
        self,
        records: list[dict],
        allele_sequences: OrderedDict[str, str],
        tokenizer: EsmSequenceTokenizer,
        k_max: int,
        min_epitope_len: int = 8,
        max_epitope_len: int = 15,
        is_sa: bool = False,
    ):
        self.allele_sequences = allele_sequences
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id
        self.chain_break_id = tokenizer.chain_break_token_id
        self.k_max = k_max
        self.max_epitope_len = max_epitope_len
        self.is_sa = is_sa

        # Detect two-chain mode
        self._two_chain = _detect_two_chain(allele_sequences)
        self._max_mhc_tokens = _compute_max_mhc_tokens(self._two_chain)
        self._mhc_token_cache: dict[str, list[int]] = {}

        # Detect whether the source records carry an affinity column (SA only)
        self._has_affinity: bool = is_sa and bool(records and "affinity" in records[0])

        self.samples: list[dict] = []
        for row in records:
            pep = row["sequence"]
            if not (min_epitope_len <= len(pep) <= max_epitope_len):
                continue
            # Parse alleles: SA has a single name; MA has comma-separated names
            if is_sa:
                alleles_raw = [row["mhc"].strip()]
            else:
                alleles_raw = [a.strip() for a in row["mhc"].split(",")]
            # Keep only alleles present in allele_sequences (or all for two-chain)
            if self._two_chain:
                alleles = alleles_raw
            else:
                alleles = [a for a in alleles_raw if a in allele_sequences]
            if not alleles:
                continue
            # Truncate to k_max
            alleles = alleles[:k_max]
            label = float(row.get("lable") if "lable" in row else row.get("label", 1.0))

            entry: dict = {"sequence": pep, "alleles": alleles, "label": label}

            # Forward affinity if present (SA only)
            if self._has_affinity:
                raw_aff = row.get("affinity", "")
                if raw_aff and str(raw_aff).strip().lower() not in ("", "nan"):
                    entry["affinity"] = float(raw_aff)
                    entry["has_affinity"] = True
                else:
                    entry["affinity"] = 0.0
                    entry["has_affinity"] = False

            self.samples.append(entry)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        pep_seq = sample["sequence"]
        alleles = sample["alleles"]
        label = sample["label"]
        k = len(alleles)

        # Build per-candidate token tensors: (K, L)
        candidate_tokens = []
        for allele in alleles:
            if allele not in self._mhc_token_cache:
                self._mhc_token_cache[allele] = _resolve_mhc_tokens(
                    allele, self.allele_sequences, self.tokenizer,
                    self.chain_break_id,
                    max_mhc_tokens=self._max_mhc_tokens,
                    pad_id=self.pad_id,
                )
            mhc_tokens = self._mhc_token_cache[allele]
            tokens = _encode_pair(
                pep_seq, "",
                self.tokenizer,
                self.cls_id, self.chain_break_id, self.eos_id, self.pad_id,
                self.max_epitope_len,
                mhc_tokens=mhc_tokens,
            )
            candidate_tokens.append(tokens)

        bag_tokens = torch.stack(candidate_tokens, dim=0)  # (K, L)
        K, L = bag_tokens.shape

        # Pad K dimension to k_max
        if K < self.k_max:
            padding = torch.full((self.k_max - K, L), self.pad_id, dtype=torch.long)
            bag_tokens = torch.cat([bag_tokens, padding], dim=0)  # (k_max, L)

        # Mask: True for valid (non-padded) candidates
        bag_mask = torch.zeros(self.k_max, dtype=torch.bool)
        bag_mask[:k] = True

        result = {
            "sequence_tokens": bag_tokens,          # (k_max, L)
            "bag_mask": bag_mask,                   # (k_max,)
            "label": torch.tensor(label, dtype=torch.float),
            "is_sa": torch.tensor(self.is_sa, dtype=torch.bool),
        }

        if self._has_affinity:
            result["affinity"] = torch.tensor(sample["affinity"], dtype=torch.float)
            result["affinity_mask"] = torch.tensor(sample["has_affinity"], dtype=torch.bool)

        return result


class CombinedBagDataset(torch.utils.data.Dataset):
    """
    Combines an SA ``BagDataset`` and an MA ``BagDataset`` into a single dataset.

    SA samples are repeated ``sa_repeat`` times to upsample them relative to MA.

    Args:
        sa_dataset: BagDataset built from SA records (is_sa=True, k_max=k_max).
        ma_dataset: BagDataset built from MA records (is_sa=False, k_max=k_max).
        sa_repeat: Number of times to repeat each SA sample (≥1).
    """

    def __init__(
        self,
        sa_dataset: BagDataset,
        ma_dataset: BagDataset,
        sa_repeat: int = 1,
    ):
        self.sa_dataset = sa_dataset
        self.ma_dataset = ma_dataset
        # Build flat index list: (source, original_index)
        sa_items = [("sa", i) for _ in range(max(1, sa_repeat)) for i in range(len(sa_dataset))]
        ma_items = [("ma", i) for i in range(len(ma_dataset))]
        self._items: list[tuple[str, int]] = sa_items + ma_items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        source, i = self._items[idx]
        if source == "sa":
            return self.sa_dataset[i]
        return self.ma_dataset[i]


def bag_collate_fn(batch: list[dict]) -> dict:
    """
    Collate a list of bag samples into a padded batch.

    Returns a dict with:
        sequence_tokens : LongTensor  (B, K_max, L)
        bag_mask        : BoolTensor  (B, K_max)
        label           : FloatTensor (B,)
        is_sa           : BoolTensor  (B,)
        affinity        : FloatTensor (B,)   — only if any sample has affinity
        affinity_mask   : BoolTensor  (B,)   — only if any sample has affinity
    """
    result = {
        "sequence_tokens": torch.stack([s["sequence_tokens"] for s in batch]),
        "bag_mask": torch.stack([s["bag_mask"] for s in batch]),
        "label": torch.stack([s["label"] for s in batch]),
        "is_sa": torch.stack([s["is_sa"] for s in batch]),
    }
    # Include affinity fields only when at least one sample carries them
    if any("affinity" in s for s in batch):
        result["affinity"] = torch.stack([
            s.get("affinity", torch.tensor(0.0)) for s in batch
        ])
        result["affinity_mask"] = torch.stack([
            s.get("affinity_mask", torch.tensor(False)) for s in batch
        ])
    return result


class Stage2DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for stage-2 combined SA + MA training.

    SA records are treated as bags with K=1 and upsampled ``sa_repeat`` times.
    MA records are bags with up to ``k_max`` candidate alleles.
    Training uses the ``bag_collate_fn`` so batches have shape (B, K_max, L).
    Validation uses the SA split only (standard per-allele AP metric).

    Args:
        sa_path: Path to SA CSV (columns: sequence, mhc, label[, affinity]).
        ma_path: Path to MA CSV (columns: sequence, mhc [comma-sep list], label).
        allele_sequences: Ordered dict mapping allele name → pseudo-sequence.
        batch_size: Batch size.
        num_workers: Number of DataLoader workers.
        pin_memory: Whether to pin CUDA memory.
        min_epitope_len: Minimum peptide length.
        max_epitope_len: Maximum peptide length.
        sa_repeat: SA upsampling factor (each SA sample repeated this many times).
        k_max: Maximum number of candidate alleles per bag.
        balance_sampling: If True, use :class:`BalancedBatchSampler` to keep
            approximately ``pos_frac_target`` SA positives per batch.  MA
            samples are added to every batch unchanged on top of the balanced
            SA draws.  Has no effect on validation.
        pos_frac_target: Target positive fraction for SA samples when
            ``balance_sampling=True`` (default 0.2).
    """

    def __init__(
        self,
        sa_path: str,
        ma_path: str,
        allele_sequences: OrderedDict[str, str],
        batch_size: int = 64,
        num_workers: int = 2,
        pin_memory: bool = True,
        min_epitope_len: int = 8,
        max_epitope_len: int = 15,
        sa_repeat: int = 1,
        k_max: int = 6,
        balance_sampling: bool = False,
        pos_frac_target: float = 0.2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.min_epitope_len = min_epitope_len
        self.max_epitope_len = max_epitope_len
        self.sa_repeat = sa_repeat
        self.k_max = k_max
        self.balance_sampling = balance_sampling
        self.pos_frac_target = pos_frac_target

        self.tokenizer = EsmSequenceTokenizer()
        self.allele_sequences = allele_sequences

        # ── Load SA data ──
        with open(sa_path, "r") as f:
            sa_raw = list(csv.DictReader(f))

        sa_filtered = [
            s for s in sa_raw
            if min_epitope_len <= len(s["sequence"]) <= max_epitope_len
        ]
        self._sa_train, self._sa_val = train_test_split(
            sa_filtered, test_size=0.05, random_state=42
        )

        # ── Load MA data ──
        with open(ma_path, "r") as f:
            ma_raw = list(csv.DictReader(f))

        self._ma_train = [
            s for s in ma_raw
            if min_epitope_len <= len(s["sequence"]) <= max_epitope_len
        ]

        # ── Label column detection (handle 'lable' typo) ──
        _lbl_col = "lable" if self._sa_train and "lable" in self._sa_train[0] else "label"
        self._sa_train_labels: list[float] = [
            float(s.get(_lbl_col) or s.get("label") or 0.0)
            for s in self._sa_train
        ]

        n_pos = sum(1 for l in self._sa_train_labels if l > 0.5)
        n_neg = len(self._sa_train_labels) - n_pos
        pos_frac_real = n_pos / max(len(self._sa_train_labels), 1)
        print(f"Stage2 SA train: {len(self._sa_train)}, val: {len(self._sa_val)}")
        print(f"Stage2 MA train: {len(self._ma_train)}")
        print(f"Stage2 SA label dist: pos={n_pos}, neg={n_neg}, "
              f"pos_frac={pos_frac_real:.4f}")
        if balance_sampling:
            print(f"Stage2 balance_sampling enabled: target pos_frac={pos_frac_target:.2f}")

        # ── Allele group map (for SA validation metrics) ──
        _two_chain = _detect_two_chain(allele_sequences,
                                       [s["mhc"] for s in self._sa_train[:100]])
        if _two_chain:
            all_mhc_names = list(OrderedDict.fromkeys(s["mhc"] for s in sa_filtered))
            self.group_map = OrderedDict((i, k) for i, k in enumerate(all_mhc_names))
        else:
            self.group_map = OrderedDict(
                (i, k) for i, k in enumerate(allele_sequences.keys())
            )
        self.group_map_inv = OrderedDict(
            (k, i) for i, k in self.group_map.items()
        )

    # ──────────────────────────────────────────────
    # DataModule interface
    # ──────────────────────────────────────────────
    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        common_kw = dict(
            allele_sequences=self.allele_sequences,
            tokenizer=self.tokenizer,
            k_max=self.k_max,
            min_epitope_len=self.min_epitope_len,
            max_epitope_len=self.max_epitope_len,
        )
        sa_train_ds = BagDataset(self._sa_train, is_sa=True, **common_kw)
        ma_train_ds = BagDataset(self._ma_train, is_sa=False, **common_kw)
        self._train_ds = CombinedBagDataset(sa_train_ds, ma_train_ds, sa_repeat=self.sa_repeat)

        # Validation: SA only, using SA process method (flat, with groups)
        self._val_ds = _SaValDataset(
            self._sa_val,
            allele_sequences=self.allele_sequences,
            tokenizer=self.tokenizer,
            group_map_inv=self.group_map_inv,
            min_epitope_len=self.min_epitope_len,
            max_epitope_len=self.max_epitope_len,
        )

    def _get_ddp_rank_info(self) -> tuple[int, int]:
        """Return (rank, num_ranks) from the attached Trainer or torch.distributed."""
        if self.trainer is not None:
            return self.trainer.global_rank, self.trainer.world_size
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return torch.distributed.get_rank(), torch.distributed.get_world_size()
        except Exception:
            pass
        return 0, 1

    def train_dataloader(self):
        if self.balance_sampling:
            # Build a label list aligned with CombinedBagDataset's flat index.
            # CombinedBagDataset._items is [(source, orig_idx), ...]:
            #   SA entries → use sa_train_ds label; MA entries → placeholder 1.0
            combined_labels: list[float] = []
            sa_ds = self._train_ds.sa_dataset
            for source, i in self._train_ds._items:
                if source == "sa":
                    combined_labels.append(sa_ds.samples[i]["label"])
                else:
                    # MA samples are always treated as positives for sampling purposes
                    combined_labels.append(1.0)

            rank, num_ranks = self._get_ddp_rank_info()

            batch_sampler = BalancedBatchSampler(
                labels=combined_labels,
                batch_size=self.batch_size,
                pos_frac_target=self.pos_frac_target,
                seed=42,
                rank=rank,
                num_ranks=num_ranks,
            )
            if rank == 0:
                print(
                    f"Stage2 BalancedBatchSampler: {batch_sampler.num_batches} batches/epoch, "
                    f"local batch={batch_sampler.batch_size}/GPU × {num_ranks} GPUs "
                    f"= global {batch_sampler.global_batch_size}, "
                    f"n_pos={batch_sampler.n_pos}, n_neg={batch_sampler.n_neg} per global batch "
                    f"(target SA pos_frac={self.pos_frac_target:.2f})"
                )
            # batch_sampler already handles DDP sharding
            return torch.utils.data.DataLoader(
                self._train_ds,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=bag_collate_fn,
            )
        return torch.utils.data.DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=bag_collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class _SaValDataset(torch.utils.data.Dataset):
    """SA validation dataset that returns flat (non-bag) samples with group IDs."""

    def __init__(
        self,
        records: list[dict],
        allele_sequences: OrderedDict[str, str],
        tokenizer: EsmSequenceTokenizer,
        group_map_inv: OrderedDict[str, int],
        min_epitope_len: int = 8,
        max_epitope_len: int = 15,
    ):
        self.allele_sequences = allele_sequences
        self.tokenizer = tokenizer
        self.group_map_inv = group_map_inv
        self.max_epitope_len = max_epitope_len
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id
        self.chain_break_id = tokenizer.chain_break_token_id

        # Detect two-chain mode
        self._two_chain = _detect_two_chain(allele_sequences,
                                            [r["mhc"] for r in records[:100]])
        self._max_mhc_tokens = _compute_max_mhc_tokens(self._two_chain)

        # Pre-compute MHC token cache
        self._mhc_token_cache: dict[str, list[int]] = {}

        # For two-chain mode, we can't filter by "mhc in allele_sequences"
        # because composite names won't be in chain_seqs directly
        if self._two_chain:
            self.samples = [
                r for r in records
                if min_epitope_len <= len(r["sequence"]) <= max_epitope_len
            ]
        else:
            self.samples = [
                r for r in records
                if (
                    min_epitope_len <= len(r["sequence"]) <= max_epitope_len
                    and r["mhc"].strip() in allele_sequences
                )
            ]

    def _get_mhc_tokens(self, mhc_name: str) -> list[int]:
        if mhc_name not in self._mhc_token_cache:
            self._mhc_token_cache[mhc_name] = _resolve_mhc_tokens(
                mhc_name, self.allele_sequences, self.tokenizer,
                self.chain_break_id,
                max_mhc_tokens=self._max_mhc_tokens,
                pad_id=self.pad_id,
            )
        return self._mhc_token_cache[mhc_name]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        row = self.samples[idx]
        pep_seq = row["sequence"]
        mhc_name = row["mhc"].strip()
        # Support 'lable' spelling (user convention) with 'label' fallback
        label = float(row.get("lable") if "lable" in row else row["label"])

        mhc_tokens = self._get_mhc_tokens(mhc_name)
        tokens = _encode_pair(
            pep_seq, "",
            self.tokenizer,
            self.cls_id, self.chain_break_id, self.eos_id, self.pad_id,
            self.max_epitope_len,
            mhc_tokens=mhc_tokens,
        )
        group_idx = self.group_map_inv.get(mhc_name, -1)
        return {
            "sequence_tokens": tokens,
            "label": torch.tensor(label, dtype=torch.float),
            "groups": torch.tensor(group_idx, dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Stage-2 V2: Pre-predict MA → flatten → merge with SA  (no per-batch MIL)
# ──────────────────────────────────────────────────────────────────────────────

class _MaPredictDataset(torch.utils.data.Dataset):
    """
    Dataset that yields all (peptide, allele) pairs from MA records for batch
    prediction.  Each item is a flat token tensor for one (pep, allele) pair,
    plus metadata to map back to the original MA record and candidate index.

    Used by :func:`batch_predict_ma_winners` for efficient batched inference.
    """

    def __init__(
        self,
        ma_records: list[dict],
        allele_sequences: OrderedDict[str, str],
        tokenizer: EsmSequenceTokenizer,
        min_epitope_len: int = 8,
        max_epitope_len: int = 15,
    ):
        self.allele_sequences = allele_sequences
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id
        self.chain_break_id = tokenizer.chain_break_token_id
        self.max_epitope_len = max_epitope_len

        # Detect two-chain mode
        self._two_chain = _detect_two_chain(allele_sequences)
        self._max_mhc_tokens = _compute_max_mhc_tokens(self._two_chain)
        self._mhc_token_cache: dict[str, list[int]] = {}

        # Build flat list: [(record_idx, allele_name, pep_seq), ...]
        self.items: list[tuple[int, str, str]] = []
        for rec_idx, row in enumerate(ma_records):
            pep = row["sequence"]
            if not (min_epitope_len <= len(pep) <= max_epitope_len):
                continue
            alleles_raw = [a.strip() for a in row["mhc"].split(",")]
            if self._two_chain:
                # For class II, accept all alleles (composite names)
                alleles = alleles_raw
            else:
                alleles = [a for a in alleles_raw if a in allele_sequences]
            for allele in alleles:
                # Pre-cache MHC tokens
                if allele not in self._mhc_token_cache:
                    self._mhc_token_cache[allele] = _resolve_mhc_tokens(
                        allele, allele_sequences, tokenizer, self.chain_break_id,
                        max_mhc_tokens=self._max_mhc_tokens,
                        pad_id=self.pad_id,
                    )
                self.items.append((rec_idx, allele, pep))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        rec_idx, allele, pep_seq = self.items[idx]
        mhc_tokens = self._mhc_token_cache[allele]
        tokens = _encode_pair(
            pep_seq, "",
            self.tokenizer,
            self.cls_id, self.chain_break_id, self.eos_id, self.pad_id,
            self.max_epitope_len,
            mhc_tokens=mhc_tokens,
        )
        return {
            "sequence_tokens": tokens,
            "record_idx": rec_idx,
            "allele": allele,
        }


def _ma_predict_collate(batch: list[dict]) -> dict:
    """Collate for _MaPredictDataset — stack tokens, forward metadata."""
    return {
        "sequence_tokens": torch.stack([s["sequence_tokens"] for s in batch]),
        "record_idx": [s["record_idx"] for s in batch],
        "allele": [s["allele"] for s in batch],
    }


@torch.no_grad()
def batch_predict_ma_winners(
    model,
    ma_records: list[dict],
    allele_sequences: OrderedDict[str, str],
    tokenizer: EsmSequenceTokenizer,
    min_epitope_len: int = 8,
    max_epitope_len: int = 15,
    batch_size: int = 512,
    num_workers: int = 4,
    device: torch.device | str = "cuda",
) -> list[dict]:
    """
    Batch-predict the winner allele for each MA record.

    For each MA peptide with N candidate alleles, runs the model on all N
    (peptide, allele) pairs, picks the allele with the highest EL logit
    (winner-takes-all), and returns a list of SA-style flat records:
        ``{"sequence": pep, "mhc": winner_allele, "label": original_label}``

    Args:
        model: A PmhcuniModel (will be set to eval mode; restored afterward).
        ma_records: Raw MA record dicts with keys ``sequence``, ``mhc``
            (comma-separated alleles), ``label``.
        allele_sequences: Allele name → pseudo-sequence mapping.
        tokenizer: ESM C tokenizer.
        min_epitope_len: Min peptide length filter.
        max_epitope_len: Max peptide length filter.
        batch_size: Inference batch size (can be large since no gradients).
        num_workers: DataLoader workers for prediction.
        device: Device for inference.

    Returns:
        List of SA-style dicts, one per MA record that passed filtering.
    """
    pred_ds = _MaPredictDataset(
        ma_records, allele_sequences, tokenizer,
        min_epitope_len, max_epitope_len,
    )
    if len(pred_ds) == 0:
        return []

    loader = torch.utils.data.DataLoader(
        pred_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_ma_predict_collate,
    )

    was_training = model.training
    model.eval()
    model_device = next(model.parameters()).device

    # Collect per-(record_idx, allele) logits
    # Use a dict: record_idx -> list of (logit, allele)
    from collections import defaultdict
    record_scores: dict[int, list[tuple[float, str]]] = defaultdict(list)

    for batch in loader:
        tokens = batch["sequence_tokens"].to(model_device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(tokens)               # (B, 1)
        logits_np = logits.float().squeeze(-1).cpu().tolist()
        for logit_val, rec_idx, allele in zip(
            logits_np, batch["record_idx"], batch["allele"]
        ):
            record_scores[rec_idx].append((logit_val, allele))

    if was_training:
        model.train()

    # For each record, pick the allele with the highest logit
    # Build a lookup from record_idx → original record
    rec_lookup: dict[int, dict] = {}
    for rec_idx, row in enumerate(ma_records):
        pep = row["sequence"]
        if min_epitope_len <= len(pep) <= max_epitope_len:
            rec_lookup[rec_idx] = row

    flattened: list[dict] = []
    for rec_idx, scores in record_scores.items():
        if not scores:
            continue
        # Winner = highest logit
        best_logit, best_allele = max(scores, key=lambda x: x[0])
        orig = rec_lookup.get(rec_idx)
        if orig is None:
            continue
        _lbl_col = "lable" if "lable" in orig else "label"
        label = float(orig.get(_lbl_col, 1.0))
        flattened.append({
            "sequence": orig["sequence"],
            "mhc": best_allele,
            "label": label,
        })

    return flattened


class _FlatSaDataset(torch.utils.data.Dataset):
    """
    Flat SA-style dataset returning (sequence_tokens, label, groups) dicts.

    Used by Stage2V2DataModule to combine SA records + flattened MA records
    into a single dataset for standard (non-MIL) training.

    Tokenization is pre-computed in __init__ to avoid per-sample CPU overhead
    during training. This trades ~2GB RAM for near-zero CPU in __getitem__.
    """

    def __init__(
        self,
        records: list[dict],
        allele_sequences: OrderedDict[str, str],
        tokenizer: EsmSequenceTokenizer,
        group_map_inv: OrderedDict[str, int],
        min_epitope_len: int = 8,
        max_epitope_len: int = 15,
        has_affinity: bool = False,
    ):
        self.has_affinity = has_affinity
        pad_id = tokenizer.pad_token_id
        cls_id = tokenizer.cls_token_id
        eos_id = tokenizer.eos_token_id
        chain_break_id = tokenizer.chain_break_token_id

        # Detect two-chain mode
        two_chain = _detect_two_chain(allele_sequences,
                                      [r["mhc"] for r in records[:100]])
        max_mhc_tokens = _compute_max_mhc_tokens(two_chain)

        # Filter valid records
        if two_chain:
            self.samples = [
                r for r in records
                if min_epitope_len <= len(r["sequence"]) <= max_epitope_len
            ]
        else:
            self.samples = [
                r for r in records
                if (
                    min_epitope_len <= len(r["sequence"]) <= max_epitope_len
                    and r["mhc"].strip() in allele_sequences
                )
            ]

        # ── Pre-compute MHC tokens (only ~hundreds of unique alleles) ──
        _mhc_token_cache: dict[str, list[int]] = {}
        for row in self.samples:
            mhc_name = row["mhc"].strip()
            if mhc_name not in _mhc_token_cache:
                _mhc_token_cache[mhc_name] = _resolve_mhc_tokens(
                    mhc_name, allele_sequences, tokenizer, chain_break_id,
                    max_mhc_tokens=max_mhc_tokens,
                    pad_id=pad_id,
                )

        # ── Pre-tokenize all samples into a single tensor ──
        import time as _time
        t0 = _time.time()
        n = len(self.samples)
        # Compute fixed token length: [CLS] + mhc + [CHAIN_BREAK] + pep + [EOS] + pad
        # All sequences have the same total length due to padding
        sample_len = None
        all_tokens_list = []
        labels = []
        groups = []
        affinities = []
        affinity_masks = []

        for row in self.samples:
            pep_seq = row["sequence"]
            mhc_name = row["mhc"].strip()
            mhc_tokens = _mhc_token_cache[mhc_name]
            pep_tokens = tokenizer.encode(pep_seq, add_special_tokens=False)
            pad_len = max_epitope_len - len(pep_seq)
            full_seq = (
                [cls_id] + mhc_tokens + [chain_break_id]
                + pep_tokens + [eos_id] + [pad_id] * pad_len
            )
            all_tokens_list.append(full_seq)
            if sample_len is None:
                sample_len = len(full_seq)

            _lbl_col = "lable" if "lable" in row else "label"
            labels.append(float(row.get(_lbl_col, 0.0)))
            groups.append(group_map_inv.get(mhc_name, -1))

            if has_affinity:
                raw_aff = row.get("affinity", "")
                if raw_aff and str(raw_aff).strip().lower() not in ("", "nan"):
                    affinities.append(float(raw_aff))
                    affinity_masks.append(True)
                else:
                    affinities.append(0.0)
                    affinity_masks.append(False)

        # Store mhc names for per-allele sampling
        self._mhc_names = [row["mhc"].strip() for row in self.samples]

        # Store as contiguous tensors
        self._tokens = torch.tensor(all_tokens_list, dtype=torch.long)
        self._labels = torch.tensor(labels, dtype=torch.float)
        self._groups = torch.tensor(groups, dtype=torch.long)
        if has_affinity:
            self._affinities = torch.tensor(affinities, dtype=torch.float)
            self._affinity_masks = torch.tensor(affinity_masks, dtype=torch.bool)

        # Free the raw records to save memory (keep samples for label access)
        elapsed = _time.time() - t0
        print(f"  [_FlatSaDataset] Pre-tokenized {n} samples in {elapsed:.1f}s "
              f"(tensor: {self._tokens.shape}, "
              f"{self._tokens.nbytes / 1024**2:.0f}MB)")

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict:
        result = {
            "sequence_tokens": self._tokens[idx],
            "label": self._labels[idx],
            "groups": self._groups[idx],
        }
        if self.has_affinity:
            result["affinity"] = self._affinities[idx]
            result["affinity_mask"] = self._affinity_masks[idx]
        return result


class _LazySaDataset(torch.utils.data.Dataset):
    """
    Lazy SA-style dataset that defers tokenization to __getitem__.

    Unlike _FlatSaDataset which pre-tokenizes all samples into a giant tensor
    (infeasible for 40M+ rows due to ~100GB+ intermediate memory), this class
    stores only compact per-record data (peptide string, allele name, label)
    and tokenizes on-the-fly in __getitem__.

    Memory: ~1.7GB for 38M records vs ~100GB+ for _FlatSaDataset.
    CPU overhead: ~10-50us per sample, negligible vs GPU forward time.
    """

    def __init__(
        self,
        records,
        allele_sequences: OrderedDict[str, str],
        tokenizer: EsmSequenceTokenizer,
        group_map_inv: OrderedDict[str, int],
        min_epitope_len: int = 8,
        max_epitope_len: int = 15,
        has_affinity: bool = False,
    ):
        self.tokenizer = tokenizer
        self.allele_sequences = allele_sequences
        self.group_map_inv = group_map_inv
        self.max_epitope_len = max_epitope_len
        self.has_affinity = has_affinity
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id
        self.chain_break_id = tokenizer.chain_break_token_id

        # Pre-compute MHC token cache (~hundreds of unique alleles)
        # Detect two-chain mode from allele_sequences
        self._two_chain = _detect_two_chain(allele_sequences)
        self._max_mhc_tokens = _compute_max_mhc_tokens(self._two_chain)
        self._mhc_token_cache: dict[str, list[int]] = {}
        if not self._two_chain:
            # Class I: pre-build cache from allele_sequences
            for name, seq in allele_sequences.items():
                toks = tokenizer.encode(
                    seq[:MIN_LEN], add_special_tokens=False
                )
                # Pad or truncate to fixed length
                if len(toks) > self._max_mhc_tokens:
                    toks = toks[:self._max_mhc_tokens]
                elif len(toks) < self._max_mhc_tokens:
                    toks = toks + [self.pad_id] * (self._max_mhc_tokens - len(toks))
                self._mhc_token_cache[name] = toks
        # For two-chain, cache is built lazily during filtering

        # Store only compact data — no tokenization
        import time as _time
        t0 = _time.time()

        # Support both list[dict] and compact tuple (seqs, mhcs, labels, ...)
        if isinstance(records, tuple):
            # Compact format from Stage2V2DataModule
            seqs, mhcs, labels = records[0], records[1], records[2]
            if has_affinity and len(records) >= 5:
                affs, aff_masks = records[3], records[4]
            else:
                affs, aff_masks = None, None

            # Filter valid records
            self._pep_seqs: list[str] = []
            self._mhc_names: list[str] = []
            self._labels: list[float] = []
            self._groups: list[int] = []
            self._affinities: list[float] = []
            self._affinity_masks: list[bool] = []

            for i in range(len(seqs)):
                pep = seqs[i]
                mhc = mhcs[i]
                if not (min_epitope_len <= len(pep) <= max_epitope_len):
                    continue
                mhc_stripped = mhc.strip()
                # For two-chain, resolve and cache on the fly
                if self._two_chain:
                    if mhc_stripped not in self._mhc_token_cache:
                        self._mhc_token_cache[mhc_stripped] = _resolve_mhc_tokens(
                            mhc_stripped, allele_sequences, tokenizer,
                            self.chain_break_id,
                            max_mhc_tokens=self._max_mhc_tokens,
                            pad_id=self.pad_id,
                        )
                else:
                    if mhc_stripped not in self._mhc_token_cache:
                        continue
                self._pep_seqs.append(pep)
                self._mhc_names.append(mhc_stripped)
                self._labels.append(labels[i])
                self._groups.append(group_map_inv.get(mhc_stripped, -1))
                if has_affinity and affs is not None:
                    self._affinities.append(affs[i])
                    self._affinity_masks.append(aff_masks[i])
        else:
            # list[dict] format
            self._pep_seqs = []
            self._mhc_names = []
            self._labels = []
            self._groups = []
            self._affinities = []
            self._affinity_masks = []

            for row in records:
                pep = row["sequence"]
                mhc = row["mhc"].strip()
                if not (min_epitope_len <= len(pep) <= max_epitope_len):
                    continue
                if self._two_chain:
                    if mhc not in self._mhc_token_cache:
                        self._mhc_token_cache[mhc] = _resolve_mhc_tokens(
                            mhc, allele_sequences, tokenizer,
                            self.chain_break_id,
                            max_mhc_tokens=self._max_mhc_tokens,
                            pad_id=self.pad_id,
                        )
                else:
                    if mhc not in self._mhc_token_cache:
                        continue
                self._pep_seqs.append(pep)
                self._mhc_names.append(mhc)
                _lbl_col = "lable" if "lable" in row else "label"
                self._labels.append(float(row.get(_lbl_col, 0.0)))
                self._groups.append(group_map_inv.get(mhc, -1))
                if has_affinity:
                    raw_aff = row.get("affinity", "")
                    if raw_aff and str(raw_aff).strip().lower() not in ("", "nan"):
                        self._affinities.append(float(raw_aff))
                        self._affinity_masks.append(True)
                    else:
                        self._affinities.append(0.0)
                        self._affinity_masks.append(False)

        elapsed = _time.time() - t0
        n = len(self._pep_seqs)
        mem_est_mb = n * 46 / 1024 / 1024  # ~46 bytes per record estimate
        print(f"  [_LazySaDataset] Indexed {n} samples in {elapsed:.1f}s "
              f"(estimated memory: {mem_est_mb:.0f}MB, lazy tokenization)")

    def __len__(self) -> int:
        return len(self._pep_seqs)

    def __getitem__(self, idx: int) -> dict:
        pep_seq = self._pep_seqs[idx]
        mhc_name = self._mhc_names[idx]

        mhc_tokens = self._mhc_token_cache[mhc_name]
        pep_tokens = self.tokenizer.encode(pep_seq, add_special_tokens=False)
        pad_len = self.max_epitope_len - len(pep_seq)

        full_seq = (
            [self.cls_id]
            + mhc_tokens
            + [self.chain_break_id]
            + pep_tokens
            + [self.eos_id]
            + [self.pad_id] * pad_len
        )
        tokens = torch.tensor(full_seq, dtype=torch.long)

        result = {
            "sequence_tokens": tokens,
            "label": torch.tensor(self._labels[idx], dtype=torch.float),
            "groups": torch.tensor(self._groups[idx], dtype=torch.long),
        }
        if self.has_affinity:
            result["affinity"] = torch.tensor(self._affinities[idx], dtype=torch.float)
            result["affinity_mask"] = torch.tensor(self._affinity_masks[idx], dtype=torch.bool)
        return result


class _CombinedFlatDataset(torch.utils.data.Dataset):
    """
    Combines SA flat dataset + MA flat dataset (pre-predicted winners).

    SA samples can optionally be repeated ``sa_repeat`` times.
    """

    def __init__(
        self,
        sa_dataset: _FlatSaDataset,
        ma_dataset: _FlatSaDataset,
        sa_repeat: int = 1,
    ):
        self.sa_dataset = sa_dataset
        self.ma_dataset = ma_dataset
        self.sa_len = len(sa_dataset)
        self.ma_len = len(ma_dataset)
        self.sa_repeat = max(1, sa_repeat)
        # Flat index: first (sa_len * sa_repeat) SA items, then ma_len MA items
        self._total = self.sa_len * self.sa_repeat + self.ma_len

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> dict:
        sa_total = self.sa_len * self.sa_repeat
        if idx < sa_total:
            return self.sa_dataset[idx % self.sa_len]
        return self.ma_dataset[idx - sa_total]

    @property
    def labels(self) -> list[float]:
        """Return label list aligned with flat indexing (for BalancedBatchSampler)."""
        # Support both tensor (_FlatSaDataset) and list (_LazySaDataset) labels
        sa_lbl = self.sa_dataset._labels
        if isinstance(sa_lbl, torch.Tensor):
            sa_labels = sa_lbl.repeat(self.sa_repeat).tolist()
        else:
            sa_labels = sa_lbl * self.sa_repeat

        ma_lbl = self.ma_dataset._labels
        if isinstance(ma_lbl, torch.Tensor):
            ma_labels = ma_lbl.tolist()
        else:
            ma_labels = ma_lbl

        return sa_labels + ma_labels

    @property
    def mhc_names(self) -> list[str]:
        """Return MHC name list aligned with flat indexing (for AlleleBalancedBatchSampler)."""
        sa_mhc = self.sa_dataset._mhc_names
        if self.sa_repeat > 1:
            sa_mhcs = sa_mhc * self.sa_repeat
        else:
            sa_mhcs = sa_mhc
        ma_mhcs = self.ma_dataset._mhc_names
        return sa_mhcs + ma_mhcs


class Stage2V2DataModule(pl.LightningDataModule):
    """
    Stage-2 V2 DataModule: Pre-predict MA winners → flatten → merge with SA.

    Instead of running MIL winner-takes-all inside every training batch
    (which requires an extra no_grad forward pass over all K candidates),
    this DataModule:

    1. Before each epoch, runs batch inference to predict the winner allele
       for each MA record (the allele with highest EL logit).
    2. Flattens MA data into SA-style records (peptide, winner_allele, label).
    3. Merges flattened MA with SA data for standard flat training.
    4. After each epoch, the training script calls ``update_ma_predictions()``
       to re-predict winners with the updated model.

    This eliminates the per-batch MIL overhead, making training ~2× faster
    for the same number of effective training samples.

    Args:
        sa_path: Path to SA CSV.
        ma_path: Path to MA CSV.
        allele_sequences: Allele name → pseudo-sequence mapping.
        batch_size: Training batch size.
        num_workers: DataLoader workers.
        pin_memory: Pin CUDA memory.
        min_epitope_len: Min peptide length.
        max_epitope_len: Max peptide length.
        sa_repeat: Upsample SA data by repeating N times.
        balance_sampling: Use BalancedBatchSampler for class balance.
        pos_frac_target: Target positive fraction when balance_sampling=True.
        predict_batch_size: Batch size for MA prediction inference.
        predict_num_workers: DataLoader workers for MA prediction.
    """

    def __init__(
        self,
        sa_path: str,
        ma_path: str,
        allele_sequences: OrderedDict[str, str],
        batch_size: int = 64,
        num_workers: int = 2,
        pin_memory: bool = True,
        min_epitope_len: int = 8,
        max_epitope_len: int = 15,
        sa_repeat: int = 1,
        balance_sampling: bool = False,
        pos_frac_target: float = 0.2,
        predict_batch_size: int = 512,
        predict_num_workers: int = 4,
        hla_only: bool = False,
        allele_balanced: bool = False,
    ):
        super().__init__()
        self.hla_only = hla_only
        self.allele_balanced = allele_balanced
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.min_epitope_len = min_epitope_len
        self.max_epitope_len = max_epitope_len
        self.sa_repeat = sa_repeat
        self.balance_sampling = balance_sampling
        self.pos_frac_target = pos_frac_target
        self.predict_batch_size = predict_batch_size
        self.predict_num_workers = predict_num_workers

        self.tokenizer = EsmSequenceTokenizer()
        self.allele_sequences = allele_sequences

        # ── Load SA data into compact lists (avoid 40M+ dicts in memory) ──
        import time as _time
        t0 = _time.time()
        sa_seqs: list[str] = []
        sa_mhcs: list[str] = []
        sa_labels: list[float] = []
        sa_affinities: list[float] = []
        sa_affinity_masks: list[bool] = []
        _has_affinity = False
        _lbl_col = None

        with open(sa_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pep = row["sequence"]
                if not (min_epitope_len <= len(pep) <= max_epitope_len):
                    continue
                if hla_only and not row["mhc"].startswith("HLA-"):
                    continue
                # Detect label column and affinity on first valid row
                if _lbl_col is None:
                    _lbl_col = "lable" if "lable" in row else "label"
                    _has_affinity = "affinity" in row

                sa_seqs.append(pep)
                sa_mhcs.append(row["mhc"])
                sa_labels.append(float(row.get(_lbl_col) or row.get("label") or 0.0))
                if _has_affinity:
                    raw_aff = row.get("affinity", "")
                    if raw_aff and str(raw_aff).strip().lower() not in ("", "nan"):
                        sa_affinities.append(float(raw_aff))
                        sa_affinity_masks.append(True)
                    else:
                        sa_affinities.append(0.0)
                        sa_affinity_masks.append(False)

        self._has_affinity = _has_affinity
        n_sa = len(sa_seqs)
        elapsed = _time.time() - t0
        print(f"Stage2V2: Loaded {n_sa} SA records in {elapsed:.1f}s (compact lists)")

        # Train/val split on indices (avoid duplicating data)
        all_indices = list(range(n_sa))
        train_idx, val_idx = train_test_split(
            all_indices, test_size=0.05, random_state=42
        )
        train_idx_set = set(train_idx)
        val_idx_set = set(val_idx)

        # Build compact train/val tuples
        sa_train_seqs: list[str] = []
        sa_train_mhcs: list[str] = []
        sa_train_labels: list[float] = []
        sa_train_affs: list[float] = []
        sa_train_aff_masks: list[bool] = []

        sa_val_records: list[dict] = []  # val is small, keep as dicts

        for i in train_idx:
            sa_train_seqs.append(sa_seqs[i])
            sa_train_mhcs.append(sa_mhcs[i])
            sa_train_labels.append(sa_labels[i])
            if _has_affinity:
                sa_train_affs.append(sa_affinities[i])
                sa_train_aff_masks.append(sa_affinity_masks[i])

        for i in val_idx:
            d = {"sequence": sa_seqs[i], "mhc": sa_mhcs[i], _lbl_col: str(sa_labels[i])}
            if _has_affinity:
                d["affinity"] = str(sa_affinities[i])
            sa_val_records.append(d)

        # Free the full lists
        del sa_seqs, sa_mhcs, sa_labels, sa_affinities, sa_affinity_masks, all_indices

        # Store compact SA train data as tuple
        if _has_affinity:
            self._sa_train = (sa_train_seqs, sa_train_mhcs, sa_train_labels,
                              sa_train_affs, sa_train_aff_masks)
        else:
            self._sa_train = (sa_train_seqs, sa_train_mhcs, sa_train_labels)
        self._sa_val = sa_val_records  # small, keep as list[dict]

        # ── Load MA data (raw records, kept for re-prediction) ──
        with open(ma_path, "r") as f:
            ma_raw = list(csv.DictReader(f))
        self._ma_records = [
            s for s in ma_raw
            if min_epitope_len <= len(s["sequence"]) <= max_epitope_len
            and (not hla_only or all(
                a.strip().startswith("HLA-") for a in s["mhc"].split(",")
            ))
        ]

        # Flattened MA records (updated each epoch by update_ma_predictions)
        self._ma_flattened: list[dict] = []

        # ── Label stats ──
        self._sa_train_labels = sa_train_labels
        n_pos = sum(1 for l in sa_train_labels if l > 0.5)
        n_neg = len(sa_train_labels) - n_pos
        print(f"Stage2V2 SA train: {len(sa_train_labels)}, val: {len(sa_val_records)}")
        print(f"Stage2V2 MA records: {len(self._ma_records)} (will be flattened each epoch)")
        print(f"Stage2V2 SA label dist: pos={n_pos}, neg={n_neg}")
        if balance_sampling:
            print(f"Stage2V2 balance_sampling enabled: target pos_frac={pos_frac_target:.2f}")

        # ── Allele group map ──
        # For two-chain (class II), build group map from data MHC names
        self._two_chain = _detect_two_chain(allele_sequences, sa_train_mhcs[:100])
        if self._two_chain:
            all_mhc_names = list(OrderedDict.fromkeys(sa_train_mhcs))
            self.group_map = OrderedDict((i, k) for i, k in enumerate(all_mhc_names))
        else:
            self.group_map = OrderedDict(
                (i, k) for i, k in enumerate(allele_sequences.keys())
            )
        self.group_map_inv = OrderedDict(
            (k, i) for i, k in self.group_map.items()
        )

        # Internal refs to train dataset (rebuilt each epoch)
        self._train_ds: Optional[_CombinedFlatDataset] = None

    def initial_ma_flatten(self) -> None:
        """
        Create initial MA flattened records without model prediction.

        For the very first epoch (before a model exists or is loaded),
        randomly pick one allele per MA record as the initial "winner".
        """
        rng = random.Random(42)
        self._ma_flattened = []
        for row in self._ma_records:
            alleles_raw = [a.strip() for a in row["mhc"].split(",")]
            if self._two_chain:
                alleles = alleles_raw  # Accept all for class II
            else:
                alleles = [a for a in alleles_raw if a in self.allele_sequences]
            if not alleles:
                continue
            winner = rng.choice(alleles)
            _lbl_col = "lable" if "lable" in row else "label"
            self._ma_flattened.append({
                "sequence": row["sequence"],
                "mhc": winner,
                "label": float(row.get(_lbl_col, 1.0)),
            })
        print(f"Stage2V2: Initial MA flatten → {len(self._ma_flattened)} records "
              f"(random allele selection)")

    def save_ma_flatten(self, path: str) -> None:
        """Save current MA flattened records to a CSV file."""
        import pandas as pd
        if not self._ma_flattened:
            raise RuntimeError("No MA flattened records to save. Run initial_ma_flatten or update_ma_predictions first.")
        df = pd.DataFrame(self._ma_flattened)
        df.to_csv(path, index=False)
        print(f"Stage2V2: Saved {len(self._ma_flattened)} MA flattened records to {path}")

    def load_ma_flatten(self, path: str) -> None:
        """Load pre-computed MA flattened records from a CSV file."""
        import pandas as pd
        df = pd.read_csv(path)
        self._ma_flattened = df.to_dict("records")
        print(f"Stage2V2: Loaded {len(self._ma_flattened)} MA flattened records from {path}")

    def update_ma_predictions(self, model) -> None:
        """
        Re-predict winner alleles for all MA records using the current model.

        Called by the training script at the start of each epoch (after the
        first epoch which uses ``initial_ma_flatten``).

        Args:
            model: A PmhcuniModel on GPU, used for inference.
        """
        print("[Stage2V2] Predicting MA winners ...")
        import time
        t0 = time.time()
        self._ma_flattened = batch_predict_ma_winners(
            model=model,
            ma_records=self._ma_records,
            allele_sequences=self.allele_sequences,
            tokenizer=self.tokenizer,
            min_epitope_len=self.min_epitope_len,
            max_epitope_len=self.max_epitope_len,
            batch_size=self.predict_batch_size,
            num_workers=self.predict_num_workers,
        )
        elapsed = time.time() - t0
        print(f"[Stage2V2] MA prediction done: {len(self._ma_flattened)} records, "
              f"{elapsed:.1f}s elapsed")

    def _rebuild_train_dataset(self) -> None:
        """Rebuild training dataset from current SA + flattened MA records."""
        common_kw = dict(
            allele_sequences=self.allele_sequences,
            tokenizer=self.tokenizer,
            group_map_inv=self.group_map_inv,
            min_epitope_len=self.min_epitope_len,
            max_epitope_len=self.max_epitope_len,
        )
        # SA: use lazy dataset (40M+ records, avoid OOM)
        sa_ds = _LazySaDataset(
            self._sa_train, has_affinity=self._has_affinity, **common_kw
        )
        # MA: use pre-tokenized dataset (only ~339K records, fast & small)
        ma_ds = _FlatSaDataset(
            self._ma_flattened, has_affinity=self._has_affinity, **common_kw
        )
        self._train_ds = _CombinedFlatDataset(sa_ds, ma_ds, sa_repeat=self.sa_repeat)
        print(f"[Stage2V2] Train dataset rebuilt: "
              f"SA={len(sa_ds)}×{self.sa_repeat} + MA={len(ma_ds)} "
              f"= {len(self._train_ds)} total")

    # ──────────────────────────────────────────────
    # DataModule interface
    # ──────────────────────────────────────────────
    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        # Initial flatten (random allele) if no predictions yet
        if not self._ma_flattened:
            self.initial_ma_flatten()
        self._rebuild_train_dataset()

        # Validation: SA only
        self._val_ds = _SaValDataset(
            self._sa_val,
            allele_sequences=self.allele_sequences,
            tokenizer=self.tokenizer,
            group_map_inv=self.group_map_inv,
            min_epitope_len=self.min_epitope_len,
            max_epitope_len=self.max_epitope_len,
        )

    def _get_ddp_rank_info(self) -> tuple[int, int]:
        if self.trainer is not None:
            return self.trainer.global_rank, self.trainer.world_size
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return torch.distributed.get_rank(), torch.distributed.get_world_size()
        except Exception:
            pass
        return 0, 1

    def train_dataloader(self):
        if self._train_ds is None:
            self._rebuild_train_dataset()

        if self.balance_sampling:
            combined_labels = self._train_ds.labels
            rank, num_ranks = self._get_ddp_rank_info()

            if self.allele_balanced:
                combined_mhcs = self._train_ds.mhc_names
                batch_sampler = AlleleBalancedBatchSampler(
                    labels=combined_labels,
                    mhc_names=combined_mhcs,
                    batch_size=self.batch_size,
                    pos_frac_target=self.pos_frac_target,
                    seed=42,
                    rank=rank,
                    num_ranks=num_ranks,
                )
            else:
                batch_sampler = BalancedBatchSampler(
                    labels=combined_labels,
                    batch_size=self.batch_size,
                    pos_frac_target=self.pos_frac_target,
                    seed=42,
                    rank=rank,
                    num_ranks=num_ranks,
                )
            if rank == 0:
                sampler_name = type(batch_sampler).__name__
                print(
                    f"Stage2V2 {sampler_name}: {batch_sampler.num_batches} batches/epoch, "
                    f"local batch={batch_sampler.batch_size}/GPU × {num_ranks} GPUs "
                    f"= global {batch_sampler.global_batch_size}"
                )
            return torch.utils.data.DataLoader(
                self._train_ds,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

        return torch.utils.data.DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
