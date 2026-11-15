"""
PmhcuniModel: Epitope-level prediction model using ESM C as backbone.

Replaces the original model's ESM-2 backbone with ESM C (300M or 600M).
This is the no-flank variant — only peptide sequence features are used.

Architecture:
    Peptide tokens
        → ESM C Backbone (frozen or fine-tuned)
        → Pooling (mean / attention / concat of mean+max+attn)
        → CLS embedding concatenated with pooled embedding
        → EL head (BCEWithLogits) and/or BA head (MSE/Huber regression)

Pooling modes (``pooling_mode``):
    "mean"   — CLS + mask-mean (2 × d_model, default / backward-compat)
    "attn"   — CLS + attention-weighted (2 × d_model)
    "concat" — CLS + mean + max + attn (4 × d_model)

Multi-task learning (``use_ba_head=True``):
    Adds a Binding Affinity regression head sharing the same backbone.
    EL and BA tasks can be present simultaneously in a batch; each loss is
    computed only when the corresponding label key is present.

Reference for ESM C model:
    https://github.com/evolutionaryscale/esm/blob/main/esm/models/esmc.py
"""

from __future__ import annotations

from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

from esm.models.esmc import ESMC
from esm.tokenization import EsmSequenceTokenizer

from pmhcuni.metrics import GroupAveragePrecision


# ──────────────────────────────────────────────────────────────────
# ESM C backbone configurations
# ──────────────────────────────────────────────────────────────────
ESMC_CONFIGS = {
    "esmc_300m": {"d_model": 960, "n_heads": 15, "n_layers": 30},
    "esmc_600m": {"d_model": 1152, "n_heads": 18, "n_layers": 36},
}

# Valid pooling modes
_POOLING_MODES = ("cls", "mean", "attn", "concat", "sep_concat")


def _reinit_module(m):
    """Re-initialise weights for the random-init backbone control (CL1)."""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ──────────────────────────────────────────────────────────────────
# Attention Pooling module
# ──────────────────────────────────────────────────────────────────
class AttentionPooling(nn.Module):
    """
    Learnable attention-weighted pooling over sequence positions.

    A single linear projection maps each token embedding to a scalar score.
    Scores are masked (PAD/EOS/BOS excluded), normalised with softmax, then
    used to compute a weighted sum of the embeddings.

    Args:
        d_model: Embedding dimension.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.attn_proj = nn.Linear(d_model, 1, bias=False)

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, L, d_model)
        mask: torch.Tensor,        # (B, L) bool, True = include
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:  # (B, d_model)
        """
        Args:
            embeddings: Token embeddings of shape (B, L, d_model).
            mask: Boolean mask of shape (B, L); True = valid position.

        Returns:
            Attention-pooled tensor of shape (B, d_model).
        """
        scores = self.attn_proj(embeddings).squeeze(-1)       # (B, L)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)                # (B, L)
        # Replace NaN arising from all-masked rows with 0 (fall back to zero vec)
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = (embeddings * weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
        if return_weights:
            return pooled, weights
        return pooled


class PmhcuniModel(pl.LightningModule):
    """
    Epitope eluted-ligand (EL) prediction model based on ESM C, with optional
    Binding Affinity (BA) regression head for multi-task learning.

    No-flank mode: only uses the peptide sequence (no flanking regions).

    Args:
        esmc_model_name: Name of the pretrained ESM C model.
        hidden_dim: Hidden dimension for the classification/regression heads.
        lr: Learning rate for the classification head.
        lr_backbone: Learning rate for the top ESM C backbone layers.
        weight_decay: Weight decay for the optimizer.
        freeze_backbone: Whether to freeze the ESM C backbone entirely.
        allele_sequences: Ordered dict mapping allele name → sequence (for metrics).
        pos_weight: Positive-class weight for weighted BCE loss.  When None,
            a default is computed from ``pos_frac`` / ``pos_neg_ratio`` if
            provided; otherwise the loss is unweighted.
        pos_frac: Fraction of positives in the training set (used to derive
            ``pos_weight`` automatically when ``pos_weight`` is None).
        pos_neg_ratio: pos/neg ratio (alternative to ``pos_frac``).
        dropout: Dropout probability in the classification head.
        llrd: Layer-wise learning-rate decay factor for the backbone.
        pooling_mode: Pooling strategy for sequence embeddings.
            ``"mean"`` – CLS + mask-mean (2 × d_model, backward-compatible default).
            ``"attn"`` – CLS + attention-weighted pool (2 × d_model).
            ``"concat"`` – CLS + mean + max + attn concatenated (4 × d_model).
        use_ba_head: Whether to add a Binding Affinity regression head.
        loss_weight_el: Scalar weight applied to the EL loss.
        loss_weight_ba: Scalar weight applied to the BA loss.
        ba_loss: Loss function for BA regression; ``"mse"`` or ``"huber"``.
        ba_label_transform: Optional transform applied to BA labels before loss.
            ``"none"`` – identity (use raw label values).
            ``"log"`` – natural log of the label.
            ``"neg_log50k"`` – ``-log(label / 50000)`` (common IC50 transform).
    """

    def __init__(
        self,
        esmc_model_name: str = "esmc_300m",
        backbone_family: str = "esmc",
        esm2_name: Optional[str] = None,
        grad_checkpoint: bool = False,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        lr_backbone: float = 1e-5,
        weight_decay: float = 0.0,
        freeze_backbone: bool = False,
        allele_sequences: Optional[OrderedDict] = None,
        load_esm_from_checkpoint=False,
        esmc_checkpoint: Optional[str] = None,
        pretrained_backbone: bool = True,
        pos_weight: Optional[float] = None,
        pos_frac: Optional[float] = None,
        pos_neg_ratio: Optional[float] = None,
        dropout: float = 0.1,
        llrd: float = 0.9,
        pooling_mode: str = "mean",
        scramble_region: bool = False,
        use_ba_head: bool = False,
        use_immuno_head: bool = False,
        loss_weight_el: float = 1.0,
        loss_weight_ba: float = 0.1,
        loss_weight_immuno: float = 0.1,
        ba_loss: str = "mse",
        ba_label_transform: str = "none",
        el_loss: str = "bce",
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
        ba_huber_delta: float = 1.0,
        ba_pairwise_weight: float = 0.0,
        ba_pairwise_tau: float = 0.1,
        ba_pairwise_gap: float = 0.08,
        ba_pairwise_warmup: int = 2000,
        el_pairwise_weight: float = 0.0,
        el_pairwise_tau: float = 1.0,
        el_pairwise_gap: float = 0.5,
        el_pairwise_warmup: int = 1000,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["allele_sequences"])

        if pooling_mode not in _POOLING_MODES:
            raise ValueError(
                f"pooling_mode must be one of {_POOLING_MODES}, got {pooling_mode!r}"
            )

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.hidden_dim = hidden_dim
        self.esmc_model_name = esmc_model_name
        self.load_esm_from_checkpoint = load_esm_from_checkpoint
        self.esmc_checkpoint = esmc_checkpoint
        self.dropout = dropout
        self.llrd = llrd
        self.pooling_mode = pooling_mode
        self.scramble_region = scramble_region
        self.use_ba_head = use_ba_head
        self.use_immuno_head = use_immuno_head
        self.loss_weight_el = loss_weight_el
        self.loss_weight_ba = loss_weight_ba
        self.loss_weight_immuno = loss_weight_immuno
        self.ba_loss = ba_loss
        self.ba_label_transform = ba_label_transform
        self.el_loss = el_loss
        self.focal_gamma = focal_gamma
        # BA regression extras (all OFF by default → zero behavior change):
        #   Huber delta (1.0 ≈ MSE on our [0,1] target range) and an optional
        #   per-allele pairwise RankNet term (weight 0 = feature disabled).
        self.ba_huber_delta = ba_huber_delta
        self.ba_pairwise_weight = ba_pairwise_weight
        self.ba_pairwise_tau = ba_pairwise_tau
        self.ba_pairwise_gap = ba_pairwise_gap
        self.ba_pairwise_warmup = ba_pairwise_warmup
        #   per-allele pairwise RankNet term for the EL head (binary). For binary
        #   labels this degenerates to a uniform-weight within-allele AUC/AP
        #   surrogate (every pos-neg pair valid; pos-pos/neg-neg excluded). Needs
        #   a grouped sampler (ELGroupedBatchSampler) to have pairs to work on;
        #   tau is larger than BA's (1.0 vs 0.1) because EL logits span a wider
        #   range than 1-log50k affinities.
        self.el_pairwise_weight = el_pairwise_weight
        self.el_pairwise_tau = el_pairwise_tau
        self.el_pairwise_gap = el_pairwise_gap
        self.el_pairwise_warmup = el_pairwise_warmup
        #   linear LR warmup over the first `warmup_steps` optimizer steps (0=off).
        #   Essential for large backbones (ESM-C-600M): random heads at full LR
        #   diverge in the first few steps without it.
        self.warmup_steps = warmup_steps

        # ── Resolve focal_alpha ──
        if el_loss == "focal":
            if focal_alpha is not None:
                self.focal_alpha = float(focal_alpha)
            elif pos_frac is not None and 0.0 < pos_frac < 1.0:
                self.focal_alpha = 1.0 - pos_frac
            else:
                self.focal_alpha = 0.25
        else:
            self.focal_alpha = focal_alpha  # store but unused for BCE

        # ── Resolve pos_weight ──
        # Priority: explicit pos_weight > derived from pos_frac > derived from pos_neg_ratio
        if pos_weight is not None:
            self._pos_weight_value: Optional[float] = float(pos_weight)
        elif pos_frac is not None and 0.0 < pos_frac < 1.0:
            self._pos_weight_value = (1.0 - pos_frac) / pos_frac
        elif pos_neg_ratio is not None and pos_neg_ratio > 0.0:
            self._pos_weight_value = 1.0 / pos_neg_ratio
        else:
            self._pos_weight_value = None

        # ── Load backbone: ESM-C family (default) or ESM2 (HF transformers) ──
        self._family = backbone_family
        if backbone_family == "esm2":
            # Backbone-size/family ablation: ESM2 via HuggingFace. Use mean/cls pooling
            # (ESM2 has no [CHAIN_BREAK]; sep_concat not applicable).
            from transformers import EsmModel, AutoTokenizer
            self.backbone = EsmModel.from_pretrained(esm2_name, add_pooling_layer=False)
            self.d_model = self.backbone.config.hidden_size
            _e2 = AutoTokenizer.from_pretrained(esm2_name)
            self._pool_pad_id = _e2.pad_token_id
            self._pool_eos_id = _e2.eos_token_id
            if not pretrained_backbone:
                self.backbone.apply(_reinit_module)
                print("[INFO] ESM2 backbone RE-INITIALISED (pretrained_backbone=False)")
            print(f"[INFO] backbone=ESM2 {esm2_name} d_model={self.d_model}")
            if grad_checkpoint:
                self.backbone.gradient_checkpointing_enable()
                self.backbone.config.use_cache = False
                print("[INFO] ESM2 gradient checkpointing enabled (full-FT memory)")
        else:
            if self.load_esm_from_checkpoint:
                self.backbone: ESMC = ESMC.from_pretrained(
                    esmc_model_name, device=torch.device("cpu"),
                    from_pretrained=esmc_checkpoint, local_files_only=True)
            else:
                self.backbone: ESMC = ESMC.from_pretrained(
                    esmc_model_name, device=torch.device("cpu"))
            # Remove the original sequence_head (LM head) — keep embed + transformer.
            self.backbone.sequence_head = nn.Identity()  # type: ignore[assignment]
            # Random-init control (CL1): same ESM-C arch, discard pretrained weights.
            if not pretrained_backbone:
                self.backbone.apply(_reinit_module)
                print("[INFO] backbone RE-INITIALISED (pretrained_backbone=False) — CL1 control")
            self.d_model = ESMC_CONFIGS[esmc_model_name]["d_model"]
            self._pool_pad_id = self.backbone.tokenizer.pad_token_id
            self._pool_eos_id = self.backbone.tokenizer.eos_token_id

        # Optionally freeze the backbone (frozen-probe backbone ablation, or CL2 frozen).
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── Attention pooling ──
        if pooling_mode in ("attn", "concat"):
            self.attention_pool = AttentionPooling(self.d_model)
        elif pooling_mode == "sep_concat":
            self.mhc_attention_pool = AttentionPooling(self.d_model)
            self.pep_attention_pool = AttentionPooling(self.d_model)

        # ── Token ID constants for sep_concat mask building ──
        if pooling_mode == "sep_concat":
            if self._family == "esm2":
                # ESM2 has no chain_break token — data.py inserts ESM2's <eos> as the
                # MHC|peptide separator; _build_separate_masks' argmax finds that FIRST
                # eos (the trailing sequence-final eos is later, so it's not picked).
                self._chain_break_id = self._pool_eos_id
                self._eos_id = self._pool_eos_id
                self._pad_id = self._pool_pad_id
            else:
                from esm.tokenization import EsmSequenceTokenizer
                _tok = EsmSequenceTokenizer()
                self._chain_break_id = _tok.chain_break_token_id
                self._eos_id = _tok.eos_token_id
                self._pad_id = _tok.pad_token_id

        # ── Head input dimension ──
        # "mean" or "attn":  CLS + pooled                              → 2 × d_model
        # "concat":          CLS + mean + max + attn                   → 4 × d_model
        # "sep_concat":      CLS + MHC(mean+max+attn) + Pep(mean+max+attn) → 7 × d_model
        if pooling_mode == "sep_concat":
            in_dim = 7 * self.d_model
        elif pooling_mode == "concat":
            in_dim = 4 * self.d_model
        elif pooling_mode == "cls":
            in_dim = self.d_model
        else:
            in_dim = 2 * self.d_model

        # ── EL classification head (BCEWithLogits) ──
        # Named "classifier" to maintain checkpoint compatibility.
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

        # ── BA regression head (MSE/Huber) – optional ──
        if use_ba_head:
            self.ba_head = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, 1),
            )

        # ── Immunogenicity classification head (BCE) – optional ──
        # Joint multi-task FT: trained alongside the EL classifier (loss_weight_el
        # stays active) with a low loss_weight_immuno, rather than replacing/
        # freezing the EL objective (see experiments/immunogenicity_training/
        # REPORT.md section 7, Codex-scoped lever #2). Same shape as ba_head/
        # classifier — a separate head, not a repurposed one, so it can never
        # corrupt the EL classifier's weights.
        if use_immuno_head:
            self.immuno_head = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, 1),
            )

        # ── Metrics ──
        if allele_sequences is not None:
            group_map = OrderedDict(
                (i, k) for i, k in enumerate(allele_sequences.keys())
            )
        else:
            group_map = OrderedDict()
        self.el_avgp = GroupAveragePrecision(groups=group_map)
        if use_immuno_head:
            self.immuno_avgp = GroupAveragePrecision(groups=group_map)

        # Flag set by training.py; consumed in configure_model()
        self._use_compile: bool = False

    def configure_model(self) -> None:
        """
        Called by Lightning after DDP wrapping but before the first forward pass.
        ``torch.compile`` must be applied here (not in ``__init__``) so that
        DDP can wrap the original module first, and compilation sees the
        already-distributed module.
        """
        if self._use_compile:
            self.backbone = torch.compile(self.backbone)  # type: ignore[assignment]
            self._use_compile = False  # guard against double-compile on reload

    @property
    def tokenizer(self) -> EsmSequenceTokenizer:
        return self.backbone.tokenizer

    # ──────────────────────────────────────────────
    # Feature extraction
    # ──────────────────────────────────────────────
    def build_pooling_mask(self, sequence_tokens: torch.Tensor) -> torch.Tensor:
        """Return the pooling mask used for non-special, non-padding tokens."""
        mask = (sequence_tokens != self._pool_pad_id) & (sequence_tokens != self._pool_eos_id)
        mask[:, 0] = False  # exclude CLS/BOS (position 0 for both ESM-C and ESM2)
        return mask

    def _build_separate_masks(
        self, sequence_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build separate boolean masks for MHC and peptide token regions.

        Layout: [CLS] MHC_tokens [CHAIN_BREAK] Pep_tokens [EOS] [PAD...]
        Returns:
            mhc_mask: (B, L) True for MHC token positions
            pep_mask: (B, L) True for peptide token positions
        """
        B, L = sequence_tokens.shape
        device = sequence_tokens.device

        # Find chain_break position per row (guaranteed exactly one per sequence)
        cb_pos = (sequence_tokens == self._chain_break_id).int().argmax(dim=1)  # (B,)

        if getattr(self, "scramble_region", False):
            # CL3 negative control: split at a deterministic pseudo-random WRONG position
            # (not the real CHAIN_BREAK). Same 7×d head + same ops, wrong region boundary →
            # isolates whether RSAP's gain needs the CORRECT MHC|peptide split.
            eos_pos = (sequence_tokens == self._eos_id).int().argmax(dim=1)  # (B,)
            span = (eos_pos - 4).clamp(min=1)  # keep ≥1 token in BOTH regions (avoid empty-pool NaN)
            cb_pos = 2 + (sequence_tokens.long().sum(dim=1) % span)

        col_idx = torch.arange(L, device=device).unsqueeze(0)  # (1, L)
        cb_pos_expanded = cb_pos.unsqueeze(1)  # (B, 1)

        # MHC: positions in (0, cb_pos); exclude PAD and the real CHAIN_BREAK token
        # (the latter matters when cb_pos is scrambled away from the true break).
        mhc_mask = (col_idx > 0) & (col_idx < cb_pos_expanded)
        mhc_mask = mhc_mask & (sequence_tokens != self._pad_id) & (sequence_tokens != self._chain_break_id)

        # Peptide: positions after cb_pos, excluding EOS, PAD and the real CHAIN_BREAK
        pep_mask = (col_idx > cb_pos_expanded)
        pep_mask = pep_mask & (sequence_tokens != self._eos_id) & (sequence_tokens != self._pad_id) & (sequence_tokens != self._chain_break_id)

        return mhc_mask, pep_mask

    def extract_features_from_embeddings(
        self,
        sequence_tokens: torch.Tensor,
        embeddings: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        """
        Pool features from precomputed token embeddings.

        Args:
            sequence_tokens: (B, L) token IDs including BOS/EOS.
            embeddings: (B, L, d_model) token embeddings from the backbone.
            return_details: If True, return pooling intermediates needed for
                explainability.

        Returns:
            feats: (B, feat_dim)
            details (optional): dict with pooling mask / weights / embeddings.
        """
        cls_embedding = embeddings[:, 0, :]  # (B, d_model)
        mask = self.build_pooling_mask(sequence_tokens)
        attn_weights: torch.Tensor | None = None

        if self.pooling_mode == "cls":
            feats = cls_embedding  # (B, d_model) — CLS-only baseline (CL3)

        elif self.pooling_mode == "mean":
            mask_f = mask.unsqueeze(-1).float()         # (B, L, 1)
            denom = mask_f.sum(dim=1).clamp(min=1.0)   # (B, 1)
            pooled = (embeddings * mask_f).sum(dim=1) / denom  # (B, d_model)
            feats = torch.cat([cls_embedding, pooled], dim=-1)  # (B, 2*d_model)

        elif self.pooling_mode == "attn":
            if return_details:
                pooled, attn_weights = self.attention_pool(
                    embeddings, mask, return_weights=True
                )
            else:
                pooled = self.attention_pool(embeddings, mask)
            feats = torch.cat([cls_embedding, pooled], dim=-1)  # (B, 2*d_model)

        elif self.pooling_mode == "sep_concat":
            # Separate MHC / peptide pooling → 7 × d_model
            mhc_mask, pep_mask = self._build_separate_masks(sequence_tokens)

            parts = [cls_embedding]
            for region_mask, attn_pool in [
                (mhc_mask, self.mhc_attention_pool),
                (pep_mask, self.pep_attention_pool),
            ]:
                rm_f = region_mask.unsqueeze(-1).float()
                denom = rm_f.sum(dim=1).clamp(min=1.0)
                mean_p = (embeddings * rm_f).sum(dim=1) / denom
                masked_e = embeddings.masked_fill(~region_mask.unsqueeze(-1), float("-inf"))
                max_p = masked_e.max(dim=1).values
                max_p = torch.nan_to_num(max_p, nan=0.0, posinf=0.0, neginf=0.0)
                attn_p = attn_pool(embeddings, region_mask)
                parts.extend([mean_p, max_p, attn_p])

            feats = torch.cat(parts, dim=-1)  # (B, 7*d_model)

        else:  # "concat": CLS + mean + max + attn → 4 * d_model
            # Mean pooling
            mask_f = mask.unsqueeze(-1).float()         # (B, L, 1)
            denom = mask_f.sum(dim=1).clamp(min=1.0)   # (B, 1)
            mean_pooled = (embeddings * mask_f).sum(dim=1) / denom  # (B, d_model)
            # Max pooling (mask invalid positions with -inf before max, then
            # replace any -inf result from all-masked rows with 0)
            masked_emb = embeddings.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            max_pooled = masked_emb.max(dim=1).values   # (B, d_model)
            max_pooled = torch.nan_to_num(max_pooled, nan=0.0, posinf=0.0, neginf=0.0)
            # Attention pooling
            if return_details:
                attn_pooled, attn_weights = self.attention_pool(
                    embeddings, mask, return_weights=True
                )
            else:
                attn_pooled = self.attention_pool(embeddings, mask)
            feats = torch.cat(
                [cls_embedding, mean_pooled, max_pooled, attn_pooled], dim=-1
            )  # (B, 4*d_model)

        if return_details:
            details = {
                "embeddings": embeddings,
                "mask": mask,
                "cls_embedding": cls_embedding,
                "attention_pool_weights": attn_weights,
            }
            return feats, details
        return feats

    def extract_features(
        self,
        sequence_tokens: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        """
        Run ESM C forward and extract a pooled feature vector.

        The mask used for pooling excludes:
          - Position 0 (CLS/BOS token)
          - EOS token positions
          - PAD token positions

        Pooling behaviour is controlled by ``self.pooling_mode``:
          - ``"mean"``   → CLS + mask-mean  (B, 2 × d_model)
          - ``"attn"``   → CLS + attention-pool  (B, 2 × d_model)
          - ``"concat"`` → CLS + mean + max + attn  (B, 4 × d_model)

        Args:
            sequence_tokens: (B, L) token IDs including BOS/EOS.

        Returns:
            feats: (B, feat_dim) where feat_dim depends on ``pooling_mode``.
        """
        if self._family == "esm2":
            attn = (sequence_tokens != self._pool_pad_id).long()
            embeddings = self.backbone(
                input_ids=sequence_tokens, attention_mask=attn
            ).last_hidden_state
        else:
            embeddings = self.backbone(sequence_tokens=sequence_tokens).embeddings
        return self.extract_features_from_embeddings(
            sequence_tokens, embeddings, return_details=return_details
        )

    # ──────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────
    def forward(
        self,
        sequence_tokens: torch.Tensor,
        return_feats: bool = False,
        task: str = "el",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            sequence_tokens: (B, L) tokenized peptide sequences.
            return_feats: If True, also return the pooled feature tensor.
            task: Which head to apply; ``"el"`` (default) for EL classification,
                ``"ba"`` for BA regression (requires ``use_ba_head=True``), or
                ``"immuno"`` for immunogenicity classification (requires
                ``use_immuno_head=True``).

        Returns:
            logits: (B, 1) — EL/immuno logits or BA predictions depending on ``task``.
            feats (optional): (B, feat_dim) pooled features.
        """
        feats = self.extract_features(sequence_tokens)
        if task == "ba" and self.use_ba_head:
            logits = self.ba_head(feats)
        elif task == "immuno" and self.use_immuno_head:
            logits = self.immuno_head(feats)
        else:
            logits = self.classifier(feats)  # EL head (default)

        if return_feats:
            return logits, feats
        return logits

    # ──────────────────────────────────────────────
    # Loss
    # ──────────────────────────────────────────────
    def _focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Focal loss: -alpha_t * (1 - p_t)^gamma * BCE(logits, labels)."""
        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1).float()
        # Numerically stable BCE per element
        bce = F.binary_cross_entropy_with_logits(
            logits_flat, labels_flat, reduction="none"
        )
        p = torch.sigmoid(logits_flat)
        p_t = p * labels_flat + (1.0 - p) * (1.0 - labels_flat)
        focal_weight = (1.0 - p_t) ** self.focal_gamma
        alpha_t = (
            self.focal_alpha * labels_flat
            + (1.0 - self.focal_alpha) * (1.0 - labels_flat)
        )
        return (alpha_t * focal_weight * bce).mean()

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        EL classification loss. Dispatches to focal loss or weighted BCE
        depending on ``self.el_loss``.
        """
        if self.el_loss == "focal":
            return self._focal_loss(logits, labels)
        # ── Weighted BCE (default) ──
        pos_weight_tensor: Optional[torch.Tensor] = None
        if self._pos_weight_value is not None:
            pos_weight_tensor = torch.tensor(
                [self._pos_weight_value],
                device=logits.device,
                dtype=logits.dtype,
            )
        return F.binary_cross_entropy_with_logits(
            logits.view(-1), labels.view(-1), pos_weight=pos_weight_tensor
        )

    def _transform_ba_label(self, ba_label: torch.Tensor) -> torch.Tensor:
        """
        Apply the configured label transform to raw BA labels.

        Transforms:
            ``"none"``      — identity (returns label as float).
            ``"log"``       — natural logarithm of the label.
            ``"neg_log50k"``— ``-log(label / 50000)`` (typical IC50 transform).

        Args:
            ba_label: Raw BA label tensor.

        Returns:
            Transformed label tensor (float32).
        """
        x = ba_label.float()
        if self.ba_label_transform == "log":
            return torch.log(x.clamp(min=1e-8))
        if self.ba_label_transform == "neg_log50k":
            return -torch.log(x.clamp(min=1e-8) / 50000.0)
        return x  # "none"

    def compute_ba_loss(
        self, ba_logits: torch.Tensor, ba_labels: torch.Tensor,
        ineq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Regression loss for the BA head (MSE or Huber), with optional
        CENSORED (inequality-aware) handling for `>`/`<` measurements.

        Args:
            ba_logits: Predicted BA values (B, 1) or (B,).
            ba_labels: Ground-truth BA values (already transformed; 1-log50k,
                higher = stronger binding).
            ineq: Optional per-row censoring flag (B,): 0=exact ('='),
                +1=IC50> (weaker than threshold -> true y <= tgt; penalize only
                if pred > tgt), -1=IC50< (stronger -> true y >= tgt; penalize
                only if pred < tgt). None -> plain MSE/Huber on all rows.
        """
        pred = ba_logits.view(-1)
        tgt = ba_labels.view(-1)
        if ineq is None:
            if self.ba_loss == "huber":
                return F.huber_loss(pred, tgt, delta=self.ba_huber_delta)
            return F.mse_loss(pred, tgt)
        # censored loss: squared error on exact, one-sided squared hinge on censored
        q = ineq.view(-1).to(pred.dtype)
        diff = pred - tgt
        exact_l = diff ** 2
        gt_l = torch.clamp(diff, min=0.0) ** 2     # IC50> : penalize pred above tgt
        lt_l = torch.clamp(-diff, min=0.0) ** 2    # IC50< : penalize pred below tgt
        loss = torch.where(q > 0, gt_l, torch.where(q < 0, lt_l, exact_l))
        return loss.mean()

    def _pairwise_rank_loss(
        self, pred: torch.Tensor, tgt: torch.Tensor, groups: torch.Tensor,
        tau: Optional[float] = None, gap: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Per-allele pairwise RankNet loss for the BA head (within-allele ranking).

        BA is fundamentally a within-allele *ranking* problem, so this term
        complements the absolute-value MSE/Huber loss. For each allele (group)
        with ≥2 rows, form all pairs (i<j) whose target gap
        ``|tgt_i - tgt_j| >= self.ba_pairwise_gap``. RankNet per valid pair::

            softplus( -sign(tgt_i - tgt_j) * (pred_i - pred_j) / tau )

        weighted by ``clamp(|tgt_i - tgt_j| / 0.3, max=1.0)`` (larger true gaps
        matter more). The weighted mean is taken within each allele, then the
        plain mean across alleles that had ≥1 valid pair. Returns a grad-safe
        0.0 tensor when no valid pair exists anywhere.

        Args:
            pred:   (N,) predicted affinities for the BA-labeled rows.
            tgt:    (N,) ground-truth affinities (already transformed).
            groups: (N,) integer allele id per row (same length as pred/tgt).
        """
        # Resolve tau/gap: default to the BA-head values for backward-compat;
        # the EL head passes its own (tau=1.0 for wider-range logits, gap=0.5
        # for binary labels).
        tau = self.ba_pairwise_tau if tau is None else tau
        min_gap = self.ba_pairwise_gap if gap is None else gap
        pred = pred.view(-1)
        tgt = tgt.view(-1)
        # grad-safe zero on the right device/dtype (keeps the graph connected)
        zero = pred.sum() * 0.0
        if pred.numel() < 2:
            return zero

        allele_losses = []
        for a in torch.unique(groups):
            idx = (groups == a).nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue
            p = pred[idx]
            t = tgt[idx]
            n = idx.numel()
            iu = torch.triu_indices(n, n, offset=1, device=pred.device)
            gap = t[iu[0]] - t[iu[1]]          # tgt_i - tgt_j for all pairs i<j
            valid = gap.abs() >= min_gap
            if not bool(valid.any()):
                continue
            gap = gap[valid]
            dp = (p[iu[0]] - p[iu[1]])[valid]  # pred_i - pred_j
            w = torch.clamp(gap.abs() / 0.3, max=1.0)
            per_pair = F.softplus(-torch.sign(gap) * dp / tau)
            allele_losses.append((w * per_pair).sum() / w.sum())

        if not allele_losses:
            return zero
        return torch.stack(allele_losses).mean()

    # ──────────────────────────────────────────────
    # Training / Validation steps
    # ──────────────────────────────────────────────
    def _mil_loss(self, batch: dict) -> torch.Tensor:
        """
        Winner-takes-all MIL loss for stage-2 combined SA + MA batches.

        Memory-efficient implementation using hard argmax + no_grad strategy to
        avoid building computation graphs and storing ESM-C backbone activations
        for all K candidates simultaneously:

        - Phase A (no_grad): Score all K candidates without gradient. Use argmax
          on masked logits to identify the winner index for each MA sample.
        - Phase B (with_grad): Forward *only* the selected tokens (candidate 0
          for SA; winner candidate for MA) with full gradient, so backprop covers
          only the paths that affect the loss.

        For SA samples (is_sa=True):  use the single candidate logit directly.
        For MA samples (is_sa=False): apply bag_mask, select winner via argmax,
                                      compute BCE on the winner logit vs label.

        Args:
            batch: dict with keys
                ``sequence_tokens`` (B, K_max, L),
                ``bag_mask``        (B, K_max),
                ``label``           (B,),
                ``is_sa``           (B,).

        Returns:
            Scalar loss tensor.
        """
        seq_tokens = batch["sequence_tokens"]   # (B, K_max, L)
        bag_mask   = batch["bag_mask"]          # (B, K_max)
        labels     = batch["label"]             # (B,)
        is_sa      = batch["is_sa"]             # (B,)

        B, K_max, L = seq_tokens.shape
        ma_mask = ~is_sa

        # ── Phase A (no_grad): score all candidates to identify MA winners ──
        # Running inference without building a computation graph avoids storing
        # ESM-C backbone activations for all K*B candidates simultaneously,
        # keeping peak memory close to a single-forward-pass baseline.
        winner_idx: torch.Tensor  # populated inside the no_grad block when ma_mask.any()
        with torch.no_grad():
            logits_flat = self.forward(seq_tokens.view(B * K_max, L))  # (B*K_max, 1)
            logits_all = logits_flat.view(B, K_max)                    # (B, K_max)
            # Mask invalid candidates before argmax; ties broken by first occurrence
            logits_masked = logits_all.masked_fill(~bag_mask, float("-inf"))
            if ma_mask.any():
                winner_idx = logits_masked[ma_mask].argmax(dim=1)      # (n_ma,)

        # ── Phase B (with_grad): forward only the selected candidate tokens ──
        # Only winner / SA-0 tokens pass through the backbone with gradient,
        # so backprop is restricted to the paths that contributed to the loss.
        selected_tokens: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        # SA: use candidate index 0
        if is_sa.any():
            selected_tokens.append(seq_tokens[is_sa, 0, :])            # (n_sa, L)
            all_labels.append(labels[is_sa])

        # MA: use the winner candidate identified in Phase A
        if ma_mask.any():
            ma_tokens = seq_tokens[ma_mask]                             # (n_ma, K_max, L)
            n_ma = ma_tokens.shape[0]
            winner_tokens = ma_tokens[
                torch.arange(n_ma, device=seq_tokens.device), winner_idx, :
            ]                                                           # (n_ma, L)
            selected_tokens.append(winner_tokens)
            all_labels.append(labels[ma_mask])

        combined_tokens = torch.cat(selected_tokens, dim=0)            # (N_selected, L)
        combined_labels = torch.cat(all_labels, dim=0)                 # (N_selected,)
        selected_logits = self.forward(combined_tokens)                 # (N_selected, 1)
        return self.compute_loss(selected_logits, combined_labels)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        if "bag_mask" in batch:
            # Stage-2: MIL winner-takes-all EL loss (SA + MA combined)
            el_loss = self._mil_loss(batch)
            total_loss = self.loss_weight_el * el_loss

            # BA loss: only SA samples may carry affinity labels;
            # MA samples never have affinity, so affinity_mask is False for them.
            ba_loss = torch.tensor(0.0, device=self.device, dtype=el_loss.dtype)
            if self.use_ba_head and "affinity" in batch:
                affinity_mask = batch.get(
                    "affinity_mask",
                    torch.ones(batch["affinity"].shape[0], dtype=torch.bool, device=self.device),
                )
                # Further restrict to SA samples only (MA never carry affinity)
                sa_affinity_mask = affinity_mask & batch["is_sa"]
                if sa_affinity_mask.any():
                    # BA head operates on flat features; run a dedicated forward
                    # for the masked SA samples only (no bag dimension needed)
                    sa_tokens = batch["sequence_tokens"][sa_affinity_mask, 0, :]  # (n, L)
                    ba_feats = self.extract_features(sa_tokens)
                    ba_logits = self.ba_head(ba_feats)
                    ba_label = self._transform_ba_label(batch["affinity"][sa_affinity_mask])
                    ba_loss = self.compute_ba_loss(ba_logits, ba_label)
                    total_loss = total_loss + self.loss_weight_ba * ba_loss

            self.log("train/el_loss", el_loss, prog_bar=True)
            self.log("train/ba_loss", ba_loss, prog_bar=True)
            self.log("train/loss", total_loss, prog_bar=True)
            return total_loss

        # Stage-1 / BA training: compute shared features once
        feats = self.extract_features(batch["sequence_tokens"])
        total_loss = torch.tensor(0.0, device=self.device, dtype=feats.dtype)

        has_el = "label" in batch

        # BA: need affinity key, BA head enabled, and at least one valid label
        # (indicated by affinity_mask; if mask absent, all samples are assumed valid)
        has_ba = False
        affinity_mask: Optional[torch.Tensor] = None
        if self.use_ba_head and "affinity" in batch:
            affinity_mask = batch.get(
                "affinity_mask",
                torch.ones(batch["affinity"].shape[0], dtype=torch.bool, device=self.device),
            )
            has_ba = bool(affinity_mask.any().item())

        # Immuno: same optional/masked shape as BA (some rows in a mixed EL+immuno
        # batch carry a real immunogenicity label, some don't — see model.py's
        # use_immuno_head docstring / experiments/immunogenicity_training/REPORT.md §7).
        has_immuno = False
        immuno_mask: Optional[torch.Tensor] = None
        if self.use_immuno_head and "immuno_label" in batch:
            immuno_mask = batch.get(
                "immuno_mask",
                torch.ones(batch["immuno_label"].shape[0], dtype=torch.bool, device=self.device),
            )
            has_immuno = bool(immuno_mask.any().item())

        if not has_el and not has_ba and not has_immuno:
            raise ValueError(
                "training_step received a batch with neither 'label' (EL), "
                "'affinity' (BA), nor 'immuno_label' (immuno). Check your data pipeline."
            )

        # Always compute the EL head so its params are in the loss graph every
        # step (lets DDP run with find_unused_parameters=False + static_graph=True).
        el_logits = self.classifier(feats)
        # EL loss (present when batch contains binary EL labels)
        if has_el:
            el_loss = self.compute_loss(el_logits, batch["label"])
            total_loss = total_loss + self.loss_weight_el * el_loss
            # Optional per-allele pairwise RankNet term for EL (binary), a
            # within-allele AP/AUC surrogate. OFF by default (weight=0). Needs a
            # grouped sampler (ELGroupedBatchSampler) so batches carry many
            # within-allele pos-neg pairs; under the prod sampler it is starved.
            if (
                self.el_pairwise_weight > 0
                and self.global_step >= self.el_pairwise_warmup
                and "groups" in batch
            ):
                el_rank_loss = self._pairwise_rank_loss(
                    el_logits.view(-1),
                    batch["label"].to(el_logits.dtype),
                    batch["groups"],
                    tau=self.el_pairwise_tau,
                    gap=self.el_pairwise_gap,
                )
                total_loss = total_loss + self.el_pairwise_weight * el_rank_loss
                self.log("train/el_rank_loss", el_rank_loss, prog_bar=True)
        else:
            # keep the EL head connected to the graph even with no EL labels
            el_loss = el_logits.sum() * 0.0
            total_loss = total_loss + self.loss_weight_el * el_loss

        # Always forward the BA head (when enabled) so its params are in the loss
        # graph every step — required for DDP find_unused_parameters=False.
        ba_logits_all = self.ba_head(feats) if self.use_ba_head else None
        # BA loss: only over samples with a valid affinity label (affinity_mask == True)
        if has_ba:
            ba_logits = ba_logits_all[affinity_mask]
            ba_label = self._transform_ba_label(batch["affinity"][affinity_mask])
            ba_ineq = batch["affinity_ineq"][affinity_mask] if "affinity_ineq" in batch else None
            ba_loss = self.compute_ba_loss(ba_logits, ba_label, ineq=ba_ineq)
            total_loss = total_loss + self.loss_weight_ba * ba_loss
            # Optional per-allele pairwise RankNet term (BA specialist).
            # OFF by default (ba_pairwise_weight=0); warmup keeps it MSE-only early.
            if self.ba_pairwise_weight > 0 and self.global_step >= self.ba_pairwise_warmup:
                ba_groups = batch["groups"][affinity_mask]
                ba_rank_loss = self._pairwise_rank_loss(
                    ba_logits.view(-1), ba_label, ba_groups
                )
                total_loss = total_loss + self.ba_pairwise_weight * ba_rank_loss
                self.log("train/ba_rank_loss", ba_rank_loss, prog_bar=True)
        elif self.use_ba_head:
            # no affinity labels this batch: keep the BA head connected (0 grad)
            ba_loss = ba_logits_all.sum() * 0.0
            total_loss = total_loss + self.loss_weight_ba * ba_loss
        else:
            ba_loss = torch.tensor(0.0, device=self.device, dtype=feats.dtype)

        # Always forward the immuno head (when enabled) so its params are in the
        # loss graph every step — required for DDP find_unused_parameters=False.
        immuno_logits_all = self.immuno_head(feats) if self.use_immuno_head else None
        # Immuno loss: only over samples with a valid immuno_label (immuno_mask == True)
        if has_immuno:
            immuno_logits = immuno_logits_all[immuno_mask]
            immuno_label = batch["immuno_label"][immuno_mask].float()
            immuno_loss = F.binary_cross_entropy_with_logits(immuno_logits.view(-1), immuno_label)
            total_loss = total_loss + self.loss_weight_immuno * immuno_loss
        elif self.use_immuno_head:
            # no immuno labels this batch: keep the immuno head connected (0 grad)
            immuno_loss = immuno_logits_all.sum() * 0.0
            total_loss = total_loss + self.loss_weight_immuno * immuno_loss
        else:
            immuno_loss = torch.tensor(0.0, device=self.device, dtype=feats.dtype)

        self.log("train/el_loss", el_loss, prog_bar=True)
        self.log("train/ba_loss", ba_loss, prog_bar=True)
        self.log("train/immuno_loss", immuno_loss, prog_bar=True)
        self.log("train/loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        feats = self.extract_features(batch["sequence_tokens"])

        # EL metrics (standard path — batch always has "label" during EL validation)
        if "label" in batch:
            el_logits = self.classifier(feats)
            loss = self.compute_loss(el_logits, batch["label"])
            self.log("val/loss", loss, prog_bar=True)
            preds = torch.sigmoid(el_logits[:, 0])
            self.el_avgp(preds, batch["label"].long(), batch["groups"])

        # BA metrics (optional — only when BA head is enabled and batch has affinity labels)
        if self.use_ba_head and "affinity" in batch:
            ba_affinity_mask = batch.get(
                "affinity_mask",
                torch.ones(batch["affinity"].shape[0], dtype=torch.bool, device=self.device),
            )
            if ba_affinity_mask.any():
                ba_logits = self.ba_head(feats[ba_affinity_mask])
                ba_label = self._transform_ba_label(batch["affinity"][ba_affinity_mask])
                ba_ineq = batch["affinity_ineq"][ba_affinity_mask] if "affinity_ineq" in batch else None
                ba_loss = self.compute_ba_loss(ba_logits, ba_label, ineq=ba_ineq)
                self.log("val/ba_loss", ba_loss)
                # accumulate EXACT-only per-allele BA preds for val/ba_pearson selection
                # (censored rows' label is a threshold, not a true affinity -> exclude)
                if not hasattr(self, "_ba_val_buf"):
                    self._ba_val_buf = {"pred": [], "label": [], "grp": []}
                if ba_ineq is None:
                    exact = torch.ones(ba_label.shape[0], dtype=torch.bool, device=ba_label.device)
                else:
                    exact = (ba_ineq == 0)
                if exact.any():
                    self._ba_val_buf["pred"].append(ba_logits[exact, 0].detach().float().cpu())
                    self._ba_val_buf["label"].append(ba_label[exact].detach().float().cpu())
                    self._ba_val_buf["grp"].append(batch["groups"][ba_affinity_mask][exact].detach().cpu())

        # Immuno metrics (optional — only when immuno head enabled and batch has immuno labels)
        if self.use_immuno_head and "immuno_label" in batch:
            immuno_mask = batch.get(
                "immuno_mask",
                torch.ones(batch["immuno_label"].shape[0], dtype=torch.bool, device=self.device),
            )
            if immuno_mask.any():
                immuno_logits = self.immuno_head(feats[immuno_mask])
                immuno_label = batch["immuno_label"][immuno_mask].float()
                immuno_loss = F.binary_cross_entropy_with_logits(immuno_logits.view(-1), immuno_label)
                self.log("val/immuno_loss", immuno_loss)
                immuno_preds = torch.sigmoid(immuno_logits[:, 0])
                self.immuno_avgp(immuno_preds, immuno_label.long(), batch["groups"][immuno_mask])

    def on_validation_epoch_end(self) -> None:
        try:
            el_avgp = self.el_avgp.compute()
            avg_el_avgp = float(np.mean(list(el_avgp.values())))
        except Exception as e:
            # surface the failure instead of silently logging -1.0 (this hid a
            # DDP CPU-tensor all_gather bug in GroupAveragePrecision).
            if getattr(self, "global_rank", 0) == 0:
                print(f"[WARN] el_avgp.compute() failed: {type(e).__name__}: {e}", flush=True)
            avg_el_avgp = -1.0
        self.log("val/el_avgp", avg_el_avgp, prog_bar=True)
        self.el_avgp.reset()

        if self.use_immuno_head:
            try:
                immuno_avgp = self.immuno_avgp.compute()
                valid = [v for v in immuno_avgp.values() if v == v]  # drop NaN (empty groups)
                avg_immuno_avgp = float(np.mean(valid)) if valid else -1.0
            except Exception as e:
                if getattr(self, "global_rank", 0) == 0:
                    print(f"[WARN] immuno_avgp.compute() failed: {type(e).__name__}: {e}", flush=True)
                avg_immuno_avgp = -1.0
            self.log("val/immuno_avgp", avg_immuno_avgp, prog_bar=True)
            self.immuno_avgp.reset()

        # median per-allele BA Pearson — the BA-specialist checkpoint-selection metric
        ba_pearson = -1.0
        buf = getattr(self, "_ba_val_buf", None)
        if buf and buf["pred"]:
            p = torch.cat(buf["pred"]).numpy()
            y = torch.cat(buf["label"]).numpy()
            g = torch.cat(buf["grp"]).numpy()
            corrs = []
            for a in np.unique(g):
                m = g == a
                if m.sum() >= 5 and np.std(p[m]) > 1e-8 and np.std(y[m]) > 1e-8:
                    corrs.append(float(np.corrcoef(p[m], y[m])[0, 1]))
            if corrs:
                ba_pearson = float(np.median(corrs))
        self.log("val/ba_pearson", ba_pearson, prog_bar=True)
        self._ba_val_buf = {"pred": [], "label": [], "grp": []}

    # ──────────────────────────────────────────────
    # Optimizer with LLRD (layer-wise LR decay)
    # ──────────────────────────────────────────────
    def configure_optimizers(self):
        """
        Build AdamW optimizer with layer-wise learning-rate decay (LLRD).

        Parameter groups (highest → lowest lr):
          1. Classification head  → ``lr``
          2. Backbone transformer layers (top → bottom) → ``lr_backbone * llrd^k``
             where k=0 for the topmost layer, increasing downward.
          3. Backbone embedding layer → ``lr_backbone * llrd^n_layers``

        If the transformer block list cannot be found, falls back to two groups:
        backbone (``lr_backbone``) and head (``lr``).
        """
        param_groups: list[dict] = []

        # ── Head parameters ──
        head_params = [
            p for k, p in self.named_parameters()
            if p.requires_grad and "backbone" not in k
        ]
        if head_params:
            param_groups.append({"params": head_params, "lr": self.lr})

        # ── Backbone LLRD ──
        # Try to locate the list of transformer blocks for per-layer decay.
        transformer_blocks = None
        if (
            hasattr(self.backbone, "transformer")
            and hasattr(self.backbone.transformer, "blocks")
        ):
            transformer_blocks = list(self.backbone.transformer.blocks)

        if transformer_blocks is not None:
            assigned_ids: set[int] = set()

            # Iterate layers from top (last) to bottom (first); k=0 → lr_backbone
            for k, layer in enumerate(reversed(transformer_blocks)):
                layer_lr = self.lr_backbone * (self.llrd ** k)
                layer_params = [p for p in layer.parameters() if p.requires_grad]
                if layer_params:
                    param_groups.append({"params": layer_params, "lr": layer_lr})
                for p in layer.parameters():
                    assigned_ids.add(id(p))

            # Remaining backbone params (embed layer, final norm, etc.) get smallest lr
            embed_lr = self.lr_backbone * (self.llrd ** len(transformer_blocks))
            remaining = [
                p for p in self.backbone.parameters()
                if p.requires_grad and id(p) not in assigned_ids
            ]
            if remaining:
                param_groups.append({"params": remaining, "lr": embed_lr})
        else:
            # Fallback: single backbone group (no LLRD)
            backbone_params = [
                p for k, p in self.named_parameters()
                if p.requires_grad and "backbone" in k
            ]
            if backbone_params:
                param_groups.append({"params": backbone_params, "lr": self.lr_backbone})

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # Optional linear LR warmup (scales every LLRD group's base lr by
        # min(1, step/warmup_steps)). Config-gated; default 0 = original behavior.
        if getattr(self, "warmup_steps", 0) and self.warmup_steps > 0:
            ws = int(self.warmup_steps)
            sched = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / ws)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sched, "interval": "step"},
            }
        return optimizer

    # ──────────────────────────────────────────────
    # Convenience class methods
    # ──────────────────────────────────────────────
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Strip ``_orig_mod`` prefix that ``torch.compile`` injects.

        When a model is saved after ``torch.compile``, every sub-module key
        gets an extra ``._orig_mod`` segment (e.g.
        ``backbone._orig_mod.transformer.…``).  If the current model is *not*
        compiled, those keys will not match.  We normalise them here so that
        ``load_from_checkpoint`` / ``Trainer.fit(ckpt_path=…)`` work
        regardless of whether the saving or loading side used ``torch.compile``.
        """
        # If the model is (or WILL be) torch.compiled, its keys carry
        # ``._orig_mod`` and the compiled checkpoint already matches — stripping
        # would BREAK the load (e.g. PL Trainer.fit(ckpt_path=…) on a compiled
        # model). Note on_load_checkpoint runs BEFORE configure_model() applies
        # compile, so also honour the pending ``_use_compile`` flag. Only strip
        # when the model is genuinely uncompiled AND will stay uncompiled.
        if any("._orig_mod" in k for k in self.state_dict().keys()) \
                or getattr(self, "_use_compile", False):
            return
        state_dict = checkpoint.get("state_dict", {})
        cleaned = {
            k.replace("._orig_mod", ""): v for k, v in state_dict.items()
        }
        if len(cleaned) != len(state_dict) or set(cleaned) != set(state_dict):
            # Only overwrite when there was actually something to strip
            checkpoint["state_dict"] = cleaned

    @classmethod
    def selection_metric(cls) -> tuple[str, str]:
        return "val/el_avgp", "max"


# ──────────────────────────────────────────────────────────────────
# Utility: fix torch.compile() _orig_mod. prefix in checkpoints
# ──────────────────────────────────────────────────────────────────
def _fix_compiled_checkpoint_keys(ckpt_path: str) -> None:
    """Strip ``_orig_mod.`` prefix from state-dict keys (added by torch.compile)."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", {})
    if not any("._orig_mod." in k for k in state_dict):
        return
    checkpoint["state_dict"] = {
        k.replace("._orig_mod.", "."): v for k, v in state_dict.items()
    }
    torch.save(checkpoint, ckpt_path)


class EnsemblePmhcuniModel(nn.Module):
    """
    Ensemble wrapper: loads multiple PmhcuniModel checkpoints and
    averages their predictions.

    Args:
        checkpoints: List of paths to ``PmhcuniModel`` checkpoint files.
    """

    def __init__(self, checkpoints: list[str]):
        super().__init__()
        models = []
        for ckpt in checkpoints:
            _fix_compiled_checkpoint_keys(ckpt)
            model = PmhcuniModel.load_from_checkpoint(ckpt, map_location="cpu")
            model.eval()
            models.append(model)
        self.models = nn.ModuleList(models)
        self.d_model = models[0].d_model

    @torch.no_grad()
    def forward(
        self,
        sequence_tokens: torch.Tensor,
        return_feats: bool = False,
        task: str = "el",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Average logits across all ensemble members.

        Args:
            sequence_tokens: (B, L) tokenized peptide sequences.
            return_feats: If True, also return stacked per-model features.
            task: ``"el"`` (default) for EL classification logits,
                ``"ba"`` for BA regression predictions.

        Returns:
            stacked_logits: (n_models, B, 1) raw per-model logits.
            stacked_feats (optional): (n_models, B, feat_dim) per-model features.
        """
        all_logits = []
        all_feats = []

        for model in self.models:
            logits, feats = model(sequence_tokens, return_feats=True, task=task)
            all_logits.append(logits)
            all_feats.append(feats)

        # (n_models, B, 1)
        stacked_logits = torch.stack(all_logits, dim=0)

        if return_feats:
            stacked_feats = torch.stack(all_feats, dim=0)
            return stacked_logits, stacked_feats
        return stacked_logits
