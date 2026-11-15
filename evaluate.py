"""
Evaluate PmhcuniModel on the el_test.csv test set.

Steps:
    1. Load the test set (el_test.csv) and the training set (mhc_class1_normalized.csv).
    2. Normalize MHC allele names so they match between datasets.
    3. Remove test samples whose (peptide, mhc) pair already appeared in the training set.
    4. Filter to peptides with valid length and known alleles.
    5. Run model inference and compute metrics (AUC, AP, per-allele AP).

Usage:
    conda activate train
    python evaluate.py
"""

from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
from collections import OrderedDict

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    classification_report,
)
from tqdm import tqdm

from esm.tokenization import EsmSequenceTokenizer

from pmhcuni.model import PmhcuniModel
from pmhcuni.utils import load_allele_sequences
from pmhcuni.data import _detect_two_chain, _compute_max_mhc_tokens, _resolve_mhc_tokens

# Must match data.py / predict.py
MIN_LEN = 346


def normalize_mhc_name(name: str) -> str:
    """
    Normalize MHC allele names to a canonical format with '*'.
    e.g. 'HLA-A01:01' -> 'HLA-A*01:01', 'HLA-B07:02' -> 'HLA-B*07:02'
    Names that already contain '*' or don't match HLA pattern are left unchanged.
    """
    import re
    # Match HLA-X followed by digits (no asterisk)
    # e.g. HLA-A01:01  -> HLA-A*01:01
    #      HLA-B07:02  -> HLA-B*07:02
    #      HLA-C04:01  -> HLA-C*04:01
    m = re.match(r'^(HLA-[A-Z])(\d{2}:\d{2})$', name)
    if m:
        return f"{m.group(1)}*{m.group(2)}"
    return name


def tokenize_samples(
    peptides: list[str],
    mhc_names: list[str],
    allele_sequences: OrderedDict[str, str],
    tokenizer: EsmSequenceTokenizer,
    max_len: int = 15,
) -> torch.Tensor:
    """Tokenize peptide+MHC pairs (same as predict.py). Supports two-chain class II."""
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    chain_break_id = tokenizer.chain_break_token_id

    two_chain = _detect_two_chain(allele_sequences, mhc_names[:100])
    max_mhc_tokens = _compute_max_mhc_tokens(two_chain)
    mhc_token_cache: dict[str, list[int]] = {}

    all_tokens = []
    for pep, mhc_name in zip(peptides, mhc_names):
        if mhc_name not in mhc_token_cache:
            mhc_token_cache[mhc_name] = _resolve_mhc_tokens(
                mhc_name, allele_sequences, tokenizer, chain_break_id,
                max_mhc_tokens=max_mhc_tokens,
                pad_id=pad_id,
            )
        mhc_tokens = mhc_token_cache[mhc_name]
        pep_tokens = tokenizer.encode(pep, add_special_tokens=False)
        pad_len = max_len - len(pep)

        full_seq = (
            [cls_id]
            + mhc_tokens
            + [chain_break_id]
            + pep_tokens
            + [eos_id]
            + [pad_id] * pad_len
        )
        all_tokens.append(torch.tensor(full_seq, dtype=torch.long))

    return torch.stack(all_tokens, dim=0)


@torch.no_grad()
def run_predictions(
    peptides: list[str],
    mhc_names: list[str],
    allele_sequences: OrderedDict[str, str],
    model: PmhcuniModel,
    device: torch.device,
    batch_size: int = 32,
    max_len: int = 15,
) -> np.ndarray:
    """Run model inference and return predicted probabilities."""
    tokenizer = EsmSequenceTokenizer()
    model.eval()
    model.to(device)

    all_preds = []
    for i in tqdm(range(0, len(peptides), batch_size), desc="Predicting"):
        batch_peps = peptides[i : i + batch_size]
        batch_mhcs = mhc_names[i : i + batch_size]
        tokens = tokenize_samples(
            batch_peps, batch_mhcs, allele_sequences, tokenizer, max_len
        ).to(device)
        logits = model(tokens)
        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        all_preds.append(probs)

    return np.concatenate(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Evaluate PmhcuniModel on test set")
    parser.add_argument(
        "--test_csv", type=str,
        default="data/el_test.csv",
        help="Path to test CSV.",
    )
    parser.add_argument(
        "--train_csv", type=str,
        default="data/combined_sa_train.csv",
        help="Path to training CSV (for deduplication).",
    )
    parser.add_argument(
        "--allele_sequences", type=str,
        default="data/seqs_normalized.csv",
        help="CSV mapping allele names to pseudo-sequences.",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="output/checkpoints/epoch=0-step=48859.ckpt",
        help="Path to model checkpoint.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--min_len", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument(
        "--skip_dedup", action="store_true",
        help="Skip the deduplication step that removes test samples seen in training.",
    )
    args = parser.parse_args()

    # ──────────────────────────────────────────────
    # 1. Load allele sequences
    # ──────────────────────────────────────────────
    allele_sequences = load_allele_sequences(args.allele_sequences)
    print(f"Loaded {len(allele_sequences)} allele sequences.")

    # ──────────────────────────────────────────────
    # 2. Load training data and build dedup set
    # ──────────────────────────────────────────────
    train_pairs = set()
    if not args.skip_dedup:
        print("Loading training data for deduplication...")
        train_df = pd.read_csv(args.train_csv)
        # Training data columns: mhc, sequence, label, class
        # Build set of (peptide_sequence, normalized_mhc_name)
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building train set"):
            pep = str(row["sequence"])
            mhc = str(row["mhc"])
            train_pairs.add((pep, mhc))
        print(f"Training set has {len(train_pairs)} unique (peptide, mhc) pairs.")
    else:
        print("Skipping deduplication against training set (--skip_dedup).")

    # ──────────────────────────────────────────────
    # 3. Load test data
    # ──────────────────────────────────────────────
    print("Loading test data...")
    test_df = pd.read_csv(args.test_csv)
    # Test data columns: Unnamed: 0, mhc, pep, label, prot, left, right, new
    print(f"Total test samples: {len(test_df)}")

    # Normalize MHC names in test set (HLA-A01:01 -> HLA-A*01:01)
    test_df["mhc_normalized"] = test_df["mhc"].apply(normalize_mhc_name)

    # ──────────────────────────────────────────────
    # 4. Filter test data
    # ──────────────────────────────────────────────

    # 4a. Keep only peptides with valid lengths
    test_df["pep_len"] = test_df["pep"].str.len()
    len_mask = (test_df["pep_len"] >= args.min_len) & (test_df["pep_len"] <= args.max_len)
    test_df = test_df[len_mask].copy()
    print(f"After length filter ({args.min_len}-{args.max_len}): {len(test_df)} samples")

    # 4b. Keep only samples whose alleles are in allele_sequences
    two_chain = _detect_two_chain(allele_sequences)
    if two_chain:
        # For class II, composite names won't be in allele_sequences directly
        # Skip allele filtering (all alleles accepted)
        print(f"Two-chain (class II) mode: skipping allele filter")
    else:
        allele_mask = test_df["mhc_normalized"].isin(allele_sequences.keys())
        n_unknown_allele = (~allele_mask).sum()
        if n_unknown_allele > 0:
            unknown_alleles = test_df[~allele_mask]["mhc_normalized"].unique()
            print(f"WARNING: {n_unknown_allele} samples with unknown alleles removed: {unknown_alleles}")
        test_df = test_df[allele_mask].copy()
    print(f"After allele filter: {len(test_df)} samples")

    # 4c. Remove samples already seen in training
    if not args.skip_dedup:
        def is_in_train(row):
            return (row["pep"], row["mhc_normalized"]) in train_pairs

        seen_mask = test_df.apply(is_in_train, axis=1)
        n_leaked = seen_mask.sum()
        print(f"Removing {n_leaked} test samples that overlap with training data.")
        test_df = test_df[~seen_mask].copy()
        print(f"After deduplication: {len(test_df)} samples")
    else:
        n_leaked = 0

    # Check label distribution
    n_pos = (test_df["label"] == 1).sum()
    n_neg = (test_df["label"] == 0).sum()
    print(f"Label distribution: {n_pos} positive, {n_neg} negative (ratio: {n_pos/(n_pos+n_neg):.4f})")

    if len(test_df) == 0:
        print("ERROR: No test samples remaining after filtering. Exiting.")
        return

    # ──────────────────────────────────────────────
    # 5. Load model
    # ──────────────────────────────────────────────
    print(f"Loading model from {args.checkpoint}...")
    model = PmhcuniModel.load_from_checkpoint(
        args.checkpoint,
        allele_sequences=allele_sequences,
        map_location="cpu",
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ──────────────────────────────────────────────
    # 6. Run predictions
    # ──────────────────────────────────────────────
    peptides = test_df["pep"].tolist()
    mhc_names = test_df["mhc_normalized"].tolist()
    labels = test_df["label"].values.astype(int)

    preds = run_predictions(
        peptides, mhc_names, allele_sequences,
        model, device, args.batch_size, args.max_len,
    )

    # ──────────────────────────────────────────────
    # 7. Compute overall metrics
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)

    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    pred_labels = (preds >= 0.5).astype(int)
    acc = accuracy_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)

    print(f"  ROC-AUC:            {auc:.4f}")
    print(f"  Average Precision:  {ap:.4f}")
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  F1-Score:           {f1:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(labels, pred_labels, target_names=["Negative", "Positive"]))

    # ──────────────────────────────────────────────
    # 8. Compute per-allele metrics
    # ──────────────────────────────────────────────
    print("=" * 60)
    print("PER-ALLELE METRICS")
    print("=" * 60)

    allele_results = []
    unique_alleles = sorted(test_df["mhc_normalized"].unique())
    for allele in unique_alleles:
        mask = test_df["mhc_normalized"].values == allele
        allele_preds = preds[mask]
        allele_labels = labels[mask]

        n_total = len(allele_labels)
        n_pos_a = (allele_labels == 1).sum()
        n_neg_a = (allele_labels == 0).sum()

        result = {
            "allele": allele,
            "n_total": n_total,
            "n_positive": int(n_pos_a),
            "n_negative": int(n_neg_a),
        }

        if len(np.unique(allele_labels)) >= 2:
            result["auc"] = roc_auc_score(allele_labels, allele_preds)
            result["ap"] = average_precision_score(allele_labels, allele_preds)
        else:
            result["auc"] = float("nan")
            result["ap"] = float("nan")

        allele_results.append(result)
        print(
            f"  {allele:25s}  n={n_total:6d}  pos={n_pos_a:5d}  neg={n_neg_a:5d}  "
            f"AUC={result['auc']:.4f}  AP={result['ap']:.4f}"
        )

    # Mean per-allele AP (only alleles with both pos/neg)
    valid_aps = [r["ap"] for r in allele_results if not np.isnan(r["ap"])]
    valid_aucs = [r["auc"] for r in allele_results if not np.isnan(r["auc"])]
    if valid_aps:
        print(f"\n  Mean per-allele AP:   {np.mean(valid_aps):.4f}")
        print(f"  Mean per-allele AUC:  {np.mean(valid_aucs):.4f}")

    # ──────────────────────────────────────────────
    # 9. Save results
    # ──────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)

    # Save per-sample predictions
    result_df = pd.DataFrame({
        "peptide": peptides,
        "mhc": mhc_names,
        "label": labels,
        "pred_prob": preds,
        "pred_label": pred_labels,
    })
    result_path = os.path.join(args.outdir, "test_predictions.csv")
    result_df.to_csv(result_path, index=False)
    print(f"\nPer-sample predictions saved to: {result_path}")

    # Save per-allele metrics
    allele_df = pd.DataFrame(allele_results)
    allele_path = os.path.join(args.outdir, "test_per_allele_metrics.csv")
    allele_df.to_csv(allele_path, index=False)
    print(f"Per-allele metrics saved to:     {allele_path}")

    # Save summary metrics
    summary = {
        "total_test_samples": len(test_df),
        "n_leaked_removed": int(n_leaked),
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "roc_auc": float(auc),
        "average_precision": float(ap),
        "accuracy": float(acc),
        "f1_score": float(f1),
        "mean_per_allele_ap": float(np.mean(valid_aps)) if valid_aps else None,
        "mean_per_allele_auc": float(np.mean(valid_aucs)) if valid_aucs else None,
    }
    import json
    summary_path = os.path.join(args.outdir, "test_summary_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary metrics saved to:        {summary_path}")


if __name__ == "__main__":
    main()
