"""
Prediction script for PmhcuniModel.

Usage:
    # Mode 1: from CSV
    python predict.py --csv input.csv --allele_sequences alleles.csv --outdir results/

    # Mode 2: from command-line arguments
    python predict.py --peptide "AAAAAAAAA,CCCCCCCCC" --mhc "HLA-A0201" --allele_sequences alleles.csv

Input CSV must contain columns: sequence, mhc
"""


from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
import argparse
from typing import Optional
from collections import OrderedDict

import torch
import pandas as pd
from tqdm import tqdm

from esm.tokenization import EsmSequenceTokenizer

from pmhcuni.model import PmhcuniModel, EnsemblePmhcuniModel
from pmhcuni.utils import load_allele_sequences, normalize_allele_name
from pmhcuni.data import _detect_two_chain, _compute_max_mhc_tokens, _resolve_mhc_tokens

# Pseudo-sequence truncation length (must match data.py)
MIN_LEN = 346


def tokenize_samples(
    peptides: list[str],
    mhc_names: list[str],
    allele_sequences: OrderedDict[str, str],
    tokenizer: EsmSequenceTokenizer,
    max_len: int = 15,
) -> torch.Tensor:
    """
    Tokenize peptide+MHC pairs in the same way as data.py:
        [CLS] + mhc_tokens + [CHAIN_BREAK] + pep_tokens + [EOS] + [PAD]*pad_len

    Supports both single-chain (class I) and two-chain (class II) MHC.
    """
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    chain_break_id = tokenizer.chain_break_token_id

    two_chain = _detect_two_chain(allele_sequences, mhc_names[:100])
    max_mhc_tokens = _compute_max_mhc_tokens(two_chain)

    # Pre-cache MHC tokens
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
def predict(
    peptides: list[str],
    mhc_names: list[str],
    allele_sequences: OrderedDict[str, str],
    model: PmhcuniModel | EnsemblePmhcuniModel,
    device: torch.device,
    batch_size: int = 32,
    max_len: int = 15,
    task: str = "el",
) -> list[float]:
    """Run prediction on a list of peptide+MHC pairs.

    task: which head to read ("el" default, or "immuno" for a
        use_immuno_head=True checkpoint's immunogenicity head). Ignored for
        EnsemblePmhcuniModel (ensembles are EL-only).
    """
    tokenizer = EsmSequenceTokenizer()
    model.eval()
    model.to(device)

    all_preds: list[float] = []

    for i in tqdm(range(0, len(peptides), batch_size), desc="Predicting"):
        batch_peps = peptides[i : i + batch_size]
        batch_mhcs = mhc_names[i : i + batch_size]
        tokens = tokenize_samples(
            batch_peps, batch_mhcs, allele_sequences, tokenizer, max_len
        ).to(device)

        if isinstance(model, EnsemblePmhcuniModel):
            # stacked_logits: (n_models, B, 1)
            stacked_logits = model(tokens)
            # Average across models
            logits = stacked_logits.mean(dim=0)
        else:
            logits = model(tokens, task=task)

        probs = torch.sigmoid(logits.squeeze(-1)).cpu().tolist()
        all_preds.extend(probs)

    return all_preds


def main(args):
    device = torch.device(args.device)

    # Load allele sequences
    allele_sequences = load_allele_sequences(args.allele_sequences)
    two_chain = _detect_two_chain(allele_sequences)

    peptides: list[str] = []
    mhc_names: list[str] = []

    if args.csv:
        # Mode 1: read from CSV (must contain 'sequence' and 'mhc' columns)
        df = pd.read_csv(args.csv)
        for _, row in df.iterrows():
            pep = str(row["sequence"])
            mhc = str(row["mhc"]).strip() if two_chain else normalize_allele_name(str(row["mhc"]))
            if args.min_len <= len(pep) <= args.max_len:
                if two_chain or mhc in allele_sequences:
                    peptides.append(pep)
                    mhc_names.append(mhc)
    elif args.peptide and args.mhc:
        # Mode 2: command-line --peptide and --mhc arguments
        pep_list = [p.strip() for p in args.peptide.split(",")]
        if two_chain:
            mhc_list = [m.strip() for m in args.mhc.split(";")]  # Use ";" for class II (names contain ",")
        else:
            mhc_list = [normalize_allele_name(m.strip()) for m in args.mhc.split(",")]

        if len(mhc_list) == 1:
            mhc_list = mhc_list * len(pep_list)

        if len(pep_list) != len(mhc_list):
            raise ValueError(
                f"Number of peptides ({len(pep_list)}) and MHC alleles ({len(mhc_list)}) "
                "must match, or provide a single MHC to pair with all peptides."
            )

        for pep, mhc in zip(pep_list, mhc_list):
            if args.min_len <= len(pep) <= args.max_len:
                if two_chain or mhc in allele_sequences:
                    peptides.append(pep)
                    mhc_names.append(mhc)
    else:
        raise ValueError("Please provide either --csv or both --peptide and --mhc.")

    print(f"Loaded {len(peptides)} peptide-MHC pairs for prediction.")

    # Load model
    if args.checkpoint:
        model = PmhcuniModel.load_from_checkpoint(args.checkpoint)
    else:
        # Default: 5-model ensemble
        checkpoints = [f"models/no-flanks/model{i}.ckpt" for i in range(1, 6)]
        model = EnsemblePmhcuniModel(checkpoints)

    # Predict
    preds = predict(peptides, mhc_names, allele_sequences, model, device, args.batch_size, args.max_len)

    # Save
    os.makedirs(args.outdir, exist_ok=True)
    df_out = pd.DataFrame({"peptide": peptides, "mhc": mhc_names, "score": preds})
    out_path = os.path.join(args.outdir, "predictions.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pMHC-uni Prediction")
    parser.add_argument("--csv", type=str, default=None, help="CSV with 'sequence' and 'mhc' columns.")
    parser.add_argument("--peptide", type=str, default=None, help="Comma-separated peptide sequences, e.g. 'AAAAAAAAA,CCCCCCCCC'.")
    parser.add_argument("--mhc", type=str, default=None, help="Comma-separated MHC alleles, or a single allele for all peptides.")
    parser.add_argument("--allele_sequences", type=str, required=True, help="CSV mapping allele names to sequences.")
    parser.add_argument("--min_len", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="./results")
    args = parser.parse_args()
    main(args)
