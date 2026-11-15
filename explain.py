"""Explain peptide-MHC predictions from a checkpointed pMHC-uni model.

Examples:
    python explain.py \
        --peptide SIINFEKL,GILGFVFTL \
        --mhc HLA-A*02:01 \
        --allele_sequences /path/to/seqs.csv \
        --checkpoint ./epoch=7-step=429472.ckpt \
        --outdir ./results/explain

    python explain.py \
        --csv /path/to/input.csv \
        --allele_sequences /path/to/seqs.csv \
        --checkpoint ./epoch=7-step=429472.ckpt \
        --outdir ./results/explain
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import pandas as pd
from tqdm import tqdm

from pmhcuni.explain import explain_pair, save_explanations
from pmhcuni.model import PmhcuniModel, _fix_compiled_checkpoint_keys
from pmhcuni.utils import load_allele_sequences
from pmhcuni.visualize import generate_visual_reports


def load_inputs(args, allele_sequences: dict[str, str]) -> tuple[list[str], list[str]]:
    peptides: list[str] = []
    mhc_names: list[str] = []

    if args.csv:
        df = pd.read_csv(args.csv)
        for _, row in df.iterrows():
            peptide = str(row[args.sequence_col]).strip()
            mhc = str(row[args.mhc_col]).strip()
            if args.min_len <= len(peptide) <= args.max_len and mhc in allele_sequences:
                peptides.append(peptide)
                mhc_names.append(mhc)
    elif args.peptide and args.mhc:
        pep_list = [item.strip() for item in args.peptide.split(",") if item.strip()]
        mhc_list = [item.strip() for item in args.mhc.split(",") if item.strip()]
        if len(mhc_list) == 1:
            mhc_list = mhc_list * len(pep_list)
        if len(pep_list) != len(mhc_list):
            raise ValueError(
                f"Number of peptides ({len(pep_list)}) and MHC alleles ({len(mhc_list)}) must match, "
                "or provide a single MHC to pair with all peptides."
            )
        for peptide, mhc in zip(pep_list, mhc_list):
            if args.min_len <= len(peptide) <= args.max_len and mhc in allele_sequences:
                peptides.append(peptide)
                mhc_names.append(mhc)
    else:
        raise ValueError("Please provide either --csv or both --peptide and --mhc.")

    return peptides, mhc_names


def main(args: argparse.Namespace) -> None:
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    allele_sequences = load_allele_sequences(args.allele_sequences)
    peptides, mhc_names = load_inputs(args, allele_sequences)
    if not peptides:
        raise ValueError("No valid peptide-MHC pairs remained after filtering.")

    _fix_compiled_checkpoint_keys(args.checkpoint)
    model = PmhcuniModel.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.to(device)
    model.eval()

    explanations = []
    for sample_id, (peptide, mhc) in enumerate(
        tqdm(list(zip(peptides, mhc_names)), desc="Explaining"), start=1
    ):
        explanation = explain_pair(
            model=model,
            peptide=peptide,
            mhc_name=mhc,
            allele_sequences=allele_sequences,
            device=device,
            max_len=args.max_len,
            top_k=args.top_k,
            task=args.task,
        )
        explanation.sample_id = sample_id
        explanations.append(explanation)

    outputs = save_explanations(explanations, args.outdir, top_k=args.top_k)
    visual_outputs = generate_visual_reports(args.outdir)

    print(f"Explained {len(explanations)} peptide-MHC pairs.")
    print(f"JSON summary: {outputs['json']}")
    print(f"Token importance: {outputs['token_importance']}")
    print(f"Sample top interactions: {outputs['sample_top_interactions']}")
    print(f"Allele interaction summary: {outputs['allele_interactions']}")
    print(f"Allele token summary: {outputs['allele_token_importance']}")
    print(f"Per-sample matrices: {outputs['matrix_dir']}")
    print(f"Visualization dashboard: {visual_outputs['index']}")
    print(f"Allele comparison page: {visual_outputs['allele_comparison']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain pMHC-uni predictions")
    parser.add_argument("--csv", type=str, default=None, help="CSV with peptide/MHC columns")
    parser.add_argument("--sequence_col", type=str, default="sequence", help="Peptide column name in --csv")
    parser.add_argument("--mhc_col", type=str, default="mhc", help="MHC column name in --csv")
    parser.add_argument("--peptide", type=str, default=None, help="Comma-separated peptides")
    parser.add_argument("--mhc", type=str, default=None, help="Comma-separated MHC alleles, or one allele for all peptides")
    parser.add_argument("--allele_sequences", type=str, required=True, help="CSV mapping allele -> pseudo-sequence")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--outdir", type=str, default=str(Path("results") / "explanations"))
    parser.add_argument("--min_len", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=15)
    parser.add_argument("--top_k", type=int, default=20, help="Top residue interactions per sample/allele")
    parser.add_argument("--task", type=str, default="el", choices=["el", "ba"])
    parser.add_argument("--device", type=str, default="auto", help="cuda / cpu / mps / auto")
    main(parser.parse_args())
