from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
import einops

from esm.tokenization import EsmSequenceTokenizer

from pmhcuni.data import MIN_LEN
from pmhcuni.model import PmhcuniModel


@dataclass
class SequenceLayout:
    mhc_start: int
    mhc_end: int
    peptide_start: int
    peptide_end: int
    eos_index: int

    @property
    def mhc_length(self) -> int:
        return self.mhc_end - self.mhc_start

    @property
    def peptide_length(self) -> int:
        return self.peptide_end - self.peptide_start


@dataclass
class SampleExplanation:
    sample_id: int
    mhc: str
    peptide: str
    logit: float
    probability: float
    mhc_importance: list[dict[str, Any]]
    peptide_importance: list[dict[str, Any]]
    top_interactions: list[dict[str, Any]]
    interaction_matrix: list[list[float]]
    attention_pool_weights: dict[str, list[float] | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "mhc": self.mhc,
            "peptide": self.peptide,
            "logit": self.logit,
            "probability": self.probability,
            "mhc_importance": self.mhc_importance,
            "peptide_importance": self.peptide_importance,
            "top_interactions": self.top_interactions,
            "interaction_matrix": self.interaction_matrix,
            "attention_pool_weights": self.attention_pool_weights,
        }


def tokenize_pair(
    peptide: str,
    mhc_name: str,
    allele_sequences: dict[str, str],
    tokenizer: EsmSequenceTokenizer,
    max_len: int = 15,
) -> tuple[torch.Tensor, str]:
    """Tokenize a peptide-MHC pair exactly like the training/inference pipeline."""
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    chain_break_id = tokenizer.chain_break_token_id

    mhc_seq = allele_sequences[mhc_name][:MIN_LEN]
    pep_tokens = tokenizer.encode(peptide, add_special_tokens=False)
    mhc_tokens = tokenizer.encode(mhc_seq, add_special_tokens=False)
    # Pad or truncate to fixed length
    if len(mhc_tokens) > MIN_LEN:
        mhc_tokens = mhc_tokens[:MIN_LEN]
    elif len(mhc_tokens) < MIN_LEN:
        mhc_tokens = mhc_tokens + [pad_id] * (MIN_LEN - len(mhc_tokens))
    pad_len = max_len - len(peptide)
    if pad_len < 0:
        raise ValueError(
            f"peptide length {len(peptide)} exceeds max_len={max_len}: {peptide}"
        )

    full_seq = (
        [cls_id]
        + mhc_tokens
        + [chain_break_id]
        + pep_tokens
        + [eos_id]
        + [pad_id] * pad_len
    )
    return torch.tensor(full_seq, dtype=torch.long), mhc_seq


def extract_sequence_layout(
    sequence_tokens: torch.Tensor,
    tokenizer: EsmSequenceTokenizer,
) -> SequenceLayout:
    """Locate the MHC and peptide token spans inside one tokenized sample."""
    if sequence_tokens.ndim != 1:
        raise ValueError("extract_sequence_layout expects a 1D token tensor")

    chain_break_id = tokenizer.chain_break_token_id
    eos_id = tokenizer.eos_token_id

    chain_break_positions = (sequence_tokens == chain_break_id).nonzero(as_tuple=False)
    eos_positions = (sequence_tokens == eos_id).nonzero(as_tuple=False)
    if chain_break_positions.numel() == 0 or eos_positions.numel() == 0:
        raise ValueError("sequence does not contain the expected chain-break/EOS tokens")

    chain_break_index = int(chain_break_positions[0].item())
    eos_index = int(eos_positions[0].item())
    return SequenceLayout(
        mhc_start=1,
        mhc_end=chain_break_index,
        peptide_start=chain_break_index + 1,
        peptide_end=eos_index,
        eos_index=eos_index,
    )


def _plain_attention_forward(
    attn_module: torch.nn.Module,
    x: torch.Tensor,
    sequence_id: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay one ESMC plain-attention block and return attention weights."""
    qkv_bld3 = attn_module.layernorm_qkv(x)
    query_bld, key_bld, value_bld = torch.chunk(qkv_bld3, 3, dim=-1)
    query_bld, key_bld = (
        attn_module.q_ln(query_bld).to(query_bld.dtype),
        attn_module.k_ln(key_bld).to(query_bld.dtype),
    )
    query_bld, key_bld = attn_module._apply_rotary(query_bld, key_bld)

    reshaper = lambda t: einops.rearrange(t, "b s (h d) -> b h s d", h=attn_module.n_heads)
    query_bhld = reshaper(query_bld)
    key_bhld = reshaper(key_bld)
    value_bhld = reshaper(value_bld)

    scores = torch.matmul(query_bhld, key_bhld.transpose(-1, -2))
    scores = scores / math.sqrt(query_bhld.shape[-1])

    if sequence_id is not None:
        mask_bll = sequence_id.unsqueeze(-1) == sequence_id.unsqueeze(-2)
        scores = scores.masked_fill(~mask_bll.unsqueeze(1), float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    context_bhld = torch.matmul(attn_weights, value_bhld)
    context_bld = einops.rearrange(context_bhld, "b h s d -> b s (h d)")
    return attn_module.out_proj(context_bld), attn_weights


def forward_with_attention(
    model: PmhcuniModel,
    sequence_tokens: torch.Tensor,
    task: str = "el",
    require_grad: bool = False,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor | None],
    list[torch.Tensor],
    torch.Tensor,
]:
    """Run the checkpointed model while exposing layer attentions and pooling details."""
    tokenizer = model.tokenizer
    sequence_id = sequence_tokens != tokenizer.pad_token_id

    if require_grad:
        input_embeddings = model.backbone.embed(sequence_tokens).detach().requires_grad_(True)
    else:
        input_embeddings = model.backbone.embed(sequence_tokens)
    x = input_embeddings

    attention_layers: list[torch.Tensor] = []
    for block in model.backbone.transformer.blocks:
        if getattr(block, "use_plain_attn", False):
            attn_out, attn_weights = _plain_attention_forward(block.attn, x, sequence_id)
            x = x + attn_out / block.scaling_factor
            attention_layers.append(attn_weights)

        if getattr(block, "use_geom_attn", False):
            raise NotImplementedError(
                "Explainability currently supports plain-attention ESMC blocks only."
            )

        x = x + block.ffn(x) / block.scaling_factor

    x = model.backbone.transformer.norm(x)
    feats, details = model.extract_features_from_embeddings(
        sequence_tokens, x, return_details=True
    )

    if task == "ba" and model.use_ba_head:
        logits = model.ba_head(feats)
    else:
        logits = model.classifier(feats)
    return logits, details, attention_layers, input_embeddings


def compute_attention_rollout(
    attention_layers: Iterable[torch.Tensor],
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Aggregate per-layer attentions into an attention-rollout matrix."""
    layers = list(attention_layers)
    if not layers:
        raise ValueError("attention_layers is empty")

    batch_size, _, seq_len, _ = layers[0].shape
    eye = torch.eye(seq_len, device=layers[0].device, dtype=layers[0].dtype)
    eye = eye.unsqueeze(0).expand(batch_size, -1, -1)

    pair_mask = valid_mask.unsqueeze(-1) & valid_mask.unsqueeze(-2)
    rollout = eye.clone()
    for layer in layers:
        attn = layer.mean(dim=1)
        attn = attn * pair_mask.float()
        attn = attn + eye
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        rollout = torch.matmul(attn, rollout)
    return rollout


def compute_token_saliency(
    model: PmhcuniModel,
    sequence_tokens: torch.Tensor,
    task: str = "el",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None], list[torch.Tensor]]:
    """Gradient x input token saliency over one or more samples."""
    logits, details, attention_layers, input_embeddings = forward_with_attention(
        model, sequence_tokens, task=task, require_grad=True
    )
    score = logits[:, 0].sum()
    grads = torch.autograd.grad(score, input_embeddings, retain_graph=False)[0]
    saliency = (grads * input_embeddings).sum(dim=-1).abs()
    return saliency, logits.detach(), details, attention_layers


def _normalize_segment_scores(scores: torch.Tensor) -> torch.Tensor:
    total = scores.sum().clamp(min=1e-8)
    return scores / total


def _build_position_records(
    sequence: str,
    scores: torch.Tensor,
    prefix: str,
) -> list[dict[str, Any]]:
    rows = []
    for idx, (residue, value) in enumerate(zip(sequence, scores.tolist()), start=1):
        rows.append(
            {
                "segment": prefix,
                "position": idx,
                "residue": residue,
                "importance": float(value),
            }
        )
    return rows


def summarize_top_interactions(
    interaction_matrix: torch.Tensor,
    mhc_sequence: str,
    peptide: str,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Return the top scoring MHC-peptide residue interactions."""
    flat = interaction_matrix.reshape(-1)
    top_k = min(top_k, flat.numel())
    values, indices = torch.topk(flat, k=top_k)
    rows = []
    peptide_len = interaction_matrix.shape[1]
    for value, flat_index in zip(values.tolist(), indices.tolist()):
        mhc_idx = flat_index // peptide_len
        pep_idx = flat_index % peptide_len
        rows.append(
            {
                "mhc_position": mhc_idx + 1,
                "mhc_residue": mhc_sequence[mhc_idx],
                "peptide_position": pep_idx + 1,
                "peptide_residue": peptide[pep_idx],
                "interaction_score": float(value),
            }
        )
    return rows


def explain_pair(
    model: PmhcuniModel,
    peptide: str,
    mhc_name: str,
    allele_sequences: dict[str, str],
    device: torch.device,
    max_len: int = 15,
    top_k: int = 20,
    task: str = "el",
) -> SampleExplanation:
    """Explain one peptide-MHC prediction with saliency and rollout interactions."""
    tokenizer = model.tokenizer
    sequence_tokens, mhc_sequence = tokenize_pair(
        peptide, mhc_name, allele_sequences, tokenizer, max_len=max_len
    )
    sequence_tokens = sequence_tokens.unsqueeze(0).to(device)

    saliency, logits, details, attention_layers = compute_token_saliency(
        model, sequence_tokens, task=task
    )
    layout = extract_sequence_layout(sequence_tokens[0].detach().cpu(), tokenizer)

    token_mask = torch.zeros_like(sequence_tokens, dtype=torch.bool)
    token_mask[:, layout.mhc_start:layout.mhc_end] = True
    token_mask[:, layout.peptide_start:layout.peptide_end] = True

    rollout = compute_attention_rollout(attention_layers, token_mask)
    mhc_slice = slice(layout.mhc_start, layout.mhc_end)
    peptide_slice = slice(layout.peptide_start, layout.peptide_end)

    mhc_to_peptide = rollout[0, mhc_slice, peptide_slice]
    peptide_to_mhc = rollout[0, peptide_slice, mhc_slice].transpose(0, 1)
    interaction_matrix = 0.5 * (mhc_to_peptide + peptide_to_mhc)

    mhc_importance = _normalize_segment_scores(saliency[0, mhc_slice].detach().cpu())
    peptide_importance = _normalize_segment_scores(
        saliency[0, peptide_slice].detach().cpu()
    )

    pool_weights = details.get("attention_pool_weights")
    mhc_pool_weights: list[float] | None = None
    peptide_pool_weights: list[float] | None = None
    if pool_weights is not None:
        mhc_pool_weights = pool_weights[0, mhc_slice].detach().cpu().tolist()
        peptide_pool_weights = pool_weights[0, peptide_slice].detach().cpu().tolist()

    logit = float(logits[0, 0].item())
    probability = float(torch.sigmoid(logits[0, 0]).item())
    return SampleExplanation(
        sample_id=-1,
        mhc=mhc_name,
        peptide=peptide,
        logit=logit,
        probability=probability,
        mhc_importance=_build_position_records(mhc_sequence, mhc_importance, "mhc"),
        peptide_importance=_build_position_records(
            peptide, peptide_importance, "peptide"
        ),
        top_interactions=summarize_top_interactions(
            interaction_matrix.detach().cpu(), mhc_sequence, peptide, top_k=top_k
        ),
        interaction_matrix=interaction_matrix.detach().cpu().tolist(),
        attention_pool_weights={
            "mhc": mhc_pool_weights,
            "peptide": peptide_pool_weights,
        },
    )


def _safe_name(text: str, limit: int = 60) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:limit].strip("_") or "sample"


def save_explanations(
    explanations: list[SampleExplanation],
    outdir: str | Path,
    top_k: int,
) -> dict[str, Path]:
    """Persist per-sample and allele-level explanation outputs."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    matrix_dir = outdir / "interaction_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    json_path = outdir / "explanations.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([item.to_dict() for item in explanations], f, ensure_ascii=False, indent=2)

    token_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []

    for explanation in explanations:
        sample_prefix = f"{explanation.sample_id:04d}_{_safe_name(explanation.mhc)}_{_safe_name(explanation.peptide)}"
        matrix_path = matrix_dir / f"{sample_prefix}.csv"
        matrix_df = pd.DataFrame(explanation.interaction_matrix)
        matrix_df.index = [f"mhc_{idx}" for idx in range(1, len(matrix_df.index) + 1)]
        matrix_df.columns = [f"pep_{idx}" for idx in range(1, len(matrix_df.columns) + 1)]
        matrix_df.to_csv(matrix_path)

        for row in explanation.mhc_importance:
            token_rows.append(
                {
                    "sample_id": explanation.sample_id,
                    "mhc": explanation.mhc,
                    "peptide": explanation.peptide,
                    **row,
                    "attention_pool_weight": None
                    if explanation.attention_pool_weights["mhc"] is None
                    else explanation.attention_pool_weights["mhc"][row["position"] - 1],
                }
            )
        for row in explanation.peptide_importance:
            token_rows.append(
                {
                    "sample_id": explanation.sample_id,
                    "mhc": explanation.mhc,
                    "peptide": explanation.peptide,
                    **row,
                    "attention_pool_weight": None
                    if explanation.attention_pool_weights["peptide"] is None
                    else explanation.attention_pool_weights["peptide"][row["position"] - 1],
                }
            )
        for row in explanation.top_interactions:
            top_rows.append(
                {
                    "sample_id": explanation.sample_id,
                    "mhc": explanation.mhc,
                    "peptide": explanation.peptide,
                    **row,
                }
            )
        for mhc_idx, row in enumerate(explanation.interaction_matrix, start=1):
            for pep_idx, value in enumerate(row, start=1):
                matrix_rows.append(
                    {
                        "sample_id": explanation.sample_id,
                        "mhc": explanation.mhc,
                        "peptide": explanation.peptide,
                        "mhc_position": mhc_idx,
                        "peptide_position": pep_idx,
                        "interaction_score": float(value),
                    }
                )

    token_df = pd.DataFrame(token_rows)
    token_path = outdir / "token_importance.csv"
    token_df.to_csv(token_path, index=False)

    sample_top_df = pd.DataFrame(top_rows)
    sample_top_path = outdir / "sample_top_interactions.csv"
    sample_top_df.to_csv(sample_top_path, index=False)

    allele_matrix_df = (
        pd.DataFrame(matrix_rows)
        .groupby(["mhc", "mhc_position", "peptide_position"], as_index=False)
        .agg(mean_interaction=("interaction_score", "mean"), count=("interaction_score", "size"))
    )
    allele_matrix_path = outdir / "allele_interactions.csv"
    allele_matrix_df.to_csv(allele_matrix_path, index=False)

    allele_token_df = (
        token_df.groupby(["mhc", "segment", "position", "residue"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            mean_attention_pool_weight=("attention_pool_weight", "mean"),
            count=("importance", "size"),
        )
    )
    allele_token_path = outdir / "allele_token_importance.csv"
    allele_token_df.to_csv(allele_token_path, index=False)

    allele_top_df = (
        allele_matrix_df.sort_values(["mhc", "mean_interaction"], ascending=[True, False])
        .groupby("mhc", as_index=False)
        .head(top_k)
    )
    allele_top_path = outdir / "allele_top_interactions.csv"
    allele_top_df.to_csv(allele_top_path, index=False)

    return {
        "json": json_path,
        "token_importance": token_path,
        "sample_top_interactions": sample_top_path,
        "allele_interactions": allele_matrix_path,
        "allele_token_importance": allele_token_path,
        "allele_top_interactions": allele_top_path,
        "matrix_dir": matrix_dir,
    }
