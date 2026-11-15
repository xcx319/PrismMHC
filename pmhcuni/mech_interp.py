"""
Mechanistic interpretability infrastructure for ESM-C-based pMHC-uni.

Provides:
  - PatchSpec: declarative description of where to inject clean activations
  - forward_with_cache_and_patch: manual forward pass with caching + patching
  - Pooled-block exposure: captures the 7 sep_concat blocks separately for Shapley
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence, Optional

import einops
import torch
import torch.nn as nn

from pmhcuni.model import PmhcuniModel


PatchTarget = Literal[
    "embed",
    "resid_pre", "attn_out", "resid_mid", "ffn_out", "resid_post",
    "q", "k", "v", "alpha", "z",
]


@dataclass(frozen=True)
class PatchSpec:
    """Describes a single activation patch.

    For module-level outputs (resid_pre, attn_out, resid_mid, ffn_out, resid_post,
    embed): tensor shape is (B, L, D). Use rows + token_idx (or full tensor patch).

    For attention internals (q, k, v, z): shape (B, H, L, D_h). Use rows + head_idx
    + token_idx (token_idx is dst for q/z, src for k/v).

    For attention weights (alpha): shape (B, H, L_dst, L_src). Use rows + head_idx
    + dst_idx + src_idx.
    """
    target: PatchTarget
    layer: Optional[int] = None
    rows: Optional[torch.Tensor] = None         # (P,) which batch rows to patch
    token_idx: Optional[torch.Tensor] = None    # (P,) per-row token index
    head_idx: Optional[torch.Tensor] = None     # (P,) per-row head index
    dst_idx: Optional[torch.Tensor] = None      # (P,) per-row dst position (alpha only)
    src_idx: Optional[torch.Tensor] = None      # (P,) per-row src position (alpha only)
    clean_key: Optional[str] = None             # explicit cache key override


def _key(target: str, layer: Optional[int]) -> str:
    return f"{target}__{layer}" if layer is not None else target


def _maybe_patch(
    x: torch.Tensor,
    target: PatchTarget,
    layer: Optional[int],
    clean_cache: Optional[dict],
    patch_specs: Sequence[PatchSpec],
) -> torch.Tensor:
    """Apply any matching PatchSpecs to x (in-place safe via clone).
    x may be 3D (B, L, D) or 4D (B, H, L, D_h) or alpha (B, H, L_dst, L_src).
    Returns possibly-patched tensor."""
    relevant = [s for s in patch_specs if s.target == target and s.layer == layer]
    if not relevant or clean_cache is None:
        return x

    out = x.clone()
    for spec in relevant:
        key = spec.clean_key or _key(target, layer)
        if key not in clean_cache:
            continue
        clean_t = clean_cache[key]

        if spec.rows is None and spec.token_idx is None and spec.head_idx is None:
            # Full-tensor patch (broadcast clean to all rows)
            if clean_t.shape[0] == 1 and out.shape[0] > 1:
                clean_t = clean_t.expand_as(out)
            out = clean_t.to(out.dtype).to(out.device)
            continue

        rows = spec.rows
        if rows is None:
            rows = torch.arange(out.shape[0], device=out.device)

        # 3D (B, L, D): rows + optional token_idx
        if out.dim() == 3 and target in ("embed", "resid_pre", "attn_out",
                                          "resid_mid", "ffn_out", "resid_post"):
            # Broadcast clean (B_clean=1, L, D) → (P, L, D)
            if clean_t.shape[0] == 1:
                clean_b = clean_t.expand(rows.shape[0], -1, -1)
            else:
                clean_b = clean_t

            if spec.token_idx is not None:
                # Patch only one (row, token) per spec entry
                tok = spec.token_idx
                # Select corresponding clean values
                clean_values = clean_b[torch.arange(rows.shape[0], device=out.device), tok]
                out[rows, tok] = clean_values.to(out.dtype)
            else:
                out[rows] = clean_b.to(out.dtype)

        # 4D (B, H, L, D_h): q/k/v/z
        elif out.dim() == 4 and target in ("q", "k", "v", "z"):
            if clean_t.shape[0] == 1:
                clean_b = clean_t.expand(rows.shape[0], -1, -1, -1)
            else:
                clean_b = clean_t

            head_idx = spec.head_idx
            tok = spec.token_idx

            if head_idx is not None and tok is not None:
                # Patch (row, head, token, :)
                idx = torch.arange(rows.shape[0], device=out.device)
                clean_values = clean_b[idx, head_idx, tok]  # (P, D_h)
                out[rows, head_idx, tok] = clean_values.to(out.dtype)
            elif head_idx is not None:
                # Patch (row, head, :, :)
                idx = torch.arange(rows.shape[0], device=out.device)
                clean_values = clean_b[idx, head_idx]
                out[rows, head_idx] = clean_values.to(out.dtype)
            else:
                out[rows] = clean_b.to(out.dtype)

        # 4D alpha (B, H, L_dst, L_src)
        elif out.dim() == 4 and target == "alpha":
            if clean_t.shape[0] == 1:
                clean_b = clean_t.expand(rows.shape[0], -1, -1, -1)
            else:
                clean_b = clean_t
            head_idx = spec.head_idx
            dst = spec.dst_idx
            src = spec.src_idx
            if head_idx is not None and dst is not None and src is not None:
                idx = torch.arange(rows.shape[0], device=out.device)
                clean_values = clean_b[idx, head_idx, dst, src]
                out[rows, head_idx, dst, src] = clean_values.to(out.dtype)

    return out


def attn_forward_with_cache(
    attn_module: nn.Module,
    x: torch.Tensor,
    sequence_id: Optional[torch.Tensor],
    *,
    layer: int,
    cache: Optional[dict] = None,
    clean_cache: Optional[dict] = None,
    patch_specs: Sequence[PatchSpec] = (),
    cache_targets: Optional[set] = None,
    ablate_heads: Optional[Sequence[int]] = None,
) -> tuple[torch.Tensor, dict]:
    """Manually run one ESM-C plain attention block, exposing q/k/v/alpha/z for caching/patching.

    If ablate_heads is provided, those head indices' z output is zeroed before being
    projected back through out_proj.
    """
    aux = {}

    qkv_bld3 = attn_module.layernorm_qkv(x)
    query_bld, key_bld, value_bld = torch.chunk(qkv_bld3, 3, dim=-1)
    query_bld, key_bld = (
        attn_module.q_ln(query_bld).to(query_bld.dtype),
        attn_module.k_ln(key_bld).to(query_bld.dtype),
    )
    query_bld, key_bld = attn_module._apply_rotary(query_bld, key_bld)

    H = attn_module.n_heads
    reshaper = lambda t: einops.rearrange(t, "b s (h d) -> b h s d", h=H)
    q_bhld = reshaper(query_bld)
    k_bhld = reshaper(key_bld)
    v_bhld = reshaper(value_bld)

    # Patch q/k/v
    q_bhld = _maybe_patch(q_bhld, "q", layer, clean_cache, patch_specs)
    k_bhld = _maybe_patch(k_bhld, "k", layer, clean_cache, patch_specs)
    v_bhld = _maybe_patch(v_bhld, "v", layer, clean_cache, patch_specs)

    if cache_targets and "q" in cache_targets:
        cache[_key("q", layer)] = q_bhld.detach().cpu()
    if cache_targets and "k" in cache_targets:
        cache[_key("k", layer)] = k_bhld.detach().cpu()
    if cache_targets and "v" in cache_targets:
        cache[_key("v", layer)] = v_bhld.detach().cpu()

    scores = torch.matmul(q_bhld, k_bhld.transpose(-1, -2))
    scores = scores / math.sqrt(q_bhld.shape[-1])

    if sequence_id is not None:
        mask_bll = sequence_id.unsqueeze(-1) == sequence_id.unsqueeze(-2)
        scores = scores.masked_fill(~mask_bll.unsqueeze(1), float("-inf"))

    alpha = torch.softmax(scores, dim=-1)
    alpha = torch.nan_to_num(alpha, nan=0.0)
    alpha = _maybe_patch(alpha, "alpha", layer, clean_cache, patch_specs)
    if cache_targets and "alpha" in cache_targets:
        cache[_key("alpha", layer)] = alpha.detach().cpu()

    z_bhld = torch.matmul(alpha, v_bhld)
    z_bhld = _maybe_patch(z_bhld, "z", layer, clean_cache, patch_specs)

    # Ablation: zero out specific heads' z output
    if ablate_heads:
        z_bhld = z_bhld.clone()
        for h in ablate_heads:
            z_bhld[:, h, :, :] = 0.0

    if cache_targets and "z" in cache_targets:
        cache[_key("z", layer)] = z_bhld.detach().cpu()

    z_bld = einops.rearrange(z_bhld, "b h s d -> b s (h d)")
    attn_out = attn_module.out_proj(z_bld)
    return attn_out, aux


def forward_with_cache_and_patch(
    model: PmhcuniModel,
    sequence_tokens: torch.Tensor,
    *,
    task: str = "el",
    cache_targets: Optional[set] = None,
    clean_cache: Optional[dict] = None,
    patch_specs: Sequence[PatchSpec] = (),
    return_pooled_blocks: bool = True,
    ablate_heads: Optional[dict] = None,  # {layer: [head_idx, ...]}
) -> tuple[torch.Tensor, dict]:
    """Manual forward pass with caching and activation patching.

    Args:
        model: PmhcuniModel
        sequence_tokens: (B, L)
        cache_targets: set of strings: {"embed", "resid_pre", "attn_out", "resid_mid",
                                         "ffn_out", "resid_post", "final_norm",
                                         "q", "k", "v", "alpha", "z",
                                         "pooled_blocks", "feat"}
        clean_cache: dict with clean-run activations (indexed by _key(target, layer))
        patch_specs: list of PatchSpec to apply
        return_pooled_blocks: whether to expose the 7 sep_concat blocks

    Returns:
        logits: (B, 1) for EL task
        cache: dict containing requested activations
    """
    cache: dict[str, Any] = {}
    cache_targets = cache_targets or set()
    tokenizer = model.tokenizer

    sequence_id = sequence_tokens != tokenizer.pad_token_id

    # Embed
    x = model.backbone.embed(sequence_tokens)
    x = _maybe_patch(x, "embed", None, clean_cache, patch_specs)
    if "embed" in cache_targets:
        cache["embed"] = x.detach().cpu()

    # Process each transformer block
    for layer_i, block in enumerate(model.backbone.transformer.blocks):
        # resid_pre
        x = _maybe_patch(x, "resid_pre", layer_i, clean_cache, patch_specs)
        if "resid_pre" in cache_targets:
            cache[_key("resid_pre", layer_i)] = x.detach().cpu()

        # Attention
        layer_ablate = (ablate_heads or {}).get(layer_i)
        if getattr(block, "use_plain_attn", False):
            attn_out, _ = attn_forward_with_cache(
                block.attn, x, sequence_id,
                layer=layer_i, cache=cache, clean_cache=clean_cache,
                patch_specs=patch_specs, cache_targets=cache_targets,
                ablate_heads=layer_ablate,
            )
        elif getattr(block, "use_geom_attn", False):
            raise NotImplementedError("Geom attention not yet supported in patching.")
        else:
            attn_out = torch.zeros_like(x)

        attn_out = _maybe_patch(attn_out, "attn_out", layer_i, clean_cache, patch_specs)
        if "attn_out" in cache_targets:
            cache[_key("attn_out", layer_i)] = attn_out.detach().cpu()

        x_mid = x + attn_out / block.scaling_factor
        x_mid = _maybe_patch(x_mid, "resid_mid", layer_i, clean_cache, patch_specs)
        if "resid_mid" in cache_targets:
            cache[_key("resid_mid", layer_i)] = x_mid.detach().cpu()

        # FFN
        ffn_out = block.ffn(x_mid)
        ffn_out = _maybe_patch(ffn_out, "ffn_out", layer_i, clean_cache, patch_specs)
        if "ffn_out" in cache_targets:
            cache[_key("ffn_out", layer_i)] = ffn_out.detach().cpu()

        x = x_mid + ffn_out / block.scaling_factor
        x = _maybe_patch(x, "resid_post", layer_i, clean_cache, patch_specs)
        if "resid_post" in cache_targets:
            cache[_key("resid_post", layer_i)] = x.detach().cpu()

    # Final norm
    x = model.backbone.transformer.norm(x)
    if "final_norm" in cache_targets:
        cache["final_norm"] = x.detach().cpu()

    # Build pooled blocks
    feats, pooled_blocks = build_pooled_blocks(model, sequence_tokens, x)
    if return_pooled_blocks:
        cache["pooled_blocks"] = {k: v.detach().cpu() for k, v in pooled_blocks.items()}
    if "feat" in cache_targets:
        cache["feat"] = feats.detach().cpu()

    # Logits
    if task == "el":
        logits = model.classifier(feats)
    elif task == "ba":
        logits = model.ba_head(feats)
    else:
        raise ValueError(f"unknown task {task}")

    return logits, cache


def build_pooled_blocks(
    model: PmhcuniModel,
    sequence_tokens: torch.Tensor,
    embeddings: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Replicate sep_concat pooling but expose individual pooled blocks.

    Returns:
        feats: (B, 7*d_model)
        blocks: dict with keys cls/mhc_mean/mhc_max/mhc_attn/pep_mean/pep_max/pep_attn
    """
    if model.pooling_mode != "sep_concat":
        raise NotImplementedError(f"build_pooled_blocks only supports sep_concat, got {model.pooling_mode}")

    cls_embedding = embeddings[:, 0, :]  # (B, d_model)

    mhc_mask, pep_mask = model._build_separate_masks(sequence_tokens)

    blocks = {"cls": cls_embedding}

    for region_name, region_mask, attn_pool in [
        ("mhc", mhc_mask, model.mhc_attention_pool),
        ("pep", pep_mask, model.pep_attention_pool),
    ]:
        rm_f = region_mask.unsqueeze(-1).float()
        denom = rm_f.sum(dim=1).clamp(min=1.0)
        mean_p = (embeddings * rm_f).sum(dim=1) / denom
        masked_e = embeddings.masked_fill(~region_mask.unsqueeze(-1), float("-inf"))
        max_p = masked_e.max(dim=1).values
        max_p = torch.nan_to_num(max_p, nan=0.0, posinf=0.0, neginf=0.0)
        attn_p = attn_pool(embeddings, region_mask)
        blocks[f"{region_name}_mean"] = mean_p
        blocks[f"{region_name}_max"] = max_p
        blocks[f"{region_name}_attn"] = attn_p

    feats = torch.cat([
        blocks["cls"],
        blocks["mhc_mean"], blocks["mhc_max"], blocks["mhc_attn"],
        blocks["pep_mean"], blocks["pep_max"], blocks["pep_attn"],
    ], dim=-1)
    return feats, blocks


# ─────────────────────────────────────────────────────────────────
# Verification helper
# ─────────────────────────────────────────────────────────────────

def verify_logit_equivalence(model, seq_tokens, atol=1e-4):
    """Compare logits from forward_with_cache_and_patch vs model.forward."""
    model.eval()
    with torch.inference_mode():
        # Standard forward
        out_std = model(seq_tokens)
        if isinstance(out_std, tuple):
            std_logits = out_std[0]
        else:
            std_logits = out_std

        # Manual forward
        logits_manual, _ = forward_with_cache_and_patch(model, seq_tokens, task="el")

    diff = (std_logits - logits_manual).abs().max().item()
    return diff, diff < atol
