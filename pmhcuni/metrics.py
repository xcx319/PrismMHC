"""
Group-level Average Precision metric.

Re-implemented standalone — no dependency on the original pmhcuni package.

In DDP training each rank only sees a subset of the validation data.
This implementation extends ``torchmetrics.Metric`` so that ``compute()``
automatically gathers predictions/targets/groups across all ranks before
computing per-group AP, giving the correct global metric on every process.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Optional

import torch
import numpy as np
from sklearn.metrics import average_precision_score

try:
    from torchmetrics import Metric as _TorchMetric
    _HAS_TORCHMETRICS = True
except ImportError:
    _HAS_TORCHMETRICS = False


if _HAS_TORCHMETRICS:
    class GroupAveragePrecision(_TorchMetric):
        """
        Per-group (e.g., per-allele) average precision.

        Extends ``torchmetrics.Metric`` so that DDP ``dist_reduce_fx`` gathers
        all per-rank predictions before computing AP.

        Args:
            groups: Optional ordered mapping of group integer id → group name.
        """

        # Tell torchmetrics to concatenate state lists across ranks on compute()
        full_state_update: bool = False

        def __init__(self, groups: Optional[OrderedDict[int, str]] = None, **kwargs):
            super().__init__(**kwargs)
            self.groups = groups or OrderedDict()
            # Each add_state registers a list that torchmetrics auto-gathers across ranks
            self.add_state("_preds",   default=[], dist_reduce_fx="cat")
            self.add_state("_targets", default=[], dist_reduce_fx="cat")
            self.add_state("_groups",  default=[], dist_reduce_fx="cat")

        def update(
            self,
            preds: torch.Tensor,
            targets: torch.Tensor,
            groups: torch.Tensor,
        ) -> None:
            # Keep states on the metric's (CUDA) device: torchmetrics gathers them
            # across ranks via NCCL at compute(), which cannot all_gather CPU
            # tensors (that failure was silently swallowed → val/el_avgp=-1.0).
            self._preds.append(preds.detach().float().view(-1))
            self._targets.append(targets.detach().long().view(-1))
            self._groups.append(groups.detach().long().view(-1))

        # Keep __call__ compatible with the old API (positional args)
        def __call__(self, preds, targets, groups):  # type: ignore[override]
            return self.update(preds, targets, groups)

        def compute(self) -> OrderedDict[str, float]:
            if not self._preds:
                return OrderedDict()
            # states are gathered across ranks (still on device) → move to CPU here
            all_preds   = torch.cat(self._preds).cpu().numpy()
            all_targets = torch.cat(self._targets).cpu().numpy()
            all_groups  = torch.cat(self._groups).cpu().numpy()
            return _compute_group_ap(all_preds, all_targets, all_groups, self.groups)

        def reset(self) -> None:
            super().reset()

else:
    # Fallback: pure-Python implementation (single-GPU / CPU only)
    class GroupAveragePrecision:  # type: ignore[no-redef]
        """
        Compute per-group (e.g., per-allele) average precision.

        **Single-process only** — torchmetrics is not installed, so DDP
        cross-rank gathering is unavailable.  Install torchmetrics for
        multi-GPU support.
        """

        def __init__(self, groups: Optional[OrderedDict[int, str]] = None):
            self.groups = groups or OrderedDict()
            self.reset()

        def reset(self) -> None:
            self._preds: list[torch.Tensor] = []
            self._targets: list[torch.Tensor] = []
            self._groups: list[torch.Tensor] = []

        def __call__(
            self,
            preds: torch.Tensor,
            targets: torch.Tensor,
            groups: torch.Tensor,
        ) -> None:
            self._preds.append(preds.detach().float().cpu())
            self._targets.append(targets.detach().long().cpu())
            self._groups.append(groups.detach().long().cpu())

        def compute(self) -> OrderedDict[str, float]:
            all_preds   = torch.cat(self._preds).numpy()
            all_targets = torch.cat(self._targets).numpy()
            all_groups  = torch.cat(self._groups).numpy()
            return _compute_group_ap(all_preds, all_targets, all_groups, self.groups)


def _compute_group_ap(
    all_preds: "np.ndarray",
    all_targets: "np.ndarray",
    all_groups: "np.ndarray",
    groups: OrderedDict[int, str],
) -> OrderedDict[str, float]:
    """Shared AP computation used by both implementations."""
    results: OrderedDict[str, float] = OrderedDict()
    group_data: dict[int, tuple[list, list]] = defaultdict(lambda: ([], []))

    for pred, target, group in zip(all_preds, all_targets, all_groups):
        gid = int(group)
        if gid < 0:
            continue
        group_data[gid][0].append(pred)
        group_data[gid][1].append(target)

    for gid, name in groups.items():
        if gid not in group_data:
            continue
        preds_g   = np.array(group_data[gid][0])
        targets_g = np.array(group_data[gid][1])
        if len(np.unique(targets_g)) < 2:
            continue
        ap = average_precision_score(targets_g, preds_g)
        results[name] = float(ap)

    return results
