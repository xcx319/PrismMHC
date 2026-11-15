"""
Training script for PmhcuniModel.

Supports Stage-1 (single-allele), Stage-2 MIL (multi-allele), and
Stage-2 V2 (pre-predict MA) training modes.

Usage:
    # Stage-1 training
    python training.py config/default.yaml --output ./output

    # Stage-2 MIL training (default when sa_path + ma_path are set)
    python training.py config/default.yaml --output ./output --resume ./output/checkpoints/last.ckpt

    # Stage-2 V2: pre-predict MA mode (faster, no per-batch MIL overhead)
    python training.py config/default.yaml --output ./output --resume ./output/checkpoints/last.ckpt --ma-mode v2

    # Debug mode (single device, fast_dev_run)
    python training.py config/default.yaml --output ./output --debug
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
import copy
import yaml
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

from pmhcuni.model import PmhcuniModel
from pmhcuni.data import PmhcuniDataset, Stage2DataModule, Stage2V2DataModule
from pmhcuni.utils import load_allele_sequences


# ──────────────────────────────────────────────────────────────────
# Callback: update MA predictions at the start of each epoch (V2 mode)
# ──────────────────────────────────────────────────────────────────
class UpdateMaPredictionsCallback(Callback):
    """
    Lightning Callback that re-predicts MA winner alleles at the start of
    each training epoch.

    After prediction the DataModule's training dataset is rebuilt; the Trainer
    must set ``reload_dataloaders_every_n_epochs=1`` so the new DataLoader is
    picked up.

    Args:
        skip_first_epoch: If True, epoch 0 uses the prediction that was
            already performed in ``main()`` (or the random initialisation
            from ``setup()``), and re-prediction starts from epoch 1.
    """

    def __init__(self, skip_first_epoch: bool = False):
        super().__init__()
        self.skip_first_epoch = skip_first_epoch
        self._initial_done = False

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: PmhcuniModel
    ) -> None:
        datamodule = trainer.datamodule
        if not isinstance(datamodule, Stage2V2DataModule):
            return

        epoch = trainer.current_epoch

        if epoch == 0 and self.skip_first_epoch and not self._initial_done:
            self._initial_done = True
            print(f"[Epoch {epoch}] Using initial MA allele assignment (skip re-predict)")
            return

        print(f"\n{'='*60}")
        print(f"[Epoch {epoch}] Updating MA predictions with current model ...")
        print(f"{'='*60}")

        was_training = pl_module.training
        pl_module.eval()

        datamodule.update_ma_predictions(pl_module)
        datamodule._rebuild_train_dataset()

        if was_training:
            pl_module.train()

        self._initial_done = True


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train PmhcuniModel")
    parser.add_argument("config", type=str, help="Path to YAML config file.")
    parser.add_argument("--output", type=str, default="./output", help="Output dir.")
    parser.add_argument(
        "--allele_file", type=str, default=None,
        help="Path to allele pseudo-sequence file (CSV: allele,sequence).",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--no_compile", action="store_true", default=False,
                        help="Disable torch.compile (avoids the _orig_mod key "
                             "mismatch when true-resuming a compiled ckpt under DDP).")
    parser.add_argument("--resume", type=str, default=None, help="Resume from ckpt.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed (pl.seed_everything). "
                             "Ablation cells differ only by this for replication.")
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="PyTorch-Lightning ckpt_path: TRUE resume — restores "
                             "global_step + optimizer and CONTINUES to trainer.max_steps "
                             "(does not add another max_steps).")
    parser.add_argument(
        "--ma-mode", type=str, default=None, choices=["mil", "v2"],
        help=(
            "Stage-2 multi-allele training mode. "
            "'mil' = original MIL winner-takes-all (per-batch inference); "
            "'v2'  = pre-predict MA winners each epoch (faster). "
            "Defaults to config value data.ma_mode, or 'mil' if unset. "
            "Only takes effect when both sa_path and ma_path are configured."
        ),
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_datamodule(data_cfg: dict, allele_sequences: dict, ma_mode: str = "mil"):
    """
    Build the appropriate DataModule based on config.

    If both ``sa_path`` and ``ma_path`` are present, builds a Stage-2
    DataModule.  The ``ma_mode`` parameter selects the variant:

    - ``"mil"``  — original :class:`Stage2DataModule` with per-batch MIL
      winner-takes-all loss (slower, but simpler).
    - ``"v2"``   — :class:`Stage2V2DataModule` that pre-predicts MA winners
      before each epoch and trains on flattened SA-style data (faster).

    If only SA data is configured, builds a Stage-1 :class:`PmhcuniDataset`.

    Args:
        data_cfg: Data configuration dict (from YAML config).
        allele_sequences: Ordered dict mapping allele name to pseudo-sequence.
        ma_mode: ``"mil"`` or ``"v2"`` (only affects Stage-2).

    Returns:
        A PyTorch Lightning DataModule ready for training.
    """
    data_cfg = dict(data_cfg)
    data_cfg.pop("allele_sequences", None)
    # Pop ma_mode from data_cfg so it is not forwarded to DataModule constructors
    data_cfg.pop("ma_mode", None)

    sa_path = data_cfg.pop("sa_path", None)
    ma_path = data_cfg.pop("ma_path", None)

    if sa_path and ma_path:
        # Stage-2: combined SA + MA training
        # Remove Stage-1-only keys
        data_cfg.pop("steps_per_epoch", None)

        if ma_mode == "v2":
            # V2: pre-predict MA → flatten → merge with SA (no MIL)
            # Remove MIL-only key
            data_cfg.pop("k_max", None)
            dataset = Stage2V2DataModule(
                sa_path=sa_path,
                ma_path=ma_path,
                allele_sequences=allele_sequences or {},
                **data_cfg,
            )
        else:
            # Original MIL mode
            # Remove V2-only keys
            data_cfg.pop("predict_batch_size", None)
            data_cfg.pop("predict_num_workers", None)
            dataset = Stage2DataModule(
                sa_path=sa_path,
                ma_path=ma_path,
                allele_sequences=allele_sequences or {},
                **data_cfg,
            )
    else:
        # Stage-1 (or SA-only): use PmhcuniDataset
        # Remove stage-2-only keys that PmhcuniDataset doesn't accept
        data_cfg.pop("sa_repeat", None)
        data_cfg.pop("k_max", None)
        data_cfg.pop("predict_batch_size", None)
        data_cfg.pop("predict_num_workers", None)
        dataset = PmhcuniDataset(
            allele_sequences=allele_sequences or {},
            **data_cfg,
        )

    dataset.prepare_data()
    dataset.setup()
    return dataset


def build_model(model_cfg: dict, allele_sequences: dict, resume_path: str = None):
    """
    Build or load the PmhcuniModel.

    Args:
        model_cfg: Model configuration dict (from YAML config).
        allele_sequences: Ordered dict mapping allele name to pseudo-sequence.
        resume_path: Path to checkpoint for resuming training.

    Returns:
        A PmhcuniModel instance.
    """
    if resume_path:
        # Modules that may legitimately be absent from an older checkpoint
        _NEW_MODULE_PREFIXES = ("ba_head.", "attention_pool.", "immuno_head.", "immuno_avgp.")

        # ── Fix torch.compile key prefix ──
        # torch.compile wraps modules under ``_orig_mod``, so checkpoints
        # saved from compiled models contain keys like
        # ``backbone._orig_mod.transformer.…`` instead of
        # ``backbone.transformer.…``.  Strip the prefix so the weights
        # can be loaded into an un-compiled model.
        ckpt = torch.load(resume_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", {})
        cleaned_state_dict = {
            k.replace("._orig_mod", ""): v for k, v in state_dict.items()
        }

        # Build a fresh model, then load the cleaned state dict
        model = PmhcuniModel(
            allele_sequences=allele_sequences,
            **model_cfg,
        )
        load_result = model.load_state_dict(cleaned_state_dict, strict=False)

        # Verify that any missing keys come only from newly-added modules
        unexpected_missing = [
            k for k in load_result.missing_keys
            if not any(k.startswith(p) for p in _NEW_MODULE_PREFIXES)
        ]
        if unexpected_missing:
            raise RuntimeError(
                f"Unexpected missing keys when loading checkpoint "
                f"(not from newly-added modules): {unexpected_missing}"
            )
        if load_result.missing_keys:
            print(
                f"[INFO] Loaded checkpoint with {len(load_result.missing_keys)} "
                f"missing key(s) from new modules (randomly initialised): "
                f"{sorted(load_result.missing_keys)}"
            )
        if load_result.unexpected_keys:
            print(
                f"[WARN] Checkpoint contained {len(load_result.unexpected_keys)} "
                f"unexpected key(s) (ignored): {sorted(load_result.unexpected_keys)}"
            )
    
    else:
        model = PmhcuniModel(
            allele_sequences=allele_sequences,
            **model_cfg,
        )
    return model


def build_trainer(trainer_cfg: dict, output_dir: str, debug: bool = False,
                  ma_mode: str = "mil"):
    """
    Build a PyTorch Lightning Trainer.

    Args:
        trainer_cfg: Trainer configuration dict (from YAML config).
        output_dir: Directory for checkpoints and logs.
        debug: If True, enable anomaly detection and fast_dev_run.
        ma_mode: ``"mil"`` or ``"v2"``.  When ``"v2"``, adds
            :class:`UpdateMaPredictionsCallback` and sets
            ``reload_dataloaders_every_n_epochs=1`` so the rebuilt dataset
            is picked up each epoch.

    Returns:
        A PyTorch Lightning Trainer instance.
    """
    trainer_cfg = copy.deepcopy(trainer_cfg)
    devices = trainer_cfg.get("devices", 1)

    if debug:
        torch.autograd.set_detect_anomaly(True)
        trainer_cfg.update(devices=1, fast_dev_run=True)

    if devices > 1:
        # Both heads are now forwarded every step (see training_step), so all
        # params always get gradients → find_unused_parameters=False + static_graph.
        # bf16_compress_hook halves the gradient all-reduce volume (critical when
        # P2P is broken and traffic goes over the slow host/SHM path).
        # NOTE: static_graph=True is INCOMPATIBLE with torch.compile here
        # (expect_autograd_hooks_ INTERNAL ASSERT) — omit it. find_unused=False is
        # safe now that both heads are forwarded every step.
        ddp_kwargs = dict(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=128,
        )
        try:
            from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
                bf16_compress_hook,
            )
            ddp_kwargs["ddp_comm_hook"] = bf16_compress_hook
        except Exception as _e:  # pragma: no cover
            print(f"[WARN] bf16_compress_hook unavailable: {_e}")
        trainer_cfg["strategy"] = DDPStrategy(**ddp_kwargs)

    # Checkpoint callback. Monitor metric is config-overridable (BA-specialist
    # selects on val/ba_pearson instead of the default val/el_avgp).
    sel_metric = trainer_cfg.pop("select_metric", None)
    sel_mode = trainer_cfg.pop("select_mode", None)
    if sel_metric:
        metric, mode = sel_metric, (sel_mode or "max")
        print(f"[INFO] checkpoint selection metric overridden -> {metric} ({mode})")
    else:
        metric, mode = PmhcuniModel.selection_metric()
    # save_weights_only (default False, zero behavior change unless set): skips the
    # optimizer/scheduler state, cutting checkpoint size ~2-3x. Useful under disk
    # pressure for short exploratory FT runs that never need exact-state resume.
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        monitor=metric,
        mode=mode,
        save_top_k=1,
        save_last=True,
        save_weights_only=trainer_cfg.pop("save_weights_only", False),
    )
    trainer_cfg.setdefault("callbacks", []).append(checkpoint_cb)

    # Optional early stopping — train to CONVERGENCE instead of a hand-picked
    # max_steps. Earlier FT runs harvested checkpoints at fixed steps and then
    # picked the best one on the eval set, which selects on the test set. With a
    # protein-disjoint data.val_path, this stops on an honest signal instead.
    es_patience = trainer_cfg.pop("early_stop_patience", None)
    if es_patience:
        early_stop_cb = EarlyStopping(
            monitor=metric,
            mode=mode,
            patience=int(es_patience),
            min_delta=float(trainer_cfg.pop("early_stop_min_delta", 0.0)),
            verbose=True,
        )
        trainer_cfg["callbacks"].append(early_stop_cb)
        print(f"[INFO] early stopping on {metric} ({mode}), patience={es_patience} val checks")

    # Optional periodic checkpoints (convergence / data-scaling studies): config
    # trainer.save_every_n_steps=N -> keep a ckpt every N steps (not pruned).
    save_every = trainer_cfg.pop("save_every_n_steps", None)
    if save_every:
        # Self-pruning periodic ckpts: monitor=None + save_top_k=1 keeps only the
        # most-recent periodic and deletes the previous one (empirically reliable;
        # monitor="step" did NOT prune in PL 2.6.1 here). save_top_k>1 with
        # monitor=None raises MisconfigurationException, so k must be 1. A separate
        # nohup prune-loop is used as a belt-and-suspenders backstop on tight disk.
        periodic_cb = ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            every_n_train_steps=int(save_every),
            save_top_k=1, save_last=False,
            filename="periodic-step{step}", auto_insert_metric_name=False,
        )
        trainer_cfg["callbacks"].append(periodic_cb)
        print(f"[INFO] periodic checkpoints every {save_every} steps")

    # V2 mode: add MA prediction callback and reload dataloaders every epoch
    if ma_mode == "v2":
        ma_cb = UpdateMaPredictionsCallback(skip_first_epoch=True)
        trainer_cfg["callbacks"].append(ma_cb)
        trainer_cfg["reload_dataloaders_every_n_epochs"] = 1

    trainer_cfg["default_root_dir"] = output_dir
    # BalancedBatchSampler handles DDP sharding internally; prevent Lightning from
    # wrapping the DataLoader with an extra DistributedSampler which would conflict
    # with the custom batch_sampler.
    trainer_cfg["use_distributed_sampler"] = False

    return pl.Trainer(**trainer_cfg)


def main():
    # Improve Tensor Core utilization for speed (recommended for Ampere+ GPUs)
    torch.set_float32_matmul_precision("medium")

    args = parse_args()

    # Reproducibility — ablation cells are identical except for --seed.
    pl.seed_everything(args.seed, workers=True)

    # Load config
    config = load_config(args.config)
    os.makedirs(args.output, exist_ok=True)

    # Load allele sequences (CLI overrides config)
    data_cfg = config.get("data", {})
    allele_file = args.allele_file or data_cfg.get("allele_sequences")

    allele_sequences = None
    if allele_file:
        allele_sequences = load_allele_sequences(allele_file)

    # ── Resolve MA mode ──
    # Priority: CLI --ma-mode > config data.ma_mode > default "mil"
    ma_mode = args.ma_mode or data_cfg.get("ma_mode", "mil")
    print(f"[INFO] MA training mode: {ma_mode}")

    # Build data, model, trainer
    dataset = build_datamodule(data_cfg, allele_sequences, ma_mode=ma_mode)
    model = build_model(
        config.get("model", {}), allele_sequences, args.resume
    )
    # torch.compile is applied inside PmhcuniModel.configure_model(),
    # which Lightning calls after DDP wrapping — see model.py.
    model._use_compile = not args.debug and not args.no_compile

    # ── V2 mode: initial MA prediction with loaded checkpoint ──
    if ma_mode == "v2" and args.resume and isinstance(dataset, Stage2V2DataModule):
        print("\n[Initial] Running MA prediction with loaded checkpoint ...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        dataset.update_ma_predictions(model)
        dataset._rebuild_train_dataset()
        model.cpu()  # Move back to CPU; Lightning will place on correct device
        model.train()

    trainer = build_trainer(
        config.get("trainer", {}), args.output, args.debug,
        ma_mode=ma_mode,
    )

    # Train. resume_ckpt != None → PL TRUE resume: continue from saved global_step
    # to trainer.max_steps (restores optimizer + step), NOT another full max_steps.
    if args.resume_ckpt:
        print(f"[INFO] TRUE resume from {args.resume_ckpt} -> continue to max_steps")
    trainer.fit(model, datamodule=dataset, ckpt_path=args.resume_ckpt)


if __name__ == "__main__":
    main()
