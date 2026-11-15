"""
Training script for PmhcuniModel — Stage-2 V2 (pre-predict MA pipeline).

Instead of running MIL winner-takes-all inside every training batch (which
requires an extra no_grad forward pass over all K candidates per batch), this
script:

1. Before each epoch, batch-predicts the winner allele for every MA record.
2. Flattens MA data into SA-style records (peptide, winner_allele, label).
3. Merges flattened MA with SA for standard flat training (single forward per sample).
4. After each epoch, re-predicts MA winners with the updated model.

This eliminates the per-batch MIL overhead, making training significantly faster.

Usage:
    # Stage-2 V2 training (resume from Stage-1 checkpoint)
    python training_v2.py config/stage2_v2.yaml --output ./outputv7 \
        --resume ./checkpoints/v3/checkpoints/last.ckpt

    # Debug mode
    python training_v2.py config/stage2_v2.yaml --output ./outputv7 --debug
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
import copy
import time
import yaml
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.strategies import DDPStrategy

from pmhcuni.model import PmhcuniModel
from pmhcuni.data import Stage2V2DataModule
from pmhcuni.utils import load_allele_sequences


# ──────────────────────────────────────────────────────────────────
# Callback: update MA predictions at the start of each epoch
# ──────────────────────────────────────────────────────────────────
class ProgressLogCallback(Callback):
    """Print training progress every N steps (works with nohup/file redirect)."""

    def __init__(self, every_n_steps: int = 500):
        super().__init__()
        self.every_n_steps = every_n_steps
        self._last_print = 0
        import time as _time
        self._start_time = _time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step - self._last_print >= self.every_n_steps:
            import time as _time, sys
            elapsed = _time.time() - self._start_time
            hours = elapsed / 3600
            speed = step / elapsed if elapsed > 0 else 0
            total = trainer.estimated_stepping_batches
            eta_h = (total - step) / speed / 3600 if speed > 0 else 0
            loss = trainer.callback_metrics.get("train/loss", 0)
            el_loss = trainer.callback_metrics.get("train/el_loss", 0)
            print(
                f"[Step {step}/{total}] "
                f"loss={float(loss):.4f} el_loss={float(el_loss):.4f} "
                f"speed={speed:.2f}it/s elapsed={hours:.1f}h ETA={eta_h:.1f}h",
                flush=True,
            )
            self._last_print = step


class UpdateMaPredictionsCallback(Callback):
    """
    Lightning Callback that re-predicts MA winner alleles at the start of
    each training epoch (except epoch 0 which uses the initial prediction
    from setup or the first explicit call).

    After prediction, the DataModule's training dataset is rebuilt and the
    DataLoader is refreshed.
    """

    def __init__(self, skip_first_epoch: bool = False):
        super().__init__()
        self.skip_first_epoch = skip_first_epoch
        self._initial_done = False

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: PmhcuniModel) -> None:
        datamodule = trainer.datamodule
        if not isinstance(datamodule, Stage2V2DataModule):
            return

        epoch = trainer.current_epoch

        if epoch == 0 and self.skip_first_epoch and not self._initial_done:
            # For epoch 0, if skip_first_epoch is set, use initial random flatten
            # (already done in setup). But still mark as done.
            self._initial_done = True
            print(f"[Epoch {epoch}] Using initial random MA allele assignment")
            return

        # Re-predict MA winners using current model weights
        print(f"\n{'='*60}")
        print(f"[Epoch {epoch}] Updating MA predictions with current model ...")
        print(f"{'='*60}")

        # Temporarily disable gradient computation and set eval mode
        was_training = pl_module.training
        pl_module.eval()

        datamodule.update_ma_predictions(pl_module)
        datamodule._rebuild_train_dataset()

        if was_training:
            pl_module.train()

        self._initial_done = True


# ──────────────────────────────────────────────────────────────────
# CLI and config
# ──────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PmhcuniModel — Stage-2 V2 (pre-predict MA)"
    )
    parser.add_argument("config", type=str, help="Path to YAML config file.")
    parser.add_argument("--output", type=str, default="./output", help="Output dir.")
    parser.add_argument(
        "--allele_file", type=str, default=None,
        help="Path to allele pseudo-sequence file (CSV: allele,sequence).",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Load model weights from a checkpoint and start a new Trainer trajectory.",
    )
    resume_group.add_argument(
        "--resume_trainer",
        type=str,
        default=None,
        help=(
            "Resume the complete Lightning Trainer state (model, optimizer, "
            "scheduler, loops, and global step) from a checkpoint. Requires "
            "--load_ma_flatten so the Stage-2 V2 dataset state is explicit."
        ),
    )
    parser.add_argument(
        "--save_ma_flatten", type=str, default=None,
        help="Save MA flatten results to this CSV path (run prediction then exit).",
    )
    parser.add_argument(
        "--load_ma_flatten", type=str, default=None,
        help="Load pre-computed MA flatten results from this CSV instead of running prediction.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_datamodule(data_cfg: dict, allele_sequences: dict) -> Stage2V2DataModule:
    """Build Stage2V2DataModule from config."""
    data_cfg = dict(data_cfg)
    data_cfg.pop("allele_sequences", None)

    sa_path = data_cfg.pop("sa_path", None)
    ma_path = data_cfg.pop("ma_path", None)

    if not sa_path or not ma_path:
        raise ValueError(
            "Stage-2 V2 training requires both 'sa_path' and 'ma_path' in config."
        )

    # Remove Stage-1-only keys
    data_cfg.pop("steps_per_epoch", None)
    # Remove old stage-2 key that Stage2V2 doesn't use
    data_cfg.pop("k_max", None)

    dataset = Stage2V2DataModule(
        sa_path=sa_path,
        ma_path=ma_path,
        allele_sequences=allele_sequences or {},
        **data_cfg,
    )
    return dataset


def build_model(model_cfg: dict, allele_sequences: dict, resume_path: str = None):
    """Build or load PmhcuniModel."""
    if resume_path:
        _NEW_MODULE_PREFIXES = ("ba_head.", "attention_pool.", "mhc_attention_pool.", "pep_attention_pool.",
                                 "immuno_head.", "immuno_avgp.")

        ckpt = torch.load(resume_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", {})
        cleaned_state_dict = {
            k.replace("._orig_mod", ""): v for k, v in state_dict.items()
        }

        model = PmhcuniModel(
            allele_sequences=allele_sequences,
            **model_cfg,
        )

        # Filter out keys with shape mismatch (e.g. head dim change from
        # pooling mode switch) so they become missing keys and get re-initialised.
        _HEAD_PREFIXES = ("classifier.", "ba_head.")
        model_state = model.state_dict()
        shape_mismatched = []
        for k, v in list(cleaned_state_dict.items()):
            if k in model_state and v.shape != model_state[k].shape:
                shape_mismatched.append(k)
                del cleaned_state_dict[k]
        if shape_mismatched:
            print(
                f"[INFO] Skipped {len(shape_mismatched)} key(s) with shape "
                f"mismatch (will be re-initialised): {sorted(shape_mismatched)}"
            )

        load_result = model.load_state_dict(cleaned_state_dict, strict=False)

        unexpected_missing = [
            k for k in load_result.missing_keys
            if not any(k.startswith(p) for p in _NEW_MODULE_PREFIXES)
            and k not in shape_mismatched
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


def build_trainer(
    trainer_cfg: dict,
    output_dir: str,
    debug: bool = False,
    skip_first_epoch_predict: bool = False,
):
    """Build Trainer with UpdateMaPredictionsCallback."""
    trainer_cfg = copy.deepcopy(trainer_cfg)
    devices = trainer_cfg.get("devices", 1)

    if debug:
        torch.autograd.set_detect_anomaly(True)
        trainer_cfg.update(devices=1, fast_dev_run=True)

    if devices > 1:
        trainer_cfg["strategy"] = DDPStrategy(find_unused_parameters=False)

    # Checkpoint callback — best model by val metric
    metric, mode = PmhcuniModel.selection_metric()
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        monitor=metric,
        mode=mode,
        save_top_k=1,
        save_last=True,
    )
    trainer_cfg.setdefault("callbacks", []).append(checkpoint_cb)

    # Step-based periodic checkpoint (guard against crashes)
    save_every_n_steps = trainer_cfg.pop("save_every_n_steps", None)
    if save_every_n_steps:
        step_ckpt_cb = ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="step-{step}",
            every_n_train_steps=save_every_n_steps,
            save_top_k=-1,  # keep all step checkpoints
        )
        trainer_cfg["callbacks"].append(step_ckpt_cb)

    # MA prediction update callback
    ma_cb = UpdateMaPredictionsCallback(
        skip_first_epoch=skip_first_epoch_predict,
    )
    trainer_cfg["callbacks"].append(ma_cb)

    # Progress logging callback (flushes to file, works with nohup)
    trainer_cfg["callbacks"].append(ProgressLogCallback(every_n_steps=500))

    trainer_cfg["default_root_dir"] = output_dir
    trainer_cfg["use_distributed_sampler"] = False

    # Reload dataloaders every epoch so the rebuilt dataset is picked up
    trainer_cfg["reload_dataloaders_every_n_epochs"] = 1

    return pl.Trainer(**trainer_cfg)


def main():
    torch.set_float32_matmul_precision("medium")

    args = parse_args()
    if args.resume_trainer and not args.load_ma_flatten:
        raise ValueError(
            "--resume_trainer requires --load_ma_flatten because the Stage-2 "
            "V2 MA winner mapping is not stored in Lightning checkpoints."
        )
    if args.save_ma_flatten and args.resume_trainer:
        raise ValueError(
            "--save_ma_flatten is a weights-only prediction mode and cannot be "
            "combined with --resume_trainer."
        )
    config = load_config(args.config)
    os.makedirs(args.output, exist_ok=True)

    # Load allele sequences
    data_cfg = config.get("data", {})
    allele_file = args.allele_file or data_cfg.get("allele_sequences")
    allele_sequences = None
    if allele_file:
        allele_sequences = load_allele_sequences(allele_file)

    # Build data, model, trainer
    dataset = build_datamodule(data_cfg, allele_sequences)
    dataset.prepare_data()
    dataset.setup()

    model = build_model(
        config.get("model", {}), allele_sequences, args.resume
    )
    model._use_compile = False  # disable torch.compile to reduce CPU overhead

    # If we have a resumed model, do an initial MA prediction with it
    # (better than random allele assignment from setup)
    if args.load_ma_flatten:
        # Load pre-computed MA flatten results — skip model prediction entirely
        dataset.load_ma_flatten(args.load_ma_flatten)
        dataset._rebuild_train_dataset()
        skip_first = True
    elif args.resume:
        print("\n[Initial] Running MA prediction with loaded checkpoint ...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        dataset.update_ma_predictions(model)
        model.cpu()  # Move back to CPU; Lightning will move to correct device

        # If --save_ma_flatten is set, save and exit (skip expensive rebuild)
        if args.save_ma_flatten:
            dataset.save_ma_flatten(args.save_ma_flatten)
            print("MA flatten saved. Exiting (--save_ma_flatten mode).")
            return

        dataset._rebuild_train_dataset()
        skip_first = True
    else:
        skip_first = True  # First epoch uses random assignment from setup

    trainer = build_trainer(
        config.get("trainer", {}),
        args.output,
        args.debug,
        skip_first_epoch_predict=skip_first,
    )

    # Train
    trainer.fit(
        model,
        datamodule=dataset,
        ckpt_path=args.resume_trainer,
    )


if __name__ == "__main__":
    main()
