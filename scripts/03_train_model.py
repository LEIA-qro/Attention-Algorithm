#!/usr/bin/env python3

"""
03_train_model.py — Train DriverStateNet (Bi-LSTM + Attention) on extracted features.

Pipeline:
  1. Load train/val split CSVs
  2. Read feature .parquet files, create sliding windows (seq_len=90, stride=15)
  3. Compute class weights from training distribution
  4. Train with AdamW + OneCycleLR + AMP + early stopping
  5. TensorBoard logging, checkpoint best model by val macro F1

Usage:
    python scripts/03_train_model.py --config config/config.yaml
    python scripts/03_train_model.py --config config/config.yaml --epochs 100 --batch-size 512
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import yaml

# Project imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import DriverStateNet  # noqa: E402
from src.dataset import DriverStateDataset  # noqa: E402

logger = logging.getLogger("dms.train")

# Constants
LABEL_MAP = {"Alert": 0, "Drowsy": 1, "Distracted": 2}
LABEL_NAMES = ["Alert", "Drowsy", "Distracted"]

FEATURE_COLS = [
    "ear_left", "ear_right", "ear_avg", "mar",
    "perclos", "blink_rate", "blink_duration_avg",
    "yaw", "pitch", "roll",
    "gaze_yaw", "gaze_pitch", "gaze_stability",
    "head_pose_stability", "ear_velocity",
    "head_nod_count", "mouth_open_duration",
    "eyes_off_road_pct",
]

NUM_FEATURES = len(FEATURE_COLS)  # 18


# Helpers

def _setup_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s"
    logging.basicConfig(level=numeric, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        logger.warning("Config %s not found — using defaults.", path)
        return {}
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# Sliding-window dataset

class SlidingWindowDataset(Dataset):
    """Create overlapping sliding windows from feature .parquet files.

    Each sample is a (seq_len, n_features) tensor with a majority-vote label.
    """

    def __init__(
        self,
        split_csv: Path,
        seq_len: int = 90,
        stride: int = 15,
        feature_cols: Optional[List[str]] = None,
        label_map: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride
        self.feature_cols = feature_cols or FEATURE_COLS
        self.label_map = label_map or LABEL_MAP

        self.windows: List[Tuple[np.ndarray, int]] = []
        self._load_from_split(split_csv)

    def _load_from_split(self, split_csv: Path) -> None:
        """Read split CSV, load parquet files, create sliding windows."""
        split_df = pd.read_csv(split_csv)
        feature_files = split_df["feature_file"].tolist()

        for fpath in tqdm(feature_files, desc=f"Loading {split_csv.stem}", unit="file", leave=False):
            fpath = Path(fpath)
            if not fpath.exists():
                logger.warning("Feature file not found: %s", fpath)
                continue

            try:
                df = pd.read_parquet(fpath)
            except Exception as exc:
                logger.warning("Cannot read %s: %s", fpath, exc)
                continue

            # Ensure all feature columns are present
            missing = [c for c in self.feature_cols if c not in df.columns]
            if missing:
                logger.warning("Missing columns in %s: %s — skipping.", fpath, missing)
                continue

            features = df[self.feature_cols].values.astype(np.float32)
            labels_str = df["label"].values

            # Replace NaN with 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Map labels to ints
            labels_int = np.array([self.label_map.get(l, 0) for l in labels_str], dtype=np.int64)

            # Create sliding windows
            n = len(features)
            if n < self.seq_len:
                # Pad short sequences
                pad_len = self.seq_len - n
                features = np.pad(features, ((0, pad_len), (0, 0)), mode="edge")
                labels_int = np.pad(labels_int, (0, pad_len), mode="edge")
                n = self.seq_len

            for start in range(0, n - self.seq_len + 1, self.stride):
                end = start + self.seq_len
                window_features = features[start:end]
                window_labels = labels_int[start:end]

                # Majority vote for window label
                counts = Counter(window_labels)
                label = counts.most_common(1)[0][0]

                self.windows.append((window_features, label))

        logger.info("Created %d windows from %s", len(self.windows), split_csv.stem)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features, label = self.windows[idx]
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)


# Class weight computation

def compute_class_weights(dataset: SlidingWindowDataset, num_classes: int = 3) -> torch.Tensor:
    """Compute inverse-frequency class weights for weighted CE loss."""
    label_counts = Counter()
    for _, label in dataset.windows:
        label_counts[label] += 1

    total = sum(label_counts.values())
    weights = torch.zeros(num_classes)
    for cls_idx in range(num_classes):
        count = label_counts.get(cls_idx, 1)
        weights[cls_idx] = total / (num_classes * count)

    logger.info("Class counts: %s", dict(label_counts))
    logger.info("Class weights: %s", weights.tolist())
    return weights


# Training loop

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch.  Returns dict of metrics."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False, unit="batch")

    for features, labels in pbar:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(features)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(len(LABEL_NAMES))))

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "f1_alert": float(per_class_f1[0]) if len(per_class_f1) > 0 else 0.0,
        "f1_drowsy": float(per_class_f1[1]) if len(per_class_f1) > 1 else 0.0,
        "f1_distracted": float(per_class_f1[2]) if len(per_class_f1) > 2 else 0.0,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Evaluate model on validation set.  Returns dict of metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]  ", leave=False, unit="batch")

    for features, labels in pbar:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(features)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(len(LABEL_NAMES))))

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "f1_alert": float(per_class_f1[0]) if len(per_class_f1) > 0 else 0.0,
        "f1_drowsy": float(per_class_f1[1]) if len(per_class_f1) > 1 else 0.0,
        "f1_distracted": float(per_class_f1[2]) if len(per_class_f1) > 2 else 0.0,
    }


# Main training driver

def main() -> None:
    parser = argparse.ArgumentParser(description="Train DriverStateNet.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=90)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    _setup_logging(args.log_level)
    cfg = _load_config(args.config)

    # Override from config if present
    train_cfg = cfg.get("training", {})
    epochs = train_cfg.get("epochs", args.epochs)
    batch_size = train_cfg.get("batch_size", args.batch_size)
    lr = train_cfg.get("lr", args.lr)
    weight_decay = train_cfg.get("weight_decay", args.weight_decay)
    seq_len = train_cfg.get("seq_len", args.seq_len)
    stride = train_cfg.get("stride", args.stride)
    hidden_size = train_cfg.get("hidden_size", args.hidden_size)
    num_layers = train_cfg.get("num_layers", args.num_layers)
    dropout = train_cfg.get("dropout", args.dropout)
    patience = train_cfg.get("patience", args.patience)

    # Paths
    project_root = _PROJECT_ROOT
    data_root = Path(args.data_root) if args.data_root else Path(
        cfg.get("data", {}).get("root", project_root / "data")
    )
    splits_dir = data_root / "splits"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = project_root / "runs" / f"train_{time.strftime('%Y%m%d_%H%M%S')}"

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    for p in [train_csv, val_csv]:
        if not p.exists():
            logger.error("Split file not found: %s — run 02_build_splits.py first.", p)
            sys.exit(1)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Datasets & DataLoaders
    logger.info("Loading training data…")
    train_dataset = SlidingWindowDataset(train_csv, seq_len=seq_len, stride=stride)
    logger.info("Loading validation data…")
    val_dataset = SlidingWindowDataset(val_csv, seq_len=seq_len, stride=stride)

    if len(train_dataset) == 0:
        logger.error("Training dataset is empty!")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    logger.info("Train: %d windows, Val: %d windows", len(train_dataset), len(val_dataset))

    # Model
    model = DriverStateNet(
        input_dim=NUM_FEATURES,
        hidden_dim=hidden_size,
        num_layers=num_layers,
        num_classes=len(LABEL_NAMES),
        dropout_lstm=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s params total, %s trainable", f"{total_params:,}", f"{trainable_params:,}")

    # Loss with class weights
    class_weights = compute_class_weights(train_dataset, num_classes=len(LABEL_NAMES))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    # Resume from checkpoint
    start_epoch = 0
    best_val_f1 = 0.0

    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            logger.info("Resuming from %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_f1 = ckpt.get("best_val_f1", 0.0)
            logger.info("Resumed at epoch %d, best F1: %.4f", start_epoch, best_val_f1)

    # TensorBoard
    writer = SummaryWriter(log_dir=str(tb_dir))
    logger.info("TensorBoard logs: %s", tb_dir)

    # Log hyperparameters
    hparams = {
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "weight_decay": weight_decay, "seq_len": seq_len, "stride": stride,
        "hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout,
    }
    writer.add_text("hyperparameters", json.dumps(hparams, indent=2), 0)

    # Training loop
    patience_counter = 0
    training_start = time.time()

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING START — %d epochs, batch=%d, lr=%.1e, device=%s",
                epochs, batch_size, lr, device)
    logger.info("=" * 70)

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        logger.info(
            "Epoch %3d/%d │ train loss=%.4f acc=%.3f F1=%.3f │ "
            "val loss=%.4f acc=%.3f F1=%.3f │ lr=%.2e │ %.1fs",
            epoch + 1, epochs,
            train_metrics["loss"], train_metrics["accuracy"], train_metrics["macro_f1"],
            val_metrics["loss"], val_metrics["accuracy"], val_metrics["macro_f1"],
            current_lr, epoch_time,
        )

        # TensorBoard
        writer.add_scalars("loss", {"train": train_metrics["loss"], "val": val_metrics["loss"]}, epoch)
        writer.add_scalars("accuracy", {"train": train_metrics["accuracy"], "val": val_metrics["accuracy"]}, epoch)
        writer.add_scalars("macro_f1", {"train": train_metrics["macro_f1"], "val": val_metrics["macro_f1"]}, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        for cls_name in ["alert", "drowsy", "distracted"]:
            writer.add_scalars(
                f"f1_{cls_name}",
                {"train": train_metrics[f"f1_{cls_name}"], "val": val_metrics[f"f1_{cls_name}"]},
                epoch,
            )

        # Checkpointing
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "hparams": hparams,
                "label_map": LABEL_MAP,
                "label_names": LABEL_NAMES,
                "feature_cols": FEATURE_COLS,
                "model_config": {
                    "input_dim": NUM_FEATURES,
                    "hidden_dim": hidden_size,
                    "num_layers": num_layers,
                    "num_classes": len(LABEL_NAMES),
                    "dropout_lstm": dropout,
                },
            }
            best_path = models_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info("  ★ New best model saved (F1=%.4f) → %s", best_val_f1, best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", patience)
                break

    # Final summary
    total_time = time.time() - training_start
    writer.close()

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("  Total time:    %.1f minutes", total_time / 60)
    logger.info("  Best val F1:   %.4f", best_val_f1)
    logger.info("  Checkpoint:    %s", models_dir / "best_model.pt")
    logger.info("  TensorBoard:   tensorboard --logdir %s", tb_dir.parent)
    logger.info("=" * 70)
    logger.info("Next step: python scripts/04_evaluate.py --config %s", args.config)


if __name__ == "__main__":
    main()
