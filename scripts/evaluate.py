#!/usr/bin/env python3

"""Evaluate a trained DriverStateNet on the test split.

Writes metrics, confusion matrix, ROC curves, per-dataset breakdown, and
permutation feature importance to models/evaluation/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import yaml

# put project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import DriverStateNet  # noqa: E402

logger = logging.getLogger("dms.evaluate")

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


def _setup_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s"
    logging.basicConfig(level=numeric, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# carries dataset/subject metadata so we can break down by source dataset
class EvalSlidingWindowDataset(Dataset):
    """sliding-window dataset that also tracks dataset/subject metadata per window"""

    def __init__(
        self,
        split_csv: Path,
        seq_len: int = 90,
        stride: int = 15,
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.windows: List[Tuple[np.ndarray, int]] = []
        self.metadata: List[Dict[str, str]] = []

        split_df = pd.read_csv(split_csv)
        feature_files = split_df["feature_file"].tolist()

        for fpath in tqdm(feature_files, desc=f"Loading {split_csv.stem}", unit="file", leave=False):
            fpath = Path(fpath)
            if not fpath.exists():
                continue

            try:
                df = pd.read_parquet(fpath)
            except Exception:
                continue

            missing = [c for c in FEATURE_COLS if c not in df.columns]
            if missing:
                continue

            features = df[FEATURE_COLS].values.astype(np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            labels_str = df["label"].values
            labels_int = np.array([LABEL_MAP.get(l, 0) for l in labels_str], dtype=np.int64)

            dataset_name = df["dataset"].iloc[0] if "dataset" in df.columns else "unknown"
            subject_id = df["subject_id"].iloc[0] if "subject_id" in df.columns else "unknown"

            n = len(features)
            if n < self.seq_len:
                pad_len = self.seq_len - n
                features = np.pad(features, ((0, pad_len), (0, 0)), mode="edge")
                labels_int = np.pad(labels_int, (0, pad_len), mode="edge")
                n = self.seq_len

            for start in range(0, n - self.seq_len + 1, self.stride):
                end = start + self.seq_len
                window_features = features[start:end]
                window_labels = labels_int[start:end]
                counts = Counter(window_labels)
                label = counts.most_common(1)[0][0]

                self.windows.append((window_features, label))
                self.metadata.append({"dataset": dataset_name, "subject_id": subject_id})

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features, label = self.windows[idx]
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """returns (all_labels, all_preds, all_probs)"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for features, labels in tqdm(loader, desc="Evaluating", unit="batch"):
        features = features.to(device, non_blocking=True)
        logits = model(features)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix_heatmap(
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: Path,
) -> None:
    """confusion matrix heatmap, raw counts and row-normalized"""
    cm = confusion_matrix(labels, preds, labels=list(range(len(LABEL_NAMES))))
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Confusion Matrix (counts)")

    # normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        ax=axes[1],
    )
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix (normalized)")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved confusion matrix -> %s", save_path)


def plot_roc_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: Path,
) -> None:
    """per-class ROC curves, one-vs-rest"""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2ecc71", "#e67e22", "#e74c3c"]

    for cls_idx, (cls_name, color) in enumerate(zip(LABEL_NAMES, colors)):
        binary_labels = (labels == cls_idx).astype(int)
        cls_probs = probs[:, cls_idx]

        if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
            logger.warning("class %s has no positive/negative samples, skipping ROC", cls_name)
            continue

        fpr, tpr, _ = roc_curve(binary_labels, cls_probs)
        auc_val = roc_auc_score(binary_labels, cls_probs)

        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{cls_name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved ROC curves -> %s", save_path)


def plot_per_dataset_breakdown(
    labels: np.ndarray,
    preds: np.ndarray,
    metadata: List[Dict[str, str]],
    save_path: Path,
) -> Dict[str, Dict[str, float]]:
    """per-dataset metrics breakdown, returns the metrics dict"""
    dataset_labels: Dict[str, List[int]] = defaultdict(list)
    dataset_preds: Dict[str, List[int]] = defaultdict(list)

    for i, meta in enumerate(metadata):
        ds = meta["dataset"]
        dataset_labels[ds].append(labels[i])
        dataset_preds[ds].append(preds[i])

    results: Dict[str, Dict[str, float]] = {}
    datasets = sorted(dataset_labels.keys())

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for idx, ds in enumerate(datasets):
        ds_labels = np.array(dataset_labels[ds])
        ds_preds = np.array(dataset_preds[ds])

        acc = accuracy_score(ds_labels, ds_preds)
        macro_f1 = f1_score(ds_labels, ds_preds, average="macro", zero_division=0)
        prec, rec, f1, _ = precision_recall_fscore_support(
            ds_labels, ds_preds, labels=list(range(len(LABEL_NAMES))),
            average=None, zero_division=0,
        )

        results[ds] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "n_samples": len(ds_labels),
        }
        for ci, cn in enumerate(LABEL_NAMES):
            results[ds][f"f1_{cn.lower()}"] = float(f1[ci])

        cm = confusion_matrix(ds_labels, ds_preds, labels=list(range(len(LABEL_NAMES))))
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

        sns.heatmap(
            cm_norm, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            ax=axes[idx],
        )
        axes[idx].set_title(f"{ds}\nacc={acc:.3f}  F1={macro_f1:.3f}")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved per-dataset breakdown -> %s", save_path)

    return results


@torch.no_grad()
def permutation_feature_importance(
    model: nn.Module,
    dataset: EvalSlidingWindowDataset,
    device: torch.device,
    n_repeats: int = 5,
    save_path: Optional[Path] = None,
) -> Dict[str, float]:
    """permutation importance: macro-F1 drop when each feature is shuffled"""
    model.eval()

    # baseline preds
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    all_labels = []
    all_preds = []
    for feats, lbls in loader:
        feats = feats.to(device)
        preds = model(feats).argmax(dim=1).cpu().numpy()
        all_labels.extend(lbls.numpy())
        all_preds.extend(preds)

    baseline_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    logger.info("Baseline macro F1: %.4f", baseline_f1)

    importances: Dict[str, float] = {}

    for feat_idx, feat_name in enumerate(tqdm(FEATURE_COLS, desc="Permutation importance")):
        drops = []
        for _ in range(n_repeats):
            permuted_preds = []
            for i in range(len(dataset)):
                feats_orig, _ = dataset[i]
                feats_perm = feats_orig.clone()
                # shuffle this feature along the sequence dim
                perm_idx = torch.randperm(feats_perm.size(0))
                feats_perm[:, feat_idx] = feats_perm[perm_idx, feat_idx]
                permuted_preds.append(feats_perm)

            permuted_stack = torch.stack(permuted_preds).to(device)
            perm_preds_list = []
            for batch_start in range(0, len(permuted_stack), 512):
                batch = permuted_stack[batch_start:batch_start + 512]
                preds = model(batch).argmax(dim=1).cpu().numpy()
                perm_preds_list.extend(preds)

            perm_f1 = f1_score(all_labels, perm_preds_list, average="macro", zero_division=0)
            drops.append(baseline_f1 - perm_f1)

        importances[feat_name] = float(np.mean(drops))

    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    if save_path:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(sorted_imp.keys())
        values = list(sorted_imp.values())
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]

        ax.barh(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("delta F1 (drop when permuted)")
        ax.set_title("Feature Importance (Permutation)")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("saved feature importance -> %s", save_path)

    return sorted_imp


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DriverStateNet on test set.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint (default: models/best_model.pt)")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--skip-permutation", action="store_true", help="Skip slow permutation importance")
    parser.add_argument("--perm-repeats", type=int, default=5)
    args = parser.parse_args()

    _setup_logging(args.log_level)
    cfg = _load_config(args.config)

    project_root = _PROJECT_ROOT
    data_root = Path(args.data_root) if args.data_root else Path(
        cfg.get("data", {}).get("root", project_root / "data")
    )
    splits_dir = data_root / "splits"
    models_dir = project_root / "models"
    eval_dir = models_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    test_csv = splits_dir / "test.csv"
    if not test_csv.exists():
        logger.error("Test split not found: %s", test_csv)
        sys.exit(1)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else models_dir / "best_model.pt"
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = ckpt.get("model_config", {})

    model = DriverStateNet(
        input_dim=model_config.get("input_dim", 18),
        hidden_dim=model_config.get("hidden_dim", 64),
        num_layers=model_config.get("num_layers", 2),
        num_classes=model_config.get("num_classes", 3),
        dropout_lstm=model_config.get("dropout_lstm", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info("Loaded model from %s (epoch %d, val F1=%.4f)",
                ckpt_path, ckpt.get("epoch", -1), ckpt.get("best_val_f1", 0))

    seq_len = ckpt.get("hparams", {}).get("seq_len", 90)
    stride = ckpt.get("hparams", {}).get("stride", 15)

    test_dataset = EvalSlidingWindowDataset(test_csv, seq_len=seq_len, stride=stride)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info("Test set: %d windows", len(test_dataset))

    if len(test_dataset) == 0:
        logger.error("Test dataset is empty!")
        sys.exit(1)

    labels, preds, probs = run_inference(model, test_loader, device)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    report = classification_report(
        labels, preds,
        target_names=LABEL_NAMES,
        labels=list(range(len(LABEL_NAMES))),
        zero_division=0,
        digits=4,
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 70)
    logger.info("Accuracy:    %.4f", acc)
    logger.info("Macro F1:    %.4f", macro_f1)
    logger.info("Weighted F1: %.4f", weighted_f1)
    logger.info("")
    logger.info("Classification Report:\n%s", report)

    prec, rec, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(LABEL_NAMES))), zero_division=0
    )

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "n_test_windows": len(labels),
        "per_class": {},
    }
    for i, name in enumerate(LABEL_NAMES):
        metrics["per_class"][name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    metrics_path = eval_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("saved metrics -> %s", metrics_path)

    plot_confusion_matrix_heatmap(labels, preds, eval_dir / "confusion_matrix.png")
    plot_roc_curves(labels, probs, eval_dir / "roc_curves.png")

    ds_results = plot_per_dataset_breakdown(
        labels, preds, test_dataset.metadata, eval_dir / "per_dataset_breakdown.png"
    )

    logger.info("")
    logger.info("Per-dataset results:")
    for ds, res in ds_results.items():
        logger.info("  %s: acc=%.3f F1=%.3f (n=%d)", ds, res["accuracy"], res["macro_f1"], res["n_samples"])

    ds_metrics_path = eval_dir / "per_dataset_metrics.json"
    with open(ds_metrics_path, "w", encoding="utf-8") as fh:
        json.dump(ds_results, fh, indent=2)

    if not args.skip_permutation:
        logger.info("")
        logger.info("computing permutation feature importance (this may take a while)")
        importances = permutation_feature_importance(
            model, test_dataset, device,
            n_repeats=args.perm_repeats,
            save_path=eval_dir / "feature_importance.png",
        )
        logger.info("Top-5 features by importance:")
        for i, (name, imp) in enumerate(list(importances.items())[:5]):
            logger.info("  %d. %s: delta F1 = %.4f", i + 1, name, imp)

        imp_path = eval_dir / "feature_importance.json"
        with open(imp_path, "w", encoding="utf-8") as fh:
            json.dump(importances, fh, indent=2)
    else:
        logger.info("Skipping permutation importance (--skip-permutation)")

    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("  All plots and metrics saved to: %s", eval_dir)
    logger.info("=" * 70)
    logger.info("Next step: python scripts/export_onnx.py --config %s", args.config)


if __name__ == "__main__":
    main()
