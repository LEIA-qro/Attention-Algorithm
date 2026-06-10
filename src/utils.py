"""Shared training helpers: logging setup, seeding, checkpointing, metrics tracking, and plots."""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, safe for headless servers.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def setup_logging(
    log_dir: Union[str, Path] = "logs",
    level: str = "INFO",
    console: bool = True,
    file: bool = True,
    log_filename: str = "training.log",
) -> logging.Logger:
    """Configure the root logger with optional file and console handlers."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers so re-init doesn't duplicate them.
    root.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    if file:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            log_dir / log_filename, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)

    root.info("Logging initialised: level=%s, dir=%s", level, log_dir)
    return root


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch and force deterministic CuDNN (may reduce perf)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d (deterministic mode)", seed)


def save_checkpoint(
    state: Dict[str, Any],
    filepath: Union[str, Path],
) -> Path:
    """Save a training checkpoint and return its absolute path."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)
    size_mb = filepath.stat().st_size / (1024 * 1024)
    logger.info("Checkpoint saved: %s (%.1f MB)", filepath, size_mb)
    return filepath.resolve()


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Load a training checkpoint and restore model, optimizer, and scheduler state."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        "Model weights loaded from %s (epoch %d)",
        filepath,
        checkpoint.get("epoch", -1),
    )

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state restored")

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Scheduler state restored")

    return checkpoint


@dataclass
class MetricsTracker:
    """Tracks per-epoch loss, accuracy, precision, recall, and macro-F1 for train and val splits."""

    num_classes: int = 3
    class_names: List[str] = field(
        default_factory=lambda: ["Alert", "Drowsy", "Distracted"]
    )

    _history: Dict[str, Dict[str, List[float]]] = field(
        default_factory=lambda: {
            "train": {
                "loss": [],
                "accuracy": [],
                "precision_macro": [],
                "recall_macro": [],
                "f1_macro": [],
            },
            "val": {
                "loss": [],
                "accuracy": [],
                "precision_macro": [],
                "recall_macro": [],
                "f1_macro": [],
            },
        },
        init=False,
    )
    _f1_per_class: Dict[str, List[np.ndarray]] = field(
        default_factory=lambda: {"train": [], "val": []},
        init=False,
    )

    def update(
        self,
        split: str,
        epoch: int,
        loss: float,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        y_score: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Record metrics for one epoch and return them."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        rec = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        f1_mac = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        f1_per = f1_score(
            y_true,
            y_pred,
            average=None,
            labels=list(range(self.num_classes)),
            zero_division=0,
        )

        self._history[split]["loss"].append(loss)
        self._history[split]["accuracy"].append(acc)
        self._history[split]["precision_macro"].append(prec)
        self._history[split]["recall_macro"].append(rec)
        self._history[split]["f1_macro"].append(f1_mac)
        self._f1_per_class[split].append(f1_per)

        metrics = {
            "loss": loss,
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1_mac,
        }
        for i, name in enumerate(self.class_names):
            metrics[f"f1_{name}"] = float(f1_per[i])

        logger.info(
            "Epoch %d [%s] loss=%.4f acc=%.4f F1=%.4f",
            epoch,
            split,
            loss,
            acc,
            f1_mac,
        )
        return metrics

    def best_val_f1(self) -> float:
        """Return the best validation macro-F1 seen so far."""
        vals = self._history["val"]["f1_macro"]
        return max(vals) if vals else 0.0

    def best_val_epoch(self) -> int:
        """Return the epoch index with the best validation F1."""
        vals = self._history["val"]["f1_macro"]
        return int(np.argmax(vals)) if vals else 0

    def get_history(self, split: str) -> Dict[str, List[float]]:
        """Return the full metric history for the given split."""
        return dict(self._history[split])


def compute_class_weights(
    labels: Union[Sequence[int], np.ndarray, torch.Tensor],
    num_classes: int = 3,
    method: str = "inverse_freq",
) -> torch.Tensor:
    """Class weights for imbalanced data: "inverse_freq" (1/count) or "effective_num" (Cui et al. 2019)."""
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    labels = np.asarray(labels, dtype=np.int64)

    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # floor at 1 so empty classes don't divide by zero.

    if method == "inverse_freq":
        weights = 1.0 / counts
    elif method == "effective_num":
        beta = 0.9999
        effective = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective
    else:
        raise ValueError(f"Unknown method: {method}")

    weights = weights / weights.sum() * num_classes
    w = torch.tensor(weights, dtype=torch.float32)
    logger.info("Class weights (%s): %s", method, w.tolist())
    return w


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    normalise: bool = True,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot, and optionally save, a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1)
        cm_display = cm.astype(np.float64) / row_sums
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)

    return fig


def plot_roc_curves(
    y_true: Sequence[int],
    y_score: np.ndarray,
    class_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot per-class ROC curves with AUC scores."""
    y_true = np.asarray(y_true)
    num_classes = len(class_names)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set2(np.linspace(0, 1, num_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        binary_true = (y_true == i).astype(int)
        if y_score.ndim == 2:
            scores = y_score[:, i]
        else:
            scores = y_score

        try:
            fpr, tpr, _ = roc_curve(binary_true, scores)
            auc_val = roc_auc_score(binary_true, scores)
            ax.plot(
                fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {auc_val:.3f})",
            )
        except ValueError:
            logger.warning("ROC curve could not be computed for class '%s'", name)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ROC curves saved to %s", save_path)

    return fig


def plot_training_curves(
    tracker: MetricsTracker,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Plot training and validation loss, accuracy, and F1 over epochs."""
    train_hist = tracker.get_history("train")
    val_hist = tracker.get_history("val")
    epochs = range(1, len(train_hist["loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_hist["loss"], "b-", label="Train")
    ax.plot(epochs, val_hist["loss"], "r-", label="Val")
    ax.set_title("Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, train_hist["accuracy"], "b-", label="Train")
    ax.plot(epochs, val_hist["accuracy"], "r-", label="Val")
    ax.set_title("Accuracy", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Macro F1
    ax = axes[1, 0]
    ax.plot(epochs, train_hist["f1_macro"], "b-", label="Train")
    ax.plot(epochs, val_hist["f1_macro"], "r-", label="Val")
    ax.set_title("Macro F1", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision & Recall (val only)
    ax = axes[1, 1]
    ax.plot(
        epochs,
        val_hist["precision_macro"],
        "g-",
        label="Val Precision",
    )
    ax.plot(
        epochs,
        val_hist["recall_macro"],
        "m-",
        label="Val Recall",
    )
    ax.set_title("Val Precision & Recall", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Training Curves", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Training curves saved to %s", save_path)

    return fig


class EarlyStopping:
    """Stop training when the validation metric stops improving for `patience` epochs (mode "max" or "min")."""

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.001,
        mode: str = "max",
        verbose: bool = True,
    ) -> None:
        assert mode in ("min", "max"), f"mode must be 'min' or 'max', got '{mode}'"
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.should_stop: bool = False

    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == "max":
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta

    def __call__(self, metric: float, epoch: int = 0) -> bool:
        """Return True if training should stop."""
        if self.best_score is None:
            self.best_score = metric
            self.best_epoch = epoch
            return False

        if self._is_improvement(metric, self.best_score):
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                logger.info(
                    "EarlyStopping: improvement to %.5f at epoch %d",
                    metric,
                    epoch,
                )
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    "EarlyStopping: no improvement (%d/%d), best=%.5f @ epoch %d",
                    self.counter,
                    self.patience,
                    self.best_score,
                    self.best_epoch,
                )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "EarlyStopping triggered after %d epochs without improvement",
                    self.patience,
                )
                return True

        return False

    def reset(self) -> None:
        """Reset internal state."""
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.should_stop = False
