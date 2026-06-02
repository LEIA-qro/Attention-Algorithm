"""
Driver Monitoring System (DMS) — Core Training Pipeline
========================================================

This package provides the end-to-end pipeline for training a driver
state classification model (Alert / Drowsy / Distracted) using:

- MediaPipe Face Landmarker v2 for facial feature extraction
- CLAHE-based adaptive preprocessing
- An 18-dimensional temporal feature vector
- A Bi-LSTM with temporal attention (DriverStateNet, ~200K params)

Modules
-------
preprocessing : Frame-level image preprocessing (CLAHE, gamma, resize).
features      : Per-frame + temporal feature extraction via MediaPipe.
model         : DriverStateNet architecture definition.
dataset       : Sliding-window dataset and dataloader utilities.
utils         : Logging, checkpointing, metrics, plotting helpers.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Antigravity"

# Lazy imports to avoid heavy dependencies on ``import src``
# Users can do: ``from src import DriverStateNet, FeatureExtractor, ...``

__all__ = [
    # preprocessing
    "preprocess_frame",
    "apply_clahe",
    "adaptive_gamma_correction",
    # features
    "FeatureExtractor",
    # model
    "DriverStateNet",
    "TemporalAttention",
    # dataset
    "DriverStateDataset",
    "create_dataloaders",
    "dms_collate_fn",
    # utils
    "setup_logging",
    "MetricsTracker",
    "EarlyStopping",
    "save_checkpoint",
    "load_checkpoint",
    "seed_everything",
    "compute_class_weights",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_training_curves",
]


def __getattr__(name: str):
    """Lazy-load heavy submodules on first attribute access."""
    if name in ("preprocess_frame", "apply_clahe", "adaptive_gamma_correction"):
        from .preprocessing import preprocess_frame, apply_clahe, adaptive_gamma_correction
        return locals()[name]
    if name == "FeatureExtractor":
        from .features import FeatureExtractor
        return FeatureExtractor
    if name in ("DriverStateNet", "TemporalAttention"):
        from .model import DriverStateNet, TemporalAttention
        return locals()[name]
    if name in ("DriverStateDataset", "create_dataloaders", "dms_collate_fn"):
        from .dataset import DriverStateDataset, create_dataloaders, dms_collate_fn
        return locals()[name]
    if name in (
        "setup_logging", "MetricsTracker", "EarlyStopping",
        "save_checkpoint", "load_checkpoint", "seed_everything",
        "compute_class_weights", "plot_confusion_matrix",
        "plot_roc_curves", "plot_training_curves",
    ):
        from . import utils
        return getattr(utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
