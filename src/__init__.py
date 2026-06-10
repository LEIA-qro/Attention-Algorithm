"""DMS core training pipeline: preprocessing, feature extraction, the DriverStateNet model, dataset utilities, and training helpers."""

__version__ = "0.1.0"
__author__ = "Antigravity"

# Lazy-load submodules so `import src` stays cheap.

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
