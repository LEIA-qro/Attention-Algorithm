"""sliding-window dataset over the feature parquet files, plus collate and dataloader helpers.

window label = majority vote of frame labels; string labels mapped to int class indices.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

logger = logging.getLogger(__name__)

# standard feature order, must match features.py FEATURE_NAMES
FEATURE_COLUMNS: List[str] = [
    "ear_left",
    "ear_right",
    "ear_avg",
    "mar",
    "perclos",
    "blink_rate",
    "blink_duration_avg",
    "yaw",
    "pitch",
    "roll",
    "gaze_yaw",
    "gaze_pitch",
    "gaze_stability",
    "head_pose_stability",
    "ear_velocity",
    "head_nod_count",
    "mouth_open_duration",
    "eyes_off_road_pct",
]

# Default class mapping
DEFAULT_CLASS_MAP: Dict[str, int] = {
    "Alert": 0,
    "Drowsy": 1,
    "Distracted": 2,
}


class DriverStateDataset(Dataset):
    """sliding-window dataset over parquet feature files; window label = majority vote.

    each file needs the FEATURE_COLUMNS plus a string "label" column.
    """

    def __init__(
        self,
        parquet_paths: Sequence[Union[str, Path]],
        seq_len: int = 90,
        stride: int = 15,
        class_map: Optional[Dict[str, int]] = None,
        feature_columns: Optional[List[str]] = None,
        label_column: str = "label",
        normalise: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride
        self.class_map = class_map or dict(DEFAULT_CLASS_MAP)
        self.feature_columns = feature_columns or list(FEATURE_COLUMNS)
        self.label_column = label_column
        self.normalise = normalise

        # load and concat parquet files
        dfs: List[pd.DataFrame] = []
        for p in parquet_paths:
            p = Path(p)
            if not p.exists():
                logger.warning("Parquet file not found, skipping: %s", p)
                continue
            df = pd.read_parquet(p)
            missing = set(self.feature_columns) - set(df.columns)
            if missing:
                raise ValueError(
                    f"Missing feature columns in {p.name}: {missing}"
                )
            if self.label_column not in df.columns:
                raise ValueError(
                    f"Missing label column '{self.label_column}' in {p.name}"
                )
            dfs.append(df)
            logger.debug("Loaded %s: %d rows", p.name, len(df))

        if not dfs:
            raise FileNotFoundError(
                "No valid parquet files loaded from the provided paths"
            )

        self._df = pd.concat(dfs, ignore_index=True)
        logger.info(
            "Total frames loaded: %d from %d files", len(self._df), len(dfs)
        )

        # map string labels to ints
        self._df["_label_int"] = self._df[self.label_column].map(self.class_map)
        unmapped = self._df["_label_int"].isna()
        if unmapped.any():
            bad = self._df.loc[unmapped, self.label_column].unique()
            logger.warning(
                "Dropping %d frames with unmapped labels: %s",
                unmapped.sum(),
                bad.tolist(),
            )
            self._df = self._df[~unmapped].reset_index(drop=True)
        self._df["_label_int"] = self._df["_label_int"].astype(int)

        self._features = self._df[self.feature_columns].values.astype(
            np.float32
        )
        self._labels = self._df["_label_int"].values.astype(np.int64)

        # z-score normalize
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        if self.normalise:
            self._mean = self._features.mean(axis=0)
            self._std = self._features.std(axis=0)
            # Avoid division by zero for constant features
            # avoid divide-by-zero on constant features
            self._std[self._std < 1e-8] = 1.0
            self._features = (self._features - self._mean) / self._std
            logger.info("Feature z-score normalisation applied")

        # each entry is (start, end)
        n_frames = len(self._features)
        self._windows: List[Tuple[int, int]] = []
        for start in range(0, n_frames - seq_len + 1, stride):
            self._windows.append((start, start + seq_len))
        logger.info(
            "Created %d sliding windows (seq_len=%d, stride=%d)",
            len(self._windows),
            seq_len,
            stride,
        )

        # majority label per window
        self._window_labels = np.empty(len(self._windows), dtype=np.int64)
        for i, (s, e) in enumerate(self._windows):
            counter = Counter(self._labels[s:e].tolist())
            self._window_labels[i] = counter.most_common(1)[0][0]

        unique, counts = np.unique(self._window_labels, return_counts=True)
        inv_map = {v: k for k, v in self.class_map.items()}
        dist_str = ", ".join(
            f"{inv_map.get(u, u)}: {c}" for u, c in zip(unique, counts)
        )
        logger.info("Window label distribution: %s", dist_str)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """features (seq_len, D) and majority label for window idx."""
        start, end = self._windows[idx]
        feats = torch.from_numpy(self._features[start:end].copy())
        label = torch.tensor(self._window_labels[idx], dtype=torch.long)
        return feats, label

    @property
    def num_features(self) -> int:
        return len(self.feature_columns)

    @property
    def num_classes(self) -> int:
        return len(self.class_map)

    @property
    def class_names(self) -> List[str]:
        # sorted by class index
        return [k for k, _ in sorted(self.class_map.items(), key=lambda kv: kv[1])]

    def get_normalisation_stats(self) -> Optional[Dict[str, np.ndarray]]:
        """mean/std used for normalisation, None if disabled."""
        if self._mean is None:
            return None
        return {"mean": self._mean, "std": self._std}

    def get_sample_weights(self) -> torch.Tensor:
        """per-sample weights inverse to class frequency, for WeightedRandomSampler."""
        unique, counts = np.unique(self._window_labels, return_counts=True)
        class_weights = 1.0 / counts.astype(np.float64)
        weight_map = {cls: w for cls, w in zip(unique, class_weights)}
        sample_weights = np.array(
            [weight_map[l] for l in self._window_labels], dtype=np.float64
        )
        return torch.from_numpy(sample_weights)


def dms_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """stack windows into (B, T, D); right-pad with zeros if lengths differ."""
    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]

    lengths = [f.size(0) for f in features_list]
    max_len = max(lengths)
    feat_dim = features_list[0].size(1)

    if all(l == max_len for l in lengths):
        features = torch.stack(features_list, dim=0)
    else:
        features = torch.zeros(len(batch), max_len, feat_dim)
        for i, f in enumerate(features_list):
            features[i, : f.size(0), :] = f

    labels = torch.stack(labels_list, dim=0)
    return features, labels


def create_dataloaders(
    features_dir: Union[str, Path],
    seq_len: int = 90,
    stride: int = 15,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    class_map: Optional[Dict[str, int]] = None,
    normalise: bool = True,
    balanced_sampling: bool = True,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """build train/val/test loaders from a dir of parquet files; test = 1 - train - val."""
    features_dir = Path(features_dir)
    parquet_files = sorted(features_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No .parquet files found in {features_dir}"
        )

    logger.info(
        "Found %d parquet files in %s", len(parquet_files), features_dir
    )

    dataset = DriverStateDataset(
        parquet_paths=parquet_files,
        seq_len=seq_len,
        stride=stride,
        class_map=class_map,
        normalise=normalise,
    )

    # shuffle and split
    n = len(dataset)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train : n_train + n_val].tolist()
    test_idx = indices[n_train + n_val :].tolist()

    logger.info(
        "Split: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx)
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_sampler = None
    shuffle_train = True
    if balanced_sampling:
        all_weights = dataset.get_sample_weights()
        train_weights = all_weights[train_idx]
        train_sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_idx),
            replacement=True,
        )
        shuffle_train = False  # sampler handles shuffling

    loaders = {
        "train": DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle_train if train_sampler is None else False,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=dms_collate_fn,
            drop_last=True,
        ),
        "val": DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=dms_collate_fn,
            drop_last=False,
        ),
        "test": DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=dms_collate_fn,
            drop_last=False,
        ),
    }

    logger.info(
        "DataLoaders created: batch_size=%d, workers=%d, balanced=%s",
        batch_size,
        num_workers,
        balanced_sampling,
    )
    return loaders
