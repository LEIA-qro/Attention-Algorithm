#!/usr/bin/env python3

"""build subject-disjoint train/val/test splits from feature parquets.

groups parquets by (dataset, subject_id), then stratified subject-level
split 70/15/15 with every dataset represented in each split.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

logger = logging.getLogger("dms.build_splits")

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
LABEL_NAMES = ["Alert", "Drowsy", "Distracted"]


def _setup_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s"
    logging.basicConfig(level=numeric, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        logger.warning("config %s not found, using defaults", path)
        return {}
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_inventory(features_dir: Path) -> pd.DataFrame:
    """build a per-file inventory dataframe from feature parquets"""
    records = []
    parquet_files = sorted(features_dir.rglob("*.parquet"))

    logger.info("scanning %d parquet files in %s", len(parquet_files), features_dir)

    for pf in tqdm(parquet_files, desc="building inventory", unit="file"):
        try:
            df = pd.read_parquet(pf, columns=["dataset", "subject_id", "video_id", "label"])
        except Exception as exc:
            logger.warning("cannot read %s: %s, skipping", pf, exc)
            continue

        if df.empty:
            continue

        n_frames = len(df)
        label_counts = df["label"].value_counts()
        n_alert = int(label_counts.get("Alert", 0))
        n_drowsy = int(label_counts.get("Drowsy", 0))
        n_distracted = int(label_counts.get("Distracted", 0))
        dominant = label_counts.idxmax()

        records.append({
            "feature_file": str(pf),
            "dataset": df["dataset"].iloc[0],
            "subject_id": df["subject_id"].iloc[0],
            "video_id": df["video_id"].iloc[0],
            "n_frames": n_frames,
            "n_alert": n_alert,
            "n_drowsy": n_drowsy,
            "n_distracted": n_distracted,
            "dominant_label": dominant,
        })

    inventory = pd.DataFrame(records)
    logger.info("Inventory: %d files, %d subjects across %d datasets",
                len(inventory),
                inventory["subject_id"].nunique() if not inventory.empty else 0,
                inventory["dataset"].nunique() if not inventory.empty else 0)
    return inventory


def _compute_subject_profile(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """aggregate per-subject stats for stratification"""
    grouped = inventory.groupby(["dataset", "subject_id"]).agg(
        n_files=("feature_file", "count"),
        total_frames=("n_frames", "sum"),
        n_alert=("n_alert", "sum"),
        n_drowsy=("n_drowsy", "sum"),
        n_distracted=("n_distracted", "sum"),
    ).reset_index()

    # dominant class per subject
    grouped["dominant_class"] = grouped[["n_alert", "n_drowsy", "n_distracted"]].idxmax(axis=1)
    grouped["dominant_class"] = grouped["dominant_class"].map({
        "n_alert": "Alert",
        "n_drowsy": "Drowsy",
        "n_distracted": "Distracted",
    })

    return grouped


def stratified_subject_split(
    subject_profiles: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Tuple[str, str]]]:
    """split subjects into train/val/test, stratified by dataset x dominant_class.

    returns dict mapping split name -> list of (dataset, subject_id) tuples.
    """
    rng = np.random.RandomState(seed)
    splits: Dict[str, List[Tuple[str, str]]] = {"train": [], "val": [], "test": []}

    strata = subject_profiles.groupby(["dataset", "dominant_class"])

    for (ds, dc), group_df in strata:
        subjects = group_df[["dataset", "subject_id"]].values.tolist()
        rng.shuffle(subjects)

        n = len(subjects)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1 if n > 1 else 0, int(round(n * val_ratio)))
        n_test = n - n_train - n_val

        # keep >=1 per split where n allows
        if n_test < 0:
            n_val = max(0, n - n_train)
            n_test = 0
        if n >= 3 and n_test == 0:
            n_test = 1
            n_train = n - n_val - n_test

        train_subjs = subjects[:n_train]
        val_subjs = subjects[n_train:n_train + n_val]
        test_subjs = subjects[n_train + n_val:]

        for s in train_subjs:
            splits["train"].append(tuple(s))
        for s in val_subjs:
            splits["val"].append(tuple(s))
        for s in test_subjs:
            splits["test"].append(tuple(s))

    return splits


def create_split_csvs(
    inventory: pd.DataFrame,
    splits: Dict[str, List[Tuple[str, str]]],
    splits_dir: Path,
) -> None:
    """write train.csv, val.csv, test.csv to splits_dir"""
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split_name, subject_keys in splits.items():
        key_set = set(subject_keys)
        mask = inventory.apply(
            lambda row: (row["dataset"], row["subject_id"]) in key_set, axis=1
        )
        split_df = inventory[mask].copy()
        split_df["split"] = split_name

        out_path = splits_dir / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        logger.info("saved %s: %d files, %d subjects -> %s",
                     split_name, len(split_df), split_df["subject_id"].nunique(), out_path)


def print_split_stats(
    inventory: pd.DataFrame,
    splits: Dict[str, List[Tuple[str, str]]],
) -> None:
    """log per-split statistics"""
    logger.info("")
    logger.info("=" * 72)
    logger.info("SPLIT STATISTICS")
    logger.info("=" * 72)

    for split_name in ["train", "val", "test"]:
        subject_keys = set(splits[split_name])
        mask = inventory.apply(
            lambda row, ks=subject_keys: (row["dataset"], row["subject_id"]) in ks,
            axis=1,
        )
        split_df = inventory[mask]

        n_files = len(split_df)
        n_subjects = split_df["subject_id"].nunique()
        total_frames = int(split_df["n_frames"].sum())
        n_alert = int(split_df["n_alert"].sum())
        n_drowsy = int(split_df["n_drowsy"].sum())
        n_distracted = int(split_df["n_distracted"].sum())
        total_labelled = n_alert + n_drowsy + n_distracted

        logger.info("")
        logger.info("[ %s ]", split_name.upper())
        logger.info("  Subjects:       %d", n_subjects)
        logger.info("  Feature files:  %d", n_files)
        logger.info("  Total frames:   %s", f"{total_frames:,}")
        if total_labelled > 0:
            logger.info("  Alert:          %s  (%.1f%%)", f"{n_alert:>8,}", 100 * n_alert / total_labelled)
            logger.info("  Drowsy:         %s  (%.1f%%)", f"{n_drowsy:>8,}", 100 * n_drowsy / total_labelled)
            logger.info("  Distracted:     %s  (%.1f%%)", f"{n_distracted:>8,}", 100 * n_distracted / total_labelled)

        for ds in sorted(split_df["dataset"].unique()):
            ds_mask = split_df["dataset"] == ds
            ds_subj = split_df[ds_mask]["subject_id"].nunique()
            ds_files = int(ds_mask.sum())
            ds_frames = int(split_df[ds_mask]["n_frames"].sum())
            logger.info("  %-20s  %d subj, %d files, %s frames",
                         ds, ds_subj, ds_files, f"{ds_frames:,}")

    logger.info("")

    # subject overlap = leakage
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set

    if overlap_tv or overlap_tt or overlap_vt:
        logger.error("DATA LEAKAGE DETECTED")
        if overlap_tv:
            logger.error("  Train and Val:  %s", overlap_tv)
        if overlap_tt:
            logger.error("  Train and Test: %s", overlap_tt)
        if overlap_vt:
            logger.error("  Val and Test:   %s", overlap_vt)
    else:
        logger.info("no subject overlap between splits - data leakage check passed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create subject-disjoint train/val/test splits.",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Config YAML path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--data-root", type=str, default=None)
    args = parser.parse_args()

    _setup_logging(args.log_level)
    cfg = _load_config(args.config)

    project_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root) if args.data_root else Path(
        cfg.get("data", {}).get("root", project_root / "data")
    )
    features_dir = data_root / "features"
    splits_dir = data_root / "splits"

    if not features_dir.exists():
        logger.error("Features directory not found: %s", features_dir)
        logger.error("Run 01_extract_features.py first.")
        sys.exit(1)

    inventory = build_inventory(features_dir)
    if inventory.empty:
        logger.error("No parquet files found in %s", features_dir)
        sys.exit(1)

    subject_profiles = _compute_subject_profile(inventory)
    logger.info("Subject profiles:\n%s", subject_profiles.to_string(index=False))

    splits = stratified_subject_split(
        subject_profiles,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    create_split_csvs(inventory, splits, splits_dir)
    print_split_stats(inventory, splits)


if __name__ == "__main__":
    main()
