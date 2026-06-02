#!/usr/bin/env python3

"""
00_setup_data.py — Extract and organize raw datasets for DMS training pipeline.

Handles:
  - DMD-Distraction (tar.gz archives)
  - DMD-Drowsiness  (tar.gz archives)
  - UTA-RLDD        (zip archives)

After extraction, walks the directory tree and prints summary statistics:
  # subjects, # face videos, # annotations per dataset.

Usage:
    python scripts/00_setup_data.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import yaml
from tqdm import tqdm


# Constants

MARKER_FILENAME = ".extraction_complete"

DATASET_EXTRACTORS = {
    "DMD-Distraction": "tar.gz",
    "DMD-Drowsiness": "tar.gz",
    "UTA-RLDD": "zip",
    "AUC-V2": "zip",
}

logger = logging.getLogger("dms.setup_data")


# Helpers

def _setup_logging(level: str = "INFO") -> None:
    """Configure root + dms loggers."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s"
    logging.basicConfig(level=numeric, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def _load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        logger.warning("Config file %s not found — using defaults.", path)
        return {}
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _is_extracted(dest_dir: Path) -> bool:
    """Check whether the extraction marker file exists."""
    return (dest_dir / MARKER_FILENAME).exists()


def _write_marker(dest_dir: Path) -> None:
    """Create the marker file after successful extraction."""
    marker = dest_dir / MARKER_FILENAME
    marker.write_text(
        f"Extraction completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        encoding="utf-8",
    )


# Extraction functions

def _extract_tar_gz(archive_path: Path, dest_dir: Path) -> int:
    """Extract a .tar.gz archive with progress bar.  Returns member count."""
    with tarfile.open(archive_path, "r:gz") as tf:
        members = tf.getmembers()
        for member in tqdm(
            members,
            desc=f"  ↳ {archive_path.name}",
            unit="file",
            leave=False,
        ):
            try:
                tf.extract(member, path=dest_dir, filter="data")
            except (tarfile.TarError, OSError) as exc:
                logger.warning("Skipping %s in %s: %s", member.name, archive_path.name, exc)
        return len(members)


def _extract_zip(archive_path: Path, dest_dir: Path) -> int:
    """Extract a .zip archive with progress bar.  Returns member count."""
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.infolist()
        for member in tqdm(
            members,
            desc=f"  ↳ {archive_path.name}",
            unit="file",
            leave=False,
        ):
            try:
                zf.extract(member, path=dest_dir)
            except (zipfile.BadZipFile, OSError) as exc:
                logger.warning("Skipping %s in %s: %s", member.filename, archive_path.name, exc)
        return len(members)


def _find_archives(directory: Path, fmt: str) -> list[Path]:
    """Return sorted list of archives in *directory* matching *fmt*."""
    if fmt == "tar.gz":
        return sorted(directory.glob("*.tar.gz")) + sorted(directory.glob("*.tgz"))
    elif fmt == "zip":
        return sorted(directory.glob("*.zip"))
    return []


def extract_dataset(name: str, raw_dir: Path, fmt: str) -> None:
    """Extract all archives for a single dataset."""
    dataset_dir = raw_dir / name
    if not dataset_dir.exists():
        logger.warning("Dataset directory not found: %s — skipping.", dataset_dir)
        return

    if _is_extracted(dataset_dir):
        logger.info("✓ %s already extracted (marker found). Skipping.", name)
        return

    archives = _find_archives(dataset_dir, fmt)
    if not archives:
        # Maybe already extracted without archives present
        logger.info("No %s archives found in %s — assuming already extracted.", fmt, dataset_dir)
        _write_marker(dataset_dir)
        return

    logger.info("Extracting %s (%d archive(s))…", name, len(archives))
    total_members = 0
    for archive in archives:
        if fmt == "tar.gz":
            total_members += _extract_tar_gz(archive, dataset_dir)
        else:
            total_members += _extract_zip(archive, dataset_dir)
        logger.info("  Extracted %s (%d items cumulative)", archive.name, total_members)

    _write_marker(dataset_dir)
    logger.info("✓ %s extraction complete (%d total items).", name, total_members)


# Statistics

def _scan_dmd(dataset_dir: Path, dataset_name: str) -> Dict[str, Any]:
    """Walk a DMD dataset directory and collect stats."""
    stats: Dict[str, Any] = {
        "name": dataset_name,
        "groups": set(),
        "subjects": set(),
        "sessions": set(),
        "face_videos": 0,
        "other_videos": 0,
        "annotations": 0,
        "total_files": 0,
    }

    if not dataset_dir.exists():
        return stats

    for root, _dirs, files in os.walk(dataset_dir):
        root_path = Path(root)
        rel = root_path.relative_to(dataset_dir)
        parts = rel.parts

        # Detect group / subject / session from path hierarchy
        for part in parts:
            if part.startswith("g") and len(part) == 2 and part[1].isalpha():
                stats["groups"].add(part)
            if part.startswith("s") and len(part) == 2 and part[1].isdigit():
                stats["sessions"].add(part)

        for fname in files:
            stats["total_files"] += 1
            fl = fname.lower()
            if fl.endswith((".mp4", ".avi", ".mkv")):
                if "_face" in fl:
                    stats["face_videos"] += 1
                    # Infer subject from parent path
                    # e.g. gA/1/s1/gA_1_s1_face.mp4  → subject = "gA_1"
                    name_parts = Path(fname).stem.split("_")
                    if len(name_parts) >= 2:
                        stats["subjects"].add(f"{name_parts[0]}_{name_parts[1]}")
                else:
                    stats["other_videos"] += 1
            elif fl.endswith(".json"):
                stats["annotations"] += 1

    # Convert sets for reporting
    stats["groups"] = sorted(stats["groups"])
    stats["subjects"] = sorted(stats["subjects"])
    stats["sessions"] = sorted(stats["sessions"])
    return stats


def _scan_uta_rldd(dataset_dir: Path) -> Dict[str, Any]:
    """Walk the UTA-RLDD dataset directory and collect stats."""
    stats: Dict[str, Any] = {
        "name": "UTA-RLDD",
        "subjects": set(),
        "videos": 0,
        "videos_by_level": defaultdict(int),
        "total_files": 0,
    }

    if not dataset_dir.exists():
        return stats

    for root, _dirs, files in os.walk(dataset_dir):
        root_path = Path(root)
        rel = root_path.relative_to(dataset_dir)
        parts = rel.parts

        # Subject folders are typically numeric (e.g., "1", "2", … "60")
        for part in parts:
            if part.isdigit():
                stats["subjects"].add(part)

        for fname in files:
            stats["total_files"] += 1
            fl = fname.lower()
            if fl.endswith((".mp4", ".avi", ".mkv", ".mov")):
                stats["videos"] += 1
                # Detect drowsiness level from filename
                stem = Path(fname).stem
                for level in ["10", "5", "0"]:
                    if level in stem:
                        stats["videos_by_level"][level] += 1
                        break

    stats["subjects"] = sorted(stats["subjects"], key=lambda x: int(x) if x.isdigit() else x)
    return stats


def print_summary(raw_dir: Path) -> None:
    """Print a comprehensive summary of all datasets."""
    logger.info("=" * 70)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 70)

    # DMD-Distraction
    dmd_dist = _scan_dmd(raw_dir / "DMD-Distraction", "DMD-Distraction")
    logger.info("")
    logger.info("┌─ DMD-Distraction ────────────────────────────────────")
    logger.info("│  Groups:       %d  %s", len(dmd_dist["groups"]), dmd_dist["groups"])
    logger.info("│  Subjects:     %d  %s", len(dmd_dist["subjects"]), dmd_dist["subjects"][:10])
    if len(dmd_dist["subjects"]) > 10:
        logger.info("│                … and %d more", len(dmd_dist["subjects"]) - 10)
    logger.info("│  Sessions:     %s", dmd_dist["sessions"])
    logger.info("│  Face videos:  %d", dmd_dist["face_videos"])
    logger.info("│  Other videos: %d", dmd_dist["other_videos"])
    logger.info("│  Annotations:  %d", dmd_dist["annotations"])
    logger.info("│  Total files:  %d", dmd_dist["total_files"])
    logger.info("└──────────────────────────────────────────────────────")

    # DMD-Drowsiness
    dmd_drow = _scan_dmd(raw_dir / "DMD-Drowsiness", "DMD-Drowsiness")
    logger.info("")
    logger.info("┌─ DMD-Drowsiness ─────────────────────────────────────")
    logger.info("│  Groups:       %d  %s", len(dmd_drow["groups"]), dmd_drow["groups"])
    logger.info("│  Subjects:     %d  %s", len(dmd_drow["subjects"]), dmd_drow["subjects"][:10])
    if len(dmd_drow["subjects"]) > 10:
        logger.info("│                … and %d more", len(dmd_drow["subjects"]) - 10)
    logger.info("│  Sessions:     %s", dmd_drow["sessions"])
    logger.info("│  Face videos:  %d", dmd_drow["face_videos"])
    logger.info("│  Other videos: %d", dmd_drow["other_videos"])
    logger.info("│  Annotations:  %d", dmd_drow["annotations"])
    logger.info("│  Total files:  %d", dmd_drow["total_files"])
    logger.info("└──────────────────────────────────────────────────────")

    # UTA-RLDD
    uta = _scan_uta_rldd(raw_dir / "UTA-RLDD")
    logger.info("")
    logger.info("┌─ UTA-RLDD ───────────────────────────────────────────")
    logger.info("│  Subjects:     %d", len(uta["subjects"]))
    logger.info("│  Videos:       %d", uta["videos"])
    for lvl in sorted(uta["videos_by_level"].keys()):
        label = {"0": "Alert", "5": "Drowsy (low)", "10": "Drowsy (high)"}.get(lvl, lvl)
        logger.info("│    Level %s (%s): %d", lvl, label, uta["videos_by_level"][lvl])
    logger.info("│  Total files:  %d", uta["total_files"])
    logger.info("└──────────────────────────────────────────────────────")

    logger.info("")
    logger.info("=" * 70)


# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and organize raw DMS datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root directory (default: from config or data/)",
    )
    args = parser.parse_args()

    _setup_logging(args.log_level)
    cfg = _load_config(args.config)

    # Resolve data root
    project_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root) if args.data_root else Path(
        cfg.get("data", {}).get("root", project_root / "data")
    )
    raw_dir = data_root / "raw"

    if not raw_dir.exists():
        logger.error("Raw data directory does not exist: %s", raw_dir)
        sys.exit(1)

    logger.info("Project root : %s", project_root)
    logger.info("Data root    : %s", data_root)
    logger.info("Raw dir      : %s", raw_dir)
    logger.info("")

    # Extract each dataset
    for dataset_name, fmt in DATASET_EXTRACTORS.items():
        try:
            extract_dataset(dataset_name, raw_dir, fmt)
        except Exception:
            logger.exception("Failed to extract %s — continuing with remaining datasets.", dataset_name)

    # Print summary
    print_summary(raw_dir)

    logger.info("Setup complete.  Next step: python scripts/01_extract_features.py --config %s", args.config)


if __name__ == "__main__":
    main()
