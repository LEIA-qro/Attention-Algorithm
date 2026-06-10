#!/usr/bin/env python3

"""Extract per-frame feature vectors from face videos and save as parquet."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# put project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features import FeatureExtractor  # noqa: E402
from src.preprocessing import preprocess_frame  # noqa: E402

logger = logging.getLogger("dms.extract_features")

# mediapipe model url and local path
MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
MEDIAPIPE_MODEL_FILENAME = "face_landmarker.task"

# DMD label mapping
DMD_DISTRACTION_LABEL_MAP: Dict[str, str] = {
    "Safe Driving": "Alert",
    "Stand Still-Waiting": "Alert",
    "Texting (Right)": "Distracted",
    "Texting (Left)": "Distracted",
    "Phonecall (Right)": "Distracted",
    "Phonecall (Left)": "Distracted",
    "Radio": "Distracted",
    "Drinking": "Distracted",
    "Reach Side": "Distracted",
    "Hair and Makeup": "Distracted",
    "Talking to passenger": "Distracted",
    "Reach Backseat": "Distracted",
    "Change Gear": "Alert",
    "Unclassified": "Alert",
}

DMD_DROWSINESS_CLOSED_ACTIONS = {"Closing", "Closed", "Yawning", "Yawning-Closing",
                                  "Yawning-Closed", "Microsleep"}

UTA_RLDD_LABEL_MAP = {"0": "Alert", "5": "Drowsy", "10": "Drowsy"}

# auc v2: 10 classes -> 3-class mapping
# 0=drive safe -> alert, else -> distracted

AUC_V2_LABEL_MAP: Dict[int, str] = {
    0: "Alert",       # Drive Safe
    1: "Distracted",  # Text Left
    2: "Distracted",  # Talk Left
    3: "Distracted",  # Text Right
    4: "Distracted",  # Talk Right
    5: "Distracted",  # Adjust Radio
    6: "Distracted",  # Drink
    7: "Distracted",  # Hair & Makeup
    8: "Distracted",  # Reach Behind
    9: "Distracted",  # Talk Passenger
}

AUC_V2_FOLDER_LABEL: Dict[str, str] = {
    "Drive Safe": "Alert",
    "Text Left": "Distracted",
    "Talk Left": "Distracted",
    "Text Right": "Distracted",
    "Talk Right": "Distracted",
    "Adjust Radio": "Distracted",
    "Drink": "Distracted",
    "Hair & Makeup": "Distracted",
    "Reach Behind": "Distracted",
    "Talk Passenger": "Distracted",
    "c0": "Alert",
    "c1": "Distracted",
    "c2": "Distracted",
    "c3": "Distracted",
    "c4": "Distracted",
    "c5": "Distracted",
    "c6": "Distracted",
    "c7": "Distracted",
    "c8": "Distracted",
    "c9": "Distracted",
}

# Feature column order
FEATURE_COLUMNS = [
    "frame_idx", "timestamp_s",
    "ear_left", "ear_right", "ear_avg", "mar",
    "perclos", "blink_rate", "blink_duration_avg",
    "yaw", "pitch", "roll",
    "gaze_yaw", "gaze_pitch", "gaze_stability",
    "head_pose_stability", "ear_velocity",
    "head_nod_count", "mouth_open_duration",
    "eyes_off_road_pct",
    "label",
]


# Setup helpers

def _setup_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s"
    logging.basicConfig(level=numeric, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        logger.warning("Config %s not found, using defaults", path)
        return {}
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def ensure_mediapipe_model(models_dir: Path) -> Path:
    """Download the MediaPipe Face Landmarker .task file if not present."""
    model_path = models_dir / MEDIAPIPE_MODEL_FILENAME
    if model_path.exists():
        logger.info("MediaPipe model already present: %s", model_path)
        return model_path

    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MediaPipe Face Landmarker model...")
    try:
        urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, str(model_path))
        logger.info("downloaded to %s", model_path)
    except Exception as exc:
        logger.error("Failed to download MediaPipe model: %s", exc)
        logger.error("Please download manually from %s", MEDIAPIPE_MODEL_URL)
        sys.exit(1)
    return model_path


# DMD annotation parsing

def parse_dmd_distraction_labels(json_path: Path, total_frames: int) -> List[str]:
    """Parse OpenLABEL JSON for a distraction session and return per-frame labels."""
    labels = ["Alert"] * total_frames  # default

    if not json_path.exists():
        logger.warning("Annotation file not found: %s, all frames labelled Alert", json_path)
        return labels

    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cannot parse %s: %s, all frames labelled Alert", json_path, exc)
        return labels

    # OpenLABEL structure: data -> openlabel -> actions
    openlabel = data.get("openlabel", data)
    actions = openlabel.get("actions", {})

    for _action_id, action_info in actions.items():
        action_type = action_info.get("type", "")
        mapped_label = DMD_DISTRACTION_LABEL_MAP.get(action_type)
        if mapped_label is None:
            # Try partial matching
            for key, val in DMD_DISTRACTION_LABEL_MAP.items():
                if key.lower() in action_type.lower():
                    mapped_label = val
                    break
            if mapped_label is None:
                mapped_label = "Alert"

        frame_intervals = action_info.get("frame_intervals", [])
        for interval in frame_intervals:
            start = int(interval.get("frame_start", interval.get("start", 0)))
            end = int(interval.get("frame_end", interval.get("end", start)))
            for f in range(start, min(end + 1, total_frames)):
                # Distracted overrides Alert but not vice-versa
                if mapped_label == "Distracted" or labels[f] == "Alert":
                    labels[f] = mapped_label

    return labels


def parse_dmd_drowsiness_labels(json_path: Path, total_frames: int) -> List[str]:
    """Parse OpenLABEL JSON for a drowsiness session (s5)."""
    labels = ["Alert"] * total_frames

    if not json_path.exists():
        logger.warning("Drowsiness annotation not found: %s, all frames labelled Alert", json_path)
        return labels

    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cannot parse %s: %s", json_path, exc)
        return labels

    openlabel = data.get("openlabel", data)
    actions = openlabel.get("actions", {})

    for _action_id, action_info in actions.items():
        action_type = action_info.get("type", "")
        is_drowsy_action = any(kw in action_type for kw in DMD_DROWSINESS_CLOSED_ACTIONS)

        if is_drowsy_action:
            frame_intervals = action_info.get("frame_intervals", [])
            for interval in frame_intervals:
                start = int(interval.get("frame_start", interval.get("start", 0)))
                end = int(interval.get("frame_end", interval.get("end", start)))
                for f in range(start, min(end + 1, total_frames)):
                    labels[f] = "Drowsy"

    return labels


# Video discovery

def discover_dmd_videos(dataset_dir: Path, dataset_name: str) -> List[Dict[str, Any]]:
    """Find all face videos in a DMD dataset directory.

    Actual DMD filenames look like:
        gA_1_s1_2019-03-08T09;31;15+01;00_rgb_face.mp4
        gA_1_s1_2019-03-08T09;31;15+01;00_rgb_ann_distraction.json

    Returns a list of dicts with keys:
        video_path, annotation_path, dataset, subject_id, video_id, session
    """
    entries: List[Dict[str, Any]] = []
    if not dataset_dir.exists():
        return entries

    # Match *_rgb_face.{mp4,avi,mkv}
    for video_path in sorted(dataset_dir.rglob("*_rgb_face.*")):
        if video_path.suffix.lower() not in (".mp4", ".avi", ".mkv"):
            continue

        stem = video_path.stem  # e.g. gA_1_s1_2019-03-08T09;31;15+01;00_rgb_face

        # Extract group, subject number, and session from the leading part
        # Pattern: gX_N_sM_..._rgb_face
        leading = stem.split("_rgb_face")[0]  # gA_1_s1_2019-...
        parts = leading.split("_")
        if len(parts) >= 3:
            group = parts[0]       # gA
            subj_num = parts[1]    # 1
            session = parts[2]     # s1
            subject_id = f"{group}_{subj_num}"
        else:
            subject_id = stem
            session = "unknown"

        # Locate annotation JSON in the same directory
        # DMD annotations are named: *_rgb_ann_distraction.json or *_rgb_ann_drowsiness.json
        annotation_path = None
        parent = video_path.parent
        ann_candidates = list(parent.glob("*_ann_*.json"))
        if ann_candidates:
            annotation_path = ann_candidates[0]  # typically one annotation per session
        else:
            # Fallback: try any .json file in the directory
            json_files = list(parent.glob("*.json"))
            if json_files:
                annotation_path = json_files[0]

        entries.append({
            "video_path": video_path,
            "annotation_path": str(annotation_path) if annotation_path else None,
            "dataset": dataset_name,
            "subject_id": subject_id,
            "video_id": stem,
            "session": session,
        })

    logger.info("Found %d face videos in %s", len(entries), dataset_name)
    return entries


def discover_uta_rldd_videos(dataset_dir: Path) -> List[Dict[str, Any]]:
    """Find all videos in UTA-RLDD dataset.

    Actual structure:
        UTA-RLDD/Fold1_part1/01/0.mov   (Alert)
        UTA-RLDD/Fold1_part1/01/5.mov   (Drowsy low)
        UTA-RLDD/Fold1_part1/01/10.MOV  (Drowsy high)

    The filename stem IS the drowsiness level.
    """
    entries: List[Dict[str, Any]] = []
    if not dataset_dir.exists():
        return entries

    for video_path in sorted(dataset_dir.rglob("*")):
        if video_path.suffix.lower() not in (".mp4", ".avi", ".mkv", ".mov"):
            continue

        # Derive subject from directory structure
        # Path: UTA-RLDD / FoldN_partM / subject_num / level.mov
        rel = video_path.relative_to(dataset_dir)
        parts = rel.parts  # e.g. ('Fold1_part1', '01', '0.mov')

        # Subject folder is the numeric directory (second-to-last component)
        subject_id = "unknown"
        if len(parts) >= 2:
            subject_dir = parts[-2]  # '01'
            if subject_dir.isdigit():
                subject_id = subject_dir

        # Drowsiness level IS the filename stem: '0', '5', or '10'
        stem = video_path.stem  # '0', '5', '10'
        level = stem.strip()    # the stem itself is the level

        label = UTA_RLDD_LABEL_MAP.get(level, "Alert")

        # Build a unique video_id including fold info
        fold = parts[0] if len(parts) >= 1 else "unknown"  # 'Fold1_part1'
        video_id = f"{fold}_{subject_id}_level{level}"

        entries.append({
            "video_path": video_path,
            "annotation_path": None,
            "dataset": "UTA-RLDD",
            "subject_id": f"uta_{subject_id}",
            "video_id": video_id,
            "session": f"level_{level}",
            "uniform_label": label,
        })

    logger.info("Found %d videos in UTA-RLDD", len(entries))
    return entries


def discover_auc_v2_images(dataset_dir: Path) -> List[Dict[str, Any]]:
    """Find all images in AUC v2 dataset using the CSV manifests.

    Structure after unzip:
        AUC-V2/v2_cam1/{Drive Safe,Text Left,...}/*.jpg
        AUC-V2/auc.distracted.driver.train.csv
        AUC-V2/auc.distracted.driver.test.csv

    We use the CSV for label mapping, but discover images on disk.
    """
    entries: List[Dict[str, Any]] = []
    if not dataset_dir.exists():
        return entries

    # Build label lookup from CSVs if available
    csv_labels: Dict[str, str] = {}  # image_basename -> label
    for csv_name in ["auc.distracted.driver.train.csv", "auc.distracted.driver.test.csv"]:
        csv_path = dataset_dir / csv_name
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    img_path_str = row["Image"]
                    label_int = int(row["Label"])
                    basename = Path(img_path_str).name
                    csv_labels[basename] = AUC_V2_LABEL_MAP.get(label_int, "Distracted")
            except Exception as exc:
                logger.warning("Cannot read %s: %s", csv_path, exc)

    # Find all images on disk
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for img_path in sorted(dataset_dir.rglob("*")):
        if img_path.suffix.lower() not in image_exts:
            continue
        # Skip macOS artifacts
        if "__MACOSX" in str(img_path):
            continue
        # Skip irrelevant folders
        if "Trash" in img_path.parts or "skin_nonskin_pixels" in img_path.parts:
            continue

        basename = img_path.name
        parent_folder = img_path.parent.name  # e.g. 'Drive Safe', 'Text Left', 'c0'

        # Determine label: prefer CSV, fallback to folder name
        if basename in csv_labels:
            label = csv_labels[basename]
        else:
            label = AUC_V2_FOLDER_LABEL.get(parent_folder, "Distracted")

        # Subject ID: AUC v2 has no real subjects, create synthetic buckets
        img_num = int(''.join(filter(str.isdigit, basename)) or '0')
        subject_id = f"auc_{img_num % 60:02d}"

        entries.append({
            "image_path": img_path,
            "dataset": "AUC-V2",
            "subject_id": subject_id,
            "image_id": img_path.stem,
            "label": label,
        })

    logger.info("Found %d images in AUC-V2", len(entries))
    return entries


def process_auc_v2_images(
    entries: List[Dict[str, Any]],
    mediapipe_model_path: str,
    output_root: Path,
) -> Tuple[int, int]:
    """Process AUC v2 images and save as parquet.

    Since these are single images (not video sequences), temporal features
    (PERCLOS, blink_rate, etc.) will be zero. Only per-frame geometric
    features are meaningful.
    """
    success = 0
    fail = 0

    # Group by subject for batch parquet files
    subject_entries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        subject_entries[entry["subject_id"]].append(entry)

    extractor = FeatureExtractor(
        model_path=mediapipe_model_path,
        fps=1.0,  # not meaningful for images
    )

    global_frame_idx = 0

    for subject_id, images in tqdm(subject_entries.items(), desc="AUC-V2 subjects", unit="subj"):
        out_dir = output_root / "AUC-V2" / subject_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{subject_id}_images.parquet"

        if out_path.exists():
            success += len(images)
            global_frame_idx += len(images)
            continue

        records = []
        # Reset extractor buffers for each subject group
        extractor.reset()

        for entry in images:
            img_path = Path(entry["image_path"])
            try:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    fail += 1
                    continue

                preprocessed = preprocess_frame(frame)
                # mediapipe needs strictly increasing timestamps across the landmarker lifetime
                global_frame_idx += 1
                timestamp_ms = global_frame_idx * 100

                feat = extractor.extract(preprocessed, timestamp_ms=timestamp_ms)

                record = {
                    "dataset": "AUC-V2",
                    "subject_id": subject_id,
                    "video_id": entry["image_id"],
                    "frame_idx": global_frame_idx,
                    "timestamp_s": timestamp_ms / 1000.0,
                    "ear_left": feat.get("ear_left", 0.0),
                    "ear_right": feat.get("ear_right", 0.0),
                    "ear_avg": feat.get("ear_avg", 0.0),
                    "mar": feat.get("mar", 0.0),
                    "perclos": feat.get("perclos", 0.0),
                    "blink_rate": feat.get("blink_rate", 0.0),
                    "blink_duration_avg": feat.get("blink_duration_avg", 0.0),
                    "yaw": feat.get("yaw", 0.0),
                    "pitch": feat.get("pitch", 0.0),
                    "roll": feat.get("roll", 0.0),
                    "gaze_yaw": feat.get("gaze_yaw", 0.0),
                    "gaze_pitch": feat.get("gaze_pitch", 0.0),
                    "gaze_stability": feat.get("gaze_stability", 0.0),
                    "head_pose_stability": feat.get("head_pose_stability", 0.0),
                    "ear_velocity": feat.get("ear_velocity", 0.0),
                    "head_nod_count": feat.get("head_nod_count", 0),
                    "mouth_open_duration": feat.get("mouth_open_duration", 0),
                    "eyes_off_road_pct": feat.get("eyes_off_road_pct", 0.0),
                    "label": entry["label"],
                }
                records.append(record)
                success += 1

            except Exception as exc:
                logger.debug("AUC-V2 image %s failed: %s", img_path.name, exc)
                fail += 1

        if records:
            df = pd.DataFrame(records)
            # Ensure column order matches video parquets
            cols = ["dataset", "subject_id", "video_id"] + FEATURE_COLUMNS
            for c in cols:
                if c not in df.columns:
                    df[c] = 0.0 if c != "label" else "Alert"
            df = df[[c for c in cols if c in df.columns]]
            df.to_parquet(out_path, index=False, engine="pyarrow")
            logger.debug("Saved %d images -> %s", len(records), out_path)

    extractor.close()
    return success, fail


# Per-frame interpolation for missing faces

def interpolate_missing_frames(
    features_list: List[Optional[Dict[str, float]]],
    max_gap: int = 5,
) -> List[Optional[Dict[str, float]]]:
    """Linearly interpolate up to *max_gap* consecutive missing frames.

    Longer gaps remain as None.
    """
    n = len(features_list)
    if n == 0:
        return features_list

    result = list(features_list)  # shallow copy

    i = 0
    while i < n:
        if result[i] is not None:
            i += 1
            continue

        # found a none, measure gap length
        gap_start = i
        while i < n and result[i] is None:
            i += 1
        gap_end = i  # exclusive
        gap_len = gap_end - gap_start

        if gap_len > max_gap:
            continue  # leave as None

        # Find bounding valid frames
        prev_idx = gap_start - 1
        next_idx = gap_end

        if prev_idx < 0 or next_idx >= n:
            # cant interpolate at boundaries, copy nearest valid
            valid = result[prev_idx] if prev_idx >= 0 else (result[next_idx] if next_idx < n else None)
            if valid is not None:
                for j in range(gap_start, gap_end):
                    result[j] = dict(valid)
            continue

        prev_feat = result[prev_idx]
        next_feat = result[next_idx]
        if prev_feat is None or next_feat is None:
            continue

        # Linear interpolation
        for j in range(gap_start, gap_end):
            alpha = (j - prev_idx) / (next_idx - prev_idx)
            interpolated = {}
            for key in prev_feat:
                pv = prev_feat[key]
                nv = next_feat[key]
                if isinstance(pv, (int, float)) and isinstance(nv, (int, float)):
                    interpolated[key] = pv + alpha * (nv - pv)
                else:
                    interpolated[key] = pv
            result[j] = interpolated

    return result


# Process a single video

def process_video(
    entry: Dict[str, Any],
    mediapipe_model_path: str,
    output_root: Path,
    max_interp_gap: int = 5,
) -> Optional[Path]:
    """Process one video file and save features as .parquet.

    Returns the output path on success, None on failure.
    """
    video_path = Path(entry["video_path"])
    dataset = entry["dataset"]
    subject_id = entry["subject_id"]
    video_id = entry["video_id"]

    # Output path
    out_dir = output_root / dataset / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.parquet"

    if out_path.exists():
        return out_path  # already processed

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.76
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        logger.warning("Video %s reports 0 frames, skipping", video_path)
        cap.release()
        return None

    # Determine per-frame labels
    if dataset == "DMD-Distraction":
        ann_path = entry.get("annotation_path")
        labels = parse_dmd_distraction_labels(
            Path(ann_path) if ann_path else Path(""), total_frames
        )
    elif dataset == "DMD-Drowsiness":
        ann_path = entry.get("annotation_path")
        labels = parse_dmd_drowsiness_labels(
            Path(ann_path) if ann_path else Path(""), total_frames
        )
    elif dataset == "UTA-RLDD":
        uniform = entry.get("uniform_label", "Alert")
        labels = [uniform] * total_frames
    else:
        labels = ["Alert"] * total_frames

    # Create feature extractor
    extractor = FeatureExtractor(
        model_path=mediapipe_model_path,
        fps=fps,
    )

    # Read frames and extract features
    raw_features: List[Optional[Dict[str, float]]] = []
    frame_idx = 0
    prev_ts_ms = -1  # track for monotonicity

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Ensure timestamps are strictly monotonically increasing
        if timestamp_ms <= prev_ts_ms:
            timestamp_ms = prev_ts_ms + int(1000.0 / fps)
        prev_ts_ms = timestamp_ms

        try:
            # CLAHE preprocessing
            preprocessed = preprocess_frame(frame)

            # Extract features via MediaPipe + landmark calculations
            feat = extractor.extract(
                preprocessed,
                timestamp_ms=timestamp_ms,
            )

            if feat is not None:
                feat["frame_idx"] = frame_idx
                feat["timestamp_s"] = timestamp_ms / 1000.0
                raw_features.append(feat)
            else:
                raw_features.append(None)

        except Exception as exc:
            logger.debug("Frame %d of %s failed: %s", frame_idx, video_id, exc)
            raw_features.append(None)

        frame_idx += 1

    cap.release()
    extractor.close()

    if frame_idx == 0:
        logger.warning("No frames read from %s", video_path)
        return None

    # Interpolate short gaps
    interpolated = interpolate_missing_frames(raw_features, max_gap=max_interp_gap)

    # Build DataFrame
    records = []
    for i, feat in enumerate(interpolated):
        if feat is None:
            continue  # discard frames in long gaps

        label = labels[i] if i < len(labels) else "Alert"
        record = {
            "frame_idx": int(feat.get("frame_idx", i)),
            "timestamp_s": feat.get("timestamp_s", i / fps),
            "ear_left": feat.get("ear_left", np.nan),
            "ear_right": feat.get("ear_right", np.nan),
            "ear_avg": feat.get("ear_avg", np.nan),
            "mar": feat.get("mar", np.nan),
            "perclos": feat.get("perclos", 0.0),
            "blink_rate": feat.get("blink_rate", 0.0),
            "blink_duration_avg": feat.get("blink_duration_avg", 0.0),
            "yaw": feat.get("yaw", 0.0),
            "pitch": feat.get("pitch", 0.0),
            "roll": feat.get("roll", 0.0),
            "gaze_yaw": feat.get("gaze_yaw", 0.0),
            "gaze_pitch": feat.get("gaze_pitch", 0.0),
            "gaze_stability": feat.get("gaze_stability", 0.0),
            "head_pose_stability": feat.get("head_pose_stability", 0.0),
            "ear_velocity": feat.get("ear_velocity", 0.0),
            "head_nod_count": feat.get("head_nod_count", 0),
            "mouth_open_duration": feat.get("mouth_open_duration", 0),
            "eyes_off_road_pct": feat.get("eyes_off_road_pct", 0.0),
            "label": label,
        }
        records.append(record)

    if not records:
        logger.warning("No valid frames extracted from %s", video_path)
        return None

    df = pd.DataFrame(records, columns=FEATURE_COLUMNS)

    # Add metadata columns
    df.insert(0, "dataset", dataset)
    df.insert(1, "subject_id", subject_id)
    df.insert(2, "video_id", video_id)

    # Save
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.debug("Saved %d frames -> %s", len(df), out_path)
    return out_path


def _process_video_wrapper(args: Tuple) -> Optional[str]:
    """Wrapper for ProcessPoolExecutor (must be top-level picklable)."""
    entry, model_path, output_root, max_gap = args
    try:
        result = process_video(entry, model_path, Path(output_root), max_gap)
        return str(result) if result else None
    except Exception as exc:
        logger.error("Error processing %s: %s", entry.get("video_id", "?"), exc)
        return None


# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-frame features from face videos using MediaPipe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config/config.yaml", help="Config YAML path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel video workers (default: 1)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["DMD-Distraction", "DMD-Drowsiness", "UTA-RLDD", "AUC-V2"],
                        help="Process only this dataset (default: all)")
    parser.add_argument("--max-interp-gap", type=int, default=5,
                        help="Max consecutive missing frames to interpolate (default: 5)")
    parser.add_argument("--data-root", type=str, default=None, help="Override data root dir")
    args = parser.parse_args()

    _setup_logging(args.log_level)
    cfg = _load_config(args.config)

    project_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root) if args.data_root else Path(
        cfg.get("data", {}).get("root", project_root / "data")
    )
    raw_dir = data_root / "raw"
    features_dir = data_root / "features"
    models_dir = project_root / "models"

    # Ensure MediaPipe model is available
    mp_model_path = ensure_mediapipe_model(models_dir)

    # Discover all videos
    all_entries: List[Dict[str, Any]] = []
    auc_entries: List[Dict[str, Any]] = []

    if args.dataset is None or args.dataset == "DMD-Distraction":
        all_entries.extend(discover_dmd_videos(raw_dir / "DMD-Distraction", "DMD-Distraction"))

    if args.dataset is None or args.dataset == "DMD-Drowsiness":
        all_entries.extend(discover_dmd_videos(raw_dir / "DMD-Drowsiness", "DMD-Drowsiness"))

    if args.dataset is None or args.dataset == "UTA-RLDD":
        all_entries.extend(discover_uta_rldd_videos(raw_dir / "UTA-RLDD"))

    if args.dataset is None or args.dataset == "AUC-V2":
        auc_entries = discover_auc_v2_images(raw_dir / "AUC-V2")

    if not all_entries and not auc_entries:
        logger.error("No videos/images found. Check your data directory: %s", raw_dir)
        sys.exit(1)

    logger.info("Total videos to process: %d", len(all_entries))
    if auc_entries:
        logger.info("Total AUC-V2 images to process: %d", len(auc_entries))
    logger.info("Output directory: %s", features_dir)
    logger.info("Workers: %d", args.workers)

    # Process AUC-V2 images first (separate pipeline)
    if auc_entries:
        logger.info("Processing AUC-V2 images...")
        auc_ok, auc_fail = process_auc_v2_images(
            auc_entries, str(mp_model_path), features_dir
        )
        logger.info("AUC-V2: %d success, %d failed", auc_ok, auc_fail)

    # Process videos
    success_count = 0
    fail_count = 0

    if args.workers <= 1:
        # sequential, simpler debugging, progress bar per video
        for entry in tqdm(all_entries, desc="Processing videos", unit="video"):
            result = process_video(
                entry, str(mp_model_path), features_dir, args.max_interp_gap
            )
            if result:
                success_count += 1
            else:
                fail_count += 1
    else:
        # parallel, video-level parallelism
        work_items = [
            (entry, str(mp_model_path), str(features_dir), args.max_interp_gap)
            for entry in all_entries
        ]

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process_video_wrapper, item): item[0]["video_id"]
                for item in work_items
            }
            with tqdm(total=len(futures), desc="Processing videos", unit="video") as pbar:
                for future in as_completed(futures):
                    video_id = futures[future]
                    try:
                        result = future.result()
                        if result:
                            success_count += 1
                        else:
                            fail_count += 1
                    except Exception as exc:
                        logger.error("Video %s raised: %s", video_id, exc)
                        fail_count += 1
                    pbar.update(1)

    logger.info("feature extraction complete")
    logger.info("  Successful: %d", success_count)
    logger.info("  Failed:     %d", fail_count)
    logger.info("  Output dir: %s", features_dir)

    # Per-dataset feature file stats
    if features_dir.exists():
        for ds_dir in sorted(features_dir.iterdir()):
            if ds_dir.is_dir():
                parquet_files = list(ds_dir.rglob("*.parquet"))
                total_rows = 0
                label_counts: Dict[str, int] = defaultdict(int)
                for pf in parquet_files:
                    try:
                        df = pd.read_parquet(pf, columns=["label"])
                        total_rows += len(df)
                        for lbl, cnt in df["label"].value_counts().items():
                            label_counts[lbl] += cnt
                    except Exception:
                        pass
                logger.info(
                    "  %s: %d parquet files, %d total frames, labels=%s",
                    ds_dir.name, len(parquet_files), total_rows, dict(label_counts),
                )

    logger.info("Next step: python scripts/build_splits.py --config %s", args.config)


if __name__ == "__main__":
    main()
