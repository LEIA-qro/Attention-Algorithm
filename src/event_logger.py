"""
event_logger.py — CSV and JSON Event Log Writer
================================================

Public API
----------
- ``EventLogger(output_dir, base_name)``
    - ``.log(event_type, confidence, trigger_timestamp_s, clip_path, driver_box)``
    - ``.close()``
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

__all__ = ["EventLogger"]

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "session_timestamp", "event_type", "confidence",
    "trigger_timestamp_s", "clip_path",
    "driver_box_x1", "driver_box_y1", "driver_box_x2", "driver_box_y2",
]


class EventLogger:
    """Appends one row per ``log()`` call to a CSV and a JSON file.

    Both files share the same session-timestamp suffix so they're paired.

    Parameters
    ----------
    output_dir : Path  — created if absent
    base_name : str    — file prefix (e.g. "events")
    """

    def __init__(self, output_dir: Path, base_name: str = "events") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = self._output_dir / f"{base_name}_{ts}.csv"
        self._json_path = self._output_dir / f"{base_name}_{ts}.json"
        self._records: List[dict] = []
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        self._csv_file.flush()
        logger.info("EventLogger CSV  → %s", self._csv_path)
        logger.info("EventLogger JSON → %s", self._json_path)

    def log(
        self,
        event_type: str,
        confidence: float,
        trigger_timestamp_s: float,
        clip_path: Optional[Path],
        driver_box: Optional[np.ndarray],
    ) -> None:
        """Append one event record."""
        now = datetime.now().isoformat(timespec="milliseconds")
        box = driver_box if driver_box is not None else [None, None, None, None]
        row = {
            "session_timestamp": now,
            "event_type": event_type,
            "confidence": round(float(confidence), 4),
            "trigger_timestamp_s": round(float(trigger_timestamp_s), 3),
            "clip_path": str(clip_path) if clip_path else "",
            "driver_box_x1": float(box[0]) if box[0] is not None else "",
            "driver_box_y1": float(box[1]) if box[1] is not None else "",
            "driver_box_x2": float(box[2]) if box[2] is not None else "",
            "driver_box_y2": float(box[3]) if box[3] is not None else "",
        }
        self._writer.writerow(row)
        self._csv_file.flush()

        record = {k: row[k] for k in
                  ["session_timestamp", "event_type", "confidence",
                   "trigger_timestamp_s", "clip_path"]}
        self._records.append(record)
        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=2)

        logger.info("Logged: %s conf=%.2f clip=%s", event_type, confidence, clip_path)

    def close(self) -> None:
        """Flush and close the CSV file handle."""
        self._csv_file.close()
        logger.info("EventLogger closed.")
