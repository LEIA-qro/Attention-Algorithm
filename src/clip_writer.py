"""
clip_writer.py — Ring-Buffer Frame Store and MP4 Clip Saver
============================================================

Public API
----------
- ``ClipWriter(output_dir, fps, pre_buffer_seconds, post_buffer_seconds, fourcc)``
    - ``.push_frame(frame, timestamp_s)``
    - ``.save_clip(event_type, confidence, trigger_timestamp_s, post_frames) -> Path | None``
    - ``.reset()``
    - ``._ring_buffer``  — deque[tuple[ndarray, float]] (accessible for tests)
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

__all__ = ["ClipWriter"]

logger = logging.getLogger(__name__)


class ClipWriter:
    """Ring-buffer of frames + MP4 writer.

    Parameters
    ----------
    output_dir : Path    — directory for saved clips (created if absent)
    fps : float          — source video frame rate
    pre_buffer_seconds   — seconds before trigger to include
    post_buffer_seconds  — seconds after trigger (caller provides post_frames)
    fourcc : str         — OpenCV FourCC codec string, e.g. "mp4v"
    """

    def __init__(
        self,
        output_dir: Path,
        fps: float,
        pre_buffer_seconds: float,
        post_buffer_seconds: float,
        fourcc: str = "mp4v",
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be greater than 0")
        if pre_buffer_seconds <= 0:
            raise ValueError("pre_buffer_seconds must be greater than 0")
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._pre_s = pre_buffer_seconds
        self._post_s = post_buffer_seconds
        self._fourcc_str = fourcc
        max_frames = int(pre_buffer_seconds * fps) + 1
        self._ring_buffer: Deque[Tuple[np.ndarray, float]] = deque(maxlen=max_frames)

        self._queue: queue.Queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def push_frame(self, frame: np.ndarray, timestamp_s: float) -> None:
        """Add one BGR uint8 frame to the ring buffer."""
        self._ring_buffer.append((frame.copy(), timestamp_s))

    def get_clip_path(
        self,
        event_type: str,
        confidence: float,
        trigger_timestamp_s: float,
    ) -> Path:
        """Calculate the file path for the saved clip."""
        dt = datetime.fromtimestamp(trigger_timestamp_s)
        ts_str = dt.strftime("%Y%m%d_%H%M%S_%f")[:19]
        conf_pct = int(confidence * 100)
        return self._output_dir / f"{ts_str}_{event_type}_{conf_pct:02d}pct.mp4"

    def save_clip_async(
        self,
        event_type: str,
        confidence: float,
        trigger_timestamp_s: float,
        post_frames: List[np.ndarray],
    ) -> Path:
        """Asynchronously write ring-buffer frames + post_frames to an MP4 file.
        
        Returns
        -------
        Path where the file will be written.
        """
        # Defensive copy of frames
        all_frames = [f.copy() for f, _ in self._ring_buffer] + [f.copy() for f in post_frames]
        out_path = self.get_clip_path(event_type, confidence, trigger_timestamp_s)
        self._queue.put((event_type, confidence, trigger_timestamp_s, all_frames))
        return out_path

    def flush(self) -> None:
        """Block until all queued clip writing tasks are completed."""
        self._queue.join()

    def close(self) -> None:
        """Stop the background worker thread cleanly after flushing pending tasks."""
        self.flush()
        self._queue.put(None)
        self._worker_thread.join()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            event_type, confidence, trigger_timestamp_s, all_frames = item
            try:
                self._write_clip_sync(event_type, confidence, trigger_timestamp_s, all_frames)
            except Exception as e:
                logger.error("Error writing clip in background: %s", e)
            finally:
                self._queue.task_done()

    def save_clip(
        self,
        event_type: str,
        confidence: float,
        trigger_timestamp_s: float,
        post_frames: List[np.ndarray],
    ) -> Optional[Path]:
        """Write ring-buffer frames + post_frames to an MP4 file.

        Returns
        -------
        Path of written file, or None if writing failed.
        """
        all_frames = [f.copy() for f, _ in self._ring_buffer] + [f.copy() for f in post_frames]
        return self._write_clip_sync(event_type, confidence, trigger_timestamp_s, all_frames)

    def _write_clip_sync(
        self,
        event_type: str,
        confidence: float,
        trigger_timestamp_s: float,
        all_frames: List[np.ndarray],
    ) -> Optional[Path]:
        if not all_frames:
            logger.warning("save_clip: no frames to write")
            return None

        h, w = all_frames[0].shape[:2]
        out_path = self.get_clip_path(event_type, confidence, trigger_timestamp_s)

        fourcc = cv2.VideoWriter_fourcc(*self._fourcc_str)
        writer = cv2.VideoWriter(str(out_path), fourcc, self._fps, (w, h))
        if not writer.isOpened():
            logger.error("Could not open VideoWriter for %s", out_path)
            return None
        try:
            for frame in all_frames:
                if frame.shape[0] != h or frame.shape[1] != w:
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
        finally:
            writer.release()

        logger.info("Clip saved: %s  (%d frames)", out_path, len(all_frames))
        return out_path

    def reset(self) -> None:
        """Clear the ring buffer."""
        self._ring_buffer.clear()

