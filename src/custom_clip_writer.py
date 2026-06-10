"""ClipWriter variant that accepts explicit pre_frames instead of the ring buffer."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from src.clip_writer import ClipWriter

__all__ = ["CustomClipWriter"]


class CustomClipWriter(ClipWriter):
    """ClipWriter that accepts explicit pre_frames instead of pulling from the ring buffer."""

    def _prepare_frames(
        self,
        post_frames: List[np.ndarray],
        pre_frames: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """Copy and concatenate pre_frames (or the ring buffer) with post_frames."""
        if pre_frames is None:
            pre_frames_copied = [f.copy() for f, _ in self._ring_buffer]
        else:
            pre_frames_copied = [f.copy() for f in pre_frames]

        return pre_frames_copied + [f.copy() for f in post_frames]

    def save_clip_async(
        self,
        event_type: str,
        confidence: float,
        trigger_timestamp_s: float,
        post_frames: List[np.ndarray],
        pre_frames: Optional[List[np.ndarray]] = None,
    ) -> Path:
        """Queue pre_frames (or the ring buffer) plus post_frames for background writing and return the target path."""
        all_frames = self._prepare_frames(post_frames, pre_frames)
        out_path = self.get_clip_path(event_type, confidence, trigger_timestamp_s)
        self._queue.put((event_type, confidence, trigger_timestamp_s, all_frames))
        return out_path

    def save_clip(
        self,
        event_type: str,
        confidence: float,
        trigger_timestamp_s: float,
        post_frames: List[np.ndarray],
        pre_frames: Optional[List[np.ndarray]] = None,
    ) -> Optional[Path]:
        """Write pre_frames (or the ring buffer) plus post_frames to an MP4 and return its path, or None on failure."""
        all_frames = self._prepare_frames(post_frames, pre_frames)
        return self._write_clip_sync(event_type, confidence, trigger_timestamp_s, all_frames)
