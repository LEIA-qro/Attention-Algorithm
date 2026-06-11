"""Ring buffer of recent frames plus a background MP4 clip writer."""

from __future__ import annotations

import functools
import logging
import queue
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

__all__ = ["ClipWriter"]

logger = logging.getLogger(__name__)

# Browsers (HTML5 <video>) and WhatsApp only play H.264 in an MP4 container.
# OpenCV can emit H.264 only when its bundled FFmpeg was built with an encoder,
# which is frequently missing on a plain `pip install opencv-python`. When that
# is the case avc1/H264 fail to open and the pipeline falls back to mp4v
# (MPEG-4 Part 2): the file plays in VLC but shows up as a broken/corrupt clip
# in the browser (only the poster/snapshot renders). To stay robust across
# machines we resolve, once, the most compatible encoder available here.
_H264_FOURCCS = {"avc1", "h264", "x264"}


@functools.lru_cache(maxsize=1)
def _ffmpeg_h264_exe() -> Optional[str]:
    """Locate an ffmpeg binary that can encode H.264 (libx264).

    Prefers a system ffmpeg on PATH, then the binary bundled by the
    ``imageio-ffmpeg`` pip package (a pure-pip way to ship ffmpeg+libx264).
    """
    candidates: List[str] = []
    system = shutil.which("ffmpeg")
    if system:
        candidates.append(system)
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass

    for exe in candidates:
        try:
            out = subprocess.run(
                [exe, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=10,
            )
            if "libx264" in out.stdout:
                return exe
        except Exception:
            continue
    return None


@functools.lru_cache(maxsize=8)
def _opencv_can_open(fourcc_str: str) -> bool:
    """Return True if OpenCV can actually open a VideoWriter for this codec.

    A failed probe makes OpenCV/FFmpeg print C-level errors straight to stderr,
    so we redirect fd 2 to /dev/null for the duration to keep startup logs clean.
    """
    import os

    tmp = Path(tempfile.gettempdir()) / f"._clipwriter_probe_{fourcc_str}.mp4"
    saved_fd = None
    devnull_fd = None
    try:
        try:
            saved_fd = os.dup(2)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, 2)
        except OSError:
            saved_fd = None

        writer = cv2.VideoWriter(
            str(tmp), cv2.VideoWriter_fourcc(*fourcc_str), 25.0, (320, 240)
        )
        opened = writer.isOpened()
        writer.release()
        return opened
    except Exception:
        return False
    finally:
        if saved_fd is not None:
            try:
                os.dup2(saved_fd, 2)
                os.close(saved_fd)
            except OSError:
                pass
        if devnull_fd is not None:
            try:
                os.close(devnull_fd)
            except OSError:
                pass
        try:
            tmp.unlink()
        except OSError:
            pass


class ClipWriter:
    """Buffers recent frames and writes triggered clips to MP4 on a worker thread."""

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
        self._encode = self._resolve_encoder()
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
        """Queue ring-buffer plus post_frames for background MP4 writing and return the target path."""
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
        """Write ring-buffer plus post_frames to an MP4 and return its path, or None on failure."""
        all_frames = [f.copy() for f, _ in self._ring_buffer] + [f.copy() for f in post_frames]
        return self._write_clip_sync(event_type, confidence, trigger_timestamp_s, all_frames)

    def _resolve_encoder(self) -> Tuple[str, str]:
        """Pick the most browser-compatible encoding strategy available here.

        Returns a ``(mode, fourcc)`` tuple where ``mode`` is either ``"opencv"``
        (write directly with the given fourcc) or ``"ffmpeg"`` (write mp4v, then
        transcode to H.264). The goal is always H.264/MP4 output.
        """
        # 1. Native OpenCV H.264 — fastest path, used when the backend supports it.
        if self._fourcc_str.lower() in _H264_FOURCCS and _opencv_can_open(self._fourcc_str):
            logger.info("ClipWriter: encoding clips natively with OpenCV '%s' (H.264).", self._fourcc_str)
            return ("opencv", self._fourcc_str)

        # 2. OpenCV can't emit H.264 here -> write reliable mp4v and let ffmpeg transcode.
        if _ffmpeg_h264_exe():
            logger.info(
                "ClipWriter: OpenCV cannot encode '%s' on this machine; "
                "writing mp4v and transcoding to H.264 with ffmpeg.",
                self._fourcc_str,
            )
            return ("ffmpeg", "mp4v")

        # 3. No H.264 path at all. Keep recording, but warn the clips won't play
        #    in browsers or WhatsApp until ffmpeg/libx264 is available.
        codec = self._fourcc_str if _opencv_can_open(self._fourcc_str) else "mp4v"
        logger.warning(
            "ClipWriter: no H.264 encoder available (OpenCV lacks it and no ffmpeg+libx264 "
            "was found); clips will be written as '%s' (MPEG-4 Part 2), which most browsers "
            "and WhatsApp cannot play. Install ffmpeg or run `pip install imageio-ffmpeg`.",
            codec,
        )
        return ("opencv", codec)

    def _transcode_to_h264(self, src: Path, dst: Path) -> bool:
        """Transcode ``src`` to a browser-safe H.264 MP4 at ``dst`` using ffmpeg.

        Produces yuv420p (the only chroma format browsers decode) with the moov
        atom moved to the front (``+faststart``) for progressive playback, and
        rounds odd dimensions down to even (H.264 requires even width/height).
        Removes ``src`` on success. Returns False if the transcode fails.
        """
        exe = _ffmpeg_h264_exe()
        if not exe:
            return False
        cmd = [
            exe, "-y", "-loglevel", "error",
            "-i", str(src),
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-movflags", "+faststart",
            str(dst),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, OSError) as e:
            stderr = getattr(e, "stderr", b"") or b""
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            logger.error("ffmpeg H.264 transcode failed: %s", stderr)
            return False
        try:
            src.unlink()
        except OSError:
            pass
        return True

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
        mode, fourcc_str = self._encode

        # In "opencv" mode we write the final file directly; in "ffmpeg" mode we
        # first write a temporary mp4v file that ffmpeg then transcodes to H.264.
        raw_path = out_path if mode == "opencv" else out_path.with_suffix(".raw.mp4")

        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(raw_path), fourcc, self._fps, (w, h))
        if not writer.isOpened():
            logger.error("Could not open VideoWriter for %s", raw_path)
            return None
        try:
            for frame in all_frames:
                if frame.shape[0] != h or frame.shape[1] != w:
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
        finally:
            writer.release()

        if mode == "ffmpeg" and not self._transcode_to_h264(raw_path, out_path):
            # Transcode failed — keep the raw mp4v footage rather than losing the clip.
            raw_path.replace(out_path)
            logger.warning(
                "Kept mp4v clip %s after ffmpeg transcode failure (may not play in browsers).",
                out_path,
            )

        logger.info("Clip saved: %s  (%d frames)", out_path, len(all_frames))
        return out_path

    def reset(self) -> None:
        """Clear the ring buffer."""
        self._ring_buffer.clear()

