from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageFilter

try:
    from .layout import COVER_H, COVER_W, TEAL, render_color_field
except ImportError:
    from layout import COVER_H, COVER_W, TEAL, render_color_field  # type: ignore[no-redef]


class VideoStreamReader:
    """Streams decoded RGB24 frames from a video file over an FFmpeg pipe."""

    def __init__(
        self,
        video_path: Path,
        width: int,
        height: int,
        fps: int = 30,
        ffmpeg_exe: Optional[str] = None,
        loop: bool = True,
    ) -> None:
        self.video_path = video_path
        self.width = width
        self.height = height
        self.fps = fps
        self.ffmpeg_exe = ffmpeg_exe or imageio_ffmpeg.get_ffmpeg_exe()
        self.loop = loop
        self.frame_bytes = width * height * 3
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._buf = bytearray(self.frame_bytes)

    def _open(self) -> None:
        if self._proc is not None:
            return
        cmd: List[str] = [self.ffmpeg_exe, "-loglevel", "error"]
        if self.loop:
            cmd.extend(["-stream_loop", "-1"])
        cmd.extend(
            [
                "-i",
                str(self.video_path),
                "-vf",
                f"fps={self.fps},scale={self.width}:{self.height}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={self.width}:{self.height},eq=contrast=1.06:brightness=-0.02:saturation=1.06",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ]
        )
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_bytes * 2,
        )

    def read_frame(self) -> Optional[Image.Image]:
        if not self.video_path.is_file():
            return None
        self._open()
        if self._proc is None or self._proc.stdout is None:
            return None
        view = memoryview(self._buf)
        offset = 0
        while offset < self.frame_bytes:
            chunk = self._proc.stdout.read(self.frame_bytes - offset)
            if not chunk:
                self.close()
                return None
            view[offset : offset + len(chunk)] = chunk
            offset += len(chunk)
        return Image.frombuffer("RGB", (self.width, self.height), bytes(self._buf), "raw", "RGB", 0, 1)

    def iter_frames(self, count: int, fallback_accent: Tuple[int, int, int] = TEAL) -> Iterator[Image.Image]:
        try:
            for idx in range(count):
                frame = self.read_frame()
                if frame is not None:
                    yield frame
                else:
                    yield render_color_field(idx, self.width, self.height, accent=fallback_accent)
        finally:
            self.close()

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdout is not None:
                    self._proc.stdout.close()
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                pass
            self._proc = None

    def __enter__(self) -> "VideoStreamReader":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


def decode_video_segment(
    video_path: Path,
    count: int,
    target_w: int = 1080,
    target_h: int = 620,
    accent: Tuple[int, int, int] = TEAL,
    ffmpeg_exe: Optional[str] = None,
) -> List[Image.Image]:
    if video_path.is_file():
        with VideoStreamReader(video_path, target_w, target_h, ffmpeg_exe=ffmpeg_exe) as reader:
            frames: List[Image.Image] = []
            for _ in range(count):
                fr = reader.read_frame()
                if fr is None:
                    break
                frames.append(fr)
            if frames:
                while len(frames) < count:
                    frames.append(frames[len(frames) % len(frames)])
                return frames
    return [render_color_field(i, target_w, target_h, accent=accent) for i in range(count)]


def decode_audio_file(
    audio_path: Path,
    sample_rate: int = 48000,
    ffmpeg_exe: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Decodes an external audio file (.wav, .mp3, .m4a, .flac) into float32 stereo PCM [-1, 1]."""
    if not audio_path.is_file():
        return None
    exe = ffmpeg_exe or imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        exe,
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-ac",
        "2",
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        raw = subprocess.check_output(cmd, timeout=30)
        pcm = np.frombuffer(raw, dtype=np.float32)
        if pcm.size < 2:
            return None
        return pcm.reshape(-1, 2)
    except Exception:
        return None


def load_still_cover(
    path: Path,
    cover_w: int = COVER_W,
    cover_h: int = COVER_H,
    accent: Tuple[int, int, int] = TEAL,
) -> Image.Image:
    if path.is_file():
        im = Image.open(path).convert("RGB")
    else:
        im = render_color_field(0, cover_w // 2, cover_h // 2, accent)
    scale = max(cover_w / im.width, cover_h / im.height)
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
    return im.filter(ImageFilter.UnsharpMask(radius=3, percent=70, threshold=2))
