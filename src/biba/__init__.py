from __future__ import annotations

from .compositor import ReelCompositor
from .decoder import VideoStreamReader, decode_audio_file, decode_video_segment
from .layout import EMERALD, FPS, GOLD, H, MUTED, TEAL, W, WHITE, render_color_field
from .spec import ReelTimeline

__version__ = "1.1.0"
__all__ = [
    "EMERALD",
    "FPS",
    "GOLD",
    "H",
    "MUTED",
    "ReelCompositor",
    "ReelTimeline",
    "TEAL",
    "VideoStreamReader",
    "W",
    "WHITE",
    "decode_audio_file",
    "decode_video_segment",
    "render_color_field",
]
