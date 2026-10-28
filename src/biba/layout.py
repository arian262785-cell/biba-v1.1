from __future__ import annotations
import math
from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
COVER_W, COVER_H = 1620, 2880

HN_PATH = "/System/Library/Fonts/HelveticaNeue.ttc"
SFNS_PATH = "/System/Library/Fonts/SFNS.ttf"
MONO_PATH = "/System/Library/Fonts/Menlo.ttc"

WHITE = (255, 255, 255)
TEAL = (64, 224, 208)
EMERALD = (34, 197, 94)
GOLD = (255, 214, 10)
AMBER = (255, 176, 92)
MUTED = (182, 194, 206)


def clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return max(a, min(b, x))


def ease_out(x: float) -> float:
    x = clamp(x)
    return 1.0 - (1.0 - x) ** 3


def ease_in_out(x: float) -> float:
    x = clamp(x)
    return 4.0 * x ** 3 if x < 0.5 else 1.0 - (-2.0 * x + 2.0) ** 3 / 2.0


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def prog(f: int, start: int, dur: int) -> float:
    return clamp((f - start) / max(1, dur))


def quantize_sub_lsb_dither(
    field_f32: np.ndarray,
    amplitude: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Applies monochromatic triangular PDF (TPDF) sub-LSB dither before rounding float32 RGB arrays to uint8."""
    arr = np.asarray(field_f32, dtype=np.float32)
    rng = np.random.default_rng(seed)
    dither_shape = arr.shape[:2] + ((1,) if arr.ndim == 3 else ())
    tpdf = (rng.random(dither_shape, dtype=np.float32) - rng.random(dither_shape, dtype=np.float32)) * amplitude
    return np.clip(np.rint(arr + tpdf), 0.0, 255.0).astype(np.uint8)


def load_sans(size: int, idx: int = 1) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(HN_PATH, size, index=idx)
    except Exception:
        pass
    for fallback in (
        SFNS_PATH,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(fallback, size)
        except Exception:
            continue
    return ImageFont.load_default(size=size)


def load_mono(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        MONO_PATH,
        SFNS_PATH,
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "DejaVuSansMono.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default(size=size)


F_HOOK = load_sans(118, 9)
F_HOOK_MD = load_sans(92, 9)
F_HOOK2 = load_sans(64, 1)
F_TITLE = load_sans(52, 1)
F_BODY = load_sans(36, 10)
F_BODY_L = load_sans(33, 7)
F_SMALL = load_sans(26, 10)
F_TINY = load_sans(22, 7)
F_STEP = load_sans(28, 1)
F_MONO = load_mono(28)
F_MONO_S = load_mono(22)


def text_width(font: ImageFont.FreeTypeFont, s: str) -> int:
    bbox = font.getbbox(s)
    return bbox[2] - bbox[0]


def render_color_field(
    frame_idx: int,
    target_w: int = 1080,
    target_h: int = 540,
    accent: Tuple[int, int, int] = TEAL,
) -> Image.Image:
    t = float(frame_idx) / float(FPS)
    y = np.linspace(-1.0, 1.0, target_h, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, target_w, dtype=np.float32)[None, :]

    r2 = x * x + y * y
    w1 = np.sin(x * 6.5 + t * 2.2) * np.cos(y * 5.0 - t * 1.7)
    w2 = np.sin(np.sqrt(r2 + 0.04) * 11.0 - t * 3.4)
    glow = np.exp(-r2 * 1.85) * (0.65 + 0.35 * np.sin(t * 1.9))
    field = np.clip(0.5 + 0.28 * w1 + 0.22 * w2 + 0.35 * glow, 0.0, 1.0)

    ar, ag, ab = accent[0] / 255.0, accent[1] / 255.0, accent[2] / 255.0
    ch_r = np.clip((0.04 + field * ar * 0.85 + glow * 0.25) * 255.0, 0.0, 255.0)
    ch_g = np.clip((0.07 + field * ag * 0.90 + glow * 0.30) * 255.0, 0.0, 255.0)
    ch_b = np.clip((0.12 + field * ab * 0.95 + glow * 0.35) * 255.0, 0.0, 255.0)

    rgb_f32 = np.stack([ch_r, ch_g, ch_b], axis=-1)
    rgb = quantize_sub_lsb_dither(rgb_f32, amplitude=0.5, seed=42 + int(frame_idx))
    return Image.fromarray(rgb, mode="RGB")


def draw_shadow_text(
    img: Image.Image,
    xy: Tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int],
    anchor: str = "mm",
    shadow: int = 12,
    alpha: float = 1.0,
) -> None:
    if alpha <= 0:
        return
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((xy[0], xy[1] + 5), text, font=font, fill=(0, 0, 0, 215), anchor=anchor)
    layer = layer.filter(ImageFilter.GaussianBlur(shadow))
    d = ImageDraw.Draw(layer)
    d.text(xy, text, font=font, fill=fill + (255,), anchor=anchor)
    if alpha < 1.0:
        layer.putalpha(layer.getchannel("A").point(lambda v: int(v * alpha)))
    img.alpha_composite(layer)


def draw_card(
    frame: Image.Image,
    box: Tuple[int, int, int, int],
    alpha: float = 1.0,
    radius: int = 34,
    accent: Tuple[int, int, int] = TEAL,
) -> None:
    if alpha <= 0:
        return
    x0, y0, x1, y1 = [int(v) for v in box]
    region = frame.crop((x0, y0, x1, y1)).convert("RGB").filter(ImageFilter.GaussianBlur(22))
    region = ImageEnhance.Brightness(region).enhance(0.40)
    tint = Image.new("RGB", region.size, (7, 18, 26))
    region = Image.blend(region, tint, 0.38).convert("RGBA")
    mask = Image.new("L", region.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, region.width - 1, region.height - 1), radius=radius, fill=int(255 * alpha)
    )
    region.putalpha(mask)
    frame.alpha_composite(region, (x0, y0))
    d = ImageDraw.Draw(frame, "RGBA")
    a = int(255 * alpha)
    d.rounded_rectangle((x0, y0, x1, y1), radius=radius, outline=(130, 225, 235, int(85 * alpha)), width=2)
    d.line((x0 + 44, y1 - 2, x0 + 210, y1 - 2), fill=accent + (a,), width=4)
