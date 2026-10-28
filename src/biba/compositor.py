from __future__ import annotations
import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

try:
    from .audio import build_audio_track
    from .decoder import decode_video_segment, load_still_cover
    from .layout import (
        AMBER,
        COVER_H,
        COVER_W,
        EMERALD,
        F_BODY,
        F_BODY_L,
        F_HOOK,
        F_HOOK2,
        F_HOOK_MD,
        F_MONO,
        F_MONO_S,
        F_SMALL,
        F_STEP,
        F_TINY,
        F_TITLE,
        FPS,
        GOLD,
        H,
        MUTED,
        TEAL,
        W,
        WHITE,
        clamp,
        draw_card,
        draw_shadow_text,
        ease_in_out,
        ease_out,
        lerp,
        load_sans,
        prog,
        render_color_field,
        text_width,
    )
    from .spec import ReelTimeline
except ImportError:
    from audio import build_audio_track  # type: ignore[no-redef]
    from decoder import decode_video_segment, load_still_cover  # type: ignore[no-redef]
    from layout import (  # type: ignore[no-redef]
        AMBER,
        COVER_H,
        COVER_W,
        EMERALD,
        F_BODY,
        F_BODY_L,
        F_HOOK,
        F_HOOK2,
        F_HOOK_MD,
        F_MONO,
        F_MONO_S,
        F_SMALL,
        F_STEP,
        F_TINY,
        F_TITLE,
        FPS,
        GOLD,
        H,
        MUTED,
        TEAL,
        W,
        WHITE,
        clamp,
        draw_card,
        draw_shadow_text,
        ease_in_out,
        ease_out,
        lerp,
        load_sans,
        prog,
        render_color_field,
        text_width,
    )
    from spec import ReelTimeline  # type: ignore[no-redef]


class ReelCompositor:
    def __init__(self, ffmpeg_exe: Optional[str] = None) -> None:
        self.ffmpeg_exe = ffmpeg_exe or imageio_ffmpeg.get_ffmpeg_exe()
        self._photo_cache: Dict[str, Image.Image] = {}
        self._grad_cache: Dict[Tuple[float, float], Image.Image] = {}
        self._mask_cache: Dict[Tuple[int, int, int], Image.Image] = {}
        self._frame_buf = np.empty((H, W, 3), dtype=np.uint8)
        self._orb = render_color_field(12, 420, 420, TEAL).filter(ImageFilter.GaussianBlur(14))

    def measure_frame_pass_ms(self, frames: int = 30) -> float:
        t0 = time.perf_counter()
        for idx in range(frames):
            _ = render_color_field(idx, 540, 270)
        return round((time.perf_counter() - t0) * 1000.0, 2)

    def _rounded_mask(self, width: int, height: int, radius: int) -> Image.Image:
        key = (width, height, radius)
        if key not in self._mask_cache:
            m = Image.new("L", (width, height), 0)
            ImageDraw.Draw(m).rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, fill=255)
            self._mask_cache[key] = m
        return self._mask_cache[key]

    def _pan_zoom(
        self,
        path: Path,
        t: float,
        z0: float,
        z1: float,
        c0: Tuple[float, float],
        c1: Tuple[float, float],
        blur_radius: int = 0,
        brightness: float = 1.0,
    ) -> Image.Image:
        cache_key = f"{path}_{blur_radius}_{brightness}"
        if cache_key not in self._photo_cache:
            im = load_still_cover(path, COVER_W, COVER_H, TEAL)
            if blur_radius > 0:
                im = im.filter(ImageFilter.GaussianBlur(blur_radius))
            if brightness != 1.0:
                im = ImageEnhance.Brightness(im).enhance(brightness)
            self._photo_cache[cache_key] = im
        im = self._photo_cache[cache_key]
        t = ease_in_out(t)
        z = lerp(z0, z1, t)
        cx = clamp(lerp(c0[0], c1[0], t), 0.05, 0.95) * im.width
        cy = clamp(lerp(c0[1], c1[1], t), 0.05, 0.95) * im.height
        cw, ch = COVER_W / z, COVER_H / z
        if cw > im.width:
            k = im.width / cw
            cw, ch = cw * k, ch * k
        if ch > im.height:
            k = im.height / ch
            cw, ch = cw * k, ch * k
        x0 = clamp(cx - cw / 2, 0, im.width - cw)
        y0 = clamp(cy - ch / 2, 0, im.height - ch)
        return im.resize((W, H), Image.Resampling.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))

    def _vignette(self, frame: Image.Image, top: float, bottom: float) -> Image.Image:
        key = (round(top, 2), round(bottom, 2))
        if key not in self._grad_cache:
            y = np.arange(H, dtype=np.float32)
            a_top = np.where(y < 780.0, top * np.power(np.maximum(0.0, 1.0 - y / 780.0), 1.5), 0.0)
            a_bot = np.where(
                y > (H - 1060.0),
                bottom * np.power(np.maximum(0.0, (y - (H - 1060.0)) / 1060.0), 1.3),
                0.0,
            )
            alpha = np.clip(np.maximum(a_top, a_bot) * 255.0, 0.0, 255.0).astype(np.uint8)[:, None]
            ov = Image.fromarray(alpha, mode="L").resize((W, H), Image.Resampling.NEAREST)
            self._grad_cache[key] = ov
        return Image.composite(Image.new("RGB", (W, H), (4, 8, 12)), frame, self._grad_cache[key])

    def _paste_viewport(
        self,
        fr: Image.Image,
        vframe: Optional[Image.Image],
        y_top: int = 240,
        accent: Tuple[int, int, int] = TEAL,
        label_left: str = "BIBA v1.1",
        label_right: str = "1080x1920 - 30 FPS",
    ) -> None:
        if vframe is None:
            return
        _, vh = vframe.size
        pad = 24
        vx0, vy0, vx1, vy1 = pad, y_top, W - pad, y_top + vh
        cropped = vframe.crop((pad, 0, W - pad, vh)).convert("RGBA")
        mask = self._rounded_mask(cropped.width, cropped.height, 28)
        fr.paste(cropped, (vx0, vy0), mask)

        d = ImageDraw.Draw(fr, "RGBA")
        d.rounded_rectangle((vx0, vy0, vx1, vy1), radius=28, outline=accent + (180,), width=3)
        d.rounded_rectangle((vx0 + 20, vy0 + 18, vx0 + 360, vy0 + 62), radius=14, fill=(5, 10, 16, 205))
        d.ellipse((vx0 + 36, vy0 + 32, vx0 + 52, vy0 + 48), fill=EMERALD + (255,))
        d.text((vx0 + 64, vy0 + 40), label_left, font=F_MONO_S, fill=WHITE + (255,), anchor="lm")

        rw = text_width(F_MONO_S, label_right) + 36
        d.rounded_rectangle((vx1 - 20 - rw, vy0 + 18, vx1 - 20, vy0 + 62), radius=14, fill=(5, 10, 16, 205))
        d.text((vx1 - 38, vy0 + 40), label_right, font=F_MONO_S, fill=accent + (255,), anchor="rm")

    def _draw_progress_bar(self, frame: Image.Image, step: int, f_local: int, accent: Tuple[int, int, int] = TEAL) -> None:
        d = ImageDraw.Draw(frame, "RGBA")
        x, y = 54, 158
        d.text((x, y), f"PART 0{step} // BIBA v1.1", font=F_STEP, fill=WHITE + (245,))
        seg_w, gap = 128, 12
        sx = W - 54 - 3 * seg_w - 2 * gap
        for i in range(3):
            bx = sx + i * (seg_w + gap)
            d.rounded_rectangle((bx, y + 10, bx + seg_w, y + 18), radius=4, fill=(255, 255, 255, 65))
            if i + 1 < step:
                fill_w = seg_w
            elif i + 1 == step:
                fill_w = int(seg_w * ease_out(f_local / 35.0))
            else:
                fill_w = 0
            if fill_w > 0:
                d.rounded_rectangle((bx, y + 10, bx + fill_w, y + 18), radius=4, fill=accent + (255,))

    def _draw_check(self, d: ImageDraw.ImageDraw, x: int, y: int, s: int, fill: Tuple[int, int, int, int], width: int = 5) -> None:
        d.line(
            [(x, y + s * 0.5), (x + s * 0.38, y + s * 0.88), (x + s, y + s * 0.1)],
            fill=fill,
            width=width,
            joint="curve",
        )

    def _flash(self, frame: Image.Image, f_local: int, length: int = 7, strength: float = 0.58) -> Image.Image:
        if f_local >= length:
            return frame
        k = strength * (1.0 - f_local / length) ** 2
        return Image.blend(frame, Image.new("RGB", frame.size, (235, 250, 255)), k)

    def _phone_card(
        self,
        f_global: int,
        h1: str,
        h2: str,
        sub: str,
        accent: Tuple[int, int, int] = TEAL,
    ) -> Image.Image:
        ph_w, ph_h, scr_pad = 540, 1020, 16
        ph = Image.new("RGBA", (ph_w + 80, ph_h + 80), (0, 0, 0, 0))
        g = Image.new("RGBA", ph.size, (0, 0, 0, 0))
        ImageDraw.Draw(g).rounded_rectangle((40, 40, 40 + ph_w, 40 + ph_h), radius=82, fill=accent + (68,))
        ph.alpha_composite(g.filter(ImageFilter.GaussianBlur(28)))

        d = ImageDraw.Draw(ph)
        d.rounded_rectangle(
            (40, 40, 40 + ph_w, 40 + ph_h),
            radius=82,
            fill=(20, 23, 28, 255),
            outline=(135, 145, 158, 255),
            width=3,
        )
        sx0, sy0 = 40 + scr_pad, 40 + scr_pad
        sw, sh = ph_w - 2 * scr_pad, ph_h - 2 * scr_pad
        screen = Image.new("RGB", (sw, sh), (4, 7, 11))

        glow = Image.new("RGB", (sw, sh), 0)
        ImageDraw.Draw(glow).ellipse((sw * 0.1, sh * 0.10, sw * 0.9, sh * 0.58), fill=(12, 65, 72))
        screen = ImageChops.add(screen, glow.filter(ImageFilter.GaussianBlur(70)))

        t = f_global / FPS
        scale = 0.56 + 0.03 * math.sin(t * 2.4)
        b = self._orb.resize(
            (int(self._orb.width * scale), int(self._orb.height * scale)),
            Image.Resampling.BICUBIC,
        )
        b = b.rotate(math.sin(t * 1.4) * 15.0 + t * 12.0, resample=Image.Resampling.BICUBIC)
        bx = (sw - b.width) // 2
        by = int(sh * 0.34 - b.height / 2 + 12.0 * math.sin(t * 2.0))
        tmp = Image.new("RGB", (sw, sh), 0)
        tmp.paste(b, (bx, by))
        screen = ImageChops.screen(screen, tmp)

        sd = ImageDraw.Draw(screen)
        sd.text((36, 62), "BIBA // v1.1", font=load_sans(24, 1), fill=WHITE)
        sd.rounded_rectangle((sw - 210, 56, sw - 34, 92), radius=16, fill=(10, 38, 30), outline=EMERALD, width=2)
        sd.text((sw - 122, 74), "LOCAL BUILD", font=F_MONO_S, fill=EMERALD, anchor="mm")

        sd.text((36, int(sh * 0.63)), h1, font=load_sans(56, 1), fill=WHITE)
        sd.text((36, int(sh * 0.63) + 68), h2, font=load_sans(56, 1), fill=accent)
        sd.text((36, int(sh * 0.63) + 150), sub, font=load_sans(25, 7), fill=MUTED)
        sd.rounded_rectangle((36, sh - 134, 340, sh - 72), radius=30, outline=accent, width=2)
        sd.text((188, sh - 103), "PYTHON + FFMPEG", font=load_sans(24, 10), fill=accent, anchor="mm")

        m = self._rounded_mask(sw, sh, 68)
        ph.paste(screen, (sx0, sy0), m)
        d.rounded_rectangle((40 + ph_w // 2 - 64, sy0 + 18, 40 + ph_w // 2 + 64, sy0 + 52), radius=17, fill=(0, 0, 0, 255))
        return ph

    def render_reel(self, spec: ReelTimeline, output_dir: Path, output_file: Optional[Path] = None) -> Dict[str, object]:
        t0 = time.perf_counter()
        output_dir.mkdir(parents=True, exist_ok=True)
        out_mp4 = output_file if output_file is not None else (output_dir / spec.output_filename)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)

        cuts = spec.cuts if len(spec.cuts) == 6 else [0, 75, 234, 420, 585, 720]
        total_frames = cuts[-1]
        acc = spec.accent_color

        v_act1 = decode_video_segment(spec.video_plate_act1, cuts[1] - cuts[0], 1080, 660, accent=acc, ffmpeg_exe=self.ffmpeg_exe)
        v_act2 = decode_video_segment(spec.video_plate_act2, cuts[2] - cuts[1], 1080, 540, accent=acc, ffmpeg_exe=self.ffmpeg_exe)
        v_act3 = decode_video_segment(spec.video_plate_act3, cuts[3] - cuts[2], 1080, 540, accent=acc, ffmpeg_exe=self.ffmpeg_exe)
        v_act4 = decode_video_segment(spec.video_plate_act4, cuts[4] - cuts[3], 1080, 520, accent=acc, ffmpeg_exe=self.ffmpeg_exe)
        v_act5 = decode_video_segment(spec.video_plate_act5, cuts[5] - cuts[4], 1080, 660, accent=acc, ffmpeg_exe=self.ffmpeg_exe)

        def render_frame_at(fi: int) -> Image.Image:
            if fi < cuts[1]:
                f = fi - cuts[0]
                n = cuts[1] - cuts[0]
                base = self._pan_zoom(spec.photo_plate_act1, f / n, 1.52, 1.18, (0.52, 0.58), (0.50, 0.52))
                fr = self._vignette(base, 0.88, 0.82).convert("RGBA")

                vf = v_act1[f % len(v_act1)] if v_act1 else None
                self._paste_viewport(fr, vf, y_top=640, accent=acc)

                d = ImageDraw.Draw(fr, "RGBA")
                p0 = ease_out(prog(f, 0, 10))
                kw = text_width(F_SMALL, spec.hook_kicker) + 54
                kx0 = (W - kw) // 2
                d.rounded_rectangle((kx0, 146, kx0 + kw, 202), radius=28, fill=(8, 24, 30, int(220 * p0)), outline=acc + (int(255 * p0),), width=2)
                d.text((W // 2, 174), spec.hook_kicker, font=F_SMALL, fill=acc + (int(255 * p0),), anchor="mm")

                p1 = ease_out(prog(f, 3, 12))
                draw_shadow_text(fr, (W // 2, 300 + 36 * (1 - p1)), spec.hook_line1, F_HOOK, WHITE, "mm", alpha=p1)
                draw_shadow_text(fr, (W // 2, 426 + 36 * (1 - p1)), spec.hook_line2, F_HOOK, acc, "mm", alpha=p1)

                p2 = ease_out(prog(f, 18, 14))
                draw_shadow_text(fr, (W // 2, 552 + 24 * (1 - p2)), spec.hook_sub, F_HOOK2, WHITE, "mm", alpha=p2)

                box = (54, 1360, W - 54, 1720)
                draw_card(fr, box, p2, radius=30, accent=acc)
                d = ImageDraw.Draw(fr, "RGBA")
                d.text((W // 2, 1445), "BIBA v1.1 REEL COMPOSITOR", font=F_TITLE, fill=WHITE + (int(255 * p2),), anchor="mm")
                d.text((W // 2, 1525), "Local 9:16 Video & 48 kHz Audio Builder.", font=F_BODY_L, fill=MUTED + (int(255 * p2),), anchor="mm")
                d.text((W // 2, 1615), "PILLOW - NUMPY - FFMPEG", font=F_MONO_S, fill=GOLD + (int(255 * p2),), anchor="mm")
                return self._flash(fr.convert("RGB"), f, 8, 0.72)

            if fi < cuts[2]:
                f = fi - cuts[1]
                n = cuts[2] - cuts[1]
                base = self._pan_zoom(spec.photo_plate_act2, f / n, 1.38, 1.54, (0.46, 0.46), (0.48, 0.42), blur_radius=22, brightness=0.52)
                fr = self._vignette(base, 0.90, 0.92).convert("RGBA")
                self._draw_progress_bar(fr, 1, f, accent=acc)

                vf = v_act2[f % len(v_act2)] if v_act2 else None
                self._paste_viewport(fr, vf, y_top=245, accent=acc)

                p = ease_out(prog(f, 3, 12))
                y_off = int(55 * (1 - p))
                box = (48, 835 + y_off, W - 48, 1715 + y_off)
                draw_card(fr, box, p, radius=34, accent=acc)
                d = ImageDraw.Draw(fr, "RGBA")
                x = box[0] + 46
                d.text((x, box[1] + 42), "01", font=F_TITLE, fill=acc + (int(255 * p),))
                d.text((x + 82, box[1] + 42), spec.step1_title, font=F_TITLE, fill=WHITE + (int(255 * p),))
                d.text((x, box[1] + 118), spec.step1_sub, font=F_BODY_L, fill=MUTED + (int(255 * p),))

                pb = (x, box[1] + 190, box[2] - 46, box[1] + 760)
                d.rounded_rectangle(pb, radius=22, fill=(4, 8, 14, int(195 * p)), outline=(255, 255, 255, int(48 * p)), width=2)
                d.ellipse((pb[0] + 24, pb[1] + 22, pb[0] + 40, pb[1] + 38), fill=(239, 68, 68, int(255 * p)))
                d.ellipse((pb[0] + 50, pb[1] + 22, pb[0] + 66, pb[1] + 38), fill=(245, 158, 11, int(255 * p)))
                d.ellipse((pb[0] + 76, pb[1] + 22, pb[0] + 92, pb[1] + 38), fill=(34, 197, 94, int(255 * p)))
                d.text((pb[0] + 116, pb[1] + 18), spec.terminal_cmd, font=F_MONO_S, fill=EMERALD + (int(255 * p),))
                d.line((pb[0], pb[1] + 56, pb[2], pb[1] + 56), fill=(40, 56, 72, int(180 * p)), width=1)

                total_chars = sum(len(s) for s in spec.terminal_lines)
                typed = int(total_chars * clamp((f - 12) / 105.0))
                yy = pb[1] + 84
                shown = 0
                cursor_pos = None
                for idx_l, s_line in enumerate(spec.terminal_lines):
                    take = int(clamp(typed - shown, 0, len(s_line)))
                    part = s_line[:take]
                    col = GOLD if idx_l == len(spec.terminal_lines) - 1 else (215, 238, 245)
                    d.text((pb[0] + 26, yy), part, font=F_MONO, fill=col + (int(255 * p),))
                    if 0 < take < len(s_line):
                        cursor_pos = (pb[0] + 26 + text_width(F_MONO, part) + 4, yy)
                    shown += len(s_line)
                    yy += 78
                if cursor_pos is None and typed >= total_chars and spec.terminal_lines:
                    cursor_pos = (pb[0] + 26 + text_width(F_MONO, spec.terminal_lines[-1]) + 6, yy - 78)
                if cursor_pos and (f // 7) % 2 == 0:
                    d.rectangle((cursor_pos[0], cursor_pos[1] + 2, cursor_pos[0] + 15, cursor_pos[1] + 36), fill=acc + (int(255 * p),))
                return self._flash(fr.convert("RGB"), f, 6, 0.48)

            if fi < cuts[3]:
                f = fi - cuts[2]
                n = cuts[3] - cuts[2]
                base = self._pan_zoom(spec.photo_plate_act3, f / n, 1.26, 1.44, (0.45, 0.42), (0.54, 0.45), blur_radius=24, brightness=0.50)
                fr = self._vignette(base, 0.90, 0.92).convert("RGBA")
                self._draw_progress_bar(fr, 2, f, accent=acc)

                vf = v_act3[f % len(v_act3)] if v_act3 else None
                self._paste_viewport(fr, vf, y_top=245, accent=acc)

                p = ease_out(prog(f, 3, 12))
                y_off = int(55 * (1 - p))
                box = (48, 835 + y_off, W - 48, 1715 + y_off)
                draw_card(fr, box, p, radius=34, accent=acc)
                d = ImageDraw.Draw(fr, "RGBA")
                x = box[0] + 46
                d.text((x, box[1] + 42), "02", font=F_TITLE, fill=acc + (int(255 * p),))
                d.text((x + 82, box[1] + 42), spec.step2_title, font=F_TITLE, fill=WHITE + (int(255 * p),))
                d.text((x, box[1] + 118), spec.step2_sub, font=F_BODY_L, fill=MUTED + (int(255 * p),))

                cx = x
                for i, tag_str in enumerate(spec.chip_tags):
                    pc = ease_out(prog(f, 10 + i * 6, 9))
                    if pc <= 0:
                        continue
                    w_chip = text_width(F_MONO_S, tag_str) + 34
                    d.rounded_rectangle(
                        (cx, box[1] + 195, cx + w_chip, box[1] + 245),
                        radius=12,
                        fill=(0, 0, 0, int(175 * pc)),
                        outline=GOLD + (int(185 * pc),),
                        width=2,
                    )
                    d.text((cx + 17, box[1] + 220), tag_str, font=F_MONO_S, fill=(255, 240, 190, int(255 * pc)), anchor="lm")
                    cx += w_chip + 12

                yy = box[1] + 305
                for i, item in enumerate(spec.check_items):
                    st = 22 + i * 28
                    pi = ease_out(prog(f, st, 10))
                    if pi <= 0:
                        yy += 108
                        continue
                    done = f > st + 14
                    cxm, cym = x + 26, yy + 28
                    if done:
                        d.ellipse((cxm - 25, cym - 25, cxm + 25, cym + 25), fill=EMERALD + (int(255 * pi),))
                        self._draw_check(d, cxm - 12, cym - 11, 24, (5, 22, 16, 255), 5)
                    else:
                        ang = (f * 26) % 360
                        d.arc((cxm - 23, cym - 23, cxm + 23, cym + 23), ang, ang + 270, fill=acc + (int(255 * pi),), width=5)
                    d.text((x + 76, cym), item, font=F_BODY, fill=WHITE + (int(255 * pi),), anchor="lm")
                    yy += 108
                return self._flash(fr.convert("RGB"), f, 6, 0.48)

            if fi < cuts[4]:
                f = fi - cuts[3]
                n = cuts[4] - cuts[3]
                base = self._pan_zoom(spec.photo_plate_act4, f / n, 1.18, 1.32, (0.50, 0.55), (0.50, 0.52), blur_radius=36, brightness=0.46)
                fr = self._vignette(base, 0.88, 0.94).convert("RGBA")
                self._draw_progress_bar(fr, 3, f, accent=acc)

                ph = self._phone_card(fi, spec.device_headline1, spec.device_headline2, spec.device_sub, accent=acc)
                ang = lerp(-4.5, 2.0, ease_in_out(f / n))
                ph = ph.rotate(ang, resample=Image.Resampling.BICUBIC, expand=False)
                ph_s = lerp(0.66, 0.72, ease_out(f / n))
                ph = ph.resize((int(ph.width * ph_s), int(ph.height * ph_s)), Image.Resampling.BICUBIC)
                px = (W - ph.width) // 2
                py = 215 + int(8 * math.sin(fi / 10.0))
                fr.alpha_composite(ph, (px, py))

                p = ease_out(prog(f, 3, 12))
                box = (48, 995, W - 48, 1745)
                draw_card(fr, box, p, radius=34, accent=acc)
                d = ImageDraw.Draw(fr, "RGBA")
                x = box[0] + 46
                d.text((x, box[1] + 38), "03", font=F_TITLE, fill=acc + (int(255 * p),))
                d.text((x + 82, box[1] + 38), spec.step3_title, font=F_TITLE, fill=WHITE + (int(255 * p),))
                d.text((x, box[1] + 108), spec.step3_sub, font=F_BODY_L, fill=MUTED + (int(255 * p),))

                yy = box[1] + 178
                for idx_c, (who, msg) in enumerate(spec.chat_pairs):
                    st = 14 + idx_c * 26
                    pb = ease_out(prog(f, st, 9))
                    if pb <= 0:
                        yy += 116
                        continue
                    a_b = int(255 * pb)
                    tw = text_width(F_BODY, msg)
                    pad_b = 28
                    extra = 44 if who == "ai" else 0
                    bw = min(box[2] - box[0] - 90, tw + pad_b * 2 + extra)
                    bh = 86
                    y_draw = yy + int(22 * (1 - pb))
                    if who == "user":
                        x1_b = box[2] - 56
                        x0_b = x1_b - bw
                        d.rounded_rectangle((x0_b, y_draw, x1_b, y_draw + bh), radius=28, fill=(238, 245, 250, int(235 * pb)))
                        d.text((x0_b + pad_b, y_draw + bh // 2), msg, font=F_BODY, fill=(10, 18, 24, a_b), anchor="lm")
                    else:
                        x0_b = x
                        x1_b = x0_b + bw
                        d.rounded_rectangle((x0_b, y_draw, x1_b, y_draw + bh), radius=28, fill=(8, 38, 42, int(230 * pb)), outline=acc + (a_b,), width=2)
                        self._draw_check(d, x0_b + pad_b, y_draw + bh // 2 - 12, 24, EMERALD + (a_b,), 5)
                        d.text((x0_b + pad_b + extra, y_draw + bh // 2), msg, font=F_BODY, fill=WHITE + (a_b,), anchor="lm")
                    yy += 116
                return self._flash(fr.convert("RGB"), f, 7, 0.55)

            f = fi - cuts[4]
            n = cuts[5] - cuts[4]
            base = self._pan_zoom(spec.photo_plate_act5, f / n, 1.24, 1.12, (0.50, 0.52), (0.50, 0.48), blur_radius=34, brightness=0.45)
            fr = self._vignette(base, 0.85, 0.95).convert("RGBA")

            vf = v_act5[f % len(v_act5)] if v_act5 else None
            self._paste_viewport(fr, vf, y_top=520, accent=GOLD)

            p1 = ease_out(prog(f, 4, 12))
            draw_shadow_text(fr, (W // 2, 255 + 30 * (1 - p1)), spec.payoff_top1, F_HOOK_MD, WHITE, "mm", alpha=p1)
            draw_shadow_text(fr, (W // 2, 365 + 30 * (1 - p1)), spec.payoff_top2, F_HOOK_MD, GOLD, "mm", alpha=p1)

            p2 = ease_out(prog(f, 16, 12))
            draw_shadow_text(fr, (W // 2, 1310 + 28 * (1 - p2)), spec.cta_line1, F_HOOK2, WHITE, "mm", alpha=p2)
            draw_shadow_text(fr, (W // 2, 1395 + 28 * (1 - p2)), spec.cta_line2, F_HOOK2, acc, "mm", alpha=p2)

            p3 = ease_out(prog(f, 26, 12))
            d = ImageDraw.Draw(fr, "RGBA")
            lw = text_width(F_BODY, spec.cta_pill) + 84
            lx = (W - lw) // 2
            d.rounded_rectangle((lx, 1515, lx + lw, 1608), radius=46, fill=GOLD + (int(245 * p3),))
            d.text((W // 2, 1561), spec.cta_pill, font=F_BODY, fill=(8, 12, 18, int(255 * p3)), anchor="mm")
            d.text(
                (W // 2, 1685),
                "BIBA v1.1 - PYTHON REEL COMPOSITOR",
                font=F_TINY,
                fill=(180, 194, 206, int(220 * p3)),
                anchor="mm",
            )
            out = self._flash(fr.convert("RGB"), f, 8, 0.78)
            if f > n - 8:
                out = Image.blend(out, Image.new("RGB", out.size, 0), (f - (n - 8)) / 8.0 * 0.65)
            return out

        duration_sec = total_frames / FPS
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "audio.wav"
            build_audio_track(
                spec.voice_segments,
                duration_sec,
                wav_path,
                ffmpeg_exe=self.ffmpeg_exe,
                use_tts=spec.use_local_macos_tts,
                fps=FPS,
                external_audio_path=spec.external_audio_path,
            )

            cmd = [
                self.ffmpeg_exe,
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{W}x{H}",
                "-r",
                str(FPS),
                "-i",
                "-",
                "-i",
                str(wav_path),
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-level",
                "4.1",
                "-preset",
                "fast",
                "-crf",
                "16",
                "-g",
                "60",
                "-bf",
                "2",
                "-aq-mode",
                "3",
                "-pix_fmt",
                "yuv420p",
                "-tag:v",
                "avc1",
                "-movflags",
                "+faststart",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                "-ar",
                "48000",
                "-shortest",
                str(out_mp4),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            assert proc.stdin is not None
            sample_frames: List[Image.Image] = []
            sample_indices = {24, 150, 330, 500, 650}
            buf_view = memoryview(self._frame_buf)
            for fi in range(total_frames):
                fr_img = render_frame_at(fi)
                if fi in sample_indices:
                    sample_frames.append(fr_img.copy())
                np.copyto(self._frame_buf, np.asarray(fr_img, dtype=np.uint8))
                proc.stdin.write(buf_view)
            proc.stdin.close()
            proc.stdin = None
            _, stderr_bytes = proc.communicate(timeout=120)
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg render error: {stderr_bytes.decode('utf-8', errors='ignore')}")

            if sample_frames:
                sheet = Image.new("RGB", (270 * len(sample_frames), 480), (6, 10, 16))
                for idx_s, sf in enumerate(sample_frames):
                    thumb = sf.resize((270, 480), Image.Resampling.LANCZOS)
                    sheet.paste(thumb, (idx_s * 270, 0))
                sheet.save(out_mp4.parent / f"{spec.reel_id}_CONTACT_SHEET.jpg", quality=92)

        dt_s = round(time.perf_counter() - t0, 2)
        size_mb = round(out_mp4.stat().st_size / (1024 * 1024), 2)
        return {
            "reel_id": spec.reel_id,
            "output_path": str(out_mp4),
            "file_size_mb": size_mb,
            "resolution": f"{W}x{H}",
            "fps": FPS,
            "duration_sec": duration_sec,
            "render_time_sec": dt_s,
        }


BibaLocalVideoEngine = ReelCompositor
