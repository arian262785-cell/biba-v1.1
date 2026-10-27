from __future__ import annotations
import math
import random
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import List, Optional
import numpy as np

try:
    from .decoder import decode_audio_file
except ImportError:
    from decoder import decode_audio_file  # type: ignore[no-redef]


def split_clauses(text: str) -> List[str]:
    parts = re.split(r"([,;:\u2014\-\.!\?]+)", text.strip())
    clauses: List[str] = []
    cur = ""
    for part in parts:
        if not part:
            continue
        cur += part
        if re.fullmatch(r"[,;:\u2014\-\.!\?]+", part):
            if cur.strip():
                clauses.append(cur.strip())
            cur = ""
    if cur.strip():
        clauses.append(cur.strip())
    return clauses if clauses else [text.strip()]


def stft_formant_smooth(signal: np.ndarray, sr: int = 48000) -> np.ndarray:
    if signal.size < 2048:
        return signal

    n_fft = 2048
    hop = 512
    win = np.hanning(n_fft).astype(np.float32)
    pad_len = n_fft
    padded = np.pad(signal.astype(np.float32), (pad_len, pad_len), mode="reflect")
    n_frames = 1 + (padded.size - n_fft) // hop

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)
    chest_boost = 1.0 + 0.28 * np.exp(-((freqs - 195.0) / 85.0) ** 2)
    presence_curve = 1.0 + 0.14 * np.exp(-((freqs - 2850.0) / 650.0) ** 2)
    de_harsh = 1.0 - 0.24 * np.exp(-((freqs - 4600.0) / 1100.0) ** 2)
    roll_off = 1.0 / (1.0 + (freqs / 11200.0) ** 4)
    eq_gain = (chest_boost * presence_curve * de_harsh * roll_off).astype(np.float32)

    out = np.zeros_like(padded, dtype=np.float32)
    norm = np.zeros_like(padded, dtype=np.float32)
    prev_mag: Optional[np.ndarray] = None

    for m in range(n_frames):
        start = m * hop
        frame = padded[start : start + n_fft] * win
        spec = np.fft.rfft(frame)
        mag = np.abs(spec).astype(np.float32)
        phase = np.angle(spec).astype(np.float32)

        if prev_mag is not None:
            mag = 0.72 * mag + 0.28 * prev_mag
        prev_mag = mag

        mag_eq = mag * eq_gain
        reconstructed = np.fft.irfft(mag_eq * np.exp(1j * phase), n=n_fft).astype(np.float32)
        out[start : start + n_fft] += reconstructed * win
        norm[start : start + n_fft] += win * win

    norm = np.maximum(norm, 1e-6)
    restored = out / norm
    return restored[pad_len : pad_len + signal.size]


def apply_vocal_warmth(signal: np.ndarray, sr: int = 48000) -> np.ndarray:
    if signal.size < 256:
        return signal

    n = signal.size
    t = np.arange(n, dtype=np.float32) / float(sr)
    drift = 0.0012 * np.sin(2.0 * np.pi * 1.35 * t) + 0.0006 * np.cos(2.0 * np.pi * 2.8 * t)
    warped_idx = np.clip(np.arange(n, dtype=np.float32) + drift * sr * 0.015, 0.0, float(n - 1))
    idx0 = np.floor(warped_idx).astype(np.int32)
    idx1 = np.minimum(idx0 + 1, n - 1)
    frac = warped_idx - idx0
    modulated = (1.0 - frac) * signal[idx0] + frac * signal[idx1]

    smoothed = stft_formant_smooth(modulated, sr=sr)

    d1 = int(0.011 * sr)
    d2 = int(0.019 * sr)
    room = smoothed.copy()
    if n > d1:
        room[d1:] += 0.045 * smoothed[:-d1]
    if n > d2:
        room[d2:] += 0.025 * smoothed[:-d2]

    return np.tanh(room * 1.18).astype(np.float32)


def _read_wav_mono_float(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    non_silent = np.where(np.abs(arr) > 0.012)[0]
    if non_silent.size > 10:
        arr = arr[max(0, non_silent[0] - 320) : min(arr.size, non_silent[-1] + 640)]
    fade_len = min(360, arr.size // 6)
    if fade_len > 8:
        arr[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        arr[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    return arr


def synthesize_voice_segment(
    text: str,
    tmpdir: Path,
    seg_idx: int,
    ffmpeg_exe: str,
    sr: int = 48000,
    voice_name: str = "Samantha",
) -> np.ndarray:
    piper_bin = shutil.which("piper")
    local_models = list(Path.cwd().glob("voices/*.onnx"))
    if piper_bin and local_models:
        out_wav = tmpdir / f"piper_{seg_idx}.wav"
        proc = subprocess.run(
            [piper_bin, "--model", str(local_models[0]), "--output_file", str(out_wav)],
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=12,
        )
        if proc.returncode == 0 and out_wav.exists():
            resampled = tmpdir / f"piper_{seg_idx}_48k.wav"
            subprocess.run(
                [ffmpeg_exe, "-y", "-i", str(out_wav), "-ar", str(sr), "-ac", "1", str(resampled)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            return apply_vocal_warmth(_read_wav_mono_float(resampled), sr=sr)

    say_bin = shutil.which("say") or ("/usr/bin/say" if Path("/usr/bin/say").exists() else None)
    espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")
    if not say_bin and not espeak_bin:
        return np.zeros((0,), dtype=np.float32)

    clauses = split_clauses(text)
    clause_chunks: List[np.ndarray] = []
    pause_samples = int(0.095 * sr)

    for c_idx, clause in enumerate(clauses):
        raw_path = tmpdir / f"seg_{seg_idx}_c_{c_idx}.aiff"
        wav_path = tmpdir / f"seg_{seg_idx}_c_{c_idx}.wav"
        rate = str(184 + (c_idx % 3) * 3)
        if say_bin:
            subprocess.run(
                [say_bin, "-v", voice_name, "-r", rate, "-o", str(raw_path), clause],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        else:
            raw_path = tmpdir / f"seg_{seg_idx}_c_{c_idx}_espeak.wav"
            subprocess.run(
                [str(espeak_bin), "-s", rate, "-w", str(raw_path), clause],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i",
                str(raw_path),
                "-af",
                "highpass=f=80,lowpass=f=12000,acompressor=threshold=-16dB:ratio=2.2",
                "-ar",
                str(sr),
                "-ac",
                "1",
                "-f",
                "wav",
                str(wav_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        chunk = _read_wav_mono_float(wav_path)
        if chunk.size > 0:
            clause_chunks.append(chunk)
            if c_idx < len(clauses) - 1:
                clause_chunks.append(np.zeros((pause_samples,), dtype=np.float32))

    if not clause_chunks:
        return np.zeros((0,), dtype=np.float32)

    joined = np.concatenate(clause_chunks)
    processed = apply_vocal_warmth(joined, sr=sr)
    peak = float(np.max(np.abs(processed))) if processed.size else 1.0
    if peak > 0.01:
        processed = (processed / peak) * 0.86
    return processed


def build_audio_track(
    voice_segments: List[str],
    duration_sec: float,
    out_wav: Path,
    ffmpeg_exe: str,
    use_tts: bool = True,
    fps: int = 30,
    external_audio_path: Optional[Path] = None,
) -> None:
    sr = 48000
    total_samples = int(duration_sec * sr)

    ext_pcm = decode_audio_file(external_audio_path, sample_rate=sr, ffmpeg_exe=ffmpeg_exe) if external_audio_path else None
    if ext_pcm is not None and ext_pcm.shape[0] > 0:
        if ext_pcm.shape[0] < total_samples:
            reps = (total_samples // ext_pcm.shape[0]) + 1
            ext_pcm = np.tile(ext_pcm, (reps, 1))
        l_arr = ext_pcm[:total_samples, 0].astype(np.float32)
        r_arr = ext_pcm[:total_samples, 1].astype(np.float32)
    else:
        left = [0.0] * total_samples
        right = [0.0] * total_samples
        random.seed(42)

        def add_sample(idx: int, l_val: float, r_val: Optional[float] = None) -> None:
            if 0 <= idx < total_samples:
                left[idx] += l_val
                right[idx] += l_val if r_val is None else r_val

        for i in range(total_samples):
            t = i / sr
            sw = 0.5 + 0.5 * math.sin(t * 0.45)
            amp = 0.055 + 0.03 * sw
            if t > 18.5:
                amp += 0.035 * min(1.0, (t - 18.5) / 1.8)
            if t > duration_sec - 1.2:
                amp *= max(0.0, (duration_sec - t) / 1.2)
            val = amp * (
                math.sin(2.0 * math.pi * 55.0 * t)
                + 0.6 * math.sin(2.0 * math.pi * 82.41 * t + 0.3 * math.sin(t * 0.7))
            )
            add_sample(i, val, val * 0.97)

        chords = [
            (220.0, 261.63, 329.63),
            (174.61, 220.0, 261.63),
            (196.0, 261.63, 329.63),
            (196.0, 246.94, 293.66),
        ]
        for i in range(int(2.2 * sr), total_samples):
            t = i / sr
            ci = int((t - 2.2) // 4.0) % 4
            ct = (t - 2.2) % 4.0
            fade = min(1.0, ct / 0.5) * min(1.0, (4.0 - ct) / 0.5 + 1e-4)
            amp = 0.028 * min(1.0, (t - 2.2) / 1.8) * fade
            if t > 19.5:
                amp *= 1.55
            vl = vr = 0.0
            for k, freq in enumerate(chords[ci]):
                vl += math.sin(2.0 * math.pi * freq * t * 1.0015 + k)
                vr += math.sin(2.0 * math.pi * freq * t * 0.9985 + k * 1.7)
            add_sample(i, amp * vl, amp * vr)

        def add_kick(t0: float, gain: float = 0.52) -> None:
            s = int(t0 * sr)
            ph = 0.0
            for j in range(int(0.42 * sr)):
                x = j / sr
                f = 45.0 + 110.0 * math.exp(-x * 30.0)
                ph += 2.0 * math.pi * f / sr
                add_sample(s + j, gain * math.sin(ph) * math.exp(-x * 7.2))

        def add_hit(t0: float, gain: float = 0.82) -> None:
            add_kick(t0, gain)
            s = int(t0 * sr)
            lp = 0.0
            for j in range(int(1.4 * sr)):
                x = j / sr
                lp += 0.04 * (random.uniform(-1.0, 1.0) - lp)
                add_sample(
                    s + j,
                    gain * 0.85 * lp * math.exp(-x * 2.6),
                    gain * 0.85 * lp * math.exp(-x * 2.4),
                )
            for j in range(int(1.8 * sr)):
                x = j / sr
                add_sample(s + j, gain * 0.34 * math.sin(2.0 * math.pi * 36.0 * x) * math.exp(-x * 1.6))

        def add_sweep(t_peak: float, length: float = 0.45, gain: float = 0.26) -> None:
            s = int((t_peak - length) * sr)
            lp = 0.0
            n = int(length * 1.3 * sr)
            for j in range(n):
                x = j / n
                env = (x / 0.77) ** 2 if x < 0.77 else math.exp(-(x - 0.77) * 18.0)
                coef = 0.02 + 0.25 * min(1.0, x / 0.77)
                lp += coef * (random.uniform(-1.0, 1.0) - lp)
                add_sample(s + j, gain * env * lp * (1.0 - x * 0.6), gain * env * lp * (0.4 + x * 0.6))

        def add_click(t0: float, gain: float = 0.06, bright: float = 0.5) -> None:
            s = int(t0 * sr)
            hp_prev = 0.0
            for j in range(int(0.028 * sr)):
                r = random.uniform(-1.0, 1.0)
                hp = r - bright * hp_prev
                hp_prev = r
                env = math.exp(-j / sr * 165.0)
                add_sample(s + j, gain * hp * env, gain * hp * env * 0.82)

        def add_riser(t0: float, t1: float, gain: float = 0.24) -> None:
            s, e_ = int(t0 * sr), int(t1 * sr)
            lp = 0.0
            ph = 0.0
            for i in range(s, e_):
                x = (i - s) / max(1, e_ - s)
                lp += (0.02 + 0.3 * x) * (random.uniform(-1.0, 1.0) - lp)
                ph += 2.0 * math.pi * (200.0 + 920.0 * x * x) / sr
                val = gain * x * x * (0.7 * lp + 0.25 * math.sin(ph))
                add_sample(i, val, val)

        add_hit(0.0, 0.88)
        add_sweep(2.5, 0.45, 0.28)
        add_hit(2.5, 0.58)
        add_sweep(7.8, 0.48, 0.30)
        add_hit(7.8, 0.60)
        add_sweep(14.0, 0.48, 0.30)
        add_hit(14.0, 0.58)
        add_riser(17.4, 19.5, 0.26)
        add_hit(19.5, 0.95)

        t_pulse = 2.5
        while t_pulse < 19.0:
            add_kick(t_pulse, 0.38)
            add_click(t_pulse + 0.25, 0.055, 0.82)
            t_pulse += 0.5

        for k in range(75):
            tt = 3.1 + k * 0.042 + random.uniform(-0.006, 0.006)
            if tt < 6.8:
                add_click(tt, 0.048, 0.35)

        for fr_chk in [234 + 34 + k * 28 for k in range(4)]:
            s0 = fr_chk / fps
            st = int(s0 * sr)
            for j in range(int(0.12 * sr)):
                x = j / sr
                val = 0.075 * math.sin(2.0 * math.pi * 1318.5 * x) * math.exp(-x * 30.0)
                add_sample(st + j, val, val)

        l_arr = np.array(left, dtype=np.float32)
        r_arr = np.array(right, dtype=np.float32)

    act_starts = [0.15, 2.65, 7.95, 14.15, 19.65]
    act_max_durs = [2.25, 5.05, 5.90, 5.20, 4.20]
    voice_track = np.zeros((total_samples,), dtype=np.float32)

    if use_tts and voice_segments:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for idx, seg_text in enumerate(voice_segments[:5]):
                arr = synthesize_voice_segment(
                    seg_text,
                    tmpdir=tmp_path,
                    seg_idx=idx,
                    ffmpeg_exe=ffmpeg_exe,
                    sr=sr,
                )
                if arr.size > 0:
                    s_idx = int(act_starts[idx] * sr)
                    max_len = int(act_max_durs[idx] * sr)
                    clip_len = min(arr.size, max_len, total_samples - s_idx)
                    if clip_len > 0:
                        voice_track[s_idx : s_idx + clip_len] = arr[:clip_len]

    peak_score = max(float(np.max(np.abs(l_arr))), float(np.max(np.abs(r_arr))), 1e-6)
    l_arr = (l_arr / peak_score) * 0.44
    r_arr = (r_arr / peak_score) * 0.44

    vo_env = np.convolve(np.abs(voice_track), np.ones(3200) / 3200.0, mode="same")
    duck = np.clip(1.0 - 0.58 * (vo_env > 0.035).astype(np.float32), 0.36, 1.0)

    mix_l = np.tanh((l_arr * duck + voice_track) * 1.12) * 0.91
    mix_r = np.tanh((r_arr * duck + voice_track) * 1.12) * 0.91

    stereo = np.empty((total_samples * 2,), dtype=np.int16)
    stereo[0::2] = (mix_l * 32767.0).astype(np.int16)
    stereo[1::2] = (mix_r * 32767.0).astype(np.int16)

    with wave.open(str(out_wav), "wb") as out_wf:
        out_wf.setnchannels(2)
        out_wf.setsampwidth(2)
        out_wf.setframerate(sr)
        out_wf.writeframes(stereo.tobytes())
