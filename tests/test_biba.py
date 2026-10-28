from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from biba.audio import build_audio_track, split_clauses, stft_formant_smooth
from biba.cli import build_default_timeline, main
from biba.compositor import ReelCompositor
from biba.decoder import VideoStreamReader
from biba.layout import quantize_sub_lsb_dither, render_color_field
from biba.spec import ReelTimeline


class TestBibaPackage(unittest.TestCase):
    def test_sub_lsb_dither_quantization(self) -> None:
        ramp = np.linspace(10.0, 200.0, 32 * 32 * 3, dtype=np.float32).reshape(32, 32, 3)
        u8_f0 = quantize_sub_lsb_dither(ramp, amplitude=0.5, seed=42)
        u8_f1 = quantize_sub_lsb_dither(ramp, amplitude=0.5, seed=43)
        self.assertEqual(u8_f0.dtype, np.uint8)
        self.assertEqual(u8_f0.shape, (32, 32, 3))
        self.assertTrue(np.allclose(u8_f0.astype(np.float32), ramp, atol=1.0))
        self.assertFalse(np.array_equal(u8_f0, u8_f1))

    def test_timeline_json_roundtrip(self) -> None:
        tl = build_default_timeline()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "timeline.json"
            p.write_text(json.dumps(tl.to_dict()), encoding="utf-8")
            loaded = ReelTimeline.from_json(p)
            self.assertEqual(loaded.reel_id, tl.reel_id)
            self.assertEqual(loaded.width, 1080)
            self.assertEqual(loaded.height, 1920)
            self.assertEqual(loaded.cuts, [0, 75, 234, 420, 585, 720])

    def test_clause_splitter_and_stft_smoothing(self) -> None:
        clauses = split_clauses("First clause, second clause; and final sentence.")
        self.assertEqual(len(clauses), 3)
        sr = 48000
        t = np.linspace(0.0, 0.25, int(sr * 0.25), dtype=np.float32)
        sig = 0.4 * np.sin(2.0 * np.pi * 220.0 * t)
        smoothed = stft_formant_smooth(sig, sr=sr)
        self.assertEqual(smoothed.shape, sig.shape)
        self.assertFalse(np.isnan(smoothed).any())

    def test_color_field_and_audio_mixer(self) -> None:
        img = render_color_field(5, 320, 180)
        self.assertEqual(img.size, (320, 180))
        comp = ReelCompositor()
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "test.wav"
            build_audio_track(
                voice_segments=[],
                duration_sec=2.0,
                out_wav=wav,
                ffmpeg_exe=comp.ffmpeg_exe,
                use_tts=False,
            )
            self.assertTrue(wav.is_file())
            self.assertGreater(wav.stat().st_size, 10000)

    def test_cli_init_and_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "init.json"
            rc1 = main(["init", str(out_json)])
            self.assertEqual(rc1, 0)
            self.assertTrue(out_json.is_file())
            rc2 = main(["probe"])
            self.assertEqual(rc2, 0)

    def test_missing_video_stream_graceful_fallback(self) -> None:
        with VideoStreamReader(Path("nonexistent_clip.mp4"), 320, 180) as reader:
            self.assertIsNone(reader.read_frame())
            frames = list(reader.iter_frames(3))
            self.assertEqual(len(frames), 3)
            self.assertEqual(frames[0].size, (320, 180))


if __name__ == "__main__":
    unittest.main()
