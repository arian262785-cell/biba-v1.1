from __future__ import annotations
import argparse
import json
import platform
import sys
from pathlib import Path
from typing import List, Optional

try:
    from .compositor import ReelCompositor
    from .spec import ReelTimeline
except ImportError:
    from compositor import ReelCompositor  # type: ignore[no-redef]
    from spec import ReelTimeline  # type: ignore[no-redef]


def build_default_timeline() -> ReelTimeline:
    return ReelTimeline(
        reel_id="product_reel",
        output_filename="product_reel.mp4",
        voice_segments=[
            "Biba is a command line timeline compositor for vertical video.",
            "It decodes local video clips over FFmpeg pipes and layers 2D cards in Pillow.",
            "Audio can be loaded from an external file or synthesized at forty-eight kilohertz.",
            "All timeline sections, text fields, and media paths are configured in a JSON file.",
            "Run biba render with your timeline file to export an H.264 MP4.",
        ],
        hook_kicker="BIBA v1.1 - TIMELINE COMPOSITOR",
        hook_line1="VERTICAL",
        hook_line2="VIDEO BUILD.",
        hook_sub="Assemble 9:16 clips from JSON timelines.",
        step1_title="Timeline Input",
        step1_sub="Pass a JSON timeline or media paths on the CLI.",
        terminal_cmd="biba render examples/product_reel.json -o output/reel.mp4",
        terminal_lines=[
            "input: examples/product_reel.json",
            "video: 1080x1920 @ 30 fps (rgb24 pipe)",
            "audio: 48000 Hz stereo PCM -> AAC 256k",
            "encoder: libx264 (high, crf 16)",
            "status: complete",
        ],
        step2_title="Pipeline Stages",
        step2_sub="Streaming frame decode, 2D overlay pass, and H.264 encode.",
        chip_tags=["FFmpeg Pipe", "Pillow RGBA", "NumPy STFT", "48 kHz Audio"],
        check_items=[
            "Streaming RGB24 video clip decoder",
            "Pre-allocated frame buffer to FFmpeg stdin",
            "External WAV/MP3 import or synthesized backing track",
            "Clause-segmented offline narration with STFT smoothing",
        ],
        step3_title="Configuration",
        step3_sub="Use JSON timelines for repeatable builds.",
        chat_pairs=[
            ("user", "How do I pass my own video clips and audio?"),
            ("ai", "Set the plate paths in your JSON file or pass --video and --audio."),
            ("user", "What happens if no video clip is provided?"),
            ("ai", "The compositor generates a procedural color field background."),
        ],
        device_headline1="biba",
        device_headline2="v1.1",
        device_sub="9:16 timeline compositor.",
        payoff_top1="JSON TIMELINE.",
        payoff_top2="DIRECT MP4.",
        cta_line1="Run biba init to generate a template",
        cta_line2="and render your own 9:16 video.",
        cta_pill="github.com/arian262785-cell/biba-v1.1",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="biba",
        description="Assemble 1080x1920 vertical video reels from JSON timelines or media files.",
    )
    sub = parser.add_subparsers(dest="command")

    p_render = sub.add_parser("render", help="Render a timeline JSON file or CLI arguments to MP4")
    p_render.add_argument("timeline", nargs="?", type=Path, help="Path to timeline JSON file")
    p_render.add_argument("-o", "--output", type=Path, default=Path("output/reel.mp4"), help="Output MP4 file path")
    p_render.add_argument("--video", type=Path, help="Override video clip path for all sections")
    p_render.add_argument("--photo", type=Path, help="Override still background image path")
    p_render.add_argument("--audio", type=Path, help="External audio file (.wav, .mp3, .m4a)")
    p_render.add_argument("--title", type=str, help="Override main title line")
    p_render.add_argument("--subtitle", type=str, help="Override subtitle line")
    p_render.add_argument("--no-tts", action="store_true", help="Disable spoken narration")

    p_init = sub.add_parser("init", help="Write a starter timeline JSON file")
    p_init.add_argument("output", nargs="?", type=Path, default=Path("timeline.json"), help="Target JSON path")

    sub.add_parser("probe", help="Print FFmpeg binary path and 30-frame render timing")

    parser.add_argument("--info", action="store_true", help="Alias for 'biba probe'")
    parser.add_argument("--render-sample", action="store_true", help="Render default template to --out directory")
    parser.add_argument("--out", type=Path, default=Path("./output"), help="Output directory for --render-sample")

    args = parser.parse_args(argv)
    compositor = ReelCompositor()

    if args.command == "init":
        tl = build_default_timeline()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(tl.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"initialized": str(args.output)}, indent=2))
        return 0

    if args.command == "probe" or args.info:
        frame_ms = compositor.measure_frame_pass_ms(frames=30)
        info = {
            "tool": "biba v1.1",
            "architecture": platform.machine(),
            "os": platform.platform(),
            "ffmpeg_binary": compositor.ffmpeg_exe,
            "color_field_30_frames_ms": frame_ms,
        }
        print(json.dumps(info, indent=2))
        return 0

    if args.command == "render" or args.render_sample:
        if getattr(args, "timeline", None) is not None:
            tl = ReelTimeline.from_json(args.timeline)
        else:
            tl = build_default_timeline()

        if getattr(args, "video", None) is not None:
            tl.video_plate_act1 = args.video
            tl.video_plate_act2 = args.video
            tl.video_plate_act3 = args.video
            tl.video_plate_act4 = args.video
            tl.video_plate_act5 = args.video
        if getattr(args, "photo", None) is not None:
            tl.photo_plate_act1 = args.photo
            tl.photo_plate_act2 = args.photo
            tl.photo_plate_act3 = args.photo
            tl.photo_plate_act4 = args.photo
            tl.photo_plate_act5 = args.photo
        if getattr(args, "audio", None) is not None:
            tl.external_audio_path = args.audio
        if getattr(args, "title", None) is not None:
            tl.hook_line1 = args.title
        if getattr(args, "subtitle", None) is not None:
            tl.hook_sub = args.subtitle
        if getattr(args, "no_tts", False):
            tl.use_local_macos_tts = False

        if args.render_sample:
            res = compositor.render_reel(tl, args.out)
        else:
            out_file: Path = args.output
            res = compositor.render_reel(tl, out_file.parent, output_file=out_file)
        print(json.dumps(res, indent=2))
        return 0

    parser.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
