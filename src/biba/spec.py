from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TEAL: Tuple[int, int, int] = (64, 224, 208)
EMERALD: Tuple[int, int, int] = (34, 197, 94)
GOLD: Tuple[int, int, int] = (255, 214, 10)
AMBER: Tuple[int, int, int] = (255, 176, 92)
WHITE: Tuple[int, int, int] = (255, 255, 255)
MUTED: Tuple[int, int, int] = (182, 194, 206)


@dataclass
class ReelTimeline:
    reel_id: str = "reel_01"
    output_filename: str = "output.mp4"
    voice_segments: List[str] = field(default_factory=list)
    hook_kicker: str = "BIBA v1.1 - VIDEO COMPOSITOR"
    hook_line1: str = "VERTICAL"
    hook_line2: str = "TIMELINE."
    hook_sub: str = "Frame-accurate 9:16 video assembly."
    video_plate_act1: Path = field(default_factory=lambda: Path(""))
    photo_plate_act1: Path = field(default_factory=lambda: Path(""))
    step1_title: str = "Timeline Input"
    step1_sub: str = "Load video clips, still plates, and audio."
    terminal_cmd: str = "biba render examples/product_reel.json -o output/reel.mp4"
    terminal_lines: List[str] = field(default_factory=list)
    video_plate_act2: Path = field(default_factory=lambda: Path(""))
    photo_plate_act2: Path = field(default_factory=lambda: Path(""))
    step2_title: str = "Pipeline Stages"
    step2_sub: str = "Streaming decode, 2D overlay pass, and H.264 encode."
    chip_tags: List[str] = field(default_factory=list)
    check_items: List[str] = field(default_factory=list)
    video_plate_act3: Path = field(default_factory=lambda: Path(""))
    photo_plate_act3: Path = field(default_factory=lambda: Path(""))
    step3_title: str = "Configuration"
    step3_sub: str = "JSON project file or command-line flags."
    chat_pairs: List[Tuple[str, str]] = field(default_factory=list)
    device_headline1: str = "biba"
    device_headline2: str = "v1.1"
    device_sub: str = "9:16 timeline compositor."
    video_plate_act4: Path = field(default_factory=lambda: Path(""))
    photo_plate_act4: Path = field(default_factory=lambda: Path(""))
    payoff_top1: str = "CLEAN TIMELINE."
    payoff_top2: str = "DIRECT EXPORT."
    cta_line1: str = "Pass a JSON timeline or media paths"
    cta_line2: str = "to render 1080x1920 MP4 clips."
    cta_pill: str = "github.com/arian262785-cell/biba-v1.1"
    video_plate_act5: Path = field(default_factory=lambda: Path(""))
    photo_plate_act5: Path = field(default_factory=lambda: Path(""))
    accent_color: Tuple[int, int, int] = TEAL
    use_local_macos_tts: bool = True
    external_audio_path: Optional[Path] = None
    width: int = 1080
    height: int = 1920
    fps: int = 30
    cuts: List[int] = field(default_factory=lambda: [0, 75, 234, 420, 585, 720])

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: Optional[Path] = None) -> "ReelTimeline":
        root = base_dir or Path.cwd()

        def _p(key: str) -> Path:
            raw = data.get(key, "")
            if not raw:
                return Path("")
            p = Path(raw)
            return p if p.is_absolute() else (root / p)

        acc_raw = data.get("accent_color", list(TEAL))
        accent = (int(acc_raw[0]), int(acc_raw[1]), int(acc_raw[2]))
        pairs = [(str(k), str(v)) for k, v in data.get("chat_pairs", [])]
        ext_audio = data.get("external_audio_path")
        ext_path = (_p("external_audio_path") if ext_audio else None)

        return cls(
            reel_id=str(data.get("reel_id", "reel_01")),
            output_filename=str(data.get("output_filename", "output.mp4")),
            voice_segments=[str(s) for s in data.get("voice_segments", [])],
            hook_kicker=str(data.get("hook_kicker", "BIBA v1.1 - VIDEO COMPOSITOR")),
            hook_line1=str(data.get("hook_line1", "VERTICAL")),
            hook_line2=str(data.get("hook_line2", "TIMELINE.")),
            hook_sub=str(data.get("hook_sub", "Frame-accurate 9:16 video assembly.")),
            video_plate_act1=_p("video_plate_act1"),
            photo_plate_act1=_p("photo_plate_act1"),
            step1_title=str(data.get("step1_title", "Timeline Input")),
            step1_sub=str(data.get("step1_sub", "Load video clips, still plates, and audio.")),
            terminal_cmd=str(data.get("terminal_cmd", "biba render examples/product_reel.json -o output/reel.mp4")),
            terminal_lines=[str(s) for s in data.get("terminal_lines", [])],
            video_plate_act2=_p("video_plate_act2"),
            photo_plate_act2=_p("photo_plate_act2"),
            step2_title=str(data.get("step2_title", "Pipeline Stages")),
            step2_sub=str(data.get("step2_sub", "Streaming decode, 2D overlay pass, and H.264 encode.")),
            chip_tags=[str(s) for s in data.get("chip_tags", [])],
            check_items=[str(s) for s in data.get("check_items", [])],
            video_plate_act3=_p("video_plate_act3"),
            photo_plate_act3=_p("photo_plate_act3"),
            step3_title=str(data.get("step3_title", "Configuration")),
            step3_sub=str(data.get("step3_sub", "JSON project file or command-line flags.")),
            chat_pairs=pairs,
            device_headline1=str(data.get("device_headline1", "biba")),
            device_headline2=str(data.get("device_headline2", "v1.1")),
            device_sub=str(data.get("device_sub", "9:16 timeline compositor.")),
            video_plate_act4=_p("video_plate_act4"),
            photo_plate_act4=_p("photo_plate_act4"),
            payoff_top1=str(data.get("payoff_top1", "CLEAN TIMELINE.")),
            payoff_top2=str(data.get("payoff_top2", "DIRECT EXPORT.")),
            cta_line1=str(data.get("cta_line1", "Pass a JSON timeline or media paths")),
            cta_line2=str(data.get("cta_line2", "to render 1080x1920 MP4 clips.")),
            cta_pill=str(data.get("cta_pill", "github.com/arian262785-cell/biba-v1.1")),
            video_plate_act5=_p("video_plate_act5"),
            photo_plate_act5=_p("photo_plate_act5"),
            accent_color=accent,
            use_local_macos_tts=bool(data.get("use_local_macos_tts", True)),
            external_audio_path=ext_path,
            width=int(data.get("width", 1080)),
            height=int(data.get("height", 1920)),
            fps=int(data.get("fps", 30)),
            cuts=[int(c) for c in data.get("cuts", [0, 75, 234, 420, 585, 720])],
        )

    @classmethod
    def from_json(cls, path: Path) -> "ReelTimeline":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw, base_dir=path.parent)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reel_id": self.reel_id,
            "output_filename": self.output_filename,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "cuts": self.cuts,
            "accent_color": list(self.accent_color),
            "use_local_macos_tts": self.use_local_macos_tts,
            "external_audio_path": str(self.external_audio_path) if self.external_audio_path else None,
            "voice_segments": self.voice_segments,
            "hook_kicker": self.hook_kicker,
            "hook_line1": self.hook_line1,
            "hook_line2": self.hook_line2,
            "hook_sub": self.hook_sub,
            "video_plate_act1": str(self.video_plate_act1) if str(self.video_plate_act1) != "." else "",
            "photo_plate_act1": str(self.photo_plate_act1) if str(self.photo_plate_act1) != "." else "",
            "step1_title": self.step1_title,
            "step1_sub": self.step1_sub,
            "terminal_cmd": self.terminal_cmd,
            "terminal_lines": self.terminal_lines,
            "video_plate_act2": str(self.video_plate_act2) if str(self.video_plate_act2) != "." else "",
            "photo_plate_act2": str(self.photo_plate_act2) if str(self.photo_plate_act2) != "." else "",
            "step2_title": self.step2_title,
            "step2_sub": self.step2_sub,
            "chip_tags": self.chip_tags,
            "check_items": self.check_items,
            "video_plate_act3": str(self.video_plate_act3) if str(self.video_plate_act3) != "." else "",
            "photo_plate_act3": str(self.photo_plate_act3) if str(self.photo_plate_act3) != "." else "",
            "step3_title": self.step3_title,
            "step3_sub": self.step3_sub,
            "chat_pairs": [list(p) for p in self.chat_pairs],
            "device_headline1": self.device_headline1,
            "device_headline2": self.device_headline2,
            "device_sub": self.device_sub,
            "video_plate_act4": str(self.video_plate_act4) if str(self.video_plate_act4) != "." else "",
            "photo_plate_act4": str(self.photo_plate_act4) if str(self.photo_plate_act4) != "." else "",
            "payoff_top1": self.payoff_top1,
            "payoff_top2": self.payoff_top2,
            "cta_line1": self.cta_line1,
            "cta_line2": self.cta_line2,
            "cta_pill": self.cta_pill,
            "video_plate_act5": str(self.video_plate_act5) if str(self.video_plate_act5) != "." else "",
            "photo_plate_act5": str(self.photo_plate_act5) if str(self.photo_plate_act5) != "." else "",
        }


CinemaReelSpec = ReelTimeline
