from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class VideoMeta:
    duration_sec: float
    fps: float
    width: int
    height: int
    codec: Optional[str]


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{p.stderr[:400]}")
    return p.stdout


def probe_video(path: Path) -> VideoMeta:
    """
    Use ffprobe to extract minimal stream/container metadata.
    """
    out = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name,r_frame_rate,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    j = json.loads(out)
    stream = (j.get("streams") or [{}])[0] or {}
    fmt = j.get("format") or {}

    def _parse_rate(s: str) -> float:
        if not s or s == "0/0":
            return 0.0
        if "/" in s:
            a, b = s.split("/", 1)
            try:
                return float(a) / float(b)
            except Exception:
                return 0.0
        try:
            return float(s)
        except Exception:
            return 0.0

    duration = float(fmt.get("duration") or 0.0)
    fps = _parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0")
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    codec = stream.get("codec_name")
    return VideoMeta(duration_sec=duration, fps=fps, width=width, height=height, codec=codec)


def extract_frames(
    *,
    video_path: Path,
    out_dir: Path,
    sample_fps: float,
    max_frames: int,
    image_ext: str = "jpg",
) -> list[Path]:
    """
    Extract frames with ffmpeg.
    - Use fps filter + max_frames cap.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"frame_%06d.{image_ext}"

    # -vsync vfr to respect fps filter output
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={sample_fps}",
        "-vsync",
        "vfr",
        "-frames:v",
        str(max_frames),
        str(pattern),
    ]
    _run(cmd)

    return sorted(out_dir.glob(f"frame_*.{image_ext}"))

