import json
import logging
import subprocess
from pathlib import Path

from modules.audio_sync import TimeRegion

logger = logging.getLogger(__name__)


def create_variable_speed_video(
    video_path: str,
    regions: list[TimeRegion],
    output_path: str,
) -> str:
    """Create a video with variable playback speed per region.

    Uses a single ffmpeg filter graph: split -> trim+setpts per region -> concat.
    Returns the original video_path unchanged if no slowdowns are needed.
    """
    needs_processing = any(r.video_speed < 0.99 for r in regions)
    if not needs_processing:
        logger.info("No video speed changes needed")
        return video_path

    # Filter out negligible regions (< 30ms)
    valid = [r for r in regions if r.end - r.start > 0.03]
    n = len(valid)

    if n == 0:
        return video_path

    # Build ffmpeg filter_complex:
    #   [0:v]split=N[v0]...[vN-1];
    #   [v0]trim=S:E,setpts=(PTS-STARTPTS)*F[s0];
    #   ...
    #   [s0]...[sN-1]concat=n=N:v=1:a=0[out]
    parts = []

    # Split input into N copies
    split_outputs = "".join(f"[v{i}]" for i in range(n))
    parts.append(f"[0:v]split={n}{split_outputs}")

    # Trim + setpts for each region
    for i, r in enumerate(valid):
        pts_factor = 1.0 / r.video_speed  # >1.0 = slow motion
        parts.append(
            f"[v{i}]trim={r.start:.3f}:{r.end:.3f},"
            f"setpts=(PTS-STARTPTS)*{pts_factor:.4f}[s{i}]"
        )

    # Concatenate
    concat_inputs = "".join(f"[s{i}]" for i in range(n))
    parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[out]")

    filter_complex = ";".join(parts)

    slowed = sum(1 for r in valid if r.video_speed < 0.99)
    logger.info(f"Processing video: {n} regions ({slowed} slowed)")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Variable-speed video failed: {result.stderr}")

    if not Path(output_path).exists():
        raise FileNotFoundError(f"Speed-adjusted video not created: {output_path}")

    return output_path


def merge_audio_video(video_path: str, audio_path: str, output_path: str) -> str:
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg merge failed: {result.stderr}")

    if not Path(output_path).exists():
        raise FileNotFoundError(f"Output video not created: {output_path}")

    return output_path


def verify_output(output_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    probe = json.loads(result.stdout)

    streams = probe.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    if not has_video:
        raise RuntimeError("Output video has no video stream")
    if not has_audio:
        raise RuntimeError("Output video has no audio stream")

    return {
        "streams": len(streams),
        "has_video": has_video,
        "has_audio": has_audio,
    }
