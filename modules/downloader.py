import subprocess
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yt_dlp


@dataclass
class VideoMetadata:
    title: str
    duration: float
    video_id: str
    uploader: str
    description: str


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def get_metadata(url: str) -> VideoMetadata:
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return VideoMetadata(
        title=info.get("title", "Unknown"),
        duration=float(info.get("duration", 0)),
        video_id=info.get("id", extract_video_id(url)),
        uploader=info.get("uploader", "Unknown"),
        description=info.get("description", ""),
    )


def download_video(
    url: str,
    output_dir: str,
    progress_cb: Callable[[float], None] | None = None,
) -> str:
    output_template = str(Path(output_dir) / "video.%(ext)s")

    def _progress_hook(d: dict):
        if progress_cb and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                progress_cb(downloaded / total)

    opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_progress_hook],
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # Ensure .mp4 extension
        mp4_path = Path(filename).with_suffix(".mp4")
        if mp4_path.exists():
            return str(mp4_path)
        # Fallback: find any video file in the output dir
        for f in Path(output_dir).iterdir():
            if f.suffix in (".mp4", ".mkv", ".webm"):
                return str(f)
        raise FileNotFoundError(f"Downloaded video not found in {output_dir}")


def extract_audio(video_path: str, output_dir: str) -> str:
    output_path = str(Path(output_dir) / "audio.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path
