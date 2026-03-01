import json
import logging
import shutil
import subprocess
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yt_dlp

logger = logging.getLogger(__name__)


def _download_file(url: str, dest: str, timeout: int = 60):
    """Download a URL to a local file with a timeout."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)


@dataclass
class VideoMetadata:
    title: str
    duration: float
    video_id: str
    uploader: str
    description: str
    channel_url: str = ""


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
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
        channel_url=info.get("channel_url", ""),
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
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
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


def get_video_duration(video_path: str) -> float:
    """Get exact video duration via ffprobe."""
    info = _ffprobe_json(video_path, show_streams=False)
    duration = float(info["format"]["duration"])
    logger.info(f"Actual video duration (ffprobe): {duration:.2f}s")
    return duration


@dataclass
class VideoFileInfo:
    width: int
    height: int
    file_size_bytes: int


def get_video_file_info(video_path: str) -> VideoFileInfo:
    """Get resolution and file size of a video file."""
    info = _ffprobe_json(video_path, show_streams=True)
    width = 0
    height = 0
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            width = int(stream.get("width", 0))
            height = int(stream.get("height", 0))
            break
    file_size = Path(video_path).stat().st_size
    return VideoFileInfo(width=width, height=height, file_size_bytes=file_size)


def _ffprobe_json(video_path: str, show_streams: bool = False) -> dict:
    """Run ffprobe and return parsed JSON."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
    ]
    if show_streams:
        cmd.append("-show_streams")
    cmd.append(video_path)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed for {video_path}: {e.stderr}") from e
    return json.loads(result.stdout)


def download_thumbnail(url: str, output_path: str) -> str:
    """Download the highest quality thumbnail for a YouTube video."""
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    video_id = info.get("id", extract_video_id(url))

    # Try thumbnails in quality order (maxresdefault > sddefault > hqdefault)
    thumbnail_urls = [
        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/sddefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
    ]

    # Also check yt-dlp's thumbnail list for original quality
    thumbnails = info.get("thumbnails", [])
    if thumbnails:
        # Sort by preference/resolution (highest last in yt-dlp)
        best = sorted(thumbnails, key=lambda t: t.get("preference", 0))
        if best:
            thumbnail_urls.insert(0, best[-1]["url"])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    for thumb_url in thumbnail_urls:
        try:
            # Determine extension from URL
            ext = ".jpg"
            if ".webp" in thumb_url:
                ext = ".webp"
            elif ".png" in thumb_url:
                ext = ".png"

            final_path = output.with_suffix(ext)
            _download_file(thumb_url, str(final_path), timeout=30)

            # Verify it's not a placeholder (YouTube returns a tiny default image for missing thumbs)
            size = final_path.stat().st_size
            if size > 10_000:  # Real thumbnails are >10KB
                logger.info(f"Downloaded thumbnail ({size // 1024}KB): {final_path}")
                return str(final_path)
            else:
                final_path.unlink()
        except Exception:
            continue

    raise FileNotFoundError(f"Could not download thumbnail for {video_id}")


def extract_audio(video_path: str, output_dir: str) -> str:
    output_path = str(Path(output_dir) / "audio.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path
