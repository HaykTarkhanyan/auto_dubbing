import json
import subprocess
from pathlib import Path


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
