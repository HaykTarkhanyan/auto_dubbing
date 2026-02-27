import subprocess
import tempfile
import os
from pydub import AudioSegment


def speed_change(audio: AudioSegment, factor: float) -> AudioSegment:
    if abs(factor - 1.0) < 0.01:
        return audio

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as inp, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
        input_path = inp.name
        output_path = out.name

    try:
        audio.export(input_path, format="wav")

        # ffmpeg atempo only accepts [0.5, 100.0], chain filters for extreme values
        atempo_filters = _build_atempo_chain(factor)
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter:a", atempo_filters,
            "-vn", output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        return AudioSegment.from_wav(output_path)
    finally:
        for p in (input_path, output_path):
            if os.path.exists(p):
                os.unlink(p)


def _build_atempo_chain(factor: float) -> str:
    # atempo accepts [0.5, 100.0]; chain multiple for extreme values
    filters = []
    remaining = factor
    while remaining > 100.0:
        filters.append("atempo=100.0")
        remaining /= 100.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)


def generate_silence(duration_ms: int, sample_rate: int = 44100) -> AudioSegment:
    return AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)
