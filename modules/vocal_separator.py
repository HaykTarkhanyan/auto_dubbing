import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def separate_vocals(audio_path: str, output_dir: str) -> str:
    """Separate vocals from audio using demucs.

    Returns the path to the no-vocals (accompaniment) track.
    Uses the htdemucs model with --two-stems vocals for speed.
    """
    audio_name = Path(audio_path).stem
    output_dir = str(Path(output_dir))

    logger.info("Running vocal separation with demucs...")
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-o", output_dir,
        audio_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs vocal separation failed: {result.stderr}")

    # Demucs output structure: {output_dir}/htdemucs/{audio_name}/no_vocals.wav
    no_vocals_path = Path(output_dir) / "htdemucs" / audio_name / "no_vocals.wav"
    if not no_vocals_path.exists():
        raise FileNotFoundError(
            f"Demucs output not found at {no_vocals_path}. "
            f"Contents: {list((Path(output_dir) / 'htdemucs').rglob('*')) if (Path(output_dir) / 'htdemucs').exists() else 'htdemucs dir missing'}"
        )

    logger.info(f"Vocal separation complete: {no_vocals_path}")
    return str(no_vocals_path)
