import hashlib
import json
import logging
import shutil
from pathlib import Path

from modules.transcript import TranscriptSegment

logger = logging.getLogger(__name__)


class PipelineCache:
    """Simple file-system cache for pipeline artifacts, scoped per video_id."""

    def __init__(self, cache_root: str, video_id: str):
        self.root = Path(cache_root) / video_id
        self.root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache directory: {self.root}")

    # --- Video ---

    def get_video(self) -> str | None:
        for ext in (".mp4", ".mkv", ".webm"):
            path = self.root / f"video{ext}"
            if path.exists():
                logger.info(f"CACHE HIT: video ({path})")
                return str(path)
        logger.info("CACHE MISS: video")
        return None

    def put_video(self, source_path: str) -> str:
        ext = Path(source_path).suffix or ".mp4"
        dest = self.root / f"video{ext}"
        shutil.copy2(source_path, dest)
        logger.info(f"CACHE STORE: video -> {dest}")
        return str(dest)

    # --- Audio ---

    def get_audio(self) -> str | None:
        path = self.root / "audio.wav"
        if path.exists():
            logger.info(f"CACHE HIT: audio ({path})")
            return str(path)
        logger.info("CACHE MISS: audio")
        return None

    def put_audio(self, source_path: str) -> str:
        dest = self.root / "audio.wav"
        shutil.copy2(source_path, dest)
        logger.info(f"CACHE STORE: audio -> {dest}")
        return str(dest)

    # --- Transcript ---

    def get_transcript(self, key: str) -> list[TranscriptSegment] | None:
        path = self.root / f"transcript_{key}.json"
        if not path.exists():
            logger.info(f"CACHE MISS: transcript (key={key})")
            return None
        logger.info(f"CACHE HIT: transcript (key={key})")
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            TranscriptSegment(text=s["text"], start=s["start"], duration=s["duration"])
            for s in data
        ]

    def put_transcript(self, key: str, segments: list[TranscriptSegment]) -> None:
        path = self.root / f"transcript_{key}.json"
        data = [
            {"text": s.text, "start": s.start, "duration": s.duration}
            for s in segments
        ]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"CACHE STORE: transcript ({len(segments)} segments)")

    # --- Translation ---

    @staticmethod
    def _translation_hash(
        segments: list[TranscriptSegment], provider: str, model: str,
    ) -> str:
        content = json.dumps(
            [s.text for s in segments] + [provider, model],
            ensure_ascii=False,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_translation(
        self, original_segments: list[TranscriptSegment], provider: str, model: str,
    ) -> list[TranscriptSegment] | None:
        h = self._translation_hash(original_segments, provider, model)
        path = self.root / f"translation_{h}.json"
        if not path.exists():
            logger.info(f"CACHE MISS: translation (provider={provider}, model={model})")
            return None
        logger.info(f"CACHE HIT: translation (provider={provider}, model={model})")
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            TranscriptSegment(text=s["text"], start=s["start"], duration=s["duration"])
            for s in data
        ]

    def put_translation(
        self,
        original_segments: list[TranscriptSegment],
        provider: str,
        model: str,
        translated_segments: list[TranscriptSegment],
    ) -> None:
        h = self._translation_hash(original_segments, provider, model)
        path = self.root / f"translation_{h}.json"
        data = [
            {"text": s.text, "start": s.start, "duration": s.duration}
            for s in translated_segments
        ]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"CACHE STORE: translation ({len(translated_segments)} segments)")

    # --- TTS (per-segment) ---

    @staticmethod
    def _tts_hash(text: str, voice: str, model: str) -> str:
        content = json.dumps([text, voice, model], ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _tts_dir(self) -> Path:
        d = self.root / "tts"
        d.mkdir(exist_ok=True)
        return d

    def get_tts_segment(self, text: str, voice: str, model: str) -> str | None:
        h = self._tts_hash(text, voice, model)
        path = self._tts_dir() / f"{h}.wav"
        if path.exists():
            return str(path)
        return None

    def put_tts_segment(self, text: str, voice: str, model: str, source_path: str) -> str:
        h = self._tts_hash(text, voice, model)
        dest = self._tts_dir() / f"{h}.wav"
        shutil.copy2(source_path, dest)
        return str(dest)

    # --- Background audio (vocal separation) ---

    def get_background(self) -> str | None:
        path = self.root / "no_vocals.wav"
        if path.exists():
            logger.info(f"CACHE HIT: background audio ({path})")
            return str(path)
        logger.info("CACHE MISS: background audio")
        return None

    def put_background(self, source_path: str) -> str:
        dest = self.root / "no_vocals.wav"
        shutil.copy2(source_path, dest)
        logger.info(f"CACHE STORE: background audio -> {dest}")
        return str(dest)
