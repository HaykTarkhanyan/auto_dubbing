import logging
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydub import AudioSegment

from config import Config
from modules.transcript import TranscriptSegment
from utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    audio_path: str
    duration: float  # seconds
    segment_index: int


def _save_wav(path: str, pcm_data: bytes, rate: int = 24000):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)


_gemini_client = None


def _get_gemini_client(api_key: str):
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def synthesize_gemini(
    text: str,
    output_path: str,
    config: Config,
    cost_tracker: CostTracker | None = None,
) -> float:
    from google.genai import types

    client = _get_gemini_client(config.google_api_key)
    voice_name = config.tts_voice_name or "Kore"

    response = client.models.generate_content(
        model=config.gemini_tts_model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name,
                    )
                )
            ),
        ),
    )

    if cost_tracker and response.usage_metadata:
        cost_tracker.add_tts_usage(
            response.usage_metadata.prompt_token_count or 0,
            response.usage_metadata.candidates_token_count or 0,
        )

    try:
        pcm_data = response.candidates[0].content.parts[0].inline_data.data
    except (AttributeError, IndexError, TypeError):
        logger.warning(f"Gemini TTS returned empty response for text: {text[:80]}...")
        silence = AudioSegment.silent(duration=500)
        silence.export(output_path, format="wav")
        return 0.5

    _save_wav(output_path, pcm_data)

    audio = AudioSegment.from_wav(output_path)
    return len(audio) / 1000.0


def _synthesize_single(
    text: str,
    output_path: str,
    config: Config,
    cost_tracker: CostTracker | None = None,
) -> float:
    if not text.strip():
        silence = AudioSegment.silent(duration=100)
        silence.export(output_path, format="wav")
        return 0.1

    return synthesize_gemini(text, output_path, config, cost_tracker)


def synthesize_all_segments(
    segments: list[TranscriptSegment],
    output_dir: str,
    config: Config,
    cost_tracker: CostTracker | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> list[TTSResult]:
    results: list[TTSResult] = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, segment in enumerate(segments):
        audio_path = str(output_path / f"tts_{i:04d}.wav")
        logger.info(f"TTS {i+1}/{len(segments)}: {segment.text[:80]}...")

        try:
            duration = _synthesize_single(segment.text, audio_path, config, cost_tracker)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                logger.warning(f"Rate limited on segment {i}, waiting 60s...")
                time.sleep(60)
                duration = _synthesize_single(segment.text, audio_path, config, cost_tracker)
            else:
                raise

        logger.info(f"TTS {i+1}/{len(segments)}: generated {duration:.2f}s audio")

        results.append(TTSResult(
            audio_path=audio_path,
            duration=duration,
            segment_index=i,
        ))

        if progress_cb:
            progress_cb((i + 1) / len(segments))

    return results
