from __future__ import annotations

import logging
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from pydub import AudioSegment

if TYPE_CHECKING:
    from modules.cache import PipelineCache

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

    tts_config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name,
                )
            )
        ),
    )

    # Try up to 3 times before falling back to silence.
    # Handles: empty responses, 400 INVALID_ARGUMENT (model tries to
    # generate text instead of audio), and rate-limit 429 errors.
    pcm_data = None
    contents = text

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=config.gemini_tts_model,
                contents=contents,
                config=tts_config,
            )

            if cost_tracker and response.usage_metadata:
                cost_tracker.add_tts_usage(
                    config.gemini_tts_model,
                    response.usage_metadata.prompt_token_count or 0,
                    response.usage_metadata.candidates_token_count or 0,
                )

            pcm_data = response.candidates[0].content.parts[0].inline_data.data
            break

        except (AttributeError, IndexError, TypeError):
            # Empty response from the API
            if attempt < 2:
                logger.warning(f"Gemini TTS empty response (attempt {attempt + 1}), retrying: {text[:60]}...")
                time.sleep(2)
            else:
                logger.error(f"Gemini TTS empty after {attempt + 1} attempts: {text[:60]}...")

        except Exception as e:
            err_str = str(e)
            if "INVALID_ARGUMENT" in err_str or "generate text" in err_str:
                # Model confused TTS with text generation — retry with
                # an explicit "speak this" prefix so intent is unambiguous.
                logger.warning(f"TTS INVALID_ARGUMENT (attempt {attempt + 1}): {text[:60]}...")
                contents = f"Read this text aloud: {text}"
                time.sleep(1)
            elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                if attempt < 2:
                    logger.warning(f"TTS rate limited (attempt {attempt + 1}), waiting 30s...")
                    time.sleep(30)
                else:
                    raise
            elif "500" in err_str or "INTERNAL" in err_str or "ServerError" in type(e).__name__:
                # Gemini server error — retry after short delay, fall back
                # to silence if all attempts fail.
                if attempt < 2:
                    logger.warning(f"TTS server error (attempt {attempt + 1}), retrying in 5s: {text[:60]}...")
                    time.sleep(5)
                else:
                    logger.error(f"TTS server error after {attempt + 1} attempts: {text[:60]}...")
            else:
                raise

    if pcm_data is None:
        logger.warning(f"TTS fallback to silence for: {text[:60]}...")
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


def _synthesize_with_retry(
    index: int,
    text: str,
    audio_path: str,
    config: Config,
    cost_tracker: CostTracker | None,
) -> TTSResult:
    """Synthesize a single segment with rate-limit / server-error retry. Thread-safe."""
    try:
        duration = _synthesize_single(text, audio_path, config, cost_tracker)
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            logger.warning(f"Rate limited on segment {index}, waiting 60s...")
            time.sleep(60)
            duration = _synthesize_single(text, audio_path, config, cost_tracker)
        elif "500" in err_str or "INTERNAL" in err_str or "ServerError" in type(e).__name__:
            logger.warning(f"Server error on segment {index}, retrying in 10s...")
            time.sleep(10)
            duration = _synthesize_single(text, audio_path, config, cost_tracker)
        else:
            raise
    return TTSResult(audio_path=audio_path, duration=duration, segment_index=index)


def synthesize_all_segments(
    segments: list[TranscriptSegment],
    output_dir: str,
    config: Config,
    cost_tracker: CostTracker | None = None,
    progress_cb: Callable[[float], None] | None = None,
    cache: PipelineCache | None = None,
    max_workers: int = 4,
) -> list[TTSResult]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    voice = config.tts_voice_name or "Kore"
    model = config.gemini_tts_model
    total = len(segments)

    # Pre-allocated results list (filled by index to maintain order)
    results: list[TTSResult | None] = [None] * total
    to_synthesize: list[tuple[int, str, str]] = []  # (index, text, audio_path)
    cached_count = 0

    # Pass 1: collect cache hits
    for i, segment in enumerate(segments):
        if cache:
            cached_path = cache.get_tts_segment(segment.text, voice, model)
            if cached_path:
                audio = AudioSegment.from_wav(cached_path)
                duration = len(audio) / 1000.0
                results[i] = TTSResult(
                    audio_path=cached_path, duration=duration, segment_index=i,
                )
                cached_count += 1
                continue
        to_synthesize.append((i, segment.text, str(output_path / f"tts_{i:04d}.wav")))

    if cached_count > 0:
        logger.info(f"TTS: {cached_count}/{total} segments from cache")

    if not to_synthesize:
        if progress_cb:
            progress_cb(1.0)
        return [r for r in results if r is not None]

    # Pass 2: synthesize non-cached segments concurrently
    completed = cached_count
    lock = threading.Lock()

    def _on_done():
        nonlocal completed
        with lock:
            completed += 1
            if progress_cb:
                progress_cb(completed / total)

    logger.info(f"TTS: synthesizing {len(to_synthesize)} segments with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, text, audio_path in to_synthesize:
            logger.info(f"TTS {idx+1}/{total}: {text[:80]}...")
            future = executor.submit(
                _synthesize_with_retry, idx, text, audio_path, config, cost_tracker,
            )
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            tts_result = future.result()  # raises if the task failed
            results[idx] = tts_result
            logger.info(f"TTS {idx+1}/{total}: generated {tts_result.duration:.2f}s audio")

            # Store in cache
            if cache:
                segment_text = segments[idx].text
                cache.put_tts_segment(segment_text, voice, model, tts_result.audio_path)

            _on_done()

    return [r for r in results if r is not None]
