import json
import time
import logging
from typing import Callable

import anthropic
from google import genai

from config import Config
from modules.transcript import TranscriptSegment
from utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)

TRANSLATION_SYSTEM_PROMPT = (
    "You are a professional translator specializing in English to Armenian translation. "
    "You translate spoken dialogue naturally, preserving tone and meaning. "
    "Translate colloquial speech naturally — do not make it overly formal."
)

BATCH_SIZE = 15


def _build_translation_prompt(texts: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. \"{t}\"" for i, t in enumerate(texts))
    return (
        "Translate each of the following numbered segments from English to Armenian. "
        "Return ONLY a JSON array of translated strings in the same order. "
        "Do not add or remove segments. Keep the exact same count.\n\n"
        f"Segments:\n{numbered}\n\n"
        f"Return format: [\"translation1\", \"translation2\", ...]"
    )


def _parse_translations(response_text: str, expected_count: int) -> list[str]:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    translations = json.loads(text)
    if not isinstance(translations, list) or len(translations) != expected_count:
        raise ValueError(
            f"Expected {expected_count} translations, got {len(translations) if isinstance(translations, list) else 'non-list'}"
        )
    return [str(t) for t in translations]


def translate_segments_claude(
    segments: list[TranscriptSegment],
    config: Config,
    cost_tracker: CostTracker | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> list[TranscriptSegment]:
    client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def call_fn(prompt: str) -> str:
        response = client.messages.create(
            model=config.claude_model,
            max_tokens=4096,
            system=TRANSLATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        if cost_tracker:
            cost_tracker.add_translation_usage(
                config.claude_model,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
        return response.content[0].text

    return _translate_batched(segments, call_fn, progress_cb)


def translate_segments_gemini(
    segments: list[TranscriptSegment],
    config: Config,
    cost_tracker: CostTracker | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> list[TranscriptSegment]:
    client = genai.Client(api_key=config.google_api_key)

    def call_fn(prompt: str) -> str:
        response = client.models.generate_content(
            model=config.gemini_model,
            contents=f"{TRANSLATION_SYSTEM_PROMPT}\n\n{prompt}",
        )
        if cost_tracker and response.usage_metadata:
            cost_tracker.add_translation_usage(
                config.gemini_model,
                response.usage_metadata.prompt_token_count or 0,
                response.usage_metadata.candidates_token_count or 0,
            )
        return response.text

    return _translate_batched(segments, call_fn, progress_cb)


def _translate_batched(
    segments: list[TranscriptSegment],
    call_fn: Callable[[str], str],
    progress_cb: Callable[[float], None] | None = None,
) -> list[TranscriptSegment]:
    all_translated: list[TranscriptSegment] = []
    total_batches = (len(segments) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(0, len(segments), BATCH_SIZE):
        batch = segments[batch_idx : batch_idx + BATCH_SIZE]
        texts = [seg.text for seg in batch]
        prompt = _build_translation_prompt(texts)

        translations = _translate_with_retry(call_fn, prompt, len(texts))

        for seg, translated_text in zip(batch, translations):
            all_translated.append(TranscriptSegment(
                text=translated_text,
                start=seg.start,
                duration=seg.duration,
            ))

        if progress_cb:
            current_batch = batch_idx // BATCH_SIZE + 1
            progress_cb(current_batch / total_batches)

    return all_translated


def _translate_with_retry(
    call_fn: Callable[[str], str],
    prompt: str,
    expected_count: int,
    max_retries: int = 3,
) -> list[str]:
    for attempt in range(max_retries):
        try:
            response_text = call_fn(prompt)
            return _parse_translations(response_text, expected_count)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Translation parse error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to parse translation after {max_retries} attempts: {e}")
            time.sleep(1)
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                wait = 2 ** (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            else:
                raise

    raise RuntimeError("Translation failed after all retries")


def translate_segments(
    segments: list[TranscriptSegment],
    config: Config,
    cost_tracker: CostTracker | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> list[TranscriptSegment]:
    if config.translation_provider == "gemini":
        return translate_segments_gemini(segments, config, cost_tracker, progress_cb)
    return translate_segments_claude(segments, config, cost_tracker, progress_cb)
