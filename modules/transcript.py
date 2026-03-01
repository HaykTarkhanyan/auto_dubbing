import logging
import re
from dataclasses import dataclass
from typing import Callable

from youtube_transcript_api import YouTubeTranscriptApi

from utils.text_utils import clean_caption_text

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


def get_youtube_transcript(
    video_id: str, languages: tuple[str, ...] = ("en",)
) -> list[TranscriptSegment] | None:
    try:
        entries = YouTubeTranscriptApi.get_transcript(video_id, languages=list(languages))
        segments = []
        for entry in entries:
            text = clean_caption_text(entry["text"])
            if text:
                segments.append(TranscriptSegment(
                    text=text,
                    start=float(entry["start"]),
                    duration=float(entry["duration"]),
                ))
        return segments if segments else None
    except Exception:
        return None


def get_whisper_transcript(audio_path: str, model_size: str = "base") -> list[TranscriptSegment]:
    import whisper

    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, word_timestamps=False)

    segments = []
    for seg in result.get("segments", []):
        text = clean_caption_text(seg.get("text", ""))
        if text:
            start = float(seg["start"])
            end = float(seg["end"])
            segments.append(TranscriptSegment(
                text=text,
                start=start,
                duration=end - start,
            ))
    return segments


def resegment_by_sentences(
    raw_segments: list[TranscriptSegment],
    min_duration: float = 5.0,
    max_duration: float = 30.0,
) -> list[TranscriptSegment]:
    """Merge all captions into full text, then re-split at sentence boundaries
    with timestamps interpolated from the original caption timing."""
    if not raw_segments:
        return []

    # Step 1: Build a character-position → timestamp map
    # Each raw segment contributes its text at a known start time.
    # We interpolate: each character within a segment gets a proportional time.
    char_timestamps: list[tuple[int, float]] = []  # (char_position, time)
    full_text_parts: list[str] = []
    char_offset = 0

    for seg in raw_segments:
        # Mark the start of this segment's text
        char_timestamps.append((char_offset, seg.start))
        full_text_parts.append(seg.text)
        char_offset += len(seg.text) + 1  # +1 for the space we'll join with

    # Mark the end
    last_seg = raw_segments[-1]
    char_timestamps.append((char_offset, last_seg.end))

    full_text = " ".join(full_text_parts)

    # Step 2: Split full text at sentence boundaries
    sentences = _split_sentences(full_text)
    if not sentences:
        return raw_segments

    logger.info(f"Re-segmented {len(raw_segments)} raw captions into {len(sentences)} sentences")

    # Step 3: Map each sentence back to a timestamp
    result: list[TranscriptSegment] = []
    current_char_pos = 0

    for i, sentence in enumerate(sentences):
        # Find this sentence's position in the full text
        sent_start_char = full_text.find(sentence, current_char_pos)
        if sent_start_char == -1:
            logger.warning(f"Could not locate sentence in full text at pos {current_char_pos}, using fallback position")
            sent_start_char = current_char_pos
        sent_end_char = sent_start_char + len(sentence)
        current_char_pos = sent_end_char

        # Interpolate timestamp from character position
        start_time = _interpolate_time(sent_start_char, char_timestamps)
        end_time = _interpolate_time(sent_end_char, char_timestamps)

        result.append(TranscriptSegment(
            text=sentence.strip(),
            start=start_time,
            duration=max(end_time - start_time, 0.1),
        ))

    # Step 4: Merge short sentences, split long ones
    result = _enforce_duration_bounds(result, min_duration, max_duration)

    for seg in result:
        logger.debug(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.text[:80]}")

    return result


def _split_sentences(text: str) -> list[str]:
    """Split text at sentence-ending punctuation (.!?) followed by a space."""
    # Split at . ! ? followed by space or end of string
    parts = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty strings
    return [p.strip() for p in parts if p.strip()]


def _interpolate_time(char_pos: int, char_timestamps: list[tuple[int, float]]) -> float:
    """Given a character position, interpolate its timestamp from the mapping."""
    # Find the two surrounding anchor points
    for i in range(len(char_timestamps) - 1):
        pos_a, time_a = char_timestamps[i]
        pos_b, time_b = char_timestamps[i + 1]
        if pos_a <= char_pos <= pos_b:
            if pos_b == pos_a:
                return time_a
            fraction = (char_pos - pos_a) / (pos_b - pos_a)
            return time_a + fraction * (time_b - time_a)

    # Past the end — return last timestamp
    return char_timestamps[-1][1]


def _enforce_duration_bounds(
    segments: list[TranscriptSegment],
    min_duration: float,
    max_duration: float,
) -> list[TranscriptSegment]:
    """Merge segments that are too short, split ones that are too long."""
    if not segments:
        return []

    # Pass 1: merge short segments with next neighbor
    merged: list[TranscriptSegment] = []
    i = 0
    while i < len(segments):
        current = segments[i]
        while current.duration < min_duration and i + 1 < len(segments):
            next_seg = segments[i + 1]
            current = TranscriptSegment(
                text=f"{current.text} {next_seg.text}",
                start=current.start,
                duration=(next_seg.end - current.start),
            )
            i += 1
        merged.append(current)
        i += 1

    # Pass 2: split segments that are too long at comma/semicolon boundaries
    result: list[TranscriptSegment] = []
    for seg in merged:
        if seg.duration <= max_duration:
            result.append(seg)
            continue

        # Try splitting at commas or semicolons
        parts = re.split(r'(?<=[,;])\s+', seg.text)
        if len(parts) <= 1:
            result.append(seg)
            continue

        # Group parts so each chunk is under max_duration
        total_chars = sum(len(p) for p in parts)
        offset = seg.start
        chunk_texts: list[str] = []
        chunk_start = seg.start

        for part in parts:
            part_duration = seg.duration * (len(part) / total_chars)
            projected_end = offset + part_duration

            if chunk_texts and (projected_end - chunk_start) > max_duration:
                # Flush current chunk
                result.append(TranscriptSegment(
                    text=" ".join(chunk_texts),
                    start=chunk_start,
                    duration=offset - chunk_start,
                ))
                chunk_texts = []
                chunk_start = offset

            chunk_texts.append(part)
            offset += part_duration

        if chunk_texts:
            result.append(TranscriptSegment(
                text=" ".join(chunk_texts),
                start=chunk_start,
                duration=seg.end - chunk_start,
            ))

    return result


def _apply_trim_to_segments(
    segments: list[TranscriptSegment],
    trim_start: float,
    trim_end: float | None,
) -> list[TranscriptSegment]:
    """Filter segments to a time range and offset timestamps to start from 0."""
    result = []
    for seg in segments:
        # Skip segments entirely outside the trim range
        if trim_end is not None and seg.start >= trim_end:
            continue
        if seg.end <= trim_start:
            continue
        # Clamp and offset
        new_start = max(seg.start, trim_start) - trim_start
        new_end = (min(seg.end, trim_end) if trim_end is not None else seg.end) - trim_start
        result.append(TranscriptSegment(
            text=seg.text,
            start=new_start,
            duration=max(new_end - new_start, 0.1),
        ))
    return result


def extract_transcript(
    video_id: str,
    audio_path: str | None,
    whisper_model_size: str = "base",
    segment_min_duration: float = 5.0,
    segment_max_duration: float = 30.0,
    progress_cb: Callable[[float], None] | None = None,
    trim_start: float | None = None,
    trim_end: float | None = None,
) -> list[TranscriptSegment]:
    if progress_cb:
        progress_cb(0.1)

    # Try YouTube captions first
    raw_segments = get_youtube_transcript(video_id)
    if raw_segments:
        # YouTube captions have full-video timestamps — filter to trim range
        if trim_start is not None or trim_end is not None:
            raw_segments = _apply_trim_to_segments(raw_segments, trim_start or 0, trim_end)
            if not raw_segments:
                logger.info("No YouTube captions in trim range, falling back to Whisper")
                raw_segments = None

    if raw_segments:
        if progress_cb:
            progress_cb(0.5)
        segments = resegment_by_sentences(raw_segments, segment_min_duration, segment_max_duration)
        if progress_cb:
            progress_cb(1.0)
        return segments

    # Fallback to Whisper (audio is already trimmed, timestamps are correct)
    if not audio_path:
        raise RuntimeError("No YouTube captions found and no audio path provided for Whisper fallback")

    if progress_cb:
        progress_cb(0.2)

    raw_segments = get_whisper_transcript(audio_path, model_size=whisper_model_size)
    if progress_cb:
        progress_cb(0.8)

    segments = resegment_by_sentences(raw_segments, segment_min_duration, segment_max_duration)
    if progress_cb:
        progress_cb(1.0)

    return segments
