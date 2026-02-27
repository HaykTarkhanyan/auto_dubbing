from dataclasses import dataclass
from typing import Callable

from youtube_transcript_api import YouTubeTranscriptApi

from utils.text_utils import clean_caption_text, split_at_sentence


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


def merge_short_segments(
    segments: list[TranscriptSegment],
    min_duration: float = 5.0,
    max_duration: float = 30.0,
) -> list[TranscriptSegment]:
    if not segments:
        return []

    # Pass 1: merge short segments with their next neighbor
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

    # Pass 2: split long segments at sentence boundaries
    result: list[TranscriptSegment] = []
    for seg in merged:
        if seg.duration <= max_duration:
            result.append(seg)
            continue

        chunks = split_at_sentence(seg.text)
        if len(chunks) <= 1:
            result.append(seg)
            continue

        # Distribute duration proportionally by text length
        total_chars = sum(len(c) for c in chunks)
        offset = seg.start
        for chunk in chunks:
            chunk_duration = seg.duration * (len(chunk) / total_chars)
            result.append(TranscriptSegment(
                text=chunk,
                start=offset,
                duration=chunk_duration,
            ))
            offset += chunk_duration

    return result


def extract_transcript(
    video_id: str,
    audio_path: str | None,
    whisper_model_size: str = "base",
    progress_cb: Callable[[float], None] | None = None,
) -> list[TranscriptSegment]:
    if progress_cb:
        progress_cb(0.1)

    # Try YouTube captions first
    segments = get_youtube_transcript(video_id)
    if segments:
        if progress_cb:
            progress_cb(0.8)
        segments = merge_short_segments(segments)
        if progress_cb:
            progress_cb(1.0)
        return segments

    # Fallback to Whisper
    if not audio_path:
        raise RuntimeError("No YouTube captions found and no audio path provided for Whisper fallback")

    if progress_cb:
        progress_cb(0.2)

    segments = get_whisper_transcript(audio_path, model_size=whisper_model_size)
    if progress_cb:
        progress_cb(0.9)

    segments = merge_short_segments(segments)
    if progress_cb:
        progress_cb(1.0)

    return segments
