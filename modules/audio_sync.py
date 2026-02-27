import logging
from dataclasses import dataclass
from typing import Callable

from pydub import AudioSegment

from config import Config
from modules.transcript import TranscriptSegment
from modules.tts import TTSResult
from utils.audio_utils import speed_change, generate_silence

logger = logging.getLogger(__name__)

FADE_MS = 10


@dataclass
class AlignedSegment:
    original_start: float
    original_end: float
    tts_audio: AudioSegment
    speed_factor: float
    adjusted_audio: AudioSegment


def align_segment(
    tts_result: TTSResult,
    original_segment: TranscriptSegment,
    config: Config,
) -> AlignedSegment:
    available_ms = original_segment.duration * 1000
    tts_audio = AudioSegment.from_file(tts_result.audio_path)
    tts_ms = len(tts_audio)

    if tts_ms == 0 or available_ms <= 0:
        silence = generate_silence(int(max(available_ms, 0)))
        return AlignedSegment(
            original_start=original_segment.start,
            original_end=original_segment.end,
            tts_audio=tts_audio,
            speed_factor=1.0,
            adjusted_audio=silence,
        )

    speed_factor = tts_ms / available_ms

    if speed_factor > config.speed_max:
        # TTS is too long — speed up to max, then truncate
        adjusted = speed_change(tts_audio, config.speed_max)
        if len(adjusted) > available_ms:
            adjusted = adjusted[: int(available_ms)]
        logger.info(
            f"Segment at {original_segment.start:.1f}s: speed capped at {config.speed_max}x "
            f"(needed {speed_factor:.2f}x), truncating"
        )
    elif speed_factor < config.speed_min:
        # TTS is too short — slow down to min, silence fills the rest
        adjusted = speed_change(tts_audio, config.speed_min)
        logger.info(
            f"Segment at {original_segment.start:.1f}s: speed capped at {config.speed_min}x "
            f"(needed {speed_factor:.2f}x), padding with silence"
        )
    else:
        # Within acceptable range
        adjusted = speed_change(tts_audio, speed_factor)

    # Ensure it doesn't exceed the window
    if len(adjusted) > available_ms:
        adjusted = adjusted[: int(available_ms)]

    # Apply fade in/out to prevent clicks
    if len(adjusted) > FADE_MS * 2:
        adjusted = adjusted.fade_in(FADE_MS).fade_out(FADE_MS)

    return AlignedSegment(
        original_start=original_segment.start,
        original_end=original_segment.end,
        tts_audio=tts_audio,
        speed_factor=speed_factor,
        adjusted_audio=adjusted,
    )


def assemble_full_audio(
    aligned_segments: list[AlignedSegment],
    total_duration: float,
    sample_rate: int = 44100,
) -> AudioSegment:
    canvas = generate_silence(int(total_duration * 1000), sample_rate)

    for i, seg in enumerate(aligned_segments):
        position_ms = int(seg.original_start * 1000)

        audio = seg.adjusted_audio

        # Prevent overlap with next segment
        if i + 1 < len(aligned_segments):
            next_start_ms = int(aligned_segments[i + 1].original_start * 1000)
            max_duration_ms = next_start_ms - position_ms
            if len(audio) > max_duration_ms and max_duration_ms > 0:
                audio = audio[:max_duration_ms]

        # Don't exceed canvas bounds
        if position_ms + len(audio) > len(canvas):
            audio = audio[: len(canvas) - position_ms]

        if len(audio) > 0:
            canvas = canvas.overlay(audio, position=position_ms)

    return canvas


def create_dubbed_audio(
    tts_results: list[TTSResult],
    original_segments: list[TranscriptSegment],
    total_duration: float,
    output_path: str,
    config: Config,
    progress_cb: Callable[[float], None] | None = None,
) -> str:
    aligned: list[AlignedSegment] = []

    for i, (tts_result, orig_seg) in enumerate(zip(tts_results, original_segments)):
        aligned_seg = align_segment(tts_result, orig_seg, config)
        aligned.append(aligned_seg)
        if progress_cb:
            progress_cb((i + 1) / len(tts_results) * 0.7)

    if progress_cb:
        progress_cb(0.7)

    full_audio = assemble_full_audio(aligned, total_duration)

    if progress_cb:
        progress_cb(0.9)

    full_audio.export(output_path, format="wav")

    if progress_cb:
        progress_cb(1.0)

    return output_path
