import logging
from dataclasses import dataclass
from typing import Callable

from pydub import AudioSegment

from config import Config
from modules.transcript import TranscriptSegment
from modules.tts import TTSResult
from utils.audio_utils import speed_change, generate_silence

logger = logging.getLogger(__name__)


@dataclass
class TimeRegion:
    """A region of the video with a specific playback speed."""
    start: float        # original video time
    end: float          # original video time
    video_speed: float  # 1.0 = normal, <1.0 = video plays slower
    new_start: float    # position in the new (stretched) timeline
    new_end: float      # position in the new (stretched) timeline


@dataclass
class AlignedSegment:
    original_start: float
    original_end: float
    tts_audio: AudioSegment
    speed_factor: float
    adjusted_audio: AudioSegment
    video_speed: float  # 1.0 = normal, <1.0 = video slows down for this segment


def _apply_fades(audio: AudioSegment, fade_ms: int = 50) -> AudioSegment:
    if len(audio) > fade_ms * 3:
        return audio.fade_in(fade_ms).fade_out(fade_ms)
    return audio


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
            video_speed=1.0,
        )

    speed_factor = tts_ms / available_ms

    if speed_factor > config.speed_max:
        # TTS is too long — speed up audio to max, slow down video to compensate
        adjusted = speed_change(tts_audio, config.speed_max)
        video_speed = available_ms / len(adjusted)
        logger.info(
            f"Segment at {original_segment.start:.1f}s: audio at {config.speed_max}x, "
            f"video slowed to {video_speed:.2f}x (needed {speed_factor:.2f}x)"
        )
    elif speed_factor < config.speed_min:
        # TTS is too short — slow down to min, silence fills the rest
        adjusted = speed_change(tts_audio, config.speed_min)
        video_speed = 1.0
        logger.info(
            f"Segment at {original_segment.start:.1f}s: speed capped at {config.speed_min}x "
            f"(needed {speed_factor:.2f}x), padding with silence"
        )
    else:
        # Within acceptable range — adjust audio speed, video stays normal
        adjusted = speed_change(tts_audio, speed_factor)
        video_speed = 1.0

    adjusted = _apply_fades(adjusted, config.fade_ms)

    return AlignedSegment(
        original_start=original_segment.start,
        original_end=original_segment.end,
        tts_audio=tts_audio,
        speed_factor=speed_factor,
        adjusted_audio=adjusted,
        video_speed=video_speed,
    )


def calculate_time_regions(
    aligned_segments: list[AlignedSegment],
    total_duration: float,
) -> tuple[list[TimeRegion], list[float], float]:
    """Build the new timeline accounting for video slowdowns.

    Returns (regions, segment_new_starts, new_total_duration).
    """
    regions: list[TimeRegion] = []
    segment_new_starts: list[float] = []
    current = 0.0
    prev_end = 0.0

    for seg in aligned_segments:
        # Gap before this segment (plays at normal speed)
        if seg.original_start - prev_end > 0.01:
            gap = seg.original_start - prev_end
            regions.append(TimeRegion(
                start=prev_end, end=seg.original_start,
                video_speed=1.0,
                new_start=current, new_end=current + gap,
            ))
            current += gap

        # Record new start position for this segment's audio
        segment_new_starts.append(current)

        # Segment region (may be slowed)
        orig_dur = seg.original_end - seg.original_start
        new_dur = orig_dur / seg.video_speed  # slower video = longer duration
        regions.append(TimeRegion(
            start=seg.original_start, end=seg.original_end,
            video_speed=seg.video_speed,
            new_start=current, new_end=current + new_dur,
        ))
        current += new_dur
        prev_end = seg.original_end

    # Trailing portion after last segment
    if total_duration - prev_end > 0.01:
        gap = total_duration - prev_end
        regions.append(TimeRegion(
            start=prev_end, end=total_duration,
            video_speed=1.0,
            new_start=current, new_end=current + gap,
        ))
        current += gap

    return regions, segment_new_starts, current


def warp_background_audio(
    background: AudioSegment,
    regions: list[TimeRegion],
) -> AudioSegment:
    """Warp background audio to match the variable-speed video timeline.

    Each region of the background is sliced and speed-adjusted to match
    the corresponding region in the new (stretched) timeline.
    """
    chunks: list[AudioSegment] = []
    bg_len_ms = len(background)

    for region in regions:
        start_ms = int(region.start * 1000)
        end_ms = int(region.end * 1000)

        # Clamp to background audio bounds
        start_ms = min(start_ms, bg_len_ms)
        end_ms = min(end_ms, bg_len_ms)

        if end_ms <= start_ms:
            # Region is beyond background audio — fill with silence
            new_dur_ms = int((region.new_end - region.new_start) * 1000)
            chunks.append(generate_silence(max(new_dur_ms, 0), background.frame_rate))
            continue

        chunk = background[start_ms:end_ms]

        if region.video_speed < 0.99:
            # Video is slowed — slow down background by the same factor
            chunk = speed_change(chunk, region.video_speed)

        chunks.append(chunk)

    if not chunks:
        return background

    warped = chunks[0]
    for chunk in chunks[1:]:
        warped += chunk

    return warped


def assemble_full_audio(
    aligned_segments: list[AlignedSegment],
    segment_new_starts: list[float],
    new_total_duration: float,
    background: AudioSegment | None = None,
    sample_rate: int = 44100,
) -> AudioSegment:
    """Place each segment's audio at its new (stretched) position."""
    canvas_ms = int(new_total_duration * 1000)

    if background is not None:
        # Use the warped background as the canvas
        canvas = background
        # Pad or trim to match expected duration
        if len(canvas) < canvas_ms:
            canvas += generate_silence(canvas_ms - len(canvas), sample_rate)
        elif len(canvas) > canvas_ms:
            canvas = canvas[:canvas_ms]
    else:
        canvas = generate_silence(canvas_ms, sample_rate)

    for seg, new_start in zip(aligned_segments, segment_new_starts):
        position_ms = int(new_start * 1000)
        audio = seg.adjusted_audio

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
    background_audio_path: str | None = None,
) -> tuple[str, list[TimeRegion], float]:
    """Create dubbed audio and return (path, time_regions, new_total_duration)."""
    # Clamp segments that extend beyond the actual video duration
    for seg in original_segments:
        if seg.end > total_duration:
            logger.info(f"Clamping segment end {seg.end:.2f}s -> {total_duration:.2f}s (video duration)")
            seg.duration = total_duration - seg.start
        if seg.start >= total_duration:
            logger.warning(f"Segment starts at {seg.start:.2f}s, past video end {total_duration:.2f}s")

    aligned: list[AlignedSegment] = []

    for i, (tts_result, orig_seg) in enumerate(zip(tts_results, original_segments)):
        aligned_seg = align_segment(tts_result, orig_seg, config)
        aligned.append(aligned_seg)
        if progress_cb:
            progress_cb((i + 1) / len(tts_results) * 0.5)

    if progress_cb:
        progress_cb(0.5)

    # Calculate the new timeline with video slowdowns
    regions, segment_new_starts, new_duration = calculate_time_regions(
        aligned, total_duration,
    )

    extra = new_duration - total_duration
    if extra > 0.1:
        logger.info(
            f"Timeline stretched: {total_duration:.1f}s -> {new_duration:.1f}s "
            f"(+{extra:.1f}s from video slowdowns)"
        )

    if progress_cb:
        progress_cb(0.6)

    # Prepare background audio if provided
    background = None
    if background_audio_path:
        logger.info("Warping background audio to match new timeline...")
        raw_bg = AudioSegment.from_file(background_audio_path)
        raw_bg = raw_bg + config.background_volume_db  # reduce volume
        background = warp_background_audio(raw_bg, regions)

    if progress_cb:
        progress_cb(0.8)

    full_audio = assemble_full_audio(aligned, segment_new_starts, new_duration, background)

    if progress_cb:
        progress_cb(0.9)

    full_audio.export(output_path, format="wav")

    if progress_cb:
        progress_cb(1.0)

    return output_path, regions, new_duration
