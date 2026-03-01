import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from config import Config
from modules.cache import PipelineCache
from modules.downloader import get_metadata, download_video, extract_audio, get_video_duration, download_thumbnail, VideoMetadata
from modules.transcript import TranscriptSegment, extract_transcript
from modules.translator import translate_segments
from modules.tts import TTSResult, synthesize_all_segments
from modules.audio_sync import create_dubbed_audio
from modules.vocal_separator import separate_vocals
from modules.video_merge import create_variable_speed_video, merge_audio_video, verify_output
from modules.temp_manager import TempManager
from utils.cost_tracker import CostTracker

VIDEOS_DIR = Path(__file__).parent / "videos"

logger = logging.getLogger(__name__)


@dataclass
class Phase1Result:
    """Intermediate result after translation (steps 1-4)."""
    metadata: VideoMetadata
    video_path: str
    original_segments: list[TranscriptSegment]
    translated_segments: list[TranscriptSegment]
    cost_tracker: CostTracker
    timings: list[tuple[str, float]]
    temp: TempManager
    config: Config
    cache: PipelineCache | None = None


@dataclass
class PipelineResult:
    output_video_path: str
    output_dir: str
    metadata: VideoMetadata
    original_segments: list[TranscriptSegment]
    translated_segments: list[TranscriptSegment]
    cost_tracker: CostTracker = field(default_factory=CostTracker)
    error: str | None = None


def _sanitize_dirname(title: str) -> str:
    """Create a safe directory name from a video title."""
    name = re.sub(r'[<>:"/\\|?*]', '', title)
    name = name.strip().replace(' ', '_')[:60]
    return name or "untitled"


def _save_artifacts(
    video_dir: Path,
    metadata: VideoMetadata,
    original_segments: list[TranscriptSegment],
    translated_segments: list[TranscriptSegment],
    cost: CostTracker,
    dubbed_video_path: str,
    original_video_path: str,
    step_timings: list[tuple[str, float]],
    background_audio_path: str | None = None,
) -> str:
    """Save all outputs to the video's directory. Returns the final video path."""
    video_dir.mkdir(parents=True, exist_ok=True)

    # Save dubbed video
    final_video_path = str(video_dir / "dubbed.mp4")
    shutil.copy2(dubbed_video_path, final_video_path)

    # Save original English video
    orig_ext = Path(original_video_path).suffix or ".mp4"
    shutil.copy2(original_video_path, str(video_dir / f"original{orig_ext}"))

    # Save transcripts
    transcript_data = []
    for orig, trans in zip(original_segments, translated_segments):
        transcript_data.append({
            "start": round(orig.start, 2),
            "end": round(orig.end, 2),
            "original": orig.text,
            "translated": trans.text,
        })

    with open(video_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)

    # Save metadata
    meta_data = {
        "title": metadata.title,
        "video_id": metadata.video_id,
        "duration": metadata.duration,
        "uploader": metadata.uploader,
    }
    with open(video_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)

    # Save cost + timing summary
    with open(video_dir / "cost.txt", "w", encoding="utf-8") as f:
        f.write(cost.summary())
        if step_timings:
            f.write("\n\n=== Step Timings ===\n")
            for name, elapsed in step_timings:
                f.write(f"{name}: {elapsed:.1f}s\n")
            total_time = sum(t for _, t in step_timings)
            f.write(f"TOTAL: {total_time:.1f}s\n")

    # Save background music if available
    if background_audio_path and Path(background_audio_path).exists():
        shutil.copy2(background_audio_path, str(video_dir / "background_music.wav"))

    # Download thumbnail
    try:
        video_url = f"https://www.youtube.com/watch?v={metadata.video_id}"
        download_thumbnail(video_url, str(video_dir / "thumbnail"))
    except Exception as e:
        logger.warning(f"Failed to download thumbnail: {e}")

    # Save video info text file
    with open(video_dir / "video_info.txt", "w", encoding="utf-8") as f:
        f.write(f"Title: {metadata.title}\n")
        f.write(f"Channel: {metadata.uploader}\n")
        f.write(f"Channel URL: {metadata.channel_url}\n")
        f.write(f"Video URL: https://www.youtube.com/watch?v={metadata.video_id}\n")
        f.write(f"\n--- Description ---\n{metadata.description}\n")

    logger.info(f"Saved all artifacts to {video_dir}")
    return final_video_path


# (step_name, global_start, global_end)
PIPELINE_STEPS = [
    ("Extracting metadata", 0.00, 0.05),       # 0
    ("Downloading video", 0.05, 0.20),          # 1
    ("Extracting transcript", 0.20, 0.35),      # 2
    ("Translating to Armenian", 0.35, 0.55),    # 3
    ("Generating speech", 0.55, 0.75),          # 4
    ("Separating vocals", 0.75, 0.82),          # 5
    ("Synchronizing audio", 0.82, 0.90),        # 6
    ("Merging video", 0.90, 0.98),              # 7
    ("Finalizing", 0.98, 1.00),                 # 8
]


def _make_step_cb(
    global_cb: Callable[[float, str], None] | None,
    step_name: str,
    global_start: float,
    global_end: float,
) -> Callable[[float], None] | None:
    if not global_cb:
        return None

    def step_cb(local_progress: float):
        global_progress = global_start + local_progress * (global_end - global_start)
        global_cb(global_progress, step_name)

    return step_cb


def run_pipeline_phase1(
    url: str,
    config: Config,
    progress_cb: Callable[[float, str], None] | None = None,
    prefetched_metadata: VideoMetadata | None = None,
) -> Phase1Result:
    """Run steps 1-4: metadata, download, transcript, translation."""
    temp = TempManager(config.temp_dir)
    cost = CostTracker()
    temp.create_session()

    try:
        timings: list[tuple[str, float]] = []

        # Step 1: Metadata
        step = PIPELINE_STEPS[0]
        if progress_cb:
            progress_cb(step[1], step[0])
        t0 = time.time()
        if prefetched_metadata:
            metadata = prefetched_metadata
            logger.info(f"Using prefetched metadata: {metadata.title}")
        else:
            logger.info("Extracting metadata...")
            metadata = get_metadata(url)
        timings.append(("Metadata", time.time() - t0))
        logger.info(f"  -> {timings[-1][1]:.1f}s")
        if progress_cb:
            progress_cb(step[2], step[0])

        # Initialize cache
        cache = PipelineCache(config.cache_dir, metadata.video_id)

        # Step 2: Download video
        step = PIPELINE_STEPS[1]
        if progress_cb:
            progress_cb(step[1], step[0])
        t0 = time.time()

        cached_video = cache.get_video()
        if cached_video:
            video_path = cached_video
            logger.info(f"Using cached video for: {metadata.title}")
        else:
            logger.info(f"Downloading: {metadata.title}")
            video_dir = str(temp.subdirectory("video"))
            video_path = download_video(
                url, video_dir,
                progress_cb=_make_step_cb(progress_cb, step[0], step[1], step[2]),
            )
            cache.put_video(video_path)

        # Get exact duration from the downloaded/cached file
        actual_duration = get_video_duration(video_path)
        if abs(actual_duration - metadata.duration) > 0.5:
            logger.info(f"Duration corrected: {metadata.duration:.1f}s (metadata) -> {actual_duration:.2f}s (ffprobe)")
        metadata.duration = actual_duration

        timings.append(("Download", time.time() - t0))
        logger.info(f"  -> {timings[-1][1]:.1f}s")

        # Step 3: Extract transcript
        step = PIPELINE_STEPS[2]
        if progress_cb:
            progress_cb(step[1], step[0])
        t0 = time.time()

        transcript_key = f"{config.whisper_model_size}_{config.segment_min_duration}_{config.segment_max_duration}"
        cached_transcript = cache.get_transcript(transcript_key)
        if cached_transcript:
            original_segments = cached_transcript
            logger.info(f"Using cached transcript: {len(original_segments)} segments")
        else:
            logger.info("Extracting transcript...")
            cached_audio = cache.get_audio()
            if cached_audio:
                audio_path = cached_audio
            else:
                video_dir_for_audio = str(temp.subdirectory("video"))
                audio_path = extract_audio(video_path, video_dir_for_audio)
                cache.put_audio(audio_path)

            original_segments = extract_transcript(
                video_id=metadata.video_id,
                audio_path=audio_path,
                whisper_model_size=config.whisper_model_size,
                segment_min_duration=config.segment_min_duration,
                segment_max_duration=config.segment_max_duration,
                progress_cb=_make_step_cb(progress_cb, step[0], step[1], step[2]),
            )
            cache.put_transcript(transcript_key, original_segments)

        timings.append(("Transcript", time.time() - t0))
        logger.info(f"Got {len(original_segments)} segments -> {timings[-1][1]:.1f}s")

        # Step 4: Translate
        step = PIPELINE_STEPS[3]
        if progress_cb:
            progress_cb(step[1], step[0])
        t0 = time.time()

        trans_model = config.gemini_model if config.translation_provider == "gemini" else config.claude_model
        cached_translation = cache.get_translation(original_segments, config.translation_provider, trans_model)
        if cached_translation:
            translated_segments = cached_translation
            logger.info(f"Using cached translation: {len(translated_segments)} segments")
        else:
            logger.info(f"Translating with {config.translation_provider}...")
            translated_segments = translate_segments(
                original_segments, config,
                cost_tracker=cost,
                progress_cb=_make_step_cb(progress_cb, step[0], step[1], step[2]),
            )
            cache.put_translation(original_segments, config.translation_provider, trans_model, translated_segments)

        timings.append(("Translation", time.time() - t0))
        logger.info(f"  -> {timings[-1][1]:.1f}s")

        if progress_cb:
            progress_cb(PIPELINE_STEPS[3][2], "Translation complete — review translations")

        return Phase1Result(
            metadata=metadata,
            video_path=video_path,
            original_segments=original_segments,
            translated_segments=translated_segments,
            cost_tracker=cost,
            timings=timings,
            temp=temp,
            config=config,
            cache=cache,
        )

    except Exception as e:
        logger.error(f"Pipeline phase 1 failed: {e}", exc_info=True)
        temp.cleanup()
        raise


def run_pipeline_phase2(
    phase1: Phase1Result,
    translated_segments: list[TranscriptSegment],
    progress_cb: Callable[[float, str], None] | None = None,
) -> PipelineResult:
    """Run steps 5-8: TTS, audio sync, video merge, finalization."""
    temp = phase1.temp
    config = phase1.config
    cost = phase1.cost_tracker
    timings = phase1.timings
    metadata = phase1.metadata
    video_path = phase1.video_path
    original_segments = phase1.original_segments

    try:
        # Step 5: TTS
        step = PIPELINE_STEPS[4]
        if progress_cb:
            progress_cb(step[1], step[0])
        t0 = time.time()
        logger.info("Generating Armenian speech...")
        tts_dir = str(temp.subdirectory("tts"))
        tts_results = synthesize_all_segments(
            translated_segments, tts_dir, config,
            cost_tracker=cost,
            progress_cb=_make_step_cb(progress_cb, step[0], step[1], step[2]),
            cache=phase1.cache,
        )
        timings.append(("TTS", time.time() - t0))
        logger.info(f"  -> {timings[-1][1]:.1f}s")

        # Step 6: Vocal separation (optional)
        background_audio_path = None
        step = PIPELINE_STEPS[5]
        if progress_cb:
            progress_cb(step[1], step[0])
        t0 = time.time()

        if config.keep_background_music:
            cache = phase1.cache
            cached_bg = cache.get_background() if cache else None
            if cached_bg:
                background_audio_path = cached_bg
                logger.info("Using cached background audio")
            else:
                logger.info("Separating vocals from original audio...")
                # Get original audio (from cache or extract)
                audio_path = cache.get_audio() if cache else None
                if not audio_path:
                    audio_dir = str(temp.subdirectory("audio"))
                    audio_path = extract_audio(video_path, audio_dir)
                    if cache:
                        cache.put_audio(audio_path)

                separator_dir = str(temp.subdirectory("separator"))
                background_audio_path = separate_vocals(
                    audio_path, separator_dir,
                    method=config.vocal_separator,
                    api_key=config.lalal_api_key,
                )
                if cache:
                    background_audio_path = cache.put_background(background_audio_path)
        else:
            logger.info("Background music preservation disabled, using silence")

        timings.append(("Vocal separation", time.time() - t0))
        logger.info(f"  -> {timings[-1][1]:.1f}s")

        # Step 7: Sync audio
        step = PIPELINE_STEPS[6]
        if progress_cb:
            progress_cb(step[1], step[0])
        t0 = time.time()
        logger.info("Synchronizing dubbed audio...")
        dubbed_audio_path = temp.get_path("dubbed_audio.wav")
        _, time_regions, new_duration = create_dubbed_audio(
            tts_results, original_segments, metadata.duration,
            dubbed_audio_path, config,
            progress_cb=_make_step_cb(progress_cb, step[0], step[1], step[2]),
            background_audio_path=background_audio_path,
        )
        timings.append(("Audio sync", time.time() - t0))
        logger.info(f"  -> {timings[-1][1]:.1f}s")

        # Step 8: Merge (with variable-speed video if needed)
        step = PIPELINE_STEPS[7]
        if progress_cb:
            progress_cb(step[1], step[0])
        t0 = time.time()
        logger.info("Merging audio and video...")

        speed_video_path = temp.get_path("speed_video.mp4")
        speed_video_path = create_variable_speed_video(
            video_path, time_regions, speed_video_path,
        )

        output_path = temp.get_path("dubbed_video.mp4")
        merge_audio_video(speed_video_path, dubbed_audio_path, output_path)
        verify_output(output_path)
        timings.append(("Video merge", time.time() - t0))
        logger.info(f"  -> {timings[-1][1]:.1f}s")

        # Step 9: Finalize — save to videos/<id>_<title>/
        step = PIPELINE_STEPS[8]
        if progress_cb:
            progress_cb(step[1], step[0])

        if not re.match(r'^[a-zA-Z0-9_-]{1,20}$', metadata.video_id):
            raise ValueError(f"Invalid video ID format: {metadata.video_id!r}")
        dir_name = f"{metadata.video_id}_{_sanitize_dirname(metadata.title)}"
        video_out_dir = VIDEOS_DIR / dir_name

        final_path = _save_artifacts(
            video_out_dir, metadata,
            original_segments, translated_segments,
            cost, output_path, video_path, timings,
            background_audio_path=background_audio_path,
        )

        if progress_cb:
            progress_cb(1.0, "Done!")

        # Log cost + timing summary
        total_time = sum(t for _, t in timings)
        timing_lines = " | ".join(f"{n}: {t:.1f}s" for n, t in timings)
        logger.info(f"Timings: {timing_lines} | TOTAL: {total_time:.1f}s")
        logger.info("\n" + cost.summary())

        logger.info(f"Dubbing complete: {final_path}")

        return PipelineResult(
            output_video_path=final_path,
            output_dir=str(video_out_dir),
            metadata=metadata,
            original_segments=original_segments,
            translated_segments=translated_segments,
            cost_tracker=cost,
        )

    except Exception as e:
        logger.error(f"Pipeline phase 2 failed: {e}", exc_info=True)
        logger.info("\n" + cost.summary())
        return PipelineResult(
            output_video_path="",
            output_dir="",
            metadata=metadata,
            original_segments=original_segments,
            translated_segments=translated_segments,
            cost_tracker=cost,
            error=str(e),
        )
    finally:
        temp.cleanup()


def run_pipeline(
    url: str,
    config: Config,
    progress_cb: Callable[[float, str], None] | None = None,
    prefetched_metadata: VideoMetadata | None = None,
) -> PipelineResult:
    """Run the full pipeline (both phases) without pausing."""
    try:
        phase1 = run_pipeline_phase1(url, config, progress_cb, prefetched_metadata)
        return run_pipeline_phase2(phase1, phase1.translated_segments, progress_cb)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return PipelineResult(
            output_video_path="",
            output_dir="",
            metadata=VideoMetadata("", 0, "", "", ""),
            original_segments=[],
            translated_segments=[],
            cost_tracker=CostTracker(),
            error=str(e),
        )
