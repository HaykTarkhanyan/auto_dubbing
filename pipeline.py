import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from config import Config
from modules.downloader import get_metadata, download_video, extract_audio, get_video_duration, VideoMetadata
from modules.transcript import TranscriptSegment, extract_transcript
from modules.translator import translate_segments
from modules.tts import TTSResult, synthesize_all_segments
from modules.audio_sync import create_dubbed_audio
from modules.video_merge import create_variable_speed_video, merge_audio_video, verify_output
from modules.temp_manager import TempManager
from utils.cost_tracker import CostTracker

VIDEOS_DIR = Path(__file__).parent / "videos"

logger = logging.getLogger(__name__)


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

    logger.info(f"Saved all artifacts to {video_dir}")
    return final_video_path


# (step_name, global_start, global_end)
PIPELINE_STEPS = [
    ("Extracting metadata", 0.00, 0.05),
    ("Downloading video", 0.05, 0.20),
    ("Extracting transcript", 0.20, 0.35),
    ("Translating to Armenian", 0.35, 0.55),
    ("Generating speech", 0.55, 0.80),
    ("Synchronizing audio", 0.80, 0.90),
    ("Merging video", 0.90, 0.98),
    ("Finalizing", 0.98, 1.00),
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


def run_pipeline(
    url: str,
    config: Config,
    progress_cb: Callable[[float, str], None] | None = None,
) -> PipelineResult:
    temp = TempManager(config.temp_dir)
    cost = CostTracker()

    with temp.session():
        try:
            timings: list[tuple[str, float]] = []

            # Step 1: Metadata
            step = PIPELINE_STEPS[0]
            if progress_cb:
                progress_cb(step[1], step[0])
            t0 = time.time()
            logger.info("Extracting metadata...")
            metadata = get_metadata(url)
            timings.append(("Metadata", time.time() - t0))
            logger.info(f"  -> {timings[-1][1]:.1f}s")
            if progress_cb:
                progress_cb(step[2], step[0])

            # Step 2: Download video
            step = PIPELINE_STEPS[1]
            if progress_cb:
                progress_cb(step[1], step[0])
            t0 = time.time()
            logger.info(f"Downloading: {metadata.title}")
            video_dir = str(temp.subdirectory("video"))
            video_path = download_video(
                url, video_dir,
                progress_cb=_make_step_cb(progress_cb, step[0], step[1], step[2]),
            )
            # Get exact duration from the downloaded file (yt-dlp metadata can be approximate)
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
            logger.info("Extracting transcript...")

            audio_path = extract_audio(video_path, video_dir)

            original_segments = extract_transcript(
                video_id=metadata.video_id,
                audio_path=audio_path,
                whisper_model_size=config.whisper_model_size,
                progress_cb=_make_step_cb(progress_cb, step[0], step[1], step[2]),
            )
            timings.append(("Transcript", time.time() - t0))
            logger.info(f"Got {len(original_segments)} segments -> {timings[-1][1]:.1f}s")

            # Step 4: Translate
            step = PIPELINE_STEPS[3]
            if progress_cb:
                progress_cb(step[1], step[0])
            t0 = time.time()
            logger.info(f"Translating with {config.translation_provider}...")
            translated_segments = translate_segments(
                original_segments, config,
                cost_tracker=cost,
                progress_cb=_make_step_cb(progress_cb, step[0], step[1], step[2]),
            )
            timings.append(("Translation", time.time() - t0))
            logger.info(f"  -> {timings[-1][1]:.1f}s")

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
            )
            timings.append(("TTS", time.time() - t0))
            logger.info(f"  -> {timings[-1][1]:.1f}s")

            # Step 6: Sync audio
            step = PIPELINE_STEPS[5]
            if progress_cb:
                progress_cb(step[1], step[0])
            t0 = time.time()
            logger.info("Synchronizing dubbed audio...")
            dubbed_audio_path = temp.get_path("dubbed_audio.wav")
            _, time_regions, new_duration = create_dubbed_audio(
                tts_results, original_segments, metadata.duration,
                dubbed_audio_path, config,
                progress_cb=_make_step_cb(progress_cb, step[0], step[1], step[2]),
            )
            timings.append(("Audio sync", time.time() - t0))
            logger.info(f"  -> {timings[-1][1]:.1f}s")

            # Step 7: Merge (with variable-speed video if needed)
            step = PIPELINE_STEPS[6]
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

            # Step 8: Finalize — save to videos/<id>_<title>/
            step = PIPELINE_STEPS[7]
            if progress_cb:
                progress_cb(step[1], step[0])

            dir_name = f"{metadata.video_id}_{_sanitize_dirname(metadata.title)}"
            video_out_dir = VIDEOS_DIR / dir_name

            final_path = _save_artifacts(
                video_out_dir, metadata,
                original_segments, translated_segments,
                cost, output_path, video_path, timings,
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
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            logger.info("\n" + cost.summary())
            return PipelineResult(
                output_video_path="",
                output_dir="",
                metadata=metadata if "metadata" in dir() else VideoMetadata("", 0, "", "", ""),
                original_segments=original_segments if "original_segments" in dir() else [],
                translated_segments=translated_segments if "translated_segments" in dir() else [],
                cost_tracker=cost,
                error=str(e),
            )
