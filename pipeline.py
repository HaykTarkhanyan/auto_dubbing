import logging
import shutil
from dataclasses import dataclass, field
from typing import Callable

from config import Config
from modules.downloader import get_metadata, download_video, extract_audio, VideoMetadata
from modules.transcript import TranscriptSegment, extract_transcript
from modules.translator import translate_segments
from modules.tts import TTSResult, synthesize_all_segments
from modules.audio_sync import create_dubbed_audio
from modules.video_merge import merge_audio_video, verify_output
from modules.temp_manager import TempManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    output_video_path: str
    metadata: VideoMetadata
    original_segments: list[TranscriptSegment]
    translated_segments: list[TranscriptSegment]
    error: str | None = None


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

    with temp.session():
        try:
            # Step 1: Metadata
            step = PIPELINE_STEPS[0]
            if progress_cb:
                progress_cb(step[1], step[0])
            logger.info("Extracting metadata...")
            metadata = get_metadata(url)
            if progress_cb:
                progress_cb(step[2], step[0])

            # Step 2: Download video
            step = PIPELINE_STEPS[1]
            if progress_cb:
                progress_cb(step[1], step[0])
            logger.info(f"Downloading: {metadata.title}")
            video_dir = str(temp.subdirectory("video"))
            video_path = download_video(
                url, video_dir,
                progress_cb=_make_step_cb(progress_cb, step[0], step[1], step[2]),
            )

            # Step 3: Extract transcript
            step = PIPELINE_STEPS[2]
            if progress_cb:
                progress_cb(step[1], step[0])
            logger.info("Extracting transcript...")

            # Extract audio for potential Whisper fallback
            audio_path = extract_audio(video_path, video_dir)

            original_segments = extract_transcript(
                video_id=metadata.video_id,
                audio_path=audio_path,
                whisper_model_size=config.whisper_model_size,
                progress_cb=_make_step_cb(progress_cb, step[0], step[1], step[2]),
            )
            logger.info(f"Got {len(original_segments)} transcript segments")

            # Step 4: Translate
            step = PIPELINE_STEPS[3]
            if progress_cb:
                progress_cb(step[1], step[0])
            logger.info(f"Translating with {config.translation_provider}...")
            translated_segments = translate_segments(
                original_segments, config,
                progress_cb=_make_step_cb(progress_cb, step[0], step[1], step[2]),
            )

            # Step 5: TTS
            step = PIPELINE_STEPS[4]
            if progress_cb:
                progress_cb(step[1], step[0])
            logger.info("Generating Armenian speech...")
            tts_dir = str(temp.subdirectory("tts"))
            tts_results = synthesize_all_segments(
                translated_segments, tts_dir, config,
                progress_cb=_make_step_cb(progress_cb, step[0], step[1], step[2]),
            )

            # Step 6: Sync audio
            step = PIPELINE_STEPS[5]
            if progress_cb:
                progress_cb(step[1], step[0])
            logger.info("Synchronizing dubbed audio...")
            dubbed_audio_path = temp.get_path("dubbed_audio.wav")
            create_dubbed_audio(
                tts_results, original_segments, metadata.duration,
                dubbed_audio_path, config,
                progress_cb=_make_step_cb(progress_cb, step[0], step[1], step[2]),
            )

            # Step 7: Merge
            step = PIPELINE_STEPS[6]
            if progress_cb:
                progress_cb(step[1], step[0])
            logger.info("Merging audio and video...")
            output_path = temp.get_path("dubbed_video.mp4")
            merge_audio_video(video_path, dubbed_audio_path, output_path)
            verify_output(output_path)

            # Step 8: Finalize — copy to a persistent location
            step = PIPELINE_STEPS[7]
            if progress_cb:
                progress_cb(step[1], step[0])

            # Copy output out of temp dir before cleanup
            import tempfile
            final_path = tempfile.mktemp(suffix=".mp4", prefix="dubbed_")
            shutil.copy2(output_path, final_path)

            if progress_cb:
                progress_cb(1.0, "Done!")

            logger.info(f"Dubbing complete: {final_path}")

            return PipelineResult(
                output_video_path=final_path,
                metadata=metadata,
                original_segments=original_segments,
                translated_segments=translated_segments,
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return PipelineResult(
                output_video_path="",
                metadata=metadata if "metadata" in dir() else VideoMetadata("", 0, "", "", ""),
                original_segments=original_segments if "original_segments" in dir() else [],
                translated_segments=translated_segments if "translated_segments" in dir() else [],
                error=str(e),
            )
