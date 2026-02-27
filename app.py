import logging
import io
import sys
import shutil
import gradio as gr

from config import Config
from pipeline import run_pipeline, run_pipeline_phase1, run_pipeline_phase2, Phase1Result
from modules.transcript import TranscriptSegment

# Fix Windows console encoding for Armenian text
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# File handler
file_handler = logging.FileHandler("auto_dubbing.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

logging.basicConfig(level=logging.DEBUG, handlers=[console_handler, file_handler])

# Suppress noisy third-party loggers
for _name in ("httpcore", "httpx", "google_genai", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _build_config(translation_provider, tts_provider, whisper_model, speed_max, keep_background):
    config = Config.from_env()
    config.translation_provider = "claude" if translation_provider == "Claude Sonnet" else "gemini"
    config.tts_provider = "edge_tts" if tts_provider == "Edge TTS (Free)" else "gemini"
    config.whisper_model_size = whisper_model
    config.speed_max = speed_max
    config.keep_background_music = keep_background
    errors = config.validate()
    if errors:
        raise gr.Error(f"Configuration error: {'; '.join(errors)}")
    return config


def _build_table(original_segments, translated_segments):
    table_data = []
    for orig, trans in zip(original_segments, translated_segments):
        time_str = f"{orig.start:.1f}s"
        table_data.append([time_str, orig.text, trans.text])
    return table_data


def _build_cost_text(metadata, cost):
    return (
        f"Dubbed: **{metadata.title}** ({metadata.duration:.0f}s)\n\n"
        f"**API Costs:** Translation: ${cost.translation_cost:.4f} | "
        f"TTS: ${cost.tts_cost:.4f} ({cost.tts_calls} calls) | "
        f"**Total: ${cost.total_cost:.4f}**"
    )


def run_phase1(
    url, translation_provider, tts_provider, whisper_model, speed_max,
    keep_background, skip_review, state, progress=gr.Progress(),
):
    if not url.strip():
        raise gr.Error("Please enter a YouTube URL")
    if not check_ffmpeg():
        raise gr.Error("ffmpeg is not installed. Please install ffmpeg and add it to your PATH.")

    config = _build_config(translation_provider, tts_provider, whisper_model, speed_max, keep_background)

    def progress_cb(frac, status):
        progress(frac, desc=status)

    if skip_review:
        # Run full pipeline without pausing
        result = run_pipeline(url, config, progress_cb=progress_cb)
        if result.error:
            raise gr.Error(f"Dubbing failed: {result.error}")

        table_data = _build_table(result.original_segments, result.translated_segments)
        cost_text = _build_cost_text(result.metadata, result.cost_tracker)

        return (
            state,                                                  # state (unchanged)
            table_data,                                             # transcript_table
            cost_text,                                              # status_text
            result.output_video_path,                               # output_video
            gr.update(visible=True, value=result.output_video_path),  # download_btn
            gr.update(visible=True),                                # dub_button
            gr.update(visible=False),                               # continue_btn
        )

    # Run phase 1 only (steps 1-4)
    phase1 = run_pipeline_phase1(url, config, progress_cb=progress_cb)

    table_data = _build_table(phase1.original_segments, phase1.translated_segments)
    cost = phase1.cost_tracker
    status = (
        f"Translation complete for **{phase1.metadata.title}** ({phase1.metadata.duration:.0f}s)\n\n"
        f"Review and edit the Armenian translations below, then click **Continue Dubbing**.\n\n"
        f"Translation cost: ${cost.translation_cost:.4f}"
    )

    return (
        phase1,                         # state
        table_data,                     # transcript_table
        status,                         # status_text
        None,                           # output_video (no video yet)
        gr.update(visible=False),       # download_btn
        gr.update(visible=False),       # dub_button (hide during review)
        gr.update(visible=True),        # continue_btn
    )


def run_phase2(state, edited_table, progress=gr.Progress()):
    phase1: Phase1Result = state
    if phase1 is None:
        raise gr.Error("No translation to continue. Please run Start Dubbing first.")

    # Rebuild translated_segments from the edited table
    translated_segments = []
    for i, row in enumerate(edited_table):
        orig = phase1.original_segments[i]
        edited_text = str(row[2]) if len(row) > 2 else orig.text
        translated_segments.append(TranscriptSegment(
            text=edited_text,
            start=orig.start,
            duration=orig.duration,
        ))

    def progress_cb(frac, status):
        progress(frac, desc=status)

    result = run_pipeline_phase2(phase1, translated_segments, progress_cb=progress_cb)

    if result.error:
        raise gr.Error(f"Dubbing failed: {result.error}")

    table_data = _build_table(result.original_segments, result.translated_segments)
    cost_text = _build_cost_text(result.metadata, result.cost_tracker)

    return (
        None,                                                   # state (clear it)
        table_data,                                             # transcript_table
        cost_text,                                              # status_text
        result.output_video_path,                               # output_video
        gr.update(visible=True, value=result.output_video_path),  # download_btn
        gr.update(visible=True),                                # dub_button
        gr.update(visible=False),                               # continue_btn
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Auto Dubbing - EN → Armenian", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Auto Dubbing Tool\n### English → Armenian Video Dubbing")

        state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    lines=1,
                )

                with gr.Accordion("Translation Settings", open=True):
                    translation_provider = gr.Radio(
                        choices=["Gemini Pro", "Claude Sonnet"],
                        value="Gemini Pro",
                        label="Translation Model",
                    )

                with gr.Accordion("Advanced Settings", open=False):
                    tts_provider = gr.Radio(
                        choices=["Gemini TTS", "Edge TTS (Free)"],
                        value="Gemini TTS",
                        label="TTS Provider",
                    )
                    whisper_model = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large"],
                        value="base",
                        label="Whisper Model (fallback if no captions)",
                    )
                    speed_max = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.35,
                        step=0.05,
                        label="Max Speed Factor",
                    )
                    keep_background = gr.Checkbox(
                        label="Keep background music",
                        value=True,
                    )
                    skip_review = gr.Checkbox(
                        label="Skip translation review",
                        value=False,
                    )

                dub_button = gr.Button("Start Dubbing", variant="primary", size="lg")
                continue_btn = gr.Button("Continue Dubbing", variant="primary", size="lg", visible=False)

            with gr.Column(scale=1):
                status_text = gr.Markdown("Ready. Paste a YouTube URL and click **Start Dubbing**.")

                with gr.Tabs():
                    with gr.TabItem("Video"):
                        output_video = gr.Video(label="Dubbed Video")
                    with gr.TabItem("Transcript"):
                        transcript_table = gr.Dataframe(
                            headers=["Time", "Original (EN)", "Armenian (HY)"],
                            label="Translation Comparison",
                            interactive=True,
                        )

                download_btn = gr.DownloadButton("Download Dubbed Video", visible=False)

        phase1_outputs = [state, transcript_table, status_text, output_video, download_btn, dub_button, continue_btn]

        dub_button.click(
            fn=run_phase1,
            inputs=[url_input, translation_provider, tts_provider, whisper_model, speed_max, keep_background, skip_review, state],
            outputs=phase1_outputs,
        )

        continue_btn.click(
            fn=run_phase2,
            inputs=[state, transcript_table],
            outputs=phase1_outputs,
        )

    return app


if __name__ == "__main__":
    if not check_ffmpeg():
        print("WARNING: ffmpeg is not installed. The app will not work without it.")
        print("Install ffmpeg: https://ffmpeg.org/download.html")

    app = build_ui()
    app.launch()
