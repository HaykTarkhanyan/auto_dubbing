import logging
import os
import shutil
import gradio as gr

from config import Config
from pipeline import run_pipeline

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# File handler
file_handler = logging.FileHandler("auto_dubbing.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

logging.basicConfig(level=logging.DEBUG, handlers=[console_handler, file_handler])


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def run_dubbing(
    url: str,
    translation_provider: str,
    tts_provider: str,
    whisper_model: str,
    speed_max: float,
    progress=gr.Progress(),
):
    if not url.strip():
        raise gr.Error("Please enter a YouTube URL")

    if not check_ffmpeg():
        raise gr.Error("ffmpeg is not installed. Please install ffmpeg and add it to your PATH.")

    # Build config from env + UI overrides
    config = Config.from_env()

    if translation_provider == "Claude Sonnet":
        config.translation_provider = "claude"
    else:
        config.translation_provider = "gemini"

    if tts_provider == "Edge TTS (Free)":
        config.tts_provider = "edge_tts"
    else:
        config.tts_provider = "gemini"

    config.whisper_model_size = whisper_model
    config.speed_max = speed_max

    # Validate
    errors = config.validate()
    if errors:
        raise gr.Error(f"Configuration error: {'; '.join(errors)}")

    def progress_cb(frac: float, status: str):
        progress(frac, desc=status)

    result = run_pipeline(url, config, progress_cb=progress_cb)

    if result.error:
        raise gr.Error(f"Dubbing failed: {result.error}")

    # Build transcript comparison table
    table_data = []
    for orig, trans in zip(result.original_segments, result.translated_segments):
        time_str = f"{orig.start:.1f}s"
        table_data.append([time_str, orig.text, trans.text])

    return (
        result.output_video_path,
        table_data,
        f"Dubbed: **{result.metadata.title}** ({result.metadata.duration:.0f}s)",
        gr.update(visible=True, value=result.output_video_path),
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Auto Dubbing - EN → Armenian", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Auto Dubbing Tool\n### English → Armenian Video Dubbing")

        with gr.Row():
            with gr.Column(scale=1):
                url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    lines=1,
                )

                with gr.Accordion("Translation Settings", open=True):
                    translation_provider = gr.Radio(
                        choices=["Claude Sonnet", "Gemini Pro"],
                        value="Claude Sonnet",
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

                dub_button = gr.Button("Start Dubbing", variant="primary", size="lg")

            with gr.Column(scale=1):
                status_text = gr.Markdown("Ready. Paste a YouTube URL and click **Start Dubbing**.")

                with gr.Tabs():
                    with gr.TabItem("Video"):
                        output_video = gr.Video(label="Dubbed Video")
                    with gr.TabItem("Transcript"):
                        transcript_table = gr.Dataframe(
                            headers=["Time", "Original (EN)", "Armenian (HY)"],
                            label="Translation Comparison",
                        )

                download_btn = gr.DownloadButton("Download Dubbed Video", visible=False)

        dub_button.click(
            fn=run_dubbing,
            inputs=[url_input, translation_provider, tts_provider, whisper_model, speed_max],
            outputs=[output_video, transcript_table, status_text, download_btn],
        )

    return app


if __name__ == "__main__":
    if not check_ffmpeg():
        print("WARNING: ffmpeg is not installed. The app will not work without it.")
        print("Install ffmpeg: https://ffmpeg.org/download.html")

    app = build_ui()
    app.launch()
