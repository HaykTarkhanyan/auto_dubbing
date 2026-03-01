import html as html_mod
import logging
import io
import sys
import shutil
import threading
import traceback
import gradio as gr

from config import Config
from pipeline import run_pipeline, run_pipeline_phase1, run_pipeline_phase2, Phase1Result
from modules.downloader import get_metadata, get_video_file_info
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


# ── TTS Voices grouped by gender with style descriptors ───────────
VOICE_INFO: dict[str, tuple[str, str]] = {
    # Male voices — (gender, style)
    "Charon":         ("Male",   "Informative"),
    "Puck":           ("Male",   "Upbeat"),
    "Fenrir":         ("Male",   "Excitable"),
    "Orus":           ("Male",   "Firm"),
    "Enceladus":      ("Male",   "Breathy"),
    "Iapetus":        ("Male",   "Clear"),
    "Umbriel":        ("Male",   "Easy-going"),
    "Algieba":        ("Male",   "Smooth"),
    "Algenib":        ("Male",   "Gravelly"),
    "Rasalgethi":     ("Male",   "Informative"),
    "Alnilam":        ("Male",   "Firm"),
    "Schedar":        ("Male",   "Even"),
    "Achird":         ("Male",   "Friendly"),
    "Zubenelgenubi":  ("Male",   "Casual"),
    "Sadachbia":      ("Male",   "Lively"),
    "Sadaltager":     ("Male",   "Knowledgeable"),
    # Female voices
    "Kore":           ("Female", "Firm"),
    "Zephyr":         ("Female", "Bright"),
    "Leda":           ("Female", "Youthful"),
    "Aoede":          ("Female", "Breezy"),
    "Callirrhoe":     ("Female", "Easy-going"),
    "Autonoe":        ("Female", "Bright"),
    "Despina":        ("Female", "Smooth"),
    "Erinome":        ("Female", "Clear"),
    "Laomedeia":      ("Female", "Upbeat"),
    "Achernar":       ("Female", "Soft"),
    "Gacrux":         ("Female", "Mature"),
    "Pulcherrima":    ("Female", "Forward"),
    "Vindemiatrix":   ("Female", "Gentle"),
    "Sulafat":        ("Female", "Warm"),
}
MALE_VOICES = [v for v, (g, _) in VOICE_INFO.items() if g == "Male"]
FEMALE_VOICES = [v for v, (g, _) in VOICE_INFO.items() if g == "Female"]
TTS_VOICE_CHOICES = (
    [(f"{v} — {VOICE_INFO[v][1]} ({VOICE_INFO[v][0]})", v) for v in MALE_VOICES]
    + [(f"{v} — {VOICE_INFO[v][1]} ({VOICE_INFO[v][0]})", v) for v in FEMALE_VOICES]
)


# ── In-memory log capture ──────────────────────────────────────────
class LogCapture(logging.Handler):
    """Thread-safe ring buffer of recent log lines for the Gradio UI."""

    def __init__(self, max_lines: int = 200):
        super().__init__(level=logging.INFO)
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._max = max_lines

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        with self._lock:
            self._lines.append(msg)
            if len(self._lines) > self._max:
                self._lines = self._lines[-self._max:]

    def get_text(self) -> str:
        with self._lock:
            return "\n".join(self._lines)

    def clear(self):
        with self._lock:
            self._lines.clear()


log_capture = LogCapture()
logging.getLogger().addHandler(log_capture)


# ── Number of outputs shared by all handlers ──────────────────────
# state, transcript_table, status_text, output_video,
# download_btn, dub_button, continue_btn, log_output, video_metadata
_N_OUTPUTS = 9


def _no_change_tuple():
    """Return a tuple of gr.update() for all outputs (no visual change)."""
    return tuple(gr.update() for _ in range(_N_OUTPUTS))


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _build_config(translation_provider, whisper_model, speed_max, keep_background, tts_voice, vocal_sep):
    config = Config.from_env()
    config.translation_provider = "claude" if translation_provider == "Claude Sonnet" else "gemini"
    config.whisper_model_size = whisper_model
    config.speed_max = speed_max
    config.keep_background_music = keep_background
    config.tts_voice_name = tts_voice
    config.vocal_separator = vocal_sep
    errors = config.validate()
    if errors:
        raise gr.Error(f"Configuration error: {'; '.join(errors)}")
    return config


def _build_table(original_segments, translated_segments):
    table_data = []
    for orig, trans in zip(original_segments, translated_segments):
        time_str = f"{orig.start:.1f}–{orig.end:.1f}s"
        table_data.append([time_str, orig.text, trans.text])
    return table_data


def _build_metadata_html(metadata) -> str:
    """Build an HTML card showing video metadata with thumbnail."""
    thumb_url = f"https://img.youtube.com/vi/{html_mod.escape(metadata.video_id)}/hqdefault.jpg"
    video_url = f"https://www.youtube.com/watch?v={html_mod.escape(metadata.video_id)}"
    title = html_mod.escape(metadata.title)
    uploader = html_mod.escape(metadata.uploader)
    mins, secs = divmod(int(metadata.duration), 60)
    duration_str = f"{mins}:{secs:02d}"

    return (
        '<div class="meta-card">'
        f'<a href="{video_url}" target="_blank" class="meta-thumb">'
        f'<img src="{thumb_url}" /></a>'
        '<div class="meta-info">'
        f'<div class="meta-title">{title}</div>'
        f'<div class="meta-channel">{uploader}</div>'
        f'<div class="meta-extra">'
        f'{duration_str} &bull; '
        f'<a href="{video_url}" target="_blank">YouTube &#8599;</a>'
        '</div></div></div>'
    )


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _build_cost_text(metadata, cost, timings=None, video_path=None):
    lines = [
        f"### Dubbing complete!",
        f"**{metadata.title}** ({metadata.duration:.0f}s)",
        "",
    ]

    # Video file info
    if video_path:
        try:
            info = get_video_file_info(video_path)
            lines.append(
                f"**Output:** {info.width}x{info.height} &bull; "
                f"{_format_file_size(info.file_size_bytes)}"
            )
        except Exception:
            pass

    lines.append(
        f"**Cost:** Translation ${cost.translation_cost:.4f} + "
        f"TTS ${cost.tts_cost:.4f} ({cost.tts_calls} calls) = "
        f"**${cost.total_cost:.4f}**"
    )

    if timings:
        total_time = sum(t for _, t in timings)
        timing_parts = " | ".join(f"{n}: {t:.1f}s" for n, t in timings)
        lines.append(f"\n**Timings:** {timing_parts} | **Total: {total_time:.1f}s**")
    return "\n".join(lines)


def _format_error(e: Exception) -> str:
    """Build a user-friendly error message with the full traceback."""
    tb = traceback.format_exception(type(e), e, e.__traceback__)
    short = str(e)

    # Detect common errors and add guidance
    if "Numba needs NumPy" in short or "numba" in short.lower():
        hint = "Fix: `pip install 'numpy<2.1'` to match numba requirements."
    elif "RESOURCE_EXHAUSTED" in short or "429" in short:
        hint = "Your Gemini API quota is exhausted. Wait for it to reset or use a different API key."
    elif "RequestBlocked" in short:
        hint = "YouTube is blocking transcript requests from your IP. The pipeline will use Whisper instead."
    elif "ANTHROPIC_API_KEY" in short or "GOOGLE_API_KEY" in short:
        hint = "Set the missing API key in your .env file."
    elif "ffmpeg" in short.lower() or "ffprobe" in short.lower():
        hint = "Make sure ffmpeg is installed and on your PATH."
    else:
        hint = ""

    msg = f"**Error:** {short}"
    if hint:
        msg += f"\n\n**Hint:** {hint}"
    msg += f"\n\n<details><summary>Full traceback</summary>\n\n```\n{''.join(tb)}```\n</details>"
    return msg


def _yield_log_updates():
    """Build a tuple that only updates the log_output component (index 7)."""
    updates = list(_no_change_tuple())
    updates[7] = log_capture.get_text()
    return tuple(updates)


# ── Phase 1: Metadata → Download → Transcript → Translate ────────
def run_phase1(
    url, translation_provider, whisper_model, speed_max,
    keep_background, skip_review, tts_voice, vocal_sep, state,
):
    if not url.strip():
        raise gr.Error("Please enter a YouTube URL")
    if not check_ffmpeg():
        raise gr.Error(
            "ffmpeg is not installed. Install it from https://ffmpeg.org/download.html "
            "and make sure it's on your PATH."
        )

    log_capture.clear()
    config = _build_config(translation_provider, whisper_model, speed_max, keep_background, tts_voice, vocal_sep)

    # ── Immediate yield: fetch metadata and show it right away ──
    metadata = None
    meta_html = ""
    try:
        metadata = get_metadata(url.strip())
        meta_html = _build_metadata_html(metadata)
    except Exception:
        pass

    yield (
        state, [], f"Starting pipeline{f' for **{metadata.title}**' if metadata else ''}...",
        None, gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=False), "", meta_html,
    )

    # ── Run pipeline in background thread, yield log updates ──
    result_box: dict = {}

    def _run():
        try:
            if skip_review:
                result_box["value"] = run_pipeline(url, config, prefetched_metadata=metadata)
            else:
                result_box["value"] = run_pipeline_phase1(url, config, prefetched_metadata=metadata)
        except Exception as e:
            result_box["error"] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    while thread.is_alive():
        thread.join(timeout=2)
        if thread.is_alive():
            yield _yield_log_updates()

    # ── Thread finished — yield final result ──
    if "error" in result_box:
        yield (
            state, [], _format_error(result_box["error"]),
            None, gr.update(visible=False),
            gr.update(visible=True), gr.update(visible=False),
            log_capture.get_text(), meta_html,
        )
        return

    if skip_review:
        result = result_box["value"]
        if result.metadata.video_id:
            meta_html = _build_metadata_html(result.metadata)
        if result.error:
            yield (
                state, [], _format_error(RuntimeError(result.error)),
                None, gr.update(visible=False), gr.update(visible=True),
                gr.update(visible=False), log_capture.get_text(), meta_html,
            )
            return

        table_data = _build_table(result.original_segments, result.translated_segments)
        cost_text = _build_cost_text(result.metadata, result.cost_tracker, video_path=result.output_video_path)

        yield (
            state, table_data, cost_text,
            result.output_video_path,
            gr.update(visible=True, value=result.output_video_path),
            gr.update(visible=True), gr.update(visible=False),
            log_capture.get_text(), meta_html,
        )
    else:
        phase1 = result_box["value"]
        meta_html = _build_metadata_html(phase1.metadata)

        table_data = _build_table(phase1.original_segments, phase1.translated_segments)
        cost = phase1.cost_tracker
        n = len(phase1.translated_segments)
        total_dur = sum(seg.duration for seg in phase1.original_segments)
        status = (
            f"### Translation ready for review\n\n"
            f"{n} segments, {total_dur:.0f}s of speech\n\n"
            f"Edit the **Armenian (HY)** column below, then click **Continue Dubbing**.\n\n"
            f"Translation cost so far: ${cost.translation_cost:.4f}"
        )

        yield (
            phase1, table_data, status,
            None, gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=True),
            log_capture.get_text(), meta_html,
        )


# ── Phase 2: TTS → Vocal separation → Audio sync → Video merge ───
def run_phase2(state, edited_table):
    phase1: Phase1Result = state
    if phase1 is None:
        raise gr.Error("No translation to continue. Please run Start Dubbing first.")

    log_capture.clear()
    meta_html = _build_metadata_html(phase1.metadata)

    translated_segments = []
    for i, row in enumerate(edited_table):
        orig = phase1.original_segments[i]
        edited_text = str(row[2]) if len(row) > 2 else orig.text
        translated_segments.append(TranscriptSegment(
            text=edited_text,
            start=orig.start,
            duration=orig.duration,
        ))

    yield (
        gr.update(), gr.update(), "Generating dubbed video...",
        gr.update(), gr.update(), gr.update(visible=False),
        gr.update(visible=False), "", meta_html,
    )

    # ── Run phase 2 in background thread, yield log updates ──
    result_box: dict = {}

    def _run():
        try:
            result_box["value"] = run_pipeline_phase2(phase1, translated_segments)
        except Exception as e:
            result_box["error"] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    while thread.is_alive():
        thread.join(timeout=2)
        if thread.is_alive():
            yield _yield_log_updates()

    # ── Thread finished — yield final result ──
    if "error" in result_box:
        yield (
            state, gr.update(), _format_error(result_box["error"]),
            None, gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=True),
            log_capture.get_text(), meta_html,
        )
        return

    result = result_box["value"]
    if result.error:
        yield (
            state, gr.update(), _format_error(RuntimeError(result.error)),
            None, gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=True),
            log_capture.get_text(), meta_html,
        )
        return

    table_data = _build_table(result.original_segments, result.translated_segments)
    timings = getattr(phase1, "timings", None)
    cost_text = _build_cost_text(result.metadata, result.cost_tracker, timings, video_path=result.output_video_path)

    yield (
        None, table_data, cost_text,
        result.output_video_path,
        gr.update(visible=True, value=result.output_video_path),
        gr.update(visible=True), gr.update(visible=False),
        log_capture.get_text(), meta_html,
    )


# ── Reset ─────────────────────────────────────────────────────────
def reset_ui():
    log_capture.clear()
    return (
        None,
        [],
        "Paste a YouTube URL and click **Start Dubbing** to begin.",
        None,
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        "",
        "",
    )


CSS = """
/* ── Hide footer ── */
footer { display: none !important; }

/* ── Header ── */
.main-header { text-align: center; margin-bottom: 0.2em; }
.main-header h1 { margin: 0; font-size: 1.6em; }
.main-header p { color: #888; margin: 0.15em 0 0 0; font-size: 0.95em; }

/* ── Metadata card ── */
.meta-card {
    display: flex; gap: 14px; padding: 12px 14px;
    border-radius: 10px; border: 1px solid var(--border-color-primary);
    background: var(--background-fill-secondary);
}
.meta-thumb img {
    width: 178px; height: 100px; border-radius: 8px;
    object-fit: cover; display: block;
}
.meta-info { flex: 1; min-width: 0; display: flex; flex-direction: column; justify-content: center; }
.meta-title { font-weight: 600; font-size: 14px; line-height: 1.35; }
.meta-channel { color: var(--body-text-color-subdued); font-size: 13px; margin-top: 3px; }
.meta-extra { color: var(--body-text-color-subdued); font-size: 12px; margin-top: 5px; }
.meta-extra a { color: var(--link-text-color); text-decoration: none; }
.meta-extra a:hover { text-decoration: underline; }

/* ── Status area ── */
.status-box { min-height: 60px; }

/* ── Settings panel ── */
.settings-panel { padding-top: 4px; }
.settings-panel .gr-group { margin-bottom: 4px !important; }

/* ── Buttons ── */
.action-buttons { margin-top: 8px; }

/* ── Tabs ── */
.output-tabs .tabitem { padding-top: 8px !important; }

/* ── Transcript table ── */
.transcript-table { font-size: 13px; }
"""


def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )

    with gr.Blocks(title="Auto Dubbing — EN → HY", theme=theme, css=CSS) as app:

        gr.HTML(
            '<div class="main-header">'
            "<h1>Auto Dubbing</h1>"
            "<p>English &rarr; Armenian YouTube Video Dubbing</p>"
            "</div>"
        )

        state = gr.State(value=None)

        with gr.Row(equal_height=False):
            # ── Left panel: inputs & settings ──
            with gr.Column(scale=2, min_width=320, elem_classes=["settings-panel"]):
                url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    lines=1,
                    max_lines=1,
                )

                translation_provider = gr.Radio(
                    choices=["Gemini Pro", "Claude Sonnet"],
                    value="Gemini Pro",
                    label="Translation Model",
                )
                tts_voice = gr.Dropdown(
                    choices=TTS_VOICE_CHOICES,
                    value="Charon",
                    label="TTS Voice",
                    info="Gemini voice for Armenian speech",
                )

                with gr.Row():
                    keep_background = gr.Checkbox(
                        label="Keep background music",
                        value=True,
                        info="Mix TTS over original music/SFX",
                    )
                    skip_review = gr.Checkbox(
                        label="Skip translation review",
                        value=False,
                        info="No pause to edit translations",
                    )

                with gr.Accordion("Advanced Settings", open=False):
                    vocal_sep = gr.Radio(
                        choices=["lalal", "demucs", "mdx"],
                        value="lalal",
                        label="Vocal Separator",
                        info="LALAL.AI (cloud) / Demucs (local, GPU) / MDX-Net (local)",
                    )
                    whisper_model = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large"],
                        value="base",
                        label="Whisper Model",
                        info="Only used when YouTube captions are unavailable",
                    )
                    speed_max = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.35,
                        step=0.05,
                        label="Max TTS Speed",
                        info="Above this threshold the video is slowed down",
                    )

                with gr.Row(elem_classes=["action-buttons"]):
                    dub_button = gr.Button("Start Dubbing", variant="primary", size="lg", scale=3)
                    reset_btn = gr.Button("New Video", variant="secondary", size="lg", scale=1)
                continue_btn = gr.Button(
                    "Continue Dubbing", variant="primary", size="lg", visible=False,
                )

            # ── Right panel: output ──
            with gr.Column(scale=3, min_width=480):
                video_metadata = gr.HTML(value="", visible=True)
                status_text = gr.Markdown(
                    "Paste a YouTube URL and click **Start Dubbing** to begin.",
                    elem_classes=["status-box"],
                )

                with gr.Tabs(elem_classes=["output-tabs"]):
                    with gr.TabItem("Video"):
                        output_video = gr.Video(label="Dubbed Video")
                        download_btn = gr.DownloadButton(
                            "Download Dubbed Video", visible=False, variant="secondary",
                        )
                    with gr.TabItem("Transcript"):
                        transcript_table = gr.Dataframe(
                            headers=["Time", "Original (EN)", "Armenian (HY)"],
                            label="Edit the Armenian column, then click Continue Dubbing",
                            interactive=True,
                            column_widths=["90px", "1fr", "1fr"],
                            elem_classes=["transcript-table"],
                        )
                    with gr.TabItem("Logs"):
                        log_output = gr.Textbox(
                            label="Pipeline Log",
                            lines=18,
                            max_lines=30,
                            interactive=False,
                        )

        # ── Wiring ──
        phase1_outputs = [
            state, transcript_table, status_text, output_video,
            download_btn, dub_button, continue_btn, log_output, video_metadata,
        ]

        dub_button.click(
            fn=run_phase1,
            inputs=[
                url_input, translation_provider, whisper_model, speed_max,
                keep_background, skip_review, tts_voice, vocal_sep, state,
            ],
            outputs=phase1_outputs,
            show_progress="full",
        )

        continue_btn.click(
            fn=run_phase2,
            inputs=[state, transcript_table],
            outputs=phase1_outputs,
            show_progress="full",
        )

        reset_btn.click(
            fn=reset_ui,
            outputs=phase1_outputs,
        )

    return app


if __name__ == "__main__":
    if not check_ffmpeg():
        print("WARNING: ffmpeg is not installed. The app will not work without it.")
        print("Install ffmpeg: https://ffmpeg.org/download.html")

    app = build_ui()
    app.launch()
