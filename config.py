import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # --- API Keys (loaded from .env) ---
    anthropic_api_key: str = ""
    google_api_key: str = ""
    lalal_api_key: str = ""

    # --- Translation ---
    translation_provider: str = "gemini"    # Options: "gemini", "claude"
    claude_model: str = "claude-sonnet-4-20250514"
    gemini_model: str = "gemini-2.5-pro"
    translation_batch_size: int = 15        # Number of segments per LLM call

    # --- TTS ---
    gemini_tts_model: str = "gemini-2.5-flash-preview-tts"
    # Gemini TTS voices (all support Armenian):
    #   Zephyr, Puck, Charon, Kore, Fenrir, Leda, Orus, Aoede,
    #   Callirrhoe, Autonoe, Enceladus, Iapetus, Umbriel, Algieba,
    #   Despina, Erinome, Algenib, Rasalgethi, Laomedeia, Achernar,
    #   Alnilam, Schedar, Gacrux, Pulcherrima, Achird,
    #   Zubenelgenubi, Vindemiatrix, Sadachbia, Sadaltager, Sulafat
    tts_voice_name: str = "Charon"
    tts_speaking_rate: float = 1.0

    # --- Whisper (fallback when no YouTube captions) ---
    # Options: "tiny", "base", "small", "medium", "large"
    # Larger = more accurate but slower and more VRAM
    whisper_model_size: str = "base"

    # --- Audio sync ---
    speed_min: float = 0.75     # Min TTS speedup (below this, pad with silence)
    speed_max: float = 1.35     # Max TTS speedup (above this, slow down video)
    fade_ms: int = 50           # Crossfade duration in ms to prevent clicks
    keep_background_music: bool = True  # Preserve background music via vocal separation
    vocal_separator: str = "lalal"     # Options: "demucs", "mdx", "lalal" (LALAL.AI API)
    background_volume_db: float = -3.0  # Volume adjustment for background music (dB)

    # --- Transcript segmentation ---
    segment_min_duration: float = 5.0   # Merge short sentences until this minimum
    segment_max_duration: float = 30.0  # Split sentences longer than this

    # --- Processing ---
    temp_dir: str = ""
    cache_dir: str = "cache"            # Root cache directory (per-video caching)

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            lalal_api_key=os.getenv("LALAL_API_KEY", ""),
        )

    def validate(self) -> list[str]:
        errors = []
        if self.translation_provider == "claude" and not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required when using Claude for translation")
        if self.translation_provider == "gemini" and not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required when using Gemini for translation")
        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required for Gemini TTS")
        if self.vocal_separator == "lalal" and not self.lalal_api_key:
            errors.append("LALAL_API_KEY is required when using LALAL.AI for vocal separation")
        return errors
