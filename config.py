import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # --- API Keys (loaded from .env) ---
    anthropic_api_key: str = ""
    google_api_key: str = ""

    # --- Translation ---
    translation_provider: str = "gemini"  # "gemini" or "claude"
    claude_model: str = "claude-sonnet-4-20250514"
    gemini_model: str = "gemini-2.5-pro"
    translation_batch_size: int = 15

    # --- TTS ---
    tts_provider: str = "gemini"  # "gemini" or "edge_tts"
    gemini_tts_model: str = "gemini-2.5-flash-preview-tts"
    tts_voice_name: str = "Kore"
    tts_speaking_rate: float = 1.0

    # --- Whisper (fallback transcription) ---
    whisper_model_size: str = "base"

    # --- Audio sync ---
    speed_min: float = 0.75
    speed_max: float = 1.35
    fade_ms: int = 50

    # --- Transcript segmentation ---
    segment_min_duration: float = 5.0
    segment_max_duration: float = 30.0

    # --- Processing ---
    temp_dir: str = ""

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        )

    def validate(self) -> list[str]:
        errors = []
        if self.translation_provider == "claude" and not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required when using Claude for translation")
        if self.translation_provider == "gemini" and not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required when using Gemini for translation")
        if self.tts_provider == "gemini" and not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required when using Gemini TTS")
        return errors
