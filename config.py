import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # API Keys
    anthropic_api_key: str = ""
    google_api_key: str = ""

    # Translation settings
    translation_provider: str = "claude"  # "claude" or "gemini"
    claude_model: str = "claude-sonnet-4-20250514"
    gemini_model: str = "gemini-2.5-pro"

    # TTS settings
    tts_provider: str = "gemini"  # "gemini" or "edge_tts"
    tts_voice_name: str = "Kore"  # Gemini voice name
    tts_speaking_rate: float = 1.0

    # Whisper settings
    whisper_model_size: str = "base"

    # Processing
    temp_dir: str = ""
    speed_min: float = 0.75
    speed_max: float = 1.35

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            translation_provider=os.getenv("TRANSLATION_PROVIDER", "claude"),
            tts_provider=os.getenv("TTS_PROVIDER", "gemini"),
            tts_voice_name=os.getenv("TTS_VOICE_NAME", "Kore"),
            whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "base"),
            speed_max=float(os.getenv("MAX_SPEED_FACTOR", "1.35")),
            speed_min=float(os.getenv("MIN_SPEED_FACTOR", "0.75")),
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
