import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (pay-as-you-go)
PRICING = {
    # Claude
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    # Gemini translation
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    # Gemini TTS
    "gemini-2.5-flash-preview-tts": {"input": 0.50, "output": 10.00},
}


@dataclass
class CostTracker:
    translation_input_tokens: int = 0
    translation_output_tokens: int = 0
    translation_model: str = ""
    tts_input_tokens: int = 0
    tts_output_tokens: int = 0
    tts_model: str = ""
    tts_calls: int = 0

    def add_translation_usage(self, model: str, input_tokens: int, output_tokens: int):
        self.translation_model = model
        self.translation_input_tokens += input_tokens
        self.translation_output_tokens += output_tokens

    def add_tts_usage(self, model: str, input_tokens: int, output_tokens: int):
        self.tts_model = model
        self.tts_input_tokens += input_tokens
        self.tts_output_tokens += output_tokens
        self.tts_calls += 1

    def _cost_for(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = PRICING.get(model)
        if not pricing:
            logger.warning(f"No pricing data for model: {model}")
            return 0.0
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    @property
    def translation_cost(self) -> float:
        return self._cost_for(
            self.translation_model,
            self.translation_input_tokens,
            self.translation_output_tokens,
        )

    @property
    def tts_cost(self) -> float:
        return self._cost_for(
            self.tts_model,
            self.tts_input_tokens,
            self.tts_output_tokens,
        )

    @property
    def total_cost(self) -> float:
        return self.translation_cost + self.tts_cost

    def summary(self) -> str:
        lines = [
            "=== API Cost Summary ===",
            f"Translation ({self.translation_model}):",
            f"  Input:  {self.translation_input_tokens:,} tokens (${self._cost_for(self.translation_model, self.translation_input_tokens, 0):.4f})",
            f"  Output: {self.translation_output_tokens:,} tokens (${self._cost_for(self.translation_model, 0, self.translation_output_tokens):.4f})",
            f"  Subtotal: ${self.translation_cost:.4f}",
            f"TTS ({self.tts_model}, {self.tts_calls} calls):",
            f"  Input:  {self.tts_input_tokens:,} tokens (${self._cost_for(self.tts_model, self.tts_input_tokens, 0):.4f})",
            f"  Output: {self.tts_output_tokens:,} tokens (${self._cost_for(self.tts_model, 0, self.tts_output_tokens):.4f})",
            f"  Subtotal: ${self.tts_cost:.4f}",
            f"TOTAL: ${self.total_cost:.4f}",
        ]
        return "\n".join(lines)
