"""
Local LM Studio runner for PsyLLM-8B GGUF served by LM Studio.
"""

from .base import GenerationConfig
from .lmstudio_qwen3 import Qwen3LMStudioRunner


class PsyLLMLMStudioRunner(Qwen3LMStudioRunner):
    """LM Studio-backed PsyLLM runner using the loaded `psyllm-8b` API identifier."""

    def __init__(
        self,
        model_name: str = "psyllm-8b",
        api_base: str = "http://127.0.0.1:1234/v1",
        config: GenerationConfig = None,
    ):
        super().__init__(model_name=model_name, api_base=api_base, config=config)
