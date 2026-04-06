"""
Local LM Studio runner for Psych Qwen 32B.

Defaults to the LM Studio API identifier used by the local GGUF loadout.
"""

import os

from .lmstudio_qwen3 import Qwen3LMStudioRunner


class PsychQwen32BLMStudioRunner(Qwen3LMStudioRunner):
    """Local LM Studio inference for Psych Qwen 32B."""

    def __init__(self, api_base: str = "http://127.0.0.1:1234/v1", config=None):
        super().__init__(
            model_name=os.getenv("LMSTUDIO_PSYCH_QWEN_MODEL", "psych_qwen_32b"),
            api_base=api_base,
            config=config,
        )
