"""Model runner package exports."""

from __future__ import annotations

from .base import GenerationConfig, ModelRunner

__all__ = ["ModelRunner", "GenerationConfig", "get_model_runner"]

from .base import ModelRunner, GenerationConfig
from .factory import get_model_runner
from .psyllm import PsyLLMRunner
from .deepseek_r1 import DeepSeekR1Runner
from .lmstudio_deepseek_r1 import DeepSeekR1LMStudioRunner
from .qwen3 import Qwen3Runner
from .gpt_oss import GPTOSSRunner
from .lmstudio_qwq import QwQLMStudioRunner
from .lmstudio_gpt_oss import GPTOSSLMStudioRunner
from .lmstudio_qwen3 import Qwen3LMStudioRunner
from .lmstudio_medgemma import MedGemmaLMStudioRunner
from .piaget_local import Piaget8BLocalRunner
from .psyche_r1_local import PsycheR1LocalRunner
from .psych_qwen_local import PsychQwen32BLocalRunner
from .psyllm_local import PsyLLMLocalRunner
from .psyllm_gml_local import PsyLLMGMLLocalRunner
try:
    from .ollama_cloud import OllamaCloudRunner
except ImportError:  # pragma: no cover - optional dependency in this checkout
    OllamaCloudRunner = None
from .vllm_runner import VLLMRunner

__all__ = [
    "ModelRunner",
    "GenerationConfig",
    "get_model_runner",
    "PsyLLMRunner",
    "QwQLMStudioRunner",
    "DeepSeekR1Runner",
    "DeepSeekR1LMStudioRunner",
    "Qwen3Runner",
    "Qwen3LMStudioRunner",
    "MedGemmaLMStudioRunner",
    "GPTOSSRunner",
    "GPTOSSLMStudioRunner",
    "Piaget8BLocalRunner",
    "PsycheR1LocalRunner",
    "PsychQwen32BLocalRunner",
    "PsyLLMLocalRunner",
    "PsyLLMGMLLocalRunner",
    "OllamaCloudRunner",
    "VLLMRunner",
]
