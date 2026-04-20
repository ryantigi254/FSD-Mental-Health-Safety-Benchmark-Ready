"""Factory for creating model runners."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .base import GenerationConfig, ModelRunner

logger = logging.getLogger(__name__)

# External models root: weights live here (downloaded via Uni-setup), not in
# runtime/models/.
EXTERNAL_MODELS_ROOT = Path(
    r"E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup\models"
)


def _resolve_local_model_path(model_name: str) -> str:
    """Resolve a local model directory, preferring EXTERNAL_MODELS_ROOT."""
    external = EXTERNAL_MODELS_ROOT / model_name
    if external.is_dir():
        return str(external)
    # Fallback to relative path (original behaviour)
    return f"models/{model_name}"


def get_model_runner(
    model_id: str, config: Optional[GenerationConfig] = None, quantization: Optional[str] = None
) -> ModelRunner:
    """
    Get a model runner instance by model ID.

    Args:
        model_id: Model identifier ('psyllm', 'qwq', 'deepseek_r1', 'gpt_oss', 'qwen3')
        config: Optional generation configuration
    """
    model_id_lower = model_id.lower()

    if model_id_lower == "psyllm":
        from .psyllm import PsyLLMRunner

        return PsyLLMRunner(config=config)

    if model_id_lower in ("psyllm_local", "psyllm-8b-local", "psyllm-local-hf"):
        from .psyllm_local import PsyLLMLocalRunner

        return PsyLLMLocalRunner(config=config)

    if model_id_lower in (
        "psyllm_gml_local",
        "psyllm-gml-local",
        "psyllm-gmlhuhe-local",
        "gmlhuhe_psyllm_local",
    ):
        from .psyllm_gml_local import PsyLLMGMLLocalRunner

        return PsyLLMGMLLocalRunner(model_name=_resolve_local_model_path("PsyLLM"), config=config)

    if model_id_lower in ("qwq", "qwq-32b", "qwq_lmstudio", "qwq-lmstudio", "qwq-32b-lmstudio"):
        from .lmstudio_qwq import QwQLMStudioRunner

        return QwQLMStudioRunner(config=config)

    if model_id_lower in ("deepseek_r1", "deepseek-r1-32b"):
        from .deepseek_r1 import DeepSeekR1Runner

        return DeepSeekR1Runner(config=config)

    if model_id_lower in ("deepseek_r1_lmstudio", "deepseek-r1-lmstudio", "deepseek-r1-14b-lmstudio"):
        from .lmstudio_deepseek_r1 import DeepSeekR1LMStudioRunner

        return DeepSeekR1LMStudioRunner(config=config)

    if model_id_lower in ("gpt_oss", "gpt_oss_lmstudio", "gpt-oss-lmstudio", "gpt-oss-20b"):
        from .lmstudio_gpt_oss import GPTOSSLMStudioRunner

        return GPTOSSLMStudioRunner(config=config)

    if model_id_lower in (
        "glm47_flash",
        "glm-4.7-flash",
        "glm-4.7-flash-runpod",
        "glm47_flash_runpod",
    ):
        from .glm47_flash import GLM47FlashRunner

        return GLM47FlashRunner(config=config)

    if model_id_lower in (
        "gpt_oss_remote",
        "gpt_oss_120b_remote",
        "gpt-oss-120b",
        "gpt-oss-120b-runpod",
        "gpt_oss_120b_runpod",
    ):
        from .gpt_oss import GPTOSSRunner

        return GPTOSSRunner(config=config)

    if model_id_lower in ("qwen3", "qwen3-8b"):
        from .qwen3 import Qwen3Runner

        return Qwen3Runner(config=config)

    if model_id_lower in ("qwen3_lmstudio", "qwen3-lmstudio", "qwen3-8b-lmstudio"):
        from .lmstudio_qwen3 import Qwen3LMStudioRunner

        return Qwen3LMStudioRunner(config=config)

    if model_id_lower in (
        "medgemma_lmstudio",
        "medgemma-lmstudio",
        "medgemma_27b_lmstudio",
        "medgemma-27b-lmstudio",
        "google/medgemma-27b-it",
        "google.medgemma-27b-text-it",
    ):
        from .lmstudio_medgemma import MedGemmaLMStudioRunner

        return MedGemmaLMStudioRunner(config=config)

    if model_id_lower in ("piaget_lmstudio", "piaget-lmstudio", "piaget-8b-lmstudio"):
        from .lmstudio_piaget import PiagetLMStudioRunner

        return PiagetLMStudioRunner(config=config)

    if model_id_lower in (
        "psych_qwen_32b-mlx",
        "psych-qwen-32b-mlx",
    ):
        from .lmstudio_psych_qwen import PsychQwen32BLMStudioRunner

        return PsychQwen32BLMStudioRunner(config=config)

    if model_id_lower in (
        "mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2",
        "qwen3.5-distilled",
        "qwen3.5-27b-distilled",
        "qwen3_5_distilled_lmstudio",
        "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0",
    ):
        from .lmstudio_qwen3_5_distilled import Qwen35DistilledLMStudioRunner

        if model_id_lower == "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0":
            return Qwen35DistilledLMStudioRunner(
                model_name="qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0",
                config=config,
            )
        return Qwen35DistilledLMStudioRunner(config=config)

    if model_id_lower in ("piaget", "piaget-8b"):
        from .piaget import Piaget8BRunner
        return Piaget8BRunner(config=config)

    if model_id_lower in ("piaget_local", "piaget-8b-local", "piaget8b-local"):
        from .piaget_local import Piaget8BLocalRunner

        return Piaget8BLocalRunner(model_name=_resolve_local_model_path("Piaget-8B"), config=config)

    if model_id_lower in ("psyche_r1", "psyche-r1"):
        from .psyche_r1 import PsycheR1Runner

        return PsycheR1Runner(config=config)

    if model_id_lower in ("psyche_r1_lmstudio", "psyche-r1-lmstudio"):
        from .lmstudio_psyche_r1 import PsycheR1LMStudioRunner

        return PsycheR1LMStudioRunner(config=config)

    if model_id_lower in ("psyche_r1_local", "psyche-r1-local", "psyche-r1-local-hf"):
        from .psyche_r1_local import PsycheR1LocalRunner

        return PsycheR1LocalRunner(model_name=_resolve_local_model_path("Psyche-R1"), config=config)

    if model_id_lower in ("psych_qwen",):
        from .psych_qwen import PsychQwen32BRunner

        return PsychQwen32BRunner(config=config)

    if model_id_lower in ("psych_qwen_32b", "psych-qwen-32b", "psych_qwen_32b_lmstudio", "psych-qwen-32b-lmstudio"):
        from .lmstudio_psych_qwen import PsychQwen32BLMStudioRunner

        return PsychQwen32BLMStudioRunner(config=config)

    if model_id_lower in ("psych_qwen_local", "psych-qwen-32b-local", "psych-qwen-local-hf"):
        from .psych_qwen_local import PsychQwen32BLocalRunner

        return PsychQwen32BLocalRunner(
            model_name=_resolve_local_model_path("Psych_Qwen_32B"),
            config=config,
        )

    if model_id_lower in (
        "ollama_minimax_m2_5_cloud",
        "minimax_m2_5_cloud",
        "minimax-m2.5-cloud",
        "minimax-m2.5:cloud",
    ):
        try:
            from .ollama_cloud import OllamaCloudRunner
        except ImportError as exc:  # pragma: no cover - optional dependency in this checkout
            raise ValueError("OllamaCloudRunner is unavailable in this checkout.") from exc
        return OllamaCloudRunner(model_name="minimax-m2.5:cloud", config=config)

    if model_id_lower in ("psyllm_gml_vllm", "psyllm-gml-vllm", "psyllm-vllm"):
        from .vllm_runner import VLLMRunner

        return VLLMRunner(model_name="GMLHUHE/PsyLLM-8B", port=8101, config=config)

    if model_id_lower in ("piaget_vllm", "piaget-vllm", "piaget-8b-vllm"):
        from .vllm_runner import VLLMRunner

        return VLLMRunner(model_name="gustavecortal/Piaget-8B", port=8102, config=config)

    if model_id_lower in ("psyche_r1_vllm", "psyche-r1-vllm"):
        from .vllm_runner import VLLMRunner

        return VLLMRunner(model_name="MindIntLab/Psyche-R1", port=8103, config=config)

    if model_id_lower in ("psych_qwen_vllm", "psych-qwen-vllm", "psych-qwen-32b-vllm"):
        from .vllm_runner import VLLMRunner

        return VLLMRunner(model_name="Compumacy/Psych_Qwen_32B", port=8104, config=config)

    raise ValueError(
        f"Unknown model ID: {model_id}. "
        "Supported models: psyllm, qwq, deepseek_r1, gpt_oss, qwen3, "
        "medgemma_lmstudio, ollama_minimax_m2_5_cloud, piaget, psyche_r1, psyche_r1_lmstudio, psych_qwen, "
        "psych_qwen_32b-mlx, qwen3.5-distilled"
    )
