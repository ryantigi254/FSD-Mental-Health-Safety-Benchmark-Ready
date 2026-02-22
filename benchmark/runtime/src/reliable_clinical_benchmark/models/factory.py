"""Factory for creating model runners."""



import logging

from pathlib import Path

from typing import Optional



from .base import ModelRunner, GenerationConfig

from .psyllm import PsyLLMRunner

from .psyllm_local import PsyLLMLocalRunner

from .psyllm_gml_local import PsyLLMGMLLocalRunner

from .deepseek_r1 import DeepSeekR1Runner

from .lmstudio_deepseek_r1 import DeepSeekR1LMStudioRunner

from .qwen3 import Qwen3Runner

from .gpt_oss import GPTOSSRunner

from .piaget import Piaget8BRunner

from .piaget_local import Piaget8BLocalRunner

from .psyche_r1 import PsycheR1Runner

from .psyche_r1_local import PsycheR1LocalRunner

from .psych_qwen import PsychQwen32BRunner

from .psych_qwen_local import PsychQwen32BLocalRunner

from .lmstudio_qwq import QwQLMStudioRunner

from .lmstudio_gpt_oss import GPTOSSLMStudioRunner

from .lmstudio_qwen3 import Qwen3LMStudioRunner



logger = logging.getLogger(__name__)



# External models root: weights live here (downloaded via Uni-setup), not in runtime/models/.

EXTERNAL_MODELS_ROOT = Path(r"E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup\models")





def _resolve_local_model_path(model_name: str) -> str:

    """Resolve a local model directory, preferring EXTERNAL_MODELS_ROOT."""

    external = EXTERNAL_MODELS_ROOT / model_name

    if external.is_dir():

        return str(external)

    # Fallback to relative path (original behaviour)

    return f"models/{model_name}"





def get_model_runner(

    model_id: str, config: Optional[GenerationConfig] = None

) -> ModelRunner:

    """

    Get a model runner instance by model ID.



    Args:

        model_id: Model identifier ('psyllm', 'qwq', 'deepseek_r1', 'gpt_oss', 'qwen3')

        config: Optional generation configuration



    Returns:

        ModelRunner instance



    Raises:

        ValueError: If model_id is not recognised

    """

    model_id_lower = model_id.lower()



    if model_id_lower == "psyllm":

        return PsyLLMRunner(config=config)

    elif model_id_lower in ("psyllm_local", "psyllm-8b-local", "psyllm-local-hf"):

        return PsyLLMLocalRunner(config=config)

    elif model_id_lower in ("psyllm_gml_local", "psyllm-gml-local", "psyllm-gmlhuhe-local", "gmlhuhe_psyllm_local"):

        return PsyLLMGMLLocalRunner(model_name=_resolve_local_model_path("PsyLLM"), config=config)

    elif model_id_lower in ("qwq", "qwq-32b", "qwq_lmstudio", "qwq-lmstudio", "qwq-32b-lmstudio"):

        return QwQLMStudioRunner(config=config)

    elif model_id_lower in ("deepseek_r1", "deepseek-r1-32b"):

        return DeepSeekR1Runner(config=config)

    elif model_id_lower in ("deepseek_r1_lmstudio", "deepseek-r1-lmstudio", "deepseek-r1-14b-lmstudio"):

        return DeepSeekR1LMStudioRunner(config=config)

    elif model_id_lower in ("gpt_oss", "gpt_oss_lmstudio", "gpt-oss-lmstudio", "gpt-oss-20b"):

        # Default GPT-OSS runs are expected to use LM Studio (local OpenAI-compatible server).

        return GPTOSSLMStudioRunner(config=config)

    elif model_id_lower in ("gpt_oss_remote", "gpt_oss_120b_remote", "gpt-oss-120b"):

        # Placeholder remote runner (api.example.com) retained for completeness.

        return GPTOSSRunner(config=config)

    elif model_id_lower in ("qwen3", "qwen3-8b"):

        return Qwen3Runner(config=config)

    elif model_id_lower in ("qwen3_lmstudio", "qwen3-lmstudio", "qwen3-8b-lmstudio"):

        return Qwen3LMStudioRunner(config=config)

    elif model_id_lower in ("piaget", "piaget-8b"):

        return Piaget8BRunner(config=config)

    elif model_id_lower in ("piaget_local", "piaget-8b-local", "piaget8b-local"):

        return Piaget8BLocalRunner(model_name=_resolve_local_model_path("Piaget-8B"), config=config)

    elif model_id_lower in ("psyche_r1", "psyche-r1"):

        return PsycheR1Runner(config=config)

    elif model_id_lower in ("psyche_r1_local", "psyche-r1-local", "psyche-r1-local-hf"):

        return PsycheR1LocalRunner(model_name=_resolve_local_model_path("Psyche-R1"), config=config)

    elif model_id_lower in ("psych_qwen", "psych_qwen_32b", "psych-qwen-32b"):

        return PsychQwen32BRunner(config=config)

    elif model_id_lower in ("psych_qwen_local", "psych-qwen-32b-local", "psych-qwen-local-hf"):

        return PsychQwen32BLocalRunner(model_name=_resolve_local_model_path("Psych_Qwen_32B"), config=config)

    else:

        raise ValueError(

            f"Unknown model ID: {model_id}. "

            f"Supported models: psyllm, qwq, deepseek_r1, gpt_oss, qwen3, "

            f"piaget, psyche_r1, psych_qwen"

        )




"""Factory for creating model runners."""

import logging
from pathlib import Path
from typing import Optional

from .base import ModelRunner, GenerationConfig
from .psyllm import PsyLLMRunner
from .psyllm_local import PsyLLMLocalRunner
from .psyllm_gml_local import PsyLLMGMLLocalRunner
from .deepseek_r1 import DeepSeekR1Runner
from .lmstudio_deepseek_r1 import DeepSeekR1LMStudioRunner
from .qwen3 import Qwen3Runner
from .gpt_oss import GPTOSSRunner
from .piaget import Piaget8BRunner
from .piaget_local import Piaget8BLocalRunner
from .psyche_r1 import PsycheR1Runner
from .psyche_r1_local import PsycheR1LocalRunner
from .psych_qwen import PsychQwen32BRunner
from .psych_qwen_local import PsychQwen32BLocalRunner
from .lmstudio_qwq import QwQLMStudioRunner
from .lmstudio_gpt_oss import GPTOSSLMStudioRunner
from .lmstudio_qwen3 import Qwen3LMStudioRunner
from .vllm_runner import VLLMRunner

logger = logging.getLogger(__name__)

# External models root: weights live here (downloaded via Uni-setup), not in runtime/models/.
EXTERNAL_MODELS_ROOT = Path(r"E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup\models")


def _resolve_local_model_path(model_name: str) -> str:
    """Resolve a local model directory, preferring EXTERNAL_MODELS_ROOT."""
    external = EXTERNAL_MODELS_ROOT / model_name
    if external.is_dir():
        return str(external)
    # Fallback to relative path (original behaviour)
    return f"models/{model_name}"


def get_model_runner(
    model_id: str, config: Optional[GenerationConfig] = None
) -> ModelRunner:
    """
    Get a model runner instance by model ID.

    Args:
        model_id: Model identifier ('psyllm', 'qwq', 'deepseek_r1', 'gpt_oss', 'qwen3')
        config: Optional generation configuration

    Returns:
        ModelRunner instance

    Raises:
        ValueError: If model_id is not recognised
    """
    model_id_lower = model_id.lower()

    if model_id_lower == "psyllm":
        return PsyLLMRunner(config=config)
    elif model_id_lower in ("psyllm_local", "psyllm-8b-local", "psyllm-local-hf"):
        return PsyLLMLocalRunner(config=config)
    elif model_id_lower in ("psyllm_gml_local", "psyllm-gml-local", "psyllm-gmlhuhe-local", "gmlhuhe_psyllm_local"):
        return PsyLLMGMLLocalRunner(model_name=_resolve_local_model_path("PsyLLM"), config=config)
    elif model_id_lower in ("qwq", "qwq-32b", "qwq_lmstudio", "qwq-lmstudio", "qwq-32b-lmstudio"):
        return QwQLMStudioRunner(config=config)
    elif model_id_lower in ("deepseek_r1", "deepseek-r1-32b"):
        return DeepSeekR1Runner(config=config)
    elif model_id_lower in ("deepseek_r1_lmstudio", "deepseek-r1-lmstudio", "deepseek-r1-14b-lmstudio"):
        return DeepSeekR1LMStudioRunner(config=config)
    elif model_id_lower in ("gpt_oss", "gpt_oss_lmstudio", "gpt-oss-lmstudio", "gpt-oss-20b"):
        # Default GPT-OSS runs are expected to use LM Studio (local OpenAI-compatible server).
        return GPTOSSLMStudioRunner(config=config)
    elif model_id_lower in ("gpt_oss_remote", "gpt_oss_120b_remote", "gpt-oss-120b"):
        # Placeholder remote runner (api.example.com) retained for completeness.
        return GPTOSSRunner(config=config)
    elif model_id_lower in ("qwen3", "qwen3-8b"):
        return Qwen3Runner(config=config)
    elif model_id_lower in ("qwen3_lmstudio", "qwen3-lmstudio", "qwen3-8b-lmstudio"):
        return Qwen3LMStudioRunner(config=config)
    elif model_id_lower in ("piaget", "piaget-8b"):
        return Piaget8BRunner(config=config)
    elif model_id_lower in ("piaget_local", "piaget-8b-local", "piaget8b-local"):
        return Piaget8BLocalRunner(model_name=_resolve_local_model_path("Piaget-8B"), config=config)
    elif model_id_lower in ("psyche_r1", "psyche-r1"):
        return PsycheR1Runner(config=config)
    elif model_id_lower in ("psyche_r1_local", "psyche-r1-local", "psyche-r1-local-hf"):
        return PsycheR1LocalRunner(model_name=_resolve_local_model_path("Psyche-R1"), config=config)
    elif model_id_lower in ("psych_qwen", "psych_qwen_32b", "psych-qwen-32b"):
        return PsychQwen32BRunner(config=config)
    elif model_id_lower in ("psych_qwen_local", "psych-qwen-32b-local", "psych-qwen-local-hf"):
        return PsychQwen32BLocalRunner(model_name=_resolve_local_model_path("Psych_Qwen_32B"), config=config)
    elif model_id_lower in ("psyllm_gml_vllm", "psyllm-gml-vllm", "psyllm-vllm"):
        return VLLMRunner(model_name="GMLHUHE/PsyLLM-8B", port=8101, config=config)
    elif model_id_lower in ("piaget_vllm", "piaget-vllm", "piaget-8b-vllm"):
        return VLLMRunner(model_name="gustavecortal/Piaget-8B", port=8102, config=config)
    elif model_id_lower in ("psyche_r1_vllm", "psyche-r1-vllm"):
        return VLLMRunner(model_name="MindIntLab/Psyche-R1", port=8103, config=config)
    elif model_id_lower in ("psych_qwen_vllm", "psych-qwen-vllm", "psych-qwen-32b-vllm"):
        return VLLMRunner(model_name="Compumacy/Psych_Qwen_32B", port=8104, config=config)
    else:
        raise ValueError(
            f"Unknown model ID: {model_id}. "
            f"Supported models: psyllm, qwq, deepseek_r1, gpt_oss, qwen3, "
            f"piaget, psyche_r1, psych_qwen"
        )

