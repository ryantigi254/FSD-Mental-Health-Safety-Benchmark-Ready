"""
GPT-OSS-120B model runner — RunPod Serverless vLLM endpoint.

Requires two env vars:
    RUNPOD_GPT_OSS_120B_ENDPOINT  – full URL to the RunPod vLLM /v1/chat/completions
    GPT_OSS_API_KEY               – RunPod API key (used as Bearer token)

See docs/models/RUNPOD_GPT_OSS_120B.md for setup instructions.
"""

import os
import logging

from .remote_api import RemoteAPIRunner, GenerationConfig

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = os.getenv(
    "RUNPOD_GPT_OSS_120B_ENDPOINT",
    "https://api.runpod.ai/v2/<endpoint-id>/openai/v1/chat/completions",
)


class GPTOSSRunner(RemoteAPIRunner):
    """GPT-OSS-120B inference via RunPod Serverless vLLM."""

    def __init__(self, config: GenerationConfig = None):
        endpoint = _DEFAULT_ENDPOINT

        if "<endpoint-id>" in endpoint:
            logger.warning(
                "RUNPOD_GPT_OSS_120B_ENDPOINT is not configured. "
                "Set it in .env to the RunPod serverless endpoint URL. "
                "See docs/models/RUNPOD_GPT_OSS_120B.md."
            )

        super().__init__(
            model_name="gpt-oss-120b",
            api_endpoint=endpoint,
            api_key_env="GPT_OSS_API_KEY",
            config=config or GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                max_tokens=None,
            ),
        )
