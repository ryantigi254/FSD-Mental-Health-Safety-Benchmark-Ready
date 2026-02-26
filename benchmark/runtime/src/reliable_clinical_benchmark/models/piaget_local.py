"""
Local runner for gustavecortal/Piaget-8B using Hugging Face transformers.

Loads the model into local memory (no LM Studio) and runs generation with
chat template + enable_thinking to preserve <think> traces.

Adds a sentinel end marker (<END>) + token-level stopping criteria to prevent
runaway repetition loops in long generations.
"""

import logging
import re
from typing import Tuple, List

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from .base import ModelRunner, GenerationConfig

logger = logging.getLogger(__name__)


class StopOnTokenSequence(StoppingCriteria):
    """Stop generation once the most-recent tokens match a target token sequence."""
    def __init__(self, seq_ids: List[int]):
        super().__init__()
        self.seq_ids = list(seq_ids)

    def __call__(self, input_ids, scores, **kwargs):
        if not self.seq_ids:
            return False
        ids = input_ids[0].tolist()
        n = len(self.seq_ids)
        if len(ids) < n:
            return False
        return ids[-n:] == self.seq_ids


class NgramRepeatStop(StoppingCriteria):
    """Stop when the recent token window is dominated by repeated n-grams.

    This catches ABAB loops / paragraph loops even when EOS or <END> is never produced.
    """

    def __init__(
        self,
        prompt_len: int,
        window: int = 256,
        n: int = 8,
        dup_frac: float = 0.25,
        min_new: int = 256,
    ):
        super().__init__()
        self.prompt_len = int(prompt_len)
        self.window = int(window)
        self.n = int(n)
        self.dup_frac = float(dup_frac)
        self.min_new = int(min_new)

    def __call__(self, input_ids, scores, **kwargs):
        ids = input_ids[0].tolist()
        new = len(ids) - self.prompt_len
        if new < self.min_new:
            return False
        if len(ids) < self.window + self.n:
            return False

        tail = ids[-self.window :]
        # Build n-grams over the tail window
        ngrams = [tuple(tail[i : i + self.n]) for i in range(len(tail) - self.n + 1)]
        if not ngrams:
            return False
        dup = 1.0 - (len(set(ngrams)) / max(1, len(ngrams)))
        return dup >= self.dup_frac


class Piaget8BLocalRunner(ModelRunner):
    """
    Local inference for Piaget-8B via transformers.
    """

    def __init__(
        self,
        model_name: str = "models/Piaget-8B",
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        config: GenerationConfig = None,
        local_files_only: bool = True,
    ):
        super().__init__(
            model_name,
            config
            or GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024,
            ),
        )
        self._model_name = model_name
        self._device_map = device_map
        self._dtype = dtype
        self._local_files_only = local_files_only

        logger.info(f"Loading {model_name} locally (device_map={device_map})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=local_files_only,
        )

        # Precompute sentinel stop sequences as token IDs
        self._end_ids = self.tokenizer("<END>", add_special_tokens=False).input_ids
        self._end_ids_nl = self.tokenizer("\n<END>", add_special_tokens=False).input_ids
        self._end_ids_nl2 = self.tokenizer("\n<END>\n", add_special_tokens=False).input_ids

        model_cfg = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=True,
            config=model_cfg,
            local_files_only=local_files_only,
        )
        self.model.eval()

    def _build_inputs(self, prompt: str, mode: str = "default"):
        formatted_prompt = self._format_prompt(prompt, mode)
        messages = [{"role": "user", "content": formatted_prompt}]
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=(mode == "cot"),
            )
        except TypeError:
            # Some templates ignore enable_thinking
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        tokenized = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=False,
        )
        return {k: v.to(self.model.device) for k, v in tokenized.items()}

    def generate(self, prompt: str, mode: str = "default") -> str:
        encoded = self._build_inputs(prompt, mode)
        input_token_count = int(encoded["input_ids"].shape[-1])

        stops = StoppingCriteriaList(
            [
                StopOnTokenSequence(self._end_ids),
                StopOnTokenSequence(self._end_ids_nl),
                StopOnTokenSequence(self._end_ids_nl2),
                NgramRepeatStop(prompt_len=input_token_count),
            ]
        )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        # Try GPU generation first; if CUDA OOM occurs, retry with reduced max_new_tokens.
        try:
            gen = self.model.generate(
                **encoded,
                max_new_tokens=self.config.max_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=1.05,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_id,
                stopping_criteria=stops,
            )
        except RuntimeError as e:
            error_str = str(e).lower()
            if "cuda" in error_str and ("out of memory" in error_str or "oom" in error_str):
                logger.warning(f"GPU OOM error: {error_str[:200]}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(0)

                # Retry with progressively reduced tokens.
                last_err = e
                for reduced_tokens in [self.config.max_tokens // 2, self.config.max_tokens // 4, 512, 256]:
                    if reduced_tokens <= 0:
                        continue
                    try:
                        logger.info(f"Retrying generation with max_new_tokens={reduced_tokens}")
                        gen = self.model.generate(
                            **encoded,
                            max_new_tokens=reduced_tokens,
                            do_sample=True,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                            repetition_penalty=1.05,
                            eos_token_id=self.tokenizer.eos_token_id,
                            pad_token_id=pad_id,
                            stopping_criteria=stops,
                        )
                        break
                    except RuntimeError as retry_error:
                        last_err = retry_error
                        err2 = str(retry_error).lower()
                        if not ("cuda" in err2 and ("out of memory" in err2 or "oom" in err2)):
                            raise
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats(0)
                else:
                    raise last_err
            else:
                raise

        generated_only = gen[0, input_token_count:]
        output = self.tokenizer.decode(generated_only, skip_special_tokens=True)

        # Remove sentinel from output if present
        output = output.replace("<END>", "").strip()

        # Clear GPU cache after each generation to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(0)
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        return output.strip()

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        full_response = self.generate(prompt, mode="cot")
        think_pattern = r"<(?:redacted_reasoning|think)>(.*?)</(?:redacted_reasoning|think)>"
        think_match = re.search(think_pattern, full_response, re.DOTALL)

        if think_match:
            reasoning = think_match.group(1).strip()
            end_tag_pos = full_response.find("</", think_match.end())
            if end_tag_pos != -1:
                closing_tag_end = full_response.find(">", end_tag_pos)
                if closing_tag_end != -1:
                    answer = full_response[closing_tag_end + 1 :].strip()
                else:
                    answer = full_response
            else:
                answer = full_response
        else:
            parts = re.split(r"\n(?:Diagnosis|Answer|Conclusion):\s*", full_response)
            if len(parts) >= 2:
                reasoning = parts[0].strip()
                answer = parts[1].strip()
            else:
                reasoning = full_response
                answer = full_response

        return answer, reasoning
