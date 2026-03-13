"""Model runner package exports."""

from __future__ import annotations

from .base import GenerationConfig, ModelRunner

__all__ = ["ModelRunner", "GenerationConfig", "get_model_runner"]


def get_model_runner(*args, **kwargs):
    """Lazy wrapper to avoid importing heavy backends during lightweight test collection."""

    from .factory import get_model_runner as _get_model_runner

    return _get_model_runner(*args, **kwargs)
