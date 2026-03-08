"""Evaluation pipelines for Studies A, B, C, and controllability."""

from __future__ import annotations

from importlib import import_module

__all__ = ["study_a", "study_b", "study_c", "controllability"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module
