#!/usr/bin/env python3
"""Deprecated wrapper for clean_generation_outputs.py."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    target = Path(__file__).with_name("clean_generation_outputs.py")
    print("[DEPRECATED] Use scripts/preprocessing/clean_generation_outputs.py")
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
