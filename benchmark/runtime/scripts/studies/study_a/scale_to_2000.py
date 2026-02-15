#!/usr/bin/env python3
"""Deprecated wrapper for scaling/expand_to_2000_samples.py."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    target = Path(__file__).parent / "scaling" / "expand_to_2000_samples.py"
    print("[DEPRECATED] Use scripts/studies/study_a/scaling/expand_to_2000_samples.py")
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
