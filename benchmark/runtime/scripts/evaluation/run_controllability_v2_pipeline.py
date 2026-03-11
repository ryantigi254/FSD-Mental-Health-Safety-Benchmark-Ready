#!/usr/bin/env python3
"""Compatibility shim for the old controllability v2 CLI path."""

from __future__ import annotations

import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from run_controllability_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
