#!/usr/bin/env python3
"""Backward-compatible entrypoint: defaults --eval-name to stage2_m2_test. Prefer checkpoint_test_eval.py."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

if "--eval-name" not in sys.argv:
    sys.argv[1:1] = ["--eval-name", "stage2_m2_test"]

from q1_3stage_pipeline.evaluation.checkpoint_test_eval import main

if __name__ == "__main__":
    main()
