"""
Retinal AI: Training and Calibration Pipeline Entrypoint (train_calibrate.py)
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening
Primary Specification: MATLAB Retinal AI SIH Presentation Guide.pdf & AI_SPEC.md

This module provides the canonical alias / execution wrapper for the
single combined training and calibration pipeline implemented in
`aiml/scripts/train_and_calibrate.py`.
"""

import os
import sys

# Ensure root is in path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from aiml.scripts.train_and_calibrate import (
    generate_fundus_sample,
    build_cohort,
    compute_comprehensive_metrics,
    main,
)

if __name__ == "__main__":
    main()
