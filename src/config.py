"""Shared configuration loaded from .env and sensible defaults."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# ── Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DB_PATH = OUTPUTS_DIR / "reports.db"
POLICY_PDF = PROJECT_ROOT / "Compliance_Policy_Manual.pdf"
POLICY_RULES_JSON = OUTPUTS_DIR / "policy_rules.json"

# ── Load .env ───────────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ── Detection thresholds ────────────────────────────────────────────────
PERSON_CONFIDENCE_THRESHOLD: float = 0.25
YOLO_MODEL: str = "yolov8m.pt"
FRAME_STRIDE: int = 30
MAX_FRAMES: int | None = None

# Camera-specific calibration values — these cannot be derived from
# policy text as they represent physical measurements of this facility's
# camera angles and lighting conditions. The policy defines WHAT to
# detect (vest color, walkway boundary) but not the pixel-level
# parameters required to detect it in this specific camera feed.
# All behavior classes, severity tiers, and numeric thresholds that
# ARE stated in policy text are loaded dynamically from
# outputs/policy_rules.json at runtime.

# Walkway polygon ratios (fraction of frame dimensions)
WALKWAY_TOP_Y_RATIO: float = 0.55
WALKWAY_LEFT_X_RATIO: float = 0.12
WALKWAY_RIGHT_X_RATIO: float = 0.88
WALKWAY_OUTER_LEFT_RATIO: float = 0.02
WALKWAY_OUTER_RIGHT_RATIO: float = 0.98

# Vest detector HSV ranges — two overlapping green ranges to handle
# factory lighting variation (shadows, direct light, yellow-green vests)
GREEN_VEST_HSV_LOW_1 = (25, 30, 30)
GREEN_VEST_HSV_HIGH_1 = (70, 255, 255)
GREEN_VEST_HSV_LOW_2 = (70, 20, 40)
GREEN_VEST_HSV_HIGH_2 = (95, 255, 255)

RED_VEST_HSV_LOW_1 = (0, 50, 50)
RED_VEST_HSV_HIGH_1 = (10, 255, 255)
RED_VEST_HSV_LOW_2 = (170, 50, 50)
RED_VEST_HSV_HIGH_2 = (180, 255, 255)

# Green pixel ratio threshold for vest classification
GREEN_VEST_RATIO_THRESHOLD: float = 0.06

# Debug mode: saves torso crops and green masks to outputs/vest_debug/
VEST_DEBUG_MODE: bool = False

# Forklift overload — threshold is loaded dynamically from
# outputs/policy_rules.json (parsed from the Compliance Policy Manual).
# No hardcoded default; if the policy JSON is missing the value the
# detector is skipped with a clear warning.

# Groq vision model for panel/forklift detection
GROQ_VISION_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_TEXT_MODEL: str = "llama-3.1-8b-instant"

# ── Facility zone label (default for single-camera setup) ───────────────
DEFAULT_ZONE: str = "Production Floor"
