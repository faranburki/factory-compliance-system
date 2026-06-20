"""
Core Configuration | config.py
Centralized constants and paths for the compliance system.
"""

from __future__ import annotations

from pathlib import Path

# Resolve absolute paths based on the location of this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input data paths
DATA_DIR = PROJECT_ROOT / "data"
POLICIES_DIR = DATA_DIR / "policies"
VIDEOS_DIR = DATA_DIR / "videos"

# Output data paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DB_PATH = OUTPUTS_DIR / "reports.db"
CSV_EXPORT_PATH = OUTPUTS_DIR / "compliance_report.csv"
POLICY_RULES_JSON = OUTPUTS_DIR / "policy_rules.json"

# Models
YOLO_MODEL_NAME = "yolov8m.pt"
GROQ_TEXT_MODEL = "llama-3.1-8b-instant"
GROQ_VISION_MODEL = "llama-3.2-90b-vision-preview"

# Default behavior thresholds
CONFIDENCE_THRESHOLD = 0.40
WALKWAY_STRIDE = 30
VISION_STRIDE = 60
