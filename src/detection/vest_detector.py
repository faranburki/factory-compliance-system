"""Vest color detector for identifying unauthorized equipment intervention.

Class 1 — Unauthorized Intervention: A person interacting with equipment
while wearing a red-black vest (or any non-green vest) is a violation.

Authorized Intervention: A person wearing a green safety vest is authorized
to interact with equipment. This is a safe, compliant behavior that must NOT
generate any violation report, database entry, or alert. (Section 4.2 / 4.3)

Uses OpenCV HSV color masking on the torso region of detected persons.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .video_utils import iter_video_frames
from .walkway_detector import PersonDetection, detect_people, point_in_polygon, default_walkway_polygon


@dataclass(frozen=True)
class VestDetection:
	frame_index: int
	is_violation: bool
	vest_color: Literal["green", "red-black", "unknown"]
	confidence: float
	person_box: tuple[int, int, int, int]
	green_ratio: float
	red_ratio: float

	def to_dict(self) -> dict:
		return asdict(self)


# ── HSV ranges ──────────────────────────────────────────────────────────
# Two overlapping green ranges combined with bitwise OR to handle factory
# lighting variation — shadows, direct light, and yellow-green vest
# material shifts.
#   Range 1: core green through yellow-green  (H 25-70)
#   Range 2: teal / dark-green overflow       (H 70-95)
# Lower S and V bounds catch shadowed / underexposed vests.
GREEN_HSV_LOW_1 = np.array([25, 30, 30])
GREEN_HSV_HIGH_1 = np.array([70, 255, 255])
GREEN_HSV_LOW_2 = np.array([70, 20, 40])
GREEN_HSV_HIGH_2 = np.array([95, 255, 255])

# HSV ranges for red vest detection (red wraps around hue 0/180)
RED_HSV_LOW_1 = np.array([0, 50, 50])
RED_HSV_HIGH_1 = np.array([10, 255, 255])
RED_HSV_LOW_2 = np.array([170, 50, 50])
RED_HSV_HIGH_2 = np.array([180, 255, 255])

# Black detection (low saturation, low value)
BLACK_HSV_LOW = np.array([0, 0, 0])
BLACK_HSV_HIGH = np.array([180, 255, 50])

# Green pixel ratio threshold — kept low to catch partially occluded or
# shadowed vests without false-positiving on other colors.
GREEN_RATIO_THRESHOLD = 0.06

# ── Debug configuration ────────────────────────────────────────────────
# Controlled by config.VEST_DEBUG_MODE.  When True, saves the torso crop
# and the green color mask to outputs/vest_debug/ so the HSV range can
# be visually verified.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEBUG_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "vest_debug"


def _is_debug_mode() -> bool:
	"""Check debug flag from config (import deferred to avoid circular deps)."""
	try:
		from ..config import VEST_DEBUG_MODE  # type: ignore[import]
		return VEST_DEBUG_MODE
	except Exception:
		pass
	# Fallback: check env var
	return os.getenv("VEST_DEBUG_MODE", "0").lower() in ("1", "true", "yes")


def _save_debug_images(
	torso: np.ndarray,
	green_mask: np.ndarray,
	frame_index: int,
	person_idx: int,
) -> None:
	"""Save torso crop and green mask to the debug output directory."""
	DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	prefix = f"frame{frame_index:06d}_person{person_idx}"
	cv2.imwrite(str(DEBUG_OUTPUT_DIR / f"{prefix}_torso.png"), torso)
	cv2.imwrite(str(DEBUG_OUTPUT_DIR / f"{prefix}_green_mask.png"), green_mask)


def _extract_torso_region(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
	"""Crop the center torso region of a person bounding box.

	The crop targets the vertical band from 20% to 55% of the bounding box
	height (below the head, above the waist) and the horizontal center 60%
	(avoids arms/background on edges).  This reliably lands on the vest
	area for persons at varying distances from the camera.
	"""
	x1, y1, x2, y2 = box
	height = y2 - y1
	width = x2 - x1

	if height < 10 or width < 5:
		return None

	# Vertical: 20-55% of bbox height  (vest / upper torso region)
	torso_top = y1 + int(height * 0.20)
	torso_bottom = y1 + int(height * 0.55)
	# Horizontal: center 60% of bbox width  (avoids arms and background)
	torso_left = x1 + int(width * 0.20)
	torso_right = x2 - int(width * 0.20)

	# Clamp to frame boundaries
	h, w = frame.shape[:2]
	torso_top = max(0, torso_top)
	torso_bottom = min(h, torso_bottom)
	torso_left = max(0, torso_left)
	torso_right = min(w, torso_right)

	if torso_bottom <= torso_top or torso_right <= torso_left:
		return None

	return frame[torso_top:torso_bottom, torso_left:torso_right]


def classify_vest_color(
	torso: np.ndarray,
) -> tuple[Literal["green", "red-black", "unknown"], float, float, np.ndarray | None]:
	"""Classify the dominant vest color from a torso crop using HSV masking.

	Returns (color_label, green_ratio, red_black_ratio, green_mask).
	The green_mask is returned for debug saving.
	"""
	hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
	total_pixels = hsv.shape[0] * hsv.shape[1]

	if total_pixels == 0:
		return "unknown", 0.0, 0.0, None

	# Green mask — two overlapping ranges OR'd together
	green_mask_1 = cv2.inRange(hsv, GREEN_HSV_LOW_1, GREEN_HSV_HIGH_1)
	green_mask_2 = cv2.inRange(hsv, GREEN_HSV_LOW_2, GREEN_HSV_HIGH_2)
	green_mask = cv2.bitwise_or(green_mask_1, green_mask_2)
	green_pixels = int(cv2.countNonZero(green_mask))

	# Red mask (two ranges since red wraps around hue boundary)
	red_mask_1 = cv2.inRange(hsv, RED_HSV_LOW_1, RED_HSV_HIGH_1)
	red_mask_2 = cv2.inRange(hsv, RED_HSV_LOW_2, RED_HSV_HIGH_2)
	red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

	# Black mask
	black_mask = cv2.inRange(hsv, BLACK_HSV_LOW, BLACK_HSV_HIGH)

	# Combined red-black
	red_black_mask = cv2.bitwise_or(red_mask, black_mask)
	red_black_pixels = int(cv2.countNonZero(red_black_mask))

	green_ratio = green_pixels / total_pixels
	red_black_ratio = red_black_pixels / total_pixels

	# ── Decision logic ──────────────────────────────────────────────
	# Green vest is the *sole observable indicator* of authorization
	# (Section 4.2).  Use a low threshold (0.06) so partially occluded
	# or shadowed vests are still detected.  The green_ratio > 0.06
	# is enough because the dual HSV range is already selective for
	# actual green; false positives on other colors are rare.
	#
	# If green is present above threshold, classify as authorized
	# regardless of red-black ratio — a person with a visible green
	# vest is always authorized.
	if green_ratio >= GREEN_RATIO_THRESHOLD:
		return "green", green_ratio, red_black_ratio, green_mask

	# Red-black classification: any combination of red and/or black
	# pixels above threshold.  Per Section 4.3.2, a "red-black vest
	# or any vest other than the designated green authorization vest"
	# is unauthorized — so pure black vests must also be caught.
	if red_black_ratio > 0.20:
		return "red-black", green_ratio, red_black_ratio, green_mask
	else:
		return "unknown", green_ratio, red_black_ratio, green_mask


def classify_vest_violations(
	frame: np.ndarray,
	person_detections: list[PersonDetection],
	walkway_polygon: list[tuple[int, int]],
	*,
	frame_index: int = 0,
) -> list[VestDetection]:
	"""Classify vest color for each detected person in a frame.

	Authorized persons (green vest) are returned with is_violation=False
	and vest_color='green'.  They must NOT be passed downstream to the
	report / escalation pipeline.

	Unauthorized persons (red-black vest near equipment) are returned
	with is_violation=True.
	"""
	debug = _is_debug_mode()
	results: list[VestDetection] = []

	for person_idx, detection in enumerate(person_detections):
		torso = _extract_torso_region(frame, detection.box)
		if torso is None:
			results.append(
				VestDetection(
					frame_index=frame_index,
					is_violation=False,
					vest_color="unknown",
					confidence=detection.confidence * 0.3,
					person_box=detection.box,
					green_ratio=0.0,
					red_ratio=0.0,
				)
			)
			continue

		vest_color, green_ratio, red_ratio, green_mask = classify_vest_color(torso)

		# ── Debug output ────────────────────────────────────────────
		if debug and green_mask is not None:
			_save_debug_images(torso, green_mask, frame_index, person_idx)

		# ────────────────────────────────────────────────────────────
		# AUTHORIZED INTERVENTION GUARD (Section 4.2 / 4.3)
		#
		# If the person is wearing a green vest, they are AUTHORIZED.
		# This is a hard stop — no violation, no report, no alert.
		# The detection is still returned so the annotator can draw
		# a green box with "Authorized Intervention", but is_violation
		# is False and vest_color is "green", which the pipeline must
		# check and skip.
		# ────────────────────────────────────────────────────────────
		if vest_color == "green":
			color_confidence = min(green_ratio * 3, 1.0)
			results.append(
				VestDetection(
					frame_index=frame_index,
					is_violation=False,
					vest_color="green",
					confidence=detection.confidence * color_confidence,
					person_box=detection.box,
					green_ratio=green_ratio,
					red_ratio=red_ratio,
				)
			)
			continue   # ← hard stop — skip all violation logic

		# ── Violation logic (non-green vest) ────────────────────────
		# Per Section 4.3.2: "a red-black vest or any vest other than
		# the designated green authorization vest" near equipment is
		# Unauthorized Intervention.  So ANY non-green vest (including
		# pure black, unknown, etc.) near equipment is a violation.
		# Equipment is on the left side of the factory floor.
		h, w = frame.shape[:2]
		is_near_equipment = detection.foot_point[0] < (w * 0.55)
		is_violation = (vest_color != "green") and is_near_equipment

		# Confidence is based on how dominant the detected color is
		if vest_color == "red-black":
			color_confidence = min(red_ratio * 2, 1.0)
		else:
			color_confidence = 0.3

		results.append(
			VestDetection(
				frame_index=frame_index,
				is_violation=is_violation,
				vest_color=vest_color,
				confidence=detection.confidence * color_confidence,
				person_box=detection.box,
				green_ratio=green_ratio,
				red_ratio=red_ratio,
			)
		)

	return results


def detect_vest_violations_for_frame(
	frame: np.ndarray,
	*,
	model_name: str = "yolov8m.pt",
	confidence_threshold: float = 0.25,
	frame_index: int = 0,
) -> list[VestDetection]:
	"""Detect vest color violations in a single frame."""
	person_detections = detect_people(frame, model_name=model_name, confidence_threshold=confidence_threshold)
	
	h, w = frame.shape[:2]
	walkway_poly = default_walkway_polygon(w, h)
	
	return classify_vest_violations(frame, person_detections, walkway_poly, frame_index=frame_index)


def detect_vest_violations_in_video(
	video_path: str | Path,
	*,
	stride: int = 30,
	max_frames: int | None = None,
	model_name: str = "yolov8m.pt",
	confidence_threshold: float = 0.25,
) -> list[VestDetection]:
	"""Run vest color detection over sampled frames from a video."""
	violations: list[VestDetection] = []
	for frame_index, frame in iter_video_frames(video_path, stride=stride, max_frames=max_frames):
		violations.extend(
			detect_vest_violations_for_frame(
				frame,
				model_name=model_name,
				confidence_threshold=confidence_threshold,
				frame_index=frame_index,
			)
		)
	return violations
