"""
Module 1 — Detection Engine | vest_detector.py
Vest color detector for identifying unauthorized equipment intervention.
"""

from __future__ import annotations

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
		"""
		Convert detection object to a standard dictionary format.
		"""
		return asdict(self)


# Policy Section 4.2 — green vest is the sole observable indicator
# of authorized intervention status
GREEN_HSV_LOW = np.array([35, 40, 40])
GREEN_HSV_HIGH = np.array([85, 255, 255])

# HSV ranges for red vest detection (red wraps around hue 0/180)
RED_HSV_LOW_1 = np.array([0, 50, 50])
RED_HSV_HIGH_1 = np.array([10, 255, 255])
RED_HSV_LOW_2 = np.array([170, 50, 50])
RED_HSV_HIGH_2 = np.array([180, 255, 255])

# Black detection (low saturation, low value)
BLACK_HSV_LOW = np.array([0, 0, 0])
BLACK_HSV_HIGH = np.array([180, 255, 50])


def _extract_torso_region(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
	"""
	Crop the upper-middle portion of a person bounding box (torso area).

	Args:
		frame: The full image frame.
		box: A tuple of (x1, y1, x2, y2) defining the person.

	Returns:
		The cropped torso image as a numpy array, or None if invalid dimensions.
	"""
	x1, y1, x2, y2 = box
	height = y2 - y1
	width = x2 - x1

	if height < 10 or width < 5:
		return None

	# Torso is roughly the upper 30-60% of the bounding box
	torso_top = y1 + int(height * 0.15)
	torso_bottom = y1 + int(height * 0.55)
	torso_left = x1 + int(width * 0.1)
	torso_right = x2 - int(width * 0.1)

	# Clamp to frame boundaries to prevent OpenCV slicing errors
	h, w = frame.shape[:2]
	torso_top = max(0, torso_top)
	torso_bottom = min(h, torso_bottom)
	torso_left = max(0, torso_left)
	torso_right = min(w, torso_right)

	if torso_bottom <= torso_top or torso_right <= torso_left:
		return None

	return frame[torso_top:torso_bottom, torso_left:torso_right]


def classify_vest_color(torso: np.ndarray) -> tuple[Literal["green", "red-black", "unknown"], float, float]:
	"""
	Classify the dominant vest color from a torso crop using HSV masking.

	Args:
		torso: Cropped torso image array.

	Returns:
		A tuple containing (color_label, green_ratio, red_black_ratio).
	"""
	hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
	total_pixels = hsv.shape[0] * hsv.shape[1]

	if total_pixels == 0:
		return "unknown", 0.0, 0.0

	# Green mask
	green_mask = cv2.inRange(hsv, GREEN_HSV_LOW, GREEN_HSV_HIGH)
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

	# Decision: if green is dominant, it's authorized
	if green_ratio > 0.15 and green_ratio > red_black_ratio * 0.5:
		return "green", green_ratio, red_black_ratio
	elif red_black_ratio > 0.20:
		return "red-black", green_ratio, red_black_ratio
	else:
		return "unknown", green_ratio, red_black_ratio


def classify_vest_violations(
	frame: np.ndarray,
	person_detections: list[PersonDetection],
	walkway_polygon: list[tuple[int, int]],
	*,
	frame_index: int = 0,
) -> list[VestDetection]:
	"""
	Classify vest color for each detected person in a frame.

	Args:
		frame: The video frame.
		person_detections: A list of detected persons.
		walkway_polygon: The safe walkway boundary polygon.
		frame_index: Current frame index.

	Returns:
		A list of VestDetection results.
	"""
	results: list[VestDetection] = []

	for detection in person_detections:
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

		vest_color, green_ratio, red_ratio = classify_vest_color(torso)
		
		# A violation occurs if they are wearing a red-black vest AND they are near equipment.
		# Equipment is on the left side of the factory floor (x < width * 0.55).
		# People just stepping slightly outside the walkway (e.g. x > width * 0.55) 
		# should only get a Walkway Deviation (MEDIUM), not an Unauthorized Intervention (CRITICAL).
		h, w = frame.shape[:2]
		is_near_equipment = detection.foot_point[0] < (w * 0.55)
		is_violation = (vest_color == "red-black") and is_near_equipment

		# Confidence is based on how dominant the detected color is
		if vest_color == "green":
			color_confidence = min(green_ratio * 3, 1.0)
		elif vest_color == "red-black":
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
	model_name: str = "yolov8n.pt",
	confidence_threshold: float = 0.25,
	frame_index: int = 0,
) -> list[VestDetection]:
	"""
	Detect vest color violations in a single frame.

	Args:
		frame: The video frame.
		model_name: The YOLO model.
		confidence_threshold: Cutoff for valid YOLO person detections.
		frame_index: Current frame index.

	Returns:
		A list of VestDetection instances.
	"""
	person_detections = detect_people(frame, model_name=model_name, confidence_threshold=confidence_threshold)
	
	h, w = frame.shape[:2]
	walkway_poly = default_walkway_polygon(w, h)
	
	return classify_vest_violations(frame, person_detections, walkway_poly, frame_index=frame_index)


def detect_vest_violations_in_video(
	video_path: str | Path,
	*,
	stride: int = 30,
	max_frames: int | None = None,
	model_name: str = "yolov8n.pt",
	confidence_threshold: float = 0.25,
) -> list[VestDetection]:
	"""
	Run vest color detection over sampled frames from a video.

	Args:
		video_path: Path to the video file.
		stride: Step interval for sampling frames.
		max_frames: Optional cap on the number of frames evaluated.
		model_name: YOLO model file name.
		confidence_threshold: Cutoff for valid YOLO person detections.

	Returns:
		An aggregated list of VestDetection objects.
	"""
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
