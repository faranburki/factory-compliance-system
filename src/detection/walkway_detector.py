"""
Module 1 — Detection Engine | walkway_detector.py
Detects Safe Walkway Violations by checking whether a detected person's
feet position falls outside the green-marked walkway boundary polygon.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

try:
	from ultralytics import YOLO
except ImportError:  # pragma: no cover - depends on installed environment
	YOLO = None

from .video_utils import iter_video_frames


@dataclass(frozen=True)
class PersonDetection:
	box: tuple[int, int, int, int]
	confidence: float
	foot_point: tuple[int, int]


@dataclass(frozen=True)
class WalkwayDetection:
	frame_index: int
	is_violation: bool
	confidence: float
	person_box: tuple[int, int, int, int]
	foot_point: tuple[int, int]
	polygon: list[tuple[int, int]]

	def to_dict(self) -> dict:
		"""
		Convert detection object to a standard dictionary format.
		"""
		return asdict(self)


def default_walkway_polygon(width: int, height: int) -> list[tuple[int, int]]:
	"""
	Build a default walkway polygon mapping to the green painted lines on the right wall.

	Args:
		width: Video frame width.
		height: Video frame height.

	Returns:
		A list of 4 (x,y) coordinates defining the safe walkway boundary.
	"""
	# The designated safe walkway is bounded by the green line on the far right.
	# The left boundary of this polygon represents the green line itself.
	top_left_x = int(width * 0.85)
	top_right_x = width
	top_y = int(height * 0.25)
	
	bottom_left_x = int(width * 0.90)
	bottom_right_x = width
	bottom_y = height

	return [
		(top_left_x, top_y),       # Top-left (near door)
		(top_right_x, top_y),      # Top-right (near door)
		(bottom_right_x, bottom_y),# Bottom-right
		(bottom_left_x, bottom_y), # Bottom-left
	]


def person_foot_point(box: tuple[int, int, int, int]) -> tuple[int, int]:
	"""
	Calculate the lowest center point (the feet) of a bounding box.

	Args:
		box: A tuple of (x1, y1, x2, y2).

	Returns:
		The calculated (x, y) coordinates of the feet.
	"""
	x1, y1, x2, y2 = box
	# Average the x coordinates for horizontal center, use the bottom y coordinate
	return int((x1 + x2) / 2), int(y2)


def point_in_polygon(point: tuple[int, int], polygon: list[tuple[int, int]]) -> bool:
	"""
	Determine if a specific point lies inside a defined polygon.

	Args:
		point: The (x, y) coordinate to check.
		polygon: A list of (x, y) coordinates defining the boundary.

	Returns:
		True if the point is inside or on the boundary, False otherwise.
	"""
	polygon_array = np.array(polygon, dtype=np.int32)
	# cv2.pointPolygonTest returns >= 0 if the point is inside or on the edge
	return cv2.pointPolygonTest(polygon_array, point, False) >= 0


def classify_walkway_violations(
	person_detections: list[PersonDetection],
	polygon: list[tuple[int, int]],
	*,
	frame_index: int = 0,
) -> list[WalkwayDetection]:
	"""
	Classify person detections as violations if their feet are outside the polygon.

	Args:
		person_detections: A list of detected persons.
		polygon: The safe walkway polygon.
		frame_index: The current frame index.

	Returns:
		A list of WalkwayDetection records denoting violation status.
	"""
	results: list[WalkwayDetection] = []
	for detection in person_detections:
		is_inside = point_in_polygon(detection.foot_point, polygon)
		results.append(
			WalkwayDetection(
				frame_index=frame_index,
				# Violation occurs strictly when the person's feet fall outside the polygon
				is_violation=not is_inside,
				confidence=detection.confidence,
				person_box=detection.box,
				foot_point=detection.foot_point,
				polygon=polygon,
			)
		)
	return results


@lru_cache(maxsize=2)
def _load_model(model_name: str):
	"""
	Load and cache the YOLO model to prevent redundant initializations.

	Args:
		model_name: The name or path of the YOLO model file.

	Raises:
		RuntimeError: If ultralytics is not installed.
	"""
	if YOLO is None:
		raise RuntimeError("ultralytics is not installed in the current Python environment")
	return YOLO(model_name)


def detect_people(frame: np.ndarray, *, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.25) -> list[PersonDetection]:
	"""
	Run YOLO person detection on a single frame.

	Args:
		frame: The image frame to process.
		model_name: The YOLO model file to load. Defaults to 'yolov8n.pt'.
		confidence_threshold: Minimum confidence score to register a detection.

	Returns:
		A list of extracted PersonDetection objects representing detected people.
	"""
	model = _load_model(model_name)
	results = model.predict(frame, conf=confidence_threshold, verbose=False)
	detections: list[PersonDetection] = []

	for result in results:
		boxes = getattr(result, "boxes", None)
		if boxes is None or boxes.xyxy is None:
			continue

		xyxy_values = boxes.xyxy.cpu().numpy()
		conf_values = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy_values), dtype=float)
		class_values = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy_values), dtype=float)

		for xyxy, detection_confidence, class_id in zip(xyxy_values, conf_values, class_values):
			# Class 0 corresponds to 'person' in the standard COCO dataset
			if int(class_id) != 0:
				continue
			x1, y1, x2, y2 = (int(value) for value in xyxy)
			detections.append(
				PersonDetection(
					box=(x1, y1, x2, y2),
					confidence=float(detection_confidence),
					foot_point=person_foot_point((x1, y1, x2, y2)),
				)
			)

	return detections


def detect_walkway_violations_for_frame(
	frame: np.ndarray,
	polygon: list[tuple[int, int]],
	*,
	model_name: str = "yolov8n.pt",
	confidence_threshold: float = 0.25,
	frame_index: int = 0,
) -> list[WalkwayDetection]:
	"""
	Detect people and classify walkway violations within a single frame.

	Args:
		frame: The video frame to process.
		polygon: The safe walkway boundary polygon.
		model_name: The YOLO model. Defaults to 'yolov8n.pt'.
		confidence_threshold: The minimum confidence for YOLO detections.
		frame_index: The index of the frame being processed.

	Returns:
		A list of classified WalkwayDetection instances.
	"""
	person_detections = detect_people(frame, model_name=model_name, confidence_threshold=confidence_threshold)
	return classify_walkway_violations(person_detections, polygon, frame_index=frame_index)


def detect_walkway_violations_in_video(
	video_path: str | Path,
	polygon: list[tuple[int, int]],
	*,
	stride: int = 30,
	max_frames: int | None = None,
	model_name: str = "yolov8n.pt",
	confidence_threshold: float = 0.25,
) -> list[WalkwayDetection]:
	"""
	Run walkway violation detection sequentially over sampled frames from a video.

	Args:
		video_path: Path to the video file.
		polygon: The predefined safe walkway boundary polygon.
		stride: Step interval for sampling frames. Defaults to 30.
		max_frames: Optional cap on the number of frames evaluated.
		model_name: YOLO model file name. Defaults to 'yolov8n.pt'.
		confidence_threshold: Cutoff for valid YOLO person detections.

	Returns:
		An aggregated list of WalkwayDetection objects.
	"""
	violations: list[WalkwayDetection] = []
	for frame_index, frame in iter_video_frames(video_path, stride=stride, max_frames=max_frames):
		violations.extend(
			detect_walkway_violations_for_frame(
				frame,
				polygon,
				model_name=model_name,
				confidence_threshold=confidence_threshold,
				frame_index=frame_index,
			)
		)
	return violations
