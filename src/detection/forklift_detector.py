"""Forklift overload detector for counting blocks on forklift forks.

Class 3 — Carrying Overload with Forklift: Uses YOLO to detect vehicles
and the Groq vision API to count standardized blocks on the forks.
A count of 3 or more is a violation. Below the confidence threshold,
the event is flagged for human review.
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

try:
	from ultralytics import YOLO
except ImportError:  # pragma: no cover
	YOLO = None

try:
	from groq import Groq
except ImportError:  # pragma: no cover
	Groq = None

from .video_utils import iter_video_frames


OVERLOAD_THRESHOLD = 3
CONFIDENCE_THRESHOLD = 0.5

# YOLO COCO class IDs for vehicles that could be forklifts
VEHICLE_CLASS_IDS = {2, 5, 7}  # car, bus, truck — forklift often detected as truck


@dataclass(frozen=True)
class ForkliftDetection:
	frame_index: int
	is_violation: bool
	block_count: int
	confidence: float
	needs_review: bool
	vehicle_box: tuple[int, int, int, int] | None
	description: str

	def to_dict(self) -> dict:
		return asdict(self)


@lru_cache(maxsize=2)
def _load_model(model_name: str):
	if YOLO is None:
		raise RuntimeError("ultralytics is not installed in the current Python environment")
	return YOLO(model_name)


def _get_groq_client() -> "Groq":
	if Groq is None:
		raise RuntimeError("groq is not installed in the current Python environment")
	api_key = os.getenv("GROQ_API_KEY", "")
	if not api_key:
		raise RuntimeError("Set GROQ_API_KEY before calling Groq vision API")
	return Groq(api_key=api_key)


def _encode_frame_base64(frame: np.ndarray, *, max_side: int = 512) -> str:
	"""Resize and encode a frame as a base64 JPEG string for the Groq API."""
	h, w = frame.shape[:2]
	scale = min(max_side / max(h, w), 1.0)
	if scale < 1.0:
		frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
	_, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
	return base64.b64encode(buffer).decode("utf-8")


def detect_vehicles(
	frame: np.ndarray,
	*,
	model_name: str = "yolov8n.pt",
	confidence_threshold: float = 0.25,
) -> list[tuple[int, int, int, int]]:
	"""Detect vehicles (potential forklifts) in a frame using YOLO."""
	model = _load_model(model_name)
	results = model.predict(frame, conf=confidence_threshold, verbose=False)
	vehicle_boxes: list[tuple[int, int, int, int]] = []

	for result in results:
		boxes = getattr(result, "boxes", None)
		if boxes is None or boxes.xyxy is None:
			continue

		xyxy_values = boxes.xyxy.cpu().numpy()
		class_values = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy_values))

		for xyxy, class_id in zip(xyxy_values, class_values):
			if int(class_id) in VEHICLE_CLASS_IDS:
				x1, y1, x2, y2 = (int(v) for v in xyxy)
				vehicle_boxes.append((x1, y1, x2, y2))

	return vehicle_boxes


def count_blocks_with_groq(
	frame: np.ndarray,
	vehicle_box: tuple[int, int, int, int] | None = None,
	*,
	model: str = "llama-3.2-90b-vision-preview",
) -> tuple[int, float, str]:
	"""Use Groq vision to count standardized blocks on a forklift.

	Returns (block_count, confidence, description).
	"""
	# Crop to vehicle region if available, otherwise use full frame
	if vehicle_box is not None:
		x1, y1, x2, y2 = vehicle_box
		h, w = frame.shape[:2]
		# Add padding around the vehicle
		pad = 30
		x1 = max(0, x1 - pad)
		y1 = max(0, y1 - pad)
		x2 = min(w, x2 + pad)
		y2 = min(h, y2 + pad)
		crop = frame[y1:y2, x1:x2]
	else:
		crop = frame

	client = _get_groq_client()
	image_b64 = _encode_frame_base64(crop)

	response = client.chat.completions.create(
		model=model,
		messages=[
			{
				"role": "user",
				"content": [
					{
						"type": "text",
						"text": (
							"Look at this image from a factory. Is there a forklift visible? "
							"If so, count how many standardized blocks (rectangular cargo blocks) "
							"are stacked on the forklift's forks.\n\n"
							"Respond with ONLY one of these exact formats:\n"
							"- BLOCKS: <number> - <brief description>\n"
							"- NO_FORKLIFT: no forklift visible\n"
							"- UNCLEAR: cannot determine block count\n"
						),
					},
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{image_b64}",
						},
					},
				],
			}
		],
		temperature=0.1,
		max_tokens=150,
	)

	content = (response.choices[0].message.content or "").strip()
	content_upper = content.upper()

	# Parse block count from response
	if "BLOCKS:" in content_upper:
		# Try to extract the number
		match = re.search(r"BLOCKS:\s*(\d+)", content_upper)
		if match:
			count = int(match.group(1))
			return count, 0.75, content
		return 0, 0.3, content
	elif "NO_FORKLIFT" in content_upper:
		return 0, 0.7, content
	else:
		return 0, 0.2, content


def detect_forklift_violations_for_frame(
	frame: np.ndarray,
	*,
	frame_index: int = 0,
	model_name: str = "yolov8n.pt",
	vision_model: str = "llama-3.2-90b-vision-preview",
	confidence_threshold: float = 0.25,
	overload_threshold: int = OVERLOAD_THRESHOLD,
	review_confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> list[ForkliftDetection]:
	"""Detect forklift overload violations in a single frame."""
	detections: list[ForkliftDetection] = []

	# First try to detect vehicles with YOLO
	vehicle_boxes = detect_vehicles(frame, model_name=model_name, confidence_threshold=confidence_threshold)

	if vehicle_boxes:
		# Analyze each detected vehicle
		for vehicle_box in vehicle_boxes:
			try:
				block_count, confidence, description = count_blocks_with_groq(
					frame, vehicle_box, model=vision_model
				)
				is_violation = block_count >= overload_threshold
				needs_review = confidence < review_confidence_threshold

				detections.append(
					ForkliftDetection(
						frame_index=frame_index,
						is_violation=is_violation,
						block_count=block_count,
						confidence=confidence,
						needs_review=needs_review,
						vehicle_box=vehicle_box,
						description=description,
					)
				)
			except Exception as exc:
				detections.append(
					ForkliftDetection(
						frame_index=frame_index,
						is_violation=False,
						block_count=0,
						confidence=0.0,
						needs_review=True,
						vehicle_box=vehicle_box,
						description=f"Detection failed: {exc}",
					)
				)
	else:
		# No vehicles detected by YOLO — try full-frame analysis with Groq
		try:
			block_count, confidence, description = count_blocks_with_groq(frame, model=vision_model)
			if block_count > 0:
				is_violation = block_count >= overload_threshold
				needs_review = confidence < review_confidence_threshold
				detections.append(
					ForkliftDetection(
						frame_index=frame_index,
						is_violation=is_violation,
						block_count=block_count,
						confidence=confidence,
						needs_review=needs_review,
						vehicle_box=None,
						description=description,
					)
				)
		except Exception:
			pass  # No forklift detected at all — not an error

	return detections


def detect_forklift_violations_in_video(
	video_path: str | Path,
	*,
	stride: int = 60,
	max_frames: int | None = 5,
	model_name: str = "yolov8n.pt",
	vision_model: str = "llama-3.2-90b-vision-preview",
	confidence_threshold: float = 0.25,
) -> list[ForkliftDetection]:
	"""Run forklift overload detection over sampled frames from a video.

	Uses a larger stride and fewer frames since each frame may require
	a Groq API call for block counting.
	"""
	all_detections: list[ForkliftDetection] = []
	for frame_index, frame in iter_video_frames(video_path, stride=stride, max_frames=max_frames):
		frame_detections = detect_forklift_violations_for_frame(
			frame,
			frame_index=frame_index,
			model_name=model_name,
			vision_model=vision_model,
			confidence_threshold=confidence_threshold,
		)
		all_detections.extend(frame_detections)
	return all_detections
