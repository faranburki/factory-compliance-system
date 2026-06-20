"""
Module 1 — Detection Engine | panel_detector.py
Uses the Groq vision API to detect whether an electrical panel cover is open.
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

try:
	from groq import Groq
except ImportError:  # pragma: no cover
	Groq = None

from .video_utils import iter_video_frames


@dataclass(frozen=True)
class PanelDetection:
	frame_index: int
	is_violation: bool
	panel_state: str  # "open", "closed", or "unknown"
	confidence: float
	description: str

	def to_dict(self) -> dict:
		"""
		Convert detection object to a standard dictionary format.
		"""
		return asdict(self)


def _get_groq_client() -> "Groq":
	"""
	Retrieve the Groq API key and initialize the client.

	Returns:
		An initialized Groq client instance.

	Raises:
		RuntimeError: If GROQ_API_KEY is not set or groq is not installed.
	"""
	if Groq is None:
		raise RuntimeError("groq is not installed in the current Python environment")
	api_key = os.getenv("GROQ_API_KEY", "")
	if not api_key:
		raise RuntimeError("Set GROQ_API_KEY before calling Groq vision API")
	return Groq(api_key=api_key)


def _encode_frame_base64(frame: np.ndarray, *, max_side: int = 512) -> str:
	"""
	Resize and encode a frame as a base64 JPEG string for the Groq API.

	Args:
		frame: The image array to encode.
		max_side: Maximum dimension for the scaled image.

	Returns:
		Base64 encoded string of the JPEG image.
	"""
	h, w = frame.shape[:2]
	scale = min(max_side / max(h, w), 1.0)
	if scale < 1.0:
		frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
	_, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
	return base64.b64encode(buffer).decode("utf-8")


def classify_panel_state(frame: np.ndarray, *, model: str = "llama-3.2-90b-vision-preview") -> PanelDetection:
	"""
	Use Groq vision to classify whether a panel cover is open or closed.

	Args:
		frame: The video frame.
		model: The language/vision model to use.

	Returns:
		A PanelDetection specifying the determined panel state.
	"""
	client = _get_groq_client()
	image_b64 = _encode_frame_base64(frame)

	response = client.chat.completions.create(
		model=model,
		messages=[
			{
				"role": "user",
				"content": [
					{
						"type": "text",
						"text": (
							"Look at this factory/industrial image. Is there an electrical panel "
							"cover visible? If so, is it OPEN or CLOSED?\n\n"
							"Respond with ONLY one of these exact formats:\n"
							"- PANEL_OPEN: <brief description>\n"
							"- PANEL_CLOSED: <brief description>\n"
							"- NO_PANEL: no electrical panel visible\n"
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

	# Parse response
	content_upper = content.upper()
	if "PANEL_OPEN" in content_upper:
		return PanelDetection(
			frame_index=0,
			is_violation=True,
			panel_state="open",
			confidence=0.8,
			description=content,
		)
	elif "PANEL_CLOSED" in content_upper:
		return PanelDetection(
			frame_index=0,
			is_violation=False,
			panel_state="closed",
			confidence=0.8,
			description=content,
		)
	else:
		return PanelDetection(
			frame_index=0,
			is_violation=False,
			panel_state="unknown",
			confidence=0.3,
			description=content,
		)


def detect_panel_violations_for_frame(
	frame: np.ndarray,
	*,
	frame_index: int = 0,
	model: str = "llama-3.2-90b-vision-preview",
) -> PanelDetection:
	"""
	Check a single frame for open panel covers using Groq vision.

	Args:
		frame: The video frame to process.
		frame_index: The index of the frame being processed.
		model: The Groq vision model to use.

	Returns:
		A PanelDetection specifying the violation status for this frame.
	"""
	try:
		detection = classify_panel_state(frame, model=model)
		# Override frame_index since classify_panel_state defaults to 0
		return PanelDetection(
			frame_index=frame_index,
			is_violation=detection.is_violation,
			panel_state=detection.panel_state,
			confidence=detection.confidence,
			description=detection.description,
		)
	except Exception as exc:
		return PanelDetection(
			frame_index=frame_index,
			is_violation=False,
			panel_state="unknown",
			confidence=0.0,
			description=f"Detection failed: {exc}",
		)


def detect_panel_violations_in_video(
	video_path: str | Path,
	*,
	stride: int = 60,
	max_frames: int | None = 5,
	model: str = "llama-3.2-90b-vision-preview",
) -> list[PanelDetection]:
	"""
	Run panel cover detection over sampled frames from a video.

	Uses a larger stride and fewer frames than other detectors since each
	frame requires a Groq API call, and panel state is persistent (state-based).

	Args:
		video_path: Path to the video file.
		stride: Step interval for sampling frames.
		max_frames: Optional cap on the number of frames evaluated.
		model: The Groq vision model to use.

	Returns:
		An aggregated list of PanelDetection objects.
	"""
	detections: list[PanelDetection] = []
	for frame_index, frame in iter_video_frames(video_path, stride=stride, max_frames=max_frames):
		detection = detect_panel_violations_for_frame(frame, frame_index=frame_index, model=model)
		detections.append(detection)
	return detections
