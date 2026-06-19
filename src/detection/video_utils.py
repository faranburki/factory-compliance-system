"""Video loading helpers for the detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
	fps: float
	frame_count: int
	width: int
	height: int


def load_video_metadata(video_path: str | Path) -> VideoMetadata:
	path = Path(video_path)
	cap = cv2.VideoCapture(str(path))
	try:
		if not cap.isOpened():
			raise FileNotFoundError(f"Unable to open video: {path}")

		return VideoMetadata(
			fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
			frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
			width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
			height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
		)
	finally:
		cap.release()


def iter_video_frames(video_path: str | Path, *, stride: int = 1, max_frames: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
	"""Yield sampled frames from a video."""

	if stride < 1:
		raise ValueError("stride must be at least 1")

	cap = cv2.VideoCapture(str(Path(video_path)))
	try:
		if not cap.isOpened():
			raise FileNotFoundError(f"Unable to open video: {video_path}")

		frame_index = 0
		sampled = 0
		while True:
			ok, frame = cap.read()
			if not ok:
				break

			if frame_index % stride == 0:
				yield frame_index, frame
				sampled += 1
				if max_frames is not None and sampled >= max_frames:
					break

			frame_index += 1
	finally:
		cap.release()


def load_first_frame(video_path: str | Path) -> tuple[int, np.ndarray]:
	"""Return the first frame from a video."""

	for frame_index, frame in iter_video_frames(video_path, stride=1, max_frames=1):
		return frame_index, frame
	raise ValueError(f"No frames found in video: {video_path}")
