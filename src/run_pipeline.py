"""
Pipeline Controller | run_pipeline.py
End-to-end execution script for the factory compliance system.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import (
	DB_PATH,
	CSV_EXPORT_PATH,
	VIDEOS_DIR,
	YOLO_MODEL_NAME,
	CONFIDENCE_THRESHOLD,
	WALKWAY_STRIDE,
	VISION_STRIDE,
)
from src.detection.run_detection import run_all_detectors
from src.escalation.route_event import route_detection_event
from src.reports.db import ReportDB
from src.reports.export import export_events_to_csv


def main():
	"""
	Execute the detection pipeline on a single video clip.
	"""
	parser = argparse.ArgumentParser(description="Run Factory Compliance Detection Pipeline")
	parser.add_argument("--video", type=str, default="clip1.mp4", help="Video filename in data/videos")
	parser.add_argument("--zone", type=str, default="Production Floor", help="Facility zone label")
	parser.add_argument("--skip-vision", action="store_true", help="Skip Groq API vision detectors")
	args = parser.parse_args()

	video_path = VIDEOS_DIR / args.video
	if not video_path.exists():
		print(f"Error: Video not found at {video_path}")
		return

	print(f"Starting pipeline on {video_path.name}...")
	print("Running detection models...")

	# Phase 1: Detection
	detections = run_all_detectors(
		video_path,
		zone=args.zone,
		stride=WALKWAY_STRIDE,
		vision_stride=VISION_STRIDE,
		model_name=YOLO_MODEL_NAME,
		confidence_threshold=CONFIDENCE_THRESHOLD,
		skip_vision=args.skip_vision,
		step_callback=lambda msg: print(f"  {msg}"),
	)

	if not detections:
		print("No violations detected.")
		return

	print(f"Routing {len(detections)} events through escalation pipeline...")

	# Phase 2: Severity lookup & Escalation
	db = ReportDB(DB_PATH)
	for detection in detections:
		# Route event handles severity lookup and pushes high-severity items to AlertQueue
		event_payload = route_detection_event(detection)
		
		# Persist to historical SQLite DB
		db.insert_event(event_payload)
		
		sev = event_payload["severity"]
		print(f"  -> Logged {sev} event: {event_payload['behavior_class']}")

	# Phase 3: Reporting
	print("Exporting report...")
	export_events_to_csv(db, CSV_EXPORT_PATH)
	print(f"Pipeline complete. Report saved to {CSV_EXPORT_PATH.relative_to(DB_PATH.parent.parent)}")


if __name__ == "__main__":
	main()