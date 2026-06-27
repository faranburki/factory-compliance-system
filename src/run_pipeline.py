"""Full end-to-end pipeline: detect → severity → escalate → report.

Scans all video clips in data/, runs all four detectors on each clip,
classifies severity, routes events through the escalation pipeline,
and writes structured reports to SQLite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is on the path when run as a script
_SRC_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_ROOT.parent
sys.path.insert(0, str(_SRC_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from detection.run_detection import run_all_detectors
from escalation.alert_queue import create_alert_queue
from escalation.route_event import build_report_event, route_event
from reports.db import fetch_events


# ── Defaults ────────────────────────────────────────────────────────────
DATA_DIR = _PROJECT_ROOT / "data"
DB_PATH = _PROJECT_ROOT / "outputs" / "reports.db"
POLICY_RULES_JSON = _PROJECT_ROOT / "outputs" / "policy_rules.json"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}


def discover_clips(data_dir: Path) -> list[Path]:
	"""Find all video files in the data directory."""
	clips = sorted(
		p for p in data_dir.iterdir()
		if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
	)
	return clips


def run_pipeline(
	*,
	video_path: Path | None = None,
	data_dir: Path = DATA_DIR,
	db_path: Path = DB_PATH,
	zone: str = "Production Floor",
	stride: int = 30,
	max_frames: int | None = None,
	vision_stride: int = 60,
	vision_max_frames: int = 5,
	skip_vision: bool = False,
	step_callback = None,
) -> None:
	"""Execute the full pipeline across one or all clips."""

	# Determine which clips to process
	if video_path is not None:
		clips = [Path(video_path)]
	else:
		clips = discover_clips(data_dir)

	if not clips:
		print(f"No video clips found in {data_dir}")
		return

	print(f"{'=' * 60}")
	print(f"Factory Compliance Pipeline")
	print(f"{'=' * 60}")
	print(f"Clips to process: {len(clips)}")
	print(f"Database: {db_path}")
	print(f"Vision API: {'DISABLED' if skip_vision else 'ENABLED (Groq)'}")
	print(f"{'=' * 60}\n")

	alert_queue = create_alert_queue()
	total_events = 0

	for clip_idx, clip in enumerate(clips, 1):
		print(f"\n{'─' * 60}")
		print(f"[{clip_idx}/{len(clips)}] Processing: {clip.name}")
		print(f"{'─' * 60}")

		# ── Step 1: Detect violations ───────────────────────────────
		detection_results = run_all_detectors(
			clip,
			zone=zone,
			stride=stride,
			max_frames=max_frames,
			vision_stride=vision_stride,
			vision_max_frames=vision_max_frames,
			skip_vision=skip_vision,
			step_callback=step_callback,
		)

		if not detection_results:
			print(f"  No violations detected in {clip.name}")
			continue

		# ── Step 2 & 3: Severity + Escalation ──────────────────────
		print(f"\n  Routing {len(detection_results)} event(s) through escalation pipeline...")
		last_alert_frame: dict[str, int] = {}
		
		for det in detection_results:
			# Debounce DB + Alert generation (max 1 per 60 frames / 2 seconds per behavior class)
			if det.frame_index is not None:
				last_frame = last_alert_frame.get(det.behavior_class, -999)
				if det.frame_index - last_frame < 60:
					continue  # Throttle: skip for this frame
				last_alert_frame[det.behavior_class] = det.frame_index

			report_event = build_report_event(
				clip_id=det.clip_id,
				zone=det.zone,
				behavior_class=det.behavior_class,
				policy_rule_ref=det.policy_rule_ref,
				event_description=det.event_description,
				observable_indicator_ref=det.observable_indicator_ref,
			)

			# Route to DB (and alert queue if HIGH/CRITICAL)
			route_event(db_path, report_event, alert_queue)
			total_events += 1

			severity_indicator = {
				"LOW": "🟢",
				"MEDIUM": "🟡",
				"HIGH": "🟠",
				"CRITICAL": "🔴",
			}.get(report_event.severity, "⚪")

			print(
				f"    {severity_indicator} [{report_event.severity}] {report_event.behavior_class} "
				f"(event_id={report_event.event_id[:8]}...)"
			)

	# ── Summary ─────────────────────────────────────────────────────
	print(f"\n{'=' * 60}")
	print(f"Pipeline Complete")
	print(f"{'=' * 60}")
	print(f"Total events logged: {total_events}")

	# Count alerts in queue
	alert_count = 0
	while not alert_queue.empty():
		alert_queue.pop()
		alert_count += 1
	print(f"HIGH/CRITICAL alerts queued: {alert_count}")

	# Show DB summary
	all_events = fetch_events(db_path)
	print(f"Events in database: {len(all_events)}")

	# Breakdown by severity
	severity_counts: dict[str, int] = {}
	for ev in all_events:
		severity_counts[ev.severity] = severity_counts.get(ev.severity, 0) + 1
	for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
		if sev in severity_counts:
			print(f"  {sev}: {severity_counts[sev]}")

	print(f"\nDatabase saved to: {db_path}")
	print(f"{'=' * 60}")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Factory Compliance Pipeline — detect → severity → escalate → report"
	)
	parser.add_argument(
		"--video", type=Path, default=None,
		help="Path to a single video clip (default: scan all clips in data/)."
	)
	parser.add_argument(
		"--data-dir", type=Path, default=DATA_DIR,
		help="Directory containing video clips."
	)
	parser.add_argument(
		"--db", type=Path, default=DB_PATH,
		help="Path to the SQLite database."
	)
	parser.add_argument(
		"--zone", type=str, default="Production Floor",
		help="Facility zone label."
	)
	parser.add_argument(
		"--stride", type=int, default=30,
		help="Frame sampling stride for walkway/vest detectors."
	)
	parser.add_argument(
		"--max-frames", type=int, default=None,
		help="Max frames to sample for walkway/vest detectors."
	)
	parser.add_argument(
		"--vision-stride", type=int, default=60,
		help="Frame sampling stride for vision-based detectors."
	)
	parser.add_argument(
		"--vision-max-frames", type=int, default=5,
		help="Max frames for vision-based detectors."
	)
	parser.add_argument(
		"--skip-vision", action="store_true",
		help="Skip Groq vision API calls (panel and forklift detectors)."
	)
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	run_pipeline(
		video_path=args.video,
		data_dir=args.data_dir,
		db_path=args.db,
		zone=args.zone,
		stride=args.stride,
		max_frames=args.max_frames,
		vision_stride=args.vision_stride,
		vision_max_frames=args.vision_max_frames,
		skip_vision=args.skip_vision,
	)


if __name__ == "__main__":
	main()