"""
Module 4 — Reporting | export.py
Exports historical compliance events to a CSV file.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .db import ReportDB


def export_events_to_csv(db: ReportDB, output_path: str | Path, limit: int = 1000) -> Path:
	"""
	Query the database and write recent events to a CSV format.

	Args:
		db: An active ReportDB instance.
		output_path: Path where the CSV should be saved.
		limit: Maximum number of events to export.

	Returns:
		The path to the generated CSV file.
	"""
	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	events = db.get_recent_events(limit=limit)

	if not events:
		# Write empty CSV with headers if no events exist
		with path.open("w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow(["event_id", "timestamp", "behavior_class", "severity", "zone"])
		return path

	fieldnames = list(events[0].keys())

	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for event in events:
			writer.writerow(event)

	return path
