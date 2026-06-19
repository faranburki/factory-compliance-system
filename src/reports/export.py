"""CSV and JSON export for compliance events stored in SQLite."""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd

from .db import fetch_events
from .schema import ReportEvent


def events_to_dicts(events: list[ReportEvent]) -> list[dict]:
	"""Convert a list of ReportEvent objects to a list of dicts."""
	return [event.model_dump() for event in events]


def export_events_json(
	db_path: str | Path,
	output_path: str | Path | None = None,
	*,
	severity: Optional[str] = None,
	behavior_class: Optional[str] = None,
) -> str:
	"""Export events from SQLite to a JSON string (and optionally a file).

	Returns the JSON string.
	"""
	events = fetch_events(db_path, severity=severity, behavior_class=behavior_class)
	data = events_to_dicts(events)
	json_str = json.dumps(data, indent=2, ensure_ascii=False)

	if output_path is not None:
		path = Path(output_path)
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(json_str, encoding="utf-8")

	return json_str


def export_events_csv(
	db_path: str | Path,
	output_path: str | Path | None = None,
	*,
	severity: Optional[str] = None,
	behavior_class: Optional[str] = None,
) -> str:
	"""Export events from SQLite to a CSV string (and optionally a file).

	Returns the CSV string.
	"""
	events = fetch_events(db_path, severity=severity, behavior_class=behavior_class)
	data = events_to_dicts(events)

	if not data:
		return ""

	df = pd.DataFrame(data)
	csv_str = df.to_csv(index=False)

	if output_path is not None:
		path = Path(output_path)
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(csv_str, encoding="utf-8")

	return csv_str


def get_events_dataframe(
	db_path: str | Path,
	*,
	severity: Optional[str] = None,
	behavior_class: Optional[str] = None,
) -> pd.DataFrame:
	"""Return events as a pandas DataFrame for filtering and analysis."""
	events = fetch_events(db_path, severity=severity, behavior_class=behavior_class)
	data = events_to_dicts(events)
	return pd.DataFrame(data) if data else pd.DataFrame()
