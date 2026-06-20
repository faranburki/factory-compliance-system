"""
Module 4 — Reporting | db.py
Provides CRUD operations for historical compliance events in the SQLite database.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .schema import init_db


class ReportDB:
	"""
	Manager for persisting and querying historical compliance events.
	"""

	def __init__(self, db_path: str | Path):
		"""
		Initialize the database connection wrapper.

		Args:
			db_path: Path to the SQLite database.
		"""
		self.db_path = Path(db_path)
		init_db(self.db_path)

	def insert_event(self, event: dict[str, Any]) -> None:
		"""
		Insert a new compliance event into the historical log.

		Args:
			event: Standardized event dictionary.
		"""
		with sqlite3.connect(self.db_path) as conn:
			conn.execute(
				"""
				INSERT INTO compliance_events (
					event_id, timestamp, behavior_class, severity, policy_rule_ref,
					event_description, zone, confidence, clip_id, frame_index, needs_review
				)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""",
				(
					event["event_id"],
					event["timestamp"],
					event["behavior_class"],
					event["severity"],
					event.get("policy_rule_ref"),
					event.get("event_description"),
					event.get("zone"),
					event.get("confidence"),
					event.get("clip_id"),
					event.get("frame_index"),
					1 if event.get("needs_review") else 0,
				),
			)

	def get_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
		"""
		Retrieve the most recent compliance events.

		Args:
			limit: Maximum number of rows to return.

		Returns:
			A list of dictionary records representing the events.
		"""
		with sqlite3.connect(self.db_path) as conn:
			# Row factory allows dict-like access to columns
			conn.row_factory = sqlite3.Row
			cursor = conn.execute(
				"SELECT * FROM compliance_events ORDER BY timestamp DESC LIMIT ?",
				(limit,),
			)
			return [dict(row) for row in cursor.fetchall()]

	def get_events_by_severity(self, severity: str, limit: int = 50) -> list[dict[str, Any]]:
		"""
		Retrieve recent events filtered by a specific severity tier.

		Args:
			severity: The severity tier to filter on (e.g. HIGH).
			limit: Maximum number of rows to return.

		Returns:
			A list of dictionary records representing the filtered events.
		"""
		with sqlite3.connect(self.db_path) as conn:
			conn.row_factory = sqlite3.Row
			cursor = conn.execute(
				"SELECT * FROM compliance_events WHERE severity = ? ORDER BY timestamp DESC LIMIT ?",
				(severity, limit),
			)
			return [dict(row) for row in cursor.fetchall()]
