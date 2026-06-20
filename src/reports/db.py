"""SQLite helpers for report events."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from .schema import ReportEvent


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    clip_id TEXT NOT NULL,
    zone TEXT NOT NULL,
    behavior_class TEXT NOT NULL,
    policy_rule_ref TEXT NOT NULL,
    event_description TEXT NOT NULL,
    severity TEXT NOT NULL,
    escalation_action TEXT NOT NULL
)
"""


def get_connection(db_path: str | Path) -> sqlite3.Connection:
	path = Path(db_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	return sqlite3.connect(path)


def initialize_database(db_path: str | Path) -> None:
	with get_connection(db_path) as connection:
		connection.execute(CREATE_TABLE_SQL)
		connection.commit()


def insert_event(db_path: str | Path, event: ReportEvent) -> None:
	initialize_database(db_path)
	with get_connection(db_path) as connection:
		connection.execute(
			"""
			INSERT OR REPLACE INTO events (
			    event_id,
			    timestamp,
			    clip_id,
			    zone,
			    behavior_class,
			    policy_rule_ref,
			    event_description,
			    severity,
			    escalation_action
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
			""",
			(
				event.event_id,
				event.timestamp,
				event.clip_id,
				event.zone,
				event.behavior_class,
				event.policy_rule_ref,
				event.event_description,
				event.severity,
				event.escalation_action,
			),
		)
		connection.commit()


def fetch_events(
	db_path: str | Path,
	*,
	severity: Optional[str] = None,
	behavior_class: Optional[str] = None,
) -> list[ReportEvent]:
	"""Fetch events from the database with optional filtering."""
	initialize_database(db_path)
	query = "SELECT event_id, timestamp, clip_id, zone, behavior_class, policy_rule_ref, event_description, severity, escalation_action FROM events"
	conditions: list[str] = []
	params: list[str] = []

	if severity:
		conditions.append("severity = ?")
		params.append(severity)
	if behavior_class:
		conditions.append("behavior_class = ?")
		params.append(behavior_class)

	if conditions:
		query += " WHERE " + " AND ".join(conditions)

	query += " ORDER BY timestamp DESC"

	with get_connection(db_path) as connection:
		rows = connection.execute(query, params).fetchall()

	return [
		ReportEvent.model_validate(
			{
				"event_id": row[0],
				"timestamp": row[1],
				"clip_id": row[2],
				"zone": row[3],
				"behavior_class": row[4],
				"policy_rule_ref": row[5],
				"event_description": row[6],
				"severity": row[7],
				"escalation_action": row[8],
			}
		)
		for row in rows
	]
