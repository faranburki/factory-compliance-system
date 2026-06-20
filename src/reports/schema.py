"""
Module 4 — Reporting | schema.py
Defines the SQLite schema and initialization logic for the historical database.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def init_db(db_path: str | Path) -> None:
	"""
	Initialize the SQLite database schema if it doesn't already exist.

	Args:
		db_path: Path to the SQLite database file.
	"""
	path = Path(db_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	
	with sqlite3.connect(path) as conn:
		# Use a comprehensive schema to track all necessary event context
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS compliance_events (
				event_id TEXT PRIMARY KEY,
				timestamp TEXT NOT NULL,
				behavior_class TEXT NOT NULL,
				severity TEXT NOT NULL,
				policy_rule_ref TEXT,
				event_description TEXT,
				zone TEXT,
				confidence REAL,
				clip_id TEXT,
				frame_index INTEGER,
				needs_review BOOLEAN DEFAULT 0
			)
			"""
		)
		# Add indices to optimize common reporting queries (filtering by time or severity)
		conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON compliance_events(timestamp)")
		conn.execute("CREATE INDEX IF NOT EXISTS idx_severity ON compliance_events(severity)")
