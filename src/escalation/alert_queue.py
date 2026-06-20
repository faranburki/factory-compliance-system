"""
Module 3 — Escalation Pipeline | alert_queue.py
In-memory FIFO queue for buffering real-time HIGH and CRITICAL alerts.
"""

from __future__ import annotations

import queue
from typing import Any

# Global queue instance accessible across the application
# queue.SimpleQueue is thread-safe and non-blocking for puts,
# making it ideal for the real-time escalation requirement.
AlertQueue: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()


def push_alert(alert_payload: dict[str, Any]) -> None:
	"""
	Push a high-severity alert onto the real-time buffer queue.

	Args:
		alert_payload: A dictionary representing the alert event.
	"""
	AlertQueue.put(alert_payload)


def pop_all_alerts() -> list[dict[str, Any]]:
	"""
	Retrieve and drain all pending alerts from the queue.

	Returns:
		A list of alert dictionaries in FIFO order.
	"""
	alerts = []
	try:
		while True:
			# Non-blocking get; raises queue.Empty when the queue is exhausted
			alerts.append(AlertQueue.get_nowait())
	except queue.Empty:
		pass
	return alerts
