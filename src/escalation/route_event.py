"""
Module 3 — Escalation Pipeline | route_event.py
Routes detected violations based on their dynamically looked-up severity tier.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from ..detection.run_detection import DetectionResult
from ..severity.classify_severity import get_severity_for_behavior
from .alert_queue import push_alert


def create_event_payload(detection: DetectionResult, severity: str) -> dict[str, Any]:
	"""
	Format a raw detection result into a standardized database/alert event schema.

	Args:
		detection: The unified detection result.
		severity: The corresponding risk severity tier.

	Returns:
		A standardized event dictionary payload.
	"""
	return {
		"event_id": str(uuid.uuid4()),
		# Store time in ISO 8601 format with UTC timezone for consistency
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"behavior_class": detection.behavior_class,
		"severity": severity,
		"policy_rule_ref": detection.policy_rule_ref,
		"event_description": detection.event_description,
		"zone": detection.zone,
		"confidence": detection.confidence,
		"clip_id": detection.clip_id,
		"frame_index": detection.frame_index,
		"needs_review": detection.needs_review,
	}


def route_detection_event(detection: DetectionResult) -> dict[str, Any]:
	"""
	Determine the severity of a detection and route it accordingly.
	
	HIGH and CRITICAL events are dispatched to the real-time alert queue.
	All events are returned for persistence in the historical report database.

	Args:
		detection: The unified detection result to process.

	Returns:
		The fully formed event payload ready for database insertion.
	"""
	severity = get_severity_for_behavior(detection.behavior_class)
	payload = create_event_payload(detection, severity)

	# Escalation routing: only severe events get real-time attention
	if severity in ("HIGH", "CRITICAL"):
		push_alert(payload)

	return payload
