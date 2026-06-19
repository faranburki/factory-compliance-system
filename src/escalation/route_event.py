"""Route report events to SQLite and the in-process alert queue."""

from __future__ import annotations

from pathlib import Path

from policy_parsing.schema import BehaviorClass
from reports.db import insert_event
from reports.schema import ReportEvent
from severity.classify_severity import classify_severity

from .alert_queue import AlertQueue


def build_report_event(
	clip_id: str,
	zone: str,
	behavior_class: BehaviorClass,
	policy_rule_ref: str,
	event_description: str,
	*,
	escalation_action: str | None = None,
) -> ReportEvent:
	"""Create a fully populated ReportEvent with auto-generated event_id and timestamp."""
	severity = classify_severity(behavior_class)
	if escalation_action is None:
		escalation_action = "Queued for real-time alert" if severity in {"HIGH", "CRITICAL"} else "Logged to database"
	return ReportEvent(
		clip_id=clip_id,
		zone=zone,
		behavior_class=behavior_class,
		policy_rule_ref=policy_rule_ref,
		event_description=event_description,
		severity=severity,
		escalation_action=escalation_action,
	)


def route_event(db_path: str | Path, event: ReportEvent, alert_queue: AlertQueue | None = None) -> ReportEvent:
	"""Persist every event to SQLite and queue HIGH/CRITICAL events."""

	insert_event(db_path, event)
	if alert_queue is not None and event.severity in {"HIGH", "CRITICAL"}:
		alert_queue.push(event)
	return event
