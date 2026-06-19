"""Schema for report events written to SQLite and surfaced in the dashboard."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from policy_parsing.schema import BehaviorClass
from severity.classify_severity import SeverityTier


def _new_event_id() -> str:
	return str(uuid.uuid4())


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


class ReportEvent(BaseModel):
	event_id: str = Field(default_factory=_new_event_id, description="Unique identifier for this violation event.")
	timestamp: str = Field(default_factory=_now_iso, description="Wall-clock time the violation was detected (ISO 8601).")
	clip_id: str = Field(..., description="Source video clip identifier.")
	zone: str = Field(..., description="Facility zone label.")
	behavior_class: BehaviorClass = Field(..., description="Detected unsafe behavior class.")
	policy_rule_ref: str = Field(..., description="Policy section reference for the rule.")
	event_description: str = Field(..., description="Plain-English description of what was observed.")
	severity: SeverityTier = Field(..., description="Severity tier assigned to the event.")
	escalation_action: str = Field(..., description="What the escalation pipeline did with the event.")
