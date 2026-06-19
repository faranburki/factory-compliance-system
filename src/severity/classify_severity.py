"""Map detected behavior classes to severity tiers dynamically."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

SeverityTier = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POLICY_RULES_JSON = _PROJECT_ROOT / "outputs" / "policy_rules.json"

def classify_severity(behavior_class: str) -> SeverityTier:
	"""Dynamically look up severity tier from parsed policy rules."""
	
	if not POLICY_RULES_JSON.exists():
		return "MEDIUM"  # Default fallback if rules aren't parsed yet
		
	try:
		data = json.loads(POLICY_RULES_JSON.read_text(encoding="utf-8"))
		for rule in data.get("rules", []):
			if rule.get("behavior_class") == behavior_class:
				# ensure it returns a valid SeverityTier or defaults to MEDIUM
				sev = rule.get("severity", "MEDIUM").upper()
				if sev in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
					return sev
				return "MEDIUM"
	except Exception:
		pass
		
	return "MEDIUM"
