"""
Module 2 — Risk Severity | classify_severity.py
Looks up the dynamically generated severity tier for an extracted violation.
"""

from __future__ import annotations

import json
from pathlib import Path

# Paths are resolved relative to this module to locate the dynamic policy rules
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POLICY_RULES_JSON = _PROJECT_ROOT / "outputs" / "policy_rules.json"


def get_severity_for_behavior(behavior_class: str, *, default_severity: str = "MEDIUM") -> str:
	"""
	Lookup the severity tier for a given behavior class from the parsed policy rules.

	Args:
		behavior_class: The name of the unsafe behavior.
		default_severity: The fallback severity if the rule is missing or malformed.

	Returns:
		The severity string (LOW, MEDIUM, HIGH, CRITICAL).
	"""
	if not POLICY_RULES_JSON.exists():
		return default_severity

	try:
		data = json.loads(POLICY_RULES_JSON.read_text(encoding="utf-8"))
		rules = data.get("rules", [])
		
		# Find the exact behavior class in the extracted rules
		for rule in rules:
			if rule.get("behavior_class") == behavior_class:
				return rule.get("severity", default_severity)
	except Exception:
		# If the JSON is invalid or unreadable, fail gracefully
		pass

	return default_severity
