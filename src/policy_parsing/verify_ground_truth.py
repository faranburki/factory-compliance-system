"""Ground-truth reference and comparison helpers for policy-rule extraction."""

from __future__ import annotations

from .schema import PolicyRule, PolicyRuleExtraction


GROUND_TRUTH = PolicyRuleExtraction(
	rules=[
		PolicyRule(
			behavior_class="Safe Walkway Violation",
			policy_rule_ref="Section 3.3.2",
			rule_text="A person on foot must stay within the green-marked designated safe walkway.",
			source_excerpt="movement or presence outside the boundaries of the green-marked Designated Safe Walkway",
			severity="MEDIUM",
			observable_indicators=["green marked walkway", "outside boundaries"],
		),
		PolicyRule(
			behavior_class="Unauthorized Intervention",
			policy_rule_ref="Section 4.3.2",
			rule_text="Personnel may only interact with equipment while wearing the green authorization vest and required safety equipment.",
			source_excerpt="interacting with or adjusting production equipment while wearing a red-black vest or any vest other than the designated green authorization vest",
			severity="HIGH",
			observable_indicators=["green vest", "red-black vest"],
		),
		PolicyRule(
			behavior_class="Opened Panel Cover",
			policy_rule_ref="Section 5.2.2",
			rule_text="An electrical panel cover must not be left open during production operations.",
			source_excerpt="the cover of an electrical panel connected to a production machine has been left in the open position",
			severity="LOW",
			observable_indicators=["electrical panel cover", "open position"],
		),
		PolicyRule(
			behavior_class="Carrying Overload with Forklift",
			policy_rule_ref="Section 6.3.2",
			rule_text="A forklift carrying three or more standardized blocks is an overload and unsafe.",
			source_excerpt="operating a forklift while carrying three (3) or more standardized blocks in a single load",
			severity="CRITICAL",
			observable_indicators=["forklift", "3 or more blocks"],
		),
	]
)


def get_ground_truth_rules() -> PolicyRuleExtraction:
	"""Return the hand-built ground-truth rule set from the policy manual."""

	return GROUND_TRUTH


def _normalize(value: str) -> str:
	return " ".join(value.split()).strip().lower()


def compare_policy_rules(extracted: PolicyRuleExtraction, expected: PolicyRuleExtraction | None = None) -> list[str]:
	"""Compare extracted rules against the hand-built reference field by field."""

	expected_rules = (expected or GROUND_TRUTH).rules
	extracted_by_class = {rule.behavior_class: rule for rule in extracted.rules}
	expected_by_class = {rule.behavior_class: rule for rule in expected_rules}
	mismatches: list[str] = []

	for behavior_class in expected_by_class:
		if behavior_class not in extracted_by_class:
			mismatches.append(f"Missing rule: {behavior_class}")
			continue

		extracted_rule = extracted_by_class[behavior_class]
		expected_rule = expected_by_class[behavior_class]

		for field_name in ("policy_rule_ref", "rule_text", "source_excerpt"):
			extracted_value = _normalize(getattr(extracted_rule, field_name))
			expected_value = _normalize(getattr(expected_rule, field_name))
			if extracted_value != expected_value:
				mismatches.append(
					f"{behavior_class} {field_name} mismatch: expected {getattr(expected_rule, field_name)!r}, got {getattr(extracted_rule, field_name)!r}",
				)

	return mismatches
