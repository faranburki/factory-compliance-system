"""
Module 1 — Policy Parsing | schema.py
Defines the Pydantic models used to structure policy-rule extraction.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Dynamic string for any extracted behavior class
BehaviorClass = str

SeverityTier = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

class PolicyRule(BaseModel):
	"""
	A single extracted policy rule defining a behavior class and its severity.
	"""

	behavior_class: BehaviorClass = Field(
		...,
		description="The name of the unsafe behavior class as defined in the policy.",
	)
	policy_rule_ref: str = Field(
		...,
		description="Policy section reference that defines the rule, such as Section 3.3.2.",
	)
	rule_text: str = Field(
		...,
		description="Plain-English summary of the rule extracted from the policy text.",
	)
	source_excerpt: str = Field(
		...,
		description="Short quote from the source policy text that supports the rule.",
	)
	severity: SeverityTier = Field(
		...,
		description="Risk severity tier (LOW, MEDIUM, HIGH, CRITICAL) derived from the policy's hazard context.",
	)
	observable_indicators: list[str] = Field(
		...,
		description="List of observable visual indicators to detect this behavior (e.g., 'green vest', '3 or more blocks').",
	)

class PolicyRuleExtraction(BaseModel):
	"""
	Structured Groq output schema for the entire policy-rule extraction step.
	"""

	rules: list[PolicyRule] = Field(
		...,
		description="All unsafe behaviors extracted from the policy document.",
	)
