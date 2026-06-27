"""Groq-based policy-rule extraction helpers."""

from __future__ import annotations

import os
from pathlib import Path

try:
	from groq import Groq
except ImportError:  # pragma: no cover - depends on installed environment
	Groq = None

from .schema import PolicyRuleExtraction


def _get_api_key() -> str:
	api_key = os.getenv("GROQ_API_KEY")
	if not api_key:
		raise RuntimeError("Set GROQ_API_KEY before calling Groq")
	return api_key


def build_policy_rules_prompt(policy_text: str) -> str:
	"""Build the dynamic prompt used to extract all policy rules."""

	return f"""You are an AI tasked with analyzing an occupational health and safety policy to extract compliance rules for automated video monitoring.

Task:
1. Identify ALL distinct unsafe behavior categories (Behavior Classes) defined in the policy.
2. For each unsafe behavior, extract its rule reference, a plain-English rule text, and a source excerpt.
3. Determine the Risk Severity Tier (LOW, MEDIUM, HIGH, CRITICAL) for each behavior. Base this strictly on the hazard context, warning callouts, frequency data, and alerting language described in the policy. IMPORTANT: "WARNING" callouts must map to MEDIUM severity. "CRITICAL SAFETY NOTICE" must map to CRITICAL severity.
4. Extract the observable visual indicators needed to detect this behavior from a video feed (e.g., 'green vest', '3 or more blocks', 'green marked walkway').
5. Extract any numeric thresholds mentioned in the observable indicators into a `detection_parameters` dictionary (e.g., {{"overload_threshold": 3}}).

Return JSON that matches this structure:
{{
  "rules": [
    {{
      "behavior_class": "...",
      "policy_rule_ref": "...",
      "rule_text": "...",
      "source_excerpt": "...",
      "severity": "...",
      "observable_indicators": ["...", "..."],
      "detection_parameters": {{"overload_threshold": 3}}
    }}
  ]
}}

Do not invent facts or paraphrase section numbers.

Policy text:
{policy_text}
"""


def extract_policy_rules(policy_text: str, *, model: str = "llama-3.1-8b-instant") -> PolicyRuleExtraction:
	"""Send one prompt to Groq and parse the structured dynamic policy-rule response."""

	if Groq is None:
		raise RuntimeError("groq is not installed in the current Python environment")

	client = Groq(api_key=_get_api_key())
	prompt = build_policy_rules_prompt(policy_text)
	response = client.chat.completions.create(
		model=model,
		messages=[
			{
				"role": "system",
				"content": "You extract structured policy rules dynamically and return JSON only.",
			},
			{
				"role": "user",
				"content": prompt,
			},
		],
		temperature=0.0,
		response_format={"type": "json_object"},
	)

	content = response.choices[0].message.content or "{}"
	return PolicyRuleExtraction.model_validate_json(content)


def write_policy_rules_json(extraction: PolicyRuleExtraction, output_path: str | Path) -> Path:
	"""Write the extracted rules to a JSON file on disk."""

	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(extraction.model_dump_json(indent=2), encoding="utf-8")
	return path


def review_policy_rules(extraction: PolicyRuleExtraction) -> str:
	"""Format the Groq output so it can be sanity-checked by eye."""

	return extraction.model_dump_json(indent=2)

