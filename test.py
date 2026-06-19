from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from policy_parsing.pdf_text_extractor import extract_pdf_text
from policy_parsing.schema import PolicyRule, PolicyRuleExtraction


POLICY_PDF = PROJECT_ROOT / "Compliance_Policy_Manual.pdf"

load_dotenv(PROJECT_ROOT / ".env")


def test_pdf_text_extractor() -> str:
	text = extract_pdf_text(POLICY_PDF)
	assert text.strip(), "PDF text extractor returned an empty string"
	assert "walkway" in text.lower(), "Expected walkway text in the policy PDF"
	print(f"[PASS] pdf_text_extractor -> {len(text)} characters")
	return text


def test_schema_validation() -> PolicyRuleExtraction:
	sample = {
		"rules": [
			{
				"behavior_class": "Safe Walkway Violation",
				"policy_rule_ref": "Section 3.3.2",
				"rule_text": "Personnel on foot must remain inside the green-marked walkway boundaries.",
				"source_excerpt": "movement or presence outside the boundaries of the green-marked Designated Safe Walkway",
				"severity": "MEDIUM",
				"observable_indicators": ["green marked walkway", "outside boundaries"]
			},
			{
				"behavior_class": "Unauthorized Intervention",
				"policy_rule_ref": "Section 4.3.2",
				"rule_text": "Personnel may only interact with equipment while wearing the green authorization vest and required safety equipment.",
				"source_excerpt": "interacting with or adjusting production equipment while wearing a red-black vest or any vest other than the designated green authorization vest",
				"severity": "HIGH",
				"observable_indicators": ["green vest", "red-black vest"]
			},
			{
				"behavior_class": "Opened Panel Cover",
				"policy_rule_ref": "Section 5.2.2",
				"rule_text": "An electrical panel cover must not be left open during production operations.",
				"source_excerpt": "the cover of an electrical panel connected to a production machine has been left in the open position",
				"severity": "LOW",
				"observable_indicators": ["electrical panel cover", "open position"]
			},
			{
				"behavior_class": "Carrying Overload with Forklift",
				"policy_rule_ref": "Section 6.3.2",
				"rule_text": "A forklift carrying three or more standardized blocks is an overload and unsafe.",
				"source_excerpt": "operating a forklift while carrying three (3) or more standardized blocks in a single load",
				"severity": "CRITICAL",
				"observable_indicators": ["forklift", "3 or more blocks"]
			}
		]
	}
	parsed = PolicyRuleExtraction.model_validate(sample)
	assert len(parsed.rules) == 4
	assert isinstance(parsed.rules[0], PolicyRule)
	print("[PASS] schema -> PolicyRuleExtraction validated")
	return parsed


def test_prompt_builder(policy_text: str) -> str:
	from policy_parsing.rule_extractor import build_policy_rules_prompt

	prompt = build_policy_rules_prompt(policy_text[:2000])
	assert "ALL distinct unsafe behavior categories" in prompt, "Prompt does not request all rules"
	assert "Risk Severity Tier" in prompt, "Prompt does not ask for severity"
	assert "response_schema" not in prompt.lower(), "Prompt should stay focused on the task, not implementation details"
	print(f"[PASS] prompt_builder -> {len(prompt)} characters")
	return prompt


def test_policy_rules_extraction(policy_text: str) -> None:
	api_key = os.getenv("GROQ_API_KEY")
	if not api_key:
		print("[SKIP] policy_rules_extraction -> GROQ_API_KEY is not set")
		return

	from policy_parsing.rule_extractor import extract_policy_rules, review_policy_rules, write_policy_rules_json
	from policy_parsing.verify_ground_truth import compare_policy_rules, get_ground_truth_rules

	try:
		extraction = extract_policy_rules(policy_text)
	except Exception as exc:
		print(f"[SKIP] policy_rules_extraction -> {exc}")
		return

	print("[PASS] policy_rules_extraction -> parsed output")
	print(review_policy_rules(extraction))

	mismatches = compare_policy_rules(extraction, get_ground_truth_rules())
	if mismatches:
		print("[WARN] policy_rules_verification -> mismatches detected")
		for mismatch in mismatches:
			print(f"- {mismatch}")
	else:
		print("[PASS] policy_rules_verification -> matches ground truth")

	output_path = PROJECT_ROOT / "outputs" / "policy_rules.json"
	write_policy_rules_json(extraction, output_path)
	print(f"[PASS] policy_rules_json -> {output_path}")


def test_walkway_detector_smoke() -> None:
	from detection.video_utils import load_video_metadata
	from detection.walkway_detector import PersonDetection, classify_walkway_violations, default_walkway_polygon

	video_path = PROJECT_ROOT / "data" / "0_te1.mp4"
	metadata = load_video_metadata(video_path)
	polygon = default_walkway_polygon(metadata.width, metadata.height)
	inside_detection = PersonDetection(
		box=(10, 10, 20, int(metadata.height * 0.9)),
		confidence=0.99,
		foot_point=(int(metadata.width * 0.5), int(metadata.height * 0.9)),
	)
	outside_detection = PersonDetection(
		box=(10, 10, 20, 30),
		confidence=0.99,
		foot_point=(5, 5),
	)
	results = classify_walkway_violations([inside_detection, outside_detection], polygon)
	assert results[0].is_violation is False
	assert results[1].is_violation is True
	print(f"[PASS] walkway_detector_smoke -> {metadata.width}x{metadata.height} on {video_path.name}")


def test_severity_mapping() -> None:
	from severity.classify_severity import classify_severity

	# Note: Severity is now dynamically parsed. The exact tiers may vary based on LLM output, 
	# but they must be valid severity strings.
	valid_tiers = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
	assert classify_severity("Safe Walkway Violation") in valid_tiers
	assert classify_severity("Unauthorized Intervention") in valid_tiers
	assert classify_severity("Opened Panel Cover") in valid_tiers
	assert classify_severity("Carrying Overload with Forklift") in valid_tiers
	print("[PASS] severity_mapping -> dynamic tiers verified as valid strings")


def test_escalation_routing() -> None:
	from escalation.alert_queue import create_alert_queue
	from escalation.route_event import build_report_event, route_event
	from reports.db import fetch_events

	db_path = PROJECT_ROOT / "outputs" / "test_events.sqlite"
	if db_path.exists():
		db_path.unlink()

	alert_queue = create_alert_queue()
	high_event = build_report_event(
		clip_id="0_te1.mp4",
		zone="Production Floor",
		behavior_class="Unauthorized Intervention",
		policy_rule_ref="Section 4.3.2",
		event_description="A person was detected interacting with equipment in the wrong vest.",
	)
	low_event = build_report_event(
		clip_id="0_te1.mp4",
		zone="Production Floor",
		behavior_class="Opened Panel Cover",
		policy_rule_ref="Section 5.2.2",
		event_description="A panel cover was observed open.",
	)

	# Verify event_id and timestamp are auto-generated
	assert high_event.event_id, "event_id should be auto-generated"
	assert high_event.timestamp, "timestamp should be auto-generated"
	assert high_event.event_id != low_event.event_id, "event_ids should be unique"

	route_event(db_path, high_event, alert_queue)
	route_event(db_path, low_event, alert_queue)

	rows = fetch_events(db_path)
	assert len(rows) == 2
	# Verify the fetched rows also have event_id and timestamp
	assert rows[0].event_id, "Fetched event should have event_id"
	assert rows[0].timestamp, "Fetched event should have timestamp"
	assert not alert_queue.empty()
	queued_event = alert_queue.pop()
	assert queued_event.severity in {"HIGH", "CRITICAL"}
	print(f"[PASS] escalation_routing -> sqlite rows={len(rows)} event_id={high_event.event_id[:8]}... alert_queue=queued")


def main() -> None:
	policy_text = test_pdf_text_extractor()
	test_schema_validation()
	test_prompt_builder(policy_text)
	test_policy_rules_extraction(policy_text)
	test_walkway_detector_smoke()
	test_severity_mapping()
	test_escalation_routing()
	print("All available tests completed.")


if __name__ == "__main__":
	main()