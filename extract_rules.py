"""
Policy Parsing Controller | extract_rules.py
CLI script to trigger the LLM policy-extraction pipeline.
"""

import argparse
from pathlib import Path

from src.config import POLICIES_DIR, POLICY_RULES_JSON, GROQ_TEXT_MODEL
from src.policy_parsing.pdf_text_extractor import extract_pdf_text
from src.policy_parsing.rule_extractor import extract_policy_rules, write_policy_rules_json
from src.policy_parsing.verify_ground_truth import compare_policy_rules


def main():
	"""
	Run the policy rule extraction process and compare against ground truth.
	"""
	parser = argparse.ArgumentParser(description="Extract policy rules from PDF")
	parser.add_argument("--pdf", type=str, default="Compliance_Policy_Manual.pdf", help="PDF filename in data/policies")
	args = parser.parse_args()

	pdf_path = POLICIES_DIR / args.pdf
	if not pdf_path.exists():
		print(f"Error: Policy PDF not found at {pdf_path}")
		return

	print(f"Extracting text from {pdf_path.name}...")
	text = extract_pdf_text(pdf_path)

	print(f"Parsing rules dynamically via Groq ({GROQ_TEXT_MODEL})...")
	try:
		extraction = extract_policy_rules(text, model=GROQ_TEXT_MODEL)
	except RuntimeError as exc:
		print(f"API Error: {exc}")
		return

	print(f"Extracted {len(extraction.rules)} rules.")
	
	write_policy_rules_json(extraction, POLICY_RULES_JSON)
	print(f"Rules saved to {POLICY_RULES_JSON.relative_to(POLICY_RULES_JSON.parent.parent)}")

	print("\nVerifying against Ground Truth...")
	mismatches = compare_policy_rules(extraction)
	if mismatches:
		print(f"Found {len(mismatches)} mismatches:")
		for mm in mismatches:
			print(f"  - {mm}")
	else:
		print("Perfect match with ground truth!")


if __name__ == "__main__":
	main()
