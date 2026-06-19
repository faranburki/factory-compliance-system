import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure src/ is on the path
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
load_dotenv(_PROJECT_ROOT / ".env")

from policy_parsing.pdf_text_extractor import extract_pdf_text
from policy_parsing.rule_extractor import extract_policy_rules, write_policy_rules_json

def main():
    pdf_path = _PROJECT_ROOT / "Compliance_Policy_Manual.pdf"
    out_json = _PROJECT_ROOT / "outputs" / "policy_rules.json"
    
    print(f"Reading {pdf_path.name}...")
    text = extract_pdf_text(pdf_path)
    
    print("Extracting rules using Groq Vision API...")
    rules = extract_policy_rules(text)
    
    write_policy_rules_json(rules, out_json)
    print(f"Successfully extracted rules and saved to {out_json}")

if __name__ == "__main__":
    main()
