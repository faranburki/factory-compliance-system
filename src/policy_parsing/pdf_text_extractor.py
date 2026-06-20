"""
Module 1 — Policy Parsing | pdf_text_extractor.py
Extracts plain text from policy PDFs to be processed by the rule extraction pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber


def extract_pdf_text(pdf_path: str | Path) -> str:
	"""
	Extract raw text from a PDF file using pdfplumber.

	Args:
		pdf_path: Absolute or relative path to the PDF file.

	Returns:
		Full extracted text as a single string with page breaks preserved
		as newline characters.

	Raises:
		FileNotFoundError: If the PDF does not exist at the given path.
	"""
	path = Path(pdf_path)
	chunks: list[str] = []

	with pdfplumber.open(path) as pdf:
		for page in pdf.pages:
			page_text = page.extract_text()
			# Ignore blank pages that yield no text
			if page_text:
				chunks.append(page_text)

	return "\n".join(chunks)
