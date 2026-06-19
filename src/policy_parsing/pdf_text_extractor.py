"""Helpers for extracting plain text from policy PDFs."""

from __future__ import annotations

from pathlib import Path

import pdfplumber


def extract_pdf_text(pdf_path: str | Path) -> str:
	"""Extract all text from a PDF into a single plain string."""

	path = Path(pdf_path)
	chunks: list[str] = []

	with pdfplumber.open(path) as pdf:
		for page in pdf.pages:
			page_text = page.extract_text()
			if page_text:
				chunks.append(page_text)

	return "\n".join(chunks)
