from __future__ import annotations

import io
import requests
from typing import Optional

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


def extract_pdf_text(url: str, timeout: int = 15) -> str:
    """
    Download a PDF from the given URL and extract its text.

    Returns an empty string if anything fails.
    """
    if not url:
        return ""

    if PdfReader is None:
        print("[pdf_utils] pypdf not installed; cannot extract PDF text")
        return ""

    try:
        print(f"[pdf_utils] Downloading PDF: {url}")
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        print(f"[pdf_utils] Error downloading {url}: {e}")
        return ""

    try:
        reader = PdfReader(io.BytesIO(resp.content))
        pages_text = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                pages_text.append(txt)
        full_text = "\n\n".join(pages_text).strip()
        print(f"[pdf_utils] Extracted {len(full_text)} chars from {url}")
        return full_text
    except Exception as e:
        print(f"[pdf_utils] Error parsing PDF {url}: {e}")
        return ""
