"""Document loading and text preprocessing.

Supports .pdf, .docx and .txt input, either as a file path or as a
file-like object (e.g. what Streamlit's file_uploader returns).
"""

from __future__ import annotations

import io
import re
from datetime import date
from pathlib import Path

# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def extract_text(source, filename: str | None = None) -> str:
    """Return the plain text of a PDF, DOCX or TXT document.

    `source` may be a path (str/Path) or a file-like object with .read().
    `filename` is needed for file-like objects so the type can be detected.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        name = path.name.lower()
        data = path.read_bytes()
    else:
        name = (filename or getattr(source, "name", "")).lower()
        data = source.read()

    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        import docx
        document = docx.Document(io.BytesIO(data))
        parts = [p.text for p in document.paragraphs]
        for table in document.tables:  # many resumes use tables for layout
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts)

    # Fallback: treat as text
    return data.decode("utf-8", errors="ignore")


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------

_EMAIL = re.compile(r"\S+@\S+\.\S+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_PHONE = re.compile(r"\+?\d[\d\s\-()]{8,}\d")
_BULLETS = re.compile(r"[•●▪■◦►✓✔·]")


def clean_text(text: str) -> str:
    """Remove contact details / noise and normalise whitespace.

    Case is preserved here; vectorizers lowercase on their own and skill
    extraction is case-insensitive.
    """
    text = _EMAIL.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _PHONE.sub(" ", text)
    text = _BULLETS.sub("\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def split_into_chunks(text: str, min_words: int = 4) -> list[str]:
    """Split a document into sentence/bullet-sized chunks.

    Resumes and JDs are mostly bullet points, so we split on newlines first
    and then on sentence boundaries. Very short fragments (headers like
    "EDUCATION") are dropped because they carry little meaning.
    """
    chunks: list[str] = []
    for line in clean_text(text).split("\n"):
        for sent in re.split(r"(?<=[.!?;])\s+", line):
            sent = sent.strip(" -–*:\t")
            if len(sent.split()) >= min_words:
                chunks.append(sent)
    return chunks


# ----------------------------------------------------------------------------
# Experience extraction (simple rule-based information extraction)
# ----------------------------------------------------------------------------

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

_RANGE = re.compile(
    r"(?:(?P<m1>[A-Za-z]{3})[a-z]*\.?\s+)?(?P<y1>(?:19|20)\d{2})\s*(?:-|–|—|to)\s*"
    r"(?:(?:(?P<m2>[A-Za-z]{3})[a-z]*\.?\s+)?(?P<y2>(?:19|20)\d{2})|(?P<now>present|current|now|till date))",
    re.IGNORECASE,
)


def estimate_years_experience(resume_text: str, today: date | None = None) -> float:
    """Estimate total years of experience from date ranges like
    'Jan 2020 - Present' or '2018 – 2021'. Overlapping ranges are merged.

    Ranges inside an EDUCATION section are ignored.
    """
    today = today or date.today()
    text = resume_text
    # Drop the education section (up to the next all-caps header) if present
    text = re.sub(r"(?s)\b(?:EDUCATION|Education)\b.*?(?=\n[A-Z][A-Z &]{3,}\s*\n|\Z)", " ", text)

    intervals = []
    for m in _RANGE.finditer(text):
        m1 = _MONTHS.get((m.group("m1") or "jan")[:3].lower(), 1)
        start = int(m.group("y1")) + (m1 - 1) / 12
        if m.group("now"):
            end = today.year + (today.month - 1) / 12
        else:
            m2 = _MONTHS.get((m.group("m2") or "dec")[:3].lower(), 12)
            end = int(m.group("y2")) + (m2 - 1) / 12
        if end > start:
            intervals.append((start, end))

    # merge overlaps
    intervals.sort()
    total, cur_s, cur_e = 0.0, None, None
    for s, e in intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return round(total, 1)


_REQ_YEARS = re.compile(r"(\d+)\s*\+?\s*(?:-\s*\d+\s*)?(?:years|yrs)", re.IGNORECASE)


def required_years(jd_text: str) -> int | None:
    """Return the minimum years of experience a JD asks for, if stated."""
    found = [int(x) for x in _REQ_YEARS.findall(jd_text)]
    return min(found) if found else None
