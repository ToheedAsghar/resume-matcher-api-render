import io
import re
from typing import Optional, Tuple

import requests
from langdetect import detect
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document


EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}")


def _extract_text_from_docx_bytes(data: bytes) -> str:
    stream = io.BytesIO(data)
    doc = Document(stream)
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_text_from_pdf_bytes(data: bytes) -> str:
    stream = io.BytesIO(data)
    return pdf_extract_text(stream)


def extract_text_from_url(file_url: str) -> Tuple[str, Optional[str]]:
    resp = requests.get(file_url, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "").lower()
    data = resp.content

    if "application/pdf" in content_type or file_url.lower().endswith(".pdf"):
        return _extract_text_from_pdf_bytes(data), "pdf"
    if (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in content_type
        or file_url.lower().endswith(".docx")
    ):
        return _extract_text_from_docx_bytes(data), "docx"
    # Fallback: treat as text
    try:
        return data.decode("utf-8", errors="ignore"), "text"
    except Exception:
        return "", None


def extract_contacts(text: str) -> Tuple[list, list]:
    emails = EMAIL_REGEX.findall(text or "")
    phones = PHONE_REGEX.findall(text or "")
    # Deduplicate and simple cleanup
    return sorted(set(emails)), sorted(set(p.strip() for p in phones if len(p.strip()) >= 7))


def detect_language(text: str) -> Optional[str]:
    try:
        if text and text.strip():
            return detect(text)
    except Exception:
        return None
    return None


COMMON_SKILLS = {
    # Programming
    "python", "java", "javascript", "typescript", "node.js", "node", "c++", "c#", "go", "rust",
    # Data/ML
    "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "pytorch", "llms", "large language models", "gpt", "nlp",
    # Cloud/DevOps
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform",
    # Web/Backend
    "rest", "graphql", "sql", "nosql", "postgres", "mysql", "mongodb",
}


def extract_skills(text: str) -> list:
    text_lower = (text or "").lower()
    found = set()
    for skill in COMMON_SKILLS:
        if skill in text_lower:
            found.add(skill)
    return sorted(found)


def parse_resume_text(text: str) -> dict:
    emails, phones = extract_contacts(text)
    skills = extract_skills(text)
    language = detect_language(text)
    # Very light heuristics; name/location are non-trivial without ML, leave None
    return {
        "emails": emails,
        "phones": phones,
        "skills": skills,
        "language": language,
        "raw_text": text,
    }


