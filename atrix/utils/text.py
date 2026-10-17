"""Shared text cleaning and parsing helpers."""
from __future__ import annotations

import re
from typing import Optional

# Full DSML / XML tool-call blocks.
_DSML_INVOKE_RE = re.compile(
    r'<[\|\s｜]*DSML[\|\s｜]*invoke[^>]*>.*?</[\|\s｜]*DSML[\|\s｜]*invoke>',
    re.IGNORECASE | re.DOTALL,
)
_DSML_PARAMETER_RE = re.compile(
    r'<[\|\s｜]*DSML[\|\s｜]*parameter[^>]*>.*?</[\|\s｜]*DSML[\|\s｜]*parameter>',
    re.IGNORECASE | re.DOTALL,
)
_PLAIN_INVOKE_RE = re.compile(r'<invoke[^>]*>.*?</invoke>', re.IGNORECASE | re.DOTALL)
_PLAIN_PARAMETER_RE = re.compile(r'<parameter[^>]*>.*?</parameter>', re.IGNORECASE | re.DOTALL)
_DSML_LONE_OPEN_RE = re.compile(r'<[\|\s｜]*DSML[\|\s｜]*[^>]*>', re.IGNORECASE)
_DSML_LONE_CLOSE_RE = re.compile(r'</[\|\s｜]*DSML[\|\s｜]*[^>]*>', re.IGNORECASE)
_TOOL_CLOSE_RE = re.compile(
    r'</?(?:invoke|parameter|tool_call|function_call|call|calls)[^>]*>',
    re.IGNORECASE,
)
_LONE_TAG_RE = re.compile(r'</?[a-zA-Z0-9_\-:\s]+>')


def clean_model_output(text: Optional[str]) -> str:
    """Strip tool-call residue from a model response. Idempotent."""
    if not text:
        return ""
    cleaned = text
    cleaned = _DSML_INVOKE_RE.sub("", cleaned)
    cleaned = _DSML_PARAMETER_RE.sub("", cleaned)
    cleaned = _PLAIN_INVOKE_RE.sub("", cleaned)
    cleaned = _PLAIN_PARAMETER_RE.sub("", cleaned)
    cleaned = _DSML_LONE_OPEN_RE.sub("", cleaned)
    cleaned = _DSML_LONE_CLOSE_RE.sub("", cleaned)
    cleaned = _TOOL_CLOSE_RE.sub("", cleaned)
    cleaned = _LONE_TAG_RE.sub("", cleaned)
    return cleaned.strip()


def looks_like_xml_garbage(text: str) -> bool:
    """Cheap detector — returns True if *text* is tool-call residue."""
    if not text:
        return True
    upper = text.upper()
    if "DSML" in upper:
        return True
    lowered = text.lower()
    return any(tag in lowered for tag in (
        "<invoke", "</invoke>", "<parameter", "</parameter>",
        "tool_call", "<call", "</call",
    ))


def strip_url(text: str) -> str:
    """Remove the first http(s) URL from *text* and return what remains."""
    return re.sub(r'https?://\S+', '', text).strip()