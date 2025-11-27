# Checks if there are any homoglyph characters

from urllib.parse import urlparse
import unicodedata
from log import get_logger

logger = get_logger(__name__)

FEATURE_NAME = "homoglyph_impersonation"
WEIGHT = 0.40  # critical weight in the critical bucket

def _has_non_latin(char):
    try:
        name = unicodedata.name(char)
    except ValueError:
        return True
    # If the Unicode name contains "LATIN" it's likely a normal ascii/latin char
    return "LATIN" not in name

def _punycode_indicator(host: str) -> bool:
    # IDN punycode labels start with "xn--"
    labels = host.split(".")
    return any(lbl.startswith("xn--") for lbl in labels)

def extract(url: str, context: dict = None) -> dict:
    """
    Heuristic homoglyph detector:
    - strong suspicion if domain contains punycode (xn--) labels
    - medium suspicion if domain contains non-Latin Unicode characters or mixed-scripts
    - lower suspicion if domain only ASCII letters/digits
    Returns score 0-100.
    """
    try:
        context = context or {}
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = (parsed.hostname or "").strip()

        if not host:
            return {"feature_name": FEATURE_NAME, "score": 0, "weight": WEIGHT, "error": False,
                    "message": "no-host"}

        # Quick high-signal checks
        if _punycode_indicator(host):
            # punycode strongly indicates IDN usage → high risk
            return {"feature_name": FEATURE_NAME, "score": 95, "weight": WEIGHT, "error": False,
                    "message": "punycode_detected"}

        # Check for non-latin characters and script mixing
        has_non_latin = False
        scripts = set()
        for ch in host:
            if ch == "." or ch == "-":
                continue
            try:
                name = unicodedata.name(ch)
            except ValueError:
                # unnameable codepoint -> treat as suspicious
                has_non_latin = True
                continue
            if "LATIN" not in name:
                has_non_latin = True
                scripts.add(name.split()[0])
            else:
                scripts.add("LATIN")

        if has_non_latin and len(scripts) >= 1:
            # If host contains characters from other scripts or mixed scripts -> suspicious
            score = 80 if len(scripts) > 1 else 65
            return {"feature_name": FEATURE_NAME, "score": score, "weight": WEIGHT, "error": False,
                    "message": f"non_latin_chars scripts={list(scripts)[:4]}"}

        # Very basic homoglyph heuristic: look for characters outside ASCII range
        if any(ord(c) > 127 for c in host):
            return {"feature_name": FEATURE_NAME, "score": 70, "weight": WEIGHT, "error": False,
                    "message": "non_ascii_chars_present"}

        # Otherwise low risk
        return {"feature_name": FEATURE_NAME, "score": 0, "weight": WEIGHT, "error": False,
                "message": "ascii_only"}

    except Exception as e:
        logger.exception("Homoglyph feature failed for URL=%s : %s", url, e)
        return {"feature_name": FEATURE_NAME, "score": None, "weight": WEIGHT, "error": True,
                "message": str(e)}
