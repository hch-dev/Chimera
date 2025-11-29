# Checks if there are any homoglyph characters

from urllib.parse import urlparse
import unicodedata
from log import get_logger

logger = get_logger(__name__)

FEATURE_NAME = "homoglyph_impersonation"
WEIGHT = 0.40

def _punycode_indicator(host: str) -> bool:
    labels = host.split(".")
    return any(lbl.startswith("xn--") for lbl in labels)

def extract(url: str, context: dict = None) -> dict:
    try:
        context = context or {}
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = (parsed.hostname or "").strip()

        if not host:
            return {"feature_name": FEATURE_NAME, "score": 0, "weight": WEIGHT, "error": False, "message": "no-host"}

        # 1. Punycode Check
        if _punycode_indicator(host):
            return {"feature_name": FEATURE_NAME, "score": 95, "weight": WEIGHT, "error": False, "message": "punycode_detected"}

        # 2. Script Mixing Check
        has_non_latin = False
        scripts = set()

        for ch in host:
            # IGNORE standard ASCII characters (Letters, Digits, Hyphen, Dot)
            if ch.isascii():
                scripts.add("LATIN") # Group digits/ascii as Latin for simplicity
                continue

            # If we get here, it is a SPECIAL character (Cyrillic, Greek, Emoji)
            try:
                name = unicodedata.name(ch)
                has_non_latin = True
                scripts.add(name.split()[0])
            except ValueError:
                has_non_latin = True
                continue

        # If we found real non-ascii stuff
        if has_non_latin:
            # If we have mixed scripts (e.g. LATIN + CYRILLIC) -> High Risk
            if len(scripts) > 1:
                 return {
                    "feature_name": FEATURE_NAME,
                    "score": 80,
                    "weight": WEIGHT,
                    "error": False,
                    "message": f"mixed_scripts_detected {list(scripts)}"
                }
            # If just non-latin (e.g. pure Chinese domain) -> Low Risk (valid IDN)
            else:
                return {
                    "feature_name": FEATURE_NAME,
                    "score": 10, # Low score for pure foreign domains
                    "weight": WEIGHT,
                    "error": False,
                    "message": "non_latin_domain"
                }

        return {"feature_name": FEATURE_NAME, "score": 0, "weight": WEIGHT, "error": False, "message": "ascii_only"}

    except Exception as e:
        logger.exception("Homoglyph feature failed: %s", e)
        return {"feature_name": FEATURE_NAME, "score": None, "weight": WEIGHT, "error": True, "message": str(e)}
