#Checks for encoded strings and various tyoes of encoding

import re
import base64
import binascii
from urllib.parse import unquote, urlparse, parse_qs
from log import get_logger

logger = get_logger(__name__)

FEATURE_NAME = "obfuscation_analysis"
WEIGHT = 0.40

SENSITIVE_KEYWORDS = {
    "login", "signin", "verify", "account", "secure",
    "update", "password", "credential", "paypal",
    "google", "microsoft", "bank", "alert", "confirm",
    "drive", "doc", "sheet"
}

def try_base64_decode(s: str) -> str:
    """
    Attempts to decode a string as Base64 (Standard or URL-Safe).
    """
    if len(s) < 6: return None

    # Fix padding
    padding = 4 - (len(s) % 4)
    if padding != 4:
        s += "=" * padding

    try:
        # 1. Try URL-Safe decode first (handles - and _)
        # This is common in URL parameters
        decoded_bytes = base64.urlsafe_b64decode(s)
        decoded_str = decoded_bytes.decode('utf-8', errors='ignore')

        # Check readability
        printable = sum(1 for c in decoded_str if c.isprintable())
        if len(decoded_str) > 3 and (printable / len(decoded_str) > 0.75):
            return decoded_str.lower()

    except (binascii.Error, ValueError):
        # 2. Fallback: Standard Base64 (handles + and /)
        # Some encoders might use standard b64 in params even if unsafe
        try:
            # Standard b64 doesn't like - or _, so we swap them if present
            s_std = s.replace('-', '+').replace('_', '/')
            decoded_bytes = base64.b64decode(s_std, validate=False)
            decoded_str = decoded_bytes.decode('utf-8', errors='ignore')

            printable = sum(1 for c in decoded_str if c.isprintable())
            if len(decoded_str) > 3 and (printable / len(decoded_str) > 0.75):
                return decoded_str.lower()
        except Exception:
            pass

    return None

def extract(url: str, context: dict = None) -> dict:
    try:
        # ---------------------------------------------------------
        # 1. Percent Decoding Check (Hex)
        # ---------------------------------------------------------
        decoded_url = unquote(url).lower()
        raw_url_lower = url.lower()

        found_hex_keywords = []
        for kw in SENSITIVE_KEYWORDS:
            if kw in decoded_url and kw not in raw_url_lower:
                found_hex_keywords.append(kw)

        if found_hex_keywords:
             return {
                "feature_name": FEATURE_NAME,
                "score": 80,
                "weight": WEIGHT,
                "error": False,
                "message": f"hidden_keywords_in_hex: {found_hex_keywords}"
            }

        # ---------------------------------------------------------
        # 2. Smart Base64 Extraction
        # ---------------------------------------------------------
        candidates = set()

        # Method A: URL Params (parse_qs handles encoded values automatically)
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        for key, values in params.items():
            for val in values:
                candidates.add(val)

        # Method B: Path Regex (CRITICAL FIX)
        # We look for alphanumeric chars plus -, _, +, =
        # We intentionally EXCLUDE '/' to ensure we grab tokens BETWEEN slashes.
        # e.g. /v2/dXBkYXRlX3Bhc3N3b3Jk/config -> grabs "dXBkYXRlX3Bhc3N3b3Jk"
        path_candidates = re.findall(r'[a-zA-Z0-9+\-_=]{10,}', url)
        candidates.update(path_candidates)

        # ---------------------------------------------------------
        # 3. Analyze Candidates
        # ---------------------------------------------------------
        for candidate in candidates:
            # Strip any accidental trailing equals signs that might be doubled
            candidate = candidate.rstrip('=')

            decoded = try_base64_decode(candidate)

            if decoded:
                # Check A: Hidden URL (Redirect Evasion)
                if any(x in decoded for x in ["http:", "https:", "www.", ".com", ".net", ".org"]):
                     return {
                        "feature_name": FEATURE_NAME,
                        "score": 90,
                        "weight": WEIGHT,
                        "error": False,
                        "message": f"hidden_redirect_in_base64: {decoded[:30]}..."
                    }

                # Check B: Hidden Keywords
                hidden_kws = [kw for kw in SENSITIVE_KEYWORDS if kw in decoded]
                if hidden_kws:
                    return {
                        "feature_name": FEATURE_NAME,
                        "score": 85,
                        "weight": WEIGHT,
                        "error": False,
                        "message": f"hidden_keywords_in_base64: {hidden_kws}"
                    }

        return {
            "feature_name": FEATURE_NAME,
            "score": 0,
            "weight": WEIGHT,
            "error": False,
            "message": "no_obfuscation_detected"
        }

    except Exception as e:
        logger.exception(f"Obfuscation check failed: {e}")
        return {"feature_name": FEATURE_NAME, "score": None, "weight": WEIGHT, "error": True, "message": str(e)}
