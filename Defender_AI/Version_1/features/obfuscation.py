#Checks for encoded strings and various tyoes of encoding

import re
import base64
import binascii
from urllib.parse import unquote
from log import get_logger

logger = get_logger(__name__)

FEATURE_NAME = "obfuscation_analysis"
WEIGHT = 0.40

# Keywords that are suspicious if found hidden inside encoding
SENSITIVE_KEYWORDS = {
    "login", "signin", "verify", "account", "secure",
    "update", "password", "credential", "paypal",
    "google", "microsoft", "bank", "alert", "confirm"
}

def try_base64_decode(s: str) -> str:
    """
    Attempts to decode a string as Base64.
    Returns the decoded string if successful and readable, else None.
    """
    # Basic heuristic: Base64 usually isn't very short
    if len(s) < 8:
        return None

    # Fix padding if necessary
    padding = 4 - (len(s) % 4)
    if padding != 4:
        s += "=" * padding

    try:
        # Attempt decode
        decoded_bytes = base64.b64decode(s, validate=True)
        decoded_str = decoded_bytes.decode('utf-8', errors='ignore')

        # Validation: Is the result actually readable text?
        # If > 80% of chars are printable, it's likely text/code.
        # If it's binary garbage, ignore it.
        printable = sum(1 for c in decoded_str if c.isprintable())
        if len(decoded_str) > 0 and (printable / len(decoded_str) > 0.8):
            return decoded_str.lower()

    except (binascii.Error, UnicodeDecodeError):
        pass

    return None

def extract(url: str, context: dict = None) -> dict:
    """
    Comprehensive Obfuscation Scanner.
    1. Decodes Percent-Encoding (%61 -> a) and checks for hidden keywords.
    2. Hunts for Base64 strings, decodes them, and checks for hidden URLs (Redirects) or keywords.
    """
    try:
        # ---------------------------------------------------------
        # 1. Percent / Hex Decoding Analysis
        # ---------------------------------------------------------
        # e.g. "http://site.com/%6c%6f%67%69%6e" -> "login"
        decoded_url = unquote(url).lower()
        raw_url_lower = url.lower()

        found_hex_keywords = []
        for kw in SENSITIVE_KEYWORDS:
            # If keyword exists in DECODED version but NOT in RAW version, it was hidden.
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
        # 2. Base64 Analysis (Hidden Redirects & Payloads)
        # ---------------------------------------------------------
        # Regex to find potential Base64 strings (alphanumeric + '+' '/' '=')
        # We look for chunks of length 10+ to avoid false positives on short IDs
        potential_b64_strings = re.findall(r'[a-zA-Z0-9+/=]{10,}', url)

        for b64 in potential_b64_strings:
            decoded = try_base64_decode(b64)

            if decoded:
                # A. Check for Hidden Redirects (URL inside URL)
                # e.g. site.com?ref=aHR0cDovL2V2aWwuY29t (http://evil.com)
                if "http://" in decoded or "https://" in decoded or "www." in decoded:
                    return {
                        "feature_name": FEATURE_NAME,
                        "score": 90,  # High risk: Hiding a destination URL
                        "weight": WEIGHT,
                        "error": False,
                        "message": f"hidden_redirect_in_base64: {decoded[:50]}..."
                    }

                # B. Check for Hidden Keywords
                hidden_b64_kws = [kw for kw in SENSITIVE_KEYWORDS if kw in decoded]
                if hidden_b64_kws:
                    return {
                        "feature_name": FEATURE_NAME,
                        "score": 85,
                        "weight": WEIGHT,
                        "error": False,
                        "message": f"hidden_keywords_in_base64: {hidden_b64_kws}"
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
