#Checks if there is '@' charcter in url or raw IP Address

from urllib.parse import urlparse
from log import get_logger

logger = get_logger(__name__)

FEATURE_NAME = "malicious_structure"
WEIGHT = 0.50

def extract(url: str, context: dict = None) -> dict:
    try:
        if "://" not in url:
            url = "http://" + url

        parsed = urlparse(url)

        # --- CHECK 1: Authority Abuse ('@' in the domain part) ---
        # Good: medium.com/@user (The @ is in parsed.path)
        # Bad:  user:pass@evil.com (The @ is in parsed.netloc)

        if "@" in parsed.netloc:
            user_info = parsed.netloc.split("@")[0]

            # Check for brand keywords in the fake username part
            suspicious_keywords = ["google", "drive", "dropbox", "secure", "login", "verify", "account", "bank"]

            if any(kw in user_info.lower() for kw in suspicious_keywords):
                 return {
                    "feature_name": FEATURE_NAME,
                    "score": 100,
                    "weight": WEIGHT,
                    "error": False,
                    "message": f"CRITICAL: Authority Abuse detected. Fake credentials '{user_info}' used before '@'."
                }

            return {
                "feature_name": FEATURE_NAME,
                "score": 90,
                "weight": WEIGHT,
                "error": False,
                "message": "suspicious_at_symbol_in_authority"
            }

        # --- CHECK 2: Raw IP Address Hostname ---
        hostname = parsed.hostname
        if hostname:
            # Exclude generic names, check if it looks like an IP (4 parts, all digits)
            parts = hostname.split('.')
            if len(parts) == 4 and all(part.isdigit() for part in parts):
                # Whitelist common DNS resolvers just in case (optional)
                if hostname not in ['1.1.1.1', '8.8.8.8', '8.8.4.4']:
                    return {
                        "feature_name": FEATURE_NAME,
                        "score": 85,
                        "weight": WEIGHT,
                        "error": False,
                        "message": "raw_ip_address_used"
                    }

        return {
            "feature_name": FEATURE_NAME,
            "score": 0,
            "weight": WEIGHT,
            "error": False,
            "message": "structure_normal"
        }

    except Exception as e:
        logger.exception(f"Structure check failed: {e}")
        return {"feature_name": FEATURE_NAME, "score": None, "weight": WEIGHT, "error": True, "message": str(e)}
