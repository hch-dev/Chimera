#Checks for "data:" links

import re
from log import get_logger

logger = get_logger(__name__)

FEATURE_NAME = "data_uri_scheme"
WEIGHT = 0.50  # Extremely High Weight (Protocol Abuse)

def extract(url: str, context: dict = None) -> dict:
    """
    Detects usage of 'data:' URI schemes.
    Attackers use these to embed the phishing page code directly in the URL,
    bypassing the need for a hosting server.

    Format: data:[<mediatype>][;base64],<data>
    """
    try:
        clean_url = url.lower().strip()

        # --- CHECK 1: Is it a Data URI? ---
        if clean_url.startswith("data:"):

            # --- CHECK 2: Analyze MIME Type ---
            # Extract the part between 'data:' and the first ';' or ','
            # e.g., "data:text/html;base64,..." -> "text/html"
            match = re.match(r'data:([^;,]+)', clean_url)
            mime_type = match.group(1) if match else "unknown"

            # Dangerous types used to render pages or execute scripts
            high_risk_mimes = {
                'text/html', 'application/xhtml+xml',
                'application/javascript', 'text/javascript',
                'application/x-javascript', 'image/svg+xml'
            }

            if any(risk in mime_type for risk in high_risk_mimes):
                 return {
                    "feature_name": FEATURE_NAME,
                    "score": 100,
                    "weight": WEIGHT,
                    "error": False,
                    "message": f"CRITICAL: Executable Data URI detected (MIME: {mime_type})"
                }

            # Even if it's just an image (data:image/png), finding this as a
            # PRIMARY navigation URL (not an embedded resource) is highly suspicious.
            return {
                "feature_name": FEATURE_NAME,
                "score": 90,
                "weight": WEIGHT,
                "error": False,
                "message": f"suspicious_data_uri_navigation ({mime_type})"
            }

        # --- CHECK 3: Blob URI (Often used with data URIs) ---
        if clean_url.startswith("blob:"):
            return {
                "feature_name": FEATURE_NAME,
                "score": 95,
                "weight": WEIGHT,
                "error": False,
                "message": "suspicious_blob_uri"
            }

        return {
            "feature_name": FEATURE_NAME,
            "score": 0,
            "weight": WEIGHT,
            "error": False,
            "message": "standard_scheme"
        }

    except Exception as e:
        logger.exception(f"Data URI check failed: {e}")
        return {"feature_name": FEATURE_NAME, "score": None, "weight": WEIGHT, "error": True, "message": str(e)}
