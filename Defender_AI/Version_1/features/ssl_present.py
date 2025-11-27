# Version_1/features/ssl/ssl_present.py
from datetime import datetime
from urllib.parse import urlparse
from log import get_logger

logger = get_logger(__name__)

FEATURE_NAME = "ssl_presence_and_validity"
WEIGHT = 0.25  # critical weight in the critical bucket

def _parse_cert_notafter(notafter):
    """
    Try to parse typical OpenSSL 'notAfter' format like:
      'Jun  1 12:00:00 2025 GMT'
    Returns datetime or None.
    """
    if not notafter:
        return None
    fmts = [
        "%b %d %H:%M:%S %Y %Z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S"
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(notafter, fmt)
        except Exception:
            continue
    # try isoformat fallback
    try:
        return datetime.fromisoformat(notafter)
    except Exception:
        return None

def extract(url: str, context: dict = None) -> dict:
    """
    Evaluate SSL presence and basic validity.
    Scoring logic (risk score = higher when SSL absent/expired):
      - No cert detected -> 100
      - Cert present but expired -> 90
      - Cert present and expires soon (<7 days) -> 70
      - Cert present and long-valid -> 0
    Returns 0-100 risk score (higher => worse).
    """
    try:
        context = context or {}
        ssl_info = (context.get("ssl", {}) or {})  # orchestrator should populate ssl dict
        ssl_present = int(bool(ssl_info.get("ssl_present", 0)))
        not_after = ssl_info.get("ssl_valid_to") or ssl_info.get("ssl_valid_to")
        if not ssl_present:
            return {"feature_name": FEATURE_NAME, "score": 100, "weight": WEIGHT, "error": False,
                    "message": "no_ssl_present"}

        # If cert present, try to parse expiration
        dt = _parse_cert_notafter(not_after)
        if not dt:
            # cannot parse notAfter -> treat as moderate concern
            return {"feature_name": FEATURE_NAME, "score": 40, "weight": WEIGHT, "error": False,
                    "message": "ssl_present_unparsed_expiry"}

        days_left = (dt - datetime.utcnow()).days
        if days_left < 0:
            score = 90
        elif days_left < 7:
            score = 70
        elif days_left < 30:
            score = 30
        else:
            score = 0

        return {"feature_name": FEATURE_NAME, "score": score, "weight": WEIGHT, "error": False,
                "message": f"days_left={days_left}"}

    except Exception as e:
        logger.exception("SSL presence feature failed for URL=%s : %s", url, e)
        return {"feature_name": FEATURE_NAME, "score": None, "weight": WEIGHT, "error": True,
                "message": str(e)}
