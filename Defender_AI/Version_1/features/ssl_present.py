from datetime import datetime
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
            return datetime.strptime(str(notafter), fmt)
        except Exception:
            continue
    # try isoformat fallback
    try:
        return datetime.fromisoformat(str(notafter))
    except Exception:
        return None

def extract(url: str, context: dict = None) -> dict:
    """
    Evaluate SSL presence and basic validity.
    Scoring logic:
      - No cert detected -> 100
      - Cert present but INVALID (Expired/Self-Signed) -> 95
      - Cert present and expires soon (<30 days) -> 30
      - Cert present and valid -> 0
    """
    try:
        context = context or {}
        ssl_info = (context.get("ssl", {}) or {})

        ssl_present = int(bool(ssl_info.get("ssl_present", 0)))

        # Default to True if key is missing to avoid false positives on mock data,
        # but False if explicit in real scan.
        is_valid_cert = ssl_info.get("is_valid_cert", False)
        not_after = ssl_info.get("ssl_valid_to")

        # Case 1: No SSL at all (HTTP)
        if not ssl_present:
            return {"feature_name": FEATURE_NAME, "score": 100, "weight": WEIGHT, "error": False,
                    "message": "no_ssl_present"}

        # Case 2: SSL Present, but INVALID (Expired, Self-Signed, Wrong Host)
        # This catches expired.badssl.com
        if not is_valid_cert:
            return {"feature_name": FEATURE_NAME, "score": 95, "weight": WEIGHT, "error": False,
                    "message": "ssl_invalid_or_expired"}

        # Case 3: SSL Valid -> Check expiration date logic (Safety check)
        dt = _parse_cert_notafter(not_after)
        if dt:
            # Use utcnow (or now(timezone.utc) if available)
            days_left = (dt - datetime.utcnow()).days
            if days_left < 30:
                return {"feature_name": FEATURE_NAME, "score": 30, "weight": WEIGHT, "error": False,
                        "message": f"expires_soon_days={days_left}"}

        # Case 4: Good SSL
        return {"feature_name": FEATURE_NAME, "score": 0, "weight": WEIGHT, "error": False,
                "message": "valid_ssl"}

    except Exception as e:
        logger.exception("SSL feature failed for URL=%s : %s", url, e)
        return {"feature_name": FEATURE_NAME, "score": None, "weight": WEIGHT, "error": True,
                "message": str(e)}
