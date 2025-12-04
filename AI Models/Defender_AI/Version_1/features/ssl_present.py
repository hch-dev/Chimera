# features/ssl_present.py

from datetime import datetime
from log import get_logger

logger = get_logger(__name__)

FEATURE_NAME = "ssl_presence_and_validity"
WEIGHT = 0.25

def _parse_cert_notafter(notafter):
    if not notafter: return None
    fmts = ["%b %d %H:%M:%S %Y %Z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"]
    for fmt in fmts:
        try:
            return datetime.strptime(str(notafter), fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(str(notafter))
    except Exception:
        return None

def extract(url: str, context: dict = None) -> dict:
    try:
        context = context or {}
        ssl_info = (context.get("ssl", {}) or {})

        # --- FIX 1: Ignore Score on Fetch Error ---
        # If the context loader failed to reach the site (timeout, connection reset),
        # we cannot penalize it for "No SSL". We must ignore it.
        if ssl_info.get("error"):
            return {
                "feature_name": FEATURE_NAME,
                "score": None, # Returning None tells the engine to skip this
                "weight": WEIGHT,
                "error": True, # Mark as error/skip
                "message": f"ssl_fetch_ignored: {ssl_info.get('error')}"
            }

        ssl_present = int(bool(ssl_info.get("ssl_present", 0)))
        is_valid_cert = ssl_info.get("is_valid_cert", False)
        not_after = ssl_info.get("ssl_valid_to")

        # Case 1: No SSL at all (HTTP)
        if not ssl_present:
            return {"feature_name": FEATURE_NAME, "score": 100, "weight": WEIGHT, "error": False, "message": "no_ssl_present"}

        # Case 2: SSL Present, but INVALID (Expired, Self-Signed)
        if not is_valid_cert:
            return {"feature_name": FEATURE_NAME, "score": 95, "weight": WEIGHT, "error": False, "message": "ssl_invalid_or_expired"}

        # Case 3: SSL Valid -> Check expiration
        dt = _parse_cert_notafter(not_after)
        if dt:
            days_left = (dt - datetime.utcnow()).days
            if days_left < 30:
                return {"feature_name": FEATURE_NAME, "score": 30, "weight": WEIGHT, "error": False, "message": f"expires_soon_days={days_left}"}

        return {"feature_name": FEATURE_NAME, "score": 0, "weight": WEIGHT, "error": False, "message": "valid_ssl"}

    except Exception as e:
        logger.exception("SSL feature failed: %s", e)
        # Return error=True and score=None to ignore
        return {"feature_name": FEATURE_NAME, "score": None, "weight": WEIGHT, "error": True, "message": str(e)}
