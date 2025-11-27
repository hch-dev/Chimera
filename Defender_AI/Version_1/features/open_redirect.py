# Version_1/features/redirects/open_redirect.py
from urllib.parse import urlparse
from log import get_logger

logger = get_logger(__name__)

FEATURE_NAME = "open_redirect_detection"
WEIGHT = 0.35  # critical weight in the critical bucket


def _host_of(u):
    try:
        return urlparse(u if "://" in u else "http://" + u).hostname or ""
    except Exception:
        return ""


def extract(url: str, context: dict = None) -> dict:
    """
    Detect open-redirect / suspicious redirect chains.
    Heuristics:
      - If redirect chain length > 1 and hosts change -> suspicious
      - If chain length >= 4 -> more suspicious
      - If redirect leads to an IP host or to a different TLD + login words -> raise score
    Returns score 0-100.
    """
    try:
        context = context or {}
        http = context.get("http", {}) or {}
        chain = http.get("redirect_chain", []) or []

        if not chain or len(chain) <= 1:
            return {
                "feature_name": FEATURE_NAME,
                "score": 0,
                "weight": WEIGHT,
                "error": False,
                "message": "no_redirects"
            }

        # Normalize host list
        hosts = [_host_of(h).lower() for h in chain if _host_of(h)]
        if not hosts:
            return {
                "feature_name": FEATURE_NAME,
                "score": 0,
                "weight": WEIGHT,
                "error": False,
                "message": "no_hosts_in_chain"
            }

        orig = hosts[0]
        final = hosts[-1]

        diff_hosts = orig != final

        # FIXED: removed invalid inline if-else inside generator expression
        external_count = sum(
            1 for h in hosts
            if orig and h and not h.endswith(orig.split(".", 1)[-1])
        )

        # Final is IP → suspicious
        final_is_ip = all(part.isdigit() for part in final.split(".")) if final else False

        score = 0

        if diff_hosts:
            score += 50
        if len(chain) >= 3:
            score += 20
        if external_count > 0:
            score += min(20, external_count * 10)
        if final_is_ip:
            score += 15

        suspicious_tokens = ("login", "secure", "verify", "signin", "account")
        token_present = any(
            any(tok in part.lower() for tok in suspicious_tokens)
            for part in chain
        )
        if token_present:
            score += 15

        score = max(0, min(100, score))

        msg = (
            f"len_chain={len(chain)} diff_hosts={diff_hosts} "
            f"external_count={external_count} final_ip={final_is_ip} token={token_present}"
        )

        return {
            "feature_name": FEATURE_NAME,
            "score": score,
            "weight": WEIGHT,
            "error": False,
            "message": msg
        }

    except Exception as e:
        logger.exception("Open redirect feature failed for URL=%s : %s", url, e)
        return {
            "feature_name": FEATURE_NAME,
            "score": None,
            "weight": WEIGHT,
            "error": True,
            "message": str(e)
        }
