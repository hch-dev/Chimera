#Checks the number of redirects in a link

from urllib.parse import urlparse
import tldextract
from log import get_logger

logger = get_logger(__name__)

FEATURE_NAME = "open_redirect_detection"
WEIGHT = 0.35

def _get_root_domain(url):
    """
    Extracts 'google.com' from 'www.google.com' or 'mail.google.co.uk'
    """
    try:
        ext = tldextract.extract(url)
        if not ext.domain:
            return ""
        return f"{ext.domain}.{ext.suffix}".lower()
    except:
        return ""

def _host_of(u):
    try:
        return urlparse(u if "://" in u else "http://" + u).hostname or ""
    except Exception:
        return ""

def extract(url: str, context: dict = None) -> dict:
    """
    Detect open-redirect / suspicious redirect chains.
    Improved Heuristics:
      - Ignore redirects within the same root domain (google.com -> www.google.com is SAFE).
      - Ignore HTTP -> HTTPS upgrades.
      - Flag only if redirect jumps to a completely different external domain.
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

        # Get the start and end of the chain
        original_url = chain[0]
        final_url = chain[-1]

        orig_root = _get_root_domain(original_url)
        final_root = _get_root_domain(final_url)

        # 1. Same Root Domain Check (The Fix)
        # If we start at google.com and end at www.google.co.in, it's technically the same entity.
        # Checking if the 'domain' part matches handles this (google == google).
        orig_ext = tldextract.extract(original_url)
        final_ext = tldextract.extract(final_url)

        same_entity = (orig_ext.domain == final_ext.domain)

        if same_entity:
             return {
                "feature_name": FEATURE_NAME,
                "score": 0,
                "weight": WEIGHT,
                "error": False,
                "message": f"internal_redirect_safe ({orig_root} -> {final_root})"
            }

        # 2. External Redirect Analysis
        score = 0

        # If we jumped to a totally different domain
        if not same_entity:
            score += 50

            # Check for known safe redirectors (optional whitelist)
            safe_redirectors = {'bit.ly', 'goo.gl', 't.co', 'youtu.be', 'linkedin.com'}
            if orig_root in safe_redirectors:
                score -= 30 # Lower risk for known shorteners (they are meant to redirect)

        # Chain length check
        if len(chain) >= 4:
            score += 20  # Long chains are suspicious regardless of domain

        # Final destination is an IP address? (High Risk)
        final_host = _host_of(final_url)
        final_is_ip = all(part.isdigit() for part in final_host.split(".")) if final_host else False
        if final_is_ip:
            score += 30

        # Suspicious keywords in the chain
        suspicious_tokens = ("login", "secure", "verify", "signin", "account", "update")
        token_present = any(
            any(tok in u.lower() for tok in suspicious_tokens)
            for u in chain
        )
        if token_present and not same_entity:
            score += 20

        score = max(0, min(100, score))

        msg = (
            f"len_chain={len(chain)} same_entity={same_entity} "
            f"orig={orig_root} final={final_root} final_ip={final_is_ip}"
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
