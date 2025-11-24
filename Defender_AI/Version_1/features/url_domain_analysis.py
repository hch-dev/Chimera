""" URL DOMAIN ANALYSIS AGGREGATOR (calls homoglyph, shortener, subdomain) """
# Path: Version_1/features/url_domain_analysis.py

from __future__ import annotations
import logging
from typing import Dict, Optional
from urllib.parse import urlparse, urlunparse

from Version_1.features.homoglyph import analyze_punycode
from Version_1.features.shortener import check_shortener, DEFAULT_TIMEOUT
from Version_1.features.subdomain import analyze_subdomain

logger = logging.getLogger(__name__)


def _refang(url: str) -> str:
    """Convert common defanged forms to parseable URL (best-effort)."""
    if not url:
        return url
    s = url.replace("[.]", ".").replace("(.)", ".").replace("[dot]", ".")
    s = s.replace("hxxp://", "http://").replace("hxxps://", "https://")
    return s.strip()


def _safe_parse(url: str) -> Dict[str, Optional[str]]:
    """
    Best-effort parse using urllib.parse. Returns dict with scheme, host, path, netloc.
    """
    out = {"raw": url, "scheme": None, "host": None, "path": None, "netloc": None}
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        out["scheme"] = parsed.scheme
        out["host"] = parsed.hostname
        out["path"] = parsed.path
        out["netloc"] = parsed.netloc
    except Exception as exc:
        logger.debug("parse error for %s: %s", url, exc)
        out["parse_error"] = str(exc)
    return out


def extract(url: str, shortener_timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Optional[object]]:
    """
    High-level extractor combining:
      - refang & parse
      - homoglyph/punycode analysis
      - known shortener detection (+ lazy redirect tracing)
      - subdomain/root extraction

    Returns a merged dict (best-effort). Never raises.
    """
    result: Dict[str, Optional[object]] = {"url_raw": url, "error": None}

    try:
        # 1) refang -> parse
        url_refanged = _refang(url)
        result["url_refanged"] = url_refanged
        parsed = _safe_parse(url_refanged)
        result.update(parsed)

        host = parsed.get("host") or ""
        # 2) punycode / homoglyph detection
        try:
            puny = analyze_punycode(host)
            result.update(puny)
        except Exception as e:
            logger.debug("punycode check failed: %s", e)
            result["punycode_error"] = str(e)

        # 3) shortener check (only triggers network calls if known shortener)
        try:
            short = check_shortener(url_refanged, timeout=shortener_timeout)
            result.update(short)
        except Exception as e:
            logger.debug("shortener check failed: %s", e)
            result["shortener_error"] = str(e)

        # 4) subdomain/root domain analysis
        try:
            sub = analyze_subdomain(host)
            result.update(sub)
        except Exception as e:
            logger.debug("subdomain analysis failed: %s", e)
            result["subdomain_error"] = str(e)

    except Exception as exc:
        logger.debug("extract overall error: %s", exc)
        result["error"] = str(exc)

    return result
