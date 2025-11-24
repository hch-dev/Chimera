""" URL SHORTENERS & REDIRECT TRACKING (HEAD request + redirect chain) """
# Path: Version_1/features/shortener.py

from __future__ import annotations
import logging
from typing import Dict, List, Optional
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Offline-known shortener hostnames (expand as needed)
KNOWN_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "youtu.be", "goo.gl", "ow.ly", "is.gd", "buff.ly", "tiny.cc",
    "rebrand.ly", "rb.gy", "shorturl.at", "lnkd.in", "short.cm"
}

# Default head request timeout
DEFAULT_TIMEOUT = 2.0


def _domain_of(url: str) -> str:
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def check_shortener(url: str, timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Optional[object]]:
    """
    Best-effort check whether `url` is a known shortener and (if so) follows redirects (HEAD).
    Returns a dict:
      {
        "url_raw": str,
        "is_shortener": 0|1,
        "shortener_domain": str | None,
        "redirect_chain": [url1, url2, ...] | [],
        "final_url": str | None,
        "shortener_timed_out": 0|1,
        "error": None | "descr"
      }

    Behavior:
      - If domain matches KNOWN_SHORTENERS, performs a HEAD request with allow_redirects=True.
      - On timeout or request failure: sets shortener_timed_out=1 and error field, but returns best-effort.
    """
    out: Dict[str, Optional[object]] = {
        "url_raw": url,
        "is_shortener": 0,
        "shortener_domain": None,
        "redirect_chain": [],
        "final_url": None,
        "shortener_timed_out": 0,
        "error": None,
    }

    try:
        host = _domain_of(url).lower()
        if host in KNOWN_SHORTENERS:
            out["is_shortener"] = 1
            out["shortener_domain"] = host

            try:
                # Use HEAD first to reduce payload; some services don't respond to HEAD — fall back to GET once
                resp = requests.head(url, allow_redirects=True, timeout=timeout)
                chain = [r.url for r in resp.history] + [resp.url]
                out["redirect_chain"] = chain
                out["final_url"] = resp.url
                return out
            except requests.RequestException as e_head:
                logger.debug("HEAD failed for %s: %s. Trying GET", url, e_head)
                try:
                    resp = requests.get(url, allow_redirects=True, timeout=timeout)
                    chain = [r.url for r in resp.history] + [resp.url]
                    out["redirect_chain"] = chain
                    out["final_url"] = resp.url
                    return out
                except requests.RequestException as e_get:
                    # Timed out or blocked — mark as timed out, but return hint
                    out["shortener_timed_out"] = 1
                    out["error"] = str(e_get)
                    logger.debug("Shortener GET failed for %s: %s", url, e_get)
                    return out
        else:
            # Not a known shortener — no network calls by default
            out["is_shortener"] = 0
            return out
    except Exception as exc:
        out["error"] = str(exc)
        logger.debug("check_shortener error: %s", exc)
        return out
