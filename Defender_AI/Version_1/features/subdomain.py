""" SUBDOMAIN ABUSE & ROOT DOMAIN ISOLATION (tldextract) """
# Path: Version_1/features/subdomain.py

from __future__ import annotations
import logging
from typing import Dict, Optional
import tldextract

logger = logging.getLogger(__name__)

# small brand tokens to detect suspicious root composition (expandable)
BRAND_TOKENS = {"paypal", "amazon", "microsoft", "google", "apple", "facebook", "instagram", "bank", "chase"}


def analyze_subdomain(host: str) -> Dict[str, Optional[object]]:
    """
    Returns:
      {
        "host_raw": str,
        "registered_domain": str,  # e.g., 'paypal.com' or 'secure-login.example.com' -> 'example.com'
        "subdomain": str,          # pre-registered domain prefix
        "root_domain": str,        # registered_domain (domain + suffix)
        "subdomain_depth": int,
        "is_subdomain_abuse": 0|1, # heuristics: brand token in subdomain or odd composition
        "notes": str | None
      }
    """
    out: Dict[str, Optional[object]] = {
        "host_raw": host,
        "registered_domain": None,
        "subdomain": None,
        "root_domain": None,
        "subdomain_depth": 0,
        "is_subdomain_abuse": 0,
        "notes": None,
    }

    try:
        if not host:
            return out
        te = tldextract.extract(host)
        domain = te.domain or ""
        suffix = te.suffix or ""
        sub = te.subdomain or ""

        root_domain = ".".join(part for part in (domain, suffix) if part)
        out["registered_domain"] = root_domain
        out["subdomain"] = sub
        out["root_domain"] = root_domain
        out["subdomain_depth"] = 0 if not sub else sub.count(".") + 1

        # Heuristic: subdomain abuse -> subdomain contains brand token or domain contains hyphen + brand token
        host_lower = host.lower()
        abused = 0
        notes = []

        # Case A: paypal-secure-login.com (typosquatting in second-level domain)
        if "-" in domain and any(bt in domain for bt in BRAND_TOKENS):
            abused = 1
            notes.append("hyphenated domain contains brand token")

        # Case B: brand token inside subdomain (e.g., paypal.secure.example.com)
        if sub and any(bt in sub for bt in BRAND_TOKENS):
            abused = 1
            notes.append("brand token present in subdomain")

        out["is_subdomain_abuse"] = 1 if abused else 0
        out["notes"] = "; ".join(notes) if notes else None
    except Exception as exc:
        logger.debug("analyze_subdomain error: %s", exc)
        out["notes"] = str(exc)
    return out
