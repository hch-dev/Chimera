""" HOMOGLYPH / CHARACTER LOOKALIKES (PUNYCODE ANALYSIS) """
# Path: Version_1/features/homoglyph.py

from __future__ import annotations
import logging
from typing import Dict, Optional
import idna

logger = logging.getLogger(__name__)


def _to_unicode(domain: str) -> str:
    """
    Convert an ASCII/punycode domain (xn--...) to Unicode.
    If conversion fails, return the original domain.
    """
    try:
        # idna.decode expects one label at a time; idna.decode on whole domain works too
        return idna.decode(domain)
    except Exception:
        try:
            # fallback: decode each label separated by dots
            labels = domain.split(".")
            return ".".join(idna.decode(lbl) if lbl.startswith("xn--") else lbl for lbl in labels)
        except Exception:
            logger.debug("punycode decode failed for %s", domain)
            return domain


def analyze_punycode(host: str) -> Dict[str, Optional[str]]:
    """
    Returns:
      {
        "host_raw": str,
        "is_punycode": bool,
        "punycode": str | None,
        "unicode": str | None,
        "homoglyph_suspected": 0|1
      }
    Best-effort: never raises. Caller can examine `homoglyph_suspected`.
    """
    out: Dict[str, Optional[str]] = {
        "host_raw": host,
        "is_punycode": False,
        "punycode": None,
        "unicode": None,
        "homoglyph_suspected": 0,
    }

    try:
        if not host:
            return out

        # If host already contains 'xn--' then it's punycode
        if "xn--" in host.lower():
            out["is_punycode"] = True
            out["punycode"] = host
            uni = _to_unicode(host)
            out["unicode"] = uni
            # If Unicode differs and contains non-ascii letters, suspect homoglyphs
            if uni and any(ord(c) > 127 for c in uni):
                out["homoglyph_suspected"] = 1
            return out

        # Otherwise, check whether punycode encoding of the unicode form differs
        try:
            # Attempt to encode -> this will raise on invalid characters
            encoded = idna.encode(host).decode("ascii")
            # If encoding produces xn-- anywhere, mark punycode representation
            if "xn--" in encoded:
                out["punycode"] = encoded
                out["unicode"] = host
                out["is_punycode"] = True
                out["homoglyph_suspected"] = 1
            else:
                out["unicode"] = host
                out["is_punycode"] = False
        except Exception:
            # If encoding fails, still provide unicode guess
            out["unicode"] = host

        # Heuristic: if unicode contains non-ascii or rare characters, suspect homoglyph
        if out["unicode"] and any(ord(c) > 127 for c in out["unicode"]):
            out["homoglyph_suspected"] = 1

    except Exception as exc:
        logger.debug("analyze_punycode error: %s", exc)
    return out
