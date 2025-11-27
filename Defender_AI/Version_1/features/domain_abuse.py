#Checks for domain abuse like paypal-secure-login.com

import tldextract
from log import get_logger

logger = get_logger(__name__)

FEATURE_NAME = "domain_abuse_detection"
WEIGHT = 0.45

# 1. The Brands we want to protect (The "Victims")
# In a real app, this would be a much larger database
TARGET_BRANDS = {
    "paypal", "google", "microsoft", "apple", "amazon",
    "netflix", "chase", "wellsfargo", "facebook", "instagram",
    "linkedin", "dropbox", "adobe", "docusign"
}

# 2. The Keywords attackers use to look official
SENSITIVE_KEYWORDS = {
    "login", "signin", "secure", "verify", "update",
    "account", "service", "support", "auth", "billing",
    "confirm", "wallet", "safe", "bank"
}

def extract(url: str, context: dict = None) -> dict:
    """
    Detects 'Combosquatting' and Subdomain Abuse.
    Example: paypal-secure.com or paypal.com.verify-me.net
    """
    try:
        # tldextract splits: 'www.google.co.uk' -> sub='www', domain='google', suffix='co.uk'
        ext = tldextract.extract(url)
        subdomain = ext.subdomain.lower()
        root_domain = ext.domain.lower()
        suffix = ext.suffix.lower()

        full_domain = f"{root_domain}.{suffix}"

        # ----------------------------------------------
        # CHECK 1: Hyphenated Domain Abuse (Combosquatting)
        # e.g. "paypal-secure-login.com"
        # ----------------------------------------------
        for brand in TARGET_BRANDS:
            # Check if brand is IN the root domain, but isn't the EXACT root domain
            # e.g. "paypal" is in "paypal-secure", but "paypal-secure" != "paypal"
            if brand in root_domain and root_domain != brand:

                # If it also contains a sensitive keyword or a hyphen, risk is HIGH
                if "-" in root_domain or any(kw in root_domain for kw in SENSITIVE_KEYWORDS):
                    return {
                        "feature_name": FEATURE_NAME,
                        "score": 95,
                        "weight": WEIGHT,
                        "error": False,
                        "message": f"combosquatting_detected: {brand} found in {root_domain}"
                    }

        # ----------------------------------------------
        # CHECK 2: Subdomain Abuse (Shadowing)
        # e.g. "paypal.com.security-check.net"
        # ----------------------------------------------
        # Reconstruct the look of a domain inside the subdomain
        # We look for patterns like "brand.com" appearing inside the subdomain part
        for brand in TARGET_BRANDS:
            if brand in subdomain:
                # If "paypal" is in the subdomain, but the main domain is NOT paypal
                if root_domain != brand:
                    return {
                        "feature_name": FEATURE_NAME,
                        "score": 85,
                        "weight": WEIGHT,
                        "error": False,
                        "message": f"subdomain_impersonation: {brand} found in subdomain of {root_domain}"
                    }

        return {
            "feature_name": FEATURE_NAME,
            "score": 0,
            "weight": WEIGHT,
            "error": False,
            "message": "brand_integrity_verified"
        }

    except Exception as e:
        logger.exception(f"Domain abuse check failed: {e}")
        return {"feature_name": FEATURE_NAME, "score": None, "weight": WEIGHT, "error": True, "message": str(e)}
