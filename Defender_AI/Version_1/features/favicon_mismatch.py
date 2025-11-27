from urllib.parse import urlparse
import tldextract
from log import get_logger

logger = get_logger(__name__)

FEATURE_NAME = "favicon_mismatch"
WEIGHT = 0.45

KNOWN_FAVICONS = {
    "f3418a443f7d80105cc1804616b5d835": "google",
    "6e7458646c1888546f0393350096793d": "microsoft",
    "33e5e8d308e8e6856b2147053073a65e": "paypal",
    "04e7233b33d2a0965583c33325b42064": "facebook",
    "70320d122a4f3308f874b1d39811108d": "apple",
    "5f39224567661e2452a12490267082cf": "amazon",
    "23134b1828507a950b92832884434026": "chase",
}

def _get_domain_root(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}".lower()

def extract(url: str, context: dict = None) -> dict:
    try:
        context = context or {}
        favicon_info = context.get("favicon", {})

        if not favicon_info or "error" in favicon_info:
             return {"feature_name": FEATURE_NAME, "score": 0, "weight": WEIGHT, "error": False, "message": "no_favicon_found"}

        img_hash = favicon_info.get("md5")
        icon_url = favicon_info.get("url")

        # Extract root domains (e.g., 'google.com' from 'mail.google.com')
        page_root = _get_domain_root(url)
        page_name = page_root.split('.')[0] # 'google' from 'google.com'

        # --- STRATEGY 1: Hash Database Check (The "Smoking Gun") ---
        # This is high confidence. If we KNOW it's the Google logo, the domain MUST be Google.
        detected_brand = KNOWN_FAVICONS.get(img_hash)

        if detected_brand:
            # Does the page domain contain the brand name?
            # e.g. 'google' in 'google.com' -> True
            # e.g. 'google' in 'go0gle.com' -> False
            if detected_brand in page_root:
                 return {
                    "feature_name": FEATURE_NAME,
                    "score": 0,
                    "weight": WEIGHT,
                    "error": False,
                    "message": f"valid_{detected_brand}_icon"
                }
            else:
                # Critical Mismatch: Google Logo on random domain
                return {
                    "feature_name": FEATURE_NAME,
                    "score": 100,
                    "weight": WEIGHT,
                    "error": False,
                    "message": f"MISMATCH: Found {detected_brand} icon on {page_root}"
                }

        # --- STRATEGY 2: Cross-Origin Hotlinking Check (The "Heuristic") ---
        if icon_url:
            icon_root = _get_domain_root(icon_url)

            # If the icon comes from a different domain
            if icon_root and icon_root != page_root:

                # HEURISTIC A: Organization Keyword Match
                # e.g. page="chase.com", icon="assets-chase.com" -> Safe
                if page_name in icon_root or icon_root.split('.')[0] in page_root:
                    return {"feature_name": FEATURE_NAME, "score": 0, "weight": WEIGHT, "error": False, "message": "cross_origin_related_domain"}

                # HEURISTIC B: Whitelist
                common_hosts = {
                    'imgur.com', 'cloudfront.net', 'wordpress.com', 'wix.com', 'squarespace.com',
                    'gstatic.com', 'googleusercontent.com', 'fbcdn.net', 'akamaihd.net',
                    'twimg.com', 'azureedge.net', 'amazonaws.com', 'shopify.com'
                }

                # Check if icon_root ends with any common host (handles subdomains like my-bucket.s3.amazonaws.com)
                if any(icon_root.endswith(host) for host in common_hosts):
                    return {"feature_name": FEATURE_NAME, "score": 0, "weight": WEIGHT, "error": False, "message": "valid_cdn_hotlink"}

                # HEURISTIC C: Unknown Mismatch (Risk Adjustment)
                # If we are here, it's a mismatch we don't recognize.
                # Don't panic (Score 100), but be suspicious (Score 40).
                return {
                    "feature_name": FEATURE_NAME,
                    "score": 40,  # Lowered from 85 to prevent false positives
                    "weight": WEIGHT,
                    "error": False,
                    "message": f"unknown_hotlink_from_{icon_root}"
                }

        return {
            "feature_name": FEATURE_NAME,
            "score": 0,
            "weight": WEIGHT,
            "error": False,
            "message": "consistent_domain"
        }

    except Exception as e:
        logger.exception("Favicon check failed: %s", e)
        return {"feature_name": FEATURE_NAME, "score": None, "weight": WEIGHT, "error": True, "message": str(e)}
