# features/threat_intel.py

import os
import requests
import tldextract
from log import get_logger
from dotenv import load_dotenv

load_dotenv()
OTX_API_KEY = os.getenv("OTX_API_KEY")

logger = get_logger(__name__)

FEATURE_NAME = "threat_intelligence"
WEIGHT = 0.50

# --- FIX 2: Whitelist for Popular Domains ---
# Major sites appear in many OTX pulses simply because they are referenced
# (e.g., "Phishing hosted on Google Forms"). We must whitelist them.
WHITELIST = {
    "google.com", "facebook.com", "youtube.com", "twitter.com", "instagram.com",
    "linkedin.com", "apple.com", "microsoft.com", "amazon.com", "wikipedia.org",
    "baidu.com", "yahoo.com", "yandex.ru", "netflix.com", "whatsapp.com"
}

def check_alienvault(domain: str) -> dict:
    if not OTX_API_KEY:
        return None

    # Check Whitelist
    if domain in WHITELIST:
        return {"score": 0, "message": "whitelisted_major_platform"}

    # Proceed with API
    url = f"https://otx.alienvault.com/api/v1/indicators/domain/{domain}/general"
    headers = {"X-OTX-API-KEY": OTX_API_KEY}

    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            pulse_info = data.get("pulse_info", {})
            count = pulse_info.get("count", 0)

            # --- FIX: Threshold Adjustment ---
            # 1-5 pulses is often noise. 10+ is suspicious.
            if count > 10:
                return {
                    "score": 100,
                    "message": f"flagged_in_{count}_otx_pulses"
                }
            elif count > 0:
                 return {
                    "score": 40, # Lower score for low counts
                    "message": f"low_otx_activity_{count}_pulses"
                }

    except Exception as e:
        logger.debug(f"OTX query failed: {e}")

    return None

def extract(url: str, context: dict = None) -> dict:
    try:
        ext = tldextract.extract(url)
        if not ext.domain:
             return {"feature_name": FEATURE_NAME, "score": 0, "weight": WEIGHT, "error": False, "message": "invalid_domain"}

        domain = f"{ext.domain}.{ext.suffix}".lower()

        otx_result = check_alienvault(domain)

        if otx_result:
             return {
                "feature_name": FEATURE_NAME,
                "score": otx_result['score'],
                "weight": WEIGHT,
                "error": False,
                "message": otx_result['message']
            }

        return {
            "feature_name": FEATURE_NAME,
            "score": 0,
            "weight": WEIGHT,
            "error": False,
            "message": "clean_in_threat_feeds"
        }

    except Exception as e:
        logger.exception(f"Threat intel check failed: {e}")
        return {"feature_name": FEATURE_NAME, "score": None, "weight": WEIGHT, "error": True, "message": str(e)}
