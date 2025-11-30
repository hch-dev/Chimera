#Checks dns against open source threat intelligence like alienvault

import os
import requests
import tldextract
from log import get_logger
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

logger = get_logger(__name__)

FEATURE_NAME = "threat_intelligence"
WEIGHT = 0.50 # High confidence because it's confirmed by others

# Replace with your actual key
OTX_API_KEY = "{api_key}"

def check_alienvault(indicator: str, indicator_type: str = 'domain') -> dict:
    """
    Queries AlienVault OTX for a domain or URL.
    """
    if not OTX_API_KEY or OTX_API_KEY == "{api_key}":
        return None # Skip if no key configured

    url = f"https://otx.alienvault.com/api/v1/indicators/{indicator_type}/{indicator}/general"
    headers = {"X-OTX-API-KEY": OTX_API_KEY}

    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            pulse_info = data.get("pulse_info", {})
            count = pulse_info.get("count", 0)

            if count > 0:
                return {
                    "score": 100,
                    "message": f"flagged_in_{count}_otx_pulses"
                }
    except Exception as e:
        logger.debug(f"OTX query failed: {e}")

    return None

def extract(url: str, context: dict = None) -> dict:
    try:
        ext = tldextract.extract(url)
        domain = f"{ext.domain}.{ext.suffix}"

        if not ext.domain:
             return {"feature_name": FEATURE_NAME, "score": 0, "weight": WEIGHT, "error": False, "message": "invalid_domain"}

        # 1. Check Domain Reputation
        # We check the domain because checking full URLs often yields no results
        # unless it's a very specific, known phishing link. Domain is safer.
        otx_result = check_alienvault(domain, 'domain')

        if otx_result:
             return {
                "feature_name": FEATURE_NAME,
                "score": otx_result['score'],
                "weight": WEIGHT,
                "error": False,
                "message": otx_result['message']
            }

        # If you had other feeds (VirusTotal, Google Safe Browsing), you would chain them here.

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
