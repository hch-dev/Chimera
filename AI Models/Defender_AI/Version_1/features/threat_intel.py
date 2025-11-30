import os
import requests
import tldextract
from log import get_logger
from dotenv import load_dotenv

# 1. Load the variables
load_dotenv()

# 2. Get the specific key directly from the environment
# Ensure your .env file has a variable named OTX_API_KEY (see below)
OTX_API_KEY = os.getenv("OTX_API_KEY")

logger = get_logger(__name__)

FEATURE_NAME = "threat_intelligence"
WEIGHT = 0.50

def check_alienvault(indicator: str, indicator_type: str = 'domain') -> dict:
    """
    Queries AlienVault OTX for a domain or URL.
    """
    # 3. Simplified check: Just check if the key exists and isn't empty
    if not OTX_API_KEY:
        logger.warning("OTX_API_KEY is missing. Skipping Threat Intelligence check.")
        return None

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
        # Ideally handle 403 (Forbidden) explicitly to know if key is bad
        elif response.status_code == 403:
             logger.error("OTX API Key is invalid or expired.")

    except Exception as e:
        logger.debug(f"OTX query failed: {e}")

    return None

def extract(url: str, context: dict = None) -> dict:
    try:
        ext = tldextract.extract(url)
        # Handle cases where tldextract fails to find a domain (e.g. raw IPs)
        if not ext.domain:
             return {"feature_name": FEATURE_NAME, "score": 0, "weight": WEIGHT, "error": False, "message": "invalid_domain"}

        domain = f"{ext.domain}.{ext.suffix}"

        otx_result = check_alienvault(domain, 'domain')

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
