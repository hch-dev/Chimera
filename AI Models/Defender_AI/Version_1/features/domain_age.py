#Checks the age of domain

import requests
import tldextract
from datetime import datetime
from dateutil import parser # pip install python-dateutil
from log import get_logger

logger = get_logger(__name__)

FEATURE_NAME = "domain_age_analysis"
WEIGHT = 0.45

def get_creation_date_from_rdap(domain):
    """
    Fetches domain age using RDAP (JSON WHOIS).
    Works on Windows/Linux/Mac without installing system binaries.
    """
    try:
        # RDAP is the modern, JSON-based WHOIS.
        # rdap.org is a free redirector to the official registry API.
        response = requests.get(f"https://rdap.org/domain/{domain}", timeout=5)

        if response.status_code != 200:
            return None

        data = response.json()

        # Parse the 'events' list to find registration date
        # Standard RDAP format: "events": [{"eventAction": "registration", "eventDate": "..."}]
        events = data.get('events', [])
        for event in events:
            if event.get('eventAction') in ['registration', 'last changed']:
                date_str = event.get('eventDate')
                # Parse ISO 8601 date format
                return parser.parse(date_str).replace(tzinfo=None)

    except Exception as e:
        logger.debug(f"RDAP lookup failed: {e}")

    return None

def extract(url: str, context: dict = None) -> dict:
    try:
        ext = tldextract.extract(url)
        # Reconstruct domain (e.g. google.com)
        if not ext.domain or not ext.suffix:
             return {"feature_name": FEATURE_NAME, "score": 0, "weight": WEIGHT, "error": False, "message": "invalid_domain"}

        domain = f"{ext.domain}.{ext.suffix}"

        # 1. Fetch Date via API
        created_at = get_creation_date_from_rdap(domain)

        if not created_at:
            # If API fails or TLD doesn't support RDAP, we default to neutral.
            # We don't want to block a site just because the registry is slow.
            return {"feature_name": FEATURE_NAME, "score": 10, "weight": WEIGHT, "error": False, "message": "age_data_unavailable"}

        # 2. Calculate Age
        now = datetime.utcnow()
        age_days = (now - created_at).days

        # 3. Scoring Logic
        if age_days < 14:
            # Baby domains (under 2 weeks) are extremely suspicious
            return {
                "feature_name": FEATURE_NAME,
                "score": 100,
                "weight": WEIGHT,
                "error": False,
                "message": f"baby_domain_detected ({age_days} days old)"
            }
        elif age_days < 60:
            return {
                "feature_name": FEATURE_NAME,
                "score": 80,
                "weight": WEIGHT,
                "error": False,
                "message": f"young_domain ({age_days} days old)"
            }
        elif age_days < 180:
            return {
                "feature_name": FEATURE_NAME,
                "score": 40,
                "weight": WEIGHT,
                "error": False,
                "message": f"immature_domain ({age_days} days old)"
            }

        return {
            "feature_name": FEATURE_NAME,
            "score": 0,
            "weight": WEIGHT,
            "error": False,
            "message": f"established_domain ({age_days} days old)"
        }

    except Exception as e:
        logger.warning(f"Domain age check failed: {e}")
        return {"feature_name": FEATURE_NAME, "score": None, "weight": WEIGHT, "error": True, "message": str(e)}
