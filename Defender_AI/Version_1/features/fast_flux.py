#Checks the TTL for a site

import dns.resolver
import tldextract
from log import get_logger

logger = get_logger(__name__)

FEATURE_NAME = "fast_flux_dns"
WEIGHT = 0.35

def extract(url: str, context: dict = None) -> dict:
    """
    Detects Fast Flux characteristics:
    - Extremely short TTL (Time To Live) < 300s.
    - High number of IP addresses returned in a single query.
    """
    try:
        # 1. Extract Hostname
        ext = tldextract.extract(url)
        if not ext.domain:
             return {"feature_name": FEATURE_NAME, "score": 0, "weight": WEIGHT, "error": False, "message": "invalid_domain"}

        hostname = f"{ext.subdomain}.{ext.domain}.{ext.suffix}".lstrip(".")

        # 2. Perform DNS Lookup
        resolver = dns.resolver.Resolver()
        resolver.lifetime = 3  # 3 second timeout to keep it fast

        try:
            answers = resolver.resolve(hostname, 'A')
        except Exception:
            # If DNS fails, we can't judge Fast Flux. Return Neutral.
            return {"feature_name": FEATURE_NAME, "score": 0, "weight": WEIGHT, "error": False, "message": "dns_resolution_failed"}

        ttl = answers.rrset.ttl
        ip_count = len(answers)

        score = 0
        flags = []

        # CHECK 1: TTL Analysis
        # Fast Flux networks need low TTL (0-180s) to rotate IPs quickly.
        # Normal sites usually have 3600s or more.
        if ttl < 60:
            score += 80
            flags.append(f"critical_low_ttl_{ttl}s")
        elif ttl < 300:
            score += 40
            flags.append(f"suspicious_low_ttl_{ttl}s")

        # CHECK 2: Volume of IPs
        # Returning 10+ IPs in a single query is distinctively "Fluxy" or CDN-like.
        if ip_count > 10:
            score += 30
            flags.append(f"mass_ip_rotation_{ip_count}_ips")
        elif ip_count > 5:
            score += 10
            flags.append(f"high_ip_count_{ip_count}_ips")

        # WHITELIST CHECK (CDN Protection)
        # CDNs (Cloudflare, Akamai) use Fast-Flux techniques legally for load balancing.
        # We must whitelist known giants to avoid False Positives.
        safe_domains = {
            'google', 'facebook', 'amazon', 'cloudflare', 'fastly',
            'akamai', 'microsoft', 'apple', 'netflix'
        }

        if ext.domain.lower() in safe_domains:
            return {
                "feature_name": FEATURE_NAME,
                "score": 0,
                "weight": WEIGHT,
                "error": False,
                "message": f"whitelisted_cdn_behavior_ttl_{ttl}"
            }

        if score > 0:
             return {
                "feature_name": FEATURE_NAME,
                "score": min(100, score),
                "weight": WEIGHT,
                "error": False,
                "message": ", ".join(flags)
            }

        return {
            "feature_name": FEATURE_NAME,
            "score": 0,
            "weight": WEIGHT,
            "error": False,
            "message": f"normal_dns (ttl={ttl}, ips={ip_count})"
        }

    except Exception as e:
        logger.warning(f"Fast flux check error: {e}")
        return {"feature_name": FEATURE_NAME, "score": None, "weight": WEIGHT, "error": True, "message": str(e)}
