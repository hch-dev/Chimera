import re
from urllib.parse import urlparse
from log import LOG

class URLScanner:
    """
    Performs static analysis on a URL based on heuristics (Refactored V1 logic).
    Returns a score from 0 (Safe) to 100 (High Risk).
    """

    def __init__(self):
        self.max_v1_score = 100
        self.risk_factors = []
        LOG.info("URLScanner (V1 Heuristics) initialized.")

    def _check_ip_address(self, url_parts):
        """Checks if the domain is an IP address instead of a domain name (V1 Feature)."""
        host = url_parts.netloc
        # Regex for simple IPv4 address check
        ip_match = re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?$", host)
        if ip_match:
            self.risk_factors.append(("IP Address in Host", 30))
            return 30
        return 0

    def _check_long_url(self, url):
        """Checks if the URL length is excessive (V1 Feature)."""
        if len(url) > 75:
            # Score scales based on length beyond 75, up to a max of 20
            score = min(20, (len(url) - 75) // 5) 
            self.risk_factors.append(("Long URL Length", score))
            return score
        return 0

    def _check_suspicious_keywords(self, url):
        """Checks for suspicious words in the path (V1 Feature)."""
        keywords = ['login', 'verify', 'secure', 'account', 'update', 'webscr']
        score = 0
        url_path = urlparse(url).path.lower()
        url_netloc = urlparse(url).netloc.lower()
        
        for kw in keywords:
            # Only flag if the keyword is in the path/query and not just the domain itself
            if kw in url_path and kw not in url_netloc:
                score += 10
                self.risk_factors.append((f"Suspicious Keyword: {kw}", 10))
        return min(score, 40) # Cap score for this check

    def _check_subdomain_abuse(self, url_parts):
        """Checks for excessive or suspicious subdomains (V1 Feature)."""
        host = url_parts.netloc.lower()
        # Count dots to estimate subdomain depth. We expect 1 (domain.tld) or 2 (www.domain.tld)
        dot_count = host.count('.')
        
        # A high dot count (e.g., > 3) suggests obfuscation
        if dot_count > 3:
            score = (dot_count - 3) * 5
            self.risk_factors.append((f"Excessive Subdomains ({dot_count})", score))
            return min(score, 30)
        return 0

    def get_static_score(self, url):
        """Runs all heuristic checks and returns the aggregated V1 score."""
        self.risk_factors = []
        total_score = 0
        
        try:
            url_parts = urlparse(url)
            
            # Run individual checks
            total_score += self._check_ip_address(url_parts)
            total_score += self._check_long_url(url)
            total_score += self._check_suspicious_keywords(url)
            total_score += self._check_subdomain_abuse(url_parts)
            
            # Cap the final score at the maximum possible
            final_score = min(total_score, self.max_v1_score)
            
            LOG.info(f"V1 Heuristic Score for '{url}': {final_score}/100")
            LOG.debug(f"V1 Risk Factors: {self.risk_factors}")
            return final_score

        except Exception as e:
            LOG.error(f"Error in V1 URL scanning: {e}")
            return 0 # Default to safe if processing fails