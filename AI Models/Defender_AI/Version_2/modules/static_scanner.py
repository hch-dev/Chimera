import re
from urllib.parse import urlparse
from log import LOG

class StaticScanner:
    """
    Performs static (V1) analysis on a URL string using heuristic rules.
    Scores based on syntactic abnormalities common in phishing attempts.
    """

    def __init__(self):
        LOG.info("StaticScanner (V1) initialized.")

    def _check_ip_address(self, url: str) -> int:
        """Checks if the hostname is an IP address instead of a domain name."""
        try:
            domain = urlparse(url).netloc
            # Simple IPv4 check pattern
            if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?$", domain):
                return 30
        except:
            pass
        return 0

    def _check_at_symbol(self, url: str) -> int:
        """Checks for the presence of the '@' symbol in the URL."""
        return 20 if '@' in url else 0

    def _check_long_url(self, url: str) -> int:
        """Checks if the URL length exceeds a common limit (e.g., 60 characters)."""
        return 15 if len(url) > 60 else 0

    def _check_subdomain_count(self, url: str) -> int:
        """Checks for excessive subdomains, often used to hide the true domain."""
        try:
            domain = urlparse(url).netloc
            # Count dots in the domain name
            count = domain.count('.')
            # Example: google.com (1 dot), www.google.com (2 dots)
            # Anything >= 4 dots is often suspicious
            return 25 if count >= 4 else 0
        except:
            return 0

    def get_static_score(self, url: str) -> tuple[int, dict]:
        """
        Runs all heuristic checks against the URL.

        :param url: The URL string to analyze.
        :return: Tuple of (Total Score, Detailed Analysis Dictionary)
        """
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
            
        analysis = {
            "ip_address": self._check_ip_address(url),
            "at_symbol": self._check_at_symbol(url),
            "long_url": self._check_long_url(url),
            "subdomain_count": self._check_subdomain_count(url)
        }
        
        total_score = sum(analysis.values())
        
        # Cap the score at 100
        final_score = min(total_score, 100)
        
        LOG.debug(f"V1 Heuristic Analysis for {url}: {analysis}")

        return final_score, analysis