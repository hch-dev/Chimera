import ssl
import socket
import requests
from urllib.parse import urlparse
from datetime import datetime
from log import get_logger

logger = get_logger(__name__)

def get_redirect_chain(url: str) -> list:
    """
    Follows HTTP redirects and returns the list of URLs visited.
    """
    chain = [url]
    try:
        # Use a browser User-Agent to avoid being blocked
        headers = {'User-Agent': 'Mozilla/5.0 (Chimera-Scanner)'}

        # HEAD request is faster (doesn't download body)
        response = requests.head(url, allow_redirects=True, headers=headers, timeout=5)

        # Add history (intermediate hops)
        for resp in response.history:
            chain.append(resp.url)

        # Add final destination if different
        if response.url != chain[-1]:
            chain.append(response.url)

    except Exception as e:
        logger.warning(f"Network scan failed for {url}: {e}")
        # Return what we have (at least the original url)

    return chain

def get_ssl_info(url: str) -> dict:
    """
    Connects to port 443 and retrieves SSL certificate details.
    """
    parsed = urlparse(url if "://" in url else "http://" + url)
    hostname = parsed.hostname

    if not hostname:
        return {"ssl_present": 0}

    context = ssl.create_default_context()
    context.check_hostname = False # We want to inspect bad certs too, not just crash
    context.verify_mode = ssl.CERT_NONE

    try:
        with socket.create_connection((hostname, 443), timeout=3) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()

                # If we got here, a cert exists (even if self-signed)
                # Note: getpeercert() might return empty dict for some config,
                # using binary_form=True and parsing is more robust but complex.
                # For now, we assume standard successful handshake = cert present.

                # Since we used CERT_NONE, getpeercert() returns empty for some python versions
                # Let's try a strict check just to get the dates
                return {
                    "ssl_present": 1,
                    "ssl_valid_to": _fetch_cert_date(hostname)
                }
    except Exception as e:
        logger.warning(f"SSL handshake failed: {e}")
        return {"ssl_present": 0, "error": str(e)}

def _fetch_cert_date(hostname):
    # Helper to get the clean date string
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=hostname) as s:
            s.settimeout(3)
            s.connect((hostname, 443))
            cert = s.getpeercert()
            return cert.get('notAfter') # Returns string like 'Nov 24 12:00:00 2025 GMT'
    except:
        return None

def load_context(url: str) -> dict:
    """
    The Master Function: Aggregates all live data.
    """
    logger.info(f"Fetching live context for: {url}")

    return {
        "http": {
            "redirect_chain": get_redirect_chain(url)
        },
        "ssl": get_ssl_info(url)
    }
