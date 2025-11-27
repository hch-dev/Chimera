import ssl
import socket
import requests
from urllib.parse import urlparse
from datetime import datetime
from log import get_logger

logger = get_logger(__name__)

def get_redirect_chain(url: str) -> list:
    chain = [url]
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    try:
        session = requests.Session()
        session.headers.update(headers)

        # Try GET with stream=True directly as it's more robust than HEAD for shorteners
        response = session.get(url, timeout=10, allow_redirects=True, stream=True)
        response.close()

        for resp in response.history:
            chain.append(resp.url)

        if response.url != chain[-1]:
            chain.append(response.url)

    except Exception as e:
        logger.warning(f"Redirect trace failed for {url}: {e}")

    return chain

def get_ssl_info(url: str) -> dict:
    parsed = urlparse(url if "://" in url else "http://" + url)
    hostname = parsed.hostname

    if not hostname:
        return {"ssl_present": 0}

    # Try secure connection first (Strict Mode)
    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=3) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                return {
                    "ssl_present": 1,
                    "ssl_valid_to": cert.get('notAfter'),
                    "is_valid_cert": True
                }

    except ssl.SSLCertVerificationError:
        # Connection failed due to bad cert (Expired, Self-Signed)
        logger.info(f"Bad SSL detected for {hostname}")
        return {
            "ssl_present": 1,
            "ssl_valid_to": None,
            "is_valid_cert": False,
            "error_type": "verification_failed"
        }

    except Exception as e:
        # Connection failed completely (No SSL port, timeout)
        return {"ssl_present": 0, "error": str(e)}

def load_context(url: str) -> dict:
    logger.info(f"Fetching live context for: {url}")

    return {
        "http": {
            "redirect_chain": get_redirect_chain(url)
        },
        "ssl": get_ssl_info(url)
    }
