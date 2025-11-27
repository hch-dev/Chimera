import ssl
import socket
import requests
import hashlib
from urllib.parse import urlparse, urljoin
from datetime import datetime
from bs4 import BeautifulSoup
from log import get_logger

logger = get_logger(__name__)

# Shared headers to mimic a real browser
BROWSER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

def get_redirect_chain(url: str) -> list:
    chain = [url]
    try:
        session = requests.Session()
        session.headers.update(BROWSER_HEADERS)

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

def get_favicon_hash(url: str) -> dict:
    """
    Downloads the favicon and returns its MD5 hash for brand comparison.
    """
    try:
        session = requests.Session()
        session.headers.update(BROWSER_HEADERS)

        # 1. Get the page content to find the link tag
        response = session.get(url, timeout=5, stream=True)
        # Read only first 50kb to find <head> tags, saves bandwidth
        content = next(response.iter_content(50000), b"")

        soup = BeautifulSoup(content, 'html.parser')

        # Try to find <link rel="icon" ...>
        icon_link = soup.find("link", rel=lambda x: x and 'icon' in x.lower())
        favicon_url = None

        if icon_link and icon_link.get("href"):
            favicon_url = icon_link.get("href")
            # Handle relative URLs (e.g. /static/icon.png)
            favicon_url = urljoin(url, favicon_url)
        else:
            # Fallback: Try standard root location
            parsed = urlparse(url)
            favicon_url = f"{parsed.scheme}://{parsed.netloc}/favicon.ico"

        # 2. Download the image
        if favicon_url:
            img_resp = session.get(favicon_url, timeout=5)
            if img_resp.status_code == 200:
                # Calculate MD5
                md5_hash = hashlib.md5(img_resp.content).hexdigest()
                return {"md5": md5_hash, "url": favicon_url}

    except Exception as e:
        logger.warning(f"Favicon fetch error for {url}: {e}")

    return {"md5": None, "error": "fetch_failed"}

def load_context(url: str) -> dict:
    logger.info(f"Fetching live context for: {url}")

    return {
        "http": {
            "redirect_chain": get_redirect_chain(url)
        },
        "ssl": get_ssl_info(url),
        "favicon": get_favicon_hash(url)  # <--- Added Feature
    }
