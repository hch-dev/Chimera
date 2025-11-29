#Checks the path of the phishing site

from urllib.parse import urlparse
import os
from log import get_logger

logger = get_logger(__name__)

FEATURE_NAME = "path_anomaly_detection"
WEIGHT = 0.40

# Directories often targeted by exploit kits
SUSPICIOUS_DIRS = {
    "wp-content", "wp-includes", "wp-admin", # WordPress
    "images", "img", "css", "js", "assets", # Asset folders shouldn't have login pages
    "plugins", "modules", "components",     # CMS plugins
    "well-known", "cgi-bin"
}

# Extensions that shouldn't be in asset folders
EXECUTABLE_EXTS = {".php", ".html", ".htm", ".jsp", ".asp", ".aspx", ".exe"}

def extract(url: str, context: dict = None) -> dict:
    """
    Detects phishing pages hosted on compromised legitimate sites
    by analyzing the URL path structure.
    """
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        path = parsed.path.lower()

        if not path or path == "/":
             return {"feature_name": FEATURE_NAME, "score": 0, "weight": WEIGHT, "error": False, "message": "root_path"}

        segments = [s for s in path.split("/") if s]
        filename = segments[-1] if segments else ""
        _, ext = os.path.splitext(filename)

        # 1. Depth Check (Buried Pages)
        # Legitimate login pages are usually /login or /auth (Depth 1)
        # Phishing pages are often /wp-content/uploads/2024/05/secure/login.php (Depth 5+)
        if len(segments) > 4:
            return {
                "feature_name": FEATURE_NAME,
                "score": 60,
                "weight": WEIGHT,
                "error": False,
                "message": f"deeply_nested_path_depth_{len(segments)}"
            }

        # 2. Tilde Usage (User Directories)
        # http://university.edu/~student/login.html
        if "~" in path:
            return {
                "feature_name": FEATURE_NAME,
                "score": 70,
                "weight": WEIGHT,
                "error": False,
                "message": "user_directory_tilde_detected"
            }

        # 3. Context Mismatch (Code in Image Folders)
        # Finding a .php file inside /images/ is 99% malware
        for seg in segments[:-1]: # Check folders, excluding the file itself
            if seg in SUSPICIOUS_DIRS and ext in EXECUTABLE_EXTS:
                return {
                    "feature_name": FEATURE_NAME,
                    "score": 90,
                    "weight": WEIGHT,
                    "error": False,
                    "message": f"code_in_asset_folder ({ext} inside /{seg})"
                }

        return {"feature_name": FEATURE_NAME, "score": 0, "weight": WEIGHT, "error": False, "message": "path_structure_normal"}

    except Exception as e:
        logger.exception(f"Path analysis failed: {e}")
        return {"feature_name": FEATURE_NAME, "score": None, "weight": WEIGHT, "error": True, "message": str(e)}
