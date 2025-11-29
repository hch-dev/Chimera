import sys
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- 1. CRITICAL PATH SETUP ---
# This MUST happen before importing anything from Version_1
# Structure: Chimera/Backend/main.py
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(current_file)
project_root = os.path.dirname(backend_dir)
version1_path = os.path.join(project_root, "Defender_AI", "Version_1")

# Forcefully add Version_1 to the start of sys.path
if version1_path not in sys.path:
    sys.path.insert(0, version1_path)

# --- 2. LOGGER SETUP ---
# We define logger here to avoid circular import issues
import logging
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(backend_dir, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chimera_api")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# --- 3. IMPORT ENGINE ---
try:
    # Now we can import directly
    from context_loader import load_context
    from score_engine import evaluate_score

    # Feature Imports
    from features.homoglyph import extract as homoglyph_extract
    from features.open_redirect import extract as redirect_extract
    from features.ssl_present import extract as ssl_extract
    from features.favicon_mismatch import extract as favicon_extract
    from features.url_structure import extract as structure_extract
    from features.domain_abuse import extract as domain_abuse_extract
    from features.data_uri import extract as data_uri_extract
    from features.obfuscation import extract as obfuscation_extract
    from features.fast_flux import extract as fast_flux_extract
    from features.random_domain import extract as random_domain_extract

    logger.info("✅ Defender Engine Version 1 Loaded Successfully")
except ImportError as e:
    logger.critical(f"❌ Failed to import engine modules: {e}")
    # We define dummy functions so the app doesn't crash on startup
    def load_context(url): return {}
    def evaluate_score(results): return 0
    homoglyph_extract = redirect_extract = ssl_extract = lambda u, c: {}
    favicon_extract = structure_extract = domain_abuse_extract = lambda u, c: {}
    data_uri_extract = obfuscation_extract = fast_flux_extract = random_domain_extract = lambda u, c: {}


# --- 4. API SETUP ---
# Import schemas safely
try:
    from schemas import ScanRequest, ScanResponse
except ImportError:
    # Fallback if running as module
    from Backend.schemas import ScanRequest, ScanResponse

# *** CRITICAL: 'app' must be defined at module level ***
app = FastAPI(title="Chimera API", version="1.0")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "Chimera System Online"}

@app.post("/scan", response_model=ScanResponse)
async def scan_endpoint(request: ScanRequest):
    url = request.url
    logger.info(f"Received scan request for: {url}")

    # A. Load Context
    try:
        context = load_context(url)
    except Exception as e:
        logger.error(f"Context load failed: {e}")
        context = {}

    # B. Run Features
    results_list = []
    extractors = [
        homoglyph_extract, redirect_extract, ssl_extract,
        favicon_extract, structure_extract, domain_abuse_extract,
        data_uri_extract, obfuscation_extract, fast_flux_extract,
        random_domain_extract
    ]

    for extractor in extractors:
        try:
            res = extractor(url, context)
            # Ensure result is valid dict
            if not isinstance(res, dict): res = {}
            results_list.append(res)
        except Exception as e:
            logger.error(f"Extractor failed: {e}")

    # C. Calculate Score
    try:
        final_score = evaluate_score(results_list)
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        final_score = 0

    # D. Verdict
    if final_score > 80:
        verdict = "PHISHING"
        risk = "CRITICAL"
    elif final_score > 40:
        verdict = "SUSPICIOUS"
        risk = "MEDIUM"
    else:
        verdict = "SAFE"
        risk = "LOW"

    # E. Format Details
    details_map = {}
    for r in results_list:
        name = r.get('feature_name', 'unknown')
        details_map[name] = r

    response = {
        "url": url,
        "verdict": verdict,
        "confidence": final_score,
        "risk_level": risk,
        "details": details_map
    }

    logger.info(f"Scan complete. Verdict: {verdict} ({final_score})")
    return response

if __name__ == "__main__":
    # When running directly with python main.py
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
