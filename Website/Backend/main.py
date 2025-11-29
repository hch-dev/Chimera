# backend/main.py
import asyncio
from concurrent.futures import ThreadPoolExecutor

# --- IMPORTS (Same as before) ---
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
from features.domain_age import extract as domain_age_extract
from features.threat_intel import extract as threat_intel_extract
from features.path_analysis import extract as path_extract

from score_engine import evaluate_score
from context_loader import load_context

# Wrapper to run sync features in async loop
async def run_feature(executor, func, url, context):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, func, url, context)

# --- RENAMED & MODIFIED FUNCTION ---
async def scan_url(url: str):
    """
    Runs the scan and returns a DICTIONARY (JSON-ready), not print statements.
    """
    print(f"📡 Backend Scanning: {url}...")

    # 1. Context Loading
    loop = asyncio.get_running_loop()
    try:
        with ThreadPoolExecutor() as pool:
            context = await loop.run_in_executor(pool, load_context, url)
    except Exception as e:
        context = {}

    # 2. Parallel Feature Extraction
    feature_functions = [
        homoglyph_extract, redirect_extract, ssl_extract,
        favicon_extract, structure_extract, domain_abuse_extract,
        data_uri_extract, obfuscation_extract, fast_flux_extract,
        random_domain_extract, domain_age_extract, threat_intel_extract,
        path_extract
    ]

    with ThreadPoolExecutor(max_workers=10) as executor:
        tasks = [run_feature(executor, func, url, context) for func in feature_functions]
        results = await asyncio.gather(*tasks)

    # 3. Scoring
    final_score = evaluate_score(results)

    # 4. Determine Verdict
    verdict = "SAFE"
    if final_score > 80: verdict = "PHISHING"
    elif final_score > 40: verdict = "SUSPICIOUS"

    # 5. Return Structured Data
    return {
        "url": url,
        "final_score": final_score,
        "verdict": verdict,
        "features": results  # List of dicts: [{'feature_name':..., 'score':...}, ...]
    }

# Keep CLI functionality for testing
if __name__ == "__main__":
    url = input("Enter URL: ")
    print(asyncio.run(scan_url(url)))
