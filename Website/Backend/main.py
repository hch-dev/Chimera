# Backend/main.py
import sys
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

# --- 1. PATH FIX BLOCK (CRITICAL FIX) ---
# We need to navigate from: .../Chimera/Website/Backend/
# To: ..................... .../Chimera/AI_Models/Defender_AI/Version_1/

# Get the directory where this script is running (.../Backend)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up two levels to reach the project root (.../Chimera)
# Step 1: Go up to 'Website'
website_dir = os.path.dirname(current_dir)
# Step 2: Go up to 'Chimera'
project_root = os.path.dirname(website_dir)

# Construct the path to the 'Version_1' folder where 'features' lives
target_path = os.path.join(project_root, "AI Models", "Defender_AI", "Version_1")

# Add it to Python's search path
if target_path not in sys.path:
    sys.path.append(target_path)

print(f"🔗 Bridging connection to AI Models at: {target_path}")
# -----------------------------------------------------------

# --- IMPORTS (Now these will work) ---
try:
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
except ModuleNotFoundError as e:
    print(f"\n❌ CRITICAL IMPORT ERROR: {e}")
    print(f"ℹ️  Python path includes: {sys.path}")
    print("💡 Please verify the 'features' folder is inside: Chimera/AI_Models/Defender_AI/Version_1/\n")
    sys.exit(1)

# Wrapper to run sync features in async loop
async def run_feature(executor, func, url, context):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, func, url, context)

async def scan_url(url: str):
    """
    Runs the full scanning pipeline on a URL.
    Returns a dictionary containing the final score, verdict, and raw feature data.
    """
    print(f"🔍 Backend Logic Scanning: {url}...")

    # --- PHASE 1: CONTEXT LOADING ---
    loop = asyncio.get_running_loop()
    try:
        with ThreadPoolExecutor() as pool:
            context = await loop.run_in_executor(pool, load_context, url)
    except Exception as e:
        print(f"⚠️ Context Load Error: {e}")
        context = {}

    # --- PHASE 2: PARALLEL FEATURE EXTRACTION ---
    feature_functions = [
        homoglyph_extract, redirect_extract, ssl_extract,
        favicon_extract, structure_extract, domain_abuse_extract,
        data_uri_extract, obfuscation_extract, fast_flux_extract,
        random_domain_extract, domain_age_extract, threat_intel_extract,
        path_extract
    ]

    with ThreadPoolExecutor(max_workers=15) as executor:
        tasks = [run_feature(executor, func, url, context) for func in feature_functions]
        results = await asyncio.gather(*tasks)

    # --- PHASE 3: SCORING ---
    final_score = evaluate_score(results)

    # --- PHASE 4: VERDICT GENERATION ---
    verdict = "SAFE"
    if final_score > 80:
        verdict = "PHISHING"
    elif final_score > 40:
        verdict = "SUSPICIOUS"

    # --- RETURN DATA ---
    return {
        "url": url,
        "final_score": final_score,
        "verdict": verdict,
        "features": results
    }

# CLI Test Block
if __name__ == "__main__":
    print("--- CLI Manual Scan Mode ---")
    url_in = input("Enter URL to scan: ")
    if not url_in.startswith(("http", "data:")):
        url_in = "https://" + url_in

    data = asyncio.run(scan_url(url_in))

    print("\n" + "="*30)
    print(f"VERDICT: {data['verdict']}")
    print(f"SCORE:   {data['final_score']}/100")
    print("="*30)
