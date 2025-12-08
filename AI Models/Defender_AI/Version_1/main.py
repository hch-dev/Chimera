# Version_1/main_async.py

import asyncio
from concurrent.futures import ThreadPoolExecutor

# --- 1. FEATURE IMPORTS ---
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

# --- 2. CORE IMPORTS ---
from score_engine import evaluate_score
from log import get_logger
from context_loader import load_context

logger = get_logger(__name__)

# Wrapper to run sync features in async loop
async def run_feature(executor, func, url, context):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, func, url, context)

async def run_scan(url: str):
    print(f"\n🚀 Initializing Chimera Defense System (Async Mode)...")
    print(f"📡 Connecting to live target: {url}...")

    # --- PHASE 1: CONTEXT LOADING ---
    loop = asyncio.get_running_loop()
    try:
        with ThreadPoolExecutor() as pool:
            context = await loop.run_in_executor(pool, load_context, url)

        redirect_chain = context.get('http', {}).get('redirect_chain', [])
        print(f"✅ Context Acquired. Redirects: {len(redirect_chain)}")
    except Exception as e:
        print(f"⚠️ Context Load Warning: {e}")
        context = {}

    # --- PHASE 2: PARALLEL FEATURE EXTRACTION ---
    print("🧠 Analyzing patterns (Parallel Execution)...")

    # List of all feature functions to run
    feature_functions = [
        homoglyph_extract, redirect_extract, ssl_extract,
        favicon_extract, structure_extract, domain_abuse_extract,
        data_uri_extract, obfuscation_extract, fast_flux_extract,
        random_domain_extract, domain_age_extract, threat_intel_extract,
        path_extract
    ]

    # Run features in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        tasks = [
            run_feature(executor, func, url, context)
            for func in feature_functions
        ]
        results = await asyncio.gather(*tasks)

    # --- PHASE 3: SCORING ---
    final_score = evaluate_score(results)

    # --- PHASE 4: REPORTING ---
    print("\n" + "="*50)
    print("       🛡️  CHIMERA SECURITY REPORT  🛡️")
    print("="*50)

    for r in results:
        score = r.get('score')
        if score is None:
            status = "⚠️"
            score_display = "ERR"
        else:
            status = "🔴" if score > 50 else "🟢"
            score_display = str(int(score))

        feature_name = r['feature_name'].replace("_", " ").title()
        print(f"{status} {feature_name.ljust(30)}: Risk {score_display}/100")

        # --- RESTORED: Justification Message ---
        if r.get('message'):
            print(f"    └── {r['message']}")

    print("-" * 50)

    # Verdict Logic (Updated for new thresholds)
    if final_score > 75:
        print(f"❌ VERDICT: PHISHING DETECTED (Score: {final_score})")
        print("   Action: BLOCK IMMEDIATELY")
    elif final_score > 50:
        print(f"⚠️ VERDICT: SUSPICIOUS (Score: {final_score})")
        print("   Action: WARN USER & SANDBOX")
    else:
        print(f"✅ VERDICT: SAFE (Score: {final_score})")
        print("   Action: ALLOW")
    print("="*50 + "\n")

if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    url_in = input("Enter URL to scan: ").strip()
    if not url_in.lower().startswith(("http://", "https://", "data:", "blob:")):
        url_in = "https://" + url_in

    asyncio.run(run_scan(url_in))
