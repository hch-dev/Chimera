# Version_1/main.py
from features.homoglyph import extract as homoglyph_extract
from features.open_redirect import extract as redirect_extract
from features.ssl_present import extract as ssl_extract
from features.favicon_mismatch import extract as favicon_extract
from features.url_structure import extract as structure_extract
from features.domain_abuse import extract as domain_abuse_extract
from features.data_uri import extract as data_uri_extract
from features.obfuscation import extract as obfuscation_extract

from score_engine import evaluate_score
from log import get_logger

# IMPORT THE NEW LOADER
from context_loader import load_context

logger = get_logger(__name__)

def run(url: str):
    print(f"🚀 Initializing Chimera Defense System...")
    print(f"📡 Connecting to live target: {url}...")

    # 1. GET REAL DATA (This takes 1-3 seconds)
    # We must load context FIRST because features rely on it
    # Note: For Data URIs, context loading will likely fail (no network), but that's expected.
    context = load_context(url)

    print(f"✅ Context Acquired. Redirects found: {len(context['http']['redirect_chain'])}")

    results = []

    # 2. RUN ANALYSIS
    print("🧠 Analyzing patterns...")
    results.append(homoglyph_extract(url, context))
    results.append(redirect_extract(url, context))
    results.append(ssl_extract(url, context))
    results.append(favicon_extract(url, context))
    results.append(structure_extract(url, context))
    results.append(domain_abuse_extract(url, context))
    results.append(data_uri_extract(url, context))
    results.append(obfuscation_extract(url, context))

    # 3. SCORE
    final_score = evaluate_score(results)

    # 4. REPORT
    print("\n" + "="*40)
    print("   🛡️  CHIMERA SECURITY REPORT  🛡️")
    print("="*40)

    for r in results:
        # Visual indicator: Red circle for high risk, Green for safe
        status = "🔴" if r['score'] > 50 else "🟢"
        print(f"{status} {r['feature_name'].ljust(25)}: Risk {r['score']}/100")

        if r.get('message'):
            print(f"    └── {r['message']}")

    print("-" * 40)
    if final_score > 70:
        print(f"❌ VERDICT: PHISHING DETECTED (Score: {final_score})")
    elif final_score > 40:
        print(f"⚠️ VERDICT: SUSPICIOUS (Score: {final_score})")
    else:
        print(f"✅ VERDICT: SAFE (Score: {final_score})")
    print("="*40 + "\n")

if __name__ == "__main__":
    # Ensure user puts http if missing, BUT skip for data/blob URIs
    url_in = input("\nEnter URL to scan (e.g. google.com): ").strip()

    # Improved protocol check to avoid breaking Data URIs
    if not url_in.lower().startswith(("http://", "https://", "data:", "blob:")):
        url_in = "https://" + url_in

    run(url_in)
