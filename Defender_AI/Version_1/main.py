from features.homoglyph import extract as homoglyph_extract
from features.open_redirect import extract as redirect_extract
from features.ssl_present import extract as ssl_extract

from score_engine import evaluate_score
from log import get_logger

# IMPORT THE NEW LOADER
from context_loader import load_context

logger = get_logger(__name__)

def run(url: str):
    print(f"🚀 Initializing Chimera Defense System...")
    print(f"📡 Connecting to live target: {url}...")

    # 1. GET REAL DATA (This takes 1-3 seconds)
    context = load_context(url)

    print(f"✅ Context Acquired. Redirects found: {len(context['http']['redirect_chain'])}")

    results = []

    # 2. RUN ANALYSIS
    print("🧠 Analyzing patterns...")
    results.append(homoglyph_extract(url, context))
    results.append(redirect_extract(url, context))
    results.append(ssl_extract(url, context))

    # 3. SCORE
    final_score = evaluate_score(results)

    # 4. REPORT
    print("\n" + "="*40)
    print("   🛡️  CHIMERA SECURITY REPORT  🛡️")
    print("="*40)

    for r in results:
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
    # Ensure user puts http if missing
    url_in = input("\nEnter URL to scan (e.g. google.com): ").strip()
    if not url_in.startswith("http"):
        url_in = "https://" + url_in

    run(url_in)
