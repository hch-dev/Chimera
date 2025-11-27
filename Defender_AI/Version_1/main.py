# Version_1/main.py

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

def run(url: str):
    print(f"\n🚀 Initializing Chimera Defense System...")
    print(f"📡 Connecting to live target: {url}...")

    # --- PHASE 1: CONTEXT LOADING ---
    # Fetches live data (SSL certs, redirect chains, favicons) from the internet.
    # Note: If the URL is a Data URI, this step might fail or return empty context, which is handled.
    try:
        context = load_context(url)
        redirect_chain = context.get('http', {}).get('redirect_chain', [])
        redirect_count = len(redirect_chain)
        print(f"✅ Context Acquired. Redirects found: {redirect_count}")
    except Exception as e:
        # Fallback for Data URIs or Network Errors
        print(f"⚠️ Context Load Warning: {e}")
        context = {}

    results = []

    # --- PHASE 2: FEATURE EXTRACTION ---
    print("🧠 Analyzing patterns...")

    # 1. Homoglyph Check (Punycode & Script Mixing)
    results.append(homoglyph_extract(url, context))

    # 2. Redirect Chain Analysis (Open Redirects & Shorteners)
    results.append(redirect_extract(url, context))

    # 3. SSL Verification (Expired, Self-Signed, Wrong Host)
    results.append(ssl_extract(url, context))

    # 4. Favicon Analysis (Brand Impersonation via Icon)
    results.append(favicon_extract(url, context))

    # 5. URL Structure (Authority Abuse '@', Raw IPs)
    results.append(structure_extract(url, context))

    # 6. Domain Abuse (Combosquatting & Subdomain Shadowing)
    results.append(domain_abuse_extract(url, context))

    # 7. Data URI & Protocol Check (Malicious Protocols)
    results.append(data_uri_extract(url, context))

    # 8. Obfuscation Decoder (Base64/Hex Hidden Payloads)
    results.append(obfuscation_extract(url, context))

    # 9. Fast Flux DNS (Dynamic Botnet IPs)
    results.append(fast_flux_extract(url, context))

    # 10. Random Domain DGA (Entropy & Gibberish)
    results.append(random_domain_extract(url, context))
    results.append(domain_age_extract(url, context))
    results.append(threat_intel_extract(url, context))



    # --- PHASE 3: SCORING ---
    final_score = evaluate_score(results)

    # --- PHASE 4: REPORTING ---
    print("\n" + "="*50)
    print("       🛡️  CHIMERA SECURITY REPORT  🛡️")
    print("="*50)

    for r in results:
        # Safely handle None scores (Error state)
        score = r.get('score')

        if score is None:
            status = "⚠️"
            score_display = "ERR"
        else:
            # Visual indicator: Red circle for high risk (>50), Green for safe
            status = "🔴" if score > 50 else "🟢"
            score_display = str(int(score)) # Display as integer

        # Format output for readability
        feature_name = r['feature_name'].replace("_", " ").title()
        print(f"{status} {feature_name.ljust(30)}: Risk {score_display}/100")

        if r.get('message'):
            print(f"    └── {r['message']}")

    print("-" * 50)

    # Final Verdict Logic
    if final_score > 80:
        print(f"❌ VERDICT: PHISHING DETECTED (Score: {final_score})")
        print("   Action: BLOCK IMMEDIATELY")
    elif final_score > 40:
        print(f"⚠️ VERDICT: SUSPICIOUS (Score: {final_score})")
        print("   Action: WARN USER & SANDBOX")
    else:
        print(f"✅ VERDICT: SAFE (Score: {final_score})")
        print("   Action: ALLOW")
    print("="*50 + "\n")

if __name__ == "__main__":
    print("\n--- Chimera CLI Scanner v1.0 ---\n")
    url_in = input("Enter URL to scan (e.g. google.com): ").strip()

    # Smart Protocol Handling:
    # If user types 'google.com', add https://
    # If user types 'data:text/html...', leave it alone.
    if not url_in.lower().startswith(("http://", "https://", "data:", "blob:")):
        url_in = "https://" + url_in

    run(url_in)
