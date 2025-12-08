# score_engine.py

from log import get_logger

logger = get_logger(__name__)

# --- THRESHOLDS ---
# Updated Thresholds
SAFE_LIMIT = 50       # 0 to 50 is Safe
PHISHING_LIMIT = 75   # Above 75 is Phishing

FEATURE_WEIGHTS = {
    "homoglyph_impersonation": 0.60,
    "open_redirect_detection": 0.55,
    "ssl_presence_and_validity": 0.55,
    "data_uri_scheme": 0.60,
    "threat_intelligence": 0.60,
    "url_structure_analysis": 0.55,
    "favicon_mismatch": 0.45,
    "domain_abuse_detection": 0.45,
    "obfuscation_analysis": 0.45,
    "fast_flux_dns": 0.35,
    "domain_age_analysis": 0.35,
    "path_anomaly_detection": 0.35,
    "random_domain_detection": 0.30,
}

# --- SMOKING GUNS ---
# Technical facts that trigger immediate Phishing verdict (> 75)
SMOKING_GUNS = {
    "data_uri_scheme",
    "homoglyph_impersonation",
    "domain_abuse_detection"
}

def evaluate_score(feature_results: list) -> float:
    """
    Robust Scoring Engine v2 (Updated for SIH Logic)

    Ranges:
    - 0  to 50: Safe
    - 50 to 75: Suspicious (Uncertain/Uncorroborated)
    - 75 to 100: Phishing
    """

    valid_results = []

    # 1. Filtering Phase
    for r in feature_results:
        if r.get("score") is not None and not r.get("error"):
            valid_results.append(r)
        else:
            logger.info(f"Ignoring feature {r.get('feature_name')} (Score is None/Error)")

    if not valid_results:
        return 0.0

    weighted_sum = 0
    total_weight = 0
    max_score = 0
    max_feature = ""

    high_risk_count = 0

    # 2. Calculation Phase
    for r in valid_results:
        name = r.get("feature_name", "unknown")
        score = r.get("score", 0)
        weight = r.get("weight") or FEATURE_WEIGHTS.get(name, 0.30)

        # Track statistics
        if score > max_score:
            max_score = score
            max_feature = name

        # We count a feature as "risky" if it pushes past the Safe zone (> 50)
        if score > SAFE_LIMIT:
            high_risk_count += 1

        weighted_sum += (score * weight)
        total_weight += weight

    base_avg = weighted_sum / total_weight if total_weight > 0 else 0

    # 3. Decision Phase (New Logic)
    final_score = base_avg

    # RULE A: The "Smoking Gun" Override
    # If a technical feature is very high, we force it into the Phishing zone (>75).
    if max_score >= 90 and max_feature in SMOKING_GUNS:
        logger.info(f"Smoking Gun Triggered: {max_feature}")
        # Force a minimum of 90, regardless of average
        final_score = max(final_score, 90.0)

    # RULE B: The "Corroboration" Check
    # We check if the max score exceeds the Phishing Limit (75)
    elif max_score > PHISHING_LIMIT:

        if high_risk_count >= 2:
            # Case 1: Corroborated (Multiple features agree it's bad)
            # We push the score higher into the Phishing zone
            final_score = max(base_avg, max_score * 0.95)
            logger.info(f"Corroborated Phishing Detected. Score boosted.")

        else:
            # Case 2: Uncorroborated (One feature says 90, others say 0)
            # This is likely a False Positive or a grey area.
            # We must force this into the SUSPICIOUS zone (50 - 75).

            logger.info(f"Uncorroborated High Score ({max_feature}). Clamping to Suspicious.")

            # Simple average dampening
            dampened_score = (base_avg + max_score) / 2

            # Clamp logic: Ensure it stays strictly within 50 and 75
            # max(51, ...) ensures it's slightly above safe.
            # min(..., 74) ensures it's slightly below phishing.
            final_score = max(SAFE_LIMIT + 1, min(dampened_score, PHISHING_LIMIT - 1))

    return round(final_score, 2)

# Helper function (Optional, can be used in your API)
def get_risk_label(score):
    if score <= SAFE_LIMIT:
        return "SAFE"
    elif score <= PHISHING_LIMIT:
        return "SUSPICIOUS"
    else:
        return "PHISHING"
