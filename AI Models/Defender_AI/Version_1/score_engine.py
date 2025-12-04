# score_engine.py

from log import get_logger

logger = get_logger(__name__)

# Weights remain the same, but we add logic to handle them differently
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
# If any of these score 100, we don't need corroboration.
# These are technical facts that are rarely false positives.
SMOKING_GUNS = {
    "data_uri_scheme",
    "homoglyph_impersonation",
    "domain_abuse_detection" # e.g., paypal-login.com
}

def evaluate_score(feature_results: list) -> float:
    """
    Robust Scoring Engine v2

    1. Filter out errors/ignored features (Score=None).
    2. Check for "Smoking Guns" (Instant Fail).
    3. Calculate Weighted Average.
    4. Apply Corroboration Logic for "Soft" High Scores (like Threat Intel).
    """

    valid_results = []

    # 1. Filtering Phase
    for r in feature_results:
        # We explicitly check for None. If score is 0, we keep it!
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

    high_risk_count = 0 # How many features think this is bad (>50)?

    # 2. Calculation Phase
    for r in valid_results:
        name = r.get("feature_name", "unknown")
        score = r.get("score", 0)
        weight = r.get("weight") or FEATURE_WEIGHTS.get(name, 0.30)

        # Track statistics
        if score > max_score:
            max_score = score
            max_feature = name

        if score > 60:
            high_risk_count += 1

        weighted_sum += (score * weight)
        total_weight += weight

    base_avg = weighted_sum / total_weight if total_weight > 0 else 0

    # 3. Decision Phase (The Rewrite)
    final_score = base_avg

    # RULE A: The "Smoking Gun" Override
    # If a technical feature is 100% sure, we assume it's true regardless of average.
    if max_score >= 90 and max_feature in SMOKING_GUNS:
        logger.info(f"Smoking Gun Triggered: {max_feature}")
        final_score = max(final_score, 95.0)

    # RULE B: The "Corroboration" Override
    # If Threat Intel says "Bad" (100), but everything else says "Safe" (0),
    # and we have no other high risks, we trust the average (which will be low),
    # or cap the max score.
    elif max_score >= 80:
        if high_risk_count >= 2:
            # We have corroboration (e.g., Threat Intel + Bad SSL)
            # Boost the score towards the max
            final_score = max(base_avg, max_score * 0.90)
        else:
            # Isolated incident (False Positive protection)
            # We dampen the single high score.
            logger.info(f"Uncorroborated High Score: {max_feature}. Dampening.")
            # Average of (Base, Max) helps pull it up but not to 100
            final_score = (base_avg + max_score) / 2

    return round(final_score, 2)
