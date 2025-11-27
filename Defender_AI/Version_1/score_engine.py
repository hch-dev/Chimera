from log import get_logger

logger = get_logger(__name__)

# Centralized Configuration for Feature Weights
# Higher Weight = This feature has a stronger say in the final vote.
FEATURE_WEIGHTS = {
    # --- CRITICAL (The "Smoking Guns") ---
    "homoglyph_impersonation": 0.60,       # Visual deception (User Request)
    "open_redirect_detection": 0.55,       # Chain abuse (User Request)
    "ssl_presence_and_validity": 0.55,     # Security posture (User Request)
    "data_uri_scheme": 0.60,               # Protocol abuse (Almost always malicious)
    "threat_intelligence": 0.60,           # Confirmed by global community
    "url_structure_analysis": 0.55,        # Authority abuse (@ symbol)

    # --- HIGH (Strong Indicators) ---
    "favicon_mismatch": 0.45,              # Brand impersonation
    "domain_abuse_detection": 0.45,        # Combosquatting/Shadowing
    "obfuscation_analysis": 0.45,          # Hiding intent (Base64/Hex)

    # --- MEDIUM (Contextual/Infrastructure) ---
    "fast_flux_dns": 0.35,                 # Evasive infrastructure
    "domain_age_analysis": 0.35,           # "Born Yesterday"
    "path_anomaly_detection": 0.35,        # Compromised site patterns
    "random_domain_detection": 0.30,       # DGA/Gibberish
}

def evaluate_score(feature_results: list) -> float:
    """
    Calculates the Final Phishing Risk Score (0-100).

    Algorithm: 'Weighted Average with Critical Override'
    1. Calculate the weighted average of all active features.
    2. Identify the single highest risk score found (The High-Water Mark).
    3. If the High-Water Mark is Critical (>80), boost the final score.
       This prevents 'Safe' features (like a valid SSL on a homoglyph domain)
       from diluting the detection of a confirmed threat.
    """

    total_weight = 0
    weighted_score_sum = 0
    max_individual_score = 0
    critical_trigger = None

    for r in feature_results:
        # Skip errored features so they don't drag down the score
        if r.get("error"):
            logger.warning(f"Skipping feature due to error: {r.get('feature_name')}")
            continue

        name = r.get("feature_name", "unknown")
        raw_score = r.get("score", 0)

        # Use the weight from the result, or fallback to our central config, or default to 0.1
        weight = r.get("weight") or FEATURE_WEIGHTS.get(name, 0.30)

        # Track the "Smoking Gun"
        if raw_score > max_individual_score:
            max_individual_score = raw_score
            critical_trigger = name

        weighted_score_sum += raw_score * weight
        total_weight += weight

    # Avoid division by zero if all features failed
    if total_weight == 0:
        logger.error("No usable features available. Returning score 0.")
        return 0.0

    # 1. Base Calculation
    base_score = weighted_score_sum / total_weight

    # 2. Critical Override Logic
    # If we found a critical threat (Score > 85), the final score must be High.
    # We allow a small damping factor (0.98) but we do NOT let the average pull it down to "Safe".
    if max_individual_score >= 85:
        final = max(base_score, max_individual_score * 0.98)
        logger.info(f"CRITICAL OVERRIDE triggered by {critical_trigger} (Score: {max_individual_score})")

    # If we found a High threat (Score > 65), ensure we don't drop below Suspicious levels.
    elif max_individual_score >= 65:
        final = max(base_score, max_individual_score * 0.85)

    else:
        final = base_score

    final = round(final, 2)

    logger.info(f"Scoring Complete. Base: {base_score:.2f} | Max: {max_individual_score} | Final: {final}")
    return final
