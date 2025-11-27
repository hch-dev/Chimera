from log import get_logger

logger = get_logger(__name__)

def evaluate_score(feature_results: list) -> float:
    """
    Final risk score calculation.
    Strategy: 'High-Water Mark with Weighted Average'

    1. Calculate Weighted Average (General Risk).
    2. Find Maximum Single Feature Score (Specific Risk).
    3. If Max Score > 80 (Critical Threat), boost the final score to match it.
       This prevents 'Safe' features (like SSL) from diluting a 'Critical' feature (like Homoglyph).
    """

    total_weight = 0
    weighted_sum = 0
    max_individual_score = 0

    for r in feature_results:
        if r.get("error"):
            logger.warning(f"Feature error ignored: {r}")
            continue

        score = r.get("score", 0)
        weight = r.get("weight", 1)

        # Track the highest single risk found
        if score > max_individual_score:
            max_individual_score = score

        weighted_sum += score * weight
        total_weight += weight

    if total_weight == 0:
        logger.error("No usable features available. Returning score 0.")
        return 0

    # Base calculation
    average_score = weighted_sum / total_weight

    # Critical Override Logic
    # If we found a "Smoking Gun" (Score > 80), the final verdict should reflect that.
    if max_individual_score > 80:
        # Boost the score closer to the max risk
        final = max(average_score, max_individual_score * 0.95)
    else:
        final = average_score

    final = round(final, 2)

    logger.info(f"Computed final score: {final} (Avg: {average_score:.2f}, Max: {max_individual_score})")
    return final
