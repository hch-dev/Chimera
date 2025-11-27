# Version_1/score_engine.py
from log import get_logger

logger = get_logger(__name__)


def evaluate_score(feature_results: list) -> float:
    """
    Final risk score:
    Weighted average of feature scores.
    If feature returns error -> ignored but logged.
    """

    total_weight = 0
    weighted_score = 0

    for r in feature_results:
        if r.get("error"):
            logger.warning(f"Feature error ignored: {r}")
            continue

        score = r.get("score")
        weight = r.get("weight", 1)

        if score is not None:
            weighted_score += score * weight
            total_weight += weight

    if total_weight == 0:
        logger.error("No usable features available. Returning score 0.")
        return 0

    final = round(weighted_score / total_weight, 2)
    logger.info(f"Computed final weighted score: {final}")
    return final
