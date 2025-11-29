from log import LOG

class ScoreFusion:
    """
    Fuses the scores from the Static Scanner (V1) and Visual Scanner (V2)
    to produce a final, weighted risk score.
    """

    def __init__(self):
        LOG.info("ScoreFusion initialized.")

    def fuse_scores(self, score_v1: int, score_v2: int, weight_v1: float = 0.4, weight_v2: float = 0.6) -> int:
        """
        Calculates the weighted average of the two scores.
        
        :param score_v1: Risk score from Static Scanner (0-100).
        :param score_v2: Risk score from Visual Scanner (0-100).
        :param weight_v1: Weight for V1 score (must be between 0 and 1).
        :param weight_v2: Weight for V2 score (must be between 0 and 1).
        :return: Final hybrid risk score (0-100).
        """
        
        # Input validation for weights
        if weight_v1 + weight_v2 == 0:
            LOG.error("Weights cannot both be zero. Defaulting to equal weights (0.5, 0.5).")
            weight_v1 = 0.5
            weight_v2 = 0.5
        elif weight_v1 + weight_v2 != 1.0:
            LOG.warning("Weights do not sum to 1. Scores will be normalized based on provided weights.")
            
        try:
            # Weighted average formula: (W1*S1 + W2*S2) / (W1 + W2)
            final_score_raw = (score_v1 * weight_v1 + score_v2 * weight_v2) / (weight_v1 + weight_v2)
            
            # The score is rounded to the nearest integer and clamped between 0 and 100
            final_score = int(round(max(0, min(100, final_score_raw))))
            
            LOG.debug(f"Scores fused: V1={score_v1} (W={weight_v1}), V2={score_v2} (W={weight_v2}) -> Final={final_score}")
            return final_score
            
        except Exception as e:
            LOG.error(f"Error during score fusion: {e}")
            return max(score_v1, score_v2) # Return the max score as a safe fallback