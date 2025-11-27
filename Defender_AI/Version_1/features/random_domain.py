#Checks the entropy of domain generated

import math
import tldextract
from collections import Counter
from log import get_logger

logger = get_logger(__name__)

FEATURE_NAME = "random_domain_detection"
WEIGHT = 0.35

def calculate_entropy(text: str) -> float:
    if not text: return 0.0
    length = len(text)
    counts = Counter(text)
    entropy = 0.0
    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    return entropy

def extract(url: str, context: dict = None) -> dict:
    try:
        ext = tldextract.extract(url)
        domain = ext.domain.lower()

        if not domain:
             return {"feature_name": FEATURE_NAME, "score": 0, "weight": WEIGHT, "error": False, "message": "no_domain"}

        # Clean domain: remove hyphens/numbers to analyze linguistic structure
        clean_domain = domain.replace("-", "").replace(".", "")

        # 1. Calculate Entropy
        entropy = calculate_entropy(clean_domain)

        # 2. Calculate Consonant Ratio (Gibberish Check)
        vowels = set("aeiouy")
        consonants = sum(1 for c in clean_domain if c.isalpha() and c not in vowels)
        total_alpha = sum(1 for c in clean_domain if c.isalpha())

        consonant_ratio = consonants / total_alpha if total_alpha > 0 else 0

        # --- SCORING LOGIC ---
        score = 0
        flags = []

        # A. Entropy Scoring (Dynamic)
        # Normal English domains are usually 2.5 - 3.5
        # Random strings are usually > 3.8.
        # Adjusted threshold: > 3.8 is suspicious, > 4.0 is critical.
        if entropy > 4.0:
            score += 85
            flags.append(f"high_entropy_{entropy:.2f}")
        elif entropy > 3.7:
             # Map 3.7 -> 4.0 to score 40 -> 80
            entropy_score = 40 + (entropy - 3.7) * 130
            score += entropy_score
            flags.append(f"elevated_entropy_{entropy:.2f}")

        # B. Gibberish Scoring (Consonants)
        # English is usually ~60% consonants. >80% is suspicious.
        if consonant_ratio > 0.80:
             ratio_score = (consonant_ratio - 0.80) * 300 # Steeper penalty
             score += ratio_score
             flags.append(f"gibberish_{consonant_ratio:.2f}")

        # C. Length Penalty
        # Long strings with high entropy are worse than short ones
        if len(domain) > 12 and entropy > 3.8:
            score += 20
            flags.append("long_random")

        # Cap score at 100
        score = min(100, max(0, score))

        # Verdict Threshold
        if score > 40:
            return {
                "feature_name": FEATURE_NAME,
                "score": int(score),
                "weight": WEIGHT,
                "error": False,
                "message": ", ".join(flags)
            }

        return {
            "feature_name": FEATURE_NAME,
            "score": 0,
            "weight": WEIGHT,
            "error": False,
            "message": f"normal_linguistics (E={entropy:.2f})"
        }

    except Exception as e:
        logger.exception(f"Random domain check failed: {e}")
        return {"feature_name": FEATURE_NAME, "score": None, "weight": WEIGHT, "error": True, "message": str(e)}
