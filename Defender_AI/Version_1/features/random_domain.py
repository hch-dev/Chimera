#Checks the entropy of domain generated

import math
import tldextract
from collections import Counter
from log import get_logger

logger = get_logger(__name__)

FEATURE_NAME = "random_domain_detection"
WEIGHT = 0.35

def calculate_entropy(text: str) -> float:
    """
    Calculates Shannon Entropy.
    Higher value = more random (DGA-like).
    Normal English words usually have entropy < 3.5.
    Random strings often have entropy > 3.8 or 4.0.
    """
    if not text:
        return 0.0

    length = len(text)
    counts = Counter(text)

    entropy = 0.0
    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy

def extract(url: str, context: dict = None) -> dict:
    """
    Detects Algorithmically Generated Domains (DGA) using Entropy Analysis.
    Also checks for high consonant-to-vowel ratios.
    """
    try:
        ext = tldextract.extract(url)
        domain = ext.domain.lower()

        if not domain:
             return {
                "feature_name": FEATURE_NAME,
                "score": 0,
                "weight": WEIGHT,
                "error": False,
                "message": "no_domain"
            }

        # 1. Remove hyphens/numbers to focus on linguistic randomness
        # Some legit domains use numbers (123-reg), so we treat them carefully.
        clean_domain = domain.replace("-", "")

        # 2. Calculate Entropy
        entropy = calculate_entropy(clean_domain)

        # 3. Consonant/Vowel Analysis (Gibberish Detector)
        vowels = set("aeiouy")
        consonants = 0
        for char in clean_domain:
            if char.isalpha() and char not in vowels:
                consonants += 1

        total_chars = len([c for c in clean_domain if c.isalpha()])
        if total_chars > 0:
            consonant_ratio = consonants / total_chars
        else:
            consonant_ratio = 0

        # --- SCORING LOGIC ---
        score = 0
        flags = []

        # Thresholds based on common DGA research
        if entropy > 4.2:
            score += 80
            flags.append(f"high_entropy_{entropy:.2f}")
        elif entropy > 3.8:
            score += 40
            flags.append(f"elevated_entropy_{entropy:.2f}")

        # Gibberish check: Too many consonants (e.g. "zbxght")
        if consonant_ratio > 0.85:
             score += 50
             flags.append(f"gibberish_consonants_{consonant_ratio:.2f}")

        # Length check: Very long random domains are suspicious
        if len(domain) > 25 and entropy > 3.5:
            score += 30
            flags.append("long_random_string")

        # Cap score
        score = min(100, score)

        if score > 0:
            return {
                "feature_name": FEATURE_NAME,
                "score": score,
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
