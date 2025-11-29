import sys
from log import LOG
from modules.static_scanner import StaticScanner
from modules.visual_scanner import VisualScanner
from utils.score_fusion import ScoreFusion

# Configuration
THRESHOLD = 50 

def run_analysis(url: str):
    """
    Orchestrates the V1 (static) and V2 (visual) analysis, fuses the scores, 
    and determines the final verdict.
    """
    # 1. Initialize Scanners
    static_scanner = StaticScanner()
    visual_scanner = VisualScanner()
    fusion = ScoreFusion()

    # 2. Run V1 Static Analysis (Heuristics)
    # The static scanner performs syntactic checks on the URL structure.
    v1_score, analysis_v1 = static_scanner.get_static_score(url)
    LOG.info(f"V1 Static Heuristic Score: {v1_score}/100")
    
    # 3. Run V2 Visual Analysis (CNN Model - Currently Mocked)
    # The visual scanner captures the page and analyzes its content layout.
    v2_score = visual_scanner.get_visual_score(url)
    LOG.info(f"V2 Visual CNN Score: {v2_score}/100")

    # 4. Fuse Scores
    # Weights: V1 (40%) and V2 (60%) for balanced detection
    final_score = fusion.fuse_scores(v1_score, v2_score, weight_v1=0.4, weight_v2=0.6)
    
    # 5. Determine Verdict
    if final_score >= THRESHOLD:
        verdict = "HIGH RISK - PHISHING"
    else:
        verdict = "LOW RISK - Legitimate"

    # 6. Report Results
    
    # Print the detailed analysis to the console
    print("\n" + "=" * 50)
    print(f"| URL Analyzed: {url}")
    print(f"| Final V2 Hybrid Score: {final_score}/100")
    print(f"| Verdict: {verdict}")
    print(f"| V1 Heuristic Score: {v1_score}")
    print(f"| V2 Visual CNN Score: {v2_score}")
    print("=" * 50 + "\n")

    # Log the final result
    LOG.info(f"ANALYSIS COMPLETE for {url}. Final Score: {final_score}, Verdict: {verdict}")


if __name__ == "__main__":
    
    target_url = None

    # Check for command-line argument (e.g., python main.py https://example.com)
    if len(sys.argv) > 1:
        target_url = sys.argv[1]
    else:
        # If no argument is provided, prompt the user for input
        print("\n--- Defender V2 Phishing Analysis ---")
        target_url = input("Please enter the URL to analyze: ").strip()

    if target_url:
        LOG.info(f"Starting Defender V2 Analysis on: {target_url}")
        run_analysis(target_url)
    else:
        print("No URL provided. Exiting analysis.")