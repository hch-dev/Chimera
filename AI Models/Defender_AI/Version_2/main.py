import sys
import os

# --- 1. PATH FIX (Crucial for Server Integration) ---
# This ensures Python can find 'log.py' and 'modules' inside Version_2
# even when the server runs from a different folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# --- 2. IMPORTS ---
try:
    from log import LOG
    from modules.visual_scanner import VisualScanner
    # Note: We REMOVED StaticScanner and ScoreFusion (Standalone Mode)
except ImportError as e:
    # Fallback logger if file is missing during setup
    print(f"⚠️ V2 Import Warning: {e}")
    class DummyLogger:
        def info(self, msg): print(f"[V2] {msg}")
        def error(self, msg): print(f"[V2 ERROR] {msg}")
    LOG = DummyLogger()
    VisualScanner = None

# --- 3. STANDALONE ENGINE LOGIC ---
def predict_url(url: str):
    """
    Runs ONLY the Visual Analysis (CNN) on the URL.
    """
    LOG.info(f"--- Starting Standalone V2 Analysis on: {url} ---")

    try:
        if not VisualScanner:
            raise ImportError("VisualScanner module is missing.")

        # Initialize Visual Scanner (Selenium + TensorFlow)
        scanner = VisualScanner()

        # Run Prediction
        # We only use the CNN score now. No weighting, no fusion.
        visual_score = scanner.get_visual_score(url)

        # Determine Verdict (Simple Threshold)
        # Assuming score is 0-100.
        verdict = "SAFE"
        if visual_score > 50:
            verdict = "PHISHING"

        LOG.info(f"Analysis Complete. Score: {visual_score}, Verdict: {verdict}")

        return {
            "version": "v2_visual_standalone",
            "verdict": verdict,
            "confidence": visual_score,
            "details": {
                "scan_type": "Visual CNN",
                "image_analysis_score": visual_score
            }
        }

    except Exception as e:
        LOG.error(f"Scan Failed: {e}")
        return {
            "version": "v2_visual_standalone",
            "verdict": "ERROR",
            "confidence": 0,
            "details": {"error": str(e)}
        }

# --- 4. CLI TESTER (Optional) ---
if __name__ == "__main__":
    test_url = input("Enter URL for Visual Scan: ")
    print(predict_url(test_url))
