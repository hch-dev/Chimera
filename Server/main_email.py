import sys
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

# --- 1. PATH SETUP ---
# Navigate from: .../Chimera/Website/Backend/
# To: ........... .../Chimera/AI Models/Defender_AI/Version_3/

current_dir = os.path.dirname(os.path.abspath(__file__))

# Move up to 'Chimera' root.
# Adjusting logic to ensure we find the "AI Models" folder irrespective of exact depth
root_path = current_dir
while os.path.basename(root_path) != "Chimera" and os.path.dirname(root_path) != root_path:
    root_path = os.path.dirname(root_path)

# Construct target path
model_path = os.path.join(root_path, "AI Models", "Defender_AI", "Version_3")

# Add to system path
if model_path not in sys.path:
    sys.path.append(model_path)
    print(f"🔗 Connected to Version 3 Engine at: {model_path}")

# --- 2. IMPORT MODEL ---
try:
    # Importing 'analyze_text' from the predict.py file you uploaded earlier
    from predict import analyze_text
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Could not import 'predict.py'.")
    print(f"   Ensure 'predict.py' is in: {model_path}")
    print(f"   Error details: {e}")
    sys.exit(1)

# --- 3. SCANNING LOGIC ---
async def scan_email_content(subject: str, body: str, sender: str):
    """
    Combines email fields and runs them through the RoBERTa model.
    """
    # 1. Combine fields for the AI
    # The model was trained on text, so we concatenate the most relevant parts.
    full_text_to_analyze = f"Subject: {subject}\nSender: {sender}\nBody: {body}"

    print(f"🔍 Analyzing Email from: {sender}")

    # 2. Run the synchronous model in a thread (to avoid blocking the server)
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        # analyze_text returns (score, verdict_string)
        score, verdict = await loop.run_in_executor(pool, analyze_text, full_text_to_analyze)

    # 3. Refine the verdict for the frontend
    # The model returns raw probability (0.0 to 1.0)

    risk_level = "Low"
    if score > 0.75:
        risk_level = "Critical"
    elif score > 0.40:
        risk_level = "Medium"

    return {
        "score": round(score * 100, 2), # Convert 0.95 -> 95.0
        "verdict": verdict,             # e.g., "Safe", "Phishing"
        "risk_level": risk_level,
        "analysis_summary": {
            "analyzed_length": len(full_text_to_analyze),
            "model_version": "V3 (RoBERTa)"
        }
    }

if __name__ == "__main__":
    # simple CLI test
    print("--- Email Logic Test ---")
    res = asyncio.run(scan_email_content("Test Subject", "Click this suspicious link now!", "hacker@bad.com"))
    print(res)
