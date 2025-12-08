import sys
import os
import uvicorn
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ==========================================
# 1. LOGGING CONFIGURATION
# ==========================================
# We configure a simple logger here since 'logger_email' might not exist
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
email_logger = logging.getLogger("email_server")

# ==========================================
# 2. DYNAMIC PATH SETUP (The Critical Fix)
# ==========================================
# Current: .../Chimera/Server/server_email.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Try to find the Project Root ('Chimera') by going up one level
#    If the folder is named 'Server', the parent is 'Chimera' (or the repo root)
project_root = os.path.dirname(current_dir)

# 2. Construct path to AI Models
#    Target: .../Chimera/AI Models/Defender_AI/Version_3
ai_model_path = os.path.join(project_root, "AI Models", "Defender_AI", "Version_3")

# 3. Verify and Append to System Path
if os.path.exists(ai_model_path):
    if ai_model_path not in sys.path:
        sys.path.append(ai_model_path)
    email_logger.info(f"✅ AI Engine found at: {ai_model_path}")
else:
    # Fallback: Search recursively if directory structure is slightly different
    email_logger.warning(f"⚠️ Direct path not found: {ai_model_path}")
    found = False
    for root, dirs, files in os.walk(project_root):
        if "Version_3" in dirs:
            ai_model_path = os.path.join(root, "Version_3")
            sys.path.append(ai_model_path)
            email_logger.info(f"✅ AI Engine located via search at: {ai_model_path}")
            found = True
            break
    if not found:
        email_logger.error(f"❌ CRITICAL: Could not find 'Version_3' folder in {project_root}")

# ==========================================
# 3. IMPORT PREDICT MODULE
# ==========================================
try:
    import predict
    email_logger.info("✅ Module 'predict' imported successfully.")
except ImportError as e:
    email_logger.error("❌ Failed to import 'predict'. Check if __init__.py exists or path is correct.")
    # We don't exit here to allow the server to start in 'Diagnostic Mode' if needed
    predict = None

# ==========================================
# 4. APP & DATA MODELS
# ==========================================
app = FastAPI(title="Chimera Email Scanner", version="3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmailRequest(BaseModel):
    subject: str = ""
    body: str
    sender: str = ""
    headers: Optional[str] = ""

# ==========================================
# 5. CORE LOGIC (Merged from main_email.py)
# ==========================================
def run_ai_analysis(full_text):
    """Wrapper to run the AI model safely."""
    if predict is None:
        return 0.0, "System Error: Model not loaded"
    return predict.analyze_text(full_text)

async def scan_email_content(subject: str, body: str, sender: str):
    full_text_to_analyze = f"Subject: {subject}\nSender: {sender}\nBody: {body}"
    email_logger.info(f"🔍 Analyzing email from: {sender}")

    # Run blocking AI inference in a separate thread
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        score, verdict = await loop.run_in_executor(pool, run_ai_analysis, full_text_to_analyze)

    # Risk Logic
    risk_level = "Low"
    if score > 0.75:
        risk_level = "Critical"
    elif score > 0.40:
        risk_level = "Medium"

    return {
        "score": round(score * 100, 2),
        "verdict": verdict,
        "risk_level": risk_level,
        "analysis_summary": {
            "analyzed_length": len(full_text_to_analyze),
            "model_version": "V3 (RoBERTa)"
        }
    }

# ==========================================
# 6. ENDPOINTS
# ==========================================
@app.get("/")
def home():
    status = "Active" if predict else "Model Missing"
    return {"status": "Chimera Email Defender V3 Running", "ai_engine": status}

@app.post("/api/email/scan")
async def scan_email_endpoint(request: EmailRequest):
    try:
        if not request.body and not request.subject:
            raise HTTPException(status_code=400, detail="Email content cannot be empty.")

        result = await scan_email_content(
            subject=request.subject,
            body=request.body,
            sender=request.sender
        )
        
        email_logger.info(f"Scan Finished: {result['verdict']} ({result['score']}%)")
        return {"success": True, "data": result}

    except Exception as e:
        email_logger.error(f"Scan failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 7. SERVER START
# ==========================================
if __name__ == "__main__":
    # Render provides the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting Server on Port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
