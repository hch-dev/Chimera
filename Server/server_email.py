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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
email_logger = logging.getLogger("email_server")

# ==========================================
# 2. DYNAMIC PATH SETUP (Localhost Version)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
ai_model_path = os.path.join(project_root, "AI Models", "Defender_AI", "Version_3")

if os.path.exists(ai_model_path):
    if ai_model_path not in sys.path:
        sys.path.append(ai_model_path)
    email_logger.info(f"✅ AI Engine found at: {ai_model_path}")
else:
    email_logger.warning(f"⚠️ Direct path not found: {ai_model_path}")
    for root, dirs, files in os.walk(project_root):
        if "Version_3" in dirs:
            ai_model_path = os.path.join(root, "Version_3")
            sys.path.append(ai_model_path)
            email_logger.info(f"✅ AI Engine located via search at: {ai_model_path}")
            break

# ==========================================
# 3. IMPORT PREDICT MODULE
# ==========================================
try:
    import predict
    email_logger.info("✅ Module 'predict' imported successfully.")
except ImportError as e:
    email_logger.error(f"❌ Failed to import 'predict'. Error: {e}")
    predict = None

# ==========================================
# 4. APP SETUP
# ==========================================
app = FastAPI(title="Chimera Email Scanner (Local)", version="3.2-Local")

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
# 5. CORE LOGIC
# ==========================================
def run_ai_analysis(full_text):
    if predict is None:
        return 0.0, "System Error: Model not loaded"
    return predict.analyze_text(full_text)

async def scan_email_content(subject: str, body: str, sender: str):
    full_text_to_analyze = f"Subject: {subject}\nSender: {sender}\nBody: {body}"
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        score, verdict = await loop.run_in_executor(pool, run_ai_analysis, full_text_to_analyze)

    risk_level = "Low"
    if score > 0.75: risk_level = "Critical"
    elif score > 0.40: risk_level = "Medium"

    return {
        "score": round(score * 100, 2),
        "verdict": verdict,
        "risk_level": risk_level
    }

# ==========================================
# 6. ENDPOINTS
# ==========================================
@app.get("/")
def home():
    return {"status": "Chimera Local Server Running", "port": 5000, "ai_engine": "Active" if predict else "Missing"}

@app.post("/api/email/scan")
async def scan_email_endpoint(request: EmailRequest):
    try:
        if not request.body and not request.subject:
            raise HTTPException(status_code=400, detail="Email content cannot be empty.")

        # 1. Run the scan
        result = await scan_email_content(
            subject=request.subject,
            body=request.body,
            sender=request.sender
        )

        # 2. Log the success
        email_logger.info(f"Scan Success: {result['verdict']} ({result['score']}%)")

        # 3. Return data to Frontend
        return {"success": True, "data": result}

    except Exception as e:
        email_logger.error(f"Scan failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 STARTING CHIMERA EMAIL SERVER (Localhost:5000)")
    print("="*50)
    uvicorn.run(app, host="0.0.0.0", port=5000)
