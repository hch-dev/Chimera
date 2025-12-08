import uvicorn
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize the App
app = FastAPI(title="Chimera Email Server")

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("==================================================")
print("🚀 STARTING CHIMERA EMAIL SERVER")
print("==================================================")

# --- 1. Attempt to load the AI Module ---
# Your logs showed this failed because 'torch' was missing.
# We wrap it in try/except so the server doesn't crash immediately if dependencies are missing.
try:
    import torch
    import predict # This assumes you have a predict.py file in the same folder
    logger.info("✅ AI Engine loaded successfully")
    ai_available = True
except ImportError as e:
    logger.error(f"❌ Failed to import AI modules: {e}")
    logger.warning("Server will start, but scanning will be disabled until 'torch' is installed.")
    ai_available = False

# --- 2. Define Request Model ---
# This defines what data you expect to receive on the /scan endpoint
class ScanRequest(BaseModel):
    url: str
    # Add other fields here if your AI needs them (e.g., email_body, sender)

# --- 3. Routes ---

@app.get("/")
def home():
    """Health Check Endpoint"""
    return {
        "status": "online",
        "message": "Chimera Phishing Detection Server is Running",
        "ai_status": "Active" if ai_available else "Disabled (Missing Dependencies)"
    }

@app.post("/scan")
def scan_url(request: ScanRequest):
    """The Missing Endpoint that caused the 404 error"""
    if not ai_available:
        raise HTTPException(status_code=503, detail="AI Engine is not loaded. Check server logs.")
    
    try:
        # CALL YOUR AI FUNCTION HERE
        # Assuming predict.py has a function like 'predict_phishing(url)'
        # You might need to adjust 'predict.analyze' to whatever your function is actually named.
        result = predict.analyze(request.url) 
        
        return {"url": request.url, "result": result, "status": "scanned"}
    except Exception as e:
        logger.error(f"Error during scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 4. Server Startup ---
if __name__ == "__main__":
    # CRITICAL: Render provides the PORT variable. Default to 10000 locally.
    port = int(os.environ.get("PORT", 10000))
    
    # Run Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
