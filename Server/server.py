import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize the App
app = FastAPI()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- REQUEST MODEL ---
class ScanRequest(BaseModel):
    url: str

# --- ROUTES ---

@app.get("/")
def home():
    return {"status": "online", "message": "Chimera Backend is Running"}

@app.post("/scan")
def scan_url(request: ScanRequest):
    """
    This is the endpoint your frontend is trying to reach.
    """
    logger.info(f"Received scan request for: {request.url}")
    
    try:
        # ---------------------------------------------------------
        # AI INTEGRATION SECTION
        # ---------------------------------------------------------
        # If you have your 'predict.py' file, uncomment the lines below:
        # import predict
        # result = predict.analyze(request.url)
        
        # FOR NOW: We return a dummy response so the 404 error goes away.
        # Once you have your AI code back, replace this line with the real logic.
        result = "AI Scan Placeholder - Logic Missing"
        
        return {
            "url": request.url, 
            "result": result, 
            "status": "safe" # or "phishing"
        }
    except Exception as e:
        logger.error(f"Error processing scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- SERVER STARTUP ---
if __name__ == "__main__":
    # Get the PORT from Render (default to 10000)
    port = int(os.environ.get("PORT", 10000))
    # Listen on 0.0.0.0 (Required for Render)
    uvicorn.run(app, host="0.0.0.0", port=port)
