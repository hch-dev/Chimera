import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Import the AI logic (we will create this file next)
import predict

# Initialize App
app = FastAPI(title="Chimera URL Scanner")

# --- CRITICAL: FIX CORS ---
# This allows your frontend to talk to this backend without "Access Denied" errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request Model
class UrlRequest(BaseModel):
    url: str

@app.get("/")
def home():
    return {"message": "Chimera URL Scanner is Online", "status": "active"}

@app.post("/scan")
def scan_url(request: UrlRequest):
    logger.info(f"Scanning URL: {request.url}")
    
    try:
        # Call the AI analysis function from predict.py
        result = predict.analyze_url(request.url)
        return result
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Use Render's PORT variable, default to 10000 locally
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
