import os
import sys
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- CRITICAL: SETUP PATH TO Version_1 ---
# This tells Python to look inside the 'Version_1' folder for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "Version_1"))

# Try to import the main.py from Version_1
try:
    import main  # This looks for Version_1/main.py
    ai_loaded = True
    print("✅ Successfully loaded Version_1/main.py")
except ImportError as e:
    ai_loaded = False
    print(f"❌ Failed to load AI Model: {e}")

# Initialize App
app = FastAPI(title="Chimera Phishing Scanner")

# Setup CORS (Allows your frontend to talk to this backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UrlRequest(BaseModel):
    url: str

@app.get("/")
def home():
    return {
        "status": "online", 
        "ai_engine": "Version_1/main.py", 
        "loaded": ai_loaded
    }

@app.post("/scan")
def scan_url(request: UrlRequest):
    logger.info(f"Scanning URL: {request.url}")

    if not ai_loaded:
        raise HTTPException(status_code=500, detail="AI Model (Version_1/main.py) failed to load.")

    try:
        # --- CALLING YOUR AI FUNCTION ---
        # IMPORTANT: Check the actual function name inside your main.py!
        # I am assuming the function is named 'predict_url' or 'scan'. 
        # If your function is named 'analyze', change the line below to: result = main.analyze(request.url)
        
        if hasattr(main, 'predict_url'):
            result = main.predict_url(request.url)
        elif hasattr(main, 'analyze'):
            result = main.analyze(request.url)
        elif hasattr(main, 'scan'):
            result = main.scan(request.url)
        else:
            raise AttributeError("Could not find a function named 'predict_url', 'analyze', or 'scan' in main.py")

        return result

    except Exception as e:
        logger.error(f"Error during scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
