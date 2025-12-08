import os
import sys
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- 1. SETUP PATH TO 'Version_1' ---
# This line is crucial. It tells Python: "Look inside the Version_1 folder for files."
sys.path.append(os.path.join(os.path.dirname(__file__), "Version_1"))

# --- 2. IMPORT MAIN.PY ---
# Now that we added the path, we can import 'main' directly.
try:
    import main  # This actually imports Version_1/main.py
    ai_loaded = True
    print("✅ Successfully loaded Version_1/main.py")
except ImportError as e:
    ai_loaded = False
    print(f"❌ Failed to load main.py: {e}")
    print("Make sure 'Version_1/main.py' exists and has no syntax errors.")

# --- 3. SERVER SETUP ---
app = FastAPI(title="Chimera Phishing Scanner")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow your frontend (GitHub Pages/Locally) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScanRequest(BaseModel):
    url: str

@app.get("/")
def home():
    return {
        "status": "online", 
        "ai_engine": "Version_1/main.py", 
        "ai_loaded": ai_loaded
    }

@app.post("/scan")
def scan_url(request: ScanRequest):
    logger.info(f"Scanning URL: {request.url}")

    if not ai_loaded:
        raise HTTPException(status_code=500, detail="AI Model (main.py) failed to load. Check server logs.")

    try:
        # --- 4. CALLING THE FUNCTION INSIDE MAIN.PY ---
        # IMPORTANT: I've added a check to find the right function name in your main.py.
        # It will try 'predict_url', 'analyze', or 'scan'.
        
        if hasattr(main, 'predict_url'):
            result = main.predict_url(request.url)
        elif hasattr(main, 'analyze'):
            result = main.analyze(request.url)
        elif hasattr(main, 'scan'):
            result = main.scan(request.url)
        else:
            # If your function is named something else (like 'check_phishing'), 
            # you will see this error.
            raise AttributeError("Could not find a recognized function (predict_url, analyze, scan) in main.py")

        return result

    except Exception as e:
        logger.error(f"Error during scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Standard Render Port Setup
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
