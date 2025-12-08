import os
import sys
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- 1. SETUP AI PATH ---
# This adds the 'Version_1' folder to the system path so we can import 'main.py'
sys.path.append(os.path.join(os.path.dirname(__file__), "Version_1"))

# Try to import the AI logic
try:
    import main as ai_model  # Imports Version_1/main.py
    ai_loaded = True
    print("✅ AI Engine (Version_1/main.py) loaded successfully.")
except ImportError as e:
    ai_loaded = False
    print(f"❌ Failed to load AI Engine: {e}")

# --- 2. INITIALIZE APP ---
app = FastAPI(title="Chimera Phishing Scanner")

# --- 3. CORS SETTINGS (CRITICAL) ---
# This allows your frontend JS to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all connections
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the data format expecting from JS
class ScanRequest(BaseModel):
    url: str

# --- 4. ROUTES ---

@app.get("/")
def home():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Chimera Backend is Running",
        "ai_status": "Loaded" if ai_loaded else "Failed"
    }

@app.post("/scan")
def scan_url(request: ScanRequest):
    """
    This is the endpoint your scanner.js calls.
    It takes the URL and passes it to your AI model.
    """
    logger.info(f"Received scan request for: {request.url}")

    # Check if AI loaded correctly
    if not ai_loaded:
        logger.error("AI model not loaded. Returning dummy data.")
        # If AI is missing, return error or dummy data so app doesn't crash
        raise HTTPException(status_code=503, detail="AI Engine is not active on server.")

    try:
        # CALL YOUR AI FUNCTION
        # Adjust 'predict_url' to whatever the function is named in Version_1/main.py
        # Common names: predict(), analyze(), scan(), predict_url()
        
        if hasattr(ai_model, 'predict_url'):
            result = ai_model.predict_url(request.url)
        elif hasattr(ai_model, 'analyze'):
            result = ai_model.analyze(request.url)
        elif hasattr(ai_model, 'scan'):
            result = ai_model.scan(request.url)
        else:
            # Fallback if we can't find the specific function
            raise AttributeError("Function 'predict_url' or 'analyze' not found in main.py")
            
        return result

    except Exception as e:
        logger.error(f"Scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. STARTUP ---
if __name__ == "__main__":
    # Render assigns the PORT. Default to 10000 if testing locally.
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
