<<<<<<< HEAD
# Chimera/Server/server.py
=======
>>>>>>> 0765b8e5d541abec14b860f2e2f8b531960add38
import os
import sys
import uvicorn
<<<<<<< HEAD
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import Logic
from main import scan_url

app = FastAPI()

# --- 1. CORS (CRITICAL FOR EXTENSION) ---
# This allows your Chrome Extension to talk to this server
=======
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
>>>>>>> 0765b8e5d541abec14b860f2e2f8b531960add38
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all extensions/websites
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

<<<<<<< HEAD
# --- 2. API ENDPOINTS ---
class ScanRequest(BaseModel):
    url: str

@app.get("/api/status")
def health_check():
    return {"status": "Chimera Server Online", "mode": "Unified Server"}

@app.post("/scan")
async def api_scan(request: ScanRequest):
    try:
        # Run the AI Scan
        result = await scan_url(request.url)

        # Format for Frontend/Extension
        response = {
            "verdict": result["verdict"],
            "confidence": result["final_score"],
            "details": {}
        }

        for feat in result["features"]:
            name = feat.get("feature_name")
            if name:
                response["details"][name] = feat

        return response

    except Exception as e:
        print(f"❌ Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 3. SERVE FRONTEND ---
# Locates the 'Website/Frontend' folder relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../Server
project_root = os.path.dirname(current_dir)              # .../Chimera
frontend_path = os.path.join(project_root, "Website", "Frontend")

if os.path.exists(frontend_path):
    print(f"📂 Hosting Website from: {frontend_path}")
    # This serves index.html at http://localhost:5000/
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
else:
    print(f"⚠️ FRONTEND NOT FOUND AT: {frontend_path}")
    print("   Server running in API-Only mode.")

if __name__ == "__main__":
    # Use the PORT environment variable provided by Render, or default to 5000
    port = int(os.environ.get("PORT", 5000))
=======
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
            result = main.run_feature(request.url)
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
>>>>>>> 0765b8e5d541abec14b860f2e2f8b531960add38
    uvicorn.run(app, host="0.0.0.0", port=port)
