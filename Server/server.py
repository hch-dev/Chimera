# Chimera/Server/server.py
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import Logic
from main import scan_url

app = FastAPI()

# --- 1. CORS (CRITICAL FOR EXTENSION) ---
# This allows your Chrome Extension to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all extensions/websites
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    uvicorn.run(app, host="0.0.0.0", port=port)
