# backend/server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import the scan logic from the file above
from main import scan_url

app = FastAPI()

# --- CORS CONFIGURATION ---
# This is crucial. It allows your frontend (file:// or localhost:5500)
# to talk to this backend (localhost:8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (good for hackathons)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],
)

class ScanRequest(BaseModel):
    url: str

@app.get("/")
def health_check():
    return {"status": "Chimera System Online", "version": "1.0"}

@app.post("/scan")
async def api_scan(request: ScanRequest):
    try:
        # Run the scan using your existing logic
        result = await scan_url(request.url)

        # Transform it into the exact JSON structure your Frontend expects
        # (Based on mapBackendToUI in scanner.js)
        response = {
            "verdict": result["verdict"],
            "confidence": result["final_score"],
            "details": {}
        }

        # Re-map features to a dictionary keyed by feature name for easier JS access
        for feat in result["features"]:
            name = feat.get("feature_name")
            if name:
                response["details"][name] = feat

        return response

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
