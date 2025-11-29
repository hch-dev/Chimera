# Backend/server.py
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the scan logic from main.py
# This requires main.py to be in the same directory
from main import scan_url

app = FastAPI()

# --- CORS CONFIGURATION ---
# This allows your frontend (running on port 5500 or file://) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (simplest for hackathons)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],
)

# --- REQUEST MODELS ---
class ScanRequest(BaseModel):
    url: str

# --- ROUTES ---

@app.get("/")
def health_check():
    """
    Simple check to see if the server is running.
    """
    return {"status": "Chimera System Online", "version": "1.0"}

@app.post("/scan")
async def api_scan(request: ScanRequest):
    """
    Receives a URL, runs the full analysis, and returns the JSON report.
    """
    try:
        print(f"📡 API Received Scan Request: {request.url}")

        # Run the scan using the logic in main.py
        result = await scan_url(request.url)

        # Transform the result into the format the Frontend expects
        response = {
            "verdict": result["verdict"],
            "confidence": result["final_score"],
            "details": {}
        }

        # Re-map the list of features to a dictionary keyed by feature_name
        # This makes it easier for the frontend to access specific scores (e.g., data.details.ssl_presence_and_validity)
        for feat in result["features"]:
            name = feat.get("feature_name")
            if name:
                response["details"][name] = feat

        return response

    except Exception as e:
        print(f"❌ Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Runs the server on localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
