import sys
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- 1. Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
ai_module_path = os.path.join(project_root, "AI Models", "Defender_AI")

# Keep the insert(0) fix!
sys.path.insert(0, ai_module_path)

# --- 2. Import your AI Model ---
try:
    from main import analyze_data as run_ai_model
    print(f"✅ Successfully imported Defender_AI module from: {ai_module_path}")
except ImportError as e:
    print(f"❌ Error importing AI module: {e}")
    # Mock fallback must also be async now
    async def run_ai_model(data):
        return {"status": "mock_response", "confidence": 0.99}

# --- 3. App Setup ---
app = FastAPI(title="Chimera Backend", version="1.0.0")

# --- 4. CORS Policy ---
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. Data Models ---
class InputData(BaseModel):
    input_text: str

# --- 6. Routes ---
@app.get("/")
async def root():
    return {"message": "Chimera Server is running", "status": "online"}

@app.post("/analyze")
async def analyze(data: InputData):
    try:
        print(f"Received input: {data.input_text}")

        # CRITICAL FIX: Added 'await' here!
        result = await run_ai_model(data.input_text)

        return {"success": True, "result": result}

    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
