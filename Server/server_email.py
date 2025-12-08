import uvicorn
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Import the logic we just wrote
from main_email import scan_email_content
from logger_email import email_logger

app = FastAPI(title="Chimera Email Scanner", version="3.0")

# --- 1. CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your frontend (and ngrok) to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. DATA MODELS ---
class EmailRequest(BaseModel):
    subject: str = ""
    body: str
    sender: str = ""
    headers: Optional[str] = ""

# --- 3. ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "Chimera Email Defender V3 is Running"}

@app.post("/api/email/scan")
async def scan_email_endpoint(request: EmailRequest):
    """
    Receives email data, runs AI analysis, and returns risk assessment.
    """
    try:
        email_logger.info(f"Received scan request. Subject len: {len(request.subject)}")

        if not request.body and not request.subject:
            raise HTTPException(status_code=400, detail="Email content cannot be empty.")

        # Call the logic from main_email.py
        result = await scan_email_content(
            subject=request.subject,
            body=request.body,
            sender=request.sender
        )

        email_logger.info(f"Scan complete. Verdict: {result['verdict']} (Score: {result['score']})")

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        email_logger.error(f"Scanning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Running on Port 4000 as requested
    print("🚀 Starting Email Defender Server on Port 4000...")
    uvicorn.run(app, host="0.0.0.0", port=4000)
