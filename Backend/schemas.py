from pydantic import BaseModel
from typing import Dict, Any, Optional, List

class ScanRequest(BaseModel):
    url: str

class FeatureResult(BaseModel):
    feature_name: str
    score: float
    weight: float
    error: bool = False
    message: Optional[str] = None

class ScanResponse(BaseModel):
    url: str
    verdict: str
    confidence: float
    risk_level: str
    details: Dict[str, FeatureResult]
