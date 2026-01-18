from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class PredictionRequest(BaseModel):
    R: float
    W: float

class PredictionResponse(BaseModel):
    predicted_ppv: float
    min_range: Optional[float] = None
    max_range: Optional[float] = None
    features: Optional[Dict[str, float]] = None

class TrainingResponse(BaseModel):
    status: str
    message: str
    metrics: Optional[Dict[str, Any]] = None
    model_info: Optional[Dict[str, Any]] = None

class ModelStatusResponse(BaseModel):
    model_exists: bool
    model_info: Optional[Dict[str, Any]] = None

