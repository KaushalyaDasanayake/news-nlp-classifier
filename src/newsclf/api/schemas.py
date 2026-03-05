# Pydantic models for API

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., description="News article text")
class PredictResponse(BaseModel):
    label: str
    confidence: float
    request_id: str