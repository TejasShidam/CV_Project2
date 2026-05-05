from __future__ import annotations

from typing import Dict

from pydantic import BaseModel


class PredictResponse(BaseModel):
    plate_text: str
    track_id: int
    zone: str
    timestamp: str
    occupancy: Dict[str, int]


class OccupancyResponse(BaseModel):
    occupancy: Dict[str, int]


class LatencyResponse(BaseModel):
    count: float
    mean_ms: float
    p95_ms: float


class DriftResponse(BaseModel):
    js_divergence: float
