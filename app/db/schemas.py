# app/db/schemas.py
from datetime import datetime
from pydantic import BaseModel
from typing import Optional

class LogCreate(BaseModel):
    person_name: str
    tracking_id: int
    confidence_score: Optional[float]
    camera_id: str
    event_type: str
    snapshot_path: str
    timestamp: datetime