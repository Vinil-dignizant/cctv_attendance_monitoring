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
class DailySummaryBase(BaseModel):
    person_name: str
    date: date
    camera_id: str  # New field
    first_login: Optional[datetime] = None
    last_logout: Optional[datetime] = None

class DailySummaryCreate(DailySummaryBase):
    pass

class DailySummary(DailySummaryBase):
    id: int
    total_logins: int
    total_logouts: int
    working_hours: Optional[timedelta] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True