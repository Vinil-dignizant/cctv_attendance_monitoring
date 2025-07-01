# models.py
from sqlalchemy import Column, Integer, String, Float, Boolean, Date, DateTime, Interval, func
from sqlalchemy import UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from app.db.database import Base
from sqlalchemy.sql import expression
from datetime import datetime
import pytz

Base = declarative_base()

class AttendanceLog(Base):
    __tablename__ = 'attendance_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    person_name = Column(String(100), nullable=False)
    tracking_id = Column(Integer)
    confidence_score = Column(Float(5,4))
    camera_id = Column(String(50), nullable=False)
    event_type = Column(String(20), nullable=False, default='login')
    snapshot_path = Column(String(255))
    timestamp = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Person(Base):
    __tablename__ = 'persons'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    email = Column(String(150))
    department = Column(String(100))
    employee_id = Column(String(50))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class Camera(Base):
    __tablename__ = 'cameras'
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(50), unique=True, nullable=False)
    camera_name = Column(String(100))
    location = Column(String(100))
    url = Column(String)
    event_type = Column(String(20), default='login')
    is_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# class DailySummary(Base):
#     __tablename__ = 'daily_summary'
    
#     id = Column(Integer, primary_key=True, index=True)
#     person_name = Column(String(100), nullable=False)
#     date = Column(Date, nullable=False)
#     first_login = Column(DateTime(timezone=True))
#     last_logout = Column(DateTime(timezone=True))
#     total_logins = Column(Integer, default=0)
#     total_logouts = Column(Integer, default=0)
#     working_hours = Column(Interval)
#     created_at = Column(DateTime(timezone=True), server_default=func.now())
#     updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# whithout camera id
# ==================================================

class DailySummary(Base):
    __tablename__ = 'daily_summary'
    
    id = Column(Integer, primary_key=True, index=True)
    person_name = Column(String(100), nullable=False)
    date = Column(Date, nullable=False)
    camera_id = Column(String(50), nullable=False)  # New field
    first_login = Column(DateTime(timezone=True))
    last_logout = Column(DateTime(timezone=True))
    total_logins = Column(Integer, default=0)
    total_logouts = Column(Integer, default=0)
    working_hours = Column(Interval)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Add composite unique constraint (optional)
    __table_args__ = (
        UniqueConstraint('person_name', 'date', 'camera_id', name='unique_person_date_camera'),
    )

#create a image table
class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, index=True)
    image_url = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func
                        .now())
    # create a table for camera
