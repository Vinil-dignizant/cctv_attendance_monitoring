# app/db/crud.py
from sqlalchemy.orm import Session
import csv
from datetime import datetime
from pathlib import Path
import pytz
from . import models
from .database import engine, SessionLocal  # Add this import
from typing import Optional, List, Dict, Any

def init_db():
    """Initialize database tables"""
    try:
        models.Base.metadata.create_all(bind=engine)
        print("[INFO] Database initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize database: {e}")
        raise

def insert_log(
    db: Session,
    person_name: str,
    tracking_id: int,
    confidence_score: Optional[float],
    camera_id: str,
    event_type: str,
    snapshot_path: str,
    timestamp: datetime
):
    """Insert attendance log into database"""
    try:
        # Handle timestamp conversion if it's a string
        if isinstance(timestamp, str):
            try:
                if '+' in timestamp:
                    timestamp = datetime.fromisoformat(timestamp)
                else:
                    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    timestamp = pytz.timezone('Asia/Kolkata').localize(timestamp)
            except ValueError:
                timestamp = datetime.now(pytz.timezone('Asia/Kolkata'))
        
        log = models.AttendanceLog(
            person_name=person_name,
            tracking_id=tracking_id,
            confidence_score=confidence_score,
            camera_id=camera_id,
            event_type=event_type,
            snapshot_path=snapshot_path,
            timestamp=timestamp
        )
        
        db.add(log)
        update_daily_summary(db, person_name, event_type, timestamp)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to insert log: {e}")
        return False

def update_daily_summary(db: Session, person_name: str, camera_id: str, event_type: str, timestamp: datetime):
    """Update daily summary statistics with camera_id"""
    date = timestamp.date()
    
    summary = db.query(models.DailySummary).filter(
        models.DailySummary.person_name == person_name,
        models.DailySummary.date == date,
        models.DailySummary.camera_id == camera_id  # New filter
    ).first()
    
    if summary:
        if event_type == 'login':
            if not summary.first_login or timestamp < summary.first_login:
                summary.first_login = timestamp
            summary.total_logins += 1
        elif event_type == 'logout':
            if not summary.last_logout or timestamp > summary.last_logout:
                summary.last_logout = timestamp
            summary.total_logouts += 1
        
        if summary.first_login and summary.last_logout:
            summary.working_hours = summary.last_logout - summary.first_login
    else:
        summary = models.DailySummary(
            person_name=person_name,
            date=date,
            camera_id=camera_id,  # New field
            first_login=timestamp if event_type == 'login' else None,
            last_logout=timestamp if event_type == 'logout' else None,
            total_logins=1 if event_type == 'login' else 0,
            total_logouts=1 if event_type == 'logout' else 0
        )
        db.add(summary)
    
    db.commit()

def export_to_csv(db: Session, output_dir="exports"):
    """Export attendance logs to CSV file"""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"attendance_export_{timestamp}.csv"
    filepath = Path(output_dir) / filename
    
    logs = db.query(models.AttendanceLog).order_by(models.AttendanceLog.timestamp.desc()).all()
    
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ['id', 'person_name', 'camera_id', 'confidence', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for log in logs:
            writer.writerow({
                'id': log.id,
                'person_name': log.person_name,
                'camera_id': log.camera_id,
                'confidence': log.confidence_score,
                'timestamp': log.timestamp.isoformat()
            })
    
    return str(filepath.absolute())