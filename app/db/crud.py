# crud.py
from sqlalchemy import inspect
from sqlalchemy.orm import Session
import csv
from datetime import datetime
from pathlib import Path
import pytz
from . import models
from .database import SessionLocal, engine
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from sqlalchemy.orm import joinedload
import numpy as np
import io
from .models import AttendanceLog, DailySummary, Person, FaceFeature, FaceImage, Camera
from sqlalchemy import text

@contextmanager
def get_db():
    """Provide a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def init_db():
    """Initialize database tables and handle migrations"""
    try:
        # Check if tables exist
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        # Create all tables if none exist (new installation)
        if not existing_tables:
            models.Base.metadata.create_all(bind=engine)
            print("[INFO] Database tables created successfully")
        else:
            # For existing installations, Alembic will handle migrations
            print("[INFO] Database already exists, using Alembic for migrations")
            
    except Exception as e:
        print(f"[ERROR] Failed to initialize database: {e}")
        raise

def insert_log(
    person_id: int,
    camera_id: str,
    tracking_id: Optional[int] = None,
    confidence_score: Optional[float] = None,
    event_type: str = 'login',
    snapshot_path: Optional[str] = None,
    timestamp: Optional[datetime] = None
) -> bool:
    """Insert attendance log into database"""
    with get_db() as db:
        try:
            # Handle timestamp conversion
            if timestamp is None:
                timestamp = datetime.now(pytz.timezone('Asia/Kolkata'))
            elif isinstance(timestamp, str):
                try:
                    if '+' in timestamp:
                        timestamp = datetime.fromisoformat(timestamp)
                    else:
                        timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                        timestamp = pytz.timezone('Asia/Kolkata').localize(timestamp)
                except ValueError:
                    timestamp = datetime.now(pytz.timezone('Asia/Kolkata'))
            
            # Get person name
            person = db.query(Person).filter(Person.id == person_id).first()
            if not person:
                raise ValueError(f"Person with ID {person_id} not found")
            
            log = AttendanceLog(
                person_id=person_id,
                camera_id=camera_id,
                tracking_id=tracking_id,
                confidence_score=confidence_score,
                event_type=event_type,
                snapshot_path=snapshot_path,
                timestamp=timestamp
            )
            
            db.add(log)
            _update_daily_summary(db, person_id, camera_id, event_type, timestamp)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            print(f"[ERROR] Failed to insert log: {e}")
            return False

def _update_daily_summary(db: Session, person_id: int, camera_id: str, event_type: str, timestamp: datetime):
    """Internal function to update daily summary statistics"""
    date = timestamp.date()
    
    summary = db.query(models.DailySummary).filter(
        models.DailySummary.person_id == person_id,
        models.DailySummary.date == date,
        models.DailySummary.camera_id == camera_id
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
            person_id=person_id,
            camera_id=camera_id,
            date=date,
            first_login=timestamp if event_type == 'login' else None,
            last_logout=timestamp if event_type == 'logout' else None,
            total_logins=1 if event_type == 'login' else 0,
            total_logouts=1 if event_type == 'logout' else 0
        )
        db.add(summary)

# Rest of the functions (export_to_csv, get_attendance_logs) can remain the same

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


def get_attendance_logs(limit: int = None) -> List[models.AttendanceLog]:
    """Retrieve attendance logs from database"""
    db = SessionLocal()
    try:
        query = db.query(models.AttendanceLog).order_by(models.AttendanceLog.timestamp.desc())
        if limit:
            return query.limit(limit).all()
        return query.all()
    except Exception as e:
        print(f"[ERROR] Failed to fetch attendance logs: {e}")
        return []
    finally:
        db.close()

def create_person(
    db: Session,
    name: str,
    email: str = None,
    department: str = None,
    employee_id: str = None
) -> Person:
    """Create a new person record"""
    person = Person(
        name=name,
        email=email,
        department=department,
        employee_id=employee_id
    )
    db.add(person)
    db.commit()
    db.refresh(person)
    return person

def add_face_feature(
    db: Session,
    person_id: int,
    embedding: np.ndarray
) -> FaceFeature:
    """Add face embedding for a person"""
    # Convert numpy array to bytes
    buf = io.BytesIO()
    np.save(buf, embedding)
    buf.seek(0)
    
    feature = FaceFeature(
        person_id=person_id,
        embedding=buf.read()
    )
    db.add(feature)
    db.commit()
    db.refresh(feature)
    return feature

def add_face_image(
    db: Session,
    person_id: int,
    image_path: str,
    thumbnail: np.ndarray = None
) -> FaceImage:
    """Add face image for a person"""
    img = FaceImage(
        person_id=person_id,
        image_path=image_path,
        thumbnail=thumbnail.tobytes() if thumbnail is not None else None
    )
    db.add(img)
    db.commit()
    db.refresh(img)
    return img

def get_person_by_id(db: Session, person_id: int) -> Optional[Person]:
    """Get person by ID with face features"""
    return db.query(Person).options(
        joinedload(Person.face_features),
        joinedload(Person.face_images)
    ).filter(Person.id == person_id).first()

def get_all_persons(db: Session) -> List[Person]:
    """Get all persons with their face data"""
    return db.query(Person).options(
        joinedload(Person.face_features),
        joinedload(Person.face_images)
    ).order_by(Person.name).all()

def delete_person(db: Session, person_id: int) -> bool:
    """Delete a person and all associated face data"""
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        return False
    
    # Delete associated face features and images
    db.query(FaceFeature).filter(FaceFeature.person_id == person_id).delete()
    db.query(FaceImage).filter(FaceImage.person_id == person_id).delete()
    
    # Delete person
    db.delete(person)
    db.commit()
    return True


#crud for Camera Management

# crud.py additions
def create_camera(
    db: Session,
    camera_id: str,
    camera_name: str = None,
    location: str = None,
    url: str = None,
    event_type: str = 'login',
    is_enabled: bool = True
) -> Camera:
    """Create a new camera record"""
    camera = Camera(
        camera_id=camera_id,
        camera_name=camera_name,
        location=location,
        url=url,
        event_type=event_type,
        is_enabled=is_enabled
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera

def get_camera(db: Session, camera_id: str) -> Optional[Camera]:
    """Get camera by ID"""
    return db.query(Camera).filter(Camera.camera_id == camera_id).first()

def get_all_cameras(db: Session) -> List[Camera]:
    """Get all cameras"""
    return db.query(Camera).order_by(Camera.camera_id).all()

def update_camera(
    db: Session,
    camera_id: str,
    **kwargs
) -> Optional[Camera]:
    """Update camera properties"""
    camera = db.query(Camera).filter(Camera.camera_id == camera_id).first()
    if not camera:
        return None
    
    for key, value in kwargs.items():
        if hasattr(camera, key):
            setattr(camera, key, value)
    
    db.commit()
    db.refresh(camera)
    return camera

def delete_camera(db: Session, camera_id: str) -> bool:
    """Delete a camera"""
    camera = db.query(Camera).filter(Camera.camera_id == camera_id).first()
    if not camera:
        return False
    
    db.delete(camera)
    db.commit()
    return True


# camera db usage functions


def get_system_config(db: Session) -> Dict:
    """Get all system configuration"""
    try:
        config_entries = db.execute(
            text("SELECT config_key, config_value FROM system_config")
        ).fetchall()
        return dict(config_entries)
    except Exception as e:
        print(f"[ERROR] Failed to load system config: {e}")
        return {}

def update_system_config(db: Session, key: str, value: str) -> bool:
    """Update system configuration"""
    try:
        db.execute(
            "INSERT INTO system_config (config_key, config_value) "
            "VALUES (:key, :value) "
            "ON CONFLICT (config_key) DO UPDATE SET config_value = EXCLUDED.config_value",
            {"key": key, "value": str(value)}
        )
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to update system config: {e}")
        return False