import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import numpy as np
from datetime import datetime
import pytz
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

from .models import DB_CONFIG, TABLE_SCHEMAS, INDEXES, UPDATE_TIMESTAMP_FUNCTION, TRIGGERS

# Setup logging
logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def convert_numpy_types(value):
    """Convert numpy types to Python native types"""
    if isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    return value

def init_db():
    """Initialize database tables and indexes"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Create tables
                for table, schema in TABLE_SCHEMAS.items():
                    cursor.execute(schema)
                
                # Create indexes
                for table, index_queries in INDEXES.items():
                    for query in index_queries:
                        cursor.execute(query)
                
                # Create timestamp update function
                cursor.execute(UPDATE_TIMESTAMP_FUNCTION)
                
                # Create triggers
                for trigger in TRIGGERS.values():
                    cursor.execute(trigger)
                
                conn.commit()
                logger.info("Database initialized successfully")
                print("[INFO] Database initialized successfully")
                
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def insert_log(
    person_name: str,
    tracking_id: int,
    confidence_score: Optional[float],
    camera_id: str,
    event_type: str,
    snapshot_path: str,
    timestamp: datetime
) -> bool:
    """Insert attendance log into database"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Convert numpy types if needed
                person_name = convert_numpy_types(person_name)
                tracking_id = convert_numpy_types(tracking_id)
                confidence_score = convert_numpy_types(confidence_score) if confidence_score else None
                
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
                
                cursor.execute("""
                    INSERT INTO attendance_logs 
                    (person_name, tracking_id, confidence_score, camera_id, 
                     event_type, snapshot_path, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (person_name, tracking_id, confidence_score, camera_id, 
                      event_type, snapshot_path, timestamp))
                
                update_daily_summary(cursor, person_name, event_type, timestamp)
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Failed to insert log: {e}")
        return False

def update_daily_summary(cursor, person_name: str, event_type: str, timestamp: datetime):
    """Update daily summary statistics"""
    date = timestamp.date()
    
    cursor.execute("""
        SELECT id, first_login, last_logout, total_logins, total_logouts 
        FROM daily_summary 
        WHERE person_name = %s AND date = %s
    """, (person_name, date))
    
    result = cursor.fetchone()
    
    if result:
        # Update existing record
        summary_id, first_login, last_logout, total_logins, total_logouts = result
        
        if event_type == 'login':
            first_login = min(first_login or timestamp, timestamp)
            total_logins += 1
        elif event_type == 'logout':
            last_logout = max(last_logout or timestamp, timestamp)
            total_logouts += 1
        
        working_hours = (last_logout - first_login) if first_login and last_logout else None
        
        cursor.execute("""
            UPDATE daily_summary 
            SET first_login = %s, last_logout = %s, 
                total_logins = %s, total_logouts = %s, 
                working_hours = %s
            WHERE id = %s
        """, (first_login, last_logout, total_logins, total_logouts, working_hours, summary_id))
    else:
        # Create new record
        first_login = timestamp if event_type == 'login' else None
        last_logout = timestamp if event_type == 'logout' else None
        total_logins = 1 if event_type == 'login' else 0
        total_logouts = 1 if event_type == 'logout' else 0
        
        cursor.execute("""
            INSERT INTO daily_summary 
            (person_name, date, first_login, last_logout, total_logins, total_logouts)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (person_name, date, first_login, last_logout, total_logins, total_logouts))

# [Keep all other functions from original postgresql_db_utils1.py]
# get_attendance_logs(), get_daily_summary(), add_person(), 
# add_camera(), get_statistics(), test_connection() etc.
# Just update their imports to use the new DB_CONFIG from models.py