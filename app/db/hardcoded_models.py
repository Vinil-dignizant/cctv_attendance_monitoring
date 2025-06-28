import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'face_recognition_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'your_password_here'),
    'options': '-c timezone=Asia/Kolkata'
}

# Table creation SQL
TABLE_SCHEMAS = {
    'attendance_logs': """
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id SERIAL PRIMARY KEY,
            person_name VARCHAR(100) NOT NULL,
            tracking_id INTEGER,
            confidence_score DECIMAL(5,4),
            camera_id VARCHAR(50) NOT NULL,
            event_type VARCHAR(20) NOT NULL DEFAULT 'login',
            snapshot_path VARCHAR(255),
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
    """,
    'persons': """
        CREATE TABLE IF NOT EXISTS persons (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) UNIQUE NOT NULL,
            email VARCHAR(150),
            department VARCHAR(100),
            employee_id VARCHAR(50),
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
    """,
    'cameras': """
        CREATE TABLE IF NOT EXISTS cameras (
            id SERIAL PRIMARY KEY,
            camera_id VARCHAR(50) UNIQUE NOT NULL,
            camera_name VARCHAR(100),
            location VARCHAR(100),
            url TEXT,
            event_type VARCHAR(20) DEFAULT 'login',
            is_enabled BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
    """,
    'daily_summary': """
        CREATE TABLE IF NOT EXISTS daily_summary (
            id SERIAL PRIMARY KEY,
            person_name VARCHAR(100) NOT NULL,
            date DATE NOT NULL,
            first_login TIMESTAMP WITH TIME ZONE,
            last_logout TIMESTAMP WITH TIME ZONE,
            total_logins INTEGER DEFAULT 0,
            total_logouts INTEGER DEFAULT 0,
            working_hours INTERVAL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
    """
}

# Index creation SQL
INDEXES = {
    'attendance_logs': [
        "CREATE INDEX IF NOT EXISTS idx_person_name ON attendance_logs (person_name)",
        "CREATE INDEX IF NOT EXISTS idx_camera_id ON attendance_logs (camera_id)",
        "CREATE INDEX IF NOT EXISTS idx_timestamp ON attendance_logs (timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_event_type ON attendance_logs (event_type)",
        "CREATE INDEX IF NOT EXISTS idx_created_at ON attendance_logs (created_at)"
    ],
    'persons': [
        "CREATE INDEX IF NOT EXISTS idx_persons_name ON persons (name)",
        "CREATE INDEX IF NOT EXISTS idx_persons_employee_id ON persons (employee_id)",
        "CREATE INDEX IF NOT EXISTS idx_persons_department ON persons (department)"
    ],
    'cameras': [
        "CREATE INDEX IF NOT EXISTS idx_cameras_camera_id ON cameras (camera_id)",
        "CREATE INDEX IF NOT EXISTS idx_cameras_enabled ON cameras (is_enabled)"
    ],
    'daily_summary': [
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_daily_summary ON daily_summary (person_name, date)",
        "CREATE INDEX IF NOT EXISTS idx_daily_person_date ON daily_summary (person_name, date)",
        "CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_summary (date)"
    ]
}

# Database functions
UPDATE_TIMESTAMP_FUNCTION = """
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = CURRENT_TIMESTAMP;
        RETURN NEW;
    END;
    $$ language 'plpgsql';
"""

TRIGGERS = {
    'persons': """
        CREATE TRIGGER IF NOT EXISTS update_persons_updated_at 
        BEFORE UPDATE ON persons 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """,
    'cameras': """
        CREATE TRIGGER IF NOT EXISTS update_cameras_updated_at 
        BEFORE UPDATE ON cameras 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """,
    'daily_summary': """
        CREATE TRIGGER IF NOT EXISTS update_daily_summary_updated_at 
        BEFORE UPDATE ON daily_summary 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """
}