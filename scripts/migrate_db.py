# scripts/migrate_db.py
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    # Add foreign key to attendance_logs
    op.create_foreign_key(
        'fk_attendance_logs_person_id',
        'attendance_logs', 'persons',
        ['person_id'], ['id']
    )
    
    op.create_foreign_key(
        'fk_attendance_logs_camera_id',
        'attendance_logs', 'cameras',
        ['camera_id'], ['camera_id']
    )
    
    # Add foreign key to daily_summary
    op.create_foreign_key(
        'fk_daily_summary_person_id',
        'daily_summary', 'persons',
        ['person_id'], ['id']
    )
    
    op.create_foreign_key(
        'fk_daily_summary_camera_id',
        'daily_summary', 'cameras',
        ['camera_id'], ['camera_id']
    )

def downgrade():
    op.drop_constraint('fk_attendance_logs_person_id', 'attendance_logs', type_='foreignkey')
    op.drop_constraint('fk_attendance_logs_camera_id', 'attendance_logs', type_='foreignkey')
    op.drop_constraint('fk_daily_summary_person_id', 'daily_summary', type_='foreignkey')
    op.drop_constraint('fk_daily_summary_camera_id', 'daily_summary', type_='foreignkey')