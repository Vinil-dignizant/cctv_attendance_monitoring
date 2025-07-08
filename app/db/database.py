# app/db/database.py
# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# from dotenv import load_dotenv
# import os

# # Load environment variables from .env file
# load_dotenv()

# # Get database configuration from environment variables
# DB_HOST = os.getenv('DB_HOST', 'localhost')
# DB_PORT = os.getenv('DB_PORT', '5432')
# DB_NAME = os.getenv('DB_NAME', 'face_recognition_db')
# DB_USER = os.getenv('DB_USER', 'face_recognition_user')
# DB_PASSWORD = os.getenv('DB_PASSWORD', 'vinil123')

# SQLALCHEMY_DATABASE_URL = (
#     f"postgresql://{DB_USER}:{DB_PASSWORD}@"
#     f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
# )

# engine = create_engine(SQLALCHEMY_DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base = declarative_base()

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config.config_manager import ConfigManager

# Initialize config manager
config = ConfigManager()

SQLALCHEMY_DATABASE_URL = config.get_db_url()

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"options": f"-c timezone={config.get_config()['PGTZ']}"}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()