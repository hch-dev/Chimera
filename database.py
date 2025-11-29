# Backend/database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import JSONB
import redis
import datetime

# --- CONFIGURATION ---
# Change 'password' to your actual Postgres password
DATABASE_URL = "postgresql://postgres:password@localhost:5432/chimera_db"
REDIS_URL = "redis://localhost:6379/0"

# --- POSTGRES SETUP ---
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- REDIS SETUP ---
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# --- MODELS (THE TABLES) ---
class ScanResult(Base):
    __tablename__ = "scan_history"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, index=True)
    verdict = Column(String)
    score = Column(Float)
    # This stores the detailed feature list from Version 1
    details = Column(JSONB) 
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

# Dependency for API routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
