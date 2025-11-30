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


# Backend/database.py
import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSONB
import redis

# --- CONFIGURATION ---
# Replace 'password' with your actual PostgreSQL password
DATABASE_URL = "postgresql://postgres:password@localhost:5432/chimera_db"

# Redis Configuration (Default settings)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

# --- DATABASE SETUP ---
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- REDIS CLIENT ---
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    redis_client.ping() # Test connection
except redis.ConnectionError:
    print("⚠️  Redis not available. Caching disabled.")
    redis_client = None

# --- MODELS (Python versions of your SQL tables) ---

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role = Column(String, default="user")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class ScanResult(Base):
    __tablename__ = "scan_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    target = Column(String, index=True) # The URL or Email
    scan_type = Column(String) # "URL" or "EMAIL"
    
    verdict = Column(String)
    confidence_score = Column(Float)
    risk_level = Column(String)
    
    # Stores V1, V2, V3, V4 feature data dynamically
    ai_details = Column(JSONB)
    
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class CommunityReport(Base):
    __tablename__ = "community_reports"
    id = Column(Integer, primary_key=True, index=True)
    reporter_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    suspicious_url = Column(String)
    report_type = Column(String)
    description = Column(Text)
    evidence = Column(Text)
    status = Column(String, default="Pending")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Create tables automatically if they don't exist
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
