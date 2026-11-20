import os
import sqlite3
from app.config import settings

# Ensure database directory exists
db_file_path = settings.DATABASE_URL.replace("sqlite:///", "")
if db_file_path.startswith("./"):
    db_file_path = db_file_path[2:]
db_dir = os.path.dirname(db_file_path)
if db_dir:
    os.makedirs(db_dir, exist_ok=True)

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import declarative_base, sessionmaker

    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
            
except ImportError:
    # Lightweight standard library SQLite fallback
    engine = None
    SessionLocal = None
    Base = None

    def get_db():
        conn = sqlite3.connect(db_file_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
