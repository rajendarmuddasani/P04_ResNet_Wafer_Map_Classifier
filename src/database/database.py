"""
Database Connection and Session Management

Handles PostgreSQL connection pooling and session lifecycle.
Uses SQLAlchemy for ORM and Alembic for migrations.

Environment Variables Required:
    POSTGRES_USER: Database user (default: postgres)
    POSTGRES_PASSWORD: Database password
    POSTGRES_HOST: Database host (default: localhost)
    POSTGRES_PORT: Database port (default: 5432)
    POSTGRES_DB: Database name (default: wafer_classifier)
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator
import os
from pathlib import Path
import logging

# Import models to register with Base
from src.database.models import Base

logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """
    Build PostgreSQL database URL from environment variables.
    
    Returns:
        Database URL string for SQLAlchemy
    
    Example:
        postgresql://user:password@localhost:5432/wafer_classifier
    """
    from config import settings
    return settings.database_url


def create_database_engine(
    database_url: str = None,
    pool_size: int = 10,
    max_overflow: int = 20,
    echo: bool = False,
):
    """
    Create SQLAlchemy engine with connection pooling.
    
    Args:
        database_url: PostgreSQL connection string (default: from environment)
        pool_size: Number of persistent connections (default: 10)
        max_overflow: Maximum overflow connections (default: 20)
        echo: Log all SQL statements (default: False, set True for debugging)
    
    Returns:
        SQLAlchemy engine instance
    """
    if database_url is None:
        database_url = get_database_url()
    
    engine = create_engine(
        database_url,
        poolclass=QueuePool,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,  # Verify connections before using
        echo=echo,
        future=True,  # SQLAlchemy 2.0 style
    )
    
    # Log connection pool events (optional)
    @event.listens_for(engine, "connect")
    def receive_connect(dbapi_conn, connection_record):
        logger.debug("Database connection established")
    
    @event.listens_for(engine, "checkout")
    def receive_checkout(dbapi_conn, connection_record, connection_proxy):
        logger.debug("Connection checked out from pool")
    
    return engine


# Global engine and session factory
DATABASE_URL = get_database_url()
engine = create_database_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_database():
    """
    Initialize database by creating all tables.
    
    Note: In production, use Alembic migrations instead.
    This is useful for development and testing.
    """
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables created successfully")


def drop_all_tables():
    """
    Drop all database tables.
    
    WARNING: This will delete all data. Use only in development/testing.
    """
    logger.warning("Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.info("✅ All tables dropped")


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Automatically handles commit/rollback and session cleanup.
    
    Usage:
        with get_db_session() as db:
            wafer = db.query(WaferMap).filter_by(wafer_id="W12345").first()
            # Session automatically closed at end of with block
    
    Yields:
        SQLAlchemy Session
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to inject database sessions.
    
    Usage in FastAPI:
        from fastapi import Depends
        from src.database.database import get_db
        
        @app.get("/wafers/{wafer_id}")
        def get_wafer(wafer_id: str, db: Session = Depends(get_db)):
            return db.query(WaferMap).filter_by(wafer_id=wafer_id).first()
    
    Yields:
        SQLAlchemy Session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_connection() -> bool:
    """
    Verify database connection is working.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


def get_database_stats() -> dict:
    """
    Get database connection pool statistics.
    
    Returns:
        Dictionary with pool statistics
    """
    pool = engine.pool
    return {
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "total_connections": pool.size() + pool.overflow(),
    }


class DatabaseManager:
    """
    High-level database management class.
    
    Provides convenience methods for common operations.
    """
    
    def __init__(self, engine=engine):
        self.engine = engine
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    def create_all_tables(self):
        """Create all tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("✅ All tables created")
    
    def drop_all_tables(self):
        """Drop all tables (WARNING: deletes all data)."""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("⚠️  All tables dropped")
    
    def reset_database(self):
        """Drop and recreate all tables (WARNING: deletes all data)."""
        self.drop_all_tables()
        self.create_all_tables()
        logger.info("✅ Database reset complete")
    
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Get database session as context manager."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def check_connection(self) -> bool:
        """Check if database connection is working."""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False


# Create global database manager instance
db_manager = DatabaseManager()


if __name__ == "__main__":
    # Test database connection and setup
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("=" * 60)
    print("Database Connection Test")
    print("=" * 60)
    print(f"\nDatabase URL: {DATABASE_URL}")
    
    # Check connection
    print("\n1. Checking database connection...")
    if check_database_connection():
        print("   ✅ Connection successful")
    else:
        print("   ❌ Connection failed")
        sys.exit(1)
    
    # Get pool stats
    print("\n2. Connection pool statistics:")
    stats = get_database_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test session creation
    print("\n3. Testing session creation...")
    try:
        with get_db_session() as db:
            result = db.execute("SELECT current_database(), current_user, version()")
            row = result.fetchone()
            print(f"   Database: {row[0]}")
            print(f"   User: {row[1]}")
            print(f"   Version: {row[2][:50]}...")
        print("   ✅ Session test successful")
    except Exception as e:
        print(f"   ❌ Session test failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ All database tests passed!")
    print("=" * 60)
