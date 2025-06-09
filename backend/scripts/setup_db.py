#!/usr/bin/env python3
"""
Database setup script for Perfect10k

This script creates the database tables and performs initial setup.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from core.config import settings
from core.database import Base
from models.models import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_database():
    """Create the database if it doesn't exist."""
    try:
        # Extract database name from URL
        db_url = settings.DATABASE_URL
        db_name = db_url.split('/')[-1]
        
        # Connect to postgres database to create our database
        base_url = '/'.join(db_url.split('/')[:-1]) + '/postgres'
        engine = create_engine(base_url)
        
        with engine.connect() as conn:
            # Use autocommit mode to create database
            conn = conn.execution_options(autocommit=True)
            
            # Check if database exists
            result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'"))
            if not result.fetchone():
                conn.execute(text(f"CREATE DATABASE {db_name}"))
                logger.info(f"Created database: {db_name}")
            else:
                logger.info(f"Database {db_name} already exists")
                
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        return False
    
    return True


def create_tables():
    """Create all tables."""
    try:
        engine = create_engine(settings.DATABASE_URL)
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Created all database tables")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        return False


def setup_extensions():
    """Setup PostgreSQL extensions if needed."""
    try:
        engine = create_engine(settings.DATABASE_URL)
        
        with engine.connect() as conn:
            # Create extensions for better performance (if they don't exist)
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
                logger.info("Created pg_trgm extension")
            except Exception as e:
                logger.warning(f"Could not create pg_trgm extension: {e}")
            
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS btree_gin;"))
                logger.info("Created btree_gin extension")
            except Exception as e:
                logger.warning(f"Could not create btree_gin extension: {e}")
            
            conn.commit()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup extensions: {e}")
        return False


def create_sample_data():
    """Create some sample data for testing."""
    try:
        from sqlalchemy.orm import sessionmaker
        from services.auth import AuthService
        
        engine = create_engine(settings.DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        with SessionLocal() as db:
            auth_service = AuthService(db)
            
            # Create a test user if it doesn't exist
            existing_user = db.query(User).filter(User.email == "test@perfect10k.com").first()
            if not existing_user:
                test_user = auth_service.create_user(
                    email="test@perfect10k.com",
                    password="testpassword123"
                )
                logger.info("Created test user: test@perfect10k.com")
            else:
                logger.info("Test user already exists")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create sample data: {e}")
        return False


def main():
    """Main setup function."""
    logger.info("Starting Perfect10k database setup...")
    
    # Step 1: Create database
    if not create_database():
        logger.error("Database creation failed")
        sys.exit(1)
    
    # Step 2: Setup extensions
    if not setup_extensions():
        logger.warning("Extension setup failed, but continuing...")
    
    # Step 3: Create tables
    if not create_tables():
        logger.error("Table creation failed")
        sys.exit(1)
    
    # Step 4: Create sample data
    if not create_sample_data():
        logger.warning("Sample data creation failed, but continuing...")
    
    logger.info("Database setup completed successfully!")
    logger.info("You can now start the server with: python run.py")


if __name__ == "__main__":
    main()