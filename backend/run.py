#!/usr/bin/env python3
"""
Perfect10k Backend Runner

This script provides a simple way to run the Perfect10k backend server
with proper error handling and logging.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from main import app
from core.config import settings

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def check_environment():
    """Check if the environment is properly configured."""
    required_vars = []
    
    # Check for critical environment variables
    if not os.getenv('SECRET_KEY') or os.getenv('SECRET_KEY') == 'your-secret-key-change-in-production':
        logger.warning("SECRET_KEY is not set or using default value. Please set a secure secret key.")
    
    if not os.getenv('DATABASE_URL'):
        logger.warning("DATABASE_URL is not set. Using default PostgreSQL connection.")
    
    return True


def main():
    """Main entry point for the application."""
    logger.info("Starting Perfect10k Route Planner API")
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed")
        sys.exit(1)
    
    # Display startup information
    logger.info(f"Server will start on {settings.HOST}:{settings.PORT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Allowed origins: {settings.ALLOWED_ORIGINS}")
    
    try:
        # Run the server
        uvicorn.run(
            "main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.DEBUG,
            log_level=settings.LOG_LEVEL.lower(),
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()