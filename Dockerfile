# Use Python 3.11 with optimized libraries
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_CACHE_DIR=/tmp/uv-cache
ENV UV_HTTP_TIMEOUT=120

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libspatialindex-dev \
    libgeos-dev \
    libproj-dev \
    gdal-bin \
    libgdal-dev \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies with uv (includes optimized numpy/scipy)
RUN uv sync --frozen

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY run.py ./

# Create necessary directories
RUN mkdir -p logs cache/graphs cache/overlays

# Set Python path to include backend directory
ENV PYTHONPATH=/app:/app/backend

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set working directory to backend for module imports
WORKDIR /app/backend

# Run the application with optimized async settings
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--http", "httptools"]