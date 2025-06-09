FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UV_CACHE_DIR=/tmp/uv-cache

# Set work directory
WORKDIR /app

# Install system dependencies including geospatial libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libpq-dev \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install uv
RUN pip install --no-cache-dir uv

# Copy only dependency files first for better caching
COPY pyproject.toml .
COPY .python-version .

# Install Python dependencies with uv (this layer will be cached)
RUN uv pip install --system -e .

# Copy the rest of the project
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash perfect10k && \
    chown -R perfect10k:perfect10k /app
USER perfect10k

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "run.py"]