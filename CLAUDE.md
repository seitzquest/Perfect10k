# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Perfect10k has evolved from a Jupyter prototype to a full-stack intelligent route planner. It generates optimal walking routes for achieving 10,000 daily steps using AI-powered semantic place matching and advanced optimization algorithms.

## Architecture

### Backend (Python FastAPI)
- **API Layer**: REST endpoints in `backend/api/endpoints/`
- **Core Algorithms**: Advanced route optimization in `backend/core/`
- **Semantic Engine**: Place embedding and matching in `backend/services/`
- **Database**: PostgreSQL with SQLAlchemy models in `backend/models/`

### Frontend (Vanilla JavaScript)
- **Clean HTML/CSS**: No build tools, runs directly in browser
- **Interactive Maps**: Leaflet integration with custom markers
- **Real-time Planning**: AJAX calls to backend API
- **Responsive Design**: Works on mobile and desktop

## Key Features

### Advanced Route Planning
- **Multiple Optimization Methods**: Genetic algorithm, simulated annealing, value-guided search
- **Resolution-Based Complexity Reduction**: Handles dense road networks by adaptively reducing edges
- **Value Function Framework**: Spatial optimization incorporating user preferences, elevation, accessibility

### Semantic Intelligence
- **Natural Language Preferences**: Users describe preferences like "I like quiet parks near water"
- **Embedding-Based Matching**: Uses sentence transformers to find semantically similar places
- **Dynamic Value Function**: Applies user preferences to spatial route optimization

## Development Commands

### Backend Setup
```bash
cd backend
# Quick setup
./scripts/setup.sh

# Or manual setup
uv pip install -e ".[dev]"
python scripts/setup_db.py  # First time only
uv run uvicorn main:app --reload

# Or use dev helper
./scripts/dev.sh dev
```

### With Docker
```bash
cd backend
docker-compose up -d
```

### Frontend
```bash
cd frontend
# Open index.html in browser or serve with:
python -m http.server 3000
```

### Database Migrations
```bash
cd backend
./scripts/dev.sh db-migrate
# or manually:
uv run alembic revision --autogenerate -m "description"
uv run alembic upgrade head
```

### Testing
```bash
cd backend
./scripts/dev.sh test
# or:
uv run pytest tests/
```

### Development Tools
```bash
cd backend
./scripts/dev.sh format    # Format code
./scripts/dev.sh lint      # Run linting
./scripts/dev.sh check     # Run all checks
./scripts/dev.sh clean     # Clean cache files
```

## Core Components

### Route Planning Pipeline
1. **Load OSM Data**: `core/route_algorithms.py` - MapLoader class
2. **Complexity Reduction**: `core/resolution_strategies.py` - Multiple reduction strategies
3. **Value Function**: `core/value_function.py` - Spatial preference optimization
4. **Advanced Optimization**: `core/advanced_optimizer.py` - Genetic algorithms, simulated annealing
5. **Semantic Matching**: `services/embedding_service.py` - Place preference matching

### API Endpoints
- **Auth**: `/api/auth/` - User registration, login, JWT tokens
- **Routes**: `/api/routes/` - Route planning, history, GPX export
- **Places**: `/api/places/` - Semantic place search, nearby places
- **Preferences**: `/api/preferences/` - User preference management

## Dependencies

### Backend
- `fastapi`: Web framework
- `osmnx`: OpenStreetMap data processing
- `networkx`: Graph algorithms
- `sentence-transformers`: Semantic embeddings
- `sqlalchemy`: Database ORM
- `rasterio`: Elevation data (optional)

### Frontend
- `leaflet`: Interactive maps
- Vanilla JavaScript (no frameworks)
- CSS custom properties for theming

## Configuration

Key settings in `backend/.env`:
- `DATABASE_URL`: PostgreSQL connection
- `SECRET_KEY`: JWT signing key
- `EMBEDDING_MODEL`: Sentence transformer model
- `SRTM_DATA_PATH`: Elevation data (optional)

## Debugging

### Common Issues
1. **Route planning slow**: Check `MAX_ROUTE_COMPLEXITY` setting
2. **Semantic search not working**: Verify embedding model download
3. **Database errors**: Run `python scripts/setup_db.py`
4. **Frontend API calls failing**: Check CORS settings in backend

### Logs
- Backend: Uvicorn logs show API requests
- Frontend: Browser dev tools for client issues
- Database: PostgreSQL logs for query issues

## Performance Notes

- Route planning: 2-5 seconds for complex routes
- Semantic search: ~100-500ms with caching
- Memory usage: ~200-500MB depending on area
- First run downloads ~90MB embedding model