# Perfect10k

Interactive route planner with semantic preferences for daily walks.

## Setup

### Local

```bash
uv sync
python run.py
```

### Docker

```bash
docker build -t perfect10k .
docker run -p 8000:8000 -v $(pwd)/backend/cache:/app/cache perfect10k
```

Access at <http://localhost:8000>

## Usage

1. Set starting point on map
2. Add preferences (e.g., "scenic parks", "quiet streets")
3. Build route by selecting from waypoint suggestions
4. Complete loop and export GPX

## Tech Stack

- **Backend**: Python 3.10+ with FastAPI
- **Frontend**: JavaScript with Leaflet maps
- **Geospatial**: OSMNx + NetworkX
- **Semantic**: Scikit-learn

## Requirements

- Python 3.10+
- ~2GB RAM
- Internet connection

## Core API

- `POST /api/start-session` - Initialize route planning
- `POST /api/add-waypoint` - Add waypoint to route
- `POST /api/finalize-route` - Complete circular route
- `GET /health` - Health check
- `GET /docs` - API documentation

## Development

```bash
# Linting
ruff check

# Formatting
black .
```

Dependencies managed via `uv` in `pyproject.toml`.

## Architecture

```text
backend/
├── main.py                 # FastAPI app
├── clean_router.py         # Routing logic
├── semantic_overlays.py    # Feature management
└── core/                   # Utilities

frontend/
├── index.html
├── js/
└── css/
```

Features semantic preference matching, multi-layer caching, and real-time route construction.
