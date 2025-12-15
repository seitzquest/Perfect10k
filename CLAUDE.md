# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Perfect10k is an interactive route planner for daily walks with semantic preferences. Users set a starting point, add preferences like "scenic parks" or "quiet streets", and iteratively build a circular route by selecting from waypoint suggestions.

## Development Commands

```bash
# Run the application
uv sync
uv run run.py
# Access at http://localhost:8000

# Linting and formatting
ruff check                    # Lint
ruff check --fix             # Lint with auto-fix
ruff format .                # Format code

# Run pre-commit hooks manually
pre-commit run --all-files

# Type checking
pyright
```

## Architecture

### Backend (`backend/`)

The backend is a FastAPI application with lazy initialization for fast startup.

**Core Request Flow:**
1. `main.py` - FastAPI app with API endpoints, lazy-loads CleanRouter and SemanticOverlayManager
2. `clean_router.py` - CleanRouter manages client sessions and orchestrates route building
3. `clean_candidate_generator.py` - Generates scored waypoint candidates using spatial indexing
4. `interpretable_scorer.py` - Translates natural language preferences into weighted feature scores
5. `feature_database.py` - Pre-computes discrete features (forest proximity, water, parks, path quality) per grid cell

**Spatial Data Pipeline:**
- `core/spatial_tile_storage.py` - Geohash-based tile storage with SQLite index for permanent graph data
- `spatial_grid.py` - Spatial indexing for efficient node lookups
- `smart_cache_manager.py` - Multi-layer caching (LRU memory + persistent) for graphs and features
- `semantic_overlays.py` - Fetches OSM features (forests, rivers, lakes, parks) via Overpass API

**Async Processing:**
- `async_job_manager.py` - Background job system for cache warming and route generation

### Frontend (`frontend/`)

Vanilla JavaScript with Leaflet maps:
- `js/interactive_map.js` - Main map and route building UI
- `js/api.js` - Backend API client
- `js/semantic_overlays.js` - Nature feature overlay rendering
- `js/app.js` - Application initialization

### Key Data Structures

- `ClientSession` - Per-user session with graph center and active route
- `ActiveRoute` - Current route being built (waypoints, path, distance)
- `ScoredCandidate` - Waypoint candidate with score breakdown and explanation
- `CellFeatures` - Pre-computed features per spatial grid cell (FeatureType enum)

### API Workflow

1. `POST /api/start-session` - Initialize with location/preferences, returns 3 candidates
2. `POST /api/add-waypoint` - Add selected candidate, get 3 new candidates
3. `POST /api/finalize-route` - Complete circular route back to start

## Code Patterns

- Global singletons accessed via `get_route_builder()` and `get_overlay_manager()` (lazy init)
- Profile operations with `@profile_function` decorator or `with profile_operation("name")`
- Feature scores are FeatureType enums, converted to strings for JSON serialization
- Pathfinding uses NetworkX with avoidance of already-visited nodes to prevent out-and-back routes
