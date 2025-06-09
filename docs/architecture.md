# Perfect10k System Architecture

## Overview
Perfect10k evolves from a Jupyter prototype to a full-stack application for intelligent route planning with semantic place matching.

## System Components

### Backend (Python FastAPI)
- **API Layer**: REST endpoints for route planning, user preferences, place search
- **Core Engine**: Route planning algorithms with resolution-based complexity reduction
- **Semantic Engine**: Place embedding and semantic matching system
- **Models**: Data models for routes, places, user preferences
- **Services**: External integrations (OSM, elevation data, embeddings)

### Frontend (React + Leaflet)
- **Route Planner**: Interactive map interface for route generation
- **Preference Manager**: UI for describing favorite places and interests
- **Route Visualizer**: Display generated routes with semantic highlights
- **Profile Manager**: User preference storage and management

### Database (PostgreSQL + Vector Extensions)
- **Users**: User profiles and authentication
- **Routes**: Generated route history and metadata
- **Places**: Cached place data with semantic embeddings
- **Preferences**: User preference vectors and descriptions

## Key Algorithms

### Resolution-Based Route Planning
```
1. Load OSM data for region
2. Apply resolution filter (randomly hide edges based on density)
3. Use value function estimation for route optimization
4. Refine route with full resolution in selected areas
```

### Semantic Place Matching
```
1. User describes preferences ("I like lakes, quiet parks")
2. Generate embedding from description
3. Match against place embeddings in area
4. Weight route planning toward semantically similar places
```

### Value Function Framework
```
- Tile-based spatial representation
- User preference weighting
- Elevation and terrain factors
- Route diversity scoring
```

## API Design

### Core Endpoints
- `POST /api/routes/plan` - Generate route with preferences
- `GET/POST /api/places/search` - Semantic place search
- `POST /api/preferences` - Store user preferences
- `GET /api/routes/history` - User route history

### Request/Response Flow
```
Frontend → API Gateway → Route Planner → Semantic Engine → Database
                     ↓
                 OSM Service ← Elevation Service
```

## Technology Stack
- **Backend**: FastAPI, SQLAlchemy, PostgreSQL, sentence-transformers
- **Frontend**: React, Leaflet, Axios
- **ML/Embeddings**: sentence-transformers, numpy, scipy
- **Geospatial**: OSMnx, geopandas, rasterio
- **Infrastructure**: Docker, Redis (caching)