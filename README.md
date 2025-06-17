# Perfect10k - Interactive Route Planner

AI-powered route builder for 10k daily walks with step-by-step interactive planning.

## Quick Setup

```bash
# Install dependencies
uv sync

# Create SSL certificates (for mobile geolocation)
./create_ssl_certs.sh

# Start server
python run.py
```

Access at **http://localhost:8000**

## How It Works

1. **Set Location** - Click map or use current location
2. **Start Route** - Get 3 smart waypoint candidates ~1km away  
3. **Build Interactively** - Tap candidates to add waypoints
4. **Complete Loop** - Finalize circular route back to start
5. **Export** - Download GPX file

## Features

- 🤖 **AI Preferences** - Describe your ideal route ("scenic parks", "quiet streets")
- 📱 **Mobile-First** - Touch-optimized interface
- 🗺️ **Global Coverage** - Works worldwide via OpenStreetMap
- ⚡ **Real-time** - Interactive route building with instant feedback
- 📍 **Smart Planning** - Conflict avoidance and distance optimization

## API Endpoints

- `POST /api/start-session` - Initialize route with location & preferences
- `POST /api/add-waypoint` - Add waypoint, get new candidates
- `POST /api/finalize-route` - Complete circular route
- `GET /api/route-status/{session_id}` - Get current route stats

## Tech Stack

- **Backend**: Python FastAPI + OSMNx + NetworkX
- **Frontend**: Vanilla JS + Leaflet maps
- **AI**: Semantic preference matching with embeddings