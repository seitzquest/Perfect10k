# Perfect10k Backend

Intelligent route planning API with semantic place matching and AI-powered optimization.

## Features

- **Advanced Route Planning**: Uses multiple optimization algorithms (genetic, simulated annealing, value-guided search)
- **Semantic Place Matching**: Natural language preferences ("I like quiet parks near water")
- **Resolution-Based Complexity Reduction**: Handles dense road networks efficiently
- **Value Function Optimization**: Spatial optimization with user preferences, elevation, and accessibility
- **User Management**: Authentication, preferences, route history
- **GPX Export**: Export routes for navigation apps

## Quick Start

### Using Docker (Recommended)

1. **Clone and setup**:
   ```bash
   cd backend
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Start services**:
   ```bash
   docker-compose up -d
   ```

3. **Setup database** (first time only):
   ```bash
   docker-compose exec api python scripts/setup_db.py
   ```

4. **Access the API**:
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

**Note**: The Docker setup uses a simplified build without elevation support for faster builds. For elevation features, use the manual setup with `uv pip install -e ".[elevation]"`.

### Manual Setup (with uv - Recommended)

1. **Quick setup** (automated):
   ```bash
   ./scripts/setup.sh
   ```

2. **Manual setup**:
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install dependencies
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   
   # Configure environment
   cp .env.example .env
   # Edit .env with your settings
   
   # Setup database
   python scripts/setup_db.py
   
   # Start development server
   uv run uvicorn main:app --reload
   ```

3. **Development commands**:
   ```bash
   ./scripts/dev.sh dev      # Start dev server
   ./scripts/dev.sh test     # Run tests
   ./scripts/dev.sh lint     # Run linting
   ./scripts/dev.sh format   # Format code
   ```

## API Documentation

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/me` - Get current user
- `POST /api/auth/logout` - Logout user

### Route Planning
- `POST /api/routes/plan` - Plan a new route
- `GET /api/routes/history` - Get route history
- `GET /api/routes/{id}` - Get specific route
- `POST /api/routes/{id}/export/gpx` - Export route as GPX

### Places
- `POST /api/places/search` - Semantic place search
- `GET /api/places/nearby` - Get nearby places
- `POST /api/places/refresh` - Refresh places cache

### Preferences
- `POST /api/preferences` - Create preference
- `GET /api/preferences` - Get user preferences
- `PUT /api/preferences/{id}` - Update preference
- `DELETE /api/preferences/{id}` - Delete preference

## Route Planning Request

```json
{
  "latitude": 49.807880,
  "longitude": 8.989109,
  "target_distance": 8000,
  "tolerance": 1000,
  "preference_query": "I love quiet parks near water and scenic viewpoints",
  "avoid_roads": true
}
```

## Configuration

Key environment variables:

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/perfect10k

# Security
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Server
HOST=localhost
PORT=8000
DEBUG=true

# ML/Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Route Planning
DEFAULT_TARGET_DISTANCE=8000
MAX_ROUTE_COMPLEXITY=1000
```

## Architecture

### Core Components

- **Route Algorithms** (`core/route_algorithms.py`): Basic route planning logic
- **Advanced Optimizer** (`core/advanced_optimizer.py`): Genetic algorithm, simulated annealing
- **Value Function** (`core/value_function.py`): Spatial optimization with preferences
- **Resolution Strategies** (`core/resolution_strategies.py`): Complexity reduction
- **Embedding Service** (`services/embedding_service.py`): Semantic place matching

### Optimization Strategies

1. **Resolution-Based Reduction**: Adaptively removes edges in dense areas
2. **Value Function Guided**: Uses spatial preferences for route optimization
3. **Multi-Algorithm Approach**: Combines genetic algorithms and simulated annealing
4. **Semantic Matching**: Embeds user preferences and matches to place descriptions

## Development

### Running Tests
```bash
./scripts/dev.sh test
# or
uv run pytest tests/
```

### Database Migrations
```bash
./scripts/dev.sh db-migrate
# or manually:
uv run alembic revision --autogenerate -m "description"
uv run alembic upgrade head
```

### Code Quality
```bash
./scripts/dev.sh format  # Format code
./scripts/dev.sh lint    # Run linting
./scripts/dev.sh check   # Run all checks
```

### Adding New Features

1. **Models**: Add to `models/models.py`
2. **Schemas**: Add to `models/schemas.py` 
3. **Services**: Add to `services/`
4. **Endpoints**: Add to `api/endpoints/`
5. **Tests**: Add to `tests/`

## Performance

- **Route Planning**: ~2-5 seconds for complex routes
- **Semantic Search**: ~100-500ms with embedding cache
- **Complexity Reduction**: Handles graphs with 10k+ nodes
- **Memory Usage**: ~200-500MB depending on area size

## Troubleshooting

### Common Issues

1. **Database Connection**:
   ```bash
   # Check PostgreSQL is running
   pg_isready -h localhost -p 5432
   ```

2. **Missing Dependencies**:
   ```bash
   uv pip install -e ".[dev]"
   ```

3. **Port Already in Use**:
   ```bash
   # Change PORT in .env or kill existing process
   lsof -ti:8000 | xargs kill -9
   ```

4. **Embedding Model Download**:
   ```bash
   # First run downloads ~90MB model
   # Ensure internet connection
   ```

### Logs

Check logs for debugging:
```bash
# Docker
docker-compose logs api

# Manual
tail -f /var/log/perfect10k.log
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `pytest`
5. Submit pull request

## License

MIT License - see LICENSE file for details.