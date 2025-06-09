#!/bin/bash
set -e

# Perfect10k Development Script
# Quick commands for common development tasks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_help() {
    echo "Perfect10k Development Helper"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  install     Install dependencies with uv"
    echo "  dev         Start development server with hot reload"
    echo "  test        Run tests"
    echo "  lint        Run linting (ruff, black, mypy)"
    echo "  format      Format code (black, isort)"
    echo "  db-setup    Setup database"
    echo "  db-migrate  Create and run database migrations"
    echo "  db-reset    Reset database (WARNING: deletes all data)"
    echo "  docker      Start with Docker Compose"
    echo "  clean       Clean cache and temp files"
    echo "  check       Run all checks (lint, test, type)"
    echo ""
}

function run_install() {
    echo -e "${BLUE}📦 Installing dependencies...${NC}"
    uv pip install -e ".[dev,test]"
    echo -e "${GREEN}✅ Dependencies installed${NC}"
}

function run_dev() {
    echo -e "${BLUE}🚀 Starting development server...${NC}"
    echo -e "${YELLOW}Server will be available at: http://localhost:8000${NC}"
    echo -e "${YELLOW}API docs available at: http://localhost:8000/docs${NC}"
    uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
}

function run_test() {
    echo -e "${BLUE}🧪 Running tests...${NC}"
    uv run pytest tests/ -v
}

function run_lint() {
    echo -e "${BLUE}🔍 Running linting...${NC}"
    echo "Running ruff..."
    uv run ruff check .
    echo "Running black..."
    uv run black --check .
    echo "Running mypy..."
    uv run mypy .
    echo -e "${GREEN}✅ Linting complete${NC}"
}

function run_format() {
    echo -e "${BLUE}🎨 Formatting code...${NC}"
    uv run black .
    uv run isort .
    uv run ruff check --fix .
    echo -e "${GREEN}✅ Code formatted${NC}"
}

function run_db_setup() {
    echo -e "${BLUE}🗄️  Setting up database...${NC}"
    uv run python scripts/setup_db.py
    echo -e "${GREEN}✅ Database setup complete${NC}"
}

function run_db_migrate() {
    echo -e "${BLUE}🔄 Creating and running migrations...${NC}"
    uv run alembic revision --autogenerate -m "Auto migration $(date +%Y%m%d_%H%M%S)"
    uv run alembic upgrade head
    echo -e "${GREEN}✅ Migrations complete${NC}"
}

function run_db_reset() {
    echo -e "${RED}⚠️  WARNING: This will delete ALL data!${NC}"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}🗑️  Resetting database...${NC}"
        uv run alembic downgrade base
        uv run alembic upgrade head
        uv run python scripts/setup_db.py
        echo -e "${GREEN}✅ Database reset complete${NC}"
    else
        echo "Cancelled"
    fi
}

function run_docker() {
    echo -e "${BLUE}🐳 Starting with Docker Compose...${NC}"
    docker-compose up -d
    echo -e "${GREEN}✅ Services started${NC}"
    echo "API: http://localhost:8000"
    echo "Postgres: localhost:5432"
    echo "Redis: localhost:6379"
}

function run_clean() {
    echo -e "${BLUE}🧹 Cleaning cache and temp files...${NC}"
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    rm -rf .pytest_cache 2>/dev/null || true
    rm -rf .mypy_cache 2>/dev/null || true
    rm -rf .ruff_cache 2>/dev/null || true
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

function run_check() {
    echo -e "${BLUE}🔍 Running all checks...${NC}"
    run_format
    run_lint
    run_test
    echo -e "${GREEN}✅ All checks passed${NC}"
}

# Main script logic
case "$1" in
    install)
        run_install
        ;;
    dev)
        run_dev
        ;;
    test)
        run_test
        ;;
    lint)
        run_lint
        ;;
    format)
        run_format
        ;;
    db-setup)
        run_db_setup
        ;;
    db-migrate)
        run_db_migrate
        ;;
    db-reset)
        run_db_reset
        ;;
    docker)
        run_docker
        ;;
    clean)
        run_clean
        ;;
    check)
        run_check
        ;;
    *)
        print_help
        exit 1
        ;;
esac