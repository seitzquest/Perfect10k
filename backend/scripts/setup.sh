#!/bin/bash
set -e

# Perfect10k Backend Setup Script
# This script sets up the development environment using uv

echo "🚀 Setting up Perfect10k Backend with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "✅ uv version: $(uv --version)"

# Check Python version
echo "🐍 Checking Python version..."
if ! uv python list | grep -q "3.11"; then
    echo "📥 Installing Python 3.11..."
    uv python install 3.11
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🏗️  Creating virtual environment..."
    uv venv --python 3.11
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
uv pip install -e .

# Install development dependencies
echo "🛠️  Installing development dependencies..."
uv pip install -e ".[dev,test]"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "📝 Please edit .env with your configuration"
fi

# Setup pre-commit hooks
echo "🔗 Setting up pre-commit hooks..."
pre-commit install

echo ""
echo "✅ Setup complete! Next steps:"
echo ""
echo "1. Edit .env with your configuration:"
echo "   nano .env"
echo ""
echo "2. Start PostgreSQL (if not using Docker):"
echo "   # On macOS with Homebrew:"
echo "   brew services start postgresql"
echo "   # On Ubuntu:"
echo "   sudo systemctl start postgresql"
echo ""
echo "3. Setup the database:"
echo "   python scripts/setup_db.py"
echo ""
echo "4. Start the development server:"
echo "   uv run python run.py"
echo ""
echo "   Or with hot reload:"
echo "   uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "5. Alternative: Use Docker Compose:"
echo "   docker-compose up -d"
echo ""
echo "🎉 Happy coding!"