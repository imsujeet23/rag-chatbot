#!/bin/bash

# RAG Chatbot Startup Script
set -e

echo "🚀 RAG Chatbot System - Startup Script"
echo "======================================"

# Check Python version
echo "✓ Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Found Python $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "✓ Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "✓ Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "✓ Installing dependencies..."
pip install -q -r requirements.txt

# Create necessary directories
echo "✓ Creating directories..."
mkdir -p uploads vector_store models tests

# Show usage instructions
echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "==========="
echo ""
echo "1. Add your documents (PDF/TXT) to the 'uploads' folder"
echo ""
echo "2. Start the API server:"
echo "   uvicorn app.api:app --reload"
echo ""
echo "3. Open browser and visit:"
echo "   - API: http://localhost:8000"
echo "   - Docs: http://localhost:8000/docs"
echo ""
echo "4. Or run the example script:"
echo "   python example.py"
echo ""
echo "5. For development with testing:"
echo "   pip install -r requirements-dev.txt"
echo "   pytest tests/ -v"
echo ""
echo "Commands quick reference:"
echo "========================"
echo "  uvicorn app.api:app --reload          # Start dev server with auto-reload"
echo "  python example.py                      # Run example pipeline"
echo "  pytest tests/ -v                      # Run all tests"
echo "  pytest tests/ -v --cov                # Run tests with coverage"
echo "  docker-compose up                     # Run in Docker"
echo ""
