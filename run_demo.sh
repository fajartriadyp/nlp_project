#!/bin/bash

# Crypto Sentiment Analysis - Demo Runner
# Author: Fajar Triady Putra

echo "🎯 Starting Crypto Sentiment Analysis Demo..."
echo "=============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements if not already installed
if ! python -c "import streamlit" &> /dev/null; then
    echo "📚 Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the demo
echo "🎯 Starting demo application..."
echo "📱 The demo will open in your browser at http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the demo"
echo "=============================================="

streamlit run demo.py

# Deactivate virtual environment when demo stops
deactivate
echo "👋 Demo stopped. Virtual environment deactivated." 