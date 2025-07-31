#!/bin/bash

# Crypto Sentiment Analysis - Notebook Results Viewer
# Author: Fajar Triady Putra

echo "📊 Starting Crypto Sentiment Analysis - Notebook Results Viewer..."
echo "=================================================================="

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

# Run the notebook results viewer
echo "📊 Starting notebook results viewer..."
echo "📱 The app will open in your browser at http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the app"
echo "=================================================================="

streamlit run app_simple.py

# Deactivate virtual environment when app stops
deactivate
echo "👋 App stopped. Virtual environment deactivated." 