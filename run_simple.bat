@echo off
REM Crypto Sentiment Analysis - Notebook Results Viewer (Windows)
REM Author: Fajar Triady Putra

echo 📊 Starting Crypto Sentiment Analysis - Notebook Results Viewer...
echo ==================================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements if not already installed
if not exist "venv\Scripts\streamlit.exe" (
    echo 📚 Installing dependencies...
    pip install -r requirements.txt
)

REM Run the notebook results viewer
echo 📊 Starting notebook results viewer...
echo 📱 The app will open in your browser at http://localhost:8501
echo 🛑 Press Ctrl+C to stop the app
echo ==================================================================

streamlit run app_simple.py

REM Deactivate virtual environment when app stops
call venv\Scripts\deactivate.bat
echo 👋 App stopped. Virtual environment deactivated.
pause 