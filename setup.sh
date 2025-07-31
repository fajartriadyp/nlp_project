#!/bin/bash

# Setup script untuk deployment Crypto Sentiment Analysis
echo "🚀 Setting up Crypto Sentiment Analysis..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p .streamlit

# Create Streamlit config
echo "⚙️ Creating Streamlit configuration..."
cat > .streamlit/config.toml << EOF
[global]
developmentMode = false

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
EOF

echo "✅ Setup completed successfully!"
echo "🎯 To run the app: streamlit run streamlit_app.py" 