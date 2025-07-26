#!/bin/bash

# Smart Waste Classifier - Run Script
echo "🚀 Starting Smart Waste Classifier"
echo "=================================="

# Check if model file exists
if [ ! -f "waste_classifier.pth" ]; then
    echo "❌ Model file 'waste_classifier.pth' not found!"
    echo "Please run the training script first:"
    echo "python model_training.py"
    exit 1
fi

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        return 0
    else
        return 1
    fi
}

# Start FastAPI backend
echo "🔧 Starting FastAPI backend..."
if check_port 8000; then
    echo "⚠️  Port 8000 is already in use. Killing existing process..."
    kill $(lsof -ti:8000) 2>/dev/null || true
    sleep 2
fi

# Start API in background
python fastapi_backend.py &
API_PID=$!
echo "📡 FastAPI started with PID: $API_PID"

# Wait for API to start
echo "⏳ Waiting for API to start..."
sleep 5

# Test API health
echo "🩺 Testing API health..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API is healthy!"
else
    echo "❌ API health check failed!"
    kill $API_PID 2>/dev/null
    exit 1
fi

# Start Streamlit frontend
echo "🌐 Starting Streamlit frontend..."
if check_port 8501; then
    echo "⚠️  Port 8501 is already in use. Killing existing process..."
    kill $(lsof -ti:8501) 2>/dev/null || true
    sleep 2
fi

streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
STREAMLIT_PID=$!
echo "🎨 Streamlit started with PID: $STREAMLIT_PID"

echo ""
echo "🎉 Application started successfully!"
echo "=================================="
echo "📡 API Backend: http://localhost:8000"
echo "🌐 Web Interface: http://localhost:8501"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $API_PID 2>/dev/null || true
    kill $STREAMLIT_PID 2>/dev/null || true
    echo "✅ Services stopped"
    exit 0
}

# Trap interrupt signal
trap cleanup INT

# Wait for interrupt
wait