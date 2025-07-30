#!/bin/bash

echo "Starting WasteSort AI Development Environment"

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "Port $1 is already in use"
        return 1
    fi
    return 0
}

# Function to kill processes on specific ports
kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port)
    if [ ! -z "$pid" ]; then
        echo "Killing process on port $port (PID: $pid)"
        kill -9 $pid
    fi
}

# Check if required files exist
if [ ! -f "package.json" ]; then
    echo "Error: package.json not found. Run this script from the project root."
    exit 1
fi

if [ ! -f "fastapi_backend.py" ]; then
    echo "Error: fastapi_backend.py not found."
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install Node.js dependencies"
        exit 1
    fi
fi

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import fastapi, uvicorn, torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing Python dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install Python dependencies"
        exit 1
    fi
fi

# Clean up any existing processes on our ports
kill_port 3000
kill_port 8000

# Start backend server
echo "Starting FastAPI backend on port 8000..."
python3 fastapi_backend.py &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to initialize..."
sleep 3

# Check if backend started successfully
for i in {1..10}; do
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        echo "Backend started successfully"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "Error: Backend failed to start after 30 seconds"
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
    sleep 3
done

# Start frontend development server
echo "Starting Next.js development server on port 3000..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "WasteSort AI is running:"
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    kill_port 3000
    kill_port 8000
    echo "Services stopped"
    exit 0
}

# Set up signal handling
trap cleanup INT TERM

# Wait for processes
wait
