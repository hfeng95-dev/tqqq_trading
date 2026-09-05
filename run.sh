#!/bin/bash

# Navigate to script directory
cd "$(dirname "$0")"

# Define the path to the virtual environment
VENV_DIR=".venv"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install -q -r requirements.txt
else
    source "$VENV_DIR/bin/activate"
fi

# Find next available port starting from 8505
PORT=8505
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; do
    PORT=$((PORT+1))
done

echo "Launching TQQQ Trading AI Terminal on port $PORT..."
streamlit run app.py --server.port $PORT --server.headless true
