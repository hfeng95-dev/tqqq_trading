#!/usr/bin/env bash
set -euo pipefail

# Navigate to script directory
cd "$(dirname "$0")"

VENV_DIR=".venv"
REQ_FILE="requirements.txt"
HASH_FILE="$VENV_DIR/.req_hash"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# POSIX-safe activation (works under both `sh run.sh` and `./run.sh`)
. "$VENV_DIR/bin/activate"

# Hash-gated dependency sync (skips pip when requirements.txt is unchanged)
if [ -f "$REQ_FILE" ]; then
    CURR_HASH=$(python -c "import hashlib, pathlib; print(hashlib.sha256(pathlib.Path('$REQ_FILE').read_bytes()).hexdigest())")
    PREV_HASH=""
    if [ -f "$HASH_FILE" ]; then
        PREV_HASH=$(cat "$HASH_FILE")
    fi
    if [ "$CURR_HASH" != "$PREV_HASH" ]; then
        echo "Syncing dependencies from $REQ_FILE..."
        python -m pip install -q -r "$REQ_FILE"
        printf "%s" "$CURR_HASH" > "$HASH_FILE"
    fi
fi

# Find next available port starting from 8505 via Python socket probe
PORT=$(python -c "
import socket
for port in range(8505, 8605):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('127.0.0.1', port)) != 0:
            print(port)
            break
else:
    print(8505)
")

echo "Launching TQQQ Trading AI Terminal on http://localhost:$PORT ..."
exec streamlit run app.py --server.port "$PORT" --server.headless true
