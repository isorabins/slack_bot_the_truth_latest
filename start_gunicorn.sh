#!/bin/bash

# Function to handle termination signal
term_handler() {
    echo "Stopping Gunicorn..."
    kill -SIGTERM "$gunicorn_pid" 2>/dev/null
    wait "$gunicorn_pid"
    exit 0
}

# Trap termination signals
trap term_handler SIGTERM SIGINT

# Trap termination signals
trap term_handler SIGTERM SIGINT

# Start Gunicorn to serve the Flask app
gunicorn -b :3000 "app:flask_app" &
gunicorn_pid=$!

# Wait for processes to exit
wait "$gunicorn_pid"

