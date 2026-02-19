#!/bin/bash

# Start Ollama server in background
ollama serve &

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
sleep 5

# Pull required models
echo "Pulling Ollama models..."
ollama pull nomic-embed-text
ollama pull mistral

# Start FastAPI application
echo "Starting FastAPI application..."
uvicorn app:app --host 0.0.0.0 --port 8000
