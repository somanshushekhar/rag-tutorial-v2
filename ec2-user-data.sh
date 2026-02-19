#!/bin/bash

# AWS EC2 User Data Script for RAG Application (t3.large optimized)
# This script runs automatically when the EC2 instance starts

set -e

echo "=== Starting RAG Application Setup ==="
echo "Instance Type: t3.large (2 vCPU, 8 GB RAM)"

echo "=== Updating System ==="
apt-get update
apt-get upgrade -y

echo "=== Installing Dependencies ==="
apt-get install -y curl git python3-pip python3-venv htop

echo "=== Installing Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh

# Wait for Ollama to start
sleep 10

echo "=== Pulling Ollama Models (this takes 5-10 minutes) ==="
ollama pull nomic-embed-text
ollama pull mistral

echo "=== Cloning Application Repository ==="
cd /home/ubuntu
git clone https://github.com/somanshushekhar/rag-tutorial-v2.git
chown -R ubuntu:ubuntu rag-tutorial-v2

echo "=== Installing Python Dependencies ==="
cd rag-tutorial-v2
sudo -u ubuntu python3 -m pip install -r requirements.txt
echo "=== Creating systemd Service ==="
cat > /etc/systemd/system/rag-app.service <<EOF
[Unit]
Description=RAG Application
After=network.target ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/rag-tutorial-v2
Environment="OLLAMA_BASE_URL=http://127.0.0.1:11434"
Environment="OLLAMA_EMBED_MODEL=nomic-embed-text"
ExecStart=/usr/bin/python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable rag-app
systemctl start rag-app

echo "=== Installation Complete ==="
echo "Application running on http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
echo "=== t3.large Setup Successful ===\"
