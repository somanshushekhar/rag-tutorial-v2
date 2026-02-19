#!/bin/bash

# Quick AWS Deployment Script for Aflac Assist Chat
# Usage: ./deploy-to-aws.sh <EC2_IP_ADDRESS> <PATH_TO_SSH_KEY>

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <EC2_IP_ADDRESS> <SSH_KEY_PATH>"
    echo "Example: $0 54.123.45.67 ~/.ssh/my-key.pem"
    exit 1
fi

EC2_IP=$1
SSH_KEY=$2

echo "=== Deploying Aflac Assist Chat to AWS ==="
echo "Target: ubuntu@$EC2_IP"
echo ""

# Test SSH connection
echo "Testing SSH connection..."
ssh -i $SSH_KEY -o ConnectTimeout=10 ubuntu@$EC2_IP "echo 'Connection successful'" || {
    echo "ERROR: Cannot connect to EC2 instance"
    exit 1
}

# Upload deployment script
echo "Uploading deployment script..."
scp -i $SSH_KEY ec2-user-data.sh ubuntu@$EC2_IP:~/setup.sh

# Run installation
echo "Running installation on EC2..."
ssh -i $SSH_KEY ubuntu@$EC2_IP "chmod +x ~/setup.sh && sudo ~/setup.sh"

# Upload PDF files (if they exist locally)
if [ -d "data" ] && [ "$(ls -A data/*.pdf 2>/dev/null)" ]; then
    echo "Uploading PDF files..."
    scp -i $SSH_KEY data/*.pdf ubuntu@$EC2_IP:/opt/aflac-assist/data/
    
    # Ingest documents
    echo "Ingesting documents..."
    ssh -i $SSH_KEY ubuntu@$EC2_IP "cd /opt/aflac-assist && source .venv/bin/activate && python populate_database.py"
fi

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Application URL: http://$EC2_IP:8000"
echo ""
echo "To check status:"
echo "  ssh -i $SSH_KEY ubuntu@$EC2_IP"
echo "  sudo systemctl status aflac-assist"
echo ""
echo "To view logs:"
echo "  sudo journalctl -u aflac-assist -f"
