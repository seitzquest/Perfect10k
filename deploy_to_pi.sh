#!/bin/bash

# Deploy Perfect10k Docker image to Raspberry Pi
# Usage: ./deploy_to_pi.sh

set -e

PI_IP="192.168.178.87"
PI_USER="pi"
IMAGE_NAME="perfect10k-app"
TAR_FILE="perfect10k-app.tar"

echo "🚀 Deploying Perfect10k to Raspberry Pi at $PI_IP"

echo "🔨 Building Docker image..."
docker build -t "$IMAGE_NAME:latest" .

echo "💾 Saving Docker image to tar file..."
docker save "$IMAGE_NAME:latest" -o "$TAR_FILE"

echo "📦 Transferring Docker image to Raspberry Pi..."
scp "$TAR_FILE" "$PI_USER@$PI_IP:/home/$PI_USER/"

echo "🐳 Loading Docker image and restarting service on Raspberry Pi..."
ssh "$PI_USER@$PI_IP" << EOF
    echo "Loading Docker image..."
    docker load -i /home/$PI_USER/$TAR_FILE
    
    echo "Restarting perfect10k-docker service..."
    sudo systemctl restart perfect10k-docker
    
    echo "Cleaning up tar file..."
    rm /home/$PI_USER/$TAR_FILE
    
    echo "Service restarted successfully!"
    sudo systemctl status perfect10k-docker
EOF

echo "🧹 Cleaning up local tar file..."
rm "$TAR_FILE"

echo "✅ Deployment complete! Your app should be running at http://$PI_IP:8000"
echo "📊 Check status with: ssh $PI_USER@$PI_IP 'docker logs perfect10k-app'"