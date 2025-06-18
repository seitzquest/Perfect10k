#!/bin/bash

# Perfect10k Deployment Script
# Pulls latest code, rebuilds containers, and restarts the service

set -e  # Exit on any error

echo "🚀 Starting Perfect10k deployment..."

# Function to print colored output
print_step() {
    echo -e "\n\033[1;34m==> $1\033[0m"
}

print_error() {
    echo -e "\033[1;31mERROR: $1\033[0m"
}

print_success() {
    echo -e "\033[1;32m✅ $1\033[0m"
}

# Change to project directory
cd "$(dirname "$0")"
PROJECT_DIR=$(pwd)
print_step "Working in directory: $PROJECT_DIR"

# Step 1: Git pull
print_step "Pulling latest changes from git..."
if git pull; then
    print_success "Git pull completed successfully"
else
    print_error "Git pull failed"
    exit 1
fi

# Step 2: Stop the systemd service
print_step "Stopping perfect10k-docker.service..."
if sudo systemctl stop perfect10k-docker.service; then
    print_success "Service stopped successfully"
else
    print_error "Failed to stop service (continuing anyway)"
fi

# Step 3: Kill any running Perfect10k containers
print_step "Killing existing Docker containers..."
CONTAINER_NAME="perfect10k-backend"

# Find and kill containers by name pattern
RUNNING_CONTAINERS=$(docker ps -q --filter "name=$CONTAINER_NAME" 2>/dev/null || true)
if [ ! -z "$RUNNING_CONTAINERS" ]; then
    echo "Found running containers: $RUNNING_CONTAINERS"
    docker kill $RUNNING_CONTAINERS
    print_success "Killed running containers"
else
    echo "No running containers found"
fi

# Remove stopped containers
STOPPED_CONTAINERS=$(docker ps -aq --filter "name=$CONTAINER_NAME" 2>/dev/null || true)
if [ ! -z "$STOPPED_CONTAINERS" ]; then
    echo "Removing stopped containers: $STOPPED_CONTAINERS"
    docker rm $STOPPED_CONTAINERS
    print_success "Removed stopped containers"
fi

# Step 4: Docker compose build
print_step "Building Docker containers..."
if docker-compose build --no-cache; then
    print_success "Docker build completed successfully"
else
    print_error "Docker build failed"
    exit 1
fi

# Step 5: Clean up any dangling images (optional, saves space)
print_step "Cleaning up dangling Docker images..."
DANGLING_IMAGES=$(docker images -q --filter "dangling=true" 2>/dev/null || true)
if [ ! -z "$DANGLING_IMAGES" ]; then
    docker rmi $DANGLING_IMAGES
    print_success "Cleaned up dangling images"
else
    echo "No dangling images to clean"
fi

# Step 6: Start the systemd service
print_step "Starting perfect10k-docker.service..."
if sudo systemctl start perfect10k-docker.service; then
    print_success "Service started successfully"
else
    print_error "Failed to start service"
    exit 1
fi

# Step 7: Check service status
print_step "Checking service status..."
sleep 3  # Give service time to start

if sudo systemctl is-active --quiet perfect10k-docker.service; then
    print_success "Service is running successfully!"
    
    # Show service status
    echo ""
    sudo systemctl status perfect10k-docker.service --no-pager -l
    
    # Show recent logs
    print_step "Recent service logs:"
    sudo journalctl -u perfect10k-docker.service --no-pager -l -n 20
    
else
    print_error "Service failed to start properly"
    echo ""
    print_step "Service status:"
    sudo systemctl status perfect10k-docker.service --no-pager -l
    
    print_step "Recent error logs:"
    sudo journalctl -u perfect10k-docker.service --no-pager -l -n 20
    exit 1
fi

print_success "🎉 Perfect10k deployment completed successfully!"
echo ""
echo "Service should be available at the configured endpoint."
echo "Use 'sudo journalctl -u perfect10k-docker.service -f' to follow logs."