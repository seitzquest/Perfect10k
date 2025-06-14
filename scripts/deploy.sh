#!/bin/bash

# Perfect10k Deployment Script
# Run this script to deploy updates to your Raspberry Pi server

set -e  # Exit on any error

# Configuration
DOMAIN="seitzquest.com"  # Change this to your domain
REPO_DIR="/var/www/Perfect10k"
NGINX_CONFIG="/etc/nginx/sites-available/perfect10k"
BACKUP_DIR="/var/backups/perfect10k"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_error "This script should not be run as root"
   exit 1
fi

# Create backup directory
sudo mkdir -p "$BACKUP_DIR"

# Backup current installation
log_info "Creating backup..."
sudo tar -czf "$BACKUP_DIR/perfect10k-backup-$(date +%Y%m%d-%H%M%S).tar.gz" -C /var/www Perfect10k

# Update system packages
log_info "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Pull latest code
log_info "Pulling latest code..."
cd "$REPO_DIR"
sudo git fetch origin
sudo git reset --hard origin/main
sudo chown -R www-data:www-data "$REPO_DIR"
sudo chmod -R 755 "$REPO_DIR"

# Update Nginx configuration if needed
if [ -f "$REPO_DIR/config/nginx/perfect10k.conf" ]; then
    log_info "Updating Nginx configuration..."
    sudo cp "$REPO_DIR/config/nginx/perfect10k.conf" "$NGINX_CONFIG"
    sudo sed -i "s/seitzquest.com/$DOMAIN/g" "$NGINX_CONFIG"
fi

# Test Nginx configuration
log_info "Testing Nginx configuration..."
sudo nginx -t

# Update systemd service if it exists
if [ -f "$REPO_DIR/config/systemd/perfect10k-api.service" ] && systemctl is-enabled perfect10k-api >/dev/null 2>&1; then
    log_info "Updating systemd service..."
    sudo cp "$REPO_DIR/config/systemd/perfect10k-api.service" /etc/systemd/system/
    sudo systemctl daemon-reload
    
    # Update Python dependencies if requirements.txt exists
    if [ -f "$REPO_DIR/requirements.txt" ]; then
        log_info "Updating Python dependencies..."
        sudo -u www-data "$REPO_DIR/venv/bin/pip" install -r "$REPO_DIR/requirements.txt"
    fi
    
    # Restart API service
    log_info "Restarting API service..."
    sudo systemctl restart perfect10k-api
fi

# Reload Nginx
log_info "Reloading Nginx..."
sudo systemctl reload nginx

# Check service status
log_info "Checking service status..."
if systemctl is-active --quiet nginx; then
    log_info "Nginx is running"
else
    log_error "Nginx is not running!"
    exit 1
fi

if systemctl is-enabled perfect10k-api >/dev/null 2>&1; then
    if systemctl is-active --quiet perfect10k-api; then
        log_info "API service is running"
    else
        log_error "API service is not running!"
        exit 1
    fi
fi

# Test HTTPS connection
log_info "Testing HTTPS connection..."
if curl -f -s -I "https://$DOMAIN/health" >/dev/null; then
    log_info "HTTPS connection successful"
else
    log_warn "HTTPS connection test failed"
fi

# Clean up old backups (keep last 10)
log_info "Cleaning up old backups..."
sudo find "$BACKUP_DIR" -name "perfect10k-backup-*.tar.gz" -type f -mtime +30 -delete

log_info "Deployment completed successfully!"
log_info "Your app should be available at: https://$DOMAIN"