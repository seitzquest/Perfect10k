#!/bin/bash

# Perfect10k Server Setup Script
# Run this script on your Raspberry Pi to set up the complete server environment

set -e

# Configuration - CHANGE THESE VALUES
DOMAIN="seitzquest.com"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_error "This script should not be run as root. Run as a regular user with sudo privileges."
   exit 1
fi

# Domain is already set to seitzquest.com - no changes needed

log_info "Starting Perfect10k server setup for domain: $DOMAIN"

# Step 1: Update system
log_step "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Step 2: Install required packages
log_step "Installing required packages..."
sudo apt install -y nginx git curl htop iotop fail2ban ufw python3-pip python3-venv

# Step 2b: Install geospatial system dependencies
log_step "Installing geospatial system dependencies..."
# Add newer repository for updated geospatial libraries
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable -y
sudo apt update

# Install updated geospatial libraries
sudo apt install -y libproj-dev proj-data proj-bin libgeos-dev libspatialindex-dev libffi-dev libgdal-dev gdal-bin

# Step 3: Install uv globally
log_step "Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sudo sh

# Step 3: Configure firewall
log_step "Configuring firewall..."
sudo ufw --force enable
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'

# Step 4: Enable and start Nginx
log_step "Setting up Nginx..."
sudo systemctl enable nginx
sudo systemctl start nginx

# Step 5: Clone repository
log_step "Cloning Perfect10k repository..."
cd /var/www/
if [ -d "Perfect10k" ]; then
    log_warn "Perfect10k directory already exists, backing up..."
    sudo mv Perfect10k Perfect10k.backup.$(date +%Y%m%d-%H%M%S)
fi

sudo git clone https://github.com/yourusername/Perfect10k.git
sudo chown -R www-data:www-data Perfect10k
sudo chmod -R 755 Perfect10k

# Step 6: Set up directory structure and dependencies
log_step "Setting up directory structure..."
sudo mkdir -p /var/www/Perfect10k/logs
sudo chown -R www-data:www-data /var/www/Perfect10k/logs

log_step "Installing Python dependencies with uv..."
cd /var/www/Perfect10k
sudo chown -R $USER:$USER /var/www/Perfect10k
uv sync
sudo chown -R www-data:www-data /var/www/Perfect10k

# Step 7: Configure Nginx
log_step "Configuring Nginx..."
sudo cp /var/www/Perfect10k/config/nginx/perfect10k-cloudflare.conf /etc/nginx/sites-available/perfect10k

# Enable site and remove default
sudo ln -sf /etc/nginx/sites-available/perfect10k /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
sudo nginx -t

# Step 8: Start Nginx
log_step "Starting Nginx with Cloudflare configuration..."
sudo systemctl reload nginx

# Step 12: Set up log rotation
log_step "Setting up log rotation..."
sudo cp /var/www/Perfect10k/config/logrotate/perfect10k /etc/logrotate.d/

# Step 13: Set up fail2ban
log_step "Setting up fail2ban..."
sudo cp /var/www/Perfect10k/config/fail2ban/perfect10k.conf /etc/fail2ban/jail.d/
sudo systemctl enable fail2ban
sudo systemctl restart fail2ban


# Step 15: Test installation
log_step "Testing installation..."
sleep 5

# Test HTTP to HTTPS redirect
if curl -s -I "http://$DOMAIN" | grep -q "301"; then
    log_info "✓ HTTP to HTTPS redirect working"
else
    log_warn "⚠ HTTP to HTTPS redirect may not be working"
fi

# Test HTTPS
if curl -f -s -I "https://$DOMAIN/health" >/dev/null; then
    log_info "✓ HTTPS connection successful"
else
    log_warn "⚠ HTTPS connection test failed"
fi

# SSL is handled by Cloudflare - no local certificates needed
log_info "✓ SSL handled by Cloudflare proxy"

# Final instructions
log_info "🎉 Setup completed successfully!"
echo
echo -e "${GREEN}Your Perfect10k application is now available at:${NC}"
echo -e "${BLUE}https://$DOMAIN${NC}"
echo
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Test your application in a web browser"
echo "2. If you have a backend API, set it up using the systemd service configuration"
echo "3. Monitor logs: sudo tail -f /var/log/nginx/perfect10k_access.log"
echo "4. Use the deploy script for future updates: /var/www/Perfect10k/scripts/deploy.sh"
echo
echo -e "${YELLOW}Useful commands:${NC}"
echo "- SSL is managed by Cloudflare - no local certificate management needed"
echo "- Check fail2ban status: sudo fail2ban-client status"
echo "- View Nginx logs: sudo tail -f /var/log/nginx/perfect10k_error.log"