#!/bin/bash

# Perfect10k Server Setup Script
# Run this script on your Raspberry Pi to set up the complete server environment

set -e

# Configuration - CHANGE THESE VALUES
DOMAIN="seitzquest.com"
CLOUDFLARE_API_TOKEN="your_cloudflare_api_token_here"
EMAIL="your-email@example.com"

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

# Check if configuration is set
if [[ "$DOMAIN" == "seitzquest.com" || "$CLOUDFLARE_API_TOKEN" == "your_cloudflare_api_token_here" ]]; then
    log_error "Please edit this script and set your DOMAIN and CLOUDFLARE_API_TOKEN"
    exit 1
fi

log_info "Starting Perfect10k server setup for domain: $DOMAIN"

# Step 1: Update system
log_step "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Step 2: Install required packages
log_step "Installing required packages..."
sudo apt install -y nginx git certbot python3-certbot-nginx python3-certbot-dns-cloudflare \
    curl htop iotop fail2ban ufw python3-pip python3-venv

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

# Step 6: Set up directory structure
log_step "Setting up directory structure..."
sudo mkdir -p /var/www/Perfect10k/logs
sudo chown -R www-data:www-data /var/www/Perfect10k/logs

# Step 7: Configure Cloudflare credentials
log_step "Setting up Cloudflare credentials..."
sudo mkdir -p /etc/letsencrypt
echo "dns_cloudflare_api_token = $CLOUDFLARE_API_TOKEN" | sudo tee /etc/letsencrypt/cloudflare.ini
sudo chmod 600 /etc/letsencrypt/cloudflare.ini

# Step 8: Configure Nginx
log_step "Configuring Nginx..."
sudo cp /var/www/Perfect10k/config/nginx/perfect10k.conf /etc/nginx/sites-available/perfect10k
sudo sed -i "s/seitzquest.com/$DOMAIN/g" /etc/nginx/sites-available/perfect10k

# Enable site and remove default
sudo ln -sf /etc/nginx/sites-available/perfect10k /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
sudo nginx -t

# Step 9: Start with HTTP-only configuration for certificate generation
log_step "Starting with HTTP-only configuration..."
sudo systemctl reload nginx

# Step 10: Obtain SSL certificate
log_step "Obtaining SSL certificate from Let's Encrypt..."
sudo certbot certonly \
  --dns-cloudflare \
  --dns-cloudflare-credentials /etc/letsencrypt/cloudflare.ini \
  --dns-cloudflare-propagation-seconds 60 \
  --email "$EMAIL" \
  --agree-tos \
  --non-interactive \
  -d "$DOMAIN"

# Step 11: Update Nginx configuration for HTTPS
log_step "Enabling HTTPS configuration..."
# The configuration is already set up for HTTPS, just reload
sudo nginx -t
sudo systemctl reload nginx

# Step 12: Set up log rotation
log_step "Setting up log rotation..."
sudo cp /var/www/Perfect10k/config/logrotate/perfect10k /etc/logrotate.d/

# Step 13: Set up fail2ban
log_step "Setting up fail2ban..."
sudo cp /var/www/Perfect10k/config/fail2ban/perfect10k.conf /etc/fail2ban/jail.d/
sudo systemctl enable fail2ban
sudo systemctl restart fail2ban

# Step 14: Set up automatic certificate renewal
log_step "Setting up automatic certificate renewal..."
sudo cp /var/www/Perfect10k/scripts/ssl-renew.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/ssl-renew.sh
sudo sed -i "s/seitzquest.com/$DOMAIN/g" /usr/local/bin/ssl-renew.sh

# Add to crontab for automatic renewal
(sudo crontab -l 2>/dev/null; echo "0 3 * * 0 /usr/local/bin/ssl-renew.sh") | sudo crontab -

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

# Check certificate
CERT_EXPIRY=$(sudo openssl x509 -enddate -noout -in "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" | cut -d= -f2)
log_info "✓ SSL certificate expires: $CERT_EXPIRY"

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
echo "- Check SSL certificate: sudo certbot certificates"
echo "- Renew SSL certificate: sudo /usr/local/bin/ssl-renew.sh"
echo "- Check fail2ban status: sudo fail2ban-client status"
echo "- View Nginx logs: sudo tail -f /var/log/nginx/perfect10k_error.log"