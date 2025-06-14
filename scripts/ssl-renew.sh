#!/bin/bash

# Perfect10k SSL Certificate Renewal Script
# This script handles SSL certificate renewal and Nginx reload

set -e

# Configuration
DOMAIN="seitzquest.com"  # Change this to your domain
LOG_FILE="/var/log/perfect10k-ssl-renewal.log"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | sudo tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    log "[SUCCESS] $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    log "[ERROR] $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   log_error "This script must be run as root"
   exit 1
fi

# Create log file if it doesn't exist
touch "$LOG_FILE"
chmod 644 "$LOG_FILE"

log "Starting SSL certificate renewal process..."

# Renew certificates
if certbot renew --quiet --no-self-upgrade; then
    log_success "Certificate renewal completed"
    
    # Test Nginx configuration
    if nginx -t; then
        log_success "Nginx configuration test passed"
        
        # Reload Nginx to apply new certificates
        if systemctl reload nginx; then
            log_success "Nginx reloaded successfully"
        else
            log_error "Failed to reload Nginx"
            exit 1
        fi
    else
        log_error "Nginx configuration test failed"
        exit 1
    fi
else
    log_error "Certificate renewal failed"
    exit 1
fi

# Check certificate expiry
CERT_PATH="/etc/letsencrypt/live/$DOMAIN/fullchain.pem"
if [ -f "$CERT_PATH" ]; then
    EXPIRY_DATE=$(openssl x509 -enddate -noout -in "$CERT_PATH" | cut -d= -f2)
    EXPIRY_EPOCH=$(date -d "$EXPIRY_DATE" +%s)
    CURRENT_EPOCH=$(date +%s)
    DAYS_UNTIL_EXPIRY=$(( (EXPIRY_EPOCH - CURRENT_EPOCH) / 86400 ))
    
    log_success "Certificate expires in $DAYS_UNTIL_EXPIRY days ($EXPIRY_DATE)"
    
    # Send alert if certificate expires soon (less than 7 days)
    if [ $DAYS_UNTIL_EXPIRY -lt 7 ]; then
        log_error "WARNING: Certificate expires in less than 7 days!"
    fi
else
    log_error "Certificate file not found: $CERT_PATH"
fi

log "SSL renewal process completed"