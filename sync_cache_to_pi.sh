#!/bin/bash
"""
Sync Cache Data to Raspberry Pi
===============================

This script copies cache and storage data from your fast computer to the Pi
to avoid the 5+ minute generation time on first load.
"""

set -e  # Exit on any error

PI_USER_HOST=192.168.178.87
PI_PROJECT_PATH=/var/www/Perfect10k

if [ -z "$PI_USER_HOST" ] || [ -z "$PI_PROJECT_PATH" ]; then
    echo "❌ Usage: $0 <pi_user@pi_host> <pi_project_path>"
    echo ""
    echo "Examples:"
    echo "  $0 pi@192.168.1.100 /home/pi/Perfect10k"
    echo "  $0 pi@raspberry.local /var/www/Perfect10k"
    exit 1
fi

echo "🚀 Syncing cache data to Raspberry Pi..."
echo "📍 Target: $PI_USER_HOST:$PI_PROJECT_PATH"

# Check if local cache directories exist
if [ ! -d "backend/cache" ]; then
    echo "❌ Error: backend/cache directory not found locally"
    echo "   Make sure you're running this from the Perfect10k project root"
    exit 1
fi

if [ ! -d "backend/storage" ]; then
    echo "❌ Error: backend/storage directory not found locally"
    echo "   Make sure you're running this from the Perfect10k project root"
    exit 1
fi

# Calculate sizes
CACHE_SIZE=$(du -sh backend/cache | cut -f1)
STORAGE_SIZE=$(du -sh backend/storage | cut -f1)

echo "📊 Local cache sizes:"
echo "   📁 backend/cache: $CACHE_SIZE"
echo "   📁 backend/storage: $STORAGE_SIZE"

# Confirm with user
echo ""
read -p "🤔 Continue with sync? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Sync cancelled"
    exit 1
fi

echo ""
echo "⏳ Starting sync (this may take a few minutes)..."

# Create temp directories in home directory first
echo "📁 Creating temp directories on Pi..."
ssh "$PI_USER_HOST" "mkdir -p ~/perfect10k_cache_temp/cache ~/perfect10k_cache_temp/storage"

# Copy to home directory first (no permission issues)
echo "🔄 Copying cache directory to home..."
scp -r backend/cache/* "$PI_USER_HOST:~/perfect10k_cache_temp/cache/"

echo "🔄 Copying storage directory to home..."
scp -r backend/storage/* "$PI_USER_HOST:~/perfect10k_cache_temp/storage/"

# Now move to final location with sudo
echo "🔧 Moving files to final location with sudo..."
ssh "$PI_USER_HOST" "
    sudo mkdir -p $PI_PROJECT_PATH/backend/cache/smart_cache $PI_PROJECT_PATH/backend/cache/overlays $PI_PROJECT_PATH/backend/storage/tiles &&
    sudo cp -r ~/perfect10k_cache_temp/cache/* $PI_PROJECT_PATH/backend/cache/ &&
    sudo cp -r ~/perfect10k_cache_temp/storage/* $PI_PROJECT_PATH/backend/storage/ &&
    sudo chown -R \$(id -u docker):\$(id -g docker) $PI_PROJECT_PATH/backend/cache $PI_PROJECT_PATH/backend/storage 2>/dev/null || 
    sudo chown -R 1000:1000 $PI_PROJECT_PATH/backend/cache $PI_PROJECT_PATH/backend/storage &&
    rm -rf ~/perfect10k_cache_temp
"

echo ""
echo "✅ Sync completed successfully!"
echo ""
echo "🔄 Now restart your Docker container on the Pi:"
echo "   ssh $PI_USER_HOST"
echo "   cd $PI_PROJECT_PATH"
echo "   docker-compose down && docker-compose up -d"
echo ""
echo "⚡ Route generation should now be fast (< 1 second) for cached areas!"