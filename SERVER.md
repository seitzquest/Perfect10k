# Perfect10k Server Setup Guide

This guide will help you deploy the Perfect10k application on a Raspberry Pi using Nginx with HTTPS and Cloudflare DNS/SSL integration.

## Prerequisites

- Raspberry Pi with Raspberry Pi OS installed
- Domain name managed by Cloudflare
- Cloudflare account with API access
- Internet connection with port forwarding capability

## Step 1: Initial Server Setup

### 1.1 Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### 1.2 Install Required Packages
```bash
sudo apt install -y nginx git curl htop iotop fail2ban ufw python3-pip python3-venv

# Install geospatial system dependencies (required for OSM routing)
# Add newer repository for updated geospatial libraries
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
sudo apt update

# Install updated geospatial libraries
sudo apt install -y libproj-dev proj-data proj-bin libgeos-dev libspatialindex-dev libffi-dev libgdal-dev gdal-bin

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1.3 Enable Nginx
```bash
sudo systemctl enable nginx
sudo systemctl start nginx
```

## Step 2: Domain and DNS Setup

### 2.1 Get Your Raspberry Pi's Public IPv4 Address

Try these commands in order until you get an IPv4 address:

```bash
curl -4 ifconfig.me
```

Note this IP address for the next step.

### 2.2 Configure Cloudflare DNS
1. Log into your Cloudflare dashboard
2. Select your domain
3. Go to DNS > Records
4. Create an A record:
   - **Type**: A
   - **Name**: your-subdomain (e.g., `perfect10k` for `seitzquest.com`)
   - **IPv4 address**: Your Raspberry Pi's public IP
   - **Proxy status**: Proxied (orange cloud) - This enables Cloudflare SSL
   - **TTL**: Auto

### 2.3 Configure Router Port Forwarding

Forward these ports to your Raspberry Pi's local IP:
- Port 80 (HTTP) → Raspberry Pi IP:80
- Port 443 (HTTPS) → Raspberry Pi IP:443

## Step 3: Configure Cloudflare SSL

### 3.1 Enable Cloudflare SSL
1. In your Cloudflare dashboard, go to SSL/TLS → Overview
2. Set SSL/TLS encryption mode to **"Flexible"** or **"Full"**
   - **Flexible**: Cloudflare to visitors is encrypted, Cloudflare to server is HTTP
   - **Full**: Both connections encrypted (recommended if you set up HTTPS on your server)
3. Go to SSL/TLS → Edge Certificates
4. Enable **"Always Use HTTPS"** to redirect HTTP to HTTPS automatically

## Step 4: Deploy Application

### 4.1 Clone Repository
```bash
cd /var/www/
sudo git clone https://github.com/yourusername/Perfect10k.git
sudo chown -R www-data:www-data Perfect10k
sudo chmod -R 755 Perfect10k
```

### 4.2 Set Up Directory Structure
```bash
sudo mkdir -p /var/www/Perfect10k/logs
sudo chown -R www-data:www-data /var/www/Perfect10k/logs
```

## Step 5: Nginx Configuration

### 5.1 Create Nginx Site Configuration
```bash
sudo cp /var/www/Perfect10k/config/nginx/perfect10k-cloudflare.conf /etc/nginx/sites-available/perfect10k
sudo sed -i "s/seitzquest.com/your-actual-domain.com/g" /etc/nginx/sites-available/perfect10k
```

### 5.2 Enable Site
```bash
sudo ln -s /etc/nginx/sites-available/perfect10k /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default  # Remove default site
```

### 5.3 Test Nginx Configuration
```bash
sudo nginx -t
```

### 5.4 Reload Nginx
```bash
sudo systemctl reload nginx
```

## Step 6: Test Your Setup

Your site should now be accessible at both:
- `http://seitzquest.com` (will redirect to HTTPS via Cloudflare)
- `https://seitzquest.com` (secured by Cloudflare SSL)

### 6.1 Test HTTP Connection
```bash
curl -I http://seitzquest.com
```

### 6.2 Test HTTPS Connection
```bash
curl -I https://seitzquest.com
```

## Step 7: Set Up Python Backend

### 7.1 Install Python Dependencies
```bash
# Change ownership temporarily to install dependencies
sudo chown -R pi:pi /var/www/Perfect10k
cd /var/www/Perfect10k

# Install dependencies with uv
uv sync

# Change ownership back to www-data for production
sudo chown -R www-data:www-data /var/www/Perfect10k
```

### 7.2 Create Systemd Service for Backend
```bash
sudo cp /var/www/Perfect10k/config/systemd/perfect10k-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable perfect10k-api
sudo systemctl start perfect10k-api
```

### 7.3 Check Backend Status
```bash
sudo systemctl status perfect10k-api
```

The backend should now be running on `http://localhost:8000` and accessible through Nginx at your domain.

## Step 8: Security and Optimization

### 8.1 Configure Firewall
```bash
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
```

### 8.2 Set Up Log Rotation
```bash
sudo cp /var/www/Perfect10k/config/logrotate/perfect10k /etc/logrotate.d/
```

## Step 9: Monitoring and Maintenance

### 9.1 Check Service Status
```bash
sudo systemctl status nginx
sudo systemctl status perfect10k-api
```

### 9.2 View Logs
```bash
# Nginx logs
sudo tail -f /var/log/nginx/perfect10k_access.log
sudo tail -f /var/log/nginx/perfect10k_error.log

# Backend logs
sudo journalctl -u perfect10k-api -f
```

### 9.3 Test API Connection
```bash
# Test backend directly
curl http://localhost:8000/health

# Test through Nginx
curl https://seitzquest.com/api/health
```

### 9.4 Update Application
```bash
cd /var/www/Perfect10k
sudo git pull
sudo systemctl restart perfect10k-api
sudo systemctl reload nginx
```

## Troubleshooting

### Common Issues

1. **Only getting IPv6 address**:
   - Try `curl -4 icanhazip.com` or other IPv4 services
   - Check if your ISP provides IPv4 connectivity
   - Contact ISP or use a VPN for IPv4 access
   - Check router settings for IPv4/dual-stack mode

2. **SSL not working**: Ensure Cloudflare proxy is enabled (orange cloud) and SSL mode is set to "Flexible" or "Full"

3. **502 Bad Gateway**: Check backend API is running

4. **Access denied**: Check file permissions and ownership

5. **DNS not resolving**: Wait for DNS propagation (up to 24 hours)

### Useful Commands
```bash
# Check Nginx syntax
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx

# Test HTTPS connection
curl -I https://seitzquest.com

# Check if ports are open
sudo netstat -tlnp | grep :443
```

## Security Considerations

1. **Keep system updated**: Regular `apt update && apt upgrade`
2. **Monitor logs**: Regular log review for suspicious activity
3. **Backup certificates**: Consider backing up `/etc/letsencrypt/`
4. **Use fail2ban**: Consider installing fail2ban for additional security
5. **Regular updates**: Keep the application code updated

## Performance Optimization

1. **Enable gzip compression** (included in config)
2. **Set up caching headers** (included in config)
3. **Monitor resource usage**: Use `htop` and `iotop`
4. **Consider CDN**: Use Cloudflare's CDN features

Your Perfect10k application should now be accessible at `https://seitzquest.com` with automatic HTTP to HTTPS redirection!