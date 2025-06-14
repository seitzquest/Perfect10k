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
sudo apt install -y nginx git certbot python3-certbot-nginx python3-certbot-dns-cloudflare curl
```

### 1.3 Enable Nginx
```bash
sudo systemctl enable nginx
sudo systemctl start nginx
```

## Step 2: Domain and DNS Setup

### 2.1 Get Your Raspberry Pi's Public IP
```bash
curl ifconfig.me
```
Note this IP address.

### 2.2 Configure Cloudflare DNS
1. Log into your Cloudflare dashboard
2. Select your domain
3. Go to DNS > Records
4. Create an A record:
   - **Type**: A
   - **Name**: your-subdomain (e.g., `perfect10k` for `seitzquest.com`)
   - **IPv4 address**: Your Raspberry Pi's public IP
   - **Proxy status**: DNS only (gray cloud) - Important for Let's Encrypt
   - **TTL**: Auto

### 2.3 Configure Router Port Forwarding
Forward these ports to your Raspberry Pi's local IP:
- Port 80 (HTTP) → Raspberry Pi IP:80
- Port 443 (HTTPS) → Raspberry Pi IP:443

## Step 3: Cloudflare API Setup

### 3.1 Get Cloudflare API Token
1. Go to Cloudflare Dashboard → My Profile → API Tokens
2. Click "Create Token"
3. Use "Custom token" template
4. Set permissions:
   - Zone:Zone:Read
   - Zone:DNS:Edit
5. Include your domain in Zone Resources
6. Copy the token

### 3.2 Create Cloudflare Credentials File
```bash
sudo mkdir -p /etc/letsencrypt
sudo nano /etc/letsencrypt/cloudflare.ini
```

Add your token:
```ini
dns_cloudflare_api_token = YOUR_CLOUDFLARE_API_TOKEN_HERE
```

Set secure permissions:
```bash
sudo chmod 600 /etc/letsencrypt/cloudflare.ini
```

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
sudo nano /etc/nginx/sites-available/perfect10k
```

Copy the configuration from `config/nginx/perfect10k.conf` (see configuration files section below).

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

## Step 6: SSL Certificate Setup

### 6.1 Obtain SSL Certificate
Replace `seitzquest.com` with your actual domain:

```bash
sudo certbot certonly \
  --dns-cloudflare \
  --dns-cloudflare-credentials /etc/letsencrypt/cloudflare.ini \
  --dns-cloudflare-propagation-seconds 60 \
  -d seitzquest.com
```

### 6.2 Update Nginx Configuration for HTTPS
Edit the Nginx configuration to uncomment the HTTPS server block:
```bash
sudo nano /etc/nginx/sites-available/perfect10k
```

### 6.3 Test and Reload Nginx
```bash
sudo nginx -t
sudo systemctl reload nginx
```

## Step 7: Automatic Certificate Renewal

### 7.1 Test Renewal
```bash
sudo certbot renew --dry-run
```

### 7.2 Set Up Automatic Renewal
The renewal should work automatically via systemd timer. Verify:
```bash
sudo systemctl status certbot.timer
```

## Step 8: Security and Optimization

### 8.1 Configure Firewall
```bash
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
```

### 8.2 Set Up Log Rotation
```bash
sudo nano /etc/logrotate.d/perfect10k
```

Add the log rotation configuration from `config/logrotate/perfect10k` (see configuration files section).

## Step 9: Backend API Setup (Optional)

If you have a backend API, set up the Python environment:

### 9.1 Install Python Dependencies
```bash
sudo apt install -y python3-pip python3-venv
cd /var/www/Perfect10k
sudo -u www-data python3 -m venv venv
sudo -u www-data ./venv/bin/pip install -r requirements.txt
```

### 9.2 Create Systemd Service
```bash
sudo nano /etc/systemd/system/perfect10k-api.service
```

Copy the configuration from `config/systemd/perfect10k-api.service` (see configuration files section).

### 9.3 Enable and Start Service
```bash
sudo systemctl enable perfect10k-api
sudo systemctl start perfect10k-api
```

## Step 10: Monitoring and Maintenance

### 10.1 Check Service Status
```bash
sudo systemctl status nginx
sudo systemctl status perfect10k-api  # If using backend
sudo certbot certificates  # Check SSL certificates
```

### 10.2 View Logs
```bash
sudo tail -f /var/log/nginx/perfect10k_access.log
sudo tail -f /var/log/nginx/perfect10k_error.log
sudo journalctl -u perfect10k-api -f  # If using backend
```

### 10.3 Update Application
```bash
cd /var/www/Perfect10k
sudo git pull
sudo systemctl reload nginx
sudo systemctl restart perfect10k-api  # If using backend
```

## Troubleshooting

### Common Issues

1. **Certificate not working**: Ensure Cloudflare proxy is disabled (gray cloud)
2. **502 Bad Gateway**: Check backend API is running
3. **Access denied**: Check file permissions and ownership
4. **DNS not resolving**: Wait for DNS propagation (up to 24 hours)

### Useful Commands
```bash
# Check Nginx syntax
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx

# Check SSL certificate expiry
sudo certbot certificates

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