#!/bin/bash

# Create SSL certificates for local HTTPS development
# This enables geolocation API access on mobile devices via IP address

echo "Creating SSL certificates for local HTTPS development..."

# Create certs directory
mkdir -p certs

# Generate private key
openssl genrsa -out certs/key.pem 2048

# Generate certificate signing request
openssl req -new -key certs/key.pem -out certs/csr.pem -subj "/C=US/ST=Local/L=Local/O=Perfect10k/CN=localhost"

# Generate self-signed certificate valid for 365 days
# Include SAN (Subject Alternative Name) for IP addresses
openssl x509 -req -days 365 -in certs/csr.pem -signkey certs/key.pem -out certs/cert.pem \
    -extensions v3_req -extfile <(cat <<EOF
[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
IP.2 = 192.168.1.100
IP.3 = 192.168.0.100
IP.4 = 10.0.0.100
EOF
)

# Clean up CSR file
rm certs/csr.pem

echo "SSL certificates created in certs/ directory"
echo "cert.pem - Certificate file"
echo "key.pem  - Private key file"
echo ""
echo "Note: You'll need to accept the security warning in your browser"
echo "since this is a self-signed certificate."