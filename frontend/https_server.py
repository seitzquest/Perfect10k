#!/usr/bin/env python3
"""
Simple HTTPS static file server for Perfect10k frontend
Enables geolocation API access on mobile devices
"""

import http.server
import os
import ssl
import sys
from pathlib import Path


def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    cert_dir = project_root / "certs"

    # Check if certificates exist
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    if not cert_file.exists() or not key_file.exists():
        print("SSL certificates not found!")
        print("Run the following command to create them:")
        print(f"cd {project_root} && ./create_ssl_certs.sh")
        sys.exit(1)

    # Change to frontend directory
    os.chdir(Path(__file__).parent)

    # Create HTTPS server
    server_address = ('0.0.0.0', 3000)
    httpd = http.server.HTTPServer(server_address, http.server.SimpleHTTPRequestHandler)

    # Create SSL context
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(str(cert_file), str(key_file))

    # Wrap the socket with SSL
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    print("Starting Perfect10k HTTPS frontend server...")
    print("This enables geolocation API access on mobile devices")
    print("")
    print("Frontend available at:")
    print("  https://localhost:3000")
    print("  https://YOUR_LOCAL_IP:3000")
    print("")
    print("Make sure the backend is running at:")
    print("  https://localhost:8000 (or YOUR_LOCAL_IP:8000)")
    print("")
    print("Note: You'll need to accept the security warning for self-signed certificates")
    print("")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()

if __name__ == "__main__":
    main()
