#!/usr/bin/env python3
"""
Simple HTTP static file server for Perfect10k frontend
SSL is handled by Cloudflare in production
"""

import http.server
import os
from pathlib import Path


def main():
    # Change to frontend directory
    os.chdir(Path(__file__).parent)

    # Create HTTP server
    server_address = ('0.0.0.0', 3000)
    httpd = http.server.HTTPServer(server_address, http.server.SimpleHTTPRequestHandler)

    print("Starting Perfect10k HTTP frontend server...")
    print("SSL handled by Cloudflare in production")
    print("")
    print("Frontend available at:")
    print("  http://localhost:3000")
    print("  http://YOUR_LOCAL_IP:3000")
    print("")
    print("Make sure the backend is running at:")
    print("  http://localhost:8000 (or YOUR_LOCAL_IP:8000)")
    print("")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()

if __name__ == "__main__":
    main()