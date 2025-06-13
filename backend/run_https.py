#!/usr/bin/env python3
"""
Simple script to run HTTPS backend with uv
"""

import subprocess
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
        print("Run: ./create_ssl_certs.sh")
        sys.exit(1)

    print("Starting Perfect10k HTTPS backend...")
    print("Access at: https://localhost:8000 or https://YOUR_IP:8000")
    print("")

    # Run with uv
    cmd = [
        "uv", "run", "uvicorn", "main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--ssl-keyfile", str(key_file),
        "--ssl-certfile", str(cert_file),
        "--reload"
    ]

    subprocess.run(cmd)

if __name__ == "__main__":
    main()
