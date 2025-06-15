#!/usr/bin/env python3
"""
Simple script to run HTTP backend with uv
SSL is handled by Cloudflare in production
"""

import subprocess
import sys


def main():
    print("Starting Perfect10k HTTP backend...")
    print("Access at: http://localhost:8000")
    print("SSL handled by Cloudflare in production")
    print("")

    # Run with uv
    cmd = [
        "uv", "run", "uvicorn", "main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]

    subprocess.run(cmd)


if __name__ == "__main__":
    main()