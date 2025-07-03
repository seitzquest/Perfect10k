#!/usr/bin/env python3
"""
Perfect10k Launcher
Starts the backend server which serves both the API and frontend.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


# Colors for terminal output
class Colors:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_colored(text, color):
    print(f"{color}{text}{Colors.END}")


def print_banner():
    print_colored("=" * 60, Colors.BLUE)
    print_colored("🚀 Perfect10k Interactive Route Builder", Colors.BOLD)
    print_colored("   Full-stack server with integrated frontend", Colors.BLUE)
    print_colored("=" * 60, Colors.BLUE)
    print()


class ProcessManager:
    def __init__(self):
        self.processes = []
        self.running = True

    def add_process(self, process, name):
        self.processes.append((process, name))

    def cleanup(self):
        print_colored("\n🛑 Shutting down servers...", Colors.YELLOW)
        self.running = False

        for process, name in self.processes:
            if process.poll() is None:  # Process is still running
                print(f"  Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"  Force killing {name}...")
                    process.kill()

        print_colored("✅ All servers stopped", Colors.GREEN)




def start_backend():
    """Start the backend server."""
    print_colored("🔧 Starting backend server...", Colors.BLUE)

    backend_dir = Path("backend")
    if not backend_dir.exists():
        print_colored("❌ Backend directory not found", Colors.RED)
        return None

    # Change to backend directory
    os.chdir(backend_dir)

    # Start backend with HTTP
    cmd = [sys.executable, "main.py"]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=None,  # Don't capture stdout - let it print directly
            stderr=None,  # Don't capture stderr - let it print directly  
            universal_newlines=True,
            bufsize=0,   # No buffering
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}  # Force Python unbuffered output
        )
        print_colored("✅ Backend server starting on http://localhost:8000", Colors.GREEN)
        return process
    except OSError as e:
        if "Address already in use" in str(e):
            print_colored("⚠️ Port 8000 already in use, killing existing process...", Colors.YELLOW)
            try:
                # Find and kill process using port 8000
                kill_cmd = ["lsof", "-ti", "tcp:8000"]
                pid = subprocess.check_output(kill_cmd).strip()
                if pid:
                    subprocess.run(["kill", "-9", pid.decode()], check=True)
                    print_colored("✅ Killed process using port 8000", Colors.GREEN)
                    time.sleep(1)  # Give OS time to release the port
                    # Try starting the backend again
                    return start_backend()
            except Exception as kill_err:
                print_colored(f"❌ Failed to kill process: {kill_err}", Colors.RED)
        print_colored(f"❌ Failed to start backend: {e}", Colors.RED)
        return None
    except Exception as e:
        print_colored(f"❌ Failed to start backend: {e}", Colors.RED)
        return None




def monitor_process(process, name, manager):
    """Monitor a process and print its output."""
    try:
        for line in iter(process.stdout.readline, ""):
            if not manager.running:
                break
            if line.strip():
                print(f"[{name}] {line.strip()}")
    except Exception:
        pass


def wait_for_server():
    """Wait for server to start up."""
    print_colored("⏳ Waiting for server to start...", Colors.YELLOW)

    # Wait a bit for server to start
    time.sleep(3)

    # Test backend
    try:
        import requests

        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print_colored("✅ Backend is responding", Colors.GREEN)
        else:
            print_colored("⚠️  Backend health check failed", Colors.YELLOW)
    except Exception:
        print_colored("⚠️  Cannot connect to backend", Colors.YELLOW)


def main():
    print_banner()

    # Change to project root directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Process manager for cleanup
    manager = ProcessManager()

    def signal_handler(signum, frame):
        manager.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start backend
        backend_process = start_backend()
        if not backend_process:
            return 1
        manager.add_process(backend_process, "Backend")

        # No need for monitoring thread since output goes directly to console

        # Wait for server to be ready
        wait_for_server()

        print()
        print_colored("🎉 Perfect10k is running!", Colors.GREEN)
        print_colored("   Application: http://localhost:8000", Colors.BLUE)
        print_colored("   API Docs: http://localhost:8000/docs", Colors.BLUE)
        print()
        print_colored("Press Ctrl+C to stop the server", Colors.BLUE)

        # Keep running until interrupted
        while manager.running:
            time.sleep(1)

            # Check if process is still running
            if backend_process.poll() is not None:
                print_colored("❌ Backend process died", Colors.RED)
                break

    except KeyboardInterrupt:
        pass
    finally:
        manager.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
