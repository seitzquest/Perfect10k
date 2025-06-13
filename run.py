#!/usr/bin/env python3
"""
Perfect10k Unified Launcher
Starts both backend and frontend servers with a single command.
"""

import os
import sys
import time
import signal
import subprocess
import threading
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
    print_colored("   User-driven step-by-step route construction", Colors.BLUE)
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


def check_ssl_certificates():
    """Check if SSL certificates exist, create them if not."""
    cert_dir = Path("certs")
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    if cert_file.exists() and key_file.exists():
        print_colored("✅ SSL certificates found", Colors.GREEN)
        return True

    print_colored("⚠️  SSL certificates not found, creating them...", Colors.YELLOW)

    # Create certificates directory
    cert_dir.mkdir(exist_ok=True)

    # Generate self-signed certificate
    openssl_cmd = [
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:4096",
        "-keyout",
        str(key_file),
        "-out",
        str(cert_file),
        "-days",
        "365",
        "-nodes",
        "-subj",
        "/C=US/ST=CA/L=SF/O=Perfect10k/CN=localhost",
    ]

    try:
        subprocess.run(openssl_cmd, check=True, capture_output=True)
        print_colored("✅ SSL certificates created", Colors.GREEN)
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Failed to create SSL certificates: {e}", Colors.RED)
        print_colored("   Please install openssl or create certificates manually", Colors.YELLOW)
        return False


def start_backend():
    """Start the backend server."""
    print_colored("🔧 Starting backend server...", Colors.BLUE)

    backend_dir = Path("backend")
    if not backend_dir.exists():
        print_colored("❌ Backend directory not found", Colors.RED)
        return None

    # Change to backend directory
    os.chdir(backend_dir)

    # Start backend with HTTPS
    cmd = [sys.executable, "run_https.py"]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        print_colored("✅ Backend server starting on https://localhost:8000", Colors.GREEN)
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


def start_frontend():
    """Start the frontend server."""
    print_colored("🌐 Starting frontend server...", Colors.BLUE)

    frontend_dir = Path("../frontend")
    if not frontend_dir.exists():
        print_colored("❌ Frontend directory not found", Colors.RED)
        return None

    # Start frontend HTTPS server
    cmd = [sys.executable, "https_server.py"]

    try:
        process = subprocess.Popen(
            cmd,
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        print_colored("✅ Frontend server starting on https://localhost:3000", Colors.GREEN)
        return process
    except Exception as e:
        print_colored(f"❌ Failed to start frontend: {e}", Colors.RED)
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


def wait_for_servers():
    """Wait for servers to start up."""
    print_colored("⏳ Waiting for servers to start...", Colors.YELLOW)

    # Wait a bit for servers to start
    time.sleep(3)

    # Test backend
    try:
        import requests

        requests.packages.urllib3.disable_warnings()
        response = requests.get("https://localhost:8000/health", verify=False, timeout=5)
        if response.status_code == 200:
            print_colored("✅ Backend is responding", Colors.GREEN)
        else:
            print_colored("⚠️  Backend health check failed", Colors.YELLOW)
    except Exception:
        print_colored("⚠️  Cannot connect to backend", Colors.YELLOW)

    # Test frontend
    try:
        response = requests.get("https://localhost:3000", verify=False, timeout=5)
        if response.status_code == 200:
            print_colored("✅ Frontend is responding", Colors.GREEN)
        else:
            print_colored("⚠️  Frontend health check failed", Colors.YELLOW)
    except Exception:
        print_colored("⚠️  Cannot connect to frontend", Colors.YELLOW)


def main():
    print_banner()

    # Change to project root directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Check SSL certificates
    if not check_ssl_certificates():
        print_colored("❌ SSL setup failed. Exiting.", Colors.RED)
        return 1

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

        # Start frontend
        frontend_process = start_frontend()
        if not frontend_process:
            manager.cleanup()
            return 1
        manager.add_process(frontend_process, "Frontend")

        # Start monitoring threads
        backend_thread = threading.Thread(
            target=monitor_process, args=(backend_process, "Backend", manager)
        )
        frontend_thread = threading.Thread(
            target=monitor_process, args=(frontend_process, "Frontend", manager)
        )

        backend_thread.daemon = True
        frontend_thread.daemon = True

        backend_thread.start()
        frontend_thread.start()

        # Wait for servers to be ready
        wait_for_servers()

        print()
        print_colored("🎉 Perfect10k is running!", Colors.GREEN)
        print_colored("   API Docs: https://localhost:8000/docs", Colors.BLUE)
        print()
        print_colored("💡 Open https://localhost:8000 in your browser", Colors.YELLOW)
        print_colored("   (Accept the self-signed certificate warning)", Colors.YELLOW)
        print()
        print_colored("Press Ctrl+C to stop all servers", Colors.BLUE)

        # Keep running until interrupted
        while manager.running:
            time.sleep(1)

            # Check if processes are still running
            if backend_process.poll() is not None:
                print_colored("❌ Backend process died", Colors.RED)
                break
            if frontend_process.poll() is not None:
                print_colored("❌ Frontend process died", Colors.RED)
                break

    except KeyboardInterrupt:
        pass
    finally:
        manager.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
