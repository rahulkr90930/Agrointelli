"""
AgroIntelli Launcher Script
Starts the Flask backend server and automatically opens the frontend in the default web browser.

Usage:
    python run.py
"""

import os
import sys
import time
import socket
import subprocess
import webbrowser
from threading import Thread

# Reconfigure stdout/stderr to UTF-8 on Windows to prevent UnicodeEncodeError with emojis
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def start_backend():
    print("🌿 Starting AgroIntelli Backend Server...")
    
    # Resolve the correct Python interpreter (Windows or Linux/Codespaces)
    candidates = [
        os.path.join("venv", "Scripts", "python.exe"),
        os.path.join(".venv", "Scripts", "python.exe"),
        os.path.join("venv", "bin", "python"),
        os.path.join(".venv", "bin", "python"),
        sys.executable
    ]
    venv_python = sys.executable
    for candidate in candidates:
        if os.path.exists(candidate):
            venv_python = candidate
            break
        
    print(f"🚀 Using Python: {venv_python}")
    
    backend_script = os.path.join("backend", "app.py")
    if not os.path.exists(backend_script):
        print(f"❌ Error: Could not find backend script at {backend_script}")
        return

    cmd = [venv_python, backend_script]
    
    # Configure environment with UTF-8 encoding support & quiet TF initialization
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    
    try:
        p = subprocess.Popen(cmd, env=env)
        p.wait()
    except KeyboardInterrupt:
        print("\n🌿 AgroIntelli stopped. Have a nice day!")
    except Exception as e:
        print(f"❌ Server error: {e}")

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def open_frontend(port):
    print("⏳ Waiting for backend and TensorFlow to initialize...")
    import urllib.request
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(40):  # Poll for up to 20 seconds
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AgroIntelli-Launcher"})
            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            pass
        time.sleep(0.5)
    
    app_url = f"http://127.0.0.1:{port}/"

    # Detect cloud headless environments (Codespaces, Replit, Binder, Gitpod, Docker)
    is_headless_cloud = any(
        os.environ.get(k) for k in [
            "CODESPACES", "REPLIT_ENVIRONMENT", "REPL_ID", "BINDER_PORT", 
            "JUPYTER_SERVER_URL", "GITPOD_WORKSPACE_ID"
        ]
    ) or not (sys.platform.startswith("win") or os.environ.get("DISPLAY"))

    if is_headless_cloud:
        print(f"🌐 AgroIntelli is live! Open port 5000 in your cloud platform to view the web app: {app_url}")
        return

    print(f"🌐 Backend is live! Opening AgroIntelli web app: {app_url}")
    try:
        webbrowser.open(app_url)
    except Exception:
        frontend_path = os.path.abspath(os.path.join("frontend", "index.html"))
        if os.path.exists(frontend_path):
            try:
                webbrowser.open(f"file:///{frontend_path.replace(os.sep, '/')}")
            except Exception:
                pass

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", "5000"))
        if is_port_in_use(port):
            print(
                f"❌ Port {port} is already in use. Stop the existing AgroIntelli "
                "server before restarting; no browser was opened."
            )
            sys.exit(1)

        # Start a background daemon thread to trigger opening the frontend
        opener_thread = Thread(target=open_frontend, args=(port,), daemon=True)
        opener_thread.start()
        
        # Start backend blocking on the main thread
        start_backend()
    except KeyboardInterrupt:
        print("\n🌿 AgroIntelli stopped. Have a nice day!")
