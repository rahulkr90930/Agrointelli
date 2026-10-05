"""
AgroIntelli Launcher Script
Starts the Flask backend server and automatically opens the frontend in the default web browser.

Usage:
    python run.py
"""

import os
import sys
import time
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
    
    # Configure environment with UTF-8 encoding support
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    try:
        p = subprocess.Popen(cmd, env=env)
        p.wait()
    except KeyboardInterrupt:
        print("\n🌿 AgroIntelli stopped. Have a nice day!")
    except Exception as e:
        print(f"❌ Server error: {e}")

def open_frontend():
    print("⏳ Waiting for backend and TensorFlow to initialize...")
    import urllib.request
    url = "http://127.0.0.1:5000/health"
    for _ in range(40):  # Poll for up to 20 seconds
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AgroIntelli-Launcher"})
            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            pass
        time.sleep(0.5)
    
    # In GitHub Codespaces or headless environments
    if os.environ.get("CODESPACES") == "true":
        print("🌐 AgroIntelli is running inside GitHub Codespaces! Open port 5000 to view the app.")
        return

    frontend_path = os.path.abspath(os.path.join("frontend", "index.html"))
    if os.path.exists(frontend_path):
        browser_url = f"file:///{frontend_path.replace(os.sep, '/')}"
        print(f"🌐 Backend is live! Opening frontend in your browser: {browser_url}")
        try:
            webbrowser.open(browser_url)
        except Exception:
            pass
    else:
        print(f"⚠ Warning: Could not find frontend at {frontend_path}")

if __name__ == "__main__":
    try:
        # Start a background daemon thread to trigger opening the frontend
        opener_thread = Thread(target=open_frontend, daemon=True)
        opener_thread.start()
        
        # Start backend blocking on the main thread
        start_backend()
    except KeyboardInterrupt:
        print("\n🌿 AgroIntelli stopped. Have a nice day!")
