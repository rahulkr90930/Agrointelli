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
    
    # Resolve the correct Python interpreter
    # Look for virtual environment first, fallback to standard python executable
    venv_python = os.path.join("venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        venv_python = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        venv_python = sys.executable
        
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
    # Wait for the backend and TensorFlow to initialize (typically ~3.5 seconds)
    print("⏳ Waiting for backend to initialize before opening frontend...")
    time.sleep(3.5)
    
    frontend_path = os.path.abspath(os.path.join("frontend", "index.html"))
    if os.path.exists(frontend_path):
        url = f"file:///{frontend_path.replace(os.sep, '/')}"
        print(f"🌐 Opening frontend in your web browser: {url}")
        webbrowser.open(url)
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
