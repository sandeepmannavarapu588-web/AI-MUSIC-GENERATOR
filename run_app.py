"""
Root launcher for AI Music Generator Studio
Automatically launches backend/app.py and opens the browser.
"""

import os
import sys
import time
import webbrowser

backend_dir = os.path.join(os.path.dirname(__file__), "backend")
sys.path.append(backend_dir)

from app import app, model_mgr, audio_converter, BASE_DIR
import socket

def find_available_port(start_port=5001):
    for p in [start_port, 5002, 5050, 8080]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', p)) != 0:
                return p
    return 5001

if __name__ == "__main__":
    port = find_available_port(5001)
    url = f"http://127.0.0.1:{port}"

    print("=" * 65)
    print("  🎵  AI Music Generator Using Deep Learning (LSTM)")
    print("=" * 65)
    print(f"  Deep Learning Engine : {model_mgr.engine_type}")
    print(f"  FluidSynth Available : {audio_converter.has_fluidsynth}")
    print("-" * 65)
    print(f"  👉 Opening Studio at: {url}")
    print("=" * 65)

    # Open browser automatically after a short delay
    def open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    app.run(host="0.0.0.0", port=port, debug=False)
