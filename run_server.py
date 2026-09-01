"""
Production Server Launcher (Multi-threaded WSGI via Waitress)
Recommended for Windows PC and Server environments.
"""
import sys
import logging
from app import app
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("server_launcher")

if __name__ == "__main__":
    host = "0.0.0.0"
    port = 5000
    threads = 8

    print("=" * 65)
    print("🚀 NavWiz & DFleet Multimodal RAG Server starting...")
    print(f"📍 Local Access:   http://localhost:{port}")
    print(f"🌐 Network Access: http://<Your_Server_IP>:{port}")
    print(f"⚙️  QA Model:       {config.GEMINI_QA_MODEL}")
    print(f"🧵 Worker Threads: {threads}")
    print("=" * 65)

    try:
        from waitress import serve
        logger.info(f"Serving with Waitress WSGI on http://{host}:{port} ({threads} worker threads)...")
        serve(app, host=host, port=port, threads=threads)
    except ImportError:
        logger.warning("Waitress not found. Falling back to Flask built-in server (pip install waitress for production).")
        app.run(host=host, port=port, debug=False)
