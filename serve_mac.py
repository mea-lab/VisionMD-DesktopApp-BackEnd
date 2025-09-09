import os
import sys
import io
import logging
from multiprocessing import freeze_support
import pathlib
import mimetypes

os.environ["PYTHONTZPATH"] = "/usr/share/zoneinfo"

log_dir = pathlib.Path.home() / "Library" / "Application Support" / "VisionMD" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
LOG_FILE = str(log_dir / "django.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8")]
)

print("Logging set up for MAS sandbox:")
print("All messages now logged to file:", LOG_FILE)

sys.stdout = open(LOG_FILE, "a", buffering=1)
sys.stderr = open(LOG_FILE, "a", buffering=1)

print("Starting backend...", LOG_FILE)

def main():
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'VideoAnalysisToolBackend.settings')
    MIMEFILE = pathlib.Path(__file__).resolve().parent / "mime.types"
    mimetypes.init(files=[str(MIMEFILE)])

    try:
        from waitress import serve
        from VideoAnalysisToolBackend.wsgi import application

        logging.info("WSGI application loaded.")
        logging.info("Starting Waitress on http://127.0.0.1:8000")
        serve(application, host='127.0.0.1', port=8000, threads=4)
    except Exception:
        logging.exception("Server failed to stsrt.")
        sys.exit(1)

if __name__ == "__main__":
    freeze_support()
    main()