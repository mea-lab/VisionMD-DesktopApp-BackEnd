import os
import sys
import logging
from multiprocessing import freeze_support

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'VideoAnalysisToolBackend.settings')

    try:
        from waitress import serve
        from VideoAnalysisToolBackend.wsgi import application

        logging.info("WSGI application loaded.")
        logging.info("Starting Waitress on http://127.0.0.1:8000")
        serve(application, host='127.0.0.1', port=8000, threads=4)
    except Exception:
        logging.exception("Server failed to start.")
        sys.exit(1)

if __name__ == "__main__":
    freeze_support()
    main()