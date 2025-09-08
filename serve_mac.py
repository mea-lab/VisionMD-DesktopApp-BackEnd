import os
import sys
import logging
from multiprocessing import freeze_support
import mimetypes
import pathlib
import tzdata

zoneinfo_dir = pathlib.Path(tzdata.__file__).parent / "zoneinfo"
os.environ.setdefault("PYTHONTZPATH", str(zoneinfo_dir))
os.environ.setdefault("TZDIR", str(zoneinfo_dir))

mimetypes.init(files=[])
mimetypes.add_type('image/jpeg', '.jpg')
mimetypes.add_type('image/jpeg', '.jpeg')
mimetypes.add_type('image/png',  '.png')
mimetypes.add_type('image/webp', '.webp')
mimetypes.add_type('video/mp4',  '.mp4')

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
        logging.exception("Server failed to stsrt.")
        sys.exit(1)

if __name__ == "__main__":
    freeze_support()
    main()