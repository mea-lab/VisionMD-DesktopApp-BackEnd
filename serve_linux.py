import os
import sys
import gunicorn
import gunicorn.app.wsgiapp
import gunicorn.glogging
import gunicorn.workers.gthread
from gunicorn.app.wsgiapp import run
import django
import VideoAnalysisToolBackend.wsgi

sys.path.append(os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'VideoAnalysisToolBackend.settings')

sys.argv = [
    'gunicorn',
    '--workers', '1',
    '--threads', '4',
    '--worker-class', 'gthread',
    '--bind', '127.0.0.1:8000',
    'VideoAnalysisToolBackend.wsgi:application'
]

run()
