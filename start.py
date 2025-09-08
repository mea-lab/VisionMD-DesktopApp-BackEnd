import os
import time
from multiprocessing import freeze_support
from django.core.management import execute_from_command_line

"""Run administrative tasks."""
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")

def run_server():
    execute_from_command_line(["manage.py", "runserver", "--noreload"])

if __name__ == "__main__":
    freeze_support()
    run_server()
    
