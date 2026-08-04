@echo off
echo Starting Celery Worker for AI-Powered Recruitment System...
cd ..
celery -A app.tasks.worker.celery_app worker --loglevel=info -P threads
