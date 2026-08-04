import os
import logging
from celery import Celery
from app.core.config import settings

logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery(
    "recruitment_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

# Optional configuration overrides
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=1800,  # 30 minutes max time limit
)

# Auto-discover tasks in the tasks package
celery_app.autodiscover_tasks(["app.tasks"], related_name="jobs")
