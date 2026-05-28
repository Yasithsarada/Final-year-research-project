from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime

class TaskStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    filename: str
    resume_id: Optional[str] = None
    error: Optional[str] = None

class AsyncUploadResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str
    poll_url: str
