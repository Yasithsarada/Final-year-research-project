import uuid
import datetime
import logging
import json
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Query, Request

from app.core.config import settings
from app.db import db_client
from app.models.resume import ResumeDocument
from app.models.status import TaskResult, TaskStatus, AsyncUploadResponse
from app.services.pipeline import ResumeParsingPipeline
from app.core.exceptions import ParserException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/parser", tags=["Resume Parser"])
pipeline = ResumeParsingPipeline()

# Simple task tracker dictionary (in-memory caching combined with file-system/db caching)
# We will write tasks to disk or Mongo to ensure persistence
async def save_task_state(task: TaskResult):
    try:
        if settings.DB_TYPE.lower() == "mongodb" and db_client.db is not None:
            doc = task.model_dump()
            doc["created_at"] = doc["created_at"].isoformat()
            doc["updated_at"] = doc["updated_at"].isoformat()
            await db_client.db["tasks"].update_one(
                {"task_id": task.task_id},
                {"$set": doc},
                upsert=True
            )
        else:
            task_dir = settings.LOCAL_STORAGE_DIR / "tasks"
            task_dir.mkdir(parents=True, exist_ok=True)
            file_path = task_dir / f"{task.task_id}.json"
            
            doc = task.model_dump()
            doc["created_at"] = doc["created_at"].isoformat()
            doc["updated_at"] = doc["updated_at"].isoformat()
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save task state: {str(e)}")

async def get_task_state(task_id: str) -> Optional[TaskResult]:
    try:
        if settings.DB_TYPE.lower() == "mongodb" and db_client.db is not None:
            doc = await db_client.db["tasks"].find_one({"task_id": task_id})
            if doc:
                doc["created_at"] = datetime.datetime.fromisoformat(doc["created_at"])
                doc["updated_at"] = datetime.datetime.fromisoformat(doc["updated_at"])
                return TaskResult.model_validate(doc)
        else:
            file_path = settings.LOCAL_STORAGE_DIR / "tasks" / f"{task_id}.json"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                doc["created_at"] = datetime.datetime.fromisoformat(doc["created_at"])
                doc["updated_at"] = datetime.datetime.fromisoformat(doc["updated_at"])
                return TaskResult.model_validate(doc)
    except Exception as e:
        logger.error(f"Failed to retrieve task state: {str(e)}")
    return None


async def run_async_pipeline(task_id: str, file_bytes: bytes, filename: str):
    """Worker job running inside FastAPI BackgroundTasks."""
    task = await get_task_state(task_id)
    if not task:
        return
        
    task.status = TaskStatus.PROCESSING
    task.updated_at = datetime.datetime.utcnow()
    await save_task_state(task)
    
    try:
        # Run parsing pipeline
        resume_doc = await pipeline.process_resume(file_bytes, filename)
        
        # Complete task
        task.status = TaskStatus.COMPLETED
        task.resume_id = resume_doc.id
        task.updated_at = datetime.datetime.utcnow()
        await save_task_state(task)
        logger.info(f"Background task {task_id} completed. Resume stored with ID {resume_doc.id}")
        
    except Exception as e:
        # Fail task
        task.status = TaskStatus.FAILED
        task.error = str(e)
        task.updated_at = datetime.datetime.utcnow()
        await save_task_state(task)
        logger.error(f"Background task {task_id} failed: {str(e)}")


# ==========================================
# API Route Handlers
# ==========================================

@router.post("/upload", response_model=ResumeDocument, summary="Upload & Parse Resume (Synchronous)")
async def upload_resume_sync(file: UploadFile = File(...)):
    """Uploads and parses a resume PDF or DOCX file synchronously.
    
    Returns the full structured resume metadata instantly.
    """
    # Read file content
    contents = await file.read()
    if not contents:
         raise HTTPException(status_code=400, detail="Uploaded file is empty.")
         
    try:
        resume_doc = await pipeline.process_resume(contents, file.filename)
        return resume_doc
    except ParserException as pe:
        raise HTTPException(status_code=422, detail=pe.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@router.post("/upload-async", response_model=AsyncUploadResponse, summary="Upload & Parse Resume (Asynchronous)")
async def upload_resume_async(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Uploads a resume file and schedules parsing in a background execution queue.
    
    Returns a ticket (task ID) and polling URL immediately.
    """
    contents = await file.read()
    if not contents:
         raise HTTPException(status_code=400, detail="Uploaded file is empty.")
         
    task_id = str(uuid.uuid4())
    now = datetime.datetime.utcnow()
    
    # Initialize background task state
    task = TaskResult(
        task_id=task_id,
        status=TaskStatus.PENDING,
        created_at=now,
        updated_at=now,
        filename=file.filename
    )
    await save_task_state(task)
    
    # Enqueue task using FastAPI BackgroundTasks
    background_tasks.add_task(run_async_pipeline, task_id, contents, file.filename)
    
    # Construct polling URL
    base_url = str(request.base_url).rstrip("/")
    poll_url = f"{base_url}/api/v1/parser/tasks/{task_id}"
    
    return AsyncUploadResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="Resume parsing queued in the background.",
        poll_url=poll_url
    )


@router.get("/tasks/{task_id}", response_model=TaskResult, summary="Check Async Task Status")
async def get_task_status(task_id: str):
    """Checks the status of an asynchronous parsing ticket."""
    task = await get_task_state(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task ticket not found.")
    return task


@router.get("/resumes/{resume_id}", response_model=ResumeDocument, summary="Retrieve Parsed Resume")
async def get_resume_by_id(resume_id: str):
    """Retrieves a previously parsed resume document from the database."""
    resume = await db_client.get_resume(resume_id)
    if not resume:
        raise HTTPException(status_code=404, detail="Resume record not found.")
    return resume


@router.get("/resumes", response_model=List[ResumeDocument], summary="List Parsed Resumes")
async def list_resumes(
    limit: int = Query(20, ge=1, le=100, description="Max records to retrieve"),
    skip: int = Query(0, ge=0, description="Offset index for pagination")
):
    """Retrieves a paginated list of all parsed resume documents."""
    return await db_client.list_resumes(limit=limit, skip=skip)
