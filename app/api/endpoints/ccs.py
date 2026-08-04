import uuid
import datetime
import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Request

from app.core.config import settings
from app.db import db_client
from app.models import CCSJobDocument, CCSTaskStatus, CCSAsyncUploadResponse
from app.tasks.jobs import process_audio_interview

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ccs", tags=["Interview Audio Claim Extraction (CCS)"])

ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a"}

def validate_audio_file(filename: str):
    """Ensures file extension is supported."""
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

# ==========================================
# API Route Handlers
# ==========================================

@router.post("/upload", response_model=CCSAsyncUploadResponse, summary="Upload Interview Audio (Asynchronous)")
async def upload_interview_audio(
    request: Request,
    file: UploadFile = File(...),
    candidate_id: Optional[str] = Form(None, description="Optional Candidate ID to correlate against Resume and GitHub")
):
    """Uploads candidate interview audio and schedules transcription & claim extraction in background queue.
    
    Returns a ticket (job ID) and polling URL immediately.
    """
    validate_audio_file(file.filename)
    
    # Ensure temp dir exists
    temp_dir = settings.LOCAL_STORAGE_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    job_id = str(uuid.uuid4())
    
    # Save uploaded file to temp path
    file_ext = file.filename.split(".")[-1]
    temp_file_path = temp_dir / f"{job_id}.{file_ext}"
    
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        with open(temp_file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Failed to save uploaded audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {str(e)}")

    # Initialize background task state in DB
    now = datetime.datetime.utcnow()
    job_doc = CCSJobDocument(
        id=job_id,
        candidate_id=candidate_id,
        filename=file.filename,
        status=CCSTaskStatus.PENDING,
        created_at=now
    )
    
    try:
        await db_client.save_ccs_job(job_doc)
    except Exception as e:
        # Cleanup temp file
        if temp_file_path.exists():
            temp_file_path.unlink()
        logger.error(f"Failed to register CCS job in DB: {str(e)}")
        raise HTTPException(status_code=500, detail="Database register failure.")
        
    # Dispatch Celery background task
    try:
        process_audio_interview.delay(job_id, str(temp_file_path), candidate_id)
        logger.info(f"Dispatched Celery job {job_id} for audio file {file.filename}")
    except Exception as e:
        # Clean up temp file and update DB status
        if temp_file_path.exists():
            temp_file_path.unlink()
        job_doc.status = CCSTaskStatus.FAILED
        job_doc.error = f"Celery dispatch failed: {str(e)}"
        await db_client.save_ccs_job(job_doc)
        logger.error(f"Celery dispatch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Task queue dispatch failed: {str(e)}")
        
    # Construct polling URL
    base_url = str(request.base_url).rstrip("/")
    poll_url = f"{base_url}/api/v1/ccs/tasks/{job_id}"
    
    return CCSAsyncUploadResponse(
        job_id=job_id,
        status=CCSTaskStatus.PENDING,
        message="Interview audio upload successful. Transcription and claim extraction queued.",
        poll_url=poll_url
    )


@router.get("/tasks/{job_id}", response_model=CCSJobDocument, summary="Check CCS Processing Task Status")
async def get_task_status(job_id: str):
    """Checks the status of an asynchronous audio analysis ticket."""
    job = await db_client.get_ccs_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ticket not found.")
    return job


@router.get("/jobs", response_model=List[CCSJobDocument], summary="List CCS Jobs")
async def list_ccs_jobs(
    limit: int = Query(20, ge=1, le=100, description="Max records to retrieve"),
    skip: int = Query(0, ge=0, description="Offset index for pagination")
):
    """Retrieves a paginated list of all interview audio extraction jobs."""
    return await db_client.list_ccs_jobs(limit=limit, skip=skip)


@router.get("/jobs/{job_id}", response_model=CCSJobDocument, summary="Retrieve Extracted CCS Claims")
async def get_ccs_job_by_id(job_id: str):
    """Retrieves a previously parsed CCS claims record from the database."""
    job = await db_client.get_ccs_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="CCS record not found.")
    return job
