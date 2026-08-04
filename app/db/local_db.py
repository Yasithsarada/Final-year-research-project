import json
import uuid
from typing import List, Optional
from pathlib import Path
from datetime import datetime

from app.db.base import BaseDatabase
from app.models.resume import ResumeDocument
from app.models.ccs import CCSJobDocument
from app.models.candidate_profile import CandidateProfileDocument
from app.core.config import settings

class LocalFileDatabase(BaseDatabase):
    """File-system based JSON database implementation for easy development/testing."""

    def __init__(self):
        pass

    @property
    def storage_dir(self) -> Path:
        """Dynamically resolve storage directory to allow settings modifications in tests."""
        path = settings.LOCAL_STORAGE_DIR / "resumes"
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def connect(self) -> None:
        # Resolve to trigger folder creation
        _ = self.storage_dir

    async def disconnect(self) -> None:
        pass

    async def save_resume(self, document: ResumeDocument) -> str:
        if not document.id:
            document.id = str(uuid.uuid4())
            
        file_path = self.storage_dir / f"{document.id}.json"
        
        # Serialize with Pydantic's model_dump
        data = document.model_dump(by_alias=True)
        # Convert datetime objects to string format
        data["uploaded_at"] = data["uploaded_at"].isoformat()
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        return document.id

    async def get_resume(self, resume_id: str) -> Optional[ResumeDocument]:
        file_path = self.storage_dir / f"{resume_id}.json"
        if not file_path.exists():
            return None
            
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Parse datetime back
        data["uploaded_at"] = datetime.fromisoformat(data["uploaded_at"])
        return ResumeDocument.model_validate(data)

    async def list_resumes(self, limit: int = 20, skip: int = 0) -> List[ResumeDocument]:
        resumes = []
        file_paths = list(self.storage_dir.glob("*.json"))
        
        # Sort files by modification time (newest first)
        file_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Implement pagination
        paginated_paths = file_paths[skip:skip + limit]
        
        for file_path in paginated_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["uploaded_at"] = datetime.fromisoformat(data["uploaded_at"])
                resumes.append(ResumeDocument.model_validate(data))
            except Exception:
                continue
                
        return resumes

    @property
    def ccs_storage_dir(self) -> Path:
        """Dynamically resolve CCS storage directory."""
        path = settings.LOCAL_STORAGE_DIR / "ccs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def save_ccs_job(self, document: CCSJobDocument) -> str:
        if not document.id:
            document.id = str(uuid.uuid4())
            
        file_path = self.ccs_storage_dir / f"{document.id}.json"
        
        data = document.model_dump(by_alias=True)
        data["created_at"] = data["created_at"].isoformat()
        if data.get("finished_at"):
            data["finished_at"] = data["finished_at"].isoformat()
            
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        return document.id

    async def get_ccs_job(self, job_id: str) -> Optional[CCSJobDocument]:
        file_path = self.ccs_storage_dir / f"{job_id}.json"
        if not file_path.exists():
            return None
            
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("finished_at"):
            data["finished_at"] = datetime.fromisoformat(data["finished_at"])
        return CCSJobDocument.model_validate(data)

    async def list_ccs_jobs(self, limit: int = 20, skip: int = 0) -> List[CCSJobDocument]:
        jobs = []
        file_paths = list(self.ccs_storage_dir.glob("*.json"))
        file_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        paginated_paths = file_paths[skip:skip + limit]
        
        for file_path in paginated_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["created_at"] = datetime.fromisoformat(data["created_at"])
                if data.get("finished_at"):
                    data["finished_at"] = datetime.fromisoformat(data["finished_at"])
                jobs.append(CCSJobDocument.model_validate(data))
            except Exception:
                continue
                
        return jobs

    # ---------------------------------------------------------------
    # Candidate Profile
    # ---------------------------------------------------------------

    @property
    def profile_storage_dir(self) -> Path:
        path = settings.LOCAL_STORAGE_DIR / "profiles"
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def save_candidate_profile(self, document: CandidateProfileDocument) -> str:
        if not document.id:
            document.id = str(uuid.uuid4())

        file_path = self.profile_storage_dir / f"{document.id}.json"
        data = document.model_dump(by_alias=True)
        data["created_at"] = data["created_at"].isoformat()
        data["updated_at"] = data["updated_at"].isoformat()

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        return document.id

    async def get_candidate_profile(self, profile_id: str) -> Optional[CandidateProfileDocument]:
        file_path = self.profile_storage_dir / f"{profile_id}.json"
        if not file_path.exists():
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return CandidateProfileDocument.model_validate(data)

    async def list_candidate_profiles(self, limit: int = 20, skip: int = 0) -> List[CandidateProfileDocument]:
        profiles = []
        file_paths = list(self.profile_storage_dir.glob("*.json"))
        file_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        for file_path in file_paths[skip: skip + limit]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["created_at"] = datetime.fromisoformat(data["created_at"])
                data["updated_at"] = datetime.fromisoformat(data["updated_at"])
                profiles.append(CandidateProfileDocument.model_validate(data))
            except Exception:
                continue

        return profiles
