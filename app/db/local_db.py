import json
import uuid
from typing import List, Optional
from pathlib import Path
from datetime import datetime

from app.db.base import BaseDatabase
from app.models.resume import ResumeDocument
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
