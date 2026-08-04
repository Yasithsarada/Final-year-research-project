from abc import ABC, abstractmethod
from typing import List, Optional
from app.models.resume import ResumeDocument
from app.models.ccs import CCSJobDocument
from app.models.candidate_profile import CandidateProfileDocument

class BaseDatabase(ABC):
    """Abstract Base Class defining the database interface for resume and CCS documents."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish database connections."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connections."""
        pass

    @abstractmethod
    async def save_resume(self, document: ResumeDocument) -> str:
        """Saves a parsed resume document and returns its unique ID."""
        pass

    @abstractmethod
    async def get_resume(self, resume_id: str) -> Optional[ResumeDocument]:
        """Retrieves a parsed resume document by its ID."""
        pass

    @abstractmethod
    async def list_resumes(self, limit: int = 20, skip: int = 0) -> List[ResumeDocument]:
        """Lists parsed resume documents with pagination."""
        pass

    @abstractmethod
    async def save_ccs_job(self, document: CCSJobDocument) -> str:
        """Saves an audio CCS job and returns its unique ID."""
        pass

    @abstractmethod
    async def get_ccs_job(self, job_id: str) -> Optional[CCSJobDocument]:
        """Retrieves a CCS job by its ID."""
        pass

    @abstractmethod
    async def list_ccs_jobs(self, limit: int = 20, skip: int = 0) -> List[CCSJobDocument]:
        """Lists CCS jobs with pagination."""
        pass

    @abstractmethod
    async def save_candidate_profile(self, document: CandidateProfileDocument) -> str:
        """Saves a candidate profile and returns its unique ID."""
        pass

    @abstractmethod
    async def get_candidate_profile(self, profile_id: str) -> Optional[CandidateProfileDocument]:
        """Retrieves a candidate profile by its ID."""
        pass

    @abstractmethod
    async def list_candidate_profiles(self, limit: int = 20, skip: int = 0) -> List[CandidateProfileDocument]:
        """Lists candidate profiles with pagination."""
        pass
