from abc import ABC, abstractmethod
from typing import List, Optional
from app.models.resume import ResumeDocument

class BaseDatabase(ABC):
    """Abstract Base Class defining the database interface for resume documents."""

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
