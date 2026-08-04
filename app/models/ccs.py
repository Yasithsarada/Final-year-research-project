from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class CCSClaimSet(BaseModel):
    technical_skills: List[str] = Field(default_factory=list, description="Explicit technical skills mentioned (e.g. Docker, Kubernetes, CI/CD)")
    frameworks: List[str] = Field(default_factory=list, description="Frameworks mentioned (e.g. FastAPI, React, Angular)")
    tools: List[str] = Field(default_factory=list, description="Tools or utilities mentioned (e.g. Git, Jenkins, Webpack)")
    programming_languages: List[str] = Field(default_factory=list, description="Programming languages mentioned (e.g. Python, Java, Go)")
    cloud_platforms: List[str] = Field(default_factory=list, description="Cloud platforms mentioned (e.g. AWS, GCP, Azure)")
    databases: List[str] = Field(default_factory=list, description="Databases or datastores mentioned (e.g. PostgreSQL, MongoDB, Redis)")
    years_of_experience: Optional[str] = Field(None, description="Verbal claims about duration/years of experience (e.g. '5 years', 'since 2018')")
    job_roles: List[str] = Field(default_factory=list, description="Job titles or roles candidate claimed to have held (e.g. Backend Engineer, Team Lead)")
    projects_claimed: List[str] = Field(default_factory=list, description="Projects candidate claimed to have led or worked on")
    leadership_claims: List[str] = Field(default_factory=list, description="Claims of leading teams, mentoring, or managing (e.g. 'managed a team of 6')")
    certifications: List[str] = Field(default_factory=list, description="Certifications, courses, or licenses mentioned")
    soft_skills: List[str] = Field(default_factory=list, description="Soft skills mentioned (e.g. communication, problem solving)")
    confidence_score: float = Field(default=1.0, description="LLM extraction self-reported confidence score (0.0 to 1.0)")
    supporting_sentences: List[str] = Field(default_factory=list, description="The literal sentences from the transcript that back up the extracted claims")

class CCSTaskStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class CCSJobDocument(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    candidate_id: Optional[str] = None
    filename: str
    status: CCSTaskStatus = CCSTaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    
    # Transcription fields
    raw_transcript: Optional[str] = None
    cleaned_transcript: Optional[str] = None
    
    # Extracted Claims
    claims: Optional[CCSClaimSet] = None
    
    # Explanations and research-grade comparisons
    explainability_report: List[str] = Field(default_factory=list, description="Explainable AI output and rationale mapping")
    comparisons: Dict[str, Any] = Field(default_factory=dict, description="Pre-calculated comparison vectors/mismatch flags against resume or GitHub")
    
    error: Optional[str] = None

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class CCSAsyncUploadResponse(BaseModel):
    job_id: str
    status: CCSTaskStatus
    message: str
    poll_url: str
