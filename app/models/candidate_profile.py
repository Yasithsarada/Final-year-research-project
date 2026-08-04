import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.models.resume import ResumeDocument
from app.models.ccs import CCSJobDocument
from app.models.github_metrics import GitHubProfileMetrics
from app.models.evaluation import TechnicalSkillExpertise, NonTechnicalSkills, FraudIndicators, DecisionEngineResult

class CandidateProfileDocument(BaseModel):
    id: Optional[str] = None
    candidate_name: str
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

    # Core Data Sources
    resume_id: Optional[str] = None
    ccs_job_ids: List[str] = Field(default_factory=list) # Audio interview processing jobs
    github_username: Optional[str] = None
    
    # Processed Metrics
    github_metrics: Optional[GitHubProfileMetrics] = None
    technical_skills: Optional[TechnicalSkillExpertise] = None
    non_technical_skills: Optional[NonTechnicalSkills] = None
    fraud_indicators: Optional[FraudIndicators] = None
    
    # Final Decision Output
    decision_result: Optional[DecisionEngineResult] = None
