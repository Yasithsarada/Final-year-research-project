from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime

# ==========================================
# Sub-models for structured LLM extraction
# ==========================================

class EducationItem(BaseModel):
    institution: str = Field(description="Name of the university, college, or school")
    degree: Optional[str] = Field(None, description="Degree type, e.g., Bachelor of Science, MSc, PhD")
    field_of_study: Optional[str] = Field(None, description="Major or field of study, e.g., Computer Science")
    start_date: Optional[str] = Field(None, description="Start date of study (e.g. Year or MM/YYYY)")
    end_date: Optional[str] = Field(None, description="End date or 'Present' if current")
    gpa: Optional[str] = Field(None, description="GPA or grade if listed")

class WorkExperienceItem(BaseModel):
    company: str = Field(description="Name of the company or employer")
    role: str = Field(description="Job title or role held")
    start_date: Optional[str] = Field(None, description="Start date of employment (e.g. Month/Year or Year)")
    end_date: Optional[str] = Field(None, description="End date of employment, or 'Present'")
    description: Optional[str] = Field(None, description="Summary of responsibilities and achievements")
    technologies: List[str] = Field(default_factory=list, description="Technologies, programming languages, and tools explicitly mentioned in this job context")

class ProjectItem(BaseModel):
    name: str = Field(description="Name of the project")
    description: Optional[str] = Field(None, description="Description of what was built and what technologies were used")
    technologies: List[str] = Field(default_factory=list, description="List of technologies used specifically in this project")
    link: Optional[str] = Field(None, description="URL to repository or live site if listed")

class CertificationItem(BaseModel):
    name: str = Field(description="Name of the certification, license, or course")
    issuing_organization: Optional[str] = Field(None, description="Organization that issued the certification, e.g., AWS, Coursera, Google")
    issue_date: Optional[str] = Field(None, description="Date of issuance")

class LanguageItem(BaseModel):
    language: str = Field(description="Language name, e.g. English, Spanish")
    proficiency: Optional[str] = Field(None, description="Proficiency level, e.g., Native, Fluent, Intermediate, Professional")

# ==========================================
# Main Extracted Resume Schema (LLM Direct)
# ==========================================

class ExtractedResumeSchema(BaseModel):
    full_name: str = Field(..., description="First and last name of candidate")
    email: str = Field(..., description="Email address of candidate")
    phone: str = Field(..., description="Phone number or contact number")
    github: str = Field(..., description="GitHub profile URL or username")
    linkedin: str = Field(..., description="LinkedIn profile URL")
    skills: List[str] = Field(default_factory=list, description="All technical skills, programming languages, and tools listed")
    education: List[EducationItem] = Field(default_factory=list, description="Academic background details")
    work_experience: List[WorkExperienceItem] = Field(default_factory=list, description="Professional work history")
    projects: List[ProjectItem] = Field(default_factory=list, description="Personal or academic projects")
    certifications: List[CertificationItem] = Field(default_factory=list, description="Certifications and courses")
    languages: List[LanguageItem] = Field(default_factory=list, description="Languages spoken")
    total_years_experience: str = Field(..., description="Calculated total years of experience, e.g. '3 years', '8 years'")
    claimed_role: str = Field(..., description="The main role the candidate claims to be, e.g. 'Senior Fullstack Engineer', 'Data Scientist'")

# ==========================================
# Normalized & Enhanced Resume Schema (For Database & Fraud Detection)
# ==========================================

class NormalizedSkill(BaseModel):
    original: str = Field(..., description="Original skill name extracted from the resume")
    canonical: str = Field(..., description="Mapped canonical name from the skill database/taxonomy")
    category: str = Field(..., description="General category, e.g., Backend, Frontend, DevOps, ML")
    similarity_score: float = Field(..., description="Cosine similarity score of the mapping")

class TraceSpan(BaseModel):
    field: str = Field(..., description="The Pydantic field path, e.g., 'full_name' or 'work_experience[0].company'")
    extracted_value: str = Field(..., description="The extracted value")
    raw_snippet: str = Field(..., description="The raw segment from the original text where this was found")
    confidence: float = Field(..., description="The certainty of this trace mapping")

class ResumeDocument(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    filename: str
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Pure extracted entities
    entities: ExtractedResumeSchema
    
    # Research-grade Enhancements
    normalized_skills: List[NormalizedSkill] = Field(default_factory=list)
    explainability_traces: List[TraceSpan] = Field(default_factory=list)
    confidence_score: float = Field(..., description="Overall pipeline confidence score (0.0 to 1.0)")
    
    # Section Segmentation Storage
    segmented_sections: Dict[str, str] = Field(default_factory=dict)
    
    # Verification and Fraud Precheck flags
    validation_report: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Results of comparing regex-anchors vs LLM, detecting formatting inconsistencies, or date anomalies."
    )
    
    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
