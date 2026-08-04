import logging
import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.db import db_client
from app.models.candidate_profile import CandidateProfileDocument
from app.services.msr_agent import MSRAgent
from app.services.sca_agent import SCAAgent
from app.services.feature_engineering import FeatureEngineeringLayer
from app.services.decision_engine import DecisionEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["Candidate Evaluation (MSR + FEL + DE)"])

# Service singletons
fel = FeatureEngineeringLayer()
de = DecisionEngine()


class EvaluationRequest(BaseModel):
    candidate_name: str
    github_url: str
    resume_id: Optional[str] = None
    ccs_job_id: Optional[str] = None
    sonarqube_component_key: Optional[str] = None


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------

@router.post(
    "/analyze",
    response_model=CandidateProfileDocument,
    summary="Run Full Candidate Evaluation (MSR + SCA + FEL + DE)",
)
async def analyze_candidate(request: EvaluationRequest):
    """Runs the complete evaluation pipeline for a candidate:

    1. **MSR Agent** – mines commit history, PRs, and repo metadata from GitHub.
    2. **SCA Agent** – fetches code quality metrics from SonarQube (falls back to mock if offline).
    3. **Feature Engineering Layer** – derives TSE, NTS, and Fraud Indicators.
    4. **Decision Engine** – cross-references claims from CV and Audio Interview to
       produce Technical Skill Score, Collaboration Score, and Fraud Risk Level.
    5. Saves the `CandidateProfileDocument` and returns it.
    """
    logger.info(f"Evaluation pipeline starting for candidate: '{request.candidate_name}'")

    # ------------------------------------------------------------------
    # 1. Optionally load resume & audio documents for cross-verification
    # ------------------------------------------------------------------
    resume_doc = None
    audio_doc = None

    if request.resume_id:
        resume_doc = await db_client.get_resume(request.resume_id)
        if not resume_doc:
            raise HTTPException(status_code=404, detail=f"Resume '{request.resume_id}' not found.")

    if request.ccs_job_id:
        audio_doc = await db_client.get_ccs_job(request.ccs_job_id)
        if not audio_doc:
            raise HTTPException(status_code=404, detail=f"CCS job '{request.ccs_job_id}' not found.")

    # ------------------------------------------------------------------
    # 2. MSR — mine GitHub signals
    # ------------------------------------------------------------------
    try:
        msr = MSRAgent()
        github_metrics = await msr.analyze_profile(request.github_url)
    except Exception as e:
        logger.error(f"MSR Agent failed: {e}")
        raise HTTPException(status_code=502, detail=f"GitHub analysis failed: {str(e)}")

    # ------------------------------------------------------------------
    # 3. SCA — fetch code quality from SonarQube
    # ------------------------------------------------------------------
    try:
        sca = SCAAgent()
        component_key = request.sonarqube_component_key or github_metrics.username
        code_quality = await sca.analyze_project(component_key)
    except Exception as e:
        logger.warning(f"SCA Agent failed (falling back to mock): {e}")
        code_quality = SCAAgent._mock_metrics()

    # ------------------------------------------------------------------
    # 4. Feature Engineering Layer
    # ------------------------------------------------------------------
    tse = fel.compute_tse(github_metrics, code_quality)
    nts = fel.compute_nts(github_metrics)
    fi = fel.compute_fraud_indicators(github_metrics)

    # ------------------------------------------------------------------
    # 5. Decision Engine
    # ------------------------------------------------------------------
    decision = de.evaluate(tse, nts, fi, github_metrics, resume_doc, audio_doc)

    # ------------------------------------------------------------------
    # 6. Persist CandidateProfile
    # ------------------------------------------------------------------
    profile = CandidateProfileDocument(
        candidate_name=request.candidate_name,
        resume_id=request.resume_id,
        ccs_job_ids=[request.ccs_job_id] if request.ccs_job_id else [],
        github_username=github_metrics.username,
        github_metrics=github_metrics,
        technical_skills=tse,
        non_technical_skills=nts,
        fraud_indicators=fi,
        decision_result=decision,
        updated_at=datetime.datetime.utcnow(),
    )

    profile_id = await db_client.save_candidate_profile(profile)
    profile.id = profile_id
    logger.info(f"Evaluation complete. Profile saved with ID: {profile_id}")
    return profile


@router.get(
    "/profiles/{profile_id}",
    response_model=CandidateProfileDocument,
    summary="Retrieve Candidate Profile",
)
async def get_profile(profile_id: str):
    """Retrieves a previously evaluated candidate profile."""
    profile = await db_client.get_candidate_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Candidate profile not found.")
    return profile


@router.get(
    "/profiles",
    response_model=List[CandidateProfileDocument],
    summary="List All Candidate Profiles",
)
async def list_profiles(
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
):
    """Returns a paginated list of all evaluated candidate profiles."""
    return await db_client.list_candidate_profiles(limit=limit, skip=skip)
