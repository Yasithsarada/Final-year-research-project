from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.msr_agent import MSRAgent
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/msr", tags=["MSR"])

class MSRRequest(BaseModel):
    github_url: str

@router.post("/analyze", summary="Analyze a GitHub profile using MSR Agent")
async def analyze_github_profile(request: MSRRequest):
    """
    Fetches raw signals from the GitHub API using the MSRAgent:
    - Commit history
    - Pull Requests
    - Repository metadata
    """
    agent = MSRAgent()
    try:
        metrics = await agent.analyze_profile(request.github_url)
        return metrics
    except Exception as e:
        logger.error(f"Error analyzing profile {request.github_url}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
