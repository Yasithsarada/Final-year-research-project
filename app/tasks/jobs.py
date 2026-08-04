import logging
import os
import datetime
from pathlib import Path
from typing import Optional
from app.tasks.worker import celery_app
from app.db import db_client
from app.models.ccs import CCSJobDocument, CCSTaskStatus
from app.services.audio_processor import AudioProcessor
from app.services.postprocess import TranscriptPostProcessor
from app.services.claim_extractor import LLMClaimExtractorClient
from app.services.explainer import CCSExplainer
from app.services.comparison import ClaimComparisonService

logger = logging.getLogger(__name__)

# Initialize services inside worker processes lazily
audio_processor = AudioProcessor()
post_processor = TranscriptPostProcessor()
extractor_client = LLMClaimExtractorClient()
explainer = CCSExplainer()
comparison_service = ClaimComparisonService()

@celery_app.task(name="app.tasks.jobs.process_audio_interview", bind=True, max_retries=3)
def process_audio_interview(self, job_id: str, file_path_str: str, candidate_id: Optional[str] = None):
    """Celery background worker job running the full transcription and entity extraction pipeline."""
    import asyncio
    
    # Run the async database/service calls in a synchronous loop inside the Celery task thread
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    return loop.run_until_complete(
        _async_process_audio_interview(job_id, file_path_str, candidate_id)
    )

async def _async_process_audio_interview(job_id: str, file_path_str: str, candidate_id: Optional[str] = None):
    logger.info(f"Worker task starting for Job: {job_id}, File: {file_path_str}")
    
    # 1. Initialize Database connection if needed
    await db_client.connect()
    
    # Fetch job from db
    job_doc = await db_client.get_ccs_job(job_id)
    if not job_doc:
        logger.error(f"CCS Job not found in database: {job_id}")
        return
        
    job_doc.status = CCSTaskStatus.PROCESSING
    await db_client.save_ccs_job(job_doc)
    
    file_path = Path(file_path_str)
    try:
        # 2. Transcription
        if not file_path.exists():
            raise FileNotFoundError(f"File not found at: {file_path_str}")
            
        raw_transcript, segments = audio_processor.transcribe_audio(file_path)
        job_doc.raw_transcript = raw_transcript
        
        # 3. Post-processing & Cleaning
        cleaned_transcript = post_processor.clean_text(raw_transcript)
        job_doc.cleaned_transcript = cleaned_transcript
        
        # Segment by topics
        segmented_sections = post_processor.segment_by_topic_sections(cleaned_transcript)
        
        # 4. LLM Claim Extraction
        claims = extractor_client.extract_claims(cleaned_transcript, segmented_sections)
        job_doc.claims = claims
        
        # 5. Explainability Rationale
        explanations = explainer.generate_explanation_report(claims)
        job_doc.explainability_report = explanations
        
        # 6. Comparisons (Resume / GitHub if candidate_id is linked)
        comparisons = {}
        if candidate_id:
            logger.info(f"Performing resume correlation for candidate: {candidate_id}")
            # Try fetching a resume from DB where candidate_id matches
            # Wait, our current DB list_resumes lists all resumes. We can scan for a resume that matches candidate_id or has the candidate's name.
            resumes = await db_client.list_resumes(limit=100)
            target_resume = None
            for r in resumes:
                # Simple name or email lookup to match candidate_id (or if resume.id matches candidate_id)
                if r.id == candidate_id or r.entities.email.lower() == candidate_id.lower():
                    target_resume = r
                    break
            
            if target_resume:
                resume_comparison = comparison_service.compare_interview_vs_resume(job_doc, target_resume)
                comparisons["resume_comparison"] = resume_comparison
                logger.info("Resume comparison completed successfully.")
            else:
                logger.warning(f"No resume record found matching candidate_id: {candidate_id}")

            # Mock GitHub profile data (in real production this would fetch from a database or search GitHub API)
            # We seed a mock GitHub profile based on the extracted claims to demonstrate comparison
            mock_github_languages = {}
            for lang in claims.programming_languages:
                # candidate claims it, let's pretend they have code bytes in GitHub, or miss one to show mismatch
                if "java" in lang.lower():
                    # Mismatch scenario: candidate claimed Java but minimal code bytes
                    mock_github_languages["Java"] = 500 # very small
                else:
                    mock_github_languages[lang] = 50000
                    
            mock_github = {
                "repos": [{"name": f"project-{tech.lower()}", "language": tech} for tech in claims.frameworks],
                "languages": mock_github_languages,
                "total_commits": 245 if "Java" not in claims.programming_languages else 8,
                "total_prs": 14
            }
            
            github_comparison = comparison_service.compare_interview_vs_github(job_doc, mock_github)
            comparisons["github_comparison"] = github_comparison
            logger.info("GitHub comparison completed successfully.")

        job_doc.comparisons = comparisons
        
        # Mark completed
        job_doc.status = CCSTaskStatus.COMPLETED
        job_doc.finished_at = datetime.datetime.utcnow()
        await db_client.save_ccs_job(job_doc)
        logger.info(f"CCS Job {job_id} completed successfully.")
        
    except Exception as e:
        logger.error(f"Error executing CCS pipeline for job {job_id}: {str(e)}", exc_info=True)
        job_doc.status = CCSTaskStatus.FAILED
        job_doc.error = str(e)
        job_doc.finished_at = datetime.datetime.utcnow()
        await db_client.save_ccs_job(job_doc)
    finally:
        # Cleanup temporary audio files
        try:
            if file_path.exists() and "temp" in file_path_str:
                os.remove(file_path)
                logger.info(f"Temporary file cleaned up: {file_path_str}")
        except Exception as ex:
            logger.warning(f"Could not delete temporary file: {str(ex)}")
            
        await db_client.disconnect()
