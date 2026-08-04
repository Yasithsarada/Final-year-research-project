import asyncio
import unittest
from unittest.mock import MagicMock, patch
import datetime
import os
import shutil

from app.core.config import settings, BASE_DIR
from app.models.ccs import CCSClaimSet, CCSJobDocument, CCSTaskStatus
from app.models.resume import ExtractedResumeSchema, ResumeDocument, WorkExperienceItem
from app.services.postprocess import TranscriptPostProcessor
from app.services.explainer import CCSExplainer
from app.services.comparison import ClaimComparisonService
from app.tasks.jobs import _async_process_audio_interview
from app.db.local_db import LocalFileDatabase

class TestCCSPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Set up a test directory for database testing
        cls.test_db_dir = BASE_DIR / "app_data_test_ccs"
        settings.LOCAL_STORAGE_DIR = cls.test_db_dir
        
        cls.post_processor = TranscriptPostProcessor()
        cls.explainer = CCSExplainer()
        cls.comparison_service = ClaimComparisonService()

    @classmethod
    def tearDownClass(cls):
        # Clean up test directories
        if cls.test_db_dir.exists():
            shutil.rmtree(cls.test_db_dir)

    def test_transcript_cleaning(self):
        """Tests that cleaner removes filler words and excessive pauses."""
        raw_text = "So, uh, I basically worked with, you know, Kubernetes, and, um, Docker."
        cleaned = self.post_processor.clean_text(raw_text)
        
        self.assertNotIn("uh", cleaned.lower())
        self.assertNotIn("um", cleaned.lower())
        self.assertNotIn("you know", cleaned.lower())
        self.assertIn("Kubernetes", cleaned)
        self.assertIn("Docker", cleaned)

    def test_explainability_generation(self):
        """Tests that explainer generates human-readable reports linking claims to evidence."""
        claims = CCSClaimSet(
            technical_skills=["Docker", "Kubernetes"],
            frameworks=["FastAPI"],
            tools=["Git"],
            programming_languages=["Python"],
            cloud_platforms=["AWS"],
            databases=["PostgreSQL"],
            years_of_experience="4 years",
            job_roles=["Backend Engineer"],
            projects_claimed=["CRM Migration"],
            leadership_claims=["Managed 6 engineers"],
            certifications=[],
            soft_skills=[],
            confidence_score=0.9,
            supporting_sentences=["I have been using Docker and Kubernetes for 4 years.", "I managed 6 engineers."]
        )
        
        report = self.explainer.generate_explanation_report(claims)
        self.assertTrue(len(report) > 0)
        self.assertTrue(any("4 years" in r for r in report))
        self.assertTrue(any("Managed 6 engineers" in r for r in report))

    def test_comparison_logic(self):
        """Tests the comparison engine for resume/interview mismatches and contradictions."""
        job_doc = CCSJobDocument(
            filename="interview.mp3",
            status=CCSTaskStatus.COMPLETED,
            claims=CCSClaimSet(
                technical_skills=["Kubernetes"],
                frameworks=[],
                tools=[],
                programming_languages=["Java"],
                cloud_platforms=[],
                databases=[],
                years_of_experience="5 years",
                job_roles=[],
                projects_claimed=[],
                leadership_claims=["Managed a team of 10"],
                certifications=[],
                soft_skills=[],
                confidence_score=0.9,
                supporting_sentences=[]
            )
        )
        
        # Scenario A: Contradicting experience and leadership
        resume = ResumeDocument(
            filename="resume.pdf",
            entities=ExtractedResumeSchema(
                full_name="John Doe",
                email="john@doe.com",
                phone="123",
                github="",
                linkedin="",
                skills=["Java"],
                education=[],
                work_experience=[
                    WorkExperienceItem(company="A", role="Developer", start_date="2022", end_date="Present")
                ],
                projects=[],
                certifications=[],
                languages=[],
                total_years_experience="2 years", # Discrepancy (5 years claimed vs 2 in resume)
                claimed_role="Developer"
            ),
            confidence_score=1.0
        )
        
        res = self.comparison_service.compare_interview_vs_resume(job_doc, resume)
        self.assertTrue(res["experience_discrepancy"])
        self.assertEqual(len(res["contradictions"]), 2) # Experience discrepancy and leadership role mismatch
        self.assertTrue(res["mismatch_index"] > 0.5)

    def test_local_db_operations(self):
        """Tests saving and retrieving CCS jobs in local file database."""
        async def run_db_test():
            db = LocalFileDatabase()
            await db.connect()
            
            job = CCSJobDocument(
                filename="interview.wav",
                status=CCSTaskStatus.PENDING,
                candidate_id="c_123"
            )
            
            job_id = await db.save_ccs_job(job)
            self.assertIsNotNone(job_id)
            
            retrieved = await db.get_ccs_job(job_id)
            self.assertEqual(retrieved.filename, "interview.wav")
            self.assertEqual(retrieved.status, CCSTaskStatus.PENDING)
            self.assertEqual(retrieved.candidate_id, "c_123")
            
            jobs = await db.list_ccs_jobs()
            self.assertTrue(len(jobs) >= 1)
            
        asyncio.run(run_db_test())

    @patch('app.services.audio_processor.AudioProcessor.transcribe_audio')
    @patch('app.services.claim_extractor.LLMClaimExtractorClient.extract_claims')
    def test_pipeline_task(self, mock_extract, mock_transcribe):
        """Test task pipeline with mocked transcription and LLM clients."""
        mock_transcribe.return_value = (
            "I worked with Kubernetes and Python for 3 years at Stripe.", 
            [{"start": 0.0, "end": 5.0, "text": "I worked with Kubernetes and Python for 3 years at Stripe."}]
        )
        
        mock_claims = CCSClaimSet(
            technical_skills=["Kubernetes"],
            frameworks=[],
            tools=[],
            programming_languages=["Python"],
            cloud_platforms=[],
            databases=[],
            years_of_experience="3 years",
            job_roles=["Backend Engineer"],
            projects_claimed=[],
            leadership_claims=[],
            certifications=[],
            soft_skills=[],
            confidence_score=0.95,
            supporting_sentences=["I worked with Kubernetes and Python for 3 years at Stripe."]
        )
        mock_extract.return_value = mock_claims
        
        async def run_pipeline():
            db = LocalFileDatabase()
            await db.connect()
            
            # Create a dummy mock_file.wav
            with open("mock_file.wav", "w") as f:
                f.write("dummy audio content")
            
            try:
                # Pre-register a job in database
                job = CCSJobDocument(
                    id="test-pipeline-job-id",
                    filename="test.wav",
                    status=CCSTaskStatus.PENDING
                )
                await db.save_ccs_job(job)
                
                # Execute pipeline
                await _async_process_audio_interview("test-pipeline-job-id", "mock_file.wav", None)
                
                # Verify database update
                finished_job = await db.get_ccs_job("test-pipeline-job-id")
                self.assertEqual(finished_job.status, CCSTaskStatus.COMPLETED)
                self.assertEqual(finished_job.raw_transcript, "I worked with Kubernetes and Python for 3 years at Stripe.")
                self.assertEqual(finished_job.claims.technical_skills, ["Kubernetes"])
                self.assertEqual(finished_job.claims.programming_languages, ["Python"])
            finally:
                # Ensure dummy file is deleted
                if os.path.exists("mock_file.wav"):
                    os.remove("mock_file.wav")
            
        asyncio.run(run_pipeline())

if __name__ == "__main__":
    unittest.main()
