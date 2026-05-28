import asyncio
import unittest
from unittest.mock import MagicMock, patch
import json
import os
import shutil

from app.core.config import settings, BASE_DIR
from app.models.resume import ExtractedResumeSchema, EducationItem, WorkExperienceItem, ProjectItem, LanguageItem
from app.services.cleaner import ResumeCleaner
from app.services.normalizer import SkillNormalizer
from app.services.explainability import ExplainabilityEngine
from app.services.pipeline import ResumeParsingPipeline
from app.db.local_db import LocalFileDatabase

class TestResumeParserPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Set up a test directory for database testing
        cls.test_db_dir = BASE_DIR / "app_data_test"
        settings.LOCAL_STORAGE_DIR = cls.test_db_dir
        
        cls.cleaner = ResumeCleaner()
        cls.normalizer = SkillNormalizer()
        cls.explainability = ExplainabilityEngine()

    @classmethod
    def tearDownClass(cls):
        # Clean up test directories
        if cls.test_db_dir.exists():
            shutil.rmtree(cls.test_db_dir)

    def test_text_cleaning(self):
        """Tests that text cleaner removes bad characters and collapses spaces."""
        raw_text = "John   Doe\xa0\xa0\n\n\nSoftware   Engineer\r\nLocation: NY"
        cleaned = self.cleaner.clean_text(raw_text)
        
        self.assertIn("John Doe", cleaned)
        self.assertIn("Software Engineer", cleaned)
        self.assertNotIn("\xa0", cleaned)
        self.assertNotIn("\r", cleaned)
        self.assertNotIn("   ", cleaned)

    def test_anchor_extraction(self):
        """Tests that contact regex anchors are extracted successfully."""
        test_text = """
        John Doe
        Email: john.doe@example.com
        Cell: +1-555-0199 or 555.222.1111
        GitHub: https://github.com/johndoe
        LinkedIn: linkedin.com/in/johndoe-dev
        """
        anchors = self.cleaner.extract_anchors(test_text)
        
        self.assertIn("john.doe@example.com", anchors["emails"])
        self.assertTrue(len(anchors["phones"]) >= 1)
        self.assertTrue(any("github.com/johndoe" in link for link in anchors["github_links"]))
        self.assertTrue(any("linkedin.com/in/johndoe-dev" in link for link in anchors["linkedin_links"]))

    def test_section_segmentation(self):
        """Tests that the cleaner segments text into sections by headers."""
        resume_text = """
        John Doe
        john@doe.com
        
        EDUCATION
        B.S. Computer Science - MIT (2018-2022)
        
        WORK EXPERIENCE
        Software Engineer at Google (2022-Present)
        Built cool search queries.
        
        SKILLS
        Python, React, Docker, SQL
        """
        sections = self.cleaner.segment_sections(resume_text)
        
        self.assertIn("education", sections)
        self.assertIn("work_experience", sections)
        self.assertIn("skills", sections)
        self.assertIn("B.S. Computer Science", sections["education"])
        self.assertIn("Software Engineer at Google", sections["work_experience"])

    def test_skill_normalization(self):
        """Tests both exact matching and Jaccard similarity fallback for skills."""
        # Exact match of registered alias
        norm1 = self.normalizer.normalize_skill("reactjs")
        self.assertEqual(norm1.canonical, "React")
        self.assertEqual(norm1.category, "Frontend")
        self.assertEqual(norm1.similarity_score, 1.0)
        
        # Exact match of canonical
        norm2 = self.normalizer.normalize_skill("Python")
        self.assertEqual(norm2.canonical, "Python")
        self.assertEqual(norm2.similarity_score, 1.0)

        # Jaccard matching fallback check
        norm3 = self.normalizer.normalize_skill("python 3 interpreter")
        self.assertEqual(norm3.canonical, "Python")

    def test_explainability_and_confidence(self):
        """Tests trace generation and overall confidence rating."""
        full_text = """
        Jane Smith
        jane.smith@email.com
        Technical Lead at OpenAI. Worked from 2020 to present.
        BSc in Artificial Intelligence from Stanford.
        """
        
        mock_entities = ExtractedResumeSchema(
            full_name="Jane Smith",
            email="jane.smith@email.com",
            phone="",
            github="",
            linkedin="",
            skills=["Python"],
            education=[EducationItem(institution="Stanford", degree="BSc", field_of_study="Artificial Intelligence", start_date="2016", end_date="2020")],
            work_experience=[WorkExperienceItem(company="OpenAI", role="Technical Lead", start_date="2020", end_date="Present", description="Lead teams")],
            projects=[],
            certifications=[],
            languages=[],
            total_years_experience="4 years",
            claimed_role="Tech Lead"
        )
        
        traces = self.explainability.trace_entities(mock_entities, full_text)
        self.assertTrue(len(traces) > 0)
        
        # Verify trace mapping
        name_trace = next(t for t in traces if t.field == "full_name")
        self.assertEqual(name_trace.extracted_value, "Jane Smith")
        self.assertIn("Jane Smith", name_trace.raw_snippet)
        
        # Evaluate confidence
        anchors = self.cleaner.extract_anchors(full_text)
        score, report = self.explainability.evaluate_confidence(mock_entities, anchors, traces)
        
        self.assertGreater(score, 0.5)
        self.assertTrue(report["email_verified"])

    def test_local_database_operations(self):
        """Tests that local file database successfully saves and retrieves documents."""
        # Run test asynchronously
        async def run_db_test():
            db = LocalFileDatabase()
            await db.connect()
            
            mock_entities = ExtractedResumeSchema(
                full_name="Database Tester", email="db@test.com", phone="123", github="", linkedin="",
                skills=[], education=[], work_experience=[], projects=[], certifications=[], languages=[],
                total_years_experience="0", claimed_role="QA"
            )
            
            from app.models.resume import ResumeDocument
            doc = ResumeDocument(
                filename="test.pdf",
                entities=mock_entities,
                confidence_score=1.0
            )
            
            doc_id = await db.save_resume(doc)
            self.assertIsNotNone(doc_id)
            
            retrieved = await db.get_resume(doc_id)
            self.assertEqual(retrieved.entities.full_name, "Database Tester")
            
            all_resumes = await db.list_resumes()
            self.assertTrue(len(all_resumes) >= 1)
            
        asyncio.run(run_db_test())

    @patch('app.services.llm_client.LLMResumeParserClient.extract_entities')
    @patch('app.services.extractor.ResumeExtractor.extract_text')
    def test_full_pipeline_with_mock_llm(self, mock_extract_text, mock_extract):
        """Integrative test validating end-to-end processing with a mocked LLM endpoint."""
        mock_entities = ExtractedResumeSchema(
            full_name="Pipeline Candidate",
            email="candidate@pipeline.com",
            phone="123-456-7890",
            github="https://github.com/candidate",
            linkedin="https://linkedin.com/in/candidate",
            skills=["python", "reactjs", "kubernetes"],
            education=[EducationItem(institution="MIT", degree="B.S.", field_of_study="CS")],
            work_experience=[WorkExperienceItem(company="Stripe", role="Backend Engineer", start_date="2021", end_date="Present")],
            projects=[],
            certifications=[],
            languages=[LanguageItem(language="English", proficiency="Native")],
            total_years_experience="5 years",
            claimed_role="Backend Engineer"
        )
        mock_extract.return_value = mock_entities
        
        test_resume = """
        Pipeline Candidate
        candidate@pipeline.com
        Phone: 123-456-7890
        github.com/candidate
        linkedin.com/in/candidate
        
        Work Experience
        Backend Engineer at Stripe (2021-Present)
        
        Education
        B.S. in CS from MIT
        
        Skills: python, reactjs, kubernetes
        """
        mock_extract_text.return_value = test_resume
        
        async def run_pipeline():
            pipeline = ResumeParsingPipeline()
            doc = await pipeline.process_resume(b"mock_bytes_here", "candidate_cv.pdf")
            
            # Verify pipeline mappings
            self.assertEqual(doc.filename, "candidate_cv.pdf")
            self.assertEqual(doc.entities.full_name, "Pipeline Candidate")
            self.assertEqual(len(doc.normalized_skills), 3)
            
            # Check normalized skills
            skills_canon = [s.canonical for s in doc.normalized_skills]
            self.assertIn("Python", skills_canon)
            self.assertIn("React", skills_canon)
            self.assertIn("Kubernetes", skills_canon)
            
            self.assertTrue(doc.confidence_score > 0.8)
            self.assertTrue(doc.validation_report["email_verified"])
            self.assertTrue(doc.validation_report["github_verified"])
            
        asyncio.run(run_pipeline())


if __name__ == "__main__":
    print("Running Resume Parser Test Suite...")
    unittest.main()
