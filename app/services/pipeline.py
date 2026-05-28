import logging
from typing import Dict, Any, Tuple
import datetime

from app.db import db_client
from app.models.resume import ResumeDocument, ExtractedResumeSchema
from app.services.extractor import ResumeExtractor
from app.services.cleaner import ResumeCleaner
from app.services.llm_client import LLMResumeParserClient
from app.services.normalizer import SkillNormalizer
from app.services.explainability import ExplainabilityEngine
from app.core.exceptions import ParserException

logger = logging.getLogger(__name__)

class ResumeParsingPipeline:
    """Orchestrator pipeline coordinating extraction, cleaning, LLM parsing,
    normalization, and audit tracing.
    """

    def __init__(self):
        self.extractor = ResumeExtractor()
        self.cleaner = ResumeCleaner()
        self.llm_client = LLMResumeParserClient()
        self.normalizer = SkillNormalizer()
        self.explainability = ExplainabilityEngine()

    async def process_resume(self, file_bytes: bytes, filename: str) -> ResumeDocument:
        """Runs the complete ingestion pipeline for a given resume file.
        
        Extracts, cleans, parses, validates, normalizes, traces, and stores.
        """
        logger.info(f"Starting pipeline execution for file: {filename}")
        
        try:
            # 1. Text Extraction
            raw_text = self.extractor.extract_text(file_bytes, filename)
            logger.info("Step 1: Text extraction completed successfully.")
            
            # 2. Text Cleaning & Anchor Extraction
            cleaned_text = self.cleaner.clean_text(raw_text)
            anchors = self.cleaner.extract_anchors(cleaned_text)
            logger.info(f"Step 2: Cleaned text. Found contact anchors: {list(anchors.keys())}")
            
            # 3. Section Segmentation (Section-Aware parsing)
            segmented_sections = self.cleaner.segment_sections(cleaned_text)
            logger.info(f"Step 3: Segmented text into {len(segmented_sections)} sections.")
            
            # 4. LLM Entity Extraction (with dynamic fallback)
            entities = self.llm_client.extract_entities(cleaned_text, segmented_sections)
            logger.info("Step 4: LLM structured entities extracted.")
            
            # 5. Semantic Skill Normalization
            normalized_skills = self.normalizer.normalize_all(entities.skills)
            logger.info(f"Step 5: Normalized {len(entities.skills)} raw skills down to {len(normalized_skills)} canonical skills.")
            
            # 6. Explainable AI (Trace Mapping)
            explainability_traces = self.explainability.trace_entities(entities, cleaned_text)
            logger.info(f"Step 6: Created {len(explainability_traces)} entity trace links.")
            
            # 7. Confidence Evaluation & Pre-Fraud Assessment
            confidence_score, validation_report = self.explainability.evaluate_confidence(
                entities, anchors, explainability_traces
            )
            logger.info(f"Step 7: Evaluation complete. Confidence score: {confidence_score}")
            
            # 8. Store and Cache
            doc = ResumeDocument(
                filename=filename,
                uploaded_at=datetime.datetime.utcnow(),
                entities=entities,
                normalized_skills=normalized_skills,
                explainability_traces=explainability_traces,
                confidence_score=confidence_score,
                segmented_sections=segmented_sections,
                validation_report=validation_report
            )
            
            doc_id = await db_client.save_resume(doc)
            doc.id = doc_id
            logger.info(f"Step 8: Document stored successfully with ID: {doc_id}")
            
            return doc
            
        except ParserException as pe:
            logger.error(f"Pipeline failure for {filename}: {pe.message}")
            raise pe
        except Exception as e:
            logger.error(f"Unexpected error in pipeline: {str(e)}")
            raise ParserException(f"Pipeline processing failed: {str(e)}")
