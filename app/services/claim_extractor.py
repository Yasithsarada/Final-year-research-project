import json
import logging
from typing import Optional, Dict, Any
from openai import OpenAI
from google import genai
from google.genai import types as genai_types

from app.core.config import settings
from app.core.exceptions import LLMException
from app.models.ccs import CCSClaimSet
from app.services.llm_client import get_gemini_compatible_schema

logger = logging.getLogger(__name__)

class LLMClaimExtractorClient:
    """LLM client for extracting self-reported claims from transcripts into structured JSON.
    
    Guarantees strict adherence to CCSClaimSet using OpenAI/Gemini Structured Outputs.
    """

    def __init__(self):
        self.openai_key = settings.OPENAI_API_KEY
        self.gemini_key = settings.GEMINI_API_KEY
        
        self._openai_client: Optional[OpenAI] = None
        if self.openai_key:
            self._openai_client = OpenAI(api_key=self.openai_key)
            
        self._gemini_client = None
        if self.gemini_key:
            self._gemini_client = genai.Client(api_key=self.gemini_key)

    def _build_prompt(self, cleaned_transcript: str, segmented_sections: Dict[str, str]) -> str:
        """Constructs a research-grade prompt optimized for self-reported claim extraction."""
        sections_dump = json.dumps(segmented_sections, indent=2)
        
        prompt = f"""
You are a research-grade Candidate Claim Extraction AI. Your task is to analyze the candidate's verbal transcript and extract all self-reported professional claims into the requested JSON schema structure.

Below is the candidate's transcript pre-segmented into topic-aware sections (use this context to align claims):
{sections_dump}

Here is the full cleaned transcript of the candidate's interview:
--- START TRANSCRIPT ---
{cleaned_transcript}
--- END TRANSCRIPT ---

Instructions for Claim Extraction:
1. **Claims Definition**: A claim is any statement where the candidate asserts their expertise, experience, tools, roles, responsibilities, or leadership history (e.g. "I managed a team", "I have 5 years of experience", "I worked with Kubernetes").
2. **Entity Lists**: Extract skills, languages, tools, frameworks, databases, and cloud platforms strictly mentioned by the candidate during the interview. Keep them as canonical, concise nouns.
3. **Years of Experience**: Extract any verbal claim indicating time length (e.g., "5 years of experience in Java", "using Docker for 3 years").
4. **Leadership Claims**: Extract specific leadership and management claims (e.g. "I led a team of 6 engineers", "I was scrum master for 2 years").
5. **Projects Claimed**: Document any explicit projects the candidate mentions having worked on or built.
6. **Supporting Evidence**: Crucially, identify the EXACT literal sentences from the transcript that contain or justify each claim, and include them in `supporting_sentences`.
7. **Confidence Score**: Provide a score between 0.0 and 1.0 reflecting:
   - The candidate's verbal confidence/assertion level (e.g., "I know a bit of Docker" -> lower score vs "I led the Docker migration" -> high score).
   - The ambiguity or vagueness of the claim statement in the context of the transcript.

Rules:
- DO NOT invent, assume, or extrapolate. If the candidate does not mention a skill or topic, DO NOT extract it.
- Only extract sentences from the transcript that were actually spoken.
"""
        return prompt

    def _parse_with_openai(self, prompt: str) -> CCSClaimSet:
        """Call OpenAI Structured Outputs API to extract claims."""
        if not self._openai_client:
            raise LLMException("OpenAI API client is not configured (missing key).")
            
        try:
            logger.info("Sending claim extraction request to OpenAI (gpt-4o-mini)")
            response = self._openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise professional claim extractor."},
                    {"role": "user", "content": prompt}
                ],
                response_format=CCSClaimSet,
                temperature=0.1
            )
            parsed_data = response.choices[0].message.parsed
            if not parsed_data:
                raise LLMException("OpenAI failed to parse claims into schema.")
            return parsed_data
        except Exception as e:
            logger.error(f"OpenAI claim parsing failed: {str(e)}")
            raise LLMException(f"OpenAI claim parse failure: {str(e)}")

    def _parse_with_gemini(self, prompt: str) -> CCSClaimSet:
        """Call Google Gemini Structured Outputs API (google-genai SDK) to extract claims."""
        if not self._gemini_client:
            raise LLMException("Gemini API is not configured (missing key).")
            
        try:
            logger.info("Sending claim extraction request to Gemini (gemini-2.5-flash)")
            compatible_schema = get_gemini_compatible_schema(CCSClaimSet)
            
            response = self._gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=compatible_schema,
                    temperature=0.1
                )
            )
            
            text_response = response.text
            if not text_response:
                raise LLMException("Gemini returned an empty response.")
                
            data_dict = json.loads(text_response)
            return CCSClaimSet.model_validate(data_dict)
        except Exception as e:
            logger.error(f"Gemini claim parsing failed: {str(e)}")
            raise LLMException(f"Gemini claim parse failure: {str(e)}")

    def extract_claims(self, cleaned_transcript: str, segmented_sections: Dict[str, str]) -> CCSClaimSet:
        """Extracts claims from a transcript using LLM structured output and failover strategy."""
        prompt = self._build_prompt(cleaned_transcript, segmented_sections)
        
        errors = []
        
        if self._openai_client:
            try:
                return self._parse_with_openai(prompt)
            except Exception as e:
                errors.append(f"OpenAI error: {str(e)}")
                logger.warning("OpenAI claim extraction failed, attempting failover to Gemini...")
                
        if self.gemini_key:
            try:
                return self._parse_with_gemini(prompt)
            except Exception as e:
                errors.append(f"Gemini error: {str(e)}")
                logger.error("Gemini claim extraction failed as well.")
                
        if not errors:
            raise LLMException("No LLM providers are configured for CCS claim extraction.")
        else:
            raise LLMException(f"LLM Claim Extraction pipeline failed: {'; '.join(errors)}")
