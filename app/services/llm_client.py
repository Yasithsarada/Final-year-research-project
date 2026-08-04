import json
import logging
from typing import Optional, Dict, Any
from openai import OpenAI
from google import genai
from google.genai import types as genai_types

from app.core.config import settings
from app.core.exceptions import LLMException
from app.models.resume import ExtractedResumeSchema

logger = logging.getLogger(__name__)

def get_gemini_compatible_schema(model_class) -> dict:
    """Generates a JSON schema from a Pydantic model class and makes it Gemini-compatible.
    
    1. Inlines/dereferences all $ref pointers.
    2. Simplifies 'anyOf' and list types to single, concrete type definitions.
    3. Removes unsupported keywords like 'default' and 'title'.
    """
    raw_schema = model_class.model_json_schema()
    defs = raw_schema.get("$defs", {})
    
    def resolve_refs(schema_part: Any) -> Any:
        if isinstance(schema_part, list):
            return [resolve_refs(item) for item in schema_part]
        elif isinstance(schema_part, dict):
            # Resolve $ref pointer
            if "$ref" in schema_part:
                ref_path = schema_part["$ref"]
                def_name = ref_path.split("/")[-1]
                if def_name in defs:
                    return resolve_refs(defs[def_name])
            
            # Simplify anyOf (Gemini prefers concrete single types over anyOf unions)
            if "anyOf" in schema_part:
                any_of_list = schema_part["anyOf"]
                non_null_schemas = [s for s in any_of_list if isinstance(s, dict) and s.get("type") != "null"]
                if non_null_schemas:
                    first_schema = resolve_refs(non_null_schemas[0])
                    # Keep metadata fields from parent part
                    cleaned = {k: v for k, v in schema_part.items() if k not in ("anyOf", "default", "title")}
                    cleaned.update(first_schema)
                    return cleaned
            
            # Normalize list types (e.g. type: ["string", "null"])
            if "type" in schema_part and isinstance(schema_part["type"], list):
                non_null_types = [t for t in schema_part["type"] if t != "null"]
                if non_null_types:
                    schema_part["type"] = non_null_types[0]
            
            cleaned = {}
            for k, v in schema_part.items():
                if k in ("default", "title", "$defs"):
                    continue
                cleaned[k] = resolve_refs(v)
            return cleaned
        return schema_part

    return resolve_refs(raw_schema)

class LLMResumeParserClient:
    """LLM client for parsing raw resume text into structured JSON.
    
    Guarantees strict adherence to ExtractedResumeSchema using OpenAI Structured Outputs
    and Gemini Structured Schema capabilities.
    """

    def __init__(self):
        self.openai_key = settings.OPENAI_API_KEY
        self.gemini_key = settings.GEMINI_API_KEY
        
        # Initialize clients lazily if keys are available
        self._openai_client: Optional[OpenAI] = None
        if self.openai_key:
            self._openai_client = OpenAI(api_key=self.openai_key)
            
        self._gemini_client = None
        if self.gemini_key:
            self._gemini_client = genai.Client(api_key=self.gemini_key)

    def _build_prompt(self, cleaned_text: str, segmented_sections: Dict[str, str]) -> str:
        """Constructs an optimized prompt for structured entity extraction."""
        sections_dump = json.dumps(segmented_sections, indent=2)
        
        prompt = f"""
You are a research-grade Resume Parsing AI. Your task is to analyze the following resume text and extract all required entities into the requested JSON schema structure.

Below are the pre-segmented sections identified by a rule-based parser:
{sections_dump}

Here is the full text of the resume:
--- START RESUME TEXT ---
{cleaned_text}
--- END RESUME TEXT ---

Instructions for Extraction:
1. **Name**: Look for the name at the beginning of the resume or contact section.
2. **Contact (Email, Phone)**: Extract email and phone. Do not include spaces or parentheses in phone numbers unless they are part of the country code.
3. **Links**: Extract GitHub and LinkedIn profile URLs. Look for usernames if full links aren't written (e.g. github.com/username).
4. **Skills**: Extract all distinct tech skills, frameworks, tools, and languages (e.g. Python, Docker, React, Git, AWS). Keep them as concise nouns.
5. **Work Experience**: Extract company, role, start/end dates, a descriptions of tasks, and the specific technologies/languages used in that role.
6. **Projects**: Identify distinct personal or academic projects. Extract description, technology tags, and links if available.
7. **Education**: Capture school, degree, major, start/end dates, and GPA if listed.
8. **Languages**: Capture spoken languages and proficiency levels.
9. **Total Experience**: Sum up non-overlapping professional employment spans and express this logically (e.g., "3.5 years" or "7 years").
10. **Claimed Role**: Infer the main job role/title the candidate is targeting or qualifies for based on experience (e.g., "DevOps Engineer", "Frontend Developer", "Machine Learning Researcher").

Crucial Extraction Rules:
- DO NOT invent, hallucinate, or extrapolate facts. If a field is not mentioned, leave it empty or null.
- Ensure all dates are parsed as standard readable formats (e.g., "MM/YYYY", "Year", or "Present").
- If the resume text is malformed or contains garbage, try your best to extract readable pieces.
"""
        return prompt

    def _parse_with_openai(self, prompt: str) -> ExtractedResumeSchema:
        """Call OpenAI Structured Outputs API."""
        if not self._openai_client:
            raise LLMException("OpenAI API client is not configured (missing key).")
            
        try:
            logger.info("Sending request to OpenAI (gpt-4o-mini)")
            # Using OpenAI beta parse API for guaranteed JSON schema conformance
            response = self._openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise data extractor."},
                    {"role": "user", "content": prompt}
                ],
                response_format=ExtractedResumeSchema,
                temperature=0.1
            )
            parsed_data = response.choices[0].message.parsed
            if not parsed_data:
                raise LLMException("OpenAI failed to parse response into schema.")
            return parsed_data
        except Exception as e:
            logger.error(f"OpenAI parsing failed: {str(e)}")
            raise LLMException(f"OpenAI parse failure: {str(e)}")

    def _parse_with_gemini(self, prompt: str) -> ExtractedResumeSchema:
        """Call Google Gemini Structured Outputs API (google-genai SDK)."""
        if not self._gemini_client:
            raise LLMException("Gemini API is not configured (missing key).")
            
        try:
            logger.info("Sending request to Gemini (gemini-2.5-flash)")
            
            # Generate Gemini-compatible schema by dereferencing & cleaning
            compatible_schema = get_gemini_compatible_schema(ExtractedResumeSchema)
            
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
                
            # Parse the JSON string into the Pydantic schema
            data_dict = json.loads(text_response)
            return ExtractedResumeSchema.model_validate(data_dict)
        except Exception as e:
            logger.error(f"Gemini parsing failed: {str(e)}")
            raise LLMException(f"Gemini parse failure: {str(e)}")

    def extract_entities(self, cleaned_text: str, segmented_sections: Dict[str, str]) -> ExtractedResumeSchema:
        """Extracts resume entities using LLM structured generation, handling failover."""
        prompt = self._build_prompt(cleaned_text, segmented_sections)
        
        # Failover Strategy:
        # 1. If OpenAI key is available, try OpenAI first.
        # 2. If OpenAI fails or is not available, try Gemini if key is available.
        # 3. If both fail/unavailable, raise LLMException.
        
        errors = []
        
        if self._openai_client:
            try:
                return self._parse_with_openai(prompt)
            except Exception as e:
                errors.append(f"OpenAI error: {str(e)}")
                logger.warning("OpenAI extraction failed, attempting failover to Gemini if configured...")
                
        if self.gemini_key:
            try:
                return self._parse_with_gemini(prompt)
            except Exception as e:
                errors.append(f"Gemini error: {str(e)}")
                logger.error("Gemini extraction failed as well.")
                
        # If we reach here, extraction has failed on all configured channels
        if not errors:
            raise LLMException("No LLM providers are configured. Please set OPENAI_API_KEY or GEMINI_API_KEY in .env.")
        else:
            raise LLMException(f"LLM Entity Extraction pipeline failed: {'; '.join(errors)}")
