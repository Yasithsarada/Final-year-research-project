import logging
import re
import spacy
from typing import List, Dict
from app.core.config import settings

logger = logging.getLogger(__name__)

# Compile regex for filler-word removal
FILLER_PATTERN = re.compile(
    r"\b(uh|um|you know|like|so|basically|actually|literally|you see|sort of|kind of|anyway)\b",
    re.IGNORECASE
)

# Compile regex for excessive white space and pauses
PUNCTUATION_PAUSES_PATTERN = re.compile(r"(\s*,\s*|\s*\.\s*)+")

class TranscriptPostProcessor:
    """Handles text preprocessing, sentence segmentation, and section categorization of transcripts."""

    def __init__(self):
        try:
            self.nlp = spacy.load(settings.SPACY_MODEL)
        except Exception as e:
            logger.error(f"Failed to load spaCy model '{settings.SPACY_MODEL}': {str(e)}")
            # Fallback simple sentencizer
            self.nlp = None

    def clean_text(self, text: str) -> str:
        """Removes filler words and cleans whitespace, preserving basic semantic structure."""
        if not text:
            return ""

        # Remove filler words
        cleaned = FILLER_PATTERN.sub("", text)
        
        # Clean double punctuation or pauses (e.g. "...," or ".. .")
        cleaned = PUNCTUATION_PAUSES_PATTERN.sub(". ", cleaned)

        # Normalize spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    def segment_sentences(self, text: str) -> List[str]:
        """Segments raw/cleaned transcript text into formal sentences using spaCy."""
        if not text:
            return []

        if self.nlp:
            try:
                doc = self.nlp(text)
                return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            except Exception as e:
                logger.error(f"spaCy sentence segmentation failed: {str(e)}")
                
        # Simple fallback sentencizer if spaCy fails
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def segment_by_topic_sections(self, text: str) -> Dict[str, str]:
        """Section-aware transcript parser: splits transcripts into potential topic regions.
        
        Uses semantic anchors/keywords frequently found in technical interviews.
        """
        sections = {
            "background": [],
            "experience": [],
            "projects": [],
            "technical": [],
            "leadership": [],
            "general": []
        }

        sentences = self.segment_sentences(text)
        current_section = "general"

        # Keywords that shift the topic section
        section_triggers = {
            "background": ["tell me about yourself", "my background", "introduce yourself", "started coding"],
            "experience": ["years of experience", "past roles", "worked at", "employment history", "previous job"],
            "projects": ["personal projects", "side projects", "built an app", "github repository", "developed a project"],
            "technical": ["kubernetes", "docker", "python", "aws", "architecture", "microservices", "database", "ci/cd"],
            "leadership": ["managed", "led 3", "led a team", "mentored", "scrum master", "product manager", "leadership"]
        }

        for sentence in sentences:
            sent_lower = sentence.lower()
            
            # Check if this sentence triggers a section transition
            transition_found = False
            for sec, keywords in section_triggers.items():
                if any(kw in sent_lower for kw in keywords):
                    current_section = sec
                    transition_found = True
                    break
                    
            sections[current_section].append(sentence)

        # Join the list of sentences into strings
        return {k: " ".join(v) for k, v in sections.items() if v}
