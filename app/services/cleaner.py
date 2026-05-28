import re
from typing import Dict, Any, List
from app.core.exceptions import CleanerException

class ResumeCleaner:
    """Service to handle cleaning raw text and segmenting it into logical resume sections."""

    # Common section header keywords (regex pattern dictionary)
    SECTION_KEYWORDS = {
        "contact_info": re.compile(
            r"\b(contact|personal info|about me|links|profiles)\b", re.IGNORECASE
        ),
        "summary": re.compile(
            r"\b(summary|objective|professional summary|about me|profile|overview)\b", re.IGNORECASE
        ),
        "education": re.compile(
            r"\b(education|academic background|academic qualifications|studies|credentials)\b", re.IGNORECASE
        ),
        "work_experience": re.compile(
            r"\b(work experience|professional experience|employment history|experience|career history|job history|work history)\b", re.IGNORECASE
        ),
        "skills": re.compile(
            r"\b(skills|technical skills|skills & expertise|core competencies|technologies|proficiencies|expertise)\b", re.IGNORECASE
        ),
        "projects": re.compile(
            r"\b(projects|academic projects|personal projects|key projects|featured projects)\b", re.IGNORECASE
        ),
        "certifications": re.compile(
            r"\b(certifications|licenses|courses|awards|accomplishments|achievements)\b", re.IGNORECASE
        ),
        "languages": re.compile(
            r"\b(languages|linguistic skills|languages spoken)\b", re.IGNORECASE
        )
    }

    @staticmethod
    def clean_text(text: str) -> str:
        """Removes garbage characters, normalizes whitespaces, but preserves essential structure."""
        if not text:
            return ""
        
        # Replace non-breaking spaces and tabs with standard space
        cleaned = text.replace("\xa0", " ").replace("\t", " ")
        
        # Clean up multi-byte or weird character representations
        cleaned = re.sub(r"[^\x00-\x7F]+", " ", cleaned)  # Replace non-ASCII characters with space
        
        # Normalize newline characters
        cleaned = re.sub(r"\r\n", "\n", cleaned)
        cleaned = re.sub(r"\r", "\n", cleaned)
        
        # Clean trailing white spaces on each line
        lines = [line.strip() for line in cleaned.split("\n")]
        
        # Keep non-empty lines, or only single separator empty lines
        cleaned_lines = []
        for i, line in enumerate(lines):
            if line:
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1] != "":
                # Keep a single empty line to preserve paragraph boundaries
                cleaned_lines.append("")
                
        cleaned_text = "\n".join(cleaned_lines)
        
        # Collapse multiple spaces (but not newlines)
        cleaned_text = re.sub(r"[ ]+", " ", cleaned_text)
        
        return cleaned_text.strip()

    @staticmethod
    def extract_anchors(text: str) -> Dict[str, Any]:
        """Extracts high-confidence email, phone, and links using deterministic regex.
        
        These anchors are used to validate and verify the LLM's outputs.
        """
        # Email Regex
        email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
        emails = re.findall(email_pattern, text)
        
        # Phone numbers: matches formats like +1-555-555-5555, (555) 555-5555, 555.555.5555
        phone_pattern = r"((?:\+\d{1,3}[\s.-]?)?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4})"
        phones = re.findall(phone_pattern, text)
        
        # Filter phone numbers that are too short/long or just numbers
        valid_phones = []
        for phone in phones:
            digits = re.sub(r"\D", "", phone)
            if 7 <= len(digits) <= 15:
                valid_phones.append(phone.strip())
                
        # Link anchors (GitHub, LinkedIn)
        github_pattern = r"(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9-_]+"
        linkedin_pattern = r"(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9-_]+"
        
        github_links = re.findall(github_pattern, text, re.IGNORECASE)
        linkedin_links = re.findall(linkedin_pattern, text, re.IGNORECASE)
        
        return {
            "emails": list(dict.fromkeys(emails)),
            "phones": list(dict.fromkeys(valid_phones)),
            "github_links": list(dict.fromkeys(github_links)),
            "linkedin_links": list(dict.fromkeys(linkedin_links))
        }

    def segment_sections(self, text: str) -> Dict[str, str]:
        """Segments raw text into structured sections by identifying section headers.
        
        Returns a dictionary of section names and their corresponding text.
        """
        try:
            lines = text.split("\n")
            sections: Dict[str, List[str]] = {key: [] for key in self.SECTION_KEYWORDS}
            sections["header_contact"] = []  # Default section for anything before the first header
            
            current_section = "header_contact"
            
            for line in lines:
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                
                # Check if the line matches any section keyword (must be a short line, typically < 6 words)
                is_header = False
                words = stripped_line.split()
                if len(words) <= 6:
                    for section_name, pattern in self.SECTION_KEYWORDS.items():
                        # The line matches the header pattern
                        if pattern.search(stripped_line):
                            current_section = section_name
                            is_header = True
                            break
                
                if not is_header:
                    sections[current_section].append(line)
                    
            # Join list lines into string segments
            return {key: "\n".join(val).strip() for key, val in sections.items() if val}
        except Exception as e:
            raise CleanerException(f"Failed to segment resume sections: {str(e)}")
