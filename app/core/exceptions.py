class ParserException(Exception):
    """Base exception class for the Resume Parser pipeline."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

class ExtractionException(ParserException):
    """Exception raised during file reading or text extraction (PDF/DOCX)."""
    pass

class CleanerException(ParserException):
    """Exception raised during text preprocessing or segmentation."""
    pass

class LLMException(ParserException):
    """Exception raised during LLM structure parsing or response generation."""
    pass

class NormalizationException(ParserException):
    """Exception raised during skill normalization or taxonomy mapping."""
    pass

class DatabaseException(ParserException):
    """Exception raised during database CRUD operations."""
    pass
