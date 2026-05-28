import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import spacy

from app.core.config import settings
from app.db import db_client
from app.api import api_router
from app.core.exceptions import (
    ParserException,
    ExtractionException,
    CleanerException,
    LLMException,
    NormalizationException,
    DatabaseException
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def ensure_spacy_model():
    """Checks if the required spaCy model is installed, and downloads it if missing."""
    model_name = settings.SPACY_MODEL
    try:
        spacy.load(model_name)
        logger.info(f"spaCy model '{model_name}' is already installed and loaded.")
    except OSError:
        logger.info(f"spaCy model '{model_name}' not found. Downloading...")
        try:
            spacy.cli.download(model_name)
            logger.info(f"spaCy model '{model_name}' downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download spaCy model '{model_name}': {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager handling app setup and cleanup hooks."""
    logger.info("Initializing Resume Parser services...")
    
    # 1. Ensure spaCy model is present
    ensure_spacy_model()
    
    # 2. Connect Database
    await db_client.connect()
    
    yield
    
    # 3. Clean up Database connections
    await db_client.disconnect()
    logger.info("Resume Parser services shut down successfully.")

app = FastAPI(
    title="AI-Powered Resume Parser & Entity Extractor API",
    description="Research-grade backend module for profile extraction, skill normalization, and explainable AI verification.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend/cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount central API routing
app.include_router(api_router)


# ==========================================
# Global Exception Handlers
# ==========================================

@app.exception_handler(ExtractionException)
async def extraction_exception_handler(request: Request, exc: ExtractionException):
    logger.error(f"Extraction error: {exc.message}")
    return JSONResponse(
        status_code=400,
        content={"error": "ExtractionError", "message": exc.message, "details": exc.details}
    )

@app.exception_handler(CleanerException)
async def cleaner_exception_handler(request: Request, exc: CleanerException):
    logger.error(f"Text cleaning error: {exc.message}")
    return JSONResponse(
        status_code=422,
        content={"error": "PreprocessingError", "message": exc.message, "details": exc.details}
    )

@app.exception_handler(LLMException)
async def llm_exception_handler(request: Request, exc: LLMException):
    logger.error(f"LLM parsing error: {exc.message}")
    return JSONResponse(
        status_code=502,
        content={"error": "LLMExtractionError", "message": exc.message, "details": exc.details}
    )

@app.exception_handler(NormalizationException)
async def normalization_exception_handler(request: Request, exc: NormalizationException):
    logger.error(f"Normalization error: {exc.message}")
    return JSONResponse(
        status_code=422,
        content={"error": "NormalizationError", "message": exc.message, "details": exc.details}
    )

@app.exception_handler(DatabaseException)
async def database_exception_handler(request: Request, exc: DatabaseException):
    logger.error(f"Database error: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={"error": "DatabaseError", "message": exc.message, "details": exc.details}
    )

@app.exception_handler(ParserException)
async def parser_exception_handler(request: Request, exc: ParserException):
    logger.error(f"Pipeline error: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={"error": "PipelineError", "message": exc.message, "details": exc.details}
    )

# Root Index Redirect
@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "AI Resume Parser & Entity Extractor API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/api/v1/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
