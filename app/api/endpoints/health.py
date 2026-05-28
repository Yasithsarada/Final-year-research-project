from fastapi import APIRouter
from typing import Dict, Any

from app.core.config import settings
from app.db import db_client
from app.services.normalizer import HAS_SENTENCE_TRANSFORMERS
import spacy

router = APIRouter(prefix="/health", tags=["Health Checks"])

@router.get("", response_model=Dict[str, Any], summary="Get System Health")
async def health_check():
    """Checks overall application, database, and NLP model statuses."""
    health_status = {
        "status": "healthy",
        "database": {
            "type": settings.DB_TYPE,
            "status": "disconnected"
        },
        "nlp_services": {
            "spacy": "not_loaded",
            "sentence_transformers": "not_loaded"
        }
    }

    # Check Database Connection
    try:
        if settings.DB_TYPE.lower() == "mongodb":
            if db_client.client:
                await db_client.client.admin.command('ping')
                health_status["database"]["status"] = "connected"
            else:
                health_status["database"]["status"] = "uninitialized"
                health_status["status"] = "degraded"
        else:
            await db_client.connect() # Verify local directory exists
            health_status["database"]["status"] = "connected"
    except Exception as e:
        health_status["database"]["status"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    # Check spaCy load status
    try:
        if spacy.util.is_package(settings.SPACY_MODEL):
            health_status["nlp_services"]["spacy"] = f"loaded ({settings.SPACY_MODEL})"
        else:
            health_status["nlp_services"]["spacy"] = f"missing ({settings.SPACY_MODEL})"
            health_status["status"] = "degraded"
    except Exception:
        health_status["nlp_services"]["spacy"] = "error"
        health_status["status"] = "degraded"

    # Check Sentence Transformers availability
    if HAS_SENTENCE_TRANSFORMERS:
        health_status["nlp_services"]["sentence_transformers"] = f"installed ({settings.EMBEDDING_MODEL})"
    else:
        health_status["nlp_services"]["sentence_transformers"] = "missing_library"
        health_status["status"] = "degraded"

    return health_status
