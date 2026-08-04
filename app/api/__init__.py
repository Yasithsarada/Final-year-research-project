from fastapi import APIRouter
from app.api.endpoints.parser import router as parser_router
from app.api.endpoints.health import router as health_router
from app.api.endpoints.ccs import router as ccs_router
from app.api.endpoints.evaluation import router as evaluation_router
from app.api.endpoints.msr import router as msr_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(parser_router)
api_router.include_router(health_router)
api_router.include_router(ccs_router)
api_router.include_router(evaluation_router)
api_router.include_router(msr_router)
