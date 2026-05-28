from fastapi import APIRouter
from app.api.endpoints.parser import router as parser_router
from app.api.endpoints.health import router as health_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(parser_router)
api_router.include_router(health_router)
