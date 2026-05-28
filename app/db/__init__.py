from app.core.config import settings
from app.db.base import BaseDatabase
from app.db.local_db import LocalFileDatabase
from app.db.mongodb import MongoDBDatabase

def get_db_client() -> BaseDatabase:
    """Exposes the active database client based on the environment configuration."""
    if settings.DB_TYPE.lower() == "mongodb":
        return MongoDBDatabase()
    return LocalFileDatabase()

# Singleton DB Client instance
db_client = get_db_client()
