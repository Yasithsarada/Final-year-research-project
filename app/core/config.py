import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # API Configuration
    OPENAI_API_KEY: Optional[str] = ""
    GEMINI_API_KEY: Optional[str] = ""

    # Database Configuration
    DB_TYPE: str = "local"  # 'mongodb' or 'local'
    MONGODB_URL: str = "mongodb://localhost:27017"
    DB_NAME: str = "recruitment_fraud_db"

    # ML & NLP Configuration
    SKILL_SIMILARITY_THRESHOLD: float = 0.75
    SPACY_MODEL: str = "en_core_web_sm"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # API Settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True

    # Storage Paths
    LOCAL_STORAGE_DIR: Path = BASE_DIR / "app_data"
    TAXONOMY_PATH: Path = BASE_DIR / "data" / "taxonomy.json"

    # Pydantic settings configuration
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def model_post_init(self, __context) -> None:
        """Ensure storage and data directories exist."""
        self.LOCAL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.TAXONOMY_PATH.parent.mkdir(parents=True, exist_ok=True)

# Instantiate settings
settings = Settings()
