"""Centralized Configuration loader for Project 3 Agentic Assistant."""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


@dataclass
class AppConfig:
    """Application Configuration Settings for Project 3."""

    api_key: str
    llm_model_name: str
    embedding_model_name: str
    project_1_dir: Path
    project_2_dir: Path
    log_level: str
    base_dir: Path

    @property
    def openai_api_key(self) -> str:
        return self.api_key

    @property
    def openai_base_url(self) -> str:
        return "https://generativelanguage.googleapis.com/v1beta/openai/"

    @property
    def chroma_db_dir(self) -> Path:
        return self.project_1_dir / "chroma_db"

    @classmethod
    def load(cls) -> "AppConfig":
        """Loads configuration from environment variables with sensible defaults."""
        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("OPENAI_API_KEY", "")
        )

        llm_model = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
        embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-001")

        p1_raw = os.getenv("PROJECT_1_DIR", "../project-1-rag")
        p2_raw = os.getenv("PROJECT_2_DIR", "../project-2-mcp")

        project_1_dir = (BASE_DIR / p1_raw).resolve() if not Path(p1_raw).is_absolute() else Path(p1_raw)
        project_2_dir = (BASE_DIR / p2_raw).resolve() if not Path(p2_raw).is_absolute() else Path(p2_raw)

        log_level = os.getenv("LOG_LEVEL", "INFO")

        return cls(
            api_key=api_key,
            llm_model_name=llm_model,
            embedding_model_name=embedding_model,
            project_1_dir=project_1_dir,
            project_2_dir=project_2_dir,
            log_level=log_level,
            base_dir=BASE_DIR,
        )

    def validate_api_key(self) -> None:
        """Validates if API key is set properly."""
        invalid_placeholders = {
            "",
            "your_openai_api_key_here",
            "your_gemini_api_key_here",
            "your_api_key_here",
        }
        if not self.api_key or self.api_key in invalid_placeholders:
            raise ValueError(
                "API key is not set in project-3-agent/.env. "
                "Please configure GEMINI_API_KEY in your .env file."
            )


config = AppConfig.load()
