"""Configuration loader for Project 2 (LangGraph + MCP Server)."""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


@dataclass
class AppConfig:
    """Settings data class for environment variables and model parameters."""

    openai_api_key: str
    openai_base_url: str
    llm_model_name: str
    log_level: str
    base_dir: Path

    @classmethod
    def load(cls) -> "AppConfig":
        """Loads configuration settings from environment variables."""
        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("OPENAI_API_KEY", "")
        )

        is_gemini = (
            bool(os.getenv("GEMINI_API_KEY"))
            or bool(os.getenv("GOOGLE_API_KEY"))
            or api_key.startswith("AIza")
            or "generativelanguage.googleapis.com" in os.getenv("OPENAI_BASE_URL", "")
            or "gemini" in os.getenv("LLM_MODEL_NAME", "").lower()
        )

        if is_gemini:
            default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            default_llm_model = "gemini-1.5-flash"
        else:
            default_base_url = "https://api.openai.com/v1"
            default_llm_model = "gpt-4o-mini"

        base_url = os.getenv("OPENAI_BASE_URL", default_base_url)
        llm_model = os.getenv("LLM_MODEL_NAME", default_llm_model)
        log_level = os.getenv("LOG_LEVEL", "INFO")

        return cls(
            openai_api_key=api_key,
            openai_base_url=base_url,
            llm_model_name=llm_model,
            log_level=log_level,
            base_dir=BASE_DIR,
        )

    def validate_api_key(self) -> None:
        """Validates API key presence."""
        invalid_placeholders = {
            "",
            "your_openai_api_key_here",
            "your_gemini_api_key_here",
            "your_api_key_here",
        }
        if not self.openai_api_key or self.openai_api_key in invalid_placeholders:
            raise ValueError(
                "API key is missing or invalid in project-2-mcp/.env. "
                "Please configure GEMINI_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY."
            )


config = AppConfig.load()
