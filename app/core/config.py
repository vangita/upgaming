from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    groq_api_key: SecretStr
    groq_model: str = "openai/gpt-oss-120b"
    # Answers when the main model's daily limit is used up; empty = no fallback
    groq_fallback_model: str | None = "openai/gpt-oss-20b"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "documents"

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("qdrant_api_key", "groq_fallback_model", mode="before")
    @classmethod
    def empty_to_none(cls, value: str | None) -> str | None:
        return value or None

    @property
    def main_collection(self) -> str:
        # The collection the app answers from and uploads go to: the structured chunking one
        return f"{self.qdrant_collection}_structured"


@lru_cache
def get_settings() -> Settings:
    return Settings()
