import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# MPS allocator limits: must be set before torch is imported anywhere.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.1")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "1.0")  # must be <= high
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class ModelConfig(BaseModel):
    REPO: str = "briaai/Fibo-2.0-edit-turbo"
    MODEL_DIR: Path = Path("./models")
    DEVICE: Literal["mps", "cuda", "cpu"] = "mps"
    QUANT: Literal["int8", "none"] = "int8"
    AUTO_COMPILE: bool = False
    BUCKET_BASE: int = 1024
    WARMUP_ASPECTS: list[str] = ["1:1"]
    MAX_OUTPUT_SIDE: int = 2048
    MAX_STRETCH: float = 0.15  # warn above 15% aspect distortion (panoramas etc.)
    WARMUP_STEPS: int = 1


class LLMConfig(BaseModel):
    VGL_LLM: str = "gemini/gemini-3.5-flash-lite"


class PathsConfig(BaseModel):
    OUTPUT_DIR: Path = Path("./static")


class Settings(ModelConfig, LLMConfig, PathsConfig, BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

    GEMINI_API_KEY: SecretStr | None = Field(None, validation_alias="GEMINI_API_KEY")
    LOG_DIR: str = "./logs"

    @property
    def use_llm(self) -> bool:
        return self.GEMINI_API_KEY is not None and bool(
            self.GEMINI_API_KEY.get_secret_value().strip()
        )

    def export_env(self) -> None:
        """litellm (inside bria-vgl) reads the key from os.environ, not from this object."""
        key = self.GEMINI_API_KEY
        if key is not None and key.get_secret_value().strip():
            os.environ.setdefault("GEMINI_API_KEY", key.get_secret_value())


SETTINGS = Settings()
SETTINGS.export_env()
