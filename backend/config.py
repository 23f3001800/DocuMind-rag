# from pydantic_settings import BaseSettings
# from pathlib import Path


# class Settings(BaseSettings):
#     gemini_api_key: str
#     embedding_model: str = "all-MiniLM-L6-v2"
#     reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#     chroma_path: str = "./chroma_db"
#     top_k_retrieve: int = 10
#     top_k_rerank: int = 3
#     chunk_size: int = 512
#     chunk_overlap: int = 64

#     class Config:
#         env_file = ".env"


# settings = Settings()


from dataclasses import dataclass, field
from typing import Optional
import os

@dataclass
class LLMConfig:
    model: str = "gemini-pro"
    max_tokens: int = 1000
    temperature: float = 0.7
    api_key: str = field(default_factory=lambda: os.environ["GEMINI_API_KEY"])
    timeout_seconds: int = 30

    def __post_init__(self):
        if not 0 <= self.temperature <= 2:
            raise ValueError(f"Temperature {self.temperature} out of range [0, 2]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")

# Usage — no magic strings, IDE autocomplete works, validation on creation
config = LLMConfig(temperature=0.9)
print(config)  # LLMConfig(model='gemini-pro', max_tokens=1000, ...)