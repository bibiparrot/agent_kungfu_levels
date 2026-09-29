from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings shared by every level.

    Ollama exposes an OpenAI-compatible endpoint at ``/v1``. LiteLLM receives
    the raw model name plus an explicit ``custom_llm_provider='openai'`` so the
    provider prefix is never forwarded to Ollama as part of the model name.
    """

    db_path: Path = PROJECT_ROOT / "data" / "sakila.db"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "ornith-1.5:9b"
    ollama_api_key: str = "ollama"
    output_dir: Path = PROJECT_ROOT / "outputs"
    omnigent_url: str | None = None
    omnigent_runner_id: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_path=Path(os.getenv("SAKILA_DB_PATH", PROJECT_ROOT / "data" / "sakila.db")),
            ollama_base_url=os.getenv("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1"),
            ollama_model=os.getenv("AGENT_LLM_MODEL", "ornith-1.5:9b"),
            ollama_api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            output_dir=Path(os.getenv("AGENT_OUTPUT_DIR", PROJECT_ROOT / "outputs")),
            omnigent_url=os.getenv("OMNIGENT_URL"),
            omnigent_runner_id=os.getenv("OMNIGENT_RUNNER_ID"),
        )

    @property
    def sqlite_uri(self) -> str:
        return f"sqlite:///{self.db_path.resolve().as_posix()}"

    @property
    def litellm_model(self) -> str:
        return self.ollama_model

    def validate(self) -> None:
        if not self.db_path.is_file():
            raise FileNotFoundError(f"Sakila database not found: {self.db_path}")
