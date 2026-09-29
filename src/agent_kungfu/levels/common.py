from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config import PROJECT_ROOT, Settings


LEVELS_DIR = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one level configuration and require a mapping at the root."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def load_level_config(level: int) -> dict[str, Any]:
    if level not in range(1, 11):
        raise ValueError("level must be between 1 and 10")
    return load_yaml(LEVELS_DIR / f"level{level}" / "config.yaml")


def resolve_from(path_value: str, *, config_path: Path) -> Path:
    """Resolve a path relative to the YAML file that declares it."""

    path = Path(path_value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


@dataclass(frozen=True, slots=True)
class BaselineAssets:
    config_path: Path
    database_path: Path
    database_uri: str
    sql_path: Path
    sql: str
    config: dict[str, Any]


def load_baseline_assets(settings: Settings | None = None) -> BaselineAssets:
    config_path = LEVELS_DIR / "baseline" / "baseline.yaml"
    config = load_yaml(config_path)
    configured_db = resolve_from(str(config["database"]["path"]), config_path=config_path)
    database_path = (settings.db_path if settings is not None else configured_db).resolve()
    sql_path = resolve_from(str(config["sql"]["revenue"]), config_path=config_path)
    if not database_path.is_file():
        raise FileNotFoundError(f"Sakila database not found: {database_path}")
    if not sql_path.is_file():
        raise FileNotFoundError(f"Baseline SQL not found: {sql_path}")
    return BaselineAssets(
        config_path=config_path,
        database_path=database_path,
        database_uri=f"sqlite:///{database_path.as_posix()}",
        sql_path=sql_path,
        sql=sql_path.read_text(encoding="utf-8").strip(),
        config=config,
    )


def default_settings_from_baseline() -> Settings:
    assets = load_baseline_assets()
    defaults = Settings.from_env()
    return Settings(
        db_path=assets.database_path,
        ollama_base_url=defaults.ollama_base_url,
        ollama_model=defaults.ollama_model,
        ollama_api_key=defaults.ollama_api_key,
        output_dir=defaults.output_dir,
        omnigent_url=defaults.omnigent_url,
        omnigent_runner_id=defaults.omnigent_runner_id,
    )


def resolved_settings(settings: Settings | None = None) -> Settings:
    resolved = settings or default_settings_from_baseline()
    resolved.validate()
    return resolved


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
