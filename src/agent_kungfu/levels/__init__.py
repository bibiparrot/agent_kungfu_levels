"""Importable package facade for the ten Agent Kung-fu level directories."""

from __future__ import annotations

from ..config import Settings
from ..contracts import LevelResult
from . import legacy
from .level1 import run as level_01
from .level2 import run as level_02
from .level3 import run as level_03
from .level4 import run as level_04
from .level5 import run as level_05
from .level6 import run as level_06
from .level7 import run as level_07
from .level8 import run as level_08
from .level9 import run as level_09
from .level10 import run as level_10


DEFAULT_QUESTION = "找出收入最高的前 5 个电影类别，并给出收入。"
PROMPT_DIR = legacy.PROMPT_DIR
WorkflowState = legacy.WorkflowState

LEVELS = {
    1: level_01,
    2: level_02,
    3: level_03,
    4: level_04,
    5: level_05,
    6: level_06,
    7: level_07,
    8: level_08,
    9: level_09,
    10: level_10,
}


def run_level(
    level: int,
    question: str = DEFAULT_QUESTION,
    *,
    settings: Settings | None = None,
    offline: bool = False,
    use_deepeval: bool = False,
) -> LevelResult:
    try:
        runner = LEVELS[level]
    except KeyError as exc:
        raise ValueError("level must be between 1 and 10") from exc
    if level in {6, 7, 8}:
        return runner(question, settings=settings, offline=offline, use_deepeval=use_deepeval)
    return runner(question, settings=settings, offline=offline)


__all__ = [
    "DEFAULT_QUESTION",
    "LEVELS",
    "PROMPT_DIR",
    "WorkflowState",
    "level_01",
    "level_02",
    "level_03",
    "level_04",
    "level_05",
    "level_06",
    "level_07",
    "level_08",
    "level_09",
    "level_10",
    "run_level",
]
