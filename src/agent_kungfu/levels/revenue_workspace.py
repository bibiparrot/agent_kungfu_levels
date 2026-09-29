from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from ..config import Settings
from ..database import SakilaDB
from .common import load_level_config
from .level1.workflow import calculate_revenue, merge_tables, read_tables, top_n_categories


@dataclass(slots=True)
class RevenueAnalysisWorkspace:
    """Neutral shared state used by both the Agents SDK and Codex SDK lessons."""

    db: SakilaDB
    queries: dict[str, str]
    minimum_n: int = 1
    maximum_n: int = 16
    tables: dict[str, pl.DataFrame] = field(default_factory=dict)
    merged: pl.DataFrame | None = None
    revenue: pl.DataFrame | None = None
    selected: pl.DataFrame | None = None
    events: list[str] = field(default_factory=list)

    def load_tables(self) -> str:
        if self.tables:
            raise RuntimeError("The table-read skill may run only once")
        self.tables = read_tables(self.db, self.queries)
        self.events.append("read_tables")
        return json.dumps(
            {
                "status": "tables_loaded",
                "rows": {name: frame.height for name, frame in self.tables.items()},
            },
            ensure_ascii=False,
        )

    def calculate(self, n: int) -> str:
        if not self.tables:
            raise RuntimeError("Read the five tables before calculating revenue")
        if self.revenue is not None:
            raise RuntimeError("The calculate skill may run only once")
        self.merged = merge_tables(self.tables)
        self.revenue = calculate_revenue(self.merged)
        self.selected = top_n_categories(
            self.revenue,
            n,
            minimum=self.minimum_n,
            maximum=self.maximum_n,
        )
        self.events.append("calculate_revenue")
        return json.dumps(
            {
                "status": "revenue_calculated",
                "joined_rows": self.merged.height,
                "category_rows": self.revenue.height,
                "top_n": n,
                "rows": self.selected.to_dicts(),
            },
            ensure_ascii=False,
        )

    def has_state(self, name: str) -> bool:
        if name == "tables":
            return bool(self.tables)
        if name == "revenue":
            return self.revenue is not None
        raise ValueError(f"Unknown workspace state: {name}")


def create_workspace(
    settings: Settings,
    *,
    config: dict[str, Any] | None = None,
) -> RevenueAnalysisWorkspace:
    level_config = config or load_level_config(2)
    level1_config = load_level_config(1)
    tool = level_config["tool"]
    return RevenueAnalysisWorkspace(
        db=SakilaDB(settings),
        queries=dict(level1_config["workflow"]["tables"]),
        minimum_n=int(tool["min_n"]),
        maximum_n=int(tool["max_n"]),
    )
