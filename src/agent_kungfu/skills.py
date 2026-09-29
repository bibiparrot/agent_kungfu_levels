from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .database import QUERY_CATALOG, SakilaDB


SkillHandler = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    handler: SkillHandler

    def invoke(self, argument: str = "") -> str:
        return self.handler(argument)


class SkillRegistry:
    """Small code-owned skill surface used from level 2 onward."""

    def __init__(self, db: SakilaDB) -> None:
        self.db = db
        self._skills: dict[str, Skill] = {
            "inspect_schema": Skill(
                "inspect_schema",
                "Return the actual SQLite DDL for Sakila tables.",
                lambda _: db.schema_text(),
            ),
            "run_readonly_sql": Skill(
                "run_readonly_sql",
                "Run exactly one read-only SELECT/CTE through Polars and ConnectorX.",
                self._run_sql,
            ),
            "business_query": Skill(
                "business_query",
                "Run one reviewed business query: store_revenue, category_revenue, top_films, customer_value.",
                self._business_query,
            ),
        }

    def names(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(f"Unknown skill {name!r}; available: {', '.join(self.names())}") from exc

    def _run_sql(self, sql: str) -> str:
        frame = self.db.query(sql, limit=50)
        return json.dumps(self.db.records(frame, max_rows=50), ensure_ascii=False, default=str)

    def _business_query(self, name: str) -> str:
        if name not in QUERY_CATALOG:
            raise ValueError(f"Unknown business query: {name}")
        return self._run_sql(QUERY_CATALOG[name])

