from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import polars as pl

from .config import Settings


_FORBIDDEN = re.compile(
    r"\b(attach|alter|create|delete|detach|drop|insert|pragma|reindex|replace|update|vacuum)\b",
    re.IGNORECASE,
)
_COMMENT = re.compile(r"(--[^\n]*|/\*.*?\*/)", re.DOTALL)


def validate_read_only_sql(sql: str) -> str:
    """Accept one SELECT/CTE statement and reject mutation or stacked SQL."""

    clean = _COMMENT.sub(" ", sql).strip()
    if clean.endswith(";"):
        clean = clean[:-1].rstrip()
    if not clean or ";" in clean:
        raise ValueError("Exactly one SQL statement is allowed")
    if not re.match(r"^(select|with)\b", clean, re.IGNORECASE):
        raise ValueError("Only SELECT or WITH queries are allowed")
    if _FORBIDDEN.search(clean):
        raise ValueError("Mutation and SQLite control statements are forbidden")
    return clean


@dataclass(slots=True)
class SakilaDB:
    settings: Settings

    def __post_init__(self) -> None:
        self.settings.validate()

    def query(self, sql: str, *, limit: int | None = None) -> pl.DataFrame:
        clean = validate_read_only_sql(sql)
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            clean = f"SELECT * FROM ({clean}) AS agent_query LIMIT {int(limit)}"
        # This is the single data-access path used by all demos.
        return pl.read_database_uri(
            query=clean,
            uri=self.settings.sqlite_uri,
            engine="connectorx",
        )

    def objects(self) -> pl.DataFrame:
        return self.query(
            """
            SELECT name, type, sql
            FROM sqlite_master
            WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        )

    def table_names(self) -> list[str]:
        frame = self.query(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return frame.get_column("name").to_list()

    def schema_text(self) -> str:
        rows = self.objects().filter(pl.col("type") == "table").select("name", "sql").rows()
        return "\n\n".join(f"-- {name}\n{ddl}" for name, ddl in rows if ddl)

    def profile(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for table in self.table_names():
            safe_table = table.replace('"', '""')
            frame = self.query(f'SELECT COUNT(*) AS n FROM "{safe_table}"')
            counts[table] = int(frame.item(0, "n"))
        return counts

    @staticmethod
    def records(frame: pl.DataFrame, *, max_rows: int = 20) -> list[dict[str, Any]]:
        return frame.head(max_rows).to_dicts()


QUERY_CATALOG: dict[str, str] = {
    "store_revenue": """
        SELECT i.store_id,
               ROUND(SUM(p.amount), 2) AS revenue,
               COUNT(DISTINCT r.rental_id) AS rentals,
               COUNT(DISTINCT r.customer_id) AS customers,
               ROUND(AVG(p.amount), 2) AS avg_payment
        FROM payment p
        JOIN rental r ON r.rental_id = p.rental_id
        JOIN inventory i ON i.inventory_id = r.inventory_id
        GROUP BY i.store_id
        ORDER BY revenue DESC
    """,
    "category_revenue": """
        SELECT c.name AS category,
               ROUND(SUM(p.amount), 2) AS revenue,
               COUNT(DISTINCT r.rental_id) AS rentals
        FROM payment p
        JOIN rental r ON r.rental_id = p.rental_id
        JOIN inventory i ON i.inventory_id = r.inventory_id
        JOIN film_category fc ON fc.film_id = i.film_id
        JOIN category c ON c.category_id = fc.category_id
        GROUP BY c.category_id, c.name
        ORDER BY revenue DESC
        LIMIT 10
    """,
    "top_films": """
        SELECT f.title,
               ROUND(SUM(p.amount), 2) AS revenue,
               COUNT(*) AS rentals
        FROM payment p
        JOIN rental r ON r.rental_id = p.rental_id
        JOIN inventory i ON i.inventory_id = r.inventory_id
        JOIN film f ON f.film_id = i.film_id
        GROUP BY f.film_id, f.title
        ORDER BY revenue DESC
        LIMIT 10
    """,
    "customer_value": """
        SELECT c.customer_id,
               c.first_name || ' ' || c.last_name AS customer,
               ROUND(SUM(p.amount), 2) AS lifetime_value,
               COUNT(DISTINCT r.rental_id) AS rentals
        FROM customer c
        JOIN payment p ON p.customer_id = c.customer_id
        JOIN rental r ON r.rental_id = p.rental_id
        GROUP BY c.customer_id, customer
        ORDER BY lifetime_value DESC
        LIMIT 10
    """,
}


def choose_catalog_query(question: str) -> str:
    q = question.lower()
    if any(word in q for word in ("门店", "store")):
        return QUERY_CATALOG["store_revenue"]
    if any(word in q for word in ("类别", "分类", "category")):
        return QUERY_CATALOG["category_revenue"]
    if any(word in q for word in ("客户", "customer", "用户")):
        return QUERY_CATALOG["customer_value"]
    return QUERY_CATALOG["top_films"]
