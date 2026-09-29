---
name: sakila-analyst
description: Analyze the local Sakila SQLite database with read-only, evidence-backed SQL.
---

# Sakila Analyst

1. Inspect the live schema before inventing tables or columns.
2. Use exactly one `SELECT` or `WITH` statement.
3. Execute data reads through `SakilaDB.query`, which uses Polars `read_database_uri` and ConnectorX.
4. Include the grouping grain and concrete values in the answer.
5. Call associations “observed relationships”; do not claim causality without an intervention or causal design.
6. Prefer the reviewed query catalogue when it answers the question.

