---
agent: python demos/level_05_ralph_loop.py
commands:
  - name: tests
    run: pytest -q
  - name: smoke
    run: python demos/level_05_ralph_loop.py --offline
args:
  - question
---

# Sakila Table Exploration and Category-Revenue Loop

Goal: inspect every table in the canonical Sakila catalogue, calculate category revenue from
the evidence-backed join path, verify the complete result against the independent SQL baseline,
and stop immediately. The user question is `{{ args.question }}`.

## Test feedback

{{ commands.tests }}

## Smoke feedback

{{ commands.smoke }}

## Runtime contract

The host snapshots the sorted SQLite table catalogue before iteration. For the bundled Sakila
database this snapshot contains 16 tables, so the complete budget is 17 iterations: one inspection
per table and one final calculation. A configuration whose budget cannot cover
`table_count + 1` is rejected before the loop starts.

Each iteration advances exactly one state transition:

1. Canonically serialize the previous evaluator feedback and compute its SHA-256 digest; also bind the loaded Polars skill digest.
2. While an unseen table remains, request one typed `inspect_table` action for exactly the next
   catalogue entry. The host records its columns, row count, one-row sample boundary, and
   revenue-related column signals. It does not materialize every row from every table.
3. After all tables are inspected, request one typed `calculate_revenue` action with the validated
   Top-N value.
4. Validate the action shape, expected table/action, Top-N, feedback digest, and skill digest before any host
   operation. The model never receives arbitrary SQL, shell, Python, or database authority.
5. Execute only the two host allowlist branches. Revenue calculation loads `category`,
   `film_category`, `inventory`, `rental`, and `payment`, then performs the fixed lazy Polars
   join, aggregation, and descending sort.
6. Evaluate the evidence. Inspection iterations return `continue` because the global goal remains
   incomplete; this is progress, not a failed table read. The calculation passes only when the
   full 16-category Polars result equals `baseline/revenue.sql`, the join has 16,044 rows, and the
   requested Top-N shape is correct.
7. Stop immediately on the verified calculation. The 17-step budget has no speculative retry slot:
   an invalid action raises immediately rather than creating an eighteenth repair turn. If the
   budget is exhausted, delegation evidence is missing, or baseline parity fails, deliver no result.

## OmniGenT and Codex boundary

In live mode, `omnigent-client` uploads the reproducible agent bundle, resolves an online Codex
runner (or uses `OMNIGENT_RUNNER_ID`), binds the session, and invokes exactly one bundled Codex
subagent per iteration. Only server-recorded tool calls are accepted.

In offline teaching mode, the same action trajectory and feedback binding are generated
deterministically and delegation records are marked `simulated`. SQLite inspection, Polars
calculation, and SQL-baseline verification still execute for real. Offline evidence must never be
reported as a connected OmniGenT or Codex session.

Never modify the database, continue after verified success, wrap exhaustion as success, or claim
causal certainty from observational differences.
