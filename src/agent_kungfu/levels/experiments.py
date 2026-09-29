"""Clickable, isolated interventions for the executable textbook (no model service required)."""
import json
import shutil
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from ..database import SakilaDB
from ..graph_engineering import build_schema_graph, generate_sakila_ontology
from . import run_level
from .common import resolved_settings


def run_experiment(level, *, settings=None):
    if level not in range(1, 11):
        raise ValueError("Experiment level must be 1–10")
    resolved = resolved_settings(settings)
    checks = []

    def check(name, passed, evidence):
        checks.append({"check": name, "passed": bool(passed), "evidence": str(evidence)})

    def rejected(name, call):
        try:
            call()
        except (ValueError, RuntimeError, TypeError) as exc:
            check(name, True, str(exc))
        else:
            check(name, False, "Unexpectedly accepted")

    with TemporaryDirectory(prefix="kungfu-experiment-") as scratch:
        root = Path(scratch)
        local = replace(resolved, output_dir=root / "outputs")
        if level == 1:
            from .level1.workflow import top_n_categories
            result = run_level(1, "top 3 categories", settings=local, offline=True)
            check("Changing N / 改变 N", result.row_count == 3, result.metadata["rows"])
            rejected("Invalid N / 非法 N", lambda: top_n_categories(pl.DataFrame(result.metadata["rows"]), 0))
        elif level == 2:
            result = run_level(2, "top 3 categories", settings=local, offline=True)
            check("Real SDK delegation / SDK 委派", len(result.metadata["sdk_tool_calls"]) == 2,
                  result.metadata["sdk_tool_calls"])
            check("Computed evidence / 计算证据", result.metadata["rows"][0]["revenue"] == 5314.21,
                  result.metadata["rows"])
        elif level == 3:
            from .level3.rpc import RevenueRpc
            from .revenue_workspace import create_workspace
            rpc = RevenueRpc(create_workspace(local), 3)
            request = json.dumps({"jsonrpc": "2.0", "id": "1", "method": "read_tables", "params": {}})
            first, second = json.loads(rpc.handle(request)), json.loads(rpc.handle(request))
            check("Replay rejected / 拒绝重放", "result" in first and "error" in second, second)
            unknown = json.loads(rpc.handle(json.dumps({"jsonrpc": "2.0", "id": "2", "method": "shell"})))
            check("Tool allowlist / 工具白名单", unknown.get("error", {}).get("code") == -32601, unknown)
        elif level == 4:
            result = run_level(4, "top 3 categories", settings=local, offline=True)
            decision = result.metadata["conflict_decision"]
            check("Evidence beats confidence / 证据胜过置信度", decision["selected_agent"] == "reference-backed", decision)
        elif level == 5:
            from .level5 import run
            result = run("top 3 categories", settings=local, offline=True, fault="early-calculation")
            check("Repair and terminate / 修复并停止", len(result.metadata["repairs"]) == 1
                  and result.metadata["stop_reason"] == "goal_satisfied", result.metadata["repairs"])
            rejected("Bounded failure / 有界失败", lambda: run("top 3 categories", settings=local,
                     offline=True, fault="persistent-invalid"))
        elif level == 6:
            from .level6 import run
            graph = build_schema_graph(SakilaDB(local))
            result = run("top 3 categories", settings=local, offline=True, graph=graph)
            check("Graph execution / 图驱动执行", result.metadata["rows"][0]["revenue"] == 5314.21,
                  result.metadata["task_order"])
            graph.remove_edge("film_category", "category")
            rejected("Missing relation / 缺失关系", lambda: run("top 3 categories", settings=local, offline=True, graph=graph))
        elif level == 7:
            local = replace(local, db_path=root / "sakila-copy.db")
            shutil.copyfile(resolved.db_path, local.db_path)
            first = run_level(7, "top 3 categories", settings=local, offline=True)
            second = run_level(7, "top 3 categories", settings=local, offline=True)
            check("Cross-run retrieval / 跨次检索", not first.metadata["memory"]["hit"]
                  and second.metadata["memory"]["hit"], second.metadata["memory"])
            # Only this owned temporary copy is mutated; all data reads use Polars.
            with closing(sqlite3.connect(local.db_path)) as connection, connection:
                connection.execute("UPDATE payment SET amount=amount+1 WHERE payment_id=1")
            third = run_level(7, "top 3 categories", settings=local, offline=True)
            check("Freshness / 数据失效", not third.metadata["memory"]["hit"], third.metadata["memory"])
        elif level == 8:
            from .level8.ontology_agent import publish_knowledge
            db = SakilaDB(local)
            ontology = generate_sakila_ontology(db, root / "ontology")
            claim = {"subject_type": "Rental", "predicate": "rental_inventory_id", "object_type": "Inventory"}
            page = publish_knowledge(db, ontology.owl, [claim], root / "accepted.md")
            check("Grounded knowledge / 有证据的知识", "16044" in page.read_text(encoding="utf-8"), claim)
            rejected("Invalid range / 非法值域", lambda: publish_knowledge(db, ontology.owl,
                     [{**claim, "object_type": "Customer"}], root / "rejected.md"))
        elif level == 9:
            from .level9.causal_agent import estimate_effect, simulate_rentals
            data = simulate_rentals(seed=42, effect=2.0, n=4000)
            estimate = estimate_effect(data)
            check("Recover known effect / 恢复已知效应", abs(estimate["adjusted"] - 2) < 0.2
                  and abs(estimate["naive"] - 2) > 3, estimate)
            rejected("Positivity / 共同支持", lambda: estimate_effect(data.filter(
                (pl.col("demand") == 0) | (pl.col("treatment") == 1))))
        else:
            from .level10.principles_agent import search_interventions
            params = dict(copies=[5, 5], price=4, unit_cost=1, transfer_cost=0.25, purchase_cost=20)
            report = search_interventions(demand=[12, 2], **params)
            check("Falsify revenue-only objective / 否证收入目标", report["selected"]["kind"] == "rebalance"
                  and report["revenue_winner"]["kind"] == "purchase", report)
            control = search_interventions(demand=[5, 5], **params)
            check("Negative control / 阴性对照", control["selected"] is None, control["stop_reason"])
    return {"level": level, "checks": checks, "model_evidence": "deterministic; live services not tested"}
