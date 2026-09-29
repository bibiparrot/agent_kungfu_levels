"""FK graph → typed join plan → dependent Polars tasks → numeric evidence gate."""
import networkx as nx
import polars as pl
from pydantic import BaseModel

from ...contracts import HarnessManifest, LevelResult, Protocol
from ...database import SakilaDB
from ...evaluation import evaluate_answer_with_deepeval, summarize_frame
from ...graph_engineering import build_schema_graph, graph_summary, shortest_join_path
from ...llm import OllamaLLM
from ..common import load_baseline_assets, resolved_settings
from ..level1.workflow import _requested_n, top_n_categories


class JoinPlan(BaseModel):
    path: list[str]
    joins: list[tuple[str, str, str, str]]
    sql: str


def plan_revenue(graph: nx.DiGraph) -> JoinPlan:
    try:
        path = shortest_join_path(graph, "payment", "category")
    except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
        raise ValueError("No join path from payment to category") from exc
    joins = []
    sql = 'SELECT category.name AS name, ROUND(SUM(payment.amount), 2) AS revenue FROM payment'
    for left, right in zip(path, path[1:]):
        source, target = (left, right) if graph.has_edge(left, right) else (right, left)
        edge = graph.edges[source, target]
        # Identifiers must come from this schema; forbid SQL-shaped graph mutations.
        names = (source, edge["source_column"], target, edge["target_column"])
        if not all(name.isidentifier() for name in names):
            raise ValueError("Invalid graph identifier")
        joins.append(names)
        sql += f' JOIN "{right}" ON "{source}"."{names[1]}" = "{target}"."{names[3]}"'
    sql += ' GROUP BY category.name ORDER BY revenue DESC, name ASC'
    return JoinPlan(path=path, joins=joins, sql=sql)


def execute_plan(db: SakilaDB, plan: JoinPlan) -> tuple[pl.DataFrame, list[str]]:
    tasks = nx.DiGraph()
    tasks.add_edges_from((f"read:{table}", "join") for table in plan.path)
    tasks.add_edge("join", "aggregate")
    frames = {}
    order = list(nx.topological_sort(tasks))
    for task in order:
        if task.startswith("read:"):
            table = task.split(":", 1)[1]
            frame = db.query(f'SELECT * FROM "{table}"')
            frames[table] = frame.rename({column: f"{table}__{column}" for column in frame.columns})
        elif task == "join":
            joined = frames[plan.path[0]]
            for new_table, (source, source_col, target, target_col) in zip(plan.path[1:], plan.joins):
                left, right = f"{source}__{source_col}", f"{target}__{target_col}"
                if source == new_table:
                    left, right = right, left
                joined = joined.join(frames[new_table], left_on=left, right_on=right, how="inner", coalesce=False)
        else:
            revenue = (joined.group_by("category__name")
                       .agg(pl.col("payment__amount").sum().round(2).alias("revenue"))
                       .rename({"category__name": "name"})
                       .sort(["revenue", "name"], descending=[True, False]))
    return revenue, order


def run(question, *, settings=None, offline=False, use_deepeval=False, graph=None):
    resolved = resolved_settings(settings)
    db = SakilaDB(resolved)
    graph = build_schema_graph(db) if graph is None else graph
    plan = plan_revenue(graph)
    revenue, order = execute_plan(db, plan)
    baseline = db.query(load_baseline_assets(resolved).sql)
    if not revenue.equals(baseline):
        raise ValueError("Graph plan disagrees with baseline evidence")
    selected = top_n_categories(revenue, _requested_n(question, 5))
    answer = summarize_frame(selected)
    evaluation = {"passed": True, "score": 1.0, "evaluator": "graph-baseline-parity", "feedback": []}
    metadata = {
        "manifest": HarnessManifest(name="sakila-graph", allowed_protocols=[Protocol.SKILL]).model_dump(mode="json"),
        "graph": graph_summary(graph), "join_path_payment_to_film": shortest_join_path(graph, "payment", "film"),
        "join_plan": plan.model_dump(), "task_order": order, "rows": selected.to_dicts(),
        "evaluation": evaluation, "execution": "graph-derived-polars",
    }
    if use_deepeval and not offline:
        metadata["deepeval"] = evaluate_answer_with_deepeval(question, answer, OllamaLLM(resolved)).model_dump(mode="json")
    return LevelResult(level=6, driver="Graph Engineering Agent", question=question,
                       answer=answer, sql=plan.sql, row_count=selected.height, metadata=metadata)
