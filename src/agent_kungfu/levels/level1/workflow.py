from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, TypedDict

import polars as pl
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ...config import Settings
from ...contracts import LevelResult
from ...database import SakilaDB
from ..common import load_baseline_assets, load_level_config, resolved_settings


class RevenueWorkflowState(TypedDict, total=False):
    question: str
    tables: dict[str, pl.DataFrame]
    table_rows: dict[str, int]
    merged: pl.DataFrame
    revenue: pl.DataFrame


def read_tables(db: SakilaDB, queries: dict[str, str]) -> dict[str, pl.DataFrame]:
    """Read the five configured Sakila tables into independent Polars frames."""

    required = {"category", "film_category", "inventory", "rental", "payment"}
    if set(queries) != required:
        raise ValueError(f"Level 1 requires exactly these tables: {sorted(required)}")
    return {name: db.query(sql) for name, sql in queries.items()}


def merge_tables(tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Apply the four specified joins after projecting collision-free columns."""

    category = tables["category"].select("category_id", "name")
    film_category = tables["film_category"].select("category_id", "film_id")
    inventory = tables["inventory"].select("film_id", "inventory_id")
    rental = tables["rental"].select("inventory_id", "rental_id")
    payment = tables["payment"].select("rental_id", "amount")
    return (
        category.join(film_category, on="category_id", how="inner")
        .join(inventory, on="film_id", how="inner")
        .join(rental, on="inventory_id", how="inner")
        .join(payment, on="rental_id", how="inner")
    )


def calculate_revenue(merged: pl.DataFrame) -> pl.DataFrame:
    """Group by category name, sum payments, and sort revenue descending."""

    return (
        merged.group_by("name")
        .agg(pl.col("amount").sum().round(2).alias("revenue"))
        .sort(["revenue", "name"], descending=[True, False])
    )


def top_n_categories(revenue: pl.DataFrame, n: int, *, minimum: int = 1, maximum: int = 16) -> pl.DataFrame:
    """Typed local function exposed to the LLM as a function tool."""

    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if not minimum <= n <= maximum:
        raise ValueError(f"n must be between {minimum} and {maximum}")
    return revenue.head(n)


def _requested_n(question: str, default: int) -> int:
    patterns = (r"(?:top|前)\s*(\d+)", r"\b(\d+)\s*(?:categories|类别|分类)")
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return default


def _format_answer(frame: pl.DataFrame, *, source: str) -> str:
    rows = frame.to_dicts()
    ranking = "；".join(
        f"{index}. {row['name']} ({float(row['revenue']):.2f})"
        for index, row in enumerate(rows, start=1)
    )
    return f"{source}选择了 {len(rows)} 个收入最高的类别：{ranking}"


def run_workflow(
    question: str,
    *,
    settings: Settings | None = None,
) -> RevenueWorkflowState:
    """Run exactly three LangGraph data steps: read, merge, calculate."""

    from langgraph.graph import END, START, StateGraph

    resolved = resolved_settings(settings)
    config = load_level_config(1)
    db = SakilaDB(resolved)
    queries = dict(config["workflow"]["tables"])

    def read_node(_state: RevenueWorkflowState) -> RevenueWorkflowState:
        tables = read_tables(db, queries)
        return {"tables": tables, "table_rows": {name: frame.height for name, frame in tables.items()}}

    def merge_node(state: RevenueWorkflowState) -> RevenueWorkflowState:
        return {"merged": merge_tables(state["tables"])}

    def calculate_node(state: RevenueWorkflowState) -> RevenueWorkflowState:
        return {"revenue": calculate_revenue(state["merged"])}

    graph = StateGraph(RevenueWorkflowState)
    graph.add_node("read_tables", read_node)
    graph.add_node("merge_tables", merge_node)
    graph.add_node("calculate_revenue", calculate_node)
    graph.add_edge(START, "read_tables")
    graph.add_edge("read_tables", "merge_tables")
    graph.add_edge("merge_tables", "calculate_revenue")
    graph.add_edge("calculate_revenue", END)
    return graph.compile().invoke({"question": question})


def _live_tool_call(
    question: str,
    requested_n: int,
    revenue: pl.DataFrame,
    settings: Settings,
    config: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any], str]:
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from litellm import completion

    prompt_dir = Path(__file__).resolve().parent
    environment = Environment(
        loader=FileSystemLoader(prompt_dir),
        undefined=StrictUndefined,
        autoescape=False,
    )
    prompt = environment.get_template(str(config["llm"]["prompt"])).render(
        question=question,
        requested_n=requested_n,
    )
    tool_config = config["tool"]
    tool_name = str(tool_config["name"])
    tool_schema = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": "Return the N categories with the highest already-calculated revenue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "minimum": int(tool_config["min_n"]),
                        "maximum": int(tool_config["max_n"]),
                    }
                },
                "required": ["n"],
                "additionalProperties": False,
            },
        },
    }
    response = completion(
        model=settings.litellm_model,
        custom_llm_provider="openai",
        base_url=settings.ollama_base_url,
        api_key=settings.ollama_api_key,
        messages=[
            {"role": "system", "content": "Choose N only by calling the supplied function."},
            {"role": "user", "content": prompt},
        ],
        tools=[tool_schema],
        tool_choice={"type": "function", "function": {"name": tool_name}},
        parallel_tool_calls=False,
        temperature=0,
        timeout=180,
    )
    message = response.choices[0].message
    calls = getattr(message, "tool_calls", None) or []
    if len(calls) != 1 or calls[0].function.name != tool_name:
        raise RuntimeError(f"Ollama did not make the required {tool_name} function call")
    raw_arguments = calls[0].function.arguments
    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
    if not isinstance(arguments, dict) or set(arguments) != {"n"}:
        raise ValueError("Tool arguments must contain exactly the requested n")
    n = arguments.get("n")
    if isinstance(n, bool) or not isinstance(n, int) or n != requested_n:
        raise ValueError(f"Model must honor requested n={requested_n}; received {n!r}")
    selected = top_n_categories(
        revenue,
        n,
        minimum=int(tool_config["min_n"]),
        maximum=int(tool_config["max_n"]),
    )
    record = {"name": tool_name, "arguments": {"n": n}, "source": "litellm/ollama"}
    return selected, record, _format_answer(selected, source="Ollama function call")


def run(
    question: str,
    *,
    settings: Settings | None = None,
    offline: bool = False,
    top_n: int | None = None,
) -> LevelResult:
    """Prompt Agent: run the three-step Polars workflow and LiteLLM Top-N tool."""

    resolved = resolved_settings(settings)
    config = load_level_config(1)
    state = run_workflow(question, settings=resolved)
    tool_config = config["tool"]
    requested_n = top_n if top_n is not None else _requested_n(question, int(tool_config["default_n"]))
    if offline:
        selected = top_n_categories(
            state["revenue"],
            requested_n,
            minimum=int(tool_config["min_n"]),
            maximum=int(tool_config["max_n"]),
        )
        tool_call = {
            "name": str(tool_config["name"]),
            "arguments": {"n": requested_n},
            "source": "offline-simulation",
        }
        answer = _format_answer(selected, source="离线 function call")
    else:
        selected, tool_call, answer = _live_tool_call(
            question, requested_n, state["revenue"], resolved, config
        )
    assets = load_baseline_assets(resolved)
    return LevelResult(
        level=1,
        driver="Prompt Agent：Polars 三步工作流 + LiteLLM function call",
        question=question,
        answer=answer,
        sql=assets.sql,
        row_count=selected.height,
        metadata={
            "graph_nodes": ["read_tables", "merge_tables", "calculate_revenue"],
            "table_rows": state["table_rows"],
            "joined_rows": state["merged"].height,
            "revenue_rows": state["revenue"].height,
            "rows": selected.to_dicts(),
            "tool_call": tool_call,
            "config": str(Path(__file__).with_name("config.yaml")),
        },
    )
