from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Coroutine, TypeVar

from ...config import Settings
from ...contracts import LevelResult
from ..common import load_baseline_assets, load_level_config, resolved_settings
from ..level1.workflow import _requested_n
from ..revenue_workspace import RevenueAnalysisWorkspace, create_workspace


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class AgentTeam:
    leader: Any
    specialists: tuple[Any, ...]
    workspace: RevenueAnalysisWorkspace

def _sdk_model(settings: Settings) -> tuple[Any, Any]:
    from agents import ModelSettings, set_tracing_disabled
    from agents.extensions.models.litellm_model import LitellmModel

    set_tracing_disabled(True)
    model = LitellmModel(
        model=settings.litellm_model,
        base_url=settings.ollama_base_url,
        api_key=settings.ollama_api_key,
    )
    model_settings = ModelSettings(
        temperature=0,
        parallel_tool_calls=False,
        extra_args={"custom_llm_provider": "openai"},
    )
    return model, model_settings


def build_agent_team(
    workspace: RevenueAnalysisWorkspace,
    settings: Settings,
    *,
    config: dict[str, Any] | None = None,
    model: Any = None,
) -> AgentTeam:
    """Build a QA manager whose two specialist agents are exposed as tools."""

    from agents import Agent, ModelSettings, function_tool

    level_config = config or load_level_config(2)
    if model is None:
        model, common_settings = _sdk_model(settings)
    else:
        common_settings = ModelSettings(temperature=0, parallel_tool_calls=False)
    specialist_settings = ModelSettings(
        temperature=0,
        tool_choice="required",
        parallel_tool_calls=False,
        extra_args={"custom_llm_provider": "openai"},
    )

    @function_tool
    def load_revenue_tables() -> str:
        """Load category, film_category, inventory, rental, and payment into Polars."""

        return workspace.load_tables()

    @function_tool
    def calculate_category_revenue(n: int) -> str:
        """Join staged tables and return the N categories with highest revenue."""

        return workspace.calculate(n)

    specialists_config = level_config["orchestration"]["specialists"]
    table_agent = Agent(
        name=str(specialists_config[0]["name"]),
        instructions="Call load_revenue_tables exactly once and return its compact JSON result.",
        tools=[load_revenue_tables],
        model=model,
        model_settings=specialist_settings,
        tool_use_behavior="stop_on_first_tool",
    )
    calculate_agent = Agent(
        name=str(specialists_config[1]["name"]),
        instructions=(
            "Read the requested N from the task, call calculate_category_revenue exactly once, "
            "and return its JSON evidence."
        ),
        tools=[calculate_category_revenue],
        model=model,
        model_settings=specialist_settings,
        tool_use_behavior="stop_on_first_tool",
    )

    async def reader_enabled(_context: Any, _agent: Any) -> bool:
        return not workspace.tables

    async def calculator_enabled(_context: Any, _agent: Any) -> bool:
        return bool(workspace.tables) and workspace.revenue is None

    leader_config = level_config["orchestration"]["leader"]
    leader = Agent(
        name=str(leader_config["name"]),
        instructions=str(leader_config["instructions"]),
        tools=[
            table_agent.as_tool(
                tool_name=str(specialists_config[0]["exposed_tool"]),
                tool_description="Ask the table-read specialist to stage the five Sakila tables.",
                max_turns=2,
                is_enabled=reader_enabled,
            ),
            calculate_agent.as_tool(
                tool_name=str(specialists_config[1]["exposed_tool"]),
                tool_description="Ask the calculation specialist to join and aggregate category revenue.",
                max_turns=2,
                is_enabled=calculator_enabled,
            ),
        ],
        model=model,
        model_settings=common_settings,
    )
    return AgentTeam(leader=leader, specialists=(table_agent, calculate_agent), workspace=workspace)


def _build_result(
    question: str,
    workspace: RevenueAnalysisWorkspace,
    settings: Settings,
    answer: str,
    *,
    mode: str,
) -> LevelResult:
    if workspace.selected is None or workspace.merged is None or workspace.revenue is None:
        raise RuntimeError("The QA leader did not complete both required specialist calls")
    if workspace.events != ["read_tables", "calculate_revenue"]:
        raise RuntimeError(f"Unexpected specialist call order: {workspace.events}")
    assets = load_baseline_assets(settings)
    return LevelResult(
        level=2,
        driver="Code Agent：OpenAI Agents QA leader + reader/calculator agent tools",
        question=question,
        answer=answer,
        sql=assets.sql,
        row_count=workspace.selected.height,
        metadata={
            "sdk": "openai-agents",
            "pattern": "manager-as-tools",
            "agents": ["Sakila QA Leader", "Sakila Table Read Agent", "Sakila Revenue Calculate Agent"],
            "agent_calls": workspace.events,
            "joined_rows": workspace.merged.height,
            "revenue_rows": workspace.revenue.height,
            "rows": workspace.selected.to_dicts(),
            "mode": mode,
        },
    )


async def arun(
    question: str,
    *,
    settings: Settings | None = None,
    offline: bool = False,
    top_n: int | None = None,
) -> LevelResult:
    resolved = resolved_settings(settings)
    config = load_level_config(2)
    n = top_n if top_n is not None else _requested_n(question, int(config["tool"]["default_n"]))
    workspace = create_workspace(resolved, config=config)
    from agents import Runner, set_tracing_disabled

    set_tracing_disabled(True)
    model = None
    if offline:
        from .offline_model import LessonModel
        model = LessonModel(n)
    team = build_agent_team(workspace, resolved, config=config, model=model)
    prompt = (
        f"{question}\nThe requested top_n is {n}. You must call read_sakila_tables first, "
        "then calculate_category_revenue, then answer from its JSON rows."
    )
    result = await Runner.run(team.leader, prompt, max_turns=8)
    report = _build_result(question, workspace, resolved, str(result.final_output),
                           mode="offline" if offline else "ollama")
    report.metadata["model_boundary"] = "scripted" if offline else "ollama"
    report.metadata["sdk_tool_calls"] = [
        {"agent": item.agent.name, "tool": item.raw_item.name}
        for item in result.new_items if item.type == "tool_call_item"
    ]
    return report


def _run_coroutine_sync(coroutine: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-kungfu") as executor:
        return executor.submit(asyncio.run, coroutine).result()


def run(
    question: str,
    *,
    settings: Settings | None = None,
    offline: bool = False,
    top_n: int | None = None,
) -> LevelResult:
    """Code Agent: let a QA leader call reader and calculator agent-tools."""

    return _run_coroutine_sync(arun(question, settings=settings, offline=offline, top_n=top_n))
