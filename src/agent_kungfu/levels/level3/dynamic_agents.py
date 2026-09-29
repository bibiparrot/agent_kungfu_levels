from __future__ import annotations

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...config import PROJECT_ROOT, Settings
from ...contracts import LevelResult, Protocol
from ...orchestration import ProtocolBus
from ..common import load_baseline_assets, load_level_config, resolved_settings
from ..level1.workflow import _requested_n
from ..revenue_workspace import RevenueAnalysisWorkspace, create_workspace
from .rpc import RevenueRpc


class LeaderDefinition(BaseModel):
    name: str
    instructions: str


class SubagentDefinition(BaseModel):
    id: str
    name: str
    exposed_tool: str
    description: str
    runtime_tool: str
    requires: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    instructions: str


class AgentManifest(BaseModel):
    schema_version: int = 1
    leader: LeaderDefinition
    subagents: list[SubagentDefinition]

    @model_validator(mode="after")
    def validate_contract(self) -> "AgentManifest":
        if len(self.subagents) != 2:
            raise ValueError("agent.md must declare exactly two subagents")
        for field_name in ("id", "name", "exposed_tool"):
            values = [getattr(item, field_name) for item in self.subagents]
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate subagent {field_name}")
        allowed_tools = {"read_tables", "calculate_revenue"}
        allowed_state = {"tables", "revenue"}
        if {item.runtime_tool for item in self.subagents} != allowed_tools:
            raise ValueError("agent.md must declare one reader and one calculator")
        for item in self.subagents:
            if item.runtime_tool not in allowed_tools:
                raise ValueError(f"Unknown runtime_tool: {item.runtime_tool}")
            if not set(item.requires + item.produces) <= allowed_state:
                raise ValueError(f"Unknown state dependency in subagent: {item.id}")
        return self


class CodexAction(BaseModel):
    """Structured action proposed by one read-only Codex thread."""

    model_config = ConfigDict(extra="forbid")

    runtime_tool: Literal["read_tables", "calculate_revenue"]
    n: int | None = None
    summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_arguments(self) -> "CodexAction":
        if self.runtime_tool == "read_tables" and self.n is not None:
            raise ValueError("read_tables requires n=null")
        if self.runtime_tool == "calculate_revenue" and (self.n is None or self.n < 1):
            raise ValueError("calculate_revenue requires a positive n")
        return self


@dataclass(frozen=True, slots=True)
class CodexWorker:
    """A dynamic agent.md definition backed by one Codex Python SDK thread."""

    definition: SubagentDefinition
    timeout_seconds: float = 240.0
    reasoning_effort: str = "none"

    def run(
        self,
        codex: Any,
        *,
        settings: Settings,
        requested_n: int,
        workspace: RevenueAnalysisWorkspace,
    ) -> tuple[CodexAction, dict[str, Any]]:
        from openai_codex import ApprovalMode, Sandbox

        state = {
            "tables": workspace.has_state("tables"),
            "revenue": workspace.has_state("revenue"),
        }
        prompt = (
            f"Agent id: {self.definition.id}\n"
            f"Task description: {self.definition.description}\n"
            f"Current shared state: {json.dumps(state)}\n"
            f"Requested Top-N: {requested_n}\n"
            f"Your only allowed runtime action is {self.definition.runtime_tool!r}. "
            "Do not run shell commands, inspect files, call tools, or modify anything. "
            "Return only one JSON object with exactly runtime_tool, n, and summary fields. "
            "For calculate_revenue, set n to the requested Top-N; otherwise set n to null. "
            "Do not wrap the JSON in Markdown."
        )
        thread = codex.thread_start(
            approval_mode=ApprovalMode.deny_all,
            base_instructions=self.definition.instructions,
            model=settings.ollama_model,
            model_provider="ollama",
            sandbox=Sandbox.read_only,
            ephemeral=True,
        )
        # This turn performs narrow action routing, not open-ended reasoning.  ``none`` keeps
        # local Ollama latency bounded while Pydantic and the host allowlist retain control.
        handle = thread.turn(prompt, effort=self.reasoning_effort)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"codex-l3-{self.definition.id}")
        future = executor.submit(handle.run)
        try:
            turn_result = future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError as exc:
            handle.interrupt()
            try:
                future.result(timeout=10)
            except Exception:
                pass
            raise RuntimeError(
                f"Codex worker {self.definition.id!r} timed out after {self.timeout_seconds}s"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if not turn_result.final_response:
            raise RuntimeError(f"Codex worker {self.definition.id!r} returned no final response")
        action = _parse_codex_action(turn_result.final_response)
        if action.runtime_tool != self.definition.runtime_tool:
            raise RuntimeError(
                f"Codex worker {self.definition.id!r} requested disallowed action "
                f"{action.runtime_tool!r}"
            )
        if action.runtime_tool == "calculate_revenue" and action.n != requested_n:
            raise RuntimeError(
                f"Codex calculator returned n={action.n!r}; expected requested n={requested_n}"
            )
        usage = turn_result.usage
        usage_payload = (
            usage.model_dump(mode="json")
            if usage is not None and hasattr(usage, "model_dump")
            else None
        )
        return action, {
            "agent": self.definition.name,
            "runtime_tool": action.runtime_tool,
            "thread_id": thread.id,
            "turn_id": turn_result.id,
            "status": getattr(turn_result.status, "value", str(turn_result.status)),
            "duration_ms": turn_result.duration_ms,
            "usage": usage_payload,
            "summary": action.summary,
            "simulated": False,
        }


def load_agent_manifest(path: Path | None = None) -> AgentManifest:
    manifest_path = path or Path(__file__).with_name("agent.md")
    text = manifest_path.read_text(encoding="utf-8")
    match = re.match(r"\A---\s*\n(.*?)\n---(?:\s*\n|\Z)", text, re.DOTALL)
    if not match:
        raise ValueError(f"Missing YAML frontmatter in {manifest_path}")
    raw = yaml.safe_load(match.group(1))
    return AgentManifest.model_validate(raw)


def build_codex_workers_from_manifest(
    manifest: AgentManifest,
    *,
    timeout_seconds: float = 240.0,
    reasoning_effort: str = "none",
) -> tuple[CodexWorker, ...]:
    """Generate exactly two Codex workers from validated agent.md data."""

    return tuple(
        CodexWorker(
            definition=item,
            timeout_seconds=timeout_seconds,
            reasoning_effort=reasoning_effort,
        )
        for item in manifest.subagents
    )


def _parse_codex_action(response: str) -> CodexAction:
    clean = response.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
    try:
        return CodexAction.model_validate_json(clean)
    except Exception as first_error:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            raise ValueError("Codex worker did not return a JSON action") from first_error
        return CodexAction.model_validate_json(match.group(0))


def _run_offline_workers(
    workers: tuple[CodexWorker, ...],
    workspace: RevenueAnalysisWorkspace,
    *,
    requested_n: int,
    rpc: RevenueRpc,
) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for worker in workers:
        definition = worker.definition
        if not all(workspace.has_state(state) for state in definition.requires):
            raise RuntimeError(f"Unmet state dependency for {definition.id}")
        action = CodexAction(
            runtime_tool=definition.runtime_tool,
            n=requested_n if definition.runtime_tool == "calculate_revenue" else None,
            summary=f"Offline simulation selected allowlisted action {definition.runtime_tool}.",
        )
        rpc.call(action.runtime_tool, action.n)
        turns.append(
            {
                "agent": definition.name,
                "runtime_tool": action.runtime_tool,
                "thread_id": None,
                "turn_id": None,
                "status": "simulated",
                "duration_ms": 0,
                "usage": None,
                "summary": action.summary,
                "simulated": True,
            }
        )
    return turns


def _run_live_workers(
    workers: tuple[CodexWorker, ...],
    workspace: RevenueAnalysisWorkspace,
    settings: Settings,
    *,
    requested_n: int,
    rpc: RevenueRpc,
) -> list[dict[str, Any]]:
    from openai_codex import Codex, CodexConfig

    config = CodexConfig(
        cwd=str(PROJECT_ROOT),
        env={
            "OPENAI_BASE_URL": settings.ollama_base_url,
            "OPENAI_API_KEY": settings.ollama_api_key,
        },
        client_title="Agent Kung-fu Level 3",
    )
    turns: list[dict[str, Any]] = []
    with Codex(config) as codex:
        for worker in workers:
            definition = worker.definition
            if not all(workspace.has_state(state) for state in definition.requires):
                raise RuntimeError(f"Unmet state dependency for {definition.id}")
            action, turn = worker.run(
                codex,
                settings=settings,
                requested_n=requested_n,
                workspace=workspace,
            )
            rpc.call(action.runtime_tool, action.n)
            turns.append(turn)
    return turns


def _trace(
    manifest: AgentManifest,
    workspace: RevenueAnalysisWorkspace,
    codex_turns: list[dict[str, Any]],
) -> list[Any]:
    bus = ProtocolBus()
    request = bus.send(
        Protocol.A2A,
        "user",
        manifest.leader.name,
        "analysis.request",
        {"goal": "top category revenue"},
    )
    for definition, event, turn in zip(
        manifest.subagents,
        workspace.events,
        codex_turns,
        strict=True,
    ):
        bus.send(
            Protocol.A2A,
            manifest.leader.name,
            definition.name,
            "task.delegate",
            {"runtime_tool": definition.runtime_tool},
            correlation_id=request.id,
        )
        bus.send(
            Protocol.CODEX_JSONRPC,
            definition.name,
            "codex-app-server",
            "turn.completed" if not turn["simulated"] else "turn.simulated",
            {
                "runtime_tool": turn["runtime_tool"],
                "thread_id": turn["thread_id"],
                "turn_id": turn["turn_id"],
                "status": turn["status"],
                "duration_ms": turn["duration_ms"],
            },
            correlation_id=request.id,
        )
        bus.send(
            Protocol.MCP,
            definition.name,
            manifest.leader.name,
            "tool.result",
            {"event": event},
            correlation_id=request.id,
        )
    return bus.messages


def _build_result(
    question: str,
    answer: str,
    settings: Settings,
    manifest: AgentManifest,
    workspace: RevenueAnalysisWorkspace,
    codex_turns: list[dict[str, Any]],
    *,
    mode: str,
) -> LevelResult:
    if workspace.selected is None or workspace.merged is None or workspace.revenue is None:
        raise RuntimeError("The generated Codex workers did not complete their data contract")
    if workspace.events != ["read_tables", "calculate_revenue"]:
        raise RuntimeError(f"Unexpected generated-agent call order: {workspace.events}")
    trace = _trace(manifest, workspace, codex_turns)
    assets = load_baseline_assets(settings)
    return LevelResult(
        level=3,
        driver="Protocol Agent：agent.md + Codex Python SDK + A2A/MCP/JSON-RPC envelopes",
        question=question,
        answer=answer,
        sql=assets.sql,
        row_count=workspace.selected.height,
        trace=trace,
        metadata={
            "definition_source": str(Path(__file__).with_name("agent.md")),
            "generated_agents": [item.name for item in manifest.subagents],
            "runtime_tools": [item.runtime_tool for item in manifest.subagents],
            "sdk": "openai-codex",
            "transport": "codex-app-server-jsonrpc",
            "model_provider": "ollama",
            "sandbox": "read-only",
            "approval_mode": "deny_all",
            "ephemeral_threads": True,
            "codex_threads_started": sum(not item["simulated"] for item in codex_turns),
            "codex_turns": codex_turns,
            "protocols": sorted({message.protocol.value for message in trace}),
            "agent_calls": workspace.events,
            "joined_rows": workspace.merged.height,
            "revenue_rows": workspace.revenue.height,
            "rows": workspace.selected.to_dicts(),
            "mode": mode,
        },
    )


def run(
    question: str,
    *,
    settings: Settings | None = None,
    offline: bool = False,
    top_n: int | None = None,
) -> LevelResult:
    """Generate two agent.md workers and run them through the Codex Python SDK."""

    resolved = resolved_settings(settings)
    config = load_level_config(3)
    requested_n = (
        top_n
        if top_n is not None
        else _requested_n(question, int(config["tool"]["default_n"]))
    )
    manifest = load_agent_manifest()
    workers = build_codex_workers_from_manifest(
        manifest,
        timeout_seconds=float(config["codex"]["turn_timeout_seconds"]),
        reasoning_effort=str(config["codex"]["reasoning_effort"]),
    )
    workspace = create_workspace(resolved, config={"tool": config["tool"]})
    rpc = RevenueRpc(workspace, requested_n)
    codex_turns = (
        _run_offline_workers(workers, workspace, requested_n=requested_n, rpc=rpc)
        if offline
        else _run_live_workers(workers, workspace, resolved, requested_n=requested_n, rpc=rpc)
    )
    rows = workspace.selected.to_dicts() if workspace.selected is not None else []
    prefix = (
        "agent.md 生成的两个 Codex 子智能体已按离线协议模拟依次完成读取与计算"
        if offline
        else "Codex Python SDK 的两个只读临时线程已依次完成读取与计算授权"
    )
    answer = prefix + "：" + "；".join(
        f"{index}. {row['name']} ({float(row['revenue']):.2f})"
        for index, row in enumerate(rows, start=1)
    )
    result = _build_result(
        question,
        answer,
        resolved,
        manifest,
        workspace,
        codex_turns,
        mode="offline-simulated-codex" if offline else "codex-ollama",
    )
    result.metadata["rpc_exchanges"] = rpc.exchanges
    result.metadata["protocol_scope"] = "JSON-RPC loopback; A2A/MCP envelopes are illustrative"
    return result


async def arun(
    question: str,
    *,
    settings: Settings | None = None,
    offline: bool = False,
    top_n: int | None = None,
) -> LevelResult:
    return await asyncio.to_thread(
        run,
        question,
        settings=settings,
        offline=offline,
        top_n=top_n,
    )
