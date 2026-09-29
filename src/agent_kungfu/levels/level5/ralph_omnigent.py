from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import json
import re
import tarfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Coroutine, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...config import Settings
from ...contracts import EvalResult, LevelResult, Protocol
from ...database import SakilaDB
from ...orchestration import ProtocolBus
from ..common import load_baseline_assets, load_level_config, resolved_settings
from ..level1.workflow import _requested_n
from ..level4.omnigent_harness import (
    CodexExecutor,
    CodexOSEnvConfig,
    InlineAgentTool,
    LoadedSkill,
    PolarsHarnessWorkspace,
    RootBundleManifest,
    SubagentBundleManifest,
    TopLevelExecutor,
    _portable_skill_bytes,
    load_polars_skill,
)


_WORKER_ORDER = ("inspect_table", "calculate_revenue")
_WORKER_NAMES = {
    "inspect_table": "Codex Sakila Table Explorer",
    "calculate_revenue": "Codex Category Revenue Solver",
}
_REVENUE_CHAIN = ("category", "film_category", "inventory", "rental", "payment")
_REQUIRED_REVENUE_COLUMNS = {
    "category": {"category_id", "name"},
    "film_category": {"category_id", "film_id"},
    "inventory": {"film_id", "inventory_id"},
    "rental": {"inventory_id", "rental_id"},
    "payment": {"rental_id", "amount"},
}
T = TypeVar("T")


class RalphOmnigentManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    spec_version: int = 1
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    skills: list[str]
    os_env: CodexOSEnvConfig
    executor: TopLevelExecutor
    prompt: str = Field(min_length=1)
    tools: dict[str, InlineAgentTool]

    @model_validator(mode="after")
    def validate_level5_contract(self) -> "RalphOmnigentManifest":
        if self.skills != ["polars"]:
            raise ValueError("Level 5 must load the bundled Polars skill")
        if tuple(self.tools) != _WORKER_ORDER:
            raise ValueError("Level 5 must declare inspect_table then calculate_revenue")
        return self


class RalphToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration: int = Field(ge=1)
    name: Literal["inspect_table", "calculate_revenue"]
    call_id: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    executed_by: Literal["server", "simulated"]
    completed: bool = False
    output_sha256: str | None = None


class RalphCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class RalphRunEvaluation(BaseModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    checks: list[RalphCheck]
    feedback: list[str]
    evaluator: str = "ralph-loop-deterministic-harness"


def load_omnigent_manifest(path: Path | None = None) -> RalphOmnigentManifest:
    manifest_path = path or Path(__file__).with_name("omnigent-agent") / "config.yaml"
    raw = _load_yaml_mapping(manifest_path)
    root = RootBundleManifest.model_validate(raw)
    if root.skills != ["polars"] or tuple(root.tools.agents) != _WORKER_ORDER:
        raise ValueError("Level 5 root must expose Explorer then Revenue Solver with Polars")
    tools: dict[str, InlineAgentTool] = {}
    for agent_id in root.tools.agents:
        child_path = manifest_path.parent / "agents" / agent_id / "config.yaml"
        child = SubagentBundleManifest.model_validate(_load_yaml_mapping(child_path))
        if child.name != agent_id or child.skills != ["polars"]:
            raise ValueError(f"Invalid Level 5 subagent contract in {child_path}")
        tools[agent_id] = InlineAgentTool(
            type="agent",
            description=child.description,
            prompt=child.prompt,
            executor=CodexExecutor(
                harness=child.executor.config.harness,
                model=child.executor.model,
            ),
            os_env=child.os_env,
            pass_history=False,
            max_sessions=1,
        )
    return RalphOmnigentManifest(
        spec_version=root.spec_version,
        name=root.name,
        description=root.description,
        skills=root.skills,
        os_env=root.os_env,
        executor=root.executor,
        prompt=root.prompt,
        tools=tools,
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def build_agent_bundle(
    manifest_path: Path | None = None,
    skill: LoadedSkill | None = None,
) -> bytes:
    """Build a byte-for-byte reproducible OmniGenT bundle for the Ralph loop."""

    source_manifest = manifest_path or Path(__file__).with_name("omnigent-agent") / "config.yaml"
    loaded_skill = skill or load_polars_skill()
    entries: dict[str, bytes] = {"config.yaml": source_manifest.read_bytes()}
    source_root = source_manifest.parent
    for worker_id in _WORKER_ORDER:
        relative_config = f"agents/{worker_id}/config.yaml"
        entries[relative_config] = (source_root / relative_config).read_bytes()
    scopes = ["skills/polars", *(f"agents/{worker}/skills/polars" for worker in _WORKER_ORDER)]
    for scope in scopes:
        entries[f"{scope}/SKILL.md"] = _portable_skill_bytes(
            (loaded_skill.root / "SKILL.md").read_text(encoding="utf-8"),
            source="SKILL.md",
        )
        for relative in loaded_skill.references:
            entries[f"{scope}/{relative}"] = _portable_skill_bytes(
                (loaded_skill.root / relative).read_text(encoding="utf-8"),
                source=relative,
            )
        entries[f"{scope}/SOURCE_SHA256.txt"] = (loaded_skill.digest_sha256 + "\n").encode("ascii")

    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name in sorted(entries):
                payload = entries[name]
                info = tarfile.TarInfo(name=name)
                info.size = len(payload)
                info.mtime = 0
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class RalphAction(BaseModel):
    """One typed, host-validated action proposed for a Ralph iteration."""

    model_config = ConfigDict(extra="forbid")

    agent: Literal["inspect_table", "calculate_revenue"]
    runtime_tool: Literal["inspect_table", "calculate_revenue"]
    table: str | None = None
    n: int | None = None
    feedback_digest_sha256: str = Field(min_length=64, max_length=64)
    skill_digest_sha256: str = Field(min_length=64, max_length=64)
    summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_action_shape(self) -> "RalphAction":
        if self.agent != self.runtime_tool:
            raise ValueError("agent and runtime_tool must match")
        if self.runtime_tool == "inspect_table":
            if not self.table or self.n is not None:
                raise ValueError("inspect_table requires table and n=null")
        elif self.table is not None or self.n is None or isinstance(self.n, bool) or self.n < 1:
            raise ValueError("calculate_revenue requires table=null and positive n")
        return self


class RalphIterationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration: int = Field(ge=1)
    action: RalphAction
    evidence: dict[str, Any]
    evaluation: EvalResult
    decision: Literal["continue", "stop"]


@dataclass(slots=True)
class SakilaExplorationWorkspace:
    """Host-owned state: Codex proposes; this object alone reads SQLite and Polars."""

    db: SakilaDB
    requested_n: int
    table_names: list[str] = field(init=False)
    inspected_tables: list[str] = field(default_factory=list)
    repairs: list[dict[str, Any]] = field(default_factory=list)
    observations: dict[str, dict[str, Any]] = field(default_factory=dict)
    revenue_workspace: PolarsHarnessWorkspace = field(init=False)

    def __post_init__(self) -> None:
        self.table_names = self.db.table_names()
        table_queries = dict(load_level_config(1)["workflow"]["tables"])
        self.revenue_workspace = PolarsHarnessWorkspace(
            db=self.db,
            queries=table_queries,
            minimum_n=1,
            maximum_n=16,
        )

    @property
    def all_tables_inspected(self) -> bool:
        return self.inspected_tables == self.table_names

    @property
    def next_table(self) -> str | None:
        if len(self.inspected_tables) >= len(self.table_names):
            return None
        return self.table_names[len(self.inspected_tables)]

    @property
    def discovered_revenue_chain(self) -> list[str]:
        for table, required_columns in _REQUIRED_REVENUE_COLUMNS.items():
            observation = self.observations.get(table)
            if observation is None or not required_columns.issubset(observation["columns"]):
                return []
        return list(_REVENUE_CHAIN)

    def inspect_table(self, table: str) -> dict[str, Any]:
        expected = self.next_table
        if expected is None:
            raise RuntimeError("all Sakila tables have already been inspected")
        if table != expected:
            raise ValueError(f"expected next table {expected!r}, got {table!r}")
        safe_table = table.replace('"', '""')
        sample = self.db.query(f'SELECT * FROM "{safe_table}"', limit=1)
        count = self.db.query(f'SELECT COUNT(*) AS n FROM "{safe_table}"').item(0, "n")
        observation = {
            "table": table,
            "row_count": int(count),
            "columns": sample.columns,
            "revenue_signals": sorted(
                set(sample.columns)
                & {"category_id", "film_id", "inventory_id", "rental_id", "amount", "name"}
            ),
        }
        self.inspected_tables.append(table)
        self.observations[table] = observation
        return observation

    def calculate_revenue(self) -> dict[str, Any]:
        if not self.all_tables_inspected:
            raise RuntimeError("calculate_revenue is blocked until every table is inspected")
        if self.discovered_revenue_chain != list(_REVENUE_CHAIN):
            raise RuntimeError("the inspected schema does not support the category-revenue chain")
        self.revenue_workspace.load_tables()
        return self.revenue_workspace.calculate_revenue(self.requested_n)


def _feedback_digest(feedback: list[str]) -> str:
    payload = json.dumps(feedback, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _offline_action(
    workspace: SakilaExplorationWorkspace,
    previous: EvalResult | None,
    skill_digest_sha256: str,
) -> RalphAction:
    feedback = [] if previous is None else previous.feedback
    digest = _feedback_digest(feedback)
    if workspace.next_table is not None:
        return RalphAction(
            agent="inspect_table",
            runtime_tool="inspect_table",
            table=workspace.next_table,
            n=None,
            feedback_digest_sha256=digest,
            skill_digest_sha256=skill_digest_sha256,
            summary=f"Inspect the next unseen Sakila table: {workspace.next_table}.",
        )
    return RalphAction(
        agent="calculate_revenue",
        runtime_tool="calculate_revenue",
        table=None,
        n=workspace.requested_n,
        feedback_digest_sha256=digest,
        skill_digest_sha256=skill_digest_sha256,
        summary="All tables are covered; calculate category revenue and verify the goal.",
    )


def _parse_action(text: str) -> RalphAction:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
    try:
        return RalphAction.model_validate_json(clean)
    except Exception as first_error:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            raise ValueError("OmniGenT did not return a typed Ralph action") from first_error
        return RalphAction.model_validate_json(match.group(0))


def _validate_action(
    action: RalphAction,
    workspace: SakilaExplorationWorkspace,
    previous: EvalResult | None,
    skill_digest_sha256: str,
) -> None:
    feedback = [] if previous is None else previous.feedback
    if action.feedback_digest_sha256 != _feedback_digest(feedback):
        raise ValueError("Ralph action did not acknowledge the previous evaluator feedback")
    if action.skill_digest_sha256 != skill_digest_sha256:
        raise ValueError("Ralph action did not acknowledge the loaded Polars skill")
    if workspace.next_table is not None:
        if action.runtime_tool != "inspect_table" or action.table != workspace.next_table:
            raise ValueError(
                f"Ralph action must inspect next table {workspace.next_table!r} before calculation"
            )
        return
    if action.runtime_tool != "calculate_revenue" or action.n != workspace.requested_n:
        raise ValueError("Ralph action must calculate the requested Top-N after full coverage")


def _execute_action(
    workspace: SakilaExplorationWorkspace,
    action: RalphAction,
) -> dict[str, Any]:
    if action.runtime_tool == "inspect_table":
        return workspace.inspect_table(action.table or "")
    return workspace.calculate_revenue()


def _evaluate_iteration(
    workspace: SakilaExplorationWorkspace,
    action: RalphAction,
    evidence: dict[str, Any],
) -> EvalResult:
    if action.runtime_tool == "inspect_table":
        remaining = len(workspace.table_names) - len(workspace.inspected_tables)
        feedback = (
            [f"Coverage incomplete: inspect {remaining} remaining table(s)."]
            if remaining
            else ["All tables inspected; calculate category revenue next."]
        )
        coverage = len(workspace.inspected_tables) / (len(workspace.table_names) + 1)
        return EvalResult(passed=False, score=round(coverage, 4), feedback=feedback)

    baseline = workspace.db.query(load_baseline_assets(workspace.db.settings).sql)
    revenue = workspace.revenue_workspace.revenue
    selected = workspace.revenue_workspace.selected
    parity = revenue is not None and revenue.equals(baseline)
    passed = bool(
        parity
        and selected is not None
        and selected.height == workspace.requested_n
        and evidence.get("category_rows") == 16
        and evidence.get("joined_rows") == 16044
    )
    return EvalResult(
        passed=passed,
        score=1.0 if passed else 0.8,
        feedback=(
            ["Goal satisfied: every table was inspected and category revenue matches baseline."]
            if passed
            else [
                "Revenue evidence failed deterministic baseline parity; replan or exhaust safely."
            ]
        ),
        evaluator="ralph-deterministic-baseline",
    )


def _run_offline_loop(
    workspace: SakilaExplorationWorkspace,
    skill: LoadedSkill,
    fault: str = "none",
) -> list[RalphIterationRecord]:
    history: list[RalphIterationRecord] = []
    previous: EvalResult | None = None
    max_iterations = len(workspace.table_names) + 1
    for iteration in range(1, max_iterations + 1):
        retry_budget = int(load_level_config(5)["loop"]["retry_budget"])
        for attempt in range(retry_budget + 1):
            action = _offline_action(workspace, previous, skill.digest_sha256)
            if fault == "persistent-invalid" or (fault == "early-calculation" and iteration == 1 and attempt == 0):
                action = action.model_copy(update={"agent": "calculate_revenue",
                    "runtime_tool": "calculate_revenue", "table": None, "n": workspace.requested_n})
            try:
                _validate_action(action, workspace, previous, skill.digest_sha256)
                break
            except ValueError as exc:
                previous = EvalResult(passed=False, score=0, feedback=[str(exc)])
                workspace.repairs.append({"iteration": iteration, "attempt": attempt + 1,
                    "feedback": previous.feedback, "rejected": action.model_dump(mode="json")})
                if attempt == retry_budget:
                    raise RuntimeError("Ralph proposal retry budget exhausted") from exc
        evidence = _execute_action(workspace, action)
        evaluation = _evaluate_iteration(workspace, action, evidence)
        decision: Literal["continue", "stop"] = "stop" if evaluation.passed else "continue"
        history.append(
            RalphIterationRecord(
                iteration=iteration,
                action=action,
                evidence=evidence,
                evaluation=evaluation,
                decision=decision,
            )
        )
        if evaluation.passed:
            return history
        previous = evaluation
    raise RuntimeError("Ralph loop exhausted without satisfying the category-revenue goal")


def _live_prompt(
    *,
    workspace: SakilaExplorationWorkspace,
    previous: EvalResult | None,
    iteration: int,
    skill: LoadedSkill,
) -> str:
    feedback = [] if previous is None else previous.feedback
    expected_action = "inspect_table" if workspace.next_table is not None else "calculate_revenue"
    expected_table = workspace.next_table or "null"
    return (
        "Advance exactly one bounded Ralph-loop step. Call exactly the named bundled Codex "
        "subagent, then return only one JSON RalphAction. Never access the database or invent rows.\n"
        f"N={workspace.requested_n}\n"
        f"ITERATION={iteration}\n"
        f"EXPECTED_ACTION={expected_action}\n"
        f"EXPECTED_TABLE={expected_table}\n"
        f"FEEDBACK_DIGEST={_feedback_digest(feedback)}\n"
        f"SKILL_DIGEST={skill.digest_sha256}\n"
        f"PREVIOUS_FEEDBACK={json.dumps(feedback, ensure_ascii=False)}\n"
        "JSON fields: agent, runtime_tool, table, n, feedback_digest_sha256, "
        "skill_digest_sha256, summary. Echo FEEDBACK_DIGEST and SKILL_DIGEST exactly."
    )


async def _run_live_loop(
    workspace: SakilaExplorationWorkspace,
    settings: Settings,
    manifest: RalphOmnigentManifest,
    skill: LoadedSkill,
    bundle: bytes,
    *,
    timeout_seconds: float,
) -> tuple[list[RalphIterationRecord], list[RalphToolCall], str, str]:
    if not settings.omnigent_url:
        raise RuntimeError(
            "Level 5 live mode requires OMNIGENT_URL; use offline=True for the deterministic lesson"
        )
    from omnigent_client import OmnigentClient, StreamHooks

    history: list[RalphIterationRecord] = []
    tool_calls: list[RalphToolCall] = []
    active_iteration = 0

    def record_tool_call(context: Any) -> None:
        try:
            tool_calls.append(
                RalphToolCall(
                    iteration=active_iteration,
                    name=context.name,
                    call_id=context.call_id,
                    agent_name=context.agent_name,
                    executed_by=context.executed_by,
                )
            )
        except Exception as exc:
            raise RuntimeError("OmniGenT emitted a call outside the Level 5 contract") from exc

    def record_tool_end(context: Any) -> None:
        matches = [call for call in tool_calls if call.call_id == context.call_id]
        if len(matches) != 1:
            raise RuntimeError("OmniGenT completed an unknown or duplicate Level 5 tool call")
        call = matches[0]
        # OmniGenT 0.11 only guarantees call_id and output on a
        # function_call_output item.  Some producers repeat name/agent_name;
        # validate them when present without rejecting the minimal wire shape.
        completed_name = getattr(context, "name", "")
        completed_agent_name = getattr(context, "agent_name", "")
        if (completed_name and call.name != completed_name) or (
            completed_agent_name and call.agent_name != completed_agent_name
        ):
            raise RuntimeError("OmniGenT tool completion does not match its start event")
        call.completed = True
        call.output_sha256 = hashlib.sha256(str(context.output).encode("utf-8")).hexdigest()

    hooks = StreamHooks(
        on_tool_call_start=record_tool_call,
        on_tool_call_end=record_tool_end,
    )
    async with OmnigentClient(settings.omnigent_url, timeout=timeout_seconds) as client:
        runner_id = settings.omnigent_runner_id or await client.sessions.resolve_online_runner(
            harness=manifest.executor.config.harness
        )
        if not runner_id:
            raise RuntimeError(
                "Level 5 live mode requires an online Codex runner; set OMNIGENT_RUNNER_ID "
                "or connect a runner to the OmniGenT server"
            )
        chat = await client.sessions_chat(
            bundle=bundle,
            filename=f"{manifest.name}.tar.gz",
            hooks=hooks,
        )
        await client.sessions.bind_runner(chat.session_id, runner_id=runner_id)
        previous: EvalResult | None = None
        max_iterations = len(workspace.table_names) + 1
        for iteration in range(1, max_iterations + 1):
            active_iteration = iteration
            retry_budget = int(load_level_config(5)["loop"]["retry_budget"])
            for attempt in range(retry_budget + 1):
                call_count = len(tool_calls)
                prompt = _live_prompt(workspace=workspace, previous=previous,
                                      iteration=iteration, skill=skill)
                result = await chat.query(prompt)
                expected_tool = "inspect_table" if workspace.next_table is not None else "calculate_revenue"
                iteration_calls = tool_calls[call_count:]
                if len(iteration_calls) != 1 or iteration_calls[0].name != expected_tool:
                    raise RuntimeError(f"OmniGenT iteration {iteration} must invoke exactly {expected_tool} once")
                if iteration_calls[0].executed_by != "server" or not iteration_calls[0].completed:
                    raise RuntimeError("Level 5 live mode accepts only completed server-executed Codex calls")
                try:
                    action = _parse_action(result.text)
                    _validate_action(action, workspace, previous, skill.digest_sha256)
                    break
                except ValueError as exc:
                    previous = EvalResult(passed=False, score=0, feedback=[str(exc)])
                    workspace.repairs.append({"iteration": iteration, "attempt": attempt + 1,
                        "feedback": previous.feedback, "rejected": result.text,
                        "tool_calls": [call.model_dump(mode="json") for call in iteration_calls]})
                    del tool_calls[call_count:]  # Rejected attempts remain in the repair audit trail.
                    if attempt == retry_budget:
                        raise RuntimeError("Ralph proposal retry budget exhausted") from exc
            evidence = await asyncio.to_thread(_execute_action, workspace, action)
            evaluation = _evaluate_iteration(workspace, action, evidence)
            decision: Literal["continue", "stop"] = "stop" if evaluation.passed else "continue"
            history.append(
                RalphIterationRecord(
                    iteration=iteration,
                    action=action,
                    evidence=evidence,
                    evaluation=evaluation,
                    decision=decision,
                )
            )
            if evaluation.passed:
                return history, tool_calls, runner_id, chat.session_id
            previous = evaluation
    raise RuntimeError("Ralph loop exhausted without satisfying the category-revenue goal")


def _offline_tool_calls(history: list[RalphIterationRecord]) -> list[RalphToolCall]:
    return [
        RalphToolCall(
            iteration=item.iteration,
            name=item.action.runtime_tool,
            call_id=f"offline-{item.iteration}-{item.action.runtime_tool}",
            agent_name="sakila-level5-loop",
            executed_by="simulated",
            completed=True,
            output_sha256=hashlib.sha256(b"offline simulated completion").hexdigest(),
        )
        for item in history
    ]


def evaluate_level5_run(
    workspace: SakilaExplorationWorkspace,
    history: list[RalphIterationRecord],
    tool_calls: list[RalphToolCall],
    manifest: RalphOmnigentManifest,
    skill: LoadedSkill,
    *,
    max_iterations: int,
    connected: bool,
) -> RalphRunEvaluation:
    expected_actions = ["inspect_table"] * len(workspace.table_names) + ["calculate_revenue"]
    previous_feedback: list[str] = []
    feedback_bound = True
    for item in history:
        repairs = [repair for repair in workspace.repairs if repair["iteration"] == item.iteration]
        if repairs:
            previous_feedback = repairs[-1]["feedback"]
        feedback_bound &= (
            item.action.feedback_digest_sha256 == _feedback_digest(previous_feedback)
            and item.action.skill_digest_sha256 == skill.digest_sha256
        )
        previous_feedback = item.evaluation.feedback
    selected = workspace.revenue_workspace.selected
    revenue = workspace.revenue_workspace.revenue
    baseline = workspace.db.query(load_baseline_assets(workspace.db.settings).sql)
    calls_match = [call.name for call in tool_calls] == expected_actions
    evidence_kind = "server" if connected else "simulated"
    sandboxed = (
        manifest.os_env.sandbox.type != "none"
        and manifest.os_env.sandbox.write_paths == []
        and manifest.os_env.sandbox.write_files == []
        and manifest.os_env.sandbox.allow_network is False
        and manifest.os_env.start_in_scratch is True
        and all(
            tool.os_env.sandbox.type != "none"
            and tool.os_env.sandbox.write_paths == []
            and tool.os_env.sandbox.write_files == []
            and tool.os_env.sandbox.allow_network is False
            and tool.os_env.start_in_scratch is True
            for tool in manifest.tools.values()
        )
    )
    checks = [
        RalphCheck(
            name="catalog-snapshot",
            passed=len(workspace.table_names) == 16 and len(set(workspace.table_names)) == 16,
            detail=f"catalog contains {len(workspace.table_names)} unique Sakila tables",
        ),
        RalphCheck(
            name="complete-canonical-coverage",
            passed=workspace.inspected_tables == workspace.table_names,
            detail=f"inspected {len(workspace.inspected_tables)}/{len(workspace.table_names)} tables",
        ),
        RalphCheck(
            name="typed-action-trajectory",
            passed=[item.action.runtime_tool for item in history] == expected_actions,
            detail="one inspect action per table, followed by one calculation",
        ),
        RalphCheck(
            name="feedback-binding",
            passed=feedback_bound,
            detail=("every proposal echoes the previous-feedback and loaded-skill SHA-256 digests"),
        ),
        RalphCheck(
            name="omnigent-codex-delegation",
            passed=calls_match
            and len(tool_calls) == len(history)
            and all(call.executed_by == evidence_kind for call in tool_calls)
            and all(call.completed and call.output_sha256 for call in tool_calls)
            and len({call.call_id for call in tool_calls}) == len(tool_calls)
            and [call.iteration for call in tool_calls] == list(range(1, len(history) + 1)),
            detail=(
                f"{len(tool_calls)} completed {evidence_kind} tool-call records match the trajectory"
            ),
        ),
        RalphCheck(
            name="read-only-worker-boundary",
            passed=sandboxed and tuple(manifest.tools) == _WORKER_ORDER,
            detail="two Codex tools declare no writes and no network",
        ),
        RalphCheck(
            name="table-observations",
            passed=set(workspace.observations) == set(workspace.table_names)
            and all(
                "columns" in item and "row_count" in item
                for item in workspace.observations.values()
            )
            and workspace.discovered_revenue_chain == list(_REVENUE_CHAIN),
            detail="each table has evidence and the five-table revenue chain is supported",
        ),
        RalphCheck(
            name="revenue-shape",
            passed=revenue is not None
            and revenue.height == 16
            and selected is not None
            and selected.height == workspace.requested_n
            and workspace.revenue_workspace.joined_rows == 16044,
            detail=(
                f"categories={0 if revenue is None else revenue.height}; "
                f"joined_rows={workspace.revenue_workspace.joined_rows}; top_n={workspace.requested_n}"
            ),
        ),
        RalphCheck(
            name="baseline-parity",
            passed=revenue is not None and revenue.equals(baseline),
            detail="full 16-category Polars result equals revenue.sql",
        ),
        RalphCheck(
            name="bounded-immediate-stop",
            passed=bool(history)
            and len(history) <= max_iterations
            and all(item.decision == "continue" for item in history[:-1])
            and history[-1].decision == "stop"
            and history[-1].evaluation.passed
            and history[-1].action.runtime_tool == "calculate_revenue",
            detail=f"stopped at iteration {len(history)}/{max_iterations} on verified calculation",
        ),
    ]
    failed = [check.name for check in checks if not check.passed]
    return RalphRunEvaluation(
        passed=not failed,
        score=round(sum(check.passed for check in checks) / len(checks), 4),
        checks=checks,
        feedback=[] if not failed else [f"Failed checks: {', '.join(failed)}"],
    )


def _trace(
    history: list[RalphIterationRecord],
    *,
    connected: bool,
) -> list[Any]:
    bus = ProtocolBus()
    request = bus.send(
        Protocol.A2A,
        "user",
        "sakila-level5-loop",
        "loop.request",
        {"goal": "inspect all tables, verify category revenue, then stop"},
    )
    for item in history:
        bus.send(
            Protocol.OMNIGENT_HTTP if connected else Protocol.SKILL,
            "sakila-level5-loop",
            item.action.agent,
            "loop.delegate",
            {"iteration": item.iteration, "table": item.action.table},
            correlation_id=request.id,
        )
        bus.send(
            Protocol.CODEX_JSONRPC,
            item.action.agent,
            "host-action-gate",
            "turn.completed" if connected else "turn.simulated",
            {"runtime_tool": item.action.runtime_tool},
            correlation_id=request.id,
        )
        bus.send(
            Protocol.MCP,
            "host-action-gate",
            "ralph-evaluator",
            "evidence.observed",
            {"passed": item.evaluation.passed, "decision": item.decision},
            correlation_id=request.id,
        )
    bus.send(
        Protocol.OMNIGENT_HTTP if connected else Protocol.SKILL,
        "ralph-evaluator",
        "user",
        "loop.stopped",
        {"reason": "goal_satisfied", "iterations": len(history)},
        correlation_id=request.id,
    )
    return bus.messages


def _run_coroutine_sync(coroutine: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-kungfu-l5") as executor:
        return executor.submit(asyncio.run, coroutine).result()


async def arun(
    question: str,
    *,
    settings: Settings | None = None,
    offline: bool = False,
    top_n: int | None = None,
    fault: str = "none",
) -> LevelResult:
    if fault not in {"none", "early-calculation", "persistent-invalid"}:
        raise ValueError("Unknown teaching fault scenario")
    if fault != "none" and not offline:
        raise ValueError("Fault injection requires offline teaching mode")
    resolved = resolved_settings(settings)
    config = load_level_config(5)
    config_path = Path(__file__).with_name("config.yaml")
    definition_path = (config_path.parent / str(config["definition"]["path"])).resolve()
    skill_path = (config_path.parent / str(config["skill"]["path"])).resolve()
    manifest = load_omnigent_manifest(definition_path)
    skill = load_polars_skill(skill_path)
    bundle = build_agent_bundle(definition_path, skill)
    tool = config["tool"]
    requested_n = top_n if top_n is not None else _requested_n(question, int(tool["default_n"]))
    if isinstance(requested_n, bool) or not int(tool["min_n"]) <= requested_n <= int(tool["max_n"]):
        raise ValueError(f"n must be from {tool['min_n']} to {tool['max_n']}")
    workspace = SakilaExplorationWorkspace(SakilaDB(resolved), requested_n)
    expected_table_count = int(config["loop"]["expected_table_count"])
    if len(workspace.table_names) != expected_table_count:
        raise RuntimeError(
            f"Level 5 expected {expected_table_count} Sakila tables, "
            f"found {len(workspace.table_names)}"
        )
    required_iterations = len(workspace.table_names) + 1
    if required_iterations > int(config["loop"]["max_iterations"]):
        raise RuntimeError(
            f"Level 5 budget {config['loop']['max_iterations']} cannot cover "
            f"{len(workspace.table_names)} tables plus calculation"
        )
    if offline:
        history = _run_offline_loop(workspace, skill, fault=fault)
        tool_calls = _offline_tool_calls(history)
        runner_id = None
        session_id = None
    else:
        live_timeout = float(config["harness"]["timeout_seconds"])
        try:
            async with asyncio.timeout(live_timeout):
                history, tool_calls, runner_id, session_id = await _run_live_loop(
                    workspace,
                    resolved,
                    manifest,
                    skill,
                    bundle,
                    timeout_seconds=live_timeout,
                )
        except TimeoutError as exc:
            raise RuntimeError(
                f"Level 5 live Ralph loop exceeded its {live_timeout:g}s wall-clock budget"
            ) from exc
    selected = workspace.revenue_workspace.selected
    if selected is None:
        raise RuntimeError("Ralph loop stopped without a revenue result")
    rows = selected.to_dicts()
    ranking = "；".join(
        f"{index}. {row['name']} ({float(row['revenue']):.2f})"
        for index, row in enumerate(rows, start=1)
    )
    assets = load_baseline_assets(resolved)
    evaluation = evaluate_level5_run(
        workspace,
        history,
        tool_calls,
        manifest,
        skill,
        max_iterations=int(config["loop"]["max_iterations"]),
        connected=not offline,
    )
    if not evaluation.passed:
        raise RuntimeError(f"Level 5 Ralph harness rejected the run: {evaluation.feedback}")
    trace = _trace(history, connected=not offline)
    return LevelResult(
        level=5,
        driver="Loop Agent：Ralph feedback loop + OmniGenT client + Codex harness",
        question=question,
        answer=f"Ralph loop 检查全部 16 张表后满足停止条件：{ranking}",
        sql=assets.sql,
        row_count=selected.height,
        trace=trace,
        metadata={
            "iterations": [item.model_dump(mode="json") for item in history],
            "repairs": workspace.repairs,
            "fault_scenario": fault,
            "inspected_tables": workspace.inspected_tables,
            "table_count": len(workspace.table_names),
            "table_rows": {
                name: int(observation["row_count"])
                for name, observation in workspace.observations.items()
            },
            "table_observations": workspace.observations,
            "revenue_chain": workspace.discovered_revenue_chain,
            "all_tables_inspected": workspace.all_tables_inspected,
            "iterations_used": len(history),
            "max_iterations": int(config["loop"]["max_iterations"]),
            "terminated": True,
            "stop_reason": "goal_satisfied",
            "evaluation": evaluation.model_dump(mode="json"),
            "rows": rows,
            "joined_rows": workspace.revenue_workspace.joined_rows,
            "revenue_rows": workspace.revenue_workspace.revenue.height,
            "mode": ("offline-simulated-omnigent-codex" if offline else "omnigent-codex-ollama"),
            "omnigent_connected": not offline,
            "omnigent_url": resolved.omnigent_url if not offline else None,
            "omnigent_runner_id": runner_id,
            "omnigent_session_id": session_id,
            "omnigent_tool_calls": [call.model_dump(mode="json") for call in tool_calls],
            "harness": "omnigent-client",
            "harness_executor": "codex",
            "generated_agents": [_WORKER_NAMES[name] for name in _WORKER_ORDER],
            "runtime_tools": list(_WORKER_ORDER),
            "agent_calls": [item.action.runtime_tool for item in history],
            "skill_name": skill.name,
            "skill_source": str(skill.root / "SKILL.md"),
            "skill_references": list(skill.references),
            "skill_digest_sha256": skill.digest_sha256,
            "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
            "definition_source": str(definition_path),
            "codex_subagent_calls": (len(tool_calls) + sum(len(repair.get("tool_calls", []))
                                     for repair in workspace.repairs)) if not offline else 0,
            "sandbox_intent": "read-only",
            "sandbox_backend_declared": manifest.tools["inspect_table"].os_env.sandbox.type,
            "sandbox_write_paths_declared": manifest.tools[
                "inspect_table"
            ].os_env.sandbox.write_paths,
            "sandbox_write_files_declared": manifest.tools[
                "inspect_table"
            ].os_env.sandbox.write_files,
            "sandbox_allow_network_declared": manifest.tools[
                "inspect_table"
            ].os_env.sandbox.allow_network,
            "sandbox_start_in_scratch_declared": manifest.tools[
                "inspect_table"
            ].os_env.start_in_scratch,
            "approval_policy_declared": "never",
            "protocols": sorted({message.protocol.value for message in trace}),
        },
    )


def run(
    question: str,
    *,
    settings: Settings | None = None,
    offline: bool = False,
    top_n: int | None = None,
    fault: str = "none",
) -> LevelResult:
    return _run_coroutine_sync(arun(question, settings=settings, offline=offline, top_n=top_n, fault=fault))
