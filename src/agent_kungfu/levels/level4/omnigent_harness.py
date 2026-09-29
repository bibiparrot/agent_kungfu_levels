from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import math
import re
import tarfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Coroutine, Literal, Mapping, TypeVar

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...config import Settings
from ...contracts import LevelResult, Protocol
from ...database import SakilaDB
from ...orchestration import ProtocolBus
from ..common import load_baseline_assets, load_level_config, resolve_from, resolved_settings
from ..level1.workflow import _requested_n


_REQUIRED_TABLES = {"category", "film_category", "inventory", "rental", "payment"}
_REQUIRED_REFERENCES = (
    "references/contexts.md",
    "references/expressions.md",
    "references/insight-recipes.md",
    "references/lazy-api.md",
    "references/pandas-to-polars.md",
)
_WORKER_ORDER = ("read_tables", "calculate_revenue")
_WORKER_NAMES = {
    "read_tables": "Polars Table Reader",
    "calculate_revenue": "Polars Revenue Calculator",
}
_ASCII_TYPOGRAPHY = str.maketrans(
    {
        "—": "--",
        "–": "-",
        "→": "->",
        "×": "x",
        "÷": "/",
        "…": "...",
    }
)
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class LoadedSkill:
    root: Path
    name: str
    description: str
    instructions: str
    references: Mapping[str, str]
    digest_sha256: str

    def instructions_for(self, runtime_tool: str) -> str:
        selected = {
            "read_tables": ("references/lazy-api.md", "references/contexts.md"),
            "calculate_revenue": (
                "references/contexts.md",
                "references/expressions.md",
                "references/insight-recipes.md",
            ),
        }
        if runtime_tool not in selected:
            raise ValueError(f"Unknown runtime tool: {runtime_tool}")
        sections = [self.instructions]
        sections.extend(self.references[name] for name in selected[runtime_tool])
        return "\n\n---\n\n".join(sections)


class CodexExecutor(BaseModel):
    model_config = ConfigDict(extra="allow")

    harness: Literal["codex"]
    model: str


class CodexSandboxConfig(BaseModel):
    """Portable, least-privilege sandbox declaration for an OmniGenT Codex agent."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["auto"] = "auto"
    read_paths: list[str] = Field(default_factory=lambda: ["."])
    write_paths: list[str] = Field(default_factory=list)
    write_files: list[str] = Field(default_factory=list)
    allow_network: Literal[False] = False

    @model_validator(mode="after")
    def validate_read_only_contract(self) -> "CodexSandboxConfig":
        if self.read_paths != ["."]:
            raise ValueError("Level 4 Codex agents may read only their scoped workspace")
        if self.write_paths or self.write_files:
            raise ValueError("Level 4 Codex agents must not declare writable paths")
        return self


class CodexOSEnvConfig(BaseModel):
    """OS-environment declaration passed by OmniGenT to its Codex harness."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["caller_process"] = "caller_process"
    cwd: Literal["."] = "."
    sandbox: CodexSandboxConfig
    fork: Literal[False] = False
    start_in_scratch: Literal[True] = True


class InlineAgentTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["agent"]
    description: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    executor: CodexExecutor
    os_env: CodexOSEnvConfig
    pass_history: bool = False
    max_sessions: int = Field(default=1, ge=1, le=1)


class TopLevelExecutorConfig(BaseModel):
    harness: Literal["codex"]


class TopLevelExecutor(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["omnigent"]
    model: str
    config: TopLevelExecutorConfig


class RootToolsConfig(BaseModel):
    agents: list[str]


class RootBundleManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    spec_version: int = 1
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    skills: list[str]
    os_env: CodexOSEnvConfig
    executor: TopLevelExecutor
    prompt: str = Field(min_length=1)
    tools: RootToolsConfig


class SubagentBundleManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    spec_version: int = 1
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    skills: list[str]
    os_env: CodexOSEnvConfig
    executor: TopLevelExecutor
    prompt: str = Field(min_length=1)


class OmnigentAgentManifest(BaseModel):
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
    def validate_level4_contract(self) -> "OmnigentAgentManifest":
        if self.skills != ["polars"]:
            raise ValueError("Level 4 must load exactly the bundled polars skill")
        if tuple(self.tools) != _WORKER_ORDER:
            raise ValueError("Level 4 must declare read_tables then calculate_revenue")
        return self


@dataclass(frozen=True, slots=True)
class CodexWorkerDefinition:
    id: str
    name: str
    prompt: str
    base_instructions: str
    skill_digest_sha256: str
    sandbox: str = "read-only"
    sandbox_backend: str = "auto"
    write_paths: tuple[str, ...] = ()
    write_files: tuple[str, ...] = ()
    allow_network: bool = False
    start_in_scratch: bool = True
    approval_policy: str = "never"
    session_scope: str = "omnigent-session"


class WorkerAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: Literal["read_tables", "calculate_revenue"]
    runtime_tool: Literal["read_tables", "calculate_revenue"]
    n: int | None
    skill_digest_sha256: str = Field(min_length=64, max_length=64)
    summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_action(self) -> "WorkerAction":
        if self.agent != self.runtime_tool:
            raise ValueError("Agent id and runtime tool must match")
        if self.runtime_tool == "read_tables" and self.n is not None:
            raise ValueError("read_tables requires n=null")
        if self.runtime_tool == "calculate_revenue" and (
            self.n is None or isinstance(self.n, bool) or self.n < 1
        ):
            raise ValueError("calculate_revenue requires a positive integer n")
        return self


class PlanEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    feedback: list[str] = Field(default_factory=list)


class OmnigentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[WorkerAction]
    evaluation: PlanEvaluation


class OmnigentToolCall(BaseModel):
    """Auditable evidence that OmniGenT delegated to a bundled Codex subagent."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["read_tables", "calculate_revenue"]
    call_id: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    executed_by: Literal["server", "simulated"]


class HarnessCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class HarnessEvaluation(BaseModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    checks: list[HarnessCheck]
    feedback: list[str]
    evaluator: str = "deterministic-harness"


@dataclass(slots=True)
class PolarsHarnessWorkspace:
    db: SakilaDB
    queries: dict[str, str]
    minimum_n: int
    maximum_n: int
    tables: dict[str, pl.DataFrame] = field(default_factory=dict)
    table_rows: dict[str, int] = field(default_factory=dict)
    revenue: pl.DataFrame | None = None
    selected: pl.DataFrame | None = None
    joined_rows: int = 0
    events: list[str] = field(default_factory=list)

    def load_tables(self) -> dict[str, Any]:
        if self.tables:
            raise RuntimeError("read_tables may run only once")
        if set(self.queries) != _REQUIRED_TABLES:
            raise ValueError(f"Expected exactly these Sakila tables: {sorted(_REQUIRED_TABLES)}")
        self.tables = {name: self.db.query(sql) for name, sql in self.queries.items()}
        self.table_rows = {name: frame.height for name, frame in self.tables.items()}
        self.events.append("read_tables")
        return {"status": "tables_loaded", "rows": self.table_rows}

    def calculate_revenue(self, n: int) -> dict[str, Any]:
        if not self.tables:
            raise RuntimeError("read_tables must complete before calculate_revenue")
        if self.revenue is not None:
            raise RuntimeError("calculate_revenue may run only once")
        if (
            isinstance(n, bool)
            or not isinstance(n, int)
            or not self.minimum_n <= n <= self.maximum_n
        ):
            raise ValueError(f"n must be an integer from {self.minimum_n} to {self.maximum_n}")

        category = self.tables["category"].lazy().select("category_id", "name")
        film_category = self.tables["film_category"].lazy().select("category_id", "film_id")
        inventory = self.tables["inventory"].lazy().select("film_id", "inventory_id")
        rental = self.tables["rental"].lazy().select("inventory_id", "rental_id")
        payment = self.tables["payment"].lazy().select("rental_id", "amount")

        # One optimized lazy plan and one collect: category event counts are summed after
        # collection to expose joined cardinality without collecting the intermediate join.
        summary = (
            category.join(film_category, on="category_id", how="inner")
            .join(inventory, on="film_id", how="inner")
            .join(rental, on="inventory_id", how="inner")
            .join(payment, on="rental_id", how="inner")
            .group_by("name")
            .agg(
                pl.col("amount").sum().round(2).alias("revenue"),
                pl.len().alias("_joined_rows"),
            )
            .sort(["revenue", "name"], descending=[True, False])
            .collect()
        )
        self.joined_rows = int(summary.get_column("_joined_rows").sum())
        self.revenue = summary.drop("_joined_rows")
        self.selected = self.revenue.head(n)
        self.events.append("calculate_revenue")
        return {
            "status": "revenue_calculated",
            "joined_rows": self.joined_rows,
            "category_rows": self.revenue.height,
            "top_n": n,
            "rows": self.selected.to_dicts(),
        }


def _frontmatter(text: str, *, source: Path) -> tuple[dict[str, Any], str]:
    match = re.match(r"\A---\s*\n(.*?)\n---(?:\s*\n|\Z)(.*)\Z", text, re.DOTALL)
    if not match:
        raise ValueError(f"Missing YAML frontmatter in {source}")
    raw = yaml.safe_load(match.group(1))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {source}")
    body = match.group(2).strip()
    if not body:
        raise ValueError(f"Skill instructions are empty in {source}")
    return raw, body


def load_polars_skill(path: Path | None = None) -> LoadedSkill:
    root = (path or Path(__file__).with_name("polars")).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Polars skill path is not a directory: {root}")
    skill_path = root / "SKILL.md"
    if not skill_path.is_file():
        raise FileNotFoundError(f"Missing Polars SKILL.md: {skill_path}")
    skill_text = skill_path.read_text(encoding="utf-8")
    metadata, body = _frontmatter(skill_text, source=skill_path)
    name = str(metadata.get("name", ""))
    description = str(metadata.get("description", "")).strip()
    if name != "polars" or name != root.name:
        raise ValueError("Polars skill name must match its directory")
    if not description:
        raise ValueError("Polars skill description must not be empty")

    payloads: dict[str, str] = {"SKILL.md": skill_text}
    references: dict[str, str] = {}
    for relative in _REQUIRED_REFERENCES:
        candidate = root / relative
        resolved = candidate.resolve(strict=True)
        if candidate.is_symlink() or not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(f"Unsafe Polars skill reference: {relative}")
        content = resolved.read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError(f"Empty Polars skill reference: {relative}")
        payloads[relative] = content
        references[relative] = content

    digest = hashlib.sha256()
    for relative in sorted(payloads):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payloads[relative].encode("utf-8"))
        digest.update(b"\0")
    return LoadedSkill(
        root=root,
        name=name,
        description=description,
        instructions=body,
        references=MappingProxyType(references),
        digest_sha256=digest.hexdigest(),
    )


def load_omnigent_manifest(path: Path | None = None) -> OmnigentAgentManifest:
    manifest_path = path or Path(__file__).with_name("omnigent-agent") / "config.yaml"
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected an OmniGenT YAML mapping in {manifest_path}")
    root = RootBundleManifest.model_validate(raw)
    if root.skills != ["polars"] or tuple(root.tools.agents) != _WORKER_ORDER:
        raise ValueError("The OmniGenT root must expose Reader then Calculator with polars")
    tools: dict[str, InlineAgentTool] = {}
    for agent_id in root.tools.agents:
        child_path = manifest_path.parent / "agents" / agent_id / "config.yaml"
        child_raw = yaml.safe_load(child_path.read_text(encoding="utf-8"))
        if not isinstance(child_raw, dict):
            raise ValueError(f"Expected an OmniGenT YAML mapping in {child_path}")
        child = SubagentBundleManifest.model_validate(child_raw)
        if child.name != agent_id or child.skills != ["polars"]:
            raise ValueError(f"Invalid Polars subagent contract in {child_path}")
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
    return OmnigentAgentManifest(
        spec_version=root.spec_version,
        name=root.name,
        description=root.description,
        skills=root.skills,
        os_env=root.os_env,
        executor=root.executor,
        prompt=root.prompt,
        tools=tools,
    )


def build_codex_workers(
    manifest: OmnigentAgentManifest,
    skill: LoadedSkill,
) -> tuple[CodexWorkerDefinition, CodexWorkerDefinition]:
    worker_items = [
        CodexWorkerDefinition(
            id=runtime_tool,
            name=_WORKER_NAMES[runtime_tool],
            prompt=manifest.tools[runtime_tool].prompt,
            base_instructions=skill.instructions_for(runtime_tool),
            skill_digest_sha256=skill.digest_sha256,
            sandbox_backend=manifest.tools[runtime_tool].os_env.sandbox.type,
            write_paths=tuple(manifest.tools[runtime_tool].os_env.sandbox.write_paths),
            write_files=tuple(manifest.tools[runtime_tool].os_env.sandbox.write_files),
            allow_network=manifest.tools[runtime_tool].os_env.sandbox.allow_network,
            start_in_scratch=manifest.tools[runtime_tool].os_env.start_in_scratch,
        )
        for runtime_tool in manifest.tools
    ]
    if tuple(worker.id for worker in worker_items) != _WORKER_ORDER:
        raise ValueError("OmniGenT must generate exactly the Reader then Calculator")
    return worker_items[0], worker_items[1]


def build_agent_bundle(
    manifest_path: Path | None = None,
    skill: LoadedSkill | None = None,
) -> bytes:
    source_manifest = manifest_path or Path(__file__).with_name("omnigent-agent") / "config.yaml"
    loaded_skill = skill or load_polars_skill()
    entries = {"config.yaml": source_manifest.read_bytes()}
    source_root = source_manifest.parent
    for worker_id in _WORKER_ORDER:
        relative_config = f"agents/{worker_id}/config.yaml"
        entries[relative_config] = (source_root / relative_config).read_bytes()
    skill_scopes = ["skills/polars"] + [
        f"agents/{worker_id}/skills/polars" for worker_id in _WORKER_ORDER
    ]
    for scope in skill_scopes:
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


def _portable_skill_bytes(text: str, *, source: str) -> bytes:
    """Keep the bundle readable by OmniGenT 0.11's locale-based Windows parser."""

    portable = text.translate(_ASCII_TYPOGRAPHY)
    try:
        return portable.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(
            f"Unsupported non-ASCII typography in bundled Polars skill file {source}"
        ) from exc


def _offline_plan(skill: LoadedSkill, requested_n: int) -> OmnigentPlan:
    return OmnigentPlan(
        actions=[
            WorkerAction(
                agent="read_tables",
                runtime_tool="read_tables",
                n=None,
                skill_digest_sha256=skill.digest_sha256,
                summary="Offline harness selected the allowlisted table-read action.",
            ),
            WorkerAction(
                agent="calculate_revenue",
                runtime_tool="calculate_revenue",
                n=requested_n,
                skill_digest_sha256=skill.digest_sha256,
                summary="Offline harness selected the allowlisted lazy-Polars calculation.",
            ),
        ],
        evaluation=PlanEvaluation(
            passed=True,
            feedback=["The deterministic plan matches the two-worker Level 4 contract."],
        ),
    )


def _offline_tool_calls() -> list[OmnigentToolCall]:
    return [
        OmnigentToolCall(
            name=name,
            call_id=f"offline-{name}",
            agent_name="sakila-level4-harness",
            executed_by="simulated",
        )
        for name in _WORKER_ORDER
    ]


def _parse_plan(text: str) -> OmnigentPlan:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
    try:
        return OmnigentPlan.model_validate_json(clean)
    except Exception as first_error:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            raise ValueError("OmniGenT did not return a JSON plan") from first_error
        return OmnigentPlan.model_validate_json(match.group(0))


def _validate_plan(plan: OmnigentPlan, skill: LoadedSkill, requested_n: int) -> None:
    if not plan.evaluation.passed:
        raise RuntimeError("OmniGenT rejected its two-worker action plan")
    if tuple(action.runtime_tool for action in plan.actions) != _WORKER_ORDER:
        raise ValueError("OmniGenT must return exactly Reader then Calculator actions")
    if any(action.skill_digest_sha256 != skill.digest_sha256 for action in plan.actions):
        raise ValueError("OmniGenT action used an unexpected Polars skill digest")
    if plan.actions[1].n != requested_n:
        raise ValueError(
            f"OmniGenT calculator returned n={plan.actions[1].n}; expected {requested_n}"
        )


async def _live_plan(
    settings: Settings,
    manifest: OmnigentAgentManifest,
    skill: LoadedSkill,
    requested_n: int,
) -> tuple[OmnigentPlan, list[OmnigentToolCall], str]:
    if not settings.omnigent_url:
        raise RuntimeError(
            "Level 4 live mode requires OMNIGENT_URL; use offline=True for the deterministic lesson"
        )
    from omnigent_client import OmnigentClient, StreamHooks

    prompt = (
        "Generate and evaluate exactly two host actions. Call read_tables first and "
        "calculate_revenue second. Return only JSON with this shape: "
        '{"actions":[{"agent":"read_tables","runtime_tool":"read_tables","n":null,'
        '"skill_digest_sha256":"...","summary":"..."},{"agent":"calculate_revenue",'
        '"runtime_tool":"calculate_revenue","n":N,"skill_digest_sha256":"...",'
        '"summary":"..."}],"evaluation":{"passed":true,"feedback":["..."]}}. '
        f"N={requested_n}. The exact skill digest is {skill.digest_sha256}. "
        "Do not execute the host actions and do not return database rows."
    )
    bundle = build_agent_bundle(skill=skill)
    observed_calls: list[OmnigentToolCall] = []

    def record_tool_call(context: Any) -> None:
        try:
            observed_calls.append(
                OmnigentToolCall(
                    name=context.name,
                    call_id=context.call_id,
                    agent_name=context.agent_name,
                    executed_by=context.executed_by,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "OmniGenT emitted a tool call outside the Level 4 subagent contract"
            ) from exc

    hooks = StreamHooks(on_tool_call_start=record_tool_call)
    async with OmnigentClient(
        settings.omnigent_url,
        timeout=float(load_level_config(4)["harness"]["timeout_seconds"]),
    ) as client:
        runner_id = settings.omnigent_runner_id or await client.sessions.resolve_online_runner(
            harness=manifest.executor.config.harness
        )
        if not runner_id:
            raise RuntimeError(
                "Level 4 live mode requires an online Codex runner; set "
                "OMNIGENT_RUNNER_ID or connect a runner to the OmniGenT server"
            )
        chat = await client.sessions_chat(
            bundle=bundle,
            filename=f"{manifest.name}.tar.gz",
            hooks=hooks,
        )
        await client.sessions.bind_runner(chat.session_id, runner_id=runner_id)
        result = await chat.query(prompt)
    if [call.name for call in observed_calls] != list(_WORKER_ORDER):
        raise RuntimeError(
            "OmniGenT must invoke exactly read_tables then calculate_revenue once each"
        )
    if any(call.executed_by != "server" for call in observed_calls):
        raise RuntimeError("Level 4 accepts only server-executed OmniGenT subagent calls")
    return _parse_plan(result.text), observed_calls, runner_id


def _execute_plan(
    plan: OmnigentPlan,
    workers: tuple[CodexWorkerDefinition, CodexWorkerDefinition],
    workspace: PolarsHarnessWorkspace,
    *,
    requested_n: int,
    simulated: bool,
) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for worker, action in zip(workers, plan.actions, strict=True):
        if action.runtime_tool == "read_tables":
            evidence = workspace.load_tables()
        elif action.runtime_tool == "calculate_revenue":
            if action.n != requested_n:
                raise ValueError("Calculator n changed after plan validation")
            evidence = workspace.calculate_revenue(requested_n)
        else:  # pragma: no cover - Pydantic and Literal make this unreachable.
            raise ValueError(f"Unknown host action: {action.runtime_tool}")
        turns.append(
            {
                "agent": worker.name,
                "agent_id": worker.id,
                "runtime_tool": action.runtime_tool,
                "skill_digest_sha256": action.skill_digest_sha256,
                "sandbox": worker.sandbox,
                "sandbox_backend": worker.sandbox_backend,
                "write_paths": list(worker.write_paths),
                "write_files": list(worker.write_files),
                "allow_network": worker.allow_network,
                "start_in_scratch": worker.start_in_scratch,
                "approval_policy": worker.approval_policy,
                "session_scope": worker.session_scope,
                "status": "simulated" if simulated else "completed",
                "simulated": simulated,
                "summary": action.summary,
                "evidence": evidence,
            }
        )
    return turns


def evaluate_level4_run(
    skill: LoadedSkill,
    workers: tuple[CodexWorkerDefinition, CodexWorkerDefinition],
    workspace: PolarsHarnessWorkspace,
    turns: list[dict[str, Any]],
    omnigent_tool_calls: list[OmnigentToolCall],
    *,
    requested_n: int,
) -> HarnessEvaluation:
    if workspace.revenue is None or workspace.selected is None:
        raise RuntimeError("Cannot evaluate an incomplete Level 4 workspace")
    baseline_assets = load_baseline_assets(workspace.db.settings)
    baseline = workspace.db.query(baseline_assets.sql)
    revenues = [float(value) for value in workspace.selected.get_column("revenue").to_list()]

    conditions = [
        (
            "omnigent-delegation",
            [call.name for call in omnigent_tool_calls] == list(_WORKER_ORDER)
            and all(
                call.executed_by == omnigent_tool_calls[0].executed_by
                for call in omnigent_tool_calls
            )
            and omnigent_tool_calls[0].executed_by in {"server", "simulated"},
            "OmniGenT delegated exactly once to Reader then Calculator.",
        ),
        (
            "exactly-two-workers",
            len(workers) == 2 and tuple(worker.id for worker in workers) == _WORKER_ORDER,
            "Reader and Calculator are the only generated agents.",
        ),
        (
            "skill-provenance",
            len(turns) == 2
            and all(turn.get("skill_digest_sha256") == skill.digest_sha256 for turn in turns),
            "Both actions acknowledge the validated Polars skill digest.",
        ),
        (
            "declared-codex-boundary",
            len(turns) == 2
            and all(
                turn.get("sandbox") == "read-only"
                and turn.get("sandbox_backend") == "auto"
                and turn.get("write_paths") == []
                and turn.get("write_files") == []
                and turn.get("allow_network") is False
                and turn.get("start_in_scratch") is True
                and turn.get("approval_policy") == "never"
                and turn.get("session_scope") == "omnigent-session"
                for turn in turns
            ),
            "The bundle declares an active auto sandbox, no writes/network, scratch start, "
            "approval-never, and session scope for both Codex subagents.",
        ),
        (
            "action-order",
            workspace.events == list(_WORKER_ORDER),
            "Host execution is read_tables followed by calculate_revenue.",
        ),
        (
            "five-tables",
            set(workspace.tables) == _REQUIRED_TABLES
            and all(rows > 0 for rows in workspace.table_rows.values()),
            "All five required Sakila tables were loaded once.",
        ),
        (
            "top-n-shape",
            workspace.selected.height == requested_n
            and workspace.selected.get_column("name").n_unique() == requested_n,
            f"The result contains {requested_n} unique categories.",
        ),
        (
            "revenue-order-and-domain",
            revenues == sorted(revenues, reverse=True)
            and all(math.isfinite(value) and value >= 0 for value in revenues),
            "Revenue is finite, non-negative, and descending.",
        ),
        (
            "lazy-join-cardinality",
            workspace.joined_rows == 16044 and workspace.revenue.height == 16,
            "The one-collect lazy plan produced 16 categories from 16,044 joined rows.",
        ),
        (
            "baseline-parity",
            workspace.revenue.to_dicts() == baseline.to_dicts(),
            "The full Polars result matches the independent SQL baseline to cents.",
        ),
    ]
    checks = [
        HarnessCheck(name=name, passed=passed, detail=detail) for name, passed, detail in conditions
    ]
    score = sum(check.passed for check in checks) / len(checks)
    feedback = [check.detail for check in checks if not check.passed]
    return HarnessEvaluation(
        passed=all(check.passed for check in checks),
        score=round(score, 4),
        checks=checks,
        feedback=feedback,
    )


def _trace(
    manifest: OmnigentAgentManifest,
    turns: list[dict[str, Any]],
    evaluation: HarnessEvaluation,
    *,
    omnigent_connected: bool,
) -> list[Any]:
    bus = ProtocolBus()
    request = bus.send(
        Protocol.A2A,
        "user",
        manifest.name,
        "analysis.request",
        {"goal": "top category revenue"},
    )
    bus.send(
        Protocol.OMNIGENT_HTTP if omnigent_connected else Protocol.SKILL,
        manifest.name,
        "polars",
        "skill.loaded",
        {"connected": omnigent_connected},
        correlation_id=request.id,
    )
    for turn in turns:
        bus.send(
            Protocol.A2A,
            manifest.name,
            str(turn["agent"]),
            "task.delegate",
            {"runtime_tool": turn["runtime_tool"]},
            correlation_id=request.id,
        )
        bus.send(
            Protocol.CODEX_JSONRPC,
            str(turn["agent"]),
            "codex-harness",
            "turn.simulated" if turn["simulated"] else "turn.completed",
            {
                "runtime_tool": turn["runtime_tool"],
                "status": turn["status"],
                "via": "omnigent" if omnigent_connected else "offline",
            },
            correlation_id=request.id,
        )
        bus.send(
            Protocol.MCP,
            str(turn["agent"]),
            manifest.name,
            "tool.result",
            {"event": turn["runtime_tool"]},
            correlation_id=request.id,
        )
    bus.send(
        Protocol.OMNIGENT_HTTP if omnigent_connected else Protocol.SKILL,
        manifest.name,
        "user",
        "harness.evaluated",
        {"passed": evaluation.passed, "score": evaluation.score},
        correlation_id=request.id,
    )
    return bus.messages


def _run_coroutine_sync(coroutine: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-kungfu-l4") as executor:
        return executor.submit(asyncio.run, coroutine).result()


async def arun(
    question: str,
    *,
    settings: Settings | None = None,
    offline: bool = False,
    top_n: int | None = None,
) -> LevelResult:
    resolved = resolved_settings(settings)
    config = load_level_config(4)
    config_path = Path(__file__).with_name("config.yaml")
    skill_path = resolve_from(str(config["skill"]["path"]), config_path=config_path)
    manifest_path = resolve_from(str(config["definition"]["path"]), config_path=config_path)
    skill = load_polars_skill(skill_path)
    manifest = load_omnigent_manifest(manifest_path)
    workers = build_codex_workers(manifest, skill)
    tool = config["tool"]
    requested_n = top_n if top_n is not None else _requested_n(question, int(tool["default_n"]))
    if isinstance(requested_n, bool) or not int(tool["min_n"]) <= requested_n <= int(tool["max_n"]):
        raise ValueError(f"n must be from {tool['min_n']} to {tool['max_n']}")

    if offline:
        plan = _offline_plan(skill, requested_n)
        omnigent_tool_calls = _offline_tool_calls()
        omnigent_runner_id = None
    else:
        plan, omnigent_tool_calls, omnigent_runner_id = await _live_plan(
            resolved,
            manifest,
            skill,
            requested_n,
        )
    _validate_plan(plan, skill, requested_n)
    level1_config = load_level_config(1)
    workspace = PolarsHarnessWorkspace(
        db=SakilaDB(resolved),
        queries=dict(level1_config["workflow"]["tables"]),
        minimum_n=int(tool["min_n"]),
        maximum_n=int(tool["max_n"]),
    )
    turns = _execute_plan(
        plan,
        workers,
        workspace,
        requested_n=requested_n,
        simulated=offline,
    )
    evaluation = evaluate_level4_run(
        skill,
        workers,
        workspace,
        turns,
        omnigent_tool_calls,
        requested_n=requested_n,
    )
    if not evaluation.passed:
        raise RuntimeError(f"Level 4 harness rejected the run: {evaluation.feedback}")
    if workspace.selected is None or workspace.revenue is None:
        raise RuntimeError("Level 4 did not produce a selected revenue frame")

    rows = workspace.selected.to_dicts()
    prefix = (
        "离线 OmniGenT harness 模拟的两个 Codex 子智能体"
        if offline
        else "OmniGenT 通过 Codex harness 生成的两个 Polars 子智能体"
    )
    ranking = "；".join(
        f"{index}. {row['name']} ({float(row['revenue']):.2f})"
        for index, row in enumerate(rows, start=1)
    )
    bundle = build_agent_bundle(manifest_path, skill)
    trace = _trace(manifest, turns, evaluation, omnigent_connected=not offline)
    assets = load_baseline_assets(resolved)
    from ...contracts import AgentProposal
    from ...orchestration import ConflictResolver

    reference = assets.sql.rstrip(";")
    conflict = ConflictResolver(workspace.db).resolve([
        AgentProposal(agent="injected-confident-error", confidence=1,
                      sql=f"SELECT name, revenue * 2 AS revenue FROM ({reference})"),
        AgentProposal(agent="reference-backed", confidence=0, sql=reference),
    ], question, reference_sql=reference)
    return LevelResult(
        level=4,
        driver="Harness Agent：OmniGenT client + Codex subagents + Polars skill + evaluation",
        question=question,
        answer=f"{prefix}完成读取、计算与评估：{ranking}",
        sql=assets.sql,
        row_count=workspace.selected.height,
        trace=trace,
        metadata={
            "definition_source": str(manifest_path),
            "conflict_fixture": "injected numeric disagreement; not model-generated proposals",
            "conflict_decision": conflict.model_dump(mode="json"),
            "generated_agents": [worker.name for worker in workers],
            "runtime_tools": [worker.id for worker in workers],
            "agent_calls": workspace.events,
            "skill_name": skill.name,
            "skill_source": str(skill.root / "SKILL.md"),
            "skill_references": list(skill.references),
            "skill_digest_sha256": skill.digest_sha256,
            "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
            "harness": "omnigent-client",
            "harness_executor": "codex",
            "omnigent_connected": not offline,
            "omnigent_url": resolved.omnigent_url if not offline else None,
            "omnigent_runner_id": omnigent_runner_id,
            "omnigent_tool_calls": [
                call.model_dump(mode="json") for call in omnigent_tool_calls
            ],
            "codex_threads_started": 0 if offline else len(workers),
            "codex_turns": turns,
            "sandbox": "read-only",
            "sandbox_backend": manifest.tools["read_tables"].os_env.sandbox.type,
            "sandbox_write_paths": manifest.tools["read_tables"].os_env.sandbox.write_paths,
            "sandbox_write_files": manifest.tools["read_tables"].os_env.sandbox.write_files,
            "sandbox_allow_network": manifest.tools["read_tables"].os_env.sandbox.allow_network,
            "sandbox_start_in_scratch": manifest.tools[
                "read_tables"
            ].os_env.start_in_scratch,
            "approval_policy": "never",
            "codex_session_scope": "omnigent-session",
            "plan_evaluation": plan.evaluation.model_dump(mode="json"),
            "evaluation": evaluation.model_dump(mode="json"),
            "table_rows": workspace.table_rows,
            "joined_rows": workspace.joined_rows,
            "revenue_rows": workspace.revenue.height,
            "rows": rows,
            "protocols": sorted({message.protocol.value for message in trace}),
            "mode": ("offline-simulated-omnigent-codex" if offline else "omnigent-codex-ollama"),
        },
    )


def run(
    question: str,
    *,
    settings: Settings | None = None,
    offline: bool = False,
    top_n: int | None = None,
) -> LevelResult:
    return _run_coroutine_sync(arun(question, settings=settings, offline=offline, top_n=top_n))
