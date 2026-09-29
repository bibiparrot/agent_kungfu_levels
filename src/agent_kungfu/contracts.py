from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Protocol(StrEnum):
    PROMPT = "prompt"
    SKILL = "skill"
    MCP = "mcp"
    A2A = "a2a"
    CODEX_JSONRPC = "codex-jsonrpc"
    OMNIGENT_HTTP = "omnigent-http"


class ProtocolMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    protocol: Protocol
    sender: str
    recipient: str
    kind: str
    payload: dict[str, Any]
    correlation_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentProposal(BaseModel):
    agent: str
    sql: str
    rationale: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    protocol: Protocol = Protocol.PROMPT


class ConflictDecision(BaseModel):
    selected_agent: str
    selected_sql: str
    scores: dict[str, float]
    conflicts: list[str] = Field(default_factory=list)
    reason: str


class EvalResult(BaseModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    feedback: list[str] = Field(default_factory=list)
    evaluator: str = "deterministic"


class LoopIteration(BaseModel):
    iteration: int
    proposal: AgentProposal
    evaluation: EvalResult


class LevelResult(BaseModel):
    level: int
    driver: str
    question: str
    answer: str
    sql: str | None = None
    row_count: int | None = None
    trace: list[ProtocolMessage] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HarnessManifest(BaseModel):
    name: str
    version: str = "1"
    allowed_protocols: list[Protocol]
    read_only: bool = True
    max_rows: int = Field(default=50, ge=1, le=1000)
    max_iterations: int = Field(default=3, ge=1, le=10)
    conflict_policy: str = "safe-executable-highest-confidence"
    evidence_required: bool = True

