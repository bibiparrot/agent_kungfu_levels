from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Settings
from .contracts import (
    AgentProposal,
    ConflictDecision,
    EvalResult,
    LoopIteration,
    Protocol,
    ProtocolMessage,
)
from .database import SakilaDB, choose_catalog_query
from .evaluation import evaluate_sql
from .llm import OllamaLLM, extract_sql


@dataclass(slots=True)
class ProtocolBus:
    messages: list[ProtocolMessage] = field(default_factory=list)

    def send(
        self,
        protocol: Protocol,
        sender: str,
        recipient: str,
        kind: str,
        payload: dict[str, Any],
        *,
        correlation_id: str | None = None,
    ) -> ProtocolMessage:
        message = ProtocolMessage(
            protocol=protocol,
            sender=sender,
            recipient=recipient,
            kind=kind,
            payload=payload,
            correlation_id=correlation_id,
        )
        self.messages.append(message)
        return message


@dataclass(slots=True)
class CodexReviewer:
    settings: Settings
    timeout_seconds: float = 45.0

    def review(self, question: str, sql: str, schema: str) -> AgentProposal:
        """Review SQL through the stable ``openai-codex`` Python SDK.

        ``model_provider='ollama'`` is the Codex local-provider protocol; the
        thread is read-only and ephemeral because this demo only reviews SQL.
        """

        from openai_codex import Codex, CodexConfig, Sandbox

        config = CodexConfig(
            cwd=str(self.settings.db_path.parent.parent),
            env={
                "OPENAI_BASE_URL": self.settings.ollama_base_url,
                "OPENAI_API_KEY": self.settings.ollama_api_key,
            },
        )
        relevant_schema = _relevant_schema(schema, sql)
        prompt = (
            "Review this Sakila SQL for correctness and read-only safety. Return only the final "
            "SELECT/CTE SQL, without Markdown. Never execute commands or edit files.\n"
            f"Question: {question}\nRelevant schema:\n{relevant_schema}\nSQL:\n{sql}"
        )
        with Codex(config) as codex:
            thread = codex.thread_start(
                model=self.settings.ollama_model,
                model_provider="ollama",
                sandbox=Sandbox.read_only,
                ephemeral=True,
            )
            handle = thread.turn(prompt, effort="low")
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="codex-review")
            future = executor.submit(handle.run)
            try:
                result = future.result(timeout=self.timeout_seconds)
                reviewed = extract_sql(result.final_response)
                rationale = "Reviewed over Codex app-server JSON-RPC using the Ollama provider."
                confidence = 0.8
            except (FutureTimeoutError, ValueError, RuntimeError) as exc:
                handle.interrupt()
                try:
                    future.result(timeout=10)
                except Exception:
                    pass
                reviewed = sql
                rationale = f"Codex reviewer fallback kept the guarded proposal: {type(exc).__name__}"
                confidence = 0.55
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        return AgentProposal(
            agent="codex-reviewer",
            sql=reviewed,
            rationale=rationale,
            confidence=confidence,
            protocol=Protocol.CODEX_JSONRPC,
        )


@dataclass(slots=True)
class OmnigentObserver:
    """Optional Omnigent HTTP observer; no server is required for core demos."""

    base_url: str | None
    agent_name: str = "sakila-governor"

    async def publish(self, event: dict[str, Any]) -> bool:
        if not self.base_url:
            return False
        from omnigent_client import OmnigentClient

        async with OmnigentClient(self.base_url) as client:
            session = client.session(self.agent_name)
            await session.send(json.dumps(event, ensure_ascii=False))
        return True


@dataclass(slots=True)
class ConflictResolver:
    db: SakilaDB

    def resolve(self, proposals: list[AgentProposal], question: str, *,
                reference_sql: str | None = None) -> ConflictDecision:
        if not proposals:
            raise ValueError("At least one proposal is required")
        scores: dict[str, float] = {}
        conflicts: list[str] = []
        normalized = {" ".join(item.sql.lower().split()) for item in proposals}
        if len(normalized) > 1:
            conflicts.append("Agents proposed different SQL plans")
        reference = self.db.query(reference_sql) if reference_sql else None
        eligible = []
        for proposal in proposals:
            evaluation = evaluate_sql(self.db, proposal.sql, question)
            accepted = evaluation.passed
            if accepted and reference is not None:
                actual = self.db.query(proposal.sql)
                accepted = actual.columns == reference.columns and actual.sort(actual.columns).equals(
                    reference.sort(reference.columns)
                )
                if not accepted:
                    conflicts.append(f"{proposal.agent}: result disagrees with reference evidence")
            if not accepted:
                scores[proposal.agent] = 0.0
                continue
            eligible.append(proposal)
            scores[proposal.agent] = round(0.7 * evaluation.score + 0.3 * proposal.confidence, 4)
        if not eligible:
            raise ValueError("No eligible proposal passed the evidence gate")
        winner = max(eligible, key=lambda item: scores[item.agent])
        return ConflictDecision(
            selected_agent=winner.agent,
            selected_sql=winner.sql,
            scores=scores,
            conflicts=conflicts,
            reason="Selected the safest executable proposal, then used declared confidence as tie-breaker.",
        )


@dataclass(slots=True)
class RalphLoop:
    db: SakilaDB
    max_iterations: int = 3

    def run(
        self,
        question: str,
        propose: Callable[[int, EvalResult | None], AgentProposal],
    ) -> list[LoopIteration]:
        history: list[LoopIteration] = []
        previous: EvalResult | None = None
        for index in range(1, self.max_iterations + 1):
            proposal = propose(index, previous)
            evaluation = evaluate_sql(self.db, proposal.sql, question)
            history.append(LoopIteration(iteration=index, proposal=proposal, evaluation=evaluation))
            if evaluation.passed:
                break
            previous = evaluation
        return history


def prompt_sql_proposal(llm: OllamaLLM, db: SakilaDB, question: str) -> AgentProposal:
    response = llm.complete(
        f"Question: {question}\nSQLite schema:\n{db.schema_text()}\nReturn only one read-only SELECT query.",
        system="You are a Sakila SQL planner. Use only columns present in the schema.",
    )
    return AgentProposal(
        agent="prompt-planner",
        sql=extract_sql(response),
        rationale="Generated by LiteLLM through Ollama's OpenAI-compatible endpoint.",
        confidence=0.65,
        protocol=Protocol.PROMPT,
    )


def offline_proposal(question: str, *, agent: str = "offline-skill") -> AgentProposal:
    return AgentProposal(
        agent=agent,
        sql=choose_catalog_query(question),
        rationale="Deterministic reviewed query used for offline learning and tests.",
        confidence=0.9,
        protocol=Protocol.SKILL,
    )


def _relevant_schema(schema: str, sql: str) -> str:
    """Keep only DDL blocks for tables named in SQL to reduce local review latency."""

    tokens = set(re.findall(r"\b[a-z_][a-z0-9_]*\b", sql.lower()))
    blocks = []
    for block in schema.split("\n\n"):
        header = re.match(r"--\s+([a-z_][a-z0-9_]*)", block.strip(), re.IGNORECASE)
        if header and header.group(1).lower() in tokens:
            blocks.append(block)
    return "\n\n".join(blocks) if blocks else schema
