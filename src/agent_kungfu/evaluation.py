from __future__ import annotations

import asyncio
from dataclasses import dataclass

import polars as pl

from .contracts import EvalResult
from .database import SakilaDB, validate_read_only_sql
from .llm import OllamaLLM


def evaluate_sql(db: SakilaDB, sql: str, question: str) -> EvalResult:
    feedback: list[str] = []
    score = 0.0
    try:
        validate_read_only_sql(sql)
        score += 0.35
    except ValueError as exc:
        return EvalResult(passed=False, score=0.0, feedback=[str(exc)])

    try:
        frame = db.query(sql, limit=50)
        score += 0.35
    except Exception as exc:  # SQLite/ConnectorX supplies the actionable error.
        return EvalResult(passed=False, score=score, feedback=[f"Execution failed: {exc}"])

    if frame.height:
        score += 0.15
    else:
        feedback.append("The query returned no evidence")
    if any(col.lower() in {"revenue", "rentals", "customers", "lifetime_value"} for col in frame.columns):
        score += 0.15
    else:
        feedback.append("Result lacks a decision-oriented metric")
    if "门店" in question and "store_id" not in frame.columns:
        score = min(score, 0.74)
        feedback.append("Question asks about stores but result is not grouped by store_id")
    return EvalResult(passed=score >= 0.9, score=min(score, 1.0), feedback=feedback)


@dataclass(slots=True)
class OllamaDeepEvalModel:
    """Factory for a DeepEvalBaseLLM backed by the same local Ollama model."""

    llm: OllamaLLM

    def build(self):
        from deepeval.models import DeepEvalBaseLLM

        outer = self

        class LocalModel(DeepEvalBaseLLM):
            def load_model(self):
                return outer.llm

            def generate(self, prompt: str, *args, **kwargs) -> str:
                return outer.llm.complete(prompt, system="You are a strict evaluation judge.")

            async def a_generate(self, prompt: str, *args, **kwargs) -> str:
                return await asyncio.to_thread(self.generate, prompt)

            def get_model_name(self) -> str:
                return outer.llm.settings.ollama_model

        return LocalModel(model=outer.llm.settings.ollama_model)


def evaluate_answer_with_deepeval(question: str, answer: str, llm: OllamaLLM) -> EvalResult:
    """Run a real DeepEval GEval metric against the local Ollama endpoint."""

    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    metric = GEval(
        name="Evidence grounded Sakila answer",
        criteria=(
            "Judge whether the answer directly addresses the question, reports concrete evidence, "
            "and avoids causal certainty unsupported by observational data. Return a calibrated score."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=OllamaDeepEvalModel(llm).build(),
        threshold=0.7,
        async_mode=False,
    )
    metric.measure(LLMTestCase(input=question, actual_output=answer))
    score = float(metric.score or 0.0)
    reason = str(metric.reason or "")
    return EvalResult(
        passed=score >= 0.7,
        score=max(0.0, min(score, 1.0)),
        feedback=[reason] if reason else [],
        evaluator="deepeval-geval+ollama",
    )


def summarize_frame(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "No rows returned."
    return f"Returned {frame.height} rows. Evidence: {frame.head(5).to_dicts()}"
