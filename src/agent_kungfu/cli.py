from __future__ import annotations

import json

import typer

from .levels import DEFAULT_QUESTION, LEVELS, run_level


app = typer.Typer(help="Run ten progressive Sakila agent-engineering demos.")


@app.command("list")
def list_levels() -> None:
    for number, runner in LEVELS.items():
        typer.echo(f"{number:02d}  {runner.__doc__ or runner.__name__}")


@app.command("run")
def run(
    level: int = typer.Argument(..., min=1, max=10),
    question: str = typer.Option(DEFAULT_QUESTION, "--question", "-q"),
    offline: bool = typer.Option(False, help="Skip LLM/server calls but exercise the real workflow."),
    deepeval: bool = typer.Option(False, help="At level 6, run GEval using local Ollama."),
) -> None:
    result = run_level(level, question, offline=offline, use_deepeval=deepeval)
    typer.echo(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str))


@app.command("experiment")
def experiment(level: int = typer.Argument(..., min=1, max=10)) -> None:
    """Run the isolated, deterministic capability intervention for one level."""
    from .levels.experiments import run_experiment

    report = run_experiment(level)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(check["passed"] for check in report["checks"]):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
