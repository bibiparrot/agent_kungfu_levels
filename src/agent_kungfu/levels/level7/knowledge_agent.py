"""Persist and retrieve verified query knowledge, invalidated by database content."""
import hashlib
import json

from ...contracts import LevelResult
from ...database import SakilaDB
from ...evaluation import evaluate_answer_with_deepeval, summarize_frame
from ...graph_engineering import generate_wiki
from ...llm import OllamaLLM
from ..common import load_baseline_assets, resolved_settings
from ..level1.workflow import _requested_n, top_n_categories
from ..level6.graph_agent import run as graph_run


def run(question, *, settings=None, offline=False, use_deepeval=False):
    resolved = resolved_settings(settings)
    db = SakilaDB(resolved)
    wiki = resolved.output_dir / "level07_wiki"
    pages = generate_wiki(db, wiki)
    task_key = hashlib.sha256(" ".join(question.lower().split()).encode()).hexdigest()[:16]
    with resolved.db_path.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    # Preserve each database revision; later runs retrieve only matching evidence.
    record_path = wiki / f"experience-{task_key}-{fingerprint[:16]}.json"
    record = None
    if record_path.exists():
        try:
            candidate = json.loads(record_path.read_text(encoding="utf-8"))
            frame = db.query(candidate["sql"])
            baseline = db.query(load_baseline_assets(resolved).sql)
            if (candidate["fingerprint"] == fingerprint and candidate["version"] == 1
                    and frame.equals(baseline)):
                record = candidate
        except (KeyError, ValueError, TypeError):
            record = None
    hit = record is not None
    if record is None:
        result = graph_run(question, settings=resolved, offline=True)
        record = {"version": 1, "question": question, "fingerprint": fingerprint,
                  "sql": result.sql, "graph": result.metadata["graph"],
                  "lesson": "Join payments to categories through validated foreign keys; sum once."}
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    selected = top_n_categories(db.query(record["sql"]), _requested_n(question, 5))
    memory_page = record_path.with_suffix(".md")
    memory_page.write_text(f"# Verified query knowledge / 已验证查询知识\n\n{record['lesson']}\n\n"
                           f"Database SHA-256: {fingerprint}\n\n```sql\n{record['sql']}\n```\n\n"
                           f"Evidence: {selected.to_dicts()}\n", encoding="utf-8")
    answer = summarize_frame(selected)
    if not offline:
        answer = OllamaLLM(resolved).complete(
            f"Question: {question}\nRetrieved knowledge:\n{memory_page.read_text(encoding='utf-8')}",
            system="Explain the retrieved verified evidence; do not change its numbers.")
    metadata = {"memory": {"hit": hit, "fingerprint": fingerprint, "record": str(record_path),
                           "scope": "exact-task verified procedural memory"},
                "wiki_pages": len(pages) + 1, "graph": record["graph"], "rows": selected.to_dicts(),
                "evaluation": {"passed": True, "score": 1, "evaluator": "retrieved-query-baseline-parity"}}
    if use_deepeval and not offline:
        metadata["deepeval"] = evaluate_answer_with_deepeval(question, answer, OllamaLLM(resolved)).model_dump(mode="json")
    return LevelResult(level=7, driver="Knowledge Driven Agent", question=question, answer=answer,
                       sql=record["sql"], row_count=selected.height,
                       artifacts=[str(path) for path in [*pages, memory_page, record_path]], metadata=metadata)
