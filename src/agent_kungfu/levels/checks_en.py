"""English descriptions for the shared deterministic checks."""

CHECKS = {
    "baseline.shape": (
        "Result contract",
        "16 rows with name/revenue columns",
        "Check baseline.yaml, revenue.sql and sakila.db paths.",
    ),
    "baseline.order": (
        "Revenue ordering",
        "Revenue is monotonically non-increasing",
        "Check ORDER BY revenue DESC.",
    ),
    "baseline.top": (
        "Leading category",
        "Sports / 5314.21",
        "Check the five-table join and SUM(payment.amount).",
    ),
    "baseline.total": (
        "Total reconciliation",
        "67406.56",
        "Check for missing rental/payment join records.",
    ),
    "common.answer": (
        "Nonempty answer",
        "An evidence-based answer",
        "Check that the final-answer stage completed.",
    ),
    "common.sql": (
        "Read-only SQL boundary",
        "One SELECT/WITH statement",
        "Remove stacked statements and database-control statements.",
    ),
    "common.level": ("Supported level", "Level 1–10", "Choose a supported lesson."),
    "l1.graph": (
        "Three-step graph",
        "read → merge → calculate",
        "Check LangGraph nodes and edges.",
    ),
    "l1.cardinality": (
        "Data cardinality",
        "Five source counts match; join=16044; category=16",
        "Check projections, four join keys and aggregation groups.",
    ),
    "l1.tool": (
        "Top-N tool boundary",
        "tool=top_n_categories; n=returned row count",
        "Check the prompt, tool schema and n bounds.",
    ),
    "l2.pattern": (
        "Manager pattern",
        "openai-agents / manager-as-tools",
        "Verify specialist delegation through Agent.as_tool.",
    ),
    "l2.order": (
        "Agent call order",
        "read_tables → calculate_revenue",
        "Enable calculation only after tables exist in the workspace.",
    ),
    "l2.evidence": (
        "Shared data contract",
        "join=16044; revenue=16; rows=Top-N",
        "Keep read/merge/calculate state in the shared workspace.",
    ),
    "l3.factory": (
        "Dynamic Codex team",
        "2 Codex workers; SDK=openai-codex; read/calculate only",
        "Check agent.md, worker generation, typed actions and the host allowlist.",
    ),
    "l3.protocols": (
        "Protocol envelopes",
        "a2a,codex-jsonrpc,mcp; 7 messages; 2 turns",
        "Record delegation, Codex turns and MCP evidence in ProtocolBus.",
    ),
    "l3.correlation": (
        "Request correlation",
        "The six child events share correlation_id",
        "Link delegation, turns and tool results with one request ID.",
    ),
    "l3.sandbox": (
        "Codex permission boundary",
        "read-only / deny_all / ephemeral; offline=0 threads, live=2",
        "Reject workspace-write/full-access; use ephemeral worker threads.",
    ),
    "l4.workers": (
        "Two Codex specialists",
        "Reader → Calculator",
        "Generate only the two allowlisted roles from OmniGenT YAML.",
    ),
    "l4.skill": (
        "Polars skill provenance",
        "polars + SHA-256 + 5 references",
        "Explicitly validate and package level4/polars.",
    ),
    "l4.evaluation": (
        "Harness quality gate",
        "10 checks passed; verified delegation; join=16044; categories=16",
        "Validate permissions, host actions, order, data domain and SQL parity before delivery.",
    ),
    "l4.trace": (
        "Control trace",
        "2 task.delegate + harness.evaluated",
        "Record OmniGenT, Codex, host-tool and evaluator boundaries.",
    ),
    "l5.bounded": (
        "Bounded 17-iteration trace",
        "16 inspections + 1 calculation; budget=17",
        "Check canonical table order, the final calculation and max_iterations.",
    ),
    "l5.coverage": (
        "Full-table coverage",
        "16 unique table observations and the five-table revenue chain",
        "Fix catalog order and inspect next_table only.",
    ),
    "l5.feedback": (
        "Feedback binding",
        "Nonempty feedback; actions echo prior-feedback and skill SHA-256",
        "Check feedback_digest_sha256 and skill_digest_sha256 together.",
    ),
    "l5.delegation": (
        "OmniGenT/Codex loop delegation",
        "17 completed tool-call records; actual Codex calls: live=17, offline=0",
        "Check bundle tools, runner evidence and executed_by.",
    ),
    "l5.boundary": (
        "Declared execution boundary",
        "read-only intent; no writes/network; scratch; approval=never",
        "Inspect final *_declared metadata and retain the host action allowlist.",
    ),
    "l5.stop": (
        "Immediate stop after verification",
        "goal_satisfied; 10/10; join=16044; categories=16; Sports=5314.21",
        "Reject partial delivery; require full coverage and baseline parity.",
    ),
    "l6.contract": (
        "Typed harness contract",
        "read_only=true, max_rows=50, max_iterations=3",
        "Check the Pydantic HarnessManifest.",
    ),
    "l6.graph": (
        "Schema graph",
        "16 tables / 21 relationships / 2 components",
        "Rebuild graph_summary from live foreign keys.",
    ),
    "l6.path": (
        "Collaboration path",
        "payment→rental→inventory→film",
        "Check edge direction and undirected path projection.",
    ),
    "l6.eval": ("Quality gate", "evaluation passed", "Inspect evaluator feedback before retrying."),
    "l7.pages": (
        "Persistent knowledge and fingerprint",
        "Schema pages + verified experience + database fingerprint",
        "Check experience records and fingerprints; retain old versions for audit.",
    ),
    "l7.artifacts": (
        "Persistent artifacts",
        "All Markdown files exist and are nonempty",
        "Check output permissions and generate_wiki.",
    ),
    "l7.eval": (
        "Evidence gate for knowledge",
        "evaluation passed",
        "Correct Wiki with SQL evidence, not the reverse.",
    ),
    "l8.core": (
        "Core ontology assets",
        "OWL, OBDA and two annotation files",
        "Check the four ontology generation writes.",
    ),
    "l8.bundle": (
        "Knowledge production bundle",
        "4 ontology + 17 Wiki files, all present",
        "Check ontology_dir, wiki_dir and write permissions.",
    ),
    "l8.eval": (
        "Evidence gate after semantics",
        "evaluation passed",
        "Validate mappings and query evidence; RDF syntax alone is insufficient.",
    ),
    "l9.evidence": ("Observational evidence", "2 stores", "Check the store aggregation SQL."),
    "l9.dag": (
        "Causal graph structure",
        "6 edges; directed and acyclic; required confounder paths",
        "Repair cycles or missing treatment/outcome/confounder edges.",
    ),
    "l9.caveat": (
        "Association is not causation",
        "The answer explicitly distinguishes association from a causal effect",
        "State identification assumptions, confounders and required experiments.",
    ),
    "l10.causal": (
        "Inherited causal boundary",
        "L9 DAG and two-store observational evidence",
        "Retain evidence and identification limits when proposing innovation.",
    ),
    "l10.structure": (
        "First-principles structure",
        "1 objective / 3 invariants / 4 constraints",
        "Complete objectives, invariants and constraints before mechanisms.",
    ),
    "l10.falsifiable": (
        "Mechanisms and experiments",
        "3 mechanisms / 3 experiments",
        "Pair each mechanism with a falsifiable experiment, guardrails and stopping conditions.",
    ),
    "l2.mechanism": (
        "Actual SDK delegation",
        "Observable execution evidence",
        "Run the capability experiment and inspect the intervention.",
    ),
    "l3.mechanism": (
        "JSON-RPC execution boundary",
        "Observable execution evidence",
        "Run the capability experiment and inspect the intervention.",
    ),
    "l4.mechanism": (
        "Evidence-based conflict resolution",
        "Observable execution evidence",
        "Run the capability experiment and inspect the intervention.",
    ),
    "l6.mechanism": (
        "Graph-driven execution",
        "Observable execution evidence",
        "Run the capability experiment and inspect the intervention.",
    ),
    "l8.mechanism": (
        "Ontology publication gate",
        "Observable execution evidence",
        "Run the capability experiment and inspect the intervention.",
    ),
    "l9.mechanism": (
        "Synthetic causal identification",
        "Observable execution evidence",
        "Run the capability experiment and inspect the intervention.",
    ),
    "l10.mechanism": (
        "Innovation negative control",
        "Observable execution evidence",
        "Run the capability experiment and inspect the intervention.",
    ),
}


def check_record(check, language, tr):
    if language != "en":
        return check.as_record()
    title, expected, recovery = CHECKS[check.check_id]
    return {
        "Status": "PASS" if check.passed else "FAIL",
        "Check": title,
        "Expected": expected,
        "Actual": tr(check.actual),
        "Recovery": "—" if check.passed else recovery,
    }
