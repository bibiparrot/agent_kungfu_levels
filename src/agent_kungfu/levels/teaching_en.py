"""English instructional text; execution and source files stay shared."""

from dataclasses import replace

from .teaching import LearningStep


# Each entry follows the same instructional contract as the Chinese material.
LESSONS = [
    dict(
        title="Baseline · Establish a trusted data reference",
        goal="Understand Sakila's rental relationships and use YAML to locate the database and SQL. The baseline is a deterministic reference for agent results.",
        change_from_previous="Start with a deterministic data baseline that every subsequent level must reproduce.",
        prerequisites=(
            "data/sakila.db is readable; baseline.yaml resolves the database and SQL paths.",
            "Polars and ConnectorX are installed; queries pass the single-statement SELECT/WITH guard.",
        ),
        steps=(
            (
                "1 · Locate",
                "Resolve SQLite and revenue.sql from baseline.yaml.",
                "Both paths exist.",
            ),
            (
                "2 · Understand",
                "Follow the five-table foreign-key chain in the ER diagram and SQL.",
                "Explain why revenue comes from payment.amount.",
            ),
            (
                "3 · Execute",
                "Run revenue.sql with Polars and ConnectorX.",
                "Return all 16 categories in descending order.",
            ),
            (
                "4 · Reconcile",
                "Check the leader and total revenue.",
                "Sports=5314.21; total=67406.56.",
            ),
        ),
        checkpoints=(
            "SQLite is the source of truth; the two ER diagrams explain relationships without replacing the live schema.",
            "Revenue follows category → film_category → inventory → rental → payment.",
            "Expect 16 categories and total revenue of 67406.56.",
        ),
        observations=(
            "YAML defines paths, SQL defines computation, and images explain relationships.",
            "Categories must be unique and ordered by revenue descending.",
            "Agent Top-N results must match the first N baseline rows.",
        ),
        exercises=(
            "Predict the top three categories, then run L1 with N=3 and compare with the baseline.",
            "Explain why payment must pass through rental to join inventory without changing the SQL.",
        ),
        question="Which film category earns the most revenue?",
        explanation="The deterministic baseline starts with Sports at 5314.21.",
    ),
    dict(
        title="Level 1 · Prompt Agent: direct a fixed workflow",
        goal="Fix computation as read → merge → calculate. The LLM only chooses how many rows to display through top_n_categories(n).",
        change_from_previous="Replace one deterministic SQL query with a fixed three-step workflow; the LLM only chooses N.",
        prerequisites=(
            "Complete the baseline and understand the five-table revenue chain.",
            "Understand DataFrame joins and group_by.",
        ),
        steps=(
            (
                "1 · Read",
                "Read category, film_category, inventory, rental and payment separately.",
                "Keep five Polars DataFrames and their row counts.",
            ),
            (
                "2 · Merge",
                "Apply four foreign-key inner joins in order.",
                "The payment chain contains 16044 rows.",
            ),
            (
                "3 · Calculate",
                "Sum amount by category.name and sort descending.",
                "Produce 16 category-revenue rows.",
            ),
            (
                "4 · Tool call",
                "Request only top_n_categories(n).",
                "Return the first N baseline rows.",
            ),
        ),
        checkpoints=(
            "Project required columns before joining to avoid duplicate last_update columns.",
            "The model neither writes SQL nor computes money; it supplies a typed n argument.",
            "The inner joins produce 16044 payment-chain records.",
        ),
        observations=(
            "Graph-node order is fixed in code.",
            "The LLM does not generate SQL or perform arithmetic.",
            "The n argument has lower and upper bounds.",
        ),
        exercises=(
            "Run N=1, 5 and 16; verify every result equals Baseline.head(N).",
            "Explain why N=0 and N=17 must be rejected at the tool boundary.",
        ),
        question="How many rows should the joined payment chain contain?",
        explanation="payment has 16049 rows, but only 16044 enter the complete rental–inventory–category chain.",
    ),
    dict(
        title="Level 2 · Code Agent: grant capabilities through code",
        goal="The QA leader retains answer ownership and delegates reading and calculation to two specialist agent-tools. Large DataFrames remain in shared runtime state.",
        change_from_previous="Move data capabilities into explicitly authorized reader and calculator agent-tools.",
        prerequisites=(
            "Understand L1's three-step data contract.",
            "Understand answer ownership in manager-as-tools versus handoff.",
        ),
        steps=(
            (
                "1 · Team",
                "Create a QA leader, Table Read Agent and Calculate Agent.",
                "One manager and two specialist roles.",
            ),
            (
                "2 · Read",
                "The leader calls the reader first; DataFrames stay in the workspace.",
                "The first event is read_tables.",
            ),
            (
                "3 · Calculate",
                "Enable the calculator only when its state dependency is satisfied.",
                "The second event is calculate_revenue.",
            ),
            (
                "4 · Answer",
                "The leader answers from structured results.",
                "Results match baseline Top-N.",
            ),
        ),
        checkpoints=(
            "Agent.as_tool supports the manager pattern; handoff transfers final-answer ownership.",
            "is_enabled and parallel_tool_calls=False enforce reader-before-calculator ordering.",
            "Check events == [read_tables, calculate_revenue].",
        ),
        observations=(
            "Tools exchange small control messages, not large DataFrames.",
            "Tool availability implements authorization in code.",
            "The leader owns the final answer.",
        ),
        exercises=(
            "Diagram the state error when the calculator runs before the reader.",
            "Compare who owns the final answer under manager-as-tools and handoff.",
        ),
        question="What is the valid specialist call order?",
        explanation="The calculator depends on the reader populating the shared workspace, so execution must be sequential.",
    ),
    dict(
        title="Level 3 · Protocol Agent: generate collaborators from agent.md",
        goal="Declare roles, dependencies and tool names in reviewable agent.md data. Create two read-only, ephemeral Codex SDK workers. A2A records delegation, Codex JSON-RPC records turns, and MCP records evidence.",
        change_from_previous="Roles come from agent.md rather than hard-coded definitions; live protocol traces cross the app-server boundary.",
        prerequisites=(
            "Understand L2's shared workspace and tool order.",
            "Understand YAML frontmatter, Pydantic and allowlists.",
        ),
        steps=(
            ("1 · Parse", "Read agent.md frontmatter.", "A leader and two subagent definitions."),
            (
                "2 · Validate",
                "Validate typed contracts and the runtime_tool allowlist.",
                "Arbitrary function names cannot execute.",
            ),
            (
                "3 · Generate",
                "Build two Codex thread specifications in dependency order.",
                "Live mode starts two read-only, ephemeral threads.",
            ),
            (
                "4 · Dispatch",
                "Codex proposes a typed action; the host allowlist executes Polars.",
                "The model has no arbitrary Python or shell capability.",
            ),
            (
                "5 · Trace",
                "Record delegation, turns and results in A2A/JSON-RPC/MCP envelopes.",
                "Child events share a correlation_id.",
            ),
        ),
        checkpoints=(
            "agent.md is data: do not execute imports, eval or arbitrary function names.",
            "Codex threads use a read-only sandbox and deny_all approval; they only propose structured actions.",
            "The host runtime_tool allowlist permits read_tables and calculate_revenue only.",
            "Messages for one request share a correlation_id across protocols.",
        ),
        observations=(
            "L3 does not import or run openai-agents.",
            "Offline Codex JSON-RPC is explicitly simulated; only live mode starts app-server.",
            "Polars execution remains under host control.",
        ),
        exercises=(
            "In a copy, change runtime_tool to an unknown value and predict validation failure.",
            "Find the six related child events for one request in the trace.",
        ),
        question="How many specialist subagents should the factory generate?",
        explanation="agent.md declares one reader and one calculator. The leader is not a specialist.",
    ),
    dict(
        title="Level 4 · Harness Agent: enforce capability boundaries",
        goal="Bundle two Codex subagents with an explicitly loaded Polars skill. Declare minimal permissions, execute only host-allowlisted actions, and evaluate contracts, order, data and baseline parity before delivery.",
        change_from_previous="Add an executable OmniGenT/Codex harness that brings skills, permissions, actions and quality evidence under one control layer.",
        prerequisites=(
            "Understand L3's action allowlist and protocol trace.",
            "Understand bundles, skill digests and fail-closed evaluation.",
        ),
        steps=(
            (
                "1 · Load",
                "Validate Polars SKILL.md, five references and SHA-256.",
                "Skill provenance is auditable.",
            ),
            (
                "2 · Bundle",
                "Package agent configuration and the skill into an OmniGenT tar bundle.",
                "Codex can discover skills/polars.",
            ),
            (
                "3 · Generate",
                "Call Reader and Calculator Codex agent-tools in order.",
                "The model returns strict JSON actions.",
            ),
            (
                "4 · Execute",
                "Read five tables and execute a one-collect lazy Polars plan through the host allowlist.",
                "Large tables never enter agent messages.",
            ),
            (
                "5 · Evaluate",
                "Check environment/approval contracts, order, cardinality, sorting and SQL parity.",
                "Any failed invariant blocks delivery.",
            ),
        ),
        checkpoints=(
            "Agent YAML may declare only read_tables and calculate_revenue Codex tools.",
            "The loader must validate level4/polars and package it under skills/polars; it is not automatically discovered.",
            "All three configurations declare sandbox=auto, no write_paths, no network and scratch startup. OmniGenT 0.11's Codex executor uses approvalPolicy=never.",
            "Windows Job Objects do not enforce filesystem/network isolation. The cross-platform boundary is typed proposals plus host-allowlisted execution.",
            "Models only propose Pydantic actions; the host reads SQLite and computes with Polars.",
            "Failed evaluation blocks delivery. Live mode requires OMNIGENT_URL and an available Codex runner; offline mode does not claim a connection.",
        ),
        observations=(
            "omnigent-client provides the actual bundle/session boundary.",
            "Configuration declares sandbox=auto, no writes/network, scratch startup and approvalPolicy=never.",
            "Windows Job Objects isolate process trees/resources; typed proposals and host allowlists provide the portable execution boundary.",
            "The host evaluator is a deterministic quality gate, not a third agent.",
        ),
        exercises=(
            "Alter a reference and predict changes to the skill digest and evaluator.",
            "Move the calculator before the reader and locate the manifest or state-gate failure.",
        ),
        question="How many Codex specialist agents does L4 create?",
        explanation="Only Reader and Calculator are subagents; evaluation is a deterministic host-harness stage.",
    ),
    dict(
        title="Level 5 · Loop Agent: feed failures into the next iteration",
        goal="Use Ralph state, previous-evaluation digests and a 17-iteration budget to inspect all 16 Sakila tables. Then compute category revenue and stop immediately when Polars matches the SQL baseline.",
        change_from_previous="Extend the one-shot L4 harness into a stateful OmniGenT/Codex Ralph loop with feedback binding and an evidenced stopping condition.",
        prerequisites=(
            "Understand L4 bundles, Codex agent-tools, typed actions and the host allowlist.",
            "Understand catalog snapshots, loop invariants, budgets and fail-closed stopping.",
        ),
        steps=(
            (
                "1 · Snapshot",
                "Read the SQLite catalog and fix canonical table order.",
                "Exactly 16 unique table names.",
            ),
            (
                "2 · Bundle",
                "Bundle two Codex tools and the Polars skill for OmniGenT.",
                "Explorer and Solver permissions are consistent and traceable.",
            ),
            (
                "3 · Inspect",
                "Inspect only the next unseen table and evaluate coverage.",
                "Produce 16 table observations and feedback digests.",
            ),
            (
                "4 · Calculate",
                "Allow five-table Polars revenue computation only after full coverage.",
                "join=16044; category=16.",
            ),
            (
                "5 · Verify & Stop",
                "Run ten harness checks and reconcile with revenue.sql.",
                "Stop at iteration 17 with goal_satisfied.",
            ),
        ),
        checkpoints=(
            "The successful trace contains 16 canonical inspect_table actions followed by one calculate_revenue.",
            "Every action echoes SHA-256 digests of prior feedback and the loaded Polars skill.",
            "OmniGenT/Codex proposes Pydantic actions; the host executes SQLite and Polars.",
            "Live mode accepts server-executed tool-call evidence only; offline calls are explicitly simulated.",
            "Iteration 17 may stop only after coverage, cardinality, sorting and baseline parity pass.",
            "Budget exhaustion or invariant failure must reject delivery, not return a partial answer.",
        ),
        observations=(
            "evaluation.passed=false during the first 16 iterations means incomplete coverage, not a system error.",
            "feedback_digest_sha256 binds the next action to the previous evaluation.",
            "Offline calls are simulated, but all table inspections, Polars calculations and baseline checks execute.",
            "Success requires terminated=true, stop_reason=goal_satisfied and ten passing checks.",
        ),
        exercises=(
            "Run Top-N=1 and 16; verify full coverage and the 17-iteration trace remain unchanged.",
            "Explain why bounded-immediate-stop prevents exit at iteration 16.",
        ),
        question="After inspecting 16 tables and calculating once, on which iteration should success stop?",
        explanation="Iterations 1–16 inspect one table each; iteration 17 calculates revenue and stops after baseline parity passes.",
    ),
    dict(
        title="Level 6 · Graph Engineering Agent: execute structured collaboration",
        goal="Constrain execution with a typed harness, derive cross-table paths from the schema graph, and pass deterministic evaluation.",
        change_from_previous="Enter the intelligence layer: control shifts from execution traces to typed data and collaboration relationships.",
        prerequisites=(
            "Understand HarnessManifest permission and budget fields.",
            "Understand graph nodes, edges and shortest paths.",
        ),
        steps=(
            (
                "1 · Contract",
                "Instantiate a typed HarnessManifest.",
                "Validate read_only, row limits and iteration budgets.",
            ),
            (
                "2 · Graph",
                "Build table relationships from the live SQLite schema.",
                "16 tables and 21 relationships.",
            ),
            (
                "3 · Plan",
                "Derive the payment-to-category JoinPlan from foreign keys.",
                "Graph edges determine join conditions.",
            ),
            (
                "4 · Execute",
                "Schedule reads, joins and aggregation from the task graph; compare with SQL.",
                "Removing a required relationship prevents computation.",
            ),
        ),
        checkpoints=(
            "Pydantic moves permissions, row limits and budgets from prompts into code contracts.",
            "A shortest path is a structural candidate, not automatically a correct business path.",
            "Structured relationships do not yet provide ontology semantics.",
        ),
        observations=(
            "Typed contracts enforce key controls in code.",
            "Structural paths still require business interpretation.",
            "Graph summaries can support subsequent knowledge production.",
        ),
        exercises=(
            "Choose another pair of tables and predict the shortest join path.",
            "Explain why max_rows=0 should fail model validation.",
        ),
        question="What is the shortest relationship path from payment to film?",
        explanation="payment joins rental, then inventory, then film.",
    ),
    dict(
        title="Level 7 · Knowledge Driven Agent: remember previous work",
        goal="Persist schema, relationships and execution experience as readable, diffable and reusable LLM Wiki knowledge.",
        change_from_previous="Turn transient schema and experience into persistent knowledge that can be reviewed and reused.",
        prerequisites=(
            "Understand the L6 schema graph.",
            "Understand knowledge provenance, freshness and review.",
        ),
        steps=(
            (
                "1 · Inspect",
                "Read the live schema and table sizes.",
                "Trace knowledge back to the database.",
            ),
            (
                "2 · Generate",
                "Save verified query knowledge, database fingerprints and pages.",
                "17 static pages plus experience Markdown/JSON.",
            ),
            (
                "3 · Reuse",
                "Run the same question and retrieve persistent query knowledge.",
                "memory.hit=true; numeric evidence is still queried again.",
            ),
            (
                "4 · Invalidate",
                "Modify payment in a temporary copy and rerun.",
                "A changed fingerprint invalidates old evidence.",
            ),
        ),
        checkpoints=(
            "Wiki artifacts persist across runs; they are more than expanded prompt context.",
            "Persistent knowledge requires freshness, provenance and change review.",
            "Query evidence takes precedence over Wiki statements.",
        ),
        observations=(
            "Retrieval is scoped to verified knowledge for the same task.",
            "Old database-version experience is retained while new versions are learned.",
            "Offline mode also persists knowledge; model explanations cannot override validation.",
        ),
        exercises=(
            "Run twice and compare memory.hit.",
            "Run the capability experiment, modify a payment in the database copy, and observe invalidation.",
        ),
        question="How should old query knowledge be handled after the database changes?",
        explanation="A different fingerprint requires regeneration, validation and a new knowledge version.",
    ),
    dict(
        title="Level 8 · Ontology Driven Agent: produce knowledge with semantics",
        goal="Organize knowledge through classes, relations and constraints, then generate OWL, OBDA, annotations and Wiki pages.",
        change_from_previous="Move from recording knowledge to producing it under explicit semantic structure.",
        prerequisites=(
            "Understand schema versus ontology.",
            "Understand the roles of OWL, OBDA and annotations.",
        ),
        steps=(
            (
                "1 · Model",
                "Generate ontology classes and relations from the schema.",
                "Obtain semantic candidates.",
            ),
            (
                "2 · Map",
                "Generate OWL, OBDA and two annotation files.",
                "Four core ontology assets exist.",
            ),
            (
                "3 · Propose",
                "Propose typed relations from the OWL vocabulary.",
                "Live mode uses LLM selection; offline enumerates allowed relations.",
            ),
            (
                "4 · Validate",
                "Check classes, domain/range and foreign-key mappings, then count relations.",
                "Unknown concepts or invalid ranges block publication.",
            ),
        ),
        checkpoints=(
            "Schema describes storage; ontology also expresses business concepts, types, relations and constraints.",
            "Valid OWL syntax does not guarantee correct knowledge.",
            "Keep the canonical ontology separate from LLM candidate proposals.",
        ),
        observations=(
            "Schema explains storage; ontology explains concept meaning.",
            "Structural validity does not guarantee domain correctness.",
            "Canonical knowledge and model proposals occupy separate layers.",
        ),
        exercises=(
            "Trace payment from table to class to OBDA mapping.",
            "Describe a business relationship that requires human confirmation beyond foreign keys.",
        ),
        question="How many core ontology asset files are produced?",
        explanation="One OWL file, one OBDA file and two annotation files: four core assets.",
    ),
    dict(
        title="Level 9 · Causal Driven Agent: identify the problem before answering",
        goal="Start with a causal question and explicitly organize assumptions, causes, outcomes, confounders and identification gaps.",
        change_from_previous="Make knowledge production answer a causal question, separating association, assumptions, confounding and identifiability.",
        prerequisites=(
            "Understand DAGs and confounding.",
            "Accept identification gaps when observational evidence is insufficient.",
        ),
        steps=(
            (
                "1 · Evidence",
                "Query inventory, rentals, customers and revenue for both stores.",
                "Obtain an observational comparison.",
            ),
            (
                "2 · DAG",
                "Model causes, outcomes and confounders.",
                "Six directed edges and no cycle.",
            ),
            (
                "3 · Experiment",
                "Use a fixed seed to generate confounded synthetic rental data with a known effect.",
                "The naive difference departs from the true effect.",
            ),
            (
                "4 · Adjust",
                "Stratify by demand and report intervals and overlap.",
                "Recover the known effect; reject estimation without positivity.",
            ),
        ),
        checkpoints=(
            "Observed store differences do not establish inventory-intervention effects.",
            "A DAG exposes assumptions and identification gaps; it is not a decorative report.",
            "When identification fails, report missing data or an experimental design rather than false certainty.",
        ),
        observations=(
            "Real two-store evidence supports association only.",
            "Causal estimates come from a clearly labeled synthetic SCM.",
            "Report known effects, identification assumptions and intervals together.",
        ),
        exercises=(
            "Add revenue→inventory_breadth and explain the resulting cycle.",
            "Design a randomized or quasi-experiment for inventory intervention and list confounding controls.",
        ),
        question="Is the default causal graph a DAG?",
        explanation="Its six edges form a directed acyclic graph; graph validity alone does not identify an effect.",
    ),
    dict(
        title="Level 10 · First-Principles Driven Agent: reframe and test mechanisms",
        goal="Derive mechanisms from objectives, invariants and constraints, then connect causal models, ontology, Wiki and graph engineering through falsifiable experiments.",
        change_from_previous="Reconsider the given problem and derive falsifiable mechanisms from objectives and constraints.",
        prerequisites=(
            "Complete L9 and distinguish association from causation.",
            "Understand falsifiable hypotheses, guardrails and stopping conditions.",
        ),
        steps=(
            (
                "1 · Reframe",
                "Compare revenue with net contribution per inventory cycle.",
                "Synthetic costs can falsify revenue-growth proposals.",
            ),
            (
                "2 · Decompose",
                "List invariants and constraints and challenge default assumptions.",
                "Innovation does not remove execution boundaries.",
            ),
            (
                "3 · Derive",
                "Enumerate inventory-preserving transfers; compare purchasing and retaining the baseline.",
                "Actually simulate every feasible candidate.",
            ),
            (
                "4 · Falsify",
                "Compare net contribution and test balanced-demand negative controls.",
                "Use selected=null when no candidate improves the baseline.",
            ),
        ),
        checkpoints=(
            "First principles require a stated way for every mechanism to fail.",
            "Connect candidates to estimands, experiments, guardrails and stopping conditions.",
            "Reframing the objective does not expand database permissions.",
        ),
        observations=(
            "Candidates follow constraint enumeration; conclusions apply only to the declared synthetic model.",
            "Maximum revenue may produce the worst net contribution.",
            "Stop after evaluating finite candidates; no improvement is a valid outcome.",
        ),
        exercises=(
            "Change transfer costs and find when the selected policy loses its advantage.",
            "Balance demand and verify that no intervention is recommended.",
        ),
        question="How many mechanism families appear in the default first-principles output?",
        explanation="The three families are transfer, purchase and retain-baseline; resource constraints determine their candidates.",
    ),
]


def english_material(material, tr):
    text = dict(LESSONS[material.level])
    question = text.pop("question")
    explanation = text.pop("explanation")
    text["steps"] = tuple(LearningStep(*step) for step in text["steps"])
    return replace(
        material,
        **text,
        architecture=tr(material.architecture),
        sources=tuple(replace(source, label=tr(source.label)) for source in material.sources),
        prediction=replace(
            material.prediction,
            question=question,
            explanation=explanation,
            options=tuple((tr(label), value) for label, value in material.prediction.options),
        ),
    )
