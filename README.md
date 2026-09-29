# Agent Kung-fu Levels: Ten Layers of Sakila Agent Engineering

This is not ten chat scripts that differ only in name. It is a runnable teaching project that adds one more "control plane" at each layer. All layers share the same read-only data kernel:

- Database: `data/sakila.db`
- Data access: `polars.read_database_uri` + ConnectorX
- Local model: Ollama OpenAI-compatible API `http://localhost:11434/v1`
- Fixed model: `ornith-1.5:9b`
- SQL boundary: only a single `SELECT` / `WITH` statement is allowed; modification statements and SQLite control statements are forbidden

## The Ten-Level Map

| Level | Driver | Main implementation | Control plane added |
|---:|---|---|---|
| 1 | Prompt agent | LiteLLM + LangGraph + Jinja2 | Code fixes the workflow; the prompt/tool calls decide Top-N |
| 2 | Code agent | `openai-agents` + function tools | Code owns skills and tool permissions |
| 3 | Protocol agent | `agent.md` + Codex Python SDK + A2A/MCP/JSON-RPC | Dynamic Codex workers and protocol message boundaries |
| 4 | Harness Agent | `omnigent-client` + Codex + Polars skill | Two-sub-agent plan, host action allowlist, and fail-closed evaluation |
| 5 | Iteration flywheel | `omnigent-client` + Codex + Ralph loop | Table-exploration trajectory, feedback binding, and deterministic stopping |
| 6 | Data + relational logic | Pydantic schema + NetworkX + DeepEval | Type contracts, relationship paths, quality gates |
| 7 | Knowledge accumulation | LLM wiki + graph engineer | Retrieves verified query knowledge; invalidated by database fingerprint |
| 8 | Knowledge production | OWL + OBDA + annotations + wiki | Ontology-constrained knowledge generation |
| 9 | Problem-driven | Causal DAG + ontology/wiki/graph | Shifts from correlation to identification problems |
| 10 | Innovation-driven | First principles + causal experiments | Derives falsifiable mechanisms from invariants |

For detailed data flows and boundaries, see [docs/architecture.md](docs/architecture.md).

For the per-level revised design, interventions, and acceptance criteria, see the [Ten-Level Experiment Plan / Experiment plan](docs/level-revision-plan.md).
Each level in Marimo has a new "Run capability experiment" button; you can also run:

```powershell
kungfu-agent experiment 5
```

Experiments use isolated temporary directories. L2 runs the real Agents SDK offline, replacing only the model replies; L3 verifies the local JSON-RPC boundary; L4 injects conflicts that are adjudicated by evidence; L5 verifies error feedback and bounded retries; in L6 the relationship graph actually determines joins and computation; L7 verifies knowledge reuse and invalidation on data change; L8 validates ontology relationships before publishing knowledge. L9–L10 use explicitly labeled synthetic causal and resource-allocation models, and do not present experimental effects or candidate improvements as real-world conclusions about Sakila.

Each level now includes a controlled intervention. Offline reproducibility and scripted transport
tests are separate from live model validation. L9–L10 use labeled synthetic experiments; their
effects and improvements do not establish real-world causality or innovation.

## Installation

Python 3.12+:

```powershell
python -m pip install -e ".[harness,eval,dev]"
```

If you only want to study a single level, you can also install that level's full runtime dependencies from the repository root. The `requirements.txt` in each Level 1–10 directory contains both the common runnable-tutorial dependencies and that level's specific frameworks. For example:

```powershell
python -m pip install -r src/agent_kungfu/levels/level5/requirements.txt
```

Confirm Ollama and the model:

```powershell
ollama list
Invoke-RestMethod http://localhost:11434/v1/models
```

All environment variables have usable defaults, and can be overridden:

```powershell
$env:OLLAMA_OPENAI_BASE_URL = "http://localhost:11434/v1"
$env:AGENT_LLM_MODEL = "ornith-1.5:9b"
$env:SAKILA_DB_PATH = "D:\Python\agent_kungfu_levels\data\sakila.db"
```

Ollama does not validate API keys; the code only passes the placeholder value `ollama` to compatible clients.

## Running

### Browser demo console (recommended, no Python editor needed)

Double-click in File Explorer:

```text
start_marimo.cmd
```

After startup, visit `http://127.0.0.1:2718`. The home page contains only setup guidance, environment checks, and the Demo index.
Click each lesson's **index** to open its standalone page; you can also jump from the index directly to that lesson's architecture diagram, code walkthrough, or run results.
The main entry and every standalone Demo have a **中文 / English** toggle at the top. Setup guidance, teaching content, architecture-diagram text, controls, and check descriptions all switch together; the index, previous-lesson, and next-lesson links preserve the language choice. You can also visit `/?lang=en` or `/level_01/index/?lang=en` directly, and the language in the URL is retained on refresh.
Source code and raw run evidence stay in their original language; offline answers come with an English counterpart, and live model output keeps the text the model actually returned.

```text
demos/
├── index.py             # Setup and main index
├── baseline/index.py    # SQL baseline
├── level_01/index.py    # Prompt Agent
├── level_02/index.py    # Code Agent
├── level_03/index.py    # Protocol Agent
├── level_04/index.py    # Harness Agent
├── level_05/index.py    # Ralph Loop
├── level_06/index.py    # Graph Engineering
├── level_07/index.py    # Knowledge Driven
├── level_08/index.py    # Ontology Driven
├── level_09/index.py    # Causal Driven
└── level_10/index.py    # First Principles
```

Each index loads only its own lesson, and contains the architecture diagram, real source code with step-by-step explanations, input controls, run results, automatic checks, and the capability experiment.
Click **Run lesson** to produce that lesson's results; changing a control marks the old results as stale, and they update after you run again.
The default offline mode still executes the real database, Polars, graph, and artifact generation. Each page's run state is independent; reopening a page requires running it again.

The unified launch logic is maintained only in `serve_demos.py`; the two shell entry points only locate the directory, set the Python path, and pass along arguments.

| Environment | Launch command |
|---|---|
| Windows Batch / PowerShell | `.\start_marimo.cmd` |
| Linux / macOS / Bash | `bash start_marimo.sh` |
| Python | `python serve_demos.py` |

All of them accept `--port 2720` to specify a port, and `--help` to list parameters. Bash uses `python3` by default and Batch uses `python`; you can specify an interpreter path via the `PYTHON` environment variable. Press Ctrl+C to stop the server.

To launch the whole directory from PowerShell:

```powershell
python serve_demos.py --port 2718
```

`/` is the main index, `/baseline/index/` is the Baseline, and `/level_01/index/` through
`/level_10/index/` are the individual lessons. `start_levels_marimo.cmd` is a forwarding entry point under the old name, and also uses the unified parameters and default port.
The multi-page service uses [marimo's native ASGI routing](https://docs.marimo.io/guides/deploying/programmatically/).

To run only one lesson (cross-lesson navigation requires starting the multi-page service above):

```powershell
python -m marimo run demos/level_01/index.py
```

`marimo_app.py` is reused as the home page content, and `marimo_levels.py` is reused as the single-lesson tutorial component;
each lesson's index fixes its lesson number, and no longer puts all demos behind a single dropdown.
The original flat CLI entry points and the dedicated L4/L5 tutorials are retained for compatibility.

List the levels:

```powershell
kungfu-agent list
```

Run with the real local model:

```powershell
kungfu-agent run 1
kungfu-agent run 2 -q "收入最高的电影有哪些？"
kungfu-agent run 3 -q "比较门店收入"
```

Teaching mode, which calls no model or external server but still executes the real LangGraph, Polars, ConnectorX, graph, OWL generation, and harness:

```powershell
kungfu-agent run 1 --offline
python demos/level_05_ralph_loop.py --offline
python demos/level_10_first_principles.py --offline
```

The original command-line and dedicated-tutorial entry points remain usable:

```text
demos/level_01_prompt_workflow.py
demos/level_02_skills_openai_agents.py
demos/level_03_multi_protocol_codex.py
demos/level_04_omnigent_codex_polars_harness.py
demos/level_04_harness_conflict_omnigent.py
demos/level_05_ralph_omnigent_codex_loop.py
demos/level_05_ralph_loop.py
demos/level_06_schema_graph_deepeval.py
demos/level_07_wiki_graph.py
demos/level_08_ontology_wiki_graph.py
demos/level_09_causal_problem.py
demos/level_10_first_principles.py
```

`level_04_omnigent_codex_polars_harness.py` is the main teaching entry point for Level 4; the older
`level_04_harness_conflict_omnigent.py` is kept only as a backward-compatible launcher and runs the same new implementation.
To open the Level 4 executable tutorial on its own:

```powershell
marimo run demos/level_04_omnigent_codex_polars_harness.py
```

`level_05_ralph_omnigent_codex_loop.py` is the standalone Marimo executable tutorial for Level 5;
`level_05_ralph_loop.py` is kept as the command-line entry point. To open Level 5 on its own:

```powershell
marimo run demos/level_05_ralph_omnigent_codex_loop.py
```