from __future__ import annotations

import asyncio
import json
from typing import Any, TypedDict

import networkx as nx
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..config import PROJECT_ROOT, Settings
from ..contracts import AgentProposal, HarnessManifest, LevelResult, Protocol
from ..database import SakilaDB, choose_catalog_query
from ..evaluation import evaluate_answer_with_deepeval, evaluate_sql, summarize_frame
from ..graph_engineering import (
    build_schema_graph,
    generate_sakila_ontology,
    generate_wiki,
    graph_summary,
    shortest_join_path,
)
from ..llm import OllamaLLM, extract_sql
from ..orchestration import (
    CodexReviewer,
    ConflictResolver,
    OmnigentObserver,
    ProtocolBus,
    RalphLoop,
    offline_proposal,
    prompt_sql_proposal,
)
from ..skills import SkillRegistry


DEFAULT_QUESTION = "比较两个门店的收入、租赁量和客户数，并说明观察到的差异。"
PROMPT_DIR = PROJECT_ROOT / "prompts"


class WorkflowState(TypedDict, total=False):
    question: str
    prompt: str
    sql: str
    rows: list[dict[str, Any]]
    answer: str


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(PROMPT_DIR),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _runtime(settings: Settings | None = None) -> tuple[Settings, SakilaDB, OllamaLLM]:
    resolved = settings or Settings.from_env()
    resolved.validate()
    return resolved, SakilaDB(resolved), OllamaLLM(resolved)


def level_01(question: str, *, settings: Settings | None = None, offline: bool = False) -> LevelResult:
    """Prompt + Jinja2 + LangGraph: the prompt controls the agent."""

    from langgraph.graph import END, START, StateGraph

    _, db, llm = _runtime(settings)
    template = _environment().get_template("sql_analyst.j2")

    def render(state: WorkflowState) -> WorkflowState:
        return {
            "prompt": template.render(question=state["question"], schema=db.schema_text(), max_rows=20)
        }

    def plan(state: WorkflowState) -> WorkflowState:
        if offline:
            return {"sql": choose_catalog_query(state["question"])}
        return {"sql": extract_sql(llm.complete(state["prompt"], max_tokens=600))}

    def execute(state: WorkflowState) -> WorkflowState:
        frame = db.query(state["sql"], limit=20)
        return {"rows": db.records(frame, max_rows=20)}

    def explain(state: WorkflowState) -> WorkflowState:
        mode = "离线计划" if offline else "Ollama 计划"
        return {"answer": f"{mode}返回 {len(state['rows'])} 行证据：{state['rows'][:5]}"}

    graph = StateGraph(WorkflowState)
    graph.add_node("render_jinja_prompt", render)
    graph.add_node("llm_plan_sql", plan)
    graph.add_node("polars_execute", execute)
    graph.add_node("llm_explain", explain)
    graph.add_edge(START, "render_jinja_prompt")
    graph.add_edge("render_jinja_prompt", "llm_plan_sql")
    graph.add_edge("llm_plan_sql", "polars_execute")
    graph.add_edge("polars_execute", "llm_explain")
    graph.add_edge("llm_explain", END)
    state = graph.compile().invoke({"question": question})
    return LevelResult(
        level=1,
        driver="提示词智能体：Jinja2 prompt + LangGraph workflow + LiteLLM",
        question=question,
        answer=state["answer"],
        sql=state["sql"],
        row_count=len(state["rows"]),
        metadata={"graph_nodes": ["render", "plan", "execute", "explain"]},
    )


def level_02(question: str, *, settings: Settings | None = None, offline: bool = False) -> LevelResult:
    """Code-owned skills + OpenAI Agents SDK Runner."""

    resolved, db, _ = _runtime(settings)
    registry = SkillRegistry(db)
    if offline:
        proposal = offline_proposal(question)
        evidence = registry.get("run_readonly_sql").invoke(proposal.sql)
        return LevelResult(
            level=2,
            driver="代码智能体：skills + openai-agents Runner",
            question=question,
            answer=f"Skill registry executed reviewed SQL. Evidence: {evidence}",
            sql=proposal.sql,
            row_count=len(json.loads(evidence)),
            metadata={"skills": registry.names(), "sdk": "openai-agents"},
        )

    from agents import Agent, ModelSettings, Runner, function_tool, set_tracing_disabled
    from agents.extensions.models.litellm_model import LitellmModel

    set_tracing_disabled(True)

    @function_tool
    def inspect_sakila_schema() -> str:
        """Return the actual Sakila SQLite DDL."""

        return registry.get("inspect_schema").invoke()

    @function_tool
    def query_sakila(sql: str) -> str:
        """Execute one read-only SELECT/CTE and return up to 50 rows as JSON."""

        return registry.get("run_readonly_sql").invoke(sql)

    model = LitellmModel(
        model=resolved.litellm_model,
        base_url=resolved.ollama_base_url,
        api_key=resolved.ollama_api_key,
    )
    agent = Agent(
        name="sakila-skill-agent",
        instructions=(
            "Inspect schema, then call query_sakila exactly once. Answer in Chinese with concrete "
            "evidence and never claim observational differences are causal."
        ),
        tools=[inspect_sakila_schema, query_sakila],
        model=model,
        model_settings=ModelSettings(extra_args={"custom_llm_provider": "openai"}),
    )
    run = Runner.run_sync(agent, question, max_turns=6)
    return LevelResult(
        level=2,
        driver="代码智能体：skills + openai-agents Runner",
        question=question,
        answer=str(run.final_output),
        metadata={"skills": registry.names(), "sdk": "openai-agents"},
    )


def level_03(question: str, *, settings: Settings | None = None, offline: bool = False) -> LevelResult:
    """Skill calls cross prompt/A2A/Codex JSON-RPC protocol boundaries."""

    resolved, db, llm = _runtime(settings)
    bus = ProtocolBus()
    request = bus.send(Protocol.A2A, "user", "planner", "analysis.request", {"question": question})
    if offline:
        planned = offline_proposal(question, agent="skill-planner")
        reviewed = AgentProposal(
            agent="codex-reviewer-simulated",
            sql=planned.sql,
            rationale="Offline protocol simulation",
            confidence=0.9,
            protocol=Protocol.CODEX_JSONRPC,
        )
    else:
        planned = prompt_sql_proposal(llm, db, question)
        reviewed = CodexReviewer(resolved).review(question, planned.sql, db.schema_text())
    bus.send(
        planned.protocol,
        planned.agent,
        "codex-reviewer",
        "sql.proposed",
        planned.model_dump(mode="json"),
        correlation_id=request.id,
    )
    bus.send(
        Protocol.CODEX_JSONRPC,
        reviewed.agent,
        "polars-worker",
        "sql.reviewed",
        reviewed.model_dump(mode="json"),
        correlation_id=request.id,
    )
    frame = db.query(reviewed.sql, limit=20)
    bus.send(
        Protocol.MCP,
        "polars-worker",
        "answer-agent",
        "evidence.rows",
        {"rows": db.records(frame)},
        correlation_id=request.id,
    )
    return LevelResult(
        level=3,
        driver="协议智能体：skills + A2A/MCP envelopes + openai-codex JSON-RPC",
        question=question,
        answer=summarize_frame(frame),
        sql=reviewed.sql,
        row_count=frame.height,
        trace=bus.messages,
        metadata={"protocols": sorted({message.protocol.value for message in bus.messages})},
    )


def level_04(question: str, *, settings: Settings | None = None, offline: bool = False) -> LevelResult:
    """Harness arbitrates competing agents; Omnigent observes the decision."""

    resolved, db, llm = _runtime(settings)
    bus = ProtocolBus()
    first = offline_proposal(question, agent="skill-agent") if offline else prompt_sql_proposal(llm, db, question)
    if offline:
        second = AgentProposal(
            agent="codex-agent",
            sql=first.sql.replace("ORDER BY revenue DESC", "ORDER BY store_id"),
            rationale="Alternative presentation order to demonstrate conflict arbitration.",
            confidence=0.72,
            protocol=Protocol.CODEX_JSONRPC,
        )
    else:
        second = CodexReviewer(resolved).review(question, first.sql, db.schema_text())
    decision = ConflictResolver(db).resolve([first, second], question)
    for proposal in (first, second):
        bus.send(proposal.protocol, proposal.agent, "harness", "proposal", proposal.model_dump(mode="json"))
    bus.send(Protocol.A2A, "harness", decision.selected_agent, "conflict.resolved", decision.model_dump(mode="json"))
    observed = False
    if not offline and resolved.omnigent_url:
        observed = asyncio.run(OmnigentObserver(resolved.omnigent_url).publish(decision.model_dump(mode="json")))
        bus.send(Protocol.OMNIGENT_HTTP, "harness", "omnigent", "audit", {"published": observed})
    frame = db.query(decision.selected_sql, limit=20)
    return LevelResult(
        level=4,
        driver="协议+冲突控制：agent orchestration + harness + omnigent-client",
        question=question,
        answer=summarize_frame(frame),
        sql=decision.selected_sql,
        row_count=frame.height,
        trace=bus.messages,
        metadata={"decision": decision.model_dump(mode="json"), "omnigent_published": observed},
    )


def level_05(question: str, *, settings: Settings | None = None, offline: bool = False) -> LevelResult:
    """Ralph loop turns evaluator feedback into a bounded improvement flywheel."""

    resolved, db, llm = _runtime(settings)
    loop = RalphLoop(db, max_iterations=3)

    def propose(index: int, previous) -> AgentProposal:
        if offline:
            if index == 1:
                return AgentProposal(
                    agent="loop-agent",
                    sql="SELECT title FROM film LIMIT 10",
                    rationale="Intentionally weak first draft: no business metric.",
                    confidence=0.4,
                )
            return offline_proposal(question, agent="loop-agent")
        feedback = "none" if previous is None else "; ".join(previous.feedback)
        response = llm.complete(
            f"Question: {question}\nSchema: {db.schema_text()}\nPrevious feedback: {feedback}\n"
            "Return an improved read-only SELECT with a decision-oriented metric.",
            system="You are the worker inside a bounded Ralph improvement loop.",
        )
        draft = AgentProposal(
            agent="loop-agent",
            sql=extract_sql(response),
            rationale=f"Ralph iteration {index}",
            confidence=min(0.55 + index * 0.1, 0.9),
        )
        return CodexReviewer(resolved).review(question, draft.sql, db.schema_text())

    history = loop.run(question, propose)
    final = history[-1]
    frame = db.query(final.proposal.sql, limit=20)
    observer = OmnigentObserver(resolved.omnigent_url)
    published = False
    if not offline and resolved.omnigent_url:
        published = asyncio.run(
            observer.publish({"loop": [item.model_dump(mode="json") for item in history]})
        )
    return LevelResult(
        level=5,
        driver="冲突控制+迭代飞轮：harness + Omnigent + Ralph loop",
        question=question,
        answer=summarize_frame(frame),
        sql=final.proposal.sql,
        row_count=frame.height,
        metadata={
            "iterations": [item.model_dump(mode="json") for item in history],
            "omnigent_published": published,
        },
    )


def level_06(
    question: str,
    *,
    settings: Settings | None = None,
    offline: bool = False,
    use_deepeval: bool = False,
) -> LevelResult:
    """Typed harness schema + relationship graph + deterministic/DeepEval gates."""

    _, db, llm = _runtime(settings)
    manifest = HarnessManifest(
        name="sakila-graph-harness",
        allowed_protocols=[Protocol.SKILL, Protocol.MCP, Protocol.A2A],
    )
    graph = build_schema_graph(db)
    loop_history = RalphLoop(db, max_iterations=manifest.max_iterations).run(
        question, lambda _index, _previous: offline_proposal(question, agent="graph-loop-agent")
    )
    proposal = loop_history[-1].proposal
    frame = db.query(proposal.sql, limit=manifest.max_rows)
    answer = summarize_frame(frame)
    evaluation = (
        evaluate_answer_with_deepeval(question, answer, llm)
        if use_deepeval and not offline
        else evaluate_sql(db, proposal.sql, question)
    )
    return LevelResult(
        level=6,
        driver="数据控制+关系逻辑：Pydantic harness schema + NetworkX + DeepEval",
        question=question,
        answer=answer,
        sql=proposal.sql,
        row_count=frame.height,
        metadata={
            "manifest": manifest.model_dump(mode="json"),
            "graph": graph_summary(graph),
            "join_path_payment_to_film": shortest_join_path(graph, "payment", "film"),
            "evaluation": evaluation.model_dump(mode="json"),
            "ralph_iterations": [item.model_dump(mode="json") for item in loop_history],
        },
    )


def level_07(
    question: str,
    *,
    settings: Settings | None = None,
    offline: bool = False,
    use_deepeval: bool = False,
) -> LevelResult:
    """Persist schema and graph knowledge as an LLM-readable wiki."""

    resolved, db, llm = _runtime(settings)
    wiki_dir = resolved.output_dir / "level07_wiki"
    pages = generate_wiki(db, wiki_dir)
    graph = build_schema_graph(db)
    loop_history = RalphLoop(db, max_iterations=2).run(
        question, lambda _index, _previous: offline_proposal(question, agent="wiki-loop-agent")
    )
    sql = loop_history[-1].proposal.sql
    frame = db.query(sql, limit=20)
    evidence = summarize_frame(frame)
    answer = evidence
    if not offline:
        answer = llm.complete(
            _environment().get_template("wiki_answer.j2").render(
                question=question,
                graph=graph_summary(graph),
                wiki_index=pages[0].read_text(encoding="utf-8"),
                evidence=evidence,
            ),
            system="Use the generated Sakila wiki as durable knowledge, but ground claims in query evidence.",
        )
        llm_page = wiki_dir / "LLM_OVERVIEW.md"
        llm_page.write_text(f"# LLM-generated analysis\n\n{answer}\n", encoding="utf-8")
        pages.append(llm_page)
    evaluation = (
        evaluate_answer_with_deepeval(question, answer, llm)
        if use_deepeval and not offline
        else evaluate_sql(db, sql, question)
    )
    return LevelResult(
        level=7,
        driver="知识积累：LLM wiki + graph engineer + Ralph-compatible artifacts",
        question=question,
        answer=answer,
        sql=sql,
        row_count=frame.height,
        artifacts=[str(path) for path in pages],
        metadata={
            "wiki_pages": len(pages),
            "graph": graph_summary(graph),
            "ralph_iterations": [item.model_dump(mode="json") for item in loop_history],
            "evaluation": evaluation.model_dump(mode="json"),
        },
    )


def level_08(
    question: str,
    *,
    settings: Settings | None = None,
    offline: bool = False,
    use_deepeval: bool = False,
) -> LevelResult:
    """Produce OWL/OBDA/annotations, then ground the wiki and graph in that ontology."""

    resolved, db, llm = _runtime(settings)
    ontology_dir = resolved.output_dir / "level08_ontology"
    ontology = generate_sakila_ontology(db, ontology_dir)
    wiki_pages = generate_wiki(db, resolved.output_dir / "level08_wiki")
    graph = build_schema_graph(db)
    sql = choose_catalog_query(question)
    frame = db.query(sql, limit=20)
    answer = summarize_frame(frame)
    if not offline:
        answer = llm.complete(
            _environment().get_template("ontology_answer.j2").render(
                question=question,
                ontology_files=ontology.paths(),
                graph=graph_summary(graph),
                evidence=answer,
            ),
            system="Reason using the Sakila ontology and evidence; do not invent ontology terms.",
        )
        llm_page = resolved.output_dir / "level08_wiki" / "LLM_ONTOLOGY_OVERVIEW.md"
        llm_page.write_text(f"# LLM ontology-grounded analysis\n\n{answer}\n", encoding="utf-8")
        wiki_pages.append(llm_page)
    evaluation = (
        evaluate_answer_with_deepeval(question, answer, llm)
        if use_deepeval and not offline
        else evaluate_sql(db, sql, question)
    )
    return LevelResult(
        level=8,
        driver="知识生产：ontology-driven + LLM wiki generation + graph engineer",
        question=question,
        answer=answer,
        sql=sql,
        row_count=frame.height,
        artifacts=[*ontology.paths(), *(str(path) for path in wiki_pages)],
        metadata={
            "graph": graph_summary(graph),
            "evaluation": evaluation.model_dump(mode="json"),
        },
    )


def level_09(question: str, *, settings: Settings | None = None, offline: bool = False) -> LevelResult:
    """Start from a causal problem and make assumptions/confounders explicit."""

    resolved, db, llm = _runtime(settings)
    output = resolved.output_dir / "level09_causal"
    ontology = generate_sakila_ontology(db, output / "ontology")
    wiki = generate_wiki(db, output / "wiki")
    causal_sql = """
        SELECT i.store_id,
               COUNT(DISTINCT i.inventory_id) AS inventory_copies,
               COUNT(DISTINCT r.rental_id) AS rentals,
               COUNT(DISTINCT r.customer_id) AS customers,
               ROUND(SUM(p.amount), 2) AS revenue
        FROM inventory i
        LEFT JOIN rental r ON r.inventory_id = i.inventory_id
        LEFT JOIN payment p ON p.rental_id = r.rental_id
        GROUP BY i.store_id
        ORDER BY i.store_id
    """
    frame = db.query(causal_sql)
    dag = nx.DiGraph()
    dag.add_edges_from(
        [
            ("inventory_breadth", "rentals"),
            ("customer_demand", "rentals"),
            ("rentals", "revenue"),
            ("pricing", "revenue"),
            ("store_location", "customer_demand"),
            ("store_location", "inventory_breadth"),
        ]
    )
    caveat = (
        "门店差异是观察性关联，不是因果效应。store_location、需求和定价可能是混杂因素；"
        "要识别库存对收入的因果效应，需要随机/准实验或更完整的时间序列控制。"
    )
    answer = f"{summarize_frame(frame)} {caveat}"
    generated_artifacts: list[str] = []
    if not offline:
        answer = llm.complete(
            _environment().get_template("causal_problem.j2").render(
                question=question,
                dag=list(dag.edges),
                evidence=frame.to_dicts(),
                caveat=caveat,
            ),
            system="You are a causal analyst. Separate association, identification assumptions, and experiments.",
        )
        causal_wiki = output / "wiki" / "Causal_Problem.md"
        causal_wiki.write_text(f"# Causal problem\n\n{answer}\n", encoding="utf-8")
        ontology_proposal = output / "ontology" / "LLM_CAUSAL_ONTOLOGY_PROPOSAL.md"
        ontology_proposal.write_text(
            "# LLM-generated candidate semantic overlay\n\n"
            "This proposal is not merged into canonical OWL until validation.\n\n"
            f"DAG edges: {list(dag.edges)}\n\n{answer}\n",
            encoding="utf-8",
        )
        generated_artifacts.extend([str(causal_wiki), str(ontology_proposal)])
    return LevelResult(
        level=9,
        driver="问题驱动：causal problem + generated ontology/wiki + graph engineer",
        question=question,
        answer=answer,
        sql=causal_sql,
        row_count=frame.height,
        artifacts=[*ontology.paths(), *(str(path) for path in wiki), *generated_artifacts],
        metadata={"causal_dag": list(dag.edges), "is_dag": nx.is_directed_acyclic_graph(dag)},
    )


def level_10(question: str, *, settings: Settings | None = None, offline: bool = False) -> LevelResult:
    """Rebuild the business problem from invariants, constraints, and testable mechanisms."""

    resolved, db, llm = _runtime(settings)
    base = level_09(question, settings=resolved, offline=True)
    first_principles = {
        "objective": "increase expected contribution per available inventory-hour",
        "invariants": [
            "a physical inventory copy cannot serve two simultaneous rentals",
            "revenue is realized through payment linked to a rental",
            "customer demand is not created merely by adding inventory",
        ],
        "constraints": ["two stores", "observational historical data", "finite copies", "customer experience"],
        "mechanisms": [
            "rebalance copies toward unmet demand",
            "reduce return-to-next-rental idle time",
            "test category-specific price elasticity",
        ],
        "experiments": [
            "randomize inventory rebalancing by matched film-store pairs",
            "stagger reminder intervention and measure return latency",
            "run bounded price tests with retention guardrails",
        ],
    }
    answer = (
        "第一性原理方案：先优化“每个可用库存小时的预期贡献”，再分别验证库存错配、周转空闲和"
        "价格弹性三个机制。每项干预都需要随机化或分阶段实验，并以复租率/逾期率作护栏。"
    )
    generated_artifacts: list[str] = []
    if not offline:
        answer = llm.complete(
            _environment().get_template("first_principles.j2").render(
                question=question,
                causal_analysis=base.answer,
                first_principles=first_principles,
            ),
            system="Challenge assumptions and propose novel but falsifiable Sakila business experiments.",
        )
        level10_dir = resolved.output_dir / "level10_first_principles"
        level10_dir.mkdir(parents=True, exist_ok=True)
        wiki_page = level10_dir / "LLM_FIRST_PRINCIPLES_WIKI.md"
        ontology_proposal = level10_dir / "LLM_FIRST_PRINCIPLES_ONTOLOGY_PROPOSAL.md"
        wiki_page.write_text(f"# First-principles innovation\n\n{answer}\n", encoding="utf-8")
        ontology_proposal.write_text(
            "# Candidate ontology extension\n\n"
            "LLM-generated candidate; validate before merging into canonical OWL.\n\n"
            f"{first_principles}\n\n{answer}\n",
            encoding="utf-8",
        )
        generated_artifacts.extend([str(wiki_page), str(ontology_proposal)])
    return LevelResult(
        level=10,
        driver="创新驱动：first principles + ontology/wiki/graph + causal experiments",
        question=question,
        answer=answer,
        sql=base.sql,
        row_count=base.row_count,
        artifacts=[*base.artifacts, *generated_artifacts],
        metadata={**base.metadata, "first_principles": first_principles},
    )


LEVELS = {
    1: level_01,
    2: level_02,
    3: level_03,
    4: level_04,
    5: level_05,
    6: level_06,
    7: level_07,
    8: level_08,
    9: level_09,
    10: level_10,
}


def run_level(
    level: int,
    question: str = DEFAULT_QUESTION,
    *,
    settings: Settings | None = None,
    offline: bool = False,
    use_deepeval: bool = False,
) -> LevelResult:
    try:
        runner = LEVELS[level]
    except KeyError as exc:
        raise ValueError("level must be between 1 and 10") from exc
    if level in {6, 7, 8}:
        return runner(question, settings=settings, offline=offline, use_deepeval=use_deepeval)
    return runner(question, settings=settings, offline=offline)
