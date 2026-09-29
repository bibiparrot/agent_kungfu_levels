import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="full",
    app_title="Level 5 · Ralph Loop + OmniGenT + Codex",
)


@app.cell
def _():
    import importlib.util
    import json
    import shutil
    import time
    import urllib.request

    import marimo as mo
    import polars as pl

    from agent_kungfu.config import Settings
    from agent_kungfu.levels import DEFAULT_QUESTION, run_level
    from agent_kungfu.levels.common import load_level_config
    from agent_kungfu.levels.teaching import (
        evaluate_level_result,
        execution_observations,
        get_teaching_material,
        load_source_documents,
        source_tree,
    )

    return (
        DEFAULT_QUESTION,
        Settings,
        evaluate_level_result,
        execution_observations,
        get_teaching_material,
        importlib,
        json,
        load_level_config,
        load_source_documents,
        mo,
        pl,
        run_level,
        shutil,
        source_tree,
        time,
        urllib,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Level 5 · Loop Agent

    ## Ralph Loop + OmniGenT + Codex：检查全部表，验证收入，然后退出

    这是一份可直接运行的独立 Marimo 教材。它把 Level 5 从“失败后重试”落实为一个
    **有状态、有反馈、有预算、有停止证明**的执行循环：

    **读取目录 → 逐表检查 → 评价覆盖率 → 绑定反馈 → 计算类别收入 → 与 Baseline 对账 → 立即停止。**

    Sakila 有 16 张业务表，因此成功轨迹固定包含 16 次 `inspect_table` 和 1 次
    `calculate_revenue`。Codex 只提出类型化动作，数据库与 Polars 始终由宿主白名单执行。
    """)
    return


@app.cell
def _(get_teaching_material, load_level_config, load_source_documents, source_tree):
    level5_material = get_teaching_material(5)
    level5_config = load_level_config(5)
    level5_sources = load_source_documents(level5_material)
    level5_source_tree = source_tree(level5_sources)
    return level5_config, level5_material, level5_source_tree, level5_sources


@app.cell(hide_code=True)
def _(level5_material, mo):
    build_rows = [
        {"步骤": step.title, "构建动作": step.action, "可观察证据": step.expected}
        for step in level5_material.steps
    ]
    prerequisites = "\n".join(f"- {item}" for item in level5_material.prerequisites)
    observations = "\n".join(f"- {item}" for item in level5_material.observations)
    checkpoints = "\n".join(f"- {item}" for item in level5_material.checkpoints)
    mo.vstack(
        [
            mo.md(f"## Goal\n\n{level5_material.goal}"),
            mo.callout(
                level5_material.change_from_previous,
                kind="info",
                title="从 L4 到 L5：控制权迁移",
            ),
            mo.md(f"### Prerequisites\n\n{prerequisites}"),
            mo.md("## Build · 把一次执行变成可验证循环"),
            mo.ui.table(
                build_rows,
                selection=None,
                pagination=False,
                show_download=False,
                wrapped_columns=["构建动作", "可观察证据"],
            ),
            mo.md("## Architecture · 17 步成功路径与反馈回边"),
            mo.mermaid(level5_material.architecture, theme="neutral"),
            mo.md(f"### 运行时重点观察\n\n{observations}"),
            mo.md(f"### 概念检查\n\n{checkpoints}"),
        ],
        gap=1.1,
    )
    return


@app.cell(hide_code=True)
def _(level5_source_tree, level5_sources, mo):
    source_tabs = {
        document.label: mo.vstack(
            [
                mo.md(f"**源码定位：** `{document.reference}`"),
                mo.ui.code_editor(
                    value=document.content,
                    language=document.language,
                    disabled=True,
                    min_height=260,
                    max_height=620,
                    show_copy_button=True,
                    label=document.reference,
                ),
            ]
        )
        for document in level5_sources
    }
    mo.vstack(
        [
            mo.md(
                "## Read · 源码结构与关键实现\n\n"
                "源码标签来自中央 `TeachingMaterial`；先读 YAML 控制合同，再读状态、"
                "动作执行与十项 fail-closed evaluator。"
            ),
            mo.md(f"```text\n{level5_source_tree}\n```"),
            mo.ui.tabs(source_tabs, lazy=False),
        ],
        gap=1,
    )
    return


@app.cell
def _(Settings, importlib, json, level5_config, shutil, urllib):
    lesson_settings = Settings.from_env()
    setup_rows = []
    table_names = []
    try:
        lesson_settings.validate()
        from agent_kungfu.database import SakilaDB

        table_names = SakilaDB(lesson_settings).table_names()
        database_status = "PASS" if len(table_names) == 16 else "WARN"
        database_detail = f"{lesson_settings.db_path.resolve()} · {len(table_names)} tables"
    except Exception as _database_exc:
        database_status = "FAIL"
        database_detail = f"{type(_database_exc).__name__}: {_database_exc}"
    setup_rows.append(
        {
            "前置条件": "Sakila database catalog",
            "Offline": database_status,
            "Live": database_status,
            "当前解析值": database_detail,
        }
    )

    omnigent_installed = importlib.util.find_spec("omnigent_client") is not None
    setup_rows.append(
        {
            "前置条件": "omnigent-client >= 0.11",
            "Offline": "OPTIONAL",
            "Live": "PASS" if omnigent_installed else "MISSING",
            "当前解析值": "installed" if omnigent_installed else "pip install -e .[harness]",
        }
    )
    codex_path = shutil.which("codex")
    setup_rows.append(
        {
            "前置条件": "Local Codex executable (advisory)",
            "Offline": "SIMULATED",
            "Live": "FOUND" if codex_path else "CHECK RUNNER HOST",
            "当前解析值": codex_path
            or "client PATH has no codex; OmniGenT runner may still provide it",
        }
    )
    setup_rows.append(
        {
            "前置条件": "OMNIGENT_URL",
            "Offline": "NOT USED",
            "Live": "PASS" if lesson_settings.omnigent_url else "MISSING",
            "当前解析值": lesson_settings.omnigent_url or "environment variable is unset",
        }
    )
    setup_rows.append(
        {
            "前置条件": "Codex runner selection",
            "Offline": "NOT USED",
            "Live": "EXPLICIT" if lesson_settings.omnigent_runner_id else "AUTO",
            "当前解析值": lesson_settings.omnigent_runner_id
            or "resolve one online runner with harness=codex",
        }
    )
    try:
        with urllib.request.urlopen(
            f"{lesson_settings.ollama_base_url.rstrip('/')}/models",
            timeout=1.5,
        ) as _ollama_response:
            _ollama_payload = json.loads(_ollama_response.read().decode("utf-8"))
        _model_items = _ollama_payload.get("data", _ollama_payload.get("models", []))
        _model_names = {str(item.get("id", item.get("name", ""))) for item in _model_items}
        model_available = any(
            lesson_settings.ollama_model == name or lesson_settings.ollama_model in name
            for name in _model_names
        )
        model_detail = f"service online; installed={model_available}"
    except Exception as _ollama_exc:
        model_available = False
        model_detail = f"{type(_ollama_exc).__name__}: {_ollama_exc}"
    setup_rows.append(
        {
            "前置条件": f"Ollama model · {lesson_settings.ollama_model}",
            "Offline": "NOT USED",
            "Live": "LOCAL PASS" if model_available else "CHECK RUNNER HOST",
            "当前解析值": model_detail,
        }
    )
    setup_rows.append(
        {
            "前置条件": "Ralph stop contract",
            "Offline": "PASS",
            "Live": "PASS",
            "当前解析值": (
                f"tables={level5_config['loop']['expected_table_count']}; "
                f"budget={level5_config['loop']['max_iterations']}; "
                f"stop={level5_config['loop']['stop_condition']}"
            ),
        }
    )
    core_ready = database_status == "PASS"
    live_ready_hint = bool(core_ready and omnigent_installed and lesson_settings.omnigent_url)
    return core_ready, lesson_settings, live_ready_hint, setup_rows, table_names


@app.cell(hide_code=True)
def _(live_ready_hint, mo, setup_rows, table_names):
    live_message = (
        "客户端前置检查已通过；运行时仍会确认在线 runner，并由 runner 主机提供 Codex 与 Ollama。"
        if live_ready_hint
        else "Live 尚缺少至少一项前置条件；Offline 可真实运行 16 表检查、Polars 与 evaluator。"
    )
    catalog = [
        {"序号": index, "待检查表": table} for index, table in enumerate(table_names, start=1)
    ]
    mo.vstack(
        [
            mo.md("## Setup · 先确认循环预算能覆盖完整目录"),
            mo.ui.table(
                setup_rows,
                selection=None,
                pagination=False,
                show_download=False,
                wrapped_columns=["当前解析值"],
            ),
            mo.callout(live_message, kind="success" if live_ready_hint else "warn"),
            mo.accordion(
                {
                    f"本次目录快照 · {len(catalog)} tables": mo.ui.table(
                        catalog,
                        selection=None,
                        pagination=False,
                        show_download=False,
                    )
                }
            ),
            mo.md(
                "**Offline：** OmniGenT/Codex tool calls 标为 `simulated`，但每张表的 SQLite "
                "检查、收入 Polars 计算、SQL baseline 对账和 10 项评价均真实执行。\n\n"
                "**Live：** 还需要 `OMNIGENT_URL`，且在线 runner 主机必须能启动 Codex 并访问 "
                "Ollama `ornith-1.5:9b`；本页对客户端 PATH/localhost 的检查只是提示，不是 runner "
                "健康证明。每轮必须恰好发生一次 completed server-executed agent-tool call。\n\n"
                "`sandbox_*_declared` 与 `approval_policy_declared` 展示 agent YAML 的声明意图；"
                "跨平台硬边界仍是 Pydantic action、宿主 allowlist 与 fail-closed evaluator。"
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(DEFAULT_QUESTION, mo):
    top_n = mo.ui.slider(1, 16, value=5, step=1, label="Top N categories")
    run_mode = mo.ui.radio(
        options={
            "Offline · 可复现教学（推荐）": "offline",
            "Live · OmniGenT + Codex + Ollama": "live",
        },
        value="Offline · 可复现教学（推荐）",
        inline=True,
        label="运行模式",
    )
    business_question = mo.ui.text_area(
        value=DEFAULT_QUESTION,
        rows=3,
        full_width=True,
        debounce=True,
        label="业务问题",
    )
    run_lesson = mo.ui.run_button(
        label="▶ Run 17-step Ralph loop",
        kind="success",
        full_width=True,
        keyboard_shortcut="Ctrl-Enter",
    )
    mo.vstack(
        [
            mo.md("## Predict & Run · 先预测停止条件，再显式执行"),
            mo.callout(
                "预测：16 张表全部检查后，计算步骤应发生在第 17 轮；只有 baseline parity "
                "通过才能停止。运行后用轨迹验证，而不是相信这句话。",
                kind="info",
            ),
            mo.hstack([run_mode, top_n], widths="equal", align="end"),
            business_question,
            run_lesson,
            mo.md("> Top-N 只改变最终选择行数，不改变目录快照、16 次逐表检查、反馈绑定和停止门。"),
        ],
        gap=1,
    )
    return business_question, run_lesson, run_mode, top_n


@app.cell
def _(mo):
    get_level5_run, set_level5_run = mo.state(None)
    return get_level5_run, set_level5_run


@app.cell
def _(
    business_question,
    core_ready,
    lesson_settings,
    run_lesson,
    run_level,
    run_mode,
    set_level5_run,
    time,
    top_n,
):
    if run_lesson.value:
        _started = time.perf_counter()
        _question = business_question.value.strip()
        _payload = {
            "inputs": {
                "question": _question,
                "mode": run_mode.value,
                "top_n": top_n.value,
            }
        }
        try:
            if not core_ready:
                raise RuntimeError("Sakila setup preflight failed")
            # Put the normalized constraint first: DEFAULT_QUESTION already contains "前 5",
            # and the runtime intentionally accepts the first valid Top-N expression.
            _prompt = f"Show the top {top_n.value} categories. Question: {_question}"
            _payload["result"] = run_level(
                5,
                _prompt,
                settings=lesson_settings,
                offline=run_mode.value == "offline",
            )
        except Exception as _run_exc:
            _payload["error"] = f"{type(_run_exc).__name__}: {_run_exc}"
        _payload["elapsed"] = time.perf_counter() - _started
        set_level5_run(_payload)
    return


@app.cell
def _(get_level5_run):
    level5_run = get_level5_run()
    return (level5_run,)


@app.cell(hide_code=True)
def _(
    business_question,
    evaluate_level_result,
    execution_observations,
    json,
    level5_run,
    mo,
    pl,
    run_mode,
    top_n,
):
    if level5_run is None:
        result_view = mo.callout(
            "选择 Top-N 与运行模式，然后点击 Run。结果会保留为快照；改变控件不会偷偷重跑。",
            kind="neutral",
            title="等待运行",
        )
    elif "error" in level5_run:
        _live_help = (
            "Live 请确认 `OMNIGENT_URL`、在线 Codex runner（或 `OMNIGENT_RUNNER_ID`）、"
            "Ollama 与 `ornith-1.5:9b`；也可切换 Offline 验证完整 17 步本地路径。"
        )
        result_view = mo.callout(
            mo.md(f"**{level5_run['error']}**\n\n{_live_help}"),
            kind="danger",
            title="运行失败",
        )
    else:
        _result = level5_run["result"]
        _metadata = _result.metadata
        _result_frame = pl.DataFrame(_metadata.get("rows", []))
        _iterations = _metadata.get("iterations", [])
        _observations = _metadata.get("table_observations", {})
        _revenue_chain = _metadata.get("revenue_chain", [])
        _harness_evaluation = _metadata.get("evaluation", {})
        _harness_checks = _harness_evaluation.get("checks", [])
        _harness_check_rows = [
            {
                "状态": "PASS" if item.get("passed") else "FAIL",
                "检查": item.get("name"),
                "说明": item.get("detail"),
            }
            for item in _harness_checks
        ]
        _passed_checks = sum(bool(item.get("passed")) for item in _harness_checks)
        _teaching_checks = evaluate_level_result(_result)
        _journal = execution_observations(5, result=_result)
        _exploration_rows = [
            {
                "顺序": index,
                "表": table,
                "行数": _observations.get(table, {}).get("row_count"),
                "列数": len(_observations.get(table, {}).get("columns", [])),
                "收入链": "YES" if table in _revenue_chain else "—",
                "收入线索": ", ".join(_observations.get(table, {}).get("revenue_signals", []))
                or "—",
            }
            for index, table in enumerate(_metadata.get("inspected_tables", []), start=1)
        ]
        _iteration_rows = [
            {
                "轮次": item.get("iteration"),
                "阶段": item.get("action", {}).get("runtime_tool"),
                "对象": item.get("action", {}).get("table")
                or f"Top-{item.get('action', {}).get('n')}",
                "反馈摘要": str(item.get("action", {}).get("feedback_digest_sha256", ""))[:12],
                "Skill摘要": str(item.get("action", {}).get("skill_digest_sha256", ""))[:12],
                "得分": item.get("evaluation", {}).get("score"),
                "评价反馈": "; ".join(item.get("evaluation", {}).get("feedback", [])),
                "决策": item.get("decision"),
            }
            for item in _iterations
        ]
        _tool_call_rows = [
            {
                "轮次": item.get("iteration"),
                "tool": item.get("name"),
                "agent": item.get("agent_name"),
                "executed_by": item.get("executed_by"),
                "completed": item.get("completed"),
                "output_sha256": str(item.get("output_sha256") or "")[:12],
                "call_id": item.get("call_id"),
            }
            for item in _metadata.get("omnigent_tool_calls", [])
        ]
        _trace_rows = [
            {
                "protocol": item.protocol.value,
                "sender": item.sender,
                "recipient": item.recipient,
                "kind": item.kind,
                "correlation_id": item.correlation_id,
                "payload": json.dumps(item.payload, ensure_ascii=False, default=str),
            }
            for item in _result.trace
        ]
        _stale = (
            level5_run["inputs"]["question"] != business_question.value.strip()
            or level5_run["inputs"]["mode"] != run_mode.value
            or level5_run["inputs"]["top_n"] != top_n.value
        )
        _freshness_panel = (
            mo.callout(
                "控件已经改变；下方仍是上一次运行快照。再次点击 Run 才会更新。",
                kind="warn",
                title="Stale result",
            )
            if _stale
            else mo.callout("结果与当前控件一致。", kind="success", title="Fresh result")
        )
        _stopped_safely = bool(
            _metadata.get("terminated")
            and _metadata.get("stop_reason") == "goal_satisfied"
            and _harness_evaluation.get("passed")
        )
        _stop_panel = mo.callout(
            (
                f"第 {_metadata.get('iterations_used')} 轮立即退出：全部表已检查，"
                "收入结果通过 deterministic baseline parity。"
                if _stopped_safely
                else "循环没有取得可交付的停止证明。"
            ),
            kind="success" if _stopped_safely else "danger",
            title=f"STOP · {_metadata.get('stop_reason', 'unknown')}",
        )
        _winner = _result_frame.row(0, named=True) if _result_frame.height else {}
        _boundary = (
            "server-executed OmniGenT/Codex delegation"
            if _metadata.get("omnigent_connected")
            else "simulated OmniGenT/Codex delegation + real host execution"
        )
        result_view = mo.vstack(
            [
                mo.md("## Observe · 全表探索、反馈轨迹与停止证明"),
                _freshness_panel,
                _stop_panel,
                mo.hstack(
                    [
                        mo.stat(
                            f"{len(_metadata.get('inspected_tables', []))}/16",
                            label="Tables inspected",
                            bordered=True,
                        ),
                        mo.stat(
                            f"{_metadata.get('iterations_used')}/{_metadata.get('max_iterations')}",
                            label="Ralph iterations",
                            bordered=True,
                        ),
                        mo.stat(
                            f"{_passed_checks}/{len(_harness_checks)}",
                            label="Harness checks",
                            bordered=True,
                        ),
                        mo.stat(
                            (
                                f"{_winner.get('name')} / {float(_winner.get('revenue', 0)):.2f}"
                                if _winner
                                else "—"
                            ),
                            label="Revenue winner",
                            bordered=True,
                        ),
                    ],
                    widths="equal",
                ),
                mo.callout(
                    f"本次边界：{_boundary}；runner={_metadata.get('omnigent_runner_id') or 'none'}；"
                    f"session={_metadata.get('omnigent_session_id') or 'none'}；"
                    f"真实 Codex subagent calls={_metadata.get('codex_subagent_calls')}。",
                    kind="info",
                ),
                mo.callout(
                    "从 16 张表的实际列证据中发现收入链：" + " → ".join(_revenue_chain),
                    kind="info",
                    title="Revenue chain discovered",
                ),
                mo.ui.tabs(
                    {
                        "16 表探索证据": mo.ui.table(
                            _exploration_rows,
                            selection=None,
                            pagination=False,
                            show_download=True,
                        ),
                        "17 轮 Ralph 轨迹": mo.ui.table(
                            _iteration_rows,
                            selection=None,
                            pagination=False,
                            show_download=True,
                            wrapped_columns=["评价反馈"],
                        ),
                        "收入结果": mo.vstack(
                            [
                                mo.md(_result.answer),
                                mo.ui.table(
                                    _result_frame,
                                    selection=None,
                                    pagination=False,
                                    show_download=True,
                                ),
                            ]
                        ),
                        "10 项停止评价": mo.vstack(
                            [
                                mo.callout(
                                    "只有十项全部通过，runtime 才返回结果。",
                                    kind="success"
                                    if _harness_evaluation.get("passed")
                                    else "danger",
                                ),
                                mo.ui.table(
                                    _harness_check_rows,
                                    selection=None,
                                    pagination=False,
                                    show_download=False,
                                    wrapped_columns=["说明"],
                                ),
                            ]
                        ),
                        "OmniGenT / Codex calls": mo.ui.table(
                            _tool_call_rows,
                            selection=None,
                            pagination=True,
                            show_download=False,
                        ),
                        "Protocol trace": mo.ui.table(
                            _trace_rows,
                            selection=None,
                            pagination=True,
                            show_download=False,
                            wrapped_columns=["payload"],
                        ),
                        "逐步运行记录": mo.ui.table(
                            _journal,
                            selection=None,
                            pagination=False,
                            show_download=False,
                            wrapped_columns=["运行证据", "意义"],
                        ),
                        "教材级验收": mo.ui.table(
                            [item.as_record() for item in _teaching_checks],
                            selection=None,
                            pagination=False,
                            show_download=False,
                            wrapped_columns=["期望", "实际", "失败后怎么做"],
                        ),
                        "关键 metadata": mo.md(
                            "```json\n"
                            + json.dumps(
                                {
                                    "stop_reason": _metadata.get("stop_reason"),
                                    "all_tables_inspected": _metadata.get("all_tables_inspected"),
                                    "iterations_used": _metadata.get("iterations_used"),
                                    "max_iterations": _metadata.get("max_iterations"),
                                    "mode": _metadata.get("mode"),
                                    "generated_agents": _metadata.get("generated_agents"),
                                    "runtime_tools": _metadata.get("runtime_tools"),
                                    "revenue_chain": _revenue_chain,
                                    "skill_digest_sha256": _metadata.get("skill_digest_sha256"),
                                    "bundle_sha256": _metadata.get("bundle_sha256"),
                                    "codex_subagent_calls": _metadata.get("codex_subagent_calls"),
                                    "sandbox_intent": _metadata.get("sandbox_intent"),
                                    "sandbox_backend_declared": _metadata.get(
                                        "sandbox_backend_declared"
                                    ),
                                    "sandbox_write_paths_declared": _metadata.get(
                                        "sandbox_write_paths_declared"
                                    ),
                                    "sandbox_write_files_declared": _metadata.get(
                                        "sandbox_write_files_declared"
                                    ),
                                    "sandbox_allow_network_declared": _metadata.get(
                                        "sandbox_allow_network_declared"
                                    ),
                                    "sandbox_start_in_scratch_declared": _metadata.get(
                                        "sandbox_start_in_scratch_declared"
                                    ),
                                    "approval_policy_declared": _metadata.get(
                                        "approval_policy_declared"
                                    ),
                                    "joined_rows": _metadata.get("joined_rows"),
                                    "revenue_rows": _metadata.get("revenue_rows"),
                                },
                                ensure_ascii=False,
                                indent=2,
                                default=str,
                            )
                            + "\n```"
                        ),
                    },
                    value="17 轮 Ralph 轨迹",
                ),
                mo.md(
                    "## Next · 两个验证练习\n\n"
                    "1. 把 Top-N 改为 1 与 16；确认目录仍检查 16 张表，且只改变最终选择行数。\n"
                    "2. 阅读 `evaluate_level5_run`，解释为什么第 16 轮仍不能停止，以及第 17 轮必须同时满足哪些条件。"
                ),
            ],
            gap=1.1,
        )
    result_view
    return


@app.cell(hide_code=True)
def _(mo):
    capability_level = 5
    capability_run = mo.ui.run_button(
        label="运行能力干预实验 / Run capability experiment",
        disabled=capability_level == 0,
    )
    mo.vstack(
        [
            mo.md(
                "## 能力实验 / Capability experiment\n\n在隔离临时目录执行正常、错误或阴性对照；L9–L10 的效应与创新结果来自合成模型。模型服务不参与本实验。"
            ),
            capability_run,
        ]
    )
    return capability_level, capability_run


@app.cell(hide_code=True)
def _(mo, capability_level, capability_run):
    mo.stop(not capability_run.value)
    from agent_kungfu.levels.experiments import run_experiment as _run_experiment

    with mo.status.spinner("Running controlled intervention…"):
        _report = _run_experiment(capability_level)
    mo.vstack(
        [
            mo.ui.table(_report["checks"], selection=None, pagination=False),
            mo.md("Evidence scope: " + _report["model_evidence"]),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
