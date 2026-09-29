import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="full",
    app_title="Level 4 · OmniGenT + Codex + Polars Harness",
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
    # Level 4 · Harness Agent

    ## OmniGenT + Codex + Polars Skill：让能力边界可裁决

    这是一份可以直接运行的独立 Marimo 教材。它不复制收入分析业务逻辑，而是复用
    `agent_kungfu.levels` 的真实 Level 4 runtime 与中央 teaching helpers，带你依次完成：

    **理解控制面 → 阅读架构与源码 → 选择 Top-N 和运行模式 → 执行 → 查看结果与 Trace → 验证 10 项质量门。**

    Level 4 的关键变化不是“再增加一个 Agent”，而是引入 Harness：OmniGenT 负责会话、
    bundle 和委派，两个 Codex 子智能体只提出严格动作，宿主白名单执行真实 Polars 计算，
    确定性 evaluator 决定结果能否交付。
    """)
    return


@app.cell
def _(
    get_teaching_material,
    load_level_config,
    load_source_documents,
    source_tree,
):
    level4_material = get_teaching_material(4)
    level4_config = load_level_config(4)
    level4_sources = load_source_documents(level4_material)
    level4_source_tree = source_tree(level4_sources)
    return level4_config, level4_material, level4_source_tree, level4_sources


@app.cell(hide_code=True)
def _(level4_material, mo):
    build_steps = [
        {
            "步骤": step.title,
            "构建动作": step.action,
            "可观察证据": step.expected,
        }
        for step in level4_material.steps
    ]
    prerequisites = "\n".join(f"- {item}" for item in level4_material.prerequisites)
    observations = "\n".join(f"- {item}" for item in level4_material.observations)
    checkpoints = "\n".join(f"- {item}" for item in level4_material.checkpoints)

    mo.vstack(
        [
            mo.md(f"## Goal\n\n{level4_material.goal}"),
            mo.callout(
                level4_material.change_from_previous,
                kind="info",
                title="从 L3 到 L4：控制权迁移",
            ),
            mo.md(f"### Prerequisites\n\n{prerequisites}"),
            mo.md("## Build · 五步搭建控制面"),
            mo.ui.table(
                build_steps,
                selection=None,
                pagination=False,
                show_download=False,
                wrapped_columns=["构建动作", "可观察证据"],
            ),
            mo.md("## Architecture · 源码调用关系"),
            mo.mermaid(level4_material.architecture, theme="neutral"),
            mo.md(f"### 运行时重点观察\n\n{observations}"),
            mo.md(f"### 概念检查\n\n{checkpoints}"),
        ],
        gap=1.1,
    )
    return


@app.cell(hide_code=True)
def _(level4_source_tree, level4_sources, mo):
    source_tabs = {
        document.label: mo.vstack(
            [
                mo.md(f"**源码定位：** `{document.reference}`"),
                mo.ui.code_editor(
                    value=document.content,
                    language=document.language,
                    disabled=True,
                    min_height=260,
                    max_height=600,
                    show_copy_button=True,
                    label=document.reference,
                ),
            ]
        )
        for document in level4_sources
    }

    mo.vstack(
        [
            mo.md(
                "## Read · 源码结构与关键源码标签\n\n"
                "下面内容由中央 `TeachingMaterial` 清单动态抽取；符号级标签只展示关键函数，"
                "完整实现标签保留可运行上下文。"
            ),
            mo.md(f"```text\n{level4_source_tree}\n```"),
            mo.ui.tabs(source_tabs, lazy=False),
        ],
        gap=1,
    )
    return


@app.cell
def _(Settings, importlib, json, level4_config, shutil, urllib):
    lesson_settings = Settings.from_env()
    setup_rows = []

    try:
        lesson_settings.validate()
        database_status = "PASS"
        database_detail = str(lesson_settings.db_path.resolve())
    except Exception as _database_exc:
        database_status = "FAIL"
        database_detail = f"{type(_database_exc).__name__}: {_database_exc}"
    setup_rows.append(
        {
            "前置条件": "Sakila database",
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
            "前置条件": "Codex executable / runner harness",
            "Offline": "SIMULATED",
            "Live": "PASS" if codex_path else "MISSING",
            "当前解析值": codex_path or "codex not found on PATH",
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
            "Live": "PASS" if model_available else "UNAVAILABLE",
            "当前解析值": model_detail,
        }
    )

    setup_rows.append(
        {
            "前置条件": "Level 4 contract",
            "Offline": "PASS",
            "Live": "PASS",
            "当前解析值": (
                f"{level4_config['workers'][0]['id']} → "
                f"{level4_config['workers'][1]['id']}; "
                f"skill={level4_config['skill']['path']}"
            ),
        }
    )
    core_ready = database_status == "PASS"
    live_ready_hint = bool(
        core_ready
        and omnigent_installed
        and codex_path
        and lesson_settings.omnigent_url
        and model_available
    )
    return core_ready, lesson_settings, live_ready_hint, setup_rows


@app.cell(hide_code=True)
def _(live_ready_hint, mo, setup_rows):
    live_message = (
        "表面前置检查已通过；运行时仍会确认 OmniGenT 中存在在线 Codex runner。"
        if live_ready_hint
        else "Live 尚缺少至少一项前置条件；Offline 仍可真实执行数据库、Polars 和 evaluator。"
    )
    mo.vstack(
        [
            mo.md("## Setup · Offline 与 Live 边界"),
            mo.ui.table(
                setup_rows,
                selection=None,
                pagination=False,
                show_download=False,
                wrapped_columns=["当前解析值"],
            ),
            mo.callout(live_message, kind="success" if live_ready_hint else "warn"),
            mo.md(
                "**Live 必需：** `OMNIGENT_URL`、`omnigent-client>=0.11`、Ollama 标准本地接口中的 "
                "`ornith-1.5:9b`，以及 OmniGenT 上一个在线 `codex` runner。可用 "
                "`OMNIGENT_RUNNER_ID` 固定 runner；不设置时 runtime 会自动解析在线 Codex runner，"
                "没有 runner 会在上传 bundle 前快速失败。\n\n"
                "**Offline 的真实边界：** OmniGenT/Codex 委派被明确标记为 `simulated`，但 skill 校验、"
                "Sakila 读取、lazy Polars join/group-by、SQL baseline 对账与 10 项评价都是真实执行。"
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
        label="▶ Run Level 4 and evaluate",
        kind="success",
        full_width=True,
        keyboard_shortcut="Ctrl-Enter",
    )
    mo.vstack(
        [
            mo.md("## Run · 选择参数并显式执行"),
            mo.hstack([run_mode, top_n], widths="equal", align="end"),
            business_question,
            run_lesson,
            mo.md(
                "> Top-N 是唯一由学习者改变的数据结果参数；两个 worker、调用顺序、"
                "数据库动作和 10 项质量门都由代码契约固定。"
            ),
        ],
        gap=1,
    )
    return business_question, run_lesson, run_mode, top_n


@app.cell
def _(mo):
    get_level4_run, set_level4_run = mo.state(None)
    return get_level4_run, set_level4_run


@app.cell
def _(
    business_question,
    core_ready,
    lesson_settings,
    run_lesson,
    run_level,
    run_mode,
    set_level4_run,
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
            _prompt = f"{_question} Show the top {top_n.value} categories."
            _payload["result"] = run_level(
                4,
                _prompt,
                settings=lesson_settings,
                offline=run_mode.value == "offline",
            )
        except Exception as _run_exc:
            _payload["error"] = f"{type(_run_exc).__name__}: {_run_exc}"
        _payload["elapsed"] = time.perf_counter() - _started
        set_level4_run(_payload)
    return


@app.cell
def _(get_level4_run):
    level4_run = get_level4_run()
    return (level4_run,)


@app.cell(hide_code=True)
def _(
    business_question,
    evaluate_level_result,
    execution_observations,
    json,
    level4_run,
    mo,
    pl,
    run_mode,
    top_n,
):
    if level4_run is None:
        result_view = mo.callout(
            "选择 Top-N 与运行模式，然后点击 Run。结果会保留为快照；改变控件不会偷偷重跑。",
            kind="neutral",
            title="等待运行",
        )
    elif "error" in level4_run:
        _live_help = (
            "Live 请确认 `OMNIGENT_URL`、在线 Codex runner（或 `OMNIGENT_RUNNER_ID`）、"
            "Ollama 服务与 `ornith-1.5:9b`；也可切换至 Offline 先学习完整数据与评价路径。"
        )
        result_view = mo.callout(
            mo.md(f"**{level4_run['error']}**\n\n{_live_help}"),
            kind="danger",
            title="运行失败",
        )
    else:
        _result = level4_run["result"]
        _metadata = _result.metadata
        _result_frame = pl.DataFrame(_metadata.get("rows", []))
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
        _journal = execution_observations(4, result=_result)
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
            level4_run["inputs"]["question"] != business_question.value.strip()
            or level4_run["inputs"]["mode"] != run_mode.value
            or level4_run["inputs"]["top_n"] != top_n.value
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
        _execution_boundary = (
            "真实 OmniGenT server delegation"
            if _metadata.get("omnigent_connected")
            else "simulated OmniGenT/Codex delegation"
        )
        result_view = mo.vstack(
            [
                mo.md("## Observe · 结果、Trace 与质量门"),
                _freshness_panel,
                mo.hstack(
                    [
                        mo.stat(
                            f"{_passed_checks}/{len(_harness_checks)}",
                            label="Harness checks",
                            bordered=True,
                        ),
                        mo.stat(
                            f"{float(_harness_evaluation.get('score', 0)):.0%}",
                            label="Evaluation score",
                            bordered=True,
                        ),
                        mo.stat(
                            str(_result_frame.height),
                            label="Top-N rows",
                            bordered=True,
                        ),
                        mo.stat(
                            f"{level4_run['elapsed']:.2f}s",
                            label="Elapsed",
                            bordered=True,
                        ),
                    ],
                    widths="equal",
                ),
                mo.callout(
                    f"本次控制面边界：{_execution_boundary}；"
                    f"runner={_metadata.get('omnigent_runner_id') or 'none'}。",
                    kind="info",
                ),
                mo.ui.tabs(
                    {
                        "Top-N 结果": mo.vstack(
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
                        "10 项 Harness 评价": mo.vstack(
                            [
                                mo.callout(
                                    "只有 10 项全部通过，runtime 才会返回 LevelResult。",
                                    kind=(
                                        "success"
                                        if _harness_evaluation.get("passed")
                                        and len(_harness_checks) == 10
                                        else "danger"
                                    ),
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
                                    "mode": _metadata.get("mode"),
                                    "generated_agents": _metadata.get("generated_agents"),
                                    "agent_calls": _metadata.get("agent_calls"),
                                    "omnigent_tool_calls": _metadata.get("omnigent_tool_calls"),
                                    "skill_digest_sha256": _metadata.get("skill_digest_sha256"),
                                    "bundle_sha256": _metadata.get("bundle_sha256"),
                                    "table_rows": _metadata.get("table_rows"),
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
                    value="10 项 Harness 评价",
                ),
                mo.md(
                    "## Next · 两个验证练习\n\n"
                    "1. 将 Top-N 分别改为 1、5、16，确认 `top-n-shape` 与 `baseline-parity` 始终通过。\n"
                    "2. 阅读 `evaluate_level4_run` 标签，解释为什么 evaluator 是宿主质量门，而不是第三个子智能体。"
                ),
            ],
            gap=1.1,
        )
    result_view
    return


@app.cell(hide_code=True)
def _(mo):
    capability_level = 4
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
