import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="Agent Kung-fu · Demo")


@app.cell
def _():
    import json
    import time
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import yaml

    from agent_kungfu.levels.checks_en import check_record
    from agent_kungfu.config import Settings
    from agent_kungfu.database import SakilaDB
    from agent_kungfu.levels import run_level
    from agent_kungfu.levels.baseline import run_baseline
    from agent_kungfu.levels.common import LEVELS_DIR, load_baseline_assets, load_level_config
    from agent_kungfu.levels.teaching import (
        baseline_image_paths,
        evaluate_baseline,
        evaluate_level_result,
        execution_observations,
        load_source_documents,
        source_tree,
    )

    return (
        LEVELS_DIR,
        check_record,
        Path,
        SakilaDB,
        Settings,
        baseline_image_paths,
        evaluate_baseline,
        evaluate_level_result,
        execution_observations,
        json,
        load_baseline_assets,
        load_level_config,
        load_source_documents,
        mo,
        pl,
        run_baseline,
        run_level,
        source_tree,
        time,
        yaml,
    )


@app.cell
def _(mo):
    query_params = mo.query_params()
    return (query_params,)


@app.cell(hide_code=True)
def _(mo, query_params):
    language = mo.ui.radio(
        {"中文": "zh", "English": "en"},
        value="English" if query_params.get("lang") == "en" else "中文",
        inline=True,
        label="语言 / Language",
        on_change=lambda value: query_params.set("lang", value),
    )
    language
    return (language,)


@app.cell
def _(language):
    ui_language = language.value
    return (ui_language,)


@app.cell
def _(ui_language):
    from functools import partial
    from agent_kungfu.levels.localization import demo_url, material_for_language, translate

    tr = partial(translate, language=ui_language)
    get_teaching_material = partial(material_for_language, language=ui_language)
    link = partial(demo_url, language=ui_language)
    return get_teaching_material, link, tr


@app.cell
def _():
    lesson_number = 0
    return (lesson_number,)


@app.cell
def _(tr, SakilaDB, Settings, load_baseline_assets):
    lesson_settings = Settings.from_env()
    setup_rows = []
    lesson_db = None
    try:
        lesson_settings.validate()
        setup_rows.append(
            {
                tr("项目"): "Sakila database",
                tr("状态"): "PASS",
                tr("解析值"): str(lesson_settings.db_path.resolve()),
                tr("恢复建议"): "—",
            }
        )
        lesson_db = SakilaDB(lesson_settings)
        table_count = lesson_db.query(
            "SELECT COUNT(*) AS n FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).item(0, "n")
        setup_rows.append(
            {
                tr("项目"): "Polars + ConnectorX",
                tr("状态"): "PASS",
                tr("解析值"): f"read-only SELECT succeeded; {table_count} tables",
                tr("恢复建议"): "—",
            }
        )
        baseline_assets = load_baseline_assets(lesson_settings)
        setup_rows.append(
            {
                tr("项目"): "Baseline YAML/SQL",
                tr("状态"): "PASS",
                tr("解析值"): str(baseline_assets.sql_path),
                tr("恢复建议"): "—",
            }
        )
    except Exception as _setup_exc:
        baseline_assets = None
        setup_rows.append(
            {
                tr("项目"): tr("本地教学运行时"),
                tr("状态"): "FAIL",
                tr("解析值"): f"{type(_setup_exc).__name__}: {_setup_exc}",
                tr("恢复建议"): tr("确认 data/sakila.db、依赖与 YAML 路径。"),
            }
        )
    setup_rows.extend(
        [
            {
                tr("项目"): "Ollama OpenAI-compatible API",
                tr("状态"): "OPTIONAL",
                tr("解析值"): lesson_settings.ollama_base_url,
                tr("恢复建议"): tr("仅实时模式需要 ollama serve。"),
            },
            {
                tr("项目"): "Model",
                tr("状态"): "OPTIONAL",
                tr("解析值"): lesson_settings.ollama_model,
                tr("恢复建议"): tr("仅实时模式需要安装该模型。"),
            },
            {
                tr("项目"): "Generated artifacts",
                tr("状态"): "WRITE",
                tr("解析值"): str(lesson_settings.output_dir.resolve()),
                tr("恢复建议"): tr("L7–L10 离线运行也会写入这里。"),
            },
        ]
    )
    setup_ready = lesson_db is not None and baseline_assets is not None
    return baseline_assets, lesson_db, lesson_settings, setup_ready, setup_rows


@app.cell
def _(get_teaching_material, lesson_number, load_source_documents, source_tree):
    teaching_material = get_teaching_material(lesson_number)
    teaching_sources = load_source_documents(teaching_material)
    teaching_tree = source_tree(teaching_sources)
    return teaching_material, teaching_sources, teaching_tree


@app.cell(hide_code=True)
def _(tr, lesson_number, mo, teaching_material, link):
    _links = [tr(f"[← Demo 总目录]({link()})")]
    if lesson_number > 0:
        _links.append(tr(f"[上一课]({link(lesson_number - 1)})"))
    if lesson_number < 10:
        _links.append(tr(f"[下一课]({link(lesson_number + 1)})"))
    mo.vstack(
        [
            mo.md(" · ".join(_links)),
            mo.md(f"# {teaching_material.title}"),
            mo.md(
                tr(
                    "**本 Demo 索引** · [架构图](#demo-architecture) · [代码说明](#demo-code) · "
                    "[运行实验](#demo-run) · [运行结果](#demo-results) · [能力实验](#demo-capability)"
                )
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(tr, mo, setup_ready, setup_rows):
    setup_kind = "success" if setup_ready else "danger"
    _setup = mo.vstack(
        [
            mo.md(tr("## Setup · 先验证环境，再开始实验")),
            mo.callout(
                tr("核心本地依赖已就绪，可以运行。")
                if setup_ready
                else tr("核心环境未就绪；先按表中建议修复。"),
                kind=setup_kind,
            ),
            mo.ui.table(setup_rows, selection=None, pagination=False, show_download=False),
            mo.md(
                tr(
                    "> “只读”指所有 SQL 经过应用层单条 `SELECT/WITH` 守卫；SQLite 文件本身并非以 `mode=ro` 打开。"
                )
            ),
        ],
        gap=1,
    )
    mo.accordion({tr("环境检查（初始化说明见总目录）"): _setup})
    return


@app.cell(hide_code=True)
def _(
    tr,
    baseline_image_paths,
    lesson_number,
    mo,
    teaching_material,
    teaching_sources,
    teaching_tree,
):
    prerequisites = "\n".join(f"- {item}" for item in teaching_material.prerequisites)
    concept_checks = "\n".join(f"- {item}" for item in teaching_material.checkpoints)
    watch_for = "\n".join(f"- {item}" for item in teaching_material.observations)
    step_rows = [
        {tr("步骤"): step.title, tr("要做什么"): step.action, tr("预期证据"): step.expected}
        for step in teaching_material.steps
    ]
    code_tabs = {
        document.label: mo.vstack(
            [
                mo.md(f"`{document.reference}`"),
                mo.ui.code_editor(
                    value=document.content,
                    language=document.language,
                    disabled=True,
                    min_height=260,
                    max_height=560,
                    show_copy_button=True,
                    label=document.reference,
                ),
            ]
        )
        for document in teaching_sources
    }
    lesson_blocks = [
        mo.md(f"## Goal · {teaching_material.title}\n\n{teaching_material.goal}"),
        mo.callout(
            teaching_material.change_from_previous,
            kind="info",
            title=tr("相对上一层，控制权发生了什么变化？"),
        ),
        mo.md(f"### Prerequisites\n\n{prerequisites}"),
        mo.Html('<div id="demo-code"></div>'),
        mo.md(tr("## Build / Read · 按步骤理解实现")),
        mo.ui.table(step_rows, selection=None, pagination=False, show_download=False),
        mo.Html('<div id="demo-architecture"></div>'),
        mo.md(tr("### Architecture · 图是源码调用关系的阅读地图")),
        mo.mermaid(teaching_material.architecture, theme="neutral"),
        mo.md(f"### Source architecture\n\n```text\n{teaching_tree}\n```"),
        mo.ui.tabs(code_tabs, lazy=False),
        mo.md(tr(f"### Run 前要观察什么？\n\n{watch_for}")),
        mo.md(tr(f"### 概念检查清单\n\n{concept_checks}")),
    ]
    if lesson_number == 0:
        sakila_erd, sakila_structure = baseline_image_paths()
        lesson_blocks[3:3] = [
            mo.md(
                tr(
                    "## Sakila database · 先读懂事实源\n\n"
                    "收入链为 `category → film_category → inventory → rental → payment`。"
                    "下图只用于教学；运行时 schema 始终来自本地 SQLite。"
                )
            ),
            mo.ui.tabs(
                {
                    "Core ER diagram": mo.image(
                        sakila_erd,
                        alt="Sakila core entity-relationship diagram",
                        width="100%",
                    ),
                    "Business-domain structure": mo.image(
                        sakila_structure,
                        alt="Sakila schema grouped by business domain",
                        width="100%",
                    ),
                },
                lazy=False,
            ),
        ]
    mo.vstack(lesson_blocks, gap=1.2)
    return


@app.cell
def _(tr, mo, teaching_material):
    prediction_options = {tr("先不作答"): "__skip__"}
    if teaching_material.prediction is not None:
        prediction_options.update(dict(teaching_material.prediction.options))
    prediction = mo.ui.radio(
        options=prediction_options,
        value=tr("先不作答"),
        label=(
            teaching_material.prediction.question
            if teaching_material.prediction is not None
            else tr("运行前预测")
        ),
    )
    top_n = mo.ui.slider(1, 16, value=5, step=1, label="Top N（L1–L5）")
    offline = mo.ui.switch(value=True, label="Offline teaching mode")
    question = mo.ui.text_area(
        value=tr("找出收入最高的电影类别，并说明结果。"),
        label=tr("业务问题（Baseline 忽略；L1–L5 会附加 Top N）"),
        full_width=True,
    )
    execute = mo.ui.run_button(label="▶ Run lesson", kind="success", full_width=True)
    mo.vstack(
        [
            mo.Html('<div id="demo-run"></div>'),
            mo.md(tr("## Predict & Run · 先预测，再执行")),
            prediction,
            mo.hstack([top_n, offline], widths="equal", align="end"),
            question,
            execute,
            mo.callout(
                tr(
                    "离线模式仍执行真实数据库/Polars/图/文件生成。关闭离线后才调用 "
                    "Ollama；L3 会启动两个 Codex SDK threads；Live L4 还需要 OMNIGENT_URL 与在线 "
                    "Codex runner（可由 OMNIGENT_RUNNER_ID 固定），并通过 OmniGenT bundle 调用两个 "
                    "Codex 子智能体。Live L5 以 16 次逐表检查和 1 次收入计算组成 17 轮 Ralph loop，"
                    "每轮恰好调用一个 Codex agent tool。"
                ),
                kind="info",
            ),
        ],
        gap=1,
    )
    return execute, offline, prediction, question, top_n


@app.cell
def _(mo):
    get_lesson_run, set_lesson_run = mo.state(None)
    return get_lesson_run, set_lesson_run


@app.cell
def _(
    LEVELS_DIR,
    execute,
    lesson_db,
    lesson_settings,
    lesson_number,
    load_baseline_assets,
    load_level_config,
    offline,
    prediction,
    question,
    run_baseline,
    run_level,
    set_lesson_run,
    setup_ready,
    time,
    top_n,
):
    if execute.value:
        _started = time.perf_counter()
        _payload = {
            "level": lesson_number,
            "inputs": {
                "question": question.value.strip(),
                "top_n": top_n.value,
                "offline": offline.value,
                "prediction": prediction.value,
            },
        }
        try:
            if not setup_ready or lesson_db is None:
                raise RuntimeError("Setup preflight failed; fix the core environment first")
            if lesson_number == 0:
                _assets = load_baseline_assets(lesson_settings)
                _payload.update(
                    {
                        "frame": run_baseline(lesson_settings),
                        "result": None,
                        "sql": _assets.sql,
                        "config": _assets.config,
                        "config_path": LEVELS_DIR / "baseline" / "baseline.yaml",
                    }
                )
            else:
                _prompt = question.value.strip()
                if lesson_number in {1, 2, 3, 4, 5}:
                    _prompt = f"Show the top {top_n.value} categories. Question: {_prompt}"
                _result = run_level(
                    lesson_number,
                    _prompt,
                    settings=lesson_settings,
                    offline=offline.value,
                )
                _frame_data = _result.metadata.get("rows")
                if _frame_data is None and _result.sql:
                    _frame_data = lesson_db.query(_result.sql, limit=50)
                _payload.update(
                    {
                        "frame": _frame_data,
                        "result": _result,
                        "config": load_level_config(lesson_number),
                        "config_path": LEVELS_DIR / f"level{lesson_number}" / "config.yaml",
                    }
                )
        except Exception as _run_exc:
            _payload["error"] = f"{type(_run_exc).__name__}: {_run_exc}"
        _payload["elapsed"] = time.perf_counter() - _started
        set_lesson_run(_payload)
    return


@app.cell
def _(get_lesson_run):
    lesson_run = get_lesson_run()
    return (lesson_run,)


@app.cell(hide_code=True)
def _(
    tr,
    check_record,
    ui_language,
    Path,
    evaluate_baseline,
    evaluate_level_result,
    execution_observations,
    get_teaching_material,
    json,
    lesson_run,
    lesson_number,
    mo,
    offline,
    pl,
    question,
    top_n,
    yaml,
):
    if lesson_run is None:
        lesson_view = mo.callout(
            tr(
                "完成预测后点击 Run lesson。运行结果会保留；调整输入后会显示“结果已过期”，不会偷偷重跑。"
            ),
            kind="neutral",
            title=tr("等待实验"),
        )
    elif "error" in lesson_run:
        lesson_view = mo.callout(
            mo.md(
                tr(
                    f"**{lesson_run['error']}**\n\n"
                    "回到 Setup 检查数据库与依赖；实时连接失败时可切回 Offline teaching mode。"
                )
            ),
            kind="danger",
            title=tr("实验失败"),
        )
    else:
        run_level_number = lesson_run["level"]
        run_material = get_teaching_material(run_level_number)
        run_result = lesson_run["result"]
        run_frame_raw = lesson_run["frame"]
        run_frame = (
            run_frame_raw
            if isinstance(run_frame_raw, pl.DataFrame)
            else pl.DataFrame(run_frame_raw or [])
        )
        if run_level_number == 0:
            run_checks = evaluate_baseline(run_frame)
            run_observations = execution_observations(0, baseline=run_frame)
            run_answer = tr("Baseline 已建立：完整类别收入表是后续 L1–L10 的确定性参照。")
            run_trace = []
            run_artifacts = []
            run_sql = lesson_run["sql"]
        else:
            run_checks = evaluate_level_result(run_result)
            run_observations = execution_observations(run_level_number, result=run_result)
            run_answer = tr(run_result.answer) if lesson_run["inputs"]["offline"] else run_result.answer
            run_trace = [
                {
                    "protocol": str(item.protocol),
                    "sender": item.sender,
                    "recipient": item.recipient,
                    "kind": item.kind,
                    "correlation_id": item.correlation_id,
                    "payload": json.dumps(item.payload, ensure_ascii=False, default=str),
                }
                for item in run_result.trace
            ]
            run_artifacts = [Path(path) for path in run_result.artifacts]
            run_sql = run_result.sql or ""
        passed_count = sum(item.passed for item in run_checks)
        stale = (
            run_level_number != lesson_number
            or lesson_run["inputs"]["question"] != question.value.strip()
            or lesson_run["inputs"]["top_n"] != top_n.value
            or lesson_run["inputs"]["offline"] != offline.value
        )
        stale_panel = (
            mo.callout(
                tr("控件已改变；下面仍是上一次运行快照。再次点击 Run lesson 才会更新。"),
                kind="warn",
                title="Stale result",
            )
            if stale
            else mo.callout(tr("结果与当前控件一致。"), kind="success", title="Fresh result")
        )
        prediction_spec = run_material.prediction
        prediction_value = lesson_run["inputs"].get("prediction")
        if prediction_spec is None or prediction_value == "__skip__":
            prediction_panel = mo.callout(
                prediction_spec.explanation if prediction_spec else tr("本课没有预测题。"),
                kind="neutral",
                title=tr("预测题解析"),
            )
        else:
            prediction_correct = prediction_value == prediction_spec.answer
            prediction_panel = mo.callout(
                prediction_spec.explanation,
                kind="success" if prediction_correct else "warn",
                title=tr("预测正确") if prediction_correct else tr("预测与证据不同"),
            )
        artifact_rows = [
            {
                "name": path.name,
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "path": str(path),
            }
            for path in run_artifacts
        ]
        artifact_panel = (
            mo.ui.table(artifact_rows, selection=None, pagination=True, show_download=False)
            if artifact_rows
            else mo.callout(tr("本层没有生成文件；L7 起会出现持久知识工件。"), kind="neutral")
        )
        trace_panel = (
            mo.ui.table(run_trace, selection=None, pagination=True, wrapped_columns=["payload"])
            if run_trace
            else mo.callout(tr("本层没有协议消息；L3 起可观察 correlation_id。"), kind="neutral")
        )
        _level5_tabs = {}
        _level5_stop_panel = mo.md("")
        if run_level_number == 5:
            _level5_metadata = run_result.metadata
            _level5_iterations = _level5_metadata.get("iterations", [])
            _level5_observations = _level5_metadata.get("table_observations", {})
            _level5_revenue_chain = _level5_metadata.get("revenue_chain", [])
            _level5_exploration_rows = [
                {
                    tr("顺序"): index,
                    tr("表"): table,
                    tr("行数"): _level5_observations.get(table, {}).get("row_count"),
                    tr("列数"): len(_level5_observations.get(table, {}).get("columns", [])),
                    tr("收入链"): "YES" if table in _level5_revenue_chain else "—",
                    tr("收入线索"): ", ".join(
                        _level5_observations.get(table, {}).get("revenue_signals", [])
                    )
                    or "—",
                }
                for index, table in enumerate(_level5_metadata.get("inspected_tables", []), start=1)
            ]
            _level5_iteration_rows = [
                {
                    tr("轮次"): item.get("iteration"),
                    tr("动作"): item.get("action", {}).get("runtime_tool"),
                    tr("对象"): item.get("action", {}).get("table")
                    or f"Top-{item.get('action', {}).get('n')}",
                    tr("反馈摘要"): str(item.get("action", {}).get("feedback_digest_sha256", ""))[
                        :12
                    ],
                    tr("Skill摘要"): str(item.get("action", {}).get("skill_digest_sha256", ""))[
                        :12
                    ],
                    tr("得分"): item.get("evaluation", {}).get("score"),
                    tr("评价反馈"): "; ".join(item.get("evaluation", {}).get("feedback", [])),
                    tr("决策"): item.get("decision"),
                }
                for item in _level5_iterations
            ]
            _level5_call_rows = [
                {
                    tr("轮次"): item.get("iteration"),
                    "tool": item.get("name"),
                    "agent": item.get("agent_name"),
                    "executed_by": item.get("executed_by"),
                    "completed": item.get("completed"),
                    "output_sha256": str(item.get("output_sha256") or "")[:12],
                    "call_id": item.get("call_id"),
                }
                for item in _level5_metadata.get("omnigent_tool_calls", [])
            ]
            _level5_runtime_evaluation = _level5_metadata.get("evaluation", {})
            _level5_check_rows = [
                {
                    tr("状态"): "PASS" if item.get("passed") else "FAIL",
                    tr("检查"): item.get("name"),
                    tr("说明"): item.get("detail"),
                }
                for item in _level5_runtime_evaluation.get("checks", [])
            ]
            _level5_stopped = bool(
                _level5_metadata.get("terminated")
                and _level5_metadata.get("stop_reason") == "goal_satisfied"
                and _level5_runtime_evaluation.get("passed")
            )
            _level5_stop_panel = mo.callout(
                (
                    tr(
                        f"检查 {len(_level5_exploration_rows)}/16 张表后，在第 "
                        f"{_level5_metadata.get('iterations_used')} 轮通过 baseline parity 并立即退出；"
                        f"收入链={' → '.join(_level5_revenue_chain)}；"
                        f"真实 Codex calls={_level5_metadata.get('codex_subagent_calls')}。"
                    )
                    if _level5_stopped
                    else tr("Level 5 没有取得可交付的停止证明。")
                ),
                kind="success" if _level5_stopped else "danger",
                title=f"STOP · {_level5_metadata.get('stop_reason', 'unknown')}",
            )
            _level5_tabs = {
                tr("L5 · 全表探索"): mo.ui.table(
                    _level5_exploration_rows,
                    selection=None,
                    pagination=False,
                    show_download=True,
                ),
                tr("L5 · Ralph 轨迹"): mo.ui.table(
                    _level5_iteration_rows,
                    selection=None,
                    pagination=False,
                    show_download=True,
                    wrapped_columns=[tr("评价反馈")],
                ),
                tr("L5 · 停止评价"): mo.ui.table(
                    _level5_check_rows,
                    selection=None,
                    pagination=False,
                    show_download=False,
                    wrapped_columns=[tr("说明")],
                ),
                "L5 · OmniGenT/Codex": mo.ui.table(
                    _level5_call_rows,
                    selection=None,
                    pagination=True,
                    show_download=False,
                ),
            }
        exercises = "\n".join(
            f"{index}. {item}" for index, item in enumerate(run_material.exercises, 1)
        )
        next_level = min(run_level_number + 1, 10)
        reproduce = (
            '$env:PYTHONPATH="src"; python -c "from agent_kungfu.levels.baseline import '
            'run_baseline; print(run_baseline())"'
            if run_level_number == 0
            else tr(f'kungfu-agent run {run_level_number} --offline --question "你的问题"')
        )
        lesson_view = mo.vstack(
            [
                mo.md(f"## Observe & Check · {run_material.title}"),
                stale_panel,
                mo.hstack(
                    [
                        mo.stat(
                            f"{passed_count}/{len(run_checks)}",
                            label="Checks passed",
                            bordered=True,
                        ),
                        mo.stat(f"{lesson_run['elapsed']:.2f}s", label="Elapsed", bordered=True),
                        mo.stat(str(run_frame.height), label="Evidence rows", bordered=True),
                        mo.stat(
                            "offline" if lesson_run["inputs"]["offline"] else "live",
                            label="Mode",
                            bordered=True,
                        ),
                    ],
                    widths="equal",
                ),
                prediction_panel,
                _level5_stop_panel,
                mo.ui.tabs(
                    {
                        tr("自动检查"): mo.ui.table(
                            [check_record(item, ui_language, tr) for item in run_checks],
                            selection=None,
                            pagination=False,
                            wrapped_columns=[tr("期望"), tr("实际"), tr("失败后怎么做")],
                        ),
                        tr("逐步运行记录"): mo.ui.table(
                            tr(run_observations),
                            selection=None,
                            pagination=False,
                            wrapped_columns=[tr("运行证据"), tr("意义")],
                        ),
                        tr("答案与数据证据"): mo.vstack(
                            [mo.md(run_answer), mo.ui.table(run_frame, pagination=True)]
                        ),
                        "Executed SQL": mo.md(f"```sql\n{run_sql}\n```"),
                        "Protocol trace": trace_panel,
                        **_level5_tabs,
                        "Artifacts": artifact_panel,
                        "Advanced": mo.accordion(
                            {
                                f"YAML · {lesson_run['config_path']}": mo.md(
                                    "```yaml\n"
                                    + yaml.safe_dump(
                                        lesson_run["config"],
                                        allow_unicode=True,
                                        sort_keys=False,
                                    )
                                    + "\n```"
                                ),
                                "Raw metadata (original evidence)": mo.md(
                                    "```json\n"
                                    + json.dumps(
                                        {} if run_result is None else run_result.metadata,
                                        ensure_ascii=False,
                                        indent=2,
                                        default=str,
                                    )
                                    + "\n```"
                                ),
                            }
                        ),
                    },
                    value=tr("自动检查"),
                ),
                mo.md(f"## Practice & Next\n\n{exercises}"),
                mo.callout(
                    tr(f"完成练习后重新运行并比较检查表。下一课建议：Level {next_level}。"),
                    kind="info",
                ),
                mo.md(tr(f"### 离开 Marimo 也能复现\n\n```powershell\n{reproduce}\n```")),
            ],
            gap=1.2,
        )
    mo.vstack([mo.Html('<div id="demo-results"></div>'), mo.md(tr("## 运行结果")), lesson_view])
    return


@app.cell(hide_code=True)
def _(tr, mo, lesson_number):
    capability_level = lesson_number
    capability_run = mo.ui.run_button(
        label=tr("运行能力干预实验 / Run capability experiment"),
        disabled=capability_level == 0,
    )
    mo.vstack(
        [
            mo.Html('<div id="demo-capability"></div>'),
            mo.md(
                tr(
                    "## 能力实验 / Capability experiment\n\n在隔离临时目录执行正常、错误或阴性对照；L9–L10 的效应与创新结果来自合成模型。模型服务不参与本实验。"
                )
            ),
            capability_run,
        ]
    )
    return capability_level, capability_run


@app.cell(hide_code=True)
def _(mo, capability_level, capability_run, tr):
    mo.stop(not capability_run.value)
    from agent_kungfu.levels.experiments import run_experiment as _run_experiment

    with mo.status.spinner("Running controlled intervention…"):
        _report = _run_experiment(capability_level)
    mo.vstack(
        [
            mo.ui.table(tr(_report["checks"]), selection=None, pagination=False),
            mo.md("Evidence scope: " + _report["model_evidence"]),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
