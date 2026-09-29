import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="Agent Kung-fu · Demo catalog")


@app.cell
def _():
    import json
    import urllib.request
    import marimo as mo
    from agent_kungfu.config import Settings
    from agent_kungfu.database import SakilaDB

    return Settings, SakilaDB, json, mo, urllib


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


@app.cell(hide_code=True)
def _(tr, mo):
    mo.md(
        tr(r"""
    # Agent Kung-fu：初始化与 Demo 目录

    首次使用先在项目根目录运行 `python -m pip install -e ".[harness,eval,dev]"`，
    确认 `data/sakila.db` 存在，再检查下面的环境状态。
    点击目录中各课的 **index**，进入该 demo 的独立页面：

    **理解控制权变化 → 阅读架构与真实源码 → 预测 → 执行 → 观察轨迹 → 自动验收 → 练习**

    一句话概括十层：L1 让 LLM 按流程工作，L2 让 Agent 执行代码，L3 让 Agent 通过协议协作，
    L4 让 Agent 处理冲突，L5 让 Agent 通过反馈迭代，L6 让 Agent 操作结构化协同关系，
    L7 让 Agent 积累知识，L8 让 Agent 按 Ontology 生产知识，L9 让 Agent 建模因果问题，
    L10 让 Agent 从第一性原理重新定义问题并产生创新。
    """)
    )
    return


@app.cell(hide_code=True)
def _(tr, get_teaching_material, mo, link):
    _rows = []
    for _number in range(11):
        _material = get_teaching_material(_number)
        _rows.append(
            tr(
                f"| {_material.title} | [index]({link(_number)}) | "
                f"[架构图]({link(_number, anchor='architecture')}) · "
                f"[代码说明]({link(_number, anchor='code')}) · "
                f"[运行结果]({link(_number, anchor='results')}) |"
            )
        )
    mo.md(
        tr(
            "## Demo 目录\n\n每个 demo 都有独立目录和 index，点击进入后只展示该课。"
            "运行结果在本课点击 **Run lesson** 后生成。\n\n"
            "| Demo | 入口 | 本课索引 |\n|---|---|---|\n"
        )
        + "\n".join(_rows)
    )
    return


@app.cell
def _(
    tr,
):
    level_catalog = [
        {
            "layer": "L1",
            "suggested_name": "Prompt Agent",
            "driver": tr("提示词 / Prompt"),
            "controls": tr("Token / 上下文"),
            "core_technology_examples": "LiteLLM + LangGraph + Jinja2",
        },
        {
            "layer": "L2",
            "suggested_name": "Code Agent",
            "driver": tr("技能 / 代码"),
            "controls": tr("工具 / 代码"),
            "core_technology_examples": "LiteLLM + OpenAI Agents",
        },
        {
            "layer": "L3",
            "suggested_name": "Protocol Agent",
            "driver": tr("协议"),
            "controls": tr("智能体通信"),
            "core_technology_examples": "Codex Python SDK + A2A/MCP/JSON-RPC",
        },
        {
            "layer": "L4",
            "suggested_name": "Harness Agent",
            "driver": tr("冲突"),
            "controls": tr("智能体行为"),
            "core_technology_examples": "Harness + OmniGenT",
        },
        {
            "layer": "L5",
            "suggested_name": "Loop Agent",
            "driver": tr("迭代"),
            "controls": tr("智能体轨迹"),
            "core_technology_examples": "Ralph Loop",
        },
        {
            "layer": "L6",
            "suggested_name": "Graph Engineering Agent",
            "driver": tr("结构"),
            "controls": tr("数据 / 协同"),
            "core_technology_examples": "Harness + Graph Engineering",
        },
        {
            "layer": "L7",
            "suggested_name": "Knowledge Driven Agent",
            "driver": tr("知识"),
            "controls": tr("知识积累"),
            "core_technology_examples": "LLM Wiki",
        },
        {
            "layer": "L8",
            "suggested_name": "Ontology Driven Agent",
            "driver": tr("语义"),
            "controls": tr("概念 / 类型 / 关系"),
            "core_technology_examples": "Ontology",
        },
        {
            "layer": "L9",
            "suggested_name": "Causal Driven Agent",
            "driver": tr("因果"),
            "controls": tr("问题 / 原因 / 结果"),
            "core_technology_examples": "Causal model",
        },
        {
            "layer": "L10",
            "suggested_name": "First-Principles Driven Agent",
            "driver": tr("创新"),
            "controls": tr("假设 / 定律 / 原理"),
            "core_technology_examples": "First Principles",
        },
    ]
    return (level_catalog,)


@app.cell(hide_code=True)
def _(tr, level_catalog, mo):
    mo.vstack(
        [
            mo.md(tr("## 十层路线图 · Execution → Control → Intelligence")),
            mo.callout(
                tr(
                    "L1–L5 是 Agent Execution & Control Layer；L6–L10 是 Agent Intelligence Layer。"
                ),
                kind="info",
            ),
            mo.ui.table(
                level_catalog,
                selection=None,
                pagination=False,
                show_download=False,
                show_search=False,
                label=tr("十层的主导驱动、控制对象与核心技术举例"),
            ),
        ]
    )
    return


@app.cell
def _(tr, SakilaDB, Settings, json, urllib):
    course_settings = Settings.from_env()
    course_db = None
    environment_rows = []
    database_counts = []
    try:
        course_settings.validate()
        course_db = SakilaDB(course_settings)
        database_counts = course_db.query(
            """
            SELECT 'film' AS entity, COUNT(*) AS rows FROM film
            UNION ALL SELECT 'customer', COUNT(*) FROM customer
            UNION ALL SELECT 'rental', COUNT(*) FROM rental
            UNION ALL SELECT 'payment', COUNT(*) FROM payment
            UNION ALL SELECT 'store', COUNT(*) FROM store
            """
        )
        environment_rows.append(
            {
                tr("检查"): "Sakila + ConnectorX",
                tr("状态"): "PASS",
                tr("实际"): str(course_settings.db_path.resolve()),
                tr("影响"): tr("Baseline/L1–L10 可运行"),
            }
        )
    except Exception as _db_exc:
        environment_rows.append(
            {
                tr("检查"): "Sakila + ConnectorX",
                tr("状态"): "FAIL",
                tr("实际"): f"{type(_db_exc).__name__}: {_db_exc}",
                tr("影响"): tr("先修复数据库/依赖"),
            }
        )
    try:
        with urllib.request.urlopen(
            f"{course_settings.ollama_base_url}/models", timeout=1.5
        ) as _response:
            _model_payload = json.loads(_response.read().decode("utf-8"))
        _model_ids = {
            str(item.get("id", item.get("name", "")))
            for item in _model_payload.get("data", _model_payload.get("models", []))
        }
        _model_found = any(
            course_settings.ollama_model == model_id or course_settings.ollama_model in model_id
            for model_id in _model_ids
        )
        environment_rows.append(
            {
                tr("检查"): "Ollama + configured model",
                tr("状态"): "PASS" if _model_found else "WARN",
                tr(
                    "实际"
                ): f"service online; model={course_settings.ollama_model}; installed={_model_found}",
                tr("影响"): tr("仅 live 模式"),
            }
        )
    except Exception as _ollama_exc:
        environment_rows.append(
            {
                tr("检查"): "Ollama + configured model",
                tr("状态"): "OPTIONAL",
                tr("实际"): f"{type(_ollama_exc).__name__}: offline mode remains available",
                tr("影响"): tr("live 模式不可用"),
            }
        )
    environment_rows.append(
        {
            tr("检查"): "Artifact output",
            tr("状态"): "WRITE",
            tr("实际"): str(course_settings.output_dir.resolve()),
            tr("影响"): tr("L7–L10 离线也写文件"),
        }
    )
    course_ready = course_db is not None
    return course_db, course_ready, course_settings, database_counts, environment_rows


@app.cell(hide_code=True)
def _(tr, course_ready, database_counts, environment_rows, mo):
    count_view = (
        mo.ui.table(
            database_counts,
            selection=None,
            pagination=False,
            show_download=False,
            label=tr("关键表规模（实时读取）"),
        )
        if course_ready
        else mo.callout(tr("数据库未就绪，无法读取表规模。"), kind="danger")
    )
    mo.vstack(
        [
            mo.md(tr("## Setup · 可执行前置检查")),
            mo.ui.table(environment_rows, selection=None, pagination=False, show_download=False),
            count_view,
            mo.md(
                tr(
                    "> Offline 是确定性教学路径，不等于 mock：数据库、Polars、图构建和文件生成均真实执行。"
                )
            ),
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
