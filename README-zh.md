# Agent Kung-fu Levels：Sakila 十层智能体工程

这不是十份仅改名称的聊天脚本，而是一套逐层增加“控制面”的可运行教学项目。所有层共享同一个只读数据内核：

- 数据库：`data/sakila.db`
- 数据读取：`polars.read_database_uri` + ConnectorX
- 本地模型：Ollama OpenAI-compatible API `http://localhost:11434/v1`
- 固定模型：`ornith-1.5:9b`
- SQL 边界：只允许单条 `SELECT` / `WITH`，禁止修改和 SQLite 控制语句

## 十层地图

| 层级 | 驱动力 | 主要实现 | 新增的控制面 |
|---:|---|---|---|
| 1 | 提示词智能体 | LiteLLM + LangGraph + Jinja2 | 代码固定流程，Prompt/工具调用决定 Top-N |
| 2 | 代码智能体 | `openai-agents` + function tools | 代码拥有 skills 与工具权限 |
| 3 | 协议智能体 | `agent.md` + Codex Python SDK + A2A/MCP/JSON-RPC | 动态 Codex workers 与协议消息边界 |
| 4 | Harness Agent | `omnigent-client` + Codex + Polars skill | 双子智能体计划、宿主动作白名单与失败即拒绝评估 |
| 5 | 迭代飞轮 | `omnigent-client` + Codex + Ralph loop | 表探索轨迹、反馈绑定与确定性停止 |
| 6 | 数据+关系逻辑 | Pydantic schema + NetworkX + DeepEval | 类型契约、关系路径、质量门 |
| 7 | 知识积累 | LLM wiki + graph engineer | 检索已验证查询知识，按数据库指纹失效 |
| 8 | 知识生产 | OWL + OBDA + annotations + wiki | 本体约束知识生成 |
| 9 | 问题驱动 | Causal DAG + ontology/wiki/graph | 从相关性转为识别问题 |
| 10 | 创新驱动 | First principles + causal experiments | 从不变量推导可证伪机制 |

详细的数据流和边界见 [docs/architecture.md](docs/architecture.md)。

逐层修订的设计、干预与验收标准见 [十层实验计划 / Experiment plan](docs/level-revision-plan.md)。
Marimo 每层新增“运行能力干预实验 / Run capability experiment”按钮；也可执行：

```powershell
kungfu-agent experiment 5
```

实验使用隔离临时目录。L2 离线运行真实 Agents SDK，仅替换模型回复；L3 验证本地 JSON-RPC
边界；L4 注入冲突由证据裁决；L5 验证错误反馈与有界重试；L6 的关系图实际决定连接与计算；
L7 验证知识复用和数据变更失效；L8 校验本体关系后发布知识。L9–L10 使用明确标记的合成
因果与资源分配模型，不把实验效应或候选改善表述成 Sakila 的真实世界结论。

Each level now includes a controlled intervention. Offline reproducibility and scripted transport
tests are separate from live model validation. L9–L10 use labeled synthetic experiments; their
effects and improvements do not establish real-world causality or innovation.

## 安装

Python 3.12+：

```powershell
python -m pip install -e ".[harness,eval,dev]"
```

若只学习某一层，也可以从仓库根目录安装该层的完整运行依赖；Level 1–10 各自目录下的
`requirements.txt` 同时包含公共可执行教材依赖和该层专属框架。例如：

```powershell
python -m pip install -r src/agent_kungfu/levels/level5/requirements.txt
```

确认 Ollama 与模型：

```powershell
ollama list
Invoke-RestMethod http://localhost:11434/v1/models
```

环境变量都有可用默认值，也可覆盖：

```powershell
$env:OLLAMA_OPENAI_BASE_URL = "http://localhost:11434/v1"
$env:AGENT_LLM_MODEL = "ornith-1.5:9b"
$env:SAKILA_DB_PATH = "D:\Python\agent_kungfu_levels\data\sakila.db"
```

Ollama 不校验 API key；代码只向兼容客户端传入占位值 `ollama`。

## 运行

### 浏览器演示台（推荐，无需 Python 编辑器）

在资源管理器中双击：

```text
start_marimo.cmd
```

启动后访问 `http://127.0.0.1:2718`。首页只保留初始化指导、环境检查和 Demo 总目录。
点击每一课的 **index** 进入独立页面；也可从目录直接跳到该课的架构图、代码说明或运行结果。
总入口和所有独立 Demo 顶部都有 **中文 / English** 切换。初始化指导、教学内容、架构图文字、
控件和检查说明同步切换；目录、上一课和下一课链接保留语言选择。也可直接访问
`/?lang=en` 或 `/level_01/index/?lang=en`，刷新后保留 URL 中的语言。
源码和原始运行证据保留原文；离线答案提供对应英文，实时模型输出保留模型实际返回的文本。

```text
demos/
├── index.py             # 初始化与总目录
├── baseline/index.py    # SQL 基准
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

每个 index 只加载本课，包含架构图、真实源码与分步说明、输入控件、运行结果、自动检查和能力实验。
点击 **Run lesson** 生成本课结果；修改控件会标记旧结果，重新运行后更新。
默认离线模式仍执行真实数据库、Polars、图与工件生成。各页面的运行状态独立，重新打开页面需重新运行。

统一启动逻辑只维护在 `serve_demos.py`，两种 shell 入口只负责定位目录、设置 Python 路径和传递参数。

| 环境 | 启动命令 |
|---|---|
| Windows Batch / PowerShell | `.\start_marimo.cmd` |
| Linux / macOS / Bash | `bash start_marimo.sh` |
| Python | `python serve_demos.py` |

都可追加 `--port 2720` 指定端口，`--help` 查看参数。Bash 默认用 `python3`，Batch 默认用 `python`；
可通过 `PYTHON` 环境变量指定解释器路径。按 Ctrl+C 停止服务。

从 PowerShell 启动整个目录：

```powershell
python serve_demos.py --port 2718
```

访问 `/` 为总目录，`/baseline/index/` 为 Baseline，`/level_01/index/` 至
`/level_10/index/` 为各课。`start_levels_marimo.cmd` 是旧名称的转发入口，也使用统一参数和默认端口。
多页服务使用 [marimo 原生 ASGI 路由](https://docs.marimo.io/guides/deploying/programmatically/)。

只运行一课（跨课导航需启动上面的多页服务）：

```powershell
python -m marimo run demos/level_01/index.py
```

`marimo_app.py` 复用为首页内容，`marimo_levels.py` 复用为单课教材组件；
各课 index 固定课程编号，不再用一个下拉框把全部 demo 放在一起。
原有平铺 CLI 入口及 L4/L5 专项教材保留兼容。

查看层级：

```powershell
kungfu-agent list
```

真实本地模型运行：

```powershell
kungfu-agent run 1
kungfu-agent run 2 -q "收入最高的电影有哪些？"
kungfu-agent run 3 -q "比较门店收入"
```

不调用模型/外部 server，但仍执行真实 LangGraph、Polars、ConnectorX、图谱、OWL 生成和 harness 的教学模式：

```powershell
kungfu-agent run 1 --offline
python demos/level_05_ralph_loop.py --offline
python demos/level_10_first_principles.py --offline
```

原有命令行与专项教材入口仍可使用：

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

`level_04_omnigent_codex_polars_harness.py` 是 Level 4 的主教学入口；旧的
`level_04_harness_conflict_omnigent.py` 仅保留为向后兼容 launcher，执行的是同一个新实现。
独立打开 Level 4 可执行教材：

```powershell
marimo run demos/level_04_omnigent_codex_polars_harness.py
```

`level_05_ralph_omnigent_codex_loop.py` 是 Level 5 的独立 Marimo 可执行教材；
`level_05_ralph_loop.py` 保留为命令行入口。独立打开 Level 5：

```powershell
marimo run demos/level_05_ralph_omnigent_codex_loop.py
```
