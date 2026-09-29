from __future__ import annotations

import ast
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import polars as pl

from ..config import PROJECT_ROOT
from ..contracts import LevelResult
from ..database import validate_read_only_sql
from .common import LEVELS_DIR


PACKAGE_DIR = LEVELS_DIR.parent


@dataclass(frozen=True, slots=True)
class SourceSpec:
    label: str
    path: str
    language: str
    symbol: str | None = None


@dataclass(frozen=True, slots=True)
class SourceDocument:
    label: str
    reference: str
    language: str
    content: str


@dataclass(frozen=True, slots=True)
class LearningStep:
    title: str
    action: str
    expected: str


@dataclass(frozen=True, slots=True)
class PredictionSpec:
    question: str
    options: tuple[tuple[str, str], ...]
    answer: str
    explanation: str


@dataclass(frozen=True, slots=True)
class CheckResult:
    check_id: str
    title: str
    passed: bool
    expected: str
    actual: str
    remediation: str = ""

    def as_record(self) -> dict[str, str]:
        return {
            "状态": "PASS" if self.passed else "FAIL",
            "检查": self.title,
            "期望": self.expected,
            "实际": self.actual,
            "失败后怎么做": "—" if self.passed else self.remediation,
        }


@dataclass(frozen=True, slots=True)
class TeachingMaterial:
    level: int
    title: str
    goal: str
    architecture: str
    sources: tuple[SourceSpec, ...]
    checkpoints: tuple[str, ...]
    change_from_previous: str = ""
    prerequisites: tuple[str, ...] = ()
    steps: tuple[LearningStep, ...] = ()
    observations: tuple[str, ...] = ()
    exercises: tuple[str, ...] = ()
    prediction: PredictionSpec | None = None


_STYLE = """
classDef input fill:#D6EAF8,stroke:#2471A3,color:#12344D,stroke-width:2px;
classDef control fill:#FCF3CF,stroke:#B7950B,color:#533F03,stroke-width:2px;
classDef compute fill:#D5F5E3,stroke:#1E8449,color:#123F2A,stroke-width:2px;
classDef knowledge fill:#E8DAEF,stroke:#7D3C98,color:#3D1F4D,stroke-width:2px;
classDef output fill:#FADBD8,stroke:#B03A2E,color:#5B1E18,stroke-width:2px;
"""


MATERIALS: dict[int, TeachingMaterial] = {
    0: TeachingMaterial(
        level=0,
        title="Baseline · 先建立可信的数据参照",
        goal=(
            "理解 Sakila 的租赁业务关系，并用 YAML 定位数据库和 SQL；Baseline 是后续智能体"
            "结果的确定性对照，而不是另一个 Agent。"
        ),
        architecture="""
flowchart LR
    Y["baseline.yaml"] --> C["路径与连接配置"]
    C --> D[("sakila.db")]
    C --> S["revenue.sql"]
    D --> R["run_baseline"]
    S --> R
    R --> P["Polars + ConnectorX"]
    P --> O["16 类收入基准"]
    class Y,C input;
    class R control;
    class D,S,P compute;
    class O output;
""" + _STYLE,
        sources=(
            SourceSpec("导读 · sakila.md", "levels/baseline/sakila.md", "markdown"),
            SourceSpec("配置 · baseline.yaml", "levels/baseline/baseline.yaml", "yaml"),
            SourceSpec("查询 · revenue.sql", "levels/baseline/revenue.sql", "sql"),
            SourceSpec("执行器 · runner.py", "levels/baseline/runner.py", "python"),
            SourceSpec("数据入口 · SakilaDB", "database.py", "python", "SakilaDB"),
        ),
        checkpoints=(
            "本地 SQLite 是事实源；两张 ER 图用于理解关系，不替代实际 schema。",
            "收入链路是 category → film_category → inventory → rental → payment。",
            "Baseline 应返回 16 个类别，总收入 67406.56。",
        ),
    ),
    1: TeachingMaterial(
        level=1,
        title="Level 1 · Prompt Agent：让提示词指挥固定流程",
        goal=(
            "把数据计算固定为 read → merge → calculate 三步；LLM 只能通过"
            " top_n_categories(n) 决定展示多少行。"
        ),
        architecture="""
flowchart LR
    U["用户问题"] --> J["Jinja2 prompt"]
    J --> L["LiteLLM tool call"]
    subgraph G["LangGraph 三步数据流"]
      R["1. read_tables"] --> M["2. merge_tables"] --> C["3. calculate_revenue"]
    end
    C --> T["top_n_categories(n)"]
    L --> T
    T --> O["Top-N 证据"]
    class U,J input;
    class L,T control;
    class R,M,C compute;
    class O output;
""" + _STYLE,
        sources=(
            SourceSpec("配置 · config.yaml", "levels/level1/config.yaml", "yaml"),
            SourceSpec("提示词 · prompt.j2", "levels/level1/prompt.j2", "plaintext"),
            SourceSpec("完整实现 · workflow.py", "levels/level1/workflow.py", "python"),
        ),
        checkpoints=(
            "五张表分别读入 Polars，连接前投影必要列，避免 last_update 重名。",
            "模型不写 SQL、不计算收入，只产生带类型约束的 n 参数。",
            "inner join 产生 16044 条支付链路记录。",
        ),
    ),
    2: TeachingMaterial(
        level=2,
        title="Level 2 · Code Agent：用代码授予能力",
        goal=(
            "QA leader 保留回答权，把读取与计算分别封装为两个 specialist agent-tools；"
            "大 DataFrame 留在共享运行上下文中。"
        ),
        architecture="""
flowchart LR
    U["用户问题"] --> Q["QA Leader"]
    Q -->|"agent tool 1"| R["Table Read Agent"]
    R --> RT["load_revenue_tables"]
    RT --> W[("共享 Polars Workspace")]
    Q -->|"agent tool 2"| C["Calculate Agent"]
    W --> C
    C --> CT["calculate_category_revenue"]
    CT --> Q
    Q --> O["证据化回答"]
    class U input;
    class Q,R,C control;
    class RT,W,CT compute;
    class O output;
""" + _STYLE,
        sources=(
            SourceSpec("配置 · config.yaml", "levels/level2/config.yaml", "yaml"),
            SourceSpec("完整实现 · agents.py", "levels/level2/agents.py", "python"),
            SourceSpec("离线模型边界 · SDK 仍真实执行", "levels/level2/offline_model.py", "python"),
            SourceSpec(
                "共享状态 · RevenueAnalysisWorkspace",
                "levels/revenue_workspace.py",
                "python",
                "RevenueAnalysisWorkspace",
            ),
            SourceSpec("复用 · read_tables", "levels/level1/workflow.py", "python", "read_tables"),
            SourceSpec("复用 · merge_tables", "levels/level1/workflow.py", "python", "merge_tables"),
            SourceSpec(
                "复用 · calculate_revenue",
                "levels/level1/workflow.py",
                "python",
                "calculate_revenue",
            ),
        ),
        checkpoints=(
            "Agent.as_tool 适合 manager 模式；handoff 会转移最终回答权。",
            "is_enabled 与 parallel_tool_calls=False 强制 reader 先于 calculator。",
            "运行后校验 events == [read_tables, calculate_revenue]。",
        ),
    ),
    3: TeachingMaterial(
        level=3,
        title="Level 3 · Protocol Agent：从 agent.md 动态生成协作者",
        goal=(
            "把角色、依赖与工具名称写成可审查的 agent.md，再生成两个只读、临时的 Codex "
            "Python SDK worker；A2A 记录委派，Codex JSON-RPC 记录 turn，MCP 记录证据。"
        ),
        architecture="""
flowchart TD
    M["agent.md"] --> P["YAML frontmatter parser"]
    P --> V["Pydantic contract + allowlist"]
    V --> F["Codex Worker Factory"]
    F --> R["Reader thread<br/>read-only · deny-all · ephemeral"]
    F --> C["Calculator thread<br/>read-only · deny-all · ephemeral"]
    R <-->|"app-server JSON-RPC"| S["Codex Python SDK"]
    C <-->|"app-server JSON-RPC"| S
    R -->|"typed action"| A["Host allowlist"]
    C -->|"typed action"| A
    A --> W[("Polars workspace")]
    W --> O["A2A + JSON-RPC + MCP correlated trace"]
    class M,P input;
    class V,F,R,C,A control;
    class S knowledge;
    class W compute;
    class O output;
""" + _STYLE,
        sources=(
            SourceSpec("配置 · config.yaml", "levels/level3/config.yaml", "yaml"),
            SourceSpec("角色清单 · agent.md", "levels/level3/agent.md", "markdown"),
            SourceSpec("动态工厂 · dynamic_agents.py", "levels/level3/dynamic_agents.py", "python"),
            SourceSpec("可执行 JSON-RPC 边界", "levels/level3/rpc.py", "python"),
            SourceSpec(
                "Codex worker · CodexWorker",
                "levels/level3/dynamic_agents.py",
                "python",
                "CodexWorker",
            ),
            SourceSpec(
                "共享状态 · RevenueAnalysisWorkspace",
                "levels/revenue_workspace.py",
                "python",
                "RevenueAnalysisWorkspace",
            ),
            SourceSpec("协议记录 · ProtocolBus", "orchestration.py", "python", "ProtocolBus"),
        ),
        checkpoints=(
            "agent.md 是数据，不执行其中的 import、eval 或任意函数名。",
            "Codex thread 只有 read-only sandbox 与 deny_all approval；它只提出结构化动作。",
            "host runtime_tool allowlist 只允许 read_tables 与 calculate_revenue。",
            "同一请求的 A2A、Codex JSON-RPC 与 MCP message 共用 correlation_id。",
        ),
    ),
    4: TeachingMaterial(
        level=4,
        title="Level 4 · Harness Agent：让能力边界可裁决",
        goal="用 OmniGenT agent bundle 生成两个 Codex 子智能体，显式加载 Polars skill；配置声明最小权限，宿主只执行白名单动作，并在交付前对契约、顺序、数据与 baseline 统一评估。",
        architecture="""
flowchart TD
    U["问题 + Top-N"] --> O["OmniGenT client<br/>解析并绑定 Codex runner"]
    P["Polars SKILL.md + references"] --> B["Bundle loader<br/>校验路径 + SHA-256"]
    B --> O
    O --> S["sessions_chat<br/>上传 agent bundle"]
    S --> R["Codex Reader<br/>read_tables proposal"]
    S --> C["Codex Calculator<br/>calculate_revenue proposal"]
    R --> A["Host action allowlist"]
    C --> A
    A --> D[("只读 Sakila")]
    D --> L["Lazy Polars joins + group_by<br/>one collect"]
    L --> E{"10 checks<br/>委派 · 权限 · 顺序 · 基数 · baseline"}
    E -->|"fail closed"| X["拒绝交付"]
    E -->|"pass"| Z["Top-N + trace + evidence"]
    class U input;
    class O,S,B,A,E control;
    class R,C,D,L compute;
    class P knowledge;
    class X,Z output;
""" + _STYLE,
        sources=(
            SourceSpec("配置 · config.yaml", "levels/level4/config.yaml", "yaml"),
            SourceSpec("OmniGenT agent bundle", "levels/level4/omnigent-agent/config.yaml", "yaml"),
            SourceSpec("Polars skill", "levels/level4/polars/SKILL.md", "markdown"),
            SourceSpec("Level 4 runtime", "levels/level4/omnigent_harness.py", "python"),
            SourceSpec("Skill loader", "levels/level4/omnigent_harness.py", "python", "load_polars_skill"),
            SourceSpec("Harness evaluator", "levels/level4/omnigent_harness.py", "python", "evaluate_level4_run"),
        ),
        checkpoints=(
            "OmniGenT agent YAML 只能声明 read_tables 与 calculate_revenue 两个 Codex agent tools。",
            "level4/polars 不在 Codex 自动发现路径；loader 校验并打包到 skills/polars 后才真正生效。",
            "三个配置都声明 sandbox=auto、无 write_paths、禁用网络并从 scratch 启动；OmniGenT 0.11 Codex executor 使用 approvalPolicy=never。",
            "Windows Job Object 不强制 FS/network；跨平台硬边界仍是模型只提议、宿主白名单执行。",
            "模型只提出 Pydantic action；数据库读取与 Polars 计算只由 host allowlist 执行。",
            "评估失败会阻止交付；live 必须有 OMNIGENT_URL 与在线/显式 Codex runner，offline 不伪装成已连接。",
        ),
    ),
    5: TeachingMaterial(
        level=5,
        title="Level 5 · Loop Agent：让失败成为下一轮输入",
        goal=(
            "用 Ralph 状态、前轮评价摘要与 17 轮硬预算，依次检查 Sakila 全部 16 张表；"
            "随后才允许计算类别收入，并在 Polars 结果与 SQL baseline 完全一致时立即退出。"
        ),
        architecture="""
flowchart TD
    Q["问题 + Top-N"] --> C["SQLite catalog snapshot<br/>16 tables"]
    K["Polars skill + agent YAML"] --> B["Reproducible OmniGenT bundle"]
    C --> R["Ralph state<br/>visited · feedback digest · skill digest · budget"]
    B --> R
    R --> N{"仍有未检查表?"}
    N -->|"是"| O["OmniGenT session"]
    O --> X["Codex Table Explorer<br/>typed inspect_table action"]
    X --> H["Host allowlist<br/>schema · row count · signals"]
    H --> E["Coverage evaluator"]
    E -->|"feedback SHA-256 bound"| R
    N -->|"否"| J["Validate discovered chain<br/>category → film_category → inventory → rental → payment"]
    J --> S["Codex Revenue Solver<br/>typed calculate_revenue action"]
    S --> P["Host Polars<br/>5-table join + group_by"]
    P --> G{"10 checks + baseline parity?"}
    G -->|"fail / budget exhausted"| F["Reject without delivery"]
    G -->|"pass"| Z["STOP goal_satisfied<br/>ranking + trace + evidence"]
    class Q,C input;
    class B,R,N,O,H,E,G control;
    class X,J,S,P compute;
    class K knowledge;
    class F,Z output;
""" + _STYLE,
        sources=(
            SourceSpec("配置 · config.yaml", "levels/level5/config.yaml", "yaml"),
            SourceSpec("OmniGenT agent bundle", "levels/level5/omnigent-agent/config.yaml", "yaml"),
            SourceSpec("Codex Explorer", "levels/level5/omnigent-agent/agents/inspect_table/config.yaml", "yaml"),
            SourceSpec("Codex Revenue Solver", "levels/level5/omnigent-agent/agents/calculate_revenue/config.yaml", "yaml"),
            SourceSpec("Polars skill", "levels/level4/polars/SKILL.md", "markdown"),
            SourceSpec("完整 runtime", "levels/level5/ralph_omnigent.py", "python"),
            SourceSpec("探索状态", "levels/level5/ralph_omnigent.py", "python", "SakilaExplorationWorkspace"),
            SourceSpec("类型化动作", "levels/level5/ralph_omnigent.py", "python", "RalphAction"),
            SourceSpec("停止评价器", "levels/level5/ralph_omnigent.py", "python", "evaluate_level5_run"),
        ),
        checkpoints=(
            "成功轨迹必须是 16 个 canonical inspect_table，再接一个 calculate_revenue。",
            "每个 action 必须回显前一轮 feedback 与已加载 Polars skill 的 SHA-256，证明同时消费状态与能力合同。",
            "OmniGenT/Codex 只提出 Pydantic action；SQLite 和 Polars 仍由宿主白名单执行。",
            "Live 只接受 server-executed tool-call evidence；Offline 必须明确标记 simulated。",
            "第 17 轮只有在完整覆盖、数据基数、排序与 baseline parity 全部通过时才能停止。",
            "预算耗尽或任何 invariant 失败，都必须拒绝交付而非返回部分答案。",
        ),
    ),
    6: TeachingMaterial(
        level=6,
        title="Level 6 · Graph Engineering Agent：操作结构化协同关系",
        goal="用 typed harness schema 约束运行，用 schema graph 找到跨表协同路径，再通过 evaluator 质量门。",
        architecture="""
flowchart LR
    S["SQLite schema"] --> M["Typed Harness Manifest"]
    S --> G["Schema Graph"]
    M --> L
    G --> J["FK-derived JoinPlan"]
    J --> L["Read → Join → Aggregate DAG"]
    L --> E["Baseline parity / optional DeepEval"]
    E --> O["Graph-controlled Polars result"]
    class S input;
    class M,L,E control;
    class G,J compute;
    class O output;
""" + _STYLE,
        sources=(
            SourceSpec("配置 · config.yaml", "levels/level6/config.yaml", "yaml"),
            SourceSpec("目录入口 · __init__.py", "levels/level6/__init__.py", "python"),
            SourceSpec("Level 6 runtime", "levels/level6/graph_agent.py", "python"),
            SourceSpec("Harness 类型契约", "contracts.py", "python", "HarnessManifest"),
            SourceSpec("Schema graph", "graph_engineering.py", "python", "build_schema_graph"),
        ),
        checkpoints=(
            "Pydantic manifest 把权限、行数与迭代预算从 prompt 迁入代码契约。",
            "图的最短路径是结构候选，不自动等于业务正确路径。",
            "Level 6 控制数据与协同关系，不等于已经具有本体语义。",
        ),
    ),
    7: TeachingMaterial(
        level=7,
        title="Level 7 · Knowledge Driven Agent：让系统记得昨天",
        goal="把 schema、关系和运行经验写入可读、可 diff、可再次注入的 LLM Wiki。",
        architecture="""
flowchart LR
    B["业务任务"] --> L["Graph / Loop Agents"]
    L --> E["运行经验"]
    E --> W[("LLM-readable Wiki")]
    W -->|"共享记忆注入"| L
    L --> V["Evaluate"]
    V --> O["结果 + durable knowledge"]
    class B input;
    class L,V control;
    class E compute;
    class W knowledge;
    class O output;
""" + _STYLE,
        sources=(
            SourceSpec("配置 · config.yaml", "levels/level7/config.yaml", "yaml"),
            SourceSpec("目录入口 · __init__.py", "levels/level7/__init__.py", "python"),
            SourceSpec("Level 7 runtime", "levels/level7/knowledge_agent.py", "python"),
            SourceSpec("Wiki 生成器", "graph_engineering.py", "python", "generate_wiki"),
        ),
        checkpoints=(
            "Wiki 是跨运行的持久工件，不是单次 prompt 中更长的 schema。",
            "持久知识需要 freshness、provenance 与变更审查。",
            "查询证据仍优先于 Wiki 中的文字陈述。",
        ),
    ),
    8: TeachingMaterial(
        level=8,
        title="Level 8 · Ontology Driven Agent：按语义生产知识",
        goal="用 classes、relations 与 constraints 组织知识生产，再生成 OWL、OBDA、annotation 与 Wiki。",
        architecture="""
flowchart TD
    H["Human semantic design"] --> O[("Ontology")]
    O --> A["OWL / OBDA / annotations"]
    O --> W[("LLM Wiki")]
    A --> G["Graph Engineering Loops"]
    W --> G
    G --> E["Evaluate"]
    E --> B["Business function"]
    class H input;
    class O,W knowledge;
    class A,G compute;
    class E control;
    class B output;
""" + _STYLE,
        sources=(
            SourceSpec("配置 · config.yaml", "levels/level8/config.yaml", "yaml"),
            SourceSpec("目录入口 · __init__.py", "levels/level8/__init__.py", "python"),
            SourceSpec("Level 8 runtime", "levels/level8/ontology_agent.py", "python"),
            SourceSpec("Ontology 生成器", "graph_engineering.py", "python", "generate_sakila_ontology"),
        ),
        checkpoints=(
            "Schema 描述存储结构；Ontology 还表达业务概念、类型、关系与约束。",
            "结构合法的 OWL 不自动成为正确知识。",
            "canonical ontology 与 LLM candidate proposal 必须分层。",
        ),
    ),
    9: TeachingMaterial(
        level=9,
        title="Level 9 · Causal Driven Agent：先识别问题，再寻找答案",
        goal="从因果问题出发，显式组织假设、原因、结果、混杂因素与识别缺口。",
        architecture="""
flowchart LR
    P["Problem + data"] --> D["Causal DAG"]
    D --> O[("Ontology candidate")]
    O --> G["Graph / Loop solving"]
    G --> E["Evaluate identification"]
    E --> C{"问题已识别?"}
    C -->|"否"| O
    C -->|"是"| W[("Causal Wiki")]
    W --> R["关联、假设与实验建议"]
    class P input;
    class D,G compute;
    class O,W knowledge;
    class E,C control;
    class R output;
""" + _STYLE,
        sources=(
            SourceSpec("配置 · config.yaml", "levels/level9/config.yaml", "yaml"),
            SourceSpec("目录入口 · __init__.py", "levels/level9/__init__.py", "python"),
            SourceSpec("Level 9 runtime", "levels/level9/causal_agent.py", "python"),
        ),
        checkpoints=(
            "观察性门店差异不是库存干预的因果效应。",
            "DAG 的作用是暴露假设与识别缺口，而不是装饰报告。",
            "无法识别时，输出应是缺失数据或实验设计，而非伪确定答案。",
        ),
    ),
    10: TeachingMaterial(
        level=10,
        title="Level 10 · First-Principles Driven Agent：重新定义问题并产生创新",
        goal="从目标、不变量和约束推导机制假设，再用可证伪实验连接因果、本体、Wiki 与图工程。",
        architecture="""
flowchart TD
    P["原始问题"] --> R["1. 重新定义目标"]
    R --> F["2. 拆到基本事实"]
    F --> A["3. 质疑现有假设"]
    A --> M["4. 推导最小机制"]
    M --> H["5. 形成可证伪假设"]
    H --> X["6. 实验与护栏"]
    X --> C["Causal model"]
    C --> O[("Ontology + Wiki + Graph")]
    O -->|"新证据反馈"| R
    X --> N["创新机制 / 新知识"]
    class P input;
    class R,F,A,M,H,X,C control;
    class O knowledge;
    class N output;
""" + _STYLE,
        sources=(
            SourceSpec("配置 · config.yaml", "levels/level10/config.yaml", "yaml"),
            SourceSpec("目录入口 · __init__.py", "levels/level10/__init__.py", "python"),
            SourceSpec("Level 10 runtime", "levels/level10/principles_agent.py", "python"),
            SourceSpec("因果实验 · L9", "levels/level9/causal_agent.py", "python"),
        ),
        checkpoints=(
            "第一性原理不是自由联想：每个机制必须能说明如何失败。",
            "创新候选要连接 estimand、实验、护栏与停止条件。",
            "Level 10 重新定义目标，但不会扩大数据库权限。",
        ),
    ),
}


_LESSON_DETAILS: dict[int, dict[str, Any]] = {
    0: {
        "change_from_previous": "先不引入智能体：建立所有层都必须回归的确定性数据基准。",
        "prerequisites": (
            "data/sakila.db 可读，且 baseline.yaml 能解析数据库与 SQL 路径。",
            "Polars 与 ConnectorX 已安装；所有查询经过单条 SELECT/WITH 只读守卫。",
        ),
        "steps": (
            LearningStep("1 · 定位", "读取 baseline.yaml，解析 SQLite 与 revenue.sql。", "两个路径存在。"),
            LearningStep("2 · 理解", "沿五张表的外键链阅读 ER 图和 SQL。", "能解释收入来自 payment.amount。"),
            LearningStep("3 · 执行", "用 Polars + ConnectorX 执行 revenue.sql。", "得到 16 个降序类别。"),
            LearningStep("4 · 对账", "核对榜首与总收入。", "Sports=5314.21，总计=67406.56。"),
        ),
        "observations": (
            "YAML 是路径契约，SQL 是计算契约，图片只帮助理解而不参与执行。",
            "类别榜单必须按 revenue 降序且 category 唯一。",
            "后续 Agent 的 Top-N 应是这张完整基准表的前 N 行。",
        ),
        "exercises": (
            "把 Top N 改为 3，先预测前三名，再运行 L1 与 Baseline 对照。",
            "在不修改 SQL 的前提下，说明 payment 为何必须通过 rental 才能连接 inventory。",
        ),
        "prediction": PredictionSpec(
            "哪一个电影类别的收入最高？",
            (("Sports", "sports"), ("Action", "action"), ("Drama", "drama")),
            "sports",
            "确定性基准的第一行应为 Sports，收入 5314.21。",
        ),
    },
    1: {
        "change_from_previous": "从一条确定性 SQL 升级为固定的 read → merge → calculate 工作流；LLM 只选择 N。",
        "prerequisites": ("先完成 Baseline 并理解五表收入链。", "理解 DataFrame join 与 group_by。"),
        "steps": (
            LearningStep("1 · Read", "分别读取 category、film_category、inventory、rental、payment。", "保留五张 Polars DataFrame 与行数。"),
            LearningStep("2 · Merge", "按四组外键依次 inner join。", "支付链产生 16044 行。"),
            LearningStep("3 · Calculate", "按 category.name 聚合 amount 并降序。", "产生 16 行收入表。"),
            LearningStep("4 · Tool call", "提示词只请求 top_n_categories(n)。", "返回 Baseline 前 N 行。"),
        ),
        "observations": ("图节点顺序固定在代码中。", "LLM 不生成 SQL，也不做金额运算。", "n 是带上下界的函数参数。"),
        "exercises": ("分别以 N=1、5、16 重跑，检查答案始终等于 Baseline.head(N)。", "解释为什么 N=0 与 N=17 应在工具边界被拒绝。"),
        "prediction": PredictionSpec(
            "合并后的支付链应该有多少行？",
            (("16044", "16044"), ("16049", "16049"), ("4581", "4581")),
            "16044",
            "payment 有 16049 行，但只有 16044 条记录进入完整租赁—库存—类别连接链。",
        ),
    },
    2: {
        "change_from_previous": "把数据能力从提示词节点搬进可授权的 reader/calculator agent-tools。",
        "prerequisites": ("理解 L1 的三步数据契约。", "理解 manager-as-tools 与 handoff 的回答权差异。"),
        "steps": (
            LearningStep("1 · 组队", "构建 QA leader、Table Read Agent、Calculate Agent。", "形成一主两专的角色边界。"),
            LearningStep("2 · 读取", "leader 先调用 reader，把大 DataFrame 留在共享 workspace。", "events 首项为 read_tables。"),
            LearningStep("3 · 计算", "状态满足后才启用 calculator。", "events 次项为 calculate_revenue。"),
            LearningStep("4 · 回答", "leader 从结构化结果生成最终回答。", "结果与 Baseline Top-N 一致。"),
        ),
        "observations": ("Agent 工具传小控制消息，不传大 DataFrame。", "工具可用性体现代码授权。", "leader 保留最终回答权。"),
        "exercises": ("画出 calculator 先于 reader 时应触发的状态错误。", "比较 manager-as-tools 与 handoff 谁负责最终回答。"),
        "prediction": PredictionSpec(
            "两个 specialist 的合法调用顺序是什么？",
            (("read → calculate", "read-calc"), ("calculate → read", "calc-read"), ("并行", "parallel")),
            "read-calc",
            "calculator 依赖 reader 写入共享 workspace，因此必须串行。",
        ),
    },
    3: {
        "change_from_previous": "角色不再硬编码：agent.md 声明两个 Codex SDK worker，协议 trace 跨越真实 app-server 边界。",
        "prerequisites": ("理解 L2 的共享 workspace 与工具顺序。", "理解 YAML frontmatter、Pydantic 与 allowlist。"),
        "steps": (
            LearningStep("1 · Parse", "读取 agent.md frontmatter。", "得到 leader 与两个 subagent 定义。"),
            LearningStep("2 · Validate", "用类型契约和 runtime_tool allowlist 校验。", "任意函数名不能执行。"),
            LearningStep("3 · Generate", "工厂按依赖生成两个 Codex thread specs。", "live 时启动两个只读临时 threads。"),
            LearningStep("4 · Dispatch", "Codex 返回 typed action，host allowlist 才执行 Polars 方法。", "模型没有任意 Python/shell 能力。"),
            LearningStep("5 · Trace", "用 A2A/JSON-RPC/MCP envelope 记录委派、turn 与结果。", "所有子事件共享 correlation_id。"),
        ),
        "observations": ("Level 3 不导入或运行 openai-agents。", "离线 Codex JSON-RPC 是显式 simulated envelope；live 才启动 app-server。", "Codex 只建议 allowlisted action，Polars 执行仍归 host 控制。"),
        "exercises": ("在副本中把 runtime_tool 改成未知值，预测校验失败位置。", "从 trace 找出同一请求的六个相关子事件。"),
        "prediction": PredictionSpec(
            "动态工厂应生成几个 specialist subagents？",
            (("2", "2"), ("3", "3"), ("5", "5")),
            "2",
            "agent.md 声明一个 reader 与一个 calculator；leader 不是 specialist。",
        ),
    },
    4: {
        "change_from_previous": "协议协作之上增加可执行的 OmniGenT/Codex harness：skill、权限、动作与质量证据进入同一个控制面。",
        "prerequisites": ("理解 L3 的 action allowlist 与协议 trace。", "理解 agent bundle、skill digest 与 fail-closed evaluation。"),
        "steps": (
            LearningStep("1 · Load", "校验 Polars SKILL.md、五个 references 与 SHA-256。", "skill provenance 可复查。"),
            LearningStep("2 · Bundle", "把 agent config 与 skill 组装成 OmniGenT tar bundle。", "Codex harness 能从 skills/polars 发现能力说明。"),
            LearningStep("3 · Generate", "OmniGenT 依次调用 Reader 与 Calculator 两个 Codex agent tools。", "模型只返回严格 JSON action。"),
            LearningStep("4 · Execute", "host allowlist 读取五表并执行 one-collect lazy Polars plan。", "大表不进入 agent 消息。"),
            LearningStep("5 · Evaluate", "检查 os_env/approval 契约、顺序、基数、排序和 SQL baseline parity。", "任一 invariant 失败即拒绝交付。"),
        ),
        "observations": ("omnigent-client 是实际 bundle/session 边界，不再只是 audit observer。", "配置声明 sandbox=auto、无写路径、禁网、scratch 启动，Codex executor 使用 approvalPolicy=never。", "Windows Job Object 仅隔离进程树/资源；真正跨平台的硬边界是类型化 proposal 与宿主 allowlist。", "宿主 evaluator 不是第三个 agent，而是确定性质量门。"),
        "exercises": ("篡改一个 reference 后预测 skill digest 和 evaluator 的变化。", "把 calculator 放到 reader 前面，定位 manifest 或 state gate 的失败位置。"),
        "prediction": PredictionSpec(
            "Level 4 生成几个 Codex specialist agents？",
            (("2", "2"), ("3（含 evaluator）", "3"), ("按表生成 5 个", "5")),
            "2",
            "只有 Reader 与 Calculator 是子智能体；evaluation 是 host harness 的确定性阶段。",
        ),
    },
    5: {
        "change_from_previous": (
            "从 L4 的一次性 Reader→Calculator 控制面，升级为可持续推进状态、消费评价反馈并"
            "证明停止原因的 OmniGenT/Codex Ralph loop。"
        ),
        "prerequisites": (
            "理解 L4 的 bundle、Codex agent-tool、typed action 与 host allowlist。",
            "理解目录快照、循环不变量、迭代预算和 fail-closed 停止语义。",
        ),
        "steps": (
            LearningStep("1 · Snapshot", "读取 SQLite catalog 并固定 canonical 顺序。", "得到 16 个唯一表名。"),
            LearningStep("2 · Bundle", "把两个 Codex tools 与 Polars skill 打包给 OmniGenT。", "Explorer 与 Solver 权限一致且可追溯。"),
            LearningStep("3 · Inspect", "每轮只检查下一个未见表，并评价覆盖率。", "连续产生 16 份表级证据与反馈摘要。"),
            LearningStep("4 · Calculate", "全表覆盖后才允许执行五表 Polars 收入计算。", "join=16044，category=16。"),
            LearningStep("5 · Verify & Stop", "执行十项 harness 检查并与 revenue.sql 对账。", "第 17 轮以 goal_satisfied 立即退出。"),
        ),
        "observations": (
            "前 16 轮 evaluation.passed=false 是未完成状态，不是系统错误；反馈指定剩余覆盖工作。",
            "feedback_digest_sha256 把下一轮 action 与上一轮评价绑定，避免伪造‘已消费反馈’。",
            "Offline 的 OmniGenT/Codex 调用标记 simulated，但 16 次数据库检查、Polars 与 baseline 评价真实执行。",
            "成功不是‘循环次数到了’，而是 terminated=true、stop_reason=goal_satisfied 且十项评价全过。",
        ),
        "exercises": (
            "把 Top-N 分别改为 1 与 16，验证目录覆盖与 17 轮轨迹不变。",
            "阅读 bounded-immediate-stop 检查，说明为什么第 16 轮不能提前退出。",
        ),
        "prediction": PredictionSpec(
            "16 张表逐一检查、再计算一次收入，成功应在第几轮停止？",
            (("16", "16"), ("17", "17"), ("18", "18")),
            "17",
            "前 16 轮各检查一张表；第 17 轮执行收入计算，并在 baseline parity 通过后立即停止。",
        ),
    },
    6: {
        "change_from_previous": "进入 Agent Intelligence Layer：控制对象从执行轨迹迁移到类型化数据与协同关系。",
        "prerequisites": ("理解 HarnessManifest 的权限与预算字段。", "理解关系图中的节点、边与最短路径。"),
        "steps": (
            LearningStep("1 · Contract", "实例化 typed HarnessManifest。", "read_only、行数和迭代预算可验证。"),
            LearningStep("2 · Graph", "从实时 SQLite schema 构建表关系图。", "得到 16 表、21 关系。"),
            LearningStep("3 · Plan", "沿外键生成 payment 到 category 的 JoinPlan。", "连接条件由图边决定。"),
            LearningStep("4 · Execute", "按任务图执行读取、连接、聚合并对照 SQL 基线。", "删去关键关系会阻止计算。"),
        ),
        "observations": ("类型契约把关键控制从 prompt 迁入代码。", "最短路径是结构候选，不自动等于业务语义。", "图摘要可以成为后续知识生产的骨架。"),
        "exercises": ("选择另一对表，手工预测最短 join path。", "尝试解释 max_rows=0 为什么应在模型验证时失败。"),
        "prediction": PredictionSpec(
            "payment 到 film 的最短关系路径是什么？",
            (("payment→rental→inventory→film", "correct"), ("payment→customer→film", "customer"), ("payment→staff→film", "staff")),
            "correct",
            "payment 先关联 rental，再由 inventory 找到 film。",
        ),
    },
    7: {
        "change_from_previous": "把一次运行中的 schema 与经验沉淀为可读、可 diff、可再次注入的持久知识。",
        "prerequisites": ("理解 L6 schema graph。", "理解持久知识的 provenance、freshness 与审查。"),
        "steps": (
            LearningStep("1 · Inspect", "读取实时 schema 与表规模。", "知识来源可追溯到数据库。"),
            LearningStep("2 · Generate", "保存已验证的查询知识、数据库指纹和知识页。", "静态 17 页 + 经验 Markdown/JSON。"),
            LearningStep("3 · Reuse", "再次执行相同问题，检索持久查询知识。", "memory.hit=true；仍重查数值证据。"),
            LearningStep("4 · Invalidate", "在临时副本修改 payment，再执行。", "数据库指纹变化，旧证据失效。"),
        ),
        "observations": ("检索范围是相同任务的已验证查询知识。", "旧数据库版本的经验保留，新版本重新学习。", "离线也保存知识，模型解释不覆盖验证结果。"),
        "exercises": ("连续运行两次，比较 memory.hit。", "运行能力实验，在数据库副本修改付款，观察指纹失效。"),
        "prediction": PredictionSpec(
            "数据库变化后，旧查询知识应怎样处理？",
            (("检索指纹相同的版本", "fresh"), ("无条件复用旧结果", "stale")),
            "fresh",
            "数据库指纹不同就重新生成、验证并保存新的知识版本。",
        ),
    },
    8: {
        "change_from_previous": "从记录知识升级为按 classes、relations、constraints 组织知识生产。",
        "prerequisites": ("理解 schema 与 ontology 的区别。", "了解 OWL、OBDA 与 annotation 的角色。"),
        "steps": (
            LearningStep("1 · Model", "从 schema 生成 ontology classes/relations。", "得到语义候选。"),
            LearningStep("2 · Map", "生成 OWL、OBDA 与两份 annotation。", "四个核心本体文件存在。"),
            LearningStep("3 · Propose", "从 OWL 词表提出类型化关系。", "Live 由 LLM 选择，Offline 枚举允许关系。"),
            LearningStep("4 · Validate", "校验类、domain/range 与外键映射后查询关系数量。", "未知概念或错误 range 阻止知识发布。"),
        ),
        "observations": ("Schema 说明怎样存，本体还说明概念意味着什么。", "结构合法不等于领域知识正确。", "canonical ontology 与 LLM proposal 必须分层。"),
        "exercises": ("选择 payment 表，追踪 table→class→OBDA mapping。", "写出一个需要人工确认而不能仅凭外键生成的业务关系。"),
        "prediction": PredictionSpec(
            "核心 ontology 资产有几类文件？",
            (("2", "2"), ("4", "4"), ("17", "17")),
            "4",
            "一份 OWL、一份 OBDA、两份 annotation，共四个核心文件。",
        ),
    },
    9: {
        "change_from_previous": "知识生产开始服从因果问题：显式区分关联、假设、混杂与可识别性。",
        "prerequisites": ("理解 DAG 与混杂因素。", "接受观察数据不足时应输出识别缺口。"),
        "steps": (
            LearningStep("1 · Evidence", "查询两个门店的库存、租赁、客户与收入。", "得到观察性对比。"),
            LearningStep("2 · DAG", "构建原因—结果—混杂关系。", "DAG 含 6 条边且无环。"),
            LearningStep("3 · Experiment", "按固定种子生成带混杂、已知效应的合成租赁数据。", "朴素差异偏离真实效应。"),
            LearningStep("4 · Adjust", "按需求分层调整，报告区间与 overlap。", "调整恢复已知效应；缺乏 positivity 时拒绝估计。"),
        ),
        "observations": ("真实两门店证据仍只支持关联。", "因果估计来自显式标记的合成 SCM。", "已知效应、识别假设与区间一起报告。"),
        "exercises": ("加入 revenue→inventory_breadth，判断为何产生循环。", "为库存干预设计随机或准实验并列出混杂控制。"),
        "prediction": PredictionSpec(
            "默认因果图是否是 DAG？",
            (("是，无环", "dag"), ("否，有环", "cycle"), ("无法判断", "unknown")),
            "dag",
            "六条边构成有向无环图，但 DAG 合法不等于效应已被识别。",
        ),
    },
    10: {
        "change_from_previous": "不再只接受既定问题：从目标、不变量与约束重新推导可证伪机制。",
        "prerequisites": ("完成 L9 并区分相关与因果。", "理解可证伪假设、实验护栏与停止条件。"),
        "steps": (
            LearningStep("1 · Reframe", "比较收入目标与每库存周期净贡献目标。", "合成模型中的成本可否证收入增长方案。"),
            LearningStep("2 · Decompose", "列出不变量与约束，质疑默认假设。", "边界不会因创新而消失。"),
            LearningStep("3 · Derive", "按库存守恒枚举调拨，并比较采购和保留基线。", "每个可行候选都实际执行模拟。"),
            LearningStep("4 · Falsify", "比较净贡献；对平衡需求运行阴性对照。", "没有改善时 selected=null，不制造创新。"),
        ),
        "observations": ("候选来自约束枚举，结论仅适用于声明的合成模型。", "收入最高的方案可能净贡献最差。", "有限候选全部评估后停止，允许没有改善。"),
        "exercises": ("修改调拨成本，观察最优方案何时失去优势。", "将需求改成平衡情形，验证不再推荐干预。"),
        "prediction": PredictionSpec(
            "默认第一性原理输出包含多少个机制候选？",
            (("2", "2"), ("3", "3"), ("6", "6")),
            "3",
            "机制族是调拨、采购和维持基线；每族的候选由资源约束推导。",
        ),
    },
}


MATERIALS = {
    level: replace(material, sources=material.sources + (
        (SourceSpec("能力干预实验 / Capability experiment", "levels/experiments.py", "python"),)
        if level else ()
    ), **_LESSON_DETAILS[level]) for level, material in MATERIALS.items()
}


def get_teaching_material(level: int) -> TeachingMaterial:
    try:
        return MATERIALS[level]
    except KeyError as exc:
        raise ValueError("teaching level must be between 0 and 10") from exc


def _extract_symbol(path: Path, symbol: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
            start = min((item.lineno for item in node.decorator_list), default=node.lineno)
            lines = text.splitlines()
            return "\n".join(lines[start - 1 : node.end_lineno])
    raise ValueError(f"Symbol {symbol!r} not found in {path}")


def load_source_documents(material: TeachingMaterial) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for spec in material.sources:
        path = (PACKAGE_DIR / spec.path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Teaching source not found: {path}")
        content = _extract_symbol(path, spec.symbol) if spec.symbol else path.read_text(encoding="utf-8")
        reference = path.relative_to(PROJECT_ROOT).as_posix()
        if spec.symbol:
            reference = f"{reference}::{spec.symbol}"
        documents.append(
            SourceDocument(
                label=spec.label,
                reference=reference,
                language=spec.language,
                content=content,
            )
        )
    return documents


def source_tree(documents: list[SourceDocument]) -> str:
    lines = ["agent_kungfu teaching sources"]
    for index, document in enumerate(documents):
        branch = "└──" if index == len(documents) - 1 else "├──"
        lines.append(f"{branch} {document.reference}")
    return "\n".join(lines)


def baseline_image_paths() -> tuple[Path, Path]:
    baseline_dir = (LEVELS_DIR / "baseline").resolve()
    paths = baseline_dir / "sakila.png", baseline_dir / "sakila_structure.png"
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Baseline teaching image not found: {path}")
    return paths


def _check(
    check_id: str,
    title: str,
    passed: bool,
    expected: str,
    actual: Any,
    remediation: str,
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        title=title,
        passed=bool(passed),
        expected=expected,
        actual=str(actual),
        remediation=remediation,
    )


def _sql_check(sql: str | None) -> tuple[bool, str]:
    if not sql:
        return False, "没有 SQL"
    try:
        validate_read_only_sql(sql)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "单条 SELECT/WITH，通过应用层只读守卫"


def evaluate_baseline(frame: pl.DataFrame) -> tuple[CheckResult, ...]:
    """Grade the deterministic reference result without any LLM dependency."""

    rows = frame.to_dicts()
    revenues = [float(value) for value in frame.get_column("revenue").to_list()]
    top = rows[0] if rows else {}
    total = sum(revenues)
    return (
        _check(
            "baseline.shape",
            "结果契约",
            frame.shape == (16, 2) and frame.columns == ["name", "revenue"],
            "16 行，列为 name/revenue",
            f"shape={frame.shape}, columns={frame.columns}",
            "检查 baseline.yaml 是否仍指向 revenue.sql 与正确的 sakila.db。",
        ),
        _check(
            "baseline.order",
            "收入降序",
            all(left >= right for left, right in zip(revenues, revenues[1:], strict=False)),
            "revenue 单调不增",
            revenues[:5],
            "检查 SQL 的 ORDER BY revenue DESC。",
        ),
        _check(
            "baseline.top",
            "榜首类别",
            top.get("name") == "Sports"
            and math.isclose(float(top.get("revenue", 0)), 5314.21, abs_tol=0.01),
            "Sports / 5314.21",
            top,
            "检查五表 join 与 SUM(payment.amount)。",
        ),
        _check(
            "baseline.total",
            "总收入对账",
            math.isclose(total, 67406.56, abs_tol=0.01),
            "67406.56",
            f"{total:.2f}",
            "确认没有丢失 rental/payment 连接记录。",
        ),
    )


def evaluate_level_result(result: LevelResult) -> tuple[CheckResult, ...]:
    """Turn a level result into deterministic, learner-visible PASS/FAIL checks."""

    metadata = result.metadata
    sql_passed, sql_actual = _sql_check(result.sql)
    checks: list[CheckResult] = [
        _check(
            "common.answer",
            "回答非空",
            bool(result.answer.strip()),
            "有基于证据的回答",
            f"{len(result.answer.strip())} characters",
            "检查该层是否完成了最终回答阶段。",
        ),
        _check(
            "common.sql",
            "SQL 只读边界",
            sql_passed,
            "一条 SELECT/WITH",
            sql_actual,
            "返回只读查询，并移除堆叠语句或控制语句。",
        ),
    ]

    if result.level == 1:
        nodes = metadata.get("graph_nodes")
        counts = metadata.get("table_rows", {})
        tool_call = metadata.get("tool_call", {})
        checks.extend(
            [
                _check(
                    "l1.graph",
                    "三步图顺序",
                    nodes == ["read_tables", "merge_tables", "calculate_revenue"],
                    "read → merge → calculate",
                    nodes,
                    "检查 LangGraph 节点与边定义。",
                ),
                _check(
                    "l1.cardinality",
                    "数据基数",
                    counts
                    == {
                        "category": 16,
                        "film_category": 1000,
                        "inventory": 4581,
                        "rental": 16044,
                        "payment": 16049,
                    }
                    and metadata.get("joined_rows") == 16044
                    and metadata.get("revenue_rows") == 16,
                    "五表行数匹配；join=16044；category=16",
                    f"tables={counts}, join={metadata.get('joined_rows')}, revenue={metadata.get('revenue_rows')}",
                    "检查表投影、四个 join key 与聚合分组。",
                ),
                _check(
                    "l1.tool",
                    "Top-N 工具边界",
                    tool_call.get("name") == "top_n_categories"
                    and tool_call.get("arguments", {}).get("n") == result.row_count,
                    "tool=top_n_categories 且 n=返回行数",
                    tool_call,
                    "检查 Jinja2 prompt、tool schema 与 n 的上下界。",
                ),
            ]
        )
    elif result.level == 2:
        checks.extend(
            [
                _check(
                    "l2.pattern",
                    "Manager 模式",
                    metadata.get("sdk") == "openai-agents"
                    and metadata.get("pattern") == "manager-as-tools",
                    "openai-agents / manager-as-tools",
                    f"{metadata.get('sdk')} / {metadata.get('pattern')}",
                    "检查 leader 是否通过 Agent.as_tool 调用 specialists。",
                ),
                _check(
                    "l2.order",
                    "Agent 调用顺序",
                    metadata.get("agent_calls") == ["read_tables", "calculate_revenue"],
                    "read_tables → calculate_revenue",
                    metadata.get("agent_calls"),
                    "让 calculator 仅在 workspace 已有 tables 时启用。",
                ),
                _check(
                    "l2.evidence",
                    "共享数据契约",
                    metadata.get("joined_rows") == 16044
                    and metadata.get("revenue_rows") == 16
                    and len(metadata.get("rows", [])) == result.row_count,
                    "join=16044；revenue=16；rows=Top-N",
                    f"join={metadata.get('joined_rows')}, revenue={metadata.get('revenue_rows')}, rows={len(metadata.get('rows', []))}",
                    "检查共享 workspace 是否完整保留 read/merge/calculate 状态。",
                ),
            ]
        )
    elif result.level == 3:
        correlated = [message.correlation_id for message in result.trace[1:]]
        checks.extend(
            [
                _check(
                    "l3.factory",
                    "Codex 动态团队",
                    len(metadata.get("generated_agents", [])) == 2
                    and metadata.get("runtime_tools") == ["read_tables", "calculate_revenue"]
                    and metadata.get("sdk") == "openai-codex"
                    and metadata.get("transport") == "codex-app-server-jsonrpc",
                    "2 个 Codex workers；SDK=openai-codex；工具仅 read/calculate",
                    f"agents={metadata.get('generated_agents')}, sdk={metadata.get('sdk')}, tools={metadata.get('runtime_tools')}",
                    "检查 agent.md、Codex worker factory、Pydantic action 与 host allowlist。",
                ),
                _check(
                    "l3.protocols",
                    "协议 envelope",
                    metadata.get("protocols") == ["a2a", "codex-jsonrpc", "mcp"]
                    and len(result.trace) == 7
                    and len(metadata.get("codex_turns", [])) == 2,
                    "协议 {a2a,codex-jsonrpc,mcp}，7 条消息，2 个 turns",
                    f"protocols={metadata.get('protocols')}, messages={len(result.trace)}",
                    "把每次 A2A 委派、Codex turn 和 MCP 证据都写入 ProtocolBus。",
                ),
                _check(
                    "l3.correlation",
                    "相关性追踪",
                    len(correlated) == 6
                    and correlated[0] is not None
                    and len(set(correlated)) == 1,
                    "后六条子事件共享 correlation_id",
                    correlated,
                    "用同一个 request id 关联 delegate、Codex turn 与 tool.result。",
                ),
                _check(
                    "l3.sandbox",
                    "Codex 权限边界",
                    metadata.get("sandbox") == "read-only"
                    and metadata.get("approval_mode") == "deny_all"
                    and metadata.get("ephemeral_threads") is True
                    and metadata.get("codex_threads_started") in {0, 2},
                    "read-only / deny_all / ephemeral；offline=0 threads，live=2",
                    f"sandbox={metadata.get('sandbox')}, approvals={metadata.get('approval_mode')}, threads={metadata.get('codex_threads_started')}",
                    "禁止 workspace-write/full-access，并保持每个 worker 使用临时 thread。",
                ),
            ]
        )
    elif result.level == 4:
        evaluation = metadata.get("evaluation", {})
        checks.extend(
            [
                _check(
                    "l4.workers",
                    "两个 Codex 子智能体",
                    metadata.get("generated_agents")
                    == ["Polars Table Reader", "Polars Revenue Calculator"]
                    and metadata.get("runtime_tools")
                    == ["read_tables", "calculate_revenue"],
                    "Reader → Calculator",
                    metadata.get("generated_agents"),
                    "从 OmniGenT agent YAML 只生成两个白名单角色。",
                ),
                _check(
                    "l4.skill",
                    "Polars Skill 可追溯",
                    metadata.get("skill_name") == "polars"
                    and len(str(metadata.get("skill_digest_sha256", ""))) == 64
                    and len(metadata.get("skill_references", [])) == 5,
                    "polars + SHA-256 + 5 references",
                    metadata.get("skill_digest_sha256"),
                    "显式加载并打包 level4/polars，而不是假设 Codex 自动发现。",
                ),
                _check(
                    "l4.evaluation",
                    "Harness 质量门",
                    evaluation.get("passed") is True
                    and evaluation.get("score") == 1.0
                    and metadata.get("joined_rows") == 16044
                    and metadata.get("revenue_rows") == 16,
                    "10 checks passed; delegation verified; join=16044; categories=16",
                    f"score={evaluation.get('score')}, join={metadata.get('joined_rows')}",
                    "在交付前验证 os_env/approval 契约、宿主白名单、顺序、输出域和 baseline parity。",
                ),
                _check(
                    "l4.trace",
                    "控制面 Trace",
                    len([item for item in result.trace if item.kind == "task.delegate"]) == 2
                    and any(item.kind == "harness.evaluated" for item in result.trace),
                    "2 task.delegate + harness.evaluated",
                    [item.kind for item in result.trace],
                    "记录 OmniGenT、Codex、host tool 与 evaluator 的边界。",
                ),
            ]
        )
    elif result.level == 5:
        iterations = metadata.get("iterations", [])
        inspected = metadata.get("inspected_tables", [])
        observations = metadata.get("table_observations", {})
        tool_calls = metadata.get("omnigent_tool_calls", [])
        evaluation = metadata.get("evaluation", {})
        final = iterations[-1] if iterations else {}
        expected_actions = ["inspect_table"] * 16 + ["calculate_revenue"]
        action_names = [
            item.get("action", {}).get("runtime_tool") for item in iterations
        ]
        evaluation_checks = {
            item.get("name"): item.get("passed")
            for item in evaluation.get("checks", [])
        }
        rows = metadata.get("rows", [])
        top = rows[0] if rows else {}
        expected_codex_calls = (17 + sum(len(repair.get("tool_calls", []))
                               for repair in metadata.get("repairs", []))) if metadata.get("omnigent_connected") else 0
        checks.extend(
            [
                _check(
                    "l5.bounded",
                    "17 轮有界轨迹",
                    len(iterations) == 17
                    and metadata.get("iterations_used") == 17
                    and metadata.get("max_iterations") == 17
                    and action_names == expected_actions,
                    "16 次 inspect_table + 1 次 calculate_revenue，预算 17",
                    f"iterations={len(iterations)}, actions={action_names}",
                    "检查 canonical table loop、最后一个计算 action 与 max_iterations。",
                ),
                _check(
                    "l5.coverage",
                    "全表探索证据",
                    metadata.get("all_tables_inspected") is True
                    and metadata.get("table_count") == 16
                    and len(inspected) == 16
                    and len(set(inspected)) == 16
                    and set(observations) == set(inspected)
                    and metadata.get("revenue_chain")
                    == ["category", "film_category", "inventory", "rental", "payment"],
                    "16/16 个唯一表均有 observation，并发现五表收入链",
                    (
                        f"inspected={len(inspected)}, observations={len(observations)}, "
                        f"chain={metadata.get('revenue_chain')}"
                    ),
                    "从 SQLite catalog 固定表序，并让每轮只检查 next_table。",
                ),
                _check(
                    "l5.feedback",
                    "反馈进入下一轮",
                    evaluation_checks.get("feedback-binding") is True
                    and all(
                        item.get("action", {}).get("skill_digest_sha256")
                        == metadata.get("skill_digest_sha256")
                        for item in iterations
                    )
                    and all(
                        bool(item.get("evaluation", {}).get("feedback"))
                        for item in iterations[:-1]
                    ),
                    "每轮反馈非空；action 回显前轮 feedback 与已加载 skill 的 SHA-256",
                    evaluation_checks.get("feedback-binding"),
                    "检查 feedback_digest_sha256 与 skill_digest_sha256 的双重绑定。",
                ),
                _check(
                    "l5.delegation",
                    "OmniGenT + Codex 循环委派",
                    metadata.get("harness") == "omnigent-client"
                    and metadata.get("harness_executor") == "codex"
                    and metadata.get("runtime_tools")
                    == ["inspect_table", "calculate_revenue"]
                    and len(tool_calls) == 17
                    and all(item.get("completed") for item in tool_calls)
                    and all(len(str(item.get("output_sha256") or "")) == 64 for item in tool_calls)
                    and metadata.get("codex_subagent_calls") == expected_codex_calls
                    and evaluation_checks.get("omnigent-codex-delegation") is True,
                    "17 条 completed tool-call evidence；真实 Codex calls：live=17、offline=0",
                    (
                        f"records={len(tool_calls)}, codex={metadata.get('codex_subagent_calls')}, "
                        f"connected={metadata.get('omnigent_connected')}"
                    ),
                    "检查 bundle tools、runner evidence 与 executed_by 模式。",
                ),
                _check(
                    "l5.boundary",
                    "声明式执行边界",
                    metadata.get("sandbox_intent") == "read-only"
                    and metadata.get("sandbox_backend_declared") != "none"
                    and metadata.get("sandbox_write_paths_declared") == []
                    and metadata.get("sandbox_write_files_declared") == []
                    and metadata.get("sandbox_allow_network_declared") is False
                    and metadata.get("sandbox_start_in_scratch_declared") is True
                    and metadata.get("approval_policy_declared") == "never",
                    "read-only intent；无写路径/文件；禁网；scratch；approval=never",
                    {
                        key: metadata.get(key)
                        for key in (
                            "sandbox_intent",
                            "sandbox_backend_declared",
                            "sandbox_write_paths_declared",
                            "sandbox_write_files_declared",
                            "sandbox_allow_network_declared",
                            "sandbox_start_in_scratch_declared",
                            "approval_policy_declared",
                        )
                    },
                    "读取最终 *_declared metadata，并保持 host typed-action allowlist。",
                ),
                _check(
                    "l5.stop",
                    "收入验证后立即停止",
                    metadata.get("terminated") is True
                    and metadata.get("stop_reason") == "goal_satisfied"
                    and bool(final)
                    and final.get("decision") == "stop"
                    and final.get("evaluation", {}).get("passed") is True
                    and evaluation.get("passed") is True
                    and evaluation.get("score") == 1.0
                    and len(evaluation.get("checks", [])) == 10
                    and metadata.get("joined_rows") == 16044
                    and metadata.get("revenue_rows") == 16
                    and top.get("name") == "Sports"
                    and math.isclose(float(top.get("revenue", 0)), 5314.21, abs_tol=0.01),
                    "goal_satisfied；10/10；join=16044；16 类；Sports=5314.21",
                    (
                        f"stop={metadata.get('stop_reason')}, score={evaluation.get('score')}, "
                        f"top={top}"
                    ),
                    "拒绝部分结果；只有完整覆盖和 baseline parity 通过才返回 LevelResult。",
                ),
            ]
        )
    elif result.level == 6:
        manifest = metadata.get("manifest", {})
        graph = metadata.get("graph", {})
        checks.extend(
            [
                _check(
                    "l6.contract",
                    "Harness 类型契约",
                    manifest.get("read_only") is True
                    and manifest.get("max_rows") == 50
                    and manifest.get("max_iterations") == 3,
                    "read_only=true, max_rows=50, max_iterations=3",
                    manifest,
                    "检查 Pydantic HarnessManifest 配置。",
                ),
                _check(
                    "l6.graph",
                    "Schema graph",
                    graph.get("tables") == 16
                    and graph.get("relationships") == 21
                    and graph.get("components") == 2,
                    "16 tables / 21 relationships / 2 components",
                    graph,
                    "从实时外键重新构建 graph_summary。",
                ),
                _check(
                    "l6.path",
                    "协同路径",
                    metadata.get("join_path_payment_to_film")
                    == ["payment", "rental", "inventory", "film"],
                    "payment→rental→inventory→film",
                    metadata.get("join_path_payment_to_film"),
                    "检查外键方向与无向路径投影。",
                ),
                _check(
                    "l6.eval",
                    "质量门",
                    metadata.get("evaluation", {}).get("passed") is True,
                    "evaluation passed",
                    metadata.get("evaluation"),
                    "查看 evaluator feedback 并进入下一轮。",
                ),
            ]
        )
    elif result.level == 7:
        artifact_paths = [Path(path) for path in result.artifacts]
        checks.extend(
            [
                _check(
                    "l7.pages",
                    "持久知识与数据指纹",
                    metadata.get("wiki_pages", 0) >= 18
                    and bool(metadata.get("memory", {}).get("fingerprint"))
                    and Path(metadata.get("memory", {}).get("record", "")).is_file(),
                    "schema pages + verified experience + database fingerprint",
                    f"metadata={metadata.get('wiki_pages')}, artifacts={len(artifact_paths)}",
                    "检查持久经验记录与数据库指纹；保留旧版本用于审计。",
                ),
                _check(
                    "l7.artifacts",
                    "持久工件",
                    bool(artifact_paths)
                    and all(path.is_file() and path.stat().st_size > 0 for path in artifact_paths),
                    "所有 Markdown 文件存在且非空",
                    f"existing={sum(path.is_file() for path in artifact_paths)}/{len(artifact_paths)}",
                    "检查 output_dir 权限与 generate_wiki 写入阶段。",
                ),
                _check(
                    "l7.eval",
                    "知识仍受证据门约束",
                    metadata.get("evaluation", {}).get("passed") is True,
                    "evaluation passed",
                    metadata.get("evaluation"),
                    "以 SQL 证据修正 Wiki，而不是反向覆盖事实源。",
                ),
            ]
        )
    elif result.level == 8:
        filenames = {Path(path).name for path in result.artifacts}
        required = {
            "ontology-man-sakilas.owl",
            "ontology-man-sakilas.obda",
            "anno_sakila.annotation",
            "anno_sakila_man3.annotation",
        }
        checks.extend(
            [
                _check(
                    "l8.core",
                    "核心本体文件",
                    required.issubset(filenames),
                    ", ".join(sorted(required)),
                    ", ".join(sorted(required & filenames)),
                    "检查 generate_sakila_ontology 的四个写入步骤。",
                ),
                _check(
                    "l8.bundle",
                    "知识生产包",
                    len(result.artifacts) >= 21
                    and all(Path(path).is_file() for path in result.artifacts),
                    "4 ontology + 17 wiki，均存在",
                    f"artifacts={len(result.artifacts)}",
                    "检查 ontology_dir、wiki_dir 与输出权限。",
                ),
                _check(
                    "l8.eval",
                    "语义输出后的证据门",
                    metadata.get("evaluation", {}).get("passed") is True,
                    "evaluation passed",
                    metadata.get("evaluation"),
                    "验证映射与查询证据，不把 RDF 语法合法当成业务正确。",
                ),
            ]
        )
    elif result.level == 9:
        edges = {tuple(edge) for edge in metadata.get("causal_dag", [])}
        required_edges = {
            ("inventory_breadth", "rentals"),
            ("rentals", "revenue"),
            ("store_location", "customer_demand"),
            ("store_location", "inventory_breadth"),
        }
        checks.extend(
            [
                _check(
                    "l9.evidence",
                    "观察性证据",
                    result.row_count == 2,
                    "2 个门店",
                    result.row_count,
                    "检查 store 聚合 SQL。",
                ),
                _check(
                    "l9.dag",
                    "因果图结构",
                    metadata.get("is_dag") is True
                    and len(edges) == 6
                    and required_edges.issubset(edges),
                    "6 条边、有向无环、含关键混杂路径",
                    sorted(edges),
                    "修复循环或补回 treatment/outcome/confounder 边。",
                ),
                _check(
                    "l9.caveat",
                    "关联不冒充因果",
                    "不是因果" in result.answer or "not causal" in result.answer.lower(),
                    "回答明确说明观察关联不是因果效应",
                    result.answer[:160],
                    "补充识别假设、混杂因素与所需实验。",
                ),
            ]
        )
    elif result.level == 10:
        principles = metadata.get("first_principles", {})
        checks.extend(
            [
                _check(
                    "l10.causal",
                    "继承因果边界",
                    metadata.get("is_dag") is True and result.row_count == 2,
                    "保留 L9 DAG 与两门店观察证据",
                    f"is_dag={metadata.get('is_dag')}, rows={result.row_count}",
                    "创新层仍需保留证据和因果识别边界。",
                ),
                _check(
                    "l10.structure",
                    "第一性原理结构",
                    bool(principles.get("objective"))
                    and len(principles.get("invariants", [])) == 3
                    and len(principles.get("constraints", [])) == 4,
                    "1 objective / 3 invariants / 4 constraints",
                    f"objective={bool(principles.get('objective'))}, invariants={len(principles.get('invariants', []))}, constraints={len(principles.get('constraints', []))}",
                    "先补齐目标、不变量和约束，再提出机制。",
                ),
                _check(
                    "l10.falsifiable",
                    "机制与实验",
                    len(principles.get("mechanisms", [])) == 3
                    and len(principles.get("experiments", [])) == 3,
                    "3 mechanisms / 3 experiments",
                    f"mechanisms={len(principles.get('mechanisms', []))}, experiments={len(principles.get('experiments', []))}",
                    "为每个机制配一个可证伪实验；再由学习者补充护栏和停止条件。",
                ),
            ]
        )
    else:
        checks.append(
            _check(
                "common.level",
                "层级有效",
                False,
                "Level 1–10",
                result.level,
                "选择受支持的教学层级。",
            )
        )
    mechanism = {
        2: ("SDK 实际委派", len(metadata.get("sdk_tool_calls", [])) == 2),
        3: ("JSON-RPC 执行边界", len(metadata.get("rpc_exchanges", [])) == 2),
        4: ("证据裁决冲突", metadata.get("conflict_decision", {}).get("selected_agent") == "reference-backed"),
        6: ("图驱动执行", metadata.get("task_order", [])[-1:] == ["aggregate"]),
        8: ("本体发布门", metadata.get("semantic_gate", {}).get("accepted_claims", 0) > 0),
        9: ("合成因果识别", abs(metadata.get("causal_experiment", {}).get("estimates", {}).get("adjusted", 100)
                              - metadata.get("causal_experiment", {}).get("known_effect", 0)) < 0.2),
        10: ("创新阴性对照", metadata.get("negative_control", {}).get("selected", "missing") is None),
    }
    if result.level in mechanism:
        title, passed = mechanism[result.level]
        checks.append(_check(f"l{result.level}.mechanism", title, passed,
                             "可观察的运行证据", passed, "执行对应能力实验，查看干预结果。"))
    return tuple(checks)


def execution_observations(
    level: int,
    *,
    baseline: pl.DataFrame | None = None,
    result: LevelResult | None = None,
) -> list[dict[str, str]]:
    """Build a small, curated execution journal instead of dumping raw metadata."""

    if level == 0:
        if baseline is None:
            return []
        top = baseline.row(0, named=True)
        return [
            {"步骤": "执行 SQL", "运行证据": f"{baseline.height} categories", "意义": "完整基准规模"},
            {"步骤": "排序", "运行证据": f"{top['name']} / {float(top['revenue']):.2f}", "意义": "榜首与金额"},
            {"步骤": "对账", "运行证据": f"{float(baseline['revenue'].sum()):.2f}", "意义": "总收入不丢失"},
        ]
    if result is None:
        return []
    metadata = result.metadata
    if level == 1:
        return [
            {"步骤": "Read", "运行证据": str(metadata.get("table_rows")), "意义": "五张源表基数"},
            {"步骤": "Merge", "运行证据": str(metadata.get("joined_rows")), "意义": "完整支付链"},
            {"步骤": "Calculate + Tool", "运行证据": str(metadata.get("tool_call")), "意义": "LLM 只控制 N"},
        ]
    if level == 2:
        return [
            {"步骤": "Team", "运行证据": " → ".join(metadata.get("agents", [])), "意义": "一主两专"},
            {"步骤": "Calls", "运行证据": " → ".join(metadata.get("agent_calls", [])), "意义": "状态依赖顺序"},
            {"步骤": "Workspace", "运行证据": f"join={metadata.get('joined_rows')}, revenue={metadata.get('revenue_rows')}", "意义": "大数据不穿过消息"},
        ]
    if level == 3:
        correlations = {item.correlation_id for item in result.trace[1:]}
        return [
            {"步骤": "Generate", "运行证据": ", ".join(metadata.get("generated_agents", [])), "意义": "agent.md 驱动角色"},
            {"步骤": "Codex SDK", "运行证据": f"sdk={metadata.get('sdk')}; threads={metadata.get('codex_threads_started')}", "意义": "live 为真实 app-server，offline 为显式模拟"},
            {"步骤": "Protocols", "运行证据": ", ".join(metadata.get("protocols", [])), "意义": "A2A / JSON-RPC / MCP 边界"},
            {"步骤": "Correlate", "运行证据": str(next(iter(correlations), None)), "意义": "跨消息追踪请求"},
        ]
    if level == 4:
        return [
            {"步骤": "Load Skill", "运行证据": str(metadata.get("skill_digest_sha256")), "意义": "能力说明可追溯"},
            {"步骤": "Generate", "运行证据": " → ".join(metadata.get("generated_agents", [])), "意义": "OmniGenT 定义两个 Codex 角色"},
            {"步骤": "Execute", "运行证据": " → ".join(metadata.get("agent_calls", [])), "意义": "host allowlist 顺序"},
            {"步骤": "Evaluate", "运行证据": str(metadata.get("evaluation")), "意义": "失败即阻止交付"},
        ]
    if level == 5:
        iterations = metadata.get("iterations", [])
        inspected = metadata.get("inspected_tables", [])
        first_feedback = (
            iterations[0].get("evaluation", {}).get("feedback", [])
            if iterations
            else []
        )
        final = iterations[-1] if iterations else {}
        tool_calls = metadata.get("omnigent_tool_calls", [])
        executed_by = sorted({item.get("executed_by", "unknown") for item in tool_calls})
        feedback_bound = next(
            (
                item.get("passed")
                for item in metadata.get("evaluation", {}).get("checks", [])
                if item.get("name") == "feedback-binding"
            ),
            False,
        )
        return [
            {
                "步骤": "Discover",
                "运行证据": f"{len(inspected)}/16: " + " → ".join(inspected),
                "意义": "目录快照按 canonical 顺序完整推进",
            },
            {
                "步骤": "Feedback",
                "运行证据": (
                    f"first={first_feedback}; feedback+skill binding={feedback_bound}; "
                    f"skill={str(metadata.get('skill_digest_sha256', ''))[:12]}"
                ),
                "意义": "前轮评价与已加载 skill 同时绑定下一 action",
            },
            {
                "步骤": "Delegate",
                "运行证据": (
                    f"records={len(tool_calls)}; actual_codex="
                    f"{metadata.get('codex_subagent_calls')}; executed_by={executed_by}"
                ),
                "意义": "live 与 simulated 边界可核验",
            },
            {
                "步骤": "Boundary",
                "运行证据": (
                    f"intent={metadata.get('sandbox_intent')}; "
                    f"backend={metadata.get('sandbox_backend_declared')}; "
                    f"network={metadata.get('sandbox_allow_network_declared')}; "
                    f"approval={metadata.get('approval_policy_declared')}"
                ),
                "意义": "展示声明边界，不把声明误写成跨平台强制保证",
            },
            {
                "步骤": "Calculate",
                "运行证据": (
                    f"action={final.get('action', {}).get('runtime_tool')}; "
                    f"join={metadata.get('joined_rows')}; categories={metadata.get('revenue_rows')}"
                ),
                "意义": "全表覆盖后才允许收入计算",
            },
            {
                "步骤": "Stop",
                "运行证据": (
                    f"{metadata.get('stop_reason')} @ "
                    f"{metadata.get('iterations_used')}/{metadata.get('max_iterations')}"
                ),
                "意义": "通过 baseline parity 后立即有界退出",
            },
        ]
    if level == 6:
        return [
            {"步骤": "Contract", "运行证据": str(metadata.get("manifest")), "意义": "权限与预算类型化"},
            {"步骤": "Graph", "运行证据": str(metadata.get("graph")), "意义": "关系结构可计算"},
            {"步骤": "Path", "运行证据": " → ".join(metadata.get("join_path_payment_to_film", [])), "意义": "跨表协同路径"},
            {"步骤": "Execute", "运行证据": str(metadata.get("task_order")), "意义": "任务图实际调度 Polars"},
        ]
    if level == 7:
        return [
            {"步骤": "Generate", "运行证据": f"{metadata.get('wiki_pages')} pages", "意义": "知识持久化"},
            {"步骤": "Persist", "运行证据": f"{len(result.artifacts)} files", "意义": "可读、可 diff"},
            {"步骤": "Retrieve", "运行证据": str(metadata.get("memory")), "意义": "跨次检索与指纹失效"},
            {"步骤": "Evaluate", "运行证据": str(metadata.get("evaluation")), "意义": "知识服从证据"},
        ]
    if level == 8:
        extensions = sorted({Path(path).suffix for path in result.artifacts})
        return [
            {"步骤": "Generate", "运行证据": ", ".join(extensions), "意义": "语义资产组合"},
            {"步骤": "Organize", "运行证据": f"{len(result.artifacts)} files", "意义": "本体指导知识生产"},
            {"步骤": "Evaluate", "运行证据": str(metadata.get("evaluation")), "意义": "语法合法不替代证据"},
            {"步骤": "Semantic gate", "运行证据": str(metadata.get("semantic_gate")), "意义": "本体约束控制发布"},
        ]
    if level == 9:
        return [
            {"步骤": "Evidence", "运行证据": f"{result.row_count} stores", "意义": "观察性对比"},
            {"步骤": "DAG", "运行证据": f"{len(metadata.get('causal_dag', []))} edges; dag={metadata.get('is_dag')}", "意义": "显式假设"},
            {"步骤": "Identify", "运行证据": "观察关联 ≠ 因果效应", "意义": "暴露识别缺口"},
            {"步骤": "Synthetic experiment", "运行证据": str(metadata.get("causal_experiment", {}).get("estimates")), "意义": "已知真值检验分层调整"},
        ]
    principles = metadata.get("first_principles", {})
    return [
        {"步骤": "Reframe", "运行证据": str(principles.get("objective")), "意义": "重新定义可度量目标"},
        {"步骤": "Derive", "运行证据": f"{len(principles.get('mechanisms', []))} mechanisms", "意义": "从不变量推导"},
        {"步骤": "Falsify", "运行证据": f"{len(principles.get('experiments', []))} experiments", "意义": "创新必须能失败"},
        {"步骤": "Select", "运行证据": str(metadata.get("innovation_experiment", {}).get("selected")), "意义": "实际比较净贡献"},
        {"步骤": "Negative control", "运行证据": str(metadata.get("negative_control", {}).get("stop_reason")), "意义": "允许没有改善"},
    ]
