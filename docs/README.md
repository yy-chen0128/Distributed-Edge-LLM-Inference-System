# docs — 调研文档与项目规划

本目录整理 `research/` 的调研成果，按主题分类。完整原始文件仍在 `research/`。

## 目录结构

```
docs/
├── research/          # 核心调研成果（文献分析、框架设计、模拟可行性）
├── plan/              # 项目规划（论文大纲、进度、阶段门）
├── sections/          # 论文章节（方法/技术方案）
├── evidence-map.md    # 证据地图（文献→研究问题的映射）
├── data-manifest.md   # 数据清单
├── table-schema.md    # 表格 schema
└── README.md
```

## research/ — 核心调研成果

| 文档 | 内容 |
|---|---|
| **framework-design.md** | 框架设计说明：架构分层、接口设计、机制vs策略、三层验证 |
| **simulation-feasibility.md** | 模拟测试可行性：论文数据集、LMCache/vLLM 无 GPU 模拟、真实对接现状 |
| **literature-notes.md** | 文献笔记：70 篇论文主题聚类 + 跨论文综合洞察 |
| **moe-literature.md** | MoE 方向文献：EP 专家并行、KV+专家联合路由、13 篇新下载 |
| **research-alignment.md** | 研究问题 D1-D6 与文献相关性映射 |
| **polylink-deep-analysis.md** | PolyLink 33 篇边缘推理论文深入分析 |
| **evidence-coverage.md** | 证据覆盖度（研究问题的文献支撑） |
| **method-experiment-traceability.md** | 方法-实验可追溯性 |
| **rubric-test-set.md** / **screening-rubric.md** | 论文筛选与评估标准 |

## plan/ — 项目规划

| 文档 | 内容 |
|---|---|
| **project-overview.md** | 项目范围、研究问题 D1-D6、约束、写作偏好 |
| **outline.md** | 论文大纲（section 结构） |
| **progress.md** | 进度跟踪 |
| **section-architecture.md** | Section 架构（文件/角色/长度） |
| **experiment-protocol.md** | 实验协议 |
| **stage-gates.md** | 阶段门（里程碑） |
| **submission-index.md** / **venue-template-profile.md** | 投稿计划与会议画像 |

## sections/ — 论文章节

| 文档 | 内容 |
|---|---|
| **02_method.md** | 技术方案（方法章节）：任务设定、机制层、策略层（E2/PriorityMigration）、验证方法 |
