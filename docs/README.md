# docs — 调研文档与项目规划

本目录整理 `research/` 的调研成果，按主题分类。完整原始文件仍在 `research/`。

> **只读一份的话读这份** →
> [`research/inference-optimization-understanding.md`](research/inference-optimization-understanding.md)
> 《推理优化：我们的理解、要解决的问题与可能路径》——会议用总纲：全貌五层地图、四个硬约束与 D1–D6、
> 已确立的实测结论（PP 不降延迟、两端固定开销、TP 弱网不可行…）、8 条可能路径与各自代价、
> 与 vLLM/LMCache/PAIR 等系统的分工、待决问题。

## 目录结构

```
docs/
├── research/          # 核心调研成果（文献分析、框架设计、模拟可行性、引擎复用评估）
├── plan/              # 项目规划（论文大纲、进度、阶段门、路线图）
├── deploy/            # 部署与实测（环境配置、四机整合、实测分析）
├── design/            # 设计讨论（单机范围、批处理/微批方案）
├── sections/          # 论文章节（方法/技术方案）
├── evidence-map.md    # 证据地图（文献→研究问题的映射）
├── data-manifest.md   # 数据清单
├── table-schema.md    # 表格 schema
└── README.md
```

## deploy/ — 部署、环境与实测

| 文档 | 内容 |
|---|---|
| **environment-setup.md** | **环境配置文档**：WSL2/Ubuntu 选型、GPU 与 venv、依赖、模型下载、起节点 agent、链路测量、常见坑、一页速查 |
| **project-structure-and-build.md** | **项目结构与构建说明**：三层架构、两条执行轨道、目录职责、关键数据契约、四种运行方式、外部参考实现来源、仓库约定与文档地图 |
| **four-laptop-integration-plan.md** | 四台笔记本整合计划：OS 决策、GPU 环境、模型选型（含"平均切失败/加权切成功"算例）、优化构件与复用情况、P0–P5 分阶段计划、风险清单 |
| **edge-4gpu-deployment-analysis.md** | 真实部署实测分析：四段流水线数值正确性、分片足迹、成本模型、节点离开重配置、并发吞吐、异构切层、PP/TP 定量对比、实验方案 |

## design/ — 设计讨论

| 文档 | 内容 |
|---|---|
| **single-machine-scope-and-batching.md** | 单机还能做什么（8 项按性价比排序）+ 批处理/微批/DP 的概念区分、为什么该做批处理（固定开销 ≈ 计算量的实测推导）、要改的引擎/协议/控制器、验收口径、单机边界（四个 agent 共用一块 GPU） |

## 当前总文档

| 文档 | 内容 |
|---|---|
| **plan/current-work-and-roadmap.md** | 当前系统实现、验证边界、旧卡多卡实验方案与后续路线图 |

## research/ — 核心调研成果

| 文档 | 内容 |
|---|---|
| **framework-design.md** | 框架设计说明：架构分层、接口设计、机制vs策略、三层验证 |
| **inference-optimization-understanding.md** ★ | **会议用总纲**：推理优化五层全貌、我们系统的四个硬约束与 D1–D6、已确立的实测结论、8 条可能路径与代价、与已有系统的分工、待决问题、术语表 |
| **simulation-feasibility.md** | 模拟测试可行性：论文数据集、LMCache/vLLM 无 GPU 模拟、真实对接现状 |
| **literature-notes.md** | 文献笔记：70 篇论文主题聚类 + 跨论文综合洞察 |
| **moe-literature.md** | MoE 方向文献：EP 专家并行、KV+专家联合路由、13 篇新下载 |
| **research-alignment.md** | 研究问题 D1-D6 与文献相关性映射 |
| **polylink-deep-analysis.md** | PolyLink 33 篇边缘推理论文深入分析 |
| **evidence-coverage.md** | 证据覆盖度（研究问题的文献支撑） |
| **method-experiment-traceability.md** | 方法-实验可追溯性 |
| **rubric-test-set.md** / **screening-rubric.md** | 论文筛选与评估标准 |
| **inference-optimization-landscape.md** | **大模型推理优化路径全景**（五层地图、43 条路径的拥挤度与四卡可行性、算子/稀疏注意力/KV 专项的归属与追问） |
| **kernels-attention-and-memory-primer.md** | 概念参照 + 成熟方向现状：融合/图优化在做什么、FA2/3/4 分代、FlexAttention、稀疏与 KV 驱逐、批 vs 数据并行、权重流式加载/P2P 分发/多级 KV；LMCache 自研算子层清单 |
| **l2-l3-customization-and-reuse.md** | **vLLM/LMCache 的自定义边界与复用评估**：免 fork 扩展点、KV Connector 层粒度接口、能力矩阵（一个实例能持有/改变什么）、必须 fork 的部分、L2 架构门槛 |
| **policies-principles-and-testability.md** | **已实现策略的原理、实现与可测性**：5 条主策略的判据/算法/参数/代码位置，实跑证据（E2 命中率 0.321→0.962、切层 3/4/5、迁移预算表），哪些数字可信哪些不可信，8 项实现缺陷的修复状态 |
| **real-workload-datasets.md** | **真实负载数据集选型**：四类信号（到达/前缀共享/长度分布/churn）各自的最佳来源，逐个核对字段与许可，前缀共享与命中率的可引用数字（Preble 85–97%、Mooncake 0.30–0.51、Codex 94.2%），门控与质量坑，以及"公开数据里没有节点 churn"这一负面结论；附模型规模/量化选型 |
| **parameter-and-kv-management-status.md** | **参数与 KV 管理的实现状态（写给不读代码的人）**：每个概念先给定义（token/prefill/decode/KV/层/词表/progress_tokens），再指出代码里的名字，最后给"实现了/只有数据结构/完全没有"三档结论；含 progress_tokens 为何失效、KV 迁移缺哪座桥、分页要不要自己做、MoE 专家卸载、词表切分，以及 7B 在四机上的实测算表 |
| **inference-engine-selection.md** | **推理引擎选型**：自研 vs vLLM/SGLang vs 边缘引擎（Ollama / LM Studio / llama.cpp RPC / exo / Mesh-LLM / Petals）；六条需求（R1–R6）逐项判定，Ollama 用**克隆的源码**核对过环境变量与调度、llama.cpp RPC 与 exo 用官方 README 核对；结论是"没有现成引擎能替我们做跨机异构+弹性"，建议三层结构（执行面自研 / 单机能力借或抄 / 基线分三层做），并附论文可直接引用的关键句 |
| **multi-machine-inference-engine-survey-2026-09.md** | **多机消费级推理引擎调研全文**（69 KB）：llama.cpp RPC / exo / distributed-llama / Petals / ktransformers / PowerInfer / Mesh-LLM / prima.cpp / SpotServe / LUMEN / DynaPipe / PipeBoost / TPI-LLM 等逐个核验，每条结论带 URL 与 V（抓到原文）/I（推断）/U（未确认）分级，含未核验清单 |
| **vllm-edge-integration.md** / **vllm-source-modification-plan.md** | vLLM 与边缘分层系统的集成边界 / 源码级动态调度改造评估 |
| **ascend-deployment-assessment.md** | 昇腾云算力部署评估（已不进入当前实验主线） |
| **nvidia-pair-analysis.md** | NVIDIA Personal AI Router 技术分析：定位、调度了什么、PP/TP/DP 归属、节点能力判定、带宽与节点离开、请求全流程（与我们的系统作对照） |

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
