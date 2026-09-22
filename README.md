# Distributed-Edge-LLM-Inference-System — 边缘分布式 LLM 推理调度系统

在边缘异构算力节点（消费级 GPU / 手机 / 边缘盒）上协作执行 LLM 推理的**管理调度层**。

> **入口文档**
> 1. [`docs/deploy/project-structure-and-build.md`](docs/deploy/project-structure-and-build.md) —— 项目由什么构成、怎么跑、改动落在哪一层
> 2. [`docs/deploy/environment-setup.md`](docs/deploy/environment-setup.md) —— 单机环境配置（Windows + WSL2 / Ubuntu）
> 3. [`docs/deploy/four-laptop-integration-plan.md`](docs/deploy/four-laptop-integration-plan.md) —— 四机怎么整合、切层怎么定、分阶段计划

## 项目结构

```
Distributed-Edge-LLM-Inference-System/
├── edge_llm_scheduler/        # 核心框架（自研调度层）
│   ├── core/                  # 机制层：类型/存储/传输/事件/节点/模型账本/任务调度/epoch 重配置
│   ├── policies/              # 策略层：放置/迁移/重并行化/中断恢复（可插拔）
│   ├── backends/              # 执行与存储：真实 HF 分层运行时 / toy / mock / TCP / LMCache / vLLM 客户端
│   ├── agents/                # 节点 agent：一个进程承载一段层（TCP 控制面 + activation 数据面）
│   ├── experiments/           # 实验入口：真实跨机流水线 / 数值一致性 / 链路测量 / 分层模拟
│   ├── deploy/                # 四机部署配置与启动脚本（cluster.example.json / start_agent.*）
│   └── tests/                 # 测试（86+ passed；skip 为可选集成未装，skip≠pass）
├── scripts/                   # 环境与工具脚本（collect_env / model_pp_fit / hf_mirror_download / WSL 自检）
├── docs/                      # 文档：research 调研 · plan 计划 · deploy 部署与实测（见 docs/README.md）
├── vllm/  LMCache/            # 第三方源码快照，仅供阅读与复用评估（上游与版本见 docs/deploy/project-structure-and-build.md）
├── moe-infinity/ preble/ spotserve/   # 参考实现 clone（已在 .gitignore 中排除）
└── README.md
```

## 问题背景

在**带宽受限 + 节点动态变化 + 内存受限**的边缘场景执行 LLM 推理，三类约束同时成立：
1. **带宽受限**：节点间链路（WiFi/毫米波）比数据中心低 1-2 个数量级 → 跨设备通信成本高
2. **节点动态变化（churn）**：节点随时加入/离开，无宽限期 → 需快速装载/卸载与状态迁移
3. **内存受限**：边缘显存小，KV 缓存无法全放 GPU → 需分级存储 + 缓存复用

## 核心设计：机制与策略分离

- **机制（core/）**：框架必须完整实现——对接接口、数据结构、流程编排
- **策略（policies/）**：可插拔算法——"怎么做"由策略决定，可对比实验、逐步替换

### 已实现策略
| 策略 | 逻辑 | 性能目标 |
|---|---|---|
| **E2Placement** | 缓存感知路由（exploit 命中 vs explore 负载均衡，参考 Preble） | 命中省 prefill（边缘最值钱），负载均衡 |
| **PriorityMigration** | KV 块价值 `PT×N`（重算时间×共享率）降序迁移，时间预算用完即丢 | 时间受限下保高价值 KV |
| **CapabilityReparallelization** | 按节点算力比例重切层区间 | 节点变化后木桶效应最小 |
| **TokenRecovery** | 保留已提交 KV，从 progress_tokens 继续 | 中断后最小重算 |
| **CapabilityPlacement** | 能力加权负载路由 | 算力强多承担 |

### 模拟实验结论
真实共享前缀负载下（Preble 风格多前缀）：
```
[default]  hit_rate=0.319  全堆单节点
[e2]       hit_rate=0.957  分散到 3 节点
```
E2 缓存感知路由命中率提升 0.64，负载更均衡。

## 快速开始

> 完整环境配置（WSL2、GPU、依赖、模型下载、链路测量）见
> [`docs/deploy/environment-setup.md`](docs/deploy/environment-setup.md)。

```bash
# ① 控制面（不需要 GPU）
PYTHONPATH=. python -m pytest edge_llm_scheduler/tests/ -q      # 86+ passed
PYTHONPATH=. python -m edge_llm_scheduler.cli --nodes 3 --requests 3

# ② 真实分层推理（需要 torch 与本地模型；device 可换 cpu）
python -m edge_llm_scheduler.experiments.verify_equivalence \
  --model .models/Qwen2.5-0.5B-Instruct --stages 4 --max-tokens 8
python -m edge_llm_scheduler.experiments.run_real_pipeline --local 4 \
  --device cuda:0 --model .models/Qwen2.5-0.5B-Instruct --max-tokens 8

# ③ 四机分布式：每台起 agent，控制端给 cluster.json
python -m edge_llm_scheduler.agents.stage_agent --node-id alpha --port 9100 \
  --host 0.0.0.0 --model .models/Qwen2.5-0.5B-Instruct --device cuda:0
python -m edge_llm_scheduler.experiments.run_real_pipeline \
  --cluster edge_llm_scheduler/deploy/cluster.example.json --max-tokens 16
```

### LMCache 真实对接（无需 GPU）
```bash
python -m pytest edge_llm_scheduler/tests/test_lmcache_real.py -v   # 未装 lmcache 时 skip
```

## 文档与后续方向

文档已按主题整理到 `docs/`，索引见 [`docs/README.md`](docs/README.md)：

| 想知道什么 | 看哪份 |
|---|---|
| 项目结构与构建方式 | [`docs/deploy/project-structure-and-build.md`](docs/deploy/project-structure-and-build.md) |
| 怎么配环境 | [`docs/deploy/environment-setup.md`](docs/deploy/environment-setup.md) |
| 四机整合与切层计划 | [`docs/deploy/four-laptop-integration-plan.md`](docs/deploy/four-laptop-integration-plan.md) |
| 实测数据与能力边界 | [`docs/deploy/edge-4gpu-deployment-analysis.md`](docs/deploy/edge-4gpu-deployment-analysis.md) |
| 推理优化路径全景 | [`docs/research/inference-optimization-landscape.md`](docs/research/inference-optimization-landscape.md) |
| 当前进度与路线图 | [`docs/plan/current-work-and-roadmap.md`](docs/plan/current-work-and-roadmap.md) |

### 后续方向（论文推进）

**近期（模拟完善）**：
1. 用 SpotServe 的节点事件 trace 驱动完整"节点加入/离开"模拟
2. 用 MoE-Infinity contextpilot 的 32K 长前缀负载测 E2 路由
3. LMCache 真实存储 + 模拟负载的端到端模拟

**中期（真实系统）**：
1. vLLM 真实对接（需 Linux/编译环境；当前 AST 静态验证已覆盖契约）
2. 无线传输（WiGig 等）真实链路适配
3. 多节点真实部署（边缘 GPU 集群）

**研究空白（论文贡献）**：
1. **KV + 专家双对象联合路由**：Preble 只管 KV，SMoE 只管专家——把两者同时纳入路由是空白
2. **边缘 churn 的事前冗余复制**：无宽限期场景"事后迁移"先天受限，探索热点 KV/专家提前复制
3. **EP 的边缘可行性**：专家按热度静态分布 + 请求路由对齐，而非 token 级 all-to-all
