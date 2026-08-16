# distributed-llm-scheduler — 边缘分布式 LLM 推理调度系统

在边缘异构算力节点（消费级 GPU / 手机 / 边缘盒）上协作执行 LLM 推理的**管理调度层**。

## 项目结构

```
distributed-llm-scheduler/
├── edge_llm_scheduler/        # 核心框架（自研调度层）
│   ├── core/                  # 机制层：类型/存储/传输/事件/节点/模型装载/任务调度
│   ├── policies/              # 策略层：放置/迁移/重并行化/中断恢复（可插拔）
│   ├── backends/              # 对接：mock / LMCache / vLLM / TCP / WiFi
│   ├── experiments/           # 模拟实验
│   └── tests/                 # 测试（64+ 通过）
├── LMCache/                   # KV 缓存存储与传输（官方 clone，纯 CPU 可测）
├── vllm/                      # LLM 推理引擎（官方 clone）
├── docs/                      # 调研文档与项目规划（见 docs/README.md）
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

```bash
# 依赖（torch CPU 版即可）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 跑测试
cd edge_llm_scheduler
python -m pytest tests/ -q          # 64 passed

# 模拟实验（对比策略命中率）
python -m edge_llm_scheduler.experiments.run_simulation
```

### LMCache 真实对接（无需 GPU）
```bash
# 真实 LMCache CPU 存储后端验证
python -m pytest tests/test_lmcache_real.py -v   # 2 passed
```

## 文档与后续方向

详细设计见 `research/plan/review/`（framework-design.md、simulation-feasibility.md 等）。

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
