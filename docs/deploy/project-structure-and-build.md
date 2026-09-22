# 项目结构与构建说明

> 用途：让任何人（含新加入的同学）在 10 分钟内搞清楚**这个项目由什么构成、怎么跑起来、改动应该落在哪一层**。
> 读完这份再看 [`environment-setup.md`](environment-setup.md) 配环境。

---

## 1. 这是什么

**在"多台异构边缘设备"上协作执行大模型推理的调度系统。**
核心不是单机推理框架，而是：把模型按层切成多段放到不同设备上、跨机传递中间激活（activation）、
管理 KV 缓存的装载与迁移，并在**节点中途加入/离开、链路波动、显存紧张**时继续服务。

当前验证状态：控制面与数据面机制已在**单机多进程 + 真实 GPU**上跑通
（`--local 4 --device cuda:0`，真实 Qwen2.5-0.5B、真实 KV、真实采样，输出正确）。

---

## 2. 三层架构

```
┌──────────────────────────────────────────────────────────────┐
│ 应用入口与实验驱动  experiments/、cli.py、deploy/              │
│   run_real_pipeline（真实跨机流水线）· run_layered_simulation   │
│   verify_equivalence（数值一致性）· measure_link（链路测量）     │
├──────────────────────────────────────────────────────────────┤
│ 动态调度与控制层  core/ + policies/            ← 本项目的核心    │
│   NodeManager        节点注册/心跳/加入离开                     │
│   ModelManager       层与专家的装载账本（谁持有哪几层）          │
│   TaskScheduler      路由→下发→回收、节点事件挂钩               │
│   PipelineReconfigurationCoordinator  epoch 化重配置（先备后切） │
│   KVStore / Transport / EventBus   统一抽象                     │
│   policies/          放置·迁移·重并行化·恢复（可插拔策略）       │
├──────────────────────────────────────────────────────────────┤
│ 执行与存储后端  backends/  + agents/                           │
│   HFLayeredEngine     ★真实 HuggingFace 分层运行时（只载本段层） │
│   TorchLayeredEngine  toy 层运行时（无模型时可跑）               │
│   LayeredMockEngine / MockEngine   纯控制流验证                 │
│   VLLMEngine          完整 vLLM endpoint 客户端（不做 stage）    │
│   stage_agent         节点进程：TCP 控制面 + activation 数据面   │
└──────────────────────────────────────────────────────────────┘
```

**机制与策略分离**是本项目的第一原则：
`core/` 是机制（必须稳定、有测试），`policies/` 是策略（可替换、用于做实验对比）。

---

## 3. 两条执行轨道

| 轨道 | 用途 | 依赖 | 能验证什么 |
|---|---|---|---|
| **A. 无硬件轨道** | `MockEngine` / `LayeredMockEngine` | 只要 Python | 控制流、epoch 切换、KV 目录、迁移与恢复策略 |
| **B. 真实执行轨道** | `HFLayeredEngine` + `stage_agent` | torch + transformers（GPU 可选） | **真实权重、真实 KV、真实 activation 传输、真实跨机流水线** |
| **C. 基线轨道**（对照） | `VLLMEngine`（HTTP 客户端） | 运行中的 vLLM 服务（Linux） | 连续批处理/前缀缓存的吞吐与延迟基线 |

> **C 不是分层执行**：`VLLMEngine` 只能调用一个完整 endpoint，会显式拒绝 `stage_count > 1`。
> 别试图用多个 vLLM endpoint 串成流水线。

---

## 4. 目录与职责

```
project/
├── edge_llm_scheduler/
│   ├── core/                    机制层（稳定，改动需谨慎）
│   │   ├── types.py             Node/Model/Request/Task/KVBlock/ActivationEnvelope/PipelinePlan
│   │   ├── node_manager.py      节点生命周期与心跳
│   │   ├── model_manager.py     层/专家装载账本 + 流水线覆盖校验
│   │   ├── task_scheduler.py    请求路由、多 stage 编排、节点事件挂钩
│   │   ├── pipeline_controller.py  epoch 化重配置（prepare→切流→drain）
│   │   ├── stage_runtime.py     StageRuntime 抽象（prepare/activate/retire/generate）
│   │   ├── storage.py           KVStore 抽象
│   │   ├── transport.py         Transport 抽象（带宽/时延/传输时间估计）
│   │   └── event_bus.py         异步事件总线
│   ├── policies/                策略层（做实验在这里加变体）
│   │   ├── placement.py         DefaultPlacement / CapabilityPlacement / E2Placement(缓存感知) / LayeredPipelinePlacement
│   │   ├── migration.py         DefaultMigration(全量) / PriorityMigration(reuse×prefill_time + 时间预算)
│   │   ├── reparallelization.py CapabilityReparallelization（按能力重切层）
│   │   └── recovery.py          DefaultRecovery / TokenRecovery（token 级续跑）
│   ├── backends/                执行与存储后端
│   │   ├── hf_layered_engine.py   ★真实模型分层运行时（meta 骨架 + 只加载本段权重）
│   │   ├── torch_layered_engine.py toy tensor 运行时
│   │   ├── layered_mock_engine.py 分层控制流模拟
│   │   ├── mock_engine.py / mock_storage.py / mock_transport.py
│   │   ├── tcp_transport.py     真实 TCP push/pull
│   │   ├── wifi_transport.py    模拟无线链路（带宽/时延/丢包）
│   │   ├── lmcache_storage.py   LMCache 适配层
│   │   └── vllm_engine.py       完整 endpoint 客户端
│   ├── agents/stage_agent.py    节点 agent：一个进程承载一段层，TCP 控制+数据面
│   ├── experiments/             实验入口（下面第 5 节）
│   ├── deploy/                  四机部署配置与启动脚本（cluster.example.json / start_agent.*）
│   └── tests/                   88 个测试（mock + CPU + 真实 tensor + 端到端）
├── scripts/                     环境与工具脚本（不含运行时依赖）
│   ├── collect_env.py           环境盘点 → env_<node>.json
│   ├── model_pp_fit.py          模型切分可行性（每层权重/两端固定开销/KV）
│   ├── model_sizing_table.py    显存/KV 总表
│   ├── hf_mirror_download.py    经镜像下载模型
│   └── wsl_*.sh                 在 WSL 里跑的自检/测试/流水线脚本（纯 ASCII）
└── docs/
    ├── research/                调研与设计（landscape / primer / L2-L3 复用 / vLLM 边界 / PAIR 分析）
    ├── plan/                    计划与路线图（current-work-and-roadmap 等）
    └── deploy/                  部署：环境配置、四机整合计划、实测分析
```

---

## 5. 怎么跑（四种方式，按依赖递增）

```bash
# ① 无硬件：mock 集群
PYTHONPATH=. python -m edge_llm_scheduler.cli --nodes 3 --requests 3
PYTHONPATH=. python -m edge_llm_scheduler.experiments.run_layered_simulation --nodes 3

# ② 数值正确性：分段执行 vs 整体执行（需要 torch + 本地模型）
python -m edge_llm_scheduler.experiments.verify_equivalence \
  --model .models/Qwen2.5-0.5B-Instruct --stages 4 --max-tokens 8

# ③ 单机多进程真实流水线（这条就是"跨机"的同构版本，只是走 loopback）
python -m edge_llm_scheduler.experiments.run_real_pipeline --local 4 \
  --device cuda:0 --model .models/Qwen2.5-0.5B-Instruct --max-tokens 8 \
  --concurrency 1,2,4 --leave-node n3 --metrics-out metrics.json

# ④ 四机分布式：每台起 agent，控制端给 cluster.json
python -m edge_llm_scheduler.agents.stage_agent --node-id alpha --port 9100 \
  --host 0.0.0.0 --model .models/Qwen2.5-0.5B-Instruct --device cuda:0
python -m edge_llm_scheduler.experiments.run_real_pipeline \
  --cluster edge_llm_scheduler/deploy/cluster.example.json --max-tokens 16
```

**测试**：

```bash
python -m pytest edge_llm_scheduler/tests/ -q      # 期望 86+ passed；skip 是可选集成未装
```

---

## 6. 构建与依赖

- **没有编译步骤**：纯 Python。`pip install` 完即可运行。
- **必需**：Python ≥3.10（本机 3.12）、`torch`（GPU 需 cu 版）、`transformers`、`accelerate`、`safetensors`、`numpy`。
- **测试需要**：`pytest`、`pytest-asyncio`；`aiohttp`（`VLLMEngine` 的可选依赖，缺了对应 4 个测试会失败）。
- **可选**：`vllm`（基线轨道）、`lmcache`（真实 KV 后端，需 Linux/WSL 且要编译 C++/CUDA 扩展）。
- 环境配置的完整步骤见 [`environment-setup.md`](environment-setup.md)。

---

## 7. 关键数据契约（改代码前先看这几个类型）

| 类型 | 位置 | 作用 |
|---|---|---|
| `StageAssignment` | `core/types.py` | 一个节点在一个 epoch 里负责的层区间 |
| `PipelinePlan` | `core/types.py` | 一次完整的层放置计划（= 一个 epoch），保证连续完整覆盖 |
| `ActivationEnvelope` | `core/types.py` | stage 间传递 hidden states 的协议：request/epoch/stage/层区间/源目标/payload/校验和 |
| `Task` / `GenerationResult` | `core/types.py` | 下发到某节点的子任务 / 执行结果（含 activation 与 layer-local KV） |
| `KVBlock` | `core/types.py` | KV 最小管理单位，**带 `layer_range`**，所以只能按层部分迁移 |
| `NodeCapability` / `NodeState` | `core/types.py` | 静态能力（算力/显存/带宽）+ 动态状态（剩余显存/负载/已载层区间/KV 块数） |

> **agent 的线协议**（`agents/stage_agent.py`）：`[4B 头长][头 JSON][4B 载荷长][载荷]`，
> 头里是 `{"op": "...", ...}`；载荷是序列化后的 hidden states（**原始 fp16 字节，非 base64**）。
> 操作：`hello / prepare_epoch / activate_epoch / retire_epoch / prefill / decode / status / release_request / shutdown`。

---

## 8. 外部参考实现（不在运行时依赖里，仅作对照与复用来源）

| 目录 | 上游 | 版本 | 在本项目里的角色 |
|---|---|---|---|
| `vllm/` | github.com/vllm-project/vllm | **源码快照（无 `.git`）** | 读源码评估"可复用到什么程度"（见 `docs/research/l2-l3-customization-and-reuse.md`）；运行时只用其 HTTP 接口 |
| `LMCache/` | github.com/LMCache/LMCache | **源码快照（无 `.git`）** | KV 分层/压缩/迁移的复用评估；`backends/lmcache_storage.py` 是其适配层 |
| `moe-infinity/` | github.com/TorchMoE/MoE-Infinity @ `fbb433a` | clone | 专家 offload/预取的设计参考（不跟踪，见 .gitignore） |
| `preble/` | github.com/WukLab/preble @ `1a35eae` | clone | 前缀缓存感知调度的设计参考（不跟踪） |
| `spotserve/` | github.com/Hsword/SpotServe @ `832b29f` | clone | 动态再并行化 / 参数与 KV 迁移的设计参考（不跟踪） |
| （独立仓库） | github.com/NVIDIA/Personal-AI-Router @ `13b6811` | clone | 生产级"请求级路由/信任/成员管理"对照，分析见 `docs/research/` |

> ⚠️ **仓库卫生现状**：`vllm/`（6492 个文件）与 `LMCache/`（972 个文件）目前**被跟踪**，
> 占本仓库文件数的 99%。它们是无 `.git` 的第三方源码快照，建议后续改为**不入库**、
> 由各人按上表自行获取（`git rm -r --cached vllm LMCache` 并加入 `.gitignore`）。
> 另外三个参考实现已经是 ignore 状态。

---

## 9. 仓库约定

1. **不入库**：模型权重（`.models/`、`*.safetensors`）、HF 缓存（`.hf/`）、
   **SSH 私钥（`.ssh/`）**、虚拟环境、`__pycache__`、运行产物（见 `.gitignore`）。
2. **所有新建张量必须显式 `device=`**。CPU 上跑通不代表 GPU 上能跑——已因此踩过两个
   `Expected all tensors to be on the same device` 类 bug。
3. **远程/自动化脚本保持纯 ASCII**（含路径），中文路径用 `find` 解析后软链成 ASCII 名。
4. **每台节点只加载自己那几层**：`prepare_epoch` 的语义就是"只 materialize 本段权重"，
   不要为了省事把整模型读进每台机器。
5. **测试口径**：`skip ≠ pass`。新加的可选集成要显式标注它为何 skip。
6. **提交信息**：说明"为什么"而不只是"改了什么"；涉及论文数字的改动要注明数据来源文件。

---

## 10. 文档地图

| 想知道什么 | 看哪份 |
|---|---|
| 怎么在自己机器上配环境 | `docs/deploy/environment-setup.md` |
| 四台机器怎么整合、切层怎么定、分阶段计划 | `docs/deploy/four-laptop-integration-plan.md` |
| 实测数据与能力边界（PP vs TP、模型选型） | `docs/deploy/edge-4gpu-deployment-analysis.md` |
| 大模型推理优化有哪些路、我们该走哪条 | `docs/research/inference-optimization-landscape.md` |
| 算子/注意力/显存的概念与成熟方向现状 | `docs/research/kernels-attention-and-memory-primer.md` |
| vLLM/LMCache 能复用多少、必须 fork 什么 | `docs/research/l2-l3-customization-and-reuse.md` |
| 当前进度与后续路线 | `docs/plan/current-work-and-roadmap.md` |
