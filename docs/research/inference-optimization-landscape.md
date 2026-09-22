# 大模型推理优化：路径全景与我们的可选路线

> 用途：把"大模型推理优化"这个大方向拆成可比较、可选择的路径，并回答两个问题：
> **① 一共有哪些路可以走？② 以我们四张消费级笔记本 GPU 的条件，哪几条走得通、走得值？**
>
> 与仓库既有文档的关系：`research-alignment.md` 把研究问题拆成 D1–D6 并给出 4 个 gap，
> 本文是它的**上游地图**——D1–D6 只是本文第 4 层（集群与调度层）的切面；
> `framework-design.md` 是机制实现，`current-work-and-roadmap.md` 是执行计划。
> 边界：本文标注的"可行性/拥挤度"是基于我们的资源条件（4 张消费级卡、WiFi/有线局域网、
> 无数据中心宽限期、单学期级周期）的判断，不是对研究价值的绝对评价。

---

## 0. 结论速览

1. **路径分五层**（模型算法 / 算子内核 / 引擎执行 / 集群调度 / 基础设施）。
   我们现在做的是**第 4 层**，PAIR 只是第 4 层里"请求级路由"一个格子。
2. **越往下（模型、算子）越拥挤**：量化格式、attention kernel、投机解码工程、
   PagedAttention 复刻已经是红海，且需要大团队/大算力才能做出差异化。
3. **第 4 层是唯一"我们的硬件反而变成优势"的层**：消费级卡 + WiFi + 笔记本会真实
   制造"装不下、带宽小、节点随时走"这三类约束，而这正是数据中心论文回避的。
4. **但第 4 层内部也有红海**：请求路由/负载均衡（PAIR 那种）已经产品化；
   真正开放的是**弹性（离开/加入）× 异构能力评估 × 带宽受限切分 × 状态迁移**的交叉地带。
5. **推荐 3 条主线 + 3 条副线**（见 §5），全部能在四张卡上做出真实实验，
   且与我们已有代码（KV 目录、epoch 重配置、迁移优先级、token 恢复、链路测量）直接衔接。

---

## 1. 五层地图

```
L5 基础设施与硬件   互联/存储/池化/功耗/能效           ← 我们没有改动权，只能"感知"
L4 集群与调度  ★★★  并行策略·放置·路由·弹性·容错·阶段分离  ← 【我们的位置】
L3 引擎与执行       批处理·KV 管理·请求调度·单实例并行度
L2 算子与内核       attention·GEMM·融合·显存复用·通信算子
L1 模型与算法       量化·压缩·KV 压缩·稀疏·解码算法·适配器
      ↑ 越往上（L4/L5）越偏"系统"，越往下（L1/L2）越偏"数学与内核"
```

每层都能独立成一篇论文，但**真正的机会在跨层**：L1 的量化会改变 L4 的最优切层点，
L2 的 kernel 效率会改变 L3 的批处理策略，L5 的链路会否决 L4 的 TP 选项。
我们已实测到一条：**PP 不降低单请求延迟**（`edge-4gpu-deployment-analysis.md` §3.5），
这就是 L4 的结论被 L2/L5 约束决定的例子。

---

## 2. 路径清单（38 条）

标注含义：**拥挤度** 红=高度竞争 · 黄=有空间但需工程积累 · 蓝=开放；
**我们的可行性** ●=四卡能做出真实验证 · ○=能仿真/部分验证 · ✗=我们的条件做不动。

### L1 模型与算法层

| # | 路径 | 优化目标 | 代表工作 | 拥挤度 | 我们 |
|---|---|---|---|---|---|
| 1 | 训练后量化（INT8/INT4/FP8、逐层混合位宽） | 显存/吞吐 | GPTQ (2210.17323)、AWQ (2306.00978)、SmoothQuant (2211.10438) | 红 | ●（当**约束**用，不当主贡献） |
| 2 | 量化感知训练 / 低比特微调 | 精度-显存 | QAT、QLoRA、LoRA 适配器 | 红 | ○ |
| 3 | 结构化剪枝 / 稀疏 / 低秩分解 | 参数/算力 | SparseGPT、蒸馏 | 红 | ✗（缺训练算力） |
| 4 | **KV 压缩与驱逐** | 显存/长上下文 | H2O (2306.14048)、StreamingLLM (2309.17453)、KIVI (2402.02750) | 黄 | ● |
| 5 | **低秩 KV 表示**（MLA 类） | 显存/带宽 | DeepSeek MLA (2405.04434) | 黄 | ○（需改模型） |
| 6 | 投机解码（draft/Medusa/EAGLE） | 延迟 | Leviathan (2211.17192)、Medusa (2401.10774)、EAGLE (2401.15077) | 红 | ●（当**加速手段**用） |
| 7 | 早退 / 层跳过 / 自适应计算 | 延迟-精度 | 各类 self-speculative | 黄 | ● |
| 8 | 约束解码 / 结构化输出 | 可用性 | JSON-schema、grammar 解码 | 黄 | ●（工程性） |
| 9 | MoE 架构稀疏化（专家数、激活数、共享专家） | 算力/显存 | Switch/GShard、DeepSeekMoE | 红（架构侧） | ○ |
| 10 | **专家替换/命中优化**（用便宜专家替代未命中专家） | 命中率 | SMoE (2508.18983) | 蓝 | ● |

### L2 算子与内核层

| # | 路径 | 优化目标 | 代表工作 | 拥挤度 | 我们 |
|---|---|---|---|---|---|
| 11 | FlashAttention / FlashDecoding 类 kernel | 延迟/显存 | FlashAttention (2205.14135) | 红 | ✗（直接用，不重写） |
| 12 | PagedAttention / 变长批 kernel | 显存/吞吐 | vLLM (2309.06180) | 红 | ✗ |
| 13 | 图编译与算子融合（CUDA Graph、torch.compile、TensorRT-LLM） | 延迟 | — | 红 | ○ |
| 14 | 显存池化 / host offload / 统一内存 | 容量 | ZeRO-Inference、LMCache | 黄 | ● |
| 15 | MoE 算子（grouped GEMM、all-to-all 融合） | 吞吐 | Megablocks、DeepEP | 红 | ✗ |
| 16 | **通信算子与压缩**（all-reduce/all-to-all、量化通信、拓扑感知） | 通信开销 | Ring Attention (2310.01889)、NCCL/DeepEP | 黄 | ●（在 WiFi 上做"通信压缩 vs 精度"） |

### L3 引擎与执行层（单实例）

| # | 路径 | 优化目标 | 代表工作 | 拥挤度 | 我们 |
|---|---|---|---|---|---|
| 17 | 连续批处理 / 迭代级调度 | 吞吐 | Orca (OSDI'22)、vLLM、Sarathi-Serve (2403.02310) | 红 | ●（**必须补上**，我们已实测气泡未消完） |
| 18 | chunked prefill / PD 内部混合 | TTFT/吞吐 | Sarathi-Serve、DistServe | 黄 | ● |
| 19 | **前缀缓存 / 缓存感知** | TTFT/成本 | SGLang RadixAttention (2312.07104)、Preble (2407.00023) | 黄 | ●（已有 `E2Placement`） |
| 20 | KV 分层存储（GPU→CPU→盘→远端） | 容量/成本 | Mooncake (2407.00079)、LMCache | 黄 | ● |
| 21 | SLO 感知调度 / 抢占 / 准入控制 | 尾延迟 | FastServe (NSDI'26)、Mooncake 早期拒绝 | 黄 | ● |
| 22 | 单实例并行度配置（TP/PP/SP/CP） | 延迟 | Megatron-LM (1909.08053) | 红（配置侧） | ○ |
| 23 | 多模型/多 LoRA 混布复用 | 成本 | MuxServe、dLoRA | 黄 | ● |
| 24 | 自适应 batch / 并行度（弹性） | goodput | Pollux (OSDI'21) | 黄 | ● |

### L4 集群与调度层 ★ 我们的位置

| # | 路径 | 优化目标 | 代表工作 | 拥挤度 | 我们 |
|---|---|---|---|---|---|
| 25 | 并行策略自动搜索与成本模型 | goodput | Alpa (2201.12023)、Galvatron、Metis (ATC'24) | 黄（数据中心侧已卷） | ●（**换成异构边缘约束**就是新的） |
| 26 | **跨节点切层/放置（层区间分配）** | 容量/均衡 | Helix (2406.01566)、EdgeShard (2405.14371) | 蓝（异构+弱网） | ●● |
| 27 | 请求路由 / KV 感知路由 | 吞吐/命中 | Preble (2407.00023)、llm-d、AIBrix | 黄（PAIR 已产品化） | ●（当 baseline） |
| 28 | PD 分离与资源配比 | goodput | DistServe (2401.09670)、Splitwise (2311.18677) | 黄（生产化中） | ● |
| 29 | **弹性：动态再并行化（离开/加入）** | 可用性/goodput | SpotServe (2311.15566)、LoongServe (2404.09526) | 蓝（**弱网+异构+无宽限期**） | ●● |
| 30 | **容错：无宽限期抢占 + token 级恢复** | 可用性 | SpotServe、Oobleck (SOSP'23)、Bamboo (NSDI'22) | 蓝 | ●● |
| 31 | **KV 跨节点迁移/复用**（策略与预算） | 恢复时间 | LMCache、Mooncake、LoongServe | 蓝（策略层开放） | ●● |
| 32 | **MoE 专家放置/负载均衡/分级 offload** | 吞吐/显存 | MoE-Infinity (2401.14361)、MoE-Lightning (2411.11217)、EPLB (2412.19437) | 黄偏蓝（边缘侧蓝） | ●● |
| 33 | **在线能力评估（运行中测新节点）** | 调度质量 | Sia/Metis 都是离线 profile → **在线是空白** | 蓝 | ●● |
| 34 | 链路/拓扑感知调度（带宽当约束） | goodput | Helix、Janus (INFOCOM'25) | 蓝 | ●● |
| 35 | 去中心化调度（无中心、RL/gossip） | 可扩展 | 去中心化卸载 (TC'24) | 蓝 | ●（**明确排除激励类**） |
| 36 | 多租户/SLO/成本（spot 实例、混布） | 成本 | Gandiva、Tiresias、Gavel | 红 | ○ |
| 37 | 仿真 / trace 驱动 / 数字孪生 | 方法学 | 各类 simulator | 黄 | ●（已有 simulation 骨架） |
| 38 | 生态接入（vLLM/SGLang connector、K8s） | 落地 | LMCache connector、llm-d、Dynamo | 黄 | ●（工程价值高、论文价值低） |

### L5 基础设施与硬件层

| # | 路径 | 我们能做的部分 |
|---|---|---|
| 39 | 互联与拓扑（NVLink/RDMA/以太/WiFi） | 我们**消费级卡没有 NVLink**，这本身是研究前提；可做"链路质量 → 策略"的映射 |
| 40 | 权重加载与分发（mmap、流式、P2P 分发） | ● 实测装载 2–8s 是重配置主要成本，**优化它有直接收益** |
| 41 | 资源隔离/池化（MIG、共享） | ✗（消费级卡不支持 MIG） |
| 42 | 功耗/热/能效调度 | ● 笔记本会降频、会过热——**能效-延迟权衡是边缘独有题目** |
| 43 | 隐私/本地化推理（TEE、联邦） | ○（是动机，不是我们的贡献点） |

---

## 3. 怎么选：三个判据

选路径不要按"哪个听起来新"选，按这三个判据过一遍：

**判据一：优化目标是什么？**（延迟 / 吞吐 / 成本 / 容量 / 鲁棒性 / 能效）
不同目标的机制完全不通用。我们已实测：**PP 改容量与吞吐，不改单请求延迟**；
想让单请求变快只能走 L1/L2（量化、kernel、投机解码）或 TP（而 TP 在 WiFi 上被否决）。
**目标选错，后面全白做。**

**判据二：约束是否"真实存在"？**（模型装不下 / 显存不够 / 链路慢 / 节点会走）
我们的四张卡能真实制造这四条约束，这是相比数据中心论文的**唯一结构性优势**。
凡是不需要这四条约束就能做的题（例如"更好的路由算法"在仿真里比数值），
我们没有优势。

**判据三：能不能做出差异化实验？**
问自己：这个题在**一台 A100 服务器**上做出来的结论，和在我们四台笔记本上做出来的，
会一样吗？如果一样，说明硬件不是变量，我们的实验结果没有不可替代性。

> 用这三条筛完，§2 的 43 条里只剩十几条；再叠加"与已有代码的衔接成本"，
> 就是 §5 的主线/副线。

---

## 4. 我们已经踩过的坑（避免重复走）

| 坑 | 实测证据 | 结论 |
|---|---|---|
| 以为 PP 能加速单请求 | 4 段 147.5ms/token vs 单机 128.5ms/token（慢 15%） | PP 讲**容量/吞吐**，不讲"加速" |
| 以为"平均切层"就行 | 12/6/3/3 切法下最慢段决定吞吐；末段 3 层却比中间 3 层慢 47% | 切层要按"固定成本 + 每层成本"建模 |
| 忽略绑定词表的双份开销 | 首/末段各背一份 544MB(fp32) / 272MB(fp16) | 两端要少分层，或做**词表切分** |
| 以为并发能线性提升吞吐 | K=1→4 只有 5.06→7.82 tok/s（1.55×），`queue_ms` 涨到 357ms | 没有连续批处理，气泡消不掉 |
| 以为重配置开销在协议 | 装载分片 2–8s ≫ 协议 0.7s | 优化点在**权重装载/缓存/量化** |
| 对照组线程数不一致 | 1 节点 24 线程 vs 每段 6 线程，出现"分段更快"假象 | 异构实验必须统一线程/dtype/上下文 |

---

## 5. 推荐路线：3 条主线 + 3 条副线

### 主线 A：**带宽与异构约束下的动态切层（含在线能力评估）**
- **一句话**：新节点加入 / 链路变化时，**在线**测出它的能力，重算层区间并切换。
- **对应 gap**：`research-alignment.md` 的 gap #1（在线评估）+ #3（切分×通信受限联合）。
- **为什么是我们**：数据中心里 GPU 同质、有宽限期，这个题不成立；我们有 4 张不同的卡 + WiFi。
- **已有基础**：`CapabilityReparallelization`、epoch 蓝绿切流、`measure_link.py`、
  §3.4 的"固定成本+每层成本"实测模型。
- **要补**：在线 profile（几十个 token 的试探性请求就够）、cost model 贝叶斯/回归校准、
  切层求解器（把"最慢段最小化"写成一个整数规划）。
- **关键实验**：加入第 5 个节点（或用 CPU 节点冒充）→ 在线评估耗时 vs 调度质量提升；
  链路从千兆降到 WiFi → 最优切分点如何移动。

### 主线 B：**节点离开下的状态迁移与无宽限期恢复**
- **一句话**：节点突然消失时，**哪些状态值得救、救到哪、花多久**。
- **对应 gap**：gap #2（弱网+异构+无宽限期迁移）。
- **已有基础**：`KVStore.move/estimate_move_cost`、`PriorityMigration`（reuse×prefill_time）、
  `TokenRecovery`、实测"重切 2.15s / drain 0.76s"。
- **要补**：**真实** KV 跨机迁移（接 LMCache）、迁移预算的在线决策、恢复后的输出一致性验证
  （分段 KV 对齐是最容易出错的地方）。
- **关键实验**：迁移预算扫描 → 恢复时间/KV 字节/重算成本三维曲线；
  不同迁移策略在**同一段真实 trace** 上的对比。

### 主线 C：**MoE 专家放置与命中率优化（边缘版）**
- **一句话**：4 台合计显存装不下稠密大模型，但装得下 MoE 的**部分专家**；
  命中率决定 all-to-all 通信量，而通信量决定延迟。
- **为什么是我们**：MoE 的 all-to-all 是**逐层串行**的，数据中心靠 NVLink 掩盖，
  我们在 WiFi 上会把它放大成主要矛盾——**这正好是可测量的研究点**。
- **已有基础**：`ModelManager.load_experts`（GPU/CPU 分级）、`NodeState.expert_ids`。
- **要补**：实现一个可跑的 MoE 小模型 stage runtime（Qwen1.5-MoE-A2.7B 级别）、
  专家级 all-to-all 的通信测量、专家预热/替换策略。
- **关键实验**：命中率 vs 每 token 延迟曲线（**这条曲线目前文献里很少在真实 WiFi 上做过**）。

### 副线 D：**PD 分离在异构边缘的资源配比**
P 机（强算力）与 D 机（大内存/多 KV）如何配比、KV 怎么传、和 PP 怎么叠。
与主线 A/B 共用同一套传输与切层机制，增量成本低，出图快。

### 副线 E：**跨层协同：量化 × 切层 × 调度**
量化不只是省显存：它**改变每层耗时和 activation 字节**，因此改变最优切层点与
放置决策。把 L1 的手段当 L4 的输入变量，是典型的"1+1>2"题，且我们已有
成本模型可复用。（文献里"量化只当加速手段"是常态，"量化改变调度决策"较少见。）

### 副线 F：**可靠性工程化的调度**
无宽限期抢占、token 级恢复、f+1 冗余副本（Oobleck 思路）在边缘的实现与权衡；
我们已有 `TokenRecovery` 接口与故障注入能力。

### 明确**不**建议投的
- ✗ 量化格式/kernel/attention 重写（红海，且需要 CUDA 内核团队）
- ✗ 通用请求路由负载均衡（PAIR/llm-d 已产品化，做不出新意）
- ✗ 需要训练算力的方向（剪枝/QAT/蒸馏）
- ✗ TP 类课题（我们没有 NVLink，WiFi 上先天不成立）
- ✗ 激励/区块链类（用户在 `research-alignment.md` 已明确排除）

---

## 6. 六条路线与 D1–D6、4 个 gap 的对照

| 路线 | D1 拆分 | D2 协作 | D3 分配 | D4 离开/加入 | D5 评估 | D6 非数据中心网 | 命中 gap |
|---|---|---|---|---|---|---|---|
| A 动态切层+在线评估 | ●● | ● | ●● | ●● | ●● | ●● | #1 #3 |
| B 状态迁移/无宽限期恢复 | ● | ● | ● | ●● | ○ | ●● | #2 |
| C MoE 专家放置与命中 | ●● | ●● | ●● | ● | ● | ● | （新增：#4 之外） |
| D PD 分离配比 | ●● | ● | ●● | ○ | ● | ● | #3 |
| E 量化×切层×调度协同 | ●● | ● | ●● | ○ | ●● | ● | #1 #3 |
| F 可靠性工程化调度 | ○ | ● | ●● | ●● | ○ | ● | #2 |

（gap #4"训练与推理统一框架"我们不建议碰：状态面差异太大，四卡做不动。）

---

## 7. 下一步（按性价比）

1. **本周**：在四台真机上跑通现有部署手册（GPU + WiFi 数据），补齐 §3 判据二的"约束真实性"证据。
2. **两周内**：接上连续批处理（L3 #17），把并发吞吐从 1.55× 拉起来——这是所有吞吐类实验的地基。
3. **一个月内**：主线 A 的在线评估 + 切层求解器（复用 `measure_link` 与成本模型）。
4. **并行推进**：主线 B 的真实 KV 迁移（接 LMCache），因为它同时是副线 F 的基础。
5. **视人力**：主线 C（MoE）单独开一条线，前置是找到一个能在 4×8GB 上跑起来的 MoE 小模型。

---

## 8. 追问：算子优化、稀疏注意力、KV 专项的归属与可用度

> 起因：数据中心（API 提供商）对推理做了大量深度优化，我们需要知道**哪些能拿来用**、
> **算子优化到底属于哪一层**、**稀疏注意力算不算我们能做的**、以及**推理侧还有哪些 KV 专门优化**。
> 本节把"技术"按 **① 归属层 ② 可用度 ③ 我们能否做** 三列拆开。

### 8.1 先定位：算子优化在哪一层，以及它有三种不同性质

**算子/内核 = L2，比引擎层（L3）更深**：引擎负责"调度哪些请求、怎么组织 KV、什么时候 forward"，
它**调用**算子；算子负责"这一步矩阵乘/注意力怎么在硬件上跑完"。所以是的，
算子优化在**引擎之下**。但"算子优化"其实混了三种性质完全不同的工作：

| 性质 | 定义 | 归属 | 例子 | 我们能否做 |
|---|---|---|---|---|
| **(a) 实现优化** | 数学语义不变，把 kernel 写得更快 | 纯 L2 | FlashAttention-2/3、FlashMLA、DeepGEMM、Marlin | ❌ 红海；需 CUDA 团队；消费卡上 FA2 已够用 |
| **(b) 融合/图优化** | 跨算子合并、消除中间张量 | L2 × 编译（L3） | norm+quant 融合、GEMM+通信重叠、CUDA graph | ✅ **官方留了口子**（见 7.2），但"为融合而融合"没有论文价值 |
| **(c) 机制优化** | 改变注意力的**数学定义**（哪些 KV 参与计算） | **L1 × L2** | 稀疏注意力、滑窗、sink、低秩 KV(MLA) | ⚠️ 取决于是否免训练（见 7.3） |

**关键判断：只有 (c) 里"免训练"的那部分，才是我们能在不改模型、不做训练的前提下碰的。**

### 8.2 我们能在 vLLM 里动算子/编译的**官方口子**（已核实）

| 口子 | 用途 | 位置 |
|---|---|---|
| `@register_backend(AttentionBackendEnum.CUSTOM)` | 注册**自己的 attention 后端**（可实现 `AttentionBackend` 的 4 个 staticmethod + `supports_compute_capability`） | `v1/attention/backends/registry.py:129,242`；接口 `v1/attention/backend.py:55,342` |
| `CustomOp.register` / `PluggableLayer` / `direct_register_custom_op` | Python 级自定义算子（可 out-of-tree 替换） | `model_executor/custom_op.py:32,103,315`；`utils/torch_utils.py:1026` |
| `CompilationConfig.inductor_passes` | **挂自定义 inductor pass**（值可为限定名或 Python 对象） | `config/compilation.py:599-604` → `__post_init__:938-953` |
| 内建融合 pass 的开关 | `fuse_norm_quant` / `fuse_act_quant` / `fuse_attn_quant` / `fuse_gemm_comms` / `fuse_allreduce_rms` / `eliminate_noops` | `compilation/pass_config`（`PassConfig`） |
| CUDA/C++ 扩展 | `csrc/` + CMake（`CMakeLists.txt` 逐内核架构交集） | `setup.py:188,194,358`、`cmake/utils.cmake:397` |
| `vllm/kernels/*` | Triton / Helion 内核目录（学习成本远低于 CUDA） | `vllm/kernels/{triton,helion,...}` |

**我们唯一有理由做的算子工作，是"与系统需求绑定的算子"**（数据中心不做，因为他们的场景不需要）：
跨 stage 的 activation **压缩/打包/直发** kernel、层粒度 **KV 提取/注入** kernel、
弱网下的 **KV 选择/稀疏化** kernel。也就是"**为跨机层粒度执行服务的算子**"。

### 8.3 稀疏注意力：分成两类，只有一类我们做得动

**第一类：训练型（模型自带 sparse indexer）——我们做不了**

vLLM 里已经有一大批稀疏后端，但**全部是"模型驱动"的**：

| 后端 | 说明 |
|---|---|
| `FLASHMLA_SPARSE` / `FLASHINFER_MLA_SPARSE(_SM120)` / `FLASH_ATTN_MLA_SPARSE` / `ROCM_AITER_MLA_SPARSE` / `XPU_MLA_SPARSE` | MLA 系稀疏（DeepSeek 类），需要 **indexer + `topk_tokens`/`topk_indices_buffer`** |
| `FLASHMLA_SPARSE_DSV4` / `FLASHINFER_MLA_SPARSE_DSV4` | 注释逐字："DeepSeek V4 sparse MLA backends (**model-driven**; selected via the V4 layer)" |
| `MINIMAX_M3_SPARSE` / `TRITON_MSA` / `CUTLASS_MSA` | MiniMax M3 的 MSA |
| `DEEPSEEK_SPARSE_SWA` | 稀疏滑窗 |

配套的 `v1/attention/backends/mla/indexer.py` 就是那个"选 top-k token 的索引器"，
且带 `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` 这类预算参数。
**共同点**：稀疏 pattern 或 indexer 是**模型的一部分**（NSA/MoBA 这类原生稀疏注意力是在训练阶段学的），
所以要么用官方权重、要么自己训练——**后者我们没有算力**。
而且这些后端多数要求 SM90+/FP8/CUTLASS，**我们手上的 sm89/sm86 基本不满足**。

**第二类：免训练型（推理时自己决定"看哪些 KV"）——我们做得了**

| 做法 | 机制 | 我们的落点 |
|---|---|---|
| 自定义稀疏 mask | vLLM 的 **FlexAttention 后端**支持 `logical_mask_mod`（**从模型层读**：`getattr(layer, "logical_mask_mod", None)`）+ `block_sparsity_hint`（把**全被 mask 的 KV block 提前剪掉**，docstring 逐字："blocks that are fully masked never get loaded"） | `v1/attention/backends/flex_attention.py:347-359,999-1008,1329-1343`；FlexAttention 走 Triton，**sm86/sm89 可用** |
| 注意：FA2/FA3 **不支持** mask_mod | `flash_attn_interface.py:313-314,345-346`（`NotImplementedError("FA2 does not support mask_mod")`）；**只有 FA4 支持**（CuTE-DSL mask_mod） | → 在我们的卡上，"自定义稀疏 mask"**只能走 FlexAttention/Triton 这条** |
| token 驱逐/选择（H2O / StreamingLLM / SnapKV / PyramidKV / TOVA 类） | **vLLM 里没有实现**：全仓搜 `h2o/heavy hitter/token evict/streamingllm` 只命中 **attention sink 作为模型特性**（gpt-oss），没有推理时驱逐策略 | 这是一个**真空白**，但要自己做；且论文已经很密 |
| prefill 稀疏 pattern（MInference 类） | 同上，需要自己在 mask/kernel 层做 | 可做，但同样拥挤 |

**结论（直接回答你的两个疑问）**：
1. **"稀疏注意力是否涉及训练"——取决于是哪一类**：原生稀疏（NSA/MoBA/DeepSeek-V4 sparse MLA）
   的 pattern 是**训练出来的**，我们只能用别人的权重；而**免训练稀疏**（自己定 mask / 挑 KV）
   不需要训练，**我们能在推理侧做**。
2. **但"再做一个免训练驱逐策略"本身已很难出增量**（2023–2024 已密集）。有增量的角度是**联合**：
   稀疏度 ↔ KV 迁移量 ↔ 切层粒度 ↔ 弱网预算（这正好是数据中心论文不关心、而我们场景必需的交集）。

### 8.4 推理侧针对 KV 的专门优化（清单 + 我们的可用度）

| 方向 | 具体技术 | vLLM 现状 | 我们能否做 |
|---|---|---|---|
| **不增长** | 分页（PagedAttention）、前缀缓存/RadixAttention、跨请求共享 | ✅ 现成（`enable_prefix_caching` 默认开） | ✅ 直接用 |
| **少占空间** | KV 量化（INT8/FP8/INT4、per-channel/per-token）、按层跳过 | ✅ `cache_dtype` + `kv_cache_dtype_skip_layers`（可**按层**跳过） | ✅ 配置级；**按层的取舍策略**可自己定 |
| **更低秩** | MLA / latent KV（把 KV 压到低维潜空间） | ✅ 有 MLA 后端（但多为 SM90+/FP8） | ❌ 需模型侧改造/重训 |
| **跨层复用** | 相邻层共享 KV（Yona 类） | ✅ 机制存在：`kv_sharing_target_layer_name`（某层直接复用另一层张量） | ⚠️ 需模型配合 |
| **丢一部分** | token 驱逐/选择（H2O/StreamingLLM/SnapKV/TOVA）、token 合并 | ❌ **没有**（只有 block 级 LRU 驱逐） | ⚠️ 可自研（见 7.3） |
| **放别处** | 分级存储 GPU→CPU→disk→远端 | ✅ `OffloadConfig`(UVA/prefetch) + `kv_offloading_*` + `SimpleCPUOffloadConnector` + LMCache | ✅ 现成为主 |
| **搬到别处** | PD 分离的 KV 传输、跨实例 KV 复用 | ✅ NIXL push（**producer 侧 PP>1 支持层区间，decode 侧 PP>1 未实现**）、KV connector 的逐层钩子 | ⚠️ 层粒度接口齐了，跨机传输要自己写 |
| **算还是搬** | 迁移 vs 重算的取舍 | ✅ 有一个二选一开关 `kv_load_failure_policy = recompute \| fail`（**无"部分抢救"**） | ✅ **我们的增量点**（按价值 + 按预算） |
| **提前知道** | KV 感知路由 / 前缀命中路由 | ✅ `get_num_new_matched_tokens` 钩子；LMCache 有 `external_lookup_client` 等 | ✅ 可做 |
| **格式/布局** | KV 布局（HND/NHD）、block 对齐、跨模型精度一致性 | ⚠️ connector 可 `get_required_kvcache_layout()` 要求布局；寻址必须 block 对齐 | ⚠️ 跨机搬运的前提，容易踩坑 |
| **长上下文** | 位置编码外推、滑窗+sink、上下文并行（CP/DCP） | ⚠️ 部分在模型侧；vLLM 有 CP/DCP 与 sparse indexer 的组合限制 | ⚠️ 需模型配合 |

### 8.5 数据中心做了、而我们前面没提到的方向（补全清单）

| 方向 | 数据中心的做法 | 我们的可用度（A=现成可用 / B=可做需自研 / C=需训练或大算力 / D=红海不建议） |
|---|---|---|
| **解码算法** | 投机解码：ngram / Medusa / EAGLE / MTP / draft model；并行草稿；拒绝采样 | **A**（vLLM 全都有：`SpeculativeMethod = ngram/ngram_gpu/medusa/draft_model/eagle/eagle3/dflash/dspark/extract_hidden_states/mtp`，含 ~25 个模型专属 MTP 变体、`parallel_drafting`、三种拒绝采样） |
| **批处理形态** | continuous batching、chunked prefill、**micro-batch / ubatching**、双批重叠 | **A**（连续批处理+chunked prefill 现成；`v1/worker/ubatching.py` 是微批基础设施） |
| **调度策略** | SLO 感知、优先级、抢占、准入控制（过载早期拒绝）、公平性 | **B**（`--scheduler-cls` 可挂，但源码声明接口非公开；抢占/调度机制已在树里） |
| **并行/通信** | Wide EP、EPLB、通信-计算重叠（DualPipe 类）、通信量化、拓扑感知放置 | **B/C**（EPLB 现成且通信后端可插拔含 NIXL；重叠与 wide EP 需要多机与高速互联） |
| **精度** | FP8/FP4 权重+激活、微缩放（MX）、逐层混合精度、KV FP8 | **B**（Ada 卡可用 FP8 CUTLASS；FP4/MX 需 SM90+；逐层混合精度可自研） |
| **编译/图** | CUDA graph、可断图（breakable cudagraph）、算子融合、TensorRT 类编译器 | **A/B**（vLLM 有 `cudagraph_mode`、`breakable_cudagraph.py`、pass 体系；自定义 pass 有官方字段） |
| **内存/加载** | 权重流式加载、P2P 权重分发、多级 KV、模型缓存 | **B**（`LoadConfig.safetensors_load_strategy=lazy/eager/prefetch`；P2P 分发需自研） |
| **可靠性** | 故障容忍、弹性伸缩、迁移一致性校验、灰度/回滚 | **B**（vLLM 只有 `engine_recovery_timeout_sec` + DP/EP-only FT；弹性只到 DP 且与 PP 互斥） |
| **能效/热** | 数据中心 PUE、功耗封顶、DVFS | **B（我们独有）**：笔记本会降频/过热，**"能效-延迟权衡"是数据中心论文不会碰的题** |
| **多租户/多模型** | 多模型混布、LoRA adapter 复用、MuxServe 类 | **A/B**（LoRA 体系完整；混布需自研） |
| **可观测性** | metrics/tracing、per-request 归因、成本核算 | **B**（vLLM 有 metrics/`perf.py`；我们要做的是把"每段/每次迁移"记进自己的指标） |
| **方法学** | trace 驱动仿真、消融、可复现实验 | **A/B**（我们已有 simulation 骨架 + 4 卡真实平台） |

### 8.6 一句话回答

- **算子优化属于 L2，确实比引擎层更深**；其中"实现优化"我们不该碰（红海 + 无 CUDA 团队），
  "融合/编译"有官方口子但论文价值低，**唯一值得做的是"为跨机层粒度执行服务的算子"**。
- **稀疏注意力要分两类**：原生稀疏（NSA/MoBA/DeepSeek-V4）是**训练出来的**，我们只能用别人的权重且后端多为 SM90+；
  **免训练稀疏（自己定 mask/挑 KV）我们做得了**——落点是 FlexAttention 的 `mask_mod` + `block_sparsity_hint`（Triton，消费卡可用），
  但单纯再做一个驱逐策略已难有增量，**要与跨机/分级/并行度联合**。
- **KV 专项优化的现成度最高**（前缀缓存、按层跳过量化、分级 offload、逐层迁移钩子、KV 感知路由都现成），
  **真正空白且属于我们的是"算还是搬"的取舍**（vLLM 只有一个二选一开关，没有按价值/按预算的部分迁移）。
- **我们前面漏掉的、且对数据中心是标配的方向**至少还有：投机解码/MTP、微批与双批重叠、
  SLO 与抢占、通信-计算重叠、编译/CUDA graph、权重流式加载、多租户混布、以及**笔记本独有的能效-热约束**。


---

> 记住 §0 的第 3、4 条：**我们的护城河是“约束真实”，不是“算法更巧”。**
> 任何一条路线（包括 §8 里的算子/稀疏/KV 专项），只要能证明
> “在一台数据中心服务器上这个问题不存在或不要紧”，它就值得我们做。
