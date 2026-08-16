# MoE 方向文献整理：专家并行（EP）与"专家+KV 联合路由"

> 日期：2026-08-09
> 依据：新下载 13 篇 MoE 相关论文全文 sidecar（`refs/papers/parsed/`）+ 已有 deep-notes
> 起因：SMoE（ISCA'26）细读后，为回答两个问题而系统检索——
> ① **EP（专家并行）**：设备不只持有层切分，还能持有部分专家，已有工作有哪些？
> ② **联合路由**：把"专家选择"与"KV 命中"同时作为请求路由目标，已有工作有哪些？

---

## 一、先说结论（速览）

**问题① EP（设备持有部分专家）——已有非常成熟的工作线：**
- 奠基（训练侧，把专家当作独立切分单位）：GShard、Switch Transformer、DeepSpeed-MoE、Megatron 的 EP 支持
- 推理侧把 EP 落到系统：MoE-Lightning、MoE-Infinity、ExpertFlow、HybriMoE、D²MoE、EC2MoE、SMoE
- 并行映射形式化：MoE Parallel Folding（Megatron-Core 异构并行映射）
- 我们的定位：以上大多是**数据中心单机/同构**或**单设备内 CPU-GPU**；**"边缘多节点、异构、每节点持有部分专家 + 部分层"的 EP 组合是空白**

**问题② 联合路由——已有最接近的工作是 Preble：**
- Preble（ICLR'25）：分布式调度里同时优化 **KV 前缀复用**与**计算负载均衡**，用 E2（Exploitation+Exploration）算法做"命中 vs 探索"决策——**但它管的是 KV，不管专家**。
- 真正把"专家选择"和"KV 命中"同时纳入路由的**尚无成熟系统**——这正是我们的研究机会点。

---

## 二、EP 相关文献（设备持有部分专家）

### 2.1 奠基：专家=可切分单位（训练侧）

| 论文 | 出处 | 核心思想 | 对我们 D1 的启示 |
|---|---|---|---|
| **GShard** | ICLR'21 | 首个提出 **expert parallelism（EP）**：FFN 层替换为 gated experts，每个 worker 持有一个专家，其余组件复制；top-2 gating + capacity factor + 辅助负载均衡损失；全 all-to-all 通信 | "专家"第一次成为独立切分轴，设备不需要持有整层，只持有其分配的专家 |
| **Switch Transformer** | JMLR'22 | 简化到 **top-1 gating**，降低路由开销，扩展到万亿参数；7× 相对 T5 dense 加速 | 路由成本极低是 EP 可落地的关键前提 |
| **DeepSpeed-MoE** | arXiv'22 | **Pyramid-Residual MoE**：MoE 层放深、共享专家 + 路由专家；推理侧用 token 分组/路由 + 动态节点并行 | 共享专家（全 token 必经）+ 路由专家（稀疏激活）的结构，是 SMoE/EC2MoE 等边缘工作的基础假设 |

### 2.2 并行映射形式化

| 论文 | 出处 | 核心思想 | 启示 |
|---|---|---|---|
| **MoE Parallel Folding** | arXiv'25 (Megatron-Core) | 把 EP 和 TP/PP/DP **解耦折叠**，支持异构并行映射（不同维度的专家并行度可不同） | 4D 并行坐标 (D,P,M,EP) 的工程实现；为"异构设备 EP 度不同"提供形式化基础 |

### 2.3 推理侧 EP/专家卸载系统（内存受限）

| 论文 | 出处 | 场景 | 核心机制 | 与我们的关系 |
|---|---|---|---|---|
| **MoE-Infinity** | arXiv'24 | 单机，显存不足 | 请求级 **Expert Activation Matrix (EAM)** 追踪激活模式 → **激活感知预取**（SSD→DRAM→GPU 多级）+ 激活感知缓存（按激活率淘汰，非 LRU） | 2-20× 延迟下降；EAM 是"预测哪些专家会被激活"的工具，可复用于分布式预取 |
| **MoE-Lightning** | ASPLOS'25 | 单机多 T4，显存不足 | **CGOPipe**：CPU-GPU-I/O 三级流水 + **paged weights**（权重分页）；**HRM 层次 Roofline 模型**找最优并行配置/batch | 单 T4 上 Mixtral 8x7B 吞吐比 FlexGen/DeepSpeed 高 10.3×；"权重也分页"的灵感来自 PagedAttention——**权重和 KV 都用分页管理** |
| **HybriMoE** | DAC'25 | 单机，CPU-GPU 混合 | 动态层内调度（CPU/GPU 权衡）+ 影响驱动跨层预取 + **分数感知缓存 MRS**（比 LRU 命中率高） | 与 SMoE 同属"分数感知缓存"线；1.70× decode 加速 |
| **D²MoE** | MobiCom'25 | 边缘设备（3060/Jetson） | **双路由**：token 自适应位宽选择（INT2/3/4）+ **MWQ Matryoshka 量化** + HEBF 热专家优先调度 | 端侧极致；精度-内存-延迟三方权衡的另一维度（量化） |
| **EC2MoE** | arXiv'25 | **端云协作**（跨设备） | **HL-GGN** 硬件感知组路由（端上过滤专家候选）+ **PO-ECC** 端云流水协作 + 低秩压缩传输 | **最接近我们 D6**：跨设备专家调度 + 网络感知，吞吐 2.2-5.1× |
| **ExpertFlow** | arXiv'24 | 单机，显存不足 | 预测式专家缓存 + token 调度 | 与 MoE-Infinity 同线，补充 |
| **SMoE** | ISCA'26 | 单机，显存不足 | **专家替换**：低分激活专家 → GPU 中相似分未激活专家；分数感知淘汰 + 保护盾 + top-score 预取 + CPU 辅助调度 | 细读笔记见 [10-smoe.md](deep-notes/10-smoe.md)；"对象（KV/专家）不重要，命中逻辑重要"的核心启发 |

### 2.4 EP 通信层面（工程，非论文）

- **DeepEP**（DeepSeek 开源通信库，GitHub 仓库无正式论文）：MoE 训练/推理的 EP 专用通信，dispatch/combine 的 all-to-all 优化到亚 200μs。
- **TensorRT-LLM 的 EP/TP 混合**（NVIDIA 文档）：`--moe_tp_size × --moe_ep_size = tp_size`，每 GPU 持专家子集且权重再分片。**这是"设备持有部分专家"在生产系统中的落地证据。**

---

## 三、"专家选择 + KV 命中"联合路由相关文献

### 3.1 最接近：Preble（ICLR'25）—— KV 命中×负载均衡

**问题**：长 prompt 场景下 85-97% token 在请求间共享。单 GPU 前缀缓存（SGLang/vLLM）无法解决分布式问题：把共享前缀请求都送同一 GPU 会过载，送不同 GPU 则重复计算。

**E2 调度算法（Exploitation + Exploration）**：
```
对每个请求：
  前缀匹配（radix tree，含哪个 GPU 缓存了哪些前缀 + 请求频率）
  若 共享前缀 token 数 > 剩余非共享 token 数：
      → EXPLOIT：送"缓存最长匹配前缀"的 GPU（命中，省 prefill）
  否则：
      → EXPLORE：送"prompt-aware 负载最轻"的 GPU
         负载 = 窗口内 GPU 计算负载
              + KV 驱逐代价（权重为复用命中率）
              + 新请求代价（只算未命中部分）
```
配套机制：
- **前缀自动缩放（autoscaling）**：热门前缀超出单 GPU 承载 → 复制到多 GPU，子树分裂
- **负载转移（load shifting）**：过载 GPU 的请求改道
- **prefill/decode 混合**：命中请求 = decode 型，未命中 = prefill 型，混搭到 GPU 上平衡两阶段计算
- **公平性**：按前缀命中率给等待请求分配优先级配额

**结果**：相对 SGLang 平均延迟 1.5×-14.5×、p99 2×-10×。**开源：github.com/WukLab/preble**（构建在 vLLM + SGLang 之上，standalone 调度层）。

**与我们的关系（最重要的一篇）**：Preble 证明"缓存感知路由"在分布式系统里不仅是 KV 管理问题，而是**调度算法**问题，且需要 `命中收益 vs 负载均衡` 的显式权衡（E2）。我们的机会是把它从"只对 KV"扩展到"**KV + 专家双对象**"——即请求路由时同时考虑目标设备的前缀 KV 命中 和 该设备持有的专家子集是否覆盖本请求所需专家。

### 3.2 部分相关：KV 复用视角的分布式路由

| 论文 | 思路 | 与"联合路由"的关系 |
|---|---|---|
| **Mooncake**（FAST'25，已有笔记） | KV-centric 调度：最大化缓存命中，请求路由到持有对应 KV 的 prefill 节点 | KV 命中作为路由唯一目标；无专家维度 |
| **Helix**（ASPLOS'25，已有笔记） | KV 水位线 mask + IWRR 路由 | 用 KV 水位约束调度，非命中驱动 |
| **SGLang / vLLM**（已有） | 单机前缀缓存；SGLang 全局调度器有 cache-aware 路由扩展 | 分布式前缀复用的基础层 |
| **LoongServe**（SOSP'24，已有笔记） | multi-master + 分布式 KV pool，路由考虑 KV 位置 | KV 位置感知路由 |

### 3.3 专家选择的"路由"视角（与 KV 无关但同构）

- SMoE 的 Expert-Cache Router：按分数替换/选择专家——**是"专家选择"决策，但只在单设备缓存内**。
- EC2MoE 的 HL-GGN：端云协作下本地过滤专家候选 + 全局 gating 融合——**跨设备的专家选择**，但主要目标是最小化传输/计算，未与 KV 命中耦合。
- DeepSeek V3 的 auxiliary-loss-free 路由：训练侧偏置更新平衡专家负载——负载均衡思想可迁移到推理时"设备级专家负载"。

---

## 四、研究机会点（结合我们的 D1-D6）

**空白 1：KV + 专家双对象联合路由**
Preble 解决了"KV 命中 vs 负载均衡"，SMoE 解决了"专家缓存命中"，但**没有一个系统把两者同时作为请求路由目标**。我们的系统恰好两个都管（KV 层用 LMCache，专家层可用类似 SMoE 的替换机制），E2 算法的"exploit/explore"框架可以直接扩展成 `exploit_KV + exploit_expert + explore` 三维决策。

**空白 2：边缘多节点的专家并行（EP）**
EC2MoE 是最接近的（端云协作专家调度），但它假设端云链路可用、专家按需拉取。我们的场景更极端：**节点可能持有部分专家 + 部分层 + 部分 KV，且节点会离开**——"专家子集的迁移/复制"作为 D4 的装载卸载对象，目前无成熟工作。

**空白 3：专家热点的"事前冗余复制"**
SMoE 的"分数感知淘汰"是事后缓存管理；结合我们之前提出的"事前冗余复制"（把热点 KV/专家提前复制到备份节点），可以做一个"专家 + KV 双热点复制"策略，应对边缘 churn。

---

## 五、下载清单与状态

### 5.1 本次新增 13 篇（PDF + sidecar 均入库）

| 文件（`refs/papers/`） | arXiv ID | 顶会/出处 | 验证 |
|---|---|---|---|
| 2021-iclr-gshard-conditional-computation.pdf | 2006.16668 | ICLR'21 | ✅ |
| 2022-deepspeed-moe-inference-training.pdf | 2201.05596 | arXiv | ✅ |
| 2022-jmlr-switch-transformers.pdf | 2101.03961 | JMLR'22 | ✅ |
| 2024-expertflow-predictive-expert-caching.pdf | 2410.17954 | arXiv | ✅ |
| 2024-moe-infinity-activation-aware-expert-offloading.pdf | 2401.14361 | arXiv | ✅ |
| 2024-promoe-proactive-caching.pdf | 2410.22134 | arXiv | ✅ |
| 2025-asplos-moe-lightning-memory-constrained-gpu.pdf | 2411.11217 | ASPLOS'25 | ✅ |
| 2025-ec2moe-end-cloud-pipeline-collaboration.pdf | 2508.06024 | arXiv | ✅ |
| 2025-hybrimoe-hybrid-cpu-gpu-scheduling.pdf | 2504.05897 | DAC'25 | ✅ |
| 2025-iclr-preble-distributed-prompt-scheduling.pdf | 2407.00023 | ICLR'25 | ✅ |
| 2025-mobicom-d2moe-dual-routing.pdf | 2504.15299 | MobiCom'25 | ✅ |
| 2025-moe-parallel-folding-heterogeneous-parallelism-mappings.pdf | 2504.14960 | arXiv (Megatron) | ✅ |
| 2026-isca-smoe-expert-substitution.pdf | 2508.18983 | ISCA'26 | ✅ |

> 验证方法：PDF `%PDF` 魔数 + PyMuPDF 首页标题词命中比对（research-assistant `pdf_parser.py` / `verify` 流程）。全部通过。
> 注：xFT（NVIDIA TP+EP 融合推理）在 arXiv 搜索解析不到，无可靠 ID，**未下载**；DeepEP 为 GitHub 仓库（无论文），仅记入 §2.4。

### 5.2 之前论文池（57 篇）不变，其中已含 MoE 相关：Alpa（GShard MoE 训练）、Metis（MoE 测试）、Galvatron（训练）。
