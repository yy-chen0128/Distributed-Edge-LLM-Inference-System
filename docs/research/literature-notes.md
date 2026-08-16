# 文献笔记：异构分布式算力上的 LLM 训练/推理调度与资源弹性

> 研究主题：大模型在分布式算力基础设施上的训练与推理；异构算力单元（GPU 等）的任务调度；资源装载/卸载（单元可能离开）
> 生成日期：2026-08-06
> 依据：`refs/papers/parsed/*.txt`（28 篇已下载全文 sidecar）+ polylink 站点清单 + CCF-A 检索

---

## 一、主题聚类总览

| 聚类 | 论文 | 细读/总结 |
|---|---|---|
| **A. 异构 GPU 集群任务调度** | Gavel、Sia、Pollux、Gandiva、Tiresias、Themis、AntMan、Metis、MLaaS in the Wild、Alibaba Hyperscale | 细读：Sia、Metis；其余总结 |
| **B. LLM 推理服务与分布式部署** | vLLM、Orca、SGLang、DistServe、FastServe、LoongServe、Splitwise、Mooncake、Helix、SpotServe、dLoRA、AlpaServe、OpenTela、EcoServe | 细读：Helix、DistServe、FastServe、SpotServe、LoongServe、Mooncake；其余总结 |
| **C. 弹性、容错与节点离开** | Bamboo、Oobleck、SpotServe、FastServe | 细读：SpotServe、FastServe；其余总结 |
| **D. 大规模训练并行与自动部署** | Alpa、PipeDream、HetPipe、Galvatron | 细读：Alpa；其余总结 |
| **E. PolyLink 站点相关（边缘/联邦/去中心化）** | EdgeShard、Distributed ML in Edge、PASTA、Janus、PolyEdge、Decentralized Task Offloading、Hybrid Redundancy 等 | 总结（部分付费） |
| **F. MoE 专家并行与边缘 MoE 推理（2026-08 新增）** | GShard、Switch、DeepSpeed-MoE、MoE Parallel Folding、MoE-Infinity、MoE-Lightning、ExpertFlow、HybriMoE、ProMoE、D²MoE、EC2MoE、SMoE、Preble | 见 [moe-literature.md](moe-literature.md) |

> 注：细读 = 深入"分布式算力部署调度细节"的 CCF-A 核心论文，精读全文；总结 = 其余相关论文的核心内容归纳。

---

## 二、细读论文详细笔记（9 篇）

> 每篇一份详细笔记，见 `plan/review/deep-notes/` 目录。以下为速览表 + 跨论文综合洞察。

| # | 论文 | 会议 | 核心机制（一句话） | 笔记 |
|---|---|---|---|---|
| 1 | Helix | ASPLOS'25 | 异构 GPU+网络上的模型放置与请求调度联合建模为 **max-flow/MILP**，per-request pipeline + IWRR | [01-helix.md](deep-notes/01-helix.md) |
| 2 | DistServe | OSDI'24 | **prefill/decode 分离**，给定 TTFT/TPOT 联合优化各阶段资源与并行策略，goodput 目标 | [02-distserve.md](deep-notes/02-distserve.md) |
| 3 | SpotServe | ASPLOS'24 | 可抢占实例上的**动态再并行化** + **二分图最优迁移（KM）** + token 级状态恢复 | [03-spotserve.md](deep-notes/03-spotserve.md) |
| 4 | FastServe | NSDI'26 | **迭代级抢占调度（skip-join MLFQ）** + KV cache 主动 GPU/host 换入换出 | [04-fastserve.md](deep-notes/04-fastserve.md) |
| 5 | Sia | SOSP'23 | **异构+弹性**联合调度，ILP 公式化 + 受限配置集 + 引导启动吞吐建模 | [05-sia.md](deep-notes/05-sia.md) |
| 6 | Metis | ATC'24 | 异构 GPU 自动并行化：**hetero-aware device group + DP-first DFS 搜索** | [06-metis.md](deep-notes/06-metis.md) |
| 7 | LoongServe | SOSP'24 | **弹性序列并行（ESP）**：迭代粒度调并行度，零开销 KV 迁移（主动留存/multi-master） | [07-loongserve.md](deep-notes/07-loongserve.md) |
| 8 | Mooncake | FAST'25 | **KV cache 为中心的分离式架构** + 全局调度 + 预测式早期拒绝 | [08-mooncake.md](deep-notes/08-mooncake.md) |
| 9 | Alpa | OSDI'22 | **自动并行部署规划**：inter-op DP + intra-op ILP 两层级联 | [09-alpa.md](deep-notes/09-alpa.md) |

### 2.1 跨论文综合洞察（对本研究的意义）

**（1）"部署/调度"可分解为三个正交维度**
- **放置（placement）**：模型分片放哪些 GPU → Helix（max-flow）、Metis（device group）、Alpa（mesh 分配）
- **任务/请求调度（scheduling）**：请求/作业何时由谁执行 → FastServe（抢占）、Sia（ILP）、Mooncake（KV-centric）
- **资源弹性（elasticity）**：GPU 数/类型变化、节点离开 → SpotServe（再并行化+迁移）、LoongServe（弹性 DoP）、Bamboo/Oobleck（容错）

**（2）"异构"的两种建模方式**
- 显式算力/带宽建模：Helix 用图边容量同时编码 GPU 算力与网络；Sia/Metis 用 per-(job, GPU_type) 吞吐模型；Alpa 用 mesh 两级带宽层次
- 负载维度异构：DistServe/Splitwise 把推理按计算特征拆成 prefill（compute-bound）与 decode（bandwidth-bound），各自独立缩放

**（3）"节点离开/装载卸载"的三个机制层次**
- 任务级抢占 + KV 换出：FastServe（GPU↔host offload/upload）
- 实例级迁移 + 状态恢复：SpotServe（二分图最优迁移 + token 级状态提交）；LoongServe（零开销主动留存 KV）
- 训练级容错：Bamboo（冗余计算填气泡）、Oobleck（f+1 副本模板）

**（4）关键张力：弹性 vs 分离**
- Mooncake 明确论证为何在长上下文 prefill 上选 CPP/静态分离而非 LoongServe 的弹性 SP（MFU、网络竞争、架构复杂度）——这是本研究定位时要回答的核心取舍问题。

**（5）研究缺口（可作为切入点）**
- 现有工作大多只处理**单一**方面：Helix/Sia 不做节点离开；FastServe/SpotServe 不做异构；Alpa/Metis 是静态规划不做运行时弹性；DistServe 无故障容错（自述 fault propagation 风险）
- 把"异构 + 弹性/节点离开 + LLM 训练/推理"三者在**一个系统**内联合处理，且有跨异构资源的状态（KV/模型分片）迁移机制，是明显的空白

---

## 三、其余论文核心内容总结

### 3.1 CCF-A 推理服务（总结级）

#### vLLM / PagedAttention（SOSP 2023）
- **核心**：KV cache 内存碎片是 LLM 高吞吐服务的瓶颈。PagedAttention 借鉴 OS 虚拟内存分页，把请求的 KV cache 切成非连续固定块，按需分配 → 近零碎片、可跨请求共享（并行采样/beam search）。
- **机制**：块表（block table）+ 分页注意力核；请求级/前缀级共享。
- **亮点**：相比 FasterTransformer/Orca 吞吐提升 2–4×，长序列/大模型更明显。
- **借鉴**：内存管理是分布式部署的基础层；为"分片如何放置在多 GPU"提供了单 GPU 内存的高效基础。

#### Orca（OSDI 2022）
- **核心**：请求级批处理不灵活（队头阻塞、完成不均）。提出**迭代级调度**——每个迭代调度器都可增删请求；**选择性批处理**——attention 逐请求算、其余算子批处理。
- **机制**：FCFS 调度 + 最大 batch 调优 + intra/inter-layer 模型并行扩展到百亿参数。
- **亮点**：vs FasterTransformer 吞吐 36.9×（GPT-3 175B）。
- **借鉴**：迭代级调度是后续 FastServe 抢占、vLLM 连续批处理的基石。

#### SGLang（NeurIPS 2024）
- **核心**：复杂 LLM 程序（agent、RAG、多轮）的执行效率。前端语言 + 运行时；**RadixAttention** 用 radix tree 自动复用前缀 KV 缓存；**压缩有限状态机**加速结构化输出（JSON）解码。
- **亮点**：多任务吞吐最高 6.4×。
- **借鉴**：KV 缓存复用与前缀调度，属于部署侧的缓存调度优化。

#### Splitwise（ISCA 2024）
- **核心**：prefill 是计算密集、decode 是访存密集，把两阶段**分离到不同类型 GPU**（A100/H100 混合），用 RDMA/InfiniBand 传 KV，优化每美元吞吐。
- **亮点**：吞吐 1.4×、每美元吞吐提升约 30–50%。
- **借鉴**：异构硬件各取所长——正是"异构算力分工"的早期代表，为 DistServe/Mooncake 铺路。

#### AlpaServe（OSDI 2023）
- **核心**：模型并行不仅用于放大单模型，也可用于**多模型服务的统计复用**。探索并行化开销与突发负载下延迟收益的权衡，自动决定多个模型的放置与并行化。
- **亮点**：相同延迟约束下请求速率 10× 或突发度 6×。
- **借鉴**：多模型共享集群的放置决策，涉及异构资源分配。

#### dLoRA（OSDI 2024）
- **核心**：多租户 LoRA 服务。**动态 merge/unmerge 适配器**（credit-based batching 决定何时切换）+ **请求与适配器协同迁移**（ILP 最小化负载不均，proactive dispatch + reactive migration）。
- **亮点**：vs vLLM 吞吐 57.9×；vs S-LoRA 平均延迟低 1.8×。
- **借鉴**：跨副本的负载均衡与适配器迁移，是"多模型在分布式集群上的部署调度"的实例。

### 3.2 CCF-A 集群调度（总结级）

#### Gavel（OSDI 2020）
- **核心**：不同模型架构在异构加速器上性能差异大。把调度策略（公平/LAS/SJF/makespan 等）统一表达为**优化问题**（吞吐矩阵 T × 分配矩阵 X），自动异质感知；round-based 机制保证作业获得接近理想分配的份额，支持空间共享。
- **亮点**：平均 JCT 提升 3.5×，makespan 1.4–2.5×。
- **借鉴**：异质感知策略的形式化框架，是本研究"异构单元调度"的经典范式。

#### Pollux（OSDI 2021）
- **核心**：调度与 DL 训练**协同优化**——调度器动态增减 GPU，框架同步调整 batchsize；引入 **goodput**（系统吞吐 × 统计效率）。监控训练状态建模 goodput 随资源变化。
- **亮点**：平均 JCT 降 37–50%，超越理想配置的 SOTA 调度器。
- **借鉴**：弹性伸缩与训练配置联动，为"资源装载/卸载 + 训练自适应"提供范式。

#### Gandiva（OSDI 2018）
- **核心**：利用 DL 训练的可预测性（重复 mini-batch 迭代），**时间片切分 GPU** 多作业共存；监控性能并**动态迁移作业**到更匹配的 GPU（迁移发生在 mini-batch 边界，内存占用最低点）。
- **亮点**：超参数搜索提速 10×，180-GPU 集群利用率 +26%。
- **借鉴**：时间片 + 迁移的"作业级资源重分配"，早期负载/卸载思想。

#### Tiresias（NSDI 2019）
- **核心**：DL 作业执行时间不可预测、all-or-nothing。提出 **Discretized 2D-Gittins index**（有部分信息）与 **Discretized 2D-LAS**（无信息）两类调度算法，同时在空间（GPU 数）与时间（attained service）两维调度。
- **亮点**：平均 JCT 比 YARN 提升 5.5×。
- **借鉴**：面向不可预测作业的调度算法设计。

#### Themis（SOSP 2020）
- **核心**：GPU 集群**公平调度**。两级 semi-optimistic 架构（AGENT + ARBITER），作业通过**拍卖竞标**获得资源，目标 finish-time fairness，减少资源碎片。
- **借鉴**：多用户公平性机制（本研究涉及多租户调度时可用）。

#### AntMan（OSDI 2020）
- **核心**：调度器与 DL 框架**协同设计**，生产部署于阿里。**动态内存缩放**（监控内存、上限回收、溢出到 host）+ **机会计算**（GpuOpManager 在空闲槽启动低优先级核），多作业共享 GPU 且不干扰。
- **亮点**：GPU 内存利用率 +42%，算力 +34%。
- **借鉴**：细粒度资源共享/弹性，资源"装卸"的底层机制。

#### MLaaS in the Wild（NSDI 2022）
- **核心**：阿里 6000+ GPU 生产 MLaaS 集群两个月 trace 的**负载表征**：低 GPU 利用率、长排队、hard-to-schedule 高要求任务、异构机间负载不均、CPU 瓶颈。
- **借鉴**：真实异构集群调度问题清单，验证研究问题的现实依据。

#### Alibaba Heterogeneity at Hyperscale（OSDI 2026）※付费
- **核心**：大规模生产 AI 集群的异构表征与调度（超大规模、多代 GPU 混布）。

### 3.3 CCF-A 弹性 / 容错（总结级）

#### Bamboo（NSDI 2022）
- **核心**：在**可抢占实例**上训练大 DNN 以降低 2.4× 成本。关键洞见：流水线并行天然有**气泡**，把冗余计算塞进气泡（邻居节点重复计算部分层），实例被抢占时无需 checkpoint 即可恢复。
- **亮点**：吞吐比传统 checkpoint 高 3.7×。
- **借鉴**：节点离开/抢占下的**无检查点恢复**——"单元可能离开"场景的核心机制。

#### Oobleck（SOSP 2023）
- **核心**：容错训练。规划-执行协同：生成**异构流水线模板**，实例化 ≥f+1 个逻辑等价副本，容忍任意 f 个并发故障；恢复利用副本已复制的模型状态，且保证故障后资源仍被模板覆盖（不空闲）。
- **亮点**：vs Bamboo/Varuna 最高 29.6×。
- **借鉴**：结构性冗余 + 快速恢复，f 故障保证，节点离开的强保证方案。

### 3.4 CCF-A 训练并行（总结级）

#### PipeDream（SOSP 2019）
- **核心**：流水线并行 DNN 训练，把层分布到多机，前向/后向 round-robin 调度，参数版本化保证正确性，通信减少最多 95%。
- **借鉴**：流水线并行是训练侧部署的基础。

#### HetPipe（ATC 2020）
- **核心**：异构（含"弱鸡"）GPU 上大模型训练。多个 GPU 组成**虚拟 worker** 流水处理，多个虚拟 worker 数据并行；提出 **Wave Synchronous Parallel (WSP)** 同步模型。
- **亮点**：收敛快 49%（VGG-19）。
- **借鉴**：异构能力整合成虚拟计算单元的"资源聚合"思想。

#### Galvatron（VLDB 2023）
- **核心**：transformer 训练的自动异构并行搜索，基于分析型成本模型（融合通信），在 DP/TP/PP 组合空间中高效搜索。

### 3.5 PolyLink 站点相关论文（边缘/联邦/去中心化）

> 以下论文来自 polylink.evan.cafe 实验室成果，除 EdgeShard 外均为付费论文（DOI 见 `plan/retrieval/download-status.json`），核心内容依据标题/摘要/公开信息归纳，建议获取全文后核实。

#### EdgeShard: Efficient LLM Inference via Collaborative Edge Computing（IEEE IoTJ 2024）✅ 已下载全文
- **核心**：把 LLM 推理按层切分，分布到多个边缘节点协作执行；边缘端模型分片调度与通信协调。

#### Distributed Machine Learning in Edge Computing（ACM CSUR 2024）※付费
- **核心**：边缘分布式机器学习综述：节点资源受限、数据异构、隐私安全、并行模式/分布式架构/通信聚合、应用与开放问题。

#### PASTA: Training Acceleration for Vertical Federated Learning via Adaptive Pipeline Parallelism（IEEE Cluster 2025）※付费
- **核心**：垂直联邦学习（VFL）训练加速，自适应流水并行，处理 VFL 各参与方的异构计算能力。

#### Janus: Collaborative Vision Transformer Under Dynamic Network Environment（INFOCOM 2025）※付费
- **核心**：动态网络环境下协同 ViT 推理，适应网络波动/节点变化的协作部署。

#### PolyEdge: A Blockchain-based Decentralized Edge AI Platform（IEEE Blockchain 2025）※付费
- **核心**：区块链去中心化边缘 AI 平台：算力聚合、任务分发、激励/信任机制。

#### Decentralized Task Offloading in Edge Computing: An Offline-to-Online RL Approach（IEEE TC 2024）※付费
- **核心**：去中心化任务卸载，离线预训练 + 在线强化学习调度，不依赖中心控制器。

#### Hybrid Redundancy for Reliable Task Offloading in Collaborative Edge Computing（IEEE TC 2025）※付费
- **核心**：协作边缘计算任务卸载的**混合冗余**机制，在可靠性（冗余）与资源开销间权衡。

#### Mitigating Unfairness in Differentially-private Federated Learning（TOMPECS 2025）※付费
- **核心**：差分隐私联邦学习中的公平性缓解。

#### Federated Class-Incremental Learning with Dynamic Feature Extractor Fusion（IEEE TMC 2024）※付费
- **核心**：联邦类增量学习，动态特征提取器融合，应对客户端异构。

---

## 四、付费论文获取情况（2026-08-06 已通过学校订阅批量下载）

### 4.1 已完成下载（57 篇 + 2026-08-09 新增 13 篇 MoE = 70 篇本地 PDF）

通过 Playwright + CDP 连接用户已登录 Chrome（学校 IEEE/ACM 订阅），逐篇下载并内容验证：
- **2026-08-09 新增 MoE 方向 13 篇**（arXiv 直连下载，`%PDF` 魔数 + 标题内容双重验证，PDF 与 sidecar 均入库）：GShard / Switch / DeepSpeed-MoE / MoE Parallel Folding / MoE-Infinity / MoE-Lightning / ExpertFlow / HybriMoE / ProMoE / D²MoE / EC2MoE / SMoE（ISCA'26）/ Preble。清单见 [moe-literature.md](moe-literature.md) §5。
- **polylink 站点：30/33 篇** 已下载（含 PASTA、Janus、PolyLink、Decentralized Task Offloading、Hybrid Redundancy、Distributed ML in Edge 等研究相关论文）
- **CCF-A：27/30 篇** 已下载；另 3 篇 OSDI'26 无 arXiv，等 USENIX 发布
- 下载脚本：`scripts/download_via_chrome.py`（IEEE 走 stampPDF 端点、ACM 走页面点击过 Cloudflare）
- 调试环境启动脚本：`scripts/launch_chrome_debug.bat`（独立 profile，登录态保留在 `.chrome-debug-profile/`）

### 4.2 仍待获取的 6 篇

| 论文 | 说明 | 获取途径 |
|---|---|---|
| OpenTela（OSDI'26） | 无 arXiv，USENIX 尚未开放 PDF | https://www.usenix.org/conference/osdi26/presentation/yao |
| Heterogeneity at Hyperscale（OSDI'26） | 同上 | https://www.usenix.org/conference/osdi26 |
| EcoServe（OSDI'26） | 同上 | https://www.usenix.org/conference/osdi26/presentation/du |
| Scalable Graph-based RAG via LSH | VLDB 2025 workshop 论文，与本研究基本无关 | 可选：arXiv 检索或跳过 |
| Explicitly Guided VQG（AAAI） | 与本研究无关 | ojs.aaai.org 开放获取 |
| Efficient Robustness Evaluation（AAAI） | 与本研究无关 | ojs.aaai.org 开放获取 |

> USENIX 会议论文在 proceedings 发布后免费公开，届时可直接从上述页面下载。

### 4.3 sidecar 说明

已为 **70 篇** 本地 PDF 全部生成全文文本 sidecar，位于 `refs/papers/parsed/`（每篇一个 `.txt`，含章节切分）。可用于全文检索、引用核对、综述写作。

---

## 五、待办

- [x] Polylink 全部 33 篇论文整合分析（`plan/review/polylink-deep-analysis.md`，含逐篇细节 + 研究版图 + 方法配方 + D1-D6 映射）
- [x] 9 篇细读论文的详细笔记（`plan/review/deep-notes/`）
- [x] 其余相关论文核心内容总结（第 3 节）
- [x] 主题聚类与综合洞察（第 1、2.1 节）
- [ ] 获取付费论文全文后补充/修正 3.5 节内容
- [ ] 基于文献笔记构建 evidence-claim map，为 Introduction/Related Work 写作做准备
- [x] MoE 方向 13 篇下载 + 整理（`plan/review/moe-literature.md`）
- [ ] SMoE 之后，可选：为 Preble / MoE-Lightning / EC2MoE 各写一份 deep-note（当前在 moe-literature.md 汇总级）
- [ ] "KV+专家双对象联合路由"作为新研究空白，纳入 research-alignment 与 proposal 写作
