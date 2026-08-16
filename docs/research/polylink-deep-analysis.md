# PolyLink 站点论文全整合分析（33 篇）

> 来源：http://polylink.evan.cafe/（IMCL PolyU 及其合作生态的论文集合）
> 分析日期：2026-08-07
> 依据：30 篇已下载全文 sidecar（`refs/papers/parsed/`）+ 3 篇未下载论文的公开信息
> 用途：回答"这个实验室/生态做了什么"；为异构分布式算力协作研究（D1–D6）提供整体坐标与逐篇细节

---

## 〇、总览：这 33 篇论文合起来做了什么

Polylink 站点不是单一课题组的工作，而是**一个"去中心化算力 + LLM"生态的成果集**：以香港理工 Jiannong Cao 组（IMCL）为核心，深度交织华南理工 Lei Yang 组、香港中文大学（Guoliang Xing / Michael Lyu 组）、香港大学 Chuan Wu 组、香港科大（Xiaofang Zhou / Bo Li）、复旦/上交等合作者，并受同一批香港 RGC 项目（TRS **T43-513/23-N**、CRS 62321166652 等）资助。

把这些论文放在一起看，它们共同回答了一条主线问题：**"如何让 AI（尤其 LLM）跑在异构、动态、去中心化的算力上"**，可拆成四层：

| 层次 | 问题 | 代表性论文 |
|---|---|---|
| **算法/应用层** | LLM 作为编排器/推理器怎么更聪明 | BDoG、LLMEPET、GAL、COCA、SocialMind、TaskSense、Multi-Agent Resilience、VQG、DS-NER |
| **协作学习层** | 异构参与者怎么一起学（训练） | PASTA、FCIDF、DistMatch、DP-FL 公平性、Distributed ML in Edge |
| **调度/卸载层** | 任务怎么拆、分给谁、节点动态怎么办 | EdgeShard、O2O-DRL、HRTO、Janus、JPDR、TSN、DSAD |
| **系统/平台层** | 去中心化平台怎么组织算力与信任 | **PolyLink/PolyEdge**（平台）、CEI（愿景）、Graph-RAG |
| **支撑/评测层** | 真实系统怎么测、怎么保安全 | zkEVM、XPS fuzzing、CAN IDS、WebAssembly、交通预测系列、ACMMM 系列 |

核心叙事：**从"边缘任务调度"（CPU 小任务）→ "边缘承载大模型"（LLM/ViT 切分协作）→ "去中心化平台"（PolyLink），并由 LLM 系统研究与安全评测两条侧翼支撑**。

对你（异构分布式算力协作执行 LLM 训练/推理）来说，**最直接的锚点**是 EdgeShard（LLM 切分+DP 调度）与 PolyLink 平台论文（真实跨地域部署）；**方法资产**在调度/卸载层（DP 精确解与 DRL 两条互补路线）；**空白点**也明确：这些工作大多**不做节点动态装载/卸载、不做新 GPU 在线评估、不做训练与推理统一弹性框架**。

---

## 一、逐篇详细分析（33 篇）

### 组 A：边缘协作与任务卸载（6 篇）★ 与本研究最相关

---

#### A1. EdgeShard: Efficient LLM Inference via Collaborative Edge Computing（IEEE IoTJ 2024）

- **作者**：Mingjin Zhang, Jiannong Cao, Xiaoming Shen, Zeyang Cui（香港理工）
- **核心贡献**：首个把 LLM **按层切分**、用**动态规划联合优化"设备选择+模型切分"**的通用协作边缘推理框架，在 15 台异构设备（Jetson 系列 + RTX 3090 云机）上把 70B 模型从"全部 OOM"变成可推理。
- **动机**：LLM 推理过度依赖云中心（高时延、高带宽、隐私）；边缘单设备放不下（Llama2-7B 全精度 28GB）；量化/云边协作受不稳定网络影响。
- **方法**：
  - 三阶段：离线 profiling（每层在每设备上的计算/激活/内存/带宽）→ 调度优化 → 协作推理（顺序 / 流水线并行）。
  - **延迟优化（DP）**：决策变量 X_ij（第 i 层分给设备 j）；通信时间 t^comm=O_{i-1}/B_{k,j}；约束含**隐私约束**（首层必须留在源节点 X_{0,0}=1）与内存约束。DP(i,j)=min_k(DP(i-1,k)+t^comp+t^comm)，回溯得最优分配。
  - **吞吐优化**：流水线并行下瓶颈=max(计算, 通信)，DP 状态 g(i,S,k)（前 i 层、已用设备集 S、末设备 k），复杂度 O(N²·2^M·M²)。
  - **EdgeShard-No-bubbles**：自回归 LLM 下允许单 micro-batch prefill 结束后立即开始 token 生成（不等整批），消气泡。
- **结果**：Llama2-7B 时延 75.88ms（比 Edge-Solo 快 ~1.85×）、吞吐 52.45 tok/s（~2.2×）；**Llama2-70B 用 11 台 AGX Orin + RTX 3090 实现 3086ms、1.25 tok/s**（基线全 OOM）。
- **与本研究（D1–D6）**：D1/D2/D3 **直接命中**（按层切分=任务拆分、设备选择=任务分配）；D6 **直接**（模拟 1–50Mbps 不稳定带宽）；D4 **间接**（静态选择，未处理成员动态离开）；D5 **间接**（无新 GPU 在线评估，但 profiling 思路可扩展）。
- **版图定位**：CEC 路线上最贴近 LLM 落地的一篇，PolyLink 平台跨设备推理直接复用它做分片。

---

#### A2. Decentralized Task Offloading in Edge Computing: An Offline-to-Online RL Approach（IEEE TC 2024）

- **作者**：Hongcai Lin, Lei Yang, Hao Guo（华南理工）, Jiannong Cao（香港理工）
- **核心贡献**：提出 **O2O-DRL**——用启发式算法日志**离线预训练** DRL，再 **on-policy 在线微调**，解决去中心化任务卸载中 DRL 的**冷启动/sim-to-real** 问题，最大化任务成功率。
- **方法**：
  - Dec-POMDP：观测（节点负载、失效率、计算能力、队列、链路速率）+ 动作（卸载到哪个节点，mask 掉不存在链路）+ 团队奖励（成功 +1、失败/超时 −1）。
  - 离线：基于 **Discrete SAC + CQL 正则化**（压低未见动作 Q 值）从启发式（RATC）日志训练；在线：**GAE + PPO clip** 微调，避免离线→在线 Q 值过估计。actor 仅 3 层全连接（<100KB），通信开销小。
  - 显式建模节点/链路失效（泊松）进可靠性。
- **结果**：仿真（20 节点）+ Kubernetes 真机测试床；所有任务规模/到达率/失效率/节点数设置下成功率最高；对启发式质量鲁棒。**future work 明确写"扩展到 LLM 分布式训练/推理的 DAG 任务"**。
- **与本研究**：D3 **直接**（去中心化任务分配）；D4 **直接/间接**（节点/链路失效显式进模型）；D6 **直接**；D1 间接。D5 无关。
- **版图定位**："任务分配与卸载决策"线的 RL 方法论代表；Dec-POMDP + 集中训练分布式执行范式与 HRTO 一脉相承。

---

#### A3. Hybrid Redundancy for Reliable Task Offloading in Collaborative Edge Computing（IEEE TC 2025）

- **作者**：Hao Guo, Lei Yang, Qingfeng Zhang（华南理工）, Jiannong Cao（香港理工）
- **核心贡献**：提出 **HRTO**——**动态自适应"主动+被动"混合冗余**卸载机制 + 带自注意力的**三头 actor 的 MAPPO** 多智能体 RL，同时权衡时延与可靠性。
- **方法**：
  - 三维决策：卸载目标 x、**主动冗余等级 y**（并行副本数）、**失败后是否重执行 z**。混合冗余可靠性 R^h 有闭式公式（k 个并行实例全失败后依次启用剩余实例）。
  - 观测含节点/链路失效模型；奖励分档（实例成功 +1、冗余终止 0、失败 −1；主动冗余 ±0.5、重执行 ±0.3）缓解稀疏奖励。
  - 自注意力对观测加权；MAPPO 集中训练、分布式执行。
- **结果**：6 个真实网络拓扑（Abilene 11 → GtsCe 149 节点）+ K8s 测试床；相对 Random 成功率提升 ≥14.6%；消融证明**"盲目加冗余反而拖垮系统"**。
- **与本研究**：D4 **直接**（冗余即应对节点/链路失效与动态负载的核心机制，对重算成本高的训练任务尤其可迁移）；D3 **直接**（多跳卸载+副本数联合决策）；D6 **直接**。D5 无关。
- **版图定位**：O2O-DRL 的升级线，把可靠性从"进奖励"提升为"专门冗余机制"。

---

#### A4. Janus: Collaborative Vision Transformer Under Dynamic Network Environment（INFOCOM 2025）

- **作者**：Linyi Jiang, Yifei Zhu（上交密西根）, Silvery D. Fu, Bo Li（港科大）
- **核心贡献**：首个云-端协作 ViT 推理系统，把 **token 剪枝与模型切分融合**，配**动态调度器**在波动网络下选择最优剪枝率与切分点，实现低时延/高精度/低通信。
- **方法**：
  - **协作感知 token 剪枝**：指数衰减控制每层剪枝数（前层多剪后层少剪），制造"中间数据缩减"机会——这是 ViT 相对 CNN 的关键难点（ViT 不改变张量尺寸）。
  - 候选切分点**细到粗生成**（前段密后段疏）+ 轻量线性 profiler（每层时延与输入 token 数强线性相关，r>0.85）。
  - 动态调度器：按 α（declining rate）扫描、对每个切分点估通信时延（带宽用调和均值），取总时延最小配置；复杂度 O(α_max/t × N)，约 1ms。
- **结果**：Jetson Orin Nano + 阿里云 V100，真实 5G/4G mmWave trace（Static/Walking/Driving）；吞吐最高 **5.15×**、时延违反率最高降 **98.7%**、精度略升。
- **与本研究**：D1/D2 **间接偏直接**（模型层切分到设备/云）；D6 **直接**（真实 5G trace，六篇中通信建模最贴近非数据中心）；D5 **间接**（边缘/云双硬件对比，新硬件评测雏形）。D4 间接（只应对带宽波动，不应对节点增删）。
- **版图定位**："模型切分 + 动态网络自适应"技术线；**"剪枝/压缩制造中间数据缩减以配合切分"对 LLM 的 KV cache / 激活值跨节点传输优化有直接迁移价值**。

---

#### A5. Distributed Machine Learning in Edge Computing: Challenges, Solutions and Future Directions（ACM CSUR 2024）

- **作者**：Jingke Tu, Lei Yang（华南理工）, Jiannong Cao（香港理工）
- **核心贡献**：系统综述边缘分布式机器学习——这是整个生态的**"设计空间地图"**。
- **分类框架**：
  - **并行模式**：模型并行（线性/非线性按层切分）、数据并行、混合并行（Gpipe/PipeDream）、专家并行（MoE）。
  - **通信架构**：集中式 PS / 簇式 / 去中心化（Ring All-Reduce、gossip）/ 分层式（HFL）；同步 BSP / 异步 / 弱同步。
  - **模型聚合**：同构（FedAvg 及变体：client-drift、q-FFL 公平性、动态网络加权）；异构（知识蒸馏：有/无公共数据）。
  - **资源受限**：剪枝、低秩分解、量化；**non-IID**：扩充、聚类、个性化、元学习。
  - **未来方向**：异构结构模型聚合、大小模型协同进化、动态通信、增量学习、轻量隐私保护。
- **与本研究**：D1 **直接**（并行模式=拆分范式全集）；D2/D3 **直接**（通信架构与聚合=分配结构）；D4 **直接/间接**（dropout 缓解、容错、动态网络加权置 0 的方法论全集）；D6 **直接**（通信效率是核心挑战）。
- **版图定位**：综述性"地图"，理解 LLM 分布式训练设计空间的最佳起点。

---

#### A6. Collaborative Edge Intelligence for Autonomous Vehicles: Opportunities and Challenges（IEEE Network 2024）

- **作者**：Luyao Bai, Jiannong Cao, Mingjin Zhang（香港理工）, Bo Li（港科大）
- **核心贡献**：提出并命名 **CEI（协作式边缘智能）** 范式——车、RSU/基站、云连成**联邦资源池**，**水平+垂直双维协作**，作为整个生态的"愿景/宣言"文件。
- **方法（框架性）**：三层架构（基础设施/中间件/应用）；中间件含资源管理、任务调度、AI 支持、激励、隐私安全。
  - **任务调度**：域内（in-edge）+ 跨域（cross-edge）混合调度；计算与网络联合；**移动感知的可靠任务迁移**（车辆在 RSU 间移动需轻量实时迁移，区别于数据中心的稳定高速迁移）——直接对应本研究 D4。
  - **AI 支持**：可扩展去中心化训练、资源感知推理（压缩/早退/切分自适应动态资源）。
  - **激励**：Proof-of-Collaboration（你已排除）。
- **与本研究**：D2/D3 **直接**（联邦资源池、域内/跨域联合调度）；D4 **直接**（任务迁移、服务随移动迁移、冗余容错是核心承诺）；D6 **直接**（V2X、低带宽不稳定链路、镜像分片传输）。D5 无关。
- **版图定位**：最顶层的范式定义；EdgeShard 与它是"系统实现 ↔ 范式定义"的两端。

**组 A 共同线索**：CEC/CEI 范式底座（Jiannong Cao × Lei Yang 深度交织）；"DP 精确解（EdgeShard） vs DRL 自适应（O2O/HRTO）"两条互补路线；可靠性/容错与动态网络自适应是共同关心；实验习惯一致（Python 仿真 + K8s 测试床 / 真实异构原型）；场景从"CPU 任务卸载"向"边缘承载大模型"演进——**六篇合起来恰好覆盖"任务怎么拆、分给谁、节点动态怎么办、用什么算法、在什么网络上跑"的完整问题链**。

---

### 组 B：联邦/分布式协作学习（4 篇）

---

#### B1. Mitigating Unfairness in Differentially-Private Federated Learning（ACM ToMPECS 2025）

- **作者**：Bingqian Du（华中科大）, Liyao Xiang（上交）, Chuan Wu（港大）
- **核心贡献**：首次把 DP 联邦学习的**裁剪阈值设计本身**与公平性缓解结合——按客户端**自适应设定梯度裁剪值 C_k**，在不增加客户端开销的前提下缓解性能不公平。
- **方法**：全局高斯 DP 机制中每客户端独立裁剪值；定理把公平性差距拆成统计异质性 + DP 参数项；优化问题（MMDM 微分乘子法）最小化损失+噪声项，约束最差客户端损失差距；客户端只需回传一个标量损失。
- **结果**：方差（×10⁴）在 Synthetic 735 vs 基线 1290；最差 10% 客户端准确率全面改善。
- **与本研究**：间接。"按参与方异质性自适应调节其在协作中的贡献"——异构算力协作中"参与方公平性/负载均衡"的理论样板。
- **版图定位**：分布式协作学习的公平/隐私维度。

---

#### B2. Federated Class-Incremental Learning with Dynamic Feature Extractor Fusion（IEEE TMC 2024）

- **作者**：Yanyan Lu, Lei Yang, Hao-Rui Chen（华南理工）, Jiannong Cao, Wanyu Lin（港理工）, Saiqin Long（暨南）
- **核心贡献**：提出 **FCIDF**——逐层可学习的**融合率向量 α** 动态融合全局与本地特征提取器 + 元学习 + 累积全局特征均值存储（AGFMS），同时解决空间 non-IID 与时序灾难性遗忘。
- **方法**：只共享特征提取器（分类器留本地）；θ = α·θ_glob + (1−α)·θ_loc；元学习式增量训练；AGFMS 上传各类别特征均值并做累积平滑，指导样例选择。
- **结果**：准确率最高 +7.69%、遗忘最高 −27.30%。
- **与本研究**：间接。"全局-本地知识按层融合/动态配比"对多源模型/知识融合有借鉴。
- **版图定位**：分布式、流式数据下模型持续适应（不遗忘）。

---

#### B3. Distributed Semi-Supervised Learning with Consensus Consistency on Edge Devices（IEEE TPDS 2024）

- **作者**：Hao-Rui Chen, Lei Yang, Xinglin Zhang（华南理工）, Jiaxing Shen（岭南）, Jiannong Cao（港理工）
- **核心贡献**：首个**完全去中心化（无中心服务器）**的边缘半监督学习 **DistMatch**——用邻居模型按**置信度加权平均**生成伪标签，一致性一致性损失过滤高熵伪标签。
- **方法**：每设备用本地有标注集评估邻居模型置信度 C_k(θ)=exp(−L_sup)，近似数据分布相似度；共识伪标签=置信度加权平均 + 阈值过滤；一致性损失只对高共识样本；按数据集大小加权聚合。
- **结果**：non-IID 下全部数据集最优；SVHN 到 80% 仅用 98 轮 vs 基线 114 轮。
- **与本研究**：D4/D6 **偏直接**（无中心服务器避免单点故障、gossip 式通信、对拓扑动态更鲁棒）；D2 间接。四篇中最贴近"非数据中心通信"之一。
- **版图定位**：无中心协调下多设备协作学习的"共识/一致性协调原语"。

---

#### B4. PASTA: Training Acceleration for Vertical Federated Learning via Adaptive Pipeline Parallelism（IEEE Cluster / IWQoS 2025）

- **作者**：Tian Wu, Han Liang, Ziwei Zhan, Xu Chen, Xiaoxi Zhang（中大）, Weijie Liu, Chuan Wu（港大）, Jingpu Duan（鹏城）, Jinhang Zuo（城大）
- **核心贡献**：VFL 中首次实现**自适应流水并行 + 陈旧性控制**——被动方在通信等待期间预计算后续 batch 的 embedding 并用陈旧梯度本地训练，按各参与方**异构算力/网络**用反馈信号自适应调节"在途 embedding 数 B"与"陈旧本地更新轮数 l"，训练提速 **1.8×–4.6×** 且不损精度。
- **方法**：
  - 流水并行 VFL：被动方三态机（前向/陈旧本地训练/后向），控制信号 C_k∈{−1,0,1} 由主动方（coordinator）发出调节 B。
  - 陈旧性衰减 s_e=s_max/√e 随 epoch 单调衰减（训练后期陈旧梯度更易振荡）。
  - 理论证明消除的气泡量与 B、延迟正相关。
- **结果**：1000 时间单位延迟下相对 FedSGD 提速 4.4/4.6/3.2/3.5×（FedBCD 仅 1.1–2.3×）；真机 4×4090 + TC 控网，异构带宽 4.04×。
- **与本研究**：**本组最直接相关**。D1/D3 **直接**（自适应流水切分=训练任务在异构单元间如何拆/分/配）；D4 间接（反馈式 B/l 调节即对负载/资源状态变化的在线再分配）；D5 直接/间接（真机多卡评估）；D6 **直接**（WAN 100ms–10s 传播延迟、带宽异构、反馈控制针对动态网络）。
- **版图定位**：最接近"异构分布式算力单元协作执行训练"目标的**方法论样板**——流水并行 + 陈旧性控制 + 反馈式自适应负载分配，可直接迁移到异构集群/多 GPU 的 LLM 训练流水与推理任务切分设计。

**组 B 共同线索**：都处理参与方数据/能力不均衡；都采用"学习得到的个性化权重"调节每个参与方在协作中的贡献或负载（裁剪值 C_k、融合率 α、置信度表、在途数 B）；都围绕"有限/昂贵通信下做多少本地工作"的权衡；共享项目资助（T43-513/23-N）与作者（Chuan Wu、Lei Yang、Jiannong Cao 交叉出现）。

---

### 组 C：LLM 系统（5 篇）

---

#### C1. SocialMind: LLM-based Proactive AR Social Assistive System（IMWUT 2025）

- **作者**：Bufang Yang, Yunqi Guo, Zhenyu Yan, Guoliang Xing（港中大）, Lilin Xu, Xiaofan Jiang（哥大）
- **核心贡献**：首个**主动式** AR 社交辅助系统——AR 眼镜多模态感知 + 服务器 LLM，实时生成 in-situ 社交建议。
- **方法**：端侧轻量模型（MediaPipe）本地提取非言语线索（省带宽保隐私）；主用户识别用**振动信号**而非语音指纹；社交因子/隐式人设解析；**社交因子感知缓存**（agent 角色扮演预生成+相似度路由）+ **意图推断**（每 2 秒把不完整语句 offload 提前推理）+ 主动响应更新。
- **结果**：engagement 比基线高 38.3%；20 人真实用户研究 95% 愿意使用。
- **与本研究**：间接。"端云算力分工 + 缓存/提前推理降低 LLM 延迟"的工程技巧，可迁移到分布式 LLM 推理的低延迟路由（D3 侧）。
- **版图定位**：端云协同 LLM 系统的完整工程范式。

---

#### C2. TaskSense: A Translation-like Approach for Tasking Heterogeneous Sensor Systems with LLMs（SenSys 2025）

- **作者**：Kaiwei Liu, Bufang Yang, ... Guoliang Xing（港中大）+ 华为诺亚
- **核心贡献**：提出"**传感器语言**"（Sensor Language）把异构传感器/工具的能力与数据依赖**翻译**给 LLM，使 LLM 生成可执行工具调用计划，并**运行时按环境反馈动态切换执行路径**。
- **方法**：工具建模为"词汇"（名称/模态/输入输出）、依赖建模为"语法"（DAG）；LLM 自动注册新工具；可解性检查 + 语法检查；**动态计划适配**（把功能等价工具聚成 adaptable groups，执行前按数据缺失/信噪比过滤、执行后结果无效则切换路径）+ 结果缓存。
- **结果**：规划精度最高 0.97（vs 0.86）、响应精度 0.75（vs 0.55）、**最高 3× 规划精度**；缓存把执行延迟 360s→31s；部署在 Jetson AGX Orin。
- **与本研究**：**本组最直接相关，整体可迁移**。传感器系统=异构算力/功能单元，LLM=中央编排器；"传感器语言"=**异构能力的形式化描述语言**（可类比算力单元注册表/能力描述）；可解性检查=任务可调度性判定；**动态计划适配=故障/数据缺失下的子任务迁移到等价替代单元（对应 D4）**；自动注册=新单元加入时的能力自描述（对应 D5 的一部分）。
- **版图定位**：完整演示"LLM 编排器 + 异构能力单元 + 形式化能力描述 + 可解性判定 + 运行时故障自适应"范式，**可整体迁移到异构分布式算力协作原型设计**。

---

#### C3. COCA: Generative Root Cause Analysis for Distributed Systems with Code Knowledge（ICSE 2025）

- **作者**：Yichen Li, ..., Zhuangbin Chen（中大）, Michael R. Lyu（港中大）
- **核心贡献**：首个把**代码知识**注入分布式系统 issue 报告根因分析的框架——从代码补执行线索，LLM 生成根因摘要并定位责任组件。
- **方法**：日志源检索（模板匹配还原源码位置）→ 执行路径重建（ICFG + **RPC bridging** 跨节点补齐调用链）→ 代码剖析（method 级索引控制进入 prompt 的代码量）→ 根因推断（BM25 相似历史 issue + in-context learning）。
- **结果**：定位 Exact Match 提升 28.3%、摘要 BLEU-4 +22.0%；5 个真实分布式系统 106 个 issue。
- **与本研究**：间接。核心洞见"分布式故障传播横跨多节点/组件、RPC bridging 重建跨节点调用链"与"跟踪跨算力单元通信与故障"可直接类比；"静态分析+结构化索引压缩代码库以适配 LLM 上下文"是通用技巧。
- **版图定位**：分布式系统故障诊断/可观测性视角，可作为算力系统**故障监测与自愈**方向的知识来源。

---

#### C4. On the Resilience of LLM-Based Multi-Agent Collaboration with Faulty Agents（ICML 2025）

- **作者**：Jen-tse Huang, Jiaxu Zhou（港中大）, Tailin Jin（清华）, Xuhui Zhou（CMU）, ..., Michael R. Lyu（港中大）, Maarten Sap（CMU）
- **核心贡献**：系统研究 LLM 多智能体协作在**线性/扁平/层次**三种结构下对"笨拙或恶意 agent"的韧性，并提出 **Challenger（互相挑战）+ Inspector（监督审查）** 两种恢复机制。
- **方法**：故障注入（AUTOTRANSFORM 改写 agent profile / AUTOINJECT 注入错误消息）；结构分类（线性 MetaGPT/Self-collab、扁平 Camel/SPP、层次 MAD/AgentVerse）；韧性增强机制；4 个下游任务（代码/数学/翻译/评估）。
- **结果**：**层次结构韧性最好**（性能下降 5.5% vs 扁平 10.5% vs 线性 23.7%）；代码生成最受影响（−22.6%）；对高层"任务分发者"注入影响更大；**Challenger+Inspector 可恢复 Self-collab 上 96.4% 的故障损失**。
- **与本研究**：**容错/弹性维度最重要的参考文献（直接类比）**。线性/扁平/层次 ↔ 流水线/全连接对等/中心化分层拓扑；故障 agent=失效算力单元；故障注入工具=硬件故障注入；Challenger/Inspector=校验/副本/监督节点；**结论可直接指导算力协作设计：引入监督/校验单元与 peer 互检大幅提升弹性；层次（有中心仲裁）最抗单点故障，线性链（pipeline）最脆弱**。
- **版图定位**：为"分布式算力单元故障弹性"提供故障注入、结构韧性度量、恢复机制的完整方法论。

---

#### C5. SAILS: A Synchronous Accessible Immersive Online Learning System（MMSys 2025）

- **作者**：Yuran Sun, Zhuoying Zhang, ..., Chuan Wu（港大）
- **核心贡献**：面向低龄学生的同步沉浸式在线学习系统（3D 虚拟教室 + 真实感头像 + 实时动作迁移），普通家用电脑双摄像头实现。
- **方法**：双摄像头方案（脸+桌面作业）；单照片真实感 3D 头部建模；**端-云架构**：Unity 客户端做实时感知/渲染，服务器做重计算（头部建模）与数据分发；HTTP/REST 低频控制 + **Kafka** 实时流；**任务按客户端处理能力与可用带宽自适应分配**。
- **结果**：90%+ 用户喜欢；90% 延迟 <135ms、平均吞吐 78.96Mbps 支撑 ~80 用户；头部建模 CPU>250%（未来卸载 GPU）。
- **与本研究**：关联最弱。间接：端-云按计算强度分工 + 按能力/带宽分配任务的最简实例；Kafka 实时分发与延迟/吞吐评估是"非数据中心通信"的工程案例。
- **版图定位**：端-云实时系统架构与**系统评估方法论**（延迟/吞吐/CPU 指标）。

**组 C 共同线索**：五篇全部来自香港高校（CUHK 3、HKU 1、跨港 1）；共享范式"**LLM 作为中央协调器编排异构底层单元**"（传感器/工具、代码库、agent、多模态感知）；全部围绕"运行时不确定性/故障下的鲁棒性"；都用"结构化中间表征压缩进 prompt"。**TaskSense 与 Multi-Agent Resilience 是最直接的两篇蓝本**。

---

### 组 D：智能交通与网络调度（5 篇）

> 方法可迁移性强，领域与本研究的物理距离远。

---

#### D1. Ultra-Flexible, Explainable, and Scalable Traffic Prediction with Dynamic Future Routes（ICDE 2025）

- **作者**：Zizhuo Xu, Lei Li, ..., Xiaofang Zhou（港科大/港科大广州/澳国立）
- **核心贡献**：提出 **RouteSys**——用"宏级交通仿真 + 一组轻量机器学习模型"替代大型深度学习时空模型，以**未来路线（根因）** 而非历史序列做预测根因，实现可解释、可扩展、对任意时域稳定。
- **方法**：SUMO 仿真生成"真实事件数据"与"手动分配数据"训练轻量模型；宏级仿真程序（事件驱动、BPR/轻量模型更新）；假设道路按静态特征聚类（KMeans 100 类）；**模型缓存**（特征→旅行时间查表，推理毫秒→微秒）。
- **结果**：15min MAE 2.27 vs 最好基线 DCRNN 2.33；各 GNN 基线在 ~6k 路时 GPU OOM，RouteSys 无限制。
- **与本研究**：间接偏直接。**模型缓存/查表替代实时推理**（对应 KV/结果缓存）、**用仿真作为 ground truth 训练与评估**、**轻量化按能力分组建模**（对应异构算力分组）。

---

#### D2. A Just-In-Time Framework for Routing-Oriented Traffic Prediction（ICDE 2025）

- **作者**：Jing Zhao, Lei Li, ..., Xiaofang Zhou（港科大）
- **核心贡献**：把"全网络周期性全局预测"改写为**查询驱动的按需（JIT）预测**——路网分区后只预测路由查询实际经过的区域、只在数据过期时预测。
- **方法**：SSE（训练模型预测 A* 实际搜索区域）+ RTSP（区域速度预测：全局 GCN + 区域内 LSTM/GCN + 速度剖面有效性权重）+ **全局区域预测调度**（过期检测 + 优先级分数决定更新顺序）。
- **结果**：北京 18 万节点；SSE F1=0.774 vs 椭圆法 0.579；基线随规模 OOM，JIT-TP 内存低平。
- **与本研究**：**五篇中最同构的一篇（直接）**。**查询/请求驱动的按需计算**（类似 LLM 推理按请求触发、按需分配算力）；**空间分区+时间过期检测+优先级调度**（对应异构算力单元按需分配、避免全量周期性重算）；**结果复用/缓存**；把大模型任务拆成**数据管理问题（DB4AI）**——与把巨型 LLM 任务拆到多算力单元协作同构。

---

#### D3. Unifying Lane-Level Traffic Prediction from a Graph Structural Perspective: Benchmark and Baseline（TKDE 2025）

- **作者**：Shuhao Li, Jingyi Xu, Weidong Yang（复旦）, Yue Cui, Xiaofang Zhou（港科大）, 等
- **核心贡献**：首个车道级预测统一基准：分类框架 + 三个公开数据集 + 统一图拓扑 + 轻量基线 **GraphMLP**，并把"训练成本"纳入基准。
- **方法**：统一图构建（距离图/二值四向邻接图/自适应图）；GraphMLP=实例归一化 + **动态图网络（自注意力动态生成邻接矩阵）** + 时间 MLP + 门控融合。
- **结果**：GraphMLP 综合最优，PeMSF 不规则车道 MAE 降 3.93%。
- **与本研究**：间接。**动态/自适应邻接矩阵建模**可类比分布式算力中节点加入/离开后拓扑与通信关系动态变化（D4/D6 的建模思路）；**benchmark 方法论**（公开数据集、复现代码、精度+成本双指标）可作为构建算力 benchmark 的模板。

---

#### D4. Joint Optimization of Pricing, Dispatching and Repositioning in Ride-Hailing（TKDE 2024）

- **作者**：Zhongyun Zhang, Lei Yang（华南理工）, Chao Ma（武大）, Jianguo Wang（普渡）
- **核心贡献**：首次联合优化网约车**定价+派单+重定位**，提出 **JPDR**——上下文多臂老虎机（定价）+ **SAC 多智能体 RL**（派单/重定位融合）+ **V-critic 跨阶段传递价值**。
- **方法**：定价=LinUCB（上下文老虎机，CNN 需求预测器）；派单+重定位=每个司机为 POMDP 智能体、共享**集中式 meta-agent 策略**、SAC 训练；**V-critic 估算订单的"未来效应"反馈给定价阶段**（互导）；无效动作掩码 + **Kuhn-Munkres 最优二分图匹配注入回放池纠偏**。
- **结果**：GMV/SR 全面最优（300 司机、多需求规模）。
- **与本研究**：**直接**。**集中式策略调度所有单元**（平台级控制面管理异构算力的架构）；**多任务联合优化互导**（调度↔定价/分配↔迁移的耦合，V-critic 连接机制可借鉴）；网格化离散动作空间+无效掩码+折扣奖励的状态/动作/奖励设计可直接用于任务分配 RL；**KM 组合优化纠偏 + RL** 是 RL 与经典调度结合的成熟配方。

---

#### D5. Reliable Routing and Scheduling in Time Sensitive Networks Based on RL（IEEE TNSE 2025）

- **作者**：Hao Cheng, Lei Yang, Qingfeng Zhang（华南理工）, Weiping Zhu（武大）
- **核心贡献**：TSN（时间敏感网络）可靠冗余路由与调度——**启发式候选路径生成 + PPO 选择多路径 + 早期时隙调度**（离线 RMRSA）+ **在线微调 + 替代安全策略**（在线 ORSJS）。
- **方法**：Top-K 最短路径生成候选冗余路径；动作=**从候选集中选路径组合**（把动作从"逐节点下一跳"重定义为"组合选择"，压缩动作空间）；奖励=负载均衡度（Umax−Umin）+失败流惩罚；早期时隙调度复用同链路时隙；在线用**替代策略**（检测不安全动作→用冲突最小的简单冗余路由替换）保证运行期安全。
- **结果**：均衡度比 RAEAP 好 38.7%；在线丢包率比基线低 8.1%/4.9%。
- **与本研究**：**直接**。**离线预训练 + 在线微调 + 安全替代策略**直接对应异构算力节点动态加入/离开、环境漂移时的"先离线训好、上线后用安全兜底 + 继续微调"（D4 调度器安全网）；**动作空间压缩设计**（逐节点→候选组合）适用于大规模任务调度；**先启发式生成候选、RL 从中挑选**的两阶段范式减少探索空间。与分布式算力任务调度几乎同一数学结构。

**组 D 共同线索**：HKUST（Xiaofang Zhou，图时空+数据管理）与 SCUT（Lei Yang，多智能体 RL）两系合流；共同方法论=**拆大模型为轻模型 + 数据管理（DB4AI）**、**按需/查询驱动计算与结果复用**、**RL 调度三件套**（集中式 meta-agent、组合优化纠偏、状态/动作/奖励设计）、**仿真器即 ground truth + 离线/在线两阶段**。对研究的可迁移资产：JPDR 与 TSN 的 RL 调度骨架 + JIT-TP 的按需计算组织 + 动态邻接矩阵建模。

---

### 组 E：安全与系统验证（5 篇）⚠ 与本研究关联最弱

> 保留价值：方法学参照（统一基准、质量指标、状态感知、LLM 辅助工具化）；共享作者网络说明它们是同一生态的安全侧翼。

#### E1. Driving State-Aware Anomaly Detection for Autonomous Vehicles（IEEE TIFS 2025）
- Xiapu Luo（港理工）等。**DSAD**：双层有限状态机建模驾驶状态 + 按状态自适应检测 + 兼容故障处理机制。4 个 ADS 上攻击检出率 >90%。
- 相关性：很弱。可借鉴"对动态运行状态的自适应建模"（对应调度中的状态感知）。

#### E2. Vehicular Intrusion Detection System for CAN: Survey and Evaluation（IEEE TITS 2025）
- Xiapu Luo 等。系统梳理 34 项攻击/53 个 VIDS，在统一真实 CAN 数据上复现评估 17 个代表方法，诚实报告基线盲区。
- 相关性：很弱。**统一基准/统一数据集横向评测的方法学**可参考（评测分布式训练系统可靠性）。

#### E3. Distinguishability-guided Test Program Generation for WebAssembly（IEEE SANER 2025）
- Michael Lyu（港中大）+ 复旦。**WarpGen**：用"可区分度"（不同运行时上执行时间比应成稳定比例）引导生成触发性能异常的 Wasm 测试程序，发现 7 个新性能问题。
- 相关性：很弱。**oracle 式性能判据**（对比不同实现/节点的执行时间比）可用于异构算力节点性能基准与回归测试（D5 评估侧）。

#### E4. Automated Soundness and Completeness Vetting of Polygon zkEVM（USENIX Security 2025）
- Xiapu Luo 等。**FreeVer**：自动形式化验证 Polygon zkEVM 的 free inputs，发现"双重执行路径攻击"新攻击面 + 7 个新高危漏洞。
- 相关性：无关。"诚实/恶意两个视角建模 + 状态图对齐比较"的验证套路可抽象用于验证分布式训练一致性/容错协议。

#### E5. Error Messages to Fuzzing: XPS Parsing Vulnerabilities in Windows Printing（ACM CCS 2025）
- HUST + Xiapu Luo。**PrintXPSurge**：LLM 辅助修复生成语义正确 XPS + 快照渐进重构闭源工作流运行时 + 回溯定位，10 个驱动发现 17 个已确认 0-day（CVE）。
- 相关性：很弱。**LLM-as-repairer** 属 LLM 应用谱系；"渐进式状态重建解决多模块异步依赖工作流"可类比分布式服务编排运行时构建。

---

### 组 F：多媒体与去中心化平台（4 篇）

---

#### F1. A Picture Is Worth a Graph: Blueprint Debate Paradigm for Multimodal Reasoning（ACM MM 2024）

- **作者**：Changmeng Zheng, ..., Xiao-Yong Wei, Tat-Seng Chua, Qing Li（港理工）
- **核心贡献**：**BDoG**——把多智能体辩论从词级归纳改为**图级演绎**（蓝图图约束辩论范围），同时解决观点平凡化与焦点漂移，ScienceQA/MMBench 达 SOTA。
- **方法**：场景图初始化（Size/Relevance 约束）；Proponent/Opponent/Moderator 三类角色在图上竞争；停止条件=图距离收敛；比纯辩论少 ~50% 推理时间。
- **与本研究**：无关（算法层）。可算间接的点：把复杂推理任务拆给多角色 agent 再合并，与把训练/推理任务拆给异构算力单元有隐喻价值；**结构化中间表示（图）约束协作过程 ≈ 用 DAG 约束调度**。

---

#### F2. Prior Knowledge Integration via LLM Encoding and Pseudo Event Regulation for VMR（ACM MM 2024）

- **作者**：Yiyang Jiang, ..., Xiao-Yong Wei, Qing Li（港理工）
- **核心贡献**：**LLMEPET**——用 LLM 编码器（而非解码器）作跨概念关系精炼器 + 伪事件边界正则，以**插件方式**插入任意 VMR 框架均一致增益。
- **结果**：QVHighlights MR mAP 44.05、HD HIT@1 65.69；插到 5 个框架均增益。
- **与本研究**：无关。"可插拔模块 + 跨框架泛化"的验证范式可参考（把调度/弹性模块做成可插拔中间件）。

---

#### F3. Generative Active Learning for Image Synthesis Personalization（ACM MM 2024）

- **作者**：Xulu Zhang, ..., Qing Li（港理工）+ CASIA
- **核心贡献**：把主动学习迁移到生成模型（图像合成个性化），**锚定方向**把开放式查询转半开放问题 + 方向不确定性采样 + 开放度平衡；开源模型超过 Google 闭源 StyleDrop。
- **与本研究**：无关。"用锚点把开放空间约束为半开放候选集"≈把无限任务空间约束为候选节点/分片集合的剪枝思想；"不确定性高的方向值得优先探索"≈"新 GPU 到来时选最值得评测的节点/分片"。

---

#### F4. PolyLink: A Blockchain-Based Decentralized Edge AI Platform for LLM Inference（IEEE Blockchain 2025）★ 生态顶点

- **作者**：Hongbo Liu, Jiannong Cao, Dongbin Bai, Yinfeng Cao, Xiaoming Shen, ..., Mingjin Zhang（港理工）+ 中国移动香港 + 中山大学
- **核心贡献**：首个带完整设计的**基于区块链的去中心化边缘 AI 平台**，支持单设备与跨设备（复用 **EdgeShard** 分片）LLM 推理，配 **TIQE** 可信推理质量评估协议与动态 Token 激励模型，**跨地域异构设备真实部署**。
- **动机**：LLM 服务高度中心化；DePIN 是去中心化方向，但有"低端设备算力不足、推理完整性校验（zkLLM 慢）、激励忽视模型开发者"三难。
- **方法（与你研究直接相关的部分）**：
  - **架构**：四方参与者——Service User / Worker（贡献空闲 GPU，含 NVIDIA、Jetson、Apple Silicon）/ Model Provider / Validator（质押+评估）。API Server 按"同模型请求"批量路由到运行该模型的 worker。区块链（Sepolia + ERC-20）作协调/结算层。
  - **算力聚合**：单设备推理（小参数模型）；跨设备推理用 **EdgeShard 顺序切分**（M=Π_i M_i，按 r 轮迭代跨 shard 执行）——即**流水线式模型并行**。
  - **任务分发/调度**：本质是"按目标模型把请求批量路由到已部署该模型的 worker"。**调度粒度粗：无动态负载均衡、无迁移算法、无数据并行**——这是你研究的空白点。
  - **节点管理**：去中心化、无 TTP；验证者委员会每 epoch 用 **VRF** 选举；中位数打分 + 偏差阈值扣质押（slashing）。
  - **TIQE 完整性协议**：Cross-Encoder（轻量 MiniLM）日常校验 + LLM-as-a-Judge（o3/DeepSeek-V2）随机抽查加权混合（Hybrid）。
- **结果**：真实部署 **20 台设备 / 10 个 worker**（香港、广州、深圳、日本金泽），含 RTX 4090/3090/3080Ti/4060Ti、P100/A100、GV100、Jetson Orin NX/AGX、Apple M3 Pro；DeepSeek-R1 1.5B/7B/14B，1000 条 H3 查询。单设备 RTX4060Ti 1.5B：TTFT 1.02s、吞吐 120.4 tok/s；**跨设备 14B 最高 160s**；失败率 <5%；深圳 M3Pro 跑 7B 达 52.84s（**地理距离显著影响**）。
- **与本研究（D1–D6）**：
  - D1/D2 **直接命中（但浅）**：任务拆分=模型级 sharding（EdgeShard 顺序分片）；协作执行=跨异构设备流水线推理。
  - D3 **直接（粗粒度）**：API 按模型批量路由，**无负载感知调度**——可借鉴的空白。
  - D4 **部分/间接**：靠去中心化设计容忍节点异构与故障（失败率 <5%），但**未做动态装载卸载/分片迁移/弹性伸缩机制**。
  - D5 **间接**：20 台异质设备的实测延迟/吞吐表本身就是"多型号 GPU 跑 LLM 推理"的评估数据集。
  - D6 **直接**：跨港/广/深/金泽真实部署，明证地理距离对延迟/吞吐的影响。
  - **可借鉴**：按模型批量路由 + worker 自主认领的简化分配模型；EdgeShard 顺序分片做跨设备推理；VRF + 中位数共识的轻量去中心化机制；实测异构设备性能表。**不宜照搬**：调度粒度粗、无弹性迁移、激励依赖区块链（你已排除）。
- **版图定位**：生态顶点，直接复用 EdgeShard，把"分片/协作机制（EdgeShard）→ 调度/卸载策略（任务卸载、Janus）→ 去中心化平台（PolyLink）→ 真实部署评估"串成完整链条。**它证明了场景的真实性，也暴露了"动态调度+弹性装载卸载"这一研究空白**。

---

### 组 G：未下载论文（3 篇，简述）

| 论文 | 会议/年份 | 核心内容 | 与本研究 |
|---|---|---|---|
| Scalable Graph-based RAG via Locality-Sensitive Hashing | VLDB 2025（LLM+Graph 研讨会） | 用**超平面 LSH** 替代 GMM 做可扩展 Graph-RAG：图构建→索引→检索增强生成；分层树构建流水线，比精确检索更快、更低内存 | 间接。RAG 属 LLM 推理应用侧；与"如何让 LLM 在受限算力上检索"相关 |
| Explicitly Guided Difficulty-Controllable Visual Question Generation | AAAI 2025 | 可控难度的视觉问答生成 | 无关 |
| Efficient Robustness Evaluation via Constraint Relaxation | AAAI 2025 | 约束松弛做高效鲁棒性评估 | 无关 |

> 两篇 AAAI 论文与本研究无关，来自合作者非核心方向；VLDB Graph-RAG 为研讨会短文。如需全文：AAAI 两篇在 ojs.aaai.org 开放获取，Graph-RAG 可查 vldb.org workshop 页面。

---

## 二、整合洞察

### 2.1 这 33 篇构成一个什么样的"研究版图"

```
                     [PolyLink 平台]  ←  去中心化 + 区块链 + 可信评估（F4）
                          ↑ 复用
   [CEI 愿景（A6）] ←——→ [EdgeShard 切分（A1）]  ←——  [Janus 剪枝切分（A4）]
   联邦资源池·任务迁移         LLM 层切分+DP 调度       ViT token剪枝+动态调度
                          ↑
   [任务卸载/分配 RL：O2O-DRL（A2）→ HRTO 混合冗余（A3）]
   [分布式学习：DistMatch（B3）、PASTA 流水并行（B4）、FCIDF（B2）、DP-FL 公平（B1）]
   [LLM 系统侧翼：TaskSense（C2）、Multi-Agent Resilience（C4）、SocialMind（C1）、COCA（C3）]
   [调度 RL 模板：JPDR（D4）、TSN（D5）、JIT-TP（D2）]
   [支撑：CSUR 综述（A5）、交通预测基准（D3）、安全评测（组 E）、多媒体（组 F）]
```

三条主脉：
1. **边缘协作算力脉**（A 组为主）：从 CEI 愿景 → EdgeShard 落地 → PolyLink 平台，是本研究的直接对话对象。
2. **调度方法论脉**（A2/A3 + D4/D5 + B4）：DP 精确解 vs DRL 自适应（SAC/MAPPO/PPO）+ 冗余/容错 + 在线微调安全兜底，构成"任务分配与弹性"的方法库。
3. **LLM 编排与鲁棒性脉**（C 组）：LLM-as-orchestrator + 形式化能力描述 + 故障弹性度量，是"异构单元协作 + 节点动态"的软件层先例。

### 2.2 方法共性（可复用的"配方"）

1. **可靠性进模型而非事后**：O2O/HRTO 把泊松失效建模进决策与奖励；PolyLink 用 TIQE 评估推理质量；Resilience 用故障注入度量韧性。
2. **离线预训练 + 在线微调 + 安全兜底**：O2O-DRL、TSN ORSJS、JPDR（预训练后在线微调）、SocialMind 缓存快路径——面对环境漂移/节点动态的统一配方。
3. **动态网络自适应**：Janus 带宽感知调度器、JIT-TP 过期检测、TSN 安全替代策略——以实时观测适配波动带宽。
4. **结构化中间表征约束开放问题**：传感器语言（C2）、蓝图图（F1）、融合率向量（B2）、锚定方向（F3）——给协作/调度加形式化约束。
5. **仿真器即 ground truth + 双验证**：SUMO/网约车仿真/NeSTiNg + K8s 测试床 / 真实异构原型（EdgeShard 15 台、PolyLink 20 台）。

### 2.3 与你的研究（D1–D6）映射与空白

| 子问题 | 可借鉴论文 | 缺口 |
|---|---|---|
| D1 任务拆分 | EdgeShard（层切分）、Janus（token剪枝切分）、PASTA（流水切分）、CSUR（并行模式全集） | 都针对**单一**推理/训练实例；无训练+推理统一弹性切分 |
| D2 协作执行 | PolyLink（跨设备流水）、DistMatch（共识协调）、Resilience（结构拓扑选择） | 通信受限下的协作执行、慢网络下的状态同步 |
| D3 任务分配/调度 | O2O-DRL、HRTO、JPDR、TSN、JIT-TP（按需调度） | **无动态负载感知调度**；PolyLink 仅按模型批量路由 |
| D4 节点离开/加入 | HRTO（冗余）、Resilience（韧性+监督）、TSN（在线微调+安全替代）、TaskSense（动态计划适配） | **无快速装载/卸载与状态（KV/分片）迁移机制**——最大空白 |
| D5 新 GPU 评估 | EdgeShard/PolyLink（profiling+实测表）、WarpGen（oracle 判据） | **无"运行中新 GPU 在线评估+任务量分配"**——都是离线评估 |
| D6 非数据中心通信 | Janus（5G trace）、PolyLink（跨地域实测）、DistMatch（gossip）、PASTA（WAN 反馈） | 无"非数据中心+慢网络下的训练/推理统一通信协议" |

**三个研究空白（与 CCF-A 层分析相互印证）**：
1. **运行中新 GPU 的在线评估 + 负载分配**（现有全部离线评估：profiling/实测表）。
2. **异构边缘 + 无宽限期 + 慢网络下的快速迁移恢复**（HRTO 的冗余不迁移状态；PolyLink 不做分片迁移）。
3. **通信受限下的训练与推理统一弹性切分**（PASTA 只训练、EdgeShard 只推理；无统一框架）。

### 2.4 对本研究写作与定位的直接用处

- **PolyLink 平台论文** = 最直接的系统先例（同领域、同平台级）：它的"按模型批量路由 + 无弹性迁移"短板，就是你的 D3/D4 切入点。
- **EdgeShard** = 推理侧任务拆分/分配的直接 baseline；它的 DP 精确解可与你的在线/动态方案对照。
- **PASTA** = 训练侧流水切分 + 反馈式自适应分配的方法论样板（可迁移到 LLM 训练流水）。
- **Multi-Agent Resilience + TaskSense** = 弹性/故障与"能力描述语言"的软件层先例，支持你论证"中心化/层次调度更抗单点故障"。
- **TSN ORSJS + O2O-DRL** = 在线调度器安全兜底与冷启动解决方案（"离线预训练+在线微调"叙事）。
- **交通预测系列（JIT-TP）** = 支持"按需计算/结果复用/把调度问题建模为数据管理问题"这一写作角度。

---

## 三、文件与后续

- 30 篇 sidecar 全文：`refs/papers/parsed/`（本分析基于它们）
- 3 篇未下载：获取途径见组 G 说明
- 与 CCF-A 层文献的关系：见 `research-alignment.md`（CCF-A 侧重数据中心集群调度/LLM 服务系统；本文侧重边缘协作/去中心化场景，两者互补构成完整坐标系）
