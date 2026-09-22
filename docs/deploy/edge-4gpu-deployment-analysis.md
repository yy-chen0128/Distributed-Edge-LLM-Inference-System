# 四张消费级笔记本 GPU 能做什么：真实部署与实测分析

> 结论先行版。本文所有"实测"数据都来自本机真实运行的端侧分层流水线
> （真实权重 Qwen2.5-0.5B-Instruct、真实 TCP、真实 KV cache、真实采样），
> 可复现命令见 §2 与附录 A。**尚未覆盖**的部分在 §1.4 明确列出，不做模糊表述。

---

## 1. 结论：这四张卡能干什么，不能干什么

### 1.1 能做的（按价值排序）

| # | 形态 | 为什么在你的硬件上成立 | 本次验证到什么程度 |
|---|---|---|---|
| 1 | **跨机流水线并行（PP）跑"单机装不下"的模型** | PP 每个 stage 边界只传 hidden states：7B 模型 decode 每 token 每边界仅 **7KB**（fp16），比 WiFi 能提供的带宽低两个数量级；而权重被切成 4 份后每台只需 3–4GB | **机制已跑通**：4 进程/真实 TCP 完成跨段推理，切层、激活传输、末段采样、结果正确（见 §3） |
| 2 | **PP + 并发请求提升吞吐** | 单请求在流水线里只有 1 段在工作；多请求交叠可把 4 段同时填满，吞吐上限≈最慢段的吞吐 | **实测 K=1→K=4 吞吐 5.06→7.82 tok/s**（同机 CPU 受限，4 台真机应更高） |
| 3 | **节点离开后的重配置与继续服务** | 真实笔记本会合盖、掉 WiFi、被抱走——这正是你的研究场景；重切层 + 新 epoch 已在真实进程上验证 | **实测 4→3 节点重切 2.15s、离开后请求正常、旧 epoch drain 0.76s** |
| 4 | **PD 分离（prefill/decode 分机）** | prefill 是计算密集、decode 是访存/KV 密集，两种机器的性价比不同；你们 `NodeRole.PREFILL/DECODE` 已有接口 | 机制可用（同一 PP 通道），本次未单独做 |
| 5 | **MoE 专家分片 / 专家分级存储（EP + offload）** | 4 台合计 16–24GB 显存装不下 14B+ 稠密模型的 fp16，但装得下 MoE 的**部分专家**；每 token 只激活 top-k | 未做；代价模型见 §4.4（all-to-all 是真正的带宽悬崖） |
| 6 | **KV 复用 / 前缀缓存 / 迁移** | 每台机器各自持有自己那几层的 KV（本次实现即如此），节点离开时按 `layer_range` **部分迁移**才是有意义的粒度 | KV 记账已实现（每段独立 KVBlock 语义）；跨机迁移未做 |
| 7 | **请求级路由/负载均衡（DP 基线，PAIR 那种）** | 小模型（0.5B–3B）每台都装得下 → 每台完整跑，代理按负载转发 | 可直接复用 PAIR 的思路与我们的 `E2Placement` |
| 8 | **真实故障注入平台** | 合盖睡眠、拔网线、抢显存、限速、限线程——消费级笔记本是最真实的不可靠环境 | 停机模拟已验证（§3.6） |

### 1.2 **做不到**的（必须在论文里说清楚，否则会被审稿人问倒）

| 做不到 | 原因（可量化） |
|---|---|
| **让单个请求更快（当模型单机装得下时）** | PP 不减少总计算量，逐 token 延迟 ≈ Σ(各段计算) + Σ(跨段通信)，与单机整跑相同**甚至更慢**。本次实测（同机同线程数）：4 段流水线 decode **147.5ms/token** vs 单节点整模型 **128.5ms/token**，即慢 15%（§3.5）。要用多卡降低单请求延迟，只能靠 **TP**；而 TP 在 WiFi 上不成立（§4.3）。 |
| **TP（张量并行）跨笔记本** | 每层都要一次 all-reduce 且是**逐层串行同步**：7B 两层 TP 时每 token 约 28×2 个同步点、每 token 通信量约 200KB，WiFi RTT 5ms 就是 **280ms/token 起步的纯等待**（§4.3）。TP 需要 NVLink/高速互联，不是笔记本 WiFi 的场景。 |
| **训练** | 本次实现与整个测试框架都只覆盖推理；训练的状态面（梯度/优化器/checkpoint）完全不在当前机制里。 |
| **承载 32B 以上的大模型** | 4 台按 8+6+4+4GB 算，可用显存约 17–19GB。32B int4 约 18GB（§4.1）已到极限，且 KV/激活还要挤进去。72B 级别直接排除。 |
| **单请求低延迟的在线服务（SLA 型）** | 消费级 GPU + WiFi：0.5B 模型 decode 约 147ms/token（CPU 实测；GPU 会快得多，但跨机同步与 OS 抖动仍在）。定位应是"能用、可扩展、可容错"，不是"低延迟"。 |

### 1.3 一句话定位

> 这四张卡的价值**不在算得快，而在够真实**：它们能真实制造"模型装不下、显存不够、
> 链路很慢、节点随时会走"这四类约束，而这正是你论文要解决的问题。用它们验证
> **"容量扩展 + 容错 + 吞吐"**（PP/EP/DP + KV），不要试图验证 **"单请求加速"**（那需要 TP + 高速互联）。

### 1.4 本次验证覆盖了什么 / 没覆盖什么（诚实边界）

| | 状态 |
|---|---|
| ✅ 真实预训练权重、真实 tokenizer、真实 KV、真实贪心采样 | 4 段切分与整体执行 **逐 token 完全一致，logits 最大绝对差 0.0** |
| ✅ 真实多进程 + 真实 TCP 的控制/数据协议（agent ↔ 控制器） | 4 个 agent 进程跨进程跑通（本机 loopback） |
| ✅ 节点离开 → 重新切层 → 新 epoch → 继续服务 | 已实测 |
| ✅ 每段显存/参数账本、activation 字节数、逐段耗时 | 已采集（作为切层与容量决策依据） |
| ⚠️ **GPU 执行**：本机 torch 是 `2.13.0+cpu`（无 CUDA 构建），本次是 **CPU fp32** | 代码已支持 `cuda:0` 与 fp16/bf16，但**GPU 数字需在四台笔记本上重跑**（§5） |
| ⚠️ **真实 WiFi 链路**：本机只有 loopback 实测（RTT 0.14ms、165MB/s） | 跨机 RTT/吞吐需用 `measure_link.py` 现场测（§3.9） |
| ⚠️ 四台机器未接入：我无法从本机访问那四台笔记本 | 提供了一键 agent 启动脚本与 cluster.json 模板 |

---

## 2. 本次真实部署记录

### 2.1 硬件与软件

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB，驱动 566.24，compute capability 8.9（Ada） |
| 本次实际使用 | **CPU**（torch 2.13.0+cpu，24 逻辑核）；GPU 路径代码就绪但本机无 CUDA 版 torch |
| Python / transformers | 3.12.6 / 5.9.0 |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct`：24 层、hidden 896、vocab 151936、**tie_word_embeddings=true**、checkpoint bf16 942MB |
| 网络 | 4 个 agent 进程 + 控制器，全部走 `127.0.0.1` 的**真实 TCP**（非进程内直调） |

### 2.2 真实部署要跑的三条命令（可复现）

```bash
# ① 数值正确性：分段执行 vs 整体执行
python -m edge_llm_scheduler.experiments.verify_equivalence \
    --model .models/Qwen2.5-0.5B-Instruct --stages 4 --max-tokens 8 \
    --json-out .models/equiv_4stage.json

# ② 真实跨进程流水线 + 节点离开 + 并发
python -m edge_llm_scheduler.experiments.run_real_pipeline --local 4 \
    --model .models/Qwen2.5-0.5B-Instruct --max-tokens 8 \
    --concurrency 1,2,4 --leave-node n3 --metrics-out .models/real_pipeline_full.json

# ③ 单机整模型基线（回答"PP 到底值不值"）
python -m edge_llm_scheduler.experiments.run_real_pipeline --local 1 \
    --model .models/Qwen2.5-0.5B-Instruct --max-tokens 8 \
    --metrics-out .models/real_pipeline_1node.json
```

四台笔记本上的等价操作见 [`runbook-4-laptops.md`](runbook-4-laptops.md)；
核心差异只是把 `--local 4` 换成 `--cluster deploy/cluster.example.json`。

---

## 3. 实测结果与分析

### 3.1 数值正确性：分段执行 ≡ 整体执行

| 指标 | 结果 |
|---|---|
| 切层 | `[0,6) [6,12) [12,18) [18,24)`（4 段平均） |
| prompt | 38 token |
| 生成 | 6 token |
| token 序列 | **完全相同**（`[86341, 32757, 62926, 22243, 101158, 62926]`） |
| 每步 logits 最大绝对差 | **0.0（6 步全部为 0，逐位一致）** |
| 输出文本 | `张量并行是一种并`（分段与整体两路一致） |

> 为什么能做到逐位一致：两路用的是同一份 fp32 权重、同一条贪心解码路径、
> 同一个因果掩码构造方式（直接复用 `transformers.masking_utils.create_causal_mask`），
> 且**没有跨机重排任何归约顺序**——PP 只搬运 hidden states，不改变数值语义。
> 这条结论很重要：它意味着**分段带来的误差为零**，任何差异都只能来自 dtype
> （如 fp16 传输）或工程实现，而不是并行度本身。

### 3.2 分片足迹：每台机器只装自己那几层

实测每段驻留参数字节（fp32）：

| 段 | 层区间 | 组成 | 参数字节 | 说明 |
|---|---|---|---|---|
| n0（首段） | 0–6 | embedding + 6 层 | **902.4 MB** | embedding = 151936×896×4 = 544MB |
| n1 | 6–12 | 6 层 | 357.9 MB | |
| n2 | 12–18 | 6 层 | 357.9 MB | |
| n3（末段） | 18–24 | 6 层 + norm + lm_head | **902.4 MB** | 绑定权重 → 末段又装了一份 embedding 当输出头 |

**这是本次最有工程价值的一条发现**：`tie_word_embeddings` 的模型在 PP 下，
**首段和末段各背一份 544MB 的词表**（fp32），占各自足迹的 60%；fp16 下是各 272MB。
后果与对策：

- 两端本来就该**少分层**（真实系统里 Megatron 类的 PP 也是这样做的）；
- 切层权重不能按显存比例简单线性分配，必须把**两端固定成本**单独算进去；
- 论文里可以把它作为"分级/重切层要考虑固定开销"的一个具体证据；进一步可做
  **vocab 维切分**（把词表也切开、末段只算自己那片 logits）来消掉这份冗余。

### 3.3 流水线时序与 activation 体积（4 节点 × 6 层，38 token prompt）

| 阶段 | 每段计算 | 跨段传输字节 | 每个 hop 的实测耗时差 |
|---|---|---|---|
| prefill（38 token） | 85–113 ms | **68,096 B（fp16）/ 136,192 B（fp32）** | hop − compute ≈ **1.5–2.6 ms**（loopback） |
| decode（每 token） | 31–57 ms | **1,792 B（fp16）/ 3,584 B（fp32）** | hop − compute ≈ **1.0–1.2 ms** |

关键比值：

- **activation : 权重 = 极小**。prefill 一次 68KB，decode 一 token 1.8KB；而每段权重
  是数百 MB。这就是"PP 适合弱链路"的定量依据。
- **fp16 传输把 payload 直接减半**（136KB→68KB、3.6KB→1.8KB），代价是数值精度；
  由于 §3.1 已证明分段本身零误差，fp16 传输的误差是可以单独评估、单独控制的一项。
- loopback 上每 hop 约 1–3ms；**跨机时还要加上线路 RTT 与传输时间**，用
  `measure_link.py` 实测后代入即可（工具会自动给出常见模型的估算表）。

### 3.4 从实测数据拟合出的成本模型（本机 CPU fp32，38 token）

用 12/6/3/3 的异构切层（§3.8）与 6/6/6/6 的数据拟合：

```
prefill 单段耗时 ≈ 20 ms(固定) + 11.5 ms × 层数        （末段再 +19 ms：norm+lm_head）
decode  单段耗时 ≈  0 ms(固定) +  5.1 ms × 层数        （末段再 +24 ms：norm+lm_head）
```

用途：**切层决策**。给定各节点实测的"每层耗时"（用同样方法在 GPU 上测一次），
令各段 `固定 + 每层×层数` 尽量相等，即可得到比"平均切"或"按显存比例切"更优的
切分点——这正是 `CapabilityReparallelization` 应该优化的目标函数，本文给出了它的
**参数标定方法与实测系数**。

### 3.5 关键结论：PP 不降低单请求延迟（实测对照）

同一台机器、**同样 6 个 torch 线程**、同一模型、同一 prompt、同样生成 8 token：

| 配置 | prefill | decode/token | 8 token 总耗时 | 吞吐 |
|---|---|---|---|---|
| **单节点整模型**（24 层在一个进程） | **369.0 ms** | **128.5 ms** | **1274.5 ms** | **6.28 tok/s** |
| **4 段流水线**（6 层 × 4 进程） | 430.8 ms（各段计算和 391.1 + 跨段 hop） | 147.5 ms（各段和 + hop） | 1493.1 ms | 5.36 tok/s |

结论：**分段比整跑慢约 15–17%**，慢的部分是"每段调用的固定开销 + 跨段通信"，
而不是计算本身。原因很朴素：流水线的逐 token 延迟 = Σ(各段计算)，而各段计算之和
就是整模型的计算量；切分只是把它摊到多个进程/机器上，**并没有减少总计算量**。

> ⚠️ 方法学提醒：我们第一版对照里 1 节点跑的是 24 线程、4 节点每段 6 线程，
> 结果出现了"分段反而更快"的假象。把线程数对齐后结论才稳定（上表）。
> 这类混淆在 GPU 上同样存在（每个 agent 的显存带宽/时钟/上下文开销），
> 四台真机做对照时必须统一 dtype、上下文长度、线程/算力占用与并发度。

> 所以：**PP 的收益是"容量"和"吞吐"，不是"单请求延迟"。** 这与 NVIDIA PAIR 文档里那句
> "Adding machines increases how many requests you can run at once. It does not make an
> individual request faster." 结论相同，但成因不同——PAIR 是请求级复制（DP），
> 我们是层切分（PP），两者都碰不到单请求延迟。要降低单请求延迟只有 TP/SP，
> 而那需要高速互联（§4.3）。
>
> 论文表述建议：把 PP 的动机写成"**装得下**"（单机放不下的模型）与
> "**吞吐扩展**"（并发请求填满流水线），不要写成"加速单个请求"。

### 3.6 节点中途离开：真实重配置

实测（4 节点 → 停掉 n3）：

| 步骤 | 实测 |
|---|---|
| 初始计划（epoch 1） | 6/6/6/6 层，4 台 prepare+activate 共 **8.67 s**（主要是各节点从 safetensors 读盘+转 dtype） |
| n3 离开后重切（epoch 2） | **8/8/8 层**，切换总耗时 **2.15 s**（3 台，且文件已在 OS 缓存） |
| 离开后的请求 | **成功**，8 token 用时 1681 ms（与离开前 1816 ms 同量级） |
| 旧 epoch drain | **762 ms**（`retire_epoch` 释放旧段参数与 KV） |
| 语义 | 新请求走新 epoch；旧 epoch 保留到 drain 完成才释放——与
`PipelineReconfigurationCoordinator` 的"先准备、再切流、最后退役"一致 |

要点：**重配置的耗时几乎全部是"装载分片"**（读盘 + dtype 转换 + 上卡），不是协议开销。
原因已实测清楚：权重放在 `/mnt/d`（Windows 盘）时**冷读只有 112.7 MiB/s、热读 114.5 MiB/s
（Windows 侧缓存透过 9p 不生效）**，而 ext4 内部是 **2319 MiB/s**，差 20 倍。
0.5B 每段 451 MB ÷ 110 MiB/s ≈ 4.1 s，与上面实测的 2.7–8.2 s 同量级。
对 7B 的后果：最大一段 12 层 ≈ 6.1 GB → **约 57 s**，每次重配置都要付。
测量方法与对策见 [`../design/single-machine-scope-and-batching.md`](../design/single-machine-scope-and-batching.md) §3.8。
这直接指向一个优化方向，也是论文可以做的实验：
**把"准备"做在后台、把"切换"压到毫秒级**（epoch 蓝绿），以及用**更小的分片/量化权重**降低装载时间。

### 3.7 并发与吞吐：气泡消除的实测效果

| 并发 K | 墙钟 | 生成 token | 吞吐 | 单请求平均延迟 |
|---|---|---|---|---|
| 1 | 1581 ms | 8 | **5.06 tok/s** | 1579 ms |
| 2 | 2447 ms | 16 | **6.54 tok/s**（1.29×） | 2379 ms |
| 4 | 4091 ms | 32 | **7.82 tok/s**（1.55×） | 3975 ms |

> **口径（重要）**：这组数据出自 `real_pipeline_conc.json`，其中 `device_kind = cpu`、`dtype = torch.float32`——
> 是**纯 CPU 运行**，不是 GPU。GPU 路径（`wsl_gpu_pipeline.json`）目前**只有单请求**数据，没有 K 扫描。
> 另外那次 GPU 运行里四个 agent 的 `device` **全是 `cuda:0`**（共用一块 8GB 卡），
> 所以"四段流水线"在单机上不等于四台机器。结论解读与单机边界见
> [`../design/single-machine-scope-and-batching.md`](../design/single-machine-scope-and-batching.md) §2.3–2.4。

- **吞吐确实随并发上升**：单请求时流水线里只有 1 段在工作（其余 3 段空转，即经典的
  PP 气泡）；多请求交叠后 4 段被填满。
- 但**远没到 4× 上限**，原因清楚且已由数据证明：4 个 agent 挤在一台 24 核 CPU 上争抢
  算力，且每段计算被串行化在各自锁内——agent 上报的 `queue_ms` 从 0 涨到 **356.9 ms**
  （K=4 时 n0 的 prefill 排队时间），这就是争用证据。
- **推论（四台真机上的预期）**：每台机器独占自己的 GPU 后，气泡消除应接近
  `吞吐上限 ≈ 1 / max(各段单token耗时)`。这也是四台笔记本最值得先做的一组实验。
- 想进一步压气泡，需要**连续批处理（continuous batching）**：让一段同时处理多个请求的
  micro-batch，而不是我们现在的"一段一次一个请求"。这也是后续工程项（§8）。

### 3.8 异构切层实验：为什么"平均切"是错的

故意按 12/6/3/3 切（模拟 4 台算力悬殊的机器）：

| 段 | 层数 | prefill 计算 | decode 计算 |
|---|---|---|---|
| n0 | 12 | **162.9 ms** | **60.9 ms** |
| n1 | 6 | 84.8 ms | 30.4 ms |
| n2 | 3 | 51.6 ms | 15.5 ms |
| n3 | 3 + norm/lm_head | 73.4 ms | 39.3 ms |

两个结论：

1. **最慢段决定吞吐**：稳态下流水线吞吐 ≈ 1/max(段耗时)，n0 的 12 层成为瓶颈，
   n1/n2 大量空等。→ 必须按实测能力切层（本框架的 `CapabilityReparallelization`）。
2. **末段不是"层数最少就最快"**：n3 只有 3 层，却因 norm+lm_head 比 3 层的 n2 慢
   47%（73.4 vs 51.6 ms）。→ 切层要按 §3.4 的"固定成本 + 每层成本"模型优化，
   而不是按层数或显存比例。

### 3.9 链路测量工具与基线

新增 `experiments/measure_link.py`（服务端 + 客户端）：测 RTT（中位/最小/最大/抖动）、
单向吞吐，并自动给出"常见模型规模下一次 activation 传输耗时（含 RTT）"的估算表。

本机 loopback 基线（用于验证工具，不代表 WiFi）：

| 指标 | 值 |
|---|---|
| RTT（64B 往返，50 次） | min 0.106 ms / **median 0.141 ms** / max 0.394 ms / σ 0.059 ms |
| 吞吐（1MiB × 256 帧） | **164.6 MB/s ≈ 1.32 Gbps**（loopback 受 CPU 序列化限制，非线路极限） |

四台笔记本要做的第一件事就是两两跑这个工具，把结果填进 §5 的切层权重。
（本机 `netsh wlan show interfaces` 读链路速率需要管理员权限，故未采集 WiFi 链路层参数。）

---

## 4. 四张消费级笔记本 GPU 的能力边界（定量）

### 4.1 选型表（按真实 config 计算，非估算口径）

权重按 `params × 2B`（fp16）/`× 0.55B`（int4 含 scale）；KV 按
`2 × 层数 × kv_heads × head_dim × 2B`（fp16）计算：

| 模型 | 层数 | hidden | kv heads | 参数量 | fp16 权重 | int4 权重 | KV/token | KV@8k ctx |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 24 | 896 | 2 | 0.63B | **1.26 GB** | 0.35 GB | 12 KB | 0.10 GB |
| Qwen2.5-1.5B | 28 | 1536 | 2 | 1.78B | **3.55 GB** | 0.98 GB | 28 KB | 0.23 GB |
| Qwen2.5-3B | 36 | 2048 | 2 | 3.40B | **6.79 GB** | 1.87 GB | 36 KB | 0.30 GB |
| Qwen2.5-7B | 28 | 3584 | 4 | 7.62B | **15.23 GB** | 4.19 GB | 56 KB | 0.47 GB |
| Qwen2.5-14B | 48 | 5120 | 8 | 14.77B | **29.54 GB** | 8.12 GB | 192 KB | 1.61 GB |
| Qwen2.5-32B | 64 | 5120 | 8 | 32.76B | **65.53 GB** | 18.02 GB | 256 KB | 2.15 GB |

MoE 参考（总量/激活量，按公开规格）：Qwen1.5-MoE-A2.7B ≈ 14.3B 总 / 2.7B 激活；
DeepSeek-V2-Lite ≈ 15.7B 总 / 2.4B 激活。MoE 的意义见 §4.4。

**按 8+6+4+4GB 四台（可用约 17–19GB）推算可承载**：

| 目标 | 方案 | 每台显存（权重/KV） | 结论 |
|---|---|---|---|
| 3B fp16 | 4 段 PP，每台 ~1.7GB + KV | 轻松 | 可行，但单机也能跑，价值在吞吐实验 |
| **7B fp16** | 4 段 PP，每台 ~3.8GB + 0.5GB KV | 可行 | ★ 首个"单机装不下（8GB 卡）"的真实案例 |
| **14B int4** | 2–4 段 PP，每台 ~2–4GB | 可行 | ★ 性价比最好的"大模型"实验 |
| 14B fp16 | 4 段，每台 ~7.4GB | 仅 8GB 卡勉强 | 可做，但 KV 余量小（长上下文会 OOM） |
| 32B int4 | 4 段，每台 ~4.5GB | 4GB 卡不可行 | 只在 8+6GB 两台 + 小上下文时考虑 |
| 32B fp16 / 72B | — | 超出 | 排除 |

> 注意：以上是**权重 + KV** 的静态预算。**activation/KV 峰值、CUDA context、
> PyTorch 碎片**还要留 1–2GB 余量；实机务必用 `-Dtype float16` 先跑通再加长上下文。

### 4.2 三段可行性计算的通用公式（把实测数字代进去即可）

```
① PP 每段边界每 token 传输量 = hidden × dtype_bytes           （decode）
                              hidden × prompt_tokens × dtype_bytes（prefill）
② PP 单请求逐 token 延迟     = Σ(各段计算) + Σ(跨段传输 + RTT)
③ PP 稳态吞吐上限           ≈ min over stages (1 / 该段单 token 耗时)   ← 最慢段决定
④ TP 每 token 通信量        = 2 × (N-1)/N × hidden × dtype_bytes × 层数 × 2（reduce-scatter+all-gather）
⑤ TP 每 token 同步点数      ≈ 2 × 层数（每层两次集合通信，且逐层串行）
```

### 4.3 为什么必须选 PP 而不是 TP（用 7B + WiFi 量化）

按 ④⑤ 代入 Qwen2.5-7B（28 层、hidden 3584、fp16）与典型笔记本 WiFi（RTT 5 ms、有效吞吐 40 MB/s）：

| 维度 | PP（4 段） | TP（2 路） |
|---|---|---|
| decode 每 token 通信量 | 3 个边界 × 7 KB = **21 KB** | 28 层 × 2 × 0.5 × 3584 × 2 ≈ **200 KB** |
| 每 token 同步点数 | 3（每个边界一次） | **≈ 56 次**（逐层串行） |
| 纯等待（RTT 5ms） | 15 ms | **280 ms** |
| prefill 512 token 通信量 | 3 × 3.7 MB ≈ 11 MB | 28 × 2 × 0.5 × 512 × 3584 × 2 ≈ **103 MB** |
| 是否降低单请求延迟 | **不降低**（Σ 段耗时 = 整模型耗时） | 降低（每层算力被分摊） |
| 结论 | ✅ 唯一可行 | ❌ 在 WiFi 上不可行；只在 NVLink/高速互联下才成立 |

这正好支撑你们汇报里的假设（"主要使用 PP，因为节点间带宽不足以支撑 TP"），
但现在是**有实测数字和公式支撑**的结论，而不是假设。

### 4.4 MoE / EP：真正的带宽悬崖，也是最有研究价值的点

- 4 台合计约 17–19GB 装不下 14B+ 稠密 fp16，但可以装下 MoE 的**部分专家**
  （Qwen1.5-MoE-A2.7B 总 14.3B，每 token 只激活 2.7B ≈ 5.4GB fp16 → 分片后每台 1–2GB）。
- 代价：每层一次 **all-to-all**（dispatch + combine 两趟）；每 token 通信量约
  `2 × hidden × top_k × experts_per_token × dtype × 层数`，是 PP 的**几十倍**，
  且同样是**逐层串行**的同步点。
- 因此四台笔记本上：**专家命中率 / 本地命中比例**直接决定成败——这与你们
  `E2Placement`（命中优先）和 SMoE（专家替换）的思路完全一致，而**这正是可以用
  这四张卡量出来的最关键曲线**：命中率 vs 每 token 延迟。
- 专家分级（GPU/CPU/盘）在笔记本上尤其自然：内存通常 16–32GB，比显存大 3–4 倍。

### 4.5 其他可跑的形态（成本低、故事完整）

| 形态 | 怎么做 | 能出的图 |
|---|---|---|
| **PD 分离** | 选算力最强的一台专做 prefill，其余做 decode；KV 从 P 机传到 D 机（`KVBlock.layer_range` 已有语义） | prefill/decode 分机 vs 混跑的 TTFT/TPOT 对比 |
| **DP 请求路由基线** | 每台装完整小模型，代理按负载路由（PAIR 范式 + 你们的 `E2Placement`） | 吞吐 vs 并发、KV 命中率收益 |
| **弹性 + 迁移** | 节点离开时按 `PriorityMigration`（reuse×prefill_time）迁移高价值 KV | 迁移量—恢复时间—重算成本的三角权衡 |
| **异构分级** | 一台无独显机器当慢节点/缓存节点，故意拖慢流水线 | 最慢段瓶颈证据、能力加权切层收益 |

---

## 5. 推荐的四机部署拓扑与切层方案

### 5.1 拓扑

```
              ┌─────────────── 控制面（任选一台，或第 5 台机器） ───────────────┐
              │  run_real_pipeline.py：能力探查 → 切层计划(epoch) → 驱动请求      │
              └───────┬───────────────┬───────────────┬───────────────┬────────┘
                      │ TCP 控制+激活  │               │               │
              ┌───────▼──────┐ ┌──────▼───────┐ ┌─────▼────────┐ ┌────▼─────────┐
              │ alpha 8GB    │ │ beta 6GB     │ │ gamma 4GB    │ │ delta CPU    │
              │ stage_agent  │ │ stage_agent  │ │ stage_agent  │ │ stage_agent  │
              │ 层 0..k      │ │ 层 k..m      │ │ 层 m..n      │ │ 层 n..末      │
              │ (+embed)     │ │              │ │              │ │ (+norm/head) │
              └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
                 真实 TCP / mTLS 可选；每台只需能被控制面与相邻段访问（9100）
```

### 5.2 切层权重怎么定（三步）

1. **测每层耗时**：先平均切一次，看各段 `compute_ms`；除以层数得各机"每层 ms"
   （含固定成本要按 §3.4 扣除）。
2. **扣掉两端固定成本**：首段 −embed、末段 −(norm+lm_head)——把这份固定开销先从
   该节点的预算里减掉，再分剩余层。
3. **按能力加权**：`层数_i ∝ 1/每层耗时_i`（不是按显存），填进 `cluster.json` 的
   `weight`。

### 5.3 显存预算模板（以 7B fp16 为例）

| 节点 | 层数 | 权重 | KV@4k | activation/工作区 | 小计 | 卡容量 |
|---|---|---|---|---|---|---|
| alpha | 8（含 embed） | 8×0.27GB + 0.31GB(embed fp16) ≈ 2.47GB | 0.23GB | ~0.5GB | ~3.2GB | 8GB ✅ |
| beta | 7 | 1.89GB | 0.23GB | ~0.5GB | ~2.6GB | 6GB ✅ |
| gamma | 6 | 1.62GB | 0.23GB | ~0.5GB | ~2.4GB | 4GB ⚠️ 需 fp16/小上下文 |
| delta | 7（含 head） | 1.89GB + 0.31GB(绑定词表) ≈ 2.20GB | 0.23GB | ~0.5GB | ~2.9GB | CPU（用内存） ✅ |

（每层权重 ≈ 7.62B/28 ≈ 0.27B 参数 ≈ 0.54GB fp16；表中按 4 段示意，实际以 §5.2 标定为准。）

---

## 6. 实验方案（直接对应论文图表）

### 6.1 必做的对照（baseline）

| 编号 | 配置 | 目的 |
|---|---|---|
| B0 | 单机整模型（8GB 卡，7B int4 或 3B fp16） | 单机上限 |
| B1 | 4 段平均切 PP | 朴素 PP |
| B2 | 能力加权切 PP（§5.2） | 证明切层策略有效 |
| B3 | PP + 并发（K=1,2,4,8） | 气泡消除/吞吐扩展 |
| B4 | DP 请求路由（每台整模型，小模型） | 与 PP 的形态对比 |
| B5 | 节点离开：无迁移 / 全量 KV 迁移 / `PriorityMigration` | 弹性的核心对照 |

### 6.2 关键指标（本次已全部有采集代码）

- 系统：吞吐（tok/s）、TTFT、TPOT（每 token 延迟）、p95/p99、成功率、
  **重配置耗时**、**离开后恢复时间**；
- 状态：**activation 字节数/段**、KV 字节数、每段显存、`queue_ms`（争用）；
- 策略：迁移字节与收益、命中率、最慢段空等比例（= max(stage)/Σ(stage)）。

### 6.3 建议优先跑的三组（按性价比）

1. **7B fp16 四段 PP**：第一个"真·容量扩展"证据（单机 8GB 装不下）+ B0/B1/B2 对照。
2. **PP + 并发扫描**：证明吞吐随并发上升、瓶颈段是哪台（`queue_ms` + 最慢段）。
3. **节点离开 + PriorityMigration**：`--leave-node` 已就绪，接上 KV 迁移策略即可出图。

---

## 7. 与 NVIDIA PAIR 的分工（我们做 PAIR 不做的那一层）

| | PAIR | 本项目 |
|---|---|---|
| 决策粒度 | 整个请求 → 某台整机（模型必须单机装得下） | 模型按层/专家切分 → 多机协同执行一个请求 |
| 并行维度 | 无 PP/TP/EP，只有请求级副本（DP） | PP（已实现真实运行）、EP、KV 分级 |
| 状态 | 不迁移、不共享 KV | KV 目录、按 `layer_range` 部分迁移、优先级迁移 |
| 链路 | 不建模带宽/延迟 | 把带宽/延迟作为切层与放置的输入（`measure_link` 提供实测） |
| 与我们的关系 | **可复用的请求路由/信任/成员控制面**（见 PAIR 分析文档） | **执行面**：PAIR 明确声明不做的那部分 |

务实建议：小模型（≤3B）用 PAIR 式路由做**服务入口与吞吐基线**；大模型/弹性实验用本
框架的 PP/EP 通道。两者不冲突，正好构成"入口 + 执行面"的完整故事。

---

## 8. 下一步工程优先级

| 优先级 | 事项 | 理由 | 预估工作量 |
|---|---|---|---|
| P0 | 在四台真机跑通本手册（GPU + WiFi），采集 §6.2 指标 | 目前所有 GPU/WiFi 数字都还是空白 | 半天 |
| P0 | `prepare_epoch` 的装载加速（预转 fp16 缓存、mmap、按需读） | 实测装载 2–8s 是重配置的主要成本 | 1–2 天 |
| P1 | 连续批处理（一段内 micro-batch） | 实测并发只到 1.55×，气泡仍未消完 | 3–5 天 |
| P1 | KV 跨机迁移（接 `LMCacheStore` + `PriorityMigration`） | 弹性实验的核心 | 3–5 天 |
| P1 | 词表切分（消掉首/末段各一份 embed） | 实测两头各 544MB(fp32)/272MB(fp16) 冗余 | 2–3 天 |
| P2 | PD 分离（prefill 机 + decode 机组） | 与 PP 正交，容易出新图 | 2–3 天 |
| P2 | MoE 专家分片 + 命中率实验 | 带宽悬崖最有论文价值 | 1 周 |
| P2 | 统一 metrics/exporter（roadmap §11.5） | 图表自动化 | 2 天 |

---

## 附录 A：命令与产物清单

| 命令 | 作用 |
|---|---|
| `scripts/hf_mirror_download.py` | 镜像直连下载模型（绕过 huggingface_hub 的 etag 问题） |
| `scripts/model_sizing_table.py` | 按真实 config 算权重/KV 显存表 |
| `edge_llm_scheduler/backends/hf_layered_engine.py` | **真实 HF 分层 stage 运行时**（meta 骨架 + 只加载本段权重 + 真实 KV） |
| `edge_llm_scheduler/agents/stage_agent.py` | 节点 agent 进程（TCP 控制面 + 数据面） |
| `edge_llm_scheduler/experiments/run_real_pipeline.py` | 控制器：探查/切层/epoch/请求/并发/节点离开 |
| `edge_llm_scheduler/experiments/verify_equivalence.py` | 分段 vs 整体数值一致性（阶段 D 验收） |
| `edge_llm_scheduler/experiments/measure_link.py` | 链路 RTT/吞吐实测 + activation 耗时估算 |
| `edge_llm_scheduler/deploy/cluster.example.json` | 四机部署配置模板 |
| `edge_llm_scheduler/deploy/start_agent.ps1` / `.sh` | 各机 agent 启动脚本 |

产物（本次已生成）：`.models/equiv_4stage.json`、`.models/real_pipeline_4stage.json`、
`.models/real_pipeline_full.json`、`.models/real_pipeline_1node.json`、
`.models/real_pipeline_hetero.json`、`.models/real_pipeline_conc.json`、
`.models/link_loopback.json`、`.models/logs/agent_*.log`。

## 附录 B：本文引用的实测数据速查

```
模型            Qwen2.5-0.5B-Instruct (24 层, hidden 896, tied embeddings, bf16 ckpt 942MB)
环境            Windows / Python 3.12.6 / torch 2.13.0+cpu / transformers 5.9.0 / CPU fp32 / 真实 TCP
数值一致性      4 段 vs 整体: token 全等, 每步 logits 最大绝对差 = 0.0
分片足迹(fp32)   [902.4, 357.9, 357.9, 902.4] MB   (首末段各含一份 544MB 词表)
prefill 传输     68,096 B/hop (fp16)  vs  136,192 B/hop (fp32)
decode 传输      1,792 B/token/hop (fp16)  vs  3,584 B/token/hop (fp32)
每 hop 开销      loopback 约 1–3 ms（hop − compute）
成本模型         prefill ≈ 20ms + 11.5ms×层数（末段 +19ms）；decode ≈ 5.1ms×层数（末段 +24ms）
单机整模型       prefill 369.0ms, decode 128.5ms/token, 6.28 tok/s（6 线程）
4 段流水线       各段计算和 prefill 391.1ms / decode 147.5ms；含 hop 后 430.8ms / 147.5ms；8 token 用时 1493ms
                 → 单请求比整跑慢约 15–17%（每段固定开销 + 跨段通信），这是 PP 的固有代价
节点离开         4→3 节点重切 2.15s；离开后请求成功；旧 epoch drain 0.76s
并发             K=1 5.06 → K=2 6.54 → K=4 7.82 tok/s；K=4 时 n0 排队 356.9ms
异构切层 12/6/3/3 prefill 162.9/84.8/51.6/73.4 ms；decode 60.9/30.4/15.5/39.3 ms
loopback 链路    RTT 中位 0.141ms；吞吐 164.6 MB/s
```
