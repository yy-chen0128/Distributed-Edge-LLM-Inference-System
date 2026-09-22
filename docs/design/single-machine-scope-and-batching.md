# 单机范围、批处理/微批，以及"框架开销占多少"的实测答案

> 这份文档回答四个问题：
> **(1) 我们实际测的是什么**（请求数、并发、batch 大小、prompt/输出长度、"实际测试情况"）；
> **(2) "绝大部分时间是框架开销"到底是什么意思，对什么模型成立**；
> **(3) 批处理 / 微批要做什么、能换来什么**；
> **(4) 只有一台机器时还能做什么有意义的工作**。
>
> §3.5 包含一处对本文档早期版本的**更正**：四机那组 prefill 数字是冷启动，不是稳态。
> 所有数字都可在本机复现，命令见 §10。

---

## 0. 结论先行

| 问题 | 答案 |
|---|---|
| 我们的真实运行用过 batch 多大？ | **全部是 batch = 1**（一个 stage 调用处理一个请求）。并发 K 是"在飞请求数"，不是批大小 |
| "框架开销占绝大部分"成立吗？ | **对 0.5B + batch=1 成立（prefill 92%、decode ~72%）**；对 7B 在 38 token 时降到 **42%**，512 token 时 **5%** |
| 是不是"推理负载很低"？ | **是，但那是我们选的配置造成的**——模型小 17 倍、batch=1、prompt 只有 38 token。真实负载下不成立 |
| 批处理能换多少？ | 实测"K 个 token 一次调用"vs"K 次单 token 调用"：K=8 → **8.9×**，K=32 → **28.6×**（上界口径，§4） |
| 现在最该做什么？ | ① 换成真实数据集（`docs/research/real-workload-datasets.md`）；② 在真实长度下重测 §3 的曲线 |

---

## 1. 我们到底测了什么：实测口径表（澄清"实际测试情况"）

**关键事实：至今没有任何一次真实推理运行使用过 batch > 1。**

| 实验 | 硬件 | 模型 | 请求数 / 并发 | **batch** | prompt | 输出 | 出处 |
|---|---|---|---|---|---|---|---|
| 策略仿真 | 无（MockEngine） | 无 | **30 个请求**（3 前缀 × 10，cold 0.2） | — | 合成文本（`shared_context_*`） | — | `run_simulation.py` |
| 分层控制流仿真 | 无（LayeredMockEngine） | 12 层虚构模型 | 1 | — | — | 8 | `run_layered_simulation.py` |
| 并发/吞吐扫描 | **CPU**（24 核，fp32） | Qwen2.5-0.5B | K = 1/2/4（**每档 K 个线程**） | **1** | 1 个固定 prompt | 8 | `real_pipeline_conc.json` |
| 四段流水线（单请求） | **GPU**（4 进程共用 cuda:0） | Qwen2.5-0.5B | **1** | **1** | **38 token** | **6** | `wsl_gpu_pipeline.json` |
| 分段 vs 整体数值等价 | CPU/GPU | Qwen2.5-0.5B | 1 | 1 | 38 token | 6 | `equiv_4stage.json` |
| 节点离开重配置 | CPU | Qwen2.5-0.5B | 2 | 1 | 38 token | 8 | `real_pipeline_hetero.json` |

即使"并发 K=4"那次，也是 **4 个线程各发 1 个请求**，每个 stage 的每次调用仍然只处理 1 个请求——
并发提高的是"流水线里同时有几个请求在飞"，不是"一次 forward 里塞了几个请求"。**这两件事完全不同**（§2）。

还有两点必须说清楚：

- **所有真实运行只用了同一个 prompt**（"用一句话解释什么是流水线并行。"，38 token），输出 6–8 token。
  这比真实负载小 **1–3 个数量级**：Azure code 服务 prompt 中位数 1500 token / 输出 13；
  Mooncake 平均 7590 / 182；Codex agentic 单次调用平均 68,329 / 520。
  在 38 token 这个尺度上得出的"切层比例""气泡比例""批处理收益"**没有迁移价值**。
- **四个 agent 现在共用同一块 GPU**（`wsl_gpu_pipeline.json` 里 n0–n3 的 `device` 全是 `cuda:0`，
  `vram_total` 都是 8.6GB）。所以"四段流水线"在单机上不等于四台机器，单机吞吐不能外推。

---

## 2. 先把三个词分开

记号：流水线 $S$ 段（我们 $S=4$），第 $i$ 段处理一个请求一个 token 的耗时 $t_i$，$t=\max_i t_i$。

| 概念 | 改的是哪个量 | 一句话 | 单机可行 |
|---|---|---|---|
| **批处理**（batching） | 改 $t$ 的**构成** | 把 $K$ 个请求塞进**一次** forward，让每次调用的固定开销被 $K$ 个请求分摊 | ✅ |
| **微批**（micro-batch） | 改**利用率** | 让 $S$ 段同时都有活干，消掉气泡；$t$ 不变 | ✅ |
| **DP**（数据并行） | 改**并行度** | 复制整条流水线 | ✅（单机无意义） |
| **TP**（张量并行） | 改 $t$，但每层要同步 | 一个矩阵乘拆到多卡 | ❌ |

- DP 是"多修几条流水线"；微批是"让一条流水线别闲着"；批处理是"让每段每次多干几个请求"。
- 微批的天花板是 $1/t$（$U = K/(K+S-1)$，$K=S=4$ 时只有 0.57），**它消不掉 $t$ 里的固定开销**。
- 我们实测的 K=1→4 吞吐 5.06→7.82 tok/s（1.55×）就是微批的收益，而那是 **CPU fp32** 下测的。

---

## 3. 固定开销 vs 模型计算：实测（核心）

### 3.1 方法

在**一个进程、一个 stage、一块 GPU** 上，用 `scripts/measure_stage_scaling.py` 测：

1. **prefill 曲线**：一次调用携带 $P$ 个 token（$P=1\ldots512$），拟合 $T(P)=c_0+c_1P$。
   $c_0$ = 每次调用的固定开销，$c_1$ = 每 token 边际成本。
2. **decode 稳态**：先 prefill 38 token，再连续 20 次单 token 调用，取中位数。
3. **调用粒度**：把 $K$ 个 token 放进**一次**调用 vs 分 $K$ 次单 token 调用（§4）。

**计时必须 `torch.cuda.synchronize()`**：CUDA 是异步的，不 fence 就只测到入队时间。
这一点在本文档早期版本里做错了，§3.5 一并更正。

### 3.2 结果（RTX 4060 Laptop 8GB，Qwen2.5-0.5B，bf16）

| 阶段 | $c_0$（ms/次调用） | $c_1$（ms/token） | $R^2$ | P=38 时开销占比 | decode 单 token |
|---|---|---|---|---|---|
| 6 层（0..6） | **5.82** | 0.0165 | 0.91 | **90.3%** | 5.12 ms |
| 24 层（全模型） | **23.81** | 0.0369 | 0.67 | **94.4%** | 19.22 ms |

**关键：$c_0$ 随层数线性增长，两次测量高度一致**：5.82/6 = 0.97 ms/层；23.81/24 = 0.99 ms/层。
所以这个"固定开销"不是一个常数，而是**每层每次调用的启动开销**（kernel launch + Python/框架 dispatch +
张量分配），与模型大小无关、与 batch 无关：

$$\boxed{T_{\text{call}} \approx 0.97\text{–}0.99\ \text{ms}\times L + c_1\cdot P}\qquad L=\text{本段层数}$$

每层权重（由两个 stage 尺寸差分得到，抵消只出现在首段的词表）：**29.8 MB/层**；
首段额外背一份 **272.3 MB 的词表**（embedding，绑定时末段共用）。

### 3.3 三个可直接用的数

| 量 | 数值 | 含义 |
|---|---|---|
| 每层每 token 边际成本（0.5B） | **≈0.00214 ms** | 真正的模型计算 |
| 每层每次调用固定成本 | **≈0.99 ms** | 无论 batch、无论 token 数都要付 |
| 盈亏平衡 prompt 长度 | **0.5B ≈ 463 token；7B ≈ 30 token** | 固定开销 = 模型计算 的临界点 |

### 3.4 对更大模型的投影（按每层权重大小线性缩放）

Qwen2.5-7B-Instruct 每层权重 **466.0 MB**（hidden 3584、intermediate 18944、
**GQA 只有 4 个 KV 头**、head_dim 128 → 注意力 29.4 M + MLP 203.7 M ≈ 233.0 M 参数，bf16）
= 0.5B 的 **15.6 倍**（28 层）。
固定开销 $c_0$ 不变，边际成本 ×15.6：

| prompt 长度 | 0.5B 开销占比 | **7B 开销占比** |
|---|---|---|
| 38 token（我们现在的测试） | 92.4% | **43.5%** |
| 512 token | 47.5% | **5.4%** |
| 4096 token | 10.2% | **0.7%** |

投影是**估算**（假设 7B 在同样的框架、同样的每层效率下运行，且只考虑 dense 权重），
但它足以说明：**"框架开销占绝大部分"是 0.5B + 38 token 这个组合的性质，不是推理系统的性质。**

### 3.5 更正：四机那组数（485–947 ms/段 prefill）是冷启动

本文档早期版本把 `wsl_gpu_pipeline.json` 里 n0–n3 的 prefill compute（946.9 / 501.4 / 485.0 / 573.0 ms）
当作稳态计算量，由此推出"GPU 利用率 0.03%"。**这个推理是错的，两点更正：**

1. **那组数是冷启动。** 四个 agent 是**四个新进程**，每个进程只有 1 次 prefill：付了 CUDA context 创建、
   cuBLAS/cuDNN handle 初始化、kernel module 加载、450MB 权重首次上卡的全部一次性成本。
   隔离测量（5 次预热后取中位数、带 synchronize）显示：同样 6 层、同样 38 token 的 prefill 只要
   **≈5.5 ms**。也就是说那 485–947 ms 里有 **99% 是一次性开销**。
   （decode 侧同样可见：首步 compute 33/20/26/80 ms，稳态 5.4/6.0/4.8/6.9 ms。）
2. **`hop ≈ compute` 不是开销证据。** `hop_ms` 的定义是 `stage.client.call(...)` 的**整段墙钟**
   （`run_real_pipeline.py:301-302`），它**本来就包含** agent 侧的排队+计算+序列化+网络。
   真正的链路+序列化开销是 `hop − compute − queue`：稳态 **0.7–1.1 ms/次**，
   而首次 prefill 约 **60 ms**（连接/线程池/首次序列化）。
   顺带一提：`measure_link.py` 测得的 loopback RTT 只有 0.141 ms，所以这 0.7–1.1 ms 主要是 Python 侧序列化。

**更正后正确的说法**：稳态下，一次调用里"固定开销 : 模型计算"≈ 5.8 : 0.6（6 层、38 token），
固定开销占 ~90%；**而不是"GPU 利用率 0.03%"**。0.03% 那个数把冷启动算进去了。

> 副作用提醒：`run_real_pipeline.py` 目前每个请求只做 1 次 prefill，所以**任何用它做的 GPU 实验
> 都在测冷启动**。要测稳态必须在请求前加预热轮，或把"进程启动 + 首次调用"单独记为 `warmup_ms`。

### 3.6 回答"是不是说明推理负载很低"

**是，但这是配置造成的，必须分两种情况说：**

- **prefill**：模型计算随 prompt 长度线性增长，固定开销不变 → 长度够长时计算占主导。
  盈亏平衡点是 **0.5B ≈ 463 token、7B ≈ 27 token**。我们测试用的 38 token 对 0.5B 来说
  远低于平衡点，所以看到 92% 是开销；换成 7B 或真实长度的 prompt，这个比例立刻反转（§3.4）。
- **decode**：每次调用只有 1 个 token，模型计算 $\approx 0.002\times L$ ms 可以忽略；
  但 decode 的真实下界是"读完本段权重"：6 层 179 MB，4060 Laptop（~256 GB/s）约 **0.7 ms**，
  而实测 **5.1 ms** → 仍高出 **7 倍**，差值主要就是每层 ~0.9 ms 的启动开销。
  **所以 decode 在 batch=1 时永远是"开销/带宽"主导，与 prompt 长度无关——这正是批处理的用武之地。**

### 3.7 显存里的参数与 KV：实现到什么程度、实测多少

**参数（已实现、已实测）**

- `prepare_epoch` 用 `torch.device("meta")` 建骨架、只把本段权重加载到真实设备（`_materialize_shard`），
  `param_bytes` 与 `torch.cuda.memory_allocated` 都上报。实测：6 层驻留 **451.2 MB**
  （其中首段额外背 **272.3 MB** 词表）、24 层 **988.1 MB**、每层 **29.8 MB**（差分测得）。
- **没有的**：没有参数换入换出（offload/evict）、没有量化、没有跨进程共享
  （4 个 agent = 4 个 CUDA context，词表在首段那边各占一份）、只在 `retire_epoch` 时释放。
- 四段同机跑：4×6 层 ≈ 716 MB + 首段词表 272 MB ≈ **988 MB 权重**，8.6GB 卡放得下，但没有余量。

**KV（模型确实在存；计量曾经是坏的，已修）**

- 结构：每个 `(epoch, request_id)` 一个 `DynamicCache`，按请求隔离；`release_request` 释放；
  epoch 退役时清理。
- **实测（GPU，6 层 stage，38 token）**：真实 KV = **116,736 B = 0.117 MB**
  （= 2 × 2 kv_heads × 64 head_dim × 2 B × 6 层 × 38 token，逐字节吻合）；
  5 次 decode 后 **132,096 B**（正好 +5×3072 B）；`release_request` 后归 0。
- **整条四段流水线**：每段各持自己 6 层的 KV，四段合计 **0.47 MB**（24 层 × 38 token × 512 B）。
  4k 上下文即 24 × 4096 × 512 = **50.3 MB**，随上下文线性增长——这是切层时必须进的预算项。
- **坑 1（已修）**：`kv_bytes` 原来写 `cache[i]`，而 transformers 5.x 的 `DynamicCache` **不可下标**
  （`TypeError: 'DynamicCache' object is not subscriptable`），异常被 `except ... continue` 吞掉后
  **恒返回 0**——四机运行里每个 stage 的 `kv_bytes_last: 0` 就是这个原因。真实路径是
  `cache.layers[i].keys/.values`。回归测试：`tests/test_kv_accounting.py`。
- **坑 2（写进纪律）**：**不能用 `torch.cuda.memory_allocated` 的增量当 KV**：同一次调用的增量是
  **8.64 MB，为真实 KV 的 74 倍**（前向中间张量 + 分配器池），而且 `release_request` 之后**不降**。
- **没有的**：**没有分页/块管理**（HF 的 cache 是每序列连续的，不是 PagedAttention）、
  **没有按 KV 预算的准入控制**（`memory_free` 只看显存，没算 KV 增长）、
  **没有跨机 KV 传输**（这是弹性与跨节点恢复的前置，见 §6）。

### 3.8 权重放 `/mnt/d` 还是 WSL 的 ext4：实测差 20 倍

| 读取对象 | 冷读（真丢 page cache） | 热读 |
|---|---|---|
| `/mnt/d`（DrvFs/9p，Windows D 盘） | **112.7 MiB/s** | **114.5 MiB/s** |
| ext4（WSL 内部） | **2319 MiB/s** | — |

（`scripts/measure_fs_read.py`：读 384 MiB，`posix_fadvise(DONTNEED)` 真丢缓存，3–4 次取中位数。）

三个结论：

1. **差 20 倍**，而且 **`/mnt/d` 热读和冷读一样慢**——Windows 侧缓存透过 9p **不生效**，
   所以每次 epoch 切换都要付全价，不能指望"第二次就快了"。
2. 这把 `prepare_epoch` 的成本解释清楚了：0.5B 每段 451 MB ÷ 110 MiB/s ≈ **4.1 s**，与实测
   3.3–8.2 s 同量级。`prepare` = **读盘 + dtype 转换 + 上卡**，其中读盘是主项。
3. **对 7B 的后果**：按 8/6/4/4 加权切分，最大一段 12 层 = 12 × 466 MB ≈ **5.6 GB**
   → 从 `/mnt/d` 读约 **52 s**，每次重配置都要付。这对"节点离开后快速重配置"是致命的。
   可选对策（都还需实测）：① 权重挪到 ext4（2.3 GiB/s → 约 2.7 s，代价是撑大 vhdx，
   而 vhdx 在 C 盘）；② 量化到 int4（体积 ÷4 → 约 15 s）；③ 预转换单文件、砍掉 dtype 转换那一段；
   ④ 让"读盘"与"继续服务旧 epoch"重叠（已经是"先 prepare 再 activate"，但节点突然离开时没有可重叠的窗口）。

> **你说的"参数载入显存后运行时不读盘，所以是一次性开销"是对的。**
> 但要注意两点：它是**每次节点集合变化**的一次性开销，所以它决定的是**重配置延迟**这个指标；
> 而"要把 7B 放到 4 台机器上"这件事本身会让这个一次性开销从 4 s 级变成 1 分钟级。

---

## 4. 批处理能换来什么（实测上界）

`scripts/measure_stage_scaling.py` 的"调用粒度"实验（6 层 stage）：把 $K$ 个 token 放进**一次**调用，
对比分 $K$ 次单 token 调用。

| K | 1 次调用（ms） | K 次调用（ms） | 加速比 | 一次调用每 token | K 次调用每 token |
|---|---|---|---|---|---|
| 2 | 5.87 | 11.02 | 1.9× | 2.94 | 5.51 |
| 4 | 5.31 | 21.10 | 4.0× | 1.33 | 5.28 |
| 8 | 5.00 | 44.51 | **8.9×** | 0.63 | 5.56 |
| 16 | 5.95 | 91.79 | **15.4×** | 0.37 | 5.74 |
| 32 | 5.88 | 168.11 | **28.6×** | 0.18 | 5.25 |
| 64 | 5.40 | 363.21 | **67.2×** | 0.084 | 5.68 |

读法：**一次调用的耗时在 K≤32 时基本不变（≈5.0–5.9 ms）**——因为它是按"层数 × 每层启动开销"计的，
与这批里放多少 token 几乎无关。所以每请求成本 ≈ $5.9/K + 0.002$。

**这是上界，不是精确预测**（必须标注）：
真实的批处理是 $K$ 个**独立序列**各推进一个 token，attention 要在各自的长上下文上做（KV 读取随上下文增长），
而本实验是单序列 $K$ 个 token 的 prefill（自注意力长度只有 $K$）。所以上式只在"上下文短、$K$ 不大"时接近；
上下文一长，真实批处理的收益会被 KV 读取摊薄。**要拿到真实数字只能实现它。**

**结论**：批处理能动的量（每层每次调用的启动开销，约 5.9 ms/次/6 层）比微批能动的量（气泡，实测只到 1.55×）
大一个量级。这就是"单机最该先做批处理"的量化理由。

---

## 5. 单机上还能做什么（按性价比）

| # | 工作 | 目的 / 产出 | 成本 | 单机是否够 |
|---|---|---|---|---|
| 1 | **换成真实数据集**（§9） | 把 38/6 换成 1K–8K/13–130 两档，重跑 §3 曲线 | 1–2 天 | ✅ |
| 2 | **单进程批处理原型** | 让一个 stage 一次处理 $K$ 个请求，验证数值正确 + 拿真实加速比 | 2–4 天 | ✅（机制），❌四机吞吐 |
| 3 | **给 `run_real_pipeline` 加预热 + `warmup_ms` 记账** | 修掉 §3.5 的口径问题，否则所有 GPU 数字都是冷启动 | 半天 | ✅ |
| 4 | **loopback 两进程真实 KV 迁移** | 接 `PriorityMigration`，把 KV 真搬一次，量字节与耗时 | 2–3 天 | ✅ |
| 5 | **量化档位 × 每层耗时表** | fp16/int8/int4 的每层 compute 与每层权重字节 → 喂给"加权切层"（现在喂的是 TFLOPS 标称值） | 1–2 天 | ✅ |
| 6 | **KV 驱逐/选择原型** | 只让部分 KV 参与注意力（长上下文权衡） | 3–5 天 | ✅ |
| 7 | **重配置原子性 / 故障注入** | epoch 切换不丢请求；`TokenRecovery` 跨节点路径语义闭合 | 2–3 天 | ✅ |
| 8 | **指标统一** | 每段 compute/queue/hop/字节 + KV + 吞吐自动出图 | 2 天 | ✅ |

**建议顺序：1 → 3 → 2 → 4 → 5。**
第 1 项最关键：**在 38 token 上做的任何优化结论都不可迁移**（§1）。
第 3 项是半天的事，但不做，后面所有 GPU 数字都还是冷启动。

---

## 6. 批处理要改什么

### 6.1 引擎侧（`hf_layered_engine.py`）

现状 `prefill(request_id, token_ids)` / `decode(request_id, token_id)` 一次一个请求，
per-request 的 `past_key_values` 已按 `(epoch, request_id)` 存在 `self._caches` —— **结构天然支持批处理**。

| 要加 | 说明 |
|---|---|
| `prefill_batch(reqs)` / `decode_batch(reqs)` | 一次 forward 处理 $K$ 个请求，各自 cache 通过 `past_key_values` 列表传入 |
| padding / position_ids / attention_mask | 变长序列必须处理（右侧 pad + mask，或 varlen 接口） |
| 输出拆分 | 返回 `{request_id: hidden}`；末段返回每请求各自的 token |
| 批内 epoch 一致 | **禁止跨 epoch 混批**（cache key 带 epoch）；现在靠 `generate` 开头的 epoch 校验挡住 |
| 显存准入 | KV 随 $K$ 线性增长，$K$ 上界由 `memory_free` 决定 |

### 6.2 数据面协议（`ActivationEnvelope`）

- **A. 一个 envelope 装一批**：省掉 $K-1$ 次往返，但要重定义校验和与 epoch 语义（**唯一有设计风险处**）；
- **B. 每请求一个 envelope、一次发多个**：协议不动，只省往返。

稳态 hop 开销只有 0.7–1.1 ms，**先做 B 收益有限，但要拿到大头必须做 A**。

### 6.3 控制器侧

攒批调度器：每段待处理队列 + 三个闸门（`max_batch_size` / `max_wait_ms` / `max_batch_tokens`）。
注意**最慢段会换人**：批处理后各段 $t_i$ 变化幅度不同（首/末段还背着词表），切层要重新优化——
这正好需要第 5 项那张量化表。

### 6.4 验收口径

| 指标 | 口径 |
|---|---|
| 加速比 | 同一段、同一模型、同一 $K$ 个请求，$T(1)/T(K)$（单进程、无流水线） |
| 固定开销 | $T(K)=c_0+c_1K$ 拟合出的 $c_0$（ms/层/次） |
| 正确性 | 批处理输出 vs 逐请求输出逐 token 比对（复用 `verify_equivalence.py` 思路） |
| 延迟代价 | 单请求 TTFT / TPOT 相对基线不劣化 > 10% |
| 显存 | 每段 KV 字节随 $K$ 的斜率 |

### 6.5 微批（比批处理小）

调度器已能并发跑 $K$ 个请求；要改的是"按请求串行"→"按段排队 + 顺序推进"，
让 $S$ 段同时在途。需处理：同 epoch 约束（`LayeredPipelinePlacement` 已按 epoch 绑任务）、
公平性（FIFO + 每请求预算）、以及与批处理叠加后的吞吐上界 $1/\max_i t_i^{(K_{\text{batch}})}$。

---

## 7. 明确不做

- **不在单机做 TP**：实测每 token 需 ~56 次串行同步、跨 WiFi 单 token ~280 ms；
- **不用单机吞吐预测四机**：四个 agent 共用一块 GPU（§1）；
- **不改 `vllm/`、`LMCache/`**：不被本仓库管理（见 `.gitignore` 说明），以库/服务方式引用；
- **批处理原型里不换 attention 实现**：先把"摊薄固定开销"做干净，varlen 等属于算子层，等瓶颈明确后再说。

---

## 8. 最小动作（只做一件事的话）

**换成真实数据集，在真实长度上重测 §3 的两条曲线（prefill 的 $c_0/c_1$ 与 decode 稳态）。**

理由：现在这三个数（0.99 ms/层/次、0.00214 ms/token/层、拐点 463/27 token）都是 38 token、单 prompt 下测的。
换成 1K / 7.6K 两档 prompt 与 13 / 130 两档输出之后，"批处理值不值得做、切层怎么切、气泡占多少"
才第一次有了可迁移的依据。这也直接连到第 2、5 项。

---

## 9. 数据集

选型、字段核对、许可与坑见 **[`../research/real-workload-datasets.md`](../research/real-workload-datasets.md)**。
一句话：**Mooncake FAST'25 对话轨迹**（唯一带真实 512-token 前缀 block hash 的生产轨迹，1.4 MB）
+ **Azure LLM Inference Trace 2024**（一周真实到达 + code/conv 两档长度）
+ **LMCache agentic traces**（真实 prompt 文本 + session + 轮间 gap）。

---

## 10. 数据出处与复现

| 数字 | 出处 |
|---|---|
| 每层固定开销 0.99 ms、边际 0.00214 ms/token、拐点 463/27 token、粒度表 | `scripts/measure_stage_scaling.py`（本次实跑，RTX 4060 Laptop） |
| 参数驻留 451.2 MB/6 层、988.1 MB/24 层、29.8 MB/层、首段词表 272.3 MB | `scripts/measure_stage_scaling.py`、`hf_layered_engine.param_bytes` |
| KV 实测 116,736 B/6 层/38 token、每步 +3072 B、四段合计 0.47 MB | `scripts/probe_kv_accounting.py`、`probe_kv_api.py`、`.models/wsl_gpu_pipeline.json` |
| `/mnt/d` 冷读 112.7 / 热读 114.5 MiB/s；ext4 2319 MiB/s | `scripts/measure_fs_read.py` |
| 四段流水线 per-stage compute/hop、38 token、6 输出、`device=cuda:0`×4 | `.models/wsl_gpu_pipeline.json` |
| `hop_ms` 的定义（含 compute） | `edge_llm_scheduler/experiments/run_real_pipeline.py:301-302` |
| K=1/2/4 吞吐 5.06/6.54/7.82 tok/s（**CPU fp32**） | `.models/real_pipeline_conc.json` |
| loopback RTT 0.141 ms、单向 164.6 MB/s | `.models/link_loopback.json`、`experiments/measure_link.py` |
| 真实负载长度（Azure/Mooncake） | `docs/research/real-workload-datasets.md` |
| 策略行为与缺陷修复 | `docs/research/policies-principles-and-testability.md` |

```bash
# 固定开销 / 边际成本 / 调用粒度（需要一个 GPU）
PYTHONPATH=. python scripts/measure_stage_scaling.py

# 四段流水线（注意：每个请求只做 1 次 prefill → 冷启动口径）
bash scripts/wsl_run_pipeline.sh
```
