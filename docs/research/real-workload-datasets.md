# 真实负载数据集：选型、字段核对与坑

> **为什么必须有这份文档**：我们现在全部真实实验都用**同一个 38-token prompt、输出 6 token**
> （见 `docs/design/single-machine-scope-and-batching.md` §1）。这个尺度比真实负载小 **1–3 个数量级**：
> Azure code 服务 prompt 中位数 **1500** token / 输出 **13**；Azure conversation **1020 / 129**；
> Mooncake 平均 **7590 / 182**（比值 ≈720:1）；Codex agentic 单次调用平均 **68,329 / 520**。
> **在这个尺度上得出的切层比例、气泡比例、批大小收益都不可迁移。**
>
> 本文所有结论都标注了来源；`[已核实]` = 抓到了页面/API 原文，`❓UNVERIFIED` = 无法确认，**不要当事实用**。

---

## 0. 结论先行

**最小可用组合（3 个，全部免申请、笔记本可处理）：**

| 用途 | 数据集 | 为什么是它 |
|---|---|---|
| 前缀共享 / KV 复用 **+** 长度分布 **+** 1 小时到达 | **Mooncake FAST'25 conversation trace** | **唯一公开的、带真实 512-token 前缀 block hash 的生产轨迹**；单文件 ~1.4 MB；vLLM 原生支持回放 |
| 到达 / 日周期 / 突发 **+** prompt-output 长度分布 | **Azure LLM Inference Trace 2024** | 免登录 CC-BY，一整周、两个语义不同的服务 |
| 真实 prompt 文本自算前缀共享 **+** 会话/轮间间隔 | **LMCache Agentic Traces** | 唯一同时给"真实 prompt 文本 + session_id + 轮间间隔 + output 长度"的小数据集 |

**覆盖面最广的单一数据集**：`eth-easl/swissai-serving-trace`（OSD'26 OpenTela）——
1632 万条请求，`created_at`/`finished_at` + 输入/输出 token 数 + 16-token `bucket_ids`（可直接算前缀复用）。
代价：总仓 21.4 GB，需挑文件下。

**我们需要的四类信号，各自的最佳来源：**

| 信号 | 用来驱动什么 | 首选 |
|---|---|---|
| (a) 到达模式（突发/日周期/并发） | 气泡、批处理、并发扫描 | Azure LLM 2024；BurstGPT（4 个月 + Session ID） |
| (b) 前缀共享 / KV 复用 | cache-aware 路由 | Mooncake（hash ground truth）；LMCache（真实文本） |
| (c) prompt : output 长度分布 | 切层比例、prefill/decode 资源配比 | Azure 2024（两档）；Mooncake（极端 prefill-heavy） |
| (d) churn / 会话结束 | 弹性、恢复 | **公开数据里没有**，只能用代理（见 §4） |

---

## 0.1 选型决定：哪些直接用、哪些明确不用

**直接用（第一轮就上）**

| 数据集 | 用途 | 动作 |
|---|---|---|
| Mooncake conversation trace | 前缀共享 ground truth + 长度分布 | 下 1.4 MB 单文件 |
| Azure LLM 2024（code + conv） | 到达模式 + 两档长度 | 下两个 CSV |
| LMCache agentic traces | 真实 prompt 文本 + session + 轮间 gap | 走 HF datasets |

**明确不用，以及为什么**（"不合适就不用"——这里把理由固化，避免以后重复讨论）

| 数据集 | 为什么不用 |
|---|---|
| **LMSYS-Chat-1M** | `schema` 里**根本没有 timestamp 字段**（已核验）→ 做不到达建模；只对前缀共享有用，而 Mooncake/LMCache 更好；且需要 HF 账号门控 |
| **ShareGPT（所有版本）** | **无时间戳、无 token 计数**，且没有权威版本；卡面自承含 canned response/raw HTML/4chan 抓取；按关键词"去对齐"的清洗会**系统性改变长度分布** |
| **BurstGPT** | 有 4 个月时间戳和 token 数，但**完全没有 prompt 文本**→ 对前缀共享零价值；到达模式可被 Azure 2024（免登录、一周）覆盖。**若需要"跨月漂移"再引入** |
| **CCL-Bench** | 记录的是 GPU 执行 trace（算子/kernel/通信），不是请求轨迹；且要"贡献换访问" |
| **FineServe / ServeGen** | 论文结论有用（突发性、分布漂移），但 FineServe **数据可得性未证实**、ServeGen **license 未证实**；不作为第一轮数据源 |
| **`swissai-serving-trace`** | 覆盖最广，但 21.4 GB 且 license 未读全文；**作为第二批**（要 16-token bucket 复用统计时再上） |

**两个"坑"的准确含义**（上一轮提到的两条，这里说清）：

1. **"LMSYS-Chat-1M 没有 timestamp"** —— 这不是"要小心使用"，而是**它不能用于到达建模**，
   所以按"不合适就不用"直接排除；替代品是 Azure/Mooncake（有真实时间）。
2. **"公开数据里没有节点 churn"** —— 这个**不是数据集选型问题，而是数据不存在**：
   没有任何公开数据集记录 LLM 服务集群里"某台机器被撤走/下线"的事件。
   所以它不能靠"换一个更合适的数据集"解决。处理方式只有两条：
   ① 用**代理信号**（WildChat 逐轮时间戳 + `hashed_ip` 推会话起止、LMCache 的 `pre_gap` 分布）；
   ② 显式**声明假设**（例如节点在线时长服从 Weibull/指数分布），并在论文里写明这是假设。
   **不做的是：假装有真实 churn 数据。**

---

## 1. 逐条核对（四个固定问题）

每个数据集都回答：**(i) 真实时间戳？(ii) 真实 prompt 文本 / 前缀 hash？(iii) in/out token 长度？(iv) 许可与门控？**

### 1.1 Mooncake FAST'25 conversation trace ★首选（b/c）

- 论文（FAST'25 Best Paper）：https://www.usenix.org/conference/fast25/presentation/qin
- 轨迹目录：https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces
- HF 镜像：https://huggingface.co/datasets/valeriol29/mooncake-traces

`[已核实]` 字段：`timestamp`（相对到达，**毫秒**，范围 0–3,600,000 = 1 小时）、
`input_length`、`output_length`、`hash_ids`。
`hash_ids` 的定义（论文原文）：把 **512-token 的块**连同其之前所有块一起哈希，再映射为全局唯一 ID；
**相同 hash ID 即表示该块及其前缀相同，可复用 KV**。
规模 **23,608 条**，平均 input **7590** / output **182**；HF `conversation` config 约 **1.4 MB**，Apache-2.0。

- (i) ✅ 毫秒级（但只有 1 小时） (ii) ❌ 无文本，但 ✅ **有前缀 block hash（对复用分析等价甚至更好）**
  (iii) ✅ (iv) Apache-2.0，免申请

`[已核实]` **vLLM 原生回放**：

```bash
curl -L -o conversation_trace.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl
vllm bench serve --model <model> --dataset-name=timed_trace \
  --dataset-path ./conversation_trace.jsonl --num-prompts 100 \
  --ignore-eos --self-timed \
  --timed-trace-chunk-hash-size 512 --timed-trace-sec-multiplier 0.001
```

**基准对标（论文 §4.2 Table 1，可直接引用）**：单全局缓存池命中率
**0.30（1000 blocks）→ 0.40（10k）→ 0.50（50k）→ 0.51（∞）**，LRU 最好；
**超过 50% 的 block 从未被命中**。我们的 cache-aware 路由至少要打赢这条曲线，否则说明路由没生效。

### 1.2 Azure LLM Inference Trace 2024 ★首选（a/c）

- 文档：https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md
- 下载（GitHub Releases，免登录）：
  `https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv`
  `https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv`

`[已核实]` 采集 2024-05-10–19（一周），两个服务（code / conversation）；CC-BY；
字段只有 `TIMESTAMP` / `ContextTokens` / `GeneratedTokens`。
`[已核实]` 论文（Splitwise, ISCA'24）给出 2023 版同源统计：**code 服务 prompt 中位数 1500 / output 13；
conversation 1020 / 129**（conversation 输出接近双峰）。2023 版只覆盖 **20 分钟**。

- (i) ✅ (ii) ❌ **无 prompt 文本**（GDPR，原文声明"看不到 prompt 内容"） (iii) ✅ (iv) CC-BY，免登录

⚠️ `❓UNVERIFIED`：2024 CSV 的具体字节数与时间戳精度未逐一核对。
⚠️ Splitwise 论文明确"为了模拟有安全保证的云服务，**不跨请求复用 KV-cache**"——所以它**不能**用来验证复用收益。

### 1.3 LMCache Agentic Traces ★首选（b/d）

- https://huggingface.co/datasets/sammshen/lmcache-agentic-traces （CC-BY-4.0）

`[已核实]` **787 个多轮 agentic session / 24,881 次 LLM 迭代**；
字段：`session_id`、`model`、**`input`（完整累积的 OpenAI `messages` 数组）**、
**`pre_gap`（上一轮响应末 token → 本轮请求发出的秒数 = 真实工具执行/思考时间）**、`output_length`。
关键性质（README 原文）：**`input[N]` 是 `input[N-1]` 的严格前缀超集** ——
**前缀共享是构造性保证的**，算出来的命中率不会因文本噪声失真。
统计：turns/session 中位 35；input 中位 **21K** token；output 中位 **104**（11.5% >500）；
`pre_gap` median 0.71 s / p95 3.73 s；README 称 **>90% 的 token 在上一轮已处理过**。

- (i) ✅（`pre_gap` 是轮间延迟，非绝对墙钟） (ii) ✅✅ **有完整真实 prompt 文本** (iii) ✅ output / 需自算 input
  (iv) CC-BY-4.0，免申请

### 1.4 BurstGPT（4 个月 Azure GPT 生产轨迹）

- https://github.com/HPMLL/BurstGPT （`[已核实]` LICENSE = **CC-BY-4.0**）

`[已核实]` v2.0 六个 CSV（3 个原始 + 3 个 `without_fails`），规模 ~5.29M + 5.34M 行，188 + 220 MB；
字段：`Timestamp`（**相对秒**）、`Session ID`（仅 conversation 模式有）、`Elapsed time`
（README 明确**不是 TTFT**）、`Model`（ChatGPT / GPT-4）、`Request tokens` / `Response tokens` / `Total tokens`、`Log Type`。

- (i) ✅（相对秒） (ii) **❌ 无任何 prompt 文本 → 不能算前缀共享** (iii) ✅ (iv) CC-BY-4.0

⚠️ 必须用 `*_without_fails_*.csv`（原始文件含 `Response tokens == 0` 的失败行）。
⚠️ vLLM 文档给的下载链接指向 **v1.1** 而非最新 v2.0。

### 1.5 `eth-easl/swissai-serving-trace`（OSDI'26 OpenTela）★覆盖最广

- https://huggingface.co/datasets/eth-easl/swissai-serving-trace
- `[已核实]` `gated: false`、`license: other`、总仓 **21.4 GB**

`[已核实]` 文件：`trace.jsonl`（**16,329,237 行 / 6.98 GB**，字段 `id`、`status`、
**`created_at` / `finished_at`（ISO-8601）**、`model`、`model_parameters`、
`reported_token_input`、`reported_token_output`）；
`qwen3-32b-buckets.jsonl`（3.99M 行 / 4.62 GB，追加 `token_count` 与 **`bucket_ids`**，
"16-token buckets and right padding"）；另有 5 个 `*-bucket-reuse.jsonl`。
README 明说用途包含 "cache locality and token reuse over bucketized input-token sequences"，
限制是 "serving metadata rather than original prompts or model outputs"。

- (i) ✅ 且**能直接算真实在飞并发**（created/finished 重叠） (ii) ❌ 文本，✅ bucket ID (iii) ✅（缺失标记 -1）
  (iv) `license: other`（未读 LICENSE 全文，❓部分 UNVERIFIED），**无门控**

### 1.6 对话文本类（用于 (b)/(d) 的补充）

| 数据集 | (i) 时间戳 | (ii) 文本 | (iii) token 数 | 许可 / 门控 | 备注 |
|---|---|---|---|---|---|
| WildChat-1M | ✅ **逐轮 UTC** | ✅ | ❌ 需自算 | ODC-BY / **无门控**，3.36 GB | `hashed_ip` 可串会话 → **(d) 最佳公开代理** |
| WildChat-1M-Full | ✅ | ✅ | ❌ | ODC-BY / **manual gate**（要说明为何需要 toxic 数据） | 4.30 GB |
| LMSYS-Chat-1M | **❌ schema 里没有该字段** | ✅ | ❌ | 自定义协议 / **auto gate** | 只能做 (b)，**不能做到达建模** |
| chatbot_arena_conversations | ✅ `tstamp` | ✅ | ❌ | cc / auto gate | 41.6 MB；`conversation_a/b` 是同 prompt 两条回复 → 天然的"同前缀不同输出"对照 |
| ShareGPT 各版本 | **❌ 全部无** | ✅（HTML/清洗） | **❌ 全部无** | CC0 / Apache-2.0 | **无权威版本**；卡面自承含 canned response、raw HTML、4chan 抓取；清洗按关键词删"对齐话术"会**改变长度分布** |

### 1.7 2025–2026 新增（可选）

- **ServeGen**（Alibaba，NSDI'26）：https://github.com/alibaba/ServeGen —
  2025 年 1–4 月、12 个模型、**35.4 亿请求**；`data/conversations/conversations_hashed.json`
  保留多轮结构（内容换 hash）。结论可直接引用：**短窗口 IAT 的 CV ≫ 1，且不存在单一最优随机过程**；
  5 分钟窗口有日周期；input ≈ Pareto+Log-normal 混合、output ≈ Exponential；**2,412 个 client 中前 29 个贡献 90% 请求**。
  `❓UNVERIFIED`：**license 未在 README/论文页找到**；数据体积未测。
- **FineServe**（2026）：https://github.com/hihiztc1/FineServe — 4 个月 / 57 模型 / 14.8 亿请求。
  `[已核实]` 明确**看不到明显日周期**（全球平台平滑掉了），**与 Azure/ServeGen 结论相反**。
  `❓UNVERIFIED`：`raw.githubusercontent.com/.../{main,master}/README.md` 均 404，**无法确认是否真的发布了数据文件**。
- **`Inferact/codex_swebenchpro_traces`**（MIT）：610 trials，公开材料里**最详细的 agentic 前缀共享统计**（见 §3）。
- **CCL-Bench**：**不要用**——它记录的是 GPU 执行 trace（算子/kernel/通信），不是请求轨迹，且需贡献换访问。

---

## 2. 横向对比

| 数据集 | 时间戳 | prompt 文本 | 前缀 hash | in/out token | 许可 | 门控 | 规模 |
|---|---|---|---|---|---|---|---|
| **Mooncake FAST25** | ✅ ms（1h） | ❌ | **✅ 512-token** | ✅ | Apache-2.0 | 无 | 23.6k / 1.4 MB |
| **Azure LLM 2024** | ✅ | ❌ (GDPR) | ❌ | ✅ | CC-BY | 无 | 1 周 × 2 服务 |
| **LMCache agentic** | ✅ 轮间 gap | **✅ 全文** | 可自算 | ✅ out | CC-BY-4.0 | 无 | 787 session |
| BurstGPT | ✅ 相对秒 | ❌ | ❌ | ✅ | CC-BY-4.0 | 无 | 10.6M / 408 MB |
| `swissai-serving-trace` | ✅ + finished | ❌ | **✅ 16-token** | ✅ | other | 无 | 16.3M / 6.98 GB |
| WildChat-1M | ✅ 逐轮 | ✅ | ❌ | ❌ | ODC-BY | 无 | 838k / 3.36 GB |
| LMSYS-Chat-1M | ❌ | ✅ | ❌ | ❌ | 自定义 | auto | 1M / 1.49 GB |
| ShareGPT 各版 | ❌ | ✅ | ❌ | ❌ | 混杂 | 无 | 53k–90k |
| ServeGen | ✅ | hash | ✅ 结构 | ✅ | ❓ | 无 | 3.54B |
| FineServe | ✅ | ❌ | ❌ | ✅ | ❓ | ❓ | 1.48B |

---

## 3. 前缀共享 / 缓存命中率：可引用的数字（写论文直接用）

| 来源 | 数字 |
|---|---|
| **Preble**（arXiv 2407.00023） | 5 类真实负载中 **85%–97% 的 prompt token 与其他请求共享**；同一序列平均被 **8.6–126** 个请求共享；prompt:output 比 **37×–2494×** |
| **Mooncake**（FAST'25 §4.2） | 命中率 **0.30→0.51**（容量 1k→∞ blocks，LRU 最优）；**>50% block 从未命中** |
| **vLLM × Mooncake**（2026-05-06 博客） | Codex/SWE-bench Pro：**94.2% 命中率**、**131:1** in:out、每轮 context +2242 token、轮间延迟 median **5.2 s** / P99 **81.4 s**；生产实验命中率 **1.7%→92.2%**、吞吐 **3.8×**、P50 TTFT **−46×** |
| **Inferact/codex traces**（MIT） | 总命中 **94.2%**；**93.8% 的 trial 首次调用就有跨 trial 命中**（共享 11520-token 主前缀）；逐轮命中 87.4%(t1) → 97.8%(t50)；top 5% 调用占 49.6% 的 uncached 计算 |
| **LMCache agentic 卡** | "**>90% 的 token 在上一轮已处理过**" |
| **SGLang HiCache**（2025-09-10） | Novita AI（Qwen3-Coder-480B）：命中率 **40%→80%**、TTFT **−56%**、吞吐 **2×**；Ant Group：命中带来 **84% TTFT 下降** |
| **llm-d**（2025-09-24） | 150 客户 × 6000-token 前缀、8 个 vLLM pod：P90 TTFT **precise 0.542 s** vs approximate **31.1 s** vs random **92.6 s**（**快 57–170×**） |
| **CacheBlend**（EuroSys'25） | "update fraction <15% 通常可生成同质量回复"；重算 10–20% 的 HKVD token，TTFT **2.2–3.3×**、吞吐 **2.8–5×** |

**对我们的含义**：
1. cache-aware 路由的判据 `matched_prefix_len > remaining_len` 与 Preble 一致，且真实负载里这个条件**经常成立**（85–97% 共享）；
2. 但命中率**随缓存容量强相关**（0.30→0.51），所以实验必须把"集群总缓存容量"作为自变量，否则不可比；
3. RAG/非前缀命中场景要额外预留 **10–20% 的重算预算**（CacheBlend），切层时不能假设"命中即零成本"。

---

## 4. 一个必须写进论文的负面结论：**没有公开的节点 churn 数据**

`[已核实 + 全量搜索]` **没有找到任何公开数据集直接建模 LLM 服务集群的节点 churn / 节点离开。**

可用代理：

| 优先级 | 数据/字段 | 怎么用 |
|---|---|---|
| 1 | **WildChat-1M**：`hashed_ip` + 逐轮 UTC 时间戳 | 聚出"客户端会话"，得到会话起止、轮间空闲、超时断连 |
| 2 | **LMCache**：`session_id` + `pre_gap` | gap 分布尾部 = "会话可能已结束" |
| 3 | **BurstGPT**：`Session ID` + `Timestamp` + `Elapsed time` | 近似 session 生命周期 |
| 4 | **Azure Functions Invocation Trace 2021** | 通用 serverless 到达/离开模型（`app`/`func`/`end_timestamp`/`duration`），可类比"会话持有" |

**建议把 (d) 拆成两个可测子问题**：①**会话级**（用户不再回来）→ 用 WildChat/LMCache 的 gap 分布拟合；
②**节点级**（机器被抢占/下线）→ 公开数据没有，**只能假设**（Weibull/指数占空比等）。
**论文里必须显式声明这是假设，不能声称有真实数据支撑。**

---

## 5. 坑清单

1. **门控**：LMSYS-Chat-1M、chatbot_arena_conversations 都是 `gated: auto`（要 HF 账号 + 填单位/国家 + 同意协议）；
   WildChat-1M-Full 是 `manual`（要说明为何需要 toxic 数据）。**不想走流程就用 Mooncake / Azure / WildChat-1M / BurstGPT / LMCache。**
2. **同一数据集有多个版本**：WildChat 在 2024 年两次删减内容（去 toxic、去 PII），**做可复现实验必须固定 revision**。
   ShareGPT 更糟——没有权威版本。
3. **LMSYS-Chat-1M 根本没有 timestamp 字段**（已用 schema 核验）→ 不能做到达建模，只能做前缀共享/多轮文本。
4. **Azure 2023 只有 20 分钟**，用它外推绝对吞吐会失真；且它声明不复用 KV。
5. **vLLM 官方警告**（文档原文）：重复对同一 server 跑 `vllm bench serve` 会**复用 prefix cache 从而虚高吞吐** ——
   做 cache 实验必须每次重启/重置 server。
6. **BurstGPT 必须用 `*_without_fails_*.csv`**；`Elapsed time` ≠ TTFT。
7. **vLLM `Sonnet` 数据集已 deprecated**。

---

## 6. 落地清单

```bash
# ① Mooncake 对话轨迹（前缀共享 + 长度分布 + 1 小时到达，~1.4 MB）
curl -L -o data/conversation_trace.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl

# ② Azure LLM 2024（一周到达/日周期 + code/conv 两档长度，免登录）
curl -L -O https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv
curl -L -O https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv

# ③ LMCache agentic（真实文本 + session + 轮间 gap）
python -c "from datasets import load_dataset; print(load_dataset('sammshen/lmcache-agentic-traces'))"

# ④ 可选：BurstGPT（4 个月 + Session ID，CC-BY）
wget https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_without_fails_2.csv
```

**接入我们现有 loader 的位置**：`edge_llm_scheduler/backends/datasets.py`
（已有 `load_contextpilot_json` / `load_preble_workload` / `load_node_trace` / `generate_synthetic`；
统一格式 `DatasetRequest = {request_id, arrival_time, input_ids, prefix_id, suffix_ids, max_new_tokens, raw}`），
需要新增：Mooncake `hash_ids → prefix_id/cached_prefix_tokens`、Azure `TIMESTAMP → arrival_time`、
LMCache `messages → input_ids` + `pre_gap → arrival_time`。

**至少保留两档长度**，否则 prefill:decode 配比结论只在一个点上成立：

| 档 | prompt | output | 对应现实 |
|---|---|---|---|
| completion/code | ~1.5K | ~13 | Azure code |
| chat | ~1K | ~130 | Azure conversation |
| 长上下文 / RAG | ~7.6K | ~182 | Mooncake |
| **我们现在** | **38** | **6** | — （差 1–3 个数量级） |

---

## 7. 配套的模型参数选型：要不要量化

数据集和模型要一起定，否则长度分布对不上模型容量。

### 7.1 结论

| 目标 | 需要 int4 吗 | 理由 |
|---|---|---|
| **四机异构 PP 跑 7B** | **不需要** | 7B fp16 = 15.2 GB 权重；按 8/6/4/4 GB 的加权切分（12/4/4/8 层）已验证可行；引擎的 fp16/bf16 路径是**已经跑通**的那条 |
| 更大模型（14B）或更长上下文 | 需要 | 14B fp16 = 28 GB，四机合计约 22 GB 装不下；int4 ≈ 8 GB 才放得下 |
| 提升并发 / KV 余量 | 需要（KV 量化或权重量化） | 4k 上下文下 7B 的 KV 约 24 层×4096×512 B ≈ 50 MB，主要压力还是权重 |
| 与 vLLM baseline 对齐比较 | 必须同 dtype | 否则吞吐数字不可比 |

### 7.2 正在下载什么

`Qwen/Qwen2.5-7B-Instruct` fp16（约 15.2 GB，4 个 safetensors 分片），
用 `scripts/hf_mirror_download.py`（**已修 User-Agent**：hf-mirror 会 403 掉默认的 `Python-urllib` UA），
落到 `.models/Qwen2.5-7B-Instruct`（D 盘，当前剩余 48 GB）。
下载后要做的第一件事：**用 `model_pp_fit.py` 按四个节点的显存重新算切分**，而不是沿用 8/6/4/4 的假设。

### 7.3 量化不是"下个开源的就能用"——引擎会静默出错

**这是必须先说清的风险**：我们的 `_materialize_shard` 用
`AutoModelForCausalLM.from_config(config)` 在 `torch.device("meta")` 上建**标准**骨架，
再 `load_state_dict(state, strict=False, assign=True)` 灌入本段权重。

AWQ/GPTQ 的 checkpoint：
- `config.json` 里有 `quantization_config`，需要构造 `WQLinear_GEMM` 之类的**量化层**，而不是标准 `Linear`；
- 权重张量是**打包过的** int32（`qweight`/`qzeros`/`scales`），键名与标准模型不同。

在 `strict=False` 下，这些键匹配不上 → **被静默忽略**，层里留下 meta/未初始化权重 → 输出是垃圾但**不报错**。
所以"直接下个 AWQ 版本就能跑"是错的。int4 要落地，必须做三件事：

1. 识别 `quantization_config` 并按量化类型构造正确的层（引入 autoawq / gptqmodel / llmcompressor 的 kernel）；
2. 按打包格式映射张量名（`qweight`/`qzeros`/`scales`，含 group size / bit packing）；
3. **加数值等价测试**：固定 prompt 下量化版与 fp16 版逐 token 比对（像 `verify_equivalence.py` 那样），
   防止"能跑但错"。

预估 **3–5 天**，且它对"每层耗时"的影响必须重新实测（`docs/design/single-machine-scope-and-batching.md` §5 第 5 项）。

### 7.4 顺带记下的两个事实（影响末端切层）

- **词表大小决定首/末段的固定占用**：Qwen2.5-7B 词表 152064 → fp16 的 embedding 约 **0.54 GB**；
  Mistral-7B 词表 32768 → 约 **0.23 GB**。四机异构时，首段往往不是"层数最多"的那台。
- encoder/decoder 的 `tie_word_embeddings` 决定末段是否要再背一份输出头（Qwen2.5 的
  0.5B/1.5B/3B 绑定，**7B/14B 不绑定** → 末段多一份 0.54 GB）。

---

## 8. 复核状态

| 项 | 状态 |
|---|---|
| Mooncake 字段/规模/许可、vLLM 回放命令 | ✅ 已核实 |
| Azure 2023/2024 元信息、字段、GDPR 声明、论文统计值 | ✅ 已核实 |
| LMCache agentic 字段、规模、前缀超集性质 | ✅ 已核实 |
| BurstGPT 许可、字段、规模（README 声明值） | ✅ 已核实（release 资产的精确字节数因 GitHub API 限流未核） |
| `swissai-serving-trace` 文件清单与字段 | ✅ 已核实；license 全文 ❓ |
| ShareGPT 各版许可与缺陷 | ✅ 已核实 |
| LMSYS-Chat-1M 无 timestamp | ✅ 已核实（schema） |
| FineServe 数据可得性、ServeGen license | ❓ **UNVERIFIED** |
| Azure 2024 CSV 字节数与时间戳精度 | ❓ **UNVERIFIED** |
