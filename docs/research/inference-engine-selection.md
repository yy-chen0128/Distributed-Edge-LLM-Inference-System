# 推理引擎选型：自研、数据中心引擎，还是边缘引擎

> **问题**：我们的执行面（stage 运行时）该自己写，还是用现成引擎？候选不只是 vLLM——
> 还有 Ollama、LM Studio、llama.cpp 的 RPC、exo、Petals、Mesh-LLM 这类"边缘/家用集群"系统。
>
> **证据分级**：✅ 源码或官方文档原文核对过；🔶 由已核对事实推断；❓查不到（不等于不存在）。
> 本文的详细取证见同目录 [`multi-machine-inference-engine-survey-2026-09.md`](multi-machine-inference-engine-survey-2026-09.md)（69 KB，逐条带 URL 与 V/I/U 分级）。
> 本仓库已克隆 `ollama/`（MIT，Go 服务端）、`lmstudio-python/`、`lms/`（MIT，客户端 SDK）供源码核对；**第三方源码不入库**（见 `.gitignore`）。

---

## 0. 结论先行

### 0.1 三条硬结论

1. **没有现成引擎同时满足 R1（异构 + 运行时重算的跨机层切分）和 R2（在途请求可恢复 + 幸存者上重建流水线）。**
2. **更具体地说：R2 是空白。** 所有真正实现了"在途请求恢复"的系统（SpotServe / LUMEN / PipeBoost / DynaPipe）
   都在**数据中心或单机内部**；所有跑在**消费级多机**上的系统（llama.cpp RPC / exo / distributed-llama /
   Mesh-LLM / prima.cpp）遇到节点消失时，要么**崩溃**、要么**取消请求**、要么**只撤销拓扑**——
   **没有一个真的把在途请求恢复出来**。这个不对称本身就是可发表的贡献点。
3. **我们这条设计路线的价值有第三方数字支撑**：prima.cpp 在 70B 上对比"异构感知的不等分切分"与
   exo 式"按比例分层"，得到 **674 ms/token vs 20,848 ms/token（15–31×）**
   （<https://ar5iv.labs.arxiv.org/html/2504.08791>）。也就是说"切分方式"在这个场景里能差一个数量级以上。

### 0.2 建议（三层结构）

| 层 | 做什么 | 为什么 |
|---|---|---|
| **执行面** | **继续自研**（我们的 HF 分层引擎） | 只有它能同时满足 R1 + R2；已经走到"真实 7B 单段在 GPU 上跑通" |
| **单机能力** | **能借就借、能抄就抄**：量化走 GGUF 生态、stage 内批处理照抄 llama.cpp slot / LM Studio Unified KV Cache、前缀缓存照抄"固定边界检查点 + LRU" | 这些是成熟轮子，重造没有收益 |
| **基线** | **分三层做**：单机 Ollama（MIT）+ 跨机 llama.cpp RPC（PoC）+ 数据中心 vLLM | 直接回应"自研引擎不好对比、脱离生产"的担心 |

**贡献边界**：我们的贡献被限定在**"调度 + 弹性"**这一层——不声称"比 vLLM 快"，
而是"在 vLLM / llama.cpp **无法表达**的动态异构场景下能跑、能恢复"。

---

## 1. 判定用的尺子（需求）

| 编号 | 需求 | 为什么 |
|---|---|---|
| **R1** | 跨机、异构、运行时可重算的层切分 | 四台机器显存 8/6/4/4，层数必须不等；节点变化后要重切。**项目立足点** |
| **R2** | 弹性/容错：机器随时被抱走；在途请求可恢复；流水线在幸存机器上重组 | 边缘场景的核心卖点 |
| **R3** | 前缀 KV 复用 + 按缓存局部性路由 | 策略贡献之一 |
| **R4** | stage 内连续批处理，批大小可配 | 实测 batch=1 时固定开销主导（7B 在 38 token 下占 75.6%） |
| **R5** | 可对照的知名基线 + 调度策略可替换/可插桩 | 论文需要可比数字，且贡献要能插进去 |
| **R6** | 低工程风险 | 人力有限；自研引擎的 bug 与性能不确定是真实风险 |

---

## 2. 候选清单（按"能否当底座"分档）

### 2.1 第一档：可以考虑当底座或基线的

| 系统 | 跨机层切分 | 异构/可重算 | **在途恢复** | 可插桩 | 许可 | 成熟度 |
|---|---|---|---|---|---|---|
| **我们自研的 HF 分层引擎** | ✅ 原生 | ✅ | ⚠️ 有框架、缺 KV 搬运 | ✅ | 自有 | 原型（7B 单段已跑通） |
| **llama.cpp + RPC** | ✅ 层切分（默认 `-sm layer`） | ⚠️ 按显存比例，装载时定死 | ❌ **节点消失 = 进程 abort** | ⚠️ 源码 MIT、无插件层 | MIT | 极活跃，但 RPC 自称 PoC |
| **Mesh-LLM**（2025–2026 新） | ✅ 连续层段（Skippy stage splits） | ✅ 规划器枚举节点子集、时延感知 | ⚠️ **只撤销拓扑**，未见在途恢复 | ⚠️ | Apache-2.0 | 自称 experimental |
| exo | ✅ Pipeline 分片 + 拓扑感知 | ⚠️ 按**空闲主机 RAM** 不等分，不能指定 | ❌ ≤10s 整个实例 `InstanceDeleted`、请求 Cancelled | ⚠️ 硬编码，需 fork | Apache-2.0 | 活跃，但 **Linux 仅 CPU** |
| vLLM / SGLang | ✅ 但**要求环境一致 + 无运行期重切** | ⚠️ **两者都支持不等分**：vLLM 官方文档有 "uneven GPU splits" 条目；SGLang 有 `SGLANG_PP_LAYER_PARTITION=15,15,15,16`（**启动期** env var） | ❌ | ⚠️ 免 fork 面无"运行时改 PP" | Apache-2.0 | 生产级 |
| Petals | ✅ **能精确表达我们的切分** | ✅ 自动/手工 `--block_indices 0:12/12:16/16:20/20:28` | ✅ **设计上真有**（见 §3.6），但**要求每段 ≥2 持有者** | ⚠️ | MIT | **休眠**（2024-08 后无提交） |

### 2.2 第二档：明确排除（附一句话理由）

| 系统 | 为什么排除 |
|---|---|
| **Ollama** | 源码核对：**没有任何跨机机制**（详见 §3.1） |
| **LM Studio** | LM Link 是"**整模型**远程服务"；服务端**闭源 + EULA 禁改禁再分发**（§3.2） |
| **distributed-llama** | **等分张量并行**：`buildLlmNet()` 是 node 循环在外、layer 循环在内，且 `assert(d % nNodes == 0)` —— **我们的 12/4/4/7 无法表达**；容错是每 3 s 整簇重传权重、永不缩容；实测 4 节点 1GbE：单机 11.00 t/s → 4 机 **1.85 t/s**（issue #294） |
| **ktransformers / PowerInfer** | **严格单机**；ktransformers 是 MoE **专家级** CPU/GPU offload（"expert-based, not layer-based"）；PowerInfer 仓库已迁移且"支持多 GPU"仍是未勾选的 TODO |
| **PowerInfer-2** | **公开仓库不存在**（候选 URL 均 404）→ 没有 license/release 可引 |
| **LocalAI 分布式** | 基于 llama.cpp RPC，官方劝退："Only a single model is supported currently"、"additional workers cannot be added once inference has begun" |
| **Ghostlink** | 包装 llama.cpp RPC；容错是**失败即清**（"active requests fail/drain cleanly"）；自承 "Usable Multi-Node Speed Unproven" |
| **MLC-LLM** | 有远端多 worker 通路代码，但**官方多节点文档 404** → 不能宣称"受支持的跨机 PP" |
| **TPI-LLM** | 4 笔记本张量并行的**负面结论**："The bottleneck in allreduce is not network bandwidth, but **link latency**" —— 正好支持我们"跨 WiFi 不做 TP"的既定结论 |
| **prima.cpp** | 算法最贴我们的场景（4 节点家庭集群、WiFi 3–7 ms、Halda ILP 解不等分、调度时延 10–12 ms），**但论文给的代码 URL 404**，无法当引擎；**只作算法参照**（见 §3.9） |
| **ServerlessLLM** | 成熟多节点，但是**模型复用/迁移**，不是跨机层切分 |
| **MLX distributed** | 有真实分布式运行时，但**仅 Apple Silicon**，与我们无关 |
| **rpc_manager** | 仅确认存在（README 404 / API 403），能力未知 |
| **DistServe / Splitwise / Mooncake / NVIDIA Dynamo / llm-d / KubeAI / KServe / Ray Serve LLM** | 数据中心 PD 分离或 K8s/Ray 运维层，与"消费级跨机层切分"不是一回事 |

> ⚠️ **不要引用**："ServerlessLLM Flex" 经专门搜索**未能确认存在**；Parrot / Metis / Tetris / Helix / Ayo / PipeSwitch 本轮未核验。

---

## 3. 逐条核对（证据）

### 3.1 Ollama —— 源码核对：**没有跨机机制**

`✅` 我们克隆了源码（commit `6383a0f`，2026-09-18）逐项核对：

- **全部 Go 源码里 `grep -i rpc` 只有一处注释命中**（webview.go 的说明文字），没有任何 RPC 分布式计算路径。
- 环境变量（`ollama/envconfig/config.go`）里唯一带 remote 字样的两个**不是远程算力**：
  `OLLAMA_REMOTES` = "允许从哪些主机**取模型**（默认 ollama.com）"；`OLLAMA_CREATE_REMOTE` = "强制走服务端 API 创建模型"。
- 层切分：`ollama/llm/llama_server.go` 把层数作为 `-ngl` **透传给子进程 `llama-server`**；
  默认 `NumGPU = -1` 时**根本不传**，让 llama-server 自己探测。用户**不可按设备指定层数**。
- 并发：`OLLAMA_NUM_PARALLEL` 默认 **1**；`OLLAMA_MAX_QUEUE` 512。文档说**上下文按并发数线性放大**
  （2K × 4 并发 = 8K）——这正是我们做"KV 预算"对比时的好反例。
- 有用的部分：`/api/generate` 响应含 `prompt_eval_cached_count`（从缓存读了多少 prompt token）→ **指标口径可借鉴**。

**判定**：R1 ❌ / R2 ❌ / R3 半 / R4 ✅ / R5 ⚠️ / R6 单机低、四机要 fork。**用途：单机 MIT 基线。**

### 3.2 LM Studio —— 闭源，跨设备是"整模型远程"

`✅` 底层 GGUF 走 llama.cpp、Apple Silicon 走 MLX；**服务端闭源**，开源的是客户端 SDK 与 mlx-engine（MIT）。
我们克隆的 `lmstudio-python` / `lms` 确认**只是客户端**（websocket 连本地实例，无任何调度器/引擎代码）。

`✅` **LM Link**（0.4.6，2026-02-27）机制是账号配对 + Tailscale 网格；文档原话是
"在远端设备上**加载整个模型**，像本地一样使用"；同模型在多机上是**多个独立条目**；
设备选择是**每台机器静态 preferred device**（整模型放置，不按层、不按请求）。崩溃只表现为 "disconnected"。

`✅` 值得抄：连续批处理可配（Max Concurrent Predictions 默认 4；**Unified KV Cache 默认开**，
关闭会按 slot 硬切上下文）；SSE 事件含 `prompt_processing.progress`；MLX 引擎的 KV
**以 256 token 为边界做检查点落盘 + LRU + 恢复最长可用前缀**。

`✅` 许可：**专有**，EULA 禁止修改/派生、禁止以非公开接口集成、禁止逆向、禁止再分发与 SaaS；责任上限 50 美元。

**判定**：R1 ❌ / R2 ❌ / R5 ❌（**法律上不允许改**）。**用途：只能当黑盒单机基线。**

### 3.3 llama.cpp + RPC —— 最省事的跨机基线（但官方自称 PoC，且节点消失会 abort）

`✅` **切分方式（这一条我上一版写错了，已更正）**：官方参数表有
`-sm, --split-mode {none,layer,row,tensor}`，**默认是 `layer`** —— "split layers and KV across GPUs (**pipelined**)"。
RPC README 原文："By default, llama.cpp distributes model weights **and the KV cache** across all available
devices -- both local and remote -- **in proportion to each device's available memory**"，可用
`--tensor-split` 覆盖比例。所以它是**层切分 + 该层的 KV 就留在那台机器上**，
跨机搬的是 activation（与我们的 PP 形态接近）——比"张量并行"要友好得多。

`✅` **但它不是"真流水线"**：**没有跨机 stage overlap**——一个 token 必须依次穿过所有设备；
跨机搬的是 activation，按 ubatch 级（commit #28789 提到双机 split 时每个 prefill ubatch >10 MB）。
`-sm tensor`（同时切权重与 KV）官方标注 **EXPERIMENTAL**。
这意味着我们"多个请求同时在途、把流水线填满"的做法在它上面表达不出来。

`✅` **不等分**：`-ts/--tensor-split` 给比例；**要精确钉死"某台机器恰好拿 12 层"**，维护者在 discussion #12714
里的回答是"可以用 `-ot/--override-tensor`"，但那是**hedged 的非承诺说法**，且 `-ot` 的 RPC device 名格式
要从源码拼（`"RPC" + index + "[" + endpoint + "]"`）。→ 精确不等分需要 `-ot` 或小改源码。

`✅` **运行时重算：不能。** 切分发生在加载期（`llm_load_tensors`），控制手段全是启动参数；
server 的 endpoint 清单里**没有任何放置 API**。

`✅` **远端机器消失 = 进程 abort**（这是 R2 的硬伤）：源码里
`#define RPC_STATUS_ASSERT(x) if (!(x)) GGML_ABORT("Remote RPC server crashed ...")` 用在每条命令上；
连接失败也是 `GGML_ABORT("Failed to connect to %s")`。
**未合入的 PR #26724 也只是把"崩溃"换成"该设备永久失效"**，其注释原话：
"the buffers on the server are gone, so a lost connection can never be recovered"，
之后每个请求返回 503 "Compute device failed, the server must be restarted"。
✅ 核对 2026-09-22 的 master **仍含 `RPC_STATUS_ASSERT`、无 `is_failed` 符号 → 该 PR 未合入**。
discussion #12714 里有人问"能不能有冗余层"→ **0 条回复**。

`✅` **没有官方性能数字**。第三方 10 GbE 双机实测（M2 Ultra + DGX Spark）：Qwen2.5-7B Q4_K_M
本地 76.1 tok/s → RPC **52.7** tok/s；72B 本地 28.2 → RPC 5.9 tok/s；作者结论
"**RPC is for capacity, not speed**"、"1GbE 下 decode 的网络开销还要再差约 10×"
（<https://github.com/kjaiswal/llama-cpp-distributed-benchmarks>）。
⚠️ 那次是**两台性能差异很大的机器**，所以降速里有多少是网络、多少是异构不均衡，**无法从这组数字里分开**。

`✅` 其他：`-cb/--cont-batching` 默认开、`-np/--parallel N` 分槽（R4 ✅）；
`--metrics` 暴露 Prometheus 指标（可观测性尚可）；**router 只按模型名路由，无 cache-aware 路由**；
llama-server 自身有槽级 KV 存取（`POST /slots/{id}?action=save|restore`）。

`✅` **R6 风险（源码级）**：`ggml_backend_rpc_device_get_type` 有
`// TODO: obtain value from the server` 且恒返回 GPU；`supports_op` 有
`//TODO: call the remote backend and cache the results` 且**恒返回 true**（不支持的算子要到运行期才炸）；
**无鉴权、无加密**，且 2026-09-16 刚修了一个 RPC 的未授权 RCE（#24292）；协议不版本化。
→ 只能放在受控局域网/VPN 里。

**判定**：R1 ⚠️（能层切分，但异构比例与运行期重算受限）/ R2 ❌ / R4 ✅ / R5 ⚠️ / R6 ⚠️（安全 + PoC）。
**用途：跨机基线的最优选择**——代价最低，顺带把 GGUF 量化道路免费打开；
它官方自述的 fragile/insecure 与"节点消失即 abort"正好是论文里"现有跨机方案缺调度与容错"的最好引证。

### 3.4 Mesh-LLM（2025–2026 新发现，值得做 spike）

`✅` 跨机连续层段（文档 `docs/SKIPPY_SPLITS.md`）；规划器"enumerates node_count in
minimum_nodes..=usable_nodes and **every node subset**"、时延感知（默认
`DEFAULT_TARGET_DECODE_TPOT_MS = 33`）；提供手工 lock JSON **接受任意半开层区间**——
实测自动放置里出现过 `0..65 / 65..66` 这种极端不等分。
容错是"stage 丢失 → 宽限期后撤销拓扑"+ 回收孤儿 session/lane；**但未发现在途请求的 checkpoint/resume**。
有 suffix/prefix KV 复用（3531 token 精确回放）。自称 "experimental distributed-systems software"。Apache-2.0。

**判定**：R1 ✅ / R2 ⚠️（只撤拓扑）/ 值得花 1–2 天做 spike，验证"stage 内是否有连续批处理"与
"stage 消失后在途请求能否继续"（现证据倾向"不能"）。

### 3.5 exo —— 架构最像，但平台不对、容错缺失

`✅` 官方 README："**Topology-Aware Auto Parallel**：figures out the best way to split your model across all
available devices based on a realtime view of your device topology. It takes into account device resources and
network latency/bandwidth between each link."；API 里有 `sharding: "Pipeline"` 与 `/instance/previews`
（**放置方案可预览、可影响**，这点值得借鉴）；后端 MLX（`MlxRing`）。

`✅` 但：层数是 `allocate_layers_proportionally` 按**空闲主机 RAM**（不是显存、不是速度）自动决定，
**无法指定 12/4/4/7**，也没有 CLI flag；重算是破坏性的。
**节点消失 ≤10 s → 整个实例 `InstanceDeleted` → 客户端流关闭、任务 `Cancelled`，无 checkpoint/re-prefill/重试**；
KV 有 `KVPrefixCache`（LRU）但**不可迁移、不持久，随节点死亡**；**无 cache-aware 路由**；
策略是硬编码函数（要 fork `master/placement.py`）。
**决定性阻塞**：README 原文 "**On Linux, exo currently runs on CPU.** GPU support for Linux platforms is
under development"，且旗舰特性依赖 Thunderbolt-5 全互联 RDMA → 我们四台 NVIDIA 笔记本用不了。

**判定**：R1 ✅ / R2 ❌ / 平台 ❌。**用途：参考设计**（拓扑感知切分 + 放置预览 API + 异构资源建模）。

### 3.6 Petals —— **唯一在设计上真做了在途恢复的**，但不能当引擎

`✅` 它是**唯一"能精确表达我们切分"**的现成系统：`--block_indices 0:12/12:16/16:20/20:28`；
也支持自动不等分（吞吐贪心、`--balance_quality 0` 可关重平衡）。

`✅` **它的容错设计正是我们要做的**：客户端缓存"已经发给该 stage 的历史输入" + 服务端 KV，
**失败时只重跑失败的那个 stage，而不是从第 0 层重来**。论文（2312.08361 Table 1）给了量化收益：
BLOOM-7.1B、4 段 **(8,7,8,7)**（几乎就是我们的拓扑）、1024 token、失效率 p=1e-3：
**7.76 steps/s vs 整段重跑 0.48 steps/s**。

`✅` **但它有两个致命障碍**：
1. 论文 §3.2 明写 "**we assume that every block is hosted on several servers**" ——
   我们 4 段各一台机器时，B 一死就 `MissingBlocksError`。**要容错就必须每段 ≥2 持有者**
   （即至少 8 台机器，或每段冗余部署）。
2. issue **#587 仍 open**，维护者原话："the issue certainly exists and I can confirm that it **breaks fault
   tolerance on my side**. We have not yet fixed it"；代码自称 "not fault-tolerant out of the box"。
   且项目**休眠**（最后提交 2024-08-25，`transformers` 硬锁在 4.43.x）。

**判定**：R1 ✅ / R2 ✅（设计）/ R6 ❌（休眠 + 每段需冗余）。
**用途**：**最值得读的相关工作**，尤其是"只重跑失败 stage"这个机制；
可以作为对照 baseline（能精确表达我们的切分），但必须接受"每段 ≥2 副本"或"无恢复"。

### 3.7 vLLM / SGLang —— 数据中心路线（**更正：不是"必须同构"**）

`✅` **先更正我上一版写错的一条**：vLLM **并不要求每段层数相同**——官方文档里有
"Edge case: **uneven GPU splits** ... splits the model along layers and supports uneven splits"；
SGLang 也有 `SGLANG_PP_LAYER_PARTITION=15,15,15,16`（官方还建议"把大的分片放在更高的 PP rank"）。
所以"同构静态"这个说法不准确。

`✅` **真正的阻塞是另外三条**：

1. **官方要求各节点执行环境完全一致**：原文 *"Ensure that every node provides an identical execution
   environment ... Using container images is recommended ... to **hide host heterogeneity**."*
   —— 这与我们 8/6/4/4 的异构集群**直接冲突**（它把异构当成要藏起来的东西，我们把它当成要利用的东西）。
   原文还有 *"closing any shell terminates the cluster."*
2. **无运行期重切**：切分是配置期决定的（env var / 启动参数）。
3. **无弹性**：容错甩给 Ray；没有"节点被抱走后恢复在途请求"这回事。
   另外本仓库此前核对过：弹性 EP 与 PP 在配置层**互斥**（`config/parallel.py:846-850`）。

`✅` 换来的是我们缺的东西：PagedAttention、连续批处理、前缀缓存（Radix Cache）、KV connector 层粒度接口
（`save_kv_layer(layer_name)` / `wait_for_layer_load(layer_name)`）。

**判定**：R1 ⚠️（能不等分，但环境要一致、运行期不能改）/ R2 ❌ / R4 ✅✅ / R5 ✅。**用途：数据中心基线。**

### 3.9 R1 / R2 的论文级参照（算法可借鉴，硬件不可用）

这一节是**相关工作**，不是候选引擎：

| 工作 | 对我们的价值 |
|---|---|
| **prima.cpp**（arXiv 2504.08791） | **测试床就是我们的场景**：4 节点家庭集群（2 笔记本 + 1 台式 + 1 手机）、一个 WiFi 路由器、320–610 Mbps、3–7 ms 时延。用 **Halda ILP** 解层分配，**4–32 设备调度时延 10–12 ms**。**70B Q4K：674 ms/token**；换 llama.cpp **10,120 ms/token**；**改成 exo 式按比例分层 → 20,848 ms/token** → **异构感知的不等分值 15–31×**。⚠️ 但它**无容错、无 KV 缓存**，且明确"mini-batching is not yet implemented"；**论文给的代码 URL 404**，我们无法复现 |
| **SpotServe**（ASPLOS'24） | **R1+R2 的经典参照**：运行期重并行化 + 二分图匹配迁移 + *"stateful inference recovery ... commits inference progress at a much **finer granularity** and allows SpotServe to **cheaply resume inference upon preemption**"*；P99 尾时延改善 2.4–9.1×。⚠️ **代码未发布**（原文"will release the code soon"），且是数据中心 spot 实例场景 |
| **DynaPipe**（NeurIPS 2025） | R1 的算法形态：*"dynamic layer redistribution ... adaptively balances computation by predicting execution latency in real time"* + *"an **asynchronous KV cache migration** coordinator to enable **non-blocking** layer redistribution during inference"*；时延降 8–49%。❓跨机范围与代码未确认 |
| **LUMEN**（2026） | 负载感知 **KV checkpoint 到别的 worker 的 CPU 内存** + locality-aware 恢复调度 + **用 draft model 渐进恢复**；4 worker Qwen3-32B TTFT −44.4%。⚠️ **门槛极高**：*"80 GB per worker for Qwen3-14B and 160 GB per worker for Qwen3-32B"*；代码未开源 |
| **PipeBoost** | 崩溃时层重分配 + KV 重建，但**仅单机内**（原文：*"heterogeneous GPUs ... a more sophisticated model partitioning strategy is required"*） |

**这五条合起来说明**：R1（运行期重并行化）与 R2（有状态恢复）在**数据中心**都有人做过，
在**消费级多机**上是空白；而我们的场景同时要"异构 + 弱网 + 节点会走"，正好落在两者之间。

### 3.8 我们自己的 HF 分层引擎 —— 唯一同时满足 R1+R2 的

`✅` 已实测（见 `single-machine-scope-and-batching.md`、`parameter-and-kv-management-status.md`）：
meta 骨架只加载本段权重、per-request KV、epoch 隔离、activation 带校验和与 epoch、
节点离开触发重切 + 旧 epoch drain、**真实 7B 单段已在 GPU 上跑通（4 层驻留 1864.5 MB）**。
缺的是：KV 导出/导入、显存可行性约束、批处理、准入控制。

---

## 4. 关键区分：为什么"跨机"这件事各家做得都不是我们要的

| 维度 | 数据中心（vLLM/SGLang） | 边缘 offload（llama.cpp RPC） | 家用集群（exo / Mesh-LLM） | Petals | **我们** |
|---|---|---|---|---|---|
| 切分单位 | 层（**同构**） | 层（按**显存比例**） | 层（按 RAM / 拓扑） | 层（**可精确指定**） | **层（异构、按实测能力）** |
| 进程模型 | 单控制器 + 多 worker | **单进程** + 远端设备 | 多进程 + 自动发现 | 多进程 + DHT | **多独立进程 + epoch** |
| 节点离开 | 任务失败/重启 | **进程 abort** | **撤销拓扑 / 实例删除** | **只重跑失败 stage**（需冗余） | **重切 + drain + 请求恢复** |
| 切分可变 | ❌ 装载时定死 | ❌ 装载时定死 | ⚠️ 可重放放置 | ⚠️ 每 ~60s 重平衡 | ✅ **运行时换 epoch** |
| 交换的东西 | 张量/RPC | activation | activation | activation | **activation（带 epoch + 校验和）** |

**要点**：**"跨机"不是难点，"跨机 + 运行时可变 + 节点会走"才是。**
前两者别人做了；第三个只有 Petals 认真做过（但要求每段冗余、且已休眠）。
**"在途请求恢复"在消费级多机系统里是空白——这就是我们的贡献点。**

---

## 5. 建议

### 5.1 路线 A（推荐）：llama.cpp RPC 当执行底座 + 自研控制面 —— **备选**，见 §5.3 权衡

- **做法**：健康探测/故障判定 → 用新的 `--rpc` 列表重启并缩容；在途恢复走 **Petals 式**
  "客户端保存已发送输入 + 重算受影响前缀"。
- **收益**：MIT、极活跃、`llama-bench` 给可复现 baseline、R4 ✅、R3 部分 ✅、量化免费。
- **代价**：**R2 全部自研**；必须放受控 LAN/VPN（无鉴权 + 有 RCE 史）；层边界不精确需 `-ot` 或小改源码。

### 5.2 路线 B / C

- **B（评估项）**：Mesh-LLM 做 1–2 天 spike，验证"stage 内连续批处理"与"stage 消失后在途请求能否继续"。
- **C（不推荐单独用）**：Petals 私有 swarm 作对照 baseline（能精确表达我们的切分，论文给了我们拓扑的数字），
  但须接受"每段 ≥2 副本"或"无恢复"。

### 5.3 我的建议：**A 与"继续自研"并不互斥，但要先做一个决定性的实测**

上面的路线 A 与"执行面自研"的差别，本质是**"stage 内部要不要换成 llama.cpp"**：

| | 继续自研 HF 引擎 | 换成 llama.cpp RPC |
|---|---|---|
| R1 异构切分 | ✅ 完全可控（想切几层切几层） | ⚠️ 需 `-ot` 或改源码才能精确 |
| R2 弹性 | ✅ 已在做（epoch + drain） | ❌ 节点消失即 abort，要从头在控制面重做 |
| R4 批处理 | ❌ 要自己写 | ✅ 白拿（`-cb` + `-np`） |
| 量化 | ❌ 要自己接（AWQ 会静默出错） | ✅ GGUF 生态白拿 |
| 可插桩 | ✅ 全部 | ⚠️ 无插件层 |
| 性能 | ❓ 未知（每层固定开销 2.55 ms 偏高） | ❓ 未知（第三方显示跨机 decode 约 2× 下降） |

**所以决定这个选择的关键数据是：WiFi 下"每 token 跨机同步"到底要多少开销。** 这必须实测：

### 5.4 前置实测（会决定架构，优先级最高）

1. **WiFi 下 4 机 `llama-bench --rpc`**：量每 token 的跨机同步开销。
   第三方在 10 GbE 上已见 7B decode 约 2× 下降，并称 1 GbE 还要再差约 10×。
   **若 WiFi 下 decode 被网络主导，那"一个 token 依次跨 3 跳"这条设计本身就要重新考虑**
   （例如改成 stage micro-batch / 多 token 流水）。
2. 我们**自己**已有的可对照数字（单机 loopback）：每跳 `hop − compute` 稳态 **0.7–1.1 ms**
   （主要是 Python 序列化，loopback RTT 只有 0.14 ms）。
   把它加上真实 WiFi RTT（按部署文档分档 2–10 ms）× 3 跳 → **每 token 多 6–30 ms**。
   拿这个与"每 token 计算时间"（0.5B 四段约 30 ms；7B 四段约 90 ms）比，就能判断 WiFi 是不是瓶颈。
   ⚠️ 这只是估算，**必须用真实 4 台机器实测**（对应部署文档里的 P0 项）。

---

## 6. 风险与未决问题

| 风险 | 说明 | 缓解 |
|---|---|---|
| 自研引擎性能不如生产引擎 | 我们 7B 每层固定开销 2.55 ms 偏高 | 贡献限定在调度/弹性；用三层基线把差距量化出来 |
| 自研引擎 bug | 已修 9 处 + KV 计量（曾恒报 0） | 保持"每个能力都有实测数字"；等价性测试 |
| "脱离生产"的质疑 | 用的是 HF 分层运行时而非生产引擎 | 基线含 Ollama / llama.cpp / vLLM 三个真实系统 |
| **是否换 stage 内部实现** | 见 §5.3 权衡表 | **先做 §5.4 的实测再定** |
| 引用风险 | 调研中的"实测数字"多来自第三方仓库或论文，非我们复现 | 论文引用时标明来源；不冒称我们实测过 |

---

## 6.5 可直接引用的关键句（写论文时用）

| 论点 | 出处原话 |
|---|---|
| 现有跨机方案**没有容错** | llama.cpp RPC：远端机器消失即 `GGML_ABORT`；未合入的 PR #26724 注释 *"the buffers on the server are gone, so **a lost connection can never be recovered**"*，合入后也只是"设备永久失效 + 503 必须重启" |
| 现有家用集群方案的容错是**删除** | exo：节点消失 ≤10 s → 整个实例 `InstanceDeleted` → 客户端流关闭、任务 `Cancelled`；**无 checkpoint、无 re-prefill、无重试** |
| 唯一做了"只重跑失败段"的设计**结构上要求冗余** | Petals 论文 §3.2：*"we assume that **every block is hosted on several servers**"* → 我们 4 段各一台时 B 一死就 `MissingBlocksError`；且 issue #587 至今未修，维护者确认"breaks fault tolerance on my side" |
| 异构不等分的价值**有数字** | prima.cpp：70B Q4K，Halda ILP 不等分 **674 ms/token** vs exo 式按比例 **20,848 ms/token**（**15–31×**） |
| 数据中心方案**要求把异构藏起来** | vLLM 官方文档：*"Ensure that every node provides an identical execution environment ... to **hide host heterogeneity**"* |
| 跨 WiFi 不该做张量并行 | TPI-LLM：*"The bottleneck in allreduce is **not network bandwidth, but link latency**"* |
| 跨机层切分在弱网下的代价**没人公测** | llama.cpp RPC 官方**没有任何性能数字**；第三方 10 GbE 上 7B decode 约 2× 下降，作者称 1 GbE 还要再差约 10×，并结论 *"RPC is for capacity, not speed"* |
| 单机引擎**解决的是相反的问题** | ktransformers：*"Unlike traditional **layer-based** or KVCache offloading (as seen in llama.cpp), we offload the **expert** computation to the CPU"* |

---

## 7. 附录：核对记录

| 对象 | 版本 / 日期 | 许可 | 我们做了什么 |
|---|---|---|---|
| Ollama | 源码 commit `6383a0f`（2026-09-18） | MIT | **克隆源码**，逐项核对 envconfig / sched.go / llama_server.go / api/types.go |
| lmstudio-python | `1679683`（2025-10-27） | MIT | 克隆；确认是客户端 SDK |
| lms（CLI） | `1017bcb`（2026-09-21） | MIT | 克隆 |
| LM Studio 服务端 | 0.4.25（2026-09-19） | **专有** | 只读官方文档与 EULA |
| llama.cpp（含 RPC） | master @ 2026-09-22，release b11103 | MIT | 抓取 RPC/server README + `ggml-rpc.cpp` 源码片段 |
| exo | master README，release v1.0.71（2026-04-23） | Apache-2.0 | 抓取 README / PLATFORMS.md |
| Petals | arXiv 2209.01188 / 2312.08361 | MIT | 抓取论文 + issue #587 |
| Mesh-LLM | 仓库 + `docs/SKIPPY_SPLITS.md` | Apache-2.0 | 抓取文档 |
| prima.cpp | arXiv 2504.08791 | — | 抓取论文（异构切分对比数字） |
| distributed-llama | commit 2026-07-05，v0.16.5 | MIT | 抓取 `src/llm.cpp`、`src/nn/nn-core.cpp` |
| ktransformers / PowerInfer | v0.7.1（2026-09-15）/ 2026-05-11 | Apache-2.0 / MIT | 抓取 README |
| vLLM / SGLang | — | Apache-2.0 | 官方文档 + 本仓库既有核对结论 |
| 详细取证 | — | — | [`multi-machine-inference-engine-survey-2026-09.md`](multi-machine-inference-engine-survey-2026-09.md) |
