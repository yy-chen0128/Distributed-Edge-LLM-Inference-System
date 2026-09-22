# 多机消费级硬件 LLM 推理引擎调研（面向 4 笔记本异构流水线）

调研日期：2026-09-22　　调研对象：可用作「4 台消费级笔记本（8/6/4/4 GB GPU）、WiFi、异构不等分层流水线（A:0-11 / B:12-15 / C:16-19 / D:20-27）、单 token 依次流过 A→B→C→D」引擎的现成系统。

需求编号（下文统一使用）：

- **R1** 异构、运行时可重算的分层切分
- **R2** 弹性/容错：任一台笔记本随时可能被合盖/带走；在途请求必须可恢复，流水线要在幸存者上重组
- **R3** 共享前缀的 KV 复用 + cache-aware 路由
- **R4** 单 stage 内的 continuous batching
- **R5** 可作可复现 baseline，且调度策略能被替换/插桩
- **R6** 工程风险低

---

## 0. 结论速览

**核心结论：在公开可核验的系统中，没有任何一个同时满足 R1 与 R2。** 尤其是 **R2 在消费级硬件上是空白**：所有实现了「在途请求恢复」的系统（SpotServe / LUMEN / PipeBoost / DynaPipe）都在数据中心或单机内部；所有跑在消费级多机上的系统（llama.cpp RPC / exo / distributed-llama / Mesh-LLM / prima.cpp）**要么直接崩溃/取消请求，要么只是撤销拓扑，没有任何一个真的把在途请求恢复出来**。这正是本课题的贡献空位。

| 系统 | 并行方式 | R1 不等分? | R1 运行时重算? | R2 在途恢复 | R3 前缀缓存 | R4 continuous batching | R5 策略可替换 | R6 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| **llama.cpp RPC** | 层切分（默认 `--split-mode layer`）+ 权重/KV 按内存比例分摊 | ✅ 可以（`-ts` / `-ot`），但边界不精确 | ❌ 仅加载时 | ❌ **进程 abort**（未合入的 PR #26724 也只是「设备永久失效」） | ✅ llama-server 有 prompt cache；❌ 无 cache-aware 路由 | ✅ `-cb`（默认开） | ❌ 无插件 API，只有 `-ot` 这一「准策略」旋钮 | 中：MIT、超活跃，但 RPC 自称 proof-of-concept / fragile / insecure | **最现实的执行底座，容错必须自研** |
| **exo** | 层切分（`Sharding.Pipeline`，默认）+ 张量并行 | ⚠️ 自动按**空闲主机 RAM** 比例（非 VRAM/速度），无法指定 12/4/4/7 | ⚠️ 可重算但**破坏性**（删旧实例、取消任务） | ❌ 节点消失 ≤10 s 后整个实例被删、任务 Cancelled、客户端流被关 | ⚠️ 有 `KVPrefixCache`（LRU），但不可迁移、不持久、无 cache-aware 路由 | ⚠️ 有 `--no-batch` 开关，未核验连续批处理细节 | 观测 ✅（events/traces/bench），策略 ❌（硬编码函数，需 fork） | 高：Linux 上**只能跑 CPU**，旗舰特性要 Thunderbolt-5 RDMA | **架构最像，平台不对，容错缺失** |
| **distributed-llama** | **张量并行**（每节点持有**每一层**的等分切片） | ❌ 硬 assert 等分，无法表达 A/B/C/D | ❌ 启动时定死 | ❌ 整簇 re-init、权重全量重传、请求丢失、永不缩容到 3 节点 | ⚠️ 单一全局 `NaiveCache`，无 cache-aware 路由 | ❌ 无（单线程、请求串行） | ❌ 无 API、无 metrics | 中：MIT、活跃，但自成 `.m` 格式、同步量巨大 | **不能满足「分层流水线」，直接排除** |
| **Petals** | 层切分流水线（互联网） | ✅ 自动吞吐贪心 + 可 `--block_indices` 手工钉死 | ✅ 每 ~60 s rebalancing（`--balance_quality`） | ⚠️ **设计上有**（客户端缓存输入 + 服务端 KV，只重跑失败 stage），但**结构上要求每个 block 有 ≥2 个持有者**，且实现**公认已坏**（issue #587 仍 open） | ⚠️ 仅 session 内；无跨请求前缀缓存；✅ cache-aware 路由（`cache_tokens_left`） | ⚠️ 服务端有 batching，未核验 continuous | 观测 ✅（DHT ServerInfo/route 日志），策略 ❌（需静态覆盖） | 高：**休眠项目**（2024-08 最后提交、2023 依赖锁死） | **设计参考价值最高，不能当引擎** |
| **ktransformers** | 单机 CPU/GPU **专家级** offload | ❌ 无跨机分层 | ❌ | ❌ 无 | ✅ 三层 GPU-CPU-Disk 前缀缓存（单机） | ✅ balance_serve 连续批处理 | ❌ | — | **严格单机，排除** |
| **PowerInfer / -2** | 单机 neuron/FFN 级 offload；-2 为单手机 | ❌ | ❌ | ❌ | ✅（继承 llama.cpp）/ -2 仅 KV 常驻 | ✅（继承 llama.cpp）/ -2 无 | ❌ | — | **严格单机，PowerInfer-2 无代码，排除** |
| **Mesh-LLM** | 跨机连续层段（Skippy stage splits） | ✅ 时延感知规划器，自动出现 0..65/65..66 这种极端不等分 | ✅（枚举节点子集与 stage 数的规划器） | ⚠️ stage 丢失 → 宽限期后撤销拓扑 + 回收会话/`split_prefill_tokens`；**无在途请求 checkpoint/resume** | ✅ suffix/prefix 复用（有 3531 token 精确回放记录） | ❓ 未核验 | ✅ 文档化的 planner + 手工 lock JSON | — | **2025-2026 最值得评估的消费级底座** |
| **prima.cpp** | piped-ring（连续层段环形） | ✅ Halda 用 ILP 解出每设备层窗口，10–12 ms | ✅ | ❌ 无 | ❌ | ❌ 明确「mini-batching is not yet implemented」 | ❓ | — | 算法最佳、**代码仓库 404** |
| vLLM / SGLang | 跨机 PP | ✅ 配置期不等分 | ❌ | ❌（vLLM 把容错甩给 Ray） | ✅ Radix Cache | ✅ | ❌ | — | 数据中心，且 vLLM 官方**要求各节点环境完全一致** |

---

## 1. 方法与证据分级

- 工具：`web_fetch`（原始 README / 源码 / arXiv / 官方文档）、真实浏览器（GitHub 讨论页）、GitHub **Atom feed**（`releases.atom` / `commits/<branch>.atom`，用于绕过 HTML 渲染）。本机 shell **无外网**（Schannel `SEC_E_NO_CREDENTIALS`，curl 退出码 35），因此所有「下载后本地 grep」的路径都不可用，一律走抓取。
- 证据分级：**V** = 实际抓到该页/该文件并读到原文；**I** = 由 V 级事实推出的结论；**U** = 未能确认（不臆造）。
- 标注来源：**[自核]** = 本报告作者亲自核验；**[子代理核验]** = 并行子代理抓取核验（同样是 V 级，但未由作者二次打开）。
- 已知证据局限：`api.github.com` 全程 HTTP 403（本机 IP 限流），所以星标数/`archived` 标志等多取自 UI 或 Atom feed；`health.petals.dev` 在 2026-09-22 不可达。

---

## 2. llama.cpp RPC backend（**[自核]**，本报告重点）

### 2.1 是什么、并行的是什么

`ggml-rpc-server`（旧名 `rpc-server`，2026-06-27 由 PR #25045 更名）把远端主机的 ggml device 暴露给主机的 llama.cpp 进程；主机用 `--rpc host:port[,...]` 挂载这些远端 device。
文档：<https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/rpc/README.md>（**V**）

- 远端按 `-DGGML_RPC=ON` 构建；`ggml-rpc-server` 默认暴露该机所有加速器，无加速器则暴露一个 CPU device（README, 同上）。可用 `CUDA_VISIBLE_DEVICES` 或 `-d/--device` 限定；`-c/--cache` 开本地张量文件缓存（默认 `$HOME/.cache/llama.cpp/rpc`）；`GGML_RPC_DEBUG=1` 开调试；`GGML_RPC_NO_RDMA=1` 强制纯 TCP。**V**
- 主/远混合执行示例（README 原文）：`llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -ngl 99 --rpc 192.168.88.10:50052,192.168.88.11:50052`。**V**
- 老版 README（b4900）写得更直白：*"This way you can offload model layers to both local and remote devices."* 以及 *"Each host can run a different backend, e.g. one with CUDA and another with Metal."*
  <https://raw.githubusercontent.com/ggml-org/llama.cpp/b4900/examples/rpc/README.md>（**V**）
- 切分模式（官方参数表原文）：`-sm, --split-mode {none,layer,row,tensor}`，其中 **`layer`（默认）：split layers and KV across GPUs (pipelined)**；`row`：按行切权重；`tensor`：切权重与 KV（**EXPERIMENTAL**）。
  <https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md>（**V**）
- 默认分摊规则（README 原文）：*"By default, llama.cpp distributes model weights and the KV cache across all available devices -- both local and remote -- in proportion to each device's available memory. You can override this behavior with the `--tensor-split` option"*。**V**

**判定：确实支持把「一个模型的特定层」放到远端机器上**——远端机器是被当成一个 device 参与 `--split-mode layer` 的层分配，且**该层的 KV cache 也留在那台机器上**。
⚠️ 但这不是「真正的流水线并行」：llama.cpp 用的是 ggml 的调度器（把图节点分派到各 backend），远端 device 上算完的中间张量要经主机中转/拷贝。**没有跨机 stage 重叠（overlap）机制**（**I**：源码里 RPC backend 只实现 backend/device 接口与 `graph_compute`/`graph_recompute`，见下）。

### 2.2 不等分与运行时重算（R1）

- **不等分：可以。** 主机制是 `-ts, --tensor-split N0,N1,N2,...`（"fraction of the model to offload to each GPU, comma-separated list of proportions"）。**V**（参数表）
- 维护者 rgerganov 在讨论 #12714 中对「32 GB 与 64 GB 节点想拿到不同层数」的提问直接回答：
  *"You can use the -ts SPLIT, --tensor-split SPLIT command line option to specify how tensors are split across devices and RPC servers."*
  并对「per-host 层数」补充：*"It could be possible with the --override-tensor option which @slaren added, see PR #11397 for details."*
  <https://github.com/ggml-org/llama.cpp/discussions/12714>（**V**）
  → 即：**按比例不等分是官方路径；「精确指定每台机器持有哪几层」只有维护者的「可能可以」级建议（`-ot, --override-tensor <tensor name pattern>=<buffer type>`），未见官方文档承诺。** 该讨论中同一提问者追问 *"Would it be possible to have redudant layers?"* —— **0 条回复**（对 R2 是重要负面证据）。**V**
- **精确边界不保证**（**I**）：`layer` 模式下每层的归属由内存/比例决定，而非由「层号区间」指定。第三方基准仓库实测日志与之吻合：双机时打印 `offloading 28 layers to GPU (RPC)` / `offloading 4 layers to GPU (Metal)`，且其说明写 *"Layer assignment is automatic: llama.cpp queries available memory on each device and distributes accordingly"*。
  <https://raw.githubusercontent.com/kjaiswal/llama-cpp-distributed-benchmarks/refs/heads/main/README.md>（**V**，第三方 MIT 仓库）
- 相关辅助参数：`--fit/-fit`、`-fitt/--fit-target`（自动按显存调整未指定参数）、`-ngl/--gpu-layers`、`-ot`。**V**
- **运行时重算：不能**（**I**，但证据强）：切分发生在模型加载期（`llm_load_tensors`），控制手段全是**启动命令行参数**；server README 的完整 endpoint 清单里**没有任何**与设备放置相关的 API（只有 `/props`、`/metrics`、`/slots/*`、`/v1/*` 等）。要换切分=重启进程=重新加载权重（对 RPC 还要重新把权重推到远端）。

### 2.3 容错行为（R2）——最关键的一节

**当前 master（2026-09-22 抓取）的行为：远端机器消失 = 本机进程 abort。**
源码原文：

```cpp
// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) if (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")
```

该宏被用在 dispatcher 的工作循环里对**每一条命令**做断言；连接阶段失败则是 `GGML_ABORT("Failed to connect to %s")`、`GGML_ABORT("RPC handshake failed for %s")`。
来源：`ggml/src/ggml-rpc/ggml-rpc.cpp`
<https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-rpc/ggml-rpc.cpp>（**V**）

**有一个正在推进但未合入的 PR #26724 "rpc : do not abort the process when the remote server fails"（作者 erwinzhang7），其 diff 本身就是最好的「没有恢复」证据**（**V**，抓到完整 diff）：
<https://patch-diff.githubusercontent.com/raw/ggml-org/llama.cpp/pull/26724.diff>

- PR 在 `rpc_dispatcher` 上加 `std::atomic_bool failed`，注释直言：*"the buffers on the server are gone, so a lost connection can never be recovered"*；失败时打印 *"lost connection to RPC server %s, the device is now unusable"*。
- `ggml_backend_rpc_graph_compute` 遇到 failed 直接返回 `GGML_STATUS_FAILED`。
- server 侧新增 `has_error`，一旦触发，**之后每个 HTTP 请求都返回 503** 且消息为 *"Compute device failed, the server must be restarted"*。
- 新增测试 `tests/test-rpc-crash.cpp`：`kill(server_pid, SIGKILL)` 后轮询 `ggml_backend_graph_compute`，要求最终返回 `GGML_STATUS_FAILED`。
- **该 PR 未合入 master**：2026-09-22 抓取的 master 里仍有 `RPC_STATUS_ASSERT` 宏、且没有任何 `is_failed` 符号（**V**）。即便合入，语义也只是「把崩溃变成永久失效 + 503」，**在途请求不恢复、流水线不重组**。

其他相关证据：

- RPC 核心仍在高频维护，且近期一次提交修的是 RPC 的**远程代码执行**漏洞：`rpc : invalidate cached compute graph when a referenced buffer is freed (#24292)`（2026-09-16）原文称 *"The bug is reachable by an unauthenticated remote client ... yielding remote code execution."*
  <https://github.com/ggml-org/llama.cpp/commits/master/ggml/src/ggml-rpc.atom>（**V**）
- 同一 feed 显示 2026-08-25 的 Apple RDMA PR 里明确 *"remove transparent reconnect"*（去掉透明重连）。**V**
- README 顶部警告框：*"This example and the RPC backend are currently in a proof-of-concept development stage. As such, the functionality is fragile and insecure. **Never run the RPC server on an open network or in a sensitive environment!**"*（**V**）—— 对「WiFi 局域网跑 4 台机器」是直接相关的安全约束。
- 第三方基准仓库「Known Issues」还记录：*"Same commit matters. RPC protocol isn't versioned — mismatched builds between host and remote can cause silent failures or crashes."*（**V**，第三方）

**小结（R2）：** 现状是**崩溃**；合入 #26724 后是**永久失效 + 503，必须重启 server**。冗余层（redundant layers）被问到但无人回答。**没有任何在途请求恢复机制。**

### 2.4 KV / 前缀缓存（R3）

- KV cache 本身**跟着层一起被分摊到各 device（含远端 RPC device）**（RPC README 原文，§2.1 已引）。含义：远端机器一走，它那部分层的 KV 就没了。**V**
- llama-server 侧确有**前缀复用**能力（**V**，参数表与 endpoint 文档）：
  - `-cram, --cache-ram N`（最大缓存 MiB，默认 8192，-1 无限，0 关闭）
  - `--cache-idle-slots`（新任务时把空闲 slot 存入 prompt cache）
  - `POST /slots/{id_slot}?action=save|restore|erase`（把 slot 的 prompt cache 存/取文件，需要 `--slot-save-path`）
  - 响应里 `timings.cache_n` = "number of prompt tokens reused from cache"，以及 `usage.prompt_tokens_details.cached_tokens`
- **cache-aware 路由：没有**（**V/I**）：router 模式原文只说 *"Requests are routed according to the requested model name"*，按模型名路由，不带缓存亲和性；`--tags` 也明确标注 "informational, not used for routing"。

### 2.5 调度可替换性 / 可观测性（R5）

- **不可插拔**（**I**，证据强）：切分/调度逻辑在 llama.cpp + ggml 内部（加载期层分配 + `ggml_backend_sched`），**没有** scheduler 接口、注册表或策略插件；要改只能改源码。最接近「策略旋钮」的是 `-ot/--override-tensor`（按张量名模式指定 buffer type）。
- **可观测性尚可**（**V**）：`--metrics` 暴露 Prometheus 指标，文档列出的指标名如 `llamacpp:prompt_tokens_total`、`llamacpp:tokens_predicted_seconds_total`、`llamacpp:predicted_tokens_seconds`、`llamacpp:requests_processing`、`llamacpp:requests_deferred`、`llamacpp:n_busy_slots_per_decode`、`llamacpp:spec_decode_*`；另有 `--perf`、`-lv/--log-verbosity`、`--log-jsonl`；RPC 自身只有 `GGML_RPC_DEBUG`。

### 2.6 许可 / 成熟度 / 日期（**V**）

- 许可：**MIT**，"Copyright (c) 2023-2026 The ggml authors"
  <https://raw.githubusercontent.com/ggml-org/llama.cpp/master/LICENSE>
- 最新 release：**b11103，2026-09-22T14:56:42Z**（同日多次发版）；master 最新提交 2026-09-22T13:54:45Z
  <https://github.com/ggml-org/llama.cpp/releases.atom>、<https://github.com/ggml-org/llama.cpp/commits/master.atom>
- RPC 路径提交活跃度（`tools/rpc` feed）：2026-09-17 `rpc : skip ACCEL devices (#29020)`；2026-08-30 `rpc: avoid serializing buffers from other servers (#26500)`；2026-08-25 Apple RDMA (#26421)；2026-06-27 更名为 `ggml-rpc-server` (#25045)。`ggml/src/ggml-rpc` feed：2026-09-16 安全修复 (#24292)、2026-09-15 hash-cache 只缓存权重 (#28789)、2026-08-26 event/async backend API (#18626)、2026-08-18 use_count (#27142)、2026-07-29 tensor_memset (#25912)、2026-05-19 / 2026-05-05 graph uid、2026-04-19 transport 重构 (#21998)、2026-04-15 RoCEv2 RDMA (#20590)、2026-04-09 **backend-agnostic tensor parallelism（experimental）(#19378)**。
  → **项目极度活跃**；但 **RPC backend 自身仍自称 proof-of-concept / fragile / insecure**。
- 仓库体量（讨论页读到）：Star ~129k、Fork ~23.6k。

### 2.7 性能数字（g）

- **官方 README 不给 RPC 性能数字**：b4900 版与当前 `tools/rpc/README.md` 都**没有** benchmark 表（**V**，两份都读过）。所以「官方文档化的 RPC 性能数字」= **不存在**。
- 第三方实测（10 GbE 直连，iperf3 9.41 Gbps；Mac Studio M2 Ultra/Metal + DGX Spark GB10/CUDA 13）。**V**（第三方 MIT 仓库 <https://github.com/kjaiswal/llama-cpp-distributed-benchmarks>）：

| 模型 | 模式 | Prompt tok/s | 生成 tok/s | 切分 |
|---|---|---|---|---|
| Qwen2.5-7B Q4_K_M (4.4 GB) | 本地 Metal | 76.1 | **91.8** | 全本地 |
| 同上 | RPC Metal+CUDA | **317.7** | 52.7 | — |
| Qwen2.5-72B Q4_K_M (44.2 GB) | 本地 Metal | 28.2 | **11.1** | 44 GB 全 Metal |
| 同上 | RPC Metal+CUDA | 29.5 | 5.9 | 30.7 GB Metal + 13.8 GB CUDA |

  其结论原文：*"RPC is for capacity, not speed."*、*"Decode (token generation) is slower with RPC … each token requires a network round-trip (~0.17 ms on 10GbE) … Latency compounds: ~2x slower decode on 7B, ~47% slower on 72B."*、以及 **"For 1GbE instead of 10GbE: expect ~10x more network overhead on decode."**
  → **对 WiFi 的推论（I）**：本项目的 WiFi 往返时延通常 2–10 ms，比 10 GbE 的 0.17 ms 差 1–2 个数量级，比文中给 1 GbE 的「~10× decode 网络开销」还要更差。**单 token 依次跨 3 跳 WiFi 的 decode 时延极可能被网络主导**，这一点必须先做实测再定架构。
- 另有一条同源证据（**V**）：`rpc : hash-cache only weights (#28789)` 的提交说明提到双机 split Qwen3.8-Flash-Next 时，调度器在两机之间搬运的 activation 每个 prefill ubatch 都 >10 MB，一天往 worker 缓存目录写了 1.4 TB。→ 侧面证明**跨机传的是 activation（每 ubatch 级），且历史上存在严重的 I/O 放大 bug**。

### 2.8 工程风险（R6）具体条目（**V**，源码）

`ggml-rpc.cpp` 里两处 TODO 直接说明成熟度：

```cpp
static enum ggml_backend_dev_type ggml_backend_rpc_device_get_type(...) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}
static bool ggml_backend_rpc_device_supports_op(...) {
    //TODO: call the remote backend and cache the results
    return true;
}
```

即：**远端 device 一律被当成 GPU 上报，且 `supports_op` 无条件返回 true**——不会被正确拒绝的算子会在运行期炸掉。加上无鉴权/无加密（且有已修复的 RCE）、协议不版本化、失败即 abort，这些都是必须计入的风险。

---

## 3. exo（**[子代理核验]**）

仓库：<https://github.com/exo-explore/exo>　README：<https://raw.githubusercontent.com/exo-explore/exo/main/README.md>

**(a) 并行方式**：`Sharding` 枚举只有 `{Tensor, Pipeline}`，**Pipeline 是默认**（`GET /instance/placement` 默认 `sharding: Sharding = Sharding.Pipeline`）；`PipelineShardMetadata` 是半开区间 `[start_layer, end_layer)`；执行在 `src/exo/worker/engines/mlx/generator/generate.py` 的 `pipeline_parallel_prefill()`，有 `PipelineFirstLayer/PipelineLastLayer` 与 rank 交错 dummy iteration。无 expert parallelism。
源码：`src/exo/shared/types/worker/shards.py`、`src/exo/worker/engines/mlx/generator/generate.py`（`https://raw.githubusercontent.com/exo-explore/exo/main/...`）

**(b) 不等分 / 运行时重算**：`allocate_layers_proportionally(total_layers, memory_fractions)` 用最大余数法按 `ram_available / total_memory` 分配——**按空闲主机 RAM，不是 VRAM，也不是速度**；**无法指定 A/B/C/D 边界**。可经 API 重算（`GET /instance/previews`、`GET /instance/placement`、`POST /place_instance`、`POST /instance`、`DELETE /instance/{id}`），但**重算是破坏性的**：新建实例、删旧实例、任务被取消。**没有 CLI flag**（`src/exo/main.py` 的完整参数表里没有 `--shard/--placement/--layers`）。
源码：`src/exo/master/placement.py`、`src/exo/master/placement_utils.py`、`src/exo/main.py`

**(c) 容错（关键）**：`src/exo/master/main.py` 的 `_plan()` 每 10 s 扫一次：发现某节点不在 topology 里 → 对整个实例发 `InstanceDeleted`；另设 30 s 无活动 → `NodeTimedOut`。随后 `src/exo/api/main.py` 对 `InstanceDeleted` 执行 `self._close_streams_for_instance(...)`，把 Pending/Running 任务标为 `TaskStatus.Cancelled`。**没有在途序列 checkpoint、没有重新 prefill、没有重新安置、没有重试。**
「无 master 节点」为真但范围很窄：每个节点都起一个 Master，选举按 `(clock, seniority, commands_seen, node id)`，默认 `DEFAULT_ELECTION_TIMEOUT=3.0`；新 master 上任后重启 worker/download-coordinator/API 并重置 API 状态——**只提供领导权 failover**。
源码：`src/exo/master/main.py`、`src/exo/api/main.py`、`src/exo/shared/election.py`

**(d) KV/前缀缓存**：`KVPrefixCache`（`src/exo/worker/engines/mlx/cache.py`）做最长公共前缀匹配 + LRU + 按设备大小的阈值（`EXO_MEMORY_THRESHOLD` 默认按 ≥128/≥64/≥32/<32 GB 取 0.85/0.80/0.75/0.70）。**唯一「分布式」的部分**是用 `mx.distributed.all_gather` 让驱逐策略取集群最大压力；tensor 只在本进程内，**不可迁移、不持久**（只有 event log 与 traces 落盘）。**路由不是 cache-aware**：master 选「在途任务最少」的实例（`sorted(instance_task_counts.keys(), ...)[0]`）。**缓存随节点消失而消失。** README 里的 "RDMA over Thunderbolt"/"MLX ring" 是传输层，不是分布式 KV 存储。
`ENABLE_DISAGGREGATION=false`（默认关），prefill/decode 分离为实验特性，远端 prefill 失败会回退本地（"Remote prefill failed, falling back to local prefill"）。

**(e) 调度可替换/可观测**：可观测 **✅**——事件溯源状态、`GET /events`、`GET /state` + `/state/{path}`、磁盘 event log、`EXO_TRACING_ENABLED` + `/v1/traces`、`/v1/traces/{task_id}`（含 by_category/by_rank 的 count/min/max/avg）、`/v1/traces/{task_id}/raw`、`POST /bench/chat/completions`（prompt_tps/generation_tps/peak_memory_usage）、按阶段能耗拆分。**不可插拔 ❌**——`place_instance(...)` 是硬编码函数，被 API 与 `Master._command_processor` 直接 import 调用；打分是内联 `max(candidate_cycles, key=lambda cycle: (_cycle_download_score(...), sum(ram_available)))`；`Sharding` 只有 2 个值。替换策略 = fork `master/placement.py` + `placement_utils.py`。

**(f) 许可/成熟度/日期**：**Apache-2.0**（LICENSE 附录 "Copyright 2025 Exo Technologies Ltd"）；最新 release **v1.0.71，2026-04-23T15:04:26Z**；master 最新提交 **21a54c5，2026-08-25T18:59:53Z**。成熟度：研究/爱好者级 beta（v1.0.66、v1.0.68 自称 stability release；PLATFORMS.md 标题为 "partial roadmap"）。版本号漂移：`pyproject.toml` 写 `version = "0.3.70"` 而 tag 是 v1.0.x；docs/api.md 指向的 `src/exo/master/api.py` 已 404（真实文件是 `src/exo/api/main.py`）。

**(g) 性能**：README 称张量并行 "up to 1.8x speedup on 2 devices and 3.2x speedup on 4 devices"；"RDMA over Thunderbolt 5, enabling 99% reduction in latency"；旗舰 benchmark 是 4×M3 Ultra Mac Studio 512 GB 跑 Qwen3-235B / DeepSeek v3.1 671B / Kimi K2（图片，具体 tok/s **未核验**）。v1.0.69 记 "pipeline parallel prefill up to 1.98x faster on 2 nodes"。**没有 WiFi 或混合 GPU 的公开数字**（U）。

**(h) 平台/R6（决定性）**：README 原文 *"Currently, exo runs on CPU on Linux. GPU support for Linux platforms is under development."*、*"On macOS, exo uses the GPU. On Linux, exo currently runs on CPU."*；PLATFORMS.md 里 Tier 1 只有 Apple Silicon，Tier 2/3 为空，"Linux CUDA Support" 仍在 **Planned**。即 **4 台 NVIDIA 笔记本默认只能跑 CPU**（除非使用来自个人 fork 的非官方 `mlx-cuda` wheels）。且旗舰特性要求 **Thunderbolt-5 全互联 RDMA**（*"Devices that wish to be part of an RDMA cluster must be connected to all other devices"*），WiFi 只能拿到邻居级 `MlxRing`。另外 `--bootstrap-peers` 在当前 main 是死代码（`raise ValueError("Bootstrap peers has been temporarily removed")`），静态 peer 发现不可用。要求 Python **`==3.13.*`**（精确锁定）。

---

## 4. distributed-llama（**[子代理核验]**）

仓库：<https://github.com/b4rtaz/distributed-llama>　README：<https://raw.githubusercontent.com/b4rtaz/distributed-llama/main/README.md>

**(a) 并行方式是张量并行，不是流水线（重大更正）**：
- README：*"More devices mean faster performance, leveraging **tensor parallelism** and high-speed synchronization over Ethernet."*；v0.11.0 README 开篇 *"Tensor parallelism is all you need."*
- 源码 `src/llm.cpp` 的 `buildLlmNet()`：**node 循环在外、layer 循环在内**，即**每个节点都为所有层构建 segment**：`for (nodeIndex...) { ... for (layerIndex = 0; layerIndex < h->nLayers; layerIndex++) { ... } }`
- 被切的是每层的权重矩阵（`sliceRowMatmul` q/k/v/w1/w3/wcls、`sliceColMatmul` wo/w2）、KV cache（`sliceKvCache`）、heads（`sliceMultiHeadAtt`）、rope（`sliceRope`）、logits（`vocabSize / nNodes`）。
- 同步：每层每 token **两次 all-gather**（`addSync(zqPipeIndex, SYNC_NODE_SLICES)` 于 att 与 ff），`syncNodeSlices()` 与**所有** peer socket 交换 → 底层是**全网状**，不是链/环。
源码：`src/llm.cpp`、`src/nn/nn-core.cpp`、`src/nn/nn-executor.cpp`（`https://raw.githubusercontent.com/b4rtaz/distributed-llama/main/src/...`）
- 精确 CLI（`src/app.cpp` 的 `AppCliArgs::parse()`）：模式是位置参数 `dllama inference|chat|worker|perplexity`，另有 `dllama-api`；flag 仅 `--model --tokenizer --buffer-float-type {f32|f16|q40|q80} --workers <ip:port ...> --port --host --nthreads --steps --prompt --temperature --topp --seed --chat-template --max-seq-len --gpu-index --gpu-segments <from>:<to> --net-turbo`。**不存在** `--nlayers` / `--layer-*` / `--compute` / `--sync` / `--kv-cache*` / `--cache-reuse` / `--nbatches`（`nBatches` 硬编码 32）。

**(b) 不等分：不可能**。`src/nn/nn-core.cpp` 里全是硬 assert 等分：`sliceRowMatmul: assert(d % nNodes == 0); s.d0 = d / nNodes;`、`sliceColMatmul: assert(n % nNodes == 0);`、`sliceKvCache: assert(kvDim % nNodes == 0);`、`sliceMultiHeadAtt: assert(nHeads % nNodes == 0);`、`sliceRope: assert(qDim % nNodes == 0); assert(kvDim % nNodes == 0)`。无显存探测、无配置、无 runner flag。切分在 root 启动时算一次（`buildLlmNet()` → `NnRootConfigWriter::writeToWorkers`），**无运行期重规划**。节点数还受 README「only on 1, 2, 4... 2^n nodes」「maximum number of nodes is equal to the number of KV heads」约束（`if (nNodes > header.nKvHeads) throw`；issue <https://github.com/b4rtaz/distributed-llama/issues/70>）。**唯一的「不等分」是节点内部 GPU/CPU 的 `--gpu-segments <from>:<to>`**（代码里有、文档里没有）。
→ **A:0-11 / B:12-15 / C:16-19 / D:20-27 无法表达**，要它就得改 `buildLlmNet` + 所有 slicer + 权重加载器 + 同步逻辑。

**(c) 容错**：作者 v0.16.3（2025-10-26）发布说明称 *"If any worker crashes, the API automatically attempts to reconnect to the failed node and reinitialize the cluster."*
<https://github.com/b4rtaz/distributed-llama/releases/tag/v0.16.3>
代码实际行为（**V**）：worker 中途死亡 → `NnTransferSocketException` → `NnExecutorException("Execution failed in one of the threads")` → `dllama-api` 的 `main()` 捕获后 **3 秒后整簇 re-init**；`nNodes = nWorkers + 1` **永不改变**，永远连同一份 worker 列表，**永不缩容到 3 节点**；每次成功重连都**重传全部权重**（Llama-3.1-8B 约 6.32 GB）；在途请求的 socket 直接被销毁 → 客户端看到连接断开，**没有错误 JSON、没有最后一个 SSE frame、没有重放**；KV 与 `NaiveCache` 全部丢失；`dllama inference/chat` 不重试，直接 `EXIT_FAILURE`。另有一个未捕获路径：`NnTransferSocketException` 不是 `NnExecutorException` 子类，主线程 socket 错误会逃出 catch → `std::terminate`（层级关系 V，触发概率 I）。
Issue 佐证：#48、#140、#209、#157、#105、#108、#87、#8（均为 closed，内容见各 issue）。**没有任何 issue/文档/代码路径描述重规划、请求恢复、重放或 checkpoint。**

**(d) KV/前缀缓存**：当前**没有** KV cache flag（v0.10.0 曾有 `--kv-cache-storage <ram|disc>`，v0.12 重构后删除）。KV 是**分片**的（每节点每层 `kvDim/nNodes`），因此复用要求集群成员完全一致。同一 root 进程内跨请求/跨轮次**能**复用：`chat` 靠递增 `pos` 隐式复用；`dllama-api` 有显式 `NaiveCache`（`resolveDeltaPrompt` 命中前缀则只 forward delta，打印 `🐤 Found naive cache for ...`；不匹配则 `cache.clear()` 全量重算）。限制：**单一全局缓存**（同时只能一个会话）、无 per-session key、无哈希块缓存、无驱逐、**无 cache-aware 路由**；任何重启/恢复即丢失。**无并发批处理**：accept 循环单线程、一请求跑完再收下一个。

**(e) 调度可替换/可观测**：**不可替换**——`std::vector<NnExecutorStep>` 由 C++ 硬编码，worker 通过线缆收到同一份 plan；无配置文件、无接口、无运行时重排。可观测性仅打印：`🔷️ Eval <ms> Sync <ms> | Sent <kB> Recv <kB>`、tokens/s、字节计数器、`📀 RequiredMemory: N MB`；**无 metrics endpoint、无 JSON 日志、无 tracing、无 Prometheus/OTel**（issue #39 曾专门要求，未实现）。

**(f) 许可/成熟度/日期**：**MIT**（"Copyright (c) 2024 Bartłomiej Tadych (b4rtaz)"）；最新 release **v0.16.5，2026-02-02**；最后提交 **2026-07-05T16:47:20Z**（`59af889`）。活跃但非生产级：2^n 节点、nKvHeads 上限、只支持两种量化组合、Vulkan 标 "🌋 Experimental"、转换器自称 "in the early stages of development. After conversion, the model may not work correctly."

**(g) 性能（含关键负面数字）**：
- Raspberry Pi 5 8 GB（来自 v0.11.0 README）：Llama 2 7B 1×2.26 / 2×2.92 / 4×4.56 t/s；Llama 3 8B 1×1.77 / 2×2.25 / 4×3.01 t/s。
- 每生成 token 的网络传输量（Llama-3-8B）：Q80 2/4/8 节点 = 544/1632/3808 kB。
- **真实 4 节点 1 GbE 集群（issue #294，Intel N150，llama3.2-1b-q40）：1 节点 Pred 11.00 t/s；4 节点 Pred 只有 1.85 t/s，逐 token `Pred 22–26 ms` vs **`Sync 442–561 ms`**。** <https://github.com/b4rtaz/distributed-llama/issues/294>
  → **多节点 decode 完全被同步主导（比单机慢约 6 倍）**。

**(h) GPU**：**仅 Vulkan**（Makefile 无 CUDA/HIP/Metal 目标），`--gpu-index` 在 root 或 worker 都可用；`--gpu-segments` 切一层内部 GPU/CPU。CUDA 相关 issue #275、#35 仍 open。**模型格式是自研 `.m`**（magic `0xA00ABCD`），**不支持 GGUF**（issue #148 关闭为 not planned），运行时只接受「q40 权重 + q80 buffer」或「f32 + f32」。API 只有 `POST /v1/chat/completions` 与 `GET /v1/models`，无鉴权。

**判定**：可作为「等分张量并行」的 baseline/引用对象（Pi 与 RTX 数字都是好的对照点），**不能**作为异构分层流水线引擎。

---

## 5. Petals（**[子代理核验]**）

仓库：<https://github.com/bigscience-workshop/petals>　论文：<https://arxiv.org/abs/2209.01188>、伴随论文（NeurIPS'23）<https://arxiv.org/abs/2312.08361>（正文读 ar5iv HTML）

**(a) 并行方式**：跨互联网的**层切分流水线并行**；跨 peer 无张量并行（只有单机内实验性 `--tensor_parallel_devices`）。论文 §1：*"A server hosts a subset of model layers ... A client can form a chain of pipeline-parallel consecutive servers"*；2312.08361 §3.2：客户端只持输入/输出 embedding（BLOOM-176B 上 <3%），把 transformer block 委托给远端。激活可绕过客户端（默认 `use_server_to_server=True`）。

**(b) 不等分 / 运行时重算**：**两者都支持，且是本项目见到的最完整的「可钉死 + 可自动重平衡」实现**。机制名："server load balancing"/"Automatic load balancing"/"rebalancing"；规则是每个 server「装尽可能多的 block 装到显存放不下」；新 server 认领「覆盖吞吐最差 block 的连续区间」；贪心可达最优的 90–100%；重平衡仅在吞吐提升 ≥ p 时提交（论文取 p=20%，"each block replacement resets attention caches for this block"）；**每 ~60 s 重算一次**。相关 flag（**V**，来自 `petals/cli/run_server.py`）：`--balance_quality`（默认 0.75，0.0 关闭重平衡）、`--mean_balance_check_period 60`、`--update_period 120`、`--throughput auto|eval|dry_run|<RPS>`、`--num_blocks N`、`--block_indices START:END`（end 开区间）。FAQ 原文：*"disable block switching completely by setting --balance_quality 0 or pin the server to a certain range of blocks ... `--block_indices 43:46`"*。
→ **我们的 A/B/C/D 可直接表达**：`--block_indices 0:12 / 12:16 / 16:20 / 20:28` + `--balance_quality 0`（**I**，机制 V、组合方式为推断）。

**(c) 容错（最关键）**：**设计上有，工程上不成立。**
- 设计（2312.08361 §3.2「dual attention caches」）：服务端缓存 KV，**客户端缓存「发给该 stage 的历史输入」**；*"If a server disconnects, a client can find another server with that pipeline stage and use client-side cache to restore the server state."* 重跑范围**不是从第 0 层**：*"the algorithm needs to send O(t) data (in one round) for each failed server and compute only the stages held by the failed servers."* 2209.01188 §3.2：*"the client sends all previous inputs to the replacement server, so that it has the same attention keys and values."* 术语：inference session / client-side cache / server-side cache / `replace_failed_server` / rebalancing。
- **量化（2312.08361 §4.1 Table 1；BLOOM-7.1B，4 stage (8,7,8,7) 于 4×1080Ti——几乎就是我们的拓扑）**：1024 token 时，p=1e-3：Petals **7.76** vs 「Caching with restarts」（整段重跑）**0.48** steps/s；p=1e-2：2.17 vs 1 小时内跑不完。**没有**任何「单次失败的恢复时延(ms)」数字（U）。
- **致命障碍 1（结构）**：论文 §3.2 明写 *"For simplicity, we assume that every block is hosted on several servers."* 我们的 A/B/C/D 各只有一台，B 一死就**没有**持有 12-15 的 peer，客户端抛 `MissingBlocksError`（`sequence_manager.py`："No servers holding blocks {block_indices} are online..."）。**要容错就必须给每个 block range ≥2 个持有者（等于 8 个 peer）。**
- **致命障碍 2（实现已坏）**：issue **#587 "Petals doesn't deal with server failure properly"** 仍 **open**；维护者 justheuristic：*"the issue certainly exists and I can confirm that it breaks fault tolerance on my side. We have not yet fixed it..."* 代码里 `_ServerInferenceSession` 文档字符串自己写 *"This class is not fault-tolerant out of the box"*；恢复路径把 `updated_sessions[0].history = failed.history` 赋给一个 `_position=0` 的新 session，违反 `assert server_session.position == self.position` 与 `assert self.history.shape[1] == self._position + n_input_tokens`（"Broken input cache"）。PR #588（2024-07-09 合入）/#594 只加了**手动**的 `InferenceSession.position` 回滚。
→ **对我们的结论（I）：B 笔记本在生成中途被合盖 = 请求死亡。论文的恢复机制是「该抄的设计」，不是「已发布的功能」。**

**(d) 加入 swarm**：公开 swarm 的 bootstrap 常量在 `petals/shared/constants.py`（`/dns/bootstrap1.petals.dev/tcp/31337/p2p/QmedTaZ...`、`bootstrap2...:31338`），可达性 API `https://health.petals.dev`（**注意：2026-09-22 该域名不可达，公开 swarm 存活情况 U**）。**私有/局域网 swarm 有官方文档**（wiki "Launch your own swarm"），明确推荐用于 *"geo-distributed and/or unreliable GPU machines (e.g., spot instances)"*；需 `python -m petals.cli.run_dht ...` + `run_server`（首台 `--new_swarm`，其余 `--initial_peers`），每 peer 需唯一 `--identity_path`；局域网原文：*"If you run your swarm in a local network only, it's fine to don't have a public IP and ports as long as you use local network's IP addresses everywhere."* **DHT 必需；health monitor 不必需**（`--skip_reachability_check`，私有 swarm 默认跳过）。

**(e) KV/前缀缓存**：服务端 per-session attention KV（`--attn_cache_tokens` 默认 MQA 16384 / 其他 4096，`--inference_max_length`、`--session_timeout 1800`、`--step_timeout 300`、`--max_alloc_timeout 600`）；客户端 `_ServerInferenceSession.history` 存「已发送的输入」，用途明确是 *"Used in case of server failures to regenerate attention caches on new servers"*。**前缀复用仅限同一 session 内**（`remote_generation.py`：*"remote servers store your attention caches and you don't have to rerun the prefix ... Supports multiple .generate() calls inside one InferenceSession"*）；客户端 cache 是假的（*"pretends to be a legit cache"*）。**未发现跨请求/跨用户前缀缓存**（I）。**cache-aware 路由：有**——`make_sequence(..., mode="min_latency", cache_tokens_needed=self._max_length)` 与 `_has_cache_for(span, ...)` 使用服务端在 DHT 公告的 `cache_tokens_left`，缓存不足时加 `alloc_delay=10` 惩罚。

**(f) 时延数字（V）**：2209.01188 Table 3（BLOOM-176B，seq 128/2048，steps/s）：3×A100 1 Gbit/s <5 ms → 1.71/1.54；100 Mbit/s <5 ms → 1.66/1.49；100 Mbit/s/100 ms → 1.23/1.11；12 虚拟 server → 1.24/1.06、0.57/0.53；**14 台真实异构 server 跨欧美 → 0.83/0.79**（并行 forward 32.6/179.4 tok/s）。2312.08361：Llama-2-70B on 3×T4 → 2.29/2.02（1 Gbit/s <5 ms）、1.57/1.44（100 ms RTT）。关键结论原文：*"For inference, performance does not depend much on bandwidth or sequence length but degrades with higher latency."* → **局域网 RTT <5 ms 那一行适用**，但 4 跳 WiFi 仍然主导 step time。

**(g) 维护状态（V）**：最后提交 **2024-08-25**（"Upgrade Pydantic to >= 2.0.0 (#607)"）；最新 release **v2.2.0（2023-09）**，PyPI `2.2.0.post1`（2023-11-20）；main 版本串 `2.3.0.dev2` 未发布。**未归档**（UI 显示 Public，非 archive；machine-readable `archived` 标志因 API 403 未核验 → U），README 无弃用横幅。依赖锁死：`transformers>=4.43.1,<4.44.0`（`petals/__init__.py` 里硬 assert）、`bitsandbytes==0.41.1`、`hivemind==1.1.10.post2`、`peft==0.5.0`。第三方 fork Kwaai-AI-Lab/OpenAI-Petal 2025-11 宣布维护模式，理由是 transformers 4.43.1 的 CVE 与 *"Dependency hell: Every update breaks"*（第三方，非上游）。

**(h) 许可**：**MIT**（"Copyright (c) 2022 Petals authors and collaborators"）。

**(i) 4 笔记本可行性**：显存预算经验值 ~1.1 GiB/1B 参数（int8）、~0.7 GiB/1B（nf4），LLaMA-65B 约 50 GiB、BLOOM-176B 约 200 GiB → **22 GB 总显存（8+6+4+4）约能放 15–20B int8 / 30B nf4 上限，实际应规划 7–13B 模型**；70B/176B 放不下。硬件门槛为 server ≥16 GB 内存/≥8 GB VRAM/100 Mbit/s（FAQ 放宽到「有 GPU + 4 GB 显存即可」）。无文档化的最小 block 数（U）；约束是每台至少装 1 个 block 且 0..N-1 全覆盖，否则 `MissingBlocksError`。

**(j) 策略可替换/可观测**：可观测 **✅**——DHT 的 ServerInfo 记录含 `state/throughput/start_block/end_block/network_rps/forward_rps/inference_rps/cache_tokens_left/...`（`data_structures.py`）；客户端打印选定路由（`show_route="inference"` → "Route found: 0:12 via ..."）；服务端 `--stats_report_interval`；`--throughput auto|eval|dry_run`；`health.petals.dev` 可自建。**可替换 ❌** —— 没有 block 分配算法的插件接口（`--custom_module_path` 只用于自定义 nn.Module）；替代方案是用 `--block_indices` + `--balance_quality 0` 静态覆盖。客户端路由策略可通过 `ClientConfig` 字段（`initial_peers/allowed_servers/blocked_servers/use_server_to_server/max_retries/...`）调节。

**判定**：架构上「就是我们的设计」（层切分、不等分、可钉死、局域网私有 swarm、int8/nf4 适配 4–8 GB 显存、MIT），但**不能当引擎**：恢复需要每区间 ≥2 副本（4 个唯一 stage 做不到）、实现被维护者承认已坏且未修、项目休眠、依赖锁死、显存上限只到 ~13B。

---

## 6. ktransformers / PowerInfer / PowerInfer-2（**[子代理核验]**）

**结论：三者全是严格单机/单设备，都不能承载跨机分层。**

### ktransformers　<https://github.com/kvcache-ai/ktransformers>（org 未迁移）
- **(a) 严格单机**：官方博客 *"This hybrid design allows trillion-parameter models to be deployed on a single machine with limited GPU memory"*、*"While KTransformers focuses on single-GPU setups and high-efficiency CPU cooperation, SGLang excels at scaling across multiple GPUs"*（<https://www.lmsys.org/blog/2025-10-22-KTransformers/>）。搜索 README / kt-kernel README / 教程 / FAQ / issue（中英文「多机」「分布式」「multi-node」「distributed」）**均无多节点推理模式**。唯一的 "distributed" 是 SFT 训练的 FSDP2（v0.7.0 发布说明）。
- **(b) 并行方式**：单机内 **MoE 专家级 CPU/GPU offload**（hot expert→GPU、cold expert→CPU），不是层 offload。参数名（**V**）：`--kt-method {AMXINT4|AMXINT8|RAWINT4|FP8|...|LLAMAFILE}`、`--kt-weight-path`、`--kt-cpuinfer`、`--kt-threadpool-count`、`--kt-num-gpu-experts`、`--kt-gpu-experts-ratio`、`--kt-expert-placement-strategy {uniform|frequency|front-loading|random}`、`--kt-max-deferred-experts-per-token`、`--kt-enable-dynamic-expert-update`。**不存在** `--multi-gpu` / `--mode` 这类 flag（子代理明确更正了任务书里的假设）。核名：Intel AMX INT4/INT8、AVX512 VNNI/BF16/VBMI、llamafile CPU 后端、AMD BLIS、oneDNN、CUDA。（<https://raw.githubusercontent.com/kvcache-ai/ktransformers/main/kt-kernel/README.md>）
- **(c) 异构放置**：跨机 **无**；机内是 **per-expert / per-operator，不是 per-layer**。决定性引文：*"**Unlike traditional layer-based or KVCache offloading (as seen in llama.cpp), we offload the expert computation to the CPU and MLA/KVCache to GPU**"*（<https://raw.githubusercontent.com/kvcache-ai/ktransformers/main/doc/en/DeepseekR1_V3_tutorial.md>）。
- **(d) 容错**：无任何恢复/重试/检查点文档（U，未找到明确否认，未臆断）。
- **(e) 前缀缓存**：**有**，三层 GPU-CPU-Disk KV 复用，配置 `kvc2: {gpu_only: false, cpu_memory_size_GB: 500, disk_path: ...}`、`attn: {page_size: 16, chunk_size: 256}`，需 `USE_BALANCE_SERVE=1`；原文 *"Deleting KVCache is not supported now."*（<https://raw.githubusercontent.com/kvcache-ai/ktransformers/main/doc/en/prefix_cache.md>）——纯单机 DRAM/磁盘，**无跨机 KV 传输**。
- **(f) 批处理**：**有 continuous batching + chunked prefill**（`balance_serve`，`--max_batch_size`、`--chunk_size`）；**SGLang 集成是当前主路径，但必须用其 fork `sglang-kt`**（原文：*"Use `sglang-kt` (kvcache-ai fork), not the official `sglang` package. ... `pip uninstall sglang -y`"*）。**无 vLLM 集成**（原文明说 vLLM 是不同定位）。
- **(g) 许可/日期**：**Apache-2.0**；最新 release **v0.7.1，2026-09-15**；`main` 最后提交 **2026-09-19**；SOSP'25 论文。
- **(h) 性能**：DeepSeek-R1-0528 FP8、8×L20 + Xeon Gold 6454S、TP=8：227.85 tok/s 总 / 87.58 tok/s 输出；Qwen3-Next-80B-A3B-FP8、4×RTX 4090、TP4：52.72–114.26 tok/s；SOSP'25 摘要：prefill 4.62–19.74×、decode 1.25–4.09× 加速。
- **(i) 判定**：**严格单机**。它解决的是相反的问题：让**一台**机器靠 CPU DRAM 承载巨大 MoE。若要借用，只能作为「某台机器内部的 serving 组件」，而且它无法给出跨机异构层数。

### PowerInfer　**仓库已迁移**：`SJTU-IPADS/PowerInfer` → **<https://github.com/Tiiny-AI/PowerInfer>**
- **(a) 严格单机**：README 摘要 *"on a personal computer (PC) equipped with a single consumer-grade GPU"*；README 的 TODO 里 **`- [ ] Support Multi-GPU` 仍未勾选**——多 GPU 都没做完，跨机更无从谈起。论文 <https://arxiv.org/abs/2312.12456>（SOSP 2024）。
- **(b) 并行方式**：单机内 **hot/cold neuron 级** offload（`adaptive predictors`、`neuron-aware sparse operators`），`--vram-budget <GB>` **取代** llama.cpp 的 `-ngl`；粒度是 FFN 级、非层放置。只支持 ReLU/ReGLU/Squared ReLU 激活的模型（模型动物园冻结在 LLaMA-2 7/13/70B、Falcon-40B 等）。
- **(d)(e)(f)**：无容错；`cache_prompt`/`tokens_cached` 前缀复用与 `-cb/--cont-batching`、`-np` 是**从 llama.cpp server 继承**的；无 vLLM/SGLang 集成。
- **(g)**：**MIT**；**零 release**（"There aren't any releases here"）；`main` 最后提交 **2026-05-11**；实质冻结（真实代码工作止于 2024-09-06，2025 只有 README，2026 是 Tiiny 品牌迁移）。
- **(h)**：单 RTX 4090 上 *"average token generation rate of 13.20 tokens/s, with a peak of 29.08 tokens/s"*，相对 llama.cpp 最高 11.69×。

### PowerInfer-2　**公开仓库不存在**
- `https://github.com/SJTU-IPADS/PowerInfer-2` 与 `https://github.com/Tiiny-AI/PowerInfer-2` **均 404**；issue #207（"Will the related code open-sourced?"）**仍 open 且无维护者回复**。论文 <https://arxiv.org/abs/2406.06282>（单台手机 OnePlus 12，NPU/CPU/UFS 的 neuron-cluster offload，需要 root 权限的 mlock）。
- **没有仓库 ⇒ 没有 license、没有 release、没有 commit 日期可引**。评测：TurboSparse-Mixtral-47B 在 OnePlus 12 上 11.68 tok/s；相对 llama.cpp 最高 27.8×。
- **判定：论文级、闭源、严格单设备，完全不可用。**

---

## 7. 2025–2026 其他项目（**[子代理核验]**）

### 7.1 消费级多机分层：最相关的两个

**Mesh-LLM**　<https://github.com/Mesh-LLM/mesh-llm>　**Apache-2.0**（"Copyright 2024 Block, Inc."）
- 跨机**连续层段**（"Skippy stage splits"），走 Iroh/QUIC P2P；不是张量并行。
- **不等分 + 运行期重算：本次调研中最强**。规划器 *"enumerates `node_count in minimum_nodes..=usable_nodes` and every node subset"*，是时延感知的（按 `estimated_decode_network_ms_per_token` 排序，默认 `DEFAULT_TARGET_DECODE_TPOT_MS = 33`）；手工 lock JSON 接受任意半开区间（`layer_start` 含、`layer_end` 不含，必须连续且覆盖 0..layer_count）。有一次日志化的自动放置把 66 层分成 **`0..65 / 65..66`**（极端不等分）。
  <https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/docs/skippy/TOPOLOGY_PLANNER.md>、<https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/docs/SKIPPY_SPLITS.md>
- **容错（具体且有限）**：*"If a locked stage is lost, the topology becomes unavailable and is withdrawn after the normal stage-loss grace period."* PR #1523 加 `ConnectionSessionOwnership`，回收失败 stage 连接的孤儿 session、execution lane 与缓冲的 `split_prefill_tokens`，替换连接在清理完成后可认领同一 session key（抓到原始 diff）。**未发现**在途请求的 checkpoint/恢复。
  <https://patch-diff.githubusercontent.com/raw/Mesh-LLM/mesh-llm/pull/1523.diff>
- **KV/前缀**：有 suffix cache 复用（文档记录精确回放恢复全部 3531 个 prompt token），但文档也自承 *"concurrent admission and suffix cache reuse as active validation gaps"*；Q4_0 K/V cache。
- **成熟度**：自称 *"experimental distributed-systems software"*；stable v0.75.1 / prerelease v0.76.0-rc8；3 个已知 Windows 加载 bug（#1510/#1511/#1512）。最新提交日期 **U**（API 403）。
- **判定：需要人工评估的首选消费级底座**（文档化 planner + 可替换放置策略），但「stage 内 continuous batching」与「在途恢复」两项待验证。

**prima.cpp（PRIMA.CPP）**　<https://ar5iv.labs.arxiv.org/html/2504.08791>（ICLR 2026 poster 10008093）
- **唯一一个测试床就是我们这类集群的已验证系统**：4 节点家庭集群（2 笔记本 + 1 台式 + 1 手机），一个 WiFi 路由器，320–610 Mbps，3–7 ms 时延，最强设备 2080Ti（11 GB）。
- 并行方式：**piped-ring parallelism（PRP）**——连续层段在环上多轮传递；每设备 CPU+GPU。
- **不等分 + 运行期重算：有**，Halda 把 Layer-to-Device Assignment 建模为 ILP，解出每设备层窗口 `w_m` 与 GPU 层数 `n_m`；枚举 L 的因数（≤11 个合法 k）把 NP-hard ILFP 化为标准 ILP；**4–32 设备调度时延 10–12 ms**；含标定与弱设备剪枝；环境漂移（后台程序、磁盘老化）时需重新标定。
- 数字：70B Q4K **674 ms/token** vs llama.cpp **10,120 ms/token**；去掉 Halda（改成 exo 式按比例分层）70B 变成 **20,848 ms/token**（即**异构感知的不等分价值 15–31×**）；8B 15 ms/token 与单机持平。
- **容错：无**。**KV/前缀缓存：无**。**批处理：明确没有**（*"mini-batching is not yet implemented"*）。
- **代码可达性：论文给的 `https://github.com/Lizonghang/prima.cpp` 抓取为 404**；论文注里另给 gitee.com/zonghang-li/prima.cpp（**未核验**）→ 代码按不可用对待。

### 7.2 数据中心参照（算法可借鉴，硬件不可用）

| 系统 | 与 R1/R2 的关系 | 关键引文/数字 | URL |
|---|---|---|---|
| **vLLM** | 跨机 PP；**官方要求各节点环境完全一致** | *"Ensure that every node provides an identical execution environment ... Using container images is recommended ... to hide host heterogeneity."*；*"Edge case: uneven GPU splits ... pipeline parallelism, which splits the model along layers and supports uneven splits."*；*"closing any shell terminates the cluster."*；容错被明确甩给 Ray（*"production-grade fault tolerance, scaling, and distributed observability"*）。**注意：当前官方文档没有「PP 是实验性/不支持」的表述**（任务书里的假设不成立于现文档） | <https://raw.githubusercontent.com/vllm-project/vllm/main/docs/serving/parallelism_scaling.md> |
| **SGLang** | 跨节点 PP 一等公民；不等分是**启动期 env var**，非运行期 | `--pp-size/--nnodes/--node-rank/--dist-init-addr`；`SGLANG_PP_LAYER_PARTITION=15,15,15,16`，官方建议 *"Put the larger partition in the higher PP rank when the layers are not evenly divisible"*；Radix Cache 前缀缓存；`--enable-dynamic-chunking` + `SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.75` 减少 PP 气泡。**无弹性/无容错** | <https://docs.sglang.io/docs/advanced_features/pipeline_parallelism.md> |
| **LocalAI** | **基于 llama.cpp RPC** 的分布式，"Worker Mode" 权重按内存比例分片 | 官方劝退语：*"Only a single model is supported currently."*、*"Ensure the server detects new workers before starting inference. Currently, additional workers cannot be added once inference has begun."*、federated **"still experimental ... tech preview quality"** | <https://localai.io/docs/features/distribute/> |
| **Ghostlink** | 消费级/异构局域网，包装 llama.cpp `ggml-rpc`，UDP/mDNS 发现 + 层分配规划器 | 容错是**失败即清**：*"`ggml-rpc-server` is actively supervised by `RpcSupervisor`; if a worker crashes, discovery advertisement is revoked immediately and **active requests fail/drain cleanly**."* 自承 *"Usable Multi-Node Speed Unproven"*；2 节点 LAN：Qwen2.5-1.5B-Q4_K_M **53.57 tok/s**；Qwen3-Coder-30B-A3B-Q3_K_L **1.47–2.52 tok/s**（单节点 OOM） | <https://raw.githubusercontent.com/rwilliamspbg-ops/Ghostlink/main/README.md> |
| **DynaPipe**（NeurIPS 2025） | **R1 的算法形态**：*"a dynamic layer redistribution scheme that adaptively balances computation by predicting execution latency in real time"* + *"an asynchronous key-value (KV) cache migration coordinator to enable non-blocking layer redistribution during inference"*；8–49% 时延下降。跨机范围与代码发布 **U** | — | <https://proceedings.neurips.cc/paper_files/paper/2025/hash/c80873f613504606542457ad715ac3c3-Abstract-Conference.html> |
| **SpotServe**（ASPLOS'24） | **R1+R2 的经典参照**：运行期重并行化 + 二分图匹配（Kuhn-Munkres）迁移 + *"stateful inference recovery ... commits inference progress at a much finer granularity and allows SpotServe to cheaply resume inference upon preemption"*；P99 尾时延 2.4–9.1× 改善；成本省 54%。**代码未发布**：*"We are still preparing artifact evalution and will release the code soon!"* | — | <https://arxiv.org/abs/2311.15566>、<https://raw.githubusercontent.com/Hsword/SpotServe/main/README.md> |
| **LUMEN**（2026） | 恢复机制 SOTA（基于 SGLang v0.5.6，~13K LoC）：负载感知的 **KV checkpoint 到别的 worker 的 CPU 内存**、locality-aware 恢复调度、**用 draft model 做渐进式恢复**。4 worker Qwen3-32B：平均 TTFT −44.4%（vs Stop-and-Restart）；8 worker Qwen3-14B：恢复时间 −64.1%。**门槛**：每 worker checkpoint 预算 *"80 GB per worker for Qwen3-14B and 160 GB per worker for Qwen3-32B"*；故障模型是整 worker。代码未开源 | — | <https://ar5iv.labs.arxiv.org/html/2606.17787> |
| **PipeBoost** | 崩溃时的层重分配 + KV 重建，但**仅单机内** | *"This paper focuses on scenarios where homogeneous GPUs are deployed on a single GPU server ... For heterogeneous GPUs, a more sophisticated model partitioning strategy is required."*；恢复时间比重启实例少 51% | — | <https://ar5iv.labs.arxiv.org/html/2503.17707> |
| **TPI-LLM** | 4 笔记本**张量并行**（非 PP）的负面结论很有用 | *"The bottleneck in allreduce is not network bandwidth, but link latency"*；70B@300 Mbps 数据传输只需 3.4 ms，链路时延主导（ring allreduce 56τ vs star 8τ）；测试床即 4 台笔记本、178 Mbps LAN | — | <https://ar5iv.labs.arxiv.org/html/2410.00531> |
| **ServerlessLLM**（OSDI'24） | 成熟、Apache-2.0、多节点，但是**模型复用 + 迁移**，不是跨机层切分 PP | — | <https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/main/README.md> |
| **MLC-LLM / TVM Disco** | **部分核验**：`python/mlc_llm/cli/disco_remote_socket_session.py` 存在（`<server_host> <server_port> <num_workers>` → `runtime.disco.RemoteSocketSession`），说明有真实的远端多 worker 通路；但 **`https://llm.mlc.ai/docs/deploy/distributed.html` 404**，README 只讲单设备 → **不能宣称 MLC-LLM 有受支持的跨机 PP** | — | <https://raw.githubusercontent.com/mlc-ai/mlc-llm/main/python/mlc_llm/cli/disco_remote_socket_session.py> |
| **MLX distributed** | 有真实分布式运行时（`mlx.core.distributed.init/Group/all_sum/all_gather/send/recv/...`）；第三方 `mlx-dist-setup` 用 Open MPI + `mlx.launch --backend mpi ... pipeline_generate.py` 跨 2 台 macOS 跑流水线（需免密 SSH、一致 conda 路径、匹配网卡、开防火墙）。**仅 Apple Silicon，与 4 台 NVIDIA 笔记本无关** | — | <https://raw.githubusercontent.com/alexziskind1/mlx-dist-setup/main/README.md> |
| **rpc_manager** | `https://github.com/arseniy0924/rpc_manager`（"Web UI for orchestrating distributed llama.cpp RPC GPU clusters with auto node discovery, telemetry, and one-click deployment"）——**仅确认存在**，README 各分支 404、API 403，能力 **U** | — | — |

### 7.3 明确排除（一句话理由）

- **DistServe / Splitwise / Mooncake**：数据中心 prefill/decode 分离，面向 GPU 集群（I，未读原文）。
- **NVIDIA Dynamo**：NVIDIA 数据中心框架；有 KV-cache-aware routing 但需数据中心 GPU。
- **llm-d / KubeAI / KServe / Ray Serve LLM / vLLM production stack / SGLang router**：K8s/Ray 集群运维层，解决的不是消费级笔记本异构。
- **Ollama / LM Studio / Jan**：未发现分布式层切分支持。
- **Hive**（Elsevier SoftwareX）：仅在搜索结果中出现，架构与是否层切分 **U**，未抓取。
- **"ServerlessLLM Flex"**：专门搜索**未能确认其存在**，不要引用。
- **moe-engine**：**训练**（不是推理），超大规模 MoE，Zenodo 自发布预印本。
- 任务书点名的 **Parrot / Metis / Tetris / Helix / Ayo / PipeSwitch**：本轮**未能核验**，不作任何断言。

---

## 8. 横向对照：对 R1–R6 的逐条判定

| 需求 | 谁满足 | 具体机制与证据 |
|---|---|---|
| **R1a 不等分分层** | llama.cpp RPC（`-ts` / `-ot`，边界不精确）；exo（自动按 RAM，边界不可控）；Petals（`--block_indices` 精确+自动）；Mesh-LLM（planner/lock JSON）；prima.cpp（ILP）；SGLang/vLLM（配置期） | 见 §2.2 / §3(b) / §5(b) / §7.1 |
| **R1b 运行期重算** | **只有** Petals（~60 s rebalancing）、Mesh-LLM（规划器）、prima.cpp（Halda 10–12 ms）、DynaPipe（论文） | llama.cpp RPC/exo/distributed-llama/SGLang/vLLM **均不支持运行期重算**（exo 的「重算」是破坏性重建） |
| **R2 在途请求恢复** | **无人满足**。设计上最接近的是 Petals（dual cache + 只重跑失败 stage，需每区间 ≥2 副本；issue #587 未修）；机制上最完整的是 SpotServe/LUMEN/PipeBoost（数据中心/单机） | llama.cpp RPC：abort（或 #26724 的永久失效）；exo：实例删除 + 任务取消；distributed-llama：整簇 re-init 且不缩容；Mesh-LLM：撤销拓扑 + 回收 token |
| **R3 KV 前缀复用** | llama.cpp server（`--cache-ram`、`/slots/*/save|restore`、`timings.cache_n`）、exo（`KVPrefixCache`）、SGLang（Radix Cache）、ktransformers（三层 KV）、Petals（仅 session 内） | 见各节 |
| **R3 cache-aware 路由** | Petals（`cache_tokens_left`）、SGLang（radix + 文档化的 DP router，**内容未读 → I**）；**llama.cpp router 只按模型名路由** | — |
| **R4 stage 内 continuous batching** | llama.cpp（`-cb` 默认开 + `-np`）、SGLang、vLLM、ktransformers（`balance_serve`）；Petals 有服务端批处理 | **distributed-llama 无**（单线程串行）；exo/Mesh-LLM/prima.cpp 未核验或明确没有（prima.cpp "mini-batching is not yet implemented"） |
| **R5 可复现 baseline** | llama.cpp（`llama-bench` + 明确 CLI + MIT + 129k star）；PowerInfer 继承 llama.cpp | — |
| **R5 调度策略可替换/插桩** | **没有一个能真正插拔**。观测最好的是 exo（events/traces/bench/能耗）；策略最接近可配置的是 Mesh-LLM（planner + lock JSON）与 Petals（`--block_indices` + `--balance_quality 0` 静态覆盖）；llama.cpp 只有 `-ot` 这一个「准策略」旋钮，其余要改源码 | — |
| **R6 工程风险** | llama.cpp：MIT、极活跃、社区巨大，但 RPC 自称 proof-of-concept/fragile/insecure、无鉴权、协议不版本化、`supports_op` 恒真、曾有 RCE；exo：Linux 只能 CPU + 需 TB5 RDMA；distributed-llama：自研格式 + 网络主导；Petals：休眠 + 依赖锁死；Mesh-LLM：自称 experimental | — |

---

## 9. 结论与工程建议

### 9.1 对「能不能直接拿现成系统当引擎」的回答

**不能。** 没有任何现成系统同时具备「跨机不等分层 + 运行期重算 + 在途恢复」。具体地说：

1. **R2 是公开空白。** 我们在 2025–2026 的公开项目/论文里**没有找到任何一个**在消费级多机上实现「节点离开后在途请求可恢复 + 幸存者上重建流水线」的系统。实现恢复的系统（SpotServe / LUMEN / PipeBoost / DynaPipe）都在数据中心或单机内；跑在消费级多机上的系统**要么崩溃、要么取消、要么只撤销拓扑**。这个不对称正是可发表的贡献点。
2. **最接近我们架构的两个「设计参考」是 Petals 与 prima.cpp**：
   - Petals 给了**恢复该怎么做**：客户端缓存「发给该 stage 的历史输入」+ 服务端 KV，失败时**只重跑失败的 stage**（不是从第 0 层重来），并有量化收益（p=1e-3 时 7.76 vs 整段重跑 0.48 steps/s）。它同时暴露了**关键约束：每个 block range 必须有 ≥2 个持有者**——这直接决定我们把流水线做成「4 个唯一 stage」还是「每段 2 副本」。
   - prima.cpp 给了**切分该怎么做**：ILP 求解不等分（10–12 ms 可重算），并且用数字证明异构感知的不等分相对「按比例分层」有 15–31× 差距——这正是我们 A:0-11/B:12-15/C:16-19/D:20-27 设计需要引用的动机。
   - SpotServe（stateful inference recovery / 细粒度 commit 进度）与 DynaPipe（异步 KV 迁移驱动的运行期层重分配）是 R1+R2 的论文级参照。
3. **launcher/parallelism 层面，distributed-llama 与 ktransformers/PowerInfer 应从「引擎候选」中移除**：前者是等分张量并行（无法表达 A/B/C/D，且 4 节点 1 GbE 实测 decode 从 11.00 掉到 1.85 t/s），后者是严格单机（PowerInfer-2 连代码都没有）。

### 9.2 三条可选路线（按 R6 从低到高排序）

**路线 A（推荐，R6 最低）：以 llama.cpp RPC 为执行底座，自研控制面。**
- 执行：`ggml-rpc-server` 各机常驻，主机 `llama-server --rpc A,B,C,D -ngl 99 --split-mode layer -ts <按 8/6/4/4 显存比例>`；用 `-ot`（维护者建议、非官方文档）尝试把层精确钉到指定机器，**这条必须实测确认能否按 buffer type 名 `RPC<device>[<endpoint>]` 匹配**（名字格式 **V**：源码 `std::string buft_name = "RPC" + std::to_string(device) + "[" + endpoint + "]"`）。
- 控制面自研（这是我们的贡献）：健康探测、故障判定、**重新拉起的 `llama-server` + 新的 `--rpc` 列表**（缩容到幸存者）、以及**在途请求的恢复**——llama.cpp 侧唯一可用的原语是 `/slots/{id}?action=save|restore` 与 `--cache-ram` 前缀缓存，但远端层的 KV 随机器丢失，所以恢复必须走「客户端侧保存已发送输入 + 重算受影响前缀」的 Petals 式路线。**注意 llama.cpp 的 router 模式不支持按缓存亲和路由，需要我们自己写在前面。**
- 收益：MIT、极度活跃、`llama-bench` 直接给可复现 baseline、`--metrics` 可插桩、continuous batching 直接可用（R4 ✅）、前缀缓存可用（R3 部分 ✅）。代价：R2 全部自研；RPC 无鉴权且有 RCE 史 → **必须在受控 LAN/VPN 内运行**；`--split-mode layer` 的层边界不精确 → 可能需要 `-ot` 或小改源码（R6 上升但可控）。
- **先做的前置实测（决定架构）**：在 WiFi 上用 4 台机器跑 `llama-bench --rpc ...`，量出「每 token 的跨机同步开销」。第三方在 10 GbE 上已经看到 7B decode 从 91.8 → 52.7 t/s（≈2× 下降），并称 1 GbE 会再差 ~10×；WiFi(2–10 ms RTT) 很可能比这更差。**如果 WiFi 下 decode 被网络主导，那么「单 token 依次跨 3 跳」这条指标本身需要重新设计（例如改为按 stage 的 microbatch/多 token 流水）。**

**路线 B（评估项）：把 Mesh-LLM 当作候选底座先做 spike。**
- 它是目前唯一被核验为「消费级 + 跨机连续层段 + 不等分 + 规划器可重算 + Apache-2.0 + 活跃」的系统，且有文档化的 stage-loss 宽限期与 session/token 回收。需要人工验证两件事：**stage 内是否有 continuous batching**、以及**在途请求到底能不能在 stage 消失后继续**（目前证据是「不能」）。若这两项都不行，它的价值主要落在「放置/规划策略可借鉴 + 观测设施」。

**路线 C（不推荐单独使用，但可作对照 baseline）：Petals 私有 swarm。**
- 能精确表达我们的不等分（`--block_indices`）并在局域网跑（无公网 IP 也行、DHT 必需、health monitor 可不装），MIT，且论文给了我们拓扑的量化数字（BLOOM-7.1B、4 stage (8,7,8,7)）。但**在途恢复结构上要求 ≥2 副本**（我们要么复制每段，要么接受无恢复），且 issue #587 未修、项目 2024-08 起休眠、依赖锁死、22 GB 显存上限只到 ~13B 模型。适合作为「Petals vs 我们」的对照实验，而不是引擎。

### 9.3 报告中可直接引用的「so what」句子

- **llama.cpp RPC 的官方定位就是「能跑但脆弱」**：proof-of-concept / fragile / insecure；远端消失时 `GGML_ABORT`，未合入的 PR #26724 也只是把「崩溃」变成「设备永久失效 + 503 必须重启」，其注释原话是 *"the buffers on the server are gone, so a lost connection can never be recovered"*。
- **exo 的容错是显式删除**：节点消失 ≤10 s → 整个实例 `InstanceDeleted` → 客户端流关闭、任务 `Cancelled`。
- **Petals 的恢复设计可行但受副本约束**：论文 *"we assume that every block is hosted on several servers"*，因此「每台机器一个唯一层段」的部署**结构上无法恢复**。
- **ktransformers/PowerInfer 解决的是相反的问题**（单机内把权重摊到 CPU/磁盘/NPU），与跨机层切分正交。
- **我们的贡献空位有实证支撑**：prima.cpp 证明异构感知不等分相对按比例分层差 15–31×（70B：674 ms/token vs 20,848 ms/token）；LLM 服务系统里唯一同时做「运行期重并行化 + 有状态恢复」的 SpotServe 代码未发布；2026 年的 LUMEN 恢复机制要求每 worker 80–160 GB 主机内存。

---

## 10. 未核验清单（不要当事实引用）

1. exo 星标数（~47.6k 来自第三方 star-history）；exo benchmark 图片里的具体 tok/s；issue #284/#1816 正文；`docs/architecture.md`（404）。
2. distributed-llama：discussion #147（4×Mac Mini M4 Pro 跑 Llama-3.3-70B 的 tok/s，页面 JS 渲染 + API 403）；`report/report.pdf`。
3. Petals：`health.petals.dev` 2026-09-22 不可达 → 公开 swarm 当前存活情况；GitHub 机器可读 `archived` 标志（API 403，仅从 UI 推断）；单次失败恢复时延(ms) —— 论文里不存在。
4. ktransformers：roadmap issue #1921 正文；星标数。
5. PowerInfer-2：同行评审状态（arXiv 无 journal-ref / comments 字段）。
6. Mesh-LLM：最新 release/commit 的精确日期（API 403）；stage 内 continuous batching。
7. prima.cpp：代码可达性（GitHub 404，gitee 未核验）；DynaPipe 的跨机范围与代码发布。
8. MLC-LLM 官方多节点/PP 文档（404）；`mlx-lm --pipeline` 官方 flag；LocalAI 的 license；Hive 的架构。
9. Parrot / Metis / Tetris / Helix / Ayo / PipeSwitch：本轮未核验，不作断言。
10. 「ServerlessLLM Flex」：**未能确认存在，请勿引用**。

### 证据层面的工具限制（影响可复现性）

- `api.github.com` 全程 **HTTP 403（IP 限流）**，因此所有仓库元数据改用 GitHub 的 `releases.atom` / `commits/<branch>.atom` 与页面 UI。
- 本机 shell **无外网**（`curl: (35) schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS`），无法把源码下载到本地 grep；所有源码结论均来自 `raw.githubusercontent.com` 的抓取文本（必要时把抓取结果落到平台临时文件再用 grep/read 检索）。
- 所有条目**未经编译/运行验证**：本报告是「文档与源码级」调研，任何以「实测数字」形式出现的内容都明确标注了来源（官方论文 / 第三方基准仓库 / 项目自述）。
