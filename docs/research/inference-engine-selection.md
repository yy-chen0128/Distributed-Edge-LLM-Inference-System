# 推理引擎选型：自研、数据中心引擎，还是边缘引擎

> **问题**：我们的执行面（stage 运行时）该自己写，还是用现成引擎？候选不只是 vLLM——
> 还有 Ollama、LM Studio、llama.cpp 的 RPC、exo 这类"边缘/家用集群"系统。
>
> **证据分级**：✅ 源码或官方文档原文核对过；🔶 由已核对事实推断；❓查不到（不等于不存在）。
> 本仓库已克隆 `ollama/`（源码，MIT）、`lmstudio-python/`、`lms/`（SDK，MIT）供核对；第三方源码不入库。

---

## 0. 结论先行

| 判断 | 结论 |
|---|---|
| 有没有现成引擎直接满足"跨机异构层切分 + 节点随时离开" | **没有**。最接近的 exo 只支持 Apple/MLX（Linux 上仅 CPU），llama.cpp RPC 官方自称 proof-of-concept 且无容错 |
| 那 vLLM 呢 | vLLM 的 PP 是**同构静态**的（每段层数相同、节点集合固定），且**没有**"节点离开后在途请求恢复"这回事——那正是我们要证明的东西 |
| Ollama / LM Studio 能当底座吗 | **不能**。Ollama 完全没有跨机机制；LM Studio 的 LM Link 是"整个模型放一台机器、其余当客户端"，**不是切分**；且 LM Studio 闭源、EULA 禁止修改与再分发 |
| 建议 | **执行面继续自研**（只有它能同时满足 R1+R2），但把成熟能力按需"借或抄"：stage 内批处理照抄 llama.cpp/LM Studio 的做法、量化走 GGUF 生态；**基线分三层**做（单机 Ollama / 跨机 llama.cpp RPC / 数据中心 vLLM-PP） |

**一句话**：**没有现成引擎能替我们做"跨机异构 + 弹性"这件事，但有现成引擎能替我们做"单机引擎"这件事**——
所以合适的结构是"我们做调度与弹性，stage 内部尽量不自研"。

---

## 1. 判定用的尺子（需求）

| 编号 | 需求 | 为什么 |
|---|---|---|
| **R1** | **跨机、异构、运行时可重算的层切分** | 四台机器显存 8/6/4/4，层数必须不等；节点变化后要重切。**这是项目立足点** |
| **R2** | **弹性/容错**：机器随时被合盖抱走；在途请求可恢复；流水线在幸存机器上重组 | 边缘场景的核心卖点 |
| **R3** | 前缀 KV 复用 + **按缓存局部性路由** | 缓存感知路由是我们的策略贡献之一 |
| **R4** | stage 内连续批处理，批大小可配 | 实测 batch=1 时固定开销主导（`single-machine-scope-and-batching.md` §3） |
| **R5** | 可对照的知名基线 + 调度策略可替换/可插桩 | 论文需要可比数字，且我们的贡献要能插进去 |
| **R6** | 低工程风险 | 人力有限；自研引擎的 bug 与性能不确定是真实风险 |

---

## 2. 候选清单

| 系统 | 类型 | 跨机层切分 | 异构/可重算 | 弹性(在途恢复) | 可插桩/换调度 | 许可 | 成熟度 |
|---|---|---|---|---|---|---|---|
| **我们自研的 HF 分层引擎** | 研究原型 | ✅ 原生 | ✅ | ⚠️ 有框架无 KV 搬运 | ✅ 全部自己写 | 自有 | 原型，四机未跑 |
| vLLM | 数据中心 | ✅（PP，需 Ray） | ❌ 同构静态 | ❌ | ⚠️ 免 fork 面有限（无运行时改 PP） | Apache-2.0 | 生产级 |
| SGLang | 数据中心 | ✅（PP/DP/EP） | ❌ 同构 | ❌ | ⚠️ 同上 | Apache-2.0 | 生产级 |
| **llama.cpp + RPC backend** | 边缘 | ✅ **设备级 offload** | ⚠️ 按显存比例，**装载时定死** | ❌ 无 | ⚠️ 源码开放但无插件层 | MIT | **官方标注 PoC / fragile / insecure** |
| **exo** | 边缘集群 | ✅ **Pipeline 分片 + 拓扑感知自动切分** | ✅ 按设备资源与链路自动 | ⚠️ 文档未承诺在途恢复 | ⚠️ 有 placement 预览 API，调度可影响 | Apache-2.0 | 活跃，但 **Linux 上仅 CPU** |
| Ollama | 单机服务 | ❌ 完全没有 | ❌ 层数自动、用户不可指定 | ❌ | ❌ **无插件 API** | MIT | 生产级（单机） |
| LM Studio | 单机服务 | ❌（LM Link 是整模型远程） | ❌ | ❌ | ❌ **闭源，EULA 禁改禁再分发** | 专有（SDK/MIT） | 生产级（单机） |
| Petals | 研究型 P2P | ✅ 流水线分片 | ⚠️ 面向公网志愿者异构 | ✅ **论文卖点就是在节点加入/离开/失败下保持可靠** | ⚠️ | Apache-2.0 | 研究，维护状态待核 ❓ |
| distributed-llama / ktransformers / PowerInfer | 边缘/单机 | 见 §3.7 | — | — | — | — | — |

---

## 3. 逐条核对（证据）

### 3.1 Ollama —— 源码核对：**没有跨机机制**

`✅` 官方文档全量索引（`docs.ollama.com/llms.txt`）里**没有** distributed / cluster / multi-machine 页面；
FAQ 里唯一相关条目是**单机内多 GPU**（"模型装不进单卡时 spread across all the available GPUs"）。

`✅` 我们克隆了源码并逐个核对环境变量（`ollama/envconfig/config.go`）：

| 变量 | 实际语义 |
|---|---|
| `OLLAMA_NUM_PARALLEL`（默认 **1**） | 并行请求数上限 |
| `OLLAMA_MAX_QUEUE`（默认 512） | 排队上限 |
| `OLLAMA_SCHED_SPREAD` | "Always schedule model across all GPUs"（**单机内**多卡分摊） |
| `OLLAMA_REMOTES` / `OLLAMA_CREATE_REMOTE` | **不是远程算力**：前者是"允许从哪些主机**取模型**（默认 ollama.com）"，后者是"强制走服务端 API 创建模型" |
| `CUDA_VISIBLE_DEVICES` / `ROCR_*` / `GGML_VK_*` / `OLLAMA_GPU_OVERHEAD` | 设备可见性与显存余量 |

`✅` 在全部 Go 源码里 `grep -i rpc`：**只有一处注释命中**（webview.go 的说明文字），没有任何 RPC 分布式计算路径。

`✅` 层切分怎么做的：`ollama/llm/llama_server.go` 把层数作为 `-ngl` **透传给子进程 `llama-server`**；
`NumGPU == -1`（默认）时**根本不传** `-ngl`，让 llama-server 自己探测。
`api/types.go` 里还留着 `num_gpu` 字段，但 Modelfile 的有效参数表里已经不再文档化它；社区工单
（ollama#12010）标题就是要求恢复"手动覆盖层切分"的能力 → **用户不可指定每台设备的层数**。

`✅` 有用的部分：`/api/generate` 响应含 `prompt_eval_cached_count`（从缓存读了多少 prompt token）、
`prompt_eval_count` / `eval_count` / 各段 duration → **指标口径可以借鉴**；
`OLLAMA_NUM_PARALLEL` 是批处理开关，但**上下文按并发数线性放大**（2K 上下文 × 4 并发 = 8K）——
这正好是我们做"KV 预算"对比时的一个反例。

**判定**：R1 ❌ / R2 ❌ / R3 半（复用有、跨机路由无）/ R4 ✅ / R5 ⚠️（MIT 可 fork，但无扩展点）/ R6 单机低、四机要 fork。
**用途**：单机 MIT 基线 + 指标口径对照。

### 3.2 LM Studio —— **闭源，且跨设备是"整模型远程"**

`✅` 底层：GGUF 走 llama.cpp（runtime pack 分版本），Apple Silicon 走 MLX；**服务端闭源**，
开源的是 `lmstudio-python` / `lms` / `mlx-engine`（MIT，我们已克隆前两个，确认**它们只是客户端 SDK**：
README 明说通过 websocket 连本地 LM Studio 实例，代码里没有任何调度器/引擎实现）。

`✅` 跨设备特性 **LM Link**（0.4.6，2026-02-27）：机制是**账号配对 + Tailscale 网格**，
文档原话是"在远端设备上**加载整个模型**，然后像本地一样使用"；同模型出现在多台机器上是**多个独立条目**；
设备选择是**每台机器静态的 preferred device**（整模型放置，不是按层、也不是按请求）。
崩溃只表现为 "disconnected" 状态，无请求迁移。

`✅` 值得抄的地方：**连续批处理可配**（Max Concurrent Predictions 默认 4；"Unified KV Cache" 默认开，
关闭则会按 slot 硬切上下文）；SSE 事件里有 `prompt_processing.progress`（0–1）与逐 token delta；
MLX 引擎的 KV 以 **256 token 为边界做检查点落盘 + LRU + 恢复最长可用前缀**。

`✅` 许可：**专有**，EULA 禁止修改/派生作品、禁止以非公开接口集成、禁止逆向、禁止再分发与 SaaS；
责任上限 50 美元。SDK 是 MIT。

**判定**：R1 ❌ / R2 ❌ / R5 ❌（**法律上不允许我们改**）。
**用途**：只能当**黑盒单机基线**，绝不可作为底座或修改对象。

### 3.3 llama.cpp + RPC backend —— **最省事的跨机基线（但官方自称 PoC）**

`✅` 官方 README（`llama.cpp/tools/rpc/README.md`）原文要点：

- `ggml-rpc-server` 把远端主机上的 ggml 设备暴露出来；主节点用
  `llama-cli ... -ngl 99 --rpc 192.168.88.10:50052,192.168.88.11:50052` 接入（构建时加 `-DGGML_RPC=ON`）。
- **"By default, llama.cpp distributes model weights and the KV cache across all available devices --
  both local and remote -- in proportion to each device's available memory."**
  可以用 `--tensor-split` 自定义比例。
- 支持本地缓存（`-c`）避免反复传张量；支持 RDMA（Linux RoCEv2 / macOS Thunderbolt 5）。
- **顶部警告原文**：*"This example and the RPC backend are currently in a proof-of-concept development
  stage. As such, the functionality is fragile and insecure. **Never run the RPC server on an open network
  or in a sensitive environment!**"*

🔶 与我们形态的差别：它是**单进程 + 设备级 offload**（一个 token 的前向路径跨多个远端设备同步完成），
KV 也按同一比例切开；**没有**"每台机器独立进程、独立重载、独立 epoch"，**没有**容错（rpc-server 掉了就失败），
切分在**装载时定死**，且模型必须是 **GGUF**。

**判定**：R1 ⚠️（能做跨机层切分，但异构比例与运行时重算受限）/ R2 ❌ / R5 ⚠️（源码 MIT、无插件层）。
**用途**：**跨机基线的最优选择**——代价最低，而且顺带把"量化模型"这条路免费打开（GGUF 有成熟的 Q4_K_M 等）。
它官方自述的 fragile/insecure 正好是论文里"现有跨机方案缺调度与容错"的最好引证。

### 3.4 exo —— **架构上最接近，但平台不对**

`✅` 官方 README（已抓取）原文要点：

- "exo connects all your devices into an AI cluster"，目标是"跑单台装不下的模型"。
- **"Topology-Aware Auto Parallel：exo figures out the best way to split your model across all available
  devices based on a realtime view of your device topology. It takes into account device resources and
  network latency/bandwidth between each link."**
- API 里有 **`sharding: "Pipeline"`** 与 `sharding: "tensor"` 两种（`/instance/previews` 可预览所有合法放置方案，
  再 POST `/instance` 选定）→ **放置方案对外可见、可影响**，这一点值得我们借鉴。
- 后端是 **MLX / MLX distributed**（`instance_meta: MlxRing`）。
- **`✅` 硬伤**："**On macOS, exo uses the GPU. On Linux, exo currently runs on CPU.** We are working on
  extending hardware accelerator support."
  → 我们的四台是 **Windows + WSL + NVIDIA**，用不了它的 GPU 路径。
- 许可 Apache-2.0；有 `exo-bench`（可按 `--sharding pipeline|tensor` 分别测 pp/tg 吞吐）。
- ❓ 文档**没有**承诺"节点离开后在途请求恢复"。

**判定**：R1 ✅ / R2 ⚠️（未承诺）/ R3 ⚠️ / R5 ⚠️，但**平台不匹配**。
**用途**：**参考设计**（拓扑感知自动切分 + 放置方案预览 API + 异构设备资源建模），不是引擎。

### 3.5 vLLM / SGLang —— 数据中心路线，与 R1/R2 冲突

`✅`（我们此前在 `l2-l3-customization-and-reuse.md` 里逐条核对过）：

- vLLM 的流水线并行**假定每段层数相同、节点集合固定**；免 fork 的扩展面里**没有**"运行时改 PP 切分"的接口。
- 弹性专家并行（EP）与 PP 在配置层**互斥**（`config/parallel.py:846-850`），
  所以"分层跨机 + 节点离开缩容"**无法原生表达**。
- 换来的是：PagedAttention、连续批处理、前缀缓存、KV connector 层粒度接口
  （`save_kv_layer(layer_name)` / `wait_for_layer_load(layer_name)`）——**这些正是我们缺的**（见 §5）。

**判定**：R1 ❌（同构静态）/ R2 ❌ / R4 ✅✅ / R5 ✅（作为数据中心基线）。
**用途**：数据中心基线；以及"如果放弃弹性，能白拿到什么"的参照。

### 3.6 Petals —— 研究型，容错是它的卖点

`✅` 论文摘要（ACL demo / arXiv 2209.01188）明确写："Another challenge is to provide **reliable** inference
and training **despite nodes joining, leaving or failing at any time**." 用 DHT 做分片发现、流水线分片，
面向公网志愿者异构节点。
❓ 仓库当前维护状态、是否支持我们的异构层切分比例、在 4 机局域网下的实测延迟——待核实。

**判定**：R1 ✅ / R2 ✅（设计目标）/ R6 ⚠️（研究代码，公网假设与局域网不同）。
**用途**：**最值得读的相关工作**（它的容错设计正是我们要做的事），但不太可能直接当引擎。

### 3.7 distributed-llama / ktransformers / PowerInfer

❓ 本轮的官方文档核对尚未完成（调研进行中）。已知定位：
distributed-llama 面向消费级设备（含树莓派集群）的跨机推理；ktransformers / PowerInfer 是**单机** CPU/GPU
混合 + 专家/神经元卸载。**待补**：它们的切分方式（层/张量）、能否不均等、容错、许可。

### 3.8 我们自己的 HF 分层引擎 —— 唯一同时满足 R1+R2 的

`✅` 已实测（见 `single-machine-scope-and-batching.md`、`parameter-and-kv-management-status.md`）：
meta 骨架只加载本段权重、per-request KV、epoch 隔离、activation 带校验和与 epoch、
节点离开触发重切 + 旧 epoch drain、真实 7B 单段已在 GPU 上跑通（4 层驻留 1864.5 MB）。
缺的是：KV 导出/导入、显存可行性约束、批处理、准入控制。

---

## 4. 一个关键区分：为什么"跨机"这件事各家做得都不是我们要的

| 维度 | 数据中心（vLLM/SGLang） | 边缘 offload（llama.cpp RPC） | 家用集群（exo） | **我们** |
|---|---|---|---|---|
| 切分单位 | 层（同构） | 设备（按显存比例） | 层/张量（拓扑感知） | **层（异构、按实测能力）** |
| 进程模型 | 单控制器 + 多 worker | **单进程** + 远端设备 | 多进程 + 自动发现 | **多独立进程 + epoch** |
| 节点离开 | 任务失败/重启 | **推理失败** | 未承诺 | **重切 + 旧 epoch drain + 请求恢复** |
| 切分可变 | ❌ 装载时定死 | ❌ 装载时定死 | ⚠️ 可重放放置 | ✅ **运行时换 epoch** |
| 交换的东西 | 张量/RPC | ggml 算子调用 + 张量 | activation | **activation（带 epoch + 校验和）** |

**要点**：**"跨机"不是难点，"跨机 + 运行时可变 + 节点会走"才是**。前两者别人做了，
第三个（弹性）目前只有 Petals 在公网志愿者场景认真做过，而它的假设与我们的局域网异构不同。

---

## 5. 建议：三层结构

### 层1 —— 执行面：**继续自研**（但要补三件事）

理由：R1（异构、可重算）+ R2（弹性）**只有自研能同时满足**；从零写执行面我们已经走通到"真实 7B 单段在 GPU 上跑"，
沉没成本已经付了。要补：

1. **KV 导出/导入**（跨机恢复与迁移的前置，见状态文档 §4.4）；
2. **显存可行性约束**（重切层必须校验"层数×每层字节 + 词表 ≤ 可用显存"，否则 7B 会切出装不下的段）；
3. **stage 内批处理**（摊薄每层每次调用 2.55 ms 的固定开销）。

### 层2 —— 能借就借，不要重造

| 借什么 | 从哪借 | 怎么借 |
|---|---|---|
| 量化模型 | **GGUF 生态 / llama.cpp** | 直接下 Q4_K_M 等量化权重，**不要自己实现 AWQ/GPTQ**（我们的引擎在 `strict=False` 下会静默出错，见数据集文档 §7.3） |
| stage 内批处理 | llama.cpp server 的 slot / LM Studio 的 **Unified KV Cache** 设计 | 照抄"默认不按 slot 硬切上下文"的做法，避免上下文被并发数除 |
| 前缀缓存实现 | LM Studio mlx-engine：**按固定 token 边界做检查点 + LRU + 恢复最长可用前缀** | 作为设计参照（我们的 KV 还没做检查点） |
| 指标口径 | Ollama（`prompt_eval_cached_count` 等） | 让我们的指标能与单机基线对齐 |

### 层3 —— 基线要分三层做，不能只挑一个

| 基线 | 用途 | 代价 |
|---|---|---|
| **Ollama（单机，MIT）** | 单机吞吐/延迟/缓存命中口径的对照 | 低 |
| **llama.cpp RPC（跨机，PoC）** | **证明"跨机层切分在 WiFi 上到底能不能用"**，并提供量化模型的对照 | 低；官方自称 fragile，正好作为"缺调度/容错"的引证 |
| **vLLM / SGLang（数据中心）** | 说明"同构静态方案在动态边缘场景的差距" | 中（需 Linux 服务端） |
| LM Studio（黑盒单机） | 可选：连续批处理 + 统一 KV 的对照 | 低（但**不可修改**） |

**这样做的意义**：论文里我们的贡献被**限定在"调度 + 弹性"这一层**，
不声称"比 vLLM 快"，而是"在 vLLM/llama.cpp 无法表达的动态异构场景下能跑、能恢复"。
这既避开了自研引擎性能不如生产引擎的短板，也回应了"脱离真实生产场景"的担心——
因为基线里包含了两个真实生产引擎。

---

## 6. 风险与未决问题

| 风险 | 说明 | 缓解 |
|---|---|---|
| 自研引擎性能不如生产引擎 | 我们的 decode 每层 2.6 ms（7B），vLLM 在同类卡上会更优 | 贡献限定在调度/弹性；用三层基线把差距量化出来 |
| 自研引擎 bug | 已修 9 处（策略文档 §7）+ KV 计量（`status` 曾恒报 0） | 保持"每个能力都有实测数字"的纪律；等价性测试 |
| "脱离生产"的质疑 | 我们用的是 HF 分层运行时而非生产引擎 | 基线含 Ollama / llama.cpp / vLLM 三个真实系统 |
| **是否把 stage 内部换成 llama.cpp** | 换掉能白拿批处理+量化，但会丢掉"层可寻址、可插桩、epoch 可切" | **未决**：建议先做层1的批处理原型，若收益足够就不换 |

---

## 7. 附录：核对记录

| 对象 | 版本 / 日期 | 许可 | 我们做了什么 |
|---|---|---|---|
| Ollama | 源码 commit `6383a0f`（2026-09-18），当日发布 v0.34.2/v0.34.3-rc1 | MIT | **克隆源码**，逐项核对 envconfig / sched.go / llama_server.go / api/types.go |
| lmstudio-python | `1679683`（2025-10-27） | MIT | 克隆；确认是客户端 SDK（websocket 连本地实例） |
| lms（CLI） | `1017bcb`（2026-09-21，@Release-56） | MIT | 克隆 |
| LM Studio 服务端 | 0.4.25（2026-09-19） | **专有**（EULA 禁改/禁再分发） | 只读官方文档与 EULA |
| llama.cpp RPC | 官方 master README | MIT | 抓取 README 原文 |
| exo | 官方 master README | Apache-2.0 | 抓取 README 原文 |
| vLLM / SGLang | — | Apache-2.0 | 复用本仓库既有核对结论（`l2-l3-customization-and-reuse.md`） |
| Petals | arXiv 2209.01188 | Apache-2.0 | 只核对了论文摘要中的容错声明 |
| distributed-llama / ktransformers / PowerInfer | — | — | ❓ 待补 |
