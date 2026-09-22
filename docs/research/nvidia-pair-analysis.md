# NVIDIA Personal AI Router (PAIR) 技术分析

> 分析对象：`PAIR/Personal-AI-Router` 仓库，当前检出 tag `v0.1.1`
> （HEAD `13b6811`，仓库时间戳 2026 年）。仓库 URL:
> https://github.com/NVIDIA/Personal-AI-Router
>
> 本文面向"边缘/多算力节点上执行 LLM 推理"的研究型读者，试图回答：
> **PAIR 在我们熟悉的"多算力单元推理体系"（PP/TP/DP/EP、模型跨机切分、
> KV 迁移、节点动态增删）坐标中处于什么位置？它调度了什么？哪些决策它做、
> 哪些决策它根本不做？如果真来了一个推理请求，系统内部到底怎么走？**
>
> 所有论断均可在源码/官方文档中找到对应证据，文末附「源码地图」。
> 官方文档以仓库内 `docs/architecture.mdx`、`docs/overview.mdx`、
> 各组件 `README.md` 与 Go 源码为准（README 与源码冲突时以源码为准）。

---

## 0. 关键结论速览（先给答案）

| # | 问题 | 一句话答案 |
|---|---|---|
| 1 | 位于多算力执行体系的什么位置？ | 它是**"整机级、请求级"的推理路由/负载均衡控制面**：位于"执行引擎"之上、应用之下；它**不**做任何跨机模型切分（无 PP/TP/EP/序列切分），也不做 KV/状态共享。加机器 = 增加可并发的请求数，**不**加速单请求。 |
| 2 | 它调度了什么？ | 调度**每一个完整推理请求（job/workload）去往哪一台节点**，且只在"持有该模型且引擎在跑"的节点集合内按负载序选路。**不是**调度算力切分、不是调度显存、不是调度模型放置（放置由人逐节点准备）。 |
| 3 | PP/TP/DP 是管理层判断的吗？ | **都不是 PAIR 管理层判断的**。PAIR 里不存在 PP/TP/EP；单机内多 GPU 的 TP/PP 由引擎（Ollama/LM Studio，对 PAIR 不透明）决定；跨机的"DP"只是结果：多台机器各持有完整模型副本，PAIR 在请求粒度上把请求分给其中一台（= 副本间负载均衡，且不做跨机 batch）。 |
| 4 | 新节点能力（算力/内存）如何被确定？ | 只做**遥测盘点，不做能力基准**：入网后由发现组件抓取 `/v1/node-info`（GPU 型号/显存总量与占用/利用率、CPU、内存），用于**展示**与"该节点能否服务某模型"的资格判定（资格 = 可达 + 引擎在跑 + 模型在册）；调度排序**不用**显存/算力/GPU 型号，只用"待处理请求数 + 粗粒度 GPU 利用率"。 |
| 5 | 节点间带宽是管理层的优化考量吗？ | **不是。** 系统不测量带宽/延迟，路由输入中没有任何网络质量信号；官方明说调度器 "not VRAM-capacity- or latency-aware"。它默认同 LAN、弱假设链路同质。 |
| 6 | 支持节点中途离开吗？ | **成员层面支持**（cluster:leave / nodes:remove / 掉线驱逐都有完整机制、且防抖动）；**对"已经开始流式返回"的请求不迁移、不续传**——连接截断、job 记 failed、客户端自重重试。任何离开都不会"杀掉"系统，只会造成短暂的最终一致空窗。 |
| 7 | 来了一个真实推理任务怎么跑？ | 客户端 → 本机代理（loopback）→ 模型资格门（谁在册）→ 按调度优先级+预留排序 → mTLS 转发到选中的远端代理 → 直通其本地引擎 → 流式回传；事件沿 workload 链路复制到全网并反馈给调度器。详见第 9 节全流程。 |

> 对你们项目的最大启示（详见 §11）：PAIR 是一个**干净、可借鉴的"请求路由/信任/成员管理"生产实现**，但它**刻意不含**你们研究的核心——
> 把放不下的模型切到多节点、KV 缓存跨节点管理、带宽受限下的放置与迁移。对照着看最能厘清"路由控制面"与"模型并行执行面"的分界。

---

## 1. 项目是什么 / 不是什么

### 1.1 是什么（README 原文精神 + 源码确认）

NVIDIA Personal AI Router（PAIR）把一个局域网内若干台装有 Ollama / LM Studio 的
机器组织成一个**对应用透明的"单入口推理集群"**：

- 每台机器跑同一套软件（对等节点，**无中心/无主节点**）；
- 应用照旧访问 `http://127.0.0.1:11434`（Ollama 兼容）或 `http://127.0.0.1:1234`
  （OpenAI 兼容），PAIR 的**代理进程占住这两个端口**，把真正的引擎挤到
  `11435+` / `1235+`；
- PAIR 决定"这个请求由集群中哪台机器执行"，应用不知道、也不需要知道；
- 路由后的请求与响应在节点之间走**双向 mTLS**（PIN 配对引导信任）；
- 每节点上的 GUI/终端可看到全集群节点、GPU、模型、正在跑的 job。

它定位为 **multi-agent / 并发本地工作负载**的路由器：多个 AI 应用/智能体同时
打进来，PAIR 把它们摊到多台机器的 GPU 上。整个提示与响应若所有组件都在本网，
则不出局域网。

### 1.2 不是什么（README 明示的四条边界 + 源码佐证）

1. **不聚合 GPU 显存**——若干张卡不会变成一张大卡；
2. **不把单个模型跨机器切分**——一个请求永远在一台节点上整跑；
3. **不把已经在跑的请求搬到另一台节点**（无在途迁移/抢占）；
4. **不自己存模型、不提供模型服务**——引擎自己下载并持有权重，模型在每台节点
   是独立副本。

代码层面的直接体现：代理的 failover 循环注释明确写
*"We can only retry before the first byte reaches the client; once a response
starts streaming we're committed"*（`services/ollama-proxy/proxy.go`），而
README 的功能表第一行即 *"routes each independent request to one node"*。

另外：**它只服务于推理**（Ollama/LM Studio 都是推理引擎），**没有训练/微调**支持。

---

## 2. 系统形态（为后面的问答提供词汇）

### 2.1 四层抽象（`docs/architecture.mdx` "Cluster Shape"）

```
Cluster  —— 一组已配对的节点（无 founder、无 primary、成员关系对称）
  └─ Node —— 一台跑 PAIR 的机器（peer，既能服务也能路由；UUID 是唯一身份）
       └─ Engine —— 本机推理服务器：Ollama 或 LM Studio（可同时跑两个/一个都不跑）
            └─ Model —— 该引擎上的模型（节点间不共享；同模型多副本=可互换性来源）
```

关键句："**集群成员资格让节点可达，但并不让它有能力**。" 资格在底层判定：
请求点名一个模型 → 只有"引擎在跑 **且** 当前库存里有该模型"的节点能服务。

### 2.2 进程与两个数据面（`docs/architecture.mdx`、`services/readme.md`）

13 个 Go 二进制，1 个 Electron/React 桌面应用。每节点：

- **`nvpair-ui-broker`**：进程总管 + JSON-RPC API（stdio/命名管道）。启动并监督
  其余 worker，有界重启（1s→16s 退避、5 次预算/60s 窗口）。**broker 是本机
  的监督者，不是集群的中枢**——节点间控制不走这条 JSON-RPC 路径。
- **`ollama-proxy` / `lmstudio-proxy`**：HTTP 反向代理，Ollama 兼容 / OpenAI 兼容，
  **请求决策的实际持有者**。
- **`nvpair-node-scanner`**：唯一的 mDNS 广播/浏览点，维护"目录"，并向目录条目
  注入硬件与模型信息；还负责 2 秒一次的调度遥测采样。
- **`nvpair-node-info`**：本机硬件清单 HTTP 服务 `:14318 /v1/node-info`
  （DXGI / nvidia-smi / IORegistry 采集）。
- **`nvpair-cluster-manager`**：身份（UUID+密钥+自签证书）、PIN 配对（EAP-NOOB）、
  可信节点库（pinned 证书），`cluster:leave` / `nodes:remove` 也归它。
- **`nvpair-engine-manager`**：本地引擎/模型生命周期（安装、启停、拉模型、
  模型清单 `em /v1/models`）、健康探测、集群内远程引擎控制。
- **`nvpair-workload-manager`**：集群范围 workload 事件复制。
- **`nvpair-job-scheduler`**：节点级"待处理负载优先"排序，产出优先级表。
- **`nvpair-errors`、`nvpair-manual-nodes`、`nvpair-node-settings`、`nvpair-tui`**
  等其余组件。
- 桌面端（Electron）只启动 broker，UI 永远不重复实现路由/调度/发现/密码学。

**两条分开的路径**（架构文档最重要的一张概念图）：

| | 控制面（control） | 推理面（inference） |
|---|---|---|
| 内容 | 命令下发（下行）+ 事件上报（上行）：发现快照、workload 事件、错误同步、调度优先级、节点遥测 | 用户的 HTTP 推理请求与流式响应 |
| 载体 | 本机 stdio JSON-RPC + 节点间 mTLS HTTP（`14318`~`14323`） | 本机 loopback HTTP → 集群 mTLS |
| 谁碰它 | broker/各 worker | 客户端 ↔ 代理 ↔ 远端代理 ↔ 引擎 |

**控制面不走集群中心**：每个节点运行相同的 HTTP 服务，peer 直接互调。唯一的
"每节点都会产生一份"的全局视图（workload 目录、调度序）靠**复制 + 最终一致**
实现，没有共享调度表。

### 2.3 端口表（默认安装）

| 端口 | 监听者 |
|---|---|
| `11434` | Ollama 兼容代理（Ollama 引擎被挤到 `11435+`） |
| `1234` | OpenAI 兼容代理（LM Studio 引擎到 `1235+`） |
| `14318` | 节点硬件与模型库存（node-info） |
| `14319` | 服务错误同步 |
| `14320` | workload 复制 |
| `14321` | 配对与集群成员 |
| `14322` | 集群 peer 的模型清单（em） |
| `14323` | 集群范围远程引擎控制（ec） |

引擎由 PAIR 启动时只绑 loopback，所以 **peer 永远无法直接打另一节点的引擎**，
只能走它的代理（代理是集群唯一入口、唯一挂 mTLS 的地方）。

### 2.4 一个端口两个"人格"（first-byte 分叉）

代理/集群管理器监听在 0.0.0.0，用连接首字节区分：TLS handshake 首字节 `0x16` →
按 pinned 证书做 mTLS（集群 peer 用）；否则视为明文 HTTP → **只接受 loopback**
（`127.0.0.1`/`::1`），非 loopback 明文一律 `403`。后果：

- 非 PAIR 节点的机器没有"进群的路"（明文被 403、TLS 无证书）——想让工作站享用
  整个集群，**它自己必须装 PAIR 并成为节点**，由其本地代理代它路由；
- peer 的 mTLS 入站请求**直达该节点自己的引擎，绝不再路由到第三台**（"被服务，
  不被转售"）——因此路由决策永远属于请求发起的那台机器。

---

## 3. 问题一：它在"多算力单元推理/训练体系"里处于什么位置？

把一套完整的"多算力执行栈"按切分粒度从粗到细排，PAIR 的位置一目了然：

```
应用层 / 智能体（并发调用方）
   │  OpenAI/Ollama 兼容 API
┌──────────────────────────────────────────────────────────────┐
│ 【PAIR 所在层：请求路由控制面】                              │
│  把"一个完整请求"分配给"一台持模型且空闲的整机"             │
│  = 机器副本间的、请求粒度的负载均衡（相当于 batch=1 的 DP）  │
│  不触碰模型内部、不触碰 KV、不做 batch/排队（立即转发）      │
└──────────────────────────────────────────────────────────────┘
   │  HTTP / mTLS
   ▼
每台节点 = 一台独立推理引擎（Ollama / LM Studio）
   ├─ 引擎内部（单机内，若有多个 GPU）：TP / PP / batch 排队由引擎自己处理
   │     （对 PAIR 完全不透明；PAIR 只知道"这机器引擎在跑、库存里有什么模型、
   │        忙不忙"）
   └─ 节点之间：无模型切分、无张量通信、无 KV 共享 —— 只有整包请求转发
```

对照你们（边缘分布式算力）研究的体系：

- 你们研究的 **TP/PP/EP 是把一个模型横向切开放在 N 台设备上**，需要设备间
  每层/每 token 级通信——PAIR **这一层完全不存在**；
- 你们研究的 **DP（同模型多执行流）+ 请求路由 + KV 管理 + 动态管线**，PAIR
  只覆盖了其中"**同模型多副本之间的请求路由**"这一个子集；
- PAIR 甚至没有"把模型装上某节点"的自动决策：**副本放置是人工逐节点完成的**
  （每台节点手动装引擎、手动拉模型）。

所以准确说法是：PAIR 是**"多节点推理集群的流量入口 + 信任与成员管理 + 请求级
调度"**，位于执行引擎之上、应用之下；它是你们系统里"任务调度/路由"那一小块的
一个**生产级、但刻意保持简单**的实现样例。官方自我定位的佐证（README）：

> "It does not pool GPU memory, combine GPUs into a larger logical GPU, shard
> one model across machines, or split an in-flight inference request between
> nodes." / "Adding machines therefore increases how many requests you can run
> at once. It does not make an individual request faster."

也即：**它能服务单个节点装得下的模型**；单机装不下的模型，PAIR 集群也无能为力
（模型必须整份存在于某节点）。

---

## 4. 问题二：它"调度了什么"？谁在决策？

### 4.1 调度单元 = 一个推理请求（= job / workload）

每次 `POST`（如 `/api/chat`、`/v1/chat/completions`）就是一个调度单元。调度动作
只有一句话：**在这台节点本地选一个"最不忙的、持该模型、引擎在跑、可达"的节点，
把整个请求转过去，等它流式返回。** 代理不排队（"forwards immediately rather
than queueing"），引擎内部才排队。

> 补充：`GET /v1/models` 与 `/api/tags`（模型列表）**不是**路由，而是对所有
> 候选节点**并发扇出后合并**——所以你看到的"模型列表"是集群库存的并集。

### 4.2 决策的"多级分工"（关键架构事实）

PAIR **没有一个**"调度器拍板这台跑"的中央仲裁。决策分三层，各自只做一部分：

```
┌ nvpair-job-scheduler（每节点一个，无监听端口，不直接与别的节点说话）
│   输入：broker 喂来的全集群 workload 事件 + 每节点最大 GPU 利用率
│   输出：每引擎一份 "node/set-priority" 排序表（pending + gpuPressure）
│   性质：**模型盲**——它根本不知道哪个模型在哪台机器上
▼
┌ broker（本地转发）：把排序表下发给本机两个代理
▼
┌ ollama-proxy / lmstudio-proxy（每请求决策的实际所有者）
│   ① 能力门（capability gate）：只保留"库存广告了该模型"的节点（模型匹配）
│   ② 在门内套用：手动 pin（若有且合格）> 调度器排序 > 稳定 UUID 兜底
│   ③ 加"预留"(reservation)：自己刚派发但反馈未回的在途计数
│   ④ 依次尝试（failover），首字节前可重试/换节点
```

架构文档原话佐证：

- "The scheduler is **model-blind**… The proxy **enforces model capability**… so
  load balancing happens **among the nodes that can serve the request**, not
  across the cluster as a whole."
- "the broker and the scheduler… neither touches the request, and their only job
  is to keep the proxy's preference order current."

### 4.3 排序表长什么样、怎么算（`nvpair-job-scheduler/schedule.go`）

每 1 秒（可调，下限 200ms）重算 + 事件驱动重算（节点集/workload/遥测变化）；
只有顺序或数字真变了才发 `schedule:priority`（每引擎一份，两引擎当前收同一份
节点级排序，因为共享节点资源）。

排序键（对每个节点）：

```
load = pending(node) + gpuPressure(node)
先按 load 升序；同 load 按 gpuPressure 升序；再同则按稳定 UUID（确定性兜底）
```

其中：

- `pending(node)` = 目录中所有 `queued`+`running` 且 `scheduledOn==node` 的
  workload 数；**Ollama 与 LM Studio 的负载合并成一个池**（互可见）；
- `gpuPressure` 的输入是**该节点所有 GPU 里最大的利用率**（scanner 每 2 秒采样，
  见 §8.2），经 EWMA(α=0.35) 平滑，按阈值压成 0~3：
  利用率 <40% → 0；40–69% → 1；70–84% → 2；≥85% → 3；**下降沿阈值更小**
  （35/65/80）防抖动（hysteresis）；缺失/无效/超过 10 秒的遥测按**中性压力 1**
  处理（既不当空闲也不当忙）。

**调度器明确不看的信号**（README/架构文档/known-issues 反复强调）：
GPU 型号、可用显存/VRAM、measured latency、模型是否已 warm（虽然系统知道哪些
模型在内存里，`LoadedByEngine`，但路由不用它）、单请求的代价（一个 3-token 的
回答和一个长生成都算 1）。这些是"能力"与"代价"维度，见问题四/五。

### 4.4 代理侧的自旋：预留机制（防"羊群效应"）

调度器排序天然滞后一拍：两台节点同时收到 5 个并发请求，都可能看同一台"最闲"
节点并全打过去（负载报告还没回来）。代理的解法（`proxy.go` 注释 + `reservation_
test.go`）：

- 每个代理在本地维护"自己刚派发出去、workload 反馈还没回来"的请求计数；
- 选路时把该计数加到候选的估计负载上，**选中后先 +1 再转发**；
- 于是同一时刻的一串并发请求会摊开，而不是全部扑向快照里最闲的节点。

注意它的边界：这是**本地、每代理**的预留。两个不同的节点同时派发仍可能撞车
（各自都看不到对方的预留）；系统靠 workload 复制 + 周期对账自愈，而不是预防。
官方在 Scheduler Limitations 里老实承认了这一点。

### 4.5 Failover 的确切规则（回答"调度是否带重试"）

候选 = 资格门内、按上述顺序排好的列表（首个 = 本次选中并预留的节点）。

- **什么时候能换**：只有在**响应首字节到达客户端之前**。触发条件：
  传输错误（dial/连接失败），或可重试状态码（`408/429/502/503/504`；模型级
  `404`——库存过期的"持有者"答没有该模型），或 5xx。换节点时请求体从内存重放
  （`bodyBytes`）。
- **什么时候不能换**：已经开始流式返回后（"committed"）。上游中途死掉 →
  流被截断，job 记为 failed（`workload:errored`），**没有续传、没有迁移**；
  客户端要恢复只能自己重试（重试即新请求，重新走完整选择；死掉的节点此时应已
  从发现目录消失，新请求自然绕开它）。
- 真正不可重试的客户端错误（`400/422` 等）不重试——换哪台都一样失败。
- 若资格门内一个可达的都没有 → 代理**本地直接回 502**（不带请求去任何引擎），
  不扩大候选范围，等下一次发现更新让新 owner 入池。

---

## 5. 问题三：PP / TP / DP —— 这些是"管理层"判断的吗？

**都不是。** 分三种情况讲清楚：

1. **TP（张量并行）、PP（流水线并行）、EP（专家并行）——PAIR 完全没有。**
   它不切任何模型、不做任何跨设备张量通信。README 第一行功能边界即
   "does not shard one model across machines"。因此 PAIR 的管理组件
   （scheduler/broker/proxy）**连"要不要 PP/TP、切几段"这种判断入口都没有**。

2. **单机多 GPU 的 TP/PP**：如果某台节点本身有多个 GPU，Ollama/LM Studio
   在自己进程内是否做张量并行/多 GPU 分载，是**引擎与该机器之间的事**，
   对 PAIR 完全不透明（README："Whether a particular engine and model work on a
   particular machine is between that engine and that machine"）。PAIR 只知道
   这台节点的"最大 GPU 利用率"这一个汇总信号。

3. **DP（数据并行）**：PAIR 的集群天然构成"**模型副本级的数据并行**"——同一个
   模型放在 N 台节点上就有 N 个可互换副本。但：
   - **复制度（N=几）由人决定**：在哪台节点上装哪个模型是人工逐节点准备的
     （`Add model`），管理层不做模型放置决策；
   - **每个请求去哪份副本由 PAIR 运行时决定**（资格门 + 负载排序）——这是它
     唯一真正"调度"的自由度；
   - **副本之间不做 batch 聚合、不共享 KV、不合并结果**：每个副本独立服务
     独立请求（batch 只发生在单机引擎内部队列里）。所以 PAIR 的"DP"是
     *请求粒度*的，不是你们论文意义上的 *batch/流水线内数据并行*。

结论一句话：**管理层的判断对象是"请求放哪台整机"，不是"模型如何被切开"。
模型切分维度（TP/PP/EP/段数）在这个系统里根本不存在于管理层；它们要么在引擎
内部（单机、人工/引擎配置决定），要么在你们要做的更底层的执行面里。**

顺带说明为什么 PAIR 可以"不做切分判断还成立"：它**假设单节点装得下整模型**，
路由只是把并发请求横向摊开。一旦模型放不下任何单节点，就必须引入你们研究的
执行面（跨机 PP/TP + 段间通信 + 显存/KV 管理），PAIR 在此处明确止步。

---

## 6. 问题四：接入一个新节点时，管理层如何确定它的能力（算力/内存）？

"接入一个节点"在 PAIR 里其实包含两条不同的线，**能力盘点只挂在发现线上，
与配对/信任线无关**：

### 6.1 配对 = 信任，不做能力评估

`cluster:invite-node` + 6 位 PIN（EAP-NOOB）只解决一件事：把对方 UUID 与证书
pin 下来，让之后的节点间流量可以走 mTLS。配对流程**从不读对方的 GPU/内存，
不做基准测试**。一个算力为零的机器也能被"配对"成功——它只是之后永远当不了
请求的合格 owner。

### 6.2 发现 + 富化 = 能力盘点（只读、用于展示与资格）

1. 每节点经 mDNS 广播**一条** `_nvpair-node._tcp` 记录（`noderec.go`）：只含
   身份（`uuid=`、成簇后 `cluster-uuid=`）与各服务端口（`ni/ol/lm/er/wl/cl/em/ec`）
   和候选地址列表——**不含模型清单**（TXT 太小），也不含硬件规格。
2. peer 的 scanner（每 5 秒扫描；丢失后防抖见 §10）看到新节点后，**HTTP 抓取**：
   - `:14318 /v1/node-info`（`nvpair-node-info`）→ GPU（名称、`vram_bytes` 总量、
     `vram_used_bytes`、`utilization_percent`）、CPU（名称、核数、利用率）、内存
     （总量/已用）。采集后端：Windows=DXGI+PDH，Linux=nvidia-smi（无 NVIDIA 驱动
     则 ghw 名称兜底，无动态数据），macOS=ioreg（Apple Silicon 统一内存）；
   - `:14322 em /v1/models`（engine-manager）→ 按引擎归类的模型清单 +
     **当前已加载进内存的模型**（`modelsByEngine` / `loadedByEngine`）。
     引擎没在跑则该引擎键缺失 → 其模型不算数。
3. 每条富化都保留 **last-good 缓存**：一次抓取失败不清空节点卡片；
   遥测采样失败有退避（2s→30s）。
4. `DirectoryNode` 把 `GPUs/CPU/Memory/Models/ModelsByEngine/LoadedByEngine` 全带
   上，UI 据此显示节点卡与 GPU 图表；代理根据 `ModelsByEngine[engine]` 判断资格。

### 6.3 关键事实：**"算力与内存"不参与调度排序**

- 调度器（§4.3）的输入只有 `pending` 与 `max GPU utilization`；
  **GPU 型号、显存容量、内存容量一个都不是输入**（known-issues 明说 "Routing
  does not consider VRAM at all, because the scheduler counts workloads"）。
- "节点能不能服务模型 X"的资格判定 = **可达 + 跑着对应引擎 + 引擎库存里在册**
  （以及隐含的"装得下"，那是引擎加载时自己失败的事，不是路由判的）。
- 所以如果你们的问题是想确认"管理层会不会像 vLLM/你们的系统那样按空闲显存决定
  能不能放这个模型/这组 KV"——**PAIR 不会**。它既不做显存约束检查，也不自动把
  模型放到"显存够的节点"。

> 意义：PAIR 对"异构能力"的态度是**只展示、不建模**。官方 roadmap 说得很坦白：
> 现政策 "does not consider GPU model, available memory, model warmness, or how
> expensive a request looks, which still makes it a better fit for similar
> machines than a highly mixed cluster"。

### 6.4 手动节点（mDNS 被禁的网络）

`nvpair-manual-nodes` 接受手输地址，每 10 秒探测；能答 `/v1/node-info` 就并入
同一目录，先按地址、读到 `hostUuid` 后改按真实 UUID 为键——身份仍然是 UUID 而非
地址。

---

## 7. 问题五：节点之间的带宽是管理层的优化考量吗？

**不是，而且是有意不做。** 逐条证据：

1. **没有任何带宽/链路质量测量**：全部节点间面是"请求转发 + 遥测/事件复制"，
   没有任何吞吐/延迟采样组件；`node-info` 里没有网卡速率（只有 GPU/CPU/内存）。
2. **路由输入不含网络信号**：scheduler 排序键 = pending + GPU pressure（§4.3）；
   代理选择 = 资格门 + 排序 + 预留。文档原话：调度器 "is not VRAM-capacity- or
   latency-aware"（scheduler README），Scheduler Limitations 列 "GPU model,
   available VRAM, and measured latency are not inputs"。
3. **连"离自己近/同机"都不加分**：路由不看拓扑、不看跳数；本地节点与远端节点
   一视同仁地被排序（本地只是走 loopback 而非 mTLS 的实现差别）。
4. 代理确实测量了 `ttfb_ms/duration_ms`（`proxy/request` 事件），但**只用于
   UI 展示/日志，不反馈给路由**。

隐含设计假设：**默认同 LAN、链路够用且同质**（mDNS 发现本身就是局域网机制）。
它唯一"网络感知"的地方是**连通性**层面的：每 peer 发布候选地址列表
（`ips=`，最多 4 个），拨号按序尝试、记住成功者；节点掉线有探活与驱逐（§10）。
这些是**可达性/容错**机制，与"带宽优化"无关。

对照你们的系统：PAIR 对带宽的态度恰好说明它**不做**你们最关心的事情——
在"带宽受限、链路异构"下按通信代价决定切分与放置（如 Helix 的最大流、EP 的
all-to-all 代价）。你们若要把它当参照物，这一条是**明摆着的缺口/分工边界**。

---

## 8. 问题六：支持节点中途离开吗？

要分三层回答，PAIR 三层都有明确机制，但"支持"的程度不同：

### 8.1 成员主动离开（cluster:leave / nodes:remove）

- 本机 `cluster:leave`：向所有成员宣告离开、删光自己的 pins/members、回到
  未成簇状态（发 `cluster:identity-changed{clusterId:""}`）。成员由 peer
  `nodes:remove {nodeId}` 移除亦可（本地先生效，离线 peer 之后按 roster 同步）。
- 离开后：该节点的 `cluster-uuid=` 从 mDNS 记录消失；所有集群门（mTLS、roster、
  引擎远程控制）立刻把它当外人；peer 若仍持有它的旧 principal 会因 node-info 的
  `clusterUuid` 上报而纠正（noderec 里专门为此设计了 "absent/空/有值"三态）。
- 代价面：**正在它上面跑的请求不会被打断通知、不会被搬走**——只是它离开后，
  后续请求不可能再选它（trust 层已断，mTLS 也过不了）。

### 8.2 机器掉线/失联（非主动）——发现目录的"驱逐防抖"

这是 PAIR 设计里最讲究的部分（`nvpair-node-scanner/daemon.go`）：

- scanner 每 5 秒广播/浏览一轮；连续 **3 次没见到**某节点的公告才开始怀疑它
  （≈15s）。
- 即使到阈值，也**不立即驱逐**：先找"还活着的证据"，按序检查：
  1. **最近 1 分钟内它曾流式返回推理字节**（`node/activity`，代理每 2s 合并上报
     一次，scanner 视为最强生命证据——忙的节点恰恰最答不上探针，但正在吐 token）；
  2. 10 秒内的成功 node-info 富化缓存；
  3. TCP 直连探测其 `ni`(14318) / `em`(14322) 端口（**绝不探测推理代理端口**
     ——避免把连接尝试压到服务路径上）；
  4. 探到东西在听时，再读一次它的 `hostUuid` 确认"还是原来那台机器"
     （防止机器被重置/换身份后，旧记录被地址顶替成永久的幽灵副本）。
- 确证离开后：从目录删除（`forget()` 顺带清掉它的遥测/活动/富化缓存），broker
  向各代理推送新的 `discovery:nodes` 全量快照，代理整体替换路由表——**离开的
  节点只是从下一个快照里消失**，不单独"下线"。
- 为什么这么防抖：多播本来就丢包，Wi-Fi/睡眠/换网卡都会造成假离线；一句话——
  "mDNS miss ≠ 节点死了"。

### 8.3 进行中的请求——**不迁移、不续传**

回到 §4.5 的结论，与成员离开叠加：

- 请求**尚未提交流式返回**（还在排队/尝试候选）时节点消失 → 传输错误 → failover
  自动试下一个合格 owner（首字节前）；全部失败 → 本机 502；
- 请求**已经开始流式返回**时节点消失 → TCP 断 → 该 job 记
  `workload:errored`（错误如 "upstream error" / 连接丢失），UI Jobs 可见；
  **没有自动在另一节点重放/续传**（已发出去的中间 token 无法回收），客户端自己
  重试后才会以"新请求"身份重新选路，而这时目录里已没有那台节点了。
- 系统层面：无共享调度表、最终一致 → 节点消失后的一两秒内，别的节点可能还拿
  旧快照往它头上派（撞一下然后 failover/502），随后自愈。官方在
  Scheduler Limitations 承认："There is no shared schedule… The system
  self-corrects through workload relay and periodic reconciliation rather than
  preventing the collision."

> 一句话回答"是否支持节点中途离开"：**支持（成员/目录/路由三层都有设计，且
> 专门做了防抖动），但对正在生成中的请求不提供迁移/续传——掉链子的那一次请求
> 失败，靠客户端重试与目录自愈来恢复。** 这跟你们设计中"节点离开时迁移模型参数
> 与 KV 缓存、把流水线重新接上"的目标是完全不同的能力层次。

---

## 9. 问题七：真来了一个推理请求，系统从头到尾怎么跑？

假设：2~3 台 Windows/Linux/macOS 机器已配对成簇；每台装了 Ollama 并 pull 了
`qwen4:12b`（“同模型多副本”是让路由有得选的先决条件）；工作站 A 上的应用通过
Ollama 客户端发请求。下面走 A 的视角。

### 9.1 时序总览

```mermaid
sequenceDiagram
    autonumber
    participant App as 应用/智能体 (A 机)
    participant P as ollama-proxy (A 机, :11434)
    participant S as job-scheduler (A 机)
    participant W as workload-manager (A 机→全网)
    participant E_A as 本地引擎 Ollama (A 机, :11435 loopback)
    participant P_B as ollama-proxy (B 机, :11434)
    participant E_B as 引擎 Ollama (B 机, :11435 loopback)

    Note over S,P: 常驻背景：scanner 每2s采样各节点GPU利用率→broker→S;<br/>各节点workload事件→W复制→broker→S;<br/>S算好排序→broker→P 下发 node/set-priority
    App->>P: POST /v1/chat/completions {model:"qwen4:12b",...} (127.0.0.1)
    P->>P: 解析model；取本机最新发现快照；<br/>资格门：保留“跑着Ollama且库存含qwen4:12b”的节点
    P->>P: 排序：手动pin? > scheduler优先级(加预留) > UUID兜底；<br/>把选中节点+1预留，组成本请求failover列表
    P->>P: 发出 workload:started (scheduledOn=选中节点) → broker → W → 全网
    alt 选中节点 = B（远端）
        P->>P_B: 首字节0x16 → mTLS(该peer的pinned证书) 转发整请求
        P_B->>E_B: 直通本机引擎（loopback），不再路由
        E_B-->>P_B: 流式 SSE/JSON token
        P_B-->>P: 流式回传；期间代理上报 node/activity（防驱逐证据）
        P-->>App: 流式响应（客户端只见普通回复）
        Note over P,W: 完成 → workload:completed → W → 全网（B机该节点pending-1）
    else 选中节点 = A 本机
        P->>E_A: loopback 直连本地引擎（node/set-local-backend 指向 :11435）
        E_A-->>P: 流式 token → 回给 App
    else 无任何合格owner
        P-->>App: 本地 502（不出网）
    end
    Note over P: 若首字节前失败/5xx/模型404 → 自动换failover列表下一节点(重放请求体)<br/>若已开始流式后断链 → 不迁移，job记failed，客户端自重重试
```

### 9.2 分步说明（含真实端口与代码依据）

**前置（一次性，做一次后常驻）：**

- **发现**：每节点 scanner 广播 `_nvpair-node._tcp`，互为 peer 建立目录；
  节点互相富化（node-info 硬件 + em 模型清单），本机 broker 把"谁在跑 Ollama、
  都有什么模型"推给本机 `ollama-proxy`（`discovery:nodes` 快照，整体替换）。
- **信任**：成簇后，节点间推理入口开 mTLS（pinned 证书）；没配对过的机器连
  端口都会被拒。
- **背景控制回路**（一直转，决定“偏好序”，不碰请求）：
  ① 每节点 scanner 每 ~2s 抓各 peer 的 `/v1/node-info`，取**最大 GPU 利用率**，
  发 `discovery:node-telemetry` → broker（缓存去重）→ `scheduler:telemetry`；
  ② 各代理每个推理请求发 `workload:started/completed/errored` → broker →
  workload-manager（:14320，mTLS）→ **全网每台 broker 都复制一份** → 各自的
  scheduler；scheduler 每 ~1s + 事件驱动重算
  `load=pending+gpuPressure`，变了才发 `schedule:priority` → broker →
  `node/set-priority` 下发给两个代理。

**请求时刻（A 机视角）：**

1. 应用照常打 `http://127.0.0.1:11434/v1/chat/completions`。因为代理占了 11434、
   引擎被挤到 11435，请求**必然先进代理**（这也是"不配置任何东西就透明获得整个
   集群"的关键：`port takeover`）。
2. 代理解析请求体里的 `model`（Ollama 的 `:latest` 隐式标签会归一化）。取一份
   **请求局部的发现快照**，执行能力门：`EngineModels("ollama")` 里广告了
   `qwen4:12b` 的节点才留下；空库存/其它模型的节点出局；LM Studio 的模型名要
   精确匹配。若一个 owner 都没有 → 直接 `502`。
3. 排序并选路（§4.2/4.4）：owner 集合里，手动 pin 优先（仅当 pin 的节点合格；
   TUI 才有此能力，桌面 UI 从不 pin）；否则按调度器下发序；调度器没排过的
   （冷启动/手动节点）按稳定 UUID 兜底排在最后。代理把自己"刚派发未回执"的计数
   加进去（预留），选中后 +1 再发。
4. 转发：选中 B → 用给 B 的 pinned 证书做 mTLS 打 `B:11434`（B 的代理）；
   选中本机 → loopback 直连本地引擎 `127.0.0.1:11435`。B 的代理收到 mTLS 请求后
   **只转发给本机引擎**（`node/set-local-backend` 目标），绝不再次选路。
   请求体保留在内存副本（`bodyBytes`），供 failover 重放。
5. 流式回传 + 可观测：token 从 B 引擎 → B 代理 → A 代理 → 应用，路径与去程一致。
   回传字节驱动 `node/activity`（防 B 被误驱逐）；`proxy/request-started` 与
   `proxy/request`（ttfb/时长）供 UI 与统计；workload 状态机
   `started→completed/errored` 经 workload-manager 全网可见（UI 的 Jobs 视图即
   此），也喂回调度器让 B 的 pending 减一。
6. 失败处理：首字节前的传输错/可重试码 → 换下一 owner（重放 body）；全失败 →
   502。流式开始后断链 → 截断 + job failed + 无迁移（§8.3）。

**模型列表请求（如 `GET /v1/models`）**：并发扇出到所有候选并合并去重 —— 应用
因此看到"整个集群的模型并集"，而不是某一台的。

### 9.3 训练任务呢？

**没有。** 引擎是推理服务器；PAIR 控制面与数据面都没有训练/微调/job 编排。
如果你们的课程里"执行推理/**训练**任务"必须都覆盖，PAIR 只示范了前者。

---

## 10. 成员、发现与信任细节（支撑以上结论的机制）

- **身份**：一切以稳定 UUID 为键（mDNS instance name 只是 hostname，仅显示用；
  重命名主机 ≠ 换节点；hostname 撞名不合并）。配对也 pin UUID↔证书。
- **可信注解**：浏览到的 peer 若 `cluster-uuid=` 匹配本机 pin 则标 trusted；
  目录里"看得到"与"可信任"是两件事（未配对也可见）。
- **离开/移除的传播**：cluster-manager 广播 departure/roster；其余节点 mTLS 门
  立刻拒；发现记录的 `cluster-uuid=` 变化 + node-info 的 `clusterUuid` 三态上报
  保证陈旧 principal 被纠正。
- **运行期 worker 故障**：broker 只保活进程（退出→退避重启，最多 5 次/60s），
  **不检测"卡死但没退"的 worker**（known issue：UI 状态停更时手动重启）。
- **引擎层**：engine-manager 做启动就绪探测 + 周期健康探测，`healthy` 标志驱动
  错误登记；PAIR 只把“引擎在跑”体现在 em 模型清单的键存在性上。
- **安全面摘要**（对“在局域网部署”的读者有意义）：明文只有 node-info 遥测与
  配对引导（PIN）；推理、workload、errors、roster、远程引擎控制均 mTLS；
  代理端口非 loopback 明文 403；引擎只绑 loopback。

---

## 11. 与你们的"边缘分布式算力（跨机 PP/TP/EP + KV + 动态管线）"对照

你们的汇报里，系统要做的核心是：模型在内存受限的边缘设备上**切分装载**
（PP 为主、辅以 MoE/EP、分级加载）、把多条**动态流水线**组织起来、
请求/任务调度 + **KV 缓存命中与迁移**、节点离开时的**参数与 KV 补救**、
新节点**评估并入**、以及带宽受限下的**放置/路径优化**。

PAIR 与这个蓝图的关系：

| 维度 | PAIR | 你们的目标系统 |
|---|---|---|
| 模型放置 | 人工逐节点放整模型；单机装不下就没戏 | 按算力/内存/带宽**自动切分与放置**（PP/EP/分级） |
| 单请求执行 | 一台节点整跑，模型内部不可见 | 跨节点流水线/张量/专家协同执行 |
| 并行度判断 | 无（引擎内部、人工、对系统不透明） | 管理层（并行度/段数/批大小随状态调整） |
| 调度输入 | pending + 最大GPU利用率 | 空闲显存、KV命中、专家命中、链路带宽、队列 |
| 网络 | 不做带宽/延迟建模；默认 LAN 同质 | 把带宽当一等公民（PP>TP 的选择即出于此） |
| KV | 每节点自管；路由不看 loaded 状态；节点离开即丢 | 跨节点 KV 复用、分级存储、离开时按价值迁移 |
| 节点离开 | 成员层优雅；在途请求失败不迁移；目录防抖驱逐 | 流水线重接、参数去重装载、KV 按序迁移 |
| 弹性 | 加副本 = 加并发（请求级） | 加节点 = 加流水线段/副本，可加速单请求 |
| 一致性 | 最终一致、无共享调度表（官方自认会短暂撞车） | 需要更强的一致放置视图（你们自研管理层） |
| 信任/产品化 | PIN+mTLS、GUI、跨平台、安装器——完整 | （通常不涉及） |

**值得借鉴的实现细节（若你们自研"管理层"）：**

1. **身份与目录分离**：稳定 UUID 贯穿发现/成员/workload/遥测/路由；mDNS 只当
   引导，重信息走 HTTP 富化 + last-good 缓存。你们的节点注册表可以直接照搬。
2. **驱逐防抖范式**（对"节点中途离开"最有价值）：公告丢失 ≠ 离线；离线判定前
   先找"正忙着吐 token/最近回包/还能拨通"的证据，且**探测绝不打扰服务端口**；
   确认前再验一次身份防“重置后的幽灵”。这条直接能改进你们“节点离开→迁移”的
   触发时机，避免把瞬时抖动的健康节点误判为离开而白做迁移。
3. **资格门与负载均衡分层**：调度器 model-blind、只出偏好序；代理在“确实能服务
   该请求的 owner 集合”内套用排序——负载均衡永远只在合格者之间发生。你们同样
   应该把“谁有资格（持参数/专家/KV）”和“谁更该去（负载/代价）”分开。
4. **预留(per-proxy reservation) + 事件驱动最终一致**：用“自己刚派发的在途数”
   抵消反馈延迟的羊群效应；配合只读“负载盲板”与稳态排序的 hysteresis，是
   轻量去中心化负载均衡的成熟做法。
5. **活动即生命**（activity as liveness）：用推理回传字节当生命信号——对你们
   “正在跑长生成/迁移中”的节点同样适用。
6. **端口接管（proxy 占引擎默认端口、引擎 loopback 化）**：让存量应用零改造获得
   集群能力 + 天然把引擎锁在本机。你们的客户端接入也可以这么做。
7. **诚实的能力边界声明**：PAIR 把“不做什么”写成文档与代码注释（不聚合显存、
   不切模型、不迁移在途请求、不看显存/带宽），这本身就是架构清晰度。

**PAIR 没有、而你们必须自研的（也是它止步之处）：**
模型切分与装载器、段间通信抽象、KV 缓存管理与迁移、基于显存/带宽/代价的放置
与路径决策、对单请求的跨节点执行、batch/排队、抢占与在途迁移。也就是说：PAIR
给你的是一个可借鉴的**“请求路由 + 信任 + 成员 + 观测”控制面外壳**，其内芯
（模型并行执行面）正是你们论文的主体工作。

---

## 12. 源码地图（每个论断去哪看）

| 主题 | 位置 |
|---|---|
| 定位/边界/路线 | `README.md`（"What is supported / What PAIR does not do / Where PAIR is going"） |
| 总览与概念 | `docs/overview.mdx` |
| 架构总纲（分层/进程/端口/信任/发现/驱逐/调度局限） | `docs/architecture.mdx` |
| 已知局限 | `docs/known-issues.mdx` |
| 服务总览/构建/测试 | `services/readme.md` |
| 调度排序与遥测平滑 | `services/nvpair-job-scheduler/schedule.go`、`telemetry.go`、`README.md` |
| 代理路由/资格门/预留/failover/流式语义 | `services/ollama-proxy/proxy.go`（failover loop 1200–1454 行附近）、`README.md`；`lmstudio-proxy` 为刻意克隆（同契约） |
| 路由目标集与快照整体替换 | `services/ollama-proxy/discovery.go` |
| workload 状态机 | `services/nvpair-workload-manager/workload.go`、`README.md` |
| 节点能力/遥测盘点 | `services/nvpair-node-info/main.go`、`README.md`（含各平台采集说明） |
| 发现目录与驱逐防抖 | `services/nvpair-node-scanner/daemon.go`（`reachable()` 约 991 行起、telemetry 约 1576 行起）、`noderec` 相关测试 |
| 记录格式与身份/TXT/端口策略 | `services/shared/noderec/noderec.go` |
| 配对/成员/leave/remove/信任 | `services/nvpair-cluster-manager/README.md`（methods 与 flows 部分） |
| 模型清单/引擎生命周期 | `services/nvpair-engine-manager`（目录内 README） |
| 手动节点 | `services/nvpair-manual-nodes` |
| 推理压测客户端（演示路径） | `docs/inference-dispatcher.mdx`、`scripts/inference-dispatcher*` |

---

## 13. 一句话总结

PAIR = **“多台整机 × 整模型副本”之上的请求级路由控制面**：它把"每台节点跑
引擎、每请求选一台”这件事做成了一套安全、可观测、可容忍节点漂移的生产系统；
**它不切模型、不共享 KV、不感知带宽**，所以 PP/TP/EP 与显存/网络优化在它的
管理层里不存在——那是你们的研究要做、且必须放在 PAIR 这层“之下/之外”的执行面。
读它的价值在于：看一个真实系统如何干净地做请求调度、成员信任、节点漂移容忍与
观测，并把其中可复用的范式（UUID 目录、资格门、驱逐防抖、预留、活动即生命、
端口接管）搬进你们的管理层设计。
