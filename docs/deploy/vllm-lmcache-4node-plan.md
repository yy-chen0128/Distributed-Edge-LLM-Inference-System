# vLLM + LMCache 四机推理：方案与执行手册

> **决策（2026-09-22）**：执行面先不自研，改用 **vLLM 作为 stage 引擎 + LMCache 做 KV** 把四机推理跑起来；
> 节点脱离按 **档0「气泡重启流水线」** 处理（停接纳 → drain → 幸存者重组 → 重启 → 恢复接纳）。
> 自研分层引擎的结论保留在 `docs/research/`，作为对照与备用。
>
> 本文的脚本都是可直接跑的；标 `❓待验证` 的地方必须装完 vLLM 后按实际版本核对，不要当已知事实。

---

## 0. 先摆清楚四条硬约束（它们决定方案能做多大）

| # | 约束 | 后果 | 我们的对策 |
|---|---|---|---|
| 1 | **vLLM 的 PP 并行度在启动期定死**，改 PP 必须重启进程 | "节点变化"＝**整条流水线重启**，这正是档0 的语义 | 接受它；把优化目标定为**让重启的气泡尽量小**（见 §5） |
| 2 | 官方要求**各节点执行环境一致**（原文 "to **hide host heterogeneity**"） | 8/6/4/4 的**算力差异 vLLM 不利用**；每 rank 拿到的是**均分后的连续层区间**，不是按能力加权 | 异构适配改从**模型量化**入手（见约束 3），而不是指望 vLLM 的切分 |
| 3 | 每 rank 的层数由 vLLM 尽量均分；**它不会按显存能力加权** | 7B fp16 每层 466 MB → 均分 7 层 = **3.26 GB + activation + KV**，**4 GB 的卡会 OOM** | **用 AWQ/GPTQ int4 的 7B**（每层约 0.13 GB → 7 层 ≈ 0.9 GB，4 GB 卡无压力）。vLLM **原生支持**量化权重，这恰好是自研引擎的短板 |
| 4 | vLLM 官方只支持 Linux | 必须在 WSL 里跑（本机 WSL 已具备 CUDA） | 用**独立 venv** 装 vLLM，别污染现有 `~/venvs/pair` |

> **一句话**：vLLM 路线把"异构"这件事从"切分策略"挪到了"选哪个量化模型"上。
> 好处是**今天就能跑**（量化权重是现成的）；代价是**放弃了"按设备能力加权切层"这个卖点**（那是自研引擎的贡献）。
> 这不是矛盾——两条路线本来就是"能跑起来"与"能发表"的分工。

---

## 1. 目标与验收

| 阶段 | 目标 | 验收标准 |
|---|---|---|
| **V1** | 单机跑通 vLLM + LMCache（1 卡） | 起服务、发请求有正确输出；重复长前缀的第二个请求**命中 LMCache**（指标/日志可见） |
| **V2** | 单机模拟多 rank 的 PP（1 卡超订 2/4 rank） | PP=2/4 能起来并出正确结果 → **验证机制**（不代表真机性能） |
| **V2b** | **本地 Ray 集群**（head + worker 同机）+ PP over Ray | 验证**四机要用的那条代码路径**（`--distributed-executor-backend ray`）；单机就能做，是风险最大的一步 |
| **V3** | **四台真机**跑 PP=4 + LMCache | 端到端正确；记录 TTFT、吞吐、每 rank 显存、启动/装载耗时 |
| **V4** | 档0 气泡重启 | 手动停掉一台 → 剩余机器重建流水线 → 记录**气泡时长**与恢复后吞吐 |

### 1.1 完整验证清单（单机能做的 / 必须真机的）

| # | 验证项 | 单机可做？ | 状态 |
|---|---|---|---|
| 1 | vLLM 单实例出正确输出 | ✅ | 待装完 |
| 2 | LMCache 命中（同前缀第 2 次更快） | ✅ | 待装完 |
| 3 | **LMCache 能否跨重启命中**（决定档1 气泡） | ✅ | 待装完 |
| 4 | 单卡超订 PP=4 出正确结果（机制） | ✅ | 待装完 |
| 5 | **本地 Ray + PP over Ray**（四机同款路径） | ✅ | 待装完 |
| 6 | **vLLM 启动/装载耗时**（档0 气泡第一项） | ✅ | 待装完 |
| 7 | **AWQ int4 7B 加载成功 + 每 rank 显存**（决定 4GB 机器能否上） | ✅ | 模型下载中 |
| 8 | **网关端到端**：杀进程 → 重启 → 重放 → 客户端输出完整不重复 | ✅ | 网关已测（单测）；端到端待 vLLM |
| 9 | 重放保真度 `replay_diverged` 是否为 0 | ✅ | 同上 |
| 10 | **TPOT：4 机 PP=4 vs 单机 PP=1**（整个方案的成败判据） | ❌ **必须真机** | 等机器 |
| 11 | 每台显存占用是否符合预期 | ❌ 必须真机 | 等机器 |
| 12 | WiFi 下两两 RTT/吞吐/抖动/丢包 | ❌ 必须真机 | 等机器 |

> 第 10 条是唯一一个**无法用单机替代**的关键数字。单机上的 PP=4 是 4 个 rank 抢同一块卡，
> 性能没有意义；只有真机能回答"跨机到底值不值"。

> ⚠️ **本机只有一块 GPU（RTX 4060 Laptop）**。V1/V2 我能做，**V3/V4 需要你那四台笔记本**——
> 我这边做不了"四机"，只能把 V3 的手册与脚本准备到"复制粘贴即可执行"的程度。

---

## 2. 部署架构（四机）

```
        客户端
          │  HTTP（OpenAI 兼容）
          ▼
   ┌──────────────────┐
   │ 入口网关（我们写） │  ← 档0 的"停接纳/drain"在这里做
   └────────┬─────────┘
            │
   ┌────────▼─────────────────────────────────┐
   │ Ray head（机器 A）+ 3 个 Ray worker       │
   │ vllm serve --pipeline-parallel-size 4     │
   │   rank0: A 层 0..6   ─┐                   │
   │   rank1: B 层 7..13   │ 每 rank 一张卡     │
   │   rank2: C 层 14..20  │                   │
   │   rank3: D 层 21..27 ─┘                   │
   └──────────────────────────────────────────┘
   每台机器上的 LMCache：本地 CPU/磁盘缓存 + 可选的远端共享层
```

要点：
- **Ray 负责跨机通信**（vLLM 的 `--distributed-executor-backend ray`）；
- **LMCache 挂在 vLLM 的 KV connector 上**（`--kv-transfer-config`，`LMCacheConnectorV1`）❓具体字段名装完后按版本核对；
- **入口网关必须我们自己写**——vLLM 不提供"停接纳 + drain + 重建"的语义，档0 的关键就在这个网关上。

---

## 3. 安装（WSL，独立 venv）

见 `scripts/wsl_install_vllm.sh`。要点：

- **独立 venv**（`~/venvs/vllm`）——vLLM 会拉自己的 torch，不能污染 `~/venvs/pair`；
- **`PIP_CACHE_DIR` 指到 `/mnt/d/pipcache`**——避免 WSL 的 ext4.vhdx 被 pip 缓存撑大（那只增不减）；
- 装完核对 `torch.cuda.is_available()` 与 `vllm.__version__`；
- 预期体积：**venv 约 6–9 GB**（torch + vllm + flashinfer 等），全部落在 C 盘的 vhdx 里。

---

## 4. 单机验证（V1/V2，我能做）

### V1：单实例 + LMCache 命中

1. 起服务（0.5B 先跑通，再换量化 7B）；
2. 同一个**长前缀**发两次请求，比较第二次的 TTFT 与 vLLM 的 prefix-cache 指标；
3. 关掉 LMCache 再发一次，作为对照 → **得到"缓存命中带来多少 TTFT 下降"**。

### V2：单卡超订跑 PP

在单卡上用 Ray 报出 4 张 GPU（超订），让 vLLM 的 PP=4 落在一块物理卡上。
**目的仅是验证机制**（能起来、结果正确、能观察到 rank 间 activation 传递），
**性能数字无意义**（4 个 rank 抢同一块卡）。

见 `scripts/wsl_vllm_pp_local.sh`。

---

## 5. 节点脱离：档0 + 档1（**已确认**）

**决策（用户 2026-09-22 确认）**：节点脱离按"气泡重启流水线"处理，且**在途请求不丢弃**——
恢复方式就是 **把已生成的内容拼回提示词、重新 prefill 一遍**（即 `elasticity-cost-benefit.md` 里的档1）。

### 5.1 完整流程

```
① 检测到节点离开（心跳超时 / Ray 报 worker 掉线 / nvidia-smi 轮询失败）
② 入口网关：停止接纳新请求（503 或排队），但**不断开已建立的客户端流**
③ drain：等在途请求自然结束；超过 grace（例如 30 s）则把它们交给档1
④ 用幸存节点重建：ray stop → ray start（新的 --address 列表）→ vllm serve
      --pipeline-parallel-size = 幸存者数量
⑤ 预热（发一条短请求，把权重/编译缓存热起来）
⑥ **档1 重放**：对每个被中断的请求，用 [原 prompt + 已生成内容] 作为新 prompt 重新提交，
   只把**新增**的输出转发给客户端；客户端那条 SSE 流全程不断（用户只感到一次卡顿）
⑦ 入口网关恢复接纳
```

### 5.2 入口网关要实现的四件事（档0+档1 的实现全在这里）

**已实现**：`edge_llm_scheduler/gateway/vllm_gateway.py`（纯逻辑单测见
`edge_llm_scheduler/tests/test_gateway_replay.py`，9 个用例）。
启动：`python -m edge_llm_scheduler.gateway.vllm_gateway --upstream http://127.0.0.1:8000 --port 8100`
（需要 `fastapi/uvicorn/httpx`，已装进 `~/venvs/pair`）。

| 职责 | 说明 | 实现位置 |
|---|---|---|
| **累积** | 边转发边记录：`原 prompt` + `已生成的内容`（重放的原料；也顺便解决了 progress 读数问题） | `_stream_with_recovery` 里的 `sent` |
| **判故障** | 下游连接错误 / 流中断 / HTTP >= 400 → 置"流水线不可用" | 异常捕获 + `state.mark_down` |
| **停接纳 + drain** | 新请求 503；在途请求进入重放流程 | `/admin/drain`、`/admin/resume`、`GET /admin/status` |
| **重放** | 用 `[原 prompt + 已生成内容]` 重新提交，**只转发新增部分**，客户端那条 SSE 流不断 | `build_replay_body` + `skip_already_sent` |

**额外做的一件事——把"分词器不保证恒等"变成指标**：重放输出与"已发送内容"做前缀比对，
不一致就计数 `replay_diverged`（`GET /admin/status` 可见）。
这样 §5.3 里那个坑不再只是口头风险，而是**每次实验都能看到的保真度数字**。

### 5.3 一个必须先查清的技术点：重放用**文本**还是**token id**

vLLM 的 OpenAI 兼容接口返回的是**文本**。把文本重新拼回 prompt 会**重新分词**，
而 detokenize → tokenize **不保证是恒等变换**（byte-level BPE、特殊字符、被跳过的特殊 token）。
所以：

- **优先**用 **token ids** 重放（精确）；要查：`/v1/completions` 是否支持 `prompt_token_ids`、
  是否有 `/tokenize` 端点、流式响应里能否拿到 token id（`logprobs` 给的是 token **文本**，不是 id）；
- **退路**：用文本重放，并在实验里**声明**"续写结果可能与未中断的世界略有不同"
  （采样本身也有随机性；温度=0 时差异只来自分词与数值）。

### 5.4 LMCache 在这里的作用（正好接上 §4 的 D 段测量）

重建之后重放时：

- **若 LMCache 仍持有该前缀的 KV** → 重放的 prefill **命中缓存**，几乎免费；
- **若没有** → 是一次完整 prefill（我们算过：Azure conv 画像 0.77 s、Mooncake 4.8 s，7B）。

**所以 §4 里 D 段（重启后 LMCache 还能不能命中）直接决定档1 的气泡大小。** 这也是这次冒烟测试最值钱的一条。

### 5.5 决定"气泡"大小的四个量（都要实测）

| 量 | 我们现在知道的 | 怎么测 |
|---|---|---|
| **权重装载** | 自研引擎实测 7B 每 4 层 35.7 s；**vLLM 要单独测**（它会做 CUDA graph 捕获，可能更慢） | 记录 `vllm serve` 从启动到 ready 的时间 |
| drain 时长 | 与请求长度分布有关 | 按负载画像（Azure code/conv、Mooncake）算 p50/p99 |
| Ray 集群重建 | 未知 | 计时 |
| **档1 重放的 prefill** | 0.77–4.8 s（有 LMCache 命中时≈0） | 见 §4 的 D 段 + 网关打点 |

### 5.6 未在这里做的：档2

冗余镜像（每段 ≥2 持有者）不做——只在无人机/抢占式那种极端频率下才值，
依据见 `docs/design/elasticity-cost-benefit.md` §5。

---

## 6. 与我们已有代码的接口

- `edge_llm_scheduler/backends/vllm_engine.py`（已有 `VLLMEngine`，HTTP 客户端）——
  把 vLLM 当"某台机器上的节点引擎"接入我们的 `TaskScheduler`；
- 我们的调度器继续负责：请求路由（含缓存感知）、epoch/档0 编排、指标；
  **计算全部交给 vLLM**；
- 好处：**我们的策略层（E2Placement / PriorityMigration / CapabilityReparallelization）可以原封不动地拿 vLLM 当执行面**，
  而且基线对比变得自然（同一批策略 vs 纯 vLLM 调度）。

---

## 7. 未决问题（需要你定）

| # | 问题 | 我的建议 |
|---|---|---|
| 1 | 四台真机什么时候可用？IP/账号怎么给我？ | 给我一台的 ssh 访问即可开始；四台的接入方式见 §2 |
| 2 | 用哪个模型？ | **先 0.5B 跑通**（最快暴露问题）→ 再上 **7B 的 AWQ/GPTQ int4**（`Qwen2.5-7B-Instruct-AWQ`，约 4.4 GB，每 rank ≈1.1 GB，4 GB 卡放得下） |
| 3 | LMCache 用本地还是要跨机共享？ | 先**本地 CPU 缓存**（最简单，先拿到命中率数字）；跨机共享（`lmcache_server` + `remote_url="lm://"`）作为第二步 |
| 4 | 入口网关谁来写？ | 我写一个最小的（FastAPI：队列 + 503 + drain 开关），半天级 |

---

## 8. 执行顺序（WSL 一恢复就按这个走）

1. `bash scripts/wsl_install_vllm.sh` —— 装 vLLM + LMCache（**长下载，放后台**）
2. `bash scripts/wsl_vllm_smoke.sh` —— V1：单实例 + LMCache 命中验证（拿到第一组数字）
3. `bash scripts/wsl_vllm_pp_local.sh` —— V2：单卡超订 PP=4（验证机制）
4. 按 §5 写档0 网关 + 重启脚本
5. 四机手册（按 §2 参数化）→ 你在真机上跑 V3/V4
