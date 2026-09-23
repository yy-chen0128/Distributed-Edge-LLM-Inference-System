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
| **V3** | **四台真机**跑 PP=4 + LMCache | 端到端正确；记录 TTFT、吞吐、每 rank 显存、启动/装载耗时 |
| **V4** | 档0 气泡重启 | 手动停掉一台 → 剩余机器重建流水线 → 记录**气泡时长**与恢复后吞吐 |

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

## 5. 档0「气泡重启」的可执行流程

```
① 检测到节点离开（心跳超时 / Ray 报 worker 掉线 / nvidia-smi 轮询失败）
② 入口网关：停止接纳新请求（返回 503 或排队），但**不断开已建立的流**
③ drain：等在途请求自然结束；超过 grace（例如 30 s）则取消剩余请求
④ 用幸存节点重建：ray stop → ray start（新的 --address 列表）→ vllm serve
      --pipeline-parallel-size = 幸存者数量
⑤ 预热（发一条短请求，把权重/编译缓存热起来）
⑥ 入口网关恢复接纳
```

**决定"气泡"大小的三个量（都要实测）**：

| 量 | 我们现在知道的 | 怎么测 |
|---|---|---|
| **权重装载** | 自研引擎实测 7B 每 4 层 35.7 s、按 12 层推算 ≈126 s；**vLLM 的装载时间要单独测**（它会做 CUDA graph 捕获，可能更慢） | 记录 `vllm serve` 从启动到 ready 的时间 |
| drain 时长 | 与请求长度分布有关 | 按负载画像（Azure code/conv、Mooncake）算 p50/p99 |
| 重建 Ray 集群 | 未知 | 计时 |

**要如实记录的一点**：档0 **必然丢弃**在途请求（客户端重试）。这是接受档0 的既定代价；
想不丢就得上档1（把已生成 token 拼回 prompt 重放），见 `docs/design/elasticity-cost-benefit.md` §5。

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
