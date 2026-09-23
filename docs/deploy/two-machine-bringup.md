# 两台机器整合跑推理（先跑通，再量"跨机值不值"）

> 目标：用 **2 台机器 = 2 个真 GPU** 把 vLLM 的 PP=2 真正跑起来，并量出**跨机的每 token 代价**。
>
> 为什么两台特别重要：**vLLM 的 PP>1 在单卡上无法验证**（它的 PP 是按设备序号的 NCCL 进程组，
> 见 [`vllm-lmcache-4node-plan.md`](vllm-lmcache-4node-plan.md) §0.5 发现③）。两台机器各有 1 块卡，
> 正好满足"一 rank 一卡"，所以 **PP=2 的功能验证 + 跨机开销测量这两件事，两台就能做完**。

---

## 0. 先给我这几样（缺一样都开不了工）

| # | 需要 | 怎么拿 |
|---|---|---|
| 1 | **B 机的 LAN IP** | B 机 WSL 里 `ip route get 1.1.1.1 \| grep -oP 'src \K[0-9.]+'` |
| 2 | **B 机的 GPU / 显存 / 驱动 CUDA 版本** | B 机跑 `python scripts/check_node_readiness.py`（推荐），或 `nvidia-smi` |
| 3 | **B 机能不能被我直接操作** | 若可以：把 `scripts/two_node_this_machine_info.sh` 打印出的公钥加到 B 的 `~/.ssh/authorized_keys`；否则你在 B 上手动跑脚本 |
| 4 | **B 机 WSL 是不是 mirrored 网络** | 自检脚本会判定（不是的话别人连不进来，必须改） |
| 5 | **B 机有没有同一份模型** | 每个 PP rank 都从**本机磁盘**装自己那段，所以 B 必须有同一目录 |

**最快**：在 B 机上跑
```bash
python scripts/check_node_readiness.py --peer <A的IP> --json-out node_readiness_$(hostname).json
```
把 `### SEND THESE BACK TO THE CONTROLLER ###` 那段发我。

---

## 1. 两台能跑什么模型（按"两机各 8GB、可用 7.2GB"算）

| 模型 | PP=2 每段 | 每段显存 | 结论 |
|---|---|---|---|
| **0.5B fp16** | 12 层 | ~0.36 GB + 词表 0.27 | ✅ **先用它跑通**（最快暴露问题） |
| 3B fp16 | 18 层 | ~3.4 GB + 词表 0.55 | ✅ 可以 |
| **7B AWQ int4** | 14 层 | ~1.8 GB + 词表 1.09 | ✅ **真正的目标**（宽裕） |
| 7B fp16 | 14 层 | **~6.5 GB + 词表 1.09 = 7.6 GB** | ⚠️ **超过 7.2GB 可用**（勉强失败）→ 要么 PP=3（需要第三台），要么换 int4 |

> 注意词表成本只压在**首段和末段**（7B 未绑定词表 → 各约 1.09 GB），所以 PP=2 时两台的负载天然不等。

---

## 2. 开工前的四条硬前提（任何一条不满足都别往下走）

1. **两台能互 ping，0% 丢包**（校园 WiFi 常有客户端隔离 → 直接阻断集群）；
2. **两台都开了 `networkingMode=mirrored`**（否则别的机器连不进 WSL）；
3. **端口放行**：6379（Ray GCS）+ 10002–10100（Ray worker）+ 8000（vLLM）+ 8100（网关）；
4. **两台都能 `nvidia-smi` 且 `torch.cuda.is_available() == True`**（可用自检脚本确认）。

---

## 3. 启动顺序

### 3.1 先量链路（30 秒，别跳过）

```bash
# B 机
python -m edge_llm_scheduler.experiments.measure_link --serve --port 9200
# A 机
python -m edge_llm_scheduler.experiments.measure_link --host <B的IP> --port 9200 \
       --json-out .models/link_A_B.json
```
判读：吞吐 ≥40MB/s 且 RTT ≤3ms → 放心；8–40MB/s / 3–10ms → 可行但盯 TPOT；<5MB/s 或 >30ms → decode 会被网络主导。

> 提醒：PP 的**带宽需求极低**（7B 每 token 跨 1 跳 7.2KB；20 tok/s 下 <1.5 Mbit/s），
> **真正贵的是延迟**——每个 token 都要过这一跳。

### 3.2 跑两次，拿对比（一条命令）

```bash
# A 机（head）
WORKER_IP=<B的IP> bash scripts/two_node_head.sh
```
它自动做四件事：
1. **PP=1 单机基线**（本机跑同一模型）→ 记录 TPOT / tok/s；
2. 起 Ray head，提示（或经 ssh 自动）在 B 上跑 `HEAD_IP=<A的IP> bash scripts/two_node_worker.sh`；
3. 等 2 个节点加入 → 起 `vllm serve --pipeline-parallel-size 2 --distributed-executor-backend ray`；
4. 用同样的 prompt / 同样 token 数（`--ignore-eos`）再测一次，**打印两者对比**。

**脚本会自动做的事**（都是踩过的坑）：从"去对方的路由"里取接口名与源 IP，
导出 `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` / `VLLM_HOST_IP`——
WSL 有多个网卡会挑错；而 `VLLM_HOST_IP` 设错（比如 127.0.0.1）会让 Ray 的 placement group 永远无法满足。

### 3.3 看什么数字

| 指标 | 含义 |
|---|---|
`PP=1` 的 warm TPOT | 单卡基线（每 token 时间） |
`PP=2` 的 warm TPOT | 跨机后每 token 时间 |
**两者之差** | **跨机的每 token 代价** = 这次实验的核心产出 |
warm TTFT | 前缀缓存起作用后的首 token 时间 |
`ray status` 的节点/GPU 数 | 必须 2 nodes / 2 GPU，否则 PP=2 不会真跨机 |

**判据**：把"每 token 代价差"与"每 token 计算时间"比。若差值占主导 → 说明"单 token 依次跨机"
在这张网上不划算，需要改设计（减少跳数 / 加大 batch / 换切分）。**这就是"先量再建"的意义。**

---

## 4. 节点脱离演练（两台版，比四台更极端）

PP=2 时**丢掉任意一台就等于丢了一半流水线**，所以两台的容错更有意思：

| 场景 | 处置 | 结果 |
|---|---|---|
| **B 走了** | 停接纳 → drain → 在 A 上以 **PP=1** 重启（前提：A 装得下整个模型，0.5B/3B/int4-7B 都行） | 服务降级为单机，容量减半但可用 |
| **A（head）走了** | 只能在 B 上重建：起新 head + worker 角色互换 | 服务中断时间 = 重建时间 |
| 两台都在但链路抖 | 网关的 stall-timeout 触发重放（档1） | 请求保住，但重放可能改变答案（见 §0.5 发现②） |

演练命令（在 PP=2 已跑起来之后）：
```bash
# 网关（A 机）
PYTHONPATH=. nohup python -m edge_llm_scheduler.gateway.vllm_gateway \
    --upstream http://127.0.0.1:8000 --port 8100 --stall-timeout 10 &

# 客户端：400 token 流式请求，第 3 秒杀掉 B 的 vLLM（模拟机器离开）
python scripts/gateway_drill.py --gateway-url http://127.0.0.1:8100 \
    --model <模型> --max-tokens 400 --kill-after 3.0
```
PP=2 下的预期与单机不同：杀掉 B 会让**整个 PP 组失败**（不是只有一段），
所以这正好检验"网关能否撑住 + 重建后能否重放"。**这一步的观测值请记录下来**。

---

## 5. 要回传给我的数据

1. §3.1 的**链路 JSON**（RTT p50/p99、吞吐）；
2. §3.2 打印的**对比表**（PP=1 vs PP=2 的 warm TPOT / tok/s / TTFT）；
3. `ray status` 的 2 节点输出（确认真的跨机了）；
4. §4 演练的网关状态（`/admin/status`：failed / replayed / replay_exact / replay_diverged）；
5. 任何一次失败的**完整报错**（尤其含 `NCCL`、`c10d`、`placement group`、`invalid device` 字样）。

---

## 6. 相关文档

| 想知道 | 看 |
|---|---|
| 四机版手册（同源，多了 TPOT 成败判据与故障表） | [`four-machine-interconnect.md`](four-machine-interconnect.md) |
| vLLM 路线的硬约束、档0+档1、单机验证边界 | [`vllm-lmcache-4node-plan.md`](vllm-lmcache-4node-plan.md) |
| 每台机器的环境配置（含 vLLM 路线） | [`environment-setup.md`](environment-setup.md) |
| 自检脚本 | `scripts/check_node_readiness.py`、`scripts/fleet_plan_from_readiness.py` |
