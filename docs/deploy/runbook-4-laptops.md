# 四台笔记本端侧部署运行手册（真实分层流水线推理）

> 适用：4 张消费级笔记本 GPU（显存 4–8GB 级别）组成的异构小集群。
> 目标：把**一个模型按层切成 4 段**分别放到 4 台机器上，通过真实网络传递
> activation，跨机跑完一次完整推理，并能在**某台机器中途离开**后重新切层继续服务。
>
> 本文只讲怎么跑。为什么这样设计、能做什么/做不到什么，见
> [`edge-4gpu-deployment-analysis.md`](edge-4gpu-deployment-analysis.md)。

---

## 0. 一分钟版本

```powershell
# 每台笔记本（第 1 步，只做一次）：准备代码与模型
cd <project>
python scripts\hf_mirror_download.py --repo Qwen/Qwen2.5-0.5B-Instruct `
    --dest .models\Qwen2.5-0.5B-Instruct

# 每台笔记本（每次实验）：起 agent，node-id/端口各机不同
powershell -ExecutionPolicy Bypass -File edge_llm_scheduler\deploy\start_agent.ps1 `
    -NodeId alpha -Port 9100 -Model .models\Qwen2.5-0.5B-Instruct -Device cuda:0

# 控制端（任一台，先改 cluster.json 里的 IP）：跑一次跨机流水线
python -m edge_llm_scheduler.experiments.run_real_pipeline `
    --cluster edge_llm_scheduler\deploy\cluster.example.json --max-tokens 16

# 想在单机上先验证全流程（无需 4 台机器）：本机起 4 个 agent 进程走真实 TCP
python -m edge_llm_scheduler.experiments.run_real_pipeline --local 4 `
    --model .models/Qwen2.5-0.5B-Instruct --max-tokens 8 --concurrency 1,2,4
```

---

## 1. 环境准备（每台机器都做）

| 项 | 要求 | 说明 |
|---|---|---|
| Python | 3.10+ | 本机验证用 3.12.6 |
| PyTorch | CPU 可跑，GPU 需装 CUDA 版 | `pip install torch --index-url https://download.pytorch.org/whl/cu124`（按驱动选 cu124/cu126） |
| transformers | ≥ 4.4x（本机 5.9.0） | `pip install transformers accelerate safetensors` |
| 网络 | 同一局域网，agent 端口互通 | Windows 首次运行需放行入站；`start_agent` 绑 `0.0.0.0` |

**模型下载**：`scripts/hf_mirror_download.py` 走镜像直连（不依赖
`huggingface_hub` 的 etag 握手，校园网/镜像站下更稳）。四台机器各自准备
**同一份权重目录**——agent 只加载自己那段层，但仍需能读到完整 safetensors 文件。

```bash
# Linux / macOS 等价操作
python scripts/hf_mirror_download.py --repo Qwen/Qwen2.5-0.5B-Instruct \
    --dest .models/Qwen2.5-0.5B-Instruct
# 更大模型同理，例如 7B（约 15GB，建议先确认磁盘与带宽）
python scripts/hf_mirror_download.py --repo Qwen/Qwen2.5-7B-Instruct --dest .models/Qwen2.5-7B-Instruct
```

---

## 2. 先测链路（决定切层方案，别跳过）

在 A 机起服务端：

```bash
python -m edge_llm_scheduler.experiments.measure_link --serve --port 9200
```

在 B 机测 A→B（并查看它给出的 activation 传输耗时估算）：

```bash
python -m edge_llm_scheduler.experiments.measure_link --host <A机IP> --port 9200 \
    --payload-mb 128 --rtt-count 50 --json-out link_b2a.json
```

判读标准（我们的经验阈值）：

| 结果 | 含义 | 建议 |
|---|---|---|
| 吞吐 ≥ 40MB/s，RTT ≤ 3ms | 有线千兆/好 WiFi | 放心做 PP，甚至可试 4 段以上的细切 |
| 吞吐 8–40MB/s，RTT 3–10ms | 典型 WiFi 5/6 | PP 完全可行（activation 只有 KB 级） |
| 吞吐 < 5MB/s 或 RTT > 30ms | 弱链路/穿墙/跨网段 | PP 仍可行但 decode 会受 RTT 影响；不要考虑 TP |

记录每台机器两两之间的结果，填进后面的切层权重里。

---

## 3. 起 agent（每台机器）

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File edge_llm_scheduler\deploy\start_agent.ps1 `
    -NodeId alpha -Port 9100 -Model .models\Qwen2.5-0.5B-Instruct -Device cuda:0
```

```bash
# Linux / macOS
bash edge_llm_scheduler/deploy/start_agent.sh alpha 9100 .models/Qwen2.5-0.5B-Instruct cuda:0
```

常用参数：

| 参数 | 用途 |
|---|---|
| `-Device cuda:0` | 用本机 GPU；先试 `cpu` 确认能起，再换 GPU |
| `-Dtype auto` | CPU→float32；CUDA→bfloat16。显存紧张可强制 `float16` |
| `-WireDtype float16` | activation 传输精度（省一半带宽）；`none` 表示保持计算精度 |
| `-Threads 4` | 单机多 agent 时按核数分摊，避免线程超订（4 机部署时给 4–8） |

agent 起好后，控制端会先 `hello` 探查，回报真实设备、dtype、层数、显存总量。

---

## 4. 跑控制面

改 `edge_llm_scheduler/deploy/cluster.example.json` 里的 IP / node_id / weight：

```json
{
  "model_path": ".models/Qwen2.5-0.5B-Instruct",
  "nodes": [
    {"node_id": "alpha", "host": "192.168.1.11", "port": 9100, "weight": 3.0, "device": "cuda:0"},
    {"node_id": "beta",  "host": "192.168.1.12", "port": 9100, "weight": 2.0, "device": "cuda:0"},
    {"node_id": "gamma", "host": "192.168.1.13", "port": 9100, "weight": 1.0, "device": "cuda:0"},
    {"node_id": "delta", "host": "192.168.1.14", "port": 9100, "weight": 1.0, "device": "cpu"}
  ]
}
```

`weight` = 该节点应承担的**层数比例**。建议按实测"每层耗时"的倒数填（快机器拿更多层），
而不是简单按显存比例——注意**首段要额外装 embedding、末段要额外装 lm_head**，
两端本就该少分几层（见分析文档 §2.7）。

```bash
python -m edge_llm_scheduler.experiments.run_real_pipeline \
    --cluster edge_llm_scheduler/deploy/cluster.example.json \
    --prompt "用一句话解释什么是流水线并行。" --max-tokens 16 \
    --concurrency 1,2,4 --metrics-out metrics_4laptops.json
```

控制台会依次打印：集群盘点 → epoch 1 的切层表（含每节点 prepare 耗时）→
每个请求的各段 compute / queue / hop / activation 字节 → 并发实验吞吐。
指标同时写入 `--metrics-out` 指定的 JSON。

### 节点中途离开（论文里最有价值的那组实验）

```bash
# 跑 1 个请求后停掉 delta（或直接用 -LeaveNode 让控制面发 shutdown）
python -m edge_llm_scheduler.experiments.run_real_pipeline --cluster ... \
    --leave-node delta --leave-after 1 --max-tokens 16
```

输出会包含：剩余节点、**新 epoch 的切层与切换耗时**、离开后请求是否成功、
旧 epoch 的 drain 耗时。也可以手动在目标机器 `Ctrl+C` 掉 agent，或直接拔网线/
合盖睡眠——控制面会在下一次调用时报错，这正是要观察的恢复路径。

### 其他可做的故障注入

| 注入 | 做法 | 观察指标 |
|---|---|---|
| 节点突然掉线 | `kill` agent 进程 / 关 WiFi | 失败请求数、重新切层耗时、恢复后首 token 延迟 |
| 链路变差 | 路由器限速、远离 AP、下载大文件抢占 | hop_ms 上升、decode 每 token 延迟 |
| 显存压力 | 另一进程占用显存，或换更大的层区间 | `prepare_epoch` 是否 OOM、是否需要重切 |
| 慢节点 | 给某台 `-Device cpu` 或限制 `-Threads 1` | 该段成为瓶颈（bottleneck stage）的证据 |
| 计算异常 | 在 agent 侧抛异常（改一行） | 控制面报错路径、恢复策略是否生效 |

---

## 5. 单机验证（没有 4 台机器时也能全流程复现）

`--local N` 会在本机起 N 个 agent 进程，走**真实 TCP**（不是进程内直调），
因此协议、序列化、时序、指标口径与跨机完全一致，只有链路时延不同：

```bash
# 数值正确性：分段执行必须与整体执行逐 token 一致
python -m edge_llm_scheduler.experiments.verify_equivalence \
    --model .models/Qwen2.5-0.5B-Instruct --stages 4 --max-tokens 8 \
    --json-out .models/equiv_4stage.json

# 完整流水线 + 节点离开 + 并发
python -m edge_llm_scheduler.experiments.run_real_pipeline --local 4 \
    --model .models/Qwen2.5-0.5B-Instruct --max-tokens 8 \
    --concurrency 1,2,4 --leave-node n3 --metrics-out .models/real_pipeline_full.json

# 异构切层（模拟 4 台算力不同的机器）
python -m edge_llm_scheduler.experiments.run_real_pipeline --local 4 --weights 12,6,3,3 ...
```

---

## 6. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `ConnectionRefusedError` | agent 未起好/端口被占 | 等 2–3 秒；`netstat -ano \| findstr 9100` 查占用 |
| 首字符乱码（控制台） | Windows 控制台代码页非 UTF-8 | 指标 JSON 是 UTF-8，正常；或 `chcp 65001` |
| `CUDA out of memory` 于 `prepare_epoch` | 该段层数太多 | 减少该节点 weight；或 `-Dtype float16` |
| prepare 很慢（数秒~数十秒） | 从磁盘读 safetensors 并转 dtype | 首次慢正常；把模型放 SSD |
| 首段/末段显存比别人高 | 绑定的 embedding 要装两份（首段+末段） | 见分析文档 §2.3，按此调整 weight |
| 并发上不去 | 同一台机器跑多个 agent 抢 CPU | 4 台真机各一个 agent；单机模拟时调小 `-Threads` |
| 端口不通 | 防火墙 | Windows 放行入站 9100（及 9200 测量端口） |

---

## 7. 采集清单（跑完把这几样留下来）

1. 每台机器的 `nvidia-smi` 输出（型号、显存）。
2. 两两 `measure_link` 的 JSON（吞吐/RTT）。
3. `--metrics-out` 的完整 JSON（含 inventory / plans / requests / concurrency）。
4. `verify_equivalence` 的 JSON（证明分段数值正确）。
5. agent 日志（`.models/logs/agent_*.log`）。
6. 一句结论：这次实验的瓶颈在哪一段、为什么（用 `compute_ms` vs `hop_ms` 说话）。

这些正对应论文里的系统指标（延迟/吞吐/恢复时间）与状态指标（activation 字节、
显存占用、每段负载），见分析文档 §5。
