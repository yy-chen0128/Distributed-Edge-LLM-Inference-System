# 四机互联手册：怎么连起来、需要你提供什么

> 目标：把 4 台笔记本连成**一条 vLLM 流水线**（PP=4，每台一段），前面挂我们的入口网关
> （档0 气泡重启 + 档1 拼接提示词重放）。
>
> 本文分三段：**① 你要提供什么** → **② 能不能连（网络前置实测）** → **③ 怎么启动、怎么验收、怎么演练节点脱离**。
> 单机部分见 [`vllm-lmcache-4node-plan.md`](vllm-lmcache-4node-plan.md)；每台机器的环境配置见
> [`environment-setup.md`](environment-setup.md)。

---

## ① 需要你提供什么

### 1.1 填这张表（每台一行）

| 字段 | 例 | 机器 A | 机器 B | 机器 C | 机器 D |
|---|---|---|---|---|---|
| 谁在用/怎么叫它 | `A`（head） | | | | |
| 操作系统 | Win11 + WSL2 / 原生 Ubuntu | | | | |
| **NVIDIA 驱动版本** | `566.24`（→ CUDA 12.7） | | | | |
| GPU 型号 | RTX 4060 Laptop | | | | |
| **显存** | 8 GB | | | | |
| **LAN IP**（不是 127.0.0.1） | `10.20.112.129` | | | | |
| 连接方式 | WiFi 5 / WiFi 6 / 有线 | | | | |
| 是否插电 | 是 | | | | |
| 磁盘剩余（系统盘） | 40 GB+ | | | | |
| **我能否 ssh 进这台** | 是/否 | | | | |
| 备注 | 校园网认证 | | | | |

> **显存这一列最重要**：它决定模型档位。7B fp16 均分 4 段时每段 7 层 = 3.26 GB，
> **4 GB 的卡会 OOM**，必须用 AWQ int4（每层约 0.13 GB，7 层 ≈ 0.9 GB）。

> **驱动版本这一列同样关键**（2026-09-23 实测踩到）：vLLM 0.30 与 LMCache 0.5.5
> 当前都会拉 **torch 2.13/2.14 + `cu130`** 的 wheel，而 **CUDA 大版本不向下兼容**：
> 本机驱动 566.24（CUDA 12.7）装完实测 `cuda available: False`，报
> *"The NVIDIA driver on your system is too old (found version 12070)"*。
> **四台机器要么都升到驱动 ≥580（支持 CUDA 13），要么统一用 torch 为 cu124/cu126 的旧组合**，
> 不能混。查法：`nvidia-smi` 右上角的 `CUDA Version`。

### 1.2 三项权限/环境确认（这三条不过，后面全白搭）

| # | 要确认什么 | 怎么确认 | 不过怎么办 |
|---|---|---|---|
| 1 | **4 台能互相 ping 通**（同一子网、无客户端隔离） | 在两台上互 `ping <对方IP>` | 校园网常开**客户端隔离**；换到同一台手机热点/路由器，或找实验室交换机 |
| 2 | **每台允许开放端口**（TCP 6379 / 8000 / 8100 + 一段 10002–10100） | 关掉或放行防火墙 | 关掉 Windows 防火墙的"公用网络"拦截，或加放行规则 |
| 3 | **每台的 WSL 用 mirrored 网络** | WSL 里 `ip -4 addr`，看地址是否**等于** Windows 的 LAN IP | 见 §2.3（默认 NAT 模式下别的机器连不进 WSL） |

### 1.4 【重要】异构分三种，性质完全不同

四台笔记本不可能完全一样。但"异构"不是一个东西，**三种异构对方案的影响方式完全不同**：

| 种类 | 例子 | 影响 | 能不能"利用" |
|---|---|---|---|
| **① 显存/算力异构** | 8 / 6 / 4 / 4 GB | 决定每台能放几层 | **能部分利用**：按能力加权切分（vLLM 不行——它均分；我们自研引擎可以） |
| **② GPU 架构异构** | 3060 = sm86，4060 = sm89 | 决定可用内核与量化格式（**FP8 只在 sm89+**；int4 AWQ 在 sm86/sm89 都行） | **不能利用，但也不阻塞**：各机自动选后端，性能不同 |
| **③ 驱动 / CUDA 版本异构** | 566.24(CUDA 12.7) vs 580+(13.x) | 决定**能不能装上同一套库** | **完全无法利用**，只能取"最低共同版本" |

#### ③ 的规则（实测得出，2026-09-23）

**硬规则：驱动报告的 CUDA 版本必须 ≥ 我们要装的 wheel 的 CUDA 版本。**（运行时不能比驱动新。）

| 我们要装的东西 | 需要的 CUDA | 对驱动的要求 |
|---|---|---|
| vLLM 0.8.5 → torch 2.6.0 | cu124 | 驱动报告 ≥ 12.4 |
| vLLM 0.9.2 / **0.10.0 / 0.10.1** | cu126 | 驱动报告 ≥ **12.6** |
| vLLM 0.10.2 → torch 2.8.0 | cu128 | 驱动报告 ≥ 12.8（≈570+） |
| vLLM 0.30 → torch 2.13/2.14 | cu130 | 驱动报告 ≥ **13.0**（≈580+） |

**所以：**
- **四台的驱动不必相同**，NCCL/CUDA 的跨机集合通信不要求驱动版本一致；
  但它们**必须都满足同一套 wheel 的下限**——而这套 wheel 是我们在四台上装**同一份**，不能一台一个版本。
- **实际约束由"最老的那台"决定**：如果三台是 580+、一台是 566.24，那么整个集群最高只能用 **cu126 档**
  （即 vLLM ≤ 0.10.1）；不能用 0.10.2/0.30。
- 反过来，**驱动异构会把集群的上限压到最低者**，而这一点**无法像显存异构那样"利用"**——
  它不是性能差异，而是"能不能装"的二元门槛。
- 这也是**自研引擎路线在异构机队上的一个隐藏优势**：它只要 torch 2.6+cu124（CUDA ≥12.4），
  对驱动的容忍度比 vLLM 当前版本高得多。

#### 开工前必须收齐的一条数据

```bash
# 每台都要跑，把 CUDA Version 报给我（右上角那个数）
nvidia-smi | head -4
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total \
    --format=csv,noheader
```

**收集后的决策规则**：取四台里 **CUDA 版本最低** 的那台，查上表，选定 vLLM 版本 → 四台装**同一个**版本。
若某台低到连 cu124 都不满足，只能**升级那台的驱动**（或把它排除出集群）。

> **同一台机器上，Windows 侧驱动决定 WSL 里的 CUDA 能力**——所以"升级驱动"是在 Windows 上做，
> 不是 `apt upgrade`。装完 `wsl --shutdown` 再进，`nvidia-smi` 的 CUDA Version 就会变。

- 每台的 **ssh 接入方式**（IP + 端口 + 用户名 + 私钥/密码），或者你选择**自己执行我给的一键脚本**；
- 每台的 **`collect_env.py` 输出**（脚本已在仓库里，产出 `env_<节点名>.json`）；
- **两两链路实测结果**（§2.2 的 JSON）。

---

## ② 能不能连：先做网络前置实测（约 30 分钟）

### 2.1 先说结论：**带宽从来不是问题，延迟才是**

PP 每生成一个 token，activation 要沿着 4 段依次过 3 跳。按模型的 hidden 维度算每 token 的跨机流量：

| 模型 | hidden | 每跳每 token（bf16） | 3 跳合计 | 20 tok/s 时的带宽 |
|---|---|---|---|---|
| Qwen2.5-0.5B | 896 | 1,792 B（我们实测过这个数） | 5.4 KB | **0.9 Mbit/s** |
| Qwen2.5-7B | 3584 | 7,168 B | 21.5 KB | **3.4 Mbit/s** |

连 5 Mbit/s 的弱 WiFi 都够。**真正的成本是延迟**：每个 token 串行过 3 跳，
每跳至少一个 RTT + 序列化。WiFi RTT 2–5 ms → 每 token 多 6–15 ms；
RTT 20 ms → 多 60 ms，那时 decode 就被网络主导了。

这也和已有研究一致：TPI-LLM 在 4 台笔记本上做张量并行的结论是
*"The bottleneck in allreduce is **not network bandwidth, but link latency**"*。

### 2.2 两两测链路（用仓库里的工具）

```bash
# A 机（服务端）
python -m edge_llm_scheduler.experiments.measure_link --serve --port 9200

# B 机（客户端）：测到 A 的 RTT 与单向吞吐
python -m edge_llm_scheduler.experiments.measure_link --host <A的IP> --port 9200 \
    --json-out .models/link_<A>_<B>.json
```

判定（沿用 `runbook-4-laptops.md` 的分档）：

| 实测 | 含义 | 能不能做 PP |
|---|---|---|
| 吞吐 ≥ 40 MB/s 且 RTT ≤ 3 ms | 有线千兆 / 好 WiFi | 放心做，甚至可以细切更多段 |
| 吞吐 8–40 MB/s 且 RTT 3–10 ms | 典型 WiFi 5/6 | 可行，但要盯 TPOT 退化 |
| 吞吐 < 5 MB/s 或 RTT > 30 ms | 弱链路/穿墙 | 仍可跑，但 decode 会很差；不要考虑 TP |

**另外要记两个数**：RTT 的 **p99** 与抖动（WiFi 会突刺），以及**丢包率**——
NCCL 对丢包比对我们自己的 TCP 敏感得多。

### 2.3 WSL 的网络模式（本机已确认的坑）

**默认 WSL2 是 NAT**：WSL 拿到的是 `172.x` 私网地址，**别的机器根本连不进来**，
Ray 集群和 vLLM 的跨机通信就起不来。本机实测已有的正确配置是 **mirrored**：

```ini
# C:\Users\<你>\.wslconfig
[wsl2]
networkingMode=mirrored
```

改完执行 `wsl --shutdown` 再重进。验证：WSL 里 `ip -4 addr` 的地址应**等于** Windows 的 LAN IP
（本机实测：Windows WLAN `10.20.112.129`，WSL 里 `eth3` 也是 `10.20.112.129` ✓）。

**副作用（必须知道）**：
- mirrored 模式下 Windows 与 WSL **共用端口**，所以冲突的端口要避让
  （本机 sshd 就因为 `Port 2222` 而非 22——见 `AGENTS.md`）；
- WSL 里的接口名不是 `eth0`（本机是 `eth3`，另有 6 个接口），
  **NCCL/GLOO 很可能挑错网卡** → 必须显式指定（见 §3.4）。

> 若某台机器的 Windows 版本不支持 mirrored（需 Win11 22H2+），退路是用
> `netsh portproxy` 把 6379/8000/8100 转发进 WSL；但 Ray 还要用一段 worker 端口，
> 转发很麻烦，所以**优先想办法开 mirrored**。

---

## ③ 每台机器的准备（在 environment-setup.md 的基础上补 vLLM 路线）

### 3.1 角色分配

| 角色 | 建议放哪台 | 为什么 |
|---|---|---|
| **Ray head + vLLM rank0** | 显存最大的那台 | head 要跑 GCS + 网关，CPU/内存占用最高 |
| Ray worker + rank1..3 | 其余三台 | — |
| **入口网关** | 与 head 同一台（默认） | 少一跳；也可以单独放一台 |
| LMCache | **每台本地** | 先拿本地 CPU 缓存的命中率；跨机共享作为第二步 |

### 3.2 每台要装什么（与自研引擎路线的差异）

| 项 | 自研引擎路线（旧） | **vLLM 路线（现在）** |
|---|---|---|
| Python 环境 | `~/venvs/pair`（torch 2.6.0+cu124） | **另建 `~/venvs/vllm`**（vLLM 0.30 会拉自己的 torch 2.13，别混装） |
| pip 源 | 默认 | **显式 https 源**：本机 `/etc/pip.conf` 指向 `http://mirrors.aliyun.com`（http + trusted-host 作用域不对）会让 pip 卡死；用 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| pip 缓存 | 默认（进 WSL ext4） | **`PIP_CACHE_DIR=/mnt/d/pipcache`**（ext4.vhdx 只增不减） |
| 模型 | 0.5B（`.models/Qwen2.5-0.5B-Instruct`） | 0.5B 先跑通 → 换 **AWQ int4 的 7B** |
| 额外依赖 | — | `ray`（vLLM 的 ray 后端）、`fastapi/uvicorn/httpx`（网关，只装在 head 那台也行） |
| 启动方式 | 起我们的 stage agent（TCP 9100+） | `ray start` + `vllm serve` |

**模型分发（4 台都要有同一份）——三种方案：**

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **各自下**（推荐先用） | 每台跑 `scripts/hf_mirror_download.py`（已修 UA 的 403 问题） | 简单、并行 | 4 份磁盘、4 次下载 |
| 一台下好再拷 | `rsync -a --info=progress2 .models/<模型> <对方>:~/pair/.models/` | 下 1 次 | 需要 ssh 通 |
| 共享盘（NFS/SMB） | 挂到同一个目录 | 只有一份 | **`/mnt/*` 读速很慢**（本机实测 `/mnt/d` 冷读 112 MiB/s vs ext4 2319 MiB/s，差 20 倍），加载会成为瓶颈 |

### 3.3 每台要开放的端口

| 端口 | 谁用 | 备注 |
|---|---|---|
| **6379** | Ray GCS（head） | 所有 worker 要能连 head |
| **10002–10100** | Ray worker 端口段 | 启动时用 `--min-worker-port/--max-worker-port` 收窄，便于放行 |
| **8000** | vLLM OpenAI API（head） | 网关与客户端连它 |
| **8100** | 我们的网关 | 对外入口 |
| 9200 | 链路测量（临时） | 测完可关 |
| 2222 | 本机 WSL 的 sshd | 与集群无关，是 agent 通道 |

> 最省事的做法：这 4 台在**同一个受控子网**里，防火墙对彼此**全放行**（都是自己人）。

### 3.4 通信相关环境变量（跨机 PP 的常见坑）

```bash
# NCCL/GLOO 必须显式绑定正确的网卡，否则多网卡机器（尤其 WSL）会挑错口而挂住
export NCCL_SOCKET_IFNAME=eth3          # 按各台 ip -4 addr 的实际名字改！
export GLOO_SOCKET_IFNAME=eth3
export NCCL_DEBUG=WARN                  # 出问题时临时改 INFO
export VLLM_HOST_IP=<本机 LAN IP>        # 让 vLLM 广播正确地址
export RAY_...                          # 如需指定，见 Ray 文档
```

（本机 WSL 里 WLAN 对应 `eth3`；**每台都要用 `ip -4 addr` 自己确认**，
因为接口名会随适配器数量变化。）

### 3.5 时钟

Ray 与日志对齐要求各机时间接近：`sudo systemctl enable --now systemd-timesyncd`。

---

## ④ 启动顺序（逐条命令）

```bash
# ---------- 每台：自检 ----------
/usr/lib/wsl/lib/nvidia-smi            # GPU 空闲、显存够
ls ~/venvs/vllm/bin/vllm               # vLLM 已装
ls -d .models/<模型>                    # 模型已就位
ip -4 addr | grep 'inet '              # 确认本机 LAN IP（mirrored 模式下 == Windows IP）

# ---------- head 那台 ----------
ray start --head --port=6379 --num-gpus=1 \
  --min-worker-port=10002 --max-worker-port=10100

# ---------- 其余三台 ----------
ray start --address=<head的IP>:6379 --num-gpus=1 \
  --min-worker-port=10002 --max-worker-port=10100

# ---------- 校验：head 上应看到 4 个节点 / 4 个 GPU ----------
ray status

# ---------- head：起 vLLM（PP=4，跨机） ----------
vllm serve .models/<模型> \
  --pipeline-parallel-size 4 \
  --distributed-executor-backend ray \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.55
# 若用 LMCache：
#   --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
#   并设置 LMCACHE_CONFIG_FILE

# ---------- head：起入口网关 ----------
python -m edge_llm_scheduler.gateway.vllm_gateway \
  --upstream http://127.0.0.1:8000 --port 8100
```

---

## ⑤ 验收清单（逐条打勾，缺一不可）

| # | 检查 | 期望 | 命令 |
|---|---|---|---|
| 1 | Ray 集群 | 4 nodes / 4 GPU | `ray status` |
| 2 | 模型服务 | HTTP 200 且列出模型 | `curl -s <head>:8000/v1/models` |
| 3 | **一次请求输出正确** | 正常中文句子 | `scripts/vllm_bench_prefix.py --base-url http://<head>:8000 --requests 1` |
| 4 | **每台显存占用符合预期** | AWQ 7B 每 rank 权重 ≈1.1GB + KV + activation | 各台 `nvidia-smi` |
| 5 | **前缀缓存命中** | 同前缀第 2 次 TTFT 明显更低 | `scripts/vllm_bench_prefix.py --requests 3` |
| 6 | **LMCache 跨重启是否命中**（决定档1 的气泡） | 记录 A/B/C/D 四段数字 | `scripts/wsl_vllm_smoke.sh` 的对应改造版 |
| 7 | 网关可用 | `/admin/status` 返回 available=true | `curl <head>:8100/admin/status` |
| 8 | **TPOT 跨机代价**（最关键的一个数） | 与 PP=1 单机对比 | 同一模型：单机 PP=1 跑一次，4 机 PP=4 跑一次，比较每 token 时间 |

> **第 8 条是整个四机方案成败的判据**：如果 4 机 PP=4 的每 token 时间**比单机 PP=1 还差**，
> 那"跨机"在这个网络上就不值得做（这也正是 llama.cpp RPC 第三方实测"RPC is for capacity, not speed"的含义）。

---

## ⑥ 节点脱离演练（档0 + 档1）

```
① 通过网关发一个长输出请求（例如 max_tokens=200），让它跑到一半
② 在某一台上 kill 掉 vLLM 进程（模拟"机器被抱走"）
   —— 更彻底的做法：直接合盖/断网，但那台就再也连不上了
③ 观察网关：/admin/status 应显示 available=false、reason 是连接错误
④ 幸存 3 台重建：
     ray stop && ray start --head/--address ...（3 台的列表）
     vllm serve --pipeline-parallel-size 3 ...
⑤ 网关：POST /admin/resume
⑥ 观察客户端：那条流是否延续、内容是否完整且不重复
⑦ 记录四个数（这就是"气泡"）：
     检测耗时 / 权重装载耗时 / Ray 重建耗时 / 重放 prefill 耗时
   以及 replay_diverged 是否为 0（重放保真度）
```

**注意**：PP 从 4 改成 3 需要改 `--pipeline-parallel-size` 并重启 vLLM——**目前是手动**。
自动化（网关触发重建脚本）是下一步；先把手动流程跑通、把气泡量出来。

---

## ⑦ 常见故障与排查

| 症状 | 可能原因 | 怎么办 |
|---|---|---|
| `ray status` 只看到 1 台 | worker 连不上 head：防火墙 / 不同子网 / 校园网隔离 | §1.2 的三条确认；先 `telnet <head> 6379` 试 |
| worker 连上又掉 | Ray worker 端口段被拦 | 收窄 `--min-worker-port/--max-worker-port` 并放行 |
| vLLM 起不来 / 卡在初始化 | NCCL 挑错网卡 | §3.4 的 `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` + `VLLM_HOST_IP` |
| **日志出现 `[c10d] The hostname of the client socket cannot be retrieved. err=-3`** | **mirrored 网络下 WSL 解析不了自己的主机名**；vLLM 的分布式初始化走 c10d/TCPStore，单机只是警告，**多机可能直接卡住或选错地址**（本机单机跑时已实测出现） | 显式指定 `MASTER_ADDR=<head 的 LAN IP>`、`VLLM_HOST_IP=<本机 LAN IP>`；必要时在 `/etc/hosts` 里手工加主机名映射 |
| **4GB 那台 OOM** | 7B fp16 均分 7 层 = 3.26GB | 换 **AWQ int4**；或把 PP 改成 2 并把小机器排除 |
| 请求很慢、每 token 几百 ms | 网络延迟主导 | 用 §2.2 的实测数字定位；考虑减段数、或改"多 token 流水" |
| WSL 里服务别的机器连不上 | 那台还是 NAT 模式 | §2.3：开 `networkingMode=mirrored` |
| 端口冲突（8000 起不来） | mirrored 模式下 Windows 占了同名端口 | `netstat -ano | findstr :8000` 找占用者并换端口 |
| 重启后 prefix cache 全丢 | 用的是 vLLM 内置缓存（进程内） | 用 LMCache，并确认 §5 第 6 条命中 |

---

## ⑧ 交接清单（我这边要收到什么才能开工）

1. §1.1 的**四行表格**（含每台 LAN IP 与显存）；
2. §1.2 的**三条确认**（互 ping / 端口 / mirrored）；
3. 每台的 `collect_env.py` 输出；
4. §2.2 的**两两链路 JSON**（至少 head 与其余三台之间）；
5. 每台是否允许我 ssh（否则我给你一键脚本，你逐台粘贴）。
