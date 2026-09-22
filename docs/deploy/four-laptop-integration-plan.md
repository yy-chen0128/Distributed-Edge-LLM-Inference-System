# 四台笔记本整合计划：环境、模型、构件与分阶段实施

> 目标：把四台消费级笔记本 GPU 整合成一个可运行的**异构跨机分层推理验证平台**，
> 用于验证"动态切层 + 状态迁移 + 弱网/异构自适应"这套机制。
>
> 本计划回答七件事：**① 分阶段计划 ② 有哪些优化构件 ③ 已有项目能复用什么
> ④ 四台机器各自配什么环境 ⑤ 要不要 WSL / Windows 是否不便 ⑥ GPU 环境怎么配 ⑦ 选什么模型**。
>
> 配套脚本（本计划新增，均可直接在各机运行）：
> `scripts/collect_env.py`（环境盘点）、`scripts/model_pp_fit.py`（模型切分可行性）、
> `scripts/hf_mirror_download.py`（镜像下载模型）、`scripts/model_sizing_table.py`（显存/KV 总表）。

---

## 0. 一句话定位（先说清楚不做什么）

这个平台的价值**不在算得快，而在够真实**：它能真实制造"**模型装不下、显存不够、链路慢、
节点随时走**"这四类约束。因此计划的目标是验证**容量扩展 + 弹性 + 吞吐**，
**不追求**单请求延迟优化（那需要 TP + 高速互联，我们的 WiFi 不成立）。

---

## 1. 现状盘点：已经有什么（本机已真实验证）

| 能力 | 状态 | 产物 |
|---|---|---|
| **真实 HF 分层 stage 运行时**（只加载本段层权重、真实 KV、真实采样） | ✅ 已跑通 | `backends/hf_layered_engine.py` |
| **数值正确性**：4 段 vs 整体执行 | ✅ 逐 token 一致，logits 最大绝对差 **0.0** | `experiments/verify_equivalence.py` |
| **节点 agent（独立进程 + TCP 控制/数据面）** | ✅ 4 进程真实 TCP 跑通 | `agents/stage_agent.py` |
| **控制器**：切层 → epoch → 请求 → 并发 → 节点离开重配置 | ✅ 含 4→3 节点重切（2.15s）与 drain（0.76s） | `experiments/run_real_pipeline.py` |
| **链路测量**（RTT/吞吐 + activation 耗时估算） | ✅ 工具就绪（loopback 基线 0.141ms / 165MB/s） | `experiments/measure_link.py` |
| **实测成本模型** | ✅ prefill ≈ 20ms+11.5ms×层；decode ≈ 5.1ms×层（CPU fp32） | 见分析文档 §3.4 |
| **测试** | ✅ 88 passed / 1 skipped | `edge_llm_scheduler/tests/` |
| **未做** | ⚠️ GPU 执行（本机 torch 是 CPU 版）、真实 WiFi 链路、真实 KV 跨机迁移、连续批处理、vLLM 基线 | — |

> 也就是说：**"机制骨架"已经在单机上真实验证过了**；剩下的是"上 GPU + 上四台机 + 补三类缺口"。

---

## 2. 环境决策：要不要 WSL？Windows 是否不便？

### 2.1 先分三条轨道，各自对 OS 的要求完全不同

| 轨道 | 内容 | OS 要求 | 结论 |
|---|---|---|---|
| **A. 自研 stage runtime**（主线） | 我们的 agent + HF 分层执行 + 自研 TCP 传输 | **三平台都行**（纯 PyTorch + socket，本机已在 Windows 原生跑通） | Windows 原生可用 |
| **B. vLLM / SGLang 基线** | 连续批处理/前缀缓存/PD 分离的对照引擎 | **必须 Linux**（vLLM 无 Windows 官方支持；SGLang 同） | 需 **WSL2** 或原生 Ubuntu |
| **C. LMCache** | KV 分层/压缩/跨机搬运 | **必须 Linux**：它的算子层是 C++/CUDA（`csrc/` 走 CMake），且用了 `fcntl.flock` 等 POSIX 接口 | 需 **WSL2** 或原生 Ubuntu |

**本机现状（实测）**：WSL **已安装且服务在跑**，已有发行版 `Ubuntu-24.04`、`Ubuntu-22.04`、
`Ubuntu`、`opp_env`、`wsl-vpnkit`；WSL 版本 2.7.3.0。**所以"要不要 WSL"这个问题在本机已经有答案：装好了，直接用。**

### 2.2 建议方案（按优先级）

1. **首选：主力节点用 WSL2（Ubuntu 24.04）**——B/C 两条轨道都能跑，
   A 轨道也能跑（WSL2 里跑 PyTorch + socket 完全没问题，且 Linux 下少了控制台编码/防火墙的麻烦）。
2. **次选：原生 Ubuntu 双系统**——如果某台机器磁盘宽松、且愿意折腾，
   原生 Linux 的 GPU 性能与稳定性更好（无 GPU-PV 虚拟化开销），多机 NCCL/调试也更顺。
3. **可以保留 Windows 原生**：只在"某台机器不跑 vLLM/LMCache、只跑我们的 agent"时用。
   **混合 OS 反而是好事**——它本身就是"异构"的一部分，我们的 TCP 协议不依赖 OS。

### 2.3 Windows 原生的具体不便（我们踩过的，不是传闻）

| 问题 | 表现 | 处理 |
|---|---|---|
| **控制台编码** | 中文/emoji 输出直接抛 `UnicodeEncodeError: 'gbk' codec` | 脚本里 `sys.stdout.reconfigure(encoding="utf-8")`（新脚本已内置） |
| **防火墙** | agent 端口从别的机器连不进来 | 首次运行放行入站（9100/9200/14318 等），或改用 WSL + mirrored 网络 |
| **线程超订** | 单机多 agent 时互相抢 CPU，吞吐不升反降 | `torch.set_num_threads(核数/agent数)`（`start_agent` 已支持 `--threads`） |
| **无 `fcntl`** | LMCache 的示例 connector 用 `fcntl.flock`，Windows 没有 | 别在 Windows 上跑 LMCache |
| **无官方 vLLM** | pip 上没有 Windows wheel | 走 WSL2 |
| **路径长度/中文路径** | 深路径 + 中文目录偶发问题 | 模型放短路径（如 `D:\models`） |

### 2.4 WSL2 的四个注意点（用之前必须知道）

| 注意点 | 说明 | 处理 |
|---|---|---|
| **驱动只装在 Windows 侧** | WSL 内**不要**再装 NVIDIA 驱动，用的是宿主驱动 + `/usr/lib/wsl/lib/libcuda.so` | 只在 Windows 装驱动；WSL 内 `nvidia-smi` 应能直接跑 |
| **内存上限默认是宿主的 ~50%** | 16GB 宿主 → WSL 默认只给 ~8GB，装 7B fp16 会 OOM | `%UserProfile%\.wslconfig` 里 `[wsl2] memory=12GB swap=8GB` |
| **网络默认 NAT，多机互联麻烦** | 别的机器访问不到 WSL 内的端口 | 用 Win11 的 **mirrored 网络模式**（`.wslconfig` 里 `networkingMode=mirrored`），或在 Windows 侧做端口转发 |
| **虚拟磁盘会膨胀** | `ext4.vhdx` 只增不减，模型反复下载会吃满盘 | 定期 `wsl --manage <distro> --set-sparse true` 或 `Optimize-VHD` |

> ⚠️ **磁盘是本机当前最紧的资源**：实测 C 24.6GB / D 29.9GB 可用。
> 7B fp16 权重 ~15GB × 每台一份 + WSL 的 vhdx + torch CUDA 轮子（~3GB）+ vLLM（~1GB），
> **一台机器会很紧张**。建议：**模型放外置 SSD 或让各机各下一份**，并先清理磁盘；
> 或者先用 1.5B/3B 跑通全流程，7B 只在一两台上下载。

---

## 3. GPU 环境怎么配

### 3.1 装什么、不装什么

| 组件 | 装吗 | 说明 |
|---|---|---|
| **NVIDIA 驱动** | ✅ 必装（Windows 侧；WSL 内不装） | 本机 566.24，支持 CUDA 12.6 级 |
| **CUDA Toolkit（nvcc）** | ❌ 通常不需要 | 只有**从源码编译** vLLM/LMCache 的 C++/CUDA 扩展才需要 |
| **PyTorch（CUDA 版）** | ✅ 必装 | `pip install torch --index-url https://download.pytorch.org/whl/cu124`（按驱动选 cu124 / cu126）。**本机当前是 `2.13.0+cpu`，必须换** |
| transformers / accelerate / safetensors | ✅ | `pip install transformers accelerate safetensors` |
| **vLLM** | ✅（轨道 B） | **优先用预编译 wheel**：`pip install vllm`；从源码装时 `VLLM_USE_PRECOMPILED=1` 跳过编译 |
| **LMCache** | ✅（轨道 C，Linux/WSL） | `pip install -e . --no-build-isolation`（先装 torch）；`NO_CUDA_EXT=1` 可只用 Python 版（**会失去快 kernel**） |

### 3.2 验证顺序（每台机器都跑一遍）

```bash
nvidia-smi                      # 驱动与显存
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python scripts/collect_env.py --node-id alpha --port 9100     # 一次性盘点，产出 env_alpha.json
```

`collect_env.py` 会自动给出**这台机器能用的 attention 后端与量化路径**，判定依据是 vLLM 的架构门槛：

| 计算能力 | attention | 量化 | 典型卡 |
|---|---|---|---|
| **8.9（Ada）** | FA2 + Triton + FlashInfer（无 sink） | **FP8(CUTLASS/Marlin) + int4(W4A16) + int8** | RTX 4060 Laptop（本机） |
| 8.6（Ampere） | FA2 + Triton + FlashInfer | int4(W4A16/W8A16) + int8；**FP8 基本不可用** | RTX 3060 |
| 7.5（Turing） | **仅 Triton**（FA2/FlashInfer 需 ≥8.0） | int4 Marlin(≥75) + int8 | GTX 1650 |
| ≥9.0（Hopper+） | FA3/FA4 | FP8/FP4 | 不在我们范围 |

> **异构混部注意**：混合架构下，**每个节点各自挑后端**（vLLM 按 `supports_compute_capability` 自动选）；
> 若要从源码编译任何扩展，用 `TORCH_CUDA_ARCH_LIST` 指定各自架构（例如 `8.6;8.9`）。

### 3.3 建议的统一 dtype

- Ada/Ampere（≥8.0）：**bfloat16**（权重/KV/激活）
- Turing（7.5）：**float16**
- 跨机 activation 传输：**float16**（体积减半，`--wire-dtype float16` 已是默认）

---

## 4. 选什么模型合适

### 4.1 三层阶梯（按"用途"选，不按"越大越好"）

| 层 | 模型 | 用途 | 为什么选它 |
|---|---|---|---|
| **校准层** | **Qwen2.5-0.5B-Instruct**（已下载） | 数值正确性、CI、协议回归 | 24 层、**绑定词表**（正好覆盖"末端要装两份词表"这个坑）；CPU 也能跑 |
| **迭代层** | **Qwen2.5-1.5B-Instruct**（28 层）或 **3B**（36 层） | 日常调策略/改代码 | 层数够多（切 4 段后 7–9 层/段）、单层权重小、GPU 上很快 |
| **旗舰层** | **Qwen2.5-7B-Instruct（fp16）** | **容量扩展的主实验**（单机 8GB 装不下） | 见 §4.2：它**平均切装不下、加权切装得下**，正好验证我们的能力加权切层 |
| **大模型层** | **Qwen2.5-14B-Instruct（int4）** | "大模型跨机" | fp16 需 29.5GB（超预算），int4 约 8.1GB → 四机可容 |
| **MoE 层** | **Qwen1.5-MoE-A2.7B-Chat**（60 专家，激活 2.7B） | 专家并行/命中率实验 | 单机放不下全部专家，天然逼出 EP；见 §4.4 注意点 |
| **对照** | **Mistral-7B-Instruct-v0.3** | PP 的"低固定开销"对照 | vocab 仅 32k → 两端固定开销只有 Qwen 的 1/4 |

### 4.2 真实数字（`model_pp_fit.py` 实测，按 8+6+4+4GB 预算、4 段平均切）

| 模型 | 层数 | 每层 fp16 | 词表 | 绑定? | KV/token | 4 段平均切 |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 24 | 0.030GB | 0.27GB | **是** | 12KB | ✅ |
| Qwen2.5-1.5B | 28 | 0.094GB | 0.47GB | **是** | 28KB | ✅ |
| Qwen2.5-3B | 36 | 0.154GB | 0.62GB | **是** | 36KB | ✅ |
| **Qwen2.5-7B** | 28 | 0.466GB | **1.09GB** | **否** | 56KB | ❌（4GB 卡装不下 7 层） |
| Qwen2.5-14B | 48 | 0.551GB | 1.56GB | 否 | 192KB | ❌（需 int4） |
| Mistral-7B-v0.3 | 32 | 0.436GB | **0.27GB** | 否 | 128KB | ❌（平均切同样不行） |
| Qwen1.5-MoE-A2.7B | 24 | 0.051GB（**单专家口径**） | 0.62GB | 否 | 192KB | ✅（但见 §4.4） |

### 4.3 关键算例：同一模型"平均切失败、加权切成功"

Qwen2.5-7B fp16 在 8/6/4/4GB 上的预算（每机留 0.8GB 给激活/上下文/碎片）：

```
总权重 28 × 0.466GB = 13.05GB
两端固定：首段 embedding 1.09GB ＋ 末段 lm_head 1.09GB（该模型未绑定词表，是独立权重）
KV：56KB/token → 4k 上下文约 0.23GB/机

可用预算：8-0.8=7.2 / 6-0.8=5.2 / 4-0.8=3.2 / 4-0.8=3.2 GB

① 平均切（7/7/7/7）：每段 7×0.466 = 3.26GB 权重
   4GB 机上 3.26 + KV 0.23 = 3.49GB  >  3.2GB   →  【装不下】

② 能力加权切（12 / 4 / 4 / 8，把带 lm_head 的末段放到 6GB 机）：
   8GB 机（首段）：12×0.466 + embed 1.09 + KV 0.23 = 6.91GB  ≤ 7.2GB  ✅
   4GB 机：        4×0.466 + KV 0.23               = 2.09GB  ≤ 3.2GB  ✅
   4GB 机：        4×0.466 + KV 0.23               = 2.09GB  ≤ 3.2GB  ✅
   6GB 机（末段）：8×0.466 + lm_head 1.09 + KV 0.23 = 5.05GB  ≤ 5.2GB  ✅
   （层数合计 12+4+4+8 = 28 ✅）
```

**这个算例里有两条可直接写进论文的观察**：
1. **平均切会失败、能力加权切能成功** —— 这就是 `CapabilityReparallelization` 的第一个真实用武之地，
   而且自带对照组（不需要额外构造 baseline）。
2. **最优切分对显存并非单调**：末段要额外背一份 1.09GB 的 lm_head，
   所以**把末段放在显存更大的机器上反而更优**（示例里 6GB 机拿 8 层，4GB 机只拿 4 层）。
   同理首段要背 embedding。→ **切层求解器必须把"两端固定开销"显式建模**，不能只按显存比例。

把这些抄成人话：**"两端是贵的，中间是便宜的"**，这正好也解释了真实系统（Megatron 类）为什么
总把 embedding 和输出层单独对待。

### 4.4 三条选型经验（都是算出来的，不是猜的）

1. **优先选词表小的模型**：PP 下首段要装 embedding、末段要装 lm_head，
   **未绑定词表时两端各背一份**。Qwen 的 152k 词表 = 每端 1.09GB；Mistral 的 32k 只有 0.27GB。
   → 同等参数量下，**Mistral 系在 PP 里比 Qwen 系省 ~1.6GB**。
2. **注意"绑定/未绑定"**：Qwen2.5-**0.5B/1.5B/3B 绑定**（末段要再装一份词表当输出头），
   **7B/14B 不绑定**（末段是独立 lm_head）。这直接改变两端该分几层。
3. **MoE 的坑**：`model_pp_fit.py` 对 MoE 只按"**单个专家**"算每层权重
   （Qwen1.5-MoE 每层 60 个专家），真实总量要 **×本地专家数**；
   所以 MoE 的选型必须用"专家并行(EP)"的视角重算——这正是 MoE 实验要验证的东西。

### 4.5 怎么下载（每台各下一份同一目录）

```bash
python scripts/hf_mirror_download.py --repo Qwen/Qwen2.5-7B-Instruct --dest .models/Qwen2.5-7B-Instruct
```

> 走镜像直连（不依赖 `huggingface_hub` 的 etag 握手）。**每台机器都要有同一份权重目录**——
> agent 只加载自己那段层，但仍需能读到完整 safetensors 文件。

---

## 5. 优化构件清单与复用情况

### 5.1 我们已有（自研，直接可用）

| 构件 | 位置 | 状态 |
|---|---|---|
| 真实分层 stage 运行时（只载本段层） | `backends/hf_layered_engine.py` | ✅ |
| 节点 agent（TCP 控制+数据面，长连接、可按层做准备/激活/退役） | `agents/stage_agent.py` | ✅ |
| 控制器（切层计划 / epoch / 请求 / 并发 / 节点离开重配置） | `experiments/run_real_pipeline.py` | ✅ |
| 数值一致性验证 | `experiments/verify_equivalence.py` | ✅ |
| 链路测量（RTT/吞吐 + activation 估算表） | `experiments/measure_link.py` | ✅ |
| 环境盘点 / 模型切分可行性 / 镜像下载 | `scripts/collect_env.py`、`model_pp_fit.py`、`hf_mirror_download.py` | ✅ |
| 骨架机制：节点/模型账本/KV 目录/epoch 重配置/迁移与恢复策略 | `core/`、`policies/` | ✅（无 GPU 验证） |
| 三平台混部 | `deploy/start_agent.ps1` / `.sh` | ✅ |

### 5.2 可复用的外部构件（Linux/WSL）

| 构件 | 来源 | 复用方式 | 注意 |
|---|---|---|---|
| 连续批处理 + 前缀缓存 | vLLM | 配置级直接用 | 作为**吞吐 baseline**；我们的 agent 不提供批处理 |
| KV 逐层加载/保存钩子 | vLLM `KVConnectorBase_V1` | 自写 connector（`kv_connector_module_path`，免 fork） | block 对齐；HMA 需声明 |
| KV 加载失败策略 | vLLM `kv_load_failure_policy` | 配置级 | 只有 `recompute|fail` 二选一 |
| KV 分层存储/压缩 | LMCache | 层粒度键 + `storage_manager.batched_*` + CacheGen 压缩 | **只能进程内嵌库**；上层 API 只认"全层命中" |
| 跨机 KV 字节通道 | LMCache `lmcache_server` + `remote_url="lm://"` | 纯 TCP、CPU-only、零硬件前提 | 需要 `PYTHONHASHSEED=0` |
| 专家重排（运行中） | vLLM EPLB `rearrange` + NIXL 通信后端 | 配置 + 调用 | elastic EP 与 PP 互斥 |
| 量化 | vLLM（int4 Marlin / FP8） | 配置级 | 按卡分档（见 §3.2） |
| 投机解码 | vLLM（ngram/MTP/EAGLE…） | 配置级 | 需对应 draft head |
| 专家 offload 到 CPU | vLLM `OffloadConfig`（UVA/Prefetch，可按 `experts` 匹配） | 配置级 | **节点内**，不跨设备 |

### 5.3 必须自研的缺口（按优先级）

| 缺口 | 为什么必须自研 | 优先级 |
|---|---|---|
| **真实 KV 跨机迁移**（层粒度 + 部分迁移 + 预算） | vLLM 只有二选一开关；LMCache 上层 API 不认部分层 | P1 |
| **连续批处理 / micro-batch**（把并发从 1.55× 拉起来） | 我们的 agent 是一段一次一个请求 | P1 |
| **权重装载加速 / P2P 权重分发** | 实测装载 2–8s 是重配置主要成本；vLLM 无节点间 P2P | P1 |
| **能力加权切层求解器**（含两端固定开销） | 现成的是平均切；已证明 7B 平均切会失败 | P1 |
| **P2P 权重分发 + 加入时在线能力评估** | gap #1（运行中评估新节点） | P2 |
| **KV 稀疏/驱逐**（与迁移量联合） | vLLM 无 token 级驱逐 | P2 |
| **PD 分离 + KV 传输** | 与 PP 正交，容易出图 | P2 |
| **能效/热约束调度** | 笔记本独有（降频/过热） | P3 |
| **观测统一化**（每段/每迁移入库） | 论文图表自动化 | P2 |

---

## 6. 分阶段实施计划（含验收标准）

### P0 环境与联通（0.5–1 天）
- 每台跑 `collect_env.py` → 收集 4 份 `env_*.json`；填 §7 的角色表
- 每台装 CUDA 版 PyTorch + transformers；**验证 `torch.cuda.is_available()==True`**
- 决策 OS：主力节点用 WSL2/Ubuntu；确认 `.wslconfig` 内存与 mirrored 网络（若用 WSL）
- **两两跑 `measure_link.py`**（一次有线、一次 WiFi），得到真实 RTT/吞吐表
- **验收**：4 份 env JSON；两两链路表；`nvidia-smi` 与 torch CUDA 均正常

### P1 单机多进程（GPU 版，1 天）
- `run_real_pipeline.py --local 4 --device cuda:0`（4 个 agent 进程共享一张卡，只为验证 GPU 路径）
- 再跑 `--local 4 --device cpu` 做对照，确认 GPU 路径正确
- **验收**：GPU 上 `verify_equivalence` 仍逐 token 一致；得到 GPU 上的成本模型（§3.4 那套系数）

### P2 四机真实 PP（2–3 天）
- 每台 `start_agent`（各自 GPU、各自 wire-dtype、`--threads` 按核数分摊）
- 控制端改 `cluster.json`（IP 来自 env JSON），先跑 **1.5B/3B 平均切**，再跑 **7B 加权切**
- 加入 `--concurrency 1,2,4` 扫描
- **验收**：7B fp16 四机跑通（证明"加权切成功"）；记录每段 compute/queue/hop、activation 字节、吞吐

### P3 弹性与迁移（3–5 天）
- `--leave-node` 触发重切；把 `PriorityMigration` 接到**真实 KV**（先 MockKVStore，再接 LMCache）
- 加"节点加入"路径：新机器起 agent → 控制器在线评估 → 新 epoch
- **验收**：离开后恢复时间、迁移字节、重算成本三条曲线；加入后吞吐变化

### P4 vLLM 基线对照（2–3 天，Linux/WSL）
- 单机 vLLM 跑同一模型（连续批处理 + 前缀缓存），记录吞吐/延迟
- 配 LMCache connector，验证 KV offload 与 `lm://` 跨机通道
- **验收**：基线表（vLLM 单机 vs 我们四机 PP vs 四机 DP 路由）

### P5 策略与论文实验（持续）
- 能力加权切层求解器（用 §3.4 实测系数）
- 稀疏/驱逐 × 迁移预算联合
- 能效-热约束调度
- **验收**：baseline + 消融 + 节点 churn/带宽变化下的完整结果

---

## 7. 角色分配表（按 `collect_env.py` 的实测值填）

| 机器 | GPU / 显存 | 计算能力 | 角色建议 | dtype | 建议层数（7B fp16 为例，见 §4.3） |
|---|---|---|---|---|---|
| alpha（本机实测） | RTX 4060 Laptop 8GB | 8.9 | **首段**（含 embedding）+ 最多层 | bf16 | **12** |
| beta |  |  | 中间段（按每层耗时倒数加权） |  | 4 |
| gamma |  |  | 中间段 / 弱段 |  | 4 |
| delta |  |  | **末段**（含 lm_head，建议放显存较大的那台） |  | 8（若 delta 是 6GB） |

> 切层权重不要按显存比例填，按**"每层耗时"的倒数**填（用 P1 的实测系数），
> 并**先扣掉两端的固定开销**（首段 embed、末段 lm_head）——
> 注意示例中"末段放在 6GB 机"比"放在 4GB 机"能多拿 4 层，因为 lm_head 本身要占 1.09GB。

---

## 8. 风险与坑清单

| 风险 | 现象 | 处理 |
|---|---|---|
| **磁盘不足** | C 24.6GB / D 29.9GB，7B 权重 15GB | 模型放外置 SSD；各机各下一份；先跑小模型 |
| **WSL 内存上限** | 7B fp16 在 WSL 里 OOM | `.wslconfig` 提 `memory`；或原生 Linux |
| **WSL 网络 NAT** | 别的机器连不进 agent | mirrored 网络模式 或 Windows 侧端口转发 |
| **WiFi 省电/漫游** | 链路抖动、吞吐骤降 | 插电、关省电、固定频段；用 `measure_link` 量化；把它当作实验变量而非故障 |
| **合盖休眠** | 节点"无声"离开（相当于免费故障注入） | 设置不休眠；用它做节点离开实验 |
| **架构差异** | 某些后端在某台机器不可用 | `collect_env` 的判定表；每节点各自选后端 |
| **模型目录不一致** | 各机权重版本不同 → 数值不一致 | 固定 commit/文件哈希，用 `hf_mirror_download.py` 统一 |
| **KV 布局耦合** | 跨机搬 KV 时布局不匹配 | 先统一 block_size 与 dtype；参考 LMCache 的 `GPUKVFormat` 做法 |
| **线程超订** | 单机多 agent 吞吐不升 | `--threads` 分摊 |
| **时钟/时区** | 事件时间线错乱 | 各机开 NTP；指标里用相对时间 |

---

## 9. 每台机器的一次性清单

```text
[ ] 1. 系统：Windows+WSL2(Ubuntu24.04) 或原生 Ubuntu；.wslconfig 调内存/网络
[ ] 2. 驱动：Windows 侧 NVIDIA 驱动（WSL 内不装）
[ ] 3. Python 3.10–3.12；pip install torch --index-url .../cu124
[ ] 4. pip install transformers accelerate safetensors numpy
[ ] 5. 验证：python -c "import torch;print(torch.cuda.is_available())"  → True
[ ] 6. python scripts/collect_env.py --node-id <name> --port 9100  → 产出 env_<name>.json
[ ] 7. 下载模型：python scripts/hf_mirror_download.py --repo ... --dest .models/...
[ ] 8. 放行入站端口：9100（agent）、9200（链路测量）；关休眠/省电
[ ] 9. 起 agent：deploy/start_agent.ps1|.sh（device=cuda:0, wire-dtype=float16, threads=核数）
[ ] 10. 两两测链路：measure_link --host <对方IP> --port 9200
[ ] 11. 把 env JSON + 链路结果交给控制端，更新 cluster.json
```
