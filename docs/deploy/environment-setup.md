# 环境配置文档（每台笔记本各做一遍）

> 目标：把一台笔记本变成集群里的**一个算力节点**，能跑通"四段流水线推理"，
> 并能被控制端调度。
>
> 全流程约 **40–60 分钟**（不含模型下载）。步骤**按顺序执行**，每步都有验收标准；
> 某一步不通过时先停下来排查，不要跳到下一步。
>
> 适用：Windows 11 笔记本（推荐走 WSL2 + Ubuntu 24.04）或原生 Ubuntu 22.04/24.04。

---

## 0. 前置检查（2 分钟）

```powershell
nvidia-smi                                    # 必须有输出，记下 GPU 型号与显存
[System.IO.DriveInfo]::GetDrives() | Where-Object {$_.Name -eq 'C:\'} |
  ForEach-Object { "C: free = {0:N1} GB" -f ($_.AvailableFreeSpace/1GB) }
```

- **必须有 NVIDIA 显卡**（本项目的执行后端依赖 CUDA）。AMD/Intel 独显暂时不行。
- **C 盘至少留 40GB**（WSL 发行版 + venv + 依赖；模型放 D 盘）。
- 记下你的 GPU：后面 `collect_env.py` 会自动判定能用哪些后端/量化路径。

> **不要安装 CUDA Toolkit**（不需要 `nvcc`）。只需要 NVIDIA 驱动——PyTorch 的 wheel 自带 CUDA 运行时。

---

## 1. 系统：Windows + WSL2（推荐）

### 1.1 为什么用 WSL2

| 组件 | 能否在 Windows 原生跑 | 说明 |
|---|---|---|
| 我们自己的 stage runtime（主线实验） | ✅ 能 | 纯 PyTorch + socket，跨平台 |
| **vLLM / SGLang**（吞吐基线） | ❌ 不能 | 官方只支持 Linux |
| **LMCache**（KV 分层/压缩/迁移） | ❌ 不能 | C++/CUDA 扩展 + POSIX `fcntl` |

所以统一走 **WSL2**：一套环境三条路都能跑，也避免"Windows 上跑不了基线"的返工。

### 1.2 安装 WSL2 与 Ubuntu 24.04

管理员 PowerShell：

```powershell
wsl --install -d Ubuntu-24.04     # 已装过会提示，忽略
wsl --update
wsl --set-default-version 2
wsl -l -v                          # 确认 Ubuntu-24.04 的 VERSION 是 2
```

首次进入 Ubuntu 会让你设用户名/密码。进去后启用 systemd（部分功能需要）：

```bash
sudo tee /etc/wsl.conf >/dev/null <<'EOF'
[boot]
systemd=true
[user]
default=<你的WSL用户名>
EOF
```

### 1.3 `.wslconfig`（重要：防止 WSL 吃满内存与 C 盘）

在 **Windows** 上创建/改写 `C:\Users\<你的用户名>\.wslconfig`：

```ini
[wsl2]
networkingMode=mirrored
memory=10GB
processors=8
swap=8GB
swapFile=D:\\WSL\\swap.vhdx
sparseVhd=true
```

- `memory` 限制 WSL 占用（不设它会吃宿主一半内存）；
- `swapFile` 必须挪出 C 盘；
- `networkingMode=mirrored` 让**其他机器能直接访问 WSL 里的端口**（否则默认 NAT 模式别人连不进来），
  同时 Windows 的 `127.0.0.1` 能直达 WSL（后面自检要用）。

改完执行 `wsl --shutdown` 再重新进入，配置才生效。

> **可选但推荐**：把发行版整体放到 D 盘，避免 ext4.vhdx 撑大 C 盘：
> ```powershell
> wsl --shutdown
> wsl --manage Ubuntu-24.04 --move D:\WSL\Ubuntu-24.04
> wsl --export Ubuntu-24.04 D:\WSL\ubuntu2404-clean.tar   # 备份一份
> ```

---

## 2. GPU 与 Python 环境（在 WSL 里执行）

```bash
# 2.1 基础工具
sudo apt update
sudo apt install -y python3-venv python3-pip build-essential git curl jq

# 2.2 确认 GPU 直通（WSL 里不要装驱动的任何部分）
/usr/lib/wsl/lib/nvidia-smi            # 注意：非交互 shell 里 nvidia-smi 不在 PATH，用绝对路径
python3 -V                             # Ubuntu 24.04 自带 3.12

# 2.3 虚拟环境（放 WSL 原生盘，不要放 /mnt/*）
python3 -m venv ~/venvs/pair
source ~/venvs/pair/bin/activate
pip install -U pip

# 2.4 PyTorch（CUDA 版；按你的驱动选 cu124 或 cu126）
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2.5 其余依赖
pip install "transformers==5.9.0" accelerate safetensors numpy \
            pytest pytest-asyncio aiohttp psutil
```

**验收（三个都必须通过）**：

```bash
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：2.x.x+cu124 True <你的 GPU 名>

/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
# 记下 compute_cap，后面判断量化/后端用
```

> **GPU 架构能用到什么**（`scripts/collect_env.py` 会自动判定）：
>
> | compute_cap | attention | 量化 |
> |---|---|---|
> | 8.9（Ada，RTX 40 系） | FA2 + Triton + FlashInfer | FP8(CUTLASS/Marlin) + int4 + int8 |
> | 8.6（Ampere，RTX 30 系） | FA2 + Triton + FlashInfer | int4 + int8（**FP8 基本不可用**） |
> | 7.5（Turing，GTX 16 系） | **仅 Triton** | int4 Marlin + int8 |
> | ≥9.0（Hopper+） | FA3/FA4 | FP8/FP4 |

---

## 3. 获取代码与模型

```bash
# 3.1 克隆仓库（放到 WSL 原生盘，别放 /mnt/d，否则 IO 很慢）
cd ~
git clone git@github.com:yy-chen0128/Distributed-Edge-LLM-Inference-System.git pair-repo
cd pair-repo

# 3.2 下载模型（走 HF 镜像直连，脚本已内置重试）
python scripts/hf_mirror_download.py --repo Qwen/Qwen2.5-0.5B-Instruct \
       --dest ~/models/Qwen2.5-0.5B-Instruct
```

> **模型权重不要放进 WSL 的 ext4.vhdx**（vhdx 只增不减，会把 C 盘写满）。
> 放 `~/models` 时请确认它是 WSL 原生盘、或是指向 `/mnt/d` 的软链：`ln -s /mnt/d/models ~/models`

---

## 4. 起节点 agent 并自检

```bash
cd ~/pair-repo
source ~/venvs/pair/bin/activate

# 4.1 环境盘点（产出 env_<你的节点名>.json，后面交给控制端）
python scripts/collect_env.py --node-id <你的节点名> --port 9100

# 4.2 单元测试（应显示 86 passed 左右，少量 skip 是可选集成未装，正常）
python -m pytest edge_llm_scheduler/tests/ -q

# 4.3 单机四段流水线（真实 GPU 推理，验证执行链）
python -m edge_llm_scheduler.experiments.run_real_pipeline \
  --local 4 --device cuda:0 --model ~/models/Qwen2.5-0.5B-Instruct --max-tokens 6
# 期望：打印集群盘点 + 各段耗时，并输出一段中文

# 4.4 起 agent，等控制端来连
python -m edge_llm_scheduler.agents.stage_agent \
  --node-id <你的节点名> --port 9100 --host 0.0.0.0 \
  --model ~/models/Qwen2.5-0.5B-Instruct --device cuda:0 --wire-dtype float16
```

**网络**：
- 用 `networkingMode=mirrored` 时，WSL 里的监听端口在局域网直接可达；否则要在 Windows 侧做端口转发。
- 放行入站端口：`9100`（agent）、`9200`（链路测量）。
- 你的**局域网 IP**：`collect_env.py` 会打印；把它一起交回控制端（填进 `cluster.json`）。

---

## 5. 两两测链路（每个节点都要做）

```bash
# A 机：起服务端
python -m edge_llm_scheduler.experiments.measure_link --serve --port 9200

# B 机：测到 A
python -m edge_llm_scheduler.experiments.measure_link \
  --host <A机IP> --port 9200 --payload-mb 64 --json-out link_B2A.json
```

把 `link_*.json` 一起交回。判读参考：

| 吞吐 | RTT | 结论 |
|---|---|---|
| ≥40MB/s | ≤3ms | 有线千兆/好 WiFi，切层可细 |
| 8–40MB/s | 3–10ms | 典型 WiFi，PP 完全可行 |
| <5MB/s 或 >30ms | — | 弱链路：PP 仍可行，但 decode 会受 RTT 影响 |

---

## 6. 交回控制端的东西（清单）

1. `env_<节点名>.json`（`collect_env.py` 产出）
2. `link_*.json`（两两链路测量）
3. 一句结论：你的 GPU 型号/显存、`cuda.is_available()` 是否为 True、`pytest` 通过数
4. 你的 agent 局域网 IP 与端口

---

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `bash: nvidia-smi: command not found` | 非交互 shell 的 PATH 不含 WSL 库目录 | 用 `/usr/lib/wsl/lib/nvidia-smi`，或 `export PATH=$PATH:/usr/lib/wsl/lib` |
| `torch.cuda.is_available()` 为 False | 装成了 CPU 版 torch | 重装：`pip install torch --index-url .../cu124`（**先确认不是 +cpu 版本**） |
| `CUDA out of memory` | 分到本机的层太多 / 显存被别的程序占 | 减小该节点的切层数（见部署计划 §4.3）；关掉占显存的程序 |
| 别人连不上你的 agent | WSL 是 NAT 模式 或 防火墙 | 用 `networkingMode=mirrored`；放行入站 9100 |
| 中文输出乱码 | Windows 控制台是 GBK | 脚本里输出用英文；或 `chcp 65001` |
| **脚本报 `$'\r': command not found`** | 文件是 CRLF | 保存为 LF；或在远端 `tr -d '\r'` 过滤 |
| **脚本内容被"吞行"/路径变乱码** | 脚本里含中文注释或中文路径 | 远程脚本保持**纯 ASCII**；中文路径用 `find` 解析后软链成 ASCII 名 |
| `git clone` 卡住/报权限 | GitHub SSH 未配 | 用 `ssh -T git@github.com` 验证；或改用 HTTPS + token |
| 磁盘被 WSL 吃满 | ext4.vhdx 只增不减 | 模型放 `/mnt/d`；`wsl --manage <distro> --set-sparse true` |

---

## 8. 一页速查（复制即用）

```bash
# 环境
source ~/venvs/pair/bin/activate
export PATH=$PATH:/usr/lib/wsl/lib

# 盘点 / 测试 / 单机流水线 / 起 agent
python scripts/collect_env.py --node-id NODE --port 9100
python -m pytest edge_llm_scheduler/tests/ -q
python -m edge_llm_scheduler.experiments.run_real_pipeline --local 4 --device cuda:0 \
       --model ~/models/Qwen2.5-0.5B-Instruct --max-tokens 6
python -m edge_llm_scheduler.agents.stage_agent --node-id NODE --port 9100 \
       --host 0.0.0.0 --model ~/models/Qwen2.5-0.5B-Instruct --device cuda:0
```

> 更完整的部署计划、切层算法与实验设计见
> [`four-laptop-integration-plan.md`](four-laptop-integration-plan.md)；
> 项目结构与构建方式见 [`project-structure-and-build.md`](project-structure-and-build.md)。
