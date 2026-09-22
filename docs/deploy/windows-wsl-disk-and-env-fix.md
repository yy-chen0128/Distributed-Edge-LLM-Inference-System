# Windows/WSL 环境整理：torch 装在哪、C 盘被谁吃了、pagefile 怎么改

> 场景：4 台笔记本要装 GPU torch + vLLM/LMCache，但 **C 盘只剩 22.9GB、D 盘只剩 27.8GB**。
> 本文基于本机（alpha）实测诊断，给出**可直接照做的步骤**与**正确执行顺序**。
> 所有需要管理员权限的命令都标注了 **[管理员]**；沙箱里我无法代跑，请自行执行。

---

## 1. 先回答：CPU 的 torch 在哪？GPU 的 torch 该装哪？

### 1.1 现状（实测）

```
python  : C:\Users\12032\AppData\Local\Programs\Python\Python312\python.exe
torch   : C:\Users\12032\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\__init__.py
version : 2.13.0+cpu        ← 纯 CPU 版，cuda_available=False
```

**结论：那个 CPU torch 在 Windows 侧**（Windows 自带的 Python 3.12）。我前面跑的所有测试、
脚本、验证，用的都是这个 Windows Python——**WSL 里根本没装 torch**。

### 1.2 Windows Python 与 WSL Python 是两套完全独立的环境

| | Windows 侧 | WSL 侧 |
|---|---|---|
| 解释器 | `C:\...\Programs\Python\Python312\python.exe` | `/usr/bin/python3`（Ubuntu 自带） |
| 包目录 | `...\Python312\Lib\site-packages` | `/usr/lib/python3/dist-packages`、`~/.local/lib/...`、venv 里 |
| 文件系统 | `C:\` / `D:\` | 虚拟磁盘 `ext4.vhdx` 内部（映射到 `/mnt/c`、`/mnt/d` 访问 Windows 盘） |
| 在一边装包 | **另一边看不到** | **另一边看不到** |

**所以在哪里装 GPU torch，取决于你要在哪里跑：**

| 你要跑的 | 装在哪 | 命令 |
|---|---|---|
| 自研 agent / 脚本（Windows 原生） | Windows Python | `python -m pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| vLLM / SGLang / LMCache（**只能 Linux**） | **WSL 的 Python** | 进 WSL：`python3 -m pip install torch --index-url .../cu124` 然后 `pip install vllm` |
| 两边都想跑 | 两边各装一份 | 注意**磁盘**：CUDA 版 torch 安装后约 7GB/份 |

**建议**：
- **主力装在 WSL**（vLLM/LMCache 只有这里能跑，且 Linux 下没有控制台编码/防火墙那些麻烦）；
- Windows 侧**也装一份 GPU torch**，方便快速跑我们的 agent 和脚本（避免 `D:\` 经 `/mnt/d` 访问时的 I/O 慢）；
- **不要装 CUDA Toolkit（nvcc）**：跑 PyTorch 不需要；只有从源码编译 vLLM/LMCache 的
  C++/CUDA 扩展才需要（vLLM 用预编译 wheel、LMCache 用 `NO_CUDA_EXT=1` 都能绕开）。
- 换 torch 前先卸载 CPU 版：`python -m pip uninstall -y torch`（避免残留混装）。

### 1.3 一个坑：仓库在 D:\ 上，在 WSL 里访问会慢

WSL 通过 `/mnt/d/...` 访问 Windows 盘是走 9P 协议，**小文件随机 I/O 很慢**
（Python 导入、pytest 收集、模型 safetensors 逐块读都会受影响）。
两个选择：
- **省事**：代码放 `/mnt/d/...` 直接用（可接受，但 pytorch 的 import 会明显变慢）；
- **快**：把代码 `cp -r` 到 WSL 内部（如 `~/work/project`），Windows 侧改动用 `git` 或 `rsync` 同步。
  模型权重则相反：**放在 Windows 的 D 盘**（`/mnt/d/models`），因为 WSL 内部磁盘在本机 C 盘上、且会膨胀。

---

## 2. C 盘到底被谁吃了（实测诊断）

C 盘 300GB 总 / **22.9GB 可用**（已用约 277GB）。逐项实测（**完整账目**）：

| 占用 | 大小 | 可回收性 |
|---|---|---|
| **Ubuntu-24.04 的 ext4.vhdx** | **69.3 GB** | ⭐ 最大单项：压缩可回收约 14GB；搬走可回收全部 |
| `AppData\Local`（不含 wsl） | 28.0 GB | `Temp` **6.2GB 可清**；`Programs` 6.6GB（Python 等，别动）；`Microsoft` 4.5GB（缓存可清） |
| `Program Files` | 31.6 GB | 已装软件，按需卸载 |
| `Program Files (x86)` | 23.8 GB | 同上 |
| `AppData\Roaming` | 18.3 GB | `Code` 3.1GB（VS Code 缓存可清）、kingsoft 3.0、Tencent 2.8、miHoYo 1.6、baidu 1.4、ClassIn 1.1 |
| `ProgramData` | 13.0 GB | `Microsoft` 4.6GB（Defender 等，别动）、`Lenovo` 4.4GB（驱动包可清）、`NVIDIA` 1.5GB |
| `AppData\LocalLow` | 8.9 GB | `NVIDIA` **3.7GB 着色器缓存可直接删**；游戏缓存 Owlcat 2.6 / Ludeon 1.6 / miHoYo 2.1 |
| **Ubuntu（老）ext4.vhdx** | 5.3 GB | 不用就 export 后注销 |
| **Ubuntu-22.04 的 ext4.vhdx** | 4.0 GB | 同上 |
| 用户目录 Documents/Downloads/桌面/图片 | 4.3 GB | 自行判断 |
| wsl-vpnkit | 0.1 GB | 别动（VPN 依赖） |
| **小计** | **≈206 GB** | |
| `C:\Windows` + 其它根目录 | ≈71 GB（推算） | WinSxS 别手动删 |

> ⚠️ **更正一个我先前给错的估计**：我原以为 69.3GB 里大部分是"膨胀空间"、压缩能回收 20–50GB。
> 实测 `df` 显示**里面真实数据 55GB**，所以：**`69.3 − 55 ≈ 14GB` 才是压缩能回收的部分**。
> 想拿回更多只有两条路——**删掉 WSL 里的真实数据**，或**把数据/发行版挪到 D 盘**。

### 2.1 为什么 WSL 会占这么多（而不是"不应该"）

你有 **5 个发行版**，每个都是一块独立的虚拟磁盘：

| 发行版 | 磁盘位置（实测） | 大小（实测） |
|---|---|---|
| **Ubuntu-24.04** | `C:\Users\12032\AppData\Local\wsl\{26b130cb-bad5-4ce1-a866-28bb829eecd1}\ext4.vhdx` | **69.3 GB**（其中真实数据 55GB） |
| Ubuntu-22.04 | `...\Packages\CanonicalGroupLimited.Ubuntu22.04LTS_79rhkp1fndgsc\LocalState\ext4.vhdx` | **4.0 GB** |
| Ubuntu（老） | `...\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\LocalState\ext4.vhdx` | **5.3 GB** |
| opp_env | `D:\wsl\opp_env\ext4.vhdx` | 4.1 GB（已在 D 盘） |
| wsl-vpnkit | `C:\Users\12032\wsl-vpnkit\ext4.vhdx` | 0.1 GB |

五个都是 **WSL2**（注册表 `Version=2`），所以都是 `ext4.vhdx`。

> ⚠️ **`df` 里的 `Size 1007G` / `Avail 901G` 是个陷阱**：那是 WSL 虚拟磁盘的**上限**，
> 而它的真实文件在 C 盘上只有 69.3GB、**C 盘也只剩 22.9GB**。
> 你在 WSL 里再写入超过约 20GB 的新数据，vhdx 就会试图增长**把 C 盘写满**（系统会卡死）。
> → **绝不要在 WSL 内部下载模型或存大型仿真结果**，一律放 `/mnt/d`。

三个原因叠加：
1. **WSL2 的 `ext4.vhdx` 只增不减**：在里面删文件，Windows 侧看到的文件大小**不会变小**；
2. **里面装了环境**：conda/venv、apt 缓存、pip 缓存、下载的模型、Docker 镜像都会算进去；
3. **5 个发行版各占一份**，其中 22.04 和老 Ubuntu 可能早就没用了。

> 注：`Ubuntu-22.04` 与 `Ubuntu` 的实际大小我读不到（`LocalState` 目录 ACL 保护），
> 需要用 **[管理员]** 权限看，或用 §2.3 的命令。

### 2.2 清理顺序（**顺序很重要，别跳步**）

> 核心矛盾：**D 盘也只剩 27.8GB，装不下一个 69GB 的发行版**。
> 所以必须**先用 pagefile 腾出 D 的空间，再压缩 WSL，最后才考虑搬迁**。

```
第 1 步（腾 D 空间）：把 D 盘的 50GB pagefile 改小 → 见 §3，可释放约 35GB
第 2 步（清里面）：进 WSL 清缓存/无用环境 → 见 §2.3
第 3 步（压缩 vhdx）：把删掉的空间真正还给 Windows → 见 §2.4
第 4 步（可选搬迁）：把 Ubuntu-24.04 移到 D 盘 → 见 §2.5
第 5 步（清 Windows）：Temp 5GB、旧发行版 → 见 §2.6
```

### 2.3 第 2 步：进 WSL 看什么大、清什么

```bash
wsl -d Ubuntu-24.04                     # 进入发行版

# ① 看谁最大（从根和 home 两个角度看）
du -h --max-depth=1 / 2>/dev/null | sort -h | tail -15
du -sh ~/.cache/* ~/.local/* ~/miniconda3 ~/anaconda3 /opt/* /usr/local/* 2>/dev/null | sort -h | tail -15
docker system df 2>/dev/null            # 装过 Docker 的话，镜像/卷常是最大头

# ② 清缓存（安全）
pip cache purge
rm -rf ~/.cache/pip ~/.cache/torch ~/.cache/huggingface ~/.cache/uv
conda clean -a -y                        # 有 conda 才需要
sudo apt clean && sudo apt autoremove -y # 需要 sudo
sudo journalctl --vacuum-size=100M       # 需要 sudo

# ③ 再看一次，记下清理前后的差
df -h / | tail -1
```

> ⚠️ **不要**在 WSL 里下载大模型权重（15GB 的 7B 会直接把 vhdx 撑大且不回收）。
> 模型放 Windows 的 D 盘，在 WSL 里用 `/mnt/d/models/...` 访问。

### 2.4 第 3 步：压缩 vhdx（把空间真正还给 Windows）

在 **Windows** 侧 **[管理员]** PowerShell：

```powershell
wsl --shutdown                                  # 必须先关掉所有发行版

# 方式 A（推荐，WSL 2.7 支持）：把磁盘设为稀疏，Windows 会按实际使用量计算占用
wsl --manage Ubuntu-24.04 --set-sparse true

# 方式 B：如果 --set-sparse 不支持，用 Optimize-VHD（需 Hyper-V 模块/Win11 专业版）
Optimize-VHD -Path "$env:LOCALAPPDATA\wsl\{26b130cb-bad5-4ce1-a866-28bb829eecd1}\ext4.vhdx" -Mode Full

# 方式 C：Hyper-V 也没有时，用 diskpart
#   diskpart
#   > select vdisk file="C:\Users\12032\AppData\Local\wsl\{26b130cb-...}\ext4.vhdx"
#   > attach vdisk readonly
#   > compact vdisk
#   > detach vdisk
#   > exit
```

压缩后重新 `wsl -d Ubuntu-24.04`，再用 §1 的清单重建环境（这一步一般能拿回几 GB 到几十 GB，
取决于 §2.3 清掉了多少）。

### 2.5 第 4 步：把发行版搬到 D 盘（可选但最彻底）

前提：**D 盘要有 ≥ 该 vhdx 当前大小的空闲空间**（所以先做 §3 释放 pagefile 空间）。

```powershell
# 方式 A（WSL 2.7 可能支持 --move；先看帮助）
wsl --manage Ubuntu-24.04 --move D:\wsl\Ubuntu-24.04

# 方式 B（通用，export/import）
wsl --shutdown
wsl --export Ubuntu-24.04 D:\wsl\ubuntu2404.tar          # 导出（耗时且占空间）
wsl --unregister Ubuntu-24.04                            # 注销（原 vhdx 会被删除，C 盘立即释放）
wsl --import Ubuntu-24.04 D:\wsl\Ubuntu-24.04 D:\wsl\ubuntu2404.tar --version 2
del D:\wsl\ubuntu2404.tar                                # 确认能进系统后再删 tar
# 注意：import 进来的发行版默认用 root 登录，需要按需补一个普通用户
```

> 回滚：import 失败就删掉 `D:\wsl\Ubuntu-24.04`，用 tar 重新 import；tar 没删之前都可回退。
> **最保险的做法**：先 `wsl --export` 到**外置硬盘**，再操作。

### 2.6 第 5 步：清 Windows 侧

```powershell
# 清 Temp（5GB）
Remove-Item "$env:LOCALAPPDATA\Temp\*" -Recurse -Force -ErrorAction SilentlyContinue
# 或用系统工具（更安全，含更新缓存、缩略图等）
cleanmgr            # 勾选：临时文件、Windows 更新清理、传递优化文件、缩略图

# 看还有哪些大目录（[管理员] 才能看全 Packages）
Get-ChildItem "$env:LOCALAPPDATA\Packages" -Filter "ext4.vhdx" -Recurse -Force -ErrorAction SilentlyContinue |
    ForEach-Object { "{0,8:N1} GB  {1}" -f ($_.Length/1GB), $_.FullName }

# 确认某个老发行版不再需要后注销它（会删除其所有数据，先 export 备份）
wsl --export Ubuntu-22.04 D:\backup\u2204.tar
wsl --unregister Ubuntu-22.04
```

---

## 3. D 盘的 50GB 虚拟内存怎么改

### 3.1 现状（实测注册表）

```
PagingFiles = d:\pagefile.sys 50000 50000      # 初始 50000MB、最大 50000MB = 固定 48.8GB
```

也就是说：**D 盘被一个固定 50GB 的页面文件占着**，这就是 D 盘"看起来很小"的原因之一。

### 3.2 该改成多少？

| 因素 | 说明 |
|---|---|
| 物理内存 | 16.9GB |
| 现状 | 50GB ≈ 3× 内存，**超出常规需要**（一般 1–1.5× 即可，或让系统托管） |
| **不能直接关掉** | **WSL2 的虚拟机内存依赖 Windows 的 pagefile 支撑**；关掉 pagefile 可能导致 WSL 起不来 |
| **未来跑 7B fp16 的约束** | 权重约 15GB **内存** + KV/激活 → 16.9GB 物理内存会不够，此时 pagefile 会被大量使用（性能很差）。所以**要么接受慢，要么只跑 int4/小模型，要么加内存** |

**建议：改成 8–16GB（初始 8192MB / 最大 16384MB），留在 D 盘。** 立即释放约 33–42GB。

### 3.3 图形界面改法（最稳）

1. `Win+R` → 输入 `sysdm.cpl` → 回车（或 `SystemPropertiesPerformance.exe` 直达性能设置）
2. **高级** 选项卡 → **性能** 区域点 **设置**
3. 性能选项里选 **高级** 选项卡 → **虚拟内存** 区域点 **更改**
4. **取消勾选** "自动管理所有驱动器的分页文件大小"
5. 选中 **D:** → 选 **自定义大小** → 初始大小 `8192`、最大值 `16384`（单位 MB）→ 点 **设置**
6. 若 C: 上也有页面文件且你不需要，选 C: → **无分页文件** → 设置（**但注意**：完全不设会失去崩溃转储能力；C 盘紧张时这样最省空间）
7. **确定** → **重启** 生效

### 3.4 命令行改法 **[管理员]** PowerShell

```powershell
# 查看现状
Get-CimInstance Win32_PageFileSetting | Select-Object Name, InitialSize, MaximumSize
Get-CimInstance Win32_PageFileUsage   | Select-Object Name, AllocatedBaseSize, CurrentUsage

# 关掉"自动管理"
$cs = Get-CimInstance Win32_ComputerSystem
if ($cs.AutomaticManagedPagefile) {
    Set-CimInstance -InputObject $cs -Property @{AutomaticManagedPagefile = $false}
}

# 把 D: 的页面文件改成 8GB/16GB
Set-CimInstance -CimInstance (Get-CimInstance Win32_PageFileSetting -Filter "Name='d:\\pagefile.sys'") `
                -Property @{InitialSize = 8192; MaximumSize = 16384}

# 若要在 C: 上新增（不推荐，C 更紧张）
# New-CimInstance -ClassName Win32_PageFileSetting -Property @{Name='c:\\pagefile.sys'; InitialSize=4096; MaximumSize=8192}

# 校验（改完需重启生效）
Get-CimInstance Win32_PageFileSetting | Select-Object Name, InitialSize, MaximumSize
```

也可直接改注册表（效果相同，重启生效）：
`HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management` → `PagingFiles`
（值形如 `d:\pagefile.sys 8192 16384`）。

> ⚠️ 页面文件是**系统级配置**，改错可能影响休眠/崩溃转储；建议改完保留一次重启前的系统还原点，
> 并确认 `wsl -d Ubuntu-24.04` 仍能启动（不能的话把最大值调回 16GB 以上再试）。

---

## 4. 顺便：给 4 台机器的"环境"定个统一规格

| 项 | 规格 | 说明 |
|---|---|---|
| Windows Python | 保留（3.12） | 跑我们自己的 agent/脚本；把 torch 换成 CUDA 版 |
| WSL 发行版 | **只留一个**（Ubuntu-24.04），装在 **D 盘** | 跑 vLLM/LMCache；22.04/老 Ubuntu 确认无用后注销 |
| WSL 内存 | `.wslconfig`：`[wsl2] memory=12GB swap=8GB`（按物理内存调） | 默认只给宿主 ~50%，7B 会 OOM |
| WSL 网络 | `.wslconfig`：`networkingMode=mirrored` | 否则别的机器连不进 agent 端口 |
| 模型位置 | **Windows 的 D 盘**（如 `D:\models`），WSL 里用 `/mnt/d/models` | 避免撑大 vhdx |
| 代码位置 | WSL 内 `~/work/project`（快）；或 `/mnt/d/...`（省事但慢） | 见 §1.3 |
| pagefile | D 盘 8–16GB | 释放约 35GB |
| CUDA Toolkit | **不装** | 除非编译扩展 |
| 入站端口 | 9100（agent）、9200（链路测量） | 防火墙放行 |

### 一次性校验清单
```bash
# Windows 侧
nvidia-smi
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"   # 换 CUDA 版后应为 True
wsl -d Ubuntu-24.04 -- nvidia-smi                                             # WSL 内也应能看到 GPU
wsl -d Ubuntu-24.04 -- df -h /                                                # 看 WSL 内部使用量

# WSL 侧
python3 -c "import torch;print(torch.__version__, torch.cuda.is_available())"
pip show vllm lmcache 2>/dev/null | head -20
```
