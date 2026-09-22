# 算子、注意力与显存：概念参照 + 成熟方向的现状综述

> 用途：作为"组件手册"用。分两部分：
> **上篇（§1–§6）解释概念**——融合/图优化到底在做什么、FlashAttention 各代、FlexAttention、
> 稀疏与 KV 驱逐、批 vs 数据并行、权重加载与 KV 分层；
> **下篇（§7）给成熟方向的现状**——拥挤不等于没价值：每个方向写清"谁做到什么程度、
> 有哪些现成实现可当组件、若要参与差异化在哪"。
>
> 口径声明：本文不再用"红海所以别碰"这种说法。**成熟度是投入产出判断，不是价值判断**；
> 了解成熟方向是系统设计的前提——它们要么是可复用的组件，要么是我们方案的 baseline。

---

# 上篇：概念

## 1. 融合（fusion）与图优化到底在做什么

### 1.1 先纠正一个常见误解

"融合是为了减少计算单元之间的通信"——**这个理解不对**。融合解决的是**同一个设备内部**的两件事：

| 融合消除的开销 | 机制 | 量级 |
|---|---|---|
| **HBM 往返流量** | 不融合时，每个算子都要把中间结果写回显存再读出来（如 `LayerNorm → MatMul`、`SiLU → Mul`、`Residual → Norm`）。融合后中间结果留在寄存器/shared memory | 现代 LLM 推理**大部分时间不是算力受限，而是显存带宽受限**（decode 阶段尤其：每生成 1 个 token 要把全部权重从 HBM 读一遍）。所以省 HBM 往返往往比省 FLOPs 更值钱 |
| **Kernel 启动开销与调度空隙** | 每个 kernel launch 有固定成本（几微秒）与 CPU 侧开销；小算子多时（norm/激活/量化）CPU 会成为瓶颈 | decode 时单 kernel 可能只有几十微秒，启动开销占比可观 |

**跨设备通信的优化是另一件事**，叫**通信-计算重叠（overlap）**或**通信融合**：
把 `AllReduce` 拆碎、与 GEMM 的计算交错执行（vLLM 的 pass 里就有 `fuse_allreduce_rms`、`fuse_gemm_comms`），
目标是"**通信时间被计算时间盖住**"，而不是"把两个算子合成一个"。
两者常被混为一谈，但机制完全不同：前者是编译器/内核层面，后者是并行策略层面。

### 1.2 "图优化"指两种东西

| 名称 | 做什么 | 解决什么 | 与硬件的关系 |
|---|---|---|---|
| **计算图优化**（inductor pass、算子融合、常量折叠、死代码消除） | 在编译期改写计算图 | 上表的 HBM 流量 + kernel 数量 | 与**算子语义+张量布局**强相关，与具体 SM 架构弱相关 |
| **执行图 / CUDA Graph** | 把一串 kernel launch 录制成一个图，之后**一次 launch 回放整串** | CPU 侧 launch 开销、launch 抖动（对 decode 小 kernel 尤其明显） | 与架构无关，但**要求形状静态** |

**"可断图"（breakable CUDA graph）为什么存在**：CUDA graph 要求捕获期间不出现"形状变化/主机同步/动态分支"。
推理里恰恰有这类东西——动态 batch、变长序列、**逐层异步 KV 操作**（KV connector 的
`wait_for_layer_load`/`save_kv_layer`）、通信等。所以 vLLM 有 `CUDAGraphMode.PIECEWISE`：
**把图在不能捕获的地方切开**，能捕获的段仍用 graph，切口处回落到 eager。
官方甚至要求"做逐层异步 KV 的 connector 必须声明 `requires_piecewise_for_cudagraph`"
（`distributed/kv_transfer/kv_connector/v1/base.py:630`），因为异步逐层操作天然无法被整图捕获。

### 1.3 这三件事"底层"到什么程度、是否硬件耦合

| 工作 | 需要什么知识 | 硬件耦合度 | 对学生/我们 |
|---|---|---|---|
| 写融合 kernel（Triton/CUDA） | 张量布局、shared memory、occupancy | **中**（布局 > 架构） | 门槛高、收益与架构强相关 |
| 挂自定义 inductor pass | torch.compile 的 IR、图改写 | **低** | 可行，但"为融合而融合"没有系统价值 |
| 用 CUDA graph / 处理图断裂 | 理解动态形状与主机同步 | **低**（概念清晰） | **该懂**，因为它直接约束我们的动态重配置与逐层异步方案 |
| TensorRT-LLM 类推理编译器 | 每个 kernel 的**架构专属实现 + 逐架构编译 cubin** | **高** | 不做；但要知道它代表"极致静态优化"的形态 |

**一句话**：融合/图优化里，**"概念与约束"很值得懂（尤其是图断裂与动态性的冲突）**，
**"写高性能融合 kernel"则高度依赖架构，不适合我们**。

---

## 2. FlashAttention 2/3/4 是什么

FlashAttention 解决的根本问题是：**注意力不要在显存里物化那个 N×N 的注意力矩阵**。
它用 **tiling（分块）+ online softmax（在线归一化）** 把注意力做成"流式"计算，
显存从 O(N²) 降到 O(N)，同时因为减少了 HBM 往返而更快。三代差别主要在**并行划分**与**硬件特性利用**：

| 版本 | 目标架构 | 关键改动 | vLLM 里的门槛（源码证据） |
|---|---|---|---|
| **FA1** | Ampere(sm80)+ | 分块 + 在线 softmax，不物化 N×N | — |
| **FA2** | Ampere/Ada(sm80/sm86/sm89) | 更好的**工作划分与并行度**（减少非矩阵乘开销、支持更大 head dim、更好的 warp 分工） | **≥ 8.0**：`flash_attn_interface.py:52-59`；后端级 `flash_attn.py:250-252`（`capability >= (8,0)`） |
| **FA3** | **Hopper(sm90) 专属** | 利用 Hopper 的 **TMA（异步张量拷贝）+ warp specialization（生产者/消费者 warp 分工）+ FP8** | `_is_fa3_supported()`：`"FA3 is only supported on devices with compute capability 9.x"`（`flash_attn_interface.py:62-69`） |
| **FA4** | **Blackwell(sm100/12x)** | 用 **CuTe-DSL** 生成 kernel，支持**自定义 mask_mod**（token 级精确稀疏 mask） | 选择逻辑 `fa_utils.py:93-101`：major==9→FA3、major==10→FA4、**否则 FA2** |

**对我们的直接含义**：sm89（4060）与 sm86（3060）**只能用 FA2**；FA3/FA4 都不行。
这不是"少个优化"，而是**决定了"自定义稀疏 mask"能不能做**：FA2/FA3 明确不支持 `mask_mod`
（`NotImplementedError("FA2 does not support mask_mod")`、FA3 同），**只有 FA4 和 FlexAttention 支持**。

---

## 3. FlexAttention：细节与它在 vLLM 里的接入

### 3.1 它是什么

FlexAttention 是 **PyTorch 官方**（`torch.nn.attention.flex_attention`）提供的一种注意力接口：
**你不写 kernel，而是写一个 Python 函数描述"哪些 query-key 对参与计算"**，
由 `torch.compile` 把它**编译成一个 Triton kernel**。

三个核心概念：

| 概念 | 作用 |
|---|---|
| `score_mod(score, b, h, q_idx, kv_idx)` | 改**分数**：相对位置偏置、ALiBi、soft-cap、tanh 门控等 |
| `mask_mod(b, h, q_idx, kv_idx) -> bool` | 改**可见性**：因果、滑窗、前缀 LM、任意自定义稀疏 pattern |
| `BlockMask` / `create_block_mask(...)` | 把 mask **按块（如 128×128）摘要**成元数据；**整块全被 mask 的块直接跳过**，不加载、不计算 |

关键点：**它的价值不是"比 FlashAttention 快"，而是"能表达 FlashAttention 表达不了的 pattern"**。
密集注意力下它通常略慢于手写 FA2；稀疏/自定义 pattern 下它是唯一能用的路子。

### 3.2 vLLM 怎么接它（已核实）

- FlexAttention 在 vLLM 里是一个**正式 attention 后端**（`AttentionBackendEnum.FLEX_ATTENTION`，
  `v1/attention/backends/flex_attention.py`）。
- **`mask_mod` 从模型层读**：`getattr(layer, "logical_mask_mod", None)`（`:1329`），
  并且可与内建掩码**组合**（`and_masks`/`or_masks`：paged/causal、bidirectional、sliding_window、
  prefix_lm、R-SWA，见 `:631-652`）。
- **`block_sparsity_hint`**（`:347-359`）是给稀疏 pattern 配套的"块级剪枝提示"，
  docstring 逐字：*"This prunes KV blocks from the BlockMask before the flex_attention kernel is
  invoked, so that blocks that are fully masked never get loaded. Use this with custom mask_mods
  that are sparse to avoid the kernel iterating over all KV blocks unnecessarily."*
  —— 也就是说：**用它可以做到"被 mask 掉的 KV 块连读都不读"**，
  这正是"稀疏 → 省显存带宽"的关键（只 mask 不剪块，仍然要读）。

### 3.3 代价与限制（要如实知道）

- 它走 **Triton**（不是 CUDA 手写），所以在消费级卡上**可用**——这是我们能碰稀疏的**唯一现实入口**。
- 稀疏/Triton 路径与 **CUDA graph 捕获**有冲突（动态 mask/块表），通常要配合 `PIECEWISE`。
- 它的性能**依赖块稀疏度**：pattern 稀疏度不够时，"跳过块"的收益会被块级粒度（如 128）吃掉。

---

## 4. 稀疏注意力与 KV 驱逐

### 4.1 两种"稀疏"，归属完全不同

| 类型 | 稀疏 pattern 从哪来 | 是否需要训练 | 例 |
|---|---|---|---|
| **原生稀疏**（learned/native sparse attention） | **模型的一部分**：训练时就带一个 indexer 学"该看哪些 token" | **需要** | NSA（DeepSeek）、MoBA（Moonshot）、BigBird/Longformer（固定 pattern 预训练）、DeepSeek-V4 sparse MLA、MiniMax M3 的 MSA |
| **免训练稀疏**（inference-time sparse / KV selection） | **推理时按统计量选**：注意力分数累积、位置（sink+recent）、预算 | **不需要** | H2O、StreamingLLM、SnapKV、PyramidKV、TOVA、MInference（prefill 阶段 pattern） |

对第一类，vLLM 的后端**全部是"model-driven"**（源码注释逐字："DeepSeek V4 sparse MLA backends
(model-driven; selected via the V4 layer)"），且 `mla/indexer.py` 就是那个 indexer；
这些后端多数要求 SM90+/FP8/CUTLASS。→ **我们只能用别人的开源权重，且多半跑不动**。

对第二类，**vLLM 里没有任何实现**（全仓搜 `h2o/heavy hitter/token evict/streamingllm` 只命中
"attention sink 作为模型特性"，没有推理时驱逐策略）→ 空白，但也是"要自己做"。

### 4.2 驱逐是否会降低 KV 存储占用？——会，但有三个前提

你的理解方向对：**"后续计算不使用某些 token 的 KV"**。但要精确区分三种情形：

| 做法 | 省计算? | 省显存? | 说明 |
|---|---|---|---|
| 只在 attention mask 里排除（不释放块） | ✅ | ❌ | 只省了注意力计算与带宽（若块被剪），**占用不降** |
| 真的释放/复用那些 KV block | ✅ | ✅ | 占用才下降。vLLM 的 `BlockPool` 支持 block 回收（LRU 驱逐），但**按块不按 token** |
| 减少需要"跨机搬运"的 KV | ✅ | ✅ | 在我们的场景里这是额外收益：**驱逐直接减少迁移字节与迁移时间** |

**三个必须知道的耦合**（这会直接影响设计）：
1. **块粒度**：KV 按 block（如 16 tokens）分配，驱逐只能整块驱逐 → 实际稀疏度会被块大小量化。
2. **前缀缓存冲突**：`enable_prefix_caching` 下，**一个 block 可能被多个请求共享**（相同前缀）。
   你为了请求 A 驱逐某块，可能会破坏请求 B 的命中 → **驱逐必须与引用计数/pin 机制配合**
   （LMCache 里就有 `pin`/`unpin`，vLLM 的 block 有 `ref_cnt`）。
3. **位置语义**：StreamingLLM 类方法发现要保留 **sink token + 最近窗口**，否则质量崩塌；
   而"最近窗口"是**必须连续**的 → 驱逐后 KV 的物理布局与逻辑位置会错位，
   要么保留位置映射，要么在 mask 里表达"跳跃可见"。

### 4.3 为什么"哪些 KV 参与计算"在我们场景里比在数据中心更值钱

在单机数据中心里，这个旋钮主要影响**注意力计算量与显存**。
在我们"跨机 + 分层 + 弱网"的场景里，同一个旋钮**同时决定四件事**：

```
哪些 KV 参与计算
   ├─ 注意力计算量（延迟）
   ├─ 本机显存占用（能不能装下更长上下文）
   ├─ 节点离开时要迁移的字节数（恢复时间）
   └─ 前缀缓存能复用的比例（跨请求收益）
```

**四个效应耦合**，且第 3、4 项是数据中心论文不关心的——这就是差异化所在，
而不是"再提出一个驱逐评分函数"。

---

## 5. 批（batch） vs 数据并行（data parallel）：两个正交的维度

这是最容易混的一处，先把定义钉死：

| 概念 | 在哪发生 | 含义 | 目的 |
|---|---|---|---|
| **批 / 连续批处理（batching）** | **单设备内部** | 把**多个请求/序列**打包进同一次前向（同一份权重、同一次 kernel 调用） | 提高算术强度、摊薄每 token 的权重读取与 kernel 启动开销 → **降低单 token 成本** |
| **数据并行（DP）** | **跨设备** | **复制整个模型**到多张卡/多台机，每个副本处理**不同请求** | 线性扩展**吞吐** |
| **流水线并行（PP）** | 跨设备，**切模型** | 一个请求横跨多段，段间传激活 | 装下更大的模型 / 用多机算一个请求 |
| **微批（micro-batch / ubatch）** | **单设备内部**（在 PP/EP 场景下特指切分后的小批） | 把一个批**再切成 M 份**，逐份推进 | **填满流水线（消气泡）**、让**通信与计算重叠** |

**所以你的疑问"批次上的并行就是数据并行吗？"——不是。**
- "开几条流水线" = **DP 副本数**（每个副本自带一条流水线）；
- "一条流水线里同时有多少个批在飞" = **微批/并发批**，它决定**同一条流水线的利用率**。

两者是乘法的关系，不是同一个东西：

```
总吞吐 ≈ (DP 副本数) × (单副本吞吐)
单副本吞吐 ≈ 批大小 × (1 / 每 token 时间)     ← 受最慢段与气泡限制
```

**我们实测过的现象正好解释这一点**：之前 K=1→4 并发只从 5.06 提到 7.82 tok/s（1.55×），
因为那时每个请求都是"顺序走完全部 stage"，且每段有一把锁——**单条流水线的利用率没被填满**。
真正提高要靠：**微批（把多个请求的同一段合成一次前向）** + **让段间通信与计算重叠**。

**"双批重叠"（dual-batch overlap）**：让**两个批反向流动**（批 A 在第 i 段时，批 B 在第 i+1 段），
使每张卡"永远有活干"、通信与计算互相遮盖。DeepSeek 的 **DualPipe** 是这个思路在**训练**里的著名实现；
推理侧对应的做法就是 **2 个以上并发批 + 通信/计算交错**（vLLM 的 `v1/worker/ubatching.py`
就是这套：把 batch 切成 ubatch 以便把 all-to-all 与计算重叠，`gpu_ubatch_wrapper.py:160-184`
里还能给通信留 SM）。

**一句话**：DP 是"多开几条流水线"，微批/双批是"让一条流水线别闲着"。我们现在缺的是后者。

### 5.1 把"请求 × 设备"的切分关系列全（避免继续混淆）

| 切的是什么 | 跨不跨设备 | 名称 | 一个请求会怎样 |
|---|---|---|---|
| **模型层** | 跨设备 | **PP**（流水线并行） | **一个请求被切成多段**，逐段流经不同设备 |
| **张量/权重** | 跨设备 | **TP**（张量并行） | 一个请求的**同一层**被切到多设备，层内要通信 |
| **专家** | 跨设备 | **EP**（专家并行） | 一个请求激活的专家分布在不同设备，每层 all-to-all |
| **上下文/序列** | 跨设备 | **CP/SP**（上下文/序列并行）、Ring Attention | 一个请求的**长序列**被切开分到多设备算注意力 |
| **阶段** | 跨设备 | **PD 分离** | 一个请求的 prefill 与 decode 在不同设备上 |
| **请求集合** | **同一设备内** | **batch / 连续批处理** | **多个不同请求**并进**同一次前向**，共享一次权重读取 |
| **批内子集** | 同一设备内（PP/EP 场景） | **micro-batch / ubatch** | 批再切小，用于**填满流水线、让通信与计算重叠** |
| **时间步** | 同一设备内 | **chunked prefill** | **长 prompt 分块**跨多个 step 处理（仍是同设备，不是分给别的设备） |
| **同一请求的多个候选** | 同一设备内 | **beam search / n>1 采样** | 一个请求展开成多条序列，一起批处理（vLLM 的 `parallel_sampling.py`） |

**判别口诀**：
- "**把一个请求拆给多台设备**" → 名字是 **PP / TP / EP / CP**，**不是 batch**；
- "**把多个请求合到一台设备的一次计算里**" → 才是 **batch**；
- "**把一批再切小**" → **micro-batch**，目的是重叠与填流水线；
- "**把多个请求分给多台设备**" → **请求级路由 / DP**。

**为什么 batch 能提升吞吐**：decode 阶段是**显存带宽受限**——每生成 1 个 token 都要把全部权重从 HBM 读一遍。
批处理让**多个请求摊薄同一次权重读取**，所以提的是**吞吐与单位成本**，它并不"拆"任何请求。

---

## 6.5 LMCache 自己的算子层（回答"KV 操作有没有专门算子"）

**有，而且这是一层完整的 C++/CUDA 扩展**（`lmcache.c_ops`，pybind 绑定；无 CUDA 构建时回落到
`lmcache.non_cuda_equivalents`）。这一点很关键：**一个 KV 管理系统真的会自己去写算子。**

`csrc/` 的构成与用途：

| 源文件 | 是什么算子 |
|---|---|
| `mem_kernels.cu/.cuh`、`pos_kernels.cu/.cuh` | KV **搬运/定位** kernel（把 KV 在 GPU↔host 之间搬、按块/按层定位） |
| `ac_enc.cu`、`ac_dec.cu`、`cal_cdf.cu`、`cachegen_kernels.cuh` | **CacheGen 的算术编码/解码与 CDF 计算**（KV 压缩的熵编码部分） |
| `pybind.cpp` | Python 绑定与枚举（`TransferDirection`、`GPUKVFormat`） |
| `storage_manager/{bitmap,ttl_lock}.cpp` | 存储管理辅助（位图、TTL 锁） |
| `storage_backends/redis/connector.cpp` | **直接在 C++ 里实现的 Redis 连接器** |
| `rust/raw_block/` | Rust 扩展：裸块设备后端（O_DIRECT） |

Python 侧实际调用的算子名（这就是"KV 操作算子"的真实清单）：

| 算子 | 作用 |
|---|---|
| `lmc_ops.multi_layer_kv_transfer(...)` | **多层 KV** 在 GPU↔host 之间批量搬（含 `TransferDirection.H2D/D2H`） |
| `lmc_ops.multi_layer_kv_transfer_unilateral(...)` | **单边**多层传输（SGLang 路径用） |
| `lmc_ops.single_layer_kv_transfer(...)` / `..._sgl(...)` | **单层** KV 搬运（layerwise 模式的核心） |
| `lmc_ops.lmcache_memcpy_async(...)` | 异步 memcpy（GPU↔host） |
| `lmc_ops.rotary_embedding_k_fused(...)` | **融合的 RoPE-K 算子**（把位置编码写进 K 的动作融进搬运路径） |
| `lmc_ops.encode_fast_new` / `decode_fast_prefsum` / `calculate_cdf` | CacheGen 压缩/解压 |
| `lmc_ops.alloc_pinned_ptr` / `alloc_shm_pinned_ptr` / `alloc_pinned_numa_ptr`（+free） | **pinned / NUMA 感知 / 共享内存分配器** |
| `lmc_ops.GPUKVFormat`（6 种布局枚举） | 识别并适配 vLLM/SGLang 的不同 KV 内存布局 |

**三个结论**：
1. **这些算子的性质全是"数据搬运 / 布局转换 / 压缩 / 分配"**，而不是 GEMM/attention 那类计算算子——
   和上一节说的"数据搬运类算子耦合度低"完全吻合。
2. **它耦合的是"KV 内存布局"，不是"SM 架构"**：`GPUKVFormat` 要认得 vLLM 的
   `[num_blocks, block_size, num_heads, head_size]`、SGLang 的指针列表、MLA 的压缩布局等。
   所以维护成本来自**布局随引擎/版本变化**，而不是硬件移植——这对"异构集群"是好消息：
   同一份搬运 kernel 在 sm75/sm86/sm89 上都能跑。
3. **它还做了算子融合**（`rotary_embedding_k_fused`）——说明"把 KV 相关的小算子融进搬运路径"
   是这类系统的常规手法，也正是我们能碰的那一类优化。

---

## 6. 权重流式加载 / P2P 权重分发 / 多级 KV

| 概念 | 是什么 | 解决什么 | 现成实现 |
|---|---|---|---|
| **权重流式加载**（streaming / lazy load） | 权重**不一次性全读进显存/内存**，而是按需（逐层）加载：用哪层读哪层，用完可换出 | **峰值内存**；让"显存装不下的大模型"能跑（代价是吞吐） | vLLM `LoadConfig.safetensors_load_strategy ∈ {lazy, eager, prefetch, torchao}`；`model_loader/reload/layerwise.py` 的逐层 materialize；`config/offload.py` 的 UVA/Prefetch（GPU↔CPU） |
| **P2P 权重分发** | 新加入/重切层的节点**不再从模型仓库下载**，而是**从已持有该分片的对端直接拉**（类似 BitTorrent / 分片互传） | 加入/重配置时的**装载时间**（我们从模型库下载 942MB 用了 80s，实测装载 2–8s/节点） | vLLM **没有**（它只有 trainer→inference 的 `WeightTransferEngine`，且 RDMA 后端 TODO）；这属于**要自研**的部分，和我们 `StageRuntime.prepare_epoch(model_source=...)` 里"同区间换节点可取旧节点为 source"的语义完全对应 |
| **多级 KV**（KV tiering） | KV 按热度在 **GPU → CPU → 本地盘/SSD → 远端** 之间升降级 | 用**便宜存储**换**上下文长度/并发量**；命中时省 prefill | vLLM：`kv_offloading_size/backend ∈ {native, lmcache}`、`SimpleCPUOffloadConnector`（CPU/disk）、`v1/kv_offload/tiering/p2p`（第二级，可 P2P）；LMCache：`local_cpu`/`local_disk`/`remote_url`/`lm://` |

**与我们的关系**（只说事实，不给方案）：
- 8GB 笔记本上，**多级 KV 是"上下文长度"的直接杠杆**；
- **P2P 权重分发**是"节点加入/离开后重切层"快慢的关键，而它是 vLLM 的空白；
- **权重流式加载**已经在 vLLM 里有配置级支持，可作为对照。

---

# 下篇：成熟方向的现状（了解 ≠ 要参与）

> 下面每个方向给三栏：**现状（谁做到什么程度）→ 可当组件的部分 → 若要参与的差异化角度**。
> 目的是让你在系统设计里知道"这块钱该用谁的、自己做到哪一层"，而不是"该不该研究"。

## 7.1 注意力 kernel（FlashAttention / MLA / 稀疏）

- **现状**：FA2/FA3/FA4 分代明确（§2）；MLA 系 kernel 由 DeepSeek 开源（FlashMLA）；
  稀疏注意力在 vLLM 里已有一批**模型驱动**后端（§4.1）；FlexAttention 提供了"自定义 mask 编译成 Triton"的通用路径。
- **可当组件**：直接选后端（配置级）；要自定义 pattern 就用 FlexAttention + `mask_mod`/`block_sparsity_hint`。
- **差异化角度**：不是"更快的注意力"，而是**"注意力 pattern 由系统状态决定"**——
  例如稀疏度随链路质量/迁移预算动态调整（数据中心没有这个输入）。

## 7.2 量化（权重/KV/激活）

- **现状**：格式与方法已收敛（GPTQ/AWQ/Marlin 做 int4；FP8 在 Ada/Hopper 起可用；
  KV 量化有 KIVI 等；vLLM 有 30 个量化方法注册表 + `register_quantization_config` 可扩展）。
- **可当组件**：`--quantization` 直接选；KV 用 `cache_dtype` + `kv_cache_dtype_skip_layers`。
- **差异化角度**：**量化改变每层耗时与 activation 字节 → 改变最优切层点**。
  "把量化当调度输入"而不是"当加速手段"，是数据中心不会做的组合。

## 7.3 KV 压缩 / 驱逐 / 复用

- **现状**：压缩（CacheGen 类）与驱逐（H2O/SnapKV 类）**方法很多**，
  但**引擎里普遍只有 block 级 LRU**；逐 token 驱逐在 vLLM 里没有实现。
- **可当组件**：前缀缓存（现成）、KV 量化（现成）、分级 offload（现成）、
  LMCache 的层粒度键与分层压缩（见另一份文档 §4）。
- **差异化角度**：**驱逐/选择与"跨机迁移"耦合**——"不参与计算的 KV 也不用搬"，
  这把显存、迁移字节、恢复时间三个指标绑在一起。

## 7.4 投机解码

- **现状**：非常成熟。vLLM 里 `SpeculativeMethod` 覆盖 ngram/medusa/draft_model/eagle/eagle3/mtp/dflash/dspark，
  还有 ~25 个模型专属 MTP 变体、`parallel_drafting`、三种拒绝采样（standard/synthetic/block）、
  `draft_tensor_parallel_size`、草稿模型独立量化与 attention 后端。
- **可当组件**：**配置级即可用**（有对应 draft head 的模型）。
- **差异化角度**：草稿模型本身也可以**跨机放置**（草稿在弱机、验证在强机？或反之），
  以及"投机收益 vs 跨机通信"的联合取舍——这属于调度层。

## 7.5 服务/调度（SLO、抢占、准入控制、路由）

- **现状**：连续批处理/chunked prefill 已成标配；SLO 感知与抢占（FastServe 类）、
  过载早拒（Mooncake）、KV 感知路由（Preble/llm-d 类）都有成熟工作。
- **可当组件**：vLLM 的 `--scheduler-cls`（接口标注非公开）、KV connector 的前缀命中钩子。
- **差异化角度**：**调度决策的输入里加入链路与节点不稳定性**（数据中心假设链路稳定）。

## 7.6 并行与通信（TP/PP/EP/CP、重叠、EPLB）

- **现状**：TP/PP/EP/CP 与 all-to-all 后端（DeepEP/FlashInfer-NVLink/MoRI/NIXL）都已工程化；
  EPLB 做专家负载均衡且通信后端可插拔；通信-计算重叠在 MoE 里用 ubatch 实现。
- **可当组件**：EPLB 的 `rearrange` + 通信后端；`all2all_backend` 选项。
- **差异化角度**：**我们场景没有 NVLink**，所以"哪些并行维度在弱网下还成立"本身就是可用实验回答的问题
  （我们已有结论：PP 成立、TP 不成立）。

## 7.7 编译 / 图 / 运行时（CUDA graph、融合、TRT 类）

- **现状**：CUDA graph 与 PIECEWISE 断图是 vLLM 的日常工程；融合由 torch.compile/inductor 主导；
  TensorRT-LLM 走"逐架构编译专属 kernel"的极致静态路线。
- **可当组件**：`cudagraph_mode`、`compilation_config.inductor_passes`。
- **差异化角度**：**"可断图"的断点恰好在我们的动态操作处**（逐层 KV、重配置）——
  研究"重配置后的图如何重建/复用"是有系统价值的。

## 7.8 显存与加载（多级 KV、权重流式、P2P 分发）

- **现状**：多级 KV 在 vLLM/LMCache/Mooncake 都已实现；权重流式加载有配置级支持；
  **P2P 权重分发在 vLLM 里是空白**。
- **可当组件**：`kv_offloading_*`、`safetensors_load_strategy`。
- **差异化角度**：**"节点加入时从对端拉分片，而不是从云端/磁盘重读"**，
  以及"迁移 KV 与迁移权重的成本对比"（论文里前瞻里就写过这个权衡）。
