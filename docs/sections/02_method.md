# 02 Method：技术方案 —— 边缘分布式算力上的 LLM 推理调度框架

> 状态：草稿 v2（2026-08-16；v1 框架 + v2 策略实装）
> 对应实现：`project/edge_llm_scheduler/`（50 测试通过，含 E2 与 PT×N 策略）
> 设计详情：`plan/review/framework-design.md`
> 本章节把"框架实现"提升到论文方法层面：问题 → 系统模型 → 机制 → 策略 → 验证。

---

## 1. 任务设定与系统模型

**问题**：在边缘异构算力节点（消费级 GPU/手机/边缘盒，非数据中心）上协作执行 LLM 推理。核心挑战是三类约束同时成立：
1. **带宽受限**：节点间链路（WiFi/毫米波）比数据中心低 1-2 个数量级 → TP 不可用，跨设备通信成本高。
2. **节点动态变化（churn）**：节点随时加入/离开，无数据中心式的宽限期 → 需要快速装载/卸载与状态迁移。
3. **内存受限**：边缘显存小，KV 缓存无法全放 GPU → 必须分级存储 + 缓存复用。

**系统模型**：把整个系统看作一张图。节点是算力单元（静态能力：算力/显存/带宽；动态状态：剩余显存/负载/KV 水位/持有层/持有专家），链路有带宽限制。调度层在这张图上做三类决策：**放置**（模型分片/专家放哪）、**路由**（请求去哪）、**迁移**（节点变化时状态去向）。

**设计原则：机制与策略分离。**
- **机制（机制层 core/）**：框架必须完整实现——对接接口、数据结构、流程编排。缺了它系统跑不起来。
- **策略（策略层 policies/）**：可插拔算法——"具体怎么做"由策略决定。复杂算法（缓存感知路由、优先级迁移）作为策略注入，可逐步替换、可对比实验。

## 2. 机制层设计（框架骨架）

### 2.1 核心抽象（三个接口，对接真实系统）

**KV 存储抽象 `KVStore`**（对接 LMCache）：
```python
async def save(block, location) / load(block_hash, location)
async def lookup(prefix_hashes, locations=None) -> int   # 连续命中块数
async def move(block_hash, src, dst) / evict(block_hash, location)
async def get_index() -> Dict[int, List[str]]  # 全局索引：block_hash -> 位置
```
- `location` 是不透明字符串（`node_a:gpu/cpu/disk`），支持分级存储。
- 框架不直接依赖 LMCache 的 StorageBackendInterface（它耦合 torch/MemoryObj），自定义轻量接口让 LMCache 作为 backend 实现——可测、可换。

**传输抽象 `Transport`**（链路抽象：有线/无线）：
```python
bandwidth_mbps; async def push(data, dst, tag) / pull(src, tag)
async def measure_bandwidth(dst) / estimate_transfer_time(bytes, dst)
```
- 调度器只依赖抽象，不知道底层有线/无线——适配在 backend（TCP 有线 / WiFi 无线）。

**推理引擎抽象 `Engine`**（对接 vLLM，OpenAI API 驱动）：
```python
async def generate(task) -> GenerationResult
```
- 参考 Preble 模式：调度层不碰 vLLM 内部，纯 HTTP 驱动。

### 2.2 模型装载（精细控制）
`ModelManager` 支持控制"装哪些层、哪些专家、专家进哪级存储"：
```python
load_layers(node_id, layer_range)          # 指定层区间
load_experts(node_id, expert_ids, tier)    # 专家进 gpu/cpu/disk
```
依据：MoE 模型中**专家参数占 97%+、注意力可全复制**（Qwen2-57B: 注意力+嵌入 1.9B vs 专家 55.5B）。因此注意力全复制到每节点，专家按热度分布——这决定了装载的粒度。

### 2.3 事件模型与节点生命周期
`EventBus` + `NodeManager`：节点注册 → 发布 NODE_JOINED（触发放置）；心跳超时/主动离开 → 发布 NODE_LEFT（触发迁移/重并行化）。这是 Preble `runtime_request_queue` 队列模式的泛化。

## 3. 策略层设计（核心研究内容）

框架机制是骨架，**研究贡献集中在策略层**。本文详述两个核心策略：E2 缓存感知路由（放置策略）与 PT×N 优先级迁移（迁移策略）。另两个策略（重并行化、中断恢复）为接口+默认实现，后续研究。

### 3.1 放置策略：E2 缓存感知路由

**目标**：请求到达时，决定"发到哪个节点"，同时优化 **KV 缓存命中**（省 prefill 计算）与 **负载均衡**（避免热点节点过载）。

**动机**（来自 Preble 的观察 + 我们的场景）：边缘场景下 prefill 计算量大（长 prompt），KV 命中省下的 prefill 比 decode 优化更值钱；但都去命中节点会过载。所以路由必须在"命中收益"与"负载均衡"间权衡。

**决策逻辑（exploit/explore 两分法）**：
```
对每个请求：
  前缀匹配：查询 KVStore 全局索引，找最长连续命中（lookup 返回命中块数）
  if 命中 token 数 > 剩余非共享 token 数:      # exploit 收益 > 负载均衡收益
      → EXPLOIT：送"持有最长命中前缀"的节点
  else:
      → EXPLORE：送"prompt-aware 负载最轻"的节点
```

**实现为 PlacementPolicy（`policies/placement.py::E2Placement`，已实装+测试）**：
```python
class E2Placement(PlacementPolicy):
    def __init__(self, storage: KVStore):
        self.storage = storage

    async def place(self, model, nodes, request, storage, hit_tokens=0):
        # 1. 前缀匹配：查全局 KV 索引，找最长命中
        prefix_hashes = hash_blocks(request.prompt)          # prompt → 块哈希序列
        hit = await storage.lookup(prefix_hashes)            # 连续命中块数
        matched_tokens = hit * BLOCK_SIZE                    # 命中 token 数
        remaining_tokens = len(request.prompt) - matched_tokens

        # 2. exploit/explore 决策
        if matched_tokens > remaining_tokens:
            # EXPLOIT：找持有最长命中前缀缓存的节点
            index = await storage.get_index()
            candidates = nodes_holding(index, prefix_hashes[:hit])
            target = min(candidates, key=load)               # 命中节点中选负载轻的
        else:
            # EXPLORE：prompt-aware 负载最轻（考虑驱逐代价）
            target = min(nodes, key=lambda n: load_cost(n, storage))

        # 3. 生成任务
        return [Task(node_id=target.node_id, ...)]

    async def load_cost(self, node, storage):
        # prompt-aware 负载 = 已有负载 + 驱逐代价（见 3.2 的 PT×N）+ 新请求成本
        base = node.state.load
        evict_cost = await self.eviction_cost(node, storage)  # 预驱逐的 KV 块价值
        return base + evict_cost
```

**关键机制细节**：
1. **前缀匹配用 KVStore.lookup**：全局索引记录"block_hash → 持有节点"，lookup 返回连续命中块数。这比 Preble 的 radix tree 更轻量（hash 查表 O(1)/块），代价是不支持"部分匹配分支"——对我们的块级粒度足够。
2. **exploit 的"收益 > 负载均衡"阈值**：Preble 用"匹配 token > 剩余 token"。这是启发式阈值，可调（后续可加权重参数）。
3. **explore 的负载定义含驱逐代价**：去轻节点可能要先逐它已有的 KV——这本身有代价（重算）。所以"负载"不只看现在忙不忙，还要看"腾地方要牺牲多少已缓存价值"。

### 3.2 迁移策略：PT×N 优先级迁移

**目标**：节点离开时，决定它持有的 KV 块"传不传、传多少、传给谁"。这是 D4（节点离开装载卸载）的核心。

**优先级公式**（参考 Preble 驱逐价值公式，用于"该先传谁"）：
```
迁移价值(KV块 j) = PT_j × N_j
  PT_j = 重算这段 KV 的 prefill 耗时
  N_j  = 该前缀块的历史共享请求数（热度）
```
直觉：一个前缀被越多请求共享、越长，重算越贵 → 越该传走。这是"冷热"的量化形式（比 LRU/LFU 更精确）。

**实现为 MigrationPolicy（`policies/migration.py::PriorityMigration`，已实装+测试）**：
```python
class PriorityMigration(MigrationPolicy):
    async def decide(self, node_id, kv_block_ids, storage):
        plan = MigrationPlan()
        # 1. 收集该节点持有的 KV 块，算每块价值
        index = await storage.get_index()
        blocks = [await storage.load(bid, f"{node_id}:gpu") for bid in kv_block_ids]
        valued = [(b, b.reuse_count * b.prefill_time_ms) for b in blocks if b]

        # 2. 按价值降序排序
        valued.sort(key=lambda x: -x[1])

        # 3. 目标节点：剩余显存够的、链路近的
        targets = available_nodes_with_memory(...)

        # 4. 按价值从高到低分配（时间受限时只传前 K 个）
        deadline = get_grace_period()                 # 宽限期（边缘可能为 0）
        budget = deadline * bandwidth_mbps            # 可传的总字节
        for block, value in valued:
            cost = await storage.estimate_move_cost(block, src, dst)
            if cost <= budget_remaining:
                plan.block_moves.append({block_hash, dst, priority=value})
                budget_remaining -= cost
            else:
                plan.block_drops.append(block_hash)   # 传不完则丢（可重算）
        return plan
```

**关键机制细节**：
1. **价值排序**：`reuse_count × prefill_time_ms`（PT×N）。`reuse_count` 从请求历史统计（SlidingWindowHistogram 类似 Preble），`prefill_time` 用 profiling 的每 token prefill 时长 × 块 token 数。
2. **时间受限部分迁移**：宽限期（deadline）× 带宽 = 可传字节预算。按价值从高到低传，传不完的丢（重算成本已在价值里体现——价值低的丢了不心疼）。
3. **目标节点选择**：剩余显存够（KV 落不下就别传）、链路近（传得快）。这结合了我们之前分析的"目标可行性"维度。
4. **token 级截断**（后续增强）：时间极端紧时，块的 KV 可按 token 截断传（已提交的 token 稳定，参考 SpotServe token 级恢复）。

### 3.3 另两个策略（接口+默认，后续研究）

| 策略 | 接口 | 默认 | 后续研究内容 |
|---|---|---|---|
| 重并行化 | `reconfigure(nodes, placements)` | 放置不变 | 节点变化后 (D,P,M,B) 调整（SpotServe solver）；流水线拓扑重排（HotSwitch） |
| 中断恢复 | `recover(task)` | 重跑 | token 级恢复（SpotServe stateful recovery）；跨节点重调度 |

## 4. 验证方法（分层，无需 GPU）

| 层级 | 方法 | 覆盖 |
|---|---|---|
| L1 框架逻辑 | mock 后端 | 请求生命周期/节点事件/装载/KV/传输（20 测试） |
| L2 对接契约 | fake 后端 | VLLMEngine 请求格式+解析；LMCacheStore 方法调用+参数；TCP/WiFi loopback（18 测试） |
| L3 真实集成 | 真实 vLLM+LMCache+GPU | 端到端（需 GPU 环境） |

**为什么 L2 不需要 GPU**：适配层的逻辑是"调用下层接口、传对参数、解析返回值"。用 fake 实现模拟真实接口签名即可完整验证适配层。真实 GPU 只是最后的端到端冒烟。

**实验矩阵（后续）**：单节点基线 → 两节点 KV 迁移 → 节点离开恢复（对比优先级迁移 vs 全传 vs 全丢）→ 专家覆盖路由。指标：KV 命中率、TTFT/TPOT、节点离开恢复时间、带宽占用。

## 5. 与相关工作对比

| 参考 | 我们借鉴 | 我们不同 |
|---|---|---|
| Preble | 独立调度层 + OpenAI API 驱动 vLLM；exploit/explore 决策 | KV 存储/传输抽象为接口可换后端；边缘场景加传输成本维度 |
| SpotServe | 重并行化求解 + 迁移指令 | asyncio 控制面；迁移优先级用 PT×N 而非贪心 |
| MoE-Infinity | 专家分级装载 | 专家装载收进 ModelManager 统一接口 |
| Helix | 图建模 + 机制/策略分离 | 机制更薄，复杂算法全走策略接口 |

## 6. 预期贡献

1. **机制**：一个接口完整、可对接 vLLM+LMCache、可扩展到真实硬件与仿真验证的管理调度框架。
2. **E2 缓存感知路由**：把"KV 命中 + 负载均衡"的权衡形式化为 placement 策略，适配边缘带宽受限场景。
3. **PT×N 优先级迁移**：把"节点离开时迁什么"形式化为带时间预算的优先级分配，支持边缘无宽限期场景的部分迁移。
4. **验证框架**：三层验证（逻辑/契约/集成），对接测试无需 GPU。
