# edge_llm_scheduler — 分布式算力调度框架

在边缘异构节点（GPU/手机/边缘盒）上执行 LLM 推理的管理调度层框架。

## 架构分层

```
┌─────────────────────────────────────────────┐
│ 管理调度层（本框架 = 你要自研的层）            │
│   core/     机制层（类型/存储/传输/事件/节点/  │
│            模型装载/任务调度/请求流）          │
│   policies/ 策略层（可插拔算法：放置/迁移/    │
│            重并行化/中断恢复）                │
│   backends/ 对接实现（mock / vLLM / LMCache / │
│            TCP / WiFi）                      │
├─────────────────────────────────────────────┤
│ 推理引擎层：vLLM（backends/vllm_engine.py）   │
│ KV 缓存层：LMCache（backends/lmcache_storage.py）│
│ 传输层：TCP 有线 / WiFi 无线（backends/）     │
└─────────────────────────────────────────────┘
```

**机制 vs 策略**：机制（core/）是框架必须完整实现的——对接接口、数据结构、流程编排。
策略（policies/）是可插拔算法——具体怎么做由策略决定，默认实现占位，复杂逻辑后续补充。

## 快速开始

```bash
# 无硬件验证（mock 集群跑请求）
cd project/
PYTHONPATH=. python -m edge_llm_scheduler.cli --nodes 3 --requests 3

# 无硬件验证（真正按层区间串行执行的 pipeline mock）
PYTHONPATH=. python -m edge_llm_scheduler.experiments.run_layered_simulation --nodes 3

# 跑测试
PYTHONPATH=. python -m pytest edge_llm_scheduler/tests/ -v
```

## 核心接口速览

### KV 存储抽象（core/storage.py）—— 对接 LMCache
```python
class KVStore(ABC):
    async def save(self, block: KVBlock, location: str) -> None
    async def load(self, block_hash: int, location: str) -> Optional[KVBlock]
    async def lookup(self, prefix_hashes: List[int], locations=None) -> int  # 命中块数
    async def move(self, block_hash: int, src: str, dst: str) -> None
    async def evict(self, block_hash: int, location: str) -> None
    async def get_index(self) -> Dict[int, List[str]]  # 全局索引（缓存感知路由）
    async def estimate_move_cost(self, block, src, dst) -> float
```
实现：`MockKVStore`（dict）/ `LMCacheStore`（对接 LMCacheEngine）

### 传输抽象（core/transport.py）—— 链路抽象（有线/无线）
```python
class Transport(ABC):
    bandwidth_mbps: float
    async def push(self, data: bytes, dst_node: str, tag: str) -> None
    async def pull(self, src_node: str, tag: str) -> Optional[bytes]
    async def measure_bandwidth(self, dst_node: str) -> float
    async def estimate_transfer_time(self, data_bytes: int, dst_node: str) -> float
```
实现：`MockTransport` / `TCPTransport` / `WiFiTransport`

### 模型装载（core/model_manager.py）—— 控制层/专家/分级
```python
class ModelManager:
    async def load_model(self, placements: List[ModelPlacement]) -> None
    async def load_layers(self, node_id, layer_range, source="cloud") -> None
    async def load_experts(self, node_id, expert_ids, tier="gpu")  # tier: gpu/cpu/disk
    async def unload(self, node_id, layer_range=None, expert_ids=None) -> None
```
- `ModelPlacement`: node_id + layer_range + experts_gpu + experts_cpu
- 专家可装入 GPU 内 或 CPU/盘（下级存储）——满足"控制装哪些专家、进哪级存储"

### 任务调度（core/task_scheduler.py）
```python
class TaskScheduler:
    async def handle_request(self, req) -> GenerationResult   # 路由→下发→回收
    async def route_request(self, req) -> List[Task]
    def attach_engine(self, node_id, engine) -> None
    # 节点事件：NODE_LEFT→迁移策略，NODE_JOINED→放置策略
```

## 策略接口（policies/）

| 策略 | 接口 | 已实现 | TODO |
|---|---|---|---|
| 放置 | `PlacementPolicy.place()` | `DefaultPlacement`（负载最轻）；**`E2Placement`（缓存感知路由）** | 按节点能力分任务；专家覆盖路由 |
| 迁移 | `MigrationPolicy.decide()/execute()` | `DefaultMigration`（全传）；**`PriorityMigration`（PT×N 优先级+时间预算）** | token 级截断；参数从云端 vs 设备权衡；冗余复制 |
| 重并行化 | `ReparallelizationPolicy.reconfigure()` | **`CapabilityReparallelization`（按能力重切层）** | 节点变化后 (D,P,M,B) 调整 + 动态流水线拓扑 |
| 中断恢复 | `RecoveryPolicy.recover()` | **`TokenRecovery`（保留进度）** | 活节点重路由 + KV 恢复一致性 |

### E2Placement（缓存感知路由，参考 Preble）
```
if 命中 token 数 > 剩余非共享 token 数:  → EXPLOIT：送持有最长命中前缀的节点
else:                                     → EXPLORE：送 prompt-aware 负载最轻的节点
```
- EXPLOIT 省 prefill（KV 命中）；EXPLORE 均衡负载（含驱逐代价）。
- 前缀匹配用 `KVStore.lookup`；`hash_fn` 可注入（测试/真实可换）。

### PriorityMigration（PT×N 优先级迁移）
```
优先级(块) = reuse_count × prefill_time_ms    # 重算成本 × 共享热度
按优先级降序传，时间预算（deadline_ms）用完即丢
```
- `reuse_count`：历史共享请求数；`prefill_time_ms`：重算这段 KV 的耗时。
- 时间受限（边缘无宽限期）时只传高价值块，低价值块丢弃（重算便宜）。

## 分层执行边界

`LayeredPipelinePlacement + LayeredMockEngine` 用于无 GPU 验证：每个 stage
只执行自己的 `layer_range`，stage 之间通过 `Transport.push/pull` 传递并校验真实
activation payload，并为每段层区间登记独立 KV 块。`PipelineReconfigurationCoordinator`
以 epoch 执行预装、KV 副本、切流和 drain，支持在节点能力/成员变化后重分层。
这是真实的控制流与状态生命周期模拟，但不执行矩阵运算。

`VLLMEngine` 是 OpenAI HTTP 客户端。它可以调用一个已经由 vLLM 自己配置好
`--pipeline-parallel-size` 的完整分布式实例，但不能把一个请求拆成多个独立
HTTP 端点来执行层区间；如果误把多个 stage 任务交给它，会明确报错。

vLLM 源码边界、静态 PP 启动方式与本项目的低耦合重配置方案见
[`docs/research/vllm-edge-integration.md`](../docs/research/vllm-edge-integration.md)。

当前实现状态、旧卡多卡验证方案和后续路线见
[`docs/plan/current-work-and-roadmap.md`](../docs/plan/current-work-and-roadmap.md)。

## 真实对接（有硬件时）

- **vLLM**：`VLLMEngine` 走 OpenAI API（`POST /v1/completions`）。启动时配
  `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`
- **LMCache**：`LMCacheStore` 对接 `LMCacheEngine`（store/retrieve/lookup/move）。
  注意跨进程统一 `PYTHONHASHSEED`（KV 键是 token 哈希）。
- **命名避让**：不使用 LMCache 自带名字（LMCacheConnectorV1/kv_role/pd_role 等）。

## 目录结构

```
edge_llm_scheduler/
├── config.py          # 配置加载
├── cli.py             # 演示入口
├── core/              # 机制层
│   ├── types.py       # Node/Model/Request/Task/KVBlock/Event
│   ├── storage.py     # KVStore 抽象
│   ├── transport.py   # Transport 抽象
│   ├── event_bus.py   # 异步事件总线
│   ├── node_manager.py# 节点生命周期
│   ├── model_manager.py # 模型装载
│   ├── task_scheduler.py # 任务调度
│   └── request_flow.py  # 请求生命周期
├── policies/          # 策略层（接口+默认实现）
├── backends/          # mock/vLLM/LMCache/TCP/WiFi
└── tests/             # 当前 76 个测试（mock/CPU/Torch tensor 验证）
```
