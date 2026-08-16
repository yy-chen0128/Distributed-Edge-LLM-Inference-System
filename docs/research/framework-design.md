# 框架设计说明：edge_llm_scheduler — 分布式算力调度层

> 日期：2026-08-16
> 代码位置：`project/edge_llm_scheduler/`
> 设计依据：论文池分析（Helix/SpotServe/Preble/MoE-Infinity/LMCache）+ 相关开源实现代码分析
> 状态：框架机制层完整实现，38 个测试通过；策略层为接口+默认实现（复杂算法后续补充）

---

## 一、设计目标与定位

在边缘异构算力节点（GPU/手机/边缘盒）上执行 LLM 推理。本框架是**管理调度层**——
它不碰矩阵运算，只做"请求发给谁、模型参数/KV 迁到哪、节点怎么组织"的决策，并通过对接接口执行。

**核心设计原则：机制（mechanism）与策略（policy）分离。**
- **机制**（`core/`）：框架必须完整实现——对接接口、数据结构、流程编排。缺了它系统跑不起来。
- **策略**（`policies/`）：可插拔算法——具体怎么做由策略决定，默认实现占位，复杂逻辑后续补充。

**为什么这样分**：论文池的复杂算法（E2 缓存感知路由、PT×N 迁移优先级、KM 重并行化）都集中在
"决策"上。把它们做成策略接口，框架骨架可以先完整搭好并验证，算法再逐个替换。

## 二、架构分层

```
┌─────────────────────────────────────────────┐
│ 管理调度层（本框架）                          │
│   core/     机制层：类型/存储/传输/事件/节点/  │
│            模型装载/任务调度/请求流            │
│   policies/ 策略层：放置/迁移/重并行化/恢复     │
│   backends/ 对接：mock / vLLM / LMCache /     │
│            TCP(有线) / WiFi(无线)            │
├─────────────────────────────────────────────┤
│ 推理引擎层：vLLM（backends/vllm_engine.py）    │
│ KV 缓存层：LMCache（backends/lmcache_storage.py）│
│ 传输层：TCP 有线 / WiFi 无线（backends/）      │
└─────────────────────────────────────────────┘
```

## 三、模块与职责

| 模块 | 文件 | 职责 |
|---|---|---|
| 核心类型 | `core/types.py` | Node/Capability/State、ModelSpec/Placement、KVBlock、Request/Task/Result、Event |
| KV 存储抽象 | `core/storage.py` | `KVStore` 接口：save/load/lookup/move/evict/get_index/estimate_move_cost |
| 传输抽象 | `core/transport.py` | `Transport` 接口：push/pull/measure_bandwidth/estimate_transfer_time |
| 事件总线 | `core/event_bus.py` | 异步事件分发（节点/请求/任务事件），Preble 队列模式的泛化 |
| 节点管理 | `core/node_manager.py` | 注册/心跳/状态/超时检测，发布 JOINED/LEFT 事件 |
| 模型装载 | `core/model_manager.py` | 控制装哪些层/哪些专家/专家进 gpu/cpu/disk 分级存储 |
| 任务调度 | `core/task_scheduler.py` | 路由→下发→回收；节点事件触发策略；中断恢复 |
| 请求流 | `core/request_flow.py` | 请求生命周期编排 + 计时 |
| 放置策略 | `policies/placement.py` | 请求去哪个节点（默认：负载最轻+KV命中优先） |
| 迁移策略 | `policies/migration.py` | 节点离开时 KV/参数去向（默认：全传） |
| 重并行化 | `policies/reparallelization.py` | 节点变化后放置调整（默认：不变） |
| 中断恢复 | `policies/recovery.py` | 任务中断重执行（默认：重跑） |
| Mock 后端 | `backends/mock_*` | 无硬件验证 |
| 真实后端 | `backends/vllm_engine/lmcache_storage/tcp_transport/wifi_transport` | 有硬件对接 |

## 四、关键接口设计

### 4.1 KV 存储抽象（对接 LMCache）
```python
class KVStore(ABC):
    async def save(self, block: KVBlock, location: str) -> None
    async def load(self, block_hash: int, location: str) -> Optional[KVBlock]
    async def lookup(self, prefix_hashes: List[int], locations=None) -> int  # 连续命中块数
    async def move(self, block_hash: int, src: str, dst: str) -> None
    async def evict(self, block_hash: int, location: str) -> None
    async def get_index(self) -> Dict[int, List[str]]  # 全局索引（缓存感知路由）
    async def estimate_move_cost(self, block, src, dst) -> float
```
- `location` 是不透明字符串（`node_a:gpu`/`node_a:cpu`/`node_a:disk`），底层解释
- **不直接依赖 LMCache 的 StorageBackendInterface**（它耦合 torch/MemoryObj）——框架自定义轻量接口，LMCache 作为一个 backend 实现，可测、可换

### 4.2 传输抽象（链路抽象：有线/无线）
```python
class Transport(ABC):
    bandwidth_mbps: float
    async def push(self, data: bytes, dst_node: str, tag: str) -> None
    async def pull(self, src_node: str, tag: str) -> Optional[bytes]
    async def measure_bandwidth(self, dst_node: str) -> float
    async def estimate_transfer_time(self, data_bytes: int, dst_node: str) -> float
```
- 调度器只依赖抽象，不知道底层有线/无线——适配在 backend

### 4.3 模型装载（用户明确要求的能力）
```python
class ModelManager:
    async def load_model(self, placements: List[ModelPlacement]) -> None
    async def load_layers(self, node_id, layer_range, source="cloud") -> None
    async def load_experts(self, node_id, expert_ids, tier="gpu")  # tier: gpu/cpu/disk
    async def unload(self, node_id, layer_range=None, expert_ids=None) -> None
```
- `ModelPlacement` = node_id + layer_range + experts_gpu + experts_cpu
- 支持"装哪些层、哪些专家、专家进哪级存储"的精细控制

### 4.4 任务调度编排
```python
class TaskScheduler:
    async def handle_request(self, req) -> GenerationResult   # 完整生命周期
    async def route_request(self, req) -> List[Task]          # 策略路由
    def attach_engine(self, node_id, engine) -> None          # 挂引擎
    # NODE_LEFT → 迁移策略；NODE_JOINED → 放置/重并行化策略
```

## 五、验证分层（关键：对接测试不需要 GPU）

| 层级 | 方法 | 覆盖 | 状态 |
|---|---|---|---|
| **L1 框架逻辑** | mock 后端 | 请求生命周期/节点事件/模型装载/KV管理/传输抽象 | ✅ 20 测试 |
| **L2 对接契约** | fake 后端验证适配层 | VLLMEngine 请求格式+解析+错误；LMCacheStore 方法调用+参数；TCP/WiFi loopback | ✅ 18 测试 |
| **L3 真实集成** | 真实 vLLM+LMCache+GPU | 端到端 | ⏳ 需 GPU 环境 |

**为什么 L2 不需要 GPU**：适配层（VLLMEngine/LMCacheStore）的逻辑是"调用下层接口、传对参数、解析返回值"。
用 fake 实现（模拟真实接口签名）就能完整验证适配层——真实 GPU 只是最后的端到端冒烟。
- VLLMEngine：用 aiohttp fake OpenAI server 验证 `POST /v1/completions` 请求体、响应 JSON 解析、错误处理
- LMCacheStore：用 fake LMCacheEngine 验证 `store/retrieve/lookup/move/clear` 方法调用和参数
- TCP：loopback 真实 socket 验证 TcpReceiver 收到数据

**L2 测试明细**（`tests/test_vllm_engine.py` / `test_lmcache_store.py` / `test_tcp_wifi_transport.py` / `test_policy_interfaces.py`）：
- vLLMEngine：请求格式 / 错误处理 / 连接拒绝 / 状态查询
- LMCacheStore：store/load(命中+未命中)/lookup/move/evict/成本估计/未attach报错
- TCP/WiFi：真实传输 / 时延估算 / 带宽测量抖动 / 丢包
- 策略可插拔：自定义策略替换默认，框架正确调用

## 六、与相关工作的对比

| 参考 | 我们借鉴 | 我们不同 |
|---|---|---|
| **Preble** | 独立调度层 + OpenAI API 驱动 vLLM；请求队列异步模式 | 我们把 KV 存储/传输抽象为接口（可换后端），Preble 直接用 radix tree |
| **Helix** | 仿真器+真实系统双实现；机制/策略分离 | 我们机制更薄，复杂算法全走策略接口 |
| **SpotServe** | Python 决策+C++ 执行分层；迁移指令 | 我们用 asyncio 控制面，迁移逻辑留 MigrationPolicy |
| **MoE-Infinity** | 专家分级装载（GPU/CPU） | 我们把专家装载收进 ModelManager 统一接口 |
| **LMCache** | 高层 API（store/retrieve/lookup/move） | 我们包一层 KVStore 抽象，不直接依赖其 StorageBackendInterface |

## 七、策略实现状态与后续 TODO

### 已实装（2026-08-16）
| 策略 | 实现 | 逻辑 |
|---|---|---|
| **E2 缓存感知路由** | `policies/placement.py::E2Placement` | exploit（命中>剩余→去缓存节点）/ explore（去 prompt-aware 负载最轻，含驱逐代价）；前缀匹配用 KVStore.lookup，hash_fn 可注入 |
| **PT×N 优先级迁移** | `policies/migration.py::PriorityMigration` | 价值 = reuse_count × prefill_time_ms；按价值降序传，时间预算（deadline_ms）用完即丢 |

### 待补充（研究内容）
| 策略 | 当前 | 待补充 |
|---|---|---|
| 放置 | E2 已实现 | 按节点能力分任务；专家覆盖路由（MoE-Infinity/SMoE 参考） |
| 迁移 | PT×N 已实现 | token 级截断（极端时间紧按 token 传）；参数从云端 vs 设备权衡；冗余复制 |
| 重并行化 | 放置不变 | 节点变化后 (D,P,M,B) 调整（SpotServe solver）；流水线拓扑（HotSwitch） |
| 中断恢复 | 重跑 | token 级恢复（SpotServe stateful recovery）；跨节点重调度 |

> 实装发现的有价值 bug：`MigrationPolicy.execute` 的 `evict(bid, src="*")` 参数名与接口 `location` 不符，
> 被 `except: pass` 吞掉导致丢弃不生效。端到端测试抓到并修复（统一 `location="*"`）。这印证了端到端测试的必要性。

## 八、目录结构

```
project/edge_llm_scheduler/
├── __init__.py / config.py / cli.py
├── core/          # types/storage/transport/event_bus/node_manager/model_manager/task_scheduler/request_flow
├── policies/      # placement/migration/reparallelization/recovery
├── backends/      # mock_* + vllm_engine/lmcache_storage/tcp_transport/wifi_transport
└── tests/         # 38 个测试（L1 逻辑 + L2 契约）
```

## 九、验证运行

```bash
cd project/
PYTHONPATH=. python -m edge_llm_scheduler.cli --nodes 3 --requests 3   # 端到端演示
PYTHONPATH=. python -m pytest edge_llm_scheduler/tests/ -q            # 38 测试
```
