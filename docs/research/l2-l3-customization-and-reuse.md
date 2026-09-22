# L2/L3 自定义边界与复用评估（vLLM / LMCache 及参考实现）

> 回答三个问题：**① L2/L3 我们能自定义到什么程度？② 已有实现是什么设计？③ 适不适合我们的场景？**
> L1（模型与算法层）按要求不涉及。
>
> 方法：结论全部来自本地 clone 的源码（`project/vllm`，dev 版；`project/LMCache`），
> 每条给 `路径::符号` 级证据。**未在本仓核实的部分会显式标注**（如 SGLang 未 clone）。
> 前置约束来自我们已实测的结论：4 张消费级笔记本 GPU（本机 RTX 4060 Laptop = **sm89/Ada**，
> 其余机器可能落在 sm75/sm86；均无 NVLink，WiFi/局域网）、
> PP 分段的 activation 极小（7B decode 7KB/token）、PP 不降低单请求延迟、重配置成本主要是装载。

---

## 0. 结论速览

### 0.1 自定义程度（三档）

| 档位 | 手段 | 能改到哪 | 代价 |
|---|---|---|---|
| **A. 配置级**（不改代码） | CLI/配置字段 | 并行度、PP 层区间(静态)、KV 传输/卸载、调度器开关、编译、量化、attention 后端、权重 offload | 低；但改动只影响"启动时的形状" |
| **B. 注册/插件级**（写代码但不 fork） | entry points + 注册函数 + 类名注入 | 自定义 **Scheduler**、**Worker 扩展方法**、**KV Connector**（可从 module path 加载）、**Executor**、**Attention Backend**、**量化方法**、**平台/设备**、模型 | 中；需要跟随 vLLM 内部接口，有版本耦合 |
| **C. fork 级** | 改内部执行路径 | 运行中改变 PP 拓扑、layer 级权重的运行中装载/卸载、把 PP 的 activation 换成外部链路 | 高；涉及 worker/model_runner/parallel_state/KV 生命周期 |

**关键判断：我们真正需要的能力里，一半在 B 档就能做（尤其 KV 迁移与调度策略），
只有"运行中改 PP 拓扑 / layer 级权重热插拔 / 外部 activation 链路"必须进 C 档。**

### 0.2 已有实现的设计（一句话各表）

| 实现 | 设计要点 | 对我们的价值 |
|---|---|---|
| vLLM EngineCore | Scheduler（调度）与 Worker/ModelRunner（执行）分离，中间用 `SchedulerOutput` 契约 | 是"调度层与执行层分离"的工业级范本；也解释了为什么它的调度器可替换（B 档） |
| vLLM KV Connector | **按层粒度**的 KV 加载/保存钩子（`wait_for_layer_load(layer_name)`/`save_kv_layer(layer_name,...)`）+ 调度器侧前缀命中查询 | **直接命中我们的"层粒度部分 KV 迁移"需求**，是最值得复用的扩展面 |
| vLLM Offload（节点内） | `cpu_offload_gb`（UVA 零拷贝）/ `PrefetchOffloadConfig`（按层组 offload + 异步预取，通过 patch module forward 插入 custom op） | 证明"层组驻留 + 预取"已被工业实现；但介质是 PCIe，**不是网络**——这正是我们的空间 |
| LMCache | 前缀分块的 KV 缓存层。**存储层原生逐层**（键含 `layer_id`、逐层 allocate/put/get）、分层压缩（CacheGen int8）、多层后端（CPU/disk/remote/P2P）、可插拔存储后端与传输通道、带 worker 心跳的缓存控制器、PD 传输。**但上层 API 只认"全层命中"**（`lookup` 显式判定），且**无外部接口可注入 KV 张量**→ 只能当进程内嵌库 | 可复用的是**存储层 + 键方案 + `lm://` 纯 TCP 跨机通道**；层区间/迁移预算语义要我们在上层重建 |
| 参考实现 | SpotServe（C++ ParamsClient + Context Daemon 做权重/KV 迁移）、MoE-Infinity（专家 offload+prefetch）、Preble（前缀感知调度） | 都是"进入执行引擎内部"的证据：动态调度不是请求路由层能解决的 |

### 0.3 是否适合我们（速判）

| 我们的需求 | 复用点 | 适合度 |
|---|---|---|
| 连续批处理 + 前缀缓存（吞吐基线） | 直接用 stock vLLM（A 档）：`enable_prefix_caching` 默认 True + `BlockPool.cache_full_blocks` + `KVCacheManager.get_computed_blocks` | ✅ 最适合，直接当 baseline |
| **KV 按层粒度迁移** | 自定义 **KV Connector**（`kv_connector_module_path`，B 档）+ 逐层钩子 + `kv_load_failure_policy=recompute`；且**已有 PP 感知的层区间寻址先例**（`EngineTransferInfo.start_layer/end_layer`、NIXL push producer 侧 PP>1） | ✅ 接口层粒度是真的，值得优先做；增量在"双向 + 按价值 + 有预算" |
| 自定义调度策略（异构/命中/带宽感知） | `--scheduler-cls`（B 档） | ⚠️ 可行，但源码声明"接口非公开、兼容性不保证"；且只能调度 vLLM 内部请求 |
| 节点内分级装载（显存不够） | `OffloadConfig`（UVA / prefetch，A 档） | ✅ 现成，可当"跨节点分级"的对照基线 |
| **层粒度权重驻留（hot swap）** | 自写 `SleepModeBackend`（经 `vllm.general_plugins` 注册）+ 每层打 CuMem tag 复用 `wake_up(tags)` | ⚠️ 唯一可能的免 fork 路径，**未端到端验证**；否则要改 `device_allocator/` |
| 运行中改 PP 拓扑 / 跨节点 activation | 无 B 档方案；PP transport 无 hook（唯一旁路被 `is_cpu()` 锁死） | ❌ 必须自研 stage runtime；
**且原生 vLLM 里"分层跨机(PP)" 与 "节点离开缩容(elastic EP)" 硬互斥**（`parallel.py:846-850`） |
| MoE 专家放置 / 跨节点 offload | 运行中重排可用（EPLB `rearrange` + 可插拔通信后端含 **NIXL**）；**但跨节点常驻专家池不存在**，且 elastic EP 与 PP 互斥 | ⚠️ 重排可复用，跨节点池需自研 |
| 我们的消费级 GPU 上用最快 kernel | FA3/FA4 不适用；4060(sm89) 可用 FA2/Triton/FlashInfer/FP8-CUTLASS，3060(sm86) 的 FP8 基本无望（见 §3.3） | ⚠️ 影响性能预期与量化选择，不影响机制 |

---

## 1. 判断依据：我们已实测的约束

（来自 `docs/deploy/edge-4gpu-deployment-analysis.md`，此处只列对 L2/L3 选择有影响的）

1. **PP 的 activation 极小**：7B decode 每 token 每边界 7KB（fp16）→ 跨机链路不是 PP 的瓶颈。
2. **PP 不降低单请求延迟**：4 段 147.5ms/token vs 单机 128.5ms/token（同线程数）。
3. **重配置成本 = 权重装载**（2–8s），不是协议（0.7s）→ 优化点在装载/缓存/量化。
4. **并发只到 1.55×**（K=1→4）→ 缺连续批处理；这是所有吞吐实验的地基。

**推论**：我们需要从 L3 拿的是"**批处理 + KV 生命周期 + 可插拔调度**"，
从 L2 拿的是"**能在 sm86/sm89 上跑得动的 kernel 与量化**"；
需要自己造的是"**跨节点的层放置与 activation 链路**"——这三条正好对应 B 档、A 档、C 档。

---

## 2. L3 引擎与执行层

### 2.1 vLLM V1 的职责划分

```
请求 → API/入口 → EngineCore(进程内循环)
        ├── Scheduler        ← 决定这一轮跑哪些请求、分配多少 token/KV 块（可替换：--scheduler-cls）
        │     └── KVCacheManager / BlockPool（分页 KV）
        ├── KVConnector（调度器侧一半）← 前缀命中查询：get_num_new_matched_tokens()
        └── Executor         ← 进程/actor 编排（可替换：distributed_executor_backend）
              └── Worker × N ← 模型加载、KV cache 初始化、execute_model（可扩展：--worker-extension-cls）
                    └── ModelRunner → attention backend（可替换：register_backend）
                          └── KVConnector（worker 侧一半）← start_load_kv/save_kv_layer/wait_for_*
```

证据：`vllm/v1/engine/core.py::EngineCore`（`Scheduler = vllm_config.scheduler_config.get_scheduler_cls()`，line 148）、
`vllm/v1/executor/abstract.py::Executor.get_class`、`vllm/v1/worker/worker_base.py`、
`vllm/v1/worker/gpu_model_runner.py`。

**设计含义**：vLLM 把"调度决策"和"执行"用 `SchedulerOutput` 这个契约切开，
这正是**我们能插进去的地方**——但也正因为它只认自己的 `SchedulerOutput`，
外部自定义的"跨机 stage 任务"无法直接塞进去（见 §2.6）。

### 2.2 免 fork 的扩展点（B 档，逐条已核实）

| 扩展点 | 用法 | 源码位置 |
|---|---|---|
| **自定义 Scheduler** | `--scheduler-cls <QualName>` | `vllm/config/scheduler.py:117,170`（`get_scheduler_cls()` → `resolve_obj_by_qualname`）；装配点 `vllm/v1/engine/core.py:148` |
| **Worker 扩展类** | `--worker-extension-cls <QualName>`（给 worker 增加自定义 RPC 方法） | `vllm/config/parallel.py:265`；合并逻辑 `vllm/v1/worker/worker_base.py:265-288`（含方法名冲突检查） |
| **自定义 Executor** | `distributed_executor_backend="<module.Class>"` | `vllm/v1/executor/abstract.py:83-87` |
| **自定义 KV Connector** | `kv_transfer_config.kv_connector` + **`kv_connector_module_path`** | `vllm/config/kv_transfer.py:26,62` |
| **自定义 Attention Backend** | `@register_backend(AttentionBackendEnum.CUSTOM)` | `vllm/v1/attention/backends/registry.py:129`（`CUSTOM = None`）、`:242 register_backend()`（docstring 明确给出三方后端用法） |
| **自定义量化方法** | 量化注册表 | `vllm/model_executor/layers/quantization/`（`Fp8Config` 等；`get_min_capability()` 见 §3.3） |
| **自定义平台/设备** | entry point `vllm.platform_plugins`；`Platform` 基类 | `vllm/plugins/__init__.py:23`、`vllm/platforms/interface.py` |
| **通用插件** | entry point `vllm.general_plugins`（+ endpoint 插件组） | `vllm/plugins/__init__.py:18,36,77` |
| **自定义算子** | `CustomOp`/`PluggableLayer`；`direct_register_custom_op` | `vllm/model_executor/custom_op.py:32,103`；`vllm/utils/torch_utils.py:1026` |

**补充：本轮普查又核实到一批同样免 fork 的扩展点**（都可能被我们直接利用）：

| 扩展点 | 用法 / 字段 | 源码位置 | 对我们的用处 |
|---|---|---|---|
| **Sleep 后端可插拔** | 自定义 backend 经 `vllm.general_plugins` 注册；选择字段 `model_config.sleep_mode_backend` | `vllm/device_allocator/sleep_mode_backend.py:142-196`（docstring 明说"lets third-party backends register"）、`vllm/config/model.py:330` | ★ 见 §2.4：这是"层粒度驻留"唯一可能的免 fork 路径 |
| **KV offload spec 可插拔** | `OffloadingSpecFactory`（`spec_module_path`）+ `SecondaryTierFactory`（docstring 逐字"skip registration entirely and pass a module_path at lookup time (out-of-tree, no vLLM fork/patch required)"） | `vllm/v1/kv_offload/factory.py:14`、`vllm/v1/kv_offload/tiering/factory.py:16-24` | 自写 KV 分级存储层 |
| **KV Connector 工厂** | `KVConnectorFactory.register_connector`；**外部模块优先于内建注册表**；构造签名必须 3 参（含 `kv_cache_config`）否则 ValueError | `vllm/distributed/kv_transfer/kv_connector/factory.py:27,31,105-107,115-123` | 自写 connector 的落地细节 |
| 权重传输引擎 | `WeightTransferEngineFactory.register_engine`；`weight_transfer_config.backend` | `vllm/distributed/weight_transfer/factory.py:41`、`vllm/config/weight_transfer.py:12` | 按参数名 chunk 寻址（见 §2.4） |
| 量化 / 模型加载器注册 | `register_quantization_config`、`register_model_loader`、`ModelRegistry.register_model` | `vllm/model_executor/layers/quantization/__init__.py:58`、`vllm/model_executor/model_loader/__init__.py:66`、`vllm/model_executor/models/registry.py:1083` | 低优先 |
| 插件 entry point 组（共 6 个） | `vllm.general_plugins` / `platform_plugins` / `io_processor_plugins` / `stat_logger_plugins` / `endpoint_plugins` / `logits_processors` | `vllm/plugins/__init__.py:18-30`、`vllm/v1/sample/logits_processor/__init__.py:48` | 挂载点 |
| 平台插件细节 | **没有 `register_platform()` 函数**；靠 entry point 返回类全限定名，OOT 优先于内建且**只能激活一个** | `vllm/platforms/__init__.py:229-254`；基类 `vllm/platforms/interface.py::Platform:134`（可覆盖 `pre_register_and_update:546`、`get_attn_backend_cls:373`、`register_custom_kv_cache_specs:941`…） | 若要引入新设备类型 |

> ⚠️ **一个必须写进风险清单的警告**：`--scheduler-cls` 虽然存在，但源码自带免责声明——
> `vllm/config/scheduler.py:182-186`「This scheduler interface is not public and compatibility
> may not be maintained」。**即"免 fork"不等于"接口稳定"**：自定义 Scheduler 的维护成本
> 要按"每次升级 vLLM 都可能改"来估。抽象接口见 `vllm/v1/core/sched/interface.py::SchedulerInterface:38`
> （`schedule` / `add_request` / `finish_requests` / `update_from_output` / `pause_state` … 13 个抽象方法）。
>
> 同理，自定义 Executor 若要支持 PP，必须自己设 `supports_pp = True`
> （`vllm/v1/executor/abstract.py:46`；内建 `multiproc_executor.py:112` 等三处如此声明），
> 否则 `vllm/engine/arg_utils.py:2535-2550` 直接抛错。

### 2.3 KV Connector：与我们需求最贴合的扩展面

`vllm/distributed/kv_transfer/kv_connector/v1/base.py::KVConnectorBase_V1`，
**调度器侧**（抽象方法）：

| 方法 | 作用 |
|---|---|
| `get_num_new_matched_tokens(request, num_computed_tokens)` | 问外部存储：这个请求的前缀**已经有多少 token 的 KV 可用**（前缀命中） |
| `update_state_after_alloc(request, blocks, num_external_tokens)` | KV 块分配后同步状态 |
| `build_connector_meta(scheduler_output)` | 生成本轮传给 worker 的连接器元数据 |

**Worker 侧**（抽象方法）：

| 方法 | 作用 | 为什么对我们重要 |
|---|---|---|
| `start_load_kv(forward_context, **kwargs)` | 本轮开始加载 KV | 可挂"从别的机器取 KV" |
| **`wait_for_layer_load(layer_name)`** | **按层**等待该层 KV 加载完成 | **接口本身是层粒度的**——正好对上"每台机器只负责自己那几层" |
| **`save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)`** | **按层**保存 KV | 层粒度导出，支持"只搬某段层的 KV" |
| `wait_for_save()` | 等保存完成 | |

另有一批非抽象但有默认实现的能力：`register_kv_caches`（拿到 KV cache 张量）、
`register_cross_layers_kv_cache`、`set_host_xfer_buffer_ops(CopyBlocksOp)`（host 侧拷贝算子注入）、
`handle_preemptions`、`get_block_ids_with_load_errors`、
`set_xfer_handshake_metadata_pp_aware`（**PP 感知的握手元数据**）、
`bind_gpu_block_pool`、`get_required_kvcache_layout`、`supports_hma`（异构内存分配器）。
角色由 `KVConnectorRole.SCHEDULER/WORKER` 区分（`base.py:124`）。

**由此得到的三个具体结论：**

1. **"按层粒度迁移 KV"不需要 fork**——`save_kv_layer(layer_name, ...)` / `wait_for_layer_load(layer_name)`
   就是层粒度的钩子。我们要做的是写一个 connector，把本节点负责的层区间映射到对端节点。
2. **"节点离开导致 KV 取不到"有官方策略开关**：`KVTransferConfig.kv_load_failure_policy = "recompute" | "fail"`
   （`vllm/config/kv_transfer.py:69`），实现在 `vllm/v1/core/sched/scheduler.py:148-149`
   （`self.recompute_kv_load_failures = policy == "recompute"`）。
   这正好对应我们论文里的"KV 抢救 vs 重算"权衡——**vLLM 已经把它做成了一个布尔策略**，
   我们的增量在于"按价值排序的部分迁移 + 迁移预算"，而不是从零做这个开关。
3. **前缀命中路由的官方钩子**是 `get_num_new_matched_tokens`——我们的 `E2Placement`
   （exploit/explore）如果要接生产引擎，这就是接口。

**但必须注意 KV 的寻址粒度**（决定我们的 `KVBlock(layer_range, tokens)` 能不能直接映射）：

- **层维度**：由"你传的是哪一层的 KV 张量"表达（`save_kv_layer(layer_name, kv_layer, …)`）→ 与我们的
  `layer_range` 天然对应，**层粒度没问题**。
- **token 维度**：是 **(block_id, block_offset) 的槽位寻址，且必须按 `block_size` 对齐**
  （`example_connector.py::ReqMeta.make_meta` 里 `align_to_block_size(...)` + 由 block_ids 生成 slot_mapping）。
  → 我们的"任意 token 区间"要落到"块对齐的区间"，**不能任意切**。
- **布局维度**：connector 可以**要求**一种 KV cache layout
  （`KVConnectorBase_V1.get_required_kvcache_layout()`，docstring 举例 `HND`/`NHD`，`base.py:611`）
  → 跨节点搬运要求两端布局一致，否则要先转置。
- **可调项**：`CacheConfig.cache_dtype`（KV 量化，`config/cache.py:77`）、
  **`kv_cache_dtype_skip_layers`**（按层号或 attention 类型跳过量化，`:114`）、
  `num_gpu_blocks_override`（显存紧张时控制 KV 池大小，`:90`）。
  → "弱网下压缩 KV 再传"在引擎侧有现成开关可做对照，我们只需在**跨机传输**这一段做文章。

**逐层钩子的唯一调用点**（说明它确实在每层 attention 前后被调用，不是摆设）：
`vllm/model_executor/layers/attention/kv_transfer_utils.py::maybe_transfer_kv_layer`
（装饰器包住 attention 层）——`:51 connector.wait_for_layer_load(layer_name)` →
执行 attention → `:57 connector.save_kv_layer(layer_name, kv_cache, attn_metadata)`。
两侧调用点：scheduler 侧在 `v1/core/sched/scheduler.py:830,1061,1320,2698`；
worker 侧在 `v1/worker/kv_connector_model_runner_mixin.py:89-103`
（`bind_connector_metadata` → `start_load_kv` → … → `wait_for_save` → `get_finished`）。

#### 已实现的先例："按层区间寻址的远端 KV"在 vLLM 里**已经存在**

- `vllm/distributed/kv_transfer/kv_connector/utils.py::EngineTransferInfo:401-405` 就带
  `start_layer` / `end_layer`（注释逐字："Global index of the first layer owned by this PP rank"），
  `TransferTopology:412` 按 `(engine_id, pp_rank)` 存远端引擎信息。
- NIXL push 连接器**就是为"PP>1 的生产者持有一段连续层 → push 到 PP=1 远端的对应子区间"设计的**
  （`v1/nixl/base_worker.py:440-455` 注释逐字）。**但 decode 侧 PP>1 是 `NotImplementedError`**（`:451-455`），
  且 HMA 布局下 PP>1 也不支持（`:445-449`）。

> 意义：需求 4（层粒度部分迁移）**不是"没人做过"**，而是"已有 PP 感知的层区间寻址先例，
> 但只在 prefill→decode 单向、且 decode 侧不支持 PP"。我们的增量应落在
> **"节点离开时的双向、按价值、有预算的层区间迁移"**，而不是从零造层区间寻址。

#### 另一条 KV 搬运路线（`v1/kv_offload/tiering/p2p/`）—— 已核实**做不到层区间**

`vllm/v1/kv_offload/` 是一套**完全绕开 torch.distributed 的 KV P2P 交换子系统**，
有文档化的抽象基类：`p2p/data/base.py::DataTransport:96`（用法示例
`write_blocks("peer:1", local_idxs=[0,3], remote_idxs=[5,7])`，约定"write_blocks 不得阻塞、无后台线程、全靠 poll()"）、
`p2p/control/base.py::ControlTransport:113`（peer 发现与消息路由，对内容不透明）。
但它的键粒度是 **`OffloadKey = block_hash + group_idx`**（`kv_offload/base.py:26-31`），
`group_idx` 是 KV cache group 而**不是层区间** → **走 offload 路线拿不到"按层区间选择"**。
（transport 在 `tiering/p2p/manager.py:291,298` 写死为 Nixl/Zmq，但 `SecondaryTierFactory` 支持
out-of-tree `module_path`，可零 fork 挂自研实现。）

#### 内建 connector 清单（16 条，挑对我们有用的）

| connector | 作用 |
|---|---|
| `example_connector.py::ExampleConnector` | 最简 debug 实现，**同时演示 load 与 store 两个方向**（逐层从 paged buffer 抽出后 `safetensors.save_file` 落盘）→ **写自研 connector 的最佳模板** |
| `lmcache_connector.py` / `lmcache_mp_connector.py` | 接 LMCache（进程内 / 多进程模式，见 §4.6） |
| `offloading_connector.py` + `simple_cpu_offload_connector.py` | KV 分层卸载门面（转发到 `v1/kv_offload/`）；后者带 CPU/disk 后端与 BlockPool LRU，默认 8GB CPU 池 |
| `multi_connector.py::MultiConnector` | 组合器：**从第一个报告可用的 connector 加载、保存到所有 connector**（`Load KV from the first connector that advertises available tokens ... Save to all connectors.`） |
| `nixl/*` | RDMA 路线（PP 感知的层区间 push，见上） |
| `flexkv_connector.py` | 外部 FlexKV 分布式 KV store |
| `decode_bench_connector.py` | **纯基准工具**：用 dummy 填 KV 模拟 PD 分离下长 ISL decode，无真实传输 |
| `example_hidden_states_connector.py` | **不是实例间搬 activation**：见 §2.6 的更正 |

### 2.4 权重驻留与分级：vLLM 的做法与我们的边界

`vllm/config/offload.py::OffloadConfig` 有两个后端：

| 后端 | 字段 | 设计 |
|---|---|---|
| UVA | `cpu_offload_gb`、**`cpu_offload_params: set[str]`** | 用统一虚拟寻址做零拷贝；`cpu_offload_gb` 被官方描述为"虚拟地增大显存"；`cpu_offload_params` **按参数名片段**选择性 offload（docstring 举例 `mlp.experts.w2_weight` 可用 `"experts"` 命中）→ 可做**专家选择性 offload** |
| Prefetch | `offload_group_size`、`offload_num_in_group`、`offload_prefetch_step`、`offload_params` | **按层组** offload（docstring 例：group=8/num=2 → 卸载 6,7,14,15,22,23…）+ **异步预取**；实现方式是在计算图里插入 `wait_prefetch`/`start_prefetch` 自定义算子并 patch module forward（见 `compute_hash()` 注释） |

**这就是我们的边界所在，请务必写进论文定位：**
vLLM 已经实现了"**层组 / 参数级驻留 + 预取**"，但介质是 **GPU↔CPU（PCIe）**，范围是**单节点内**。
我们要做的是把同一条轴延伸到 **节点↔节点（网络）**——问题形状相同（驻留账本 + 预取 + 释放），
但约束完全不同：带宽低 2–3 个数量级、没有 DMA/零拷贝、对端可能随时消失。
**"同一个机制、换一种介质、增加不可靠性"是清晰且可辩护的增量**。

#### 更正：`sleep`/`wake_up` 的粒度是**整个 engine**，不是层

| 事实 | 证据 |
|---|---|
| `sleep(level)` 的 level 是**裸 int，没有枚举**；语义只写在 docstring：0=只暂停调度、1=权重下到 host RAM 且丢弃 KV、2=丢弃全部显存 | `v1/worker/gpu_worker.py:203-219`；`v1/engine/core.py:871-875`；`device_allocator/sleep_mode_backend.py:56-58` |
| `wake_up(tags)` 的 tags **只有 `"weights"` 与 `"kv_cache"`**（CuMem 内存池 tag） | `gpu_worker.py:238-259`；`Executor.sleeping_tags = {"weights","kv_cache"}`（`v1/executor/abstract.py:336`） |
| 调用是 **collective_rpc 到每个 worker 一次** → 粒度=整 engine，**最细到 weights/kv_cache 两类** | `v1/executor/abstract.py:329`（`collective_rpc("sleep", ...)`）；HTTP 面 `entrypoints/serve/dev/sleep/api_router.py:21,32` |
| level 不做范围校验（`sleep(3)` 等价"level2 但保留 buffer"） | `sleep_mode_backend.py:125`（`offload_tags=("weights",) if level == 1 else tuple()`） |

**但机制是可插拔的**（这是需求 1"层粒度驻留"唯一可能的免 fork 路径，标注为**未端到端验证**）：
`SleepModeBackendFactory` 允许第三方经 `vllm.general_plugins` 注册自己的后端
（`device_allocator/sleep_mode_backend.py:142-196`，docstring 明说 "lets third-party backends register"），
由 `model_config.sleep_mode_backend` 选择；而 `CuMemAllocator` 的 tag 机制**支持按 tag 部分唤醒**
（`device_allocator/cumem.py:298` `if tags is None or data.tag in tags`）、
`use_memory_pool(tag=...)`（`:313`）可给不同模块打不同 tag。
→ **理论上**可以"每层一个 tag"，从而 `wake_up(tags=["layer.17"])` 实现层粒度驻留。
**但当前树上只注册了 `weights`/`kv_cache` 两个 tag（`"cumec"` 一个后端），这条路需要我们自己实现并验证。**

#### 其它两个相关事实

- **layer 级运行中 load/unload 不存在**：`gpu_model_runner.py:5643 reload_weights(weights_iterator, weights_path, is_checkpoint_format)`
  **没有 layer 参数**；`model_loader/reload/layerwise.py::initialize_layerwise_reload:84` 只是全量重载时的按层
  materialize（触发条件是"该层 checkpoint 权重到齐"）。**但"逐层 materialize 到设备"的原语已经在树里**
  （`reload/meta.py::materialize_layer` / `restore_layer_on_meta`），可以复用。
- **可寻址的层粒度权重搬运 = weight-transfer 引擎**：chunk 按**参数名**寻址
  （`distributed/weight_transfer/nccl_engine.py:86-88`），所以一个 chunk 可以正好是一层；
  HTTP 面 `entrypoints/serve/dev/rlhf/api_router.py:157-205`。**但方向是 trainer→inference 单向**，
  不是"推理节点之间互搬"。`WeightTransferConfig.backend` 只有 `nccl/ipc/sparse_nccl`，**RDMA 仍是 TODO**
  （`distributed/weight_transfer/base.py:190-191`）。
- **专家级有真正的运行中机制（且通信后端可插拔）**：EPLB `EplbState.rearrange():733`
  + `EplbCommunicator(ABC)`/`create_eplb_communicator:660`，后端 `torch_nccl|torch_gloo|**nixl**|pynccl`
  ——**`nixl` 后端不走 torch.distributed**，是可复用的专家权重点对点搬运通道；
  elastic EP 走 `elastic_execute.py::batch_transfer_weights:59`（本质是 rank 加入/退出时搬权重）。

### 2.5 Pipeline Parallel 的真实边界

| 事实 | 证据 |
|---|---|
| PP 层区间由**环境变量**静态决定；override 是"**每 rank 的层数计数**"（不是层号），必须**连续、整分**（`sum(partitions)==num_hidden_layers`，否则 `ValueError`） | `vllm/distributed/utils.py::get_pp_indices:127-171`（读 `envs.VLLM_PP_LAYER_PARTITION:143`）；`vllm/envs.py:50`；`vllm/config/model.py::get_layers_start_end_indices:1528` |
| 层分配落到模型树是**固定 ModuleList + 占位层** | `vllm/model_executor/models/utils.py:806-816`（真实层夹在 `PPMissingLayer()` 之间，`:773` 是 `nn.Identity`） |
| PP 通信**写死 torch.distributed，没有 hook**：`GroupCoordinator` 直连 `torch.distributed.isend/irecv`；`_PP` 由 `init_model_parallel_group:1902` 建立、非注入点；**唯一旁路 `use_cpu_custom_send_recv` 被 `current_platform.is_cpu()` 锁死**（CUDA 上恒 False） | `vllm/distributed/parallel_state.py:1067,1170,1902,529-533`；调用点 `v1/worker/gpu_worker.py:1098-1139`（`irecv/isend_tensor_dict`）、`gpu_model_runner.py:4612-4627` |
| **PP 不只是前向 activation**：last rank 会把 sampled logits **广播回所有 rank**；async 路径甚至裸调 `torch.distributed.broadcast` | `gpu_model_runner.py:4625-4629`；`:4964,4977`；`IntermediateTensors` 定义在 `vllm/sequence.py:11-18` |
| PP 是**per-model 硬门禁**（不是集中白名单）：模型须实现 `SupportsPP` 协议，否则配置期 `NotImplementedError` | `vllm/model_executor/models/interfaces.py:719`；`vllm/config/model.py:1377-1384` |
| **弹性 EP 与 PP 互斥** | `vllm/config/parallel.py:846-850`：`if self.pipeline_parallel_size > 1: raise ValueError("Elastic EP is not supported with pipeline parallelism ...")` |
| 容错是 **DP/EP-only，不感知 PP** | `v1/worker/sentinel/gpu_worker_sentinel.py:26`（`FT_BACKEND_SET = {"deepep_low_latency","nixl_ep"}`）、`:52-68`（retry 重建 DP group） |
| async scheduling + PP **允许但 V1 runner 下"不完整支持"**（退化而非报错）；chunked prefill 无不兼容 | `vllm/config/vllm.py:551-560`（注释逐字 "V1 Model Runner does not fully support async scheduling with PP"） |
| KV connector + PP **允许**（NIXL push 的 producer 侧 PP>1 即层区间 push；**decode 侧 PP>1 是 `NotImplementedError`**） | `v1/nixl/base_worker.py:440-455` |

**结论**：vLLM 的 PP 是"启动时固定、连续整分、torch.distributed 承载、per-model opt-in"的形态。
我们的"跨机 + 运行中重切 + 外部链路"三个特征它都不覆盖——这是自研 stage runtime 的正当性。
**并且这三条在原生 vLLM 里不能同时成立**：`elastic EP`（缩容/离开）与 `PP`（分层跨机）是硬互斥的，
所以"分层跨机 + 节点离开后缩容"这个组合**必须由我们的控制面承担**。

### 2.6 必须 fork（或自研）才能做的事

| 需求 | 为什么 B 档做不到 | 若 fork 要碰的文件 |
|---|---|---|
| 运行中改变 PP 拓扑（层区间 / world size） | PP group、model layers、worker rank 在初始化时固定；elastic EP 与 PP 互斥 | `vllm/config/parallel.py`、`vllm/distributed/parallel_state.py`、`vllm/v1/worker/gpu_model_runner.py`、`vllm/v1/executor/*` |
| **layer 级权重运行中装载/卸载** | `Worker` 抽象只有 `load_model`/`initialize_from_config`/`execute_model`；`reload_weights` **无 layer 参数**。**唯一可能的免 fork 绕法**：自写 `SleepModeBackend`（经 `general_plugins` 注册）+ 给每层打 CuMem tag，复用它按 tag 部分唤醒的能力（**未验证**，见 §2.4） | 若上述绕法不通，需改 `vllm/device_allocator/{cumem,sleep_mode_backend}.py` 让 tag 变层粒度；以及 `v1/worker/gpu_model_runner.py::reload_weights:5643` |
| 把 PP 的 activation 交给外部链路 | 通信写死在 `get_pp_group()` 的 `send/recv_tensor_dict`；`Executor.collective_rpc` 官方只建议传控制消息 | `vllm/distributed/parallel_state.py`、`vllm/v1/worker/gpu_model_runner.py`（或在 runner 层旁路） |
| 专家放到另一台机器上（跨节点 EP） | vLLM 的 EP 是"同一实例内多个 rank 分专家"，`enable_elastic_ep` 还需要 EPLB 且排斥 PP | 参考 MoE-Infinity 的做法自研专家驻留 runtime |

> ⚠️ **更正一处我上轮的说法**：`example_hidden_states_connector.py` **不是"实例间搬 activation"的示范**。
> 经全文核实：它是 **store-only**（`start_load_kv`/`wait_for_layer_load`/`save_kv_layer` 全是 `pass`），
> 真实工作发生在 `get_finished` 里把 hidden states 抽出来**写成 safetensors 文件**
> （`_submit_async_write:351-431`，专用 CUDA stream 做 DtoH + 线程池落盘），
> 供**外部客户端**用同文件的 `load_hidden_states(path):45`（flock 同步）读取；
> 且它**硬绑定 speculative decoding 的 `extract_hidden_states` 方法**
> （`assert self._vllm_config.speculative_config is not None`，`:186-189`；
> 依赖 `v1/spec_decode/extract_hidden_states.py::ExtractHiddenStatesProposer`
> 与 `CacheOnlyAttentionLayer:237`）。
> **对我们的需求 2 是"可借鉴的形状"，不是可复用的实现**——它没有 stage 间依赖驱动、没有流控、不走 PP。
>
> 顺带核实：**EC = Encoder Cache**（`distributed/ec_transfer/ec_connector/base.py:4-5` 逐字
> "Distributed Encoder Cache & P2P Encoder cache communication"），传的是**多模态 encoder 输出**
> （按 `mm_hash` 索引的 `encoder_cache`），典型用途是"把 vision encoder 从 LLM 实例剥离到专门实例"
> （`ec_transfer.py:39-41`："0 for encoder, 1 for pd instance. Currently only 1P1D is supported"），
> 且 `is_encode_only` 的实例**不分配 KV cache**。我们不涉及多模态，**价值低**——
> 但"把某类计算整体搬到另一台机器"的思路可参考。
> （另：`v1/executor/vllm_net_devices.py` **不是"网络设备当执行设备"**，只是 GPU→NIC 的 PCIe 亲和建议表，
> 用于给 RDMA 设 `UCX_NET_DEVICES`/`NCCL_IB_HCA`；对 WiFi 场景无意义。）

---

### 2.7 能力矩阵：一个实例（一张卡/一台机）到底能持有、能变化什么

> 本表只回答"**支持什么 / 不支持什么 / 边界在哪**"，不给实施建议。
> 标注：✅ 支持 · ⚠️ 部分支持（有条件） · ❌ 不支持。

#### A. 模型参数驻留

| # | 能力 | 判定 | 机制与证据 | 边界 |
|---|---|---|---|---|
| A1 | **一个实例只持有模型的若干层**（不是全部） | ✅ | PP：`get_pp_indices` 决定本 rank 的 `[start,end)`（`distributed/utils.py:127-171`）；模型树里非本段的层被替换为 `PPMissingLayer`（`models/utils.py:773,806-816`，`nn.Identity` 占位） | 必须**连续**且各 rank 层数之和 `== num_hidden_layers` |
| A2 | 层区间**任意/非连续**分配 | ❌ | override `VLLM_PP_LAYER_PARTITION` 的语义是"**每 rank 的层数**"而非层号（`utils.py:143-161`：`len(partitions)==pp_size`、`sum(partitions)==num_hidden_layers`），代码按前缀和切连续段 | 无法表达"机器 A 拿 3 和 17 层" |
| A3 | **运行中**改变层区间（加一层/少一层 → 改并行度） | ❌ | PP group（`parallel_state.py:1890-1902`）、模型层区间（`config/model.py:1528`）、worker rank 都在初始化时固定；运行中重配置请求体只覆盖 DP（`v1/engine/__init__.py::ReconfigureDistributedRequest:289-295` 全是 `new_data_parallel_*`） | 这是"改变并行度"在 vLLM 里的硬边界 |
| A4 | **一个实例只持有部分专家** | ✅ | EP：`compute_local_expert_ids(num_experts, ep_size, ep_rank, placement)`（`default_loader.py:399-404`）；`expert_map` 形状 `(global_num_experts,)`，**非本 rank 的专家填 `-1`**（`expert_map_manager.py:49-51,71-93`） | 专家是**均匀切**的：`base = N // ep_size`，前 `N % ep_size` 个 rank 多 1 个 |
| A5 | **专家放在哪台设备可指定** | ⚠️ | 只有两种策略：`expert_placement_strategy ∈ {linear, round_robin}`（`parallel.py:36,178`）；`linear`=每 rank 一段连续专家，`round_robin`=按 `ep_rank` 步长交错（`expert_map_manager.py:75-87`） | `round_robin` 还有额外条件：需 `num_expert_group>1`、无冗余专家、未开 EPLB、且 all2all 后端须为 DeepEP-LL 或 NIXL EP，否则**静默回落 linear**（`:116-147`） |
| A6 | **EP 度数独立设置** | ❌ | **没有 `expert_parallel_size` 字段**；`ep_size = dp_size × pcp_size × tp_size`、`ep_rank = dp_rank×pcp×tp + pcp_rank×tp + tp_rank`（`default_loader.py:381-397`） | EP 是"被推导"的，不是独立旋钮 |
| A7 | 参数**按名字选择性 offload 到 CPU**（含 experts） | ✅ | UVA：`cpu_offload_params`，段匹配 `".{param}." in f".{name}."`（`offloader/uva.py:86-92`，docstring 举例 `mlp.experts.w2_weight` 可用 `experts` 命中）；Prefetch：`offload_params`（`offloader/prefetch.py:181-185`）+ `_CpuParamOffloader:600` | **节点内 GPU↔CPU**，不跨设备 |
| A8 | **运行中装载/卸载单层参数** | ❌ | `gpu_model_runner.py:5643 reload_weights(...)` **无 layer 参数**（只支持整模型重载）；`sleep`/`wake_up` 粒度为**整个 engine**，tags 只有 `weights`/`kv_cache`（`gpu_worker.py:203-259`、`executor/abstract.py:336`） | 逐层 materialize 的内部原语存在（`model_loader/reload/layerwise.py:84`、`reload/meta.py`），但不是对外 API |
| A9 | **运行中改变并行度** | ⚠️ 仅 DP 与专家 | ① DP：`scale_elastic_ep(new_data_parallel_size, drain_timeout)`（`async_llm.py:1051`）→ `_commit_scale_up/down_elastic_ep`（`core_client.py:1757,1815`）；② elastic EP：本质也是改 **DP 度数**（因为 EP=DP×PCP×TP），并重算专家冗余槽（`core_client.py:1642-1649`） | 硬条件：必须 `enable_eplb=True`、**`pipeline_parallel_size == 1`**（`parallel.py:843-850`）、async 模式需 NIXL、不兼容 DP external/hybrid LB。**TP/PP 度数无法运行中改变** |

#### B. KV 缓存

| # | 能力 | 判定 | 机制与证据 | 边界 |
|---|---|---|---|---|
| B1 | **KV 缓存按层管理** | ✅ | `get_kv_cache_spec() -> dict[str, KVCacheSpec]`（键是 `layer_name`，`v1/worker/gpu/attn_utils.py:61-79`、`worker_base.py:98`）；分组 `KVCacheGroupSpec(layer_names=[...])`；每个 `layer_name` 一份 KV 张量 | 本实例**只为自己持有的层**建 spec（非本段层是 `PPMissingLayer`，不在 `attn_layers` 里）→ **PP 与 KV 按层分配天然一致** |
| B2 | 层间 KV 复用/共享 | ✅ | `kv_sharing_target_layer_name`：某层的 KV 直接复用另一层的张量（`attn_utils.py:65-66,86-88`、`initialize_kv_cache_tensors:7716-7718`） | 属模型结构特性（如跨层共享注意力） |
| B3 | KV 的寻址粒度 | ⚠️ | **block × (block 内偏移)**，必须 `block_size` 对齐（`example_connector.py::ReqMeta.make_meta` 的 `align_to_block_size`+slot_mapping）；offload 路线的键是 `block_hash + group_idx`（`v1/kv_offload/base.py:26-31`） | `group_idx` 是 **KV cache group 而非层区间** → offload 路线拿不到"按层区间"；层维度要靠 connector 的逐层钩子 |
| B4 | KV 的**逐层**加载/保存钩子 | ✅ | `wait_for_layer_load(layer_name)` / `save_kv_layer(layer_name, kv_layer, attn_metadata)`（`kv_connector/v1/base.py:332,346`）；唯一调用点 `layers/attention/kv_transfer_utils.py:51,57`（在每层 attention 前后） | docstring 明确 "useful for layer-by-layer pipelining" |
| B5 | KV 加载失败的取舍 | ✅ | `kv_load_failure_policy: "recompute" \| "fail"`（`config/kv_transfer.py:69` → `v1/core/sched/scheduler.py:148-149`） | 二选一，无"部分抢救" |
| B6 | KV 量化 / 按层跳过量化 | ✅ | `CacheConfig.cache_dtype`（`config/cache.py:77`）+ **`kv_cache_dtype_skip_layers`**（按层号或 attention 类型跳过，`:114`） | Triton 后端的 FP8 KV 需 SM89+ |
| B7 | KV 池大小受控 | ✅ | `num_gpu_blocks_override`（`cache.py:90`）、`gpu_memory_utilization`（`:69`）、`kv_cache_memory_bytes`（`:180`）、`kv_offloading_size/backend`（`:189,195`） | — |

#### C. 专家并行的通信代价（EP 的固有性质）

| # | 事实 | 判定 | 证据 |
|---|---|---|---|
| C1 | 某请求激活的专家**不在同一设备**时，每层需要 **dispatch + combine 两次 all-to-all** | ✅ 固有机制 | `fused_moe/all2all_utils.py`（`dispatch`/`combine`，`:146-351`）；`get_ep_all2all_manager()` 提供通信句柄 |
| C2 | all-to-all 后端可选，且**分档明显** | ✅ | `All2AllBackend` 字面量（`parallel.py:42`）；**默认 `allgather_reducescatter`**（`:188`，`naive`/`pplx` 会被归一化到它，`:467-473`） |
| C3 | 高速后端（DeepEP / FlashInfer-NVLink / MoRI）的存在意味着**快路径假设 NVLink 级互联** | ✅ | 选择逻辑 `distributed/device_communicators/cuda_communicator.py:140-201`（`deepep_high_throughput`/`deepep_low_latency`/`deepep_v2`/`mori_*`/`flashinfer_nvlink_two_sided`/`flashinfer_nvlink_one_sided`）；`parallel.py:675-705` 有相应拓扑校验 |
| C4 | 有**不依赖 NVLink** 的 all-to-all 后端 | ⚠️ | `nixl_ep`（`NixlEPAll2AllManager`，`:175-178`，NIXL/RDMA 路线）；通用回落 `allgather_reducescatter`（走集合通信）；CPU/XPU 也有 AgRs 实现 | WiFi 上的实际可用性**未验证** |
| C5 | **专家分级（offload）是节点内的**：把部分专家放 CPU/下级存储 | ✅ | A7 的 UVA/Prefetch offload（参数名段匹配可打到 `experts`） | **不跨设备**；这正是"专家不跨设备"的现有形态 |
| C6 | **跨设备的专家常驻池**（专家住在别的机器上、按需取） | ❌ | 没有这种抽象；跨设备只能靠 C1 的 all-to-all 现算，或 ②③ 的搬运机制 | — |
| C7 | 运行中**搬运专家权重**（rank 间） | ✅ | ① EPLB 重排：`EplbState.rearrange()`（`distributed/eplb/eplb_state.py:733`），通信后端可插拔（`torch_nccl/torch_gloo/nixl/pynccl`，`eplb_communicator.py:47,660`）；② elastic EP：`batch_transfer_weights`（`distributed/elastic_ep/elastic_execute.py:59`） | 目的是**负载均衡 / 加减 rank**，不是"把专家迁到指定机器" |
| C8 | EPLB 与 PP 的关系 | ✅ 可共存 | 普通（非弹性）EPLB **未被 PP 排除**，且已 PP-aware：组名带 `-pp{rank}` 后缀（`eplb_communicator.py:357-358`） | 但 **elastic EP 与 PP 互斥**（A9） |
| C9 | 容错（FT）对 all-to-all 后端的依赖 | ⚠️ | `FT_BACKEND_SET = {"deepep_low_latency","nixl_ep"}`（`v1/worker/sentinel/gpu_worker_sentinel.py:26,40-44`） | 只有这两个后端支持；且 FT 是 **DP/EP-only，不感知 PP** |

#### D. 其它必要情况

| # | 能力 | 判定 | 证据 |
|---|---|---|---|
| D1 | 连续批处理 + 前缀缓存（baseline） | ✅ | `enable_prefix_caching` 默认 True（`config/cache.py:96`）；`BlockPool.cache_full_blocks`、`KVCacheManager.get_computed_blocks` |
| D2 | 多机部署 | ⚠️ | `nnodes > 1` **只允许** `distributed_executor_backend ∈ {mp, uni, external_launcher}`（`parallel.py:963-971`）；Ray 路线另算 |
| D3 | PD 分离（KV 跨实例传输） | ⚠️ | NIXL push：**producer（prefill）侧 PP>1 支持"层区间 → 远端对应子区间"**，**consumer（decode）侧 PP>1 是 `NotImplementedError`**（`v1/nixl/base_worker.py:440-455`） |
| D4 | 逐层异步 KV 操作与 CUDA graph | ⚠️ | connector 需声明 `requires_piecewise_for_cudagraph`（`base.py:630` docstring：异步逐层操作无法被 CUDA graph 捕获，必须切 PIECEWISE） |
| D5 | PP 的模型支持 | ⚠️ | per-model 硬门禁：须实现 `SupportsPP` 协议，否则配置期 `NotImplementedError`（`models/interfaces.py:719`、`config/model.py:1377-1384`） |
| D6 | PP 上的额外通信 | ⚠️ | 不止前向 activation：**last rank 会把 sampled logits 广播回所有 rank**（`gpu_model_runner.py:4625-4629`）；async 路径裸调 `torch.distributed.broadcast`（`:4964,4977`） |
| D7 | 容错范围 | ⚠️ | 只有 `FaultToleranceConfig.engine_recovery_timeout_sec`（默认 120s，`config/fault_tolerance.py:12`）；FT 后端集见 C9；**不覆盖 PP** |
| D8 | 权重热更新 | ⚠️ | `WeightTransferEngine`（chunk 按**参数名**寻址，`distributed/weight_transfer/nccl_engine.py:86-88`），HTTP 面 `entrypoints/serve/dev/rlhf/api_router.py:157-205`；**方向是 trainer→inference 单向**，RDMA 后端仍是 TODO（`weight_transfer/base.py:190-191`） |
| D9 | 权重驻留的 tag 机制 | ⚠️ | `CuMemAllocator` 支持按 tag 部分唤醒（`device_allocator/cumem.py:298,313`），`SleepModeBackend` 可插拔（`sleep_mode_backend.py:142-196`） | **当前只注册了 `weights`/`kv_cache` 两个 tag、"cumec" 一个后端**；"每层一个 tag"需要自己实现，**未验证** |

**一句话总结这张表**：
vLLM 支持"**一个实例只持有部分层 + 部分专家**"，也支持"**KV 按层管理/逐层搬运**"；
但它把**层区间、TP/PP 度数、专家分布公式**都固定在启动期，
运行中唯一能改的并行度是 **DP（进而带动 EP 度数）**，且**与 PP 互斥**；
而"**专家不跨设备、只在节点内分级**"恰好是它现有的形态（A7/C5），跨设备专家只能靠 all-to-all 现算（C1）。

---

## 3. L2 算子与内核层

### 3.1 可自定义的五档

| 档 | 手段 | 典型场景 |
|---|---|---|
| 1 | 选后端（配置/环境变量） | `--attention-backend`、`flash_attn_version`、量化方法、dtype |
| 2 | **注册新 attention 后端** | `@register_backend(AttentionBackendEnum.CUSTOM)`（`registry.py:242`，官方 docstring 给了三方后端示例） |
| 3 | 注册量化方法 | `vllm/model_executor/layers/quantization/` 下的 `QuantizationConfig` 子类 |
| 4 | 自定义算子（Python 级） | `CustomOp`/`PluggableLayer`（`model_executor/custom_op.py`）、`direct_register_custom_op`（`utils/torch_utils.py:1026`） |
| 5 | CUDA/C++ 扩展 | `csrc/` + 构建系统（`setup.py`，需编译环境与目标架构） |

编译相关（介于 L2/L3）：`vllm/config/compilation.py`（`mode`、`cudagraph_mode`、`custom_ops`、`pass_config`……）、
`vllm/compilation/*`（`cuda_graph.py`、`piecewise_backend.py`、`partition_rules.py`、`decorators.py`、`caching.py`）、
`vllm/ir/*`（IR 层）、`vllm/kernels/*`（`triton`、`helion`、`aiter_ops`、`oink_ops`、`vllm_c`）。

### 3.2 已有实现的设计

- **attention 后端是一张注册表**：`vllm/v1/attention/backends/registry.py:34 AttentionBackendEnum`
  注册了 **37 个成员**（FlashAttention/FlashInfer/Triton/MLA 系列/CUTLASS/ROCm/XPU/CPU/稀疏 等，
  以及 `CUSTOM`、`NO_ATTENTION`、`FLEX_ATTENTION`、`TURBOQUANT`），另有独立的 `MambaAttentionBackendEnum`。
  设计要点：**枚举 → 类路径的映射 + 覆盖注册表（`_ATTN_OVERRIDES`）**，
  支持"覆盖内置后端"与"新增三方后端"两种用法。
- **版本选择逻辑是显式分档**：`vllm/v1/attention/backends/fa_utils.py::get_flash_attn_version`
  —— SM90 → FA3；SM100+(限 SM100) → FA4；**否则回退 FA2**；并允许 `attention_config.flash_attn_version` 覆盖。
- **量化是"配置类 + 能力声明"**：如 `Fp8Config.get_min_capability() -> 75`（`fp8.py:144`），
  平台据此校验；kernel 层面再按架构分派（Marlin 等）。
- **编译是可组合的 pass/后端体系**（`compilation/pass_config`、inductor 图切分、CUDA graph 捕获尺寸），
  并会把并行度、offload、PP 层区间等纳入**编译缓存 key**（`compilation/caching.py:576` 就提到
  `VLLM_PP_LAYER_PARTITION` 会影响计算图）——这也解释了为什么"运行中改拓扑"不能只改配置。

### 3.3 我们硬件的架构门槛（决定"适不适合"）

> **先更正一处**：本文早稿把手上的卡笼统写成"sm86/sm89"。实测本机 **RTX 4060 Laptop = sm89（Ada）**，
> 不是 sm86。这个区别直接决定 FP8 能否用，所以下面按卡分别给结论。

| 能力 | 架构门槛（源码原文级证据） |
|---|---|
| **FA3** | 仅 **9.x（Hopper）**：`vllm_flash_attn/flash_attn_interface.py:62-69 _is_fa3_supported()` → `"FA3 is only supported on devices with compute capability 9.x"`；选择逻辑 `v1/attention/backends/fa_utils.py:93-101`（major==9→FA3；major==10→FA4；**else→FA2**） |
| **FA2** | **>= 8.0**：`flash_attn_interface.py:52-59`；后端级 `v1/attention/backends/flash_attn.py:250-252`（`capability >= (8,0)`）；attention sink 需 **>= 9.0**（`:267-268`） |
| **FA4** | 9.x / 10.x / 11.x |
| **FlashInfer** | 探测函数名是 **`has_flashinfer()`**（`vllm/utils/flashinfer.py:47-63`；**不存在** `is_flashinfer_available`）；架构范围 `>= (8,0) 且 <= (12,1)`（`flashinfer.py:531-540`）；**sink 需 SM100/120**（`:555`） |
| **FP8（CUTLASS 路径，真正决定"跑不跑得动"）** | **>= 8.9**：`vllm/platforms/cuda.py:564-566 supports_fp8() → has_device_capability(89)`；C++ 判定 `csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu:145-159`（**cc 86 → `return false`**；sm90 需 CUDA≥12.0，sm89 需 CUDA≥12.4） |
| FP8 的"配置层最低能力" | **只有 75**：`layers/quantization/fp8.py:143-145 get_min_capability() → 75` → ⚠️ **vLLM 会"放行" sm86 上的 FP8，但静默换掉 activation 量化 key**（`fp8.py:284-292`：`elif cutlass_fp8_supported(): kFp8DynamicTokenSym else: kFp8DynamicTensorSym`）。**配置不报错 ≠ 硬件加速可用。** |
| block-FP8 / FP8 KV cache | block-FP8 需 **SM90+**（`w8a8_utils.py:161-174`）；Triton 后端的 FP8 KV cache 需 **SM89+**（`v1/attention/backends/triton_attn.py:540-550`，MLA 同理 `mla/triton_mla.py:212`） |
| **Marlin（W4A16/W8A16）** | 最低 **75（Turing）**：`model_executor/kernels/linear/mixed_precision/marlin.py:35-38`；强制于 `kernels/linear/__init__.py:814-822` |
| Marlin W4A8-**FP8** | **SM89 或 SM12x**（`utils/marlin_utils.py:672-680` 报错原文 "only support SM89 or SM12x"）；且在 ≥89 的设备上默认被主动拒绝除非 `VLLM_TEST_FORCE_FP8_MARLIN`（`kernels/linear/scaled_mm/marlin.py:42-55`） |
| Machete | min **90**（`kernels/linear/mixed_precision/machete.py:24-27`）→ sm89 也不可用 |
| AllSpark W4A16 | 含 **8.6**（`allspark.py:19-22`；`CMakeLists.txt:778 "8.0;8.6;8.7;8.9"`） |

**按我们手上的三种典型卡给结论**：

| 卡 | 架构 | 可用的 attention | 可用的量化 | 关键限制 |
|---|---|---|---|---|
| **RTX 4060 Laptop（本机）** | **sm89** | FA2、Triton、FlashInfer（**无 sink**） | **W4A16/W8A16 Marlin、W4A8-FP8（Marlin）、CUTLASS FP8（需 CUDA≥12.4）、int8、Triton FP8 KV cache** | 无 FA3/FA4、无 Machete、无 FP4、无 CUTLASS block-FP8/MoE/MLA（≥SM90） |
| RTX 3060 等 | sm86 | FA2、Triton、FlashInfer（无 sink）、AllSpark | **W4A16/W8A16 int4、int8** | **FP8 基本无望**：CUTLASS 判定 false、Triton FP8 KV 需 89、`MARLIN_FP8_ARCHS "8.9"` 不含 8.6 |
| GTX 1650 等 | sm75 | **FA2 不可用**（需 ≥8.0）、FlashInfer 不可用 → 实际只能 **Triton attention** | Marlin W4A16（≥75）、int8 | 老旧 Turing：attention 与量化选择都最窄 |

> ⚠️ **两个"配置说支持、实际可能跑不了"的坑（建议实测）**：
> ① `Fp8Config.get_min_capability()==75` 会在 sm86 上放行 FP8 但走非 CUTLASS 回落路径；
> ② Marlin FP8 的构建架构是 `MARLIN_FP8_ARCHS "8.9;…"`（`CMakeLists.txt:608-634`，注释逐字
> "we only enable fp8 computation for SM89 (e.g. RTX 40x0)"），cmake 架构交集（`cmake/utils.cmake:475-499`）
> **在只按 sm86 构建时不会产出 FP8 Marlin cubin**，而 `is_fp8_marlin_supported()`（≥75）仍会报告"支持"。
> **该不一致在 sm86 上的运行期表现未确认**——若真在 3060 上用 FP8，务必先实测。

**结论**：L2 对我们不是"改什么"的问题，而是"**能用什么**"的问题。
理性做法是：**照抄 vLLM 的后端选择与量化配方**
（4060 这类 Ada 卡可试 W4A8-FP8，3060 及更老一律 W4A16/int8；attention 用 FA2/Triton/FlashInfer），
把精力留在 L3/L4；**不要在 L2 上做贡献**
（既没有 CUDA 内核团队，也做不出别人跑不了的实验）。
唯一值得动 L2 的场景是"**为自研 stage runtime 写一个自定义 attention backend**"——
那时用 `register_backend(AttentionBackendEnum.CUSTOM)` + 实现
`AttentionBackend` 的四个 staticmethod（`get_name`/`get_impl_cls`/`get_builder_cls`/`get_kv_cache_shape`）
即可，架构门控钩子是 `supports_compute_capability()`（`v1/attention/backend.py:342-344`，在
`validate_configuration:409` 被消费）。

---

## 4. LMCache 的设计与适配性

> 定位（`LMCache/AGENTS.md`）：**LLM 服务的 KV cache 管理引擎**，把 KV 存在
> GPU/CPU/disk/S3 等多个层级，集成 vLLM 与 SGLang，目标是降 TTFT、提吞吐。
> 本节结论均直接来自源码核实；仍不确定的项在文中逐条标注（如 NIXL 在 WiFi 上的实际可用性、
> vLLM 侧精确支持的版本范围）。

### 4.1 数据模型：**存储层原生逐层，但上层 API 不是**（这决定了它适不适合我们）

> 先给结论免得误读：**"每层一个键"是真的，"只取某几层"在存储层可行**；
> 但 `LMCacheEngine.lookup` 会把"部分层命中"判成不命中（§4.4⑤），
> 所以**不能直接用它的上层 API 表达层区间**。

| 事实 | 证据 |
|---|---|
| KV 缓存键**自带 layer_id** | `lmcache/utils.py:325` 键格式 `model_name@world_size@worker_id@chunk_hash@dtype@layer_id[@tag%value…]`；`:508 LayerCacheEngineKey.layer_id`（默认 0）；`:399/:539 split_layers(num_layers)` 可把一个键拆成每层一个键 |
| 可**逐层**存/取 | `lmcache/v1/cache_engine.py::store_layer`（`:567`，"Store the KV cache in a layerwise manner"）与 `retrieve_layer`（`:897`）都是**生成器**，内部 `for layer_id in range(self.num_layers)` 逐层推进（`:621`、`:715`、`:993`） |
| 有开关与专用连接器 | 配置 `use_layerwise`（`lmcache/v1/config.py:85`）；GPU connector 分 `VLLMPagedMemLayerwiseGPUConnector` / `VLLMBufferLayerwiseGPUConnector` / `SGLangLayerwiseGPUConnector`（`lmcache/v1/gpu_connector/__init__.py`） |
| 层组抽象 | `lmcache/v1/kv_layer_groups.py::KVLayerGroupInfo`（`num_layers`、`contains_layer(layer_idx)`、`get_layer_shape(layer_idx)`、`get_layer_dtype(layer_idx)`） |

**对我们的意义**：LMCache 的 KV **不是"整模型一坨"**，而是"**每层一个键**"。
我们的 stage 只持有 `[12,18)` 这几层，理论上只需要读写这几个 `layer_id` 的条目——
**这正是我们 `KVBlock.layer_range` 想要的粒度**。

### 4.2 压缩与序列化：弱网下"压缩 KV 再传"有现成实现

- **CacheGen 序列化器**：`lmcache/v1/storage_backend/serde/cachegen_encoder.py` /
  `cachegen_decoder.py` / `cachegen_basics.py`。里面用
  **`QuantizationSpec(start_layer, end_layer, bins)`** 表达"**哪几层用几个量化桶**"
  （如 `start_layer=0,end_layer=10,bins=32`＋`10..32,bins=16`）→ **按层区间分配压缩预算**。
- tensor 布局是显式的层维：`[num_layers, 2, num_tokens, num_heads, head_size]`
  （`cachegen_encoder.py:95`），或 `KV_2LTD` 的 `[2, num_layers, num_tokens, hidden]`
  （`lmcache/v1/check/utils.py:54-66`）。
- 序列化器可选：配置 `remote_serde`（默认 `"naive"`，`config.py:83`）。

**对我们的意义**：论文里"按价值分配 KV 迁移预算"可以直接**复用这个分层压缩能力**：
高价值层高精度、低价值层低精度或不传——需要一个"层粒度带宽预算"的决策器，
而压缩执行是现成的。

### 4.3 扩展点（免 fork）

| 扩展点 | 内容 | 证据 |
|---|---|---|
| 存储后端接口 | `StorageBackendInterface`：`contains(key,pin)`、`exists_in_put_tasks(key)`、`batched_submit_put_task(keys,objs,transfer_spec)`… | `lmcache/v1/storage_backend/abstract_backend.py:26-75` |
| **后端插件注册** | 配置 `storage_plugins`、`remote_storage_plugins`、`runtime_plugin_locations` | `lmcache/v1/config.py:315,320,325` |
| **可插拔传输通道** | `transfer_channel` 配置；**工厂只接受 `["nixl","mock_memory"]`**（`transfer_channel/__init__.py:41`），NIXL 默认 `backends=["UCX"]`（RDMA 取向） | `lmcache/v1/config.py:213`；`lmcache/v1/transfer_channel/*` |
| 现成后端 | `local_cpu`/`local_disk`/`remote_url`(Redis/Mooncake 类)/`gds`(GPUDirect Storage)/`p2p`/`nixl`/`pd`/`audit` | `lmcache/v1/storage_backend/*.py`；`config.py` 对应字段 |
| **分布式控制面** | `enable_controller` + `controller_pull_url`/`controller_reply_url` + `lmcache_worker_ports`/`worker_ids` + 心跳参数 | `lmcache/v1/config.py:141-186` |
| **PD 分离传输** | `enable_pd` + `pd_role`/`pd_buffer_size`/`pd_peer_host`/`pd_peer_init_port`/`pd_alloc_port`/`pd_proxy_host`/`pd_proxy_port` | `lmcache/v1/config.py:187-211` |
| **路由/查找钩子** | `external_lookup_client`、`lookup_timeout_ms`、`min_retrieve_tokens`、`hit_miss_ratio`、`lookup_server_worker_ids`、`enable_scheduler_bypass_lookup` | `lmcache/v1/config.py:265,331,336,347,352,357` |
| 复用与重算混合 | `enable_blending`/`blend_thresholds`/`blend_recompute_ratios`/`blend_check_layers` | `lmcache/v1/config.py:101-122` |
| KV 量化/事件 | `cache_policy`、`numa_mode`、`priority_limit`、`enable_kv_events`、`pin_timeout_sec` | `lmcache/v1/config.py:275-300,448,460` |

> ⚠️ **更正一处容易误判的地方**：`transfer_channel/` 目录里确实有 `py_socket_channel.py`，
> 但**通道工厂只接受 `nixl` 与 `mock_memory`**，socket 通道并未接线；而 NIXL 默认走 **UCX（RDMA 取向）**。
> 所以在 WiFi + 消费级笔记本上，**P2P 通道是不可指望的**（是否可用未在仓库中验证）。
> 真正零硬件前提的跨机路径是 **`lmcache_server` + `remote_url="lm://host:port"`**（纯 TCP、CPU-only，见 §4.4③）。

### 4.4 非 vLLM 复用路径（已核实，结论比预期更受限）

**① `cache_interface.py` 不是接口**：全文 19 行，只有一个 msgspec 结构体
`LMCacheModelRequest{store_cache: bool=True, ttl: Optional[float]=None}`，
且全仓**无任何调用方**。真正的公共 API 是 `LMCacheEngine`
（`store`/`store_layer`/`retrieve`/`retrieve_layer`/`lookup`/`move`/`compress`…）。
`LMCacheEngineBuilder.get_or_create(...)` 建实例后**必须先调 `post_init()`**（内部才建 `StorageManager`）。

**② `standalone/` 不能当独立 KV 服务**（它的 docstring 声称 "Works without vLLM or GPU"，
但只有"不崩"这一半是真的）：
- 确实不需要 vLLM（`super().__init__(..., vllm_config=None, role="worker")`）也不需要 CUDA
  （用 `EngineType.MOCK` 的 connector）；
- **但 KV 内容是假的**：`_generate_multi_group_kvcaches` 用 `torch.rand(...)` 造随机张量；
- **且张量根本没交给 engine**：`post_init()` 没有 `kvcaches=` 参数（全仓只有 vLLM/SGLang 适配器传），
  `MockGPUConnector` 的四个搬运方法全是 `pass`；
- `[project.scripts]` 里没有 standalone 入口，测试也零覆盖。
→ **它是"无 vLLM 启动器 + 监控面"，不是可用的 KV 服务。**

**③ 唯一真正的独立 KV 服务是 `lmcache_server`（`lmcache/v1/server/`）**：
裸 TCP + 不透明字节块（按 `CacheEngineKey` 索引），后端是 CPU-only 的 `LMSLocalBackend`，
**不需要 GPU、不懂张量语义**；客户端是 `storage_backend/connector/lm_connector.py`，
由 `remote_url="lm://host:port"` 触发。**这是零硬件前提的跨机路径**（对我们最实用）。
另外 `api_server/`（`lmcache_controller`，默认 9000）只做 token/instance 级编排（含 `/move`），
**不含 KV 载荷**；`internal_api_server` 的 `/cache/store` 只接受 `tokens_mock` 并从
`engine.gpu_connector.kvcaches` 取数据 → **没有任何外部接口能注入外部 KV 张量**。

**④ 由此得出两个硬性结论**：
- **LMCache 只能作为"进程内嵌库"使用**（除了 `lm://` 那条字节通道）；
- 若要在自研 runtime 里用它的存储层，**必须自己实现 `GPUConnectorInterface`**
  （`to_gpu`/`from_gpu`/`batched_from_gpu`/`batched_to_gpu`/`get_shape`/`initialize_kvcaches_ptr`），
  因为现成非 MOCK connector **全部无条件创建 `torch.cuda.Stream()`**（部分还 `assert device.type=="cuda"`），
  且 `CreateGPUConnector` 对 `EngineType` 是硬编码 if/elif，**绕不过工厂就插不进去**。

**⑤ 层粒度的真实边界（这是本节最重要的修正）**：
层粒度在**存储层是真的**（键、分配、后端都是层无关的），但**上层语义只有"全层"**：

| 卡点 | 代码证据 |
|---|---|
| `lookup` **不认部分层命中** | `cache_engine.py:1108-1110`：`# Only all layers are hit and hit in one location, we consider this key as a hit` / `if hit_chunks == self.num_layers and len(block_mapping) == 1:` → 少一层就返回 0 命中 |
| 没有"层区间"API | `split_layers(num_layers)` 只能产出 `0..N-1` 全部层，无子区间 |
| `retrieve_layer` 不允许跨位置拼装 | `assert location == current_location, (...)`（`:964-968`） |
| `move` **不重映射 key** | `keys` 原样透传（`:1204-1211`），`worker_id` 不在路径里 → 能把已有 key 的 KV 搬到对端，**但无法把"stage A 的第 k 层"落到"stage B 的命名空间"** |
| 迁移通道受限 | `transfer_channel` 只支持 `["nixl","mock_memory"]`；NIXL 默认 `backends=["UCX"]`（RDMA 取向），**WiFi 可用性未确认** |

**→ 因此正确的用法是：绕开 `store_layer`/`retrieve_layer`/`lookup`/`move` 这套上层 API，
自己构造 `LayerCacheEngineKey`，直接调用 `storage_manager` 的 key 级、层无关原语**
（`batched_allocate` / `batched_put` / `batched_get` / `batched_contains` / `batched_remove` / `pin`）。
层区间 = 自选的 key 子集；`metadata.kv_shape[0]` 报**本 stage 的层数**，
本地按段索引，`split_layers` 就自然只覆盖本段。

**⑥ 自定义存储后端怎么做（无 entry point，是 config 驱动的 importlib 动态导入）**：
继承 `StoragePluginInterface`（`abstract_backend.py:394`，docstring 明确为"可插拔后端"基类），
构造签名必须是 `(dst_device=, config=, metadata=, local_cpu_backend=, loop=)`；
然后 `config.storage_plugins=["MyBackend"]` + `extra_config["storage_plugin.MyBackend.module_path"/".class_name"]`
（`storage_backend/__init__.py::storage_plugin_launcher`）。
注意：`get_allocator_backend()` 实际**只能返回 `LocalCPUBackend`**（`storage_manager.py:317`）；
插件加载失败只记日志不抛。

**⑦ 跨进程前缀命中有一个必须做的前提**：`PYTHONHASHSEED=0`
（或把 `pre_caching_hash_algorithm` 指向与 vLLM 一致的 hash 函数）。
否则跨进程会回落到 Python 内置 `hash`，**hash 不一致 → 永远不命中**（`token_database.py:142-148,282-289` 有明确警告）。

**⑧ 压缩的真实适用范围**：只有 `remote_serde="cachegen"` 真的压缩（int8 量化 + 熵编码，
`cachegen_encoder.py::torch_quant`），且**依赖 CUDA**（走 `lmcache.c_ops`）；
`naive`/`kivi` 目前是空操作；**压缩只在 `RemoteBackend` 生效，P2P 路径不压缩**。
`config.py` 里**没有 `compression` 字段**，也**没有任何"传输预算/按带宽调节"的概念**——
那部分要我们自己在后端/connector 层加。

### 4.5 对我们四个需求的判断（据源码核实后的修正版）

| 需求 | 判断 | 依据 / 要补什么 |
|---|---|---|
| ① 节点离开时按层迁移 KV | **能，但必须绕开上层 API** | 键/分配/后端**全都原生逐层**；但 `lookup` 只认"全层命中"、`move` 不重映射 key。**做法**：自己构造 `LayerCacheEngineKey`，直接调 `storage_manager.batched_contains/batched_get/batched_put/batched_remove/pin`；层区间 = 自选的 key 子集。跨机通道用 `RemoteBackend("lm://")`（纯 TCP，无硬件前提）或自研 `StoragePluginInterface` 后端 |
| ② 跨机前缀缓存复用 | **能（最成熟的一条）** | 链式前缀哈希 + `batched_contains` 的"遇到首个 miss 即停"语义；跨进程是**被显式设计**的场景。**前提**：`PYTHONHASHSEED=0`（或统一 hash 算法），`model_name`/`world_size`/`worker_id`/`dtype`/`chunk_size` 全一致，mask 必须 `FFFF…TTTT` 且 F 数为 `chunk_size` 整数倍 |
| ③ 作为自研 stage runtime 的 KV 后端 | **只能"部分复用"才划算** | 复用它的**存储层 + 键方案 + 多层后端 + 驱逐/pin/serde**；**不要**整体替换我们的 `KVStore`。理由：必须自研 `GPUConnectorInterface`（现成 connector 一律 `torch.cuda.Stream()`）；走 `store_layer` 还得继承三个 Layerwise connector 之一才过断言；且**没有任何外部接口能注入外部 KV 张量**（只能进程内嵌库） |
| ④ KV 压缩以适配弱网 | **部分能** | 只有 `remote_serde="cachegen"` 真压缩（int8+熵编码，确实先于网络传输），但**依赖 CUDA**，且**只在 `RemoteBackend` 生效、P2P 不压缩**；`config.py` 无 `compression` 字段、**无任何"传输预算"概念** → 预算决策要我们自己加 |

**两个必须我们自定的口径（LMCache 不给）**：
1. **各 stage 的 `worker_id`/`world_size` 一致性**——它们进 key，不一致则键空间**永不相交**；
2. **`kv_shape` 不在 key 里**——"层数不同但 `model_name` 相同"会**撞键**，需自行加命名空间约定。

**最值得拿走的两个东西**（净结论）：
**① 前缀缓存的 key 方案**（链式前缀哈希 + 层维度 `layer_id`）——层粒度语义在 `LayerCacheEngineKey`
这一层重建，正好对应我们的 `KVBlock(layer_range, …)`；
**② `lmcache_server` + `remote_url="lm://host:port"` 这条零硬件前提的纯 TCP 跨机字节通道**。
传输预算、按价值选层迁移、按带宽决策，都由我们在上层补——LMCache 的 `lookup` 认死"全层齐了才算命中"，
这一条决定了它**不能整体替掉我们的 `KVStore`，但可以当我们 KV 层的执行底座**。

### 4.6 与 vLLM 的分工（为什么两个都要看）

- **vLLM 提供**：执行引擎（批处理、attention/KV 布局、PP/TP）＋ **KV connector 钩子**
  （层粒度、可从 module path 加载）。
- **LMCache 提供**：**KV 的存储/层级/压缩/跨实例搬运/查找**，并已有分布式控制器与 PD 传输。
- 两者是**互补**的：vLLM 的 `LMCacheConnectorV1` 就是把 LMCache 接到 vLLM 的 connector 接口上
  （`vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py` ↔
  `LMCache/lmcache/integration/vllm/lmcache_connector_v1.py`）。
- **对我们的结论**：
  - 走"**vLLM 作为每台机器的执行器**"路线 → "跨机 KV 迁移"可大部分交给 LMCache
    （vLLM 侧就是 `LMCacheConnectorV1`，它实现的正是 `wait_for_layer_load`/`save_kv_layer`
    那套层粒度钩子）。
  - 继续走"**自研 stage runtime**"路线 → **不要整体套用 LMCacheEngine**；
    只复用它的**存储层与键方案**（自造 `LayerCacheEngineKey` + `storage_manager.batched_*`）
    与 **`lm://` 跨机字节通道**，把"层区间 / 迁移预算 / 按价值选层"这些语义留在我们自己的控制面里。
    原因见 §4.4④⑤：上层 API 只认"全层命中"，且没有任何外部接口能注入外部 KV 张量。

---

## 5. 需求 × 复用点 × 代价（决策表）

| 需求 | 推荐做法 | 档位 | 主要代价/风险 |
|---|---|---|---|
| 吞吐基线（连续批处理/前缀缓存） | stock vLLM（`enable_prefix_caching` 默认开） | A | 消费级卡无 FA3，性能预期要下调；8GB 卡上 prefix caching 的 block 哈希开销与 `gpu_memory_utilization` 需调 |
| 前缀命中感知路由 | 实现 `get_num_new_matched_tokens` 的自定义 connector，或把 `E2Placement` 作为 `--scheduler-cls` | B | 版本耦合；**Scheduler 接口源码明说"非公开、兼容性不保证"** |
| **层粒度 KV 迁移**（vLLM 路线） | 自定义 KV Connector（`save_kv_layer`/`wait_for_layer_load`，`kv_connector_module_path` 零 fork 加载）+ `kv_load_failure_policy`；层区间对齐可复用 `EngineTransferInfo.start_layer/end_layer` | B | block 对齐（不能任意切 token）；跨机传输要自己写；**HMA 布局下若 connector 未声明 `SupportsHMA` 会被强制报错**（`factory.py:55-60`） |
| **层粒度 KV 迁移**（自研 runtime 路线） | LMCache **存储层 + 键方案**（自造 `LayerCacheEngineKey` + `storage_manager.batched_*`）+ `lm://` 跨机通道 | B（部分复用） | 上层 API 不可用（全层命中语义）；必须自研 `GPUConnectorInterface`；只能进程内嵌库 |
| **层粒度权重驻留**（hot swap） | 自写 `SleepModeBackend` 注册到 `vllm.general_plugins` + 给每层打 CuMem tag，复用 `wake_up(tags)` 的部分唤醒 | B（未验证） | 现成只有 `weights`/`kv_cache` 两个 tag；**若不通就要改 `device_allocator/`** |
| 弱网 KV 压缩 | `remote_serde="cachegen"`（仅 `RemoteBackend`）或自写 serde | B | cachegen 依赖 CUDA；**无"传输预算"概念，要自己加** |
| 异构/带宽感知调度 | `--scheduler-cls` | B | 只能调度 vLLM 内部请求；跨机放置仍需我们控制面 |
| 节点内分级装载（对照实验） | `OffloadConfig`（UVA 或 prefetch） | A | 现成，可直接作为我们"跨节点分级"的基线 |
| 运行中重切 PP / 跨机 activation | 自研 stage runtime（现有路线）；**推荐用 `worker_extension_cls` + `collective_rpc` 暴露 `get/set_hidden_states`，让 vLLM 退化为"单机执行器+KV 管理器"** | C / B | 原生 PP transport 无 hook；**弹性 EP 与 PP 硬互斥，靠 vLLM 无法同时满足"分层跨机"与"节点离开缩容"** |
| MoE 专家放置 / 跨节点 offload | 运行中重排复用 EPLB `rearrange` + 可插拔通信后端（含 NIXL，不走 torch.distributed）；跨节点常驻池自研 | B/C | 跨节点专家池不存在；elastic EP 与 PP 互斥 |
| 自定义 attention/量化 | 只在必要时用 B 档注册（`register_backend(CUSTOM)` / `register_quantization_config`）；否则用现成配方 | B/A | 投入产出比低；L2 不做贡献 |

---

## 6. 复用边界怎么划（六条，属"选择建议"，与 §2.7 的事实矩阵分开看）

1. **把 vLLM 当"节点内高性能执行器 + 可插拔 KV/调度宿主"，不当"跨节点控制器"。**
   它的扩展点（Scheduler/KVConnector/WorkerExtension）都在**单实例语义**内；
   跨节点语义（谁放哪几层、节点走了怎么重切）vLLM 没有也不打算有。
2. **优先复用 KV Connector 这一条线**：它同时命中我们的"层粒度 KV 迁移"与"前缀命中路由"，
   而且是官方扩展点（`kv_connector_module_path`），投入产出比最高。
3. **L2 只做选择不做创新**；把算力与时间投在 §5 里标 B/C 的地方，
   因为这些才是"在一台数据中心服务器上不成立"的问题。
4. **LMCache 只"部分复用"**：拿它的**存储层 + 键方案（含 `layer_id`）+ `lm://` 纯 TCP 跨机通道**，
   不要整体套用 `LMCacheEngine`（它的 `lookup` 认死"全层命中"，且无法从外部注入 KV 张量）。
   接入前先定死两个口径：**各 stage 的 `worker_id`/`world_size` 一致性**、
   以及**给 `kv_shape` 不同但 `model_name` 相同的部署加命名空间**（否则撞键）。
   另外记得 `PYTHONHASHSEED=0`，否则跨进程前缀命中恒为 0。
5. **"分层跨机 + 节点离开缩容"这个组合，在原生 vLLM 里不可能同时成立**
   （`elastic EP` 与 `PP` 硬互斥，`parallel.py:846-850`；容错也只覆盖 DP/EP）。
   这既是自研控制面的正当性依据，也是论文里**最有力的一句话定位**：
   我们不是"再做一个 vLLM 插件"，而是补上 vLLM 结构上不覆盖的那一格。
6. **推荐的零 fork 集成形态**：vLLM 当"**每台机器跑一段层的执行器 + KV 管理器**"——
   用 `worker_extension_cls`（`parallel.py:265`）+ `Executor.collective_rpc`（`executor/abstract.py:155`）
   在 worker 上挂 `load_layers` / `set_hidden_states` 等自定义 RPC，**流水线编排由我们的控制面做**，
   绕开写死 `torch.distributed` 的原生 PP。这样需求 2/3 用 B 档解决，需求 4 用自带 connector 解决，
   全程**只需 fork `parallel_state.py` 与 `expert_map_manager.py`，且都可避免**。

---

## 附录：源码定位索引（本文引用到的文件）

vLLM（`project/vllm/`）：
```
vllm/plugins/__init__.py                                  插件 entry point 组
vllm/platforms/interface.py                               Platform 基类（自定义设备）
vllm/config/scheduler.py                                  scheduler_cls / get_scheduler_cls
vllm/config/parallel.py                                   worker_cls/worker_extension_cls；elastic EP×PP 互斥(843-849)
vllm/config/kv_transfer.py                                KVTransferConfig（connector/role/module_path/load_failure_policy）
vllm/config/ec_transfer.py                                ECTransferConfig（encoder cache 跨实例传输）
vllm/config/offload.py                                    UVA / prefetch 权重分级与预取
vllm/config/compilation.py, vllm/compilation/*            编译与 CUDA graph
vllm/v1/engine/core.py                                    EngineCore；装配 Scheduler(148)
vllm/v1/core/sched/scheduler.py                           recompute_kv_load_failures(148-149)
vllm/v1/executor/abstract.py                              Executor.get_class（自定义执行器）
vllm/v1/worker/worker_base.py                             worker extension 合并(265-288)
vllm/v1/worker/gpu_model_runner.py, gpu_worker.py         PP 通信与 KV 初始化、模型加载
vllm/distributed/utils.py::get_pp_indices                 PP 层区间（VLLM_PP_LAYER_PARTITION）
vllm/config/model.py::get_layers_start_end_indices        层区间计算(1516)
vllm/distributed/kv_transfer/kv_connector/v1/base.py      KVConnectorBase_V1（层粒度钩子）
vllm/distributed/kv_transfer/kv_connector/v1/*.py         各 connector；example_hidden_states_connector.py
vllm/v1/attention/backends/registry.py                    后端枚举(34) + register_backend(242)
vllm/v1/attention/backends/fa_utils.py                    FA 版本选择（SM90→FA3 等）
vllm/model_executor/layers/quantization/fp8.py            get_min_capability(144)
vllm/model_executor/layers/quantization/utils/marlin_utils.py  W4A8-FP8 需 SM89/SM12x(672)
vllm/model_executor/custom_op.py, vllm/utils/torch_utils.py   自定义算子（direct_register_custom_op 在 torch_utils:1026）
vllm/distributed/kv_transfer/kv_connector/factory.py      KVConnectorFactory（外部 module_path 优先）
vllm/model_executor/layers/attention/kv_transfer_utils.py 逐层钩子的唯一调用点(51,57)
vllm/v1/kv_offload/{factory,tiering/factory}.py           offload spec / secondary tier 可插拔（out-of-tree）
vllm/v1/kv_offload/tiering/p2p/{data,control}/base.py     DataTransport / ControlTransport（绕开 torch.distributed）
vllm/device_allocator/sleep_mode_backend.py               SleepModeBackendFactory（general_plugins 可注册）
vllm/device_allocator/cumem.py                            CuMem tag 级部分唤醒(298,313)
vllm/model_executor/model_loader/reload/{layerwise,meta}.py  逐层 materialize 原语
vllm/distributed/eplb/{eplb_state,eplb_communicator}.py   专家重排 + 可插拔通信后端（含 nixl）
vllm/distributed/weight_transfer/*                        按参数名寻址的权重传输（trainer→inference）
vllm/distributed/ec_transfer/ec_connector/base.py         EC = Encoder Cache（多模态 encoder 输出）
vllm/v1/attention/backend.py                              AttentionBackend 抽象 + supports_compute_capability(342)
vllm/model_executor/kernels/linear/mixed_precision/marlin.py  Marlin min capability = 75
vllm/platforms/cuda.py                                    supports_fp8 → has_device_capability(89)
csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu  FP8 CUTLASS 的 C++ 架构判定(145-159)
vllm/entrypoints/llm.py                                   进程内嵌 LLM（离线批 generate/enqueue）
```

LMCache（`project/LMCache/`）：
```
lmcache/utils.py                                   CacheEngineKey / LayerCacheEngineKey(layer_id) / split_layers
lmcache/v1/token_database.py                       链式前缀哈希 _prefix_hash / chunk_size 对齐 / PYTHONHASHSEED 警告
lmcache/v1/cache_engine.py                         store/retrieve/store_layer/retrieve_layer/lookup/move/post_init
                                                   ★ lookup 只认"全层命中"（:1108-1110）；move 原样透传 keys（:1204）
lmcache/v1/kv_layer_groups.py                      KVLayerGroupInfo（按 shape/dtype 自动分组，非层区间）
lmcache/v1/config.py                               use_layerwise / storage_plugins / transfer_channel / remote_serde /
                                                   enable_controller / enable_pd / external_lookup_client（无 compression 字段）
lmcache/v1/storage_backend/abstract_backend.py     StorageBackendInterface(:26) / AllocatorBackendInterface(:295) /
                                                   StoragePluginInterface(:394) = 自定义后端基类
lmcache/v1/storage_backend/__init__.py             storage_plugin_launcher（importlib 动态导入插件）
lmcache/v1/storage_backend/storage_manager.py      batched_allocate/put/get/contains、get_block_mapping（命中即回写 CPU）
lmcache/v1/storage_backend/naive_serde/__init__.py CreateSerde：仅 naive / kivi / cachegen（前两者当前是空操作）
lmcache/v1/storage_backend/serde/cachegen_*.py     CacheGen int8 量化+熵编码（tp_quant 走 lmcache.c_ops，需 CUDA）
lmcache/v1/storage_backend/connector/lm_connector.py  lm:// 客户端 → lmcache_server
lmcache/v1/server/__main__.py                      ★ 唯一真正的独立 KV 服务（裸 TCP + 不透明字节块，CPU-only）
lmcache/v1/standalone/manager.py                   "无 vLLM 启动器"（KV 是 torch.rand 造的，不能当服务用）
lmcache/v1/api_server/__main__.py                  lmcache_controller（token/instance 级编排，不含 KV 载荷）
lmcache/v1/internal_api_server/                    进程内 HTTP 面（/cache/store 只吃 tokens_mock，无法注入 KV）
lmcache/v1/gpu_connector/gpu_connectors.py         GPUConnectorInterface（非 vLLM 必须自研）+ 各 Layerwise 变体
lmcache/v1/gpu_connector/utils.py                  assert_layerwise_gpu_connector（只认三个特定类）
lmcache/integration/vllm/lmcache_connector_v1.py   ↔ vLLM 的 LMCacheConnectorV1（层粒度钩子的对接实现）
```
