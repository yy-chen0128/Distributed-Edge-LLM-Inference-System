# 当前工作总览与后续路线图

更新日期：2026-09-09

本文档用于统一当前项目口径：说明已经完成了什么、系统目前能验证到什么程度、还缺什么，以及后续如何用多张淘汰旧卡组成验证系统。此前讨论过的昇腾云算力路线暂不进入当前实验主线；后续验证资源假设改为多张旧 GPU 卡构成的异构小集群。

## 1. 项目定位

本项目研究的是异构分布式算力环境下的大模型推理调度与弹性资源管理。核心目标不是再做一个单机推理框架，而是在多个性能、显存、链路状态可能不同的设备之间，动态安排模型层、专家、KV cache 和推理任务，使系统在节点加入、节点离开、带宽波动和资源紧张时仍能继续服务。

当前主线聚焦推理系统。训练相关能力保留为研究背景和远期扩展方向，因为训练涉及梯度、优化器状态、同步一致性和 checkpoint 恢复，状态面远大于推理。目前原型先把推理侧的参数驻留、KV 管理、activation 传输、流水线执行和中断恢复跑通。

项目的核心研究问题可以概括为六点：

1. 如何把大模型推理任务拆成多个可调度阶段，例如按层切分、按专家切分、按 prefill/decode 切分。
2. 多个算力节点如何协同执行同一个请求，而不是每个节点独立运行完整模型。
3. 调度层如何根据算力、显存、队列、KV 命中和链路状态分配任务。
4. 节点加入或离开时，如何快速装载、卸载、迁移和恢复模型状态。
5. 新节点到来时，如何评估其能力，并决定它承担多大的模型片段或任务比例。
6. 在非数据中心网络中，如何把通信延迟、带宽和不稳定性纳入调度决策。

## 2. 当前资源假设修正

当前不再把华为昇腾云算力作为近期验证资源。昇腾相关分析仍有参考价值，但它不再决定近期实现路径，也不再作为测试计划中的必需环境。

新的近期资源假设是：使用多张淘汰的旧 GPU 卡组成一个小型异构系统。这个选择更适合当前阶段，原因如下：

- 多张旧卡可以真实验证多设备调度、跨设备 activation 传输、KV 迁移、节点离开和重新切分。
- 旧卡之间性能和显存通常不一致，天然符合异构调度问题。
- 即使单卡性能较弱，也足够运行小模型、随机权重模型、toy Transformer 或分层 mock runtime。
- 旧卡更容易做故障注入，例如手动停止某个 worker、限制显存、限制带宽、制造延迟和丢包。
- CUDA/PyTorch/vLLM/LMCache 生态在旧 NVIDIA 卡上更容易完成第一阶段验证。

这意味着当前实验目标不是追求单次推理绝对性能，而是验证系统机制是否成立：调度层能否正确观察状态、制定计划、切换 epoch、传输 activation、管理 KV、恢复任务，并在资源变化后继续工作。

## 3. 总体架构

当前系统分为三层：

```text
应用入口与实验负载
  - CLI demo
  - layered simulation
  - 数据集/trace loader
  - 后续真实多卡实验驱动

动态调度与控制层
  - NodeManager: 节点注册、心跳、加入/离开
  - ModelManager: 层、专家、显存账本、装载/卸载语义
  - TaskScheduler: 请求路由、任务下发、流水线执行、中断恢复
  - PipelineReconfigurationCoordinator: epoch 化重配置
  - KVStore: KV 目录、查询、迁移、逐出
  - Transport: 节点间数据传输抽象
  - Policies: 放置、迁移、重并行化、恢复策略

执行与存储后端
  - MockEngine / LayeredMockEngine
  - TorchLayeredEngine
  - VLLMEngine
  - MockKVStore / LMCacheStore
  - MockTransport / TCPTransport / WiFiTransport
```

这个分层的核心原则是机制和策略分离。机制层必须稳定，负责把请求、节点、模型状态、KV 和链路组织成可运行的系统；策略层可以逐步替换，从简单启发式发展到更复杂的优化算法。

## 4. 已实现的核心机制

### 4.1 节点状态与事件

`NodeManager` 已实现节点注册、心跳更新、超时检测、节点加入和节点离开事件。每个节点包含静态能力和动态状态：

- 静态能力：算力、总显存、带宽、延迟、支持角色。
- 动态状态：剩余显存、当前负载、存活状态、KV 块数量、已装载专家、已装载层区间、队列深度、正在执行的任务。

这为后续旧卡集群实验提供了基础：每张旧卡可以注册为一个独立节点，也可以在单机多进程环境下注册为多个逻辑节点。

### 4.2 模型装载与卸载语义

`ModelManager` 已实现层和专家的装载账本：

- `load_model`: 按放置计划批量装载。
- `load_layers`: 指定节点装载某个连续层区间。
- `load_experts`: 指定节点装载若干专家，并区分 GPU/CPU 层级。
- `unload`: 卸载指定层或专家。
- `apply_pipeline_plan`: 原子更新流水线计划的显存和层覆盖账本。
- `validate_pipeline_coverage`: 校验层区间是否连续完整覆盖模型。

当前真实权重尚未接入生产级 Transformer runtime，但控制层已经明确知道“哪个节点持有哪些层、哪些专家、占用了多少模型显存”。这正是后续旧卡实验中需要保留的接口。

### 4.3 Pipeline epoch 与动态重配置

`PipelineReconfigurationCoordinator` 已经实现 epoch 化重配置。一个 pipeline epoch 表示一次完整的模型层放置计划，新的请求会绑定当前 epoch，旧请求继续使用旧 epoch，避免中途切换导致 activation 被错误解释。

重配置流程是：

```text
计算新放置计划
  -> 在目标节点 prepare_epoch
  -> 更新模型层账本
  -> 复制可复用 KV
  -> activate_epoch 切入新计划
  -> 旧请求排空后 drain_old_epoch
  -> retire_epoch 释放旧资源
```

这个机制已经覆盖了动态调度系统最关键的控制语义：新计划先准备好，再切流；旧计划不立即销毁；状态迁移和资源释放分阶段进行。

### 4.4 Activation 数据面协议

系统已定义 `ActivationEnvelope` 作为 stage 之间传输 hidden states 的统一协议。它包含：

- request id
- pipeline epoch
- stage index
- layer range
- source node
- destination node
- payload
- metadata
- checksum

`TaskScheduler` 在多 stage 流水线中按层顺序执行任务。前一个 stage 产生 activation，后一个 stage 通过 `Transport.push/pull` 接收 activation，并在执行前校验 request、epoch、目标节点和 checksum。

这说明系统已经不是简单的“把请求发给某个节点”，而是具备了跨节点串联执行一个模型的控制和数据协议。

### 4.5 KV cache 目录与部分迁移基础

系统已定义 `KVBlock` 作为 KV 管理单位。每个 KV block 可以携带：

- block hash
- token 数
- 字节大小
- layer range
- 数据 payload
- token key
- reuse count
- prefill time
- hot 标记

`KVStore` 提供统一接口：

- `save`
- `load`
- `lookup`
- `move`
- `evict`
- `get_index`
- `estimate_move_cost`

当前 `MockKVStore` 可以完整验证目录和迁移语义；`LMCacheStore` 作为真实 KV 后端适配层，已经通过契约测试。关键设计是 KV block 带有 `layer_range`，因此节点离开或重分层时，可以只迁移与某段层相关的 KV，而不是粗暴迁移整个请求状态。

### 4.6 传输抽象

`Transport` 抽象屏蔽底层通信方式：

- `MockTransport`: 用于控制带宽、延迟和本地模拟。
- `TCPTransport`: 用 loopback socket 验证真实 push/get 协议。
- `WiFiTransport`: 用于模拟无线链路的带宽、延迟和丢包。

旧卡多机实验可以先走普通 TCP；如果多张卡在同一台机器上，则可以用 loopback 或本地 IPC 模拟节点间传输；如果多台旧机器组成集群，则可以直接测实际以太网带宽和延迟。

## 5. 已实现的执行后端

### 5.1 MockEngine

`MockEngine` 用于最基础的端到端请求验证。它把一个请求当成完整模型任务执行，适合测试请求路由、KV 登记、错误处理和恢复逻辑。

### 5.2 LayeredMockEngine

`LayeredMockEngine` 用于无 GPU 的分层流水线验证。它不做矩阵运算，但严格模拟以下语义：

- 每个节点只执行自己的 `layer_range`。
- stage 按顺序消费前一个 stage 的 activation。
- 每个 stage 生成属于自己层区间的 KVBlock。
- activation payload 有大小、来源、目标和 epoch 校验。
- 旧 epoch 可以在 drain 后释放。

这是目前验证动态调度控制流的主要后端。

### 5.3 TorchLayeredEngine

`TorchLayeredEngine` 是当前更接近真实执行的后端。它已经具备：

- `prepare_epoch`: 按 layer range 创建并驻留 stage 层权重。
- `activate_epoch`: 激活某个 epoch。
- `retire_epoch`: 释放旧 epoch 的参数。
- `generate`: stage 0 从 prompt 生成 embedding，后续 stage 反序列化前一段 hidden states。
- 使用 PyTorch tensor 进行实际数值计算。
- 序列化真实 hidden states 作为 activation payload。
- 为每段层生成 layer-local KVBlock。
- 支持 `device="cpu"`、`device="npu"` 和 `device="auto"`。

虽然它现在还是 toy layer，不是完整 Transformer，但它已经把“动态 stage runtime 必须提供的执行契约”落成了可测试实现。旧卡实验中，下一步应把它扩展为 CUDA 设备上的真实小模型 stage runtime，优先跑通小模型和随机权重模型。

### 5.4 VLLMEngine

`VLLMEngine` 是 OpenAI HTTP 客户端，用来连接一个已经启动的完整 vLLM 服务。它可以作为完整模型服务后端参与测试，但它明确拒绝 `stage_count > 1`。

原因是 vLLM 的 OpenAI endpoint 表示一个完整 engine，而不是任意模型层 stage。不能把多个 vLLM HTTP endpoint 简单串起来当成分层流水线。vLLM 内部支持静态 TP/PP，但层边界、worker 组和通信组在启动时固定；运行中动态改层切分、外部传 activation、迁移部分 KV 和参数，都不是公开接口能力。

## 6. 已实现的策略层

### 6.1 放置策略

当前包含：

- `DefaultPlacement`: 选择负载最轻的健康节点。
- `CapabilityPlacement`: 按节点算力加权，算力更强的节点承担更多任务。
- `E2Placement`: 参考 Preble 的 exploit/explore 思路，综合 KV 前缀命中和负载。
- `LayeredPipelinePlacement`: 将一个完整层放置计划展开成多 stage pipeline task。

`E2Placement` 已经把 KV 命中纳入路由决策：命中收益大时优先送到持有缓存的节点，否则送到综合负载更低的节点。

### 6.2 迁移策略

当前包含：

- `DefaultMigration`: 节点离开时尽量全量迁移 KV。
- `PriorityMigration`: 按 `reuse_count * prefill_time_ms` 计算 KV 迁移价值，在时间预算有限时优先迁移高价值块，低价值块可以丢弃并在后续重算。

这对应了动态环境下的核心权衡：不是所有状态都值得抢救，特别是在旧卡、低带宽或节点突然离开的场景中，迁移预算必须花在最有价值的 KV 上。

### 6.3 重并行化策略

`CapabilityReparallelization` 已实现按节点算力比例重新切分层区间。它会根据存活节点的相对算力，把模型总层数分配给不同节点，使每个节点预计计算时间更接近。

这适合旧卡实验：不同旧卡性能差异明显，用算力比例切层比平均切层更合理。

### 6.4 恢复策略

当前包含：

- `DefaultRecovery`: 从头重跑任务。
- `TokenRecovery`: 保留已提交 token 和 KV，从 `progress_tokens` 继续。

当前 token 级恢复还没有完全和真实生成引擎绑定，但接口已经为中断恢复保留了进度字段和重试语义。

## 7. 当前验证状态

截至当前整理，项目测试覆盖了以下能力：

- mock 端到端请求流。
- 节点注册、心跳、加入、离开。
- KV 保存、查询、迁移、逐出。
- E2 缓存感知放置。
- PriorityMigration 迁移优先级和时间预算。
- CapabilityReparallelization 按能力重切层。
- TokenRecovery 中断恢复接口。
- LayeredMockEngine 分层流水线。
- PipelineReconfigurationCoordinator epoch 切换、KV copy、drain。
- ActivationEnvelope 序列化、校验和传输。
- TCP/WiFi/MockTransport 契约。
- VLLMEngine OpenAI payload、错误处理、连接失败和 stage 拒绝。
- LMCacheStore 适配层契约。
- TorchLayeredEngine 真实 tensor activation 传输、层权重驻留和 epoch 释放。

最近一次完整测试结果：

```text
76 passed, 1 skipped
```

唯一警告来自 Windows 下 pytest cache 目录权限，不影响功能判断。

## 8. 与现有框架的关系

### 8.1 vLLM

vLLM 是当前推理服务的重要基线，适合做高吞吐完整模型服务。我们复用它的方向包括：

- 完整模型 OpenAI endpoint。
- PagedAttention 思想。
- KV connector / LMCache 生态。
- 静态 TP/PP 的性能基线。

但 vLLM 不是当前动态 stage 的直接执行接口。运行时动态层迁移、外部 activation 流水线、部分 KV 迁移和节点离开恢复都需要我们自己的控制层和 runtime。

### 8.2 SGLang

SGLang 与 vLLM 同属高性能推理服务框架，值得作为第二个完整模型服务后端。它的前缀缓存、结构化生成和 serving runtime 对我们有参考价值。短期可以作为对照后端，不建议替换核心调度层。

### 8.3 LMCache

LMCache 适合作为真实 KV 管理后端。我们的 `KVStore` 抽象避免直接绑定某个实现，使 MockKVStore、LMCacheStore 和后续自研 KV backend 可以互换。

### 8.4 SpotServe

SpotServe 对本项目最关键的启发是：真正的动态调度必须进入执行引擎级别，处理参数、KV buffer 和恢复状态，而不是只在 HTTP 请求路由层做选择。它也说明动态再并行化、状态迁移和中断恢复需要全链路设计。

### 8.5 MoE-Infinity 与 Preble

MoE-Infinity 对专家缓存、专家预取和专家分级存储有参考价值。Preble 对 KV 前缀命中与负载均衡之间的 exploit/explore 权衡有直接参考价值。

我们的机会是把这些思想合在一起：同时管理层、专家、KV 和链路，而不是只优化其中一个对象。

## 9. 旧卡多卡验证方案

### 9.1 目标

旧卡验证的目标不是追求大模型极限吞吐，而是构建一个真实、有差异、有故障注入能力的小型系统，用它证明动态调度机制成立。

最小可用目标：

- 至少两张 GPU 卡，最好三张或以上。
- 可以是同机多卡，也可以是多台旧机器各带一张卡。
- 每张卡作为一个 `Node` 注册。
- 每个节点运行一个 stage runtime 或完整模型 endpoint。
- 调度器能够观察每个节点的显存、负载、队列和存活状态。
- 请求可以按层流水线跨节点执行。
- 某个节点离开后，系统能重新切分层并继续处理后续请求。

### 9.2 推荐硬件组织

优先级从高到低：

1. 单机多张旧 NVIDIA GPU。
2. 多台机器，每台一张旧 NVIDIA GPU，通过有线局域网连接。
3. 混合 GPU + CPU 节点，用 CPU 节点模拟慢设备或 cache 节点。
4. 如果有非 NVIDIA 卡，先作为调度可见节点或完整服务 endpoint，暂不放入动态 stage 主链路。

如果旧卡显存较小，可以使用以下方法降低压力：

- 使用小模型，例如 0.5B、1B、3B 级别模型。
- 使用 dummy/random 权重进行流程验证。
- 使用量化权重。
- 只运行 toy Transformer stage。
- 使用 `LayeredMockEngine` 和 `TorchLayeredEngine` 混合验证。
- 把大模型实验拆成控制面实验和真实性能实验两类。

### 9.3 单机多卡实验

单机多卡适合第一阶段，因为网络因素更少，便于定位问题。

实验拓扑：

```text
controller process
  -> node_0: GPU 0, layers 0..k
  -> node_1: GPU 1, layers k..m
  -> node_2: GPU 2, layers m..n
```

先用 `MockTransport` 或 loopback TCP 传 activation，再逐步替换为更接近真实的数据通道。

可验证内容：

- 每张卡的显存探测。
- 按能力切分层。
- stage 间 activation 传递。
- 每段层生成独立 KV。
- 停止某个 stage 后重配置。
- 旧 epoch drain 后释放资源。

### 9.4 多机多卡实验

多机多卡适合第二阶段，因为它能验证真实链路。

实验拓扑：

```text
controller
  -> worker-a: old GPU A
  -> worker-b: old GPU B
  -> worker-c: old GPU C
```

每个 worker 需要运行一个节点 agent，负责：

- 上报 GPU 型号、显存、利用率和进程状态。
- 接收 `StageAssignment`。
- 装载或卸载指定层/专家。
- 执行 stage task。
- 发送或接收 activation。
- 上报 KV/cache 占用。

可验证内容：

- TCP 链路实际带宽和延迟。
- 慢链路下的流水线瓶颈。
- 节点掉线后的事件传播。
- KV 迁移成本估计是否合理。
- 重配置是否能在真实网络延迟下完成。

### 9.5 故障注入

旧卡系统非常适合故障注入。建议保留以下实验：

- 直接停止某个 worker 进程，模拟节点离开。
- 人为降低某个节点心跳频率，模拟不稳定节点。
- 限制某张卡可用显存，模拟显存压力。
- 给 `Transport` 注入带宽下降和丢包。
- 在 stage 执行中抛出异常，模拟推理失败。
- 在 epoch 切换期间触发请求，验证新旧请求隔离。

评价指标包括：

- 请求成功率。
- 平均延迟和 tail latency。
- 重配置耗时。
- KV 迁移量。
- KV 命中率。
- 被丢弃 KV 的重算成本。
- 节点离开后的恢复时间。
- 每张旧卡的利用率和等待时间。

## 10. 近期实施计划

### 阶段 A：旧卡环境探测

目标是把旧卡注册为可观测节点。

需要完成：

- 编写 GPU 探测模块，读取型号、显存、可用显存、驱动、CUDA 可用性。
- 把探测结果映射到 `NodeCapability` 和 `NodeState`。
- 实现周期心跳，上报显存、负载、队列和存活状态。
- 输出一份 cluster inventory，记录每张卡的能力。

验收标准：

- 多张旧卡能被注册为多个节点。
- 调度器能实时看到节点状态变化。
- 停止一个 worker 后能触发 `NODE_LEFT`。

### 阶段 B：单机多卡 mock + torch stage

目标是在多张旧卡上验证动态层流水线。

需要完成：

- 让 `TorchLayeredEngine` 支持明确指定 CUDA 设备，例如 `cuda:0`、`cuda:1`。
- 为每张卡创建一个 runtime。
- 使用 `CapabilityReparallelization` 生成层切分。
- 跑通三段或多段流水线。
- 注入节点离开，触发新 epoch。

验收标准：

- 请求跨多个 stage 完成。
- activation 真实序列化、传输、反序列化。
- 每段层产生独立 KVBlock。
- 节点离开后新请求走新 epoch。
- 旧 epoch drain 后旧 stage 释放。

### 阶段 C：真实 worker agent

目标是从单进程模拟走向多进程或多机器。

需要完成：

- 定义 controller 到 worker 的控制 RPC。
- worker 暴露 prepare/activate/retire/generate/status 接口。
- Transport 从进程内 mock 迁移到 TCP。
- 节点状态由 worker 主动上报。
- 调度器不再直接持有 runtime 对象，而是通过 agent client 调用。

验收标准：

- 多个 worker 进程可独立启动和停止。
- controller 能远程下发 stage task。
- worker 掉线能被检测并触发重配置。

### 阶段 D：小模型真实 stage runtime

目标是把 toy Torch layer 替换成可解释的小模型层。

需要完成：

- 选择一个小模型或自定义小 Transformer。
- 将模型层拆成可独立加载的 stage。
- stage 之间传递真实 hidden states。
- 每层或每段层维护 KV。
- 对比单机完整执行输出，验证数值路径合理。

验收标准：

- 分段执行输出与完整执行在可接受误差内一致，或对随机权重模型保持确定性一致。
- 层切分变化后仍能执行。
- KV 管理和 activation 传输仍通过现有接口。

### 阶段 E：vLLM/SGLang 完整后端基线

目标是建立完整模型服务基线，而不是动态 stage。

需要完成：

- 在旧卡上启动小模型 vLLM 或 SGLang 服务。
- 用 `VLLMEngine` 或新增 `SGLangEngine` 接入完整 endpoint。
- 记录吞吐、延迟、显存和失败行为。
- 与动态 stage runtime 的机制实验分开报告。

验收标准：

- 至少一个完整模型 endpoint 可被调度器调用。
- OpenAI API 路径稳定。
- 明确区分完整 endpoint 基线与动态 stage 实验。

### 阶段 F：策略增强

目标是把目前的机制验证推进到研究贡献。

需要完成：

- 将 KV 命中、专家覆盖、显存压力、链路带宽和节点算力合入统一打分。
- 扩展 `CapabilityReparallelization`，引入通信成本。
- 对 `PriorityMigration` 加入参数迁移和专家迁移。
- 引入热点 KV/专家冗余复制。
- 支持更细粒度的任务恢复和重试。

验收标准：

- 有明确 baseline。
- 有消融实验。
- 有节点 churn、带宽变化、异构性能变化下的结果。

## 11. 当前需要补齐的工程模块

### 11.1 Node Agent

当前调度器直接持有 runtime 对象，适合单进程测试。真实旧卡系统需要独立 worker agent。

Node Agent 应提供：

- `status`: 返回节点状态。
- `prepare_epoch`: 装载层/专家。
- `activate_epoch`: 切换当前 epoch。
- `retire_epoch`: 释放旧 epoch。
- `generate_stage`: 执行 stage task。
- `load_kv` / `save_kv` / `evict_kv`: KV 管理。
- `shutdown`: 用于测试和清理。

### 11.2 设备探测

需要新增设备探测后端，将旧卡状态转成调度器可理解的字段：

- GPU 名称。
- 显存总量。
- 当前可用显存。
- CUDA 是否可用。
- 驱动版本。
- 当前进程占用。
- 粗略算力评分。

旧卡不一定支持新算子，因此探测模块还应记录 PyTorch/vLLM 可用性，而不仅是硬件存在。

### 11.3 真实层运行时

`TorchLayeredEngine` 需要从 toy layer 演进到真实模型层。建议先做两级：

1. 小型自定义 Transformer，用于确认 hidden states、KV 和层切分语义。
2. HuggingFace 小模型拆层，用于确认真实模型结构适配。

暂不建议一开始就 fork vLLM。等控制语义稳定后，再选择性复用 vLLM 的 kernel、KV 管理或 worker 内部。

### 11.4 通信与序列化

当前 activation 用 PyTorch 序列化为 bytes，可靠但不一定高效。旧卡实验第一阶段可以接受，后续需要优化：

- tensor metadata 与 tensor bytes 分离。
- 支持 pinned memory。
- 支持 chunk 传输。
- 支持压缩或量化 activation。
- 支持不同 transport 后端的带宽测量。

### 11.5 观测与日志

需要统一记录：

- 每次请求经过哪些 stage。
- 每个 stage 的层区间和节点。
- activation 大小和传输耗时。
- KV 生成、命中、迁移、丢弃。
- epoch 切换耗时。
- 节点状态变化。
- 错误、重试和恢复路径。

这些日志会直接成为论文实验图表的数据来源。

## 12. 实验设计

### 12.1 Baseline

建议设置以下 baseline：

- 单节点完整模型执行。
- 静态平均切层流水线。
- 按算力比例切层流水线。
- 无 KV 迁移的节点离开恢复。
- 全量 KV 迁移。
- PriorityMigration 部分迁移。
- vLLM/SGLang 完整 endpoint 服务。

### 12.2 Workload

负载可以分三类：

- 合成短 prompt，用于快速测试吞吐和调度正确性。
- 共享前缀 prompt，用于测试 KV 命中和 E2 路由。
- 长 prompt 或多轮会话，用于测试 KV 容量压力和迁移收益。

已有文档建议优先使用 MoE-Infinity 的共享前缀 fixture、Preble 的 workload generator，以及 SpotServe 的节点 add/remove trace 思路。

### 12.3 关键指标

系统指标：

- throughput
- 平均 latency
- p95/p99 latency
- 请求成功率
- 节点离开后的恢复时间
- epoch 切换时间

状态指标：

- KV 命中率
- KV 迁移字节数
- 被丢弃 KV 的重算成本
- activation 传输字节数
- 每节点显存占用
- 每节点负载和队列长度

策略指标：

- 算力利用率均衡程度
- 慢节点造成的流水线等待
- 迁移预算使用率
- 热点 KV/专家复制收益

## 13. 当前边界与风险

当前已经完成的是动态调度框架和可执行契约，不是生产级 LLM runtime。

主要边界：

- `TorchLayeredEngine` 仍是 toy layer，不是完整 Transformer。
- `ModelManager` 管理的是控制账本，真实权重下载和释放需要 Node Agent 实现。
- `TokenRecovery` 的接口已存在，但真实 token 级恢复要与生成引擎深度结合。
- `LMCacheStore` 适配层已有契约测试，但真实多进程 KV 服务还需要旧卡环境验证。
- `VLLMEngine` 只能调用完整 endpoint，不能执行任意 layer stage。
- 当前还没有统一 metrics/exporter。

主要风险：

- 旧卡显存太小，无法直接运行目标模型。
- 旧卡算子兼容性不足，新版本 vLLM/SGLang 可能不支持较老架构。
- 多机旧卡网络带宽低，activation 传输可能成为主要瓶颈。
- 分段 Transformer 的数值一致性和 KV 对齐需要细致验证。
- 动态迁移如果粒度过细，控制开销可能超过收益。

对应缓解：

- 第一阶段使用小模型、随机权重和 toy Transformer。
- 先验证机制，再验证性能。
- 保留 mock/CPU 实验作为 ground truth。
- 所有优化策略都配 baseline 和消融实验。
- 将完整 endpoint 实验与动态 stage 实验分开报告。

## 14. 下一步最优先任务

近期最值得直接开始的工程任务如下：

1. 新增旧卡节点探测模块，把 GPU 状态映射到 `NodeCapability` 和 `NodeState`。
2. 扩展 `TorchLayeredEngine`，明确支持 `cuda:0`、`cuda:1` 等设备选择，并补测试。
3. 编写单机多卡 demo：每张卡一个逻辑 stage，跑通 layer pipeline。
4. 实现最小 Node Agent，让 worker 可以独立进程运行。
5. 将 `MockTransport` 替换为 TCP loopback，再替换为多机 TCP。
6. 设计小模型 stage runtime，逐步替换 toy layer。
7. 增加统一 metrics 记录，为后续实验图表做准备。
8. 在旧卡上部署 vLLM/SGLang 小模型完整 endpoint，作为服务基线。

如果只能先做一个任务，建议优先做“旧卡节点探测 + CUDA TorchLayeredEngine 多卡 demo”。它能最快验证资源是否可用，也能最直接暴露旧卡环境问题。

## 15. 当前口径总结

当前项目已经完成了动态调度系统的骨架：节点、模型状态、KV、传输、任务、流水线、epoch 重配置和策略接口都已连成可测试闭环。它已经能在无 GPU 条件下模拟多节点分层执行，也能用 PyTorch tensor 验证真实 activation 数据流。

接下来不再围绕昇腾云算力推进，而是使用多张淘汰旧卡搭建小型异构验证系统。旧卡系统的价值不在于性能强，而在于它能真实制造异构、显存限制、慢链路和节点不稳定这些研究场景。我们的近期目标是把当前控制面接到真实旧 GPU runtime 上，先证明动态切分、状态迁移和恢复机制成立，再逐步替换更高性能的推理后端。
