# vLLM 与边缘分层系统的集成边界

更新：2026-08-24

## 结论

vLLM 已经在一个 Engine 内部实现了 tensor parallelism（TP）和 pipeline
parallelism（PP）。一个 Engine 可以管理 `TP x PP` 个 worker/GPU；HTTP/OpenAI
端点暴露的是完整 Engine，而不是“可把任意层的 activation 交给外部”的 stage API。

因此，不能把多个 `VLLMEngine` HTTP 实例串成一个模型流水线。那样每个端点都会把
自己当成完整模型；activation、layer-local KV 与模型中间状态都没有公开成 HTTP
协议。项目中的 `VLLMEngine` 明确拒绝 `stage_count > 1`，以防形成这一错误配置。

对于论文要求的节点 join/leave、异构能力、层区间动态重划分、跨边缘链路 activation
和 KV 迁移，本项目不 fork vLLM。采用两个执行轨道：

1. 静态生产轨道：把 stock vLLM 作为一个不可在运行中改变 PP 拓扑的完整执行组。启动
   新组时可设置 TP/PP，并用 `VLLM_PP_LAYER_PARTITION=8,12,4` 指定不等长的静态层区间。
2. 动态研究轨道：控制面维护独立的 stage 协议、activation 数据面与 epoch 化重配置。
   当前 `LayeredMockEngine` 在 CPU 上完整验证协议；未来可实现独立的 CUDA/Transformers
   stage runtime，或把每次新 epoch 落地为新启动的静态 vLLM 组。

这不是回避 vLLM，而是尊重它的扩展边界：用 vLLM 做它擅长的高性能静态执行，用本项目
实现论文真正需要的弹性控制面。

## 本地源码核验

| 需求 | stock vLLM 的能力 | 结论 |
|---|---|---|
| 多 GPU 的 TP/PP 执行 | `vllm/v1/executor/*` 创建固定 worker 组 | 可直接使用 |
| 异构静态层区间 | `distributed/utils.py::get_pp_indices` 读取 `VLLM_PP_LAYER_PARTITION` | 可直接使用，但只在启动时生效 |
| 自定义 worker 编排 | `Executor.get_class` 接受自定义限定名 | 可做，但绑定内部 `SchedulerOutput/WorkerBase`，不作为首选控制面 |
| worker 控制命令 | `Executor.collective_rpc` | 官方注释要求只传控制消息，数据面另设 |
| 跨实例 KV | KV Connector / LMCache，特别是 disaggregated prefill/decode | 可直接复用作 KV 传输层 |
| 运行时改 PP 世界大小或层边界 | PP group、model layers 和 worker rank 在初始化时固定 | 不支持为公开弹性 API |
| PP 下 elastic scaling | `config/parallel.py` 明确拒绝 `enable_elastic_ep` 与 PP 组合 | 不支持 |
| 将任意 PP stage activation 交给外部链路 | `gpu_model_runner.py` 用内部 PP group 的 `irecv_tensor_dict/isend_tensor_dict` | 未暴露；直接替换需侵入内部执行路径 |

`Executor` 是可扩展点，不等于弹性 stage API。写自定义 Executor 仍需跟随 vLLM 的内部
调度输出、worker 生命周期、KV cache 初始化和固定的 distributed group，版本耦合很高。

## 已实现的边缘控制面

### Pipeline epoch

- `core/types.py::PipelinePlan`、`StageAssignment`：一个计划是完整模型连续层区间的版本。
- `policies/placement.py::LayeredPipelinePlacement`：新请求在进入时绑定 `pipeline_epoch`。
- `core/pipeline_controller.py::PipelineReconfigurationCoordinator`：执行
  `prepare -> model ledger update -> KV copy -> activate -> ingress switch`。
- 旧 epoch 不会在切流时立即释放；入口确认在途请求排空后调用 `drain_old_epoch`。这避免
  churn 时旧 activation 被新层边界错误解释。

### Activation 数据面

- `core/types.py::ActivationEnvelope`：包含 request、epoch、stage、层区间、源/目标、payload
  和 SHA-256 校验和。
- `core/task_scheduler.py`：stage 间调用 `Transport.push/pull`，下一 stage 在执行前校验来源、
  epoch 与目标节点。
- `backends/tcp_transport.py`：已具备 push/get 的真实 loopback 协议；`MockTransport` 与
  `WiFiTransport` 可模拟带宽、时延和丢包。

### KV 与模型状态

- KV 保持 layer-local 的 `KVBlock.layer_range`；重配置时只将一个完整落在新 stage 范围内的
  块复制到新 owner。迁移使用 copy 而不是 move，旧副本在 drain 前保留。
- `KVStore` 是目录和状态操作的统一接口；`MockKVStore` 用于无 GPU 验证，`LMCacheStore`
  负责将目录动作映射到 LMCache。LMCache 管的是 KV，不负责模型权重。
- 模型状态通过 `StageRuntime.prepare_epoch(..., model_source=...)` 描述来源：同范围换节点可
  取旧节点为 source，其它范围从模型仓库重装。当前 CPU runtime 验证该控制协议；真正权重
  加载应在节点 agent 内完成，而不是由调度器进程直接拷贝权重。

## 有 GPU 时的实施顺序

1. 先部署一组静态 vLLM PP：固定参与节点、TP/PP、模型版本和
   `VLLM_PP_LAYER_PARTITION`。通过一个完整 `VLLMEngine` HTTP endpoint 服务请求。
2. 为该组配置 LMCache KV connector，先验证 disaggregated prefill/decode 的 KV 复用与目录。
3. 节点能力或成员变化时，控制面计算下一计划，在空闲资源预装一个新静态组，复制/预热可用
   KV，等待健康检查后把入口切到新组，最后 drain 并停止旧组。
4. 只有当论文必须评估“一个请求在不同独立边缘进程间逐 stage 流动”时，再实现项目自己的
   GPU `StageRuntime`。此时 activation 协议、epoch、KV catalog 和实验已经现成；不需要先
   修改 vLLM 源码。

若目标是“不中断地在同一个 vLLM Engine 中移动任意 PP layer 到新节点”，则必须深入修改
vLLM 的 worker/model-runner/parallel-state/KV 生命周期。这是一个独立的 vLLM fork 研究方向，
不应作为当前系统的第一条实现路径。
