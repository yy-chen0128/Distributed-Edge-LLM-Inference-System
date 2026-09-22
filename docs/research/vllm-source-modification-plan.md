# vLLM 源码级动态调度改造评估

更新：2026-08-25

## 结论

当前项目不应该把第一条实现路线押在 fork vLLM 上。vLLM 的公开 OpenAI/HTTP 接口只能把
一个 engine 当作完整模型调用；它没有公开“按节点装载任意层或专家、把 activation 交给
外部链路、运行中改变 PP 拓扑、部分迁移 KV/cache”的 stage 控制 API。

推荐路线是：

1. 主线采用自研 `StageRuntime`：调度层直接控制 `prepare_epoch / activate_epoch /
   retire_epoch / generate`，先把参数装载、KV/cache 生命周期、activation 传输、任务中断
   和重配置语义跑通。
2. vLLM 作为可借鉴/可嵌入的高性能组件来源：优先复用其 paged KV、attention kernels、
   静态 PP 经验和 KV connector 思路，而不是直接使用 OpenAI 接口串 stage。
3. 等系统语义稳定后，再选择性 fork vLLM 的局部模块。这个 fork 应被视为独立执行后端，
   不是调度层本身。

## 对照我们的控制接口

| 控制需求 | stock vLLM 对应位置 | 源码判断 | 若 fork vLLM 需要改什么 | 自研路线需要补什么 |
|---|---|---|---|---|
| 装载参数/层 | `vllm/v1/worker/gpu_worker.py::load_model` -> `gpu_model_runner.load_model`，层范围来自 `config/model.py::get_layers_start_end_indices` | 层范围在 worker 初始化和模型构造期确定 | 让 worker 支持运行中加载/卸载 layer module 或 expert shard，并更新模型图/编译缓存/显存池 | 节点 agent 实现按 layer/expert 的权重加载，调度层只下发 `StageAssignment` |
| 卸载参数 | `Executor.sleep/wake_up` 可按 tags 处理 `weights`/`kv_cache` | 粒度是 engine/worker 内部资源，不是任意层区间 | 增加 layer/expert 级 residency ledger、引用计数、drain 后释放 | `retire_epoch` 已有；补真实 runtime 的释放和账本校验 |
| KV/cache | `gpu_worker.initialize_from_config`、`gpu_model_runner.initialize_kv_cache`、KV connector 注册 | KV cache 与 runner、attention backend、block table 绑定紧密 | 暴露 layer-local KV block 的导出/导入/部分迁移，处理旧 epoch 和新 epoch 并存 | `KVBlock.layer_range`、`KVStore`、重配置 copy 已有；补真实 tensor KV 存储后端 |
| 任务分配 | `vllm/v1/executor/abstract.py::execute_model(SchedulerOutput)` | 输入是 vLLM 内部 scheduler 输出，不接受外部 stage task | 新增外部 stage task schema 到 engine/core/scheduler/worker 全链路 | `TaskScheduler` 已能按 placement 生成 stage task |
| 流水线执行 | `gpu_model_runner.execute_model` 返回/接收 `IntermediateTensors`，`parallel_state.get_pp_group().send_tensor_dict/recv_tensor_dict` | PP 通信在固定 torch.distributed group 内完成 | 把 PP tensor-dict 通信替换/扩展成可插拔外部 transport，并保留 TP all-gather 语义 | `ActivationEnvelope` + `Transport` 已有；新增了 Torch 后端验证真实 hidden-state payload |
| 中断与重排 | vLLM 有 preemption/scheduling，但面向 engine 内请求和 KV blocks | 没有节点离开后按外部调度器重新切 layer epoch 的公开路径 | engine core、scheduler、worker、block manager 同时支持 epoch drain/resume | `PipelineReconfigurationCoordinator`、node join/left hook、recovery 框架已有 |
| 动态专家 | `enable_elastic_ep` 与 EPLB 存在，但 `config/parallel.py` 明确排斥 PP 组合 | 更偏 MoE EP 负载均衡，不等价于边缘节点 layer/expert 任意迁移 | 解开 EP/PP 限制，加入专家参数迁移、router 状态和跨节点 expert dispatch | 可参考 MoE-Infinity 的 expert offload/prefetch，实现 expert residency runtime |

## vLLM 中必须牵动的源码面

- `vllm/v1/executor/abstract.py`：`collective_rpc` 和 `execute_model` 是 worker 控制入口。
  注释也说明 RPC 适合控制消息，不能把大 activation 数据面塞进这里。
- `vllm/v1/executor/uniproc_executor.py`、`multiproc_executor.py`、`ray_executor.py`：
  worker 生命周期、分布式初始化、进程/actor 编排在这些 executor 中固定下来。
- `vllm/v1/worker/worker_base.py`：worker 抽象只有 `load_model`、`initialize_from_config`、
  `execute_model` 等内部接口，没有 layer-level load/unload。
- `vllm/v1/worker/gpu_worker.py`：显存 profiling、模型加载、KV cache 初始化、sleep/wake
  和 weight transfer engine 都在这里发生；要做细粒度参数迁移必须进入这一层。
- `vllm/v1/worker/gpu_model_runner.py`：`execute_model`、`_preprocess`、
  `sync_and_gather_intermediate_tensors`、`initialize_kv_cache` 是 PP hidden states 和 KV 的核心路径。
- `vllm/distributed/parallel_state.py`：`get_pp_group()`、`send_tensor_dict()`、
  `recv_tensor_dict()` 绑定固定 PP group；外部链路要替换这一层或在 runner 层旁路它。
- `vllm/distributed/utils.py::get_pp_indices` 与 `VLLM_PP_LAYER_PARTITION`：
  支持不均匀静态层分区，但只适合启动期配置。
- `vllm/config/parallel.py`：`pipeline_parallel_size`、world size、elastic EP 约束都在配置阶段
  固定；动态 PP/EP 混用会触碰配置校验。

## SpotServe 给出的路线证据

SpotServe 没有依赖 vLLM，而是走了更低层路线：Python 全局调度器计算新并行配置，C++ 参数
客户端和 Context Daemon 执行权重/KV buffer 迁移，推理引擎基于修改后的 FasterTransformer。
也就是说，真正的动态调度不是“请求路由器”层能解决的，而是必须有 execution engine 级别的
参数、缓存和中断恢复控制。

这和我们目前的判断一致：控制面可以先独立于具体控制语义成型，但执行后端必须愿意暴露
layer/expert/KV/activation 这些内部对象。stock vLLM OpenAI 接口不暴露这些对象。

## 本次落地

新增 `edge_llm_scheduler.backends.TorchLayeredEngine`：

- `prepare_epoch` 按 `StageAssignment.layer_range` 生成并驻留该 stage 的层权重；
- `activate_epoch` 只允许已准备 epoch 接收新任务；
- `retire_epoch` 释放旧 epoch 的参数，覆盖节点离开或重分层后的卸载路径；
- `generate` 在 stage 0 将 prompt 转成 embedding，后续 stage 校验并反序列化
  `ActivationEnvelope` 中的 PyTorch hidden states；
- 每个 stage 产生 layer-local `KVBlock`，继续走现有 `KVStore` 登记和迁移路径；
- 新测试覆盖三段流水线的真实 tensor activation 传输，以及 epoch drain 后参数卸载。

这个实现不是最终推理内核，但它把“自研后端必须提供的控制接口”落成了可执行契约。下一步
若接真实模型，可以把里面的 toy layer 替换成 Transformers/vLLM kernel 包装，而调度层不需要
再改大结构。
