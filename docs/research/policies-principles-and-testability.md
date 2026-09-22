# 已实现策略：原理、实现与当前可测性

> **对象**：`edge_llm_scheduler/policies/` 下已经写出来的全部策略（5 条主策略 + 3 条默认基线 + 1 个支撑机制）。
> **目的**：说清每条策略**要解决什么问题、判据是什么、代码落在哪一行、参数含义**，以及
> **今天到底能不能测出它的效果**——测得出的给出实测数字，测不出的给出缺口和最小补齐方案。
> **事实源**：策略源码 + 本机实跑输出。所有数字都可用 §8 的命令复现，没有估算值。

---

## 0. 结论先行

| 问题 | 答案 |
|---|---|
| 5 条主策略都实现了吗 | 实现了，且都能跑通 |
| 现在能测效果吗 | **能测三种东西**：路由选择差异（E2 的命中率）、切层划分（算力比例）、恢复/迁移的**语义正确性** |
| 能测端到端收益吗 | **不能**。仿真没有负载回馈，也没把策略接到真实流水线上——见 §6 |
| 仿真里的 `load_std` 可信吗 | **不可信**，是并列取首的产物，不是均衡效果——见 §5.2 |
| 有没有实现缺陷 | 查出 9 处，**已全部修复**（含 2 处语义变更：迁移成本含链路、`reuse_count=0` 的估值）——见 §7 |

一句话：**策略的"机制"已经可测，"收益"还不可测**。二者之间差的不是算法，是三处接线。

---

## 1. 策略全景

| # | 策略 | 文件 | 决策对象 | 输入信号 | 目标 | 参考来源 |
|---|---|---|---|---|---|---|
| 1 | `E2Placement` | `placement.py:76-215` | 请求去哪个节点 | KV 前缀命中块数 + 节点负载 + 驱逐代价 | 命中收益 vs 负载均衡 | Preble E2（exploit/explore） |
| 2 | `CapabilityPlacement` | `placement.py:218-256` | 请求去哪个节点 | `compute_flops` + `load` | 异构下按算力加权 | 经典加权轮询思想 |
| 3 | `PriorityMigration` | `migration.py:124-254` | 节点离开时哪些 KV 块传走 | `reuse_count × prefill_time_ms` + 时间预算 | 时限内保住价值最高的块 | Preble 驱逐价值 $M_i=\sum PT_j N_j$ |
| 4 | `CapabilityReparallelization` | `reparallelization.py:48-107` | 每个节点分几层 | `compute_flops` 占比 | 消除木桶效应 | SpotServe StrategyOptimizer（简化版） |
| 5 | `TokenRecovery` | `recovery.py:57-90` | 中断后从哪继续 | `progress_tokens` + 已提交 KV 块 | 少重算 | SpotServe stateful recovery |
| — | `LayeredPipelinePlacement` | `placement.py:259-311` | 放置计划→任务链 | `ModelPlacement` 列表 + epoch | 支撑机制（非路由决策） | — |
| 基线 | `DefaultPlacement` / `DefaultMigration` / `DefaultRecovery` / `DefaultReparallelization` | 各文件上部 | 最轻负载 / 全传 / 从头重跑 / 不变 | — | 对照组 | — |

共同接口：所有放置策略实现 `place(model, nodes, request, storage, hit_tokens) -> List[Task]`；
迁移实现 `decide(...)/execute(...)`；恢复实现 `recover(task) -> Optional[Task]`；
重并行化实现 `reconfigure(model, nodes, current_placements) -> List[ModelPlacement]`。
**都能通过构造参数注入替换**（`TaskScheduler(placement_policy=..., migration_policy=..., recovery_policy=...)`，
`pipeline_controller` 持有 `placement_policy` 与 `repartition` 策略），这是仿真能对比策略的前提。

---

## 2. 逐条拆解

### 2.1 `E2Placement` — KV 缓存感知路由

**要解决的问题**：多请求共享长前缀时（系统提示、few-shot 示例、同一文档的连续提问），
命中缓存的节点能省掉整段 prefill；但把同一前缀的所有请求都压到那一个节点上，热点节点会过载。

**原理**：把决策写成二选一，判据是"命中省下的 token 数"与"还要现算的 token 数"谁多：

```
matched_tokens   = 命中块数 × block_size_tokens
remaining_tokens = prompt_len − matched_tokens

if matched_tokens > remaining_tokens:   # 省得比算得多
    走 EXPLOIT：在"持有该前缀的节点"里选负载最轻的
else:
    走 EXPLORE：选"prompt-aware 负载"最轻的
```

- EXPLOIT 只用收益明显时才用，避免"为了命中 2 个 token 把请求塞进热点"。
- EXPLORE 的"负载"不是裸 `load`，而是 `load + min(1, 驱逐代价 / 最大驱逐代价)`；
  驱逐代价取该节点上**价值最低**的块（`reuse_count × prefill_time_ms`）——因为去轻节点往往要顶掉它已有的 KV，
  而被顶掉的块是要重算的，这也是成本。

**实现要点**（`placement.py`）：

| 位置 | 作用 |
|---|---|
| `:119-122` | `_hash_prompt` 把 prompt 切成块哈希序列 → `storage.lookup(prefix_hashes)` 返回**连续**命中块数 |
| `:125-126` | 调用方传入的 `hit_tokens` 更大时覆盖（调度器已经查过一遍，见 §3） |
| `:131-139` | exploit 分支：`_find_hit_holders` 从 KV 索引反查持有者集合，过滤 `load < load_threshold(0.95)`，`min(load)` 选一个；无可用者退回 explore |
| `:189-215` | explore 分支：遍历索引算每节点驱逐代价，`total_cost` 取最小 |
| `:95` | `hash_fn` 可注入——**测试与仿真必须注入**，见下面的坑 |

**参数**：`block_size_tokens=16`（块↔token 换算）、`load_threshold=0.95`、`storage_hit_weight=1.0`（预留未用）、
`hash_fn`（默认 `_default_hash`）。

**⚠️ 默认 hash 对 `str` prompt 是退化的**：`_default_hash` 对字符串走
`tokens = list(range(len(prompt)))`——只取长度，丢掉内容。两个词数相同、内容不同的 prompt 会算出同样的块哈希。
当前调用方（仿真、调度器）都传 token 列表，所以没有暴露；但 `InferenceRequest.prompt` 的类型是 `Any`（`types.py:239`），
字符串是允许的。真实语义应对齐 LMCache 的 key 构造，不该依赖这个默认实现。

### 2.2 `CapabilityPlacement` — 算力加权路由

**要解决的问题**：异构集群里"负载相同"不等于"占用相同"。同样 `load=0.5`，算力 2× 的节点实际更闲。

**原理**：定义有效负载

$$\text{effective\_load}(n) = \frac{\text{load}(n)}{w(n)},\qquad w(n) = \frac{\text{compute\_flops}(n)}{\text{avg}(\text{compute\_flops})}$$

取 `effective_load` 最小的节点。$w>1$（强于平均）时有效负载被压低 → 优先被选中。权重相对平均值归一化，
所以集群整体算力水平不影响排序，只有**相对**算力影响。

**实现**：`placement.py:241-247`，共 7 行核心逻辑。权重下限 `1e-9` 防止除零。

**与 E2 的关系**：两者都返回**单个** Task（整请求放一个节点），互斥使用；`CapabilityPlacement` 不看 KV，
`E2Placement` 不看算力。**"缓存感知 + 算力加权"的组合决策还没有实现**——这是一个明确的空白。

### 2.3 `PriorityMigration` — 价值优先 KV 迁移

**要解决的问题**：节点会走（合盖、抱走、断网），它上面的 KV 不会等你。全传可能来不及，
所以要**在时限内挑价值最高的块传**，剩下的丢掉（丢了以后按需重算）。

**原理**：块价值沿用 Preble 的驱逐价值公式

$$\text{priority}(b) = \text{reuse\_count}(b)\times \text{prefill\_time\_ms}(b)$$

- `reuse_count`：历史被复用次数（热度 $N_j$）
- `prefill_time_ms`：重算这段 KV 的 prefill 耗时（$PT_j$）

注意这个公式的**后果**：`reuse_count=0` 的块优先级恒为 0——再贵也会最先被丢。
对"只被用过一次的长 prompt"这可能是错的（它不可复用但重算很贵）。是否需要把"不可复用但重算贵"单独建项，是一个待定问题。

**算法**（`decide`，`:148-201`）分四步：

1. 逐块 `_load_block`（先试 `node:gpu`/`:cpu`/`:disk`），拿不到就算丢弃；
2. 按 priority 降序排序；
3. 选目标节点：`available_nodes` 中 `alive` 且通过 `target_check_fn`（默认 `memory_free > 0`）；
4. **贪心装预算**：`cost_ms = storage.estimate_move_cost(block, src, dst)`，`cost_ms <= budget_ms` 就装进去并扣减预算，否则丢弃。

**参数**：`deadline_ms`（默认 `inf`＝全传）、`bandwidth_mbps`（默认 100，**只用于兜底**，实际成本由 storage 算）、
`target_check_fn`。

**实测行为**（4 个块，`byte_size=8192`，100 MB/s → 每块 0.08192 ms）：

| 块 | reuse_count | prefill_ms | priority | `deadline=inf` | `0.25ms` | `0.10ms` |
|---|---|---|---|---|---|---|
| 11 | 5 | 40 | **200** | 传 | 传 | 传 |
| 13 | 9 | 10 | **90** | 传 | 传 | 丢 |
| 12 | 1 | 40 | **40** | 传 | 传 | 丢 |
| 14 | 0 | 99 | **0** | 传 | 丢 | 丢 |

预算 0.25 ms 装得下 3 块（0.24576 ms），第 4 块剩 0.00424 ms 装不下 → 丢。降序 + 贪心的语义符合预期。

**⚠️ 三个实现局限**：

1. **所有块都去了同一个目标**：`_assign_target` 用 `min(targets, key=lambda n: n.state.load)`，
   而 `load` 是心跳里的静态值，**不会因为本次已分配了块而增加**，并列时 `min` 取列表第一个。
   实测 4 个块全部落到 `n1`，`n2` 一个都没拿到。要真正分摊，得在分配循环里维护"已分配字节"的局部账本。
2. **成本模型看不见链路**：`MockKVStore.estimate_move_cost = byte_size / bandwidth_mbps × 0.001`，
   **完全忽略 `src`/`dst`**（实测传 `n0:gpu→n1:gpu` 与任意组合同价）。`NodeCapability.bandwidth`（MB/s）
   在整个迁移路径上没有被读过。结果是"跨机 vs 本机"成本一样，预算分配失去了它最该有的区分度。
3. **没有多副本/冗余**：`move` 是搬移语义（删源）。源节点正要走，这没错；但"热点块提前复制到备份节点"
   这条真正的容错手段还没实现（文件头部 TODO 里记着）。

### 2.4 `CapabilityReparallelization` — 算力比例重切层

**要解决的问题**：节点集合变化后（加入/离开）若不重排，就会出现"强节点闲着、弱节点拖尾"。
流水线的吞吐由**最慢的那一段**决定，所以层数必须按算力占比分。

**原理**：把 `num_layers` 按 $w(n)$ 切成整数块，每个节点拿到**连续**层区间：

$$
\text{raw}(n) = w(n)\cdot L,\qquad \text{assigned}(n)=\lfloor \text{raw}(n)\rfloor
$$

余数 $L-\sum\lfloor\text{raw}\rfloor$ 按**小数部分从大到小**补给。区间按节点顺序累积成
`(start, start+cnt)`，保证连续覆盖 $[0,L)$。

**实测切分**（$L=12$）：

| 算力 | 结果 | 说明 |
|---|---|---|
| 100 / 100 / 100 | 4 / 4 / 4 | 同构 → 均分 |
| 100 / 150 / 200 | 3 / 4 / 5 | 比例 2:3:4，余数 1 给小数部分最大的 n0（0.667） |
| 20 / 100 / 400 | 1 / 2 / 9 | 比例近 1:2:9 |
| 50 / 400 | 1 / 11 | 两节点，弱节点只拿 1 层 |
| 100 × 4 | 3 / 3 / 3 / 3 | 四机同构 → 均分（12/4 整除） |

**⚠️ 注释与实现不符**：`:74` 的注释写"把余数给**权重最大**的"，实现（`:79-82`）是按**小数部分**降序。
上面 `100/150/200` 的例子就是反例：权重最大的是 n2（0.444），但余数给了 n0（小数部分 0.667 最大）。
实现更合理（它才是真正的"最大余数法"），**要改的是注释**。

**⚠️ 它用的不是实测算力**：输入是 `NodeCapability.compute_flops`——一个由调用方填的数字。
仿真里是 `100 + i*50`（人为设定），真实部署里必须先 profiling 才能填对。
我们已经实测到"**7B 在 8+6+4+4GB 上，平均切分失败、按实测每层成本加权切成 12/4/4/8 才成功**"，
说明该策略的**输入质量**比算法本身更关键：真实接入应喂实测的每层耗时，而不是 TFLOPS 标称值。
另注：该策略只按**算力**分，不按**显存**分——显存不够时会切出装不下的区间（仿真里没暴露，因为 mock 不校验）。

### 2.5 `TokenRecovery` — token 级恢复

**要解决的问题**：生成到一半节点走了，已经算出来的 token 不该丢。从头重跑把已完成的 prefill 和 decode 全废掉。

**原理**：自回归的因果性保证——前 `progress_tokens` 个 token 的 KV 一旦写成就不会再变，恢复时接着往下算即可。

**实现**（`recovery.py:70-90`）：新建一个 Task，改三处：

| 字段 | `DefaultRecovery` | `TokenRecovery` |
|---|---|---|
| `task_id` | 新 uuid | 新 uuid |
| `node_id` | 原节点 | `resume_node_id or 原节点` |
| `kv_block_ids` | 原样带上 | 原样带上（已提交 KV 复用） |
| `progress_tokens` | **0（从头）** | **原值（从这继续）** |
| `max_retries` | 2 | 3 |

实测：`Task(progress_tokens=25, kv_block_ids=[7,8])` 恢复后，
`DefaultRecovery` 给出 `progress=0, kv=[7,8]`，`TokenRecovery` 给出 `progress=25, kv=[7,8]`；
`_retries` 达到 `max_retries` 时返回 `None`（放弃），实测 `_retries=3` → `None`。

**⚠️ 跨节点恢复在真实引擎上是错的**（重要）：
`hf_layered_engine.generate`（`:258-263`）在 stage 0 看到 `progress_tokens > 0` 时只喂**最后一个 token**：

```python
if task.progress_tokens > 0:
    tokens = tokens[-1:]          # 只喂最后一个 token
out = self._forward(tokens=None, token_ids=tokens, request_id=task.request_id)
```

而 `_forward` 用的是**进程内 per-request 的 `past_key_values`**（`_cache_for`，`:487-495`；key 是 `(epoch, request_id)`）。
于是：

- **同节点、同 epoch、引擎进程没重启** → cache 还在，喂最后一个 token 就是正确的 decode 步，**真能少算**；
- **换到别的节点（`resume_node_id`）或引擎重启** → `_cache_for` 会建一个**空** `DynamicCache`，
  只喂最后一个 token 等于让模型看一个 1-token 序列，输出相对原上下文是错的。
  没有任何代码把 `kv_block_ids` 里的 KV 装回目标节点的 cache。

所以 `TokenRecovery` 目前是「**字段带对了，数据没带过去**」。跨节点恢复要成立，必须补一条真实的 KV 传输/装载路径
（这也正是 §6 里"真实 KV 迁移未实现"的同一件事）。

### 2.6 支撑机制：`LayeredPipelinePlacement` 与 `pipeline_epoch`

它不是路由决策，而是把**已经决定好的**层划分展开成任务链（`placement.py:280-311`）：
按 `layer_range[0]` 排序、过滤掉已不健康的节点、为每段生成一个 Task，并写入
`stage_index` / `stage_count` / `pipeline_epoch`；`hit_tokens` **只给 stage 0**（其余段没有 prompt，命中无意义）。

`set_plan(plan)` 让**后续进入**的请求绑定新 epoch，而**旧请求继续带旧 epoch 执行完**——
这是"重配置不打断在途请求"的关键（`pipeline_controller.py:97` 调它）。
真实引擎侧 `hf_layered_engine.generate` 开头就校验 epoch（`:248-251`），
任务 epoch 与节点活跃 epoch 不符直接抛错——**串线会被立刻发现，而不是静默算错**。

另：调度器 `task_scheduler.py:127-133` 会**覆盖**放置策略写好的 `prompt/max_tokens/prompt_len/hit_tokens/stage_index/stage_count`。
其中 `task.hit_tokens = hit`（`:131`）对**所有**段都赋了同一个命中值，抹掉了 `LayeredPipelinePlacement` 的"只给 stage 0"。
当前不构成 bug——因为三个引擎都自己重新判了（`hf_layered_engine.py:292`、`torch_layered_engine.py:178`、
`layered_mock_engine.py:163` 都是 `hit_tokens if start/stage_index == 0 else 0`）——但两处重复计算同一件事，
将来改一处忘一处就会出问题。

---

## 3. 一条请求走完所有策略（接缝在哪）

```
generate(req)
  └─ route_request(req)                                  task_scheduler.py:114
       ├─ prefix_hashes = _hashes_from_prompt(prompt)
       ├─ hit = storage.lookup(prefix_hashes)            ← 命中的连续块数
       ├─ tasks = placement.place(model=None, nodes=healthy,
       │                          request=req, storage, hit_tokens=hit)
       │        ↑ 这里决定：E2 走 exploit 还是 explore；Capability 按算力加权
       └─ 回填 task 的 prompt / max_tokens / prompt_len / hit_tokens / stage_*
  ├─ 单任务 → _run_task(tasks[0])                        :102
  └─ 多任务 → _run_pipeline(tasks)                       :136
       ├─ 按 layer_range[0] 排序
       ├─ 跨节点时 _deliver_activation() 送 ActivationEnvelope（带 epoch + 校验和）
       └─ 逐段 _run_task → engine.generate()
            ├─ 失败/中断 → recovery.recover(task) → 递归重跑       :228
            └─ 成功 → KV 块 save 到 "{node}:gpu" + 发 TASK_DONE     :210-217

节点离开（NODE_LEFT 事件）_on_node_left()                 :238
  ├─ 从 KV 索引收集该节点持有的块
  ├─ 每个未完成任务 → recovery.recover() → _run_task(retry)   :256-261
  ├─ migration.decide(node_id, blocks, storage, remaining)     :264
  │     └─ PriorityMigration：PT×N 降序 + deadline 预算
  ├─ migration.execute(plan, storage, transport)               :268
  └─ pipeline_controller.reconfigure_for_topology()            :270
        └─ repartition.reconfigure(...) → 新 ModelPlacement 列表
             └─ placement_policy.set_plan(plan)  ← epoch+1，新请求走新切分   pipeline_controller.py:97

节点加入（NODE_JOINED 事件）_on_node_joined()              :272
  └─ reconfigure_for_topology()（同上）
```

**接缝上发现的一处断链**：`:256-261` 里恢复出来的 retry task 是 `await self._run_task(retry)` **直接丢掉返回值**的——
被中断的那个请求的结果不会回到调用 `generate()` 的一方，`self._results[request_id]` 也不会被更新。
恢复动作发生了、KV 迁移发生了，但**恢复的效果没有任何地方被观测到**。这是把策略接上"可测量"的关键堵点之一。

---

## 4. 现在能测什么（全部为实跑输出）

命令见 §8。环境：WSL2 Ubuntu 24.04 / `~/venvs/pair` / torch 2.6.0+cu124。

### 4.1 路由选择：E2 vs 基线（无需 GPU）

```
dataset: 30 requests, 9 shared prefixes

[default]    hit_rate=0.321  node_load={'node_0': 30}                 load_std=0.0
[e2]         hit_rate=0.962  node_load={'node_0': 14, 'node_1': 8, 'node_2': 8}  load_std=2.8
[capability] hit_rate=0.321  node_load={'node_0': 30}                 load_std=0.0

compare: E2 vs Default hit_rate improvement = 0.641
```

`hit_rate` = 被路由到的节点**确实持有**该前缀缓存的那部分 prompt token 占比（口径：命中 token / 总 prompt token）。
这个比较是**非平凡**的：ground truth（哪个前缀在哪台机器）由数据集初始化时独立决定（`run_simulation.py:90-102`），
策略只是被"给了或没给"KV 索引的可见性——`E2` 用它，`Default`/`Capability` 完全不看它。
所以 0.321 → 0.962 测的是「**缓存感知 vs 缓存盲**」这件事本身，不是调参调出来的。

### 4.2 切层划分：谁拿几层（无需 GPU）

`run_layered_simulation.py` 端到端跑通控制流：

```
node registered: Node(node_0, mem_free=1932735283, load=0.00) ×3
reparallelized 12 layers over 3 nodes: node_0=(0, 3), node_1=(3, 7), node_2=(7, 12)
EventBus started
request layered-demo done: tokens=8 hit=0 latency=86.5ms err=None
result=[layered mock output from node_2] tokens=8 stages=3
  kv_blocks=3 locations={...: ['node_0:gpu'], ...: ['node_1:gpu'], ...: ['node_2:gpu']}
```

三层任务链建起来了、activation 逐段传递了、每段的 KV 落在了各自节点上。
配合 §2.4 的 5 组切分探针，"按算力比例分层 + 连续覆盖 + 余数分配"这个算法**行为已完全确定**。

### 4.3 策略相关单测

```
22 passed in 4.49s
```

（`test_e2_placement` / `test_capability_placement` / `test_priority_migration` /
`test_reparallelization_recovery` / `test_pipeline_reconfiguration`）

**全量套件**（含 §7 八处修复的回归测试 `test_policy_fixes.py`）：

```
92 passed, 3 skipped in 17.29s
```

跳过项：LMCache ×2、vLLM ×1（对应库未装）。

---

## 5. 哪些数字可信，哪些不可信

这一节比 §4 重要：**同一个输出里，有的指标有效，有的无效。**

### 5.1 可信：`hit_rate`

理由如上：ground truth 独立于策略，指标口径与策略目标一致（省下的 prefill token 数）。
**上限**：它是"命中 token 比例"，不是"TTFT 降低比例"。要换算成时间，还需要"每 token prefill 耗时"这个系数；
而命中并不等于白拿——从 `node_x:gpu` 取回 KV 也要传数据，这部分没进模型。

### 5.2 不可信：`load_std` / `node_load` 作为均衡指标

两个原因，都能从代码直接判定：

1. **仿真从不回馈负载**。初始 `load = 0.05 + 0.05*i`（`run_simulation.py:45`），
   之后只调用 `scheduler.route_request()`，**没有任何地方写回 `node.state.load`**。
   于是 `DefaultPlacement`（`min(load)`）永远是 node_0，`CapabilityPlacement`
   （`effective_load`：0.075 / 0.100 / 0.1125）**也**永远选 node_0 —— 所以上面两者输出**完全相同**（都是 30/0/0）。
   也就是说：**这次实验根本无法区分负载均衡类策略的优劣**，`[capability]` 那一行等于 `[default]` 的复制。
2. **E2 的 14/8/8 不是均衡的结果**。它的分布来自 exploit（去持有者）与 explore，
   而 explore 的 `total_cost` 在三个节点驱逐代价相同时并列，`min` 取**列表第一个** → 又回到 node_0。
   所以 `load_std=2.8` 是"并列取首 + 前缀归属"的产物，**不能读作"E2 更均衡"**。

**结论**：现在这份仿真只能支撑 §5.1 的命中率结论；任何关于"负载更均衡 / 吞吐更高"的说法都缺证据。

### 5.3 不可信：迁移成本

`estimate_move_cost` 忽略 `src`/`dst`（§2.3 局限 2），实测 1 MB 块在 100 MB/s 下恒为 10.0 ms，
换任何源/目标都一样。所以"deadline 内传了几块"这个结果**只对块大小和 deadline 敏感**，
对网络差异完全不敏感——用它来论证"跨机迁移代价高"是循环论证。

---

## 6. 现在**测不了**的，以及最小补齐方案

按"投入产出比"排序。前三项都不需要新硬件，也不需要模型。

| # | 测不了的东西 | 为什么 | 最小补齐 | 规模 |
|---|---|---|---|---|
| 1 | 均衡类策略的优劣（Default vs Capability vs E2-explore） | 仿真无负载回馈（§5.2） | 任务完成后写回 `node.state.load`（按 `compute_ms` 或 token 数增量），并让 `_assign_target`/`_explore_target` 用局部账本 | 小，主要是 `run_simulation.py` |
| 2 | 策略在波动下的稳定性 | 没有 churn 场景、没有重复与方差 | 加"每 N 个请求触发一次节点离开/加入"+ 多 seed 重复 + 报均值/标准差 | 小，复用已有 `NODE_LEFT` 事件 |
| 3 | 切层收益的**端到端**效果 | `run_real_pipeline.py` 用 `--weights` **手工**给切分、`--leave-node` **手工**触发，**不经过** `CapabilityReparallelization` | 让 `run_real_pipeline` 用策略算切分，输入喂**实测每层耗时**而非 TFLOPS 标称值 | 中，接线为主 |
| 4 | 迁移的**真实**收益 | 真实 KV 迁移未实现；成本是线性估算（§5.3） | 两个 agent 进程 + loopback 搬真实 KV 块，量 `move` 实际耗时与网络字节 | 中，是独立工作项 |
| 5 | 恢复的收益 | 跨节点恢复语义未闭合（§2.5）；恢复结果被丢弃（§3 断链） | 先修断链（把 retry 的结果回填），再决定是否实现 KV 装载 | 小修 + 大实现 |

**注意 #3 的性质**：它不改算法，只是把已有的两半接起来——策略那一半写好了、真实流水线那一半跑通了，
中间少一根线。这是"策略从可测机制变成可测收益"最短的一步。

---

## 7. 实现缺陷清单与修复状态

全部已修复，回归测试在 `edge_llm_scheduler/tests/test_policy_fixes.py`（每条对应一个测试）。

| # | 位置 | 问题 | 修法 | 副作用 / 注意 |
|---|---|---|---|---|
| 1 | `placement.py` | **可复现崩溃**：`prompt=None` 且 `hit_tokens>0` 时读未绑定的 `prefix_hashes`，抛 `UnboundLocalError` | 函数开头初始化 `prefix_hashes: List[int] = []`，并在 exploit 判据里加 `and prefix_hashes` | 无 |
| 2 | `reparallelization.py:74` | 注释"余数给权重最大的"与实现（小数部分最大）不符 | 改注释为"最大余数法" | 无（实现本来就是对的） |
| 3 | `migration.py` `_assign_target` | 所有块落到同一个目标节点（静态 `load` 并列时 `min` 取列表第一个） | 引入本次分配账本 `assigned_bytes`，评分 = `load + 已分配字节/剩余显存` | 分配会摊开；同等负载下不再全压一个节点 |
| 4 | `mock_storage.py` `estimate_move_cost` | 完全忽略 `src`/`dst`，本机与跨机同价；`NodeCapability.bandwidth` 在迁移路径上从未被使用 | 传输按两端较慢链路算，跨机加一次 RTT；新增 `link_bandwidth_mbps` / `link_rtt_ms` / `local_rtt_ms=0.14` / `remote_rtt_ms=5.0`（取自本机 loopback 实测与 WiFi 分档） | **成本变大**：跨机一块 1000B 从 0.01ms 变成 ~5.01ms。两个依赖旧硬编码预算的测试改为按 storage 报的实际成本定预算 |
| 5 | `migration.py` priority 公式 | `reuse_count=0` 的块优先级恒为 0，再贵也最先丢 | 新增 `min_reuse_count`（默认 **1**）：显存里存在的块至少被用过一次，按"未来至少再用一次"估值。实测 `reuse=0,prefill=99`（99）现在优先于 `reuse=9,prefill=10`（90） | **语义变更**，论文里要说明；设 `min_reuse_count=0` 可退回 Preble 原式 |
| 6 | `hf_layered_engine.py` | 跨节点恢复只带字段不带数据：目标节点 `_cache_for` 建**空** cache，stage 0 仍只喂最后一个 token → 输出语义错误 | 新增 `_has_cache()`；cache 不存在时**退回整段重算**并打日志（慢但正确） | 跨节点恢复现在是"正确但慢"，不是"快但错" |
| 7 | `task_scheduler.py` `_on_node_left` | 恢复出的 retry 结果被丢弃，中断请求的结果不回传、`self._results` 不更新 | 回填 `self._results[request_id]` 并发 `TASK_DONE` 事件 | 恢复动作第一次变得可观测 |
| 8 | `task_scheduler.py:131` + 三个引擎 | `hit_tokens` 被调度器对**所有**段覆盖为同一值 | 统一到调度器：`hit_tokens = hit if index == 0 else 0`；引擎侧的重判保留为冗余保险 | 行为不变（引擎本来就会纠正），但不再是两处各判一次 |
| 9 | `hf_layered_engine.py` `kv_bytes` | **KV 计量恒返回 0**：用 `cache[i]` 取值，而 transformers 5.x 的 `DynamicCache` 不可下标，`TypeError` 被 `except ... continue` 吞掉。四机运行里每个 stage 的 `kv_bytes_last: 0` 就是这个 | 改为遍历 `cache.layers[i].keys/.values`（兼容 4.4x 的 `key_cache/value_cache`）；`status()` 里按请求上报 KV 字节 | 实测修复后：6 层 38 token = 116,736 B，每 decode 步 +3072 B，释放归 0。**注意显存增量不能当 KV**（实测是真实 KV 的 74 倍且释放后不降）。回归测试 `tests/test_kv_accounting.py` |

**⚠️ 仍未闭合的一条（不是可修的 bug，是缺的能力）**：`hf_layered_engine.generate` 把"从 `progress_tokens` 续算"
近似为"只喂 `task.prompt` 的最后一个 token"。这只有在**恢复时 `task.prompt` 已经是"到中断点为止的完整序列"**
时才正确；控制器目前并没有把已生成的 token 拼回 prompt。所以跨节点恢复这条路径，
**字段传递对了、引擎侧不再产生错误输出，但"少算"的收益还没真正拿到**——
要拿到必须补一条真实的 KV 传输/装载路径（与 §6 表里的真实 KV 迁移是同一件事）。

---

## 8. 复现命令

WSL 内（`~/pair` 是 WSL 里的 ASCII 软链，指向 Windows 侧工作区）：

```bash
# ① 路由策略对比（mock，无需 GPU）
python -m edge_llm_scheduler.experiments.run_simulation --policy default e2 capability --nodes 3

# ② 分层流水线控制流（mock）
python -m edge_llm_scheduler.experiments.run_layered_simulation --nodes 3

# ③ 策略单测
python -m pytest edge_llm_scheduler/tests/test_e2_placement.py \
  edge_llm_scheduler/tests/test_capability_placement.py \
  edge_llm_scheduler/tests/test_priority_migration.py \
  edge_llm_scheduler/tests/test_reparallelization_recovery.py \
  edge_llm_scheduler/tests/test_pipeline_reconfiguration.py -q

# ④ 策略行为探针（§2.4 切分表、§2.3 预算表、§7 缺陷 1/3/4 的证据）
PYTHONPATH=. python scripts/probe_policies.py
```

Windows 侧一键（走工作区密钥 + ssh）：

```powershell
$script = Get-Content project\scripts\wsl_run_policy_sim.sh -Raw
$script | & "$env:SystemRoot\System32\OpenSSH\ssh.exe" `
  -i project\.ssh\agent_ed25519 -o UserKnownHostsFile=project\.ssh\known_hosts `
  -o BatchMode=yes -p 22 yychen@127.0.0.1 "tr -d '\r' | bash -s"
```

---

## 9. 相关文档

| 想知道 | 看 |
|---|---|
| 五层全景、四条硬约束、10 条实测结论、8 条路径汇总 | `docs/research/inference-optimization-understanding.md` |
| 算子层细节、请求×设备切分、LMCache 算子清单 | `docs/research/kernels-attention-and-memory-primer.md` |
| L2/L3 可定制面与可复用实现、能力矩阵 | `docs/research/l2-l3-customization-and-reuse.md` |
| 真实四机部署实测（PP/TP/并发/异构切层） | `docs/deploy/edge-4gpu-deployment-analysis.md` |
| 四机整合计划与分阶段安排 | `docs/deploy/four-laptop-integration-plan.md` |
