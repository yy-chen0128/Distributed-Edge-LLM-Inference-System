# 模拟测试可行性分析

> 日期：2026-08-16
> 目的：回答两个问题——① 用什么数据集模拟测试（真实数据集 vs 合成）；② vLLM/LMCache 能否在无 GPU 下模拟（离线/mock/纯 CPU）。
> 结论先行：**可以完全无 GPU 做模拟测试**——LMCache 有纯 CPU 后端，vLLM 有离线+随机权重模式；数据集优先用 MoE-Infinity 的共享前缀 JSON fixture。

---

## 一、数据集：选什么、为什么

### 1.1 各论文实现的数据集盘点

| 来源 | 数据集 | 格式 | 是否适合我们 |
|---|---|---|---|
| **MoE-Infinity** `benchmarks/contextpilot/fixtures/` | shared_prefix_rag、multi_turn_conversation、longctx_shared_prefix_32k（32K 共享前缀）、longctx_tinysuffix_32k、batch_with_overlap | **JSON，自带 `metadata.overlap_ratio` + 每条 `context_overlap_with_prev`** | ★**最适合**——就是为测"跨请求前缀缓存复用"设计的 |
| **Preble** `benchmark_workload_gen.py` | WorkloadPrefixDataLoader（多前缀+冷请求）、ToolBench（EVEN/ZIPF/NORMAL）、LooGLE 长文档 QA、APPS 编程、multi_document_qa（真实 PDF RAG） | prompt + sampling_params | ★很合适——前缀结构可精确控制；`is_hot()/get_prefix_index()` 可复用 |
| **Preble** `multi_turn_chat` | 合成随机字符串 | 每轮 256~512 token 会话 | 部分——测多轮，但 prompt 是乱码 |
| **Helix** trace_generator | SharedGPT/Alpaca/AzureCode/AzureConversation 的**长度统计**（pkl） | 只有 (到达时间, input_len, output_len) | ✗长度-only，无前缀结构——但**到达率采样**有价值 |
| **SpotServe** trace | query_trace.csv（长度）+ node add/remove 事件流 | 只有序列长度 | ✗长度-only，但 **node 伸缩事件 trace** 极有价值 |

### 1.2 结论与选择

**主负载**：MoE-Infinity `contextpilot/fixtures/` 的 JSON（共享前缀结构最清晰，可直接喂给 KVStore.lookup 算命中）。用 `gen_longctx_workload.py` 可按参数生成任意"共享前缀 token 数 / 独特后缀"比例的长 prompt——正好测我们的 E2 缓存感知路由。

**补充负载**：Preble 的 WorkloadPrefixDataLoader（多前缀+冷请求混合，测"热前缀 vs 冷请求"的路由权衡）。

**到达时间**：用 Preble 的指数分布（`np.random.exponential(1/rps)`）或 Helix 的 Azure trace 到达率。

**节点动态轨迹**：用 SpotServe 的 node add/remove trace 驱动我们的 NODE_LEFT/NODE_JOINED 事件——这是测重并行化/迁移/恢复策略的关键输入。

### 1.3 统一请求格式（归一化）

论文负载格式五花八门，我们归一化成统一格式，让所有数据源可喂进同一个模拟器：

```json
{
  "request_id": "...",
  "arrival_time": 0.0,
  "input_ids": [1,2,3,...],          // 或 "prefix_id" + "suffix_ids"
  "prefix_id": "hot_prefix_0",       // 前缀标识（共享请求用同一 id → 可命中）
  "suffix_ids": [...],               // 独特部分
  "max_new_tokens": 64
}
```
- `prefix_id` 相同的请求 → KV 前缀可复用（对齐我们的 KVBlock.hash + KVStore.lookup）
- `input_ids` 用于真实语义；`prefix_id` 用于模拟命中判断

## 二、vLLM / LMCache 能否无 GPU 模拟？—— 能

### 2.1 LMCache 纯 CPU（已核实存在）
| 组件 | 路径 | 说明 |
|---|---|---|
| **MockConnector** | `lmcache/v1/storage_backend/connector/mock_connector.py` | 纯 CPU"远端"后端，`mock://SIZE/?peeking_latency=L&read_throughput=G&write_throughput=G` 用延迟/吞吐参数模拟网络 |
| **LocalCPUBackend** | `lmcache/v1/storage_backend/local_cpu_backend.py` | 默认分配器，无 CUDA 时自动 `dst_device="cpu"` |
| **LocalDiskBackend** | `lmcache/v1/storage_backend/local_disk_backend.py` | 磁盘后端（O_DIRECT），纯 CPU |
| **MockMemoryChannel** | `lmcache/v1/transfer_channel/mock_memory_channel.py` | 单进程内全局 dict 模拟跨节点张量传输 + sleep 模拟延迟 |
| **Standalone Starter** | `lmcache/v1/standalone/__main__.py` | 明确"Works without vLLM or GPU"，`python -m lmcache.v1.standalone --device=cpu` |
| 内部 API server | `docs/.../internal_api_server/vllm_apis.rst` | `/cache/store`（mock token）等 HTTP 端点，无 vLLM 可验证存储 |

**结论**：LMCache 的 KV 存储/传输逻辑可完全在 CPU 上验证（MockConnector + LocalCPUBackend + standalone）。

### 2.2 vLLM 模拟模式
| 模式 | 方式 | 说明 |
|---|---|---|
| **离线模式** | `vllm.entrypoints.llm.LLM` 类 + `llm.generate()` | 进程内推理，不起 HTTP server |
| **随机权重** | `--load-format dummy` | 不用下载模型，随机权重跑流程 |
| **CPU 推理** | `LLM(..., device="cpu")` | 真实但慢 |

**结论**：vLLM 有离线 + dummy 权重模式，可以无 GPU 跑流程（CPU 推理慢但可验证逻辑）。

### 2.3 但本机现状
- **本机未安装 vLLM/LMCache/sglang**（`pip show` 全空），只有 `torch 2.5.1+cu121`。
- 两条路：① 安装 LMCache（纯 CPU 可跑，不依赖 vLLM/GPU）；② 安装 vLLM（离线+dummy，CPU 慢但可用）。
- 我们的 `LMCacheStore` 已做契约测试（FakeLMCacheEngine），真实 LMCache 纯 CPU 是下一步验证。

## 三、我们的模拟测试架构

```
┌─────────────────────────────────────────────┐
│ 负载源（归一化 JSON）：                       │
│   MoE-Infinity contextpilot fixtures        │
│   Preble WorkloadPrefixDataLoader           │
│   SpotServe node add/remove trace           │
├─────────────────────────────────────────────┤
│ 模拟器（edge_llm_scheduler + 数据集加载器）   │
│   请求流 → E2路由 → 引擎 → KV存取/迁移       │
├─────────────────────────────────────────────┤
│ 存储：LMCache (CPU: MockConnector/LocalCPU) │
│       或 MockKVStore                        │
│ 引擎：vLLM 离线(dummy) 或 MockEngine         │
└─────────────────────────────────────────────┘
```

## 四、下一步
1. 写数据集加载器 `edge_llm_scheduler/backends/datasets.py`（解析 contextpilot JSON + 归一化）
2. 装 LMCache 纯 CPU 跑真实存储逻辑（可选，先 MockKVStore）
3. 用共享前缀负载测 E2Placement 的命中率提升（对比 DefaultPlacement）
4. 用 SpotServe node trace 测节点离开/加入时的迁移/恢复

## 五、真实对接测试现状（2026-08-16 补充）

**项目已 clone**：`project/LMCache`（官方）、`project/vllm`（官方，含内置 `lmcache_connector.py`）。

**阻塞**：本机 torch 损坏（目录空的只有元数据），且网络下载 torch wheel 反复失败
（官方源 hash 错误 / 清华源超时 / curl 断连，122MB 3 种方式均失败）。vLLM/LMCache 均依赖 torch。

**已做（不依赖 torch）**：
1. **AST 静态签名验证**（严格）：用 Python ast 解析真实源码方法参数，与我们的调用逐一比对——
   `LMCacheStore` 的 store/retrieve/lookup/move/clear 全部匹配真实 `LMCacheEngine` 签名；
   `VLLMEngine` payload 字段（model/prompt/max_tokens/temperature/stream）匹配真实
   vLLM `CompletionRequest`。
2. **真实对接测试脚本**：`test_lmcache_real.py`（真实 LMCacheEngine CPU 模式）、
   `test_vllm_real.py`（真实 vLLM 离线 LLM() + dummy 权重）——检测到 torch/vllm 就自动跑，
   否则 skip。该记录反映当时的阶段性快照；当前完整测试以
   `docs/plan/current-work-and-roadmap.md` 中记录的结果为准。

**环境就绪后**（torch 装好）：这两个测试自动生效，即可真实验证对接。
修复 torch 建议：`pip install torch --index-url https://download.pytorch.org/whl/cpu`
（需稳定的网络，或手动下载 wheel 后本地安装）。
