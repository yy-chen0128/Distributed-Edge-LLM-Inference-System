# 项目概览

> 用途：初始化项目范围，减少后续上下文丢失。

## 一、基础信息

- 项目名称：异构分布式算力上大模型训练与推理的任务调度与弹性资源管理
- 创建日期：2026-08-06
- 最后更新：2026-08-06
- 当前负责人：

## 二、研究定位

- 论文题目：（待定）
- 学科类型：计算机系统 / 分布式系统 / 高性能计算
- 论文类型：研究论文
- 目标院校/期刊/会议：（待定，倾向 CCF-A 系统类会议）
- 截止日期：（待定）

## 三、本项目约束

- 目标字数：（待定）
- 引用格式（GB/T / APA / IEEE / MLA）：IEEE
- 语言要求（中文 / 英文 / 中英双语）：英文为主
- 文件格式要求（Markdown / Word / LaTeX）：LaTeX

## 四、写作偏好（必须记录）

- 正文是否允许加粗：默认不允许
- 段落间空行：默认必须
- 正文是否允许列表：默认不允许
- 其他风格偏好：

## 五、研究问题与方法

- 核心研究问题（2026-08-06 重定位，详见 `plan/review/research-alignment.md`）：
  - 大目标：**分布式算力单元协作执行训练/推理任务**。
  - D1 任务拆分：把训练/推理任务拆成更小的阶段/部分（模型切分、流水线、阶段分离）。
  - D2 协作执行：多个算力单元如何协同完成一个任务（pipeline、batching、通信协同）。
  - D3 任务分配/调度：任务如何分配给不同的异构算力单元（放置、调度、负载均衡）。
  - D4 节点离开/加入：中途有节点离开/加入时，任务如何快速装载/卸载、状态如何迁移恢复。
  - D5 新单元性能评估：新 GPU 到来时如何评估其能力、决定分配多大任务。
  - D6 非数据中心通信：非数据中心场景（边缘/去中心化/地理分布式）下节点间如何通信。
  - 排除项：激励/信誉机制（如 PolyLink 区块链代币）、与分布式算力协作无关的方向。
- 研究方法：系统研究与实验、调度算法设计（启发式 / 强化学习 / 图优化）、原型实现
- 数据来源：相关文献（polylink 站点 33 篇 + CCF-A 30 篇，共 57 篇本地 PDF）
- 预期贡献：（待定；研究空白见 `plan/review/research-alignment.md` 第四节——运行中新 GPU 在线评估、异构边缘无宽限期快速迁移恢复、通信受限下任务切分）

## 六、当前阶段

- 当前阶段：文献调研（相关工作开展）
- 当前任务：
  1. 逐个了解并下载 polylink.evan.cafe 列出的论文（33 篇，15 期刊 + 18 会议）
  2. 在 CCF-A 会议（ATC/OSDI/SOSP/NSDI/HPCA/PPoPP/SC/ICSE 等）检索异构分布式算力调度与弹性资源相关论文并下载
- 下一个里程碑：建立完整 source pool 与 evidence map

## 七、Paper Support Pack 状态

- Submission index 状态：`plan/submission-index.md`
- Evidence map 状态：`refs/evidence-map.md`
- Source pool 状态：`plan/retrieval/source-pool.md`
- Screening rubric 状态：`plan/review/screening-rubric.md`
- Reviewed papers 状态：`plan/review/reviewed-papers.json`
- Relevant papers 状态：`plan/review/relevant-papers.json`
- Citation ledger 状态：`plan/bibliography/citation-ledger.md`
- Full-text papers 目录：`refs/papers/`
- Parsed paper sidecars 目录：`refs/papers/parsed/`
- Venue template profile 状态：`plan/venue-template-profile.md`
- Evidence coverage 状态：`plan/review/evidence-coverage.md`
- Section architecture 状态：`plan/section-architecture.md`
- Experiment protocol 状态：`plan/experiment-protocol.md`
- Method-experiment traceability 状态：`plan/review/method-experiment-traceability.md`
- Table schema 状态：`tables/table-schema.md`
- Data manifest 状态：`figures/data-manifest.md`

## 八、目录约定

```text
项目目录/
├── plan/
│   ├── project-overview.md
│   ├── stage-gates.md
│   ├── progress.md
│   ├── outline.md
│   ├── notes.md
│   ├── submission-index.md
│   ├── venue-template-profile.md
│   ├── retrieval/
│   │   ├── source-pool.md
│   │   └── polylink-site-papers.md   <- polylink.evan.cafe 提取的论文清单
│   ├── bibliography/
│   │   └── citation-ledger.md
│   ├── task-packets/
│   ├── section-blueprints/
│   └── review/
├── sections/
├── tables/
├── figures/
├── figure-logs/
├── submissions/
└── refs/
    ├── evidence-map.md
    └── papers/                       <- 下载的论文 PDF（year-venue-title 命名）
        └── parsed/
```

## 九、初始化确认清单

- [x] 本轮写作是否使用 `projects/<task-root>/` 独立任务根目录
- [ ] 当前激活 submission 是否已登记到 `plan/submission-index.md`
- [x] 已与用户确认研究主题
- [ ] 已与用户确认输出物与截止时间
- [ ] 已确认写作偏好（无加粗、段间空一行等）
- [x] 已确定本轮优先任务（文献调研）
