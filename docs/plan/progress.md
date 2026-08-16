# 进度追踪

> 用途：记录每轮任务执行前后的状态变化。

## 一、当前状态

- 最后更新：2026-08-06
- 当前阶段：文献调研（相关工作收集与整理）✅ 完成本轮
- 当前任务：下一步进入证据整理（evidence map / 综述初稿）
- 下一步行动：构建 evidence-claim map；对 3 篇 OSDI'26 等待 USENIX 发布后获取
- 阶段门禁文件：`plan/stage-gates.md`

## 二、本轮任务卡（每轮都要更新）

- 任务名称：文献调研 —— 异构分布式算力上 LLM 训练/推理的任务调度与弹性资源管理
- 任务类型（写作/润色/翻译/画图/环境配置/文献整理）：文献整理
- 输入文件：
  - `http://polylink.evan.cafe/`（React SPA，bundle.js 提取 33 篇论文）
  - CCF-A 会议相关论文（WebSearch 检索 + 多源下载）
- 输出文件：
  - `plan/retrieval/polylink-site-papers.md`（polylink 论文清单，33 篇）
  - `plan/retrieval/ccfa-candidates.md`（CCF-A 候选清单，30 篇，已核实会议）
  - `plan/retrieval/source-pool.md`（来源矩阵，63 行）
  - `plan/retrieval/download-status.json` + `download-status-ccfa.json`
  - `refs/papers/`（28 篇已下载 PDF，全部内容验证通过）
  - `scripts/`（download_papers.py / download_ccfa.py / redownload_verify.py / build_source_pool.py）
- Stage-gate updates：无
- Required skills：`shared-research-retrieval`、`paper-literature-review`
- Evidence/data inputs：arXiv（API + 搜索 UI + 直连 PDF）、Semantic Scholar、CrossRef、WebSearch、USENIX/CMU 官方 PDF
- Required artifacts：source pool、download-status、论文 PDF
- Review gate 1（spec compliance）：文件名符合框架约定（year-venue-title），状态记录完整
- Review gate 2（quality）：28 篇 PDF 全部通过首页标题内容验证（PyMuPDF）
- Verification run：`scripts/redownload_verify.py`（内容验证）+ `build_source_pool.py`
- 验收标准：✅ 相关论文 PDF 本地可用、内容验证通过、错误下载全部修正、source-pool 完整

### Capability-use audit（能力使用审计）

- Required skills：`shared-research-retrieval`、`paper-literature-review`
- Skills actually used：`shared-research-retrieval` 的存储约定、命名规则、download-status 记录方式
- Inputs consumed：polylink bundle.js、arXiv API/搜索 UI/直连 PDF、Semantic Scholar、CrossRef、WebSearch、USENIX/CMU PDL 官网
- Inputs not used and why：`scholar_search.py`（https 访问 arXiv/CrossRef 在本网络超时）；DBLP API（连接超时/429）
- Artifacts produced：见上
- Verification run：PyMuPDF 首页标题验证，28/28 通过
- Remaining risk：3 篇 OSDI'26 无 arXiv（OpenTela/Alibaba/EcoServe）需机构订阅手动获取；polylink 网站 32 篇付费论文记录了 DOI/登录页，仅 1 篇可 OA 获取

## 三、执行记录

### 2026-08-06 文献调研（文献收集）

- 执行动作：
  1. 解析 polylink.evan.cafe（React SPA），从 bundle.js 提取站点描述与 33 篇论文清单
  2. 按 research-assistant 框架 init_plan.ps1 建立 `D:\Newproject\分布式算力\research\` 项目结构
  3. WebSearch 检索 CCF-A 会议论文（OSDI/SOSP/NSDI/ASPLOS/ISCA/ATC/SOSP/FAST/NeurIPS 等）
  4. 编写 4 个脚本下载与整理论文，多源下载（arXiv / USENIX 官网 / CMU PDL / friendli.ai）
  5. 用 PyMuPDF 对全部下载 PDF 做首页标题内容验证，删除并重下错误内容
- 完成结果：
  - polylink：33 篇清单，1 篇下载（EdgeShard，OA），32 篇记录 DOI/登录页
  - CCF-A：30 篇清单（会议归属已核实），27 篇下载且内容验证通过，3 篇 manual-needed（OSDI'26）
  - source-pool 矩阵 63 行，进度与状态记录完整
- 产物路径：`plan/retrieval/*`、`refs/papers/*.pdf`（28 篇）、`scripts/*`
- 遇到问题：
  1. 学术 API 全面限流（Semantic Scholar/CrossRef/DBLP/arXiv API 429 或超时）
  2. **凭记忆填写的 arXiv ID 大量错误**，导致 13 篇 PDF 下载到无关论文（首次内容验证发现）
  3. 会议归属需多方核实（SGLang=NeurIPS 非 SOSP；Gavel=OSDI 非 SOSP；AntMan=OSDI 非 SIGCOMM；Sia=SOSP 非 EuroSys；FastServe=NSDI'26；Metis 标题实为 "Fast Automatic Distributed Training"）
- 解决方案：
  1. 改用 arXiv http API / arxiv.org 搜索 UI / 直连 arxiv.org/pdf/<id> / USENIX 官网开放 PDF
  2. **放弃记忆 ID**，改用 arXiv 搜索解析 + PyMuPDF 首页内容双重验证
  3. 用 WebSearch + USENIX 官方页核实会议归属，逐一更正
- 下一步：PDF 解析生成 sidecar → 主题聚类 → evidence-claim map → 综述初稿

### 2026-08-06 付费论文批量下载（浏览器自动化）

- 执行动作：
  1. 安装 Playwright，用 CDP 连接用户已登录的调试 Chrome（独立 profile）
  2. 解析 26 个缺失 DOI（CrossRef 带退避），18 个成功
  3. 编写 download_via_chrome.py：IEEE 走 stampPDF 端点、ACM 走页面点击下载（过 Cloudflare）
  4. 批量下载付费论文，逐篇内容验证
- 完成结果：
  - polylink 站点：30/33 下载；3 篇需手动（Graph-RAG=VLDB workshop、2 篇 AAAI 开放获取）
  - CCF-A：27/30 下载；3 篇 OSDI'26 等 USENIX 发布
  - 共 57 篇 PDF 在 refs/papers/，全部内容验证通过 + 生成 sidecar
- 产物路径：`scripts/download_via_chrome.py`、`scripts/resolve_dois.py`、`scripts/launch_chrome_debug.bat`
- 遇到问题：Chrome"本地调试保护"拦截默认 profile → 改用独立 profile；ACM 被 Cloudflare 拦截 context.request → 改用页面点击 + expect_download；IEEE PDF 链接不在初始 HTML → 用 stampPDF 端点
- 解决方案：见上；全部解决
- 下一步：构建 evidence-claim map

## 四、里程碑

| 里程碑 | 计划日期 | 实际日期 | 状态 |
|---|---|---|---|
| 选题确定 | — | 2026-08-06 | ✅ |
| 文献收集（本阶段） | 2026-08-06 | 2026-08-06 | ✅ |
| 文献综述完成 | — | — | ⏳ |
| 方法设计完成 | — | — | ⏳ |
| 初稿完成 | — | — | ⏳ |
| 润色完成 | — | — | ⏳ |
| 定稿提交 | — | — | ⏳ |

状态说明：`⏳待开始` `🔄进行中` `✅已完成` `🔧需返工`

## 五、Section 完成度（按 outline 实际填写）

| Section | 状态 | 字数 | 备注 |
|---|---|---|---|
| Abstract | ⏳待开始 | 0 | |
| Introduction | ⏳待开始 | 0 | |
| Related Work | ⏳待开始 | 0 | 需先完成 evidence map |
| 方法 | ⏳待开始 | 0 | |
| Experimental Setup | ⏳待开始 | 0 | |
| Results and Analysis | ⏳待开始 | 0 | |
| Ablation / Discussion | ⏳待开始 | 0 | |
| Conclusion | ⏳待开始 | 0 | |
| References | ⏳待开始 | 0篇 | |

## 六、待办事项

### 高优先级
- [x] 解析已下载 PDF（`scripts/pdf_parser.py`）生成 sidecar（57 篇 → `refs/papers/parsed/`）
- [x] 细读 9 篇部署调度核心论文，输出详细笔记（`plan/review/deep-notes/`）
- [x] 按主题聚类 + 综合文献笔记（`plan/review/literature-notes.md`）
- [x] 通过浏览器自动化（Playwright + CDP + 学校登录）批量下载付费论文
- [ ] 构建 evidence-claim map，为 Introduction/Related Work 写作做准备

### 中优先级
- [ ] 手动获取 3 篇 OSDI'26（OpenTela / Alibaba Hyperscale / EcoServe）
- [ ] 手动获取 polylink 站点中与研究主题相关的付费论文（PASTA / Janus / PolyEdge / Decentralized Task Offloading 等，DOI 已记录在 download-status.json）

### 低优先级
- [ ] 补查补充背景（2026 预印本：LUMEN / CheckFree 等，非 CCF-A 但主题相关）
