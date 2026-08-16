# Section Architecture

> 用途：把整篇论文切成执行 section，并在中型或整篇任务开始前锁定文件、角色、长度和 owner。

## 一、当前状态

- 目标 venue：
- 结构版本：v1
- 当前任务类型（新稿 / 整篇重写 / 多节联动返工）：
- 最后更新：

## 二、Required section files

| Section file | Role | Min body length / chars | Owner | Planning placeholders allowed | Status |
|---|---|---|---|---|---|
| sections/01_introduction.md | problem-gap-contribution | 4500 |  | no | pending |
| sections/02_method.md | method flow | 4500 |  | no | **draft v1** |
| sections/03_experiments.md | setup-results-analysis | 4500 |  | no | pending |
| sections/04_discussion.md | meaning-boundary-limits | 2500 |  | limited | pending |

> 注：`sections/02_method.md` 已于 2026-08-16 起草 v1（框架设计 → 方法层论述）。对应实现 `project/edge_llm_scheduler/`，设计详情 `plan/review/framework-design.md`。

## 三、Notes

- 如目标 venue 不要求独立 Related Work，可把其并入 Introduction。
- 如 section 结构变化，先更新本文件，再更新 `plan/outline.md` 与正文。
- 缺失 section 或额外 section 都应视为 review item，而不是静默接受。
