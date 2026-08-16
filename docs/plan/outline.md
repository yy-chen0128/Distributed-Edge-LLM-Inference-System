# 论文大纲

> 用途：定义论文 section 结构，`progress.md` 的 section 进度需与本文件一致。

## 一、当前采用结构

- 结构类型（conference paper / journal paper / survey / 中文期刊 / 其他）：
- 目标 venue：
- 版本号：v1
- 最后更新：

## 二、Section 目录

### Abstract
- 目标字数：
- 要点：问题、方法、结果、结论

### 01 Introduction
- 研究背景
- 问题定义
- 主要挑战
- 贡献与 paper 结构

### 02 Related Work
- 路线一
- 路线二
- 路线三
- 未覆盖边界

### 03 Method / Approach
- 任务设定：边缘异构节点（带宽受限/churn/内存受限）上执行 LLM 推理
- 系统结构：机制层（KVStore/Transport/Engine 抽象 + 模型装载 + 事件模型）
- 策略层：
  - 放置策略：E2 缓存感知路由（exploit/explore + prompt-aware 负载）
  - 迁移策略：PT×N 优先级迁移（价值排序 + 时间预算部分迁移）
  - 重并行化 / 中断恢复（接口 + 默认，后续）
- 验证方法：三层（L1 逻辑 / L2 契约 / L3 集成），对接测试无需 GPU
- 对应文档：`sections/02_method.md`（draft v1）

### 04 Experimental Setup
- 数据集
- baseline
- 指标
- 实现细节

### 05 Results and Analysis
- 主结果
- 对比分析
- 失败案例或边界条件

### 06 Ablation / Discussion
- 消融
- 效率或鲁棒性
- 局限性

### 07 Conclusion
- 主要结论
- 局限与未来工作

### References
- 引用规范：

### Appendix（可选）
- 

## 三、变更记录

### [日期] [变更主题]
- 变更内容：
- 变更原因：
- 影响 section：
