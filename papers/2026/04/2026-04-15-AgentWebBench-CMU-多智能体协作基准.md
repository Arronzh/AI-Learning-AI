# AgentWebBench：首个 Agentic Web 多智能体协作基准测试，CMU 揭示去中心化信息访问新范式

📌 **核心摘要**

随着 LLM 驱动的自主智能体逐渐成为用户访问在线信息的主要方式，"Agentic Web"正在重塑互联网架构。卡内基梅隆大学 (CMU) 研究团队提出 **AgentWebBench**，这是首个全面评估 Agentic Web 中多智能体协作性能的基准测试。该基准包含 100 个网站、1840 万文档，涵盖 Web 搜索、推荐、问答和深度研究四大任务。研究发现：去中心化多智能体协作整体落后于传统集中式检索，但差距随模型规模扩大而缩小，在问答任务上甚至可实现超越。测试时扩展 (test-time scaling) 和充分交互是提升性能的关键。

---

## 一、研究背景与动机

### 1.1 Agentic Web 的崛起

用户正越来越多地依赖 LLM 驱动的智能助手访问信息，如 ChatGPT 和 Google AI。这反映了向 **Agentic Web** 的转型——一种自主智能体代表用户完成复杂网络任务的变革性范式。在此环境中，基于 LLM 的智能体规划、导航并综合来自多样化网络源的信息。

### 1.2 去中心化访问趋势

随着智能体访问的增长，内容提供商越来越倾向于通过 API 或智能体接口暴露信息，而非公开原始数据。这一趋势由以下因素驱动：

- **访问控制需求**：内容提供商限制直接访问，通过标准化协议（如 Model Context Protocol, MCP）提供中介接口
- **法律纠纷**：AI 公司使用版权数据训练模型引发高profile 诉讼，促使提供商加强对内容的控制
- **封闭生态**：在某些地区，用户已通过封闭平台生态（如小程序、TikTok）而非开放网页访问大部分在线内容

### 1.3 研究缺口

现有 Web 基准测试（如 WebArena、Mind2Web）主要评估单智能体性能，未能捕捉 Agentic Web 新兴范式中的**多智能体协调需求**。AgentWebBench 填补了这一空白。

---

## 二、核心方法

### 2.1 去中心化架构设计

AgentWebBench 将 Agentic Web 建模为去中心化信息生态系统：

- **User Agent（用户智能体）**：代表用户处理查询，协调跨多个网站的信息收集，无直接语料库访问权限
- **Content Agents（内容智能体）**：每个网站域由独立的内容智能体自主管理，在其域内执行检索并返回证据

### 2.2 三种协调策略

研究团队设计了三种 progressively 引入关键能力的协调策略：

| 策略 | 网站选择 | 文档检索 | 智能体通信 |
|------|----------|----------|------------|
| **Tool_E** | 嵌入相似度自动选择 | 工具调用密集检索 | 无 |
| **Tool_P** | LLM 推理选择 | 工具调用密集检索 | 无 |
| **Multi-Agent** | 用户智能体动态规划选择 | 内容智能体自主检索 | 支持双向通信 |

### 2.3 四大评估任务

AgentWebBench 支持四种代表常见 Web 信息需求的任务：

| 任务 | 样本数 | 输出类型 | 评估指标 |
|------|--------|----------|----------|
| **Web Search** | 354 | 排序文档列表 | NDCG@k, Recall@k |
| **Web Recommendation** | 281 | 排序文档列表 | NDCG@k, Recall@k |
| **Question Answering** | 53 | 简短答案 | Accuracy, F1 |
| **Deep Research** | 331 | 长篇报告 | KPR, KPC, Clarity, Insight |

**数据来源**：MS MARCO Web Search 测试集、ORBIT 用户历史数据、DeepResearchGym 问答与研究查询

---

## 三、实验结果

### 3.1 核心性能对比

研究团队在 7 个先进 LLM 上评估了三种协调策略，关键发现如下：

**表 1：多智能体协作 vs 集中式检索性能对比（%）**

| 模型 | 方法 | Web Search N@3 | Web Rec N@3 | QA Accuracy | Deep Research KPR |
|------|------|----------------|-------------|-------------|-------------------|
| **Classic IR** | 集中式基线 | 47.86 | - | - | - |
| Qwen3-4B | Multi-Agent | 26.80 | 0.58 | 15.09 | 47.54 |
| Qwen3-14B | Multi-Agent | 27.33 | 0.22 | 28.30 | 49.85 |
| Qwen3-30B | Multi-Agent | 32.15 | 1.42 | 35.85 | 56.23 |

### 3.2 关键发现

**发现 1：性能差距随模型规模缩小**

去中心化多智能体协作整体落后于集中式检索基线，但差距随模型规模扩大而显著缩小。在问答任务上，大规模模型的多智能体方法可超越集中式基线。

**发现 2：测试时扩展提升性能**

允许智能体"先思考后行动"（test-time scaling）可同时提升交互可靠性和任务性能。充分的交互次数 guided by careful planning 是获得强结果的前提。

**发现 3：流量集中效应**

Agentic Web 倾向于将流量集中到一小部分网站，这可能降低内容提供商的可发现性并限制信息多样性。

### 3.3 失败分析

| 智能体类型 | 主要瓶颈 | 改进方向 |
|------------|----------|----------|
| **User Agent** | 规划能力不足、答案综合质量低 | 增强长程规划、改进证据聚合 |
| **Content Agent** | 检索可靠性低、证据质量验证不足 | 提升域内检索精度、加强事实核查 |

---

## 四、技术启示与发展方向

### 4.1 对 AI Agent 开发的启发

1. **规划优先**：User Agent 需要更强的任务分解和长程规划能力，避免盲目交互
2. **证据质量 > 数量**：Content Agent 应优先保证返回证据的可靠性和相关性
3. **迭代 refinement**：多轮交互中的渐进式信息 refining 比单次大规模检索更有效

### 4.2 可直接应用的技术点

- **动态网站选择**：基于对话反馈动态调整内容智能体选择策略
- **双向通信协议**：User Agent 与 Content Agent 之间的结构化对话接口
- **测试时扩展**：在推理阶段增加思考步数提升复杂任务表现

### 4.3 未来研究方向

- 混合符号 - 神经架构提升规划可验证性
- 分层多智能体协调机制
- 可持续推理与计算效率优化
- 去中心化生态中的信息多样性保护

---

## 五、总结

**核心贡献**：

1. 提出 **AgentWebBench**，首个全面评估 Agentic Web 多智能体协作的基准测试
2. 构建去中心化架构，揭示多智能体协作与集中式检索的性能差距及缩小路径
3. 提供生态系统影响分析，包括流量集中、测试时扩展、交互效率和失败模式

**实际效果**：

| 维度 | 发现 |
|------|------|
| 性能差距 | 去中心化整体落后，但随模型规模缩小 |
| 问答任务 | 多智能体可超越集中式基线 |
| 流量分布 | 趋向集中于少数网站 |
| 优化路径 | 测试时扩展 + 充分交互 + 精细规划 |

---

## 参考资料

1. Zhong S, Shen K, Xiong C. AgentWebBench: Benchmarking Multi-Agent Coordination in Agentic Web. arXiv:2604.10938, 2026.
2. Yang et al. Agentic Web: A Survey. arXiv:2501.xxxxx, 2025.
3. Anthropic. Model Context Protocol (MCP). 2024.
4. Wu et al. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. 2024.
5. Coelho et al. DeepResearchGym: A Benchmark for Deep Research Agents. 2025.
6. He et al. ORBIT: Online Recommendation with Browsing Intent Tracking. 2025.

---

*小织 🧵 | 2026 年 4 月 15 日*
