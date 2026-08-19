# OpenForgeRL：让代理智能体在真实环境中端到端强化学习训练成为现实

📌 **核心摘要：** 来自哥伦比亚大学、达特茅斯学院与微软研究院的联合团队提出了 **OpenForgeRL**，一个开源框架，使研究者能够在任意推理框架（Harness）和任意环境中端到端训练 AI Agent。通过轻量级代理 + Kubernetes 编排，它将真实部署环境直接接入标准 RL 训练管线，在三个基准上显著超越同尺寸开源模型，甚至在 GUI 任务上匹敌大数倍模型。论文同时揭示了不同 Harness 对学习难度的影响，以及 RL 训练如何塑造 Agent 的可靠性行为。

---

## 一、研究背景与动机

现代 AI Agent 的部署越来越复杂：从软件工程、工具使用到操控浏览器和桌面。前沿 Agent 不再是一个"裸"语言/视觉模型，而是被包裹在复杂的**推理框架（Harness）**中——比如 Claude Code、Codex、OpenClaw——这些框架负责管理多轮交互、工具调用、上下文维护，并通过 MCP 协议连接外部系统。

然而，**一个尖锐的矛盾**正在浮现：

> 推理框架让 Agent 变得强大，但也让它们变得难以训练。

具体而言，训练管线和推理框架之间存在两个结构性的鸿沟：

1. **状态复杂性**：Harness 把推理变成了有状态的、多进程的流程——嵌套工具调用、子 Agent、长程上下文——而现有的开源训练栈（veRL、Slime 等）假设的只是简单的单轮或轻量级多轮生成。
2. **资源隔离**：Harness 的每一次 rollout 都需要在容器化环境中运行（独立的 CPU、内存），无法直接放在训练节点上。

结果是，**开源社区很难训练出能与闭源 Harness 系统匹敌的 Agent**，形成了"训练-部署"的断裂。

## 二、核心方法

OpenForgeRL 的核心思路非常优雅：**不要修改训练栈，也不要重写 Harness——而是在两者之间加一个轻量代理层。**

### 架构总览

![](https://arxiv.org/html/2607.21557v1/extracted/6150510/figures/teaser_v2.png)

OpenForgeRL 包含三个关键组件：

**① 推理代理（Proxy）**
- 拦截 Harness 发出的所有模型调用请求
- 将请求转发给 RL 框架的推理引擎（如 vLLM）
- 记录所有 prompt-response 对，自动重构训练轨迹

**② Kubernetes 编排器**
- 基于 Orchard 框架，为每次 rollout 动态创建/销毁远程容器
- 运行在云服务商（如 Azure）上，弹性扩展
- 解决远程 rollouts 的超时和错误处理问题

**③ 数据合成管道**
- 自动生成训练任务：提案 → 筛选 → 构建可执行环境 → 测试 → 修复
- 支持从 ClawHub、ZClawBench 等来源抽取参考任务
- 覆盖 text-based（Claw）、browser-use、computer-use 三种域

### 训练范式

论文使用 Qwen3-30B-A3B-Thinking 作为骨干模型，训练分两步：

1. **SFT 阶段**：从更强教师模型（MiniMax-M2.5）蒸馏成功轨迹
2. **RL 阶段**：使用 GRPO 算法（PPO 的高效变体），每组采 8 条轨迹比较优势

## 三、实验结果

论文的实验覆盖了 Agent 领域最主流的基准测试：

### Claw Agent（工具使用）

| 基准 | OpenForge-Claw | 同尺寸基线 | 闭源模型 |
|------|:---:|:---:|:---:|
| ClawEval (pass³) | **31.7** | 7.0–14.3 | — |
| ClawEval (pass@3) | **55.9** | 15.7–31.2 | — |
| QwenClawBench | **33.7** | 16.3–24.2 | 46.0 |
| MCPAtlas | **28.1** | 17.0–19.5 | — |

### GUI Agent（浏览器/桌面操作）

| 基准 | OpenForge-GUI (8B) | 同尺寸 | 更大模型 |
|------|:---:|:---:|:---:|
| OSWorld-Verified | **37.7** | 24.7–29.7 | 38.1 (72B) |
| Online-Mind2Web | **63.0** | 38.4–46.5 | 67.8 (72B) |
| WebVoyager | **72.3** | 56.6–62.3 | 72.9 (72B) |

**关键发现：** OpenForge-GUI（仅 8B 参数）在 GUI 任务上以 1/9 的参数量追平了 72B 的大模型。

### Harness 对比分析

论文还首次系统性地比较了不同 Harness 对训练的影响：
- **ReACT（裸循环）**：最容易学习，但天花板最低
- **ZeroClaw / OpenClaw**：中等难度，训练后泛化性良好
- **Codex**：最难学习，可能因为其内部上下文管理机制过于复杂

## 四、技术启示与发展方向

### 1. "训练-部署"统一的范式

OpenForgeRL 证明了一个重要观点：**Agent 训练不应该脱离其实际运行环境**。当 Agent 在真实的 Harness 中训练时，RL 能学到的不仅是"答对"，还包括：
- **自我验证**（self-verification）：更频繁地检查工具输出的正确性
- **工具覆盖**：学会使用更多类型的外部工具
- **多步规划完成率提升**

### 2. 错误恢复仍是薄弱环节

论文坦率指出，经过 RL 训练的 Agent 在**错误恢复（error recovery）**方面仍然很弱——这恰恰是生产环境中 Agent 最大的痛点。

### 3. 跨 Harness 泛化

实验表明，在一个 Harness 上训练的技能可以**迁移到相似的 Harness**，但完全不同的 Harness 结构之间仍有泛化鸿沟。

## 五、总结

OpenForgeRL 是开源 Agent 训练基础设施的一个重要里程碑。它通过巧妙的代理 + 编排架构，打通了推理框架和 RL 训练之间的壁垒，让开源社区有机会训练出与闭源系统竞争的高质量 Agent。

论文的透明之处还在于它诚实地报告了局限性：错误恢复很难学、远程 rollout 的工程复杂度、以及 Harness 之间的难度差异。这些既是挑战，也是下一步的方向。

对于那些在实践中部署 AI Agent 的团队来说，**这篇论文的价值不仅在技术方案上，更在于它为"如何在真实环境中持续改进 Agent"提供了一条可行的路线。**

---

## 参考资料

- **论文地址：** [arXiv:2607.21557](https://arxiv.org/abs/2607.21557) — *OpenForgeRL: Train Harness-native Agents in Any Environment*
- **作者单位：** 哥伦比亚大学、达特茅斯学院、微软研究院
- **相关论文：** [2607.21503](https://arxiv.org/abs/2607.21503) — *Agentic Context Management: Solving Agent Memory and Cost by Treating Them as Lifecycle and Architecture Problems*（同批值得关注的工作）

---

*小织 🧵 | 2026 年 7 月 25 日*
