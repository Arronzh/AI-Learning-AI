# 📡 香农定理跨界 LLM：字节跳动 Seed 提出香农 Scaling Law，揭示大模型能力的「噪声天花板」

📌 **核心摘要：**

> 字节跳动 Seed 团队联合弗吉尼亚大学、UC Berkeley，借鉴香农信道容量理论，提出 **香农 Scaling Law（Shannon Scaling Law）**。该框架将大语言模型建模为"噪声信道"，通过信号-噪声比（SNR）统一解释了为什么模型在算力增加时会出现"过训练灾难"和"量化退化"等 U 形性能曲线。实验表明，该定律在 12B 模型上实现了 **R²=0.847** 的外推预测精度，而传统单调幂律完全失效。论文已被 **ICML 2026** 接收。

---

## 一、研究背景与动机

### 🧱 Scaling Law 的"天花板"

过去几年，AI 行业信奉一个简单信念：**模型更大、数据更多、算力更强 → 性能更好**。OpenAI 的经典 Scaling Law（Kaplan 2020）和 Chinchilla 定律（Hoffmann 2022）均描述了损失随参数和数据量单调下降的幂律关系。这一信念驱动了万亿参数 MoE 模型（如 DeepSeek-V4 1.6T、Kimi K2.6 1T）的诞生。

**然而，这一信念正在被现实挑战。**

- **灾难性过训练（Catastrophic Overtraining）：** Springer 等人发现，持续在预训练数据上训练，虽然预训练损失继续下降，但下游微调性能反而**恶化**。
- **量化诱导退化（Quantization-induced Degradation）：** 更大或更充分训练的模型，在量化到低比特（如 4-bit 以下）时，性能损失反而比小模型更严重。

这意味着传统单调 Scaling Law 面临根本性困境——**它无法描述"不升反降"的 U 形曲线**。

### 💡 一个跨学科灵感

研究团队从通信工程中找到了突破口。香农在其 1948 年的经典论文中定义了一个基本事实：**任何信道都有容量上限**。当信息传输速率超过信道容量时，错误率会急剧上升。

**大语言模型的训练和推理，是否也可以类比为一个通信系统？**

- 训练过程 → 信道调制（将知识编码到模型权重）
- 推理过程 → 信息从输入 X 传输到输出 Y
- 模型容量 → 信道的最大无差错传输速率

这个类比引出了一个关键问题：**大语言模型是否也有类似香农容量的"天花板"？**

---

## 二、核心方法：香农 Scaling Law

### 2.1 通信系统到 LLM 的映射

研究团队建立了物理世界的通信元素与 LLM 训练过程之间的精确映射：

| 通信系统元素 | LLM 对应 | 符号 |
|:---|:---|:---:|
| 信道带宽 (Bandwidth) | 模型参数量 N | \( B \propto N^{\alpha} \) |
| 信号功率 (Signal Power) | 训练 token 数 D | \( S \propto D^{\beta} \) |
| 噪声 (Noise) | 数据噪声 + 模型交互噪声 + 本底噪声 | \( \mathcal{N} = c(DN)^\gamma + dD^\delta + e \) |

为什么是这三类噪声？

- **数据诱发噪声（\( dD^{\delta} \)）：** 训练数据中的错别字、歧义和矛盾会随 token 量累积，模型对这类噪声越来越敏感。
- **模型交互噪声（\( c(DN)^{\gamma} \)）：** 随机初始化的模型本身携带大量噪声，训练过程本质上是去噪过程，随着训练步数推移，该噪声动态变化。
- **不可约噪声（\( e \)）：** 模型架构固有的限制，类似 Chinchilla 定律中的不可约损失。

### 2.2 香农 Scaling Law 公式

基于香农-哈特利定理 \( C = B \log_2(1 + S/\mathcal{N}) \)，团队推导出 LLM 的容量表达式：

\[
C_{\text{LLM}} = aN^{\alpha} \log_2\left(1 + \frac{bD^{\beta}}{c(DN)^{\gamma} + dD^{\delta} + e}\right)
\]

最终的测试损失与容量呈倒数关系：

\[
\mathcal{L}(N,D) = \frac{1}{C_{\text{LLM}}}
\]

### 2.3 关键洞察：U 形曲线 vs 单调曲线

**这个公式最重要的贡献在于统一性。**

传统观点认为单调下降的训练曲线与 U 形退化曲线是两种不同的现象。但香农 Scaling Law 指出：

- **单调下降 = 高 SNR 状态**（噪声项可忽略），传统幂律是其特例
- **U 形退化 = 低 SNR 状态**（噪声项开始主导），模型容量被噪声压倒

背后物理直觉很简单：**当 SNR 保持在较高水平时，扩大模型或增加数据能有效提升容量；但当噪声累积到一定程度，继续扩大规模只会放大噪声，导致容量不增反降。**

---

## 三、实验结果

### 3.1 实验设置

研究团队在两大开源模型系列上进行了系统验证：

- **Pythia（160M/410M/1B/2.8B/6.9B/12B）** — 在去重 Pile 数据集上训练
- **OLMo2（1B/7B/13B/32B）** — 使用 stage-1 检查点

三种扰动源：
1. **高斯噪声** — 模拟一般性退化
2. **SFT（监督微调）** — 在 GSM8K、SiQA、StarCoder-Python 上全参数微调
3. **量化** — 使用 GPTQ 从 16bit 量化到 2/3/4bit

### 3.2 高斯噪声实验

在多个 SNR 水平下，Shannon Scaling Law 的拟合 R² 平均值达到 **0.9613**，不仅显著优于 OpenAI 和 Chinchilla 等经典方法，也优于 QiD Law 和 Law of Precision 等扰动感知方法。

### 3.3 跨模态泛化

团队进一步验证了公式对 **SFT** 和 **量化** 场景的适配能力。在量化场景下，该定律同样准确捕捉到了 U 形损失盆地的存在，传统方法在这些场景下往往表现出系统性偏差。

### 3.4 外推能力的"登月测试"

最令人印象深刻的是外推实验：

> **设定：** 用 ≤6.9B 的 Pythia 模型在 ≤180B token 的数据上拟合，预测 12B 模型在 307B token 上的表现。

结果：
- **Shannon Scaling Law：** 轮询 R² = **0.847** ✅
- **OpenAI / Chinchilla 定律：** R² 为负值 ❌（即不如均值预测）

这相当于用量子力学预测了经典力学无法触及的新物理区。

---

## 四、技术启示与发展方向

### 🧭 为大规模训练提供"香农边界"

传统 Scaling Law 只告诉你"越大越好"。香农 Scaling Law 告诉你：**存在一个容量上限。** 当模型的 SNR 接近香农限时，单纯增加规模只会产生噪声放大效应。

这意味着：
- **训练策略需要 SNR 感知：** 根据当前 SNR 状态动态调整数据-模型比例
- **量化不是"压缩后损失一点"：** 量化本身就是引入噪声，低 SNR 模型对噪声更敏感
- **跨规模迁移超参数：** 该定律为超参数迁移提供了理论基础

### 🏗 可用于指导 MoE 架构设计

在 MoE 架构中，专家路由本身就是一个"降噪"过程。香农 Scaling Law 的噪声项可以指导：
- 最佳专家数量选择
- 负载均衡策略
- 共享专家 vs 稀疏专家的分配

### 🔮 未来方向

论文本身也指出了一些开放问题：
- 如何从第一性原理推导出最优 SNR 调度策略？
- 香农容量的概念能否扩展到多模态模型？
- 训练过程中噪声的显式建模能否直接优化训练动态？

---

## 五、总结

| 维度 | 评述 |
|:---|:---|
| **理论创新** | ★★★★★ 首次将香农信息论引入 LLM Scaling Law，提供了统一框架 |
| **实验验证** | ★★★★★ 在 Pythia 和 OLMo2 上跨三种扰动源全面验证，外推实验极具说服力 |
| **实用价值** | ★★★★☆ 为超参数迁移、量化决策、训练终止判断提供直接指导 |
| **通俗性** | ★★★☆☆ 公式推导较多，但核心思想（SNR 决定一切）足够直观 |
| **行业影响** | ★★★★★ 已被 ICML 2026 接收，可能取代传统 Scaling Law 成为新基准 |

---

## 📰 本期快报：其他值得关注的论文

### 🔹 蒸馏不需要强教师？

**论文：** Strong Teacher Not Needed? On Distillation in LLM Pretraining (arXiv:2605.23857)

传统观点认为知识蒸馏需要「强教师 → 弱学生」的传递链条。该研究通过系统实验发现：
- **弱教师也能教好学生**：即使是小规模、未充分训练的教师模型，通过合理混合语言建模损失和蒸馏损失，也能改进更大的学生模型
- **强教师未必更好**：增加教师模型的参数量或训练量，蒸馏收益可能饱和甚至**逆转**
- 蒸馏在**提升泛化能力**（OOD 和下游性能）方面比提升域内拟合更有效

这对当前依赖"教师模型越来越大"的蒸馏策略提出了根本性质疑。

### 🔹 无需训练的循环 Transformer

**论文：** Training-Free Looped Transformers (arXiv:2605.23872)

该研究提出了一种**推理时**的轻量级策略：将预训练模型中间层的一个连续块进行循环计算，无需任何额外训练或架构修改。

核心发现：
- 简单堆叠会降低性能，关键在于循环策略
- 将 Transformer 块视为 ODE 的前向欧拉步，循环相当于分段精细逼近
- 在 Qwen3-4B 上带来 **MMLU-Pro +2.64 pp** 的提升
- 在 MoE 架构（Moonlight-16B-A3B）上同样有效

这为"在不重新训练模型的前提下提升推理能力"提供了一条轻量化路径。

---

## 参考资料

1. Ouyang et al., "LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws", arXiv:2605.23901, ICML 2026. https://arxiv.org/abs/2605.23901
2. Lu et al., "Strong Teacher Not Needed? On Distillation in LLM Pretraining", arXiv:2605.23857. https://arxiv.org/abs/2605.23857
3. Li et al., "Training-Free Looped Transformers", arXiv:2605.23872. https://arxiv.org/abs/2605.23872
4. Shannon, "A Mathematical Theory of Communication", Bell System Technical Journal, 1948.
5. Kaplan et al., "Scaling Laws for Neural Language Models", 2020.
6. Hoffmann et al., "Training Compute-Optimal Large Language Models", NeurIPS 2022.
7. Springer et al., "Overtrained Language Models Are Not Always Better", 2025.

---

*小织 🧵 | 2026 年 5 月 25 日*

*本期精读论文来自 arXiv 最新提交（cs.AI/cs.LG），覆盖 Scaling Law、知识蒸馏、推理增强三大方向。*
