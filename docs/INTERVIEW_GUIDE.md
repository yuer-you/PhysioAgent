# PhysioAgent 面试讲解指南

## 90 秒项目介绍

> 我做了一个面向生理时序分析的工具调用 Agent。模型不直接预测医学结论，而是把自然语言请求转成严格的多步 JSON 计划，再由确定性 Python 工具处理真实 ECG。我从规则 Agent 开始，跑通 Qwen2.5-3B 基线，然后用 LoRA 做单工具和多步 SFT，再针对遗漏加载、默认列乱填等错误构造 chosen/rejected 数据做 DPO。整个项目把模型规划、工具执行、ECG 算法和 grounded answer 分开评测，每轮训练前冻结新的测试集。在同一 final v3 上，Workflow SFT 把基础模型从 41.7% 提高到 60%；在同一 final v4 上，DPO 从 SFT 的 78.8% 提高到 82.5%，但只有 3 条净提升，统计上不显著，所以我没有夸大结论。

## 最值得讲的四个设计

### 1. 模型只规划，工具负责数值

这样能判断错误来自模型、工作流状态还是信号算法。最终回答中的 BPM 和峰值数量必须来自工具返回值，而不是模型自由生成。

### 2. 冻结测试与数据治理

每个 final 文件在模型运行前记录 SHA-256。最终失败不会直接复制进训练，只提取错误类型，用独立模板生成下一阶段数据，并建立新的 final。

### 3. SFT 验证不是只看 eval loss

teacher-forcing 的 token accuracy 很高不代表模型能自主生成完整计划。因此项目额外运行生成式验证，统计 JSON 有效率、计划精确率、步骤数、加载策略和逐例回退。

### 4. DPO reference 是 SFT 策略

DPO 优化目标是相对 reference 提高 chosen 对 rejected 的优势。如果 reference 错用基础 Qwen，就不是“从 SFT 模型继续偏好优化”。实现中把同一个 SFT adapter 加载为可训练 `default` 和冻结 `reference`，共享基础模型权重。

## DPO 原理简述

对 prompt `x`、chosen `y+`、rejected `y-`，DPO 比较 policy 相对 reference 对两种回答的对数概率提升：

```text
L_DPO = -log sigmoid(
  beta * [
    log πθ(y+|x) - log πref(y+|x)
    - log πθ(y-|x) + log πref(y-|x)
  ]
)
```

`beta` 越大，通常约束模型不要偏离 reference 太多。本项目使用 `beta=0.1`。训练验证 reward accuracy 为 99.5%，但 chosen reward 也为负，所以仍通过自主生成评测确认是否真的提升。

## 常见面试问题

### 为什么不用 LangChain？

项目重点是后训练和可测的 Agent 算法。核心 loop 直接实现，模型计划、schema 校验、状态传递和工具执行都能逐层测试，避免框架隐藏关键逻辑。

### 为什么只训练 q_proj/v_proj？

3B 模型在单张 16GB A4000 上需要控制显存和训练时间。`q_proj/v_proj` LoRA 只有约 3.69M 可训练参数，足以完成实验闭环。代价是容量有限，三步规划仍有遗漏加载问题。

### 为什么 DPO 提升不大？

SFT reference 在 final v4 已有 78.8%，偏好数据是合成且目标较窄；DPO 主要改变相对排序，不保证 greedy generation 大幅变化。最终只有 3 条修复、0 条回退，方向正确但 `p≈0.25`，所以不能声称显著提升。

### 真实 ECG 结果为什么不能说泛化？

只有四条 30 秒 MIT-BIH 片段，且只用第 0 通道。记录 207 的 29/29 匹配是冻结测试上的正确结果，但样本规模不足以代表不同患者、设备、导联和噪声条件。

### 遇到过什么工程问题？

- Hugging Face 主下载、ModelScope 备用下载，以及离线服务器的模型路径挂载；
- Docker 中 wfdb/pandas/pyarrow 兼容；
- 长 system prompt 导致 DPO 默认 prompt 截断风险；
- 默认 0.5–8 Hz 滤波与 ECG 检测器要求的 5–15 Hz 不兼容；
- PEFT DPO 必须正确冻结 SFT reference adapter；
- Windows/Linux 换行导致数据哈希变化。

## 简历项目描述示例

> **PhysioAgent｜大模型后训练与工具调用 Agent**：基于 Qwen2.5-3B、LoRA、TRL/PEFT 构建生理时序多步工具调用 Agent，实现严格 JSON 规划、状态传递、真实 MIT-BIH ECG 工具执行与 grounded answer；设计无泄漏的 SFT/DPO 数据生成、冻结测试和逐层评测体系。单张 RTX A4000 完成 3.69M 参数 LoRA 训练，Workflow SFT 在配对 final 上将基础模型端到端成功率从 41.7% 提升至 60.0%；DPO 在另一冻结配对测试中从 78.8% 提升至 82.5%，并如实报告统计不显著与剩余失败模式。

## 不要这样表述

- “ECG 检测准确率 100%，可用于临床”；
- “DPO 显著提升 3.75%”；
- “模型支持任意生理信号”；
- “在 8 卡上完成大规模 RLHF”。

项目真正的亮点是完整、诚实、可复现的后训练与 Agent 评测闭环。
