# Workflow SFT v2 数据说明

这份数据用于训练 Qwen2.5-3B-Instruct 生成严格的多步工具计划，主要修复 Workflow LoRA v1 对 `load_signal` 新表达泛化不足的问题。

## 文件

- `train.jsonl`：1800 条训练样例；
- `validation.jsonl`：300 条验证样例；
- `manifest.json`：版本、随机种子、分布、列名隔离策略和文件哈希。

每条记录采用 TRL prompt-completion 格式。模型输入包含 system 和 user 消息，标签是一个严格的 `{"steps":[...]}` JSON 对象。只对 completion 计算 SFT loss。

## 数据设计

训练和验证均覆盖单步、两步和三步工作流，以及三类加载策略：

- `load_explicit`：调用 `load_signal` 并复制明确列名；
- `load_default`：调用 `load_signal`，参数为空对象；
- `no_load`：不调用 `load_signal`。

验证集的加载短语和显式列名不出现在训练集中，用于发现模板记忆。生成器不会读取开发集、final v1 或 final v2；final v2 只提供“加载语义是主要失败类型”这一诊断结论，没有任何问题文本被复制。

冻结 ECG 检测器要求上游滤波完整保留 5–15 Hz，因此数据不会生成“默认 0.5–8 Hz 滤波后检测峰值或心率”的不可执行标签。默认滤波只出现在单独滤波和滤波后统计任务中。

## 复现

从项目根目录运行：

```bash
python scripts/generate_sft_workflow_data_v2.py
```

预期 SHA-256：

```text
train.jsonl      66da58371d627997e13a1bd37b7c59c7a4f062e2d8d84171a7b1e350125e63a8
validation.jsonl aead0dbb6f1069e1ea5f899399578455b6612aaedd955409b9096c0722190a74
```

该数据仅用于学习和软件评测，不用于临床诊断。
