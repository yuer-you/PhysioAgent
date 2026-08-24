# Workflow LoRA v2 冻结 final v3 报告

## 实验设置

- 基础模型：Qwen2.5-3B-Instruct
- Adapter：`outputs/sft/qwen2.5-3b-workflow-lora-v2/final_adapter`
- 冻结测试：`evaluation/workflow_final_cases_v3.jsonl`
- 测试 SHA-256：`b6a62188d1f039b984f5a050c3cd451e2ac349b1a5f92ff16b942352dec30025`
- 提示词：workflow prompt v2
- 最大生成长度：128 token
- 解码：确定性生成
- 主指标：严格计划匹配、工具执行与回答依据全部通过
- 恢复层：不用于主指标，也没有运行必要

final v3 在 Workflow LoRA v2 训练前冻结。基础模型和 adapter 使用完全相同的测试、提示词与解码配置，各运行一次并保留全部原始输出。

## 受控对照结果

| 指标 | 基础 Qwen | Workflow LoRA v2 | 变化 |
|---|---:|---:|---:|
| 有效计划率 | 93.3% | 100% | +6.7 pp |
| 计划精确率 | 41.7% | 60.0% | +18.3 pp |
| 工具执行成功率 | 85.0% | 100% | +15.0 pp |
| 参考检查通过率 | 85.0% | 98.3% | +13.3 pp |
| 回答有依据率 | 85.0% | 100% | +15.0 pp |
| 严格端到端成功率 | 25 / 60（41.7%） | 36 / 60（60.0%） | +11 条 / +18.3 pp |

逐例配对结果：

- LoRA 修复基础模型失败：12 条；
- LoRA 使基础模型正确样例回退：1 条；
- 两者共同通过：24 条；
- 两者共同失败：23 条。

对 13 条配对结果不一致的样例做双侧精确检验，12 次改善、1 次回退对应 `p≈0.0034`。这支持“本次 LoRA 在该冻结测试上确有提升”，但不能外推到其他模型、信号类型或临床任务。

## 分组结果

| 期望步骤数 | 基础 Qwen | Workflow LoRA v2 |
|---:|---:|---:|
| 1 | 6 / 8 | 8 / 8 |
| 2 | 17 / 26 | 23 / 26 |
| 3 | 2 / 26 | 5 / 26 |

| 加载策略 | 基础 Qwen | Workflow LoRA v2 |
|---|---:|---:|
| 不加载 | 19 / 24 | 23 / 24 |
| 默认列加载 | 3 / 18 | 8 / 18 |
| 显式列加载 | 3 / 18 | 5 / 18 |

| 类别 | 基础 Qwen | Workflow LoRA v2 |
|---|---:|---:|
| single_step | 6 / 8 | 8 / 8 |
| filter_then_heart_rate | 5 / 6 | 6 / 6 |
| filter_then_peaks | 6 / 6 | 6 / 6 |
| filter_then_statistics | 4 / 6 | 5 / 6 |
| load_then_statistics | 2 / 8 | 6 / 8 |
| load_filter_heart_rate | 1 / 8 | 2 / 8 |
| load_filter_peaks | 0 / 8 | 1 / 8 |
| load_filter_statistics | 1 / 10 | 2 / 10 |

## LoRA 失败分析

LoRA 的 24 条失败均可严格解析：

- 23 条步骤数量错误；
- 1 条 `load_signal` 参数错误；
- 22 条直接遗漏用户要求的 `load_signal`；
- 1 条不加载任务遗漏最终 `calculate_statistics`；
- 1 条生成了 `load_signal`，但漏掉明确列名 `signal`；
- 没有未知工具、schema 错误、JSON 闭合错误或生成截断；最长输出为 52 token。

唯一的逐例回退是 `wf_final_v3_load_stats_004`：基础模型正确生成加载后统计，LoRA 只生成统计步骤。

参考检查唯一失败为 `wf_final_v3_filter_stats_006`。模型只执行滤波并遗漏最终统计，因此工具执行本身成功，但最终工具与参考要求不一致。严格端到端指标已将其计为失败。

## 结论与限制

Workflow SFT v2 显著改善了输出合法性、单步/两步规划和加载识别，净提升 11 条。但三步任务仍只有 `5/26`，主要瓶颈从“格式错误与乱填参数”转为“在较长计划中省略开头加载步骤”。

final v3 已用于最终评测，不能再作为未见测试重跑调参。若继续进入偏好优化或下一版 SFT，应从独立模板构建数据，冻结新的 final v4。测试复用已有 MIT-BIH 文件，因此结论只涉及未见语言与规划组合，不代表未见患者信号或临床泛化。

机器可读结果分别位于：

- `outputs/workflow/workflow_base_final_v3/base_qwen/summary.json`
- `outputs/workflow/workflow_base_final_v3/base_qwen/results.jsonl`
- `outputs/workflow/workflow_lora_v2_final_v3/lora/summary.json`
- `outputs/workflow/workflow_lora_v2_final_v3/lora/results.jsonl`
