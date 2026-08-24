# Workflow LoRA v1 开发集报告

## 实验设置

- 基础模型：Qwen2.5-3B-Instruct
- Adapter：`outputs/sft/qwen2.5-3b-workflow-lora-v1/final_adapter`
- 数据：`evaluation/workflow_planning_cases_v1.jsonl`
- 提示词：workflow prompt v2
- 最大生成长度：128 token
- 解析方式：严格 JSON 和 schema 校验，不启用保守恢复

## 结果

严格端到端成功率为 `20/20`（100%）。计划 JSON 有效率、计划精确率、工具执行成功率、参考检查通过率和回答有依据率均为 100%，没有失败样例。

| 类别 | 正确数 / 总数 |
|---|---:|
| single_step | 3 / 3 |
| filter_then_heart_rate | 6 / 6 |
| filter_then_peaks | 4 / 4 |
| filter_then_statistics | 4 / 4 |
| load_then_statistics | 3 / 3 |

## 结论与限制

这个结果说明多步 SFT adapter 已经学会开发集中的单步和两步计划格式，且解决了基础模型在 `load_signal` 和 JSON 闭合方面暴露的开发集错误。

但该开发集没有三步任务，而且已经参与过提示词和系统开发，所以 100% 不能作为最终泛化成绩。模型必须在预先冻结、包含大量三步计划的 `evaluation/workflow_final_cases_v2.jsonl` 上只评测一次，才能得到本阶段的主要结论。

原始机器可读结果保存在同目录的 `lora/summary.json` 和 `lora/results.jsonl`。
