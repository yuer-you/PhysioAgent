# Workflow DPO v1 冻结 final v4 报告

## 实验设置

- 冻结测试：`evaluation/workflow_final_cases_v4.jsonl`
- SHA-256：`3016c73fb509e88d1a670185e2e4086f7e2b8ce8c2f62330ede8d8c3c4ee087f`
- Reference：Workflow LoRA v2
- Candidate：Workflow DPO v1
- 提示词：workflow prompt v2
- 最大生成长度：128 token
- 解码：确定性生成
- 主指标：严格端到端成功率
- 恢复层：不使用

测试在 DPO 训练前冻结。SFT reference 和 DPO candidate 使用相同问题、提示词、生成预算与执行器，各运行一次并保存原始输出。

## 受控对照

| 指标 | SFT LoRA v2 | DPO v1 | 变化 |
|---|---:|---:|---:|
| 有效计划率 | 98.8% | 98.8% | 0 pp |
| 计划精确率 | 78.8% | 82.5% | +3.75 pp |
| 工具执行成功率 | 92.5% | 92.5% | 0 pp |
| 参考检查通过率 | 92.5% | 92.5% | 0 pp |
| 回答有依据率 | 92.5% | 92.5% | 0 pp |
| 严格端到端成功率 | 63 / 80 | 66 / 80 | +3 条 |

逐例配对结果：DPO 修复 3 条、回退 0 条、两者共同通过 63 条、共同失败 14 条。三个不一致样例均偏向 DPO，但双侧精确检验为 `p=0.25`。该结果与“小幅正向变化”一致，样本证据不足以声称显著提升。

## 分组结果

| 期望步骤数 | SFT LoRA v2 | DPO v1 |
|---:|---:|---:|
| 1 | 10 / 10 | 10 / 10 |
| 2 | 21 / 28 | 23 / 28 |
| 3 | 32 / 42 | 33 / 42 |

| 加载策略 | SFT LoRA v2 | DPO v1 |
|---|---:|---:|
| 不加载 | 25 / 26 | 25 / 26 |
| 默认加载 | 23 / 27 | 25 / 27 |
| 显式加载 | 15 / 27 | 16 / 27 |

| 类别 | SFT LoRA v2 | DPO v1 |
|---|---:|---:|
| single_step | 10 / 10 | 10 / 10 |
| filter_then_heart_rate | 6 / 6 | 6 / 6 |
| filter_then_peaks | 6 / 6 | 6 / 6 |
| filter_then_statistics | 5 / 6 | 5 / 6 |
| load_then_statistics | 4 / 10 | 6 / 10 |
| load_filter_heart_rate | 10 / 14 | 11 / 14 |
| load_filter_peaks | 12 / 14 | 12 / 14 |
| load_filter_statistics | 10 / 14 | 10 / 14 |

## DPO 修复的三条案例

1. `wf_final_v4_load_then_statistics_001`：SFT 遗漏显式 `load_signal`；DPO 正确生成 `signal` 列加载后统计。
2. `wf_final_v4_load_then_statistics_007`：SFT 输出两个重复的顶层 `steps` 键；DPO 生成单一合法两步计划。
3. `wf_final_v4_load_filter_heart_rate_009`：SFT 遗漏默认加载；DPO 正确生成加载、滤波、心率三步链路。

## DPO 剩余失败

DPO 共有 14 条严格失败：

- 参数错误 10 条；
- 步骤数量错误 2 条；
- 工具序列错误 1 条；
- schema 无效 1 条；
- 没有生成长度截断，最长输出 56 token。

具体错误仍集中于加载参数，例如把描述词组合成不存在的 `auto_parsed_waveform`、`key_signal`、`key signal` 或 `ecg` 列，以及生成 schema 不允许的 `signal_key` 参数。另有一条不加载任务把最终统计错误替换为心率，导致上游窄带滤波与 ECG 检测器不兼容。

## 结论

DPO v1 在不产生逐例回退的情况下净修复 3 条，说明受控偏好对能够轻微改善加载完整性和显式列处理。但它没有改变整体执行/依据指标，且统计证据有限。项目应保留这一负责任的结论，而不是继续针对 final v4 调参。

final v4 已被使用，不能再作为未见测试。若继续研究，应建立新的训练来源与 final v5；对于当前求职项目，更合适的下一步是整理统一实验报告、可复现命令和最终演示，而不是反复追逐同一测试分数。

机器可读结果：

- `outputs/workflow/workflow_sft_v2_final_v4/lora/summary.json`
- `outputs/workflow/workflow_sft_v2_final_v4/lora/results.jsonl`
- `outputs/workflow/workflow_dpo_v1_final_v4/lora/summary.json`
- `outputs/workflow/workflow_dpo_v1_final_v4/lora/results.jsonl`
