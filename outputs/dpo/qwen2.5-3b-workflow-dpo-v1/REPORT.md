# Workflow DPO v1 训练报告

## 配置

- 基础模型：Qwen2.5-3B-Instruct
- 初始与 reference policy：Workflow LoRA v2
- 训练/验证偏好对：1000 / 200
- Epoch：1
- 学习率：`5e-6`
- Beta：`0.1`
- DPO loss：sigmoid
- 有效 batch size：16
- 最大 prompt/completion/full：896 / 128 / 1024
- 实际最大 prompt/completion/full：640 / 60 / 694
- 可训练参数：3,686,400（0.1192%）
- 设备：单张 NVIDIA RTX A4000

训练耗时 1210.98 秒，约 20 分 11 秒。最终训练 loss 为 `0.60014`，验证 loss 为 `0.56687`。验证偏好准确率为 99.5%，reward margin 为 `0.27722`。

验证 chosen/rejected reward 分别为 `-0.16725` 和 `-0.44446`。两者都为负表示相对冻结 SFT reference 的序列概率都下降，但 rejected 下降更多；因此偏好指标不能代替自主生成评测。

## 生成回归

- 旧工作流开发集：SFT `20/20`，DPO `20/20`；
- 300 条 SFT 留出释义：SFT `275/300`，DPO `276/300`；
- 留出集逐例：修复 1 条、回退 0 条；
- final v4：SFT `63/80`，DPO `66/80`；
- final v4 逐例：修复 3 条、回退 0 条。

训练配置、环境、偏好数据哈希和指标保存在 `run_manifest.json`、`all_results.json` 与 `trainer_state.json`。最终冻结分析见 `outputs/workflow/workflow_dpo_v1_final_v4/REPORT.md`。
