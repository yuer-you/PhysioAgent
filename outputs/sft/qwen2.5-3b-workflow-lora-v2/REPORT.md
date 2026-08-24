# Workflow LoRA v2 训练报告

- 训练样例：1800
- 验证样例：300
- Epoch：2
- 有效 batch size：16
- 学习率：`2e-4`
- 最大长度：1024，实际最大 699 token
- 可训练参数：3,686,400（0.1193%）
- 最终训练 loss：`0.0036205`
- 最佳验证 loss：`0.0134034`，位于 epoch 2 / checkpoint 226
- 训练设备：单张 NVIDIA RTX A4000
- 训练时间：1953.19 秒（约 32 分 33 秒）

Adapter 已通过 PEFT 配置和权重完整性检查。旧开发集严格端到端结果为 `20/20`；300 条留出释义的自主生成结果为 `275/300`。完整生成式分析见 `outputs/workflow/workflow_lora_v2_sft_validation/REPORT.md`。

训练未读取 final v3。训练配置、环境和哈希的机器可读记录保存在 `run_manifest.json`，指标保存在 `all_results.json`，trainer 状态保存在 `trainer_state.json`。
