# PhysioAgent 服务器复现指南

## 1. 固定环境

```text
Python 3.11.9
PyTorch 2.3.1+cu118
Transformers 4.57.1
TRL 0.24.0
PEFT 0.17.1
GPU: NVIDIA RTX A4000 16GB
```

进入克隆后的项目根目录：

```text
cd /absolute/path/to/PhysioAgent
```

设置本地基础模型路径：

```text
export PHYSIOAGENT_MODEL_PATH=/absolute/path/to/Qwen2.5-3B-Instruct
```

确保 Python 可以导入当前项目：

```bash
export PYTHONPATH="$PWD"
```

## 2. 完整测试

```bash
python -m pytest -q
```

本地冻结版本预期为 `135 passed`；Windows 上可能出现无法写 `.pytest_cache` 的无害警告。

## 3. 关键数据哈希

```text
Workflow SFT v2 train:
66da58371d627997e13a1bd37b7c59c7a4f062e2d8d84171a7b1e350125e63a8

Workflow SFT v2 validation:
aead0dbb6f1069e1ea5f899399578455b6612aaedd955409b9096c0722190a74

Workflow DPO v1 train:
b6c7cd4244943cdfeb92b5dcb7ee805f4ffab8e212754b76ba4b1f3cae59ddd2

Workflow DPO v1 validation:
b2a12a322522ccee388e7cbc82c9a7f939ec7212fc9ef90ad81fa738f84f053e

Workflow final v3:
b6a62188d1f039b984f5a050c3cd451e2ac349b1a5f92ff16b942352dec30025

Workflow final v4:
3016c73fb509e88d1a670185e2e4086f7e2b8ce8c2f62330ede8d8c3c4ee087f
```

校验：

```bash
sha256sum \
  data/sft_workflow_v2/train.jsonl \
  data/sft_workflow_v2/validation.jsonl \
  data/dpo_workflow_v1/train.jsonl \
  data/dpo_workflow_v1/validation.jsonl \
  evaluation/workflow_final_cases_v3.jsonl \
  evaluation/workflow_final_cases_v4.jsonl
```

## 4. Workflow SFT v2

```bash
python -m physioagent.train_sft \
  --model-path "$PHYSIOAGENT_MODEL_PATH" \
  --train-file data/sft_workflow_v2/train.jsonl \
  --validation-file data/sft_workflow_v2/validation.jsonl \
  --output-dir outputs/sft/qwen2.5-3b-workflow-lora-v2 \
  --epochs 2 \
  --learning-rate 2e-4 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --max-length 1024
```

## 5. Workflow DPO v1

```bash
python -m physioagent.train_dpo \
  --model-path "$PHYSIOAGENT_MODEL_PATH" \
  --sft-adapter-path outputs/sft/qwen2.5-3b-workflow-lora-v2/final_adapter \
  --train-file data/dpo_workflow_v1/train.jsonl \
  --validation-file data/dpo_workflow_v1/validation.jsonl \
  --output-dir outputs/dpo/qwen2.5-3b-workflow-dpo-v1 \
  --epochs 1 \
  --learning-rate 5e-6 \
  --beta 0.1 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --max-prompt-length 896 \
  --max-completion-length 128 \
  --max-length 1024
```

不要对已有非空输出目录使用覆盖参数。训练前先执行相同命令并附加 `--dry-run --inspect-token-lengths`。

## 6. 最终演示

```bash
python -m physioagent.final_demo
```

自定义问题：

```bash
python -m physioagent.final_demo \
  --question "请先读取 signal 列，再保留 1 到 30 Hz，最后检测 R 峰" \
  --signal-file data/real/mitdb/207_30s/signal.csv \
  --output outputs/demo/custom_record207.json
```

输出始终写入项目路径，包含模型原始计划、逐步输入来源、参数、结果摘要、停止原因和最终回答。

## 7. 关键结果位置

```text
EXPERIMENT_REPORT.md
outputs/real_signal/REPORT.md
outputs/final_evaluation_v1/REPORT.md
outputs/workflow/workflow_lora_v2_final_v3/REPORT.md
outputs/workflow/workflow_dpo_v1_final_v4/REPORT.md
outputs/sft/qwen2.5-3b-workflow-lora-v2/run_manifest.json
outputs/dpo/qwen2.5-3b-workflow-dpo-v1/run_manifest.json
```

所有模型与数据仅用于学习和软件评测，不用于临床决策。
