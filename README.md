# PhysioAgent

[English](README_EN.md) | 简体中文

基于 Qwen2.5-3B、LoRA SFT 和 DPO 的生理时序多步工具调用 Agent。

PhysioAgent 将中文或英文问题转换为严格的多步 JSON 计划，再由确定性 Python 工具处理真实 ECG。模型只负责规划，不直接生成医学数值；心率、R 峰和统计量均来自工具结果。

> 仅用于学习与软件评测，不用于诊断、治疗或任何临床决策。

## 项目亮点

- 五个 CPU 工具：加载、滤波、统计、R 峰检测、平均心率；
- 自研线性 Agent loop：严格 schema、跨步骤状态传递、停止条件与完整轨迹；
- 后训练闭环：Qwen 基线 → LoRA SFT → Workflow SFT → DPO；
- DPO reference 是冻结的 SFT adapter，而不是基础模型；
- 每轮训练前冻结新测试集并记录 SHA-256；
- 模型规划、工具执行、ECG 算法和 grounded answer 分层评测；
- 单张 16GB RTX A4000 完成全部 Workflow SFT/DPO 训练；
- 135 个自动化测试覆盖工具、数据、训练 dry-run、评测和最终演示。

## 系统结构

```text
用户问题 + signal_profile
        │
        ▼
Qwen2.5-3B / LoRA SFT / DPO Planner
        │
        ▼
严格 {"steps":[...]} JSON 与参数校验
        │
        ▼
WorkflowExecutor
        │
        ├── load_signal
        ├── filter_signal
        ├── calculate_statistics
        ├── detect_peaks
        └── calculate_heart_rate
        │
        ▼
grounded answer + 原始计划 + 执行轨迹 + 停止原因
```

当前 Agent 是可测试的单轮线性规划器，不包含动态重规划、长期记忆或多智能体协作。

## 主要结果

### Workflow SFT：同一 final v3

| 配置 | 严格端到端成功率 | 有效计划率 |
|---|---:|---:|
| 基础 Qwen | 25/60（41.7%） | 93.3% |
| Workflow LoRA v2 | 36/60（60.0%） | 100% |

SFT 修复 12 条、回退 1 条；双侧配对精确检验约 `p=0.0034`。

### Workflow DPO：同一 final v4

| 配置 | 严格端到端 | 单步 | 两步 | 三步 |
|---|---:|---:|---:|---:|
| Workflow LoRA v2 | 63/80（78.8%） | 10/10 | 21/28 | 32/42 |
| Workflow DPO v1 | 66/80（82.5%） | 10/10 | 23/28 | 33/42 |

DPO 修复 3 条、回退 0 条，但配对检验约 `p=0.25`。因此结论是“小幅正向、未达到统计显著”，而不是 DPO 已经解决多步规划。

final v3 与 final v4 的问题分布不同，不能跨测试集直接比较绝对分数。

### 真实 ECG 演示

MIT-BIH 记录 207 的最终三步轨迹：

```text
load_signal(signal)
  → filter_signal(0.5–40 Hz)
  → calculate_heart_rate()
```

结果为 29 个心搏峰、平均心率 56.847 BPM，并以 `plan_completed` 停止。该演示是工程验收，不是新的未见临床泛化测试。

更完整的实验设计、统计比较与失败分析见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)。

## 仓库结构

```text
.
├── physioagent/              # 工具、Agent、SFT/DPO训练与评测代码
├── scripts/                  # 数据生成和 MIT-BIH 准备脚本
├── tests/                    # 135 个单元/集成测试
├── evaluation/               # 开发集、冻结测试与 manifest
├── data/
│   ├── sample_ecg.csv        # CPU MVP 玩具信号
│   ├── real/                 # 带来源与许可记录的 MIT-BIH 片段
│   └── */manifest.json       # 可再生成训练数据的哈希与分布
├── outputs/**/REPORT.md      # 精简实验报告，不含模型权重
├── docs/
│   ├── REPRODUCIBILITY.md    # 服务器复现命令
│   ├── INTERVIEW_GUIDE.md    # 面试讲解与刁钻问题
│   └── GITHUB_UPLOAD.md      # GitHub上传白名单与检查
├── Dockerfile
├── EXPERIMENT_REPORT.md
├── README_EN.md
└── README.md
```

模型权重、checkpoint、optimizer、原始逐例输出和可再生成的大型训练 JSONL 被 `.gitignore` 排除。

## 快速开始：CPU 工具与规则 Agent

推荐 Python 3.10+。

```bash
git clone https://github.com/yuer-you/PhysioAgent.git
cd PhysioAgent
python -m venv .venv
```

Linux/macOS：

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m physioagent.demo
pytest -q
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m physioagent.demo
pytest -q
```

CPU MVP 不需要下载 Qwen，也不会使用 GPU。

## 准备本地 Qwen

模型权重不随 GitHub 仓库发布。推荐从 Hugging Face 下载官方
[`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)：

```bash
python -m pip install -U huggingface_hub
hf download Qwen/Qwen2.5-3B-Instruct \
  --local-dir models/Qwen2.5-3B-Instruct
```

该模型是公开模型，通常不需要登录。下载完成后，项目默认从
`models/Qwen2.5-3B-Instruct` 读取权重。

如果服务器访问 Hugging Face 不稳定，可以改用 ModelScope：

```bash
python -m pip install -U modelscope
modelscope download \
  --model "Qwen/Qwen2.5-3B-Instruct" \
  --local_dir "models/Qwen2.5-3B-Instruct"
```

两种方式得到的目录均可直接使用。如果模型已经下载到其他位置，可以通过环境变量指定：

```bash
export PHYSIOAGENT_MODEL_PATH=/absolute/path/to/Qwen2.5-3B-Instruct
```

Windows PowerShell：

```powershell
$env:PHYSIOAGENT_MODEL_PATH = "D:\models\Qwen2.5-3B-Instruct"
```

所有模型加载默认使用 `local_files_only=True`，适合离线计算节点。

## 生成训练数据

训练 JSONL 不进入 Git 历史，可由固定种子和 manifest 复现：

```bash
python scripts/generate_sft_workflow_data_v2.py
python scripts/generate_workflow_dpo_data.py
```

关键哈希：

```text
Workflow SFT v2 train
66da58371d627997e13a1bd37b7c59c7a4f062e2d8d84171a7b1e350125e63a8

Workflow SFT v2 validation
aead0dbb6f1069e1ea5f899399578455b6612aaedd955409b9096c0722190a74

Workflow DPO v1 train
b6c7cd4244943cdfeb92b5dcb7ee805f4ffab8e212754b76ba4b1f3cae59ddd2

Workflow DPO v1 validation
b2a12a322522ccee388e7cbc82c9a7f939ec7212fc9ef90ad81fa738f84f053e
```

## 训练

安装训练依赖：

```bash
pip install -r requirements.txt -r requirements-train.txt
```

### Workflow SFT v2

先检查数据和 token 长度：

```bash
python -m physioagent.train_sft \
  --model-path "$PHYSIOAGENT_MODEL_PATH" \
  --train-file data/sft_workflow_v2/train.jsonl \
  --validation-file data/sft_workflow_v2/validation.jsonl \
  --output-dir outputs/sft/qwen2.5-3b-workflow-lora-v2 \
  --epochs 2 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --max-length 1024 \
  --dry-run \
  --inspect-token-lengths
```

移除 `--dry-run --inspect-token-lengths` 后开始训练。

### Workflow DPO v1

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
  --max-length 1024 \
  --dry-run \
  --inspect-token-lengths
```

DPO reference policy 是同一个 SFT adapter 的冻结副本；policy/reference 共享基础模型权重。通过 dry-run 后再移除两个检查参数。

完整环境、训练和评测命令见 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。

## 最终 Agent 演示

最终 adapter 不提交到 Git 历史。请自行训练，或从项目 Release/Hugging Face/ModelScope 下载后放到：

```text
outputs/dpo/qwen2.5-3b-workflow-dpo-v1/final_adapter
```

运行：

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

## Docker

公开 Dockerfile 默认基于 `pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime`：

```bash
docker build -t physioagent:latest .
docker run --rm --gpus all physioagent:latest python -m pytest -q
```

模型和 adapter 建议以只读 volume 挂载，不要复制进镜像。

## 数据来源与许可

真实 ECG 来自 [MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/)，数据文件遵循 [Open Data Commons Attribution License v1.0](https://physionet.org/content/mitdb/view-license/1.0.0/)。仓库只包含记录 100、101、200、207 第0通道的前30秒派生片段，并在 `data/real/README.md` 与每个 `reference.json` 中保留来源、许可和署名。

引用：

```text
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
IEEE Engineering in Medicine and Biology Magazine. 2001;20(3):45-50.
DOI: 10.13026/C2F305
```

## 限制

- 当前是单轮线性计划，不支持动态重规划和长期记忆；
- SFT/DPO 数据主要由受控模板合成，不等价于真实用户偏好；
- DPO 在 final v4 上只净提升3条，统计上不显著；
- ECG 只评测了四段30秒单通道数据，不能声称临床或跨设备泛化；
- `load_signal` 在当前执行器中同时承担来源选择语义与状态转换，未来应进一步拆分资源获取和内存处理。

## 文档

- [统一实验报告](EXPERIMENT_REPORT.md)
- [复现指南](docs/REPRODUCIBILITY.md)
- [面试讲解](docs/INTERVIEW_GUIDE.md)
- [GitHub发布清单](docs/GITHUB_UPLOAD.md)

## License

项目代码采用 [MIT License](LICENSE)。MIT-BIH 派生数据不受代码许可证覆盖，仍遵循 ODC Attribution 1.0。
