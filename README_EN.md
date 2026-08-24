# PhysioAgent

English | [简体中文](README.md)

A multi-step physiological time-series tool-calling agent built with Qwen2.5-3B, LoRA SFT, and DPO.

PhysioAgent converts Chinese or English questions into strict multi-step JSON plans, which are then executed by deterministic Python tools on real ECG signals. The model plans the workflow but does not generate medical measurements directly; heart rate, R peaks, and statistics all come from tool outputs.

> For education and software evaluation only. Not intended for diagnosis, treatment, or any clinical decision-making.

## Highlights

- Five CPU-friendly tools for loading, filtering, statistics, R-peak detection, and mean heart rate;
- A custom linear agent loop with strict schemas, cross-step state transfer, stopping conditions, and complete execution traces;
- An end-to-end post-training path: Qwen baseline → LoRA SFT → Workflow SFT → DPO;
- A frozen SFT adapter as the DPO reference policy, rather than the base model;
- A newly frozen test set with a recorded SHA-256 hash before each training round;
- Layered evaluation of model planning, tool execution, ECG algorithms, and grounded answers;
- All Workflow SFT and DPO training completed on a single 16 GB RTX A4000;
- 135 automated tests covering tools, data, training dry runs, evaluation, and the final demo.

## Architecture

```text
User question + signal_profile
        │
        ▼
Qwen2.5-3B / LoRA SFT / DPO Planner
        │
        ▼
Strict {"steps":[...]} JSON and argument validation
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
grounded answer + raw plan + execution trace + stop reason
```

The current agent is a testable, single-turn linear planner. It does not include dynamic replanning, long-term memory, or multi-agent collaboration.

## Main Results

### Workflow SFT on the same final v3 set

| Configuration | Strict end-to-end success | Valid-plan rate |
|---|---:|---:|
| Base Qwen | 25/60 (41.7%) | 93.3% |
| Workflow LoRA v2 | 36/60 (60.0%) | 100% |

SFT fixed 12 cases and regressed on 1 case. The two-sided exact paired test gives approximately `p=0.0034`.

### Workflow DPO on the same final v4 set

| Configuration | Strict end-to-end | One step | Two steps | Three steps |
|---|---:|---:|---:|---:|
| Workflow LoRA v2 | 63/80 (78.8%) | 10/10 | 21/28 | 32/42 |
| Workflow DPO v1 | 66/80 (82.5%) | 10/10 | 23/28 | 33/42 |

DPO fixed 3 cases with no regressions, but the paired test gives approximately `p=0.25`. The supported conclusion is therefore “a small positive change without statistical significance,” not that DPO solved multi-step planning.

The question distributions of final v3 and final v4 differ, so their absolute scores must not be compared directly across test sets.

### Real ECG demo

The final three-step trace on MIT-BIH record 207 is:

```text
load_signal(signal)
  → filter_signal(0.5–40 Hz)
  → calculate_heart_rate()
```

The run detects 29 heartbeat peaks, reports a mean heart rate of 56.847 BPM, and stops with `plan_completed`. This demo is an engineering acceptance check, not a new unseen clinical-generalization test.

See [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) for the complete experimental design, statistical comparisons, and failure analysis.

## Repository Layout

```text
.
├── physioagent/              # Tools, agent, SFT/DPO training, and evaluation
├── scripts/                  # Data generation and MIT-BIH preparation scripts
├── tests/                    # 135 unit and integration tests
├── evaluation/               # Development sets, frozen tests, and manifests
├── data/
│   ├── sample_ecg.csv        # Toy signal for the CPU MVP
│   ├── real/                 # MIT-BIH excerpts with provenance and license records
│   └── */manifest.json       # Hashes and distributions for reproducible datasets
├── outputs/**/REPORT.md      # Compact reports without model weights
├── docs/
│   ├── REPRODUCIBILITY.md    # Server reproduction commands
│   ├── INTERVIEW_GUIDE.md    # Interview narrative and challenging questions
│   └── GITHUB_UPLOAD.md      # GitHub upload allowlist and checks
├── Dockerfile
├── EXPERIMENT_REPORT.md
├── README_EN.md
└── README.md
```

Model weights, checkpoints, optimizer states, raw per-case outputs, and large reproducible training JSONL files are excluded by `.gitignore`.

## Quick Start: CPU Tools and Rule-Based Agent

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/yuer-you/PhysioAgent.git
cd PhysioAgent
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m physioagent.demo
pytest -q
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m physioagent.demo
pytest -q
```

The CPU MVP does not require Qwen or a GPU.

## Download Qwen Locally

Model weights are not distributed with this GitHub repository. The recommended source is the official
[`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) repository on Hugging Face:

```bash
python -m pip install -U huggingface_hub
hf download Qwen/Qwen2.5-3B-Instruct \
  --local-dir models/Qwen2.5-3B-Instruct
```

This is a public model and normally does not require authentication. PhysioAgent reads it from `models/Qwen2.5-3B-Instruct` by default.

If Hugging Face connectivity is unreliable on your server, use ModelScope as a fallback:

```bash
python -m pip install -U modelscope
modelscope download \
  --model "Qwen/Qwen2.5-3B-Instruct" \
  --local_dir "models/Qwen2.5-3B-Instruct"
```

Both methods produce a directly usable local directory. If the model already exists elsewhere, set an environment variable:

```bash
export PHYSIOAGENT_MODEL_PATH=/absolute/path/to/Qwen2.5-3B-Instruct
```

Windows PowerShell:

```powershell
$env:PHYSIOAGENT_MODEL_PATH = "D:\models\Qwen2.5-3B-Instruct"
```

All model loaders use `local_files_only=True` by default, which is suitable for offline compute nodes.

## Generate Training Data

Training JSONL files are excluded from Git history. They can be reproduced from fixed seeds and manifests:

```bash
python scripts/generate_sft_workflow_data_v2.py
python scripts/generate_workflow_dpo_data.py
```

Key hashes:

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

## Training

Install the training dependencies:

```bash
pip install -r requirements.txt -r requirements-train.txt
```

### Workflow SFT v2

Inspect the data and token lengths first:

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

Remove `--dry-run --inspect-token-lengths` to start training.

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

The DPO reference policy is a frozen copy of the same SFT adapter. The policy and reference share the base-model weights. Remove the two inspection flags after a successful dry run.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete environment, training, and evaluation commands.

## Final Agent Demo

The final adapter is not committed to Git history. Train it locally, or download it from a project release, Hugging Face, or ModelScope and place it at:

```text
outputs/dpo/qwen2.5-3b-workflow-dpo-v1/final_adapter
```

Run:

```bash
python -m physioagent.final_demo
```

Custom question:

```bash
python -m physioagent.final_demo \
  --question "First read the signal column, retain 1 to 30 Hz, and then detect R peaks" \
  --signal-file data/real/mitdb/207_30s/signal.csv \
  --output outputs/demo/custom_record207.json
```

## Docker

The public Dockerfile uses `pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime` by default:

```bash
docker build -t physioagent:latest .
docker run --rm --gpus all physioagent:latest python -m pytest -q
```

Mount models and adapters as read-only volumes instead of copying them into the image.

## Data Provenance and Licensing

The real ECG signals come from the [MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/), whose data files are distributed under the [Open Data Commons Attribution License v1.0](https://physionet.org/content/mitdb/view-license/1.0.0/). This repository contains only derived 30-second excerpts from channel 0 of records 100, 101, 200, and 207. Provenance, licensing, and attribution are retained in `data/real/README.md` and each `reference.json`.

Citation:

```text
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
IEEE Engineering in Medicine and Biology Magazine. 2001;20(3):45-50.
DOI: 10.13026/C2F305
```

## Limitations

- The current implementation uses single-turn linear plans without dynamic replanning or long-term memory;
- Most SFT and DPO data come from controlled templates and do not represent real user preferences;
- DPO provides a net improvement of only three cases on final v4, without statistical significance;
- ECG evaluation covers only four 30-second, single-channel excerpts and does not establish clinical or cross-device generalization;
- In the current executor, `load_signal` combines source-selection semantics with state transition. Future work should separate resource acquisition from in-memory processing.

## Documentation

- [Unified experiment report](EXPERIMENT_REPORT.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Interview guide](docs/INTERVIEW_GUIDE.md)
- [GitHub release checklist](docs/GITHUB_UPLOAD.md)

## License

The project code is released under the [MIT License](LICENSE). The MIT-BIH derived data is not covered by the code license and remains subject to ODC Attribution 1.0.
