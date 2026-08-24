# GitHub 发布清单

## 应提交

### 根目录

```text
.dockerignore
.gitattributes
.gitignore
Dockerfile
README.md
README_EN.md
EXPERIMENT_REPORT.md
requirements.txt
requirements-train.txt
requirements-real-data.txt
```

### 源码、脚本与测试

```text
physioagent/*.py
scripts/*.py
tests/*.py
```

不要提交任何 `__pycache__`、`.pyc` 或 `.pytest_cache`。

### 冻结评测

提交整个 `evaluation/`。这些 JSON/JSONL 文件体积小，包含开发集、冻结测试、数据划分和 SHA-256 manifest，是项目可复现性的核心证据。

### 文档

```text
docs/INTERVIEW_GUIDE.md
docs/REPRODUCIBILITY.md
docs/GITHUB_UPLOAD.md
```

### 数据

提交：

```text
data/sample_ecg.csv
data/real/README.md
data/real/mitdb/*/signal.csv
data/real/mitdb/*/reference.json
data/**/README.md
data/**/manifest.json
```

四段 MIT-BIH 片段总计不到 0.4 MB，可以随仓库发布，但必须保留 `data/real/README.md` 和每条 `reference.json` 中的来源、许可与署名。MIT-BIH 文件使用 Open Data Commons Attribution License v1.0。

合成 SFT/DPO 的 `train.jsonl`、`validation.jsonl`、`test.jsonl` 不提交。它们接近 19 MB，且可由 `scripts/generate_*.py` 和 manifest 确定性复现。

### 实验报告

`.gitignore` 只允许 `outputs/` 下的 `REPORT.md` 进入仓库。提交这些报告即可；原始逐例输出、训练状态、checkpoint 和权重不进入源码仓库。

## 不应提交

```text
outputs/**/checkpoint-*/
outputs/**/final_adapter/
adapter_model.safetensors
optimizer.pt
training_args.bin
trainer_state.json
results.jsonl
模型 tokenizer 副本
基础 Qwen 权重
.env 或访问令牌
DOCKER_SETUP_INSIS.md
Dockerfile.base
```

`DOCKER_SETUP_INSIS.md`、`Dockerfile.base` 和旧输出中包含实验室私有镜像、NFS 路径或内部环境细节，已由 `.gitignore` 排除。公开的 `Dockerfile` 使用 PyTorch 官方镜像。

## Adapter 发布建议

最终 DPO adapter 约 15 MB，完整 adapter 目录约 30 MB。不要直接提交到 Git 历史。推荐二选一：

1. 上传到 Hugging Face Hub 或 ModelScope，README 中填写下载链接；
2. 压缩后作为 GitHub Release 附件发布。

发布内容只需最终 DPO `final_adapter/`，不要发布 optimizer 或 checkpoint。基础 Qwen 权重由用户从原模型仓库自行下载。

## 许可证

项目代码已采用根目录 `LICENSE` 中的 MIT License。代码许可证不覆盖 MIT-BIH 数据，数据仍遵循其 ODC Attribution 1.0。

## 提交前检查

在 PowerShell 中：

```powershell
git init
git branch -M main
git add .
git status --short
git status --ignored
```

确认暂存区没有权重或 checkpoint：

```powershell
git diff --cached --name-only | Select-String -Pattern "safetensors|optimizer.pt|checkpoint|__pycache__|\.pyc$"
```

该命令应没有输出。再检查暂存文件大小：

```powershell
git diff --cached --stat
```

确认后提交：

```powershell
git commit -m "feat: publish PhysioAgent SFT-DPO workflow agent"
git remote add origin https://github.com/yuer-you/PhysioAgent.git
git push -u origin main
```
