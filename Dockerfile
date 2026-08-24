# 默认使用公开 PyTorch CUDA 镜像；实验室可通过 --build-arg BASE_IMAGE=... 覆盖。
# Use the public PyTorch CUDA image by default; labs can override it with --build-arg BASE_IMAGE=...
ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
FROM ${BASE_IMAGE}

WORKDIR /workspace/physioagent
ENV PYTHONPATH=/workspace/physioagent

# 先复制依赖文件：代码变化不会让 Docker 重装依赖。
# Copy dependency files first so code changes do not trigger dependency reinstallation.
COPY requirements.txt requirements-train.txt requirements-real-data.txt ./
# PyArrow 从源码构建需要庞大的 C++/Rust 工具链。训练镜像只接受官方预编译 wheel。
# Building PyArrow from source requires a large C++/Rust toolchain; use the official prebuilt wheel.
RUN python -m pip install --only-binary=:all: pyarrow==17.0.0
RUN python -m pip install -r requirements.txt -r requirements-train.txt -r requirements-real-data.txt

COPY physioagent ./physioagent
COPY data ./data
COPY scripts ./scripts
COPY tests ./tests
COPY evaluation ./evaluation
COPY docs ./docs
COPY EXPERIMENT_REPORT.md ./
COPY README.md README_EN.md ./

# OpenPAI 会覆盖 CMD 并运行任务命令；CMD 只用于手动进入容器时的默认行为。
# OpenPAI overrides CMD with the job command; this default is only for manual container sessions.
CMD ["bash"]
