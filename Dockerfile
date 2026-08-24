# 默认使用公开 PyTorch CUDA 镜像；实验室可通过 --build-arg BASE_IMAGE=... 覆盖。
ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
FROM ${BASE_IMAGE}

WORKDIR /workspace/physioagent
ENV PYTHONPATH=/workspace/physioagent

# 先复制依赖文件：代码变化不会让 Docker 重装依赖。
COPY requirements.txt requirements-train.txt requirements-real-data.txt ./
# PyArrow 从源码构建需要庞大的 C++/Rust 工具链。训练镜像只接受官方预编译 wheel。
RUN python -m pip install --only-binary=:all: pyarrow==17.0.0
RUN python -m pip install -r requirements.txt -r requirements-train.txt -r requirements-real-data.txt

COPY physioagent ./physioagent
COPY data ./data
COPY scripts ./scripts
COPY tests ./tests
COPY evaluation ./evaluation
COPY docs ./docs
COPY EXPERIMENT_REPORT.md ./
COPY README.md ./

# OpenPAI 会覆盖 CMD 并运行任务命令；CMD 只用于手动进入容器时的默认行为。
CMD ["bash"]
