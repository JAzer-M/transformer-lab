# transformer-lab

从零实现 Llama/Qwen 风格的 Decoder-only Transformer：先在 TinyStories 上预训练小模型并做结构消融，再用自研架构载入 Qwen3-0.6B 权重，手写 LoRA 完成 GSM8K SFT。

这是学习项目。MHA、Transformer block、主训练循环和 `LoRALinear` 由用户亲手实现；具体边界见 [AGENTS.md](AGENTS.md)。路线与验收见 [docs/PLAN.md](docs/PLAN.md)，实际进度和交接证据见 [PROGRESS.md](PROGRESS.md)。

## 当前阶段

M0 第一批的工程骨架、依赖锁定、设备选择与单元测试已实现，Mac 本地检查及独立审查通过，待用户阅读改动验收。模型、训练入口和训练配置将在后续任务实现；完整 M0 仍需 Windows/WSL 真实 CUDA 自检、该设备上的 pytest，以及 CPU 10 步训练验收。

## 环境与安装

- Python 3.11 或 3.12，uv 管理项目环境与锁文件。
- MacBook Air M4：开发、单元测试和短时间 CPU/MPS 冒烟检查。
- Windows + RTX 5060 8GB（WSL2 Ubuntu）：正式训练、消融与微调。
- PyTorch 固定为 2.9.1；macOS 从 PyPI 安装平台版，Linux/Windows 从官方 CUDA 12.8 源安装。平台选择规则保存在 `pyproject.toml`，锁文件应一并同步。

官方依据：[PyTorch 安装版本](https://pytorch.org/get-started/previous-versions/)、[uv 的 PyTorch 平台源配置](https://docs.astral.sh/uv/guides/integration/pytorch/)。Windows 的驱动与 GPU 可运行性仍须在目标设备实测。

尚未安装 uv 时，macOS 或 WSL 可按 [官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
export PATH="$HOME/.local/bin:$PATH"
```

在项目根目录安装并检查：

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

如果本机尚无符合要求的 Python，先用 uv 安装 Python 3.11，再同步环境。`uv.lock` 记录解析后的具体依赖；日常复现使用 `--locked`，避免无意更新锁文件。

## 设备选择

```python
from tlab.device import get_device

device = get_device()
```

按 CUDA → MPS → CPU 的顺序返回当前可用的 `torch.device`。这一步只选择设备；后续创建张量和放置模型时再使用它。单元测试模拟各分支，Mac 实机检查另行验证了 CPU/MPS 上的小张量分配；证据见进度文档。

## Windows/WSL GPU 自检

在 RTX 5060 所在的 WSL 环境安装依赖后执行：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); torch.randn(1).cuda()"
```

验收时记录 PyTorch/CUDA 版本、GPU 名称与实际分配结果。Mac 的成功不能替代此项验证。

## 协作与结果

当前由 Codex 继续实施，用户决定切换时再交接 Claude Code。每次接手读取项目规范、计划与进度，并核对实际工作区。新功能先写测试，再做最小实现，之后独立审查。

实验只通过经确认的配置运行；真实结果追加到 `experiments/results.md`。数据、模型权重、checkpoint 和运行日志留在本地，不进入 Git。
