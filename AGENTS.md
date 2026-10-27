# AGENTS.md

本文件是 Codex 和 Claude Code 共用的唯一规范（`CLAUDE.md` 只引用本文件）。完整计划、里程碑和验收标准见 `docs/PLAN.md`，当前状态与交接记录见 `PROGRESS.md`。开始任何任务前先读本文件、项目计划和进度文档（如存在），再核对当前工作区。

## 项目是什么

`transformer-lab`：从零实现 Llama/Qwen 风格的 Decoder-only Transformer，在 TinyStories 上预训练小模型并做结构消融；再让自研架构载入 Qwen3-0.6B 权重，用手写 LoRA 在 GSM8K 上做 SFT。

这是一个**学习项目和简历项目**。用户是数学系本科生，能读代码，正在练习亲手写代码。目标是用户最终能在面试中讲清每一行，而不是最快写完。因此：

- 用中文回复用户。
- 解释时讲清张量形状、为什么这样写、换一种写法会出什么问题。
- 不确定的地方先问，不要猜着往下写。

## 硬性规则

1. **不用现成实现做核心模块。** 模型、注意力、归一化、位置编码、训练循环、LoRA 中禁止使用 `nn.Transformer`、`nn.TransformerEncoder/DecoderLayer`、`nn.MultiheadAttention`，以及 `transformers` / `peft` 的模型类。
    - `transformers` 只用于下载和读取 Qwen3 权重、在测试中做对照。
    - `peft` 只用于 F4 对照实验（`scripts/` 或 `tests/` 中），不得出现在 `src/tlab/` 的实现里。
    - `F.scaled_dot_product_attention` 只能在测试中当参考答案，或作为配置开关的可选加速路径；默认路径必须是手写实现。
2. **用户手写的文件，agent 不得改写实现。** 以下文件由用户亲手写：agent 只能讲解、写测试、做 review、在回复里给修改建议，不能直接编辑实现代码：
    - `src/tlab/model/attention.py` 中的 MHA 版本（M1）
    - `src/tlab/model/block.py`（M1）
    - `src/tlab/train/trainer.py` 的主训练循环（M2；日志、checkpoint 等外围代码 agent 可以写）
    - `src/tlab/finetune/lora.py` 中的 `LoRALinear`（M5）
    - 文件顶部会有注释 `# USER-WRITTEN: agent 只提建议，不改实现`。如果任务需要改这些文件，停下来告诉用户。
3. **先计划，后代码。** 每个任务先列出：要改哪些文件、实现思路、要新增哪些测试。等用户确认后再写。
4. **先测试，后实现。** 每个新功能都配单元测试。提交前 `pytest` 必须全部通过；不得为了让测试通过而删改已有测试的断言。
5. **形状注释。** 每个张量操作旁写形状，例如 `# (B, n_head, T, head_dim)`。
6. **实验只通过配置文件运行。** 不得擅自修改超参数；需要改时先说明理由并等用户确认。训练报错、loss 出现 NaN 或发散时，停下来汇报，不要自行调参重跑。
7. **结果只追加，不改写。** `experiments/results.md` 只能追加新行，不得修改或删除已有结果。只记录真实跑出的数字。
8. **不提交大文件。** `data/`、`checkpoints/`、`runs/`、模型权重、数据集都不进 git。

## 环境

| 设备 | 用途 |
| --- | --- |
| Windows + RTX 5060 8GB（WSL2 Ubuntu） | 所有正式训练、消融和微调 |
| MacBook Air M4 | 写代码、跑单元测试、tiny 配置的冒烟测试（CPU/MPS） |

- Python 3.11/3.12，`uv` 管理依赖。
- PyTorch 2.7+，**必须是 CUDA 12.8 及以上的构建**（RTX 5060 是 Blackwell / sm_120，旧构建会报 `no kernel image is available`）。
- GPU 训练统一用 bf16 autocast。显存上限 8GB，任何新配置先估算显存。
- 代码必须在没有 GPU 的机器上可导入、测试可运行：设备选择写成 `cuda` → `mps` → `cpu` 的回退。

常用命令（M0 建好后以 `pyproject.toml` 为准）：

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format .
uv run python scripts/train.py --config configs/tinystories_base.yaml
```

GPU 自检：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); torch.randn(1).cuda()"
```

## 仓库结构

```text
transformer-lab/
├── AGENTS.md / CLAUDE.md / README.md / pyproject.toml
├── configs/                 # 每个实验一个 YAML；ablations/、finetune/ 子目录
├── src/tlab/
│   ├── model/               # attention.py, layers.py, block.py, gpt.py
│   ├── tokenizer/bpe.py
│   ├── data/
│   ├── train/trainer.py
│   ├── finetune/            # lora.py, sft.py
│   ├── convert/qwen3.py     # HF 权重 → 自研模块的映射
│   └── eval/                # gsm8k.py, generate.py
├── scripts/                 # prepare_data.py, train.py, sample.py, eval_gsm8k.py
├── tests/
├── experiments/results.md
└── docs/
    ├── PLAN.md              # 计划、里程碑、验收标准
    └── notes/               # 用户自己的推导笔记，agent 不改
```

## 代码约定

- 类型标注；`ruff` 格式化。
- 配置用 dataclass + YAML，命令行可覆盖单个字段。
- 所有组件可通过配置开关切换（位置编码、归一化、FFN、注意力类型），方便消融。
- 每次训练在输出目录保存：完整配置、git commit 哈希、随机种子、tokens/s、显存峰值。
- 优先写简单、可读的代码，而不是最快的代码；性能优化单独成任务。

## 测试要求

至少覆盖：

- 注意力：因果性（改动未来 token 不影响前面位置的输出）；与 `F.scaled_dot_product_attention(is_causal=True)` 数值一致。
- GQA：`n_kv_heads == n_heads` 时与 MHA 输出一致；有无 KV cache 的生成结果一致。
- 参数量函数与手算一致。
- BPE：编码后解码还原原文。
- 权重转换：与 `transformers` 的 Qwen3 输出逐层对比，fp32 下 logits 最大误差在 1e-4 量级。
- LoRA：B 零初始化时输出与原层相同；merge 后输出与 unmerged 一致。
- SFT：prompt 部分的 label 为 -100。

## 实验记录格式

`experiments/results.md` 每个实验追加一行：

```markdown
| 日期 | 编号 | 配置文件 | commit | 种子 | val loss / 准确率 | tokens/s | 显存峰值 | 一句结论 |
```

对比实验要求：同 token 预算、同超参、同种子，只改一个变量；主结论至少 2 个种子。

## 工作流程

1. 一个任务 = 一个模块 + 它的测试 = 一个分支（`m1-attention`、`a1-rope` 这类命名）。
2. 先给计划，等确认。
3. 先写测试，再写实现，跑 `pytest`。
4. 另一个 agent 做 review（只列问题，不直接改代码）。
5. 用户读完 diff、能讲清楚后再合并到 `main`。`main` 必须始终可运行。

提交信息写清做了什么，例如 `add GQA attention with KV cache and tests`。

## 交接与协作

- 每个任务在 `PROGRESS.md` 标明实现方、审查方、范围和状态。Codex 与 Claude Code 可以轮换角色；审查方只列问题，由实现方修复。
- 同一工作区同一时间只由一个实现方写入。需要并行实现时使用独立分支和 checkout/worktree；共享工作区里的两个终端不算隔离。
- 接手时核对当前分支、commit、未提交改动及相关代码；Git 尚未初始化时如实记录。进度文档与实际文件不符时先核实，保留更正依据。
- 交接前更新 `PROGRESS.md` 顶部的当前状态，并按其中模板追加记录。历史记录保留失败与未验证项；更正旧结论时追加说明，不静默改写原始证据。
- 原对话只在需要追溯决策时定向补查。外部讨论确定的计划变更同步到 `docs/PLAN.md` 后交接；用户当前明确指令优先。
- 只把本次实际执行的检查写成本次验证；转述、历史结果和未运行的命令分别标明。Claude Code 未审查时不得写成已通过，其他 agent 的审查单独署名。

## 完成任务时汇报

- 改了哪些文件、为什么。
- `pytest` 结果。
- 如果跑了实验：关键数字和是否已追加到 `experiments/results.md`。
- 用户需要自己理解的 1–3 个关键点，以及面试官可能追问的问题（不给答案）。
