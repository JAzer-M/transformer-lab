# M5 微调实验手册（交给 Codex 或用户执行）

本手册由 Claude Code 于 2026-11-07 编写，在 Windows + RTX 5060（WSL2 Ubuntu）上运行 M5 的实验，并把结果写回仓库。Codex 额度不够时由用户直接运行。下面的命令都假设仓库在 `~/transformer-lab`。

## 通用规则

1. **不改超参数和配置。** 只运行本手册列出的命令，命令行只能用这里写出的覆盖项。
2. **出错就停。** 报错、OOM、loss 出现 NaN 或发散，立即停止，原样记录，不自行调参重跑。
3. **只记录真实跑出的数字。** `experiments/results.md` 只追加，不改已有行。
4. **不提交大文件。** `data/`、`runs/`、权重不进 git。
5. **不改用户手写的文件**（顶部有 `# USER-WRITTEN`）。
6. **运行前不改动已跟踪的文件**，保证结果里的 `git_commit` 不带 `-dirty`。
7. **提交和推送前先征得用户同意。** 不修改全局 git 配置、系统代理或网络设置。
8. **用了 `tee` 就看不到命令的退出码，** 所以先执行一次 `set -o pipefail`，并且看每条命令的输出。

环境里设了 SOCKS 代理（`ALL_PROXY=socks5://...`）时，下载会报缺 socksio。如果同时有 HTTP 代理，在命令前加 `env -u ALL_PROXY -u all_proxy`，这条命令就只走 HTTP 代理。

## 第 1 批：基座准确率（F1）和 bf16 对比（不训练）

前提是 `m5-baseline` 已合并到 `main`。这一批只做评测：先看基座的 fp32 和 bf16 与 `transformers` 差多少，再测基座在固定的 500 题上的准确率，确认有提升空间。

### 1. 同步和检查

```bash
cd ~/transformer-lab && git fetch origin && git switch main && git pull --ff-only && uv sync --locked
```

```bash
uv run pytest -q
```

预期 **570 passed**（Mac 上提交前实测），全部通过才继续。

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); torch.randn(1).cuda()"
```

```bash
mkdir -p experiments/m5_gsm8k runs && set -o pipefail
```

### 2. 准备数据（约 2.7 MB）

```bash
uv run python scripts/prepare_gsm8k.py
```

应打印 `train：7473 条` 和 `test：1319 条`。文件大小或 SHA-256 不符时脚本会报错。

### 3. 和 transformers 对比（第一次下载基座权重，约 1.2 GB）

fp32，有门槛：

```bash
uv run python scripts/check_qwen3.py --repo Qwen/Qwen3-0.6B-Base --device cuda --out experiments/m5_gsm8k/base_alignment_cuda_fp32.json 2>&1 | tee runs/m5_align_fp32.txt
```

最后一行应是 `通过`。不通过就停下，把输出交给用户。

bf16，只报告差多少，不判定：

```bash
uv run python scripts/check_qwen3.py --repo Qwen/Qwen3-0.6B-Base --device cuda --autocast bf16 --out experiments/m5_gsm8k/base_alignment_cuda_bf16.json 2>&1 | tee runs/m5_align_bf16.txt
```

最后一行应含 `只报告，不判定`。贪心生成不相同也不算错，原样记录，这正是要看的内容。

### 4. 冒烟：每种提示 5 题

```bash
uv run python scripts/eval_gsm8k.py --config configs/finetune/f1_base_fewshot.yaml --out runs/m5_smoke_fewshot.json limit=5 2>&1 | tee runs/m5_smoke_fewshot.txt
```

```bash
uv run python scripts/eval_gsm8k.py --config configs/finetune/f1_base_zeroshot.yaml --out runs/m5_smoke_zeroshot.json limit=5 2>&1 | tee runs/m5_smoke_zeroshot.txt
```

记下打印的 tokens/s。每种提示最多生成 500 × 256 = 128,000 个 token，用它除以 tokens/s 就是耗时上限。Mac 的 CPU 上是 15–20 tokens/s，GPU 上还没测过。**任何一种估算超过 3 小时，先问用户再继续。** 这 5 题的准确率没有意义，不要记录。

### 5. 正式评测

每次评测期间记录 GPU 状态，结束后结束这个后台进程：

```bash
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,utilization.gpu,memory.used,power.draw --format=csv -l 5 > runs/m5_f1_gpu.csv &
```

```bash
uv run python scripts/eval_gsm8k.py --config configs/finetune/f1_base_fewshot.yaml --out experiments/m5_gsm8k/f1_base_fewshot.json 2>&1 | tee runs/m5_f1_fewshot.txt
```

```bash
uv run python scripts/eval_gsm8k.py --config configs/finetune/f1_base_zeroshot.yaml --out experiments/m5_gsm8k/f1_base_zeroshot.json 2>&1 | tee runs/m5_f1_zeroshot.txt
```

```bash
kill %1
```

每 50 题打印一行进度。结果只在跑完时写出，中途出错要从头再跑。

### 6. 核对和记录

确认四个 json 里的 `git_commit` 都不带 `-dirty`，`device` 是 `cuda`，两个 F1 结果的 `total` 是 500。

`experiments/results.md` 追加 3 行，数字从 json 里抄，一个字也不要估：

```markdown
| 日期 | F1-base-fewshot | configs/finetune/f1_base_fewshot.yaml | commit | — | 准确率（accuracy，correct/500） | tokens_per_second | peak_memory_mb | 基座 few-shot |
| 日期 | F1-base-zeroshot | configs/finetune/f1_base_zeroshot.yaml | commit | — | 准确率（accuracy，correct/500） | tokens_per_second | peak_memory_mb | 基座不给例子 |
| 日期 | M5-align-bf16 | Qwen/Qwen3-0.6B-Base（scripts/check_qwen3.py --autocast bf16） | commit | — | 分数最大误差 max_error（bf16，GPU） | — | — | 只报告，不判定 |
```

征得用户同意后提交推送：

```bash
git add experiments/m5_gsm8k experiments/results.md && git commit -m "record the GSM8K baseline accuracy of the base model" && git push
```

`PROGRESS.md` 不用在这边改，由 Claude Code 核对时记录。完成后回到 Mac，让 Claude Code 从原始文件核对。
