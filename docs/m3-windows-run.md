# M3 消融实验手册（交给 Codex 执行）

本手册由 Claude Code 于 2026-11-02 编写，供 Windows + RTX 5060（WSL2 Ubuntu）上的 Codex 分批执行 M3 消融训练，并把结果写回仓库。M3 的组件按分支逐个完成，**每完成一个分支，本手册就追加一批**；只执行用户让你执行的那一批。开始前先读 `AGENTS.md`、`docs/PLAN.md` 的「M3 实施计划（2026-11-02 用户确认）」，以及 `PROGRESS.md` 顶部「当前状态」。

## 通用规则（每一批都适用）

1. **不改超参数和配置。** 只运行本手册列出的命令；命令行只允许覆盖 `seed` 和 `out_dir`。
2. **出错就停。** 训练报错、OOM、loss 出现 NaN 或发散，立即停止，原样记录报错和已有输出，不自行调参重跑。例外：A2 的高学习率加测本来就是要看会不会发散，按该批的说明处理。
3. **只记录真实跑出的数字。** `experiments/results.md` 只追加；每次训练一行。
4. **不提交大文件。** `data/`、`runs/`、`*.pt`、`*.bin` 不进 git。
5. **不改用户手写的文件**（顶部有 `# USER-WRITTEN`）。
6. **训练前不改动已跟踪的文件**，保证 `run.json` 的 `git_commit` 不带 `-dirty`；笔记先写在 `runs/` 下，跑完再整理。
7. **提交和推送前先征得用户同意。** 不修改全局 git 配置、系统代理或网络设置。
8. 每次训练期间用 `nvidia-smi` 每 5 秒记录一次 GPU 温度、频率、利用率、显存与功耗（命令见下面的固定流程），记录里写明平均和最高温度。速度对比另做短测（后面的批次会给），不要直接拿长训练的 tokens/s 下速度结论。

## 每次训练的固定流程

以变体 `<变体>`、种子 `<种子>` 为例（`<变体>` 就是 `configs/ablations/<变体>.yaml` 的文件名）。先在后台开始记录 GPU 状态：

```bash
mkdir -p runs && nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,utilization.gpu,memory.used,power.draw --format=csv -l 5 > runs/m3_<变体>_seed<种子>_gpu.csv & GPU_LOG=$!
```

再训练：

```bash
uv run python scripts/train.py --config configs/ablations/<变体>.yaml seed=<种子> out_dir=runs/m3/<变体>/seed<种子> 2>&1 | tee runs/m3_<变体>_seed<种子>_console.txt
```

首行必须是 `设备 cuda，参数量 …`，参数量与该批说明一致，否则立即中断。训练结束（或中断）后停止 GPU 记录：

```bash
kill $GPU_LOG
```

把小文件复制进仓库：

```bash
mkdir -p experiments/m3_ablations/<变体>/seed<种子> && cp runs/m3/<变体>/seed<种子>/run.json runs/m3/<变体>/seed<种子>/metrics.csv experiments/m3_ablations/<变体>/seed<种子>/ && cp runs/m3_<变体>_seed<种子>_console.txt experiments/m3_ablations/<变体>/seed<种子>/train_console.txt && cp runs/m3_<变体>_seed<种子>_gpu.csv experiments/m3_ablations/<变体>/seed<种子>/gpu.csv
```

在 `experiments/results.md` 末尾追加一行（数字取自 `run.json`）：

```markdown
| 运行日期 | M3-<变体>-s<种子> | configs/ablations/<变体>.yaml（seed=<种子>） | run.json 中的 git_commit | <种子> | val {final_val_loss，保留 4 位} | {tokens_per_sec，取整} | {peak_memory_mb，取整} MB | 一句如实的结论 |
```

一批全部跑完后，生成汇总表：

```bash
uv run python scripts/summarize_ablations.py experiments/m3_ablations --out experiments/m3_ablations/SUMMARY.md
```

最后更新 `PROGRESS.md` 顶部「当前状态」，并在交接记录末尾（「后续交接模板」一节之前）追加一条署名 Codex 的记录：设备、分支与 commit、每次训练的退出码、最终 train/val loss、tokens/s、显存峰值、训练时长、GPU 平均与最高温度、是否出现异常。经用户同意后提交并推送当前分支：

```bash
git add experiments/ PROGRESS.md && git commit -m "record M3 <批次名> ablation runs on RTX 5060" && git push
```

## 第 1 批：对照组（2 次训练）

- 前提：用户已把 `m3-ablate` 合并到 `main`，或让你在 `m3-ablate` 上执行。先 `git fetch origin && git switch <用户指定的分支> && git pull --ff-only`，再 `uv sync --locked`。
- GPU 自检（预期含 `2.9.1+cu128`、`12.8`、`RTX 5060`，且 `.cuda()` 不报错）：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); torch.randn(1).cuda()"
```

- 单元测试：`uv run pytest -q`，预期 **132 passed**（Mac 上提交前实测的数字），全部通过才继续。
- 数据复用 M2 已准备好的 `data/tinystories/`；先 `cat data/tinystories/info.json` 确认 train 536,174,492、val 5,414,548 个 token。
- 配置 `configs/ablations/a0_control.yaml`：M2 基线结构，参数量 **13,969,152**，3052 步、每次 100,007,936 个 token，预计每次约 20 分钟、显存峰值约 5 GB（按 M2 实测推算）。
- 依次运行种子 **1337** 和 **2024**（用上面的固定流程，`<变体>` = `a0_control`）。两次之间不需要等待降温。

完成后告诉用户，由用户回到 Mac 让 Claude Code 核对。
