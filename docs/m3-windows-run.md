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

（第 2 批起）在完整验证集上给训练完的模型打分。训练命令退出码为 0 才做（loss 变成 NaN 但跑完了也算）；整批的训练与打分期间不改动任何已跟踪文件，`run.json` 与 `full_val.json` 里记录的 commit 才不带 dirty。记下这一步的耗时：

```bash
uv run python scripts/eval_full_val.py --ckpt runs/m3/<变体>/seed<种子>/ckpt.pt --out runs/m3/<变体>/seed<种子>/full_val.json 2>&1 | tee runs/m3_<变体>_seed<种子>_full_val.txt
```

输出应为 `完整验证集 val loss …（5,414,547 个 token，第 3052 步的存档）`；token 数或步数不对就停下汇报。

把小文件复制进仓库（第 2 批起多了 `full_val.json`）：

```bash
mkdir -p experiments/m3_ablations/<变体>/seed<种子> && cp runs/m3/<变体>/seed<种子>/run.json runs/m3/<变体>/seed<种子>/metrics.csv runs/m3/<变体>/seed<种子>/full_val.json experiments/m3_ablations/<变体>/seed<种子>/ && cp runs/m3_<变体>_seed<种子>_console.txt experiments/m3_ablations/<变体>/seed<种子>/train_console.txt && cp runs/m3_<变体>_seed<种子>_gpu.csv experiments/m3_ablations/<变体>/seed<种子>/gpu.csv
```

整批的训练与打分都完成后，再按运行顺序为每次训练在 `experiments/results.md` 末尾各追加一行（与第 1 批的做法相同；数字取自 `run.json` 与 `full_val.json`；第 1 批没有完整验证集分数，不补写旧行）：

```markdown
| 运行日期 | M3-<变体>-s<种子> | configs/ablations/<变体>.yaml（seed=<种子>） | run.json 中的 git_commit | <种子> | val {final_val_loss，保留 4 位}（完整验证集 {full_val.json 的 val_loss，保留 4 位}） | {tokens_per_sec，取整} | {peak_memory_mb，取整} MB | 一句如实的结论 |
```

一批全部跑完后，生成汇总表：

```bash
uv run python scripts/summarize_ablations.py experiments/m3_ablations --configs configs/ablations --out experiments/m3_ablations/SUMMARY.md
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

## 第 2 批：对照组补评 + A2 归一化（7 次训练）

本批由 Claude Code 于 2026-11-03 追加，对应分支 `m3-norm`（RMSNorm、norm / norm_position 开关、A2 配置）与 `m3-evalfull`（完整验证集评估）。

- 前提：用户已把这两个分支合并到 `main`，或指定了要用的分支。先 `git fetch origin && git switch <用户指定的分支> && git pull --ff-only`，再 `uv sync --locked`。
- GPU 自检同第 1 批；单元测试 `uv run pytest -q`，预期 **170 passed**（Mac 上提交前实测），全部通过才继续。
- 数据仍复用 `data/tinystories/`（train 536,174,492、val 5,414,548 个 token）。

### 第 1 步：给对照组的两个存档补评（不训练）

第 1 批的两个模型还在 `runs/m3/a0_control/seed1337/` 和 `seed2024/` 里。对种子 **1337** 和 **2024** 各运行一次上面固定流程里的打分命令（`<变体>` = `a0_control`），再只把 `full_val.json` 复制过去：

```bash
cp runs/m3/a0_control/seed<种子>/full_val.json experiments/m3_ablations/a0_control/seed<种子>/
```

- 找不到 `ckpt.pt`、`ckpt_step` 不是 3052，或 token 数不是 5,414,547：停下汇报，不要重新训练。
- 这一步不往 `results.md` 加行（不是新训练）；两个分数写进本批的 Codex 交接记录。

### 第 2 步：A2 归一化对比（4 次训练，按固定流程，含打分）

| 变体 | 种子 | 首行参数量 | 与对照组的唯一区别 |
| --- | --- | --- | --- |
| `a2_postln` | 1337、2024 | 13,968,384 | LayerNorm 放在残差加法之后（Post-LN），没有最终 ln_f |
| `a2_rmsnorm` | 1337、2024 | 13,964,160 | 13 个归一化全部换成 RMSNorm |

### 第 3 步：高学习率加测（3 次训练，只跑种子 1337，按固定流程，含打分）

| 变体 | 首行参数量 | 结构 |
| --- | --- | --- |
| `a2_preln_lr3e-3` | 13,969,152 | 对照组结构 |
| `a2_postln_lr3e-3` | 13,968,384 | Post-LN |
| `a2_rmsnorm_lr3e-3` | 13,964,160 | Pre-RMSNorm |

三者的学习率都是 3e-3、min_lr 3e-4，其余与对照组相同。**本步是通用规则 2 的例外**：这一组本来就是看会不会发散，loss 出现 NaN 或上升时**不要中断、不要调参**，让它跑完 3052 步，照常打分、复制、追加 results.md（数字原样写，NaN 就写 NaN），在结论里如实写“发散”或“未发散”。如果是程序报错退出（不是 loss 变成 NaN），照通用规则停下汇报。

### 收尾

- 全部完成后运行汇总命令更新 `SUMMARY.md`。对照组与 A2 变体的每个种子都有 `full_val.json` 时，表里“依据”一列应为“完整验证集”。
- 交接记录里写明：对照组两个补评分数与耗时；7 次训练各自的退出码、最终 train/val loss、完整验证集分数、tokens/s、显存峰值、训练时长、GPU 平均与最高温度；高学习率组是否发散。
- 预计耗时：7 次训练各约 21 分钟（共约 2.5 小时），每次打分预计不到 1 分钟（以实测为准）。
- 提交与推送前先问用户。完成后告诉用户，由用户回到 Mac 让 Claude Code 核对。

## 第 2 批补充：高学习率组第 2 个种子（3 次训练）

本节由 Claude Code 于 2026-11-04 追加。第 2 批里三个 `a2_*_lr3e-3` 配置只有种子 1337，而 `AGENTS.md` 要求主结论至少 2 个种子，所以补种子 2024（决定见 `docs/PLAN.md`「M3 实施计划」的“A2 结果之后的决定”）。这三次只用 `main` 上已有的代码和配置，与正在开发的 `m3-mlp` 无关。

- 前提：在 `main` 上执行，先 `git fetch origin && git switch main && git pull --ff-only`，再 `uv sync --locked`。
- GPU 自检同第 1 批；单元测试 `uv run pytest -q`，预期 **170 passed**，全部通过才继续。
- 数据仍复用 `data/tinystories/`（train 536,174,492、val 5,414,548 个 token）。
- 依次运行下表 3 次，每次按上面的固定流程（记录 GPU、训练、完整验证集打分、复制小文件），种子都是 **2024**，命令行只覆盖 `seed=2024` 和 `out_dir=runs/m3/<变体>/seed2024`。

| 变体 | 首行参数量 | 结构 |
| --- | --- | --- |
| `a2_preln_lr3e-3` | 13,969,152 | 对照组结构 |
| `a2_postln_lr3e-3` | 13,968,384 | Post-LN |
| `a2_rmsnorm_lr3e-3` | 13,964,160 | Pre-RMSNorm |

- **例外同第 2 批第 3 步**：loss 出现 NaN、上升或长期不降时不中断、不调参，跑完 3052 步，照常打分、复制、记录；程序报错退出才按通用规则停下汇报。
- 只做这 3 次，不重跑种子 1337，也不跑任何其他配置。
- 收尾同第 2 批：全部完成后统一追加 `results.md`（每次一行）、运行汇总命令更新 `SUMMARY.md`，并在 `PROGRESS.md` 追加署名 Codex 的记录。汇总表的高学习率部分应有这三个配置各 2 行。如果某次的 loss 一直停在高位（如第 2 批 Post-LN 的约 5.84），在记录里如实写明曲线，不要只写“未发散”。
- 预计耗时：3 次训练各约 22 分钟，共约 1.1 小时，每次打分约 30 秒。
- 提交与推送前先问用户。完成后告诉用户，由用户回到 Mac 让 Claude Code 核对。

## 第 3 批：A3 SwiGLU（2 次训练）

本批由 Claude Code 于 2026-11-04 追加，对应分支 `m3-mlp`（SwiGLU、`mlp` 开关、`a3_swiglu` 配置，以及汇总表的“未收敛”标记）。

- 前提：用户已把 `m3-mlp` 合并到 `main`，或指定了要用的分支。先 `git fetch origin && git switch <用户指定的分支> && git pull --ff-only`，再 `uv sync --locked`。
- GPU 自检同第 1 批；单元测试 `uv run pytest -q`，预期 **190 passed**（`efb4883` 上实测；如果在之后的 `main` 上执行，以最新一批写的数字为准），全部通过才继续。
- 数据仍复用 `data/tinystories/`（train 536,174,492、val 5,414,548 个 token）。
- 按固定流程（记录 GPU、训练、完整验证集打分、复制小文件）依次运行：

| 变体 | 种子 | 首行参数量 | 与对照组的唯一区别 |
| --- | --- | --- | --- |
| `a3_swiglu` | 1337、2024 | 13,969,152 | 前馈子层换成 SwiGLU（隐藏维度 1024），参数量与对照组相同 |

- **参数量和对照组相同，首行查不出跑错结构。** 每次训练结束后多核对一步，必须打印 `swiglu`，否则停下汇报：

```bash
uv run python -c "import json; print(json.load(open('runs/m3/a3_swiglu/seed<种子>/run.json'))['config']['mlp'])"
```

- 通用规则 2 照常适用：出错、OOM、loss NaN 或发散都立即停下汇报。高学习率加测的例外不适用于本批。
- 收尾同第 2 批：全部完成后统一追加 `results.md`（每次一行）、运行汇总命令更新 `SUMMARY.md`，并在 `PROGRESS.md` 追加署名 Codex 的记录。这次重新生成的 `SUMMARY.md`，`a3_swiglu` 一行的“依据”应为“完整验证集”；高学习率表里 Post-LN 两行会变成“未发散，但未收敛”，这是汇总脚本的更新，不是结果变了。
- 预计耗时：2 次训练各约 22 分钟，共约 45 分钟。显存峰值估计约 5.2 GB（对照组实测 5.0 GB，SwiGLU 为反向传播多留约 190 MiB，是 Mac 上按真实形状算的，不是 GPU 实测；上限 8 GB），超过 7 GB 时在记录里提醒。
- 提交与推送前先问用户。完成后告诉用户，由用户回到 Mac 让 Claude Code 核对。

## 第 4 批：A1 位置编码（4 次训练 + 4 次外推评测）

本批由 Claude Code 于 2026-11-04 追加，对应分支 `m3-pos`（RoPE、正弦位置编码、`pos_encoding` 开关、外推评测）。

- 前提：用户已把 `m3-pos` 合并到 `main`，或指定了要用的分支。先 `git fetch origin && git switch <用户指定的分支> && git pull --ff-only`，再 `uv sync --locked`。
- GPU 自检同第 1 批；单元测试 `uv run pytest -q`，预期 **235 passed**（Mac 上提交前实测），全部通过才继续。
- 数据仍复用 `data/tinystories/`（train 536,174,492、val 5,414,548 个 token）。
- 按固定流程（记录 GPU、训练、完整验证集打分、复制小文件）依次运行：

| 变体 | 种子 | 首行参数量 | 与对照组的唯一区别 |
| --- | --- | --- | --- |
| `a1_sinusoidal` | 1337、2024 | 13,772,544 | 原论文的正弦位置编码（token 向量乘 √C），没有位置表 |
| `a1_rope` | 1337、2024 | 13,772,544 | RoPE，每层注意力里旋转 q、k，没有位置表 |

- **两个变体参数量相同，首行分不开。** 每次训练结束后多核对一步，打印结果必须和变体一致（`sinusoidal` 或 `rope`），否则停下汇报：

```bash
uv run python -c "import json; print(json.load(open('runs/m3/<变体>/seed<种子>/run.json'))['config']['pos_encoding'])"
```

- **外推评测**：每次复制完小文件之后，再在 2 倍长度（1024）上评测一次，并把结果复制进仓库。这一步同样要在改动任何已跟踪文件之前做：

```bash
uv run python scripts/eval_extrapolation.py --ckpt runs/m3/<变体>/seed<种子>/ckpt.pt --out runs/m3/<变体>/seed<种子>/extrapolation.json 2>&1 | tee runs/m3_<变体>_seed<种子>_extrapolation.txt
```

```bash
mkdir -p experiments/m3_ablations/<变体>/seed<种子> && cp runs/m3/<变体>/seed<种子>/extrapolation.json experiments/m3_ablations/<变体>/seed<种子>/
```

  输出应为 `长度 1024：前半段 …，后半段 …（每半段 2,706,944 个 token，第 3052 步的存档）`；长度、token 数或步数不对就停下汇报。显存不足时停下汇报，不要自行改 batch 大小。

- 通用规则 2 照常适用：出错、OOM、loss NaN 或发散都立即停下汇报。高学习率加测的例外不适用于本批。
- 收尾同第 2 批：全部完成后统一追加 `results.md`（每次一行）、运行汇总命令更新 `SUMMARY.md`，并在 `PROGRESS.md` 追加署名 Codex 的记录，其中写明每次的外推结果和耗时。汇总命令会把每次结果与配置文件核对，不一致时报错，此时停下汇报，不要去掉 `--configs`。`SUMMARY.md` 应出现“A1 外推”一节；两个 `a1_*` 行的“依据”应为“完整验证集”。
- 预计耗时：4 次训练各约 22 分钟，共约 1.5 小时，外推评测每次预计 1–2 分钟（以实测为准）。显存峰值估计约 5.0 GB（对照组实测 5.0 GB，RoPE 为反向传播多留的内存很少；是 Mac 上按真实形状算的，不是 GPU 实测），超过 7 GB 时在记录里提醒。
- 提交与推送前先问用户。完成后告诉用户，由用户回到 Mac 让 Claude Code 核对。
