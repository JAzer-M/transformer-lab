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

## 第 2 批：LoRA SFT 短测（显存和速度，不正式训练）

本节由 Claude Code 于 2026-11-08 追加，对应分支 `m5-sft`。前提是 `m5-sft` 已合并到 `main`，否则脚本不存在。这一批只做短测：看 8 GB 显存够不够、训练和评测有多快，再由用户确认超参数。**这一批不跑完整训练，也不追加 `results.md` 的 F1、F2 行。**

### 1. 同步和检查

```bash
cd ~/transformer-lab && git fetch origin && git switch main && git pull --ff-only && uv sync --locked
```

```bash
uv run pytest -q
```

预期 **690 passed**（Mac 上提交前实测），全部通过才继续。

```bash
mkdir -p experiments/m5_gsm8k/short runs/m5 && set -o pipefail
```

数据和基座权重第 1 批已经准备好；如果 `data/gsm8k/train.jsonl` 不在，先跑一次 `uv run python scripts/prepare_gsm8k.py`。

### 2. 开关打开后，bf16 下和 transformers 差多少

第 1 批测的是加 `rmsnorm_fp32` 开关之前（分数最大误差 0.656）。现在载入 Qwen3 时会自动打开，再测一次：

```bash
uv run python scripts/check_qwen3.py --repo Qwen/Qwen3-0.6B-Base --device cuda --autocast bf16 --out experiments/m5_gsm8k/base_alignment_cuda_bf16_rmsnorm_fp32.json 2>&1 | tee runs/m5_align_bf16_rmsnorm_fp32.txt
```

最后一行应含 `只报告，不判定`。json 里的 `rmsnorm_fp32` 应是 `true`。

### 3. 显存上限：只跑最长的那一步

`memory_probe=true` 只跑一步，用训练集里最长的 16 条（最长 445 个 token），显存峰值就是正式训练的上限。r=8 和 r=32 各跑一次（F2 里 r 最大是 32）：

```bash
uv run python scripts/train_sft.py --config configs/finetune/sft_r8.yaml memory_probe=true out_dir=runs/m5/probe_r8 2>&1 | tee runs/m5_probe_r8.txt
```

```bash
uv run python scripts/train_sft.py --config configs/finetune/sft_r8.yaml memory_probe=true r=32 out_dir=runs/m5/probe_r32 2>&1 | tee runs/m5_probe_r32.txt
```

最后一行会打印显存峰值。**OOM 或报错就停下，不要改 batch、精度等设置重跑**，把输出交给用户：退路（微批 2 × 累积 8、基座转 bf16、gradient checkpointing）要用户决定。

### 4. 训练速度：正式配置跑 20 步

```bash
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,utilization.gpu,memory.used,power.draw --format=csv -l 5 > runs/m5_speed_gpu.csv &
```

```bash
uv run python scripts/train_sft.py --config configs/finetune/sft_r8.yaml max_steps=20 out_dir=runs/m5/speed_r8 2>&1 | tee runs/m5_speed_r8.txt
```

```bash
kill %1
```

第 20 步会看一次 val loss。记下最后一行的 tokens/s 和用时；正式训练一个 epoch 是 454 步，按每步用时估算总时长。

### 5. 评测速度：用刚才 20 步的 LoRA 答 20 题

微调后的模型会在 eos 停下，评测应该比第 1 批快。这 20 题的准确率没有意义（只训练了 20 步），只看速度和是否在 eos 停下：

```bash
uv run python scripts/eval_gsm8k.py --config configs/finetune/f1_sft_r8.yaml --out runs/m5/speed_r8_eval.json lora=runs/m5/speed_r8/lora.pt limit=20 2>&1 | tee runs/m5_speed_r8_eval.txt
```

### 6. 整理结果

把短测的原始结果拷进仓库（都很小）：

```bash
for run in probe_r8 probe_r32 speed_r8; do cp runs/m5/$run/run.json experiments/m5_gsm8k/short/${run}_run.json && cp runs/m5/$run/metrics.csv experiments/m5_gsm8k/short/${run}_metrics.csv; done && cp runs/m5/speed_r8_eval.json experiments/m5_gsm8k/short/speed_r8_eval.json && cp runs/m5_speed_gpu.csv experiments/m5_gsm8k/short/speed_gpu.csv
```

确认这些 json 里的 `git_commit` 都不带 `-dirty`，`device` 是 `cuda`。

`experiments/results.md` 只追加 1 行（对齐检查；短测不算实验，不记）：

```markdown
| 日期 | M5-align-bf16-rmsnorm32 | Qwen/Qwen3-0.6B-Base（scripts/check_qwen3.py --autocast bf16，rmsnorm_fp32 打开） | commit | — | 分数最大误差 max_error（bf16，GPU） | — | — | 只报告，不判定 |
```

征得用户同意后提交推送：

```bash
git add experiments/m5_gsm8k experiments/results.md && git commit -m "record the M5 SFT short tests on RTX 5060" && git push
```

跑完告诉用户这几个数：bf16 对齐的分数最大误差和贪心是否相同；两次 probe 的显存峰值；speed 的 tokens/s、每步用时和第 20 步的 train、val loss；评测 20 题的 tokens/s 和停在 eos 的题数（`stopped_by_eos`）。`PROGRESS.md` 不用在这边改，由 Claude Code 核对时记录。

## 第 3 批：F2（4 种 r × 2 个种子）和 F1 基座重测

本节由 Claude Code 于 2026-11-08 追加，对应分支 `m5-f2`。前提是 `m5-f2` 已合并到 `main`。超参数已由用户确认（2026-11-08）：r 为 4、8、16、32，alpha 固定 16，学习率 2e-4，warmup 20 步，cosine 降到 2e-5，**微批 2 × 累积 8**（第 2 批里 4 × 4 的最长一批要 11 GB，显卡只有 8 GB），1 个 epoch（454 步），种子 1337 和 2024。粗算整批约 7–8 小时。

**停下的条件（除了通用规则）：** 第 2 步显存短测的峰值不低于 7,500 MiB；训练时 loss 出现 nan；某次训练最后的 val loss 比它第一次打印的 val loss 还高。遇到这些都停下，把输出交给用户。

### 1. 同步和检查

```bash
cd ~/transformer-lab && git fetch origin && git switch main && git pull --ff-only && uv sync --locked
```

```bash
uv run pytest -q
```

预期 **694 passed**（Mac 上提交前实测），全部通过才继续。

```bash
mkdir -p experiments/m5_gsm8k/f2 runs/m5 && set -o pipefail
```

整批都记录 GPU 状态，全部跑完后再结束：

```bash
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,utilization.gpu,memory.used,power.draw --format=csv -l 5 > runs/m5_batch3_gpu.csv &
```

### 2. 关口：用新的 2 × 8 再测一次显存

```bash
uv run python scripts/train_sft.py --config configs/finetune/f2_r32.yaml memory_probe=true out_dir=runs/m5/probe_r32_b2 2>&1 | tee runs/m5_probe_r32_b2.txt
```

最后一行打印的显存峰值**低于 7,500 MiB 才继续**，否则停下问用户。

### 3. 先跑 r=8、种子 1337：它同时是 F1 里“微调后”的结果

```bash
uv run python scripts/train_sft.py --config configs/finetune/f2_r8.yaml seed=1337 out_dir=runs/m5/f2_r8_seed1337 2>&1 | tee runs/m5_f2_r8_seed1337.txt
```

```bash
uv run python scripts/eval_gsm8k.py --config configs/finetune/f1_sft_r8.yaml --out experiments/m5_gsm8k/f2/r8_seed1337_eval.json 2>&1 | tee runs/m5_f2_r8_seed1337_eval.txt
```

`f1_sft_r8.yaml` 里写的就是 `runs/m5/f2_r8_seed1337/lora.pt`。

### 4. F1 基座重测（`rmsnorm_fp32` 已打开）

第 1 批的基座结果是加开关之前跑的，保留不动；这次写到新文件：

```bash
uv run python scripts/eval_gsm8k.py --config configs/finetune/f1_base_fewshot.yaml --out experiments/m5_gsm8k/f1_base_fewshot_rmsnorm32.json 2>&1 | tee runs/m5_f1_fewshot_rmsnorm32.txt
```

```bash
uv run python scripts/eval_gsm8k.py --config configs/finetune/f1_base_zeroshot.yaml --out experiments/m5_gsm8k/f1_base_zeroshot_rmsnorm32.json 2>&1 | tee runs/m5_f1_zeroshot_rmsnorm32.txt
```

### 5. 其余 7 次：每次训练完马上评测

某一次出错就停下，不往下跑：

```bash
for run in r4:1337 r4:2024 r8:2024 r16:1337 r16:2024 r32:1337 r32:2024; do
  r=${run%%:*}; seed=${run##*:}; name=${r}_seed${seed}
  uv run python scripts/train_sft.py --config configs/finetune/f2_${r}.yaml seed=$seed out_dir=runs/m5/f2_$name 2>&1 | tee runs/m5_f2_$name.txt \
    && uv run python scripts/eval_gsm8k.py --config configs/finetune/f1_sft_r8.yaml --out experiments/m5_gsm8k/f2/${name}_eval.json lora=runs/m5/f2_$name/lora.pt 2>&1 | tee runs/m5_f2_${name}_eval.txt \
    || { echo "$name 失败，停下"; break; }
done
```

`f1_sft_r8.yaml` 后面的 `lora=...` 换成这一次的存档；评测设置（题目、提示、生成）和其他几次完全相同。

```bash
kill %1
```

### 6. 整理结果

训练的记录拷进仓库（LoRA 存档留在 `runs/`，不提交）：

```bash
for run in probe_r32_b2 f2_r8_seed1337 f2_r4_seed1337 f2_r4_seed2024 f2_r8_seed2024 f2_r16_seed1337 f2_r16_seed2024 f2_r32_seed1337 f2_r32_seed2024; do cp runs/m5/$run/run.json experiments/m5_gsm8k/f2/${run#f2_}_run.json && cp runs/m5/$run/metrics.csv experiments/m5_gsm8k/f2/${run#f2_}_metrics.csv; done && cp runs/m5_batch3_gpu.csv experiments/m5_gsm8k/f2/batch3_gpu.csv
```

确认所有 json 的 `git_commit` 都不带 `-dirty`，`device` 是 `cuda`，评测的 `total` 都是 500。

`experiments/results.md` 追加 10 行，数字从 json 里抄。训练的 tokens/s 和显存峰值用 `*_run.json` 里的，准确率用 `*_eval.json` 里的：

```markdown
| 日期 | F1-base-fewshot-rmsnorm32 | configs/finetune/f1_base_fewshot.yaml | commit | — | 准确率（correct/500） | tokens_per_second | peak_memory_mb | 基座 few-shot，rmsnorm_fp32 打开 |
| 日期 | F1-base-zeroshot-rmsnorm32 | configs/finetune/f1_base_zeroshot.yaml | commit | — | 准确率（correct/500） | tokens_per_second | peak_memory_mb | 基座不给例子，rmsnorm_fp32 打开 |
| 日期 | F2-r4-s1337 | configs/finetune/f2_r4.yaml | commit | 1337 | val loss final_val_loss；准确率（correct/500） | tokens_per_second | peak_memory_mb | 待 Claude 汇总 |
```

F2 每次一行，共 8 行（r 为 4、8、16、32，种子 1337、2024），格式同第 3 行。

征得用户同意后提交推送：

```bash
git add experiments/m5_gsm8k experiments/results.md && git commit -m "record M5 F2 rank runs and the F1 base rerun on RTX 5060" && git push
```

跑完告诉用户：显存短测的峰值；每次训练的最后 val loss、用时和显存峰值；每次评测的准确率；基座两种提示的准确率。`PROGRESS.md` 不用在这边改，由 Claude Code 核对时记录。

## 第 4 批：F4，自研 LoRA 和 PEFT 对比 loss 曲线

本节由 Claude Code 于 2026-11-08 追加，对应分支 `m5-peft`。前提：`m5-peft` 已合并到 `main`，**第 3 批已经跑完**（要用到 `runs/m5/f2_r8_seed1337/` 和 `runs/m5/f2_r8_seed2024/`，它们只在这台 WSL 上）。约 20 分钟。

PEFT 那边从自研那次训练存下的 `lora_init.pt` 出发，数据、顺序、训练循环和超参数都和 `f2_r8` 种子 1337 那次相同，只有 LoRA 的实现不同。“基本重合”的门槛已经定好（2026-11-08 用户确认）：每一步 train loss 之差的平均 ≤ 0.01，最后的 val loss 之差 ≤ 0.01。

### 1. 同步和检查

```bash
cd ~/transformer-lab && git fetch origin && git switch main && git pull --ff-only && uv sync --locked
```

```bash
uv run pytest -q
```

预期 **718 passed**（Mac 上提交前实测），全部通过才继续。`uv sync` 会装上 `peft` 0.21.2 和它依赖的几个包。

```bash
ls runs/m5/f2_r8_seed1337/lora_init.pt runs/m5/f2_r8_seed2024/metrics.csv && mkdir -p experiments/m5_gsm8k/f4 && set -o pipefail
```

两个文件都在才继续。

### 2. 关口：PEFT 的显存

官方模型的实现和我们的不同，先测一次最长那一步的显存：

```bash
uv run python scripts/train_peft.py --config configs/finetune/f4_peft_r8.yaml memory_probe=true out_dir=runs/m5/probe_peft_r8 2>&1 | tee runs/m5_probe_peft_r8.txt
```

显存峰值**低于 7,500 MiB 才继续**，否则停下问用户。

### 3. 正式训练

```bash
uv run python scripts/train_peft.py --config configs/finetune/f4_peft_r8.yaml 2>&1 | tee runs/m5_f4_peft_r8_seed1337.txt
```

### 4. 比较曲线

```bash
uv run python scripts/compare_curves.py --ours runs/m5/f2_r8_seed1337 --theirs runs/m5/f4_peft_r8_seed1337 --reference runs/m5/f2_r8_seed2024 --out experiments/m5_gsm8k/f4/compare.json 2>&1 | tee runs/m5_f4_compare.txt
```

脚本先核对两次训练的起点和配置是否相同，不同就报错（这时停下问用户）。最后一行是“达标”或“未达标”；未达标时退出码是 1，**这不是出错，结果照样记录**，不要重跑或改门槛。

### 5. 整理结果

```bash
for run in probe_peft_r8 f4_peft_r8_seed1337; do cp runs/m5/$run/run.json experiments/m5_gsm8k/f4/${run}_run.json && cp runs/m5/$run/metrics.csv experiments/m5_gsm8k/f4/${run}_metrics.csv; done
```

确认 json 里的 `git_commit` 不带 `-dirty`，`device` 是 `cuda`，`implementation` 是 `peft`。

`experiments/results.md` 追加 1 行，数字从 `f4_peft_r8_seed1337_run.json` 和 `compare.json` 里抄：

```markdown
| 日期 | F4-peft-r8-s1337 | configs/finetune/f4_peft_r8.yaml | commit | 1337 | val loss final_val_loss；和自研比：每步平均差 mean_abs_train_diff，最后 val 差 final_val_diff（达标/未达标） | tokens_per_second | peak_memory_mb | 待 Claude 核对 |
```

征得用户同意后提交推送：

```bash
git add experiments/m5_gsm8k/f4 experiments/results.md && git commit -m "record the M5 F4 comparison between our LoRA and PEFT" && git push
```

跑完告诉用户：PEFT 显存短测的峰值；正式训练最后的 train、val loss 和用时；`compare.json` 里的第 1 步差、每步平均差和最大差、最后的 val 差、换种子的差距，以及是否达标。
