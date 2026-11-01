# M2 任务 D：TinyStories 基线训练手册（交给 Codex 执行）

本手册由 Claude Code 于 2026-11-01 编写，供 Windows + RTX 5060（WSL2 Ubuntu）上的 Codex 执行 M2 任务 D：全量数据准备 → 吞吐短跑 → **停下等用户确认** → 正式训练 → 回写结果。开始前先读 `AGENTS.md`、`docs/PLAN.md`（「M2」「M2 实施计划」及任务 B、C 的决定）和 `PROGRESS.md` 顶部「当前状态」与「会话约定」。

## 背景

- M2 任务 A（手写 BPE）、B（TinyStories 数据管线）、C（学习率调度、梯度累积、梯度裁剪、断点续训）已合并 `main`（8056ac0）。本任务在分支 **`m2-baseline`** 上进行，它包含本手册和吞吐短跑配置 `configs/tinystories_probe.yaml`。
- Mac 上已验证：pytest 109 passed；V2 valid 小样本（21.5 MiB）端到端跑通；默认配置下训练结果与 M1 逐步一致。**Windows 上尚未用真实 TinyStories 数据跑过任何一步**（上次审查时网络不通）。
- 用户手写的文件（顶部有 `# USER-WRITTEN`）：`src/tlab/model/attention.py`、`block.py`、`gpt.py`、`src/tlab/train/trainer.py`、`src/tlab/tokenizer/bpe.py`。**不得修改。**
- PLAN 的 M2 验收标准：单次训练 1–2 小时内完成，生成连贯英文短故事，loss 曲线和吞吐有记录。

## 硬性规则（摘自 AGENTS.md）

1. **不擅自改超参数和配置。** 第 1–3 步只运行下面列出的命令。正式训练的配置必须等用户在第 4 步确认后才能写。
2. **出错就停。** 下载或校验失败、测试失败、CUDA 不可用、OOM、训练报错、loss 出现 NaN 或发散，都立即停止，原样记录报错和已有输出，不自行调参重跑。
3. **只记录真实跑出的数字。** `experiments/results.md` 只追加新行；吞吐短跑不写入结果表。
4. **不提交大文件。** `data/`、`runs/`、`*.pt`、`*.bin` 不进 git；需要回传的结果复制成小文本文件（第 6 步）。
5. **提交和推送前先征得用户同意。** 不修改全局 git 配置、系统代理或网络设置。
6. 正式训练前尽量不改动已跟踪的文件（包括 `PROGRESS.md`），这样 `run.json` 里的 `git_commit` 不带 `-dirty`；需要做笔记先写在 `runs/` 下，最后再整理进 `PROGRESS.md`。

## 第 1 步：更新代码与环境检查

```bash
cd ~/transformer-lab && git fetch origin && git switch m2-baseline && git pull --ff-only && git log --oneline -3
```

确认 HEAD 与 `origin/m2-baseline` 一致后：

```bash
uv sync --locked
```

GPU 自检（预期含 `2.9.1+cu128`、`12.8`、`RTX 5060`，且 `.cuda()` 不报错）：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); torch.randn(1).cuda()"
```

单元测试（预期 `109 passed`）：

```bash
uv run pytest -q
```

磁盘与内存（需要约 3.5 GB 空闲磁盘；记录 WSL 可用内存）：

```bash
df -h ~ && free -h
```

## 第 2 步：准备全量 TinyStories 数据

**2.1 先确认能连上 Hugging Face。** 上次在 Windows 上 HF 请求报 `Network is unreachable`，所以先测：

```bash
curl -sI --max-time 20 https://huggingface.co/api/datasets/roneneldan/TinyStories | head -1
```

- 返回 `HTTP/2 200` 即可继续。
- 失败时停下来告诉用户，请用户自己开启代理或决定是否使用镜像；agent 不启动代理、不改系统网络设置。用户同意用镜像时，在下面的命令前加 `HF_ENDPOINT=https://hf-mirror.com`（文件仍会核对 SHA-256，镜像文件不对会报错）。
- 若报 `Using SOCKS proxy, but the 'socksio' package is not installed`（`ALL_PROXY` 是 socks5、`HTTPS_PROXY` 是 http），在命令前加 `env -u ALL_PROXY -u all_proxy`，让下载走 HTTP 代理（Mac 上就是这样解决的）。
- 也可以由用户用浏览器或 curl 把两个文件下载到 `data/tinystories/raw/`，脚本发现文件已存在时只核对、不联网。

**2.2 下载、训练词表、编码。** 会下载约 2.1 GiB 的训练集和 21.5 MiB 的验证集，并核对大小与 SHA-256；词表在完整训练集上训练（8192，含 `<|endoftext|>`）。用 `/usr/bin/time -v` 记录内存峰值：

```bash
mkdir -p runs && /usr/bin/time -v uv run python scripts/prepare_tinystories.py 2>&1 | tee runs/prepare_tinystories_console.txt
```

预期（按 Mac 小样本约 4.16 字节/token 外推，**均为估算**）：训练集约 5.4 亿 token、`train.bin` 约 1.07 GB；验证集是整个 valid 文件，约 540 万 token。以 `data/tinystories/info.json` 实测为准。记录：`info.json` 全文、`Maximum resident set size`、总耗时。若内存不足被杀（WSL 约 7.6 GiB），停下记录。

**2.3 快速核对。** 结束标记个数应等于篇数，文件能读回：

```bash
cat data/tinystories/info.json && ls -la data/tinystories/
```

## 第 3 步：吞吐与显存短跑（不是正式实验）

配置 `configs/tinystories_probe.yaml`：PLAN 的例子模型（6 层、6 头、d_model 384、上下文 512、8192 词表），200 步、只在第 0 步和最后一步评估。

**显存粗估（未实测，误差可能很大）**：参数、梯度、AdamW 状态约 0.23 GB；注意力是手写实现，会显式保存 B × 6 × 512 × 512 的分数矩阵；lm_head 输出 B × 512 × 8192。按 M1（batch 64、上下文 256，实测 2597 MB）外推：batch 16 约 3 GB，batch 32 约 5–6 GB，batch 48 可能超过 8 GB，所以不测 48。

**3.1 batch 16。** 首行必须是 `设备 cuda，参数量 13,969,152`，否则立即中断：

```bash
uv run python scripts/train.py --config configs/tinystories_probe.yaml 2>&1 | tee runs/tinystories_probe_b16_console.txt
```

**3.2 batch 32**（只覆盖 batch_size 与输出目录；OOM 就停，记录报错，这本身就是要测的结果）：

```bash
uv run python scripts/train.py --config configs/tinystories_probe.yaml batch_size=32 out_dir=runs/tinystories_probe_b32 2>&1 | tee runs/tinystories_probe_b32_console.txt
```

每次记录 `run.json` 中的 `tokens_per_sec`、`peak_memory_mb`、`train_seconds`，以及第 0 步与最后一步的 loss（第 0 步应接近 ln 8192 ≈ 9.01）。

## 第 4 步：停下，交给用户确认正式配置

把第 2、3 步的实测数字报告给用户，并按下面的方法算出建议，**等用户明确确认后**才进入第 5 步。

**算法**：每步 token 数 = `batch_size × grad_accum_steps × 512`；`max_iters = 训练时长（秒） × 实测 tokens/s ÷ 每步 token 数`，训练时长按 1–2 小时内取（例如 75 分钟 = 4500 秒，给评估和存档留余量）。

**供用户确认的建议值**（Claude Code 提出，尚未确认；用户可改任意一项）：

| 项目 | 建议 | 说明 |
| --- | --- | --- |
| 模型 | 6 层、6 头、d_model 384、上下文 512 | PLAN 的例子；吞吐太低时由用户决定是否缩小 |
| batch_size × grad_accum_steps | 16 × 4（batch 32 放得下且更快时用 32 × 2） | 每步 64 条、32,768 个 token |
| learning_rate / lr_schedule / min_lr | 6e-4 / cosine / 6e-5 | 余弦降到峰值的十分之一 |
| warmup_iters | 约总步数的 2%–3%（例如 200–500） | |
| grad_clip | 1.0 | GPT-2、Llama 的常用值 |
| weight_decay | 0.01 | 与 M1 相同，不新增变化 |
| eval_interval / eval_iters | 500 / 20 | |
| ckpt_interval | 1000 | 中断后可用 `resume=true` 续训 |
| seed | 1337 | |

用户确认后，按确认的数值新建 `configs/tinystories_base.yaml`（字段与 `tinystories_probe.yaml` 相同，再加上 `grad_accum_steps`、`grad_clip`、`lr_schedule`、`warmup_iters`、`min_lr`、`ckpt_interval`，`data_dir: data/tinystories`，`out_dir: runs/tinystories_base`，`device: auto`），在文件顶部注释写明确认日期与参数量、显存实测依据。新建配置会让工作区变 dirty：先经用户同意提交这一个配置文件，再开始训练。

## 第 5 步：正式训练与生成

```bash
uv run python scripts/train.py --config configs/tinystories_base.yaml 2>&1 | tee runs/tinystories_base_console.txt
```

- 首行必须是 `设备 cuda，参数量 …`（与确认的模型一致）。
- 观察每次评估的 train/val loss：持续下降为正常；NaN、突然变大或 val 持续上升，立即中断并记录。
- 如果中途意外中断（不是因为 loss 异常），同一条命令加 `resume=true` 从最近存档继续，并在记录里写明中断与续训的步数：

```bash
uv run python scripts/train.py --config configs/tinystories_base.yaml resume=true 2>&1 | tee -a runs/tinystories_base_console.txt
```

训练完成后生成 3 个样例（生成参数固定，不要改动）：

```bash
for s in 1 2 3; do uv run python scripts/sample.py --ckpt runs/tinystories_base/ckpt.pt --prompt "Once upon a time" --max-new-tokens 300 --temperature 0.8 --top-k 50 --seed $s; echo; echo '-----'; done 2>&1 | tee runs/tinystories_base_samples.txt
```

## 第 6 步：把结果写回仓库

```bash
mkdir -p experiments/m2_tinystories_base && cp runs/tinystories_base/run.json runs/tinystories_base/metrics.csv experiments/m2_tinystories_base/ && cp runs/tinystories_base_console.txt experiments/m2_tinystories_base/train_console.txt && cp runs/tinystories_base_samples.txt experiments/m2_tinystories_base/samples.txt && cp data/tinystories/info.json experiments/m2_tinystories_base/data_info.json && cp runs/prepare_tinystories_console.txt experiments/m2_tinystories_base/prepare_console.txt
```

吞吐短跑的两个 `run.json` 也复制过去，便于以后核对：

```bash
cp runs/tinystories_probe/run.json experiments/m2_tinystories_base/probe_b16_run.json && cp runs/tinystories_probe_b32/run.json experiments/m2_tinystories_base/probe_b32_run.json
```

（batch 32 OOM 时没有它的 `run.json`，改为复制 `runs/tinystories_probe_b32_console.txt` 中的报错。）

另新建 `experiments/m2_tinystories_base/env_check.txt`，写入第 1 步 GPU 自检、pytest 最后一行、`df`/`free` 的输出，以及第 2.1 步的网络检查结果。

**在 `experiments/results.md` 末尾追加一行**（只在正式训练正常跑完时追加；数字取自 `run.json`）：

```markdown
| 运行日期 | M2-1 | configs/tinystories_base.yaml | run.json 中的 git_commit | 1337 | val {final_val_loss，保留 4 位} | {tokens_per_sec，取整} | {peak_memory_mb，取整} MB | 一句如实的结论 |
```

**更新 `PROGRESS.md` 顶部「当前状态」，并在交接记录末尾（「后续交接模板」一节之前）追加一条署名 Codex 的记录**，至少写明：设备、分支与 commit；每条命令的退出码与关键输出（网络检查、数据统计与内存峰值、两次短跑的 tokens/s 与显存、用户确认的正式配置、每次评估的 train/val loss、最终 val loss、tokens/s、显存峰值、训练时长、是否续训）；val loss 是否稳定下降；3 个样例是否是通顺的英文短故事；失败或未完成的步骤原样保留报错。

经用户同意后提交并推送到 `m2-baseline`：

```bash
git add experiments/ PROGRESS.md && git commit -m "record M2 TinyStories baseline training on RTX 5060" && git push
```

## 第 7 步：交回

告诉用户已完成或在哪一步停下。用户回到 Mac 后由 Claude Code `git pull` 核对结果，再一起判断 M2 是否达到验收标准，以及是否合并到 `main`。
