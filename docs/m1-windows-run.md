# M1 Windows 正式训练手册（交给 Codex 执行）

本手册由 Claude Code 于 2026-10-30 编写，供 Windows + RTX 5060（WSL2 Ubuntu）上的 Codex 按步骤执行 M1 的正式训练，并把结果写回仓库，之后由 Claude Code 在 Mac 上读取。开始前先读 `AGENTS.md`、`docs/PLAN.md` 和 `PROGRESS.md` 顶部的「当前状态」。

## 背景

- M1 代码已全部完成并推送：分支 `m1-generate`（提交 `dbd7c78` 及之后的文档提交）包含注意力、Block、GPT、字符级数据、训练循环、生成函数。各 M1 分支**尚未合并 `main`**，所以必须在 `m1-generate` 上工作。
- 用户手写的文件（顶部有 `# USER-WRITTEN` 标记）：`src/tlab/model/attention.py`、`src/tlab/model/block.py`、`src/tlab/model/gpt.py`、`src/tlab/train/trainer.py`。**不得修改。**
- Mac 上已验证：`pytest` 48 passed；tiny_smoke 在 CPU 上 10 步跑通；Windows/CUDA 尚未验证过任何一项。

## 硬性规则（摘自 AGENTS.md，务必遵守）

1. **不改任何超参数和配置文件。** 只运行下面列出的命令；需要改动时停下来问用户。
2. **出错就停。** 安装失败、测试失败、CUDA 不可用、训练报错、loss 出现 NaN 或明显发散（比如 val loss 持续上升或突然变大很多），都立即停止，把原始报错和已有输出记录下来，不要自行调参重跑。
3. **只记录真实跑出的数字。** `experiments/results.md` 只追加新行，不改已有内容。
4. **不提交大文件。** `data/`、`runs/`、`*.pt`、`*.bin` 不进 git；需要回传的结果按「第 4 步」复制成小文本文件。
5. **提交和推送前先征得用户同意。** 不修改全局 git 配置（如 user.name/email），缺少时问用户。

## 第 1 步：准备环境（在 WSL Ubuntu 终端中）

工作目录放在 Linux 主目录（`~`），不要放在 `/mnt/c/...` 下。

```bash
cd ~
```

如未安装 uv（已有则跳过）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
```

```bash
export PATH="$HOME/.local/bin:$PATH"
```

获取代码（私有仓库；需要凭据时由用户本人输入用户名 `JAzer-M` 和访问令牌，agent 不代填）：

```bash
git clone https://github.com/JAzer-M/transformer-lab.git
```

```bash
cd ~/transformer-lab && git checkout m1-generate && git log --oneline -3
```

安装依赖（Linux 上会安装 PyTorch 2.9.1+cu128，约 2–3 GB）：

```bash
uv sync --locked
```

## 第 2 步：M0 在 Windows 上的验收

**2.1 GPU 自检**。预期输出含 `2.9.1+cu128`、`12.8` 和 `RTX 5060`，且最后的 `.cuda()` 不报错：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); torch.randn(1).cuda()"
```

若报 `no kernel image is available` 或 CUDA 不可用，停止并记录（见 AGENTS.md 环境说明）。

**2.2 单元测试**。预期 `48 passed`：

```bash
uv run pytest -q
```

## 第 3 步：数据、冒烟、正式训练、生成

**3.1 准备数据**。预期打印「字符总数 1,115,394，字表大小 65」以及训练集 1,003,854、验证集 111,540：

```bash
uv run python scripts/prepare_shakespeare_char.py
```

若下载报 SSL 证书错误，用 curl 下载同一文件后重跑上面的命令：

```bash
mkdir -p data/shakespeare_char && curl -fsSL -o data/shakespeare_char/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

核对原文哈希，应为 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`（与 Mac 上一致）：

```bash
sha256sum data/shakespeare_char/input.txt
```

**3.2 冒烟测试**（CPU 上 10 步，补 M0 的「CPU 10 步训练」验收）。预期打印第 0、5、9 步的 loss 并以「完成」结尾：

```bash
mkdir -p runs && uv run python scripts/train.py --config configs/tiny_smoke.yaml 2>&1 | tee runs/tiny_smoke_console.txt
```

**3.3 正式训练**（`configs/shakespeare_char.yaml`：6 层、6 头、d_model 384、block_size 256、batch 64、lr 1e-3、2000 步、每 250 步评估、种子 1337；参数量应为 10,750,080）：

```bash
uv run python scripts/train.py --config configs/shakespeare_char.yaml 2>&1 | tee runs/shakespeare_char_console.txt
```

开头一行必须是 `设备 cuda，参数量 10,750,080`；如果设备不是 cuda，立即中断并记录（说明 GPU 没被用上）。显存粗估 2–3 GB、时长估计为几分钟，均为估算，以实测为准。

**3.4 生成样例**（固定参数，不要改动）：

```bash
uv run python scripts/sample.py --ckpt runs/shakespeare_char/ckpt.pt --prompt "ROMEO:" --max-new-tokens 500 2>&1 | tee runs/shakespeare_char_sample.txt
```

## 第 4 步：把结果写回仓库

`runs/` 不进 git，所以把需要回传的小文件复制到 `experiments/m1_shakespeare_char/`：

```bash
mkdir -p experiments/m1_shakespeare_char && cp runs/shakespeare_char/run.json runs/shakespeare_char/metrics.csv experiments/m1_shakespeare_char/ && cp runs/shakespeare_char_console.txt experiments/m1_shakespeare_char/train_console.txt && cp runs/shakespeare_char_sample.txt experiments/m1_shakespeare_char/sample.txt && cp runs/tiny_smoke_console.txt experiments/m1_shakespeare_char/tiny_smoke_console.txt
```

另外新建 `experiments/m1_shakespeare_char/env_check.txt`，写入第 2 步 GPU 自检的完整输出和 `pytest` 的最后一行。

**在 `experiments/results.md` 末尾追加一行**（数字全部取自 `run.json`；只在正式训练正常跑完时追加）：

```markdown
| 运行日期 | M1-1 | configs/shakespeare_char.yaml | run.json 中的 git_commit | 1337 | val {final_val_loss，保留 4 位} | {tokens_per_sec，取整} | {peak_memory_mb，取整} MB | 一句如实的结论 |
```

**在 `PROGRESS.md` 更新顶部「当前状态」，并按文末模板追加一条交接记录**，至少写明：

- 执行者为 Codex、设备为 Windows/WSL + RTX 5060、分支与 commit；
- 每条命令的退出码和关键输出（GPU 名称、PyTorch/CUDA 版本、pytest 结果、数据统计与哈希、冒烟的 loss、正式训练每次评估的 train/val loss、最终 val loss、tokens/s、显存峰值、训练时长）；
- val loss 是否稳定下降，是否出现过拟合迹象（val 先降后升）；
- 失败或未完成的步骤，原样保留报错；
- 生成样例是否有剧本格式（人名、冒号、换行）。

经用户同意后提交并推送到 `m1-generate`：

```bash
git add experiments/ PROGRESS.md && git commit -m "record M1 shakespeare_char training on RTX 5060" && git push
```

## 第 5 步：交回

告诉用户已完成（或在哪一步停下），由用户回到 Mac 让 Claude Code `git pull` 读取结果，再一起判断 M1 是否达到验收标准（`docs/PLAN.md`：5060 上几分钟内训练出格式像剧本的文本，val loss 稳定下降）以及是否合并到 `main`。
