# Transformer 项目规划与投递前待办

2026-10-25 导出自 Claude 文档。

2026-10-26 起，本文件作为 Codex 与 Claude Code 共同执行的项目计划。外部文档与对话用于讨论和追溯；用户确定的计划变更同步到本文件后交接，用户当前明确指令优先。当前进度与交接记录统一维护在根目录 `PROGRESS.md`。

## 项目定位与最终交付物

5 周内（10/26–11/30）做出一个能在 12 月初投递时写进简历的 Transformer 项目：从零实现并预训练一个 Llama/Qwen 风格的小语言模型，做 4 组以上结构消融，再用自己的架构加载 Qwen3-0.6B 权重，并用手写 LoRA 做数学题微调。

它和 PaperLens 互补：PaperLens 证明你会搭 LLM 应用，这个项目证明你懂模型内部。核心原则是理解优先，`transformers` 只用于加载权重和做对照，不用来实现核心模块。

暂定仓库名 `transformer-lab`。最终交付物：

| 交付物 | 内容 | 截止 |
| --- | --- | --- |
| GitHub 仓库 | 干净代码、单元测试、README（结构图、结果表、loss 曲线、生成样例、复现命令） | 11/30 |
| 从零预训练模型 | 约 20M 参数，TinyStories 上训练，能生成连贯英文短故事 | 11/9 |
| 消融实验报告 | 位置编码、归一化、FFN、注意力四组对比，同 token 预算 | 11/16 |
| 权重对齐（亮点） | 自研架构载入 Qwen3-0.6B，输出与 HF 官方实现对齐 | 11/20 |
| 手写 LoRA + SFT | GSM8K 微调前后准确率对比，并与 PEFT 结果对照 | 11/27 |
| 学习笔记 + 面试问答 | 你自己的话写的推导与问答清单 | 持续 |

## 投递前总待办

到 12 月初投递前有五条线并行：Transformer 项目、刷题、PaperLens 收尾、面试准备、简历。Transformer 的具体任务见后文各里程碑，这里只列关口；其余四条线的事项都在这里，做完一项勾一项。

### Transformer 项目

- [ ] M0 环境与仓库就绪，写好 `AGENTS.md` 和 `CLAUDE.md`（2026-10-27）
- [x] MVP 关口：A1–A4 结果表可复现（2026-11-15；2026-11-07 由用户确认勾选）
- [x] 自研架构载入 Qwen3-0.6B，输出对齐（2026-11-20；2026-11-07 由用户确认勾选）
- [ ] 手写 LoRA + GSM8K SFT 结果出来（2026-11-27）
- [ ] README、实验报告定稿（2026-11-30）

### 刷题

- [ ] 每天固定时段刷题，与项目并行
- [ ] 优先覆盖高频题型：数组与哈希、双指针、链表、二叉树、动态规划
- [ ] 练会两道手写题：multi-head attention、最小 ReAct Agent

### PaperLens 收尾（开发已结束，不再加功能）

- [ ] 回代码确认：重排后保留几块、RRF 的 k、BM25 的 k₁ 和 b
- [ ] 回代码确认：怎么识别问题点名的论文、问题分解怎么实现、引用检查核对哪些内容
- [ ] 亲自读一遍复测报告 `report.md`
- [ ] 检查 README、PROGRESS 里有无残留「使用 LangGraph」的说法，有就改掉

### 面试准备

- [ ] 继续 PaperLens 模拟面试：切块方式、响应速度优化、为什么不用 LangGraph、手写最小 ReAct Agent
- [ ] 按复测后的新评测数字再模拟一轮
- [ ] Transformer 完成后，按「学习护栏」和「简历写法与面试准备」里的问题模拟一轮

### 简历

- [ ] 填姓名、邮箱（先改简历文档这个主版本，再导出）
- [ ] 按本规划重写 Transformer 的三条占位：技术栈和消融项要与实际做的一致
- [ ] 填入真实数字后重新导出 Word / PDF

### 长期

- [ ] 期中前后保证课业，GPA 不为实习让步

## 硬件分工与环境配置

正式训练和微调全部放在 RTX 5060 8GB 上，MacBook Air M4 只负责写代码、跑测试和几分钟内结束的小规模调试。8GB 显存是整个计划的主约束，所有模型规模都按它来定。

| 设备 | 用途 | 注意事项 |
| --- | --- | --- |
| Windows + RTX 5060 8GB | 预训练、消融实验、Qwen3-0.6B 的 LoRA 微调 | Blackwell 架构（sm_120），必须装用 CUDA 12.8 及以上构建的 PyTorch 2.7+，否则会报 no kernel image 错误；训练统一用 bf16 autocast |
| MacBook Air M4 | 写代码、单元测试、tiny 配置的 CPU/MPS 冒烟测试 | 无风扇，长时间训练会降频，不跑正式实验 |

环境建议：

- Windows 端用 WSL2（Ubuntu）+ `uv` 管理 Python 3.11/3.12 环境。`torch.compile`、部分训练库在原生 Windows 上支持较差，Codex CLI 和 Claude Code 在 WSL 里也更顺。
- 安装 PyTorch 时从官网安装矩阵选 CUDA 12.8 的命令，不要直接 `pip install torch` 了事。装好后跑这一行确认：`python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); torch.randn(1).cuda()"`
- 两台电脑以 GitHub 为唯一代码来源：Mac 上开发、push，Windows 上 pull 后训练；数据和 checkpoint 不进仓库。
- 实验日志用 TensorBoard + CSV 作为保底；Weights & Biases 体验更好，但国内访问可能不稳。下载 Hugging Face 模型和数据集如遇网络问题，可设置镜像环境变量。
- 显存预算在 W4 第一天实测：Qwen3-0.6B 的 bf16 权重约 1.2GB，配合 gradient checkpointing、序列长度 512、小 batch + 梯度累积，LoRA 训练预计能放进 8GB。

[PyTorch 论坛：sm_120 需 CUDA 12.8+ 构建](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099?page=4)

## 技术路线总览

两条线在权重对齐处汇合：自研架构跑通工业级模型权重。

| 线 | 模块 | 内容 | 产出给谁 |
| --- | --- | --- | --- |
| 从零预训练线（W1–W3） | 最小 GPT | 字符级莎士比亚，手写注意力与训练循环 | → BPE + 基线 |
| | BPE + 基线 | TinyStories 预训练，约 20M 参数 | → 现代组件 |
| | 现代组件 | RoPE · RMSNorm · SwiGLU · GQA | → 消融实验；完整架构 → 权重对齐 |
| | 消融实验 | 4 组必做 + 可选，同 token 预算 | 结果表 → README · 简历 |
| 汇合点（亮点） | 权重对齐 | 载入 Qwen3-0.6B，logits 与 HF 一致 | 作为基座 → GSM8K SFT |
| 微调线（W4–W5） | 手写 LoRA | LoRALinear，与 PEFT 对照 | → GSM8K SFT |
| | GSM8K SFT | Qwen3-0.6B-Base，微调前后准确率 | 准确率 → README · 简历 |
| 终点 | README · 简历 | 结果表与曲线 | 12 月初投递 |

预训练线 W1–W3 做出一个可写进简历的完整结果；微调线 W4–W5 把这套架构用到真实模型上。权重对齐是整个项目的故事核心：它证明你写的架构正确到可以直接运行 Qwen3-0.6B 的官方权重，后面的 LoRA 也跑在你自己的模型代码上。

## 五周时间表

W3 末达到 MVP，之后的工作只影响出彩程度。

| 周 | 日期 | 任务 | 所属线 |
| --- | --- | --- | --- |
| W1 | 10/26–11/1 | M0 环境与仓库；M1 最小 GPT；手写注意力 | 从零预训练线 |
| W2 | 11/2–11/8 | M2 BPE 分词器；TinyStories 基线；训练基建 | 从零预训练线 |
| W3 | 11/9–11/15 | M3 现代组件；A1–A4 消融；README 初版 | 从零预训练线 |
| **MVP 关口 · 11/15** | | A1–A4 结果表可复现，可写进简历 | |
| W4 | 11/16–11/22 | M4 权重对齐；测基座准确率；手写 LoRA | 微调线 |
| W5 | 11/23–11/30 | M5 SFT 与 F 组；M6 README·简历；2–3 天缓冲 | 微调线 |
| **完成 · 11/30** | | 真实数字填入简历，12 月初投递 | |

按每周 15–20 小时估算（课业之外），训练任务尽量放在夜里和上课时跑。如果 W3 末 MVP 关口没过，W4 先补齐 MVP，再按「风险与降级方案」砍可选实验。

## 各里程碑任务与验收标准

每个里程碑都以「测试通过 + 你能讲清楚」为验收，不以「agent 说写完了」为验收。M0–M3 是必须完成的 MVP，M4–M5 是让项目出彩的部分。

### M0 环境与仓库（W1，1–2 天）

1. WSL2 + CUDA 版 PyTorch 装好，GPU 验证命令通过。
2. 建仓库骨架（见「仓库结构」），写好 `AGENTS.md` 和 `CLAUDE.md`。
3. 配好 `pytest`、`ruff`，可选 GitHub Actions 在 CPU 上跑测试。

验收：`pytest` 在 Mac 和 Windows 都通过；一个 tiny 配置能在 CPU 上跑 10 步训练。

#### M0 实施计划（第一批已于 2026-10-26 获用户确认）

用户已确认继续实施第一批，由 Codex 推进；用户决定切换时再由 Claude Code 读取交接。分两批落实，原有 M0 验收标准保留，实际进度见 `PROGRESS.md`。

**第一批：工程基础与设备选择。**

| 文件或目录 | 实现范围 |
| --- | --- |
| `pyproject.toml`、`uv.lock` | 声明 Python 3.11/3.12 与当前阶段的 PyTorch、pytest、ruff 依赖，配置 src 布局；lock 由 uv 生成。具体版本与平台安装来源在实施时核对并记录 |
| `.gitignore` | 排除虚拟环境、缓存、数据、checkpoint、运行日志和本地私有配置 |
| `README.md` | 写明项目阶段、Mac 开发与 Windows 正式训练的分工，以及经过验证的安装和检查命令 |
| `src/tlab/__init__.py` 与各子包的 `__init__.py` | 按既定结构建立可导入的包，不提前实现模型、分词器或训练循环 |
| `src/tlab/device.py` | 用普通函数 `get_device()` 返回 `torch.device`，按 CUDA → MPS → CPU 选择可用设备 |
| `tests/test_import.py`、`tests/test_device.py` | 先写测试，再实现包与设备选择；测试内容见下文 |
| `configs/`、`scripts/`、`experiments/` 等既定目录 | 保留必要的空目录占位；`experiments/results.md` 只建表头，暂不填写实验行。`docs/notes/` 留给用户，不代写笔记 |

执行顺序：

1. 定位已有 uv；当前 shell 找不到时先检查安装位置，确认没有后再安装。建立项目虚拟环境，核对并锁定依赖。
2. 初始化本地 Git，使用 `m0-bootstrap` 作为首个任务分支；首个可运行基线经用户验收后建立 `main`。提交、配置远程和推送按用户授权执行。
3. 先写导入与设备选择测试，确认失败来自尚未实现的功能，再做最小实现。
4. 运行完整 `uv run pytest`、`uv run ruff check .`、`uv run ruff format --check .`，记录命令、结果和平台。
5. 交给另一方审查，再由用户阅读 diff 并验收；Windows 检查由能访问该设备的一方执行并记录。

新增测试覆盖：

- 在没有可用 GPU 的环境中，安装后的 `tlab` 包可以导入，导入不触发模型下载或 GPU 初始化。
- 模拟 CUDA 和 MPS 均可用时优先选 CUDA；仅 MPS 可用时选 MPS；两者均不可用时选 CPU。
- 设备函数返回 `torch.device`。模拟测试只证明选择逻辑，不能替代真实设备上的运行检查。

**第二批：补全原有 M0 验收。**

CPU 训练 10 步依赖 M1 的最小模型与可运行训练路径。相关代码就绪后，通过另行确认的 tiny 配置补做这项检查；涉及用户手写的实现仍由用户完成。Windows/WSL 的真实 CUDA 自检、两台设备的 pytest 与 CPU 10 步训练未全部验证前，M0 保持部分完成，不用占位测试或无关模型替代验收。

第一批不提前安装后期微调依赖，不下载模型或数据，不运行正式实验。后续训练配置与超参数仍按原规则单独确认。

### M1 最小 GPT（W1）

1. 字符级 Tiny Shakespeare 数据。
2. 手写组件：token embedding、可学习位置编码、因果自注意力（不用 `nn.MultiheadAttention`）、MLP、Pre-LN、残差、weight tying。
3. AdamW 训练循环；生成函数支持 temperature 和 top-k。
4. 测试：注意力输出与 `F.scaled_dot_product_attention(is_causal=True)` 一致；因果性测试（改动未来 token 不影响前面位置的输出）；参数量函数与手算一致。

验收：5060 上几分钟内训练出格式像剧本的文本，val loss 稳定下降。

### M2 BPE + TinyStories 基线 + 训练基建（W2）

1. 手写 byte-level BPE（训练、编码、解码），在小语料上与 HF `tokenizers` 的结果对照；正式训练用 `tokenizers` 训练的 8k 词表（速度原因）。
2. TinyStories 预处理成 uint16 memmap 二进制文件。
3. 训练基建：bf16 autocast、梯度累积、warmup + cosine 学习率、梯度裁剪、checkpoint 保存与恢复、定期 eval、日志、固定随机种子、记录 tokens/s 和显存峰值。
4. 基线模型约 20M 参数（例如 6 层、d_model 384、6 头、上下文 512），以 W2 实测吞吐为准调整。

验收：单次训练 1–2 小时内完成，生成连贯英文短故事，loss 曲线和吞吐有记录。

#### M2 实施计划（2026-10-31 用户确认）

原有 M2 任务与验收保留，实施按四个分支依次推进，每个分支从 `main` 分出；实际进度见 `PROGRESS.md`。

| 任务 | 分支 | 内容 | 实现方 |
| --- | --- | --- | --- |
| A 手写 BPE | `m2-bpe` | `src/tlab/tokenizer/bpe.py` 的 `get_stats`、`merge`、`BPETokenizer.train/encode/decode`（字节级、无预分词，并列时取最先出现的一对）；保存加载与对照脚本 | 核心由用户手写；测试、保存加载、对照脚本由 agent 写 |
| B TinyStories 数据 | `m2-data` | 下载 `roneneldan/TinyStories`（约 2 GB，下载前核实文件名与大小，仅在 Windows 上全量下载）；HF `tokenizers` 训练 8k 词表；编码为 uint16 `.bin`；`np.memmap` 读取与 get_batch | agent |
| C 训练基建 | `m2-infra` | warmup + cosine 学习率与 checkpoint 保存恢复（含优化器、步数、随机数状态）由 agent 写；梯度累积、梯度裁剪及在主循环中调用学习率调度需改用户手写的 `trainer.py`，由用户写 | agent + 用户 |
| D 基线训练 | 沿用 `m2-infra` 或单独分支 | 参照 `docs/m1-windows-run.md` 写手册，由 Codex 在 RTX 5060 上执行并回写结果 | Codex 执行，agent 核对 |

已确认的决定：

- 手写 BPE 与 HF BPE 的对照标准：两者都能无损还原；相同词表大小下压缩率（每 token 平均字节数）相差不超过 5%。不要求逐 token 一致（预分词与并列规则不同）。
- 新增依赖 `tokenizers`；TinyStories 只在 Windows 上全量下载，Mac 只用小样本测试。
- 模型大小以实测吞吐为准：计划例子（6 层、d_model 384、6 头、上下文 512、8k 词表）按公式约为 13,969,152 参数，而非 20M；先用该配置在 5060 上短跑测吞吐，再按“1–2 小时内完成”确定层数、宽度与步数，调整前需用户确认。
- 上下文 512 时注意力表每层每份约 96 MB（bf16，batch 32），batch 不足时用梯度累积补足。

任务 B 的决定（2026-11-01 用户确认）：

- 数据用 TinyStories V2（`TinyStoriesV2-GPT4-train.txt` / `-valid.txt`，全部由 GPT-4 生成）；验证集直接用官方 valid 文件，不从训练集切分。
- 正式词表用 HF `tokenizers` 训练：字节级 BPE，先用 GPT-2 正则预分词，总大小 8192（含 `<|endoftext|>`），在完整训练集上训练。编码时每篇故事末尾加 `<|endoftext|>`。
- Mac 只下载 V2 valid（约 21.5 MiB）做端到端冒烟；全量下载与编码写入任务 D 的 Windows 手册，由 Codex 执行。冒烟配置 `configs/tinystories_smoke.yaml` 的超参数与 `tiny_smoke.yaml` 相同，只改 `data_dir` 与 `out_dir`。
- `get_batch` 移到 `src/tlab/data/batch.py`，同时支持张量与 `np.memmap`，起点抽取方式与 M1 相同；`char.py` 保留同名导入，用户手写的 `trainer.py` 不改。

任务 C 的决定（2026-11-01 用户确认）：

- `TrainConfig` 新增 `grad_accum_steps`（默认 1）、`grad_clip`（默认 0，表示不裁剪）、`lr_schedule`（`constant` / `cosine`，默认 `constant`）、`warmup_iters`（默认 0）、`min_lr`（默认 0）、`ckpt_interval`（默认 0，表示只在结束时存）、`resume`（默认 False）。默认值与 M1 行为相同，已有配置文件不改。
- agent 写 `src/tlab/train/schedule.py` 的 `get_lr(step, config)`：前 `warmup_iters` 步线性升到 `learning_rate`；warmup 对 constant 与 cosine 都生效；cosine 在剩余步数内从 `learning_rate` 降到 `min_lr`，循环的最后一步（第 `max_iters - 1` 步）恰为 `min_lr`（2026-11-01 提交前审查后用户确认）。`lr_schedule` 在建配置时检查，拼错立即报错。
- agent 写 utils 中的存档与读档（模型、优化器、下一步的 step、训练用随机数生成器与全局随机数状态；先写临时文件再改名），`RunLogger` 增加 `lr` 列并支持续训时追加；`run.py` 负责读档、把起始步数与生成器交给 `train()`，tokens 统计计入梯度累积。
- 用户在 `trainer.py` 主循环中写：每步调用 `get_lr` 设定学习率、梯度累积（loss 除以累积次数）、梯度裁剪，并把 lr 传给 `logger.log_eval`。用户写完后，经用户同意由 agent 在 `trainer.py` 中加入断点续训的三处外围改动（起始步数、外部传入的生成器、按 `ckpt_interval` 调用存档），并标注“agent 编写”。
- TinyStories 正式训练的超参数与 `tinystories_base.yaml` 留到任务 D，实测吞吐后由用户确认。

### M3 现代组件 + 消融（W3）——MVP 关口

1. 实现 RoPE、RMSNorm、SwiGLU、GQA/MQA、KV cache，全部可通过配置开关切换。
2. 按「实验设计」跑 A1–A4 四组消融。
3. README 初版：结构图、结果表、loss 曲线。

验收：结果表可复现；到这一步项目已经可以写进简历，后面的部分出问题也不影响投递。

#### M3 实施计划（2026-11-02 用户确认）

原有 M3 任务与验收保留；实际进度见 `PROGRESS.md`。

已确认的决定：

- **分工**：用户手写 RMSNorm、SwiGLU、RoPE（`src/tlab/model/layers.py`）与 GQA（`attention.py`）；正弦位置编码、Post-LN、KV cache、配置开关、评测与汇总脚本、消融配置、Codex 手册、README 由 agent 写。改到用户手写文件的部分标注“agent 编写”。
- **基准**：以 M2 基线结构（可学习位置编码、Pre-LN LayerNorm、GELU、MHA）为对照组，每组只换一个组件；对照组各组共用。另跑一组 RoPE + RMSNorm + SwiGLU + GQA 的组合，仅作参考，不算单变量结论。
- **每次训练**：1 亿 token。上下文 512，batch 32 × 累积 2，3052 步（100,007,936 token），学习率 6e-4 按 cosine 降到 6e-5，warmup 92 步（3%），梯度裁剪 1.0，weight decay 0.01，每 250 步评估。
- **种子与判定**：每个变体 2 个种子（1337、2024）。两个变体均值之差大于两者各自种子间差距中的较大者时才算有结论，否则记“差异不显著”；同种子配对比较方向相反时补第 3 个种子。每个变体一个配置文件，命令行只覆盖 `seed` 与 `out_dir`。
- **A4**：GQA-2（参数少 8.4%）、MQA（少 10.6%）直接比较，表中写明参数量。
- **速度与显存**：val loss 取自正式训练；训练吞吐、生成吞吐、KV cache 显存另做短测（空闲降温后测，每项 3 次取中位数，记录温度）。
- **记录**：`experiments/results.md` 每次训练追加一行；脚本生成 `experiments/m3_ablations/SUMMARY.md`。

| 步骤 | 分支 | 内容 |
| --- | --- | --- |
| 0 | `m3-ablate` | 对照组配置 `configs/ablations/a0_control.yaml`、汇总逻辑与脚本、Codex 手册第一版；完成后 Codex 先跑对照组 |
| 1 | `m3-norm` | 用户写 RMSNorm；agent 写 `norm`（layernorm/rmsnorm）与 `norm_position`（pre/post）开关，Post-LN 为 `LN(x + f(x))` 且不用最终 ln_f |
| 2 | `m3-mlp` | 用户写 SwiGLU（隐藏层 1024，与 GELU MLP 参数量相同）；agent 写 `mlp` 开关 |
| 3 | `m3-pos` | 用户写 RoPE；agent 写正弦位置编码、`pos_encoding` 与 `rope_theta` 开关、评估时允许更长序列、长度外推评测脚本 |
| 4 | `m3-attn` | 用户写 GQA；agent 写 `n_kv_heads` 开关、KV cache（含位置偏移）、生成时的缓存开关、速度与显存短测脚本 |
| 5 | — | Codex 每完成一个分支跑一组：对照 2、A1 4（另在 1024 长度上评测外推）、A2 4 + 学习率 3e-3 加测 3（各 1 个种子）、A3 2、A4 4、组合 2，共 21 次；A7 只做短测 |
| 6 | — | 汇总表、每组 loss 曲线叠加图、生成样例、README 初版、M3 自测提醒 |

- 每步测试都包含：默认配置下参数与 tiny_smoke 结果与 main 逐字一致，保证 M1、M2 可复现。
- A2 发散判定：任一 loss 为 NaN；或最终 val loss 高于第 250 步；或 val loss 比此前最低点高 0.5 以上。收敛速度用“首次 val ≤ 2.0 的步数”。
- 变体名（即 `configs/ablations/<变体>.yaml` 的文件名和结果目录名，2026-11-02 第 0 步定下）：`a0_control`；`a1_sinusoidal`、`a1_rope`；`a2_postln`、`a2_rmsnorm`；`a3_swiglu`；`a4_gqa2`、`a4_mqa`；高学习率加测 `a2_postln_lr3e-3`、`a2_preln_lr3e-3`、`a2_rmsnorm_lr3e-3`；组合 `a9_modern`。
- 判定规则的补充（第 0 步提交前自查后明确）：任一种子最终 val 为 NaN 时记“无法比较”；任一方不足 2 个种子时记“种子不足”；补到 3 个种子后不再看配对方向，只按均值规则判定。速度与显存只看短测，汇总表不放长训练的 tokens/s。
- A2 结果之后的决定（2026-11-04 用户回复“接下来的话就按你说的来做吧”确认）：① 补三个 `a2_*_lr3e-3` 的第 2 个种子 2024，因为 AGENTS 要求主结论至少 2 个种子（同日用户决定不等 m3-mlp，先让 Codex 在 main 上跑，称“第 2 批补充”；这三次不依赖 m3-mlp 的代码）；② 汇总表的高学习率加测增加“未收敛”标记（周期评估和最终评估都没到 val ≤ 2.0；已判发散的只写发散原因，不再叠加），既定的发散判定不改；③ A1、A3、A4 继续用学习率 6e-4，保持与对照组可比。
- M3 结束时（第 6 步）把本阶段所有题统一再问、再讲一遍，含 `docs/self-test-qa.md` 第 31–33 题（2026-11-04 用户要求）；此前不逐步追问。
- 第 2 步 `m3-mlp` 的决定（2026-11-04 用户回复“按推荐开始吧”确认）：用户在 `layers.py` 写 `SwiGLU(d_model, hidden_dim)`，层名 `gate_proj`、`up_proj`、`down_proj`，都不带 bias；`mlp` 开关（`gelu`/`swiglu`）由 agent 写，标注“agent 编写”；隐藏层按 `8 * C // 3`（C = 384 时 1024），暂不加配置项；手册第 3 批只有 `a3_swiglu` 2 个种子，显存峰值估计约 5.2 GB。实现为 16 个测试函数（SwiGLU 4 个、开关 9 个、提交前自查补 3 个），配置测试在已有参数化里各加 1 项。
- 第 3 步 `m3-pos` 的决定（2026-11-04 用户回复“按推荐开始”确认）：
  - 用户在 `layers.py` 写 `RotaryEmbedding(head_dim, theta=10000.0)`，作用于拆头后的 q、k，形状 (B, h, T, d)，没有参数；采用 HF Qwen3 的 `rotate_half` 写法（对半分、前半段与后半段配对），M4 才能对上（2026-11-04 已核对 HF 源码）。
  - 注意力接入、`sinusoidal` 分支、`pos_encoding`（`learned`/`sinusoidal`/`rope`）与 `rope_theta` 开关由 agent 写，标注“agent 编写”。`learned` 为默认，超过 `block_size` 仍报错；`sinusoidal`、`rope` 可超过。
  - `rope_theta` 在 A1 用 10000；Qwen3-0.6B 的配置是 1,000,000（已核对），留给 M4。
  - `sinusoidal` 时 token 向量乘 √C（原论文做法）：正弦向量均方根 0.707，token 向量初始标准差 0.02，不缩放时位置会盖过内容。小实验（字符级，4 层 C=128，300 步，种子 0、1）：可学习 2.357 / 2.461，正弦不缩放 3.350 / 3.043，正弦乘 √C 2.302 / 2.309（`runs/m3-pos/01-`）。只影响这一组。
  - 外推：只测 `sinusoidal`、`rope`，用 1024 长度的窗口打分，分别报前半段（位置 0–511）和后半段（512–1023）的 loss；`learned` 明确报不支持。
  - 参数量：`a1_sinusoidal`、`a1_rope` 都是 13,772,544。手册第 4 批：两个变体各 2 个种子，共 4 次训练加 4 次外推评测。
  - 测试接口 17 个（RoPE 6 个、注意力 3 个、开关 5 个、外推与配置 3 个），实现为 28 个测试函数；提交前自查后又补 7 个：续训时结构开关必须一致（2 个）、结果与配置文件核对（3 个）、外推的奇数长度与数据不足、正弦的奇数维度。
  - 自查后增加的两道检查（RoPE 和正弦参数量相同，只看数字发现不了跑错结构）：`resume=True` 时结构开关与存档不同就报错；`summarize_ablations.py --configs configs/ablations` 把每次结果与变体的配置文件核对。
  - 外推的读法：后半段的上下文也更长，“后半段 − 前半段”同时包含位置超出训练范围和能看的内容变多两个影响，不能全算到其中一个头上。
- 第 4 步 `m3-attn` 的决定（2026-11-05 用户回复“好的， 那我们接下来开始吧”确认）：
  - 用户在 `attention.py` 写 GQA：`CausalSelfAttention(d_model, n_heads, rope_theta=None, n_kv_heads=None)`，`None` 表示等于 `n_heads`（即 MHA）；`k_proj`、`v_proj` 是 C → n_kv·d，`q_proj`、`out_proj` 不变。每组复制给相邻的头（`repeat_interleave`，k0 k0 k0 k1 k1 k1），2026-11-05 已核对 HF `modeling_qwen3.py` 的 `repeat_kv` 与此等价，M4 才能对上。
  - RoPE 偏移由用户改（原分工写 agent 负责，因 `RotaryEmbedding` 是用户手写的类，只改两行，用户同意后改由用户写）：`forward(x, offset=0)`。
  - KV cache 由 agent 写，标注“agent 编写”：每层一个 `KVCache`，存旋转后、复制前的 k、v，形状 (B, n_kv, T, d)，顺序与 HF 相同（先旋转，再存缓存，最后复制）；`model(idx, cache=None)` 返回值不变，训练路径不受影响；掩码 `diagonal=T_past + 1`。
  - 生成超过 `block_size`：窗口满后每一步丢掉缓存，用最后 `block_size` 个 token 重建。此时不再提速，但结果与不用缓存相同；让缓存继续变长会破坏一致性，还会用到训练时没见过的距离。`generate(..., use_cache=False)` 默认关，`sample.py` 加 `--cache`。
  - `n_kv_heads` 开关由 agent 写（`int | None`，默认 `None`）；`load_config` 现在直接用字段类型转换，遇到这种类型要小改。
  - 配置与手算参数量：`a4_gqa2`（`n_kv_heads: 2`）12,789,504，少 8.44%；`a4_mqa`（`n_kv_heads: 1`）12,494,592，少 10.56%；`a9_modern`（RoPE、RMSNorm、SwiGLU、GQA-2，`rope_theta` 10000，只作参考）12,587,904。
  - 完整缓存手算（B=1、T=512、bf16，6 层）：层数 × n_kv × T × d × (k 的字节数 + v 的字节数)，MHA 4.5 MiB，GQA-2 1.5 MiB，MQA 0.75 MiB。RoPE 的 cos、sin 是 fp32，bf16 的 k 旋转后被提升成 fp32，v 仍是 bf16（`q @ k` 又回到 bf16），所以 `a1_rope` 是 6.75 MiB，`a9_modern` 是 2.25 MiB（2026-11-06 提交前审查发现，CPU bf16 autocast 实测）。不改这一点：改了会让第 4 批的 RoPE 结果无法复现，节省显存属于性能优化，单独成任务。
  - 短测脚本 `scripts/bench_attention.py`：训练吞吐（和正式训练一样累积、裁剪、更新，token 随机生成）、生成吞吐（有无缓存，开头 32 + 新生成 480，正好存满窗口）、缓存显存（`generate_cache` 的显存峰值，另记完整缓存的字节数和类型），每项 3 次取中位数，每次测量都记录温度，每个配置开始前空闲 60 秒；用随机权重，不需要存档；跑对照、各单变量变体和 `a9_modern`。
  - 手册第 5 批：`a4_gqa2`、`a4_mqa`、`a9_modern` 各 2 个种子，共 6 次训练，都带完整验证打分，外加短测；短测在 `results.md` 追加一行 `M3-bench`。
  - 提交前审查后的决定（2026-11-06）：整数配置字段不再接受 2.9、true（`int()` 会悄悄截断，等于换了结构）；`n_layers=0` 时不支持 `use_cache`（没有注意力，也没有 k、v 可存）；短测参数走命令行不走 YAML，与 `eval_extrapolation.py` 等评测脚本相同，它们不是训练超参数。
  - 测试接口 20 个（GQA 6 个、RoPE 偏移 1 个、缓存 5 个、生成 4 个、开关配置与脚本 4 个）。
- 第 5 批短测之后的决定（2026-11-06 用户回复“按推荐开始”确认）：
  - 原因：第 5 批短测在 RTX 5060、B=1 时，缓存提速大多是 0.98–1.05 倍。每步耗时大致跟操作数走（13.4 µs × kernel 数 + 1.17 ms，R² = 0.80），而缓存几乎不减少操作数；Mac 的 CPU、MPS 上同样的代码明显提速，batch 越大越明显。这是推断，这一批用来验证。
  - 分支 `m3-bench-batch`，全部由 agent 写：短测脚本加 `--generate-batch-size`，只影响生成吞吐和生成的显存峰值，完整缓存字节数仍按 B=1 算；已有的 `bench.json` 不动。
  - 手册“第 5 批补充”：`a0_control`、`a4_gqa2` 在 RTX 5060 上 B=1、8、32，每项 5 次；`a0_control` 在同一台机器的 CPU 上 B=1，每项 3 次。结果写到 `experiments/m3_ablations/bench_batch/`。
  - 每项 5 次只用于这组 GPU 加测（上次 B=1 的噪声大：同一项三次是 146、308、223），与“每项 3 次”不同；训练部分只跑 5 步，这次不看训练吞吐。
  - 测试接口 4 个，实现为 5 个新测试函数。改动只有十几行且不碰模型，不开两个审查子 agent，用测试、故意改错和干净导出检查。
- 第 6 步 `m3-report` 的决定（2026-11-06 用户回复“按推荐开始”确认）：
  - 曲线：`scripts/plot_ablations.py`，每组一张 PNG（`a1`、`a2`、`a2_lr`、`a3`、`a4`、`a9`），和对照叠在一起，左边整条曲线，右边放大第 2000 步之后（末端差别只有 0.01–0.05，整条曲线上看不出来），写到 `experiments/m3_ablations/figures/`。用 matplotlib，放在 `dev` 依赖，只在画图时才导入。配色取自 dataviz 技能的参考色板前 3 色；这台 Mac 没有 Node，校验脚本没有运行，依据是色板文档写明前 3 色通过两两校验。
  - 汇总：`SUMMARY.md` 末尾加“速度与显存短测”和“A7 KV 缓存的生成速度”两节，原有内容逐字不变。
  - 样例：`scripts/sample_ablations.py`，全部 12 个变体，每个用种子 1337 的最终存档，参数沿用 M2。存档只在 WSL 上，由用户直接运行（Codex 额度将尽）；手册“第 6 步”。样例回来后再写进 README，并再提交一次。
  - README 用中文，结构图用 Mermaid（GitHub 渲染，可以用中文）；曲线图是 PNG，上面的文字用英文。`tests/test_readme.py` 核对磁盘上的 SUMMARY.md 由原始结果生成，README 的分数、均值、外推、速度和链接与原始结果一致。
  - GQA-2 和 MQA 之间：两个种子方向相反（+0.0041、-0.0060），按判定规则要补第 3 个种子才能比较。计划里的 A4 结论是和对照比（配对方向一致，不用补），所以这里记为“没有结论”，补不补由用户决定。
  - MVP 关口：README 合并后由用户确认是否勾选，agent 不自己勾。
- MVP 关口（2026-11-07 用户回复“可以勾掉MVP关口了”确认）：结果表、曲线、样例、README 初版都已合并。两点如实记录：“可复现”验证的是由原始结果重新生成表，没有重新训练一次来比较数字；M3 清单对应的第 34、35、36、38、44、45、50 题，第一轮还没有全部讲出来，用户决定进入 M4，这些题之后自己回顾（`docs/self-test-qa.md`）。
- 本次不做：A5、A6（可选，MVP 后再定），性能优化（单独成任务）。
- 第 1 步 `m3-norm` 的决定（2026-11-03 用户回复“按推荐开始吧”确认）：
  - 用户在新建的 `src/tlab/model/layers.py` 手写 `RMSNorm(d_model, eps=1e-6)`：只有名为 `weight` 的 (C,) 参数，初始全 1，没有 bias；eps 加在根号里，默认 1e-6 与 Qwen3 一致，LayerNorm 保持 PyTorch 默认 1e-5。内部暂不转 fp32（训练时残差主干本就是 fp32），M4 再定。
  - 开关由 agent 写进用户手写的 `block.py`、`gpt.py` 与 `layers.py`（`make_norm`），逐处标注“agent 编写”。Post-LN 下 `ln_f` 换成 `nn.Identity()`，不产生参数。
  - 高学习率加测：`learning_rate` 3e-3、`min_lr` 3e-4（保持降到峰值的 1/10），warmup 仍为 92 步，其余与对照组相同。
  - 测试接口 14 个（RMSNorm 5 个、Block 4 个、GPT 与配置 4 个、消融配置 1 个），2026-11-02 列给用户确认。实现为 20 个测试函数：接口 11、13 各拆成两个；提交前自查后又补 3 个（汇总时旧种子缺新字段按默认值比较、GPTConfig 建配置时检查取值、每个模型开关都能在训练配置里设置）。
- 第 2 批之前加“完整验证集评估”（2026-11-03 用户选择“加，单独分支”，随后对计划回复“提交并按计划开始”）：
  - 原因：训练中的评估只按种子随机抽 20 个 batch（约占验证集 6%），两个种子抽到的子集不同，抽样误差约 0.02，与对照组种子间差距 0.0175 同一量级。
  - 分支 `m3-evalfull`，从 m3-norm（0e65b62）分出，全部由 agent 写，不出自测题。
  - 做法：验证集切成首尾相接、互不重叠的窗口（长度 = 存档的 block_size，最后不满一窗的部分单独算），每个 token 恰好预测一次，按 token 平均；GPU 上与训练相同用 bf16。`src/tlab/eval/full_val.py` 的 `evaluate_full`、`evaluate_checkpoint`，脚本 `scripts/eval_full_val.py` 写 `full_val.json`。
  - 汇总：每个种子目录有 `full_val.json` 就读入；对照与变体的每个种子都有时用完整分数判定，缺一个则这一行两边都用 run.json 的抽样估计；表中加“依据”列。
  - 第 2 批：先给对照组两个存档补评（不重新训练），之后每次训练完都打分。
  - 测试接口 6 个：每个 token 恰好算一次（手算 50·ln2/22）、batch 大小与数据格式不影响结果、评估时 eval 模式且之后恢复、能给真实存档打分（token 数 = 验证集长度 − 1）、汇总优先用完整分数、缺一个就整行退回。
  - 实现为 7 个测试函数（接口 4 拆成函数与脚本两个）；提交前自查后又补 3 个：任一种子的抽样估计或完整分数为 NaN 都记“无法比较”、full_val.json 的 ckpt_step 不等于 max_iters 时报错、退回抽样估计的行写出对照的抽样均值；脚本同时打印存档步数。
  - 高学习率加测的例外（2026-11-03 用户回复“同意并提交”确认）：出现 NaN 或发散时不中断、不调参，跑完 3052 步并照常打分、记录，结论如实写“发散”或“未发散”；程序报错退出仍按 AGENTS 规则 6 停下汇报。只适用于 `a2_*_lr3e-3` 三个配置，其余训练照常遵守规则 6。

### M4 自研架构载入 Qwen3-0.6B（W4 前半）

1. 以 HF 的 `config.json` 为准，逐项补齐 Qwen3 的细节：QK-Norm、GQA 头数、RoPE 的 theta、tied embeddings 等。
2. 写权重映射脚本：HF 参数名 → 自己的模块名，处理转置和合并/拆分。
3. 写对齐测试：同一输入下逐层对比 hidden states，最后对比 logits。

验收：fp32 下 logits 与 `transformers` 输出的最大误差在 1e-4 量级；greedy 生成结果一致。卡住超过 3 天就走降级方案。

#### M4 实施计划（2026-11-07 用户回复“按推荐开始”确认）

已核对 HF 上的 `config.json`、`model.safetensors` 文件头和 `modeling_qwen3.py`：28 层，C=1024，16 头，8 组 k、v，`head_dim` 128（不等于 C/h = 64），`intermediate_size` 3072，`rope_theta` 1,000,000，`rms_norm_eps` 1e-6，词表 151,936，`tie_word_embeddings` true，q、k 在拆头之后、旋转之前各做一次 RMSNorm，v 不做。

- **参数量手算：** 每层 15,730,944，28 层 440,466,432，加词表 155,582,464、`ln_f` 1,024，共 596,049,920。文件里存了 751,632,384，因为 `lm_head.weight` 单独存了一份。
- **分工：** 用户在 `attention.py` 写 `head_dim` 和 QK-Norm；`head_dim`、`qk_norm`、`mlp_hidden_dim` 三个开关和转换、对齐、配置都由 agent 写，改到用户手写文件的地方标注“agent 编写”。所有开关的默认值和 M1–M3 相同。
- **分支：** `m4-arch`（结构）、`m4-convert`（转换和对齐）。
- **依赖：** `safetensors` 放主依赖；`transformers` 只放 `dev`，只用于测试和对照。
- **真实权重：** 用户同意下载 `model.safetensors`（来自 `huggingface.co/Qwen/Qwen3-0.6B`，1,503,300,328 字节）、`config.json` 和分词器文件，放在 HF 默认缓存目录，不进仓库。对齐检查在 Mac 的 CPU 上跑，结果写到 `experiments/m4_qwen3/`，`results.md` 追加一行。
- **`RMSNorm` 不改：** HF 在里面先转成 fp32，我们没有。fp32 下对齐不受影响，bf16 下的差别留到 M5 训练时再看。
- **配置：** 不单独写 YAML，直接读模型目录的 `config.json`。`rms_norm_eps`、`attention_bias`、`attention_dropout`、`hidden_act`、`tie_word_embeddings`、`use_sliding_window`、`rope_scaling` 这些和我们的结构不符时就报错，否则权重能载入、结果却对不上。
- **验收门槛：** 只看分数和贪心生成，分数最大误差小于 1e-3（即 1e-4 量级）。隐藏状态的数值可能上千，只报告绝对误差和相对误差，不算失败。
- **误差的来源（2026-11-07 实测）：** 真实权重上分数最大误差 1.73e-04。在副本里把注意力的 `scores / d**0.5` 改成官方的 `scores * d**-0.5`，三个提示每一层的误差都是 0.0，所以误差只来自这一处的浮点舍入，结构和权重映射没有差别。不改这一行：它是用户手写的代码，改了 M1–M3 的结果就无法逐字复现，而误差已经满足验收。
- **测试接口 14 个：** 不传 `head_dim` 时和现在逐位相同；形状；和手工算的 SDPA 数值一致；QK-Norm 的位置；开了之后分批算和一次算完一致；开关的读取、校验、旧存档、续训；`mlp_hidden_dim`；0.6B 配置的参数量；权重映射；`config.json`；`lm_head` 不相同时报错；小型随机 Qwen3 逐层对比；贪心生成和 `transformers` 相同，有无缓存都相同；检查脚本。

### M5 手写 LoRA + SFT（W4 后半–W5 前半）

1. 手写 `LoRALinear`：A 随机初始化、B 零初始化、缩放 alpha/r、merge 与 unmerge。
2. SFT 数据管线：prompt 模板、只对答案部分算 loss（prompt 位置的 label 设为 -100）、padding。
3. 任务：GSM8K，基座 Qwen3-0.6B-Base。评测：固定 500 题测试子集，greedy 解码，抽取最终数值算准确率。
4. 先测基座准确率（W4 第一天就测，确认有提升空间），再在自研模型上做 LoRA SFT；用 HF + PEFT 同超参跑对照组。

验收：微调前后准确率对比、可训练参数占比、显存和速度；自研 LoRA 与 PEFT 的 loss 曲线基本重合。

#### M5 实施计划（2026-11-07 用户回复“可以按推荐开始”确认）

已核对（2026-11-07，从 HF 读配置和文件大小，权重还没下载）：`Qwen/Qwen3-0.6B-Base` 结构和 M4 相同（28 层，C=1024，16 头，8 组 k、v，`head_dim` 128，H=3072，`rope_theta` 1,000,000，词表 151,936），`model.safetensors` 1,192,135,096 字节，`generation_config` 是贪心，eos 是 151643。`openai/gsm8k` 的 `main`：训练 7,473 条，测试 1,319 条，答案里有 `<<16-3-4=9>>` 这样的计算标注，最后一行是 `#### 18`。

**分工：** 用户手写 `LoRALinear`（`src/tlab/finetune/lora.py`，文件顶部写 `# USER-WRITTEN`）。其余由 agent 写：评测、数据管线、把 LoRA 装进模型、SFT 训练循环、PEFT 对照、手册。`peft` 和 `transformers` 只能出现在 `scripts/`、`tests/`。

| 步骤 | 分支 | 内容 |
| --- | --- | --- |
| 1 | `m5-baseline` | GSM8K 评测和基座准确率 |
| 2 | `m5-lora` | 用户写 `LoRALinear`；agent 写 `apply_lora` 和只存 LoRA 权重的存档 |
| 3 | `m5-sft` | SFT 数据管线和训练循环，配置，显存短测 |
| 4 | `m5-peft` | HF + PEFT 对照脚本（F4） |
| 5 | — | WSL 上跑 F1、F2、F4；汇总、README、M5 复习 |

**第 1 步 `m5-baseline`（计划要求最先做，确认有提升空间）：**

- `scripts/prepare_gsm8k.py` 下载数据到 `data/gsm8k/`（不进仓库）；评测用的 500 题下标由固定种子选出，写进 `configs/finetune/gsm8k_eval_subset.json` 并提交，以后都用这 500 题。
- `src/tlab/eval/gsm8k.py`：提示模板、答案抽取、准确率。基座常常答完不停、接着编下一题，所以先在第一处续写截断（换行后跟 `Question:`、`[Question]` 或 `Q:`），再依次看：第一个 `####` 后面的数；第一个 `answer is` 后面紧跟的数；最后一个数；去掉逗号和 `$`。截断是生成完在文本上做的，速度不够再单独优化。结果里留着原始生成文本，以后换规则可以重新打分，不用重跑。
- **规则的依据（2026-11-07，Mac CPU，dirty，只是试跑）：** 确认计划时的规则只认 `\nQuestion:` 和 `####`，而基座不给例子时常写 `The answer is 55.\n[Question]...`，答对的题也被判错。同样 40 题的原始文本，旧规则 5/40，新规则 22/40；40 题的标准误约 8 个百分点，这不是结论，正式数字看 WSL 的 500 题。few-shot 的 5 题都按例子写了 `####`，两种规则结果相同。
- **贪心：** `top_k=1` 原来也走 `multinomial`，分数并列时会随机选，bf16 下分数很粗，真的可能并列。现在 `top_k=1` 直接取 argmax，和 `transformers` 相同。抽样和 fp32 的贪心结果都和 main 逐位相同（2026-11-07 实测）。
- `generate` 加 `eos_token_id`，默认 `None`，行为不变；SFT 之后的模型靠 eos 停下。
- 贪心解码，KV 缓存，最多 256 个新 token，一次一题。模型没有 attention mask，不同长度的题不能直接拼 batch；一次一题最简单，速度先实测。
- F1 的基座要测两种：few-shot（4 个固定的训练集例子）和不给例子（和 SFT 后的格式相同）。
- `scripts/check_qwen3.py` 加 `--repo`，先确认 Base 权重也能对齐，并在 GPU 上看 bf16 下和 `transformers` 差多少。这决定 M4 留下的问题：`RMSNorm` 要不要像官方一样在内部转 fp32。改它要用户同意，而且会影响 `a2_rmsnorm`、`a9_modern` 的逐字复现；如果要改，做成默认关闭的开关，M3 才能逐字复现。
- **在哪跑：** Mac 只跑单元测试、fp32 对齐和几道题的冒烟；500 题的准确率和 bf16 对比放 WSL。
- 如果基座 few-shot 准确率已经很高，按“风险与降级方案”如实报告，再决定换不换任务。

**手册：** `docs/m5-windows-run.md` 的第 1 批。

**第 1 批的结果和决定（2026-11-08 用户回复“按推荐开始”确认）：** 基座 few-shot 46.8%、不给例子 49.8%（各 500 题，GPU，bf16），继续用 GSM8K。GPU bf16 下和 `transformers` 的分数最大误差 0.656；副本实验发现差别来自 `RMSNorm`，于是加开关 `rmsnorm_fp32`（分支 `m5-norm32`）：默认关闭，M1–M3 逐字复现；载入 Qwen3 时打开。`layers.py` 里的几行由用户写，开关的其余部分由 agent 写。打开后真实 Base 在 Mac CPU 的 bf16 下，3 个短提示每层误差都是 0.0；审查实测 83 个 token 时注意力内部还有 bf16 舍入差别，原因没查。autocast 下残差主干是 fp32，所以这个开关实际影响的是 q_norm、k_norm；如果以后把基座转成 bf16 省显存，所有 RMSNorm 都会受影响。GPU 上的 bf16 对比和 F1 基座重测，和 SFT 之后的评测放在同一批 WSL 上跑，第 1 批的结果保留。

**第 2 步 `m5-lora`：**

- 用户写 `LoRALinear(base, r, alpha)`：A 形状 (r, in)，kaiming uniform 初始化（和 PEFT 相同）；B 形状 (out, r)，全 0；缩放 alpha/r；基座冻结；`merge()`、`unmerge()`。输出 `base(x) + (alpha/r) · x Aᵀ Bᵀ`。
- 测试接口：B 全 0 时输出和基座逐位相同；只有 A、B 能训练；只有 A、B 拿到梯度；merge 后输出和未 merge 一致（容差内）；unmerge 后恢复；参数量等于手算值。
- agent 写 `apply_lora(model, r, alpha, targets)`：默认替换每层的 `q_proj`、`k_proj`、`v_proj`、`out_proj`；词表和其他层冻结；存档只存 A、B。
- 可训练参数手算（默认目标）：每层 (1024+2048)·r × 2 + (1024+1024)·r × 2 = 10,240·r，28 层 286,720·r。r=4、8、16、32 分别是 1,146,880、2,293,760、4,587,520、9,175,040 个，占 596,049,920 的 0.19%、0.38%、0.77%、1.54%。

- **实现（2026-11-08，分支 `m5-lora`）：** 用户写的 `LoRALinear` 在 `src/tlab/finetune/lora.py`，只放这一个类；`apply_lora`、`save_lora`、`load_lora`、`merge_lora` 在 agent 写的 `src/tlab/finetune/lora_model.py`。A、B 先在 CPU 上随机初始化，再搬到 W 所在的设备，类型保持 fp32；基座转 bf16 时要开 autocast。合并要求 W 是 fp32，否则报错：bf16 下 ΔW 会被舍掉一部分（审查实测丢掉 3–31%）。存档只有 A、B 和 r、alpha、targets，载入出错时模型保持原样。真实 Base 上 r=8 时可训练参数 2,293,760（0.385%），和手算一致；装上后 fp32 和 bf16 下的分数都和原模型逐位相同（审查实测）。
- **给第 4 步：** A、B 的形状和 PEFT 的 `lora_A`、`lora_B` 权重相同，不用转置。审查凭记忆说 PEFT 里同一个种子抽出的 A 不同（没装 peft，没实测），所以仍按计划把初始值拷过去。

**第 3 步 `m5-sft`：**

- 模板 `Question: {q}\nAnswer: {a}<eos>`，去掉 `<<...>>` 标注。提示和 padding 位置的 label 都是 -100（`F.cross_entropy` 默认忽略 -100）。
- 在右边 padding：因果掩码下，右边的 pad 影响不到前面的 token，所以不需要 attention mask。测试要证明带 pad 和不带 pad 时真实 token 的分数相同。
- 训练循环写在 `src/tlab/finetune/sft.py`，因为 `trainer.py` 的 `train` 是按随机窗口取数的语言模型训练，形状不同；bf16 autocast、梯度累积、裁剪、cosine、日志和存档的写法照它来。
- 从训练集留 200 条看 val loss，测试集的 500 题只用于最后的准确率。
- 建议超参数（要先做短测，再由用户确认）：r=8，alpha=16，学习率 2e-4，weight decay 0，batch 4 × 累积 4，序列最长 512，1 个 epoch。
- 显存：参数是 fp32 时基座约 2.4 GB；不够再把基座转 bf16（约 1.2 GB），注意不要把 RoPE 的 `inv_freq` 一起转掉；还不够才加 gradient checkpointing。

**第 4 步 `m5-peft`：** `scripts/train_peft.py`，同样的数据顺序和超参数。LoRA 的初始值从自研模型拷过去，否则两边随机数不同，曲线本来就不会重合。验收是 loss 曲线基本重合；“基本重合”的数字门槛在正式跑之前，根据短测结果定下来，并且要先决定 `RMSNorm`，否则 bf16 下的差别会混进来。

**第 5 步，实验（WSL + RTX 5060）：** F1；F2（r=4、8、16、32，各 2 个种子，共 8 次）；F4（r=8，1 个种子）。F3、F5 可选，时间不够先砍。F2 沿用 M3 的判定规则：均值之差大于种子间差距才算有结论。500 题的标准误在准确率 50% 附近约 2.2 个百分点（√(0.5·0.5/500)），报告时一起写。手册写到 `docs/m5-windows-run.md`；Codex 额度不够时由用户直接运行。

**验收：** 微调前后准确率对比、可训练参数占比、显存和速度；自研 LoRA 与 PEFT 的 loss 曲线基本重合；`results.md` 和 README 更新；M5 清单的题统一复习一遍。

### M6 收尾（W5）

1. README 完整版、实验报告、代码清理。
2. 简历条目填入真实数字，面试问答清单过一遍。
3. 预留 2–3 天缓冲。

## 实验设计

所有对比实验遵守同一规则：同一 token 预算、同一超参、同一随机种子，只改一个变量；主结论至少跑 2 个种子。单次预训练暂定约 100M tokens，按 W2 实测吞吐调整到每次 30–60 分钟。

### 预训练消融（A 组）

| 编号 | 对比 | 观察指标 | 面试能讲的点 | 优先级 |
| --- | --- | --- | --- | --- |
| A1 | 可学习位置编码 vs 正弦 vs RoPE | val loss；在 2 倍训练长度上评测的外推表现 | RoPE 为什么编码相对位置 | 必做 |
| A2 | Post-LN vs Pre-LN vs Pre-RMSNorm | 收敛速度；调高学习率后是否发散 | 现代 LLM 为什么用 Pre-Norm | 必做 |
| A3 | GELU MLP vs SwiGLU（隐藏层取 8/3·d，参数量对齐） | val loss | 如何做公平比较 | 必做 |
| A4 | MHA vs GQA（2 组 KV）vs MQA | val loss；KV cache 显存；生成 tokens/s | 推理显存与质量的权衡 | 必做 |
| A5 | 有无 warmup；cosine vs 常数学习率；有无梯度裁剪 | loss 曲线、尖刺 | 训练稳定性 | 可选 |
| A6 | 3–4 个模型大小，同 token 预算 | 参数量 vs loss | scaling law 直觉 | 可选 |
| A7 | 有无 KV cache 的生成 | tokens/s | KV cache 原理 | 必做（很便宜） |

### 微调实验（F 组）

| 编号 | 对比 | 观察指标 | 优先级 |
| --- | --- | --- | --- |
| F1 | 基座 few-shot vs LoRA SFT 后 | GSM8K 子集准确率 | 必做 |
| F2 | LoRA rank 4 / 8 / 16 / 32 | 准确率、可训练参数量、显存 | 必做 |
| F3 | 只加 q、v vs 全部 attention vs attention + MLP | 准确率 | 可选 |
| F4 | 自研 LoRA vs PEFT，同超参 | loss 曲线是否重合 | 必做 |
| F5 | 全参微调 vs LoRA | 准确率、显存 | 可选，8GB 很紧，可能需租卡 |

每个实验对应 `configs/` 下一个配置文件，跑完在 `experiments/results.md` 记一行：日期、配置名、commit、关键指标、一句结论。简历和 README 里只写真实跑出的数字。

## Codex 与 Claude Code 的协作方式

用「一个实现、另一个审查」的交叉方式：每个任务由一个 agent 写，另一个 agent review，你读完 diff、能讲清每一处改动再合并。这样既能利用两个工具，又能避免单个 agent 自说自话。

双方通过根目录 `PROGRESS.md` 共享当前状态与交接记录，按 `AGENTS.md` 的「交接与协作」执行；原对话按需补查，不要求每次加载全部历史。

### 规范文件

- `AGENTS.md` 作为唯一主规范（Codex 读这个文件）；`CLAUDE.md` 里只写一行 `@AGENTS.md` 引用它，两边规则保持一致。
- `AGENTS.md` 必须写明的规则：
    - 项目以学习为目的，核心模块禁止调用 `nn.Transformer`、`nn.MultiheadAttention` 或 `transformers` 的模型类；`transformers` / `peft` 只用于加载权重和做对照实验。
    - 每个模块写张量形状注释，每个新功能配单元测试，提交前 `pytest` 必须通过。
    - 所有实验通过配置文件运行；不得擅自改超参数、不得改写 `experiments/results.md` 里的已有结果。
    - 标记为「用户手写」的文件（见「学习护栏」）只能提建议和写测试，不能直接改实现。

### 每个任务的流程

1. 一个任务 = 一个模块 + 它的测试 = 一个分支。
2. 先让 agent 只讲计划、不写代码，你确认。
3. 先写测试，再写实现。
4. 换另一个 agent 做 review。
5. 你读 diff，把不懂的地方让 agent 讲到懂，再合并。

### 提示词模板

实现任务：

```markdown
任务：实现 src/tlab/model/attention.py 中的 GQA 注意力，支持 n_kv_heads 配置和 KV cache。
要求：遵守 AGENTS.md；先列出实现计划和需要新增的测试，等我确认后再写代码；
每个张量标注形状；新增测试覆盖 n_kv_heads = n_heads 时与 MHA 输出一致、因果性、cache 与无 cache 生成结果一致。
完成后运行 pytest 并汇报结果。
```

讲解：

```markdown
逐段解释 src/tlab/model/attention.py。对每一步说明张量形状如何变化、为什么这样写、
如果换一种写法会出什么问题。最后给我 3 个面试官可能会追问的问题，先不要给答案。
```

Review：

```markdown
请 review 当前分支相对 main 的改动。检查：是否违反 AGENTS.md；数值正确性和边界情况；
测试是否真正覆盖了行为；是否有不必要的复杂度。按严重程度列出问题，不要直接修改代码。
```

跑实验：

```markdown
用 configs/ablations/a1_rope.yaml 训练，跑完把验证 loss、tokens/s、显存峰值按 experiments/results.md 的格式追加一行。
不要修改配置里的任何超参数；如果训练报错或 loss 出现 NaN，停下来汇报，不要自行调整。
```

## 学习护栏

这个项目的价值在于面试时你能讲清楚，所以 4 个核心模块由你亲手敲，其余交给 agent。每个都在 100 行以内：可以让 agent 讲解、可以看参考资料，但键盘在你手里。

| 用户手写的文件 | 里程碑 | agent 能做的 |
| --- | --- | --- |
| 因果自注意力 `attention.py`（MHA 版本） | M1 | 讲解、写测试、review |
| Transformer block `block.py` | M1 | 讲解、写测试、review |
| 训练循环 `trainer.py` 的主循环部分 | M2 | 讲解、补日志和 checkpoint 等外围代码 |
| `LoRALinear` | M5 | 讲解、写测试、review |

你的数学背景是优势：每个里程碑在 `docs/notes/` 写一篇自己的推导笔记，比如注意力的公式、softmax 的梯度、RoPE 的旋转矩阵。

```latex
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
```

```latex
h = W_0x + \frac{\alpha}{r}BAx,\quad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ B_{\text{init}}=0
```

每个里程碑结束时，不看代码自测下面的问题，讲不出来就先不进入下一阶段：

M1

- [ ] 手推注意力每一步的张量形状，(B, T, C) 到 (B, h, T, d) 再回来
- [ ] 为什么除以 √d_k
- [ ] 因果 mask 怎么实现，为什么填 -inf
- [ ] 残差连接和 LayerNorm 各起什么作用
- [ ] 模型参数量怎么估算（约 12·L·d²）
- [ ] weight tying 是什么，省了什么

M2

- [ ] BPE 的训练和编码过程
- [ ] 交叉熵 loss 和 perplexity 的关系
- [ ] AdamW 和 Adam 的区别；为什么需要 warmup
- [ ] bf16 和 fp16 的区别；梯度累积等价于什么

M3

- [ ] RoPE 的公式，以及它为什么编码相对位置
- [ ] RMSNorm 比 LayerNorm 少了什么
- [ ] SwiGLU 的公式和参数量对齐
- [ ] GQA 如何减小 KV cache；KV cache 显存怎么算
- [ ] 训练计算量约 6·N·D 是怎么来的

M4

- [ ] 权重映射中遇到的具体问题（转置、QK-Norm、RoPE 实现差异）和你怎么定位的

M5

- [ ] 为什么 B 初始化为 0；alpha/r 的作用
- [ ] LoRA 省下的是哪部分显存
- [ ] SFT 为什么要 mask 掉 prompt 部分的 loss
- [ ] LoRA、全参微调、QLoRA 的区别
- [ ] RLHF 的三个阶段（SFT、奖励模型、PPO），和 SFT 是什么关系（2026-11-07 用户确认：不做 RLHF 实验，只在 M5 复习时讲清概念）

## 仓库结构与工程规范

仓库按「模型 / 数据 / 训练 / 微调 / 转换 / 评测」分包，配置与代码分离，M0 一开始就把这个骨架建好，之后 agent 往里填。

```text
transformer-lab/
├── AGENTS.md              # 主规范，Codex 读取
├── CLAUDE.md              # 一行：@AGENTS.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── shakespeare_char.yaml
│   ├── tinystories_base.yaml
│   ├── ablations/         # a1_rope.yaml, a2_rmsnorm.yaml ...
│   └── finetune/          # gsm8k_lora_r8.yaml ...
├── src/tlab/
│   ├── model/
│   │   ├── attention.py   # MHA / GQA / MQA + KV cache
│   │   ├── layers.py      # RMSNorm, SwiGLU, RoPE
│   │   ├── block.py
│   │   └── gpt.py
│   ├── tokenizer/bpe.py
│   ├── data/
│   ├── train/trainer.py
│   ├── finetune/          # lora.py, sft.py
│   ├── convert/qwen3.py   # HF 权重映射
│   └── eval/              # gsm8k.py, generate.py
├── scripts/               # prepare_data.py, train.py, sample.py, eval_gsm8k.py
├── tests/
├── experiments/results.md
└── docs/notes/            # 你自己的推导笔记
```

- 环境：Python 3.11/3.12，PyTorch 2.7+（CUDA 12.8 构建），`uv` 管理依赖并锁版本。
- 代码：类型标注、张量形状注释、`ruff` 格式化；配置用 dataclass + YAML，命令行可覆盖单个字段。
- 可复现：每次训练保存完整配置、git commit 哈希、随机种子到输出目录。
- `.gitignore` 排除 `data/`、`checkpoints/`、日志；如需展示模型，可选把小 checkpoint 上传到 Hugging Face Hub。
- Git：`main` 始终可运行；每个任务一个分支、一个小 PR，提交信息写清做了什么。

## 简历写法与面试准备

简历条目先按下面的骨架写，方括号里的数字全部等实验跑完再填；没跑出来的那一条就删掉，不写预期值。申请英美硕士时需要英文版，W5 再一起翻译。

**Transformer 从零实现与高效微调**（PyTorch，个人项目，GitHub 链接）

- 从零实现 Llama/Qwen 风格 Decoder-only Transformer（RoPE、RMSNorm、SwiGLU、GQA、KV Cache），不依赖 `nn.Transformer`，单元测试覆盖注意力因果性与数值正确性。
- 在单张 RTX 5060（8GB）上用 TinyStories 预训练 [X]M 参数模型；同 token 预算下完成 [N] 组结构消融，[具体发现，如 RoPE 较可学习位置编码 val loss 降低 X%]。
- 编写 HF 权重映射，将 Qwen3-0.6B 载入自研架构，logits 与官方实现最大误差 [X]。
- 手写 LoRA 与 SFT 流程，在 GSM8K 上将 Qwen3-0.6B-Base 准确率从 [X]% 提升到 [Y]%，可训练参数占 [Z]%，结果与 PEFT 对照一致。

面试高频问题，W5 逐个用自己的话写答案：

1. 现场手写 multi-head attention（白板或共享文档）。
2. KV cache 的原理和显存计算；GQA 为什么能省显存。
3. RoPE 和绝对位置编码的区别。
4. Pre-Norm 和 Post-Norm 的区别，你的实验看到了什么。
5. LoRA 的原理、初始化、rank 怎么选。
6. 预训练和 SFT 的 loss 计算有什么不同。
7. 权重对齐时最难排查的 bug 是什么，怎么定位的。
8. 你的实验有什么局限（规模小、结论未必能迁移到大模型），如果有更多算力下一步做什么。

## 风险与降级方案

最大的风险是进度：只要 W3 末 MVP 完成，后面任何一项出问题都只影响「出彩程度」，不影响投递。

| 风险 | 早期信号 | 降级方案 |
| --- | --- | --- |
| 进度落后 | W3 末 A1–A4 没跑完 | 砍掉 A5、A6、F3、F5，保留 A1–A4、A7 和 F1、F2、F4 |
| 5060 环境装不上 | M0 第一天 GPU 验证不通过 | 改原生 Windows、不用 `torch.compile`；仍不行就用 AutoDL 等租卡 |
| LoRA 微调显存不够 | Qwen3-0.6B 训练 OOM | gradient checkpointing、序列长度 512、batch 1 + 梯度累积；仍不够则换 Qwen2.5-0.5B 或租卡 |
| 权重对齐卡住 | M4 超过 3 天无进展 | 改为载入 GPT-2 small（124M，结构更简单），或把 M4 挪到投递之后 |
| GSM8K 提升不明显 | 基座准确率已经较高，或微调后几乎不变 | 如实报告；改用格式更明确、提升空间更大的任务（如结构化输出或分类），重点展示流程和 PEFT 对照 |
| 课业冲突 | 期中考试周 | 那一周只跑已写好的实验，不开新模块 |
| 依赖 agent、讲不清 | 学习护栏的自测问题答不上 | 不合并、不进入下一阶段，先让 agent 讲到懂 |
