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
- [ ] MVP 关口：A1–A4 结果表可复现（2026-11-15）
- [ ] 自研架构载入 Qwen3-0.6B，输出对齐（2026-11-20）
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

### M3 现代组件 + 消融（W3）——MVP 关口

1. 实现 RoPE、RMSNorm、SwiGLU、GQA/MQA、KV cache，全部可通过配置开关切换。
2. 按「实验设计」跑 A1–A4 四组消融。
3. README 初版：结构图、结果表、loss 曲线。

验收：结果表可复现；到这一步项目已经可以写进简历，后面的部分出问题也不影响投递。

### M4 自研架构载入 Qwen3-0.6B（W4 前半）

1. 以 HF 的 `config.json` 为准，逐项补齐 Qwen3 的细节：QK-Norm、GQA 头数、RoPE 的 theta、tied embeddings 等。
2. 写权重映射脚本：HF 参数名 → 自己的模块名，处理转置和合并/拆分。
3. 写对齐测试：同一输入下逐层对比 hidden states，最后对比 logits。

验收：fp32 下 logits 与 `transformers` 输出的最大误差在 1e-4 量级；greedy 生成结果一致。卡住超过 3 天就走降级方案。

### M5 手写 LoRA + SFT（W4 后半–W5 前半）

1. 手写 `LoRALinear`：A 随机初始化、B 零初始化、缩放 alpha/r、merge 与 unmerge。
2. SFT 数据管线：prompt 模板、只对答案部分算 loss（prompt 位置的 label 设为 -100）、padding。
3. 任务：GSM8K，基座 Qwen3-0.6B-Base。评测：固定 500 题测试子集，greedy 解码，抽取最终数值算准确率。
4. 先测基座准确率（W4 第一天就测，确认有提升空间），再在自研模型上做 LoRA SFT；用 HF + PEFT 同超参跑对照组。

验收：微调前后准确率对比、可训练参数占比、显存和速度；自研 LoRA 与 PEFT 的 loss 曲线基本重合。

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
