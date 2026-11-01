# M2 TinyStories 数据管线审查

审查时间：2026-11-01 12:29（Asia/Hong_Kong，UTC+08:00）。

已在 Ubuntu `/home/j/transformer-lab` 完成 GitHub 拉取，并阅读 `AGENTS.md`、`CLAUDE.md`、`docs/PLAN.md` 的 M2 计划及 `PROGRESS.md` 中 Claude 的审查入口和最新交接。

固定范围为 `bd46e1a783ea544ef03c0803a3868702c8a938e4...6ecc28f712ca3175dccc1533b1f45f0663eb0522`，共 18 个文件。实现提交为 `619bbdd`，后续 `6ecc28f` 仅修改交接文档。拉取后的本地 `m2-data` 与 `origin/m2-data` 完全一致；审查前工作区干净。

规范轴：0 个需要修复的确定问题。需求轴：0 个需要修复的确定问题。本次未发现阻塞合并的问题；全量数据处理的资源表现尚未验证，不能据此宣称整个 M2 已验收。

规范审查由 Codex 子 agent `review_standards` 只读执行，需求审查由 `review_spec` 独立只读执行。以下运行验证由主 Codex 实际执行；未将 Claude 在 Mac 上的历史结果当作本次结果。

已核对的规范要求：

- attention、block、gpt、trainer、手写 bpe、全部原有测试、旧配置和实验结果表共 18 个受保护文件，与基线逐字节一致。
- 新冒烟配置除 `data_dir`、`out_dir` 外，与 `tiny_smoke.yaml` 的参数完全一致。
- Git 中没有数据、checkpoint 或运行产物；新增张量操作有形状说明。
- loader 的共用加载逻辑、tokenizer 包装与 `char.get_batch` 兼容再导出均有明确用途，未发现值得报告的过度设计。

已核对的需求要求：

- 全量模式在官方训练集训练词表，官方 valid 独立用于验证；sample 的 90/10 切分只用于冒烟。
- 默认 8192 词表包含 EOT，采用字节级 BPE 和 GPT-2 正则；逐篇追加 EOT，以 uint16 写盘并通过只读 memmap 加载。
- 随机起点算法与 M1 保持一致，返回连续 int64 张量；未修改用户训练循环。
- 故事拆分支持同行多个分隔符、分隔符周围空白、CRLF、空文件和故事内部空行。首尾 `strip()` 属于明确约定；字面 EOT 在原始数据中被视为分隔符，在直接调用 tokenizer 时被视为特殊 token。
- 下载固定 revision，并核对大小和 SHA-256；已有文件只做本地校验。残缺文件会报告实际大小与预期大小。未发现额外实现任务 C/D 的情况。

本次运行证据：

| 检查 | 结果 |
| --- | --- |
| `uv sync --locked` | 退出 0；锁定依赖安装成功 |
| `uv run --locked pytest -q` | 退出 0；84 passed in 5.66s |
| `uv run --locked ruff check .` | 退出 0；All checks passed |
| `uv run --locked ruff format --check .` | 退出 0；51 files already formatted |
| 独立 M1 采样对照 | 40 个种子 × 5 种形状 × 连续 3 次，共 600 组；旧实现、新张量路径、新 memmap 路径逐个相同 |
| dtype、连续性和真实 CUDA 转移 | CPU 与 RTX 5060 Laptop GPU 均通过，包括超过 32767 的 uint16 编号 |
| 额外故事边界检查 | 9 组通过，包含空文件、连续/同行分隔符、内部空行、CRLF、中文和 emoji |
| 下载行为探针 | 固定 revision 参数、已有文件不联网、残缺文件报错均通过；此项使用临时假下载，不代表真实 HF 下载成功 |
| 编码与写盘探针 | Unicode/空白/EOT 无损还原；无字面 EOT 的故事每篇恰好追加一个 EOT，通过 |
| Git 差异和保护检查 | `git diff --check` 通过；追加交接前 tracked 工作区无改动 |

环境为 Python 3.12.14、PyTorch 2.9.1+cu128 / CUDA 12.8、tokenizers 0.23.3、huggingface-hub 2.4.0；WSL 可用总内存约 7.6 GiB。

限制与未完成检查：

- 本次尝试 `uv run --locked python -u scripts/prepare_tinystories.py --sample`，HF HEAD 请求报 `[Errno 101] Network is unreachable`。为避免继续无效重试，定向 SIGINT 中断该进程，命令退出 130。原始错误保留在日志中。
- 改用 Windows `curl.exe` 获取同一固定 revision 的官方 valid 文件，连接约 10 秒后超时，退出 28，HTTP 000、下载 0 字节。当前系统代理未启用，7897 没有监听；本轮未启动代理或修改系统网络设置。
- 因此本轮没有在真实 TinyStories valid 数据上完成准备、原样 10 步冒烟及 sample.py 生成。84 项测试中包含在合成 BPE 数据上训练 3 步的集成测试，不能替代真实数据冒烟。Claude 的 Mac 冒烟仍仅作为历史证据。
- 没有下载约 2.2 GB 的训练集，没有运行正式训练，没有向结果表追加实验数字。全量内存、耗时和 huggingface-hub 0.23 下限版本未实测。

非阻塞建议，不计作确定缺陷：

- [scripts/prepare_tinystories.py:73](../../scripts/prepare_tinystories.py#L73) 和第 75 行统计 EOT 时会产生全长布尔数组；按交接中的约 5.4 亿 token 估算，额外内存约 0.54 GB。正式全量运行时应测峰值；需要节省时可分块计数或在写盘时同时累计。这是静态估算，本轮未观察到 OOM。
- [src/tlab/train/trainer.py:20](../../src/tlab/train/trainer.py#L20) 和第 21 行仍标注 `torch.Tensor`，实际现在也会收到 memmap；运行测试通过，但注解可由用户后续更新。该文件为用户手写，本轮没有编辑。

交付与后续：审查阶段只追加 Ubuntu 仓库的 `PROGRESS.md` 记录，没有提交、推送或合并。用户随后要求提交供 Claude 读取，因此将完整报告归档到本文件，与 `PROGRESS.md` 一起提交并推送 `m2-data`；实际提交和远端状态以 Git 记录为准。源码、配置、依赖声明和测试均未改动。建议网络恢复后补真实 valid 冒烟，任务 D 再验证全量资源表现；是否合并仍由用户决定。

原始运行证据保存在 `/home/j/transformer-lab/runs/m2_data_review_6ecc28f/`，包括依赖安装、pytest、两项 ruff、失败下载、行为探针与受保护文件哈希审计。

供用户理解与自测：

1. 为什么硬盘中的 uint16 数据进入 embedding 前要转成 int64？
2. 移动 get_batch 后，需要保持哪些随机数行为才能复现 M1？
3. 为什么使用 memmap 后，某些 NumPy 运算仍然会分配与整个数据集一样长的数组？
