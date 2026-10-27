# 项目进度与交接

本文件记录当前状态和交接证据。规则见 [AGENTS.md](AGENTS.md)，里程碑、验收标准与下一步计划见 [docs/PLAN.md](docs/PLAN.md)。实验数字归档到 `experiments/results.md`，这里引用对应实验，不复制完整结果表。

## 当前状态

更新时间：2026-10-27 17:17（Asia/Shanghai，UTC+08:00）。

| 项目 | 当前可核实状态 |
| --- | --- |
| 项目阶段 | 用户已确认保存 M0 Mac 基础环境，基线保存后进入 M1 注意力练习；完整 M0 仍有剩余验收 |
| 本轮范围 | M0 本地 Git 基线、m1-attention 分支与首个注意力测试 |
| 实现方 | 用户手写 MHA；Codex 准备测试、讲解并 review；用户决定切换时再交接 Claude Code |
| 审查方 | Codex 子 agent review_m0_bootstrap 已审查，无待修问题；Claude Code 尚未接手 |
| Git 分支 / commit | 当前 m0-bootstrap；用户已授权首个本地基线提交与 m1-attention 分支创建，实际提交以 Git 记录为准 |
| 本轮改动 | 先记录 M0 复核与授权；基线保存后新增 tests/test_attention.py，注意力实现由用户创建 |
| 工程代码与依赖 | 项目已可编辑安装；PyTorch 2.9.1、NumPy 2.4.6、pytest 9.1.1、ruff 0.17.2；锁文件共 38 个包条目，包含其他平台的依赖 |
| 本机环境 | Python 3.11.9，arm64；uv 0.13.5 已安装到 `/Users/j/.local/bin`，未修改 shell 配置 |
| 范围与保护检查 | 本轮 AGENTS.md / CLAUDE.md 字节未改；用户手写实现未创建，用户笔记目录为空，结果表无实验行 |
| pytest / 代码检查 | 本次 M0 提交前复核：5 passed in 1.09s；ruff check 与 format --check 均通过 |
| Mac 真实设备检查 | MPS 可用且被选中；CPU 与 MPS 的 (2, 3) 张量分配、求和均成功 |
| 训练 / 实验 | 均未运行；`experiments/results.md` 仅有表头，没有实验结果 |
| Windows / CUDA | 本轮未访问或验证，不从 Mac 状态推断 |
| 下一步 | 保存基线并加入首测后，由用户创建 CausalSelfAttention 并先写构造函数 |

## 当前任务与下一步

- 计划位置：`docs/PLAN.md` 的「M0 实施计划」，第一批已获用户确认。
- 当前操作：用户已授权保存 M0 本地基线并创建 `m1-attention`，执行后记录实际 commit。
- 分工：用户手写 MHA；Codex 准备首测、逐段讲解并审查。M0 工程已有独立审查记录。
- 首批本地验证已通过：包安装和隔离导入、设备选择逻辑、pytest、ruff、CPU/MPS 小张量检查；本次提交前再次复核测试和代码检查。
- Mac pytest 已通过；完整 M0 仍需补充 Windows/WSL 真实 CUDA 自检、该设备的 pytest 与 CPU 10 步训练证据；后者依赖 M1 代码就绪。
- 当前 M1 约定：`CausalSelfAttention(d_model=4, n_heads=1)`，输入与输出均为 `(B, T, C)`。先加入形状测试，再由用户亲手实现；后续逐项加入因果性、数值一致性与梯度测试。
- 本轮改动差异见 `runs/m0-bootstrap/m0-changes.patch`；该文件和检查日志留在本机，不进 Git。

## 交接记录

### 2026-10-26 18:26（UTC+08:00）｜Codex｜建立协作约定与 M0 提案

- 用户确认：采用共享进度文档交接，开始落实文档与下一步具体计划。
- 工作区：`/Users/j/Desktop/transformer-lab`；按项目约定属于 Mac 开发侧，本次只读检查与文档写入。
- 起点：目录中只有 `AGENTS.md`、单行 `@AGENTS.md` 的 `CLAUDE.md` 和 `docs/PLAN.md`，另有系统文件 `.DS_Store`。
- 改动：新增共享进度文档；规范中补充任务归属、工作区隔离与交接要求；计划改为仓库执行版本，并加入待确认的 M0 文件、测试和验收安排。
- 保护边界：未创建或改写用户手写实现、用户笔记、训练配置或实验结果；未安装依赖、初始化 Git、提交或推送。
- 本次实际检查：
  - `ls -la` 和过滤系统、数据等目录的 `rg --files --hidden`：确认上述起点文件，无源码或测试。
  - `git status --short --branch`：退出码 128，`fatal: not a git repository (or any of the parent directories): .git`。
  - `uv --version`：退出码 127，`zsh:1: command not found: uv`；只说明当前 PATH 找不到，尚未排查其他安装位置。
  - 使用 `python3` 读取 `sys.executable`、`platform.python_version()`、`platform.machine()` 与 Git 路径：退出码 0；Python 为 3.11.9，arm64，Git 位于 `/usr/bin/git`。
- 验证边界：以上不代表 PyTorch、MPS、Windows/CUDA、pytest 或训练已经通过。文档检查与交叉审查尚待执行。
- 决策依据：单份进度文档减少重复维护；原对话作为决策背景按需检索。M0 的 10 步训练依赖 M1，在代码就绪前保留待验证状态，不改低验收标准。
- 下一步：检查本轮文档，交叉审查，确认 M0 具体实施计划。

### 2026-10-26 18:31（UTC+08:00）｜Codex｜文档验证与独立审查

- 本次验证：运行内联 Python 文档检查，退出码 0，15 项全部通过。基于修改前的文档快照核对，原规范与计划正文除本轮明确新增及版本说明更正外完整保留，原里程碑、学习护栏与验收标准未改写。
- 检查范围：`CLAUDE.md` 与修改前字节一致，仍为单行 `@AGENTS.md`；本地 Markdown 链接有效；无行尾空格或未闭合代码块；仅更新或新增本轮三份文档，没有源码、依赖配置、测试文件或 Git 初始化。
- 独立审查：实际审查者为 Codex 子 agent `review_collaboration_docs`。首次发现 M0 补全摘要漏列真实 CUDA 自检，已在计划及下一任务清单中补齐；该审查者复核后确认问题解决，无剩余可操作问题。
- 状态更正：上一条记录中的「文档检查与交叉审查尚待执行」，其中本轮文档检查与 Codex 子 agent 审查已完成；Claude Code 审查仍未进行，不以其他审查替代其结论。
- 未执行：pytest、PyTorch/GPU 检查、训练、正式实验、依赖安装与 Git 操作。当前交付是协作文档与 M0 提案，M0 尚未实施。
- 下一步：让 Claude Code 阅读这三份文档并审查；用户确认 M0 的具体文件与测试计划后，再指定实现方与审查方开始首批工程工作。

### 2026-10-26 18:43（UTC+08:00）｜Codex｜用户确认开始 M0 第一批

- 用户指示：当前继续由 Codex 推进，额度将尽时用户再告知并切换 Claude Code；已确认开始下一步计划。此前等待 Claude 审查或 M0 计划确认的下一步安排已被这条指示更新。
- 范围：落实 `docs/PLAN.md` 中第一批工程基础与设备选择，测试接口为包导入及 `get_device() -> torch.device`。实现后由 Codex 子 agent 进行独立审查，Claude Code 尚未接手。
- 起点检查：原四份项目文档存在，无其他源码或测试；系统 Python 3.11.9 / arm64。当前 PATH 与常见安装位置均未发现 uv，系统 Python 也未安装 torch、pytest、ruff。
- 环境准备：从 Astral 官方地址下载并检查 uv 安装脚本，安装 uv 0.13.5 到用户本地工具目录，设置 `UV_NO_MODIFY_PATH=1` 保留 shell 配置；项目依赖安装到 `.venv`，缓存放项目 `.cache/uv`。
- Git：`git init -b m0-bootstrap` 成功；尚无首个 commit，未配置远程或推送，`main` 尚未建立。
- 失败与处理：普通沙箱中的下载无法连接当前代理端口，Git 初始化遇到写权限限制；在已确认的环境准备范围内通过工具权限复核后重试成功，未改动代理设置。
- 版本选择：依据 PyTorch 官方安装页确认 2.9.1 同时有 macOS 与 CUDA 12.8 构建；平台来源按 uv 官方 PyTorch 配置设置，具体传递依赖由 uv 锁定。未安装 torchvision、torchaudio 或后期微调依赖。
- 下一步：依赖准备完成后先确认导入测试因缺少 tlab 失败，再实现最小包与设备选择，逐步跑测试及代码检查；目前尚无测试通过结论。

### 2026-10-26 18:52（UTC+08:00）｜Codex｜M0 第一批 Mac 本地实现与验证完成

- 实现：`pyproject.toml` 与 `uv.lock` 管理 src 包和依赖；`.gitignore` 排除根目录数据、checkpoint、运行日志与本地私有配置，同时保留 `src/tlab/data/`；建立既定包和配置目录；新增普通函数 `get_device() -> torch.device`，按 CUDA → MPS → CPU 返回设备。
- 依赖：实际安装 uv 0.13.5、Python 3.11.9、torch 2.9.1、numpy 2.4.6、pytest 9.1.1、ruff 0.17.2。初次导入 PyTorch 出现缺少 NumPy 的警告，已补充基础依赖；原失败日志保留。锁文件区分 macOS/PyPI 的 torch 2.9.1 与 Linux/Windows CUDA 源的 2.9.1+cu128；这只证明解析配置，不代表 Windows 已安装或验证。
- 先红后绿的证据（均为 `.venv/bin/python -m pytest <测试文件> -q`）：导入测试先因缺少 tlab 失败，建包并可编辑安装后通过；设备测试先因缺少 tlab.device 出现收集错误，随后逐项添加 CPU、MPS、CUDA 优先级逻辑，对应失败均在最小实现后通过。未删除或削弱已有断言。
- 本次实际检查：
  - `uv run --locked pytest`：退出码 0，5 passed in 0.91s；涵盖隔离进程导入与 4 个设备可用性组合。
  - `uv run --locked ruff check .`：退出码 0，All checks passed。
  - `uv run --locked ruff format --check .`：退出码 0，17 files already formatted。
  - Mac 实机 Python 检查：退出码 0，MPS built/available 均为 true，选择 mps；CPU 和 MPS 上各创建形状 (2, 3) 的全 1 张量，求和均为 6.0。它是极小设备检查，没有训练步骤。
  - 忽略规则与范围：数据、checkpoint、运行日志、私有配置和虚拟环境被忽略，源码数据包未被忽略；AGENTS.md、CLAUDE.md 与本轮起点字节一致；用户手写文件未创建，笔记未写入，实验表无数据行。
- 命令环境：uv 位于 `/Users/j/.local/bin/uv`；本轮 uv 命令设置 `UV_CACHE_DIR=/Users/j/Desktop/transformer-lab/.cache/uv`，避免写入项目外缓存；依赖安装使用已存在的 Python 3.11.9。未修改 shell 配置，新终端找不到 uv 时按 README 临时加入 PATH。
- 证据位置：`runs/m0-bootstrap/01-import-red.txt` 至 `08-device-cuda-green.txt` 保留逐步失败与通过结果；`09-pytest-all.txt`、`10-ruff-check.txt`、`11-ruff-format.txt` 为最终检查；`environment.json` 为实机环境与张量检查，`gitignore-check.json`、`lock-summary-final.json` 为范围与平台源核验。运行日志不进 Git，以上摘要保留在本文件用于跨设备交接。
- 独立审查：Codex 子 agent `review_m0_bootstrap` 只读核对实现、规范、计划、测试与实际日志，未发现值得修复的问题，未重复运行测试；不代表 Claude Code 已审查。
- 未完成边界：用户尚需阅读本轮改动；Windows/WSL 的 CUDA 自检和 pytest 未执行；CPU 10 步训练等待 M1 最小模型与训练路径就绪。没有正式训练、模型/数据下载或实验结果，未填任何预估指标。
- Git：当前符号分支 `m0-bootstrap`，尚无 commit，所有项目文件仍为未提交状态；未配置远程、推送或建立 main。首个基线提交与 main 建立仍按用户后续验收与授权执行。
- 学习与下一步：理解设备描述和实际张量分配的区别，以及模拟选择分支与实机检查的证据边界；本轮验收后提出 M1 计划，保留用户亲手写 MHA 与 block 的安排。用户告知切换时，Claude Code 从本文件当前状态继续。

### 2026-10-27 17:17（UTC+08:00）｜Codex｜M0 提交前复核与 M1 准备授权

- 用户确认：同意将当前 M0 内容保存为一次本地 Git 提交，并创建 `m1-attention` 分支；随后由 Codex 加入首个测试，用户亲手写 MHA。
- 提交前复核：`uv run --locked pytest` 退出码 0，5 passed in 1.09s；`uv run --locked ruff check .` 退出码 0；`uv run --locked ruff format --check .` 退出码 0，17 files already formatted。
- 证据：`runs/m0-bootstrap/12-pytest-before-baseline.txt`、`13-ruff-before-baseline.txt`、`14-format-before-baseline.txt`。
- 提交范围：既有 M0 工程与项目文档。工作区新增的 `.idea/` 保留给用户，不纳入基线；注意力实现与测试目前均未创建。
- 首测约定：调用 `CausalSelfAttention(d_model=4, n_heads=1)`，给出形状 `(2, 3, 4)` 的输入，检查输出为 `(2, 3, 4)`。测试先前已在临时目录形成草稿；正式加入 M1 后重新记录结果。
- 历史记录更正：2026-10-26 依赖记录中“实际安装”列表包含 Python 3.11.9，表述不准确；Python 3.11.9 是机器原有解释器，本项目复用它建立虚拟环境，额外安装的是 uv 与项目依赖包。
- 下一步：完成本地基线和分支操作后，在 M1 记录实际 commit；加入首测并确认预期失败，然后交给用户编写注意力模块。

## 后续交接模板

每次结束或切换任务时，先更新当前状态，再在「交接记录」末尾追加一条。审查结果可以使用同一模板，署名实际审查者。

```markdown
### YYYY-MM-DD HH:MM（UTC+08:00）｜实际执行者｜任务

- 范围与用户确认：本次任务、已确认的计划；实现方 / 审查方。
- 工作区与设备：绝对路径；Mac 或 Windows/WSL；分支、commit、未提交改动。
- 改动：文件与行为；关联计划、diff 或提交。
- 本次验证：实际命令、退出码、结果摘要与证据位置。
- 未验证与失败：未运行项、失败证据、当前限制；历史或转述结果注明来源。
- 决策与原因：影响后续实现的选择；需要更正的旧记录与依据。
- 审查状态：未请求 / 待审查 / 已审查；实际审查者、问题与修复状态。
- 学习状态：用户仍需亲手实现或理解的内容。
- 下一步：具体动作、负责人、所需确认或前置条件。
```
