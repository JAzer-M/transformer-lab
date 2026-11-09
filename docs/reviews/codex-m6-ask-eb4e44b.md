# M6 ask.py 只读审查

审查日期为 2026-11-09（Asia/Shanghai，UTC+08:00）。审查者为 Codex，规范轴由只读子 agent `m6_ask_standards_review` 独立核对，需求轴由 `m6_ask_spec_review` 独立核对。全量测试、真实问答及完整评测入口的前后对照由主 Codex 执行。

## 固定范围与依据

- 基线为 `origin/main` 的 `e04bc79d2724dd06c4c9d15c79de3d94fed9fb7d`。
- 目标为 `origin/m6-ask` 的 `eb4e44b13e78785e7644d74ba47d265fea776667`，单个提交，共 15 个文件。
- 差异命令为 `git diff origin/main...origin/m6-ask`，实际审查同时以完整 SHA 固定范围。
- 规范依据为 `AGENTS.md` 硬性规则 1、2、4、5 及代码约定。需求依据为目标提交的 `PROGRESS.md` 顶部当前状态和第 1986 行起的“M6：自己提问的脚本 ask.py”交接记录。
- WSL 的 Git fetch 两次遇到 GnuTLS 握手中断。随后使用 Windows 现有 Git 成功 fetch，确认上述远端 SHA，再经本地 bundle 导入 WSL。两棵隔离工作树分别固定在基线和目标提交。

实际执行目录为 `/home/j/transformer-lab/runs/m6_ask_review_20261109/target`，基线目录为同级 `baseline`。复用现有虚拟环境，通过 `PYTHONPATH` 指向各自的 `src`。实际导入路径已记录并核对为目标工作树。

## 规范

### 确定问题

**P3，非阻塞。`tests/test_ask.py:17` 的新辅助函数 `ask` 缺少返回类型注解。**

`AGENTS.md:85` 要求类型标注。该函数已有参数类型，但定义缺少返回类型。实际用 AST 检查确认 `returns=None`，函数返回的是开启 `text=True` 的 `subprocess.run` 结果，可以补为 `-> subprocess.CompletedProcess[str]`。本项通过 AST 和规范逐项核对发现，Ruff 启用的 E4/E7/E9/F/I 规则另行通过。[目标代码](https://github.com/JAzer-M/transformer-lab/blob/eb4e44b13e78785e7644d74ba47d265fea776667/tests/test_ask.py#L17) [规范依据](https://github.com/JAzer-M/transformer-lab/blob/eb4e44b13e78785e7644d74ba47d265fea776667/AGENTS.md#L85)

### 已核验事项

- 硬性规则 1、2。实际比较 Git 对象，8 个带 `USER-WRITTEN` 标记的实现文件与基线逐字节相同。新增生成路径使用自研 GPT 和生成函数，`tests/conftest.py` 中的 `Qwen3ForCausalLM` 用于构造测试参考模型。
- 硬性规则 4。实际 AST 对照确认，修改的测试文件中原有测试函数均保留，164 条原有 `assert` 和 21 处 `pytest.raises`、`pytest.warns`、`torch.testing.assert_close` 调用相同。目标提交全量测试实测 778 passed。TDD 编写先后顺序的依据为交接历史陈述，证据等级记为推断。
- 用户指定的 RMSNorm 测试。`tests/test_gsm8k_scripts.py:331` 起的 `test_eval_script_records_the_actual_rmsnorm_setting` 适配了新加载函数的参数、元组返回值和 monkeypatch 目标，第 370 行仍断言实际记录为 `False`。实际 AST 对照及全量测试均通过。[对应测试](https://github.com/JAzer-M/transformer-lab/blob/eb4e44b13e78785e7644d74ba47d265fea776667/tests/test_gsm8k_scripts.py#L331)
- 三个迁移辅助函数。`tests/conftest.py:88` 的 `load_script` 主体 AST 相同。第 97 行的 `big_lora_model` 保留模型构造、种子、LoRA 安装和 B 初始化。第 114 行的 `write_eval_config` 在归一化参数改名及默认题目列表后，主体 AST 相同，默认题目仍为 `(1, 3)`。实际全量测试及下面的四组合入口对照通过。[共用辅助函数](https://github.com/JAzer-M/transformer-lab/blob/eb4e44b13e78785e7644d74ba47d265fea776667/tests/conftest.py#L88)
- 硬性规则 5。抽出的 `generate_answer` 保留输入、输出和新增 token 列表的形状注释。新的加载包装函数复用现有权重加载和合并实现。

规范轴共 1 个确定问题，最高级别为 P3。额外设计判断项为 0 个。

## 需求与正确性

### 确定问题

新增确定问题为 0 个。

### 已核验事项

- **评测入口行为。** 实际用同一个本地随机微型模型和同一份随机 LoRA 张量，运行基线与分支的完整 `scripts/eval_gsm8k.py`。few-shot、zero-shot 分别配基座、LoRA，共 4 组。去除 `seconds`、`tokens_per_second`、`git_commit` 三个易变字段后，全部 JSON 值和字段顺序逐字相同。模型与 LoRA 通过随机初始化构造，验证执行前向生成。
- **生成结果字段及停止行为。** `src/tlab/eval/gsm8k.py:161` 的 `generate_answer` 和第 195 行的 `answer_question` 经轻量 CPU 探针验证，cache 开关与 eos、长度截止组成的 4 组返回值及字段顺序均与基线相同。标准答案检查提前符合交接决定。[生成与评测路径](https://github.com/JAzer-M/transformer-lab/blob/eb4e44b13e78785e7644d74ba47d265fea776667/src/tlab/eval/gsm8k.py#L161)
- **提示、截断和抽答案。** `scripts/ask.py:63` 的 `ask_one` 调用共用的 `zero_shot_prompt` 和 `generate_answer`。第 49 行的 `describe_reply` 用同一个 `truncate_continuation` 显示回答，抽取结果直接使用共用生成结果中的 `pred`。CPU 假生成器探针实际确认，续编的新题被隐藏，截断后抽到 `29`。真实 GPU 问答也验证了提示及截断显示，结果见下一节。[提问入口](https://github.com/JAzer-M/transformer-lab/blob/eb4e44b13e78785e7644d74ba47d265fea776667/scripts/ask.py#L49)
- **LoRA 路径检查顺序。** 实际向 `ask.main`、`eval.main` 和 `load_qwen3_with_lora` 传入缺失路径，并以哨兵替换基座解析或加载函数。三处均先抛出 `FileNotFoundError`，哨兵调用次数为 0。对应位置为 `scripts/ask.py:95`、`scripts/eval_gsm8k.py:66` 和 `src/tlab/finetune/lora_model.py:175`。[ask 检查](https://github.com/JAzer-M/transformer-lab/blob/eb4e44b13e78785e7644d74ba47d265fea776667/scripts/ask.py#L95) [评测检查](https://github.com/JAzer-M/transformer-lab/blob/eb4e44b13e78785e7644d74ba47d265fea776667/scripts/eval_gsm8k.py#L66) [共用加载检查](https://github.com/JAzer-M/transformer-lab/blob/eb4e44b13e78785e7644d74ba47d265fea776667/src/tlab/finetune/lora_model.py#L175)
- **交互退出。** `scripts/ask.py:36` 的等待输入路径经实际探针验证，EOF 和 Ctrl-C 均结束输入。778 项测试还覆盖空行结束及连续提问。此项验证范围为等待输入阶段。[输入循环](https://github.com/JAzer-M/transformer-lab/blob/eb4e44b13e78785e7644d74ba47d265fea776667/scripts/ask.py#L36)

需求轴共 0 个新增确定问题。

## 本次运行证据

模型与测试命令设置了 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1` 和 `UV_OFFLINE=1`，uv 使用 `--offline --no-sync`。真实问答从既有缓存载入 `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`。

| 检查 | 实际结果 |
| --- | --- |
| `uv run --offline --no-sync pytest -q` | 退出 0，778 passed in 82.01s |
| `uv run --offline --no-sync ruff check --no-cache .` | 退出 0，All checks passed |
| `uv run --offline --no-sync ruff format --check --no-cache .` | 退出 0，131 files already formatted |
| 完整评测入口的基线与分支对照 | 4/4 组合通过，非易变 JSON 字段及顺序完全相同 |
| CPU 生成路径探针 | cache 开关 × eos、长度截止共 4 组通过，使用小 CPU 张量和假生成器 |
| 缺失 LoRA 路径哨兵探针 | 3 处均先报路径错误，基座解析或加载调用为 0 |
| Git 只读状态 | 测试后的目标及基线工作树均干净，原 WSL 仓库的 tracked 文件无改动 |

### 真实问答

两次使用同一道题、默认 `max_new_tokens=256`、eos 编号 151643，在 RTX 5060 Laptop GPU 上执行。

```text
Tom has 3 boxes with 12 pens each. He gives away 7 pens. How many pens does he have left?
```

LoRA 使用 `/home/j/transformer-lab/runs/m5/f2_r8_seed1337/lora.pt`，SHA-256 为 `1cfba5102bcc0337d9d37794f11ed80b1f294a4a889886361c62c139e77b55bf`。

| 模型 | 抽出的答案 | eos 停止 | 新 token 数 | 生成用时 | 隐藏的续编字符数 |
| --- | --- | --- | --- | --- | --- |
| SFT LoRA，r=8、seed=1337 | 29 | 是 | 34 | 2.6 秒 | 0 |
| 基座 | 29 | 否，达到 256-token 上限 | 256 | 9.8 秒 | 600 |

用时采用脚本显示值，精度为 0.1 秒。两次提示均为 `Question: Tom has 3 boxes with 12 pens each. He gives away 7 pens. How many pens does he have left?\nAnswer:`。LoRA 的回答包含 `#### 29`，基座显示的第一段以 `The answer is 29.` 结束。两次抽出的答案相同。

## 已知限制与证据边界

- **静态推断，交接已声明。** `--max-new-tokens 0` 会使 `src/tlab/eval/gsm8k.py:183` 的 `new_ids[-1]` 访问空列表。对应的基线 `answer_question` 同样访问空列表。该边界已在交接第 2003 行明确保留，本次计入已知限制。0-token 入口的结论来自代码核对，实际问答使用默认 256。[空列表访问位置](https://github.com/JAzer-M/transformer-lab/blob/eb4e44b13e78785e7644d74ba47d265fea776667/src/tlab/eval/gsm8k.py#L183) [交接记录](https://github.com/JAzer-M/transformer-lab/blob/eb4e44b13e78785e7644d74ba47d265fea776667/PROGRESS.md#L2003)
- 完整评测入口的四组合对照使用 CPU 微型模型。真实 GPU 验证为上表的一道题及两种模型。Claude 交接中的 Mac 测试与历史冒烟结果属于历史记录，本报告的运行表列出本次实际执行项。

## 交付与后续

本次交付为 `docs/reviews/codex-m6-ask-eb4e44b.md`。源码、配置和实验结果经 Git 差异检查与各自的固定提交一致。用户随后确认“提交并推送报告到 m6-ask”，发布范围为本文件，代码建议供实现方后续处理。实际提交和远端状态以 Git 记录为准。

持久运行证据位于 `/home/j/transformer-lab/runs/m6_ask_review_20261109/`，包括 `review_source.json`、`review.diff`、`runtime_checks.json`、`source.log`、`pytest.log`、两份 Ruff 日志、`ask_lora.log`、`ask_base.log` 和 `eval_equivalence.json`。四组合的原始评测 JSON 与日志位于其下 `eval_equivalence/`。

规范结论为 1 项 P3，需求结论为 0 项新增确定问题。两条轴分别核对和列示。
