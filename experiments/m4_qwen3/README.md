# M4：载入 Qwen3-0.6B 的对齐结果

- `alignment.json`：正式验收。在干净的提交上运行 `scripts/check_qwen3.py`，Mac CPU，fp32。
- `alignment_official_scaling.json`：一次对照。在代码副本里把 `attention.py` 的 `scores = scores / (self._head_dim**0.5)` 改成官方的 `scores = scores * self._head_dim**-0.5` 后运行，所以 `git_commit` 带 `-dirty`，那行改动也没有进仓库。三个提示每一层的误差都是 0.0，说明正式结果里的误差只来自这一处的浮点舍入。

正式代码不改这一行：它是用户手写的，改了 M1–M3 的结果就不能逐字复现，而现在的误差已经满足验收。
