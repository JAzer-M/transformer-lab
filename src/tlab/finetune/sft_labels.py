# USER-WRITTEN: agent 只提建议， 不改实现
"""SFT 的输入和目标：只有答案部分算 loss（M5）"""

# F.cross_entropy 默认跳过这个值的位置
IGNORE_INDEX = -100


def build_example(
    prompt_ids: list[int], answer_ids: list[int], eos_id: int
) -> tuple[list[int], list[int]]:
    # 提示至少要有 1 个 token： 没有提示时 n_prompt 是 -1， 切片会取错
    if not prompt_ids:
        raise ValueError("prompt_ids 不能为空")
    # 整条序列: 提示 + 答案 + eos
    ids = prompt_ids + answer_ids + [eos_id]
    # 输入是前 n - 1个， 目标是后 n - 1 个：第 i 个位置要预测第 i + 1 个 token
    inputs = ids[:-1]
    targets = ids[1:]
    # 目标里的前 len(prompt) - 1 个还是提示， 设成 -100 不算 loss
    n_prompt = len(prompt_ids) - 1
    targets = [IGNORE_INDEX] * n_prompt + targets[n_prompt:]
    return inputs, targets
