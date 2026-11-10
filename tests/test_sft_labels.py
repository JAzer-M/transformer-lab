"""SFT 的输入和目标（build_example）：只有答案部分算 loss。预期值都是手写的。"""

import pytest

# build_example 在 tlab/finetune/sft_labels.py；在各测试内导入，文件出错时每个测试单独报错


def test_prompt_targets_are_ignored_and_answer_targets_are_kept() -> None:
    from tlab.finetune.sft_labels import IGNORE_INDEX, build_example

    inputs, targets = build_example([10, 11, 12], [20, 21], eos_id=2)

    # 整条序列 [10, 11, 12, 20, 21, 2]：输入去掉最后一个，目标去掉第一个
    assert inputs == [10, 11, 12, 20, 21]
    # 目标的前 2 个（11、12）还是提示；第 3 个（20）已经是答案的第一个 token
    assert targets == [IGNORE_INDEX, IGNORE_INDEX, 20, 21, 2]


def test_ignore_index_is_what_cross_entropy_skips() -> None:
    from tlab.finetune.sft_labels import IGNORE_INDEX

    # F.cross_entropy 的 ignore_index 默认就是 -100
    assert IGNORE_INDEX == -100


def test_one_token_prompt_ignores_nothing() -> None:
    from tlab.finetune.sft_labels import build_example

    inputs, targets = build_example([10], [20, 21], eos_id=2)

    # 提示只有 1 个 token：它只出现在输入里，目标从答案开始，一个都不用遮
    assert inputs == [10, 20, 21]
    assert targets == [20, 21, 2]


def test_answer_with_only_eos_still_learns_to_stop() -> None:
    from tlab.finetune.sft_labels import IGNORE_INDEX, build_example

    inputs, targets = build_example([10, 11], [], eos_id=2)

    assert inputs == [10, 11]
    assert targets == [IGNORE_INDEX, 2]


@pytest.mark.parametrize(("n_prompt", "n_answer"), [(1, 1), (3, 2), (60, 90)])
def test_inputs_and_targets_have_the_same_length(n_prompt: int, n_answer: int) -> None:
    from tlab.finetune.sft_labels import IGNORE_INDEX, build_example

    inputs, targets = build_example(
        list(range(100, 100 + n_prompt)), list(range(500, 500 + n_answer)), eos_id=2
    )

    # 长度都是 提示 + 答案 + eos − 1；算 loss 的正好是答案和 eos
    assert len(inputs) == len(targets) == n_prompt + n_answer
    assert sum(t != IGNORE_INDEX for t in targets) == n_answer + 1


def test_inputs_are_not_masked() -> None:
    from tlab.finetune.sft_labels import IGNORE_INDEX, build_example

    inputs, _ = build_example([10, 11, 12], [20, 21], eos_id=2)

    # 只遮目标，不遮输入：模型要看到完整的题目才能回答
    assert IGNORE_INDEX not in inputs


def test_empty_prompt_is_an_error() -> None:
    from tlab.finetune.sft_labels import build_example

    # 没有提示时 n_prompt 是 -1：[-100] * -1 是空列表，targets[-1:] 只剩最后一个，
    # 输入 2 个、目标 1 个，长度对不上。模型总要先看到点什么才能回答
    with pytest.raises(ValueError, match="prompt"):
        build_example([], [20, 21], eos_id=2)
