"""SFT 的数据管线（agent 编写）：模板、分词、丢掉太长的、拼 batch、验证集、训练顺序。"""

import json
from pathlib import Path

import pytest
import torch

from tlab.eval.gsm8k import Example, zero_shot_prompt
from tlab.finetune.sft_data import (
    TRAIN_TOTAL,
    VAL_SIZE,
    collate,
    epoch_order,
    format_example,
    make_split,
    optimizer_steps,
    tokenize_examples,
)
from tlab.finetune.sft_labels import IGNORE_INDEX
from tlab.model.gpt import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "configs" / "finetune" / "gsm8k_sft_split.json"
EOS = 256  # 假分词器：0–255 是字节，256 是 eos


class ByteTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


# ---- 模板 ----


def test_format_example_uses_the_eval_prompt_and_strips_annotations() -> None:
    example = Example("How many?", "16-3-4=<<16-3-4=9>>9\n#### 9")

    prompt, answer = format_example(example)

    # 提示和评测时完全相同；答案前面加一个空格，接在 "Answer:" 后面
    assert prompt == zero_shot_prompt("How many?") == "Question: How many?\nAnswer:"
    assert answer == " 16-3-4=9\n#### 9"


# ---- 分词 ----


def test_tokenized_example_keeps_the_prompt_tokens_of_the_eval() -> None:
    example = Example("Q?", "#### 7")

    items, dropped = tokenize_examples([example], ByteTokenizer(), EOS, max_len=64)

    assert dropped == 0
    inputs, targets = items[0]
    prompt_ids = ByteTokenizer().encode("Question: Q?\nAnswer:")  # 评测时的提示
    answer_ids = ByteTokenizer().encode(" #### 7")
    assert inputs == prompt_ids + answer_ids  # 最后一个 eos 不在输入里
    # 算 loss 的正好是答案和 eos
    assert [t for t in targets if t != IGNORE_INDEX] == answer_ids + [EOS]


def test_examples_longer_than_max_len_are_dropped_and_counted() -> None:
    short = Example("Q?", "#### 1")  # 输入 20 + 7 = 27 个 token
    long = Example("Q" * 40 + "?", "#### 1")  # 输入 66 个 token

    items, dropped = tokenize_examples([short, long], ByteTokenizer(), EOS, max_len=30)

    assert dropped == 1
    assert len(items) == 1
    assert len(items[0][0]) == 27


# ---- 拼 batch ----


def test_collate_pads_on_the_right_and_ignores_padding() -> None:
    batch = [([1, 2, 3], [IGNORE_INDEX, 3, 4]), ([5, 6, 7, 8, 9], [6, 7, 8, 9, 10])]

    inputs, targets = collate(batch, pad_id=0)

    assert inputs.tolist() == [[1, 2, 3, 0, 0], [5, 6, 7, 8, 9]]  # (B, T) = (2, 5)
    assert targets.tolist() == [
        [IGNORE_INDEX, 3, 4, IGNORE_INDEX, IGNORE_INDEX],
        [6, 7, 8, 9, 10],
    ]
    assert inputs.dtype == targets.dtype == torch.long


@torch.no_grad()
def test_right_padding_does_not_change_the_scores_of_real_tokens() -> None:
    torch.manual_seed(0)
    model = GPT(
        GPTConfig(
            vocab_size=EOS + 1,
            block_size=32,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pos_encoding="rope",
            norm="rmsnorm",
        )
    ).eval()
    short = ([1, 2, 3], [2, 3, 4])
    long = ([5, 6, 7, 8, 9, 10], [6, 7, 8, 9, 10, 11])

    padded, _ = collate([short, long], pad_id=0)  # (2, 6)
    alone, _ = collate([short], pad_id=0)  # (1, 3)

    # 因果掩码：右边的 pad 在后面，前面的 token 看不到它；RoPE 的位置也都从 0 数起
    torch.testing.assert_close(model(padded)[0, :3], model(alone)[0])  # (3, V)


# ---- 验证集 ----


def test_committed_split_has_200_validation_examples_and_can_be_regenerated() -> None:
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))

    assert split == make_split(split["seed"])
    val = split["val_indices"]
    assert len(val) == len(set(val)) == VAL_SIZE == 200
    assert 0 <= min(val) and max(val) < TRAIN_TOTAL == 7473
    assert split["train_indices"] == sorted(set(range(TRAIN_TOTAL)) - set(val))


# ---- 训练顺序 ----


def test_epoch_order_is_a_permutation_fixed_by_the_seed() -> None:
    order = epoch_order(50, seed=1337, epoch=0)

    assert sorted(order) == list(range(50))  # 每条用一次
    assert order == epoch_order(50, seed=1337, epoch=0)
    assert order != epoch_order(50, seed=2024, epoch=0)
    assert order != epoch_order(50, seed=1337, epoch=1)  # 每个 epoch 重新打乱
    assert order != list(range(50))


def test_optimizer_steps_group_micro_batches_and_drop_the_last_partial_step() -> None:
    order = list(range(10))

    steps = optimizer_steps(order, batch_size=2, grad_accum_steps=2)

    # 每步 2 × 2 = 4 条；10 条凑成 2 步，剩下 2 条丢掉
    assert steps == [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]


@pytest.mark.parametrize(("n", "expected"), [(7273, 454), (16, 1), (15, 0)])
def test_number_of_optimizer_steps(n: int, expected: int) -> None:
    steps = optimizer_steps(list(range(n)), batch_size=4, grad_accum_steps=4)

    assert len(steps) == expected


def test_example_exactly_max_len_long_is_kept() -> None:
    example = Example("Q?", "#### 1")  # 输入正好 27 个 token

    items, dropped = tokenize_examples([example], ByteTokenizer(), EOS, max_len=27)

    assert dropped == 0 and len(items) == 1
