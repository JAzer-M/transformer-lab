"""SFT 的数据管线（M5）：模板、分词、拼 batch、验证集、训练顺序。

哪些位置算 loss 由 build_example 决定（sft_labels.py）。
"""

import random
from collections.abc import Sequence

import torch

from tlab.eval.gsm8k import (
    TRAIN_TOTAL,
    Example,
    Tokenizer,
    strip_calculator,
    zero_shot_prompt,
)
from tlab.finetune.sft_labels import IGNORE_INDEX, build_example

VAL_SIZE = 200  # 从训练集里留出来看 val loss；测试集只用于最后的准确率

Item = tuple[list[int], list[int]]  # (输入, 目标)


def format_example(example: Example) -> tuple[str, str]:
    """提示和评测时完全相同；答案去掉 <<...>> 标注，前面加空格接在 "Answer:" 后面。"""
    return zero_shot_prompt(example.question), " " + strip_calculator(example.answer)


def tokenize_examples(
    examples: Sequence[Example], tokenizer: Tokenizer, eos_id: int, max_len: int
) -> tuple[list[Item], int]:
    """分词并做成 (输入, 目标)；输入比 max_len 长的丢掉，返回样本和丢掉的条数。

    提示和答案分开编码，提示部分才和评测时逐 token 相同。
    """
    items: list[Item] = []
    dropped = 0
    for example in examples:
        prompt, answer = format_example(example)
        inputs, targets = build_example(
            tokenizer.encode(prompt), tokenizer.encode(answer), eos_id
        )
        if len(inputs) > max_len:
            dropped += 1
            continue
        items.append((inputs, targets))
    return items, dropped


def collate(batch: Sequence[Item], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """在右边补齐到这一批最长的长度：输入补 pad_id，目标补 -100（不算 loss）。

    因果掩码下右边的 pad 影响不到前面的 token，所以不需要 attention mask。
    """
    length = max(len(inputs) for inputs, _ in batch)
    inputs = torch.full((len(batch), length), pad_id, dtype=torch.long)  # (B, T)
    targets = torch.full((len(batch), length), IGNORE_INDEX, dtype=torch.long)  # (B, T)
    for row, (x, y) in enumerate(batch):
        inputs[row, : len(x)] = torch.tensor(x)  # 这一行的前 len(x) 个
        targets[row, : len(y)] = torch.tensor(y)  # 同上，len(y) == len(x)
    return inputs, targets


def make_split(seed: int) -> dict:
    """用固定种子从训练集里留出 VAL_SIZE 条做验证，其余用来训练；下标都排好序。"""
    val = sorted(random.Random(seed).sample(range(TRAIN_TOTAL), VAL_SIZE))
    return {
        "seed": seed,
        "train_total": TRAIN_TOTAL,
        "val_indices": val,
        "train_indices": sorted(set(range(TRAIN_TOTAL)) - set(val)),
    }


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    """第 epoch 个 epoch 的训练顺序：0 到 n-1 的一个排列，只由种子和 epoch 决定。

    PEFT 对照（F4）用同一个函数，两边的数据顺序才完全相同。
    """
    order = list(range(n))
    random.Random(f"{seed}:{epoch}").shuffle(order)
    return order


def optimizer_steps(
    order: Sequence[int], batch_size: int, grad_accum_steps: int
) -> list[list[list[int]]]:
    """把顺序切成一步一步：每步 grad_accum_steps 个微批次，每个微批次 batch_size 条。

    最后凑不满一步的丢掉，每一步的有效 batch 大小都相同。
    """
    per_step = batch_size * grad_accum_steps
    steps = []
    for start in range(0, len(order) - per_step + 1, per_step):
        chunk = list(order[start : start + per_step])
        steps.append(
            [chunk[i : i + batch_size] for i in range(0, per_step, batch_size)]
        )
    return steps
