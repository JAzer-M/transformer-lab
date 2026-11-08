"""GSM8K 评测：提示模板、答案抽取、准确率、固定的 500 题。"""

import json
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import torch
from torch import nn

from tlab.eval.generate import generate
from tlab.train.config import load_dataclass
from tlab.train.utils import autocast_context, peak_memory_mb

TEST_TOTAL = 1319  # openai/gsm8k main 测试集的条数
TRAIN_TOTAL = 7473  # 训练集的条数
EVAL_SIZE = 500
N_SHOTS = 4

ANSWER_MARK = "####"
# 一个数：可选负号和美元符号，整数部分（带千分位逗号或不带），可选小数。
# 前面不能紧贴数字，也不能是“数字 + 小数点或逗号”：10-5 里的减号不是负号，
# 1,000 不能从中间算起；单独的省略号不算，...18 还是 18。
# 小数部分必须有数字，所以句末的句号不会被当成小数点
NUMBER = re.compile(r"(?<!\d)(?<!\d[.,])(-?)\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")
CALCULATOR = re.compile(r"<<[^>]*>>")
# 基座答完常接着编下一题。换行之后这几种开头都是编的（2026-11-07 试跑时见到的形式）
CONTINUATION = re.compile(r"\n\s*(?:\[Question\]|Question:|Q:)")
# 不给例子时基座常这样结尾；后面只能隔空白、冒号或 markdown 的星号
ANSWER_PHRASE = re.compile(r"answer is[\s:*]*", re.IGNORECASE)
PROMPT_MODES = ("few_shot", "zero_shot")


class Example(NamedTuple):
    question: str
    answer: str  # 原始答案：带 <<...>> 计算标注，最后一行是 #### 数字


@dataclass
class Gsm8kEvalConfig:
    """一次评测的全部设置；贪心解码、一次一题是固定的，不是配置。"""

    repo: str = "Qwen/Qwen3-0.6B-Base"
    prompt_mode: str = "few_shot"  # few_shot 或 zero_shot
    max_new_tokens: int = 256
    eos_token_id: int | None = None
    data_dir: str = "data/gsm8k"
    subset_file: str = "configs/finetune/gsm8k_eval_subset.json"
    limit: int | None = None  # 冒烟时只评前几题；正式评测不设
    lora: str | None = None  # LoRA 存档；给了就先载基座，再装上并合并
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.prompt_mode not in PROMPT_MODES:
            raise ValueError(
                f"prompt_mode 应为 {PROMPT_MODES}，收到 {self.prompt_mode!r}"
            )
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens 至少是 1，收到 {self.max_new_tokens}")
        if self.limit is not None and self.limit < 1:
            raise ValueError(f"limit 至少是 1，收到 {self.limit}")


def load_eval_config(
    path: str | Path, overrides: list[str] | None = None
) -> Gsm8kEvalConfig:
    """读取评测配置，命令行可覆盖单个字段。"""
    return load_dataclass(Gsm8kEvalConfig, path, overrides)


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


def _normalize(match: re.Match[str]) -> str:
    """去掉逗号和多余的 0：18.00 和 18 是同一个答案，-0 和 0 也是。"""
    sign, integer, fraction = match.groups()
    value = Decimal(sign + integer.replace(",", "") + (fraction or ""))
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def extract_answer(text: str) -> str | None:
    """取最终答案，依次看：第一个 #### 后面的数；第一个 answer is 后面紧跟的数；最后一个数。

    都没有时返回 None。
    """
    mark = text.find(ANSWER_MARK)
    if mark != -1:
        match = NUMBER.search(text, mark + len(ANSWER_MARK))
        if match is not None:
            return _normalize(match)
    phrase = ANSWER_PHRASE.search(text)
    if phrase is not None:
        match = NUMBER.match(text, phrase.end())
        if match is not None:
            return _normalize(match)
    matches = list(NUMBER.finditer(text))
    return _normalize(matches[-1]) if matches else None


def truncate_continuation(text: str) -> str:
    """基座答完可能不停，接着编下一题；只留第一处续写之前的部分。"""
    return CONTINUATION.split(text, maxsplit=1)[0]


def strip_calculator(answer: str) -> str:
    """去掉 <<16-3-4=9>> 这样的计算标注：16-3-4=<<16-3-4=9>>9 → 16-3-4=9。"""
    return CALCULATOR.sub("", answer)


def zero_shot_prompt(question: str) -> str:
    """不给例子。以冒号结尾，不留空格：训练时答案前的空格要和答案一起分词。"""
    return f"Question: {question}\nAnswer:"


def few_shot_prompt(question: str, examples: Sequence[Example]) -> str:
    """把例子按同样的格式写在前面，例子之间隔一个空行。"""
    shots = "".join(
        f"{zero_shot_prompt(example.question)} {strip_calculator(example.answer)}\n\n"
        for example in examples
    )
    return shots + zero_shot_prompt(question)


def accuracy(preds: Sequence[str | None], golds: Sequence[str]) -> float:
    """答对的比例；None 表示没答出，算错。"""
    if len(preds) != len(golds):
        raise ValueError(f"预测 {len(preds)} 条，答案 {len(golds)} 条，数量不同")
    if not golds:
        raise ValueError("没有题目，无法计算准确率")
    return sum(p == g for p, g in zip(preds, golds, strict=True)) / len(golds)


def make_eval_subset(seed: int) -> dict[str, Any]:
    """用固定种子选出评测的 500 题和 few-shot 的 4 个例子，下标都排好序。"""
    rng = random.Random(seed)
    return {
        "seed": seed,
        "test_total": TEST_TOTAL,
        "test_indices": sorted(rng.sample(range(TEST_TOTAL), EVAL_SIZE)),
        "train_total": TRAIN_TOTAL,
        "few_shot_indices": sorted(rng.sample(range(TRAIN_TOTAL), N_SHOTS)),
    }


def load_examples(path: Path) -> list[Example]:
    """读 prepare_gsm8k.py 写出的 jsonl，每行一条。"""
    with path.open(encoding="utf-8") as f:
        return [Example(row["question"], row["answer"]) for row in map(json.loads, f)]


def answer_question(
    model: nn.Module,
    tokenizer: Tokenizer,
    example: Example,
    prompt: str,
    max_new_tokens: int,
    eos_token_id: int | None,
    device: torch.device,
    use_cache: bool,
) -> dict[str, Any]:
    """贪心回答一题，返回这一题的完整记录。"""
    prompt_ids = tokenizer.encode(prompt)
    ids = torch.tensor([prompt_ids], device=device)  # (1, T)
    with autocast_context(device):
        out = generate(  # (1, T + n)，n ≤ max_new_tokens
            model,
            ids,
            max_new_tokens,
            top_k=1,  # 只留分数最高的一个，就是贪心
            use_cache=use_cache,
            eos_token_id=eos_token_id,
        )
    new_ids = out[0, ids.size(1) :].tolist()  # n 个新 token
    stopped_by_eos = new_ids[-1] == eos_token_id
    # 原始文本整段留着，将来可以换规则重新打分；eos 不算文本
    output = tokenizer.decode(new_ids[:-1] if stopped_by_eos else new_ids)
    pred = extract_answer(truncate_continuation(output))
    gold = extract_answer(example.answer)
    if gold is None:
        # 否则 None == None，模型没答出也会算对
        raise ValueError(f"标准答案里没有数：{example.answer!r}")
    return {
        "prompt": prompt,
        "prompt_tokens": len(prompt_ids),
        "output": output,
        "new_tokens": len(new_ids),
        "stopped_by_eos": stopped_by_eos,
        "pred": pred,
        "gold": gold,
        "correct": pred == gold,
    }


def evaluate_gsm8k(
    model: nn.Module,
    tokenizer: Tokenizer,
    questions: Sequence[tuple[int, Example]],
    build_prompt: Callable[[str], str],
    max_new_tokens: int,
    eos_token_id: int | None,
    device: torch.device,
    use_cache: bool = True,
    log_every: int | None = None,
) -> dict[str, Any]:
    """一次一题地评测。questions 是 (在数据集里的下标, 题目)。

    模型没有 attention mask，长度不同的题不能直接拼成一批，所以一次一题，最简单。
    """
    start = time.perf_counter()
    items: list[dict[str, Any]] = []
    for index, example in questions:
        prompt = build_prompt(example.question)
        item = answer_question(
            model,
            tokenizer,
            example,
            prompt,
            max_new_tokens,
            eos_token_id,
            device,
            use_cache,
        )
        items.append({"index": index, **item})
        if log_every and len(items) % log_every == 0:
            correct = sum(i["correct"] for i in items)
            print(
                f"已答 {len(items)}/{len(questions)}，目前答对 {correct}，"
                f"用时 {time.perf_counter() - start:.0f} 秒",
                flush=True,
            )

    seconds = time.perf_counter() - start
    new_tokens = sum(item["new_tokens"] for item in items)
    golds = [item["gold"] for item in items]
    return {
        "accuracy": accuracy([item["pred"] for item in items], golds),
        "correct": sum(item["correct"] for item in items),
        "total": len(items),
        "new_tokens": new_tokens,
        "seconds": seconds,
        "tokens_per_second": new_tokens / seconds,
        "peak_memory_mb": peak_memory_mb(device),
        "items": items,
    }
