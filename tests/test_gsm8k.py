"""GSM8K 评测：答案抽取、提示模板、准确率、固定的题目。预期值都是手写的具体例子。"""

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch import nn

from tlab.eval.gsm8k import (
    Example,
    accuracy,
    evaluate_gsm8k,
    extract_answer,
    few_shot_prompt,
    generate_answer,
    load_examples,
    make_eval_subset,
    strip_calculator,
    truncate_continuation,
    zero_shot_prompt,
)
from tlab.model.gpt import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[1]
SUBSET_FILE = ROOT / "configs" / "finetune" / "gsm8k_eval_subset.json"


# ---- 答案抽取 ----


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 标准格式：#### 后面的数
        ("16-3-4=9\n9*2=18\n#### 18", "18"),
        # 有多个 #### 时取第一个：后面的可能是模型接着编的
        ("#### 18\nmore words\n#### 7", "18"),
        # 没有 #### 时取最后一个数
        ("so 3 + 4 = 7 apples.", "7"),
        # 千分位逗号和美元符号
        ("#### $1,000", "1000"),
        ("#### 12,345,678", "12345678"),
        # 18.00 和 18 是同一个答案；18.50 和 18.5 也是
        ("#### 18.00", "18"),
        ("#### 18.50", "18.5"),
        # 前面的省略号不是小数点
        ("so ...18", "18"),
        # 句末的句号不是小数点
        ("The answer is 18.", "18"),
        # 负数
        ("#### -5", "-5"),
        ("It lost -$5", "-5"),
        ("#### 0.0", "0"),
        ("#### -0", "0"),
        # 数字中间的减号不是负号
        ("so we get 10-5", "5"),
        # #### 后面没有数时，退回到取最后一个数
        ("3 apples #### none", "3"),
        # 没有 #### 时，先看 The answer is：基座不给例子时常这样结尾，后面可能还有别的数
        ("The answer is 55 after 3 more days", "55"),
        ("The answer is $1,280.", "1280"),
        ("The Answer is 60 after 3 days", "60"),
        ("The answer is: 7 not 8", "7"),
        ("the answer is **7** not 8", "7"),
        # 有多个时取第一个；#### 优先
        ("The answer is 7. The answer is 8", "7"),
        ("#### 18\nThe answer is 7", "18"),
        # answer is 后面紧跟的不是数：退回到取最后一个数
        ("The answer is five, 3 more than 2", "2"),
        # 没有数字：没答出
        ("no digits at all", None),
        ("", None),
    ],
)
def test_extract_answer(text: str, expected: str | None) -> None:
    assert extract_answer(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A #### 18\nQuestion: foo #### 7", "A #### 18"),
        # few-shot 的例子之间隔一个空行，接着编题时也常这样
        ("A #### 18\n\nQuestion: foo", "A #### 18"),
        # 接着编了好几题时，也是在第一处截断
        ("A #### 18\nQuestion: b #### 7\nQuestion: c", "A #### 18"),
        ("A #### 18", "A #### 18"),
        ("\nQuestion: foo", ""),
        # 基座不给例子时的几种续写形式（2026-11-07 试跑时见到的）
        ("The answer is 55.\n[Question]A car", "The answer is 55."),
        ("x 7\n\n[Question] foo", "x 7"),
        ("x 7\nQ: foo 9", "x 7"),
        ("x 7\n  Question: foo", "x 7"),
        # 不是续写：换行后只是以这些字母开头
        ("x 7\nQuestions are hard", "x 7\nQuestions are hard"),
        ("x 7\nQuite so", "x 7\nQuite so"),
        # 只有换行之后的 Question: 才是接着编的新题
        ("Question: inside the answer", "Question: inside the answer"),
    ],
)
def test_truncate_continuation(text: str, expected: str) -> None:
    assert truncate_continuation(text) == expected


def test_truncating_first_keeps_numbers_of_the_invented_question_out() -> None:
    # 不截断的话，“最后一个数”会取到模型自己编的题里的 9
    text = "so 3 + 4 = 7\n\nQuestion: how many are 9"

    assert extract_answer(text) == "9"
    assert extract_answer(truncate_continuation(text)) == "7"


def test_strip_calculator_removes_every_annotation() -> None:
    answer = "16-3-4=<<16-3-4=9>>9\n9*2=<<9*2=18>>18\n#### 18"

    assert strip_calculator(answer) == "16-3-4=9\n9*2=18\n#### 18"


# ---- 提示模板 ----


def test_zero_shot_prompt_ends_right_after_the_colon() -> None:
    # 不能以空格结尾：训练时是 "Answer: {a}"，答案前的空格要和答案一起分词
    assert zero_shot_prompt("How many?") == "Question: How many?\nAnswer:"


def test_few_shot_prompt_is_exactly_the_examples_then_the_question() -> None:
    examples = [
        Example("Q1?", "1+1=<<1+1=2>>2\n#### 2"),
        Example("Q2?", "#### 3"),
    ]

    assert few_shot_prompt("How many?", examples) == (
        "Question: Q1?\nAnswer: 1+1=2\n#### 2\n\n"
        "Question: Q2?\nAnswer: #### 3\n\n"
        "Question: How many?\nAnswer:"
    )


# ---- 准确率 ----


def test_accuracy_counts_unanswered_as_wrong() -> None:
    assert accuracy(["18", "7", None], ["18", "7", "1"]) == pytest.approx(2 / 3)


def test_accuracy_rejects_mismatched_lengths_and_empty_input() -> None:
    with pytest.raises(ValueError):
        accuracy(["1"], ["1", "2"])
    with pytest.raises(ValueError):
        accuracy([], [])


# ---- 固定的题目 ----


def test_committed_subset_has_500_distinct_test_questions() -> None:
    subset = json.loads(SUBSET_FILE.read_text(encoding="utf-8"))
    indices = subset["test_indices"]

    assert len(indices) == 500
    assert len(set(indices)) == 500
    assert min(indices) >= 0
    assert max(indices) <= 1318  # 测试集 1,319 条


def test_committed_subset_has_4_distinct_few_shot_examples() -> None:
    subset = json.loads(SUBSET_FILE.read_text(encoding="utf-8"))
    indices = subset["few_shot_indices"]

    assert len(indices) == 4
    assert len(set(indices)) == 4
    assert min(indices) >= 0
    assert max(indices) <= 7472  # 训练集 7,473 条


def test_committed_subset_can_be_regenerated_from_its_seed() -> None:
    subset = json.loads(SUBSET_FILE.read_text(encoding="utf-8"))

    assert make_eval_subset(subset["seed"]) == subset


def test_different_seeds_choose_different_questions() -> None:
    assert make_eval_subset(1)["test_indices"] != make_eval_subset(2)["test_indices"]


# ---- 评测流程 ----

EOS = 256  # 假分词器的词表是 0–255 这些字节，256 留给 eos


class ByteTokenizer:
    """每个字节一个 token，不用下载也能测流程。"""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


class ScriptedModel(nn.Module):
    """按提示决定回答：把回答一个字节一个字节地“生成”出来，写完后给 eos。"""

    block_size = 4096

    def __init__(self, replies: dict[str, str]) -> None:
        super().__init__()
        self.replies = {
            prompt: list(reply.encode("utf-8")) for prompt, reply in replies.items()
        }

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        text = bytes(idx[0].tolist()).decode("utf-8")  # idx 是 (1, T)
        prompt = next(p for p in self.replies if text.startswith(p))
        written = len(text.encode("utf-8")) - len(prompt.encode("utf-8"))
        reply = self.replies[prompt]
        next_id = reply[written] if written < len(reply) else EOS
        logits = torch.zeros(1, idx.size(1), EOS + 1)  # (1, T, V)
        logits[0, -1, next_id] = 1.0
        return logits


def test_evaluate_scores_every_question_and_records_what_was_generated() -> None:
    questions = [
        (7, Example("Right?", "#### 18")),  # 答对，但接着编了一题
        (9, Example("Wrong?", "#### 5")),  # 答错
        (20, Example("Silent?", "#### 1")),  # 没有数字
    ]
    model = ScriptedModel(
        {
            "Question: Right?\nAnswer:": " 9*2=18\n#### 18\n\nQuestion: other #### 7",
            "Question: Wrong?\nAnswer:": " 3 + 4 = 7",
            "Question: Silent?\nAnswer:": " no idea",
        }
    )

    result = evaluate_gsm8k(
        model,
        ByteTokenizer(),
        questions,
        zero_shot_prompt,
        max_new_tokens=64,
        eos_token_id=EOS,
        device=torch.device("cpu"),
        use_cache=False,
    )

    assert result["correct"] == 1
    assert result["total"] == 3
    assert result["accuracy"] == pytest.approx(1 / 3)
    first, second, third = result["items"]
    assert first["index"] == 7
    assert first["prompt"] == "Question: Right?\nAnswer:"
    assert first["prompt_tokens"] == 24  # 字节数
    # 原始生成文本都留着，将来可以换规则重新打分；eos 不在文本里
    assert first["output"] == " 9*2=18\n#### 18\n\nQuestion: other #### 7"
    assert first["new_tokens"] == 40  # 39 个字节 + eos
    assert first["stopped_by_eos"] is True
    assert first["pred"] == "18"  # 不是后面编的 7
    assert first["gold"] == "18"
    assert first["correct"] is True
    assert (second["pred"], second["gold"], second["correct"]) == ("7", "5", False)
    assert (third["pred"], third["correct"]) == (None, False)
    assert result["new_tokens"] == sum(item["new_tokens"] for item in result["items"])
    assert result["tokens_per_second"] > 0
    assert "peak_memory_mb" in result


def test_generate_answer_needs_no_gold_answer() -> None:
    # 自己提问时没有标准答案：只续写、抽答案，不打分
    model = ScriptedModel(
        {"Question: Right?\nAnswer:": " 9*2=18\n#### 18\n\nQuestion: other #### 7"}
    )

    result = generate_answer(
        model,
        ByteTokenizer(),
        "Question: Right?\nAnswer:",
        max_new_tokens=64,
        eos_token_id=EOS,
        device=torch.device("cpu"),
        use_cache=False,
    )

    assert result == {
        "prompt_tokens": 24,
        "output": " 9*2=18\n#### 18\n\nQuestion: other #### 7",
        "new_tokens": 40,  # 39 个字节 + eos
        "stopped_by_eos": True,
        "pred": "18",  # 抽之前先截掉编出来的下一题
    }


def test_generate_answer_cut_off_by_the_token_limit() -> None:
    model = ScriptedModel({"Q": " 9*2=18"})

    result = generate_answer(
        model,
        ByteTokenizer(),
        "Q",
        max_new_tokens=5,
        eos_token_id=EOS,
        device=torch.device("cpu"),
        use_cache=False,
    )

    assert result["output"] == " 9*2="
    assert result["new_tokens"] == 5
    assert result["stopped_by_eos"] is False
    assert result["pred"] == "2"  # 没写完时只能取到最后一个数


def test_evaluate_ignores_numbers_of_the_invented_question() -> None:
    # 没有 #### 时取最后一个数，所以一定要先截断，否则会取到编出来的 9
    model = ScriptedModel(
        {"Question: Sum?\nAnswer:": " 3 + 4 = 7\n\nQuestion: how many are 9"}
    )

    result = evaluate_gsm8k(
        model,
        ByteTokenizer(),
        [(0, Example("Sum?", "#### 7"))],
        zero_shot_prompt,
        max_new_tokens=64,
        eos_token_id=EOS,
        device=torch.device("cpu"),
        use_cache=False,
    )

    assert result["items"][0]["pred"] == "7"
    assert result["accuracy"] == 1.0


def test_evaluate_reads_the_answer_before_a_bracketed_continuation() -> None:
    # 试跑里的真实形式：第一题答对了 60，接着用 [Question] 编题，不能取到编出来的数
    model = ScriptedModel(
        {
            "Question: Dance?\nAnswer:": " 12 / 20 * 100 = 60%. The answer is 60.\n"
            "[Question]A group of 10 friends\n[Answer]10 / 5 = 2"
        }
    )

    result = evaluate_gsm8k(
        model,
        ByteTokenizer(),
        [(14, Example("Dance?", "#### 60"))],
        zero_shot_prompt,
        max_new_tokens=128,
        eos_token_id=EOS,
        device=torch.device("cpu"),
        use_cache=False,
    )

    assert result["items"][0]["pred"] == "60"
    assert result["accuracy"] == 1.0


def test_prompt_tokens_counts_tokens_not_characters() -> None:
    prompt = "Question: 几个？\nAnswer:"
    model = ScriptedModel({prompt: " #### 1"})

    result = evaluate_gsm8k(
        model,
        ByteTokenizer(),
        [(0, Example("几个？", "#### 1"))],
        zero_shot_prompt,
        max_new_tokens=16,
        eos_token_id=EOS,
        device=torch.device("cpu"),
        use_cache=False,
    )

    # 每个字节一个 token：10 + 9 + 1 + 7 = 27；如果数字符就是 21
    assert result["items"][0]["prompt_tokens"] == 27


def test_evaluate_marks_answers_cut_off_by_the_token_limit() -> None:
    model = ScriptedModel({"Question: Long?\nAnswer:": " 1 2 3 4 5 6"})

    result = evaluate_gsm8k(
        model,
        ByteTokenizer(),
        [(0, Example("Long?", "#### 3"))],
        zero_shot_prompt,
        max_new_tokens=4,
        eos_token_id=EOS,
        device=torch.device("cpu"),
        use_cache=False,
    )

    item = result["items"][0]
    assert item["output"] == " 1 2"
    assert item["new_tokens"] == 4
    assert item["stopped_by_eos"] is False
    assert item["pred"] == "2"


def test_evaluate_gives_the_same_outputs_with_and_without_cache() -> None:
    torch.manual_seed(0)
    model = GPT(
        GPTConfig(
            vocab_size=EOS + 1,
            block_size=128,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pos_encoding="rope",
        )
    )
    questions = [(0, Example("Q1?", "#### 1")), (1, Example("Q2 longer?", "#### 2"))]

    runs = [
        evaluate_gsm8k(
            model,
            ByteTokenizer(),
            questions,
            zero_shot_prompt,
            max_new_tokens=12,
            eos_token_id=EOS,
            device=torch.device("cpu"),
            use_cache=use_cache,
        )
        for use_cache in (False, True)
    ]

    assert [i["output"] for i in runs[0]["items"]] == [
        i["output"] for i in runs[1]["items"]
    ]
    assert [i["new_tokens"] for i in runs[0]["items"]] == [12, 12]


def test_evaluate_passes_the_prompt_builder_each_question() -> None:
    shots = [Example("Q?", "#### 2")]
    prompt = few_shot_prompt("Real?", shots)
    model = ScriptedModel({prompt: " #### 4"})

    result = evaluate_gsm8k(
        model,
        ByteTokenizer(),
        [(3, Example("Real?", "#### 4"))],
        lambda question: few_shot_prompt(question, shots),
        max_new_tokens=16,
        eos_token_id=EOS,
        device=torch.device("cpu"),
        use_cache=False,
    )

    assert result["items"][0]["prompt"] == prompt
    assert result["accuracy"] == 1.0


def test_load_examples_reads_question_and_answer_per_line(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"question": "Q1?", "answer": "#### 1"}\n'
        '{"question": "问题？", "answer": "#### 2"}\n',
        encoding="utf-8",
    )

    assert load_examples(path) == [
        Example("Q1?", "#### 1"),
        Example("问题？", "#### 2"),
    ]


def test_a_gold_answer_without_a_number_is_an_error() -> None:
    # 否则 None == None，模型没答出也会算对
    model = ScriptedModel({"Question: Q?\nAnswer:": " no idea"})

    with pytest.raises(ValueError, match="标准答案"):
        evaluate_gsm8k(
            model,
            ByteTokenizer(),
            [(0, Example("Q?", "no number here"))],
            zero_shot_prompt,
            max_new_tokens=16,
            eos_token_id=EOS,
            device=torch.device("cpu"),
            use_cache=False,
        )


def test_evaluate_reports_progress_every_few_questions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = ScriptedModel({"Question: Q?\nAnswer:": " #### 1"})
    questions = [(i, Example("Q?", "#### 1")) for i in range(5)]

    evaluate_gsm8k(
        model,
        ByteTokenizer(),
        questions,
        zero_shot_prompt,
        max_new_tokens=16,
        eos_token_id=EOS,
        device=torch.device("cpu"),
        use_cache=False,
        log_every=2,
    )

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2  # 第 2、4 题之后各一行
    assert "2/5" in lines[0]
    assert "4/5" in lines[1]
    assert "答对 4" in lines[1]


def test_evaluate_uses_the_cache_by_default() -> None:
    torch.manual_seed(0)
    model = GPT(
        GPTConfig(
            vocab_size=EOS + 1,
            block_size=128,
            d_model=32,
            n_layers=2,
            n_heads=4,
            pos_encoding="rope",
        )
    )
    lengths: list[int] = []
    # 记下每一步送进模型的长度 T'：用了缓存，第一步之后每步只送 1 个 token
    model.register_forward_pre_hook(lambda _, args: lengths.append(args[0].size(1)))

    evaluate_gsm8k(
        model,
        ByteTokenizer(),
        [(0, Example("Q?", "#### 1"))],
        zero_shot_prompt,
        max_new_tokens=3,
        eos_token_id=EOS,
        device=torch.device("cpu"),
    )

    assert lengths == [len("Question: Q?\nAnswer:"), 1, 1]


def test_evaluate_generates_inside_the_autocast_of_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CPU 上 autocast_context 什么也不做，所以换成会记录的版本，确认生成确实在它里面
    entered: list[torch.device] = []
    calls_inside: list[bool] = []

    @contextlib.contextmanager
    def recording(device: torch.device) -> Iterator[None]:
        entered.append(device)
        calls_inside.append(True)
        yield
        calls_inside.append(False)

    monkeypatch.setattr("tlab.eval.gsm8k.autocast_context", recording)
    model = ScriptedModel({"Question: Q?\nAnswer:": " #### 1"})
    original_forward = model.forward

    def checked_forward(idx: torch.Tensor) -> torch.Tensor:
        assert calls_inside[-1] is True, "模型要在 autocast 里运行"
        return original_forward(idx)

    model.forward = checked_forward  # type: ignore[method-assign]

    evaluate_gsm8k(
        model,
        ByteTokenizer(),
        [(0, Example("Q?", "#### 1"))],
        zero_shot_prompt,
        max_new_tokens=16,
        eos_token_id=EOS,
        device=torch.device("cpu"),
        use_cache=False,
    )

    assert entered == [torch.device("cpu")]
