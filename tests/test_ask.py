"""自己提问的脚本 scripts/ask.py。

用 conftest.py 里结构和 Qwen3 相同的小模型，不联网。
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import ROOT, TINY_EOS, big_lora_model, load_script, write_eval_config

from tlab.finetune.lora_model import save_lora


def ask(
    model_dir: Path, *args: str, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ask.py"),
            "--model-dir",
            str(model_dir),
            "--eos-token-id",
            str(TINY_EOS),
            "--max-new-tokens",
            "6",
            "--device",
            "cpu",
            *args,
        ],
        input=stdin,
        capture_output=True,
        text=True,
    )


def eval_first_question(directory: Path, model_dir: Path, lora: Path | None) -> dict:
    """用评测脚本答小数据里的第 3 题，返回这一题的记录。"""
    config = write_eval_config(directory, lora, test_indices=(3,))
    out = directory / "eval.json"
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_gsm8k.py"),
            "--config",
            str(config),
            "--model-dir",
            str(model_dir),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    (item,) = json.loads(out.read_text(encoding="utf-8"))["items"]
    return item


def question_text(tmp_path: Path, index: int) -> str:
    lines = (tmp_path / "data" / "test.jsonl").read_text(encoding="utf-8").splitlines()
    return json.loads(lines[index])["question"]


def test_ask_answers_exactly_like_the_eval(
    tmp_path: Path, tiny_model_dir: Path
) -> None:
    lora = tmp_path / "lora.pt"
    save_lora(big_lora_model(tiny_model_dir), lora)

    outputs = {}
    for name, path in (("base", None), ("lora", lora)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        item = eval_first_question(run_dir, tiny_model_dir, path)
        assert item["output"], "小模型一上来就给了 eos，这个测试看不出区别"
        extra = ["--lora", str(path)] if path else []
        done = ask(tiny_model_dir, *extra, question_text(run_dir, 3))

        assert done.returncode == 0, done.stderr
        # 自己提问看到的，就是评测时模型的回答：同样的提示格式、贪心、截断和抽答案。
        # 小分词器不认识 Question、Answer，换了格式也编成同样的 [UNK]，所以提示要单独核对
        assert f"提示：{item['prompt']!r}\n" in done.stdout
        assert f"回答：{item['output']}\n" in done.stdout
        assert f"抽出的答案：{item['pred']}\n" in done.stdout
        outputs[name] = item["output"]

    assert outputs["base"] != outputs["lora"]


def test_without_a_question_it_asks_until_an_empty_line(tiny_model_dir: Path) -> None:
    done = ask(tiny_model_dir, stdin="t2 ?\nt3 ?\n\nt4 ?\n")

    assert done.returncode == 0, done.stderr
    # 空行就退出，后面的 t4 不再回答
    assert done.stdout.count("回答：") == 2


def test_end_of_input_also_ends_the_questions(tiny_model_dir: Path) -> None:
    done = ask(tiny_model_dir, stdin="t2 ?\n")

    assert done.returncode == 0, done.stderr
    assert done.stdout.count("回答：") == 1


def test_missing_lora_file_is_reported(tmp_path: Path, tiny_model_dir: Path) -> None:
    done = ask(tiny_model_dir, "--lora", str(tmp_path / "nope.pt"), "t2 ?")

    assert done.returncode != 0
    assert "nope.pt" in done.stderr


def test_reply_shows_only_the_part_before_the_invented_question() -> None:
    script = load_script("ask")
    reply = {
        "output": " 3*12=36\n#### 36\n\nQuestion: next one",
        "new_tokens": 30,
        "stopped_by_eos": False,
        "pred": "36",
    }

    # 截掉的是 "\n\nQuestion: next one"：2 个换行 + 18 个字符
    text = script.describe_reply(reply, seconds=2.34)

    assert text == (
        "回答： 3*12=36\n#### 36\n"
        "（后面又编了 20 个字符的新题，没显示）\n"
        "抽出的答案：36\n"
        "到了长度上限还没停；生成 30 个 token，用时 2.3 秒"
    )


def test_reply_without_a_number() -> None:
    script = load_script("ask")
    reply = {
        "output": " no idea",
        "new_tokens": 3,
        "stopped_by_eos": True,
        "pred": None,
    }

    assert script.describe_reply(reply, seconds=0.5) == (
        "回答： no idea\n抽出的答案：没有找到数\n在 eos 停下；生成 3 个 token，用时 0.5 秒"
    )


def test_missing_lora_is_found_before_looking_for_the_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script("ask")

    def must_not_download(*args: object) -> None:
        raise AssertionError("LoRA 路径不对时不该去找或下载基座")

    monkeypatch.setattr(script, "resolve_model_dir", must_not_download)
    monkeypatch.setattr(
        sys, "argv", ["ask.py", "--lora", str(tmp_path / "nope.pt"), "t2 ?"]
    )

    with pytest.raises(FileNotFoundError, match="nope.pt"):
        script.main()
