"""GSM8K 的数据脚本、F1 配置和评测脚本。都不联网，用测试里生成的小文件和小模型。"""

import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Qwen3Config, Qwen3ForCausalLM

from tlab.data.tinystories import SourceFile
from tlab.eval.gsm8k import Example, Gsm8kEvalConfig, load_eval_config, load_examples

ROOT = Path(__file__).resolve().parents[1]
EOS = 1  # 小分词器里 <|endoftext|> 的编号


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- 数据脚本 ----


def write_parquet(path: Path, rows: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), path)


def test_parquet_to_jsonl_keeps_question_and_answer(tmp_path: Path) -> None:
    prepare = load_script("prepare_gsm8k")
    source = tmp_path / "raw.parquet"
    write_parquet(
        source,
        {
            "question": ["Q1?", "问题？"],
            "answer": ["#### 1", "#### 2"],
            "other": ["不要", "不要"],
        },
    )
    out = tmp_path / "out.jsonl"

    count = prepare.parquet_to_jsonl(source, out)

    assert count == 2
    assert load_examples(out) == [Example("Q1?", "#### 1"), Example("问题？", "#### 2")]
    # 中文原样写出，不写成 \\uXXXX，打开文件能直接看
    assert "问题？" in out.read_text(encoding="utf-8")
    assert "不要" not in out.read_text(encoding="utf-8")


def test_parquet_without_the_columns_is_an_error(tmp_path: Path) -> None:
    prepare = load_script("prepare_gsm8k")
    source = tmp_path / "raw.parquet"
    write_parquet(source, {"question": ["Q1?"]})

    with pytest.raises(KeyError):
        prepare.parquet_to_jsonl(source, tmp_path / "out.jsonl")


def pin_small_file(
    monkeypatch: pytest.MonkeyPatch, prepare: ModuleType, tmp_path: Path, rows: int
) -> None:
    """写一个小 parquet 放到原始文件的位置，并把校验信息换成它的。文件已在，就不联网。"""
    filename = "main/test-00000-of-00001.parquet"
    path = tmp_path / "raw" / filename
    write_parquet(path, {"question": ["Q?"] * 2, "answer": ["#### 1"] * 2})
    data = path.read_bytes()
    source = SourceFile(filename, len(data), hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(prepare, "SOURCE_FILES", {"test": source})
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", {"test": rows})


def test_prepare_split_writes_the_jsonl_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare = load_script("prepare_gsm8k")
    pin_small_file(monkeypatch, prepare, tmp_path, rows=2)

    out = prepare.prepare_split("test", tmp_path)

    assert out == tmp_path / "test.jsonl"
    assert len(load_examples(out)) == 2


def test_prepare_split_rejects_a_wrong_row_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare = load_script("prepare_gsm8k")
    pin_small_file(monkeypatch, prepare, tmp_path, rows=3)

    with pytest.raises(ValueError, match="3"):
        prepare.prepare_split("test", tmp_path)


def test_prepare_split_rejects_a_file_that_does_not_match(tmp_path: Path) -> None:
    prepare = load_script("prepare_gsm8k")
    # 用真实的校验信息，文件却是别的内容：不应该继续往下转
    path = tmp_path / "raw" / prepare.SOURCE_FILES["test"].filename
    write_parquet(path, {"question": ["Q?"], "answer": ["#### 1"]})

    with pytest.raises(ValueError):
        prepare.prepare_split("test", tmp_path)
    assert not (tmp_path / "test.jsonl").exists()


# ---- F1 配置 ----


def test_f1_configs_differ_only_in_the_prompt_mode() -> None:
    few = load_eval_config(ROOT / "configs" / "finetune" / "f1_base_fewshot.yaml")
    zero = load_eval_config(ROOT / "configs" / "finetune" / "f1_base_zeroshot.yaml")

    assert few.prompt_mode == "few_shot"
    assert zero.prompt_mode == "zero_shot"
    assert replace(few, prompt_mode="zero_shot") == zero
    assert few.repo == "Qwen/Qwen3-0.6B-Base"
    assert few.max_new_tokens == 256
    assert few.eos_token_id == 151643
    assert few.limit is None


def test_eval_config_can_be_overridden_from_the_command_line() -> None:
    config = load_eval_config(
        ROOT / "configs" / "finetune" / "f1_base_fewshot.yaml", ["limit=5"]
    )

    assert config.limit == 5


@pytest.mark.parametrize(
    "fields",
    [
        {"prompt_mode": "two_shot"},
        {"max_new_tokens": 0},
        {"limit": 0},
    ],
)
def test_eval_config_rejects_bad_values(fields: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Gsm8kEvalConfig(**fields)


def test_eval_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("prompt_modee: few_shot\n", encoding="utf-8")

    with pytest.raises(ValueError, match="未知配置项"):
        load_eval_config(path)


# ---- 评测脚本 ----


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """结构和 Qwen3 相同的小模型，加一个词表 97 的小分词器。"""
    path = tmp_path_factory.mktemp("tiny_qwen3_eval")
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        rope_theta=1_000_000.0,
        attn_implementation="eager",
    )
    Qwen3ForCausalLM(config).save_pretrained(path)
    vocab = {"[UNK]": 0, "<|endoftext|>": EOS}
    vocab |= {f"t{i}": i for i in range(2, 97)}
    tokenizer = Tokenizer(WordLevel(vocab, "[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    return path


@pytest.fixture()
def eval_config(tmp_path: Path) -> Path:
    """小数据：训练集 5 条、测试集 6 条，评测第 1、3、5 题，例子用训练集第 0、2 条。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for split, count in (("train", 5), ("test", 6)):
        rows = [
            json.dumps({"question": f"{split}{i}?", "answer": f"#### {i}"})
            for i in range(count)
        ]
        (data_dir / f"{split}.jsonl").write_text("\n".join(rows) + "\n", "utf-8")
    subset = tmp_path / "subset.json"
    subset.write_text(
        json.dumps(
            {
                "seed": 7,
                "test_total": 6,
                "test_indices": [1, 3, 5],
                "train_total": 5,
                "few_shot_indices": [0, 2],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"repo: some/repo\nprompt_mode: few_shot\nmax_new_tokens: 4\n"
        f"eos_token_id: {EOS}\ndata_dir: {data_dir}\nsubset_file: {subset}\n"
        "device: cpu\n",
        encoding="utf-8",
    )
    return config


def run_eval(
    config: Path, model_dir: Path, out: Path, *overrides: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_gsm8k.py"),
            "--config",
            str(config),
            "--model-dir",
            str(model_dir),
            "--out",
            str(out),
            *overrides,
        ],
        capture_output=True,
        text=True,
    )


def test_eval_script_writes_a_report_with_results_and_provenance(
    tmp_path: Path, eval_config: Path, tiny_model_dir: Path
) -> None:
    out = tmp_path / "result" / "f1.json"

    done = run_eval(eval_config, tiny_model_dir, out)

    assert done.returncode == 0, done.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert [item["index"] for item in report["items"]] == [1, 3, 5]
    assert [item["gold"] for item in report["items"]] == ["1", "3", "5"]
    assert report["total"] == 3
    assert report["correct"] == sum(item["correct"] for item in report["items"])
    assert report["accuracy"] == pytest.approx(report["correct"] / 3)
    assert report["new_tokens"] == sum(item["new_tokens"] for item in report["items"])
    for item in report["items"]:
        assert item["correct"] == (item["pred"] == item["gold"])
        assert item["output"]  # 原始文本要留着，将来可以换规则重新打分
        assert item["new_tokens"] == 4  # 配置里的 max_new_tokens，小模型碰不到 eos
        assert item["stopped_by_eos"] is False
    # 例子是训练集第 0、2 条，题目是测试集第 1 条
    prompt = report["items"][0]["prompt"]
    assert prompt == (
        "Question: train0?\nAnswer: #### 0\n\n"
        "Question: train2?\nAnswer: #### 2\n\n"
        "Question: test1?\nAnswer:"
    )
    assert report["config"]["prompt_mode"] == "few_shot"
    assert report["subset_seed"] == 7
    assert report["git_commit"]
    assert report["model"] == str(tiny_model_dir)
    assert report["device"] == "cpu"
    assert report["dtype"] == "torch.float32"
    assert report["n_params"] > 0
    assert report["torch_version"] == torch.__version__
    assert report["tokens_per_second"] > 0
    assert report["peak_memory_mb"] is None  # CPU 上读不到显存峰值
    assert "准确率" in done.stdout


def test_eval_script_can_run_only_the_first_questions(
    tmp_path: Path, eval_config: Path, tiny_model_dir: Path
) -> None:
    out = tmp_path / "f1.json"

    done = run_eval(
        eval_config, tiny_model_dir, out, "limit=2", "prompt_mode=zero_shot"
    )

    assert done.returncode == 0, done.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert [item["index"] for item in report["items"]] == [1, 3]
    assert report["config"]["limit"] == 2
    assert report["items"][0]["prompt"] == "Question: test1?\nAnswer:"


def test_eval_script_stops_at_the_eos_in_the_config(
    tmp_path: Path, eval_config: Path, tiny_model_dir: Path
) -> None:
    out = tmp_path / "f1.json"
    done = run_eval(eval_config, tiny_model_dir, out)
    assert done.returncode == 0, done.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    # 小分词器里 t5 的编号就是 5。把第一题生成的第一个 token 设成 eos，它应该马上停下
    first_token = report["items"][0]["output"].split()[0]
    assert first_token.startswith("t")
    eos = first_token[1:]

    done = run_eval(eval_config, tiny_model_dir, out, f"eos_token_id={eos}")

    assert done.returncode == 0, done.stderr
    item = json.loads(out.read_text(encoding="utf-8"))["items"][0]
    assert item["stopped_by_eos"] is True
    assert item["new_tokens"] == 1
    assert item["output"] == ""


def test_eval_script_downloads_the_repo_in_the_config(
    tmp_path: Path,
    eval_config: Path,
    tiny_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = load_script("eval_gsm8k")
    asked: list[str] = []

    def fake_download(repo: str, allow_patterns: list[str]) -> str:
        asked.append(repo)
        assert "model.safetensors" in allow_patterns
        return str(tiny_model_dir)

    monkeypatch.setattr(script, "snapshot_download", fake_download)
    # device=auto 时按 cuda → mps → cpu 选，这里固定成 CPU，免得测试用到显卡
    monkeypatch.setattr(script, "get_device", lambda: torch.device("cpu"))
    out = tmp_path / "f1.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_gsm8k.py",
            "--config",
            str(eval_config),
            "--out",
            str(out),
            "device=auto",
        ],
    )

    script.main()

    assert asked == ["some/repo"]
    report = json.loads(out.read_text(encoding="utf-8"))
    # 目录名就是这份权重在 HF 上的版本号，换一台机器也知道用的是哪份
    assert report["model"] == f"some/repo@{tiny_model_dir.name}"
    assert report["config"]["device"] == "auto"
    assert report["device"] == "cpu"
