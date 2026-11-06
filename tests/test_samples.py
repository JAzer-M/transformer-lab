import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch

# 待实现模块在各测试内导入，与其他测试文件保持一致

if TYPE_CHECKING:
    from tlab.eval.samples import SampleParams

ROOT = Path(__file__).resolve().parents[1]


def train_tiny_run(root: Path, variant: str, seed: int = 1337, **overrides) -> Path:
    """在字符级小数据上训 2 步，存档放在 <root>/<变体>/seed<种子>/ckpt.pt。"""
    from tlab.data.char import CharTokenizer, save_ids, split_ids
    from tlab.train.config import TrainConfig
    from tlab.train.run import run

    text = "to be or not to be, that is the question.\n" * 20
    tokenizer = CharTokenizer.from_text(text)
    train, val = split_ids(tokenizer.encode(text))
    data_dir = root / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        save_ids(train, data_dir / "train.bin")
        save_ids(val, data_dir / "val.bin")
        tokenizer.save(data_dir / "meta.json")
    fields = {
        "data_dir": str(data_dir),
        "out_dir": str(root / variant / f"seed{seed}"),
        "n_layers": 1,
        "n_heads": 2,
        "d_model": 16,
        "block_size": 64,
        "batch_size": 2,
        "max_iters": 2,
        "eval_interval": 1,
        "eval_iters": 1,
        "device": "cpu",
        "seed": seed,
    }
    fields.update(overrides)
    return run(TrainConfig(**fields))


def params() -> "SampleParams":
    from tlab.eval.samples import SampleParams

    return SampleParams(
        prompt="to be", max_new_tokens=8, temperature=0.8, top_k=5, seeds=[1, 2]
    )


def test_same_seed_gives_the_same_text_and_matches_no_cache(tmp_path: Path) -> None:
    from tlab.data.loader import load_tokenizer
    from tlab.eval.generate import generate
    from tlab.eval.samples import sample_checkpoint
    from tlab.train.utils import load_model

    out_dir = train_tiny_run(tmp_path, "a1_rope", pos_encoding="rope")
    device = torch.device("cpu")

    first = sample_checkpoint(out_dir / "ckpt.pt", params(), device)
    second = sample_checkpoint(out_dir / "ckpt.pt", params(), device)

    assert first == second
    assert first["ckpt_step"] == 2
    assert list(first["texts"]) == [1, 2]
    assert all(text.startswith("to be") for text in first["texts"].values())
    # 和不用缓存、同一个种子生成的文字一样
    model, ckpt = load_model(out_dir / "ckpt.pt", device)
    tokenizer = load_tokenizer(Path(ckpt["config"]["data_dir"]))
    prompt = torch.tensor([tokenizer.encode("to be")])  # (1, T)
    plain = generate(
        model,
        prompt,
        8,
        temperature=0.8,
        top_k=5,
        generator=torch.Generator().manual_seed(1),
    )  # (1, T + 8)
    assert first["texts"][1] == tokenizer.decode(plain[0].tolist())


def test_sampling_uses_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tlab.eval import samples

    out_dir = train_tiny_run(tmp_path, "a0_control")
    calls: list[bool] = []
    real_generate = samples.generate

    def spying_generate(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(kwargs["use_cache"])
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(samples, "generate", spying_generate)

    samples.sample_checkpoint(out_dir / "ckpt.pt", params(), torch.device("cpu"))

    assert calls == [True, True]


def test_sampling_rejects_a_checkpoint_from_before_the_end(tmp_path: Path) -> None:
    from tlab.eval.samples import sample_checkpoint

    out_dir = train_tiny_run(tmp_path, "a0_control")
    ckpt = torch.load(out_dir / "ckpt.pt", weights_only=False)
    ckpt["step"] = 1
    torch.save(ckpt, out_dir / "ckpt.pt")

    # 用训到一半的存档生成，样例就不代表结果表里的模型了
    with pytest.raises(ValueError, match="max_iters"):
        sample_checkpoint(out_dir / "ckpt.pt", params(), torch.device("cpu"))


def test_rendered_markdown_records_everything_needed_to_reproduce() -> None:
    from tlab.eval.samples import render_samples

    results = {"a1_rope": {"ckpt_step": 3052, "texts": {1: "once", 2: "twice"}}}

    text = render_samples(results, params(), "abc123", "cuda", run_seed=1337)

    for expected in (
        "abc123",
        "cuda",
        "1337",
        "3052",
        "to be",
        "8",
        "0.8",
        "5",
        "KV 缓存",
    ):
        assert expected in text
    assert "## a1_rope" in text
    assert "once" in text and "twice" in text


def run_script(
    tmp_path: Path, runs: Path, *variants: str
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "sample_ablations.py"),
            "--runs",
            str(runs),
            "--out",
            str(tmp_path / "samples.md"),
            "--device",
            "cpu",
            "--prompt",
            "to be",
            "--max-new-tokens",
            "4",
            "--seeds",
            "1",
            "--variants",
            *variants,
        ],
        capture_output=True,
        text=True,
    )


def test_script_writes_one_section_per_variant(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    train_tiny_run(runs, "a0_control")
    train_tiny_run(runs, "a1_rope", pos_encoding="rope")
    # 输出目录不存在时脚本要自己建
    out = tmp_path / "not" / "yet"

    done = run_script(out, runs, "a0_control", "a1_rope")

    assert done.returncode == 0, done.stderr
    text = (out / "samples.md").read_text(encoding="utf-8")
    assert "## a0_control" in text and "## a1_rope" in text


def test_script_names_the_missing_checkpoint_before_generating_anything(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    train_tiny_run(runs, "a0_control")

    done = run_script(tmp_path, runs, "a0_control", "a4_mqa")

    assert done.returncode != 0
    assert "a4_mqa" in done.stderr
    assert not (tmp_path / "samples.md").exists()
    # 第一个变体也没有开始生成：先确认存档都在
    assert "a0_control" not in done.stdout


def test_default_variants_are_every_planned_variant() -> None:
    from tlab.eval.ablation import CONTROL, GROUPS, REFERENCE, STRESS
    from tlab.eval.samples import VARIANTS

    planned = {CONTROL, *STRESS, *REFERENCE, *(n for g in GROUPS.values() for n in g)}
    assert set(VARIANTS) == planned
    assert len(VARIANTS) == 12
