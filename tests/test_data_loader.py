import json
import math
from pathlib import Path

import numpy as np
import torch

# 按数据目录里的文件选择字符级或 BPE 的加载方式（agent 编写）

STORIES = [
    "Once upon a time, Tom met a little cat.",
    "Lily liked the big red ball. She was happy.",
] * 20


def write_char_dir(data_dir: Path) -> None:
    from tlab.data.char import CharTokenizer, save_ids, split_ids

    text = "to be or not to be, that is the question.\n" * 20
    tokenizer = CharTokenizer.from_text(text)
    train, val = split_ids(tokenizer.encode(text))
    data_dir.mkdir(parents=True)
    save_ids(train, data_dir / "train.bin")
    save_ids(val, data_dir / "val.bin")
    tokenizer.save(data_dir / "meta.json")


def write_bpe_dir(data_dir: Path):
    from tlab.data.tinystories import write_tokens
    from tlab.tokenizer.hf_bpe import train_hf_tokenizer

    tokenizer = train_hf_tokenizer(STORIES, 300)
    data_dir.mkdir(parents=True)
    tokenizer.save(data_dir / "tokenizer.json")
    write_tokens(STORIES[:30], tokenizer, data_dir / "train.bin")
    write_tokens(STORIES[30:], tokenizer, data_dir / "val.bin")
    return tokenizer


def test_char_data_dir_loads_char_tokenizer_and_tensors(tmp_path: Path) -> None:
    from tlab.data.char import CharTokenizer
    from tlab.data.loader import load_splits, load_tokenizer

    data_dir = tmp_path / "data"
    write_char_dir(data_dir)

    train, val = load_splits(data_dir)

    assert isinstance(load_tokenizer(data_dir), CharTokenizer)
    # 字符级数据很小，仍按 M1 的方式整个读进内存
    assert isinstance(train, torch.Tensor) and train.dtype == torch.long
    assert isinstance(val, torch.Tensor) and val.dtype == torch.long


def test_bpe_data_dir_loads_hf_tokenizer_and_memmaps(tmp_path: Path) -> None:
    from tlab.data.loader import load_splits, load_tokenizer
    from tlab.tokenizer.hf_bpe import HFTokenizer

    data_dir = tmp_path / "data"
    tokenizer = write_bpe_dir(data_dir)

    loaded = load_tokenizer(data_dir)
    train, val = load_splits(data_dir)

    assert isinstance(loaded, HFTokenizer)
    assert loaded.vocab_size == tokenizer.vocab_size
    # BPE 数据可能很大：用 memmap，按需从硬盘读
    assert isinstance(train, np.memmap) and isinstance(val, np.memmap)
    assert int(train[-1]) == tokenizer.eot_id


def test_run_trains_on_bpe_data_dir(tmp_path: Path) -> None:
    from tlab.train.config import TrainConfig
    from tlab.train.run import run

    tokenizer = write_bpe_dir(tmp_path / "data")
    config = TrainConfig(
        data_dir=str(tmp_path / "data"),
        out_dir=str(tmp_path / "out"),
        n_layers=1,
        n_heads=2,
        d_model=32,
        block_size=8,
        batch_size=4,
        max_iters=3,
        eval_interval=2,
        eval_iters=1,
        seed=0,
        device="cpu",
    )

    out_dir = run(config)

    # 模型的词表大小来自 tokenizer.json，而不是字符级的 meta.json
    ckpt = torch.load(out_dir / "ckpt.pt", weights_only=False)
    assert ckpt["vocab_size"] == tokenizer.vocab_size
    meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert math.isfinite(meta["final_val_loss"])
