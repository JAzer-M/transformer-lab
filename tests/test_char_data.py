from pathlib import Path

import pytest
import torch

# 待实现模块在各测试内导入，与其他测试文件保持一致


def test_vocab_is_sorted_unique_characters() -> None:
    from tlab.data.char import CharTokenizer

    tokenizer = CharTokenizer.from_text("hello")

    # 字表按字符排序，同一段文本总是得到同一个字表
    assert tokenizer.chars == ["e", "h", "l", "o"]
    assert tokenizer.vocab_size == 4


def test_encode_then_decode_restores_text() -> None:
    from tlab.data.char import CharTokenizer

    text = "First Citizen:\nBefore we proceed any further, hear me speak.\n"
    tokenizer = CharTokenizer.from_text(text)

    ids = tokenizer.encode(text)

    assert all(isinstance(i, int) for i in ids)
    assert len(ids) == len(text)
    assert tokenizer.decode(ids) == text


def test_encode_rejects_unknown_character() -> None:
    from tlab.data.char import CharTokenizer

    tokenizer = CharTokenizer.from_text("abc")

    # 字表里没有的字符没有编号，应直接报错，而不是悄悄编成别的字
    with pytest.raises(KeyError):
        tokenizer.encode("abz")


def test_tokenizer_save_and_load(tmp_path: Path) -> None:
    from tlab.data.char import CharTokenizer

    tokenizer = CharTokenizer.from_text("To be, or not to be\n")
    path = tmp_path / "meta.json"

    tokenizer.save(path)
    loaded = CharTokenizer.load(path)

    assert loaded.chars == tokenizer.chars


def test_split_keeps_order_and_ratio() -> None:
    from tlab.data.char import split_ids

    ids = list(range(100))

    train, val = split_ids(ids, val_fraction=0.1)  # (90,), (10,)

    assert train.dtype == torch.long and val.dtype == torch.long
    assert len(train) == 90 and len(val) == 10
    # 前 90% 做训练集、后 10% 做验证集，顺序不打乱，拼回去就是原序列
    assert torch.cat([train, val]).tolist() == ids


def test_ids_file_round_trip(tmp_path: Path) -> None:
    from tlab.data.char import load_ids, save_ids

    ids = torch.tensor([0, 1, 64, 7, 7, 3])  # (N,)
    path = tmp_path / "train.bin"

    save_ids(ids, path)
    loaded = load_ids(path)  # (N,)

    assert loaded.dtype == torch.long
    assert torch.equal(loaded, ids)


def test_get_batch_targets_are_inputs_shifted_by_one() -> None:
    from tlab.data.char import get_batch

    # 用 0, 1, 2, ... 当数据：每个位置的“下一个 token”恰好是自己加 1
    data = torch.arange(100)  # (N,)
    generator = torch.Generator().manual_seed(0)

    x, y = get_batch(data, batch_size=4, block_size=8, generator=generator)

    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert x.dtype == torch.long and y.dtype == torch.long
    # 每一行都是原文中连续的一段
    assert torch.equal(x[:, 1:] - x[:, :-1], torch.ones(4, 7, dtype=torch.long))
    # y 就是 x 往后挪一位
    assert torch.equal(y, x + 1)
    # 截取的窗口不会越过数据末尾
    assert int(y.max()) <= 99


def test_get_batch_is_reproducible_with_same_seed() -> None:
    from tlab.data.char import get_batch

    data = torch.arange(1000)  # (N,)

    x1, _ = get_batch(data, 8, 16, generator=torch.Generator().manual_seed(42))
    x2, _ = get_batch(data, 8, 16, generator=torch.Generator().manual_seed(42))

    assert torch.equal(x1, x2)
