import hashlib
from pathlib import Path

import numpy as np
import pytest

# TinyStories 预处理；待实现模块在各测试内导入


class ByteTokenizer:
    """测试用的假词表：每个字节就是一个 token，256 是故事结束标记。"""

    def __init__(self, vocab_size: int = 257) -> None:
        self.vocab_size = vocab_size
        self.eot_id = 256

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [list(t.encode("utf-8")) for t in texts]


# ---------- 拆分故事 ----------


def test_iter_stories_splits_on_eot_and_skips_blanks(tmp_path: Path) -> None:
    from tlab.data.tinystories import iter_stories

    path = tmp_path / "stories.txt"
    path.write_text(
        "<|endoftext|>\n"  # 开头就是分隔符：前面是空段，跳过
        "Once upon a time.\nThe end.\n"  # 故事内部的换行要保留
        "<|endoftext|>\n\n\n"  # 分隔符后的空行不算一篇
        "Tom ran.\n"
        "<|endoftext|>\n"
        "A.<|endoftext|>B.\n"  # 分隔符和文字在同一行
        "<|endoftext|>\n"
        "Last one",  # 结尾没有分隔符也算一篇
        encoding="utf-8",
    )

    assert list(iter_stories(path)) == [
        "Once upon a time.\nThe end.",
        "Tom ran.",
        "A.",
        "B.",
        "Last one",
    ]


# ---------- 编码写盘 ----------


def test_write_tokens_appends_eot_after_each_story(tmp_path: Path) -> None:
    from tlab.data.tinystories import write_tokens

    path = tmp_path / "train.bin"

    n_tokens = write_tokens(["ab", "c", "你"], ByteTokenizer(), path)

    # 每篇故事的编号后面跟一个 256；"你" 是 3 个 UTF-8 字节
    expected = [97, 98, 256, 99, 256, 228, 189, 160, 256]
    assert n_tokens == len(expected)
    assert np.fromfile(path, dtype=np.uint16).tolist() == expected


def test_write_tokens_does_not_depend_on_chunk_size(tmp_path: Path) -> None:
    from tlab.data.tinystories import write_tokens

    stories = [f"story number {i}" for i in range(25)]

    write_tokens(stories, ByteTokenizer(), tmp_path / "a.bin", chunk_size=1)
    write_tokens(stories, ByteTokenizer(), tmp_path / "b.bin", chunk_size=7)

    assert (tmp_path / "a.bin").read_bytes() == (tmp_path / "b.bin").read_bytes()


def test_write_tokens_overwrites_existing_file(tmp_path: Path) -> None:
    from tlab.data.tinystories import write_tokens

    path = tmp_path / "train.bin"

    write_tokens(["ab"], ByteTokenizer(), path)
    write_tokens(["ab"], ByteTokenizer(), path)

    # 重跑预处理不能把新结果接在旧文件后面
    assert np.fromfile(path, dtype=np.uint16).tolist() == [97, 98, 256]


def test_write_tokens_rejects_vocab_too_big_for_uint16(tmp_path: Path) -> None:
    from tlab.data.tinystories import write_tokens

    # uint16 只能表示 0–65535，词表最多 65536 个
    with pytest.raises(ValueError):
        write_tokens(["a"], ByteTokenizer(vocab_size=2**16 + 1), tmp_path / "x.bin")


def test_open_tokens_returns_read_only_memmap(tmp_path: Path) -> None:
    from tlab.data.tinystories import open_tokens, write_tokens

    path = tmp_path / "train.bin"
    write_tokens(["ab"], ByteTokenizer(), path)

    data = open_tokens(path)  # (N,)

    assert isinstance(data, np.memmap)
    assert data.dtype == np.uint16
    assert data.tolist() == [97, 98, 256]
    # 只读打开：训练时不会意外改坏数据文件
    with pytest.raises(ValueError):
        data[0] = 1


def test_count_token_does_not_depend_on_chunk_size(tmp_path: Path) -> None:
    from tlab.data.tinystories import count_token, write_tokens

    path = tmp_path / "train.bin"
    write_tokens([f"story {i}" for i in range(25)], ByteTokenizer(), path)

    # 每篇末尾一个 256，数出来就是篇数；块与块的交界处不能漏数或重复数
    for chunk_size in (1, 7, 1000):
        assert count_token(path, 256, chunk_size=chunk_size) == 25
    # 每篇都只有一个 "s"（来自 "story"）
    assert count_token(path, ord("s"), chunk_size=7) == 25


# ---------- 下载校验 ----------


def test_verify_file_checks_size_and_sha256(tmp_path: Path) -> None:
    from tlab.data.tinystories import verify_file

    path = tmp_path / "f.txt"
    path.write_bytes(b"hello")
    sha256 = hashlib.sha256(b"hello").hexdigest()

    verify_file(path, 5, sha256)  # 大小和哈希都对：不报错
    with pytest.raises(ValueError, match="大小"):
        verify_file(path, 6, sha256)
    with pytest.raises(ValueError, match="SHA-256"):
        verify_file(path, 5, "0" * 64)


def test_download_skips_network_when_file_exists(tmp_path: Path, monkeypatch) -> None:
    import huggingface_hub

    from tlab.data import tinystories

    # 换成一个 5 字节的假文件，并让任何联网下载都直接报错
    fake = tinystories.SourceFile("f.txt", 5, hashlib.sha256(b"hello").hexdigest())
    monkeypatch.setitem(tinystories.SOURCE_FILES, "valid", fake)

    def no_network(*args, **kwargs):
        raise AssertionError("文件已存在，不应联网")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_network)
    (tmp_path / "f.txt").write_bytes(b"hello")

    # 已下载过：只核对大小与哈希，直接返回路径
    assert tinystories.download("valid", tmp_path) == tmp_path / "f.txt"
    # 文件被改过：核对不通过就报错，不会拿坏文件去训练
    (tmp_path / "f.txt").write_bytes(b"hellO")
    with pytest.raises(ValueError, match="SHA-256"):
        tinystories.download("valid", tmp_path)


def test_source_files_match_verified_metadata() -> None:
    from tlab.data.tinystories import SOURCE_FILES

    train, valid = SOURCE_FILES["train"], SOURCE_FILES["valid"]

    # 2026-11-01 从 HF API 核实的文件名、大小与哈希
    assert (train.filename, train.size) == ("TinyStoriesV2-GPT4-train.txt", 2227753162)
    assert (valid.filename, valid.size) == ("TinyStoriesV2-GPT4-valid.txt", 22502601)
    assert train.sha256 == (
        "6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443"
    )
    assert valid.sha256 == (
        "6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d"
    )
