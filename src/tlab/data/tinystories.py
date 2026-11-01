"""TinyStories：下载与校验、按 <|endoftext|> 拆故事、编码写成 uint16 文件、memmap 读取。"""

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from tlab.tokenizer.hf_bpe import EOT

REPO_ID = "roneneldan/TinyStories"
# 固定数据集版本：以后仓库有更新，下载到的仍是同一份文件
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"


@dataclass(frozen=True)
class SourceFile:
    filename: str
    size: int  # 字节数
    sha256: str


# 2026-11-01 从 HF API 核实；V2 的故事全部由 GPT-4 生成
SOURCE_FILES = {
    "train": SourceFile(
        "TinyStoriesV2-GPT4-train.txt",
        2_227_753_162,
        "6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443",
    ),
    "valid": SourceFile(
        "TinyStoriesV2-GPT4-valid.txt",
        22_502_601,
        "6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d",
    ),
}


class TokenEncoder(Protocol):
    """write_tokens 只需要词表的这三样东西。"""

    @property
    def vocab_size(self) -> int: ...

    @property
    def eot_id(self) -> int: ...

    def encode_batch(self, texts: list[str]) -> list[list[int]]: ...


# ---------- 下载与校验 ----------


def verify_file(path: Path, size: int, sha256: str) -> None:
    """大小或 SHA-256 不符时报错：下载不完整、文件被换过都能发现。"""
    actual_size = path.stat().st_size
    if actual_size != size:
        raise ValueError(f"{path.name} 大小为 {actual_size:,} 字节，应为 {size:,}")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        # 每次读 1 MB：2 GB 的文件不必整个读进内存
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise ValueError(
            f"{path.name} 的 SHA-256 为 {digest.hexdigest()}，应为 {sha256}"
        )


def download(split: str, out_dir: Path) -> Path:
    """下载 train 或 valid 到 out_dir，核对大小与 SHA-256 后返回路径。

    文件已存在时不联网、只核对，所以也可以先用浏览器或 curl 手动下载到 out_dir。
    """
    source = SOURCE_FILES[split]
    path = out_dir / source.filename
    if not path.exists():
        # 只有真要下载时才需要；网络不通时可设环境变量 HF_ENDPOINT 换镜像
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            REPO_ID,
            source.filename,
            repo_type="dataset",
            revision=REVISION,
            local_dir=out_dir,
        )
    verify_file(path, source.size, source.sha256)
    return path


# ---------- 拆分故事 ----------


def iter_stories(path: Path) -> Iterator[str]:
    """逐行读文件，按 <|endoftext|> 切出一篇篇故事；去掉首尾空白，跳过空段。"""
    lines: list[str] = []  # 当前这篇故事已经读到的部分
    with path.open(encoding="utf-8") as f:
        for line in f:
            # 一行里可能有 0 个或多个分隔符：前面几段各自收尾，最后一段留给下一篇
            *finished, rest = line.split(EOT)
            for part in finished:
                story = ("".join(lines) + part).strip()
                if story:
                    yield story
                lines = []
            lines.append(rest)
    # 文件末尾没有分隔符时，最后一篇也要算上
    story = "".join(lines).strip()
    if story:
        yield story


# ---------- 编码写盘与读取 ----------


def _chunks(items: Iterable[str], size: int) -> Iterator[list[str]]:
    # 每次凑够 size 篇交给 encode_batch；最后不足 size 篇也要交出去
    chunk: list[str] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def write_tokens(
    stories: Iterable[str],
    tokenizer: TokenEncoder,
    path: Path,
    chunk_size: int = 1024,
) -> int:
    """把故事分块编码，每篇末尾加 <|endoftext|>，以 uint16 写入 path；返回 token 总数。"""
    # uint16 能表示 0–65535，所以词表最多 65536 个
    if tokenizer.vocab_size > 2**16:
        raise ValueError(f"词表大小 {tokenizer.vocab_size} 超出 uint16 的 65536 个编号")
    n_tokens = 0
    # "wb" 覆盖旧文件；一块一块地编码、写入，内存里始终只有一块
    with path.open("wb") as f:
        for chunk in _chunks(stories, chunk_size):
            ids: list[int] = []
            for story_ids in tokenizer.encode_batch(chunk):
                ids.extend(story_ids)
                ids.append(tokenizer.eot_id)
            np.array(ids, dtype=np.uint16).tofile(f)  # (n,)
            n_tokens += len(ids)
    return n_tokens


def open_tokens(path: Path) -> np.memmap:
    # 只读映射：不把整个文件读进内存，取 batch 时才从硬盘读用到的几段
    return np.memmap(path, dtype=np.uint16, mode="r")  # (N,)


def count_token(path: Path, token_id: int, chunk_size: int = 1 << 24) -> int:
    """数 path 中 token_id 出现了几次；分块比较，额外内存只有一块那么大。"""
    data = open_tokens(path)  # (N,)
    total = 0
    for start in range(0, len(data), chunk_size):
        # data == token_id 会得到和 data 一样长的布尔数组（每个元素 1 字节）；
        # 一次只比较一块，布尔数组就只有 chunk_size 长（默认约 1600 万，16 MB）
        block = data[start : start + chunk_size]  # (≤ chunk_size,)
        total += int((block == token_id).sum())
    return total
