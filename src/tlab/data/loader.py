"""按数据目录里的文件选择加载方式：有 tokenizer.json 是 BPE 数据，否则是字符级数据。"""

from pathlib import Path

import numpy as np
import torch

from tlab.data.char import CharTokenizer, load_ids
from tlab.data.tinystories import open_tokens
from tlab.tokenizer.hf_bpe import HFTokenizer


def is_bpe_dir(data_dir: Path) -> bool:
    return (data_dir / "tokenizer.json").exists()


def load_tokenizer(data_dir: Path) -> CharTokenizer | HFTokenizer:
    if is_bpe_dir(data_dir):
        return HFTokenizer.load(data_dir / "tokenizer.json")
    return CharTokenizer.load(data_dir / "meta.json")


def load_splits(
    data_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[np.memmap, np.memmap]:
    """返回 (train, val)，都是一维的 token 编号序列。"""
    if is_bpe_dir(data_dir):
        # BPE 数据可能有上亿个 token：memmap 按需读，不整个读进内存
        return open_tokens(data_dir / "train.bin"), open_tokens(data_dir / "val.bin")
    # 字符级数据很小：与 M1 相同，整个读进内存
    return load_ids(data_dir / "train.bin"), load_ids(data_dir / "val.bin")
