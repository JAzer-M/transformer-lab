"""字符级分词与数据切分（Tiny Shakespeare 用）；批次采样见 batch.py。"""

import json
from pathlib import Path

import numpy as np
import torch

# get_batch 已移到 batch.py；trainer.py 等仍从这里导入，保留同名的再导出
from tlab.data.batch import get_batch as get_batch


class CharTokenizer:
    """每个字符对应一个整数编号；字表按字符排序，同一段文本总是得到同一个字表。"""

    def __init__(self, chars: list[str]) -> None:
        self.chars = chars
        # 字符 → 编号，编号 → 字符
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = dict(enumerate(chars))

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        # 去重后排序，得到字表
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        # 字表里没有的字符会抛 KeyError
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: Path) -> None:
        payload = json.dumps({"chars": self.chars}, ensure_ascii=False)
        path.write_text(payload, encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        return cls(json.loads(path.read_text(encoding="utf-8"))["chars"])


def split_ids(
    ids: list[int], val_fraction: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    # 前 (1 - val_fraction) 做训练集，后面做验证集；不打乱，保持原文顺序
    data = torch.tensor(ids, dtype=torch.long)  # (N,)
    n_train = int(len(data) * (1 - val_fraction))
    return data[:n_train], data[n_train:]  # (N_train,), (N_val,)


def save_ids(ids: torch.Tensor, path: Path) -> None:
    # 字符级词表远小于 65536，用 uint16 存，每个编号 2 字节
    assert int(ids.max()) < 2**16, "编号超出 uint16 范围"
    ids.numpy().astype(np.uint16).tofile(path)


def load_ids(path: Path) -> torch.Tensor:
    # 读回 uint16，再转成 embedding 需要的 int64
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int64))  # (N,)
