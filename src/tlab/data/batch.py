"""从一维 token 序列里截窗口：训练时随机取 batch，评估时也复用 read_window；
数据可以是内存里的张量，也可以是硬盘上的 np.memmap。
"""

import numpy as np
import torch


def read_window(
    data: torch.Tensor | np.ndarray, start: int, length: int
) -> torch.Tensor:
    """取出 data[start : start + length]：张量原样返回，memmap 转成 int64 张量。

    随机取 batch 和完整验证集评估（tlab.eval.full_val）都用它。
    """
    piece = data[start : start + length]  # (length,)
    if isinstance(piece, torch.Tensor):
        return piece
    # memmap 切片时才从硬盘读这一小段；uint16 先转 int64，embedding 要求 long
    return torch.from_numpy(piece.astype(np.int64))  # (length,)


def get_batch(
    data: torch.Tensor | np.ndarray,
    batch_size: int,
    block_size: int,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机截取 batch_size 段长为 block_size 的文本；y 是 x 往后挪一位。"""
    assert len(data) > block_size, "数据长度必须大于 block_size"
    # 起点最大取 len - block_size - 1，保证 y 的最后一个位置不越界；
    # 抽取方式与 M1 相同，同一个种子取到同样的起点
    starts = torch.randint(
        len(data) - block_size, (batch_size,), generator=generator
    ).tolist()  # B 个起点
    x = torch.stack([read_window(data, s, block_size) for s in starts])  # (B, T)
    y = torch.stack([read_window(data, s + 1, block_size) for s in starts])  # (B, T)
    return x.to(device), y.to(device)
