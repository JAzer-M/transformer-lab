from pathlib import Path

import numpy as np
import torch

# 通用的 get_batch：同时支持内存里的张量和硬盘上的 memmap


def write_memmap(tmp_path: Path, values: np.ndarray) -> np.memmap:
    path = tmp_path / "ids.bin"
    values.astype(np.uint16).tofile(path)
    return np.memmap(path, dtype=np.uint16, mode="r")  # (N,)


def test_memmap_and_tensor_give_identical_batches(tmp_path: Path) -> None:
    from tlab.data.batch import get_batch

    values = np.arange(1000) % 300  # (N,)
    data_memmap = write_memmap(tmp_path, values)
    data_tensor = torch.from_numpy(values.astype(np.int64))

    x1, y1 = get_batch(data_memmap, 8, 16, torch.Generator().manual_seed(7))
    x2, y2 = get_batch(data_tensor, 8, 16, torch.Generator().manual_seed(7))

    assert x1.shape == (8, 16) and y1.shape == (8, 16)
    assert x1.dtype == torch.long and y1.dtype == torch.long
    # 同一个种子，两种数据来源取到完全相同的 batch：M1 的结果仍可复现
    assert torch.equal(x1, x2) and torch.equal(y1, y2)


def test_starts_are_drawn_exactly_as_in_m1(tmp_path: Path) -> None:
    from tlab.data.batch import get_batch

    data = write_memmap(tmp_path, np.arange(1000))  # (N,)，第 i 个位置的值就是 i

    x, _ = get_batch(data, 8, 16, torch.Generator().manual_seed(7))  # (B, T)

    # M1 的抽取方式：起点 ~ randint(len - block_size)；换了抽法，旧实验就无法复现
    starts = torch.randint(1000 - 16, (8,), generator=torch.Generator().manual_seed(7))
    assert x[:, 0].tolist() == starts.tolist()


def test_memmap_targets_are_inputs_shifted_by_one(tmp_path: Path) -> None:
    from tlab.data.batch import get_batch

    data = write_memmap(tmp_path, np.arange(100))  # (N,)

    x, y = get_batch(data, 4, 8, torch.Generator().manual_seed(0))  # (B, T), (B, T)

    assert torch.equal(x[:, 1:] - x[:, :-1], torch.ones(4, 7, dtype=torch.long))
    assert torch.equal(y, x + 1)
    assert int(y.max()) <= 99


def test_large_uint16_ids_are_read_as_positive(tmp_path: Path) -> None:
    from tlab.data.batch import get_batch

    # 超过 32767 的编号如果被当成有符号数，会变成负数
    data = write_memmap(tmp_path, np.arange(65400, 65536))  # (N,)

    x, _ = get_batch(data, 4, 8, torch.Generator().manual_seed(0))  # (B, T)

    assert int(x.min()) >= 65400 and int(x.max()) <= 65535


def test_char_module_reexports_the_same_get_batch() -> None:
    from tlab.data import batch, char

    # trainer.py 和 utils.py 仍从 tlab.data.char 导入，拿到的是同一个函数
    assert char.get_batch is batch.get_batch
