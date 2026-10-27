"""运行设备选择。"""

import torch


def get_device() -> torch.device:
    """按 CUDA、MPS、CPU 的优先级选择当前可用设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
