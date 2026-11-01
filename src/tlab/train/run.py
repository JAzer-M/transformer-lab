"""把数据、模型、优化器、主循环和记录串起来，完成一次训练。"""

import time
from pathlib import Path

import torch

from tlab.data.loader import load_splits, load_tokenizer
from tlab.device import get_device
from tlab.model.gpt import GPT, GPTConfig, count_parameters
from tlab.train.config import TrainConfig
from tlab.train.trainer import train
from tlab.train.utils import (
    RunLogger,
    estimate_loss,
    peak_memory_mb,
    save_checkpoint,
    write_run_info,
)


def run(config: TrainConfig) -> Path:
    """训练一次，输出目录里写 metrics.csv、run.json、ckpt.pt，返回输出目录。"""
    torch.manual_seed(config.seed)
    device = get_device() if config.device == "auto" else torch.device(config.device)

    # 数据：词表决定 vocab_size；字符级整个读进内存，BPE 用 memmap 按需读
    data_dir = Path(config.data_dir)
    tokenizer = load_tokenizer(data_dir)
    train_data, val_data = load_splits(data_dir)  # (N_train,), (N_val,)

    # 模型与优化器
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=config.block_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
    )
    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    n_params = count_parameters(model)
    print(f"设备 {device}，参数量 {n_params:,}")

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(out_dir)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # 主循环（用户手写）；计时包含其中的评估
    start = time.perf_counter()
    train_losses = train(model, optimizer, train_data, val_data, config, device, logger)
    elapsed = time.perf_counter() - start

    final = estimate_loss(model, train_data, val_data, config, device)
    tokens = config.max_iters * config.batch_size * config.block_size
    save_checkpoint(out_dir / "ckpt.pt", model, config, tokenizer.vocab_size)
    write_run_info(
        out_dir,
        config,
        {
            "device": str(device),
            "torch_version": torch.__version__,
            "n_params": n_params,
            "train_seconds": elapsed,
            "tokens_per_sec": tokens / elapsed,
            "peak_memory_mb": peak_memory_mb(device),
            "final_train_loss": final["train"],
            "final_val_loss": final["val"],
            "last_step_loss": train_losses[-1],
        },
    )
    print(f"完成：val loss {final['val']:.4f}，{tokens / elapsed:,.0f} tokens/s")
    return out_dir
