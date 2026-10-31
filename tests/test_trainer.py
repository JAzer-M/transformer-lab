import csv
import json
import math
from pathlib import Path

import pytest
import torch

# 待实现模块在各测试内导入：外围工具先写好时，它们的测试可以先单独通过


def write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def tiny_config(tmp_path: Path, **overrides):
    from tlab.train.config import TrainConfig

    # CPU 上几秒内跑完的极小配置
    fields = {
        "data_dir": str(tmp_path / "data"),
        "out_dir": str(tmp_path / "out"),
        "n_layers": 1,
        "n_heads": 2,
        "d_model": 32,
        "block_size": 8,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "max_iters": 10,
        "eval_interval": 5,
        "eval_iters": 2,
        "seed": 0,
        "device": "cpu",
    }
    fields.update(overrides)
    return TrainConfig(**fields)


def write_tiny_dataset(data_dir: Path) -> None:
    from tlab.data.char import CharTokenizer, save_ids, split_ids

    # 一段会重复的短文本，足够让 get_batch 截出 block_size + 1 的窗口
    text = "to be or not to be, that is the question.\n" * 20
    tokenizer = CharTokenizer.from_text(text)
    train, val = split_ids(tokenizer.encode(text))
    data_dir.mkdir(parents=True)
    save_ids(train, data_dir / "train.bin")
    save_ids(val, data_dir / "val.bin")
    tokenizer.save(data_dir / "meta.json")


# ---------- 配置 ----------


def test_load_config_from_yaml_with_overrides(tmp_path: Path) -> None:
    from tlab.train.config import load_config

    path = write_yaml(
        tmp_path / "c.yaml",
        "n_layers: 6\nd_model: 384\nlearning_rate: 1.0e-3\nmax_iters: 2000\n",
    )

    config = load_config(path, ["max_iters=5", "learning_rate=3e-4", "device=cpu"])

    assert config.n_layers == 6 and config.d_model == 384
    # 命令行覆盖的值优先，并转换成字段声明的类型
    assert config.max_iters == 5 and isinstance(config.max_iters, int)
    assert config.learning_rate == 3e-4 and isinstance(config.learning_rate, float)
    assert config.device == "cpu"


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    from tlab.train.config import load_config

    path = write_yaml(tmp_path / "c.yaml", "n_layer: 6\n")  # 少了个 s 的拼写错误

    # 拼错的字段名必须报错，不能被悄悄忽略
    with pytest.raises(ValueError):
        load_config(path)
    with pytest.raises(ValueError):
        load_config(write_yaml(tmp_path / "ok.yaml", "n_layers: 6\n"), ["lr=1e-3"])


# ---------- 外围工具 ----------


def test_estimate_loss_restores_train_mode_and_builds_no_graph(tmp_path: Path) -> None:
    from tlab.model.gpt import GPT, GPTConfig
    from tlab.train.utils import estimate_loss

    torch.manual_seed(0)
    config = tiny_config(tmp_path)
    model = GPT(GPTConfig(11, config.block_size, config.d_model, 1, 2))
    data = torch.randint(0, 11, (200,))  # (N,)

    losses = estimate_loss(model, data, data, config, torch.device("cpu"))

    assert set(losses) == {"train", "val"}
    assert all(math.isfinite(v) for v in losses.values())
    # 评估结束后模型回到训练模式，且评估没有留下任何梯度
    assert model.training
    assert all(p.grad is None for p in model.parameters())


def test_run_logger_writes_csv_rows(tmp_path: Path) -> None:
    from tlab.train.utils import RunLogger

    logger = RunLogger(tmp_path)
    logger.log_eval(0, {"train": 4.2, "val": 4.3})
    logger.log_eval(5, {"train": 3.1, "val": 3.4})

    with open(tmp_path / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    assert [r["step"] for r in rows] == ["0", "5"]
    assert float(rows[1]["val_loss"]) == pytest.approx(3.4)


# ---------- 主循环（用户手写的 train） ----------


def test_train_runs_and_returns_one_loss_per_step(tmp_path: Path) -> None:
    from tlab.model.gpt import GPT, GPTConfig
    from tlab.train.trainer import train
    from tlab.train.utils import RunLogger

    torch.manual_seed(0)
    config = tiny_config(tmp_path, max_iters=10)
    model = GPT(GPTConfig(11, config.block_size, config.d_model, 1, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    data = torch.randint(0, 11, (200,))  # (N,)
    before = [p.detach().clone() for p in model.parameters()]

    losses = train(
        model, optimizer, data, data, config, torch.device("cpu"), RunLogger(tmp_path)
    )

    assert len(losses) == 10
    assert all(isinstance(v, float) and math.isfinite(v) for v in losses)
    # 参数确实被更新了
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    # 按 eval_interval 记了评估：第 0、5 步，以及最后一步
    with open(tmp_path / "metrics.csv", newline="") as f:
        steps = [int(r["step"]) for r in csv.DictReader(f)]
    assert steps == [0, 5, 9]


def test_train_can_memorize_a_repeating_sequence(tmp_path: Path) -> None:
    from tlab.model.gpt import GPT, GPTConfig
    from tlab.train.trainer import train
    from tlab.train.utils import RunLogger

    torch.manual_seed(0)
    config = tiny_config(
        tmp_path, max_iters=150, eval_interval=1000, learning_rate=1e-2
    )
    model = GPT(GPTConfig(5, config.block_size, config.d_model, 1, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # 0 1 2 3 4 0 1 2 3 4 ...：下一个 token 完全由当前 token 决定
    data = torch.arange(5).repeat(40)  # (N,)

    losses = train(
        model, optimizer, data, data, config, torch.device("cpu"), RunLogger(tmp_path)
    )

    # 初始约为 ln 5 ≈ 1.61；能学会规律时 loss 应降到很低
    assert losses[0] > 1.0
    assert sum(losses[-10:]) / 10 < 0.1


# ---------- 端到端：tiny 配置在 CPU 上跑 10 步（M0 验收项） ----------


def test_run_writes_metrics_metadata_and_checkpoint(tmp_path: Path) -> None:
    from tlab.train.run import run

    write_tiny_dataset(tmp_path / "data")
    config = tiny_config(tmp_path)

    out_dir = run(config)

    assert (out_dir / "metrics.csv").exists()
    meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["config"]["max_iters"] == 10 and meta["config"]["seed"] == 0
    assert meta["tokens_per_sec"] > 0
    assert math.isfinite(meta["final_val_loss"])
    assert "git_commit" in meta
    ckpt = torch.load(out_dir / "ckpt.pt", weights_only=False)
    assert {"model", "config", "vocab_size"} <= set(ckpt)
