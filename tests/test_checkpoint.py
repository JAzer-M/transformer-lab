import csv
from pathlib import Path

import pytest
import torch

# 配置新字段、存档读档、日志的 lr 列与续训追加


def tiny_model_and_optimizer(seed: int):
    from tlab.model.gpt import GPT, GPTConfig

    torch.manual_seed(seed)
    model = GPT(GPTConfig(11, 8, 16, 1, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def take_one_step(model, optimizer) -> None:
    idx = torch.randint(0, 11, (2, 8))  # (B, T)
    model(idx).float().mean().backward()  # (B, T, V) → () 随便一个标量，只为产生梯度
    optimizer.step()


# ---------- 配置 ----------


def test_new_config_fields_default_to_m1_behavior(tmp_path: Path) -> None:
    from tlab.train.config import TrainConfig, load_config

    config = TrainConfig()

    # 不写这些字段的旧配置文件，行为与 M1 完全相同
    assert config.grad_accum_steps == 1
    assert config.grad_clip == 0.0
    assert config.lr_schedule == "constant"
    assert config.warmup_iters == 0
    assert config.min_lr == 0.0
    assert config.ckpt_interval == 0
    assert config.resume is False
    # 命令行写 resume=true 得到布尔值 True
    path = tmp_path / "c.yaml"
    path.write_text("max_iters: 5\n", encoding="utf-8")
    assert load_config(path, ["resume=true"]).resume is True


# ---------- 存档与读档 ----------


def test_checkpoint_round_trip_restores_training_state(tmp_path: Path) -> None:
    from tlab.train.config import TrainConfig
    from tlab.train.utils import load_checkpoint, save_checkpoint

    model, optimizer = tiny_model_and_optimizer(seed=0)
    take_one_step(model, optimizer)
    generator = torch.Generator().manual_seed(7)
    torch.randint(10, (5,), generator=generator)  # 让生成器走到中间某个状态
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, TrainConfig(), 11, optimizer, 42, generator)
    # 存档之后各自再往前走，用来对照恢复后是否完全一致
    expected_draw = torch.randint(10, (5,), generator=generator)
    expected_global = torch.rand(3)

    new_model, new_optimizer = tiny_model_and_optimizer(seed=1)  # 不同的初始化
    new_generator = torch.Generator().manual_seed(0)
    torch.manual_seed(123)
    step = load_checkpoint(path, new_model, new_optimizer, new_generator)

    assert step == 42
    for a, b in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(a, b)
    # AdamW 的一阶、二阶动量与步数也恢复了，续训后的更新才和不中断时一样
    old_state = optimizer.state_dict()["state"]
    new_state = new_optimizer.state_dict()["state"]
    assert old_state.keys() == new_state.keys()
    for key in old_state:
        for name in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(old_state[key][name], new_state[key][name])
    # 训练用生成器与全局随机数都回到存档时的位置
    assert torch.equal(torch.randint(10, (5,), generator=new_generator), expected_draw)
    assert torch.equal(torch.rand(3), expected_global)


def test_checkpoint_keeps_sampling_keys_and_leaves_no_temp_file(
    tmp_path: Path,
) -> None:
    from tlab.train.config import TrainConfig
    from tlab.train.utils import save_checkpoint

    model, optimizer = tiny_model_and_optimizer(seed=0)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, TrainConfig(), 11, optimizer, 3, torch.Generator())

    ckpt = torch.load(path, weights_only=False)
    # sample.py 需要的三项仍在
    assert {"model", "config", "vocab_size"} <= set(ckpt)
    # 先写临时文件再改名：成功后目录里只剩 ckpt.pt
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt.pt"]


# ---------- 日志 ----------


def read_rows(out_dir: Path) -> list[dict]:
    with open(out_dir / "metrics.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_run_logger_records_lr_column(tmp_path: Path) -> None:
    from tlab.train.utils import RunLogger

    logger = RunLogger(tmp_path)
    logger.log_eval(0, {"train": 4.2, "val": 4.3}, lr=1e-3)
    logger.log_eval(5, {"train": 3.1, "val": 3.4})  # 不传 lr 也能用

    rows = read_rows(tmp_path)

    assert float(rows[0]["lr"]) == pytest.approx(1e-3)
    assert rows[1]["lr"] == ""


def test_run_logger_resume_appends_instead_of_overwriting(tmp_path: Path) -> None:
    from tlab.train.utils import RunLogger

    RunLogger(tmp_path).log_eval(0, {"train": 4.2, "val": 4.3}, lr=1e-3)
    RunLogger(tmp_path, resume=True).log_eval(5, {"train": 3.1, "val": 3.4}, lr=5e-4)

    rows = read_rows(tmp_path)

    # 续训时接着写，旧记录还在，表头只有一行
    assert [r["step"] for r in rows] == ["0", "5"]


def test_run_logger_resume_drops_rows_after_the_checkpoint(tmp_path: Path) -> None:
    from tlab.train.utils import RunLogger

    # 第一次跑到第 4 步评估后崩溃，但最近的存档在第 3 步
    logger = RunLogger(tmp_path)
    for step in (0, 2, 4):
        logger.log_eval(step, {"train": 1.0, "val": 1.0}, lr=1e-3)

    resumed = RunLogger(tmp_path, resume=True, start_step=3)
    resumed.log_eval(4, {"train": 0.9, "val": 0.9}, lr=1e-3)

    # 存档之后的旧记录会被重新跑一遍，所以先删掉，曲线里同一步只出现一次
    rows = read_rows(tmp_path)
    assert [r["step"] for r in rows] == ["0", "2", "4"]
    assert float(rows[-1]["val_loss"]) == pytest.approx(0.9)
