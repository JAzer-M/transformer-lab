import math
from pathlib import Path

import pytest
import torch
from torch import nn

# 完整验证集评估（agent 编写）：把验证集切成首尾相接、互不重叠的窗口，
# 每个 token 恰好被预测一次，按 token 求平均。预期值都是手算的具体数字。


class FixedProbs(nn.Module):
    """假模型：不管输入是什么，都给 4 个 token 固定概率 [1/2, 1/4, 1/8, 1/8]。

    于是每个位置的 loss 只取决于要预测的 token 是谁：ln2、2ln2、3ln2、3ln2。
    """

    def __init__(self) -> None:
        super().__init__()
        # 一个用不到的参数：让模型和真模型一样有参数、有 train/eval 模式
        self.unused = nn.Parameter(torch.zeros(()))
        self.register_buffer("log_probs", torch.tensor([0.5, 0.25, 0.125, 0.125]).log())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        return self.log_probs.expand(B, T, 4) + self.unused  # (B, T, V)


def test_every_target_token_is_scored_exactly_once() -> None:
    from tlab.eval.full_val import evaluate_full

    # 23 个 token、窗口长 5：4 个满窗口预测 t1…t20，最后一个窗口只剩 t21、t22。
    # 第一个 token 故意取 1：它只当输入、不当预测目标；y 若没往后挪一位，它就会被算进去
    data = torch.tensor([1, 1, 2, 3] + [0, 1, 2, 3] * 4 + [0, 1, 2])  # (23,)

    loss, n_tokens = evaluate_full(
        FixedProbs(), data, block_size=5, batch_size=2, device=torch.device("cpu")
    )

    # 要预测的 22 个 token 里，0 有 5 个、1 有 6 个、2 有 6 个、3 有 5 个：
    # 总 loss = (5×1 + 6×2 + 6×3 + 5×3)·ln2 = 50·ln2，按 token 平均 ≈ 1.5753
    # （漏掉最后半个窗口得 2.25·ln2，按批平均得 2.3333·ln2，y 没挪一位得 2.3182·ln2，
    #   窗口重叠一个 token 得 2.2273·ln2：都和正确答案 2.2727·ln2 不同）
    assert n_tokens == 22
    assert loss == pytest.approx(50 * math.log(2) / 22)


def test_result_does_not_depend_on_batch_size_or_data_format(tmp_path: Path) -> None:
    import numpy as np

    from tlab.eval.full_val import evaluate_full
    from tlab.model.gpt import GPT, GPTConfig

    torch.manual_seed(0)
    model = GPT(
        GPTConfig(vocab_size=11, block_size=8, d_model=16, n_layers=1, n_heads=2)
    )
    # 40 个 token、窗口长 8：4 个满窗口 + 最后 7 个，满窗口数除不尽 3 和 7
    data = torch.randint(0, 11, (40,))  # (N,)
    cpu = torch.device("cpu")

    results = [evaluate_full(model, data, 8, b, cpu) for b in (1, 3, 7)]

    # 分批方式只影响速度：每种分法都覆盖全部 39 个 token，loss 相同
    assert [n for _, n in results] == [39, 39, 39]
    for loss, _ in results[1:]:
        assert loss == pytest.approx(results[0][0], rel=1e-6)
    # TinyStories 的验证集是硬盘上 uint16 的 memmap，结果应与张量相同
    data.numpy().astype(np.uint16).tofile(tmp_path / "val.bin")
    memmap = np.memmap(tmp_path / "val.bin", dtype=np.uint16, mode="r")  # (N,)
    loss_np, n_np = evaluate_full(model, memmap, 8, 3, cpu)
    assert n_np == 39
    assert loss_np == pytest.approx(results[0][0], rel=1e-6)


class RecordsMode(FixedProbs):
    """记下每次前向时模型处在 train 还是 eval 模式。"""

    def __init__(self) -> None:
        super().__init__()
        self.modes: list[bool] = []

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        self.modes.append(self.training)
        return super().forward(idx)


def test_scores_in_eval_mode_and_restores_the_previous_mode() -> None:
    from tlab.eval.full_val import evaluate_full

    model = RecordsMode()
    model.train()
    data = torch.tensor([0, 1, 2, 3] * 5)  # (20,)

    evaluate_full(model, data, block_size=5, batch_size=2, device=torch.device("cpu"))

    # 评估时关掉 dropout 等训练行为；评估完回到原来的 train 模式，训练才能接着进行
    assert model.modes and not any(model.modes)
    assert model.training


def train_tiny_run(tmp_path: Path, **overrides: int | str) -> Path:
    """在字符级小数据上用 run() 训 2 步，返回输出目录（里面有 ckpt.pt）。"""
    from tlab.data.char import CharTokenizer, save_ids, split_ids
    from tlab.train.config import TrainConfig
    from tlab.train.run import run

    text = "to be or not to be, that is the question.\n" * 20
    tokenizer = CharTokenizer.from_text(text)
    train, val = split_ids(tokenizer.encode(text))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    save_ids(train, data_dir / "train.bin")
    save_ids(val, data_dir / "val.bin")
    tokenizer.save(data_dir / "meta.json")
    fields = {
        "data_dir": str(data_dir),
        "out_dir": str(tmp_path / "out"),
        "n_layers": 1,
        "n_heads": 2,
        "d_model": 16,
        "block_size": 8,
        "batch_size": 4,
        "max_iters": 2,
        "eval_interval": 1000,
        "eval_iters": 1,
        "seed": 0,
        "device": "cpu",
    }
    fields.update(overrides)
    return run(TrainConfig(**fields))


def test_checkpoint_is_scored_on_its_whole_validation_set(tmp_path: Path) -> None:
    from tlab.data.loader import load_splits
    from tlab.eval.full_val import evaluate_checkpoint

    # 用 RMSNorm 训：评估时若没按存档里的配置重建结构，参数载入会直接失败
    out_dir = train_tiny_run(tmp_path, norm="rmsnorm")

    result = evaluate_checkpoint(out_dir / "ckpt.pt", 4, torch.device("cpu"))

    # 用的是存档训练时的验证集（不是训练集），除第一个 token 外每个都预测一次
    _, val = load_splits(tmp_path / "data")
    assert result["n_tokens"] == len(val) - 1
    assert result["block_size"] == 8
    assert math.isfinite(result["val_loss"])


def test_script_writes_full_val_json(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    out_dir = train_tiny_run(tmp_path)
    out = tmp_path / "full_val.json"

    # 每次训练完运行这条命令，再把 json 复制进 experiments/
    done = subprocess.run(
        [
            sys.executable,
            "scripts/eval_full_val.py",
            "--ckpt",
            str(out_dir / "ckpt.pt"),
            "--out",
            str(out),
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    # 终端里直接看到评的是第几步的存档，手册要求不是 max_iters 就停下
    assert "第 2 步" in done.stdout
    result = json.loads(out.read_text(encoding="utf-8"))
    assert {"val_loss", "n_tokens", "block_size", "ckpt_step", "eval_commit"} <= set(
        result
    )
    assert result["ckpt_step"] == 2


class ProbsByPosition(nn.Module):
    """假模型：前半段位置给 token 1 概率 1/2，后半段给 3/4；还记下前向时的模式。"""

    def __init__(self) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(()))
        self.register_buffer("first", torch.tensor([0.5, 0.5]).log())
        self.register_buffer("second", torch.tensor([0.25, 0.75]).log())
        self.modes: list[bool] = []

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        self.modes.append(self.training)
        B, T = idx.shape
        logits = self.first.repeat(B, T, 1)  # (B, T, V)
        logits[:, T // 2 :] = self.second  # (V,) 广播到后半段
        return logits + self.unused  # (B, T, V)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_extrapolation_reports_each_half_separately(batch_size: int) -> None:
    from tlab.eval.full_val import evaluate_extrapolation

    model = ProbsByPosition()
    model.train()
    # 11 个 token 全是 1、窗口长 4：2 个满窗口，最后不满一窗的部分不算（它没有后半段可比）
    data = torch.ones(11, dtype=torch.long)  # (N,)

    result = evaluate_extrapolation(model, data, 4, batch_size, torch.device("cpu"))

    # 前半段 -ln(1/2) ≈ 0.6931，后半段 -ln(3/4) ≈ 0.2877，每半段 2 个窗口 × 2 个 token
    assert result["first_half_loss"] == pytest.approx(math.log(2))
    assert result["second_half_loss"] == pytest.approx(-math.log(0.75))
    assert result["n_tokens_per_half"] == 4
    # 评估时是 eval 模式，评估完回到原来的模式
    assert model.modes and not any(model.modes)
    assert model.training


def test_first_half_of_a_long_window_equals_a_normal_short_forward() -> None:
    from tlab.eval.full_val import evaluate_extrapolation
    from tlab.model.gpt import GPT, GPTConfig
    from tlab.train.utils import cross_entropy_loss

    torch.manual_seed(0)
    model = GPT(GPTConfig(11, 8, 16, 1, 2, pos_encoding="rope"))
    data = torch.randint(0, 11, (33,))  # (N,)
    model.eval()

    result = evaluate_extrapolation(model, data, 16, 2, torch.device("cpu"))

    # 模型是因果的：长窗口前半段的位置看不到后半段，和只输入前 8 个 token 算出来一样
    with torch.no_grad():
        losses = [
            cross_entropy_loss(
                model(data[s : s + 8][None]), data[s + 1 : s + 9][None]
            ).item()
            for s in (0, 16)
        ]
    assert result["first_half_loss"] == pytest.approx(sum(losses) / 2, rel=1e-5)


@pytest.mark.parametrize("pos_encoding", ["sinusoidal", "rope"])
def test_checkpoint_extrapolation_uses_twice_the_training_length(
    tmp_path: Path, pos_encoding: str
) -> None:
    from tlab.eval.full_val import extrapolate_checkpoint

    out_dir = train_tiny_run(tmp_path, pos_encoding=pos_encoding)

    result = extrapolate_checkpoint(out_dir / "ckpt.pt", 4, torch.device("cpu"))

    assert result["block_size"] == 8 and result["length"] == 16
    assert result["ckpt_step"] == 2
    # 记下评的是哪种位置编码：两种参数量相同，文件名之外还要有据可查
    assert result["pos_encoding"] == pos_encoding
    assert result["n_tokens_per_half"] > 0
    assert math.isfinite(result["first_half_loss"])
    assert math.isfinite(result["second_half_loss"])


def test_learned_positions_cannot_be_extrapolated(tmp_path: Path) -> None:
    from tlab.eval.full_val import extrapolate_checkpoint

    out_dir = train_tiny_run(tmp_path)

    # 位置表只有 block_size 行，超出的位置根本没有向量：直接说明白，不要算出一个假数
    with pytest.raises(ValueError, match="位置表"):
        extrapolate_checkpoint(out_dir / "ckpt.pt", 4, torch.device("cpu"))


def test_extrapolation_script_writes_json(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    out_dir = train_tiny_run(tmp_path, pos_encoding="rope")
    out = tmp_path / "extrapolation.json"

    done = subprocess.run(
        [
            sys.executable,
            "scripts/eval_extrapolation.py",
            "--ckpt",
            str(out_dir / "ckpt.pt"),
            "--out",
            str(out),
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "第 2 步" in done.stdout
    result = json.loads(out.read_text(encoding="utf-8"))
    expected = {"first_half_loss", "second_half_loss", "length", "ckpt_step"}
    assert expected <= set(result)


def test_extrapolation_rejects_an_odd_length_and_too_little_data() -> None:
    from tlab.eval.full_val import evaluate_extrapolation

    cpu = torch.device("cpu")
    data = torch.ones(11, dtype=torch.long)  # (N,)

    # 前后半段要一样长，才好比较
    with pytest.raises(AssertionError, match="偶数"):
        evaluate_extrapolation(ProbsByPosition(), data, 5, 1, cpu)
    # 数据连一个满窗口都不够时，要说明原因，不是除以 0
    with pytest.raises(ValueError, match="满窗口"):
        evaluate_extrapolation(ProbsByPosition(), data, 16, 1, cpu)
