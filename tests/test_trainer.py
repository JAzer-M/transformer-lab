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


def test_load_config_reads_norm_settings_and_rejects_typos(tmp_path: Path) -> None:
    from tlab.train.config import load_config

    path = write_yaml(tmp_path / "c.yaml", "norm: rmsnorm\nnorm_position: post\n")

    config = load_config(path)

    assert (config.norm, config.norm_position) == ("rmsnorm", "post")
    # 取值拼错在读配置时就报错，不用等到建模型、更不会悄悄换成别的结构
    with pytest.raises(ValueError):
        load_config(path, ["norm=rms"])
    with pytest.raises(ValueError):
        load_config(path, ["norm_position=middle"])


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


# ---------- 主循环（train） ----------


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


def test_gpt_config_from_saved_config_defaults_missing_norm_fields() -> None:
    from tlab.train.utils import gpt_config_from

    # M1/M2 存档里的配置没有这两项：按默认值重建，才能和存档里的参数对上
    old = {"block_size": 8, "d_model": 16, "n_layers": 1, "n_heads": 2, "seed": 0}
    config = gpt_config_from(old, vocab_size=11)
    assert (config.vocab_size, config.block_size, config.d_model) == (11, 8, 16)
    assert (config.n_layers, config.n_heads) == (1, 2)
    assert (config.norm, config.norm_position) == ("layernorm", "pre")

    # M3 的配置写了这两项，就照写的来
    new = dict(old, norm="rmsnorm", norm_position="post")
    config = gpt_config_from(new, vocab_size=11)
    assert (config.norm, config.norm_position) == ("rmsnorm", "post")


def test_run_builds_the_configured_norm_and_checkpoint_rebuilds_it(
    tmp_path: Path,
) -> None:
    from tlab.model.gpt import GPT
    from tlab.model.layers import RMSNorm
    from tlab.train.run import run
    from tlab.train.utils import gpt_config_from

    write_tiny_dataset(tmp_path / "data")
    out_dir = run(tiny_config(tmp_path, max_iters=2, norm="rmsnorm"))

    # 按 sample.py 的做法，用存档里的配置重建模型，再严格载入参数：
    # 训练时若没用上 RMSNorm，存档里会多出 LayerNorm 的 bias，严格载入就会报错
    ckpt = torch.load(out_dir / "ckpt.pt", weights_only=False)
    model = GPT(gpt_config_from(ckpt["config"], ckpt["vocab_size"]))
    model.load_state_dict(ckpt["model"])
    assert isinstance(model.ln_f, RMSNorm)


def model_switches() -> list[str]:
    from dataclasses import MISSING, fields

    from tlab.model.gpt import GPTConfig

    # GPTConfig 里有默认值的字段是 M3 起加的结构开关，M1、M2 的旧存档里都没有
    return [f.name for f in fields(GPTConfig) if f.default is not MISSING]


def test_every_model_switch_is_a_train_config_field() -> None:
    from dataclasses import MISSING, fields

    from tlab.model.gpt import GPTConfig
    from tlab.train.config import TrainConfig

    # GPTConfig 里有默认值的字段是 M3 起加的结构开关：每一个都必须能在训练配置里设置，
    # 否则 YAML 里写了也传不进模型（A1、A3、A4 还会再加开关）
    switches = {f.name for f in fields(GPTConfig) if f.default is not MISSING}
    assert switches
    assert switches <= {f.name for f in fields(TrainConfig)}


def test_load_config_reads_mlp_setting_and_rejects_typos(tmp_path: Path) -> None:
    from tlab.train.config import load_config

    path = write_yaml(tmp_path / "c.yaml", "mlp: swiglu\n")

    assert load_config(path).mlp == "swiglu"
    with pytest.raises(ValueError):
        load_config(path, ["mlp=glu"])


def test_old_saved_config_rebuilds_with_gelu_mlp() -> None:
    from tlab.train.utils import gpt_config_from

    # M1、M2 和 A2 的存档里都没有 mlp 这一项：按默认的 GELU MLP 重建
    old = {"block_size": 8, "d_model": 16, "n_layers": 1, "n_heads": 2, "seed": 0}
    assert gpt_config_from(old, vocab_size=11).mlp == "gelu"
    assert gpt_config_from(dict(old, mlp="swiglu"), vocab_size=11).mlp == "swiglu"


def test_run_with_swiglu_saves_a_checkpoint_that_rebuilds(tmp_path: Path) -> None:
    from tlab.model.layers import SwiGLU
    from tlab.train.run import run
    from tlab.train.utils import load_model

    write_tiny_dataset(tmp_path / "data")
    out_dir = run(tiny_config(tmp_path, max_iters=2, mlp="swiglu"))

    # 按存档重建并严格载入：训练时若没用上 SwiGLU，参数名就对不上，这里会报错
    model, _ = load_model(out_dir / "ckpt.pt", torch.device("cpu"))
    assert isinstance(model.blocks[0].mlp, SwiGLU)


def test_checkpoint_saved_before_the_switches_still_loads(tmp_path: Path) -> None:
    from tlab.model.block import MLP
    from tlab.train.run import run
    from tlab.train.utils import load_model

    write_tiny_dataset(tmp_path / "data")
    path = run(tiny_config(tmp_path, max_iters=2)) / "ckpt.pt"
    # 做成 M1、M2 存档的样子：配置里没有后来加的开关
    ckpt = torch.load(path, weights_only=False)
    for key in model_switches():
        del ckpt["config"][key]
    torch.save(ckpt, path)

    # 严格载入：缺的开关按默认值重建，参数名必须对得上
    model, _ = load_model(path, torch.device("cpu"))
    assert isinstance(model.blocks[0].mlp, MLP)


def test_load_config_reads_position_settings_and_rejects_typos(tmp_path: Path) -> None:
    from tlab.train.config import load_config

    path = write_yaml(tmp_path / "c.yaml", "pos_encoding: rope\nrope_theta: 1000000\n")

    config = load_config(path)

    assert config.pos_encoding == "rope"
    # 配置文件里的整数按字段类型转成小数
    assert config.rope_theta == 1000000.0 and isinstance(config.rope_theta, float)
    with pytest.raises(ValueError):
        load_config(path, ["pos_encoding=abs"])


def test_old_saved_config_rebuilds_with_learned_positions() -> None:
    from tlab.train.utils import gpt_config_from

    old = {"block_size": 8, "d_model": 16, "n_layers": 1, "n_heads": 2, "seed": 0}
    assert gpt_config_from(old, vocab_size=11).pos_encoding == "learned"
    new = dict(old, pos_encoding="rope", rope_theta=100.0)
    rebuilt = gpt_config_from(new, vocab_size=11)
    assert (rebuilt.pos_encoding, rebuilt.rope_theta) == ("rope", 100.0)


@pytest.mark.parametrize("pos_encoding", ["sinusoidal", "rope"])
def test_run_with_fixed_positions_saves_a_checkpoint_that_rebuilds(
    tmp_path: Path, pos_encoding: str
) -> None:
    from tlab.train.run import run
    from tlab.train.utils import load_model

    write_tiny_dataset(tmp_path / "data")
    out_dir = run(tiny_config(tmp_path, max_iters=2, pos_encoding=pos_encoding))

    # 严格载入：训练时若没用上这个开关，存档里会多出 pos_emb，这里就会报错
    model, _ = load_model(out_dir / "ckpt.pt", torch.device("cpu"))
    assert not hasattr(model, "pos_emb")


def test_resume_refuses_a_different_position_setting(tmp_path: Path) -> None:
    from tlab.train.run import run

    write_tiny_dataset(tmp_path / "data")
    run(tiny_config(tmp_path, max_iters=2, pos_encoding="rope"))

    # RoPE 和正弦都没有位置参数，严格载入看不出区别：不查的话会悄悄换结构接着训
    with pytest.raises(ValueError, match="pos_encoding"):
        run(tiny_config(tmp_path, max_iters=3, pos_encoding="sinusoidal", resume=True))
    with pytest.raises(ValueError, match="rope_theta"):
        run(
            tiny_config(
                tmp_path,
                max_iters=3,
                pos_encoding="rope",
                rope_theta=100.0,
                resume=True,
            )
        )
    # 结构相同才能接着训
    out_dir = run(tiny_config(tmp_path, max_iters=3, pos_encoding="rope", resume=True))
    ckpt = torch.load(out_dir / "ckpt.pt", weights_only=False)
    assert ckpt["step"] == 3


def test_resume_of_an_old_checkpoint_counts_missing_switches_as_default(
    tmp_path: Path,
) -> None:
    from tlab.train.run import run

    write_tiny_dataset(tmp_path / "data")
    path = run(tiny_config(tmp_path, max_iters=2)) / "ckpt.pt"
    ckpt = torch.load(path, weights_only=False)
    for key in model_switches():
        del ckpt["config"][key]
    torch.save(ckpt, path)

    # 旧存档没有这些字段，等于当时的默认结构：默认配置可以接着训，换结构不行
    with pytest.raises(ValueError, match="pos_encoding"):
        run(tiny_config(tmp_path, max_iters=3, pos_encoding="rope", resume=True))
    run(tiny_config(tmp_path, max_iters=3, resume=True))


def test_load_config_reads_n_kv_heads_and_rejects_bad_values(tmp_path: Path) -> None:
    from tlab.train.config import load_config

    path = write_yaml(tmp_path / "c.yaml", "n_heads: 6\nn_kv_heads: 2\n")

    config = load_config(path)
    assert config.n_kv_heads == 2 and isinstance(config.n_kv_heads, int)

    # 命令行可以改成别的组数，也可以用 null 改回 MHA
    assert load_config(path, ["n_kv_heads=3"]).n_kv_heads == 3
    assert load_config(path, ["n_kv_heads=null"]).n_kv_heads is None
    # 不写就是 None，即 MHA
    assert (
        load_config(write_yaml(tmp_path / "d.yaml", "n_heads: 6\n")).n_kv_heads is None
    )
    # 6 个头分不成 4 组
    with pytest.raises(ValueError, match="n_kv_heads"):
        load_config(path, ["n_kv_heads=4"])


def test_old_saved_config_rebuilds_with_mha() -> None:
    from tlab.train.utils import gpt_config_from

    old = {"block_size": 8, "d_model": 16, "n_layers": 1, "n_heads": 2, "seed": 0}
    assert gpt_config_from(old, vocab_size=11).n_kv_heads is None
    new = dict(old, n_kv_heads=1)
    assert gpt_config_from(new, vocab_size=11).n_kv_heads == 1


def test_run_with_n_kv_heads_saves_a_checkpoint_that_rebuilds(tmp_path: Path) -> None:
    from tlab.train.run import run
    from tlab.train.utils import load_model

    write_tiny_dataset(tmp_path / "data")
    path = run(tiny_config(tmp_path, max_iters=2, n_kv_heads=1)) / "ckpt.pt"

    model, _ = load_model(path, torch.device("cpu"))

    assert model.blocks[0].attn.n_kv_heads == 1


def test_resume_refuses_a_different_n_kv_heads(tmp_path: Path) -> None:
    from tlab.train.run import run

    write_tiny_dataset(tmp_path / "data")
    run(tiny_config(tmp_path, max_iters=2, n_kv_heads=1))

    with pytest.raises(ValueError, match="n_kv_heads"):
        run(tiny_config(tmp_path, max_iters=3, resume=True))


@pytest.mark.parametrize(
    "override", ["n_kv_heads=2.9", "n_kv_heads=true", "max_iters=2.9", "n_heads=true"]
)
def test_load_config_rejects_values_that_are_not_whole_numbers(
    tmp_path: Path, override: str
) -> None:
    from tlab.train.config import load_config

    path = write_yaml(tmp_path / "c.yaml", "n_heads: 6\n")

    # 以前 int(2.9) 会悄悄变成 2，int(True) 会变成 1，等于悄悄换了结构
    with pytest.raises(ValueError, match="整数"):
        load_config(path, [override])


def test_load_config_still_accepts_whole_numbers_written_as_floats(
    tmp_path: Path,
) -> None:
    from tlab.train.config import load_config

    path = write_yaml(tmp_path / "c.yaml", "n_heads: 6.0\nn_kv_heads: 2.0\n")

    config = load_config(path)

    assert config.n_heads == 6 and isinstance(config.n_heads, int)
    assert config.n_kv_heads == 2 and isinstance(config.n_kv_heads, int)


def test_load_config_reads_qwen3_switches(tmp_path: Path) -> None:
    from tlab.train.config import load_config

    path = write_yaml(
        tmp_path / "c.yaml",
        "n_heads: 16\nd_model: 1024\nmlp: swiglu\n"
        "head_dim: 128\nqk_norm: true\nmlp_hidden_dim: 3072\n",
    )

    config = load_config(path)
    assert config.head_dim == 128 and isinstance(config.head_dim, int)
    assert config.qk_norm is True
    assert config.mlp_hidden_dim == 3072 and isinstance(config.mlp_hidden_dim, int)

    # 命令行可以覆盖，null 改回默认
    assert load_config(path, ["head_dim=null"]).head_dim is None
    assert load_config(path, ["qk_norm=false"]).qk_norm is False
    assert load_config(path, ["mlp_hidden_dim=null"]).mlp_hidden_dim is None

    # 不写时和 M3 相同
    plain = load_config(write_yaml(tmp_path / "d.yaml", "n_heads: 6\n"))
    assert plain.head_dim is None
    assert plain.qk_norm is False
    assert plain.mlp_hidden_dim is None


@pytest.mark.parametrize(
    "override",
    ["head_dim=0", "head_dim=2.5", "mlp_hidden_dim=0", "mlp_hidden_dim=true"],
)
def test_load_config_rejects_bad_qwen3_switches(tmp_path: Path, override: str) -> None:
    from tlab.train.config import load_config

    path = write_yaml(tmp_path / "c.yaml", "n_heads: 6\nmlp: swiglu\n")

    with pytest.raises(ValueError):
        load_config(path, [override])


def test_load_config_rejects_mlp_hidden_dim_with_gelu(tmp_path: Path) -> None:
    from tlab.train.config import load_config

    path = write_yaml(tmp_path / "c.yaml", "n_heads: 6\nmlp_hidden_dim: 96\n")

    with pytest.raises(ValueError, match="mlp_hidden_dim"):
        load_config(path)


QWEN3_SWITCHES = {
    "n_kv_heads": 1,
    "head_dim": 32,
    "qk_norm": True,
    "norm": "rmsnorm",
    "mlp": "swiglu",
    "mlp_hidden_dim": 48,
    "pos_encoding": "rope",
}


def test_run_with_qwen3_switches_saves_a_checkpoint_that_rebuilds(
    tmp_path: Path,
) -> None:
    from tlab.train.run import run
    from tlab.train.utils import load_model

    write_tiny_dataset(tmp_path / "data")
    path = run(tiny_config(tmp_path, max_iters=2, **QWEN3_SWITCHES)) / "ckpt.pt"

    model, _ = load_model(path, torch.device("cpu"))

    attn = model.blocks[0].attn
    assert attn.q_proj.weight.shape == (64, 32)
    assert attn.q_norm.weight.shape == (32,)
    assert model.blocks[0].mlp.gate_proj.weight.shape == (48, 32)


@pytest.mark.parametrize(
    "changed", [{"head_dim": 16}, {"qk_norm": False}, {"mlp_hidden_dim": 64}]
)
def test_resume_refuses_different_qwen3_switches(tmp_path: Path, changed: dict) -> None:
    from tlab.train.run import run

    write_tiny_dataset(tmp_path / "data")
    run(tiny_config(tmp_path, max_iters=2, **QWEN3_SWITCHES))

    name = next(iter(changed))
    with pytest.raises(ValueError, match=name):
        run(
            tiny_config(
                tmp_path, max_iters=3, resume=True, **{**QWEN3_SWITCHES, **changed}
            )
        )


@pytest.mark.parametrize(
    "override",
    ["qk_norm=flase", "qk_norm=maybe", "qk_norm=2", 'qk_norm="false"', "resume=2"],
)
def test_load_config_rejects_values_that_are_not_booleans(
    tmp_path: Path, override: str
) -> None:
    from tlab.train.config import load_config

    path = write_yaml(tmp_path / "c.yaml", "n_heads: 6\n")

    # bool() 只要不是空值都得到 True，拼错的取值会被悄悄当成开启
    with pytest.raises(ValueError, match="布尔"):
        load_config(path, [override])


def test_load_config_rejects_an_odd_head_dim_with_rope(tmp_path: Path) -> None:
    from tlab.train.config import load_config

    path = write_yaml(
        tmp_path / "c.yaml", "n_heads: 6\npos_encoding: rope\nhead_dim: 15\n"
    )

    with pytest.raises(ValueError, match="head_dim"):
        load_config(path)


def test_train_config_checks_rmsnorm_fp32_before_building_the_model() -> None:
    from tlab.train.config import TrainConfig

    # 默认是 LayerNorm、不开 QK-Norm，模型里没有 RMSNorm：建配置时就报错，不用等到建模型
    with pytest.raises(ValueError, match="rmsnorm_fp32"):
        TrainConfig(rmsnorm_fp32=True)
    with pytest.raises(ValueError, match="rmsnorm_fp32"):
        TrainConfig(norm="rmsnorm", rmsnorm_fp32="true")
    assert TrainConfig(norm="rmsnorm", rmsnorm_fp32=True).rmsnorm_fp32 is True
