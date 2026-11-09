"""把 LoRA 装进模型（agent 编写）：替换哪些层、冻结、参数量、只存 A 和 B、合并。"""

import hashlib
from pathlib import Path

import pytest
import torch
from conftest import big_lora_model
from torch import nn

from tlab.convert.qwen3 import load_qwen3
from tlab.eval.generate import generate
from tlab.finetune.lora import LoRALinear
from tlab.finetune.lora_model import (
    apply_lora,
    count_trainable,
    load_lora,
    load_qwen3_with_lora,
    merge_lora,
    save_lora,
)
from tlab.model.gpt import GPT, GPTConfig

TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")


def small_qwen3(n_layers: int = 2) -> GPT:
    """结构和 Qwen3 相同的小模型：q 的总维度 h·d = 64，不等于 C = 32。"""
    torch.manual_seed(0)
    return GPT(
        GPTConfig(
            vocab_size=50,
            block_size=32,
            d_model=32,
            n_layers=n_layers,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            qk_norm=True,
            norm="rmsnorm",
            mlp="swiglu",
            pos_encoding="rope",
            rmsnorm_fp32=True,
        )
    )


def with_trained_b(model: GPT) -> GPT:
    """A、B 都变了，像训练过几步之后：B 是 0 时合并、存档前后的输出本来就一样；
    A 还是初始值时，同一个种子新建的模型也一样，测不出有没有载入 A。"""
    torch.manual_seed(1)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.normal_(std=0.1)
                module.lora_B.normal_(std=0.1)
    return model


# ---- 装在哪里 ----


def test_apply_lora_replaces_the_four_projections_of_every_layer() -> None:
    model = small_qwen3()

    replaced = apply_lora(model, r=4, alpha=8)

    assert replaced == [f"blocks.{i}.attn.{t}" for i in range(2) for t in TARGETS]
    for i in range(2):
        attn = model.blocks[i].attn
        assert all(isinstance(getattr(attn, t), LoRALinear) for t in TARGETS)
    # 前馈子层不动
    assert isinstance(model.blocks[0].mlp.gate_proj, nn.Linear)


def test_apply_lora_can_target_only_some_projections() -> None:
    model = small_qwen3()

    replaced = apply_lora(model, r=4, alpha=8, targets=("q_proj", "v_proj"))

    assert replaced == [
        f"blocks.{i}.attn.{t}" for i in range(2) for t in ("q_proj", "v_proj")
    ]
    assert isinstance(model.blocks[0].attn.k_proj, nn.Linear)


@pytest.mark.parametrize("targets", [("q_proj", "qkv"), ()])
def test_apply_lora_rejects_bad_targets(targets: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="targets"):
        apply_lora(small_qwen3(), r=4, alpha=8, targets=targets)


def test_apply_lora_twice_is_an_error() -> None:
    model = small_qwen3()
    apply_lora(model, r=4, alpha=8)

    # 再装一次会把 LoRALinear 里的 base 也包一层，参数量和存档都乱掉
    with pytest.raises(ValueError, match="已经"):
        apply_lora(model, r=4, alpha=8)


def test_apply_lora_to_a_model_without_attention_is_an_error() -> None:
    with pytest.raises(ValueError, match="没有"):
        apply_lora(nn.Sequential(nn.Linear(4, 4)), r=2, alpha=4)


# ---- 刚装上时什么都不变 ----


@torch.no_grad()
def test_new_lora_model_gives_the_same_logits_and_generation() -> None:
    base = small_qwen3()
    model = small_qwen3()
    apply_lora(model, r=4, alpha=8)
    idx = torch.randint(0, 50, (2, 10), generator=torch.Generator().manual_seed(2))

    assert torch.equal(model(idx), base(idx))  # (B, T, V)
    prompt = idx[:1, :5]  # (1, 5)
    expected = generate(base, prompt, 8, top_k=1)  # (1, 13)
    assert torch.equal(generate(model, prompt, 8, top_k=1), expected)
    assert torch.equal(generate(model, prompt, 8, top_k=1, use_cache=True), expected)


@torch.no_grad()
def test_trained_lora_model_gives_the_same_text_with_and_without_cache() -> None:
    model = small_qwen3()
    apply_lora(model, r=4, alpha=8)
    with_trained_b(model)
    prompt = torch.randint(0, 50, (1, 5), generator=torch.Generator().manual_seed(3))

    plain = generate(model, prompt, 8, top_k=1)  # (1, 13)

    assert torch.equal(generate(model, prompt, 8, top_k=1, use_cache=True), plain)


# ---- 只训练 A 和 B ----


def test_only_lora_parameters_are_trainable_and_get_gradients() -> None:
    model = small_qwen3()
    apply_lora(model, r=4, alpha=8)
    idx = torch.randint(0, 50, (2, 10), generator=torch.Generator().manual_seed(2))

    model(idx).float().pow(2).mean().backward()  # (B, T, V) → ()

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == [
        f"blocks.{i}.attn.{t}.{ab}"
        for i in range(2)
        for t in TARGETS
        for ab in ("lora_A", "lora_B")
    ]
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
        else:
            assert parameter.grad is None, name  # 词表、归一化、MLP、W 都不算梯度


def test_trainable_count_of_the_small_model() -> None:
    model = small_qwen3()
    apply_lora(model, r=4, alpha=8)

    # 每层：q 4·(32+64)，k 4·(32+32)，v 4·(32+32)，out 4·(64+32)，共 1,280；2 层 2,560
    assert count_trainable(model) == 2_560


def test_trainable_count_of_qwen3_0_6b_at_rank_8() -> None:
    from tlab.convert.qwen3 import gpt_config_from_hf

    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "max_position_embeddings": 40960,
        "model_type": "qwen3",
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "rope_theta": 1000000,
        "tie_word_embeddings": True,
        "use_sliding_window": False,
        "vocab_size": 151936,
    }
    # 用 meta 设备建模型：只有形状，不分配内存
    with torch.device("meta"):
        model = GPT(gpt_config_from_hf(config))
        apply_lora(model, r=8, alpha=16)

    # 计划里手算的：每层 10,240·r，28 层 286,720·r，r=8 时 2,293,760
    assert count_trainable(model) == 2_293_760


# ---- 只存 A 和 B ----


def test_saved_lora_holds_only_a_and_b_and_the_settings(tmp_path: Path) -> None:
    model = small_qwen3()
    apply_lora(model, r=4, alpha=8, targets=("q_proj", "v_proj"))
    path = tmp_path / "lora.pt"

    save_lora(model, path)

    saved = torch.load(path, weights_only=True)
    # 不存基座：除了 A、B 只有三个设置
    assert set(saved) == {"r", "alpha", "targets", "weights"}
    assert saved["r"] == 4
    assert saved["alpha"] == 8
    assert saved["targets"] == ["q_proj", "v_proj"]
    assert sorted(saved["weights"]) == sorted(
        f"blocks.{i}.attn.{t}.{ab}"
        for i in range(2)
        for t in ("q_proj", "v_proj")
        for ab in ("lora_A", "lora_B")
    )


@torch.no_grad()
def test_loading_saved_lora_gives_the_same_output(tmp_path: Path) -> None:
    trained = small_qwen3()
    apply_lora(trained, r=4, alpha=8)
    with_trained_b(trained)
    path = tmp_path / "lora.pt"
    save_lora(trained, path)
    idx = torch.randint(0, 50, (2, 10), generator=torch.Generator().manual_seed(2))

    # 先载基座（这里用同一个种子建，相当于同一份基座权重），再装 LoRA、载存档
    fresh = small_qwen3()
    settings = load_lora(fresh, path)

    assert settings == {"r": 4, "alpha": 8, "targets": list(TARGETS)}
    assert torch.equal(fresh(idx), trained(idx))  # (B, T, V)


def test_loading_lora_into_a_different_structure_is_an_error(tmp_path: Path) -> None:
    trained = small_qwen3(n_layers=2)
    apply_lora(trained, r=4, alpha=8)
    path = tmp_path / "lora.pt"
    save_lora(trained, path)

    # 3 层的模型多出第 2 层的 A、B，存档里没有：不能悄悄留着随机值
    model = small_qwen3(n_layers=3)
    with pytest.raises(ValueError, match="blocks.2"):
        load_lora(model, path)
    # 出错时模型保持原样，没有装上一半的 LoRA
    assert not any(isinstance(m, LoRALinear) for m in model.modules())


def test_loading_lora_whose_rank_does_not_match_its_weights_is_an_error(
    tmp_path: Path,
) -> None:
    trained = small_qwen3()
    apply_lora(trained, r=4, alpha=8)
    path = tmp_path / "lora.pt"
    save_lora(trained, path)
    saved = torch.load(path, weights_only=True)
    saved["r"] = 8  # 设置写的是 8，权重却是 r=4 的形状
    torch.save(saved, path)

    model = small_qwen3()
    with pytest.raises(ValueError, match="形状"):
        load_lora(model, path)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())


def test_loading_lora_into_a_model_that_already_has_lora_is_an_error(
    tmp_path: Path,
) -> None:
    trained = small_qwen3()
    apply_lora(trained, r=4, alpha=8)
    path = tmp_path / "lora.pt"
    save_lora(trained, path)

    with pytest.raises(ValueError, match="已经"):
        load_lora(trained, path)


# ---- 合并回普通模型 ----


@torch.no_grad()
def test_merge_lora_gives_a_plain_model_with_the_same_output() -> None:
    model = small_qwen3()
    plain_names = list(small_qwen3().state_dict())
    apply_lora(model, r=4, alpha=8)
    with_trained_b(model)
    idx = torch.randint(0, 50, (2, 10), generator=torch.Generator().manual_seed(2))
    before = model(idx)  # (B, T, V)

    merged = merge_lora(model)

    assert merged is model
    assert not any(isinstance(m, LoRALinear) for m in merged.modules())
    # 参数名和原模型完全相同：可以按普通模型存档，评测脚本也能直接用
    assert list(merged.state_dict()) == plain_names
    torch.testing.assert_close(merged(idx), before)


def test_merge_lora_without_lora_is_an_error() -> None:
    with pytest.raises(ValueError, match="没有"):
        merge_lora(small_qwen3())


def test_saving_lora_with_different_ranks_is_an_error(tmp_path: Path) -> None:
    model = small_qwen3()
    apply_lora(model, r=4, alpha=8)
    # 有一层换成 r=2：存档只能记一个 r，这样存下去载回来会对不上
    attn = model.blocks[0].attn
    attn.q_proj = LoRALinear(attn.q_proj.base, 2, 8)

    with pytest.raises(ValueError, match="不同"):
        save_lora(model, tmp_path / "lora.pt")


def test_lora_follows_the_device_of_the_base_weights() -> None:
    # meta 设备只有形状没有数据，不用显卡也能测“A、B 和 W 在同一个设备上”
    model = small_qwen3().to("meta")

    apply_lora(model, r=4, alpha=8)

    for parameter in lora_weights_of(model):
        assert parameter.device.type == "meta"
        assert parameter.dtype == torch.float32


def lora_weights_of(model: nn.Module) -> list[torch.Tensor]:
    return [
        p
        for m in model.modules()
        if isinstance(m, LoRALinear)
        for p in (m.lora_A, m.lora_B)
    ]


def test_merge_lora_refuses_bf16_weights() -> None:
    model = small_qwen3()
    apply_lora(model, r=4, alpha=8)
    model.to(torch.bfloat16)

    # bf16 下 ΔW 会被舍掉一部分，合并后的模型就和训练时不一样了
    with pytest.raises(ValueError, match="fp32"):
        merge_lora(model)


@pytest.mark.parametrize(("r", "alpha"), [(49, 1), (25, 7), (3, 1)])
def test_saved_alpha_is_exactly_the_one_given(
    tmp_path: Path, r: int, alpha: float
) -> None:
    model = small_qwen3()
    apply_lora(model, r=r, alpha=alpha)
    path = tmp_path / "lora.pt"

    save_lora(model, path)

    # 不能从 alpha / r 反推：0.0204... × 49 不一定正好是 1
    assert torch.load(path, weights_only=True)["alpha"] == alpha


# ---- 载入基座、装上 LoRA、合并（评测和自己提问共用）----


IDS = torch.tensor([[5, 9, 2, 33, 7]])  # (1, T)


def test_load_qwen3_with_lora_records_the_file_and_merges(
    tmp_path: Path, tiny_model_dir: Path
) -> None:
    path = tmp_path / "lora.pt"
    lora_model = big_lora_model(tiny_model_dir)
    save_lora(lora_model, path)
    with torch.no_grad():
        expected = lora_model(IDS)  # 合并前的 logits，(1, T, V)

    model, info = load_qwen3_with_lora(tiny_model_dir, torch.device("cpu"), path)

    assert info == {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "r": 4,
        "alpha": 8.0,
        "targets": list(TARGETS),
    }
    # 已经合并成普通的 Linear，算出来和没合并时一样
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    with torch.no_grad():
        torch.testing.assert_close(model(IDS), expected)


def test_load_qwen3_without_lora_is_the_plain_base(tiny_model_dir: Path) -> None:
    model, info = load_qwen3_with_lora(tiny_model_dir, torch.device("cpu"), None)

    assert info is None
    base = load_qwen3(tiny_model_dir, torch.device("cpu"))
    with torch.no_grad():
        assert torch.equal(model(IDS), base(IDS))


def test_missing_lora_file_is_an_error_before_loading_the_base(
    tmp_path: Path,
) -> None:
    # 基座目录也不存在：先载基座的话，报的会是找不到 config.json
    with pytest.raises(FileNotFoundError, match="nope.pt"):
        load_qwen3_with_lora(
            tmp_path / "no_model", torch.device("cpu"), tmp_path / "nope.pt"
        )
