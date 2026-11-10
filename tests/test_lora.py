"""LoRALinear：B 全 0 时和原来相同、手算、梯度、merge 和 unmerge、参数量。"""

import math

import pytest
import torch
from torch import nn

# LoRALinear 在 tlab/finetune/lora.py；在各测试内导入，文件出错时每个测试单独报错


def make_lora(in_features: int, out_features: int, r: int, alpha: float):
    from tlab.finetune.lora import LoRALinear

    torch.manual_seed(0)
    base = nn.Linear(in_features, out_features, bias=False)
    return LoRALinear(base, r, alpha)


def test_new_lora_gives_exactly_the_output_of_the_base_layer() -> None:
    lora = make_lora(16, 8, r=4, alpha=8)
    x = torch.randn(2, 5, 16)  # (B, T, in)

    # B 全 0，旁路的输出正好是 0：模型刚装上 LoRA 时和原来逐位相同
    assert torch.equal(lora(x), lora.base(x))  # (B, T, out)


def test_shapes_names_and_frozen_base() -> None:
    lora = make_lora(16, 8, r=4, alpha=8)

    assert lora.lora_A.shape == (4, 16)  # (r, in)：把 in 维降到 r 维
    assert lora.lora_B.shape == (8, 4)  # (out, r)：把 r 维升回 out 维
    # 名字要固定：存档和 PEFT 对照都按这两个名字找
    assert {name for name, _ in lora.named_parameters()} == {
        "base.weight",
        "lora_A",
        "lora_B",
    }
    assert lora.base.weight.requires_grad is False
    assert lora.lora_A.requires_grad is True
    assert lora.lora_B.requires_grad is True
    assert torch.equal(lora.lora_B, torch.zeros(8, 4))
    assert lora.scaling == 2.0  # alpha / r = 8 / 4


def test_a_is_initialized_with_kaiming_uniform_like_peft() -> None:
    from tlab.finetune.lora import LoRALinear

    base = nn.Linear(16, 8, bias=False)
    torch.manual_seed(3)
    lora = LoRALinear(base, 4, 8)
    # PEFT 的 LoRA 层用的就是这个初始化；这里还没装 peft，只按它的写法对照，
    # 和真的 PEFT 对比放在 F4（那时 A、B 的初始值会直接拷过去）
    torch.manual_seed(3)
    expected = torch.empty(4, 16)  # (r, in)
    nn.init.kaiming_uniform_(expected, a=math.sqrt(5))

    assert torch.equal(lora.lora_A, expected)
    # 范围是 ±1/√in：in=16 时是 ±0.25
    assert lora.lora_A.abs().max() <= 0.25


def test_hand_computed_example() -> None:
    lora = make_lora(2, 2, r=1, alpha=2)  # scaling = 2 / 1 = 2
    with torch.no_grad():
        lora.base.weight.copy_(torch.eye(2))  # W = I
        lora.lora_A.copy_(torch.tensor([[1.0, 1.0]]))  # (r, in) = (1, 2)
        lora.lora_B.copy_(torch.tensor([[1.0], [2.0]]))  # (out, r) = (2, 1)
    x = torch.tensor([[1.0, 2.0]])  # (1, in)

    # 原来的输出：x Wᵀ = [1, 2]
    # 旁路：x Aᵀ = 1 + 2 = 3；再乘 Bᵀ = [3, 6]；乘 scaling 2 = [6, 12]
    # 相加：[7, 14]
    assert torch.equal(lora(x), torch.tensor([[7.0, 14.0]]))  # (1, out)


def test_only_b_changes_in_the_first_step() -> None:
    lora = make_lora(16, 8, r=4, alpha=8)
    x = torch.randn(2, 5, 16)  # (B, T, in)

    lora(x).pow(2).sum().backward()

    # B 全 0，所以 A 的梯度要乘上 B，也是 0；B 的梯度要乘上 A·x，不是 0
    assert lora.base.weight.grad is None  # 冻结：根本不算
    assert torch.equal(lora.lora_A.grad, torch.zeros(4, 16))
    assert lora.lora_B.grad.abs().sum() > 0


def trained_lora():
    """B 不再是 0 的 LoRA，像训练过几步之后。"""
    lora = make_lora(16, 8, r=4, alpha=8)
    with torch.no_grad():
        lora.lora_B.normal_()
    return lora


def test_merge_gives_the_same_output_and_changes_the_weight() -> None:
    lora = trained_lora()
    x = torch.randn(2, 5, 16)  # (B, T, in)
    before = lora(x)  # (B, T, out)
    weight_before = lora.base.weight.clone()  # (out, in)

    lora.merge()

    assert lora.merged is True
    assert not torch.equal(lora.base.weight, weight_before)
    # 浮点舍入的顺序不同，所以用容差
    torch.testing.assert_close(lora(x), before)
    # 已经合并：W 里包含了旁路，输出就是 base 的输出，不再另算旁路
    assert torch.equal(lora(x), lora.base(x))


def test_merging_twice_does_not_add_twice() -> None:
    lora = trained_lora()
    lora.merge()
    once = lora.base.weight.clone()  # (out, in)

    lora.merge()

    assert torch.equal(lora.base.weight, once)


def test_unmerge_restores_the_weight_and_the_output() -> None:
    lora = trained_lora()
    x = torch.randn(2, 5, 16)  # (B, T, in)
    weight_before = lora.base.weight.clone()  # (out, in)
    before = lora(x)  # (B, T, out)

    lora.merge()
    lora.unmerge()

    assert lora.merged is False
    torch.testing.assert_close(lora.base.weight, weight_before)
    torch.testing.assert_close(lora(x), before)


def test_unmerge_without_merge_does_nothing() -> None:
    lora = trained_lora()
    weight_before = lora.base.weight.clone()  # (out, in)

    lora.unmerge()

    assert lora.merged is False
    assert torch.equal(lora.base.weight, weight_before)


@pytest.mark.parametrize(
    ("in_features", "out_features", "r", "expected"),
    [
        (1024, 2048, 8, 24_576),  # Qwen3 的 q_proj：8 × (1024 + 2048)
        (1024, 1024, 8, 16_384),  # k_proj、v_proj：8 × (1024 + 1024)
        (2048, 1024, 16, 49_152),  # out_proj，r=16：16 × (2048 + 1024)
    ],
)
def test_trainable_parameter_count(
    in_features: int, out_features: int, r: int, expected: int
) -> None:
    lora = make_lora(in_features, out_features, r, alpha=2 * r)

    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)

    assert trainable == expected


def test_scaling_is_alpha_over_r() -> None:
    lora = make_lora(2, 2, r=2, alpha=6)  # scaling = 6 / 2 = 3
    with torch.no_grad():
        lora.base.weight.zero_()  # 只看旁路
        lora.lora_A.copy_(torch.eye(2))  # (r, in) = (2, 2)
        lora.lora_B.copy_(torch.eye(2))  # (out, r) = (2, 2)
    x = torch.tensor([[1.0, 2.0]])  # (1, in)

    # B A = I，旁路就是 3x：乘 alpha 会得到 [6, 12]，忘了缩放会得到 [1, 2]
    assert torch.equal(lora(x), torch.tensor([[3.0, 6.0]]))  # (1, out)


def test_lora_remembers_r_and_alpha() -> None:
    lora = make_lora(16, 8, r=4, alpha=7)

    # 存档要记下原来的 r 和 alpha；用 scaling × r 反推会有舍入误差
    assert lora.r == 4
    assert lora.alpha == 7


def test_bias_of_the_base_is_frozen_too() -> None:
    from tlab.finetune.lora import LoRALinear

    base = nn.Linear(16, 8, bias=True)
    lora = LoRALinear(base, 4, 8)

    # Qwen3 的投影没有 bias；但传进来的 Linear 有 bias 时，它也不该被训练
    assert base.bias.requires_grad is False
    trainable = [name for name, p in lora.named_parameters() if p.requires_grad]
    assert sorted(trainable) == ["lora_A", "lora_B"]
