import pytest
import torch
import torch.nn.functional as F

# RMSNorm 由用户手写在 tlab/model/layers.py；在各测试内导入，文件还没写时每个测试单独报错


def test_rmsnorm_has_only_a_weight_initialized_to_ones() -> None:
    from tlab.model.layers import RMSNorm

    norm = RMSNorm(8)

    # 只有逐分量缩放 weight，没有平移 bias：参数量是 C，LayerNorm 是 2C
    # 名字必须叫 weight：M4 载入 Qwen3 时按名字对应 "...norm.weight"
    params = dict(norm.named_parameters())
    assert list(params) == ["weight"]
    assert params["weight"].shape == (8,)
    torch.testing.assert_close(params["weight"].detach(), torch.ones(8))


def test_rmsnorm_matches_hand_computed_values() -> None:
    from tlab.model.layers import RMSNorm

    norm = RMSNorm(4, eps=0.0)
    # 两个 token，各自的 RMS 不同：每个 token 只用自己的 C 个分量归一化
    x = torch.tensor([[[1.0, 2.0, 2.0, 4.0], [6.0, 0.0, 0.0, 8.0]]])  # (1, 2, 4)

    # token 0：均方 (1 + 4 + 4 + 16) / 4 = 6.25，RMS = 2.5，每个分量除以 2.5
    # token 1：均方 (36 + 0 + 0 + 64) / 4 = 25，RMS = 5，每个分量除以 5
    y = norm(x)  # (1, 2, 4)
    expected = torch.tensor([[[0.4, 0.8, 0.8, 1.6], [1.2, 0.0, 0.0, 1.6]]])  # (1, 2, 4)
    torch.testing.assert_close(y, expected)

    # 再逐分量乘以 weight = [1, 2, 3, 4]
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))  # (C,)
    y = norm(x)  # (1, 2, 4)
    expected = torch.tensor([[[0.4, 1.6, 2.4, 6.4], [1.2, 0.0, 0.0, 6.4]]])  # (1, 2, 4)
    torch.testing.assert_close(y, expected)


def test_rmsnorm_does_not_subtract_the_mean() -> None:
    from tlab.model.layers import RMSNorm

    norm = RMSNorm(4, eps=0.0)
    x = torch.full((1, 1, 4), 5.0)  # (1, 1, 4)，四个分量都是 5

    # RMS = 5，结果全是 1；LayerNorm 会先减去均值 5，输出全 0
    torch.testing.assert_close(norm(x), torch.ones(1, 1, 4))


def test_rmsnorm_adds_eps_inside_the_square_root() -> None:
    from tlab.model.layers import RMSNorm

    # 用默认 eps（1e-6）；分量很小时 eps 的位置才看得出来
    norm = RMSNorm(4)
    x = torch.full((1, 1, 4), 1e-3)  # (1, 1, 4)

    # 均方 1e-6，加在根号里：1e-3 / √(1e-6 + 1e-6) = 1/√2 ≈ 0.7071
    # 加在根号外会得到 1e-3 / (1e-3 + 1e-6) ≈ 0.999
    expected = torch.full((1, 1, 4), 2**-0.5)
    torch.testing.assert_close(norm(x), expected, rtol=1e-5, atol=1e-6)

    # 全零输入：有 eps 就不会 0 / 0，输出 0 而不是 NaN
    zeros = torch.zeros(1, 1, 4)  # (1, 1, 4)
    torch.testing.assert_close(norm(zeros), zeros)


def test_rmsnorm_matches_reference_rms_norm() -> None:
    from tlab.model.layers import RMSNorm

    torch.manual_seed(0)
    norm = RMSNorm(8)
    # 随机化 weight，才能检查 weight 有没有逐分量乘对
    with torch.no_grad():
        norm.weight.normal_()
    x = torch.randn(2, 5, 8)  # (B, T, C)

    # F.rms_norm 只在测试里当参考答案（与注意力测试用 SDPA 的做法相同）
    expected = F.rms_norm(x, (8,), norm.weight, eps=1e-6)  # (B, T, C)
    torch.testing.assert_close(norm(x), expected)


# ---- RMSNorm 内部转 fp32（M5，和 HF 的 Qwen3 相同）----


def random_norm(d_model: int, **kwargs: bool) -> torch.nn.Module:
    """weight 随机的 RMSNorm：全 1 时乘不乘 weight、先乘后转都看不出差别。"""
    from tlab.model.layers import RMSNorm

    torch.manual_seed(0)
    norm = RMSNorm(d_model, **kwargs)
    with torch.no_grad():
        norm.weight.normal_()
    return norm


def old_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """M1–M3 用的写法，用来检查默认行为没变。"""
    ms = x.pow(2).mean(dim=-1, keepdim=True)  # (B, T, 1)
    return x * torch.rsqrt(ms + 1e-6) * weight  # (B, T, C)


def test_compute_fp32_is_off_by_default() -> None:
    from tlab.model.layers import RMSNorm

    assert RMSNorm(8).compute_fp32 is False


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_without_fp32_is_bit_for_bit_unchanged(dtype: torch.dtype) -> None:
    norm = random_norm(128)
    x = torch.randn(2, 5, 128).to(dtype)  # (B, T, C)

    assert torch.equal(norm(x), old_rmsnorm(x, norm.weight))


def test_compute_fp32_changes_nothing_for_fp32_inputs() -> None:
    on = random_norm(128, compute_fp32=True)
    off = random_norm(128)
    x = torch.randn(2, 5, 128)  # (B, T, C)，fp32

    assert torch.equal(on(x), off(x))


def test_compute_fp32_normalizes_bf16_inputs_in_fp32_then_casts_back() -> None:
    on = random_norm(128, compute_fp32=True)
    off = random_norm(128)
    torch.manual_seed(1)
    x = torch.randn(2, 5, 128).to(torch.bfloat16)  # (B, T, C)

    # 手工按 HF 的顺序算：转 fp32 → 归一化 → 转回 bf16 → 乘 weight
    x32 = x.float()  # (B, T, C)
    normed = x32 * torch.rsqrt(
        x32.pow(2).mean(dim=-1, keepdim=True) + 1e-6
    )  # (B, T, C)
    expected = normed.to(torch.bfloat16) * on.weight  # (B, T, C)
    y = on(x)  # (B, T, C)

    assert torch.equal(y, expected)
    assert y.dtype == expected.dtype
    # 开不开确实有差别，否则这个开关没有意义
    assert not torch.equal(y, off(x))


def test_compute_fp32_still_passes_gradients_back() -> None:
    norm = random_norm(16, compute_fp32=True)
    x = torch.randn(2, 3, 16).to(torch.bfloat16).requires_grad_()  # (B, T, C)
    x_ref = x.detach().clone().requires_grad_()  # (B, T, C)
    weight_ref = norm.weight.detach().clone().requires_grad_()  # (C,)

    norm(x).float().sum().backward()
    # 参考：同样的写法手工算一遍。中间漏了梯度（比如 detach）时两边会不同
    x32 = x_ref.float()  # (B, T, C)
    normed = x32 * torch.rsqrt(
        x32.pow(2).mean(dim=-1, keepdim=True) + 1e-6
    )  # (B, T, C)
    (normed.to(torch.bfloat16) * weight_ref).float().sum().backward()

    assert x.grad is not None and norm.weight.grad is not None
    assert torch.equal(x.grad, x_ref.grad)
    assert torch.equal(norm.weight.grad, weight_ref.grad)


def test_compute_fp32_is_not_saved_in_the_checkpoint() -> None:
    from tlab.model.layers import RMSNorm

    on = RMSNorm(8, compute_fp32=True)

    # 只是一个普通属性：存档里只有 weight，旧存档也能直接载入
    assert list(on.state_dict()) == ["weight"]
    on.load_state_dict(RMSNorm(8).state_dict())
    assert on.compute_fp32 is True


def test_swiglu_has_three_linear_layers_without_bias() -> None:
    from tlab.model.layers import SwiGLU

    swiglu = SwiGLU(6, 16)

    # Linear 的 weight 形状是 (输出, 输入)：两路把 C 升到隐藏维度 H，再把 H 降回 C
    assert swiglu.gate_proj.weight.shape == (16, 6)
    assert swiglu.up_proj.weight.shape == (16, 6)
    assert swiglu.down_proj.weight.shape == (6, 16)
    # 没有 bias：3·C·H = 288，C = 6、H = 8C/3 = 16 时正好等于 GELU MLP 的 8C²
    assert sum(p.numel() for p in swiglu.parameters()) == 288


def test_swiglu_matches_hand_computed_values() -> None:
    from tlab.model.layers import SwiGLU

    swiglu = SwiGLU(2, 1)
    with torch.no_grad():
        # 门只看第 0 个分量，内容只看第 1 个分量
        swiglu.gate_proj.weight.copy_(torch.tensor([[1.0, 0.0]]))  # (H, C)
        swiglu.up_proj.weight.copy_(torch.tensor([[0.0, 1.0]]))  # (H, C)
        swiglu.down_proj.weight.copy_(torch.tensor([[1.0], [2.0]]))  # (C, H)
    x = torch.tensor([[[1.0, 2.0], [-1.0, 2.0]]])  # (1, 2, 2)

    y = swiglu(x)  # (1, 2, 2)

    # token 0：SiLU(1) = 1·σ(1) = 0.731059，乘内容 2 得 1.462117，缩回 [×1, ×2]
    # token 1：SiLU(−1) = −1·σ(−1) = −0.268941，乘内容 2 得 −0.537883（ReLU 会得 0）
    expected = torch.tensor(
        [[[1.462117, 2.924234], [-0.537883, -1.075766]]]
    )  # (1, 2, 2)
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)


def test_swiglu_output_is_zero_when_the_gate_is_closed() -> None:
    from tlab.model.layers import SwiGLU

    torch.manual_seed(0)
    swiglu = SwiGLU(6, 16)
    # 门的权重全为 0：SiLU(0) = 0，不管内容那一路是什么，乘完都是 0
    with torch.no_grad():
        swiglu.gate_proj.weight.zero_()  # (H, C)
    x = torch.randn(2, 5, 6)  # (B, T, C)

    y = swiglu(x)  # (B, T, C)
    torch.testing.assert_close(y, torch.zeros(2, 5, 6))


def test_swiglu_keeps_shape_and_does_not_mix_tokens() -> None:
    from tlab.model.layers import SwiGLU

    torch.manual_seed(0)
    swiglu = SwiGLU(6, 16)
    x = torch.randn(2, 5, 6)  # (B, T, C)
    # 只改第 0 个 token
    x_changed = x.clone()  # (B, T, C)
    x_changed[:, 0, :] = torch.randn(2, 6)  # (B, C)

    y = swiglu(x)  # (B, T, C)
    y_changed = swiglu(x_changed)  # (B, T, C)

    assert y.shape == (2, 5, 6)
    # 和 MLP 一样，每个 token 各自加工：其余 token 的输出不变，被改的那个要变
    torch.testing.assert_close(y_changed[:, 1:, :], y[:, 1:, :])
    assert not torch.allclose(y_changed[:, 0, :], y[:, 0, :])


def rotate_one(
    rope: torch.nn.Module, vector: torch.Tensor, position: int, T: int
) -> torch.Tensor:
    """把 vector 放在第 position 个位置（其余位置填 0），返回旋转后这个位置的向量。"""
    x = torch.zeros(1, 1, T, vector.numel())  # (B, h, T, d)
    x[0, 0, position] = vector  # (d,)
    return rope(x)[0, 0, position]  # (d,)


def test_rotary_has_no_parameters_and_stores_nothing_in_checkpoints() -> None:
    from tlab.model.layers import RotaryEmbedding

    rope = RotaryEmbedding(8)

    # RoPE 只用公式旋转，没有要学的东西；转速表也不进存档，旧存档才不受影响
    assert sum(p.numel() for p in rope.parameters()) == 0
    assert rope.state_dict() == {}
    # 分量要两两配对旋转，维度必须是偶数
    with pytest.raises(AssertionError):
        RotaryEmbedding(7)


def test_rotary_matches_hand_computed_values_for_one_pair() -> None:
    from tlab.model.layers import RotaryEmbedding

    # d = 2 只有一对分量，转速 1：位置 t 转 t 弧度
    x = torch.tensor([1.0, 0.0]).repeat(1, 1, 3, 1)  # (1, 1, 3, 2)

    y = RotaryEmbedding(2)(x)  # (1, 1, 3, 2)

    # [1, 0] 转 t 弧度是 [cos t, sin t]；位置 0 不转
    expected = torch.tensor(
        [[[[1.0, 0.0], [0.540302, 0.841471], [-0.416147, 0.909297]]]]
    )  # (1, 1, 3, 2)
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)


def test_rotary_pairs_the_two_halves_at_different_speeds() -> None:
    from tlab.model.layers import RotaryEmbedding

    # d = 4：x0 与 x2 一对，x1 与 x3 一对。x = [1, 1, 0, 0]，输出是
    # [cos(快), cos(慢), sin(快), sin(慢)]
    x = torch.tensor([1.0, 1.0, 0.0, 0.0]).repeat(1, 1, 101, 1)  # (1, 1, 101, 4)

    # theta = 10000：转速 1 和 1/100，位置 100 分别转 100 和 1 弧度
    y = RotaryEmbedding(4)(x)[0, 0, 100]  # (4,)
    expected = torch.tensor([0.862319, 0.540302, -0.506366, 0.841471])  # (4,)
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)

    # theta = 100：转速 1 和 1/10，位置 10 分别转 10 和 1 弧度
    y = RotaryEmbedding(4, theta=100.0)(x)[0, 0, 10]  # (4,)
    expected = torch.tensor([-0.839072, 0.540302, -0.544021, 0.841471])  # (4,)
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)


def test_rotary_keeps_every_vector_length() -> None:
    from tlab.model.layers import RotaryEmbedding

    torch.manual_seed(0)
    x = torch.randn(2, 3, 16, 8)  # (B, h, T, d)

    y = RotaryEmbedding(8)(x)  # (B, h, T, d)

    # 旋转只改方向，不改长度
    assert y.shape == x.shape
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))  # (B, h, T)


def test_rotary_score_depends_only_on_the_relative_distance() -> None:
    from tlab.model.layers import RotaryEmbedding

    torch.manual_seed(0)
    rope = RotaryEmbedding(64)
    q, k = torch.randn(64), torch.randn(64)  # (d,), (d,)

    def score(q_position: int, k_position: int) -> torch.Tensor:
        return rotate_one(rope, q, q_position, 64) @ rotate_one(rope, k, k_position, 64)

    # RoPE 的核心性质：绝对位置变了，只要距离相同，点积就相同
    base = score(1, 3)  # ()
    torch.testing.assert_close(score(5, 7), base, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(score(40, 42), base, rtol=1e-4, atol=1e-4)
    # 距离不同，点积就不同（这组向量实测约 -7.09 和 -5.99）
    assert abs(score(1, 4) - base) > 0.5


def test_rotary_has_no_length_limit_and_treats_heads_alike() -> None:
    from tlab.model.layers import RotaryEmbedding

    torch.manual_seed(0)
    rope = RotaryEmbedding(8)
    x = torch.randn(2, 3, 1024, 8)  # (B, h, T, d)

    y = rope(x)  # (B, h, T, d)

    # 1024 个位置也能算（可学习位置表做不到）；角度只看位置，不看序列总长
    assert y.shape == (2, 3, 1024, 8)
    torch.testing.assert_close(y[:, :, :8], rope(x[:, :, :8]))
    # 每条序列、每个头都用同一套角度
    torch.testing.assert_close(y[1, 2], rope(x[1:2, 2:3])[0, 0])


def test_sinusoidal_positions_match_hand_computed_values() -> None:
    from tlab.model.layers import sinusoidal_positions

    pe = sinusoidal_positions(3, 4, torch.device("cpu"))  # (T, C)

    # 原论文公式，C = 4：[sin(t), cos(t), sin(t/100), cos(t/100)]
    expected = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.841471, 0.540302, 0.0099998, 0.99995],
            [0.909297, -0.416147, 0.0199987, 0.9998],
        ]
    )  # (T, C)
    torch.testing.assert_close(pe, expected, rtol=1e-5, atol=1e-5)


def test_sinusoidal_positions_need_an_even_dimension() -> None:
    from tlab.model.layers import sinusoidal_positions

    # sin、cos 成对出现；奇数维时要给出看得懂的报错，不是形状对不上
    with pytest.raises(AssertionError, match="偶数"):
        sinusoidal_positions(3, 5, torch.device("cpu"))


def test_rope_offset_continues_the_positions_of_the_full_sequence() -> None:
    from tlab.model.layers import RotaryEmbedding

    torch.manual_seed(0)
    rope = RotaryEmbedding(8)
    x = torch.randn(2, 3, 7, 8)  # (B, h, T, d)

    full = rope(x)  # (B, h, T, d)

    # 只转后面 4 个 token，从第 3 个位置数起，结果必须和一次转完的后 4 个相同
    tail = rope(x[:, :, 3:], offset=3)  # (B, h, 4, d)
    torch.testing.assert_close(tail, full[:, :, 3:])
    # 不传 offset 就是从 0 数起，和以前一样
    assert torch.equal(rope(x, offset=0), full)


def test_sinusoidal_positions_offset_continues_the_full_sequence() -> None:
    from tlab.model.layers import sinusoidal_positions

    device = torch.device("cpu")
    full = sinusoidal_positions(7, 8, device)  # (T, C)

    tail = sinusoidal_positions(4, 8, device, offset=3)  # (4, C)

    torch.testing.assert_close(tail, full[3:])
