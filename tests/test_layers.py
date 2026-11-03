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
    expected = torch.tensor([[[0.4, 0.8, 0.8, 1.6], [1.2, 0.0, 0.0, 1.6]]])
    torch.testing.assert_close(y, expected)

    # 再逐分量乘以 weight = [1, 2, 3, 4]
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    y = norm(x)  # (1, 2, 4)
    expected = torch.tensor([[[0.4, 1.6, 2.4, 6.4], [1.2, 0.0, 0.0, 6.4]]])
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
