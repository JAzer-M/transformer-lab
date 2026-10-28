import pytest
import torch
import torch.nn.functional as F

from tlab.model.attention import CausalSelfAttention


def test_single_head_preserves_input_shape() -> None:
    attention = CausalSelfAttention(d_model=4, n_heads=1)
    x = torch.randn(2, 3, 4)  # (B, T, C)

    y = attention(x)  # (B, T, C)

    assert y.shape == (2, 3, 4)


def test_future_tokens_do_not_affect_earlier_outputs() -> None:
    torch.manual_seed(0)
    attention = CausalSelfAttention(d_model=4, n_heads=1)
    x = torch.randn(2, 3, 4)  # (B, T, C)

    # 只改最后一个 token（下标 2），前两个 token 保持不变
    x_changed = x.clone()  # (B, T, C)
    x_changed[:, 2, :] = torch.randn(2, 4)  # (B, C)

    y = attention(x)  # (B, T, C)
    y_changed = attention(x_changed)  # (B, T, C)

    # 前两个位置看不到未来，输出必须不变
    torch.testing.assert_close(y_changed[:, :2, :], y[:, :2, :])
    # 被改的位置输出应当变化，否则说明这次改动根本没传进模型，上一条断言就没有意义
    assert not torch.allclose(y_changed[:, 2, :], y[:, 2, :])


def test_matches_scaled_dot_product_attention() -> None:
    torch.manual_seed(0)
    attention = CausalSelfAttention(d_model=4, n_heads=1)
    x = torch.randn(2, 5, 4)  # (B, T, C)

    # 参考答案：复用同一组投影层，中间的注意力计算交给 PyTorch 官方实现
    q = attention.q_proj(x)  # (B, T, C)
    k = attention.k_proj(x)  # (B, T, C)
    v = attention.v_proj(x)  # (B, T, C)
    z_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, T, C)
    y_ref = attention.out_proj(z_ref)  # (B, T, C)

    y = attention(x)  # (B, T, C)

    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-5)


def test_gradients_reach_all_projections() -> None:
    torch.manual_seed(0)
    attention = CausalSelfAttention(d_model=4, n_heads=1)
    x = torch.randn(2, 3, 4)  # (B, T, C)

    # 用输出之和当作一个最简单的 loss，反向传播
    loss = attention(x).sum()  # ()
    loss.backward()

    # 四个投影层都参与了计算，都应拿到有限且不全为 0 的梯度
    for layer in (
        attention.q_proj,
        attention.k_proj,
        attention.v_proj,
        attention.out_proj,
    ):
        grad = layer.weight.grad  # (C, C)
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


def test_single_token_sequence() -> None:
    torch.manual_seed(0)
    attention = CausalSelfAttention(d_model=4, n_heads=1)
    x = torch.randn(2, 1, 4)  # (B, T=1, C)

    y = attention(x)  # (B, 1, C)

    assert y.shape == (2, 1, 4)
    assert torch.isfinite(y).all()
    # 只有一个 token 时注意力权重恒为 1，输出等于 out_proj(v)
    expected = attention.out_proj(attention.v_proj(x))  # (B, 1, C)
    torch.testing.assert_close(y, expected)


def test_multi_head_preserves_input_shape() -> None:
    attention = CausalSelfAttention(d_model=8, n_heads=2)
    x = torch.randn(2, 5, 8)  # (B, T, C)

    y = attention(x)  # (B, T, C)

    assert y.shape == (2, 5, 8)


def test_multi_head_matches_scaled_dot_product_attention() -> None:
    torch.manual_seed(0)
    B, T, C, n_heads = 2, 5, 8, 2
    head_dim = C // n_heads
    attention = CausalSelfAttention(d_model=C, n_heads=n_heads)
    x = torch.randn(B, T, C)  # (B, T, C)

    # 参考答案：复用同一组投影层
    q = attention.q_proj(x)  # (B, T, C)
    k = attention.k_proj(x)  # (B, T, C)
    v = attention.v_proj(x)  # (B, T, C)

    # 在测试里自己拆头
    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)  # (B, h, T, d)
    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)  # (B, h, T, d)
    v = v.view(B, T, n_heads, head_dim).transpose(1, 2)  # (B, h, T, d)

    # 各头的注意力交给 PyTorch 官方实现，它默认按 head_dim 缩放
    z_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, h, T, d)

    # 合头后再过输出投影
    z_ref = z_ref.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
    y_ref = attention.out_proj(z_ref)  # (B, T, C)

    y = attention(x)  # (B, T, C)

    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-5)


def test_multi_head_future_tokens_do_not_affect_earlier_outputs() -> None:
    torch.manual_seed(0)
    attention = CausalSelfAttention(d_model=8, n_heads=2)
    x = torch.randn(2, 5, 8)  # (B, T, C)

    # 只改最后一个 token（下标 4），前四个 token 保持不变
    x_changed = x.clone()  # (B, T, C)
    x_changed[:, 4, :] = torch.randn(2, 8)  # (B, C)

    y = attention(x)  # (B, T, C)
    y_changed = attention(x_changed)  # (B, T, C)

    # 前四个位置在每个头里都看不到未来，输出必须不变
    torch.testing.assert_close(y_changed[:, :4, :], y[:, :4, :])
    # 被改的位置输出应当变化，保证这次改动确实传进了模型
    assert not torch.allclose(y_changed[:, 4, :], y[:, 4, :])


def test_rejects_d_model_not_divisible_by_n_heads() -> None:
    # 6 // 4 = 1，拆头时 4 × 1 ≠ 6，应在构造时就报错
    with pytest.raises(AssertionError):
        CausalSelfAttention(d_model=6, n_heads=4)
