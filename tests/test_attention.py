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


def test_rope_rotates_q_and_k_but_not_v() -> None:
    from tlab.model.layers import RotaryEmbedding

    torch.manual_seed(0)
    B, T, C, h = 2, 5, 8, 2
    attn = CausalSelfAttention(C, h, rope_theta=10000.0)
    x = torch.randn(B, T, C)  # (B, T, C)

    # 参考答案：用同样的投影层拆头，只旋转 q、k，再交给 SDPA（只在测试里当参考）
    def split(y: torch.Tensor) -> torch.Tensor:
        return y.view(B, T, h, C // h).transpose(1, 2)  # (B, h, T, d)

    rope = RotaryEmbedding(C // h)
    q = rope(split(attn.q_proj(x)))  # (B, h, T, d)
    k = rope(split(attn.k_proj(x)))  # (B, h, T, d)
    v = split(attn.v_proj(x))  # (B, h, T, d)
    z = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, h, T, d)
    y_ref = attn.out_proj(z.transpose(1, 2).contiguous().view(B, T, C))  # (B, T, C)

    torch.testing.assert_close(attn(x), y_ref, rtol=1e-5, atol=1e-5)


def test_rope_changes_nothing_about_parameters_and_stays_causal() -> None:
    torch.manual_seed(0)
    attn = CausalSelfAttention(8, 2, rope_theta=10000.0)

    # RoPE 没有参数，也不进存档：参数名和不开 RoPE 时完全相同
    assert set(attn.state_dict()) == set(CausalSelfAttention(8, 2).state_dict())

    # 只改最后一个 token，前面位置的输出不变
    x = torch.randn(2, 5, 8)  # (B, T, C)
    x_changed = x.clone()  # (B, T, C)
    x_changed[:, 4, :] = torch.randn(2, 8)  # (B, C)
    y = attn(x)  # (B, T, C)
    y_changed = attn(x_changed)  # (B, T, C)
    torch.testing.assert_close(y_changed[:, :4, :], y[:, :4, :])
    assert not torch.allclose(y_changed[:, 4, :], y[:, 4, :])


def test_attention_can_tell_the_order_of_earlier_tokens_only_with_rope() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8)  # (B, T, C)
    swapped = x[:, [1, 0, 2, 3], :]  # (B, T, C)，交换第 0、1 个 token

    # 因果掩码只决定能看谁：不开 RoPE 时，最后一个位置分不出前面 token 的先后
    plain = CausalSelfAttention(8, 2)
    torch.testing.assert_close(plain(swapped)[:, -1], plain(x)[:, -1])

    # 开了 RoPE，同样的 token 换了位置，最后一个位置的输出就会变
    rope = CausalSelfAttention(8, 2, rope_theta=10000.0)
    assert not torch.allclose(rope(swapped)[:, -1], rope(x)[:, -1])


# ---- GQA：n_kv_heads 是 k、v 的组数，注释里写作 g ----


def gqa_reference(
    attn: CausalSelfAttention,
    x: torch.Tensor,
    h: int,
    g: int,
    order: list[int],
    rope: torch.nn.Module | None = None,
) -> torch.Tensor:
    """参考答案：用同样的投影层拆头，按 order 把每组分给各个头，再交给 SDPA（只在测试里用）。"""
    B, T, C = x.shape
    d = C // h
    q = attn.q_proj(x).view(B, T, h, d).transpose(1, 2)  # (B, h, T, d)
    k = attn.k_proj(x).view(B, T, g, d).transpose(1, 2)  # (B, g, T, d)
    v = attn.v_proj(x).view(B, T, g, d).transpose(1, 2)  # (B, g, T, d)
    if rope is not None:
        q = rope(q)  # (B, h, T, d)
        k = rope(k)  # (B, g, T, d)
    z = F.scaled_dot_product_attention(
        q, k[:, order], v[:, order], is_causal=True
    )  # (B, h, T, d)
    return attn.out_proj(z.transpose(1, 2).contiguous().view(B, T, C))  # (B, T, C)


# 第 i 个头用第几组：同组的头排在一起，和 HF 的 repeat_kv 相同
HEAD_ORDERS = [
    (1, [0, 0, 0, 0, 0, 0]),
    (2, [0, 0, 0, 1, 1, 1]),
    (3, [0, 0, 1, 1, 2, 2]),
]


def test_n_kv_heads_equal_to_n_heads_is_the_same_as_default_mha() -> None:
    torch.manual_seed(0)
    default = CausalSelfAttention(24, 6)
    explicit = CausalSelfAttention(24, 6, n_kv_heads=6)

    # 严格载入：参数名或形状对不上会报错，M1–M3 的旧存档靠这一点才载得进来
    explicit.load_state_dict(default.state_dict())

    x = torch.randn(2, 5, 24)  # (B, T, C)
    assert torch.equal(explicit(x), default(x))


@pytest.mark.parametrize(
    ("n_kv_heads", "g", "expected_params"),
    [
        (None, 6, 589_824),
        (6, 6, 589_824),
        (3, 3, 442_368),
        (2, 2, 393_216),
        (1, 1, 344_064),
    ],
)
def test_only_k_and_v_projections_shrink(
    n_kv_heads: int | None, g: int, expected_params: int
) -> None:
    # 手算（C = 384，d = 64）：q、out 各 C² = 147,456；k、v 各 C × g × d
    attn = CausalSelfAttention(384, 6, n_kv_heads=n_kv_heads)

    # nn.Linear 的 weight 形状是 (输出维度, 输入维度)
    assert attn.q_proj.weight.shape == (384, 384)
    assert attn.k_proj.weight.shape == (g * 64, 384)
    assert attn.v_proj.weight.shape == (g * 64, 384)
    assert attn.out_proj.weight.shape == (384, 384)
    assert sum(p.numel() for p in attn.parameters()) == expected_params


@pytest.mark.parametrize(("g", "order"), HEAD_ORDERS)
def test_gqa_matches_scaled_dot_product_attention(g: int, order: list[int]) -> None:
    torch.manual_seed(0)
    attn = CausalSelfAttention(24, 6, n_kv_heads=g)
    x = torch.randn(2, 5, 24)  # (B, T, C)

    y = attn(x)  # (B, T, C)

    assert y.shape == (2, 5, 24)
    y_ref = gqa_reference(attn, x, 6, g, order)  # (B, T, C)
    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("rope_theta", [None, 10000.0])
@pytest.mark.parametrize("g", [1, 2, 3])
def test_gqa_future_tokens_do_not_affect_earlier_outputs(
    g: int, rope_theta: float | None
) -> None:
    torch.manual_seed(0)
    attn = CausalSelfAttention(24, 6, rope_theta=rope_theta, n_kv_heads=g)
    x = torch.randn(2, 5, 24)  # (B, T, C)

    # 只改最后一个 token（下标 4），前四个 token 保持不变
    x_changed = x.clone()  # (B, T, C)
    x_changed[:, 4, :] = torch.randn(2, 24)  # (B, C)

    y = attn(x)  # (B, T, C)
    y_changed = attn(x_changed)  # (B, T, C)

    torch.testing.assert_close(y_changed[:, :4, :], y[:, :4, :])
    # 被改的位置输出应当变化，保证这次改动确实传进了模型
    assert not torch.allclose(y_changed[:, 4, :], y[:, 4, :])


@pytest.mark.parametrize("n_kv_heads", [4, 0, -1, 12])
def test_rejects_n_kv_heads_that_cannot_divide_n_heads(n_kv_heads: int) -> None:
    # 4：6 个头分不成 4 个等大的组；0、-1：组数没有意义；12：组数比头数还多
    with pytest.raises(AssertionError):
        CausalSelfAttention(24, 6, n_kv_heads=n_kv_heads)


@pytest.mark.parametrize(("g", "order"), HEAD_ORDERS)
def test_gqa_with_rope_rotates_q_and_k_but_not_v(g: int, order: list[int]) -> None:
    from tlab.model.layers import RotaryEmbedding

    torch.manual_seed(0)
    attn = CausalSelfAttention(24, 6, rope_theta=10000.0, n_kv_heads=g)
    x = torch.randn(2, 5, 24)  # (B, T, C)

    y_ref = gqa_reference(attn, x, 6, g, order, RotaryEmbedding(4))  # (B, T, C)

    torch.testing.assert_close(attn(x), y_ref, rtol=1e-5, atol=1e-5)


# ---- KV cache：测试默认用 C=24、h=6、d=4 ----


def test_kv_cache_appends_new_tokens_after_the_old_ones() -> None:
    from tlab.model.attention import KVCache

    torch.manual_seed(0)
    cache = KVCache()
    assert cache.length == 0

    k1, v1 = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 4, 8)  # (B, g, T, d)
    cache.update(k1, v1)
    assert cache.length == 4

    k2, v2 = torch.randn(2, 3, 1, 8), torch.randn(2, 3, 1, 8)  # (B, g, 1, d)
    k, v = cache.update(k2, v2)  # (B, g, 5, d)

    assert cache.length == 5
    assert k.shape == v.shape == (2, 3, 5, 8)
    # 旧的在前，新的接在后面
    assert torch.equal(k[:, :, :4], k1) and torch.equal(k[:, :, 4:], k2)
    assert torch.equal(v[:, :, :4], v1) and torch.equal(v[:, :, 4:], v2)


@pytest.mark.parametrize("g", [6, 2, 1])
def test_attention_stores_rotated_k_and_plain_v_before_copying(g: int) -> None:
    from tlab.model.attention import KVCache
    from tlab.model.layers import RotaryEmbedding

    torch.manual_seed(0)
    attn = CausalSelfAttention(24, 6, rope_theta=10000.0, n_kv_heads=g)
    x = torch.randn(2, 5, 24)  # (B, T, C)
    cache = KVCache()

    attn(x, cache=cache)

    # 存的是复制前的 g 组，这正是 GQA 能省缓存的原因
    assert cache.k.shape == cache.v.shape == (2, g, 5, 4)
    # k 存旋转后的，以后不用再转；v 本来就不旋转
    k_plain = attn.k_proj(x).view(2, 5, g, 4).transpose(1, 2)  # (B, g, T, d)
    v_plain = attn.v_proj(x).view(2, 5, g, 4).transpose(1, 2)  # (B, g, T, d)
    torch.testing.assert_close(cache.k, RotaryEmbedding(4)(k_plain))
    torch.testing.assert_close(cache.v, v_plain)


@pytest.mark.parametrize("chunks", [(6,), (3, 1, 2), (1, 1, 1, 1, 1, 1)])
@pytest.mark.parametrize("rope_theta", [None, 10000.0])
@pytest.mark.parametrize("g", [6, 2, 1])
def test_attention_in_chunks_matches_one_pass(
    g: int, rope_theta: float | None, chunks: tuple[int, ...]
) -> None:
    from tlab.model.attention import KVCache

    torch.manual_seed(0)
    attn = CausalSelfAttention(24, 6, rope_theta=rope_theta, n_kv_heads=g)
    x = torch.randn(2, 6, 24)  # (B, T, C)
    full = attn(x)  # (B, T, C)

    cache = KVCache()
    # 每段 (B, 这一段的长度, C)
    parts = [attn(part, cache=cache) for part in x.split(chunks, dim=1)]

    # 同一段里的 token 也不能偷看段内后面的 token，掩码写错这里会不一致
    torch.testing.assert_close(torch.cat(parts, dim=1), full, rtol=1e-5, atol=1e-5)
