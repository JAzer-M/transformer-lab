import torch

# 待实现模块在各测试内导入：先写好 MLP 时，MLP 的测试就能单独通过


def test_mlp_preserves_shape_and_expands_hidden_to_4x() -> None:
    from tlab.model.block import MLP

    mlp = MLP(d_model=8)
    x = torch.randn(2, 5, 8)  # (B, T, C)

    y = mlp(x)  # (B, T, C)

    assert y.shape == (2, 5, 8)
    # Linear 的 weight 形状是 (out_features, in_features)
    assert mlp.fc.weight.shape == (32, 8)  # C -> 4C
    assert mlp.proj.weight.shape == (8, 32)  # 4C -> C


def test_block_preserves_shape() -> None:
    from tlab.model.block import Block

    block = Block(d_model=8, n_heads=2)
    x = torch.randn(2, 5, 8)  # (B, T, C)

    y = block(x)  # (B, T, C)

    assert y.shape == (2, 5, 8)


def test_block_matches_pre_ln_formula() -> None:
    from tlab.model.block import Block

    torch.manual_seed(0)
    block = Block(d_model=8, n_heads=2)
    x = torch.randn(2, 5, 8)  # (B, T, C)

    # 两个 LayerNorm 初始化时完全相同；随机化它们的参数，才能分辨 ln1 和 ln2 是否用对位置
    with torch.no_grad():
        for ln in (block.ln1, block.ln2):
            ln.weight.normal_()
            ln.bias.normal_()

    # 参考答案：用 block 自己的子模块，按 Pre-LN 公式手动算一遍
    x1 = x + block.attn(block.ln1(x))  # (B, T, C)
    y_ref = x1 + block.mlp(block.ln2(x1))  # (B, T, C)

    y = block(x)  # (B, T, C)

    torch.testing.assert_close(y, y_ref)


def test_block_is_identity_when_branch_outputs_are_zero() -> None:
    from tlab.model.block import Block

    torch.manual_seed(0)
    block = Block(d_model=8, n_heads=2)
    x = torch.randn(2, 5, 8)  # (B, T, C)

    # 把注意力和 MLP 的最后一层置 0，两条分支都输出 0，只剩残差这条捷径
    with torch.no_grad():
        block.attn.out_proj.weight.zero_()
        block.mlp.proj.weight.zero_()

    y = block(x)  # (B, T, C)

    torch.testing.assert_close(y, x)


def test_block_future_tokens_do_not_affect_earlier_outputs() -> None:
    from tlab.model.block import Block

    torch.manual_seed(0)
    block = Block(d_model=8, n_heads=2)
    x = torch.randn(2, 5, 8)  # (B, T, C)

    # 只改最后一个 token（下标 4），前四个 token 保持不变
    x_changed = x.clone()  # (B, T, C)
    x_changed[:, 4, :] = torch.randn(2, 8)  # (B, C)

    y = block(x)  # (B, T, C)
    y_changed = block(x_changed)  # (B, T, C)

    # LayerNorm 和 MLP 都只看单个 token，因果性只取决于注意力，套上 block 后仍应成立
    torch.testing.assert_close(y_changed[:, :4, :], y[:, :4, :])
    # 被改的位置输出应当变化，保证这次改动确实传进了模型
    assert not torch.allclose(y_changed[:, 4, :], y[:, 4, :])


def test_block_parameter_count() -> None:
    from tlab.model.block import Block

    C = 8
    block = Block(d_model=C, n_heads=2)

    n_params = sum(p.numel() for p in block.parameters())

    # 注意力 4 个 (C, C) = 4C²；MLP 两个 (C, 4C) = 8C²；两个 LayerNorm 各有 weight、bias = 4C
    assert n_params == 12 * C * C + 4 * C
