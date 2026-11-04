import pytest
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


def test_block_with_rmsnorm_uses_it_for_both_norms() -> None:
    from tlab.model.block import Block
    from tlab.model.layers import RMSNorm

    C = 8
    block = Block(d_model=C, n_heads=2, norm="rmsnorm")

    assert isinstance(block.ln1, RMSNorm)
    assert isinstance(block.ln2, RMSNorm)
    # RMSNorm 只有 weight：两个 norm 共 2C，比 LayerNorm 少 2C（C=8 时 784）
    n_params = sum(p.numel() for p in block.parameters())
    assert n_params == 12 * C * C + 2 * C == 784


def test_post_ln_block_matches_post_ln_formula() -> None:
    from tlab.model.block import Block

    torch.manual_seed(0)
    block = Block(d_model=8, n_heads=2, norm_position="post")
    x = torch.randn(2, 5, 8)  # (B, T, C)

    # 随机化两个 LayerNorm 的参数，才能分辨 ln1、ln2 是否用对位置
    with torch.no_grad():
        for ln in (block.ln1, block.ln2):
            ln.weight.normal_()
            ln.bias.normal_()

    # 参考答案：Post-LN 先把子层输出加回 x，再归一化（原论文的 Add & Norm）
    x1 = block.ln1(x + block.attn(x))  # (B, T, C)
    y_ref = block.ln2(x1 + block.mlp(x1))  # (B, T, C)

    y = block(x)  # (B, T, C)

    torch.testing.assert_close(y, y_ref)


def test_post_ln_block_is_still_causal() -> None:
    from tlab.model.block import Block

    torch.manual_seed(0)
    block = Block(d_model=8, n_heads=2, norm_position="post")
    x = torch.randn(2, 5, 8)  # (B, T, C)

    # 只改最后一个 token（下标 4）
    x_changed = x.clone()  # (B, T, C)
    x_changed[:, 4, :] = torch.randn(2, 8)  # (B, C)

    y = block(x)  # (B, T, C)
    y_changed = block(x_changed)  # (B, T, C)

    # 归一化挪到加法之后，仍然只在单个 token 内部计算，不会把未来泄露给前面
    torch.testing.assert_close(y_changed[:, :4, :], y[:, :4, :])
    assert not torch.allclose(y_changed[:, 4, :], y[:, 4, :])


def test_block_rejects_unknown_norm_settings() -> None:
    from tlab.model.block import Block

    # 拼错时要直接报错；否则 forward 里的分支会把它悄悄当成某一种结构
    with pytest.raises(ValueError):
        Block(d_model=8, n_heads=2, norm="rms")
    with pytest.raises(ValueError):
        Block(d_model=8, n_heads=2, norm_position="middle")


def test_block_with_swiglu_has_the_same_parameter_count_as_gelu() -> None:
    from tlab.model.block import Block
    from tlab.model.layers import SwiGLU

    torch.manual_seed(0)
    # C = 6 能被 3 整除：隐藏维度 H = 8C/3 = 16，SwiGLU 3·C·H = 288，等于 GELU MLP 的 8C²
    block = Block(d_model=6, n_heads=2, mlp="swiglu")

    assert isinstance(block.mlp, SwiGLU)
    assert block.mlp.down_proj.weight.shape == (6, 16)
    # 注意力 4C² + 前馈 8C² + 两个 LayerNorm 4C = 144 + 288 + 24 = 456，与 GELU 版相同
    assert sum(p.numel() for p in block.parameters()) == 456
    gelu_block = Block(d_model=6, n_heads=2)
    assert sum(p.numel() for p in gelu_block.parameters()) == 456

    # 只换前馈子层，Pre-LN 的结构不变：参考答案用 block 自己的子模块按公式算一遍
    x = torch.randn(2, 5, 6)  # (B, T, C)
    x1 = x + block.attn(block.ln1(x))  # (B, T, C)
    y_ref = x1 + block.mlp(block.ln2(x1))  # (B, T, C)
    torch.testing.assert_close(block(x), y_ref)


def test_block_rejects_unknown_mlp() -> None:
    from tlab.model.block import Block

    # 拼错时要直接报错，不能被悄悄当成 GELU MLP
    with pytest.raises(ValueError):
        Block(d_model=6, n_heads=2, mlp="glu")


def test_make_mlp_rejects_unknown_kind() -> None:
    from tlab.model.block import make_mlp

    # 直接调用时也要报错，不能悄悄退回 GELU MLP（与 make_norm 的行为一致）
    with pytest.raises(ValueError):
        make_mlp("glu", 6)


def test_default_block_parameter_names_are_unchanged() -> None:
    from tlab.model.block import Block

    # M1、M2 的存档按这些名字载入；这份名单写死，改名或挪动结构时测试会报错
    assert set(Block(d_model=6, n_heads=2).state_dict()) == {
        "ln1.weight",
        "ln1.bias",
        "attn.q_proj.weight",
        "attn.k_proj.weight",
        "attn.v_proj.weight",
        "attn.out_proj.weight",
        "ln2.weight",
        "ln2.bias",
        "mlp.fc.weight",
        "mlp.proj.weight",
    }
