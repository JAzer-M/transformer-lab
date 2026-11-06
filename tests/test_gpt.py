import math

import pytest
import torch
import torch.nn.functional as F

# 待实现模块在各测试内导入，与 test_block.py 保持一致


def small_config(**overrides: int | float | str):
    from tlab.model.gpt import GPTConfig

    # 测试用的小模型：词表 11、最长 8 个 token、C=16、2 层、2 个头
    fields = {
        "vocab_size": 11,
        "block_size": 8,
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 2,
    }
    fields.update(overrides)
    return GPTConfig(**fields)


def test_logits_shape_and_gradients_reach_embeddings_and_final_norm() -> None:
    from tlab.model.gpt import GPT

    torch.manual_seed(0)
    model = GPT(small_config())
    idx = torch.randint(0, 11, (2, 5))  # (B, T)

    logits = model(idx)  # (B, T, V)

    assert logits.shape == (2, 5, 11)

    # 两张查表和最终 LayerNorm 都参与了计算：反向传播后都应拿到不全为 0 的梯度
    # （只定义不使用的模块，参数量照样计入，只有梯度能发现它没用上）
    logits.sum().backward()
    for module in (model.tok_emb, model.pos_emb, model.ln_f):
        grad = module.weight.grad
        assert grad is not None
        assert grad.abs().sum() > 0


def test_lm_head_shares_weight_with_token_embedding() -> None:
    from tlab.model.gpt import GPT

    model = GPT(small_config())

    # weight tying：两处必须是同一个参数对象，而不是数值碰巧相等的两份拷贝
    assert model.lm_head.weight is model.tok_emb.weight


def test_count_parameters_matches_formula() -> None:
    from tlab.model.gpt import GPT, count_parameters

    V, block_size, C, L = 11, 8, 16, 2
    model = GPT(small_config())

    # 查表 V·C；位置表 block_size·C；每个 block 12C² + 4C；最终 LayerNorm 2C；
    # lm_head 与 tok_emb 共用权重，不再额外计数
    expected = V * C + block_size * C + L * (12 * C * C + 4 * C) + 2 * C

    assert count_parameters(model) == expected


def test_future_tokens_do_not_affect_earlier_logits() -> None:
    from tlab.model.gpt import GPT

    torch.manual_seed(0)
    model = GPT(small_config())
    idx = torch.randint(0, 11, (2, 6))  # (B, T)

    # 只改最后一个 token 的编号，前五个 token 保持不变
    idx_changed = idx.clone()  # (B, T)
    idx_changed[:, -1] = (idx[:, -1] + 1) % 11  # (B,)

    logits = model(idx)  # (B, T, V)
    logits_changed = model(idx_changed)  # (B, T, V)

    # 叠了多层 block 之后，前面的位置仍然看不到未来
    torch.testing.assert_close(logits_changed[:, :-1, :], logits[:, :-1, :])
    # 被改的位置输出应当变化，保证这次改动确实传进了模型
    assert not torch.allclose(logits_changed[:, -1, :], logits[:, -1, :])


def test_rejects_sequence_longer_than_block_size() -> None:
    from tlab.model.gpt import GPT

    model = GPT(small_config(block_size=8))
    idx = torch.zeros(1, 9, dtype=torch.long)  # (B, T=9)

    # 位置表只有 block_size 行，第 9 个位置没有对应的位置向量
    with pytest.raises(AssertionError):
        model(idx)


def test_initial_loss_is_close_to_log_vocab_size() -> None:
    from tlab.model.gpt import GPT

    torch.manual_seed(0)
    V = 65  # 字符级莎士比亚的词表大小
    model = GPT(small_config(vocab_size=V, block_size=16, d_model=32))
    idx = torch.randint(0, V, (4, 16))  # (B, T)
    targets = torch.randint(0, V, (4, 16))  # (B, T)

    logits = model(idx)  # (B, T, V)
    loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))  # ()

    # 初始化合理时 logits 都接近 0，softmax 近似均匀分布，交叉熵约为 ln(V)
    assert abs(loss.item() - math.log(V)) < 0.1


def test_rmsnorm_config_replaces_every_norm() -> None:
    from tlab.model.gpt import GPT, count_parameters
    from tlab.model.layers import RMSNorm

    V, block_size, C, L = 11, 8, 16, 2
    model = GPT(small_config(norm="rmsnorm"))

    # 每层两个、最后一个 ln_f，共 2L + 1 个归一化，必须全部换成 RMSNorm，一个都不漏
    norms = [m for m in model.modules() if isinstance(m, (RMSNorm, torch.nn.LayerNorm))]
    assert len(norms) == 2 * L + 1
    assert all(isinstance(m, RMSNorm) for m in norms)

    # 每个归一化少了 bias：每层 12C² + 2C，ln_f 只剩 C
    expected = V * C + block_size * C + L * (12 * C * C + 2 * C) + C
    assert count_parameters(model) == expected


def test_post_ln_config_normalizes_every_block_output() -> None:
    from tlab.model.gpt import GPT

    torch.manual_seed(0)
    model = GPT(small_config(norm_position="post"))
    idx = torch.randint(0, 11, (2, 5))  # (B, T)

    # Post-LN 的每层都以 LayerNorm 结尾（初始 bias = 0）：每个 token 的均值都是 0
    # Pre-LN 的 block 输出是 x + 改动量，均值不为 0，所以这条能看出开关有没有传到每一层
    x = model.tok_emb(idx) + model.pos_emb(torch.arange(5))  # (B, T, C)
    for block in model.blocks:
        x = block(x)  # (B, T, C)
        torch.testing.assert_close(x.mean(dim=-1), torch.zeros(2, 5), atol=1e-5, rtol=0)


def test_post_ln_config_has_no_final_norm() -> None:
    from tlab.model.gpt import GPT, count_parameters

    V, block_size, C, L = 11, 8, 16, 2
    torch.manual_seed(0)
    model = GPT(small_config(norm_position="post"))

    # 每层已经以归一化结尾，不再接最终 ln_f：比默认公式少 2C
    expected = V * C + block_size * C + L * (12 * C * C + 4 * C)
    assert count_parameters(model) == expected

    # 随机化所有 LayerNorm 的参数，最后一层的输出就不再是“已归一化”的，
    # 这时如果模型偷偷多做一次归一化，logits 一定对不上
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.LayerNorm):
                m.weight.normal_()
                m.bias.normal_()

    # 参考答案：最后一层 block 的输出直接进 lm_head
    idx = torch.randint(0, V, (2, 5))  # (B, T)
    x = model.tok_emb(idx) + model.pos_emb(torch.arange(5))  # (B, T, C)
    for block in model.blocks:
        x = block(x)  # (B, T, C)
    torch.testing.assert_close(model(idx), model.lm_head(x))


def test_gpt_config_rejects_unknown_norm_settings() -> None:
    # 建配置时就报错；即使 0 层、没有 Block 来检查，也不能被悄悄当成某种结构
    with pytest.raises(ValueError):
        small_config(norm="rms")
    with pytest.raises(ValueError):
        small_config(norm_position="middle", n_layers=0)


def test_gpt_config_rejects_unknown_mlp() -> None:
    # 与 norm 开关相同：建配置时就报错，0 层时也不例外
    with pytest.raises(ValueError):
        small_config(mlp="glu", n_layers=0)


def test_swiglu_config_replaces_every_mlp() -> None:
    from tlab.model.gpt import GPT, count_parameters
    from tlab.model.layers import SwiGLU

    V, block_size, C, L = 11, 8, 16, 2
    model = GPT(small_config(mlp="swiglu"))

    assert all(isinstance(block.mlp, SwiGLU) for block in model.blocks)
    # C = 16 不能被 3 整除：H = 8C//3 = 42，前馈 3·C·H = 2016（比 8C² = 2048 略少）；
    # 每层注意力 4C²、两个 LayerNorm 4C，再加查表、位置表和 ln_f 的 2C
    block_params = 4 * C * C + 3 * C * 42 + 4 * C
    assert count_parameters(model) == V * C + block_size * C + L * block_params + 2 * C


def test_swiglu_initial_loss_is_close_to_log_vocab_size() -> None:
    from tlab.model.gpt import GPT

    torch.manual_seed(0)
    V = 65
    model = GPT(small_config(vocab_size=V, block_size=16, d_model=32, mlp="swiglu"))
    idx = torch.randint(0, V, (4, 16))  # (B, T)
    targets = torch.randint(0, V, (4, 16))  # (B, T)

    logits = model(idx)  # (B, T, V)
    loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))  # ()

    # SwiGLU 的三个 Linear 也走 N(0, 0.02) 初始化，开局仍是均匀乱猜的水平
    assert abs(loss.item() - math.log(V)) < 0.1


def test_switches_can_be_combined_and_stay_causal() -> None:
    from tlab.model.gpt import GPT
    from tlab.model.layers import RMSNorm, SwiGLU

    torch.manual_seed(0)
    # 以后的组合实验 a9_modern 要同时打开多个开关；这里用 Post-LN 把所有分支都走到
    model = GPT(small_config(norm="rmsnorm", norm_position="post", mlp="swiglu"))
    block = model.blocks[0]
    assert isinstance(block.ln1, RMSNorm) and isinstance(block.mlp, SwiGLU)

    idx = torch.randint(0, 11, (2, 6))  # (B, T)
    idx_changed = idx.clone()  # (B, T)
    idx_changed[:, -1] = (idx[:, -1] + 1) % 11  # (B,)
    logits = model(idx)  # (B, T, V)
    logits_changed = model(idx_changed)  # (B, T, V)

    torch.testing.assert_close(logits_changed[:, :-1, :], logits[:, :-1, :])
    assert not torch.allclose(logits_changed[:, -1, :], logits[:, -1, :])


def run_blocks_and_head(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """参考答案的后半段：已经算好入口向量 x，走完所有 block、ln_f 和 lm_head。"""
    for block in model.blocks:
        x = block(x)  # (B, T, C)
    return model.lm_head(model.ln_f(x))  # (B, T, V)


def test_default_model_keeps_its_top_level_parameter_names() -> None:
    from tlab.model.gpt import GPT

    # M1、M2 的存档按这些名字载入；默认仍是可学习的位置表
    names = {
        name
        for name in GPT(small_config()).state_dict()
        if not name.startswith("blocks.")
    }
    assert names == {
        "tok_emb.weight",
        "pos_emb.weight",
        "ln_f.weight",
        "ln_f.bias",
        "lm_head.weight",
    }


def test_sinusoidal_scales_tokens_and_adds_fixed_positions() -> None:
    from tlab.model.gpt import GPT, count_parameters
    from tlab.model.layers import sinusoidal_positions

    V, C, L = 11, 16, 2
    torch.manual_seed(0)
    model = GPT(small_config(pos_encoding="sinusoidal"))

    # 没有位置表：比默认公式少 block_size·C
    assert not hasattr(model, "pos_emb")
    assert count_parameters(model) == V * C + L * (12 * C * C + 4 * C) + 2 * C

    # 入口向量 = token 向量 × √C + 正弦位置（原论文做法）
    idx = torch.randint(0, V, (2, 5))  # (B, T)
    pe = sinusoidal_positions(5, C, idx.device)  # (T, C)
    x = model.tok_emb(idx) * C**0.5 + pe  # (B, T, C)
    torch.testing.assert_close(model(idx), run_blocks_and_head(model, x))


def test_rope_config_adds_nothing_at_the_entrance_and_rotates_in_every_layer() -> None:
    from tlab.model.gpt import GPT, count_parameters
    from tlab.model.layers import RotaryEmbedding

    V, C, L = 11, 16, 2
    torch.manual_seed(0)
    model = GPT(small_config(pos_encoding="rope", rope_theta=100.0))

    assert not hasattr(model, "pos_emb")
    assert count_parameters(model) == V * C + L * (12 * C * C + 4 * C) + 2 * C
    # 每层注意力都旋转；d = 8、theta = 100 时第 1 对的转速是 1 / 100^(2/8) ≈ 0.316228
    for block in model.blocks:
        assert isinstance(block.attn.rope, RotaryEmbedding)
        assert block.attn.rope.inv_freq[1].item() == pytest.approx(0.316228, rel=1e-5)

    # 入口只有 token 向量，不缩放、不加位置
    idx = torch.randint(0, V, (2, 5))  # (B, T)
    torch.testing.assert_close(
        model(idx), run_blocks_and_head(model, model.tok_emb(idx))
    )


@pytest.mark.parametrize("pos_encoding", ["sinusoidal", "rope"])
def test_fixed_position_encodings_accept_longer_sequences_and_stay_causal(
    pos_encoding: str,
) -> None:
    from tlab.model.gpt import GPT

    torch.manual_seed(0)
    model = GPT(small_config(block_size=8, pos_encoding=pos_encoding))
    # 比 block_size 长一倍：这是 A1 外推评测要用的
    idx = torch.randint(0, 11, (2, 16))  # (B, T)
    idx_changed = idx.clone()  # (B, T)
    idx_changed[:, -1] = (idx[:, -1] + 1) % 11  # (B,)

    logits = model(idx)  # (B, T, V)
    logits_changed = model(idx_changed)  # (B, T, V)

    assert logits.shape == (2, 16, 11)
    torch.testing.assert_close(logits_changed[:, :-1, :], logits[:, :-1, :])
    assert not torch.allclose(logits_changed[:, -1, :], logits[:, -1, :])


@pytest.mark.parametrize("pos_encoding", ["sinusoidal", "rope"])
def test_fixed_position_encodings_start_near_log_vocab_size(pos_encoding: str) -> None:
    from tlab.model.gpt import GPT

    torch.manual_seed(0)
    V = 65
    model = GPT(
        small_config(vocab_size=V, block_size=16, d_model=32, pos_encoding=pos_encoding)
    )
    idx = torch.randint(0, V, (4, 16))  # (B, T)
    targets = torch.randint(0, V, (4, 16))  # (B, T)

    logits = model(idx)  # (B, T, V)
    loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))  # ()

    assert abs(loss.item() - math.log(V)) < 0.1


def test_rope_rejects_an_odd_head_dim_and_unknown_pos_encoding() -> None:
    from tlab.model.gpt import GPT

    # C = 6、2 个头：每头 3 维，没法两两配对
    with pytest.raises(AssertionError):
        GPT(small_config(d_model=6, n_heads=2, pos_encoding="rope"))
    # 拼错时建配置就报错
    with pytest.raises(ValueError):
        small_config(pos_encoding="abs")
    # theta 是转速公式的底数，必须大于 0，否则角度不是有限数
    with pytest.raises(ValueError):
        small_config(pos_encoding="rope", rope_theta=0.0)


@pytest.mark.parametrize("pos_encoding", ["learned", "sinusoidal"])
def test_only_the_rope_config_rotates_in_attention(pos_encoding: str) -> None:
    from tlab.model.gpt import GPT

    # RoPE 没有参数，开错了参数量也看不出来，所以要直接检查：其余两种不能旋转
    model = GPT(small_config(pos_encoding=pos_encoding))
    assert all(block.attn.rope is None for block in model.blocks)


@pytest.mark.parametrize("n_kv_heads", [3, 0, -1, 4])
def test_gpt_config_rejects_n_kv_heads_that_cannot_divide_n_heads(
    n_kv_heads: int,
) -> None:
    # 建配置时就报错，0 层的模型也不会被放过
    with pytest.raises(ValueError, match="n_kv_heads"):
        small_config(n_kv_heads=n_kv_heads, n_layers=0)


def test_n_kv_heads_config_reaches_every_layer() -> None:
    from tlab.model.gpt import GPT

    model = GPT(small_config(d_model=24, n_heads=6, n_kv_heads=2))

    # k_proj 的 weight 形状是 (g * d, C)：2 组，每组 4 维
    for block in model.blocks:
        assert block.attn.k_proj.weight.shape == (8, 24)
        assert block.attn.v_proj.weight.shape == (8, 24)


@pytest.mark.parametrize("n_kv_heads", [None, 2, 1])
@pytest.mark.parametrize("pos_encoding", ["learned", "sinusoidal", "rope"])
@pytest.mark.parametrize("norm_position", ["pre", "post"])
def test_model_with_cache_matches_one_pass_token_by_token(
    norm_position: str, pos_encoding: str, n_kv_heads: int | None
) -> None:
    from tlab.model.gpt import GPT

    torch.manual_seed(0)
    model = GPT(
        small_config(
            d_model=24,
            n_heads=6,
            norm_position=norm_position,
            pos_encoding=pos_encoding,
            n_kv_heads=n_kv_heads,
        )
    )
    idx = torch.randint(0, 11, (2, 7))  # (B, T)
    full = model(idx)  # (B, T, V)

    cache = model.new_cache()
    # 先送开头 4 个，再一个一个送，每一步的分数都要和一次算完的对应位置相同
    prompt = model(idx[:, :4], cache=cache)  # (B, 4, V)
    torch.testing.assert_close(prompt, full[:, :4], rtol=1e-5, atol=1e-5)
    for t in range(4, 7):
        step = model(idx[:, t : t + 1], cache=cache)  # (B, 1, V)
        torch.testing.assert_close(step[:, 0], full[:, t], rtol=1e-5, atol=1e-5)


def test_learned_positions_reject_a_cache_longer_than_block_size() -> None:
    from tlab.model.gpt import GPT

    torch.manual_seed(0)
    model = GPT(small_config())  # block_size = 8，位置表只有 8 行
    cache = model.new_cache()
    model(torch.zeros(1, 8, dtype=torch.long), cache=cache)

    # 新 token 是第 9 个，位置表里没有，要明确报错，而不是查表越界
    with pytest.raises(AssertionError, match="block_size"):
        model(torch.zeros(1, 1, dtype=torch.long), cache=cache)


@pytest.mark.parametrize("pos_encoding", ["learned", "sinusoidal", "rope"])
def test_empty_cache_gives_the_same_output_as_no_cache(pos_encoding: str) -> None:
    from tlab.model.gpt import GPT

    torch.manual_seed(0)
    model = GPT(small_config(pos_encoding=pos_encoding))
    idx = torch.randint(0, 11, (2, 5))  # (B, T)

    # 第一次送进去时缓存是空的，计算过程应该和不用缓存完全一样
    assert torch.equal(model(idx, cache=model.new_cache()), model(idx))
    # 缓存不是模型的一部分：不进存档，建了也不改变参数名
    names = set(model.state_dict())
    model(idx, cache=model.new_cache())
    assert set(model.state_dict()) == names
