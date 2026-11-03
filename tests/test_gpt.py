import math

import pytest
import torch
import torch.nn.functional as F

# 待实现模块在各测试内导入，与 test_block.py 保持一致


def small_config(**overrides: int | str):
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
