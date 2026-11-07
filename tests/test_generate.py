import pytest
import torch
from torch import nn

# 待实现模块在各测试内导入，与其他测试文件保持一致


def small_model() -> nn.Module:
    from tlab.model.gpt import GPT, GPTConfig

    torch.manual_seed(0)
    return GPT(
        GPTConfig(vocab_size=11, block_size=8, d_model=16, n_layers=1, n_heads=2)
    )


class FixedLogits(nn.Module):
    """无论输入什么，每个位置都输出同一组分数：3 号最高，7 号次之。"""

    def __init__(self) -> None:
        super().__init__()
        self.block_size = 8
        scores = torch.zeros(11)
        scores[3], scores[7] = 5.0, 4.0
        self.register_buffer("scores", scores)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        return self.scores.expand(B, T, -1)  # (B, T, V)


def test_output_appends_new_tokens_after_prompt() -> None:
    from tlab.eval.generate import generate

    model = small_model()
    prompt = torch.randint(0, 11, (2, 3))  # (B, T)

    out = generate(model, prompt, max_new_tokens=5)  # (B, T + 5)

    assert out.shape == (2, 8)
    assert out.dtype == torch.long
    # 开头原样保留，新 token 都在词表范围内
    assert torch.equal(out[:, :3], prompt)
    assert int(out.min()) >= 0 and int(out.max()) < 11


def test_top_k_one_equals_greedy_argmax() -> None:
    from tlab.eval.generate import generate

    model = small_model()
    prompt = torch.randint(0, 11, (2, 3))  # (B, T)

    out = generate(model, prompt, max_new_tokens=4, top_k=1)

    # 参考答案：每一步手动取最后位置分数最大的 token
    expected = prompt
    with torch.no_grad():
        for _ in range(4):
            next_id = model(expected)[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
            expected = torch.cat([expected, next_id], dim=1)
    assert torch.equal(out, expected)


def test_prompt_longer_than_block_size_is_cropped() -> None:
    from tlab.eval.generate import generate

    model = small_model()  # block_size = 8
    prompt = torch.randint(0, 11, (1, 10))  # (B, T=10)，已经超过 block_size

    # 只把最后 block_size 个 token 送进模型，不应触发位置表越界的断言
    out = generate(model, prompt, max_new_tokens=5)

    assert out.shape == (1, 15)


def test_top_k_restricts_sampling_to_highest_scores() -> None:
    from tlab.eval.generate import generate

    model = FixedLogits()
    prompt = torch.zeros(4, 1, dtype=torch.long)  # (B, T)
    generator = torch.Generator().manual_seed(0)

    out = generate(model, prompt, max_new_tokens=50, top_k=2, generator=generator)

    # 只允许分数最高的两个 token：3 和 7；温度为 1 时两者都应该出现
    new_tokens = set(out[:, 1:].flatten().tolist())
    assert new_tokens == {3, 7}


def test_low_temperature_concentrates_on_the_best_token() -> None:
    from tlab.eval.generate import generate

    model = FixedLogits()
    prompt = torch.zeros(4, 1, dtype=torch.long)  # (B, T)
    generator = torch.Generator().manual_seed(0)

    out = generate(
        model, prompt, max_new_tokens=50, temperature=0.05, generator=generator
    )

    # 分数差 1 除以 0.05 变成差 20，几乎总是选 3 号
    assert set(out[:, 1:].flatten().tolist()) == {3}


def test_same_seed_gives_same_text() -> None:
    from tlab.eval.generate import generate

    model = small_model()
    prompt = torch.zeros(1, 1, dtype=torch.long)  # (B, T)

    out1 = generate(model, prompt, 20, generator=torch.Generator().manual_seed(1))
    out2 = generate(model, prompt, 20, generator=torch.Generator().manual_seed(1))

    assert torch.equal(out1, out2)


def test_rejects_non_positive_temperature_and_restores_train_mode() -> None:
    from tlab.eval.generate import generate

    model = small_model()
    model.train()
    prompt = torch.zeros(1, 1, dtype=torch.long)  # (B, T)

    with pytest.raises(AssertionError):
        generate(model, prompt, 3, temperature=0.0)

    generate(model, prompt, 3)
    # 生成期间切到 eval 模式，结束后恢复原来的模式
    assert model.training


# ---- KV cache ----


def cache_model(
    pos_encoding: str = "learned", n_kv_heads: int | None = None
) -> nn.Module:
    from tlab.model.gpt import GPT, GPTConfig

    torch.manual_seed(0)
    return GPT(
        GPTConfig(
            vocab_size=11,
            block_size=8,
            d_model=24,
            n_layers=2,
            n_heads=6,
            pos_encoding=pos_encoding,
            n_kv_heads=n_kv_heads,
        )
    )


@pytest.mark.parametrize("n_kv_heads", [None, 2, 1])
@pytest.mark.parametrize("pos_encoding", ["learned", "sinusoidal", "rope"])
def test_cache_gives_the_same_text_as_no_cache(
    pos_encoding: str, n_kv_heads: int | None
) -> None:
    from tlab.eval.generate import generate

    model = cache_model(pos_encoding, n_kv_heads)
    prompt = torch.randint(0, 11, (2, 3))  # (B, T)

    # 贪心：每一步都取分数最高的
    plain = generate(model, prompt, 5, top_k=1)  # (B, T + 5)
    cached = generate(model, prompt, 5, top_k=1, use_cache=True)  # (B, T + 5)
    assert torch.equal(cached, plain)

    # 抽样：同一个种子，抽到的也要一样
    plain = generate(
        model, prompt, 5, generator=torch.Generator().manual_seed(1)
    )  # (B, T + 5)
    cached = generate(
        model, prompt, 5, generator=torch.Generator().manual_seed(1), use_cache=True
    )  # (B, T + 5)
    assert torch.equal(cached, plain)


@pytest.mark.parametrize("prompt_len", [3, 10])
@pytest.mark.parametrize("pos_encoding", ["learned", "sinusoidal", "rope"])
def test_cache_gives_the_same_text_past_block_size(
    pos_encoding: str, prompt_len: int
) -> None:
    from tlab.eval.generate import generate

    model = cache_model(pos_encoding, 2)  # block_size = 8
    prompt = torch.randint(0, 11, (2, prompt_len))  # (B, T)

    plain = generate(model, prompt, 12, top_k=1)  # (B, T + 12)
    cached = generate(model, prompt, 12, top_k=1, use_cache=True)  # (B, T + 12)

    assert torch.equal(cached, plain)


def lengths_sent_to_the_model(model: nn.Module) -> list[int]:
    """记录每一步送进模型的 token 数，用来确认缓存真的用上了。"""
    lengths: list[int] = []
    model.register_forward_pre_hook(lambda _, args: lengths.append(args[0].size(1)))
    return lengths


def test_cache_sends_only_the_new_token_after_the_prompt() -> None:
    from tlab.eval.generate import generate

    model = cache_model()
    lengths = lengths_sent_to_the_model(model)

    generate(model, torch.zeros(1, 3, dtype=torch.long), 5, top_k=1, use_cache=True)

    # 不检查这个的话，什么都不缓存的假实现也能让“结果相同”通过
    assert lengths == [3, 1, 1, 1, 1]


def test_cache_rebuilds_from_the_window_once_it_is_full() -> None:
    from tlab.eval.generate import generate

    model = cache_model()  # block_size = 8
    lengths = lengths_sent_to_the_model(model)

    generate(model, torch.zeros(1, 3, dtype=torch.long), 8, top_k=1, use_cache=True)

    # 前 6 步把缓存从 3 个填到 8 个；之后每一步都要丢掉缓存，用最后 8 个重建
    assert lengths == [3, 1, 1, 1, 1, 1, 8, 8]


def test_cache_starts_fresh_each_call_and_restores_train_mode() -> None:
    from tlab.eval.generate import generate

    model = cache_model("rope")
    model.train()
    prompt = torch.randint(0, 11, (2, 3))  # (B, T)

    first = generate(model, prompt, 4, top_k=1, use_cache=True)  # (B, T + 4)
    second = generate(model, prompt, 4, top_k=1, use_cache=True)  # (B, T + 4)

    # 连续生成两次互不影响：缓存不留在模型里
    assert torch.equal(first, second)
    assert model.training


class NextIsLength(nn.Module):
    """下一个 token 的编号等于当前长度：长度 2 时生成 2，长度 3 时生成 3，以此类推。"""

    block_size = 64

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        logits = torch.zeros(B, T, 64)  # (B, T, V)
        logits[:, -1, T] = 1.0
        return logits


def test_eos_stops_right_after_it_is_generated() -> None:
    from tlab.eval.generate import generate

    idx = torch.tensor([[9, 9]])  # (1, 2)

    out = generate(NextIsLength(), idx, 6, top_k=1, eos_token_id=4)  # (1, 5)

    # eos 留在结果里，由调用方决定怎么处理
    assert out.tolist() == [[9, 9, 2, 3, 4]]


def test_eos_that_never_comes_gives_the_full_length() -> None:
    from tlab.eval.generate import generate

    idx = torch.tensor([[9, 9]])  # (1, 2)

    out = generate(NextIsLength(), idx, 3, top_k=1, eos_token_id=50)  # (1, 5)

    assert out.tolist() == [[9, 9, 2, 3, 4]]


def test_without_eos_nothing_stops_early() -> None:
    from tlab.eval.generate import generate

    idx = torch.tensor([[9, 9]])  # (1, 2)

    out = generate(NextIsLength(), idx, 6, top_k=1)  # (1, 8)

    assert out.tolist() == [[9, 9, 2, 3, 4, 5, 6, 7]]


def test_eos_none_samples_the_same_tokens_as_not_passing_it() -> None:
    from tlab.eval.generate import generate

    idx = torch.tensor([[1, 2]])  # (1, 2)
    default = generate(  # (1, 14)
        small_model(), idx, 12, generator=torch.Generator().manual_seed(5)
    )
    explicit = generate(  # (1, 14)
        small_model(),
        idx,
        12,
        generator=torch.Generator().manual_seed(5),
        eos_token_id=None,
    )

    assert torch.equal(default, explicit)
    assert default.shape == (1, 14)


@pytest.mark.parametrize("use_cache", [False, True])
def test_eos_stops_at_the_same_token_with_and_without_cache(use_cache: bool) -> None:
    from tlab.eval.generate import generate

    model = cache_model("rope")
    idx = torch.tensor([[1, 2, 3]])  # (1, 3)
    # 随机小模型贪心时总是同一个 token，所以这里用固定种子抽样，生成的 token 才有变化
    full = generate(model, idx, 6, generator=torch.Generator().manual_seed(5))
    full = full[0, 3:].tolist()  # (1, 9) → 不停时生成的 6 个 token
    # 取第 2 种 token 当 eos，才有“停下之前已经生成了别的 token”的情况
    eos = next(token for token in full if token != full[0])
    stop = full.index(eos)

    out = generate(
        model,
        idx,
        6,
        generator=torch.Generator().manual_seed(5),
        use_cache=use_cache,
        eos_token_id=eos,
    )

    assert stop >= 1
    assert out[0, 3:].tolist() == full[: stop + 1]


def test_eos_is_only_for_one_row_at_a_time() -> None:
    from tlab.eval.generate import generate

    idx = torch.tensor([[1, 2], [3, 4]])  # (2, 2)

    # 各行会在不同位置结束，先不处理
    with pytest.raises(ValueError, match="eos"):
        generate(NextIsLength(), idx, 3, top_k=1, eos_token_id=4)


def test_stopping_at_eos_restores_train_mode() -> None:
    from tlab.eval.generate import generate

    model = NextIsLength().train()

    generate(model, torch.tensor([[9, 9]]), 6, top_k=1, eos_token_id=2)

    assert model.training is True


class TiedLogits(nn.Module):
    """3 号和 7 号分数完全相同：bf16 下分数很粗，真实模型也会这样并列。"""

    block_size = 8

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        logits = torch.zeros(B, T, 11)  # (B, T, V)
        logits[:, :, 3] = 5.0
        logits[:, :, 7] = 5.0
        return logits


def test_greedy_is_deterministic_when_scores_are_tied() -> None:
    from tlab.eval.generate import generate

    idx = torch.tensor([[1]])  # (1, 1)

    outputs = {
        tuple(
            generate(
                TiedLogits(),
                idx,
                4,
                top_k=1,
                generator=torch.Generator().manual_seed(seed),
            )[0].tolist()
        )
        for seed in range(20)
    }

    # 和 torch.argmax、transformers 的贪心一样，并列时取编号最小的
    assert outputs == {(1, 3, 3, 3, 3)}
