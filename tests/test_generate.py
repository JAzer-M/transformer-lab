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
