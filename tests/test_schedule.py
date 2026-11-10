import pytest

# 学习率调度；待实现模块在各测试内导入


def make_config(**overrides):
    from tlab.train.config import TrainConfig

    fields = {"learning_rate": 1e-2, "max_iters": 100}
    fields.update(overrides)
    return TrainConfig(**fields)


def test_default_is_constant_like_m1() -> None:
    from tlab.train.schedule import get_lr

    config = make_config()

    # 默认 constant、无 warmup：每一步都是 learning_rate，与 M1 相同
    assert all(get_lr(s, config) == 1e-2 for s in range(100))


def test_warmup_rises_linearly_then_holds_for_constant() -> None:
    from tlab.train.schedule import get_lr

    config = make_config(warmup_iters=4)

    # 第 0 步就不为 0；第 warmup_iters - 1 步恰好升到 learning_rate
    assert [get_lr(s, config) for s in range(4)] == pytest.approx(
        [2.5e-3, 5e-3, 7.5e-3, 1e-2]
    )
    assert all(get_lr(s, config) == 1e-2 for s in range(4, 100))


def test_cosine_decays_from_lr_to_min_lr_by_the_last_step() -> None:
    from tlab.train.schedule import get_lr

    # 共 101 步：第 0–9 步 warmup，第 10–100 步走完 cosine
    config = make_config(
        lr_schedule="cosine", warmup_iters=10, min_lr=1e-3, max_iters=101
    )
    after_warmup = [get_lr(s, config) for s in range(10, 101)]

    # warmup 结束时是最大值，正中间是两者的平均，循环的最后一步正好是 min_lr
    assert get_lr(10, config) == pytest.approx(1e-2)
    assert get_lr(55, config) == pytest.approx((1e-2 + 1e-3) / 2)
    assert get_lr(config.max_iters - 1, config) == pytest.approx(1e-3)
    # 一路只降不升
    assert all(a >= b for a, b in zip(after_warmup, after_warmup[1:]))


def test_cosine_without_warmup_starts_at_lr() -> None:
    from tlab.train.schedule import get_lr

    config = make_config(lr_schedule="cosine", min_lr=0.0)

    assert get_lr(0, config) == pytest.approx(1e-2)
    assert get_lr(config.max_iters - 1, config) == pytest.approx(0.0)


def test_unknown_schedule_is_rejected_when_config_is_built() -> None:
    from tlab.train.config import TrainConfig

    # 拼错的调度名在建配置时就报错，不用等到开始训练
    with pytest.raises(ValueError):
        TrainConfig(lr_schedule="cosin")
