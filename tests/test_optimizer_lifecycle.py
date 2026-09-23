"""GA / PSO 单次运行生命周期的联动回归测试。

覆盖的问题（两种优化器共享同一组行为契约）：

1. 同一实例重复运行时，历史长度不再翻倍，最优解/找到代数不混入上一次运行；
2. 固定种子的普通执行可精确重放，接续执行由 ``continued`` 明确标记；
3. 显式接续（export_state / continue_run）与一次性连续执行逐位等价；
4. 返回的最优位置是独立拷贝，不会被种群/粒子数组的后续写入悄悄改动；
5. 代数、历史长度、最终种群与最佳适应度彼此一致；
6. 接续状态带标签与结构校验，GA/PSO 不能互换，被篡改时拒绝接续。
"""

import numpy as np
import pytest

from wind_farm_opt.constraints.boundary import create_rectangular_boundary
from wind_farm_opt.optimization import (
    GAConfig,
    GeneticAlgorithm,
    OptimizeResult,
    PSOConfig,
    ParticleSwarmOptimizer,
)

N_TURB = 5
ROTOR_DIAMETERS = np.full(N_TURB, 80.0)
BOUNDARY = create_rectangular_boundary(4000.0, 4000.0)

# 每个索引对应一个彼此远离的吸引点（间距 600m > 5D=400m），
# 最优点唯一且可行，适应度完全确定。
_TARGETS = np.column_stack(
    [np.linspace(-1200.0, 1200.0, N_TURB), np.zeros(N_TURB)]
)


def make_fitness():
    """确定性的解析适应度：负的各风机到指定目标点的平方距离和。"""

    def fitness(positions: np.ndarray) -> float:
        return -float(np.sum((positions - _TARGETS) ** 2)) / 1e6

    return fitness


def make_ga(generations, seed=7, population=10):
    return GeneticAlgorithm(
        n_turbines=N_TURB,
        rotor_diameters=ROTOR_DIAMETERS,
        boundary=BOUNDARY,
        fitness_fn=make_fitness(),
        config=GAConfig(
            population_size=population,
            max_generations=generations,
            seed=seed,
        ),
    )


def make_pso(iterations, seed=7, swarm=10):
    return ParticleSwarmOptimizer(
        n_turbines=N_TURB,
        rotor_diameters=ROTOR_DIAMETERS,
        boundary=BOUNDARY,
        fitness_fn=make_fitness(),
        config=PSOConfig(
            swarm_size=swarm,
            max_iterations=iterations,
            seed=seed,
        ),
    )


def results_equal(a: OptimizeResult, b: OptimizeResult) -> bool:
    """数组/列表/字典感知的结果比较。"""
    if a.best_fitness != b.best_fitness:
        return False
    if a.best_generation != b.best_generation:
        return False
    if a.generations != b.generations:
        return False
    if a.continued != b.continued:
        return False
    for key in (
        "best_positions",
        "convergence_history",
        "mean_history",
        "final_population",
        "final_fitness",
    ):
        if not np.array_equal(getattr(a, key), getattr(b, key)):
            return False
    if a.rng_state != b.rng_state:
        return False
    return True


def _make(kind, steps, seed=7, size=10):
    if kind == "ga":
        return make_ga(steps, seed=seed, population=size)
    return make_pso(steps, seed=seed, swarm=size)


@pytest.fixture(params=["ga", "pso"])
def optimizer_kind(request):
    return request.param


# ---------------------------------------------------------------------------
# 1. 单次运行：独立状态，不泄漏到第二次运行
# ---------------------------------------------------------------------------


def test_repeated_runs_do_not_accumulate_history(optimizer_kind):
    """第二次普通执行的历史长度必须与第一次相同，而不是翻倍。"""
    opt = _make(optimizer_kind, 4)
    first = opt.optimize(verbose=False)
    second = opt.optimize(verbose=False)

    assert len(first.convergence_history) == 4
    assert len(second.convergence_history) == 4
    assert len(first.mean_history) == 4
    assert len(second.mean_history) == 4
    assert second.generations == 4
    assert first.continued is False
    assert second.continued is False


def test_repeated_runs_on_same_instance_are_identical(optimizer_kind):
    """同一实例连跑两次 = 两次独立重放，最优解与找到代数不混入上次运行。"""
    opt = _make(optimizer_kind, 4)
    first = opt.optimize(verbose=False)
    second = opt.optimize(verbose=False)

    assert results_equal(first, second)


def test_fixed_seed_replays_bit_identically_on_new_instances(optimizer_kind):
    """固定种子：新建实例重放必须逐位一致（这是重放，不是接续）。"""
    first = _make(optimizer_kind, 4, seed=123).optimize(verbose=False)
    second = _make(optimizer_kind, 4, seed=123).optimize(verbose=False)

    assert first.continued is second.continued is False
    np.testing.assert_array_equal(
        first.convergence_history, second.convergence_history
    )
    np.testing.assert_array_equal(first.final_population, second.final_population)
    np.testing.assert_array_equal(first.best_positions, second.best_positions)
    assert first.best_generation == second.best_generation


def test_optimizer_instance_holds_no_run_state_after_optimize(optimizer_kind):
    """运行结束后实例上不得残留历史/RNG/全局最佳等易变状态。"""
    opt = _make(optimizer_kind, 3)
    opt.optimize(verbose=False)

    for attr in (
        "rng",
        "convergence_history",
        "mean_history",
        "_best_positions",
        "_best_fitness",
        "_best_generation",
        "_best_global_pos",
        "_best_global_fitness",
        "_best_iteration",
    ):
        assert not hasattr(opt, attr), f"运行状态泄漏到实例: {attr}"


def test_fresh_run_after_a_continued_run_starts_clean(optimizer_kind):
    """接续运行之后再做普通执行，仍然是从零开始的独立运行。"""
    opt = _make(optimizer_kind, 3)
    part1 = opt.optimize(verbose=False)
    state = opt.export_state(part1)
    continued = opt.continue_run(state, 3, verbose=False)
    fresh = opt.optimize(verbose=False)

    assert len(continued.convergence_history) == 6
    assert continued.continued is True
    assert len(fresh.convergence_history) == 3
    assert fresh.continued is False
    # 全新运行应与最初那次普通重放一致
    baseline = _make(optimizer_kind, 3).optimize(verbose=False)
    assert results_equal(fresh, baseline)


# ---------------------------------------------------------------------------
# 2. 返回值内部一致性
# ---------------------------------------------------------------------------


def test_result_fields_are_mutually_consistent(optimizer_kind):
    """代数、历史长度、最终种群、最佳适应度必须彼此一致。"""
    size = 10
    result = _make(optimizer_kind, 5, size=size).optimize(verbose=False)

    assert result.generations == 5
    assert len(result.convergence_history) == result.generations
    assert len(result.mean_history) == result.generations
    assert result.final_population.shape == (size, 2 * N_TURB)
    assert result.final_fitness.shape == (size,)

    # 收敛历史是“至今最优”，必须单调不减且末值就是最佳适应度
    history = np.asarray(result.convergence_history)
    assert np.all(np.diff(history) >= 0)
    assert history[-1] == pytest.approx(result.best_fitness)

    # 找到代数合法：0 表示初始代；最佳一旦在第 k 代被找到，
    # 该代结束时（history[k-1]）起历史值即为最优值
    assert 0 <= result.best_generation <= result.generations
    first_idx = max(result.best_generation - 1, 0)
    assert history[first_idx] == pytest.approx(result.best_fitness)

    # 最优位置必须确实取得所声称的适应度
    assert make_fitness()(result.best_positions) == pytest.approx(
        result.best_fitness
    )
    assert result.best_positions.shape == (N_TURB, 2)


def test_best_position_is_not_aliased_by_population_arrays(optimizer_kind):
    """最优位置是独立拷贝：改写最终种群/适应度不得影响已返回的最优解。"""
    result = _make(optimizer_kind, 4).optimize(verbose=False)
    snapshot = result.best_positions.copy()

    result.final_population.fill(999.0)
    result.final_fitness.fill(-999.0)

    np.testing.assert_array_equal(result.best_positions, snapshot)
    # 种群被改后最优位置仍然成立（自含的拷贝）
    assert make_fitness()(result.best_positions) == pytest.approx(
        result.best_fitness
    )


def test_returned_arrays_are_independent_copies(optimizer_kind):
    """再次运行不会改动上一次返回结果中的任何数组。"""
    opt = _make(optimizer_kind, 3)
    first = opt.optimize(verbose=False)
    first_pop = first.final_population.copy()
    first_best = first.best_positions.copy()

    opt.optimize(verbose=False)

    np.testing.assert_array_equal(first.final_population, first_pop)
    np.testing.assert_array_equal(first.best_positions, first_best)


# ---------------------------------------------------------------------------
# 3. 显式接续：可验证、且与一次性连续执行等价
# ---------------------------------------------------------------------------


def test_continue_run_matches_single_long_run(optimizer_kind):
    """3+3 分段接续必须与一次性运行 6 步逐位等价。"""
    steps = 3

    # 一次性运行
    one_shot = _make(optimizer_kind, 2 * steps, seed=7).optimize(verbose=False)

    # 分段：同一实例上先跑一段，导出状态，再接续
    opt = _make(optimizer_kind, steps, seed=7)
    part1 = opt.optimize(verbose=False)
    assert part1.continued is False
    assert len(part1.convergence_history) == steps

    state = opt.export_state(part1)
    part2 = opt.continue_run(state, steps, verbose=False)

    assert part2.continued is True
    assert len(part2.convergence_history) == 2 * steps
    assert part2.generations == 2 * steps
    # 前半段历史保持不变
    np.testing.assert_array_equal(
        part2.convergence_history[:steps], part1.convergence_history
    )

    np.testing.assert_array_equal(
        part2.convergence_history, one_shot.convergence_history
    )
    np.testing.assert_array_equal(part2.mean_history, one_shot.mean_history)
    np.testing.assert_array_equal(part2.final_population, one_shot.final_population)
    np.testing.assert_array_equal(part2.final_fitness, one_shot.final_fitness)
    np.testing.assert_array_equal(part2.best_positions, one_shot.best_positions)
    assert part2.best_fitness == pytest.approx(one_shot.best_fitness)
    assert part2.best_generation == one_shot.best_generation
    assert part2.rng_state == one_shot.rng_state


def test_continue_via_optimize_keyword_matches_helper(optimizer_kind):
    """optimize(continue_from=...) 与 continue_run(...) 行为一致。"""
    opt = _make(optimizer_kind, 2)
    part1 = opt.optimize(verbose=False)
    state = opt.export_state(part1)

    via_keyword = opt.optimize(verbose=False, continue_from=state)
    via_helper = opt.continue_run(opt.export_state(part1), 2, verbose=False)

    assert results_equal(via_keyword, via_helper)
    assert via_keyword.continued is True


def test_same_state_can_continue_twice_without_corruption(optimizer_kind):
    """接续不会消费/改写状态对象本身：同一状态接续两次结果相同。"""
    opt = _make(optimizer_kind, 2)
    part1 = opt.optimize(verbose=False)
    state = opt.export_state(part1)

    run_a = opt.continue_run(state, 2, verbose=False)
    key = "population" if optimizer_kind == "ga" else "positions"
    snapshot = np.array(state[key])
    run_b = opt.continue_run(state, 2, verbose=False)

    assert results_equal(run_a, run_b)
    # 状态数组未被接续过程改写
    np.testing.assert_array_equal(state[key], snapshot)


def test_export_state_returns_independent_copies(optimizer_kind):
    """导出状态后篡改状态数组，不得污染原结果，且篡改会被校验拒绝。"""
    opt = _make(optimizer_kind, 2)
    result = opt.optimize(verbose=False)
    state = opt.export_state(result)
    best_snapshot = result.best_positions.copy()

    key = "population" if optimizer_kind == "ga" else "positions"
    state[key].fill(999.0)

    np.testing.assert_array_equal(result.best_positions, best_snapshot)
    # 种群形状没变，仍可运行，但历史保持独立；这里再篡改历史长度
    state["convergence_history"].append(12345.0)
    assert len(result.convergence_history) == 2
    with pytest.raises(ValueError, match="历史长度"):
        opt.optimize(verbose=False, continue_from=state)


def test_continue_run_does_not_mutate_config(optimizer_kind):
    """continue_run 的临时步数不得写回配置。"""
    opt = _make(optimizer_kind, 2)
    part1 = opt.optimize(verbose=False)
    state = opt.export_state(part1)

    opt.continue_run(state, 5, verbose=False)

    if optimizer_kind == "ga":
        assert opt.config.max_generations == 2
    else:
        assert opt.config.max_iterations == 2
    with pytest.raises(ValueError):
        opt.continue_run(state, 0, verbose=False)


# ---------------------------------------------------------------------------
# 4. 接续状态校验：GA/PSO 联动，拒绝串用与伪造
# ---------------------------------------------------------------------------


def test_cross_algorithm_state_is_rejected():
    """GA 的状态不能喂给 PSO，反之亦然。"""
    ga = make_ga(2)
    ga_result = ga.optimize(verbose=False)
    ga_state = ga.export_state(ga_result)

    pso = make_pso(2)
    pso_result = pso.optimize(verbose=False)
    pso_state = pso.export_state(pso_result)

    with pytest.raises(ValueError, match="接续状态"):
        pso.optimize(verbose=False, continue_from=ga_state)
    with pytest.raises(ValueError, match="接续状态"):
        ga.optimize(verbose=False, continue_from=pso_state)


def test_foreign_and_corrupt_states_are_rejected(optimizer_kind):
    opt = _make(optimizer_kind, 2)

    with pytest.raises(ValueError):
        opt.optimize(verbose=False, continue_from={"not": "a state"})

    result = opt.optimize(verbose=False)
    state = opt.export_state(result)

    # 篡改标签
    bad_tag = dict(state)
    bad_tag["tag"] = "forged-tag"
    with pytest.raises(ValueError, match="标签"):
        opt.optimize(verbose=False, continue_from=bad_tag)

    # 篡改种群形状
    bad_shape = dict(state)
    key = "population" if optimizer_kind == "ga" else "positions"
    bad_shape[key] = np.zeros((3, 2 * N_TURB))
    with pytest.raises(ValueError, match="形状"):
        opt.optimize(verbose=False, continue_from=bad_shape)

    # 缺字段
    missing = dict(state)
    del missing["rng_state"]
    with pytest.raises(ValueError, match="字段"):
        opt.optimize(verbose=False, continue_from=missing)


def test_export_rejects_tampered_result(optimizer_kind):
    """结果内部不一致（历史长度被改）时禁止导出接续状态。"""
    opt = _make(optimizer_kind, 2)
    result = opt.optimize(verbose=False)
    result.convergence_history.pop()

    with pytest.raises(ValueError):
        opt.export_state(result)


# ---------------------------------------------------------------------------
# 5. PSO 专项：个体最佳与速度同样独立、且参与接续
# ---------------------------------------------------------------------------


def test_pso_carries_velocities_and_personal_bests():
    result = make_pso(3).optimize(verbose=False)

    assert result.velocities.shape == (10, 2 * N_TURB)
    assert result.personal_best_positions.shape == (10, 2 * N_TURB)
    assert result.personal_best_fitness.shape == (10,)

    # 个体最佳适应度必须不低于各自当前适应度
    assert np.all(result.personal_best_fitness + 1e-12 >= result.final_fitness)


def test_pso_personal_best_not_leaked_between_runs():
    """第二次普通运行的个体最佳不得残留上一次运行的值。"""
    pso = make_pso(3, seed=11)
    first = pso.optimize(verbose=False)
    second = pso.optimize(verbose=False)

    np.testing.assert_array_equal(
        first.personal_best_positions, second.personal_best_positions
    )
    np.testing.assert_array_equal(first.velocities, second.velocities)


# ---------------------------------------------------------------------------
# 6. 对外接口（含绘图依赖字段）保持可用
# ---------------------------------------------------------------------------


def test_result_supports_legacy_fields_used_by_plotting():
    """plotting 依赖的字段语义不变：代数轴与历史等长。"""
    result = make_ga(3).optimize(verbose=False)
    generations_axis = np.arange(1, len(result.convergence_history) + 1)
    assert len(generations_axis) == result.generations
    assert isinstance(result.best_generation, int)
