"""GA / PSO 单次运行生命周期的联动回归测试。

同一组测试以参数化方式同时施加于遗传算法与粒子群优化器，验证两者一致的
生命周期契约：

1. 普通执行彼此独立：固定种子重放逐位一致，历史不翻倍，最优解不串台；
2. 返回结果内部自洽：代数 / 历史长度 / 最终种群 / 最佳适应度彼此一致；
3. best_positions 是独立副本，不会被种群数组的后续写入悄悄改动；
4. 接续运行必须显式声明 (continue_run=True)，且与一次不间断长运行等价，
   可验证、可与重放区分。
"""

import numpy as np
import pytest

from tests.factories import make_ga, make_pso


# (工厂, 单次运行步数, 接续步数)
FACTORIES = {
    "ga": (make_ga, 5, 3),
    "pso": (make_pso, 6, 4),
}


def make(name, steps=None, **kwargs):
    factory, default_steps, _ = FACTORIES[name]
    if name == "ga":
        return make_ga(
            max_generations=steps if steps is not None else default_steps, **kwargs
        )
    return make_pso(
        max_iterations=steps if steps is not None else default_steps, **kwargs
    )


def set_steps(optimizer, name, steps):
    """配置接续运行本次执行的步数（代数/迭代数）。"""
    if name == "ga":
        optimizer.config.max_generations = steps
    else:
        optimizer.config.max_iterations = steps


@pytest.mark.parametrize("name", ["ga", "pso"], ids=["GA", "PSO"])
class TestSingleRunLifecycle:
    def test_history_length_matches_generations(self, name):
        optimizer, config, _, _ = make(name)
        steps = (
            config.max_generations if name == "ga" else config.max_iterations
        )
        result = optimizer.optimize(verbose=False)

        assert result.n_generations == steps
        assert len(result.convergence_history) == steps + 1
        assert len(result.mean_history) == steps + 1

    def test_result_fields_mutually_consistent(self, name):
        optimizer, _, _, _ = make(name)
        result = optimizer.optimize(verbose=False)

        # 历史末位 == 报告的最佳适应度 == 最终种群的最大适应度
        assert result.convergence_history[-1] == result.best_fitness
        assert np.isclose(
            result.best_fitness, float(np.max(result.final_fitness))
        )

        # best_positions 确实是最终种群中取得最佳适应度的个体
        best_row = result.best_positions.reshape(1, -1)
        distances = np.linalg.norm(result.final_population - best_row, axis=1)
        match_idx = int(np.argmin(distances))
        assert distances[match_idx] < 1e-9
        assert result.final_fitness[match_idx] == pytest.approx(
            result.best_fitness
        )

        # 最终种群与适应度一一对应
        pop_size = (
            result.final_population.shape[0]
        )
        assert pop_size == result.final_fitness.shape[0]

        # 找到最优解的代数合法，且该代历史首次达到最佳适应度
        assert 0 <= result.best_generation <= result.n_generations
        assert result.convergence_history[result.best_generation] == pytest.approx(
            result.best_fitness
        )
        if result.best_generation > 0:
            assert (
                result.convergence_history[result.best_generation]
                > result.convergence_history[result.best_generation - 1]
            )

        # “至今最优”历史单调不降
        diffs = np.diff(result.convergence_history)
        assert np.all(diffs >= 0)

    def test_repeated_runs_are_independent_replays(self, name):
        optimizer, _, _, _ = make(name)

        first = optimizer.optimize(verbose=False)
        second = optimizer.optimize(verbose=False)

        # 第二次是全新运行：历史长度不翻倍
        assert len(second.convergence_history) == len(first.convergence_history)
        assert second.n_generations == first.n_generations

        # 固定种子重放：逐位一致
        np.testing.assert_array_equal(
            np.array(second.convergence_history),
            np.array(first.convergence_history),
        )
        np.testing.assert_array_equal(
            np.array(second.mean_history), np.array(first.mean_history)
        )
        np.testing.assert_array_equal(second.best_positions, first.best_positions)
        assert second.best_fitness == first.best_fitness
        assert second.best_generation == first.best_generation
        np.testing.assert_array_equal(
            second.final_population, first.final_population
        )

    def test_later_run_does_not_mutate_previous_result(self, name):
        optimizer, _, _, _ = make(name)

        first = optimizer.optimize(verbose=False)
        snapshot_best = first.best_positions.copy()
        snapshot_history = np.array(first.convergence_history)
        snapshot_mean = np.array(first.mean_history)
        snapshot_pop = first.final_population.copy()
        snapshot_fitness = first.final_fitness.copy()

        # 在同一实例上再跑一次全新运行，第一次返回的结果不得被改动
        optimizer.optimize(verbose=False)

        np.testing.assert_array_equal(first.best_positions, snapshot_best)
        np.testing.assert_array_equal(
            np.array(first.convergence_history), snapshot_history
        )
        np.testing.assert_array_equal(np.array(first.mean_history), snapshot_mean)
        np.testing.assert_array_equal(first.final_population, snapshot_pop)
        np.testing.assert_array_equal(first.final_fitness, snapshot_fitness)

    def test_best_positions_independent_of_population_writes(self, name):
        optimizer, _, _, _ = make(name)
        result = optimizer.optimize(verbose=False)

        best_snapshot = result.best_positions.copy()
        # 外部写入返回的种群数组不得波及 best_positions
        result.final_population[:] = 12345.0
        np.testing.assert_array_equal(result.best_positions, best_snapshot)

        # 优化器内部再次初始化/运行也不得改写已返回的最优位置
        optimizer.optimize(verbose=False)
        np.testing.assert_array_equal(result.best_positions, best_snapshot)

    def test_gen0_best_not_aliased_to_population(self, name):
        """回归：初代最优若之后未被超越，其位置不得被种群后续写入改动。

        使用平坦适应度（所有个体相同），最优解必然停留在第 0 代且
        argmax 取到初始种群第 0 行；旧实现中 GA 把 _best_positions
        绑定为种群行的视图，会被后续代对种群行的覆盖悄悄改写。
        """
        optimizer, config, _, _ = make(name, steps=4)

        evaluated = []

        def flat_fitness(positions):
            evaluated.append(np.asarray(positions, dtype=float).copy())
            return 1.0

        optimizer.fitness_fn = flat_fitness

        result = optimizer.optimize(verbose=False)

        assert result.best_generation == 0
        assert result.best_fitness == 1.0
        # 第 0 代最优即初始种群中第一个被评估的个体（适应度全等，argmax
        # 取首个）；若 best_positions 是种群行的视图，此时已被后代覆盖
        gen0_best = evaluated[0]
        np.testing.assert_array_equal(result.best_positions, gen0_best)

    def test_optimizer_actually_improves(self, name):
        optimizer, _, _, _ = make(name)
        result = optimizer.optimize(verbose=False)

        assert result.best_fitness >= result.convergence_history[0]
        # 在该玩具问题上两步算法都应找到明显优于初始最优的解
        assert result.best_fitness > result.convergence_history[0]


@pytest.mark.parametrize("name", ["ga", "pso"], ids=["GA", "PSO"])
class TestContinuation:
    def test_continue_without_prior_run_raises(self, name):
        optimizer, _, _, _ = make(name)
        with pytest.raises(RuntimeError, match="接续"):
            optimizer.optimize(verbose=False, continue_run=True)

    def test_continuation_appends_history_with_cumulative_counts(self, name):
        factory, first_steps, more_steps = FACTORIES[name]
        optimizer, _, _, _ = make(name, steps=first_steps)

        first = optimizer.optimize(verbose=False)
        set_steps(optimizer, name, more_steps)
        continued = optimizer.optimize(verbose=False, continue_run=True)

        total = first_steps + more_steps
        assert len(continued.convergence_history) == total + 1
        assert len(continued.mean_history) == total + 1
        assert continued.n_generations == total

        # 前 first_steps+1 条历史必须与首次运行完全一致（接续而非重放）
        np.testing.assert_array_equal(
            np.array(continued.convergence_history[: first_steps + 1]),
            np.array(first.convergence_history),
        )
        # 接续产生了新的历史条目
        assert len(continued.convergence_history) > len(first.convergence_history)
        # 累计最优代数合法且不早于首次运行的判定
        assert 0 <= continued.best_generation <= total
        assert continued.convergence_history[continued.best_generation] == pytest.approx(
            continued.best_fitness
        )

    def test_continuation_equals_uninterrupted_run(self, name):
        """接续运行必须与一次不间断的等长运行逐位等价（可验证的接续）。"""
        first_steps, more_steps = FACTORIES[name][1], FACTORIES[name][2]

        stepped, _, _, _ = make(name, steps=first_steps)
        stepped.optimize(verbose=False)
        set_steps(stepped, name, more_steps)
        continued = stepped.optimize(verbose=False, continue_run=True)

        uninterrupted, _, _, _ = make(name, steps=first_steps + more_steps)
        single = uninterrupted.optimize(verbose=False)

        np.testing.assert_array_equal(
            np.array(continued.convergence_history),
            np.array(single.convergence_history),
        )
        np.testing.assert_array_equal(
            np.array(continued.mean_history), np.array(single.mean_history)
        )
        np.testing.assert_allclose(
            continued.best_positions, single.best_positions, atol=1e-9
        )
        np.testing.assert_array_equal(
            continued.final_population, single.final_population
        )
        assert continued.best_fitness == single.best_fitness
        assert continued.best_generation == single.best_generation

    def test_replay_after_continuation_is_independent(self, name):
        """接续之后的普通调用重新独立开始：固定种子可与接续区分。"""
        first_steps, more_steps = FACTORIES[name][1], FACTORIES[name][2]

        optimizer, _, _, _ = make(name, steps=first_steps)
        optimizer.optimize(verbose=False)
        set_steps(optimizer, name, more_steps)
        optimizer.optimize(verbose=False, continue_run=True)

        # 普通调用：重置为当前步数的独立运行，而不是在接续态上继续
        fresh, _, _, _ = make(name, steps=more_steps)
        expected = fresh.optimize(verbose=False)

        replay = optimizer.optimize(verbose=False)
        assert replay.n_generations == more_steps
        assert len(replay.convergence_history) == more_steps + 1
        np.testing.assert_array_equal(
            np.array(replay.convergence_history),
            np.array(expected.convergence_history),
        )
        assert replay.best_generation == expected.best_generation

    def test_continue_with_changed_population_size_raises(self, name):
        optimizer, _, _, _ = make(name, steps=2)
        optimizer.optimize(verbose=False)

        optimizer.config = (
            optimizer.config.__class__(
                **{
                    **optimizer.config.__dict__,
                    ("population_size" if name == "ga" else "swarm_size"): (
                        optimizer.config.population_size + 1
                        if name == "ga"
                        else optimizer.config.swarm_size + 1
                    ),
                }
            )
        )
        with pytest.raises(ValueError, match="不一致|种群大小|粒子群大小"):
            optimizer.optimize(verbose=False, continue_run=True)


class TestGaPsoLinkedContract:
    """GA 与 PSO 在同一问题上的联动契约。"""

    def test_same_problem_both_optimizers_return_consistent_results(self):
        ga, _, fitness_fn, _ = make_ga(max_generations=8)
        pso, _, _, _ = make_pso(max_iterations=8)

        ga_result = ga.optimize(verbose=False)
        pso_result = pso.optimize(verbose=False)

        for result in (ga_result, pso_result):
            # 两者都满足统一的结果结构与自洽性契约
            assert len(result.convergence_history) == 9
            assert result.n_generations == 8
            assert result.convergence_history[-1] == result.best_fitness
            assert np.isclose(
                result.best_fitness,
                float(np.max(result.final_fitness)),
            )
            # 报告的最优位置重新评估确实得到报告的适应度
            assert fitness_fn(result.best_positions) == pytest.approx(
                result.best_fitness
            )
