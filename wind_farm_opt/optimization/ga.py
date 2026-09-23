"""遗传算法优化器。"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..constraints.boundary import SiteBoundary
from ..constraints.spacing import (
    check_min_spacing,
    compute_min_spacing_from_diameters,
    enforce_min_spacing,
)


@dataclass
class GAConfig:
    """遗传算法配置参数。

    Parameters
    ----------
    population_size : int
        种群大小
    max_generations : int
        最大迭代代数
    crossover_rate : float
        交叉概率
    mutation_rate : float
        变异概率
    mutation_strength : float
        变异强度（坐标标准差占场地范围的比例）
    elite_ratio : float
        精英保留比例
    tournament_size : int
        锦标赛选择的规模
    min_spacing_multiple : float
        最小间距倍数（相对于转子直径）
    penalty_factor : float
        约束违反惩罚因子
    seed : Optional[int]
        随机种子
    """

    population_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    mutation_strength: float = 0.1
    elite_ratio: float = 0.1
    tournament_size: int = 3
    min_spacing_multiple: float = 5.0
    penalty_factor: float = 1e6
    seed: Optional[int] = None


@dataclass
class OptimizeResult:
    """优化结果。

    Parameters
    ----------
    best_positions : np.ndarray
        最优风机位置 (N_turb, 2)
    best_fitness : float
        最优适应度（净AEP，MWh/year）
    best_generation : int
        找到最优解的代数
    convergence_history : list[float]
        每代最优适应度历史
    mean_history : list[float]
        每代平均适应度历史
    final_population : np.ndarray
        最终种群 (pop_size, N_turb*2)
    final_fitness : np.ndarray
        最终种群适应度 (pop_size,)
    n_generations : int
        本次运行实际执行的代数（含接续运行的累计代数），
        与 convergence_history / mean_history 的长度一致
    """

    best_positions: np.ndarray
    best_fitness: float
    best_generation: int
    convergence_history: list[float]
    mean_history: list[float]
    final_population: np.ndarray
    final_fitness: np.ndarray
    n_generations: int = 0


class GeneticAlgorithm:
    """遗传算法机位优化器。

    优化目标：最大化年净发电量（等价于最小化尾流损失）。
    约束：最小间距、场地边界内。

    运行生命周期
    ------------
    每次调用 :meth:`optimize`（``continue_run=False``，默认）都从完全独立的
    状态开始：随机数发生器按 ``config.seed`` 重新播种，收敛历史、全局最优等
    运行状态全部清空，同一实例上的重复运行互不泄漏。固定种子下重复调用
    ``optimize()`` 得到逐位一致的结果（重放）。

    需要连续试验（在已有种群基础上继续进化）时，显式传入
    ``continue_run=True``：本次运行从上次运行保留的种群、最优解与随机数
    状态接续执行，历史记录追加而非重置。接续运行的结果中
    ``best_generation`` 与 ``n_generations`` 为累计代数，可与重放区分。
    """

    def __init__(
        self,
        n_turbines: int,
        rotor_diameters: np.ndarray,
        boundary: SiteBoundary,
        fitness_fn: Callable[[np.ndarray], float],
        config: Optional[GAConfig] = None,
    ) -> None:
        """
        Parameters
        ----------
        n_turbines : int
            风机台数
        rotor_diameters : np.ndarray
            每台风机的转子直径
        boundary : SiteBoundary
            场地边界
        fitness_fn : Callable[[np.ndarray], float]
            适应度函数，输入位置数组 (N_turb, 2)，返回净AEP
        config : Optional[GAConfig]
            算法配置参数
        """
        self.n_turbines = n_turbines
        self.rotor_diameters = np.asarray(rotor_diameters, dtype=np.float64)
        self.boundary = boundary
        self.fitness_fn = fitness_fn
        self.config = config if config is not None else GAConfig()

        self.min_spacing = compute_min_spacing_from_diameters(
            self.rotor_diameters,
            self.config.min_spacing_multiple,
        )

        self.n_dim = n_turbines * 2
        self.x_range = boundary.x_max - boundary.x_min
        self.y_range = boundary.y_max - boundary.y_min

        # 单次运行状态（每次普通 optimize() 调用前由 _reset_run_state 重置）
        self.rng = np.random.default_rng(self.config.seed)
        self._reset_run_state()

    def _reset_run_state(self) -> None:
        """重置单次运行的全部状态，使新一次运行从独立状态开始。"""
        self.rng = np.random.default_rng(self.config.seed)

        self._best_positions: Optional[np.ndarray] = None
        self._best_fitness = -np.inf
        self._best_generation = 0

        self.convergence_history: list[float] = []
        self.mean_history: list[float] = []

        # 上一次运行保留的种群（供 continue_run=True 接续使用）
        self._last_population: Optional[np.ndarray] = None
        self._last_fitness: Optional[np.ndarray] = None
        self._generations_completed = 0

    def _initialize_population(self, pop_size: int) -> np.ndarray:
        """初始化种群。

        每个个体是展平的位置向量：[x1, y1, x2, y2, ..., xn, yn]
        """
        population = np.zeros((pop_size, self.n_dim), dtype=np.float64)

        for i in range(pop_size):
            positions = self._generate_valid_layout()
            population[i] = positions.flatten()

        return population

    def _generate_valid_layout(self) -> np.ndarray:
        """生成一个满足约束的初始布局。"""
        max_attempts = 100

        for _ in range(max_attempts):
            try:
                positions = self.boundary.sample_random_points(
                    self.n_turbines, self.rng, max_attempts=50
                )
                valid, _ = check_min_spacing(positions, self.min_spacing)
                if valid:
                    return positions
            except RuntimeError:
                continue

            try:
                positions = self.boundary.sample_random_points(
                    self.n_turbines, self.rng, max_attempts=50
                )
                positions = enforce_min_spacing(
                    positions, self.min_spacing, self.boundary, self.rng
                )
                return positions
            except RuntimeError:
                continue

        raise RuntimeError("无法生成满足约束的初始布局")

    def _compute_penalty(self, positions_flat: np.ndarray) -> float:
        """计算约束违反惩罚。"""
        positions = positions_flat.reshape(self.n_turbines, 2)

        penalty = 0.0

        inside = self.boundary.contains_all(positions)
        if not inside.all():
            n_violations = np.sum(~inside)
            penalty += n_violations * self.config.penalty_factor

        valid, violations = check_min_spacing(positions, self.min_spacing)
        if not valid:
            for i, j in violations:
                dist = np.linalg.norm(positions[i] - positions[j])
                penalty += (self.min_spacing - dist) * self.config.penalty_factor

        return penalty

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """评估整个种群的适应度（带惩罚）。"""
        pop_size = population.shape[0]
        fitness = np.zeros(pop_size, dtype=np.float64)

        for i in range(pop_size):
            positions = population[i].reshape(self.n_turbines, 2)

            penalty = self._compute_penalty(population[i])

            if penalty > 0:
                fitness[i] = -penalty
            else:
                try:
                    fitness[i] = self.fitness_fn(positions)
                except Exception:
                    fitness[i] = -self.config.penalty_factor

        return fitness

    def _tournament_selection(
        self, population: np.ndarray, fitness: np.ndarray, n_select: int
    ) -> np.ndarray:
        """锦标赛选择。"""
        pop_size = population.shape[0]
        selected = np.zeros((n_select, self.n_dim), dtype=np.float64)

        for i in range(n_select):
            candidates = self.rng.integers(0, pop_size, size=self.config.tournament_size)
            best_idx = candidates[np.argmax(fitness[candidates])]
            selected[i] = population[best_idx]

        return selected

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """均匀交叉。"""
        if self.rng.random() > self.config.crossover_rate:
            return parent1.copy()

        mask = self.rng.integers(0, 2, size=self.n_dim, dtype=bool)
        child = np.where(mask, parent1, parent2)

        return child

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """高斯变异。"""
        mutated = individual.copy()

        for i in range(self.n_dim):
            if self.rng.random() < self.config.mutation_rate:
                range_sigma = (
                    self.x_range if i % 2 == 0 else self.y_range
                ) * self.config.mutation_strength
                mutated[i] += self.rng.normal(0.0, range_sigma)

        return mutated

    def _repair(self, individual: np.ndarray) -> np.ndarray:
        """修复违反约束的个体。"""
        positions = individual.reshape(self.n_turbines, 2)

        for i in range(self.n_turbines):
            if not self.boundary.contains_point(positions[i]):
                positions[i] = self.boundary.project_to_boundary(positions[i])

        valid, _ = check_min_spacing(positions, self.min_spacing)
        inside = self.boundary.contains_all(positions).all()

        if not (valid and inside):
            try:
                positions = enforce_min_spacing(
                    positions, self.min_spacing, self.boundary, self.rng
                )
            except RuntimeError:
                pass

        return positions.flatten()

    def optimize(self, verbose: bool = True, continue_run: bool = False) -> OptimizeResult:
        """执行优化。

        Parameters
        ----------
        verbose : bool
            是否打印进度信息
        continue_run : bool
            False（默认）：开始一次全新的独立运行，重置历史、最优解与
            随机数状态（按 config.seed 重新播种）。
            True：从上一次运行保留的种群、最优解与随机数状态接续进化，
            历史记录追加；要求此前已完成至少一次运行。

        Returns
        -------
        OptimizeResult
            优化结果。保证 ``n_generations == len(convergence_history)
            == len(mean_history)``，``best_fitness == convergence_history[-1]``
            且等于最终种群的最大适应度，``best_positions`` 为独立副本，
            不会被后续运行或种群数组的写入改动。
        """
        if continue_run:
            if self._last_population is None:
                raise RuntimeError(
                    "没有可接续的运行状态：请先完成一次普通运行 "
                    "(continue_run=False)，再使用 continue_run=True 接续。"
                )
            if self._last_population.shape[0] != self.config.population_size:
                raise ValueError(
                    f"接续运行的种群大小 ({self.config.population_size}) "
                    f"与上次运行保留的种群大小 ({self._last_population.shape[0]}) "
                    f"不一致；如需修改 population_size 请使用普通运行 (continue_run=False)。"
                )
        else:
            self._reset_run_state()

        pop_size = self.config.population_size
        max_gen = self.config.max_generations

        n_elite = max(1, int(pop_size * self.config.elite_ratio))

        if verbose:
            print(f"\n=== 遗传算法优化开始 ===")
            print(f"风机台数: {self.n_turbines}")
            print(f"种群大小: {pop_size}")
            print(f"本次代数: {max_gen}"
                  + (f"（接续运行，已完成 {self._generations_completed} 代）"
                     if continue_run else ""))
            print(f"最小间距: {self.min_spacing:.1f} m "
                  f"({self.config.min_spacing_multiple:.1f}倍转子直径)")
            print(f"场地面积: {self.boundary.area / 1e6:.2f} km²")
            print("=" * 35)

        if continue_run:
            population = self._last_population.copy()
            fitness = self._last_fitness.copy()
        else:
            population = self._initialize_population(pop_size)
            fitness = self._evaluate_population(population)

            best_idx = np.argmax(fitness)
            self._best_fitness = float(fitness[best_idx])
            # 必须拷贝：population[best_idx] 是种群数组的视图，
            # 不拷贝会被后续对种群行的写入悄悄改动最优位置
            self._best_positions = population[best_idx].reshape(
                self.n_turbines, 2
            ).copy()
            self._best_generation = 0

            # 第 0 代（初始种群）的历史记录；之后每完成一代追加一条，
            # 保证 len(history) == n_generations + 1 且末位等于 best_fitness
            self.convergence_history.append(float(self._best_fitness))
            self.mean_history.append(float(np.mean(fitness)))

        for gen in range(max_gen):
            elite_idx = np.argsort(fitness)[-n_elite:]
            elites = population[elite_idx].copy()

            parents = self._tournament_selection(population, fitness, pop_size - n_elite)

            offspring = np.zeros((pop_size - n_elite, self.n_dim), dtype=np.float64)
            for i in range(0, pop_size - n_elite, 2):
                p1 = parents[i]
                p2 = parents[(i + 1) % (pop_size - n_elite)]
                c1 = self._crossover(p1, p2)
                c2 = self._crossover(p2, p1)
                offspring[i] = self._mutate(c1)
                if i + 1 < pop_size - n_elite:
                    offspring[i + 1] = self._mutate(c2)

            for i in range(len(offspring)):
                offspring[i] = self._repair(offspring[i])

            population[:n_elite] = elites
            population[n_elite:] = offspring

            fitness = self._evaluate_population(population)

            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > self._best_fitness:
                self._best_fitness = float(fitness[current_best_idx])
                self._best_positions = population[current_best_idx].reshape(
                    self.n_turbines, 2
                ).copy()
                self._best_generation = self._generations_completed + gen + 1

            self.convergence_history.append(float(self._best_fitness))
            self.mean_history.append(float(np.mean(fitness)))

            if verbose and (gen % 5 == 0 or gen == max_gen - 1):
                print(
                    f"Gen {self._generations_completed + gen + 1:3d} | "
                    f"Best: {self._best_fitness/1e3:8.2f} GWh | "
                    f"Mean: {np.mean(fitness)/1e3:8.2f} GWh | "
                    f"Found@Gen {self._best_generation}"
                )

        self._generations_completed += max_gen

        # 为可能的接续运行保留最终种群（拷贝，避免结果被后续运行改动）
        self._last_population = population.copy()
        self._last_fitness = fitness.copy()

        if verbose:
            print("=" * 35)
            print(f"优化完成!")
            print(f"最优净AEP: {self._best_fitness/1e3:.2f} GWh")
            print(f"找到最优解的代数: {self._best_generation}")

        result = OptimizeResult(
            best_positions=self._best_positions.copy(),
            best_fitness=float(self._best_fitness),
            best_generation=self._best_generation,
            convergence_history=self.convergence_history.copy(),
            mean_history=self.mean_history.copy(),
            final_population=population.copy(),
            final_fitness=fitness.copy(),
            n_generations=self._generations_completed,
        )
        self._check_result_consistency(result)
        return result

    @staticmethod
    def _check_result_consistency(result: OptimizeResult) -> None:
        """校验结果内部一致性（代数、历史长度、最终种群与最优适应度）。"""
        assert len(result.convergence_history) == result.n_generations + 1, (
            f"收敛历史长度 ({len(result.convergence_history)}) "
            f"与代数 ({result.n_generations}) 不一致"
        )
        assert len(result.mean_history) == result.n_generations + 1, (
            f"平均适应度历史长度 ({len(result.mean_history)}) "
            f"与代数 ({result.n_generations}) 不一致"
        )
        assert result.convergence_history[-1] == result.best_fitness, (
            f"收敛历史末位 ({result.convergence_history[-1]}) "
            f"与最优适应度 ({result.best_fitness}) 不一致"
        )
        assert 0 <= result.best_generation <= result.n_generations, (
            f"最优解代数 ({result.best_generation}) 超出 [0, {result.n_generations}]"
        )
        assert result.final_population.shape[0] == result.final_fitness.shape[0], (
            "最终种群与最终适应度数量不一致"
        )
        assert np.isclose(
            result.best_fitness, np.max(result.final_fitness), rtol=1e-9, atol=1e-9
        ), (
            f"最优适应度 ({result.best_fitness}) 与最终种群最大适应度 "
            f"({np.max(result.final_fitness)}) 不一致"
        )
