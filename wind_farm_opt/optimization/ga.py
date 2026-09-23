"""遗传算法优化器。

单次运行生命周期
------------------
``GeneticAlgorithm`` 实例本身不携带任何一次运行的易变状态：历史记录、
最优解、随机数流都由 :meth:`optimize` 在启动时新建、在结束时丢弃。因此
同一个实例（以及同一进程内）连续调用 ``optimize`` 两次，第二次得到的
结果与新建实例后运行完全一致，不会混入上一次运行的历史或最优解。

需要把一次长运算拆成多段连续执行时，使用显式的 :meth:`export_state` /
:meth:`continue_run`（或 :meth:`optimize` 的 ``continue_from`` 参数）。
接续状态带有代数计数与校验信息，普通执行与接续执行可以明确区分。
"""

from dataclasses import dataclass
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
        随机种子。每次 ``optimize`` 都会以该种子重新播种，因此固定种子
        的普通执行可精确重放；只有显式接续（``continue_from``）才会
        沿用上次运行结束时的随机数状态。
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
    """单次优化运行的结果。

    所有数组字段均为运行结束时刻的独立拷贝，调用方修改返回值不会影响
    优化器内部，也不会污染后续运行。

    Parameters
    ----------
    best_positions : np.ndarray
        最优风机位置 (N_turb, 2)
    best_fitness : float
        最优适应度（净AEP，MWh/year）
    best_generation : int
        找到最优解的代数（首次运行从 0 计起，接续运行按累计代数计）
    generations : int
        本次结果覆盖的总代数；等于 ``len(convergence_history)``
    convergence_history : list[float]
        每代最优适应度历史（长度等于 ``generations``）
    mean_history : list[float]
        每代平均适应度历史（长度等于 ``generations``）
    final_population : np.ndarray
        最终种群 (pop_size, N_turb*2)
    final_fitness : np.ndarray
        最终种群适应度 (pop_size,)
    continued : bool
        本次运行是否为接续执行
    rng_state : Optional[dict]
        本次运行结束时的随机数状态（numpy BitGenerator 内部状态），
        供 export_state() / 接续运行使用
    velocities : Optional[np.ndarray]
        仅 PSO 使用：最终粒子速度 (pop_size, N_turb*2)；GA 为 None
    personal_best_positions : Optional[np.ndarray]
        仅 PSO 使用：最终个体最佳位置 (pop_size, N_turb*2)；GA 为 None
    personal_best_fitness : Optional[np.ndarray]
        仅 PSO 使用：最终个体最佳适应度 (pop_size,)；GA 为 None
    """

    best_positions: np.ndarray
    best_fitness: float
    best_generation: int
    convergence_history: list
    mean_history: list
    final_population: np.ndarray
    final_fitness: np.ndarray
    generations: int = 0
    continued: bool = False
    rng_state: Optional[dict] = None
    velocities: Optional[np.ndarray] = None
    personal_best_positions: Optional[np.ndarray] = None
    personal_best_fitness: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.generations <= 0:
            self.generations = len(self.convergence_history)


class _RunState:
    """一次运行的易变状态。仅存在于 ``optimize`` 执行期间。"""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.best_positions: Optional[np.ndarray] = None
        self.best_fitness: float = -np.inf
        self.best_generation: int = 0
        self.convergence_history: list = []
        self.mean_history: list = []


class GeneticAlgorithm:
    """遗传算法机位优化器。

    优化目标：最大化年净发电量（等价于最小化尾流损失）。
    约束：最小间距、场地边界内。

    优化器实例只保存问题定义与配置（均不可在运行间泄漏易变状态）；
    每次 :meth:`optimize` 都从独立的随机数流与空历史开始。
    """

    _STATE_TAG = "ga-run-state-v1"

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

    def _new_rng(self) -> np.random.Generator:
        """为一次普通运行建立独立的随机数流。"""
        return np.random.default_rng(self.config.seed)

    def _initialize_population(
        self, pop_size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """初始化种群。

        每个个体是展平的位置向量：[x1, y1, x2, y2, ..., xn, yn]
        """
        population = np.zeros((pop_size, self.n_dim), dtype=np.float64)

        for i in range(pop_size):
            positions = self._generate_valid_layout(rng)
            population[i] = positions.flatten()

        return population

    def _generate_valid_layout(self, rng: np.random.Generator) -> np.ndarray:
        """生成一个满足约束的初始布局。"""
        max_attempts = 100

        for _ in range(max_attempts):
            try:
                positions = self.boundary.sample_random_points(
                    self.n_turbines, rng, max_attempts=50
                )
                valid, _ = check_min_spacing(positions, self.min_spacing)
                if valid:
                    return positions
            except RuntimeError:
                continue

            try:
                positions = self.boundary.sample_random_points(
                    self.n_turbines, rng, max_attempts=50
                )
                positions = enforce_min_spacing(
                    positions, self.min_spacing, self.boundary, rng
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
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        n_select: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """锦标赛选择。"""
        pop_size = population.shape[0]
        selected = np.zeros((n_select, self.n_dim), dtype=np.float64)

        for i in range(n_select):
            candidates = rng.integers(0, pop_size, size=self.config.tournament_size)
            best_idx = candidates[np.argmax(fitness[candidates])]
            selected[i] = population[best_idx]

        return selected

    def _crossover(
        self, parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """均匀交叉。"""
        if rng.random() > self.config.crossover_rate:
            return parent1.copy()

        mask = rng.integers(0, 2, size=self.n_dim, dtype=bool)
        child = np.where(mask, parent1, parent2)

        return child

    def _mutate(self, individual: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """高斯变异。"""
        mutated = individual.copy()

        for i in range(self.n_dim):
            if rng.random() < self.config.mutation_rate:
                range_sigma = (
                    self.x_range if i % 2 == 0 else self.y_range
                ) * self.config.mutation_strength
                mutated[i] += rng.normal(0.0, range_sigma)

        return mutated

    def _repair(
        self, individual: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
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
                    positions, self.min_spacing, self.boundary, rng
                )
            except RuntimeError:
                pass

        return positions.flatten()

    def export_state(self, result: OptimizeResult) -> dict:
        """把一次运行的结果导出为可校验的接续状态。

        Parameters
        ----------
        result : OptimizeResult
            上一次 :meth:`optimize` 或 :meth:`continue_run` 的返回值

        Returns
        -------
        dict
            不透明的接续状态（独立拷贝，可跨进程序列化）。随机数状态取自
            产生该结果的那次运行结束时刻，因此接续后随机序列与一次性
            连续执行完全相同。

        Raises
        ------
        ValueError
            结果被外部改动、内部一致性校验失败时抛出
        """
        self._validate_result(result)
        rng_state = result.rng_state
        return {
            "tag": self._STATE_TAG,
            "population": np.array(result.final_population, dtype=np.float64, copy=True),
            "fitness": np.array(result.final_fitness, dtype=np.float64, copy=True),
            "best_positions": np.array(result.best_positions, dtype=np.float64, copy=True),
            "best_fitness": float(result.best_fitness),
            "best_generation": int(result.best_generation),
            "generations": int(result.generations),
            "convergence_history": list(result.convergence_history),
            "mean_history": list(result.mean_history),
            "rng_state": rng_state,
        }

    def _validate_state(self, state: dict) -> None:
        """校验接续状态的归属与结构。"""
        if not isinstance(state, dict) or state.get("tag") != self._STATE_TAG:
            raise ValueError(
                f"无效的接续状态：期望标签 {self._STATE_TAG!r}，"
                "请使用同类型优化器的 export_state() 生成"
            )
        required = (
            "population",
            "fitness",
            "best_positions",
            "best_fitness",
            "best_generation",
            "generations",
            "convergence_history",
            "mean_history",
            "rng_state",
        )
        missing = [k for k in required if k not in state]
        if missing:
            raise ValueError(f"接续状态缺少字段: {missing}")

        population = np.asarray(state["population"], dtype=np.float64)
        fitness = np.asarray(state["fitness"], dtype=np.float64)
        if population.shape != (self.config.population_size, self.n_dim):
            raise ValueError(
                "接续状态的种群形状与当前配置不一致: "
                f"{population.shape} != {(self.config.population_size, self.n_dim)}"
            )
        if fitness.shape != (self.config.population_size,):
            raise ValueError("接续状态的适应度形状与种群大小不一致")
        if len(state["convergence_history"]) != int(state["generations"]):
            raise ValueError("接续状态的历史长度与已运行代数不一致")

    def _validate_result(self, result: OptimizeResult) -> None:
        """校验返回结果内部各字段彼此一致。"""
        if not isinstance(result, OptimizeResult):
            raise ValueError("需要 OptimizeResult（optimize 的返回值）")
        if result.generations != len(result.convergence_history):
            raise ValueError("结果的 generations 与收敛历史长度不一致")
        if len(result.mean_history) != result.generations:
            raise ValueError("结果的平均适应度历史长度与代数不一致")
        if np.asarray(result.best_positions).shape != (self.n_turbines, 2):
            raise ValueError("结果的最优位置形状不正确")
        if np.asarray(result.final_population).shape != (
            self.config.population_size,
            self.n_dim,
        ):
            raise ValueError("结果的最终种群形状与当前配置不一致")
        if np.asarray(result.final_fitness).shape != (self.config.population_size,):
            raise ValueError("结果的最终适应度形状与种群大小不一致")
        if result.rng_state is None:
            raise ValueError("结果缺少随机数状态，无法接续")

    def optimize(
        self,
        verbose: bool = True,
        continue_from: Optional[dict] = None,
    ) -> OptimizeResult:
        """执行优化。

        Parameters
        ----------
        verbose : bool
            是否打印进度信息
        continue_from : Optional[dict]
            由 :meth:`export_state` 产生的接续状态。默认 ``None`` 表示
            一次全新的独立运行：重新播种随机数、清空历史、重置最优解；
            传入状态时则从该状态显式接续，随机数流、种群、历史和最优解
            都从上一次结束处继续。

        Returns
        -------
        OptimizeResult
            优化结果
        """
        pop_size = self.config.population_size
        max_gen = self.config.max_generations
        n_elite = max(1, int(pop_size * self.config.elite_ratio))

        if continue_from is None:
            # —— 全新独立运行：全新 RNG、空历史、无历史最优解 ——
            state = _RunState(self._new_rng())
            population = self._initialize_population(pop_size, state.rng)
            fitness = self._evaluate_population(population)

            best_idx = int(np.argmax(fitness))
            # .copy()：最优解是独立副本，种群数组之后的写入不会改动它
            state.best_positions = population[best_idx].reshape(
                self.n_turbines, 2
            ).copy()
            state.best_fitness = float(fitness[best_idx])
            state.best_generation = 0
            continued = False
        else:
            # —— 显式接续：恢复上次结束时的全部状态 ——
            self._validate_state(continue_from)
            rng = np.random.default_rng()
            rng.bit_generator.state = continue_from["rng_state"]
            state = _RunState(rng)
            population = np.array(continue_from["population"], copy=True)
            fitness = np.array(continue_from["fitness"], copy=True)
            state.best_positions = np.array(
                continue_from["best_positions"], dtype=np.float64, copy=True
            )
            state.best_fitness = float(continue_from["best_fitness"])
            state.best_generation = int(continue_from["best_generation"])
            state.convergence_history = list(continue_from["convergence_history"])
            state.mean_history = list(continue_from["mean_history"])
            continued = True

        if verbose:
            print(f"\n=== 遗传算法优化{'（接续）' if continued else ''}开始 ===")
            print(f"风机台数: {self.n_turbines}")
            print(f"种群大小: {pop_size}")
            print(f"本次运行代数: {max_gen}")
            print(
                f"累计代数: {len(state.convergence_history)} -> "
                f"{len(state.convergence_history) + max_gen}"
            )
            print(f"最小间距: {self.min_spacing:.1f} m "
                  f"({self.config.min_spacing_multiple:.1f}倍转子直径)")
            print(f"场地面积: {self.boundary.area / 1e6:.2f} km²")
            print("=" * 35)

        gen_offset = len(state.convergence_history)

        for gen in range(max_gen):
            elite_idx = np.argsort(fitness)[-n_elite:]
            elites = population[elite_idx].copy()

            parents = self._tournament_selection(
                population, fitness, pop_size - n_elite, state.rng
            )

            offspring = np.zeros((pop_size - n_elite, self.n_dim), dtype=np.float64)
            for i in range(0, pop_size - n_elite, 2):
                p1 = parents[i]
                p2 = parents[(i + 1) % (pop_size - n_elite)]
                c1 = self._crossover(p1, p2, state.rng)
                c2 = self._crossover(p2, p1, state.rng)
                offspring[i] = self._mutate(c1, state.rng)
                if i + 1 < pop_size - n_elite:
                    offspring[i + 1] = self._mutate(c2, state.rng)

            for i in range(len(offspring)):
                offspring[i] = self._repair(offspring[i], state.rng)

            population[:n_elite] = elites
            population[n_elite:] = offspring

            fitness = self._evaluate_population(population)

            current_best_idx = int(np.argmax(fitness))
            if fitness[current_best_idx] > state.best_fitness:
                state.best_fitness = float(fitness[current_best_idx])
                # 关键：拷贝，避免后续世代覆写种群时改动已记录的最优位置
                state.best_positions = population[current_best_idx].reshape(
                    self.n_turbines, 2
                ).copy()
                state.best_generation = gen_offset + gen + 1

            # 在世代结束时记录，保证历史末值 == 最终 best_fitness，
            # 且长度恰为 max_gen
            state.convergence_history.append(float(state.best_fitness))
            state.mean_history.append(float(np.mean(fitness)))

            if verbose and (gen % 5 == 0 or gen == max_gen - 1):
                print(
                    f"Gen {gen_offset + gen + 1:3d} | "
                    f"Best: {state.best_fitness/1e3:8.2f} GWh | "
                    f"Mean: {np.mean(fitness)/1e3:8.2f} GWh | "
                    f"Found@Gen {state.best_generation}"
                )

        if verbose:
            print("=" * 35)
            print("优化完成!")
            print(f"最优净AEP: {state.best_fitness/1e3:.2f} GWh")
            print(f"找到最优解的代数: {state.best_generation}")

        rng_state = state.rng.bit_generator.state
        result = OptimizeResult(
            best_positions=state.best_positions.copy(),
            best_fitness=float(state.best_fitness),
            best_generation=state.best_generation,
            convergence_history=state.convergence_history.copy(),
            mean_history=state.mean_history.copy(),
            final_population=population.copy(),
            final_fitness=fitness.copy(),
            continued=continued,
            rng_state=rng_state,
        )
        return result

    def continue_run(
        self,
        state: dict,
        additional_generations: Optional[int] = None,
        verbose: bool = True,
    ) -> OptimizeResult:
        """从已有接续状态继续运行。

        Parameters
        ----------
        state : dict
            :meth:`export_state` 的返回值
        additional_generations : Optional[int]
            追加运行的代数。``None`` 时使用 ``config.max_generations``。
            注意该参数只改变本次追加的代数，不修改配置。
        verbose : bool
            是否打印进度

        Returns
        -------
        OptimizeResult
            覆盖完整累计历史的结果
        """
        original = self.config.max_generations
        if additional_generations is not None:
            if additional_generations <= 0:
                raise ValueError("additional_generations 必须为正整数")
            self.config.max_generations = int(additional_generations)
        try:
            return self.optimize(verbose=verbose, continue_from=state)
        finally:
            self.config.max_generations = original
