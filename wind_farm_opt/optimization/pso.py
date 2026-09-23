"""粒子群优化器。"""

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
class PSOConfig:
    """粒子群算法配置参数。

    Parameters
    ----------
    swarm_size : int
        粒子群大小
    max_iterations : int
        最大迭代次数
    inertia_weight : float
        惯性权重 w
    cognitive_coeff : float
        认知系数 c1
    social_coeff : float
        社会系数 c2
    max_velocity : float
        最大速度（占场地范围的比例）
    min_spacing_multiple : float
        最小间距倍数（相对于转子直径）
    penalty_factor : float
        约束违反惩罚因子
    seed : Optional[int]
        随机种子
    """

    swarm_size: int = 40
    max_iterations: int = 150
    inertia_weight: float = 0.7
    cognitive_coeff: float = 1.49
    social_coeff: float = 1.49
    max_velocity: float = 0.2
    min_spacing_multiple: float = 5.0
    penalty_factor: float = 1e6
    seed: Optional[int] = None


class ParticleSwarmOptimizer:
    """粒子群算法机位优化器。

    运行生命周期
    ------------
    每次调用 :meth:`optimize`（``continue_run=False``，默认）都从完全独立的
    状态开始：随机数发生器按 ``config.seed`` 重新播种，收敛历史、全局最优、
    个体最优等运行状态全部清空，同一实例上的重复运行互不泄漏。固定种子下
    重复调用 ``optimize()`` 得到逐位一致的结果（重放）。

    需要连续试验（在已有粒子群基础上继续迭代）时，显式传入
    ``continue_run=True``：本次运行从上次运行保留的粒子位置、速度、个体
    最优、全局最优与随机数状态接续执行，历史记录追加而非重置。接续运行的
    结果中 ``best_generation`` 与 ``n_generations`` 为累计迭代数，可与
    重放区分。
    """

    def __init__(
        self,
        n_turbines: int,
        rotor_diameters: np.ndarray,
        boundary: SiteBoundary,
        fitness_fn: Callable[[np.ndarray], float],
        config: Optional[PSOConfig] = None,
    ) -> None:
        self.n_turbines = n_turbines
        self.rotor_diameters = np.asarray(rotor_diameters, dtype=np.float64)
        self.boundary = boundary
        self.fitness_fn = fitness_fn
        self.config = config if config is not None else PSOConfig()

        self.min_spacing = compute_min_spacing_from_diameters(
            self.rotor_diameters,
            self.config.min_spacing_multiple,
        )

        self.n_dim = n_turbines * 2
        self.x_range = boundary.x_max - boundary.x_min
        self.y_range = boundary.y_max - boundary.y_min

        self.vel_range = np.zeros(self.n_dim, dtype=np.float64)
        for i in range(self.n_dim):
            self.vel_range[i] = (
                self.x_range if i % 2 == 0 else self.y_range
            ) * self.config.max_velocity

        self.pos_bounds = np.zeros((self.n_dim, 2), dtype=np.float64)
        for i in range(self.n_dim):
            if i % 2 == 0:
                self.pos_bounds[i] = [boundary.x_min, boundary.x_max]
            else:
                self.pos_bounds[i] = [boundary.y_min, boundary.y_max]

        # 单次运行状态（每次普通 optimize() 调用前由 _reset_run_state 重置）
        self.rng = np.random.default_rng(self.config.seed)
        self._reset_run_state()

    def _reset_run_state(self) -> None:
        """重置单次运行的全部状态，使新一次运行从独立状态开始。"""
        self.rng = np.random.default_rng(self.config.seed)

        self._best_global_pos: Optional[np.ndarray] = None
        self._best_global_fitness = -np.inf
        self._best_iteration = 0

        self.convergence_history: list[float] = []
        self.mean_history: list[float] = []

        # 上一次运行保留的粒子群状态（供 continue_run=True 接续使用）
        self._last_positions: Optional[np.ndarray] = None
        self._last_velocities: Optional[np.ndarray] = None
        self._last_personal_best_pos: Optional[np.ndarray] = None
        self._last_personal_best_fitness: Optional[np.ndarray] = None
        self._iterations_completed = 0

    def _initialize_swarm(self, swarm_size: int) -> tuple[np.ndarray, np.ndarray]:
        """初始化粒子群。"""
        positions = np.zeros((swarm_size, self.n_dim), dtype=np.float64)
        velocities = np.zeros((swarm_size, self.n_dim), dtype=np.float64)

        for i in range(swarm_size):
            pos = self._generate_valid_layout()
            positions[i] = pos.flatten()
            velocities[i] = self.rng.uniform(
                -self.vel_range, self.vel_range, self.n_dim
            )

        return positions, velocities

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

    def _evaluate_particles(self, positions: np.ndarray) -> np.ndarray:
        """评估所有粒子的适应度。"""
        swarm_size = positions.shape[0]
        fitness = np.zeros(swarm_size, dtype=np.float64)

        for i in range(swarm_size):
            penalty = self._compute_penalty(positions[i])

            if penalty > 0:
                fitness[i] = -penalty
            else:
                pos_reshaped = positions[i].reshape(self.n_turbines, 2)
                try:
                    fitness[i] = self.fitness_fn(pos_reshaped)
                except Exception:
                    fitness[i] = -self.config.penalty_factor

        return fitness

    def _repair(self, positions_flat: np.ndarray) -> np.ndarray:
        """修复违反约束的粒子。"""
        positions = positions_flat.reshape(self.n_turbines, 2)

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

    def optimize(self, verbose: bool = True, continue_run: bool = False) -> "OptimizeResult":
        """执行优化。

        Parameters
        ----------
        verbose : bool
            是否打印进度信息
        continue_run : bool
            False（默认）：开始一次全新的独立运行，重置历史、全局最优、
            个体最优与随机数状态（按 config.seed 重新播种）。
            True：从上一次运行保留的粒子位置、速度、个体最优、全局最优与
            随机数状态接续迭代，历史记录追加；要求此前已完成至少一次运行。

        Returns
        -------
        OptimizeResult
            优化结果。保证 ``n_generations == len(convergence_history)
            == len(mean_history)``，``best_fitness == convergence_history[-1]``
            且等于最终粒子群的最大适应度，``best_positions`` 为独立副本，
            不会被后续运行或粒子位置数组的写入改动。
        """
        from .ga import OptimizeResult

        if continue_run:
            if self._last_positions is None:
                raise RuntimeError(
                    "没有可接续的运行状态：请先完成一次普通运行 "
                    "(continue_run=False)，再使用 continue_run=True 接续。"
                )
            if self._last_positions.shape[0] != self.config.swarm_size:
                raise ValueError(
                    f"接续运行的粒子群大小 ({self.config.swarm_size}) "
                    f"与上次运行保留的粒子群大小 ({self._last_positions.shape[0]}) "
                    f"不一致；如需修改 swarm_size 请使用普通运行 (continue_run=False)。"
                )
        else:
            self._reset_run_state()

        swarm_size = self.config.swarm_size
        max_iter = self.config.max_iterations

        w = self.config.inertia_weight
        c1 = self.config.cognitive_coeff
        c2 = self.config.social_coeff

        if verbose:
            print(f"\n=== 粒子群优化开始 ===")
            print(f"风机台数: {self.n_turbines}")
            print(f"粒子群大小: {swarm_size}")
            print(f"本次迭代: {max_iter}"
                  + (f"（接续运行，已完成 {self._iterations_completed} 次迭代）"
                     if continue_run else ""))
            print(f"最小间距: {self.min_spacing:.1f} m "
                  f"({self.config.min_spacing_multiple:.1f}倍转子直径)")
            print(f"w={w}, c1={c1}, c2={c2}")
            print("=" * 35)

        if continue_run:
            positions = self._last_positions.copy()
            velocities = self._last_velocities.copy()
            best_personal_pos = self._last_personal_best_pos.copy()
            best_personal_fitness = self._last_personal_best_fitness.copy()
            fitness = self._evaluate_particles(positions)
        else:
            positions, velocities = self._initialize_swarm(swarm_size)
            fitness = self._evaluate_particles(positions)

            best_personal_pos = positions.copy()
            best_personal_fitness = fitness.copy()

            best_global_idx = np.argmax(fitness)
            self._best_global_pos = positions[best_global_idx].reshape(
                self.n_turbines, 2
            ).copy()
            self._best_global_fitness = float(fitness[best_global_idx])
            self._best_iteration = 0

            # 第 0 次迭代（初始粒子群）的历史记录
            self.convergence_history.append(float(self._best_global_fitness))
            self.mean_history.append(float(np.mean(fitness)))

        for iteration in range(max_iter):
            r1 = self.rng.random((swarm_size, self.n_dim))
            r2 = self.rng.random((swarm_size, self.n_dim))

            best_global_flat = self._best_global_pos.flatten()

            velocities = (
                w * velocities
                + c1 * r1 * (best_personal_pos - positions)
                + c2 * r2 * (best_global_flat - positions)
            )

            velocities = np.clip(velocities, -self.vel_range, self.vel_range)

            positions = positions + velocities

            positions = np.clip(
                positions,
                self.pos_bounds[:, 0],
                self.pos_bounds[:, 1],
            )

            for i in range(swarm_size):
                positions[i] = self._repair(positions[i])

            fitness = self._evaluate_particles(positions)

            improved_mask = fitness > best_personal_fitness
            best_personal_pos[improved_mask] = positions[improved_mask].copy()
            best_personal_fitness[improved_mask] = fitness[improved_mask].copy()

            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > self._best_global_fitness:
                self._best_global_fitness = float(fitness[current_best_idx])
                self._best_global_pos = positions[current_best_idx].reshape(
                    self.n_turbines, 2
                ).copy()
                self._best_iteration = self._iterations_completed + iteration + 1

            self.convergence_history.append(float(self._best_global_fitness))
            self.mean_history.append(float(np.mean(fitness)))

            if verbose and (iteration % 5 == 0 or iteration == max_iter - 1):
                print(
                    f"Iter {self._iterations_completed + iteration + 1:3d} | "
                    f"Best: {self._best_global_fitness/1e3:8.2f} GWh | "
                    f"Mean: {np.mean(fitness)/1e3:8.2f} GWh | "
                    f"Found@Iter {self._best_iteration}"
                )

        self._iterations_completed += max_iter

        # 为可能的接续运行保留粒子群状态（拷贝，保持接续动态与单次长运行
        # 完全一致，故不能在内部状态上做任何替换）
        self._last_positions = positions.copy()
        self._last_velocities = velocities.copy()
        self._last_personal_best_pos = best_personal_pos.copy()
        self._last_personal_best_fitness = best_personal_fitness.copy()

        # 返回用副本：粒子可能已飞离历史最优点，用全局最优位置替换最差
        # 粒子，使 final_population / final_fitness 与 best_positions /
        # best_fitness 严格一致（仅影响返回值，不影响接续运行的动态）
        result_positions = positions.copy()
        result_fitness = fitness.copy()
        if self._best_global_fitness > float(np.max(result_fitness)):
            worst_idx = int(np.argmin(result_fitness))
            result_positions[worst_idx] = self._best_global_pos.flatten()
            result_fitness[worst_idx] = self._best_global_fitness

        if verbose:
            print("=" * 35)
            print(f"优化完成!")
            print(f"最优净AEP: {self._best_global_fitness/1e3:.2f} GWh")
            print(f"找到最优解的迭代: {self._best_iteration}")

        result = OptimizeResult(
            best_positions=self._best_global_pos.copy(),
            best_fitness=float(self._best_global_fitness),
            best_generation=self._best_iteration,
            convergence_history=self.convergence_history.copy(),
            mean_history=self.mean_history.copy(),
            final_population=result_positions,
            final_fitness=result_fitness,
            n_generations=self._iterations_completed,
        )
        self._check_result_consistency(result)
        return result

    @staticmethod
    def _check_result_consistency(result: "OptimizeResult") -> None:
        """校验结果内部一致性（迭代数、历史长度、最终粒子群与最优适应度）。"""
        assert len(result.convergence_history) == result.n_generations + 1, (
            f"收敛历史长度 ({len(result.convergence_history)}) "
            f"与迭代数 ({result.n_generations}) 不一致"
        )
        assert len(result.mean_history) == result.n_generations + 1, (
            f"平均适应度历史长度 ({len(result.mean_history)}) "
            f"与迭代数 ({result.n_generations}) 不一致"
        )
        assert result.convergence_history[-1] == result.best_fitness, (
            f"收敛历史末位 ({result.convergence_history[-1]}) "
            f"与最优适应度 ({result.best_fitness}) 不一致"
        )
        assert 0 <= result.best_generation <= result.n_generations, (
            f"最优解迭代 ({result.best_generation}) 超出 [0, {result.n_generations}]"
        )
        assert result.final_population.shape[0] == result.final_fitness.shape[0], (
            "最终粒子群与最终适应度数量不一致"
        )
        assert np.isclose(
            result.best_fitness, np.max(result.final_fitness), rtol=1e-9, atol=1e-9
        ), (
            f"最优适应度 ({result.best_fitness}) 与最终粒子群最大适应度 "
            f"({np.max(result.final_fitness)}) 不一致"
        )
