"""粒子群优化器。

单次运行生命周期
------------------
``ParticleSwarmOptimizer`` 实例本身不携带任何一次运行的易变状态：速度、
个体最佳、全局最佳、历史记录与随机数流都由 :meth:`optimize` 在启动时
新建、在结束时丢弃。同一实例连续调用两次 ``optimize``，第二次结果与
新建实例后运行完全一致，不会混入上一次运行的全局最佳或个体最佳。

需要把一次长运算拆成多段连续执行时，使用显式的 :meth:`export_state` /
:meth:`continue_run`（或 :meth:`optimize` 的 ``continue_from`` 参数）。
接续状态带有迭代计数与校验信息，普通执行与接续执行可以明确区分。
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
from .ga import OptimizeResult


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
        随机种子。每次 ``optimize`` 都会以该种子重新播种，因此固定种子
        的普通执行可精确重放；只有显式接续（``continue_from``）才会
        沿用上次运行结束时的随机数状态。
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


class _RunState:
    """一次运行的易变状态。仅存在于 ``optimize`` 执行期间。"""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.best_global_pos: Optional[np.ndarray] = None
        self.best_global_fitness: float = -np.inf
        self.best_iteration: int = 0
        self.convergence_history: list = []
        self.mean_history: list = []


class ParticleSwarmOptimizer:
    """粒子群算法机位优化器。

    优化器实例只保存问题定义与配置；每次 :meth:`optimize` 都从独立的
    随机数流、空历史与全新的粒子状态开始。
    """

    _STATE_TAG = "pso-run-state-v1"

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

    def _new_rng(self) -> np.random.Generator:
        """为一次普通运行建立独立的随机数流。"""
        return np.random.default_rng(self.config.seed)

    def _initialize_swarm(
        self, swarm_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """初始化粒子群。"""
        positions = np.zeros((swarm_size, self.n_dim), dtype=np.float64)
        velocities = np.zeros((swarm_size, self.n_dim), dtype=np.float64)

        for i in range(swarm_size):
            pos = self._generate_valid_layout(rng)
            positions[i] = pos.flatten()
            velocities[i] = rng.uniform(
                -self.vel_range, self.vel_range, self.n_dim
            )

        return positions, velocities

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

    def _repair(
        self, positions_flat: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
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
        return {
            "tag": self._STATE_TAG,
            "positions": np.array(result.final_population, dtype=np.float64, copy=True),
            "velocities": np.array(result.velocities, dtype=np.float64, copy=True),
            "fitness": np.array(result.final_fitness, dtype=np.float64, copy=True),
            "best_personal_pos": np.array(
                result.personal_best_positions, dtype=np.float64, copy=True
            ),
            "best_personal_fitness": np.array(
                result.personal_best_fitness, dtype=np.float64, copy=True
            ),
            "best_global_pos": np.array(result.best_positions, dtype=np.float64, copy=True),
            "best_global_fitness": float(result.best_fitness),
            "best_iteration": int(result.best_generation),
            "iterations": int(result.generations),
            "convergence_history": list(result.convergence_history),
            "mean_history": list(result.mean_history),
            "rng_state": result.rng_state,
        }

    def _validate_state(self, state: dict) -> None:
        """校验接续状态的归属与结构。"""
        if not isinstance(state, dict) or state.get("tag") != self._STATE_TAG:
            raise ValueError(
                f"无效的接续状态：期望标签 {self._STATE_TAG!r}，"
                "请使用同类型优化器的 export_state() 生成"
            )
        required = (
            "positions",
            "velocities",
            "fitness",
            "best_personal_pos",
            "best_personal_fitness",
            "best_global_pos",
            "best_global_fitness",
            "best_iteration",
            "iterations",
            "convergence_history",
            "mean_history",
            "rng_state",
        )
        missing = [k for k in required if k not in state]
        if missing:
            raise ValueError(f"接续状态缺少字段: {missing}")

        shape = (self.config.swarm_size, self.n_dim)
        for key in ("positions", "velocities", "best_personal_pos"):
            if np.asarray(state[key]).shape != shape:
                raise ValueError(f"接续状态的 {key} 形状与当前配置不一致")
        if np.asarray(state["fitness"]).shape != (self.config.swarm_size,):
            raise ValueError("接续状态的适应度形状与粒子群大小不一致")
        if np.asarray(state["best_personal_fitness"]).shape != (
            self.config.swarm_size,
        ):
            raise ValueError("接续状态的个体最佳适应度形状与粒子群大小不一致")
        if len(state["convergence_history"]) != int(state["iterations"]):
            raise ValueError("接续状态的历史长度与已运行迭代数不一致")

    def _validate_result(self, result: OptimizeResult) -> None:
        """校验返回结果内部各字段彼此一致。"""
        if not isinstance(result, OptimizeResult):
            raise ValueError("需要 OptimizeResult（optimize 的返回值）")
        if result.generations != len(result.convergence_history):
            raise ValueError("结果的 generations 与收敛历史长度不一致")
        if len(result.mean_history) != result.generations:
            raise ValueError("结果的平均适应度历史长度与迭代数不一致")
        if np.asarray(result.best_positions).shape != (self.n_turbines, 2):
            raise ValueError("结果的最优位置形状不正确")
        shape = (self.config.swarm_size, self.n_dim)
        if np.asarray(result.final_population).shape != shape:
            raise ValueError("结果的最终粒子位置形状与当前配置不一致")
        if np.asarray(result.final_fitness).shape != (self.config.swarm_size,):
            raise ValueError("结果的最终适应度形状与粒子群大小不一致")
        if result.rng_state is None:
            raise ValueError("结果缺少随机数状态，无法接续")
        if result.velocities is None or np.asarray(result.velocities).shape != shape:
            raise ValueError("结果缺少最终速度，无法接续")
        if (
            result.personal_best_positions is None
            or np.asarray(result.personal_best_positions).shape != shape
        ):
            raise ValueError("结果缺少个体最佳位置，无法接续")
        if (
            result.personal_best_fitness is None
            or np.asarray(result.personal_best_fitness).shape != (self.config.swarm_size,)
        ):
            raise ValueError("结果缺少个体最佳适应度，无法接续")

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
            一次全新的独立运行：重新播种随机数、清空历史、重置全局最佳
            与个体最佳；传入状态时则从该状态显式接续，随机数流、粒子
            速度、个体/全局最佳和历史都从上一次结束处继续。

        Returns
        -------
        OptimizeResult
            优化结果（``best_generation`` 字段对 PSO 表示找到全局最佳的
            迭代次数，接续时按累计迭代计数）
        """
        swarm_size = self.config.swarm_size
        max_iter = self.config.max_iterations

        w = self.config.inertia_weight
        c1 = self.config.cognitive_coeff
        c2 = self.config.social_coeff

        if continue_from is None:
            # —— 全新独立运行：全新 RNG、空历史、全新的粒子状态 ——
            state = _RunState(self._new_rng())
            positions, velocities = self._initialize_swarm(swarm_size, state.rng)
            fitness = self._evaluate_particles(positions)

            best_personal_pos = positions.copy()
            best_personal_fitness = fitness.copy()

            best_global_idx = int(np.argmax(fitness))
            # .copy()：全局最佳是独立副本，粒子数组后续写入不会改动它
            state.best_global_pos = positions[best_global_idx].reshape(
                self.n_turbines, 2
            ).copy()
            state.best_global_fitness = float(fitness[best_global_idx])
            state.best_iteration = 0
            continued = False
        else:
            # —— 显式接续：恢复上次结束时的全部状态 ——
            self._validate_state(continue_from)
            rng = np.random.default_rng()
            rng.bit_generator.state = continue_from["rng_state"]
            state = _RunState(rng)
            positions = np.array(continue_from["positions"], copy=True)
            velocities = np.array(continue_from["velocities"], copy=True)
            fitness = np.array(continue_from["fitness"], copy=True)
            best_personal_pos = np.array(
                continue_from["best_personal_pos"], copy=True
            )
            best_personal_fitness = np.array(
                continue_from["best_personal_fitness"], copy=True
            )
            state.best_global_pos = np.array(
                continue_from["best_global_pos"], dtype=np.float64, copy=True
            )
            state.best_global_fitness = float(continue_from["best_global_fitness"])
            state.best_iteration = int(continue_from["best_iteration"])
            state.convergence_history = list(continue_from["convergence_history"])
            state.mean_history = list(continue_from["mean_history"])
            continued = True

        if verbose:
            print(f"\n=== 粒子群优化{'（接续）' if continued else ''}开始 ===")
            print(f"风机台数: {self.n_turbines}")
            print(f"粒子群大小: {swarm_size}")
            print(f"本次运行迭代: {max_iter}")
            print(
                f"累计迭代: {len(state.convergence_history)} -> "
                f"{len(state.convergence_history) + max_iter}"
            )
            print(f"最小间距: {self.min_spacing:.1f} m "
                  f"({self.config.min_spacing_multiple:.1f}倍转子直径)")
            print(f"w={w}, c1={c1}, c2={c2}")
            print("=" * 35)

        iter_offset = len(state.convergence_history)

        for iteration in range(max_iter):
            r1 = state.rng.random((swarm_size, self.n_dim))
            r2 = state.rng.random((swarm_size, self.n_dim))

            best_global_flat = state.best_global_pos.flatten()

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
                positions[i] = self._repair(positions[i], state.rng)

            fitness = self._evaluate_particles(positions)

            improved_mask = fitness > best_personal_fitness
            best_personal_pos[improved_mask] = positions[improved_mask].copy()
            best_personal_fitness[improved_mask] = fitness[improved_mask].copy()

            current_best_idx = int(np.argmax(fitness))
            if fitness[current_best_idx] > state.best_global_fitness:
                state.best_global_fitness = float(fitness[current_best_idx])
                # 关键：拷贝，避免后续迭代覆写粒子位置时改动已记录的全局最佳
                state.best_global_pos = positions[current_best_idx].reshape(
                    self.n_turbines, 2
                ).copy()
                state.best_iteration = iter_offset + iteration + 1

            # 在迭代结束时记录，保证历史末值 == 最终 best_fitness，
            # 且长度恰为 max_iter
            state.convergence_history.append(float(state.best_global_fitness))
            state.mean_history.append(float(np.mean(fitness)))

            if verbose and (iteration % 5 == 0 or iteration == max_iter - 1):
                print(
                    f"Iter {iter_offset + iteration + 1:3d} | "
                    f"Best: {state.best_global_fitness/1e3:8.2f} GWh | "
                    f"Mean: {np.mean(fitness)/1e3:8.2f} GWh | "
                    f"Found@Iter {state.best_iteration}"
                )

        if verbose:
            print("=" * 35)
            print("优化完成!")
            print(f"最优净AEP: {state.best_global_fitness/1e3:.2f} GWh")
            print(f"找到最优解的迭代: {state.best_iteration}")

        rng_state = state.rng.bit_generator.state
        result = OptimizeResult(
            best_positions=state.best_global_pos.copy(),
            best_fitness=float(state.best_global_fitness),
            best_generation=state.best_iteration,
            convergence_history=state.convergence_history.copy(),
            mean_history=state.mean_history.copy(),
            final_population=positions.copy(),
            final_fitness=fitness.copy(),
            continued=continued,
            rng_state=rng_state,
            velocities=velocities.copy(),
            personal_best_positions=best_personal_pos.copy(),
            personal_best_fitness=best_personal_fitness.copy(),
        )
        return result

    def continue_run(
        self,
        state: dict,
        additional_iterations: Optional[int] = None,
        verbose: bool = True,
    ) -> OptimizeResult:
        """从已有接续状态继续运行。

        Parameters
        ----------
        state : dict
            :meth:`export_state` 的返回值
        additional_iterations : Optional[int]
            追加运行的迭代数。``None`` 时使用 ``config.max_iterations``。
            注意该参数只改变本次追加的迭代数，不修改配置。
        verbose : bool
            是否打印进度

        Returns
        -------
        OptimizeResult
            覆盖完整累计历史的结果
        """
        original = self.config.max_iterations
        if additional_iterations is not None:
            if additional_iterations <= 0:
                raise ValueError("additional_iterations 必须为正整数")
            self.config.max_iterations = int(additional_iterations)
        try:
            return self.optimize(verbose=verbose, continue_from=state)
        finally:
            self.config.max_iterations = original
