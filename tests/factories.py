"""GA / PSO 优化器回归测试的共享构造工具。

两个优化器面对完全相同的确定性优化问题，因此可以验证它们具有一致的
"单次运行生命周期" 语义：普通执行独立重放、接续执行显式可验证。
"""

import numpy as np

from wind_farm_opt.constraints.boundary import create_rectangular_boundary
from wind_farm_opt.optimization.ga import GAConfig, GeneticAlgorithm
from wind_farm_opt.optimization.pso import PSOConfig, ParticleSwarmOptimizer


N_TURBINES = 5
ROTOR_DIAMETER = 126.0
SEED = 42


def make_fitness():
    """确定性的纯函数适应度：距固定目标布局越近越好（越大越好）。

    不修改入参、不依赖任何外部状态，因此固定种子下的重放必须逐位一致。
    """
    target = np.array(
        [
            [-800.0, -800.0],
            [800.0, -800.0],
            [-800.0, 800.0],
            [800.0, 800.0],
            [0.0, 0.0],
        ]
    )

    def fitness_fn(positions: np.ndarray) -> float:
        return -float(np.sum((positions - target) ** 2))

    return fitness_fn, target


def make_boundary():
    return create_rectangular_boundary(width=4000.0, height=4000.0)


def make_rotor_diameters():
    return np.full(N_TURBINES, ROTOR_DIAMETER)


def make_ga(max_generations: int = 5, population_size: int = 12, seed: int = SEED):
    fitness_fn, target = make_fitness()
    config = GAConfig(
        population_size=population_size,
        max_generations=max_generations,
        min_spacing_multiple=5.0,
        seed=seed,
    )
    optimizer = GeneticAlgorithm(
        n_turbines=N_TURBINES,
        rotor_diameters=make_rotor_diameters(),
        boundary=make_boundary(),
        fitness_fn=fitness_fn,
        config=config,
    )
    return optimizer, config, fitness_fn, target


def make_pso(max_iterations: int = 6, swarm_size: int = 12, seed: int = SEED):
    fitness_fn, target = make_fitness()
    config = PSOConfig(
        swarm_size=swarm_size,
        max_iterations=max_iterations,
        min_spacing_multiple=5.0,
        seed=seed,
    )
    optimizer = ParticleSwarmOptimizer(
        n_turbines=N_TURBINES,
        rotor_diameters=make_rotor_diameters(),
        boundary=make_boundary(),
        fitness_fn=fitness_fn,
        config=config,
    )
    return optimizer, config, fitness_fn, target
