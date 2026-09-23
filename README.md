# 风电场布局优化工具

这个项目用于估算风电场的年发电量，并比较不同风机布局和尾流模型的结果。项目包含风机与风资源模型、场地边界和间距约束、遗传算法与粒子群优化、经济性分析以及无界面图表输出。

## 安装

建议使用 Python 3.10 或更新版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 可以使用 `.venv\\Scripts\\Activate.ps1` 激活环境。

## 快速验证

```bash
python quick_test.py
```

快速验证会覆盖模型、约束、年发电量、优化、经济性和图表生成，并在 `test_output/` 写入临时图片。该目录不会纳入版本控制。

## 回归测试

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

`tests/test_optimizer_lifecycle.py` 约束 GA 与 PSO 的单次运行生命周期：
同一实例重复 `optimize()` 互不泄漏（历史不翻倍、全局/个体最佳与随机数
状态独立）、固定种子可逐位重放、`export_state()`/`continue_run()` 的
显式接续与一次性长运行逐位等价、返回的最优位置不被种群数组后续写入
改动，且代数、历史长度、最终种群与最佳适应度彼此一致。

## 优化器运行模型

GA 与 PSO 的实例只保存问题定义与配置，不保存任何一次运行的易变状态。
每次 `optimize()` 都是一次全新的独立运行：以 `config.seed` 重新播种
随机数、清空历史、重置全局/个体最佳。返回的 `OptimizeResult` 中，
`generations == len(convergence_history) == len(mean_history)`，
`convergence_history[-1] == best_fitness`，所有数组均为独立拷贝。

需要把一次长运算拆成多段时，使用显式且可校验的接续方式：

```python
result = optimizer.optimize()                 # continued=False
state = optimizer.export_state(result)        # 带标签/形状/长度校验，可 pickle
more = optimizer.continue_run(state, 50)      # continued=True，累计代数与历史
```

接续状态带有算法标签（GA/PSO 不可互换）与结构校验；篡改或跨算法使用
会抛出 `ValueError`。接续之后再次普通调用 `optimize()` 仍然从零开始。

## 完整分析

```bash
python -m wind_farm_opt --help
python -m wind_farm_opt --n-turbines 15 --iterations 100 --population 50 --output-dir output
```

也可以先生成配置文件，再通过 `--config` 运行：

```bash
python -m wind_farm_opt --generate-config my_config.json
python -m wind_farm_opt --config my_config.json
```

所有运行结果默认写入 `output/`，可以用 `--no-plots` 跳过图表生成。命令行使用无界面绘图后端，适合容器和服务器环境。
