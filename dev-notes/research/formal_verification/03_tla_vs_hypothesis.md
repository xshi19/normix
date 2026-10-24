# normix 的软件架构：TLA+ 与 Hypothesis 各能检查什么

写给 Xiang Shi · 2026-10-24

建议先给现有代码补一小组 Hypothesis 性质测试。TLA+ 值得学，但目前最合适的试点是文档发布协议。分布对象、参数转换、Bessel 内核和大部分拟合代码，不需要先建 TLA+ 模型。原因是它们的主要风险在数值、参数域和接口约定；独立模型会省略这些风险，又增加一份需要同步的描述。

这不等于“数学用 Lean，应用一律不用形式化”。应用中若有崩溃恢复、重复请求、并发写入或必须最终完成的工作，TLA+ 就有用。normix 已有增量 EM，但尚不能据此把它当成带持久化、消息确认和恢复协议的在线服务。

本文只做静态阅读。没有联网、安装依赖、运行 Python 测试、Lean 或 TLC。所有代码块都是**示意、未经运行验证**。只新增本报告；不改源码、测试、索引、配置或设计决策表。

证据分三类：

- **本地源码事实**：核对了 normix `51e053c57237a29c71b56d3a2a86988277477bd9` 的文件。读到测试源码，只说明测试已写，不能说明本次通过。
- **输入材料的说法**：[overview_report.md](00_overview_report.md) 是起点。“完整 CPU 测试约 71 分钟”和 jax-lean 有 12 次提交来自任务输入，本次未复测或核对提交数量。
- **文献与工具记忆**：下文 TLA+、TLC、PlusCal、Hypothesis 及替代工具的通用介绍来自既有知识，未在此核对外部文档。具体版本行为标作 `to verify`。数学测试条件是本文推导或仓库已有公式，不冒充已经机器证明的结论。

文中 CI 指持续集成，即提交代码后自动运行检查；PR 指请求合并代码的变更。后端是用户选择的计算实现路径，例如 `'jax'` 或 `'cpu'`。自动微分简称 AD，是按运算规则计算导数。下文提到的 bundle 只是把一次计算产生的多个矩和导数装在一起的结果对象。

## 1. 从零理解三种检查

### 1.1 TLA+：把允许发生的步骤写清楚

TLA+ 是描述系统行为的规约语言。**规约**就是精确写下系统允许做什么、必须保证什么。它以集合、函数和逻辑表达式描述**状态机**：一组状态，加上状态之间允许的转换。状态不一定是数据库，也可以是一次拟合的阶段、迭代次数和停止标志。

以文档发布为例。状态可以是“旧构建已完成、新构建已完成、网站当前版本”。两个任务先后完成的顺序不固定。每一种可能顺序都形成一条**执行轨迹**，即状态序列。

TLA+ 中常用三个名字：

- `Init`：哪些初始状态合法。例如两项任务都未开始，网站版本为 0。
- `Next`：一步可以怎样变。例如任意一项任务完成构建，或者把构建产物发布到网站。
- `Spec`：完整行为约束。常见形状是 `Init /\ [][Next]_vars`。`vars` 收集状态变量；`[]` 表示“始终”。方括号允许状态不变的一步，称为 **stuttering**。外部等待不应被误认作非法行为。

变量后的撇号表示下一状态。例如 `version' = 2` 是下一步版本为 2，不是 Python 的赋值语句。`UNCHANGED version` 表示这一步不改版本。`/\`、`\/` 分别表示“且”“或”。

**不变式**是每个可达状态都应成立的断言。例如“显示的版本不低于已经发布过的最高版本”。**安全性**是“坏事永远不发生”，不变式是表达安全性的常用方式。**时序性质**谈整条轨迹；例如“每个已经启动的发布最终完成”，其中“最终”写成 `<>`。这类保证进展的性质叫**活性**。

“始终没发布错”可以靠永远不发布实现。因此安全性不推出活性。要说“最终完成”，通常还要给出**公平性假设**：一个持续可执行的步骤，不能被调度器永远饿死。公平性是模型的前提，不是对 GitHub、操作系统或网络的证明。

**TLC** 是 TLA+ 的模型检查器。把版本数、任务数等限制为有限集合后，它可遍历可达状态和转换，检查不变式，并在支持的范围内分析时序性质。活性检查可分析状态图中的循环，不需要真的运行无限步。失败时给出反例轨迹。完成一次有限模型的穷举检查，比随机试几个调度更强；但两个任务的检查不会自动证明任意任务数都正确。随机模拟模式也不具有这种穷举保证。TLA+ 也有演绎证明路线，本文不把 TLC 的有限检查说成那种证明。

**PlusCal** 是写在 TLA+ 文件中的算法描述语言。外观更像带标签的伪代码，可以翻译成 TLA+。它便于表达循环和多个进程，但不会把 normix 的 Python 自动翻译成模型。标签如何划分原子步骤仍要由建模者判断。**原子步骤**是模型假设不可被别的任务插入的一步；划得太大，就可能藏起竞态。

一个手算反例足够说明用途：

```text
网站版本 0
新任务构建完成 → 新任务发布版本 2
旧任务构建完成 → 旧任务发布版本 1
网站退回版本 1
```

两项构建都成功，也没有数值误差。问题只在允许的顺序。第 5 节把这个例子写成一个完整的小模型。

TLC 检查的是我们写的**模型**。它不读取实际工作流、Python 分支或部署动作的实现。即便模型通过，仍需核对“模型中的一步，对应代码中的哪一步”。把这种对应关系正式证明出来叫**精化证明**；本文没有做。现实多了一种重试路径、模型少了一次崩溃，都可能破坏对应关系。

### 1.2 Hypothesis：生成输入，调用真实代码

Hypothesis 是 Python 的**基于性质的测试**工具。普通测试给定几组输入和答案；性质测试先给出一类输入，以及每次都应成立的关系。工具生成许多具体输入，再运行真正的函数。它不是均匀随机抽样器；会结合生成、边界探索、已有反例和搜索。本文与“有限模型穷举”对比时，将它简称为生成式抽样检查。

三个核心词：

- **性质**：可执行的断言。例如合法 Gamma 参数转成自然参数再转回来，应恢复原值。
- **策略**（strategy）：如何生成输入。例如 `st.floats(0.5, 5.0, allow_nan=False, allow_infinity=False)`。
- **缩减**（shrinking）：失败后继续找更小、更容易解释的失败输入。它不是保证找到全局最小反例。

下面调用仓库中的 `Gamma.natural_params()` 和 `Gamma.from_natural()`。Gamma 在 `normix/distributions/gamma.py` 里用 $\alpha$ 表示形状、$\beta$ 表示率，二者都为正。自然参数为 $\theta=(\alpha-1,-\beta)$。

```python
import numpy as np
from hypothesis import given, strategies as st
from normix import Gamma

positive = st.floats(
    min_value=0.5, max_value=5.0,
    allow_nan=False, allow_infinity=False,
)

@given(alpha=positive, beta=positive)
def test_gamma_natural_roundtrip(alpha: float, beta: float) -> None:
    original = Gamma(alpha=alpha, beta=beta)
    restored = Gamma.from_natural(original.natural_params())
    np.testing.assert_allclose(
        [float(restored.alpha), float(restored.beta)],
        [alpha, beta], rtol=1e-10, atol=1e-12,
    )
```

这里默认已按仓库 `tests/conftest.py` 开启 float64。容差适用于这个远离边界的代数往返示例，不能直接套给 GIG 反解。

若故意把 `from_natural` 中恢复 $\alpha$ 的加 1 删掉，普通正参数就能暴露问题。若正反两个函数被同时改错，同一组往返测试可能仍通过。因此还需要独立公式、SciPy 或高精度参考值。**参考实现**也常叫 oracle，意思是“用来判断答案的依据”，不是绝对可信的程序。

Hypothesis 也能测试操作序列。`RuleBasedStateMachine` 让我们定义若干操作，工具随机排列它们，并在每步后检查不变式。它适合“先更新、再读取、再更新”的真实 API。名字里有 state machine，不表示它会像 TLC 一样穷举所有调度；默认也不会自动制造线程交错。

例如 `SampleWeightedUpdate` 必须按样本数合并批次。批次 `[2]` 与 `[0, 0]` 合并后的均值应是 $2/3$。若误把两个批次均值各乘 $1/2$，会得到 1。第 4 节用这个真实类给出状态测试骨架。

`settings` 控制测试预算。`max_examples` 控制生成样例的预算，不是包含缩减在内的绝对调用上限。`deadline` 是单例时间检查。JAX 首次编译可能远慢于后续运行，建议这些测试用 `deadline=None`，另外用 CI 作业超时和耗时报告控制总预算。`derandomize=True` 便于 CI 重现；它不承诺跨 Hypothesis、JAX 版本和设备得到同一执行轨迹。版本与设置都要记录。

### 1.3 Lean 的位置

Lean 是证明助手：人或 AI 写定义、命题和证明，核心检查器检查推理。Mathlib 是其数学库。它适合表达“对所有满足前提的分布，某恒等式成立”，例如有限矩条件下的条件方差公式。它也能描述程序状态，但需要显式建立代码语义。证明实数公式，不自动证明 JAX 的舍入、编译、设备放置或随机数实现。本文不假定某个 Bessel 定理已经在 Mathlib 中；具体覆盖和引理名均需 `to verify`。TLA+ 的行为模型、Lean 的数学证明和 Hypothesis 对真实代码的测试，可以共享同一条规约，各检查不同部分。

## 2. 比较与选择

下表是工具机制比较，来自文献与工具记忆；不是本次运行结果。

| 项目 | TLA+ 配合 TLC | Hypothesis | Lean，供对照 |
|---|---|---|---|
| 直接检查什么 | 手工或生成的设计模型 | 测试调用到的真实 Python/JAX 代码 | 精确定义上的命题及证明 |
| 保证强度 | 有限范围内完整探索可达行为；受模型、配置和检查器约束 | 有限次生成检查；通过不能推出所有输入正确 | 在给定定义、前提和公理下的演绎结论 |
| 主要缺口 | 模型与实现是否一致；小规模能否推广 | 未抽到的输入、未安排到的操作顺序 | 数学/语义定义与实际运行之间的对应 |
| 擅长发现 | 重复提交、丢更新、错误顺序、死锁、饥饿 | 参数域遗漏、索引/形状错误、数值边界、接口不一致、状态更新错误 | 证明中的逻辑缺口、缺少前提、错误恒等式 |
| 不擅长直接处理 | Bessel 精度、连续参数上的误差界 | 证明任意调度下的活性、全称数学定理 | 无现成语义支持的完整 JAX/XLA 执行链 |
| 入门曲线 | 新的状态/时序思维；语法并非最大难点 | Python 用户较容易开始；策略和容差设计较难 | 需要学习类型、命题、证明和数学库 |
| 小试点成本，本文估计 | 1–2 人日建模，加 1–2 小时业主审模型 | 0.5–1 人日完成首个小组，加约 1 小时审规约 | 小代数命题也应留出学习时间；复杂分析不作几天内完成承诺 |
| 维护负担 | 实现和协议变化都要检查模型是否同步 | 维护策略、参考值、容差、版本与执行时间 | 定义、库版本和依赖证明一起维护 |
| normix 当前适配 | 发布流程较好；其余多为低收益 | 高；与现有 pytest、JAX 和契约测试直接衔接 | 数学层另设范围；不替代代码测试 |

选择规则可以直接用于评审：

1. 能否写成“给函数一个输入，返回值满足什么”？优先普通 pytest 或 Hypothesis。只有两个后端值，穷举两个值的参数化测试通常更直接。
2. 是否取决于历史顺序？先看是否能用真实 API 的短序列覆盖。累计样本数、重复调用和异常后的恢复，通常用状态式性质测试即可。
3. 是否有独立参与者、重试、崩溃、共享资源，或者“最终必须发生”的要求？先考虑 TLA+。它也可用于顺序程序，收益并不限于并发；但迭代本身不是采用它的充分理由。
4. 是否需要对无限参数域证明数学结论？考虑 Lean。先把有限矩、可微性、可积性和求解精度等前提写对。

“架构检查”也不必二选一。模块依赖方向、禁用某个调用、公共导出是否齐全，往往只需确定性的源码检查。jax-lean 自己的 `tests/test_architecture.py` 就通过解析导入关系来守住层次边界，没有给这件事另建 TLA+ 状态机。

## 3. 对照 normix 的现状

以下路径均相对仓库根。具体文件链接集中在末尾 Sources。$\theta$ 表示自然参数，$\eta$ 表示充分统计量的期望，$\psi$ 表示对数配分函数。充分统计量是指数族密度中与参数配对的观测函数。它们都是统计概念，不是三个可变对象之间的同步协议。

### 3.1 后端契约：先检真实分支

`dev-notes/design/design.md` 的 S9、S11 要求尊重显式后端选择。`backend='jax'` 的 GIG 反解不得隐藏主机回调。**主机回调**是 JAX 执行期间转回 Python/CPU 执行另一段函数；结果再放回设备，也不能算原生 JAX 求解。

需要分开检查四件事：数值相近、能否 JIT、能否求导、实际执行路径。`jax.jit` 是编译追踪到的计算；`jax.vmap` 是把单观测函数映射到一批输入。结果是 GPU 数组，不能证明中间未在 CPU 求解。`backend='jax'` 也不是“强制用 GPU”，它应保留调用方选择的 JAX 设备。

真实入口包括：

- `normix/exponential_family.py`：`expectation_params`、`fisher_information`、`from_expectation`。
- `normix/utils/bessel.py`：`log_kv`、`log_kv_moments`。
- `normix/distributions/generalized_inverse_gaussian.py`：`GIG.from_expectation`。
- `normix/mixtures/marginal.py`：`NormalMixture.e_step`；`normix/fitting/em.py` 传递 E/M 两个后端。

CPU 是显式、已有的路径。CPU E-step 仍可用 JAX 处理二次型，GIG/Bessel 部分走 NumPy；不能把契约改成“cpu 字样出现后，全栈禁止 JAX”。同样，不应为了让某个验证工具容易工作而改变用户选定的后端。

已有 `tests/test_no_host_callbacks.py`：一项词法扫描，以及递归进入 `while`、`cond`、`scan`、`jit`、自定义微分函数内部的 jaxpr 检查。**jaxpr** 是 JAX 追踪生成的中间计算表示。应扩展这项检查的冷启动、热启动和批量覆盖，不另写只扫描顶层的弱版本。它检查所追踪形状和静态配置，仍不证明所有设备编译器都正确。

一个可立即记录的缺口：本地 `log_kv_moments` 检查未知后端并抛 `ValueError`；`log_kv` 末尾只有 `'cpu'` 分支和默认 JAX 返回。因此 `log_kv(..., backend='typo')` 按源码会进入 JAX。`GIG.expectation_params_batch` 也有类似“cpu，否则 jax”的分派。前者是公共接口契约的明确核对点；后者还应确认公开范围。**这是静态发现，未执行复现，未修复。** 为后端写一个总会保持常量的 TLA+ 模型，发现不了这种代码缺口。

这里优先用有限枚举、错误参数测试、路径哨兵和 jaxpr 检查。**路径哨兵**是测试中暂时替换某个不应进入的函数，一旦进入就失败。它应在追踪前安装，避免已缓存的 JIT 程序绕过哨兵。没有 GPU 的运行要明确记为缺少加速器证据。

### 3.2 EM：有局部状态，但大多没有外部调度

`BatchEMFitter.fit` 做中心化、选择循环、运行 E/M 步并恢复位置。纯 JAX 的批量路径在双方后端为 JAX、`algorithm='em'`、`verbose<=1` 且无 `eta_update` 时用 `lax.scan`。其他情形用 Python 循环。E 步、M 步、正则化的次序可由 `_em_step`、`_mcecm_step` 直接核对。

`EMResult` 有三个不同标志：

- `converged`：满足批量 EM 的 Aitken 停止准则。Aitken 用连续三个目标值估计剩余改进量；不是到达全局最优的证明。
- `diverged`：正常参数块变化或目标变成非有限值，并保留此前有限模型。
- `solver_failed`：混合分布参数非有限，或 Python 路径捕获相应求解异常。

耗尽预算可以三个都为假。增量 EM 的 `converged` 为 `None`。不能把 `not converged` 当成 `diverged`。`_fit_scan` 在停止后冻结输出状态，但循环体仍计算候选 `_step` 再选择旧值；不应把“逻辑停止后不再改变模型”写成“硬件不再做计算”。

`EStepResult.log_lik` 是 E 步输入模型的每观测平均对数似然。返回模型通常已多做一次 M 步。比较最终似然时要重新计算，不能把 `log_likelihoods[-1]` 当作最终模型的值。Python 失败路径还可能使似然记录少于尝试步数，所以“所有诊断数组长度永远等于 `n_iter`”不是可直接采用的性质。

**热启动**是把已有参数作为求解起点；**冷启动**是重新构造起点。`GeneralizedHyperbolic.m_step_subordinator` 传入当前 GIG 的自然参数作热启动。随机测试应检查这个参数有没有正确传递，不能要求热启动和冷启动逐步走同一条轨迹。

分布是不可变的 `eqx.Module`，但拟合器是普通 Python 对象。`BatchEMFitter.fit` 和 `IncrementalEMFitter.fit` 都会暂时替换 `self.eta_update`，并在 `finally` 恢复。顺序复用、异常后恢复，值得测试。若两个线程共享同一个 fitter，这个字段可能成为交错点；本次没有验证这种用法受支持，也没有复现竞态。先明确对象的共享契约，通常比立刻为整套 EM 建模更便宜。

此外，`_fit_scan` 的计算体可编译，不等于整个 Python `fit` 包装器都可无条件放进 `jax.jit`。返回阶段含 `int(n_iter)`、布尔转换、切片和计时。测试应限定承诺的转换边界。

### 3.3 增量 EM 与未来持久化

`IncrementalEMFitter.fit(model, X, key=...)` 已实现小批量更新。`eta_rules.py` 的规则有显式状态，例如 `SampleWeightedUpdate` 的 `cumulative_n`。JAX `scan` 与 Python 循环使用预先排列的随机子键，可测试同一输入和键下的等价性。

但该入口接收完整数组 `X`，先计算全样本均值，并从数组抽取批次。它不是已经实现了消息队列消费、检查点恢复和“每条数据恰好处理一次”的服务。`docs/theory/online_em.md` 的理论也不构成这样的实现承诺。

若将来增加可恢复流式服务，才值得用 TLA+ 表达：已确认的数据偏移量不能超过已持久化的更新；恢复后模型、累计 $\eta$、步数、规则状态和随机键必须属于同一个检查点。**检查点**是恢复执行所需的一份一致快照。只保存模型而丢掉步长计数或规则状态，不足以复现后续更新。现有全样本中心化还需换成明确的流式约定。这是未来设计问题，不是本报告替仓库添加的功能。

### 3.4 不可变对象、参数布局与派生结果

`ExponentialFamily` 存储各分布的经典参数，`natural_params()` 和 `expectation_params()` 计算另两种表示。`NormalMixture` 持有 joint 对象；`replace` 用 `eqx.tree_at` 返回新对象。**pytree** 是 JAX 将对象拆成数组叶子和结构信息的约定，Equinox 帮这些类接入此约定。

主要风险是参数槽位错、把矩与自然参数混用、转换丢字段、更新误改原对象。这些是代码性质。`NormalMixtureEta` 的六项按理论顺序排列；其中 GIG 三元向量却按 $E[\log Y],E[1/Y],E[Y]$ 排列。检查映射比给变量加“状态”标签有用。

不要给所有 joint 类统一套 $\nabla\psi=\eta$。`docs/design/exponential_family.md` §2.4–2.5 指出，Joint VG、NInvG、NIG 返回更大 GH 统计空间中的矩，受限配分函数的梯度会丢掉固定槽位。它们的矩反解应走 `NormalMixtureEta` 路径。GH 尺度还存在表示不唯一；分布等价不总要求经典参数逐项相同。

这里的“缓存”也要区分：`QuantileTable` 是独立、不可变的 CDF/分位数查表对象；`MeanRiskProblem.__init__` 保存当前模型的分解和场景；JIT 缓存的是编译程序。前两者更像快照。模型改变后应重新构造相关对象。任意用 `eqx.tree_at` 换掉 `MeanRiskProblem.model` 而保留旧分解，不应凭“对象不可变”就视为合法更新协议。

优先检查旧对象未变、重建结果一致、树展开/重组无损。当前没有多写者缓存失效协议，不值得为它维护 TLA+ 模型。

### 3.5 GIG 求解：不要虚构自动回退链

实际分派在 `GIG._from_expectation_unchecked` 和 `fitting/solvers.py`：

| 条件 | 当前源码行为 |
|---|---|
| JAX，给定 `theta0`，Newton，低日志级别 | 调用缓存的 `_gig_jax_newton_jit` |
| JAX，未给 `theta0` | `_jax_cold_start` 用指定方法做有序多起点；找到有限且收敛的结果即可停止 |
| CPU，未给 `theta0` | 多起点 L-BFGS-B；这条冷启动策略不是“Newton 失败后的动作” |
| 其他热启动 | `solve_bregman` 按后端、方法分派；CPU Newton 对应 SciPy `trust-exact` |
| Newton 内部 | 有阻尼、步长搜索等数值备选；仍属该后端的求解过程 |
| GH 的 CPU 热启动抛异常 | `m_step_subordinator` 当前用 `except Exception` 保留前一个 GIG，继续 EM；正常参数块可能已经更新 |
| GH 的有限结果越过允许的参数范围 | 回用前一个 subordinator；非有限 JAX 结果保留给外层记录失败 |

因此不能把现状画成统一的“Newton → L-BFGS → CPU”自动链。多起点求解器在没有成功解时可返回最好的有限候选；GIG 构造入口还会检查是否接受该结果。CPU 的多起点也不是 JAX 那套遇到成功就停止的控制流程。

还有一处需区分文档与代码：E17 和该方法的说明着重列出求解未收敛与 `ValueError`，但当前 `except Exception` 捕获范围更宽。它也可能吞掉实现错误。应由 Xiang 确认预期，再写异常类型的负面测试；本文不把更宽的捕获自动认可为契约，也没有运行复现。

这是有分支的顺序算法，值得枚举成功、未收敛、非有限、越界四类结果，并测试外层处理。把每个数值结果抽象为这些标签可建 TLA+ 模型，但模型不会发现 Hessian 错、容差错或未尊重 `backend` 的真实调用。当前更适合小型可控求解问题、失败注入和性质测试。

### 3.6 finance、公共 API 与编译边界

`finance/` 当前是风险计算和优化，没有读到券商下单、订单确认、共享仓位账本或并发交易服务。应检验约束和数学变换，不能只说“回测够了”。

- `MeanRiskProblem.weights` 应在可达目标上满足预算和两个投影约束。构造阶段做秩判断，含主机读取；不能宣称整个构造器都可追踪。对不可达目标的异常检查也有直接执行（eager）与追踪执行（traced）的差异，应单列。追踪时 JAX 用代表数组的对象记录计算，不能随意把它读成 Python 数值。
- `CVaR.value_w`、`value_grad_hess_w` 应使用同一场景测度；正比例缩放持仓时检查风险的一次齐次性。CVaR 是按约定尾部计算的平均损失，符号应以仓库定义为准。有限矩和固定场景是测试前提。
- `TransactionCostProblem.solve` 调 `solve_transaction_cost_qp`，后者显式使用 SciPy SLSQP。其保证是局部二次近似目标不劣于持仓不动，不是保证真实风险目标全局改进。这是已有 CPU 优化边界，不应为了形式化改写整个包。
- `DiscreteMeasure.from_weights` 将正节点和非负权重整理成离散概率测度。测度就是节点及其概率；`tail_index` 记录原分布的矩存在阈值。有限节点不能用来掩盖原分布的发散矩。

公共 API 还需检查单观测输入、批量输出形状、空样本、非法后端和种子复现。值相等、`jit`、`vmap`、自动微分四类检查分开记录。某些主机校验在追踪时跳过，不能要求两种执行模式抛同一种 Python 异常。

### 3.7 发布与多 agent 工作流

真实并发候选在 `.github/workflows/`。`release-please.yml` 创建 release 后显式分发 `publish.yml` 和 `docs-full.yml`，使用 release tag。`publish.yml`、`docs-full.yml` 也声明 tag push 和手动触发。仅凭 YAML，不能断言同一 release 必然重复触发；事件是否再次触发还受 token 和平台规则影响，这些未联网验证。

`docs.yml` 的并发组按 workflow/ref 区分，允许取消同组旧任务。`docs-full.yml` 用 `pages-release`，不取消进行中的任务。二者都写 `gh-pages`。仓库内未见这两组共享一把部署锁或共同的版本接受规则。这里有“谁最后写网站”的实际协议问题。究竟希望网站追随 master 还是最新 release，必须先决定；二者不能被随意排成一个“更新”序列。

CI、release 和 publish 之间是否受分支保护、环境审批、必需检查约束，不能仅从这些文件推断。报告没有认定线上发布出错，也没有替这些平台设置作证明。

`dev-notes/plans/loops_and_orchestration.md` 把夜间 benchmark、PR 触发审查和其他自动循环列为尚未实现的计划。它提出“同一循环最多一个未结事项”等约束。未来若多个 agent 会重试创建事项、读取旧 commit 的评测或合并结果，TLA+ 可检查重复工作和过期结果被接受。当前先按工作路径隔离、记录 commit、确定所有权就够；不能把计划写成已运行的多 agent 平台。

### 3.8 各处投入是否值得

以下是未来工作的工程估计，1 人日按约 6 小时集中工作计算。时间不代表已发生的验证。

| 范围 | 状态/并发复杂度 | 当前选择 | AI 可做；Xiang 必须审 | 首轮投入 |
|---|---|---|---|---|
| 后端、jaxpr、非法配置 | 少量分派；无并发协议 | 确定性契约测试，加少量生成输入 | AI 扩已有检查；Xiang 审 S9/S11 的执行承诺 | 0.5–1 日；审 0.5–1 小时 |
| 参数、pytree、快照 | 主要是纯函数 | Hypothesis | AI 写策略、独立对照；Xiang 审参数域和表示等价 | 1–2 日；审 1–2 小时 |
| 批量 EM、GIG 失败处理 | 有局部状态；路径有限 | pytest 枚举和状态测试 | AI 注入失败、比路径；Xiang 审停止/回退含义 | 1–2 日；审 1–2 小时 |
| 增量规则 | 有序累计；当前无持久化协议 | `RuleBasedStateMachine` | AI 测计数、权重、键复现；Xiang 审统计权重 | 0.5–1 日；审约 1 小时 |
| finance | 数值约束为主 | 生成可行问题，检查残差 | AI 写场景与变换测试；Xiang 审风险符号、矩域和目标 | 1–2 日；审 1–2 小时 |
| 文档发布 | 独立任务共写目标 | 先审政策；值得一次小 TLA+ 试验 | AI 写状态图、反例；Xiang 定“哪个版本应胜出” | 1–2 日；审 1–2 小时 |
| 可恢复流式 EM | 未来可能有重试/崩溃 | 等真实需求后再建模 | AI 建恢复模型；Xiang 定偏移量、快照、原子性 | 单写者初模约 2–4 日；审半日；实现另计 |
| 多 agent 自动循环 | 目前主要是计划 | 暂不建模 | AI 可列共享资源；Xiang 决定是否建设平台 | 需求出现后先用半日界定；完整估计待定 |

## 4. 一份可落地的 Hypothesis 计划

### 4.1 先定义输入域，避免伪性质

`pyproject.toml` 当前没有 Hypothesis。未来实施时，把它作为开发依赖通过 uv 管理，不加入运行时依赖。本次不安装、不改锁文件。

初期策略从有效分布产生目标矩。不要均匀生成三个实数就称为 GIG 的 $\eta$。GIG 还要求 $E[X]E[1/X]>1$、$-\log E[1/X]<E[\log X]<\log E[X]$，这些必要条件也不是“任意这种三元组都可无误反解”的证明。先生成温和的 $(p,a,b)$，计算矩，再测试反解；非法矩另设负面测试。

正参数可在对数尺度生成，例如先取 $u\in[-3,3]$，再取 $a=e^u$。协方差可由有界矩阵 $A$ 构造 $AA^\top+\lambda I$，其中 $\lambda>0$。这能保证正定，但还要限制条件数，避免把第一批测试全部变成病态矩阵实验。固定几种维数，用值变化探索参数。不要大量 `assume` 过滤无效样本；否则生成器忙于丢弃，错误报告也难解释。

**容差**必须分开来源：浮点舍入、数值反解残差、求积截断、查表插值、随机采样误差。先要求本应有限的量有限，再用显式 `rtol`、`atol` 比较；不让“两边都是 NaN”算通过。预期发散的矩或 NaN 导数另立性质。用更宽容差掩盖失败，不算修复。

### 4.2 25 个候选性质

这是候选清单，不建议一次加入 25 组全量随机测试。优先扩展已有文件，保留固定回归样例。下表中的“快”表示计划进入 PR 默认集合，尚需测时确认；“慢”用现有 `slow` 标记；宽域和大样本用 `stress`；真实数据工作流用 `integration`；真实加速器执行用 `gpu`。重要性另用 `contract` 表示。

| # | 精确入口与应检查的性质 | 前提、参考与执行层级 |
|---|---|---|
| 1 | `Gamma`、`InverseGamma`、`InverseGaussian`、`GIG` 的 `natural_params` → `from_natural` 恢复经典参数 | 内部合法参数，远离构造器裁剪边界；独立核对参数公式。快，`contract` |
| 2 | 上述类的 `expectation_params` → `from_expectation` 恢复 $\theta$，并复算 $\eta$ 残差 | 从有效模型生成；GIG 分冷/热启动，热起点适度扰动，不能只拿精确解作起点。温和 Gamma/IG 快；GIG 先慢 |
| 3 | `ExponentialFamily.expectation_params`、`fisher_information` 与对应类 `_log_partition_from_theta` 的梯度/Hessian 一致 | 限 Gamma、IG、内部 GIG 等适用的完整指数族；不套给受限 joint VG/NInvG/NIG。矩阵对称和半正定用尺度相关容差。GIG 慢 |
| 4 | `NormalMixture.compute_eta_from_model` → `m_step_normal` 恢复正常参数块；`NormalMixture.from_expectation` 的 pytree 路径恢复同一分布 | 用 VG/NInvG/NIG 的有限所需矩区域；GH 的尺度自由度以密度/矩比较。查六字段到 GIG 三字段的顺序。快或慢，按族拆开 |
| 5 | `NormalMixture.replace` 后原对象的所有数组叶子保持不变；新对象只改变请求的参数和相应派生值 | 包含 `sigma` 到 `L_Sigma` 的转换、subordinator 字段；保存值快照，不只比对象身份。快 |
| 6 | `GIG.quantile_table`、`InverseGaussian.quantile_table`、`QuantileTable.cdf/ppf/rvs` 的重复调用与独立重建一致 | 查表对象是固定模型的快照。模型更新后重建；不能期待旧表自行更新。还需独立密度/CDF 对照。构表慢，复用查询快 |
| 7 | `log_kv(v,z)` 满足 $\log K_{-v}(z)=\log K_v(z)$ | $z>0$；先用温和范围，两后端分别检查。已有 Bessel 契约上增加输入生成。快 |
| 8 | `log_kv` 满足 $K_{v+1}(z)=K_{v-1}(z)+(2v/z)K_v(z)$ | 第一批限制 $v>0$，用 log-sum-exp 比较两边，避免先指数化造成溢出；$v=0$ 单独测。宽域 `stress` |
| 9 | `log_kv_moments` 的值和一、二阶导数与 mpmath 参考一致；`jax.grad/hessian(log_kv)` 与 bundle 一致 | PR 复用 `tests/data/bessel_reference.json`；以后夜间才生成新高精度点，参考精度加倍自检。沿用 `test_bessel_contract.py` 按分量容差，不能只用全局一个阈值。新点慢 |
| 10 | `Gamma.pdf`、`InverseGamma.pdf`、`GIG.pdf` 在足够宽、足够细的网格上积分近似 1 | 正半轴可用 $x=e^u$，积分项为 $f(e^u)e^u$。比较加宽、加密结果，并有独立尾界或参考积分；不把已归一化的 PINV 表当归一化证据。慢 |
| 11 | `Gamma.cdf`、`InverseGaussian.cdf`、`GIG.cdf` 单调且在 $[0,1]$；对应 `ppf` 在内部概率处近似反解 | 用排序后的点；CDF 外部支撑单独测。数值查表的 `cdf(ppf(q))` 是内部一致性，不是精度证明；需 SciPy/独立参考。解析族快，GIG 慢 |
| 12 | `Gamma.rvs`、`GIG.rvs`、`NormalMixture.rvs` 的样本一、二阶矩与 `mean/var/cov` 相符 | 限所需高阶矩存在且尾部适中的参数；用下节统计误差预算，不用统一 5%。固定样本长度、记录种子。`slow, stress` |
| 13 | `Gamma.rvs`、`GIG.rvs`、`JointNormalMixture.rvs`、`NormalMixture.rvs` 的形状和同种子复现 | 固定同版本、方法、设备；正支撑样本为正；joint 返回 `(X,Y)`。不同种子“必须不同”不是严格保证，更不能推出独立性。小样本快 |
| 14 | `log_kv/log_kv_moments`、`GIG.expectation_params/fisher_information`、`NormalMixture.e_step` 的 cpu/jax 值相符 | 同一合法参数和 dtype；分开比较 $\eta$、似然及导数。CPU/JAX 共用算法，不能互为唯一精度依据。内核快，完整 E 步慢 |
| 15 | `GIG.from_expectation`、`ExponentialFamily.expectation_params`、`BatchEMFitter.__init__` 和 Bessel 两入口尊重后端并拒绝非法名称 | 两种有效值穷举；路径哨兵检查路由。`log_kv` 的非法名称项预计会暴露 §3.1 的缺口，状态标记“待复现”，不能填已通过。快 |
| 16 | `GIG.from_expectation(backend='jax')` 的冷/热启动及 `vmap` jaxpr 无主机回调 | 复用 `test_no_host_callbacks.py` 的递归遍历；确认求解循环确实出现在图中，防止空壳通过。按静态配置检查一次即可，不对同形状重复 200 次。`contract` |
| 17 | `Gamma.log_prob`、`GIG.from_expectation`、`NormalMixtureEta` 在承诺范围内的 eager/jit/vmap 结果一致，树展开再重组保持值和结构 | 单观测 `log_prob` 用 `vmap` 批量。自动微分另查 S11，不强迫 CPU 构造器可微。小模型快；GIG、真实设备慢或 `gpu` |
| 18 | `NormalMixture.e_step` → `m_step`，以及 `BatchEMFitter.fit` 的普通 EM 目标不降，允许明确误差 | 固定数据、无 shrinkage、无有效边界裁剪、足够准确的 M 步；先 NIG，再 GH。检查每次实际更新的模型，不错位使用 E 步轨迹。`slow` |
| 19 | `BatchEMFitter.fit` 的停止、失败、预算耗尽与 `EMResult` 一致；异常后 fitter 的 `eta_update` 恢复 | 用可控 `_step` 或测试替身注入有限、非有限和异常输出，再用小真实模型对照。检查停止后模型/计数冻结；不要求停止后没有计算。快的控制逻辑，慢的集成 |
| 20 | `SampleWeightedUpdate.__call__` 按累计样本数合并；`IncrementalEMFitter.fit` 的 scan/Python 路径在固定键下相符 | 首先测纯规则的批次序列；它不意味着改变 EM 的批次划分仍产生同一模型。完整增量路径单列慢测，`converged is None` |
| 21 | `affine_combine` 的端点权重保持相应统计量；`Shrinkage(base, eta0, tau=0)` 等于 base；标量与各字段同值的 $\tau$ 路径一致 | 非负有效权重、有限矩；比较返回的规则状态，不能只比 $\eta$。使用非零统计量防止退化通过。快 |
| 22 | `NormalMixture.regularize_det_sigma`、`GeneralizedHyperbolic.regularize_a_eq_b` 保持边缘密度/矩，且重复应用幂等 | 幂等即第二次应用不再改变结果。限制缩放不触发 `GIG_CLAMP` 等裁剪；检目标行列式或 $a=b$。不声称固定潜变量坐标下 joint 密度逐点不变。快/慢按族拆 |
| 23 | `DiscreteMeasure.from_weights` 的概率非负、和为 1；正倍缩放权重不改测度；节点与权重共同置换不改期望 | 禁止全零权重；零权重允许。`coerce` 保持已有测度及 `tail_index`；GIG `quadrature` 不丢原分布尾指数。小测度快 |
| 24 | `MeanRiskProblem.weights` 满足 $\mathbf1^\top w=1$、$\mu^\top w=\tilde\mu$、$\gamma^\top w=\tilde\gamma$；`NormalMixture.project(w)` 的均值/方差等于投影公式 | 从可达目标生成，覆盖秩 1/2/3；另测 eager 不可达错误。$w$ 是持仓，$\tilde\mu,\tilde\gamma$ 是约化目标。复用固定小场景，快或慢 |
| 25 | `TransactionCostQP.weights_from_v` 保持自融资预算；`solve_transaction_cost_qp` 的返回满足约束，近似目标不低于 `hold_objective` | 生成保证不动组合可行的小问题。失败应显式抛异常；检查原始约束残差。不能要求真实目标必改善，也不能用不可行不动组合兜底。`slow` |

`CVaR.value_grad_hess_w` 的分量一致性与正齐次性质可扩展现有 `tests/finance/test_cvar.py`，复用第 23–25 项的生成器。先用固定、正权重场景，避免 Monte Carlo 噪声干扰数值导数。列入 finance 阶段，不挤进第一批 PR 检查。

### 4.3 三类尤其容易写错的性质

**EM 单调性。** 令 $\ell_t$ 为第 $t$ 次模型在固定样本上的平均对数似然。测试可用 $\ell_{t+1}-\ell_t\ge-(\varepsilon_a+\varepsilon_r\max(1,|\ell_t|))$。两项 $\varepsilon_a,\varepsilon_r$ 是经参考计算校准的绝对和相对误差预算；本报告不给未经测量的全族统一数字。普通精确 EM 的单调论证来自文献记忆及仓库理论，不自动覆盖近似 M 步、后验矩下限、参数裁剪、mini-batch 或任意 shrinkage。标量 MAP shrinkage 应检带惩罚目标；MAP 指最大后验估计。一般字段级规则未必对应同一标量目标。不得用“失败的样例都 `assume` 掉”来制造单调性。

**采样矩。** 设 $n$ 个独立同分布样本的均值为 $\bar X$，理论均值、方差为 $\mu,\sigma^2$。仅在方差有限时，Chebyshev 不等式给出 $\Pr(|\bar X-\mu|>\sqrt{\sigma^2/(n\delta)})\le\delta$。这里 $\delta$ 是该断言允许的误报概率。这个界较保守；不能以“矩总在宽界内”证明采样质量良好。Gamma 还可用样本和的已知分布作更有力的检查；这一分布事实来自文献记忆，使用前再核对参数约定。

二阶矩检查需要四阶矩。令 $M_2=n^{-1}\sum X_i^2$，用 $\operatorname{Var}(X^2)=E[X^4]-E[X^2]^2$ 为 $M_2$ 设置对应误差界。若同时有 $|\bar X-\mu|\le r_1$、$|M_2-E[X^2]|\le r_2$，则以分母 $n$ 计算的样本方差与理论方差之差至多为 $r_2+r_1(2|\mu|+r_1)$。这样误差有统计来源。重尾边界缺少四阶矩时，换成有界区间频数或分位数测试，不能继续套高斯近似。

多参数、多坐标、多后端会增加误报机会。给整组主要断言一个总预算，再分配各 $\delta$；并记录实际检查数。固定种子让失败可重现，不会让统计误报消失。发现失败后保留输入和种子，不能换种子一直重跑到绿。这些界假设理想独立抽样，并没有证明真实伪随机数发生器独立。

**PDF 归一化与 Bessel 参考。** 求积分格加密后“变化小”，仍可能两张网格都漏掉同一个窄峰。要限制分布尺度、定位峰并检查尾部。`build_pinv_table` 自己把累计积分除以末值，所以查表 CDF 的终点是 1，不能反过来证明 PDF 的归一化常数对。Bessel 恒等式与两个后端的一致性也可能共同通过同一种错误；冻结的高精度表提供另一种依据。mpmath 是高精度数值参考，不是形式化证明或自动的严格区间界。

### 4.4 设置草图与 JAX 的成本

以下只是未来可合入 `tests/conftest.py` 的设置草图。应保留现有数据 fixture 和已有 x64 设置，不覆盖整个文件。所列 Hypothesis API 来自记忆；最终所用版本的细节 `to verify`。

```python
import os

import jax
from hypothesis import settings

jax.config.update("jax_enable_x64", True)

settings.register_profile(
    "normix_ci",
    max_examples=20,
    stateful_step_count=12,
    deadline=None,
    derandomize=True,
    database=None,
)
settings.register_profile(
    "normix_nightly",
    max_examples=200,
    stateful_step_count=50,
    deadline=None,
    derandomize=False,
    database=None,
)
settings.load_profile(
    os.environ.get("NORMIX_HYPOTHESIS_PROFILE", "normix_ci")
)
```

这里关闭反例数据库，使 CI 行为不依赖历史缓存。代价是必须从失败日志提取反例，并用普通固定回归样例或 `@example` 保存。未来是否启用本地数据库可另定。不要通过环境变量覆写 `$HOME` 等系统路径。

JAX 还有几项具体限制：

- 在导入会创建数组的测试模块前开启 x64。本地已有 `tests/conftest.py` 这样做。若要检查 float32，另开进程并单定容差；不要在样例之间反复切全局配置。
- 一次 JIT 编译通常对应固定形状、dtype 和静态参数。把 shape、`maxiter`、方法字符串当大量随机取值，会造成许多编译。先固定这些结构，只生成数组值。
- 编译函数放在模块级。不要每个样例创建一个捕获新参数的 `jax.jit(lambda ...)`，然后误把编译时间当求解性能。
- `np.asarray`、读取标量或 `block_until_ready` 可等待异步结果。性能测量要等结果完成；只测调用返回时间可能漏掉设备执行。
- pytest 的 fixture 是准备和清理测试环境的机制。Hypothesis 下，函数作用域 fixture 通常不会按每个生成样例重建。`monkeypatch` 是测试中的临时替换；替换和状态必须逐例恢复。不要直接关闭相应健康检查。健康检查是工具用来提示生成或测试组织异常的机制。
- 日志级别也可能切换 EM 的 Python/scan 路径。做路径对照时明确固定其余配置和随机键。CPU 对照不能暗中替代用户指定的 JAX 路径。
- `deadline=None` 只关闭单例时限，不保证快速，也不应默认关闭所有健康检查。若 JAX 编译触发工具警告，先检查重编译，再记录局部处理理由。

### 4.5 一个普通性质测试文件骨架

以下拟放 `tests/test_property_contracts.py`，本次不创建。第一项调用真实 Gamma 参数转换。第二项直接编译并复用 Bessel 对称检查。数值范围和容差是首轮试验值，尚未校准；不是新发布的精度承诺。

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, strategies as st

from normix import Gamma
from normix.utils.bessel import log_kv

pytestmark = pytest.mark.contract

positive = st.floats(
    min_value=0.5, max_value=5.0,
    allow_nan=False, allow_infinity=False,
)
orders = st.floats(
    min_value=-5.0, max_value=5.0,
    allow_nan=False, allow_infinity=False,
)

@jax.jit
def bessel_pair(v: jax.Array, z: jax.Array) -> jax.Array:
    return jnp.stack((log_kv(v, z), log_kv(-v, z)))

@given(alpha=positive, beta=positive)
def test_gamma_roundtrip(alpha: float, beta: float) -> None:
    original = Gamma(alpha=alpha, beta=beta)
    restored = Gamma.from_natural(original.natural_params())
    got = np.array([float(restored.alpha), float(restored.beta)])
    assert np.isfinite(got).all()
    np.testing.assert_allclose(
        got, [alpha, beta], rtol=1e-10, atol=1e-12,
    )

@given(v=orders, z=positive)
def test_bessel_symmetry(v: float, z: float) -> None:
    pair = np.asarray(bessel_pair(
        jnp.asarray(v, dtype=jnp.float64),
        jnp.asarray(z, dtype=jnp.float64),
    ))
    assert np.isfinite(pair).all()
    np.testing.assert_allclose(
        pair[0], pair[1], rtol=1e-10, atol=1e-11,
    )
```

这两个性质不会因 AI 把实现中的一段公式原样抄到测试中而自然通过。但它们仍可能漏掉一对互相抵消的错误；必须保留现有 SciPy、高精度表和固定回归点。

### 4.6 一个真实规则的状态测试骨架

下面用真实 `SampleWeightedUpdate` 合并一维统计量。数组本身也是 pytree，所以可先用标量测试累计权重，再扩成 `NormalMixtureEta`。每个动作是一批整数；整数总和作为独立参考，不复制被测类的递推式。

```python
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from normix.fitting.eta_rules import SampleWeightedUpdate

class WeightedMeanMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.updater = SampleWeightedUpdate()
        self.state = self.updater.initial_state()
        self.eta = jnp.asarray(0.0, dtype=jnp.float64)
        self.total = 0
        self.count = 0
        self.step = 0

    @rule(batch=st.lists(st.integers(-5, 5), min_size=1, max_size=4))
    def append_batch(self, batch: list[int]) -> None:
        values = jnp.asarray(batch, dtype=jnp.float64)
        self.eta, self.state = self.updater(
            self.eta, jnp.mean(values), self.step, len(batch), self.state,
        )
        self.total += sum(batch)
        self.count += len(batch)
        self.step += 1

    @invariant()
    def agrees_with_raw_data(self) -> None:
        assert int(self.state["cumulative_n"]) == self.count
        if self.count:
            np.testing.assert_allclose(
                float(self.eta), self.total / self.count,
                rtol=1e-12, atol=1e-12,
            )

TestWeightedMean = WeightedMeanMachine.TestCase
TestWeightedMean.settings = settings()
TestWeightedMean.pytestmark = [pytest.mark.contract]
```

这是未运行示例，包含 pytest 与状态测试的设置衔接，最终收集行为需验证。一个 testcase 可以包含多步动作；`stateful_step_count` 与 `max_examples` 分别控制序列长度和序列数量的预算。缩减可删除批次或缩小元素，帮助显示“批次大小不同”这个触发条件。

它检查真实累计规则，不保证流式恢复、不证明并发安全，也不要求不同批次划分的完整 EM 结果相同。EM 的 E 步依赖当前模型，通常不具备这种划分不变性。

### 4.7 与现有测试提速工作衔接

约 71 分钟是输入给出的完整 CPU 套件耗时，不是 PR 默认集合的实测耗时，也不是本次测量。当前 `pyproject.toml` 默认排除 `slow`、`stress`、`integration`、`gpu`；`ci.yml` 使用该默认集合。这一分层必须保留。

建议第一批只增加 6–8 条快性质，每条约 8–20 个输入。先选第 1、5、7、15–17、20–21 项中较便宜的部分。GIG 冷启动反解、EM 多次完整拟合、求积分格、大样本采样全部从 `slow` 开始。未来实测后才能把某条移入默认集合。

给第一批设一个**待测的增量时间目标**：普通 CI 机器上不超过约 1–2 分钟。若超出，先找编译形状和重复 fixture，再减少预算或调整标记；不通过改成 CPU 后端来“提速”。完整 25 项分阶段约需 2–3 周实施和校准，取决于数值反例数量；不是在每条现有测试外包一层 `@given`。

未来运行示例，**本次未执行**：

```bash
uv run pytest tests/ --durations=50
NORMIX_HYPOTHESIS_PROFILE=normix_nightly uv run pytest tests/test_property_contracts.py tests/test_property_numerics.py -m "not gpu" --durations=50
```

两个 `test_property_*.py` 名称都是拟议文件。夜间 profile 只改变 Hypothesis 预算，不会自动改变 pytest 的标记选择。因此例子显式覆盖 marker 过滤。“`smoke or contract`”也可能选中同时带 `slow` 的测试；若要求快，就写出排除条件。现有 `--strict-markers` 下不凭空使用未注册的 `property` 或 `nightly` 标记。

对应仓库规则：

- 使用 uv、pytest、已有 marker 和 x64 配置；遵守单观测 API，以 `vmap` 批量。
- 按 `.cursor/rules/testing-guidelines.mdc` 保留参考对照与显式容差。性质测试针对契约和故障机制，不替代已有测试。
- 保留 `test_no_host_callbacks.py` 的递归扫描及其扫描器自检。其内部故意嵌入 callback 的测试已是负向校验实例。
- 如果未来确认并修复 bug，按仓库 TDD 流程先留下失败测试；本报告不伪造这样的红绿证据。
- 不在本次更新 `design.md`、计划或其他知识文件。实际实施时才把获准的决定和拒绝理由同步到规范位置。

### 4.8 变异式校验与记录

采用输入概览提出的做法：每条新性质至少有一次针对性“故意改坏后失败”的记录。称为**变异式校验**：小幅改变实现，确认性质能识别该类错误。它检查测试有没有作用，不证明测试完备。

未来在可丢弃副本或受控测试替身中做，当前不能改任何源码。一次只改变一个机制，并区分断言失败、编译失败和安装失败；编译不了不算已检出目标数值错误。

| 性质 | 建议变异 | 期望检出的现象 |
|---|---|---|
| 自然参数往返 | Gamma 恢复 $\alpha$ 时漏加 1 | 参数不符 |
| 不可变更新 | 替身故意覆盖旧对象快照 | 原对象值变化 |
| 后端传递 | 把 `m_step_backend` 忽略为固定 CPU | 路径哨兵失败；纯值比较可能漏检 |
| 无回调 | 只在内层条件分支加入回调 | 递归 jaxpr 扫描失败 |
| 累计规则 | 用平均批次替代按样本数加权 | 不同大小批次产生错误均值 |
| 停止状态 | 去掉停止后的标志冻结 | 后续候选翻转最终状态 |
| PDF 归一化 | 密度乘上固定的 1.1 | 积分质量偏离 1；归一化查表测试可能漏检 |

建议未来记录模板如下。本报告中的状态一律是“未执行”，不填虚构的最小反例或耗时。

```text
property_id / 对应决策行：
源码 commit / 文件 / 函数：
输入域、排除条件、dtype、设备、后端：
参考来源与容差理由：
工具版本、profile、命令、种子或重放信息：
原代码结果与耗时：未执行
变异的精确差异：
变异结果与缩减反例：未执行
失败是否来自目标断言：待检查
恢复原代码后的结果：未执行
Xiang 对规约的审核意见：
```

已有 jax-lean `tests/test_certification.py` 也故意改除数、运算符和变量引用，再要求 Lean 拒绝。可以借鉴“先验证检查器会失败”的方法；本次只读了测试，没运行这些拒绝实验。

## 5. 一个小 TLA+ 模型：旧文档不能覆盖新文档

只选文档发布这一个候选。模型对应 `docs.yml` 和 `docs-full.yml` 的共用部署目标，不把数值计算塞进状态空间。

先明确**示例假设**：Xiang 已决定这两份产物可以比较新旧，旧产物编号 1，新产物编号 2，网站不应退回较小编号。这只是教学政策，尚不是 normix 已接受的站点政策。master 与 release 谁优先，见末尾问题。

模型有三个变量：

- `phase`：每个任务处于 `queued`、`built`、`done` 哪个阶段。
- `shown`：当前网站显示的版本，初始为 0。
- `highWater`：到目前为止完成发布决策的最高版本。它只是检验性质用的历史摘要，不要求真实实现存储这个同名字段。

`Build` 可按任何顺序发生。`PublishOrSkip` 检查版本，只接受不比网站旧的产物。**比较版本和更新网站被抽象成同一个原子步骤**；这是该模型最需要审的前提。

下面是示意 TLA+，未经过语法检查或 TLC。若以后运行，应先验证错误版本能报错，再验证修正版本。

```tla
---- MODULE DocsPublish ----
EXTENDS Naturals

Jobs == {"old", "new"}
Version(job) == IF job = "old" THEN 1 ELSE 2

VARIABLES phase, shown, highWater
vars == <<phase, shown, highWater>>

Init ==
    /\ phase = [job \in Jobs |-> "queued"]
    /\ shown = 0
    /\ highWater = 0

Build(job) ==
    /\ phase[job] = "queued"
    /\ phase' = [phase EXCEPT ![job] = "built"]
    /\ UNCHANGED <<shown, highWater>>

PublishOrSkip(job) ==
    /\ phase[job] = "built"
    /\ phase' = [phase EXCEPT ![job] = "done"]
    /\ shown' = IF Version(job) >= shown THEN Version(job) ELSE shown
    /\ highWater' = IF Version(job) > highWater
                     THEN Version(job) ELSE highWater

Next == \E job \in Jobs : Build(job) \/ PublishOrSkip(job)

TypeOK ==
    /\ phase \in [Jobs -> {"queued", "built", "done"}]
    /\ shown \in 0..2
    /\ highWater \in 0..2

NoRollback == shown = highWater
AllFinished == \A job \in Jobs : phase[job] = "done"
EventuallyFinished == <>AllFinished

Spec ==
    /\ Init
    /\ [][Next]_vars
    /\ (\A job \in Jobs :
          WF_vars(Build(job)) /\ WF_vars(PublishOrSkip(job)))
====
```

`\E` 是“存在”，`\A` 是“对所有”。`[Jobs -> {...}]` 是所有从任务集合到阶段集合的函数。`EXCEPT ![job] = ...` 表示只改该任务的阶段。`WF` 是弱公平性：若某动作从某时刻起一直可执行，最终必须执行。两个任务完成后可以一直保持原状态。

未来的 TLC 配置草图同样未经运行：

```text
SPECIFICATION Spec
INVARIANT TypeOK
INVARIANT NoRollback
PROPERTY EventuallyFinished
CHECK_DEADLOCK FALSE
```

这里关闭死锁报告，是因为所有任务完成后没有下一次工作，本来就应停止。只对这个教学模型这样设置；在需要持续服务的模型里，无条件关掉死锁检查可能掩盖错误。活性仍单独检查。

把 `shown'` 那一行故意改成无条件 `shown' = Version(job)`，手算即可得到：

| 步骤 | old 阶段 | new 阶段 | shown | highWater |
|---|---|---|---:|---:|
| Init | queued | queued | 0 | 0 |
| Build(new) | queued | built | 0 | 0 |
| PublishOrSkip(new) | queued | done | 2 | 2 |
| Build(old) | built | done | 2 | 2 |
| 错误的 PublishOrSkip(old) | done | done | 1 | 2 |

最后一行违反 `NoRollback`。这是**人工推演的反例**，不是声称已经拿到 TLC 输出。修正版在最后一步跳过旧产物，保持 `shown=2`。在给定公平性下，两任务最终结束；没有公平性时，允许永久不动，所以不能得出该活性结论。

它能检查：在这个有限模型中，构建完成的各种交错不会让已显示版本回退；旧任务仍能结束；任务阶段不会走到未定义值。它不能检查构建是否真的使用正确 commit、产物内容是否新、工作流权限、网络失败、取消语义，也不能证明 `peaceiris/actions-gh-pages` 的实现。

最危险的实现误读是：先读取网站版本，隔一段时间再上传，就认为实现了原子步骤。两个任务可能都读到 0，旧任务稍后覆盖新任务。若现实分成“读取—判断—写入”，模型也必须拆开，允许这几步交错。需要一个明确的单写者发布入口，或等价的原子接受机制；是否采用它是未来设计决定。仅把构建串行化，也要检查队列顺序是否符合版本政策。

后续若值得维护模型，给每个动作建立映射表：YAML/job/step、使用的 ref、产物标识、接受条件、失败后动作。TLC 配置、模型版本、对应源码 commit、手工或机器反例都应保存。模型改动后，确认没有通过把动作删掉、增加不真实的公平性或扩大原子步骤来“修复”错误。

这个试点预计 AI 用 1–2 人日能准备并运行有限模型、整理反例和实现映射；Xiang 用 1–2 小时审版本政策与原子性假设。如果静态审查加一个统一部署入口已解决问题，也可以不长期引入 TLA+。该工具的价值由未来协议复杂度决定。

## 6. 与概览及 jax-lean 的关系

概览的“先做性质测试”方向可保留，但三句话应改得更准确。

**“TLA+ 暂时用不上”缺少范围。** 对数值内核，它省略了最重要的精度和参数问题；对发布工作流，它能检查真实存在的交错风险。对在线 EM，必须先区分现有数组批处理器与未来可恢复服务。不能用文件名或“应用层”标签作判断。

**“finance 用回测和数值对照就够”太宽。** 预算约束、不可达目标、场景权重和局部目标约定都应成为自动检查的性质。回测一条历史路径，很难覆盖退化秩、零权重、参数换单位或错误失败分支。没有下单协议时，先不引入 TLA+ 有具体理由；不是 finance 不值得严谨检查。

**“没有 Bessel 原语，所以不能验证”不是根本障碍。** Xiang 的判断正确：若 Bessel 数值实现是可追踪的 JAX 运算，可以尝试为其 jaxpr 建立 Lean 语义，不必存在一个名叫 Bessel 的原语。但“能追踪”“能翻译”“翻译等价”“近似了真实 $K_\nu$”“浮点误差受控”是五个不同问题。

本地 jax-lean 提供了这个区别的具体证据：

- `python/jaxlean/translate.py` 生成可读 Lean 函数；`python/jaxlean/importers/jaxpr.py` 的 `JaxprImporter` 生成另一份显式中间程序；`certify.py` 生成两者等值的证明义务。
- `JaxLean/Core/Jaxpr.lean` 定义该中间程序的求值；`Core/RealOps.lean` 用实数函数解释部分超越运算。`examples/certificates/generated/CertifiedMean.lean` 给出固定长度 2 的均值例子。
- `SEMANTICS_REFERENCES.md` 明说，Python 导入器与真实 JAX 执行的对应不在该等式证明内。实数语义省略浮点舍入。`tests/test_certification.py` 甚至包含浮点转换实际改变数值、实数模型仍通过恒等证书的例子。
- `SUPPORTED_JAXPR.md` 区分普通翻译和证书支持。普通翻译的 `scan` 默认只展开至 32 次；证书路径不支持 `scan`，也拒绝捕获的浮点数组常量。支持某个运算不表示所需性质已有证明。

normix `bessel.py` 的 `_edge` 使用固定次数循环，`_geometry` 使用 `arcsinh`，内核还用 `expm1`、`log1p`、求和以及浮点节点常量。JAX 怎样降低这些操作，要用对应版本的实际 jaxpr 清单核对；本次未追踪，不能宣称它已经落在支持集合中。当前 40 次窗口迭代还提示默认展开预算可能不够。即使解决翻译问题，192 节点求积与连续积分之间的误差仍需另证。

`stop_gradient` 在前向值模型中是恒等函数，但它改变自动微分规则。只证明前向值相等，不会自动验证冻结积分节点后的导数解释。这是本文性质计划同时检查 AD、矩 bundle 和独立参考的原因。

本地副本的 `pyproject.toml` 固定 JAX 0.8.0，normix 的 uv 解析约束是 0.9.1；后者不是发布依赖的精确版本上限。副本根目录没有 LICENSE 文件，但部分示例目录有单独许可。具体复用权限待核实，不据此断言所有内容均无许可。完全可以独立实现同一思路；本报告没有复制它的实现，也不建议为试验降级 normix。

对本报告的任务，jax-lean 最可借鉴的是检查方法：语义边界写清楚；支持与拒绝列表写清楚；改变生成结果后检查证书会拒绝；架构导入边界用小型源码测试守住。暂不把开发 jaxpr→Lean 翻译器并入 Hypothesis 或 TLA+ 试点。

## 7. 决策与拒绝的方案

本节遵照仓库的决策记录格式。日期均为 2026-10-24。依据是上面的静态源码、输入材料和已标明的工具记忆。**Xiang 已授权的是这份研究报告，不是以下工具迁移。** 因此“拒绝”表示本文建议不采用，除已有仓库决策外，不冒充 owner 的正式否决。

| 方案 | 状态与范围 | 理由与依据 | 谁决定；何时重开 |
|---|---|---|---|
| JAX 原生路径、用户选择的后端 | 保留，已有约束 | 用户明确要求；S9/S11；现有 no-callback 测试 | Xiang/既有决策；仅在他明确改变契约时重开 |
| 少量 Hypothesis + 既有 pytest | 推荐，待实施 | 直接测真实代码，能覆盖连续参数与调用序列；复用既有参考和 marker | Xiang 审第一批规约；若时间成本超预算，调整策略与范围 |
| TLA+ 文档发布小试点 | 有条件推荐 | 已有多个工作流共写网站；版本接受政策仍未定 | Xiang 定网站政策后再做；复杂度低时不必长期维护 |
| 为所有非数学代码建 TLA+ | REJECTED，本阶段 | 大部分是纯函数/局部顺序状态；模型遗漏数值风险且容易漂移 | 本文判断；增加持久化、并发调用承诺或恢复协议时重开 |
| 在 JAX 路径偷偷转 CPU 以便验证或提速 | REJECTED | 违反用户约束；S9 已拒绝这种 offload，不能当工具便利性选择 | 已有约束；不由本报告重开 |
| 为形式化把核心移植到别的语言 | REJECTED，本项目方向 | 增加 JAX 接口与另一实现的对应义务，破坏原生工作方式 | 用户约束；独立子系统确有迁移需求时另议 |
| 所有性质都跑 200 个样例并进 PR | REJECTED | 编译和反解会扩大已有测试成本；71 分钟输入提示应继续分层 | 本文判断；实际耗时证明便宜后局部扩大 |
| 只写往返、只比 cpu/jax | REJECTED，作为唯一证据 | 两路径可能共享错误；往返可由成对错误抵消 | 本文判断；必须保留独立参考或不同推导 |
| 原样采用 jax-lean | 暂缓，不否定思路 | 版本、证书子集、浮点语义和许可范围均有未解决项 | 独立研究试验另立范围；不等同于等待原仓库支持 Bessel |

其他工具也按实际问题选择。以下能力描述来自工具记忆，未核对最新版本、维护状态或扩展包；不据此判断其“过时”或“不可用”。

| 工具 | 从零解释 | 本阶段拒绝/暂缓理由 | 重开条件 |
|---|---|---|---|
| Alloy | 用关系和约束描述结构，在有限范围搜索实例/反例；部分版本也支持时序描述，细节 `to verify` | normix 没有复杂权限、对象关联或配置图问题；数值误差不因换成关系模型而容易检查 | 出现复杂模型注册/配置组合、结构约束比步骤顺序更重要 |
| P language | 用事件驱动状态机描述程序和并发行为，可探索调度 | 当前库没有 actor/message 服务层；为纯数组代码引入事件程序模型成本较高 | 建立独立异步拟合服务或多参与者任务协议 |
| Quint | 面向状态机规约的语言及工具，目标包括让模型表达和检查更易用；具体执行/验证后端 `to verify` | 能否改善语法学习，不会消除模型与 Python 的对应义务；首个试点同时选两套规约语言会分散精力 | TLA+ 语法成为实际主要障碍时，用同一发布例子比较一天以内的体验 |
| stateright | 以 Rust 描述系统并做模型检查的工具 | normix 不是 Rust 项目；不会自动读取 Python/JAX，仍要重写模型 | 出现已有 Rust 服务，需要在其语言内维护行为模型 |
| Dafny / Verus | 把前置条件、后置条件和循环性质放进程序，再用逻辑工具验证；Verus 面向 Rust | 直接验证纯 Python/JAX/Bessel 不是其自然路径；移植会新增维护和对应证明 | 单独出现小型关键控制组件，且本来就决定采用相应语言 |
| CrossHair | 对部分 Python 代码做符号执行：把输入当符号，通过约束求解寻找违约输入 | 能覆盖哪些 Python 操作取决于支持；JAX tracer、数组运行时、编译和特殊函数不是可默认覆盖的范围 | 一个纯 Python 配置/校验模块风险足够高时，先做半日能力探测 |

这些拒绝是投入排序，不是“只有 Lean 和 Hypothesis 正确”。尤其是状态模型语言，模型边界和规约审查通常比语法选择更决定结果。

## 8. 排名建议与下一步

1. **先把后端与调用契约补成小规模检查。** 优先处理未知 backend、已有 jaxpr 无回调、不可变更新、pytree 和累计权重。首批 6–8 条预计 3–5 人日，包含首次测时、缩减反例和变异式校验；Xiang 审约 3–5 小时。AI 可写候选规约、策略和测试，不能自行把失败路径改成 CPU 或放宽数学前提。`log_kv` 缺口先按 TDD 复现，修复另作实现任务。

2. **再扩数值性质与 finance 约束。** 把 GIG 反解、EM 目标、采样矩和 PDF 积分放慢测，复用既有 Bessel 高精度表。预计再用 5–10 人日，Xiang 审约 4–8 小时；若发现真正数值问题，修复工期另计。AI 做生成器、日志和对照，Xiang 决定可接受参数域、误差预算及重尾例外。完整清单以约 2–3 周作预算，不承诺每条都应采用。

3. **给文档发布一次 1–2 人日的 TLA+ 试验。** 先回答站点应追随什么版本，再运行第 5 节的错误/修正两模型，并把动作映射回工作流。Xiang 审 1–2 小时。AI 可以产生模型和反例；Xiang 必须审“比较和写入为何能当原子操作”。没有明确发布政策时，先停在研究模型，不能声称验证了真实部署。

4. **可恢复流式 EM 和多 agent 协议暂缓。** 若需求成立，在写恢复/共享状态代码前安排模型工作。单写者流式原型的模型可先留 2–4 人日与半日人工评审；多写者、租约、跨进程恢复需重新估计。AI 可枚举故障顺序；Xiang 决定“允许重放还是必须恰好一次”、状态持久化边界与代价。

Lean 数学验证及独立 jaxpr→Lean 试验可另立研究范围。本报告的排序没有否定它们，也没有要求它们成为性质测试合入的前置条件。

## 9. 给 Xiang 的开放问题

1. 网站应展示 master 的最新状态、最新 release，还是分别维护两份？这决定文档发布模型的性质，不能由工具替你选。
2. 是否承诺同一个 `BatchEMFitter` / `IncrementalEMFitter` 实例可被并发调用？若没有这种需要，先测试顺序复用和异常恢复即可。
3. 增量 EM 的近期需求只是内存里的小批量拟合，还是需要数据流断点恢复？若要恢复，数据能否重读，是否需要跨进程保存随机键与更新规则状态？
4. 首批性质测试可接受的 PR 增量时长是多少？本文暂提 1–2 分钟目标；夜间慢测能否独立安排？
5. 第一阶段精度契约是否明确限定 x64、温和参数域？边界分布、极重尾和 GPU 应单列哪些必须支持的情形？
6. 未知后端名是否应在包括 `log_kv`、批量 helper 在内的所有公开入口统一抛错？这与已有 E17 的方向一致，但当前源码覆盖不齐。

## Sources

所有链接都是本地文件，路径以本报告所在目录为基准。源码与测试是静态阅读依据，不代表测试已在本次执行。外部作品只列确定的名称和年份；未确认的版本或年代不补写。

**仓库规则与研究输入**

- [AGENTS.md](../../../AGENTS.md)、[dev-notes/README.md](../../README.md)、[ARCHITECTURE.md](../../ARCHITECTURE.md)。
- [概览报告，2026-10-19](00_overview_report.md)。其中社交媒体帖、行业统计及远程论文没有在本次验证，不作为本文结论的独立证据。
- [设计决策表](../../design/design.md)：E3、E9–E11、E16–E17、S4–S12、FIN-1–FIN-5；[设计索引](../../design/index.md)。
- [运行测试指引](../../../.cursor/skills/running-tests/SKILL.md)、[测试指南](../../../.cursor/rules/testing-guidelines.mdc)、[编程约定](../../../.cursor/rules/coding-conventions.mdc)、[设计记录规则](../../../.cursor/rules/maintain-design-docs.mdc)。写作与分析还按本地 `unslop`、`how`、`principles`、`agent-maintenance` 指引核对。
- [pyproject.toml](../../../pyproject.toml)、[tests/conftest.py](../../../tests/conftest.py)：依赖、JAX 约束、marker 与 x64。完整 CPU 套件约 71 分钟来自本次任务输入，未复测。

**normix 源码与既有测试**

- [exponential_family.py](../../../normix/exponential_family.py)；[gamma.py](../../../normix/distributions/gamma.py)；[generalized_inverse_gaussian.py](../../../normix/distributions/generalized_inverse_gaussian.py)；[generalized_hyperbolic.py](../../../normix/distributions/generalized_hyperbolic.py)。
- [mixtures/marginal.py](../../../normix/mixtures/marginal.py)、[mixtures/joint.py](../../../normix/mixtures/joint.py)、[fitting/eta.py](../../../normix/fitting/eta.py)：对象、公共矩、统计量及 E/M 接口。
- [fitting/em.py](../../../normix/fitting/em.py)、[fitting/eta_rules.py](../../../normix/fitting/eta_rules.py)、[fitting/solvers.py](../../../normix/fitting/solvers.py)：状态、后端分派、热启动及失败处理。
- [utils/bessel.py](../../../normix/utils/bessel.py)、[utils/rvs.py](../../../normix/utils/rvs.py)、[utils/quadrature.py](../../../normix/utils/quadrature.py)：数值内核、分位数快照和场景测度。
- [finance/optimization.py](../../../normix/finance/optimization.py)、[finance/risk.py](../../../normix/finance/risk.py)、[finance/transaction_costs.py](../../../normix/finance/transaction_costs.py)：约化持仓、CVaR、局部交易成本优化。
- [test_no_host_callbacks.py](../../../tests/test_no_host_callbacks.py)、[test_bessel_contract.py](../../../tests/test_bessel_contract.py)、[Bessel 参考表](../../../tests/data/bessel_reference.json)、[test_distribution_conversions.py](../../../tests/test_distribution_conversions.py)、[test_incremental_em.py](../../../tests/test_incremental_em.py)、[交易成本测试](../../../tests/finance/test_transaction_costs.py)。
- 相关现有测试位置：[test_em_cpu_vs_jax.py](../../../tests/test_em_cpu_vs_jax.py)、[test_expectation_autodiff.py](../../../tests/test_expectation_autodiff.py)、[test_regularizations.py](../../../tests/test_regularizations.py)、[test_cdf_ppf.py](../../../tests/test_cdf_ppf.py)、[test_gig_rvs.py](../../../tests/test_gig_rvs.py)。这些文件是未来扩展的入口；本文不声称已逐项审计其全部断言。

**仓库理论、设计与流程**

- [指数族设计](../../../docs/design/exponential_family.md)、[EM 设计](../../../docs/design/em_framework.md)、[内部求解器/Bessel 设计](../../design/solvers_and_bessel.md)。
- [GIG 理论](../../../docs/theory/gig.md)、[EM 理论](../../../docs/theory/em_algorithm.md)、[在线 EM 理论](../../../docs/theory/online_em.md)、[仓库参考文献](../../../docs/references.md)。理论页面是拟验证的陈述来源，不是机器证明。
- [Bessel 矩内核技术说明](../../tech_notes/bessel_moment_kernel.md)、[GIG 反解旧技术说明](../../tech_notes/gig_eta_to_theta.md)。后者只描述 CPU 多起点的部分内容，不能用其“不支持 JIT”概括当前 JAX 反解；当前行为以源码及 S9/S11 为准。
- [多 agent 与自动循环计划](../../plans/loops_and_orchestration.md)。
- [ci.yml](../../../.github/workflows/ci.yml)、[release-please.yml](../../../.github/workflows/release-please.yml)、[publish.yml](../../../.github/workflows/publish.yml)、[docs.yml](../../../.github/workflows/docs.yml)、[docs-full.yml](../../../.github/workflows/docs-full.yml)。未读取远程保护规则或实际运行日志。

**jax-lean 本地副本**

- [README.md](../../../.research-inputs/jax-lean/README.md) 很短，主要是项目页链接；没有访问该远程页面。[ARCHITECTURE.md](../../../.research-inputs/jax-lean/ARCHITECTURE.md)、[SUPPORTED_JAXPR.md](../../../.research-inputs/jax-lean/SUPPORTED_JAXPR.md)、[SEMANTICS_REFERENCES.md](../../../.research-inputs/jax-lean/SEMANTICS_REFERENCES.md) 是主要本地说明。
- [Core/RealOps.lean](../../../.research-inputs/jax-lean/JaxLean/Core/RealOps.lean)、[Core/Jaxpr.lean](../../../.research-inputs/jax-lean/JaxLean/Core/Jaxpr.lean)：实数语义与带形状/类型的中间程序。
- [certify.py](../../../.research-inputs/jax-lean/python/jaxlean/certify.py)、[importers/jaxpr.py](../../../.research-inputs/jax-lean/python/jaxlean/importers/jaxpr.py)、[translate.py](../../../.research-inputs/jax-lean/python/jaxlean/translate.py)：导入、生成和证书边界；[pyproject.toml](../../../.research-inputs/jax-lean/pyproject.toml) 固定 JAX 版本。
- [均值等例子的 Python 源码](../../../.research-inputs/jax-lean/examples/certificates/code.py)、[CertifiedMean.lean](../../../.research-inputs/jax-lean/examples/certificates/generated/CertifiedMean.lean)、[test_certification.py](../../../.research-inputs/jax-lean/tests/test_certification.py)、[test_architecture.py](../../../.research-inputs/jax-lean/tests/test_architecture.py)。未执行 Lean 构建或这些测试。

**外部作品与工具知识：本次离线，来自记忆或本地书目**

- Leslie Lamport，*Specifying Systems*，2002。TLA+ 规约、状态与行为的背景；来自文献记忆，未核对具体页码。
- Koen Claessen、John Hughes，*QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs*，2000。性质测试背景；不是本文 Hypothesis API 的版本依据。
- Hypothesis 文档：strategies、settings、stateful testing；TLA+ 文档：TLC、PlusCal。本文依照工具记忆介绍；当前版本的精确行为 `to verify`，不虚构文档年份。
- Dempster、Laird、Rubin，*Maximum Likelihood from Incomplete Data via the EM Algorithm*，1977；Cappé、Moulines，*On-line Expectation-Maximization Algorithm for Latent Data Models*，2009；McLachlan、Krishnan，*The EM Algorithm and Extensions*，2008。名称与年份可对照本地 `docs/references.md`；本文未读取外部全文。
- NIST DLMF：修正 Bessel 函数的对称与递推关系。公式来自数学知识与输入概览，未访问外部页面，不编造节号或 Mathlib 引理名。
- Lean/Mathlib、Alloy、P、Quint、stateright、Dafny、Verus、CrossHair、mpmath 的能力概述来自工具记忆；没有核查当前兼容矩阵或发布状态。
