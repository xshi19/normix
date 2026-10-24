# 从 jaxpr 到 Lean：normix 的可行范围、设计与路线

写给 Xiang Shi。2026-10-24。

建议做一个限期试验：自己写小型 jaxpr 导出器，先验证标量和固定长度向量的实数模型。jaxpr 是 JAX 追踪得到的基本运算程序；Lean 是检查数学证明的语言。用当前 Bessel 算法作为后续目标。同时向 Sasha Rush 询问许可和合作意向。不要等上游支持全部 normix，也不要现在承诺浮点端到端证明。

本报告只做静态阅读。没有联网、安装依赖、运行 JAX、测试或 Lean。下文的“代码事实”来自本地文件；“拟证明”表示还没有机器检查的命题。工期是估算。涉及外部工具和数学文献而未在本地核实的内容，会明确标为“文献记忆”或 `to verify`。

阅读基线是 normix 提交 `51e053c57237a29c71b56d3a2a86988277477bd9`，以及 `.research-inputs/jax-lean/` 的本地副本。输入说明称后者有 12 个提交。副本没有独立 Git 元数据，所以这里不确认其提交数或上游版本；不能把父仓库的 Git 信息当成 jax-lean 的信息。后文将此副本简称为 JL。

仓库规则已读：[AGENTS.md](../../../AGENTS.md)、[dev-notes/README.md](../../README.md)。本次只写本文件。设计建议与否决理由也只记在这里；它们没有自动成为 normix 已接受的设计决定。

## 1. 先修正概览中的判断

[起始概览](00_overview_report.md)抓住了实数与浮点之间的区别，但对当前任务有几处过于绝对。下表也回应 Xiang 关于网站数学的判断。

先说明表中几个名字。`scan` 是携带状态、执行固定次数的循环。`custom_jvp` 允许用户指定方向导数规则；`custom_jvp_call` 是相应的程序调用节点。TLA+ 是描述软件状态及其变化的规约语言。它们的验证范围在后文展开。

| 起始判断 | 本次静态阅读后的结论 |
| --- | --- |
| 没有 Bessel 原语，所以不能做 normix | 当前 `log_kv` 已展开为普通 JAX 运算。没有一个名为 Bessel 的原语并非根本障碍。真正缺口是运算覆盖、循环、常数表和数学误差证明。 |
| jax-lean 支持把 jaxpr 翻译并生成证书 | 要区分普通翻译和可生成证书的子集。普通翻译支持部分 `scan`、`custom_jvp_call`；证书路径不支持它们。 |
| 浮点数值“不值得证” | 应改为分层投入。局部误差界有价值；整个 Bessel、求解器和编译后程序的浮点证明成本很高。 |
| Xiang：网站数学都可以用 Lean 验证 | 原则上可逐条形式化。需要先补齐定义、参数域和分析假设。不能据此估计全部内容只需数周，也不能默认 Mathlib（Lean 的数学库）已有 Bessel 所需定理。 |
| normix 暂时用不上 TLA+ | 对纯数值公式，Lean 更直接。对停止、失败回退、状态保持等流程，TLA+ 有可用的小范围；它不能代替数值分析。 |

还有两个需要避免的旧印象。

第一，当前 [bessel.py](../../../normix/utils/bessel.py) 没有 `custom_jvp`、分区渐近分支或有限差分。`custom_jvp` 仍用于 [ExponentialFamily._implicit_natural_params](../../../normix/exponential_family.py)，负责期望参数反演的导数。

第二，旧 [gig_eta_to_theta.md](../../tech_notes/gig_eta_to_theta.md) 仍说多起点控制流使求解器不能 JIT。JIT 是即时编译，即把函数编译成可重复执行的程序。当前代码及设计行 S5、S11 已有 JAX 路径。公开 [solvers_and_bessel.md](../../../docs/design/solvers_and_bessel.md) 的采样段还写着“no `while_loop`”，而 `_gig_rvs_tdr` 确实使用 `lax.while_loop`，即按条件继续的循环。本报告按源码判断，不改这些文件。

## 2. jaxpr 和 Lean 分别做什么

### 2.1 用 Gamma 方差理解 jaxpr

jaxpr 是 JAX 追踪函数后得到的中间表示。中间表示就是介于 Python 源码和底层机器程序之间的程序描述。它记录输入、常数、基本运算及其连接方式。

“追踪”通常用带形状和数值类型的信息走过函数，记录数组运算。它不是对所有 Python 行为的验证。依赖静态配置的 Python 分支在此时已经选定。backend 指执行后端。`backend='jax'` 与 `backend='cpu'` 因而不是同一份待证明程序。

一个 primitive，中文称“原语”，是这一层不可再拆的基本操作。`add`、`log` 是原语；`scan` 也可以是带有子程序的原语。Python API 的函数名不一定是原语名。例如 `gammaln` 通常对应 `lgamma`，具体仍要检查所锁定 JAX 版本的输出。

[Gamma.var](../../../normix/distributions/gamma.py) 的实际 Python 函数体是：

```python
return self.alpha / self.beta**2
```

Gamma 的形状参数为 $\alpha>0$，率参数为 $\beta>0$。这一行先平方 $\beta$，再用 $\alpha$ 除以结果。下面是手写的合理 jaxpr 示意，不是本次运行 JAX 得到的输出：

```text
{ lambda ; alpha:f64[] beta:f64[]. let
    b2:f64[] = integer_pow[y=2] beta
    variance:f64[] = div alpha b2
  in (variance,) }
```

`f64[]` 表示 float64 标量。`b2` 和 `variance` 各绑定一次。这种写法叫 SSA，即“每个中间变量只赋值一次”。对象字段会成为输入或捕获的常数，取决于如何追踪；这里选择把两个字段当输入。

Lean 是能检查数学证明的语言。Mathlib 是它的数学库。我们可在 Lean 中把这个程序解释为实数函数 $V(\alpha,\beta)=\alpha/\beta^2$，再证明合法参数下 $V>0$。这证明了导出函数的实数性质。它没有证明 float64 的下溢、溢出、舍入，也没有单靠这一行证明 Gamma 分布的方差就是该公式。后者还需要积分或矩公式。

### 2.2 三个不同的证明目标

把 jaxpr 记为 $J$，导出的 Lean 程序记为 $F$，规约中的算法记为 $A$。规约就是明确写出的预期行为。

1. **翻译正确。** Lean 中对 $J$ 的解释结果等于 $F$。解释器是逐条执行这些数学运算的定义。
2. **算法正确。** $F=A$，或 $F$ 在指定参数域内以明确误差逼近目标数学量。
3. **机器实现准确。** 实际 JAX 运行结果与这个实数结果相差多少。

证书是第一项的证明对象。Lean 的内核，即最后检查证明的小型核心，会检查它。证书不是“这段 Python 已经完全正确”的标签。第二项需要另写定理。第三项需要浮点模型及对编译和执行的额外假设。

本报告建议先做前两项的有限范围。对于外部程序，最终仍要信任 JAX 追踪确实记录了预期计算。

## 3. jax-lean 的实际机制

本节依据 JL 的 [README.md](../../../.research-inputs/jax-lean/README.md)、[ARCHITECTURE.md](../../../.research-inputs/jax-lean/ARCHITECTURE.md)、[SUPPORTED_JAXPR.md](../../../.research-inputs/jax-lean/SUPPORTED_JAXPR.md)、[SEMANTICS_REFERENCES.md](../../../.research-inputs/jax-lean/SEMANTICS_REFERENCES.md) 和下列实现。README 本身只有项目页链接，主要说明在后三个文件。

### 3.1 从 Python 到证明文件

处理顺序如下。每个箭头都有自己的边界。

```text
Python 函数 + 输入形状/类型 + 静态配置
    → JAX make_jaxpr → ClosedJaxpr
                         ├→ translate.py → 可读 Lean 函数
                         └→ importers/jaxpr.py → Lean 内部程序表示
    → certify.py → 两者结果相等的待检查证明
    → Lean 检查 → 翻译证书
    → 人或 agent 另写应用定理 → 业务/数学性质
```

`ClosedJaxpr` 把程序与捕获的常数放在一起。例如 Bessel 的节点表可能是捕获常数，而不是每次调用的参数。

| 文件或函数 | 实际责任 |
| --- | --- |
| `python/jaxlean/translate.py::transpile` | 调用 `jax.make_jaxpr`。随后调用 `translate`。 |
| `translate.py::translate` | 严格要求 JAX 0.8.0，调用 JAX 的 `check_jaxpr` 做结构检查。输入是 jaxpr 对象，不是打印文本。 |
| `translate.py::Emitter.program` | 把方程生成 Lean `let` 表达式。能内联支持的调用，静态展开 `scan`。 |
| `python/jaxlean/jaxpr.py` | 检查静态形状和类型，提供命名和类型表示。没有执行原语的职责。 |
| `python/jaxlean/importers/jaxpr.py::JaxprImporter.run` | 独立读取同一 jaxpr，生成 Lean 内部语法。它不解析另一条路径生成的函数文本。 |
| `python/jaxlean/certify.py::certify` | 组合函数、内部语法和逐输入相等的证明义务。返回源文件不等于已经验证。 |
| `certify_module` | 先处理纯、单输出的 `jit` 子调用，再让父证明使用子证书，保留函数边界。 |

`translate` 和 `transpile` 本身不自动产生证书。`certify` 不能处理任意已可翻译程序。单体 `certify` 不接受普通嵌套 `jit`；调用图应走 `certify_module`。

### 3.2 Lean 中的张量和解释器

[Core/Tensor.lean](../../../.research-inputs/jax-lean/JaxLean/Core/Tensor.lean) 把张量定义为“从有界坐标到数值的函数”。例如长度为 $n$ 的向量可理解为 $\{0,\ldots,n-1\}\to\mathbb R$。形状写在类型里，越界索引必须被排除。

[Core/Jaxpr.lean](../../../.research-inputs/jax-lean/JaxLean/Core/Jaxpr.lean) 定义 `DType`、`Var`、`Atom`、`Env`、`Op` 和 `Program`：数值类型、变量、输入项、变量环境、操作、程序。`Program.eval` 读输入，解释一条操作，绑定结果，再解释后续程序。实数、布尔值和有符号 Int32 在同一个类型化环境中。

浮点类型在这里映射到 $\mathbb R$。Int32 索引则保留有符号整数与回绕运算，不能说整个解释器“所有值都是实数”。[Core/RealOps.lean](../../../.research-inputs/jax-lean/JaxLean/Core/RealOps.lean) 把 `exp`、`log`、`sqrt` 等接到 Mathlib 的实数函数。实数实例使用 `noncomputable`：它提供精确数学含义，不承诺能把任意实数表达式直接执行成小数。

[Verification/JaxprRules.lean](../../../.research-inputs/jax-lean/JaxLean/Verification/JaxprRules.lean) 给局部解释规则。[Verification/Certificate.lean](../../../.research-inputs/jax-lean/JaxLean/Verification/Certificate.lean) 的 `jaxpr_certificate` 组合这些规则。tactic 是生成证明的指令；生成结果还须由内核检查。

真实例子是 [CertifiedMean.lean](../../../.research-inputs/jax-lean/examples/certificates/generated/CertifiedMean.lean)。它定义 `certified_mean_ir`，并声明 `certified_mean_translation_correct`，比较内部程序的解释与长度为 2 的均值函数。这里确认的是文件中的定义与证明结构，本次没有重新检查其编译结果。

### 3.3 证书之外还信任什么

JL 的证书陈述是：对固定形状的所有模型输入，$\operatorname{eval}(\operatorname{import}(J),x)=F(x)$。其外部边界包括：

- Python 导入器有没有正确记录原 jaxpr。两个生成路径仍共享 `layout.py`、`static_index.py` 的形状和索引解码；共享错误可能同时进入两侧。
- 原语的实数定义是否是我们要的 JAX 抽象。Lean 可检查类型与等式，不能自行确认这个选择。
- JAX 追踪、JAX 到编译器中间表示的转换、编译器和硬件执行。
- Lean 内核、采用的逻辑基础，以及应用定理中写出的假设。未来还应审计证明依赖，排除 `sorry` 这种“暂缺证明”的占位符和未经批准的新公理。

浮点字面量用精确有理数表示。例如存储的 float32 `0.1` 保留其实际二进制值，而不改成 $1/10$。中间运算却不舍入。实数除零与 `log` 等采用 Mathlib 的全域定义；“全域”意为在通常公式的合法域之外也有规定值。这不等于 JAX 的 NaN 或无穷语义。应用定理必须写明分母非零、对数参数为正等前提。

[test_certification.py](../../../.research-inputs/jax-lean/tests/test_certification.py) 有把除数从 2 改为 3、把乘法改为加法后要求证明失败的测试。它还专门展示 float32→float16→float32 的实际结果会改变，而实数恒等证书仍可成立。测试文件是边界说明的证据；本次没有运行，不能报告“已通过”。

### 3.4 形状固定和运算覆盖

追踪会固定数组形状、静态参数和许多 Python 循环次数。长度 2 的函数证书不覆盖长度 3。JL 的库里可以有对任意长度成立的数学引理，但把引理接到另一份生成程序仍需要新的连接证明。`vmap` 是对多个输入批量应用函数的 JAX 变换；JL 处理的是变换后的程序，不重建源码中的 `vmap`。

普通确定性翻译支持的主要清单如下。限制须连同名称一起看。前向值指原函数返回的值，不含导数；`stop_gradient` 保留这个值，但阻止自动微分沿该处传播。其证明影响见第 5.5 节。

| 类别 | 本地支持清单 |
| --- | --- |
| 实数运算 | `add`、`add_any`、`sub`、`mul`、`div`、`neg`、`square`、`integer_pow`；整数指数可为负。 |
| 比较与选择 | `abs`、`min`、`max`；`eq/ne/lt/le/gt/ge`；布尔 `and/or/xor/not`；布尔、双分支 `select_n`。 |
| 初等函数 | `exp/log/sqrt/rsqrt/sin/cos/tanh`。 |
| 形状操作 | 固定形状的 `reshape/transpose/broadcast_in_dim/squeeze/slice/rev/concatenate`。 |
| 求和与乘积 | `reduce_sum/reduce_prod`；非空归约域的 `reduce_min/reduce_max`；支持批次与多个收缩轴的 `dot_general`。归约就是沿指定坐标合并值。 |
| 索引与写入 | `gather` 只接受指定裁剪模式、Int32 索引及受限维度配置；`scatter/scatter-add` 只接受静态合法位置、最多 256 个更新元素。替换不能冲突；加法可累加冲突位置。 |
| 其他 | 浮点输出的 `iota`；受限类型转换；`copy`；前向值意义上的 `stop_gradient`。 |
| 调用与循环 | 纯 `jit`；只取前向程序的 `custom_jvp_call`；`scan` 静态展开，默认每个最多 32 次，可配置。 |

证书路径更窄：不支持 `scan`、`custom_jvp_call`、`reduce_prod`、捕获的浮点数组、多输出程序或随机操作。`while`、`cond` 两条路径都不支持。运行时 Int32 有部分支持，不代表任意整数宽度、整数位运算或浮点转整数均可用。一般 NaN、无穷和副作用被拒绝。副作用指打印、外部回调等可观察的外部行为。

代码有一个窄例外：删除标准 softmax 展开中的 `max(-inf,x)` 和 `min(+inf,x)` 恒等模式。它没有建立一般的扩展实数模型，不能因此接受 normix `log_prob` 的全部支持集分支。

### 3.5 随机数处理和许可

[rewrites/random_spec.py](../../../.research-inputs/jax-lean/python/jaxlean/rewrites/random_spec.py) 不是 PRNG 验证器。PRNG 是由种子确定输出的伪随机数生成器。设置 `random_model="uniform"` 后，JL 对 JAX 0.8.0 的 `_randint` 调用体做结构匹配，再换成理想的均匀整数分布。

key 是 JAX 显式传递的随机状态。允许的主要形态是一个有类型的标量 key、一个取样调用，或先做一次固定大小拆分，再对这些 key 做批量整数取样。匹配包含原语、参数、形状、类型和字面量，不仅看函数名。重复使用 key、多次取样、动态边界、普通数组式 `randint(shape=...)` 和其他分布都不在该模型内。

[Stdlib/FiniteLaw.lean](../../../.research-inputs/jax-lean/JaxLean/Stdlib/FiniteLaw.lean) 用有限样本空间上的质量函数表示概率。[Stdlib/Random.lean](../../../.research-inputs/jax-lean/JaxLean/Stdlib/Random.lean) 和 `Independent.lean` 提供理想采样、期望、方差及独立性规则。拆分后独立是显式选择的理想模型，不是对真实 Threefry 实现的结论。复用同一抽样值也不会被解释为独立抽样。

因此这些证明不能直接覆盖 `Gamma.rvs`、GIG 拒绝采样、PINV 连续均匀采样或混合模型的正态采样。拒绝采样会反复提议样本，按接受条件保留；normix 的 PINV 路径则通过数值分布表的反函数取样。对固定 seed 证明程序输出，与对随机 seed 下的分布证明，是不同目标。

本地根目录没有覆盖整个 JL 项目的 LICENSE，`pyproject.toml` 也未给出项目许可。存在 `examples/TENSOR_PUZZLES_LICENSE` 和 `examples/noether/LICENSE`，它们是局部材料的许可，不能代替项目许可。按本任务约束：复用或再分发前先询问作者；不复制其实现。这里不作未联网核实的法律结论，也不推定作者会选择哪种许可。

## 4. normix 与 JL 的缺口

下面是源码驱动的需求清单，不是已经生成的 jaxpr 清单。logpdf 指概率密度的对数；CDF 指累积分布函数。精确原语及嵌套形态都是后续追踪时的 `to verify`。尤其不能把 NumPy/JAX API 名直接当成原语名称。

| normix 位置与需要 | 候选原语/结构 | JL 现状与影响 |
| --- | --- | --- |
| `Gamma.var`、简单矩公式 | `integer_pow`、`div` 等 | 可作为最小入口；仍需 JAX 版本适配。 |
| `Gamma._log_partition_from_theta`、梯度、Hessian | `lgamma`、`digamma`、`polygamma` 或其展开 | 不支持。Gamma logpdf 不是“只加乘除”的例子。 |
| `Gamma.cdf`、`InverseGamma.cdf`、`utils/gammainc.py::gammaincinv` | 不完全 Gamma 函数、正态逆 CDF 的展开；可能出现 `igamma`、`erf`、`erfc`、逆误差函数等 | 缺特殊函数语义和导数定理。精确命名及 `ndtri/log_ndtr` 的展开待追踪。 |
| Bessel `_ab_coeffs/_g_k/_edge/_geometry` 及矩束 | `expm1`、`log1p`、`asinh`；`hypot/sinh/cosh` 或其展开；比较、选择 | 清单中没有这些函数的直接规则。部分函数可能展开到已有原语，也可能带内部封装。应检查完整子图。 |
| Bessel `_fori`；PINV `_bisect_w` | 静态 `fori_loop` 通常成为 `scan` | Bessel 二分为 40 次，PINV 为 200 次，超过默认 32；即使提高展开上限也没有循环证书。 |
| `fitting/solvers.py::_jax_newton_raw/_backtrack/_multistart_jax`；`distributions/gamma.py::_newton_digamma` | `while`，嵌套 `cond`，计数器与多个状态值 | 不支持。须定义有上限循环的语义和停止状态。 |
| `fitting/em.py::_fit_scan`、增量 EM | `scan`、多输出、状态元组；增量路径还含随机选择 | 不能直接导出整个 fitter。外层计时、Python 类型转换和结果裁剪也不属于纯 jaxpr。 |
| GIG `_cdf_or_ppf/_gig_rvs_devroye/from_expectation` | `cond` | 不支持；不能把分支调用一概当逐元素选择。 |
| `ExponentialFamily._implicit_natural_params` | `custom_jvp_call`；求导后另有线性代数 | 普通翻译只内联前向部分。导数是否正确不随之得到证明。 |
| 缓存的 GIG 与 Newton 调用 | `jit`，版本相关的 `pjit` 等封装 | JL 只显式处理 `jit`，没有通用 `pjit` 规则。出现哪个必须实测；不能直接改名字。 |
| Bessel `logsumexp`、矩束；EM 统计量 | `reduce_sum/max`、`exp/log`、广播、`dot_general`；可能有 `is_finite` | 前几项大多可覆盖。标准库展开中的有限性检查、无穷和其他保护分支仍可能阻塞。支持 softmax 不等于已支持此版本的 logsumexp。 |
| Bessel `_GL_T_J/_GL_LOGW_J` | 捕获的 float64 数组 | 普通翻译可写数组；证书导入器拒绝。要保留每个实际存储值。 |
| PINV `jnp.interp/cumsum`；混合模型数据索引 | `gather`、动态切片、`scan` 或累计归约 | `gather` 只有裁剪子集；`cumsum` 不因支持 `reduce_sum` 就自动覆盖。 |
| `mixtures/joint.py` 的矩阵 `.at[...].set`；GIG 初值 | `scatter/scatter-add` 或相关更新展开 | 静态、合法且足够小的更新可能支持；运行时位置、更新模式和计数必须逐项核对。 |
| 正态二次型、GIG 反演及隐式导数 | `triangular_solve/cholesky/eigh`、`custom_linear_solve` 等候选 | 不支持。需数学线性代数规则，不能把求解器当作普通除法。 |
| 支持集、退化边界、失败值 | `select_n/cond`、`is_finite`、NaN、$\pm\infty$ | 有限实数模型不足。先限制成功的内部域，另建失败状态或扩展值类型。 |
| `rvs`、增量 EM | random key、拆分、位操作、Gamma/normal/uniform/choice 等 | JL 的理想 randint 模型不能复用为这些分布的证明。 |

normix 的 [pyproject.toml](../../../pyproject.toml) 在仓库约束中锁定 JAX/jaxlib 0.9.1。JL 的 `translate` 会主动拒绝非 0.8.0。初步试验应锁定 normix 版本并适配导出器，不能为了迁就 JL 降级生产依赖。

还应按“函数、形状、dtype、backend、是否求导、是否批量化”记录覆盖。证明一个标量 `log_kv` 不自动证明整个 `vmap`、GIG `log_prob` 或 GPU 执行。后续可沿用 [test_no_host_callbacks.py](../../../tests/test_no_host_callbacks.py) 遍历嵌套 jaxpr 的思路，收集需求；该测试本身没有证明其他语义。

## 5. Bessel：能翻译什么，能证明什么

### 5.1 当前算法的数学对象

以下分解来自 [bessel.py](../../../normix/utils/bessel.py)、[常数定义](../../../normix/utils/constants.py) 和 [bessel_moment_kernel.md](../../tech_notes/bessel_moment_kernel.md)。积分表示在本地公开设计文档中引用 DLMF，即 NIST 的数学函数参考资料。本次没有核验外部原文或 Mathlib 是否已有对应定理。

对实数阶数 $\nu$ 和 $z>0$，目标是第二类修正 Bessel 函数 $K_\nu(z)$。采用积分表示 $2K_\nu(z)=\int_{\mathbb R}\exp(\nu u-z\cosh u)\,du$。

积分的峰值位置为 $u_0=\operatorname{asinh}(\nu/z)$。令 $\kappa=\sqrt{\nu^2+z^2}$，中心坐标为 $x=u-u_0$，再令 $a_c=(\kappa+\nu)/2$、$b_c=(\kappa-\nu)/2$。下标用于区别 GIG 的参数 $a,b$。中心指数为 $g(x)=\nu x-a_c(\exp x-1)-b_c(\exp(-x)-1)$。

于是 $K_\nu(z)=\tfrac12\exp(\nu u_0-\kappa)I$，其中 $I=\int_{\mathbb R}\exp(g(x))\,dx$。

代码分成几步：

1. `_as_arrays` 转成 float64、广播，并把 $z$ 下限夹到最小正规 float64。证明原数学公式时必须让这个保护不改变输入。
2. `_ab_coeffs` 用 `hypot` 算 $\kappa$。较小系数用 $z(z/(\kappa+|\nu|))$，避免直接相减，也避免先形成 $z^2$。
3. `_geometry` 对整数倾斜量 $k=-2,-1,0,1,2$ 找左右窗口。倾斜量表示给指数加上 $kx$，用来照顾指数矩。
4. `_edge` 从 `log1p` 构造初始上界。做 8 次带条件的加倍，再做 40 次二分，目标是 $g(x)+kx\le-40$。搜索上界截到 800，搜索用指数的坐标截到 $[-700,700]$。
5. `_nodes_and_logw` 在 $[x_{\rm lo},0]$ 和 $[0,x_{\rm hi}]$ 各放 96 个 Gauss–Legendre 节点。Gauss–Legendre 求积是用特定节点和权重近似区间积分的方法。节点数固定，但区间和节点位置随输入变化；它不是全局等距网格。
6. `_eval_quad` 以 `stop_gradient` 冻结窗口、节点及几何量，再用 `logsumexp` 累加。`logsumexp` 的数学含义是 $\log\sum_i\exp(q_i)$；数值实现通过减去最大值避免直接指数溢出。
7. `_moments_from_weights` 将权重归一化为 $p_i$，计算统计量 $s_i=(x_i,e^{-x_i}-1,e^{x_i}-1)$ 的均值 $\bar s$。矩阵第 $i$ 行为 $R_i=\sqrt{p_i}(s_i-\bar s)$。协方差按 $R^\top R$ 构造，再对称化。函数还计算参数导数。

节点及对数权重由 NumPy 的 `leggauss(96)` 和 `np.log` 在导入时产生。它们不是追踪内的 JAX 算法。后续证明可以固定本次表的实际值，但不能据此声称已经验证了 NumPy 生成表的算法。

float64 也是一个配置前提。[normix/__init__.py](../../../normix/__init__.py) 只在未启用 `jax_enable_x64` 时警告，并不代用户启用。要研究这里的 float64 常数，必须在导入 normix 前启用它，并记录实际捕获数组的类型；不能仅凭源码写了 `float64` 就认定运行值是 64 位。

### 5.2 固定次数循环并不等于数据无关

这里的 `fori_loop` 迭代次数固定。输入改变每轮的分支选择、区间端点和最终窗口。外层五个 $k$ 的 Python 循环在追踪时展开。按源码可数出 10 次 `_edge`，每次 48 次状态更新，共 480 次；这不是运行测量。

令 $c_j$ 表示第 $j$ 轮状态，$B$ 表示循环体。可把固定次数循环定义为 $c_{j+1}=B(c_j)$，递归应用指定的 $N$ 次，或生成有限展开。Lean 可以按 $j$ 归纳证明循环不变式。不变式就是每轮前后都成立的性质。相比完全展开，保留循环体和次数通常更适合读证明，也可避免文件和检查时间膨胀。

对 `_edge` 可先证明以下条件命题：如果二分前区间已包住目标交点，且比较用的截断指数与该区间上的目标指数相符，则每轮保留包围关系。40 次后二分区间宽度至多为初始宽度的 $2^{-40}$。因上界不超过 800，宽度至多为 $800\cdot2^{-40}$。

这还不能证明交点一定被包住。需要另证 8 次加倍足够、上限 800 不阻碍搜索，以及坐标截断不会破坏所需比较。当前 `_edge` 到达迭代上限后直接返回，没有返回“已成功建立包围区间”的状态。注释中“700 处已足够小”不能充当任意参数下的定理。

对未截断的 $g(x)+kx$，有 $g(0)+k0=0$，且二阶导数为 $-a_c e^x-b_c e^{-x}<0$。严格凹性可帮助证明两侧阈值交点的结构。这是本文的数学推导；形式化及截断对应仍待做。

真正的数据依赖退出循环在 Newton、回溯和 GIG 拒绝采样中。对有次数上限的 `while_loop`，建议定义带“剩余步数”的语义：每次先看条件；条件为假即返回；预算耗尽时明确区分正常结束和截断。然后证明它与原来的计数条件相符。预算不能凭空选成观察到的迭代次数。

例如 `_jax_newton_raw` 的预检查成功时不增加 `num_steps`，而是设置停止标志。证明循环有界时要考虑这个检查轮次，不能机械地把“数值步数上限”当“循环体调用次数上限”。这与证明 Newton 收敛到根仍是两件事。

### 5.3 第一层：程序表示了预期算法

现实的首批定理是：

- 导入的程序与一个保留实际常数、截断、选择和循环次数的实数算法相等。
- 对固定窗口和正权重，输出是指定的有限和及其对数。
- 归一化权重为正且总和为 1。
- `_moments_from_weights` 给出的中心协方差为半正定。半正定意为对任意向量 $h$，$h^\top C h\ge0$。因为它是非负权重下的平方和。
- 在明确的包围前提下，二分保持交点并给出端点距离界。

这些目标适合按周推进。它们可以发现符号、尺度、节点映射、循环状态或导数投影中的错误。它们不依赖一开始就有完整的 Lean Bessel 库。

需要同时保留边界：导出器本身仍有受信任部分；精确算术下的半正定不保证机器矩阵每个特征值非负；有终止上限不保证窗口满足尾部误差要求。

### 5.4 第二层：有限和与真正的 $K_\nu$ 有多近

“jaxpr 可翻译”只解决程序表达。要验证 Bessel 值，还须证明积分存在、变量替换成立、尾部和求积误差受控。

设返回窗口是 $[L,R]$，$L<0<R$。定义截断误差 $T=\int_{-\infty}^{L}e^{g(x)}dx+\int_R^{\infty}e^{g(x)}dx$。若 $g'(L)>0$ 且 $g'(R)<0$，严格凹性给出切线控制，进而可推得 $T\le e^{g(L)}/g'(L)+e^{g(R)}/(-g'(R))$。这条不等式由切线指数积分直接得到；本次未形式化。

因此端点指数下降 40，不等于积分相对误差为 $e^{-40}$。还需要尾部斜率和 $I$ 的下界。$k=\pm1,\pm2$ 的窗口有助于控制指数矩尾部；含 $x$、$x^2$ 的阶数导数还须相应的多项式权重界。

误差应分开记录：

| 误差项 | 要证明的内容 |
| --- | --- |
| 窗口与尾部 | 实际有限次搜索给出的 $L,R$ 满足适用的尾界。若只先假设包围成功，标题必须写“条件结果”。 |
| 理想求积误差 | 精确 Gauss–Legendre 节点和权重的有限和，与窗口积分的差。按文献记忆，经典余项涉及高阶导数；每面 96 点时是 192 阶。原文条件、具体库定理和可用界：`to verify`。 |
| 表常数误差 | 实际存储节点不是精确多项式根；存储的 `logw` 也不是精确权重的对数。须界定节点扰动和有效权重扰动。 |
| 对数常数误差 | `_LOG_TWO=np.log(2.0)` 是已舍入常数。在实数模型里它仍不等于精确的 $\log2$。 |
| 运行时浮点误差 | `hypot/asinh/expm1/logsumexp`、求和和矩阵乘法等的实际舍入。此项不在前四项之内。 |

一个可审查的定理目标如下。令 $Q_{\rm tab}>0$ 是保留实际节点及存储对数权重后、在实数上计算的中心求积和。令 $c_2$ 是存储的 `_LOG_TWO`。在保护分支不改变数学输入、几何恒等式成立的域内，程序的实数值为 $L_{\rm alg}=\nu u_0-\kappa+\log Q_{\rm tab}-c_2$。

若已证明 $I\ge m>0$，并证明尾部、理想求积和表常数误差之和至多为 $\varepsilon m$，其中 $0<\varepsilon<1$，则可以得到 $|L_{\rm alg}-\log K_\nu(z)|\le-\log(1-\varepsilon)+|c_2-\log2|$。

这个陈述把真实算法与理想公式分开了。不能把 float 表导入为有理数后，就直接调用“精确 Gauss–Legendre”的误差公式。也不能用一个占位公理宣称 $Q_{\rm tab}=I$。

建议先在待 Xiang 确定的小域上研究，例如 $|\nu|\le10$、$0.1\le z\le100$。这只是试验域，不是现有包的新承诺。代码写明测试过 $|\nu|\le300$，声明数值域为 $z\ge10^{-10}$，并注明 $(0,10^{-300})$ 在 192 点下未解决。测试域没有给出对任意大的 $z$ 的统一误差定理。

实数有限和、权重和协方差性质是现实工程目标。包含所有窗口保护、实际表常数、整个参数域及二阶导数的有效误差界，属于研究工作。经典求积余项即使能写出来，也可能大到没有实际用途。可以另研究可验证的分区积分包围，但这也需要实质分析。

固定 192 点的精度，与“节点数趋于无穷时收敛”也不同。证明节点加密只收敛到截断区间积分；若要趋于整个 $K_\nu$，还须让尾误差趋零并处理固定窗口上限。现有常数不能直接支持这种渐近结论。

### 5.5 `stop_gradient` 与隐式导数必须另证

AD 是自动微分，即用运算的导数规则生成导数程序。JVP 是 Jacobian-vector product，即导数矩阵作用于一个方向向量。`custom_jvp` 允许用户指定该规则。用户指定不等于规则正确。

`stop_gradient` 的前向值等于输入，但 AD 把它的导数设为零。例如示意表达式 `stop_gradient(x) * x` 的前向值是 $x^2$，JAX 规则给出的导数是 $x$，普通数学导数却是 $2x$。所以把 `stop_gradient` 翻译为恒等后，再证明生成函数的普通导数，可能证明错对象。

Bessel 应显式引入几何参数 $\gamma$，它包含参考输入、中心、节点及权重。定义两参数实数函数 $F(\gamma;\nu,z)$。生产前向值是 $F(\gamma(\nu,z);\nu,z)$，AD 则在当前参考点固定 $\gamma$，计算后两个参数的偏导。

在几何恒等式成立时，可写成 $F(\gamma;\nu,z)=\log\sum_i w_i\exp(\nu u_i-z\cosh u_i)-c_2$，其中 $w_i>0$、$u_i$ 均固定。令 $p_i$ 为归一化后的权重，则 $\partial_\nu F=\sum_i p_i u_i$、$\partial_z F=-\sum_i p_i\cosh u_i$。二阶导数由对应统计量的协方差给出。

这是 `_eval_quad` 和矩束之间可证明的离散恒等式。它不自动等于“重新选择窗口和节点的整个前向函数”的普通导数，也不自动等于连续积分的导数。后者需要换序条件及带权积分误差界。数值值误差小也不能推出导数误差小。

[examples/autodiff/proofs/Jvp.lean](../../../.research-inputs/jax-lean/examples/autodiff/proofs/Jvp.lean) 确实给二次函数另证了数学导数，再连接生成的 `grad` 和 JVP 程序。它没有证明 JAX 的整个 AD 变换。normix 可以采用同一证明分工，不需要复制实现。

`ExponentialFamily._implicit_natural_params` 又是另一层。这里 $\theta$ 是自然参数，$\eta$ 是期望参数，$\psi$ 是对数配分函数。若 $\psi$ 在内部邻域二阶连续可微，$\theta$ 精确满足 $\nabla\psi(\theta)=\eta$，且 $H=\nabla^2\psi(\theta)$ 可逆，局部反函数的导数应满足 $H\,d\theta=d\eta$。这是拟采用的数学条件，具体 Lean 定理仍待核验。代码停止对优化历史求导，再用对角尺度平衡、特征值检查及 Cholesky 求解这个线性系统；失败或边界活跃时返回 NaN 导数。

可先证明“在成功、内部且矩阵可逆的条件下，返回切向量解了该线性系统”。不能由此宣称有限次 Newton 返回值是精确根，或其自定义导数就是截断迭代程序的普通导数。对于 GIG，$H$ 又来自离散矩束，还需控制它与连续模型 Hessian 的差，以及未消失的方程残差。

### 5.6 对 logpdf、CDF 和采样的意义

Gamma 的内部 logpdf 为 $\alpha\log\beta-\log\Gamma(\alpha)+(\alpha-1)\log x-\beta x$。GIG 在 $x,a,b>0$ 时为 $(p-1)\log x-(b/x+ax)/2-\log2-\log K_p(\sqrt{ab})-(p/2)(\log b-\log a)$。

若把 GIG 的真实 $\log K$ 换成近似值，logpdf 的误差可直接从该项传递。但近似密度并非因此严格积分为 1。还须处理 `_gig_degeneracy_flags` 选择的 Gamma/InverseGamma 极限公式及 `_unpack_safe` 的保护。仅有 $a,b>0$ 不保证走内部 Bessel 分支。

`ExponentialFamily.log_prob` 在支持集外和端点包含无穷。即使定理只要求 $x>0$，原始 jaxpr 仍可能携带未选分支中的无穷常数。不能把一句 Lean 前提当作 JL 已会删除该分支。第一版可证明内部公式；连接实际 API 时需有受前提约束的分支化简证明，或引入明确的扩展值类型。

CDF 是累积分布函数。`utils/rvs.py::build_pinv_table` 的梯形积分表、累计和与插值可先证单调性、端点及非负权重。它们与真实 CDF 的距离仍需积分、截断和插值误差。文件中的 `tail_eps` 用作密度高度阈值，不能不经尾界证明就当成尾概率界。

采样还须定义概率空间。可先假设输入是理想独立均匀变量，证明逆 CDF 或拒绝采样的分布，再研究实际 PRNG。当前 GIG 拒绝采样有最大轮数，耗尽时产生 NaN。拟证明应包含成功条件下的分布和失败概率界，而非无条件返回精确 GIG 样本。连续分布证明不受 JL 有限 randint 范例自动覆盖。

## 6. 浮点差距怎么处理

下面对浮点证明工具路线的描述来自文献记忆，未在本地验证工具兼容性或现成库覆盖。它们都不能靠安装一个工具就自动验证 JAX 编译后的 Bessel。

| 路线 | 可得到什么 | 成本与限制 |
| --- | --- | --- |
| 实数证明 + 高精度差分测试 | 精确模型性质，以及有限输入上的实际误差证据 | 首选。差分测试就是在相同输入上比较不同实现。它没有全域舍入保证。 |
| 区间算术 | 用包含真值的上下界计算；若每步向外舍入有保证，可包住结果 | 适合固定域、常数表和局部表达式。需可靠的指数、对数等区间界；参数相关性会让区间过宽。小型验证试验约 2–6 人周。 |
| Flocq 式浮点模型 | 明确建模格式、舍入、误差传播及异常情况 | Flocq 属于另一种证明助手 Coq，不是可直接接到 Lean/JAX 的插件。这里只借用其思路。小核需数月；全面覆盖目前不能可靠估工期。 |

推荐先使用现有 [scripts/gen_bessel_reference.py](../../../scripts/gen_bessel_reference.py) 和 [tests/data/bessel_reference.json](../../../tests/data/bessel_reference.json) 的结构。该生成器调用 mpmath 的高精度 Bessel 值和导数；mpmath 是任意精度数值库，不是自动给出严格误差包围的证明器。以后提高参考精度并检查结果稳定性，也仍是数值证据。

[test_bessel_contract.py](../../../tests/test_bessel_contract.py) 已区分参考值与 AD/矩束一致性：多数字段用 $10^{-11}$ 相对阈值，部分阶数导数字段用 $10^{-7}$，对称零点用绝对阈值。这些是代码中的测试标准，不是本文重新测得的误差。

未来的 3–5 人日数值验证扩展可覆盖窗口上限附近、小 $z$、大 $z$、$\nu=0$、半整数阶数、退化分支附近。也要比较逐操作执行的 eager 模式、JIT 编译、vmap 批量调用和用户选定设备。agent 可写输入生成器、误差表与故意改错的反例检查；Xiang 须审核参数域、参考精度和容差是否有统计与数值意义。这个估算不含 GPU 资源等待。

浮点证明还会遇到实数没有的问题：舍入改变窗口比较结果，归约顺序改变，GPU 初等函数精度不同，编译器可能融合运算。只给每次加法附上一个误差符号不够。若以后要证明此层，必须明确格式、次正规数、溢出、特殊函数误差契约及允许的执行方式。

不改变 S5、S8、S9、S10、S11 的既有约束。JAX 路径仍由用户选择的设备执行；验证工具在开发环境旁路运行，不进入生产调用。不用 Lean、NumPy 或 CPU callback 替换 JAX 运算。用户选择 CPU 时，也不能因为 JAX 路径有定理就宣称 CPU 执行已验证。

## 7. 四种设计选择与本次建议

估算中的一人日约为 6 小时有效工作，含 agent 产物的审阅。默认有一名熟悉 Lean 的工程人员协助。Xiang 从零学习和独自维护会拉长日历时间。下表均不含 Bessel 全域误差界。fork 指保留上游代码，再建立自己维护的版本。

| 方案 | 成本估算 | 信任边界与收益 | 本次判断 |
| --- | --- | --- | --- |
| A：直接使用 JL | 复现范例 1–3 人日；覆盖 normix 当前目标无可靠短工期 | 已有解释器和证书机制。但锁定 0.8.0，特殊函数、循环证书和常数表均阻塞。 | 不作为主路线。保留作比较基线。 |
| B：获得许可后 fork 并扩展 | 首个有用子集约 3–6 人周；Bessel 前向连接再约 4–8 人周 | 可沿用张量、索引和证书。仍信任导入与布局，承担上游变化及已有架构成本。 | 若作者愿合作、许可适合且小循环方案一致，可重新考虑。现在不能复制或再分发。 |
| C：独立小型导出器 | 标量证书原型 1–2 人周；固定向量和 Bessel 前向目标累计约 5–10 人周 | 只支持确有需要的语义，方便锁定 0.9.1。自己负责导入器、证明规则和拒绝边界。代码新写不代表少了证明义务。 | 推荐，但以两周试验和停止条件约束。 |
| D：手写 Lean 算法模型 + 差分测试 | 有用的有限和/协方差模型约 1–3 人周 | 可先做数学。Python 与模型的一致性依赖审阅和测试，没有 jaxpr 翻译证书。 | 作为 C 的前置数学模型及退出路线。不能冒充源程序已验证。 |

### 7.1 C 应有多小

Python 端读取 `ClosedJaxpr`，每个原语产生一项类型明确的 Lean 表达式。Lean 端有一份小型程序语法和实数解释器。可读函数与解释器之间的等式单独检查。最初只接受标量实数、布尔值和已确认的循环计数类型。

随后加入 `Fin n → ℝ` 形式的固定向量。`Fin n` 表示小于 $n$ 的非负整数索引。Bessel 虽然输入、输出可为标量，中间却有 192 个节点和矩束；永久坚持“只有标量”会迫使我们把数组铺成巨大表达式。暂不做任意秩张量、动态形状或通用 scatter。

每种新原语都必须有输入条件、数学定义和拒绝条件。例如实数 `expm1` 可定义成 $e^x-1$，`log1p` 可定义成 $\log(1+x)$，但这只描述目标值，不证明机器实现的数值稳定性。Gamma 的 `lgamma` 可先在正参数域解释为 $\log\Gamma$；其余域和导数是另外的义务。Mathlib 对这些函数的现有定义与引理名均为 `to verify`，不应由 agent 猜名字补齐。

常数表以实际二进制值对应的有理数导入，记录形状和内容摘要。初始数学证明可参数化于任意正权重；随后单独验证捕获表满足条件。把常数改成输入参数只证明了一个参数化函数，还要补“这些输入正是原程序捕获值”的连接。

`cond` 表示选择一个子程序执行；`select_n` 表示选择已经算出的值。两者不能简单互换，特别是未选分支可能含非法输入或异常时。`while` 先只支持有可证次数上限的纯循环。未知原语、外部副作用、未支持的类型和失败语义都明确拒绝。

每份产物记录：源码提交、函数入口、静态配置、JAX/Lean/Mathlib 版本、输入形状和类型、嵌套原语清单、捕获常数、假设及未证明项。源码或常数变更后重新生成。可读性与来源记录可以让 Xiang 审核“到底证明了哪个程序”。

两条生成路径仍可能共享错误。证书主要约束代码生成，不证明 Python 对象的读取过程。需独立的原语对照和故意破坏测试；若今后把形状解码搬到 Lean 并证明其性质，可进一步缩小信任边界。

### 7.2 设计决定与否决记录

以下记录日期均为 2026-10-24。除第一行的用户约束外，其余是本报告作者的建议，待 Xiang 决定；“否决”不冒称已有 owner 批准。

| 事项 | 状态、理由和证据 | 重开条件 |
| --- | --- | --- |
| 保持 JAX 原生与 backend 选择 | 用户已明确要求；本地设计 S5/S9/S11 也规定不隐藏 host callback。验证设施不改变运行路径。 | 只有 Xiang 明确更改契约。 |
| C 为试验路线，D 提供数学模型和退路 | 建议。JL 的版本、常数表和循环证书缺口已由本地代码确认。 | 两周试验没有一项与 normix 相连的可检查结果，则优先转 D。 |
| A 直接覆盖 Bessel | 本报告否决。提高 `scan` 上限不足以补证书、特殊函数、常数及误差分析。 | 实际覆盖变化并完成版本适配。 |
| 立即复制 JL 建 fork | 本报告否决。没有可确认的项目许可。 | 作者明确许可，双方确定维护范围。 |
| 为导出而改变生产公式或减少循环次数 | 本报告否决。会证明另一个程序，且影响现有数值契约。 | 另立算法改动任务，有独立数值理由和 owner 决策。 |
| 给 Bessel 加一个“返回正确 K”的公理 | 本报告否决为实现验证。它把最难的问题藏进前提。参数化的外层定理可以保留，但必须写明条件。 | 该边界已有独立证明或明确只做条件定理。 |
| 全面实现 IEEE 浮点与编译正确性 | 暂缓，未实验。范围远超原型，也没有本地现成衔接。 | 小核实数证明已产生价值，且具体使用场景确需保证误差。 |
| 只手写模型并称“normix 已形式化验证” | 本报告否决该表述。手写模型可保留，一致性缺口必须公开。 | 补齐源码连接证明，或明确缩小声称范围。 |

## 8. 可停止的里程碑

这些都是未来工作，不是本次已执行的命令或已完成证明。先批准约两周的 M0/M1 试验，再决定是否投入 Bessel。工期按增量估算；M2 的内部 Bessel 连接明确依赖后面的 M3/M4。

| 阶段 | 交付与验收 | 估算与分工 | 停止或改道条件 |
| --- | --- | --- | --- |
| M0：复现基线、盘点实际图 | 在隔离环境复现 JL 的均值、二次函数 AD、短 scan。区分普通生成与证书。未来运行 `lake build JaxLeanExamples` 才能检查应用证明；仅 `lake build` 不够。另在 normix 0.9.1 递归列出候选函数原语与常数。 | 1–3 人日。agent 准备记录、追踪与版本差异；Xiang 用 1–2 小时审查边界。许可问题与作者确认；不把 JL 源码纳入新项目。 | 3 人日仍无法得到清楚的版本/构建诊断，就停止修上游环境，保留静态比较，转自己的最小模型。 |
| M1：标量与 Gamma 内部 logpdf | 先连接实际 `Gamma.var`。再支持正参数 `lgamma`，导出内部 logpdf。证明公式相等与参数域，不承诺完整支持集。故意换错除数或 Gamma 项时证明必须失败。实际 `log_prob` 的无穷分支须单独说明。 | 4–8 人日。agent 写导出器、规则和证明候选；Xiang 约 3–5 小时审核公式、域及公理清单；Lean 熟手审核证书结构。 | 两周总预算内还只有生成文本，没有检查过的证书和至少一条性质，转 D。不用删除前提或加入公理来过关。 |
| M2：GIG logpdf 的外层 | 以明确的 $\ell_K$ 参数表示 Bessel 正规化项，证明内部公式组合及 logpdf 误差传播。记录其对应 `GeneralizedInverseGaussian._log_partition_from_theta` 与 `ExponentialFamily.log_prob`。清楚标为条件结果；在 M4 后补实际调用连接。 | 3–5 人日。agent 做公式与分支前提；Xiang 约 2–3 小时审查自然参数顺序、符号、退化标志与保护下限。 | 若需同时重建全部退化、CDF 和 solver，缩回内部域。不能把抽象 $\ell_K$ 当成已验证的 Bessel 实现。 |
| M3：Bessel 被积函数和固定节点 | 对 `_g_k` 的未截断内部域与 `_eval_quad` 的固定几何部分，证明中心化恒等式、权重归一化、协方差半正定。补实际常数数组导入。连接冻结几何 AD 与矩束；积分导数先不声称。 | 5–10 人日。agent 实现有限向量、`expm1` 等语义并提交证明；Xiang 约 4–6 小时审查几何、尺度和导数对象。 | 若只有人工抄写公式而未连接真实子图，降级标为 D；若需要大张量框架，先限定一维节点。 |
| M4：循环与完整前向 Bessel | 给固定次数 scan/递归加证书，保留 8、40 和 192。先条件证明 `_edge` 不变式，再分析小域内的包围成功。完成 `log_kv(..., backend='jax')` 实数前向连接，并回接 M2。动态 while 仅另做一个有界实例。 | 10–20 人日。agent 做循环规则、来源记录、反例试验；Xiang 与 Lean 审阅者共约 6–10 小时审查终止、截断和失败含义。 | 若指定环境中单个标量前向证书反复超过约 10 分钟或内存预算，先改模块化证明；两次设计迭代仍失控，停止整图展开。阈值是项目管理建议，不是性能测量。 |
| M5：有效误差界 | 在一个明确小域证明尾界、求积误差、表常数误差和对数误差。先证明值，后另做导数。再评估扩到声明域。 | 30–80 人日研究预算，可能更长。agent 整理分析、搜索本地库并写证明；Xiang 负责主要数学路线，每周约半天至一天，并需数值分析/Lean 审阅。 | 先给 5–10 人日分析预研。若无法得到非空且有用的界，停止承诺生产精度，保留已完成的实数结构性质。 |

M0–M4 大约是 5–10 人周的受控工程试验，不包括所有 CDF、随机算法或 EM。M5 的成本不应和前几项混在一起。对 Xiang，先用约 2–4 个半天学习如何读定理的量词、假设、定义和证明依赖；不要求一开始能写长证明。agent 可以多次尝试证明，Xiang 仍需决定“这个命题是否值得证明、前提是否覆盖使用场景”。

验收必须分别列出：已检查证书、已检查数学性质、仍是假设的算法条件、实测数值误差、没有覆盖的 API/设备。源码中存在证明文件、Python 测试通过、Lean 接受空缺证明的文件，都不够代表验收完成。

## 9. 与 srush 合作是否值得

值得发一个小范围技术问题。理由是双方需求互补：JL 已经探索类型化表示、逐程序证书和函数边界；normix 能提供特殊函数、固定循环和非标准导数语义的真实案例。合作价值不取决于先引入其代码。

最合适的贡献顺序是：

1. **原语与常数表实例。** 一个独立编写的标量 `expm1/log1p` 例子和捕获 float 数组例子。约 1–3 人日准备。
2. **有界循环证书。** 固定次数二分，不涉及 Bessel 全部数学。双方认可接口后约 1–3 人周研究。
3. **冻结几何与导数规格。** 一个前向等价但 AD 含义不同的最小例子，然后才连接 `custom_jvp` 和隐式反演。约 2–4 人日准备可讨论材料；完成规则的工期另估。
4. **Bessel 应用。** 作为导出器之上的应用证明，不要求作者承担完整特殊函数分析。

这些时间与路线图有重叠，不额外累加。agent 可整理最小例子、支持矩阵和草稿。Xiang 须审查共享内容、许可诉求、维护承诺，并亲自决定发送。这里没有联系作者，也没有声称外部评审已完成。

风险也明确：小型研究仓库的维护承诺未知；JAX 和 Lean 的版本变化会带来迁移；通用张量任务的优先级可能高于特殊函数；上游接受一个新 primitive 不等于接受它的数学规约。输入说明中的“12 个提交”只能说明提供时的规模描述，不能推断作者意愿或技术质量。

建议先发 GitHub issue。许可和支持范围是公共问题，讨论结果便于后来引用。若作者明确偏好邮件，或涉及尚未公开的研究安排，再发短邮件；不要同时重复发送。离线状态下不猜作者邮箱或当前联系方式。

下面是供 Xiang 审阅的英文草稿，未发送：

```text
Title: Licensing and a small scientific-computing extension experiment

Hi Sasha,

I maintain normix, a JAX-native library for generalized hyperbolic
distributions. I have been reading a local copy of jax-lean and am
interested in its separation of real semantics and translation certificates.
I have not reproduced the examples yet.

Our current log-Bessel implementation uses centered quadrature, fixed
8-step bracket expansion and 40-step bisection, captured floating-point
node tables, and stop_gradient for the quadrature geometry. Its implicit
parameter-inversion derivative uses custom_jvp elsewhere in the library.
We use JAX 0.9.1.

I did not find a project-wide license in the copy I reviewed. Could you
clarify the intended license before we reuse or redistribute any code?

Would a small, independently written example around captured constants
and fixed-length scan certificates be useful? We could also contribute
examples that distinguish forward-value equivalence from frozen-geometry
derivative claims. We would start with real arithmetic and make no claim
about floating-point or PRNG correctness.

Would you prefer an issue or a short design note before any implementation?
No commitment is expected; a narrow scope recommendation would help.

Thanks,
Xiang
```

## 10. Lean、Hypothesis、TLA+ 的分工

本报告的主线是 jaxpr→Lean。为回应概览留下的问题，这里只给与本路线直接相关的分工。以下工具定位是文献与使用知识的记忆，未在此运行验证。

Hypothesis 是 Python 的性质测试工具。它生成输入、执行真实代码，再缩小失败例子。Lean 证明对给定数学模型和前提下所有输入成立。TLA+ 是描述状态及状态之间变化的规约语言；TLC 是常用的模型检查器，会枚举指定有限模型中的可达状态。有限模型检查不等于任意规模的数学证明。

| 问题 | 合适入口 | 不能从结果外推什么 |
| --- | --- | --- |
| 固定权重协方差为何半正定；Bessel 求积误差 | Lean | 不自动覆盖 JAX 舍入和导出器。 |
| CPU/JAX、JIT/vmap 与 mpmath 是否在某些参数下不一致 | Hypothesis 或现有参数化数值测试 | 没找到反例不是全域正确。 |
| EM 停止后是否保持已接受模型；内层失败是否保留最后有限模型 | 小型 TLA+ 状态模型，也可用 Hypothesis 状态测试连接代码 | 数值步骤若被抽象为成功/失败，模型不会证明这一步实际收敛。 |

例如 [BatchEMFitter._fit_scan](../../../normix/fitting/em.py) 有 `converged/diverged/solver_failed` 三个标志，并用 `done` 冻结模型和历史。可用 TLA+ 检查“停止后状态不反转”和“失败更新不覆盖旧模型”。这不是并发系统也能建模。但若仅有这段顺序流程，用 Lean 或直接状态测试可能更省维护成本。

不建议为了这个 jaxpr 试验同步建立大型 TLA+ 工程。若 Xiang 关心失败回退，可限 2–4 人日做一个模型：agent 提取状态和转换、构造反例；Xiang 用约 2 小时审核抽象是否漏掉真实行为。若模型只复述代码而没有独立的安全性质，就停止。所谓安全性质，是“某种错误状态永远不可达”，例如失败的新模型被当成成功返回。

EM 似然不降仍需 E/M 步的数学条件。有限求解、近似 Bessel、正则化及失败回退都应进入适用范围说明。不能用“状态机没有坏状态”替代 EM 数值收敛证明，也不能用 EM 单调性直接推出参数收敛或全局最优。

## 11. 排序建议

1. **选 C，限期两周做 M0/M1。** 自写最小导出器和证书，先连实际 `Gamma.var`，再做 Gamma 内部 logpdf。保持 JAX 0.9.1、用户 backend 和现有算法。只有产生可检查结果才扩到 Bessel。
2. **并行准备 D 的小型数学模型。** 优先固定几何的 logsumexp、矩与协方差，以及条件二分不变式。约 1–3 人周。它既支持 C，也能在导出器成本失控时留下可用成果；明确保留模型与源码的一致性缺口。
3. **联系 srush，询问许可及窄范围合作。** 准备问题约半个人日，最小案例 1–3 人日。许可和方向一致后，B 才可能比 C 更省成本。不把作者回复当作路线开始的前提。
4. **M4 后再决定 M5。** 先用 5–10 人日研究一个小域的有效误差界。浮点全链路证明、连续采样器和完整 EM 导出暂缓。A 只保留为范例基线。

各项 agent 都能承担机械生成、证明搜索和失败案例整理。Xiang 必须审核命题、参数域、近似层次与被排除情形。M5 需要他直接参与数学分析。两者的职责不能用“Lean 已通过”互相替代。

## 12. 给 Xiang 的开放问题

- 第一项成果更看重什么：固定节点的矩/导数一致性，还是直接由实际 `Gamma.log_prob` 得到的函数证书？前者数学收益更早，后者暴露支持集和特殊函数接口问题。
- 能否接受先验证成功的内部域，把 NaN、无穷、退化边界和失败状态列为独立任务？这决定最小语义库的规模。
- Bessel 第一块误差证明应覆盖哪个业务参数域、哪个量和多严的误差？值、阶数导数和参数 Hessian 的难度不同。
- 两周试验之后，是否愿意给 M0–M4 约 5–10 人周，并安排 Lean 熟手审阅？若不能，D 加数值性质测试更合适。
- 若上游许可合适，你愿意维护通用循环/常数支持，还是只提供 normix 的应用和案例？这决定 C 与 B 的取舍。
- 首轮是否只验证确定性计算？采样分布与实际 PRNG 需要单独的概率规格和预算。

## Sources

以下路径均为本次静态阅读依据。它们支持源码、规格和现有测试内容的陈述，不表示测试已经运行。外部论文、工具现状及 Mathlib 覆盖没有在此核验。

| 本地来源 | 本报告使用范围 |
| --- | --- |
| [AGENTS.md](../../../AGENTS.md)、[dev-notes/README.md](../../README.md)、[ARCHITECTURE.md](../../ARCHITECTURE.md) | 仓库规则、文档边界、模块与接口。 |
| [.research-inputs/overview_report.md](00_overview_report.md) | 起始判断；其中社交媒体、论文和运行结果没有作为本次独立核实的事实。 |
| [JL README](../../../.research-inputs/jax-lean/README.md)、[ARCHITECTURE](../../../.research-inputs/jax-lean/ARCHITECTURE.md)、[SUPPORTED_JAXPR](../../../.research-inputs/jax-lean/SUPPORTED_JAXPR.md)、[SEMANTICS_REFERENCES](../../../.research-inputs/jax-lean/SEMANTICS_REFERENCES.md)、[primitive-index](../../../.research-inputs/jax-lean/docs/primitive-index.md) | 架构、覆盖与信任边界。语义参考文档也明确说引用不是一致性证明。 |
| [JL translate.py](../../../.research-inputs/jax-lean/python/jaxlean/translate.py)、[jaxpr.py](../../../.research-inputs/jax-lean/python/jaxlean/jaxpr.py)、[certify.py](../../../.research-inputs/jax-lean/python/jaxlean/certify.py)、[importers/jaxpr.py](../../../.research-inputs/jax-lean/python/jaxlean/importers/jaxpr.py)、[layout.py](../../../.research-inputs/jax-lean/python/jaxlean/layout.py) | 导出、独立导入、版本门槛、常数和布局处理。 |
| [JL random_spec.py](../../../.research-inputs/jax-lean/python/jaxlean/rewrites/random_spec.py)、[FiniteLaw.lean](../../../.research-inputs/jax-lean/JaxLean/Stdlib/FiniteLaw.lean)、[Random.lean](../../../.research-inputs/jax-lean/JaxLean/Stdlib/Random.lean) | 理想随机模型及其限制。 |
| [JL Core/Jaxpr.lean](../../../.research-inputs/jax-lean/JaxLean/Core/Jaxpr.lean)、[Tensor.lean](../../../.research-inputs/jax-lean/JaxLean/Core/Tensor.lean)、[RealOps.lean](../../../.research-inputs/jax-lean/JaxLean/Core/RealOps.lean)、[Certificate.lean](../../../.research-inputs/jax-lean/JaxLean/Verification/Certificate.lean) | 类型、解释器、实数模型和证书机制。 |
| [JL basics/code.py](../../../.research-inputs/jax-lean/examples/basics/code.py)、[CertifiedMean.lean](../../../.research-inputs/jax-lean/examples/certificates/generated/CertifiedMean.lean)、[autodiff/proofs/Jvp.lean](../../../.research-inputs/jax-lean/examples/autodiff/proofs/Jvp.lean)、[RandomProgramProofs.lean](../../../.research-inputs/jax-lean/examples/random_program/proofs/RandomProgramProofs.lean) | 短 scan、真实证书、AD 与概率应用范例。 |
| [JL test_translation.py](../../../.research-inputs/jax-lean/tests/test_translation.py)、[test_certification.py](../../../.research-inputs/jax-lean/tests/test_certification.py)、[test_common_primitives.py](../../../.research-inputs/jax-lean/tests/test_common_primitives.py)、[test_random_translation.py](../../../.research-inputs/jax-lean/tests/test_random_translation.py) | 已编写的覆盖、拒绝和故意破坏测试；未运行。 |
| [JL pyproject.toml](../../../.research-inputs/jax-lean/pyproject.toml)、[lean-toolchain](../../../.research-inputs/jax-lean/lean-toolchain)、[lakefile.toml](../../../.research-inputs/jax-lean/lakefile.toml)、[局部 puzzle 许可](../../../.research-inputs/jax-lean/examples/TENSOR_PUZZLES_LICENSE)、[局部 Noether 许可](../../../.research-inputs/jax-lean/examples/noether/LICENSE) | 固定工具版本及局部许可。副本声明 Lean/Mathlib v4.33.0；可取得性与运行状态未核验。 |
| [normix/utils/bessel.py](../../../normix/utils/bessel.py)、[constants.py](../../../normix/utils/constants.py)、[exponential_family.py](../../../normix/exponential_family.py) | Bessel 全部现行步骤、常数、支持集和隐式 JVP。 |
| [gamma.py](../../../normix/distributions/gamma.py)、[generalized_inverse_gaussian.py](../../../normix/distributions/generalized_inverse_gaussian.py)、[inverse_gamma.py](../../../normix/distributions/inverse_gamma.py)、[inverse_gaussian.py](../../../normix/distributions/inverse_gaussian.py) | 分布接口、特殊函数、退化与采样需求。 |
| [mixtures/joint.py](../../../normix/mixtures/joint.py)、[mixtures/marginal.py](../../../normix/mixtures/marginal.py)、[fitting/solvers.py](../../../normix/fitting/solvers.py)、[fitting/em.py](../../../normix/fitting/em.py)、[utils/rvs.py](../../../normix/utils/rvs.py)、[utils/gammainc.py](../../../normix/utils/gammainc.py) | 线性代数、循环、状态、CDF 和插值。 |
| [docs/theory/gig.md](../../../docs/theory/gig.md)、[docs/theory/em_algorithm.md](../../../docs/theory/em_algorithm.md)、[docs/design/solvers_and_bessel.md](../../../docs/design/solvers_and_bessel.md) | 已发布的数学目标和算法说明；不能未经审阅直接当形式化规约。 |
| [dev-notes/design/design.md](../../design/design.md)、[solvers_and_bessel.md](../../design/solvers_and_bessel.md)、[bessel_moment_kernel.md](../../tech_notes/bessel_moment_kernel.md)、[gig_eta_to_theta.md](../../tech_notes/gig_eta_to_theta.md) | S5/S8–S11、拒绝的 backend 方案、现行 Bessel 与部分旧说明。 |
| [test_bessel_contract.py](../../../tests/test_bessel_contract.py)、[test_no_host_callbacks.py](../../../tests/test_no_host_callbacks.py)、[gen_bessel_reference.py](../../../scripts/gen_bessel_reference.py)、[bessel_reference.json](../../../tests/data/bessel_reference.json)、[normix pyproject.toml](../../../pyproject.toml) | 测试阈值、现有图遍历、高精度参考来源和 JAX 版本。 |

外部背景仅列确信的名称与年份，不给未经核验的引理名：

- Boldo、Melquiond，*Flocq: A Unified Library for Proving Floating-Point Algorithms in Coq*，2011。文献记忆；这里只用于说明浮点建模路线。
- Lamport，*Specifying Systems*，2002。文献记忆；用于 TLA+ 的状态与行为规约定位。
- Dempster、Laird、Rubin，*Maximum Likelihood from Incomplete Data via the EM Algorithm*，1977。本地概览及理论文档也有引用；本次未核验原文，不借此声称当前实现满足所有条件。

DLMF 的 Bessel 积分表示、Gauss–Legendre 误差分析、Lean/Mathlib、Hypothesis、mpmath 的外部资料在本次均未联网核实。Mathlib 中 Bessel、求积余项、特殊函数导数及反函数所需的确切定义与引理：`to verify`。
