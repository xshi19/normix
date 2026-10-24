# normix 数学在 Lean 4 + Mathlib 中的形式化可行性

2026-10-24 · 写给 Xiang Shi · 离线研究报告

建议在 normix 仓库内设一个独立的 `formal/lean/` 项目。先证明有限加权矩、参数转换和混合表示，再进入连续指数族、GIG 归一化和 EM。Bessel 的数学定义可以形式化。当前求积实现的误差界也可以研究。但“JAX 能追踪出 jaxpr”只解决了程序表示问题，还没有证明它算对了 Bessel。

本报告只做静态阅读和数学分析。没有联网、安装依赖、运行测试或运行 Lean。没有修改其他文件。normix 阅读基线为 `51e053c57237a29c71b56d3a2a86988277477bd9`。提供的 jax-lean 副本没有 `.git`，因此“12 次提交”采用输入说明，未独立核实历史。根目录确实没有 `LICENSE`；个别示例有自己的许可证。

下文区分三类证据：

- **本地核对**：能在本次提供的源码或文档中找到。源码存在不表示本次编译通过。
- **数学推导／文献记忆**：本文给出推导，或依据已有数学知识。没有重新查阅外部原文。
- **to verify**：需要以后查固定版本的 Mathlib、编译或核实文献。不会把猜测的库定理名写成已有接口。

工作量是规划估计，不是基准数据。一个人周按五个工作日计。估计假设有一位熟悉 Lean 的证明工程人员，AI 辅助检索和写证明，Xiang 审数学。Xiang 若独自从零学习，日历时间应至少留出约两倍余量；特殊函数部分可能更久。

## 1. 从什么开始理解 Lean

**形式化**是把数学对象、假设和结论写成精确的语言。**验证**是让程序检查证明是否推出这个结论。例如，“GIG 的方差非负”必须说明参数范围、随机变量的分布，以及二阶矩存在。

**Lean 4**既是编程语言，也是交互式证明助手。这里用它写数学定义和证明，不用它取代 JAX 数值计算。

**Mathlib**是 Lean 的数学库。实数、线性代数、测度、积分和微积分都有基础设施。它不是“输入论文，自动得到证明”的程序。某个领域有基础设施，也不表示已有 normix 需要的整条定理。

**规约（spec）**是要证明的精确陈述。比如：当 $a,b>0$ 且 $p\in\mathbb R$ 时，某个明确写出的 GIG 密度在 $(0,\infty)$ 上积分为 $1$。规约必须独立于证明接受审查。

**策略（tactic）**是帮助构造证明的命令。`ring`处理多项式恒等式，`simp`使用已知等式化简，`positivity`处理常见的正性目标。普通 tactic 生成的证明项仍由**内核（kernel）**检查。内核是负责最后逻辑检查的小型核心。它不会判断你的定义是否表达了正确的金融或概率问题。

**公理（axiom）**是未经本项目证明而接受的前提。`sorry`是 Lean 中暂时跳过证明的占位符；包含它的文件可能构建成功。构建成功不能单独当作证明完成。

### 1.1 一条小定理长什么样

下面是本文拟议的 Lean 示例，**没有编译**。`NormixProposal`是拟用命名空间，不是现有库。

```lean
import Mathlib

namespace NormixProposal

theorem gig_b_roundtrip (b : ℝ) : -2 * (-b / 2) = b := by
  ring

end NormixProposal
```

`(b : ℝ)`表示任取实数 $b$。冒号后是结论。`by`后是证明。它对应 `GeneralizedInverseGaussian.natural_params` 和 `from_natural` 中 $\theta_2=-b/2$、$b=-2\theta_2$ 的代数关系。[N2]

这条式子对负数也成立，所以没有加 $b>0$。但要把它解释成合法 GIG 参数，还要另证构造结果留在参数域内。它没有证明密度归一化，更没有证明浮点乘除精确。

真正的概率规约还要定义“哪一个密度”。例如，可先写以下定义草图。它使用实数的指数和对数，且显式限制支持集。

```lean
import Mathlib
open MeasureTheory Set

namespace NormixProposal

noncomputable def gigKernel (p a b x : ℝ) : ℝ :=
  Real.exp ((p - 1) * Real.log x - (a * x + b / x) / 2)

noncomputable def gigMass (p a b : ℝ) : ℝ :=
  ∫ x in Ioi (0 : ℝ), gigKernel p a b x

end NormixProposal
```

这也**未经编译**。`noncomputable`表示这里的定义用于数学，不承诺生成可执行的实数积分器。`Ioi 0`表示开区间 $(0,\infty)$。随后要证明可积性、质量严格为正，以及质量等于明确的 Bessel 公式。不能只把密度定义成“核除以积分”，证归一化后就宣称验证了代码里的归一化常数。

导数也应表达存在性。jax-lean 的 `examples/autodiff/proofs/Jvp.lean`实际使用 `HasDerivAt`，它同时陈述“可导”和“导数等于什么”。其中证明的是二次多项式的 `gradient_correct`，不是所有 `jax.grad` 都正确。[J3]

### 1.2 四个不同的验证目标

| 层次 | normix 例子 | 需要证明什么 |
|---|---|---|
| 数学定理 | $\nabla\psi=E[t]$ | 在明确条件下，积分定义推出导数公式 |
| 精确实数算法 | 固定节点的加权指数和 | 代数、导数、矩关系在实数上成立 |
| 数值近似 | 192 节点近似 Bessel 积分 | 截尾、求积、常数近似造成多大误差 |
| 机器执行 | 实际 JAX/XLA 在选定设备上运行 | 舍入、编译和原语实现符合某个误差规约 |

XLA 是 JAX 使用的编译系统。第一层不自动覆盖其余三层。jax-lean 当前主要连接程序表示和第二层。第三层需要数值分析。第四层还需要机器数与编译行为的模型。

## 2. 本地阅读改变了哪些判断

起点是 [2026-10-19 概览](00_overview_report.md)。它提出了有用的区分，但以下判断需要收窄。

| 概览中的判断 | 本报告的判断与依据 |
|---|---|
| 没有 Bessel 原语，因此正态混合目前做不了 | 对这个导入器的现成支持成立；对数学形式化不成立。Bessel 可由积分定义，混合变换也不需要调用导入器。[N5, J1] |
| Lean 几周可以先证指数族对偶和 EM | 若只是已给定关键分析前提的抽象推论，可能。若包括连续积分、微分交换和 GH 实例，通常要数月。 |
| 浮点数值、求解器不值得证明 | 不建议先证整个机器执行链。但实数求积误差、局部逆映射和受限收敛命题有价值。不要把“首期不做”写成“不可能”。 |
| TLA+ 不适合 normix | 数值恒等式不适合交给它。停止标志、失败回退和后端选择协议可以建模；见第 7 节。 |
| 数学网站可直接作为 Lean 规约 | 需先审查公式、域和文字强度。形式化应允许发现文档错误。 |

静态阅读已发现以下规约审查点。这不是完整的文档审计，也不是代码修复报告。

1. `docs/theory/gig.md` 的替代参数中，定义 $\delta=\sqrt{b/a}$、$\eta=\sqrt{ab}$ 后，密度前因子写成 $\delta^p$。由同页原定义直接代入得到的是 $\delta^{-p}$。这是本文代数核对所得，应在后续单独修正文档。[N1]
2. `docs/theory/em_algorithm.md` 写“两种算法收敛到 MLE”。一般 EM 不保证全局最大似然解。需要区分似然值收敛、聚点平稳、整个参数序列收敛。该页末尾又把 Gamma／InverseGamma 的 `from_expectation`称为闭式；实际 Gamma 调用 `_newton_digamma`，InverseGamma 也涉及形状反解。[N3, N6]
3. GIG 平坦统计量顺序是 $(\log Y,1/Y,Y)$。`NormalMixtureEta` 的字段顺序是 $(1/Y,Y,\log Y,\ldots)$。文档也在两种顺序间切换。Lean 应使用命名字段和显式转换，不能默认两个数组相同。[N2, N6]
4. `_rescale` 的理想数学作用保持 $X$ 的边缘分布。若把 $Y$ 改成 $sY$，同一坐标下的联合密度不会原封不动；必须带变量变换的 Jacobian，即体积缩放因子。GH 的 `_build_rescaled` 还夹紧 $a,b$，因此连边缘分布的精确不变性也需“夹紧未触发”的条件。[N4]
5. `dev-notes/tech_notes/gig_eta_to_theta.md` 仍有“求解器不可 JIT”的旧描述。当前源码和设计 S5／S11 有 JAX 冷启动与隐式导数。本文以当前源码和带状态的设计记录为准，不把旧笔记当作当前限制。[N6, N7]

## 3. 数学清单、难度和价值

难度用四级：A 是有限代数；B 是已有分析工具的组合；C 是连续概率和积分基础工作；D 是新的特殊函数或误差理论工程。下表的工作量按“相关基础已建立后的增量”估算，不能简单相加。

| 对象 | 难度／价值 | 增量估计 | AI 可以承担 | Xiang 必须审查 |
|---|---|---|---|---|
| GIG／Gamma 参数往返、统计量顺序 | A／高 | 2–5 人日 | 定义、代数证明、映射清单 | 参数约定、夹紧边界 |
| 有限加权和的梯度、协方差、半正定性 | A–B／很高 | 1–3 人周 | 有限和与微积分证明 | 固定节点含义、严格正性条件 |
| 一般连续指数族的凸性、梯度、Fisher | C／很高 | 4–8 人周 | 分拆可积性与导数引理 | 正则性、支持集、非最小表示 |
| Legendre 对偶、局部逆映射 | B–C／高 | 2–5 人周 | 凸分析、链式法则 | 可逆域、是否宣称全局满射 |
| Gamma／InverseGamma 归一化与矩 | B–C／高 | 2–4 人周 | 变量代换、Gamma 积分复用 | rate／scale 约定、矩存在域 |
| IG 归一化、矩、闭式 CDF | C／高 | 2–5 人周 | 高斯积分与导数证明 | 参数化、端点和尾部 |
| GIG 归一化、任意实阶矩 | C／很高 | 2–4 人周；另需 Bessel 基础 | 对数代换、归一化比值 | 内部域与 Gamma 边界区别 |
| 混合表示、均值与协方差 | C／很高 | 2–4 人周 | 积测度和可测变换证明 | 独立性、所需矩 |
| GH 闭式边缘密度与条件 GIG | C／很高 | 2–5 人周；复用前两项 | 完全平方、换元、积分 | 常数、矩阵域、特殊点 |
| NIG／VG／NInvG 专门公式 | B–C／高 | 每类 1–3 人周 | 专门积分与边界证明 | 不能以数值极限代替精确边界 |
| CDF 一般性质、理想逆变换抽样 | C／高 | 2–4 人周 | 单调性、分布推前 | 不连续处、端点、量化误差范围 |
| IG／GIG 拒绝抽样算法的理想正确性 | C–D／中高 | 各 4–10 人周 | 提议分布、接受事件、终止证明 | 包络合法性、失败事件 |
| EM 单步不降、惩罚版本 | C／很高 | 3–6 人周；已有密度基础 | KL／Jensen 证明与实例条件 | E/M 步精确性、目标函数 |
| 正态块 M 步和 IG M 步 | A–C／很高 | 1–3 人周 | 代数、优化目标比较 | 非奇异条件、最优性而非仅驻点 |
| EM 聚点平稳性 | C–D／中 | 6–12 人周 | 紧性与极限论证 | 假设是否覆盖实际 GH 拟合 |
| Bessel 对称、递推、参数导数 | C–D／很高 | 基础 4–8 人周；恒等式再 2–4 人周 | 积分界、微分交换、分部积分 | 支配函数、端点项 |
| Bessel 渐近、求积与导数误差 | D／高但昂贵 | 每个受限区域 2–6 人月 | 辅助不等式和证明实现 | 误差指标、参数一致性、数值分析方案 |

这些项目不是独立岛屿。有限指数族可先支持离散 Bessel 矩；连续指数族和 Bessel 积分再支持 GIG；混合密度和条件 GIG 再支持 EM 实例。

### 3.1 指数族：定义先于恒等式

指数族英文为 exponential family，后文简称 EF。设 $\rho$ 是固定参考测度，$h(x)\ge0$ 是基准密度因子，$t(x)\in\mathbb R^k$ 是充分统计量。在这里，它收集指数中与参数配对的观测函数。定义

$$
Z(\theta)=\int h(x)e^{\langle\theta,t(x)\rangle}\,d\rho(x),\qquad
\psi(\theta)=\log Z(\theta),\qquad
p_\theta(x)=h(x)e^{\langle\theta,t(x)\rangle-\psi(\theta)}.
$$

先处理 $0<Z(\theta)<\infty$ 的参数域。涉及微分时限制到其开内部。代码中的 `log_base_measure` 返回的是 $\log h$，不是 $h$。[N2]

建议分开证明四组结论。

- **凸性**：对自然参数域中的线段，$\psi$ 凸。可由 Hölder 不等式证明；不必先证明 Hessian 存在。
- **梯度与 Hessian**：在局部可交换微分与积分的条件下，$\nabla\psi=E_\theta[t]$，$\nabla^2\psi=\operatorname{Cov}_\theta(t)$。
- **Fisher 信息**：定义得分 $s_\theta(x)=\nabla_\theta\log p_\theta(x)$。在支持集不随参数变化等条件下，$I(\theta)=E[s_\theta s_\theta^\top]=\nabla^2\psi(\theta)$。不能直接把 Fisher 定义为 Hessian 后把这个等式当成验证成果。
- **对偶**：定义 $\psi^*(\eta)=\sup_{\theta\in\Theta}\{\langle\theta,\eta\rangle-\psi(\theta)\}$。先在 $\eta\in\nabla\psi(\Theta)$ 上证明对偶等式与逆关系。要声称像等于整个矩空间内部，还需额外条件。

以上是标准理论的记忆与本文推导方向。可参照 Wainwright–Jordan（2008）；本次未查原文，也未核实 Mathlib 中是否已有整套指数族框架。

有三个 normix 特有的陷阱。

**最小性。** 统计量之间没有非平凡的几乎处处仿射关系，才称为最小表示。最小性通常是严格凸性和参数可识别的关键。`MultivariateNormal` 和联合混合都展开 $xx^\top$ 的全部矩阵元素，对称位置有重复。不能据此声称整个平坦坐标空间的 Fisher 正定。Lean 数学层宜使用对称矩阵，或明确只在可识别子空间上求逆。[N2, N4]

**受限子族。** `JointNormalMixture._ambient_moment_contract`让 joint VG／NInvG／NIG 返回共同 GH 统计量的矩与协方差。受限配分函数的梯度并不是这些环境统计量的期望。对内部光滑子族 $\theta=\theta(u)$，其内在 Fisher 是 $J^\top\operatorname{Cov}(t)J$，其中 $J$ 是参数映射的导数；边界还要另查矩是否存在。不能把该 API 的环境协方差误称为每个子族坐标下的 Fisher。[N4, N7]

**数值逆与数学逆。** `ExponentialFamily._implicit_natural_params`使用 $H(\theta)d\theta=d\eta$，这里 $H=\nabla^2\psi$。局部逆函数定理可以支持它的理想规约，但须有精确驻点、内部解和可逆 Hessian。实际求解只有残差容差，还可能触发边界或失败分支。其返回值的误差与导数误差另立命题。Bregman 散度是凸函数值与其切平面之差。`bregman_objective`计算的是 $\psi(\theta)-\theta^\top\eta$；当 $\eta=\nabla\psi(\theta_0)$ 时，它与 $D_\psi(\theta,\theta_0)$ 相差不依赖优化变量的项，不应直接声称这个函数非负。[N2, N6]

### 3.2 密度、矩和混合表示

统一用 $V>0$ 表示混合变量；源码多数叫 `Y`。$\mu,\gamma\in\mathbb R^d$，$\Sigma$ 为正定协方差矩阵。令 $Z\sim N(0,\Sigma)$ 与 $V$ 独立，定义

$$X=\mu+\gamma V+\sqrt V\,Z.$$

| 分布 | 精确数学定义与第一批目标 | 源码 |
|---|---|---|
| Gamma | $f(v)=\beta^\alpha v^{\alpha-1}e^{-\beta v}/\Gamma(\alpha)$；$\alpha,\beta>0$，$\beta$ 是 rate。证归一化和 $E[V^r]=\Gamma(\alpha+r)/(\Gamma(\alpha)\beta^r)$，条件 $r>-\alpha$ | `distributions/gamma.py`：`_log_partition_from_theta`、`raw_moment`、`cdf` |
| InverseGamma | $f(v)=\beta^\alpha v^{-\alpha-1}e^{-\beta/v}/\Gamma(\alpha)$；证倒数变换，$E[V^r]=\beta^r\Gamma(\alpha-r)/\Gamma(\alpha)$，条件 $r<\alpha$ | `distributions/inverse_gamma.py`：`raw_moments`、`mean`、`var` |
| IG，逆高斯 | 均值参数 $m>0$，形状 $\lambda>0$；$f(v)=\sqrt{\lambda/(2\pi v^3)}\exp[-\lambda(v-m)^2/(2m^2v)]$。证 $E[V]=m$、$\operatorname{Var}V=m^3/\lambda$、$E[1/V]=1/m+1/\lambda$ | `distributions/inverse_gaussian.py`：`mean`、`var`、`from_expectation` |
| GIG | $p\in\mathbb R,a,b>0$；用 Bessel 积分求归一化常数，再证任意实阶矩 | `distributions/generalized_inverse_gaussian.py`：`natural_params`、`raw_moment`、`_eta_H_from_moments` |
| GH | 取 $V\sim\mathrm{GIG}(p,a,b)$；证明混合定义与边缘密度等价 | `distributions/generalized_hyperbolic.py`：`log_prob` |
| NIG | 取 IG 混合；等价于 $p=-1/2,a=\lambda/m^2,b=\lambda$。证半整数 Bessel 归一化简化 | `distributions/normal_inverse_gaussian.py`：`log_prob` |
| VG | 取 Gamma 混合；对应精确边界 $p=\alpha,a=2\beta,b=0$。单独证明该密度，处理 $x=\mu$ 的奇点 | `distributions/variance_gamma.py`：`log_prob` |
| NInvG | 取 InverseGamma 混合；对应 $p=-\alpha,a=0,b=2\beta$。处理 $\gamma=0$ 的 Student 型边界与矩发散 | `distributions/normal_inverse_gamma.py`：`log_prob` |

表内路径均以 `normix/` 为根。[N3] IG 与 InverseGamma 是不同分布。代码保持各自公式。Lean 可以复用数学引理，但不应因此让运行时代码一律调用 GH。

GIG 的核心积分是

$$
\int_0^\infty v^{p-1}e^{-(av+b/v)/2}\,dv
=2(b/a)^{p/2}K_p(\sqrt{ab}).
$$

由 $v=\sqrt{b/a}\,e^u$ 换元可得。把 $p$ 换成 $p+r$ 后取比值，即得

$$E[V^r]=(b/a)^{r/2}\frac{K_{p+r}(\sqrt{ab})}{K_p(\sqrt{ab})}.$$

这条证明先完成所有阶的可积性。边界 $a=0$ 或 $b=0$ 不在这个公式的直接适用域内。边界密度、弱收敛和矩的收敛是三个不同命题。弱收敛指对有界连续函数的期望收敛；它不自动推出高阶矩收敛。

GH 的闭式密度不必另造特殊函数。令

$$
q(x)=(x-\mu)^\top\Sigma^{-1}(x-\mu),\quad
A=a+\gamma^\top\Sigma^{-1}\gamma,\quad B=b+q(x),\quad r=p-d/2.
$$

展开条件正态密度中的二次型，再用同一个 GIG 积分，得到

$$
f_X(x)=\frac{(a/b)^{p/2}e^{(x-\mu)^\top\Sigma^{-1}\gamma}}
{(2\pi)^{d/2}|\Sigma|^{1/2}K_p(\sqrt{ab})}
(B/A)^{r/2}K_r(\sqrt{AB}).
$$

同一个计算给出 $V\mid X=x\sim\mathrm{GIG}(r,A,B)$。这直接对应 `JointNormalMixture._posterior_gig_params`。后者的带下界版本 `_floored_posterior_gig_params`改变了目标参数，不能直接套用精确共轭定理。[N4]

归一化也可先由混合定义得到：每个条件正态密度积分为 $1$，再用非负积分交换定理交换积分。随后证明上述闭式等于这个混合积分。这样可以把“这是一个概率分布”和“代码使用的公式正确”分成两条可检查结论。

在所需矩存在时，条件期望给出

$$E[X]=\mu+\gamma E[V],\qquad
\operatorname{Cov}(X)=E[V]\Sigma+\operatorname{Var}(V)\gamma\gamma^\top.$$

统一证明可先假设 $E[V^2]<\infty$。然后另证 $\gamma=0$ 等较弱条件下的版本。对称重尾分布不能因为“正负抵消”就写均值为零。Cauchy 的均值不存在；代码用 `inf`表达不可用，也不等于数学上的有符号期望确实为 $+\infty$。[N4]

### 3.3 CDF：数学定义与近似表要分开

**累积分布函数（CDF）**是 $F(x)=P(X\le x)$。对于有密度的实值分布，$F(x)=\int_{-\infty}^x f(t)\,dt$。先证 $0\le F\le1$、单调、右连续和两端极限。密度连续处再证 $F'=f$。

本地实现分三类。[N3, N5]

- `Gamma.cdf`使用正则化不完全 Gamma 函数。需要证明该特殊函数与密度积分的联系。
- `InverseGaussian.cdf`有正态 CDF 的解析公式。`log_ndtr`只是稳定计算的路径；解析恒等式不等于该路径的舍入误差证明。
- `GIG.cdf`和 `_UnivariateNormalMixtureMixin.cdf`使用数值表。`build_pinv_table`实际做有限区间截尾、梯形积分、归一化和线性插值。PINV 是基于多项式插值的数值反演这一名称的缩写；不能据名字套用某篇高阶反演算法的误差结论。

多维 CDF 是 $P(X_1\le x_1,\ldots,X_d\le x_d)$，不应与某个一维投影的 CDF 混用。当前混合 CDF 目标先限于 `Univariate*` 类；不要把抽象定义存在写成所有多维类已实现 CDF。

表格方法可以先证明有限数据性质：网格递增、质量非负且总和正，则归一化累计量单调，插值值在 $[0,1]$ 内。再研究 $\sup_x|\widehat F(x)-F(x)|\le\varepsilon$。这里 $\widehat F$ 是表格 CDF，$\varepsilon$ 是明确的误差上界。

`build_pinv_table` 的 `tail_eps`用于密度高度阈值。密度高度降到某个比例，不自动意味着尾概率小于同一个数。Bessel 窗口的 `log_drop`也有同样区别。尾部误差必须用积分不等式另证。

### 3.4 “随机数生成正确”有三种意思

第一种是**分布变换正确**。随机变量可看成样本空间上的可测函数。可测表示事件的原像可以被赋予概率。函数把输入概率测度搬到输出空间，得到的测度叫**推前测度**。

取相互独立的 $U\sim\mathrm{Uniform}(0,1)$ 和 $G\sim N(0,I_d)$。若 $Q_V$ 是 $V$ 的广义分位函数，$L_\Sigma L_\Sigma^\top=\Sigma$，则定义可测映射

$$T(u,g)=\mu+\gamma Q_V(u)+\sqrt{Q_V(u)}L_\Sigma g.$$

证明 $T$ 在乘积输入测度下的推前等于 GH 等目标分布，就是理想混合抽样正确。广义分位函数可定义为 $Q_V(u)=\inf\{v:F_V(v)\ge u\}$；端点零概率事件与可测性要处理。也可直接从已具有正确分布的独立 $V,Z$ 出发，避免一开始就证明高斯采样算法。

第二种是**具体理想抽样算法正确**。这比混合变换多一层。

- `InverseGaussian.rvs`用正态平方、均匀数和二选一变换。需要推导该变换的密度；不是证明三个 API 名称存在。
- `_gig_rvs_tdr`使用变换密度拒绝抽样（TDR），即先从包络分布提议，再按概率接受。必须证包络在全域盖住目标、接受率公式正确，以及重复提议独立。无限重试可证几乎必然终止，即不终止事件的概率为零。
- 实际 `_gig_rvs_tdr`有最大轮数，耗尽后返回 NaN。规约应允许失败结果，并证明成功条件下的分布。若每轮接受概率至少为 $a_0>0$，独立提议下单列失败概率至多 $(1-a_0)^R$，$R$ 为轮数。对多列可再取联合上界。技术笔记的 $a_0\ge e^{-1}$ 仍需连同有限精度包络构造一起核实。
- `QuantileTable.rvs`给出的是近似分位表诱导的分布。即使输入均匀数完全理想，也不能宣称它精确服从原 GIG。[N5]

第三种是**真实伪随机数生成器（PRNG）正确**。PRNG 从有限种子确定地产生位序列。有限种子与浮点输出的分布是离散的，不可能精确等于连续高斯分布。可以研究种子到输出的实现、统计性质或有限精度分布，但那是另一套规约。特别是离散输出与连续目标的总变差距离可为 $1$，不能随意承诺这种距离很小。

`JointNormalMixture.rvs`用 `seed`采样混合变量，用 `seed + 1`生成高斯。种子不同不是独立性的数学证明。即使改为 `split`，仍需声明理想随机模型，或另证 PRNG 性质。jax-lean 的随机模型仅覆盖有限均匀整数抽样，也明确不证明真实 PRNG 独立均匀。[N4, J4]

因此首期应交付“给定正确且独立的输入分布，混合映射输出正确分布”。报告标签应写“理想抽样变换已证明”，不能写“`rvs`全部已验证”。

### 3.5 EM：先证明不下降，再讨论收敛

EM 是期望最大化算法。对固定观测 $x_1,\ldots,x_n$，令 $p_\theta(x,v)$ 为联合密度，$m_\theta(x)=\int p_\theta(x,v)\,dv$ 为边缘密度，$q_\theta(v\mid x)=p_\theta(x,v)/m_\theta(x)$ 为条件密度。定义平均对数似然

$$\ell(\theta)=\frac1n\sum_{i=1}^n\log m_\theta(x_i),\qquad
Q(\theta'\mid\theta)=\frac1n\sum_{i=1}^n
\int q_\theta(v\mid x_i)\log p_{\theta'}(x_i,v)\,dv.$$

要求 $n>0$、相关边缘密度正且有限，条件密度归一化，所用对数积分存在。对于存在零密度的更一般情形，还需处理绝对连续性，即旧后验有质量的地方新后验不能无定义。

**KL 散度**为 $D_{\mathrm{KL}}(q\|r)=\int q\log(q/r)$。其非负性可由 Jensen 不等式得到；Jensen 说凸函数作用于平均值，不大于作用后再取平均。条件齐全时，

$$
\ell(\theta')-\ell(\theta)
=Q(\theta'\mid\theta)-Q(\theta\mid\theta)
+\frac1n\sum_iD_{\mathrm{KL}}\bigl(q_\theta(\cdot\mid x_i)\|q_{\theta'}(\cdot\mid x_i)\bigr).
$$

因此只要 $Q(\theta'\mid\theta)\ge Q(\theta\mid\theta)$，似然就不下降。不需要新参数必然达到全局的 Q 最大值。这是广义 EM 的基本条件。上述理论来自 Dempster–Laird–Rubin（1977）的文献记忆；分解式在此可由代入 $\log p=\log q+\log m$ 检查。本次没有查原文或证明库。

Lean 的第一版宜直接从密度比定义条件分布。这样可以暂时避开任意测度空间的条件分布存在性理论。不能把“单步似然不下降”作为公理，然后只证循环累计不下降，并把后者称为 EM 理论完成。

**M 步。** 用 `NormalMixtureEta` 的命名字段，记 $s_1=E[1/V]$、$s_2=E[V]$、$s_4=E[X]$、$s_5=E[X/V]$、$s_6=E[XX^\top/V]$。对非退化输入，$s_1s_2>1$。正态块候选解是

$$
\mu_* = \frac{s_4-s_2s_5}{1-s_1s_2},\qquad
\gamma_* = \frac{s_5-s_1s_4}{1-s_1s_2},
$$

$$
\Sigma_*=s_6-s_5\mu_*^\top-\mu_*s_5^\top
+s_1\mu_*\mu_*^\top-s_2\gamma_*\gamma_*^\top.
$$

先证它满足正规方程，再在 $\Sigma_*\succ0$ 等条件下证明最大化 Q。正规方程为零不等于最优性。若数据不足导致散布矩阵奇异，开正定域中可能没有有限最大值。

`JointNormalMixture._mstep_normal_params`把分母设下界，并给协方差加 `SIGMA_REG`。所以理想闭式与完整实现不是精确同一函数。IG 混合部分的 $m_*=E[V]$、$\lambda_*=(E[1/V]-1/E[V])^{-1}$ 是闭式；Gamma 与 InverseGamma 的形状需要求根；GIG 需要三维反解。[N3, N4, N6]

**收敛必须拆成三个目标。**

1. 若 $\ell(\theta_k)$ 单调且有上界，则似然值有极限。
2. 若再有迭代留在开参数域内的紧集、目标可微、更新映射适当连续，以及非平稳点严格改进等条件，可证明聚点平稳。平稳表示无约束内部梯度为零；有约束时应改为相应的一阶最优条件。
3. 参数序列本身收敛、极限为局部最大值、极限为全局 MLE，都比第 2 条强。不能从单调性直接推出。

一个可实现的受限证明路线是：先针对连续单值更新映射证明紧集上的聚点结论，再单独从精确 M 步与 Q 的导数关系推出严格改进条件。若 M 步有多个最优解或不连续，需要集合值更新的理论，工作量更大。Wu（1983）的 EM 收敛工作是文献记忆中的参考；准确假设与版本 **to verify**，这里不声称已复述其完整定理。

这些条件对 GH 并非自动满足。尺度不识别、边界解和 VG 在 $x=\mu$ 处的无界密度都需要处理。`BatchEMFitter`还包括后验下界、近似 GIG 求解、回退、缩放和 Aitken 停止。Aitken 是由相邻似然值外推极限的停止估计，不是平稳性的证明。[N6]

普通 batch EM 的结论不能直接用于全部拟合器。

- 惩罚 EM 应证明 $\ell(\theta)-P(\theta)$ 不下降，$P$ 是固定惩罚函数。任意逐字段收缩不一定对应同一个 $P$。
- MCECM 是多轮期望／条件最大化；每轮必须对应正确的条件目标。
- `IncrementalEMFitter`的随机小批量与自定义 `EtaUpdateRule`一般不具有每步完整数据似然单调性。随机逼近的步长、噪声和稳定性假设另立项目。

对于误差可控的近似 E/M 步，可以研究 $\ell(\theta_{k+1})\ge\ell(\theta_k)-\varepsilon_k$。但 $\varepsilon_k$ 必须由误差分析导出；不能把测试容差直接当作理论上界。

### 3.6 Bessel：定义、导数、渐近和求积

取实阶 $\nu\in\mathbb R$，参数 $z>0$。建议先使用当前内核对应的定义：

$$K_\nu(z)=\frac12\int_{\mathbb R}e^{\nu u-z\cosh u}\,du
=\int_0^\infty e^{-z\cosh u}\cosh(\nu u)\,du.$$

两种积分的等价由拆成正负半轴得到。前者直接连接 `normix/utils/bessel.py`，后者便于读经典公式。它们是修正第二类 Bessel 函数的标准积分表示。依据本地技术笔记和文献记忆；若固定 Mathlib 版本已有其他定义，必须证明等价，不能仅复用同名符号。[N5]

建议按以下顺序立项。

| 命题 | 主要证明义务 | 难点与增量估计 |
|---|---|---|
| 积分存在且 $K_\nu(z)>0$ | 两侧尾部由 $e^{-z\cosh u}$压住；正被积函数在非零测度集合上严格正 | 基础工程 4–8 人周的一部分 |
| $K_{-\nu}(z)=K_\nu(z)$ | $u\mapsto-u$换元 | 可积性完成后 1–3 人日 |
| $K_{\nu+1}=K_{\nu-1}+(2\nu/z)K_\nu$ | 对指数核分部积分；两端边界项为零 | 1–2 人周 |
| $\partial_zK_\nu=-(K_{\nu-1}+K_{\nu+1})/2$ | 微分移入积分，再用 $\cosh u=(e^u+e^{-u})/2$ | 1–2 人周；与下一项复用 |
| $\partial_\nu K_\nu=\frac12\int u e^{\nu u-z\cosh u}\,du$ | 局部一致的可积支配函数 | 1–3 人周，二阶再增加工作 |
| $K_{1/2}(z)=\sqrt{\pi/(2z)}e^{-z}$ | 与高斯积分联系；负阶用对称性 | 1–3 人周 |
| 固定阶的大 $z$ 渐近；小 $z$ 边界 | 指定极限、余项与一致范围 | 每个目标 4–12 人周或更多 |
| 当前有限求积对值和导数的误差 | 尾积分、有限区间求积、存储常数误差 | 每个受限区域 2–6 人月 |

这些恒等式来自标准特殊函数理论的记忆与当前积分定义的推导；没有核实 Mathlib 的对应定理名。半整数公式也出现在 NIG 的 `log_prob`实现里。

微分交换可以先在紧参数盒上做：$|\nu|\le M$、$z\ge z_0>0$，其中 $M,z_0$ 固定。所需导数被形如

$$C(1+|u|^m)(1+\cosh^n u)e^{M|u|-z_0\cosh u}$$

的函数控制。这里 $m,n$ 为非负整数，$C>0$ 是常数。须证明这个函数可积，并选择包含待求导点邻域的参数盒。不能只证明某一点的被积函数可导。

令 $P_{\nu,z}$ 为密度正比于 $e^{\nu u-z\cosh u}$ 的概率测度，$L(\nu,z)=\log K_\nu(z)$。可推出

$$
\partial_\nu L=E[U],\quad \partial_{\nu\nu}L=\operatorname{Var}(U),\quad
\partial_zL=-E[\cosh U],\quad \partial_{zz}L=\operatorname{Var}(\cosh U),
$$

$$\partial_{\nu z}L=-\operatorname{Cov}(U,\cosh U).$$

这是 order derivative，即对阶 $\nu$ 求导，最自然的形式化路线。没有初等闭式并不妨碍证明。`BesselMoments.d_order`、`d2_order`、`d_arg`、`d2_arg`、`d2_order_arg`分别近似这些量。

**先证明离散版。** 固定有限节点 $u_i$ 和正权重 $w_i$，定义

$$A_Q(\nu,z)=\log\sum_iw_i e^{\nu u_i-z\cosh u_i}.$$

对它求导，得到同样的矩／协方差关系，概率改为归一化的有限权重。对任意向量 $v$，协方差 $C_Q$ 满足

$$v^\top C_Qv=\sum_ip_i\bigl(v^\top(s_i-\bar s)\bigr)^2\ge0.$$

这里 $p_i$ 是归一化权重，$s_i$ 是节点统计量，$\bar s=\sum_ip_is_i$。这正是 `_moments_from_weights`中的中心化 Gram 矩阵结构。Gram 矩阵是向量内积构成的矩阵，因而半正定。严格正定还要求统计量在这些节点上没有退化方向，不能只由正权重推出。

**冻结几何的边界。** `_eval_quad`先按输入选择中心、窗口和节点，再用 `stop_gradient`冻结它们。自动微分（AD）得到的是这组固定几何下的偏导数。把整个“随输入重新选节点”的前向函数视为普通实函数，其导数一般还含几何变化项。

准确规约应写：令 $g_0$ 是在 $(\nu_0,z_0)$选出的几何；存在固定几何函数 $L_{g_0}(\nu,z)$，代码返回值在该点等于它，AD 输出等于它在该点的偏导数。然后分别证明该值和这些偏导数逼近真实 $L$。不能把 `stop_gradient`在前向语义中解释为恒等映射后，再声称 AD 自动等于整个复合函数的经典导数。

**渐近不是误差证书。** 文献记忆中，对固定 $\nu$，$z\to\infty$ 时有

$$L(\nu,z)=-z+\tfrac12\log\!\frac{\pi}{2z}
+\frac{4\nu^2-1}{8z}+O(z^{-2}).$$

$O(z^{-2})$表示误差绝对值最终被某常数乘 $z^{-2}$控制；常数可能依赖 $\nu$。它不是 $|\nu|\le300$、所有 $z$ 上的统一误差界。小 $z$ 时 $\nu=0$与 $\nu\ne0$也要分开。对值成立的渐近式不能未经证明直接对 $\nu$或 $z$求导。

当前生产代码已经删除按渐近区域切换的实现。现在证明渐近式主要是为尾界、边界理论和独立校验服务，不是验证一条仍在使用的渐近分支。[N5, N7]

192 个节点和固定窗口下降阈值也没有一个会自动趋于极限的过程。要证求积收敛，须定义一族规则：截尾区间扩大，网格或阶数改变，并证明尾误差与求积误差共同趋零。要证当前固定配置精度，则需给出有限参数范围内的具体误差界。

Gauss–Legendre（GL）求积是用特定节点和权重的有限和近似区间积分。源码中的节点由 NumPy `leggauss`产生并存为浮点数。实数语义可以把这些值看成精确有理数，但它们不因此成为精确 Legendre 多项式根。证明有限权重协方差不需要精确 GL 节点；套用精确 Gauss–Legendre 误差公式则需要额外的节点／权重误差分析。

可以从一个明确的误差分解开始。令 $I=2K_\nu(z)$，$I_W$ 是窗口内的精确积分，$Q$ 是理想节点求积，$\widehat Q$ 是采用存储节点和权重的实数求和。则

$$|\widehat Q-I|\le |I-I_W|+|I_W-Q|+|Q-\widehat Q|.$$

三项分别是截尾、求积和常数扰动误差；还没有加入运算舍入。若进一步得到 $|\widehat Q/I-1|\le\varepsilon<1$，就能推出 $|\log\widehat Q-\log I|\le-\log(1-\varepsilon)$。代码中存储的 $\log2$ 常数误差另加。导数则需对带 $u$、$\cosh u$等因子的积分分别给界，再控制比值。这个路线比直接要求“证明 `log_kv`准确”更容易审查。

### 3.7 网站其余理论也应进入清单

“全站理论可形式化”在表达能力上是合理方向；不能作为一期交付范围。以下只是按页面主题建立的清单，不是对这些页面的逐式审计。[N8]

| 页面 | 可拆出的目标 | 优先级与增量估计 |
|---|---|---|
| `varentropy.md` | 熵、信息量方差、Rényi 密度幂公式；明确基准密度恒定和矩存在条件 | 中高；EF 基础后 2–5 人周 |
| `gig.md`、`gh.md` 的 Hellinger 部分 | 密度平方根积分公式；边缘化后的距离不增 | 中高；归一化后 1–3 人周 |
| `shrinkage.md` | 固定 KL 惩罚与收缩统计量等价；惩罚 EM 不下降 | 高；EM 基础后 1–3 人周 |
| `online_em.md` | 先证有限步统计量递推与望远镜求和，再审 regret 等式与随机收敛 | 中；代数 1–3 人周，随机收敛另计数月 |
| `factor_analysis.md` | 因子条件矩、受限协方差与 M 步 | 中；4–8 人周，不能套完整 EF 的无约束反解 |
| `mean_risk_optimization.md`、`transaction_costs.md` | 线性投影、均值风险降维、二次目标等价 | 中；每个受限命题 1–4 人周 |
| `cvar_derivatives.md` | 条件尾部期望及其导数，需密度与微分交换条件 | 中；4–10 人周 |
| `enb.md`、`generalized_enb.md` | 风险分解、齐次函数恒等式、矩阵变换 | 中低；每组 2–6 人周 |
| `quadrature_scenarios.md` | 离散测度归一化、条件积分、截尾误差 | 中高；有限代数 1–2 人周，误差界另计 |

这里 regret 指累计损失相对某个比较解的差；CVaR 指指定尾部的平均损失；ENB 是有效风险来源数量的指标。AI 可按页面抽取假设和证明依赖。Xiang 应决定哪些是库承诺，哪些只是研究推导。尤其不能把回测表现当作这些数学恒等式的替代验证。

## 4. Mathlib 已有什么，哪里尚未核实

**本次没有完整 Mathlib 源码。无法报告截至 2026-10-24 的最新覆盖。** jax-lean 的配置给出 Lean `v4.33.0`，Mathlib 清单中的提交为 `db584cd6d46c92f209a44c0f1c829460d327499d`。这是本地文件记录，不是本次安装、构建或远端版本核实。[J1]

下表的 Bochner 积分是向量值 Lebesgue 积分的推广。支配收敛是用一个共同可积上界控制函数列，从而交换极限和积分的工具。

| 项目 | 可确定的范围 | 未核实的范围 |
|---|---|---|
| 实数、exp／log／sqrt、基础微分 | 本地 `Core/RealOps.lean`引用这些 Mathlib 模块；`Jvp.lean`使用 `HasDerivAt`、`deriv`和导数合成 | 本次未编译这些导入 |
| 测度、可测函数、独立性 | `Stdlib/BatchProbability.lean`实际引用 `Measure`、`Measurable`、`iIndepFun`及可测复合规则 | 不等于本地已有连续 GH 或连续采样的完成证明 |
| Lebesgue／Bochner 积分、支配收敛、积分微分交换 | 依 Mathlib 使用经验和文献记忆，这些方向已有基础设施 | 精确模块、假设形状、可直接套用的定理名均 **to verify** |
| Gamma 函数、高斯积分、凸分析 | 依记忆已有相关定义和大量定理 | 不完全 Gamma、digamma／trigamma、GIG 所需变换、Legendre 接口覆盖均 **to verify** |
| Bessel $K_\nu$ | 能在 Lean 中自行用积分定义；本地 normix 提供目标积分式 | 固定 Mathlib 版本是否有任意实阶 $K$、递推、order 导数和渐近，全部 **to verify**。不能断言“有”，也不能断言“没有” |
| GIG／GH／EM | 可以用已有分析基础搭建 | 没有本地证据表明这些整套定理已存在于 Mathlib |

jax-lean 自己的 `FiniteLaw`和 `Rand`是有限概率模型。其有限和证明可供设计参考，但不能直接承担连续 GIG 的积分。[J4]

首个可执行阶段应花 2–4 人日盘点固定版本的依赖。AI 列出“目标、现有定义、可用定理、缺口、最小编译例子”。Xiang 审定义是否一致。不能只查名字中有没有 Bessel：要看是第一类还是第二类、实阶还是整数阶、实变量还是复变量、以及归一化约定。

最费时的地方通常不在公式末尾。

- **几乎处处。** 密度在零测集上不同仍表示同一分布。但 API 在 $x=0$或 $x=\mu$的返回值可能属于独立的点值约定。
- **可积性。** 实值 Bochner 积分在不可积情形有总化约定；不能把它当成通常意义的有限期望。需要显式可积性命题，或用允许 $+\infty$ 的非负积分表达发散。
- **导数存在性。** 单写 `deriv f x = ...`不如同时证明 `HasDerivAt`可靠。某些总化定义在不可导点也返回一个值。
- **反常积分。** 在纸面上写“边界项显然为零”，在 Lean 中要证明两个方向的极限及可积性。
- **微分进入积分。** 需要参数邻域内统一可积的控制，不是逐点有限。
- **矩阵坐标。** 对称矩阵、正定域、Cholesky 因子和展平数组之间要有显式桥接。重复制表示会让可逆性假设失效。

这些是可以逐步解决的工程问题。应先接受一个范围明确的定理，再推广；不要靠增加“已满足结论”的前提消除难点。

## 5. jaxpr 到 Lean：可行，但不是数学证明的前置依赖

**jaxpr**是 JAX 追踪函数后得到的中间程序表示。它记录输入、常量、原语操作和输出。原语是系统认识的基本操作，如加法、指数和数组归约。固定形状、数据类型和静态参数后，Python 的很多结构已经消失。

jax-lean 的思路值得独立采用，不必把该仓库作为依赖。其本地结构给出了清楚的边界。[J1, J2]

```text
JAX 函数 --追踪--> ClosedJaxpr
                      |
             Python 导入／翻译
                 /         \
          Lean IR          可读 Lean 函数
             |                  |
           eval -------- 相等证书
                                |
                         另证业务数学规约
```

IR 是中间表示；`eval`是它的解释器，即按模型执行程序定义的函数。**证书（certificate）**在这里是一条 Lean 检查的等式证明。

实际文件分工如下。

- `python/jaxlean/translate.py`生成可读函数。
- `python/jaxlean/importers/jaxpr.py`生成形状与类型受检查的 IR。
- `JaxLean/Core/Jaxpr.lean`定义 IR 及其解释器。
- `python/jaxlean/certify.py`生成两者相等的证明目标。
- `JaxLean/Verification/Certificate.lean`用局部等式组合证书。
- `examples/certificates/generated/CertifiedMean.lean`展示形状 `[2]`的具体例子。

这个证书证明的是

$$\operatorname{eval}(\text{导入的 IR},x)=\text{生成的 Lean 函数}(x).$$

Python 导入器是否忠实读取原 jaxpr、共享的布局解码是否正确、JAX 追踪是否正确，都不在这个等式内。两个生成路径共享错误，也可能得到一个正确的等式。`SEMANTICS_REFERENCES.md`对此有明确说明。

### 5.1 为什么 JAX Bessel 在原则上可以翻译

normix 的 Bessel 已展开成求积、指数、加减乘除和循环。只要每个底层操作有定义，控制流可表达，就可以给整个 jaxpr 建立 Lean 语义。没有名为 Bessel 的原语不是根本障碍。

但提供的 jax-lean 副本尚不能据此宣布直接适用。

| 当前 normix 结构 | 本地 jax-lean 边界 | 以后要做的工作 |
|---|---|---|
| 项目锁定 JAX 0.9.1 | 副本锁定 JAX 0.8.0 | 对当前版本建立导入规则；不能为验证降级 normix |
| `_edge`的 40 次二分和 `_fori` | 普通翻译默认只展开最多 32 次 `scan`；证书子集不覆盖 `scan` | 明确循环语义和归纳规则，或在小试点中显式限定范围 |
| `_GL_T_J`、`_GL_LOGW_J`为捕获的浮点数组 | `JaxprImporter`拒绝捕获的浮点数组常量 | 扩展常量编码，记录精确存储值 |
| `expm1`、`log1p`、双曲函数、`hypot`等调用 | 支持清单没有覆盖这些调用的所有可能展开 | 以后实际追踪逐项核对；本次没有运行 jaxpr，不断言具体展开结果 |
| `stop_gradient` | 前向模型把它当恒等函数 | 给冻结几何与 AD 另立规约，不能沿用普通复合函数导数 |
| GIG 反解的 `custom_jvp`及 `while_loop` | 普通翻译仅内联 `custom_jvp_call`的原函数；不支持一般 `while` | 原函数、导数规则、终止性分开处理 |
| 混合抽样、TDR | 随机替换只支持特定 `_randint`边界 | 连续随机模型与真实 PRNG 是额外项目 |

这些是静态差异清单，不是一次失败的兼容性实验。`SUPPORTED_JAXPR.md`还区分“能翻译”和“能生成证书”。不能用前者替代后者。[J1]

### 5.2 建议的小试点

等数学层有一条有用定理后，再选 `_moments_from_weights`中的小型固定形状核心。先验证权重归一化和中心化 Gram 结构，或选 `InverseGaussian.from_expectation`的纯代数部分。不要先导入整个 GIG 求解器。

预计 1–2 人周可完成 primitive 清单与一个小桥接原型；若需重写导入器核心，预算另议。AI 可生成导入代码、证书和故意破坏目标后的反例检查。Xiang 应审查实数模型的含义、裁剪条件和覆盖范围。若两周后仍只有“能生成很多 Lean 文本”，没有连接实际函数的已检查规约，就停止扩展。

**翻译后仍要证明近似。** 即使完整 `_eval_quad`成功获得证书，得到的也是一个有限和算法。要证明它近似 $\log K$，仍须完成第 3.6 节的尾部与求积分析。要证明导数准确，还须比较相应矩，不能只比较函数值。

本次建议采用思想并独立设计最小接口，不复制或分发许可证不明的实现。根目录缺少许可证只是依赖与复用决策的边界，不是“不能证明 Bessel”的数学理由。若以后考虑直接复用，先核实授权。

## 6. 项目架构与审查制度

### 6.1 同仓库独立子项目

推荐 `formal/lean/`，不先建独立仓库。理由是 normix 的公式、坐标约定和近似策略仍会变化。一次变更可以同时更新代码映射、定理陈述和文档状态。Lean 不进入 Python 的安装依赖，不改变 JAX 运行路径。

**Lake**是 Lean 的包管理和构建工具。**持续集成（CI）**是在提交后自动执行检查。

| 方案 | 决策建议 | 理由 |
|---|---|---|
| 同仓库的独立 Lake 项目 | 首选；设置与 CI 约 2–4 人日 | 用同一提交记录代码和证明；保持 Python／Lean 构建独立 |
| 单独 `normix-lean`仓库 | 暂缓 | 可以独立发布，但需维护两边提交对应和文档漂移。适合出现独立维护团队后再拆 |
| 证明放在 Python 包内部并参与运行 | 否决 | 混淆数学证明与数值执行；增加安装和设备路径负担 |

以下结构都是提案，本次没有创建这些文件。

```text
formal/lean/
  lean-toolchain
  lakefile.toml
  lake-manifest.json
  Normix.lean
  Normix/
    Foundations/       可积性、矩、有限维坐标
    ExponentialFamily/ 定义、凸性、导数、对偶
    SpecialFunctions/  Bessel 积分、恒等式
    Distributions/     Gamma、IG、GIG 及边界
    Mixtures/          混合测度、GH 密度、矩、条件分布
    EM/                Q 分解、M 步、受限收敛
    Numerics/          有限求积、冻结几何、误差界
    Bridge/            可选：特定 jaxpr 的实数模型对应
  AxiomAudit.lean
  coverage.toml
  decisions.md
  Scratch/             未完成探索；正式模块禁止导入
```

依赖方向从基础数学到分布，再到混合与 EM。`Numerics`可以引用数学对象和有限矩定理。`Bridge`引用数学定理与生成程序。数学定义不得反过来依赖某次追踪形状或生成文件。

先固定样本空间和统计量约定。不需要一次设计一个覆盖所有概率模型的 Lean 类层次。第一版用有限维实向量、明确的 Lebesgue 参考测度和命名参数结构。处理 GIG 的正参数内部域；边界分布单独定义。代码的 `log_prob`可表示负无穷，而 Lean 的普通实数不能，所以精确概率层优先以非负密度建模，log-density 等式限于正密度域。

### 6.2 固定版本、构建与缓存

`lean-toolchain`固定 Lean 版本。`lakefile.toml`声明依赖。`lake-manifest.json`固定解析到的提交。不能追随 Mathlib 的移动分支而让一个原本通过的证明每天变化。

下面是配置形状示例，**未经运行**。版本与 SHA 来自提供的 jax-lean 配置，不代表已推荐或验证这一组合适用于新项目。可执行阶段先核实兼容性，再冻结。

```toml
name = "normixformal"
version = "0.1.0"
defaultTargets = ["Normix"]

[[require]]
name = "mathlib"
scope = "leanprover-community"
rev = "db584cd6d46c92f209a44c0f1c829460d327499d"

[[lean_lib]]
name = "Normix"
```

配套工具链示例为 `leanprover/lean4:v4.33.0`。首次依赖获取需要以后具备网络或完整本地依赖；本次不执行。

建议独立 Lean CI 作业，未来的命令形状如下；缓存命令的固定版本可用性仍需核实。

```bash
cd formal/lean
lake exe cache get
lake build
lake env lean AxiomAudit.lean
```

`lake exe cache get`常用于取得 Mathlib 已编译产物，来自工具使用记忆，本次未验证。它节省依赖编译时间，不应省略本项目构建。缓存键包括操作系统、Lean 工具链和依赖清单摘要。项目证明变化必须触发重新构建，不能把旧构建成功标志当证书。

`Normix.lean`应导入所有正式模块；CI 再核对正式目录与覆盖清单，避免未导入的文件被默认构建漏掉。`AxiomAudit.lean`也必须由 CI 显式执行。升级 Mathlib 放在独立变更中，记录修复工作量；不要与新数学结论混在一起。AI 可维护构建和依赖升级，Xiang 只需审版本冻结、证明范围变化及异常公理。

### 6.3 函数与定理的双向映射

映射必须细到函数和性质。下列 Lean 名称全部是**拟议名称**，不是现有 Mathlib 或 normix 定理。

| 标识 | 现有代码 | 拟议 Lean 定理 | 只覆盖什么 |
|---|---|---|---|
| NM-EF-001 | `ExponentialFamily.expectation_params` | `Normix.EF.grad_logPartition` | 正则连续族的数学公式，不含 JAX AD 正确性 |
| NM-EF-002 | `ExponentialFamily.fisher_information` | `Normix.EF.fisher_eq_cov` | 完整自然坐标；排除受限子族接口歧义 |
| NM-GIG-001 | `GIG.raw_moment` | `Normix.GIG.moment_formula` | $a,b>0$的精确矩，非数值误差 |
| NM-MIX-001 | `JointNormalMixture.rvs`、`NormalMixture.mean/cov` | `Normix.Mixture.pushforward_law`、`mean_cov` | 理想独立输入及所需矩 |
| NM-GH-001 | `GeneralizedHyperbolic.log_prob` | `Normix.GH.density_eq_integral` | 未加数值下界的数学密度 |
| NM-EM-001 | `BatchEMFitter._em_step` | `Normix.EM.likelihood_monotone` | 精确后验且 Q 改进的抽象一步 |
| NM-BES-001 | `_moments_from_weights`、`BesselMoments.cov` | `Normix.FiniteEF.cov_psd` | 有限正权重的实数协方差 |
| NM-BES-002 | `_eval_quad`、`log_kv_moments` | `Normix.Quadrature.frozen_derivatives` | 固定几何的偏导；误差另证 |

定义自己的映射格式即可，不必发明一个 Python 装饰器。拟议 `coverage.toml`条目可长这样：

```toml
[[claim]]
id = "NM-BES-001"
python_file = "normix/utils/bessel.py"
python_symbol = "_moments_from_weights"
theorem = "Normix.FiniteEF.cov_psd"
lean_file = "Normix/Numerics/FiniteMoments.lean"
theory_page = "docs/design/solvers_and_bessel.md"
scope = "exact-real finite positive weights; no rounding bound"
status = "planned"
spec_review = "pending"
source_commit = "51e053c57237a29c71b56d3a2a86988277477bd9"
```

以后补充参数域、假设、定理依赖、公理审计结果和检查提交。状态至少分“计划”“条件定理已证明”“分布实例已证明”“代码桥接已证明”“误差界已证明”。本报告中没有任何条目已经达到后四种状态。

docstring 的拟议标签放在 `Notes`中即可，不影响函数执行：

```text
Formal specification: NM-BES-001.
Exact-real covariance identity; excludes rounding error.
See :doc:`/design/solvers_and_bessel`.
```

该标签不宣称源码已被完整验证。公开页面可以写“对应 Lean 定理：……；范围：……；检查提交：……”，链接到实际存在的公开源码固定提交或生成的 Lean 文档锚点。不要猜 URL、定理锚点或预先挂通过徽章。`docs/theory`及 docstring 不能链接到 `dev-notes`；本报告仍只作为内部研究材料。

映射与初始公开链接机制约 2–3 人日。AI 维护链接和符号存在性；Xiang 审每个标签到底承诺哪一层。源码或数学页面变化时，CI 要求重新确认相关映射；仅符号名字还在，不能视为语义未变。

### 6.4 防止规约被悄悄削弱

每个目标先有一张简短“命题卡”：数学公式、参数域、支持集、可积性、量词、对应代码、排除范围。Xiang 先审命题卡，再让 AI 写证明。

审查至少问这些问题。

- 是否把“所有合法参数”换成了一个方便的小区间，却仍保留原覆盖标签？
- 是否把密度归一化、独立性或 EM 单调性本身放进了假设？这些可作为抽象引理前提，但不能冒充具体实例已完成。
- 是否存在满足假设的例子？例如既假设所有矩有限，又包含 InverseGamma 的发散区域，会产生错误或空泛范围。
- 是否把代码中的下界、有限轮数、失败值或边界点从对应关系里删掉？
- 是否把 `fisher_information`定义成 Hessian，以此回避与得分协方差的联系？
- 是否只证了一维或固定长度 2，却将说明扩写成任意维数？

证明作者可以改证明，不可静默改已审陈述。若证明失败，先报告数学缺口、库缺口还是工程成本。修改假设必须重新审命题卡，并记录旧目标为什么没有完成。AI 可以列反例候选，Xiang 决定接受的理论范围。

**`sorry`预算。** 正式导出层预算为零。探索目录可以暂存，但每个缺口要有标识和目标，禁止正式模块导入。建议试点同时开放不超过 5 个证明洞；达到上限先收敛已有目标。这个数量是管理建议，不是 Lean 限制。只搜文本中的 `sorry`不够，因为不完整证明可能从导入文件传入。

**公理审计。** 使用下面的命令检查每个正式导出定理的传递依赖。名称仍为提案，命令本次未执行。

```lean
#print axioms Normix.FiniteEF.cov_psd
#print axioms Normix.GIG.moment_formula
```

审计要拒绝 `sorryAx`和未经批准的项目公理。依据 Lean 使用记忆，普通经典数学可能依赖 `propext`、`Classical.choice`、`Quot.sound`等基础公理；应显式列出可接受集合，不要求不现实的“所有公理为零”。任何新增项要说明来源。对内核之外计算路径的额外信任也要记录，不能仅因构建绿灯就忽略。

公理列表仍检查不出错误的定义或过强的前提。`#print axioms`不能代替 Xiang 审规约。正式标签还需记录定义与陈述摘要，AI 不得用替换定义的方式“完成”原任务。

以后做程序桥接时，还应有故意改坏运算、坐标或分支后证书失败的检查。jax-lean 的 `tests/test_certification.py`与 `test_review_regressions.py`已有这种测试源码。本次只阅读，没有运行。[J5]

### 6.5 人与 AI 的分工

| 工作 | AI 主做 | Xiang 主审 | 预计投入 |
|---|---|---|---|
| 固定版本库盘点 | 找定义、写最小编译例子、列缺口 | 归一化和参数约定 | AI／工程 2–4 人日；Xiang 2–3 小时 |
| 每张命题卡 | 从源码和文档抽取假设、给反例 | 定理是否有用且表达正确 | 简单目标 30–60 分钟审查；分析目标 2–4 小时 |
| 证明实现 | 小引理、tactic、反复编译 | 关键数学路线和非显然假设 | 依第 3 节估计 |
| 定理与代码对应 | 维护映射、标出夹紧和近似 | 是否足以给 API 打标签 | 每组 1–2 小时 |
| 特殊函数误差 | 展开已选分析方案、辅助不等式 | 支配函数、误差预算、适用区域 | 通常需每周 2–4 小时，必要时请熟悉数值分析／Lean 的人复核 |

Xiang 不必逐行审 tactic。但需要逐条读数学陈述和它的定义。AI 不能自行决定降低规格以换取通过。

建议所有变更记录“现有约束、提案、否决原因、证据日期、决定者、重开条件”。本次只把它们写在本报告。以后获准实施时，才将已接受决策迁入 `dev-notes/design/design.md`的正式记录；不要把本报告建议误标为 Xiang 已批准。

## 7. TLA+ 和 Hypothesis 各放在哪里

**TLA+**是一种描述系统状态和状态变化的规约语言。这里不要求先有并发。一个带停止标志、失败回退和执行路径选择的顺序程序，也能建成状态机。状态机就是一组状态和允许的下一步变化。

**安全性质**说坏事永不发生，例如停止后已接受模型不再被覆盖。**活性性质**说某件事最终会发生，例如满足某些调度和终止前提后最终返回。活性证明通常还需公平性条件，即某个一直可执行的步骤不能永远被忽略。

TLA+ 的模型检查工具可穷举给定有限模型中的状态和变化，寻找违反性质的路径。TLC 是这类工具。检查有限配置通过，不表示任意大小系统已经被证明，也不表示 Python 自动实现了这个模型。TLA+ 还可以配合演绎证明；本项目没有必要一开始就承担这层工作。以上是工具知识记忆，本次未运行或核实版本。

**Hypothesis**是 Python 的性质测试工具。人给出输入生成规则和应满足的性质，工具执行真实代码，并尝试缩小失败输入。它检查有限次执行，不是普遍证明。它也有状态化测试，能测试操作序列；TLA+ 对状态转移的建模与穷举仍是另一层工作。

| 工具 | 适合 normix 的问题 | 不能由此推出 | 小试点成本与分工 |
|---|---|---|---|
| Lean | GIG 归一化、混合矩、EM 的数学假设 | 实际浮点代码自动正确 | 见路线；AI 写证明，Xiang 审数学 |
| TLA+ | `_fit_scan`停止后冻结、失败保留旧模型、后端路径协议 | Bessel 精度、Q 改进、求解器数值收敛 | 2–5 人日模型试点；AI 建状态图，Xiang 审状态抽象与协议 |
| Hypothesis | 真实参数往返、合法域边界、后端一致性、停止序列 | 所有输入都满足性质 | 2–5 人日设计并接入少量性质；AI 编写策略，Xiang 审输入域与容差 |

`BatchEMFitter._fit_scan`的 carry，即循环携带状态，包含模型、`converged`、`diverged`、`solver_failed`、计数和历史似然。可以把数值步骤抽象成“候选有限／非有限”“停止／继续”等输入。先检查三个具体性质：[N6]

1. 一旦某停止标志已经生效，后续填充迭代保持模型、标志和有效迭代计数。
2. 候选混合参数非有限时，返回上一个可用模型并标记求解失败。
3. 请求的后端和设备路径不被失败回退偷偷更换。

这里把“数值步骤返回有限值”作为状态机输入是合理的抽象。但不能把“有限”解释为“参数必然合法”，也不能把状态机最终到达停止状态解释为数学上收敛。应另有实际分支与模型的对应审查，或状态化测试。

后端协议必须保持现有 S9／S11：`backend='jax'`在调用者选择的 JAX 设备执行，设备也可以是 CPU；`backend='cpu'`才选择显式 NumPy／SciPy 路径。形式化更容易不是自动 offload 的理由。本地 `tests/test_no_host_callbacks.py`已有源码和嵌套 jaxpr 检查，比单纯证明一个抽象“无回调”状态机更直接。[N7]

因此不需要给全部类层次建 TLA+ 模型。若问题只是纯函数的代数关系，Lean 或测试足够。若要审停止、恢复、批次状态，或以后加入并发调度与检查点，TLA+ 才有更明确收益。先保留小试点选项，不让它成为 Lean 数学项目的依赖。

## 8. 决策记录与不做的事

以下记录日期均为 2026-10-24。除“现有约束”外，决定者均为本报告作者，状态为建议，**不是 Xiang 已批准**。否决指本报告建议否决当前方案；不表示对工具能力作永久判断。

| 标识／状态 | 范围与决定 | 原因和本地证据 | 重开条件 |
|---|---|---|---|
| FV-00／现有约束 | normix 保持 JAX 原生，尊重用户后端和设备 | 用户本次要求；设计 S9／S11 的既有决定 | 只有 Xiang 明确修改约束 |
| FV-01／建议 | 同仓库 `formal/lean/`，Python 无 Lean 运行依赖 | 公式与实现需同提交对应；Lake 能独立构建 | 独立维护团队或跨项目数学库形成后考虑拆仓库 |
| FV-02／建议 | 先有限矩，再连续概率，再算法误差 | `_moments_from_weights`有直接有限和结构；连续 Bessel 的积分工程尚缺 | 固定 Mathlib 审计显示关键分析已齐备，可提前连续部分 |
| FV-03／建议 | 数学、算法、近似误差、机器执行分层标记 | jax-lean 自身承认实数语义与导入器边界 | 有精确机器数和编译对应证明后增加覆盖标签 |
| FV-04／建议否决 | 直接把 jax-lean 作为整包依赖 | 许可证未明、JAX 版本差异、循环和常量证书缺口 | 授权明确且小试点确实节省维护成本 |
| FV-05／建议否决 | 为了证明改为 CPU／SciPy 或全改写 Lean 运行时 | 违反 FV-00；数学证明不要求改变数值后端 | 仅限另获授权的独立参考实现，不能暗改生产路径 |
| FV-06／建议否决 | 同时形式化全部网站与所有数值路径 | 没有固定版本库盘点，也没有成本证据 | 前两阶段产出稳定，Xiang 明确批准更大预算 |
| FV-07／建议暂缓 | 全部浮点、XLA、GPU、PRNG 的端到端证明 | 所需机器语义远超本报告首期目标 | 出现必须有严格误差保证的具体短内核与预算 |
| FV-08／建议否决 | 用一套“所有分布都是完整 EF”的 Lean 抽象 | joint VG／NIG／NInvG 是受限子族，边缘混合也不具备该接口意义 | 只有在准确区分结构后才扩展统一抽象 |
| FV-09／建议暂缓 | 用 TLA+ 覆盖整个软件架构 | 大量纯函数和数学层没有状态空间收益 | 具体停止／恢复协议或并发流程需要验证 |

还应明确否决这些替代路线。Coq／Rocq 与 Isabelle/HOL 是其他证明助手。Dafny 用前置条件、后置条件和循环不变式描述程序合约；Verus 用类似方法验证 Rust 程序。前置条件是调用前的要求，后置条件是返回时的保证，循环不变式是每轮需要保持的性质。

| 替代方案 | 本项目为何暂不采用 | 不是能力否定；何时重开 |
|---|---|---|
| Coq／Rocq | 本次目标已选 Lean + Mathlib；换库要重做依赖和训练成本评估，不能免费复用 Lean 证明 | 它有成熟形式化数学生态。若有熟练维护者和现成所需分析库，花 1–2 人周做同题比较再决定 |
| Isabelle/HOL | 高阶逻辑与分析库是可行路线，但本项目没有已验证的 GIG／Bessel 基础优势；现在双栈会增加维护 | 若已有相关形式化成果，比较一个连续积分目标；估计 1–2 人周 |
| Dafny／Verus | 更适合程序合约、循环与数据结构。Verus 面向 Rust；normix 是 Python/JAX。特殊函数和测度积分仍需外部数学基础 | 若未来有独立 Rust 内核或离散状态组件，可单独评估；不为工具迁移主实现 |
| 只写在 docs 中 | 仍是必要的可读推导，但不能机械检查假设、换元和导数条件；本次已发现具体文档不一致 | 低优先级研究命题可以暂留纸面，但不能标为已形式化 |
| 只做 Hypothesis | 能触达真实代码，成本低，但有限样本不能证明积分和所有参数结论 | 保留为补充，不作为数学证明替代 |

工具比较是一般知识与项目适配判断，没有在本次做横向实验。不要从“暂不选”推断其他工具无法做概率论。

首期还不做以下承诺。

- 不写“浮点 Hessian 对所有合法输入严格半正定”。应先证实数结构，再决定舍入允许的误差。
- 不写“所有 EM 运行都单调且收敛到全局 MLE”。先区分精确算法、近似分支和不同目标。
- 不把 `backend='jax'`默认为 GPU，也不把显式 CPU 后端的成功当作 JAX 变换或设备行为的证据。
- 不把一次有限形状证书推广为任意维数；不把 `custom_jvp`的原函数翻译当作其导数规则已证明。
- 不通过 `axiom`宣告 Bessel 恒等式，再将依赖它的 GIG 标为无条件完成。
- 不将 Gamma、NIG 等专门运行时公式统一替换成 GH。数学复用与代码执行策略是两件事。

## 9. 分阶段路线与退出条件

每阶段有自己的交付与停止条件。停止某条路线不要求删除已证明的基础成果。所有时间都指今后可安装依赖、可运行 Lean 的环境；本次没有开始这些阶段。

| 阶段 | 交付与通过条件 | 预算／人审 | 停止或缩小范围的条件 |
|---|---|---|---|
| P0：核对规格与依赖 | 固定 Lean／Mathlib；盘点分析缺口；5–8 张命题卡；确认统计量顺序和文档审查项 | 约 1 人周；Xiang 3–5 小时 | 若没有能构建的固定依赖，或命题定义未获认可，不进入证明扩张 |
| P1：小而有用的基础 | 参数转换、有限正权重归一化、Gram 半正定、固定几何的一阶矩关系；至少 3 条连接真实函数的定理；零正式 `sorry` | 2–3 人周；Xiang 每周 1–2 小时 | 第 3 周仍只有抽象演示或被削弱的陈述，停止增加模块，审是否需要熟练 Lean 协作者 |
| P2：第一条连续概率链 | Gamma 归一化与矩；理想混合变换和均值／协方差；至少一个连续分布实例 | 4–8 人周；Xiang 每周 2 小时 | 4 周后仍卡在通用可积性且无实例，暂保留有限数学层，不同时开启 Bessel 与 EM |
| P3：Bessel 与 GIG | 积分可积、正性、对称；GIG 归一化与矩；再做 order／argument 导数 | 6–12 人周；Xiang 每周 2–4 小时 | 4 周后仍无关键尾界路线，或需整套新分析理论，暂停该分支；可先做条件性 GH／EM 引理并明确状态 |
| P4：GH 与 EM | GH 密度等于混合积分、条件 GIG、正态 M 步；抽象单步不降；列出实现差距 | 4–8 人周；Xiang 每周 2–3 小时 | 无法把实际下界、近似求解与抽象假设对应时，停在数学定理，不给实际 fitter 完整认证标签 |
| P5：有界数值目标 | 选一个紧参数区域和误差指标，证明 Bessel 值或某个导数的显式界 | 2–6 人月；Xiang 与数值分析审查者定期复核 | 前 2–3 周找不到可计算的尾部／求积界，或界远大于使用需求，记录失败原因并暂停 |
| 可选桥接 | 一条真实小 jaxpr 到已证数学规约；记录导入器信任边界 | 1–2 人周；Xiang 2–4 小时 | 超预算仍无可检查规约，停止导入器扩张 |
| 可选状态机 | `_fit_scan`的三个状态性质及代码映射 | 2–5 人日；Xiang 1–2 小时 | 模型只是重复 if 条件，未覆盖测试难以检查的序列，则不扩大 TLA+ 使用 |

P0–P1 是建议先批准的范围。选定核心连续理论走完 P2–P4，应按数月预算；不要承诺“几周验证整个网站”。渐近、采样器细节、EM 平稳性和数值误差全部纳入，可能超过一个熟练人员一年的工作，且不宜在未做 P0 前给固定总价。

每个阶段结束记录四件事：已证命题、尚未证明的前提、与源码的差距、实际人时。由这些数据决定下一阶段，不按定理数量判断进展。

## 10. 排序建议

1. **先做 P0–P1，同仓库独立 Lean 项目。** 预算 3–4 人周。优先参数约定、有限指数和与 `BesselMoments`的协方差结构。AI 主做实现；Xiang 先审 5–8 张命题卡，再审覆盖标签。这是离现有代码最近、分析基础负担较小的一组成果。
2. **再做连续概率与 GIG／GH 的一条完整证明链。** 按 P2–P4 分段批准，累计按数月计。优先混合表示、矩、归一化、后验共轭和 M 步。AI 可承担证明工程；Xiang 审定义、边界及可积性。
3. **EM 先做到有条件的单步不降。** 在密度基础之后投入约 3–6 人周。平稳点收敛单独立项，不与普通 EM 单调混为一个承诺。Xiang 必须决定哪些数值近似值得进一步误差分析。
4. **jaxpr 桥接只做限时试点。** 数学定理已有用处后投入 1–2 人周。采用可读函数、独立 IR 与等式证书的思路。保留导入器、机器浮点和 PRNG 的明确边界，不改变 JAX 后端契约。
5. **Bessel 误差界、真实采样器、TLA+ 按具体需求选。** 误差界是月级研究；状态机可用 2–5 人日验证收益。Hypothesis 可同时补真实代码的边界测试，但没有必要把安装它设为数学项目的前置条件。本次均未实施。

## 11. 给 Xiang 的待决问题

这些问题不影响本报告完成，也不要求现在逐项回答。

1. 首个验收目标选哪一个：有限 Bessel 矩结构，还是 Gamma／理想混合抽样的连续概率链？建议先选前者。
2. 是否接受先固定 $a,b>0$的内部域，再单独处理 Gamma／InverseGamma 精确边界？是否还需要极限过程的证明？
3. 对代码覆盖标签，你更看重公式来源可追溯，还是具体误差上界？后者需要先指定参数范围、精度和误差指标。
4. 能否每周留出约 1–2 小时审命题卡？特殊函数阶段是否有额外的 Lean／数值分析协作者预算？
5. EM 的长期目标是数学单调性、聚点平稳性，还是带当前下界和回退的实际拟合器保证？三者成本差异很大。
6. 同仓库子项目是否符合你的维护方式？若未来拆仓库，谁负责两边提交与公开定理链接同步？

## Sources

以下路径是本次静态阅读的依据。路径中的函数名用于定位，不表示已运行。N8 是主题清单，未做全站逐式核验。外部作品只列确定的名称与年份；没有访问外部链接。概览里的社交媒体、近期论文和成功率数字没有在这里重新核实，也没有用来估算证明成本。

- **规则**：[AGENTS.md](../../../AGENTS.md)、[dev-notes/README.md](../../README.md)、[项目目录规则](../../../.cursor/rules/project-overview.mdc)、[交叉链接规则](../../../.cursor/rules/docs-cross-links.mdc)、[设计记录规则](../../../.cursor/rules/maintain-design-docs.mdc)、[理论维护规则](../../../.cursor/rules/maintain-theory-docs.mdc)、[Python 约定](../../../.cursor/rules/coding-conventions.mdc)。使用了 [how](../../../.cursor/skills/how/SKILL.md)、[principles](../../../.cursor/skills/principles/SKILL.md)、[agent-maintenance](../../../.cursor/skills/agent-maintenance/SKILL.md)、[unslop](../../../.cursor/skills/unslop/SKILL.md)。
- **起点**：[overview_report.md](00_overview_report.md)，输入说明日期为 2026-10-19。
- **N1／分布理论**：[gig.md](../../../docs/theory/gig.md)、[gh.md](../../../docs/theory/gh.md)、[em_algorithm.md](../../../docs/theory/em_algorithm.md)。
- **N2／指数族**：[exponential_family.py](../../../normix/exponential_family.py)、[normal.py](../../../normix/distributions/normal.py)、[generalized_inverse_gaussian.py](../../../normix/distributions/generalized_inverse_gaussian.py)、[公开指数族设计](../../../docs/design/exponential_family.md)。重点为 triad、`natural_params`、`_implicit_natural_params`和 `raw_moment`。
- **N3／具体分布**：[gamma.py](../../../normix/distributions/gamma.py)、[inverse_gamma.py](../../../normix/distributions/inverse_gamma.py)、[inverse_gaussian.py](../../../normix/distributions/inverse_gaussian.py)、[generalized_hyperbolic.py](../../../normix/distributions/generalized_hyperbolic.py)、[normal_inverse_gaussian.py](../../../normix/distributions/normal_inverse_gaussian.py)、[variance_gamma.py](../../../normix/distributions/variance_gamma.py)、[normal_inverse_gamma.py](../../../normix/distributions/normal_inverse_gamma.py)。重点为密度、矩、CDF、抽样和期望参数反解。
- **N4／混合**：[joint.py](../../../normix/mixtures/joint.py)、[marginal.py](../../../normix/mixtures/marginal.py)、[混合设计](../../../docs/design/mixtures.md)。重点为 `rvs`、`_ambient_moment_contract`、`_posterior_gig_params`、`_mstep_normal_params`、`_rescale`和一维混入类。
- **N5／Bessel 与抽样**：[bessel.py](../../../normix/utils/bessel.py)、[rvs.py](../../../normix/utils/rvs.py)、[bessel_moment_kernel.md](../../tech_notes/bessel_moment_kernel.md)、[gig_rvs.md](../../tech_notes/gig_rvs.md)。技术笔记的历史性能和精度数字没有在本次复测。
- **N6／拟合**：[em.py](../../../normix/fitting/em.py)、[eta.py](../../../normix/fitting/eta.py)、[solvers.py](../../../normix/fitting/solvers.py)、[EM 设计](../../../docs/design/em_framework.md)、[GIG 反解旧笔记](../../tech_notes/gig_eta_to_theta.md)。重点为 `_fit_scan`、`_em_step`、`_mcecm_step`、命名统计量和求解器边界。
- **N7／既有架构决定**：[design.md](../../design/design.md)、[solvers_and_bessel.md](../../design/solvers_and_bessel.md)、[公开求解器设计](../../../docs/design/solvers_and_bessel.md)、[ARCHITECTURE.md](../../ARCHITECTURE.md)、[无 host callback 检查源码](../../../tests/test_no_host_callbacks.py)。S5／S9／S10／S11 是本地记录，相关外部 PR 链接未打开。
- **N8／其余理论主题**：[varentropy.md](../../../docs/theory/varentropy.md)、[online_em.md](../../../docs/theory/online_em.md)、[shrinkage.md](../../../docs/theory/shrinkage.md)、[factor_analysis.md](../../../docs/theory/factor_analysis.md)、[mean_risk_optimization.md](../../../docs/theory/mean_risk_optimization.md)、[cvar_derivatives.md](../../../docs/theory/cvar_derivatives.md)、[transaction_costs.md](../../../docs/theory/transaction_costs.md)、[enb.md](../../../docs/theory/enb.md)、[generalized_enb.md](../../../docs/theory/generalized_enb.md)、[quadrature_scenarios.md](../../../docs/theory/quadrature_scenarios.md)。
- **J1／jax-lean 范围与版本**：[README.md](../../../.research-inputs/jax-lean/README.md)、[ARCHITECTURE.md](../../../.research-inputs/jax-lean/ARCHITECTURE.md)、[SUPPORTED_JAXPR.md](../../../.research-inputs/jax-lean/SUPPORTED_JAXPR.md)、[SEMANTICS_REFERENCES.md](../../../.research-inputs/jax-lean/SEMANTICS_REFERENCES.md)、[lakefile.toml](../../../.research-inputs/jax-lean/lakefile.toml)、[lake-manifest.json](../../../.research-inputs/jax-lean/lake-manifest.json)、[lean-toolchain](../../../.research-inputs/jax-lean/lean-toolchain)、[pyproject.toml](../../../.research-inputs/jax-lean/pyproject.toml)。normix 版本约束另见 [pyproject.toml](../../../pyproject.toml) 与 [uv.lock](../../../uv.lock)。
- **J2／翻译与证书**：[translate.py](../../../.research-inputs/jax-lean/python/jaxlean/translate.py)、[导入器](../../../.research-inputs/jax-lean/python/jaxlean/importers/jaxpr.py)、[certify.py](../../../.research-inputs/jax-lean/python/jaxlean/certify.py)、[Core/Jaxpr.lean](../../../.research-inputs/jax-lean/JaxLean/Core/Jaxpr.lean)、[Core/RealOps.lean](../../../.research-inputs/jax-lean/JaxLean/Core/RealOps.lean)、[Verification/Certificate.lean](../../../.research-inputs/jax-lean/JaxLean/Verification/Certificate.lean)、[CertifiedMean.lean](../../../.research-inputs/jax-lean/examples/certificates/generated/CertifiedMean.lean)。
- **J3／导数示例**：[autodiff/code.py](../../../.research-inputs/jax-lean/examples/autodiff/code.py)、[autodiff/proofs/Jvp.lean](../../../.research-inputs/jax-lean/examples/autodiff/proofs/Jvp.lean)。
- **J4／随机模型**：[Stdlib/Random.lean](../../../.research-inputs/jax-lean/JaxLean/Stdlib/Random.lean)、[Stdlib/BatchProbability.lean](../../../.research-inputs/jax-lean/JaxLean/Stdlib/BatchProbability.lean)、[rewrites/random_spec.py](../../../.research-inputs/jax-lean/python/jaxlean/rewrites/random_spec.py)、[RandomProgramProofs.lean](../../../.research-inputs/jax-lean/examples/random_program/proofs/RandomProgramProofs.lean)。
- **J5／测试源码，仅阅读**：[test_certification.py](../../../.research-inputs/jax-lean/tests/test_certification.py)、[test_review_regressions.py](../../../.research-inputs/jax-lean/tests/test_review_regressions.py)、[test_random_translation.py](../../../.research-inputs/jax-lean/tests/test_random_translation.py)。

外部理论依据与待核实项：

- Dempster、Laird、Rubin，*Maximum Likelihood from Incomplete Data via the EM Algorithm*（1977）。EM 分解与单调性依据文献记忆；名称年份也见本地 [references.md](../../../docs/references.md)。
- Wu，*On the Convergence Properties of the EM Algorithm*（1983）。名称年份有把握；精确假设、定理编号及其与 GH 的适用关系 **to verify**。
- Wainwright、Jordan，*Graphical Models, Exponential Families, and Variational Inference*（2008）。用于指数族与凸对偶背景，依据文献记忆。
- *NIST Handbook of Mathematical Functions*（2010）。用于 Bessel 恒等式与渐近的文献背景，依据记忆。在线 DLMF 的当前版本和具体公式位置 **to verify**；未据此声称已查原式。
- Devroye，*Random Variate Generation for the Generalized Inverse Gaussian Distribution*（2014）。本地技术笔记和参考文献记录；原文证明及当前离散包络的对应 **to verify**。
- Cappé、Moulines，*On-line Expectation-Maximization Algorithm for Latent Data Models*（2009）。名称年份见本地参考文献；随机收敛假设与 normix 自定义更新规则的对应 **to verify**。
- Lean、Mathlib、TLA+／TLC、Hypothesis、Coq／Rocq、Isabelle/HOL、Dafny、Verus 的一般工具描述来自使用知识或文献记忆。当前版本能力没有外部核实；除本地 jax-lean 源码中明确出现的名称外，不提供未经确认的 Mathlib 定理名。
