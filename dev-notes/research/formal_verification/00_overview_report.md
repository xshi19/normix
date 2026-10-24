# 形式化验证在普及吗？normix 该怎么用（入门版）

2026-10-19 · 写给 Xiang

## 一句话结论

在升温，但不是全面普及。真正的变化在于 agent 开始替人写模型和证明，门槛降了很多。你的项目里 normix 最适合用，适合做的是验证数学，而不是验证浮点代码。顺序是：先用 hypothesis 做性质测试，再用 Lean 证几条恒等式，jax-lean 先别用。

## 概念入门（101）

先把词讲清楚。后面所有内容都建立在这些词上。

**测试和形式化验证的区别。** 测试只检查你挑出来的那几个输入。比如用 3 组参数算 GIG 方差，检查结果都大于 0。形式化验证是写一个数学命题，例如“对所有合法参数，方差 ≥ 0”，再让机器检查一份覆盖所有情况的证明。测试只能说明“这几次没出错”，证明能说明“在给定前提下不会出错”。

**规约（spec）。** 规约是一句精确的话，写明代码应该满足什么。例如“EM 每迭代一步，对数似然不下降”。规约写错了，后面的证明再严谨也白费。所以审规约比审证明更要紧。

**定理证明器。** 一种程序，它检查证明的每一步是否合乎逻辑规则。证明由谁写都可以：人、自动程序、AI。最后都要过它这一关。Lean 的核心检查器叫 kernel。

**交互式证明和自动证明。** 交互式证明是人（或 agent）一步一步告诉证明器怎么推，Lean 属于这一类。自动证明是只给出命题和少量提示，由求解器自己去找证明，Dafny 属于这一类。前者表达力强但费劲，后者省力，但遇到复杂数学容易卡住。

**Lean 和 Mathlib。** Lean 4 既是编程语言，又是证明助手。Mathlib 是社区维护的数学库，实数、积分、导数、概率等定理都在里面，你可以直接引用。

**tactic。** tactic 是写在 `by` 后面的一条证明指令，比如 `ring` 自动证明多项式恒等式，`positivity` 证明“某个量 ≥ 0”。tactic 不会跳过检查，它生成的证明照样要经过 kernel。下面是一段示意，表达“方差非负”的数学核心：

```lean
-- 示意代码，未经验证
import Mathlib
-- 方差 = E[(X-μ)^2]，单点情形：平方非负
example (x μ : ℝ) : 0 ≤ (x - μ)^2 := by positivity
```

**Dafny 的前置条件、后置条件和不变式。** `requires` 是前置条件，即调用方必须保证的东西。`ensures` 是后置条件，即函数返回时承诺的东西。`invariant` 是循环不变式，即循环每一轮都保持成立的性质。下面示意“EM 似然不降”怎么写成规约：

```dafny
// 示意代码，未经验证；LogLik、EMStep 为抽象函数
method RunEM(theta0: Params, n: nat) returns (theta: Params)
  ensures LogLik(theta) >= LogLik(theta0)
{
  theta := theta0;
  var i := 0;
  while i < n
    invariant LogLik(theta) >= LogLik(theta0)
  {
    theta := EMStep(theta);   // 需要引理：LogLik(EMStep(t)) >= LogLik(t)
    i := i + 1;
  }
}
```

真正的难点在注释里那条引理：单步不降。它本身就是 EM 理论里那个 Jensen 不等式的论证。

**SMT 求解器。** 一种自动判定逻辑公式能否成立的引擎，常用的是 Z3。Dafny、Verus 会把规约翻成公式交给它去判定。它擅长线性算术和有限结构，碰到 Bessel 函数、积分这类就基本无能为力。

**jaxpr。** JAX 对一个函数做追踪之后得到的中间表示，是一串基本运算，例如 `c = add(a, b); d = mul(c, 2.0)`。它比 Python 源码干净得多，所以 jax-lean 选它作为翻译的入口。

**实数和浮点数的差别。** 数学里 (a+b)+c = a+(b+c)。float32 里，取 a = 2^24、b = 1、c = −2^24，左边算出 0，右边算出 1，因为 a+b 被舍入回了 a。你 vault 里那篇 review 用的正是这个例子。在实数上成立的证明，不能自动搬到浮点代码上。

**基于性质的测试。** 不写具体样例，而是写一条性质，比如“θ 转成 η 再转回来，等于原来的 θ”，让工具随机生成成百上千组输入去试，找到反例后再自动缩小成最简形式。Python 里用 hypothesis。它不是证明，但成本低，而且能跑真实代码。

## 三条材料各说了什么

**1. @srush_nlp 的 jax-lean**

- 做法：用 `make_jaxpr` 取出 Jaxpr，在 Lean 里实现一个解释器，生成可读的 Lean 函数，再自动证明它和 IR 等价，也就是生成的 certificate。[仓库](https://github.com/srush/jax-lean)，[ARCHITECTURE.md](https://github.com/srush/jax-lean/blob/main/ARCHITECTURE.md)
- 例子：向量化写法和循环写法等价、Transformer 对置换等变、jax.grad 等于 mathlib 的 deriv，还有蒙特卡洛估计量的方差：16 次掷骰取均值，方差是 35/192。[项目页](https://srush.github.io/jax-lean/)，[证明文件](https://github.com/srush/jax-lean/blob/main/examples/randint_monte_carlo/proofs/RandintMonteCarloProofs.lean)
- 限制：按实数语义处理，不管舍入。它支持的是 JAX 0.8.0。随机数只认 `_randint`，并且明说不保证真实 PRNG 均匀、独立。确定性原语只有 exp、log、sqrt、sin、cos、tanh 这类，没有 Bessel 函数。[SUPPORTED_JAXPR.md](https://github.com/srush/jax-lean/blob/main/SUPPORTED_JAXPR.md)
- 仓库根目录没有 LICENSE 文件（我克隆后看过）。所以正态方差混合、肥尾分布目前都做不了。
- 我读了你 vault 里的 *jax-lean: a beginner's review*（在 wukong 上）。它补充了几点：
- 流程分两步。第一步是翻译证书，证明“导入的程序和可读函数结果一致”。第二步要另外证明可读函数满足你的规约。翻译再忠实，原算法错了照样是错的。
- Python 侧的导入器本身没有被证明，属于需要信任的部分。
- 形状在追踪时就固定了。对长度为 2 和 3 的向量证过的结论，不能自动推广到其他长度。
- 谜题集 21 题做了 20 题，剩下一题是因为证书检查太贵。

**2. @poteto 与 create-formal-verification-skill**

- 说明：这条帖子发于 10 月 18 日美东 17:05，不是今天。她今天的帖子只有转推和回复，跟这个话题无关。[今日帖子来源：她的时间线](https://x.com/poteto)
- 帖子内容：pstack v0.15.27 新增了 `/create-formal-verification-skill`，还在 beta。它是一个“生成 skill 的 skill”，会按你要验证的性质，生成一个项目内 skill 和一套验证。她举的例子有用 TLA+ 建模状态机和并发、用 Rust 的 kani 查未定义行为。她认为形式化验证过去没进主流，是因为建模和写证明太难。她已经用 TLA+ 在自己的代码里找到了几个 bug。[原帖](https://x.com/poteto/status/2111926622278521253)
- skill 文件确实在：box 上的 pstack 缓存里有 `create-formal-verification-skill`、`create-verification-skill`，以及对应的两个 maintain 版本。它的要点：
  - 挑“出 bug 代价最高、测试最难覆盖”的地方。
  - 让问题决定工具。
  - 如果这门语言没有验证器，就直说，然后用生成输入去测真实代码。
  - 每条性质都要在故意改坏的代码上跑失败一次。
  - 说清楚验证范围和前提假设。
  - 记录覆盖了哪些文件，以及同步时的 commit。
  - 来源：[cursor/plugins pstack](https://github.com/cursor/plugins/tree/main/pstack)

**3. @bcherny 的帖子（9 月 22 日）**

他用 Opus 5.5 加 Lean 对 Claude Agent SDK 做形式化验证，几条简短的 prompt 就产出了 16 个 PR，修了多处 bug 和竞态。他说 TLA+ 也好用，有时会把 Lean 和 TLA+ 合用，查数据流、并发和状态管理的问题。他自己两门语言都不熟，结尾问形式化验证是不是编程（至少是找 bug）的未来。附带的视频我没有看。[原帖](https://x.com/bcherny/status/2102543349102338309)

## 行业现状

- **工具本身**：Lean 4 加 [Mathlib](https://github.com/leanprover-community/mathlib4) 是数学形式化的主力。[Dafny](https://dafny.org)、[Verus](https://github.com/verus-lang/verus)（Rust）、[F*](https://fstar-lang.org) 都属于“写代码时顺带写规格”的路线。[TLA+](https://lamport.azurewebsites.net/tla/tla.html) 管设计层面的并发和状态机。这几样都不新，新的是 agent 能替你写。
- **LLM 写证明**：
  - AlphaProof 用强化学习在 Lean 里做竞赛题。[Nature 论文](https://www.nature.com/articles/s41586-025-09833-y)
  - DeepSeek-Prover-V2 用子目标分解加 RL。[arXiv 2504.21801](https://arxiv.org/abs/2504.21801)
  - arXiv 2610.19298 用验证器的诊断信息当稠密奖励：按证明义务完成的比例给分，再加一道完整性审计，防止模型偷改规格或写 assume。35B 模型在 DafnyBench 代码生成上从 66.1% 提到 93.5%。论文也指出，难点仍在写证明注释。[arXiv 2610.19298](https://arxiv.org/abs/2610.19298)
  - Lean4AgenticRL 把 agent 的工作流写成 Lean 里的前置和后置条件，用来给多层过程奖励。但其中不少谓词还是靠 LLM 评判，作者自己划了形式化的边界。[arXiv 2610.22448](https://arxiv.org/abs/2610.22448)
- **一个共同的坑**：模型会削弱规格来“通过验证”。2610.19298 专门做了审计来防这一点。所以你自己审规格比审证明更重要。
- **数值和浮点代码**：可以做，但范围窄。
  - [Gappa](https://gappa.gitlabpages.inria.fr/) 和 [FPTaylor](https://github.com/soarlab/FPTaylor) 能给小段直线式浮点表达式算舍入误差界。
  - 区间算术能给出有保证的包络。
  - 涉及迭代、特殊函数、优化收敛的整段代码，目前基本做不到端到端验证。jax-lean 也明确不管浮点。
- **Python/JAX 能走的路**：
  - [hypothesis](https://hypothesis.readthedocs.io/)：基于性质的测试，成本最低。
  - [CrossHair](https://github.com/pschanely/CrossHair)：用 SMT 符号执行查 Python 合约，对 JAX 数组基本用不上。
  - [Nagini](https://github.com/marcoeilers/nagini)：Python 静态验证器，2610.19298 的 SWE-Proof 用的就是它，但它面向普通 Python，不面向 JAX 数值代码。
  - jax-lean：最贴近 JAX，但还早。

## 对 normix 的建议

**为什么是 normix。** 它本身就是数学库：GIG、GH、NIG、VG 等分布，加上指数族、EM 和 Bessel 工具。[代码树](https://github.com/xshi19/normix/tree/main/normix) 它承诺的东西大多能写成定理。测试已经有 `contract` 标记，也有 Bessel 导数的专门调查。但 `pyproject.toml` 里没有 hypothesis。[pyproject.toml](https://github.com/xshi19/normix/blob/main/pyproject.toml)，[tests](https://github.com/xshi19/normix/tree/main/tests)

**值得做的部分**

- EM 每一步对数似然单调不降（[fitting/em.py](https://github.com/xshi19/normix/blob/main/normix/fitting/em.py)）。数学上这是 Jensen 不等式加 M 步取最优，可以在 Lean 里证抽象版本。代码层面用性质测试检查，允许一个小容差。
- 指数族对偶：∇ψ(θ) = η，Legendre 变换后对偶回来恒等，Fisher 信息等于 ψ 的 Hessian（[exponential_family.py](https://github.com/xshi19/normix/blob/main/normix/exponential_family.py)、[fitting/eta.py](https://github.com/xshi19/normix/blob/main/normix/fitting/eta.py)）。θ→η→θ 往返可以直接做成性质测试。
- Bessel 恒等式：K_{-ν} = K_ν，递推 K_{ν+1} = K_{ν-1} + (2ν/x)K_ν，导数关系（[utils/bessel.py](https://github.com/xshi19/normix/blob/main/normix/utils/bessel.py)）。适合在 Lean 里证明，也适合用 hypothesis 在大参数范围里检查。
- 矩公式：GIG 的矩是 Bessel 比值，混合分布的均值和方差由条件矩合成（[mixtures/](https://github.com/xshi19/normix/tree/main/normix/mixtures)）。适合在纸面推导之后用 Lean 固定下来。

**不划算的部分**

- 浮点数值本身：log-Bessel 的数值稳定性、极端参数下的精度。这些靠参考数据和误差容差测试，不要去证。
- 优化和求解器的收敛性（[fitting/solvers.py](https://github.com/xshi19/normix/blob/main/normix/fitting/solvers.py)）：证明代价远大于收益。
- finance 模块：用回测和数值对照就够了。

**路线（投入从低到高）**

1. **hypothesis 性质测试**，大约几天。新增一个 `tests/test_properties.py`，覆盖 θ↔η 往返、EM 单调、Bessel 对称和递推、矩公式与采样矩一致。按 skill 的要求，每条性质都先在故意改坏的代码上跑失败一次。
2. **Lean 证数学恒等式**，大约几周。单独开一个 `normix-lean` 仓库，不碰主仓库。先证指数族对偶和 EM 单调这类抽象定理。Bessel 部分先查 Mathlib 已经覆盖到哪里。在 docstring 里写明每条定理对应哪个函数。
3. **jax-lean**，暂缓。原因有四条：没有 Bessel 原语；只支持 randint 采样；它针对 JAX 0.8.0，而 normix 锁定的是 0.9.1；仓库没有 license。等它补上这些再看。

**agent 能做什么**

第 1 步基本可以全交给 agent。第 2 步里 agent 能写 Lean 证明并反复编译，但定理的陈述必须你来审，防止它把规格写弱。也可以在 normix 上跑一次 `/create-formal-verification-skill`，让它生成一个项目内的验证 skill。按它自己的规则，Python 没有合适的验证器时会退回到生成输入测试，这和第 1 步正好吻合。

工具按你的规则用：Cursor 只用 Grok 4.7 xhigh，Fast 关掉，不用 Auto，不用第三方模型；Codex 用 GPT 6 Astra。

## 其他项目

- **xshi-math**：适合把关键引理挂上 Lean 证明链接，但这是锦上添花。
- **Project-HG / Project-CC**：写作项目，不适用。
- **portfolio 工具**：数值和数据为主，做性质测试就够了，不值得形式化。

## 学习路径

链接我都在今天打开确认过。

1. [Natural Number Game](https://adam.math.hhu.de/#/g/leanprover-community/nng4)：在浏览器里玩，从零学 Lean 的 tactic，大约一两个晚上。
2. [Theorem Proving in Lean 4](https://lean-lang.org/theorem_proving_in_lean4/)：官方教程，讲清楚命题、证明和 tactic。
3. [Mathematics in Lean](https://leanprover-community.github.io/mathematics_in_lean/)：用 Mathlib 写真实数学，和 normix 的恒等式最接近。
4. [Dafny 教程](https://dafny.org/latest/OnlineTutorial/guide)：前置条件、后置条件、不变式，看完就能读懂上面那段示意。
5. [hypothesis 文档](https://hypothesis.readthedocs.io/)：第 1 步路线马上就能用。
6. [Learn TLA+](https://learntla.com/)：想了解 @poteto 和 @bcherny 说的并发建模，再看这个。normix 暂时用不上。

建议顺序：5 → 1 → 3。先在 normix 上看到性质测试的效果，再学 Lean。

## 来源

- jax-lean：https://github.com/srush/jax-lean ；https://srush.github.io/jax-lean/ ；SUPPORTED_JAXPR.md、ARCHITECTURE.md（仓库内）
- @poteto pstack 帖：https://x.com/poteto/status/2111926622278521253
- pstack 插件：https://github.com/cursor/plugins/tree/main/pstack
- @bcherny 帖：https://x.com/bcherny/status/2102543349102338309
- arXiv 2610.19298：https://arxiv.org/abs/2610.19298
- arXiv 2610.22448：https://arxiv.org/abs/2610.22448
- DeepSeek-Prover-V2：https://arxiv.org/abs/2504.21801
- AlphaProof：https://www.nature.com/articles/s41586-025-09833-y
- Mathlib：https://github.com/leanprover-community/mathlib4
- Dafny：https://dafny.org ；Verus：https://github.com/verus-lang/verus ；F*：https://fstar-lang.org ；TLA+：https://lamport.azurewebsites.net/tla/tla.html
- Gappa：https://gappa.gitlabpages.inria.fr/ ；FPTaylor：https://github.com/soarlab/FPTaylor
- hypothesis：https://hypothesis.readthedocs.io/ ；CrossHair：https://github.com/pschanely/CrossHair ；Nagini：https://github.com/marcoeilers/nagini
- normix：https://github.com/xshi19/normix
