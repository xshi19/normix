module

public import Mathlib.Algebra.BigOperators.Field
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Tactic.Ring

@[expose] public section

noncomputable section

namespace Normix.Numerics

open scoped BigOperators

variable {n d : ℕ}

/-- Normalize finite log weights by subtracting the log of the sum of their exponentials. -/
def normalizedWeights (logW : Fin n → ℝ) : Fin n → ℝ :=
  fun i => Real.exp (logW i - Real.log (∑ j, Real.exp (logW j)))

private theorem exp_sum_pos [NeZero n] (logW : Fin n → ℝ) :
    0 < ∑ i, Real.exp (logW i) :=
  Finset.sum_pos (fun i _ => Real.exp_pos (logW i)) Finset.univ_nonempty

/-- EF-07: every normalized exponential weight is positive. -/
theorem normalized_weights_pos (logW : Fin n → ℝ) (i : Fin n) :
    0 < normalizedWeights logW i :=
  Real.exp_pos _

/-- EF-07: subtracting the log normalizer divides each exponential by their sum. -/
theorem normalized_weights_eq_div [NeZero n] (logW : Fin n → ℝ) (i : Fin n) :
    normalizedWeights logW i = Real.exp (logW i) / ∑ j, Real.exp (logW j) := by
  rw [normalizedWeights, Real.exp_sub, Real.exp_log (exp_sum_pos logW)]

/-- EF-07: normalized exponential weights on a nonempty finite set sum to one. -/
theorem normalized_weights_sum [NeZero n] (logW : Fin n → ℝ) :
    ∑ i, normalizedWeights logW i = 1 := by
  simp_rw [normalized_weights_eq_div]
  rw [← Finset.sum_div, div_self (ne_of_gt (exp_sum_pos logW))]

/-- The weighted mean of a finite family of statistic vectors. -/
def weightedMean (p : Fin n → ℝ) (s : Fin n → Fin d → ℝ) : Fin d → ℝ :=
  fun a => ∑ i, p i * s i a

/-- The centered covariance entries used by the finite weighted moment calculation. -/
def centeredCov (p : Fin n → ℝ) (s : Fin n → Fin d → ℝ) :
    Matrix (Fin d) (Fin d) ℝ :=
  fun a b => ∑ i, p i * (s i a - weightedMean p s a) * (s i b - weightedMean p s b)

/-- EF-09: the centered quadratic form is a weighted sum of squared projections. -/
theorem centered_gram_identity (p : Fin n → ℝ) (s : Fin n → Fin d → ℝ)
    (v : Fin d → ℝ) :
    (∑ a, ∑ b, v a * centeredCov p s a b * v b) =
      ∑ i, p i * (∑ j, v j * (s i j - weightedMean p s j)) ^ 2 := by
  unfold centeredCov
  simp_rw [Finset.mul_sum, Finset.sum_mul]
  calc
    (∑ a, ∑ b, ∑ i,
        v a * (p i * (s i a - weightedMean p s a) * (s i b - weightedMean p s b)) *
          v b) =
        ∑ i, ∑ a, ∑ b,
          v a * (p i * (s i a - weightedMean p s a) * (s i b - weightedMean p s b)) *
            v b := by
      conv_lhs =>
        arg 2
        ext a
        rw [Finset.sum_comm]
      rw [Finset.sum_comm]
    _ = ∑ i, p i * (∑ j, v j * (s i j - weightedMean p s j)) ^ 2 := by
      apply Finset.sum_congr rfl
      intro i _
      rw [pow_two, Finset.sum_mul_sum, Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro a _
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro b _
      ring

/-- EF-09: nonnegative weights make every centered quadratic form nonnegative. -/
theorem centered_gram_nonneg (p : Fin n → ℝ) (s : Fin n → Fin d → ℝ)
    (hp : ∀ i, 0 ≤ p i) (v : Fin d → ℝ) :
    0 ≤ ∑ a, ∑ b, v a * centeredCov p s a b * v b := by
  rw [centered_gram_identity]
  exact Finset.sum_nonneg fun i _ => mul_nonneg (hp i) (sq_nonneg _)

/-- Centered covariance is symmetric for arbitrary real weights. -/
theorem centered_cov_symm (p : Fin n → ℝ) (s : Fin n → Fin d → ℝ) :
    (centeredCov p s).IsSymm := by
  apply Matrix.IsSymm.ext
  intro a b
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- EF-09: centered covariance is positive semidefinite for nonnegative weights. -/
theorem centered_cov_pos_semidef (p : Fin n → ℝ) (s : Fin n → Fin d → ℝ)
    (hp : ∀ i, 0 ≤ p i) : (centeredCov p s).PosSemidef := by
  apply Matrix.PosSemidef.of_dotProduct_mulVec_nonneg
  · exact Matrix.isHermitian_iff_isSymm.mpr (centered_cov_symm p s)
  · intro v
    simpa [dotProduct, Matrix.mulVec, Finset.mul_sum, mul_assoc] using
      centered_gram_nonneg p s hp v

/-- Unit total weight makes centered covariance equal the raw second moment minus the mean product. -/
theorem centered_cov_eq_raw_sub (p : Fin n → ℝ) (s : Fin n → Fin d → ℝ)
    (hp : ∑ i, p i = 1) (a b : Fin d) :
    centeredCov p s a b =
      (∑ i, p i * s i a * s i b) - weightedMean p s a * weightedMean p s b := by
  unfold centeredCov
  calc
    (∑ i, p i * (s i a - weightedMean p s a) * (s i b - weightedMean p s b)) =
        ∑ i, (p i * s i a * s i b - p i * s i a * weightedMean p s b -
          p i * s i b * weightedMean p s a + p i * (weightedMean p s a *
            weightedMean p s b)) := by
      apply Finset.sum_congr rfl
      intro i _
      ring
    _ = _ := by
      simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.sum_mul,
        hp, one_mul]
      unfold weightedMean
      ring

/-- The one-dimensional log partition with fixed positive base weights and statistic locations. -/
def fixedGeometryLogPartition (w x : Fin n → ℝ) (theta : ℝ) : ℝ :=
  Real.log (∑ i, w i * Real.exp (theta * x i))

/-- Normalized weights of the one-dimensional family with fixed geometry. -/
def fixedGeometryWeights (w x : Fin n → ℝ) (theta : ℝ) : Fin n → ℝ :=
  fun i => w i * Real.exp (theta * x i) / ∑ j, w j * Real.exp (theta * x j)

/-- EF-08: the fixed-geometry log-partition derivative is the normalized statistic mean. -/
theorem fixed_geometry_log_partition_has_deriv_at [NeZero n] (w x : Fin n → ℝ)
    (hw : ∀ i, 0 < w i) (theta : ℝ) :
    HasDerivAt (fixedGeometryLogPartition w x)
      (∑ i, fixedGeometryWeights w x theta i * x i) theta := by
  have hsum : HasDerivAt (fun t : ℝ => ∑ i, w i * Real.exp (t * x i))
      (∑ i, w i * (Real.exp (theta * x i) * x i)) theta := by
    apply HasDerivAt.fun_sum
    intro i _
    exact (hasDerivAt_mul_const (x i)).exp.const_mul (w i)
  have hpos : 0 < ∑ i, w i * Real.exp (theta * x i) :=
    Finset.sum_pos (fun i _ => mul_pos (hw i) (Real.exp_pos _)) Finset.univ_nonempty
  have hmean : (∑ i, w i * (Real.exp (theta * x i) * x i)) /
        (∑ j, w j * Real.exp (theta * x j)) =
      ∑ i, fixedGeometryWeights w x theta i * x i := by
    rw [Finset.sum_div]
    apply Finset.sum_congr rfl
    intro i _
    unfold fixedGeometryWeights
    ring
  rw [← hmean]
  exact hsum.log (ne_of_gt hpos)

end Normix.Numerics
