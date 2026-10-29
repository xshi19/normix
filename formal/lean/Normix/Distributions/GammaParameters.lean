module

public import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

@[expose] public section

noncomputable section

namespace Normix.Distributions

/-- Classical shape and rate parameters of the Gamma family. -/
structure GammaParams where
  alpha : ℝ
  beta : ℝ

/-- Gamma natural parameters in the statistic order $(\log x, x)$. -/
structure GammaNatural where
  theta1 : ℝ
  theta2 : ℝ

/-- The classical Gamma domain requires positive shape and rate. -/
def GammaParams.Valid (p : GammaParams) : Prop :=
  0 < p.alpha ∧ 0 < p.beta

/-- The natural Gamma domain corresponds to positive shape and rate. -/
def GammaNatural.Valid (t : GammaNatural) : Prop :=
  -1 < t.theta1 ∧ t.theta2 < 0

/-- Convert classical Gamma parameters to natural parameters. -/
def gammaNatural (p : GammaParams) : GammaNatural :=
  ⟨p.alpha - 1, -p.beta⟩

/-- Convert natural Gamma parameters to classical parameters. -/
def gammaFromNatural (t : GammaNatural) : GammaParams :=
  ⟨t.theta1 + 1, -t.theta2⟩

/-- Classical Gamma parameters round trip for all real coordinates. -/
theorem gamma_from_natural_round_trip (p : GammaParams) :
    gammaFromNatural (gammaNatural p) = p := by
  cases p
  simp [gammaNatural, gammaFromNatural]

/-- Natural Gamma parameters round trip for all real coordinates. -/
theorem gamma_natural_round_trip (t : GammaNatural) :
    gammaNatural (gammaFromNatural t) = t := by
  cases t
  simp [gammaNatural, gammaFromNatural]

/-- Conversion to Gamma natural parameters preserves validity in both directions. -/
theorem gamma_natural_valid_iff (p : GammaParams) :
    p.Valid ↔ (gammaNatural p).Valid := by
  dsimp [GammaParams.Valid, GammaNatural.Valid, gammaNatural]
  constructor <;> rintro ⟨ha, hb⟩ <;> constructor <;> linarith

/-- Conversion from Gamma natural parameters preserves validity in both directions. -/
theorem gamma_from_natural_valid_iff (t : GammaNatural) :
    t.Valid ↔ (gammaFromNatural t).Valid := by
  rw [gamma_natural_valid_iff, gamma_natural_round_trip]

/-- The Gamma mean in shape and rate coordinates. -/
def gammaMean (p : GammaParams) : ℝ :=
  p.alpha / p.beta

/-- The Gamma variance in shape and rate coordinates. -/
def gammaVar (p : GammaParams) : ℝ :=
  p.alpha / p.beta ^ 2

/-- Valid Gamma parameters have positive variance. -/
theorem gamma_var_pos (p : GammaParams) (h : p.Valid) :
    0 < gammaVar p :=
  div_pos h.1 (pow_pos h.2 2)

/-- Valid Gamma parameters have positive mean. -/
theorem gamma_mean_pos (p : GammaParams) (h : p.Valid) :
    0 < gammaMean p :=
  div_pos h.1 h.2

/-- The Gamma variance is its mean divided by its nonzero rate. -/
theorem gamma_var_eq_mean_div_beta (p : GammaParams) (h : p.beta ≠ 0) :
    gammaVar p = gammaMean p / p.beta := by
  dsimp [gammaVar, gammaMean]
  field_simp

end Normix.Distributions
