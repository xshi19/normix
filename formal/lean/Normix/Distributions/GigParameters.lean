module

public import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

@[expose] public section

noncomputable section

namespace Normix.Distributions

/-- Classical GIG parameters, with unrestricted order and two scale coordinates. -/
structure GigParams where
  p : ℝ
  a : ℝ
  b : ℝ

/-- GIG natural parameters in the statistic order $(\log y, 1/y, y)$. -/
structure GigNatural where
  theta0 : ℝ
  theta1 : ℝ
  theta2 : ℝ

/-- The strict classical GIG domain requires both scale coordinates to be positive. -/
def GigParams.Valid (q : GigParams) : Prop :=
  0 < q.a ∧ 0 < q.b

/-- The strict natural GIG domain requires both scale coefficients to be negative. -/
def GigNatural.Valid (t : GigNatural) : Prop :=
  t.theta1 < 0 ∧ t.theta2 < 0

/-- Convert classical GIG parameters to natural parameters. -/
def gigNatural (q : GigParams) : GigNatural :=
  ⟨q.p - 1, -q.b / 2, -q.a / 2⟩

/-- Invert GIG natural coordinates without applying the constructor clamps. -/
def gigFromNaturalUnclamped (t : GigNatural) : GigParams :=
  ⟨t.theta0 + 1, -2 * t.theta2, -2 * t.theta1⟩

/-- Invert GIG natural coordinates with the source constructor's nonnegative clamps. -/
def gigFromNatural (t : GigNatural) : GigParams :=
  ⟨t.theta0 + 1, max (-2 * t.theta2) 0, max (-2 * t.theta1) 0⟩

/-- EF-01: the inverse-scale coordinate round trips for every real value. -/
theorem gig_b_round_trip (b : ℝ) : -2 * (-b / 2) = b := by
  ring

/-- EF-02: the scale coordinate round trips for every real value. -/
theorem gig_a_round_trip (a : ℝ) : -2 * (-a / 2) = a := by
  ring

/-- EF-03: the order coordinate round trips for every real value. -/
theorem gig_p_round_trip (p : ℝ) : (p - 1) + 1 = p := by
  ring

/-- Unclamped GIG conversion recovers all classical parameters. -/
theorem gig_from_natural_unclamped_round_trip (q : GigParams) :
    gigFromNaturalUnclamped (gigNatural q) = q := by
  cases q
  simp only [gigNatural, gigFromNaturalUnclamped, gig_p_round_trip, gig_b_round_trip]

/-- Unclamped GIG conversion recovers all natural parameters. -/
theorem gig_natural_unclamped_round_trip (t : GigNatural) :
    gigNatural (gigFromNaturalUnclamped t) = t := by
  cases t
  simp [gigNatural, gigFromNaturalUnclamped]

/-- The clamped inverse-scale coordinate is always nonnegative. -/
theorem gig_from_natural_b_nonneg (t : GigNatural) :
    0 ≤ (gigFromNatural t).b :=
  le_max_right _ _

/-- The clamped scale coordinate is always nonnegative. -/
theorem gig_from_natural_a_nonneg (t : GigNatural) :
    0 ≤ (gigFromNatural t).a :=
  le_max_right _ _

/-- The inverse-scale clamp is inactive exactly when its natural coefficient is nonpositive. -/
theorem gig_b_clamp_inactive_iff (t : GigNatural) :
    (gigFromNatural t).b = (gigFromNaturalUnclamped t).b ↔ t.theta1 ≤ 0 := by
  change max (-2 * t.theta1) 0 = -2 * t.theta1 ↔ t.theta1 ≤ 0
  rw [max_eq_left_iff]
  constructor <;> intro h <;> linarith

/-- The scale clamp is inactive exactly when its natural coefficient is nonpositive. -/
theorem gig_a_clamp_inactive_iff (t : GigNatural) :
    (gigFromNatural t).a = (gigFromNaturalUnclamped t).a ↔ t.theta2 ≤ 0 := by
  change max (-2 * t.theta2) 0 = -2 * t.theta2 ↔ t.theta2 ≤ 0
  rw [max_eq_left_iff]
  constructor <;> intro h <;> linarith

/-- Both clamps are inactive throughout the strict natural GIG domain. -/
theorem gig_from_natural_eq_unclamped (t : GigNatural) (h : t.Valid) :
    gigFromNatural t = gigFromNaturalUnclamped t := by
  have ha := (gig_a_clamp_inactive_iff t).2 h.2.le
  have hb := (gig_b_clamp_inactive_iff t).2 h.1.le
  cases t
  simp_all [gigFromNatural, gigFromNaturalUnclamped]

/-- EF-06: conversion to GIG natural parameters preserves validity in both directions. -/
theorem gig_natural_valid_iff (q : GigParams) :
    q.Valid ↔ (gigNatural q).Valid := by
  dsimp [GigParams.Valid, GigNatural.Valid, gigNatural]
  constructor <;> rintro ⟨ha, hb⟩ <;> constructor <;> linarith

/-- EF-06: unclamped conversion from GIG natural parameters preserves validity. -/
theorem gig_from_natural_unclamped_valid_iff (t : GigNatural) :
    t.Valid ↔ (gigFromNaturalUnclamped t).Valid := by
  rw [gig_natural_valid_iff, gig_natural_unclamped_round_trip]

/-- EF-06: clamped conversion from GIG natural parameters preserves validity. -/
theorem gig_from_natural_valid_iff (t : GigNatural) :
    t.Valid ↔ (gigFromNatural t).Valid := by
  dsimp [GigNatural.Valid, GigParams.Valid, gigFromNatural]
  simp only [lt_max_iff, lt_self_iff_false, or_false]
  constructor <;> rintro ⟨ha, hb⟩ <;> constructor <;> linarith

/-- Valid classical GIG parameters round trip through the clamped conversion. -/
theorem gig_from_natural_round_trip (q : GigParams) (h : q.Valid) :
    gigFromNatural (gigNatural q) = q := by
  rw [gig_from_natural_eq_unclamped _ ((gig_natural_valid_iff q).1 h),
    gig_from_natural_unclamped_round_trip]

/-- Valid natural GIG parameters round trip through the clamped conversion. -/
theorem gig_natural_round_trip (t : GigNatural) (h : t.Valid) :
    gigNatural (gigFromNatural t) = t := by
  rw [gig_from_natural_eq_unclamped t h, gig_natural_unclamped_round_trip]

end Normix.Distributions
