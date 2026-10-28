module

public import Mathlib.Basic.Real.Basic

@[expose] public section

namespace Normix.Foundations

/-- GIG statistic coordinates in the order $(\log Y, 1/Y, Y)$. -/
structure GigStatistic where
  logY : ℝ
  invY : ℝ
  y : ℝ

/-- The Y-part of mixture statistics in the source order $(1/Y, Y, \log Y)$. -/
structure MixtureStatistic where
  invY : ℝ
  y : ℝ
  logY : ℝ

/-- Reorder GIG statistics into the Y-part of mixture statistics. -/
def gigToMixtureStatistic (s : GigStatistic) : MixtureStatistic :=
  ⟨s.invY, s.y, s.logY⟩

/-- Reorder the Y-part of mixture statistics into GIG statistics. -/
def mixtureToGigStatistic (s : MixtureStatistic) : GigStatistic :=
  ⟨s.logY, s.invY, s.y⟩

/-- EF-05: mixture coordinates place the inverse statistic first and the log statistic last. -/
theorem gig_statistic_coordinates (s : GigStatistic) :
    (gigToMixtureStatistic s).invY = s.invY ∧
      (gigToMixtureStatistic s).y = s.y ∧
      (gigToMixtureStatistic s).logY = s.logY :=
  ⟨rfl, rfl, rfl⟩

/-- Reordering GIG statistics to mixture order and back preserves every coordinate. -/
theorem gig_statistic_round_trip (s : GigStatistic) :
    mixtureToGigStatistic (gigToMixtureStatistic s) = s :=
  rfl

/-- Reordering mixture statistics to GIG order and back preserves every coordinate. -/
theorem mixture_statistic_round_trip (s : MixtureStatistic) :
    gigToMixtureStatistic (mixtureToGigStatistic s) = s :=
  rfl

end Normix.Foundations
