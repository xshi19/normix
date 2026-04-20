from normix.fitting.em import BatchEMFitter, IncrementalEMFitter, EMResult
from normix.fitting.eta import NormalMixtureEta, affine_combine
from normix.fitting.eta_rules import (
    EtaUpdateRule,
    AffineRule,
    IdentityUpdate,
    RobbinsMonroUpdate,
    SampleWeightedUpdate,
    EWMAUpdate,
    Shrinkage,
    AffineUpdate,
)
from normix.fitting.shrinkage_targets import (
    eta0_from_model,
    eta0_isotropic,
    eta0_diagonal,
    eta0_with_sigma,
)

__all__ = [
    "BatchEMFitter",
    "IncrementalEMFitter",
    "EMResult",
    "NormalMixtureEta",
    "affine_combine",
    "EtaUpdateRule",
    "AffineRule",
    "IdentityUpdate",
    "RobbinsMonroUpdate",
    "SampleWeightedUpdate",
    "EWMAUpdate",
    "Shrinkage",
    "AffineUpdate",
    "eta0_from_model",
    "eta0_isotropic",
    "eta0_diagonal",
    "eta0_with_sigma",
]
