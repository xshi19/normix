from normix.fitting.em import BatchEMFitter, IncrementalEMFitter, EMResult
from normix.fitting.eta import NormalMixtureEta, affine_combine
from normix.fitting.eta_rules import (
    EtaUpdateRule,
    AffineRule,
    IdentityUpdate,
    RobbinsMonroUpdate,
    SampleWeightedUpdate,
    EWMAUpdate,
    ShrinkageUpdate,
    AffineUpdate,
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
    "ShrinkageUpdate",
    "AffineUpdate",
]
