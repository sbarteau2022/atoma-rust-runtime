//! `atoma_immune`: Layer 3 Immune System, DHNG II Optical Refraction, DHNG III 12-Around-1 Silos,
//! Non-Abelian Hyperbolic Holonomy (`holonomy.rs`), Ephemeral Agent Map Deposits, Model-Agnostic
//! Centralized Memory, and First-Principles Superpowers Integration.

#[path = "../../../core_lm/src/geometry/mod.rs"]
pub mod geometry;

#[path = "../../../core_lm/src/cache/geometric.rs"]
pub mod geometric_mount;

pub mod cache {
    pub use super::geometric_mount as geometric;
}

#[path = "../../../core_lm/src/lock.rs"]
pub mod lock;

#[path = "../../../core_lm/src/metal/mod.rs"]
pub mod metal;

#[path = "../../../core_lm/src/ops.rs"]
pub mod ops;

pub mod atlas13;
pub mod compliance;
pub mod first_principles;
pub mod holonomy;
pub mod refraction;
pub mod silo;

pub use atlas13::{
    Atlas13, DualityAxis, EpistemicMark, FrameGeometry, INV_SQRT5_Q32, MEASURED_AXIS_COS_Q32,
    R_SPH_Q32, RoleId, WellDescriptor,
};
pub use compliance::{
    ComplianceAuditEntry, CompliancePermit, ComplianceViolation, ImmuneSystem,
    OverlapPolicyMatrix, PREREGISTERED_KILL_FINGERPRINT, PreregisteredEvaluation,
};
pub use first_principles::{
    ATOMIC_IR_MAX_VERIFY_TOKENS, AtomClass, AtomicIrAlphabet, ComplianceMerkleTree,
    FIRST_PRINCIPLE_STATUTES, FirstPrincipleInvariant, FirstPrincipleStatutoryRule,
    FirstPrinciplesComplianceEngine, FirstPrinciplesComplianceReport, MetaCognitiveRunbookController,
    MetaCognitiveVerdictQ32, MetaControlDecision, OpticalRiskEngine, OpticalRiskReport,
    QuotientManifoldProof, QuotientVerificationReport, RegulatoryFramework, RiskTier, RoleSignalQ32,
    RunbookTelemetryQ32, SecurityOntologyEntry, SphereHomeostasisSpec, SurgicalErasureReceipt,
    SwarmKnowledgeAndHomeostasis, digest_ct_eq, merkle_hash_leaf, merkle_hash_node,
    predict_token_envelope,
};
pub use holonomy::{
    HolonomyLoopReceipt, HolonomyVerdict, HyperbolicHolonomyEngine, IDENTITY_FRAME_Q32,
};
pub use refraction::{
    FrozenShard, OpticalLens, ProjectedTangent, ProjectionLossReport, RefractedPoint,
    RefractionError, RefractionLossReport, Sha256Digest, TangentProjector, TeacherTapRecord,
    rad_q32_to_pi_unit,
};
pub use silo::{
    CarveLossReport, CarvedSwarmMap, CarvedWeightView, DepositCompilationReceipt,
    DepositedAtomRecord, EphemeralMapDeposit, FrozenBaseWeights, GlobalAtom, ModelHotSwapReceipt,
    SiloAtom, SiloView,
};
