//! `atoma_immune`: Layer 3 Immune System, DHNG II Optical Refraction, DHNG III 12-Around-1 Silos,
//! Non-Abelian Hyperbolic Holonomy (`holonomy.rs`), Ephemeral Agent Map Deposits, Model-Agnostic
//! Centralized Memory, Geodesic MLIR Engine (`mlir_native.rs`), Open-Weights Knowledge & Reasoning
//! Distillation Engine (`distill_engine.rs`), First-Principles Superpowers Integration
//! (`first_principles.rs`), and Active Inference / Superposition Holding (`superposition.rs`).

#[path = "../../../core_lm/src/geometry/mod.rs"]
pub mod geometry;

#[path = "../../../core_lm/src/cache/geometric.rs"]
pub mod geometric_mount;

pub mod cache {
    pub use super::geometric_mount as geometric;
}

#[path = "../../../core_lm/src/lock.rs"]
pub mod lock;

// Apple Silicon Metal / Accelerate compute backend: only compiled when the
// `metal` feature is enabled (macOS-only dependency). Portable consumers build
// with default features and never see this module.
#[cfg(feature = "metal")]
#[path = "../../../core_lm/src/metal/mod.rs"]
pub mod metal;

/// Portable fallback for row dequantization when the `metal` feature is off.
#[cfg(not(feature = "metal"))]
pub mod portable_dequant;

/// Single-row zero-copy dequantization, dispatched to the tuned Metal/Accelerate
/// backend (`crate::metal`) when the `metal` feature is enabled, or to the
/// portable, bounds-checked implementation otherwise. Both paths implement the
/// identical affine group contract `(q * s_g + b_g) * row_out_scale`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dequantize_row_zero_copy(
    weight_data: &[u8],
    weight_dtype: &str,
    scales_data: &[u8],
    scales_dtype: &str,
    biases_data: &[u8],
    biases_dtype: &str,
    row_out_scale: f32,
    bits: u32,
    group_size: usize,
    row: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    #[cfg(feature = "metal")]
    {
        metal::dequantize_row_zero_copy(
            weight_data,
            weight_dtype,
            scales_data,
            scales_dtype,
            biases_data,
            biases_dtype,
            row_out_scale,
            bits,
            group_size,
            row,
            cols,
        )
    }
    #[cfg(not(feature = "metal"))]
    {
        portable_dequant::dequantize_row_zero_copy(
            weight_data,
            weight_dtype,
            scales_data,
            scales_dtype,
            biases_data,
            biases_dtype,
            row_out_scale,
            bits,
            group_size,
            row,
            cols,
        )
    }
}

#[path = "../../../core_lm/src/ops.rs"]
pub mod ops;

pub mod atlas13;
pub mod compliance;
pub mod continuous_active_mind;
pub mod distill_engine;
pub mod first_principles;
pub mod holonomy;
pub mod map_native;
pub mod mlir_native;
pub mod native_reader;
pub mod refraction;
pub mod silo;
pub mod skill_evolution;
pub mod specialist_carving;
pub mod superposition;

pub use specialist_carving::{
    BasinActivationMetric, CANONICAL_DOMAIN_PRESETS, CarvedSpecialistProfile, DomainProbePreset,
    LitAtomPoint, LitTrajectoryArc, project_qwen_token_sketch, volumetric_cone_ball_coord,
};

pub use skill_evolution::{
    AutonomousSkillCrystal, LiveVerificationReceipt, PolyglotCodeTrajectorySpec, PolyglotLanguage,
    SkillEvolutionEngine, SuperpowersAndPolyglotHarvestReceipt, VerifiedCodeSynthesisReceipt,
};

pub use continuous_active_mind::{
    ATOMA_LEX_MAGIC, ATOMA_WAL_MAGIC, ATOMA_WIRE_MAGIC, AgenticTaskCompletionReceipt,
    BoundaryLexicalTransducer, CrystallizationReceipt, DEFAULT_ACTIVE_MIND_VAULT,
    DEFAULT_BOUNDARY_LEXICON_PATH, DreamConsolidationReceipt, EpisodeWorkingTrace,
    GeodesicActiveTransformer, GeodesicCursor, NativeWireRequest, NativeWireResponse,
    SpanFoldingReceipt, SubAgentTaskReceipt, fold_high_certainty_trajectory_spans,
};

pub use map_native::{
    COUPLED_OPERATOR_MAGIC, CoupledOperatorAtomPayload, DEFAULT_MGS_ORTHO_EPSILON,
    InstrumentedLayerCapture, JacobianTangentSlot, MAX_ACCRETION_ENTROPY_BITS, MAX_JACOBIAN_RANK,
    MapNativeInferenceEngine, MapNativeLayerReceipt, MapSubstitutionMode, RECEPTOR_SKETCH_DIM,
    Rank8AccretionReceipt, VectorComparisonMetrics, align_cross_model_effector_dim,
    compute_receptor_sketch, generate_icosahedral_perturbation_ray,
};

pub use atlas13::{
    Atlas13, DualityAxis, EpistemicMark, FrameGeometry, INV_SQRT5_Q32, MEASURED_AXIS_COS_Q32,
    R_SPH_Q32, RoleId, WellDescriptor,
};
pub use compliance::{
    ComplianceAuditEntry, CompliancePermit, ComplianceViolation, ImmuneSystem, OverlapPolicyMatrix,
    PREREGISTERED_KILL_FINGERPRINT, PreregisteredEvaluation,
};
pub use distill_engine::{
    DistilledReasoningArc, FullModelDistillationReceipt, LiveNoveltyDistillationReceipt,
    OpenWeightsDistillationEngine, QuantizedDistillationReceipt, QuantizedLayerTapDescriptor,
    ReasoningChainDistillationReceipt, WeightDistillationReceipt,
};
pub use first_principles::{
    ATOMIC_IR_MAX_VERIFY_TOKENS, AtomClass, AtomicIrAlphabet, ComplianceMerkleTree,
    FIRST_PRINCIPLE_STATUTES, FirstPrincipleInvariant, FirstPrincipleStatutoryRule,
    FirstPrinciplesComplianceEngine, FirstPrinciplesComplianceReport,
    MetaCognitiveRunbookController, MetaCognitiveVerdictQ32, MetaControlDecision,
    OpticalRiskEngine, OpticalRiskReport, QuotientManifoldProof, QuotientVerificationReport,
    RegulatoryFramework, RiskTier, RoleSignalQ32, RunbookTelemetryQ32, SecurityOntologyEntry,
    SphereHomeostasisSpec, SurgicalErasureReceipt, SwarmKnowledgeAndHomeostasis, digest_ct_eq,
    merkle_hash_leaf, merkle_hash_node, predict_token_envelope,
};
pub use holonomy::{
    HolonomyLoopReceipt, HolonomyVerdict, HyperbolicHolonomyEngine, IDENTITY_FRAME_Q32,
};
pub use mlir_native::{
    ATOMA_IR_BINARY_MAGIC, GeodesicDecodeReceipt, GeodesicMlirDecoder, MLIR_SSA_WIRE_MAGIC,
    MlirAtomCatalog, MlirDialect, MlirOpDescriptor, MlirOpId, MlirSsaBlock, MlirSsaError,
    MlirSsaValidationReceipt, MlirTransductionLossReport, MlirTypeSig, NativeMlirInstruction,
};
pub use native_reader::{
    NativeContinuation, NativeRead, NativeReader, NativeStop, TrajectoryWrite,
    frame_frobenius_dist_sq_q32, normalized_frame_alignment_q32, prefix_chain_id,
};
pub use refraction::{
    FrozenShard, OpticalLens, ProjectedTangent, ProjectionLossReport, RefractedPoint,
    RefractionError, RefractionLossReport, Sha256Digest, TangentProjector, TeacherTapRecord,
    rad_q32_to_pi_unit,
};
pub use silo::{
    AdaptedKvRecord, AdapterLossReport, CarveLossReport, CarvedSwarmMap, CarvedWeightView,
    DepositCompilationReceipt, DepositedAtomRecord, EphemeralMapDeposit, FrozenBaseWeights,
    GlobalAtom, ModelAdapterLens, ModelHotSwapReceipt, SiloAtom, SiloView, TaskCarvedView,
};
pub use superposition::{
    ActiveInferenceConfig, ActiveInferenceContinuation, ActiveInferenceStepReceipt,
    ActiveInferenceStop, SuperpositionBranch, SuperpositionBranchSource,
    SuperpositionHoldingEngine, SuperpositionIterationTrace, SuperpositionResolution,
    SuperpositionState, SuperpositionTrigger, VariationalFreeEnergyScore,
};
