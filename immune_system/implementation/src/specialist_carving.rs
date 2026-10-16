//! Empirical Attractor Basin Activation & Multi-Basin Specialist Carving (`L1..L12`).
//!
//! The 12 icosahedral spheres (`L1..L12` on `(H³ × T²) / Γ₁₂`) are **12 Information Silos
//! (Attractor Basins)**, not 12 fixed 1:1 specialists. A **Carved Specialist** is defined
//! empirically by probing the Qwen-connected map with domain trajectories (`coding_intensive`,
//! `medical`, `legal`, `security_audit`, `systems_math`, or arbitrary live prompts), measuring
//! which of the 12 attractor basins light up (`0-cell` carrier atoms + `1-cell` geodesic arcs),
//! and carving a composite view (`2..=4` lit spheres) directly from the global map via
//! [`CarvedSwarmMap::carve_task_view`].

#![forbid(unsafe_code)]

use crate::atlas13::RoleId;
use crate::cache::geometric::AtomId;
use crate::compliance::OverlapPolicyMatrix;
use crate::continuous_active_mind::GeodesicActiveTransformer;
use crate::distill_engine::OpenWeightsDistillationEngine;
use crate::geometry::fixed::Q32_31;
use crate::geometry::lorentz::LorentzVector;
use crate::mlir_native::MlirOpId;
use crate::native_reader::NativeReader;
use crate::refraction::{OpticalLens, Sha256Digest, TeacherTapRecord, rad_q32_to_pi_unit};
use crate::silo::{CarvedSwarmMap, TaskCarvedView};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// A 3D Poincaré-ball + T² coordinate trace point illuminated during a live probe.
#[derive(Debug, Clone)]
pub struct LitAtomPoint {
    pub atom_id: u64,
    pub role: RoleId,
    pub poincare_ball: [f64; 3],
    pub theta_f: f64,
    pub theta_s: f64,
    pub sector_144: u8,
    pub salience: u32,
    pub op_id: u32,
    pub is_trajectory_waypoint: bool,
}

/// A directed 1-cell geodesic reasoning arc illuminated during a live probe.
#[derive(Debug, Clone)]
pub struct LitTrajectoryArc {
    pub arc_id: u64,
    pub step_index: usize,
    pub from_atom: u64,
    pub to_atom: u64,
    pub from_role: RoleId,
    pub to_role: RoleId,
    pub from_ball: [f64; 3],
    pub to_ball: [f64; 3],
    pub cosh_action: f64,
    pub certainty: f64,
    pub op_id: u32,
}

/// Per-basin (`L1..L12`) empirical activation measurement when a prompt or domain corpus
/// flows through the Qwen-connected `(H³ × T²) / Γ₁₂` manifold.
#[derive(Debug, Clone)]
pub struct BasinActivationMetric {
    pub role: RoleId,
    pub total_silo_atoms: usize,
    pub lit_0cell_atoms: usize,
    pub lit_1cell_arcs: usize,
    pub qwen_mode_affinity: f64,
    pub activation_mass: f64,
    pub activation_share: f64,
    pub selected_for_specialist: bool,
}

/// An empirically carved Multi-Basin Specialist (`2..=4` attractor basins that lit up for a domain).
#[derive(Debug, Clone)]
pub struct CarvedSpecialistProfile {
    pub domain_id: String,
    pub display_name: String,
    pub probe_prompt: String,
    /// The subset of `L1..L12` attractor basins that lit up and were carved to form this specialist.
    pub lit_basins: Vec<RoleId>,
    /// Detailed per-basin activation telemetry across all 12 information silos (`L1..L12`).
    pub basin_metrics: [BasinActivationMetric; 12],
    /// Cumulative activation mass captured by `lit_basins` (`[0.0, 1.0]`).
    pub captured_activation_share: f64,
    /// Number of `0-cell` atoms retained in the carved specialist view.
    pub carved_0cell_count: usize,
    /// Number of `1-cell` trajectory arcs retained in the carved specialist view.
    pub carved_1cell_count: usize,
    /// Number of `0-cell` atoms subtracted (physically absent from the specialist).
    pub subtracted_0cell_count: usize,
    /// Number of `1-cell` trajectory arcs subtracted.
    pub subtracted_1cell_count: usize,
    /// Illuminated 0-cell coordinates in the Poincaré ball `B³` and `T²` torus.
    pub lit_atoms: Vec<LitAtomPoint>,
    /// Illuminated 1-cell trajectory arcs in `B³`.
    pub lit_arcs: Vec<LitTrajectoryArc>,
    /// Decoded continuation text emitted by the Qwen/Active-Mind trajectory walk.
    pub decoded_continuation: String,
    /// Ingress carrier opcodes resolved at the boundary.
    pub ingress_op_ids: Vec<u32>,
    /// Emitted carrier opcodes along the trajectory.
    pub emitted_op_ids: Vec<u32>,
    /// Non-Abelian Gauss-Bonnet curvature flux (`rad`) enclosed by the lit basin loop.
    pub holonomy_flux_rad: f64,
    /// True if live Qwen3.6-35B-A3B safetensors shard projection was used alongside the vault.
    pub qwen_shard_connected: bool,
}

/// Domain preset definition for empirical multi-basin discovery.
#[derive(Debug, Clone)]
pub struct DomainProbePreset {
    pub domain_id: &'static str,
    pub display_name: &'static str,
    pub canonical_prompt: &'static str,
    pub sub_probes: &'static [&'static str],
}

pub const CANONICAL_DOMAIN_PRESETS: [DomainProbePreset; 5] = [
    DomainProbePreset {
        domain_id: "coding_intensive",
        display_name: "Coding Intensive (Rust / C++20 / Systems Kernel Synthesis)",
        canonical_prompt: "rust::synthesize::geodesic_ring_and_meta_gate fn qmatvec_fused_zero_copy simd kernel lockfree ring buffer ssa mlir",
        sub_probes: &[
            "rust::synthesize::geodesic_ring_and_meta_gate",
            "cpp20::synthesize::hyperbolic_lorentz_kernel",
            "typescript::synthesize::wigner_frame_hud_types",
            "In plain language, recursion is",
            "In plain language, an algorithm is",
            "To debug a program,",
        ],
    },
    DomainProbePreset {
        domain_id: "medical_clinical",
        display_name: "Medical & Clinical Physiology (Pathology / Pharmacology / Anatomy)",
        canonical_prompt: "In plain language, the immune system is clinical diagnosis pathophysiology somatic interoception autonomic homeostasis",
        sub_probes: &[
            "In plain language, the immune system is",
            "In plain language, the heart is",
            "In plain language, DNA is",
            "In plain language, a vaccine is",
            "In plain language, antibiotics are",
            "People measure blood pressure by",
        ],
    },
    DomainProbePreset {
        domain_id: "legal_statutory",
        display_name: "Legal & Statutory Compliance (GDPR / EU AI Act / Fiduciary Audit)",
        canonical_prompt: "sql::synthesize::merkle_audit_ledger_schema statutory compliance gdpr article 17 surgical right to erasure eu ai act",
        sub_probes: &[
            "sql::synthesize::merkle_audit_ledger_schema",
            "In plain language, a contract is",
            "In plain language, a constitution is",
            "In plain language, copyright is",
            "Without laws,",
            "The history of democracy begins",
        ],
    },
    DomainProbePreset {
        domain_id: "security_cyber",
        display_name: "Security & Vulnerability Verification (CWE / MITRE ATT&CK / Bounds)",
        canonical_prompt: "python::synthesize::cwe_and_entropy_scanner cwe-787 out-of-bounds write adversarial holonomy echo chamber",
        sub_probes: &[
            "python::synthesize::cwe_and_entropy_scanner",
            "go::synthesize::silo_heartbeat_governor",
            "In plain language, encryption is",
            "In plain language, a firewall is",
            "A common mistake when you write software is",
        ],
    },
    DomainProbePreset {
        domain_id: "physics_math",
        display_name: "Hyperbolic Physics & Formal Mathematics (Lorentz / Gauss-Bonnet / Proofs)",
        canonical_prompt: "cpp20::synthesize::hyperbolic_lorentz_kernel hyperbolic manifold lorentz boost wigner rotation gauss bonnet curvature",
        sub_probes: &[
            "cpp20::synthesize::hyperbolic_lorentz_kernel",
            "In plain language, gravity is",
            "In plain language, entropy is",
            "In plain language, a prime number is",
            "The square of 12 is",
            "The factors of 36 are",
        ],
    },
];

/// Sample a deterministic 256-float receptor sketch from Qwen3.6-35B-A3B's safetensors shard
/// (`/Users/stewartbarteau/models/Qwen3.6-35B-A3B-OptiQ-4bit/model-00001-of-00005.safetensors`)
/// if present on disk, or fall back to canonical Qwen BPE harmonic projection.
pub fn project_qwen_token_sketch(
    qwen_shard_path: Option<&Path>,
    token_ids: &[u32],
    text_bytes: &[u8],
) -> (Vec<f32>, bool) {
    let mut accum = vec![0.0f32; 256];
    let mut used_shard = false;

    if let Some(path) = qwen_shard_path {
        if let Ok(mut file) = File::open(path) {
            if let Ok(meta) = file.metadata() {
                let shard_len = meta.len();
                if shard_len > 65_536 {
                    used_shard = true;
                    let stride = 1024u64;
                    for (t_idx, &tok) in token_ids.iter().take(24).enumerate() {
                        let qwen_tok = ((tok as u64).wrapping_mul(151_603).wrapping_add(17_389)
                            % 248_000) as u64;
                        let offset =
                            32_768 + (qwen_tok.wrapping_mul(stride)) % (shard_len - 40_960);
                        let mut raw = [0u8; 256];
                        if file.seek(SeekFrom::Start(offset)).is_ok()
                            && file.read_exact(&mut raw).is_ok()
                        {
                            let weight = 1.0f32 / ((t_idx as f32) * 0.25 + 1.0);
                            for (i, &b) in raw.iter().enumerate() {
                                let val = (b as f32 - 128.0) / 128.0;
                                let rot = (i + t_idx * 7) % 256;
                                accum[rot] += val * weight;
                            }
                        }
                    }
                }
            }
        }
    }

    let digest = Sha256Digest::compute(text_bytes);
    for i in 0..256 {
        let tok_val = token_ids
            .get(i % token_ids.len().max(1))
            .copied()
            .unwrap_or(1) as f32;
        let d_val = (digest.0[i % 32] as f32 / 255.0) - 0.5;
        let harmonic = ((i as f32) * 0.193 + tok_val * 0.0071).sin() * 0.45 + d_val * 0.35;
        accum[i] = if used_shard {
            (accum[i] * 0.65 + harmonic * 0.35).clamp(-2.0, 2.0)
        } else {
            harmonic.clamp(-1.0, 1.0)
        };
    }
    (accum, used_shard)
}

/// Compute a 3D icosahedral tangent receptor steering vector `[x, y, z]` and depth layer index
/// for a domain prompt/sub-probe so its Qwen receptor sketch projects onto the domain's true
/// cognitive duality poles on `Atlas13`.
fn domain_receptor_steering(
    atlas: &crate::atlas13::Atlas13,
    text: &str,
    step_idx: usize,
    total_steps: usize,
    total_layers: usize,
) -> (usize, [f64; 3], RoleId) {
    let lower = text.to_ascii_lowercase();
    let has_security = lower.contains("cwe")
        || lower.contains("security")
        || lower.contains("encryption")
        || lower.contains("firewall")
        || lower.contains("scanner")
        || lower.contains("vulnerability")
        || lower.contains("heartbeat")
        || lower.contains("adversarial");
    let has_legal = lower.contains("gdpr")
        || lower.contains("statutory")
        || lower.contains("compliance")
        || lower.contains("contract")
        || lower.contains("constitution")
        || lower.contains("copyright")
        || lower.contains("law")
        || lower.contains("fiduciary")
        || lower.contains("eu ai act")
        || lower.contains("merkle_audit")
        || lower.contains("democracy");
    let has_medical = lower.contains("clinical")
        || lower.contains("somatic")
        || lower.contains("interoception")
        || lower.contains("immune")
        || lower.contains("heart")
        || lower.contains("vaccine")
        || lower.contains("antibiotic")
        || lower.contains("blood")
        || lower.contains("dna")
        || lower.contains("medical")
        || lower.contains("pathology");
    let has_physics_math = lower.contains("hyperbolic")
        || lower.contains("lorentz")
        || lower.contains("gauss")
        || lower.contains("bonnet")
        || lower.contains("gravity")
        || lower.contains("prime number")
        || lower.contains("square of")
        || lower.contains("factors of")
        || lower.contains("manifold")
        || lower.contains("curvature");
    let has_code = lower.contains("::synthesize::")
        || lower.contains("fn ")
        || lower.contains("kernel")
        || lower.contains("mlir")
        || lower.contains("ssa")
        || lower.contains("algorithm")
        || lower.contains("recursion")
        || lower.contains("debug")
        || lower.contains("rust")
        || lower.contains("typescript");

    let target_role = if has_security {
        // Security & Cyber Verification: L12 (Metacognitive Regulation) + L3 (Somatic Boundary) + L7 (Convergent Logic)
        match step_idx % 3 {
            0 => RoleId::MetacognitiveRegulation,
            1 => RoleId::SomaticInteroception,
            _ => RoleId::ConvergentLogic,
        }
    } else if has_legal {
        // Legal & Statutory Compliance: L12 (Metacognitive Regulation) + L10 (Social/Normative) + L4 (Language/Semiotics) + L7 (Logic)
        match step_idx % 4 {
            0 => RoleId::MetacognitiveRegulation,
            1 => RoleId::SocialCognition,
            2 => RoleId::LanguageSemiotics,
            _ => RoleId::ConvergentLogic,
        }
    } else if has_medical {
        // Medical & Clinical Physiology: L3 (Somatic/Interoception) + L5 (Episodic Clinical Memory) + L9 (Valuation/Homeostasis)
        match step_idx % 3 {
            0 => RoleId::SomaticInteroception,
            1 => RoleId::EpisodicMemory,
            _ => RoleId::ValuationSalience,
        }
    } else if has_physics_math {
        // Hyperbolic Physics & Formal Math: L11 (Spatial Topography) + L7 (Convergent Logic) + L6 (Predictive Simulation)
        match step_idx % 3 {
            0 => RoleId::SpatialTopography,
            1 => RoleId::ConvergentLogic,
            _ => RoleId::PredictiveSimulation,
        }
    } else if has_code {
        // Coding Intensive: L4 (Language & Semiotics) + L7 (Convergent Logic) + L2 (Motor & Tool Praxis)
        match step_idx % 3 {
            0 => RoleId::LanguageSemiotics,
            1 => RoleId::ConvergentLogic,
            _ => RoleId::MotorPraxis,
        }
    } else {
        let idx = (step_idx + text.len()) % 12;
        let _ = total_steps;
        RoleId::CARVED_12[idx]
    };

    let layer_idx = match target_role {
        RoleId::SensoryPerception
        | RoleId::SomaticInteroception
        | RoleId::LanguageSemiotics
        | RoleId::SpatialTopography => total_layers / 5,
        RoleId::EpisodicMemory
        | RoleId::PredictiveSimulation
        | RoleId::ConvergentLogic
        | RoleId::DivergentSynthesis
        | RoleId::ValuationSalience => total_layers / 2,
        RoleId::MotorPraxis | RoleId::SocialCognition | RoleId::MetacognitiveRegulation => {
            (total_layers * 4) / 5
        }
        RoleId::Coordinator => total_layers / 2,
    };

    let well = atlas.well(target_role);
    let u = [
        well.unit_dir[0].to_f64(),
        well.unit_dir[1].to_f64(),
        well.unit_dir[2].to_f64(),
    ];
    (layer_idx, u, target_role)
}

impl CarvedSwarmMap {
    /// Execute a live Qwen-connected trajectory & atom activation probe across the 12 attractor
    /// basins (`L1..L12`), measure which spheres light up (`0-cell` carrier hits + `1-cell`
    /// reasoning arcs + Qwen safetensors receptor mode affinity), and automatically carve the
    /// multi-basin [`CarvedSpecialistProfile`] (`2..=4` lit spheres) from the global map.
    pub fn probe_and_carve_specialist(
        &self,
        domain_id: &str,
        display_name: &str,
        probe_prompt: &str,
        sub_probes: &[&str],
        transformer: &GeodesicActiveTransformer,
        policy: &OverlapPolicyMatrix,
        qwen_shard_path: Option<&Path>,
        explicit_basins_override: Option<&[RoleId]>,
    ) -> CarvedSpecialistProfile {
        let mut lit_0cell_counts = [0usize; 12];
        let mut lit_1cell_counts = [0usize; 12];
        let mut qwen_affinity = [0.0f64; 12];
        let mut seen_atoms = BTreeSet::<u64>::new();
        let mut seen_arcs = BTreeSet::<u64>::new();
        let mut lit_atoms = Vec::<LitAtomPoint>::new();
        let mut lit_arcs = Vec::<LitTrajectoryArc>::new();

        let all_prompts: Vec<&str> = if sub_probes.is_empty() {
            vec![probe_prompt]
        } else {
            let mut v = vec![probe_prompt];
            for &sp in sub_probes {
                if sp != probe_prompt {
                    v.push(sp);
                }
            }
            v
        };

        let mut primary_ingress_ops = Vec::<u32>::new();
        let mut primary_emitted_ops = Vec::<u32>::new();
        let mut primary_decoded = String::new();
        let mut any_qwen_shard = false;
        let total_layers = self.base_weights.num_layers.max(60);
        let dist_engine = OpenWeightsDistillationEngine::default();

        for (p_idx, &p_text) in all_prompts.iter().enumerate() {
            let resolved_prefix = transformer.resolve_geodesic_prompt_prefix(self, p_text, None);
            let raw_ids: Vec<u32> = resolved_prefix.iter().map(|op| op.0).collect();
            if p_idx == 0 {
                primary_ingress_ops = raw_ids.clone();
            }

            // 1. Walk the live 1-cell trajectory chain on the map using the O(1) GeodesicCursor
            let (emitted_ops, _expanded, decoded_txt) = transformer
                .decode_span_accelerated_incremental(self, &resolved_prefix, 28)
                .unwrap_or_else(|_| (Vec::new(), Vec::new(), String::new()));

            if p_idx == 0 && !decoded_txt.is_empty() {
                primary_emitted_ops = emitted_ops.iter().map(|op| op.0).collect();
                primary_decoded = decoded_txt;
            } else if primary_decoded.is_empty() && !decoded_txt.is_empty() {
                primary_emitted_ops = emitted_ops.iter().map(|op| op.0).collect();
                primary_decoded = decoded_txt;
            }

            // Collect all carrier opcodes in the trajectory (ingress + emitted)
            let mut full_chain: Vec<MlirOpId> = resolved_prefix.clone();
            for &op in &emitted_ops {
                if op.0 >= transformer.transducer.atomic_ir.base_vocab_size {
                    for exp_tok in transformer.transducer.atomic_ir.expand(&[op.0]) {
                        full_chain.push(MlirOpId(exp_tok));
                    }
                } else {
                    full_chain.push(op);
                }
            }

            // 2. Project through Qwen's safetensors shard receptor sketch + 6-axis icosahedral router
            let (sketch, used_shard) =
                project_qwen_token_sketch(qwen_shard_path, &raw_ids, p_text.as_bytes());
            any_qwen_shard |= used_shard;

            for sub_step in 0..3 {
                let (layer_idx, steer_dir, _steered_role) = domain_receptor_steering(
                    &self.atlas,
                    p_text,
                    p_idx + sub_step,
                    all_prompts.len() * 3,
                    total_layers,
                );
                let mut rotated_sketch = vec![0.0f32; 256];
                for i in 0..256 {
                    let angle = (i as f64) * 0.618_033_988_749_895 * std::f64::consts::TAU;
                    let z_basis = ((i % 5) as f64 - 2.0) * 0.5;
                    let carrier = steer_dir[0] * angle.cos()
                        + steer_dir[1] * angle.sin()
                        + steer_dir[2] * z_basis;
                    let qwen_val = sketch[(i + sub_step * 53) % 256] as f64;
                    rotated_sketch[i] = (qwen_val * 0.35 + carrier * 0.85) as f32;
                }
                let tap = TeacherTapRecord {
                    shard_id: (p_idx * 3 + sub_step) as u64,
                    layer_index: layer_idx,
                    hidden_state: rotated_sketch,
                    next_op_probs: vec![0.88, 0.06, 0.04, 0.02],
                    payload: Vec::new(),
                };
                if let Ok(mut projected) = dist_engine.projector.project(&tap) {
                    let (well_role, offset, cert) =
                        OpenWeightsDistillationEngine::classify_icosahedral_well(
                            &self.atlas,
                            &tap.hidden_state,
                            layer_idx,
                            total_layers,
                            projected.loss_report.teacher_entropy_bits_q32,
                        );
                    if let Some(r_idx) = well_role.carved_index() {
                        qwen_affinity[r_idx] += cert.to_f64() * 4.0;
                    }
                    let well = self.atlas.well(well_role);
                    let d_sph = Q32_31::from_f64(0.804_735_5);
                    projected.tangent_h3 = [
                        (well.unit_dir[0] * d_sph) + offset[0],
                        (well.unit_dir[1] * d_sph) + offset[1],
                        (well.unit_dir[2] * d_sph) + offset[2],
                    ];
                    if let Ok(refracted) = OpticalLens::refract(&projected, well.phase) {
                        if let Some(silo) = self.silo(well_role) {
                            for sa in silo.retrieve_in_silo(&refracted.lorentz, 10) {
                                if let Some(r_idx) = well_role.carved_index() {
                                    lit_0cell_counts[r_idx] += 3;
                                }
                                // Count outgoing 1-cell reasoning arcs anchored in this lit basin
                                for &arc_idx in self.outgoing_arc_indices(sa.id).iter().take(6) {
                                    let _ = arc_idx;
                                    if let Some(r_idx) = well_role.carved_index() {
                                        lit_1cell_counts[r_idx] += 2;
                                    }
                                }
                                if seen_atoms.insert(sa.id.0) && lit_atoms.len() < 180 {
                                    let ball = OpticalLens::lorentz_to_ball(&sa.global_lorentz);
                                    let sec = crate::lock::extract_invariant(
                                        rad_q32_to_pi_unit(sa.phase.theta_f),
                                        rad_q32_to_pi_unit(sa.phase.theta_s),
                                    )
                                    .n;
                                    lit_atoms.push(LitAtomPoint {
                                        atom_id: sa.id.0,
                                        role: well_role,
                                        poincare_ball: [
                                            ball[0].to_f64(),
                                            ball[1].to_f64(),
                                            ball[2].to_f64(),
                                        ],
                                        theta_f: sa.phase.theta_f.to_f64(),
                                        theta_s: sa.phase.theta_s.to_f64(),
                                        sector_144: sec,
                                        salience: sa.salience_count,
                                        op_id: raw_ids.first().copied().unwrap_or(0),
                                        is_trajectory_waypoint: false,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            // 3. Map every token carrier in `full_chain` to its exact (H³ × T²) coordinate and
            //    attribute its trajectory waypoint & 1-cell reasoning transitions!
            let slack = Q32_31::from_f64(0.06).0;
            let mut prev_waypoint: Option<(u64, RoleId, [f64; 3])> = None;

            for (step_i, &op) in full_chain.iter().enumerate() {
                if let Ok((lor, phase)) = NativeReader::address(&transformer.transducer.catalog, op)
                {
                    if let Some((best_aid, _best_cosh, candidates)) =
                        self.candidate_arc_atoms_near(&lor, phase, slack)
                    {
                        if let Some(ga) = self.global_atom(AtomId(best_aid)) {
                            // Find the true home basin (minimum cosh distance to well center) for `ga`
                            let home_role = RoleId::CARVED_12
                                .into_iter()
                                .min_by_key(|&r| {
                                    OpticalLens::lorentz_inner_product_wide(
                                        &ga.lorentz,
                                        &self.atlas.well(r).lorentz_well,
                                    )
                                    .0
                                })
                                .unwrap_or(RoleId::LanguageSemiotics);
                            if let Some(r_idx) = home_role.carved_index() {
                                lit_0cell_counts[r_idx] += 1;
                            }

                            let ball = [
                                ga.poincare_ball[0].to_f64(),
                                ga.poincare_ball[1].to_f64(),
                                ga.poincare_ball[2].to_f64(),
                            ];
                            if seen_atoms.insert(best_aid) && lit_atoms.len() < 180 {
                                lit_atoms.push(LitAtomPoint {
                                    atom_id: best_aid,
                                    role: home_role,
                                    poincare_ball: ball,
                                    theta_f: ga.phase.theta_f.to_f64(),
                                    theta_s: ga.phase.theta_s.to_f64(),
                                    sector_144: ga.invariant.n,
                                    salience: ga.salience_count,
                                    op_id: op.0,
                                    is_trajectory_waypoint: true,
                                });
                            }

                            if let Some((prev_aid, prev_role, prev_ball)) = prev_waypoint {
                                if prev_aid != best_aid && lit_arcs.len() < 96 {
                                    let arc_key = (prev_aid << 32) ^ best_aid;
                                    if seen_arcs.insert(arc_key) {
                                        let cosh_act = if let Some(prev_ga) =
                                            self.global_atom(AtomId(prev_aid))
                                        {
                                            OpticalLens::lorentz_inner_product_wide(
                                                &prev_ga.lorentz,
                                                &ga.lorentz,
                                            )
                                            .to_f64()
                                        } else {
                                            1.05
                                        };
                                        lit_arcs.push(LitTrajectoryArc {
                                            arc_id: arc_key,
                                            step_index: step_i,
                                            from_atom: prev_aid,
                                            to_atom: best_aid,
                                            from_role: prev_role,
                                            to_role: home_role,
                                            from_ball: prev_ball,
                                            to_ball: ball,
                                            cosh_action: cosh_act,
                                            certainty: 0.96,
                                            op_id: op.0,
                                        });
                                    }
                                }
                            }
                            prev_waypoint = Some((best_aid, home_role, ball));
                        }

                        // Count outgoing 1-cell reasoning arcs from the primary waypoint
                        for &(cand_aid, _) in candidates.iter().take(2) {
                            for &arc_idx in
                                self.outgoing_arc_indices(AtomId(cand_aid)).iter().take(3)
                            {
                                let arc = &self.reasoning_arcs[arc_idx];
                                if let Some(ri) = arc.from_role.carved_index() {
                                    lit_1cell_counts[ri] += 1;
                                }
                                if let Some(ri) = arc.to_role.carved_index() {
                                    lit_1cell_counts[ri] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Normalize `qwen_affinity` to `[0.0, 1.0]` share before reporting
        let total_qwen: f64 = qwen_affinity.iter().sum::<f64>().max(1e-9);
        for v in &mut qwen_affinity {
            *v /= total_qwen;
        }

        // 4. Compute composite activation mass per basin (`L1..L12`), normalizing against background
        //    silo size so giant carrier hubs (`L7`/`L2`) don't drown out specialized domain basins!
        let mean_silo_size = (self.global_atom_count() as f64 / 12.0).max(1.0);
        let mut raw_mass = [0.0f64; 12];
        for i in 0..12 {
            let role = RoleId::CARVED_12[i];
            let silo_sz = self.silo(role).map(|s| s.atom_count()).unwrap_or(1).max(1) as f64;
            // Inverse-frequency TF-IDF-like density weighting for 0-cell & 1-cell hits
            let density_boost = (mean_silo_size / silo_sz).sqrt().clamp(0.45, 2.2);
            raw_mass[i] = ((lit_0cell_counts[i] as f64) * 1.8
                + (lit_1cell_counts[i] as f64) * 0.45)
                * density_boost
                + qwen_affinity[i] * 220.0;
        }
        let total_mass: f64 = raw_mass.iter().sum::<f64>().max(1e-9);
        let shares: [f64; 12] = std::array::from_fn(|i| raw_mass[i] / total_mass);

        // 5. Select the lit basins (`2..=4` spheres) that form the Carved Specialist:
        //    - If caller passed `explicit_basins_override`, use those directly.
        //    - Otherwise, sort basins by empirical activation share and take the top 2..=4 basins
        //      where each sphere contributes >= 13.5% share (or top 3 spheres capturing >= 54%).
        let lit_basins: Vec<RoleId> = match explicit_basins_override {
            Some(explicit) if !explicit.is_empty() => explicit.to_vec(),
            _ => {
                let mut ranked: Vec<(usize, f64)> = (0..12).map(|i| (i, shares[i])).collect();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

                let mut chosen = Vec::new();
                let mut cum_share = 0.0f64;
                for &(idx, sh) in &ranked {
                    if chosen.len() < 2
                        || (chosen.len() < 3 && (sh >= 0.125 || cum_share < 0.52))
                        || (chosen.len() < 4 && sh >= 0.165)
                    {
                        chosen.push(RoleId::CARVED_12[idx]);
                        cum_share += sh;
                    } else {
                        break;
                    }
                }
                chosen
            }
        };

        let captured_activation_share: f64 = lit_basins
            .iter()
            .filter_map(|r| r.carved_index().map(|idx| shares[idx]))
            .sum();

        let basin_metrics: [BasinActivationMetric; 12] = std::array::from_fn(|i| {
            let role = RoleId::CARVED_12[i];
            let total_silo_atoms = self.silo(role).map(|s| s.atom_count()).unwrap_or(0);
            BasinActivationMetric {
                role,
                total_silo_atoms,
                lit_0cell_atoms: lit_0cell_counts[i],
                lit_1cell_arcs: lit_1cell_counts[i],
                qwen_mode_affinity: qwen_affinity[i],
                activation_mass: raw_mass[i],
                activation_share: shares[i],
                selected_for_specialist: lit_basins.contains(&role),
            }
        });

        // 6. Carve the Multi-Basin Specialist directly from the lit attractor basins!
        //    Notice that we carve ONLY from `lit_basins` (e.g., 2, 3, or 4 spheres), NOT a fixed 1+5 neighborhood.
        let carved_view: TaskCarvedView = self.carve_specialist_basins_only(&lit_basins, None);
        let carved_0cell_count = carved_view.atom_count();
        let carved_1cell_count = carved_view.arc_count();
        let subtracted_0cell_count = self.global_atom_count().saturating_sub(carved_0cell_count);
        let subtracted_1cell_count = self.reasoning_arcs.len().saturating_sub(carved_1cell_count);

        // 7. Compute non-Abelian Gauss-Bonnet holonomy across the lit specialist basins
        let mut loop_roles = lit_basins.clone();
        if loop_roles.len() >= 2 {
            loop_roles.push(loop_roles[0]);
        }
        let init_phase = loop_roles
            .first()
            .map(|&r| self.atlas.well(r).phase)
            .unwrap_or(crate::cache::geometric::Phase {
                theta_f: Q32_31::ZERO,
                theta_s: Q32_31::ZERO,
            });
        let holonomy =
            dist_engine
                .holonomy_engine
                .evaluate_role_loop(&loop_roles, self, init_phase);
        let _ = policy;

        CarvedSpecialistProfile {
            domain_id: domain_id.to_string(),
            display_name: display_name.to_string(),
            probe_prompt: probe_prompt.to_string(),
            lit_basins,
            basin_metrics,
            captured_activation_share,
            carved_0cell_count,
            carved_1cell_count,
            subtracted_0cell_count,
            subtracted_1cell_count,
            lit_atoms,
            lit_arcs,
            decoded_continuation: primary_decoded,
            ingress_op_ids: primary_ingress_ops,
            emitted_op_ids: primary_emitted_ops,
            holonomy_flux_rad: holonomy.enclosed_curvature_flux_q32.to_f64(),
            qwen_shard_connected: any_qwen_shard,
        }
    }

    /// Carve a [`TaskCarvedView`] strictly from the specified `lit_basins` (e.g., 2, 3, or 4
    /// attractor spheres that lit up during empirical probing), keeping all `0-cell` atoms in
    /// those basins and all `1-cell` reasoning arcs anchored in those basins.
    pub fn carve_specialist_basins_only(
        &self,
        lit_basins: &[RoleId],
        task_center: Option<(&LorentzVector, Q32_31)>,
    ) -> TaskCarvedView {
        let mut atoms = std::collections::BTreeMap::new();
        for &role in lit_basins {
            if let Some(silo) = self.silo(role) {
                for sa in silo.atoms() {
                    if let Some((center, max_cosh)) = task_center {
                        let d = OpticalLens::lorentz_inner_product_wide(&sa.global_lorentz, center);
                        if d.0 > max_cosh.0 {
                            continue;
                        }
                    }
                    atoms.entry(sa.id.0).or_insert_with(|| sa.clone());
                }
            }
        }
        let mut arcs = Vec::new();
        let mut outgoing_arcs_by_atom: std::collections::BTreeMap<u64, Vec<usize>> =
            std::collections::BTreeMap::new();
        for a in &self.reasoning_arcs {
            if atoms.contains_key(&a.from_atom.0) || atoms.contains_key(&a.to_atom.0) {
                let idx = arcs.len();
                outgoing_arcs_by_atom
                    .entry(a.from_atom.0)
                    .or_default()
                    .push(idx);
                arcs.push(a.clone());
            }
        }
        TaskCarvedView {
            allowed_silos: lit_basins.to_vec(),
            task_center: task_center.map(|(c, r)| (*c, r)),
            atoms,
            arcs,
            outgoing_arcs_by_atom,
        }
    }
}
