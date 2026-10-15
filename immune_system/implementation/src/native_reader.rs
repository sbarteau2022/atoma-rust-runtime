//! Native reader: one generation step whose only model is `CarvedSwarmMap`.
//!
//! An ML-IR ID is a point on `(H³ × T²)`. The prefix is a path of those points.
//! The reader stands at the last point, queries the global atoms (every origin
//! fingerprint), and follows the 1-cell whose stored gyroscope matches that path.
//! The arc's `compiled_mlir_op` is the next ML-IR ID.
//!
//! A token trajectory plants the from-atom on that same point and stores the
//! prefix gyroscope on the arc. The hidden state rides in the atom payload.
//!
//! Layer count, hidden width, expert count, and vocabulary size are not inputs.

use crate::atlas13::RoleId;
use crate::cache::geometric::{AtomId, Phase};
use crate::compliance::OverlapPolicyMatrix;
use crate::distill_engine::DistilledReasoningArc;
use crate::geometry::fixed::Q32_31;
use crate::geometry::lorentz::LorentzVector;
use crate::holonomy::{HyperbolicHolonomyEngine, IDENTITY_FRAME_Q32};
use crate::lock::extract_invariant;
use crate::mlir_native::{MlirAtomCatalog, MlirOpId};
use crate::refraction::{
    OpticalLens, ProjectionLossReport, RefractedPoint, RefractionLossReport, Sha256Digest,
    rad_q32_to_pi_unit,
};
use crate::silo::CarvedSwarmMap;
use std::collections::BTreeMap;

/// `cosh` of the widest query the reader will treat as standing on the map.
/// Farther than this, the nearest arc is a different place and is not a continuation.
fn query_basin_cosh() -> i64 {
    Q32_31::from_f64(1.25).0
}

/// Atoms this close to the nearest arc-bearing atom (in `cosh`) share its basin.
fn neighborhood_slack() -> i64 {
    Q32_31::from_f64(0.05).0
}

/// One geometric continuation read from the global map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeRead {
    pub next_id: MlirOpId,
    pub arc_id: u64,
    pub from_atom: AtomId,
    pub to_atom: AtomId,
    pub query_cosh_q32: Q32_31,
    pub cosh_action_q32: Q32_31,
    /// Frobenius product of the prefix gyroscope and the arc's stored Wigner frame.
    pub frame_alignment_q32: Q32_31,
    pub origin_model_fingerprint: Sha256Digest,
}

/// Why a multi-step read stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeStop {
    MaxNew,
    Eos(MlirOpId),
    FixedPoint(MlirOpId),
    NoGeodesic { at: MlirOpId, detail: String },
}

/// IDs read from the map, and why the walk stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeContinuation {
    pub reads: Vec<NativeRead>,
    pub stop: NativeStop,
}

/// One token step written into the map, and the read that step now produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrajectoryWrite {
    pub read: NativeRead,
    pub arc_was_novel: bool,
}

/// The play path. Distillation fills the map. This reads it.
pub struct NativeReader;

impl NativeReader {
    /// Place an ML-IR ID on `(H³ × T²)`.
    ///
    /// A registered catalog opcode keeps the well the catalog pinned.
    /// Every other ID, including a token carrier from any vocabulary, is a
    /// pure function of the integer. There is no embedding matrix.
    pub fn address(
        catalog: &MlirAtomCatalog,
        id: MlirOpId,
    ) -> Result<(LorentzVector, Phase), String> {
        if let Some(desc) = catalog.get_op(id) {
            return Ok((desc.lorentz_well, desc.phase_well));
        }
        token_carrier_address(id.0)
    }

    /// Read the next ML-IR ID from `map` given the IDs so far.
    pub fn step(
        map: &CarvedSwarmMap,
        catalog: &MlirAtomCatalog,
        prefix: &[MlirOpId],
    ) -> Result<NativeRead, String> {
        let Some(&last) = prefix.last() else {
            return Err("native reader needs at least one ML-IR ID".to_string());
        };
        let (query, frame) = context(catalog, prefix)?;

        let mut sources = BTreeMap::<u64, ()>::new();
        for arc in &map.reasoning_arcs {
            sources.insert(arc.from_atom.0, ());
        }
        if sources.is_empty() {
            return Err("map has no reasoning arcs".to_string());
        }

        let mut atom_cosh: BTreeMap<u64, i64> = BTreeMap::new();
        for (_key, atom) in map.iter_global_atoms() {
            if !sources.contains_key(&atom.id.0) {
                continue;
            }
            let cosh = OpticalLens::lorentz_inner_product_wide(&query, &atom.lorentz);
            atom_cosh.insert(atom.id.0, cosh.0);
        }
        let Some((&best_atom, &best_cosh)) = atom_cosh.iter().min_by_key(|(_, c)| *c) else {
            return Err(format!("no arc-bearing atom on the map for {last}"));
        };
        if best_cosh > query_basin_cosh() {
            return Err(format!(
                "no geodesic within cosh {:.4} of {last} (nearest arc leaves atom {best_atom} at cosh {:.4})",
                Q32_31(query_basin_cosh()).to_f64(),
                Q32_31(best_cosh).to_f64(),
            ));
        }

        let band = best_cosh.saturating_add(neighborhood_slack());
        let winner = map
            .reasoning_arcs
            .iter()
            .filter(|arc| atom_cosh.get(&arc.from_atom.0).is_some_and(|c| *c <= band))
            .min_by(|a, b| {
                let align_a = frame_alignment(&a.wigner_frame, &frame);
                let align_b = frame_alignment(&b.wigner_frame, &frame);
                align_b
                    .cmp(&align_a)
                    .then(a.cosh_action_q32.0.cmp(&b.cosh_action_q32.0))
                    .then(b.teacher_certainty_q32.0.cmp(&a.teacher_certainty_q32.0))
                    .then(a.arc_id.cmp(&b.arc_id))
            })
            .ok_or_else(|| format!("basin of atom {best_atom} has no outgoing arc"))?;

        let query_cosh = atom_cosh
            .get(&winner.from_atom.0)
            .copied()
            .unwrap_or(best_cosh);
        Ok(NativeRead {
            next_id: winner.compiled_mlir_op,
            arc_id: winner.arc_id,
            from_atom: winner.from_atom,
            to_atom: winner.to_atom,
            query_cosh_q32: Q32_31(query_cosh),
            cosh_action_q32: winner.cosh_action_q32,
            frame_alignment_q32: Q32_31(frame_alignment(&winner.wigner_frame, &frame)),
            origin_model_fingerprint: winner.origin_model_fingerprint,
        })
    }

    /// Walk up to `max_new` continuations, appending each ID to the prefix.
    pub fn continue_ids(
        map: &CarvedSwarmMap,
        catalog: &MlirAtomCatalog,
        prefix: &[MlirOpId],
        max_new: usize,
        eos: &[MlirOpId],
    ) -> Result<NativeContinuation, String> {
        if prefix.is_empty() {
            return Err("native reader needs at least one ML-IR ID".to_string());
        }
        let mut prefix_buf = prefix.to_vec();
        let mut reads = Vec::new();
        if max_new == 0 {
            return Ok(NativeContinuation {
                reads,
                stop: NativeStop::MaxNew,
            });
        }
        for _ in 0..max_new {
            match Self::step(map, catalog, &prefix_buf) {
                Ok(read) => {
                    let next = read.next_id;
                    let at = prefix_buf[prefix_buf.len() - 1];
                    reads.push(read);
                    if eos.contains(&next) {
                        return Ok(NativeContinuation {
                            reads,
                            stop: NativeStop::Eos(next),
                        });
                    }
                    if next == at {
                        return Ok(NativeContinuation {
                            reads,
                            stop: NativeStop::FixedPoint(next),
                        });
                    }
                    prefix_buf.push(next);
                }
                Err(detail) => {
                    let at = prefix_buf[prefix_buf.len() - 1];
                    return Ok(NativeContinuation {
                        reads,
                        stop: NativeStop::NoGeodesic { at, detail },
                    });
                }
            }
        }
        Ok(NativeContinuation {
            reads,
            stop: NativeStop::MaxNew,
        })
    }

    /// Write one teacher step onto the wells the reader queries.
    ///
    /// `prefix` is the ML-IR IDs so far. `next_id` is the teacher's next ID.
    /// `hidden_state` is the teacher state at the last prefix ID.
    pub fn record_step(
        map: &mut CarvedSwarmMap,
        catalog: &MlirAtomCatalog,
        policy: &OverlapPolicyMatrix,
        prefix: &[MlirOpId],
        next_id: MlirOpId,
        hidden_state: &[f32],
        origin: Sha256Digest,
        certainty: Q32_31,
    ) -> Result<TrajectoryWrite, String> {
        if prefix.is_empty() {
            return Err("native reader needs at least one ML-IR ID".to_string());
        }
        let (_query, frame) = context(catalog, prefix)?;
        let last = prefix[prefix.len() - 1];
        let from_atom = plant_carrier(
            map,
            policy,
            catalog,
            last,
            hidden_payload(hidden_state),
            origin,
        )?;
        let to_atom = plant_carrier(map, policy, catalog, next_id, Vec::new(), origin)?;
        let from_lor = map
            .global_atom(from_atom)
            .map(|atom| atom.lorentz)
            .ok_or_else(|| format!("from atom {} missing after plant", from_atom.0))?;
        let to_lor = map
            .global_atom(to_atom)
            .map(|atom| atom.lorentz)
            .ok_or_else(|| format!("to atom {} missing after plant", to_atom.0))?;
        let cosh_action = OpticalLens::lorentz_inner_product_wide(&from_lor, &to_lor);
        let (_arc_id, arc_was_novel) = map.record_trajectory_arc(DistilledReasoningArc {
            arc_id: 0,
            chain_id: prefix_chain_id(prefix),
            step_index: prefix.len() - 1,
            from_atom,
            to_atom,
            from_role: RoleId::LanguageSemiotics,
            to_role: RoleId::PredictiveSimulation,
            cosh_action_q32: cosh_action,
            wigner_frame: frame,
            compiled_mlir_op: next_id,
            teacher_certainty_q32: certainty,
            origin_model_fingerprint: origin,
        });
        let read = Self::step(map, catalog, prefix)?;
        if read.next_id != next_id {
            return Err(format!(
                "recorded mlir_op#{} and the reader emitted {} for this prefix",
                next_id.0, read.next_id.0
            ));
        }
        Ok(TrajectoryWrite {
            read,
            arc_was_novel,
        })
    }

    /// Write each generated ID as the next step of `prefix`, extending the prefix as it goes.
    ///
    /// An empty `hidden_at_step` stores no teacher state. A non-empty slice has one state per
    /// generated ID, the state at the prefix the reader is standing on when that ID is emitted.
    pub fn record_continuation(
        map: &mut CarvedSwarmMap,
        catalog: &MlirAtomCatalog,
        policy: &OverlapPolicyMatrix,
        prefix: &[MlirOpId],
        generated: &[MlirOpId],
        hidden_at_step: &[Vec<f32>],
        origin: Sha256Digest,
        certainty: Q32_31,
    ) -> Result<Vec<TrajectoryWrite>, String> {
        if !hidden_at_step.is_empty() && hidden_at_step.len() != generated.len() {
            return Err("hidden states must cover every generated ID".to_string());
        }
        let mut prefix_buf = prefix.to_vec();
        let mut wrote = Vec::with_capacity(generated.len());
        for (i, next) in generated.iter().copied().enumerate() {
            let hidden = hidden_at_step.get(i).map(Vec::as_slice).unwrap_or(&[]);
            let step = Self::record_step(
                map,
                catalog,
                policy,
                &prefix_buf,
                next,
                hidden,
                origin,
                certainty,
            )?;
            prefix_buf.push(next);
            wrote.push(step);
        }
        Ok(wrote)
    }
}

/// Arrival point and gyroscope after walking the prefix from the hyperboloid origin.
fn context(
    catalog: &MlirAtomCatalog,
    prefix: &[MlirOpId],
) -> Result<(LorentzVector, [[Q32_31; 3]; 3]), String> {
    let (mut lor, _) = NativeReader::address(catalog, prefix[0])?;
    let mut frame = HyperbolicHolonomyEngine::transport_frame_hop(
        &LorentzVector::origin(),
        &lor,
        &IDENTITY_FRAME_Q32,
    );
    for id in &prefix[1..] {
        let (next, _) = NativeReader::address(catalog, *id)?;
        frame = HyperbolicHolonomyEngine::transport_frame_hop(&lor, &next, &frame);
        lor = next;
    }
    Ok((lor, frame))
}

/// Poincaré ball point of an unregistered ID, lifted by the integer optical map.
fn token_carrier_address(id: u32) -> Result<(LorentzVector, Phase), String> {
    const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;
    let n = (id as f64) + 1.0;
    let radius = 0.20 + 0.55 * (n * 0.618_033_988_749_895).fract();
    let theta = n * GOLDEN_ANGLE;
    let z = 0.12 * ((((id.wrapping_mul(0x9E37_79B9)) >> 16) as f64) / 65536.0 - 0.5);
    let x = radius * theta.cos();
    let y = radius * theta.sin();
    let norm_sq = x * x + y * y + z * z;
    let scale = if norm_sq > 0.81 {
        (0.81 / norm_sq).sqrt()
    } else {
        1.0
    };
    let ball = [
        Q32_31::from_f64(x * scale),
        Q32_31::from_f64(y * scale),
        Q32_31::from_f64(z * scale),
    ];
    let lorentz = OpticalLens::ball_to_lorentz(&ball).map_err(|e| e.to_string())?;
    let theta_f = ((id % 2048) as f64) * (std::f64::consts::TAU / 2048.0) - std::f64::consts::PI;
    let theta_s = ((id / 2048) as f64).rem_euclid(2048.0) * (std::f64::consts::TAU / 2048.0)
        - std::f64::consts::PI;
    let phase = Phase {
        theta_f: Q32_31::from_f64(theta_f),
        theta_s: Q32_31::from_f64(theta_s),
    };
    Ok((lorentz, phase))
}

fn plant_carrier(
    map: &mut CarvedSwarmMap,
    policy: &OverlapPolicyMatrix,
    catalog: &MlirAtomCatalog,
    id: MlirOpId,
    payload: Vec<u8>,
    origin: Sha256Digest,
) -> Result<AtomId, String> {
    let point = refract_id(catalog, id)?;
    let (atom, novel) = map.keep_and_carve_novel_only(point, payload.clone(), origin, policy);
    if !novel {
        map.store_payload_if_empty(atom, payload);
    }
    Ok(atom)
}

/// Lift an ML-IR ID to the refracted point `address` returns.
fn refract_id(catalog: &MlirAtomCatalog, id: MlirOpId) -> Result<RefractedPoint, String> {
    let (lorentz, phase) = NativeReader::address(catalog, id)?;
    Ok(RefractedPoint {
        poincare_ball: OpticalLens::lorentz_to_ball(&lorentz),
        lorentz,
        phase,
        invariant: extract_invariant(
            rad_q32_to_pi_unit(phase.theta_f),
            rad_q32_to_pi_unit(phase.theta_s),
        ),
        projection_loss: ProjectionLossReport {
            input_state_bits: 32,
            output_tangent_bits: 256,
            discarded_dimensions: 0,
            retained_energy_ratio_q32: Q32_31::ONE,
            discarded_energy_ratio_q32: Q32_31::ZERO,
            teacher_entropy_bits_q32: Q32_31::ZERO,
        },
        refraction_loss: RefractionLossReport {
            quantization_ulp_bound: 1,
            hyperboloid_residual_q32: Q32_31::ZERO,
            refractive_index_q32: Q32_31::ONE,
        },
    })
}

fn hidden_payload(hidden: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(hidden.len() * 4);
    for &v in hidden {
        let canonical = if v == 0.0 { 0.0 } else { v };
        buf.extend_from_slice(&canonical.to_le_bytes());
    }
    buf
}

fn prefix_chain_id(prefix: &[MlirOpId]) -> u64 {
    let mut acc = 0x9E37_79B9_7F4A_7C15u64;
    for id in prefix {
        acc = acc
            .wrapping_mul(0x0000_0100_0000_01B3)
            .wrapping_add(u64::from(id.0));
    }
    acc
}

fn frame_alignment(a: &[[Q32_31; 3]; 3], b: &[[Q32_31; 3]; 3]) -> i64 {
    let mut acc: i128 = 0;
    for r in 0..3 {
        for c in 0..3 {
            acc += a[r][c].0 as i128 * b[r][c].0 as i128;
        }
    }
    (acc >> 31).clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas13::{Atlas13, RoleId};
    use crate::compliance::OverlapPolicyMatrix;
    use crate::distill_engine::DistilledReasoningArc;
    use crate::geometry::fixed::Q32_31;
    use crate::holonomy::IDENTITY_FRAME_Q32;
    use crate::map_native::CoupledOperatorAtomPayload;
    use crate::refraction::Sha256Digest;
    use crate::silo::{CarvedSwarmMap, FrozenBaseWeights};
    use std::sync::Arc;

    fn fixture(
        name: &str,
        layers: usize,
    ) -> (CarvedSwarmMap, OverlapPolicyMatrix, MlirAtomCatalog) {
        let atlas = Atlas13::approved_icosahedral();
        let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
        let base = FrozenBaseWeights::freeze(name, layers, vec![0.1; 32]);
        let map = CarvedSwarmMap::new(atlas, base, Q32_31::from_f64(1.22));
        let catalog = MlirAtomCatalog::new_canonical(248_320);
        (map, policy, catalog)
    }

    fn place(
        map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
        catalog: &MlirAtomCatalog,
        id: u32,
        payload: Vec<u8>,
        origin: Sha256Digest,
    ) -> AtomId {
        let refracted = refract_id(catalog, MlirOpId(id)).unwrap();
        let (atom, novel) = map.keep_and_carve_novel_only(refracted, payload, origin, policy);
        assert!(novel, "ID {id} coalesced into an existing atom");
        atom
    }

    fn link(
        map: &mut CarvedSwarmMap,
        from: AtomId,
        to: AtomId,
        op: u32,
        frame: [[Q32_31; 3]; 3],
        action: f64,
        certainty: f64,
        origin: Sha256Digest,
    ) {
        map.record_novel_reasoning_arc(DistilledReasoningArc {
            arc_id: 0,
            chain_id: 1,
            step_index: 0,
            from_atom: from,
            to_atom: to,
            from_role: RoleId::LanguageSemiotics,
            to_role: RoleId::ConvergentLogic,
            cosh_action_q32: Q32_31::from_f64(action),
            wigner_frame: frame,
            compiled_mlir_op: MlirOpId(op),
            teacher_certainty_q32: Q32_31::from_f64(certainty),
            origin_model_fingerprint: origin,
        });
    }

    fn coupled_lie(next_id: u32, h_dim: u32, layer: u16) -> Vec<u8> {
        let mut buf = Vec::with_capacity(36);
        buf.extend_from_slice(b"ATOMA_OP");
        buf.push(0x04);
        buf.push(0);
        buf.extend_from_slice(&layer.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&10u32.to_le_bytes());
        buf.extend_from_slice(&1.0f32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&h_dim.to_le_bytes());
        buf.extend_from_slice(&next_id.to_le_bytes());
        buf
    }

    #[test]
    fn address_uses_the_catalog_well_and_ignores_vocab_for_other_ids() {
        let wide = MlirAtomCatalog::new_canonical(248_320);
        let narrow = MlirAtomCatalog::new_canonical(202_048);
        let op = MlirOpId(wide.base_vocab_size);
        let (lor, phase) = NativeReader::address(&wide, op).unwrap();
        let desc = wide.get_op(op).unwrap();
        assert_eq!(lor.x, desc.lorentz_well.x);
        assert_eq!(phase, desc.phase_well);

        let (a, _) = NativeReader::address(&wide, MlirOpId(10)).unwrap();
        let (b, _) = NativeReader::address(&narrow, MlirOpId(10)).unwrap();
        assert_eq!(a.x, b.x);
        let self_cosh = OpticalLens::lorentz_inner_product_wide(&a, &a);
        assert!(self_cosh.0 <= Q32_31::from_f64(1.01).0);
    }

    #[test]
    fn gyroscope_match_beats_a_shorter_hop() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = map.base_weights.fingerprint;
        let a = place(&mut map, &policy, &catalog, 10, b"a".to_vec(), origin);
        let b = place(&mut map, &policy, &catalog, 11, b"b".to_vec(), origin);
        let c = place(&mut map, &policy, &catalog, 99, b"c".to_vec(), origin);
        let (_pos, frame) = context(&catalog, &[MlirOpId(10)]).unwrap();
        let flipped = [
            [Q32_31::from_f64(-1.0), Q32_31::ZERO, Q32_31::ZERO],
            [Q32_31::ZERO, Q32_31::from_f64(-1.0), Q32_31::ZERO],
            [Q32_31::ZERO, Q32_31::ZERO, Q32_31::from_f64(-1.0)],
        ];
        link(&mut map, a, c, 99, flipped, 1.01, 0.99, origin);
        link(&mut map, a, b, 11, frame, 2.0, 0.10, origin);

        let read = NativeReader::step(&map, &catalog, &[MlirOpId(10)]).unwrap();
        assert_eq!(read.next_id, MlirOpId(11));
        assert_eq!(read.to_atom, b);
    }

    #[test]
    fn equal_frames_follow_the_shorter_geodesic() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = map.base_weights.fingerprint;
        let a = place(&mut map, &policy, &catalog, 10, b"a".to_vec(), origin);
        let b = place(&mut map, &policy, &catalog, 11, b"b".to_vec(), origin);
        let c = place(&mut map, &policy, &catalog, 12, b"c".to_vec(), origin);
        let (_pos, frame) = context(&catalog, &[MlirOpId(10)]).unwrap();
        link(&mut map, a, c, 12, frame, 1.80, 0.99, origin);
        link(&mut map, a, b, 11, frame, 1.20, 0.10, origin);

        let read = NativeReader::step(&map, &catalog, &[MlirOpId(10)]).unwrap();
        assert_eq!(read.next_id, MlirOpId(11));
    }

    #[test]
    fn continuation_walks_the_chain_and_stops_when_the_chart_ends() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = map.base_weights.fingerprint;
        let a = place(&mut map, &policy, &catalog, 10, b"a".to_vec(), origin);
        let b = place(&mut map, &policy, &catalog, 11, b"b".to_vec(), origin);
        let c = place(&mut map, &policy, &catalog, 12, b"c".to_vec(), origin);
        let f10 = context(&catalog, &[MlirOpId(10)]).unwrap().1;
        let f11 = context(&catalog, &[MlirOpId(10), MlirOpId(11)]).unwrap().1;
        link(&mut map, a, b, 11, f10, 1.1, 0.5, origin);
        link(&mut map, b, c, 12, f11, 1.1, 0.5, origin);

        let walk = NativeReader::continue_ids(&map, &catalog, &[MlirOpId(10)], 5, &[]).unwrap();
        let ids: Vec<u32> = walk.reads.iter().map(|r| r.next_id.0).collect();
        assert_eq!(ids, vec![11, 12]);
        assert!(matches!(walk.stop, NativeStop::NoGeodesic { at, .. } if at == MlirOpId(12)));
    }

    #[test]
    fn a_distant_atom_is_not_a_continuation() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = map.base_weights.fingerprint;
        let far = place(
            &mut map,
            &policy,
            &catalog,
            50_000,
            vec![0u8; 8_000],
            origin,
        );
        let sink = place(&mut map, &policy, &catalog, 50_001, vec![0u8; 64], origin);
        link(
            &mut map,
            far,
            sink,
            77,
            IDENTITY_FRAME_Q32,
            1.0,
            1.0,
            origin,
        );

        let err = NativeReader::step(&map, &catalog, &[MlirOpId(10)]).unwrap_err();
        assert!(err.contains("no geodesic"), "{err}");
    }

    #[test]
    fn two_origins_and_a_backend_swap_do_not_change_the_next_id() {
        let (mut map, policy, catalog) = fixture("dense-wide", 52);
        let wide = map.base_weights.fingerprint;
        let narrow = Sha256Digest::compute(b"moe-narrow");
        let a = place(&mut map, &policy, &catalog, 10, vec![1u8; 6_656], wide);
        let b = place(&mut map, &policy, &catalog, 11, vec![2u8; 2_048], narrow);
        let frame = context(&catalog, &[MlirOpId(10)]).unwrap().1;
        link(&mut map, a, b, 11, frame, 1.2, 0.8, narrow);

        let swapped = FrozenBaseWeights::freeze("swapped-backend", 40, vec![0.2; 16]);
        let receipt = map.hot_swap_model_backend(Arc::clone(&swapped));
        assert_eq!(map.base_weights.num_layers, 40);
        assert!(receipt.preserved_global_atoms >= 2);

        let read = NativeReader::step(&map, &catalog, &[MlirOpId(10)]).unwrap();
        assert_eq!(read.next_id, MlirOpId(11));
        assert_eq!(read.origin_model_fingerprint, narrow);

        let other_catalog = MlirAtomCatalog::new_canonical(202_048);
        let again = NativeReader::step(&map, &other_catalog, &[MlirOpId(10)]).unwrap();
        assert_eq!(again.next_id, MlirOpId(11));
    }

    #[test]
    fn coupled_next_id_and_hidden_width_are_not_the_read() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = map.base_weights.fingerprint;
        let lie = coupled_lie(99_999, 6_656, 51);
        assert_eq!(
            CoupledOperatorAtomPayload::peek_next_mlir_id(&lie),
            Some(99_999)
        );
        let a = place(&mut map, &policy, &catalog, 10, lie, origin);
        let b = place(&mut map, &policy, &catalog, 11, vec![0u8; 2_048], origin);
        let frame = context(&catalog, &[MlirOpId(10)]).unwrap().1;
        link(&mut map, a, b, 11, frame, 1.3, 0.4, origin);

        let read = NativeReader::step(&map, &catalog, &[MlirOpId(10)]).unwrap();
        assert_eq!(read.next_id, MlirOpId(11));
    }

    #[test]
    fn empty_prefix_is_rejected() {
        let (map, _policy, catalog) = fixture("reader", 1);
        let err = NativeReader::step(&map, &catalog, &[]).unwrap_err();
        assert!(err.contains("at least one"));
    }

    fn bare_arc(from: AtomId, to: AtomId, op: u32, action: f64) -> DistilledReasoningArc {
        DistilledReasoningArc {
            arc_id: 0,
            chain_id: 1,
            step_index: 0,
            from_atom: from,
            to_atom: to,
            from_role: RoleId::LanguageSemiotics,
            to_role: RoleId::PredictiveSimulation,
            cosh_action_q32: Q32_31::from_f64(action),
            wigner_frame: IDENTITY_FRAME_Q32,
            compiled_mlir_op: MlirOpId(op),
            teacher_certainty_q32: Q32_31::from_f64(0.5),
            origin_model_fingerprint: Sha256Digest::compute(b"dedup"),
        }
    }

    #[test]
    fn trajectory_record_keeps_a_second_well_with_the_same_opcode() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = map.base_weights.fingerprint;
        let a = place(&mut map, &policy, &catalog, 10, b"a".to_vec(), origin);
        let b = place(&mut map, &policy, &catalog, 11, b"b".to_vec(), origin);
        let sink = place(&mut map, &policy, &catalog, 30, b"n".to_vec(), origin);
        let first = bare_arc(a, sink, 30, 1.2);
        let second = bare_arc(b, sink, 30, 1.2);
        let (id1, novel1) = map.record_novel_reasoning_arc(first);
        let (id2, novel2) = map.record_novel_reasoning_arc(second.clone());
        assert!(novel1);
        assert!(!novel2);
        assert_eq!(id1, id2);
        let (id3, novel3) = map.record_trajectory_arc(second);
        assert!(novel3);
        assert_ne!(id3, id1);
        assert_eq!(map.reasoning_arcs.len(), 2);
    }

    #[test]
    fn recorded_token_step_is_what_the_reader_returns() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = Sha256Digest::compute(b"glimmer-trajectory");
        let hidden = [1.25f32, -0.5, 0.0];
        let prefix = [MlirOpId(10), MlirOpId(20)];
        let wrote = NativeReader::record_step(
            &mut map,
            &catalog,
            &policy,
            &prefix,
            MlirOpId(30),
            &hidden,
            origin,
            Q32_31::from_f64(0.8),
        )
        .unwrap();
        assert!(wrote.arc_was_novel);
        assert_eq!(wrote.read.next_id, MlirOpId(30));
        assert_eq!(wrote.read.origin_model_fingerprint, origin);
        let from = map.global_atom(wrote.read.from_atom).unwrap();
        let (well, _) = NativeReader::address(&catalog, MlirOpId(20)).unwrap();
        assert_eq!(from.lorentz.x, well.x);
        assert_eq!(from.payload.as_ref(), hidden_payload(&hidden).as_slice());

        let again = NativeReader::record_step(
            &mut map,
            &catalog,
            &policy,
            &prefix,
            MlirOpId(30),
            &hidden,
            origin,
            Q32_31::from_f64(0.9),
        )
        .unwrap();
        assert!(!again.arc_was_novel);
        assert_eq!(map.reasoning_arcs.len(), 1);
        assert_eq!(
            map.reasoning_arcs[0].teacher_certainty_q32,
            Q32_31::from_f64(0.9)
        );
    }

    #[test]
    fn two_wells_that_share_a_next_id_both_play() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = Sha256Digest::compute(b"teacher");
        let cert = Q32_31::from_f64(0.7);
        NativeReader::record_step(
            &mut map,
            &catalog,
            &policy,
            &[MlirOpId(10)],
            MlirOpId(30),
            &[1.0],
            origin,
            cert,
        )
        .unwrap();
        NativeReader::record_step(
            &mut map,
            &catalog,
            &policy,
            &[MlirOpId(11)],
            MlirOpId(30),
            &[2.0],
            origin,
            cert,
        )
        .unwrap();
        assert_eq!(map.reasoning_arcs.len(), 2);
        let a = NativeReader::step(&map, &catalog, &[MlirOpId(10)]).unwrap();
        let b = NativeReader::step(&map, &catalog, &[MlirOpId(11)]).unwrap();
        assert_eq!(a.next_id, MlirOpId(30));
        assert_eq!(b.next_id, MlirOpId(30));
        assert_ne!(a.from_atom, b.from_atom);
    }

    #[test]
    fn two_histories_of_one_token_keep_both_next_ids() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = Sha256Digest::compute(b"teacher");
        let cert = Q32_31::ONE;
        let prefix_a = [MlirOpId(7), MlirOpId(10)];
        let prefix_b = [MlirOpId(8), MlirOpId(10)];
        let frame_a = context(&catalog, &prefix_a).unwrap().1;
        let frame_b = context(&catalog, &prefix_b).unwrap().1;
        assert_ne!(frame_a, frame_b);
        NativeReader::record_step(
            &mut map,
            &catalog,
            &policy,
            &prefix_a,
            MlirOpId(11),
            &[0.1],
            origin,
            cert,
        )
        .unwrap();
        NativeReader::record_step(
            &mut map,
            &catalog,
            &policy,
            &prefix_b,
            MlirOpId(12),
            &[0.2],
            origin,
            cert,
        )
        .unwrap();
        assert_eq!(
            NativeReader::step(&map, &catalog, &prefix_a)
                .unwrap()
                .next_id,
            MlirOpId(11)
        );
        assert_eq!(
            NativeReader::step(&map, &catalog, &prefix_b)
                .unwrap()
                .next_id,
            MlirOpId(12)
        );
    }

    #[test]
    fn a_recorded_continuation_replays() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = Sha256Digest::compute(b"teacher");
        let hidden = vec![vec![1.0f32], vec![3.0], vec![4.0]];
        NativeReader::record_continuation(
            &mut map,
            &catalog,
            &policy,
            &[MlirOpId(10)],
            &[MlirOpId(11), MlirOpId(12), MlirOpId(13)],
            &hidden,
            origin,
            Q32_31::ONE,
        )
        .unwrap();
        let walk = NativeReader::continue_ids(&map, &catalog, &[MlirOpId(10)], 5, &[]).unwrap();
        let ids: Vec<u32> = walk.reads.iter().map(|r| r.next_id.0).collect();
        assert_eq!(ids, vec![11, 12, 13]);
        let standing_on_11 = map.global_atom(walk.reads[0].to_atom).unwrap();
        assert_eq!(
            standing_on_11.payload.as_ref(),
            hidden_payload(&[3.0]).as_slice()
        );
        assert!(matches!(
            walk.stop,
            NativeStop::NoGeodesic { at, .. } if at == MlirOpId(13)
        ));
    }

    #[test]
    fn a_recorded_step_survives_the_vault() {
        let (mut map, policy, catalog) = fixture("traj", 1);
        let origin = Sha256Digest::compute(b"teacher");
        NativeReader::record_step(
            &mut map,
            &catalog,
            &policy,
            &[MlirOpId(10), MlirOpId(20)],
            MlirOpId(30),
            &[0.25, -1.5],
            origin,
            Q32_31::from_f64(0.8),
        )
        .unwrap();
        let path = std::env::temp_dir().join(format!("traj-{}-30.atoma_map", std::process::id()));
        map.save_to_binary_vault(&path).unwrap();
        let (mut restored, policy2, catalog2) = fixture("restored", 1);
        restored.load_from_binary_vault(&path, &policy2).unwrap();
        let _ = std::fs::remove_file(&path);
        let read = NativeReader::step(&restored, &catalog2, &[MlirOpId(10), MlirOpId(20)]).unwrap();
        assert_eq!(read.next_id, MlirOpId(30));
        assert_eq!(read.origin_model_fingerprint, origin);
    }
}
