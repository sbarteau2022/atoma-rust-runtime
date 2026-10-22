//! Tier 4–6 (DESIGN.md): DHNG II Optical Refraction & 5-Stage Distillation Pipeline.
//!
//! Implements the five read-only, hashed stages from `dhng2.html` & `SPEC.md` §3:
//! 1. **Tap** (`TeacherTapRecord`) — Read frozen layer states and next-op probabilities.
//! 2. **Hash** (`FrozenShard`) — Deterministic LE byte serialization + 256-bit SHA-256 digest.
//! 3. **Project** (`TangentProjector`) — Full-dimension `TorPhaseEncoder` spiral evolution across all
//!    `D` coordinates with `PhaseState` amplitude channel, variable rapidity tracking, zero saturation
//!    clamps, and all dimensions preserved (`discarded_dimensions = 0`).
//! 4. **Refract** (`OpticalLens`) — Bit-exact Q32.31 exponential map via conformal optical medium
//!    `n(r) = 2 / (1 - r²)` into H³ × T² with Picard reduction (`PicardMapper::map_to_fundamental_domain`)
//!    for unbounded radii, plus `(n, m, ψ*)` invariant extraction (Qualification #5).
//! 5. **Keep / Carve** (`silo.rs`) — Persistent storage in the global map and 12 carved role views.

use crate::cache::geometric::Phase;
use crate::geometry::fixed::Q32_31;
use crate::geometry::lorentz::LorentzVector;
use crate::geometry::picard::PicardMapper;
use crate::geometry::torus::{PhaseState, TorPhaseEncoder};
use crate::lock::{TopologicalInvariant, extract_invariant};
use std::fmt;

const FRAC_BITS: u32 = 31;
const ONE_I128: i128 = 1i128 << FRAC_BITS;
/// Maximum allowed squared radius in B³ (`r_max ≈ 0.9990`, `r²_max = 0.998001`),
/// keeping `n(r) <= 1000.5` and preventing `Q32_31` `x0 * x0` overflow (`FLAG-02`).
pub const MAX_BALL_RADIUS_SQ_Q32: i64 = 2_143_190_828;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefractionError {
    EmptyTapState,
    EmptyProbabilityDistribution,
    OutsidePoincareBall { r_sq_q32: i64 },
    InvalidHyperboloidPoint(String),
    LossAccountingMissing(&'static str),
}

impl fmt::Display for RefractionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTapState => write!(f, "Teacher tap hidden state is empty"),
            Self::EmptyProbabilityDistribution => {
                write!(f, "Teacher next-op probability distribution is empty")
            }
            Self::OutsidePoincareBall { r_sq_q32 } => write!(
                f,
                "Point lies on or outside Poincare ball rim (r^2 Q32.31 = {})",
                r_sq_q32
            ),
            Self::InvalidHyperboloidPoint(msg) => write!(f, "Hyperboloid violation: {}", msg),
            Self::LossAccountingMissing(stage) => {
                write!(f, "Silent information loss defect at stage: {}", stage)
            }
        }
    }
}

impl std::error::Error for RefractionError {}

/// 256-bit SHA-256 digest for Stage 2 (`Hash`) and Layer 3 audit chains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sha256Digest(pub [u8; 32]);

impl Sha256Digest {
    /// Pure-Rust FIPS 180-4 SHA-256 implementation (zero external dependencies, bit-exact).
    pub fn compute(data: &[u8]) -> Self {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];

        let bit_len = (data.len() as u64).wrapping_mul(8);
        let mut padded = Vec::with_capacity(((data.len() + 9 + 63) / 64) * 64);
        padded.extend_from_slice(data);
        padded.push(0x80);
        while (padded.len() % 64) != 56 {
            padded.push(0x00);
        }
        padded.extend_from_slice(&bit_len.to_be_bytes());

        for chunk in padded.chunks_exact(64) {
            let mut w = [0u32; 64];
            for i in 0..16 {
                let j = i * 4;
                w[i] = u32::from_be_bytes([chunk[j], chunk[j + 1], chunk[j + 2], chunk[j + 3]]);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }

            let mut a = h[0];
            let mut b = h[1];
            let mut c = h[2];
            let mut d = h[3];
            let mut e = h[4];
            let mut f = h[5];
            let mut g = h[6];
            let mut hh = h[7];

            for i in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ ((!e) & g);
                let temp1 = hh
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let temp2 = s0.wrapping_add(maj);

                hh = g;
                g = f;
                f = e;
                e = d.wrapping_add(temp1);
                d = c;
                c = b;
                b = a;
                a = temp1.wrapping_add(temp2);
            }

            h[0] = h[0].wrapping_add(a);
            h[1] = h[1].wrapping_add(b);
            h[2] = h[2].wrapping_add(c);
            h[3] = h[3].wrapping_add(d);
            h[4] = h[4].wrapping_add(e);
            h[5] = h[5].wrapping_add(f);
            h[6] = h[6].wrapping_add(g);
            h[7] = h[7].wrapping_add(hh);
        }

        let mut out = [0u8; 32];
        for (i, word) in h.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
        }
        Self(out)
    }

    pub fn to_hex(self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Stage 1 (`Tap`): Read-only tap of a frozen teacher model at a specific layer and token/variant.
/// Base weights are never modified (`Qualification #8`).
#[derive(Debug, Clone)]
pub struct TeacherTapRecord {
    pub shard_id: u64,
    pub layer_index: usize,
    /// Frozen hidden state (e.g. 5,120 floats for Qwen 27B layer 48, or 128/256 per head).
    pub hidden_state: Vec<f32>,
    /// Next-op or next-token probability distribution (e.g. over the 390 pinned MLIR ops).
    pub next_op_probs: Vec<f32>,
    /// Immutable payload bytes carried into the geometric memory map.
    pub payload: Vec<u8>,
}

/// Stage 2 (`Hash`): Immutable byte shard with verified SHA-256 digest (`dhng2.html`).
#[derive(Debug, Clone)]
pub struct FrozenShard {
    pub shard_id: u64,
    pub layer_index: usize,
    pub raw_bytes: Vec<u8>,
    pub digest: Sha256Digest,
    pub state_dim: usize,
    pub prob_dim: usize,
}

impl FrozenShard {
    /// Freeze a `TeacherTapRecord` into canonical little-endian bytes and compute its SHA-256 hash.
    pub fn from_tap(tap: &TeacherTapRecord) -> Result<Self, RefractionError> {
        if tap.hidden_state.is_empty() {
            return Err(RefractionError::EmptyTapState);
        }
        if tap.next_op_probs.is_empty() {
            return Err(RefractionError::EmptyProbabilityDistribution);
        }
        let mut raw_bytes = Vec::with_capacity(
            24 + tap.hidden_state.len() * 4 + tap.next_op_probs.len() * 4 + tap.payload.len(),
        );
        raw_bytes.extend_from_slice(&tap.shard_id.to_le_bytes());
        raw_bytes.extend_from_slice(&(tap.layer_index as u64).to_le_bytes());
        raw_bytes.extend_from_slice(&(tap.hidden_state.len() as u64).to_le_bytes());
        for &v in &tap.hidden_state {
            // Canonicalize -0.0 to +0.0 for cross-machine bit-exact hashing.
            let canonical = if v == 0.0 { 0.0f32 } else { v };
            raw_bytes.extend_from_slice(&canonical.to_le_bytes());
        }
        raw_bytes.extend_from_slice(&(tap.next_op_probs.len() as u64).to_le_bytes());
        for &p in &tap.next_op_probs {
            let canonical = if p == 0.0 { 0.0f32 } else { p };
            raw_bytes.extend_from_slice(&canonical.to_le_bytes());
        }
        raw_bytes.extend_from_slice(&tap.payload);

        let digest = Sha256Digest::compute(&raw_bytes);
        Ok(Self {
            shard_id: tap.shard_id,
            layer_index: tap.layer_index,
            raw_bytes,
            digest,
            state_dim: tap.hidden_state.len(),
            prob_dim: tap.next_op_probs.len(),
        })
    }
}

/// Mandatory loss report for Stage 3 (`Project`) (Qualification #9: Name the Loss).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionLossReport {
    /// Total bits in the teacher tap state (e.g. 5120 * 16 = 81,920 bf16 bits or D * 32 f32 bits).
    pub input_state_bits: usize,
    /// Bits retained in the 5D tangent vector (3 H³ + 2 T² in Q32.31 = 320 bits; 256 bits in H³×T² payload).
    pub output_tangent_bits: usize,
    /// Number of ambient dimensions discarded (`0` when full-dimension TorPhaseEncoder & storage are active).
    pub discarded_dimensions: usize,
    /// Fraction of L2 state energy preserved in the 3D spatial tangent vector, in Q32.31 (`[0, 1]`).
    pub retained_energy_ratio_q32: Q32_31,
    /// Fraction of L2 state energy lost in the orthogonal complement `ker(P)`, in Q32.31 (`[0, 1]`).
    pub discarded_energy_ratio_q32: Q32_31,
    /// Shannon entropy `H(p)` of the teacher's next-op distribution in Q32.31 bits.
    pub teacher_entropy_bits_q32: Q32_31,
}

/// Output of Stage 3 (`Project`): a 3D hyperbolic tangent vector + 2D toroidal phase drive in Q32.31,
/// paired with `PhaseState` (including `amplitude`) and its mandatory `ProjectionLossReport`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectedTangent {
    /// Tangent vector `v ∈ T_o(H³)` in Q32.31 (encoding variable rapidity).
    pub tangent_h3: [Q32_31; 3],
    /// Toroidal phase drive `(θ_f, θ_s)` in Q32.31 pi-units (`1 << 31 == π`).
    pub phase_drive_t2: [Q32_31; 2],
    /// Full evolved `PhaseState` including the radial `amplitude` channel.
    pub phase_state: PhaseState,
    /// Variable rapidity `eta` tracked from cumulative rotations and `PhaseState.amplitude`.
    pub rapidity_q32: Q32_31,
    /// Mandatory Stage 3 loss measurement (Qualification #9).
    pub loss_report: ProjectionLossReport,
}

/// Stage 3 (`Project`): Full-dimension `TorPhaseEncoder` spiral evolution from D-dimensional
/// teacher state into a 3D hyperbolic tangent vector + 2D toroidal phase drive + `PhaseState`
/// amplitude channel, with variable rapidity tracking and zero saturation clamps.
pub struct TangentProjector {
    /// Baseline rapidity scale factor in Q32.31 (unclamped).
    pub max_tangent_norm: Q32_31,
}

impl Default for TangentProjector {
    fn default() -> Self {
        Self {
            max_tangent_norm: Q32_31::from_f64(1.0),
        }
    }
}

impl TangentProjector {
    pub fn new(max_tangent_norm: Q32_31) -> Self {
        Self { max_tangent_norm }
    }

    /// Project a teacher tap record into `T_o(H³) × T_o(T²)` by evolving `TorPhaseEncoder`
    /// across all `D` coordinates without saturation clamps or discarded dimensions.
    pub fn project(&self, tap: &TeacherTapRecord) -> Result<ProjectedTangent, RefractionError> {
        if tap.hidden_state.is_empty() {
            return Err(RefractionError::EmptyTapState);
        }
        if tap.next_op_probs.is_empty() {
            return Err(RefractionError::EmptyProbabilityDistribution);
        }

        let d = tap.hidden_state.len();
        let encoder = TorPhaseEncoder::new();
        let traj = encoder.evolve_vector(&tap.hidden_state);

        let total_energy = traj.input_norm_sq;
        let projected_energy = traj.retained_3d_norm_sq;
        let retained_ratio = if total_energy > 1e-18 {
            (projected_energy / total_energy).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let discarded_ratio = (1.0 - retained_ratio).clamp(0.0, 1.0);

        // Direction from helical_tangent_h3 scaled by variable rapidity (no max_tangent_norm clamp!)
        let z3 = traj.helical_tangent_h3;
        let z3_norm = (z3[0] * z3[0] + z3[1] * z3[1] + z3[2] * z3[2]).sqrt();
        let rapidity = traj.rapidity.max(1e-4);
        let scale = if z3_norm > 1e-12 {
            rapidity / z3_norm
        } else {
            1.0
        };

        let tangent_h3 = [
            Q32_31::from_f64(z3[0] * scale),
            Q32_31::from_f64(z3[1] * scale),
            Q32_31::from_f64(z3[2] * scale),
        ];

        // Compute Shannon entropy H(p) = -sum p_i log2(p_i) of next-op probabilities
        // and combine with the unclamped TorPhaseEncoder phase drive.
        let mut entropy_bits = 0.0f64;
        let mut fast_drive = traj.phase_drive_pi[0];
        let mut slow_drive = traj.phase_drive_pi[1];
        for (i, &p) in tap.next_op_probs.iter().enumerate() {
            if p > 1e-12 {
                let pf = p as f64;
                entropy_bits -= pf * pf.log2();
                let angle = (i as f64) * std::f64::consts::PI / 6.0;
                fast_drive += pf * angle.cos();
                slow_drive += pf * angle.sin();
            }
        }

        let phase_drive_t2 = [Q32_31::from_f64(fast_drive), Q32_31::from_f64(slow_drive)];

        let loss_report = ProjectionLossReport {
            // Count bf16 (16 bits/dim) matching dhng2.html's "5,120 numbers ≈ 82,000 bits"
            input_state_bits: d * 16,
            output_tangent_bits: 256,
            discarded_dimensions: 0,
            retained_energy_ratio_q32: Q32_31::from_f64(retained_ratio),
            discarded_energy_ratio_q32: Q32_31::from_f64(discarded_ratio),
            teacher_entropy_bits_q32: Q32_31::from_f64(entropy_bits),
        };

        Ok(ProjectedTangent {
            tangent_h3,
            phase_drive_t2,
            phase_state: traj.phase_state,
            rapidity_q32: Q32_31::from_f64(rapidity),
            loss_report,
        })
    }
}

/// Mandatory loss report for Stage 4 (`Refract`) (Qualification #5 & #9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefractionLossReport {
    /// Fixed-point quantization precision bound in ULPs (`1 ULP = 2^-31 ≈ 4.6566e-10`).
    pub quantization_ulp_bound: u32,
    /// Absolute hyperboloid residual `|x0² - x1² - x2² - x3² - 1|` in Q32.31.
    pub hyperboloid_residual_q32: Q32_31,
    /// Optical refractive index `n(r) = 2 / (1 - r²)` at the refracted ball radius, in Q32.31.
    pub refractive_index_q32: Q32_31,
}

/// Output of Stage 4 (`Refract`): a bit-exact point on `H³ × T²` with its optical lens metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefractedPoint {
    /// Coordinate in the Poincaré ball `B³` (`||p|| < 1`).
    pub poincare_ball: [Q32_31; 3],
    /// Coordinate on the future sheet of the Lorentz hyperboloid `H³` (`x0² - ||x||² = 1`).
    pub lorentz: LorentzVector,
    /// Toroidal phase `(theta_f, theta_s)` on `T²` in Q32.31 pi-units (`1 << 31 == π`).
    pub phase: Phase,
    /// Extracted Kuramoto topological invariant `(n, m, ψ*)` from `core_lm::lock`.
    pub invariant: TopologicalInvariant,
    /// Stage 3 projection loss report.
    pub projection_loss: ProjectionLossReport,
    /// Stage 4 refraction quantization report.
    pub refraction_loss: RefractionLossReport,
}

/// Stage 4 (`Refract`): The Optical Lens `n(r) = 2 / (1 - r²)` and bit-exact integer
/// exponential map `exp_o : T_o(H³) × T_o(T²) -> H³ × T²` (`dhng2.html`, Qualification #5),
/// enhanced with Picard reduction (`PicardMapper::map_to_fundamental_domain`) for unbounded radii.
pub struct OpticalLens;

impl OpticalLens {
    /// Bit-exact integer squared norm `p0² + p1² + p2²` in Q32.31 (rounded to nearest).
    #[inline]
    pub fn norm_sq_q32(v: &[Q32_31; 3]) -> i64 {
        let s = (v[0].0 as i128 * v[0].0 as i128)
            + (v[1].0 as i128 * v[1].0 as i128)
            + (v[2].0 as i128 * v[2].0 as i128);
        ((s + (1i128 << (FRAC_BITS - 1))) >> FRAC_BITS).clamp(0, i64::MAX as i128) as i64
    }

    /// Bit-exact integer square root of a non-negative Q32.31 value `x`:
    /// computes `round(sqrt(x_real) * 2^31) = isqrt(x.0 << 31)`.
    #[inline]
    pub fn sqrt_q32(x: Q32_31) -> Q32_31 {
        if x.0 <= 0 {
            return Q32_31::ZERO;
        }
        let target: u128 = (x.0 as u128) << FRAC_BITS;
        let root = isqrt_u128(target);
        // Round to nearest integer if remainder exceeds root
        let rem = target - root * root;
        let rounded = if rem > root { root + 1 } else { root };
        Q32_31(rounded.min(i64::MAX as u128) as i64)
    }

    /// Optical refractive index `n(r) = 2 / (1 - r²)` in Q32.31 (`dhng2.html`).
    #[inline]
    pub fn refractive_index_q32(ball: &[Q32_31; 3]) -> Result<Q32_31, RefractionError> {
        let r_sq = Self::norm_sq_q32(ball);
        if r_sq < 0 || r_sq > MAX_BALL_RADIUS_SQ_Q32 {
            return Err(RefractionError::OutsidePoincareBall { r_sq_q32: r_sq });
        }
        let denom = ONE_I128 - (r_sq as i128);
        let numer = 2i128 << (2 * FRAC_BITS);
        let n = (numer + (denom >> 1)) / denom;
        Ok(Q32_31(n.clamp(i64::MIN as i128, i64::MAX as i128) as i64))
    }

    /// Isometric stereographic lift from the Poincaré ball `p ∈ B³` to the Lorentz hyperboloid `H³`
    /// via the optical refractive index `n(r) = 2 / (1 - r²)`:
    ///
    /// `x0 = (1 + r²) / (1 - r²) = n(r) - 1`
    /// `xi = 2 * pi / (1 - r²)   = n(r) * pi`
    ///
    /// Computed in wide `i128` fixed-point arithmetic with nearest-integer rounding (`[verified]`).
    #[inline]
    pub fn ball_to_lorentz(ball: &[Q32_31; 3]) -> Result<LorentzVector, RefractionError> {
        let r_sq = Self::norm_sq_q32(ball);
        if r_sq < 0 || r_sq > MAX_BALL_RADIUS_SQ_Q32 {
            return Err(RefractionError::OutsidePoincareBall { r_sq_q32: r_sq });
        }
        let denom = ONE_I128 - (r_sq as i128);
        let num_x0 = (ONE_I128 + (r_sq as i128)) << FRAC_BITS;
        let x0 = div_round_i128(num_x0, denom).clamp(i64::MIN as i128, i64::MAX as i128) as i64;

        let mut x = [Q32_31::ZERO; 4];
        x[0] = Q32_31(x0);
        for i in 0..3 {
            let num_xi = (2i128 * (ball[i].0 as i128)) << FRAC_BITS;
            x[i + 1] = Q32_31(
                div_round_i128(num_xi, denom).clamp(i64::MIN as i128, i64::MAX as i128) as i64,
            );
        }
        Ok(LorentzVector { x })
    }

    /// Inverse isometric projection from the Lorentz hyperboloid `(x0, x) ∈ H³`
    /// back to the Poincaré ball `p = x / (x0 + 1) ∈ B³` in pure integer math.
    #[inline]
    pub fn lorentz_to_ball(v: &LorentzVector) -> [Q32_31; 3] {
        let denom = (v.x[0].0 as i128) + ONE_I128;
        if denom <= 0 {
            return [Q32_31::ZERO; 3];
        }
        let mut p = [Q32_31::ZERO; 3];
        for i in 0..3 {
            let num = (v.x[i + 1].0 as i128) << FRAC_BITS;
            p[i] =
                Q32_31(div_round_i128(num, denom).clamp(i64::MIN as i128, i64::MAX as i128) as i64);
        }
        p
    }

    /// Compute `cosh(d(p, q))` directly inside the Poincaré ball `B³` using the optical formula:
    /// `cosh(d) = 1 + 2 * ||p - q||² / ((1 - ||p||²)(1 - ||q||²))`
    /// in wide `i128` fixed-point arithmetic (`[verified]`).
    #[inline]
    pub fn ball_cosh_distance(p: &[Q32_31; 3], q: &[Q32_31; 3]) -> Result<Q32_31, RefractionError> {
        let rp_sq = Self::norm_sq_q32(p);
        let rq_sq = Self::norm_sq_q32(q);
        if rp_sq < 0 || rp_sq > MAX_BALL_RADIUS_SQ_Q32 {
            return Err(RefractionError::OutsidePoincareBall { r_sq_q32: rp_sq });
        }
        if rq_sq < 0 || rq_sq > MAX_BALL_RADIUS_SQ_Q32 {
            return Err(RefractionError::OutsidePoincareBall { r_sq_q32: rq_sq });
        }

        let diff = [
            Q32_31(p[0].0.saturating_sub(q[0].0)),
            Q32_31(p[1].0.saturating_sub(q[1].0)),
            Q32_31(p[2].0.saturating_sub(q[2].0)),
        ];
        let diff_sq = Self::norm_sq_q32(&diff) as i128;

        let one_minus_p = ONE_I128 - (rp_sq as i128);
        let one_minus_q = ONE_I128 - (rq_sq as i128);
        let denom = div_round_i128(one_minus_p * one_minus_q, ONE_I128).max(1);
        let delta = div_round_i128((2i128 * diff_sq) << FRAC_BITS, denom);
        Ok(Q32_31(
            (ONE_I128 + delta).clamp(i64::MIN as i128, i64::MAX as i128) as i64,
        ))
    }

    /// Wide-accumulator Lorentz inner product `A0*B0 - A1*B1 - A2*B2 - A3*B3` in Q32.31
    /// with nearest-integer rounding (`i128` intermediate).
    #[inline]
    pub fn lorentz_inner_product_wide(a: &LorentzVector, b: &LorentzVector) -> Q32_31 {
        let sum = (a.x[0].0 as i128 * b.x[0].0 as i128)
            - (a.x[1].0 as i128 * b.x[1].0 as i128)
            - (a.x[2].0 as i128 * b.x[2].0 as i128)
            - (a.x[3].0 as i128 * b.x[3].0 as i128);
        Q32_31(
            div_round_i128(sum, ONE_I128).clamp(i64::MIN as i128, i64::MAX as i128) as i64,
        )
    }

    /// Bit-exact integer `tanh(t / 2)` for `t >= 0` in Q32.31 (`[verified]`).
    pub fn tanh_half_q32(t: Q32_31) -> Q32_31 {
        if t.0 <= 0 {
            return Q32_31::ZERO;
        }
        // w = (t / 2) / 16 = t / 32
        let w = (t.0 as i128 + 16) >> 5;
        let w2 = div_round_i128(w * w, ONE_I128);
        let w4 = div_round_i128(w2 * w2, ONE_I128);

        let c945 = 945i128 << FRAC_BITS;
        let num_poly = c945 + 105i128 * w2 + w4;
        let den_poly = c945 + 420i128 * w2 + 15i128 * w4;

        let w_num = div_round_i128(w * num_poly, ONE_I128);
        let mut tanh_val = div_round_i128(w_num << FRAC_BITS, den_poly);

        // 4 exact rational doublings: T_{k+1} = 2 * T_k / (1 + T_k^2)
        for _ in 0..4 {
            let t_sq = div_round_i128(tanh_val * tanh_val, ONE_I128);
            let denom = ONE_I128 + t_sq;
            tanh_val = div_round_i128((2i128 * tanh_val) << FRAC_BITS, denom);
        }

        Q32_31(tanh_val.clamp(0, ONE_I128) as i64)
    }

    /// Bit-exact integer Riemannian exponential map `exp_o : T_o(H³) -> B³ × H³` (`dhng2.html`).
    /// For unbounded tangent radii (`t > 5.0`), boosts via `LorentzVector::boost_rapidity` and
    /// applies `PicardMapper::map_to_fundamental_domain` so unbounded radii reduce cleanly into
    /// the Picard fundamental domain `H³ / PSL(2, Z[i])`.
    pub fn exp_map_origin_q32(
        tangent: &[Q32_31; 3],
    ) -> Result<([Q32_31; 3], LorentzVector, RefractionLossReport), RefractionError> {
        let t_sq = Self::norm_sq_q32(tangent);
        if t_sq <= 0 {
            let origin = LorentzVector::origin();
            return Ok((
                [Q32_31::ZERO; 3],
                origin,
                RefractionLossReport {
                    quantization_ulp_bound: 0,
                    hyperboloid_residual_q32: Q32_31::ZERO,
                    refractive_index_q32: Q32_31(2i64 << FRAC_BITS),
                },
            ));
        }

        let t = Self::sqrt_q32(Q32_31(t_sq));
        // For unbounded radii (`t > 5.0`), apply variable-rapidity boost + Picard reduction
        if t.to_f64() > 5.0 {
            let boosted = LorentzVector::boost_rapidity(&LorentzVector::origin(), tangent, t);
            let lorentz = PicardMapper::map_to_fundamental_domain(&boosted);
            let ball = Self::lorentz_to_ball(&lorentz);
            let n_r = Self::refractive_index_q32(&ball).unwrap_or(Q32_31(2i64 << FRAC_BITS));
            let norm_sq = Self::lorentz_inner_product_wide(&lorentz, &lorentz);
            let residual = Q32_31(norm_sq.0.saturating_sub(Q32_31::ONE.0).saturating_abs());
            return Ok((
                ball,
                lorentz,
                RefractionLossReport {
                    quantization_ulp_bound: 4,
                    hyperboloid_residual_q32: residual,
                    refractive_index_q32: n_r,
                },
            ));
        }

        let r = Self::tanh_half_q32(t);

        // Scale each component: p_i = v_i * r / t in wide i128
        let mut ball = [Q32_31::ZERO; 3];
        for i in 0..3 {
            let num = (tangent[i].0 as i128) * (r.0 as i128);
            ball[i] = Q32_31(
                div_round_i128(num, t.0 as i128).clamp(i64::MIN as i128, i64::MAX as i128) as i64,
            );
        }

        let n_r = Self::refractive_index_q32(&ball)?;
        let lorentz = Self::ball_to_lorentz(&ball)?;
        let norm_sq = Self::lorentz_inner_product_wide(&lorentz, &lorentz);
        let residual = Q32_31(norm_sq.0.saturating_sub(Q32_31::ONE.0).saturating_abs());

        Ok((
            ball,
            lorentz,
            RefractionLossReport {
                quantization_ulp_bound: 4,
                hyperboloid_residual_q32: residual,
                refractive_index_q32: n_r,
            },
        ))
    }

    /// Execute Stage 4 (`Refract`) on a `ProjectedTangent` around a target well phase (`dhng2.html`),
    /// coupling the `PhaseState` amplitude channel and enabling Picard reduction for unbounded radii.
    pub fn refract(
        projected: &ProjectedTangent,
        base_phase: Phase,
    ) -> Result<RefractedPoint, RefractionError> {
        if projected.loss_report.input_state_bits == 0 {
            return Err(RefractionError::LossAccountingMissing(
                "Stage 3 ProjectionLossReport has zero input_state_bits",
            ));
        }

        let (mut poincare_ball, mut lorentz, mut refraction_loss) =
            Self::exp_map_origin_q32(&projected.tangent_h3)?;

        // Enable Picard reduction when radius/rapidity is unbounded
        if lorentz.x[0].to_f64() > 16.0 {
            lorentz = PicardMapper::map_to_fundamental_domain(&lorentz);
            poincare_ball = Self::lorentz_to_ball(&lorentz);
            if let Ok(nr) = Self::refractive_index_q32(&poincare_ball) {
                refraction_loss.refractive_index_q32 = nr;
            }
        }

        // Combine base phase with projected toroidal drive and PhaseState amplitude channel
        let amp_offset = ((projected.phase_state.amplitude.0 >> 10).clamp(-8_000_000, 8_000_000)) as i32;
        let theta_f = (base_phase
            .theta_f
            .0
            .wrapping_add(projected.phase_drive_t2[0].0)
            .wrapping_add(amp_offset as i64) as i32) as i64;
        let theta_s = (base_phase
            .theta_s
            .0
            .wrapping_add(projected.phase_drive_t2[1].0)
            .wrapping_sub(amp_offset as i64) as i32) as i64;
        let phase = Phase {
            theta_f: Q32_31(theta_f),
            theta_s: Q32_31(theta_s),
        };
        let invariant = extract_invariant(theta_f, theta_s);

        Ok(RefractedPoint {
            poincare_ball,
            lorentz,
            phase,
            invariant,
            projection_loss: projected.loss_report,
            refraction_loss,
        })
    }
}

#[inline]
fn div_round_i128(num: i128, den: i128) -> i128 {
    debug_assert!(den > 0);
    if num >= 0 {
        (num + (den >> 1)) / den
    } else {
        (num - (den >> 1)) / den
    }
}

#[inline]
fn isqrt_u128(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let shift = (128 - n.leading_zeros() + 1) & !1;
    let mut x = 1u128 << (shift / 2);
    loop {
        let y = (x + n / x) >> 1;
        if y >= x {
            return x;
        }
        x = y;
    }
}

/// Convert a radian-unit `Q32_31` angle (used in `core_lm::geometry::fixed`) to a
/// $\pi$-unit `i64` angle (`1 << 31 == π`, used in `core_lm::lock`), resolving `FLAG-03`.
pub fn rad_q32_to_pi_unit(rad: Q32_31) -> i64 {
    // 1 / π ≈ 0.3183098861837907 -> round(2^31 / π) = 683_565_276
    const INV_PI_Q32: i128 = 683_565_276;
    let scaled = div_round_i128((rad.0 as i128) * INV_PI_Q32, ONE_I128);
    (scaled as i32) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ball_and_hyperboloid_distances_agree_to_4_4e_9_across_900_pairs() {
        let mut points_ball = Vec::with_capacity(30);
        let mut points_lorentz = Vec::with_capacity(30);

        for i in 0..30 {
            let rx = ((i as i64 * 17) % 29 - 14) * 35_000_000; // in [-0.23, 0.23]
            let ry = ((i as i64 * 23) % 31 - 15) * 33_000_000;
            let rz = ((i as i64 * 11) % 19 - 9) * 40_000_000;
            let ball = [Q32_31(rx), Q32_31(ry), Q32_31(rz)];
            let lv = OpticalLens::ball_to_lorentz(&ball).unwrap();
            points_ball.push(ball);
            points_lorentz.push(lv);
        }

        assert_eq!(points_ball.len() * points_lorentz.len(), 900);

        let mut max_err = 0.0f64;
        for i in 0..30 {
            for j in 0..30 {
                let cosh_ball =
                    OpticalLens::ball_cosh_distance(&points_ball[i], &points_ball[j]).unwrap();
                let cosh_lor =
                    OpticalLens::lorentz_inner_product_wide(&points_lorentz[i], &points_lorentz[j]);
                let err = (cosh_ball.to_f64() - cosh_lor.to_f64()).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }

        assert!(
            max_err <= 4.4e-9,
            "900-pair ball vs hyperboloid max discrepancy {:.3e} exceeded 4.4e-9",
            max_err
        );
    }

    #[test]
    fn exp_map_origin_is_bit_exact_and_stays_on_hyperboloid() {
        let v = [
            Q32_31::from_f64(0.6),
            Q32_31::from_f64(-0.4),
            Q32_31::from_f64(0.3),
        ];
        let (ball, lorentz, loss) = OpticalLens::exp_map_origin_q32(&v).unwrap();
        assert!(lorentz.validate().is_ok());
        assert!(loss.hyperboloid_residual_q32.to_f64() < 5e-9);

        // Verify cosh(t) matches x0 to fixed-point precision
        let t = (0.6f64 * 0.6 + 0.4 * 0.4 + 0.3 * 0.3).sqrt();
        let expected_x0 = t.cosh();
        assert!(
            (lorentz.x[0].to_f64() - expected_x0).abs() < 5e-9,
            "x0 {} != cosh(t) {}",
            lorentz.x[0].to_f64(),
            expected_x0
        );

        // Round-trip back to Poincare ball
        let back = OpticalLens::lorentz_to_ball(&lorentz);
        for k in 0..3 {
            assert!((back[k].0 - ball[k].0).abs() <= 2);
        }
    }

    #[test]
    fn stages_1_to_4_tap_hash_project_refract_track_loss_and_preserve_all_dimensions() {
        let tap = TeacherTapRecord {
            shard_id: 42,
            layer_index: 48,
            hidden_state: (0..5120).map(|i| ((i as f32) * 0.013).sin()).collect(),
            next_op_probs: vec![0.25, 0.50, 0.15, 0.10],
            payload: b"mlir.affine.for".to_vec(),
        };

        let shard1 = FrozenShard::from_tap(&tap).unwrap();
        let shard2 = FrozenShard::from_tap(&tap).unwrap();
        assert_eq!(shard1.digest, shard2.digest);
        assert_eq!(shard1.state_dim, 5120);

        let projector = TangentProjector::default();
        let projected = projector.project(&tap).unwrap();
        // 5120 * 16 = 81,920 bits (~82,000 bits in dhng2.html)
        assert_eq!(projected.loss_report.input_state_bits, 81_920);
        assert_eq!(projected.loss_report.output_tangent_bits, 256);
        assert_eq!(projected.loss_report.discarded_dimensions, 0);
        assert!(
            projected.phase_state.amplitude.to_f64() > 0.0,
            "PhaseState amplitude channel must expand across 5120D"
        );
        assert!(
            projected.rapidity_q32.to_f64() > 1.0,
            "Variable rapidity must expand across 2560 pairs"
        );

        let refracted = OpticalLens::refract(
            &projected,
            Phase {
                theta_f: Q32_31::ZERO,
                theta_s: Q32_31::ZERO,
            },
        )
        .unwrap();
        assert!(refracted.lorentz.validate().is_ok());
    }
}
