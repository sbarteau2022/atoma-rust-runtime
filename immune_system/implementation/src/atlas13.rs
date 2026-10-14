//! Tier 4–6 (DESIGN.md): DHNG III "Twelve Around One" Manifold & Phase Geometry.
//!
//! Why this geometry exists (Qualification #3 & #6):
//! 1. In full teacher embedding space, the 6 cognitive duality differences (A - B)
//!    are nearly orthogonal (mean |cos| = 0.079, max = 0.177 [reported: F17]),
//!    forming a 6D cross-polytope (±e₁ … ±e₆, 12 poles around 1 coordinator).
//! 2. In 3D hyperbolic tangent space T_o(H³) ≅ ℝ³, 6 mutually orthogonal axes cannot
//!    exist. By the Welch/Haantjes bound for 6 equiangular lines in ℝ³, the minimum
//!    achievable maximum pairwise |cos θ| is 1/√5 ≈ 0.4472136, attained uniquely by
//!    the 6 five-fold axes (12 vertices) of the regular icosahedron [verified].
//! 3. Both the approved icosahedral frame (`FrameGeometry::Icosahedral`) and the legacy
//!    cuboctahedral frame (`FrameGeometry::Cuboctahedral` from `atlas13.json`) are
//!    provided in bit-exact Q32.31 fixed-point coordinates at Poincaré ball shell
//!    radius r_sph = 0.382 [verified].

use crate::cache::geometric::Phase;
use crate::geometry::fixed::Q32_31;
use crate::geometry::lorentz::LorentzVector;
use crate::lock::{TopologicalInvariant, extract_invariant};

/// Pi in Q32.31 pi-units (matching `crate::lock::PI = 1 << 31`).
pub const PI_UNIT: i64 = 1i64 << 31;

/// Golden ratio φ = (1 + √5) / 2 in Q32.31: round(1.618033988749895 * 2^31).
pub const PHI_Q32: Q32_31 = Q32_31(3_474_701_521);

/// 1 / √5 in Q32.31: round(0.447213595499957939 * 2^31) = 960_383_883 — exact icosahedral axis cosine.
pub const INV_SQRT5_Q32: Q32_31 = Q32_31(960_383_883);

/// Approved Poincaré ball shell radius for the 12 spheres: r_sph = 0.382 (F16).
/// round(0.382 * 2^31) = 820_338_754.
pub const R_SPH_Q32: Q32_31 = Q32_31(820_338_754);

/// Measured radial depth shells in the Poincaré ball (`dhng3.html`, F16 [reported]).
pub mod measured_shells {
    use crate::geometry::fixed::Q32_31;

    /// Tasks shell: r = 0.36 (1,227 nodes [reported]).
    pub const TASKS_R: Q32_31 = Q32_31(773_094_113);
    /// Ghost membrane shell: r = 0.444 ([reported]).
    pub const GHOST_MEMBRANE_R: Q32_31 = Q32_31(953_482_740);
    /// Results shell: r = 0.62 (1,335 nodes [reported]).
    pub const RESULTS_R: Q32_31 = Q32_31(1_331_439_862);
    /// Sources shell: r = 0.80 (41 nodes [reported]).
    pub const SOURCES_R: Q32_31 = Q32_31(1_717_986_918);
}

/// Epistemic mark required on every system claim (Qualification #1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicMark {
    /// Measured or mathematically proved on this machine.
    Verified,
    /// Measured and reported in reference artifacts (`dhng2.html`, `dhng3.html`, `SESSION_NOTES.md`).
    Reported,
    /// Follows logically from verified or reported evidence.
    Inferred,
    /// Unproven architectural hypothesis subject to preregistered falsification.
    Theory,
}

impl EpistemicMark {
    pub const fn as_tag(self) -> &'static str {
        match self {
            Self::Verified => "[verified]",
            Self::Reported => "[reported]",
            Self::Inferred => "[inferred]",
            Self::Theory => "[theory]",
        }
    }
}

/// The 6 independent cognitive duality axes (`dhng3.html`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DualityAxis {
    /// Axis 0 (u = 0°): Sensory Perception (L1) ↔ Motor Praxis (L2).
    PerceiveAct = 0,
    /// Axis 1 (u = 30°): Somatic / Interoception (L3) ↔ Language & Semiotics (L4).
    BodySymbol = 1,
    /// Axis 2 (u = 60°): Episodic Memory (L5) ↔ Predictive Simulation (L6).
    RememberPredict = 2,
    /// Axis 3 (u = 90°): Convergent Logic (L7) ↔ Divergent Synthesis (L8).
    ConvergeDiverge = 3,
    /// Axis 4 (u = 120°): Valuation & Salience (L9) ↔ Social Cognition (L10).
    ValueSocial = 4,
    /// Axis 5 (u = 150°): Spatial Topography (L11) ↔ Metacognitive Regulation (L12).
    SpaceSelfMonitor = 5,
}

impl DualityAxis {
    pub const ALL: [Self; 6] = [
        Self::PerceiveAct,
        Self::BodySymbol,
        Self::RememberPredict,
        Self::ConvergeDiverge,
        Self::ValueSocial,
        Self::SpaceSelfMonitor,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::PerceiveAct => "perceive <-> act",
            Self::BodySymbol => "body <-> symbol",
            Self::RememberPredict => "remember <-> predict",
            Self::ConvergeDiverge => "converge <-> diverge",
            Self::ValueSocial => "value <-> social",
            Self::SpaceSelfMonitor => "space <-> self-monitor",
        }
    }

    /// Toroidal longitude u_k = k * 30° in degrees (`dhng3.html`).
    pub const fn longitude_deg(self) -> u16 {
        (self as u16) * 30
    }

    /// Toroidal longitude u_k = k * (π / 6) in Q32.31 pi-units (`1 << 31 == π`).
    pub const fn longitude_pi_unit(self) -> i64 {
        // 30° = π / 6 => (1 << 31) / 6 = 357_913_941 in pi-units.
        (self as i64) * 357_913_941
    }

    /// Returns `(Pole A, Pole B)` for this duality axis.
    pub const fn poles(self) -> (RoleId, RoleId) {
        match self {
            Self::PerceiveAct => (RoleId::SensoryPerception, RoleId::MotorPraxis),
            Self::BodySymbol => (RoleId::SomaticInteroception, RoleId::LanguageSemiotics),
            Self::RememberPredict => (RoleId::EpisodicMemory, RoleId::PredictiveSimulation),
            Self::ConvergeDiverge => (RoleId::ConvergentLogic, RoleId::DivergentSynthesis),
            Self::ValueSocial => (RoleId::ValuationSalience, RoleId::SocialCognition),
            Self::SpaceSelfMonitor => (RoleId::SpatialTopography, RoleId::MetacognitiveRegulation),
        }
    }
}

/// The 1 + 12 architectural roles (`dhng3.html`, Qualification #6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum RoleId {
    /// L0: The 1 — central full-depth coordinator at the origin (1, 0, 0, 0).
    Coordinator = 0,
    /// L1: Pole A of `PerceiveAct`.
    SensoryPerception = 1,
    /// L2: Pole B of `PerceiveAct`.
    MotorPraxis = 2,
    /// L3: Pole A of `BodySymbol`.
    SomaticInteroception = 3,
    /// L4: Pole B of `BodySymbol`.
    LanguageSemiotics = 4,
    /// L5: Pole A of `RememberPredict`.
    EpisodicMemory = 5,
    /// L6: Pole B of `RememberPredict`.
    PredictiveSimulation = 6,
    /// L7: Pole A of `ConvergeDiverge`.
    ConvergentLogic = 7,
    /// L8: Pole B of `ConvergeDiverge`.
    DivergentSynthesis = 8,
    /// L9: Pole A of `ValueSocial`.
    ValuationSalience = 9,
    /// L10: Pole B of `ValueSocial`.
    SocialCognition = 10,
    /// L11: Pole A of `SpaceSelfMonitor`.
    SpatialTopography = 11,
    /// L12: Pole B of `SpaceSelfMonitor`.
    MetacognitiveRegulation = 12,
}

impl RoleId {
    /// All 13 levels (L0 Coordinator + L1..L12 Specialists).
    pub const ALL_13: [Self; 13] = [
        Self::Coordinator,
        Self::SensoryPerception,
        Self::MotorPraxis,
        Self::SomaticInteroception,
        Self::LanguageSemiotics,
        Self::EpisodicMemory,
        Self::PredictiveSimulation,
        Self::ConvergentLogic,
        Self::DivergentSynthesis,
        Self::ValuationSalience,
        Self::SocialCognition,
        Self::SpatialTopography,
        Self::MetacognitiveRegulation,
    ];

    /// The 12 carved specialist roles around the 1 (`Qualification #6`).
    pub const CARVED_12: [Self; 12] = [
        Self::SensoryPerception,
        Self::MotorPraxis,
        Self::SomaticInteroception,
        Self::LanguageSemiotics,
        Self::EpisodicMemory,
        Self::PredictiveSimulation,
        Self::ConvergentLogic,
        Self::DivergentSynthesis,
        Self::ValuationSalience,
        Self::SocialCognition,
        Self::SpatialTopography,
        Self::MetacognitiveRegulation,
    ];

    pub const fn level(self) -> u8 {
        self as u8
    }

    /// Index in `0..12` for carved specialist roles (`None` for `Coordinator`).
    pub const fn carved_index(self) -> Option<usize> {
        let lvl = self as usize;
        if lvl == 0 { None } else { Some(lvl - 1) }
    }

    pub const fn from_level(level: u8) -> Option<Self> {
        match level {
            0 => Some(Self::Coordinator),
            1 => Some(Self::SensoryPerception),
            2 => Some(Self::MotorPraxis),
            3 => Some(Self::SomaticInteroception),
            4 => Some(Self::LanguageSemiotics),
            5 => Some(Self::EpisodicMemory),
            6 => Some(Self::PredictiveSimulation),
            7 => Some(Self::ConvergentLogic),
            8 => Some(Self::DivergentSynthesis),
            9 => Some(Self::ValuationSalience),
            10 => Some(Self::SocialCognition),
            11 => Some(Self::SpatialTopography),
            12 => Some(Self::MetacognitiveRegulation),
            _ => None,
        }
    }

    pub const fn id_slug(self) -> &'static str {
        match self {
            Self::Coordinator => "coordinator",
            Self::SensoryPerception => "sensory_perception",
            Self::MotorPraxis => "motor_praxis",
            Self::SomaticInteroception => "somatic_interoception",
            Self::LanguageSemiotics => "language_semiotics",
            Self::EpisodicMemory => "episodic_memory",
            Self::PredictiveSimulation => "predictive_simulation",
            Self::ConvergentLogic => "convergent_logic",
            Self::DivergentSynthesis => "divergent_synthesis",
            Self::ValuationSalience => "valuation_salience",
            Self::SocialCognition => "social_cognition",
            Self::SpatialTopography => "spatial_topography",
            Self::MetacognitiveRegulation => "metacognitive_regulation",
        }
    }

    /// Stable compact identifier for wire protocols and operator-facing metrics.
    pub const fn short_code(self) -> &'static str {
        match self {
            Self::Coordinator => "L0",
            Self::SensoryPerception => "L1",
            Self::MotorPraxis => "L2",
            Self::SomaticInteroception => "L3",
            Self::LanguageSemiotics => "L4",
            Self::EpisodicMemory => "L5",
            Self::PredictiveSimulation => "L6",
            Self::ConvergentLogic => "L7",
            Self::DivergentSynthesis => "L8",
            Self::ValuationSalience => "L9",
            Self::SocialCognition => "L10",
            Self::SpatialTopography => "L11",
            Self::MetacognitiveRegulation => "L12",
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Coordinator => "Coordinator",
            Self::SensoryPerception => "Sensory Perception",
            Self::MotorPraxis => "Motor Praxis",
            Self::SomaticInteroception => "Somatic / Interoception",
            Self::LanguageSemiotics => "Language & Semiotics",
            Self::EpisodicMemory => "Episodic Memory",
            Self::PredictiveSimulation => "Predictive Simulation",
            Self::ConvergentLogic => "Convergent Logic",
            Self::DivergentSynthesis => "Divergent Synthesis",
            Self::ValuationSalience => "Valuation & Salience",
            Self::SocialCognition => "Social Cognition",
            Self::SpatialTopography => "Spatial Topography",
            Self::MetacognitiveRegulation => "Metacognitive Regulation",
        }
    }

    /// Live map member counts reported in `dhng3.html` (`~/.elle/hyper-map/map.json` [reported]).
    pub const fn reported_live_members(self) -> u32 {
        match self {
            Self::Coordinator => 1227,
            Self::SensoryPerception => 0,
            Self::MotorPraxis => 222,
            Self::SomaticInteroception => 0,
            Self::LanguageSemiotics => 412,
            Self::EpisodicMemory => 1,
            Self::PredictiveSimulation => 0,
            Self::ConvergentLogic => 27,
            Self::DivergentSynthesis => 2,
            Self::ValuationSalience => 26,
            Self::SocialCognition => 0,
            Self::SpatialTopography => 35,
            Self::MetacognitiveRegulation => 0,
        }
    }

    /// Duality axis to which this specialist belongs (`None` for `Coordinator`).
    pub const fn duality(self) -> Option<DualityAxis> {
        match self {
            Self::Coordinator => None,
            Self::SensoryPerception | Self::MotorPraxis => Some(DualityAxis::PerceiveAct),
            Self::SomaticInteroception | Self::LanguageSemiotics => Some(DualityAxis::BodySymbol),
            Self::EpisodicMemory | Self::PredictiveSimulation => Some(DualityAxis::RememberPredict),
            Self::ConvergentLogic | Self::DivergentSynthesis => Some(DualityAxis::ConvergeDiverge),
            Self::ValuationSalience | Self::SocialCognition => Some(DualityAxis::ValueSocial),
            Self::SpatialTopography | Self::MetacognitiveRegulation => {
                Some(DualityAxis::SpaceSelfMonitor)
            }
        }
    }

    /// Opposite pole on the same duality axis (`None` for `Coordinator`).
    pub const fn opposite_pole(self) -> Option<Self> {
        match self {
            Self::Coordinator => None,
            Self::SensoryPerception => Some(Self::MotorPraxis),
            Self::MotorPraxis => Some(Self::SensoryPerception),
            Self::SomaticInteroception => Some(Self::LanguageSemiotics),
            Self::LanguageSemiotics => Some(Self::SomaticInteroception),
            Self::EpisodicMemory => Some(Self::PredictiveSimulation),
            Self::PredictiveSimulation => Some(Self::EpisodicMemory),
            Self::ConvergentLogic => Some(Self::DivergentSynthesis),
            Self::DivergentSynthesis => Some(Self::ConvergentLogic),
            Self::ValuationSalience => Some(Self::SocialCognition),
            Self::SocialCognition => Some(Self::ValuationSalience),
            Self::SpatialTopography => Some(Self::MetacognitiveRegulation),
            Self::MetacognitiveRegulation => Some(Self::SpatialTopography),
        }
    }

    /// True if this role is Pole A (outer equator, v = 0) of its duality.
    pub const fn is_pole_a(self) -> bool {
        let lvl = self as u8;
        lvl >= 1 && (lvl % 2 == 1)
    }

    /// Approved toroidal phase `(theta_f = u, theta_s = v)` on T² in Q32.31 pi-units (`dhng3.html`).
    ///
    /// - Pole A (`2k+1`): `(u_k, 0)` on the outer equator.
    /// - Pole B (`2k+2`): `(u_k + π, π)` on the inner equator (antipodal in phase).
    pub const fn toroidal_phase(self) -> Phase {
        let Some(axis) = self.duality() else {
            return Phase {
                theta_f: Q32_31::ZERO,
                theta_s: Q32_31::ZERO,
            };
        };
        let u = axis.longitude_pi_unit();
        if self.is_pole_a() {
            Phase {
                theta_f: Q32_31(u),
                theta_s: Q32_31::ZERO,
            }
        } else {
            // Wrap in hardware 32-bit torus arithmetic (pi-units where 1<<31 == π).
            let u_antipode = (u.wrapping_add(PI_UNIT) as i32) as i64;
            Phase {
                theta_f: Q32_31(u_antipode),
                theta_s: Q32_31(-PI_UNIT),
            }
        }
    }

    /// Topological invariant `(n, m, ψ*)` of this role's home well on T² (`core_lm::lock`).
    pub fn home_invariant(self) -> TopologicalInvariant {
        let p = self.toroidal_phase();
        extract_invariant(p.theta_f.0, p.theta_s.0)
    }
}

/// Choice of 3D spatial frame for projecting the 6 duality axes into H³ (Qualification #3).
///
/// Neither 3D frame can make 6 axes mutually orthogonal (which requires dim >= 6).
/// - `Icosahedral`: Welch-bound optimal in 3D; every pair of distinct axes has |cos θ| = 1/√5 ≈ 0.4472.
/// - `Cuboctahedral`: Legacy vector-equilibrium frame from `atlas13.json` (`dhng3.html`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameGeometry {
    /// Approved icosahedral frame (`pami/src/frame.rs`, `dhng3.html`).
    Icosahedral,
    /// Legacy cuboctahedral frame (`atlas13.json`, `dhng3.html`).
    Cuboctahedral,
}

/// Bit-exact Q32.31 unit direction vectors for Pole A of each of the 6 dualities
/// under the approved icosahedral jitterbug map (`dhng3.html`: `ICO`).
///
/// Construction (`dhng3.html` line 204):
/// Base VE vertices: `[[1,1,0],[1,0,1],[0,1,1],[1,-1,0],[1,0,-1],[0,1,-1]]`.
/// For zero at index `z`, coordinate `(z + 2) % 3` scales by φ = (1+√5)/2, then normalized
/// by `√(1 + φ²) = √(2 + φ) ≈ 1.902113032590307`.
/// Let `a = 1 / √(2 + φ) ≈ 0.5257311121191336` -> `Q32_31(1_128_998_967)`
/// and `b = φ / √(2 + φ) ≈ 0.8506508083520400` -> `Q32_31(1_826_758_701)`.
/// Note: `a² + b² = 1.0` and `a * b = 1/√5` to within 1 ULP in Q32.31 (`[verified]`).
pub const ICO_A: i64 = 1_128_998_967;
pub const ICO_B: i64 = 1_826_758_701;

pub const ICOSAHEDRAL_POLE_A_UNIT: [[Q32_31; 3]; 6] = [
    // k=0: [1, φ, 0] / √(1+φ²) -> [a, b, 0]
    [Q32_31(ICO_A), Q32_31(ICO_B), Q32_31::ZERO],
    // k=1: [φ, 0, 1] / √(1+φ²) -> [b, 0, a]
    [Q32_31(ICO_B), Q32_31::ZERO, Q32_31(ICO_A)],
    // k=2: [0, 1, φ] / √(1+φ²) -> [0, a, b]
    [Q32_31::ZERO, Q32_31(ICO_A), Q32_31(ICO_B)],
    // k=3: [1, -φ, 0] / √(1+φ²) -> [a, -b, 0]
    [Q32_31(ICO_A), Q32_31(-ICO_B), Q32_31::ZERO],
    // k=4: [φ, 0, -1] / √(1+φ²) -> [b, 0, -a]
    [Q32_31(ICO_B), Q32_31::ZERO, Q32_31(-ICO_A)],
    // k=5: [0, 1, -φ] / √(1+φ²) -> [0, a, -b]
    [Q32_31::ZERO, Q32_31(ICO_A), Q32_31(-ICO_B)],
];

/// Legacy cuboctahedral unit direction vectors for Pole A from `atlas13.json` (`dhng3.html`).
pub const CUBOCTAHEDRAL_POLE_A_UNIT: [[Q32_31; 3]; 6] = [
    // L1: [0.4005, 0.2845, 0.8710]
    [
        Q32_31(860_067_201),
        Q32_31(610_959_098),
        Q32_31(1_870_458_257),
    ],
    // L3: [0.9096, 0.4150, 0.0203]
    [
        Q32_31(1_953_351_126),
        Q32_31(891_205_714),
        Q32_31(43_593_918),
    ],
    // L5: [0.7721, -0.5061, 0.3844]
    [
        Q32_31(1_658_072_125),
        Q32_31(-1_086_844_142),
        Q32_31(825_492_714),
    ],
    // L7: [0.1375, 0.9211, -0.3641]
    [
        Q32_31(295_279_002),
        Q32_31(1_978_047_188),
        Q32_31(-781_898_796),
    ],
    // L9: [-0.3716, 0.7906, 0.4866]
    [
        Q32_31(-798_004_924),
        Q32_31(1_697_800_572),
        Q32_31(1_044_968_211),
    ],
    // L11: [-0.5091, -0.1305, 0.8508]
    [
        Q32_31(-1_093_286_574),
        Q32_31(-280_246_616),
        Q32_31(1_827_079_001),
    ],
];

/// Empirical 6x6 pairwise axis cosine matrix reported in `dhng3.html` (`F17` [reported]),
/// stored in Q32.31 fixed-point. Off-diagonal mean |cos| = 0.079, max = 0.1769.
pub const MEASURED_AXIS_COS_Q32: [[Q32_31; 6]; 6] = [
    [
        Q32_31::ONE,
        Q32_31(-149_464_852),
        Q32_31(379_889_858),
        Q32_31(-229_136_505),
        Q32_31(-15_676_631),
        Q32_31(268_650_204),
    ],
    [
        Q32_31(-149_464_852),
        Q32_31::ONE,
        Q32_31(-85_469_848),
        Q32_31(-113_172_388),
        Q32_31(300_218_214),
        Q32_31(-301_506_704),
    ],
    [
        Q32_31(379_889_858),
        Q32_31(-85_469_848),
        Q32_31::ONE,
        Q32_31(189_193_312),
        Q32_31(-280_676_113),
        Q32_31(140_015_933),
    ],
    [
        Q32_31(-229_136_505),
        Q32_31(-113_172_388),
        Q32_31(189_193_312),
        Q32_31::ONE,
        Q32_31(27_702_539),
        Q32_31(5_368_709),
    ],
    [
        Q32_31(-15_676_631),
        Q32_31(300_218_214),
        Q32_31(-280_676_113),
        Q32_31(27_702_539),
        Q32_31::ONE,
        Q32_31(-44_238_163),
    ],
    [
        Q32_31(268_650_204),
        Q32_31(-301_506_704),
        Q32_31(140_015_933),
        Q32_31(5_368_709),
        Q32_31(-44_238_163),
        Q32_31::ONE,
    ],
];

/// Complete geometric descriptor for one of the 13 wells (1 Coordinator + 12 Specialists).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WellDescriptor {
    pub role: RoleId,
    pub frame: FrameGeometry,
    /// Unit direction in ℝ³ (ZERO for Coordinator).
    pub unit_dir: [Q32_31; 3],
    /// Position in the Poincaré ball B³ at radius `r_sph = 0.382` (origin for Coordinator).
    pub poincare_ball: [Q32_31; 3],
    /// Exact isometric image on the future sheet of the Lorentz hyperboloid H³ via optical lens `n(r) = 2/(1-r²)`.
    pub lorentz_well: LorentzVector,
    /// Approved toroidal phase `(theta_f, theta_s)` on T² (`dhng3.html`).
    pub phase: Phase,
    /// Extracted `(n, m, ψ*)` invariant at the well center.
    pub invariant: TopologicalInvariant,
}

impl WellDescriptor {
    /// Construct the well descriptor for `role` in the chosen `frame` using pure Q32.31 integer math.
    pub fn for_role(role: RoleId, frame: FrameGeometry) -> Self {
        let Some(axis) = role.duality() else {
            let phase = role.toroidal_phase();
            return Self {
                role,
                frame,
                unit_dir: [Q32_31::ZERO; 3],
                poincare_ball: [Q32_31::ZERO; 3],
                lorentz_well: LorentzVector::origin(),
                phase,
                invariant: extract_invariant(phase.theta_f.0, phase.theta_s.0),
            };
        };

        let base_a = match frame {
            FrameGeometry::Icosahedral => ICOSAHEDRAL_POLE_A_UNIT[axis.index()],
            FrameGeometry::Cuboctahedral => CUBOCTAHEDRAL_POLE_A_UNIT[axis.index()],
        };

        let sign = if role.is_pole_a() { 1i64 } else { -1i64 };
        let unit_dir = [
            Q32_31(base_a[0].0 * sign),
            Q32_31(base_a[1].0 * sign),
            Q32_31(base_a[2].0 * sign),
        ];

        // Compute Poincare ball coordinates symmetrically (`sign * (base_a * R_SPH)`) so
        // Pole B is bit-for-bit `-Pole A` without floor-shift asymmetry from `Q32_31::mul`.
        let poincare_ball = [
            Q32_31((base_a[0] * R_SPH_Q32).0 * sign),
            Q32_31((base_a[1] * R_SPH_Q32).0 * sign),
            Q32_31((base_a[2] * R_SPH_Q32).0 * sign),
        ];

        let lorentz_well = crate::refraction::OpticalLens::ball_to_lorentz(&poincare_ball)
            .expect("R_SPH = 0.382 < 1 is strictly inside the Poincare ball");

        let phase = role.toroidal_phase();
        let invariant = extract_invariant(phase.theta_f.0, phase.theta_s.0);

        Self {
            role,
            frame,
            unit_dir,
            poincare_ball,
            lorentz_well,
            phase,
            invariant,
        }
    }
}

/// Complete 13-well atlas ("Twelve Around One").
#[derive(Debug, Clone)]
pub struct Atlas13 {
    pub frame: FrameGeometry,
    pub wells: [WellDescriptor; 13],
}

impl Atlas13 {
    pub fn new(frame: FrameGeometry) -> Self {
        let wells = RoleId::ALL_13.map(|role| WellDescriptor::for_role(role, frame));
        Self { frame, wells }
    }

    pub fn approved_icosahedral() -> Self {
        Self::new(FrameGeometry::Icosahedral)
    }

    pub fn legacy_cuboctahedral() -> Self {
        Self::new(FrameGeometry::Cuboctahedral)
    }

    pub fn well(&self, role: RoleId) -> &WellDescriptor {
        &self.wells[role.level() as usize]
    }

    /// Compute the 3D dot product `⟨u_a, u_b⟩` in Q32.31 between two duality axes' Pole A unit vectors.
    pub fn axis_cosine(&self, a: DualityAxis, b: DualityAxis) -> Q32_31 {
        let ua = self.well(a.poles().0).unit_dir;
        let ub = self.well(b.poles().0).unit_dir;
        (ua[0] * ub[0]) + (ua[1] * ub[1]) + (ua[2] * ub[2])
    }

    /// Returns true if two specialist roles share an edge on the icosahedron
    /// (`⟨u_r, u_s⟩ = +1/√5 > 0`, meaning each pole has exactly 5 nearest neighbors on the shell).
    pub fn are_icosahedral_neighbors(&self, r: RoleId, s: RoleId) -> bool {
        if r == RoleId::Coordinator || s == RoleId::Coordinator || r == s {
            return false;
        }
        let ur = self.well(r).unit_dir;
        let us = self.well(s).unit_dir;
        let dot = (ur[0] * us[0]) + (ur[1] * us[1]) + (ur[2] * us[2]);
        // Nearest neighbors on the icosahedron have dot ≈ +1/√5 ≈ +0.4472 > +0.30.
        dot.0 > Q32_31::from_f64(0.30).0
    }

    /// Sample the irrational φ-winding scan orbit `v(u) = φ * u (mod 2π)` on T² (`dhng3.html`).
    /// Never closes; visits every sphere's phase neighborhood ergodically.
    pub fn sample_phi_orbit(u_pi_unit: i64) -> Phase {
        let v_wide = (u_pi_unit as i128 * PHI_Q32.0 as i128) >> Q32_31::FRACTIONAL_BITS;
        Phase {
            theta_f: Q32_31((u_pi_unit as i32) as i64),
            theta_s: Q32_31((v_wide as i32) as i64),
        }
    }

    /// Sample the resonant 2:1 phase-lock orbit `v(u) = 2*u + ψ* (mod 2π)` on T² (`dhng3.html`).
    pub fn sample_resonant_2_to_1_orbit(u_pi_unit: i64, psi_star: i64) -> Phase {
        let two_u = u_pi_unit.wrapping_mul(2);
        let v = (two_u.wrapping_add(psi_star) as i32) as i64;
        Phase {
            theta_f: Q32_31((u_pi_unit as i32) as i64),
            theta_s: Q32_31(v),
        }
    }
}

/// Toroidal phase distance squared `Δu² + Δv²` on T² in Q32.31 (where `1 << 31 == π`).
/// Because `i32` wrap maps angular differences into `[-π, π)`, `|Δu| <= π` (`1.0` in pi-units).
/// Maximum possible squared distance on T² is `1² + 1² = 2.0` (`2 << 31` in Q32.31), achieved
/// uniquely by antipodal phase pairs `(Δu, Δv) = (π, π)` (`[verified]`).
pub fn toroidal_distance_sq_pi_units(a: Phase, b: Phase) -> Q32_31 {
    let du = (a.theta_f.0.wrapping_sub(b.theta_f.0) as i32) as i64;
    let dv = (a.theta_s.0.wrapping_sub(b.theta_s.0) as i32) as i64;
    let du_sq = ((du as i128 * du as i128) >> Q32_31::FRACTIONAL_BITS) as i64;
    let dv_sq = ((dv as i128 * dv as i128) >> Q32_31::FRACTIONAL_BITS) as i64;
    Q32_31(du_sq.saturating_add(dv_sq))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twelve_around_one_structure_is_exact() {
        assert_eq!(RoleId::ALL_13.len(), 13);
        assert_eq!(RoleId::CARVED_12.len(), 12);
        assert_eq!(DualityAxis::ALL.len(), 6);

        let total_members: u32 = RoleId::ALL_13.iter().map(|r| r.reported_live_members()).sum();
        // 1227 (L0) + 222 + 412 + 1 + 27 + 2 + 26 + 35 = 1952 total reported nodes
        assert_eq!(total_members, 1952);
    }

    #[test]
    fn icosahedral_axes_are_equiangular_at_inv_sqrt5() {
        let atlas = Atlas13::approved_icosahedral();
        for i in 0..6 {
            for j in 0..6 {
                let cos = atlas.axis_cosine(DualityAxis::ALL[i], DualityAxis::ALL[j]);
                if i == j {
                    assert!((cos.to_f64() - 1.0).abs() < 1e-8);
                } else {
                    let abs_cos = cos.abs();
                    let diff = (abs_cos.0 - INV_SQRT5_Q32.0).abs();
                    // Every off-diagonal pair of axes meets at |cos| = 1/√5 ≈ 0.4472135955
                    assert!(
                        diff <= 4,
                        "axis ({i},{j}) |cos|={} expected 1/sqrt(5)={} (diff={diff} ULPs)",
                        abs_cos.to_f64(),
                        INV_SQRT5_Q32.to_f64()
                    );
                }
            }
        }
    }

    #[test]
    fn every_icosahedral_pole_has_five_nearest_neighbors_and_one_antipode() {
        let atlas = Atlas13::approved_icosahedral();
        for role in RoleId::CARVED_12 {
            let neighbors: Vec<RoleId> = RoleId::CARVED_12
                .into_iter()
                .filter(|&other| atlas.are_icosahedral_neighbors(role, other))
                .collect();
            assert_eq!(
                neighbors.len(),
                5,
                "role {:?} must have 5 icosahedral neighbors, got {:?}",
                role,
                neighbors
            );
            let opp = role.opposite_pole().unwrap();
            assert!(!neighbors.contains(&opp));
        }
    }

    #[test]
    fn duality_poles_are_antipodal_on_torus_and_hyperboloid() {
        let atlas = Atlas13::approved_icosahedral();
        for axis in DualityAxis::ALL {
            let (pole_a, pole_b) = axis.poles();
            let wa = atlas.well(pole_a);
            let wb = atlas.well(pole_b);

            // Spatial antipodes in B³ and H³
            assert_eq!(wa.poincare_ball[0].0, -wb.poincare_ball[0].0);
            assert_eq!(wa.poincare_ball[1].0, -wb.poincare_ball[1].0);
            assert_eq!(wa.poincare_ball[2].0, -wb.poincare_ball[2].0);
            assert_eq!(wa.lorentz_well.x[0], wb.lorentz_well.x[0]);
            assert_eq!(wa.lorentz_well.x[1].0, -wb.lorentz_well.x[1].0);
            assert_eq!(wa.lorentz_well.x[2].0, -wb.lorentz_well.x[2].0);
            assert_eq!(wa.lorentz_well.x[3].0, -wb.lorentz_well.x[3].0);

            // Toroidal antipodes on T²: (Δu, Δv) = (π, π) => squared distance = 2.0 in pi-units
            let d_sq = toroidal_distance_sq_pi_units(wa.phase, wb.phase);
            assert!(
                (d_sq.to_f64() - 2.0).abs() < 1e-8,
                "axis {:?} toroidal distance sq = {}, expected 2.0",
                axis,
                d_sq.to_f64()
            );
        }
    }
}
