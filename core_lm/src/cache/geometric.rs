//! Geometric Memory Map (H³/Γ × T² Replacement for KV Cache)
//! Implements the Atom, Write, Retrieval, and Eviction Contracts.
//!
//! FLAG-02, FLAG-03 & Unbounded Picard Reduction Upgrades:
//! - Uses wide-accumulator `LorentzVector::inner_product` to prevent `Q32_31` overflow at large
//!   hyperbolic radii.
//! - Applies `PicardMapper::map_to_fundamental_domain` consistently in `store()`, `write()`, and
//!   `retrieve()` when `self.use_picard` is enabled (`true` by default).
//! - Populates deterministic non-zero `Phase` in `store()` when caller does not supply an explicit
//!   phase, and adds `retrieve_5d()` for joint `H³ × T²` product-manifold retrieval.

use crate::geometry::fixed::Q32_31;
use crate::geometry::lorentz::LorentzVector;
use crate::geometry::picard::PicardMapper;
use std::collections::HashMap;
use std::fmt;
use std::time::SystemTime;

pub type PayloadRef = Vec<u8>;

#[derive(Debug)]
pub enum MemoryError {
    InvalidAddress(String),
    ReductionFailed,
    StorageFull,
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryError::InvalidAddress(e) => write!(f, "Invalid Address: {}", e),
            MemoryError::ReductionFailed => write!(f, "Picard reduction failed to converge"),
            MemoryError::StorageFull => write!(f, "Storage capacity reached and eviction refused"),
        }
    }
}

impl std::error::Error for MemoryError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AtomId(pub u64);

impl fmt::Display for AtomId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LogicalClock(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Phase {
    pub theta_f: Q32_31,
    pub theta_s: Q32_31,
}

impl Phase {
    /// Construct a `Phase` coupled with the `PhaseState` amplitude channel.
    #[inline]
    pub fn with_amplitude(theta_f: Q32_31, theta_s: Q32_31, amplitude: Q32_31) -> Self {
        let amp = amplitude.to_f64();
        let tf = theta_f.to_f64() + 0.15 * amp.sin();
        let ts = theta_s.to_f64() + 0.15 * amp.cos();
        Self {
            theta_f: Q32_31::from_f64(tf),
            theta_s: Q32_31::from_f64(ts),
        }
    }

    /// Derives a deterministic toroidal phase `(theta_f, theta_s)` from a `LorentzVector`'s
    /// spatial coordinates so `store()` never leaves `Phase` silently zeroed when `lorentz != origin`.
    #[inline]
    pub fn from_lorentz_spatial(lorentz: &LorentzVector) -> Self {
        let x1 = lorentz.x[1].to_f64();
        let x2 = lorentz.x[2].to_f64();
        let x3 = lorentz.x[3].to_f64();
        if x1.abs() < 1e-12 && x2.abs() < 1e-12 && x3.abs() < 1e-12 {
            return Self {
                theta_f: Q32_31::ZERO,
                theta_s: Q32_31::ZERO,
            };
        }
        let r_xy = (x1 * x1 + x2 * x2).sqrt();
        Self {
            theta_f: Q32_31::from_f64(x2.atan2(x1)),
            theta_s: Q32_31::from_f64(x3.atan2(r_xy)),
        }
    }

    /// Circular chord distance squared on `T²` in `Q32_31`.
    #[inline]
    pub fn torus_distance_sq(&self, other: &Self) -> Q32_31 {
        let df = self.theta_f - other.theta_f;
        let ds = self.theta_s - other.theta_s;
        let one_minus_cos_f = Q32_31::ONE - Q32_31::cos(df);
        let one_minus_cos_s = Q32_31::ONE - Q32_31::cos(ds);
        one_minus_cos_f + one_minus_cos_s
    }
}

#[derive(Debug, Clone)]
pub struct Atom {
    pub id: AtomId,
    pub clock: LogicalClock,
    pub phase: Phase,
    pub lorentz: LorentzVector,
    pub payload: PayloadRef,
    pub pinned: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Write(AtomId),
    Evict(AtomId),
}

#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub clock: LogicalClock,
    pub op: Op,
    pub timestamp: SystemTime,
}

struct VpNode {
    vantage: u64,
    /// Hyperbolic distance threshold. Children inside are at or below it.
    mu: f64,
    inside: Option<Box<VpNode>>,
    outside: Option<Box<VpNode>>,
}

pub struct GeometricMemoryMap {
    pub atoms: HashMap<u64, Atom>,
    pub max_atoms: usize,
    pub next_id: u64,
    pub clock: LogicalClock,
    pub audit_log: Vec<AuditEntry>,
    pub use_picard: bool,
    index: Option<Box<VpNode>>,
}

impl GeometricMemoryMap {
    pub fn new(max_atoms: usize) -> Self {
        Self {
            atoms: HashMap::new(),
            max_atoms,
            next_id: 0,
            clock: LogicalClock(0),
            audit_log: Vec::new(),
            use_picard: true,
            index: None,
        }
    }

    /// Store an address, respecting `self.use_picard` (`true` by default for Picard reduction
    /// of unbounded radii) and populating `Phase::from_lorentz_spatial(&lorentz)`.
    pub fn store(
        &mut self,
        lorentz: LorentzVector,
        payload: PayloadRef,
    ) -> Result<AtomId, MemoryError> {
        let phase = Phase::from_lorentz_spatial(&lorentz);
        self.write(lorentz, phase, payload)
    }

    /// Build the vantage-point tree used by [`Self::retrieve`].
    pub fn rebuild(&mut self) {
        let ids: Vec<u64> = self.atoms.keys().copied().collect();
        self.index = build_vp(&ids, &self.atoms);
    }

    pub fn write(
        &mut self,
        lorentz: LorentzVector,
        phase: Phase,
        payload: PayloadRef,
    ) -> Result<AtomId, MemoryError> {
        if let Err(e) = lorentz.validate() {
            return Err(MemoryError::InvalidAddress(e));
        }

        let addr = if self.use_picard {
            PicardMapper::map_to_fundamental_domain(&lorentz)
        } else {
            lorentz
        };

        if self.atoms.len() >= self.max_atoms {
            if self.evict().is_none() {
                return Err(MemoryError::StorageFull);
            }
        }

        let id = AtomId(self.next_id);
        self.next_id += 1;
        self.clock.0 += 1;

        let atom = Atom {
            id,
            clock: self.clock,
            phase,
            lorentz: addr,
            payload,
            pinned: false,
        };
        self.index = None;

        self.atoms.insert(id.0, atom);
        self.audit_log.push(AuditEntry {
            clock: self.clock,
            op: Op::Write(id),
            timestamp: SystemTime::now(),
        });

        Ok(id)
    }

    pub fn pin(&mut self, id: AtomId) {
        if let Some(atom) = self.atoms.get_mut(&id.0) {
            atom.pinned = true;
        }
    }

    pub fn evict(&mut self) -> Option<AtomId> {
        self.index = None;
        let victim = self.atoms.values().filter(|a| !a.pinned).max_by(|a, b| {
            a.lorentz.x[0]
                .cmp(&b.lorentz.x[0])
                .then_with(|| a.id.0.cmp(&b.id.0))
        })?;

        let id = victim.id;
        self.audit_log.push(AuditEntry {
            clock: self.clock,
            op: Op::Evict(id),
            timestamp: SystemTime::now(),
        });
        self.atoms.remove(&id.0);
        Some(id)
    }

    pub fn retrieve(
        &self,
        query: &LorentzVector,
        top_k: usize,
        exclude_query: bool,
    ) -> Result<Vec<&Atom>, MemoryError> {
        if let Err(e) = query.validate() {
            return Err(MemoryError::InvalidAddress(e));
        }

        let q_eff = if self.use_picard {
            PicardMapper::map_to_fundamental_domain(query)
        } else {
            *query
        };

        if self.index.is_some() {
            return Ok(self.retrieve_indexed(&q_eff, top_k, exclude_query));
        }

        let mut results = Vec::new();
        for atom in self.atoms.values() {
            let dist = LorentzVector::inner_product(&q_eff, &atom.lorentz);
            results.push((dist, atom));
        }

        // Ascending order for distance (smaller inner product = closer in H3)
        results.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.id.0.cmp(&b.1.id.0)));

        let mut final_results = Vec::new();
        for (_, atom) in results {
            if exclude_query && (atom.lorentz == q_eff || atom.lorentz == *query) {
                continue;
            }
            final_results.push(atom);
            if final_results.len() >= top_k {
                break;
            }
        }

        Ok(final_results)
    }

    /// Joint 5D `H³ × T²` retrieval using VP-tree candidate filtering followed by
    /// product-manifold geodesic ranking (`FLAG-03`).
    pub fn retrieve_5d(
        &self,
        query: &LorentzVector,
        query_phase: &Phase,
        top_k: usize,
        exclude_query: bool,
    ) -> Result<Vec<&Atom>, MemoryError> {
        let q_eff = if self.use_picard {
            PicardMapper::map_to_fundamental_domain(query)
        } else {
            *query
        };
        let oversample = (top_k.saturating_mul(4)).max(top_k);
        let candidates = self.retrieve(&q_eff, oversample, exclude_query)?;
        let mut scored: Vec<(Q32_31, &Atom)> = candidates
            .into_iter()
            .map(|atom| {
                let h3_ip = LorentzVector::inner_product(&q_eff, &atom.lorentz);
                let t2_dist = query_phase.torus_distance_sq(&atom.phase);
                // Weight T^2 phase distance by 0.25 in Q32_31 so H^3 primary topology leads
                // and T^2 breaks ties among angularly close hyperbolic keys.
                let combined = Q32_31(h3_ip.0.saturating_add(t2_dist.0 >> 2));
                (combined, atom)
            })
            .collect();
        scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.id.0.cmp(&b.1.id.0)));
        scored.truncate(top_k);
        Ok(scored.into_iter().map(|(_, a)| a).collect())
    }

    fn retrieve_indexed(
        &self,
        query: &LorentzVector,
        top_k: usize,
        exclude_query: bool,
    ) -> Vec<&Atom> {
        let Some(root) = &self.index else {
            return Vec::new();
        };
        let extra = if exclude_query {
            self.atoms
                .values()
                .filter(|atom| atom.lorentz == *query)
                .count()
        } else {
            0
        };
        let want = top_k.saturating_add(extra);
        let mut best = Vec::new();
        search_vp(&self.atoms, root, query, want, &mut best);
        let mut atoms: Vec<&Atom> = best
            .iter()
            .filter_map(|(_, id)| self.atoms.get(id))
            .filter(|atom| !(exclude_query && atom.lorentz == *query))
            .collect();
        atoms.sort_by(|a, b| {
            let left = LorentzVector::inner_product(query, &a.lorentz);
            let right = LorentzVector::inner_product(query, &b.lorentz);
            left.cmp(&right).then(a.id.0.cmp(&b.id.0))
        });
        atoms.truncate(top_k);
        atoms
    }
}

fn hyp_distance(a: &LorentzVector, b: &LorentzVector) -> f64 {
    let ip = LorentzVector::inner_product(a, b).to_f64().max(1.0);
    (ip + (ip * ip - 1.0).max(0.0).sqrt()).ln()
}

fn consider(best: &mut Vec<(f64, u64)>, k: usize, distance: f64, id: u64) {
    if k == 0 {
        return;
    }
    if let Some(&(worst_d, worst_id)) = best.last() {
        if best.len() == k && (distance, id) >= (worst_d, worst_id) {
            return;
        }
    }
    best.push((distance, id));
    best.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    if best.len() > k {
        best.pop();
    }
}

fn tau_of(best: &[(f64, u64)], k: usize) -> f64 {
    if best.len() < k {
        f64::INFINITY
    } else {
        best.last()
            .map(|(distance, _)| *distance)
            .unwrap_or(f64::INFINITY)
    }
}

fn build_vp(ids: &[u64], atoms: &HashMap<u64, Atom>) -> Option<Box<VpNode>> {
    if ids.is_empty() {
        return None;
    }
    let mut ordered = ids.to_vec();
    ordered.sort_unstable();
    let vantage = ordered[0];
    let rest = &ordered[1..];
    if rest.is_empty() {
        return Some(Box::new(VpNode {
            vantage,
            mu: 0.0,
            inside: None,
            outside: None,
        }));
    }
    let origin = atoms[&vantage].lorentz;
    let mut scored: Vec<(f64, u64)> = rest
        .iter()
        .map(|id| (hyp_distance(&origin, &atoms[id].lorentz), *id))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mu = scored[scored.len() / 2].0;
    let mut inside = Vec::new();
    let mut outside = Vec::new();
    for (distance, id) in scored {
        if distance <= mu {
            inside.push(id);
        } else {
            outside.push(id);
        }
    }
    Some(Box::new(VpNode {
        vantage,
        mu,
        inside: build_vp(&inside, atoms),
        outside: build_vp(&outside, atoms),
    }))
}

fn search_vp(
    atoms: &HashMap<u64, Atom>,
    node: &VpNode,
    query: &LorentzVector,
    k: usize,
    best: &mut Vec<(f64, u64)>,
) {
    let Some(atom) = atoms.get(&node.vantage) else {
        return;
    };
    let distance = hyp_distance(query, &atom.lorentz);
    consider(best, k, distance, node.vantage);

    if distance <= node.mu {
        if let Some(child) = &node.inside {
            if distance <= node.mu + tau_of(best, k) {
                search_vp(atoms, child, query, k, best);
            }
        }
        if let Some(child) = &node.outside {
            if distance + tau_of(best, k) >= node.mu {
                search_vp(atoms, child, query, k, best);
            }
        }
    } else {
        if let Some(child) = &node.outside {
            if distance + tau_of(best, k) >= node.mu {
                search_vp(atoms, child, query, k, best);
            }
        }
        if let Some(child) = &node.inside {
            if distance <= node.mu + tau_of(best, k) {
                search_vp(atoms, child, query, k, best);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::fixed::Q32_31;

    fn address(i: usize) -> LorentzVector {
        let angle = i as f64 * 0.37;
        let n = [
            Q32_31::from_f64(angle.cos()),
            Q32_31::from_f64(angle.sin()),
            Q32_31::ZERO,
        ];
        LorentzVector::boost_rapidity(
            &LorentzVector::origin(),
            &n,
            Q32_31::from_f64(0.15 + i as f64 * 0.03),
        )
    }

    #[test]
    fn vptree_matches_linear_order() {
        let mut map = GeometricMemoryMap::new(80);
        let phase = Phase {
            theta_f: Q32_31::ZERO,
            theta_s: Q32_31::ZERO,
        };
        for i in 0..40 {
            map.write(address(i), phase, vec![i as u8])
                .expect("valid boost address");
        }
        let query = address(3);
        let linear: Vec<u64> = map
            .retrieve(&query, 7, true)
            .unwrap()
            .iter()
            .map(|atom| atom.id.0)
            .collect();
        map.rebuild();
        let indexed: Vec<u64> = map
            .retrieve(&query, 7, true)
            .unwrap()
            .iter()
            .map(|atom| atom.id.0)
            .collect();
        assert_eq!(linear, indexed);
    }

    #[test]
    fn store_populates_nonzero_phase_and_retrieve_5d_works_with_picard_enabled() {
        let mut map = GeometricMemoryMap::new(32);
        assert!(
            map.use_picard,
            "Picard reduction must be enabled by default"
        );
        for i in 1..16 {
            let addr = address(i);
            let id = map.store(addr, vec![i as u8]).unwrap();
            let atom = &map.atoms[&id.0];
            assert!(
                atom.phase.theta_f != Q32_31::ZERO || atom.phase.theta_s != Q32_31::ZERO,
                "store() must populate non-zero T^2 phase for boosted address"
            );
        }
        let q_addr = address(5);
        let q_phase = Phase::from_lorentz_spatial(&q_addr);
        let hits = map.retrieve_5d(&q_addr, &q_phase, 4, false).unwrap();
        assert_eq!(hits.len(), 4);
        assert_eq!(hits[0].payload[0], 5);
    }

    #[test]
    fn retrieve_5d_and_torus_distance_saturate_on_extreme_phase_and_boost() {
        let mut map = GeometricMemoryMap::new(16);
        let extreme_phase_a = Phase {
            theta_f: Q32_31::MAX,
            theta_s: Q32_31::MIN,
        };
        let extreme_phase_b = Phase {
            theta_f: Q32_31::MIN,
            theta_s: Q32_31::MAX,
        };
        // Previously panicked with "Overflow in Q32_31 subtraction"
        let dist = extreme_phase_a.torus_distance_sq(&extreme_phase_b);
        assert!(dist.0 >= 0);

        let high_boost = LorentzVector::from_f64([100_000.0, 99_999.999_995, 0.0, 0.0]);
        map.write(high_boost, extreme_phase_a, vec![42]).unwrap();
        map.write(LorentzVector::origin(), extreme_phase_b, vec![7])
            .unwrap();
        map.rebuild();

        let hits = map
            .retrieve_5d(&high_boost, &extreme_phase_b, 2, false)
            .unwrap();
        assert_eq!(hits.len(), 2);
    }
}
