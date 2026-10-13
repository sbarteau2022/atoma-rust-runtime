use core_lm::cache::geometric::*;
use core_lm::geometry::fixed::Q32_31;
use core_lm::geometry::lorentz::LorentzVector;

#[test]
fn test_write_contract_and_audit() {
    let mut map = GeometricMemoryMap::new(10);
    let addr = LorentzVector::origin();
    let phase = Phase {
        theta_f: Q32_31::ZERO,
        theta_s: Q32_31::ZERO,
    };
    let payload = vec![1, 2, 3];

    let id = map
        .write(addr, phase, payload)
        .expect("Write should succeed");

    assert_eq!(map.atoms.len(), 1);
    assert_eq!(map.audit_log.len(), 1);
    assert!(matches!(map.audit_log[0].op, Op::Write(_)));
    assert_eq!(map.audit_log[0].clock.0, 1);
}

#[test]
fn test_eviction_furthest_from_origin() {
    let mut map = GeometricMemoryMap::new(2);
    let phase = Phase {
        theta_f: Q32_31::ZERO,
        theta_s: Q32_31::ZERO,
    };

    // Atom 1: Origin (x0 = 1.0) - Closest
    let addr1 = LorentzVector::origin();
    let id1 = map.write(addr1, phase, vec![1]).unwrap();

    // Atom 2: Distant (x0 = 2.0) - Furthest
    let addr2 = LorentzVector::from_f64([2.0, 1.732, 0.0, 0.0]);
    let id2 = map.write(addr2, phase, vec![2]).unwrap();

    // Map is now full (2/2). Next write triggers eviction.
    let addr3 = LorentzVector::origin();
    map.write(addr3, phase, vec![3]).unwrap();

    // Atom 2 (x0=2.0) should have been evicted.
    assert!(map.atoms.contains_key(&id1.0));
    assert!(!map.atoms.contains_key(&id2.0));

    // Audit log should record the eviction
    let evicted = map.audit_log.iter().any(|e| {
        if let Op::Evict(evicted_id) = e.op {
            evicted_id == id2
        } else {
            false
        }
    });
    assert!(evicted, "Audit log should record eviction of id2");
}

#[test]
fn test_pinning_prevents_eviction() {
    let mut map = GeometricMemoryMap::new(2);
    let phase = Phase {
        theta_f: Q32_31::ZERO,
        theta_s: Q32_31::ZERO,
    };

    // Atom 1: Furthest, but PINNED
    let addr1 = LorentzVector::from_f64([2.0, 1.732, 0.0, 0.0]);
    let id1 = map.write(addr1, phase, vec![1]).unwrap();
    map.pin(id1);

    // Atom 2: Closer, UNPINNED
    let addr2 = LorentzVector::origin();
    let id2 = map.write(addr2, phase, vec![2]).unwrap();

    // Trigger eviction
    map.write(LorentzVector::origin(), phase, vec![3]).unwrap();

    // Even though id1 is further, id2 must be evicted because id1 is pinned.
    assert!(map.atoms.contains_key(&id1.0));
    assert!(!map.atoms.contains_key(&id2.0));
}

#[test]
fn test_deterministic_tie_break() {
    let mut map = GeometricMemoryMap::new(10);
    let phase = Phase {
        theta_f: Q32_31::ZERO,
        theta_s: Q32_31::ZERO,
    };

    // Two atoms at same distance
    let addr = LorentzVector::origin();
    let id1 = map.write(addr, phase, vec![1]).unwrap();
    let id2 = map.write(addr, phase, vec![2]).unwrap();

    let results = map.retrieve(&addr, 10, false).unwrap();

    // Tie-break should prefer lower ID
    assert_eq!(results[0].id, id1);
    assert_eq!(results[1].id, id2);
}
