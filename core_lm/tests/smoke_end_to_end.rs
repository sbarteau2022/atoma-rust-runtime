//! Smoke Test: End-to-End Geometric Memory Map
//!
//! Proves the full contract chain on hardware:
//!   Write -> Retrieve (nearest-first) -> Pin -> Evict -> Audit
//!
//! Run: cargo test --test smoke_end_to_end -- --nocapture

use core_lm::cache::geometric::*;
use core_lm::geometry::fixed::Q32_31;
use core_lm::geometry::lorentz::LorentzVector;

fn make_phase() -> Phase {
    Phase {
        theta_f: Q32_31::ZERO,
        theta_s: Q32_31::ZERO,
    }
}

#[test]
fn smoke_write_retrieve_nearest_first() {
    let mut map = GeometricMemoryMap::new(100);
    map.use_picard = false;
    let phase = make_phase();

    let near = LorentzVector::origin();
    let mid = LorentzVector::from_f64([1.5, 1.118, 0.0, 0.0]);
    let far = LorentzVector::from_f64([2.0, 1.732, 0.0, 0.0]);

    let id_near = map.write(near, phase, b"near".to_vec()).unwrap();
    let id_mid = map.write(mid, phase, b"mid".to_vec()).unwrap();
    let id_far = map.write(far, phase, b"far".to_vec()).unwrap();

    let results = map.retrieve(&near, 3, true).unwrap();

    assert_eq!(results.len(), 2, "Should return 2 (excluding query)");
    assert_eq!(results[0].id, id_mid, "Closest should be first");
    assert_eq!(results[1].id, id_far, "Farthest should be last");

    assert!(id_near.0 < id_mid.0);
    assert!(id_mid.0 < id_far.0);

    println!("PASS Write->Retrieve: nearest-first ordering correct");
    println!(
        "PASS Atom IDs monotonic: {} < {} < {}",
        id_near.0, id_mid.0, id_far.0
    );
}

#[test]
fn smoke_pin_evict_audit() {
    let mut map = GeometricMemoryMap::new(2);
    map.use_picard = false;
    let phase = make_phase();

    let id1 = map
        .write(LorentzVector::origin(), phase, b"one".to_vec())
        .unwrap();
    let id2 = map
        .write(
            LorentzVector::from_f64([2.0, 1.732, 0.0, 0.0]),
            phase,
            b"two".to_vec(),
        )
        .unwrap();

    map.pin(id2);

    let id3 = map
        .write(LorentzVector::origin(), phase, b"three".to_vec())
        .unwrap();

    assert!(map.atoms.contains_key(&id2.0), "Pinned atom must survive");
    assert!(
        !map.atoms.contains_key(&id1.0),
        "Unpinned atom must be evicted"
    );
    assert!(map.atoms.contains_key(&id3.0), "New atom must be present");

    assert_eq!(map.audit_log.len(), 4, "Audit log should have 4 entries");

    let writes = map
        .audit_log
        .iter()
        .filter(|e| matches!(e.op, Op::Write(_)))
        .count();
    let evicts = map
        .audit_log
        .iter()
        .filter(|e| matches!(e.op, Op::Evict(_)))
        .count();
    assert_eq!(writes, 3);
    assert_eq!(evicts, 1);

    let clocks: Vec<u64> = map.audit_log.iter().map(|e| e.clock.0).collect();
    for w in clocks.windows(2) {
        assert!(w[0] <= w[1], "Logical clock must be monotonic");
    }

    println!("PASS Pin->Evict: pinned atom survived, unpinned evicted");
    println!(
        "PASS Audit: {} writes, {} evictions, clock monotonic",
        writes, evicts
    );
}

#[test]
fn smoke_determinism() {
    fn run_sequence() -> Vec<u64> {
        let mut map = GeometricMemoryMap::new(50);
        map.use_picard = false;
        let phase = make_phase();
        for i in 0..10 {
            let addr = LorentzVector::from_f64([
                1.0 + (i as f64 * 0.1),
                (i as f64 * 0.05).sin(),
                0.0,
                0.0,
            ]);
            let _ = map.write(addr, phase, format!("atom_{}", i).as_bytes().to_vec());
        }
        let results = map.retrieve(&LorentzVector::origin(), 5, true).unwrap();
        results.iter().map(|a| a.id.0).collect()
    }

    let run1 = run_sequence();
    let run2 = run_sequence();
    assert_eq!(run1, run2, "Retrieval must be deterministic across runs");
    println!("PASS Determinism: two runs produced identical top-5");
}
