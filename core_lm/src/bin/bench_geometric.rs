use core_lm::cache::geometric::{GeometricMemoryMap, Phase};
use core_lm::geometry::fixed::Q32_31;
use core_lm::geometry::lorentz::LorentzVector;

fn main() {
    let mut memory = GeometricMemoryMap::new(1000);
    let v = LorentzVector::origin();
    let phase = Phase {
        theta_f: Q32_31::ZERO,
        theta_s: Q32_31::ZERO,
    };

    for i in 0..100 {
        let _ = memory.write(v, phase, vec![i as u8; 32]);
    }

    println!(
        "Stored 100 atoms. Memory map state: {} atoms",
        memory.atoms.len()
    );
}
