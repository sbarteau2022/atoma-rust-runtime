use core_lm::cache::geometric::*;
use core_lm::geometry::fixed::Q32_31;
use core_lm::geometry::lorentz::LorentzVector;
use core_lm::geometry::torus::TorPhaseEncoder;
use std::collections::HashSet;

struct AttentionOracle {
    atoms: Vec<(LorentzVector, AtomId)>,
}

impl AttentionOracle {
    fn new(atoms: Vec<(LorentzVector, AtomId)>) -> Self {
        Self { atoms }
    }

    fn retrieve(&self, query: &LorentzVector, k: usize) -> Vec<AtomId> {
        let mut scores: Vec<(Q32_31, AtomId)> = self
            .atoms
            .iter()
            .map(|(addr, id)| (LorentzVector::inner_product(query, addr), *id))
            .collect();

        scores.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.0.cmp(&b.1.0)));

        scores.into_iter().take(k).map(|(_, id)| id).collect()
    }
}

fn generate_semantic_cluster(
    encoder: &TorPhaseEncoder,
    seed: &str,
    count: usize,
) -> Vec<(String, LorentzVector)> {
    let mut cluster = Vec::new();
    for i in 0..count {
        let mut text = seed.to_string();
        text.push_str(&format!("_{}", i));
        let (df, ds) = encoder.project_increments(text.as_bytes());
        let n = [
            Q32_31::from_f64((df.to_f64() * 10.0).cos()),
            Q32_31::from_f64((ds.to_f64() * 10.0).sin()),
            Q32_31::from_f64((df.to_f64() * 5.0).cos()),
        ];
        let addr = LorentzVector::boost(&LorentzVector::origin(), &n, Q32_31::from_f64(0.1));
        cluster.push((text, addr));
    }
    cluster
}

#[test]
fn test_retrieval_equivalence_bar() {
    let n_atoms = 1000;
    let k = 10;
    let encoder = TorPhaseEncoder::new();
    let mut memory_map = GeometricMemoryMap::new(n_atoms + 100);

    // Compare unreduced addresses with the exact oracle. Picard would fold
    // both sides by the same isometry and no longer measure the scan itself.
    memory_map.use_picard = false;

    let mut oracle_data = Vec::new();

    let seeds = vec![
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
        "honeydew",
        "kiwi",
        "lemon",
    ];
    for seed in seeds {
        let cluster = generate_semantic_cluster(&encoder, seed, 100);
        for (text, addr) in cluster {
            let phase = Phase {
                theta_f: Q32_31::ZERO,
                theta_s: Q32_31::ZERO,
            };
            let id = memory_map
                .write(addr, phase, text.as_bytes().to_vec())
                .unwrap();
            oracle_data.push((addr, id));
        }
    }

    let oracle = AttentionOracle::new(oracle_data);
    let queries = vec![
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
        "honeydew",
        "kiwi",
        "lemon",
    ];
    let mut total_recall = 0.0;

    for q_text in &queries {
        let (df, ds) = encoder.project_increments(q_text.as_bytes());
        let n = [
            Q32_31::from_f64((df.to_f64() * 10.0).cos()),
            Q32_31::from_f64((ds.to_f64() * 10.0).sin()),
            Q32_31::from_f64((df.to_f64() * 5.0).cos()),
        ];
        let query_addr = LorentzVector::boost(&LorentzVector::origin(), &n, Q32_31::from_f64(0.1));

        let oracle_top_k = oracle.retrieve(&query_addr, k);
        let oracle_set: HashSet<u64> = oracle_top_k.iter().map(|id| id.0).collect();

        let geo_top_k = memory_map.retrieve(&query_addr, k, true).unwrap();
        let geo_set: HashSet<u64> = geo_top_k.iter().map(|a| a.id.0).collect();

        let intersection = oracle_set.intersection(&geo_set).count();
        total_recall += intersection as f64 / k as f64;
    }

    let avg_recall = total_recall / queries.len() as f64;
    println!("Average Recall@{} for N={}: {:.4}", k, n_atoms, avg_recall);
    assert!(
        avg_recall >= 0.95,
        "Recall@{} was {:.4}, below bar 0.95",
        k,
        avg_recall
    );
}
