use core_lm::cache::geometric::GeometricMemoryMap;
use core_lm::geometry::fixed::Q32_31;
use core_lm::geometry::lorentz::LorentzVector;
use core_lm::geometry::picard::PicardMapper;
use core_lm::geometry::torus::TorPhaseEncoder;

#[test]
fn test_geometric_memory_flow() {
    println!("--- Geometric Memory Demonstration ---");

    // 1. Setup
    let encoder = TorPhaseEncoder::new();
    let mut map = GeometricMemoryMap::new(1000);

    // 2. Define some semantic "points" using fixed-length slices
    let concepts: Vec<(&str, &[u8])> = vec![
        ("Apple", b"fruit red sweet"),
        ("Banana", b"fruit yellow long"),
        ("Car", b"vehicle metal fast"),
        ("Truck", b"vehicle metal heavy"),
        ("Ocean", b"water blue deep"),
    ];

    println!("\nCommitting Atoms...");
    for (name, data) in &concepts {
        let (df, ds) = encoder.project_increments(data);

        // Create a unique direction based on the data
        let n = [
            Q32_31::from_f64(df.to_f64().cos()),
            Q32_31::from_f64(ds.to_f64().sin()),
            Q32_31::from_f64(0.1),
        ];

        // Boost from origin to create a point in H3
        let addr = LorentzVector::boost(&LorentzVector::origin(), &n, Q32_31::from_f64(0.5));
        let reduced_addr = PicardMapper::map_to_fundamental_domain(&addr);

        map.store(reduced_addr, name.as_bytes().to_vec())
            .expect("Store failed");
        println!(
            "Stored {}: Address x0={:.4}",
            name,
            reduced_addr.x[0].to_f64()
        );
    }

    // Force a rebuild to populate the VP-Tree
    map.rebuild();

    // 3. Retrieval Test
    println!("\nTesting Retrieval...");
    for (name, data) in &concepts {
        let (df, ds) = encoder.project_increments(data);
        let n = [
            Q32_31::from_f64(df.to_f64().cos()),
            Q32_31::from_f64(ds.to_f64().sin()),
            Q32_31::from_f64(0.1),
        ];
        let query_addr = LorentzVector::boost(&LorentzVector::origin(), &n, Q32_31::from_f64(0.5));
        let reduced_query = PicardMapper::map_to_fundamental_domain(&query_addr);

        let results = map
            .retrieve(&reduced_query, 1, false)
            .expect("Retrieve failed");

        if let Some(atom) = results.first() {
            let found_name = String::from_utf8_lossy(&atom.payload);
            println!("Query {}: Found {} (ID: {})", name, found_name, atom.id);
            assert_eq!(
                *name,
                found_name.as_ref(),
                "Retrieval failed to find the closest atom for {}",
                name
            );
        } else {
            panic!("No atoms found for query {}", name);
        }
    }

    println!("\nResult: Geometric Memory successfully retrieved the correct atoms via VP-Tree.");
}
