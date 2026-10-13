use core_lm::Engine;

#[test]
fn test_context_connected_output() {
    let mut engine = Engine::new().expect("Engine init failed");

    // Seed the geometric memory with a specific fact.
    // We seed the "query" as a coordinate and the "response" as the payload.
    engine.seed_knowledge(vec![
        ("Who is the architect?", "Stewart Barteau"),
        ("What is the geometry?", "H3/Gamma"),
    ]);

    // Test 1: Retrieve the architect
    let res1 = engine
        .generate("Who is the architect?", 1)
        .expect("Generation failed");
    assert!(
        res1.contains("Stewart Barteau"),
        "Output should contain the architect. Got: {}",
        res1
    );

    // Test 2: Retrieve the geometry
    let res2 = engine
        .generate("What is the geometry?", 1)
        .expect("Generation failed");
    assert!(
        res2.contains("H3/Gamma"),
        "Output should contain the geometry. Got: {}",
        res2
    );
}
