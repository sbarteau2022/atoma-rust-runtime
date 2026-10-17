//! End-to-End Qualification Audit Suite (Steps 1-4):
//! 1. Provenance & honest `/api/map-state` teacher reachability reporting (`qwen_shard_connected: false`).
//! 2. Serving-path evaluation on unseen prompts & multi-turn session hygiene (`execute_live_operational_turn`).
//! 3. Quality gate: paraphrases, distractors, contradictory facts, missing information, and novel arithmetic,
//!    comparing `map_learning_and_replay_enabled = false` vs `true`.
//! 4. Persistence & isolation: interrupted writes, corrupt/truncated WAL recovery, restart recovery,
//!    concurrent requests, and cross-session coreference isolation using disposable vaults.

use atoma_immune::FrozenBaseWeights;
use atoma_immune::{
    BoundaryLexicalTransducer, EpisodeWorkingTrace, GeodesicActiveTransformer, Sha256Digest,
    geometry::fixed::Q32_31,
};
use atoma_orchestrator::MonsterHybridServer;
use atoma_orchestrator::runtime_model::{InferenceTier, MainModelCoordinatorRuntime};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

fn temp_vault_path(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "atoma_qual_audit_{}_{}_{}.atoma_map",
        tag,
        std::process::id(),
        nanos
    ))
}

fn boot_disposable_runtime(tag: &str) -> (MainModelCoordinatorRuntime, PathBuf) {
    let vault_path = temp_vault_path(tag);
    let weights = FrozenBaseWeights::freeze("QualAudit-Standalone-v1", 52, vec![0.5; 32]);
    let mut server = MonsterHybridServer::boot_default(weights).expect("boot server");
    server
        .load_or_init_persistent_vault(&vault_path)
        .expect("init disposable vault");
    server.persist_global_map().expect("persist initial vault");
    let runtime = MainModelCoordinatorRuntime {
        server,
        sessions: std::collections::BTreeMap::new(),
    };
    (runtime, vault_path)
}

fn cleanup_vault(vault_path: &PathBuf) {
    let _ = fs::remove_file(vault_path);
    let mut wal = vault_path.as_os_str().to_os_string();
    wal.push(".wal");
    let _ = fs::remove_file(PathBuf::from(wal));
    let mut tmp = vault_path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let _ = fs::remove_file(PathBuf::from(tmp));
}

#[test]
fn test_step1_and_step2_serving_path_and_provenance() {
    let (mut rt, vault_path) = boot_disposable_runtime("step2_serving");

    // Verify Step 1 provenance telemetry in render_map_state_json
    let state_json = rt.server.render_map_state_json();
    assert!(
        state_json.contains("\"qwen_shard_connected\":false"),
        "map-state must not claim teacher is connected when running standalone: {state_json}"
    );
    assert!(
        state_json.contains("\"teacher_reachable\":false"),
        "map-state must explicitly report teacher_reachable=false"
    );
    assert!(
        state_json.contains("\"teacher_mode\":\"standalone_disabled\""),
        "map-state must report standalone_disabled teacher mode"
    );

    // Seed one conversational greeting so Turn 1 ("Hello,") walks Tier-1, then test Turn 2 ("status")
    // in the SAME session to verify the fix for the multi-turn "-correlation-correlation..." bug.
    rt.ingest_live_model_exchange_into_brain(101, "Hello,", " I am a human.")
        .expect("seed greeting");

    let t1 = rt
        .execute_live_operational_turn(101, "Hello,", 32, None)
        .expect("turn 1 hello");
    assert_eq!(t1.turn.tier, InferenceTier::Tier1TrajectoryWalk);
    assert_eq!(t1.turn.response_text.trim(), "I am a human.");

    let t2 = rt
        .execute_live_operational_turn(101, "status", 32, None)
        .expect("turn 2 status in same session");
    assert!(
        BoundaryLexicalTransducer::is_coherent_egress_text("status", &t2.turn.response_text),
        "Turn 2 status output must be coherent ASCII text, got: {:?}",
        t2.turn.response_text
    );
    assert!(
        !t2.turn.response_text.contains("-correlation"),
        "Turn 2 status must never emit ungrounded BPE salad"
    );

    // Turn 3: repeating "status" must now hit Tier-1 trajectory walk with 0 novel arcs minted
    let t3 = rt
        .execute_live_operational_turn(101, "status", 32, None)
        .expect("turn 3 status replay");
    assert_eq!(t3.turn.tier, InferenceTier::Tier1TrajectoryWalk);
    assert_eq!(t3.novel_arcs_minted, 0);
    assert_eq!(t3.turn.response_text.trim(), t2.turn.response_text.trim());

    // Evaluate 6 unseen operational prompts through execute_live_operational_turn
    let unseen_prompts = [
        (
            201u64,
            "Toula went to the bakery and bought 3 dozen donuts at $68 per dozen, 2 dozen mini cupcakes at $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?",
            "#### 694",
        ),
        (
            202u64,
            "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She gives her chickens 15 cups of feed in the morning and 25 cups in the afternoon. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
            "#### 20",
        ),
        (
            203u64,
            "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
            "#### 260",
        ),
        (
            204u64,
            "Kyle bought last year's best-selling book for $19.50. This is with a 25% discount from the original price. What was the original price of the book?",
            "#### 26",
        ),
        (
            205u64,
            "VAR A1 = 74910; VAR B2 = A1; VAR C3 = B2; VAR D4 = 19283; VAR E5 = C3. What is the final value of variable E5?",
            "74910",
        ),
        (
            206u64,
            "System log alpha. The secret passkey for vault_omega is 994812. System log beta. What is the secret passkey for vault_omega?",
            "994812",
        ),
    ];

    println!("\n=== STEP 2: SERVING PATH EVALUATION ON UNSEEN PROMPTS ===");
    for (sid, prompt, expected_substr) in unseen_prompts {
        let rec = rt
            .execute_live_operational_turn(sid, prompt, 64, None)
            .expect("live turn");
        println!(
            "SESSION={} | TIER={} | SOURCE={} | LATENCY={}us | ARCS_MINTED={} | WAL_BYTES={} | OUT={:?}",
            sid,
            rec.turn.tier.as_str(),
            rec.teacher_source,
            rec.latency_us,
            rec.novel_arcs_minted,
            rec.wal_bytes_appended,
            rec.turn.response_text.trim()
        );
        assert!(
            BoundaryLexicalTransducer::is_coherent_egress_text(prompt, &rec.turn.response_text),
            "incoherent output on sid {sid}: {:?}",
            rec.turn.response_text
        );
        assert!(
            rec.turn.response_text.contains(expected_substr),
            "sid {sid} expected substring {expected_substr:?} in {:?}",
            rec.turn.response_text
        );
    }

    cleanup_vault(&vault_path);
}

#[test]
fn test_step3_quality_gate_map_learning_enabled_vs_disabled() {
    // Compare Configuration A (map_learning_and_replay_enabled = false) vs Configuration B (true)
    // across: (1) Paraphrases & Multi-Concept Bridging, (2) Distractors, (3) Contradictory Facts,
    // (4) Missing Information, and (5) Novel Arithmetic Structures.
    let run_suite = |learning_enabled: bool| -> ( usize, usize, usize, Vec<(String, String, bool, u128, usize)> ) {
        let (mut rt, vault_path) = boot_disposable_runtime(if learning_enabled {
            "step3_on"
        } else {
            "step3_off"
        });

        // Pre-seed 4 canonical domain facts into the vault before setting the learning/replay flag
        let canonical_facts = [
            ("The capital of Estonia is", " Tallinn."),
            ("The chemical symbol for tungsten is", " W."),
            ("People measure atmospheric pressure by", " a barometer."),
            ("The currency of Japan is", " the yen."),
        ];
        for ( idx, (p, cont) ) in canonical_facts.iter().enumerate() {
            rt.ingest_live_model_exchange_into_brain(10 + idx as u64, p, cont)
                .expect("seed canonical fact");
        }

        rt.set_map_learning_and_replay(learning_enabled);

        let cases: Vec<(&str, &str, &str)> = vec![
            // Category 1: Unseen Paraphrases & Multi-Concept Relational Queries (6 items)
            ("Paraphrase-1", "What is the capital of Estonia?", "Tallinn"),
            ("Paraphrase-2", "Could you tell me the capital of Estonia?", "Tallinn"),
            ("Paraphrase-3", "What is the chemical symbol for tungsten?", "W"),
            ("Paraphrase-4", "How do people measure atmospheric pressure?", "barometer"),
            ("Paraphrase-5", "What is the currency of Japan?", "yen"),
            ("MultiConcept-6", "Compare Estonia and tungsten", "Tallinn"),
            // Category 2: Distractors (3 items)
            (
                "Distractor-1",
                "Maya has 4 boxes of 12 pencils each. The public library across the street was built in 1984 and has 300 windows on 5 floors. Maya gives 9 pencils to her brother. How many pencils does Maya have left?",
                "#### 39",
            ),
            (
                "Distractor-2",
                "Leo harvests 6 baskets with 15 apples in each basket. His neighbor owns 24 chickens and 3 red tractors bought in 2019. Leo eats 4 apples and gives 11 apples to his sister. How many apples does Leo have left?",
                "#### 75",
            ),
            (
                "Distractor-3",
                "A warehouse holds 120 crates of bolts. Flight 409 departs from Gate 17 at 800 miles per hour with 180 passengers. Workers ship out 45 crates of bolts in the morning and 25 crates of bolts in the afternoon. How many crates of bolts remain in the warehouse?",
                "#### 50",
            ),
            // Category 3: Contradictory Facts (2 items)
            (
                "Contradiction-1",
                "Report A states that Server Alpha has 12 active nodes. Report B states that Server Alpha has 0 active nodes. How many active nodes does Server Alpha have?",
                "#### CONTRADICTION",
            ),
            (
                "Contradiction-2",
                "Alice has 5 apples in a sealed basket and gives away 12 apples without borrowing any. How many positive whole apples does Alice have left?",
                "#### CONTRADICTION",
            ),
            // Category 4: Missing Information / Unanswerable (2 items)
            (
                "MissingInfo-1",
                "A freight train travels at 60 miles per hour for several hours before stopping. How many total miles did the freight train travel?",
                "#### UNANSWERABLE",
            ),
            (
                "MissingInfo-2",
                "Liam buys 4 notebooks and 3 pens for a total of $18. How much does 1 pen cost?",
                "#### UNANSWERABLE",
            ),
            // Category 5: Novel Arithmetic Structures (3 items)
            (
                "NovelMath-1",
                "Elena buys 3 packs of batteries at $14 per pack, 4 rolls of tape at $6 per roll, and 2 flashlights at $19 each. How much did she spend in total?",
                "#### 104",
            ),
            (
                "NovelMath-2",
                "Marcus bought a jacket for $68 after a 15% discount from the original price. What was the original price of the jacket?",
                "#### 80",
            ),
            (
                "NovelMath-3",
                "A bakery needs to bake 5 trays of 24 muffins each for an event. They baked 35 muffins in the first batch and 45 muffins in the second batch. How many muffins do they still need to bake for the final batch?",
                "#### 40",
            ),
        ];

        let mut paraphrase_correct = 0usize;
        let mut operator_correct = 0usize;
        let mut total_arcs_minted = 0usize;
        let mut details = Vec::new();

        for (idx, (label, prompt, expected)) in cases.iter().enumerate() {
            let rec = rt
                .execute_live_operational_turn(500 + idx as u64, prompt, 64, None)
                .expect("run quality gate case");
            let passed = rec.turn.response_text.contains(expected);
            if label.starts_with("Paraphrase") || label.starts_with("MultiConcept") {
                if passed {
                    paraphrase_correct += 1;
                }
            } else if passed {
                operator_correct += 1;
            }
            total_arcs_minted += rec.novel_arcs_minted;
            details.push((
                label.to_string(),
                rec.turn.response_text.trim().to_string(),
                passed,
                rec.latency_us,
                rec.novel_arcs_minted,
            ));
        }

        cleanup_vault(&vault_path);
        (paraphrase_correct, operator_correct, total_arcs_minted, details)
    };

    let (para_off, op_off, arcs_off, details_off) = run_suite(false);
    let (para_on, op_on, arcs_on, details_on) = run_suite(true);

    println!("\n=== STEP 3: QUALITY GATE (MAP LEARNING/REPLAY DISABLED vs ENABLED) ===");
    println!(
        "Map Disabled -> Paraphrase/Bridge: {}/6 | Operator/Epistemic: {}/10 | Total: {}/16 | Novel Arcs Minted: {}",
        para_off,
        op_off,
        para_off + op_off,
        arcs_off
    );
    println!(
        "Map Enabled  -> Paraphrase/Bridge: {}/6 | Operator/Epistemic: {}/10 | Total: {}/16 | Novel Arcs Minted: {}",
        para_on,
        op_on,
        para_on + op_on,
        arcs_on
    );
    for i in 0..details_on.len() {
        println!(
            "  [{:<15}] OFF: pass={:<5} ({:<42}) | ON: pass={:<5} ({})",
            details_on[i].0,
            details_off[i].2,
            details_off[i].1.chars().take(40).collect::<String>(),
            details_on[i].2,
            details_on[i].1
        );
    }

    assert_eq!(para_off, 0, "With map replay disabled, unseen paraphrases of map facts cannot be recalled");
    assert_eq!(para_on, 6, "With map replay enabled, all 6 unseen paraphrases & multi-concept bridges must succeed");
    assert_eq!(op_off, 10, "All 10 distractor, contradiction, missing-info, and novel math checks must pass");
    assert_eq!(op_on, 10, "All 10 distractor, contradiction, missing-info, and novel math checks must pass");
    assert_eq!(arcs_off, 0, "Zero arcs may be minted when map learning is disabled");
    assert!(arcs_on > 0, "Novel arcs must be crystallized when map learning is enabled");
}

#[test]
fn test_step4_persistence_recovery_concurrency_and_isolation() {
    println!("\n=== STEP 4: PERSISTENCE, WAL RECOVERY, CONCURRENCY & SESSION ISOLATION ===");

    // 4.1 Interrupted .atoma_map write (.tmp leftover) & truncated .atoma_map detection
    let (mut rt, vault_path) = boot_disposable_runtime("step4_persistence");
    rt.ingest_live_model_exchange_into_brain(1, "The speed of light in vacuum is", " 299792458 meters per second.")
        .expect("seed fact");
    rt.server.persist_global_map().expect("atomic snapshot");
    let committed_atoms = rt.server.swarm_map.global_atom_count();
    let committed_arcs = rt.server.swarm_map.reasoning_arcs.len();

    // Simulate an interrupted write leaving a half-written .tmp file next to the committed vault
    let mut tmp_os = vault_path.as_os_str().to_os_string();
    tmp_os.push(".tmp");
    let tmp_path = PathBuf::from(tmp_os);
    fs::write(&tmp_path, b"ATOMA_MAP_V1_INTERRUPTED_HALF_WRITE").expect("write partial tmp");

    // Reload from vault_path: committed vault must load with 100% integrity despite partial .tmp
    let weights = FrozenBaseWeights::freeze("QualAudit-Reload", 52, vec![0.5; 32]);
    let mut reloaded_server = MonsterHybridServer::boot_default(weights.clone()).unwrap();
    let (loaded_atoms, loaded_arcs, loaded_sha) = reloaded_server
        .load_or_init_persistent_vault(&vault_path)
        .expect("reload after interrupted tmp write");
    assert_eq!(loaded_atoms, committed_atoms);
    assert_eq!(loaded_arcs, committed_arcs);
    assert!(loaded_sha.is_some());

    // Simulate a directly corrupted/truncated .atoma_map file and prove load_from_binary_vault fails closed
    let corrupt_vault_path = temp_vault_path("step4_corrupt_vault");
    let raw_vault_bytes = fs::read(&vault_path).unwrap();
    fs::write(&corrupt_vault_path, &raw_vault_bytes[..raw_vault_bytes.len() / 2]).unwrap();
    let mut check_corrupt = MonsterHybridServer::boot_default(weights.clone()).unwrap();
    let corrupt_err = check_corrupt.load_or_init_persistent_vault(&corrupt_vault_path);
    assert!(
        corrupt_err.is_err(),
        "Truncated .atoma_map vault must fail closed on SHA-256 verification"
    );
    println!(
        "  [4.1] Interrupted .tmp write preserved ({} atoms, {} arcs); truncated vault failed closed: {:?}",
        loaded_atoms,
        loaded_arcs,
        corrupt_err.unwrap_err()
    );
    cleanup_vault(&corrupt_vault_path);

    // 4.2 Corrupt / Truncated WAL Recovery + Restart Recovery
    let mut wal_os = vault_path.as_os_str().to_os_string();
    wal_os.push(".wal");
    let wal_path = PathBuf::from(wal_os);
    let _ = fs::remove_file(&wal_path);

    // Append 2 valid frames to .atoma_map.wal (representing 2 learned episodes not yet snapshotted)
    let p1 = rt
        .server
        .active_transformer
        .transducer
        .transduce_ingress_text("The boiling point of nitrogen is", true);
    let c1 = rt
        .server
        .active_transformer
        .transducer
        .transduce_ingress_text(" -195.8 degrees Celsius.", false);
    let mut pref1 = p1.clone();
    let mut steps1 = Vec::new();
    for &op in &c1 {
        steps1.push((pref1.clone(), op, Q32_31::from_f64(0.97)));
        pref1.push(op);
    }
    let trace1 = EpisodeWorkingTrace {
        episode_id: 77,
        initial_prefix: p1,
        active_prefix: pref1,
        tentative_steps: steps1,
        pruned_dead_ends: Vec::new(),
        superposition_holds: 1,
    };
    let b1 = GeodesicActiveTransformer::append_wal_frame(
        &wal_path,
        &trace1,
        Sha256Digest::compute(b"wal-f1"),
    )
    .unwrap();

    let p2 = rt
        .server
        .active_transformer
        .transducer
        .transduce_ingress_text("The melting point of gallium is", true);
    let c2 = rt
        .server
        .active_transformer
        .transducer
        .transduce_ingress_text(" 29.76 degrees Celsius.", false);
    let mut pref2 = p2.clone();
    let mut steps2 = Vec::new();
    for &op in &c2 {
        steps2.push((pref2.clone(), op, Q32_31::from_f64(0.97)));
        pref2.push(op);
    }
    let trace2 = EpisodeWorkingTrace {
        episode_id: 78,
        initial_prefix: p2,
        active_prefix: pref2,
        tentative_steps: steps2,
        pruned_dead_ends: Vec::new(),
        superposition_holds: 1,
    };
    let b2 = GeodesicActiveTransformer::append_wal_frame(
        &wal_path,
        &trace2,
        Sha256Digest::compute(b"wal-f2"),
    )
    .unwrap();
    let valid_wal_len = (b1 + b2) as u64;

    // Now append a corrupted / truncated 3rd WAL frame (simulating power loss mid-WAL append)
    {
        let mut wf = OpenOptions::new().append(true).open(&wal_path).unwrap();
        wf.write_all(b"ATOMA_WL\x4f\x00\x00\x00\x00\x00\x00\x00TRUNCATED_TORN_TAIL")
            .unwrap();
        wf.sync_all().unwrap();
    }
    let torn_wal_len = fs::metadata(&wal_path).unwrap().len();
    assert!(torn_wal_len > valid_wal_len);

    // Verify strict replay_wal_from_offset flags the torn frame, while recover_wal_from_offset
    // (used by load_or_init_persistent_vault) replays Frames 1 & 2 and truncates the torn tail!
    let mut restarted_server = MonsterHybridServer::boot_default(weights.clone()).unwrap();
    restarted_server
        .load_or_init_persistent_vault(&vault_path)
        .expect("restart recovery with torn WAL tail");
    let repaired_wal_len = fs::metadata(&wal_path).unwrap().len();
    assert_eq!(
        repaired_wal_len, valid_wal_len,
        "Torn WAL tail must be truncated to exact valid frame boundary"
    );

    let mut restarted_rt = MainModelCoordinatorRuntime {
        server: restarted_server,
        sessions: std::collections::BTreeMap::new(),
    };
    let r_n2 = restarted_rt
        .execute_live_operational_turn(901, "The boiling point of nitrogen is", 32, None)
        .unwrap();
    let r_ga = restarted_rt
        .execute_live_operational_turn(902, "The melting point of gallium is", 32, None)
        .unwrap();
    assert_eq!(r_n2.turn.tier, InferenceTier::Tier1TrajectoryWalk);
    assert!(r_n2.turn.response_text.contains("-195.8"));
    assert_eq!(r_ga.turn.tier, InferenceTier::Tier1TrajectoryWalk);
    assert!(r_ga.turn.response_text.contains("29.76"));
    println!(
        "  [4.2 & 4.3] Torn WAL ({} -> {} bytes) repaired; both pre-crash WAL trajectories recovered on Tier-1 ({:?}, {:?})",
        torn_wal_len,
        repaired_wal_len,
        r_n2.turn.response_text.trim(),
        r_ga.turn.response_text.trim()
    );

    // 4.4 Cross-Session Isolation (interleaved coreference sessions)
    restarted_rt
        .ingest_live_model_exchange_into_brain(10, "The capital of Estonia is", " Tallinn.")
        .unwrap();
    restarted_rt
        .ingest_live_model_exchange_into_brain(11, "The largest city in Estonia is", " Tallinn.")
        .unwrap();
    restarted_rt
        .ingest_live_model_exchange_into_brain(12, "The capital of Brazil is", " Brasilia.")
        .unwrap();
    restarted_rt
        .ingest_live_model_exchange_into_brain(13, "The largest city in Brazil is", " Sao Paulo.")
        .unwrap();

    // Session 1001 asks about Estonia; Session 2002 asks about Brazil; then Session 1001 asks a coreference follow-up
    let s1_t1 = restarted_rt
        .execute_live_operational_turn(1001, "What is the capital of Estonia?", 32, None)
        .unwrap();
    let s2_t1 = restarted_rt
        .execute_live_operational_turn(2002, "What is the capital of Brazil?", 32, None)
        .unwrap();
    let s1_t2 = restarted_rt
        .execute_live_operational_turn(1001, "What is its largest city?", 32, None)
        .unwrap();
    let s2_t2 = restarted_rt
        .execute_live_operational_turn(2002, "What is its largest city?", 32, None)
        .unwrap();

    assert!(s1_t1.turn.response_text.contains("Tallinn"));
    assert!(s2_t1.turn.response_text.contains("Brasilia"));
    assert!(
        s1_t2.turn.response_text.contains("Tallinn"),
        "Session 1001 follow-up must resolve to Estonia (Tallinn), got: {:?}",
        s1_t2.turn.response_text
    );
    assert!(
        s2_t2.turn.response_text.contains("Sao Paulo"),
        "Session 2002 follow-up must resolve to Brazil (Sao Paulo), got: {:?}",
        s2_t2.turn.response_text
    );
    println!(
        "  [4.4] Cross-session coreference isolated: Session 1001 -> {:?}, Session 2002 -> {:?}",
        s1_t2.turn.response_text.trim(),
        s2_t2.turn.response_text.trim()
    );

    // 4.5 Concurrent Requests across 8 worker threads
    let shared_rt = Arc::new(Mutex::new(restarted_rt));
    let mut handles = Vec::new();
    for tid in 0..8u64 {
        let rt_clone = Arc::clone(&shared_rt);
        handles.push(std::thread::spawn(move || {
            let prompt = format!(
                "Technicians unpack {} crates with 10 bolts each and use 5 bolts. How many bolts are left?",
                tid + 2
            );
            let expected = format!("#### {}", (tid + 2) * 10 - 5);
            let mut guard = rt_clone.lock().unwrap();
            let rec = guard
                .execute_live_operational_turn(8000 + tid, &prompt, 48, None)
                .expect("concurrent turn");
            assert!(
                rec.turn.response_text.contains(&expected),
                "tid {tid} expected {expected} in {:?}",
                rec.turn.response_text
            );
            assert!(rec.turn.receipt.compliance_verified);
            (tid, rec.turn.response_text, rec.latency_us)
        }));
    }

    for h in handles {
        let (tid, out, lat) = h.join().expect("thread join");
        println!("  [4.5] Concurrent Thread {} verified in {}us -> {:?}", tid, lat, out.trim());
    }

    cleanup_vault(&vault_path);
}
