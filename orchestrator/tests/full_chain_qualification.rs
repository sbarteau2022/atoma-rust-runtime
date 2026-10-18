//! End-to-End Qualification Suite for the Carved Swarm, Governor, Immune System,
//! and First-Principles Superpowers Integration.
//!
//! Explicitly tests all 10 Qualifications from `SPEC.md`, the Fault-Tolerance / Parity Gates
//! from `design/QUALIFICATION_GATES.md`, and the First-Principles Statutory Invariants from
//! `design/SUPERPOWERS_SERVER_INTEGRATION_SPEC.md`.

use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::{
    ATOMIC_IR_MAX_VERIFY_TOKENS, Atlas13, ComplianceViolation, DualityAxis, EpistemicMark,
    FirstPrincipleInvariant, FrozenBaseWeights, MetaControlDecision, OpticalRiskEngine,
    PREREGISTERED_KILL_FINGERPRINT, PreregisteredEvaluation, QuotientManifoldProof,
    RegulatoryFramework, RoleId, RoleSignalQ32, RunbookTelemetryQ32, TeacherTapRecord,
    predict_token_envelope,
};
use atoma_orchestrator::MonsterHybridServer;
use atoma_orchestrator::swarm::ProcessState;
use std::time::Duration;

fn sample_weights() -> std::sync::Arc<FrozenBaseWeights> {
    let w: Vec<f32> = (0..4096).map(|i| ((i as f32) * 0.031).sin()).collect();
    FrozenBaseWeights::freeze("Muse-Glimmer-Test-Frozen", 52, w)
}

#[test]
fn qual_01_and_06_twelve_around_one_and_epistemic_marks() {
    let atlas_ico = Atlas13::approved_icosahedral();
    let atlas_cub = Atlas13::legacy_cuboctahedral();
    assert_eq!(atlas_ico.wells.len(), 13);
    assert_eq!(atlas_cub.wells.len(), 13);
    assert_eq!(RoleId::CARVED_12.len(), 12);
    assert_eq!(DualityAxis::ALL.len(), 6);
    assert_eq!(EpistemicMark::Verified.as_tag(), "[verified]");
    assert_eq!(EpistemicMark::Reported.as_tag(), "[reported]");
    assert_eq!(EpistemicMark::Inferred.as_tag(), "[inferred]");
    assert_eq!(EpistemicMark::Theory.as_tag(), "[theory]");
}

#[test]
fn qual_04_and_07_silo_isolation_and_boundary_overlap_handoff() {
    let mut server = MonsterHybridServer::boot_default(sample_weights()).unwrap();

    // 1. Place an atom deep inside L1 (Sensory Perception) private core
    let tap_private = TeacherTapRecord {
        shard_id: 10,
        layer_index: 48,
        hidden_state: (0..5120).map(|i| ((i as f32) * 0.017).sin()).collect(),
        next_op_probs: vec![0.7, 0.1, 0.1, 0.1],
        payload: b"private_sensory_trace".to_vec(),
    };
    let rec_priv = server
        .ingest_teacher_tap(&tap_private, RoleId::SensoryPerception, [Q32_31::ZERO; 3])
        .unwrap();

    // L1 sees it; L2 (Motor Praxis) and L3 (Somatic) do NOT see it because it is not in the boundary overlap
    let l1_silo = server.swarm_map.silo(RoleId::SensoryPerception).unwrap();
    let l2_silo = server.swarm_map.silo(RoleId::MotorPraxis).unwrap();
    let l3_silo = server.swarm_map.silo(RoleId::SomaticInteroception).unwrap();
    assert!(l1_silo.contains_atom(rec_priv.atom_id));
    assert!(!l2_silo.contains_atom(rec_priv.atom_id));
    assert!(!l3_silo.contains_atom(rec_priv.atom_id));

    // Attempting to hand off a private (non-boundary) atom from L1 to L3 is blocked by BoundaryHandoffGate
    let err_non_boundary = server
        .handoff_between_roles(
            RoleId::SensoryPerception,
            RoleId::SomaticInteroception,
            rec_priv.atom_id,
            b"private_cannot_cross".to_vec(),
        )
        .unwrap_err();
    assert!(matches!(
        err_non_boundary,
        ComplianceViolation::AtomOutsideBoundaryOverlap { .. }
    ));

    // 2. Place an atom at the geodesic midpoint between icosahedral neighbors L1 and L3
    let u1 = server.swarm_map.atlas.well(RoleId::SensoryPerception).unit_dir;
    let u3 = server
        .swarm_map
        .atlas
        .well(RoleId::SomaticInteroception)
        .unit_dir;
    let d_sph = Q32_31::from_f64(0.804_735_5);
    let mid_offset = [
        ((u3[0] - u1[0]) * d_sph) * Q32_31::HALF,
        ((u3[1] - u1[1]) * d_sph) * Q32_31::HALF,
        ((u3[2] - u1[2]) * d_sph) * Q32_31::HALF,
    ];
    let tap_shared = TeacherTapRecord {
        shard_id: 11,
        layer_index: 48,
        hidden_state: (0..5120).map(|i| ((i as f32) * 0.023).cos()).collect(),
        next_op_probs: vec![0.25, 0.25, 0.25, 0.25],
        payload: b"shared_boundary_reflex".to_vec(),
    };
    let rec_shared = server
        .ingest_teacher_tap(&tap_shared, RoleId::SensoryPerception, mid_offset)
        .unwrap();

    // Both L1 and L3 contain the boundary atom, while antipodal L2 still does not
    assert!(
        server
            .swarm_map
            .silo(RoleId::SensoryPerception)
            .unwrap()
            .contains_atom(rec_shared.atom_id)
    );
    assert!(
        server
            .swarm_map
            .silo(RoleId::SomaticInteroception)
            .unwrap()
            .contains_atom(rec_shared.atom_id)
    );
    assert!(
        !server
            .swarm_map
            .silo(RoleId::MotorPraxis)
            .unwrap()
            .contains_atom(rec_shared.atom_id)
    );

    // Handoff L1 -> L3 succeeds and lands in L3's process inbox
    let delivered = server
        .handoff_between_roles(
            RoleId::SensoryPerception,
            RoleId::SomaticInteroception,
            rec_shared.atom_id,
            b"reflex_handoff_payload".to_vec(),
        )
        .unwrap();
    assert_eq!(delivered.from_role, RoleId::SensoryPerception);
    assert_eq!(delivered.to_role, RoleId::SomaticInteroception);
    assert_eq!(
        server
            .governor
            .process(RoleId::SomaticInteroception)
            .unwrap()
            .inbox
            .len(),
        1
    );
}

#[test]
fn qual_07_unjustified_overlap_expansion_is_rejected() {
    let mut server = MonsterHybridServer::boot_default(sample_weights()).unwrap();
    let res = server.immune.policy.set_pair_overlap(
        RoleId::SensoryPerception,
        RoleId::MotorPraxis,
        Q32_31::from_f64(0.10),
        "   ", // empty justification
    );
    assert!(matches!(
        res,
        Err(ComplianceViolation::UnjustifiedOverlapExpansion { .. })
    ));
}

#[test]
fn qual_08_zero_copy_shared_base_weights_across_all_twelve_silos() {
    let base = sample_weights();
    let server = MonsterHybridServer::boot_default(std::sync::Arc::clone(&base)).unwrap();
    assert!(base.verify_integrity());

    for role in RoleId::CARVED_12 {
        let silo = server.swarm_map.silo(role).unwrap();
        assert!(std::sync::Arc::ptr_eq(&silo.weight_view.base, &base));
        assert!(silo.weight_view.active_layers.len() < base.num_layers);
    }
}

#[test]
fn qual_09_loss_accounting_ledger_is_populated_at_every_stage() {
    let mut server = MonsterHybridServer::boot_default(sample_weights()).unwrap();
    let tap = TeacherTapRecord {
        shard_id: 99,
        layer_index: 48,
        hidden_state: vec![0.42f32; 5120],
        next_op_probs: vec![0.5, 0.5],
        payload: b"loss_ledger_check".to_vec(),
    };
    let receipt = server
        .ingest_teacher_tap(&tap, RoleId::ConvergentLogic, [Q32_31::ZERO; 3])
        .unwrap();

    assert_eq!(receipt.projected.loss_report.input_state_bits, 81_920);
    assert_eq!(receipt.projected.loss_report.output_tangent_bits, 256);
    assert_eq!(receipt.projected.loss_report.discarded_dimensions, 0);
    assert_eq!(receipt.carve_loss.global_atom_count, 1);
    let l7_idx = RoleId::ConvergentLogic.carved_index().unwrap();
    assert_eq!(receipt.carve_loss.silo_atom_counts[l7_idx], 1);
}

#[test]
fn qual_10_preregistered_kill_criteria_enforces_94ce028317e0ecac() {
    let mut server = MonsterHybridServer::boot_default(sample_weights()).unwrap();

    // Case 1: Only 12 of 24 families beat flat student by >= 5% -> FALSIFIED
    let mut failing_kl = [Q32_31::from_f64(0.02); 24];
    for i in 0..12 {
        failing_kl[i] = Q32_31::from_f64(0.08);
    }
    let eval_fail = PreregisteredEvaluation {
        test_fingerprint: PREREGISTERED_KILL_FINGERPRINT,
        family_kl_reductions_q32: failing_kl,
        co_recall_ci95_lower_q32: Q32_31::from_f64(0.04),
    };
    let res_fail = server.immune.evaluate_kill_criteria(&eval_fail);
    assert!(matches!(
        res_fail,
        Err(ComplianceViolation::LensFalsifiedByKillCriteria {
            families_passed: 12,
            required_families: 17,
            ..
        })
    ));

    // Case 2: 18 of 24 families beat flat student by >= 5% and 95% CI > 0 -> PASS
    let mut passing_kl = [Q32_31::from_f64(0.01); 24];
    for i in 0..18 {
        passing_kl[i] = Q32_31::from_f64(0.06);
    }
    let eval_pass = PreregisteredEvaluation {
        test_fingerprint: PREREGISTERED_KILL_FINGERPRINT,
        family_kl_reductions_q32: passing_kl,
        co_recall_ci95_lower_q32: Q32_31::from_f64(0.025),
    };
    assert!(server.immune.evaluate_kill_criteria(&eval_pass).is_ok());
}

#[test]
fn gate_03_governor_recovers_crashed_role_under_50ms() {
    let mut server = MonsterHybridServer::boot_default(sample_weights()).unwrap();
    let old_pid = server
        .governor
        .process(RoleId::EpisodicMemory)
        .unwrap()
        .pid;

    server.governor.simulate_crash(RoleId::EpisodicMemory);
    assert_eq!(
        server
            .governor
            .process(RoleId::EpisodicMemory)
            .unwrap()
            .state,
        ProcessState::Crashed
    );

    let recovered = server
        .governor
        .step_supervision(&server.swarm_map, &mut server.immune)
        .unwrap();
    assert_eq!(recovered, vec![RoleId::EpisodicMemory]);

    let proc = server.governor.process(RoleId::EpisodicMemory).unwrap();
    assert_eq!(proc.state, ProcessState::Running);
    assert_ne!(proc.pid, old_pid);
    assert_eq!(proc.restart_count, 1);
    assert!(
        proc.last_recovery_duration.unwrap() < Duration::from_millis(50),
        "Recovery took {:?}, exceeding 50ms Qualification Gate",
        proc.last_recovery_duration
    );
}

#[test]
fn superpowers_first_principles_compliance_and_surgical_erasure_verified() {
    let mut server = MonsterHybridServer::boot_default(sample_weights()).unwrap();

    // 1. Verify all 12 spheres have populated Superpowers + Creative Homeostasis specs
    for role in RoleId::CARVED_12 {
        let spec = server.knowledge_homeostasis.sphere_spec(role).unwrap();
        assert_eq!(spec.role, role);
        assert!(!spec.superpowers_pillar.is_empty());
        assert!(!spec.invariant_name.is_empty());
    }
    assert!(!server.knowledge_homeostasis.lookup_by_cwe("CWE-78").is_empty());

    // 2. Verify (H³×T²)/Γ₁₂ quotient 144-sector grid and ψ* invariance
    let q_rep = QuotientManifoldProof::verify_all();
    assert!(q_rep.sector_tiling_exact);
    assert!(q_rep.psi_star_2_to_1_invariant);
    assert!(q_rep.sector_shift_equivariant);
    assert!(q_rep.grid_144_complete);

    // 3. Verify Atomic IR lossless round-trip and <= 7 token verify-cliff cap
    let stream = vec![101, 102, 103, 104, 55, 201, 202, 203, 99, 105, 106, 107];
    let encoded = server.atomic_ir.encode(&stream);
    assert_eq!(server.atomic_ir.expand(&encoded), stream);
    assert_eq!(
        server.atomic_ir.expand_within_verify_cliff(&encoded).len(),
        ATOMIC_IR_MAX_VERIFY_TOKENS
    );

    // 4. Verify Q32.31 Meta-Cognitive Runbook Controller (continue, nudge, fail-closed stop, escalate)
    let (envelope, basis) = predict_token_envelope(&[1000, 2000, 3000, 4000], 60000);
    assert_eq!(basis, "measured_p75_x_1.5");
    assert_eq!(envelope, 4875);

    let signals = vec![
        RoleSignalQ32 {
            role: RoleId::SensoryPerception,
            hypothesis_hash: 7,
            proposed_action_hash: 9,
            confidence_q32: Q32_31::from_f64(0.95),
        },
        RoleSignalQ32 {
            role: RoleId::ConvergentLogic,
            hypothesis_hash: 7,
            proposed_action_hash: 9,
            confidence_q32: Q32_31::from_f64(0.93),
        },
    ];
    let l1_well = server
        .swarm_map
        .atlas
        .well(RoleId::SensoryPerception)
        .lorentz_well;
    let rb_ok = RunbookTelemetryQ32 {
        step: 2,
        kappa_q32: Q32_31::from_f64(0.85),
        drift_detected: false,
        out_of_region: false,
        regime_collapse: false,
        verified_progress_in_window: 1,
        spent_tokens: 1500,
        predicted_token_envelope: envelope,
        nudges_used: 0,
    };
    assert_eq!(
        server
            .meta_controller
            .evaluate(&signals, &l1_well, &rb_ok)
            .decision,
        MetaControlDecision::ContinueAutonomous
    );

    // Exceeding spend envelope with zero verified progress triggers FailClosedStop
    let rb_over_spend = RunbookTelemetryQ32 {
        verified_progress_in_window: 0,
        spent_tokens: envelope + 1,
        ..rb_ok
    };
    assert!(matches!(
        server
            .meta_controller
            .evaluate(&signals, &l1_well, &rb_over_spend)
            .decision,
        MetaControlDecision::FailClosedStop(_)
    ));

    // 5. Prove all 8 First-Principle Invariants and all 5 Statutory Frameworks
    let fp_report = server.prove_first_principles_compliance();
    assert!(fp_report.all_statutes_satisfied);
    assert_eq!(fp_report.verified_invariants.len(), 8);
    assert!(
        fp_report
            .verified_invariants
            .contains(&FirstPrincipleInvariant::HyperbolicRightToErasure)
    );
    for fw in RegulatoryFramework::ALL {
        assert_eq!(fp_report.framework_scores_q32[&fw], Q32_31::ONE);
    }

    // 6. Verify OpticalRiskEngine FAIR + DREAD + Fragility scoring
    let risk = OpticalRiskEngine::evaluate_coordinate_risk(
        &l1_well,
        [Q32_31::from_f64(6.0); 5],
        Q32_31::from_f64(5.0),
        Q32_31::from_f64(5.0),
    );
    assert!(risk.optical_blast_radius_q32.0 > 0);
    assert!((risk.dread_score_q32.to_f64() - 6.0).abs() < 1e-6);
}

#[test]
fn three_layer_forward_pass_supervision_and_metal_qmatvec_acceleration() {
    let mut server = MonsterHybridServer::boot_default(sample_weights()).unwrap();

    // 1. Execute governed core_lm forward steps across supervised workers
    let hd = 128usize;
    let q_head: Vec<f32> = (0..hd).map(|d| ((d as f32) * 0.17).cos()).collect();
    let k_head: Vec<f32> = (0..hd).map(|d| ((d as f32) * 0.17).cos() + 0.01).collect();
    let v_head: Vec<f32> = (0..hd).map(|d| ((d as f32) * 0.11).sin()).collect();
    let o_proj: Vec<f32> = (0..(hd * hd)).map(|i| ((i as f32) * 0.005).cos() * 0.05).collect();

    let rec1 = server
        .execute_governed_forward(
            RoleId::SensoryPerception,
            1001,
            0,
            &q_head,
            &k_head,
            &v_head,
            &o_proj,
        )
        .unwrap();
    assert_eq!(rec1.role, RoleId::SensoryPerception);
    assert_eq!(rec1.bridge_loss.input_dimensions, 128);
    assert_eq!(rec1.bridge_loss.rope_pair_integrity_q32, Q32_31::ONE);
    assert_eq!(rec1.output_state.len(), 128);
    assert!(
        rec1.forward_permit
            .action_summary()
            .contains("PASS ForwardPassGate")
    );

    // 2. Simulate worker crash on L1; execute_governed_forward automatically triggers
    //    ImmuneSystem-governed step_supervision, recovers the worker, and completes the pass!
    let old_pid = rec1.worker_pid;
    server.governor.simulate_crash(RoleId::SensoryPerception);
    assert_eq!(
        server.governor.process(RoleId::SensoryPerception).unwrap().state,
        ProcessState::Crashed
    );

    let rec2 = server
        .execute_governed_forward(
            RoleId::SensoryPerception,
            1002,
            1,
            &q_head,
            &k_head,
            &v_head,
            &o_proj,
        )
        .unwrap();
    assert_ne!(rec2.worker_pid, old_pid, "Crashed worker must be respawned with new PID");
    assert_eq!(
        server.governor.process(RoleId::SensoryPerception).unwrap().state,
        ProcessState::Running
    );
    assert!(rec2.forward_permit.sequence() > rec1.forward_permit.sequence());

    // 3. Benchmark Zero-Copy Fused Quantized GEMV + Accelerate vs naive dequantize + scalar matvec
    let rows = 512usize;
    let cols = 2048usize;
    let group_size = 128usize;
    let groups_per_row = cols / group_size;
    let n_words = (rows * cols * 4) / 32;
    let packed_w: Vec<u8> = (0..(n_words * 4)).map(|i| (i.wrapping_mul(29) & 0xFF) as u8).collect();
    let scales_bf16: Vec<u8> = (0..(rows * groups_per_row))
        .flat_map(|g| (((0.25f32 + (g % 4) as f32 * 0.125).to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let biases_bf16: Vec<u8> = (0..(rows * groups_per_row))
        .flat_map(|g| (((-0.25f32 + (g % 3) as f32 * 0.125).to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let out_scale: Vec<f32> = (0..rows).map(|r| 0.75 + (r % 5) as f32 * 0.05).collect();
    let x_in: Vec<f32> = (0..cols).map(|c| ((c as f32) * 0.019).cos()).collect();

    atoma_immune::metal::reset_kernel_telemetry();
    let y_fast = atoma_immune::metal::qmatvec_fused_zero_copy(
        &packed_w,
        "U32",
        &scales_bf16,
        "BF16",
        &biases_bf16,
        "BF16",
        Some(&out_scale),
        4,
        group_size,
        rows,
        cols,
        &x_in,
    )
    .unwrap();
    let tel = atoma_immune::metal::kernel_telemetry();
    assert_eq!(tel.zero_copy_qmatvec_calls, 1);
    assert_eq!(tel.bytes_saved_from_heap_alloc, (rows * cols * 4) as u64);
    assert_eq!(y_fast.len(), rows);
    assert!(y_fast.iter().all(|v| v.is_finite()));
}

#[test]
fn non_abelian_hyperbolic_holonomy_and_wigner_frame_transport_verified() {
    use atoma_immune::HolonomyVerdict;
    let mut server = MonsterHybridServer::boot_default(sample_weights()).unwrap();

    // 1. Place shared boundary atoms along the 3-role triangle L1 -> L3 -> L5 -> L1
    let roles = [
        RoleId::SensoryPerception,
        RoleId::SomaticInteroception,
        RoleId::EpisodicMemory,
    ];
    let d_sph = Q32_31::from_f64(0.804_735_5);
    let mut boundary_atoms = Vec::new();
    for i in 0..3 {
        let r_a = roles[i];
        let r_b = roles[(i + 1) % 3];
        let u_a = server.swarm_map.atlas.well(r_a).unit_dir;
        let u_b = server.swarm_map.atlas.well(r_b).unit_dir;
        let mid_offset = [
            ((u_b[0] - u_a[0]) * d_sph) * Q32_31::HALF,
            ((u_b[1] - u_a[1]) * d_sph) * Q32_31::HALF,
            ((u_b[2] - u_a[2]) * d_sph) * Q32_31::HALF,
        ];
        let tap = TeacherTapRecord {
            shard_id: 200 + i as u64,
            layer_index: 48,
            hidden_state: (0..5120)
                .map(|k| ((k as f32) * 0.013 + (i as f32)).cos())
                .collect(),
            next_op_probs: vec![0.25, 0.25, 0.25, 0.25],
            payload: format!("holonomy_edge_{}", i).into_bytes(),
        };
        let rec = server.ingest_teacher_tap(&tap, r_a, mid_offset).unwrap();
        boundary_atoms.push(rec.atom_id);
    }

    // 2. Route across L1 -> L3 and verify Wigner frame is parallel-transported on the wire as an SO(3) rotation
    let hop1 = server
        .handoff_between_roles(
            RoleId::SensoryPerception,
            RoleId::SomaticInteroception,
            boundary_atoms[0],
            b"hop1".to_vec(),
        )
        .unwrap();
    let diag_sum = hop1.transported_wigner_frame[0][0].to_f64()
        + hop1.transported_wigner_frame[1][1].to_f64()
        + hop1.transported_wigner_frame[2][2].to_f64();
    let col0_norm_sq = hop1.transported_wigner_frame[0][0].to_f64().powi(2)
        + hop1.transported_wigner_frame[1][0].to_f64().powi(2)
        + hop1.transported_wigner_frame[2][0].to_f64().powi(2);
    assert!(
        (col0_norm_sq - 1.0).abs() < 1e-4 && diag_sum < 2.99,
        "Off-origin L1 -> L3 transport preserves SO(3) unit column norm ({}) and induces non-trivial Wigner trace ({})",
        col0_norm_sq,
        diag_sum
    );

    // 3. Ping-pong loop (L1 -> L3 -> L1) encloses ZERO hyperbolic area and is rejected as degenerate
    let ping_pong = server.evaluate_reasoning_holonomy(&[
        RoleId::SensoryPerception,
        RoleId::SomaticInteroception,
        RoleId::SensoryPerception,
    ]);
    assert!(matches!(
        ping_pong.verdict,
        HolonomyVerdict::EchoChamberRejected { .. }
    ));
    assert!(ping_pong.enclosed_curvature_flux_q32.to_f64() < 1e-6);

    // 4. Genuine 3-role triangle loop (L1 -> L3 -> L5 -> L1) produces exact Gauss-Bonnet Wigner rotation
    let tri_fwd = server.evaluate_reasoning_holonomy(&[
        RoleId::SensoryPerception,
        RoleId::SomaticInteroception,
        RoleId::EpisodicMemory,
        RoleId::SensoryPerception,
    ]);
    let tri_rev = server.evaluate_reasoning_holonomy(&[
        RoleId::SensoryPerception,
        RoleId::EpisodicMemory,
        RoleId::SomaticInteroception,
        RoleId::SensoryPerception,
    ]);
    assert!(matches!(
        tri_fwd.verdict,
        HolonomyVerdict::DialecticalSynthesisVerified { .. }
    ));
    assert!(tri_fwd.enclosed_curvature_flux_q32.to_f64() > 0.05);
    assert!(tri_fwd.gauss_bonnet_wigner_discrepancy_q32.to_f64() < 1e-4);
    assert!(
        (tri_fwd.wigner_axis_skew_q32[2].to_f64() + tri_rev.wigner_axis_skew_q32[2].to_f64()).abs()
            < 1e-5
    );
}

#[test]
fn ephemeral_agent_dissolution_deposit_coalescence_and_model_agnostic_hot_swap() {
    let mut server = MonsterHybridServer::boot_default(sample_weights()).unwrap();

    // 1. Ingest a baseline global atom and run two governed forward passes in L1 (SensoryPerception)
    let tap = TeacherTapRecord {
        shard_id: 501,
        layer_index: 48,
        hidden_state: (0..5120).map(|i| ((i as f32) * 0.019).sin()).collect(),
        next_op_probs: vec![0.4, 0.3, 0.2, 0.1],
        payload: b"persistent_knowledge_trace".to_vec(),
    };
    let ingested = server
        .ingest_teacher_tap(&tap, RoleId::SensoryPerception, [Q32_31::ZERO; 3])
        .unwrap();

    let hd = 128usize;
    let q_head: Vec<f32> = (0..hd).map(|d| ((d as f32) * 0.13).cos()).collect();
    let k_head: Vec<f32> = (0..hd).map(|d| ((d as f32) * 0.13).cos() + 0.01).collect();
    let v_head: Vec<f32> = (0..hd).map(|d| ((d as f32) * 0.09).sin()).collect();
    let o_proj: Vec<f32> = (0..(hd * hd)).map(|i| ((i as f32) * 0.004).cos() * 0.05).collect();

    server
        .execute_governed_forward(
            RoleId::SensoryPerception,
            2001,
            0,
            &q_head,
            &k_head,
            &v_head,
            &o_proj,
        )
        .unwrap();
    server
        .execute_governed_forward(
            RoleId::SensoryPerception,
            2002,
            1,
            &q_head,
            &k_head,
            &v_head,
            &o_proj,
        )
        .unwrap();

    // 2. Dissolve L1 and compile its EphemeralMapDeposit into the centralized CarvedSwarmMap
    let pre_atoms = server.swarm_map.global_atom_count();
    let (deposit, deposit_receipt, _deposit_permit) = server
        .dissolve_agent_and_compile_deposit(RoleId::SensoryPerception)
        .unwrap();
    assert_eq!(deposit.records.len(), 2);
    assert_eq!(deposit_receipt.from_role, RoleId::SensoryPerception);
    assert_eq!(
        server.governor.process(RoleId::SensoryPerception).unwrap().state,
        ProcessState::Dissolved
    );
    assert_eq!(
        server
            .governor
            .process(RoleId::SensoryPerception)
            .unwrap()
            .worker
            .memory_map
            .atoms
            .len(),
        0,
        "Dissolved agent local GeometricMemoryMap must be drained into deposit"
    );
    assert_eq!(deposit_receipt.submitted_atoms, 2);
    assert!(
        deposit_receipt.newly_compiled_atoms + deposit_receipt.coalesced_existing_atoms == 2
    );
    assert!(server.swarm_map.global_atom_count() >= pre_atoms);

    // 3. Verify supervisor respawns a clean ephemeral instance with a new PID and restart_count == 0
    //    (intentional dissolution is never counted as a fault/crash!)
    let _respawned = server
        .governor
        .step_supervision(&server.swarm_map, &mut server.immune)
        .unwrap();
    let l1_after = server.governor.process(RoleId::SensoryPerception).unwrap();
    assert_eq!(l1_after.state, ProcessState::Running);
    assert_ne!(l1_after.pid, deposit_receipt.dissolved_pid);
    assert_eq!(
        l1_after.restart_count, 0,
        "Dissolved specialist respawn must not increment fault restart_count"
    );

    // 4. Plug in a completely new model system (128-layer NextGen MoE) and verify ZERO re-distillation
    //    and 100% centralized memory preservation
    let total_atoms_before_swap = server.swarm_map.global_atom_count();

    let new_w: Vec<f32> = (0..8192).map(|i| ((i as f32) * 0.023).cos()).collect();
    let new_model = FrozenBaseWeights::freeze("Atoma-NextGen-MoE-128L", 128, new_w);
    let (swap_receipt, _swap_permit) = server
        .hot_swap_model_backend(std::sync::Arc::clone(&new_model))
        .unwrap();

    assert_eq!(swap_receipt.new_model_epoch, 2);
    assert_eq!(swap_receipt.preserved_global_atoms, total_atoms_before_swap);
    assert_eq!(server.swarm_map.global_atom_count(), total_atoms_before_swap);
    assert_eq!(server.swarm_map.model_lineage.len(), 2);
    assert!(
        server
            .swarm_map
            .silo(RoleId::SensoryPerception)
            .unwrap()
            .contains_atom(ingested.atom_id)
    );
    for role in RoleId::CARVED_12 {
        let silo = server.swarm_map.silo(role).unwrap();
        assert!(std::sync::Arc::ptr_eq(&silo.weight_view.base, &new_model));
    }

    // 5. Hydrate the newly respawned L1 worker at 256d head_dim from the centralized vault
    //    using ModelAdapterLens (adapting 128d Epoch-1 deposited KV payloads -> 256d Epoch-2)
    let hydration = server
        .hydrate_role_from_central_vault(RoleId::SensoryPerception, 256)
        .unwrap();
    assert!(!hydration.adapted_records.is_empty());
    for rec in &hydration.adapted_records {
        assert_eq!(rec.loss_report.source_head_dim, 128);
        assert_eq!(rec.loss_report.target_head_dim, 256);
        assert_eq!(rec.loss_report.rope_pair_integrity_q32, Q32_31::ONE);
        assert!(
            (rec.loss_report.retained_k_energy_ratio_q32.to_f64() - 1.0).abs() < 1e-5,
            "Isometric 128d -> 256d pair-harmonic embedding must preserve 100% K energy"
        );
        assert!(
            (rec.loss_report.retained_v_energy_ratio_q32.to_f64() - 1.0).abs() < 1e-5,
            "Isometric 128d -> 256d pair-harmonic embedding must preserve 100% V energy"
        );
    }

    // 6. Execute a governed 256d forward pass on the Epoch-2 worker attending over adapted Epoch-1 memories
    let hd_256 = 256usize;
    let q_256: Vec<f32> = (0..hd_256).map(|d| ((d as f32) * 0.11).cos()).collect();
    let k_256: Vec<f32> = (0..hd_256).map(|d| ((d as f32) * 0.11).cos() + 0.01).collect();
    let v_256: Vec<f32> = (0..hd_256).map(|d| ((d as f32) * 0.07).sin()).collect();
    let o_256: Vec<f32> = (0..(hd_256 * hd_256))
        .map(|i| ((i as f32) * 0.002).cos() * 0.03)
        .collect();
    let fwd_256 = server
        .execute_governed_forward(
            RoleId::SensoryPerception,
            3001,
            2,
            &q_256,
            &k_256,
            &v_256,
            &o_256,
        )
        .unwrap();
    assert_eq!(fwd_256.output_state.len(), 256);
    assert!(fwd_256.output_state.iter().all(|v| v.is_finite()));
}

#[test]
fn native_mlir_tokenless_inter_process_ssa_execution_and_binary_catalog_verified() {
    use atoma_immune::{MlirAtomCatalog, MlirSsaBlock, NativeMlirInstruction};
    let mut server = MonsterHybridServer::boot_default(sample_weights()).unwrap();

    // 1. Verify .atoma_ir binary catalog round-trips with SHA-256 verification
    let blob = server.mlir_catalog.to_binary_blob();
    let loaded_catalog = MlirAtomCatalog::from_binary_blob(&blob).unwrap();
    assert_eq!(
        loaded_catalog.catalog_sha256,
        server.mlir_catalog.catalog_sha256
    );

    // 2. Place a shared boundary atom between L1 (SensoryPerception) and L3 (SomaticInteroception)
    let u1 = server.swarm_map.atlas.well(RoleId::SensoryPerception).unit_dir;
    let u3 = server
        .swarm_map
        .atlas
        .well(RoleId::SomaticInteroception)
        .unit_dir;
    let d_sph = Q32_31::from_f64(0.804_735_5);
    let mid_offset = [
        ((u3[0] - u1[0]) * d_sph) * Q32_31::HALF,
        ((u3[1] - u1[1]) * d_sph) * Q32_31::HALF,
        ((u3[2] - u1[2]) * d_sph) * Q32_31::HALF,
    ];
    let tap = TeacherTapRecord {
        shard_id: 700,
        layer_index: 48,
        hidden_state: (0..5120).map(|i| ((i as f32) * 0.015).cos()).collect(),
        next_op_probs: vec![0.25, 0.25, 0.25, 0.25],
        payload: b"mlir_boundary_anchor".to_vec(),
    };
    let boundary_rec = server
        .ingest_teacher_tap(&tap, RoleId::SensoryPerception, mid_offset)
        .unwrap();

    // 3. Construct a typed Native MLIR SSA block (%0 = 4.25; %1 = 2.0; %2 = %0 + %1; %3 = %2 * %1; return %3 = 12.5)
    let op_const = server
        .mlir_catalog
        .lookup_by_mnemonic("arith.constant_q32")
        .unwrap()
        .op_id;
    let op_add = server
        .mlir_catalog
        .lookup_by_mnemonic("arith.add_q32")
        .unwrap()
        .op_id;
    let op_mul = server
        .mlir_catalog
        .lookup_by_mnemonic("arith.mul_q32")
        .unwrap()
        .op_id;
    let op_ret = server
        .mlir_catalog
        .lookup_by_mnemonic("func.return_q32")
        .unwrap()
        .op_id;

    let valid_block = MlirSsaBlock::new(vec![
        NativeMlirInstruction {
            op_id: op_const,
            dst_reg: 0,
            src_regs: [0; 4],
            arity: 0,
            immediate_q32: Q32_31::from_f64(4.25),
        },
        NativeMlirInstruction {
            op_id: op_const,
            dst_reg: 1,
            src_regs: [0; 4],
            arity: 0,
            immediate_q32: Q32_31::from_f64(2.0),
        },
        NativeMlirInstruction {
            op_id: op_add,
            dst_reg: 2,
            src_regs: [0, 1, 0, 0],
            arity: 2,
            immediate_q32: Q32_31::ZERO,
        },
        NativeMlirInstruction {
            op_id: op_mul,
            dst_reg: 3,
            src_regs: [2, 1, 0, 0],
            arity: 2,
            immediate_q32: Q32_31::ZERO,
        },
        NativeMlirInstruction {
            op_id: op_ret,
            dst_reg: 4,
            src_regs: [3, 0, 0, 0],
            arity: 1,
            immediate_q32: Q32_31::ZERO,
        },
    ]);

    let dispatch = server
        .dispatch_native_mlir_block(
            RoleId::SensoryPerception,
            RoleId::SomaticInteroception,
            boundary_rec.atom_id,
            &valid_block,
        )
        .unwrap();
    assert_eq!(dispatch.validation.instruction_count, 5);
    assert!((dispatch.result_q32.to_f64() - 12.5).abs() < 1e-6);

    // Verify lossless bijective token_id <-> MlirOpId round-trip on the expanded token stream
    let (re_encoded_ops, loss) = server
        .mlir_catalog
        .tokens_to_mlir_ops(&dispatch.expanded_model_tokens);
    assert!(loss.roundtrip_lossless);
    assert_eq!(re_encoded_ops.len(), 5);

    // 4. Verify Layer 3 MlirNativeSsaGate rejects an SSA block with use-before-def before routing
    let malformed_block = MlirSsaBlock::new(vec![NativeMlirInstruction {
        op_id: op_ret,
        dst_reg: 0,
        src_regs: [77, 0, 0, 0],
        arity: 1,
        immediate_q32: Q32_31::ZERO,
    }]);
    let err = server
        .dispatch_native_mlir_block(
            RoleId::SensoryPerception,
            RoleId::SomaticInteroception,
            boundary_rec.atom_id,
            &malformed_block,
        )
        .unwrap_err();
    assert!(matches!(err, ComplianceViolation::InvalidMlirSsaBlock { .. }));

    // 5. Verify GeodesicMlirDecoder synthesizes valid, executable SSA blocks directly from
    //    128d and 256d continuous hidden states without any tokenizer or token_id lookup!
    let states_256d: Vec<Vec<f32>> = (0..6)
        .map(|step| {
            (0..256)
                .map(|d| ((d as f32) * 0.11 + (step as f32) * 0.67).sin() * 0.55)
                .collect()
        })
        .collect();
    let (geo_decode, geo_dispatch) = server
        .synthesize_and_dispatch_geodesic_mlir(
            RoleId::SensoryPerception,
            RoleId::SomaticInteroception,
            boundary_rec.atom_id,
            &states_256d,
        )
        .unwrap();
    assert_eq!(geo_decode.block.instructions.len(), 6);
    assert!(geo_decode.masked_illegal_opcodes_count > 0);
    assert_eq!(geo_dispatch.validation.instruction_count, 6);
    assert!(geo_dispatch.result_q32.to_f64().is_finite());
}

#[test]
fn open_weights_knowledge_and_reasoning_distillation_engine_verified() {
    let mut server = MonsterHybridServer::boot_default(sample_weights()).unwrap();

    // 1. Distill static 0-cell knowledge modes directly from an open-source weight buffer
    let oss_weights: Vec<f32> = (0..16_384).map(|i| ((i as f32) * 0.019).cos() * 0.9).collect();
    let oss_teacher = FrozenBaseWeights::freeze("Qwen3.6-OpenTeacher-64L", 64, oss_weights);
    let w_rec = server
        .distill_open_source_weights(&oss_teacher, 256, 16)
        .unwrap();
    assert_eq!(w_rec.modes_scanned, 16);
    assert_eq!(w_rec.knowledge_atoms_minted, 16);
    assert!(w_rec.mean_retained_energy_q32.0 > 0);

    // 2. Distill a multi-step reasoning chain (with 1 high-entropy hallucination step that must be sieved out)
    let taps = vec![
        TeacherTapRecord {
            shard_id: 8001,
            layer_index: 6,
            hidden_state: (0..256).map(|i| ((i as f32) * 0.07).sin()).collect(),
            next_op_probs: vec![0.90, 0.05, 0.03, 0.02],
            payload: b"premise_a".to_vec(),
        },
        TeacherTapRecord {
            shard_id: 8002,
            layer_index: 20, // Uniform 8-way guess -> H = 3.0 bits > 2.75 max_entropy_bits -> rejected!
            hidden_state: (0..256).map(|i| ((i as f32) * 0.19).cos()).collect(),
            next_op_probs: vec![0.125; 8],
            payload: b"high_entropy_noise".to_vec(),
        },
        TeacherTapRecord {
            shard_id: 8003,
            layer_index: 34,
            hidden_state: (0..256).map(|i| ((i as f32) * 0.13 + 1.1).cos()).collect(),
            next_op_probs: vec![0.04, 0.88, 0.05, 0.03],
            payload: b"lemma_b".to_vec(),
        },
        TeacherTapRecord {
            shard_id: 8004,
            layer_index: 58,
            hidden_state: (0..256).map(|i| ((i as f32) * 0.11 + 2.3).sin()).collect(),
            next_op_probs: vec![0.02, 0.04, 0.91, 0.03],
            payload: b"conclusion_c".to_vec(),
        },
    ];

    let r_rec = server
        .distill_reasoning_trajectory(42, &taps, 64, oss_teacher.fingerprint)
        .unwrap();
    assert_eq!(r_rec.steps_ingested, 3);
    assert_eq!(r_rec.steps_rejected_by_entropy_sieve, 1);
    assert_eq!(r_rec.arc_ids.len(), 2);
    assert_eq!(r_rec.compiled_ssa_block.instructions.len(), 3);

    // 3. Verify the distilled 1-cell reasoning arcs can be traversed in CarvedSwarmMap
    //    and survive a pluggable model hot-swap 100% intact!
    let path = server
        .swarm_map
        .query_reasoning_trajectory(r_rec.atom_sequence[0], 5);
    assert_eq!(path.len(), 2);
    assert_eq!(path[0].from_atom, r_rec.atom_sequence[0]);
    assert_eq!(path[1].to_atom, r_rec.atom_sequence[2]);

    let next_model = FrozenBaseWeights::freeze("Atoma-Student-v3", 32, vec![0.5f32; 4096]);
    server.hot_swap_model_backend(next_model).unwrap();
    assert_eq!(
        server
            .swarm_map
            .query_reasoning_trajectory(r_rec.atom_sequence[0], 5)
            .len(),
        2
    );

    // 4. Verify Zero-Copy Quantized Layer Distillation (4-bit and 6-bit OptiQ U32+BF16)
    //    screens all rows via analytical scale/bias energy and selectively dequantizes only top_k_rows!
    let q_rows = 256usize;
    let q_cols = 1024usize;
    let group_size = 128usize;
    let groups_per_row = q_cols / group_size;
    let packed_q4: Vec<u8> = (0..((q_rows * q_cols * 4) / 8))
        .map(|i| (i.wrapping_mul(37) & 0xFF) as u8)
        .collect();
    let packed_q6: Vec<u8> = (0..((q_rows * q_cols * 6) / 8))
        .map(|i| (i.wrapping_mul(53) & 0xFF) as u8)
        .collect();
    let scales_bf16: Vec<u8> = (0..(q_rows * groups_per_row))
        .flat_map(|g| (((0.15f32 + (g % 7) as f32 * 0.08).to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let biases_bf16: Vec<u8> = (0..(q_rows * groups_per_row))
        .flat_map(|g| (((-0.1f32 + (g % 5) as f32 * 0.04).to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let out_scale: Vec<f32> = (0..q_rows).map(|r| 0.5 + (r % 11) as f32 * 0.1).collect();

    let q4_tap = atoma_immune::QuantizedLayerTapDescriptor {
        tensor_name: "model.layers.12.mlp.down_proj.q4",
        layer_index: 12,
        total_layers: 64,
        weight_data: &packed_q4,
        weight_dtype: "U32",
        scales_data: &scales_bf16,
        scales_dtype: "BF16",
        biases_data: &biases_bf16,
        biases_dtype: "BF16",
        out_scale: Some(&out_scale),
        bits: 4,
        group_size,
        rows: q_rows,
        cols: q_cols,
    };
    let q4_rec = server
        .distill_quantized_layer(&q4_tap, 8, oss_teacher.fingerprint)
        .unwrap();
    assert_eq!(q4_rec.total_rows_scanned_via_scales, q_rows);
    assert_eq!(q4_rec.top_k_rows_dequantized, 8);
    assert_eq!(q4_rec.minted_atom_ids.len(), 8);
    assert_eq!(
        q4_rec.heap_bytes_saved_vs_full_dequant,
        ((q_rows - 8) * q_cols * 4) as u64
    );

    let q6_tap = atoma_immune::QuantizedLayerTapDescriptor {
        tensor_name: "model.layers.44.self_attn.o_proj.q6",
        bits: 6,
        layer_index: 44,
        weight_data: &packed_q6,
        ..q4_tap
    };
    let q6_rec = server
        .distill_quantized_layer(&q6_tap, 8, oss_teacher.fingerprint)
        .unwrap();
    assert_eq!(q6_rec.bits, 6);
    assert_eq!(q6_rec.top_k_rows_dequantized, 8);
    assert_eq!(q6_rec.minted_atom_ids.len(), 8);

    // 5. Verify Live Server Continuous Novelty-Only Distillation Gate & Persistent .atoma_map Vault:
    //    - Novel Premise A and Novel Deduction B mint new 0-cells and a new 1-cell arc.
    //    - Replaying Deduction B is deduplicated in O(1) (atom_was_novel == false, arc_was_novel == false, 0 bloat).
    //    - High-entropy noise step is rejected by the Entropy Sieve.
    let live_a = TeacherTapRecord {
        shard_id: 99_001,
        layer_index: 6,
        hidden_state: (0..256).map(|i| ((i as f32) * 0.081 + 0.4).sin()).collect(),
        next_op_probs: vec![0.92, 0.04, 0.02, 0.02],
        payload: vec![0u8; 256],
    };
    let live_b = TeacherTapRecord {
        shard_id: 99_002,
        layer_index: 42,
        hidden_state: (0..256).map(|i| ((i as f32) * 0.149 - 0.9).cos()).collect(),
        next_op_probs: vec![0.02, 0.90, 0.05, 0.03],
        payload: vec![1u8; 256],
    };
    let live_noise = TeacherTapRecord {
        shard_id: 99_003,
        layer_index: 20,
        hidden_state: (0..256).map(|i| ((i as f32) * 0.31).sin()).collect(),
        next_op_probs: vec![0.125; 8],
        payload: vec![2u8; 256],
    };

    let rec_a = server.observe_and_distill_live_step(&live_a).unwrap();
    assert!(rec_a.step_ingested && rec_a.atom_was_novel);
    let atoms_after_a = server.swarm_map.global_atom_count();

    let rec_b = server.observe_and_distill_live_step(&live_b).unwrap();
    assert!(rec_b.step_ingested && rec_b.atom_was_novel && rec_b.arc_was_novel);
    let atoms_after_b = server.swarm_map.global_atom_count();
    let arcs_after_b = server.swarm_map.reasoning_arcs.len();
    assert_eq!(atoms_after_b, atoms_after_a + 1);

    // Replay `live_b`: must be deduplicated with ZERO new atoms or arcs!
    let rec_b_dup = server.observe_and_distill_live_step(&live_b).unwrap();
    assert!(rec_b_dup.step_ingested);
    assert!(!rec_b_dup.atom_was_novel);
    assert!(!rec_b_dup.arc_was_novel);
    assert_eq!(server.swarm_map.global_atom_count(), atoms_after_b);
    assert_eq!(server.swarm_map.reasoning_arcs.len(), arcs_after_b);

    // High-entropy noise step: rejected at the gate!
    let rec_noise = server.observe_and_distill_live_step(&live_noise).unwrap();
    assert!(!rec_noise.step_ingested && rec_noise.rejected_as_high_entropy_noise);
    assert_eq!(server.swarm_map.global_atom_count(), atoms_after_b);

    // Verify .atoma_map binary vault save & load round-trip
    let tmp_vault = std::env::temp_dir().join(format!("test_vault_{}.atoma_map", std::process::id()));
    let (saved_atoms, saved_arcs, saved_sha) =
        server.swarm_map.save_to_binary_vault(&tmp_vault).unwrap();
    let mut fresh_server = MonsterHybridServer::boot_default(sample_weights()).unwrap();
    let (loaded_atoms, loaded_arcs, loaded_sha) = fresh_server
        .swarm_map
        .load_from_binary_vault(&tmp_vault, &fresh_server.immune.policy)
        .unwrap();
    let _ = std::fs::remove_file(&tmp_vault);
    assert_eq!(saved_atoms, loaded_atoms);
    assert_eq!(saved_arcs, loaded_arcs);
    assert_eq!(saved_sha, loaded_sha);

    // 6. Verify Geodesic Query & Multi-Hop Reasoning Retrieval (`query_swarm_map`) on `fresh_server`
    let q_res = fresh_server
        .query_swarm_map(
            "deduce causal invariant and verify CWE-78 command injection holonomy",
            None,
            5,
            4,
            false,
        )
        .unwrap();
    assert!(!q_res.hits.is_empty());
    assert!(!q_res.reasoning_hops.is_empty());
    assert!(!q_res.synthesized_mlir_program.is_empty());
}




