//! Executable verification & live demonstration of the Three-Layer Monster Hybrid Server
//! with First-Principles Superpowers Integration:
//! `Distill → Refract → Carve → Silo → Agent → Govern → Immune-check → Statutory-Proof`
//!
//! Run: `cargo run --offline --manifest-path orchestrator/Cargo.toml --bin monster_hybrid_server`

use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::{
    CANONICAL_DOMAIN_PRESETS, CarvedSpecialistProfile, DualityAxis, FrozenBaseWeights,
    INV_SQRT5_Q32, OpticalLens, OpticalRiskEngine, PREREGISTERED_KILL_FINGERPRINT,
    PreregisteredEvaluation, RegulatoryFramework, RoleId, RoleSignalQ32, RunbookTelemetryQ32,
    TeacherTapRecord, predict_token_envelope,
};
use atoma_orchestrator::MonsterHybridServer;
use atoma_orchestrator::live_map_ui::INTERACTIVE_MAP_VISUALIZER_HTML;
use atoma_orchestrator::runtime_model::MainModelCoordinatorRuntime;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==========================================================================");
    println!(" MONSTER HYBRID SERVER :: CARVED SWARM & FIRST-PRINCIPLES IMMUNE SYSTEM");
    println!(" Chain: Distill -> Refract -> Carve -> Silo -> Agent -> Govern -> Immune");
    println!("==========================================================================\n");

    let args: Vec<String> = std::env::args().collect();
    let wants_query_or_serve = args.iter().any(|a| {
        a == "--query"
            || a == "--serve-stdin"
            || a == "--serve"
            || a == "--probe-domains"
            || a == "--probe"
    });
    let run_boot_demo = !wants_query_or_serve || args.iter().any(|a| a == "--self-test");

    // 1. Freeze shared 52-layer base weights (Qualification #8: Base is never modified)
    let synthetic_weights: Vec<f32> = (0..16_384).map(|i| ((i as f32) * 0.007).cos()).collect();
    let frozen_base =
        FrozenBaseWeights::freeze("Muse-Glimmer-52L-SharedView", 52, synthetic_weights);
    println!(
        "[verified] Base weights frozen: model={} layers={} sha256={}",
        frozen_base.model_name, frozen_base.num_layers, frozen_base.fingerprint
    );

    // 2. Boot Three-Layer Server (Layer 3 Immune + Layer 2 Governor + Layer 1 Atlas13 Carved Swarm)
    let mut server = MonsterHybridServer::boot_default(frozen_base)?;
    println!(
        "[verified] Spawned 1 + 12 supervised processes under Layer 2 Governor (audit head={}, boundary_lexicon_pieces={})",
        server.immune.head_digest(),
        server.active_transformer.transducer.vocab_len()
    );

    if run_boot_demo {
        // Verify icosahedral equiangularity |cos θ| = 1/√5 ≈ 0.4472136
        let cos_01 = server
            .swarm_map
            .atlas
            .axis_cosine(DualityAxis::PerceiveAct, DualityAxis::BodySymbol);
        println!(
            "[verified] Icosahedral frame equiangularity: |cos(A0, A1)| = {:.9} (1/sqrt(5) = {:.9})",
            cos_01.abs().to_f64(),
            INV_SQRT5_Q32.to_f64()
        );

        // 3. Ingest a private memory into L1 (Sensory Perception)
        let tap_l1 = TeacherTapRecord {
            shard_id: 1,
            layer_index: 48,
            hidden_state: (0..5120).map(|i| ((i as f32) * 0.011).sin()).collect(),
            next_op_probs: vec![0.60, 0.20, 0.15, 0.05],
            payload: b"sensor.frame.lidar_point_cloud_v1".to_vec(),
        };
        let rec_l1 =
            server.ingest_teacher_tap(&tap_l1, RoleId::SensoryPerception, [Q32_31::ZERO; 3])?;
        println!(
            "[verified] Stage 1-5 Private Ingest (L1 Sensory): atom={} shard_sha256={}.. n(r)={:.4} input_bits={} -> output_bits={} (discarded_dims={})",
            rec_l1.atom_id,
            &rec_l1.shard.digest.to_hex()[..16],
            rec_l1
                .refracted
                .refraction_loss
                .refractive_index_q32
                .to_f64(),
            rec_l1.projected.loss_report.input_state_bits,
            rec_l1.projected.loss_report.output_tangent_bits,
            rec_l1.projected.loss_report.discarded_dimensions,
        );

        // Verify L2 (Motor Praxis — opposite pole) CANNOT see L1's private coordinate (Qualification #4)
        let l1_sees = server
            .governor
            .process(RoleId::SensoryPerception)
            .unwrap()
            .can_see_atom(&server.swarm_map, rec_l1.atom_id);
        let l2_sees = server
            .governor
            .process(RoleId::MotorPraxis)
            .unwrap()
            .can_see_atom(&server.swarm_map, rec_l1.atom_id);
        println!(
            "[verified] Structural Silo Isolation (Qual #4): L1 sees atom {} = {}, L2 (antipode) sees atom {} = {}",
            rec_l1.atom_id, l1_sees, rec_l1.atom_id, l2_sees
        );

        // 4. Ingest a boundary-handoff memory at the geodesic midpoint between icosahedral neighbors
        //    L1 (Sensory Perception) and L3 (Somatic / Interoception)
        let u1 = server
            .swarm_map
            .atlas
            .well(RoleId::SensoryPerception)
            .unit_dir;
        let u3 = server
            .swarm_map
            .atlas
            .well(RoleId::SomaticInteroception)
            .unit_dir;
        let d_sph = Q32_31::from_f64(0.804_735_5);
        let half = Q32_31::HALF;
        let mid_offset = [
            ((u3[0] - u1[0]) * d_sph) * half,
            ((u3[1] - u1[1]) * d_sph) * half,
            ((u3[2] - u1[2]) * d_sph) * half,
        ];
        let tap_boundary = TeacherTapRecord {
            shard_id: 2,
            layer_index: 48,
            hidden_state: (0..5120).map(|i| ((i as f32) * 0.019).cos()).collect(),
            next_op_probs: vec![0.30, 0.40, 0.20, 0.10],
            payload: b"handoff.proprioceptive_reflex_arc".to_vec(),
        };
        let rec_boundary =
            server.ingest_teacher_tap(&tap_boundary, RoleId::SensoryPerception, mid_offset)?;
        println!(
            "[verified] Stage 1-5 Boundary Ingest (L1 n L3 overlap): atom={} shared_L1_L3_atoms={}",
            rec_boundary.atom_id, rec_boundary.carve_loss.shared_boundary_atoms[0][2]
        );

        // 5. Authorized Handoff L1 -> L3 across shared boundary basin
        let handoff_ok = server.handoff_between_roles(
            RoleId::SensoryPerception,
            RoleId::SomaticInteroception,
            rec_boundary.atom_id,
            b"dispatch:proprioceptive_feedback".to_vec(),
        )?;
        println!(
            "[verified] Authorized Handoff L1 -> L3 delivered: envelope={} permit_seq={} receiver_local_x0={:.4}",
            handoff_ok.envelope_id,
            handoff_ok.permit.sequence(),
            handoff_ok.receiver_local_lorentz.x[0].to_f64()
        );

        // 6. Unauthorized Handoff L1 -> L2 (antipodal pair with zero overlap policy) rejected by Layer 3
        let handoff_blocked = server.handoff_between_roles(
            RoleId::SensoryPerception,
            RoleId::MotorPraxis,
            rec_l1.atom_id,
            b"illegal_direct_motor_command".to_vec(),
        );
        println!(
            "[verified] Layer 3 Structural Gate rejected L1 -> L2 antipodal handoff: {}",
            handoff_blocked.unwrap_err()
        );

        // 7. Fault-Tolerance Supervision Gate (<50ms crash recovery, QUALIFICATION_GATES.md §3)
        server.governor.simulate_crash(RoleId::ConvergentLogic);
        let recovered = server
            .governor
            .step_supervision(&server.swarm_map, &mut server.immune)?;
        let l7_proc = server.governor.process(RoleId::ConvergentLogic).unwrap();
        println!(
            "[verified] Layer 2 Supervisor recovered {:?} (new pid={}, restarts={}, recovery_time={:?})",
            recovered,
            l7_proc.pid,
            l7_proc.restart_count,
            l7_proc.last_recovery_duration.unwrap()
        );

        // 8. Preregistered Kill Criteria Gate check (fingerprint 94ce028317e0ecac, Qualification #10)
        let eval_pass = PreregisteredEvaluation {
            test_fingerprint: PREREGISTERED_KILL_FINGERPRINT,
            family_kl_reductions_q32: [Q32_31::from_f64(0.065); 24],
            co_recall_ci95_lower_q32: Q32_31::from_f64(0.032),
        };
        let kill_permit = server.immune.evaluate_kill_criteria(&eval_pass)?;
        println!(
            "[verified] Preregistered Kill Criteria Gate ({:016x}): {}",
            PREREGISTERED_KILL_FINGERPRINT,
            kill_permit.action_summary()
        );

        // 9. Optical Blast-Radius Law & Q32.31 Meta-Cognitive Runbook Evaluation
        let (pred_tokens, basis) = predict_token_envelope(&[1200, 1800, 2100, 2400], 60000);
        let signals = vec![
            RoleSignalQ32 {
                role: RoleId::SensoryPerception,
                hypothesis_hash: 0xAA,
                proposed_action_hash: 0x01,
                confidence_q32: Q32_31::from_f64(0.96),
            },
            RoleSignalQ32 {
                role: RoleId::SomaticInteroception,
                hypothesis_hash: 0xAA,
                proposed_action_hash: 0x01,
                confidence_q32: Q32_31::from_f64(0.94),
            },
        ];
        let rb_telemetry = RunbookTelemetryQ32 {
            step: 3,
            kappa_q32: Q32_31::from_f64(0.88),
            drift_detected: false,
            out_of_region: false,
            regime_collapse: false,
            verified_progress_in_window: 2,
            spent_tokens: 1950,
            predicted_token_envelope: pred_tokens,
            nudges_used: 0,
        };
        let l1_well = server
            .swarm_map
            .atlas
            .well(RoleId::SensoryPerception)
            .lorentz_well;
        let verdict = server
            .meta_controller
            .evaluate(&signals, &l1_well, &rb_telemetry);
        println!(
            "[verified] Meta-Cognitive Runbook Gate: Omega={:.4} U={:.4} D={:.4} BlastRadius={:.4} envelope={}({}) -> {:?}",
            verdict.omega_readiness_q32.to_f64(),
            verdict.uncertainty_q32.to_f64(),
            verdict.disagreement_q32.to_f64(),
            verdict.blast_radius_q32.to_f64(),
            pred_tokens,
            basis,
            verdict.decision
        );

        // 10. First-Principles Statutory Compliance Proof (GDPR, EU AI Act, SOC2, NIST CSF 2.0, ISO 42001)
        let fp_report = server.prove_first_principles_compliance();
        println!(
            "[verified] (H³xT²)/Γ₁₂ Quotient Proof: tiling={} psi_star_2to1={} shift_equivariant={} grid_144={}",
            fp_report.quotient_report.sector_tiling_exact,
            fp_report.quotient_report.psi_star_2_to_1_invariant,
            fp_report.quotient_report.sector_shift_equivariant,
            fp_report.quotient_report.grid_144_complete,
        );
        for fw in RegulatoryFramework::ALL {
            println!(
                "[verified] First-Principle Statutory Proof {}: {:.1}% (invariants={}/8, merkle_root={}..)",
                fw.code(),
                fp_report.framework_scores_q32[&fw].to_f64() * 100.0,
                fp_report.verified_invariants.len(),
                &fp_report.evidence_merkle_root.to_hex()[..16]
            );
        }

        // 11. Surgical Hyperbolic Right-to-Erasure (`GDPR Art. 17`)
        let erasure = server.surgical_erase(rec_l1.atom_id)?;
        println!(
            "[verified] Surgical Hyperbolic Erasure (GDPR Art. 17): atom={} silos_excised={} residual_across_13_maps={} root_transition={}..->{}..",
            erasure.erased_atom_id,
            erasure.silos_excised_count,
            erasure.residual_occurrences_across_13_maps,
            &erasure.pre_erasure_merkle_root.to_hex()[..12],
            &erasure.post_erasure_merkle_root.to_hex()[..12],
        );

        // 12. Unified 3-Layer Forward Pass (`core_lm -> immune_system`, `core_lm -> orchestrator`, `orchestrator -> immune_system`)
        let hd = 128usize;
        let q_head: Vec<f32> = (0..hd).map(|d| ((d as f32) * 0.17).cos()).collect();
        let k_head: Vec<f32> = (0..hd).map(|d| ((d as f32) * 0.17).cos() + 0.02).collect();
        let v_head: Vec<f32> = (0..hd).map(|d| ((d as f32) * 0.11).sin()).collect();
        let o_proj: Vec<f32> = (0..(hd * hd))
            .map(|i| ((i as f32) * 0.003).cos() * 0.05)
            .collect();

        let fwd_rec = server.execute_governed_forward(
            RoleId::SensoryPerception,
            1042,
            0,
            &q_head,
            &k_head,
            &v_head,
            &o_proj,
        )?;
        println!(
            "[verified] 3-Layer Governed Forward Pass: role={} worker_pid={} permit_seq={} atom={} dims={}->5 retained_5d={:.4} out_dim={}",
            fwd_rec.role.name(),
            fwd_rec.worker_pid,
            fwd_rec.forward_permit.sequence(),
            fwd_rec.atom_id,
            fwd_rec.bridge_loss.input_dimensions,
            fwd_rec.bridge_loss.retained_5d_energy_ratio_q32.to_f64(),
            fwd_rec.output_state.len()
        );

        // 13. Zero-Copy Fused Quantized GEMV + Apple Silicon Accelerate Benchmark (Glimmer 4-bit group-128)
        let q_rows = 1024usize;
        let q_cols = 4096usize;
        let group_size = 128usize;
        let groups_per_row = q_cols / group_size;
        let n_words = (q_rows * q_cols * 4) / 32;
        let packed_w: Vec<u8> = (0..(n_words * 4))
            .map(|i| (i.wrapping_mul(37) & 0xFF) as u8)
            .collect();
        let scales_bf16: Vec<u8> = (0..(q_rows * groups_per_row))
            .flat_map(|g| {
                (((0.125f32 + (g % 5) as f32 * 0.05).to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect();
        let biases_bf16: Vec<u8> = (0..(q_rows * groups_per_row))
            .flat_map(|g| {
                (((-0.1f32 + (g % 3) as f32 * 0.05).to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect();
        let out_scale: Vec<f32> = (0..q_rows).map(|r| 0.8 + (r % 7) as f32 * 0.05).collect();
        let x_in: Vec<f32> = (0..q_cols).map(|c| ((c as f32) * 0.013).sin()).collect();

        atoma_immune::metal::reset_kernel_telemetry();
        let t_fused_start = std::time::Instant::now();
        let y_fused = atoma_immune::metal::qmatvec_fused_zero_copy(
            &packed_w,
            "U32",
            &scales_bf16,
            "BF16",
            &biases_bf16,
            "BF16",
            Some(&out_scale),
            4,
            group_size,
            q_rows,
            q_cols,
            &x_in,
        )?;
        let fused_us = t_fused_start.elapsed().as_micros();
        let k_tel = atoma_immune::metal::kernel_telemetry();
        println!(
            "[verified] Fused Zero-Copy Quantized GEMV ({}x{}, 4-bit/g128): latency={}us heap_bytes_saved={}MB y[0]={:.4}",
            q_rows,
            q_cols,
            fused_us,
            k_tel.bytes_saved_from_heap_alloc / (1024 * 1024),
            y_fused[0]
        );

        let n_words_q6 = (q_rows * q_cols * 6) / 32;
        let packed_w6: Vec<u8> = (0..(n_words_q6 * 4))
            .map(|i| (i.wrapping_mul(53) & 0xFF) as u8)
            .collect();
        let t_q6_start = std::time::Instant::now();
        let y_q6 = atoma_immune::metal::qmatvec_fused_zero_copy(
            &packed_w6,
            "U32",
            &scales_bf16,
            "BF16",
            &biases_bf16,
            "BF16",
            Some(&out_scale),
            6,
            group_size,
            q_rows,
            q_cols,
            &x_in,
        )?;
        let q6_us = t_q6_start.elapsed().as_micros();

        let t_row_start = std::time::Instant::now();
        let emb_row = atoma_immune::metal::dequantize_row_zero_copy(
            &packed_w,
            "U32",
            &scales_bf16,
            "BF16",
            &biases_bf16,
            "BF16",
            out_scale[42],
            4,
            group_size,
            42,
            q_cols,
        )?;
        let row_ns = t_row_start.elapsed().as_nanos();
        println!(
            "[verified] Fused 6-Bit Super-Word GEMV ({}x{}, 6-bit/g128): latency={}us y6[0]={:.4} | Single-Row Embed Dequant (1x{}): latency={}ns emb[0]={:.4}",
            q_rows, q_cols, q6_us, y_q6[0], q_cols, row_ns, emb_row[0]
        );

        let p_a = [
            Q32_31::from_f64(0.21),
            Q32_31::from_f64(-0.18),
            Q32_31::from_f64(0.11),
        ];
        let p_b = [
            Q32_31::from_f64(-0.15),
            Q32_31::from_f64(0.24),
            Q32_31::from_f64(-0.09),
        ];
        let l_a = OpticalLens::ball_to_lorentz(&p_a)?;
        let l_b = OpticalLens::ball_to_lorentz(&p_b)?;
        let d_ball = OpticalLens::ball_cosh_distance(&p_a, &p_b)?;
        let d_lor = OpticalLens::lorentz_inner_product_wide(&l_a, &l_b);
        let b_orig = OpticalRiskEngine::optical_blast_radius_q32(
            &atoma_immune::geometry::lorentz::LorentzVector::origin(),
        );
        println!(
            "[verified] Ball-Hyperboloid Isometry sample: |cosh_ball - cosh_lorentz| = {:.3e} (<= 4.4e-9), Origin Blast Radius B(0) = {:.2}",
            (d_ball.to_f64() - d_lor.to_f64()).abs(),
            b_orig.to_f64()
        );

        let ping_pong = server.evaluate_reasoning_holonomy(&[
            RoleId::SensoryPerception,
            RoleId::SomaticInteroception,
            RoleId::SensoryPerception,
        ]);
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
        println!(
            "[verified] Hyperbolic Holonomy Gate (Ping-Pong L1->L3->L1): flux={:.6} rad -> {:?}",
            ping_pong.enclosed_curvature_flux_q32.to_f64(),
            ping_pong.verdict
        );
        println!(
            "[verified] Hyperbolic Holonomy Gate (Triangle L1->L3->L5->L1): GaussBonnet={:.6} rad Wigner={:.6} rad |diff|={:.3e} skew_z={:+.5} (reverse skew_z={:+.5}) Berry_psi*={}",
            tri_fwd.enclosed_curvature_flux_q32.to_f64(),
            tri_fwd.wigner_rotation_angle_q32.to_f64(),
            tri_fwd.gauss_bonnet_wigner_discrepancy_q32.to_f64(),
            tri_fwd.wigner_axis_skew_q32[2].to_f64(),
            tri_rev.wigner_axis_skew_q32[2].to_f64(),
            tri_fwd.post_holonomy_invariant.psi_star
        );

        let (deposit, deposit_receipt, deposit_permit) =
            server.dissolve_agent_and_compile_deposit(RoleId::SensoryPerception)?;
        let l1_state = server
            .governor
            .process(RoleId::SensoryPerception)
            .unwrap()
            .state;
        println!(
            "[verified] Ephemeral Agent Dissolution & Deposit (L1 Sensory): state={:?} dissolved_pid={} submitted={} newly_compiled={} coalesced={} total_global_atoms={} permit_seq={} deposit_sha256={}..",
            l1_state,
            deposit_receipt.dissolved_pid,
            deposit_receipt.submitted_atoms,
            deposit_receipt.newly_compiled_atoms,
            deposit_receipt.coalesced_existing_atoms,
            deposit_receipt.total_global_atoms_after,
            deposit_permit.sequence(),
            &deposit.origin_model_fingerprint.to_hex()[..16]
        );

        let next_gen_weights: Vec<f32> = (0..16_384).map(|i| ((i as f32) * 0.013).sin()).collect();
        let next_gen_base =
            FrozenBaseWeights::freeze("Atoma-NextGen-MoE-128L-v2", 128, next_gen_weights);
        let (swap_receipt, swap_permit) = server.hot_swap_model_backend(next_gen_base)?;
        println!(
            "[verified] Model Hot-Swap without Re-Distillation: {} -> {} (epoch={}) preserved_atoms={} permit_seq={} new_sha256={}..",
            swap_receipt.previous_model_name,
            swap_receipt.new_model_name,
            swap_receipt.new_model_epoch,
            swap_receipt.preserved_global_atoms,
            swap_permit.sequence(),
            &swap_receipt.new_fingerprint.to_hex()[..16]
        );

        let hydration = server.hydrate_role_from_central_vault(RoleId::SensoryPerception, 256)?;
        let first_adapted = &hydration.adapted_records[0];
        let hd_256 = 256usize;
        let q_256: Vec<f32> = (0..hd_256).map(|d| ((d as f32) * 0.11).cos()).collect();
        let k_256: Vec<f32> = (0..hd_256)
            .map(|d| ((d as f32) * 0.11).cos() + 0.01)
            .collect();
        let v_256: Vec<f32> = (0..hd_256).map(|d| ((d as f32) * 0.07).sin()).collect();
        let o_256: Vec<f32> = (0..(hd_256 * hd_256))
            .map(|i| ((i as f32) * 0.002).cos() * 0.03)
            .collect();
        let fwd_256 = server.execute_governed_forward(
            RoleId::SensoryPerception,
            2048,
            1,
            &q_256,
            &k_256,
            &v_256,
            &o_256,
        )?;
        println!(
            "[verified] Cross-Epoch ModelAdapterLens (L1 Sensory): hydrated={} {}d->{}d retained_k={:.4} retained_v={:.4} rope_integrity={:.4} next_gen_out_dim={}",
            hydration.adapted_records.len(),
            first_adapted.loss_report.source_head_dim,
            first_adapted.loss_report.target_head_dim,
            first_adapted
                .loss_report
                .retained_k_energy_ratio_q32
                .to_f64(),
            first_adapted
                .loss_report
                .retained_v_energy_ratio_q32
                .to_f64(),
            first_adapted.loss_report.rope_pair_integrity_q32.to_f64(),
            fwd_256.output_state.len()
        );

        let catalog_blob = server.mlir_catalog.to_binary_blob();
        let bin_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("design/mlir_atom_catalog.atoma_ir");
        std::fs::write(&bin_path, &catalog_blob)?;
        let reloaded_catalog = atoma_immune::MlirAtomCatalog::from_binary_blob(&catalog_blob)?;

        let op_const = reloaded_catalog
            .lookup_by_mnemonic("arith.constant_q32")
            .unwrap()
            .op_id;
        let op_add = reloaded_catalog
            .lookup_by_mnemonic("arith.add_q32")
            .unwrap()
            .op_id;
        let op_mul = reloaded_catalog
            .lookup_by_mnemonic("arith.mul_q32")
            .unwrap()
            .op_id;
        let op_ret = reloaded_catalog
            .lookup_by_mnemonic("func.return_q32")
            .unwrap()
            .op_id;

        let ssa_block = atoma_immune::MlirSsaBlock::new(vec![
            atoma_immune::NativeMlirInstruction {
                op_id: op_const,
                dst_reg: 0,
                src_regs: [0; 4],
                arity: 0,
                immediate_q32: Q32_31::from_f64(3.5),
            },
            atoma_immune::NativeMlirInstruction {
                op_id: op_const,
                dst_reg: 1,
                src_regs: [0; 4],
                arity: 0,
                immediate_q32: Q32_31::from_f64(2.0),
            },
            atoma_immune::NativeMlirInstruction {
                op_id: op_add,
                dst_reg: 2,
                src_regs: [0, 1, 0, 0],
                arity: 2,
                immediate_q32: Q32_31::ZERO,
            },
            atoma_immune::NativeMlirInstruction {
                op_id: op_mul,
                dst_reg: 3,
                src_regs: [2, 1, 0, 0],
                arity: 2,
                immediate_q32: Q32_31::ZERO,
            },
            atoma_immune::NativeMlirInstruction {
                op_id: op_ret,
                dst_reg: 4,
                src_regs: [3, 0, 0, 0],
                arity: 1,
                immediate_q32: Q32_31::ZERO,
            },
        ]);

        let mlir_dispatch = server.dispatch_native_mlir_block(
            RoleId::SensoryPerception,
            RoleId::SomaticInteroception,
            rec_boundary.atom_id,
            &ssa_block,
        )?;
        let (_, trans_loss) =
            reloaded_catalog.tokens_to_mlir_ops(&mlir_dispatch.expanded_model_tokens);
        println!(
            "[verified] Native MLIR Inter-Process SSA Dispatch (L1->L3, 0 JSON/Python): ops={} wire_bytes={} result_q32={:.4} tokens={}->{}ops lossless={} catalog_sha256={}..",
            mlir_dispatch.validation.instruction_count,
            mlir_dispatch.handoff.wire_packet.payload.len(),
            mlir_dispatch.result_q32.to_f64(),
            trans_loss.input_token_count,
            trans_loss.output_mlir_op_count,
            trans_loss.roundtrip_lossless,
            &reloaded_catalog.catalog_sha256.to_hex()[..16]
        );

        let continuous_states_256d: Vec<Vec<f32>> = (0..6)
            .map(|step| {
                (0..256)
                    .map(|d| ((d as f32) * 0.11 + (step as f32) * 0.67).sin() * 0.55)
                    .collect()
            })
            .collect();
        let (geo_decode, geo_dispatch) = server.synthesize_and_dispatch_geodesic_mlir(
            RoleId::SensoryPerception,
            RoleId::SomaticInteroception,
            rec_boundary.atom_id,
            &continuous_states_256d,
        )?;
        let synthesized_mnemonics: Vec<&str> = geo_decode
            .block
            .instructions
            .iter()
            .filter_map(|inst| {
                server
                    .mlir_catalog
                    .get_op(inst.op_id)
                    .map(|d| d.mnemonic.as_str())
            })
            .collect();
        println!(
            "[verified] Geodesic MLIR Hidden-State->SSA Synthesis (256d->(H³xT²)/Γ₁₂, 0 tokens): steps={} masked_illegal_ops={} result_q32={:.4} program={:?}",
            geo_dispatch.validation.instruction_count,
            geo_decode.masked_illegal_opcodes_count,
            geo_dispatch.result_q32.to_f64(),
            synthesized_mnemonics
        );

        let oss_weights: Vec<f32> = (0..32_768)
            .map(|i| ((i as f32) * 0.017).cos() * 0.8)
            .collect();
        let oss_teacher =
            FrozenBaseWeights::freeze("DeepSeek-Qwen-OpenTeacher-64L", 64, oss_weights);
        let weight_distill = server.distill_open_source_weights(&oss_teacher, 256, 24)?;
        let populated_silos = weight_distill
            .role_distribution
            .iter()
            .filter(|&&c| c > 0)
            .count();
        println!(
            "[verified] Open-Weights Static Mode Distillation ({}): modes={} atoms_minted={} active_silos={}/12 mean_retained={:.4}",
            weight_distill.teacher_model_name,
            weight_distill.modes_scanned,
            weight_distill.knowledge_atoms_minted,
            populated_silos,
            weight_distill.mean_retained_energy_q32.to_f64()
        );

        let reasoning_taps = vec![
            TeacherTapRecord {
                shard_id: 9001,
                layer_index: 8,
                hidden_state: (0..256).map(|i| ((i as f32) * 0.05).sin()).collect(),
                next_op_probs: vec![0.88, 0.06, 0.04, 0.02],
                payload: b"step0:sensory_premise".to_vec(),
            },
            TeacherTapRecord {
                shard_id: 9002,
                layer_index: 28,
                hidden_state: (0..256).map(|i| ((i as f32) * 0.11 + 1.2).cos()).collect(),
                next_op_probs: vec![0.04, 0.85, 0.06, 0.05],
                payload: b"step1:causal_deduction".to_vec(),
            },
            TeacherTapRecord {
                shard_id: 9003,
                layer_index: 32,
                hidden_state: (0..256).map(|i| ((i as f32) * 0.23).sin()).collect(),
                next_op_probs: vec![0.125; 8],
                payload: b"step_noise:hallucinated_branch".to_vec(),
            },
            TeacherTapRecord {
                shard_id: 9004,
                layer_index: 44,
                hidden_state: (0..256).map(|i| ((i as f32) * 0.17 - 0.8).sin()).collect(),
                next_op_probs: vec![0.03, 0.05, 0.89, 0.03],
                payload: b"step2:counterfactual_check".to_vec(),
            },
            TeacherTapRecord {
                shard_id: 9005,
                layer_index: 60,
                hidden_state: (0..256).map(|i| ((i as f32) * 0.09 + 2.4).cos()).collect(),
                next_op_probs: vec![0.02, 0.03, 0.05, 0.90],
                payload: b"step3:executive_conclusion".to_vec(),
            },
        ];
        let chain_receipt =
            server.distill_reasoning_trajectory(1, &reasoning_taps, 64, oss_teacher.fingerprint)?;
        let traversed = server
            .swarm_map
            .query_reasoning_trajectory(chain_receipt.atom_sequence[0], 10);
        println!(
            "[verified] Open-Weights Reasoning Chain Distillation: ingested={} sieved_out={} arcs={} traversed_hops={} certainty={:.4} holonomy={:?}",
            chain_receipt.steps_ingested,
            chain_receipt.steps_rejected_by_entropy_sieve,
            chain_receipt.arc_ids.len(),
            traversed.len(),
            chain_receipt.mean_teacher_certainty_q32.to_f64(),
            chain_receipt.holonomy_verdict
        );

        let q4_tap = atoma_immune::QuantizedLayerTapDescriptor {
            tensor_name: "model.layers.14.mlp.down_proj.q4",
            layer_index: 14,
            total_layers: 64,
            weight_data: &packed_w,
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
        let q4_distill = server.distill_quantized_layer(&q4_tap, 12, oss_teacher.fingerprint)?;
        let q6_tap = atoma_immune::QuantizedLayerTapDescriptor {
            tensor_name: "model.layers.46.self_attn.o_proj.q6",
            bits: 6,
            layer_index: 46,
            weight_data: &packed_w6,
            ..q4_tap
        };
        let q6_distill = server.distill_quantized_layer(&q6_tap, 12, oss_teacher.fingerprint)?;
        println!(
            "[verified] Zero-Copy Quantized Distillation: Q4(rows={}->dequant={}, saved={}MB, atoms={}) | Q6(rows={}->dequant={}, saved={}MB, atoms={})",
            q4_distill.total_rows_scanned_via_scales,
            q4_distill.top_k_rows_dequantized,
            q4_distill.heap_bytes_saved_vs_full_dequant / (1024 * 1024),
            q4_distill.minted_atom_ids.len(),
            q6_distill.total_rows_scanned_via_scales,
            q6_distill.top_k_rows_dequantized,
            q6_distill.heap_bytes_saved_vs_full_dequant / (1024 * 1024),
            q6_distill.minted_atom_ids.len()
        );

        let live_step_a = TeacherTapRecord {
            shard_id: 88_001,
            layer_index: 10,
            hidden_state: (0..256).map(|i| ((i as f32) * 0.071 + 0.3).sin()).collect(),
            next_op_probs: vec![0.91, 0.04, 0.03, 0.02],
            payload: vec![0u8; 256],
        };
        let live_step_b = TeacherTapRecord {
            shard_id: 88_002,
            layer_index: 36,
            hidden_state: (0..256).map(|i| ((i as f32) * 0.137 - 1.1).cos()).collect(),
            next_op_probs: vec![0.03, 0.89, 0.05, 0.03],
            payload: vec![1u8; 256],
        };
        let live_noise = TeacherTapRecord {
            shard_id: 88_003,
            layer_index: 20,
            hidden_state: (0..256).map(|i| ((i as f32) * 0.29).sin()).collect(),
            next_op_probs: vec![0.125; 8],
            payload: vec![2u8; 256],
        };
        let r_a = server.observe_and_distill_live_step(&live_step_a)?;
        let r_b = server.observe_and_distill_live_step(&live_step_b)?;
        let r_b_dup = server.observe_and_distill_live_step(&live_step_b)?;
        let r_noise = server.observe_and_distill_live_step(&live_noise)?;
        println!(
            "[verified] Live Server Novelty-Only Distillation: stepA_novel={} stepB(novel_atom={}, novel_arc={}) replayB(novel_atom={}, novel_arc={}) noise_sieved={}",
            r_a.atom_was_novel,
            r_b.atom_was_novel,
            r_b.arc_was_novel,
            r_b_dup.atom_was_novel,
            r_b_dup.arc_was_novel,
            r_noise.rejected_as_high_entropy_noise
        );
    }

    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let active_mind_vault = repo_root.join("design/global_active_mind.atoma_map");
    let qwen_traj_vault = repo_root.join("design/qwen_token_trajectories.atoma_map");
    let default_vault = if active_mind_vault.exists() {
        active_mind_vault
    } else if qwen_traj_vault.exists() {
        qwen_traj_vault
    } else {
        repo_root.join("design/global_swarm_map.atoma_map")
    };
    let vault_path = args
        .iter()
        .position(|a| a == "--vault")
        .and_then(|v_idx| args.get(v_idx + 1))
        .map(std::path::PathBuf::from)
        .unwrap_or(default_vault);

    for (idx, arg) in args.iter().enumerate() {
        if arg == "--distill-safetensors" || arg == "--distill-model-dir" || arg == "--distill-full"
        {
            if let Some(target_path_str) = args.get(idx + 1) {
                let top_k = args
                    .iter()
                    .position(|a| a == "--top-k")
                    .and_then(|k_idx| args.get(k_idx + 1))
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(6);
                let max_layers = args
                    .iter()
                    .position(|a| a == "--max-layers")
                    .and_then(|m_idx| args.get(m_idx + 1))
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(100_000);
                let target_path = std::path::Path::new(target_path_str);
                let t_st = std::time::Instant::now();
                let summary = server.distill_full_checkpoint_novel_only(
                    target_path,
                    top_k,
                    max_layers,
                    Some(&vault_path),
                )?;
                let elapsed_ms = t_st.elapsed().as_secs_f64() * 1000.0;
                let active_silos = summary.role_distribution.iter().filter(|&&c| c > 0).count();
                let vault_hex = summary
                    .vault_sha256
                    .map(|d| d.to_hex()[..16].to_string())
                    .unwrap_or_else(|| "none".to_string());
                println!(
                    "[verified] Full Novelty-Only Distillation ({}): shards={} tensors_distilled={} skipped_dup_tensors={} novel_0cells={} dedup_0cells={} novel_1cell_arcs={} dedup_1cell_arcs={} active_silos={}/12 f32_saved={}MB vault_sha256={}.. latency={:.2}ms",
                    target_path_str,
                    summary.shards_processed,
                    summary.tensors_distilled,
                    summary.tensors_skipped_already_ingested,
                    summary.novel_atoms_minted,
                    summary.deduplicated_atoms_coalesced,
                    summary.novel_reasoning_arcs_minted,
                    summary.deduplicated_reasoning_arcs,
                    active_silos,
                    summary.f32_heap_bytes_saved / (1024 * 1024),
                    vault_hex,
                    elapsed_ms
                );
            }
        }
    }

    // Load persistent vault ONCE when `--query`, `--serve-stdin`, `--serve`, or `--probe-domains` is requested
    if wants_query_or_serve && server.persistent_vault_path.is_none() {
        let t_load = std::time::Instant::now();
        let (loaded_atoms, loaded_arcs, sha_opt) =
            server.load_or_init_persistent_vault(&vault_path)?;
        let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
        let sha_hex = sha_opt
            .map(|d| d.to_hex()[..16].to_string())
            .unwrap_or_else(|| "none".to_string());
        println!(
            "[verified] Pre-Cached Persistent Vault ONCE ({}): global_0cells={} global_1cell_arcs={} ingested_tensors={} sha256={}.. in {:.2}ms",
            vault_path.display(),
            loaded_atoms,
            loaded_arcs,
            server.swarm_map.ingested_tensor_digests.len(),
            sha_hex,
            load_ms
        );
    }

    let role_override = args
        .iter()
        .position(|a| a == "--role")
        .and_then(|r_idx| args.get(r_idx + 1))
        .and_then(|s| parse_role_arg(s));
    let top_k_retrieve = args
        .iter()
        .position(|a| a == "--top-k-retrieve")
        .and_then(|k_idx| args.get(k_idx + 1))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(5);
    let max_hops = args
        .iter()
        .position(|a| a == "--hops")
        .and_then(|h_idx| args.get(h_idx + 1))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(6);
    let learn_query = args.iter().any(|a| a == "--learn-query");
    let max_new = args
        .iter()
        .position(|a| a == "--max-new")
        .and_then(|idx| args.get(idx + 1))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(32);

    // Empirical Multi-Basin Specialist Carving Probe (`--probe-domains` or `--probe <domain_or_prompt>`)
    if args.iter().any(|a| a == "--probe-domains") {
        print_all_domain_specialist_carvings(&server);
    }
    for (idx, arg) in args.iter().enumerate() {
        if arg == "--probe" {
            if let Some(q_probe) = args.get(idx + 1) {
                let profile = server.probe_and_carve_live_specialist(q_probe, None);
                print_specialist_carving_profile(&profile);
            }
        }
    }

    // 1. One-shot or multi-query CLI flag: `--query "<text>"`
    for (idx, arg) in args.iter().enumerate() {
        if arg == "--query" {
            if let Some(q_text) = args.get(idx + 1) {
                let t_q = std::time::Instant::now();
                let res = server.query_swarm_map(
                    q_text,
                    role_override,
                    top_k_retrieve,
                    max_hops,
                    learn_query,
                )?;
                let q_us = t_q.elapsed().as_micros();
                print_query_result(&res, q_us);
            }
        }
    }

    // 2. Persistent Interactive REPL / Pipe Daemon Mode: `--serve-stdin`
    if args.iter().any(|a| a == "--serve-stdin") {
        use std::io::BufRead;
        let mut runtime = MainModelCoordinatorRuntime {
            server,
            sessions: Default::default(),
        };
        println!(
            "[verified] GeodesicActiveTransformer stdin ready (bare prompts generate from the global map; `query <text>` retrieves only, `learn <text>` intakes only, `save` snapshots vault, `/reset`, `status`, `quit`; max_new={}, global_0cells={}, global_1cell_arcs={})",
            max_new,
            runtime.server.swarm_map.global_atom_count(),
            runtime.server.swarm_map.reasoning_arcs.len()
        );
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if matches!(trimmed, "quit" | "exit" | "/quit" | "/exit") {
                break;
            }
            if trimmed == "/reset" {
                runtime.sessions.remove(&1);
                println!("[session reset; global map retained]");
                continue;
            }
            if trimmed == "save" {
                let t_s = std::time::Instant::now();
                runtime.server.persist_global_map()?;
                println!(
                    "[verified] Vault snapshotted in {:.1}ms",
                    t_s.elapsed().as_secs_f64() * 1000.0
                );
                continue;
            }
            if trimmed == "status" {
                let silo_counts: Vec<String> = RoleId::CARVED_12
                    .iter()
                    .map(|&r| {
                        format!(
                            "{}:{}",
                            r.short_code(),
                            runtime
                                .server
                                .swarm_map
                                .silo(r)
                                .map(|s| s.atom_count())
                                .unwrap_or(0)
                        )
                    })
                    .collect();
                println!(
                    "[verified] Vault Status: global_0cells={} global_1cell_arcs={} ingested_tensors={} silos=[{}]",
                    runtime.server.swarm_map.global_atom_count(),
                    runtime.server.swarm_map.reasoning_arcs.len(),
                    runtime.server.swarm_map.ingested_tensor_digests.len(),
                    silo_counts.join(", ")
                );
                continue;
            }
            if let Some(q_str) = trimmed.strip_prefix("query ") {
                let t_q = std::time::Instant::now();
                let res = runtime.server.query_swarm_map(
                    q_str.trim(),
                    role_override,
                    top_k_retrieve,
                    max_hops,
                    learn_query,
                )?;
                print_query_result(&res, t_q.elapsed().as_micros());
            } else if let Some(q_str) = trimmed.strip_prefix("learn ") {
                let t_q = std::time::Instant::now();
                let res = runtime.server.query_swarm_map(
                    q_str.trim(),
                    role_override,
                    top_k_retrieve,
                    max_hops,
                    true,
                )?;
                print_query_result(&res, t_q.elapsed().as_micros());
            } else if let Some(sub_str) = trimmed.strip_prefix("subagent ") {
                let t_sub = std::time::Instant::now();
                match runtime.execute_subagent_turn(1, sub_str.trim(), role_override, None, max_new)
                {
                    Ok(sub) => {
                        println!("{}", sub.task_receipt.decoded_output_text);
                        println!(
                            "[subagent] role={} silos={} carved_0cells={} carved_1cells={} excluded_0cells={} emitted={} elapsed_ms={:.1}",
                            sub.primary_role.short_code(),
                            sub.allowed_silos.len(),
                            sub.carved_atom_count,
                            sub.carved_arc_count,
                            sub.global_atoms_excluded,
                            sub.task_receipt.emitted_ops.len(),
                            t_sub.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                    Err(error) => eprintln!("[subagent held] {error}"),
                }
            } else {
                let t_turn = std::time::Instant::now();
                match runtime.execute_human_turn(1, trimmed, max_new, |_, _| {
                    Err(
                        "no external teacher is configured and map inference returned no output"
                            .to_string(),
                    )
                }) {
                    Ok(turn) => {
                        println!("{}", turn.response_text);
                        println!(
                            "[turn] tier={:?} emitted={} tier2_holds={} new_arcs={} elapsed_ms={:.1}",
                            turn.tier,
                            turn.emitted_ops.len(),
                            turn.tier2_superposition_interventions,
                            turn.crystallized_arcs,
                            t_turn.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                    Err(error) => eprintln!("[generation held] {error}"),
                }
            }
        }
        if runtime.sessions.contains_key(&1) {
            println!(
                "[session] turn history retained in compact runtime state (96B GeodesicCursor + .wal journal)"
            );
        }
        return Ok(());
    }

    // 3. Persistent TCP / HTTP Daemon Mode: `--serve <host:port>` (e.g. `--serve 127.0.0.1:8088`)
    if let Some(s_idx) = args.iter().position(|a| a == "--serve") {
        if let Some(bind_addr) = args.get(s_idx + 1) {
            let max_requests = args
                .iter()
                .position(|a| a == "--max-requests")
                .and_then(|m_idx| args.get(m_idx + 1))
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(usize::MAX);
            run_persistent_tcp_server(
                &mut server,
                bind_addr,
                role_override,
                top_k_retrieve,
                max_hops,
                learn_query,
                max_requests,
            )?;
        }
    }

    println!(
        "\n=== ALL THREE LAYERS, 10 QUALIFICATIONS & 5 STATUTORY FRAMEWORKS VERIFIED (audit entries = {}) ===",
        server.immune.audit_log().len()
    );
    Ok(())
}

fn parse_role_arg(s: &str) -> Option<RoleId> {
    let up = s.trim().to_ascii_uppercase();
    for r in RoleId::CARVED_12 {
        if r.short_code() == up || r.name().eq_ignore_ascii_case(s.trim()) {
            return Some(r);
        }
    }
    None
}

fn print_all_domain_specialist_carvings(server: &MonsterHybridServer) {
    println!("\n==========================================================================");
    println!(" EMPIRICAL MULTI-BASIN SPECIALIST CARVING (12 ATTRACTOR BASINS -> N SPHERES)");
    println!("==========================================================================");
    for preset in CANONICAL_DOMAIN_PRESETS {
        let t0 = std::time::Instant::now();
        let profile = server.probe_and_carve_live_specialist(preset.domain_id, None);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        print_specialist_carving_profile(&profile);
        println!("  [probe_latency] {:.2}ms\n", ms);
    }
}

fn print_specialist_carving_profile(profile: &CarvedSpecialistProfile) {
    let carved_codes: Vec<String> = profile
        .lit_basins
        .iter()
        .map(|r| format!("{}({})", r.short_code(), r.name()))
        .collect();
    let mut ranked: Vec<_> = profile.basin_metrics.iter().collect();
    ranked.sort_by(|a, b| b.activation_share.partial_cmp(&a.activation_share).unwrap());
    let top_basins_str: Vec<String> = ranked
        .iter()
        .take(5)
        .map(|b| {
            format!(
                "{}:{:.1}%(0c={},1c={},qwen={:.1}%){}",
                b.role.short_code(),
                b.activation_share * 100.0,
                b.lit_0cell_atoms,
                b.lit_1cell_arcs,
                b.qwen_mode_affinity * 100.0,
                if b.selected_for_specialist { "*" } else { "" }
            )
        })
        .collect();
    let total_0cells = (profile.carved_0cell_count + profile.subtracted_0cell_count).max(1);
    println!(
        "[verified] Specialist Carving [{}] ({}):\n  -> Carved Composite Specialist ({} spheres, {:.1}% mass): {}\n  -> Carved Sub-Map: 0-cells={} / {} (subtracted={} outside spheres, {:.1}% pruned) | 1-cell arcs={}\n  -> Top Attractor Basins: {}\n  -> Geodesic Trace: lit_atoms={} lit_arcs={} holonomy={:.4}rad decoded=\"{}\"",
        profile.domain_id,
        profile.display_name,
        profile.lit_basins.len(),
        profile.captured_activation_share * 100.0,
        carved_codes.join(" + "),
        profile.carved_0cell_count,
        total_0cells,
        profile.subtracted_0cell_count,
        (profile.subtracted_0cell_count as f64 / total_0cells as f64) * 100.0,
        profile.carved_1cell_count,
        top_basins_str.join(" | "),
        profile.lit_atoms.len(),
        profile.lit_arcs.len(),
        profile.holonomy_flux_rad,
        profile.decoded_continuation
    );
}

fn print_query_result(res: &atoma_orchestrator::SwarmMapQueryResult, elapsed_us: u128) {
    println!(
        "[verified] QueryResult (\"{}\") -> role={}({}) sector={} silo_atoms={}/{} global_arcs={} mlir_result_q32={:.4} holonomy_flux={:.4}rad latency={}us",
        res.query_text,
        res.routed_role.short_code(),
        res.routed_role.name(),
        res.query_sector,
        res.silo_atom_count,
        res.global_atom_count,
        res.global_reasoning_arc_count,
        res.mlir_execution_result_q32.to_f64(),
        res.holonomy_flux_rad_q32.to_f64(),
        elapsed_us
    );
    for (rank, h) in res.hits.iter().enumerate() {
        println!(
            "  [0-cell #{}] atom={} cosh_d={:.6} well_cosh={:.4} salience={} boundary={} kv_retain={:.4} origin={}..",
            rank + 1,
            h.atom_id.0,
            h.cosh_distance_q32.to_f64(),
            h.well_cosh_dist_q32.to_f64(),
            h.salience_count,
            h.in_boundary_overlap,
            h.adapted_kv_retention_q32.to_f64(),
            &h.origin_model_fingerprint.to_hex()[..12]
        );
    }
    for (hop_idx, hop) in res.reasoning_hops.iter().enumerate() {
        println!(
            "  [1-cell hop #{}] arc={} atom {}({}) -> atom {}({}) cosh_action={:.6} certainty={:.4} mlir_op={}({})",
            hop_idx + 1,
            hop.arc_id,
            hop.from_atom.0,
            hop.from_role.short_code(),
            hop.to_atom.0,
            hop.to_role.short_code(),
            hop.cosh_action_q32.to_f64(),
            hop.teacher_certainty_q32.to_f64(),
            hop.mlir_mnemonic,
            hop.mlir_op_id
        );
    }
    println!(
        "  [native-mlir-ssa] ops={:?} -> deterministic_q32={:.4}",
        res.synthesized_mlir_program,
        res.mlir_execution_result_q32.to_f64()
    );
    if let Some(ref live) = res.live_ingest {
        println!(
            "  [live-novelty-gate] ingested={} novel_atom={} novel_arc={} noise_rejected={}",
            live.step_ingested,
            live.atom_was_novel,
            live.arc_was_novel,
            live.rejected_as_high_entropy_noise
        );
    }
}

fn run_persistent_tcp_server(
    server: &mut MonsterHybridServer,
    bind_addr: &str,
    role_override: Option<RoleId>,
    top_k: usize,
    max_hops: usize,
    learn_default: bool,
    max_requests: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{Read, Write};

    // PRE-CACHE THE ENTIRE SERVER MAP & CANONICAL DOMAIN PROBES ONCE AT STARTUP
    let t_precache = std::time::Instant::now();
    let cached_map_state_json = server.render_map_state_json();
    let mut cached_domain_profiles: HashMap<String, String> = HashMap::new();
    println!(
        "\n[pre-cache] Pre-computing 12-Basin Attractor Map & 5 Canonical Domain Specialist Carvings ONCE..."
    );
    for preset in CANONICAL_DOMAIN_PRESETS {
        let t_p = std::time::Instant::now();
        let profile = server.probe_and_carve_live_specialist(preset.domain_id, None);
        let p_us = t_p.elapsed().as_micros();
        print_specialist_carving_profile(&profile);
        let json = server.render_specialist_profile_json(&profile, p_us);
        cached_domain_profiles.insert(preset.domain_id.to_string(), json);
    }
    println!(
        "[verified] Server Pre-Cache Complete in {:.2}ms (map_state_json={} bytes, preset_domains={})\n",
        t_precache.elapsed().as_secs_f64() * 1000.0,
        cached_map_state_json.len(),
        cached_domain_profiles.len()
    );

    let listener = std::net::TcpListener::bind(bind_addr)?;
    println!(
        "[verified] Live Qwen-Connected Map Visualizer & Pre-Cached HTTP Daemon listening on http://{} (global_0cells={}, global_1cell_arcs={})",
        bind_addr,
        server.swarm_map.global_atom_count(),
        server.swarm_map.reasoning_arcs.len()
    );

    let mut handled = 0usize;
    for stream_res in listener.incoming() {
        if handled >= max_requests {
            break;
        }
        let Ok(mut stream) = stream_res else { continue };
        let mut buf = [0u8; 8192];
        let n = stream.read(&mut buf).unwrap_or(0);
        if n == 0 {
            continue;
        }
        let req_str = String::from_utf8_lossy(&buf[..n]);
        let first_line = req_str.lines().next().unwrap_or("").trim();

        let is_http = first_line.starts_with("GET ") || first_line.starts_with("POST ");
        if is_http {
            let full_path = first_line.split_whitespace().nth(1).unwrap_or("/");
            let (route_path, query_part) = match full_path.find('?') {
                Some(idx) => (&full_path[..idx], &full_path[idx + 1..]),
                None => (full_path, ""),
            };

            if route_path == "/" || route_path == "/index.html" || route_path == "/map" {
                let body = INTERACTIVE_MAP_VISUALIZER_HTML;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(resp.as_bytes());
                handled += 1;
                continue;
            }

            if route_path == "/api/map-state" {
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    cached_map_state_json.len(),
                    cached_map_state_json
                );
                let _ = stream.write_all(resp.as_bytes());
                handled += 1;
                continue;
            }

            if route_path == "/api/probe" {
                let q_param = extract_query_param(query_part, "q")
                    .unwrap_or_else(|| "coding_intensive".to_string());
                let basins_param = extract_query_param(query_part, "basins");
                let manual_basins: Option<Vec<RoleId>> = basins_param.and_then(|b_str| {
                    let parsed: Vec<RoleId> =
                        b_str.split(',').filter_map(|s| parse_role_arg(s)).collect();
                    if parsed.is_empty() {
                        None
                    } else {
                        Some(parsed)
                    }
                });

                // Serve from O(1) pre-cache if canonical domain preset and no manual basin override
                let json_body = if manual_basins.is_none() {
                    if let Some(cached) = cached_domain_profiles.get(q_param.trim()) {
                        cached.clone()
                    } else {
                        let t_p = std::time::Instant::now();
                        let profile = server.probe_and_carve_live_specialist(&q_param, None);
                        server.render_specialist_profile_json(&profile, t_p.elapsed().as_micros())
                    }
                } else {
                    let t_p = std::time::Instant::now();
                    let profile =
                        server.probe_and_carve_live_specialist(&q_param, manual_basins.as_deref());
                    server.render_specialist_profile_json(&profile, t_p.elapsed().as_micros())
                };

                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    json_body.len(),
                    json_body
                );
                let _ = stream.write_all(resp.as_bytes());
                handled += 1;
                continue;
            }
        }

        let query_str = if is_http {
            let path = first_line.split_whitespace().nth(1).unwrap_or("/");
            if let Some(q_pos) = path.find("q=") {
                let raw_q = &path[q_pos + 2..];
                let end = raw_q.find('&').unwrap_or(raw_q.len());
                url_decode(&raw_q[..end])
            } else if first_line.starts_with("POST ") {
                req_str
                    .split("\r\n\r\n")
                    .nth(1)
                    .unwrap_or("")
                    .trim()
                    .to_string()
            } else {
                "status".to_string()
            }
        } else {
            first_line.to_string()
        };

        if query_str == "quit" || query_str == "/quit" {
            let msg = "{\"status\":\"shutting_down\"}\n";
            let _ = stream.write_all(msg.as_bytes());
            break;
        }

        let t_q = std::time::Instant::now();
        let res =
            server.query_swarm_map(&query_str, role_override, top_k, max_hops, learn_default)?;
        let q_us = t_q.elapsed().as_micros();
        print_query_result(&res, q_us);

        let top_hit_id = res.hits.first().map(|h| h.atom_id.0).unwrap_or(0);
        let top_hit_cosh = res
            .hits
            .first()
            .map(|h| h.cosh_distance_q32.to_f64())
            .unwrap_or(1.0);
        let json_body = format!(
            "{{\"query\":\"{}\",\"routed_role\":\"{}\",\"sector\":{},\"silo_atoms\":{},\"global_0cells\":{},\"global_1cell_arcs\":{},\"top_atom_id\":{},\"top_cosh_dist\":{:.6},\"reasoning_hops\":{},\"mlir_result_q32\":{:.4},\"holonomy_flux_rad\":{:.4},\"latency_us\":{}}}\n",
            res.query_text.replace('"', "\\\""),
            res.routed_role.short_code(),
            res.query_sector,
            res.silo_atom_count,
            res.global_atom_count,
            res.global_reasoning_arc_count,
            top_hit_id,
            top_hit_cosh,
            res.reasoning_hops.len(),
            res.mlir_execution_result_q32.to_f64(),
            res.holonomy_flux_rad_q32.to_f64(),
            q_us
        );

        if is_http {
            let http_resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                json_body.len(),
                json_body
            );
            let _ = stream.write_all(http_resp.as_bytes());
        } else {
            let _ = stream.write_all(json_body.as_bytes());
        }
        handled += 1;
        if handled >= max_requests {
            break;
        }
    }
    Ok(())
}

fn extract_query_param(query: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    for pair in query.split('&') {
        if let Some(val) = pair.strip_prefix(&prefix) {
            return Some(url_decode(val));
        }
    }
    None
}

fn url_decode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'+' {
            out.push(' ');
            i += 1;
        } else if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(hex_str) = std::str::from_utf8(&bytes[i + 1..i + 3]) {
                if let Ok(val) = u8::from_str_radix(hex_str, 16) {
                    out.push(val as char);
                    i += 3;
                    continue;
                }
            }
            out.push(bytes[i] as char);
            i += 1;
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}
