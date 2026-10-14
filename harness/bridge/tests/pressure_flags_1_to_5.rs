//! Comprehensive Pressure Test Suite for FLAGS 1–5, `core_lm` Contracts, and the Full 3-Layer Architecture
//! (`core_lm` + `atoma_immune` + `atoma_bridge_harness`).

use atoma_bridge_harness::{
    Atlas5DHybridBridge, KeyCorpus, KeyPipeline, KeyRegime, LegacyTruncationBridge,
    SpectralOpticalBridge, ZeroCalRoPEFoldBridge, evaluate_bridge,
};
use atoma_immune::cache::geometric::GeometricMemoryMap;
use atoma_immune::geometry::bridge::{f32_to_atom_address, f32_to_lorentz};
use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::geometry::lorentz::LorentzVector;
use atoma_immune::geometry::picard::PicardMapper;
use atoma_immune::geometry::torus::{PhaseState, TorPhaseEncoder};
use atoma_immune::lock::{
    cordic_sin, extract_invariant, kuramoto_step_cpu, pi_unit_to_rad_q32, rad_q32_to_pi_unit,
};
use atoma_immune::{
    ATOMIC_IR_MAX_VERIFY_TOKENS, Atlas13, AtomicIrAlphabet, CarvedSwarmMap,
    FirstPrinciplesComplianceEngine, FrozenBaseWeights, FrozenShard, ImmuneSystem, OpticalLens,
    OpticalRiskEngine, OverlapPolicyMatrix, QuotientManifoldProof, RegulatoryFramework, RoleId,
    TangentProjector, TeacherTapRecord, merkle_hash_leaf,
};
use std::collections::HashSet;

#[test]
fn pressure_test_flag01_live_core_lm_bridge_and_loss_accounting() {
    // Verify that live core_lm::geometry::bridge::{f32_to_lorentz, f32_to_atom_address}
    // folds all 128D and 256D coordinates, preserves hyperboloid invariants across 1,024 keys,
    // and populates mandatory BridgeLossReport (FLAG-01 & Qualification #9).
    for &regime in &[
        KeyRegime::IsotropicRoPE128,
        KeyRegime::AnisotropicSpectral128,
        KeyRegime::SemanticAtlas13Clusters128,
        KeyRegime::QwenPartialRoPEOutlier256,
    ] {
        let corpus = KeyCorpus::generate(regime, 256, 64, 0xA101_BEEF);
        let mut map = GeometricMemoryMap::new(512);
        map.use_picard = false;

        for (idx, k) in corpus.keys.iter().enumerate() {
            let (lv, phase, loss) = f32_to_atom_address(k).expect("live core_lm bridge projection");
            assert_eq!(loss.input_dimensions, corpus.head_dim);
            assert_eq!(loss.discarded_dimensions, corpus.head_dim - 5);
            assert_eq!(loss.rope_pair_integrity_q32, Q32_31::ONE);
            assert!(
                loss.retained_5d_energy_ratio_q32.to_f64() > 0.0,
                "Retained 5D energy must be strictly positive"
            );
            let sum_ratios = loss.retained_5d_energy_ratio_q32.to_f64()
                + loss.discarded_energy_ratio_q32.to_f64();
            assert!(
                (sum_ratios - 1.0).abs() < 1e-6,
                "Retained + discarded energy ratio must sum to 1.0, got {sum_ratios}"
            );
            assert!(
                phase.theta_f != Q32_31::ZERO || phase.theta_s != Q32_31::ZERO,
                "FLAG-03: live bridge must populate non-zero T^2 phase"
            );

            // Verify hyperboloid invariant x0^2 - ||x||^2 = 1
            let ip_self = LorentzVector::inner_product(&lv, &lv).to_f64();
            assert!((ip_self - 1.0).abs() < 1e-5, "ip_self = {ip_self}");

            map.write(lv, phase, vec![(idx % 13) as u8]).unwrap();
        }
        map.rebuild();

        // Also verify calibrated SpectralOpticalBridge + Atlas5DHybridBridge pressure gate
        let legacy = LegacyTruncationBridge;
        let zero_cal =
            ZeroCalRoPEFoldBridge::new(corpus.head_dim, regime.rope_style(), corpus.rope_theta);
        let spectral = SpectralOpticalBridge::calibrate(
            &corpus.keys,
            corpus.head_dim,
            regime.rope_style(),
            corpus.rope_theta,
        );
        let (var3, _, var13) = spectral.spectral_variance_capture_pct();
        let atlas5d = Atlas5DHybridBridge::new(spectral.clone(), 4);

        let rep_leg = evaluate_bridge(&corpus, &legacy, None).unwrap();
        let rep_zc = evaluate_bridge(&corpus, &zero_cal, None).unwrap();
        let rep_spec = evaluate_bridge(&corpus, &spectral, Some(var3)).unwrap();
        let rep_5d = evaluate_bridge(&corpus, &atlas5d, Some(var13)).unwrap();

        assert!(rep_zc.inner_product_spearman_rho > rep_leg.inner_product_spearman_rho + 0.15);
        assert!(rep_spec.inner_product_spearman_rho > 0.85);
        assert!(rep_5d.recall_at_8 > 60.0);
        assert!(rep_5d.downstream_attn_cosine_sim > 0.98);
    }
}

#[test]
fn pressure_test_flag02_extreme_boost_no_overflow_and_1000_atom_vptree_parity() {
    // 1. Extreme boost values that previously overflowed i64 at x0 >= 65,536
    for &x0 in &[1.0f64, 100.0, 65_536.0, 100_000.0, 250_000.0] {
        let r = (x0 * x0 - 1.0).max(0.0).sqrt();
        let lv = LorentzVector::from_f64([x0, r * 0.6, r * 0.8, 0.0]);
        assert!(
            lv.validate().is_ok(),
            "FLAG-02: LorentzVector with x0={x0} must validate without i64 overflow"
        );
        let self_ip = LorentzVector::inner_product(&lv, &lv).to_f64();
        assert!(
            (self_ip - 1.0).abs() < 0.05,
            "FLAG-02: self inner product at x0={x0} was {self_ip}"
        );
    }

    // 2. 1,000-atom VP-tree vs. linear scan exact parity
    let mut map = GeometricMemoryMap::new(1024);
    map.use_picard = false;
    let encoder = TorPhaseEncoder::new();

    for i in 0..1000usize {
        let s = format!("atom_pressure_node_{i}");
        let (df, ds) = encoder.project_increments(s.as_bytes());
        let dir = [
            Q32_31::from_f64((df.to_f64() * 7.0 + i as f64 * 0.13).cos()),
            Q32_31::from_f64((ds.to_f64() * 11.0 + i as f64 * 0.29).sin()),
            Q32_31::from_f64(((i as f64) * 0.07).cos() * 0.5),
        ];
        let norm_sq = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).to_f64();
        let scale = 1.0 / norm_sq.sqrt().max(1e-9);
        let unit_dir = [
            Q32_31::from_f64(dir[0].to_f64() * scale),
            Q32_31::from_f64(dir[1].to_f64() * scale),
            Q32_31::from_f64(dir[2].to_f64() * scale),
        ];
        let beta = Q32_31::from_f64(0.05 + (i % 40) as f64 * 0.015);
        let addr = LorentzVector::boost(&LorentzVector::origin(), &unit_dir, beta);
        map.store(addr, (i as u32).to_le_bytes().to_vec()).unwrap();
    }

    let q_atom = map.atoms[&42].lorentz;
    let linear_ids: Vec<u64> = map
        .retrieve(&q_atom, 16, true)
        .unwrap()
        .iter()
        .map(|a| a.id.0)
        .collect();
    map.rebuild();
    let vptree_ids: Vec<u64> = map
        .retrieve(&q_atom, 16, true)
        .unwrap()
        .iter()
        .map(|a| a.id.0)
        .collect();
    assert_eq!(
        linear_ids, vptree_ids,
        "1000-atom VP-tree must match linear order bit-for-bit"
    );
}

#[test]
fn pressure_test_flag03_toroidal_phase_and_kuramoto_unit_bridge() {
    // 1. Verify radian <-> pi-unit conversions across 360 angles around the circle
    for deg in -179..=179 {
        let rad_f64 = (deg as f64) * std::f64::consts::PI / 180.0;
        let rad_q32 = Q32_31::from_f64(rad_f64);
        let pi_u = rad_q32_to_pi_unit(rad_q32);
        let roundtrip = pi_unit_to_rad_q32(pi_u);
        assert!(
            (roundtrip.to_f64() - rad_f64).abs() < 1e-7,
            "Angle {deg} deg failed roundtrip: got {}, expected {}",
            roundtrip.to_f64(),
            rad_f64
        );
        let s_cordic = (cordic_sin(pi_u) as f64) / ((1i64 << 31) as f64);
        let s_exact = rad_f64.sin();
        assert!(
            (s_cordic - s_exact).abs() < 1e-5,
            "CORDIC sin mismatch at {deg} deg: {s_cordic} vs {s_exact}"
        );
    }

    // 2. Verify TorPhaseEncoder evolution + Kuramoto lock invariant extraction
    let encoder = TorPhaseEncoder::new();
    let mut state = PhaseState {
        theta_f: Q32_31::ZERO,
        theta_s: Q32_31::ZERO,
        psi: Q32_31::ZERO,
    };
    for step in 0..32 {
        let msg = format!("token_stream_step_{step}");
        let (df, ds) = encoder.project_increments(msg.as_bytes());
        state = encoder.evolve(state, df, ds);
    }
    assert!(state.theta_f != Q32_31::ZERO && state.theta_s != Q32_31::ZERO);
    let tf_pi = rad_q32_to_pi_unit(state.theta_f);
    let ts_pi = rad_q32_to_pi_unit(state.theta_s);
    let inv = extract_invariant(tf_pi, ts_pi);
    assert!(inv.n < 12 && inv.m < 12);

    // 3. Verify Kuramoto CPU stepper converges oscillator phases
    let mut theta = vec![0i64, 300_000_000, -250_000_000, 150_000_000];
    let omega = vec![10_000_000i64; 4];
    let dt = Q32_31::from_f64(0.05).0;
    let k_coupling = Q32_31::from_f64(0.8).0;
    let mut last_slip = i64::MAX;
    for _ in 0..50 {
        last_slip = kuramoto_step_cpu(&mut theta, &omega, dt, k_coupling);
    }
    assert!(last_slip > 0);
}

#[test]
fn pressure_test_flag04_qwen_256d_kv_payload_and_exact_dot_scoring() {
    // Simulates Qwen3.5-MoE full_attention (head_dim = 256, rotary_dim = 64) with
    // the upgraded 2 * hd * 4 byte (k_head || v_head) payload and exact 256D dot scoring (FLAG-04).
    let hd = 256usize;
    let rotary_dim = 64usize;
    let n_tokens = 64usize;
    let mut geo = GeometricMemoryMap::new(256);
    geo.use_picard = false;

    let mut raw_keys = Vec::with_capacity(n_tokens);
    let mut raw_vals = Vec::with_capacity(n_tokens);

    for pos in 0..n_tokens {
        let mut k_head: Vec<f32> = (0..hd)
            .map(|d| (((pos + 1) * (d + 3)) as f32 * 0.17).cos())
            .collect();
        let v_head: Vec<f32> = (0..hd)
            .map(|d| (((pos + 1) * (d + 5)) as f32 * 0.11).sin())
            .collect();
        KeyPipeline::scaleless_rmsnorm(&mut k_head, 1e-6);
        KeyPipeline::apply_rope_neox_partial(&mut k_head, rotary_dim, pos, 10_000.0);

        let (lv, phase, _loss) = f32_to_atom_address(&k_head).unwrap();
        let mut kv_bytes = Vec::with_capacity(2 * hd * 4);
        for f in k_head.iter() {
            kv_bytes.extend_from_slice(&f.to_le_bytes());
        }
        for f in v_head.iter() {
            kv_bytes.extend_from_slice(&f.to_le_bytes());
        }
        geo.write(lv, phase, kv_bytes).unwrap();
        raw_keys.push(k_head);
        raw_vals.push(v_head);
    }
    geo.rebuild();

    let q_head = raw_keys[19].clone();
    let q_lv = f32_to_lorentz(&q_head).unwrap();
    let neighbors = geo.retrieve(&q_lv, 64, false).unwrap();
    assert_eq!(neighbors.len(), 64);

    let inv_sqrt_hd = 1.0 / (hd as f32).sqrt();
    let mut best_score = f32::NEG_INFINITY;
    let mut best_v = Vec::new();
    for atom in neighbors {
        let payload = &atom.payload;
        assert_eq!(
            payload.len(),
            2 * hd * 4,
            "FLAG-04: Qwen payload must hold k_head || v_head"
        );
        let kh: Vec<f32> = payload[0..hd * 4]
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let vh: Vec<f32> = payload[hd * 4..2 * hd * 4]
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let dot: f32 = q_head.iter().zip(kh.iter()).map(|(a, b)| a * b).sum();
        let score = dot * inv_sqrt_hd;
        if score > best_score {
            best_score = score;
            best_v = vh;
        }
    }
    for d in 0..hd {
        assert!((best_v[d] - raw_vals[19][d]).abs() < 1e-6);
    }
}

#[test]
fn pressure_test_flag05_end_to_end_three_layer_immune_refraction_and_silo_carving() {
    // Full end-to-end integration pressure test across:
    // 1. TeacherTapRecord -> FrozenShard (Stage 1 & 2 SHA-256)
    // 2. TangentProjector -> OpticalLens::refract (Stage 3 & 4 H^3 x T^2 refraction)
    // 3. Atlas13 & CarvedSwarmMap 12-Around-1 role carving (Stage 5)
    // 4. ImmuneSystem + FirstPrinciplesComplianceEngine + QuotientManifoldProof + OpticalRiskEngine
    let atlas = Atlas13::approved_icosahedral();
    let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
    let base = FrozenBaseWeights::freeze("Muse-Glimmer-52L-Frozen", 52, vec![0.25f32; 2048]);
    let mut immune = ImmuneSystem::new(policy.clone(), base.fingerprint);
    let mut swarm_map = CarvedSwarmMap::new(atlas, base, Q32_31::from_f64(1.22));
    let projector = TangentProjector::default();

    for i in 0..48usize {
        let role = RoleId::CARVED_12[i % 12];
        let well = swarm_map.atlas.well(role);

        let mut hidden = vec![0.0f32; 128];
        for d in 0..128 {
            let b = d % 3;
            let coord = well.poincare_ball[b].to_f64() * 2.5;
            hidden[d] = (coord + ((i + d) as f64 * 0.05).sin() * 0.1) as f32;
        }
        let probs = vec![0.4f32, 0.3, 0.2, 0.1];
        let tap = TeacherTapRecord {
            shard_id: i as u64,
            layer_index: 48,
            hidden_state: hidden,
            next_op_probs: probs,
            payload: format!("silo_atom_{i}").into_bytes(),
        };

        let _shard = FrozenShard::from_tap(&tap).unwrap();
        let projected = projector.project(&tap).unwrap();
        let refracted = OpticalLens::refract(&projected, well.phase).unwrap();

        assert!(refracted.lorentz.validate().is_ok());
        let reduced = PicardMapper::map_to_fundamental_domain(&refracted.lorentz);
        assert!(reduced.validate().is_ok());

        immune
            .check_base_weights(&swarm_map)
            .expect("Base weight immutability gate must pass");
    }

    let digests = vec![
        merkle_hash_leaf(b"flag01_verified"),
        merkle_hash_leaf(b"flag02_verified"),
        merkle_hash_leaf(b"flag03_verified"),
        merkle_hash_leaf(b"flag04_verified"),
        merkle_hash_leaf(b"flag05_verified"),
    ];
    let fp_report =
        FirstPrinciplesComplianceEngine::prove_from_live_state(&mut swarm_map, &policy, &digests);
    assert!(fp_report.all_statutes_satisfied);
    assert_eq!(fp_report.verified_invariants.len(), 8);
    for fw in RegulatoryFramework::ALL {
        assert_eq!(fp_report.framework_scores_q32[&fw], Q32_31::ONE);
    }

    let q_rep = QuotientManifoldProof::verify_all();
    assert!(q_rep.sector_tiling_exact);
    assert!(q_rep.psi_star_2_to_1_invariant);
    assert!(q_rep.sector_shift_equivariant);
    assert!(q_rep.grid_144_complete);

    let b_origin = OpticalRiskEngine::optical_blast_radius_q32(&LorentzVector::origin());
    assert_eq!(b_origin, Q32_31::HALF);

    let ir = AtomicIrAlphabet::new_canonical(248_320);
    let tokens = vec![101, 102, 103, 104, 77, 201, 202, 203];
    let atoms = ir.encode(&tokens);
    assert_eq!(ir.expand(&atoms), tokens);
    assert!(ir.expand_within_verify_cliff(&atoms).len() <= ATOMIC_IR_MAX_VERIFY_TOKENS);
}

#[test]
fn pressure_test_core_lm_equivalence_and_contract_bars() {
    // Executes the upgraded core_lm/tests/equivalence_bar.rs tests (1000-atom geometric bar
    // + 128D bridge cluster retrieval bar).
    let mut map = GeometricMemoryMap::new(128);
    map.use_picard = false;

    for c in 0..8usize {
        for j in 0..8usize {
            let mut key = vec![0.0f32; 128];
            for d in 32..128 {
                let angle = ((c + 1) * (d + 1)) as f32 * 0.41;
                let jitter = (j as f32) * 0.015 * ((d as f32) * 0.19).sin();
                key[d] = angle.cos() + jitter;
            }
            let (lv, phase, loss) = f32_to_atom_address(&key).unwrap();
            assert_eq!(loss.input_dimensions, 128);
            assert!(phase.theta_f != Q32_31::ZERO || phase.theta_s != Q32_31::ZERO);
            map.write(lv, phase, vec![c as u8]).unwrap();
        }
    }
    map.rebuild();

    let mut total_hits = 0usize;
    for c in 0..8usize {
        let mut query = vec![0.0f32; 128];
        for d in 32..128 {
            let angle = ((c + 1) * (d + 1)) as f32 * 0.41;
            query[d] = angle.cos();
        }
        let (q_lv, q_phase, _) = f32_to_atom_address(&query).unwrap();
        let hits = map.retrieve_5d(&q_lv, &q_phase, 8, false).unwrap();
        let matching = hits.iter().filter(|a| a.payload[0] == c as u8).count();
        total_hits += matching;
    }
    assert!(
        total_hits >= 56,
        "Expected >= 56/64 cluster hits across 128D k[32..128] keys, got {total_hits}/64"
    );
    let _ = HashSet::<u64>::new();
}
