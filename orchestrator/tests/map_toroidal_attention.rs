//! Central-map one-layer attention qualification: causal, role, layer, and convexity gates.

use atoma_immune::cache::geometric::Phase;
use atoma_immune::geometry::bridge::f32_to_atom_address;
use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::{FrozenBaseWeights, RoleId, TeacherTapRecord};
use atoma_orchestrator::MonsterHybridServer;
use atoma_orchestrator::toroidal_attention::{
    MapAttentionFitSample, MapKvIdentity, PhaseSource, ToroidalLayerConfig,
};

fn id(layer: usize, position: usize) -> MapKvIdentity {
    MapKvIdentity {
        layer,
        kv_head: 0,
        sequence_id: 1,
        position,
    }
}

fn weights() -> std::sync::Arc<FrozenBaseWeights> {
    FrozenBaseWeights::freeze("toroidal-map-test", 52, vec![0.5; 128])
}

fn tap(layer: usize, seed: u64) -> TeacherTapRecord {
    TeacherTapRecord {
        shard_id: seed,
        layer_index: layer,
        hidden_state: (0..256)
            .map(|i| ((i as f32 + seed as f32) * 0.037).sin())
            .collect(),
        next_op_probs: vec![0.8, 0.2],
        payload: Vec::new(),
    }
}

fn head(sign: f32) -> Vec<f32> {
    (0..128)
        .map(|i| if i % 2 == 0 { sign } else { 0.0 })
        .collect()
}

fn config(source: PhaseSource) -> ToroidalLayerConfig {
    ToroidalLayerConfig {
        target_layer: 4,
        phase_source: source,
        gain: 1.0,
        baseline_dot_scale: 43.78405189107103 / 128.0,
        top_k: 64,
    }
}

#[test]
fn central_map_causal_role_and_layer_masks_before_convex_weighting() {
    let mut server = MonsterHybridServer::boot_default(weights()).unwrap();
    let q = head(1.0);
    let aligned = server
        .ingest_attention_kv(
            &tap(4, 1),
            RoleId::SensoryPerception,
            id(4, 0),
            &q,
            &vec![1.0; 128],
        )
        .unwrap();
    let opposed = server
        .ingest_attention_kv(
            &tap(4, 2),
            RoleId::SensoryPerception,
            id(4, 1),
            &head(-1.0),
            &vec![2.0; 128],
        )
        .unwrap();
    let wrapped_phase = |phase: Phase| Phase {
        theta_f: Q32_31(phase.theta_f.0 as i32 as i64),
        theta_s: Q32_31(phase.theta_s.0 as i32 as i64),
    };
    assert_eq!(
        aligned.refracted.phase,
        wrapped_phase(f32_to_atom_address(&q).unwrap().1)
    );
    assert_eq!(
        opposed.refracted.phase,
        wrapped_phase(f32_to_atom_address(&head(-1.0)).unwrap().1)
    );
    let future = server
        .ingest_attention_kv(
            &tap(4, 3),
            RoleId::SensoryPerception,
            id(4, 9),
            &q,
            &vec![99.0; 128],
        )
        .unwrap();
    let other_layer = server
        .ingest_attention_kv(
            &tap(3, 4),
            RoleId::SensoryPerception,
            id(3, 0),
            &q,
            &vec![88.0; 128],
        )
        .unwrap();
    let other_role = server
        .ingest_attention_kv(
            &tap(4, 5),
            RoleId::MotorPraxis,
            id(4, 0),
            &q,
            &vec![77.0; 128],
        )
        .unwrap();
    let other_head = server
        .ingest_attention_kv(
            &tap(4, 6),
            RoleId::SensoryPerception,
            MapKvIdentity {
                kv_head: 1,
                ..id(4, 0)
            },
            &q,
            &vec![66.0; 128],
        )
        .unwrap();
    let other_sequence = server
        .ingest_attention_kv(
            &tap(4, 7),
            RoleId::SensoryPerception,
            MapKvIdentity {
                sequence_id: 2,
                ..id(4, 0)
            },
            &q,
            &vec![55.0; 128],
        )
        .unwrap();
    let ordinary = server
        .ingest_teacher_tap(&tap(4, 8), RoleId::SensoryPerception, [Q32_31::ZERO; 3])
        .unwrap();
    assert_ne!(
        ordinary.atom_id, aligned.atom_id,
        "ordinary atoms must not coalesce into typed KV"
    );
    assert_ne!(
        aligned.atom_id, opposed.atom_id,
        "distinct token positions must survive central-map coalescence"
    );
    let phase = server.swarm_map.global_atom(aligned.atom_id).unwrap().phase;

    let result = server
        .evaluate_map_attention_layer(
            RoleId::SensoryPerception,
            4,
            0,
            1,
            42,
            1,
            &q,
            phase,
            config(PhaseSource::PairAngles),
        )
        .unwrap();
    assert!(result.intervention_active);
    assert_eq!(result.atom_ids.len(), 2);
    assert!(result.atom_ids.contains(&aligned.atom_id));
    assert!(result.atom_ids.contains(&opposed.atom_id));
    for excluded in [
        future.atom_id,
        other_layer.atom_id,
        other_role.atom_id,
        other_head.atom_id,
        other_sequence.atom_id,
        ordinary.atom_id,
    ] {
        assert!(!result.atom_ids.contains(&excluded));
    }
    assert!((result.toroidal_weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(result.toroidal_weights.iter().all(|w| *w >= 0.0));
    assert!(result.selected_output.iter().all(|x| x.is_finite()));
    assert_eq!(result.selected_output, result.toroidal_output);

    let control = server
        .evaluate_map_attention_layer(
            RoleId::SensoryPerception,
            3,
            0,
            1,
            42,
            1,
            &q,
            phase,
            config(PhaseSource::PairAngles),
        )
        .unwrap();
    assert!(!control.intervention_active);
    assert_eq!(control.selected_output, control.baseline_output);
    assert_eq!(control.atom_ids, vec![other_layer.atom_id]);

    let map_phase = server
        .evaluate_map_attention_layer(
            RoleId::SensoryPerception,
            4,
            0,
            1,
            42,
            1,
            &q,
            phase,
            config(PhaseSource::MapT2),
        )
        .unwrap();
    assert_eq!(map_phase.phase_source, PhaseSource::MapT2);
    assert_eq!(map_phase.map_phases.len(), 2);
    assert!((map_phase.toroidal_weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);

    let fit = server
        .fit_map_attention_gain(
            RoleId::SensoryPerception,
            4,
            &[MapAttentionFitSample {
                kv_head: 0,
                sequence_id: 1,
                token_id: 42,
                query_position: 1,
                query: q.clone(),
                query_phase: phase,
            }],
            PhaseSource::PairAngles,
            config(PhaseSource::PairAngles).baseline_dot_scale,
            64,
            &[0.5, 1.0, 2.0, 4.0],
        )
        .unwrap();
    assert_eq!(fit.samples, 1);
    assert_eq!(fit.grid_results.len(), 4);
    assert!(fit.best_mean_relative_l2_error <= fit.drop_in_mean_relative_l2_error);
}

#[test]
fn invalid_gain_empty_silo_and_future_only_fail_closed() {
    let mut server = MonsterHybridServer::boot_default(weights()).unwrap();
    let q = head(1.0);
    let phase = Phase {
        theta_f: Q32_31::ZERO,
        theta_s: Q32_31::ZERO,
    };
    let mut bad = config(PhaseSource::MapT2);
    bad.gain = f64::NAN;
    assert!(
        server
            .evaluate_map_attention_layer(RoleId::SensoryPerception, 4, 0, 1, 1, 0, &q, phase, bad)
            .is_err()
    );
    assert!(
        server
            .evaluate_map_attention_layer(
                RoleId::SensoryPerception,
                4,
                0,
                1,
                1,
                0,
                &q,
                phase,
                config(PhaseSource::MapT2)
            )
            .is_err()
    );
    server
        .ingest_attention_kv(
            &tap(4, 7),
            RoleId::SensoryPerception,
            id(4, 5),
            &q,
            &vec![1.0; 128],
        )
        .unwrap();
    assert!(
        server
            .evaluate_map_attention_layer(
                RoleId::SensoryPerception,
                4,
                0,
                1,
                1,
                4,
                &q,
                phase,
                config(PhaseSource::MapT2)
            )
            .is_err()
    );
}

#[test]
fn repeated_kv_slot_uses_newest_map_atom_once() {
    let mut server = MonsterHybridServer::boot_default(weights()).unwrap();
    let q = head(1.0);
    let old = server
        .ingest_attention_kv(
            &tap(4, 40),
            RoleId::SensoryPerception,
            id(4, 0),
            &q,
            &vec![1.0; 128],
        )
        .unwrap();
    let new = server
        .ingest_attention_kv(
            &tap(4, 41),
            RoleId::SensoryPerception,
            id(4, 0),
            &q,
            &vec![2.0; 128],
        )
        .unwrap();
    assert_ne!(old.atom_id, new.atom_id);
    let phase = server.swarm_map.global_atom(new.atom_id).unwrap().phase;
    let result = server
        .evaluate_map_attention_layer(
            RoleId::SensoryPerception,
            4,
            0,
            1,
            42,
            0,
            &q,
            phase,
            config(PhaseSource::PairAngles),
        )
        .unwrap();
    assert_eq!(result.atom_ids, vec![new.atom_id]);
    assert_eq!(result.selected_output, vec![2.0; 128]);
}

#[test]
fn vault_round_trip_and_cross_epoch_adaptation_preserve_map_attention() {
    let mut server = MonsterHybridServer::boot_default(weights()).unwrap();
    let q = head(1.0);
    let rec = server
        .ingest_attention_kv(
            &tap(4, 31),
            RoleId::SensoryPerception,
            id(4, 0),
            &q,
            &vec![0.5; 128],
        )
        .unwrap();
    assert_eq!(server.swarm_map.silo(RoleId::SensoryPerception).unwrap()
        .typed_kv_range(4, 0, 1, 0, 0).unwrap(), vec![rec.atom_id]);
    let phase = server.swarm_map.global_atom(rec.atom_id).unwrap().phase;
    let before = server
        .evaluate_map_attention_layer(
            RoleId::SensoryPerception,
            4,
            0,
            1,
            3,
            0,
            &q,
            phase,
            config(PhaseSource::PairAngles),
        )
        .unwrap();

    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "atoma-toroidal-{}-{stamp}.atoma_map",
        std::process::id()
    ));
    server.swarm_map.save_to_binary_vault(&path).unwrap();
    let mut loaded = MonsterHybridServer::boot_default(weights()).unwrap();
    loaded.load_or_init_persistent_vault(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(loaded.swarm_map.silo(RoleId::SensoryPerception).unwrap()
        .typed_kv_range(4, 0, 1, 0, 0).unwrap(), vec![rec.atom_id]);
    let after = loaded
        .evaluate_map_attention_layer(
            RoleId::SensoryPerception,
            4,
            0,
            1,
            3,
            0,
            &q,
            phase,
            config(PhaseSource::PairAngles),
        )
        .unwrap();
    assert_eq!(before.atom_ids, after.atom_ids);
    assert_eq!(before.toroidal_output, after.toroidal_output);

    let new_weights = FrozenBaseWeights::freeze("next-model-epoch", 52, vec![0.75; 128]);
    loaded.hot_swap_model_backend(new_weights).unwrap();
    let adapted = loaded
        .evaluate_map_attention_layer(
            RoleId::SensoryPerception,
            4,
            0,
            1,
            3,
            0,
            &q,
            phase,
            config(PhaseSource::PairAngles),
        )
        .unwrap();
    assert_eq!(adapted.atom_ids, before.atom_ids);
    assert!(adapted.selected_output.iter().all(|x| x.is_finite()));
}
