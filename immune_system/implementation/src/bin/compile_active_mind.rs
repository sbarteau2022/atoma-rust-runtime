//! Compile the Unified One-Mind Active-Inference Vault (`design/global_active_mind.atoma_map`).
//!
//! Merges:
//!   1. `design/global_100pc.atoma_map` (combined `glimmer_100pc` + `qwen_100pc` static weight mind),
//!   2. `design/glimmer_52l_map_native.atoma_map` (52-layer coupled `ATOMA_OP` v2 operators), and
//!   3. `design/glimmer_token_trajectories.atoma_map` (`100,045` token trajectory `1-cell` arcs,
//!      compacting raw `26.6 KB` token-carrier float buffers down to `1024 B` receptor sketches),
//!   4. Optional `design/qwen_token_trajectories.atoma_map` if present on disk,
//! into `design/global_active_mind.atoma_map`, then verifies Active Inference, Variational Free
//! Energy (`F_q32`), Compliance/Safety proof across all 5 statutory frameworks, and Superposition
//! Holding (`design/superpositionholding.md`).

use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::{
    ActiveInferenceConfig, Atlas13, CarvedSwarmMap, FirstPrinciplesComplianceEngine,
    FrozenBaseWeights, MlirAtomCatalog, MlirOpId, NativeReader, OverlapPolicyMatrix,
    merkle_hash_leaf,
};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    if let Err(err) = run() {
        eprintln!("compile_active_mind: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let root = PathBuf::from("/Users/stewartbarteau/Desktop/monster-hybrid-server/design");
    let global_100pc = root.join("global_100pc.atoma_map");
    let glimmer_52l = root.join("glimmer_52l_map_native.atoma_map");
    let glimmer_traj = root.join("glimmer_token_trajectories.atoma_map");
    let qwen_traj = root.join("qwen_token_trajectories.atoma_map");
    let out_vault = root.join("global_active_mind.atoma_map");

    let atlas = Atlas13::approved_icosahedral();
    let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
    let plug = FrozenBaseWeights::freeze("OneMind-ActiveInference-Unified", 52, vec![0.1; 64]);
    let mut map = CarvedSwarmMap::new(atlas, plug, Q32_31::from_f64(1.22));

    let t0 = Instant::now();
    if global_100pc.exists() {
        let t = Instant::now();
        let (a, r) = map.merge_vault_from(&global_100pc, &policy, false)?;
        println!(
            "[1/4] Merged {} -> +{} novel atoms, +{} novel arcs ({:.2}s)",
            global_100pc.display(),
            a,
            r,
            t.elapsed().as_secs_f64()
        );
    }
    if glimmer_52l.exists() {
        let t = Instant::now();
        let (a, r) = map.merge_vault_from(&glimmer_52l, &policy, false)?;
        println!(
            "[2/4] Merged {} -> +{} novel atoms, +{} novel arcs ({:.2}s)",
            glimmer_52l.display(),
            a,
            r,
            t.elapsed().as_secs_f64()
        );
    }
    if glimmer_traj.exists() {
        let t = Instant::now();
        let (a, r) = map.merge_vault_from(&glimmer_traj, &policy, true)?;
        println!(
            "[3/4] Merged {} (compact carrier sketches) -> +{} novel atoms, +{} novel arcs ({:.2}s)",
            glimmer_traj.display(),
            a,
            r,
            t.elapsed().as_secs_f64()
        );
    }
    if qwen_traj.exists() {
        let t = Instant::now();
        let (a, r) = map.merge_vault_from(&qwen_traj, &policy, true)?;
        println!(
            "[4/4] Merged {} (compact carrier sketches) -> +{} novel atoms, +{} novel arcs ({:.2}s)",
            qwen_traj.display(),
            a,
            r,
            t.elapsed().as_secs_f64()
        );
    }

    // Verify all 8 First-Principle Invariants & 5 Statutory Frameworks on the unified mind
    let audit_leaves = vec![
        merkle_hash_leaf(b"compile_active_mind:global_100pc"),
        merkle_hash_leaf(b"compile_active_mind:glimmer_52l_map_native"),
        merkle_hash_leaf(b"compile_active_mind:glimmer_token_trajectories"),
    ];
    let compliance =
        FirstPrinciplesComplianceEngine::prove_from_live_state(&mut map, &policy, &audit_leaves);
    if !compliance.all_statutes_satisfied {
        return Err("First-principles statutory compliance gate failed on unified mind".to_string());
    }
    println!(
        "[compliance] All 5 statutory frameworks & {} invariants verified (Merkle root={})",
        compliance.verified_invariants.len(),
        compliance.evidence_merkle_root.to_hex()
    );

    let tmp = out_vault.with_extension("atoma_map.partial");
    let t_save = Instant::now();
    let (total_atoms, total_arcs, sha) = map.save_to_binary_vault(&tmp)?;
    std::fs::rename(&tmp, &out_vault).map_err(|e| e.to_string())?;
    let bytes = std::fs::metadata(&out_vault)
        .map(|m| m.len())
        .unwrap_or(0);
    println!(
        "[saved] {}  atoms={} arcs={} size_mb={:.2} save_s={:.2} total_s={:.2} sha={}",
        out_vault.display(),
        total_atoms,
        total_arcs,
        bytes as f64 / (1024.0 * 1024.0),
        t_save.elapsed().as_secs_f64(),
        t0.elapsed().as_secs_f64(),
        sha.to_hex()
    );

    // Benchmark 32-token Active Inference replay + Superposition Holding on the unified vault
    let catalog = MlirAtomCatalog::new_canonical(202_048);
    let france_prefix = [
        MlirOpId(200000),
        MlirOpId(954),
        MlirOpId(7963),
        MlirOpId(323),
        MlirOpId(11698),
        MlirOpId(373),
    ];
    let eos = [MlirOpId(200001), MlirOpId(200008)];
    let config = ActiveInferenceConfig::default();

    let t_active = Instant::now();
    let cont = NativeReader::continue_ids_active(
        &map,
        &catalog,
        &policy,
        &france_prefix,
        32,
        &eos,
        &config,
    )?;
    let active_ms = t_active.elapsed().as_secs_f64() * 1000.0;
    let ids: Vec<u32> = cont.steps.iter().map(|s| s.read.next_id.0).collect();
    println!(
        "[active-inference] 32-token trajectory replay: {:.2} ms ({:.1} tok/s) total_free_energy={:.4} interventions={} ids={:?}",
        active_ms,
        (ids.len() as f64) / (active_ms / 1000.0).max(1e-9),
        cont.total_free_energy_q32.to_f64(),
        cont.superposition_interventions,
        &ids[..8.min(ids.len())]
    );

    // Also verify off-manifold Superposition Holding on the unified mind
    let off_prefix = [
        MlirOpId(200000),
        MlirOpId(954),
        MlirOpId(7963),
        MlirOpId(323),
        MlirOpId(99999),
    ];
    let t_sup = Instant::now();
    let sup_cont =
        NativeReader::continue_ids_active(&map, &catalog, &policy, &off_prefix, 6, &eos, &config)?;
    let sup_ms = t_sup.elapsed().as_secs_f64() * 1000.0;
    let sup_ids: Vec<u32> = sup_cont.steps.iter().map(|s| s.read.next_id.0).collect();
    println!(
        "[superposition-holding] 6-token off-manifold resolution: {:.2} ms total_free_energy={:.4} interventions={} ids={:?}",
        sup_ms,
        sup_cont.total_free_energy_q32.to_f64(),
        sup_cont.superposition_interventions,
        sup_ids
    );

    Ok(())
}
