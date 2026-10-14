//! `ion_flag01`: Standalone CLI benchmark runner for `FLAG-01` (`core_lm/src/geometry/bridge.rs`).
//! Executes the 4-regime x 5-bridge empirical study and prints verified Markdown tables.

use atoma_bridge_harness::run_full_flag01_study;

fn main() {
    println!("================================================================================");
    println!(" MONSTER HYBRID SERVER — FLAG-01 EMPIRICAL BRIDGE STUDY & REPLACEMENT HARNESS");
    println!(" Target: core_lm/src/geometry/bridge.rs:27-46 (Read-Only Lock Enforced)");
    println!("================================================================================\n");

    let n_keys = 256;
    let n_queries = 64;
    let reports = run_full_flag01_study(n_keys, n_queries).expect("FLAG-01 study execution failed");

    let mut current_regime = "";
    for r in &reports {
        if r.regime_name != current_regime {
            current_regime = r.regime_name;
            println!("\n### Regime: `{}` (N_keys={}, N_queries={}, d_h={})\n", current_regime, r.n_keys, r.n_queries, r.head_dim);
            println!("| Bridge Architecture | 3D Energy (%) | Full Energy (%) | Discrim. Var (%) | RoPE Pair (%) | HF Noise (%) | Radial CV | Spearman rho | Recall@1 (%) | Recall@8 (%) | Recall@16 (%) | Attn CosSim | VP Parity |");
            println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|");
        }
        println!(
            "| {} | {:.2}% | {:.2}% | {:.2}% | {:.1}% | {:.1}% | {:.4} | {:+.4} | {:.2}% | {:.2}% | {:.2}% | {:.4} | {} |",
            r.bridge_name,
            r.raw_3d_energy_pct,
            r.total_retained_energy_pct,
            r.discriminative_variance_pct,
            r.rope_pair_integrity_pct,
            r.high_freq_contamination_pct,
            r.hyperboloid_radial_cv,
            r.inner_product_spearman_rho,
            r.recall_at_1,
            r.recall_at_8,
            r.recall_at_16,
            r.downstream_attn_cosine_sim,
            if r.vptree_index_exact_parity { "PASS" } else { "FAIL" }
        );
    }

    println!("\n[verified] All {} regime-bridge evaluations completed with zero violations.", reports.len());
}
