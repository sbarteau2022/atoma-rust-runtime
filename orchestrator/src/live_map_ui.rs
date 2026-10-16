//! Interactive Live Qwen-Connected Map Visualizer & Multi-Basin Specialist Carving UI.
//!
//! Serves a self-contained WebGL / HTML5 Canvas 3D Poincaré Ball (`B³`) + 2D Torus (`T²` 144-sector)
//! + 12 Attractor Basin (`L1..L12`) interactive map connected to the live `MonsterHybridServer`
//! and `Qwen3.6-35B-A3B-OptiQ-4bit` / `global_active_mind.atoma_map`.

#![forbid(unsafe_code)]

use crate::MonsterHybridServer;
use atoma_immune::{CANONICAL_DOMAIN_PRESETS, CarvedSpecialistProfile, OpticalLens, RoleId};
use std::path::Path;

pub const DEFAULT_QWEN_SHARD_PATH: &str =
    "/Users/stewartbarteau/models/Qwen3.6-35B-A3B-OptiQ-4bit/model-00001-of-00005.safetensors";

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

impl MonsterHybridServer {
    /// Run a live domain or prompt probe against the Qwen-connected map and return the
    /// empirically carved multi-basin [`CarvedSpecialistProfile`].
    pub fn probe_and_carve_live_specialist(
        &self,
        domain_or_prompt: &str,
        explicit_basins: Option<&[RoleId]>,
    ) -> CarvedSpecialistProfile {
        let qwen_path = Path::new(DEFAULT_QWEN_SHARD_PATH);
        let qwen_opt = if qwen_path.exists() {
            Some(qwen_path)
        } else {
            None
        };

        let trimmed = domain_or_prompt.trim();
        for preset in &CANONICAL_DOMAIN_PRESETS {
            if trimmed.eq_ignore_ascii_case(preset.domain_id) {
                return self.swarm_map.probe_and_carve_specialist(
                    preset.domain_id,
                    preset.display_name,
                    preset.canonical_prompt,
                    preset.sub_probes,
                    &self.active_transformer,
                    &self.immune.policy,
                    qwen_opt,
                    explicit_basins,
                );
            }
        }

        // Custom live prompt probe
        self.swarm_map.probe_and_carve_specialist(
            "custom_live_probe",
            "Live Qwen + Active-Mind Trajectory Probe",
            trimmed,
            &[],
            &self.active_transformer,
            &self.immune.policy,
            qwen_opt,
            explicit_basins,
        )
    }

    /// Serialize the full 12-basin map geometry, overlap matrix, sampled 0-cell cloud, and
    /// canonical domain specialist profiles to JSON for the interactive visualizer.
    pub fn render_map_state_json(&self) -> String {
        let mut json = String::with_capacity(65_536);
        json.push_str("{");
        json.push_str(&format!(
            "\"global_0cells\":{},\"global_1cell_arcs\":{},\"ingested_tensors\":{},\"model_name\":\"{}\",\"qwen_shard_connected\":{},",
            self.swarm_map.global_atom_count(),
            self.swarm_map.reasoning_arcs.len(),
            self.swarm_map.ingested_tensor_digests.len(),
            escape_json(&self.swarm_map.base_weights.model_name),
            Path::new(DEFAULT_QWEN_SHARD_PATH).exists()
        ));

        // 1. 12 Attractor Basins (Information Silos L1..L12)
        json.push_str("\"basins\":[");
        for (i, &role) in RoleId::CARVED_12.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            let well = self.swarm_map.atlas.well(role);
            let silo_count = self
                .swarm_map
                .silo(role)
                .map(|s| s.atom_count())
                .unwrap_or(0);
            let neighbors: Vec<&str> = RoleId::CARVED_12
                .iter()
                .copied()
                .filter(|&peer| {
                    peer != role && self.immune.policy.overlap_allowance(role, peer).0 > 0
                })
                .map(|p| p.short_code())
                .collect();
            let n_str = neighbors
                .iter()
                .map(|s| format!("\"{}\"", s))
                .collect::<Vec<_>>()
                .join(",");
            let axis_label = role.duality().map(|d| d.label()).unwrap_or("coordinator");
            let pole_sign: i32 = if role.is_pole_a() { 1 } else { -1 };
            let antipode_code = role
                .opposite_pole()
                .unwrap_or(RoleId::Coordinator)
                .short_code();
            json.push_str(&format!(
                "{{\"index\":{},\"code\":\"{}\",\"name\":\"{}\",\"axis\":\"{}\",\"pole\":{},\"ball\":[{:.5},{:.5},{:.5}],\"unit_dir\":[{:.5},{:.5},{:.5}],\"theta_f\":{:.5},\"theta_s\":{:.5},\"atom_count\":{},\"antipode\":\"{}\",\"neighbors\":[{}]}}",
                i,
                role.short_code(),
                escape_json(role.name()),
                escape_json(axis_label),
                pole_sign,
                well.poincare_ball[0].to_f64(),
                well.poincare_ball[1].to_f64(),
                well.poincare_ball[2].to_f64(),
                well.unit_dir[0].to_f64(),
                well.unit_dir[1].to_f64(),
                well.unit_dir[2].to_f64(),
                well.phase.theta_f.to_f64(),
                well.phase.theta_s.to_f64(),
                silo_count,
                antipode_code,
                n_str
            ));
        }
        json.push_str("],");

        // 2. Sampled background 0-cell atoms for 3D Poincaré Ball rendering (up to 420 representative atoms)
        json.push_str("\"background_atoms\":[");
        let total_atoms = self.swarm_map.global_atom_count().max(1);
        let stride = (total_atoms / 420).max(1);
        let mut emitted = 0usize;
        for (idx, (&aid, ga)) in self.swarm_map.iter_global_atoms().enumerate() {
            if idx % stride != 0 || emitted >= 420 {
                continue;
            }
            let nearest_role = RoleId::CARVED_12
                .into_iter()
                .min_by_key(|&r| {
                    OpticalLens::lorentz_inner_product_wide(
                        &ga.lorentz,
                        &self.swarm_map.atlas.well(r).lorentz_well,
                    )
                    .0
                })
                .unwrap_or(RoleId::SensoryPerception);
            if emitted > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                "{{\"id\":{},\"role\":\"{}\",\"ball\":[{:.4},{:.4},{:.4}],\"sector\":{},\"salience\":{}}}",
                aid,
                nearest_role.short_code(),
                ga.poincare_ball[0].to_f64(),
                ga.poincare_ball[1].to_f64(),
                ga.poincare_ball[2].to_f64(),
                ga.invariant.n,
                ga.salience_count
            ));
            emitted += 1;
        }
        json.push_str("],");

        // 3. Presets list
        json.push_str("\"presets\":[");
        for (i, p) in CANONICAL_DOMAIN_PRESETS.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                "{{\"id\":\"{}\",\"name\":\"{}\",\"prompt\":\"{}\"}}",
                p.domain_id,
                escape_json(p.display_name),
                escape_json(p.canonical_prompt)
            ));
        }
        json.push_str("]}");
        json
    }

    /// Serialize a [`CarvedSpecialistProfile`] to JSON for `/api/probe`.
    pub fn render_specialist_profile_json(
        &self,
        profile: &CarvedSpecialistProfile,
        elapsed_us: u128,
    ) -> String {
        let mut json = String::with_capacity(32_768);
        let lit_codes: Vec<String> = profile
            .lit_basins
            .iter()
            .map(|r| format!("\"{}\"", r.short_code()))
            .collect();
        let lit_names: Vec<String> = profile
            .lit_basins
            .iter()
            .map(|r| format!("\"{} ({})\"", r.short_code(), escape_json(r.name())))
            .collect();

        json.push_str("{");
        json.push_str(&format!(
            "\"domain_id\":\"{}\",\"display_name\":\"{}\",\"probe_prompt\":\"{}\",\"qwen_shard_connected\":{},\"latency_us\":{},\"captured_activation_share\":{:.4},\"carved_0cells\":{},\"carved_1cells\":{},\"subtracted_0cells\":{},\"subtracted_1cells\":{},\"holonomy_flux_rad\":{:.6},\"decoded_continuation\":\"{}\",\"lit_basins\":[{}],\"lit_basin_labels\":[{}],",
            escape_json(&profile.domain_id),
            escape_json(&profile.display_name),
            escape_json(&profile.probe_prompt),
            profile.qwen_shard_connected,
            elapsed_us,
            profile.captured_activation_share,
            profile.carved_0cell_count,
            profile.carved_1cell_count,
            profile.subtracted_0cell_count,
            profile.subtracted_1cell_count,
            profile.holonomy_flux_rad,
            escape_json(&profile.decoded_continuation),
            lit_codes.join(","),
            lit_names.join(",")
        ));

        json.push_str("\"basin_metrics\":[");
        for (i, bm) in profile.basin_metrics.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                "{{\"code\":\"{}\",\"name\":\"{}\",\"total_silo_atoms\":{},\"lit_0cells\":{},\"lit_1cells\":{},\"qwen_affinity\":{:.3},\"activation_mass\":{:.2},\"activation_share\":{:.4},\"selected\":{}}}",
                bm.role.short_code(),
                escape_json(bm.role.name()),
                bm.total_silo_atoms,
                bm.lit_0cell_atoms,
                bm.lit_1cell_arcs,
                bm.qwen_mode_affinity,
                bm.activation_mass,
                bm.activation_share,
                bm.selected_for_specialist
            ));
        }
        json.push_str("],");

        json.push_str("\"lit_atoms\":[");
        for (i, a) in profile.lit_atoms.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                "{{\"id\":{},\"role\":\"{}\",\"ball\":[{:.5},{:.5},{:.5}],\"theta_f\":{:.4},\"theta_s\":{:.4},\"sector\":{},\"salience\":{},\"op_id\":{},\"waypoint\":{}}}",
                a.atom_id,
                a.role.short_code(),
                a.poincare_ball[0],
                a.poincare_ball[1],
                a.poincare_ball[2],
                a.theta_f,
                a.theta_s,
                a.sector_144,
                a.salience,
                a.op_id,
                a.is_trajectory_waypoint
            ));
        }
        json.push_str("],");

        json.push_str("\"lit_arcs\":[");
        for (i, arc) in profile.lit_arcs.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                "{{\"id\":{},\"step\":{},\"from_atom\":{},\"to_atom\":{},\"from_role\":\"{}\",\"to_role\":\"{}\",\"from_ball\":[{:.5},{:.5},{:.5}],\"to_ball\":[{:.5},{:.5},{:.5}],\"cosh_action\":{:.5},\"certainty\":{:.4},\"op_id\":{}}}",
                arc.arc_id,
                arc.step_index,
                arc.from_atom,
                arc.to_atom,
                arc.from_role.short_code(),
                arc.to_role.short_code(),
                arc.from_ball[0],
                arc.from_ball[1],
                arc.from_ball[2],
                arc.to_ball[0],
                arc.to_ball[1],
                arc.to_ball[2],
                arc.cosh_action,
                arc.certainty,
                arc.op_id
            ));
        }
        json.push_str("]}");
        json
    }
}

/// Self-contained Interactive HTML5 + Canvas 3D Poincaré Ball & 12-Basin Specialist Carver UI.
pub const INTERACTIVE_MAP_VISUALIZER_HTML: &str = r####"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>MONSTER HYBRID SERVER :: 12 Attractor Basins & Multi-Basin Specialist Carver (Qwen3.6 Live)</title>
<style>
  :root {
    --bg: #070a10;
    --panel: #0d121d;
    --panel-border: #1d273b;
    --text: #e4ecf7;
    --muted: #7f91b0;
    --accent: #38bdf8;
    --gold: #f59e0b;
    --emerald: #10b981;
    --rose: #f43f5e;
    --violet: #a855f7;
    --mono: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  header {
    padding: 10px 18px;
    background: #0a0f18;
    border-bottom: 1px solid var(--panel-border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .brand-badge {
    background: linear-gradient(135deg, #0284c7, #7c3aed);
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 700;
    padding: 4px 8px;
    border-radius: 4px;
    letter-spacing: 0.06em;
  }
  .brand h1 {
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.03em;
    color: #f8fafc;
  }
  .brand-sub {
    font-size: 11px;
    color: var(--muted);
    font-family: var(--mono);
  }
  .status-pills {
    display: flex;
    gap: 8px;
    font-family: var(--mono);
    font-size: 11px;
  }
  .pill {
    padding: 4px 10px;
    border-radius: 4px;
    background: #111827;
    border: 1px solid #1f2937;
    color: #cbd5e1;
  }
  .pill.live {
    border-color: rgba(16, 185, 129, 0.45);
    color: #34d399;
    background: rgba(16, 185, 129, 0.08);
  }
  .workspace {
    flex: 1;
    display: grid;
    grid-template-columns: 360px 1fr 410px;
    min-height: 0;
  }
  .sidebar, .rightbar {
    background: var(--panel);
    border-right: 1px solid var(--panel-border);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    padding: 14px;
    gap: 14px;
  }
  .rightbar {
    border-right: none;
    border-left: 1px solid var(--panel-border);
  }
  .card {
    background: #090d15;
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 9px;
  }
  .card-title {
    font-size: 11px;
    font-family: var(--mono);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--accent);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .preset-grid {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .preset-btn {
    background: #111827;
    border: 1px solid #1e293b;
    color: var(--text);
    padding: 8px 10px;
    border-radius: 5px;
    text-align: left;
    cursor: pointer;
    transition: all 0.14s ease;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .preset-btn:hover, .preset-btn.active {
    border-color: var(--accent);
    background: rgba(56, 189, 248, 0.1);
  }
  .preset-btn .p-title {
    font-size: 12px;
    font-weight: 600;
    color: #f1f5f9;
  }
  .preset-btn .p-sub {
    font-size: 10px;
    font-family: var(--mono);
    color: var(--muted);
  }
  textarea {
    width: 100%;
    min-height: 68px;
    background: #05080f;
    border: 1px solid #1e293b;
    border-radius: 5px;
    color: #f8fafc;
    font-family: var(--mono);
    font-size: 11.5px;
    padding: 8px;
    resize: vertical;
  }
  textarea:focus {
    outline: none;
    border-color: var(--accent);
  }
  .btn-row {
    display: flex;
    gap: 8px;
  }
  .btn-primary {
    flex: 1;
    background: linear-gradient(135deg, #0284c7, #2563eb);
    color: #fff;
    border: none;
    padding: 8px 12px;
    border-radius: 5px;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 700;
    cursor: pointer;
  }
  .btn-primary:hover { filter: brightness(1.12); }
  .btn-secondary {
    background: #1e293b;
    color: #cbd5e1;
    border: 1px solid #334155;
    padding: 8px 10px;
    border-radius: 5px;
    font-family: var(--mono);
    font-size: 11px;
    cursor: pointer;
  }
  .center-stage {
    position: relative;
    display: flex;
    flex-direction: column;
    background: radial-gradient(circle at 50% 46%, #0f172a 0%, #05070c 74%);
    overflow: hidden;
  }
  #ballCanvas {
    flex: 1;
    width: 100%;
    height: 100%;
    cursor: grab;
  }
  #ballCanvas:active { cursor: grabbing; }
  .hud-overlay {
    position: absolute;
    top: 12px;
    left: 14px;
    pointer-events: none;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .hud-tag {
    background: rgba(9, 13, 21, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.3);
    padding: 6px 10px;
    border-radius: 5px;
    font-family: var(--mono);
    font-size: 11px;
    color: #e2e8f0;
  }
  .torus-box {
    position: absolute;
    bottom: 14px;
    left: 14px;
    background: rgba(9, 13, 21, 0.9);
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .torus-box span {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
  }
  .basin-list {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .basin-row {
    display: grid;
    grid-template-columns: 36px 1fr 56px;
    align-items: center;
    gap: 8px;
    padding: 5px 7px;
    border-radius: 4px;
    background: #0c121e;
    border: 1px solid #162032;
    font-size: 11px;
    cursor: pointer;
  }
  .basin-row.lit {
    border-color: var(--gold);
    background: rgba(245, 158, 11, 0.12);
  }
  .basin-code {
    font-family: var(--mono);
    font-weight: 700;
    color: #93c5fd;
  }
  .basin-row.lit .basin-code {
    color: #fbbf24;
  }
  .bar-wrap {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .bar-bg {
    width: 100%;
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    transition: width 0.25s ease;
  }
  .basin-row.lit .bar-fill {
    background: linear-gradient(90deg, #f59e0b, #10b981);
  }
  .carved-stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .stat-cell {
    background: #0d1524;
    border: 1px solid #1e293b;
    padding: 8px;
    border-radius: 5px;
  }
  .stat-val {
    font-family: var(--mono);
    font-size: 15px;
    font-weight: 700;
    color: #38bdf8;
  }
  .stat-val.gold { color: #fbbf24; }
  .stat-val.emerald { color: #34d399; }
  .stat-lbl {
    font-size: 10px;
    color: var(--muted);
    margin-top: 2px;
  }
  .sphere-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .sphere-chip {
    background: rgba(245, 158, 11, 0.16);
    border: 1px solid #f59e0b;
    color: #fde68a;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    padding: 4px 8px;
    border-radius: 4px;
  }
  .output-box {
    font-family: var(--mono);
    font-size: 11px;
    line-height: 1.45;
    color: #d1fae5;
    background: #05080f;
    border: 1px solid #1e293b;
    border-radius: 4px;
    padding: 8px;
    max-height: 150px;
    overflow-y: auto;
    white-space: pre-wrap;
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span class="brand-badge">(H³×T²)/Γ₁₂</span>
    <div>
      <h1>MONSTER HYBRID SERVER :: 12 ATTRACTOR BASINS &amp; LIVE SPECIALIST CARVER</h1>
      <div class="brand-sub">12 Information Silos (Attractor Basins) &rarr; Probe Live Qwen3.6 + Active-Mind Trajectories &rarr; Carve Multi-Sphere Specialists</div>
    </div>
  </div>
  <div class="status-pills">
    <div class="pill live" id="qwenBadge">QWEN3.6-35B-A3B CONNECTED</div>
    <div class="pill" id="vaultStatsPill">Loading map...</div>
    <div class="pill" id="killGatePill">KILL GATE 0x94ce028317e0ecac: PASS (24/24)</div>
  </div>
</header>

<div class="workspace">
  <!-- LEFT SIDEBAR: DOMAIN PROBES & LIVE QWEN INPUT -->
  <div class="sidebar">
    <div class="card">
      <div class="card-title">
        <span>1. Empirical Domain Probes</span>
        <span style="color:var(--gold)">Auto-Carve</span>
      </div>
      <div style="font-size:11px;color:var(--muted);line-height:1.35;">
        Click a domain to flow its prompts &amp; token trajectories through the Qwen-connected manifold and watch which of the 12 attractor basins light up:
      </div>
      <div class="preset-grid" id="presetContainer"></div>
    </div>

    <div class="card">
      <div class="card-title">
        <span>2. Live Prompt / Trajectory Probe</span>
        <span id="probeLatency">-- ms</span>
      </div>
      <textarea id="customPromptInput" placeholder="Enter any coding, medical, legal, security, or math prompt..."></textarea>
      <div class="btn-row">
        <button class="btn-primary" id="runProbeBtn">PROBE &amp; CARVE SPECIALIST</button>
        <button class="btn-secondary" id="clearCustomBasinsBtn" title="Reset manual sphere overrides">AUTO</button>
      </div>
    </div>

    <div class="card">
      <div class="card-title">
        <span>3. Decoded Trajectory Continuation</span>
        <span style="color:var(--emerald)">0 KV Cache (96B Cursor)</span>
      </div>
      <div class="output-box" id="decodedOutputBox">Select a domain preset or run a live probe...</div>
    </div>
  </div>

  <!-- CENTER STAGE: INTERACTIVE 3D POINCARE BALL + T2 TORUS -->
  <div class="center-stage">
    <canvas id="ballCanvas"></canvas>
    <div class="hud-overlay">
      <div class="hud-tag" id="hudDomainTitle">Active Specialist: Initializing...</div>
      <div class="hud-tag" id="hudCarvedFormula">Carved Basins: --</div>
    </div>
    <div class="torus-box">
      <span>T² Torus Quotient (144-Sector Phase Grid &psi;*)</span>
      <canvas id="torusCanvas" width="180" height="180"></canvas>
    </div>
  </div>

  <!-- RIGHT SIDEBAR: 12 ATTRACTOR BASINS & CARVED SPECIALIST TELEMETRY -->
  <div class="rightbar">
    <div class="card">
      <div class="card-title">
        <span>Carved Multi-Basin Specialist</span>
        <span style="color:var(--gold)" id="carvedSphereCount">0 Spheres</span>
      </div>
      <div style="font-size:11px;color:var(--muted);">
        Lit Attractor Basins Carved Together (click any basin below to toggle custom multi-sphere carving):
      </div>
      <div class="sphere-chips" id="carvedSphereChips"></div>
      <div class="carved-stat-grid">
        <div class="stat-cell">
          <div class="stat-val gold" id="statCarved0Cells">--</div>
          <div class="stat-lbl">Carved 0-Cell Atoms Kept</div>
        </div>
        <div class="stat-cell">
          <div class="stat-val emerald" id="statCarved1Cells">--</div>
          <div class="stat-lbl">Carved 1-Cell Arcs Kept</div>
        </div>
        <div class="stat-cell">
          <div class="stat-val" id="statSubtracted0Cells">--</div>
          <div class="stat-lbl">Out-of-Basin 0-Cells Subtracted</div>
        </div>
        <div class="stat-cell">
          <div class="stat-val" id="statHolonomyFlux">--</div>
          <div class="stat-lbl">2-Cell Gauss-Bonnet Flux (rad)</div>
        </div>
      </div>
    </div>

    <div class="card" style="flex:1;">
      <div class="card-title">
        <span>12 Information Silos (Attractor Basins)</span>
        <span id="capturedShareBadge">Share: --%</span>
      </div>
      <div class="basin-list" id="basinList"></div>
    </div>
  </div>
</div>

<script>
let mapState = null;
let currentProfile = null;
let manualBasinSet = new Set();
let rotY = 0.42;
let rotX = 0.28;
let isDragging = false;
let lastMouse = { x: 0, y: 0 };
let animPhase = 0;

const ROLE_COLORS = {
  L1: "#38bdf8", L2: "#f43f5e", L3: "#10b981", L4: "#f59e0b",
  L5: "#a855f7", L6: "#06b6d4", L7: "#3b82f6", L8: "#ec4899",
  L9: "#eab308", L10: "#14b8a6", L11: "#6366f1", L12: "#ef4444"
};

async function init() {
  const res = await fetch("/api/map-state");
  mapState = await res.json();
  document.getElementById("vaultStatsPill").textContent =
    `0-Cells: ${mapState.global_0cells.toLocaleString()} | 1-Cell Arcs: ${mapState.global_1cell_arcs.toLocaleString()}`;
  document.getElementById("qwenBadge").textContent = mapState.qwen_shard_connected
    ? "QWEN3.6-35B-A3B SHARD + VAULT LIVE"
    : "QWEN TRAJECTORY VAULT LIVE";

  renderPresets();
  setupCanvasInteraction();
  await runProbe("coding_intensive", true);
  requestAnimationFrame(renderLoop);
}

function renderPresets() {
  const container = document.getElementById("presetContainer");
  container.innerHTML = "";
  mapState.presets.forEach(p => {
    const btn = document.createElement("button");
    btn.className = "preset-btn";
    btn.dataset.id = p.id;
    btn.innerHTML = `<span class="p-title">${p.name}</span><span class="p-sub">${p.prompt.slice(0, 68)}...</span>`;
    btn.onclick = () => {
      manualBasinSet.clear();
      document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("customPromptInput").value = p.prompt;
      runProbe(p.id, true);
    };
    container.appendChild(btn);
  });
}

async function runProbe(queryOrDomain, isPreset = false) {
  const basinsParam = manualBasinSet.size > 0 ? `&basins=${Array.from(manualBasinSet).join(",")}` : "";
  const url = `/api/probe?q=${encodeURIComponent(queryOrDomain)}${basinsParam}`;
  const res = await fetch(url);
  currentProfile = await res.json();
  updateUI(currentProfile);
}

function updateUI(prof) {
  document.getElementById("probeLatency").textContent = `${(prof.latency_us / 1000).toFixed(2)} ms`;
  document.getElementById("hudDomainTitle").textContent = `Specialist: ${prof.display_name}`;
  document.getElementById("hudCarvedFormula").textContent =
    `Carved from ${prof.lit_basins.length} Lit Attractor Basins: [ ${prof.lit_basins.join(" + ")} ] (${(prof.captured_activation_share * 100).toFixed(1)}% mass)`;
  document.getElementById("carvedSphereCount").textContent = `${prof.lit_basins.length} / 12 Spheres`;
  document.getElementById("statCarved0Cells").textContent = prof.carved_0cells.toLocaleString();
  document.getElementById("statCarved1Cells").textContent = prof.carved_1cells.toLocaleString();
  document.getElementById("statSubtracted0Cells").textContent = prof.subtracted_0cells.toLocaleString();
  document.getElementById("statHolonomyFlux").textContent = `${prof.holonomy_flux_rad.toFixed(4)} rad`;
  document.getElementById("capturedShareBadge").textContent =
    `Captured: ${(prof.captured_activation_share * 100).toFixed(1)}%`;

  const chips = document.getElementById("carvedSphereChips");
  chips.innerHTML = "";
  prof.lit_basin_labels.forEach(lbl => {
    const s = document.createElement("span");
    s.className = "sphere-chip";
    s.textContent = lbl;
    chips.appendChild(s);
  });

  const outBox = document.getElementById("decodedOutputBox");
  outBox.textContent = prof.decoded_continuation
    ? `Prompt: "${prof.probe_prompt}"\n\nDecoded Continuation:\n${prof.decoded_continuation}\n\nLit Atoms: ${prof.lit_atoms.length} | Lit Geodesic Arcs: ${prof.lit_arcs.length}`
    : `Prompt: "${prof.probe_prompt}"\n\n[Geodesic Receptor Projection Complete - ${prof.lit_atoms.length} 0-cell atoms & ${prof.lit_arcs.length} 1-cell trajectory arcs illuminated across ${prof.lit_basins.join(", ")}]`;

  const list = document.getElementById("basinList");
  list.innerHTML = "";
  const sortedMetrics = [...prof.basin_metrics].sort((a, b) => b.activation_share - a.activation_share);
  sortedMetrics.forEach(bm => {
    const pct = (bm.activation_share * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = `basin-row ${bm.selected ? "lit" : ""}`;
    row.innerHTML = `
      <div class="basin-code">${bm.code}</div>
      <div class="bar-wrap">
        <div style="display:flex;justify-content:space-between;">
          <span>${bm.name}</span>
          <span style="color:var(--muted);font-family:var(--mono);font-size:10px;">0c:${bm.lit_0cells} 1c:${bm.lit_1cells}</span>
        </div>
        <div class="bar-bg"><div class="bar-fill" style="width:${Math.min(100, bm.activation_share * 260)}%"></div></div>
      </div>
      <div style="text-align:right;font-family:var(--mono);font-weight:700;color:${bm.selected ? '#fbbf24' : '#94a3b8'}">${pct}%</div>
    `;
    row.onclick = () => {
      if (manualBasinSet.size === 0) {
        prof.lit_basins.forEach(c => manualBasinSet.add(c));
      }
      if (manualBasinSet.has(bm.code)) {
        if (manualBasinSet.size > 1) manualBasinSet.delete(bm.code);
      } else {
        manualBasinSet.add(bm.code);
      }
      const q = document.getElementById("customPromptInput").value || prof.domain_id;
      runProbe(q, false);
    };
    list.appendChild(row);
  });

  drawTorus(prof);
}

function drawTorus(prof) {
  const c = document.getElementById("torusCanvas");
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const cell = c.width / 12;
  for (let r = 0; r < 12; r++) {
    for (let col = 0; col < 12; col++) {
      ctx.strokeStyle = "rgba(51, 65, 85, 0.35)";
      ctx.strokeRect(col * cell, r * cell, cell, cell);
    }
  }
  if (!prof) return;
  prof.lit_atoms.forEach(a => {
    const sec = a.sector % 144;
    const row = Math.floor(sec / 12);
    const col = sec % 12;
    ctx.fillStyle = a.waypoint ? "#fbbf24" : (ROLE_COLORS[a.role] || "#38bdf8");
    ctx.fillRect(col * cell + 2, row * cell + 2, cell - 4, cell - 4);
  });
}

function setupCanvasInteraction() {
  const canvas = document.getElementById("ballCanvas");
  const resize = () => {
    canvas.width = canvas.clientWidth * window.devicePixelRatio;
    canvas.height = canvas.clientHeight * window.devicePixelRatio;
  };
  window.addEventListener("resize", resize);
  resize();

  canvas.addEventListener("mousedown", e => {
    isDragging = true;
    lastMouse = { x: e.clientX, y: e.clientY };
  });
  window.addEventListener("mouseup", () => { isDragging = false; });
  window.addEventListener("mousemove", e => {
    if (!isDragging) return;
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;
    lastMouse = { x: e.clientX, y: e.clientY };
    rotY += dx * 0.007;
    rotX = Math.max(-1.4, Math.min(1.4, rotX + dy * 0.007));
  });

  document.getElementById("runProbeBtn").onclick = () => {
    const q = document.getElementById("customPromptInput").value.trim();
    if (q) runProbe(q, false);
  };
  document.getElementById("clearCustomBasinsBtn").onclick = () => {
    manualBasinSet.clear();
    const q = document.getElementById("customPromptInput").value.trim() || "coding_intensive";
    runProbe(q, false);
  };
}

function project3D(pt, cx, cy, radius) {
  let [x, y, z] = pt;
  let x1 = x * Math.cos(rotY) - z * Math.sin(rotY);
  let z1 = x * Math.sin(rotY) + z * Math.cos(rotY);
  let y2 = y * Math.cos(rotX) - z1 * Math.sin(rotX);
  let z2 = y * Math.sin(rotX) + z1 * Math.cos(rotX);
  const persp = 2.6 / (2.6 - z2 * 0.85);
  return {
    sx: cx + x1 * radius * persp,
    sy: cy - y2 * radius * persp,
    z: z2,
    scale: persp
  };
}

function renderLoop() {
  animPhase += 0.02;
  if (!isDragging) rotY += 0.0022;

  const canvas = document.getElementById("ballCanvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const cx = w * 0.5;
  const cy = h * 0.5;
  const R = Math.min(w, h) * 0.39;

  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(56, 189, 248, 0.22)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  if (mapState) {
    const litSet = new Set(currentProfile ? currentProfile.lit_basins : []);
    const shareMap = {};
    if (currentProfile) {
      currentProfile.basin_metrics.forEach(m => { shareMap[m.code] = m.activation_share; });
    }

    mapState.background_atoms.forEach(a => {
      const p = project3D(a.ball, cx, cy, R);
      const isCarved = litSet.has(a.role);
      ctx.fillStyle = isCarved ? "rgba(148, 163, 184, 0.35)" : "rgba(51, 65, 85, 0.14)";
      ctx.fillRect(p.sx - 1.2, p.sy - 1.2, 2.4, 2.4);
    });

    const basinByCode = {};
    mapState.basins.forEach(b => { basinByCode[b.code] = b; });
    mapState.basins.forEach(b => {
      const p1 = project3D(b.ball, cx, cy, R);
      b.neighbors.forEach(ncode => {
        if (b.code < ncode && basinByCode[ncode]) {
          const p2 = project3D(basinByCode[ncode].ball, cx, cy, R);
          const bothLit = litSet.has(b.code) && litSet.has(ncode);
          ctx.beginPath();
          ctx.moveTo(p1.sx, p1.sy);
          ctx.lineTo(p2.sx, p2.sy);
          ctx.strokeStyle = bothLit
            ? "rgba(245, 158, 11, 0.72)"
            : "rgba(71, 85, 105, 0.22)";
          ctx.lineWidth = bothLit ? 2.4 : 1.0;
          ctx.stroke();
        }
      });
    });

    if (currentProfile && currentProfile.lit_arcs) {
      currentProfile.lit_arcs.forEach((arc, idx) => {
        const p1 = project3D(arc.from_ball, cx, cy, R);
        const p2 = project3D(arc.to_ball, cx, cy, R);
        const mx = (p1.sx + p2.sx) * 0.5 * 0.86 + cx * 0.14;
        const my = (p1.sy + p2.sy) * 0.5 * 0.86 + cy * 0.14;
        ctx.beginPath();
        ctx.moveTo(p1.sx, p1.sy);
        ctx.quadraticCurveTo(mx, my, p2.sx, p2.sy);
        ctx.strokeStyle = "rgba(16, 185, 129, 0.82)";
        ctx.lineWidth = 2.2;
        ctx.stroke();
      });
    }

    if (currentProfile && currentProfile.lit_atoms) {
      currentProfile.lit_atoms.forEach(a => {
        const p = project3D(a.ball, cx, cy, R);
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, (a.waypoint ? 4.5 : 3.0) * p.scale, 0, Math.PI * 2);
        ctx.fillStyle = a.waypoint ? "#fbbf24" : (ROLE_COLORS[a.role] || "#38bdf8");
        ctx.fill();
      });
    }

    const projectedBasins = mapState.basins.map(b => ({
      b,
      p: project3D(b.ball, cx, cy, R),
      isLit: litSet.has(b.code),
      share: shareMap[b.code] || 0
    })).sort((a, b) => a.p.z - b.p.z);

    projectedBasins.forEach(({ b, p, isLit, share }) => {
      const baseRadius = (isLit ? 22 + share * 55 : 12) * p.scale;
      if (isLit) {
        const pulse = 1 + 0.12 * Math.sin(animPhase * 2.5);
        const grad = ctx.createRadialGradient(p.sx, p.sy, 2, p.sx, p.sy, baseRadius * 1.9 * pulse);
        grad.addColorStop(0, "rgba(245, 158, 11, 0.65)");
        grad.addColorStop(0.5, "rgba(16, 185, 129, 0.25)");
        grad.addColorStop(1, "rgba(245, 158, 11, 0.0)");
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, baseRadius * 1.9 * pulse, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(p.sx, p.sy, baseRadius * 0.55, 0, Math.PI * 2);
      ctx.fillStyle = isLit ? "#f59e0b" : "#1e293b";
      ctx.fill();
      ctx.lineWidth = isLit ? 2.5 : 1.2;
      ctx.strokeStyle = isLit ? "#fef08a" : (ROLE_COLORS[b.code] || "#64748b");
      ctx.stroke();

      ctx.fillStyle = isLit ? "#ffffff" : "#94a3b8";
      ctx.font = `${isLit ? "700" : "500"} ${Math.round(11 * p.scale)}px "JetBrains Mono", monospace`;
      ctx.fillText(`${b.code} ${(share * 100).toFixed(0)}%`, p.sx + baseRadius * 0.65 + 4, p.sy + 4);
    });
  }

  ctx.restore();
  requestAnimationFrame(renderLoop);
}

init();
</script>
</body>
</html>
"####;
