-- Autonomous Monster-Hybrid Synthesized Schema: `geodesic_vault_and_merkle_ledger.sql`
CREATE TABLE IF NOT EXISTS global_atoms (
    atom_id BIGINT PRIMARY KEY,
    primary_role SMALLINT NOT NULL CHECK (primary_role BETWEEN 0 AND 11),
    lorentz_x0_q32 BIGINT NOT NULL,
    lorentz_x1_q32 BIGINT NOT NULL,
    lorentz_x2_q32 BIGINT NOT NULL,
    lorentz_x3_q32 BIGINT NOT NULL,
    phase_theta_f_q32 BIGINT NOT NULL,
    phase_theta_s_q32 BIGINT NOT NULL,
    spatial_bucket_x INT GENERATED ALWAYS AS (CAST(lorentz_x1_q32 >> 15 AS INT)) STORED,
    spatial_bucket_y INT GENERATED ALWAYS AS (CAST(lorentz_x2_q32 >> 15 AS INT)) STORED,
    spatial_bucket_z INT GENERATED ALWAYS AS (CAST(lorentz_x3_q32 >> 15 AS INT)) STORED,
    salience_reinforcements BIGINT NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_global_atoms_spatial_bucket
    ON global_atoms (spatial_bucket_x, spatial_bucket_y, spatial_bucket_z);

CREATE TABLE IF NOT EXISTS reasoning_arcs (
    arc_id BIGINT PRIMARY KEY,
    chain_id BIGINT NOT NULL,
    step_index INT NOT NULL,
    from_atom BIGINT NOT NULL REFERENCES global_atoms(atom_id),
    to_atom BIGINT NOT NULL REFERENCES global_atoms(atom_id),
    compiled_mlir_op INT NOT NULL,
    cosh_action_q32 BIGINT NOT NULL,
    teacher_certainty_q32 BIGINT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reasoning_arcs_outgoing
    ON reasoning_arcs (from_atom, chain_id);
