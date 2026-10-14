-- =============================================================================
-- ATOMA GEODESIC MLIR ATOM CATALOG SCHEMA v2 (`design/mlir_atom_catalog.sql`)
-- =============================================================================
-- Purpose:
--   Model-agnostic, tokenizer-free SQLite source-of-truth schema where every
--   MLIR opcode (`MlirOpId`) is anchored at a canonical 5D coordinate well
--   `(lorentz_x0..x3_q32, phase_theta_f_q32, phase_theta_s_q32)` inside the
--   144-sector quotient manifold `(H³ × T²) / Γ₁₂` (`sector_n`, `sector_m`).
--
--   `mlir_token_span_bindings` is retained strictly as an optional edge adapter
--   for legacy text-pretrained checkpoints (`SPEC_atomic_ir.md` Phase A).
-- =============================================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS catalog_meta (
    version             INTEGER PRIMARY KEY CHECK (version >= 1),
    base_vocab_size     INTEGER NOT NULL CHECK (base_vocab_size > 0),
    catalog_sha256_hex  TEXT    NOT NULL CHECK (length(catalog_sha256_hex) == 64),
    created_at_utc      TEXT    NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS mlir_dialects (
    dialect_id          INTEGER PRIMARY KEY CHECK (dialect_id BETWEEN 0 AND 255),
    dialect_slug        TEXT    NOT NULL UNIQUE,
    description         TEXT    NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS mlir_type_sigs (
    type_id             INTEGER PRIMARY KEY CHECK (type_id BETWEEN 0 AND 255),
    type_name           TEXT    NOT NULL UNIQUE,
    byte_width          INTEGER NOT NULL
) STRICT;

-- Primary Geodesic MLIR Opcode Table anchored in (H³ × T²) / Γ₁₂
CREATE TABLE IF NOT EXISTS mlir_opcodes (
    op_id               INTEGER PRIMARY KEY CHECK (op_id >= 0),
    dialect_id          INTEGER NOT NULL REFERENCES mlir_dialects(dialect_id),
    mnemonic            TEXT    NOT NULL UNIQUE,
    sector_n            INTEGER NOT NULL CHECK (sector_n BETWEEN 0 AND 11),
    sector_m            INTEGER NOT NULL CHECK (sector_m BETWEEN 0 AND 11),
    lorentz_x0_q32      INTEGER NOT NULL CHECK (lorentz_x0_q32 >= 2147483648),
    lorentz_x1_q32      INTEGER NOT NULL,
    lorentz_x2_q32      INTEGER NOT NULL,
    lorentz_x3_q32      INTEGER NOT NULL,
    phase_theta_f_q32   INTEGER NOT NULL,
    phase_theta_s_q32   INTEGER NOT NULL,
    arity               INTEGER NOT NULL CHECK (arity BETWEEN 0 AND 4),
    arg0_type_id        INTEGER REFERENCES mlir_type_sigs(type_id),
    arg1_type_id        INTEGER REFERENCES mlir_type_sigs(type_id),
    arg2_type_id        INTEGER REFERENCES mlir_type_sigs(type_id),
    arg3_type_id        INTEGER REFERENCES mlir_type_sigs(type_id),
    ret_type_id         INTEGER NOT NULL REFERENCES mlir_type_sigs(type_id),
    required_role_level INTEGER CHECK (required_role_level IS NULL OR required_role_level BETWEEN 0 AND 12)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_mlir_opcodes_manifold_sector
    ON mlir_opcodes(sector_n, sector_m);

-- Optional legacy text-checkpoint token_id span adapter (`SPEC_atomic_ir.md` Phase A shim)
CREATE TABLE IF NOT EXISTS mlir_token_span_bindings (
    op_id               INTEGER NOT NULL REFERENCES mlir_opcodes(op_id) ON DELETE CASCADE,
    span_order          INTEGER NOT NULL CHECK (span_order >= 0),
    token_id            INTEGER NOT NULL CHECK (token_id >= 0),
    PRIMARY KEY (op_id, span_order)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_token_span_head
    ON mlir_token_span_bindings(token_id, span_order);

-- Seed canonical dialects
INSERT OR IGNORE INTO mlir_dialects (dialect_id, dialect_slug, description) VALUES
    (0, 'token',          'Identity single-token carrier (legacy edge shim)'),
    (1, 'builtin',        'Core MLIR module, function call, and return operations'),
    (2, 'arith',          'Deterministic integer and Q32.31 fixed-point arithmetic'),
    (3, 'linalg',         'Zero-copy quantized GEMV and inner-product operations'),
    (4, 'affine',         'Structured loop and conditional control flow'),
    (5, 'atoma.manifold', 'Native (H3 x T2)/Gamma hyperbolic and toroidal operations'),
    (6, 'atoma.tool',     'Capability-gated binary tool invocation (replaces JSON/Python)');

-- Seed canonical static SSA types
INSERT OR IGNORE INTO mlir_type_sigs (type_id, type_name, byte_width) VALUES
    (0, 'Void',              0),
    (1, 'I64',               8),
    (2, 'Q32_31',            8),
    (3, 'F32Vec',           16),
    (4, 'LorentzCoord',     32),
    (5, 'PhaseCoord',       16),
    (6, 'SiloAtomHandle',    8),
    (7, 'CapabilityReceipt',32);

-- Seed canonical Geodesic MLIR opcodes anchored on (H³ × T²)/Γ₁₂ (base_vocab_size = 248320)
INSERT OR IGNORE INTO mlir_opcodes (
    op_id, dialect_id, mnemonic,
    sector_n, sector_m,
    lorentz_x0_q32, lorentz_x1_q32, lorentz_x2_q32, lorentz_x3_q32,
    phase_theta_f_q32, phase_theta_s_q32,
    arity, arg0_type_id, arg1_type_id, arg2_type_id, arg3_type_id, ret_type_id, required_role_level
) VALUES
    (248320, 2, 'arith.constant_q32',           0,  0, 2380998693,  995736342,  226303714,  226303714, -6012954214, -6012954214, 0, NULL, NULL, NULL, NULL, 2, NULL),
    (248321, 2, 'arith.add_q32',                1,  2, 2544425620,  234625864, 1313904840,  375401383, -4724464026, -3435973837, 2,    2,    2, NULL, NULL, 2, NULL),
    (248322, 2, 'arith.mul_q32',                3,  5, 2539604340, -1124565239,  655996389,  421711964, -3221225472, -1073741824, 2,    2,    2, NULL, NULL, 2, NULL),
    (248323, 5, 'atoma.manifold.embed_lorentz', 4,  6, 2678483804,  482475617, -1254436604,  868456110, -1717986918,   858993459, 3,    2,    2,    2, NULL, 4, NULL),
    (248324, 5, 'atoma.manifold.silo_query',    6,  8, 2750384621,  884736192,  884736192, -1228800267,   214748365,  2362232013, 1,    4, NULL, NULL, NULL, 6, NULL),
    (248325, 5, 'atoma.manifold.cosh_dist',     7,  9, 2655312980, -911245821, -1007166434,  767364902,  1825361101,  3758096384, 2,    4,    4, NULL, NULL, 2, NULL),
    (248326, 6, 'atoma.tool.sensory_ingest',    8, 10, 2856104212,          0,  999636474, 1599418358,  3113851289,  4724464026, 2,    6,    2, NULL, NULL, 7,    1),
    (248327, 6, 'atoma.tool.motor_actuate',    10,  1, 2856104212,          0, -999636474,-1599418358,  4509715661, -4509715661, 2,    6,    2, NULL, NULL, 7,    2),
    (248328, 1, 'func.return_q32',             11, 11, 2765110924, -394910218,  394910218, 1678368426,  5905580032,  5905580032, 1,    2, NULL, NULL, NULL, 2, NULL);

INSERT OR IGNORE INTO mlir_token_span_bindings (op_id, span_order, token_id) VALUES
    (248320, 0, 101), (248320, 1, 102),
    (248321, 0, 103), (248321, 1, 104), (248321, 2, 105),
    (248322, 0, 106), (248322, 1, 107), (248322, 2, 108),
    (248323, 0, 201), (248323, 1, 202), (248323, 2, 203), (248323, 3, 204),
    (248324, 0, 205), (248324, 1, 206), (248324, 2, 207), (248324, 3, 208),
    (248325, 0, 209), (248325, 1, 210), (248325, 2, 211),
    (248326, 0, 301), (248326, 1, 302), (248326, 2, 303), (248326, 3, 304), (248326, 4, 305),
    (248327, 0, 306), (248327, 1, 307), (248327, 2, 308), (248327, 3, 309), (248327, 4, 310),
    (248328, 0, 401), (248328, 1, 402);
