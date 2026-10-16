#!/Users/stewartbarteau/mlx_env/bin/python3
"""Live Multi-Model Trajectory Harvester for Qwen3.6-35B-A3B, Muse-Glimmer, or any Open-Weights Model.

Runs the real teacher model forward pass on Apple Silicon Metal (via mlx_lm),
extracts the last-layer hidden states that selected each token, transduces the
prompt and generated continuation into the canonical Atoma 202,048 ML-IR carrier
alphabet (so Qwen, Glimmer, and any downloaded open-source model share ONE
unified coordinate carrier space), and:
  1. Saves the trajectory vault (`design/qwen_token_trajectories.atoma_map`), AND
  2. Streams binary `ATOMA_WL` delta frames into `design/global_active_mind.atoma_map.wal`
     so a running `monster_hybrid_server` automatically tails and ingests new
     trajectories LIVE in < 1 ms while you work on other projects.

Usage:
  # Default: harvest Qwen3.6-35B-A3B to match Glimmer's 100,045+ arcs & stream live to global map
  python3 immune_system/implementation/scripts/collect_qwen_trajectories.py

  # Harvest any other local or HuggingFace/MLX open-source model live into the global map
  python3 immune_system/implementation/scripts/collect_qwen_trajectories.py \
      --model /path/to/or/hf-repo-of-model \
      --origin my-open-source-teacher \
      --target-arcs 200000
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure mlx_env virtual environment packages are accessible regardless of which python3 invoked this
for _p in [
    "/Users/stewartbarteau/mlx_env/src/mlx-lm-mtp",
    "/Users/stewartbarteau/mlx_env/lib/python3.14/site-packages",
    "/Users/stewartbarteau/mlx_env/lib/python3.12/site-packages",
    "/Users/stewartbarteau/mlx_env/lib/python3.11/site-packages",
]:
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

from collect_glimmer_trajectories import build_prompts, pack_continuation

DEFAULT_QWEN_MODEL = Path("/Users/stewartbarteau/models/Qwen3.6-35B-A3B-OptiQ-4bit")
CANONICAL_LEXICON_MODEL = Path("/Users/stewartbarteau/models/Muse-Glimmer-q4km-gs128-MLX")
ROOT = Path("/Users/stewartbarteau/Desktop/monster-hybrid-server")
IMPL = ROOT / "immune_system" / "implementation"
DEFAULT_OUT = ROOT / "design" / "qwen_token_trajectories.atoma_map"
DEFAULT_PROGRESS = ROOT / "design" / "qwen_token_trajectories.progress"
GLOBAL_WAL = ROOT / "design" / "global_active_mind.atoma_map.wal"
RECORDER = IMPL / "target" / "release" / "record_mlir_trajectory"
READER = IMPL / "target" / "release" / "read_mlir_native"
STEPS_DIR = Path("/tmp/qwen-traj-steps")

# Canonical Atoma boundary carrier space (shared across Glimmer, Qwen, and all teachers)
CANONICAL_BASE_VOCAB = 202048
CANONICAL_BOS = 200000
PROGRESS_MODE = "live_mlx_canonical_v2"

# Model-native EOS token IDs across Qwen / Glimmer / Llama / Mistral families
DEFAULT_TEACHER_EOS = {151643, 151645, 248044, 248046, 200001, 200008, 2, 128001, 128009}


def load_progress(progress_path: Path, out_path: Path, force_reset: bool) -> tuple[int, int]:
    if force_reset:
        if out_path.exists():
            bak = out_path.with_suffix(".atoma_map.bak")
            out_path.replace(bak)
            print(f"[reset] Backed up previous {out_path.name} -> {bak.name}", flush=True)
        progress_path.unlink(missing_ok=True)
        return 0, 0

    if not progress_path.exists():
        return 0, 0

    vals: dict[str, str] = {}
    for line in progress_path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep == "=":
            vals[key.strip()] = value.strip()

    # If the progress file came from the earlier synthetic Rust affine bootstrap (which lacked
    # mode=live_mlx_canonical_v2), automatically archive it and start real MLX harvesting at 0.
    if vals.get("mode") != PROGRESS_MODE:
        print(
            f"[init] Detected legacy/synthetic progress file (mode={vals.get('mode')!r}); "
            f"archiving synthetic vault and starting real teacher harvesting at index=0.",
            flush=True,
        )
        if out_path.exists():
            bak = out_path.with_suffix(".synthetic.bak")
            out_path.replace(bak)
        progress_path.unlink(missing_ok=True)
        return 0, 0

    return int(vals.get("next_index", "0")), int(vals.get("arcs", "0"))


def save_progress(
    progress_path: Path,
    next_index: int,
    arcs: int,
    base_vocab: int,
    origin: str,
) -> None:
    text = (
        f"next_index={next_index}\n"
        f"arcs={arcs}\n"
        f"base_vocab={base_vocab}\n"
        f"origin={origin}\n"
        f"mode={PROGRESS_MODE}\n"
    )
    tmp = progress_path.with_suffix(".progress.partial")
    tmp.write_text(text)
    tmp.replace(progress_path)


def encode_teacher_prefix(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=True)
    if not isinstance(ids, list):
        ids = list(ids)
    return [int(i) for i in ids]


def encode_canonical_prefix(canonical_tokenizer, text: str) -> list[int]:
    ids = canonical_tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(ids, list):
        ids = list(ids)
    return [CANONICAL_BOS] + [int(i) for i in ids if 0 <= int(i) < CANONICAL_BASE_VOCAB]


def encode_canonical_continuation(canonical_tokenizer, text: str) -> list[int]:
    ids = canonical_tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(ids, list):
        ids = list(ids)
    return [int(i) for i in ids if 0 <= int(i) < CANONICAL_BASE_VOCAB]


def _compute_step_logits(model, hidden):
    import mlx.core as mx

    head = getattr(model, "lm_head", None) or getattr(getattr(model, "language_model", None), "lm_head", None)
    if head is not None:
        logits = head(hidden[:, -1:, :]).astype(mx.float32)
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        logits = model.model.embed_tokens.as_linear(hidden[:, -1:, :]).astype(mx.float32)
    else:
        logits = model.language_model.model.embed_tokens.as_linear(hidden[:, -1:, :]).astype(mx.float32)

    args = getattr(model, "args", None) or getattr(getattr(model, "language_model", None), "args", None)
    if args is not None and getattr(args, "output_multiplier", None) is not None:
        cap = getattr(args, "output_soft_cap_temp", None)
        if cap is not None:
            logits = cap * mx.tanh(logits * args.output_multiplier / cap)
        else:
            logits = logits * args.output_multiplier
    return logits[0, -1]


def teacher_step(
    model,
    teacher_tokenizer,
    canonical_tokenizer,
    prompt_text: str,
    max_new: int,
    seen_carriers: set[int],
    eos_ids: set[int],
) -> tuple[list[int], list[int], list[list[float]], str]:
    import mlx.core as mx

    t_prefix = encode_teacher_prefix(teacher_tokenizer, prompt_text)
    cache = model.make_cache()
    prompt = mx.array(t_prefix, dtype=mx.int32)
    offset = 0
    length = int(prompt.shape[0])
    while length - offset > 1:
        take = min(2048, (length - offset) - 1)
        model.model(prompt[offset : offset + take][None], cache=cache)
        mx.eval([c.state for c in cache if hasattr(c, "state")])
        offset += take
    hidden = model.model(prompt[offset:][None], cache=cache)

    teacher_generated: list[int] = []
    raw_states: list[list[float]] = []

    for _ in range(max_new):
        last = hidden[0, -1]
        logits = _compute_step_logits(model, hidden)
        state = last.astype(mx.float32)
        mx.eval(state, logits)
        nxt = int(mx.argmax(logits).item())
        if nxt in eos_ids:
            break
        teacher_generated.append(nxt)
        raw_states.append(state.tolist())
        hidden = model.model(mx.array([nxt], dtype=mx.int32)[None], cache=cache)

    generated_text = teacher_tokenizer.decode(teacher_generated)
    if canonical_tokenizer is not None:
        c_prefix = encode_canonical_prefix(canonical_tokenizer, prompt_text)
        c_generated = encode_canonical_continuation(canonical_tokenizer, generated_text)
    else:
        c_prefix = t_prefix
        c_generated = teacher_generated

    if not c_generated:
        return c_prefix, [], [], generated_text

    # Align teacher hidden states with canonical carrier steps and compact to 256d receptor sketch
    aligned_states: list[list[float]] = []
    any_novel_carrier = False
    curr_token = c_prefix[-1]
    for idx, c_tok in enumerate(c_generated):
        if curr_token not in seen_carriers:
            any_novel_carrier = True
            seen_carriers.add(curr_token)
        if raw_states:
            src_idx = min(idx * len(raw_states) // max(1, len(c_generated)), len(raw_states) - 1)
            full_h = raw_states[src_idx]
            aligned_states.append(full_h[:256] if len(full_h) > 256 else full_h)
        curr_token = c_tok

    if not any_novel_carrier:
        aligned_states = []

    return c_prefix, c_generated, aligned_states, generated_text


def record_batch(
    records: list[tuple[list[int], list[int], list[list[float]]]],
    out_path: Path,
    base_vocab: int,
    origin: str,
    live_wal_path: Path | None,
) -> int:
    STEPS_DIR.mkdir(parents=True, exist_ok=True)
    path = STEPS_DIR / f"batch-{time.time_ns()}.bin"
    blob = b"ATOMA_TJ1" + b"".join(
        pack_continuation(prefix, generated, hidden)
        for prefix, generated, hidden in records
        if prefix and generated
    )
    path.write_bytes(blob)
    cmd = [
        str(RECORDER),
        "--out",
        str(out_path),
        "--steps",
        str(path),
        "--base-vocab",
        str(base_vocab),
        "--origin",
        origin,
        "--certainty",
        "1",
    ]
    if out_path.exists():
        cmd.extend(["--vault", str(out_path)])
    if live_wal_path is not None:
        cmd.extend(["--append-wal", str(live_wal_path)])

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    arcs = None
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith("saved ") or line.startswith("steps_written "):
            print(line, end="", flush=True)
        if line.startswith("saved "):
            for part in line.split():
                if part.startswith("arcs="):
                    arcs = int(part.split("=", 1)[1])
    code = proc.wait()
    path.unlink(missing_ok=True)
    if code != 0:
        raise SystemExit(f"GATE_FAIL recorder exit {code}")
    if arcs is None:
        raise SystemExit("GATE_FAIL recorder did not report arcs")
    return arcs


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Qwen / Open-Weights Trajectory Harvester")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_QWEN_MODEL),
        help="Path or HF repo for teacher model (default: Qwen3.6-35B-A3B-OptiQ-4bit)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output .atoma_map trajectory vault path",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=DEFAULT_PROGRESS,
        help="Progress checkpoint file path",
    )
    parser.add_argument(
        "--origin",
        type=str,
        default="qwen3.6-35b-a3b-optiq4",
        help="Origin model provenance tag",
    )
    parser.add_argument(
        "--target-arcs",
        type=int,
        default=100_045,
        help="Target 1-cell trajectory arcs to harvest (default: 100,045 matching Glimmer)",
    )
    parser.add_argument("--max-new", type=int, default=32, help="Max tokens generated per prompt")
    parser.add_argument("--batch", type=int, default=16, help="Prompts per vault checkpoint batch")
    parser.add_argument(
        "--no-live-wal",
        action="store_true",
        help="Disable streaming ATOMA_WL frames into design/global_active_mind.atoma_map.wal",
    )
    parser.add_argument(
        "--native-vocab",
        action="store_true",
        help="Use teacher's raw tokenizer IDs instead of canonical 202,048 Atoma boundary lexicon",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Start harvesting cleanly from prompt index 0",
    )
    args = parser.parse_args()

    if not RECORDER.exists() or not READER.exists():
        raise SystemExit(
            f"GATE_FAIL release binaries missing at {RECORDER}; run: "
            "cargo build --manifest-path immune_system/implementation/Cargo.toml --bins --release"
        )

    prompts = build_prompts()
    next_index, arcs = load_progress(args.progress, args.out, args.reset)
    base_vocab = 248320 if args.native_vocab else CANONICAL_BASE_VOCAB
    live_wal = None if args.no_live_wal else GLOBAL_WAL

    print(
        f"==========================================================================\n"
        f" ATOMA LIVE TEACHER TRAJECTORY HARVESTER\n"
        f" Teacher Model : {args.model}\n"
        f" Output Vault  : {args.out}\n"
        f" Live Server WAL: {live_wal if live_wal else 'disabled'}\n"
        f" Corpus Prompts: {len(prompts)} (resume_index={next_index}, current_arcs={arcs})\n"
        f" Target Arcs   : {args.target_arcs} (max_new={args.max_new}, batch={args.batch}, base_vocab={base_vocab})\n"
        f"==========================================================================",
        flush=True,
    )
    if arcs >= args.target_arcs:
        print(f"DONE arcs={arcs} prompts={next_index} reason=target", flush=True)
        return

    from mlx_lm.tokenizer_utils import load as load_tokenizer
    from mlx_lm.utils import load, load_model

    canonical_tokenizer = None
    if not args.native_vocab and CANONICAL_LEXICON_MODEL.exists():
        canonical_tokenizer = load_tokenizer(CANONICAL_LEXICON_MODEL)
        print(
            f"[verified] Loaded canonical Atoma boundary lexicon from {CANONICAL_LEXICON_MODEL.name} "
            f"(base_vocab={CANONICAL_BASE_VOCAB}, BOS={CANONICAL_BOS})",
            flush=True,
        )

    print(f"[loading] Loading teacher model {args.model} on Metal...", flush=True)
    try:
        model, teacher_tokenizer = load(str(args.model), trust_remote_code=True)
    except Exception as e:
        print(f"[note] Standard load requested strict weights ({e}); retrying with strict=False for MTP sidecar...", flush=True)
        model_path = Path(args.model)
        model, _ = load_model(model_path, lazy=False, strict=False, trust_remote_code=True)
        teacher_tokenizer = load_tokenizer(model_path)
    import mlx.core as mx

    try:
        mx.set_cache_limit(512 * 1024 * 1024)
    except Exception:
        pass

    eos_ids = set(DEFAULT_TEACHER_EOS)
    if hasattr(teacher_tokenizer, "eos_token_ids"):
        eos_ids.update(int(x) for x in teacher_tokenizer.eos_token_ids)

    seen_carriers: set[int] = set()

    while next_index < len(prompts) and arcs < args.target_arcs:
        batch_prompts = prompts[next_index : next_index + args.batch]
        records = []
        t0 = time.time()
        for offset, text in enumerate(batch_prompts):
            t_step = time.time()
            c_prefix, c_generated, hidden, piece = teacher_step(
                model,
                teacher_tokenizer,
                canonical_tokenizer,
                text,
                args.max_new,
                seen_carriers,
                eos_ids,
            )
            if c_generated:
                records.append((c_prefix, c_generated, hidden))
            clean_piece = piece.replace("\n", " ")
            print(
                f"prompt {next_index + offset} sec={time.time() - t_step:.1f} "
                f"n={len(c_generated)} prompt={text!r} -> {clean_piece!r}",
                flush=True,
            )
        if records:
            arcs = record_batch(records, args.out, base_vocab, args.origin, live_wal)
        next_index += len(batch_prompts)
        save_progress(args.progress, next_index, arcs, base_vocab, args.origin)
        print(
            f"[checkpoint] index={next_index}/{len(prompts)} arcs={arcs}/{args.target_arcs} "
            f"batch_sec={time.time() - t0:.1f}s live_wal={'synced' if live_wal else 'off'}",
            flush=True,
        )
        try:
            mx.clear_cache()
        except Exception:
            try:
                mx.metal.clear_cache()
            except Exception:
                pass

    reason = "target" if arcs >= args.target_arcs else "prompts_exhausted"
    print(f"DONE arcs={arcs} prompts={next_index} reason={reason}", flush=True)


if __name__ == "__main__":
    main()
