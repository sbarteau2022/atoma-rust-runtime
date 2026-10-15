#!/usr/bin/env python3
"""Record Qwen3.6-35B-A3B-OptiQ-4bit greedy token steps into a trajectory vault.

Each step stores the last-layer state that chose the next token id (when the
carrier token atom has not yet stored a hidden-state payload in this run).
Supports recording into its own vault (`design/qwen_token_trajectories.atoma_map`)
or directly into the shared trajectory mind via `--vault` / `--out`.

Resume reads design/qwen_token_trajectories.progress and continues.
"""

from __future__ import annotations

import array
import struct
import subprocess
import sys
import time
from pathlib import Path

from collect_glimmer_trajectories import build_prompts, pack_continuation

MODEL = Path("/Users/stewartbarteau/models/Qwen3.6-35B-A3B-OptiQ-4bit")
ROOT = Path("/Users/stewartbarteau/Desktop/monster-hybrid-server")
IMPL = ROOT / "immune_system" / "implementation"
OUT = ROOT / "design" / "qwen_token_trajectories.atoma_map"
PROGRESS = ROOT / "design" / "qwen_token_trajectories.progress"
RECORDER = IMPL / "target" / "release" / "record_mlir_trajectory"
READER = IMPL / "target" / "release" / "read_mlir_native"
STEPS_DIR = Path("/tmp/qwen-traj-steps")
BASE_VOCAB = 248320
ORIGIN = "qwen3.6-35b-a3b-optiq4"
EOS = {151643, 151645, 248044, 248046}
TARGET_ARCS = 100_000
MAX_NEW = 32
BATCH = 32


def load_progress() -> tuple[int, int]:
    if not PROGRESS.exists():
        if OUT.exists():
            raise SystemExit(
                "GATE_FAIL trajectory vault exists without a progress file"
            )
        return 0, 0
    vals = {}
    for line in PROGRESS.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep != "=" or key.strip() not in ("next_index", "arcs"):
            continue
        vals[key.strip()] = int(value.strip())
    return vals.get("next_index", 0), vals.get("arcs", 0)


def save_progress(next_index: int, arcs: int) -> None:
    text = f"next_index={next_index}\narcs={arcs}\nbase_vocab={BASE_VOCAB}\norigin={ORIGIN}\n"
    tmp = PROGRESS.with_suffix(".progress.partial")
    tmp.write_text(text)
    tmp.replace(PROGRESS)


def encode_prefix(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=True)
    if not isinstance(ids, list):
        ids = list(ids)
    return [int(i) for i in ids]


def teacher_step(model, prefix: list[int], max_new: int, seen_carriers: set[int]):
    import mlx.core as mx

    cache = model.make_cache()
    prompt = mx.array(prefix, dtype=mx.int32)
    offset = 0
    length = int(prompt.shape[0])
    while length - offset > 1:
        take = min(2048, (length - offset) - 1)
        model.model(prompt[offset : offset + take][None], cache=cache)
        mx.eval([c.state for c in cache if hasattr(c, "state")])
        offset += take
    hidden = model.model(prompt[offset:][None], cache=cache)
    generated: list[int] = []
    states: list[list[float]] = []
    any_novel_carrier = False
    curr_token = prefix[-1]
    for _ in range(max_new):
        last = hidden[0, -1]
        if hasattr(model, "lm_head"):
            logits = model.lm_head(hidden[:, -1:, :]).astype(mx.float32)[0, -1]
        else:
            logits = model.model.embed_tokens.as_linear(hidden[:, -1:, :]).astype(mx.float32)[0, -1]
        state = last.astype(mx.float32)
        mx.eval(state, logits)
        nxt = int(mx.argmax(logits).item())
        generated.append(nxt)
        if curr_token not in seen_carriers:
            any_novel_carrier = True
            seen_carriers.add(curr_token)
        states.append(state.tolist())
        curr_token = nxt
        if nxt in EOS:
            break
        hidden = model.model(mx.array([nxt], dtype=mx.int32)[None], cache=cache)
    if not any_novel_carrier:
        states = []
    return generated, states


def record(records: list[tuple[list[int], list[int], list[list[float]]]]) -> int:
    STEPS_DIR.mkdir(parents=True, exist_ok=True)
    path = STEPS_DIR / f"batch-{time.time_ns()}.bin"
    blob = b"ATOMA_TJ1" + b"".join(
        pack_continuation(prefix, generated, hidden) for prefix, generated, hidden in records
    )
    path.write_bytes(blob)
    cmd = [
        str(RECORDER),
        "--out",
        str(OUT),
        "--steps",
        str(path),
        "--base-vocab",
        str(BASE_VOCAB),
        "--origin",
        ORIGIN,
        "--certainty",
        "1",
    ]
    if OUT.exists():
        cmd.extend(["--vault", str(OUT)])
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    arcs = None
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
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
    if not RECORDER.exists() or not READER.exists():
        raise SystemExit("GATE_FAIL release reader binaries are missing")
    prompts = build_prompts()
    next_index, arcs = load_progress()
    print(
        f"corpus={len(prompts)} resume_index={next_index} arcs={arcs} "
        f"target={TARGET_ARCS} max_new={MAX_NEW} batch={BATCH} base_vocab={BASE_VOCAB}",
        flush=True,
    )
    if arcs >= TARGET_ARCS:
        print(f"DONE arcs={arcs} prompts={next_index} reason=target", flush=True)
        return

    from mlx_lm.utils import load

    print("loading qwen3.6-35b-a3b-optiq4", flush=True)
    model, tokenizer = load(str(MODEL), trust_remote_code=True)
    import mlx.core as mx

    try:
        mx.set_cache_limit(512 * 1024 * 1024)
    except Exception:
        pass

    seen_carriers: set[int] = set()

    while next_index < len(prompts) and arcs < TARGET_ARCS:
        batch_prompts = prompts[next_index : next_index + BATCH]
        records = []
        t0 = time.time()
        for offset, text in enumerate(batch_prompts):
            prefix = encode_prefix(tokenizer, text)
            generated, hidden = teacher_step(model, prefix, MAX_NEW, seen_carriers)
            records.append((prefix, generated, hidden))
            piece = tokenizer.decode(generated).replace("\n", " ")
            print(
                f"prompt {next_index + offset} sec={time.time() - t0:.1f} "
                f"n={len(generated)} text={piece!r}",
                flush=True,
            )
        arcs = record(records)
        next_index += len(records)
        save_progress(next_index, arcs)
        print(f"checkpoint index={next_index} arcs={arcs}", flush=True)
        mx.metal.clear_cache()

    reason = "target" if arcs >= TARGET_ARCS else "prompts_exhausted"
    print(f"DONE arcs={arcs} prompts={next_index} reason={reason}", flush=True)


if __name__ == "__main__":
    main()
