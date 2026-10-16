#!/usr/bin/env python3
"""Record Muse Glimmer greedy token steps into a trajectory vault.

Each step stores the last-layer state that chose the next token id (when the
carrier token atom has not yet stored a hidden-state payload in this run).
The weight vaults are not opened. generate_glimmer must not be running:
that process and this one would both resident the same checkpoint.

Resume reads design/glimmer_token_trajectories.progress and continues.
"""

from __future__ import annotations

import array
import struct
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

MODEL = Path("/Users/stewartbarteau/models/Muse-Glimmer-q4km-gs128-MLX")
ROOT = Path("/Users/stewartbarteau/Desktop/monster-hybrid-server")
IMPL = ROOT / "immune_system" / "implementation"
OUT = ROOT / "design" / "glimmer_token_trajectories.atoma_map"
PROGRESS = ROOT / "design" / "glimmer_token_trajectories.progress"
RECORDER = IMPL / "target" / "release" / "record_mlir_trajectory"
READER = IMPL / "target" / "release" / "read_mlir_native"
STEPS_DIR = Path("/tmp/glimmer-traj-steps")
BASE_VOCAB = 202048
ORIGIN = "glimmer-q4km-gs128"
BOS = 200000
EOS = {200001, 200008}
TARGET_ARCS = 100_000
MAX_NEW = 32
BATCH = 32
FRANCE = "The capital of France is"
FRANCE_PREFIX = [200000, 954, 7963, 323, 11698, 373]
FRANCE_GEN = [
    13796, 26, 1573, 373, 290, 2167, 26899, 1212, 6996, 310, 11698, 341, 290,
    2167, 26899, 1212, 6996, 9424, 310, 290, 13471, 16712, 26, 589, 13796,
    19157, 2064, 262, 7001, 323, 220, 738, 24, 27432, 24, 34939, 310, 10167,
    220, 837, 31, 24, 290, 13467, 310, 290, 13471, 16712, 24, 2467,
]

PLACES = [
    "France", "Japan", "Kenya", "Peru", "Egypt", "Canada", "Norway", "India",
    "Brazil", "Italy", "Spain", "Greece", "Mexico", "Chile", "Ghana", "Mali",
    "Nepal", "Thailand", "Vietnam", "Poland", "Sweden", "Finland", "Ireland",
    "Portugal", "Morocco", "Nigeria", "Ethiopia", "Argentina", "Colombia",
    "Cuba", "Iceland", "Austria", "Hungary", "Romania", "Turkey", "Jordan",
    "Lebanon", "Qatar", "Oman", "Mongolia", "Laos", "Cambodia", "Malaysia",
    "Indonesia", "Philippines", "Australia", "Fiji", "Samoa", "Palau", "Bhutan",
    "Bangladesh", "Pakistan", "Iran", "Iraq", "Syria", "Yemen", "Sudan",
    "Tanzania", "Uganda", "Rwanda", "Senegal", "Algeria", "Tunisia", "Libya",
    "Ukraine", "Belarus", "Latvia", "Estonia", "Denmark", "Belgium",
]
ELEMENTS = [
    "hydrogen", "helium", "lithium", "carbon", "nitrogen", "oxygen", "sodium",
    "magnesium", "aluminum", "silicon", "phosphorus", "sulfur", "chlorine",
    "potassium", "calcium", "iron", "copper", "zinc", "silver", "gold",
    "mercury", "lead", "uranium", "neon", "argon", "fluorine", "iodine",
    "tin", "nickel", "cobalt", "chromium", "manganese", "titanium", "boron",
]
ANIMALS = [
    "a wolf", "a salmon", "a honeybee", "an octopus", "a crow", "a tortoise",
    "a bat", "a whale", "a penguin", "a camel", "a jaguar", "a crane",
    "a beetle", "a shark", "a frog", "a falcon", "a bison", "a panda",
    "a gecko", "an eel", "a moth", "a heron", "a lynx", "a seal", "a yak",
    "a squid", "a robin", "a badger", "a moose", "a cobra", "a dolphin",
    "a sparrow", "a termite", "a lobster", "a hare", "a raven", "an ibis",
    "a sturgeon", "a firefly", "a mantis",
]
TOPICS = [
    "gravity", "photosynthesis", "prime numbers", "the water cycle",
    "plate tectonics", "vaccination", "supply and demand", "a hash table",
    "recursion", "public key cryptography", "the Doppler effect", "osmosis",
    "natural selection", "inflation", "a constitution", "habeas corpus",
    "double entry bookkeeping", "the scientific method", "a sonnet",
    "counterpoint", "perspective in drawing", "the silk road", "the printing press",
    "antibiotics", "pasteurization", "a glacier", "a coral reef", "a watershed",
    "compost", "crop rotation", "a ledger", "a mortgage", "an algorithm",
    "a compiler", "a database index", "a mutex", "a checksum", "latency",
    "bandwidth", "an eclipse", "a tide", "a comet", "a volcano", "an aquifer",
    "fermentation", "catalysis", "entropy", "inertia", "refraction",
    "insulation", "a vaccine trial", "informed consent", "peer review",
    "a census", "a tariff", "a cooperative", "a patent", "a trademark",
    "due process", "a jury", "a treaty", "a ceasefire", "a dialect",
    "an alphabet", "a calendar", "a map projection", "a compass", "a kiln",
    "a loom", "irrigation", "a reservoir", "a port", "a canal", "a bridge",
    "concrete", "steel", "glass", "paper", "ink", "a library", "an archive",
    "oral history", "a census form", "a ballot", "a budget", "a receipt",
    "a contract", "a will", "a lease", "interest", "a dividend", "a tax",
    "insurance", "a pension", "a union", "a strike", "a standard", "a protocol",
    "a packet", "a socket", "a filesystem", "a backup", "a checksum error",
    "version control", "a unit test", "a type system", "garbage collection",
    "a stack", "a queue", "a graph", "a proof", "a theorem", "a conjecture",
    "zero", "infinity", "probability", "a sample", "a confidence interval",
    "a placebo", "a control group", "a blind trial", "a case study",
]
TOOLS = [
    "a hammer", "a level", "a chisel", "a plane", "a lathe", "a kiln",
    "a sextant", "a scale", "a pipette", "a microscope", "a telescope",
    "a metronome", "a tuning fork", "a loom", "a spindle", "a press",
    "a compass", "a protractor", "an abacus", "a slide rule", "a wrench",
    "a saw", "a clamp", "a trowel", "a sieve", "a bellows", "a crucible",
    "a retort", "a balance", "a caliper",
]
TASKS = [
    "boil water", "set a bone", "read a map", "tune a violin", "proof a page",
    "graft a tree", "salt a fish", "fire a brick", "shear a sheep",
    "mill grain", "render tallow", "tan a hide", "splice a rope",
    "caulk a seam", "survey a field", "bind a book", "cast a bell",
    "blow glass", "quench steel", "temper a blade", "mix mortar",
    "lay a course of bricks", "prune a vine", "press a cider",
    "smoke a ham", "culture a yogurt", "proof dough", "knead bread",
    "clarify butter", "reduce a sauce",
]
WORKS = [
    "Hamlet", "The Odyssey", "Don Quixote", "Things Fall Apart",
    "One Hundred Years of Solitude", "The Tale of Genji", "Medea",
    "The Republic", "On the Origin of Species", "Principia",
    "The Art of War", "The Pillow Book", "Candide", "Frankenstein",
    "Pride and Prejudice", "Crime and Punishment", "The Stranger",
    "Season of Migration to the North", "Pedro Páramo", "Invisible Cities",
]


def build_prompts() -> list[str]:
    seen = {FRANCE}
    prompts = [FRANCE]

    def add(text: str) -> None:
        if text not in seen:
            seen.add(text)
            prompts.append(text)

    for place in PLACES:
        add(f"The capital of {place} is")
        add(f"The largest city in {place} is")
        add(f"A river in {place} is the")
        add(f"The official language of {place} is")
        add(f"Travelers in {place} often eat")
        add(f"{place} shares a border with")
        add(f"The currency of {place} is")
        add(f"Farmers in {place} grow")
        add(f"Winter in {place} brings")
        add(f"A port city in {place} is")
    for element in ELEMENTS:
        add(f"The chemical symbol for {element} is")
        add(f"The atomic number of {element} is")
        add(f"{element.capitalize()} is used to")
        add(f"At room temperature, {element} is")
    for animal in ANIMALS:
        add(f"{animal.capitalize()} lives")
        add(f"The diet of {animal} is")
        add(f"{animal.capitalize()} raises its young by")
        add(f"A distinguishing feature of {animal} is")
    for topic in TOPICS:
        add(f"In plain language, {topic} is")
        add(f"The history of {topic} begins")
        add(f"One limit of {topic} is")
        add(f"People measure {topic} by")
        add(f"A beginner learning {topic} starts with")
        add(f"Without {topic}, ")
        add(f"A textbook defines {topic} as")
        add(f"An expert uses {topic} when")
        add(f"Students confuse {topic} with")
        add(f"The unit used for {topic} is")
    for year in range(1500, 2026, 5):
        add(f"In {year}, ")
        add(f"By {year}, ")
        add(f"A widely used invention of the {year}s was")
    for tool in TOOLS:
        add(f"{tool.capitalize()} is used to")
        add(f"Before using {tool}, ")
        add(f"The measurement a person reads from {tool} is")
    for task in TASKS:
        add(f"To {task}, ")
        add(f"The first step to {task} is")
        add(f"A common mistake when you {task} is")
    for work in WORKS:
        add(f"The book {work} opens")
        add(f"{work} is about")
        add(f"A central character in {work} is")
    for n in range(2, 400):
        add(f"The square of {n} is")
        add(f"{n} times {n + 1} is")
        add(f"The factors of {n * 6} are")

    # Pass 2: Extended multi-domain reasoning, science, systems engineering, and follow-up corpus
    # (appended strictly after index 3847 so Pass 1 indices remain 100% identical to Glimmer)
    for place in PLACES:
        add(f"A major industry in {place} is")
        add(f"The geography of {place} includes")
        add(f"Historically, {place} is known for")
        add(f"The climate of {place} is characterized by")
    for element in ELEMENTS:
        add(f"In nature, {element} is commonly found in")
        add(f"When {element} reacts with oxygen, it forms")
        add(f"An important industrial application of {element} is")
        add(f"The discovery of {element} is attributed to")
    for animal in ANIMALS:
        add(f"In its ecosystem, {animal} plays the role of")
        add(f"To survive harsh conditions, {animal} adapts by")
        add(f"Predators of {animal} include")
    for topic in TOPICS:
        add(f"The fundamental mechanism behind {topic} is")
        add(f"In systems engineering, {topic} matters because")
        add(f"A real-world example of {topic} is")
        add(f"If {topic} fails, the consequence is")
        add(f"To optimize {topic}, an engineer will")
    for task in TASKS:
        add(f"The tools required to {task} are")
        add(f"You know you have properly managed to {task} when")
    for work in WORKS:
        add(f"A major theme explored in {work} is")
        add(f"The historical influence of {work} comes from")
    return prompts


def rust_running() -> bool:
    out = subprocess.check_output(["ps", "-ax", "-o", "command="], text=True)
    return any(
        "generate_glimmer" in line and "collect_glimmer_trajectories" not in line
        for line in out.splitlines()
    )


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


def pack_continuation(prefix: list[int], generated: list[int], hidden: list[list[float]]) -> bytes:
    dim = len(hidden[0]) if hidden else 0
    if hidden and any(len(row) != dim for row in hidden):
        raise RuntimeError("hidden rows differ in width")
    if hidden and len(hidden) != len(generated):
        raise RuntimeError("hidden states must cover every generated id")
    buf = struct.pack("<III", len(prefix), len(generated), dim)
    buf += struct.pack(f"<{len(prefix)}I", *prefix)
    buf += struct.pack(f"<{len(generated)}I", *generated)
    if dim:
        flat = array.array("f", (value for row in hidden for value in row))
        if sys.byteorder != "little":
            flat.byteswap()
        buf += flat.tobytes()
    return buf


def encode_prefix(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(ids, list):
        ids = list(ids)
    return [BOS] + [int(i) for i in ids]


def _logits_last(model, hidden):
    """Same head as Model.__call__, last position only. Does not touch the cache."""
    import mlx.core as mx

    logits = model.lm_head(hidden).astype(mx.float32)
    cap = model.args.output_soft_cap_temp
    if cap is not None:
        logits = cap * mx.tanh(logits * model.args.output_multiplier / cap)
    else:
        logits = logits * model.args.output_multiplier
    return logits[0, -1]


def teacher_step(model, prefix: list[int], max_new: int, seen_carriers: set[int]):
    import mlx.core as mx

    cache = model.make_cache()
    prompt = mx.array(prefix, dtype=mx.int32)
    offset = 0
    length = int(prompt.shape[0])
    while length - offset > 1:
        take = min(2048, (length - offset) - 1)
        model.model(prompt[offset : offset + take][None], cache=cache)
        mx.eval([c.state for c in cache])
        offset += take
    hidden = model.model(prompt[offset:][None], cache=cache)
    generated: list[int] = []
    states: list[list[float]] = []
    any_novel_carrier = False
    curr_token = prefix[-1]
    for _ in range(max_new):
        last = hidden[0, -1]
        logits = _logits_last(model, hidden)
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
    # If every carrier token in this continuation already had its hidden state recorded in this run,
    # omit the redundant 26.6 KB/step float vectors (`dim = 0` in ATOMA_TJ1).
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


def replay_france(expect: list[int]) -> None:
    cmd = [
        str(READER),
        "--vault",
        str(OUT),
        "--mlir-ids",
        ",".join(str(i) for i in FRANCE_PREFIX),
        "--max-new",
        str(len(expect)),
        "--base-vocab",
        str(BASE_VOCAB),
        "--eos",
        "200001,200008",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    print(proc.stdout, end="")
    if proc.returncode != 0:
        print(proc.stderr, end="", file=sys.stderr)
        raise SystemExit(f"GATE_FAIL reader exit {proc.returncode}")
    produced = None
    for line in proc.stdout.splitlines():
        if line.startswith("next_ids "):
            start = line.find("[")
            end = line.find("]")
            produced = [int(part) for part in line[start + 1 : end].split(",") if part.strip()]
    if produced != expect:
        raise SystemExit(f"GATE_FAIL replay {produced} expected {expect}")
    print(f"GATE_OK france_tokens={len(expect)}", flush=True)


def main() -> None:
    if rust_running():
        raise SystemExit("GATE_FAIL generate_glimmer is running")
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

    from mlx_lm.tokenizer_utils import load as load_tokenizer
    from mlx_lm.utils import load

    tokenizer = load_tokenizer(MODEL)
    probe = encode_prefix(tokenizer, FRANCE)
    if probe != FRANCE_PREFIX:
        raise SystemExit(f"GATE_FAIL prompt ids {probe}")
    print(f"PROMPT_IDS {len(probe)} {probe}", flush=True)

    print("loading glimmer", flush=True)
    model, tokenizer = load(str(MODEL), trust_remote_code=True)
    import mlx.core as mx

    try:
        mx.set_cache_limit(512 * 1024 * 1024)
    except Exception:
        pass

    seen_carriers: set[int] = set()

    if next_index == 0:
        t0 = time.time()
        expect = FRANCE_GEN[: min(MAX_NEW, len(FRANCE_GEN))]
        generated, hidden = teacher_step(model, FRANCE_PREFIX, len(expect), seen_carriers)
        if generated != expect:
            raise SystemExit(f"GATE_FAIL teacher {generated} expected {expect}")
        text = tokenizer.decode(generated)
        print(
            f"prompt 0 sec={time.time() - t0:.1f} n={len(generated)} text={text!r}",
            flush=True,
        )
        arcs = record([(FRANCE_PREFIX, generated, hidden)])
        replay_france(generated)
        next_index = 1
        save_progress(next_index, arcs)
        mx.metal.clear_cache()

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
    import importlib.util
    import os

    mlx_py = Path("/Users/stewartbarteau/mlx_env/bin/python3")
    if importlib.util.find_spec("mlx_lm") is None and mlx_py.exists():
        if Path(sys.executable).resolve() != mlx_py.resolve():
            os.execv(str(mlx_py), [str(mlx_py), *sys.argv])
    main()

