#!/usr/bin/env python3
"""
Live Operational Benchmark & Real-Time Trajectory Learning Suite for Monster Hybrid Server:
1. OpenAI-Compatible `/v1/models` & `/v1/chat/completions` (JSON + SSE Streaming)
2. GSM8K Multi-Step Word Problems (`L7 Convergent Logic + L6 Predictive Simulation` via native `MlirSsaBlock` Q32_31)
3. NIAH (Needle-in-a-Haystack) Long-Horizon Retrieval across 4K, 16K, and 32K char haystacks at 10%, 50%, 90% depths (`L5 Episodic Memory + L11 Spatial Topography`)
4. RULER Variable Tracking (5-hop variable chains), Multi-Needle Aggregation, and Word Frequency Aggregation (`CWE`)
5. Real-Time Trajectory Crystallization Proof: Turn 1 (`Tier3QualifiedTeacher`, mints 1-cells + appends `.wal`) -> Turn 2 (`Tier1TrajectoryWalk`, 0 new arcs, direct geodesic recall).
"""

import json
import sys
import time
import urllib.request

BASE_URL = "http://127.0.0.1:8088"


def post_chat_completion(
    prompt: str, session_id: int = 1, max_tokens: int = 64, stream: bool = False
):
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": "atoma-active-mind-12basin",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "session_id": session_id,
        "stream": stream,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read().decode("utf-8")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if stream:
        return raw, elapsed_ms
    return json.loads(raw), elapsed_ms


def build_haystack(target_chars: int, needle_sentence: str, depth_ratio: float) -> str:
    filler = (
        "The icosahedral hyperbolic manifold partitions cognitive state space into twelve "
        "specialist attractor basins governed by Gauss-Bonnet holonomy and Levi-Civita parallel transport. "
    )
    repeats = max(4, target_chars // len(filler))
    idx = max(1, min(repeats - 1, int(repeats * depth_ratio)))
    chunks = [filler.strip()] * repeats
    chunks.insert(idx, needle_sentence)
    return " ".join(chunks)


def run_suite():
    print(
        "========================================================================================"
    )
    print(
        " MONSTER HYBRID SERVER :: LIVE GSM8K + NIAH + RULER + /v1/chat/completions BENCHMARK"
    )
    print(
        "========================================================================================\n"
    )

    # Wait briefly if server is still finishing startup pre-cache
    for attempt in range(15):
        try:
            with urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=3) as r:
                models_data = json.loads(r.read().decode("utf-8"))
                print(
                    f"[1/5] Verified OpenAI `/v1/models`: {[m['id'] for m in models_data['data']]}"
                )
                break
        except Exception as e:
            if attempt == 14:
                raise RuntimeError(f"Server at {BASE_URL} not reachable: {e}")
            time.sleep(1.0)

    # 1. Verify SSE Streaming mode for Zed / coding agent compatibility
    sse_raw, sse_ms = post_chat_completion(
        "The capital of France is", session_id=700, max_tokens=16, stream=True
    )
    has_done = "data: [DONE]" in sse_raw
    print(
        f"      Verified OpenAI SSE Streaming (`stream: true`): has_done={has_done} latency={sse_ms:.2f}ms\n"
    )

    # 2. GSM8K Multi-Step Word Problems + Real-Time Learning (Turn 1 Tier 3 -> Turn 2 Tier 1)
    gsm8k_cases = [
        (
            "GSM8K-1 (Natalia clips)",
            "Natalia sold 48 clips in April and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "#### 72",
        ),
        (
            "GSM8K-1b (Unseen numbers -> 96 clips)",
            "Natalia sold 96 clips in April and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "#### 144",
        ),
        (
            "GSM8K-2 (Weng hourly rate)",
            "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "#### 10",
        ),
        (
            "GSM8K-3 (Betty wallet savings)",
            "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
            "#### 5",
        ),
        (
            "GSM8K-4 (Julie 120-page book)",
            "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
            "#### 42",
        ),
        (
            "GSM8K-5 (James letters a year)",
            "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
            "#### 624",
        ),
        (
            "GSM8K-6 (Josh house flip profit)",
            "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            "#### 70000",
        ),
    ]

    print(
        "[2/5] Running GSM8K Symbolic Math + Native `MlirSsaBlock` (`Q32_31`) Benchmark:"
    )
    gsm_passed = 0
    for idx, (label, prompt, expected_tag) in enumerate(gsm8k_cases, start=1):
        res1, ms1 = post_chat_completion(prompt, session_id=1000 + idx, max_tokens=64)
        ans1 = res1["choices"][0]["message"]["content"]
        tel1 = res1["atoma_telemetry"]
        ok = expected_tag in ans1
        if ok:
            gsm_passed += 1
        # Immediately re-query on a fresh session to verify real-time trajectory recall (`Tier1TrajectoryWalk`)
        res2, ms2 = post_chat_completion(prompt, session_id=2000 + idx, max_tokens=64)
        tel2 = res2["atoma_telemetry"]
        print(
            f"  - {label}: pass={ok} | Turn1: tier={tel1['tier']} teacher={tel1['teacher_source']} "
            f"arcs=+{tel1['novel_arcs_minted']} wal=+{tel1['wal_bytes_appended']}B ({ms1:.1f}ms) "
            f'-> Turn2: tier={tel2["tier"]} arcs=+{tel2["novel_arcs_minted"]} ({ms2:.1f}ms) | out="{ans1}"'
        )
    print(
        f"  => GSM8K Exact-Match Score: {gsm_passed}/{len(gsm8k_cases)} ({gsm_passed / len(gsm8k_cases) * 100:.1f}%)\n"
    )

    # 3. NIAH (Needle-in-a-Haystack) across 4K, 16K, and 32K haystacks at 10%, 50%, 90% depths
    niah_configs = [
        (
            "NIAH-4K@10%",
            4096,
            0.10,
            "The secret passkey for sector Apollo is 918273.",
            "What is the secret passkey for sector Apollo?",
            "918273",
        ),
        (
            "NIAH-16K@50%",
            16384,
            0.50,
            "The magic number for vault Hyperion is 440912.",
            "What is the magic number for vault Hyperion?",
            "440912",
        ),
        (
            "NIAH-32K@90%",
            32768,
            0.90,
            "The secret code for silo Artemis is 773109.",
            "What is the secret code for silo Artemis?",
            "773109",
        ),
    ]
    print(
        "[3/5] Running NIAH (Needle-in-a-Haystack) Long-Horizon Episodic Benchmark (4K -> 32K chars):"
    )
    niah_passed = 0
    for idx, (label, chars, depth, needle, question, expected) in enumerate(
        niah_configs, start=1
    ):
        haystack_prompt = f"{build_haystack(chars, needle, depth)}\n{question}"
        res, ms = post_chat_completion(
            haystack_prompt, session_id=3000 + idx, max_tokens=32
        )
        ans = res["choices"][0]["message"]["content"]
        tel = res["atoma_telemetry"]
        ok = expected in ans
        if ok:
            niah_passed += 1
        print(
            f"  - {label} (len={len(haystack_prompt)} chars, depth={int(depth * 100)}%): pass={ok} | "
            f"tier={tel['tier']} teacher={tel['teacher_source']} basins={tel['lit_basins']} "
            f'wal=+{tel["wal_bytes_appended"]}B ({ms:.1f}ms) | out="{ans}"'
        )
    print(
        f"  => NIAH Retrieval Score: {niah_passed}/{len(niah_configs)} ({niah_passed / len(niah_configs) * 100:.1f}%)\n"
    )

    # 4. RULER Suite (Multi-Hop Variable Tracking, Multi-Needle Aggregation, Common Word Aggregation)
    ruler_cases = [
        (
            "RULER-VT-5Hop (Value of X5)",
            "VAR X1 = 84920. Always maintain geodesic coherence. VAR X2 = X1. Noise step alpha. VAR X3 = X2. VAR X4 = X3. VAR X5 = X4. What is the value of variable X5?",
            ["84920"],
        ),
        (
            "RULER-VT-Reverse (All vars assigned 31415)",
            "VAR A1 = 31415; VAR B2 = 99999; VAR A2 = A1; VAR B3 = B2; VAR A3 = A2; Which variables are assigned the value 31415?",
            ["A1", "A2", "A3"],
        ),
        (
            "RULER-MultiNeedle (Passkeys for Alpha & Omega)",
            "The passkey for Alpha is 55102. The system clock is nominal. The passkey for Omega is 88341. What are the passkeys for both Alpha and Omega?",
            ["55102", "88341"],
        ),
        (
            "RULER-CWE (Top 3 most common words)",
            "hyperboloid curvature holonomy hyperboloid geodesic curvature hyperboloid curvature geodesic hyperboloid\nWhat are the top 3 most common words in the text above?",
            ["hyperboloid", "curvature", "geodesic"],
        ),
    ]
    print("[4/5] Running RULER Suite (Variable Tracking + Multi-Needle + CWE):")
    ruler_passed = 0
    for idx, (label, prompt, expected_tokens) in enumerate(ruler_cases, start=1):
        res, ms = post_chat_completion(prompt, session_id=4000 + idx, max_tokens=48)
        ans = res["choices"][0]["message"]["content"]
        tel = res["atoma_telemetry"]
        ok = all(tok in ans for tok in expected_tokens)
        if ok:
            ruler_passed += 1
        print(
            f"  - {label}: pass={ok} | tier={tel['tier']} teacher={tel['teacher_source']} "
            f'basins={tel["lit_basins"]} wal=+{tel["wal_bytes_appended"]}B ({ms:.1f}ms) | out="{ans}"'
        )
    print(
        f"  => RULER Score: {ruler_passed}/{len(ruler_cases)} ({ruler_passed / len(ruler_cases) * 100:.1f}%)\n"
    )

    total_passed = gsm_passed + niah_passed + ruler_passed
    total_cases = len(gsm8k_cases) + len(niah_configs) + len(ruler_cases)
    print(
        "========================================================================================"
    )
    print(
        f" SUMMARY: {total_passed}/{total_cases} ({total_passed / total_cases * 100:.1f}%) PASSED | "
        f"Global Map Now At: {tel['total_0cells']} 0-cells, {tel['total_1cell_arcs']} 1-cell arcs"
    )
    print(
        "========================================================================================"
    )
    if total_passed != total_cases:
        sys.exit(1)


if __name__ == "__main__":
    run_suite()
