# Runbook Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the handwritten runbook's meta-orchestrator into Elle's swarm agent loop — measured κ / drift / region / phase-space / token spend drive continue · nudge · fail-closed stop, verified results are frozen in the breathing window, and an eval A/B measures it against the current loop.

**Architecture:** A pure decision module (`runbook-controller.cjs`) and a pure spend predictor (`spend-predictor.cjs`) are driven by the atlas state the swarm worker already computes every turn. `runLoop` gains one optional hook (`onControl`) that can append a nudge to the agent's next observation or end the run. `atlas-state.cjs` gains message freezing that the breathing window and compaction respect. `swarm-worker.cjs` wires it all behind `ELLE_RUNBOOK_CONTROL=1`; `swarm-eval.cjs` gains a `--runbook off,on` arm.

**Tech Stack:** Node.js CommonJS (`.cjs`, `'use strict'`), built-in `node:test` + `node:assert`, no new dependencies. Model served by `tools/mlx-serve` (Qwen3.8-27B-4bit, MLX).

**Spec:** `docs/superpowers/specs/2026-09-14-runbook-controller-design.md`

## Global Constraints

- All paths below are relative to the Elle repo root: `~/Desktop/ELLE_STACK__CURRENT_WORK/Elle`.
- Controller default is OFF: behavior changes only when `ELLE_RUNBOOK_CONTROL=1`.
- Hard step ceiling with controller on: `ELLE_RUNBOOK_CEILING`, default `40` (runLoop's `MAX_MAX_STEPS`).
- Fallback predicted envelope: `ELLE_RUNBOOK_TOKENS`, default `60000` tokens.
- Do not modify `tools/mlx-serve/strata_kv.py` or `README.md` (they carry uncommitted owner changes). Stage only the files each task names.
- Tests: `node --test <file>`; full electron suite `npm run test:electron`.
- Commit messages end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- No result numbers may be written into docs unless they come from a file produced by a run in Task 7.

---

## File Structure

| File | Responsibility |
|---|---|
| Create `electron/native/providers/runbook-controller.cjs` | Pure per-step decision: continue / nudge / stop, with reasons and nudge text |
| Create `electron/native/providers/runbook-controller.test.cjs` | Unit tests for the decision rules |
| Create `electron/native/providers/spend-predictor.cjs` | Pure token-envelope prediction from prior `RUNBOOK.json` records + loader |
| Create `electron/native/providers/spend-predictor.test.cjs` | Unit tests |
| Modify `electron/native/providers/local-react-agent.cjs` (`runLoop`) | `onControl` hook |
| Modify `electron/native/providers/local-react-agent.test.cjs` | Hook tests |
| Modify `electron/native/providers/atlas-state.cjs` | `freezeMeta`; `compactConversation` keeps pinned messages |
| Modify `electron/native/providers/atlas-state.test.cjs` | Freezing tests |
| Create `electron/native/providers/runbook-integration.test.cjs` | Real atlas-state + controller + runLoop, no model |
| Modify `electron/native/swarm-worker.cjs` | Config, wiring, trajectory fields, `RUNBOOK.json` |
| Create `electron/native/swarm-worker-runbook.test.cjs` | Config tests |
| Modify `electron/native/swarm-eval.cjs` | `--runbook off,on` arm, `controlStats`, report columns |
| Create `electron/native/swarm-eval.test.cjs` | `controlStats` tests |
| Create `docs/validation/2026-09-14-runbook-controller/` | Live A/B packaged results (Task 7) |

---

### Task 1: Runbook controller (pure decision rules)

**Files:**
- Create: `electron/native/providers/runbook-controller.cjs`
- Test: `electron/native/providers/runbook-controller.test.cjs`

**Interfaces:**
- Consumes: nothing from other tasks. Input shapes come from `atlas-state.cjs` `summary(s)` (`{step, kappa, coherence, recovered, strain, …}`) and `phaseSpace(s)` (`{regime, runaway, …}`).
- Produces:
  - `initialState() → {nudges:number, lastNudgeStep:number, windowStartStep:number, windowStartRecovered:number}`
  - `observationFrom(summary, phase, extra) → {step, kappa:number|null, coherence:'locked'|'drift'|'stagnation', recovered, strain, regime, runaway:boolean, outOfRegion:boolean, spentTokens}` where `extra = {outOfRegion, spentTokens}`
  - `decide(obs, state, cfg) → {action:'continue'|'nudge'|'stop', reason:string, note:string, confidence:number, progress:number, state}`; `cfg = {predictedTokens, kappaLine=0.5, nudgeCooldown=3, maxNudges=2, window=5}`
  - `DEFAULTS`

- [ ] **Step 1: Create the branch**

```bash
cd ~/Desktop/ELLE_STACK__CURRENT_WORK/Elle
git checkout -b runbook-controller
```

- [ ] **Step 2: Write the failing tests**

Create `electron/native/providers/runbook-controller.test.cjs`:

```js
'use strict';
const { test } = require('node:test');
const assert = require('node:assert');
const R = require('./runbook-controller.cjs');

const obs = (o) => ({ step: 1, kappa: 0.8, coherence: 'locked', recovered: 0, strain: 0, regime: 0, runaway: false, outOfRegion: false, spentTokens: 0, ...o });

test('continue: locked, confident, under the envelope', () => {
  const d = R.decide(obs({ step: 3, kappa: 0.9, spentTokens: 1000 }), R.initialState(), { predictedTokens: 50000 });
  assert.equal(d.action, 'continue');
  assert.equal(d.confidence, 0.9);
  assert.equal(d.note, '');
});

test('stop (fail-closed): spend exceeds the prediction with no verified progress in the window', () => {
  const d = R.decide(obs({ step: 4, spentTokens: 60001, recovered: 0 }), R.initialState(), { predictedTokens: 60000 });
  assert.equal(d.action, 'stop');
  assert.match(d.reason, /spend 60001 > predicted 60000 with no verified progress/);
});

test('no stop over the envelope while verified progress is still happening', () => {
  const d = R.decide(obs({ step: 4, spentTokens: 90000, recovered: 2 }), R.initialState(), { predictedTokens: 60000 });
  assert.equal(d.action, 'continue');
  assert.equal(d.progress, 2);
});

test('predictedTokens 0 disables the spend stop', () => {
  const d = R.decide(obs({ step: 9, spentTokens: 10 ** 9 }), R.initialState(), { predictedTokens: 0 });
  assert.equal(d.action, 'continue');
});

test('progress is measured over the rolling window, not the whole run', () => {
  let st = R.initialState();
  let d = R.decide(obs({ step: 1, recovered: 3, spentTokens: 10 }), st, { predictedTokens: 100, window: 5 }); st = d.state;
  assert.equal(d.action, 'continue');
  d = R.decide(obs({ step: 5, recovered: 3, spentTokens: 20 }), st, { predictedTokens: 100, window: 5 }); st = d.state;   // window rolls here
  assert.equal(st.windowStartRecovered, 3);
  d = R.decide(obs({ step: 6, recovered: 3, spentTokens: 101 }), st, { predictedTokens: 100, window: 5 });
  assert.equal(d.action, 'stop', 'the earlier successes do not excuse a stalled window');
});

test('nudge: low κ with drift; cooldown suppresses a repeat; nudges again after the cooldown', () => {
  let st = R.initialState();
  let d = R.decide(obs({ step: 2, kappa: 0.3, coherence: 'drift' }), st, {}); st = d.state;
  assert.equal(d.action, 'nudge');
  assert.match(d.note, /^CONTROLLER \(step 2\): you have drifted from the brief \(κ=0\.30\)/);
  d = R.decide(obs({ step: 3, kappa: 0.3, coherence: 'drift' }), st, {}); st = d.state;
  assert.equal(d.action, 'continue');
  assert.match(d.reason, /cooldown/);
  d = R.decide(obs({ step: 5, kappa: 0.3, coherence: 'drift' }), st, {}); st = d.state;
  assert.equal(d.action, 'nudge');
  assert.equal(st.nudges, 2);
});

test('nudge: low κ out of the assigned region', () => {
  const d = R.decide(obs({ step: 2, kappa: 0.4, outOfRegion: true }), R.initialState(), {});
  assert.equal(d.action, 'nudge');
  assert.match(d.note, /left the area of the task/);
});

test('high κ drift is not nudged (continue + score confidence)', () => {
  const d = R.decide(obs({ step: 2, kappa: 0.9, coherence: 'drift' }), R.initialState(), {});
  assert.equal(d.action, 'continue');
});

test('collapse (runaway / regime 3 / regime 4) is nudged, then stopped once the nudge allowance is spent', () => {
  let st = R.initialState();
  let d = R.decide(obs({ step: 1, kappa: 0.7, runaway: true }), st, {}); st = d.state;
  assert.equal(d.action, 'nudge');
  assert.match(d.note, /confidence is falling/);
  d = R.decide(obs({ step: 2, kappa: 0.4, regime: 3 }), st, {}); st = d.state;
  assert.equal(d.action, 'continue');
  d = R.decide(obs({ step: 4, kappa: 0.35, regime: 4 }), st, {}); st = d.state;
  assert.equal(d.action, 'nudge');
  d = R.decide(obs({ step: 5, kappa: 0.3, regime: 3 }), st, {});
  assert.equal(d.action, 'stop');
  assert.match(d.reason, /still collapsing after 2 nudges/);
});

test('observationFrom: maps an atlas summary and phase space', () => {
  const o = R.observationFrom({ step: 8, kappa: 0.42, coherence: 'stagnation', recovered: 2, strain: 1 }, { regime: 4, runaway: false }, { outOfRegion: true, spentTokens: 1234 });
  assert.deepEqual(o, { step: 8, kappa: 0.42, coherence: 'stagnation', recovered: 2, strain: 1, regime: 4, runaway: false, outOfRegion: true, spentTokens: 1234 });
  assert.equal(R.observationFrom({}, {}, {}).kappa, null);
});
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `node --test electron/native/providers/runbook-controller.test.cjs`
Expected: FAIL — `Cannot find module './runbook-controller.cjs'`

- [ ] **Step 4: Write the implementation**

Create `electron/native/providers/runbook-controller.cjs`:

```js
'use strict';
// ============================================================
// RUNBOOK CONTROLLER — the meta-orchestrator's per-step decision for one agent
// (the handwritten Active Inference + CORDIS KV runbook; spec:
// docs/superpowers/specs/2026-09-14-runbook-controller-design.md).
//
//   high κ, low d                      → continue, confidence = κ
//   low κ with drift / out of region   → nudge (2-way: the agent's next observation + the board)
//   collapse (runaway, regime 3 or 4)  → nudge; still collapsing once the nudges are spent → stop
//   spend > predicted, no verified     → stop, fail-closed
//   progress in the recent window
//
// Pure: no I/O. Inputs are what atlas-state already measures every turn;
// progress is the count of VERIFIED actions (atlasState.observe recovered),
// never the agent's own report.
// ============================================================

const DEFAULTS = Object.freeze({ kappaLine: 0.5, nudgeCooldown: 3, maxNudges: 2, window: 5, predictedTokens: 0 });

function initialState() {
  return { nudges: 0, lastNudgeStep: -Infinity, windowStartStep: 0, windowStartRecovered: 0 };
}

function observationFrom(summary, phase, extra) {
  const s = summary || {}; const p = phase || {}; const x = extra || {};
  return {
    step: Number(s.step) || 0,
    kappa: Number.isFinite(s.kappa) ? s.kappa : null,
    coherence: s.coherence || 'locked',
    recovered: Number(s.recovered) || 0,
    strain: Number(s.strain) || 0,
    regime: Number(p.regime) || 0,
    runaway: !!p.runaway,
    outOfRegion: !!x.outOfRegion,
    spentTokens: Number(x.spentTokens) || 0,
  };
}

function nudgeNote(obs, why) {
  const k = obs.kappa == null ? '?' : obs.kappa.toFixed(2);
  const text = {
    drift: `you have drifted from the brief (κ=${k}). Re-read the GOAL and take the next step that serves it.`,
    region: `your recent work has left the area of the task (κ=${k}). Return to the brief's own subject.`,
    collapse: `confidence is falling (κ=${k}, regime ${obs.regime}${obs.runaway ? ', accelerating' : ''}). Stop repeating the current approach: verify one small thing, change strategy, or call done with an honest summary.`,
  }[why];
  return `CONTROLLER (step ${obs.step}): ${text}`;
}

function decide(obs, state, cfg) {
  const c = { ...DEFAULTS, ...(cfg || {}) };
  let st = { ...(state || initialState()) };
  const confidence = obs.kappa == null ? 0 : obs.kappa;
  const progress = obs.recovered - st.windowStartRecovered;
  if (obs.step - st.windowStartStep >= c.window) st = { ...st, windowStartStep: obs.step, windowStartRecovered: obs.recovered };
  const out = (action, reason, note) => ({ action, reason, note: note || '', confidence, progress, state: st });

  if (c.predictedTokens > 0 && obs.spentTokens > c.predictedTokens && progress <= 0) {
    return out('stop', `spend ${obs.spentTokens} > predicted ${c.predictedTokens} with no verified progress in the last ${c.window} steps`);
  }

  const collapse = obs.runaway || obs.regime === 3 || obs.regime === 4;
  const lowKappa = obs.kappa != null && obs.kappa < c.kappaLine;
  const offCourse = obs.coherence === 'drift' || obs.outOfRegion;
  if (collapse || (lowKappa && offCourse)) {
    if (collapse && st.nudges >= c.maxNudges) return out('stop', `still collapsing after ${st.nudges} nudges`);
    if (obs.step - st.lastNudgeStep < c.nudgeCooldown) return out('continue', 'nudge cooldown');
    const why = collapse ? 'collapse' : obs.outOfRegion ? 'region' : 'drift';
    st = { ...st, nudges: st.nudges + 1, lastNudgeStep: obs.step };
    return out('nudge', why, nudgeNote(obs, why));
  }
  return out('continue', 'stable');
}

module.exports = { DEFAULTS, initialState, observationFrom, decide, nudgeNote };
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `node --test electron/native/providers/runbook-controller.test.cjs`
Expected: PASS — 10 tests, 0 failures

- [ ] **Step 6: Commit**

```bash
git add electron/native/providers/runbook-controller.cjs electron/native/providers/runbook-controller.test.cjs
git commit -m "feat(runbook): pure controller — continue, nudge, fail-closed stop from measured κ, drift, phase space and spend

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Spend predictor

**Files:**
- Create: `electron/native/providers/spend-predictor.cjs`
- Test: `electron/native/providers/spend-predictor.test.cjs`

**Interfaces:**
- Consumes: `RUNBOOK.json` record shape written in Task 5: `{ ok:boolean, spentTokens:number, level:number|null, … }`
- Produces:
  - `predict(records, {level, fallback}) → {tokens:number, basis:'fallback'|'all'|'level:<n>', n:number}`
  - `loadRecords(root:string) → Array<record>` (reads `<root>/task-<n>/RUNBOOK.json`)
  - `quantile(sortedNumbers, q) → number`, constants `DEFAULT_FALLBACK=60000`, `MARGIN=1.5`, `MIN_RECORDS=3`

- [ ] **Step 1: Write the failing tests**

Create `electron/native/providers/spend-predictor.test.cjs`:

```js
'use strict';
const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const P = require('./spend-predictor.cjs');

const rec = (spentTokens, level, ok = true) => ({ ok, spentTokens, level });

test('fallback when fewer than 3 successful records', () => {
  assert.deepEqual(P.predict([rec(1000, 1), rec(2000, 1)], {}), { tokens: 60000, basis: 'fallback', n: 2 });
  assert.equal(P.predict([], { fallback: 5000 }).tokens, 5000);
});

test('p75 × 1.5 over all successful runs', () => {
  const r = P.predict([rec(1000, 1), rec(2000, 2), rec(3000, 3), rec(4000, 4)], { level: 9 });
  assert.equal(P.quantile([1000, 2000, 3000, 4000], 0.75), 3250);
  assert.deepEqual(r, { tokens: 4875, basis: 'all', n: 4 });
});

test('the same sphere level is preferred once it has 3 records', () => {
  const r = P.predict([rec(100, 7), rec(200, 7), rec(300, 7), rec(90000, 2), rec(80000, 2)], { level: 7 });
  assert.equal(r.basis, 'level:7');
  assert.equal(r.tokens, Math.ceil(P.quantile([100, 200, 300], 0.75) * 1.5));
});

test('failed runs and zero-spend records are ignored', () => {
  const r = P.predict([rec(1000, 1, false), rec(0, 1), rec(1000, 1), rec(1000, 1)], {});
  assert.equal(r.basis, 'fallback');
  assert.equal(r.n, 2);
});

test('loadRecords reads task-*/RUNBOOK.json and skips unreadable ones', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'spend-'));
  fs.mkdirSync(path.join(root, 'task-1')); fs.writeFileSync(path.join(root, 'task-1', 'RUNBOOK.json'), JSON.stringify(rec(1500, 3)));
  fs.mkdirSync(path.join(root, 'task-2')); fs.writeFileSync(path.join(root, 'task-2', 'RUNBOOK.json'), '{not json');
  fs.mkdirSync(path.join(root, 'other'));
  assert.deepEqual(P.loadRecords(root), [rec(1500, 3)]);
  assert.deepEqual(P.loadRecords(path.join(root, 'missing')), []);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node --test electron/native/providers/spend-predictor.test.cjs`
Expected: FAIL — `Cannot find module './spend-predictor.cjs'`

- [ ] **Step 3: Write the implementation**

Create `electron/native/providers/spend-predictor.cjs`:

```js
'use strict';
// ============================================================
// SPEND PREDICTOR — the runbook's "estimate benchmark + token prediction"
// before an agent runs. The envelope is measured, not guessed: p75 of the
// tokens that prior SUCCESSFUL runs spent (same sphere level once it has
// MIN_RECORDS runs, else every level), times MARGIN. With too little history
// the configured fallback is used and the basis says so.
// ============================================================

const fs = require('fs');
const path = require('path');

const DEFAULT_FALLBACK = 60000;
const MARGIN = 1.5;
const MIN_RECORDS = 3;

function quantile(sorted, q) {
  const i = (sorted.length - 1) * q; const lo = Math.floor(i); const hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

function predict(records, opts) {
  const o = opts || {};
  const fallback = Number(o.fallback) > 0 ? Number(o.fallback) : DEFAULT_FALLBACK;
  const ok = (records || []).filter((r) => r && r.ok && Number(r.spentTokens) > 0);
  const same = o.level == null ? [] : ok.filter((r) => r.level === o.level);
  const useLevel = same.length >= MIN_RECORDS;
  const pool = useLevel ? same : ok;
  if (pool.length < MIN_RECORDS) return { tokens: fallback, basis: 'fallback', n: pool.length };
  const sorted = pool.map((r) => Number(r.spentTokens)).sort((a, b) => a - b);
  return { tokens: Math.ceil(quantile(sorted, 0.75) * MARGIN), basis: useLevel ? `level:${o.level}` : 'all', n: pool.length };
}

function loadRecords(root) {
  let dirs;
  try { dirs = fs.readdirSync(root).filter((d) => /^task-\d+$/.test(d)); } catch { return []; }
  const out = [];
  for (const d of dirs) {
    try { out.push(JSON.parse(fs.readFileSync(path.join(root, d, 'RUNBOOK.json'), 'utf8'))); } catch { /* no record for this task */ }
  }
  return out;
}

module.exports = { predict, loadRecords, quantile, DEFAULT_FALLBACK, MARGIN, MIN_RECORDS };
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `node --test electron/native/providers/spend-predictor.test.cjs`
Expected: PASS — 5 tests

- [ ] **Step 5: Commit**

```bash
git add electron/native/providers/spend-predictor.cjs electron/native/providers/spend-predictor.test.cjs
git commit -m "feat(runbook): spend predictor — p75×1.5 token envelope from prior successful runs, level-first

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `onControl` hook in `runLoop`

**Files:**
- Modify: `electron/native/providers/local-react-agent.cjs:237-240` (end of the while-loop body)
- Test: `electron/native/providers/local-react-agent.test.cjs` (append)

**Interfaces:**
- Consumes: the `decide()` result shape from Task 1 (`{action, reason, note}`), but the hook is generic.
- Produces: `runLoop(goal, catalog, deps, { …, onControl })` where `onControl(steps:number, convo:Array) → {action, reason, note} | null`. `'stop'` ends the run with `{ ok:false, stopped:'controller', final:'stopped by the runbook controller: <reason>.' }`. `'nudge'` appends `\n\n<note>` to the observation the model reads next (same message object, so breathing-window intake keys still match).

- [ ] **Step 1: Write the failing tests**

Append to `electron/native/providers/local-react-agent.test.cjs`:

```js
test('runLoop: onControl nudge is appended to the observation the model reads next', async () => {
  const seen = [];
  const replies = ['{"tool":"run_shell","args":{"command":"ls"}}', '{"tool":"done","args":{"summary":"ok"}}'];
  let i = 0;
  const deps = {
    infer: async (system, messages) => { seen.push(messages.map((m) => m.content)); return { ok: true, content: replies[i++], model: 'm' }; },
    execNative: async () => ({ ok: true, exit: 0, stdout: 'a.txt', stderr: '', duration_ms: 1 }),
    callTool: async () => 'unused',
  };
  const calls = [];
  const res = await agent.runLoop('g', 'c', deps, { maxSteps: 5, onControl: (k) => { calls.push(k); return { action: 'nudge', reason: 'drift', note: 'CONTROLLER (step 1): come back' }; } });
  assert.equal(res.stopped, 'done');
  assert.deepEqual(calls, [1]);
  const lastSeen = seen[1][seen[1].length - 1];
  assert.match(lastSeen, /^OBSERVATION:\nexit 0/);
  assert.match(lastSeen, /\n\nCONTROLLER \(step 1\): come back$/);
  assert.match(res.transcript, /step 1: controller nudge — drift/);
});

test('runLoop: onControl stop ends the run fail-closed as stopped "controller"', async () => {
  const deps = {
    infer: scriptedInfer(['{"tool":"run_shell","args":{"command":"false"}}', '{"tool":"done","args":{"summary":"never"}}']),
    execNative: async () => ({ ok: false, exit: 1, stdout: '', stderr: 'no', duration_ms: 1 }),
    callTool: async () => 'unused',
  };
  const res = await agent.runLoop('g', 'c', deps, { maxSteps: 5, onControl: () => ({ action: 'stop', reason: 'spend 9 > predicted 5 with no verified progress in the last 5 steps', note: '' }) });
  assert.equal(res.ok, false);
  assert.equal(res.stopped, 'controller');
  assert.equal(res.steps, 1);
  assert.equal(res.final, 'stopped by the runbook controller: spend 9 > predicted 5 with no verified progress in the last 5 steps.');
});

test('runLoop: onControl continue changes nothing', async () => {
  const deps = {
    infer: scriptedInfer(['{"tool":"run_shell","args":{"command":"ls"}}', '{"tool":"done","args":{"summary":"ok"}}']),
    execNative: async () => ({ ok: true, exit: 0, stdout: 'x', stderr: '', duration_ms: 1 }),
    callTool: async () => 'unused',
  };
  const res = await agent.runLoop('g', 'c', deps, { maxSteps: 5, onControl: () => ({ action: 'continue', reason: 'stable', note: '' }) });
  assert.equal(res.stopped, 'done');
  assert.doesNotMatch(res.transcript, /controller/);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node --test electron/native/providers/local-react-agent.test.cjs`
Expected: FAIL — the three new tests (`stopped` is `'done'` instead of `'controller'`; nudge text absent)

- [ ] **Step 3: Implement the hook**

In `electron/native/providers/local-react-agent.cjs`, replace:

```js
    lastObs = obs;
    lastObsMsg = { role: 'user', content: `OBSERVATION:\n${obs}` };
    convo.push(lastObsMsg);
    raw.push({ role: 'user', content: `OBSERVATION:\n${obs}` });
  }
```

with:

```js
    lastObs = obs;
    lastObsMsg = { role: 'user', content: `OBSERVATION:\n${obs}` };
    convo.push(lastObsMsg);
    raw.push({ role: 'user', content: `OBSERVATION:\n${obs}` });

    // The runbook controller (swarm-worker): after each observation it may
    // nudge — appended to this same observation so intake keys still match —
    // or end the run fail-closed.
    if (typeof o.onControl === 'function') {
      const c = o.onControl(steps, convo);
      if (c && c.action === 'stop') return finish(false, `stopped by the runbook controller: ${c.reason}.`, 'controller');
      if (c && c.action === 'nudge' && c.note) {
        lastObsMsg.content += `\n\n${c.note}`;
        raw[raw.length - 1].content += `\n\n${c.note}`;
        transcript.push(`step ${steps}: controller nudge — ${c.reason}`);
      }
    }
  }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `node --test electron/native/providers/local-react-agent.test.cjs`
Expected: PASS — all tests, including the 3 new ones

- [ ] **Step 5: Commit**

```bash
git add electron/native/providers/local-react-agent.cjs electron/native/providers/local-react-agent.test.cjs
git commit -m "feat(runbook): runLoop onControl hook — nudge into the next observation or stop fail-closed

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Message-level freezing of verified results

**Files:**
- Modify: `electron/native/providers/atlas-state.cjs` (`compactConversation` at lines 556-569; add `freezeMeta`; export it)
- Test: `electron/native/providers/atlas-state.test.cjs` (append)

**Interfaces:**
- Consumes: breathing-window meta shape `{tier, kappa}` (existing `tierOf` / `intakeKappa`).
- Produces:
  - `freezeMeta(meta) → {...meta, tier: 1, frozen: true}`
  - `compactConversation(convo, s, actions, keepPairs, facts, isPinned?)` — when `isPinned(msg)` is true for a message older than the tail, that message is kept (in order) between the COMPACTED STATE line and the tail.

- [ ] **Step 1: Write the failing tests**

Append to `electron/native/providers/atlas-state.test.cjs`:

```js
test('freezing: a verified observation is Tier 1 frozen — never exhaled, never leaked, kept through compaction', () => {
  const frozenMeta = A.freezeMeta({ tier: 3, kappa: 0.05 });
  assert.deepEqual(frozenMeta, { tier: 1, kappa: 0.05, frozen: true });

  const meta = new Map();
  const mk = (i, m) => { const msg = { role: 'user', content: `OBSERVATION:\n${i} ${'y'.repeat(200)}` }; meta.set(msg, m); return msg; };
  const convo = [{ role: 'user', content: 'GOAL' }];
  const verified = mk('verified', A.freezeMeta({ tier: 3, kappa: 0.05 }));
  const fringe = mk('fringe', { tier: 3, kappa: 0.05 });
  const tailA = mk('tailA', { tier: 2, kappa: 0.5 }); const tailB = mk('tailB', { tier: 2, kappa: 0.5 });
  for (const m of [verified, fringe, tailA, tailB]) convo.push({ role: 'assistant', content: 'act' }, m);

  const b = A.breathe(convo, (m) => meta.get(m) || null, { keepPairs: 2 });
  assert.equal(b.exhaled, 1, 'only the unfrozen fringe is exhaled');
  assert.ok(b.convo.includes(verified), 'the frozen observation survives breathing verbatim');

  const s = A.turn(A.create());
  const folded = A.compactConversation(convo, s, ['a'], 2, [], (m) => !!(meta.get(m) && meta.get(m).frozen));
  assert.equal(folded[0].content, 'GOAL');
  assert.match(folded[1].content, /^COMPACTED STATE/);
  assert.equal(folded[2], verified, 'the frozen observation is kept right after the state line');
  assert.ok(!folded.includes(fringe), 'an unfrozen old observation is folded away');
  assert.deepEqual(folded.slice(-4).map((m) => m.content).filter((c) => c !== 'act'), [tailA.content, tailB.content]);

  const plain = A.compactConversation(convo, s, ['a'], 2, []);
  assert.ok(!plain.includes(verified), 'without isPinned, compaction is unchanged');
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node --test electron/native/providers/atlas-state.test.cjs`
Expected: FAIL — `A.freezeMeta is not a function`

- [ ] **Step 3: Implement**

In `electron/native/providers/atlas-state.cjs`, replace the whole `compactConversation` function with:

```js
function compactConversation(convo, s, actions, keepPairs, facts, isPinned) {
  const keep = Math.max(1, keepPairs || 2);
  const goal = convo[0];
  const rest = convo.slice(1);
  const tail = rest.slice(-2 * keep);
  const older = rest.slice(0, Math.max(0, rest.length - 2 * keep));
  // Frozen (verified) observations are the runbook's locked state: they are
  // carried across every fold instead of being summarized away.
  const pinned = typeof isPinned === 'function' ? older.filter((m) => isPinned(m)) : [];
  const sum = summary(s);
  const learned = (facts || []).filter(Boolean).slice(-8);
  const state = `COMPACTED STATE at F=${s.step} (older turns folded; the record is in the map):\n` +
    `atlas: κ=${sum.kappa} Δ=${sum.delta} (${sum.coherence}) θ=${sum.theta} W=${sum.winding} strain=${sum.strain} recovered=${sum.recovered}\n` +
    `done so far: ${(actions || []).slice(-8).join(' · ') || '(nothing yet)'}\n` +
    (learned.length ? `learned so far (results you already have — use them, do not re-run these):\n${learned.map((f) => `- ${f}`).join('\n')}\n` : '') +
    'Continue from here. Do not repeat work listed above.';
  return [goal, { role: 'user', content: state }, ...pinned, ...tail];
}

// The runbook's phase freeze, at the message level: an observation produced by
// a verified action is locked into Tier 1. breathe() never exhales or leaks
// Tier 1, and compactConversation keeps it when given an isPinned predicate.
function freezeMeta(meta) { return { ...(meta || {}), tier: 1, frozen: true }; }
```

Then in the `module.exports` object, replace:

```js
  windingCheck, lobe, lobeDiagnosis, driftClass, compactConversation, wrapTurns, DIAG_WINDOW,
```

with:

```js
  windingCheck, lobe, lobeDiagnosis, driftClass, compactConversation, freezeMeta, wrapTurns, DIAG_WINDOW,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `node --test electron/native/providers/atlas-state.test.cjs`
Expected: PASS — all tests, including the new freezing test

- [ ] **Step 5: Commit**

```bash
git add electron/native/providers/atlas-state.cjs electron/native/providers/atlas-state.test.cjs
git commit -m "feat(runbook): freeze verified observations as Tier 1 — kept by the breathing window and compaction

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Wire the controller into the swarm worker

**Files:**
- Modify: `electron/native/swarm-worker.cjs` (requires; `config()` lines 60-94; runTask state near line 658; `deps.infer` lines 503-543; `deps.execNative`/`editFile`/`callTool` lines 578-626; `loopOpts` lines 679-691; after the ATLAS.jsonl write at line 719)
- Create: `electron/native/swarm-worker-runbook.test.cjs`
- Create: `electron/native/providers/runbook-integration.test.cjs`

**Interfaces:**
- Consumes: Task 1 `initialState`, `observationFrom`, `decide`; Task 2 `predict`, `loadRecords`; Task 3 `onControl`; Task 4 `freezeMeta`, `compactConversation(…, isPinned)`; existing `atlasState.summary`, `phaseSpace`, `outOfRegion`, `observe`, `turn`.
- Produces:
  - `config()` fields `runbook:boolean`, `runbookTokens:number`, `runbookCeiling:number`
  - events: `emit('runbook', {ref, predictedTokens, basis, n, ceiling})`, `emit('control', {ref, step, action, reason, kappa, spentTokens, predictedTokens})`
  - trajectory rows (ATLAS.jsonl) gain `spent_tokens`, `control`, `control_reason`, `frozen`
  - `<task workspace>/RUNBOOK.json`: `{seq, level, ok, stopped, steps, spentTokens, predictedTokens, predictedBasis, ceiling, frozen, exhaled, leaked, control:[{step, action, reason, kappa, spentTokens, progress}]}`

- [ ] **Step 1: Write the failing config test**

Create `electron/native/swarm-worker-runbook.test.cjs`:

```js
'use strict';
const { test } = require('node:test');
const assert = require('node:assert');
const sw = require('./swarm-worker.cjs');

test('config: the runbook controller is off by default and reads its env', () => {
  const off = sw.config({});
  assert.equal(off.runbook, false);
  assert.equal(off.runbookTokens, 0);
  assert.equal(off.runbookCeiling, 40);
  const on = sw.config({ ELLE_RUNBOOK_CONTROL: '1', ELLE_RUNBOOK_TOKENS: '50000', ELLE_RUNBOOK_CEILING: '30' });
  assert.equal(on.runbook, true);
  assert.equal(on.runbookTokens, 50000);
  assert.equal(on.runbookCeiling, 30);
  assert.equal(sw.config({ ELLE_RUNBOOK_CONTROL: 'yes' }).runbook, false, 'only "1" enables it');
});
```

- [ ] **Step 2: Write the failing integration test (real modules, no model)**

Create `electron/native/providers/runbook-integration.test.cjs`:

```js
'use strict';
// The runbook loop exactly as swarm-worker wires it — real atlas-state, real
// controller, real runLoop — with a scripted model and a box whose commands
// always fail. No network, no model. Proves the pieces compose: failures
// collapse κ, the controller nudges into the observation, and the run ends
// fail-closed once spend passes the envelope with no verified progress.
const { test } = require('node:test');
const assert = require('node:assert');
const atlasState = require('./atlas-state.cjs');
const runbook = require('./runbook-controller.cjs');
const agent = require('./local-react-agent.cjs');

test('failing agent: nudged, then stopped by the controller before the ceiling', async () => {
  let st = atlasState.create();
  let spentTokens = 0; let control = runbook.initialState(); const decisions = []; const seenLast = [];
  let n = 0;
  const deps = {
    infer: async (system, messages) => {
      seenLast.push(messages[messages.length - 1].content);
      spentTokens += 500; st = atlasState.turn(st); n++;
      // a different command every turn, so runLoop's identical-action guard (LOOP_ABORT) never ends the run first
      return { ok: true, content: `{"tool":"run_shell","args":{"command":"make build-${n}"}}`, model: 'scripted' };
    },
    execNative: async () => { const r = { ok: false, exit: 2, stdout: '', stderr: 'make: *** error', duration_ms: 1 }; st = atlasState.observe(st, false, {}); return r; },
    callTool: async () => 'unused',
  };
  const res = await agent.runLoop('build it', 'c', deps, {
    maxSteps: 40,
    onControl: () => {
      const obs = runbook.observationFrom(atlasState.summary(st), atlasState.phaseSpace(st), { outOfRegion: atlasState.outOfRegion(st), spentTokens });
      const d = runbook.decide(obs, control, { predictedTokens: 3000 });
      control = d.state; decisions.push(d.action);
      return d;
    },
  });
  assert.equal(res.stopped, 'controller');
  assert.ok(res.steps < 40, `stopped at ${res.steps}, before the ceiling`);
  assert.ok(decisions.includes('nudge'), `a nudge came before the stop: ${decisions.join(',')}`);
  assert.equal(decisions[decisions.length - 1], 'stop');
  assert.ok(seenLast.some((c) => /CONTROLLER \(step \d+\)/.test(c)), 'the model actually read a nudge');
});
```

- [ ] **Step 3: Run both to verify they fail**

Run: `node --test electron/native/swarm-worker-runbook.test.cjs electron/native/providers/runbook-integration.test.cjs`
Expected: `swarm-worker-runbook.test.cjs` FAILS (`off.runbook` is `undefined`). `runbook-integration.test.cjs` PASSES already (Tasks 1–4 are in place) — it is the regression guard for the wiring below.

- [ ] **Step 4: Add the requires**

Run: `grep -n "require('./providers/atlas-state.cjs')" electron/native/swarm-worker.cjs`
On the line immediately after the match, add:

```js
const runbook = require('./providers/runbook-controller.cjs');
const spendPredictor = require('./providers/spend-predictor.cjs');
```

- [ ] **Step 5: Add the config fields**

In `config()`, replace:

```js
    ledger: (e.ELLE_SWARM_LEDGER || 'agent').toLowerCase(),   // agent | task | model | both
  };
```

with:

```js
    ledger: (e.ELLE_SWARM_LEDGER || 'agent').toLowerCase(),   // agent | task | model | both
    // The runbook controller (docs/superpowers/specs/2026-09-14-runbook-controller-design.md).
    // Off unless ELLE_RUNBOOK_CONTROL=1. With it on, the fixed step budget is
    // replaced by the controller's spend/progress rule under a hard ceiling.
    runbook: e.ELLE_RUNBOOK_CONTROL === '1',
    runbookTokens: Number(e.ELLE_RUNBOOK_TOKENS) > 0 ? Math.floor(Number(e.ELLE_RUNBOOK_TOKENS)) : 0,
    runbookCeiling: Number(e.ELLE_RUNBOOK_CEILING) > 0 ? Math.floor(Number(e.ELLE_RUNBOOK_CEILING)) : 40,
  };
```

- [ ] **Step 6: Run the config test**

Run: `node --test electron/native/swarm-worker-runbook.test.cjs`
Expected: PASS

- [ ] **Step 7: Add per-task runbook state and the prediction**

Replace:

```js
  let pendingNote = '';
  const trajectory = [];
```

with:

```js
  let pendingNote = '';
  const trajectory = [];
  // runbook controller state (inert unless cfg.runbook)
  let spentTokens = 0; let control = runbook.initialState(); let lastControl = null; const controlLog = [];
  let lastVerified = false; let frozen = 0;
  const predicted = cfg.runbook ? spendPredictor.predict(spendPredictor.loadRecords(cfg.root), { level: region ? region.level : null, fallback: cfg.runbookTokens || undefined }) : null;
  if (predicted) emit('runbook', { ref, predictedTokens: predicted.tokens, basis: predicted.basis, n: predicted.n, ceiling: cfg.runbookCeiling });
```

- [ ] **Step 8: Count spend and log control fields per turn**

In `deps.infer`, replace:

```js
      if (r && r.ok && r.content) {
        // The turn itself, for the floor: the model's text and the action it parsed to.
```

with:

```js
      if (r && r.ok) spentTokens += (r.prompt_tokens || 0) + (r.gen_tokens || 0);
      if (r && r.ok && r.content) {
        // The turn itself, for the floor: the model's text and the action it parsed to.
```

In the same function, replace:

```js
        exhaled: breathed.exhaled,
        leaked: breathed.leaked,
```

with:

```js
        exhaled: breathed.exhaled,
        leaked: breathed.leaked,
        spent_tokens: spentTokens,
        control: lastControl ? lastControl.action : null,          // the decision taken after the previous observation
        control_reason: lastControl ? lastControl.reason : null,
        frozen,
```

- [ ] **Step 9: Mark verified actions**

In `deps.editFile`, replace:

```js
      note(`edit ${String(rel || '').slice(0, 80)}`);
```

with:

```js
      note(`edit ${String(rel || '').slice(0, 80)}`);
      lastVerified = false;
```

and replace:

```js
      st = atlasState.observe(st, true, {});
      return done(`edited: ${rel} (1 occurrence)`);
```

with:

```js
      st = atlasState.observe(st, true, {});
      lastVerified = true;
      return done(`edited: ${rel} (1 occurrence)`);
```

In `deps.execNative`, replace:

```js
      st = atlasState.observe(st, !!r.ok, { expectFailure: expectFail });   // a pass, a recorded failure, or the brief's own predicted failure
```

with:

```js
      st = atlasState.observe(st, !!r.ok, { expectFailure: expectFail });   // a pass, a recorded failure, or the brief's own predicted failure
      lastVerified = !!r.ok;
```

In `deps.callTool`, replace:

```js
    callTool: async (tool, args) => {
      const a = args || {};
```

with:

```js
    callTool: async (tool, args) => {
      const a = args || {};
      lastVerified = false;   // a tool result is evidence to read, not a verified act
```

- [ ] **Step 10: Wire freezing, compaction pinning, the ceiling and `onControl` into `loopOpts`**

Replace the whole `const loopOpts = () => ({ … });` block (lines 681-691) with:

```js
  const isFrozen = (m) => { const meta = intake.get(m); return !!(meta && meta.frozen); };
  const loopOpts = () => ({ maxSteps: cfg.runbook ? cfg.runbookCeiling : cfg.maxSteps, timeoutMs: cfg.timeoutMs, persona: cfg.persona,
      onIntake: (msg, meanSurprisal) => {
        const tier = atlasState.tierOf(msg.content, meanSurprisal, invariants);
        let meta = { tier, kappa: atlasState.intakeKappa(meanSurprisal) };
        // the runbook's phase freeze: the result of a verified action is locked (Tier 1, frozen)
        if (cfg.runbook && lastVerified) { meta = atlasState.freezeMeta(meta); frozen++; }
        intake.set(msg, meta);
      },
      breathe: (convo) => { const b = atlasState.breathe(convo, metaOf, { keepPairs: 2 }); if (b.exhaled || b.leaked) { breathed = { exhaled: breathed.exhaled + b.exhaled, leaked: breathed.leaked + b.leaked }; emit('breath', { ref, exhaled: b.exhaled, leaked: b.leaked }); } return b; },
      // Fibonacci-paced GC: fold the window at F_k ≥ 8 so it never grows past
      // GOAL + state + the last two exchanges, however long the run.
      onStep: (k, convo) => (cfg.compact && atlasState.isCheckpoint(k) && k >= 8 && convo.length > 6)
        // Compaction rewrites the transcript, so any cached prefix for this
        // namespace is now stale (§R2): drop it. Fire-and-forget — onStep is
        // synchronous and must still return the replacement window.
        ? (emit('compact', { ref, step: k, before: convo.length }), void evictCache(cacheNs, ref), atlasState.compactConversation(convo, st, lastActions, 2, facts, cfg.runbook ? isFrozen : undefined))
        : null,
      // The runbook controller: continue / nudge (observation + board) / stop fail-closed.
      onControl: cfg.runbook ? (k) => {
        const obs = runbook.observationFrom(atlasState.summary(st), atlasState.phaseSpace(st), { outOfRegion: atlasState.outOfRegion(st), spentTokens });
        const d = runbook.decide(obs, control, { predictedTokens: predicted ? predicted.tokens : 0 });
        control = d.state; lastControl = d;
        if (d.action !== 'continue') {
          controlLog.push({ step: k, action: d.action, reason: d.reason, kappa: obs.kappa, spentTokens, progress: d.progress });
          emit('control', { ref, step: k, action: d.action, reason: d.reason, kappa: obs.kappa, spentTokens, predictedTokens: predicted ? predicted.tokens : 0 });
          postProgress(`step ${k}: runbook controller ${d.action.toUpperCase()} — ${d.reason} (κ=${obs.kappa == null ? '?' : obs.kappa.toFixed(2)}, spent ${spentTokens}/${predicted ? predicted.tokens : '∞'} tokens)`).catch(() => {});
        }
        return d;
      } : undefined });
```

- [ ] **Step 11: Write `RUNBOOK.json` per task**

Replace:

```js
  try { if (trajectory.length) fs.writeFileSync(path.join(root, 'ATLAS.jsonl'),
```

by inserting this line immediately BEFORE it (leave the ATLAS line unchanged):

```js
  try { if (cfg.runbook) fs.writeFileSync(path.join(root, 'RUNBOOK.json'), JSON.stringify({ seq: task.seq, level: region ? region.level : null, ok: !!res.ok, stopped: res.stopped, steps: res.steps, spentTokens, predictedTokens: predicted ? predicted.tokens : null, predictedBasis: predicted ? predicted.basis : null, ceiling: cfg.runbookCeiling, frozen, exhaled: breathed.exhaled, leaked: breathed.leaked, control: controlLog }, null, 1) + '\n'); } catch { /* ignore */ }
```

- [ ] **Step 12: Syntax check and full electron suite**

Run: `node --check electron/native/swarm-worker.cjs && npm run test:electron`
Expected: no syntax error; all electron tests pass (including `swarm-worker-runbook.test.cjs` and `runbook-integration.test.cjs`)

- [ ] **Step 13: Commit**

```bash
git add electron/native/swarm-worker.cjs electron/native/swarm-worker-runbook.test.cjs electron/native/providers/runbook-integration.test.cjs
git commit -m "feat(runbook): wire the controller into the swarm worker behind ELLE_RUNBOOK_CONTROL — spend, freezing, nudges, fail-closed stop, RUNBOOK.json

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Eval A/B arm (`--runbook off,on`)

**Files:**
- Modify: `electron/native/swarm-eval.cjs` (`args` lines 97-109; `runScope` lines 131-179; `grade` lines 198-232; `summarize` lines 234-263; `main` lines 265-291; `module.exports` line 294)
- Create: `electron/native/swarm-eval.test.cjs`

**Interfaces:**
- Consumes: Task 5 events `runbook`, `control`, existing `breath`.
- Produces: `controlStats(events) → {nudges, stops, stopReason:string|null, predictedTokens:number|null, exhaled, leaked}`; report scopes gain `arm: 'off'|'on'|null` and `controller: {nudges, stops, exhaled, leaked}`; rows gain `arm`, `control`.

- [ ] **Step 1: Write the failing test**

Create `electron/native/swarm-eval.test.cjs`:

```js
'use strict';
const { test } = require('node:test');
const assert = require('node:assert');
const E = require('./swarm-eval.cjs');

test('controlStats: counts nudges, stops, the stop reason, prediction and breathing', () => {
  const events = [
    { kind: 'runbook', predictedTokens: 4875 },
    { kind: 'control', action: 'nudge', reason: 'drift' },
    { kind: 'breath', exhaled: 2, leaked: 0 },
    { kind: 'control', action: 'nudge', reason: 'collapse' },
    { kind: 'breath', exhaled: 1, leaked: 1 },
    { kind: 'control', action: 'stop', reason: 'still collapsing after 2 nudges' },
  ];
  assert.deepEqual(E.controlStats(events), { nudges: 2, stops: 1, stopReason: 'still collapsing after 2 nudges', predictedTokens: 4875, exhaled: 3, leaked: 1 });
  assert.deepEqual(E.controlStats([]), { nudges: 0, stops: 0, stopReason: null, predictedTokens: null, exhaled: 0, leaked: 0 });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `node --test electron/native/swarm-eval.test.cjs`
Expected: FAIL — `E.controlStats is not a function`

- [ ] **Step 3: Implement `controlStats` and the arm**

In `args()`, replace:

```js
    else if (a[i] === '--reps') o.reps = Number(a[++i]) || 1;
```

with:

```js
    else if (a[i] === '--reps') o.reps = Number(a[++i]) || 1;
    else if (a[i] === '--runbook') o.runbook = a[++i].split(',').map((s) => s.trim()).filter((s) => s === 'off' || s === 'on');
```

Replace the `runScope` signature and its posting/env lines:

```js
async function runScope(scope, opts) {
```

with:

```js
async function runScope(scope, opts, arm) {
  const tag = arm ? `${scope}/runbook-${arm}` : scope;
```

```js
    const r = await bridge.createTask(`${it.title} [${scope}]`, it.brief, 'normal');
```

with:

```js
    const r = await bridge.createTask(`${it.title} [${tag}]`, it.brief, 'normal');
```

```js
  const env = { ELLE_SWARM_LEDGER: scope, ELLE_SWARM_MAX_STEPS: String(opts.maxSteps), ELLE_SWARM_AGENT_ID: `eval-${scope}`, ELLE_SWARM_DISSOLVE: 'self', ELLE_SWARM_ROOT: WS_ROOT };
```

with:

```js
  const env = { ELLE_SWARM_LEDGER: scope, ELLE_SWARM_MAX_STEPS: String(opts.maxSteps), ELLE_SWARM_AGENT_ID: `eval-${scope}`, ELLE_SWARM_DISSOLVE: 'self', ELLE_SWARM_ROOT: WS_ROOT,
    ...(arm ? { ELLE_RUNBOOK_CONTROL: arm === 'on' ? '1' : '0' } : {}) };
```

```js
  const results = await runWaves(posted.length, env, opts.parallel, scope);
  const attempt1 = grade(posted, results, scope, 1);
```

with:

```js
  const results = await runWaves(posted.length, env, opts.parallel, tag.replace('/', '-'));
  const attempt1 = grade(posted, results, scope, 1, arm);
```

```js
      const results2 = await runWaves(redo.length, env, opts.parallel, `${scope}-r2`);
      attempt2 = grade(posted.filter((p) => redo.some((r) => r.seq === p.seq)), results2, scope, 2);
```

with:

```js
      const results2 = await runWaves(redo.length, env, opts.parallel, `${tag.replace('/', '-')}-r2`);
      attempt2 = grade(posted.filter((p) => redo.some((r) => r.seq === p.seq)), results2, scope, 2, arm);
```

```js
  return summarize(scope, attempt1, attempt2);
}
```

with:

```js
  return summarize(scope, attempt1, attempt2, arm);
}
```

Immediately above `function grade(`, add:

```js
function controlStats(events) {
  const ev = events || [];
  const ctl = ev.filter((e) => e.kind === 'control');
  const rb = ev.find((e) => e.kind === 'runbook') || {};
  const br = ev.filter((e) => e.kind === 'breath');
  const stop = ctl.find((e) => e.action === 'stop');
  return {
    nudges: ctl.filter((e) => e.action === 'nudge').length,
    stops: ctl.filter((e) => e.action === 'stop').length,
    stopReason: stop ? stop.reason : null,
    predictedTokens: Number.isFinite(rb.predictedTokens) ? rb.predictedTokens : null,
    exhaled: br.reduce((a, e) => a + (e.exhaled || 0), 0),
    leaked: br.reduce((a, e) => a + (e.leaked || 0), 0),
  };
}
```

Change `function grade(posted, results, scope, attempt) {` to `function grade(posted, results, scope, attempt, arm) {`, and in its `rows.push({ key: it.key, seq: it.seq, scope, attempt,` replace `scope, attempt,` with `scope, arm: arm || null, attempt, control: controlStats(ev),`.

Change `function summarize(scope, a1, a2) {` to `function summarize(scope, a1, a2, arm) {`, and replace its return:

```js
  return { scope, n: rows.length, correct: rows.filter((r) => r.correct).length, refused: ref.length, auc: A, refusalPrecision, refusalRecall, acceptedAccuracy, energy, retry, rows };
```

with:

```js
  const controller = {
    nudges: rows.reduce((a, r) => a + r.control.nudges, 0),
    stops: rows.reduce((a, r) => a + r.control.stops, 0),
    exhaled: rows.reduce((a, r) => a + r.control.exhaled, 0),
    leaked: rows.reduce((a, r) => a + r.control.leaked, 0),
  };
  return { scope, arm: arm || null, n: rows.length, correct: rows.filter((r) => r.correct).length, refused: ref.length, auc: A, refusalPrecision, refusalRecall, acceptedAccuracy, energy, controller, retry, rows };
```

In `main()`, replace:

```js
  for (const scope of opts.scopes) {
    console.log(`\n== scope: ${scope} · driver: ${opts.driver || 'specialists'} ==`);
    const r = await runScope(scope, opts);
```

with:

```js
  for (const scope of opts.scopes) for (const arm of (opts.runbook && opts.runbook.length ? opts.runbook : [null])) {
    console.log(`\n== scope: ${scope}${arm ? ` · runbook ${arm}` : ''} · driver: ${opts.driver || 'specialists'} ==`);
    const r = await runScope(scope, opts, arm);
```

and after the `energy:` console line add:

```js
    console.log(`  controller: ${r.controller.nudges} nudges · ${r.controller.stops} stops · exhaled ${r.controller.exhaled} · leaked ${r.controller.leaked}`);
```

Replace the markdown table header and row expressions:

```js
  const md = ['| scope | n | correct | refused | AUC κ→correct | refusal precision | refusal recall | accepted accuracy | J total | J per correct | J wasted | retry fixed | retry inherited z | retry J |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|',
    ...report.scopes.map((r) => `| ${r.scope} | ${r.n} |
```

with:

```js
  const md = ['| scope | arm | n | correct | refused | AUC κ→correct | refusal precision | refusal recall | accepted accuracy | J total | J per correct | J wasted | nudges | stops | exhaled | retry fixed | retry inherited z | retry J |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|',
    ...report.scopes.map((r) => `| ${r.scope} | ${r.arm || '-'} | ${r.n} |
```

and in that same row template replace `| ${r.energy.wastedJoules} | ${r.retry ?` with `| ${r.energy.wastedJoules} | ${r.controller.nudges} | ${r.controller.stops} | ${r.controller.exhaled} | ${r.retry ?`.

Replace the exports line:

```js
module.exports = { BATTERY, SMOKE, auc };
```

with:

```js
module.exports = { BATTERY, SMOKE, auc, controlStats };
```

- [ ] **Step 4: Run the tests**

Run: `node --check electron/native/swarm-eval.cjs && node --test electron/native/swarm-eval.test.cjs && npm run test:electron`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add electron/native/swarm-eval.cjs electron/native/swarm-eval.test.cjs
git commit -m "feat(runbook): eval A/B arm --runbook off,on with controller stats in the report

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Live A/B run on Apple Silicon, packaged

**Files:**
- Create: `docs/validation/2026-09-14-runbook-controller/` (copied artifacts, `SHA256SUMS`, `RESULTS.md`)

**Interfaces:**
- Consumes: Tasks 1–6 on the branch; mlx-serve on `127.0.0.1:11437`; agent-bridge.
- Produces: the packaged evidence the spec's R9 asks for.

- [ ] **Step 1: Preconditions**

Quit the Elle app (its own swarm would claim the eval briefs). Then run:

```bash
curl -s http://127.0.0.1:11437/health | head -c 200; echo
node -e "require('./electron/native/providers/bridge-client.cjs').client().health().then((ok)=>{console.log('bridge', ok); process.exit(ok?0:1)})"
```

Expected: health JSON with `"ok": true`; `bridge true`. If either fails, start the missing service and repeat; do not continue until both pass.

- [ ] **Step 2: Run the A/B**

```bash
node electron/native/swarm-eval.cjs --scopes agent --runbook off,on --battery smoke --reps 3 --parallel 2 --max-steps 16 2>&1 | tee /tmp/runbook-ab-console.txt
```

Expected: two scope blocks (`runbook off`, `runbook on`), each ending with `controller: … nudges · … stops · exhaled … · leaked …`, and a final line `report: …/eval-<stamp>.json`.
Note: the `on` arm ignores `--max-steps` by design (hard ceiling `ELLE_RUNBOOK_CEILING`, default 40).

- [ ] **Step 3: Package the evidence**

```bash
OUT=docs/validation/2026-09-14-runbook-controller
REPORT=$(grep -o 'report: .*json' /tmp/runbook-ab-console.txt | tail -1 | cut -d' ' -f2-)
mkdir -p "$OUT/tasks"
cp "$REPORT" "${REPORT%.json}.md" /tmp/runbook-ab-console.txt "$OUT/"
node -e "const r=require(process.argv[1]); for (const s of r.scopes) for (const row of s.rows) console.log(row.seq)" "$REPORT" | while read seq; do
  src="$HOME/Library/Application Support/elle/swarm-workspace/task-$seq"
  mkdir -p "$OUT/tasks/task-$seq"
  for f in ATLAS.jsonl RUNBOOK.json TRANSCRIPT.md; do [ -f "$src/$f" ] && cp "$src/$f" "$OUT/tasks/task-$seq/"; done
done
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS && shasum -a 256 -c SHA256SUMS | grep -c ': OK')
```

Expected: the final count equals the number of files listed in `SHA256SUMS`.

- [ ] **Step 4: Write `RESULTS.md` from the report only**

Create `docs/validation/2026-09-14-runbook-controller/RESULTS.md` by running:

```bash
OUT=docs/validation/2026-09-14-runbook-controller
node -e "
const r = require(require('path').resolve(process.argv[1]));
const L = ['# Runbook controller A/B — ' + r.at, '', 'Command: see runbook-ab-console.txt. Hardware/software: Apple Silicon, mlx-serve on :11437. Every number below is copied from ' + require('path').basename(process.argv[1]) + '.', '',
  '| arm | n | correct | refused | AUC κ→correct | tokens | J total | J per correct | nudges | stops | exhaled | leaked |', '|---|---|---|---|---|---|---|---|---|---|---|---|'];
for (const s of r.scopes) L.push('| ' + [s.arm, s.n, s.correct, s.refused, s.auc == null ? 'n/a' : s.auc.toFixed(3), s.energy.tokens, s.energy.totalJoules, s.energy.joulesPerCorrect ?? 'n/a', s.controller.nudges, s.controller.stops, s.controller.exhaled, s.controller.leaked].join(' | ') + ' |');
L.push('', '## Stops', '');
for (const s of r.scopes) for (const row of s.rows) if (row.control.stops) L.push('- ' + s.arm + ' #' + row.seq + ' ' + row.key + ': ' + row.control.stopReason + ' (correct=' + row.correct + ')');
L.push('', 'Sample size: ' + r.opts.reps + ' repetition(s) of the smoke battery per arm. Treat differences as preliminary until repeated on the full battery.');
require('fs').writeFileSync(process.argv[2], L.join('\n') + '\n');
" "$OUT/$(basename "$(ls "$OUT"/eval-*.json | tail -1)")" "$OUT/RESULTS.md"
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS)
cat "$OUT/RESULTS.md"
```

Expected: a table with one `off` row and one `on` row; every value present in the eval JSON.

- [ ] **Step 5: Commit the evidence**

```bash
git add docs/validation/2026-09-14-runbook-controller
git commit -m "validation(runbook): live A/B off vs on on the smoke battery, packaged with checksums

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Follow-on plans (not in this plan)

- **B — StrataKV from measured κ:** pass breathing-window κ as `priorities` into `tools/mlx-serve/strata_kv.py` via the server's cache path; A/B inside Elle against `KVCache`.
- **C — Sub-Atlas and Master Atlas distillation:** read-only per-agent atlas view; write frozen results and controller stops (failures) back to the map with provenance.
- **D — Retrieval and entropy evals** added to the eval battery.
