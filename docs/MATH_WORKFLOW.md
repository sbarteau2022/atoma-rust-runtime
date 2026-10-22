# MATH WORKFLOW — Word Problems by Derivation, Not Resemblance

Status: DESIGN SPEC (2026-10-19, Stewart + Muse). Automaton side is
Muse's lane to build; equation/key derivation is the model's job.
Supersedes: the GSM8K phrase-template solver's role (named test-gaming
2026-10-19; its numbers must not be cited as capability).

## The workflow

**Stage 0 — Derive the key (Relevance Classification, 2026-10-20).**
Before any equation: bind symbols to the problem's entities
("r = bolts of blue fiber"). The key is TWO-SLOT:

    Key = { OperativeBindings: [x1, x2, ...],
            ContextualDistractors: [d1, ...] }

Every quantity in the problem is classified into exactly one slot.
OperativeBindings are the quantities the derivation operates on;
ContextualDistractors are quantities the problem mentions but the
asked quantity does not depend on (scene-setting numbers, irrelevant
ages, decoy counts). Strictness REVISED: every variable used later
must appear in OperativeBindings; an operative binding left unused by
Stage 1 is still a mis-parse — but a quantity sitting in
ContextualDistractors is DISCARDED BY DECISION, not by omission, and
never triggers the mis-parse loop. Mis-classification (an operative
quantity filed as a distractor) is an empirical error, judged by
outcomes — the ledger is where classification quality is measured.

**Stage 1 — Derive the equation.** From the key, state the equation(s).
Structure only; no numbers asserted as conclusions.

**Stage 2 — Reason steps, delegate computations.** The model may reason
the derivation's steps, but every computation goes immediately to the
automaton (math_eval). The model states no intermediate values of its
own. The automaton is the only component permitted to touch numbers.

**Stage 3 — Back-substitute and validate (Semantic Balancer).**
Place the computed assignment back into the original equation; the
automaton evaluates both sides. The answer exists only if the
equation balances. ADDITIONALLY (2026-10-20, Stewart): DIMENSIONAL
ANALYSIS. Quantities in the key carry units (5 apples, 3 hours,
12 dollars). The automaton tracks units through every operation:
5 apples + 3 apples = 8 apples (unit preserved); 5 apples x 3 apples
= 15 apples^2 — if the asked quantity's unit is apples, the AST is
REJECTED dimensionally BEFORE any numeric balancing. This kills the
False-Equilibrium trap: a wrong-structure equation whose numbers
happen to balance. The answer's unit must equal the asked unit, or
the turn is a named refusal (refusal_reason: dimensional_mismatch)
and files NO bet.
(Honesty boundary: numeric balancing validates the (equation, answer)
pair; dimensional analysis validates the equation's STRUCTURE against
the asked unit. Neither proves the equation matches the English —
parse fidelity remains empirical, judged by ledger outcomes.)

**Stage 4 — Print "the answer is …"** Only after Stage 3 validates.
That sentence is a claim; it files a bet in the outcome ledger like
every other claim.

## Global rule: dynamic iteration, never decline-and-guess

- Key incomplete → **loop back**: re-examine the problem, fill the
  missing bindings, re-verify key completeness, continue. Each pass
  must strictly increase key completeness (monotone progress), or the
  loop is spinning.
- Bounded: after the iteration cap, or when the keys cannot be parsed
  at all → **stop and confess**, in the workflow's own words: "I can't
  parse this correctly, and I won't give a confident-seeming answer
  that I can't validate confidently." This is a named refusal
  (refusal_reason: unparseable_keys) — diagnosable, not silent.
- Never print an answer that did not pass Stage 3.

## Implicit-Constant Lexicon (2026-10-20, Stewart)

Pragmatic arithmetic words are CONSTANTS of the language, not content
to be derived. The boundary transducer recognizes them and emits them
as native arithmetic carrier constants BEFORE AST generation:

    dozen  -> 12        twice -> x2 (operator)    half -> /2 (operator)
    pair   -> 2         weekly -> 7 (days)

These are closed-class lexical facts (like number words), handled by
the transducer/math front end — never by the model's derivation.

## Ingestion facts (verified 2026-10-19)

- The eval harness strips JSONL wrappers; the operator receives clean
  question text. JSON feeding is NOT the cause of the 4% full-set
  GSM8K result — the phrase-template binding failure is.
- The model path is different: the boundary transducer converts
  English to carrier ops, and its conversion certainty for prose is
  unproven (egress garble is standing evidence). For the model to
  derive keys/equations, either the transducer converts with certainty
  or problems must reach the model in a native-legible form.
  Verbose math (Stage 0–1 output form) is that form: exact to
  transduce, exact for the automaton to execute.

## Depth input: relational backbone only (2026-10-20, Stewart)

Breathing depth construction must NOT compute PAMI spectral spread
over the raw English sentence. Decorative narrative (adjectives,
atmospheric setup clauses) is roughness, not difficulty (the
BREATHING_DEPTH_DESIGN §5 failure mode). PAMI complexity analysis
runs ONLY over the relational/numerical clause backbone: number
tokens, operator words, and relational words (per, each, total,
left, more than, ...). The full sequence still propagates through
whatever depth is constructed; only the depth DECISION reads the
backbone.

## The criterion (from the 2026-10-19 eliminations)

Any mechanism proposed for word problems must answer: how does it
handle a problem guaranteed unseen, by derivation rather than
resemblance? Teacher serving (removed), templates (gaming), stored
trajectories (recitation) all fail this test. This workflow is the
serving discipline for a model that derives; the model's parse
competence itself is the open model-change problem (see
BREATHING_DEPTH_DESIGN.md).
