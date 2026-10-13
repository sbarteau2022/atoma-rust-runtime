# Tier 0 parity qualification

## Function

Proves that a candidate backend emits the same first 128 tokens as the designated oMLX baseline for identical tokenized prompts.

## Workflow

1. Record the exact model revision, tokenizer, backend settings, and prompt vectors.
2. Run the baseline and candidate with greedy decoding and identical stop conditions.
3. Feed their token IDs to `core_lm::qualify_token_parity`.
4. Reject the candidate if either result is under 128 tokens or any compared token differs.
5. Preserve the raw streams and environment metadata with the report.

## Edge cases

Tokenizer mismatches, different EOS policies, sampling, truncation, and backend errors invalidate the run; none may be treated as a pass.
