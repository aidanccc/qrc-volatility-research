# Reproduction status

## Modernization in progress — September 14, 2026

- Read-only project/history/source audit completed; previous superiority claims
  corrected in the README and preserved in a historical local archive.
- Yahoo/FRED payloads downloaded and retained with hashes. Daily data reaches
  September 14; completed monthly targets reach August 2026.
- January 1950 difference reconciled exactly to an omitted first return.
- Seven provider discrepancies recorded and resolved by documented FRED
  precedence; no unexplained exchange sessions are missing.
- Causal preprocessing, model adapters, CLI, run identities, checkpoints, and
  artifact-only reporting implemented. Core tests pass, including deterministic
  LSTM reconstruction independent of execution order.
- Full modern run and isolated original benchmark are executing. Completion,
  coverage, numerical failures, and results will be recorded after validation.

See [the audit](MODERNIZATION_AUDIT.md), [assumptions](ASSUMPTIONS.md), and
[run order](RUN_ORDER.md). Original ignored historical notes remain in `docs/`.
