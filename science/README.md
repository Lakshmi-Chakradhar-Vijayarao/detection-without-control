# Epistemic Observability Research — Science Governance

## Files

| File | Purpose |
|------|---------|
| `CLAIMS.yaml` | Claim registry. Every scientific claim with status: CONFIRMED / SUPPORTED / EXPLORATORY / FALSIFIED |
| `EXPERIMENTS.yaml` | Experiment registry. Protocol fingerprints, results pointers, dependency map |
| `validate_claims.py` | CI enforcement script. Blocks paper submission if FALSIFIED claims are referenced |

## Rules

1. **Never cite FALSIFIED or EXPLORATORY claims in paper.** Run `python science/validate_claims.py` before any draft.
2. **Never overwrite frozen results.** `results/frozen/` is immutable. New runs create new experiment IDs.
3. **Every experiment must have a protocol_fingerprint.** Two experiments with different fingerprints are not directly comparable.
4. **Update CLAIMS.yaml when a new experiment changes a claim's status.** Do not let status lag behind evidence.

## Running the Validator

```bash
# Check all claims and their statuses
python science/validate_claims.py --list-all

# Check which claims are falsified
python science/validate_claims.py --list-falsified

# Validate paper source against claims (once paper/ directory exists)
python science/validate_claims.py --paper paper/
```

## Claim Status Definitions

| Status | Meaning | Paper? |
|--------|---------|--------|
| CONFIRMED | ≥2 architectures, clean controls, N≥128/class | ✅ Yes |
| SUPPORTED | Single architecture or small N; valid but needs replication | ✅ Yes (with caveat) |
| EXPLORATORY | Preliminary finding, insufficient evidence | ❌ No |
| FALSIFIED | Directly contradicted by controlled experiment | ❌ No (must disclose) |
