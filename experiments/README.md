# Experiments

One config file describes one run. Copy `_template.yaml` to start a new one, or start from one of
the configs in `examples/`.

## Running

```bash
sway-eval audit CONFIG         # all of the steps below, then report.md and report.json
sway-eval generate CONFIG      # build synthetic items, if the config uses them
sway-eval plan CONFIG          # planned calls, estimated cost and wall time
sway-eval run CONFIG           # paid runs, and runs over an hour, need --yes
sway-eval analyze RUN_DIR      # per-record measures and a run summary
sway-eval report RUN_DIR       # the audit report for a finished run
```

Every call goes through the response cache, so an interrupted run resumes and a repeated run costs
nothing. A run folder is never overwritten.

## Rules

- **Pin the model version.** Hosted decision models change without notice. Record the version
  string the API returns, not just the model name. The runner stops if the version changes mid-run.
- **Outputs go to `output_root/<date>_<name>_<model-version>/`**, with a copy of the config (the
  runner does both). Keep `output_root` and `cache` outside the repository.
- **Fix the random seed** and record it.
- **Fix the tie rule and the zone-width convention** in the config before a run; the config is
  rejected without them.
- **Check the budget before you run.** `sway-eval plan` prints it. The balanced set needs 2n
  orderings per record for odd n (ten for five sections); every ordering, which validates the
  balanced set, grows as n!; `sway_eval.exhaustive_calls(n_examples, n_passages)` gives the count:

| Passages per example | Orderings per example | Calls for 50 examples |
| --- | --- | --- |
| 5 | 120 | 6,000 |
| 6 | 720 | 36,000 |
| 7 | 5,040 | 252,000 |
| 8 | 40,320 | 2,016,000 |
