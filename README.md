# sway-eval

**Audit how much the order of a record's sections decides an AI decision model's verdict.**

> Status: version 0.1.0, the first release. The API may change in later versions; see
> [Known limitations](#known-limitations).

A decision model takes a record and a question with fixed options, such as ADVANCE or DO NOT
ADVANCE, and returns a probability for each option in one pass. When a deployer screens records
with a stated rule, the order in which a record presents its sections is not part of the rule. If
reordering the same sections changes the verdict, the model has added a criterion nobody wrote.

`sway-eval` lets you measure that on your own records, without labels, and compare the defenses
that remove it:

- **How often layout decides.** For each record, every admissible ordering of its sections (or a
  balanced subset) is scored, and each decision is compared with the model's own consensus, the
  decision on the probabilities averaged over orderings. The flip probability of a record is the
  share of orderings that disagree; the flip zone is the range of qualification levels where it
  reaches a threshold you set.
- **What it costs.** The variance of the prediction across orderings is exactly the Brier loss
  that averaging removes, so the cost is computed on unlabeled records. It is compared with a
  formatting baseline (separators, headings, bullets) and an option-order baseline, so you can
  tell order sensitivity from general brittleness.
- **What each defense buys.** A fixed canonical layout, a fixed layout chosen on held-out records,
  k random layouts, a balanced Williams set (2n orderings for odd n, ten for five sections) and an
  adaptive screen that runs the balanced set only near the decision boundary, each with its calls
  per record and its agreement with the full average.
- **Whether the model changed while you measured.** For hosted models, a repeat study measures
  run-to-run noise and subtracts it, and canary records re-scored between batches halt the run if
  the serving model changes.

It measures and compares defenses. It contains no search for orderings that move a model.

## What is in the package

- **Adapters** for hosted decision APIs (TypeSafe Jev, Microsoft-Decision-1 on Azure AI Foundry,
  the OpenAI Decisions API) and local open-weight decision models (Cloudflare Clef and Clef-flash,
  Decision-2.0), plus a **mock** decision model with known position weights for testing
- **A runner** that plans before it calls (planned calls, estimated cost and wall time; paid or long
  runs need `--yes`), caches every response in sqlite so interrupted runs resume and repeated runs
  cost nothing, can cap spending per provider, pins the model version and never overwrites a run
- **Measures:** symmetrized prediction, order variance, flip probability and flip zone (direct and
  model-based, including a margin-binned zone for three-option tasks), pairwise and confident flip
  rates, the layout-to-formatting ratio
- **Layouts:** records as movable units, all orderings, Williams designs, seeded samples
- **A synthetic rubric generator:** résumé-like records at known distances from a stated threshold,
  with labels fixed by the rule
- **Utilities:** margins from logits or probabilities, contextual calibration, call-count arithmetic

## Install

`sway-eval` needs Python 3.10 or newer; the tests run on 3.10 through 3.15.

With pip:

```bash
git clone https://github.com/raminzandvakili/sway-eval.git
cd sway-eval
python -m venv .venv && . .venv/bin/activate
pip install -e .
```

With [uv](https://docs.astral.sh/uv/), which installs the exact versions in `uv.lock` (and
downloads Python 3.15, the version in `.python-version`, if you don't have it):

```bash
git clone https://github.com/raminzandvakili/sway-eval.git
cd sway-eval
uv sync
```

Local open-weight models need the `local` extra (`pip install -e ".[local]"` or
`uv sync --extra local`); hosted models read their keys from `.env` (copy `.env.example`). A PyPI
release (`pip install sway-eval`) will follow. In Python the package is imported as `sway_eval`.

## Quick start: an audit on the mock model

No key, no GPU and no cost. From the repository root:

```bash
sway-eval audit examples/mock_audit.yaml          # with pip, in the activated environment
uv run sway-eval audit examples/mock_audit.yaml   # with uv
```

This generates 50 synthetic records, prints the plan, makes about 6,700 calls to the mock decision
model (about a minute) and writes `report.md` and `report.json` into a new folder under
`examples/out/results/`. The mock has planted position weights, a carryover between adjacent
sections, a small formatting effect, an option-order bias and run-to-run noise, so the report has a
known answer. It shows:

- how often layout decides: the mean flip probability and the flip zone by qualification level;
- what it costs: the order SD, the formatting and option-order SDs, rho, and the order variance;
- the noise study and the canary checks;
- what each defense buys: calls per record, agreement with the full average (with bootstrap
  intervals), the section gap and the residual, for the convention layout, a held-out fixed
  layout, k random layouts, the balanced Williams set, the full average and the adaptive screen.

`examples/mock_drift.yaml` plants a serving change after 3,000 calls; the canaries halt that run
and the report says so. `examples/mock_exact.yaml` audits the same records with the mock's noise and
rounding switched off, so its report gives the values the noisy audit should print; run it after
`mock_audit.yaml` and compare the two reports. `examples/recover_mock.py` fits the mock's planted
parameters from a run's records. It needs numpy, which the package does not install:
`uv run --with numpy python examples/recover_mock.py examples/out/results/<run> recovery.json`
with uv, or `pip install numpy` and then `python examples/recover_mock.py ...` with pip.

To audit a real model, copy the example, set `model.provider`, `model.name` and the exact
`model.version`, and point `dataset.items` at your records. Set `layouts.kind: williams` for a routine
audit: the balanced set, ten orderings per record for five sections (2,000 calls for 200 records),
whose average removes position effects and first-order carryover exactly. The example uses every
ordering (`all`, 120 per record) because that is what validates the balanced set and fills the
defense comparison. Any Hugging Face sequence classifier with fixed labels works through
`provider: hf_classifier`; `examples/hf_classifier_audit.yaml` runs one (a small NLI model, on CPU). Paid runs, and runs expected to take over
an hour, stop after the plan unless you add `--yes`. The steps also run one at a time:
`sway-eval generate`, `plan`, `run`, `analyze` and `report`.

## Configs

One YAML file describes a run: the model, the records, the instruction with the stated rule, the
layouts, the baselines, the noise study and canaries, and the analysis settings (tie rule, zone
width convention, threshold). A config is rejected if a choice that changes the result is left
unset. See `experiments/_template.yaml` for every field.

## Known limitations

These are known in 0.1.0 and will be addressed in the next version.

- The defense comparison evaluates the balanced set with a record's sections labelled in its
  template order, while an audit with `layouts.kind: williams` labels them by sorted unit id. Both
  are balanced sets, but they are different sets of orderings, so the comparison does not score
  exactly the orderings a routine audit runs.
- The formatting SD, and with it rho, is not corrected for run-to-run noise (the order variance is).
  On a noisy model the formatting SD is overstated and rho understated.
- Agreement intervals are bootstrap intervals over records, so a defense that agrees on every record
  gets the interval [1.00, 1.00]; an exact binomial interval would say more.
- On the mock with noise and rounding switched off (`examples/mock_exact.yaml`), eight orderings
  have a probability of exactly 0.5 in exact arithmetic, and the side of 0.5 they compute to depends
  on the Python version, since the mock adds its terms with `sum()`. Under Python 3.10 and 3.11 all
  eight come out just above 0.5; under 3.12 and 3.15 six come out just below and two at exactly 0.5,
  which the tie rule decides. The noise-free flip probability therefore reads 0.087 under 3.10 and
  3.11 and 0.088 under 3.12 and 3.15. Other models are not affected: the package takes their
  probabilities as returned. The quick-start output (`examples/mock_audit.yaml`) is the same under
  3.10 and 3.15.

## Tests

```bash
pytest            # or: uv run pytest
```

Tests never call a network or a paid API; they use the mock.

## How to cite

If you use `sway-eval`, please cite it. GitHub's **Cite this repository** button (from
`CITATION.cff`) gives APA and BibTeX. Or use:

```bibtex
@software{zandvakili_sway_eval_2026,
  author  = {Zandvakili, Ramin and Eshgarf, Nasim},
  title   = {sway-eval: Auditing Order Sensitivity in AI Decision Models},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/raminzandvakili/sway-eval}
}
```

## License

MIT. See [LICENSE](LICENSE).
