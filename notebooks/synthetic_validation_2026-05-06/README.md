# GWT Synthetic Validation Sprint

Initial branch: `synthetic-validation-checks`

## Validation Spec

The sprint contract is in `validation_spec.md`. It separates correctness,
M-closed calibration, practical identifiability, later M-open robustness, and
scientific usefulness, and requires every output to label DGP / fit / leaf /
nuisance truth / design.

## Working Defaults

- Primary semantics: exact latent-state tree.
- First baseline: unpooled node-level edge betas fixed at paper prior means.
- Indicator leaf: three-state `m_j in {0, 1, 2}`.
- Observation layer for the first synthetic smoke: no expert shifts; shared `a`
  and `kappa` seeded from existing exact-tree production posterior medians.
- Pooled betas are a later regularisation/sensitivity check, not the default.
- Composite propagated-`q` fits are a later approximation comparison, not the
  first data-generating process.

## First Run Target

Generate one full-GWT synthetic dataset with the current rater design:

- Human `C = 0.999`
- ELIZA `C = 0.001`
- Chicken `C = 0.25`
- 2024 Leading Chat LLMs `C = 0.10`

The first script writes one run directory containing:

- `synthetic_stance_data.json`
- `truth.json`
- `sanity_checks.json`
- `config.json`

Run from the repo root with:

```bash
.venv/bin/python notebooks/synthetic_validation_2026-05-06/gwt_exact_unpooled_synthetic_smoke.py
```

## Oracle Internal-Node Identifiability

Run Arvo's practical-identifiability audit without PyMC sampling:

```bash
.venv/bin/python notebooks/synthetic_validation_2026-05-06/gwt_oracle_internal_identifiability.py
```

Default labels:

- DGP: exact latent tree
- Fit: oracle
- Leaf: three-state
- Nuisance truth: exact-tree production medians
- Design: current GWT rater design

The audit simulates exact-tree datasets, conditions on true nuisance
parameters, computes `Pr(z_sv=1 | y, theta*)` by exact clamped dynamic
programming, and writes:

- `summary.md`
- `node_probabilities.csv`
- `stratified_summary.csv`
- `calibration.csv`
- `overall_summary.csv`

For the paper-mean tree-transmission stress truth:

```bash
.venv/bin/python notebooks/synthetic_validation_2026-05-06/gwt_oracle_internal_identifiability.py --nuisance-truth paper_mean_tree_transmission
```

Practical reading:

- The output is not asking whether binary internal states are philosophically
  correct. It asks whether the current ratings would let the model learn those
  states if the binary tree were true.
- Strong nodes can support cautious probability statements, such as
  `Pr(feature present | ratings)`.
- Weak nodes should not be reported as recovered feature/subfeature labels;
  they are mostly prior/tree-structure inheritance under the current design.
- The most useful audience-facing summary is probability plus entropy reduction
  by node or feature block, not hard state classification.

## Oracle Root-Signal Audit

Run the fixed-nuisance root-signal audit:

```bash
.venv/bin/python notebooks/synthetic_validation_2026-05-06/gwt_oracle_root_signal.py
```

For the paper-mean tree-transmission stress truth:

```bash
.venv/bin/python notebooks/synthetic_validation_2026-05-06/gwt_oracle_root_signal.py --nuisance-truth paper_mean_tree_transmission
```

The audit computes expected ordinal rating distributions over a grid of root
`C` values and marginal KL separation curves from each system's reference
truth.

Practical reading:

- If KL is shallow near a system's reference `C`, then the current ratings do
  not strongly distinguish nearby root values on their own.
- Under the default production-median nuisance truth, LLM `C=0.10` versus
  Chicken-like `C=0.25` is weakly separated by the marginal rating design.
- Under paper-mean tree transmission, root signal is weaker still; many root
  distinctions would depend on priors, anchors, or shared nuisance learning.
- The KL here is marginal over ordinal rating slots; sibling dependence is a
  separate exact-tree block/dependence question.

## Full Exact/Exact Fake-Data Recovery

Run the full one-seed M-closed recovery pilot:

```bash
.venv/bin/python notebooks/synthetic_validation_2026-05-06/gwt_full_exact_recovery.py
```

Run the fast pipeline smoke first:

```bash
.venv/bin/python notebooks/synthetic_validation_2026-05-06/gwt_full_exact_recovery.py --smoke
```

The full run generates exact-tree synthetic data, fits the exact-tree model,
and writes `fit.nc`, `truth.json`, recovery tables, diagnostics, and
`summary.md`.

Practical reading:

- This is fake-data recovery, not a final multi-seed calibration study.
- One-seed signed errors are recovery errors for this generated dataset, not
  stable bias estimates.
- The key free-root result is whether Chicken and LLM truth values fall inside
  posterior intervals and whether the intervals are practically narrow.
- Internal-state recovery should be read as probability scoring, not hard
  latent-state classification.
