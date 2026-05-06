# GWT Exact-Tree Synthetic Validation Spec

Status: v0.1 initial sprint spec.

This sprint treats the GWT DCM as a mechanistic generative model for ordinal
expert ratings. The immediate exercise is M-closed: assume the exact latent
state tree is the data-generating process and ask whether the current GWT data
design identifies the scientific estimands.

The first target semantics are:

- DGP: exact latent binary feature/subfeature tree.
- Fit: exact latent-tree likelihood.
- Leaf: three-state indicator leaf, `m_j in {0, 1, 2}`.
- Nuisance truth: production exact-tree posterior medians by default.
- Design: current GWT rating design: Human 50, Chicken 93, LLMs 186, ELIZA 50.

Composite, marginal, and continuous-tree variants are later comparison or
misspecification models. They are not the first DGP.

## Output Label Contract

Every table, JSON artifact, plot title, and run directory must expose these
labels:

| label | required values for first sprint |
|---|---|
| `dgp` | `exact_latent_tree` |
| `fit` | `oracle`, `toy_exact_tree`, or `full_exact_tree` |
| `leaf` | `three_state_binomial_2` |
| `nuisance_truth` | `exact_tree_production_medians` or `paper_mean_tree_transmission` |
| `design` | `current_gwt_rater_design` |

For CSV or JSON outputs, include the labels as fields on every row or in a
top-level `labels` object. For plots, include the labels in the subtitle or
caption metadata file.

Recommended run id pattern:

```text
{dgp}__{fit}__{leaf}__{nuisance_truth}__{design}__seed{seed}
```

## Validation Taxonomy

### 1. Correctness

Question: does the simulator and exact-tree likelihood compute the intended
probability model?

First-sprint checks:

- Toy exact/exact recovery on a small homologous tree preserving root to
  feature/subfeature to indicator structure.
- Brute-force enumeration for tiny trees where possible.
- Dynamic-programming identities: no-transmission invariance, exact likelihood
  equals enumeration, finite probabilities under all ordinal categories.

Failure interpretation: implementation or algebra bug, not a scientific result.

### 2. M-Closed Calibration

Question: under the exact latent-tree DGP, do posterior intervals and
probabilities behave as Bayesianly calibrated summaries of the same model?

First-sprint checks:

- One full-GWT exact DGP / exact fit synthetic run.
- Coverage and interval width for root `C_s` and path summaries.
- Calibration of internal-node posterior probabilities across simulated
  binary states when conditioning regime makes the target explicit.

This is not a broad SBC campaign. SBC ranks are deferred until the simulator,
likelihood, and estimand definitions have passed narrower checks.

### 3. Practical Identifiability

Question: do the observed ratings contain enough information to learn the
scientific quantities, or are posterior summaries mostly prior-regularised?

Primary target: Arvo's question about internal feature/subfeature states.
Evaluate `Pr(z_sv = 1 | y)`, not hard labels.

Primary metrics for internal nodes:

- Prior entropy, posterior entropy, and relative entropy reduction.
- Brier score against known simulated `z_sv`.
- Log score against known simulated `z_sv`, with clipping for numerical safety.
- Probability calibration by posterior-probability bins.
- Stratification by depth, fanout, subtree rating count, and system.

Interpretation rule: weak recovery means practical non-identifiability under
the current design and exact-tree DGP. It is not evidence that binary internal
nodes are false.

### 4. M-Open Robustness

Question for later: when the real DGP lies outside the exact tree, what
pseudo-true quantities does the model learn, and are they still scientifically
useful?

Deferred DGPs:

- Rater shifts or expert-scale heterogeneity.
- Alternative ordinal layers.
- Continuous internal truth fit by binary exact tree.
- Wrong tree dependence or missing sibling dependence.

These are not first-sprint checks.

### 5. Scientific Usefulness

Question: do the model outputs support the decisions and interpretations we
actually care about?

Primary scientific estimands:

- Root consciousness parameters `C_s`, especially LLMs and Chicken.
- Threshold events such as `Pr(C_s > tau | y)`.
- Internal probabilities `Pr(z_sv = 1 | y)`.
- Path-transmission summaries, including `delta_j` and
  `q_j(0.999) - q_j(0.001)`.
- Observed-scale summaries: ordinal category probabilities, feature-block
  means, top/bottom counts, and sibling-block dependence.

Secondary targets:

- Raw edge betas.
- Cutpoint recovery.
- Hard latent-state classification.
- Full SBC ranks.

## First-Sprint Workstreams

### A. Oracle Root-Signal Audit

Purpose: isolate whether the current rating design can distinguish root
consciousness values when nuisance parameters are fixed.

Labels:

- DGP: `exact_latent_tree`
- Fit: `oracle`
- Leaf: `three_state_binomial_2`
- Nuisance truth: primary `exact_tree_production_medians`; stress
  `paper_mean_tree_transmission`
- Design: `current_gwt_rater_design`

Outputs:

- Expected rating distributions over a grid of `C_s`.
- KL separation curves by system and by feature block.
- Minimum distinguishable changes in `C_s` under the current design.

### B. Oracle Internal-Node Identifiability Audit

Purpose: answer whether internal feature/subfeature states are practically
identifiable under the exact-tree DGP.

Method:

- Generate exact-tree data under known `theta_star`.
- Condition on true nuisance parameters.
- Compute `Pr(z_sv = 1 | y, theta_star)` by exact belief propagation.
- Score probabilities against known simulated `z_sv`.

Required stratifications:

- System.
- Depth.
- Fanout.
- Subtree rating count.
- Feature/subfeature block.

Outputs:

- Node-level probability table.
- Entropy-reduction table.
- Brier/log-score table.
- Calibration plot data.
- Depth/fanout/subtree-count summaries.

### C. Exact-Tree Dependence and Block Audit

Purpose: test whether binary internal states create observed-scale dependence
among descendant indicators that is visible under the current design.

Method:

- Simulate exact-tree data under known nuisance truth.
- Summarise descendant indicator blocks by feature/subfeature.
- Compare block means, top/bottom counts, and sibling dependence to the
  implications of marginal/composite summaries.

Interpretation:

- This is a compatibility check for the observed implications of the binary
  tree structure.
- It is not raw primitive-parameter recovery.

## Full-GWT Synthetic Target

Run one full-GWT exact/exact synthetic fit only after the toy correctness gate.

Labels:

- DGP: `exact_latent_tree`
- Fit: `full_exact_tree`
- Leaf: `three_state_binomial_2`
- Nuisance truth: `exact_tree_production_medians`
- Design: `current_gwt_rater_design`

Root truth:

| system | true `C_s` |
|---|---:|
| Human | 0.999 |
| ELIZA | 0.001 |
| 2024 Leading Chat LLMs | 0.10 |
| Chicken | 0.25 |

Report:

- Root recovery for `C_s` and threshold events.
- Internal-node probability diagnostics.
- Path-transmission summaries.
- Ordinal posterior predictive checks.
- Feature-block posterior predictive checks.

## Acceptance Language

Use these phrases consistently:

- M-closed checks ask: if the exact DCM tree is true, can this design and
  inference recover the relevant quantities?
- M-open checks ask: when the real DGP lies outside the exact tree, what
  pseudo-true quantities does the model learn?
- Practical identifiability means posterior concentration caused by data, not
  merely prior regularisation.
- Internal binary states are probabilistic latent summaries, not recovered
  labels.
- A continuous-tree alternative is justified if binary internal states neither
  improve observed-scale fit nor yield stable posterior probabilities for
  interpretable features/subfeatures.
