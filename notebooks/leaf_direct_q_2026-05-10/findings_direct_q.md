# Branch 2 — `leaf-direct-q` findings

**Branch:** `leaf-direct-q` off `asymmetric-beta-prior-sweep` (commit `929063b`).
**Date:** 2026-05-10. **Stance:** GWT only.

## Question

Pro_A ablation: replace the three-state binomial-2 leaf with a parameter-free
**direct_q** leaf — η_je = a · q̃_j + b_e, no latent indicator state — and
ask whether removing the discrete-z compression alone meaningfully tightens
free C posterior contraction at the centre β prior
(`pres0.90 / abs0.10 / sig0.30`).

q̃_j is the indicator-effective probability $\tilde q_j = \beta^{\rm abs}_j +
q_j(\beta^{\rm pres}_j - \beta^{\rm abs}_j)$. In the composite path the
upstream propagation already supplies q̃_j; in the exact-tree path the
indicator's own β (β_pres conditional on parent z=1, β_abs conditional on
parent z=0) IS q̃_j.

## Bottom line

**No-marginal.** Removing the discrete-z compression delivers ~3× the
analytical per-rating Fisher information for C (and ~1.7× the mutual
information $I(Q;Y)$), but the empirical free-C posterior contraction
moves by **less than 1 percentage point** vs the three_state baseline at
the same prior and design (Chicken contraction 0.927 → 0.920, LLM
contraction 0.957 → 0.949; both still ≈1, i.e. C posterior ≈ C prior).
Per-indicator PPC chi² is essentially the same in aggregate. The posterior
medians are within 0.005 of three_state.

This is a **striking and instructive negative result**: the analytical
Fisher / mutual-info ceiling and the empirical posterior contraction are
*not* in agreement on this design. The leaf compression is not the
binding constraint on C identification at the centre β prior with the
current GWT rater design — even though analytically it looks like it
should be.

This strongly motivates Branch 3 (mixture, Pro_D) as the immediate next
session, *and* sharpens the hypothesis: if Branch 3 also fails to move
contraction, the binding constraint is design / signal-strength, not the
leaf model. The Branch 1 oracle-C ladder (running in parallel) will tell
us whether *any* leaf change can move C contraction once nuisance is
pinned.

## Implementation

- `dcm_model.py:95` — extended `INDICATOR_STATE_MODEL` Literal to include
  `"direct_q"`.
- `dcm_model.py:pt_direct_q_ll` (new) — leaf log-lik helper. Wraps
  `pt_ordinal_logp` / `pt_ordinal_logp_expert_kappa` with
  `eta_shift = a * q_j_eff`. Returns scalar log-lik for one indicator's
  ratings.
- `dcm_model.py:add_indicator_marginal_likelihood` — composite-path
  dispatcher branch. Calls `pt_direct_q_ll(ratings, expert_idx, q_j, ...)`
  where `q_j` (line ~1294) is already $\tilde q_j$ via the upstream
  `parent_prob * β_pres + (1 - parent_prob) * β_abs` propagation. Exposes
  `{name}_q_eff` deterministic for diagnostics; no latent z determinstics.
- `dcm_model.py:add_no_data_prior_deterministics` — direct_q branch
  exposes `{name}_q_eff` only.
- `dcm_model_exact_tree.py` — accepted `direct_q` in the state-model
  guard. In `_collect_indicator_leaf_lls` for direct_q we *don't*
  precompute leaf log-lik tensors (η depends on β at evaluation time);
  instead store `(ratings, expert_idx)` arrays for `leaf_log_B` to call
  `pt_direct_q_ll` with the correct β. The upstream tree DP is
  unchanged (only the per-indicator leaf message simplifies).
- `notebooks/synthetic_validation_2026-05-06/gwt_full_exact_recovery.py`
  — added `--state-model {binary, three_state, direct_q}` CLI flag
  (default `three_state` preserves the baseline) and ported
  `--beta-pres-mean / --beta-abs-mean / --beta-override-sigma` from
  Branch 1's working tree (the asymmetric-prior-sweep commit message
  flags these as intentionally not committed). Run-id appends
  `__sm{state}` when non-default.

### Verification (passed)

- `python test_label_pooled_smoke.py` — defaults preserved (3-state
  baseline still validated).
- `python notebooks/leaf_direct_q_2026-05-10/sanity_a_to_zero.py` — at
  $a → 0$ the predicted distribution is invariant in $q$ (max |Δ| = 5.6e-9
  across q ∈ {0, 0.5, 1}); at $a = 2$ the q-dependence is large
  (|Δ| = 44.8 nats over 30 ratings). Confirms q-dependence is carried by
  `a` alone.
- Smoke fit (`--smoke`, 80 tune / 80 draws, 2 chains) compiles + samples
  cleanly in 52 s with **0 divergences**.
- Full fit (4 chains × 1000 tune × 1000 draws) sampled in 199 s, **0
  divergences**, max R̂ = 1.000, min ESS_bulk = 5248.

## Pre-flight: analytical leaf information at production-median nuisance

`info_diagnostics.py --leaf direct_q --truth-json …/truth.json` and
`--leaf three_state` on the same truth (centre β prior, seed 06):

| leaf | I(Q;Y) nats | H(Y\|q) nats | Fisher (Chicken) | Fisher (LLM) | Full-vector KL Chicken (C=.9 vs C=.1) |
|---|---|---|---|---|---|
| three_state | 0.240 | 1.512 | 11.76 | 22.68 | 3.85 |
| **direct_q** | **0.418** | **1.449** | **34.69** | **66.60** | **10.98** |

direct_q delivers ~1.74× per-rating mutual information, ~3× expected
Fisher information for C, ~2.85× full-vector KL between C=0.9 and C=0.1
for Chicken at this design. The discrete-z compression is genuinely
discarding rating-level information about q at the analytical level.

(Full-vector KL here is the per-indicator-independent upper bound from
`info_diagnostics.py`; the true joint KL is no larger but this is the
right relative-comparison number.)

## Sweep design

| # | scope | run_id suffix | β_pres | β_abs | sigma | divs | max R̂ |
|---|---|---|---|---|---|---|---|
| 1 | synthetic, ref-anchored, seed 06 | `smdirect_q__pres90_abs10_sig30` | 0.90 | 0.10 | 0.30 | 0 | 1.000 |

Comparator on disk:

- Three-state baseline at this prior:
  `notebooks/asymmetric_prior_sweep_2026-05-10/runs/synthetic/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__pres90_abs10_sig30/`

## Results — synthetic seed 06 (free C recovery)

Truth: Human 0.999 (anchored), Chicken 0.25, LLM 0.10, ELIZA 0.001 (anchored).

| leaf | Chicken median | Chicken [p03, p97] | LLM median | LLM [p03, p97] | Chicken signed bias | LLM signed bias |
|---|---|---|---|---|---|---|
| three_state (baseline) | 0.118 | [0.006, 0.460] | 0.123 | [0.007, 0.484] | −0.132 | +0.023 |
| **direct_q** | **0.115** | **[0.006, 0.460]** | **0.127** | **[0.007, 0.493]** | **−0.135** | **+0.027** |

Posterior-vs-prior contraction (`posterior_vs_prior.py`):

| leaf | Chicken contraction | LLM contraction |
|---|---|---|
| three_state (baseline) | 0.927 | 0.957 |
| **direct_q** | **0.920** | **0.949** |

(Contraction = posterior_SD / prior_SD; ≪ 1 means data identifies C, ≈ 1
means likelihood is flat in C.)

**Both leaves leave C contraction at ≈1.** Direct_q is 0.7-0.8 percentage
points tighter — well within sampler noise; not a meaningful improvement.

β posterior contractions on direct_q (mean across groups): pres 1.008,
abs 1.034 — same ≈1 pattern as in the three_state baseline. β posterior ≈
β prior; data doesn't move it. Same diagnosis as the asymmetric-prior
sweep findings.

## Results — per-indicator PPC chi² (mean across indicators)

`per_indicator_ppc_direct_q.py` (mirrors the asymmetric-prior-sweep PPC
script but uses the direct_q `tilde_q → ordered probit` leaf — no
z-mixture).

| leaf | Human | Chicken | LLM | ELIZA | mean | cov94 |
|---|---|---|---|---|---|---|
| three_state (baseline) | 7.63 | 7.08 | 9.21 | 5.85 | 7.44 | 0.889 |
| **direct_q** | **6.85** | **7.36** | **9.42** | **5.44** | **7.27** | **0.879** |

Aggregate chi² is essentially the same (7.27 vs 7.44; ~2% lower).
Direct_q is slightly better on Human and ELIZA, slightly worse on Chicken
and LLM — within sampling/draw-subset noise. The PPC quality is
qualitatively unchanged: as in the asymmetric-prior sweep, the model's
predicted rating distributions look wrong across the board, just in a
slightly different way.

## Read of the experiment

1. **Analytical and empirical disagree.** The three-state binomial-2
   compression looked like it was dropping ~2/3 of the per-rating Fisher
   information for C. Removing it recovers that information *in the
   analytical leaf adapter*, but the actual posterior contraction barely
   budges (<1 point). This means the binding constraint at this prior +
   design is *not* the discrete-z compression. Branches 2 and beyond
   need to be evaluated against this empirical baseline, not the
   analytical ceiling.

2. **The β posterior is still ≈ β prior.** Both leaves leave β posterior
   contraction at ~1, so neither leaf is feeding back useful per-rating
   evidence to the q̃ propagation. This is independent confirmation of
   the asymmetric-prior-sweep diagnosis — the data isn't updating the
   tree priors regardless of leaf choice.

3. **Branch 3 (mixture, Pro_D) is still the right next step,** but the
   prior expectation has changed: it now needs to demonstrate empirical
   contraction improvement, not just an analytical ceiling. If it also
   fails to move contraction at the centre prior, the structural
   problem is upstream of the leaf — likely:
     - design (rater count per indicator too small),
     - β-prior strength (centre prior already tight enough that it
       dominates the per-edge posterior),
     - or label-pool sharing (within-pool β common to many indicators
       smoothing across systems).

4. **Cross-branch corroboration (added post-hoc).** Branch 3
   (`leaf-mixture-shared-h`) ran in parallel and reported Chicken c_only
   (oracle-C ladder, all nuisance pinned) contraction = **1.16 under
   mixture vs 0.87 under three_state** — the recommended production
   leaf is *worse* than three_state at the rung where the leaf has its
   best chance to shine. Combined with this branch's result, the
   evidence is consistent: **no leaf change tested so far meaningfully
   improves C identification.** The binding constraint is upstream of
   the leaf — most likely design (rater count) or β-prior strength /
   label-pool sharing.

5. **Implementation is clean.** No divergences, max R̂ = 1.000,
   ESS_bulk > 5000. The dispatcher pattern keeps three_state and
   direct_q both selectable at runtime via a single CLI flag; binary
   was already present.

## Reproducibility

- Branch `leaf-direct-q` off `asymmetric-beta-prior-sweep` (commit
  `929063b`). Worktree at
  `/Users/ryankelly/ryan-code/research/dcm-code-leaf-direct-q/`.
- Driver: `notebooks/synthetic_validation_2026-05-06/gwt_full_exact_recovery.py
  --state-model direct_q --seed 20260506
  --runs-dir notebooks/leaf_direct_q_2026-05-10/runs
  --beta-pres-mean 0.90 --beta-abs-mean 0.10 --beta-override-sigma 0.30
  --overwrite`
- Diagnostics: `notebooks/leaf_direct_q_2026-05-10/info_diagnostics.py`
  (copied from Branch 1's working tree),
  `notebooks/leaf_direct_q_2026-05-10/per_indicator_ppc_direct_q.py`
  (direct_q-aware PPC),
  `notebooks/asymmetric_prior_sweep_2026-05-10/posterior_vs_prior.py`
  (reused unchanged).
- Sanity: `notebooks/leaf_direct_q_2026-05-10/sanity_a_to_zero.py`.
- Run dir:
  `notebooks/leaf_direct_q_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__smdirect_q__pres90_abs10_sig30/`.

## Implementation note (for future branches)

The leaf-aware PPC script (`per_indicator_ppc_direct_q.py`) currently
forks `per_indicator_ppc.py`. When Branch 3 (mixture) lands, fold both
into a single state-model-aware PPC script keyed on
`config.INDICATOR_STATE_MODEL`. `info_diagnostics.py` already accepts
`--leaf {three_state, direct_q, mixture}` so its analytical comparators
are reusable as-is.
