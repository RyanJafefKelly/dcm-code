# Next directions after the asymmetric β prior sweep

## Diagnosis (and how much weight to put on the sweep)

**The asymmetric β prior change is *probably net good* — but the current model
has other issues that are dominating the results, so don't lean on cross-prior
comparisons.** Concretely:

- *Structural win:* analytical prior-implied gap `q̄_Human − q̄_ELIZA` at depth 3
  goes from 0.038 (paper-mu baseline) to 0.51 (90/10/sig30) to 0.73
  (95/05/sig30). The C signal *can* now reach the leaves through the prior
  alone — see `eval/analytical_qj_by_prior_depth.csv` and `figs/qj_vs_depth.png`.
- *Posterior diagnostic:* β posteriors barely move from the prior centre under
  the override (mean contraction ≈ 1 across all groups, in every fit).
  Posterior C contraction also ≈ 1 on the synthetic data — the rating
  likelihood is **nearly flat in C**.
- *Real-data effect:* C posteriors do shift on real data, but the shift makes
  LLMs and chickens essentially indistinguishable (both ≈ 0.26). Defensible
  direction or not is a research call, but it's a clear consequence of the
  prior change.
- *PPC rating distributions are bad across all priors and the baseline.* See
  `figs/ppc_rating_dist_synthetic.png` and `figs/ppc_rating_dist_real_data.png`.
  The `q_j → rating` conversion is the bottleneck — fix that first, then
  re-evaluate prior choices once the leaf model is doing its job.

**Project-level constraint Ryan flagged on 2026-05-10:** future data will be
Likert 1-7 ratings, so any "drop ordinal cutpoints / Beta likelihood on raw
ratings" path is out of scope. Stick with categorical/ordinal output.

**Cross-stance pooling on C** is also deprioritised for now (multi-week, will
be picked up separately).

## Order of operations

| order | branch | concept |
|---|---|---|
| 1 | `oracle-nuisance-c-only` (D) | Pin `(a, κ, σ_e, b_e)` at synthetic truth. If C contracts, cutpoint flexibility was eating signal. If not, the q_j → rating conversion is the problem. |
| 2 | `c-identifiability-sample-size` (A) | Generate 2x/5x/10x rater synthetic data and refit. Plot C posterior SD vs N. Tells you whether the rater budget alone is the issue, or whether the structural ceiling is the bottleneck. |
| 3 | `continuous-z-leaf` (B, **most important** — Ryan's domain focus) | Replace the 3-state Beta-Binomial leaf with a continuous z (logit-Normal or K-state with K > 2). Several specific options below. |
| 4 | `soft-anchor-defensibility` (E) | Re-run the centre prior on real data with `SOFT_REFERENCE_ANCHORS={"Human":(50,1),"ELIZA":(1,50)}` instead of hard 0.999/0.001. |

D and A are cheap diagnostics that *triage* B. Even if you skip them, run B
— the q_j→rating compression is the most plausible suspect.

E is independent and can run any time.

---

## D — Oracle nuisance (synthetic, c-only)

**Branch:** `oracle-nuisance-c-only`

**Method:** add a `--clamp-obs` flag to
`notebooks/synthetic_validation_2026-05-06/gwt_full_exact_recovery.py`. When
set:

1. Read `truth.json` for the run (already written).
2. Pass the truth `a` and `kappa` (and `sigma_by_expert` / `b_by_expert` if
   present) into the model as `pm.Deterministic` constants instead of
   sampling them. Cleanest implementation: a `CLAMP_OBSERVATION_PARAMS:
   Optional[Dict[str, Any]] = None` field on `ModelConfig`, consumed inside
   `build_ordinal_observation_layer` (`dcm_model.py:???`) — if non-None,
   short-circuit the prior construction and emit deterministic constants.
3. Run with the centre β prior:
   `--beta-pres-mean 0.90 --beta-abs-mean 0.10 --beta-override-sigma 0.30
   --clamp-obs`.

**Expected outcome — diagnostic table:**

| C posterior contraction (post SD / prior SD) | Interpretation |
|---|---|
| `> 1.5` | Cutpoints + scale were absorbing per-system signal that should have gone to C. Continuous-z (B) likely amplifies this. |
| `1.0 – 1.5` | Modest — nuisance was a partial culprit but not the main issue. B is the bigger lever. |
| `≈ 1.0` | Likelihood is genuinely flat in C even with oracle nuisance. The q_j → rating conversion is the bottleneck. Go straight to B. |

**Effort:** ~1-2h to wire `CLAMP_OBSERVATION_PARAMS`, smoke + run, write up.

---

## A — Sample-size sweep (synthetic, identification ceiling)

**Branch:** `c-identifiability-sample-size`

**Method:** add `--rater-multiplier K` to
`notebooks/synthetic_validation_2026-05-06/gwt_full_exact_recovery.py`. In
`simulate_observations_in_place` (`gwt_full_exact_recovery.py:143`), repeat
each rater K times when generating synthetic ratings. Run K ∈ {1, 2, 5, 10}
on seed 20260506 with the centre β prior. Plot Chicken / LLM C posterior SD
and signed bias vs K.

**Expected outcome:**
- SD shrinks ~`1/√K` → "we just need more data"; flag at the project level.
- SD asymptotes well above 0 → structural identification ceiling. Can't be
  fixed by more raters; only by changing the q_j → rating layer (B).

**Effort:** ~1-2h to wire the multiplier + 4 fits at ~15min each + plot.

---

## B — Continuous z / better q_j → rating mapping

**Branch:** `continuous-z-leaf`

**This is the centre of mass for fixing the model.** The current
indicator-leaf path is:

```
q_j  →  m_j ∈ {0, ½, 1}  →  latent = a·m_j + b_e  →  rating ∈ {0..6}
        (Beta-Bin(2,β))     (deterministic)         (OrderedProbit(latent, κ))
```

Three sources of compression:

1. **3-state collapse** — q_j ∈ [0, 1] reduces to one of 3 latent values.
2. **Three latent points** smeared by ordered-probit noise into 7 categories.
3. **Shared cutpoints κ** across systems (when `USE_HIERARCHICAL_EXPERT_CUTPOINTS=False`).

The end-effect on C identification: *many distinct (β, C) configurations
produce indistinguishable rating distributions because they all collapse to
one of the same 3 m_j values per indicator*. That's why the C-likelihood is
flat.

**Four candidate mappings to test, in increasing structural change.** Pick
**B.1 first** as a smoke test (smallest change, biggest leverage); fall back
to B.2 / B.3 / B.4 if B.1 doesn't tighten C.

### B.1 — Bypass z entirely (simplest)

Drop the latent z layer; let q_j parameterise the ordered-probit latent
directly:

```
latent = a · q_j + b_e
rating | q_j ~ OrderedProbit(latent, κ)
```

- Removes the 3-state compression.
- No new RVs.
- Requires writing one new `INDICATOR_STATE_MODEL = "direct_q"` branch in the
  model builder (where `pt_three_state_ll_terms` would go).
- Cheapest test: single config flag, no new hyperparameters.

### B.2 — Logit-Normal continuous z (Ryan's stated instinct)

```
z_j ~ Normal(logit(q_j), τ),  m_j = sigmoid(z_j)
latent = a · m_j + b_e
rating | z_j ~ OrderedProbit(latent, κ)
```

- One new global hyperparameter τ (or labelled).
- Requires Gauss-Hermite quadrature over z when computing the per-indicator
  marginal log-likelihood (since z is now continuous and indicators have
  multiple ratings — analytical marginalisation no longer closed-form).
- 5-10 quadrature points should be plenty given the smoothness.

### B.3 — K-state Beta-Binomial generalisation

The cleanest extension of the existing 3-state code:

```
m_j ~ BetaBinomial(K, β_pres, β_abs) / K   for some K > 2 (try K=6)
latent = a · m_j + b_e
rating | m_j ~ OrderedProbit(latent, κ)
```

- Same `pt_three_state_ll_terms` machinery generalises to K+1 latent states.
- No new continuous parameters; just a discretisation refinement.
- K=6 gives 7 latent states matching the 7 rating categories — *the latent
  state count finally matches the output cardinality.*
- Drop-in compatibility with the rest of the tree (β_pres, β_abs unchanged
  in interpretation).

### B.4 — Mixture-of-rating-distributions

```
rating | q_j ~ q_j · OrderedProbit(α_high, κ) + (1-q_j) · OrderedProbit(α_low, κ)
```

- Mixture of two ordered-probits weighted directly by q_j.
- Most flexible: handles bimodal observed distributions (cluster at 0/6 + a
  middle distribution) cleanly — and the real-data PPCs *do* show bimodality
  the current model misses.
- Two new global parameters α_high, α_low (per system if needed).

### Comparison protocol for B

For each of B.1–B.4, on synthetic seed 20260506 with the centre β prior:

1. Posterior contraction on free C (Chicken, LLM): `post_sd / prior_sd`.
   Target: contraction < 0.5 (truly informative).
2. Per-indicator chi² and KL vs observed (use existing
   `per_indicator_ppc.py`).
3. PSIS-LOO cross-comparison vs the 3-state baseline (use existing
   `per_system_loo.py`).
4. Refit on real data with the best variant; compare Human / ELIZA /
   Chicken / LLM posteriors to the production result.

---

## E — Soft anchors (defensibility)

**Branch:** `soft-anchor-defensibility`

**Method:** rerun the centre-prior real-data fit with
`SOFT_REFERENCE_ANCHORS={"Human": (50, 1), "ELIZA": (1, 50)}` (already
plumbed via `dcm_model.py:181` and
`dcm_model_exact_tree.py:69-89`). Compare Chicken / LLM C posterior medians
and intervals to the hard-anchor real-data fit (this branch's
`runs/real_data/pres90_abs10_sig30/`).

**Expected outcome:** Human / ELIZA stay near 1 / 0 with slight inward
drift; Chicken / LLM should match the hard-anchor case unless anchors were
leaking into the free systems. Useful for the methods defence.

**Effort:** trivial — single dict in the config, one ~25-min fit.

---

## Critical files for any branch

All branches start from `asymmetric-beta-prior-sweep` (which has the
override infrastructure already wired). Critical files:

- `dcm_model.py` — `ModelConfig`, `EvidenceProcessor`,
  `build_label_pool_hyperparameters`, `build_ordinal_observation_layer`,
  `pt_three_state_ll_terms`.
- `dcm_model_exact_tree.py` — `MultiSystemExactTreeBuilder`,
  `_collect_indicator_leaf_lls`.
- `notebooks/synthetic_validation_2026-05-06/gwt_full_exact_recovery.py` —
  synthetic recovery driver.
- `notebooks/asymmetric_prior_sweep_2026-05-10/run_real_data_with_override.py` —
  real-data refit driver.
- `notebooks/asymmetric_prior_sweep_2026-05-10/per_indicator_ppc.py`,
  `posterior_vs_prior.py`, `posterior_predictive_rating_dist.py` — eval
  modules; reuse across branches.
