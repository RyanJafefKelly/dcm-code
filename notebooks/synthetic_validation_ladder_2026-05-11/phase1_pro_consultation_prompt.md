# Phase 1 Pro consultation: root-evidence ladder for the binary-root DCM

## What I'm asking you to do

I'm validating a Bayesian Digital Consciousness Model (DCM). I just completed Phase 0, which settled the model's estimand semantics. Phase 1 should answer one specific gating question before I commit to a broader synthetic-validation sweep:

> **Under the current binary-root DCM, why does production-style synthetic data fail to produce correctly signed, decisive root evidence for realised $R_s = 1$ free systems?**

I want from you:

1. A concrete experimental protocol for **Phase 1A** (an "oracle evidence audit" using existing synthetic data — no refit needed).
2. A concrete experimental protocol for **Phase 1B** (the "root-evidence ladder" with 5–7 rungs, varying one source of uncertainty at a time).
3. The pass/fail thresholds for each rung — the gates that decide whether I escalate to broader validation or pause to fix something specific.
4. A judgement on whether the four candidate hypotheses I list below are the right framing, ranked correctly, or need to be replaced.

You won't have repo access. Everything you need is below. Some of what I write may be subtly wrong — please push back on premises if you spot issues rather than building on them.

---

## Project context (one paragraph)

The DCM was developed at Rethink Priorities to formalise Bayesian aggregation of expert opinion about whether a given system is conscious under each of several theoretical stances on consciousness (Global Workspace Theory, Higher-Order Thought, etc.). Each stance defines a hand-coded tree where internal nodes are features / subfeatures and leaves are operationally observable indicators that experts rate. The published work (Shiller 2026) reports per-stance, per-system $C_s$ posteriors as the headline output. Production data covers four target systems on the GWT stance: Human (anchored at $C_s \approx 0.999$), ELIZA (anchored at $\approx 0.001$), Chicken (free), 2024 chat LLMs (free) — 50, 50, 93, 186 expert ratings respectively, on a 7-point Likert scale, from 6 raters of whom only one (Rater_B) crosses systems.

## The model in concrete terms

For each system $s$ and stance $\sigma$:

$$
\pi_s \sim \mathrm{Beta}(1, 5),
\qquad
R_s \mid \pi_s \sim \mathrm{Bernoulli}(\pi_s),
$$

where $R_s \in \{0, 1\}$ is the latent binary "root state" — *system $s$ is conscious under stance $\sigma$*. The published headline parameter $C_s$ is what I call $\pi_s$ here — a hyperparameter, not a per-system trait.

Below the root, the GWT tree has depth 3 (root $C \to \sim 6$ top-level features $\to$ mixed-fanout subfeatures $\to \sim 30$ indicator leaves). Every edge $v \to u$ has two transmission probabilities $\beta^{\mathrm{pres}}_u = \Pr(z_u = 1 \mid z_v = 1)$ and $\beta^{\mathrm{abs}}_u = \Pr(z_u = 1 \mid z_v = 0)$, with Beta priors whose means are set by hand-assigned (support, demandingness) labels. The published priors give relatively narrow gaps $\beta^{\mathrm{pres}} - \beta^{\mathrm{abs}}$ for many edges (means around $0.6$ vs $0.4$). An asymmetric override exists (`BETA_PRES_OVERRIDE_MEAN = 0.90`, `BETA_ABS_OVERRIDE_MEAN = 0.10`) which fixes the structural transmission gap but does not move empirical $\rho_s$ recovery (more below).

The leaf observation layer is a 7-category ordered probit: $s_{ej} = b_e + a \cdot z_j + \varepsilon_{ej}$, $\varepsilon \sim \mathcal{N}(0, 1)$, with shared cutpoints $\boldsymbol{\kappa} \in \mathbb{R}^6$. Rater shifts $b_e \sim \mathcal{N}(0, 2)$ with one rater anchored at 0. The leaf latent state is binary or three-state (production: three-state, $m_j \sim \mathrm{Binomial}(2, q_j)$, emission centres at $\eta \in \{0, a/2, a\} + b_e$).

Discrete latents are marginalised analytically by a sum-product / belief-propagation DP over the tree. NUTS samples a continuous joint posterior over $(\pi_s, \boldsymbol{\beta}, a, \boldsymbol{\kappa}, \mathbf{b})$. The top-level marginal likelihood per system is exactly the binary-root mixture:

$$
p(y_s \mid \pi_s, \theta) = (1 - \pi_s) L_0(y_s; \theta) + \pi_s L_1(y_s; \theta).
$$

## The synthetic generator

For each system, the generator (in `gwt_exact_unpooled_synthetic_smoke.py:152`):

```python
def sample_latent_tree_for_system(rng, stance_data, edge_betas, true_c):
    root_z = int(rng.binomial(1, true_c))         # ← single Bernoulli draw
    # Then propagation through internal nodes:
    z = int(rng.binomial(1, beta))
    indicator_m[key] = int(rng.binomial(2, beta))  # leaves
```

The generator samples one realised $R_s$ per system from $\mathrm{Bernoulli}(C_s^{\text{true}})$. Synthetic "truths" used in the M-closed pilot:
Human $C^* = 0.999$, Chicken $C^* = 0.25$, LLMs $C^* = 0.10$, ELIZA $C^* = 0.001$. The realised $R_s$ is stored in `truth_payload["latent_by_system"][system]["root_z"]` for every seed. This is critical: with $\pi^* \in \{0.10, 0.25\}$ the generator produces relatively *few* realised $R = 1$ free-target cases, which itself biases the validation set.

## Phase 0 outcome: Path A confirmed (with a problem)

Phase 0 confirmed the binary-root semantics and reframed the headline output to:

$$
\rho_s = p(R_s = 1 \mid y_s)
=
\mathbb{E}_{(\pi_s, \theta) \sim \mathrm{posterior}}
\left[
\sigma\bigl(\mathrm{logit}(\pi_s) + \log B_s(\theta)\bigr)
\right],
\qquad
B_s = L_1 / L_0.
$$

A "collapsed" variant integrates the root prior odds analytically per draw:

$$
\rho_s^{\mathrm{collapsed}} = \mathbb{E}_{\theta \sim \mathrm{posterior}}\left[\sigma\bigl(\log(a/b) + \log B_s(\theta)\bigr)\right]
$$

with $a/b = 1/5$ for the default Beta(1, 5) prior. Phase 0 audit shows $\rho^{\mathrm{collapsed}}$ and $\rho^{\mathrm{sampled-}\pi}$ agree to mean abs difference 0.001 in practice; collapsed is the cleaner Path A headline because it matches the integrated binary-root estimand.

Production deterministics (`{prefix}_log_L_root0`, `{prefix}_log_L_root1`, `{prefix}_log_B`, `{prefix}_rho`) were added to `MultiSystemExactTreeBuilder._exact_tree_log_likelihood`. Every future fit exposes $\rho_s$ as a first-class output.

A binary-root toy at $J \in \{4, 16, 64\}$ leaves and $K \in \{1, 5, 20\}$ ratings per leaf, with truth $\pi_s \sim \mathrm{Beta}(1, 5)$ and $\beta^{\mathrm{pres}} = 0.9$ / $\beta^{\mathrm{abs}} = 0.1$, $\varepsilon = 0.05$ Bernoulli emission noise, recovers $\rho_s$ cleanly:

| $J$ | $K$ | median $\rho \mid R=1$ | median $\rho \mid R=0$ | mean log-score improvement (nats) |
|---|---|---|---|---|
| 4 | 1 | 0.883 | 0.000 | 0.339 |
| 4 | 5 | 0.999 | 0.000 | 0.355 |
| 16 | 1 | 1.000 | 0.000 | 0.440 |
| 64 | 20 | 1.000 | 0.000 | 0.416 |

And $\pi_s$ posterior contraction always hits the predicted one-update ceilings (0.878 for $R = 0$, 1.134 for $R = 1$). So the binary-root toy + ρ_s machinery work as expected.

**But a post-hoc reanalysis of 13 existing production-style synthetic fits (M-closed pilot + sample-size sweep + asymmetric β prior sweep, total 26 free-target system × seed combinations) against realised $R_s$ truth shows the production-style model is not doing the same job:**

| $R_{\text{true}}$ | $n$ | mean $\rho^{\mathrm{collapsed}}$ | median $\rho$ | mean $\log B$ | median $\log B$ | mean Brier |
|---|---|---|---|---|---|---|
| 0 | 16 | 0.111 | 0.117 | −1.65 | −0.57 | 0.018 |
| 1 | 10 | 0.315 | 0.369 | −1.02 | +0.99 | 0.500 |

Classification at $\rho > 0.5$: **TPR = 0.0**, TNR = 1.0, balanced accuracy = 0.5. **Mean log Bayes factor for realised $R_s = 1$ free targets is *negative*** (median is positive but small; the distribution is bimodal under $R = 1$).

Evidence-category breakdown (with prior-odds thresholds: $\log B > +1.61$ for $\rho > 0.5$, $\log B < -1.34$ for $\rho < 0.05$):

- **0** strong-present
- **0** moderate-present
- **10** ambiguous (most $R_s = 1$ cases land here)
- **10** moderate-absent
- **6** strong-absent

So the production-style synthetic model is **conservative**: it almost never produces false positives, but it *also misses every single realised $R_s = 1$ free target* in the reanalysed set. **This is the Phase 1 puzzle.** Phase 0 nailed the estimand; Phase 1 needs to nail down why production-style synthetic data fails to produce correctly signed root evidence.

## Four candidate hypotheses (please rank or replace)

1. **Tree transmission gap is too weak at production depth (3) under published priors.** Realised $R_s = 1$ might not produce visibly different leaf marginal probabilities $q_j$ from $R_s = 0$, so the data has no signal to separate them. The asymmetric $(0.90, 0.10)$ prior partially addresses this structurally but didn't fix empirical $\rho$ recovery in the prior sweep.
2. **Posterior uncertainty over $(\boldsymbol{\beta}, a, \boldsymbol{\kappa}, \mathbf{b})$ washes out root evidence.** Even if the oracle $\log B$ at true parameters is strong, marginalising over the joint posterior of nuisance parameters dilutes the signal.
3. **Three-state / ordinal observation layer discards information about $R_s$.** The per-indicator $m_j \in \{0, 1, 2\}$ marginalisation collapses leaf-state information that would otherwise distinguish root regimes; the 7-cutpoint ordered probit family adds further smoothing.
4. **The current rater design (5/6 raters single-system, only one cross-system) is the binding constraint.** Per-system rating-distribution differences get absorbed by per-rater shifts $b_e$, leaving little signal for $R_s$. (Note: this should bite hard on *real* data but might not for M-closed synthetic where the generator matches the model.)

My prior is that (1) and (2) are most likely the binding constraints, (3) is probably second-order, and (4) bites hard for real data but matters less for M-closed synthetic. But my evidence for this ranking is weak. The bimodal $\log B$ distribution under $R = 1$ (mean −1.02, median +0.99) is itself diagnostic — a substantial fraction of $R = 1$ seeds genuinely produce strong positive evidence ($\log B > 1$), but another fraction produces strong *negative* evidence — what does this suggest?

## Phase 1 structure I'm proposing (lifted from your prior turn)

### Phase 1A: oracle evidence audit on existing synthetic data

**Goal:** Compute $\log B_s$ at the *true* synthetic parameters and compare with the posterior $\log B_s$ from saved fits, on the same 26 free-target system × seed combinations from Phase 0.

For each (synthetic seed, system):

```text
log_B_oracle              # log L(y_s | R=1, β*, a*, κ*, b*) − log L(y_s | R=0, β*, a*, κ*, b*)
log_B_posterior_mean      # mean over saved posterior draws of log B from fitted parameters
log_B_posterior_median
rho_oracle                = sigmoid(logit(π_true) + log_B_oracle)
rho_posterior             = mean over draws of sigmoid(logit(π_d) + log_B_d)
```

Interpretation matrix (your turn 3):

| Result | Diagnosis |
|---|---|
| oracle $\log B$ weak for $R = 1$ | data/tree/DGP itself often lacks root evidence — even infinite Bayesian inference can't recover |
| oracle $\log B$ strong, posterior $\log B$ weak | inference / nuisance uncertainty washes out evidence — nuisance free parameters absorb $C_s$ signal |
| oracle $\log B$ strong positive, posterior negative | likely implementation, indexing, or model mismatch |
| oracle and posterior both weak | broad fitting sweeps will not fix it; need DGP / observation-layer redesign |

This is a one-day notebook against existing posterior samples and saved synthetic truths. Your earlier turn flagged this as the highest-value next diagnostic — it cleanly separates "data lacks signal" from "inference loses signal."

### Phase 1B: actual-tree ladder with nuisance clamping

**Goal:** Test the four candidate hypotheses by varying one source of uncertainty at a time, on the actual GWT tree.

| Rung | Tree | Observation | $\boldsymbol{\beta}$ | $(a, \boldsymbol{\kappa}, \mathbf{b})$ | Rater design | What it tests |
|---|---|---|---|---|---|---|
| 1 | actual GWT | binary noisy leaf | true | none (binary leaf) | fully crossed | pure tree transmission |
| 2 | actual GWT | binary noisy leaf | inferred | none | fully crossed | $\boldsymbol{\beta}$ identifiability |
| 3 | actual GWT | three-state leaf | true | true (clamped) | fully crossed | leaf-state information loss |
| 4 | actual GWT | ordinal probit ($K = 7$) | true | true (clamped) | fully crossed | ordinal information loss |
| 5 | actual GWT | ordinal probit | inferred | inferred | fully crossed | nuisance uncertainty |
| 6 | actual GWT | ordinal probit | inferred | inferred | production-like (5/6 single-system) | rater design |
| 7 | actual GWT | ordinal probit | inferred | inferred | improved crossed design | survey-design payoff |

**Crucial design choice from Phase 0:** use *balanced synthetic roots*, not the original $\pi^* \in \{0.10, 0.25\}$ which produced few realised $R = 1$ cases (10/26 in the post-hoc reanalysis). Generate $n_{\text{rep}}$ datasets each with $R_s = 0$ and with $R_s = 1$ set deterministically.

Per-rung metrics:

```text
median log_B given R=1
median log_B given R=0
mean rho^collapsed given R=1
mean rho^collapsed given R=0
Brier improvement over prior (1/6 baseline)
log-score improvement
TPR at rho > 0.5
TNR at rho ≤ 0.5
balanced accuracy
ECE (across replicates within rung, if n_rep ≥ 100)
evidence category counts
```

Pass thresholds (your turn 3):

```text
median log_B | R=1 > +1.61   (clears prior odds threshold for ρ > 0.5)
median log_B | R=0 < −1.34
balanced accuracy > 0.75
positive log-score improvement vs prior
```

## Repo-specific shortcuts the coding agent can use

1. **`composite_vs_exact_diagnostic.py:exact_loglik`** is a verified-to-numerical-precision NumPy implementation of the bottom-up tree DP. Signature: `exact_loglik(stance_data, indicator_obs, leaf_logliks, beta_pres_by_key, beta_abs_by_key, C_by_system) -> (total, ll_per_sys)`. Evaluate it per posterior draw with `C_by_system` clamped to $\{0, 1\}$ to extract $\log L_0$ and $\log L_1$ directly. **Phase 1A reuses this — no refit needed.**

2. **`MultiSystemExactTreeBuilder` in `dcm_model_exact_tree.py:40`** is the production model class. It now exposes `log_L_root0`, `log_L_root1`, `log_B`, `rho` per (system, stance) as PyMC deterministics. Phase 1B refits use this directly.

3. **Synthetic generator (`gwt_exact_unpooled_synthetic_smoke.py`)** drives all production-style synthetic data. Toy variants (different observation model, varying tree topology, varying rater design) can subclass / override `sample_latent_tree_for_system`, `simulate_observations`, and the rater-construction logic. For the toy ladder, modify `sample_latent_tree_for_system` to take `R_s` as input (instead of sampling it from `true_c`) — that's the balanced-root mod.

4. **The verified NumPy DP also works as the *generator* for the toy ladder** if I want a minimal-dependency synthetic-data pipeline (no PyMC / PyTensor compile time). This makes Phase 1B rungs cheap to iterate.

5. **Saved synthetic posterior fits** live at:
   - `notebooks/synthetic_validation_2026-05-06/runs/full_exact_recovery/` (M-closed pilot, 4 seeds × 2 fit families)
   - `notebooks/sample_size_sweep_2026-05-10/runs/` (K ∈ {1, 2, 5, 10, 20})
   - `notebooks/asymmetric_prior_sweep_2026-05-10/runs/synthetic/` (β prior sweep)

   Each is an ArviZ `fit.nc` file plus a `truth_payload.json` with the realised `root_z` per system. Phase 1A reads these directly.

6. **GWT tree topology:** depth 3 (root $C \to \sim 6$ top-level features $\to$ mixed-fanout subfeatures $\to \sim 30$ indicator leaves). The exact tree comes from the API (`https://dcm.rethinkpriorities.org/schemes/133/json`) and is cached at `data_cache.json`.

7. **Phase 0 outputs** (for context on the puzzle): `outputs/phase0_root_semantics/` — `phase0_root_semantics_report.md`, `posthoc_rho_summary.csv`, `posthoc_free_target_classification.csv`, `posthoc_free_target_stratified_metrics.csv`, `binary_root_toy_summary.csv`, `information_precheck.csv`. The `phase0_audit_collapsed_rho_report.md` has finer-grained per-system / per-seed breakdowns.

## Specific asks for your response

1. **Phase 1A protocol.** Concrete enough for a coding agent to implement without further consultation: function signatures, expected output CSV columns, pass thresholds for each interpretation cell of your matrix, and what to report if oracle and posterior $\log B$ disagree by more than X nats. Note that the existing Phase 0 reanalysis was done at posterior median nuisance, not draw-by-draw oracle; Phase 1A wants true-parameter oracle, not posterior median.

2. **Phase 1B detailed specification.** For each of rungs 1–7: tree topology to use (real GWT vs simplified depth-2 toy first?), exact $\boldsymbol{\beta}$ true values (and matching priors for inferred-$\boldsymbol{\beta}$ rungs), number of synthetic replicates per rung, expected wall-clock budget, and the exact pass/fail gates that would let me move to the next rung vs pause to fix.

3. **Hypothesis ranking.** Given Phase 0's pattern (R=1 mean log_B ≈ −1.02 but median +0.99, mean ρ ≈ 0.32, no strong-present cases, bimodal $\log B$ under $R = 1$), which of the four hypotheses is most likely the binding one? Does the bimodality under $R = 1$ implicate any specific hypothesis?

4. **Order of operations.** Some of the Phase 0 puzzle might dissolve under balanced roots (10 R=1 cases out of 26 is small; the bias estimates are noisy). Does balancing the synthetic root sampling come *first* (re-running existing seeds with $R_s$ forced), or does the oracle audit come first?

5. **A scalar signal for "Phase 1B rung passed."** With 7 rungs × multiple metrics, I want one or two scalars per rung that summarise whether to escalate. What's the cleanest summary statistic?

6. **Rung-design specifics:**
   - For rungs 1–4, should I use 1 anchored ref + 1 free system, or all 4 systems with both anchors? The all-4 variant matches production but adds anchor calibration as an extra moving piece.
   - For rung 7, what specifically is the "improved crossed design"? E.g., 3 raters each rating all 4 systems? 1 cross-rater + 5 single-system but with denser cross-rater coverage? What's the experimental sweet spot for "this would resolve the design constraint"?
   - Is a "binary leaf with ordinal-probit emission" rung worth adding between rungs 2 and 3, to isolate the ordinal observation from the three-state leaf?

7. **Anything you'd add or remove from the Phase 1 plan.** If you think a candidate hypothesis is missing, or the ladder ordering is wrong, or some rungs are redundant, say so.

## Constraints

- Hand-off to a coding agent for implementation; protocols need to be concrete (signatures, CSV columns, thresholds, replicate counts, $\boldsymbol{\beta}$ values).
- The existing Phase 0 infrastructure (deterministics, exact_loglik, generator, saved truth_payloads) is reusable. Don't propose new infrastructure unless necessary.
- If you think the four-hypothesis framing is wrong, push back on premises rather than building on them.
- 3–5 page response is fine. Don't pad with general Bayesian-modelling advice; assume working knowledge of HMC, ordered probit, identifiability, scoring rules, and PSIS-LOO.
