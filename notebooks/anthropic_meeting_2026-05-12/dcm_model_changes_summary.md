# DCM model — what changed since the paper, and what we learned

**Author:** Ryan Kelly (SPAR fellow). **Date:** 2026-05-10.
**Audience:** Tuesday 2026-05-12 meeting with Arvo + Anthropic data scientist.
**Repo:** `dcm-code`, branch `asymmetric-beta-prior-sweep` and follow-ups.

---

## Meeting brief (read this first; ~2 pages)

> *Fill on Monday morning, after reviewing overnight Branch 6 + expert-discrimination
> feasibility outputs. Should be a tight executive summary that the Anthropic
> person can absorb in 5 minutes.*

### What changed since the paper (one-line each)
- *(fill: ordinal observation layer; three-state indicator z; label pooling; soft-anchor infrastructure; …)*

### What we learned from synthetic validation
- *(fill: M-closed pilot 2026-05-07; asymmetric β prior structurally fixes q_j transmission; sample-size sweep result; expert-coverage finding; bimodal-mass deficiency)*

### Where we'd value input (the Tuesday discussion)
- *(fill: rater design implications; per-expert κ heterogeneity vs per-system κ; mixture leaf vs continuous z; cross-stance pooling priority)*

---

## 1. Original DCM paper baseline

> *Fill: model as published. Beta-Bernoulli tree with hand-coded support /
> demandingness labels. Bernoulli leaf (z_j ∈ {0, 1} given parent).
> Deterministic ordinal binning of q_j → category. Per-stance fits.
> Reference paper sections / equations. Use $\tilde q_j = \beta^{\rm abs}_j +
> q_j(\beta^{\rm pres}_j - \beta^{\rm abs}_j)$ notation throughout.*

## 2. SPAR-tenure changes shipped to production

> *Each subsection: what / why / what it bought / where it lives in code.*

### 2.1 Ordinal observation layer (replaces deterministic binning)
- **What:** ordered probit with cutpoints κ ∈ ℝ⁶ on a latent location η_je = a · z_j + b_e (when expert shifts active).
- **Why:** *(fill — original deterministic-bin produced compressed posteriors; ordinal-probit lets the rating distribution be inferred rather than hard-coded.)*
- **Lives at:** `dcm_model.py:619` (`build_ordinal_observation_layer`), config flags `N_CATEGORIES=7`, `KAPPA_PRIOR_SIGMA=2.0`, `A_PRIOR_SIGMA=…`.
- **Evidence:** *(fill — fit comparison vs binary, refer to `notebook 17` write-up.)*

### 2.2 Three-state indicator z (replaces Bernoulli z)
- **What:** m_j ~ Binomial(2, β_eff(q_j)) → z_j = m_j/2 ∈ {0, 0.5, 1}; latent location a · z_j.
- **Why:** *(fill — Bernoulli binary z over-compressed q_j ∈ [0, 1] into one of 2 latent rating locations.)*
- **Lives at:** `dcm_model.py:461` (`three_state_log_weights`), `:470` (`pt_three_state_ll_terms`), config `INDICATOR_STATE_MODEL = "three_state"`.

### 2.3 Label pooling (POOL_BETAS_BY_LABEL + BETA_ABS_BY_SUPPORT_DEMAND)
- **What:** edges sharing a (support, demandingness) label group share a single logit-Normal β_pres / β_abs RV centred at the paper's deterministic prior mean. Hyperparameter LABEL_POOL_SIGMA = 0.5 in logit space.
- **Why:** *(fill — paper's per-edge β had unidentifiable singletons; pooling lets data calibrate label-level transmission while preserving the modeller's label semantics.)*
- **Lives at:** `dcm_model.py:1068` (`build_label_pool_hyperparameters`).

### 2.4 Soft-anchor infrastructure (SOFT_REFERENCE_ANCHORS)
- **What:** per-system Beta priors on root C overriding hard fixed values, e.g., Human (50, 1) → 95% CI ≈ [0.93, 0.999].
- **Why:** defensibility — anchors-as-priors not anchors-as-constants.
- **Status:** infrastructure landed; production fit still uses hard anchors. First real-data use was in the exploratory mixture-leaf branch (Branch 3).
- **Lives at:** `dcm_model.py:181`, wired into `MultiSystemExactTreeBuilder` `dcm_model_exact_tree.py:69-89`.

### 2.5 Reference-system anchored joint fit
- **What:** Human C fixed at 0.999, ELIZA C fixed at 0.001, Chicken / LLM C free; shared (a, κ, β tree) across systems.
- **Why:** the paper fits each system separately; joint fit lets cross-system rater calibration anchor the per-system C estimates.
- **Lives at:** `gwt_reference_recovery_analysis.py` (`ANCHORED_SYSTEM_CONFIGS`), `dcm_model_exact_tree.py` (`MultiSystemExactTreeBuilder`).

### 2.6 Other infrastructure (worth flagging, not load-bearing)
- Per-system observation parameters (`task5_recovery_per_system_obs.py`) — *exploratory, not production*.
- Per-expert kappa hierarchy `USE_HIERARCHICAL_EXPERT_CUTPOINTS` — *flag, untested at production scale*.
- Per-expert sigma `USE_EXPERT_SCALES` — *flag, currently False in production*.

## 3. Synthetic validation programme

### 3.1 The 2026-05-07 pilot (M-closed, 4 seeds)
> *Fill: reference `notebooks/synthetic_validation_2026-05-06/findings_2026-05-07.md`.
> Headline: coverage holds 16/16 across (4 seeds × 2 fit models × 2 free systems)
> but Chicken-vs-LLM ordering recovers in only 1/4 seeds. Beta(1, 5) prior on
> root C dominates — uniform-prior fit gives LLM median 0.62 vs 0.16 under default.*

### 3.2 The 2026-05-10 asymmetric β prior sweep
> *Fill: reference `notebooks/asymmetric_prior_sweep_2026-05-10/findings.md`. Headline:*
- Structural prior win: prior-implied q_j(Human) − q_j(ELIZA) at depth 3 goes from 0.038 (paper baseline) to 0.51 (centre prior). The β prior change cleanly transmits C signal where the paper's prior didn't.
- But empirical C posterior contraction stays ≈ 1 across all priors. β posterior ≈ β prior on synthetic data.
- Real-data effect: LLM posterior median shifts from 0.111 → 0.263; Chicken stable at 0.26. New prior amplifies modest mid-range LLM evidence; whether the direction is defensible is a research call.
- Per-system + no-anchors: every C posterior collapses to prior mean. Anchors were carrying per-system distinguishability.

### 3.3 Leaf-model exploration (Branches 2 + 3, exploratory)
> *Fill: reference `notebooks/leaf_direct_q_2026-05-10/findings_direct_q.md` and
> `notebooks/leaf_mixture_2026-05-10/findings_mixture.md`. Headline:*
- Branch 2 (`direct_q`): 3× analytical Fisher / 1.74× I(Q;Y) / 2.85× full-vector KL, but empirical contraction barely moves. **Leaf flexibility alone doesn't translate to empirical C identifiability.**
- Branch 3 (`mixture` with shared $h_{sj}$, centred endpoints): no-clamp synthetic Chicken median 0.110 (vs three_state 0.118). Real-data fit gives Chicken 0.267 > LLM 0.183 (first ordering recovery on real data) but mixture's (δ, κ) joint mode-finds (R̂ = 1.73), so result is suspect.
- The mixture+c_only Chicken=0.227 result is **confounded by a κ-frame mismatch**: c_only pinned κ to three_state's posterior (which was fit for η ∈ {0, a/2, a}); centred mixture η ∈ {−δ/2, +δ/2} is shifted down by a/2; mixture's C is partially compensating for the wrong κ. Clean upper bound is `three_state + c_only` Chicken contraction = 0.87.
- **Conclusion:** leaf change has not been shown to do anything useful. Three_state+c_only proves leaf is not the bottleneck (contraction 0.87 even with all nuisance pinned).

### 3.4 Sample-size sweep (Branch 6, completed 2026-05-11)

> *Source: `notebooks/sample_size_sweep_2026-05-10/findings_sample_size.md`.*

**The result is the strongest negative finding in this validation programme.**
Synthetic seed-06 rerun with rater multiplier $K \in \{1, 2, 5, 10, 20\}$
(replicates each (rater, indicator) slot $K$ times with independent draws
given the latent state) and the centre β prior:

| K | Chicken median | Chicken contraction | LLM median | LLM contraction |
|---|---|---|---|---|
| 1 | *(fill from CSV)* | *(fill)* | *(fill)* | *(fill)* |
| 2 | *(fill)* | *(fill)* | *(fill)* | *(fill)* |
| 5 | *(fill)* | *(fill)* | *(fill)* | *(fill)* |
| 10 | *(fill)* | *(fill)* | *(fill)* | *(fill)* |
| 20 | *(fill)* | 1.01 | *(fill)* | 1.11 |

**Posterior SD on free $C$ asymptotes at the prior SD (~0.141) across all $K$.** Contraction never drops below 0.93. At $K = 20$ — twenty times the current rater count — Chicken contraction is 1.01 and LLM 1.11. Posterior medians plateau near the prior mean (~0.167); they do not approach truth even at $K = 20$.

**Interpretation: shrinkage is asymptotic, not parametric.** A well-identified Bayesian model should give posterior SD shrinking at $\propto 1/\sqrt K$. Here, the SD reaches a floor at the prior SD and refuses to go below it. **This means the rating likelihood is *asymptotically flat in $C$* under the current model — infinite data would not help.** This is a structural identifiability problem, not a data-volume problem.

**Mechanism (consistent with Branches 1-3):** the joint posterior over $(C_s, \beta, a, \kappa, b)$ has a flat marginal in $C_s$ because the other parameters can absorb whatever rating-distribution shift $C_s$ would induce. Per-rating Fisher *at oracle nuisance* is non-zero (~0.12 nats² per rating, see §3.1), but the *marginal* information about $C_s$ after marginalising over the other parameters' posteriors is asymptotically zero.

**Diagnostics caveat:** $K \in \{5, 10, 20\}$ show max R̂ = 1.53 / min ESS = 7 on a label-pool β-tilde *singleton* parameter (the `weak undermining + neutral` group with one edge). C-only diagnostics stay healthy across all $K$ (R̂ ≤ 1.05, ESS ≥ 55), so the SD plateau on $C$ is reliably estimated. Flagged in the findings doc.

**Implication for the project:** the path forward cannot be a survey-design change alone. The model as currently structured needs to change — and the diagnostic-build-up programme (§9) is the next step.

### 3.5 Expert-discrimination feasibility (Arvo's specific request)

> *Source: `notebooks/expert_discrimination_feasibility_2026-05-10/findings.md`.*

- **Diagonal Fisher at oracle nuisance** (i.e. $a$, $\kappa$, $\alpha_j$, $\delta_j$, $C_s$ pinned to truth, asking only "what does $d_e$'s own log-likelihood curvature look like at $d_e = 1$?"): single-system experts in the realistic design get **90-115 nats²**; Rater_B (cross-system, ~3× more ratings) gets **~396 nats²**. All far above the ~1 nat² collapse threshold. Idealised-design ratings-per-expert (200 vs 45) would scale this further but the realistic *diagonal* identifiability is already healthy.
- **Caveat — joint identifiability not yet checked.** The diagonal Fisher fixes the global discrimination $a$ at truth and asks about $d_e$ alone. The actually-stricter question is whether $d_e$ and $a$ are *jointly* identifiable: for any single expert the predictive distribution depends on the product $a \cdot d_e$, and within-system contrasts between raters are what disentangle the two factors. Per-system rater counts: LLM 4 raters, Chicken 2, Human and ELIZA only Rater_B. The $a \cdot d_e$ direction has no within-system anchor on Human / ELIZA, only cross-system via Rater_B.
- **Practical implication.** Per-expert $d_e$ is *probably* recoverable on this design *if* implemented with strong hierarchical shrinkage centred at $d_e = 1$ (so the model only takes the per-expert deviation when within-system data demands it). Without shrinkage, joint modes along the $a \cdot d_e$ confound axis could be wide on Human / ELIZA. The synthetic joint-fit verification is recommended as the next step on this thread.

## 4. Two structural findings worth surfacing

### 4.1 Single-system-expert design constraint
- **Fact:** of 6 experts, 5 rate exactly one system (3 LLM-only, 2 Chicken-only). Rater_B is the only cross-system rater (LLM, ELIZA, Human).
- **Implication:** any per-expert or per-system parameter (κ, $d_e$, η-shift) absorbs per-system rating-distribution differences that the model would otherwise route to $C_s$. This is the structural reason fit-5's per-system + no-anchors collapsed every C to the prior mean.
- **Defensibility:** **this is a survey-design constraint, not a modelling failure.** No within-system modelling lever can tighten C posteriors crisply at this design.

### 4.2 Bimodal-mass deficiency at extreme $\tilde q$
- **Fact:** at baseline-fit $\tilde q \in [0.6, 1.0]$ on real data, observed ratings concentrate 85-88% at category 6. Both three_state and mixture leaves predict only 28-45% at cat 6 — off by 40-50 percentage points. The leaf change does not fix this.
- **Mechanism:** the OrderedProbit observation layer assumes a smooth Gaussian latent error around a single location $\eta$, with shared cutpoints κ. Under this family, the predictive distribution on category $k$ is determined by where $\eta$ sits relative to the cutpoints — a single $\eta$ can put mass at one or two adjacent categories, but cannot place mass *both* at the extremes (cat 0 and cat 6) simultaneously, nor make a single category as concentrated as 85% without pushing the location well past the outermost cutpoint. With shared κ across the $\tilde q$ range, the cutpoint set has to compromise between mid-range distributions (need spread) and extreme-range distributions (need concentration).
- **This is a known limitation of the ordered-probit family.** It applies to *any* OrderedProbit observation layer, including the one added during SPAR. The choice was principled — ordered probit is the standard psychometric latent-variable model for Likert ratings — and inheriting its smoothness assumption is the trade-off you make for identifiability and interpretability of cutpoints. The empirical finding here is that *for this data*, the smoothness assumption is too restrictive at the extremes.
- **Candidate fixes** (not yet tested; ordered by surgical-ness):
  - **Heavier-tailed latent error** (Student-$t$ with low ν instead of Normal in the cumulative density) — same κ family but more mass at the extremes per location shift. Single new ν parameter. Surgical.
  - **Per-expert κ heterogeneity** (`USE_HIERARCHICAL_EXPERT_CUTPOINTS`, already a config flag) — different experts use the scale differently; with 6 experts and only Rater_B as cross-system anchor this might be partial but still useful.
  - **Two-level mixture observation** — augment the leaf with a "scale-use" latent class per expert (extreme-user vs middle-user) that controls predictive concentration independently of $\eta$.
  - **Dirichlet-Multinomial direct categorical likelihood** — most flexible, drops κ entirely; loses the latent-variable interpretation but preserves ordering via a monotonicity constraint on category probabilities given $\tilde q$.
  - **Per-system κ** — most expressive but absorbs the very signal $C_s$ should explain; flagged in §5 as the central design tension.
- *(See plot in `notebooks/leaf_mixture_2026-05-10/figs/ppc_by_qbin.png` — fill in once doc is finalised.)*

### 4.3 The per-system-parameter design tension (open question for the meeting)

**The core problem.** Per-system parameters — whether per-system $\kappa$, per-system $a$, per-system endpoint shifts, or per-expert anything when experts are mostly single-system — *absorb per-system rating-distribution differences*. But $C_s$ is also a per-system parameter encoding "how high does this system's rating distribution tend to be". So per-system observation parameters and $C_s$ are co-identified up to a *trade*: one absorbs what the other could have explained. Empirical evidence: fit 5 (per-system kappa, no anchors) collapses every $C$ posterior to the prior mean.

**Three principled mitigations** (none of which solves it cleanly at the current rater design):

1. **Strong-shrinkage hierarchical priors.** Allow per-system parameters but with a tight hyperprior centred on the global value:
   $$ \kappa_s \sim \text{Normal}(\kappa_{\rm global}, \sigma_\kappa), \quad \sigma_\kappa \text{ small (e.g. } 0.1) $$
   The data has to overcome the prior to drive system-specific divergence. Limits absorption proportional to $\sigma_\kappa$. **Pro recommended this for $d_e$ (expert discrimination) — same logic generalises.** Trade-off: tight prior → may miss real per-system differences; loose prior → absorbs C signal.

2. **Constrain the *form* of per-system variation.** Allow per-system variation only in dimensions orthogonal to the $C_s \to q_j \to \eta$ path. Concretely: per-system *scale* parameter (how spread out a system's predictive distribution is, modulating the latent error variance) but *not* per-system *location* shift. $C_s$ already controls location; if per-system flexibility is restricted to scale, the two are not co-identified.
   $$ y_{sje} \sim \text{OrderedProbit}\left(\eta_{sje}, \kappa, \sigma_s\right), \quad \log \sigma_s \sim \text{Normal}(0, 0.3) $$
   This is mechanically similar to per-expert discrimination $d_e$ from Pro's recommendation, but lifted to the system level instead of the expert level. Pure scale-only deviation is identifiable independently of $C_s$ even with single-system raters per system — the *shape* of the per-system distribution constrains $\sigma_s$ even when the *centre* is correlated with $C_s$.

3. **Cross-system rater anchoring.** With more cross-system raters (currently only Rater_B covers Human/LLM/ELIZA, no one covers Chicken cross-system), per-expert $d_e$ or per-expert κ deviation from the cross-system anchor becomes identifiable because the cross-system rater's calibration pins the global scale, and single-system raters' deviations are calibrated *relative to that anchor*. **This is a survey-design lever, not a modelling one.** The current design has no cross-system anchor for Chicken — that's the binding constraint.

**Composite proposal for next-survey-wave + future modelling.**
- *Survey side:* aim for ≥ 3 cross-system raters who each rate ≥ 2 systems with overlap. Doesn't have to be all-systems-by-all-raters; even a partial crossover design (e.g., "every system has at least 2 raters who also rate Human") would give the cross-system anchor that's currently load-bearing on a single person.
- *Modelling side:* prefer scale-only per-system flexibility (option 2 above) over per-system cutpoints. Combine with strong-shrinkage hierarchical priors on per-expert discrimination. The asymmetric β prior or its label-aware refinement (Branch 7, pending Pro review) is orthogonal to this and can be combined.

**Honest caveat.** Even with all three mitigations, $C_s$ identification at the current rater design is fundamentally limited by single-system-expert dominance. The *median* of $C_s$ posterior is recoverable to within ≈ ±0.1 of truth (per the asymmetric prior + mixture leaf experiments); the *spread* is not crisply contracted below the prior. Reporting intervals + medians, not point estimates with implied precision, is the right communication choice for the current data.

## 5. Current state of the model

### 5.1 What's working
- The model compiles, samples cleanly (modulo mixture identifiability issue), produces sensible-looking C posteriors *for hard-anchored systems* (Human, ELIZA recover when anchored).
- The label-pooling structure is well-behaved: tree β posteriors track the modeller's label semantics, Beta(α, β) priors don't blow up at boundaries.
- Synthetic validation infrastructure is robust: M-closed pilot, oracle audits, posterior-vs-prior diagnostics, per-indicator PPC, full info-diagnostic library.
- Reproducibility: every fit lives at `notebooks/<topic>_<date>/runs/<run_id>/` with `config.json`, `truth.json`, `fit.nc`, all CSVs, and a `summary.md`.

### 5.2 What's not working
- **C is asymptotically unidentified at this design (§3.4).** Sample-size sweep at $K \in \{1, 2, 5, 10, 20\}$ shows C posterior SD asymptotes at the prior SD; contraction never drops below 0.93; medians plateau near the prior mean even at $K = 20$. Shrinkage is asymptotic, not parametric — *infinite data would not help under the current model.* This is the strongest negative result in the validation programme.
- C posterior contraction stays ≈ 1 on free systems (Chicken, LLM) for any leaf / prior combination tested. The data has location information but not curvature information about C. Any reported posterior median is a prior-anchored point estimate, not a tightly-identified one.
- PPC rating distributions are qualitatively wrong on real data at the extremes — model under-predicts category-6 mass at high $\tilde q$ by 40-50 pp (§4.2).
- Without the hard reference anchors, the model can't tell systems apart (fit 5 collapsed every C posterior to the prior mean). Anchors are load-bearing; soft anchors (50, 1) / (1, 50) appear to give similar identification to hard anchors but this hasn't been formally compared yet.

### 5.3 Defensibility tally (which changes are "shipped" vs "exploratory")
> *Fill: clean table of every flag / config option, status (shipped to main / exploratory branch / scratch), and what would happen if it were reverted.*

## 6. Open questions for the meeting

1. **Survey design.** Given the 5/6 single-system-expert finding, is it worth flagging up-front that the next survey wave should aim for ≥ 3 cross-system raters? Or some other design change?
2. **OrderedProbit vs Dirichlet-Multinomial.** The bimodal-mass deficiency suggests the cutpoint family is fundamentally too rigid for these rating distributions. How much appetite is there for a more flexible categorical likelihood, given the future Likert-1-7 data constraint?
3. **Per-expert κ heterogeneity** (`USE_HIERARCHICAL_EXPERT_CUTPOINTS`) — would help with bimodal mass, but with only 6 experts the per-expert posterior might be very wide. Any guidance on when this is worth turning on? Related: per-expert discrimination $d_e$ has healthy diagonal Fisher even on the current single-system-dominant design (§3.5); joint $(a, d_e)$ identifiability is the open question, and hierarchical shrinkage on $d_e$ likely mitigates.
4. **Asymmetric β prior** — ship it (with the label-aware refinement) or leave on a branch? It structurally fixes q_j transmission, doesn't hurt fit quality, but doesn't visibly improve recovery either.
5. **Mixture leaf + identifiability fix** — worth investing more time, or shelve given the bimodal-mass deficiency suggests the cutpoint family is the deeper issue?
6. **Cross-stance pooling on C** — multi-week project; right priority for the next contract phase?

## 7. Reproducibility appendix

> *Fill: branch names (with commit hashes), what each branch contains, where
> each result lives. Tree of `notebooks/` subdirs.*

### Branch tree
- `synthetic-validation-checks` (5fc04f4) — 2026-05-07 M-closed pilot, oracle audits.
- `asymmetric-beta-prior-sweep` (929063b) — asymmetric β prior infrastructure + 5-fit sweep.
- `oracle-c-ablation-and-info-diags` — clamp infrastructure + info-diagnostic library + ladder smokes.
- `leaf-direct-q` — Pro_A direct_q ablation (no-marginal result).
- `leaf-mixture-shared-h` — Pro_D mixture leaf (confound + bimodal-mass finding).
- `c-identifiability-sample-size` — Branch 6 sample-size sweep.
- `expert-discrimination-feasibility-2026-05-10` — Arvo's discrimination feasibility check.

### Findings docs
- `notebooks/synthetic_validation_2026-05-06/findings_2026-05-07.md`
- `notebooks/asymmetric_prior_sweep_2026-05-10/findings.md`
- `notebooks/asymmetric_prior_sweep_2026-05-10/next_directions.md`
- `notebooks/asymmetric_prior_sweep_2026-05-10/external_review_framing.md` *(Pro consultation prompt)*
- `notebooks/oracle_c_ablation_2026-05-10/findings_oracle_c_ladder.md`
- `notebooks/leaf_direct_q_2026-05-10/findings_direct_q.md`
- `notebooks/leaf_mixture_2026-05-10/findings_mixture.md`
- *(Branch 6 + expert-discrimination findings, on Monday)*

## 8. SPAR-final-report mapping

> *Fill: which sections above map to which SPAR-report sections. Methods,
> Results, Discussion, Limitations, Future Work. So when the SPAR report is
> written this doc tells you what to lift.*

## 9. Forward-looking (for the next session / contract phase)

> *Fill — for cold pickup in 4-6 weeks:*
- *Most likely highest-leverage next experiment: …*
- *What's blocked on Pro consultation / Anthropic input: …*
- *What's blocked on next survey wave: …*
- *What infrastructure is in place and ready to extend: …*
