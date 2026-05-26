# Pro consultation: structural fixes for a binary-root DCM after prior-leak diagnosis

## What I'm asking you to do (priority order)

1. **Sanity-check the diagnoses for Issues 1 and 2.** Push back if there are alternative explanations for the same evidence that I'm under-weighting, or if my framing is subtly wrong.
2. **Rank the four candidate Issue 2 fixes** by principled-ness, feasibility-this-week, and downstream consequences for the headline output (per-system $P(R = 1 \mid \text{data})$).
3. **Specify a Bayesian-workflow ordering** for the next ~2 weeks of remaining work, and the cheap workflow checks that should sit in front of any structural change.
4. **Spot standard diagnostics I'm not doing** that should be high priority before the model gets handed off.

Secondary asks: how would you approach revisiting the support / demandingness → Beta translation table principally? And: is the synthetic-validation cross-seed pooling artefact (real vs synthetic per-system ratings look very different) something to fix at the validation layer or accept as a limit on what synthetic-from-prior validation can tell us?

You won't have repo access. Everything you need is in this document plus the attached meeting notebook md (see next section). Please push back on premises if anything reads as wrong rather than building on it.

---

## Materials and conventions

**Attached meeting notebook**: `spar_friday_meeting_2026-05-15.md` (rendered from the Jupyter notebook). It contains:
- The narrative framing of Issues 1 and 2
- Two key plots: **prior-predictive expert ratings** for Human (R=1) vs ELIZA (R=0), baseline prior vs targeted fix; and **real per-system rating distributions** (Human / Chicken / LLMs / ELIZA) pooled across all stances and indicators
- The targeted-fix description (high-level), 10-seed synthetic recovery summary, and four candidate Issue 2 fixes
- A Bayesian workflow reflection (Gelman et al. 2020, "Bayesian Workflow", arXiv:2011.01808, Figure 1)
- Key questions and actions

This document supplements the notebook with model spec, code-level mechanics, negative results from prior weeks, and detailed Pro questions. Please use the notebook for the framing and visuals; this document for technical depth.

**Notation.** $R_s \in \{0, 1\}$ is the binary root state ("system $s$ is conscious under stance $\sigma$") for a single draw. $C_s = P(R_s = 1)$ is the per-system prior probability of consciousness — the parameter. $\rho_s = P(R_s = 1 \mid \text{data})$ is the headline posterior output we report per (system, stance). I'll use "C" colloquially for both the parameter and the posterior; context disambiguates.

---

## Project recap

The DCM was developed at Rethink Priorities to formalise Bayesian aggregation of expert opinion about whether a system is conscious under each of several theoretical stances on consciousness (Global Workspace Theory, Higher-Order Thought, Integrated Information Theory, Attention Schema Theory, etc.). The original published model (Shiller 2026, "An initial discriminative consciousness model") reports per-(system, stance) $\rho_s$ as the headline output; it's intended to be re-run as new expert ratings come in over time.

Per stance, a hand-coded tree of binary latent features encodes how that stance would "score" a system: each indicator at the leaves is operationally observable (e.g. "the system has a winner-take-all attentional bottleneck"), and parent–child structure encodes the modeller's claim that "if this feature is present, then these subfeatures should be more likely." Edges are labelled with categorical (support, demandingness) tags from a domain-expert elicitation; those tags are converted to Beta prior parameters via a hand-tabulated translation table. **The labels themselves encode genuine domain knowledge and I do not want to discard them.** What's in scope is how that table maps into the prior, and what structure sits above the tree.

Project work this term reframed the original ad-hoc per-run Bernoulli-collapsing inference (and the Metropolis sampler for discrete latents) into a single PyMC model with the discrete latents marginalised analytically by sum-product / belief propagation. Continuous parameters are sampled with NUTS. This made local laptop sampling fast (~minutes), enabling fast iteration. That work is largely orthogonal to the issues below.

---

## Model specification (current production)

For each (system $s$, stance $\sigma$) pair:

**Root.** $C_s \sim \text{Beta}(1, 5)$, $R_s \mid C_s \sim \text{Bernoulli}(C_s)$. The Beta(1, 5) is the production "skeptical" prior (mean 1/6); under Phase 0 sensitivity work, swapping to uniform Beta(1, 1) on seed 06 moves recovered $\rho$ from $\sim 0.16$ to $\sim 0.62$ for LLMs — i.e., the prior is materially load-bearing on what we report.

**Tree edges.** For an edge from parent node $v$ to child $u$:
- $\beta^{\text{pres}}_u = P(z_u = 1 \mid z_v = 1) \sim \text{Beta}(\alpha^{\text{pres}}, \beta^{\text{pres}})$
- $\beta^{\text{abs}}_u = P(z_u = 1 \mid z_v = 0) \sim \text{Beta}(\alpha^{\text{abs}}, \beta^{\text{abs}})$

Beta hyperparameters come from the (support, demandingness) label of node $u$ via `EvidenceProcessor.get_beta_parameters(s, d)`. Concentration is fixed at $\alpha + \beta = 10$ per edge. The label-to-(α, β) map is the support / demandingness translation table that's the focus of the follow-up question.

So $z_u \mid z_v \sim \text{Bernoulli}(q_u(z_v))$ with $q_u(1) = \beta^{\text{pres}}_u$, $q_u(0) = \beta^{\text{abs}}_u$. The marginal probability that indicator $j$ is in the present state given root $C_s$ is

$$
q_j \;=\; \mathbb{E}\bigl[z_j \mid C_s, \boldsymbol{\beta}, \text{tree}\bigr]
$$

propagated by the recursion $q_u = \beta^{\text{abs}}_u + q_v \cdot (\beta^{\text{pres}}_u - \beta^{\text{abs}}_u)$ from the root to each leaf.

**Leaf — three-state.** $m_j \sim \text{Binomial}(2, q_j)$, $m \in \{0, 1, 2\}$. Emission centres in the latent score:

$$
\eta_m \in \{0,\; a/2,\; a\} \;+\; b_e
$$

where $a$ is a per-stance discrimination parameter and $b_e$ is a per-rater shift.

**Observation.** 7-category ordered probit:

$$
y_{ij} \mid \eta_{ij} \sim \text{OrderedProbit}\bigl(\eta_{ij},\; \boldsymbol{\kappa}\bigr).
$$

Priors: $a \sim \text{HalfNormal}(0, 2)$. $\boldsymbol{\kappa}$: 6 cutpoints, $\kappa_k \sim \mathcal{N}(0, 2)$ jointly under PyMC's `ordered` transform (initval $\text{linspace}(-1.5, 1.5, 6)$). $b_e \sim \mathcal{N}(0, 2)$ per rater (one rater anchored at 0 for identifiability via `USE_EXPERT_SHIFTS`).

**Marginalisation.** The discrete latents $\{R_s, z_u, m_j\}$ are integrated out analytically by a sum-product / belief-propagation pass over the tree. The structure is exact because the latent graph is a tree: per indicator we collect $(\log L(\text{ratings} \mid z_{\text{parent}} = 0), \log L(\text{ratings} \mid z_{\text{parent}} = 1))$ pairs and combine them up the tree using $\beta^{\text{pres}}, \beta^{\text{abs}}$. Implemented in `dcm_model_exact_tree.py`. NUTS samples a continuous joint posterior over $(C_s, \boldsymbol{\beta}, a, \boldsymbol{\kappa}, \mathbf{b})$.

**Cross-system structure.** Reference-system anchoring is via soft anchors: e.g. Beta(50, 1) on $C_{\text{Human}}$, Beta(1, 50) on $C_{\text{ELIZA}}$. This is the only cross-system coupling. The free systems (Chicken, 2024 LLMs) get Beta(1, 5). Tree betas can be either per-system (the original setup), pooled by (support, demandingness) label across systems (`POOL_BETAS_BY_LABEL`, the current default), or hybrid. The current production setup pools tree betas by label.

---

## Data and design

13 stances in `data_cache.json`. For GWT (the worked example throughout this consultation), the tree has ~50 indicators across depth 1–3. Total ratings on GWT: 379 across 6 raters (Rater_A–Rater_E plus one anchored).

Four target systems: Human (anchored ~ 1), ELIZA (anchored ~ 0), Chicken (free), 2024 Leading Chat LLMs (free). Cross-system rater coverage is uneven — one rater (Rater_B) covers Human / LLMs / ELIZA but not Chicken; the others are mostly system-specific.

Real per-system rating distributions, pooled across all stances and indicators (raw 0–1 scale binned into 7 equal-width ordinal categories):

| System | n | %@cat 1 | %@cat 7 | %@cats 3–5 | shape |
|---|---:|---:|---:|---:|---|
| Human | 731 | 0.1% | **92.5%** | 4.3% | extreme high |
| Chicken | 1386 | 5.6% | 37.7% | **37.8%** | upper-middle spread |
| LLMs (2024) | 2779 | **36.8%** | 20.6% | 24.9% | broad spread |
| ELIZA | 730 | **99.3%** | 0.4% | 0.0% | extreme low |

**This is what the model has to fit.** Chicken and LLMs genuinely sit between the extremes, with substantial mass in middle categories.

---

## What's been ruled out (negative results from earlier weeks)

These are the "I've already tried this and it didn't work" baseline. All on the GWT tree, on synthetic data drawn from a known-truth fit configuration.

1. **Leaf alternatives.** Two leaf-model alternatives to three-state were tested:
   - **direct-$q$**: $\eta = a \cdot \tilde{q}_j + b_e$, no latent indicator state. Analytical Fisher info for $C$ is **3× higher** than three-state, $I(Q; Y)$ is 1.74× higher, full-vector KL is 2.85× higher. _Empirical_ posterior contraction on $C$ for Chicken: 0.927 → 0.920. Essentially no improvement. The discrete-state compression is not the binding constraint.
   - **Mixture** (2-component shared-h centred-endpoint): $\eta_h = b_e \pm \delta/2$. Median moves toward Chicken's truth (0.227 vs 0.119 for three-state at oracle-C ladder rung) but **posterior broadens beyond the prior** (contraction 1.16 vs 0.87). Empirically worse than the baseline three-state for $C$ identification.
   - Conclusion from these two: **leaf-model changes aren't the lever**.
2. **Sample-size sweep.** $K \in \{1, 2, 5, 10, 20\}$ rater-multiplier on synthetic seed 20260506 with the centre asymmetric β override (β_pres = 0.90, β_abs = 0.10). **Posterior SD on free $C_s$ asymptotes at the prior SD ($\approx 0.141$) across all $K$.** Contraction ratios plateau at $\sim 1.0$. $1/\sqrt{K}$ shrinkage is _not_ observed. So **more raters at the current design will not move $C$**. The likelihood is structurally flat for $C$ at this design — not data-limited.
3. **Oracle-C ladder.** Pinning subsets of nuisance parameters (a, κ, b, β) to truth and measuring posterior contraction on the unanchored $C$. Even with all nuisances pinned, the C-likelihood is essentially flat for the free systems on the synthetic data tested. So "nuisance absorbing C signal at the free fit" isn't the explanation either; the C-information just isn't there at this design.
4. **Asymmetric-prior sweep (analytical).** Mean prior-implied $q_j$ at depth 3 under different override centres for Human (R=1) and ELIZA (R=0):

| Prior | $q_\text{Human}$ | $q_\text{ELIZA}$ | Gap |
|---|---:|---:|---:|
| Baseline (paper means) | 0.631 | 0.593 | 0.038 |
| pres0.85 / abs0.15 | 0.671 | 0.329 | 0.342 |
| pres0.90 / abs0.10 | 0.755 | 0.245 | 0.510 |
| pres0.95 / abs0.05 | 0.864 | 0.136 | 0.728 |

Each row is the prior-mean $q_j$ averaged over all depth-3 indicator nodes. This shows the prior alone — no data — can't transmit C signal under the baseline.

---

## The targeted fix — mechanics

Implemented in `dcm_model.py` and `scripts/main_synthetic_validation_metrics.py`. Two pieces:

**Identifying the targeted edges.** `collect_targeted_override_node_keys` walks the tree and returns every node key whose top-level feature is in a "strong top features" list (currently `STRONG_TOP_FEATURES = {"Coherence", "Selective Attention", "Complexity"}` — these were identified empirically as the dominant blocking edges) and whose depth $> 1$. The set is per-stance; for GWT it lands ~25–50 lower-edge nodes.

**Overriding the prior at those edges.** For each targeted node, `build_safe_gain_node_beta` replaces the Beta prior with a logit-Normal centred at $\text{logit}(\mu_\text{gain})$ with $\sigma = 0.30$:
```
tilde   ~ Normal(0, 1)
logit_β = logit(μ_gain) + σ · tilde
β       = sigmoid(logit_β)
```
with $\mu_\text{gain} = 0.90$ for $\beta^{\text{pres}}$ and $0.10$ for $\beta^{\text{abs}}$. Non-centred parameterisation; sidesteps Beta boundary singularities when the centre approaches 0 or 1. Non-targeted edges keep their paper-mu Beta priors.

**What it buys us (synthetic recovery).** 10-seed pilot, balanced accuracy 0.80 overall; all $R = 0$ cases recovered correctly; 6/10 $R = 1$ cases recovered correctly; the 4 misses are oracle-negative (the underlying simulated parameter draws were themselves weak under the truth). Sampler clean. Source: `outputs/main_synthetic_validation_pilot_20260511/pilot_decision_report.md`.

**What it doesn't fix.** The prior-predictive at the leaf is still poor — see the attached notebook plot. Even under the fix, ELIZA's prior-predictive ratings don't concentrate near the lowest categories; both Human and ELIZA still lean noticeably high. The fix moves things from "prior ignores consciousness" to "prior registers it weakly". It's the minimum-invasive intervention that lets recovery work; it doesn't represent the right joint prior over expert ratings.

---

## The synthetic-validation cross-seed pooling artefact

Worth flagging because it surprised me this morning when I went to make the Issue 2 visual.

The `posterior_predictive_rating_distribution_by_system.csv` from `outputs/main_synthetic_validation/` shows Chicken and LLMs as roughly bimodal at categories 1 and 7 — see `outputs/main_synthetic_validation/plots/ppc_rating_distribution_overlay_by_system.png` (also embedded in the attached notebook).

This is an artefact of pooling across 10 synthetic seeds where the synthetic root $R_s$ for the unanchored systems was randomly drawn $0$ or $1$ per seed. When half the seeds have $R = 1$ (Chicken / LLMs producing high ratings) and half have $R = 0$ (low ratings), the pooled distribution looks bimodal. No single synthetic seed produces the genuinely-spread-across-the-middle distribution that the real data shows for Chicken / LLMs.

So the synthetic data the validation has been generating doesn't resemble real expert ratings for middle systems at all. That's a real critique of the synthetic-validation framework — not just "synthetic data drawn from a misspecified prior is uninformative", but "synthetic data drawn from the model's own forward distribution doesn't resemble the real data even structurally."

---

## The four candidate Issue 2 fixes (with my reasoning)

In the order I currently have them in the notebook. Genuinely uncertain on the ranking — would value an external read.

1. **Continuous $C$ at the root.** Replace binary $R_s$ with a continuous root probability $C_s \in [0, 1]$ so middle systems can be fit at intermediate values directly rather than as mixtures of two extreme regimes.
   - Prior on $C_s$: most natural is $\text{Beta}(1, 5)$ (matching the current parameter-level prior).
   - Tree propagation: $q_u = \beta^{\text{abs}}_u + q_v \cdot (\beta^{\text{pres}}_u - \beta^{\text{abs}}_u)$ extends naturally with $C_s$ taking the role of the "root $q$".
   - Loss of binary $R_s$ semantics ("is system $s$ conscious?") feels like a loss to me, but maybe a clean one. Headline output becomes $E[C_s \mid \text{data}]$ rather than $P(R_s = 1 \mid \text{data})$.
   - **My weak prior**: this is the most actionable change for the time remaining, and the most likely to help middle-system identification.

2. **Graded reference systems.** Add anchor systems at intermediate root probabilities. Natural candidate: chimpanzee toward the upper end (0.8?). Lower end is harder — most of the middle-spectrum candidates (cephalopods, fish) are scientifically contested.
   - _Big asterisk:_ assigning a defensible reference probability for anything that isn't an extreme is genuinely hard, and is likely stance-dependent.
   - Implementation cost in the model is small (just more soft anchors); cost is mostly in justifying the numbers.
   - Could be combined with (1).

3. **Cross-stance pooling.** Hierarchical structure across stance variants for the same system, instead of independent per-stance $C_{s, \sigma}$. Borrows strength across stances.
   - Most likely beyond what I can implement before SPAR ends.
   - Relevant to Matilda's tree-structure work; flagged as a longer-term direction.
   - _Asterisk:_ any structural change has to remain compatible with the exact-tree (sum-product) algorithm we use for marginal likelihood — that means it has to stay a tree (no two-parents → one-child), and very deep structures hurt propagation more than wider ones.

4. **Open structural choice — joint shared tree (current) vs per-system tree (original).** Both have known issues. Per-system has historically had per-system parameters absorb rating-distribution differences without informing $C$.
   - My weak lean: stay with the joint shared tree and use (1) and / or (2) to fix middle-system identification.
   - This is orthogonal to (1)–(3) but interacts with all of them.

---

## Specific questions

I'd find it useful if you could **rank these by importance** for the goals stated at the top, and then answer each.

**A. Issue 1 alternative explanations.** "β prior collapses C signal" is my diagnosis, supported by the analytical $q_j$-vs-depth gap going from 0.038 (baseline) to 0.51 (pres0.90/abs0.10) and the prior-predictive ratings being near-identical for H/E under baseline. What competing diagnoses fit the same evidence? E.g., is it actually the cutpoints' prior (a / κ width)? The three-state vs binary leaf? Something about the $\alpha + \beta = 10$ concentration? Is "translate domain labels into wider gaps via a different α/β formula" the right framing, or is there a more upstream diagnosis I'm missing?

**B. Targeted fix — band-aid or principled?** Is overriding only the strong-lower edges defensible, or am I just hiding the problem? What would a principled prior-predictive-driven revision of the support / demandingness → β translation look like? (Concrete question: if you were to elicit the right table, what _prior-predictive at the leaf_ would you target — e.g., $E[y \mid R = 1] = 6$, $E[y \mid R = 0] = 2$? Some other prior-predictive constraint?)

**C. Issue 2 alternative failure modes.** "Binary root + extreme-only anchors → $C$ unidentified for middle systems" is my diagnosis, supported by the leaf-mixture / direct-$q$ / sample-size negative results. What other structural failure modes would produce wide $C$ posteriors for unanchored middle systems? E.g., per-system parameters absorbing signal, rater-effect identification failures, three-state-vs-binary at the leaf in some way I haven't thought of?

**D. The four candidates — rank.** Of (1) continuous $C$, (2) graded references, (3) cross-stance pooling, (4) joint vs per-system tree — which is most principled given the goal of estimating per-system $C$ with defensible uncertainty? Is continuous $C$ a clean fix or does it lose the binary "is it conscious" semantics in a way that hurts interpretation downstream? If you'd rank them differently, why?

**E. Continuous $C$ specifics.** If continuous $C$ is the right next step, what's the right parameterisation? Beta on $C$? Logit-Normal on $C$? How should the tree propagate from a continuous root — does $q_u = \beta^{\text{abs}}_u + C_s \cdot (\beta^{\text{pres}}_u - \beta^{\text{abs}}_u)$ work directly, or are there subtler identification problems that crop up when $C_s$ is no longer binary? What does the soft anchor on Human / ELIZA look like under continuous $C$ — Beta(50, 1) directly on $C$?

**F. Bayesian workflow next step ordering.** Before I implement continuous-$C$, what cheap workflow checks would tighten the case and de-risk it? (E.g., prior-predictive on continuous-$C$ model first, fake-data with known continuous truths spaced across [0, 1], identifiability profiling holding everything else fixed, …) What would Gelman et al.'s workflow figure suggest as the very next step from where I am?

**G. Standard diagnostics I'm not running.** What workflow-standard diagnostics would you expect for this kind of model that I haven't mentioned? PSIS-LOO with K-checks (we have basic LOO but I'm not sure how informative it is here)? Held-out experts? Identifiability profiling? Prior-data conflict checks (Evans & Moshonov)?

**H. Synthetic-validation cross-seed pooling.** Should we (a) generate synthetic data with $R$ values designed to match real per-system distributions (e.g., draw $C_\text{Chicken}$ from Beta tilted to ~0.4 to match the upper-middle pattern), (b) accept synthetic-from-the-model-prior as a self-consistency check only, or (c) move to fake-data validation with continuous-$C$ truths instead?

**I. Anything else.** Anything from the negative-results section that you think should be revisited differently? Specifically: if leaf-model changes empirically can't lift $C$ identification on Chicken / LLM, but analytical Fisher info says they should — what's the model-level explanation for that gap?

---

## What I'm not asking

- I'm not asking to revisit the choice of binary-root reporting estimand (that was Phase 0).
- I'm not asking about the philosophy / metaphysics of consciousness, or whether LLMs are conscious. The question is structural: given the elicitation we have, what's the right model.
- I'm not asking for a complete rewrite. The exact-tree marginalisation, ordinal probit observation, and basic tree-propagation structure all stay. The questions are about the prior, the root parameterisation, and any cross-system / cross-stance structure above the tree.

---

## Repository pointers (in case anything matters)

- Code: `github.com/arvomm/dcm-spar`, branch `ryan/spar-handover-2026-05-12`.
- Key files: `dcm_model.py` (model construction + per-system fit), `dcm_model_exact_tree.py` (joint multi-system with exact-tree marginalisation), `scripts/main_synthetic_validation.py` and `scripts/main_synthetic_validation_metrics.py` (synthetic validation + targeted fix).
- The forward-simulator that generated the prior-predictive plot in the notebook: `notebooks/spar_friday_meeting_2026-05-15/generate_prior_predictive_ratings.py`.
- Real-data per-system distribution generator: `notebooks/spar_friday_meeting_2026-05-15/generate_real_observed_ratings.py`.
