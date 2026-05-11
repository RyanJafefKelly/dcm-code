# Synthetic validation programme for a discriminative consciousness model (DCM): help scoping the simplest informative experiment

## What I'm asking you to do

I'm building a Bayesian model that estimates the probability of consciousness $C_s \in [0, 1]$ for a system $s$ based on expert ratings of indicators in a hand-coded feature tree. The model has been failing to recover $C_s$ even on synthetic data with abundant ratings, and I'm now reverting to a "build the smallest informative toy and add features back one at a time" approach to localise where things break.

I want your help with three things, in priority order:

1. **Specify the minimal model** — what is the smallest version of the DCM that is still meaningfully a DCM, such that I can verify it works on synthetic data before adding back the complexity?
2. **Specify the metrics** — what should "model is working" mean, in concrete terms a coding agent could compute and threshold, for (a) $C_s$ recovery, (b) rating-distribution recovery, (c) internal-node parameter recovery?
3. **Specify the ladder** — what is the right order to add features back, and what failure modes should I expect at each rung?

A secondary ask: I have several **hypotheses about why the current model fails**. I'd like you to reason about which are most plausible and design experiments in the toy that can adjudicate.

You won't have repo access. Everything you need is (hopefully) included below. Please reach out if there is further information I could provide that would be useful. Some of what I write may be subtly wrong — please push back on premises if you spot issues rather than building on them.

---

## Background: the DCM and its tree structure

The model originates from Shiller (2026), "An initial discriminative consciousness model," and assigns to each (system $s$, theoretical stance $\sigma$) a binary latent $C_s \in \{0, 1\}$ representing "system $s$ is conscious under stance $\sigma$." The prior on $C_s$ is stance-level, defaulting to $\mathrm{Beta}(1, 5)$.

Below $C_s$, each stance defines a hand-coded tree of binary latent features. Internal nodes are features and subfeatures; leaves are indicators that experts rate. Every edge $v \to u$ in the tree carries two conditional probabilities:

- $\beta^{\mathrm{pres}}_u = \Pr(z_u = 1 \mid z_v = 1)$ (sensitivity-like)
- $\beta^{\mathrm{abs}}_u = \Pr(z_u = 1 \mid z_v = 0)$ (false-positive-like)

So $z_u \mid z_v \sim \mathrm{Bernoulli}(q_u(z_v))$ with $q_u(1) = \beta^{\mathrm{pres}}_u$ and $q_u(0) = \beta^{\mathrm{abs}}_u$. The two parameters per edge are given independent Beta priors, with prior means set by hand-assigned (support, demandingness) labels — categorical labels that the modellers spent considerable time setting per edge. **I do not want to discard the labels** — they encode genuine domain knowledge.

The marginal probability that an indicator $j$ is present in system $s$ is then

$$ q_j = \Pr(z_j = 1 \mid C_s, \boldsymbol{\beta}, \text{tree}) $$

propagated from the root to each leaf by the tree's edge transmissions.

In the published model, observations are processed as follows. For each (system, indicator) cell, the cross-rater mean $\bar p_{sj} \in [0,1]$ is computed. Then *at the start of every PyMC sampling run*, this mean is collapsed stochastically: $y_{sj} \sim \mathrm{Bernoulli}(\bar p_{sj})$, and the latent indicator $z_j$ is observed against $y_{sj}$ via a single Bernoulli factor. Hundreds of runs are repeated with fresh $y$ draws, and the reported posterior on $C_s$ is averaged across runs. Each (system, stance) pair is fit as a separate PyMC model. Discrete latents are sampled with a Metropolis kernel; continuous $\boldsymbol{\beta}$ parameters use NUTS.

## Project context: paper, systems, data

The DCM was developed at Rethink Priorities to formalise Bayesian aggregation of expert opinion about whether a given system is conscious under each of several theoretical stances on consciousness (Global Workspace Theory, Higher-Order Thought, Integrated Information Theory, Attention Schema Theory, etc.). The hand-coded tree per stance encodes how that stance would "score" a system: each indicator at the leaves is meant to be operationally observable (e.g. "the system has a winner-take-all attentional bottleneck"), and the parent–child structure encodes the modeller's claim that "if this feature is present, then these subfeatures should be more likely." The published work (Shiller 2026, "An initial discriminative consciousness model") reports per-stance, per-system $C_s$ posteriors as the headline output, and is intended to be re-run as new expert ratings come in over time.

The current production data covers four target systems on the GWT (Global Workspace Theory) stance:

- **Human** — anchored at $C_s \approx 1$. Canonical conscious system. One rater (Rater_B, the modeller).
- **ELIZA** — anchored at $C_s \approx 0$. Canonical non-conscious chatbot. One rater (Rater_B).
- **Chicken** — free. Two raters (both Chicken-only).
- **2024 chat LLMs** — free. Four raters (LLM-only) plus Rater_B.

Total ratings on the GWT stance: Human 50, Chicken 93, LLM 186, ELIZA 50 — each rating is one (rater × indicator), on the 7-point Likert scale (strongly absent / absent / somewhat absent / unsure / somewhat present / present / strongly present). Six raters total; Rater_B is the only one with cross-system coverage (Human, LLM, ELIZA). The GWT tree has depth 3 (root $C \to$ top-level features $\to$ subfeatures $\to$ indicator leaves), with a few dozen indicators total.

The (support, demandingness) labels per edge encode two orthogonal claims:

- **Support** ∈ {overwhelming / strong / moderate / weak support, no bearing, weak / moderate / strong / overwhelming undermining}: how strongly does the parent feature being present *support* inferring the child is also present? Roughly maps to $\beta^{\mathrm{pres}}$.
- **Demandingness** ∈ {overwhelmingly / strongly / moderately / weakly demanding, neutral, weakly / moderately / strongly / overwhelmingly undemanding}: how restrictive (rare) is the child feature absent the parent? Roughly maps to $\beta^{\mathrm{abs}}$.

Each (support, demandingness) pair maps to a fixed Beta prior on $\beta^{\mathrm{pres}}$ and on $\beta^{\mathrm{abs}}$ via a hand-tabulated rule (see code excerpt below). About a dozen unique label combinations actually appear in the GWT tree. The published priors give relatively *narrow* gaps $\beta^{\mathrm{pres}} - \beta^{\mathrm{abs}}$ for many edges (means around $0.6$ vs $0.4$ for moderate-strength labels), which is what motivates the asymmetric prior override discussed below — under those default means, multiplicative collapse $q_{\mathrm{child}} = \beta^{\mathrm{abs}} + q_{\mathrm{parent}} \cdot (\beta^{\mathrm{pres}} - \beta^{\mathrm{abs}})$ shrinks the gap between $C_s = 1$ and $C_s = 0$ implied $q_j$ by a factor each level down the tree.

## SPAR-tenure changes I've made

These may matter for understanding why my current diagnostics don't easily isolate the failure:

1. **Tree marginalisation (sum-product / belief propagation).** Instead of sampling discrete latents with Metropolis, I marginalise them analytically by a bottom-up DP in PyTensor. The likelihood becomes a function of continuous parameters only. NUTS samples cleanly, $\hat R$ and ESS apply across the full posterior, and joint multi-system fits become natural. *This is a tooling change, not a modelling change* — it implements the same generative model exactly, and removes the discrete-Metropolis bottleneck. I consider this load-bearing infrastructure and intend to keep it in the toy.
2. **Ordinal-probit observation layer.** Replaces the binary-collapse + outer-loop with a proper ordinal-probit likelihood on the raw 7-point Likert ratings. For rater $e$ and indicator $j$, the latent signal is $s_{ej} = b_e + a \cdot z_j + \varepsilon_{ej}$, $\varepsilon_{ej} \sim \mathcal{N}(0, 1)$, with shared cutpoints $\boldsymbol{\kappa} \in \mathbb{R}^6$ binning to 7 categories. Marginalising $z_j$ gives a 2-component mixture rating likelihood per rating. Eliminates outer-loop variance.
3. **Three-state indicator latent.** Instead of $z_j \in \{0, 1\}$, $m_j \sim \mathrm{Binomial}(2, q_j)$, $z_j = m_j / 2 \in \{0, 0.5, 1\}$, latent emission centres at $\eta \in \{0, a/2, a\} + b_e$. Lets a moderate $q_j$ place rating mass at the middle category instead of mixing between extremes.
4. **Reference-system joint fit + anchors.** All target systems fit jointly with shared observation-layer + tree parameters. Two reference systems anchored: Human at $C_s \approx 0.999$, ELIZA at $\approx 0.001$. The free systems (Chicken, 2024 LLMs) inherit calibration from the anchored systems via shared parameters.
5. **Asymmetric $(\beta^{\mathrm{pres}}, \beta^{\mathrm{abs}})$ prior sweep.** Override the per-label prior means with values like $(0.90, 0.10)$ or $(0.95, 0.05)$ to enforce a large transmission gap. Motivation: the published priors give $\bar q_{\text{Human}} - \bar q_{\text{ELIZA}}$ at depth 3 of $\approx 0.04$ — i.e., the root anchor signal does not transmit to the leaves. The asymmetric prior fixes this analytically (gap $\approx 0.51$ at $(0.90, 0.10)$), but **doesn't visibly improve empirical $C_s$ contraction**.
6. **Continuous-mixture leaf** (exploratory). Two-component mixture with centred endpoints $\eta_h = b_e \pm \delta/2$. Recovers the Chicken > LLM ordering on real data for the first time, but $\hat R \approx 1.73$ on $(\delta, \kappa)$ — likely multimodal in observation layer. Under oracle nuisance clamp, gives Chicken contraction 1.16 (broader than prior); three-state gives 0.87 at the same clamp.

## Code reference: the load-bearing pieces

A few selected snippets so you can see exactly what is being computed. The key fact is that **the inference path is purely continuous**: discrete latents are marginalised analytically, and NUTS samples a continuous joint posterior over $(\boldsymbol{\beta}, a, \boldsymbol{\kappa}, \mathbf{b})$ plus any free $C_s$.

### Bottom-up tree DP (sum-product on the tree)

For each system, the marginal log-likelihood is computed by a depth-first recursion that returns, for each subtree rooted at $v$, the pair $(\log L(z_{\mathrm{pa}} = 0), \log L(z_{\mathrm{pa}} = 1))$ — the conditional likelihood of the subtree's data given the parent's binary state. Unrolled at model-build time into a PyTensor expression:

```python
def subtree_lls(node, path):
    """Return (log L(z_pa=0), log L(z_pa=1)) for this subtree."""
    bp, ba = node_betas[key]   # β_pres, β_abs at the edge into this node

    if ntype == "indicator":
        # Leaf mixture: B(β) = (1-β) * exp(ll_z=0) + β * exp(ll_z=1)
        leaf_lls = indicator_leaf_lls[(sys_name, key)]
        log_L_zpa1 = leaf_log_B(bp, leaf_lls)   # parent z=1 → q_self = β_pres
        log_L_zpa0 = leaf_log_B(ba, leaf_lls)   # parent z=0 → q_self = β_abs
        return log_L_zpa0, log_L_zpa1

    # Internal node: combine children, then condition on parent edge
    log_L_v0 = sum_c log_L_c_at(z_v=0)   # children's contributions when self is 0
    log_L_v1 = sum_c log_L_c_at(z_v=1)
    log_L_zpa1 = logaddexp(log(bp) + log_L_v1,  log(1-bp) + log_L_v0)
    log_L_zpa0 = logaddexp(log(ba) + log_L_v1,  log(1-ba) + log_L_v0)
    return log_L_zpa0, log_L_zpa1

# Top-level: condition on stance C
log_L = logaddexp(log(C) + log_L_top1,  log(1-C) + log_L_top0)
```

(`dcm_model_exact_tree.py:234`. The actual implementation also handles three-state leaves, label pooling, and per-system parameter sharing; the kernel is what's above. Verified to numerical precision against an independent NumPy reference per draw.)

### Leaf likelihood (per-indicator marginalisation)

For a single indicator $j$ with $n_j$ ratings, marginalising the latent state across the indicator's ratings (latent state is shared across ratings of the same indicator):

```python
# Three-state: m_j ~ Binomial(2, q_j), z_j = m_j/2 ∈ {0, 0.5, 1},
#              latent emission centres eta_m ∈ {0, a/2, a} + b_e.
# Per-indicator components (sum over the indicator's ratings):
#   ll_m = sum_i log P_OP(r_ij | eta_m, theta)

log_w0 = 2 * log(1 - q_j)
log_w1 = log(2) + log(q_j) + log(1 - q_j)
log_w2 = 2 * log(q_j)

log_lik_j = logsumexp([log_w0 + ll_0, log_w1 + ll_1, log_w2 + ll_2])
```

Binary case is the same with two components:

```python
# Binary: z_j ~ Bernoulli(q_j), eta ∈ {0, a} + b_e.
log_lik_j = logaddexp(log(1 - q_j) + ll_z0,  log(q_j) + ll_z1)
```

(`dcm_model.py:461 (three_state_log_weights)`, `dcm_model.py:470 (pt_three_state_ll_terms)`.)

### Ordinal observation layer

```python
a = pm.HalfNormal("a", sigma=A_PRIOR_SIGMA)               # default sigma = 2.0
kappa = pm.Normal("kappa", mu=0, sigma=KAPPA_PRIOR_SIGMA,  # default sigma = 2.0
                  shape=K-1, transform=transforms.ordered)

# Per-rater shifts (one rater anchored at b=0 to remove location degeneracy with kappa)
b_free = pm.Normal("b_free", mu=0, sigma=EXPERT_SHIFT_SIGMA, shape=n_experts-1)
b = pt.concatenate([pt.zeros(1), b_free])

# Per-rating likelihood given latent eta = a*z_j + b_e:
#   P(r = k | eta) = Phi(kappa[k] - eta) - Phi(kappa[k-1] - eta)
```

(`dcm_model.py:619`. K = 7 for the 7-point Likert.)

### Support / demandingness → Beta prior mapping

```python
# At default TRANSMISSION_GAIN=1.0, the published mapping. NODE_CONCENTRATION = 10
# means alpha+beta = 10 for every edge's Beta prior.
absence_alpha, absence_beta = demandingness_map[demandingness]   # → β_abs Beta params
support_factor = support_map[support][demandingness]             # multipliers for β_pres
presence_alpha = absence_alpha * support_factor[0]
presence_beta  = absence_beta  * support_factor[1]
# Then renormalised so alpha+beta = NODE_CONCENTRATION.
```

Examples of resulting prior means under the published mapping:

- "strong support / neutral demandingness" → $\beta^{\mathrm{pres}}$ mean $\approx 0.89$, $\beta^{\mathrm{abs}}$ mean $\approx 0.50$.
- "moderate support / weakly demanding" → $\beta^{\mathrm{pres}}$ mean $\approx 0.6$, $\beta^{\mathrm{abs}}$ mean $\approx 0.4$.
- "weak support / neutral" → both $\approx 0.5$ (very narrow gap; transmits almost no signal).

The asymmetric override (`BETA_PRES_OVERRIDE_MEAN=0.90, BETA_ABS_OVERRIDE_MEAN=0.10`) replaces these per-label means with a uniform $(0.90, 0.10)$ across all labels, in logit-Normal form with `LABEL_POOL_SIGMA = 0.5` (so per-edge $\beta$ remains a random variable around the override mean). This is what fixes the prior-implied transmission gap structurally but doesn't move the empirical $C_s$ posterior contraction.

(`dcm_model.py:894 (EvidenceProcessor.get_beta_parameters)`.)

### Key ModelConfig flags

```python
INDICATOR_STATE_MODEL = "binary" | "three_state"      # leaf family
USE_EXPERT_SHIFTS = True                              # b_e per rater (else b=0 for all)
EXPERT_SHIFT_SIGMA = 2.0                              # b_e prior scale
A_PRIOR_SIGMA = 2.0
KAPPA_PRIOR_SIGMA = 2.0
N_CATEGORIES = 7
NODE_CONCENTRATION = 10.0                             # alpha+beta on every edge's Beta prior
DEFAULT_ALPHA, DEFAULT_BETA = 1, 5                    # stance C ~ Beta(1, 5)
SOFT_REFERENCE_ANCHORS = {"Human": (50, 1),
                          "ELIZA": (1, 50)}            # Beta priors on free C; alternative
                                                       # to hard fixed C
POOL_BETAS_BY_LABEL = False                           # complete-pool β within label group
LABEL_POOL_SIGMA = 0.5                                # logit-Normal scale around paper mean
BETA_PRES_OVERRIDE_MEAN = None                        # set to 0.90 for asymmetric override
BETA_ABS_OVERRIDE_MEAN  = None                        # set to 0.10 for asymmetric override
TRANSMISSION_GAIN = 1.0                               # symmetric logit-gap rescaling of label means
```

## What's empirically failing

The strongest negative result is the **sample-size sweep**. Replicate each (rater, indicator) slot $K$ times for $K \in \{1, 2, 5, 10, 20\}$ on synthetic seed 06, fit with the marginalised model and the asymmetric centre $\beta$ prior. The posterior SD on free $C_s$ asymptotes at the prior SD ($\approx 0.141$). Posterior medians plateau near the prior mean ($\approx 0.17$) for both Chicken (truth 0.25) and LLM (truth 0.10), even at $K = 20$. Contraction $= \mathrm{sd}(C_s \mid r) / \mathrm{sd}(C_s)$ never drops below 0.93. **Shrinkage is not parametric** — there is no $1/\sqrt K$ effect.

This is consistent with the **oracle clamp ladder**: even when $a, \boldsymbol{\kappa}, \mathbf{b}, \boldsymbol{\beta}$ are pinned at synthetic truth and only $C_s$ is free, the three-state leaf gives Chicken contraction 0.87. A direct-$q$ leaf ($\eta_{je} = a \cdot \tilde q_j + b_e$, no latent state) has $\sim 3\times$ analytical Fisher information about $C_s$ relative to three-state, but moves empirical contraction by less than 0.01. Continuous mixture is worse than three-state at 1.16.

The **PPCs on real data** show a separate problem: at fitted $\tilde q_j \in [0.6, 1.0]$, observed ratings concentrate 85–88% at category 6, while the model predicts 28–45% at category 6. This bimodal-mass deficit appears under all leaf families tested and looks like a property of the ordered-probit family with shared cutpoints — it cannot simultaneously fit mid-range distributions (need spread) and extreme-range distributions (need concentration).

There is a structural design constraint worth flagging: of 6 raters in the dataset, 5 rate exactly one system (3 LLM-only, 2 Chicken-only). Only one rater is cross-system (LLM, Human, ELIZA). This means any per-system parameter (per-system $\kappa$, per-rater anything when raters are mostly single-system) is co-identified with $C_s$ — both absorb per-system rating-distribution shifts. An ablation with no anchors and per-system $\kappa$ collapses every $C_s$ posterior to the prior mean.

## Three hypotheses for what's actually wrong (please challenge or rank)

1. **Per-system / per-rater parameters absorb the $C_s$ signal.** The single-system-rater design means the model can't separate "this system's ratings tend to be higher" (which $C_s$ should explain) from "this rater scores generously" (which a per-rater shift would explain). Empirical support: the no-anchor + per-system $\kappa$ ablation collapses every $C_s$ posterior to the prior mean; with hard anchors removed the model loses all per-system distinguishability. Implication: a survey-design constraint, not a modelling lever.
2. **Transmission gap at depth.** Even with the asymmetric $(0.90, 0.10)$ prior, the *posterior* on $\boldsymbol{\beta}$ may stay at the prior because the data has no curvature about $\boldsymbol{\beta}$ either. So the structural gap fix only helps if the true $\beta$ values *are* themselves extreme. My conjecture: real-world $(\beta^{\mathrm{pres}}, \beta^{\mathrm{abs}})$ values for these features may be *more* extreme than the modeller-set prior means — but I don't want to discard the support/demandingness labels, which were thoughtfully assigned. *I'd particularly value your reasoning here* — is this conjecture consistent with the diagnostic pattern, or is something else more likely?
3. **Observation-layer rigidity.** Ordered probit with shared cutpoints can't fit the bimodal-mass extreme-rating concentration. This is independent of the $C_s$ identification problem (it's a PPC issue, not a contraction issue), but possibly entangled — if the observation layer is mis-specified, the inferred $\boldsymbol{\beta}$ may compensate in ways that destroy $C_s$ identifiability.

There is also a meta-concern. I previously thought (a) fixing the $\boldsymbol{\beta}$ prior to be asymmetric was the main issue (it structurally fixes transmission), and (b) the mixture leaf would fix the residual (more flexible emission). Neither has visibly moved $C_s$ recovery. I am genuinely stuck on what the binding constraint is.

## What I'm proposing: a synthetic ladder

Build the smallest model that still resembles a DCM:

- **Tiny tree:** root $C \to$ 2 internal features $\to$ 4 leaves (depth 2). Each edge has its own $\beta^{\mathrm{pres}}, \beta^{\mathrm{abs}}$.
- **Two systems:** one anchored reference (truth $C \approx 1$), one free system (truth $C$ varies across experiments).
- **Multiple raters per system**, fully crossed (every rater rates every indicator on every system) — explicitly *not* the single-system-rater design.
- **Simplest possible leaf observation:** directly observe $z_j$ as binary with small Bernoulli noise $\varepsilon$ (NOT ordinal-probit at first). I.e. $y_{ji} \sim \mathrm{Bernoulli}(z_j(1-\varepsilon) + (1-z_j)\varepsilon)$.
- **Always use tree marginalisation** (sum-product) — never sample latents.
- No rater shifts $b_e$ initially; defer.

Verify on this:

- As # raters grows, do we recover $C_s$ for the free system?
- Do we recover edge $\beta$ values?
- Coverage of credible intervals at the nominal level?

If yes, escalate one feature at a time and find where things break:

- Increase tree depth (3, 4, 5)
- Increase tree fanout
- Replace binary leaf with ordinal-probit (start $K=3$, then $K=7$)
- Replace fully-crossed raters with single-system raters per system (the live design)
- Add rater shifts $b_e$ with a single cross-system rater
- Add three-state $z$ or continuous-mixture leaf
- Switch from marginalisation to discrete-Metropolis sampling (sanity check the published inference path)

I want to find the rung at which things break and have crisp evidence about why.

## Specific questions for you

1. **Simplest model.** Is "depth-2 tree + binary leaf + tree marginalisation + 1 anchored ref + 1 free system + fully-crossed raters" the right starting point? What am I missing? Is there value in starting *even smaller* (e.g. single internal feature, one leaf)?

2. **Metrics — please be concrete enough that a coding agent can implement them and I can see "the model is broken" without staring at plots.** Specifically:
   - **For $C_s$ recovery:** I have bias, MAE/RMSE, coverage of 50/80/95% credible intervals, contraction ratio. Is there anything else? Should I run SBC (Talts et al.) — and if so, with what scope (just $C_s$, or all parameters)?
   - **For rating distribution:** I have lppd / elpd_loo via PSIS-LOO, and per-indicator $\chi^2$ histograms vs predictions. The PPC visualisations I do per (rater, indicator) feel okay but ad hoc. What is the *principled* thing here? Is there a single scalar (or small set) that catches "the model is qualitatively predicting wrong rating shapes" so a coding agent can flag regression automatically?
   - **For internal-node parameter recovery:** I have $\beta^{\mathrm{pres}}, \beta^{\mathrm{abs}}$ recovery (bias / coverage). Should I also recover marginal $q_v$ at internal nodes? Anything else?

3. **Transmission-gap hypothesis.** Is hypothesis (2) above consistent with the empirical pattern (asymmetric prior fixes the *prior* gap structurally but doesn't move *posterior* contraction)? What experiment in the toy ladder would cleanly adjudicate whether the binding constraint is (a) the prior on $\boldsymbol{\beta}$, (b) the data not having curvature information about $\boldsymbol{\beta}$, (c) the joint $(C_s, \boldsymbol{\beta})$ identifiability — the marginal $q_j$ depends on $C_s$ through a chain of $\boldsymbol{\beta}$ products, so $C_s$ and $\boldsymbol{\beta}$ may co-trade?

4. **Order of adding features.** Given the three hypotheses and the proposed ladder, what ordering most efficiently localises the failure? Are there features I should add earlier than I've sketched? Later? Are there pairs of features whose interaction matters and that I should add together?

5. **Anything else you'd consider.** I'm stuck and would value an outside view. If you think the right experiment is *not* a "minimal model + ladder" approach, say so. If you think a different diagnostic — e.g. an analytical Fisher-information walk over candidate ladder rungs, before any fitting at all — would be higher leverage, say so.

## Constraints on your response

- I have approximately two weeks of focused engineering time. Recommendations need to be implementable on that scale.
- I will hand off your recommendations to a coding agent for implementation; please make experimental protocols concrete enough to be coded directly (specify tree topology, true parameter values, sample sizes, fitting backend assumptions, exact metrics to compute and thresholds to compare against).
- It's fine — and useful — to flag where you're uncertain or where my framing is wrong.
- A 3–5 page response is fine. Show your reasoning where it informs the recommendation. Do not pad with general-purpose Bayesian-modelling advice; assume working knowledge of HMC, ordinal regression, identifiability, and PSIS-LOO.
