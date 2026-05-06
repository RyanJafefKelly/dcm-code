# DCM synthetic validation and identifiability bundle

Copy this whole file into GPT-5.5 Pro with extended reasoning. It is meant to
give enough project context for a statistically serious plan, not just generic
Bayesian workflow advice.

---

## 0. What I want from Pro

I am starting a two-day work block on synthetic-data validation / sensitivity
analysis for the Digital Consciousness Model (DCM), focused on the Global
Workspace Theory (GWT) stance.

The immediate question from Arvo was roughly: before switching the inner latent
nodes of the DCM tree to continuous variables, run synthetic validations to see
whether the current binary inner latent nodes are identifiable, and how much
interpretational cost there is if they are not.

Please give a principled Bayesian workflow plan. I want pressure on:

1. Which estimands should be primary versus diagnostic.
2. Which metrics best assess recovery / identifiability of:
   - root consciousness probability `C_s`;
   - internal feature/subfeature nodes;
   - tree transmission parameters (`beta_pres`, `beta_abs`);
   - path transmission slopes `delta_j`;
   - ordinal observation-layer parameters.
3. Whether simulation-based calibration is useful here, and for which
   quantities.
4. How to design the synthetic experiments over:
   - sample size;
   - rater-system overlap;
   - root `C_s` values;
   - exact latent-state tree versus composite marginal tree;
   - binary versus three-state indicator leaf;
   - paper priors versus pooled tree priors.
5. What decision criteria would justify saying that internal binary nodes are
   not carrying interpretable recoverable information, so switching them to
   continuous probabilities has little interpretational cost.
6. What is the best achievable analysis in two working days, as distinct from a
   larger ideal study.

Write at peer level. I have a Bayesian statistics background. Skip generic
introductions. Be sceptical and direct.

---

## 1. Project context

The Digital Consciousness Model (DCM) is Rethink Priorities' Bayesian
hierarchical model for aggregating expert evidence about whether systems may be
conscious. It is evaluated under multiple "stances" or theories of
consciousness. Each stance is a tree:

```text
stance/root consciousness C
  -> features
    -> subfeatures
      -> indicators
        -> expert ratings for each system
```

I am a SPAR fellow working with Arvo Munoz Moran. My remit started as the
bottom expert-observation layer: replace the old probability-mean plus
Bernoulli-collapse scheme with an ordinal model for 7-point expert ratings.
While doing that, I found that many fit problems were actually driven by the
tree above the indicators, especially weak root-to-indicator signal
transmission.

Most diagnostics so far are on Global Workspace Theory (GWT), because it is the
most complete stance and has the richest data.

Current GWT data geometry:

```text
GWT tree:
  root stance: 1
  internal feature/subfeature nodes: 25
  indicators: 50
  indicator depths: 9 at depth 2, 41 at depth 3

Ratings:
  rated system-indicator cells: 199
  total ratings: 379
  Human: 50 ratings
  ELIZA: 50 ratings
  Chicken: 93 ratings
  2024 leading chat LLMs: 186 ratings

Experts:
  one cross-system rater ("E_cross"): 150 ratings
  two Chicken-focused raters: 46 and 47 ratings
  three LLM-focused raters: 40, 48, 48 ratings
```

The reference systems in current GWT fits are:

```text
Human: hard anchor C = 0.999 in most fits; one rating per indicator from E_cross
ELIZA: hard anchor C = 0.001 in most fits; one rating per indicator from E_cross
Chicken: free C
2024 leading chat LLMs: free C
```

The sparse and non-overlapping rater-system design is a central identifiability
issue. Human and ELIZA are effectively single-rater reference cells. LLMs and
Chicken have more ratings but their raters mostly do not overlap across
systems, except for E_cross.

---

## 2. Current DCM model

### 2.1 Tree parameters

Each edge from parent node `u` to child node `v` has two conditional
probabilities:

```text
beta_pres[u,v] = P(child present | parent present)
beta_abs[u,v]  = P(child present | parent absent)
```

The paper version sets Beta prior hyperparameters from two qualitative labels
on the edge:

```text
support label:       strong support, moderate support, weak undermining, etc.
demandingness label: strongly demanding, neutral, weakly undemanding, etc.
```

The default paper-style prior has concentration `alpha + beta = 10` at each
edge. The prior mean is a deterministic lookup from `(support, demandingness)`.

The usual propagated marginal probability at child node `v` is:

```text
q_v = beta_abs[u,v] + q_u * (beta_pres[u,v] - beta_abs[u,v])
```

with root `q_root = C_s`.

For an indicator `j`, holding beta fixed, `q_j(C_s)` is affine:

```text
q_j(C_s) = alpha_j + delta_j * C_s

delta_j = product over path root->j of
          (beta_pres[edge] - beta_abs[edge])
```

`delta_j` is the path transmission slope: how much of a root-level difference
in consciousness probability reaches indicator `j`. It has become one of the
most important diagnostics.

### 2.2 Ordinal observation layer

For each rating by expert `e` on system `s`, indicator `j`, the bottom layer is
an ordered probit:

```text
u_e,s,j = b_e + a * z_s,j + epsilon
epsilon ~ Normal(0, 1)
rating is determined by ordered cutpoints kappa_1 ... kappa_6
```

Current baseline variants often set `USE_EXPERT_SHIFTS=False`, so `b_e = 0`,
because the existing rater-system coverage makes expert shifts/scales
high-leverage and weakly identified. There are model flags for expert shifts,
hierarchical expert cutpoints, and expert scales, but these are not the current
default.

The strict binary indicator leaf uses:

```text
z_s,j in {0, 1}
z_s,j ~ Bernoulli(q_s,j)
```

and analytically marginalises `z_s,j` in the likelihood.

The three-state indicator leaf uses:

```text
m_s,j in {0, 1, 2}
m_s,j ~ Binomial(2, q_s,j)
emission centres: 0, a/2, a
expected z = m/2
```

This is not meant as a literal claim that indicators are three-valued. It is a
low-rank graded mixture that helped with extreme reference ratings: Human has
very high top-category mass and ELIZA has very high bottom-category mass.

---

## 3. The key modelling fork: exact latent-state tree vs composite marginal tree

This is central to the current synthetic validation question.

### 3.1 Composite marginal tree

The main implementation in `dcm_model.py` propagates marginal probabilities
`q_j` top-down, then gives each indicator an independent marginal likelihood
conditional on `(C_s, beta, observation parameters)`.

Under this interpretation:

```text
internal feature/subfeature nodes are bookkeeping for marginal indicator probabilities
there are no shared realised internal binary states
sibling indicators are conditionally independent given C_s and beta
```

This is coherent if internal nodes are not meant to be real "on/off" latent
states. It is more naturally compatible with continuous internal quantities.

### 3.2 Exact latent-state tree

The exact-tree implementation in `dcm_model_exact_tree.py` treats internal
nodes as real binary latent states and marginalises them by dynamic programming.

Under this interpretation:

```text
internal feature/subfeature nodes z_v are binary latent states
sibling indicators are correlated through shared ancestors
multiple indicators under the same feature are not independent evidence
```

This exact latent-state tree is more conservative about evidence redundancy and
usually gives lower free-system `C` posteriors.

### 3.3 Why this matters for the synthetic validation

If the exact latent-state tree is the intended DGP, then we should ask whether
the internal binary node states/probabilities can actually be recovered from
realistic data designs.

If the composite marginal tree is the intended DGP, then "internal binary node
recovery" is arguably the wrong target, because the internal nodes are already
just a deterministic parameterisation of marginal probabilities.

The current project question is precisely at this boundary: are the binary
inner nodes meaningful enough to keep as discrete latent states, or are they
mostly arbitrary bookkeeping such that moving them continuous has little
interpretational cost?

---

## 4. Prior diagnostics already run

### 4.1 Paper tree prior has weak root-to-indicator transmission

Prior-predictive Monte Carlo over the GWT tree using the paper's per-edge Beta
priors, before observing ratings:

```text
Sample 10^4 tree beta realisations.
Propagate q_j through the tree at C = 0.001, 0.5, 0.999.

Mean q_j(C=0.999) - q_j(C=0.001) ~= 0.048.
Root anchor gap is ~= 0.998.

So the paper tree prior transmits only about 5% of the root Human-vs-ELIZA
gap to the indicator layer.
```

Depth matters:

```text
depth-2 paths transmit about 13% on average
depth-3 paths transmit about 5% on average
```

This is structural in the paper lookup, not just sparse-data posterior
shrinkage.

### 4.2 Posterior `delta_j` tracks prior `delta_j`

A structural-vs-data-induced check compared posterior `delta_j` to prior
mean `delta_j`:

```text
log r_j = log(|delta_post| + eps) - log(|delta_prior| + eps)

median log r_j ~= +0.20
mean log r_j   ~= +0.16
```

Interpretation so far: the posterior tracks the paper prior with modest
amplification; small `delta_j` is mostly a property of the label mapping
compounded over depth.

### 4.3 Three-state leaf improved reference-cell PPC but inflated `a`

Binary versus three-state GWT comparison:

```text
System C posteriors:
  Chicken:
    binary      0.249 [0.017, 0.655]
    three-state 0.297 [0.022, 0.699]
  LLMs:
    binary      0.081 [0.004, 0.361]
    three-state 0.118 [0.006, 0.419]

PPC focus cells:
  E_cross x Human observed top-category mass: 0.86
    binary predicted top:      0.472
    three-state predicted top: 0.637
  E_cross x ELIZA observed bottom-category mass: 0.96
    binary predicted bottom:      0.352
    three-state predicted bottom: 0.612

Observation discrimination a:
  binary median:      1.913
  three-state median: 4.344
```

The three-state leaf helps, but the large `a` increase flags possible
response-style / emission-side absorption.

---

## 5. Tree-prior interventions already run

### 5.1 Complete pooling within labels

`POOL_BETAS_BY_LABEL=True` replaces independent per-node Beta priors with one
logit-Normal parameter per label group. For `beta_pres`, groups are keyed by
`(support, demandingness)`. Initially, `beta_abs` was keyed by demandingness
alone, matching the paper's absent-state lookup.

The pooled parameterisation is:

```text
logit(beta_group) = logit(paper_prior_mean_group) + sigma_pool * beta_tilde
beta_tilde ~ Normal(0, 1)
sigma_pool = 0.5
```

This is complete pooling within each label group, not proper partial pooling
with node-level residuals. Partial pooling is conceptually preferable but was
blocked by PyTensor graph/compile cost in earlier attempts.

### 5.2 `beta_abs` by `(support, demandingness)`

The first pooled model shared `beta_abs` by demandingness only. This created a
material inheritance effect: the weak-undermining/neutral cluster inherited a
posterior shift in shared `beta_abs__neutral` driven by other neutral-demand
nodes.

A diagnostic refit, `pool_3s_abs_by_sd`, keyed `beta_abs` by
`(support, demandingness)` instead. This reduced that inheritance.

### 5.3 Headline comparison of four GWT fits

Approximate summary from the most recent meeting notes:

```text
fit                         C_Chicken   C_LLMs   mean delta_j
baseline_3s                 0.297       0.118    0.075
pool_3s                     0.362       0.170    0.155
pool_3s_abs_by_sd           0.300       0.127    0.098
exact-tree production       0.252       0.111    0.110
```

The original `pool_3s` looked like the strongest PPC improvement, but part of
that was traced to the shared `beta_abs__neutral` inheritance. The
`pool_3s_abs_by_sd` model is cleaner but less impressive on focus-cell PPC.

### 5.4 Exact-tree production fit

The exact-tree production fit used:

```text
INDICATOR_STATE_MODEL = three_state
POOL_BETAS_BY_LABEL = True
BETA_ABS_BY_SUPPORT_DEMAND = True
LABEL_POOL_SIGMA = 0.5
Human C fixed at 0.999
ELIZA C fixed at 0.001
Chicken and LLM C free
4 chains x 1000 tune x 2000 draws
target_accept = 0.95
```

Diagnostics:

```text
divergences: 0
max R-hat: 1.0000
min ESS bulk: 5826
```

Headline exact-tree posteriors:

```text
Chicken C: 0.252 [0.037, 0.613]
LLM C:     0.111 [0.005, 0.446]
```

The exact tree is more conservative than the composite pooled fits, consistent
with accounting for evidence redundancy through shared internal latent states.

---

## 6. Current scientific question

I need to plan a synthetic-data validation / sensitivity analysis for this
week. The main goal is not to get one more headline fit. It is to understand
recoverability and identifiability.

Main concern:

```text
If internal binary feature/subfeature nodes are not recoverable under realistic
data, then treating them as binary "states" may be arbitrary.

In that case, switching inner nodes to continuous probabilities may have little
interpretational cost and may better match the fact that feature/subfeature
concepts are fuzzy.
```

But root consciousness `C_s` remains crucial. If the synthetic workflow shows
that `C_s` itself is badly biased, poorly calibrated, or non-identifiable under
realistic designs, that is a much more serious model problem.

So the hierarchy of concern is:

```text
Tier 1: root C_s recovery and calibration
Tier 2: path transmission / delta_j recovery, because it controls how evidence
        flows from C to indicators
Tier 3: internal binary node recovery, conditional on exact latent-state tree
Tier 4: beta parameter recovery and observation-layer recovery, mostly as
        diagnostics for where non-identifiability enters
```

Potential synthetic DGPs:

1. Model-correct exact latent-state tree.
   - sample `C_s`;
   - sample beta parameters;
   - sample internal binary states;
   - sample indicator states;
   - sample ordinal ratings.

2. Model-correct composite marginal tree.
   - sample `C_s`;
   - sample beta parameters;
   - propagate q's;
   - sample indicator states independently from q_j;
   - sample ordinal ratings.

3. Fixed-parameter scenario simulations.
   - choose specific plausible beta / observation parameters from posterior
     medians or prior means;
   - vary `C_s`, sample size, rater overlap;
   - evaluate practical recovery under controlled conditions.

4. Prior predictive SBC-style simulations.
   - draw parameters from the model prior;
   - simulate data;
   - refit;
   - evaluate rank calibration for selected estimands.

The practical two-day question is how to choose among these and what metrics to
use.

---

## 7. Candidate metrics I am considering

For continuous parameters / probabilities:

```text
bias: posterior median minus true value
logit-scale bias for probabilities
MAE / RMSE over seeds
credible interval coverage: 50%, 80%, 94%
posterior interval width
prior-to-posterior contraction: posterior SD / prior SD
posterior entropy reduction
rank calibration / SBC ranks
```

For root `C_s` specifically:

```text
coverage and bias for C_s
calibration of Pr(C_s > threshold)
decision recovery at thresholds such as 0.01, 0.05, 0.1, 0.5
ability to distinguish Chicken vs LLM C ordering if relevant
prior sensitivity: Beta(1,5) versus alternatives
```

For internal binary states under the exact tree:

```text
posterior P(z_v = true z_v)
Brier score for P(z_v = 1)
log score for true z_v
entropy reduction: H_prior(z_v) - H_posterior(z_v | y)
classification accuracy at threshold 0.5, with caution
whether posterior is mostly prior-like for realistic data
depth-stratified recovery
fanout-stratified recovery
high-leverage-node recovery, e.g. Selective Attention, Coherence, Complexity
```

For path transmission:

```text
bias / coverage for delta_j
sign recovery for delta_j
recovery of mean delta_j
recovery of q_j(C=0.999) - q_j(C=0.001)
rank correlation of true vs posterior indicator transmission
```

For experiment design:

```text
expected posterior interval width as a function of sample size
expected information gain / entropy reduction for C_s and internal nodes
coverage and RMSE curves over number of experts / overlap design
```

I need Pro to critique which of these are actually the right metrics and which
ones are distractions.

---

## 8. Design axes I am considering

Sample size / data collection:

```text
current design:
  Human and ELIZA: one cross-system rater
  Chicken: two raters
  LLMs: four raters

scaled current design:
  duplicate the same rater-system pattern but with more ratings or more raters
  per cell

balanced overlap design:
  multiple raters rate all systems, especially Human, ELIZA, Chicken, LLMs

reference-rich design:
  add more reference systems or more ratings on Human/ELIZA-like anchors
```

Root C values:

```text
anchors:
  Human near 0.999
  ELIZA near 0.001

free systems:
  low C values: 0.01, 0.05, 0.10
  moderate C values: 0.25, 0.50
```

Tree structures / models:

```text
paper priors, binary leaf
paper priors, three-state leaf
pooled priors, three-state leaf
pooled priors with beta_abs by (support, demandingness)
exact latent-state tree
composite marginal tree
```

Synthetic truth:

```text
truth generated from same model as fitted model
truth generated from exact tree, fitted with composite tree
truth generated from composite tree, fitted with exact tree
truth generated from continuous inner nodes, fitted with binary exact tree
```

The cross-DGP fits may be especially informative for the binary-to-continuous
question, but they may be too much for two days.

---

## 9. Practical constraints

Exact-tree production fit on the full GWT setup takes about 15 minutes for a
4-chain production run on my machine under the current pooled configuration.
Shorter smoke fits are feasible.

A two-day workflow probably cannot do hundreds of full fits across a large grid
unless the fits are reduced, parallelised, or simplified. I need a plan that
separates:

```text
must do in two days
nice to do if time
larger follow-up study
```

I am open to using smaller synthetic trees for formal SBC if full GWT is too
expensive, but I do not want the main answer to be disconnected from the actual
GWT geometry, because depth and sparse rater overlap are exactly the issue.

---

## 10. Codebase landmarks

Useful files in the repository:

```text
dcm_model.py
  Main ordinal DCM implementation.
  Includes ModelConfig, ordinal log-likelihood, three-state leaf,
  MultiSystemModelBuilder for the composite marginal tree.

dcm_model_exact_tree.py
  Exact latent-state tree builder.
  Replaces per-indicator composite potentials with per-system bottom-up
  dynamic-programming potentials.

run_exact_tree_production.py
  Production exact-tree GWT fit under:
    three-state leaf
    pooled beta by label
    beta_abs keyed by (support, demandingness)

gwt_reference_recovery_analysis.py
  Reference-system recovery utilities, q_j(C) propagation, recovery curves.

tree_propagation_diagnostic.py
  Prior-predictive and posterior propagation diagnostics for delta_j.

sigma_pool_prior_predictive_sweep.py
  Prior-predictive sweep over label-pooling sigma.

composite_vs_exact_diagnostic.py
composite_vs_exact_diagnostic_extended.py
  Diagnostics showing the composite marginal likelihood and exact-tree
  likelihood are materially different.

exact_tree_ppc.py
  Proper exact-tree posterior predictive calculations using belief propagation.

test_ordinal_smoke.py
test_three_state_smoke.py
test_reference_recovery_smoke.py
  Existing small synthetic/smoke tests. These are useful scaffolds but are not
  yet a full synthetic validation framework.
```

---

## 11. Specific deliverable I want from Pro

Please return:

1. A concise diagnosis of the statistical problem.
2. A recommended two-day synthetic validation plan with a clear priority order.
3. A table of estimands and metrics, saying which metrics are primary,
   secondary, or misleading.
4. A proposed simulation grid that is ambitious but feasible.
5. Specific decision criteria for:
   - root `C_s` recoverability;
   - internal binary node non-identifiability;
   - whether switching internal nodes to continuous probabilities is
     defensible.
6. Your view on SBC:
   - what to run SBC on;
   - what not to run it on;
   - whether to use the full GWT tree or a reduced tree.
7. The strongest possible critique of my current framing. In particular:
   - Am I over-focusing on internal node "state recovery" when the more
     relevant object is posterior predictive / root C calibration?
   - Are binary internal nodes identifiable only in a degenerate sense because
     they are mostly prior-defined?
   - Would continuous internal nodes change the generative semantics enough
     that synthetic validation under the binary exact tree is the wrong test?
   - Is expected information gain / Bayesian experimental design worth the
     overhead here, or should I stick to recovery curves and coverage?

Be direct. If some proposed metric is a bad idea, say why.

