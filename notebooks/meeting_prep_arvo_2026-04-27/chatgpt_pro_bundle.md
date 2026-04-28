# DCM tree-prior intervention — context bundle

> Companion to the prompt. Self-contained. Sections 1–3 set up the model;
> section 4 is the diagnosis; section 5 is the intervention; section 6 the
> results; section 7 the honest limits.

---

## 1. Project context

The Digital Consciousness Model (DCM; Shiller, Duffy, Muñoz Morán, Moret, Percy & Clatterbuck, 2026 — arXiv:2601.17060) is Rethink Priorities' Bayesian hierarchical model that aggregates expert evidence about whether AI systems may be conscious. It is evaluated under 13 different *stances* (theories of consciousness): Global Workspace Theory (GWT), Higher-Order Theory, Recurrent Processing Theory (Pure / Perceptual), Integrated Information Theory, Attention Schema Theory, Embodied Agency, Simple Valence, Cognitive Complexity, Computational Analogy, Field Mechanisms, Person-like, Biological Analogy. Each stance has its own tree of features → subfeatures → indicators. The results in the initial paper aggregate evidence against 2024-LLM consciousness overall, but not decisively so.

The four reference systems in the current data are: Human (1 expert; rated all systems = "cross-system rater" `E_cross`), 2024 leading chat LLMs (4 experts including the cross-system rater), Chicken (2 experts including the cross-system rater), ELIZA (1 expert; the cross-system rater).

I am a SPAR fellow on the model. My remit is the **bottom (expert observation) layer**. Almost all my fitting / diagnostic work has been on **GWT** because that is the most-completed stance with the richest data; it has 50 indicators on the existing tree.

---

## 2. Arvo's original brief for the bottom layer (verbatim, lightly trimmed)

> Data Format. We could get experts to give, per indicator question on whether it's present: full distribution drawn / confidence interval e.g. 90% + central estimate (we did this) / central pr estimate (we ended up using this) / Likert 7 points (maybe we should do this) / Likert 5 points / Yes/no present. Feedback was that what we did was much too taxing, so I'm now thinking a Likert scale with 7 points makes sense for future data.
>
> Top Layer — Consciousness. The top of the model is a binary variable $C_i$ representing whether system $i$ is conscious. Rather than fixing a point estimate for the prior probability of consciousness, I'd place a Beta hyperprior on $\pi_0$.
>
> Reference systems as soft ground truth. Humans at Bernoulli(0.99999), thermostats at the opposite extreme. Octopuses at Bernoulli(0.8). The reference data updates the posterior on $\pi_0$ and all lower-layer parameters.
>
> Bottom Layer — Experts. Each expert reports $r$ on a 7-point ordinal scale for each indicator of each system. I would probably model this using an ordinal probit. Latent signal $s = \delta_k^{(Z)} + \varepsilon$, where $\delta_k^{(1)}$ is the average signal expert $k$ has when the indicator is truly present, $\delta_k^{(0)}$ when absent, $\varepsilon$ Gaussian. Six ordered cutpoints $\gamma_1, \dots, \gamma_6$ carve $s$ into 7 categories. The likelihood is the probability that $s$ falls in the corresponding interval. Parameters $\delta_k^{(1)}, \delta_k^{(0)}$ capture each expert's accuracy and bias, learnt from reference systems. Cutpoints can be shared across experts or made expert-specific.
>
> [Arvo also notes the SDT analogue and DeCarlo (1998) on ordered probit as SDT-with-rating-responses.]

This shaped the *framing* of the bottom layer (ordinal probit, expert-specific shifts, reference systems as soft ground truth). Where I have diverged from Arvo's brief, the reasons are noted below in §3 and §5.

---

## 3. The DCM, formally

### 3.1 Tree above the indicators (paper version)

For each stance $t$ and system $s$, there is a binary variable $C_s$ (consciousness). For each non-leaf node $j$ in the stance tree (feature, subfeature) there is an unobserved Bernoulli variable $z_{s,j} \in \{0, 1\}$. Every directed parent–child edge $i \to j$ carries two labels (set by RP from a literature review): a *support* label $\sigma_{ij} \in \{\text{overwhelming, strong, moderate, weak}\} \times \{\text{support, no bearing, undermining}\}$ and a *demandingness* label $\delta_{ij} \in \{\text{overwhelmingly, strongly, moderately, weakly}\} \times \{\text{demanding, neutral, undemanding}\}$.

The conditional probability of child = present given parent is parameterised by two Beta priors:

$$
\beta^{\text{pres}}_{ij} \sim \text{Beta}(\alpha^p_{ij}, \beta^p_{ij}),
\qquad
\beta^{\text{abs}}_{ij}  \sim \text{Beta}(\alpha^a_{ij}, \beta^a_{ij})
$$

with the hyperparameters $(\alpha^p, \beta^p, \alpha^a, \beta^a)$ set deterministically from $(\sigma_{ij}, \delta_{ij})$ via the paper's lookup table. Conceptually $\beta^{\text{pres}}_{ij} \approx P(z_{s,j} = 1 \mid z_{s,i} = 1)$ and $\beta^{\text{abs}}_{ij} \approx P(z_{s,j} = 1 \mid z_{s,i} = 0)$. Concentration $\alpha + \beta$ is fixed to `NODE_CONCENTRATION = 10` (paper choice).

The lookup table (paper, Table 1+2):

```
Support pseudocounts:                      Demandingness pseudocounts:
 overwhelming support  (50, 1)             overwhelmingly demanding  (50, 1)
 strong support        ( 8, 1)             strongly demanding         ( 8, 1)
 moderate support      ( 3, 1)             moderately demanding       ( 3, 1)
 weak support          (1.5, 1)            weakly demanding           (1.5, 1)
 no bearing            ( 1, 1)             neutral                    ( 1, 1)
 weak undermining      ( 1, 1.5)           weakly undemanding         ( 1, 1.5)
 ...                                       ...
```

with a small "tweak": the support factor is multiplied by a demandingness-derived factor (`STRONG`, `MODERATE`, `WEAK` from the demandingness table) so that "strong support + strongly demanding" is sharper than "strong support + weakly demanding". See `EvidenceProcessor._get_support_factor` in §5.3 for the exact code. Demandingness alone determines the absent-state Beta's pseudocounts.

### 3.2 The indicator → consciousness propagation

The original paper marginalises the discrete `z_{s,j}` Bernoullis by repeated outer-loop simulation. I rewrote this analytically (Rao-Blackwell): the continuous probability $q_{s,j} = P(z_{s,j} = 1 \mid \text{tree above})$ is propagated downward as

$$
q_{s,j}(\boldsymbol\beta) = q_{s, \mathrm{pa}(j)} \cdot \beta^{\text{pres}}_{\mathrm{pa}(j) \to j} + (1 - q_{s, \mathrm{pa}(j)}) \cdot \beta^{\text{abs}}_{\mathrm{pa}(j) \to j}
$$

with $q_{s, \mathrm{root}} = C_s$. This keeps the entire model continuous so NUTS can sample the joint posterior (over $C_s$, all $\boldsymbol\beta$, and the emission parameters) in one go.

Per-indicator, fixing $\boldsymbol\beta$ at posterior draws and varying $C_s$, the implied probability is **affine** in $C_s$:

$$
q_{s,j}(C_s) = \alpha_j(\boldsymbol\beta) + \delta_j(\boldsymbol\beta) \cdot C_s, \qquad
\delta_j = \prod_{(i \to j) \in \mathrm{path}(\mathrm{root} \to j)} (\beta^{\text{pres}}_{i \to j} - \beta^{\text{abs}}_{i \to j}).
$$

$\delta_j$ is the *propagation slope* — the share of the root anchor gap (≈ 0.998 for Human vs ELIZA) that reaches indicator $j$. **It is the central diagnostic in §4.**

### 3.3 Bottom layer (my work)

For indicator $j$, system $s$, expert $e$, the latent signal is

$$
u_{e,s,j} = b_e + a \cdot z_{s,j} + \varepsilon_{e,s,j}, \qquad \varepsilon \sim \mathcal{N}(0, 1)
$$

with shared discrimination $a$, expert-specific shift $b_e$ (one anchor pinned for identifiability), shared cutpoints $\boldsymbol\kappa = (\kappa_0, \dots, \kappa_5)$, observed rating $r_{e,s,j} \in \{1, \dots, 7\}$ via $r = k \iff \kappa_{k-1} < u \leq \kappa_k$. Marginalising the Bernoulli $z_{s,j}$ analytically, the rating likelihood becomes the binary mixture

$$
P(r_{e,s,j} = k \mid \boldsymbol\theta) = q_{s,j} \cdot P_{\mathrm{OP}}(k \mid b_e + a, \boldsymbol\kappa) + (1 - q_{s,j}) \cdot P_{\mathrm{OP}}(k \mid b_e, \boldsymbol\kappa).
$$

**Three changes to the leaf I have made beyond Arvo's brief, with reasons:**

1. **Three-state latent indicator** (binary mixture → three-state mixture with weights $\pi_m(q) = ((1-q)^2, 2q(1-q), q^2)$ and three component shifts $\eta_0 = 0, \eta_1 = a/2, \eta_2 = a$). Reason: when most observed ratings concentrate at the extremes (Human ≈ 86% top, ELIZA ≈ 96% bottom), a strict binary mixture cannot put enough mass on those extreme categories at any cutpoint configuration. Three-state lifts the emission ceiling. **It is a low-rank graded approximation, not a literal claim that indicators are 3-valued.** The component weights interpretation is purely the binomial expansion of $(q + (1-q))^2$, which can be read as "two latent presence draws, with $m$ counting how many came up present"; not a substantive position.

2. **Soft reference anchors** (Beta(50, 1) on $C_{\text{Human}}$, Beta(1, 50) on $C_{\text{ELIZA}}$ instead of fixed 0.999 / 0.001). Diagnostic: posteriors land at 0.990 / 0.010 — *closer to the hard anchors than to the prior means*. The hard anchors are not load-bearing.

3. **Per-expert noise scale (`σ_e`) — attempted, rejected.** With a single rater on Human and a single rater on ELIZA (one rater is the cross-system rater, who rates all four systems), `σ_e` is unidentifiable on the reference cells. Failure was identifiability, not sampler.

---

## 4. Diagnosis — why I concluded the tree is the bottleneck

The original PPC under the binary leaf failed worst on `E_cross × Human` and `E_cross × ELIZA`: predicted top-rate 0.47 vs observed 0.86, predicted bottom-rate 0.36 vs observed 0.96. Initial reaction was "fix the leaf." I wrote three diagnostics that together reframed the issue.

### 4.1 Per-indicator posterior propagation slope $\delta_j$

Under `baseline_bin` (binary leaf, paper tree, joint anchored fit), I computed $\delta_j$ per indicator from posterior draws. **Mean $\delta_j = 0.062$, median $0.058$, max $0.246$.** Stratified by depth:

| depth | n | mean δ_prior (paper) | mean δ_post | mean q at C=0.999 | mean q at C=0.001 |
|---|---|---|---|---|---|
| 2 | 9  | 0.091 | 0.128 | 0.687 | 0.542 |
| 3 | 41 | 0.037 | 0.047 | 0.630 | 0.571 |

Tree-implied indicator probability at the Human reference cell averages 0.64; at ELIZA, 0.57 — a 7-percentage-point separation on a root anchor gap of ~1.0. The leaf likelihood is doing essentially all of the anchoring work via the emission cutpoints; the tree is delivering ~5–13% of the root signal to the indicator layer.

### 4.2 Structural vs data-induced regime

For each indicator I compared posterior $\delta_j$ to its prior expectation (paper Beta means, $g = 1$):

$$
\log r_j = \log(|\delta_j^{\text{post}}| + \varepsilon) - \log(|\delta_j^{\text{prior}}| + \varepsilon).
$$

**Result: median $\log r_j = +0.20$, mean $+0.16$**, with a roughly symmetric distribution around zero plus 5 sign-flips on a single subfeature ancestor. The posterior is *not* flattening toward $\beta = 0.5$ (which would indicate sparse-data shrinkage to uninformative priors). It tracks the paper's prior means with modest data-driven amplification.

**Reading.** We are in the *structural* regime: the small $\delta_j$ are not symptoms of sparse data, they are properties of the label-mapping semantics compounded over depth. Prior revision (rather than more data) is the appropriate intervention.

### 4.3 Prior-predictive Monte Carlo (decisive)

Sampled $10^4$ tree realisations from the paper's per-label Beta priors (no ratings observed) and propagated through the GWT tree at $C \in \{0.001, 0.5, 0.999\}$. Marginal of $q_j$ per indicator:

```
Prior-predictive q_j medians (paper priors, no data):
 C       n   mean    median   min     max
 0.001   50  0.595   0.633    0.286   0.842
 0.500   50  0.619   0.658    0.298   0.860
 0.999   50  0.643   0.678    0.288   0.880

Prior anchor separation: mean q(0.999) − mean q(0.001) = 0.048
Root anchor gap: 0.998
=> Prior alone supports only ~5% of the root anchor gap at the indicator layer.
```

**The paper's tree prior itself produces ~0.60 indicator probability at $C = 0.001$ and ~0.64 at $C = 0.999$.** No amount of reference-rating data can pull the indicator-layer posterior far from this band without fighting the prior hard. This is the structural ceiling.

### 4.4 Cross-expert scatter check

Plotting the 50 indicators' tree-implied $(q_j(0.001), q_j(0.999))$ and colouring those observed by `E_cross` at Human and ELIZA shows the failing reference cells are **not** outliers in propagation space. They sit on the same attenuation structure as the rest of the tree. This is a global tree issue, not a rater-specific cutpoint issue masquerading as tree compression.

---

## 5. The intervention: complete-pooling-within-label (`POOL_BETAS_BY_LABEL`)

### 5.1 Motivation

Given §4: prior revision is structurally required. Two paths:

- **Sharpen the per-node prior** (transmission gain — sharpen Beta means in logit space without changing concentration). Tested: see §6.5. Gets `Chicken_C` posterior shifted by +0.32 and `LLMs_C` by +0.22 from baseline; on data-fit grounds, partially closes the focus-cell tails but at large free-system cost.
- **Replace the rigid label → Beta lookup with a model that lets the *labels themselves* be calibrated to the data** — keeping the labels (RP's literature-review judgements) but loosening the assumption that each (support, demandingness) combination has a fixed Beta(α(σ,δ), β(σ,δ)). This is the pooling intervention.

### 5.2 Formal definition

Let $G_{\text{pres}} = \{(\sigma, \delta) : \exists \text{ node with these labels}\}$ and $G_{\text{abs}} = \{\delta : \dots\}$. (Observed counts in GWT: 18 pres groups, 7 abs groups; 14 pres groups have ≥ 3 nodes, 4 are singletons; abs groups all have ≥ 6 nodes except one singleton.)

For each $(\sigma, \delta) \in G_{\text{pres}}$:

$$
\beta^{\text{pres}}_{(\sigma, \delta)} = \mathrm{logistic}\!\left( \mathrm{logit}\,\mu_{\sigma, \delta}^{\text{paper}} + \sigma_{\text{pool}} \cdot \tilde\beta^{\text{pres}}_{(\sigma, \delta)} \right), \qquad \tilde\beta^{\text{pres}}_{(\sigma, \delta)} \sim \mathcal{N}(0, 1).
$$

Similarly for each $\delta \in G_{\text{abs}}$:

$$
\beta^{\text{abs}}_{\delta} = \mathrm{logistic}\!\left( \mathrm{logit}\,\mu_{\delta}^{\text{paper, abs}} + \sigma_{\text{pool}} \cdot \tilde\beta^{\text{abs}}_{\delta} \right).
$$

For every tree edge $i \to j$ with labels $(\sigma_{ij}, \delta_{ij})$, the conditional probabilities $\beta^{\text{pres}}_{ij}, \beta^{\text{abs}}_{ij}$ are taken to be the group-level $\beta^{\text{pres}}_{(\sigma_{ij}, \delta_{ij})}, \beta^{\text{abs}}_{\delta_{ij}}$ — same realised value across all same-label edges.

Default pool $\sigma_{\text{pool}} = 0.5$. This is a **complete-pooling** model in the (σ, δ) groups: all same-label edges share one realised β. Means (centres) are the paper's; only the prior family and the sharing structure change.

### 5.3 The relevant code

#### 5.3.1 Paper's lookup (preserved exactly when `POOL_BETAS_BY_LABEL=False, TRANSMISSION_GAIN=1`)

```python
class EvidenceProcessor:
    def get_beta_parameters(self, support, demandingness):
        absence_alpha, absence_beta = self._get_demandingness_parameters(demandingness)
        support_factor = self._get_support_factor(support, demandingness)

        presence_alpha = int(absence_alpha * support_factor[0])
        presence_beta  = int(absence_beta  * support_factor[1])

        c = self.config.NODE_CONCENTRATION  # 10
        alpha_p = presence_alpha * c / (presence_alpha + presence_beta)
        beta_p  = presence_beta  * c / (presence_alpha + presence_beta)
        alpha_a = absence_alpha  * c / (absence_alpha  + absence_beta)
        beta_a  = absence_beta   * c / (absence_alpha  + absence_beta)
        return alpha_p, beta_p, alpha_a, beta_a

    def _get_demandingness_parameters(self, demandingness):
        base = self.config.BASE  # 1
        return {
            "overwhelmingly demanding": (base, int(base * 50)),
            "strongly demanding":       (base, int(base *  8)),
            "moderately demanding":     (base, int(base *  3)),
            "weakly demanding":         (base, int(base * 1.5)),
            "neutral":                  (base, base),
            "weakly undemanding":       (int(base * 1.5), base),
            "moderately undemanding":   (int(base *  3),  base),
            "strongly undemanding":     (int(base *  8),  base),
            "overwhelmingly undemanding": (int(base * 50), base),
        }[demandingness]

    def _get_support_factor(self, support, demandingness):
        # Demandingness-modulated support: "strong support + strongly demanding"
        # is sharper than "strong support + weakly demanding".
        demandingness_factor = {
            "overwhelmingly demanding": self.config.STRONG,    # 8
            "strongly demanding":       self.config.MODERATE,  # 3
            "moderately demanding":     self.config.WEAK,      # 1.5
            "weakly demanding":         1,
        }.get(demandingness, 1)
        return {
            "overwhelming support": (50 * demandingness_factor, 1),
            "strong support":       ( 8 * demandingness_factor, 1),
            "moderate support":     ( 3 * demandingness_factor, 1),
            "weak support":         (1.5 * demandingness_factor, 1),
            "no bearing":           (1, 1),
            "weak undermining":     (1, 1.5),
            "moderate undermining": (1, 3),
            "strong undermining":   (1, 8),
            "overwhelming undermining": (1, 50),
        }[support]
```

#### 5.3.2 Pooling — non-centred logit-Normal hyperparameters

```python
def build_label_pool_hyperparameters(config, evidence_processor, stance_data):
    """
    For each (support, demandingness) group in the tree, instantiate ONE
    logit-Normal beta_pres centred on the paper Beta's prior mean.
    For each demandingness group, instantiate ONE logit-Normal beta_abs.
    All same-label tree edges then SHARE that single value.
    """
    pres_groups, abs_groups = collect_tree_label_groups(stance_data)
    sigma = config.LABEL_POOL_SIGMA  # 0.5

    beta_pres_by_group = {}
    for (s, d) in pres_groups:
        alpha_p, beta_p, _, _ = evidence_processor.get_beta_parameters(s, d)
        paper_mu = alpha_p / (alpha_p + beta_p)
        raw = pm.Normal(f"beta_pres_tilde__{_san(s)}__{_san(d)}", mu=0., sigma=1.)
        logit_beta = pt.constant(_logit_np(paper_mu)) + sigma * raw
        beta_pres_by_group[(s, d)] = pm.Deterministic(
            f"beta_pres__{_san(s)}__{_san(d)}", pt.sigmoid(logit_beta)
        )

    beta_abs_by_group = {}
    for d in abs_groups:
        # NB: pulls absence pseudocounts from "no bearing" support — which is
        # the paper's parameterisation: absence depends only on demandingness.
        _, _, alpha_a, beta_a = evidence_processor.get_beta_parameters("no bearing", d)
        paper_mu = alpha_a / (alpha_a + beta_a)
        raw = pm.Normal(f"beta_abs_tilde__{_san(d)}", mu=0., sigma=1.)
        logit_beta = pt.constant(_logit_np(paper_mu)) + sigma * raw
        beta_abs_by_group[d] = pm.Deterministic(
            f"beta_abs__{_san(d)}", pt.sigmoid(logit_beta)
        )

    return beta_pres_by_group, beta_abs_by_group
```

#### 5.3.3 Tree node uses the pooled beta when flag is on

```python
def _create_node_variable(self, evidencer, parent_prob, ancestor_path):
    support = evidencer.get("support", "no bearing")
    demand  = evidencer.get("demandingness", "neutral")

    if self.config.POOL_BETAS_BY_LABEL:
        beta_present = self.beta_pres_by_group[(support, demand)]
        beta_absent  = self.beta_abs_by_group[demand]
    elif self.config.GAIN_LOGIT_NORMAL and self.config.TRANSMISSION_GAIN != 1.0:
        # "safe-gain": node-level logit-Normal centred at gained means.
        mu_p, mu_a = self.evidence_processor.get_gained_means(support, demand)
        sigma = self.config.GAIN_LOGIT_NORMAL_SIGMA
        beta_present = build_safe_gain_node_beta(name, "pres", mu_p, sigma)
        beta_absent  = build_safe_gain_node_beta(name, "abs",  mu_a, sigma)
    else:
        # Paper: node-level Beta priors from get_beta_parameters().
        a_p, b_p, a_a, b_a = self.evidence_processor.get_beta_parameters(support, demand)
        beta_present = pm.Beta(f"{name}_beta_pres", alpha=a_p, beta=b_p)
        beta_absent  = pm.Beta(f"{name}_beta_abs",  alpha=a_a, beta=b_a)

    q_j = pm.Deterministic(
        f"{name}_p",
        parent_prob * beta_present + (1 - parent_prob) * beta_absent,
    )
    ...
```

### 5.4 Comparators tested

- **`baseline_bin`** — paper tree (per-node Beta priors), binary leaf, joint anchored fit. Validated reporting baseline.
- **`baseline_3s`** — paper tree, three-state leaf.
- **`pool_bin`** — `POOL_BETAS_BY_LABEL`, binary leaf.
- **`pool_3s`** — `POOL_BETAS_BY_LABEL`, three-state leaf. **Leading library candidate.**
- **`gain` (g = 2.5, Beta)** — Beta priors with means recentred via symmetric log-odds-gap rescaling at gain $g = 2.5$. Concentration preserved at 10. **Failed: 1072 divergences.** Mechanism: at $g = 2.5$, 11/18 GWT label groups have $\min(\alpha, \beta) < 1$, producing density spikes near 0 or 1 in the Beta priors that NUTS cannot traverse (52/75 tree nodes affected).
- **`safegain_3s` (g = 2.5, logit-Normal)** — same gain, but per-node logit-Normal($\sigma = 0.5$) centred on the gained mean instead of Beta. Repairs the divergence; Tier 1 clean. The "what if we believe the paper labels literally" comparator.

---

## 6. Results

### 6.1 Sampling diagnostics

```
fit            divergences   max R-hat   min ESS bulk
baseline_bin            0       1.00          6220
baseline_3s             0       1.00          5422
pool_bin                0       1.00          9428
pool_3s                 0       1.00          4533
gain (Beta)          1072       1.00          6890   <-- REJECTED
safegain_3s             0       1.00          3687
```

### 6.2 System C posteriors (all stances under joint-anchored fit)

```
system   baseline_bin       baseline_3s        pool_bin           pool_3s            safegain_3s
Human    0.999 (fixed)      0.999              0.999              0.999              0.999
Chicken  0.249 [0.02, 0.65] 0.297 [0.02, 0.70] 0.297 [0.02, 0.71] 0.362 [0.04, 0.72] 0.621 [0.34, 0.85]
LLMs     0.081 [0.00, 0.36] 0.118 [0.01, 0.42] 0.068 [0.00, 0.31] 0.170 [0.01, 0.51] 0.342 [0.12, 0.56]
ELIZA    0.001 (fixed)      0.001              0.001              0.001              0.001
```

System ordering Human ≫ Chicken > LLMs ≫ ELIZA preserved everywhere. `pool_3s` shifts Chicken/LLMs by +0.07/+0.05 relative to the three-state baseline — well within a ±0.08 "don't destabilise the reporting baseline" guardrail. `safegain_3s` shifts free systems substantially (Chicken +0.32, LLMs +0.22) — sharp prior pulling softly-informed evidence up.

### 6.3 Per-indicator propagation slope δ_j

```
fit            mean δ_j   median δ_j   max δ_j
baseline_bin     0.062      0.058       0.346
baseline_3s      0.075      0.070       0.357
pool_bin         0.122      0.135       0.395
pool_3s          0.155      0.179       0.439
safegain_3s      0.334      0.353       0.826
```

Pooling roughly doubles the propagation; safe-gain quadruples it.

### 6.4 Focus-cell PPC closure (vs matching-leaf baseline)

```
                                                  Δ_baseline   Δ_pool_3s   closure
E_cross × Human    (right tail, observed 0.86)    -0.228       -0.163      +28.3%
E_cross × ELIZA    (left tail,  observed 0.96)    -0.344       -0.257      +25.5%
E_chickenB × Chicken (guardrail right tail)       +0.176       +0.140      +20.4%
E_llmC × LLMs        (guardrail right tail)       +0.113       +0.110       +2.7%
```

`pool_3s` materially closes both focus tails without inducing wrong-tail pathology at the guardrail cells.

### 6.5 2×2 factorial: leaf × tree-prior interaction

For each focus cell, $I_M = M_{\text{pool\_3s}} - M_{\text{baseline\_3s}} - M_{\text{pool\_bin}} + M_{\text{baseline\_bin}}$, with $M = -|\Delta|$ (larger = better PPC fit).

```
cell                       I_M       interpretation
E_cross × Human (right)   +0.002    additive
E_cross × ELIZA  (left)   +0.105    synergistic
```

Three-state and pooling compose cleanly. ELIZA is synergistic — neither fix alone gets close, the combination crosses the threshold.

### 6.6 Pool decomposition concern (the honest finding)

The five "sign-flip" indicators in §4.2 all share one subfeature ancestor labelled (`weak undermining`, `neutral`). Under pool_bin, the implied label_delta = $\beta^{\text{pres}}_{(\text{wk und, neutral})} - \beta^{\text{abs}}_{\text{neutral}}$ moves from paper $-0.07$ to posterior $+0.15$, ostensibly "resolving" the sign issue. But decomposing:

```
                         | paper |  posterior median (94% HDI)        | shift
beta_pres (wk_und, neut) | 0.429 |  0.441 [0.230, 0.672]              | +0.012
beta_abs  (neutral)      | 0.500 |  0.293 [0.164, 0.461]              | -0.207
label_delta              | -0.071|  +0.145 [-0.123, +0.424]
```

The shared `β_abs__neutral` parameter is informed by data from the 10 other neutral-demandingness nodes (mostly strong/moderate-support) and drops 0.50 → 0.29. The weak-undermining group inherits this without itself providing direct evidence (singleton, 1 node).

**Reading I'm currently giving this**: it is *not* "the sign-flip is fixed." It is "under pooling, the data wants the neutral-demandingness absence baseline (a *paper-prior* parameter) lower than 0.50, and that's a legitimate Bayesian update — but it is not specific evidence that the weak-undermining label assignment is wrong." This is one of the questions for ChatGPT.

### 6.7 Soft anchors corroborate the modelling choices (B.13)

Refit `pool_3s` with soft anchors $C_{\text{Human}} \sim \text{Beta}(50, 1)$ (prior mean 0.980), $C_{\text{ELIZA}} \sim \text{Beta}(1, 50)$ (prior mean 0.020). Posterior: $C_{\text{Human}} = 0.990$ (above prior mean, just below hard anchor), $C_{\text{ELIZA}} = 0.010$ (below prior mean, just above hard anchor). Chicken/LLM posteriors and focus-cell PPCs essentially unchanged. Hard anchors are not load-bearing.

### 6.8 Oracle emission ceiling (B.11c — sharper claim about residuals)

Even granting an oracle tree ($q_{s,j} \to 1$ for Human / $q_{s,j} \to 0$ for ELIZA on every indicator), the shared ordered-probit emission has hard ceilings on what extreme-category mass it can produce:

$$
P(r=1 \mid m=0) \leq \Phi(\kappa_0), \qquad P(r=7 \mid m=2) \leq \Phi(a - \kappa_5).
$$

Under `pool_3s` posterior median: ceiling at 0.880 (ELIZA) and 0.871 (Human). Required probit-scale gaps to *centre* the expected outer-rate on observed: Human needs $a - \kappa_5 = \Phi^{-1}(0.86) = 1.080$, posterior is $1.131$ [0.54, 1.74] (sits at posterior median, soluble on ~57% of draws). ELIZA needs $\kappa_0 = \Phi^{-1}(0.96) = 1.751$, posterior is $1.174$ [0.73, 1.69] — required value *exceeds the 97th percentile* (soluble on ~2% of draws).

**Reading.** Tree-side fixes can plausibly close the Human residual and ~70% of the ELIZA residual; the last ~0.08 of the ELIZA gap is structurally imposed by $\kappa_0$ under the single-cross-system-rater identification bottleneck. Honest emission/identifiability-misspecification framing is the correct writeup, not more tree tuning.

---

## 7. Honest limits / where I am still uncertain

1. **"Pooling" here is *complete* pooling within label, not partial pooling with node-level residuals.** I tried proper partial pooling in a draft but the PyTensor C-backend graph cost was prohibitive. Numba/JAX backend (Edwin's parallel work) could unblock this. The complete-pooling result is informative but is a stronger sharing assumption than I would otherwise pick.

2. **`pool_3s` does not solve the Human/ELIZA reference PPC tails fully** — closes ~25–28%, residual remains. §6.8 argues the residual is split between irreducible emission (~31% of the ELIZA gap) and tree-side ($q_j$ still ~0.7 at oracle Human, blocked by structural attenuation under pool).

3. **The shared-β_abs decomposition (§6.6) is the honest finding I most want pressure-tested.** I have positioned it as legitimate Bayesian behaviour given the paper's parameterisation; it could equally be argued as evidence the parameterisation itself is wrong (β_abs should depend on (σ, δ), not δ alone).

4. **Free-system C posteriors are sensitive to the top-level Beta(1, 5) stance prior.** Under weak tree (baseline, pool_bin), LLM posterior median sits *below* prior median; under strong tree (pool_3s, safegain_3s) it sits above. A stance-prior sensitivity sweep is the next-week priority and would disentangle "stance prior" from "tree" contributions.

5. **Per-expert noise scale (`σ_e`) and expert-specific cutpoints failed for the same identification reason.** Single cross-system rater on Human/ELIZA is the binding bottleneck. Real fix is more / overlapping reference data — outside SPAR scope.

6. **All my fits are GWT-only.** Generalisation to the other 12 stances requires re-running and re-diagnosing. Compatibility note: `pool_3s` is structure-agnostic — it regroups by whatever labels sit on the tree, so it should re-fit cleanly on Matilda's restructured tree when that lands.

7. **Matilda's tree restructuring (independent stream)** moves to a different feature/sub-feature layout based on Marr's levels of explanation, including conjunctive level-2 dependencies (a + b → c). Conjunctive dependencies would be a substantively bigger model change than `pool_3s` and are currently unsupported. Compatibility: the *labels* are preserved on her restructured trees; her changes are upstream of mine and would inherit `pool_3s`'s benefits if both land.

---

## 8. References (for ChatGPT to verify if needed)

- Shiller, Duffy, Muñoz Morán, Moret, Percy, Clatterbuck (2026). *Initial results of the Digital Consciousness Model.* arXiv:2601.17060.
- DeCarlo (1998). *Signal detection theory and generalized linear models.* Psychological Methods 3(2): 186–205.
- Paulewicz & Blaut (2020). *The bhsdtr package.* R package documentation.
- Butlin et al. (2025). *Identifying indicators of consciousness in AI systems.* Trends in Cognitive Sciences.
