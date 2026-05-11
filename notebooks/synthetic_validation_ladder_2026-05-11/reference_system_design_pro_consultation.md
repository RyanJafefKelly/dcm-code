# Pro consultation: reference-system / cross-system sharing design for the binary-root DCM

I want your help thinking through the cross-system inference architecture of the DCM, specifically focused on the **reference-system mechanism**, **parameter sharing across systems**, and whether the current implementation captures what was actually intended. The goal — which has not changed — is producing a defensible **per-system root probability**:

$$
\rho_s = p(R_s = 1 \mid y_s),
$$

where $R_s \in \{0, 1\}$ is "system $s$ is conscious under stance $\sigma$." This is what should be reported and validated, per Phase 0. I'm not asking you to revisit that.

What I am asking: given the empirical findings from Phase 1 (transmission gap is binding; β parameters barely move under inference; per-system parameters absorb root signal), what's the right architecture for cross-system inference such that the root probability is identifiable AND defensible? Push back on premises if any of this is wrong.

You won't have repo access. Everything is below.

---

## Project context (compact recap)

The DCM was developed at Rethink Priorities to formalise Bayesian aggregation of expert opinion about whether a system is conscious under each of several theoretical stances on consciousness (Global Workspace Theory, etc.). Each stance defines a hand-coded tree of binary latent features; experts rate the leaf indicators on a 7-point Likert scale. Four target systems on GWT: Human (anchored), ELIZA (anchored), Chicken (free), 2024 chat LLMs (free). 379 ratings total from 6 raters; only one rater (Rater_B) crosses systems, and he covers Human/LLMs/ELIZA but **not Chicken**.

Binary-root semantics (Phase 0 outcome):

$$
\pi_s \sim \mathrm{Beta}(1, 5), \quad R_s \mid \pi_s \sim \mathrm{Bernoulli}(\pi_s).
$$

The headline output is $\rho_s = p(R_s=1 \mid y_s)$. $\pi_s$ is a hyperparameter (per-system but no shared structure across systems in the current implementation), not a recoverable system trait.

Tree edges have $\beta^{\mathrm{pres}}_e, \beta^{\mathrm{abs}}_e$ — production priors from hand-tabulated support/demandingness labels via `EvidenceProcessor.get_beta_parameters`. Concentration $\alpha + \beta = 10$ per edge. 7-category ordered probit observation layer with shared discrimination $a$, shared cutpoints $\boldsymbol{\kappa}$, per-rater shifts $b_e \sim \mathcal{N}(0, 2)$ (one rater anchored at 0). Three-state leaves: $m_j \sim \mathrm{Binomial}(2, q_j)$, emission centres $\eta \in \{0, a/2, a\} + b_e$. Discrete latents marginalised analytically by sum-product DP; NUTS samples a continuous joint posterior over $(\pi_s, \boldsymbol{\beta}, a, \boldsymbol{\kappa}, \mathbf{b})$.

---

## What Phase 1 established empirically

Phase 1C localised the failure mechanism for $\rho_s$ recovery on synthetic data:

- **Top-level analytical bound** (production priors, derived from `EvidenceProcessor` per-label means): $\mathbb{E}[\log B \mid R=1] = +2.81$ nats, $\mathbb{E}[\log B \mid R=0] = -2.81$ nats. Top is fine.
- **Subfeature-observed**: median $\log B \mid R=1 = +1.02$ nats. ~64% of top-bound evidence already gone by depth 2.
- **Leaf-perfect or noisy K=1000**: ~$+0.56$ nats. Information ramp asymptotes — more data does not help under production priors.
- **Per-feature loss** (median $c_{\text{top latent}} \to c_{\text{oracle subtree}}$): Coherence loses 1.45 nats (its "Autonomous Subparts" subfeature has a *negative* β gap, $\mu_p = 0.43, \mu_a = 0.50$); Sel.Attn loses 0.94; Complexity 0.68; Integration ~0 (direct indicator children, no intermediate subfeatures).
- **Edge-profile ablation**: `strong_top_features_only_extreme_lower` PASSES, `weak_top_features_only_extreme_lower` FAILS. Lower-edge structural transmission under strong-top features is the binding constraint.
- **Asymmetric β prior posterior**: even under the asymmetric override applied to the *fitter prior*, the posterior on $\boldsymbol{\beta}$ barely moves from prior — i.e., the data has very little curvature about $\boldsymbol{\beta}$ under the production rater design. β is essentially identified by its prior, not by data.

The proposed fix being validated now (`targeted_strong_lower_override`) overrides lower edges under Coherence/Sel.Attn/Complexity/Integration with logit-Normal priors centred at $(0.90, 0.10)$. This passes the no-fit ladder; the HMC pilot is in progress.

---

## What Arvo (the project lead) originally intended

Quoting near-verbatim from his design notes:

### Top layer

> The top of the model is a binary variable $C_i$ representing whether system $i$ is conscious or not. Rather than fixing a point estimate for the prior probability of consciousness, I'd place a Beta hyperprior on $\pi_0$ — **the base-rate probability that a system is conscious**. For a completely unknown system, I think start with a diffuse prior such as Beta(1, 1) (uniform), expressing total uncertainty about the base rate.

### Reference systems

> I'd be keen to include reference systems, which wouldn't require deterministic ground-truth labels. Instead, I'd assign graded credences — for example, humans at Bernoulli(0.99999), thermostats at the opposite extreme. It's also possible to include other organisms: octopuses at Bernoulli(0.8), dogs at a similar level, and so on. This lets us include a richer set of reference systems without pretending to be certain about borderline cases. **The reference data then updates the posterior on $\pi_0$ and all lower-layer parameters**, whilst a diffuse hyperprior ensures the model doesn't overfit to a (very) small reference class.

### Bottom layer (expert observation)

> The latent signal is $s = \delta_k^{(Z)} + \varepsilon$, where $\delta_k^{(1)}$ is the average signal expert $k$ has when the indicator is truly present (ideally high), $\delta_k^{(0)}$ is the average signal when it's truly absent (ideally low), and $\varepsilon$ is Gaussian noise. Six ordered cutpoints carve the continuous signal into seven response categories. **The parameters $\delta_k^{(1)}$ and $\delta_k^{(0)}$ would capture each expert's accuracy and bias, and are learnt from the reference systems where we know (or have strong credences about) the ground truth.** The cutpoints can be shared across experts or made expert-specific depending on data availability.

---

## Where the current implementation diverged from Arvo's intent

Three specific divergences:

### 1. No shared $\pi_0$ hyperprior across systems

Arvo: $\pi_0 \sim \mathrm{Beta}(1, 1)$ shared; $C_i \mid \pi_0 \sim \mathrm{Bernoulli}(\pi_0)$ per system. Reference systems update $\pi_0$.

Current: per-system $\pi_s \sim \mathrm{Beta}(1, 5)$ independently. **No shared base-rate hyperparameter.** $C_s$ (now $\pi_s$) is per-system from the start, with no cross-system pooling on it. Reference systems update everything *downstream* of $\pi_s$ (the shared β, $a$, $\kappa$, $\mathbf{b}$) via the joint likelihood, but they cannot update a shared $\pi_0$ because there isn't one.

Note that Arvo's design also distinguished the *binary state* $C_i$ from the *base rate* $\pi_0$ — exactly the Phase 0 estimand distinction restated at the design level. The current implementation collapsed these into a single per-system Beta variable that is now (post-Phase-0) recognised as a hyperparameter.

### 2. State-conditional per-rater parameters were collapsed to global $a$ + per-rater shift

Arvo: $\delta_k^{(1)}, \delta_k^{(0)}$ — per-rater, *state-conditional* signal means. Each rater has their own "tendency when indicator is present" vs "when absent."

Current: $a$ (shared scalar, no per-rater variation), plus $b_e$ (per-rater shift, state-*in*dependent). The state effect comes only through $a \cdot (m_j / 2)$, shared across all raters. Per-rater variation is captured only by an additive offset $b_e$ that doesn't depend on the latent state.

So the current observation layer is much less flexible than Arvo's original sketch. Arvo's $(\delta_k^{(1)}, \delta_k^{(0)})$ would have given each rater their own SDT-style $d'$ (= $\delta_k^{(1)} - \delta_k^{(0)}$) plus their own bias. The current model has shared discrimination ($a$) plus per-rater bias only.

### 3. Reference systems implemented as hard or soft anchors, not graded credences with shared base rate

Arvo: graded credences via Bernoulli($p$) for $p \in (0, 1)$, where $p$ represents prior credence the system is conscious. Multiple reference systems (Human, octopuses, dogs, thermostats) at different graded values, all feeding into a shared $\pi_0$.

Current: only two reference systems used in production (Human, ELIZA). Two implementation modes:
- **Hard anchor**: $C_s$ is a `pt.constant` (e.g., 0.999 for Human, 0.001 for ELIZA). Not a random variable at all.
- **Soft anchor**: $C_s \sim \mathrm{Beta}(\alpha_s, \beta_s)$ with tight priors (e.g., Beta(50, 1) for Human, Beta(1, 50) for ELIZA).

No shared base-rate hyperparameter exists, so reference systems can only calibrate downstream parameters via the joint likelihood — they cannot pool toward a shared base rate.

---

## The empirical reality complicating things

Two facts from Phase 1 that bound how much these design choices matter in practice:

1. **β parameters barely move under inference.** The posterior on $\boldsymbol{\beta}$ is essentially the prior, regardless of which β priors are used (asymmetric, label-pooled, or per-edge). The data carries very little Fisher information about $\boldsymbol{\beta}$ under the production rater design. So "reference systems updating lower-layer parameters" — Arvo's intent — happens very weakly in practice for β.

2. **Per-system parameters absorb root signal.** An earlier ablation removed all anchors and gave each system its own $\boldsymbol{\kappa}$. Result: every $C_s$ posterior collapsed to the prior mean. This is the §4.3 design tension from the meeting prep doc: per-system parameters (per-system $\boldsymbol{\kappa}$, per-rater $b_e$ when raters are mostly single-system, per-system $a$, etc.) all absorb per-system rating-distribution shifts that the root state should explain.

So there's a structural pressure pulling in two opposite directions:
- **Add per-system / per-rater flexibility** → captures genuine system-level differences → but absorbs root signal
- **Share parameters across systems** → preserves root identification via anchors → but commits to "consciousness manifests identically in all systems"

The current model is at the "share everything" extreme (apart from the loose per-rater $b_e$). Phase 1C says the bottleneck under this extreme is the transmission gap, fixable by the targeted-override.

---

## The architectural questions I'm trying to resolve

Given Arvo's original intent + Phase 1's empirical findings + the goal of producing a defensible per-system root probability, several architectural choices are on the table. I want your view on each.

### Question A: should we restore a shared $\pi_0$ hyperprior?

Arvo's design has $\pi_0 \sim \mathrm{Beta}(1, 1)$ shared, $C_i \mid \pi_0$ per system. Currently absent.

Pros of adding it:
- More faithful to Arvo's intent
- Reference systems can pool toward a shared base rate, providing genuine cross-system regularisation
- Diffuse $\pi_0$ + multiple graded reference systems would mean the base rate is data-driven rather than fixed at Beta(1, 5)

Cons:
- With only 4 systems (2 anchored, 2 free), a shared $\pi_0$ has very little data
- May make the headline reporting more confusing — "system $s$'s consciousness probability" becomes a function of both posterior on $\pi_0$ *and* per-system likelihood
- The current Beta(1, 5) per-system prior is essentially a strong prior expressing "consciousness is rare a priori" — replacing it with a diffuse shared $\pi_0$ changes that

Does adding shared $\pi_0$ help with **root recovery** for the free systems, or is it mostly a cosmetic re-architecture?

### Question B: should we restore Arvo's state-conditional $\delta_k^{(Z)}$ per-rater observation model?

Currently: $\eta = b_e + a \cdot (m_j / 2)$ — per-rater $b_e$ shift, shared $a$.

Arvo's design: $\eta = \delta_k^{(Z)} + \varepsilon$ — per-rater, state-conditional. Equivalent to:
$$
\eta_{ej} = b_e^{(0)} + (b_e^{(1)} - b_e^{(0)}) \cdot z_j
$$
with $b_e^{(0)}, b_e^{(1)}$ both per-rater. This gives each rater both their own discrimination ($b_e^{(1)} - b_e^{(0)}$) and their own bias.

Pros:
- Restores Arvo's SDT-style design
- Calibration through reference systems would now bear directly on per-rater discrimination, which is Arvo's stated intent
- Captures genuine inter-rater differences in how they use the Likert scale

Cons:
- With single-system-dominant rater design (5/6 raters rate only one system, only Rater_B crosses), per-rater state-conditional params are co-identified with system-level signal. Worse identifiability than the current shared-$a$ model.
- Doubles the per-rater parameter count from 1 to 2; requires strong hierarchical shrinkage to remain identifiable.

Is this a defensible move under the current rater design, or does it require a better survey first?

### Question C: should the tree be split per-system?

Naive version: each system has its own $\boldsymbol{\beta}_s$. No sharing.

Compromise version: hierarchical β — $\boldsymbol{\beta}_s$ around a shared $\boldsymbol{\beta}_0$, with tight hyperprior.

Pros:
- Loosens the "consciousness manifests identically in all systems" commitment
- Could absorb genuine system-specific structural differences (e.g., maybe Coherence works differently in chickens than humans)

Cons (large):
- Naive per-system β loses the cross-system calibration that makes anchored Human/ELIZA useful at all. Each system's β identified only by its own data; with 50-186 ratings per system this is hopelessly under-identified.
- Hierarchical β with tight hyperprior reduces to "shared β" in the limit; with loose hyperprior reduces to "per-system β." The window between is narrow.
- More importantly: the current data has weak β identifiability *under shared β*. Adding per-system variation makes this worse, not better.
- Breaks cross-system comparability of $C_s$: under per-system β, $C_s = 0.5$ in Chicken doesn't mean the same thing as $C_s = 0.5$ in LLMs.

Is this just a bad idea, or is there a version that helps?

### Question D: what does Arvo's "reference systems update lower-layer parameters" actually buy in practice, given that β barely moves?

Empirically, the reference systems' anchor mechanism in the current model does pull $(a, \kappa, b_e)$ toward values consistent with Human's high ratings and ELIZA's low ratings. This is "lower-layer parameter calibration via reference systems" in practice — just not literal Bayesian updating of $\delta_k^{(Z)}$.

But β barely updates. The Phase 1C asymmetric prior sweep showed posterior $\boldsymbol{\beta} \approx$ prior $\boldsymbol{\beta}$.

So Arvo's intent — that reference systems update *all* lower-layer parameters — is empirically realised for $(a, \kappa, \mathbf{b})$ but not for $\boldsymbol{\beta}$. Does this matter? Is there an architectural change that would make β more responsive to reference-system data, or is the chain-rule Fisher-information argument (β identification needs many independent parent-child realisations, which the current design lacks) fundamentally limiting?

---

## The specific question for you

Given:
- The goal is producing a defensible per-system **root probability** $\rho_s = p(R_s=1 \mid y_s)$
- Phase 1C says the binding bottleneck is lower-edge transmission under strong-top features (currently being fixed by targeted-override)
- β parameters barely move under inference regardless of architecture
- Per-system parameters absorb root signal
- The current implementation diverged from Arvo's original intent in three specific ways (no shared $\pi_0$, collapsed $\delta_k^{(Z)}$ to shared $a$ + per-rater $b_e$, hard/soft anchors instead of graded credences with shared base rate)

What's the right architectural move? Specifically:

1. **Does restoring a shared $\pi_0$ hyperprior help with root recovery?** Or is it mostly a cleanup of the reporting story without practical inference impact?

2. **Should we restore $\delta_k^{(Z)}$ state-conditional per-rater observation params?** Or does the rater design make this infeasible until a better survey is collected?

3. **Is per-system or hierarchical β a good idea?** My intuition is no, but I want your reasoning.

4. **Is there a different parameter-sharing architecture I'm missing?** E.g., shared β with per-stance variation (we currently only model GWT, but the published model considers multiple stances); per-stance hyperpriors; partial sharing where some parts of the tree are shared and others are per-system.

5. **Given that β barely moves under any architecture we've tested**, is the "reference systems update lower-layer parameters" framing of Arvo's intent actually achievable in practice? Or is the rater design (mostly single-system raters) the binding constraint that no architectural change can overcome?

6. **Ranking**: what's the highest-leverage architectural change to recommend to Arvo for post-SPAR work, given root probability is the headline?

## Constraints on your response

- Approximately one week of SPAR engineering time left after the targeted-override pilot completes.
- Architectural changes that require survey-design changes (more cross-system raters) are explicitly *future work*, not implementable now. Flag them as such.
- The targeted-override fix is currently being validated (HMC pilot in progress). Don't propose dropping it; reason about what comes *after* assuming it passes.
- If you think one of the questions above is the wrong framing, push back rather than building on it.
- 3-5 page response. Show reasoning where it informs the ranking.
- Don't pad with general Bayesian advice; assume working knowledge of hierarchical models, identification theory, and the binary-root estimand semantics from Phase 0.
