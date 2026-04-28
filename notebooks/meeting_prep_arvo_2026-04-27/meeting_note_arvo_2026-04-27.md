# Meeting note for Arvo — Apr 27, 2026

**Author**: Ryan Kelly (SPAR fellow). **Audience**: Arvo Muñoz Morán (RP).
**Topic**: GWT tree-prior intervention — what I tested, what's flawed, what I want your input on.

5-minute read. Companion artefacts: `chatgpt_pro_bundle.md` (full setup), `diagnostics_round2_results.md` (round-2/3 diagnostics + numbers + Phase B production exact-tree fit + proper exact-tree PPC), notebook 18 (tree-propagation decision artefact).

---

## 1. What was wrong with the tree

Before any data, the paper's `(support, demandingness) → Beta` lookup gives **only ~5% root-to-indicator anchor separation** at the GWT indicator layer. Mean prior-predictive `q_j(C=0.999) − q_j(C=0.001) = 0.048` on a root anchor gap of 0.998 (over 10⁴ Monte Carlo draws from the paper prior alone, before any rating data; B.7 / Stage 3.5 of notebook 18). Compounded by depth: depth-3 paths attenuate to ~5%, depth-2 to ~13%.

This is **structural**: the structural-vs-data-induced check (`log r_j` median = +0.20) confirms the posterior tracks the prior rather than collapsing to it. The paper's tree prior itself says "indicators only weakly distinguish reference systems"; the ordinal-likelihood machinery downstream then carries almost all of the anchoring.

**Consequence for the bottom layer**: the three-state ordinal leaf got most of the focus-cell PPC closure I needed — but the residual was diagnosable as tree-side, not emission-side. So I moved upward.

## 2. What I tested

**`pool_3s` (`POOL_BETAS_BY_LABEL=True`, three-state leaf)**: keeps the `(support, demandingness)` labels but treats them as exchangeability classes. One logit-Normal `β_pres__(σ,δ)` per group, σ=0.5 *median-centred* on the paper Beta mean. Plus one `β_abs__δ` per demandingness group (matching the paper's β_abs lookup). Sampling clean. **Closes E_cross focus cells by ~25-28%** vs `baseline_3s`.

**`pool_3s_abs_by_sd`** (round-2 diagnostic refit): removes the d-only β_abs sharing — each `(σ, δ)` group gets its own β_abs RV. Tests whether the apparent `weak_undermining + neutral` "sign-flip resolution" under `pool_3s` was inheriting from other neutral-demand cells. Sampling clean.

**Production exact-tree fit** (added this morning): replaces the implementation's per-(system, indicator) marginal `pm.Potential`s with a single per-system bottom-up DP `pm.Potential` that exactly marginalises the latent-tree internal `z`'s. Same `pool_3s_abs_by_sd` config; 4 chains × 1000 tune × 2000 draws in 14.5 min; 0 divergences, R-hat 1.0, ESS 5826. Cross-checked against an independent NumPy DP at numerical precision.

**Headline numbers (all four fits)**:

| fit | C_Chicken | C_LLMs | mean δ_j | E_cross×Human Δright | E_cross×ELIZA Δleft |
|---|---:|---:|---:|---:|---:|
| `baseline_3s` | 0.297 | 0.118 | 0.075 | -0.228 | -0.344 |
| `pool_3s` | **0.362** | **0.170** | 0.155 | **-0.163** (28% closure) | **-0.257** (25%) |
| `pool_3s_abs_by_sd` | 0.300 | 0.127 | 0.098 | -0.206 (9%) | -0.324 (6%) |
| `exact-tree (prod)` | **0.252** | **0.111** | 0.110 | -0.283* | -0.263* |

*PPC for `exact-tree (prod)` reported under proper exact-tree leaf-updated `ρ_m` (computed by 2-pass belief propagation; sanity-checked against the standalone exact-tree log-lik at machine precision).

## 3. The biggest reframe of the morning

**I had been calling the original implementation "Rao-Blackwell of the latent tree" — that wording was wrong.** The implementation propagates marginal indicator probabilities `q_j` top-down and treats indicators as conditionally independent given `(C, β)`. That's a **composite marginal likelihood**, not Rao-Blackwell of the paper's latent tree. The two models give materially different posteriors (Δℓ ≈ +10 nats, ESS collapses to <0.2% of draws under reweighting; library-wide, not pool-specific).

**But** — and this is where the morning's thinking landed — there are two coherent DGPs and we (as modellers) get to pick:

> **Latent-state tree** (paper-like): internal feature/subfeature nodes are real binary latent states. Siblings are correlated through shared ancestor states. Implemented in the new exact-tree builder. More conservative on `C` (Chicken 0.36 → 0.25, LLMs 0.17 → 0.11) because it correctly accounts for evidence redundancy through shared latents.
>
> **Leaf-independent marginal tree** (current implementation): tree structure is a parameterisation device for assigning marginal indicator probabilities from `(C, β)`. No shared latents at the feature/subfeature level. Indicators conditionally independent given `(C, β)`. Better focus-cell PPC because it doesn't impose shared-ancestor correlation.

Both are coherent. Neither is automatically "the truth" — calling the latent-state tree "the right model" because it matches the paper text would be deferring to a choice we should be making explicitly. The right way to frame this is as a DGP commitment.

## 4. Possible flaws

1. **Composite-vs-exact gap is library-wide.** The leaf-independent marginal tree drops sibling correlations through internal nodes. In the latent-state-tree view, this means it overcounts evidence at well-supported cells (treating two correlated siblings as two independent updates). In the leaf-independent-tree view, this is just the model's commitment. Either way, the two posteriors differ materially.

2. **`pool_3s` "sign-flip resolution" was mostly a sharing artefact.** Round-2 diagnostics: 70% of the apparent `weak_undermining + neutral` cluster shift was inheritance from the shared `β_abs__neutral` posterior (driven by the 10 other neutral-demand cells). Under `pool_3s_abs_by_sd` the cluster collapses to sign-ambiguous (`Pr(δ̄ > 0) = 0.56`). Under exact-tree, sign-ambiguous and slightly negative (`Pr(δ̄ > 0) = 0.44`). **Right framing**: the cluster is practically neutral, not robustly positive.

3. **Pooling fixes posterior label calibration, not prior-predictive mean attenuation.** Mean prior-predictive `δ_j` is ~0.04 at every σ_pool tested. Pooling lets data move per-label means; it does NOT strengthen the structural transmission. Pooling and "transmission gain" are orthogonal levers.

4. **PPC trade-offs are real, not just a yardstick issue.**
    - Composite-fit / composite-PPC: `pool_3s` gets 25-28% closure on E_cross focus cells.
    - Exact-tree fit / exact-tree PPC: focus cells substantially under-predict (Δright = -0.28 on Human, Δleft = -0.26 on ELIZA — better than the same fit scored under composite PPC, but still substantially worse than `pool_3s` composite).
    - This is **not** a measurement artefact: the exact-tree model is genuinely more conservative about per-cell predictions because it accounts for evidence redundancy. Under-prediction at high-evidence cells is the model's commitment to not over-confidently update on correlated siblings. Whether you accept this is a DGP-choice question.

## 5. Asks for you (the wider-project context items)

I have **two killer questions**, and I'd like your view on both:

1. **Are internal feature/subfeature nodes meant to be real latent binary states, or just bookkeeping for organising marginal indicator evidence from `(C, β)`?** This is the DGP commitment. If real-states: the exact-tree builder is the cleaner formal target, and headline `C` will land more conservatively (Chicken 0.25, LLMs 0.11). If bookkeeping: the leaf-independent marginal tree is fine, and headline `C` stays closer to `pool_3s` levels (0.36, 0.17). I don't have strong domain intuitions here — I lean toward "bookkeeping" on epistemic grounds (we don't have evidence for discrete on/off states for "Selective Attention"), but you and the team know what RP wants the framework to assert.

2. **Better-per-cell-PPC vs more-conservative-on-`C`** — under either DGP, there's a tension. The leaf-independent model fits per-cell ratings better, partly because it treats correlated sibling evidence as independent (which may be over-confident). The latent-state model is more calibrated about evidence redundancy but currently doesn't fit the reference cells as well. Which side of this trade-off do you want the project to land on?

Lower-priority project asks:

3. **Is `β_abs` semantically meant to depend on `(σ, δ)` or only `δ`?** Round-2 results show the d-only sharing in the paper does material work in the headline numbers; `(σ, δ)` keying is cleaner if you're willing to accept that absence behaviour can depend on support.

4. **Matilda's restructured tree timeline?** Both my exact-tree builder and the pooling are structure-agnostic and would re-fit cleanly on whatever she lands on. Conjunctive level-2 dependencies (`a + b → c`) would be a substantively bigger model change.

5. **Edwin's JAX backend timeline?** Now that the exact-tree builder works, proper partial pooling (label hypermean + node residuals) is the next principled tightening. JAX would make the larger-graph compile time tractable.

## 6. What I'm NOT recommending today

- **Promoting any new fit to "official reporting baseline"** — that decision is downstream of (1) above and shouldn't happen at this meeting.
- **Promoting `pool_3s_abs_by_sd` as the "leading library candidate"** — earlier framing I retracted; it's a diagnostic stepping-stone, not a candidate.
- **Switching to exact-tree as a default** — same reason; it's a structurally important alternative DGP, but choosing between DGPs is the point of question 1.
- **All-stance rollout** under exact-tree — about 3-4 hours of compute; not worth doing without the DGP decision first.

---

## Appendix — Phase B production exact-tree fit (full numbers)

For technical reference. Skip unless you want the full breakdown.

### Sampling diagnostics

| metric | value |
|---|---|
| PyMC model build time | 5.7 sec |
| Sampling time | 871 sec (~14.5 min on 4 cores) |
| Divergences | 0 |
| Max R-hat | 1.0000 |
| Min ESS bulk | 5826 |
| Pre-sampling sanity check | PASS (PyTensor exact-tree vs NumPy DP: max \|Δ\| = 5.7e-7) |

### Headline C posteriors (production)

| system | exact-tree (prod) | `pool_3s_abs_by_sd` | `pool_3s` | exact vs `pool_3s` |
|---|---:|---:|---:|---:|
| Chicken | 0.252 [0.037, 0.613] | 0.300 | 0.362 | **−0.110** |
| LLMs | 0.111 [0.005, 0.446] | 0.127 | 0.170 | **−0.059** |

### Sign uncertainty for the weak-undermining cluster

| fit | cluster mean δ̄ [94% CI] | Pr(δ̄ > 0) | Pr(\|δ̄\| < 0.02) |
|---|---:|---:|---:|
| `pool_3s` | +0.060 [-0.025, +0.165] | 0.90 | 0.17 |
| `pool_3s_abs_by_sd` | +0.006 [-0.067, +0.088] | 0.56 | 0.39 |
| **exact-tree (prod)** | **−0.008 [-0.097, +0.080]** | **0.44** | **0.34** |

### Focus-cell PPC: composite-style ρ_m vs exact-tree ρ_m on the same exact-tree fit

| cell | tail | composite ρ_m Δ | exact-tree ρ_m Δ | direction |
|---|---|---:|---:|---|
| E_cross × Human | right | -0.309 | -0.283 | slightly better |
| E_cross × ELIZA | left | -0.364 | -0.263 | meaningfully better (+28%) |
| E_chickenB × Chicken | right | +0.108 | +0.248 | worse |
| E_llmC × LLMs | right | +0.133 | +0.099 | slightly better |

Apples-to-apples (proper exact-tree ρ_m) vs `pool_3s` composite-ρ_m:

| cell | tail | `baseline_3s` | `pool_3s` | exact-tree (proper) |
|---|---|---:|---:|---:|
| E_cross × Human | right | -0.228 | -0.163 | -0.283 |
| E_cross × ELIZA | left | -0.344 | -0.257 | -0.263 |

The exact-tree fit's E_cross × ELIZA (proper PPC) is comparable to `pool_3s`. E_cross × Human is substantially worse. Net: not just a measurement issue.

Outputs:
- `results/gwt_exact_tree/three_state_pooled_abs_by_sd_exact_anchored.{nc,meta.json}`
- `figs_round2/exact_tree_production_headline.csv` / `_ppc.csv` / `_ppc_BP.csv` / `_sign.json`
- Builder: `dcm_model_exact_tree.py:MultiSystemExactTreeBuilder`
- Driver: `run_exact_tree_production.py`
- Post-processing: `postprocess_exact_tree_production.py` (composite-style PPC) + `exact_tree_ppc.py` (proper exact-tree PPC via belief propagation)
