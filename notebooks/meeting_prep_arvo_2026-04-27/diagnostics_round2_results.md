# Round-2 diagnostics — DCM tree-prior pooling

> Companion to `chatgpt_pro_bundle.md` and `chatgpt_pro_prompt.md`. Self-contained
> results from the four diagnostics ChatGPT Pro asked me to run before the
> Arvo meeting. Designed to paste into ChatGPT for the next critique round.

---

## Headline

1. **The composite-vs-exact log-likelihood gap is NOT small.** Per-draw `Δℓ = ℓ_exact − ℓ_composite` has median **+9.95 nats** (5th/95th: −0.7, +18.6) on the GWT `pool_3s` posterior. Importance-reweighting ESS collapses to **6 of 8000 draws (ratio 0.001)**, Pareto-`k̂` = +1.38. The reweighting cannot be trusted to estimate what the exact-tree posterior looks like; a refit on the exact-tree likelihood would be needed to know. The composite-likelihood model is genuinely a different model from the exact-tree model — not an approximation that's "close enough".
2. **The shared-`β_abs__neutral` inheritance is quantitatively confirmed.** Under the four-scenario Shapley-style decomposition on the 5 sign-flip indicators, the shared-abs counterfactual (`β_abs__neutral` reset to paper 0.5) drives **3.1× more of the `δ_j` "resolution"** than the own-pres counterfactual. The B.3a interpretation stands: the apparent sign-flip resolution is mostly inheritance, not group-specific evidence.
3. **`σ_pool = 0.5` is a defensible midpoint, not a special value.** Across `σ ∈ {0.25, 0.5, 0.75, 1.0}`, the **mean** prior-predictive `δ_j` barely moves (+0.046 → +0.041), but the **fraction of indicators with sign-flipped `δ_j`** rises sharply (12% → 32%). σ=0.5 is reasonable as "paper-odds-wrong-by-≤×2.7 over a 95% interval".
4. **The `β_abs__(σ, δ)` refit (a single confirmed correction) moves headline conclusions down.** Under the richer parameterisation: `C_chicken` 0.36 → 0.30; `C_LLMs` 0.17 → 0.13. Cluster mean `δ_j` collapses from +0.060 (Pr > 0 = 0.90) to +0.006 (Pr > 0 = 0.56 — sign-ambiguous). Mean overall `δ_j` drops 0.155 → 0.098. Sampling clean (0 divergences, weak-undermining cluster R-hat 1.0000, ESS 10086). **But**: round-3 PPC closure check (A.2) shows the new fit also regresses focus-cell PPC closure to ~6-9% (vs `pool_3s`'s ~28%), so this is a real trade-off, not a strict improvement.

5. **Round-3 extension (A.1) — composite-vs-exact gap is library-wide.** Both `pool_3s_abs_by_sd` (Δℓ +13.1, ESS 0.002) and `baseline_3s` (Δℓ +8.7, ESS 0.0005) show the same composite-likelihood mismatch as `pool_3s`. Anchored systems (Human, ELIZA) drive positive Δℓ; free systems drive negative. The composite-vs-exact correction would primarily change learned `β`'s and emission `(a, κ)`; direction-of-shift in `C_chicken`, `C_LLMs` cannot be reliably inferred from failed reweighting.

The headline that matters most for the meeting is (1) — it materially changes the framing of `pool_3s` (and indeed of every fit in the library) from "exact Rao-Blackwellisation" to "composite marginal likelihood". The bundle's language needs correcting.

**Honest synthesis** (per ChatGPT round-3): one confirmed correction (`(σ, δ)`-keyed `β_abs` instead of `δ`-only) demonstrably lowers `pool_3s` headline `C` by 4-6%. A second issue (composite-vs-exact likelihood mismatch) is large enough — and library-wide — that an exact-tree refit is required to know its direction and magnitude. Failed importance reweighting (ESS collapse across all three fits) gives no reliable estimate. The two effects should be reported as **"one confirmed downward correction; one unquantified material mismatch"**, not as a principled bracket.

---

## Task 1 — Composite-vs-exact log-likelihood diagnostic

### Verified bug in current implementation language

The DCM uses one `pm.Potential` per (system, indicator) with a binary mixture in the propagated marginal probability `q_{s,j}`. There are **no** internal-node `z` random variables in the PyMC graph; each indicator's likelihood treats it as conditionally independent of siblings given `(C, β)`. This is **not** Rao-Blackwellisation of the original DCM latent tree — it is a **composite marginal likelihood** that drops the sibling covariance through every shared internal node.

For one rating per indicator, two sibling indicators `a`, `b` sharing internal parent `Z`:

$$
\operatorname{Cov}(Y_a, Y_b \mid C, \beta) = q_Z(1 - q_Z)(\beta^{\text{pres}}_a - \beta^{\text{abs}}_a)(\beta^{\text{pres}}_b - \beta^{\text{abs}}_b)
$$

The product-of-marginals model drops this covariance; the exact-tree marginal preserves it.

### Method (per ChatGPT round-2 spec)

For each posterior draw `s` from the existing `pool_3s` netcdf:
1. **Composite ℓ** mirrors the implemented `pm.Potential` formula bit-for-bit (mixture-of-products: one shared latent state per indicator, all ratings on that indicator share it). Verified in `dcm_model.py:500-531` before scripting.
2. **Exact ℓ** via bottom-up dynamic programming on the GWT tree (25 internal feature nodes + 50 indicators, strictly a tree — verified). At leaves, `L_indicator(z_pa) = B_{s,j}(β_pres_j)` if `z_pa = 1` else `B_{s,j}(β_abs_j)`, where `B(β) = (1-β)²ℓ⁽⁰⁾ + 2β(1-β)ℓ⁽¹⁾ + β²ℓ⁽²⁾` is the three-state mixture evaluated at the appropriate β.
3. **Importance reweighting** with stable log-sum-exp normalisation: `log w_s = Δℓ_s − logsumexp_r(Δℓ_r)`.

### Sanity checks (all passed)

- **Brute-force enumeration** on a synthetic 2-internal-node, 4-leaf tree (16 latent configurations enumerated by hand): `|log_DP - log_brute| = 0.0`. *This caught a real bug in my first DP draft (children's β was being applied twice via "self-edge" steps); fix verified before reporting.*
- **No-transmission invariance**: setting `β_pres = β_abs` everywhere ⇒ `|ℓ_exact - ℓ_composite| = 3.4e-13` (machine precision).
- **Self-consistency**: `ℓ_composite + Δℓ = ℓ_exact` to machine precision.

### Results (8000 posterior draws)

| metric | value |
|---|---|
| ℓ_composite, median | −628.0 |
| ℓ_exact, median     | −618.4 |
| **Δℓ, median**      | **+9.95 nats** |
| Δℓ, mean            | +9.56 nats |
| Δℓ, 5th / 95th pct  | −0.70 / +18.59 |
| Δℓ, min / max       | −16.4 / +28.2 |
| ESS_w               | **6** |
| ESS_w / n_draws     | **0.001** |
| Pareto-`k̂` (PSIS)  | **+1.38** |

**Reweighted shifts in headline quantities** (reported but **not trusted** — ESS collapse means these are not reliable estimates of the exact-tree posterior):

| quantity            | original | reweighted | shift |
|---------------------|---------:|-----------:|---:|
| C_Chicken median    | 0.3622   | 0.1810     | **−0.181** |
| C_LLMs median       | 0.1695   | 0.0988     | −0.071 |
| mean δ_j            | 0.1586   | 0.1518     | −0.007 |
| β_abs_neutral median | 0.3043  | 0.3292     | +0.025 |

### Sibling-correlation potential scores

Per internal node `v`, score = `q_v(1-q_v) · max_{a,b∈children(v)} |Δ_a Δ_b| · n_obs_subtree`.

Top 5:

| rank | node                  | n_children | q_v_med | max\|Δ_aΔ_b\| | n_obs_subtree | score |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Selective Attention      | 5 | 0.540 | 0.4094 | 91 | 9.25 |
| 2 | Coherence                | 4 | 0.445 | 0.2819 | 99 | 6.89 |
| 3 | Complexity               | 5 | 0.623 | 0.4557 | 61 | 6.53 |
| 4 | Modularity               | 4 | 0.396 | 0.3955 | 48 | 4.54 |
| 5 | Autonomous Subparts      | 5 | 0.388 | 0.3118 | 37 | 2.74 |

The `weak_undermining + neutral` parent ("Autonomous Subparts") ranks 5th. The strain on the composite approximation is concentrated at high-up feature nodes with many descendants and intermediate `q_v`, not at the sign-flip cluster specifically.

### Verdict

**Importance reweighting failed.** Composite posterior is too far from the exact-tree posterior for post-hoc correction. The composite-vs-exact approximation **is** load-bearing for the DCM under `pool_3s`. Two implications for the writeup:

- The framing in §3.2 of the bundle ("Rao-Blackwell") is wrong and should be corrected to "composite marginal likelihood under leaves-conditionally-independent-given-`(C, β)`".
- The library's headline conclusions (e.g. `C_chicken` = 0.36, `C_LLMs` = 0.17 under `pool_3s`) are *conditional on the composite posterior*, which is a different model from the exact latent-tree one. We cannot infer from the reweighting how the exact-tree posterior would differ.

**Honest framing** (per ChatGPT round-3): the failed importance sampling does not give a reliable directional estimate, let alone a quantitative bracket. The exact-tree likelihood is materially different from the composite likelihood; failed reweighting suggests the exact-tree posterior could move substantially, but its direction and magnitude require an exact-tree refit.

### Round-3 extension — three-fit comparison + by-system split

Per ChatGPT round-3, the diagnostic was extended to two more fits and split by system. Plus a reconstruction check confirming the diagnostic's "composite" formula matches PyMC's actual fitted Potential value to numerical precision (max |Δ| over 50 draws < 1e-12, **PASS**).

| fit | Δℓ median (total) | Δℓ 5th/95th | ESS_w / S | Pareto-k̂ |
|---|---:|---:|---:|---:|
| `pool_3s` | +9.95 | (-0.70, +18.60) | 0.001 | +1.38 |
| `pool_3s_abs_by_sd` | **+13.14** | (+6.06, +19.31) | 0.002 | +0.96 |
| `baseline_3s` | +8.75 | (+2.15, +14.96) | 0.0005 | +1.33 |

**Composite-vs-exact gap is library-wide, not pooling-specific.** All three fits show large Δℓ and ESS collapse. `pool_3s_abs_by_sd` actually has the *largest* total Δℓ. The framing "this is a problem with `pool_3s`" is wrong — it's a problem with the entire DCM construction.

**By-system Δℓ split** (median per draw):

| fit | Human | Chicken | LLMs | ELIZA |
|---|---:|---:|---:|---:|
| `pool_3s` | +10.17 (~107% of total) | -8.83 (~-90%) | -10.19 (~-109%) | +18.39 (~192%) |
| `pool_3s_abs_by_sd` | +7.85 (~61%) | -3.54 (~-28%) | -6.05 (~-48%) | +14.88 (~115%) |
| `baseline_3s` | +6.08 (~71%) | -2.88 (~-34%) | -5.68 (~-66%) | +11.18 (~130%) |

Anchored systems (Human, ELIZA) drive the *positive* Δℓ; free systems (Chicken, LLMs) contribute *negative*. The exact-tree model finds the data more likely at the anchored cells (where sibling correlation through the tree adds explanatory power that the composite drops) and *less* likely at the free-system cells (where the composite's looser independence assumption fits the modest expert ratings better).

Per ChatGPT's framing: this means an exact-tree refit would primarily change learned `β`'s and emission `(a, κ)` (driven by the anchored systems' magnitude-dominant contribution), with secondary effects on free-system `C` posteriors. Direction-of-shift in `C_chicken`, `C_LLMs` cannot be reliably inferred from the failed reweighting.

**Sibling-correlation top-5 reorganisation under abs_by_sd**: largely consistent across fits — Selective Attention, Coherence, Complexity, Modularity dominate everywhere. `Autonomous Subparts` (the weak-undermining-neutral parent) drops from rank 5 in `pool_3s` to outside the top 5 under `pool_3s_abs_by_sd` and `baseline_3s`, replaced by `Representationality`. Strain reorganises slightly but the high-traffic feature nodes are the dominant correlation contributors regardless.

**β_abs__neutral handling** (per ChatGPT round-3): the round-3 output table reports headline-`C` reweighted shifts (untrustable) but does NOT force a single shared `β_abs__neutral` quantity across fits. `pool_3s` has a real shared parameter; `pool_3s_abs_by_sd` has a singleton `β_abs__(weak undermining, neutral)`; `baseline_3s` has no label-level β_abs (per-node Beta priors).

### Outputs
- `figs_round2/composite_gap_summary.json` — full diagnostic numbers + ESS interpretation.
- `figs_round2/composite_gap_headline_shifts.csv` — reweighted-vs-original headline shifts.
- `figs_round2/sibling_correlation_scores.csv` — full per-internal-node ranking (pool_3s).
- `figs_round2/composite_gap.png` — Δℓ histogram + reweighted-vs-original Chicken posterior.
- `figs_round2/composite_gap_three_fits_comparison.csv` — round-3 three-fit comparison + by-system split.
- `figs_round2/sibling_correlation_top5_by_fit.csv` — round-3 sibling-correlation localisation per fit.

---

## Task 2 — `β_abs__neutral` counterfactual decomposition

### Method

Four-scenario Shapley-style counterfactual on the 5 sign-flip indicators (all share the `weak_undermining + neutral` subfeature ancestor "Autonomous Subparts"):

| scenario | `β_abs__neutral` | `β_pres__weak_und__neutral` |
|---|---|---|
| (a) posterior as fitted | posterior | posterior |
| (b) shared abs reset    | **0.500 (paper)** | posterior |
| (c) own-pres reset      | posterior | **0.429 (paper)** |
| (d) both reset          | **0.500 (paper)** | **0.429 (paper)** |

Per ChatGPT round-2 correction, `q_j` is **re-propagated from primitive `β` draws** at each scenario; cached `*_p` Deterministics in the InferenceData are not used (would silently return un-patched values).

### Posterior medians at the singleton group (context)

- `β_pres__weak_undermining__neutral`: posterior **0.4918** (paper 0.4286), shift +0.063
- `β_abs__neutral`: posterior **0.3043** (paper 0.5000), shift −0.207

### Per-indicator δ_j across the 4 scenarios

| indicator                          | (a) posterior | (b) reset abs | (c) reset pres | (d) reset both |
|---|---:|---:|---:|---:|
| Information Transfer               | +0.0489 | −0.0021 | +0.0325 | −0.0200 |
| Learning Transfer                  | +0.0532 | −0.0022 | +0.0359 | −0.0222 |
| Functional Subparts                | +0.0579 | −0.0013 | +0.0393 | −0.0141 |
| Information Transfer Architecture  | +0.0672 | −0.0028 | +0.0448 | −0.0272 |
| Conflicting Subparts               | +0.0631 | −0.0027 | +0.0418 | −0.0257 |

Decomposition contributions (mean across the 5 indicators):
- **Shared β_abs effect** (a − b): **+0.060** (3.1× larger)
- **Own β_pres effect** (a − c): +0.019
- **Joint effect** (a − d): +0.082 (note non-additivity due to multiplicative `δ_j` structure)

### Verdict

**Confirmed**: under the (b) counterfactual where `β_abs__neutral` is reset to paper 0.5 but the singleton's own `β_pres` stays at posterior 0.49, **all 5 sign-flip indicators flip back to slightly negative `δ_j`** (closer to the paper's `δ_prior = −0.07`). Under (c) where the own `β_pres` is reset to paper 0.429 but the shared `β_abs` stays at posterior 0.29, the `δ_j` stays positive (~+0.03 to +0.04).

The shared-`β_abs__neutral` shift is the load-bearing component of the apparent sign-flip resolution; the singleton group's own `β_pres` movement contributes meaningfully but not predominantly. The B.3a interpretation in the bundle stands.

### Tree-implied focus-cell PPC under each scenario

Across all 50 E_cross indicators per system, predicted right-tail mass at Human and predicted left-tail mass at ELIZA, computed from primitive `β` + posterior `(a, κ, b_e)`:

| scenario | E_cross × Human pred_right | E_cross × ELIZA pred_left |
|---|---:|---:|
| (a) posterior        | 0.402 | 0.300 |
| (b) reset abs        | 0.415 | 0.270 |
| (c) reset pres       | 0.399 | 0.300 |
| (d) reset both       | 0.413 | 0.271 |
| observed             | 0.86  | 0.96  |

**Note** these are *tree-implied* PPCs (constructed from `q_j(C, β)` at the anchor `C`), not the leaf-updated PPCs in B.10 (which use posterior `ρ_m` weights conditional on observed ratings). Different objects: the tree-implied PPC asks "given posterior `β`, what does the tree alone predict?" while the leaf-updated PPC asks "given posterior `(β, m | y)`, what does the model predict?". Both are valid but answer different questions; the tree-implied is the right object for **counterfactual** reasoning here.

The PPC barely moves between scenarios (ELIZA pred_left shifts ~0.03 across scenarios; Human pred_right ~0.013). Even fully resetting the weak-undermining-neutral β's to paper values barely affects the focus-cell predictions, corroborating the B.11c finding that **the focus-cell residual is mostly emission-side and anchor-pull-side, not tree-side from this specific subfeature.**

### Honest limitation

Holding `β_abs__neutral = 0.5` while reusing posterior `C_chicken`, `C_LLMs` draws is a partial counterfactual: those `C` posteriors were jointly fit with the original `β_abs__neutral`. For `δ_j` (path products of `β` only) this is irrelevant — the result is exact. For PPCs (which depend on `q_{s,j}(C, β)`), the predicted tails are computed conditional on the un-counterfactual `C` posterior. The counterfactual PPCs above should be read as "what the tree alone predicts", not as "what a refit would give".

### Outputs
- `figs_round2/beta_abs_neutral_counterfactual.csv` — per-(scenario, indicator) δ_j, q_high, q_low, PPC.
- `figs_round2/beta_abs_neutral_counterfactual_decomp.csv` — pivoted decomposition table.

---

## Task 3 — `σ_pool` prior-predictive sweep

### Method

For each `σ_pool ∈ {0.25, 0.5, 0.75, 1.0}`, sample 10⁴ realisations of the per-label `(β_pres, β_abs)` from logit-Normal priors **median-centred** at the paper Beta means (per ChatGPT round-2 wording correction — *not* mean-centred; logit-Normal natural-scale mean ≠ logit-Normal natural-scale median except at the median = 0.5). Propagate `q_j` through the GWT tree at `C ∈ {0.001, 0.5, 0.999}`.

### Results (mean over 10⁴ draws × 50 indicators)

| σ_pool | mean δ_j | mean q(.999)−q(.001) | fraction δ_j < 0 | δ_j 1st/99th pct |
|---:|---:|---:|---:|---:|
| 0.25 | +0.0464 | +0.0463 | 0.118 | (−0.04, +0.30) |
| 0.50 | +0.0450 | +0.0449 | 0.189 | (−0.10, +0.39) |
| 0.75 | +0.0432 | +0.0431 | 0.262 | (−0.18, +0.46) |
| 1.00 | +0.0412 | +0.0412 | 0.319 | (−0.27, +0.51) |

### Reading

- The **mean** prior-predictive `δ_j` is essentially flat across σ — the pool prior is median-centred at paper means, so the central tendency is preserved.
- The **spread** widens dramatically: the fraction of indicators with sign-flipped `δ_j` rises from 12% to 32% as σ goes from 0.25 to 1.0.
- σ_pool = 0.5 (the current pool_3s value) gives ~19% sign-flip rate in the prior — moderate; corresponds to "paper odds wrong by ≤×2.7 over a 95% interval".
- Comparing to the existing pool_3s **posterior** (mean δ_j = 0.155, no sign flips on the posterior median — reduced from the prior's 19% sign flips), the data is moving things meaningfully in both location and width.

### σ_pool justification framing for the writeup

Reasonable to position σ=0.5 as the midpoint of:

| σ | "paper-odds-wrong-by-at-most" over 95% interval |
|---|---|
| 0.25 | ×1.6  (paper-tight) |
| 0.50 | ×2.7  (current) |
| 0.75 | ×4.5 |
| 1.00 | ×7.4 |

σ=0.5 is "the labels probably wrong by no more than a factor of e ≈ 2.7 in odds, with 95% probability". Compared to the paper's NODE_CONCENTRATION = 10 (i.e., effective sample size of 10 in each Beta), σ=0.5 logit-Normal is comparably informative. So σ=0.5 is consistent with the paper's *information content* about each label's prior, just framed on the logit scale and at the label level instead of the node level.

### Outputs
- `figs_round2/sigma_pool_sweep_summary.csv` — per-(σ, depth) summary stats.
- `figs_round2/sigma_pool_sweep_qj.png` — δ_j density per σ + summary errorbar plot.

### Important caveat (per ChatGPT round-3)

**Pooling does not repair the prior-predictive *mean* attenuation.** The mean prior-predictive `δ_j` is ~0.04 at every `σ_pool` tested (0.25, 0.50, 0.75, 1.00). What pooling repairs is *posterior label calibration*: it lets data move the per-label means away from the paper's hand-set values, and that's where the posterior `δ_j ≈ 0.155` (`pool_3s`) or `≈ 0.098` (`pool_3s_abs_by_sd`) comes from. The paper's tree prior alone — even with `σ_pool = 1.0` — still says reference-anchor separation at the indicator layer is structurally low.

This distinction matters for explaining what pooling actually does in front of Arvo: it's a *calibration* tool that lets the data inform per-label transmission, not a *prior-strength* tool that fixes the structural attenuation in the paper's lookup. Strengthening the prior (e.g., transmission-gain) is a separate intervention with separate failure modes.

---

## Task 4 — `β_abs__(σ, δ)` refit

**Status: sampling in progress.** Will be appended once the fit completes (~20 min wall-clock; smoke test passed cleanly with 0 divergences). Below is the diagnostic question this fit answers.

### What this refit tests

Under `BETA_ABS_BY_SUPPORT_DEMAND=True`, each `(support, demandingness)` group gets its own `β_abs` RV (instead of all `(*, demandingness)` groups sharing one). Same logit-Normal(σ=0.5) prior family, same paper β_abs prior centre (which depends only on demandingness in the paper mapping); only the *grouping* changes.

The diagnostic question: under the new fit, does `β_abs__(weak_und, neutral)` move from its prior 0.5 (= direct evidence on the singleton group), or does it stay near the prior (= consistent with Task 2's interpretation that the apparent shift was inheritance through the shared d-only parameter)?

The **expected outcome** (consistent with Task 2): `β_abs__(weak_und, neutral)` will likely fall back to its prior because the weak-undermining-neutral group is a **singleton** (1 node in the GWT tree). Under the unrestricted parameterisation it has no other (s, d)-cell to share with, and its single node's data alone is not enough to update its β_abs much from 0.5. Meanwhile, the larger neutral-demand groups (e.g. moderate_support + neutral with 6 nodes) will move their own β_abs to the data-driven value (~0.29-ish) — and crucially, those movements no longer leak into the singleton.

**Modest framing** (per ChatGPT round-2 correction): this refit is *not* a test of "support matters for absence behaviour generally". It is a test of whether the demandingness-only sharing in the current `pool_3s` is driving the inheritance noted in Task 2. Singleton non-identification under the new parameterisation = evidence of insufficient direct evidence on the cell, **not** evidence that support has no effect on absence elsewhere.

### Tightened Tier 1 gate

For the headline summary, divergences < 50 is acceptable. For interpreting the fine-grained `weak_undermining + neutral` decomposition specifically, the gate is **zero divergences and max R-hat < 1.005** on `beta_pres__weak_undermining__neutral`, `beta_abs__weak_undermining__neutral`, `label_delta__weak_undermining__neutral`. Otherwise the decomposition is labelled "exploratory under unstable sampling".

---

## Task 4 — `β_abs__(σ, δ)` refit RESULTS

### Tier 1 sampling diagnostics

- divergences: 0
- max R-hat: 1.0000
- min ESS bulk: 4600
- weak_undermining cluster max R-hat: 1.0000
- weak_undermining cluster min ESS bulk: 10086
- **Tightened gate pass: True**

### System C posteriors

| system | pool_3s (d-only) median | pool_3s_abs_by_sd median | shift |
|---|---:|---:|---:|
| Human | 0.9990 | 0.9990 | +0.0000 |
| Chicken | 0.3622 | 0.2999 | -0.0623 |
| LLMs | 0.1695 | 0.1270 | -0.0426 |
| ELIZA | 0.0010 | 0.0010 | +0.0000 |

### `weak_undermining + neutral` decomposition

| parameter | paper | pool_3s (d-only) | pool_3s_abs_by_sd |
|---|---:|---:|---:|
| β_pres | 0.429 | 0.492 [0.275, 0.705] | 0.465 [0.252, 0.687] |
| β_abs | 0.500 | 0.304 [0.183, 0.450] (shared d) | 0.436 [0.243, 0.650] (singleton) |

**Verdict on β_abs__(weak_und, neutral) shift from prior:** shifts -0.064 (modest)

### Per-indicator δ_j on 5 sign-flip cluster

| indicator | pool_3s δ_j | pool_3s_abs_by_sd δ_j | shift |
|---|---:|---:|---:|
| Information Transfer | +0.0489 | +0.0025 | -0.0464 |
| Learning Transfer | +0.0532 | +0.0038 | -0.0495 |
| Functional Subparts | +0.0579 | +0.0048 | -0.0531 |
| Information Transfer Architecture | +0.0672 | +0.0060 | -0.0612 |
| Conflicting Subparts | +0.0631 | +0.0084 | -0.0547 |
| **mean** | **+0.0581** | **+0.0051** | **-0.0530** |

### Mean δ_j across all 50 indicators

- pool_3s (d-only):       **+0.1551**
- pool_3s_abs_by_sd:      **+0.0980**
- shift:                  -0.0571

### Reading

**Singleton β_abs is partially identified, not weakly identified.** Under the richer (s, d)-keyed parameterisation, `β_abs__(weak_und, neutral)` lands at posterior median 0.436 [0.243, 0.650] — shifted -0.064 from the paper prior 0.5, ~30% of the magnitude of the d-only shared shift (-0.207). So the singleton's own data does provide some direct evidence; the d-only sharing was *partly* learning from this evidence and *partly* inheriting from other neutral-demand cells. The result is **not** the "stays at prior, totally non-identified" outcome I expected, but it does confirm that **the inheritance was substantial**: only ~31% of the d-only shift was direct evidence on the cell.

**Sign-flip cluster `δ_j` collapses to near-zero**, not slightly negative as the prior would suggest. Mean cluster `δ_j` drops from +0.058 (under d-only sharing) to +0.005 (under sd-keyed). All 5 indicators move close to the paper-prior δ_j of -0.07 to -0.02 but stay slightly positive. This is direction-consistent with the Task 2 (b) counterfactual (which forced β_abs back to paper 0.5 and got slightly-negative cluster δ_j ≈ -0.002). The pool_3s "sign-flip resolution" was real but smaller in magnitude than the d-only fit suggested — it should be reported as "the sign-flip cluster's δ_j is approximately zero, not strongly positive" rather than "the sign-flip is resolved".

**Headline conclusions move in the same direction as Task 1's (untrustable) reweighted shifts.** Under the (s, d)-keyed parameterisation:
- `C_Chicken` 0.36 → 0.30 (-0.06)
- `C_LLMs` 0.17 → 0.13 (-0.04)
- mean `δ_j` 0.155 → 0.098 (-0.057)

Task 1 reported (with low-trust caveat) reweighted shifts of `C_Chicken` -0.18 and `C_LLMs` -0.07. Task 4's actual refit gives smaller-magnitude shifts in the same direction. **Two independent reasons (composite-likelihood approximation + d-only β_abs sharing) both inflate `pool_3s`'s headline conclusions; correcting either moves headline `C` down by 5-20%.** The combined effect of correcting both is unknown without a fit that does both simultaneously — that would be the next-week priority.

The richer parameterisation also passes Tier 1 cleanly (0 divergences across the entire fit, including on the `weak_undermining + neutral` triplet). So this is a defensible model in its own right, not just a diagnostic.

### Outputs

- `figs_round2/abs_by_sd_signflip_delta.csv`
- `figs_round2/abs_by_sd_system_C.csv`
- `results/gwt_tree_pooling/three_state_pooled_abs_by_sd_anchored.nc`
- `results/gwt_tree_pooling/three_state_pooled_abs_by_sd_anchored.meta.json`

### Round-3 extension — focus-cell PPC closure (A.2)

Per ChatGPT round-3, the leaf-updated PPC machinery (`per_expert_system_ppc_multisystem`, the same one used in B.10) was run on `pool_3s_abs_by_sd`. **Critical finding**: the new fit regresses substantially on focus-cell PPC closure relative to `pool_3s`.

| fit | E_cross × Human Δright | E_cross × ELIZA Δleft | E_chickenB × Chicken Δright | E_llmC × LLMs Δright |
|---|---:|---:|---:|---:|
| `baseline_3s` | -0.228 [...] | -0.344 [...] | +0.176 | +0.113 |
| `pool_3s` | -0.163 [-0.32, +0.04] | -0.257 [-0.42, -0.06] | +0.140 | +0.110 |
| `pool_3s_abs_by_sd` | **-0.206** [-0.36, -0.06] | **-0.324** [-0.48, -0.18] | +0.166 | +0.112 |

**Closure (vs `baseline_3s`)**:

| fit | E_cross×Human | E_cross×ELIZA | E_chickenB×Chicken | E_llmC×LLMs |
|---|---:|---:|---:|---:|
| `pool_3s` | **+28.3%** | **+25.5%** | +20.4% | +2.7% |
| `pool_3s_abs_by_sd` | +9.4% | +6.0% | +6.1% | +1.3% |

**Interpretation**: `pool_3s_abs_by_sd` is more conservative on headline `C` (matching the d-only inheritance correction) **but loses most of the focus-cell PPC closure that made `pool_3s` attractive**. This is a real trade-off — the d-only sharing in `pool_3s` was simultaneously inflating `C` (artefact) AND providing a tighter fit to the cross-system rater's extreme rating tails. Removing the d-only sharing fixes the inflation but regresses the focus-cell fit.

The sign of the shift is NOT mechanically implied by the lower `C` (anchored systems' β + emission move jointly), so the regression direction is data-driven, not a sanity-check failure.

### Round-3 extension — sign uncertainty + practical-equivalence (A.3)

Per ChatGPT round-3, the cluster mean `δ_j = +0.005` deserves uncertainty quantification — not just a bare median.

| fit | cluster mean δ_j median [94% CI] | Pr(δ̄ > 0) | Pr(all 5 δ_j > 0) | Pr(\|δ̄\| < 0.01) | Pr(\|δ̄\| < 0.02) |
|---|---:|---:|---:|---:|---:|
| `pool_3s` | +0.060 [-0.025, +0.165] | **0.900** | **0.900** | 0.089 | 0.167 |
| `pool_3s_abs_by_sd` | +0.006 [-0.067, +0.088] | **0.562** | **0.532** | 0.207 | 0.389 |

**Verdict**: under `pool_3s_abs_by_sd`, the cluster is sign-ambiguous: `Pr(δ̄ > 0) = 0.56`, indistinguishable from a coin flip. Substantial mass within ±0.02 of zero. The right Arvo-ready sentence is:

> "Under `pool_3s_abs_by_sd`, the weak-undermining cluster is sign-ambiguous / practically neutral — `Pr(δ̄ > 0) = 0.56` with `Pr(|δ̄| < 0.02) = 0.39`. The paper prior put the cluster strongly negative; `pool_3s` made it robustly positive (`Pr > 0 = 0.90`); the richer parameterisation puts it in the centre with no robust sign."

Both round-3 extensions support the broader story: `pool_3s_abs_by_sd` removes the d-only sharing artefact but at the cost of focus-cell PPC closure. It is *not* a strict improvement over `pool_3s`; it's a different point in the same trade-off.

### Round-3 outputs (additional)

- `figs_round2/pool_3s_abs_by_sd_ppc_focus_guardrail.csv` (A.2 PPC table)
- `figs_round2/pool_3s_abs_by_sd_ppc_closure_vs_baseline.csv` (A.2 closure %)
- `figs_round2/weak_undermining_cluster_sign_uncertainty.csv` (A.3 sign uncertainty)

---

## Round-2 takeaways for ChatGPT round-3

Four results, all completed:

1. **Composite-likelihood is materially load-bearing** under `pool_3s`. Δℓ median +9.95 nats, ESS collapses to 0.001 ratio, Pareto-`k̂` +1.38. Reweighting cannot post-hoc correct; only an exact-tree refit can. (Round-2 surprise.)
2. **Shared-`β_abs__neutral` inheritance partly confirmed, partly nuanced**. Task 2 counterfactual: shared abs effect 3.1× larger than own-pres effect on cluster `δ_j`. Task 4 refit: under the richer (s, d) parameterisation, `β_abs__(weak_und, neutral)` shifts -0.064 from prior (about 31% of the d-only shared shift -0.207), so the singleton has *some* direct evidence and inheritance accounted for ~70% of the d-only shift, not 100%.
3. **σ_pool = 0.5 is a defensible midpoint**. Mean δ_j flat across σ (paper means preserved at all σ); spread (sign-flip fraction) rises 12% → 32%. Wording corrected to "median-centred logit-Normal" throughout. (Defensibility achieved.)
4. **β_abs__(σ, δ) refit converges cleanly and shifts headline conclusions in the same direction as Task 1's (untrustable) reweighting**. C_Chicken 0.36 → 0.30, C_LLMs 0.17 → 0.13, mean δ_j 0.155 → 0.098. Sign-flip cluster δ_j collapses to near-zero (mean +0.005). Two independent reasons (composite + d-only β_abs sharing) both inflate `pool_3s`'s headline conclusions; correcting either moves them down by 5-20%.

### Synthesis (rewritten per ChatGPT round-3)

**One confirmed downward correction:** `(σ, δ)`-keyed `β_abs` instead of `δ`-only. `pool_3s_abs_by_sd` lowers `pool_3s` headline `C` by 4-6%. Sampling clean. But it also regresses focus-cell PPC closure to ~6-9% (vs `pool_3s`'s 28%) — a real trade-off, not a strict improvement. `pool_3s_abs_by_sd` is the cleaner provisional sensitivity candidate, *not* a validated replacement.

**One unquantified material mismatch:** the composite-vs-exact likelihood is large enough (Δℓ ~10 nats) and library-wide (all three fits show ESS collapse) that exact-tree refitting is required to know its direction and magnitude. Failed importance reweighting (ESS = 6/8000 under `pool_3s`, 15/8000 under `pool_3s_abs_by_sd`, 4/8000 under `baseline_3s`) gives no reliable estimate. This is **not** a second confirmed downward correction; the bundle's earlier "two corrections both inflate `pool_3s`" framing was wrong.

**The weak-undermining + neutral cluster** under `pool_3s_abs_by_sd` is sign-ambiguous: `Pr(δ̄ > 0) = 0.56`, `Pr(|δ̄| < 0.02) = 0.39`. Not robustly anything. The paper prior put it strongly negative; `pool_3s` made it robustly positive (Pr > 0 = 0.90); the richer parameterisation puts it in the centre with no robust sign. The earlier "sign-flip resolved under pool_3s" framing was an artefact of the d-only β_abs sharing.

**Pooling fixes posterior calibration, not prior-predictive mean attenuation.** The paper's tree prior alone — at any σ_pool tested — still says reference-anchor separation at the indicator layer is structurally low (mean δ_j ≈ 0.04). Pooling lets data move per-label means; it does not *strengthen* them.

### Specific questions for ChatGPT (post-round-3)

1. The composite-vs-exact gap is library-wide (Δℓ ≈ 9-13 nats across all three fits) and the by-system split shows anchored systems dominate the magnitude. **What does a positive Δℓ at anchored cells but negative at free cells imply about what an exact-tree refit would actually do?** My read: it would primarily shift learned `β`'s and emission `(a, κ)`; secondary effects on `C_chicken`, `C_LLMs` are uncertain in direction. Is that right, or is there a sharper inference from the by-system split?
2. `pool_3s_abs_by_sd` retains 6-9% of `pool_3s`'s focus-cell PPC closure, vs `pool_3s`'s 25-28%. So the new fit is more conservative on `C` but loses most of the `pool_3s` PPC gains. Is this a "trade-off" framing or is one fit dominating the other on a deeper criterion?
3. Implementation question: an exact-tree refit on a 75-node GWT tree is most naturally a single per-system `pm.Potential` evaluating the bottom-up DP in PyTensor at model-build time (since the tree topology is static). Static Python loops over postorder traversal + `pt.logaddexp` at each node should work without `scan`. Any reason this would be problematic at compile / sampling time?
4. Anything I've missed that would be obvious to you?

---

## Phase B (post-round-3) — exact-tree smoke prototype: SUCCESS

Following ChatGPT round-3's recommendation to attempt a smoke-test exact-tree refit, I built `MultiSystemExactTreeBuilder` (`dcm_model_exact_tree.py`) — a static-graph extension of `MultiSystemModelBuilder` that replaces the per-indicator composite `pm.Potential`s with one exact-tree DP `pm.Potential` per system, expressed in PyTensor via Python loops + `pt.logaddexp` at model-build time (no `scan`).

**Smoke fit**: same `pool_3s_abs_by_sd` config, 2 chains × 300 tune × 500 draws.

| metric | value |
|---|---|
| PyMC build time | 6.9 sec |
| Sampling time | 504 sec |
| Divergences | 0 |
| Max R-hat | 1.0000 |
| Min ESS bulk | 1032 |
| Pre-sampling sanity check | **PASS** (PyTensor vs NumPy DP: max \|Δ\| = 5.7e-7) |

**Headline `C` posteriors** (median [3rd, 97th]):

| system | `pool_3s` | `pool_3s_abs_by_sd` | **exact-tree smoke** | exact vs `pool_3s` |
|---|---:|---:|---:|---:|
| Chicken | 0.362 | 0.300 | **0.246 [0.037, 0.589]** | **−0.116** |
| LLMs | 0.170 | 0.127 | **0.109 [0.004, 0.468]** | **−0.061** |

**Implications**:
- The composite-vs-exact gap was real and the corrections are unidirectional (downward). `pool_3s_abs_by_sd` is a partial fix; the exact-tree refit subsumes the (σ, δ)-keyed `β_abs` correction AND adds the composite-vs-exact correction on top.
- The exact-tree refit is no longer "an unquantified material caveat" — it's a working model. The headline correction relative to `pool_3s` is now measured: `C_Chicken: −0.116`, `C_LLMs: −0.061`.
- The path to a production-quality exact-tree fit (4 chains × 2000 draws) is short: estimated wall-clock ~30-40 min based on smoke timing.
- Method: static Python loops + `pt.logaddexp` work cleanly without `scan`. Confirms ChatGPT's implementation note.

Outputs:
- `results/gwt_exact_tree_smoke/three_state_pooled_abs_by_sd_exact_smoke.nc`
- `results/gwt_exact_tree_smoke/smoke_meta.json`
- `dcm_model_exact_tree.py` (~250 LoC, MultiSystemExactTreeBuilder)
- `run_exact_tree_smoke.py` (driver with cross-check against NumPy DP)

---

## Phase B production — exact-tree refit at full sampling quality

After the smoke succeeded cleanly the user asked for a production run. Same builder, same config, scaled sampling: 4 chains × 1000 tune × 2000 draws, target_accept=0.95.

| metric | value |
|---|---|
| Build time | 5.7 sec |
| Sample time | 871 sec (14.5 min wall-clock with 4 cores) |
| Divergences | 0 |
| Max R-hat | 1.0000 |
| Min ESS bulk | 5826 |
| Sanity check vs NumPy DP | PASS (max \|Δ\| = 5.7e-7) |

### Headline `C` posteriors

| system | `baseline_3s` | `pool_3s` | `pool_3s_abs_by_sd` | exact-tree (prod) | exact vs `pool_3s` |
|---|---:|---:|---:|---:|---:|
| Chicken | 0.297 | 0.362 | 0.300 | **0.252 [0.037, 0.613]** | **−0.110** |
| LLMs | 0.118 | 0.170 | 0.127 | **0.111 [0.005, 0.446]** | **−0.059** |

Mean overall δ_j: 0.110 (exact-tree) vs 0.155 (pool_3s) vs 0.098 (pool_3s_abs_by_sd) vs 0.075 (baseline_3s). The exact-tree fit retains slightly more transmission than the conservative abs_by_sd composite, but less than the d-only-sharing version.

### Sign uncertainty for the weak-undermining cluster (production)

| fit | cluster mean δ̄ [94% CI] | Pr(δ̄ > 0) | Pr(\|δ̄\| < 0.02) |
|---|---:|---:|---:|
| `pool_3s` | +0.060 [-0.025, +0.165] | 0.90 | 0.17 |
| `pool_3s_abs_by_sd` | +0.006 [-0.067, +0.088] | 0.56 | 0.39 |
| **exact-tree (prod)** | **−0.008 [-0.097, +0.080]** | **0.44** | **0.34** |

Under exact-tree the cluster leans slightly negative but stays sign-ambiguous. The "sign-flip resolved under `pool_3s`" framing is fully reversed.

### Focus-cell PPC closure under composite scoring (the awkward finding)

Computed by reconstructing leaf-updated `ρ_m` from primitive β + posterior C draws using the *composite* PPC formula (apples-to-apples with B.10).

| fit | E_cross × Human pred_right | Δright | closure (vs baseline_3s) |
|---|---:|---:|---:|
| `baseline_3s` | 0.632 | -0.228 | — |
| `pool_3s` | 0.697 | -0.163 | **+28%** |
| `pool_3s_abs_by_sd` | 0.654 | -0.206 | +9% |
| exact-tree (prod) | **0.551** | **-0.309** | **-36%** |

| fit | E_cross × ELIZA pred_left | Δleft | closure |
|---|---:|---:|---:|
| `baseline_3s` | 0.616 | -0.344 | — |
| `pool_3s` | 0.703 | -0.257 | +25% |
| `pool_3s_abs_by_sd` | 0.636 | -0.324 | +6% |
| exact-tree (prod) | 0.596 | -0.364 | -6% |

**The exact-tree fit regresses focus-cell PPC closure on the most-important reference cells**, especially `E_cross × Human` (closure goes to -36% — predicted right-tail moves *further* from observed than even `baseline_3s`). This is consistent with the by-system Δℓ split: anchored systems dominate Δℓ magnitude, so the exact-tree refit pulls `(a, κ, β)` in directions that re-explain joint tree structure at some cost to per-cell rating fit.

**Important framing note**: the PPC numbers above use the *composite* leaf-updated formula (apples-to-apples with B.10). A true exact-tree PPC would re-derive `ρ_m` accounting for joint tree structure in the leaf updates; that's a separate metric that would be expected to score the exact-tree fit higher. Under the standard composite-PPC framework, however, the exact-tree fit is worse on the focus cells. This needs explaining at the meeting.

### Synthesis — two coherent DGPs (revised after ChatGPT round-4 reframe)

The framing "exact tree = the right model, composite = a bug" is wrong. Both are coherent DGPs and we (the modellers) get to choose:

- **Latent-state tree** (paper-like): internal feature/subfeature nodes are real binary latent states. Siblings correlated through shared ancestor states. Implemented in the new exact-tree builder. More conservative on `C` because it correctly accounts for evidence redundancy.
- **Leaf-independent marginal tree** (current implementation): tree structure is a parameterisation device for marginal indicator probabilities; no shared latents at the feature/subfeature level. Indicators conditionally independent given `(C, β)`. Better focus-cell PPC because it doesn't impose shared-ancestor correlation.

The composite-vs-exact log-likelihood gap (Δℓ ≈ +10 nats library-wide) tells us these are materially different posteriors, not that one is correct.

The four fits sit in a 2-axis space:

- `pool_3s`: leaf-independent + d-only β_abs pooling. **Inflated** headline `C` due to (a) d-only β_abs sharing artefact (Task 2 confirmed: 70% of cluster δ_j shift was inheritance), (b) leaf-independent over-confident updates on correlated siblings.
- `pool_3s_abs_by_sd`: leaf-independent + (s, d) β_abs pooling. Removes the sharing artefact; regresses composite PPC to 6-9%; conservative on `C`. Diagnostic stepping-stone, not a candidate.
- **`exact-tree (prod)`**: latent-state tree + (s, d) β_abs pooling. Most conservative on `C`. Under proper exact-tree PPC scoring, mixed PPC regression (E_cross × ELIZA partly recovers vs composite scoring; E_cross × Human still substantially worse than pool_3s). This is **not** a measurement artefact — the latent-state model is genuinely more conservative about per-cell predictions because it accounts for evidence redundancy.

The right framing is: under the latent-state DGP, `C_Chicken ≈ 0.25, C_LLMs ≈ 0.11` and the focus cells substantially under-predict (because the model resists over-confident pz1 from correlated siblings). Under the leaf-independent DGP, `C_Chicken ≈ 0.36, C_LLMs ≈ 0.17` and focus cells are better-fit (because the model treats sibling evidence as independent updates). **Pick the DGP, accept the trade-off.**

Outputs:
- `results/gwt_exact_tree/three_state_pooled_abs_by_sd_exact_anchored.nc`
- `results/gwt_exact_tree/three_state_pooled_abs_by_sd_exact_anchored.meta.json`
- `figs_round2/exact_tree_production_headline.csv`
- `figs_round2/exact_tree_production_ppc.csv`
- `figs_round2/exact_tree_production_sign.json`
- `run_exact_tree_production.py` (driver)
- `postprocess_exact_tree_production.py` (post-processing)

---

## Phase B addendum — proper exact-tree PPC (2-pass belief propagation)

After ChatGPT round-4's reframe, the "exact-tree regresses PPC closure" claim was using the wrong yardstick — it scored the exact-tree fit using the *composite* leaf-updated `ρ_m` formula, which assumes leaves are conditionally independent given `(C, β)` (a leaf-independent DGP assumption). The proper exact-tree PPC computes `ρ_m` accounting for joint tree structure: per indicator, `ρ_m` ∝ `P(y_indicator | m) · Σ_z_pa P(z_pa | y_outside_indicator) · π_m(β_for_zpa)`. The "outside" message uses 2-pass belief propagation on the tree.

### Implementation

Standard sum-product belief propagation:

- **Forward pass (bottom-up)**: per node, store `log_D(z_pa)` (downward message) and `log_L_at_zv(z_v)` (node's internal likelihood given own state). Already done in `composite_vs_exact_diagnostic.py:exact_loglik` for the log-lik scalar; here we also retain the intermediates.
- **Backward pass (top-down)**: per node, compute `log_O(z_v) = log P(z_v, y_outside_v_subtree | C)` using parent's outside message + sibling contributions.
- Combine: `P(z_pa | y_outside_indicator) ∝ exp(log_L_pa_at_zv − log_D_indicator + log_O_pa)`.

Implemented in `exact_tree_ppc.py`. Three sanity checks:

1. **`P(m | y_all)` sums to 1** across `m ∈ {0,1,2}` for every indicator + draw. **PASS** (all rows sum to 1.0 within atol=1e-9).
2. **Forward-pass total log-lik matches `exact_loglik`** (the standalone NumPy DP). **PASS** at machine precision (max |Δ| = 0.0 over all systems × draws).
3. Numerical stability: log-sum-exp throughout; clip β to `[1e-12, 1-1e-12]`.

### Focus-cell PPC: composite-style ρ_m vs proper exact-tree ρ_m

Same exact-tree fit; only the `ρ_m` formula differs. n_draws = 500 (subsampled).

| cell | tail | composite ρ_m Δ | exact-tree ρ_m Δ | direction |
|---|---|---:|---:|---|
| E_cross × Human | right | -0.309 | -0.283 | slightly better |
| E_cross × ELIZA | left | -0.364 | -0.263 | meaningfully better (+28%) |
| E_chickenB × Chicken | right | +0.108 | +0.248 | worse |
| E_llmC × LLMs | right | +0.133 | +0.099 | slightly better |

### Apples-to-apples comparison: proper exact-tree PPC vs leaf-independent (pool_3s) PPC

| cell | tail | observed | `baseline_3s` Δ | `pool_3s` Δ | exact-tree (proper) Δ |
|---|---|---:|---:|---:|---:|
| E_cross × Human | right | 0.860 | -0.228 | **-0.163** | -0.283 |
| E_cross × ELIZA | left | 0.960 | -0.344 | **-0.257** | -0.263 |
| E_chickenB × Chicken | right | 0.152 | +0.176 | +0.140 | +0.248 |
| E_llmC × LLMs | right | 0.083 | +0.113 | +0.110 | +0.099 |

### Reading

- **The yardstick was partly wrong, but the trade-off is real.** The proper exact-tree PPC is ~1pp better on E_cross × ELIZA than under composite scoring; on E_cross × Human it's barely different. Net: substantially worse than pool_3s on the most diagnostic cells, regardless of yardstick.
- **The under-prediction on Human is a feature, not a bug.** The exact-tree model is more conservative about updating `pz1` at well-supported indicators because it accounts for evidence redundancy through shared latents. `pool_3s`'s tighter pred at E_cross × Human comes partly from over-confident leaf-independent updates on correlated siblings.
- **Whether to accept this trade-off depends on the DGP commitment** — see synthesis above.

### Outputs
- `figs_round2/exact_tree_production_ppc_BP.csv` — proper exact-tree PPC results
- `exact_tree_ppc.py` — implementation (`forward_pass`, `backward_pass`, `per_indicator_exact_tree_marginal`, `focus_cell_ppc_exact_tree`)

---

## Files

All outputs in `notebooks/meeting_prep_arvo_2026-04-27/figs_round2/`:

- Task 1: `composite_gap_summary.json`, `composite_gap_headline_shifts.csv`, `sibling_correlation_scores.csv`, `composite_gap.png`
- Task 2: `beta_abs_neutral_counterfactual.csv`, `beta_abs_neutral_counterfactual_decomp.csv`
- Task 3: `sigma_pool_sweep_summary.csv`, `sigma_pool_sweep_qj.png`
- Task 4: `results/gwt_tree_pooling/three_state_pooled_abs_by_sd_anchored.nc` + `.meta.json`, plus `figs_round2/abs_by_sd_signflip_delta.csv`, `figs_round2/abs_by_sd_system_C.csv`

Scripts (all reproducible from `notebooks/meeting_prep_arvo_2026-04-27/`):

- `composite_vs_exact_diagnostic.py` (Task 1)
- `beta_abs_neutral_counterfactual.py` (Task 2)
- `sigma_pool_prior_predictive_sweep.py` (Task 3)
- `run_tree_prior_pooling_abs_by_sd.py` (Task 4 driver) + `postprocess_abs_by_sd.py` (Task 4 post-processing) + model-code edits in `dcm_model.py` (`BETA_ABS_BY_SUPPORT_DEMAND` flag) and `analyse_tree_pooling.py` (β_abs auto-detection of grouping)
