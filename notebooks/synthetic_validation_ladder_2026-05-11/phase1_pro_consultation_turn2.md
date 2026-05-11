# Phase 1 Pro consultation, turn 2: numerical corrections to the top-level bound + asymmetric-sweep mismatch + smaller clarifications

Thanks — the Phase 1A protocol (function signatures, CSV columns, interpretation matrix) is excellent and I'll implement essentially as written. Two numerical corrections from the actual GWT tree change the *expected* result of the pre-flight and the hypothesis ranking, and one repo-specific finding sharpens what your positive-control rung is actually testing. None of these invalidate the structural framework — they shift what the diagnostics are predicted to show.

## Correction 1: actual GWT top-level evidence is much higher than estimated

Your "root-child evidence bottleneck" argument assumed: 6 top-level features, all with $\mu_{\mathrm{pres}} \approx 0.6$ and $\mu_{\mathrm{abs}} \approx 0.4$, giving expected top-state evidence $\approx 0.81$ nats — close to my reported R=1 median $\log B \approx 0.99$, which made the bottleneck interpretation seem decisive.

**Actual GWT tree**, computed via `EvidenceProcessor.get_beta_parameters` on the 7 top-level labels in `data_cache.json`:

| Top-level feature | Support | Demandingness | $\mu_{\mathrm{pres}}$ | $\mu_{\mathrm{abs}}$ | gap |
|---|---|---|---|---|---|
| Representationality | weak | strongly undemanding | 0.923 | 0.889 | +0.034 |
| Hierarchical Organization | weak | moderately undemanding | 0.812 | 0.750 | +0.062 |
| Coherence | strong | strongly demanding | 0.750 | 0.111 | +0.639 |
| Modularity | weak | moderately demanding | 0.400 | 0.250 | +0.150 |
| Complexity | strong | weakly undemanding | 0.914 | 0.571 | +0.343 |
| Selective Attention | strong | moderately demanding | 0.800 | 0.250 | +0.550 |
| Integration | strong | moderately demanding | 0.800 | 0.250 | +0.550 |

**Seven top-level features, not six. Highly heterogeneous gaps:** three weak-support features have negligible gaps (≤ 0.15), four strong-support features have substantial gaps (0.34–0.64).

Analytical top-state evidence bounds (averaging $\mu \log(\mu_p / \mu_a) + (1-\mu)\log((1-\mu_p)/(1-\mu_a))$ across all seven children, with $\mu = \mu_p$ for $R=1$ and $\mu = \mu_a$ for $R=0$):

$$
\mathbb{E}[\log B_{\mathrm{top}} \mid R=1] = \mathbf{+2.81 \text{ nats}},
\qquad
\mathbb{E}[\log B_{\mathrm{top}} \mid R=0] = \mathbf{-2.81 \text{ nats}}.
$$

Both clear the prior-odds thresholds substantially: $+2.81 \gg +1.61$ ($\rho > 0.5$) and $-2.81$ is essentially at the strong-absent threshold $-2.99$ ($\rho < 0.05$). So under perfect observation of the top-level latent states, the model should produce decisive evidence in both directions on average.

**Implication:** the analytical pre-flight you recommended will *pass*, not fail, under the published priors. The bottleneck is downstream of the top-level features — somewhere among depth-3 propagation, leaf observation, ordinal emission, three-state $m_j$, nuisance marginalisation, and rater design. The empirical R=1 median $\log B \approx 0.99$ represents only ~35% of the analytical top-bound surviving — substantial information loss, but the loss is happening *below* the top-child layer, not at it.

This reorders the hypothesis ranking. Your "Hypothesis 1 (root-child / tree-transmission bottleneck)" is *partially refuted* under the published priors: the top-level layer has plenty of analytical evidence; the question is what fraction of it survives downstream. H2 (nuisance washout), H3 (observation-layer info loss), and H4 (rater design) move up.

**Caveat:** my analytical calculation is an upper bound under the prior means. Actual realised top-state distributions per replicate will have variance around it, and the *median* (not mean) realised $\log B_{\mathrm{top}}$ might still fail. The full per-replicate distribution is exactly what your pre-flight should compute.

## Correction 2: the asymmetric-prior sweep tested a different question than expected

Your debug checklist asked: "did the (0.90, 0.10) override affect the *generator true β*, or only the *fitted priors*?" I checked the generator code:

`notebooks/synthetic_validation_2026-05-06/gwt_full_exact_recovery.py:212`:

```python
truth = load_oracle_truth("exact_tree_production_medians", source_stance_data, cfg)
# ...
latent_by_system[system] = sample_latent_tree_for_system(
    rng, source_stance_data, truth.edge_betas, TRUE_C_BY_SYSTEM[system],
)
```

The synthetic generator uses `truth.edge_betas` derived from `exact_tree_production_medians` — i.e., the production posterior medians on real data, which sit close to the published-prior means. The asymmetric override `BETA_PRES_OVERRIDE_MEAN=0.90, BETA_ABS_OVERRIDE_MEAN=0.10` applied only to the *fitter's prior centre*, not to the generator's true β.

**So the asymmetric sweep was a *prior–DGP mismatch* test, not a "what if true β were extreme" test.** Neither it nor any prior synthetic experiment in the repo has tested the scenario where both generator truth and fit prior are at $(0.90, 0.10)$. Your positive-control rung is therefore a genuinely new experiment — not a redundant check on a question already explored. I'll promote it to a first-class part of the ladder.

This also means the empirical "asymmetric prior didn't visibly improve $\rho$ recovery" finding I reported earlier should be reinterpreted: it tested *robustness to a prior-DGP mismatch*, and the result that $\rho$ recovery was insensitive to the asymmetric prior is consistent with the data simply not having much curvature on β (so the posterior on β is dominated by the prior, but the resulting downstream propagation barely changed since the truth-β was unchanged).

## Smaller clarifications

3. **Chicken has zero cross-system rater coverage.** Rater_B (the only cross-system rater) covers Human + LLM + ELIZA but **not Chicken**. So Chicken is structurally the *worst* free target for rater identifiability — not just bad, but structurally unanchored across systems. Your "improved crossed design" Rung 7 should specifically include Chicken in the cross-rater coverage. The `3 crossed raters × 4 systems × all leaves` recipe naturally fixes this.

4. **GWT leaf count.** Confirmed ~30 indicators across the 7 top-level features (1 to 6 subfeatures each, with leaves at depth 2 or 3). Your "3 crossed raters × 4 systems × all GWT indicators ≈ 360 ratings" estimate lands close to the production total of 379 (Human 50 + Chicken 93 + LLM 186 + ELIZA 50 = 379). So the budget-comparable claim holds.

5. **Your `log_B_eff = logit(ρ) − logit(p₀)` derivation is correct** under collapsed-rho with $p_0 = 1/6$. I'll use this as the primary scalar comparable to oracle log_B in Phase 1A, with posterior mean / median / quantiles of per-draw log_B as secondary diagnostics, exactly as you specified.

6. **Reinterpreting the bimodality under R=1.** Given correction (1), your reading "some realised R=1 systems happen to light up enough high-level branches; others die near the top" is consistent with the analytical *mean* being high but the *variance* across realised top-state configurations being substantial. Specifically: under R=1, the realised $z_{\mathrm{top}}$ is sampled from $\prod_u \mathrm{Bernoulli}(\beta^{\mathrm{pres}}_u)$, and the resulting log_B varies per realisation. With three weak-gap features (combined evidence near 0) and four strong-gap features, the R=1 distribution of realised log_B_top will likely have substantial spread, possibly bimodal if the four strong-gap features tend to fire together vs apart. The analytical *distribution* — not just mean — should be a first output of the pre-flight.

## Concrete asks for your response

1. **Revised hypothesis ranking under correction (1).** Given the analytical mean top-bound of +2.81 nats (R=1) / −2.81 nats (R=0), which of H2–H4 is now most likely the binding downstream constraint? Does the heterogeneous-gap structure (3 weak + 4 strong) suggest any specific failure mode I should prioritise testing?

2. **Expected pre-flight outcome and decision branches.** What's the right pre-flight protocol given that the analytical *mean* passes? Specifically:
   - Should I compute the *full distribution* of realised log_B_top under R=1 and R=0 (analytical or by Monte Carlo over $z_{\mathrm{top}}$ realisations) and report the median plus tail quantiles?
   - What pre-flight gate should I use: median realised log_B_top crossing the threshold, or some fraction crossing?
   - If the pre-flight passes (as I expect), do you still want the positive-control β_true = (0.90, 0.10) rung as Rung 1A, or only as a debug check if Rung 1 fails?

3. **The genuinely new positive-control rung.** Your Rung 1 with β_true = (0.90, 0.10) is a real new experiment (not a redundant check on the asymmetric sweep). What specifically should it test: just `passes / fails recovery`, or a richer comparison against published-prior-truth Rung 1 to quantify how much downstream loss is "transmission gap" vs "observation/leaf/nuisance"? Should I run both side-by-side with matched seeds for a clean comparison?

4. **Chicken-specific Rung 7 design.** Given Chicken has zero cross-system rater coverage in production, Rung 6 (production-like) and Rung 7 (improved crossed) might show very different per-system behaviour. Should I score per-system metrics separately (Chicken vs LLM) at Rung 7 to isolate the Chicken-specific design fix? This would tell me whether a future-survey design needs to specifically target Chicken cross-rater coverage, vs uniform 3× cross-rater coverage being sufficient.

5. **Add a "matched-prior nuisance washout" rung?** The current ladder has Rung 5 = inferred β + inferred (a, κ, b) under fully crossed raters. Given correction (2), would it be cleaner to add a rung where β_true matches the prior mean exactly (so the prior-DGP mismatch from the existing asymmetric sweep is removed) — testing pure nuisance washout independent of the prior-mismatch confound?

6. **Anything else the corrections change.** If correction (1) substantially shifts your priors on which rungs are likely to fail or pass, please flag where else my Phase 1B implementation should differ from your turn-1 spec.

## Constraints unchanged

 hand-off to coding agent, push back on premises if these corrections themselves are subtly wrong, and please show your reasoning where it informs the revised ranking. Pro turn 1 protocol design (function signatures, CSV columns, interpretation matrix, scalar pass rule) stays as-is unless you flag specific changes.
