# Oracle Root-Signal Summary

Labels:

- DGP: `exact_latent_tree`
- Fit: `oracle`
- Leaf: `three_state_binomial_2`
- Nuisance truth: `paper_mean_tree_transmission`
- Design: `current_gwt_rater_design`

C grid points: 103

## Plain-English Interpretation

This audit asks whether the current ordinal ratings would visibly change if the root consciousness value `C_s` changed, assuming the exact tree and nuisance parameters are correct.

The output is a root-signal screen, not a full posterior fit. It fixes the tree and observation layer, computes the expected category distribution for each indicator at each candidate `C`, and then asks how separated those expected ratings are from the reference `C`.

The KL numbers are expected log-evidence differences, in nats, from the marginal ordinal rating distributions. Bigger means the design should more strongly distinguish the two root values. Near zero means those root values look similar on the observed rating scale.

Under the LLM rating design, the KL separation between `C=0.10` and `C=0.25` is 0.057 nats. Under the Chicken rating design, the reverse comparison is 0.030 nats. This is the practical check for whether LLM-vs-Chicken root differences are visible in the current data design before fitting.

How to use this result: if the KL curve is shallow around a system's reference `C`, then root posterior concentration in a full model will need strong help from priors, anchors, or shared nuisance learning. If the curve is steep, the observed ratings themselves contain real root signal under the exact-tree semantics.

Caveat: this is a marginal-rating KL, so it intentionally ignores the extra sibling dependence induced by shared internal states. The dependence/block audit is the separate check for that structure.

## Current Rating Counts

| system | rating_count |
|---|---|
| 2024 Leading Chat LLMs | 186 |
| Chicken | 93 |
| ELIZA | 50 |
| Human | 50 |

## KL Threshold Crossings

Minimum root-C movement needed to reach each marginal KL threshold.

| system | reference_C | kl_threshold | min_abs_delta_C | nearest_C_at_threshold | reached |
|---|---|---|---|---|---|
| 2024 Leading Chat LLMs | 0.100 | 1.000 | 0.630 | 0.730 | yes |
| 2024 Leading Chat LLMs | 0.100 | 2.000 | 0.869 | 0.969 | yes |
| 2024 Leading Chat LLMs | 0.100 | 5.000 | not reached | not reached | no |
| 2024 Leading Chat LLMs | 0.100 | 10.000 | not reached | not reached | no |
| Chicken | 0.250 | 1.000 | not reached | not reached | no |
| Chicken | 0.250 | 2.000 | not reached | not reached | no |
| Chicken | 0.250 | 5.000 | not reached | not reached | no |
| Chicken | 0.250 | 10.000 | not reached | not reached | no |
| ELIZA | 0.001 | 1.000 | not reached | not reached | no |
| ELIZA | 0.001 | 2.000 | not reached | not reached | no |
| ELIZA | 0.001 | 5.000 | not reached | not reached | no |
| ELIZA | 0.001 | 10.000 | not reached | not reached | no |
| Human | 0.999 | 1.000 | not reached | not reached | no |
| Human | 0.999 | 2.000 | not reached | not reached | no |
| Human | 0.999 | 5.000 | not reached | not reached | no |
| Human | 0.999 | 10.000 | not reached | not reached | no |

## Pairwise KL At Reference Truths

| system | reference_C | comparison_C | kl_marginal_nats | kl_per_rating |
|---|---|---|---|---|
| 2024 Leading Chat LLMs | 0.100 | 0.001 | 0.026 | 0.000 |
| 2024 Leading Chat LLMs | 0.100 | 0.100 | 0.000 | 0.000 |
| 2024 Leading Chat LLMs | 0.100 | 0.250 | 0.057 | 0.000 |
| 2024 Leading Chat LLMs | 0.100 | 0.999 | 2.156 | 0.012 |
| Chicken | 0.250 | 0.001 | 0.083 | 0.001 |
| Chicken | 0.250 | 0.100 | 0.030 | 0.000 |
| Chicken | 0.250 | 0.250 | 0.000 | 0.000 |
| Chicken | 0.250 | 0.999 | 0.776 | 0.008 |
| ELIZA | 0.001 | 0.001 | 0.000 | 0.000 |
| ELIZA | 0.001 | 0.100 | 0.007 | 0.000 |
| ELIZA | 0.001 | 0.250 | 0.041 | 0.001 |
| ELIZA | 0.001 | 0.999 | 0.686 | 0.014 |
| Human | 0.999 | 0.001 | 0.666 | 0.013 |
| Human | 0.999 | 0.100 | 0.539 | 0.011 |
| Human | 0.999 | 0.250 | 0.374 | 0.007 |
| Human | 0.999 | 0.999 | 0.000 | 0.000 |
