# Oracle Root-Signal Summary

Labels:

- DGP: `exact_latent_tree`
- Fit: `oracle`
- Leaf: `three_state_binomial_2`
- Nuisance truth: `exact_tree_production_medians`
- Design: `current_gwt_rater_design`

C grid points: 103

## Plain-English Interpretation

This audit asks whether the current ordinal ratings would visibly change if the root consciousness value `C_s` changed, assuming the exact tree and nuisance parameters are correct.

The output is a root-signal screen, not a full posterior fit. It fixes the tree and observation layer, computes the expected category distribution for each indicator at each candidate `C`, and then asks how separated those expected ratings are from the reference `C`.

The KL numbers are expected log-evidence differences, in nats, from the marginal ordinal rating distributions. Bigger means the design should more strongly distinguish the two root values. Near zero means those root values look similar on the observed rating scale.

Under the LLM rating design, the KL separation between `C=0.10` and `C=0.25` is 0.232 nats. Under the Chicken rating design, the reverse comparison is 0.121 nats. This is the practical check for whether LLM-vs-Chicken root differences are visible in the current data design before fitting.

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
| 2024 Leading Chat LLMs | 0.100 | 1.000 | 0.320 | 0.420 | yes |
| 2024 Leading Chat LLMs | 0.100 | 2.000 | 0.450 | 0.550 | yes |
| 2024 Leading Chat LLMs | 0.100 | 5.000 | 0.699 | 0.799 | yes |
| 2024 Leading Chat LLMs | 0.100 | 10.000 | not reached | not reached | no |
| Chicken | 0.250 | 1.000 | 0.440 | 0.690 | yes |
| Chicken | 0.250 | 2.000 | 0.609 | 0.859 | yes |
| Chicken | 0.250 | 5.000 | not reached | not reached | no |
| Chicken | 0.250 | 10.000 | not reached | not reached | no |
| ELIZA | 0.001 | 1.000 | 0.609 | 0.610 | yes |
| ELIZA | 0.001 | 2.000 | 0.848 | 0.849 | yes |
| ELIZA | 0.001 | 5.000 | not reached | not reached | no |
| ELIZA | 0.001 | 10.000 | not reached | not reached | no |
| Human | 0.999 | 1.000 | 0.609 | 0.390 | yes |
| Human | 0.999 | 2.000 | 0.858 | 0.141 | yes |
| Human | 0.999 | 5.000 | not reached | not reached | no |
| Human | 0.999 | 10.000 | not reached | not reached | no |

## Pairwise KL At Reference Truths

| system | reference_C | comparison_C | kl_marginal_nats | kl_per_rating |
|---|---|---|---|---|
| 2024 Leading Chat LLMs | 0.100 | 0.001 | 0.105 | 0.001 |
| 2024 Leading Chat LLMs | 0.100 | 0.100 | 0.000 | 0.000 |
| 2024 Leading Chat LLMs | 0.100 | 0.250 | 0.232 | 0.001 |
| 2024 Leading Chat LLMs | 0.100 | 0.999 | 8.940 | 0.048 |
| Chicken | 0.250 | 0.001 | 0.339 | 0.004 |
| Chicken | 0.250 | 0.100 | 0.121 | 0.001 |
| Chicken | 0.250 | 0.250 | 0.000 | 0.000 |
| Chicken | 0.250 | 0.999 | 3.245 | 0.035 |
| ELIZA | 0.001 | 0.001 | 0.000 | 0.000 |
| ELIZA | 0.001 | 0.100 | 0.027 | 0.001 |
| ELIZA | 0.001 | 0.250 | 0.169 | 0.003 |
| ELIZA | 0.001 | 0.999 | 2.875 | 0.057 |
| Human | 0.999 | 0.001 | 2.736 | 0.055 |
| Human | 0.999 | 0.100 | 2.212 | 0.044 |
| Human | 0.999 | 0.250 | 1.540 | 0.031 |
| Human | 0.999 | 0.999 | 0.000 | 0.000 |
