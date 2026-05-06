# Oracle Internal-Node Identifiability Summary

Labels:

- DGP: `exact_latent_tree`
- Fit: `oracle`
- Leaf: `three_state_binomial_2`
- Nuisance truth: `exact_tree_production_medians`
- Design: `current_gwt_rater_design`

Simulations: 50
Internal nodes per system: 25

## Plain-English Interpretation

This audit asks a practical question: if the exact binary GWT tree were really how expert ratings are generated, would the current rating design let us learn the hidden feature/subfeature states?

Short answer: the answer is mixed. On average the ratings remove about 0.439 of the prior uncertainty about internal states, with mean Brier score 0.110. That means the internal probabilities are not purely arbitrary, but they are also not uniformly strong across the tree.

Using production-median nuisance truth is a best-case version of the current exact-tree model: the observation layer and tree transmission are set to values the real-data fit already found plausible.

For Arvo's question, the important point is that internal states should not be interpreted as recovered true/false labels. They are posterior probabilities. Some feature blocks have enough descendant ratings and tree transmission to move those probabilities substantially; other nodes mostly inherit information from ancestors and priors.

The clearest weak spot is the deep/empty part of the tree: depth-3 nodes have mean relative entropy reduction 0.048, and nodes with zero subtree ratings have 0.048. Those nodes should not support strong scientific claims in the current design.

The clearest strong spot is broad, connected structure: fanout-6 nodes have mean relative entropy reduction 0.831. These are the kinds of internal summaries where the exact-tree model has actual observed-scale leverage.

What this means for the model results: root `C_s` summaries and well-observed feature-block probabilities are more defensible than hard claims about every individual subfeature. The right reporting style is probability plus uncertainty/entropy by node, not a table of binary recovered states.

How to read the metrics:

- Relative entropy reduction is the fraction of uncertainty removed by the ratings: 0 means no learning beyond the prior/tree, 1 means near complete resolution.
- Brier score is a probability-error score: lower is better; around 0.25 is what a non-informative 50/50 probability gets for balanced binary states.
- Calibration compares probability bins to the simulated truth rate. Good calibration means a bin around 0.8 contains true `z=1` states about 80% of the time.

This is a best-case information check, not a proof that the real world is an exact binary tree. If a node is weak here, a full real-data fit cannot make it strongly data-driven without relying on priors or model structure. If a node is strong here, it is at least identifiable in principle under the current design.

## Overall

| nuisance_truth | n | mean_relative_entropy_reduction | mean_brier | mean_neg_log_score |
|---|---|---|---|---|
| exact_tree_production_medians | 5000 | 0.439 | 0.110 | 0.348 |

## By System

| group_value | n | mean_relative_entropy_reduction | mean_brier | mean_neg_log_score |
|---|---|---|---|---|
| 2024 Leading Chat LLMs | 1250 | 0.509 | 0.112 | 0.354 |
| Chicken | 1250 | 0.464 | 0.108 | 0.343 |
| ELIZA | 1250 | 0.393 | 0.117 | 0.366 |
| Human | 1250 | 0.388 | 0.101 | 0.328 |

## By Depth

| group_value | n | mean_relative_entropy_reduction | mean_brier | mean_neg_log_score |
|---|---|---|---|---|
| 1 | 1400 | 0.443 | 0.097 | 0.310 |
| 2 | 3200 | 0.486 | 0.103 | 0.332 |
| 3 | 400 | 0.048 | 0.208 | 0.604 |

## By Fanout

| group_value | n | mean_relative_entropy_reduction | mean_brier | mean_neg_log_score |
|---|---|---|---|---|
| 0 | 400 | 0.048 | 0.208 | 0.604 |
| 1 | 400 | 0.341 | 0.148 | 0.454 |
| 2 | 1600 | 0.447 | 0.100 | 0.323 |
| 3 | 1200 | 0.523 | 0.105 | 0.336 |
| 4 | 800 | 0.392 | 0.101 | 0.322 |
| 5 | 400 | 0.535 | 0.088 | 0.297 |
| 6 | 200 | 0.831 | 0.024 | 0.098 |

## By Subtree Rating Count

| group_value | n | mean_relative_entropy_reduction | mean_brier | mean_neg_log_score |
|---|---|---|---|---|
| 0 | 400 | 0.048 | 0.208 | 0.604 |
| 1-2 | 1200 | 0.356 | 0.115 | 0.365 |
| 11-25 | 850 | 0.524 | 0.085 | 0.283 |
| 26-50 | 200 | 0.433 | 0.124 | 0.396 |
| 3-5 | 1300 | 0.506 | 0.108 | 0.344 |
| 6-10 | 1050 | 0.530 | 0.085 | 0.279 |

## Calibration

| prob_bin | n | mean_posterior_p | empirical_z_rate | mean_brier |
|---|---|---|---|---|
| [0.0, 0.1) | 1030 | 0.030 | 0.035 | 0.032 |
| [0.1, 0.2) | 332 | 0.145 | 0.190 | 0.154 |
| [0.2, 0.3) | 218 | 0.246 | 0.220 | 0.172 |
| [0.3, 0.4) | 210 | 0.351 | 0.319 | 0.218 |
| [0.4, 0.5) | 274 | 0.449 | 0.438 | 0.245 |
| [0.5, 0.6) | 232 | 0.550 | 0.634 | 0.239 |
| [0.6, 0.7) | 273 | 0.651 | 0.645 | 0.230 |
| [0.7, 0.8) | 453 | 0.754 | 0.759 | 0.181 |
| [0.8, 0.9) | 458 | 0.857 | 0.834 | 0.139 |
| [0.9, 1.0) | 1520 | 0.967 | 0.966 | 0.033 |
