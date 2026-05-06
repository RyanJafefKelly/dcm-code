# Oracle Internal-Node Identifiability Summary

Labels:

- DGP: `exact_latent_tree`
- Fit: `oracle`
- Leaf: `three_state_binomial_2`
- Nuisance truth: `paper_mean_tree_transmission`
- Design: `current_gwt_rater_design`

Simulations: 50
Internal nodes per system: 25

## Plain-English Interpretation

This audit asks a practical question: if the exact binary GWT tree were really how expert ratings are generated, would the current rating design let us learn the hidden feature/subfeature states?

Short answer: the answer is mixed. On average the ratings remove about 0.263 of the prior uncertainty about internal states, with mean Brier score 0.148. That means the internal probabilities are not purely arbitrary, but they are also not uniformly strong across the tree.

Using paper-mean tree transmission makes the internal states much harder to learn. This is the conservative stress case: if these edge strengths are closer to reality, many middle-layer posteriors will be only weakly data-driven.

For Arvo's question, the important point is that internal states should not be interpreted as recovered true/false labels. They are posterior probabilities. Some feature blocks have enough descendant ratings and tree transmission to move those probabilities substantially; other nodes mostly inherit information from ancestors and priors.

The clearest weak spot is the deep/empty part of the tree: depth-3 nodes have mean relative entropy reduction 0.010, and nodes with zero subtree ratings have 0.010. Those nodes should not support strong scientific claims in the current design.

The clearest strong spot is broad, connected structure: fanout-6 nodes have mean relative entropy reduction 0.634. These are the kinds of internal summaries where the exact-tree model has actual observed-scale leverage.

What this means for the model results: root `C_s` summaries and well-observed feature-block probabilities are more defensible than hard claims about every individual subfeature. The right reporting style is probability plus uncertainty/entropy by node, not a table of binary recovered states.

How to read the metrics:

- Relative entropy reduction is the fraction of uncertainty removed by the ratings: 0 means no learning beyond the prior/tree, 1 means near complete resolution.
- Brier score is a probability-error score: lower is better; around 0.25 is what a non-informative 50/50 probability gets for balanced binary states.
- Calibration compares probability bins to the simulated truth rate. Good calibration means a bin around 0.8 contains true `z=1` states about 80% of the time.

This is a best-case information check, not a proof that the real world is an exact binary tree. If a node is weak here, a full real-data fit cannot make it strongly data-driven without relying on priors or model structure. If a node is strong here, it is at least identifiable in principle under the current design.

## Overall

| nuisance_truth | n | mean_relative_entropy_reduction | mean_brier | mean_neg_log_score |
|---|---|---|---|---|
| paper_mean_tree_transmission | 5000 | 0.263 | 0.148 | 0.454 |

## By System

| group_value | n | mean_relative_entropy_reduction | mean_brier | mean_neg_log_score |
|---|---|---|---|---|
| 2024 Leading Chat LLMs | 1250 | 0.314 | 0.146 | 0.445 |
| Chicken | 1250 | 0.276 | 0.143 | 0.440 |
| ELIZA | 1250 | 0.229 | 0.157 | 0.479 |
| Human | 1250 | 0.234 | 0.147 | 0.451 |

## By Depth

| group_value | n | mean_relative_entropy_reduction | mean_brier | mean_neg_log_score |
|---|---|---|---|---|
| 1 | 1400 | 0.262 | 0.126 | 0.395 |
| 2 | 3200 | 0.296 | 0.148 | 0.454 |
| 3 | 400 | 0.010 | 0.233 | 0.657 |

## By Fanout

| group_value | n | mean_relative_entropy_reduction | mean_brier | mean_neg_log_score |
|---|---|---|---|---|
| 0 | 400 | 0.010 | 0.233 | 0.657 |
| 1 | 400 | 0.242 | 0.152 | 0.474 |
| 2 | 1600 | 0.271 | 0.144 | 0.446 |
| 3 | 1200 | 0.319 | 0.151 | 0.459 |
| 4 | 800 | 0.202 | 0.136 | 0.424 |
| 5 | 400 | 0.280 | 0.140 | 0.436 |
| 6 | 200 | 0.634 | 0.058 | 0.186 |

## By Subtree Rating Count

| group_value | n | mean_relative_entropy_reduction | mean_brier | mean_neg_log_score |
|---|---|---|---|---|
| 0 | 400 | 0.010 | 0.233 | 0.657 |
| 1-2 | 1200 | 0.222 | 0.157 | 0.484 |
| 11-25 | 850 | 0.317 | 0.134 | 0.414 |
| 26-50 | 200 | 0.196 | 0.154 | 0.467 |
| 3-5 | 1300 | 0.304 | 0.147 | 0.449 |
| 6-10 | 1050 | 0.326 | 0.118 | 0.378 |

## Calibration

| prob_bin | n | mean_posterior_p | empirical_z_rate | mean_brier |
|---|---|---|---|---|
| [0.0, 0.1) | 513 | 0.046 | 0.045 | 0.042 |
| [0.1, 0.2) | 391 | 0.151 | 0.156 | 0.132 |
| [0.2, 0.3) | 333 | 0.250 | 0.243 | 0.184 |
| [0.3, 0.4) | 311 | 0.350 | 0.354 | 0.225 |
| [0.4, 0.5) | 346 | 0.451 | 0.439 | 0.245 |
| [0.5, 0.6) | 421 | 0.549 | 0.539 | 0.248 |
| [0.6, 0.7) | 451 | 0.652 | 0.663 | 0.224 |
| [0.7, 0.8) | 550 | 0.748 | 0.738 | 0.193 |
| [0.8, 0.9) | 744 | 0.856 | 0.853 | 0.125 |
| [0.9, 1.0) | 940 | 0.949 | 0.945 | 0.051 |
