# Synthetic Validation Run Log

## exact_unpooled_paper_mean_seed20260506

- Status: generated synthetic data only; no model fit yet.
- Tree semantics: exact latent-state tree.
- Edge betas: unpooled paper prior means.
- Indicator leaf: three-state.
- Observation layer: no expert shifts; `a` and `kappa` from exact production posterior medians.
- Counts matched current rater design: Human 50, Chicken 93, LLMs 186, ELIZA 50.
- First sanity concern: this DGP is much less anchor-like than observed Human/ELIZA data. The generated Human top-category mass and ELIZA bottom-category mass should be compared against the observed-data reference in `sanity_checks.json` before treating this as a realistic smoke DGP.

## oracle_internal_identifiability / exact_tree_production_medians_seed20260506

- Status: completed oracle audit; no PyMC sampling.
- Labels: DGP `exact_latent_tree`; fit `oracle`; leaf `three_state_binomial_2`; nuisance truth `exact_tree_production_medians`; design `current_gwt_rater_design`.
- Command: `.venv/bin/python notebooks/synthetic_validation_2026-05-06/gwt_oracle_internal_identifiability.py --overwrite`
- Simulations: 50; internal nodes: 25; scored probabilities: 5,000.
- Overall mean relative entropy reduction: 0.439; mean Brier score: 0.110; mean negative log score: 0.348.
- Main pattern: depth-3 / fanout-0 nodes with zero subtree ratings are effectively weakly identified (relative entropy reduction 0.048), while high-fanout and moderate-subtree-count nodes show much stronger probability recovery.
- Practical reading: the current design supports some internal feature-block probability statements, especially broad connected parts of the tree, but it does not justify treating every internal state as a recovered true/false label.
- Output directory: `runs/oracle_internal_identifiability/exact_latent_tree__oracle__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506/`

## oracle_internal_identifiability / paper_mean_tree_transmission_seed20260506

- Status: completed oracle stress audit; no PyMC sampling.
- Labels: DGP `exact_latent_tree`; fit `oracle`; leaf `three_state_binomial_2`; nuisance truth `paper_mean_tree_transmission`; design `current_gwt_rater_design`.
- Command: `.venv/bin/python notebooks/synthetic_validation_2026-05-06/gwt_oracle_internal_identifiability.py --nuisance-truth paper_mean_tree_transmission --overwrite`
- Simulations: 50; internal nodes: 25; scored probabilities: 5,000.
- Overall mean relative entropy reduction: 0.263; mean Brier score: 0.148; mean negative log score: 0.454.
- Main pattern: paper-mean transmission makes internal states substantially less identifiable than production medians; depth-3 / fanout-0 nodes are almost entirely prior/ancestor driven (relative entropy reduction 0.010).
- Practical reading: if paper-mean transmission is closer to reality, many internal posteriors in the current model should be presented as weak, prior-regularised summaries rather than learned feature states.
- Output directory: `runs/oracle_internal_identifiability/exact_latent_tree__oracle__three_state_binomial_2__paper_mean_tree_transmission__current_gwt_rater_design__seed20260506/`

## oracle_root_signal / exact_tree_production_medians

- Status: completed deterministic oracle audit; no PyMC sampling.
- Labels: DGP `exact_latent_tree`; fit `oracle`; leaf `three_state_binomial_2`; nuisance truth `exact_tree_production_medians`; design `current_gwt_rater_design`.
- Command: `.venv/bin/python notebooks/synthetic_validation_2026-05-06/gwt_oracle_root_signal.py --overwrite`
- Grid: 103 root-`C` values including the four reference truths.
- Main metric: marginal ordinal-rating KL summed over the current rating slots.
- LLM design KL for `C=0.10` versus `C=0.25`: 0.232 nats; Chicken design KL for `C=0.25` versus `C=0.10`: 0.121 nats.
- Practical reading: the current design has weak marginal root signal for LLM-vs-Chicken-scale differences; posterior concentration near those values will need help from priors, anchors, or shared nuisance learning.
- Output directory: `runs/oracle_root_signal/exact_latent_tree__oracle__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design/`

## oracle_root_signal / paper_mean_tree_transmission

- Status: completed deterministic oracle stress audit; no PyMC sampling.
- Labels: DGP `exact_latent_tree`; fit `oracle`; leaf `three_state_binomial_2`; nuisance truth `paper_mean_tree_transmission`; design `current_gwt_rater_design`.
- Command: `.venv/bin/python notebooks/synthetic_validation_2026-05-06/gwt_oracle_root_signal.py --nuisance-truth paper_mean_tree_transmission --overwrite`
- Grid: 103 root-`C` values including the four reference truths.
- Main metric: marginal ordinal-rating KL summed over the current rating slots.
- LLM design KL for `C=0.10` versus `C=0.25`: 0.057 nats; Chicken design KL for `C=0.25` versus `C=0.10`: 0.030 nats.
- Practical reading: under paper-mean tree transmission, the current marginal rating design provides very little root signal around the LLM/Chicken range.
- Output directory: `runs/oracle_root_signal/exact_latent_tree__oracle__three_state_binomial_2__paper_mean_tree_transmission__current_gwt_rater_design/`

## full_exact_recovery / exact_tree_production_medians_seed20260506_smoke

- Status: completed pipeline smoke; diagnostics intentionally exploratory due to tiny chain budget.
- Labels: DGP `exact_latent_tree`; fit `full_exact_tree`; leaf `three_state_binomial_2`; nuisance truth `exact_tree_production_medians`; design `current_gwt_rater_design`.
- Command: `.venv/bin/python notebooks/synthetic_validation_2026-05-06/gwt_full_exact_recovery.py --smoke --overwrite`
- Sampling: 2 chains × 80 tune × 80 draws, target_accept 0.9.
- Diagnostics: 0 divergences; max R-hat 1.09; min bulk ESS 76; status `exploratory_failed`.
- Purpose: verified the synthetic generation, exact-tree fitting, netcdf writing, and all recovery postprocessing outputs before the full run.
- Output directory: `runs/full_exact_recovery/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__smoke/`

## full_exact_recovery / exact_tree_production_medians_seed20260506

- Status: completed full one-seed exact/exact fake-data recovery pilot.
- Labels: DGP `exact_latent_tree`; fit `full_exact_tree`; leaf `three_state_binomial_2`; nuisance truth `exact_tree_production_medians`; design `current_gwt_rater_design`.
- Command: `.venv/bin/python notebooks/synthetic_validation_2026-05-06/gwt_full_exact_recovery.py --overwrite`
- Sampling: 4 chains × 1000 tune × 1000 draws, target_accept 0.95.
- Diagnostics: 0 divergences; max R-hat 1.0000; min bulk ESS 3181; status `passed`.
- Free root recovery: Chicken truth 0.25, posterior median 0.113 with 94% interval [0.006, 0.439]; LLM truth 0.10, posterior median 0.159 with 94% interval [0.010, 0.546]. Both truths are covered, but intervals are wide and the one-seed medians move in opposite directions.
- Label beta recovery: beta_abs MAE 0.054, beta_pres MAE 0.039, both with 94% coverage 1.000.
- Edge beta recovery: beta_abs MAE 0.062, beta_pres MAE 0.051, both with 94% coverage 1.000.
- Path summary recovery: `delta_j` MAE 0.054, RMSE 0.062, 94% coverage 0.920; `q_j(0.999)-q_j(0.001)` has the same recovery profile.
- Internal state probability scoring: mean relative entropy reduction 0.333; mean Brier 0.111; mean negative log score 0.343.
- Observed-scale feature-block PPC: mean absolute proportion error 0.116; 94% posterior predictive coverage 0.990 after including ordinal rating noise.
- Practical reading: the full M-closed fit behaves computationally well and covers the key truths in this seed, but root C recovery around Chicken/LLM remains broad. This supports treating the current results as probabilistic/weakly identified in the middle range rather than as precise root separation.
- Output directory: `runs/full_exact_recovery/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506/`
