# Codex Audit Report: Targeted Strong-Lower Override

No HMC was launched. This audit used source reads plus one static no-sampling key-selection check.

## Summary

The targeted exact-tree path is implemented and reusable for the main synthetic validation workflow. `ModelConfig` exposes `TARGETED_OVERRIDE_NODE_KEYS`; `MultiSystemExactTreeBuilder._collect_tree_betas` dispatches targeted keys before pooled, safe-gain, or production-Beta branches; targeted edges use the existing node-level logit-Normal helper (`build_safe_gain_node_beta`), not hard-coded `Beta(18,2)` / `Beta(2,18)`.

For the intended `targeted_strong_lower_override` main-validation path, non-overridden edges stay on production `EvidenceProcessor.get_beta_parameters(...)` per-node `pm.Beta` priors because `scripts/main_synthetic_validation.py::build_fit_config` sets `POOL_BETAS_BY_LABEL=False`. If someone combines `TARGETED_OVERRIDE_NODE_KEYS` with `POOL_BETAS_BY_LABEL=True`, targeted edges still win, but non-targeted edges follow the pooled branch rather than per-node production Betas.

Static targeted-key check on current GWT:

- Total non-root edges: `75`
- Targeted overridden edges: `49`
- Root-to-top overridden edges: `0`
- Weak-top overridden edges: `0`
- Targeted by top feature: Coherence `17`, Selective Attention `17`, Complexity `13`, Integration `2`
- Targeted by depth/type: depth 2 feature edges `14`, depth 3 indicator edges `35`
- Generator/fitter beta-centre mismatches: `0` for pres and abs; targeted truth/prior centres are `0.90 / 0.10`

## Checklist

| # | Audit item | Status | Evidence |
|---|---|---|---|
| 1 | `ModelConfig` has `TARGETED_OVERRIDE_NODE_KEYS` | Pass | `dcm_model.py::ModelConfig` includes `TARGETED_OVERRIDE_NODE_KEYS: Optional[Sequence[str]] = None`. |
| 2 | `_collect_tree_betas` dispatches targeted keys before pooled/Beta branches | Pass | `dcm_model_exact_tree.py::MultiSystemExactTreeBuilder._collect_tree_betas` checks `targeted_keys is not None and key in targeted_keys` before `POOL_BETAS_BY_LABEL`, `GAIN_LOGIT_NORMAL`, and `pm.Beta`. |
| 3 | Overridden edges use logit-Normal override infrastructure, not `Beta(18,2)` / `Beta(2,18)` | Pass | Targeted branch calls `build_safe_gain_node_beta`; that helper creates `*_lnbeta_tilde ~ Normal(0,1)` and deterministic sigmoid-logit beta. |
| 4 | Non-overridden edges remain on production `EvidenceProcessor` Beta priors | Pass | In the main targeted fit config, pooling is disabled, so non-targeted exact-tree edges call `EvidenceProcessor.get_beta_parameters(...)` then `pm.Beta(...)`. |
| 5 | Function computing targeted lower-edge keys identified | Pass | `scripts/main_synthetic_validation_metrics.py::collect_targeted_override_node_keys`. |
| 6 | Edge selection is correct | Pass | Static check: `49` overridden, `0` root-to-top, all under Coherence / Selective Attention / Complexity / Integration, none under Representationality / Hierarchical Organization / Modularity. |
| 7 | Generator truth can be overridden edge-by-edge to `0.90 / 0.10` for same keys | Pass | `build_edge_beta_profile` sets truth and fitter means for targeted keys; `profile_as_truth_edge_betas` feeds generation; `generate_synthetic_dataset` writes the same keys to truth payload. |
| 8 | Generator/fitter beta-centre matching is asserted | Pass | `validate_edge_beta_profile` asserts max pres/abs difference <= tolerance; `run_one_fit` calls it before sampling. Static check returned both max diffs `0.0`. |
| 9 | `rho_collapsed` exists or can be computed post hoc | Pass | Exact-tree builder writes `{system}__global_workspace_theory_rho_collapsed`; `extract_rho_cases` falls back to `expit(LOGIT_PRIOR + log_B)` if missing. Existing run summary reports `rho_collapsed_present: true`. |
| 10 | Internal-node identifiability code identified for reuse | Pass | `gwt_oracle_internal_identifiability.py` is no-HMC and provides exact clamped DP via `exact_log_evidence_with_optional_clamp`, `posterior_internal_probability`, `simulate_and_score_once`, and summaries. |
| 11 | Attrition decomposition outputs identified for reuse | Pass | `phase1c_evidence_attrition_audit.py` writes `phase1c_evidence_attrition_cases.csv`, `phase1c_evidence_attrition_summary.csv`, `phase1c_feature_message_diagnostics.csv`, `phase1c_edge_profile_ablation.csv`, `phase1c_information_ramp.csv`, `phase1c_leaf_information_map.csv`, and report. |
| 12 | PPC functions identified | Pass | `exact_tree_ppc.py` has `forward_pass`, `backward_pass`, `per_indicator_exact_tree_marginal`, `focus_cell_ppc_exact_tree`; `posterior_predictive_rating_dist.py` has `predicted_distribution`; main validation already wraps exact-tree PPC in `compute_exact_tree_ppc`. |

## Reuse Notes

Phase 1B can be reused for top-bound/no-fit ladder checks. `scripts/phase1b_root_evidence_ladder.py` already exposes top-bound, no-fit, and fit-smoke modes and intentionally does not run HMC in fit-smoke mode.

Phase 1C can be reused directly for the attrition story. Its `strong_top_features_only_extreme_lower` edge profile is the no-fit analogue of the targeted override, and its report already states that strong-top lower-edge strengthening passes while weak-top lower-edge strengthening fails.

The May 6 synthetic scripts are reusable, but `scripts/main_synthetic_validation.py` is now the cleaner targeted harness: it already generates targeted truth, validates matched centres, fits exact tree, extracts `rho_collapsed`, computes exact-tree PPC, and computes conditional LOO.

## Exact Files / Functions To Patch Next

No patch is required to make the targeted exact-tree infrastructure itself work. Useful next patches for consolidation:

1. `notebooks/synthetic_validation_2026-05-06/gwt_oracle_internal_identifiability.py::load_oracle_truth` or a nearby helper: allow targeted `edge_betas` from `scripts.main_synthetic_validation_metrics::build_edge_beta_profile` so internal identifiability can run under the exact targeted truth without duplicating code.
2. `scripts/phase1c_evidence_attrition_audit.py::build_beta_profiles` / `edge_profile_betas`: add a canonical `targeted_strong_lower_override` profile label that delegates to `build_edge_beta_profile`, replacing the parallel `strong_top_features_only_extreme_lower` naming for main-validation reports.
3. `notebooks/synthetic_validation_2026-05-06/gwt_full_exact_recovery.py::build_fit_config` and `generate_synthetic_dataset`: only patch if this older script must run targeted fits; otherwise prefer `scripts/main_synthetic_validation.py`.
4. `notebooks/asymmetric_prior_sweep_2026-05-10/posterior_predictive_rating_dist.py::predicted_distribution`: patch only if reusing this plotter for targeted unpooled node-level fits, because it currently routes through pooled-beta extraction; `scripts.main_synthetic_validation_metrics::compute_exact_tree_ppc` is already targeted-compatible.
5. `dcm_model.py::BayesianModelBuilder._add_evidencer_node` and `dcm_model.py::MultiSystemModelBuilder._add_tree_node`: patch only if non-exact/composite builders also need targeted overrides. The current targeted main-validation path uses `MultiSystemExactTreeBuilder`, where the override is already wired.
