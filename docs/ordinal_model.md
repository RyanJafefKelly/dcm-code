# Ordinal GWT Model

This branch adds a separate ordinal implementation in `dcm_model_ordinal.py`.
It leaves the original `dcm_model.py` unchanged.

The model uses the paper tree as a marginal probability propagation device:
each node has paper-style `beta_pres` and `beta_abs` parameters, each system has
its own root `C`, and indicator leaves are conditionally independent given the
propagated `q_j` values and shared observation parameters.

Included leaf models:

- `binary`: latent indicator state in `{0, 1}`.
- `three_state`: latent indicator state in `{0, 0.5, 1}` with a Binomial(2, q)
  prior and ordered-probit emission centers `{0, a / 2, a}`.

Included reference-anchor configurations:

- `hard_anchor_3s_paper_tree`: Human fixed at `C = 0.999`, ELIZA fixed at
  `C = 0.001`.
- `soft_anchor_3s_paper_tree`: Human has `Beta(50, 1)`, ELIZA has
  `Beta(1, 50)`.

Run them explicitly:

```bash
python run_gwt_ordinal_baseline.py --config hard_anchor_3s_paper_tree
python run_gwt_ordinal_baseline.py --config soft_anchor_3s_paper_tree
```

There is deliberately no default between hard and soft anchors in this branch.

Sensitivity switches are available through `ModelConfig`, but are non-default:

- `USE_EXPERT_SHIFTS`
- `USE_HIERARCHICAL_EXPERT_CUTPOINTS`
- `USE_EXPERT_SCALES`

Out of scope for this branch:

- tree pooling
- transmission gain
- exact latent-tree fitting
- generated notebooks, figures, CSVs, and result files

Public outputs use anonymized expert labels such as `Expert A` and role labels
such as `E_cross`. Raw expert identifiers from `data_cache.json` are not written
to production metadata.
