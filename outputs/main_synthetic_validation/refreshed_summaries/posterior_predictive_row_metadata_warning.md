# Posterior Predictive Row Metadata Warning

Existing smoke `posterior_predictive.nc` files do not contain system/node/rater row metadata, and no per-run `posterior_predictive_row_metadata.csv` sidecars exist for these completed runs. Cell-TV cannot be reconstructed from NetCDF alone for the existing smoke outputs. Future runs now write the sidecar from the exact row table used to create `posterior_predictive.nc`.

Missing sidecars:
- `runs/targeted_strong_lower_override/seed_20260511`
- `runs/targeted_strong_lower_override/seed_20260512`
