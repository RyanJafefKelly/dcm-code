# Ordinal-layer explainer figures

Two schematic figures for the SPAR report (and eventual paper) that
communicate where the ordinal observation layer lives in the DCM and
what it actually does, plus three "bonus" PGM variants showing the
expert-specific extensions, plus an optional posterior-mean version
of the cutpoint diagram.

## Files

- `_style.py` — shared palette + matplotlib rcParams.
- `ordinal_explainer_pgm.py` — Figure A. Extends the paper's
  Pr(conscious) → stance → feature → indicator tree with the new
  bottom layer: q_j → m_j → s_jk → r_jk. `--variant` selects which
  shared-parameter set to display.
- `ordinal_explainer_cutpoints.py` — Figure B. Latent-signal axis
  with three Gaussian bumps, six cutpoints, seven shaded bands, and
  a stacked-bar panel of P(r | m_j). `--from-fit` swaps schematic
  values for posterior means.
- `build.sh` — fast rebuild loop.

## Outputs

All in `report_figures/ordinal_explainer/`:

| file | what it shows |
|------|----|
| `pgm.png` | **Baseline PGM** — shared `{a, κ}`, ε ~ N(0,1) |
| `pgm_expert_shifts.png` | + per-expert location shift `b_e` |
| `pgm_expert_cutpoints.png` | + per-expert cutpoints `κ_e` (hierarchical) |
| `pgm_expert_scales.png` | + per-expert noise scale `σ_e` |
| `cutpoints.png` | **Schematic** cutpoint diagram (illustrative `a, κ`) |
| `cutpoints_posterior.png` | Same diagram with posterior means from a fit |

## Commands

From `dcm-code-clean/`:

```bash
# Defaults
bash figures/build.sh                      # baseline PGM + schematic cutpoints
bash figures/build.sh open                 # same, then open the two PNGs

# Single targets
bash figures/build.sh pgm                  # baseline PGM only
bash figures/build.sh cut                  # schematic cutpoints only

# All four PGM variants
bash figures/build.sh variants

# Direct python invocation (more control)
python figures/ordinal_explainer_pgm.py --variant baseline
python figures/ordinal_explainer_pgm.py --variant expert_shifts
python figures/ordinal_explainer_pgm.py --variant expert_cutpoints
python figures/ordinal_explainer_pgm.py --variant expert_scales
python figures/ordinal_explainer_pgm.py --all      # render every variant

# Posterior-mean cutpoint version
python figures/ordinal_explainer_cutpoints.py \
    --from-fit results/gwt_ordinal/hard_anchor_3s_paper_tree_20260428T065809Z/hard_anchor_3s_paper_tree.nc
```

If the local `python` is not the project venv, set `PY` explicitly:

```bash
PY=../dcm-code/.venv/bin/python bash figures/build.sh
```

## Iterate loop

Coordinate-driven plain matplotlib. Every node position is one
number to nudge in the source. Tight feedback loop: edit → run
`bash figures/build.sh open` → look → repeat.

## Schematic parameters

Cutpoint diagram defaults at the top of
`ordinal_explainer_cutpoints.py`:

```python
SCHEMATIC_A     = 1.6
SCHEMATIC_SIGMA = 1.0
SCHEMATIC_KAPPA = [-1.6, -0.8, -0.25, 0.25, 0.8, 1.6]
```

Chosen so the three Gaussian bumps and the six cutpoints are
both visually legible. The actual posterior fit has `a ≈ 4.38`
and `κ ≈ [1.13, 1.50, 1.79, 2.24, 2.88, 3.67]` — strong learned
discrimination, but pushes all bumps right of the lower cutpoints
and is harder to read as a *mechanism* diagram. Use schematic in
the main text; posterior in an appendix or sensitivity caption.

## Design notes

- Top half (system → indicator) intentionally mirrors paper Figure 1
  in colour and style for instant continuity.
- Bottom half is drawn for ONE indicator to keep the figure legible;
  dashed plates make the "for each indicator j" / "for each expert k"
  replication explicit.
- Three nodes (`m_j`, `s_jk`, `r_jk`) carry distinct accent colours
  so a reader's eye lands on the new layer.
- `r_jk` uses the standard PGM observed-node convention (diagonal
  hatching + thicker edge) on top of its accent colour.
- Variants reserve a yellow accent for *per-expert* parameters
  drawn inside the expert plate, distinguishing them from shared
  greys outside.
