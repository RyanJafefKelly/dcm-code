# Per-expert discrimination $d_e$ — analytical Fisher feasibility (2026-05-10)

## Headline

**The per-expert $d_e$ does *not* structurally collapse onto per-system
discrimination at the diagonal Fisher metric.** Under oracle nuisance
(production-median $a$, $\kappa$, true $\alpha_j$, $\delta_j$, $C_s$
from synthetic seed-06), every expert in the realistic GWT design
contributes substantial expected Fisher info to their own $d_e$ at
$d_e = 1$ — single-system experts get $\sim 90\text{–}115$ nats$^2$,
well above the $1$ nat$^2$ collapse threshold the prompt set, and the
cross-system expert (Rater_B) gets $\sim 396$ nats$^2$ entirely
from his $\sim 3\times$ greater coverage rather than from any
qualitative cross-system bonus. Per-rating Fisher is broadly flat
across experts (2.3 nats$^2$ on LLM/Chicken raters, 2.6 nats$^2$ for
Rater_B) — the realistic design difference is one of rating volume, not
structural identifiability of the $d_e$ diagonal. **Caveat:** this
diagonal-Fisher analysis pins all upstream nuisance (especially the
global $a$) at truth; the joint identifiability of $\{d_e\}$ together
with $a$ is a separate question (single-system experts share a
$a \cdot d_e$ confounding direction with $a$, identifiable only via
within-system contrasts among co-raters — LLM has 4, Chicken 2, Human
and ELIZA 1). The diagonal result tells us $d_e$ is *locally
informative*, not that it's posteriorly resolvable in the full model.

## Per-expert Fisher info on $d_e$ at $d_e = 1$ (nats$^2$)

| expert            | systems rated | n ratings | realistic | idealised | per-rating (real.) |
|-------------------|--------------:|----------:|----------:|----------:|-------------------:|
| Rater_C      |             1 |        48 |     112.8 |     520.1 |               2.35 |
| Rater_A  |             1 |        40 |      93.5 |     520.1 |               2.34 |
| Rater_D   |             1 |        48 |     111.9 |     520.1 |               2.33 |
| Rater_E    |             1 |        46 |     112.8 |     520.1 |               2.45 |
| Hayley chickens   |             1 |        47 |     114.6 |     520.1 |               2.44 |
| **Rater_B** |         **3** |   **150** | **396.2** | **520.1** |           **2.64** |

Realistic = current GWT seed-06 design (data_cache.json `Activation
Steering Effects` mapping generalised across all 50 indicators);
idealised = every expert × every system × every indicator = 200
ratings each. Per-rating Fisher across systems: LLM 2.35, Chicken
2.45, ELIZA 2.29, Human 3.27 — slightly higher at extreme C because
$\tilde q$ saturates and the latent rating is more concentrated, so
small $d_e$ shifts produce larger relative log-likelihood swings.

## What this does and doesn't say

- **Does say:** at the production-median operating point, each
  expert's ratings carry $\mathcal{O}(10^2)$ nats$^2$ of marginal
  information about their own discrimination scale. Adding $d_e$ to
  the ordered probit is not a doomed extension on this design.
- **Doesn't say:** that the joint posterior contracts on each $d_e$
  individually. The model must impose an identification constraint
  (e.g. $\prod_e d_e = 1$ or fix one expert) to remove the global
  $a \cdot \langle d_e \rangle$ degeneracy. Within-system relative
  discrimination is constrained by between-rater contrasts — LLM
  (4 raters) and Chicken (2 raters) provide such contrasts; Human and
  ELIZA do not (Rater_B is the sole rater) and so Rater_B's $d_e$ is only
  pinned down by cross-system pooling against the LLM raters (where
  he overlaps with Rater_C/Rater_A/Rater_D).
- **Idealised vs realistic gap is volume, not structure:** every
  realistic expert moves from $\sim 100$ to $520$ nats$^2$ purely by
  going from $\sim 45$ to $200$ ratings. No expert in the realistic
  design hits a structural ceiling below the idealised one.

## Recommended follow-up if compute permits

The natural next step (Arvo's actual concern) is the *joint* Fisher
matrix for $\{d_e\}_e \cup \{a\}$ with the chosen identification
constraint, then check the conditional number / posterior contraction
on each $d_e$ from a synthetic fit. The per-rating diagonal numbers
above suggest there's enough information to make this worth doing;
the diagonal alone doesn't predict whether off-diagonal correlations
(especially Rater_B-vs-$a$ via the Human and ELIZA cells where he is
the sole rater) will dominate.

## Reproduction

```
python notebooks/expert_discrimination_feasibility_2026-05-10/fisher_de.py
```

Reads `notebooks/synthetic_validation_2026-05-06/runs/full_exact_recovery/.../truth.json`
(seed 20260506 production-median $a$, $\kappa$, true $C$) and
`data_cache.json` (rater names per indicator cell). Writes
`fisher_de_{realistic,idealised}.csv` and the
`*_per_system.csv` long-format breakdowns.
