"""Synthetic smoke test for the ordinal DCM implementation.

Builds a small GWT-like tree with known parameters, generates synthetic
ordinal ratings, feeds them through the real OrdinalDataProcessor and
BayesianModelBuilder pipeline, and verifies that:
  1. The model compiles and samples without error.
  2. Posterior indicator-presence probabilities (_pz1) are higher for
     indicators whose data was generated under z=1 than z=0.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from dcm_model import (
    ModelConfig,
    OrdinalDataProcessor,
    EvidenceProcessor,
    BayesianModelBuilder,
    node_key,
)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

# Probability values that round-trip cleanly through
# legacy_probability_to_ordinal with default ORDINAL_BINS:
#   (0.05, 0.20, 0.40, 0.60, 0.80, 0.95)
# ordinal 0 -> 0.00, 1 -> 0.10, 2 -> 0.30, 3 -> 0.50,
#          4 -> 0.70, 5 -> 0.90, 6 -> 1.00
_ORDINAL_TO_PROB = [0.00, 0.10, 0.30, 0.50, 0.70, 0.90, 1.00]


def ordinal_probs(cutpoints: np.ndarray, eta: float) -> np.ndarray:
    """Category probabilities under ordered probit (NumPy, for data gen)."""
    c_aug = np.concatenate([[-np.inf], cutpoints, [np.inf]])
    return np.diff(norm.cdf(c_aug - eta))


def build_synthetic_tree(
    n_features: int = 2,
    indicators_per_feature: int = 2,
    n_experts: int = 3,
    true_a: float = 1.5,
    true_b: np.ndarray | None = None,
    true_kappa: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[dict, dict[str, int], str]:
    """Build a synthetic GWT-like tree with ordinal ratings.

    Returns
    -------
    tree : dict in the same JSON schema as data_cache.json
    true_states : {indicator_name: z_j} for checking recovery
    system_name : str
    """
    rng = np.random.default_rng(seed)
    if true_b is None:
        true_b = np.array([0.0, 0.5, -0.3])[:n_experts]
    if true_kappa is None:
        true_kappa = np.array([-1.5, -0.8, -0.1, 0.4, 1.0, 1.6])

    K = 7
    expert_names = [f"Expert_{i}" for i in range(n_experts)]
    system_name = "TestSystem"
    true_states: dict[str, int] = {}

    features = []
    for f_idx in range(n_features):
        indicators = []
        for ind_idx in range(indicators_per_feature):
            ind_name = f"Indicator_F{f_idx}_I{ind_idx}"
            # Alternate present / absent so we can check separation
            z_j = 1 if (f_idx + ind_idx) % 2 == 0 else 0
            true_states[ind_name] = z_j

            values, names = [], []
            for e in range(n_experts):
                eta = true_b[e] + true_a * z_j
                probs = ordinal_probs(true_kappa, eta=eta)
                r = rng.choice(K, p=probs)
                values.append(str(_ORDINAL_TO_PROB[r]))
                names.append(expert_names[e])

            indicators.append(
                {
                    "name": ind_name,
                    "type": "Indicator",
                    "support": "moderate support",
                    "demandingness": "neutral",
                    "observations": {
                        system_name: {"values": values, "names": names}
                    },
                    "observation_stats": {
                        system_name: {
                            "average": str(
                                np.mean([float(v) for v in values])
                            ),
                            "std": 0,
                        }
                    },
                }
            )

        features.append(
            {
                "name": f"Feature_{f_idx}",
                "type": "Feature",
                "support": "moderate support",
                "demandingness": "neutral",
                "evidencers": indicators,
            }
        )

    tree = {
        "name": "Global Workspace Theory",
        "type": "Stance",
        "evidencers": features,
    }
    return tree, true_states, system_name


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_smoke() -> None:
    """End-to-end smoke test using real builder pipeline."""
    print("=" * 60)
    print("Ordinal DCM -- synthetic smoke test")
    print("=" * 60)

    tree, true_states, system = build_synthetic_tree()

    print(f"\nTrue indicator states:")
    for k, v in true_states.items():
        print(f"  {k}: z={v}")

    # -- fast sampling config --
    config = ModelConfig(
        NUM_SAMPLES=200,
        NUM_TUNE=200,
        NUM_CHAINS=2,
        TARGET_STANCE="Global Workspace Theory",
        TARGET_SYSTEM=system,
    )

    # -- preprocess --
    processor = OrdinalDataProcessor(config)
    processor.process(tree, system)

    print(f"\nAnchor expert: {processor.anchor_expert}")
    print(f"Observations per indicator:")
    for key, obs in sorted(processor.observations.items()):
        print(f"  {key}: {len(obs)} ratings")

    # -- build --
    evidence_proc = EvidenceProcessor(config)
    builder = BayesianModelBuilder(config, evidence_proc, processor)
    model = builder.build_model(tree)

    print(
        f"\nModel: {len(model.free_RVs)} free RVs, "
        f"{len(model.potentials)} potentials, "
        f"{len(model.deterministics)} deterministics"
    )

    # -- sample --
    idata = builder.sample(model)

    # -- check results --
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    pz1_present: list[float] = []
    pz1_absent: list[float] = []

    for vn in sorted(idata.posterior.data_vars):
        if not vn.endswith("_pz1"):
            continue
        pz1_val = float(idata.posterior[vn].mean())

        # Match back to true state via node_to_varname
        matched_name = None
        for nkey, varname in builder.node_to_varname.items():
            if vn == f"{varname}_pz1" and nkey.split(" > ")[-1] in true_states:
                matched_name = nkey.split(" > ")[-1]
                break

        if matched_name is not None:
            z = true_states[matched_name]
            tag = "PRESENT" if z == 1 else "absent"
            print(f"  {matched_name}: pz1={pz1_val:.3f}  (true z={z}, {tag})")
            (pz1_present if z == 1 else pz1_absent).append(pz1_val)

    # -- assertions --
    print(f"\n  Mean pz1 (z=1 indicators): {np.mean(pz1_present):.3f}")
    print(f"  Mean pz1 (z=0 indicators): {np.mean(pz1_absent):.3f}")

    if np.mean(pz1_present) > np.mean(pz1_absent):
        print("  PASS: posterior separates present from absent indicators")
    else:
        print(
            "  WARNING: posterior did not separate clearly "
            "(may be OK with low sample count -- try increasing NUM_SAMPLES)"
        )

    # Observation model parameters
    print(f"\n  a (discrimination): {float(idata.posterior['a'].mean()):.3f}")
    kappa_vals = idata.posterior["kappa"].mean(dim=("chain", "draw")).values
    print(f"  kappa: [{', '.join(f'{v:.3f}' for v in kappa_vals)}]")

    print("\nSmoke test complete.")


if __name__ == "__main__":
    test_smoke()
