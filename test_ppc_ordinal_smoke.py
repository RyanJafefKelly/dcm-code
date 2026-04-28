"""Smoke test for joint multi-system PPC helpers and observation configs.

Builds a tiny two-system GWT-like tree, runs a very short joint fit in both
"no shifts", "tight shifts (sigma=0.3)", and hierarchical expert-cutpoint
modes, and exercises both the default pz1-based PPC and the q_j sidecar.
Checks structural invariants only (shapes, dict keys, non-negative histograms
summing to 1).
"""

from __future__ import annotations

import numpy as np

from dcm_model_ordinal import (
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    MultiSystemModelBuilder,
)
from dcm_ppc_ordinal import (
    per_expert_ppc_multisystem,
    per_expert_system_tree_implied_ppc_multisystem,
    per_expert_tree_implied_ppc_multisystem,
    summarise_per_expert_ppc,
    summarise_per_expert_system_ppc,
)
from test_ordinal_smoke import build_synthetic_tree


def _merge_systems(tree_a: dict, tree_b: dict, system_a: str, system_b: str) -> dict:
    """Merge two single-system synthetic trees into a single tree that carries
    observations for both systems on the same indicators."""
    import copy

    merged = copy.deepcopy(tree_a)

    def walk(node_a, node_b):
        if node_a.get("type", "").lower() == "indicator":
            obs_b = node_b.get("observations", {}).get(system_b)
            if obs_b is not None:
                node_a.setdefault("observations", {})[system_b] = obs_b
        for child_a, child_b in zip(
            node_a.get("evidencers", []), node_b.get("evidencers", [])
        ):
            walk(child_a, child_b)

    walk(merged, tree_b)
    return merged


def run_one(cfg: ModelConfig, label: str) -> None:
    print(f"\n=== {label} ===")

    # Two synthetic "systems" with different separability
    tree_a, _, sys_a = build_synthetic_tree(
        n_features=2, indicators_per_feature=2, n_experts=3, true_a=1.8, seed=1
    )
    tree_b, _, _ = build_synthetic_tree(
        n_features=2, indicators_per_feature=2, n_experts=3, true_a=1.8, seed=7
    )
    sys_b = "TestSystemB"
    # Relabel sys_b observations
    def relabel(node):
        if node.get("type", "").lower() == "indicator":
            if sys_a in node.get("observations", {}):
                node["observations"][sys_b] = node["observations"].pop(sys_a)
        for c in node.get("evidencers", []):
            relabel(c)

    relabel(tree_b)
    merged = _merge_systems(tree_a, tree_b, sys_a, sys_b)

    system_configs = [(sys_a, None), (sys_b, None)]

    processor = MultiSystemDataProcessor(cfg)
    processor.process(merged, [sys_a, sys_b])

    evidence_proc = EvidenceProcessor(cfg)
    builder = MultiSystemModelBuilder(cfg, evidence_proc, processor, system_configs)
    model = builder.build_model(merged)
    idata = builder.sample(model)

    if cfg.USE_HIERARCHICAL_EXPERT_CUTPOINTS:
        assert "b_free" not in idata.posterior.data_vars
        assert "kappa_by_expert" in idata.posterior.data_vars
        kappa_means = idata.posterior["kappa_by_expert"].mean(
            dim=("chain", "draw")
        ).values.mean(axis=1)
        assert np.allclose(kappa_means, 0.0, atol=1e-6)
        print("hierarchical expert cutpoints enabled; centered per expert")
    elif cfg.USE_EXPERT_SHIFTS:
        assert "b_free" in idata.posterior.data_vars, "expected b_free with shifts"
        b_sd = float(idata.posterior["b_free"].std())
        print(f"b_free posterior sd ~ {b_sd:.3f} (prior sigma={cfg.EXPERT_SHIFT_SIGMA})")
    else:
        assert "b_free" not in idata.posterior.data_vars
        print("no b_free variable (shifts disabled)")

    ppc = per_expert_ppc_multisystem(
        idata, builder, processor, n_draws=80, seed=0
    )
    assert ppc, "PPC returned empty dict"
    print(summarise_per_expert_ppc(ppc))

    ppc_tree = per_expert_tree_implied_ppc_multisystem(
        idata, builder, processor, n_draws=80, seed=0
    )
    assert ppc_tree, "tree-implied PPC returned empty dict"

    ppc_tree_strat = per_expert_system_tree_implied_ppc_multisystem(
        idata, builder, processor, n_draws=80, seed=0
    )
    assert ppc_tree_strat, "stratified tree-implied PPC returned empty dict"
    print(summarise_per_expert_system_ppc(ppc_tree_strat))

    K = cfg.N_CATEGORIES
    for result_dict in (ppc, ppc_tree):
        for name, r in result_dict.items():
            assert r["obs_hist"].shape == (K,)
            assert r["pred_hist_mean"].shape == (K,)
            assert np.isclose(r["obs_hist"].sum(), 1.0)
            assert np.isclose(r["pred_hist_mean"].sum(), 1.0, atol=1e-6)
            assert (r["pred_hist_mean"] >= 0).all()
            assert 0 <= r["pred_mean_mean"] <= K - 1
            assert 0 <= r["pred_extreme_mean"] <= 1
    for key, r in ppc_tree_strat.items():
        assert r["obs_hist"].shape == (K,)
        assert r["pred_hist_mean"].shape == (K,)
        assert np.isclose(r["obs_hist"].sum(), 1.0)
        assert np.isclose(r["pred_hist_mean"].sum(), 1.0, atol=1e-6)
        assert (r["pred_hist_mean"] >= 0).all()
    print(f"{label}: PPC structural checks passed")


def main() -> None:
    base_cfg = dict(
        NUM_SAMPLES=150,
        NUM_TUNE=250,
        NUM_CHAINS=2,
        TARGET_ACCEPT=0.9,
    )
    run_one(
        ModelConfig(USE_EXPERT_SHIFTS=False, **base_cfg),
        "no-shifts baseline",
    )
    run_one(
        ModelConfig(USE_EXPERT_SHIFTS=True, EXPERT_SHIFT_SIGMA=0.3, **base_cfg),
        "tight shifts (sigma=0.3)",
    )
    run_one(
        ModelConfig(
            USE_EXPERT_SHIFTS=False,
            USE_HIERARCHICAL_EXPERT_CUTPOINTS=True,
            HIER_KAPPA_EXPERT_SCALE_SIGMA=0.15,
            **base_cfg,
        ),
        "hierarchical expert cutpoints",
    )


if __name__ == "__main__":
    main()
