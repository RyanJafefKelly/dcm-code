"""Reduced-sample demo: ordinal DCM on real GWT data from local cache.

Runs the full ordinal model on 'Global Workspace Theory' for a selected
system with a low sampling budget.  Verifies the model compiles and
produces indicator _pz1 outputs with the real tree structure.

No network access required -- uses data_cache.json.
"""
from __future__ import annotations

from dcm_model import (
    ModelConfig,
    OrdinalDataProcessor,
    EvidenceProcessor,
    BayesianModelBuilder,
    ResultsManager,
    load_data,
    setup_logging,
)


def main() -> None:
    logger = setup_logging("INFO")

    config = ModelConfig(
        NUM_SAMPLES=200,
        NUM_TUNE=200,
        NUM_CHAINS=2,
        TARGET_ACCEPT=0.9,
    )

    all_data = load_data(config)
    stance_data = next(
        item for item in all_data if item["name"] == config.TARGET_STANCE
    )

    processor = OrdinalDataProcessor(config)
    processor.process(stance_data, config.TARGET_SYSTEM)

    evidence_proc = EvidenceProcessor(config)
    builder = BayesianModelBuilder(config, evidence_proc, processor)
    model = builder.build_model(stance_data)

    print(
        f"\nModel: {len(model.free_RVs)} free RVs, "
        f"{len(model.potentials)} potentials, "
        f"{len(model.deterministics)} deterministics"
    )

    idata = builder.sample(model)

    results = ResultsManager(config, builder.node_to_varname)
    print(f"\n{'=' * 60}")
    print(f"GWT Ordinal Demo: {config.TARGET_SYSTEM}")
    print(f"{'=' * 60}\n")
    results.summarise(idata, stance_data)

    # Observation model parameters
    print(f"\n{'=' * 60}")
    print("Observation model parameters")
    print(f"{'=' * 60}")
    print(f"  a (discrimination): {float(idata.posterior['a'].mean()):.3f}")
    n_experts = len(processor.expert_names)
    if n_experts > 1:
        b_free = idata.posterior["b_free"].mean(dim=("chain", "draw")).values
        print(
            f"  b (expert shifts):  "
            f"[0.000 (anchor: {processor.anchor_expert}), "
            + ", ".join(f"{v:.3f}" for v in b_free)
            + "]"
        )
    kappa_vals = idata.posterior["kappa"].mean(dim=("chain", "draw")).values
    print(
        f"  kappa (cutpoints):  "
        f"[{', '.join(f'{v:.3f}' for v in kappa_vals)}]"
    )

    logger.info("Demo complete")


if __name__ == "__main__":
    main()
