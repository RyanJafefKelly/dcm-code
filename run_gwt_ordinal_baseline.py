"""Run explicit GWT ordinal baseline configurations.

This runner intentionally has no implicit default baseline. PR 1 exposes the
hard-anchor and soft-anchor paper-tree configurations so they can be rerun under
clean code with the same data snapshot, seeds, and sampling budget.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dcm_model_ordinal import (
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    MultiSystemModelBuilder,
    fetch_data,
    load_data,
)
from dcm_ppc_ordinal import per_expert_system_ppc_multisystem


TARGET_STANCE = "Global Workspace Theory"
SYSTEM_HUMAN = "Human"
SYSTEM_CHICKEN = "Chicken"
SYSTEM_LLM = "2024 Leading Chat LLMs"
SYSTEM_ELIZA = "ELIZA"
SYSTEMS = [SYSTEM_HUMAN, SYSTEM_CHICKEN, SYSTEM_LLM, SYSTEM_ELIZA]

CONFIG_CHOICES = {
    "hard_anchor_3s_paper_tree",
    "soft_anchor_3s_paper_tree",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        choices=sorted(CONFIG_CHOICES),
        help="Explicit baseline configuration to run. There is no default.",
    )
    parser.add_argument("--data-cache", default="data_cache.json")
    parser.add_argument("--output-dir", default="results/gwt_ordinal")
    parser.add_argument("--draws", type=int, default=3000)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ppc-seed", type=int, default=43)
    parser.add_argument("--ppc-draws", type=int, default=500)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Tuple[ModelConfig, List[Tuple[str, Optional[float]]]]:
    cfg = ModelConfig(
        NUM_SAMPLES=args.draws,
        NUM_TUNE=args.tune,
        NUM_CHAINS=args.chains,
        TARGET_ACCEPT=args.target_accept,
        RANDOM_SEED=args.seed,
        TARGET_STANCE=TARGET_STANCE,
        DATA_CACHE_PATH=args.data_cache,
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=True,
    )
    if args.config == "hard_anchor_3s_paper_tree":
        system_configs = [
            (SYSTEM_HUMAN, 0.999),
            (SYSTEM_CHICKEN, None),
            (SYSTEM_LLM, None),
            (SYSTEM_ELIZA, 0.001),
        ]
    elif args.config == "soft_anchor_3s_paper_tree":
        cfg.SOFT_REFERENCE_ANCHORS = {
            SYSTEM_HUMAN: (50.0, 1.0),
            SYSTEM_ELIZA: (1.0, 50.0),
        }
        system_configs = [(system, None) for system in SYSTEMS]
    else:
        raise ValueError(f"Unhandled config: {args.config}")
    return cfg, system_configs


def canonical_data_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_or_fetch_data(config: ModelConfig) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    path = Path(config.DATA_CACHE_PATH)
    if path.exists():
        return load_data(config), {"kind": "cache", "path": str(path)}
    data = fetch_data(config)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return data, {"kind": "fetched", "path": str(path), "endpoint": config.API_ENDPOINT}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def package_versions() -> Dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ["numpy", "pymc", "pytensor", "scipy", "arviz"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def system_prefix(system: str) -> str:
    return MultiSystemModelBuilder._sys_prefix(system)


def diagnostic_var_names(config_name: str) -> List[str]:
    stance = "global_workspace_theory"
    systems = [SYSTEM_CHICKEN, SYSTEM_LLM]
    if config_name.startswith("soft_anchor"):
        systems.extend([SYSTEM_HUMAN, SYSTEM_ELIZA])
    names = [f"{system_prefix(system)}__{stance}_C" for system in systems]
    names.extend(["a", "kappa"])
    names.extend(["b_free", "kappa_by_expert", "sigma_by_expert", "tau_sigma"])
    return names


def diagnostics_summary(idata: Any, config_name: str) -> Dict[str, Any]:
    try:
        import arviz as az
    except Exception as exc:
        return {"available": False, "reason": f"arviz unavailable: {exc}"}

    available = set(idata.posterior.data_vars)
    var_names = [name for name in diagnostic_var_names(config_name) if name in available]
    if not var_names:
        return {"available": False, "reason": "no declared diagnostic variables found"}
    try:
        summary = az.summary(idata, var_names=var_names, kind="diagnostics")
    except Exception as exc:
        return {"available": False, "reason": f"diagnostics failed: {exc}"}
    return {
        "available": True,
        "variables": var_names,
        "max_r_hat": float(summary["r_hat"].max(skipna=True)),
        "min_ess_bulk": float(summary["ess_bulk"].min(skipna=True)),
        "min_ess_tail": float(summary["ess_tail"].min(skipna=True)),
    }


def role_labels(processor: MultiSystemDataProcessor) -> Dict[str, str]:
    counts: Dict[Tuple[int, str], int] = {}
    for system, sys_obs in processor.system_observations.items():
        for obs_list in sys_obs.values():
            for expert_idx, _ in obs_list:
                counts[(expert_idx, system)] = counts.get((expert_idx, system), 0) + 1

    def total_for(expert_idx: int, systems: List[str]) -> int:
        return sum(counts.get((expert_idx, system), 0) for system in systems)

    def choose(systems: List[str]) -> Optional[int]:
        candidates = [
            idx
            for idx in range(len(processor.expert_names))
            if all(counts.get((idx, system), 0) > 0 for system in systems)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda idx: (total_for(idx, systems), -idx))

    roles = {
        "E_cross": choose([SYSTEM_HUMAN, SYSTEM_ELIZA]),
        "E_chicken_focus": choose([SYSTEM_CHICKEN]),
        "E_llm_focus": choose([SYSTEM_LLM]),
    }
    return {
        role: processor.expert_label(idx) if idx is not None else "unavailable"
        for role, idx in roles.items()
    }


def focus_ppc(
    ppc: Dict[Tuple[str, str], Dict[str, Any]],
    roles: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    focus_specs = {
        "cross_human": (roles["E_cross"], SYSTEM_HUMAN),
        "cross_eliza": (roles["E_cross"], SYSTEM_ELIZA),
        "chicken_focus": (roles["E_chicken_focus"], SYSTEM_CHICKEN),
        "llm_focus": (roles["E_llm_focus"], SYSTEM_LLM),
    }
    out: Dict[str, Dict[str, float]] = {}
    for label, key in focus_specs.items():
        result = ppc.get(key)
        if result is None:
            continue
        out[label] = {
            "n_obs": float(result["n_obs"]),
            "delta_left_mean": float(result["delta_left_mean"]),
            "delta_right_mean": float(result["delta_right_mean"]),
            "delta_mid_mean": float(result["delta_mid_mean"]),
        }
    return out


def find_stance(data: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for item in data:
        if item.get("name") == name:
            return item
    raise ValueError(f"Stance not found: {name}")


def main() -> None:
    args = parse_args()
    config, system_configs = build_config(args)
    data, data_source = load_or_fetch_data(config)
    stance_data = find_stance(data, TARGET_STANCE)

    processor = MultiSystemDataProcessor(config)
    processor.process(stance_data, SYSTEMS)
    builder = MultiSystemModelBuilder(
        config, EvidenceProcessor(config), processor, system_configs
    )
    model = builder.build_model(stance_data)
    idata = builder.sample(model)

    roles = role_labels(processor)
    ppc_focus: Dict[str, Dict[str, float]] = {}
    if args.ppc_draws > 0:
        ppc = per_expert_system_ppc_multisystem(
            idata, builder, processor, n_draws=args.ppc_draws, seed=args.ppc_seed
        )
        ppc_focus = focus_ppc(ppc, roles)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) / f"{args.config}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_path = output_dir / f"{args.config}.nc"
    idata.to_netcdf(inference_path)

    metadata = {
        "config_name": args.config,
        "git_sha": git_sha(),
        "timestamp_utc": timestamp,
        "seeds": {"sampling": args.seed, "ppc": args.ppc_seed},
        "sampling_budget": {
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
        },
        "target_accept": args.target_accept,
        "data_source": data_source,
        "canonical_data_hash": canonical_data_hash(data),
        "package_versions": package_versions(),
        "systems": SYSTEMS,
        "expert_labels": [processor.expert_label(i) for i in range(len(processor.expert_names))],
        "focus_roles": roles,
        "diagnostics": diagnostics_summary(idata, args.config),
        "focus_ppc": ppc_focus,
        "inference_data": str(inference_path),
    }
    metadata_path = output_dir / f"{args.config}.meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "metadata": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()
