"""First full-GWT exact-tree synthetic dataset scaffold.

This script only generates the synthetic stance-data copy and truth artifacts.
It does not fit either the exact tree or the composite model.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import arviz as az
import numpy as np
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcm_model import EvidenceProcessor, ModelConfig, load_data, node_key  # noqa: E402
from gwt_reference_recovery_analysis import ANCHORED_SYSTEM_CONFIGS  # noqa: E402


STANCE = "Global Workspace Theory"
EXACT_PROD_PATH = REPO_ROOT / "results/gwt_exact_tree/three_state_pooled_abs_by_sd_exact_anchored.nc"
DEFAULT_RUNS_DIR = REPO_ROOT / "notebooks/synthetic_validation_2026-05-06/runs"

TRUE_C_BY_SYSTEM = {
    "Human": 0.999,
    "Chicken": 0.25,
    "2024 Leading Chat LLMs": 0.10,
    "ELIZA": 0.001,
}

CATEGORY_REPRESENTATIVES = (0.025, 0.125, 0.300, 0.500, 0.700, 0.875, 0.975)


def is_missing(val: Any) -> bool:
    if val is None:
        return True
    s = str(val).strip().lower()
    return s in {"-1", "-1.0", "none", "", "unsure", "not tested"}


def iter_tree_nodes(stance_data: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], str, Tuple[str, ...]]]:
    """Yield every non-root tree node with its path-key and current path."""
    root_path = (stance_data["name"],)

    def walk(node: Dict[str, Any], ancestor_path: Tuple[str, ...]):
        key = node_key(ancestor_path, node["name"])
        current_path = ancestor_path + (node["name"],)
        yield node, key, current_path
        for child in node.get("evidencers", []):
            yield from walk(child, current_path)

    for child in stance_data.get("evidencers", []):
        yield from walk(child, root_path)


def load_observation_medians(path: Path, k: int) -> Dict[str, Any]:
    """Use existing production medians only for realistic ordinal emissions."""
    if not path.exists():
        raise FileNotFoundError(f"Missing exact production posterior: {path}")
    idata = az.from_netcdf(path)
    post = idata.posterior
    a = float(np.median(np.asarray(post["a"].values).reshape(-1)))
    kappa = np.median(
        np.asarray(post["kappa"].values).reshape(-1, k - 1),
        axis=0,
    )
    return {
        "source": str(path.relative_to(REPO_ROOT)),
        "a": a,
        "kappa": [float(x) for x in kappa],
        "use_expert_shifts": False,
        "expert_shift": 0.0,
    }


def collect_unpooled_paper_mean_betas(
    stance_data: Dict[str, Any],
    evidence_processor: EvidenceProcessor,
) -> Dict[str, Dict[str, Any]]:
    edge_betas: Dict[str, Dict[str, Any]] = {}
    for node, key, _ in iter_tree_nodes(stance_data):
        support = node.get("support", "no bearing")
        demandingness = node.get("demandingness", "neutral")
        alpha_p, beta_p, alpha_a, beta_a = evidence_processor.get_beta_parameters(
            support, demandingness,
        )
        bp = alpha_p / (alpha_p + beta_p)
        ba = alpha_a / (alpha_a + beta_a)
        edge_betas[key] = {
            "node_name": node["name"],
            "node_type": (node.get("type") or "").lower(),
            "support": support,
            "demandingness": demandingness,
            "beta_pres": float(bp),
            "beta_abs": float(ba),
            "delta": float(bp - ba),
        }
    return edge_betas


def compute_indicator_delta(
    stance_data: Dict[str, Any],
    edge_betas: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    root_path = (stance_data["name"],)

    def walk(node: Dict[str, Any], ancestor_path: Tuple[str, ...], parent_delta: float) -> None:
        key = node_key(ancestor_path, node["name"])
        current_delta = parent_delta * edge_betas[key]["delta"]
        current_path = ancestor_path + (node["name"],)
        if (node.get("type") or "").lower() == "indicator":
            out[key] = float(current_delta)
            return
        for child in node.get("evidencers", []):
            walk(child, current_path, current_delta)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path, 1.0)
    return out


def ordered_probit_probs(kappa: np.ndarray, eta: float) -> np.ndarray:
    cum = np.concatenate(([0.0], norm.cdf(kappa - eta), [1.0]))
    probs = np.diff(cum)
    probs = np.clip(probs, 1e-12, 1.0)
    return probs / probs.sum()


def sample_rating_category(rng: np.random.Generator, kappa: np.ndarray, a: float, m: int) -> int:
    eta = a * (m / 2.0)
    probs = ordered_probit_probs(kappa, eta)
    return int(rng.choice(np.arange(len(probs)), p=probs))


def sample_latent_tree_for_system(
    rng: np.random.Generator,
    stance_data: Dict[str, Any],
    edge_betas: Dict[str, Dict[str, Any]],
    true_c: float,
) -> Dict[str, Any]:
    root_z = int(rng.binomial(1, true_c))
    internal_z: Dict[str, int] = {}
    indicator_m: Dict[str, int] = {}
    root_path = (stance_data["name"],)

    def walk(node: Dict[str, Any], ancestor_path: Tuple[str, ...], parent_z: int) -> None:
        key = node_key(ancestor_path, node["name"])
        beta = edge_betas[key]["beta_pres"] if parent_z else edge_betas[key]["beta_abs"]
        ntype = (node.get("type") or "").lower()
        if ntype == "indicator":
            indicator_m[key] = int(rng.binomial(2, beta))
            return
        z = int(rng.binomial(1, beta))
        internal_z[key] = z
        current_path = ancestor_path + (node["name"],)
        for child in node.get("evidencers", []):
            walk(child, current_path, z)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path, root_z)

    return {
        "root_z": root_z,
        "internal_z": internal_z,
        "indicator_m": indicator_m,
    }


def simulate_observations(
    rng: np.random.Generator,
    stance_data: Dict[str, Any],
    latent_by_system: Dict[str, Dict[str, Any]],
    obs_params: Dict[str, Any],
) -> Dict[str, Dict[str, List[int]]]:
    """Mutate stance_data observations in place and return ordinal truth."""
    a = float(obs_params["a"])
    kappa = np.asarray(obs_params["kappa"], dtype=float)
    ordinal_by_system_indicator: Dict[str, Dict[str, List[int]]] = {
        sys_name: {} for sys_name in TRUE_C_BY_SYSTEM
    }

    def walk(node: Dict[str, Any], ancestor_path: Tuple[str, ...]) -> None:
        key = node_key(ancestor_path, node["name"])
        current_path = ancestor_path + (node["name"],)
        if (node.get("type") or "").lower() == "indicator":
            for sys_name, obs in node.get("observations", {}).items():
                if sys_name not in latent_by_system:
                    continue
                values = obs.get("values", [])
                simulated_values: List[Any] = []
                ordinal_values: List[int] = []
                m = latent_by_system[sys_name]["indicator_m"][key]
                for val in values:
                    if is_missing(val):
                        simulated_values.append(-1)
                        continue
                    category = sample_rating_category(rng, kappa, a, m)
                    ordinal_values.append(category)
                    simulated_values.append(float(CATEGORY_REPRESENTATIVES[category]))
                obs["values"] = simulated_values
                ordinal_by_system_indicator[sys_name][key] = ordinal_values
            return
        for child in node.get("evidencers", []):
            walk(child, current_path)

    for child in stance_data.get("evidencers", []):
        walk(child, (stance_data["name"],))
    return ordinal_by_system_indicator


def rating_counts_and_categories(
    stance_data: Dict[str, Any],
    systems: List[str],
    cfg: ModelConfig,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    counts = {sys_name: 0 for sys_name in systems}
    cats = {sys_name: {str(i): 0 for i in range(cfg.N_CATEGORIES)} for sys_name in systems}

    def to_category(value: Any) -> int:
        p = float(value)
        for i, edge in enumerate(cfg.ORDINAL_BINS):
            if p < edge:
                return i
        return len(cfg.ORDINAL_BINS)

    def walk(node: Dict[str, Any]) -> None:
        if (node.get("type") or "").lower() == "indicator":
            for sys_name, obs in node.get("observations", {}).items():
                if sys_name not in counts:
                    continue
                for val in obs.get("values", []):
                    if is_missing(val):
                        continue
                    c = to_category(val)
                    counts[sys_name] += 1
                    cats[sys_name][str(c)] += 1
        for child in node.get("evidencers", []):
            walk(child)

    for child in stance_data.get("evidencers", []):
        walk(child)
    return counts, cats


def build_sanity_checks(
    stance_data: Dict[str, Any],
    source_stance_data: Dict[str, Any],
    latent_by_system: Dict[str, Dict[str, Any]],
    indicator_delta: Dict[str, float],
    cfg: ModelConfig,
) -> Dict[str, Any]:
    systems = [s for s, _ in ANCHORED_SYSTEM_CONFIGS]
    counts, cats = rating_counts_and_categories(stance_data, systems, cfg)
    source_counts, source_cats = rating_counts_and_categories(source_stance_data, systems, cfg)
    delta_values = np.asarray(list(indicator_delta.values()), dtype=float)
    log_abs_delta = np.log(np.abs(delta_values) + 1e-12)

    def mass(
        count_map: Dict[str, int],
        cat_map: Dict[str, Dict[str, int]],
        sys_name: str,
        category: int,
    ) -> float:
        n = count_map[sys_name]
        return float(cat_map[sys_name][str(category)] / n) if n else math.nan

    return {
        "rating_counts_by_system": counts,
        "category_counts_by_system": cats,
        "human_top_category_mass": mass(counts, cats, "Human", cfg.N_CATEGORIES - 1),
        "eliza_bottom_category_mass": mass(counts, cats, "ELIZA", 0),
        "source_observed_reference": {
            "rating_counts_by_system": source_counts,
            "category_counts_by_system": source_cats,
            "human_top_category_mass": mass(
                source_counts, source_cats, "Human", cfg.N_CATEGORIES - 1,
            ),
            "eliza_bottom_category_mass": mass(source_counts, source_cats, "ELIZA", 0),
        },
        "delta_j": {
            "n": int(delta_values.size),
            "mean": float(delta_values.mean()),
            "median": float(np.median(delta_values)),
            "min": float(delta_values.min()),
            "max": float(delta_values.max()),
            "mean_log_abs_plus_eps": float(log_abs_delta.mean()),
            "median_log_abs_plus_eps": float(np.median(log_abs_delta)),
        },
        "internal_z_ones_by_system": {
            sys_name: int(sum(latent["internal_z"].values()))
            for sys_name, latent in latent_by_system.items()
        },
        "internal_z_count_by_system": {
            sys_name: int(len(latent["internal_z"]))
            for sys_name, latent in latent_by_system.items()
        },
        "root_z_by_system": {
            sys_name: int(latent["root_z"])
            for sys_name, latent in latent_by_system.items()
        },
    }


def print_sanity_checks(sanity: Dict[str, Any]) -> None:
    print("=== Synthetic simulator sanity checks ===")
    print("rating counts by system:")
    for sys_name, n in sanity["rating_counts_by_system"].items():
        print(f"  {sys_name:<28} {n:>4}")
    print(f"Human top-category mass:    {sanity['human_top_category_mass']:.3f}")
    print(f"ELIZA bottom-category mass: {sanity['eliza_bottom_category_mass']:.3f}")
    ref = sanity["source_observed_reference"]
    print(
        "observed-data reference masses: "
        f"Human top {ref['human_top_category_mass']:.3f}, "
        f"ELIZA bottom {ref['eliza_bottom_category_mass']:.3f}"
    )
    d = sanity["delta_j"]
    print(
        "delta_j mean/median/log-median: "
        f"{d['mean']:.4f} / {d['median']:.4f} / {d['median_log_abs_plus_eps']:.3f}"
    )
    print("internal z=1 counts by system:")
    for sys_name, n_one in sanity["internal_z_ones_by_system"].items():
        n_total = sanity["internal_z_count_by_system"][sys_name]
        print(f"  {sys_name:<28} {n_one:>2} / {n_total}")
    print("root z by system:")
    for sys_name, z in sanity["root_z_by_system"].items():
        print(f"  {sys_name:<28} {z}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or f"exact_unpooled_paper_mean_seed{args.seed}"
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else REPO_ROOT / args.runs_dir
    out_dir = runs_dir / run_id
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Run directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cfg = ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=False,
    )
    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    synthetic_stance_data = copy.deepcopy(stance_data)

    evidence_processor = EvidenceProcessor(cfg)
    edge_betas = collect_unpooled_paper_mean_betas(stance_data, evidence_processor)
    indicator_delta = compute_indicator_delta(stance_data, edge_betas)
    obs_params = load_observation_medians(EXACT_PROD_PATH, cfg.N_CATEGORIES)

    latent_by_system = {
        sys_name: sample_latent_tree_for_system(
            rng, stance_data, edge_betas, TRUE_C_BY_SYSTEM[sys_name],
        )
        for sys_name, _ in ANCHORED_SYSTEM_CONFIGS
    }
    ordinal_by_system_indicator = simulate_observations(
        rng, synthetic_stance_data, latent_by_system, obs_params,
    )
    sanity = build_sanity_checks(
        synthetic_stance_data, stance_data, latent_by_system, indicator_delta, cfg,
    )

    config_payload = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "stance": STANCE,
        "systems": [s for s, _ in ANCHORED_SYSTEM_CONFIGS],
        "tree_semantics": "exact_latent_state_tree",
        "edge_beta_source": "unpooled_paper_prior_means",
        "indicator_leaf": "three_state_binomial_2",
        "observation_parameter_source": obs_params["source"],
        "model_config": {
            "INDICATOR_STATE_MODEL": cfg.INDICATOR_STATE_MODEL,
            "USE_EXPERT_SHIFTS": cfg.USE_EXPERT_SHIFTS,
            "USE_HIERARCHICAL_EXPERT_CUTPOINTS": cfg.USE_HIERARCHICAL_EXPERT_CUTPOINTS,
            "POOL_BETAS_BY_LABEL": cfg.POOL_BETAS_BY_LABEL,
            "BETA_ABS_BY_SUPPORT_DEMAND": cfg.BETA_ABS_BY_SUPPORT_DEMAND,
            "ORDINAL_BINS": list(cfg.ORDINAL_BINS),
        },
    }
    truth_payload = {
        "true_C_by_system": TRUE_C_BY_SYSTEM,
        "latent_by_system": latent_by_system,
        "edge_betas": edge_betas,
        "indicator_delta": indicator_delta,
        "observation_parameters": obs_params,
        "category_representatives": list(CATEGORY_REPRESENTATIVES),
        "ordinal_by_system_indicator": ordinal_by_system_indicator,
    }

    write_json(out_dir / "config.json", config_payload)
    write_json(out_dir / "truth.json", truth_payload)
    write_json(out_dir / "sanity_checks.json", sanity)
    write_json(out_dir / "synthetic_stance_data.json", synthetic_stance_data)

    print_sanity_checks(sanity)
    print(f"\nwrote {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
