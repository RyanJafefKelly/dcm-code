"""Analytical prior-implied q_j check.

For each prior setting, walk the GWT tree and compute the prior-mean q_j at
each internal node for Human (C=0.999) and ELIZA (C=0.001), using the
recursion q_child = β_abs + parent_q · (β_pres − β_abs) with the prior-mean
betas (i.e., the override values, or paper means if no override).

Goal: confirm the new asymmetric prior actually transmits the root C signal
to deeper nodes. Under the production prior, average δ = β_pres − β_abs is
small enough that q_Human and q_ELIZA collapse together by depth 3. Under
β_pres ≈ 1, β_abs ≈ 0, they should stay separated.

Runs in seconds; no PyMC sampling.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcm_model import EvidenceProcessor, ModelConfig, load_data, node_key  # noqa: E402

STANCE = "Global Workspace Theory"
PRIOR_SETTINGS: List[Tuple[str, float | None, float | None]] = [
    ("baseline (paper means)", None, None),
    ("pres0.85_abs0.15", 0.85, 0.15),
    ("pres0.90_abs0.10", 0.90, 0.10),
    ("pres0.95_abs0.05", 0.95, 0.05),
]
C_VALUES = {"Human": 0.999, "ELIZA": 0.001}


def prior_mean_betas_by_node(
    stance_data: Dict, evidence: EvidenceProcessor,
    pres_override: float | None, abs_override: float | None,
) -> Dict[str, Tuple[float, float, int]]:
    """Walk tree; return {node_key: (β_pres, β_abs, depth)}.
    Uses paper_mu(s,d) unless overridden globally.
    """
    out: Dict[str, Tuple[float, float, int]] = {}
    root_path = (stance_data["name"],)

    def walk(node: Dict, ancestor_path: Tuple[str, ...], depth: int) -> None:
        key = node_key(ancestor_path, node["name"])
        s = node.get("support", "no bearing")
        d = node.get("demandingness", "neutral")
        if pres_override is not None:
            bp = float(pres_override)
        else:
            alpha_p, beta_p, _, _ = evidence.get_beta_parameters(s, d)
            bp = alpha_p / (alpha_p + beta_p)
        if abs_override is not None:
            ba = float(abs_override)
        else:
            _, _, alpha_a, beta_a = evidence.get_beta_parameters("no bearing", d)
            ba = alpha_a / (alpha_a + beta_a)
        out[key] = (bp, ba, depth)
        current = ancestor_path + (node["name"],)
        for child in node.get("evidencers", []):
            walk(child, current, depth + 1)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path, 1)
    return out


def propagate_q(
    stance_data: Dict, betas: Dict[str, Tuple[float, float, int]], c_value: float,
) -> Dict[str, Tuple[float, int]]:
    """Compute prior-implied q_j at every internal node. Returns {key: (q, depth)}."""
    out: Dict[str, Tuple[float, int]] = {}
    root_path = (stance_data["name"],)

    def walk(node: Dict, ancestor_path: Tuple[str, ...], parent_q: float) -> None:
        key = node_key(ancestor_path, node["name"])
        bp, ba, depth = betas[key]
        q = ba + parent_q * (bp - ba)
        if (node.get("type") or "").lower() in {"feature", "subfeature", "indicator"}:
            out[key] = (float(q), depth)
        current = ancestor_path + (node["name"],)
        for child in node.get("evidencers", []):
            walk(child, current, q)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path, c_value)
    return out


def main() -> None:
    cfg = ModelConfig(INDICATOR_STATE_MODEL="three_state")
    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    evidence = EvidenceProcessor(cfg)

    rows: List[Dict] = []
    for label, pres_o, abs_o in PRIOR_SETTINGS:
        betas = prior_mean_betas_by_node(stance_data, evidence, pres_o, abs_o)
        for sys_name, c_val in C_VALUES.items():
            q_by_key = propagate_q(stance_data, betas, c_val)
            depths = sorted({d for _, d in q_by_key.values()})
            for depth in depths:
                qs = [q for q, dd in q_by_key.values() if dd == depth]
                rows.append({
                    "prior": label,
                    "system": sys_name,
                    "depth": depth,
                    "n_nodes": len(qs),
                    "q_mean": float(np.mean(qs)),
                    "q_min": float(np.min(qs)),
                    "q_max": float(np.max(qs)),
                })

    df = pd.DataFrame(rows)
    out_path = Path(__file__).parent / "eval" / "analytical_qj_by_prior_depth.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    print()

    # Pivot: per prior, q_Human - q_ELIZA mean by depth
    print("Mean q_j(Human, C=0.999) − q_j(ELIZA, C=0.001) by prior × depth")
    print("(transmission gap; should stay large under the new asymmetric prior)")
    print()
    pivot_rows = []
    for label, _, _ in PRIOR_SETTINGS:
        for depth in sorted(df["depth"].unique()):
            qh = df[(df.prior == label) & (df.system == "Human") & (df.depth == depth)]["q_mean"].values
            qe = df[(df.prior == label) & (df.system == "ELIZA") & (df.depth == depth)]["q_mean"].values
            if len(qh) and len(qe):
                pivot_rows.append({"prior": label, "depth": depth, "q_Human": float(qh[0]), "q_ELIZA": float(qe[0]), "gap": float(qh[0] - qe[0])})
    pivot = pd.DataFrame(pivot_rows)
    print(pivot.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
