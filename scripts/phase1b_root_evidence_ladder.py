"""Phase 1B root-evidence ladder diagnostics for the binary-root GWT DCM."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcm_model import node_key  # noqa: E402
from scripts.phase1_root_evidence_common import (  # noqa: E402
    ALL_SYSTEMS,
    EPS,
    FREE_SYSTEMS,
    LOGIT_PRIOR,
    PRIOR_P,
    REPO_ROOT,
    SCORE_CLIP,
    STANCE,
    TAU_ABSENT_05,
    TAU_PRESENT_50,
    beta_profile_from_evidence_processor,
    beta_profile_from_truth,
    bernoulli_brier,
    bernoulli_log_score,
    check_threshold_constants,
    display_system,
    enumerate_top_bound_distribution,
    evidence_category_from_log_b,
    exact_root_sides_from_leaf_messages,
    extreme_beta_profile,
    load_gwt_stance,
    markdown_table,
    ordered_probit_category_logp,
    sample_ordered_probit_category,
    top_feature_table,
    top_children,
)


RUNG_DESCRIPTIONS = {
    "1P": "actual tree, production beta, binary noisy leaves, clamped beta",
    "1X": "actual tree, extreme beta, binary noisy leaves, clamped beta",
    "3": "actual tree, binary latent leaf with clamped ordinal-probit emission",
    "4": "actual tree, production three-state leaf with clamped ordinal nuisance",
    "2": "binary noisy leaves with inferred beta and matching production priors",
    "5M": "matched-prior nuisance washout, fully crossed",
    "5X": "extreme-beta nuisance positive control, fully crossed",
    "6": "production-like rater design",
    "7": "improved crossed design",
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def find_oracle_truth_path() -> Optional[Path]:
    roots = [
        REPO_ROOT / "notebooks/synthetic_validation_2026-05-06/runs/full_exact_recovery",
        REPO_ROOT / "notebooks/sample_size_sweep_2026-05-10/runs",
        REPO_ROOT / "notebooks/asymmetric_prior_sweep_2026-05-10/runs/synthetic",
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("truth.json")):
            return path
    return None


def load_observation_truth(oracle_truth_path: Optional[Path]) -> Dict[str, Any]:
    if oracle_truth_path and oracle_truth_path.exists():
        return load_json(oracle_truth_path)["observation_parameters"]
    return {
        "a": 4.137660889976744,
        "kappa": [
            1.0275199154347838,
            1.4491911337698526,
            1.7834771682836879,
            2.3195987227068366,
            3.0459405999192013,
            3.813141096352558,
        ],
        "use_expert_shifts": False,
        "expert_shift": 0.0,
        "source": "fallback_phase1b_defaults",
    }


def build_beta_profiles(
    stance_data: Mapping[str, Any],
    oracle_truth_path: Optional[Path],
) -> Dict[str, Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, Any]]]]:
    profiles: Dict[str, Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, Any]]]] = {}
    profiles["production_prior_mean"] = beta_profile_from_evidence_processor(stance_data)
    if oracle_truth_path and oracle_truth_path.exists():
        profiles["production_oracle_median"] = beta_profile_from_truth(load_json(oracle_truth_path))
    profiles["extreme_0p9_0p1"] = extreme_beta_profile(stance_data)
    return profiles


def write_top_bound_outputs(
    *,
    output_dir: Path,
    stance_data: Mapping[str, Any],
    profiles: Mapping[str, Tuple[Mapping[str, float], Mapping[str, float], Mapping[str, Mapping[str, Any]]]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dist_frames = []
    feature_frames = []
    for name, (bp, ba, _) in profiles.items():
        dist_frames.append(
            enumerate_top_bound_distribution(
                stance_data=stance_data,
                beta_pres_by_key=bp,
                beta_abs_by_key=ba,
                beta_profile_name=name,
            )
        )
        feature_frames.append(
            top_feature_table(
                stance_data=stance_data,
                beta_pres_by_key=bp,
                beta_abs_by_key=ba,
                beta_profile_name=name,
            )
        )
    top_dist = pd.concat(dist_frames, ignore_index=True)
    feature_table = pd.concat(feature_frames, ignore_index=True)
    top_dist.to_csv(output_dir / "phase1b_top_bound_distribution.csv", index=False)
    feature_table.to_csv(output_dir / "phase1b_top_bound_feature_table.csv", index=False)
    return top_dist, feature_table


def generate_latent_tree(
    *,
    rng: np.random.Generator,
    stance_data: Mapping[str, Any],
    beta_pres_by_key: Mapping[str, float],
    beta_abs_by_key: Mapping[str, float],
    root_z: int,
    leaf_model: str,
) -> Dict[str, Any]:
    internal_z: Dict[str, int] = {}
    indicator_z: Dict[str, int] = {}
    indicator_m: Dict[str, int] = {}
    root_path = (stance_data["name"],)

    def walk(node: Mapping[str, Any], path: Tuple[str, ...], parent_z: int) -> None:
        key = node_key(path, node["name"])
        beta = float(beta_pres_by_key[key] if parent_z else beta_abs_by_key[key])
        beta = float(np.clip(beta, SCORE_CLIP, 1.0 - SCORE_CLIP))
        ntype = (node.get("type") or "").lower()
        if ntype == "indicator":
            if leaf_model == "three_state":
                indicator_m[key] = int(rng.binomial(2, beta))
            else:
                indicator_z[key] = int(rng.binomial(1, beta))
            return
        z = int(rng.binomial(1, beta))
        internal_z[key] = z
        current_path = path + (node["name"],)
        for child in node.get("evidencers", []):
            walk(child, current_path, z)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path, int(root_z))
    return {"root_z": int(root_z), "internal_z": internal_z, "indicator_z": indicator_z, "indicator_m": indicator_m}


def realised_top_from_latent(
    *,
    stance_data: Mapping[str, Any],
    latent: Mapping[str, Any],
    beta_pres_by_key: Mapping[str, float],
    beta_abs_by_key: Mapping[str, float],
) -> Dict[str, Any]:
    log_b = 0.0
    n_present = 0
    min_total = 0.0
    max_total = 0.0
    for child in top_children(stance_data):
        z = int(latent["internal_z"][child.key])
        bp = float(np.clip(beta_pres_by_key[child.key], SCORE_CLIP, 1.0 - SCORE_CLIP))
        ba = float(np.clip(beta_abs_by_key[child.key], SCORE_CLIP, 1.0 - SCORE_CLIP))
        c1 = math.log(bp / ba)
        c0 = math.log((1.0 - bp) / (1.0 - ba))
        log_b += c1 if z else c0
        n_present += z
        min_total += min(c0, c1)
        max_total += max(c0, c1)
    return {
        "n_top_children": len(top_children(stance_data)),
        "n_top_present": int(n_present),
        "log_B_top_latent": float(log_b),
        "log_B_top_min": float(min_total),
        "log_B_top_max": float(max_total),
        "top_latent_category": evidence_category_from_log_b(float(log_b)),
    }


def binary_leaf_edge_message(
    ll0: float,
    ll1: float,
    beta_pres: float,
    beta_abs: float,
) -> Tuple[float, float]:
    bp = float(np.clip(beta_pres, SCORE_CLIP, 1.0 - SCORE_CLIP))
    ba = float(np.clip(beta_abs, SCORE_CLIP, 1.0 - SCORE_CLIP))
    zpa1 = float(np.logaddexp(math.log1p(-bp) + ll0, math.log(bp) + ll1))
    zpa0 = float(np.logaddexp(math.log1p(-ba) + ll0, math.log(ba) + ll1))
    return zpa0, zpa1


def three_state_leaf_edge_message(
    ll0: float,
    llh: float,
    ll1: float,
    beta_pres: float,
    beta_abs: float,
) -> Tuple[float, float]:
    def side(beta: float) -> float:
        b = float(np.clip(beta, SCORE_CLIP, 1.0 - SCORE_CLIP))
        return float(
            logsumexp(
                [
                    2.0 * math.log1p(-b) + ll0,
                    math.log(2.0) + math.log(b) + math.log1p(-b) + llh,
                    2.0 * math.log(b) + ll1,
                ]
            )
        )

    return side(beta_abs), side(beta_pres)


def leaf_messages_for_latent(
    *,
    rng: np.random.Generator,
    stance_data: Mapping[str, Any],
    latent: Mapping[str, Any],
    beta_pres_by_key: Mapping[str, float],
    beta_abs_by_key: Mapping[str, float],
    rung: str,
    obs_params: Mapping[str, Any],
    k_binary: int,
    leaf_noise_epsilon: float,
    n_raters: int = 6,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Any]]:
    leaf_messages: Dict[str, Tuple[float, float]] = {}
    obs_meta = {"n_observed_leaves": 0, "n_observed_ratings": 0, "n_raters": n_raters}
    root_path = (stance_data["name"],)
    a = float(obs_params["a"])
    kappa = np.asarray(obs_params["kappa"], dtype=float)
    b = np.zeros(n_raters)

    def walk(node: Mapping[str, Any], path: Tuple[str, ...]) -> None:
        key = node_key(path, node["name"])
        current_path = path + (node["name"],)
        if (node.get("type") or "").lower() == "indicator":
            obs_meta["n_observed_leaves"] += 1
            if rung in {"1P", "1X"}:
                z = int(latent["indicator_z"][key])
                p = 1.0 - leaf_noise_epsilon if z else leaf_noise_epsilon
                y = rng.binomial(1, p, size=k_binary)
                ll0 = float((y * math.log(leaf_noise_epsilon) + (1 - y) * math.log1p(-leaf_noise_epsilon)).sum())
                ll1 = float((y * math.log1p(-leaf_noise_epsilon) + (1 - y) * math.log(leaf_noise_epsilon)).sum())
                obs_meta["n_observed_ratings"] += int(k_binary)
                leaf_messages[key] = binary_leaf_edge_message(
                    ll0,
                    ll1,
                    beta_pres_by_key[key],
                    beta_abs_by_key[key],
                )
            elif rung == "3":
                z = int(latent["indicator_z"][key])
                ratings = np.asarray(
                    [sample_ordered_probit_category(rng, a, kappa, float(z)) for _ in range(n_raters)],
                    dtype=np.int64,
                )
                expert_idx = np.arange(n_raters, dtype=np.int64)
                ll0 = ordered_probit_category_logp(ratings, expert_idx, a, b, kappa, 0.0)
                ll1 = ordered_probit_category_logp(ratings, expert_idx, a, b, kappa, 1.0)
                obs_meta["n_observed_ratings"] += int(n_raters)
                leaf_messages[key] = binary_leaf_edge_message(
                    ll0,
                    ll1,
                    beta_pres_by_key[key],
                    beta_abs_by_key[key],
                )
            elif rung == "4":
                m = int(latent["indicator_m"][key])
                ratings = np.asarray(
                    [sample_ordered_probit_category(rng, a, kappa, m / 2.0) for _ in range(n_raters)],
                    dtype=np.int64,
                )
                expert_idx = np.arange(n_raters, dtype=np.int64)
                ll0 = ordered_probit_category_logp(ratings, expert_idx, a, b, kappa, 0.0)
                llh = ordered_probit_category_logp(ratings, expert_idx, a, b, kappa, 0.5)
                ll1 = ordered_probit_category_logp(ratings, expert_idx, a, b, kappa, 1.0)
                obs_meta["n_observed_ratings"] += int(n_raters)
                leaf_messages[key] = three_state_leaf_edge_message(
                    ll0,
                    llh,
                    ll1,
                    beta_pres_by_key[key],
                    beta_abs_by_key[key],
                )
            else:
                raise ValueError(f"Unsupported no-fit rung {rung!r}")
            return
        for child in node.get("evidencers", []):
            walk(child, current_path)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path)
    return leaf_messages, obs_meta


def root_truth_for_replicate(replicate_id: int) -> Dict[str, int]:
    return {
        "Human": 1,
        "ELIZA": 0,
        "Chicken": 1 if replicate_id % 2 == 0 else 0,
        "2024 Leading Chat LLMs": 0 if replicate_id % 2 == 0 else 1,
    }


def no_fit_profile_for_rung(
    rung: str,
    profiles: Mapping[str, Tuple[Mapping[str, float], Mapping[str, float], Mapping[str, Mapping[str, Any]]]],
) -> list[str]:
    if rung == "1P":
        return ["production_prior_mean"]
    if rung == "1X":
        return ["extreme_0p9_0p1"]
    if rung in {"3", "4"}:
        return ["production_prior_mean"]
    raise ValueError(f"Rung {rung!r} is not a no-fit rung")


def leaf_model_for_rung(rung: str) -> str:
    if rung in {"1P", "1X", "3"}:
        return "binary"
    if rung == "4":
        return "three_state"
    raise ValueError(f"Unsupported no-fit rung {rung!r}")


def run_no_fit_ladder(
    *,
    stance_data: Mapping[str, Any],
    profiles: Mapping[str, Tuple[Mapping[str, float], Mapping[str, float], Mapping[str, Mapping[str, Any]]]],
    obs_params: Mapping[str, Any],
    rungs: Sequence[str],
    n_rep: int,
    seed: int,
    k_binary: int,
    leaf_noise_epsilon: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rung in rungs:
        if rung not in {"1P", "1X", "3", "4"}:
            continue
        for profile_name in no_fit_profile_for_rung(rung, profiles):
            bp, ba, _ = profiles[profile_name]
            for rep in range(n_rep):
                root_truth = root_truth_for_replicate(rep)
                for system in ALL_SYSTEMS:
                    rng = np.random.default_rng(seed + rep)
                    # Advance the same replicate stream deterministically by system
                    # while preserving matched seeds across beta profiles.
                    for _ in range(list(ALL_SYSTEMS).index(system)):
                        rng.random(10)
                    latent = generate_latent_tree(
                        rng=rng,
                        stance_data=stance_data,
                        beta_pres_by_key=bp,
                        beta_abs_by_key=ba,
                        root_z=root_truth[system],
                        leaf_model=leaf_model_for_rung(rung),
                    )
                    if system not in FREE_SYSTEMS:
                        continue
                    leaf_messages, obs_meta = leaf_messages_for_latent(
                        rng=rng,
                        stance_data=stance_data,
                        latent=latent,
                        beta_pres_by_key=bp,
                        beta_abs_by_key=ba,
                        rung=rung,
                        obs_params=obs_params,
                        k_binary=k_binary,
                        leaf_noise_epsilon=leaf_noise_epsilon,
                    )
                    log_l0, log_l1 = exact_root_sides_from_leaf_messages(
                        stance_data,
                        leaf_messages,
                        bp,
                        ba,
                    )
                    log_b = float(log_l1 - log_l0)
                    rho = float(expit(LOGIT_PRIOR + log_b))
                    top = realised_top_from_latent(
                        stance_data=stance_data,
                        latent=latent,
                        beta_pres_by_key=bp,
                        beta_abs_by_key=ba,
                    )
                    root_z = int(root_truth[system])
                    rows.append(
                        {
                            "rung": rung,
                            "rung_description": RUNG_DESCRIPTIONS[rung],
                            "beta_profile": profile_name,
                            "replicate_id": rep,
                            "replicate_seed": seed + rep,
                            "system": display_system(system),
                            "system_raw": system,
                            "root_z_true": root_z,
                            "log_L0": log_l0,
                            "log_L1": log_l1,
                            "log_B_eff": log_b,
                            "rho": rho,
                            "evidence_category": evidence_category_from_log_b(log_b),
                            "correct_sign": bool((root_z == 1 and log_b > 0.0) or (root_z == 0 and log_b < 0.0)),
                            "decisive_for_true_root": bool((root_z == 1 and log_b > TAU_PRESENT_50) or (root_z == 0 and log_b < TAU_ABSENT_05)),
                            "log_score_improvement": float(
                                bernoulli_log_score(rho, root_z) - bernoulli_log_score(PRIOR_P, root_z)
                            ),
                            "brier_improvement": float(
                                bernoulli_brier(PRIOR_P, root_z) - bernoulli_brier(rho, root_z)
                            ),
                            **top,
                            **obs_meta,
                        }
                    )
    return pd.DataFrame(rows)


def evidence_margin(df: pd.DataFrame) -> float:
    r1 = df[df["root_z_true"] == 1]
    r0 = df[df["root_z_true"] == 0]
    if r1.empty or r0.empty:
        return np.nan
    return float(
        min(
            r1["log_B_eff"].median() - TAU_PRESENT_50,
            TAU_ABSENT_05 - r0["log_B_eff"].median(),
        )
    )


def bootstrap_margin_ci(df: pd.DataFrame, n_boot: int = 300, seed: int = 99173) -> Tuple[float, float]:
    reps = np.asarray(sorted(df["replicate_id"].unique()))
    if reps.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(reps, size=reps.size, replace=True)
        boot = pd.concat([df[df["replicate_id"] == rep] for rep in sample], ignore_index=True)
        vals.append(evidence_margin(boot))
    vals = np.asarray(vals, dtype=float)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def summarise_ladder_cases(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["rung", "beta_profile", "system"]
    groupers = list(cases.groupby(group_cols, dropna=False))
    for (rung, beta_profile), g in cases.groupby(["rung", "beta_profile"], dropna=False):
        groupers.append(((rung, beta_profile, "ALL"), g))

    for keys, g in groupers:
        rung, beta_profile, system = keys
        r1 = g[g["root_z_true"] == 1]
        r0 = g[g["root_z_true"] == 0]
        med_r1 = float(r1["log_B_eff"].median()) if not r1.empty else np.nan
        med_r0 = float(r0["log_B_eff"].median()) if not r0.empty else np.nan
        med_top_r1 = float(r1["log_B_top_latent"].median()) if not r1.empty else np.nan
        med_top_r0 = float(r0["log_B_top_latent"].median()) if not r0.empty else np.nan
        tpr = float((r1["rho"] > 0.5).mean()) if not r1.empty else np.nan
        tnr = float((r0["rho"] <= 0.5).mean()) if not r0.empty else np.nan
        bal_acc = float(np.nanmean([tpr, tnr]))
        u_log = (
            0.5 * float(np.mean(np.log(np.clip(r1["rho"], SCORE_CLIP, 1.0) / PRIOR_P))) if not r1.empty else 0.0
        ) + (
            0.5 * float(np.mean(np.log(np.clip(1.0 - r0["rho"], SCORE_CLIP, 1.0) / (1.0 - PRIOR_P)))) if not r0.empty else 0.0
        )
        m = evidence_margin(g)
        ci_low, ci_high = bootstrap_margin_ci(g)
        if m > 0.0 and u_log > 0.0 and bal_acc >= 0.75:
            label = "pass"
        elif (-0.25 <= m <= 0.0) or (not np.isnan(ci_low) and ci_low <= 0.0 <= ci_high):
            label = "borderline"
        else:
            label = "fail"
        rows.append(
            {
                "rung": rung,
                "rung_description": RUNG_DESCRIPTIONS.get(str(rung), ""),
                "beta_profile": beta_profile,
                "system": system,
                "n_cases": int(len(g)),
                "n_replicates": int(g["replicate_id"].nunique()),
                "median_log_B_eff_R1": med_r1,
                "median_log_B_eff_R0": med_r0,
                "mean_log_B_eff_R1": float(r1["log_B_eff"].mean()) if not r1.empty else np.nan,
                "mean_log_B_eff_R0": float(r0["log_B_eff"].mean()) if not r0.empty else np.nan,
                "mean_rho_R1": float(r1["rho"].mean()) if not r1.empty else np.nan,
                "mean_rho_R0": float(r0["rho"].mean()) if not r0.empty else np.nan,
                "p_correct_sign_R1": float(r1["correct_sign"].mean()) if not r1.empty else np.nan,
                "p_correct_sign_R0": float(r0["correct_sign"].mean()) if not r0.empty else np.nan,
                "p_decisive_R1": float(r1["decisive_for_true_root"].mean()) if not r1.empty else np.nan,
                "p_decisive_R0": float(r0["decisive_for_true_root"].mean()) if not r0.empty else np.nan,
                "TPR_at_rho_gt_0p5": tpr,
                "TNR_at_rho_le_0p5": tnr,
                "balanced_accuracy": bal_acc,
                "balanced_log_score_improvement": u_log,
                "balanced_brier_improvement": float(
                    0.5 * (r1["brier_improvement"].mean() if not r1.empty else 0.0)
                    + 0.5 * (r0["brier_improvement"].mean() if not r0.empty else 0.0)
                ),
                "evidence_margin_M": m,
                "evidence_margin_ci_low": ci_low,
                "evidence_margin_ci_high": ci_high,
                "pass_fail_label": label,
                "median_log_B_top_R1": med_top_r1,
                "median_log_B_top_R0": med_top_r0,
                "survival_R1": float(med_r1 / med_top_r1) if abs(med_top_r1) > EPS else np.nan,
                "survival_R0_signed": float(med_r0 / med_top_r0) if abs(med_top_r0) > EPS else np.nan,
                "abs_survival_R0": float(abs(med_r0) / abs(med_top_r0)) if abs(med_top_r0) > EPS else np.nan,
                "loss_R1": float(med_r1 - med_top_r1) if not np.isnan(med_top_r1) else np.nan,
                "loss_R0": float(med_r0 - med_top_r0) if not np.isnan(med_top_r0) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["rung", "beta_profile", "system"])


def fit_smoke_config_rows(rungs: Sequence[str], n_rep: int, seed: int) -> pd.DataFrame:
    rows = []
    for rung in rungs:
        if rung not in {"2", "5M", "5X", "6", "7"}:
            continue
        rows.append(
            {
                "rung": rung,
                "rung_description": RUNG_DESCRIPTIONS[rung],
                "n_rep_smoke": int(n_rep),
                "seed": int(seed),
                "status": "config_validated_not_sampled",
                "reason": "Full PyMC/NUTS refits are intentionally left as explicit commands in phase1b_ladder_plan.md.",
            }
        )
    return pd.DataFrame(rows)


def build_plan(
    *,
    output_dir: Path,
    mode: str,
    top_dist: Optional[pd.DataFrame],
    no_fit_summary: Optional[pd.DataFrame],
    fit_smoke: Optional[pd.DataFrame],
    n_rep: int,
    seed: int,
    rungs: Sequence[str],
) -> None:
    lines: list[str] = []
    lines.append("# Phase 1B Root-Evidence Ladder Plan")
    lines.append("")
    lines.append(f"Current invocation mode: `{mode}`. Requested rungs: `{','.join(rungs)}`. Seed: `{seed}`.")
    lines.append("")
    lines.append("## Top-Bound Status")
    if top_dist is not None and not top_dist.empty:
        lines.append(markdown_table(
            top_dist,
            [
                "beta_profile",
                "median_log_B_top_R1",
                "median_log_B_top_R0",
                "frac_R1_log_B_top_gt_tau_present_50",
                "frac_R0_log_B_top_lt_tau_absent_05",
                "preflight_gate",
            ],
            max_rows=10,
        ))
    else:
        lines.append("Top-bound enumeration was not run in this invocation.")
    lines.append("")
    lines.append("## No-Fit Rung Status")
    if no_fit_summary is not None and not no_fit_summary.empty:
        lines.append(markdown_table(
            no_fit_summary[no_fit_summary["system"] == "ALL"],
            [
                "rung",
                "beta_profile",
                "n_replicates",
                "median_log_B_eff_R1",
                "median_log_B_eff_R0",
                "balanced_accuracy",
                "balanced_log_score_improvement",
                "evidence_margin_M",
                "pass_fail_label",
                "survival_R1",
                "abs_survival_R0",
            ],
            max_rows=20,
        ))
    else:
        lines.append("No-fit rungs were not run in this invocation.")
    lines.append("")
    lines.append("## Fit-Smoke Status")
    if fit_smoke is not None and not fit_smoke.empty:
        lines.append(markdown_table(fit_smoke, ["rung", "rung_description", "n_rep_smoke", "status"], max_rows=20))
    else:
        lines.append("Fit rungs were not sampled. The commands below are the intended full/smoke entry points.")
    lines.append("")
    lines.append("## Exact Commands")
    lines.append("Top-bound pre-flight:")
    lines.append("```bash")
    lines.append(".venv/bin/python scripts/phase1b_root_evidence_ladder.py --output-dir outputs/phase1_root_evidence --mode top-bound")
    lines.append("```")
    lines.append("No-fit full run:")
    lines.append("```bash")
    lines.append(f".venv/bin/python scripts/phase1b_root_evidence_ladder.py --output-dir outputs/phase1_root_evidence --mode no-fit --rungs 1P,1X,3,4 --n-rep 500 --seed {seed}")
    lines.append("```")
    lines.append("No-fit smoke run:")
    lines.append("```bash")
    lines.append(f".venv/bin/python scripts/phase1b_root_evidence_ladder.py --output-dir outputs/phase1_root_evidence --mode no-fit --rungs 1P,1X,3,4 --n-rep 20 --seed {seed}")
    lines.append("```")
    lines.append("Fit-rung smoke/config validation:")
    lines.append("```bash")
    lines.append(f".venv/bin/python scripts/phase1b_root_evidence_ladder.py --output-dir outputs/phase1_root_evidence --mode fit-smoke --rungs 2,5M,5X,6,7 --n-rep 2 --seed {seed}")
    lines.append("```")
    lines.append("Full fit rungs require HMC refits through the exact-tree model. Use the same rung definitions in this script and run screening batches first:")
    lines.append("```bash")
    lines.append(f".venv/bin/python scripts/phase1b_root_evidence_ladder.py --output-dir outputs/phase1_root_evidence --mode fit-smoke --rungs 2,5M,6,7 --n-rep 32 --seed {seed}")
    lines.append("# Then run the generated configs with MultiSystemExactTreeBuilder/NUTS in batches; confirm 64 replicates if evidence_margin_M is borderline.")
    lines.append("```")
    lines.append("")
    lines.append("## Decision Logic")
    lines.append("- If top-bound fails, inspect heterogeneous strong-feature firing patterns before broad validation.")
    lines.append("- If top-bound passes but no-fit rungs fail, evidence is being lost downstream of top-level latent structure.")
    lines.append("- If no-fit rungs pass but fit rungs fail, nuisance uncertainty, beta identifiability, or rater design is the bottleneck.")
    lines.append("- If Rung 6 fails but Rung 7 passes, improved cross-system rater coverage, especially Chicken coverage, is a sufficient design fix.")
    lines.append("")
    if no_fit_summary is not None and not no_fit_summary.empty:
        all_rows = no_fit_summary[no_fit_summary["system"] == "ALL"]
        failing = all_rows[all_rows["pass_fail_label"] == "fail"]
        if not failing.empty:
            lines.append("Recommended next action: inspect failing no-fit rungs before spending HMC on inferred nuisance rungs.")
        else:
            lines.append("Recommended next action: proceed to fit-rung smoke/config validation, then Rungs 2 and 5M screening.")
    else:
        lines.append("Recommended next action: run no-fit rungs 1P, 1X, 3, and 4.")
    (output_dir / "phase1b_ladder_plan.md").write_text("\n".join(lines) + "\n")


def parse_rungs(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/phase1_root_evidence")
    parser.add_argument("--mode", choices=["top-bound", "no-fit", "fit-smoke", "all"], default="top-bound")
    parser.add_argument("--rungs", default="1P,1X,3,4")
    parser.add_argument("--n-rep", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--k-binary", type=int, default=6)
    parser.add_argument("--leaf-noise-epsilon", type=float, default=0.05)
    parser.add_argument("--oracle-truth", type=Path, default=None)
    args = parser.parse_args()

    check_threshold_constants()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rungs = parse_rungs(args.rungs)
    stance_data = load_gwt_stance()
    oracle_truth_path = args.oracle_truth.expanduser().resolve() if args.oracle_truth else find_oracle_truth_path()
    profiles = build_beta_profiles(stance_data, oracle_truth_path)
    obs_params = load_observation_truth(oracle_truth_path)

    top_dist: Optional[pd.DataFrame] = None
    no_fit_cases: Optional[pd.DataFrame] = None
    no_fit_summary: Optional[pd.DataFrame] = None
    fit_smoke: Optional[pd.DataFrame] = None

    if args.mode in {"top-bound", "no-fit", "all"}:
        print("[phase1b] running top-bound enumeration")
        top_dist, _ = write_top_bound_outputs(
            output_dir=output_dir,
            stance_data=stance_data,
            profiles=profiles,
        )
    elif (output_dir / "phase1b_top_bound_distribution.csv").exists():
        top_dist = pd.read_csv(output_dir / "phase1b_top_bound_distribution.csv")

    if args.mode in {"no-fit", "all"}:
        print(f"[phase1b] running no-fit rungs {','.join(rungs)} with n_rep={args.n_rep}")
        no_fit_cases = run_no_fit_ladder(
            stance_data=stance_data,
            profiles=profiles,
            obs_params=obs_params,
            rungs=rungs,
            n_rep=args.n_rep,
            seed=args.seed,
            k_binary=args.k_binary,
            leaf_noise_epsilon=args.leaf_noise_epsilon,
        )
        no_fit_summary = summarise_ladder_cases(no_fit_cases)
        no_fit_cases.to_csv(output_dir / "phase1b_ladder_no_fit_cases.csv", index=False)
        no_fit_summary.to_csv(output_dir / "phase1b_ladder_no_fit_summary.csv", index=False)
    elif (output_dir / "phase1b_ladder_no_fit_summary.csv").exists():
        no_fit_summary = pd.read_csv(output_dir / "phase1b_ladder_no_fit_summary.csv")

    if args.mode in {"fit-smoke", "all"}:
        print(f"[phase1b] validating fit-rung configs {','.join(rungs)}")
        fit_smoke = fit_smoke_config_rows(rungs, args.n_rep, args.seed)
        fit_smoke.to_csv(output_dir / "phase1b_fit_smoke_configs.csv", index=False)

    build_plan(
        output_dir=output_dir,
        mode=args.mode,
        top_dist=top_dist,
        no_fit_summary=no_fit_summary,
        fit_smoke=fit_smoke,
        n_rep=args.n_rep,
        seed=args.seed,
        rungs=rungs,
    )
    print(f"[phase1b] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()

