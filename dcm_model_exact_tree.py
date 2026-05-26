"""Phase B — exact-tree marginal-likelihood builder (smoke prototype).

Replaces the per-indicator composite `pm.Potential`s of `MultiSystemModelBuilder`
with a single per-system `pm.Potential` evaluating the exact bottom-up DP for
the latent-tree marginal log-likelihood.

The tree topology is known statically at model-build time, so the DP is unrolled
into a PyTensor expression via Python recursion + `pt.logaddexp` — no scan,
no custom Op. Reuses `pt_three_state_ll_terms` from `dcm_model.py` for the
per-(system, indicator) (ℓ_0, ℓ_½, ℓ_1) tensors.

Smoke-test only: not optimised, no graph fusion, no JAX. Goal is to answer
the binary question for the meeting: does the exact-tree likelihood compile
and sample at all on GWT under the `pool_3s_abs_by_sd` configuration?

Sanity check: per-draw `log_L_total` from this PyTensor expression should match
the standalone NumPy DP from `composite_vs_exact_diagnostic.py:exact_loglik`
to numerical precision.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from dcm_model import (
    MultiSystemModelBuilder,
    build_label_pool_hyperparameters,
    build_ordinal_observation_layer,
    build_safe_gain_node_beta,
    node_key,
    pt_direct_q_ll,
    pt_three_state_ll_terms,
    pt_indicator_logps,
)


class MultiSystemExactTreeBuilder(MultiSystemModelBuilder):
    """Multi-system DCM with exact latent-tree marginal likelihood per system.

    Drop-in replacement for `MultiSystemModelBuilder` for prototyping the
    exact-tree refit. Same priors / pooling / observation layer / stance C
    construction; differs only in how leaf likelihoods are aggregated.
    """

    def build_model(self, stance_data: Dict) -> pm.Model:
        n_experts = len(self.multi_data.expert_names)
        K = self.config.N_CATEGORIES
        stance_name_raw = stance_data["name"]
        stance_name = self.sanitize_name(stance_name_raw)
        self.node_to_varname[stance_name_raw] = stance_name

        self.logger.info(
            f"Building EXACT-TREE multi-system DCM: stance={stance_name_raw}, "
            f"{len(self.system_configs)} systems, {n_experts} experts, K={K}"
        )

        if self.config.INDICATOR_STATE_MODEL not in {"binary", "three_state", "direct_q"}:
            raise ValueError(
                f"Exact-tree builder supports binary, three_state, or direct_q "
                f"leaf only; got {self.config.INDICATOR_STATE_MODEL!r}"
            )

        model = pm.Model()
        with model:
            # --- Per-system stance C (identical to parent) ---
            soft = self.config.SOFT_REFERENCE_ANCHORS or {}
            stance_qs: Dict[str, pt.TensorVariable] = {}
            for sys_name, c_fixed in self.system_configs:
                sp = self._sys_prefix(sys_name)
                if sys_name in soft:
                    alpha_s, beta_s = soft[sys_name]
                    c_var = pm.Beta(
                        f"{sp}__{stance_name}_C", alpha=alpha_s, beta=beta_s,
                    )
                    stance_qs[sys_name] = c_var
                elif c_fixed is not None:
                    c_val = pt.constant(c_fixed, dtype="floatX")
                    pm.Deterministic(f"{sp}__{stance_name}_C", c_val)
                    stance_qs[sys_name] = c_val
                else:
                    c_var = pm.Beta(
                        f"{sp}__{stance_name}_C",
                        alpha=self.config.DEFAULT_ALPHA,
                        beta=self.config.DEFAULT_BETA,
                    )
                    stance_qs[sys_name] = c_var

            # --- Shared observation layer ---
            (
                self.a,
                self.b,
                self.kappa,
                self.kappa_by_expert,
                self.sigma_by_expert,
            ) = build_ordinal_observation_layer(self.config, n_experts, K)

            # --- Optional label-level beta pooling (POOL_BETAS_BY_LABEL) ---
            if self.config.POOL_BETAS_BY_LABEL:
                (
                    self.beta_pres_by_group,
                    self.beta_abs_by_group,
                ) = build_label_pool_hyperparameters(
                    self.config, self.evidence_processor, stance_data
                )

            # --- Walk the tree once to register node varnames + collect
            #     β_pres / β_abs tensors per node_key (no Potentials, no
            #     per-system propagation Deterministics).
            node_betas: Dict[str, Tuple[pt.TensorVariable, pt.TensorVariable]] = {}
            self._collect_tree_betas(stance_data, node_betas)

            # --- Per (system, indicator with data), precompute leaf
            #     log-likelihood components (ℓ_0, ℓ_½, ℓ_1) tensors.
            indicator_leaf_lls = self._collect_indicator_leaf_lls(stance_data)

            # --- For each system, build the bottom-up DP expression and
            #     add a single per-system Potential.
            for sys_name, c_var in stance_qs.items():
                log_lik = self._exact_tree_log_likelihood(
                    stance_data, sys_name, c_var, node_betas, indicator_leaf_lls,
                )
                pm.Potential(
                    f"{self._sys_prefix(sys_name)}__exact_tree_lik",
                    log_lik,
                )

        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_tree_betas(
        self,
        stance_data: Dict,
        node_betas: Dict[str, Tuple[pt.TensorVariable, pt.TensorVariable]],
    ) -> None:
        """Walk tree, register varnames, populate node_betas dict.

        For non-pooled fits, instantiates per-node Beta priors here.
        For pooled fits, references `self.beta_pres_by_group` / `_abs_by_group`.
        """
        root_path = (stance_data["name"],)

        def walk(node: Dict, path: Tuple[str, ...]) -> None:
            current_path = path + (node["name"],)
            name = self.sanitize_name(node["name"])
            key = node_key(path, node["name"])
            self.node_to_varname[key] = name

            support = node.get("support", "no bearing")
            demand = node.get("demandingness", "neutral")

            if self.config.POOL_BETAS_BY_LABEL:
                bp = self.beta_pres_by_group[(support, demand)]
                abs_key: Any = (
                    (support, demand)
                    if self.config.BETA_ABS_BY_SUPPORT_DEMAND
                    else demand
                )
                ba = self.beta_abs_by_group[abs_key]
            elif self.config.GAIN_LOGIT_NORMAL and self.config.TRANSMISSION_GAIN != 1.0:
                mu_p, mu_a = self.evidence_processor.get_gained_means(support, demand)
                sigma = self.config.GAIN_LOGIT_NORMAL_SIGMA
                bp = build_safe_gain_node_beta(name, "pres", mu_p, sigma)
                ba = build_safe_gain_node_beta(name, "abs", mu_a, sigma)
            else:
                a_p, b_p, a_a, b_a = self.evidence_processor.get_beta_parameters(
                    support, demand
                )
                bp = pm.Beta(f"{name}_beta_pres", alpha=a_p, beta=b_p)
                ba = pm.Beta(f"{name}_beta_abs", alpha=a_a, beta=b_a)

            node_betas[key] = (bp, ba)

            if (node.get("type") or "").lower() in {"feature", "subfeature"}:
                for child in node.get("evidencers", []):
                    walk(child, current_path)

        for child in stance_data.get("evidencers", []):
            walk(child, root_path)

    def _collect_indicator_leaf_lls(
        self, stance_data: Dict
    ) -> Dict[Tuple[str, str], Tuple[pt.TensorVariable, ...]]:
        """For each (system, indicator-with-data), build the per-component
        leaf log-likelihood tensors (ℓ_0, ℓ_½, ℓ_1) for three-state OR
        (ℓ_0, ℓ_1) for binary. Returned dict keyed by (sys_name, node_key).
        """
        out: Dict[Tuple[str, str], Tuple[Any, ...]] = {}
        state_model = self.config.INDICATOR_STATE_MODEL
        root_path = (stance_data["name"],)

        def walk(node: Dict, path: Tuple[str, ...]) -> None:
            current_path = path + (node["name"],)
            ntype = (node.get("type") or "").lower()
            if ntype == "indicator":
                key = node_key(path, node["name"])
                for sys_name in self.multi_data.systems:
                    obs_list = self.multi_data.system_observations.get(
                        sys_name, {}
                    ).get(key, [])
                    if not obs_list:
                        continue
                    ratings = np.array([r for _, r in obs_list], dtype=np.int64)
                    expert_indices = np.array(
                        [e for e, _ in obs_list], dtype=np.int64
                    )
                    if state_model == "three_state":
                        ll0, llh, ll1 = pt_three_state_ll_terms(
                            ratings, expert_indices, self.a, self.kappa, self.b,
                            kappa_by_expert=self.kappa_by_expert,
                            sigma_by_expert=self.sigma_by_expert,
                        )
                        out[(sys_name, key)] = (ll0, llh, ll1)
                    elif state_model == "direct_q":
                        # No precomputable leaf log-lik tensors — η depends on
                        # β, which is the parent-conditional q. Stash the raw
                        # (ratings, expert_idx) so leaf_log_B can call
                        # pt_direct_q_ll at the right β.
                        out[(sys_name, key)] = (ratings, expert_indices)
                    else:
                        ll0, ll1 = pt_indicator_logps(
                            ratings, expert_indices, self.a, self.kappa, self.b,
                            kappa_by_expert=self.kappa_by_expert,
                            sigma_by_expert=self.sigma_by_expert,
                        )
                        out[(sys_name, key)] = (ll0, ll1)
                return
            for child in node.get("evidencers", []):
                walk(child, current_path)

        for child in stance_data.get("evidencers", []):
            walk(child, root_path)
        return out

    def _exact_tree_log_likelihood(
        self,
        stance_data: Dict,
        sys_name: str,
        c_var: pt.TensorVariable,
        node_betas: Dict[str, Tuple[pt.TensorVariable, pt.TensorVariable]],
        indicator_leaf_lls: Dict[Tuple[str, str], Tuple[pt.TensorVariable, ...]],
    ) -> pt.TensorVariable:
        """Bottom-up exact-tree marginal log-likelihood for one system.

        Returns
        -------
        scalar PyTensor — log P(observed ratings for this system | C, β, a, κ, b)
        marginalising over all internal-node binary z's exactly.
        """
        state_model = self.config.INDICATOR_STATE_MODEL
        root_path = (stance_data["name"],)
        eps = pt.constant(1e-12, dtype="floatX")

        def leaf_log_B(beta: pt.TensorVariable, leaf_lls) -> pt.TensorVariable:
            """Three-state mixture B(β) = (1-β)² ℓ_0 + 2β(1-β) ℓ_½ + β² ℓ_1.

            For binary leaf: B(β) = (1-β) ℓ_0 + β ℓ_1.
            For direct_q leaf: leaf_lls is (ratings, expert_idx); evaluate
            pt_direct_q_ll at q_eff = β (no marginalisation).
            Returned as scalar log.
            """
            if state_model == "direct_q":
                ratings, expert_idx = leaf_lls
                beta_c = pt.clip(beta, eps, 1.0 - eps)
                return pt_direct_q_ll(
                    ratings, expert_idx, beta_c,
                    self.a, self.kappa, self.b,
                    kappa_by_expert=self.kappa_by_expert,
                    sigma_by_expert=self.sigma_by_expert,
                )
            beta_c = pt.clip(beta, eps, 1.0 - eps)
            if state_model == "three_state":
                ll0, llh, ll1 = leaf_lls
                log_w0 = 2.0 * pt.log(1.0 - beta_c)
                log_w1 = pt.log(2.0) + pt.log(beta_c) + pt.log(1.0 - beta_c)
                log_w2 = 2.0 * pt.log(beta_c)
                return pt.logaddexp(
                    pt.logaddexp(log_w0 + ll0, log_w1 + llh),
                    log_w2 + ll1,
                )
            ll0, ll1 = leaf_lls
            return pt.logaddexp(
                pt.log(1.0 - beta_c) + ll0,
                pt.log(beta_c) + ll1,
            )

        def subtree_lls(
            node: Dict, path: Tuple[str, ...]
        ) -> Tuple[pt.TensorVariable, pt.TensorVariable]:
            """Return (log L(z_pa = 0), log L(z_pa = 1)) for this node.

            z_pa = parent's binary state. The returned tensors include
            integration over node's own latent state (and all descendants).
            For an internal node with no observed-data subtree, returns (0, 0).
            """
            current_path = path + (node["name"],)
            key = node_key(path, node["name"])
            ntype = (node.get("type") or "").lower()
            bp, ba = node_betas[key]

            if ntype == "indicator":
                lk = (sys_name, key)
                if lk not in indicator_leaf_lls:
                    # No observations for this (system, indicator)
                    z = pt.constant(0.0, dtype="floatX")
                    return z, z
                leaf_lls = indicator_leaf_lls[lk]
                # When parent z_pa = 1, q_self = β_pres; when z_pa = 0, q_self = β_abs
                log_L_zpa1 = leaf_log_B(bp, leaf_lls)
                log_L_zpa0 = leaf_log_B(ba, leaf_lls)
                return log_L_zpa0, log_L_zpa1

            # Internal node v (feature/subfeature):
            #   L_v(z_v=k) = ∏_c L_c(z_v = k) for each k.
            # Then self-edge to parent:
            #   L_self(z_pa=1) = β_pres_v · L_v(z_v=1) + (1-β_pres_v) · L_v(z_v=0)
            #   L_self(z_pa=0) = β_abs_v  · L_v(z_v=1) + (1-β_abs_v)  · L_v(z_v=0)
            log_L_v0 = pt.constant(0.0, dtype="floatX")
            log_L_v1 = pt.constant(0.0, dtype="floatX")
            for child in node.get("evidencers", []):
                cL_zpa0, cL_zpa1 = subtree_lls(child, current_path)
                log_L_v0 = log_L_v0 + cL_zpa0
                log_L_v1 = log_L_v1 + cL_zpa1
            bp_c = pt.clip(bp, eps, 1.0 - eps)
            ba_c = pt.clip(ba, eps, 1.0 - eps)
            log_L_zpa1 = pt.logaddexp(
                pt.log(bp_c) + log_L_v1,
                pt.log(1.0 - bp_c) + log_L_v0,
            )
            log_L_zpa0 = pt.logaddexp(
                pt.log(ba_c) + log_L_v1,
                pt.log(1.0 - ba_c) + log_L_v0,
            )
            return log_L_zpa0, log_L_zpa1

        # Aggregate top-level children's contributions, conditioning on stance C
        # (parent of all top-level features = the binary stance variable).
        log_L_top0 = pt.constant(0.0, dtype="floatX")
        log_L_top1 = pt.constant(0.0, dtype="floatX")
        for child in stance_data.get("evidencers", []):
            cL_zpa0, cL_zpa1 = subtree_lls(child, root_path)
            log_L_top0 = log_L_top0 + cL_zpa0
            log_L_top1 = log_L_top1 + cL_zpa1

        c_c = pt.clip(c_var, eps, 1.0 - eps)
        return pt.logaddexp(
            pt.log(c_c) + log_L_top1,
            pt.log(1.0 - c_c) + log_L_top0,
        )
