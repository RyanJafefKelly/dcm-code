# Prompt for GPT-5.5 Pro: DCM Tree Propagation Skill Graph

Copy this whole file into GPT-5.5 Pro with extended reasoning. The goal is to
produce an initial Math-Academy-style prerequisite graph for learning the DCM
tree propagation and synthetic-validation/oracle machinery.

The assistant receiving this prompt does not have access to my repo. I have
included the relevant local context and code excerpts below.

---

## 0. What I Want From Pro

I want to build a personal mini skill graph in the style of Math Academy. The
specific target is understanding the Digital Consciousness Model (DCM) code,
especially:

1. Tree probability propagation.
2. Exact-tree bottom-up likelihood propagation / message passing.
3. The "oracle" synthetic validation audits.
4. The difference between oracle internal-node identifiability and full
   fake-data recovery.
5. The scoring and interpretation of internal binary states.

Please produce an initial prerequisite graph, not a prose syllabus.

Use this output schema for each node unless you have a strong reason to modify
it:

```yaml
- id: short_snake_case_id
  title: Human readable title
  description: What the learner must understand or be able to do
  why_it_matters_for_dcm: Why this skill matters for this DCM work
  prerequisites: [direct_prerequisite_node_ids]
  key_prerequisites: [possibly deeper prerequisites that often explain failure]
  diagnostic_questions:
    - question 1
    - question 2
    - question 3
  mastery_criteria:
    - concrete observable criterion
  resources:
    - suggested textbook/course section or other high-quality source
  status: unknown
  tags: [probability, tree_dp, oracle, validation, etc]
```

Important constraints:

- Use small, teachable nodes. Split nodes if they have more than about 3-4
  direct prerequisites.
- Separate direct prerequisites from "key prerequisites" several layers back.
- Do not assume I have all standard prerequisites. Include foundational nodes
  when they are genuinely needed.
- Prefer a directed acyclic graph (DAG), not a linear course outline.
- Include a topological learning order.
- Mark a first "learning frontier": the smallest subset I should master first
  to understand tree propagation and the oracle audit.
- Include diagnostic questions that distinguish recognition from usable mastery.
- Include at least one code-reading diagnostic for nodes that map to repo code.
- Include a few challenging textbook-style questions that could be adapted into
  Math-Academy-style practice.
- Keep resources high-quality: textbooks, official docs, reliable lecture notes.
- Do not bloat the graph with every possible Bayesian-statistics topic. Include
  only topics that are prerequisites for this DCM target or likely gap-fillers.

Please return:

1. A concise explanation of the graph design.
2. The node list in YAML-like format.
3. The edge list separately, if useful.
4. A topological learning order.
5. A first 2-week learning frontier.
6. A list of likely missing prerequisites or blind spots.
7. A short implementation note for importing this into a repo named
   `ryan-graph`.

---

## 1. Math Academy Design Pattern To Emulate

Public Math Academy material describes:

- A knowledge graph whose nodes are mathematical topics and whose arrows encode
  relationships such as prerequisites.
- A student model that overlays a learner's answer history on the graph to
  estimate a knowledge profile.
- A diagnostic algorithm that estimates the learner's "knowledge frontier":
  what they know, what they are ready to learn, and which foundational gaps
  block progress.
- Fine-grained lessons broken into smaller knowledge points.
- Key prerequisites that may be direct prerequisites or several steps back in
  the graph.
- Mastery learning, active problem solving, spaced repetition, interleaving,
  and targeted remediation.

Useful public sources:

- https://www.mathacademy.com/how-our-ai-works
- https://mathacademy.com/how-it-works
- https://mathacademy.com/pedagogy
- https://www.justinmath.com/the-most-important-thing-to-understand-about-building-educational-knowledge-graphs/

The most relevant graph-design heuristic from Justin Skycak's public note:
avoid high prerequisite fan-in. More than about 3-4 direct prerequisites for a
lesson is a warning sign for cognitive overload. The pressure point is often
the key prerequisites needed for each knowledge point, not just the top-level
lesson prerequisites.

For this task, please make nodes small enough that I can test mastery with
2-5 questions.

---

## 2. Project Context

The Digital Consciousness Model (DCM) is a Bayesian hierarchical model for
aggregating expert evidence about whether systems may be conscious. It is
evaluated under multiple "stances" or theories of consciousness. Most of the
current work is on Global Workspace Theory (GWT), because that stance has the
richest tree and data.

The GWT tree is shaped roughly like:

```text
root stance / root consciousness C_s
  -> features
    -> subfeatures
      -> indicators
        -> expert ordinal ratings for each system
```

Current GWT data geometry:

```text
root stance nodes: 1
internal feature/subfeature nodes: 25
indicators: 50
systems: Human, ELIZA, Chicken, 2024 Leading Chat LLMs

Human: 50 ratings
ELIZA: 50 ratings
Chicken: 93 ratings
LLMs: 186 ratings
```

Human and ELIZA are usually hard reference anchors:

```text
Human C = 0.999
ELIZA C = 0.001
Chicken C is free
LLM C is free
```

The central problem: I do not yet fully understand the tree probability
propagation and the oracle synthetic-validation machinery. I need a skill graph
that fills missing prerequisites and leads to being able to read and reason
about the code/results without hand-holding.

---

## 3. Core DCM Tree Model

Each edge from parent node `u` to child node `v` has two conditional
probabilities:

```text
beta_pres[u,v] = P(child present | parent present)
beta_abs[u,v]  = P(child present | parent absent)
```

The propagated marginal probability at the child is:

```text
q_v = beta_abs[u,v] + q_u * (beta_pres[u,v] - beta_abs[u,v])
```

At the root:

```text
q_root = C_s
```

For an indicator `j`, holding edge betas fixed, the propagated probability is
affine in the root probability:

```text
q_j(C_s) = alpha_j + delta_j * C_s
```

The path slope is:

```text
delta_j = product over path root->j of
          (beta_pres[edge] - beta_abs[edge])
```

`delta_j` is important because it measures how much root-level signal reaches an
indicator. Deep trees attenuate signal because these edge differences multiply.

Local code excerpt:

```python
def propagate_affine_indicator_coefficients_from_draws(...):
    """Propagate q_j(c) = intercept_j + slope_j * c through the tree."""

    def walk(node, ancestor_path, intercept_parent, slope_parent):
        key = node_key(ancestor_path, node["name"])
        beta_pres = beta_pres_by_key[key]
        beta_abs = beta_abs_by_key[key]
        delta = beta_pres - beta_abs
        intercept_child = beta_abs + intercept_parent * delta
        slope_child = slope_parent * delta
```

Source in repo:
`gwt_reference_recovery_analysis.py`, around
`propagate_affine_indicator_coefficients_from_draws`.

---

## 4. Ordinal Observation Layer And Three-State Leaf

The expert ratings are 7-point ordinal ratings. The observation model is an
ordered probit:

```text
latent utility = b_expert + a * state + noise
noise ~ Normal(0, 1)
rating category is determined by ordered cutpoints kappa_1 ... kappa_6
```

For the current synthetic validation, the relevant leaf state is three-state:

```text
m_j in {0, 1, 2}
```

In the exact tree likelihood, the indicator leaf contributes a mixture over
three likelihood terms:

```text
B(beta) = (1-beta)^2 * L_0
        + 2*beta*(1-beta) * L_half
        + beta^2 * L_1
```

where `L_0`, `L_half`, `L_1` are the rating likelihoods under the three possible
leaf states. The code works in log space using `logaddexp` / `logsumexp`.

Local code excerpt:

```python
def leaf_log_B(beta, leaf_lls):
    """Three-state mixture B(beta) = (1-beta)^2 l_0
    + 2 beta (1-beta) l_half + beta^2 l_1.
    """
    beta_c = pt.clip(beta, eps, 1.0 - eps)
    ll0, llh, ll1 = leaf_lls
    log_w0 = 2.0 * pt.log(1.0 - beta_c)
    log_w1 = pt.log(2.0) + pt.log(beta_c) + pt.log(1.0 - beta_c)
    log_w2 = 2.0 * pt.log(beta_c)
    return pt.logaddexp(
        pt.logaddexp(log_w0 + ll0, log_w1 + llh),
        log_w2 + ll1,
    )
```

Source in repo:
`dcm_model_exact_tree.py`, around `leaf_log_B`.

---

## 5. Exact-Tree Bottom-Up Likelihood Propagation

The exact-tree likelihood marginalizes over all discrete latent states in the
tree. It does this by dynamic programming / message passing.

For each node, return two log messages:

```text
message_to_parent(parent=0)
message_to_parent(parent=1)
```

For an indicator, the messages are the leaf likelihood mixture evaluated at
`beta_abs` or `beta_pres`:

```text
if parent absent:  B(beta_abs)
if parent present: B(beta_pres)
```

For an internal node `v`, first combine child messages into:

```text
L_v(z_v=0) = product over children of child_message(parent_state=0)
L_v(z_v=1) = product over children of child_message(parent_state=1)
```

Then integrate over the node's own binary state conditional on its parent:

```text
L_self(parent=1)
  = beta_pres_v * L_v(z_v=1) + (1-beta_pres_v) * L_v(z_v=0)

L_self(parent=0)
  = beta_abs_v * L_v(z_v=1) + (1-beta_abs_v) * L_v(z_v=0)
```

At the root, mix top-level messages with `C_s`:

```text
P(y | C_s, theta)
  = C_s * L_top(parent/root=1) + (1-C_s) * L_top(parent/root=0)
```

Local code excerpt:

```python
def subtree_lls(node, path):
    """Return (log L(z_pa=0), log L(z_pa=1)) for this node."""

    if ntype == "indicator":
        log_L_zpa1 = leaf_log_B(beta_pres, leaf_lls)
        log_L_zpa0 = leaf_log_B(beta_abs, leaf_lls)
        return log_L_zpa0, log_L_zpa1

    log_L_v0 = 0.0
    log_L_v1 = 0.0
    for child in node.get("evidencers", []):
        cL_zpa0, cL_zpa1 = subtree_lls(child, current_path)
        log_L_v0 += cL_zpa0
        log_L_v1 += cL_zpa1

    log_L_zpa1 = logaddexp(
        log(beta_pres) + log_L_v1,
        log(1-beta_pres) + log_L_v0,
    )
    log_L_zpa0 = logaddexp(
        log(beta_abs) + log_L_v1,
        log(1-beta_abs) + log_L_v0,
    )
    return log_L_zpa0, log_L_zpa1
```

Source in repo:
`dcm_model_exact_tree.py`, around `_exact_tree_log_likelihood`.

---

## 6. Oracle Internal-Node Identifiability Audit

The oracle audit is not a PyMC fit. It is an M-closed information check.

Question:

```text
If the exact binary GWT tree were truly the data-generating process, and if the
nuisance parameters were known, would the current rating design let us learn
the hidden internal feature/subfeature states?
```

Mechanism:

1. Simulate exact-tree datasets under known truth.
2. Condition on the true nuisance parameters `theta_star`.
3. Compute `Pr(z_sv = 1 | y, theta_star)` by exact clamped dynamic programming.
4. Score those probabilities against the simulated true binary states.

Important: this is a best-case information check, not evidence that the real
world is an exact binary tree.

The local docstring says:

```text
It is an M-closed oracle check. The DGP is the exact latent-state tree, the
leaf is the three-state indicator model, and nuisance parameters are fixed to
known truth. For each simulated dataset it computes Pr(z_sv = 1 | y, theta*)
by exact clamped dynamic programming, then scores those probabilities against
the simulated latent states.

No PyMC sampling is run.
```

Local code excerpt:

```python
def exact_log_evidence_with_optional_clamp(..., target_key=None, target_state=None):
    """Compute log P(y, optional z_target=state | theta*) exactly."""

def posterior_internal_probability(..., target_key, log_evidence=None):
    log_y = exact_log_evidence_with_optional_clamp(...)
    log_joint1 = exact_log_evidence_with_optional_clamp(
        ..., target_key=target_key, target_state=1
    )
    return exp(log_joint1 - log_y)
```

The audit outputs:

```text
summary.md
node_probabilities.csv
stratified_summary.csv
calibration.csv
overall_summary.csv
```

---

## 7. Full Exact/Exact Fake-Data Recovery

The full fake-data recovery check differs from the oracle audit:

- Data are generated from the exact GWT tree.
- The model fitted is the same exact-tree model.
- Nuisance parameters are learned rather than fixed.
- PyMC sampling is run.

This answers a different question:

```text
When the model is correctly specified, can the full inference workflow recover
root C, edge betas, label betas, path summaries, internal state probabilities,
and observed-scale PPCs?
```

One-seed summary:

```text
Mode: full
Diagnostic status: passed
Divergences: 0
Max R-hat: 1.0000
Min bulk ESS: 3181

Free root C recovery:
Chicken truth 0.250, posterior median 0.113, 94% interval [0.006, 0.439]
LLM truth 0.100, posterior median 0.159, 94% interval [0.010, 0.546]

Label betas:
beta_abs n=18, MAE 0.054, RMSE 0.064, coverage_94 1.000
beta_pres n=18, MAE 0.039, RMSE 0.053, coverage_94 1.000

Edge betas:
beta_abs n=75, MAE 0.062, RMSE 0.071, coverage_94 1.000
beta_pres n=75, MAE 0.051, RMSE 0.065, coverage_94 1.000

Path summaries:
delta_j n=50, MAE 0.054, RMSE 0.062, coverage_94 0.920
q_gap_999_001 n=50, MAE 0.054, RMSE 0.062, coverage_94 0.920

Internal state probability scores:
n=100, mean_relative_entropy_reduction 0.333,
mean_brier 0.111, mean_neg_log_score 0.343

Observed-scale feature-block PPC:
n=196, mean_abs_prop_error 0.116, coverage_94 0.990
```

Interpretation: continuous parameters get MAE/RMSE/coverage. Binary internal
states get probability scoring: Brier score, log score, entropy reduction, and
calibration.

---

## 8. Oracle Audit Results To Understand

Production-median nuisance truth, 50 simulations:

```text
Internal nodes per system: 25
Total scored states: 5000 = 50 simulations * 25 nodes * 4 systems

Overall:
mean_relative_entropy_reduction 0.439
mean_brier 0.110
mean_neg_log_score 0.348

By depth:
depth 1: relative entropy reduction 0.443
depth 2: relative entropy reduction 0.486
depth 3: relative entropy reduction 0.048

By fanout:
fanout 0: relative entropy reduction 0.048
fanout 5: relative entropy reduction 0.535
fanout 6: relative entropy reduction 0.831

By subtree rating count:
0 ratings: relative entropy reduction 0.048
1-2 ratings: 0.356
3-5 ratings: 0.506
6-10 ratings: 0.530
11-25 ratings: 0.524

Calibration examples:
bin [0.0, 0.1): mean p 0.030, empirical z rate 0.035
bin [0.9, 1.0): mean p 0.967, empirical z rate 0.966
```

Paper-mean tree-transmission stress truth is weaker:

```text
Overall:
mean_relative_entropy_reduction 0.263
mean_brier 0.148
mean_neg_log_score 0.454

By depth:
depth 1: 0.262
depth 2: 0.296
depth 3: 0.010

By fanout:
fanout 0: 0.010
fanout 6: 0.634
```

The key interpretation:

- Deep / fanout-0 / zero-descendant-rating nodes are essentially not learned
  from data.
- Broad connected nodes with many descendant ratings can be learned.
- Internal states should be reported as posterior probabilities, not hard
  recovered labels.
- Good calibration means the model's "I don't know" is honest, not broken.
- The oracle is better than full recovery because it knows the true nuisance
  parameters. The gap is the cost of estimating nuisance.

---

## 9. Validation Taxonomy Used In The Repo

The repo's validation spec separates:

1. Correctness:
   Does the simulator and exact-tree likelihood compute the intended
   probability model? Includes toy recovery, brute-force enumeration, dynamic
   programming identities, no-transmission invariance, finite probabilities.

2. M-closed calibration:
   Under exact latent-tree DGP, do posterior intervals and probabilities behave
   as calibrated Bayesian summaries of the same model? Includes root C, path
   summaries, and internal-node posterior probability calibration.

3. Practical identifiability:
   Do the observed ratings contain enough information to learn scientific
   quantities, or are posterior summaries mostly prior-regularized? For internal
   nodes, evaluate `Pr(z_sv=1 | y)`, not hard labels.

4. M-open robustness:
   When the real DGP lies outside the exact tree, what pseudo-true quantities
   does the model learn? Deferred DGPs include rater shifts, alternative ordinal
   layers, continuous internal truth fit by binary exact tree, wrong tree
   dependence, and missing sibling dependence.

5. Scientific usefulness:
   Do outputs support the decisions and interpretations that matter? Primary
   estimands include root `C_s`, threshold events, internal probabilities, path
   summaries, and observed-scale summaries.

---

## 10. Skill Inventory Already Identified

Please use this as a node bank. You may reorganize, split, merge, or add missing
prerequisites.

Probability foundations:

- Events, complements, conditional probability, independence.
- Product rule, sum rule, law of total probability.
- Bayes rule as probability reweighting.
- Bernoulli, binomial, categorical, beta, normal distributions.
- Expected value, variance, simulation, Monte Carlo error.
- Odds, log-odds, logits, probits.
- Working in log probability space.

Tree and graph foundations:

- Trees, roots, leaves, parents, children, descendants.
- Depth, fanout, subtree size.
- Directed graphical models / Bayesian networks.
- Conditional independence in a tree.
- Top-down generative sampling.
- Bottom-up dynamic programming.
- Message passing / belief propagation on trees.
- Clamping a latent variable and recomputing evidence.
- Why exact inference is cheap on trees but hard on general graphs.

DCM tree-specific skills:

- Interpret `C_s` as root probability, not a sampled root label.
- Interpret `z_sv` as internal binary state.
- Interpret `beta_pres` and `beta_abs`.
- Derive `q_child = beta_abs + q_parent * (beta_pres - beta_abs)`.
- Explain why path slopes multiply across depth.
- Explain why deep paths attenuate root signal.
- Distinguish `q_j`, `delta_j`, and `q_gap_999_001`.
- Explain why depth/fanout/subtree rating count control identifiability.

Ordinal observation layer:

- Ordinal ratings as ordered categories.
- Latent utility model behind ordered probit.
- Cutpoints and why they must be ordered.
- Discrimination/slope parameter `a`.
- Expert bias / scale heterogeneity.
- Likelihood of ratings conditional on latent state.
- Three-state leaf mixture: weights for `m=0,1,2`.
- Marginalizing discrete latent states instead of sampling them with NUTS.

Bayesian inference:

- Prior, likelihood, posterior, posterior predictive.
- Latent variables vs nuisance parameters vs estimands.
- Marginal likelihood for a subtree.
- `pm.Potential` as a custom likelihood contribution.
- HMC/NUTS at a conceptual level.
- R-hat, ESS, divergences, target accept.
- Posterior intervals and why they differ from probability scores.

Synthetic validation:

- DGP vs fit model.
- M-closed validation: data generated from the same model being fit.
- M-open robustness: real-world deviations from the model.
- Oracle audit: nuisance known/fixed.
- Full recovery: nuisance learned from fake data.
- Correctness checks: enumeration, dynamic-programming identity checks.
- Fake-data recovery vs SBC vs PPC.
- Coverage, RMSE, MAE, calibration.
- Why binary latent states get Brier/log-score/entropy, not interval coverage.

Identifiability and information:

- Practical identifiability vs philosophical truth of a model.
- Prior-regularized posterior vs data-driven posterior.
- Entropy and relative entropy reduction.
- KL separation curves.
- Proper scoring rules: Brier score and log score.
- Probability calibration bins.
- Why zero descendant ratings imply little learning.
- Why fanout helps: more descendant evidence enters the subtree message.

Code/navigation skills:

- Read `summary.md`, `config.json`, `truth.json`, and CSV outputs.
- Trace run labels: `dgp`, `fit`, `leaf`, `nuisance_truth`, `design`.
- Use pandas to group by depth/fanout/subtree count.
- Read ArviZ `InferenceData` / NetCDF posterior draws.
- Map code objects to math notation.
- Reproduce a tiny tree by hand before trusting the full GWT tree.

---

## 11. Suggested Resource Pool

Please map resources to individual nodes, not just broad clusters. Use better
resources if you know them.

Probability:

- Blitzstein and Hwang, *Introduction to Probability*.
- Harvard Stat 110 lectures.

Bayesian modeling:

- McElreath, *Statistical Rethinking*.
- Gelman et al., *Bayesian Data Analysis*.

Graphical models:

- Bishop, *Pattern Recognition and Machine Learning*, chapter 8.
- Koller and Friedman, *Probabilistic Graphical Models*, for deeper study.

Ordinal models:

- Agresti, *Analysis of Ordinal Categorical Data*.
- McElreath's ordinal regression material.

Bayesian workflow and computation:

- Betancourt's HMC and Bayesian workflow case studies.
- Talts et al. on simulation-based calibration.
- PyMC documentation on custom likelihoods / `pm.Potential`.
- ArviZ documentation on diagnostics and posterior summaries.

---

## 12. Desired First Learning Frontier

I suspect the first frontier should include only the smallest set needed to
understand the two main mechanisms:

1. Forward probability propagation:
   `q_child = beta_abs + q_parent * (beta_pres - beta_abs)`.

2. Bottom-up likelihood propagation:
   leaf likelihoods produce messages; internal nodes combine child evidence;
   the root mixes with `C_s`.

3. Oracle audit:
   fixed nuisance, exact simulated truth, clamped dynamic programming, score
   posterior probabilities against simulated binary states.

4. Internal-state scoring:
   entropy reduction, Brier score, log score, calibration bins.

Please either confirm this frontier or propose a better one.

---

## 13. Self-Assessment Will Be Added Later

After you produce the initial graph, I will mark each node myself:

```text
4 = can derive/explain/use without notes
3 = understand but slow
2 = recognize but cannot use fluently
1 = vague familiarity
0 = missing
unknown = not assessed yet
```

Please structure the output so this self-assessment can be added cleanly before
I pass the graph to a Codex instance in the `ryan-graph` repo.

