\documentclass[11pt]{article}

% ---------- Packages ----------
\usepackage[margin=1in]{geometry}
\usepackage{microtype}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage[nameinlink,noabbrev]{cleveref}
\usepackage{natbib}

% ---------- Metadata ----------
\title{Anthropic / RP Briefing: Changes Since the Public DCM}
\author{Ryan Kelly}
\date{\today}

% ---------- Convenience macros ----------
\newcommand{\DCM}{\mathrm{DCM}}
\newcommand{\GWT}{\mathrm{GWT}}
\newcommand{\Cs}{C_s}
\newcommand{\qj}{q_j}
\newcommand{\todo}[1]{\textcolor{red}{[TODO: #1]}}
\newcommand{\statusshipped}{\noindent\textbf{Status:}~\textcolor{teal}{Shipped.}\par\smallskip}
\newcommand{\statusreview}{\noindent\textbf{Status:}~\textcolor{orange}{Shipped, under review.}\par\smallskip}
\newcommand{\statusexploratory}{\noindent\textbf{Status:}~\textcolor{purple}{Exploratory.}\par\smallskip}

\begin{document}

\maketitle

% ============================================================
\section{Executive summary}
% Leave blank until the end.


% ============================================================
\section{Baseline public DCM}

\subsection{What the initial model did}
\label{subsec:baseline-model}

The published DCM \citep{shiller2026initial} models the probability of
consciousness for a system $s$ on a stance $\sigma$ as a binary latent
$\Cs \in \{0, 1\}$ with a stance-level prior $\Cs \mid \sigma \sim
\mathrm{Bernoulli}(p_\sigma)$, where $p_\sigma \sim
\mathrm{Beta}(\alpha_0, \beta_0)$ defaults to $\mathrm{Beta}(1, 5)$ --
a baseline expectation that consciousness is unlikely. Below $\Cs$,
each stance carries a hand-coded tree of binary features,
subfeatures, and indicators; every parent--child edge $v \to u$ is
parameterised by two conditional probabilities
$\beta^{\mathrm{pres}}_u = \mathrm{Pr}(z_u = 1 \mid z_v = 1)$ and
$\beta^{\mathrm{abs}}_u = \mathrm{Pr}(z_u = 1 \mid z_v = 0)$, both
given independent Beta priors whose means follow the modeller's
support (sensitivity) and demandingness (specificity) labels for that
edge (\citep{shiller2026initial} Tables 2--3). The child's latent
state is then $z_u \mid z_v \sim \mathrm{Bernoulli}(q_u(z_v))$ with
$q_u(1) = \beta^{\mathrm{pres}}_u$ and $q_u(0) = \beta^{\mathrm{abs}}_u$,
so the joint generative process threads from $\Cs$ at the root,
through a chain of conditional Bernoulli transmissions, down to each
indicator $j$ at the leaves.

Expert opinion enters as a probability $\bar p_{sj} \in [0,1]$ per
(system $s$, indicator $j$) cell, computed as the cross-rater mean of
the raw responses \citep[\S~Generating indicator values from expert
surveys]{shiller2026initial}. The cell mean is converted to a binary
observation $y_{sj} \in \{0,1\}$ at the start of every "model run" by
drawing $y_{sj} \sim \mathrm{Bernoulli}(\bar p_{sj})$ once, and the
indicator's latent state is observed against that draw inside the
run. Each stance and each system is fit independently as a separate
PyMC model: within a single run, PyMC's default sampler assigns a
Metropolis-style kernel to the discrete latents $\{\Cs, z_v, z_j\}$
and NUTS to the continuous edge probabilities $\boldsymbol\beta$, and
many such runs are repeated with fresh $y_{sj}$ draws to recover
indicator-level uncertainty by averaging the per-run posterior means.
The reported $\Cs$ probability and its uncertainty interval are taken
from the across-run distribution of those per-run means
\citep[Figures~1--3]{shiller2026initial}.

\subsection{Where expert observations entered}
\label{subsec:baseline-observations}

Observations enter the model only at the indicator (leaf) level: no
internal feature or subfeature node has a directly observed value.
The input quantity is the per-cell mean $\bar p_{sj}$ defined above,
and the likelihood contribution at indicator $j$ is the single
Bernoulli factor obtained by clamping the latent state $z_j$ to the
per-run binary draw $y_{sj}$. No rater identity, individual rating,
or full rating distribution is preserved beyond the per-cell mean,
and no rating-distribution information enters the likelihood -- only
the binary collapse does.

\subsection{What was lossy / under-instrumented}
\label{subsec:baseline-limitations}

Four aspects of the published inference are lossy or hard to
diagnose, and the SPAR changes are organised around them:

\begin{itemize}[leftmargin=*]
\item \textbf{Stochastic binary collapse of expert input.} The
$\mathrm{Bernoulli}(\bar p_{sj})$ draw at run start destroys all
within-cell information beyond the mean -- rater identity, individual
ratings, and the shape of the rating distribution are discarded
before the likelihood is evaluated. Addressed in
\S~\ref{subsec:raw-ratings} and \S~\ref{subsec:ordinal-probit}.
\item \textbf{Sampled rather than marginalised latent tree.} The
binary latents $\{\Cs, z_v, z_j\}$ are sampled by a discrete
Metropolis kernel rather than removed from the likelihood
analytically; mixing this with NUTS on the continuous parameters
makes $\hat R$, ESS, and divergence diagnostics hard to interpret
jointly, and forces each (stance, system) pair to be a separate
\texttt{pm.sample} call with no cross-system anchoring inside the
inference. Addressed in \S~\ref{subsec:tree-marginalisation}.
\item \textbf{No rating-level posterior predictive check.} Because
the likelihood lives at the level of $y_{sj} \in \{0,1\}$ rather
than at the rating distribution, observation-layer misfit cannot
be separated from tree-prior misfit. Addressed by rating-level PPCs
in \S~\ref{sec:validation}.
\item \textbf{Outer-loop variance dominates the reported uncertainty.}
The across-run dispersion that the paper reports as the consciousness
probability interval is principally Bernoulli-collapse variance over
$y$, not posterior curvature on $\Cs$. There is no native facility
for asking whether a given $\Cs$ posterior is informed by data or
inherited from the prior. Addressed by prior sweeps and posterior
contraction diagnostics in \S~\ref{sec:findings}.
\end{itemize}


% ============================================================
\section{SPAR changes to the model}

\subsection{Raw 7-point expert ratings}
\label{subsec:raw-ratings}
\statusshipped

The original DCM elicits expert judgements about each indicator as a
probability in $[0,1]$ -- experts state how likely the indicator is to be
present in a given system, and these probabilities are then averaged across
respondents to produce a single mean score per (system, indicator) cell
\citep[\S~Generating indicator values]{shiller2026initial}. Inside the
inference loop, that mean score is collapsed stochastically to a binary
$\{0,1\}$ indicator via a Bernoulli draw at the start of every model run, and
hundreds of runs are then averaged to recapture the uncertainty that the
collapse step destroys (see \texttt{dcm\_model.py} in the legacy code, lines
$\approx 308$--$310$: \texttt{binary\_value = 0 if float(obs\_average) <
random.random() else 1}). This creates two related problems: each fitted
posterior is conditional on one sampled realisation of the indicators, and most
of the uncertainty propagation happens outside the main inference problem,
through repeated outer-loop simulation rather than through the posterior
\citep[\S~Indicator independence]{shiller2026initial}.

The survey data used in this work is collected on a 7-point Likert scale
\textit{(strongly absent, absent, somewhat absent, unsure, somewhat present,
present, strongly present)}, matching the future data format the project is
expected to standardise on. Each rating is retained at the level of an
individual \emph{(rater, system, indicator)} triple: ratings are \emph{not}
averaged across raters, and they are not collapsed to a binary value. The
current dataset covers 16 expert surveys for 2024 chat LLMs (6 full, 10
partial), 2 surveys for chickens, and one survey each for Human and ELIZA
authored by the modelling team. Of the six distinct raters, five rate exactly
one system; Rater_B is the only cross-system rater, contributing
judgements on LLM, Human, and ELIZA.\footnote{The fact that 5 of 6 raters are
single-system is a structural feature of the data that has consequences for
identifiability of any per-rater parameter; we return to this when discussing
the rater location shift $b_e$ and again in the discussion of survey design.}

Keeping the raw ordinal responses serves two purposes. First, it preserves the
information in the full rating distribution, not just its mean -- two indicators
with the same mean rating but very different category histograms encode
different evidential signals. Second, it lets the rating likelihood be
\emph{specified} rather than \emph{simulated}: the tree-implied probability
that an indicator is present can be combined with the rating distribution
directly inside the posterior, removing the outer-loop Bernoulli simulation
entirely. Both motivations are addressed by the ordinal-probit observation
model in \S~\ref{subsec:ordinal-probit}, which Arvo suggested adapting from the
signal-detection literature \citep{decarlo1998signal}.

\subsection{Ordinal-probit observation model}
\label{subsec:ordinal-probit}
\statusshipped

The observation layer replaces the binary-collapse step of the original
DCM with an ordinal-probit (latent-Gaussian threshold) likelihood. For
rater $e$ and indicator $j$, we posit a latent continuous evidence
variable
\begin{equation}
    s_{ej} \;=\; b_e \,+\, a \, z_j \,+\, \varepsilon_{ej},
    \qquad \varepsilon_{ej} \sim \mathcal{N}(0,\,1),
    \label{eq:latent-signal}
\end{equation}
where $z_j \in \{0,1\}$ is the latent presence/absence of the indicator
(produced by the tree-prior in the system being modelled), $a > 0$ is a
shared discrimination parameter controlling how far apart the two latent
locations sit, and $b_e$ is a rater-specific location shift (treated in
\S~\ref{subsec:rater-structure}). The observed rating
$r_{ej} \in \{1,\dots,K\}$ ($K=7$) is recovered by ordered thresholding
against cutpoints $\boldsymbol{\kappa} = (\kappa_1 < \dots < \kappa_{K-1})$:
\begin{equation}
    r_{ej} = k \iff \kappa_{k-1} \,<\, s_{ej} \,\le\, \kappa_k,
    \label{eq:ordinal-thresholds}
\end{equation}
with $\kappa_0 = -\infty$ and $\kappa_K = +\infty$. This is the standard
latent-variable construction for ordinal regression
\citep{decarlo1998signal}.

The DCM tree for a given stance already supplies an implicit probability
\begin{equation}
    q_{j} \;=\; \mathrm{Pr}(z_j = 1 \mid \text{tree},\,C_s,\,\boldsymbol{\beta}),
    \label{eq:q-tree-implied}
\end{equation}
i.e.\ the marginal probability that indicator $j$ is present in system
$s$, marginalising over the latent state of every node on the path from
the root $C_s$ down to $j$ given the support/demandingness transmission
parameters $\boldsymbol{\beta}$. Rather than sampling $z_j$ explicitly
and conditioning on the binary realisation as the legacy model does, we
marginalise $z_j$ analytically. The rating likelihood is then the
two-component mixture
\begin{equation}
    \mathrm{Pr}(r_{ej} = k \mid \theta)
    \;=\; q_j \cdot P_{\mathrm{OP}}(k \mid b_e + a,\,\boldsymbol{\kappa})
    \;+\; (1 - q_j) \cdot P_{\mathrm{OP}}(k \mid b_e,\,\boldsymbol{\kappa}),
    \label{eq:rating-mixture}
\end{equation}
where
$P_{\mathrm{OP}}(k \mid \eta,\,\boldsymbol{\kappa}) =
\Phi(\kappa_k - \eta) - \Phi(\kappa_{k-1} - \eta)$
is the ordered-probit category probability and $\Phi$ is the standard
normal CDF. The mixture form makes the contribution of $q_j$ to the
likelihood explicit: the data shift mass between the
\emph{present-conditional} category distribution (centred at $b_e + a$)
and the \emph{absent-conditional} distribution (centred at $b_e$),
and the tree-prior parameters that govern $q_j$ are updated jointly with
the observation-layer parameters.

The shipped observation-layer parameters are pooled and shared as
follows. The discrimination $a$ and the cutpoints $\boldsymbol{\kappa}$
are \emph{shared across all systems, indicators, and raters}, with
priors $a \sim \mathrm{HalfNormal}(\sigma_a)$ and an ordered
$\boldsymbol{\kappa} \sim \mathrm{Normal}(0, \sigma_\kappa)$ vector
($\sigma_a = \sigma_\kappa = 2.0$ in the production fit); the rater
shifts $\mathbf{b}$ are specified in \S~\ref{subsec:rater-structure}.
The system-level parameters that the observation layer is meant to
recover -- the root probabilities of consciousness $\Cs$ and the tree
transmission parameters $\boldsymbol{\beta}$ -- enter only through $q_j$
in \eqref{eq:rating-mixture}, so observation-layer nuisance is
structurally separated from the consciousness signal. The
implementation lives in
\texttt{dcm\_model.py:\,build\_ordinal\_observation\_layer} (line~619),
controlled by \texttt{N\_CATEGORIES=7},
\texttt{KAPPA\_PRIOR\_SIGMA=2.0}, and
\texttt{A\_PRIOR\_SIGMA=2.0}.

Three immediate consequences are worth flagging. First, the analytical
marginalisation in \eqref{eq:rating-mixture} eliminates the outer
Bernoulli simulation loop: NUTS samples the full joint posterior over
$(\Cs, \boldsymbol{\beta}, a, \boldsymbol{\kappa}, \mathbf{b})$ in a
single inference run. Second, the implied rating distributions at
$z_j=0$ and $z_j=1$ are now \emph{inferred} from the data rather than
hard-coded by the Bernoulli draw, and can be inspected at the posterior
mean to check that the latent-presence and latent-absence categories
separate as expected (Figure~\todo{ref to absent-vs-present rating
distribution plot}). Third, posterior predictive checks can now be
performed at the rating-distribution level, which lets us separate
observation-layer misfit (the mixture in \eqref{eq:rating-mixture} fits
the rating histogram poorly) from tree-prior misfit (the tree
mis-allocates mass to particular $q_j$ values), a separation that was
not possible under the binary-collapse procedure.

\subsection{Posterior inference via marginalisation of the latent tree}
\label{subsec:tree-marginalisation}
\statusshipped

The architectural shift behind nearly every other change in this
section is the move from \emph{sampling} the discrete latents of the
DCM to \emph{marginalising} them analytically. The published
implementation (\S~\ref{subsec:baseline-model}) treats every binary
variable -- the stance state $\Cs$, every internal feature/subfeature
$z_v$, and every indicator $z_j$ -- as an explicit PyMC random
variable and samples it with a Metropolis kernel inside the same
chain that runs NUTS on the continuous edge probabilities
$\boldsymbol\beta$, conditional on a once-drawn binary indicator
vector $\mathbf y$ (\S~\ref{subsec:raw-ratings}); reported posterior
quantities are an outer-loop average of those per-run chain summaries.
The SPAR model instead removes every \emph{unobserved} discrete
latent from the joint distribution by closed-form summation over its
support, leaving a likelihood whose only random unknowns are the
continuous parameters $\theta = (\boldsymbol\beta, a,
\boldsymbol\kappa, \mathbf b)$ together with any soft-anchored
continuous $\Cs$ on free systems. NUTS then samples that continuous
posterior in a single run; system-level $\Cs$ probabilities and any
other latent-node posterior of interest (e.g.\ $\Pr(z_v = 1 \mid
\mathbf r)$ for explainability) are recovered as posterior quantities
of $\theta$ via a complementary downward sweep.

\paragraph{The marginal likelihood as a bottom-up recursion.}
For a stance tree rooted at $\Cs$ with top-level children
$\mathcal V_0$, the marginal data likelihood given $\theta$ is
\begin{align}
    P(\mathbf r \mid \theta)
    \;&=\; \sum_{c \in \{0,1\}} \mathrm{Pr}(\Cs = c)
    \prod_{v \in \mathcal V_0} M_v(c), \label{eq:tree-likelihood} \\[2pt]
    M_v(z)
    \;&=\; \sum_{z_v \in \{0,1\}} \!\! \mathrm{Pr}(z_v \mid z, \boldsymbol\beta_v) \cdot
    \begin{cases}
        \displaystyle\prod_{c \in \mathcal C_v} M_c(z_v)
            & \text{($v$ internal),} \\[6pt]
        \displaystyle\prod_{e \in E_v} \mathrm{Pr}(r_{ev} \mid z_v, \theta)
            & \text{($v$ indicator leaf),}
    \end{cases}
    \label{eq:tree-recursion}
\end{align}
where $\mathrm{Pr}(z_v = 1 \mid z, \boldsymbol\beta_v) =
\beta^{\mathrm{pres}}_v$ if $z = 1$ and $\beta^{\mathrm{abs}}_v$ if
$z = 0$. The leaf factor $\Pr(r_{ev} \mid z_v, \theta)$ is the
ordered-probit rating likelihood from \eqref{eq:rating-mixture}
(\S~\ref{subsec:ordinal-probit}) evaluated at the latent-state
parameterisation chosen for the indicator (binary
$z_j \in \{0, 1\}$, or three-state $m_j \in \{0, 1, 2\}$ from
\S~\ref{subsec:three-state}). Equations
\eqref{eq:tree-likelihood}--\eqref{eq:tree-recursion} are a standard
sum--product (or "belief propagation") recursion on a polytree
\citep{pearl1988probabilistic, koller2009probabilistic}; because the
DCM tree has no cross-edges, the recursion is \emph{exact} and
introduces no approximation. Each evaluation costs
$O(|\mathcal V|)$ time, and the entire expression is unrolled at
model-build time into a PyTensor graph via Python recursion and
\texttt{pt.logaddexp} -- no \texttt{pytensor.scan}, no custom Op --
so HMC/NUTS sees analytic gradients in $\theta$ throughout
(\texttt{dcm\_model\_exact\_tree.py:MultiSystemExactTreeBuilder}).

\paragraph{What this buys.}
\emph{Inferentially}: a single posterior over $\theta$ (plus any free
$\Cs$) replaces the mixed Metropolis-plus-outer-loop construction,
so $\hat R$, ESS, and divergence diagnostics apply cleanly across
the entire model. Joint multi-system fits become natural -- the same
shared $(a, \boldsymbol\kappa, \mathbf b, \beta\text{-pool})$ feeds
each system's tree marginalisation call, and cross-system anchors
(Human, ELIZA) transmit information to free systems through those
shared parameters rather than only through hand-set $\Cs$ priors.
\emph{Mathematically}: marginalisation is over the
\emph{joint} distribution of latent indicators within a tree, so the
likelihood respects sibling dependence among indicators sharing a
parent; the legacy form, by contrast, factorises each indicator's
likelihood independently against its own marginal $q_j$ and ignores
the joint dependence those indicators inherit from their shared
parent (the disagreement at finite data is measured in
\texttt{composite\_vs\_exact\_diagnostic.py}). \emph{Diagnostically}:
because the likelihood now lives at the rating-distribution level
and respects the tree's dependence structure, posterior predictive
checks can be performed at per-rating, per-indicator, and per-system
granularity and attributed to specific layers of the model
(\S~\ref{sec:validation}).

\paragraph{Caveats and scope.}
Marginalisation applies only to \emph{unobserved} discrete latents;
hard-anchored $\Cs$ (Human, ELIZA in the production fit) enter the
recursion as fixed \texttt{pt.constant} values and soft-anchored
$\Cs$ on free systems enter as continuous Beta variates rather than
as marginalised binaries. The legacy "composite" form in which each
indicator's likelihood is mixed independently against its own
marginal $q_j$ is retained in \texttt{MultiSystemModelBuilder} for
ablation, and a NumPy reference implementation in
\texttt{composite\_vs\_exact\_diagnostic.py} verifies that the
marginalised likelihood here matches the standalone DP to numerical
precision per draw. Sampling wall time on the current GWT dataset is
comparable to the legacy mixed-sampler model (the per-draw DP is
cheap because the tree is small), but $\hat R$ and ESS on the
continuous parameters are substantially better because the
discrete-Metropolis bottleneck is gone.

\subsection{Reference-system anchoring}
\label{subsec:reference-systems}
\statusreview

The published DCM fits each (system, stance) pair as an independent
PyMC model with the system's root $\Cs$ a free Bernoulli at the
stance-level prior $\mathrm{Beta}(1, 5)$ \citep{shiller2026initial}.
The SPAR change is to fit all four target systems jointly in a single
inference -- sharing the observation layer $(a, \boldsymbol{\kappa},
\mathbf{b})$ and the tree-transmission parameters $\boldsymbol{\beta}$
across systems through the tree-marginalisation recursion of
\S~\ref{subsec:tree-marginalisation} -- and to designate Human and
ELIZA as reference \emph{anchors} whose $\Cs$ is fixed close to a
known value rather than left free. Each system's $\Cs$ enters the
joint model through one of three branches
(\texttt{dcm\_model\_exact\_tree.py:69--89}):
\begin{itemize}[leftmargin=*]
\item \textbf{Hard anchor.} The system's $\Cs$ is supplied as a
\texttt{c\_fixed} float and enters the recursion as a
\texttt{pt.constant}; the production
\texttt{ANCHORED\_SYSTEM\_CONFIGS} sets Human $= 0.999$ and
ELIZA $= 0.001$.
\item \textbf{Soft anchor.} \texttt{SOFT\_REFERENCE\_ANCHORS} maps a
system to a tight Beta prior (e.g.\ $\mathrm{Beta}(50, 1)$ for
Human, $\mathrm{Beta}(1, 50)$ for ELIZA -- 95\% CI
$\approx [0.93, 0.999]$ and $[0.001, 0.07]$ respectively); the
prior overrides any \texttt{c\_fixed}, so the same system list
can be flipped between hard and soft for ablation.
\item \textbf{Free.} No anchor: $\Cs \sim \mathrm{Beta}(1, 5)$, the
default stance prior; used for Chicken and the 2024 LLMs.
\end{itemize}

Anchoring is structurally load-bearing in the joint fit: the shared
$(a, \boldsymbol{\kappa}, \mathbf{b}, \boldsymbol{\beta})$ are
calibrated against the anchored systems' ratings, and that
calibration is what lets cross-system rater contrasts (chiefly
Rater_B's ratings across Human / ELIZA / LLM) feed information into
the free Chicken and LLM posteriors. A no-anchor ablation -- every
system free, with per-system $\boldsymbol{\kappa}$ -- collapses every
$\Cs$ posterior to the prior mean (\S~\ref{sec:findings}),
confirming that the per-system rating distributions cannot be
separated from the prior without the anchored reference points.
Status is under review at the hard-vs-soft level rather than the
anchored-vs-free level: production runs currently use hard anchors;
the soft-anchor infrastructure is the more defensible long-term
default ("anchors are priors, not constants") and was used in the
exploratory mixture-leaf branch (\S~\ref{subsec:three-state}), but
has not yet been adopted as the production setting. The diagnostic
configurations \texttt{HUMAN\_FREE\_SYSTEM\_CONFIGS} and
\texttt{ELIZA\_FREE\_SYSTEM\_CONFIGS} in
\texttt{gwt\_reference\_recovery\_analysis.py} release one anchor at
a time, so the other anchor alone has to pull the released system's
$\Cs$ posterior back to its expected location -- the resulting
one-anchor recovery curves are the main quantitative check on how
informative each reference is in isolation (\S~\ref{sec:validation}).

\subsection{Three-state indicator leaves}
\label{subsec:three-state}
\statusreview

The binary latent indicator $z_j \in \{0, 1\}$ used in
\S~\ref{subsec:ordinal-probit} forces the ordinal-probit likelihood to
collapse the tree-implied probability $q_j \in (0,1)$ onto just two
latent rating locations -- $\eta = b_e$ when the indicator is absent and
$\eta = b_e + a$ when present -- regardless of how strongly the tree
favours one outcome over the other. The three-state leaf relaxes this
by drawing a per-indicator latent count $m_j \sim \mathrm{Binomial}(2,
q_j)$ and setting $z_j = m_j / 2 \in \{0, \tfrac{1}{2}, 1\}$ with
emission centres $\eta_m \in \{0,\, a/2,\, a\}$, so a moderate $q_j$
can place rating mass coherently around the middle category rather
than being forced to mix between the two extremes. The marginalisation
is performed at the indicator level rather than per rating, since the
latent $m_j$ is shared across all ratings of indicator $j$:
\begin{equation}
    \log \mathrm{Pr}(r_{1j},\dots,r_{n_j j} \mid q_j, \theta)
    = \operatorname{logsumexp}_{m \in \{0,1,2\}}
    \!\left[\,
        \log w_m(q_j) +
        \textstyle\sum_{i=1}^{n_j}\!\log P_{\mathrm{OP}}(r_{ij} \mid \eta_m, \theta)
    \,\right],
    \label{eq:three-state-marginal}
\end{equation}
with binomial weights $w_0 = (1-q_j)^2$, $w_1 = 2 q_j (1-q_j)$,
$w_2 = q_j^2$. The implementation lives at
\texttt{dcm\_model.py:\,three\_state\_log\_weights} (line~461) and
\texttt{pt\_three\_state\_ll\_terms} (line~470), selected via
\texttt{INDICATOR\_STATE\_MODEL = "three\_state"} (the
\texttt{"binary"} branch is retained for ablation). It is more
honestly read as an observation-layer device that smooths the latent
emission across $q_j$ values than as a literal three-valued ontology
for the indicators themselves.

The status is under review because two alternative leaf models were
prototyped in May 2026 to test whether the binary-leaf compression
was actually the binding constraint on $\Cs$ identification, and
the answer was negative for both. The \texttt{direct\_q} leaf
(branch \texttt{leaf-direct-q}; $\eta_{je} = a \cdot \tilde q_j + b_e$,
no latent state) lifts analytical Fisher information about $\Cs$ by
roughly $3\times$ at the central $\boldsymbol{\beta}$ prior but moves
empirical posterior contraction by less than 0.01 on both Chicken
and LLM under synthetic seed-06. A 2-component continuous-$z$
mixture (branch \texttt{leaf-mixture-shared-h}; centred endpoints
$\eta_h = b_e \pm \delta/2$ with shared rater shift, marginalised in
the same per-indicator form as \eqref{eq:three-state-marginal})
delivers the first real-data recovery of Chicken~$>$~LLM ordering
but with $\hat R \approx 1.73$ on $(\delta, \kappa)$, indicating
multi-modality in the observation layer; under the oracle-clamp
ablation the mixture's $\Cs$ contraction is $1.16$ -- broader than
the prior -- versus three-state's $0.87$ at the same clamp. Taken
together, three-state remains the best-tested production leaf, but
no leaf change has cleared the contraction floor of $\approx 0.87$
even with all other nuisance parameters pinned to truth, which
indicates that the binding constraint on $\Cs$ identification is
upstream of the leaf
\citep[][\texttt{findings\_mixture.md},
\texttt{findings\_direct\_q.md}]{}. The choice between three-state
and a continuous-$z$ alternative is therefore still open; the
present model ships three-state because it is the best-validated
option \emph{and} because its
$\mathrm{Binomial}(2, q_j)$ weights have a clean interpretation,
not because the leaf-flexibility motivation has been confirmed.

\subsection{Rater-specific structure}
\label{subsec:rater-structure}
\statusreview

The only rater-specific parameter in the production model is the
additive location shift $b_e$ entering the latent signal
\eqref{eq:latent-signal}. A single rater is anchored at $b_e = 0$ to
remove the global location degeneracy with the cutpoints
$\boldsymbol{\kappa}$, and the remaining shifts are given a weakly
informative prior $b_e \sim \mathrm{Normal}(0, \sigma_b)$ with
$\sigma_b = 2.0$ (config flag \texttt{USE\_EXPERT\_SHIFTS=True},
\texttt{EXPERT\_SHIFT\_SIGMA=2.0}; the
\texttt{USE\_EXPERT\_SHIFTS=False} branch sets all $b_e = 0$ and is
retained for ablation). The shift absorbs each rater's central
tendency on the 7-point scale and is the minimum rater structure
needed to keep the cutpoints from being confounded with rater-mean
differences. Status is \emph{under review} because the present rater
panel -- five of six raters rate exactly one system, with Rater_B
Shiller the only cross-system rater -- means that on Human, ELIZA,
and Chicken any per-rater location is co-identified with the
system-level parameters those raters' ratings are meant to inform.
The shipped $b_e \sim \mathrm{Normal}(0, 2)$ prior is loose enough
that an unlucky single-rater system could in principle absorb $\Cs$
signal into $b_e$; we have not seen this empirically on the anchored
fits, but it is the structural reason any \emph{further} per-rater
flexibility (below) has been kept off by default.

Three extensions to per-rater structure were prototyped but are not
in the shipped model. Hierarchical per-rater cutpoints
($\boldsymbol{\kappa}_e$ via positive log-gap offsets around a
global $\log\bar{\boldsymbol{g}}$, flag
\texttt{USE\_HIERARCHICAL\_EXPERT\_CUTPOINTS}) and per-rater scales
($\sigma_e$ on the latent error, flag \texttt{USE\_EXPERT\_SCALES})
both target the bimodal-mass deficiency at extreme $\tilde q$
documented in \S~\ref{sec:findings}; the implementations are
mutually exclusive with each other and with
\texttt{USE\_EXPERT\_SHIFTS} (see lines~640--655 of
\texttt{dcm\_model.py}), and neither has been validated end-to-end
under the current rater panel. Per-rater discrimination $d_e$
(scaling the latent presence shift, $a \cdot d_e \cdot z_j$) was
not implemented but was investigated analytically at Arvo's
request: under oracle nuisance, the diagonal Fisher information at
$d_e = 1$ is $93\text{--}115$~nats$^2$ for single-system raters and
$\sim 396$~nats$^2$ for Rater_B, well above the $\sim 1$~nat$^2$
collapse threshold, but this only measures \emph{local}
informativeness with the global $a$ pinned at truth -- the actually
binding question is joint identifiability of $\{d_e\}$ with $a$
along the $a \cdot d_e$ confounding direction, which is constrained
within-system only on LLM (4 raters) and Chicken (2 raters) and is
unanchored on Human and ELIZA (Rater_B alone)
\citep[\texttt{notebooks/expert\_discrimination\_feasibility\_2026-05-10/findings.md}]{}.
Across all three extensions the binding constraint is the same:
adding per-rater parameters is only safe when within-system rater
contrasts can disentangle them from system-level signal, and the
current panel does not provide that for most systems.

\subsection{Label-level $\boldsymbol\beta$ pooling}
\label{subsec:label-pooling}
\statusexploratory

The published DCM gives every edge $v \to u$ its own pair of
independent Beta priors on $(\beta^{\mathrm{pres}}_u,
\beta^{\mathrm{abs}}_u)$, with the prior mean set by the modeller's
hand-assigned (support, demandingness) label
(\S~\ref{subsec:baseline-model}). The pooling variant replaces these
with one logit-Normal RV per (support, demandingness) \emph{group}
present in the tree -- so all edges sharing a label share a single
$\beta^{\mathrm{pres}}_{(s,d)}$ (and, depending on
\texttt{BETA\_ABS\_BY\_SUPPORT\_DEMAND}, either one
$\beta^{\mathrm{abs}}_d$ per demandingness group or one per (s, d)
group) -- using a non-centred parameterisation
$\beta_{(s,d)} = \sigma\bigl(\mathrm{logit}(\bar\mu_{(s,d)}) +
\sigma_{\mathrm{pool}} \cdot \tilde\beta_{(s,d)}\bigr)$ with
$\tilde\beta_{(s,d)} \sim \mathcal{N}(0, 1)$ and
$\sigma_{\mathrm{pool}} = \texttt{LABEL\_POOL\_SIGMA} = 0.5$ in
logit space, where $\bar\mu_{(s,d)}$ is the paper's tabulated prior
mean for that label. The intent was to reduce the number of free
$\boldsymbol\beta$ parameters from $O(|\mathrm{edges}|)$ to
$O(|\mathrm{labels}|)$ while preserving the modeller's label
semantics as the unit of variation -- letting data calibrate
\emph{per-label} transmission strengths without ungluing each edge
from its label. In practice this is \emph{not} the baseline: the
default configuration is
\texttt{POOL\_BETAS\_BY\_LABEL=False}, the production fits use
per-edge priors at the published means, and the pooled variant has
not produced a clear win on $\Cs$ posterior contraction or
rating-level PPCs on the current data (the leaf and design
constraints discussed in \S~\ref{sec:findings} bind well before the
$\boldsymbol\beta$ parameterisation does). The implementation lives
at \texttt{dcm\_model.py:build\_label\_pool\_hyperparameters}
(line~1071) and is retained as ablation infrastructure rather than a
shipped change.

\subsection{Revised tree-transmission priors}
\label{subsec:revised-tree-priors}

% (Validation and diagnostic infrastructure described in
% \S~\ref{sec:validation}; this subsection intentionally dropped to
% avoid duplicating that material.)


% ============================================================
\section{Validation programme}
\label{sec:validation}

The checks below are organised by the stages of a principled Bayesian
workflow \citep{gelman2020bayesian, betancourt2018calibrating}: pre-data
prior-predictive checks first; then synthetic / fake-data recovery and
sampler diagnostics; then rating-level posterior predictive checks on the
real data; then prior- and structural-sensitivity sweeps, identifiability
ablations, and design-ceiling diagnostics. Most of these checks were only
possible after the move to analytical tree marginalisation
(\S~\ref{subsec:tree-marginalisation}) and the ordinal observation layer
(\S~\ref{subsec:ordinal-probit}): under the published mixed-sampler design,
there was no single posterior to diagnose end-to-end and no rating-level
predictive distribution against which to score the model. The structural
conclusions drawn from these checks are collected in
\S~\ref{sec:findings}; quantitative summaries here are pointers, not
self-contained results.

\subsection{Prior-predictive transmission audit}
\label{subsec:val-prior-predictive}

Before any data is involved, walk the GWT tree analytically at prior-mean
$\boldsymbol\beta$ values and propagate the gap between the indicator
probabilities $\qj$ implied by $\Cs = 0.999$ and by $\Cs = 0.001$ from the
root down to the leaves. Under the paper's per-edge prior means the gap
$\bar q_{\text{Human}} - \bar q_{\text{ELIZA}}$ collapses from $0.33$ at
depth $1$ to $0.04$ at depth $3$; under the asymmetric
$(\beta^{\mathrm{pres}}, \beta^{\mathrm{abs}})$ override at centre
$0.90 / 0.10$ it stays at $\approx 0.51$, and at $0.95 / 0.05$ at
$\approx 0.73$
(\texttt{notebooks/asymmetric\_prior\_sweep\_2026-05-10/eval/analytical\_qj\_by\_prior\_depth.csv},
plot at \texttt{figs/qj\_vs\_depth.png}). The check is data-independent and
sampler-independent: the published tree prior could not, even in
principle, transmit root anchor information to the leaves at the GWT
depth, and this observation alone motivates the asymmetric-prior override
in \S~\ref{subsec:revised-tree-priors}. The same walk is now run as a
sanity gate on every candidate prior before any fit is launched.

\subsection{Synthetic / M-closed recovery and sampler diagnostics}
\label{subsec:val-synthetic-recovery}

The central piece of inferential validation is a multi-seed M-closed pilot
on the current rating design (Human 50, Chicken 93, LLMs 186, ELIZA 50)
with known root truths Human $0.999$, Chicken $0.25$, 2024 LLMs $0.10$,
ELIZA $0.001$
(\texttt{notebooks/synthetic\_validation\_2026-05-06/}, branch
\texttt{synthetic-validation-checks}). Four synthetic seeds are each
fitted under two likelihood families --- the exact-tree likelihood of
\S~\ref{subsec:tree-marginalisation} and the legacy composite
\texttt{pool\_3s} likelihood with soft anchors --- yielding a $4 \times 2
\times 2$ = 16-cell recovery panel for the two free systems (Chicken and
the 2024 LLMs). Coverage of the production 94\% HDI holds in all 16
cells; Chicken-versus-LLM ordering at the posterior \emph{median}
recovers in only one of four seeds under either fit; both models bias
Chicken downward by $\approx -0.13$ and LLM upward by $+0.03$
(exact-tree) / $+0.07$ (\texttt{pool\_3s}). The misspecified
\texttt{pool\_3s} fit produces the larger bias, consistent with
leaf-independent likelihoods over-counting evidence on correlated
sibling indicators. The exact-tree likelihood evaluation is cross-checked
against an independent NumPy reference to numerical precision before any
inferential claim is built on it
(\texttt{composite\_vs\_exact\_diagnostic.py}); on real data the
exact-tree and composite posteriors differ by $\Delta \ell \approx +10$
nats per draw with importance-reweighting ESS $< 0.2\%$, so the choice
between them is a DGP commitment rather than a numerical refinement.

Two oracle audits complement the pilot by isolating design-level
information content from sampler behaviour:

\begin{itemize}[leftmargin=*]
\item \textbf{Oracle root-signal audit}
(\texttt{gwt\_oracle\_root\_signal.py}). At fixed production-median
nuisance, compute the expected marginal ordinal-rating distribution at
each candidate $\Cs$ and the marginal KL from each system's reference
truth. Under the LLM rating design, the KL between $\Cs = 0.10$ and
$\Cs = 0.25$ is $0.23$ nats summed across all 186 LLM ratings; under
the Chicken design the reverse comparison is $0.12$ nats across 93
ratings. The design itself is shallow at the $\Cs$ separations the
paper asks about, independent of any sampler or prior choice.
\item \textbf{Oracle internal-node identifiability audit}
(\texttt{gwt\_oracle\_internal\_identifiability.py}; written directly to
Arvo's question of whether the binary internal feature/subfeature
states are recoverable labels or organisational bookkeeping). At the
same oracle nuisance, $\Pr(z_v = 1 \mid \mathbf r, \theta^\star)$ is
computed by exact belief propagation for $5{,}000$ simulated
internal-state slots and scored against the simulated truth. Mean
relative entropy reduction is $0.439$ overall but breaks down sharply:
depth-3 nodes sit at $0.048$, fanout-6 nodes at $0.831$, and
calibration is preserved across the probability bins. The audit
supports reporting internal-node summaries as posterior probabilities
with depth/fanout stratification, not as recovered labels.
\end{itemize}

Sampler diagnostics --- $\hat R$, ESS, divergences --- are reported on
every fit; under the marginalised exact-tree model, production fits run
clean ($\hat R \le 1.05$ on parameters of interest, ESS bulk in the
thousands, 0 divergences). The scalar that turns out to do most of the
diagnostic work in this project is the \emph{posterior contraction ratio}
$\mathrm{sd}(\Cs \mid \mathbf r) / \mathrm{sd}(\Cs)$ on the free systems:
contraction $\approx 1$ indicates that the data add no curvature
information on $\Cs$ beyond the stance prior, and contraction $> 1$
indicates a posterior \emph{broader} than the prior (multimodality or
sampler pathology). This scalar is reported on every fit in the sections
below; its persistent location near $1$ on free $\Cs$ across leaves,
priors, anchors, and sample sizes is the empirical backbone of the
findings in \S~\ref{sec:findings}.

\subsection{Posterior predictive checks on real data}
\label{subsec:val-ppc}

The ordinal mixture likelihood (\S~\ref{subsec:ordinal-probit}) plus
analytical tree marginalisation made rating-distribution PPCs feasible
for the first time; the published binary-collapse construction did not
permit them. Four nested PPC granularities are run on every shipped fit:

\begin{itemize}[leftmargin=*]
\item \textbf{Per-system rating-distribution overlays}
(\texttt{posterior\_predictive\_rating\_dist.py},
\texttt{figs/ppc\_rating\_dist\_\{synthetic,real\_data\}.png}). The
\emph{shape} of the model's predicted rating distribution is
qualitatively wrong at the extremes: at fitted $\tilde \qj \in [0.6,
1.0]$ on real data, observed ratings concentrate $85\text{--}88\%$ at
category 6 while the model places only $28\text{--}45\%$ there. The
same bimodal-mass deficit appears under three-state and
continuous-mixture leaves (\S~\ref{subsec:three-state}) and is
therefore a property of the ordered-probit family --- a single latent
location $\eta$ cannot simultaneously place $\sim 0.85$ mass at
category 6 \emph{and} hold the other categories' mass at the levels
the same indicator's lower-$\tilde q_j$ ratings require under shared
cutpoints. This is an honest limitation of the observation family
shipped in \S~\ref{subsec:ordinal-probit} and is the most consequential
single PPC finding of the SPAR tenure.
\item \textbf{Per-indicator $\chi^2$} between predicted and observed
rating histograms (\texttt{per\_indicator\_ppc.py}). Mean $\chi^2$
across indicators sits at $7\text{--}10$ across all four systems
regardless of prior choice; the absolute values are large, so
cross-prior $\chi^2$ ranking is not a reliable model-selection signal
while the bimodal-mass deficit above is uncorrected.
\item \textbf{Focus-cell tail residuals.} For reference cells that
combine the cross-system rater (Rater_B) with a reference
system, the right-tail and left-tail mispredictions
$\Delta_{\mathrm{right}}$, $\Delta_{\mathrm{left}}$ are decomposed
against an oracle emission ceiling (50-draw lower-bound analysis) and
into tree-prior mixture weights $\pi_m(q) = ((1-q)^2,\, 2q(1-q),\,
q^2)$ versus leaf-updated responsibilities
$\rho_m \propto \pi_m(q) L_m$. The residual on \texttt{E\_cross}
$\times$ Human is $\approx 100\%$ tree-side (needs $q \to 0.97$ at
the reference); the residual on \texttt{E\_cross} $\times$ ELIZA is
$\approx 69\%$ tree-side plus an $\approx 31\%$ irreducible emission
floor that no $q$ value reaches under the shared cutpoints
(\texttt{notebooks/18\_gwt\_tree\_propagation.ipynb}~\S~B.11c;
summarised in \texttt{notebooks/friday\_meeting\_brief.md}).
\item \textbf{Apples-to-apples exact-tree PPC by belief propagation}
(\texttt{exact\_tree\_ppc.py}). Replaces composite-style
responsibilities with the correct two-pass exact-tree responsibilities,
so PPC numbers on exact-tree fits are not silently scored against a
composite counterfactual. Used in the round-2 diagnostics
(\texttt{notebooks/meeting\_prep\_arvo\_2026-04-27/diagnostics\_round2\_results.md}).
\end{itemize}

These PPCs underwrite the structural finding (\S~\ref{sec:findings})
that the indicator leaf is not the limiting layer at the current design,
and that any rating-distribution improvement at the extremes will
require an observation-family change rather than a leaf change.

\subsection{Prior and structural sensitivity sweeps}
\label{subsec:val-prior-sweeps}

Two prior sweeps quantify how strongly conclusions depend on modeller
choices upstream of the likelihood.

\begin{itemize}[leftmargin=*]
\item \textbf{Stance prior on $\Cs$.} Refit synthetic seed 06 under
$\Cs \sim \mathrm{Uniform}(0, 1)$ in place of the paper's
$\mathrm{Beta}(1, 5)$. The LLM posterior median moves from $0.16$ to
$0.62$ ($\approx 5\times$ truth) and Chicken from $0.11$ to $0.34$;
contraction on $\Cs$ stays $\approx 1$ throughout. The headline
"$\Pr(\Cs = 1) \approx 0.11$" for the 2024 LLMs is therefore
substantially prior-driven; the data does not pin the location of
$\Cs$ within $[0, 1]$ on its own (branch
\texttt{synthetic-validation-checks}).
\item \textbf{Tree-transmission $(\beta^{\mathrm{pres}},
\beta^{\mathrm{abs}})$ override.} Sweep four prior centres (paper
means; $0.85 / 0.15$ at $\sigma_{\text{logit}} = 0.5$; $0.90 / 0.10$
at $\sigma_{\text{logit}} = 0.30$; $0.95 / 0.05$ at
$\sigma_{\text{logit}} = 0.30$) on synthetic seed 06 and on real data
with reference anchors fixed
(\texttt{notebooks/asymmetric\_prior\_sweep\_2026-05-10/findings.md}).
Synthetic LLM signed bias improves monotonically from $+0.059$ to
$+0.011$ with prior strength, Chicken bias stays $\approx -0.13$, and
contraction on $\Cs$ remains $\approx 1$ across the entire sweep. On
real data the centre prior shifts the LLM posterior from $0.111$ to
$0.263$ (Chicken remains near $0.252$) --- a substantial movement on
real data of a direction the synthetic check cannot adjudicate, and a
change for which only the structural prior-predictive win
(\S~\ref{subsec:val-prior-predictive}) currently stands alone.
\end{itemize}

The PPC-$\chi^2$ surface is approximately flat across these priors;
prior selection on rating-level fit is not reliable while the
bimodal-mass deficit of \S~\ref{subsec:val-ppc} remains uncorrected.

\subsection{Identifiability ablations}
\label{subsec:val-identifiability}

Three orthogonal ablations localise where information about $\Cs$
enters the posterior under the current design.

\begin{itemize}[leftmargin=*]
\item \textbf{Reference-anchor leverage.} Vary the anchor specification:
hard ($\Cs$ fixed at $0.999 / 0.001$), soft
($\Cs \sim \mathrm{Beta}(50, 1) / \mathrm{Beta}(1, 50)$,
95\% CIs $\approx [0.93, 0.999]$ and $[0.001, 0.07]$ respectively),
one-anchor-released
(\texttt{HUMAN\_FREE\_SYSTEM\_CONFIGS} /
\texttt{ELIZA\_FREE\_SYSTEM\_CONFIGS} in
\texttt{gwt\_reference\_recovery\_analysis.py}), and no-anchor with
per-system $\boldsymbol\kappa$. Hard and soft give similar free-system
posteriors and the soft Human posterior median lands at $0.990$ ---
slight inward drift from the hard value rather than outward expansion,
so the hard/soft distinction is presentational rather than load-bearing.
The no-anchor case with per-system cutpoints, in contrast, collapses
every $\Cs$ posterior to the Beta(1, 5) prior mean on synthetic seed 06
(Human truth $0.999$ recovered as $0.215$, ELIZA truth $0.001$ as
$0.122$, Chicken at $0.109$, LLM at $0.196$). The reference anchors are
carrying every unit of per-system distinguishability that the current
design provides.
\item \textbf{Oracle nuisance clamp ladder} (branch
\texttt{oracle-c-ablation-and-info-diags}). Pin $(a,
\boldsymbol\kappa, \mathbf b, \boldsymbol\beta)$ at synthetic truth
one parameter group at a time using
\texttt{CLAMP\_OBSERVATION\_PARAMS} / \texttt{CLAMP\_TREE\_BETAS} on
\texttt{ModelConfig} and refit with only $\Cs$ free. At the
fully-clamped \texttt{c\_only} rung, three-state Chicken contraction
is $0.87$ --- the best of all leaves tested --- while the
continuous-mixture leaf is $1.16$ (\emph{broader} than the prior) and
\texttt{direct\_q} moves contraction by $< 0.01$ from the unclamped
baseline despite a $\approx 3\times$ analytical Fisher advantage.
Even at oracle nuisance, no current leaf family clears the contraction
floor of $\approx 0.87$, so the binding constraint on $\Cs$
identification is upstream of the indicator leaf (branches
\texttt{leaf-direct-q}, \texttt{leaf-mixture-shared-h}).
\item \textbf{Composite versus exact-tree DGP.} The two coherent
likelihood families disagree by $\Delta \ell \approx +10$ nats per
draw on real data with importance-reweighting ESS $< 0.2\%$
(\texttt{composite\_vs\_exact\_diagnostic.py},
\texttt{composite\_vs\_exact\_diagnostic\_extended.py}). The
exact-tree fit is more conservative on $\Cs$ for the free systems
(Chicken $0.36 \to 0.25$, LLM $0.17 \to 0.11$) precisely because it
accounts for evidence redundancy through shared internal states; the
composite fit treats correlated sibling indicators as conditionally
independent given $(\Cs, \boldsymbol\beta)$ and therefore overcounts
their evidence. Both are coherent models, but the choice is a DGP
commitment (latent binary states versus bookkeeping for marginal
indicator probabilities) rather than a numerical refinement.
\end{itemize}

Read jointly, the three ablations imply that posterior medians on the
free $\Cs$ are inherited from the reference anchors and the stance
prior, not from system-specific rating-distribution curvature against
$\Cs$ at this design.

\subsection{Design and information-ceiling diagnostics}
\label{subsec:val-design-ceiling}

Three checks ask what more or different data would buy under the
current model structure.

\begin{itemize}[leftmargin=*]
\item \textbf{Sample-size sweep.} Replicate each
\emph{(rater, indicator)} slot $K$ times with independent draws given
the latent state, on synthetic seed 06 with the centre $\beta$ prior,
for $K \in \{1, 2, 5, 10, 20\}$
(\texttt{notebooks/sample\_size\_sweep\_2026-05-10/findings\_sample\_size.md}).
The posterior SD on free $\Cs$ asymptotes at the Beta(1, 5) prior SD
$\approx 0.141$; Chicken contraction floors at $1.01$, LLM at $1.13$;
medians plateau near the prior mean $\approx 0.17$ for both free
systems even at $K = 20$. Parametric $1/\sqrt K$ shrinkage is not
observed. \emph{More raters at the current design does not
asymptotically resolve $\Cs$.} This is the strongest single negative
result of the validation programme.
\item \textbf{Information-theoretic leaf ceiling}
(\texttt{notebooks/asymmetric\_prior\_sweep\_2026-05-10/leaf\_model\_compression.py};
diagnostic library at
\texttt{notebooks/oracle\_c\_ablation\_2026-05-10/info\_diagnostics.py}).
Per-rating mutual information $I(Q; Y)$ at oracle nuisance is
$\approx 0.24$ nats against a categorical entropy ceiling of
$\log 7 \approx 1.95$ nats. Across five candidate leaves the nats
lost per rating range from $1.81$ (three-state baseline) down to
$1.35$ (continuous mixture), with \texttt{direct\_q}, logit-Normal
$z$, and $K$-state Beta-Binomial bracketing in between. No candidate
narrows the gap to the entropy ceiling enough to plausibly drive free
$\Cs$ contraction below the prior at the current rater count, which
is consistent with the empirical clamp-ladder result above.
\item \textbf{Per-expert discrimination feasibility}
(\texttt{notebooks/expert\_discrimination\_feasibility\_2026-05-10/findings.md};
written directly to Arvo's request to check whether a per-rater
discrimination $d_e$ extension is buildable on the current panel). At
oracle nuisance and $d_e = 1$, the diagonal expected Fisher
information is $93\text{--}115$ nats$^2$ for each of the five
single-system experts and $\approx 396$ nats$^2$ for the one
cross-system expert (Rater_B, $\approx 3\times$ more ratings),
all well above the $\sim 1$ nat$^2$ collapse threshold. The diagonal
verdict is encouraging; the joint identifiability of $\{d_e\}$ with
the global $a$ along the $a \cdot d_e$ confound is constrained
within-system only on LLM (4 raters) and Chicken (2 raters), and is
\emph{unanchored} on Human and ELIZA (Rater_B is the sole rater). $d_e$
is locally informative on the current panel but not posterior
resolvable without within-system rater overlap, which is a
survey-design lever rather than a modelling lever.
\end{itemize}

The three diagnostics together motivate the structural-findings claim
that the identifiability bottleneck is upstream of any single layer of
the current model, and that rater budget on the current survey design
is the wrong axis to expand (\S~\ref{sec:findings}).


% ============================================================
\section{Main structural findings}
\label{sec:findings}


% ============================================================
\section{Current best model and competing variants}


% ============================================================
\section{Open questions and risk register}


% ============================================================
\section{Diagnostic plan forward}


% ============================================================
\section{Reproducibility notes}


% ============================================================
\bibliographystyle{plainnat}
% \bibliography{references}

\end{document}