## Revisiting the Platonic Representation Hypothesis: An Aristotelian View

Fabian Gr¨oger*123 Shuo Wen*1 Maria Brbi´c1

# arXiv:2602.14486v2[cs.LG]25Jun2026

### Abstract

The Platonic Representation Hypothesis suggests that representations from neural networks are converging to a common statistical model of reality. We show that the existing metrics used to measure representational similarity are confounded by network scale: increasing model depth or width can systematically inflate representational similarity scores. To correct these effects, we introduce a permutation-based null-calibration framework that transforms any representational similarity metric into a calibrated score with statistical guarantees. We revisit the Platonic Representation Hypothesis with our calibration framework, which reveals a nuanced picture: the apparent convergence reported by global spectral measures largely disappears after calibration, while local neighborhood similarity, but not local distances, retains significant agreement across different modalities. Based on these findings, we propose the Aristotelian Representation Hypothesis: representations in neural networks are converging to shared local neighborhood relationships.

### 1. Introduction

Quantifying the similarity between neural network representations is central to understanding the geometry of learned representation spaces (Raghu et al., 2017; Nguyen et al., 2021), guiding transfer learning decisions (Kornblith et al., 2019; Neyshabur et al., 2020; Gr¨oger et al., 2025), and relating artificial representations to neural measurements in neuroscience (Schrimpf et al., 2018). The Platonic Representation Hypothesis (Huh et al., 2024) posits that as neural networks scale, representations across different modalities become increasingly similar, suggesting convergence to a shared statistical model of reality. This hypothesis has mo-

Project Page: brbiclab.epfl.ch/aristotelian Code: github.com/mlbio-epfl/aristotelian

*Equal contribution 1EPFL 2University of Basel 3HSLU. Correspondence to: Maria Brbi´c <maria.brbic@epfl.ch>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

#### The Aristotelian Representation Hypothesis

Neural networks, trained with different objectives on different data and modalities, are converging to shared local neighborhood relationships.

###### Local alignment

Space Space

[Figure 1]

Figure 1. The Aristotelian Representation Hypothesis: Local relations (“who is near whom”), rather than distances between data points, are preserved across different representation spaces. Representation learning algorithms converge toward shared local neighborhood relationships.

tivated a growing literature that uses representational similarity to study whether scaling produces universal structure across models (Huh et al., 2024; Maniparambil et al., 2024; Tjandrasuwita et al., 2025; Zhu et al., 2026). To measure representational similarity across models, different metrics have been proposed, such as Centered Kernel Alignment (Kornblith et al., 2019), Canonical Correlation Analysis (Weenink, 2003), Representational Similarity Analysis (Kriegeskorte et al., 2008), and mutual k-Nearest Neighbors (Huh et al., 2024).

In this work, we identify two pervasive confounders that distort representational similarity measurements. The first is the model width: when the embedding dimension increases relative to the sample size, interaction-matrix-based similarity metrics exhibit a systematic positive baseline even when representations are independent. This spurious similarity is a general consequence of dimensionality-driven null inflation: the expected similarity under independence does not vanish but instead depends on both the representation dimensionality and the sample size (Figure 2a). As a result, wider models can appear more aligned simply because their representations live in higher-dimensional spaces. The second confounder is the model depth. Many analyses do not compare individual layer pairs, because it is unknown

(a) Width confounder:

(b) Depth confounder:

Raw score Calibrated score

Raw

[Figure 2]

[Figure 3]

[Figure 4]

#samples

Score

0.5

Calibrated

0.0

Model depth (# comparisons)

Model width

Model width

(c) Revisit the Platonic Representation Hypothesis

Global similarity (CKA) Local similarity (mutual KNN)

0.16

|Small vision model Medium vision model| | | | | |
|---|---|---|---|---|---|
|Large vision model| | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| |Calibration| | | | |
| | | | | | |
| | | | | | |

Small vision model

Medium vision model

0.45

0.15

Large vision model

0.14

Alignmentscore

Alignmentscore

Alignmentscore

0.40

0.13

0.35

0.12

0.11

0.30

0.10

0.25

0.09

Calibration

0.08

0.20

Language model capacity

Language model capacity

Language model capacity Language model capacity

Figure 2. Null calibration removes width and depth confounders. (a) Width confounder: raw scores exhibit positive null baselines that increase with the ratio of dimension (width) of the spaces and the number of samples; calibration collapses them to zero. (b) Depth confounder: selection-based summaries (max over layers) inflate with search space size; aggregation-aware calibration removes this. (c) After calibration, global metrics lose their convergence trend, while local metrics retain significant alignment.

where similarity arises (Schrimpf et al., 2018; Huh et al.,

- 2024). Instead, they search over all pairs and report a summary statistic such as the maximum. This is widespread practice: studies routinely compute similarity over many layer pairs and report an aggregate such as the maximum or best-matching layer (Raghu et al., 2017; Kornblith et al., 2019; Schrimpf et al., 2018; Huh et al., 2024; Kapoor et al.,
- 2025). Taking a maximum over many comparisons inflates the reported score even if there is no similarity, since the expected maximum of independent draws exceeds the mean. This inflation grows with the number of comparisons, so deeper models can appear more aligned simply because more layer pairs are compared (Figure 2b). Together, these confounders undermine the comparative use of representational similarity without calibration.

To address these issues, we introduce the null-calibration for representational similarity, a general permutation-based framework that transforms any similarity metric into a calibrated score with a principled null reference, here defined as no relationship (Figure 2). The core idea is to measure how extreme an observed similarity is relative to an empirical null distribution obtained by breaking sample correspondences. For scalar comparisons (i.e., width confounder), we estimate a critical threshold from the null distribution and define a calibrated score that is zero when the observed similarity falls below this threshold and rescaled to preserve the maximum at one. For selection-based summaries (i.e., depth confounder), we apply aggregation-aware calibration. We

compute the null distribution of the same aggregate statistic that is ultimately reported (e.g., the maximum over all layer pairs), thereby calibrating the selection step itself.

These observations raise a question: Does the Platonic Representation Hypothesis still hold once similarity is calibrated? We find that, after calibration, the previously reported convergence in global metrics (Huh et al., 2024; Maniparambil et al., 2024; Tjandrasuwita et al., 2025) largely disappears, suggesting it was driven primarily by width and depth confounders, whereas local neighborhood-based metrics retain significant cross-modal alignment (Figure 2c). However, we also observe that the convergence in local distances is not preserved, suggesting that only local neighborhood relationships are aligned. Motivated by these results, we refine the original Platonic Representation Hypothesis and propose the Aristotelian Representation Hypothesis1: Neural networks, trained with different objectives on different data and modalities, converge to shared local neighborhood relationships (Figure 1). We name it after the Greek philosopher Aristotle, who was a student of Plato and, in his Categories, established the principles of relatives (Aristotle, ca. 350 B.C.E).

### 2. Related work

Representational similarity metrics. A long line of work compares representation spaces using a variety of similarity measures. Canonical Correlation Analysis (CCA) (Hotelling, 1992) and variants such as Singular Vector Canonical Correlation Analysis (SVCCA) (Raghu et al., 2017) and Projection Weighted Canonical Correlation Analysis (PWCCA) (Morcos et al., 2018) compare subspaces up to linear transformations, while Procrustesand shape-based distances compare representations up to restricted alignment classes (Ding et al., 2021; Williams et al., 2021). Centered Kernel Alignment (CKA) (Kornblith et al., 2019) has become a dominant tool for comparing deep representations, with kernelized variants extending to nonlinear similarity. Representational Similarity Analysis (RSA) (Kriegeskorte et al., 2008), originating in neuroscience, compares representational dissimilarity matrices rather than feature bases. Neighborhood-based approaches, such as mutual k-Nearest Neighbors (mKNN) (Huh et al., 2024), capture local topological consistency rather than global alignment. However, recent evaluations stress that different metrics encode different invariances and can yield qualitatively different conclusions, motivating more robust reporting practices (Klabunde et al., 2025a;b; Ding et al., 2021; Harvey et al., 2024; Bo et al., 2024). This sensitivity

1Calling this refinement Aristotelian: it emphasizes learned representations converging on relations among instances (who is near whom) rather than the idea of convergence toward a globally matching structure.

is especially consequential in the neuro-AI literature, where the choice of similarity measure can profoundly change conclusions about which models best align with the brain (Soni et al., 2024), which is why scores that are calibrated and comparable across metrics and scales are needed.

Reliability of representational similarity metrics. In finite-sample, high-dimensional regimes, raw similarity scores can be systematically biased. Recent works (Murphy

- et al., 2024; Chun et al., 2025) propose debiased CKA, but these corrections are metric-specific. For neighborhoodbased metrics, no analogous debiasing methods exist despite distance concentration effects that inflate random k-NN overlap (Beyer et al., 1999; Aggarwal et al., 2001). Other approaches address confounding from input population structure. For instance, Cui et al. (2022) propose regression-style deconfounding to remove effects of shared input statistics on RSA/CKA. Analogous concerns arise for regression-based predictivity in neuroscience, where highdimensional fits can inflate alignment scores and may fail to identify genuinely brain-like models (Schaeffer et al., 2024). A separate reliability issue arises from layer search, where max or top-k aggregation across many layer pairs introduces multiple-comparison inflation. While resampling-based “maxT” procedures (Westfall & Young, 1993; Nichols & Holmes, 2002) can calibrate such aggregates, this has not yet been applied in representational similarity studies. Our calibration framework addresses both finite-sample bias and selection inflation in a unified, metric-agnostic way.

The Platonic Representation Hypothesis. A growing body of work examines whether neural networks trained under different conditions converge toward similar representations. The Platonic Representation Hypothesis (Huh et al., 2024) posits that as models scale, their representations increasingly converge across architectures and even across modalities such as vision and language, with convergence reported under both global and local similarity measures. Follow-up work has examined factors influencing these trends, including model size, training duration, and data distribution (Raugel et al., 2025), and has explored analogous convergence effects in broader settings such as video models (Zhu et al., 2026) and comparisons to biological vision (Marcos-Manch´on & Fuentemilla, 2025). Prior work attributes convergence to shared factors such as training data distribution, scale, and objective functions (Raugel

- et al., 2025; Huh et al., 2024). Our calibration-based analysis does not contradict these findings but refines their interpretation, showing that the evidence for convergence depends on whether it is measured by global or local similarity. In this work, we revisit the Platonic Representation Hypothesis using our null-calibration framework that controls for width and depth confounders.

### 3. Problem setup

#### 3.1. Representation spaces and similarity score

y be two representation spaces, where dx and dy are the respective space dimensions. For a set of n input samples, let X ∈ Rn×d

Let X ⊆ Rd

x and Y ⊆ Rd

x and Y ∈ Rn×d

y

be the corresponding embeddings in X and Y. We assume row-wise alignment such that the i-th row of X and Y correspond to paired inputs. We use a similarity score s(X,Y) ∈ R to quantify the agreement between X and Y. In practice, we compute it from X,Y and, by a slight abuse of notation, denote it with s(X,Y).

We consider three families of metrics: (i) spectral: metrics defined on the spectrum of cross-covariance or Gram matrices (e.g., CKA, CCA), (ii) neighborhood: metrics measuring local topological overlap (e.g., mKNN), and (iii) geometric: second-order isomorphism metrics (e.g., RSA). Appendix B provides definitions of the metrics used in this paper.

#### 3.2. The null hypothesis of independence

We claim that a similarity score s(X,Y) is uninterpretable without a baseline. To provide this baseline, we define the null hypothesis H0 as the absence of a relationship between X and Y beyond their marginal statistics. We operationalize H0 via a permutation group Πn acting on sample indices: draw π ∼ Unif(Πn) independently of (X,Y) and evaluate s(X,π(Y)), where π(Y) permutes the rows of Y.

Assumption 3.1 (Exchangeability under the null). Under H0, the joint distribution of paired samples is invariant to relabeling of correspondences. For any permutation π ∈ Πn, PH

(X,π(Y)).

(X,Y) = PH

0

0

This assumption implies that if no true relationship exists, the observed pairing is statistically indistinguishable from a random shuffling of the data. It allows us to construct an empirical null distribution by holding X fixed and shuffling the rows of Y.

#### 3.3. Baseline problem: non-zero null expectations

Ideally, under H0, we desire Eπ[s(X,π(Y))] ≈ 0. However, for commonly used raw or biased estimators, the expected similarity under the null is not zero,

##### µ0(n,dx,dy) := Eπ[s(X,π(Y))]. (1)

This baseline µ0 is metric- and preprocessing-dependent and can deviate from zero in finite samples. It also varies with sample size and dimension, thus acting as a confounding variable in comparative studies.

### 4. Theoretical motivation: spurious alignment

We motivate and formalize why raw representational similarity metrics fail in cross-scale model comparisons. We identify two distinct sources of confounding: (i) the width confounder driven by representation dimension, and (ii) the depth confounder driven by the number of layers considered when comparing models.

neighborhood-based metrics such as mutual k-NN exhibit different behavior, as they rely on set comparisons rather than interactions.

Proposition 4.2 (Null baseline for neighborhood metrics). Assume the rows are i.i.d. with xi and yi independent, and that pairwise distances are almost surely distinct (e.g., under absolutely continuous distributions). Then for any k < n,

#### 4.1. The width confounder

Many spectral-family similarity metrics, e.g., linear/kernel CKA and the RV coefficient, can be written as functionals of an interaction operator constructed from two representations. One such operator is the (normalized) crosscovariance

1 n − 1

X⊤c Yc ∈ Rd

x×dy, (2)

C =

where Xc and Yc denote row-centered representations (Appendix B.1).

A common but misleading intuition is that if X and Y are independent, then C ≈ 0 and therefore spectral aggregates should be near zero. In high dimension this fails: the null interaction energy is typically non-zero.

Proposition 4.1 (Non-vanishing null interaction energy). Assume the rows are i.i.d. with E[xi] = E[yi] = 0, Cov(xi) = Id

, and xi and yi are independent. Then

, Cov(yi) = Id

x

y

##### EH

0

mKNN(X,Y) =

k n − 1

. (4)

Proof. See Appendix C.6.

In particular, neighborhood metrics have null baselines scaling as O(k/n).

The difference in null baseline between spectral and neighborhood metrics is substantial: (i) The neighborhood scale k can be fixed consistently across experiments, whereas the embedding dimension d is determined by the model architecture, making it difficult to control in comparison studies. (ii) The neighborhood metrics are much less confounded since k ≪ d in typical settings.

Geometric metrics (e.g., Procrustes), similarly to the spectral metrics, are functions of all-pairs distances or inner products, which are the interaction quantities of Proposition 4.1. They thus inherit a width-dependent O(d/n) baseline rather than the O(k/n) baseline of rank-based neighborhood metrics. We verify this empirically in Appendix E.7.

0 ∥ C∥2F =

EH

dxdy n − 1

. (3)

- Proof. See Appendix C.4.

Since CKA is scaled by the normalized self-similarity terms, which each scale as O(

√

d), the resulting null baseline for the metric is thus O(d/n) to leading order. We refer to this O(d/n) effect as the width confounder: at a fixed sample size n it grows with the representation width d, and it persists even when two representations share a genuine signal (Proposition C.5).

This aligns with insights from random matrix theory: in high-dimensional regimes (d ∼ n), the null singular spectrum of interaction operators (after centering/whitening) concentrates into a non-trivial “noise bulk” whose upper edge depends on d/n and preprocessing, rather than collapsing to zero (Wachter, 1978; M¨uller, 2002; Livan et al., 2018). Our framework estimates this null baseline directly via permutation, providing a metric- and pipeline-independent alternative to asymptotic formulas.

Neighborhood metrics follow a different regime. While spectral metrics have null baselines scaling as O(d/n),

#### 4.2. The depth confounder

- A subtle yet pervasive issue is the comparison of selection-based alignment summaries across models. Let

Sℓ,ℓ′ := s(X(ℓA),Yℓ(′B)) be the similarity between layer ℓ of model A and layer ℓ′ of model B. It is common to

summarize the similarity between two models by the maximum alignment score Tmax = maxℓ,ℓ′ Sℓ,ℓ′ (Schrimpf et al., 2018; Huh et al., 2024; Raghu et al., 2017; Kornblith et al., 2019). Let M = LALB be the number of layer pairs searched, where LA and LB are the depths of models A and

- B. Even under H0, taking a maximum over M comparisons inflates the reported score, a “look-elsewhere” effect. This is an instance of the classical multiple comparisons problem (Benjamini & Hochberg, 1995; Bonferroni, 1936): as M increases, the probability that at least one null similarity exceeds any fixed threshold grows, inflating the expected maximum. Consequently, when alignment is summarized via a max or top-k statistic without correction, unrelated representations can exhibit spuriously high reported similarity, as the inflation depends on model depth, making raw summaries non-comparable across architectures.

Characterizing this inflation does not require independence across pairs. It follows from a uniform right-tail bound.

Assume there exist a common mean µ ∈ R and σ > 0 such that the null fluctuations satisfy, for all (ℓ,ℓ′) and all t ≥ 0,

t2 2σ2

. (5)

P(Sℓ,ℓ′ − µ ≥ t) ≤ exp −

For bounded similarities Sℓ,ℓ′ ∈ [smin,smax], Hoeffding’s inequality implies a sub-Gaussian right-tail bound of the form Equation (5) with σ ≤ (smax − smin)/2. This covers many common bounded metrics (e.g., CKA/RSA/mKNN). Crucially, only the right tail is needed for bounding the maximum. Then a union bound gives

t2 2σ2

P(Tmax − µ ≥ t) ≤ M exp −

and consequently for a constant C

, (6)

[Tmax] ≤ µ + C σ log M. (7)

##### EH

0

- Proof. See Appendix C.5.

This creates a depth confounder. Deeper models (larger M = LALB) can attain higher raw “max-alignment” scores purely because of a larger search space. Correlations across neighboring layers reduce the effective number of comparisons, but the inflation remains monotone in the search space size in typical workflows. Therefore, raw scaling plots of Tmax (or top-k summaries) are not comparable across architectures unless the selection step itself is calibrated.

The two confounders are not independent but hierarchical: the depth confounder operates on top of the width confounder. The width confounder gives each individual layer pair a positive chance similarity, and deeper models supply a larger pool of such chance similarities, so the selection maximum drifts higher.

### 5. Representational similarity calibration

To overcome the issues of the width and depth confounders, we introduce the null-calibration for representational similarity. The key idea is to compare observed similarity scores against an empirical null distribution obtained by permuting sample correspondences, thereby establishing a principled zero point that accounts for finite-sample, high-dimensional artifacts.

#### 5.1. Null-calibrated similarity

We propose null-calibrated similarity measures to correct for width and depth confounders by transforming raw similarity scores into an effect size with a principled zero point.

Given representations X ∈ Rn×d

y aligned by rows, we operationalize the null hypothesis H0 (no relationship beyond marginal statistics) by permuting sample

x and Y ∈ Rn×d

correspondences. For permutations πk ∈ Πn drawn i.i.d. uniformly from Πn and independently of (X,Y), we form null scores

##### s(k) = s(X,πk(Y)), k = 1,...,K. (8)

Let sobs := s(X,Y) denote the observed score. Let s(1) ≤ s(2) ≤ ··· ≤ s(K+1) denote the order statistics of the combined multiset {sobs,s(1),...,s(K)} (with ties allowed). We define a right-tail rank-based critical value:

τα := s(⌈(1−α)(K+1)⌉), (9)

where ⌈(1 − α)(K + 1)⌉ is the (1 − α)-quantile of the (K + 1)-sized multiset and the empirical right-tail p-value:

p =

1 + #{k ∈ {1,...,K} : s(k) ≥ sobs} K + 1

. (10)

The critical value τα defines a robust zero point: values below τα are typical under H0 at level α, while p provides an evidence measure that can be combined with multipletesting correction when many comparisons are performed.

The proposed calibration framework relies on randomization (permutation) to construct a null distribution for any similarity statistic. This yields finite-sample guarantees under an exchangeability condition (Assumption 3.1), and it implies useful invariances that make calibrated scores comparable across metrics and implementations.

The permutation p-value in Equation (10) is super-uniform under H0 (i.e., PH

(p ≤ α) ≤ α for all α ∈ [0,1]), a standard consequence of randomization inference (Nichols & Holmes, 2002; Phipson & Smyth, 2010; Good, 2005) (see Appendix C.1 for formal definitions and proofs).

0

Corollary 5.1 (Type-I control for calibrated scores). Let sobs = s(X,Y) and s(k) = s(X,πk(Y)) for k = 1,...,K. Define the add-one permutation p-value p as in Equation (10), and equivalently define the rank-based critical value τα := s(⌈(1−α)(K+1)⌉) from the sorted combined set {sobs,s(1),...,s(K)}. Under Assumption 3.1,

p ≤ α ≤ α and hence PH

##### PH

sobs > τα ≤ α,

0

0

(11) so the gating rule “scal > 0” (where scal is the calibrated score defined in Equation (12), which implies p ≤ α) is a finite-sample α-level declaration of similarity above chance.

Proof. Follows directly from Lemma C.2; see Appendix C.1.

Calibrated score (scalar case). While p-values and null percentiles are rank-based and therefore invariant under monotone transformations of the raw score (Proposition C.3; see Appendix C.2), effect sizes serve a complementary purpose: they quantify how much similarity exceeds chance on

an interpretable scale. The calibrated score achieves this by rescaling the excess over the null threshold τα to the interval [0,1]. This rescaling is not monotone-invariant, by design. A purely rank-based calibration would be equivalent to a score shift and would be unable to correct for the scale-dependent null baselines identified in Section 4. The calibrated score instead adapts to the actual null distribution, providing a meaningful zero point.

For similarity metrics with a known maximum (upper bound) smax (often smax = 1), we define a max-preserving calibrated score

sobs − τα smax − τα

,0 . (12)

scal = max

This calibrated score depends on the chosen level α through τα (Equation (9)). We therefore also report the corresponding permutation p-value and/or null percentile for an α-free summary. This score satisfies scal = 0 whenever sobs ≤ τα (i.e., below the estimated right-tail critical value of the permutation null), and scal = 1 when sobs = smax (i.e., perfect similarity remains 1). When smax is unknown, or the metric is unbounded, we default to the unnormalized effect size [s − τα]+ = max(s − τα,0).

#### 5.2. Aggregation-aware null-calibration

To analyze the similarity between two models A and B with depths LA and LB, a common approach is to compute a layer-by-layer similarity matrix S ∈ RL

A×LB by evaluating a similarity score for every pair of layers:

Sℓ,ℓ′ = s X(ℓA),Yℓ(′B) , (13)

where X(ℓA) ∈ Rn×d

ℓ and Yℓ(′B) ∈ Rn×dℓ′ are the representations of models A and B at layers ℓ and ℓ′ respectively, evaluated on n samples, and s(·,·) is a similarity metric. A common practice is then to summarize S by a selectionbased aggregation operator, such as taking the maximum. These summaries are attractive because they support statements such as “there exists a layer in A that matches some layer in B” or “each layer of A best matches a layer in B”. However, selection introduces a statistical effect: even under the null hypothesis of no relationship between representations, selection-based summaries are systematically inflated.

As analyzed in Section 4.2, this inflation grows with the number of layer pairs and makes na¨ıve post-selection pvalues anti-conservative. Our aggregation-aware calibration addresses this by calibrating the reported statistic directly: the null distribution must match the entire analysis pipeline. Let the aggregate score be T(S) (e.g., a maximum), then the appropriate null is the distribution of T(S) under a valid null transformation (e.g., permuting sample correspondences). We therefore define an aggregation-aware permutation null.

When T is the maximum over layer pairs, this recovers the classical “maxT” procedure of Westfall & Young (1993). Our formulation generalizes it to arbitrary selection-based aggregates (e.g., top-k) and to any similarity metric.

Consistency of permutations across layers. For each draw πk ∈ Πn, we apply the same sample permutation to all layers of model B and define

Sℓ,ℓ(k)′ := s X(ℓA),πk Yℓ(′B) , (14) ℓ = 1,...,LA, ℓ′ = 1,...,LB,

then compute T(k) := T(S(k)). Let Tobs := T(S) denote the observed aggregate. Let T(1) ≤ ··· ≤ T(K+1) denote the order statistics of the combined set {Tobs,T(1),...,T(K)} (with ties allowed). We define

ταagg := T(⌈(1−α)(K+1)⌉), (15)

where ⌈(1 − α)(K + 1)⌉ is the (1 − α)-quantile of the (K + 1)-sized multiset. We report the right-tail permutation p-value

1 + #{k ∈ {1,...,K} : T(k) ≥ Tobs} K + 1

, (16)

pagg =

By the same exchangeability argument as for scalar calibration, pagg is super-uniform under H0 (see Proposition C.4).

Calibrated score (aggregate case). For bounded similarities with maximum smax (often smax = 1), we report a max-preserving calibrated aggregate

Tobs − ταagg smax − ταagg

, 0 . (17)

Tcal = max

This score satisfies Tcal = 0 when Tobs ≤ ταagg and Tcal = 1 when Tobs = smax. As above, Tcal depends on α via ταagg; we therefore report both Tcal (magnitude above null) and pagg (evidence against null), applying multiplicity correction (Holm, 1979; Benjamini & Hochberg, 1995) when many model pairs are evaluated.

#### 5.3. Summary

To compute a calibrated similarity score: (i) fix a significance level α (e.g., α = 0.05); (ii) generate K null scores by permuting sample correspondences; (iii) compute critical value τ as the ⌈(1 − α)(K + 1)⌉-th order statistic of the combined set (observed + null scores); (iv) return calibrated score, either scal or Tcal.

Use scalar calibration (Section 5.1) when comparing a single pair of representations. Use aggregation-aware calibration (Section 5.2) when reporting a summary statistic (e.g., maximum) over multiple layer pairs. Appendix D provides pseudocode for both procedures.

### 6. Experiments

We quantify the effects of the width and depth confounders in controlled synthetic experiments and show that our calibration framework effectively removes them. We then revisit the Platonic Representation Hypothesis using our calibration framework, assessing which convergence trends remain robust after controlling for these confounding factors.

#### 6.1. Null-calibration removes width confounder

We validate that our calibration eliminates width-related inflation of similarity across metrics, regimes, and noise distributions, without metric-specific derivations.

We design controlled synthetic experiments as follows. Under H0, we draw X,Y ∈ Rn×d independently from Gaussian and heavy-tailed (Student-t, Laplace) distributions. We vary the number of samples n ∈ {128,256,512,1024,2048,4096} and the dimension d ∈ {128,256,512,1024,2048}. Under H1, we inject a shared low-rank signal component and vary the signal-to-noise ratio. We evaluate representative metrics spanning three families. For spectral similarity, we use linear and RBF CKA, as well as CCA/SVCCA/PWCCA; for neighborhood similarity, we use mKNN (with k = 10); and for geometric similarities, we use RSA and Procrustes. Figure 3 reports a subset of these metrics for readability; additional metrics are reported in Appendix E.7. For calibration, we use K = 200 permutations with α = 0.05.

Under H0, uncalibrated scores are systematically inflated, while our calibrated scores stay at zero across settings (Figure 3). This confirms that the similarity scores of wider models can arise purely from high-dimensional finite-sample effects, and our calibration removes this spurious baseline. Importantly, the magnitude of the null baseline is metric-dependent, consistent with our theory: CKA’s baseline scales as O(d/n) (Proposition 4.1), while mKNN’s baseline scales as O(k/n) (Proposition 4.2). Intuitively,

Uncalibrated

| |
|---|
|[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>|
| |
| |
|[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]|
| |

| |
|---|
|[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]|
| |
|Calibrated|
|[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]|
| |

| |
|---|
|[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]|
| |
| |
|[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]|
| |

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Figure 3. Calibration eliminates spurious similarity across metrics. Raw scores (top) drift with d/n; calibrated scores (bottom) collapse to zero. Results for heavy-tailed distributions and additional metrics are in Appendix E.7.

mKNN compares local neighborhood overlap at a fixed k, thus only comparing relationships instead of local distances, making its null baseline insensitive to representation width d, which explains the order-of-magnitude gap observed in raw scores. The same pattern holds for heavy-tailed noise (Appendix E.7).

The same width inflation arises under a genuine shared signal, and calibration corrects it there as well, as we show analytically (Proposition C.5) and on real networks at fixed n (Appendix E.5).

Next, we verify the statistical guarantees of our empirical null calibration. For Type-I error control, rejection rates stay at or below the nominal α = 0.05 across (n,d/n) configurations (Figure 4a). Crucially, our calibration does not sacrifice sensitivity to real alignment: detection rates increase rapidly with signal strength (Figure 4b). Overall, our calibration preserves signal structure: in the high-signal regime, raw and calibrated scores show the same pattern, while in the low-signal regime, calibration correctly gates scores to zero (Appendix E.2).

Furthermore, we verify that our empirical calibration closely matches existing analytical bias corrections for CKA (Murphy et al., 2024), recovering the width correction without metric-specific derivation (Appendix E.4).

Additionally, we perform ablations on different noise distributions used in the synthetic experiments (Appendix E.1), different calibration approaches (Appendix E.3), and an ablation on the influence of the number of permutations K used for calibration (Appendix E.6).

| | | | | | |α|= 0.05|
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

1.0

0.08

Power(detectionrate)

0.8

0.07

TypeIerrorrate

0.06

0.6

0.05

0.4

0.04

0.2

0.03

0.02

0.0

0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00

1.0 1.5 2.0 2.5 3.0 3.5 4.0

d/n

Signal strength

CKA (linear) CKA (RBF) mKNN RSA

Figure 4. Statistical guarantees. (Left) Type-I error stays at or below α across configurations. (Right) Power increases rapidly with signal strength; calibration does not sacrifice sensitivity.

#### 6.2. Null-calibration removes depth confounder

We validate that our aggregation-aware null-calibration eliminates the depth confounder. To build a controlled synthetic setting, we construct two synthetic models, A and B, each with L layers. Under H0, we sample layer representations {Xℓ}Lℓ=1 and {Yℓ′}Lℓ′=1, where each Xℓ,Yℓ′ ∈ Rn×d has i.i.d. N(0,1) entries (independent across layers and between models), using d/n = 8 to match the upper range of the Platonic Representation Hypothesis setting. We then compute the layerwise similarity matrix Sℓ,ℓ′ =

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

0.6

AlignmenttoDINOv2

small base large giant

base large huge

AlignmenttoCLIP

0.6

0.5

0.5

0.4

0.3

0.4

0.2

0.3

0.1

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(a) CKA RBF: Global spectral alignment.

|sm ba lar|all<br><br>se ge| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
|gia|nt| | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

AlignmenttoDINOv2

base large huge

0.16

0.20

AlignmenttoCLIP

0.18

0.14

0.16

0.12

0.14

0.10

0.12

0.08

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(b) mKNN: Local neighborhood overlap.

- Figure 5. Revisiting the Platonic Representation Hypothesis. Models are ranked according to their language performance (Huh et al., 2024). Solid lines connect the models within the same family, while semi-transparent lines connect the models across different families. (a) Global spectral metrics lose their convergence trend; calibrated scores show no systematic increase with scale. (b) Local neighborhood metrics keep their trend even after calibration. Full results for all vision families and metrics in Appendix E.8.

CKA(Xℓ,Yℓ′) and summarize it with standard aggregates. The uncalibrated max-aggregated scores inflate with layer count even under H0 (Figure 6): raw max-scores are systematically higher at L = 128 than at L = 2, despite no true signal. Our aggregation-aware calibration eliminates this bias: calibrated aggregates remain stable regardless of depth. We further show that naively calibrating each scalar comparison still leads to inflation, highlighting the importance of calibrating the final statistic. Furthermore, since deeper models tend to be wider as well, raw comparisons are doubly confounded.

0 20 40 60 80 100 120

Number of layers

0.890

0.895

0.900

0.905

0.910

0.915

Uncalibratedscore

| | | | |ent|ry-wise|calibrat|ion|
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0 20 40 60 80 100 120

Number of layers

0.000

0.025

0.050

0.075

0.100

0.125

Calibratedscore

max row-max mean col-max mean top-5 mean top-10 mean

- Figure 6. Aggregation-aware calibration removes depth confounding. Raw max-aggregates of linear CKA scores inflate with layer count under the null; calibrated aggregates are stable and show that naive entry-wise calibration still leads to inflation.

families (BLOOM, OpenLLaMA, LLaMA) and five vision model families (ImageNet-21K, MAE, DINOv2, CLIP, CLIP-finetuned) across multiple scales. As in Huh et al. (2024), we extract per-layer representations using the class token for vision models and mean pooling over tokens for language models. This yields 204 vision–language model pairs spanning d/n ∈ [0.19,8]. For each pair, we compute layer-wise similarity and report the maximum across layers, as in the original work. We evaluate both global spectral metrics (CKA linear/RBF) and local neighborhood metrics (mKNN, cycle-kNN, CKNNA). Following Huh et al. (2024), we evaluate mKNN, cycle-kNN, and CKNNA with k = 10. We further apply Benjamini-Hochberg FDR correction (Benjamini & Hochberg, 1995) to control for multiple comparisons across model pairs.

For the global similarity, we find that uncalibrated CKA scores increase with model scale (dotted lines in Figure 5a), reproducing the trend interpreted as evidence of cross-modal convergence (Huh et al., 2024). However, this trend disappears after our calibration (solid lines): calibrated CKA shows no systematic increase with model size. This indicates that global convergence in uncalibrated CKA is largely attributable to width and depth confounders rather than a genuine increase in representational similarity. The same holds for the other global metrics, including shape- and geometry-based ones: SVCCA, the RV coefficient, and Procrustes distance all lose their apparent scaling trend after calibration (Appendix E.8).

#### 6.3. Revisiting the Platonic Representation Hypothesis

A central claim behind the Platonic Representation Hypothesis is that, as models become more capable, their representations begin to converge across modalities. We revisit this claim through our calibration framework to determine whether the observed alignment reflects genuine shared representation structure or instead arises from width and depth confounders.

In contrast, for the local similarity, evidence of cross-modal convergence remains strong for neighborhood-based metrics even under our calibration (Figure 5b). The same qualitative conclusion holds for other neighborhood-based measures (cycle-kNN and CKNNA; Appendix E.8) and different choices of α (Appendix E.11). Further analysis (Appendix E.10) reveals that models converge in local neighbor-

We follow the experimental protocol of Huh et al. (2024) using n = 1024 image–text pairs (WIT; Srinivasan et al. (2021)) and embeddings from three language model

hood structure: models increasingly agree on which points are neighbors, but do not agree on the pairwise distances, since CKA-RBF with a small bandwidth shows no alignment after calibration.

CKA and mKNN capture different invariances, so their raw magnitudes are not directly comparable. We therefore compare, for each metric separately, how strongly its score tracks language-model capability before and after calibration (its Pearson correlation with the model ranking). After calibration this correlation collapses for global metrics (for linear CKA it falls from 0.86 to 0.45, and for Procrustes distance from 0.89 to 0.39) but is essentially unchanged for local ones (mKNN stays near 0.85 and CKNNA near 0.87). Appendix E.8 reports all metrics.

To test whether these findings generalize beyond images and text, we extend our analysis to video–language alignment following Zhu et al. (2026). We compare video encoders (VideoMAE small/base/large/huge, fine-tuned on Kinetics) against the same language model families. Consistent with our previous findings, the global similarity (CKA) shows no trend with model capacity (Figure 7). In contrast, for local similarity (mKNN), a clear scaling trend emerges with VideoMAE-Large/Huge, whereas smaller video encoders appear to act as a bottleneck, limiting alignment regardless of language model size. This confirms that local neighborhood convergence extends to video–language alignment, provided that representations are sufficiently powerful. Appendix E.9 further compares non-finetuned VideoMAE versions and a variety of image models at the frame level on the same dataset, showing the same trend.

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

0.6

0.175

###### CKARBF

0.150

mKNN

0.4

0.125

0.2

0.100

560M1B11B73B7B1 3B7B13B 7B13B30B65B

560M1B11B73B7B1 3B7B13B 7B13B30B65B

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

small base large huge calibrated uncalibrated

- Figure 7. Video–language alignment. Extending the Platonic Representation Hypothesis analysis to video encoders (VideoMAE small/base/large/huge) yields the same pattern: calibrated CKA drops substantially while mKNN retains alignment.

Taken together, these results suggest a refined version of the Platonic Representation Hypothesis. After calibration, we find little evidence that representations converge in global spectral structure as models scale, at least under the considered setting. What reliably persists is local geometric alignment: different models preserve similar neighborhood relationships among inputs. We therefore propose the

alternative Aristotelian Representation Hypothesis: As models become capable, their representations converge to shared local neighborhood relationships.

### 7. Conclusion

Representational similarity metrics are widely used to study learned features, but their interpretation is systematically distorted by two artifacts: width-dependent null baselines and depth-dependent selection inflation. We introduced a unified null-calibration framework that corrects both, turning similarity scores into effect sizes with principled zero points and valid p-values. Applying our framework to the Platonic Representation Hypothesis reveals that previously reported global spectral convergence is largely confounded by width and depth, whereas local neighborhood alignment remains significant, motivating an Aristotelian Representation Hypothesis.

Relationship to the Platonic hypothesis. The Aristotelian hypothesis refines rather than refutes the Platonic one. Global convergence implies local convergence, because matching the full geometry preserves neighborhoods, but the converse does not hold. The Platonic hypothesis therefore implies the Aristotelian one, which is a weaker criterion. Our experiments support this local form and, after calibration, find little evidence for the global one. Prior reports of convergence thus remain compatible with our analysis, and the Aristotelian hypothesis offers a more precise lens on what converges across models and modalities.

Limitations and outlook. Our conclusions come with two caveats. First, representational similarity has no groundtruth scale, so we report the presence or absence of calibrated evidence for convergence rather than proving it. Second, our guarantees assume exchangeability, so grouped or clustered samples require restricted permutations that preserve their dependence structure. Why representations converge in local neighborhoods but not in global geometry remains the key open question.

### Acknowledgements

We thank Artyom Gadetsky, Siba Smarak Panigrahi, Debajyoti Dasgupta, David Fr¨uhbuss, Shin Matsushima, Rishubh Singh, Adriana Moreno Castan, and Gioele La Manno for their valuable suggestions, which helped improve the manuscript. We are especially grateful to Simone Lionetti for additional input and support. We gratefully acknowledge the support of the Swiss National Science Foundation (SNSF) starting grant TMSGI2 226252/1, SNSF grant IC00I0 231922, SNSF grant 10.004.411, and the Swiss AI Initiative Large Call #32. M.B. is a CIFAR Fellow in the Multiscale Human Program.

### Impact statement

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

Chun, C., Canatar, A., Chung, S., and Lee, D. D. Estimating Neural Representation Alignment from Sparsely Sampled Inputs and Features. arXiv preprint arXiv:2502.15104, 2025.

Cram´er, H. Mathematical methods of statistics. 1999.

### References

Aggarwal, C. C., Hinneburg, A., and Keim, D. A. On the surprising behavior of distance metrics in high dimensional space. In International Conference on Database Theory. Springer, 2001.

Aristotle. Categories, ca. 350 B.C.E.

Benjamini, Y. and Hochberg, Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B (Methodological), 1995.

Beyer, K., Goldstein, J., Ramakrishnan, R., and Shaft, U. When is “nearest neighbor” meaningful? In International Conference on Database Theory. Springer, 1999.

Bo, Y., Soni, A., Srivastava, S., and Khosla, M. Evaluating representational similarity measures from the lens of functional correspondence. arXiv preprint arXiv:2411.14633, 2024.

Bolya, D., Huang, P.-Y., Sun, P., Cho, J. H., Madotto, A., Wei, C., Ma, T., Zhi, J., Rajasegaran, J., Rasheed, H. A., Wang, J., Monteiro, M., Xu, H., Dong, S., Ravi, N., Li, S.W., Dollar, P., and Feichtenhofer, C. Perception Encoder: The best visual embeddings are not at the output of the network. Advances in Neural Information Processing Systems, 2025.

Bonferroni, C. Teoria statistica delle classi e calcolo delle probabilit`a. Pubblicazioni del R Istituto Superiore di Scienze Economiche e Commerciali di Firenze, 1936.

Cai, M. B., Schuck, N. W., Pillow, J. W., and Niv, Y. Representational structure or task structure? Bias in neural representational similarity analysis and a Bayesian method for reducing bias. PLoS Computational Biology, 2019.

Cho, J. H., Madotto, A., Mavroudi, E., Afouras, T., Nagarajan, T., Maaz, M., Song, Y., Ma, T., Hu, S., Jain, S., Martin, M., Wang, H., Rasheed, H. A., Sun, P., Huang, P.-Y., Bolya, D., Ravi, N., Jain, S., Stark, T., Moon, S., Damavandi, B., Lee, V., Westbury, A., Khan, S., Kraehenbuehl, P., Dollar, P., Torresani, L., Grauman, K., and Feichtenhofer, C. PerceptionLM: Open-Access Data and Models for Detailed Visual Understanding. Advances in Neural Information Processing Systems, 2025.

Cui, T., Kumar, Y., Marttinen, P., and Kaski, S. Deconfounded representation similarity for comparison of neural networks. Advances in Neural Information Processing Systems, 2022.

Diedrichsen, J., Berlot, E., Mur, M., Sch¨utt, H. H., Shahbazi, M., and Kriegeskorte, N. Comparing representational geometries using whitened unbiased-distance-matrix similarity. Neurons, Behavior, Data Analysis, and Theory, 2021.

Ding, F., Denain, J.-S., and Steinhardt, J. Grounding representation similarity through statistical testing. Advances in Neural Information Processing Systems, 2021.

Embrechts, P., Kl¨uppelberg, C., and Mikosch, T. Modelling Extremal Events: for Insurance and Finance. Stochastic Modelling and Applied Probability. 2013.

Good, P. Permutation, parametric and bootstrap tests of hypotheses. 2005.

Gr¨oger, F., Wen, S., Le, H., and Brbic, M. With limited data for multimodal alignment, let the STRUCTURE guide you. Advances in Neural Information Processing Systems, 2025.

Harvey, S. E., Lipshutz, D., and Williams, A. H. What Representational Similarity Measures Imply about Decodable Information. In Proceedings of UniReps: the Second Edition of the Workshop on Unifying Representations in Neural Models. PMLR, 2024.

Holm, S. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 1979.

Hotelling, H. Relations between two sets of variates. In Breakthroughs in Statistics: Methodology and Distribution. 1992.

Huh, M., Cheung, B., Wang, T., and Isola, P. Position: The platonic representation hypothesis. In International Conference on Machine Learning, 2024.

Kapoor, C., Srivastava, S., and Khosla, M. Bridging Critical Gaps in Convergent Learning: How Representational Alignment Evolves Across Layers, Training, and Distribution Shifts. Advances in Neural Information Processing Systems, 2025.

Klabunde, M., Schumacher, T., Strohmaier, M., and Lemmerich, F. Similarity of neural network models: A survey of functional and representational measures. ACM Computing Surveys, 2025a.

Klabunde, M., Wald, T., Schumacher, T., Maier-Hein, K., Strohmaier, M., and Lemmerich, F. ReSi: A Comprehensive Benchmark for Representational Similarity Measures. In International Conference on Learning Representations, 2025b.

Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. Similarity of neural network representations revisited. In International Conference on Machine Learning, 2019.

Kriegeskorte, N., Mur, M., and Bandettini, P. A. Representational similarity analysis–connecting the branches of systems neuroscience. Frontiers in Systems Neuroscience, 2008.

Lehmann, E. L. and Romano, J. P. Testing statistical hypotheses. 2005.

Livan, G., Novaes, M., and Vivo, P. Introduction to Random Matrices: Theory and Practice. SpringerBriefs in Mathematical Physics. 2018.

Maniparambil, M., Akshulakov, R., Djilali, Y. A. D., El Amine Seddik, M., Narayan, S., Mangalam, K., and O’Connor, N. E. Do Vision and Language Encoders Represent the World Similarly? In Conference on Computer Vision and Pattern Recognition, 2024.

Marcos-Manch´on, P. and Fuentemilla, L. Convergent transformations of visual representation in brains and models. arXiv preprint arXiv:2507.13941, 2025.

Morcos, A., Raghu, M., and Bengio, S. Insights on representational similarity in neural networks with canonical correlation. Advances in Neural Information Processing Systems, 2018.

M¨uller, R. R. A random matrix model of communication via antenna arrays. IEEE Transactions on information theory, 2002.

Murphy, A., Zylberberg, J., and Fyshe, A. Correcting biased centered kernel alignment measures in biological and artificial neural networks. arXiv preprint arXiv:2405.01012, 2024.

Neyshabur, B., Sedghi, H., and Zhang, C. What is being transferred in transfer learning? Advances in Neural Information Processing Systems, 2020.

Nguyen, T., Raghu, M., and Kornblith, S. Do Wide and Deep Networks Learn the Same Things? Uncovering How Neural Network Representations Vary with Width and Depth.

In International Conference on Learning Representations, 2021.

Nichols, T. E. and Holmes, A. P. Nonparametric permutation tests for functional neuroimaging: a primer with examples. Human Brain Mapping, 2002.

Phipson, B. and Smyth, G. K. Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn. Statistical Applications in Genetics & Molecular Biology, 2010.

Raghu, M., Gilmer, J., Yosinski, J., and Sohl-Dickstein, J. SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability. In Advances in Neural Information Processing Systems, 2017.

Raugel, J., Szafraniec, M., Vo, H. V., Couprie, C., Labatut, P., Bojanowski, P., Wyart, V., and King, J.-R. Disentangling the factors of convergence between brains and computer vision models. arXiv preprint arXiv:2508.18226, 2025.

Robert, P. and Escoufier, Y. A unifying tool for linear multivariate statistical methods: the RV-coefficient. Journal of the Royal Statistical Society: Series C (Applied Statistics), 1976.

Schaeffer, R., Khona, M., Chandra, S., Ostrow, M., Miranda, B., and Koyejo, S. Position: Maximizing Neural Regression Scores May Not Identify Good Models of the Brain. In NeurIPS 2024 Workshop on Unifying Representations in Neural Models (UniReps), 2024.

Schrimpf, M., Kubilius, J., Hong, H., Majaj, N. J., Rajalingham, R., Issa, E. B., Kar, K., Bashivan, P., Prescott-Roy, J., Geiger, F., et al. Brain-score: Which artificial neural network for object recognition is most brain-like? BioRxiv, 2018.

Smilde, A. K., Kiers, H. A., Bijlsma, S., Rubingh, C., and Van Erk, M. Matrix correlations for high-dimensional data: the modified RV-coefficient. Bioinformatics, 2009.

Song, L., Smola, A., Gretton, A., Bedo, J., and Borgwardt, K. Feature Selection via Dependence Maximization. Journal of Machine Learning Research, 2012.

Soni, A., Srivastava, S., Kording, K., and Khosla, M. Conclusions about Neural Network to Brain Alignment are Profoundly Impacted by the Similarity Measure. bioRxiv, 2024.

Srinivasan, K., Raman, K., Chen, J., Bendersky, M., and Najork, M. Wit: Wikipedia-based image text dataset for multimodal multilingual machine learning. In International ACM SIGIR Conference on Research and Development in Information Retrieval, 2021.

Tjandrasuwita, M., Ekbote, C., Ziyin, L., and Liang, P. P. Understanding the Emergence of Multimodal Representation Alignment. In International Conference on Machine Learning, 2025.

Tong, Z., Song, Y., Wang, J., and Wang, L. VideoMAE: Masked autoencoders are data-efficient learners for selfsupervised video pre-training. Advances in Neural Information Processing Systems, 2022.

Wachter, K. W. The strong limits of random matrix spectra for sample matrices of independent elements. The Annals of Probability, 1978.

Weenink, D. Canonical correlation analysis. Proceedings of the Institute of Phonetic Sciences of the University of Amsterdam, 2003.

Westfall, P. H. and Young, S. S. Resampling-based multiple testing: Examples and methods for p-value adjustment. 1993.

Williams, A. H., Kunz, E., Kornblith, S., and Linderman, S. W. Generalized Shape Metrics on Neural Representations. In Advances in Neural Information Processing Systems, 2021.

Zhu, T., Han, T., Guibas, L., P˘atr˘aucean, V., and Ovsjanikov, M. Dynamic Reflections: Probing Video Representations with Text Alignment. In International Conference on Learning Representations, 2026.

### A. Existing calibration approaches for representational similarity metrics

Table 1. Comparison of prior works. Y=yes, N=no, P=partial/indirect. “Debias” indicates an explicit null correction of the reported similarity. “Bounded” indicates whether the corrected score preserves an interpretable upper bound (e.g., 1 for perfect alignment). “Agg-aware” indicates calibration of selection-based aggregates (e.g., max over layer pairs).

Ref Metric(s) Debias? Bounded? Agg-aware?

Murphy et al. (2024) CKA Y N N Chun et al. (2025) CKA Y N N Cui et al. (2022) RSA/CKA P N N Diedrichsen et al. (2021) RSA (cv/WUC) Y P N Cai et al. (2019) RSA (Bayes) P N N Smilde et al. (2009) RV / adj. RV Y N N

Ours Any bounded metric Y Y Y

### B. Metrics and score definitions

This appendix gives the definitions of the similarity metrics s(X,Y) used throughout the paper. The main text focuses on the calibration procedure (Sections 5.1 and 5.2). Here we provide concrete instantiations of the metrics referenced in Section 3 and Section 6.

#### B.1. Preprocessing and basic notation

Let X ∈ Rn×d

y denote row-aligned representations evaluated on the same n inputs. We use the centering matrix

x and Y ∈ Rn×d

1 n n

⊤ n , (18)

H = In −

where In ∈ Rn×n is the identity matrix and n ∈ Rn is the all-ones vector. We define row-centered representations Xc = HX and Yc = HY. Unless stated otherwise, similarities are computed on centered representations.

#### B.2. Raw similarity metrics

This section provides formal definitions of the similarity metrics used throughout the paper. In the main text, we primarily use CKA (linear and RBF kernel) (Kornblith et al., 2019), RSA (Kriegeskorte et al., 2008), and mutual k-NN (Huh et al., 2024) as representative metrics from the spectral, geometric, and neighborhood families, respectively. Additional metrics (SVCCA, PWCCA, cycle-kNN, CKNNA, RV coefficient, Procrustes) are included for completeness and used in supplementary experiments.

- B.2.1. SPECTRAL METRICS

Linear Centered Kernel Alignment (CKA). Linear CKA (Kornblith et al., 2019) can be written as a normalized Frobenius energy of the sample cross-covariance operator. With Xc,Yc as above, define the sample (cross-)covariances

1 n − 1

1 n − 1

1 n − 1

X⊤c Xc, ΣY Y :=

Yc⊤Yc, C := ΣXY :=

#### X⊤c Yc. (19)

ΣXX :=

The biased linear Hilbert-Schmidt Independence Criterion (HSIC) energy equals ∥ C∥2F. The commonly used linear CKA normalization can be written as

= ∥X⊤c Yc∥2F ∥X⊤c Xc∥F ∥Yc⊤Yc∥F

CKAlin(X,Y) = ∥ C∥2F ∥ ΣXX∥F ∥ ΣY Y ∥F

∈ [0,1], (20)

where the second equality follows by cancellation of common 1/(n − 1) factors.

What it measures: linear CKA treats two representations as similar when pairs of points with high inner product in one space also have high inner product in the other, i.e., when their global second-order geometry agrees. It is invariant to rotations and isotropic rescaling, but not to general invertible linear maps.

Kernel Centered Kernel Alignment. Kernel CKA (Kornblith et al., 2019) generalizes linear CKA by replacing dot products with kernel functions. Let kX : Rd

→ R be positive semidefinite kernel functions (e.g., RBF kernel kX(x,x′) = exp(−∥x − x′∥2/2σ2)). Let KX ∈ Rn×n and KY ∈ Rn×n be Gram matrices with entries (KX)ij = kX(xi,xj) and (KY )ij = kY (yi,yj). Let KX = HKXH and KY = HKY H denote centered Gram matrices. Kernel CKA is defined as:

##### → R and kY : Rd

##### × Rd

##### × Rd

x

x

y

y

X,kY (X,Y) = ⟨ KX, KY ⟩F ∥ KX∥F ∥ KY ∥F

. (21)

CKAk

where ⟨A,B⟩F = tr(A⊤B). With positive semidefinite kernels and the biased HSIC estimator, the numerator is nonnegative, and kernel CKA typically lies in [0,1].

What it measures: kernel CKA treats representations as similar when their kernel-induced similarity patterns over points agree. With an RBF kernel of small bandwidth, this emphasizes local, nonlinear similarity structure rather than global linear geometry.

Unbiased Centered Kernel Alignment. The biased HSIC estimator can yield inflated similarity scores at finite sample sizes. Song et al. (2012) derived an unbiased HSIC estimator by recognizing that HSIC can be formulated as a U-statistic. Following Kornblith et al. (2019), we substitute the unbiased estimator into the CKA formula. Let K˚X denote the Gram matrix KX with its diagonal entries set to zero (and likewise K˚Y ). The unbiased HSIC estimator is:

1 n(n − 3)

HSICu(KX,KY ) =

1⊤K˚X1 · 1⊤K˚Y 1

tr(K˚XK˚Y ) +

(n − 1)(n − 2) −

2 n − 2

1⊤K˚XK˚Y 1 . (22)

Unbiased CKA replaces both numerator and denominator of Equation (21) with this estimator. Unlike the biased version, unbiased CKA can take small negative values at finite n. What it measures: the same agreement of similarity patterns as kernel CKA, but with the finite-sample inflation of HSIC removed, so a value near zero indicates no more agreement than expected by chance.

Canonical Correlation Analysis (CCA)-based similarity. CCA (Weenink, 2003) measures linear subspace alignment. The sample canonical correlations {ρi}ri=1 (with r = rank( ΣXY )) are the singular values of the whitened cross-covariance operator

- 1

- 2

- 1

- 2

TCCA = Σ−

XX ΣXY Σ−

Y Y . (23)

Common scalar summaries include the mean canonical correlation 1r ri=1 ρi or a weighted average as used in SVCCA (Raghu et al., 2017) and PWCCA (Morcos et al., 2018).

What it measures: CCA treats two representations as similar when one can be mapped onto the other by an invertible linear transformation, i.e., when they span the same linear subspace. It ignores rotations, scaling, and other invertible linear changes within each space.

Singular Vector Canonical Correlation Analysis (SVCCA). SVCCA (Raghu et al., 2017) combines dimensionality reduction via singular value decomposition (SVD) with CCA. First, truncated SVD is applied to each representation to retain the top principal components, yielding X′ ∈ Rn×p and Y′ ∈ Rn×q. Then CCA is applied to the reduced representations, yielding canonical correlations {ρi}ri=1. The SVCCA similarity is the mean canonical correlation:

r

1 r

ρi. (24)

SVCCA(X,Y) =

i=1

What it measures: SVCCA treats representations as similar when their high-variance subspaces are linearly aligned, discarding low-variance (noise) directions before measuring linear correspondence.

Projection Weighted Canonical Correlation Analysis (PWCCA). PWCCA (Morcos et al., 2018) improves upon SVCCA by weighting canonical correlations according to their importance in explaining the original representations. Let hXi and hYi denote the i-th canonical variables (projections onto canonical directions). The weight for the i-th canonical

correlation is the overall magnitude of the corresponding canonical variable across the dataset:

n

(hXi )m = ∥hXi ∥1, (25)

αi =

m=1

the ℓ1 norm of the i-th canonical variable over the n samples, following the reference implementation of Morcos et al.

(2018). The PWCCA similarity is the weighted mean:

r i=1 αiρi

. (26)

PWCCA(X,Y) =

r i=1 αi

This weighting ensures that canonical correlations corresponding to principal directions receive higher weight than those corresponding to noise dimensions. What it measures: like CCA, PWCCA treats representations as similar when they linearly correspond, but it counts agreement along high-variance directions more, so two representations are similar when their dominant directions align.

RV coefficient. The RV (“Relation between two sets of Variables”) coefficient (Robert & Escoufier, 1976; Smilde et al., 2009) is a multivariate generalization of the squared Pearson correlation. It measures the similarity between two configuration matrices via their inner-product (Gram) matrices. Let WX = XcX⊤c and WY = YcYc⊤ be the inner-product (Gram) matrices of the centered representations Xc,Yc. The RV coefficient is:

tr(WXWY ) tr(WX2 ) tr(WY2 ) ∈ [0,1]. (27)

RV(X,Y) =

What it measures: the RV coefficient treats representations as similar when their point-by-point inner-product (Gram) matrices match. Like linear CKA, it captures global second-order geometry and is invariant to rotation but sensitive to scaling. Computed on centered representations (our default convention), the RV coefficient coincides exactly with linear CKA; we list it separately as it arises from a different historical motivation and include it for completeness.

- B.2.2. GEOMETRIC METRICS

Representational Similarity Analysis (RSA) via Spearman correlation of dissimilarity matrices. RSA (Kriegeskorte et al., 2008) compares the geometry induced by pairwise dissimilarities. Let δ(·,·) be a dissimilarity on representation vectors (e.g., correlation distance δ(u,v) = 1−corr(u,v), cosine distance). Define Representational Dissimilarity Matrices (RDMs)

##### (DX)ij = δ(xi,xj), (DY )ij = δ(yi,yj), (28)

and let vec△(D) ∈ Rn(n−1)/2 denote vectorization of the strict upper triangle. RSA is then computed as a rank correlation between the two RDM vectors:

RSA(X,Y) = ρS(vec△(DX), vec△(DY )), (29) where Spearman’s ρ can be expressed as Pearson correlation of ranks,

##### ρS(u,v) = corr(rank(u), rank(v)). (30)

What it measures: RSA treats two representations as similar when they rank pairs of points in the same order of (dis)similarity. It compares the relational geometry of distances rather than the coordinates, and depends only on the pattern of pairwise dissimilarities.

Procrustes distance. The orthogonal Procrustes distance (Williams et al., 2021) measures the minimal Euclidean distance between two representations after optimal orthogonal alignment. Assuming dx = dy = d, the optimal orthogonal matrix Q∗ ∈ O(d) is:

Q∗ = argmin Q∈O(d)

∥X − YQ∥2F, (31) which has the closed-form solution Q∗ = VU⊤ where UΣV⊤ = X⊤Y is the SVD. The Procrustes distance is:

##### dProc(X,Y) = ∥X − YQ∗∥F. (32)

We convert this distance to a similarity score sProc = 1 − dProc/∥Yc∥F, evaluated on centered representations, where Yc denotes the centered Y. It equals 1 at perfect orthogonal alignment (dProc = 0) and can be negative for poorly aligned pairs. What it measures: Procrustes treats two representations as similar when one can be rotated and reflected onto the other with small residual. It compares absolute geometry up to rigid motion and, unlike neighborhood metrics, is sensitive to the actual pairwise distances.

- B.2.3. NEIGHBORHOOD METRICS

Mutual k-Nearest Neighbors (mKNN). mKNN (Huh et al., 2024) focuses on local topology. For each anchor sample i, define the set of its k nearest neighbors according to a distance measure dist(·,·) in X and Y,

##### NX(i) = KNNk(i;X), NY(i) = KNNk(i;Y), (33)

where KNNk(i;X) denotes the indices of the k samples (excluding i) that minimize dist(xi,xj). mKNN is then defined as the average fraction of shared neighbors:

n

|NX(i) ∩ NY(i)| k ∈ [0,1]. (34)

1 n

mKNNk(X,Y) =

i=1

What it measures: mKNN treats two representations as similar when each point keeps the same set of nearest neighbors in both spaces. It captures local topology, depends only on the rank order of distances, and is therefore invariant to any transformation that preserves k-nearest-neighbor sets, ignoring exact distances and global geometry.

Cycle-kNN (bidirectional k-NN). While mKNN measures one-directional neighborhood overlap, cycle-kNN enforces a round-trip consistency between the two spaces (Huh et al., 2024). An anchor i counts as consistent if at least one of its nearest neighbors j in Y has i among its nearest neighbors in X:

 i ∈

  ∈ [0,1]. (35)

n

1 n

cycle-kNNk(X,Y) =

NX(j)

i=1

j∈NY(i)

This is stricter than mKNN, as it requires neighbor relations to be mutually recognized across the two spaces. What it measures: cycle-kNN treats points as similar only when neighborhood membership is reciprocated across spaces, a stricter local-topology criterion than mKNN.

CKA with Neighborhood Alignment (CKNNA). CKNNA (Huh et al., 2024) combines the kernel formulation of CKA with local neighborhood structure, restricting the kernel interaction to mutual k-nearest-neighbor edges. Let KX = XX⊤ and KY = YY⊤ be the (linear) Gram matrices, and let MX,MY ∈ {0,1}n×n be the k-NN indicator matrices with (MX)ij = [j ∈ N X(i)] and (MY )ij = [j ∈ N Y(i)]. Let A = MX ⊙ MY retain only edges that are k-NN in both spaces, where ⊙ is the Hadamard product. CKNNA aligns the neighbor-masked Gram matrices with CKA’s normalization:

CKNNAk(X,Y) = ⟨H(A ⊙ KX)H, H(A ⊙ KY )H⟩F ∥H(MX ⊙ KX)H∥F ∥H(MY ⊙ KY )H∥F

. (36)

What it measures: CKNNA treats representations as similar when kernel values on shared (mutual) nearest-neighbor edges agree, combining CKA’s normalization with a purely local notion of neighborhood structure.

- C. Theoretical Derivations In this section, we provide the theoretical justification for the confounding factors identified in Section 4.

- C.1. Permutation validity, super-uniformity, and gating This section formalizes the finite-sample validity of permutation calibration. Definition C.1 (Super-uniformity). A p-value p is super-uniform under H0 if for all t ∈ [0,1],

(p ≤ t) ≤ t. (37) Equivalently, p-values under H0 are stochastically larger than Unif(0,1), which is sufficient for valid Type-I error control.

##### PH

0

Lemma C.2 (Permutation p-values are super-uniform). Under Assumption 3.1, the permutation p-value in Equation (10) satisfies super-uniformity: PH

(p ≤ α) ≤ α for all α ∈ [0,1] (finite-sample validity).

0

Proof of Lemma C.2. Let sobs = s(X,Y) be the observed statistic and let s(k) = s(X,πk(Y)) for k = 1,...,K be the statistics computed on permuted pairings. Under Assumption 3.1, the vector (sobs,s(1),...,s(K)) is exchangeable: its joint distribution is invariant to permutations of the indices. Consider the (upper) rank

R = 1 + #{k ∈ {1,...,K} : s(k) ≥ sobs} ∈ {1,...,K + 1}. (38)

If the scores are almost surely distinct, exchangeability implies that the rank of sobs among {sobs,s(1),...,s(K)} is uniform on {1,...,K + 1}. With possible ties, the add-one p-value of Phipson & Smyth (2010),

R K + 1

, (39)

p =

is conservative, implying PH

(p ≤ α) ≤ α for all α ∈ [0,1].

0

| |
|---|

Proof of Corollary 5.1. Let sobs = s(X,Y) and s(k) = s(X,πk(Y)) for k = 1,...,K. Under Assumption 3.1, the vector (sobs,s(1),...,s(K)) is exchangeable. Let

τα := s(⌈(1−α)(K+1)⌉)

be the (1 − α)-quantile defined via the order statistic of the combined multiset {sobs,s(1),...,s(K)}. Define the (upper) rank

R = 1 + #{k ∈ {1,...,K} : s(k) ≥ sobs} ∈ {1,...,K + 1},

and the corresponding add-one p-value p = R/(K + 1). By construction of τα, the rejection event {sobs > τα} implies that sobs lies among the largest ⌊α(K + 1)⌋ values of {sobs,s(1),...,s(K)}, hence R ≤ α(K + 1) and therefore p ≤ α. By Lemma C.2, PH

(p ≤ α) ≤ α, which yields

0

##### PH

(sobs > τα) ≤ PH

(p ≤ α) ≤ α.

0

0

| |
|---|

Restricted permutations under dependence. Assumption 3.1 treats the n row pairs as exchangeable units. In practice, exchangeability can be violated even without sequential structure (e.g., grouped or clustered samples). Validity is then recovered by using restricted permutations that preserve the dependence structure (e.g., permuting within blocks or permuting block labels) and re-running under each restricted permutation.

#### C.2. Monotone invariance of rank-based calibration

The following proposition is a standard result in randomization inference; we state it here for completeness and to clarify its role in justifying the calibrated score design.

- Proposition C.3 (Monotone invariance of rank-based calibration (Lehmann & Romano, 2005)). Let g : R → R be strictly increasing. Define pg by applying Equation (10) to the transformed statistic g ◦ s using the same permutations. Then pg = p, and likewise the null percentile (the rank of sobs among the combined set) is invariant under g.

Proof. Let g be strictly increasing. For any two real numbers a,b, we have a ≥ b if and only if g(a) ≥ g(b). Therefore, for each permutation draw k,

{s (k) ≥ sobs} = {g(s (k)) ≥ g(sobs)}. (40)

Summing over k shows that the permutation rank R (and thus the add-one p-value) is unchanged by applying g to both the observed and permuted statistics. The same argument applies to the null percentile, since the ordering of samples is preserved under g.

| |
|---|

#### C.3. Post-selection inflation and aggregation-aware validity

- Proposition C.4 (Validity for aggregation-aware calibration). Let T be any measurable aggregation operator applied to a layer-wise similarity matrix S (e.g., max, row-max, top-k). If Tobs = T(S) is calibrated against the permutation null {T(S(k))}Kk=1 as in Equation (16), then the resulting pagg is super-uniform under H0.

Proof of Proposition C.4. Let T be any measurable functional of the full data (representations across all layers), producing the scalar report Tobs. Under Assumption 3.1 and consistent layer-wise permutation of sample correspondences, the vector (Tobs,T(1),...,T(K)) is exchangeable. Applying the same rank argument as in Appendix C.1 yields super-uniformity for the add-one p-value in Equation (16).

| |
|---|

#### C.4. The width confounder

This appendix provides concrete calculations that justify the width confounder using Random Matrix Theory (RMT): even under independence, interaction operators have non-trivial magnitude and spectrum when d is not negligible relative to n.

Proof of Proposition 4.1. Let X ∈ Rn×d

x and Y ∈ Rn×d

y have i.i.d. rows with mean 0, identity covariance, and xi and

yi independent. Let H = In − n1 n ⊤n be the centering matrix, so Xc = HX and Yc = HY. Since H is symmetric and idempotent (H2 = H), the sample cross-covariance is

1 n − 1

1 n − 1

X⊤c Yc =

#### X⊤HY. (41)

C =

Denote entry (a,b) as Cab. Expanding via Hij = δij − n1:

 

 . (42)

n

n

n

1 n − 1

1 n

XiaYib −

Cab =

Xia

Yjb

i=1

i=1

j=1

We compute E[ Cab2 ] using independence of xi and yj for all i,j, zero means, and identity covariance.

- Term 1: E ( i XiaYib)2 = i,j E[XiaXja]E[YibYjb]. For i ̸= j, independence across rows and zero mean give E[XiaXja] = E[Xia]E[Xja] = 0. For i = j, we have E[Xia2 ]E[Yib2] = 1. Thus E ( i XiaYib)2 = n.
- Term 2: E ( i XiaYib)( j Xja)( k Ykb) = i,j,k E[XiaXja]E[YibYkb]. This is nonzero only when i = j and i = k, yielding i 1 · 1 = n.
- Term 3: E ( i Xia)2( j Yjb)2 = E[( i Xia)2]E[( j Yjb)2] = n · n = n2.

Combining:

n2 n2

2 n · n +

1 (n − 1)2

E Cab2 =

n −

Summing over all entries:

dy

dx

E ∥ C∥2F =

a=1

b=1

1 (n − 1)2

1 n − 1

. (43)

(n − 2 + 1) =

=

E[ Cab2 ] =

dxdy n − 1

. (44)

| |
|---|

Interpretation. The null interaction energy is O(dxdy/n). In the common regime dx,dy ≍ n, the null energy is O(n) and therefore does not vanish. Since many spectral similarity metrics aggregate singular values (e.g., via ∥ C∥2F = i σi2( C)), this already explains a positive baseline under H0 and its dependence on (n,dx,dy).

Width inflation persists under genuine signal (H1). The width confounder is not specific to the null hypothesis. Even when two representations share a fixed, genuine signal, finite-sample CKA is inflated by width, because increasing d grows the diagonal and off-diagonal Gram entries at different rates and the CKA denominator does not cancel the resulting width-dependent self-similarity.

- Proposition C.5 (Width inflation under genuine signal). Fix n ≥ 2 and a target alignment ρ ∈ (0,1). For each width d, draw ui,vi,wi i.∼i.d. N(0,Id) independently across i = 1,...,n, and set

xi = √ρui + 1 − ρvi, yi = √ρui + 1 − ρwi, (45)

so that xi and yi share the latent signal ui. Then the population linear CKA equals ρ2 < 1, whereas for fixed n the sample centered linear CKA satisfies CKA(X,Y) → 1 almost surely as d → ∞. Hence the finite-sample estimate is inflated by width, with CKA(X,Y) − ρ2 → 1 − ρ2 > 0.

Proof. By independence of ui,vi,wi, the marginal covariances are Σx = Σy = Id and the cross-covariance is Σxy = Cov(xi,yi) = ρId. The population linear CKA is therefore

CKApop = ∥Σxy∥2F ∥Σx∥F ∥Σy∥F

ρ2d √

= ρ2. (46)

√

=

d

d

For the sample statistic at fixed n, write the columns of X as g1,...,gd ∈ Rn. Each entry of X is N(0,1) (since ρ + (1 − ρ) = 1), and the columns are i.i.d. across j with gj ∼ N(0,In). By the strong law of large numbers,

d

1 d

1 d

gjgj⊤ −−→a.s. E[g1g1⊤] = In. (47)

XX⊤ =

j=1

Since n is fixed, this convergence also holds in Frobenius norm, so d1HXX⊤H → H a.s.; the identical argument gives 1

dHYY⊤H → H. By scale invariance of CKA, CKA(X,Y) = ⟨d1HXX⊤H, d1HYY⊤H⟩F ∥d1HXX⊤H∥F ∥d1HYY⊤H∥F

⟨H,H⟩F ∥H∥2F

−−→a.s.

= 1. (48)

Subtracting the population value gives CKA(X,Y) − ρ2 → 1 − ρ2 > 0.

| |
|---|

Thus, at any fixed sample size n, increasing the width drives the estimated CKA toward 1 regardless of the true alignment ρ2, inflating it by up to 1 − ρ2. This width-driven gap is distinct from the O(1/n) sample-size effect that vanishes as n grows, and it is exactly what permutation calibration removes. Appendix E.5 verifies this on real pretrained networks with nonzero true similarity.

CCA-based scores. CCA, SVCCA, and PWCCA fall outside the energy analysis above: their canonical correlations are computed from the whitened cross-covariance, which normalizes away the cross-covariance energy of Proposition 4.1. They therefore do not follow the O(d/n) baseline. Their null instead reflects the rank deficiency of the whitening near and beyond d ≈ n: mean CCA peaks at d ≈ n and decays as O(n/d) for d > n, PWCCA saturates toward its ceiling once d ≳ n, and SVCCA is largely width-insensitive because it first projects onto a fixed low-dimensional subspace (Appendix E.7). Permutation calibration removes this inflation without closed-form solutions, unlike the energy-based metrics.

Why we use permutation rather than closed forms. Closed-form bulk edges are ensemble- and normalization-specific and are brittle to the preprocessing used in practice (e.g., centering, whitening, kernelization). Moreover, finite-n corrections can be non-negligible. We therefore estimate the relevant right-tail behavior nonparametrically via permutation. This yields a conservative, implementation-faithful estimate of chance fluctuations without relying on fragile analytical formulas.

#### C.5. The depth confounder

Here we formalize why selection-based summaries (e.g., maximum similarity over layer pairs) inflate with the size of the search space using Extreme Value Theory (EVT).

Let S = {Sℓ,ℓ′ : 1 ≤ ℓ ≤ LA, 1 ≤ ℓ′ ≤ LB} denote the collection of null similarity fluctuations under H0, and let M = LALB.

Assumption C.6 (Uniform sub-Gaussian right tails and integrability). There exist µ ∈ R and σ > 0 such that for all (ℓ,ℓ′) and all t ≥ 0,

t2 2σ2

. (49)

P(Sℓ,ℓ′ − µ ≥ t) ≤ exp −

Moreover, each Sℓ,ℓ′ is integrable: E|Sℓ,ℓ′| < ∞ for all (ℓ,ℓ′). Proposition C.7 (Maximal inequality, no independence required). Under Assumption C.6 and for M ≥ 2,

Sℓ,ℓ′ ≤ µ + C σ log M, (50)

E max

ℓ,ℓ′

where C > 0 is a constant (e.g., one can take C = 3).

Proof. Let Z := maxℓ,ℓ′ Sℓ,ℓ′ − µ. Since M < ∞ and E|Sℓ,ℓ′| < ∞ for all (ℓ,ℓ′), we have

E|Z| ≤ E max

|Sℓ,ℓ′| + |µ| ≤

ℓ,ℓ′

E|Sℓ,ℓ′| + |µ| < ∞, (51)

ℓ,ℓ′

so Z is integrable, and the tail-integration formula applies. By the union bound and Assumption C.6,

t2 2σ2

P(Z ≥ t) ≤ M exp −

for all t ≥ 0. (52)

Using the tail-integration formula for an integrable real-valued random variable Z,

E[Z] =

∞

P(Z ≥ t)dt −

0

∞

P(Z ≤ −t)dt ≤

0

∞

P(Z ≥ t)dt, (53)

0

and the bound P(Z ≥ t) ≤ 1, we obtain

∞

t2 2σ2

dt. (54)

E[Z] ≤

min 1, M exp −

0

Let t0 = σ√2log M. This value of t0 is the solution of M exp −t20/2σ2 = 1, i.e., the crossover where the bound min{1,·} switches. Splitting the integral at t0 yields

∞

t2 2σ2

dt. (55)

E[Z] ≤ t0 + M

exp −

t0

Applying the standard Gaussian tail bound t ∞

2/(2σ2)dt ≤ (σ2/t0)e−t

2

0/(2σ2) gives

e−t

0

σ √2log M

. (56)

E[Z] ≤ σ 2log M +

For M ≥ 2, the right-hand side is at most 3σ√log M, proving the claim with C = 3.

| |
|---|

Remark. When the Sℓ,ℓ′ are i.i.d. (or weakly dependent), classical Extreme Value Theory yields sharper asymptotics. For example, if Sℓ,ℓ′ ∼ N(µ0,σ02) i.i.d., the centered maximum converges to a Gumbel distribution and

√

lnlnM + ln4π 2

, (57)

E[Tmax] ≈ µ0 + σ0

√

2lnM −

2lnM

as stated in standard references (Cram´er, 1999; Embrechts et al., 2013). Real layer-wise similarities are dependent, so the approximation above should be treated as heuristic; Proposition C.7 provides a dependence-robust upper bound.

#### C.6. Null Baselines for Neighborhood Metrics

The preceding analysis focused on the cross-covariance-energy metrics (CKA and the RV coefficient), whose null baselines scale with d/n. Neighborhood-based metrics such as mutual k-NN follow a fundamentally different regime, which we now characterize.

Definition C.8 (Mutual k-NN overlap). For representations X ∈ Rn×d

y and neighborhood size k < n, let NX(i) ⊆ {1,...,n} \ {i} denote the indices of the k nearest neighbors of sample i in X (e.g., Euclidean or cosine), and similarly for NY(i). The mutual k-NN overlap is

,Y ∈ Rn×d

x

1 n

mKNN(X,Y) =

n

|NX(i) ∩ NY(i)| k

. (58)

i=1

Proposition C.9 (Uniformity of k-NN index sets under i.i.d. sampling). Fix an anchor index i ∈ {1,...,n}. Let x1,...,xn ∈ Rd be i.i.d. and define the k-NN set NX(i) ⊆ {1,...,n} \ {i} using a fixed distance dist(·,·). Assume either (i) {dist(xi,xj)}j̸=i are almost surely distinct, or (ii) ties are broken by selecting a uniformly random k-subset among the set of minimizers. Then NX(i) is uniformly distributed over the n−k1 k-subsets of {1,...,n} \ {i}.

Proof. Let I := {1,...,n} \ {i} be the candidate-neighbor index set. For any permutation π of I, i.i.d. sampling implies

(xj)j∈I =d (xπ(j))j∈I.

The k-NN selection rule depends on the candidate points only through their distances to xi, so permuting the candidate indices permutes the resulting neighbor set. Under either the no-ties assumption or the stated uniform tie-break rule, for any two k-subsets S,S′ ⊆ I there exists a permutation π with π(S) = S′ and hence

P NX(i) = S = P NX(i) = S′ .

Since the events {NX(i) = S} over all |S| = k partition the sample space, each has probability n−k1 −1.

| |
|---|

Theorem C.10 (Null baseline for mutual k-NN). Let X,Y ∈ Rn×d have i.i.d. rows, with X independent of Y. Define NX(i) and NY(i) as in Definition C.8, using either almost sure absence of distance ties or uniform random tie-breaking. Then

k n − 1

##### EH

.

mKNN(X,Y) =

0

Proof. Fix an anchor i. By Proposition C.9, NX(i) and NY(i) are each uniform random k-subsets of the (n − 1)-element set {1,...,n} \ {i}. Moreover, since X and Y are independent and NX(i) (resp. NY(i)) is a measurable function of X (resp. Y), the sets NX(i) and NY(i) are independent.

Therefore |NX(i) ∩ NY(i)| has a hypergeometric distribution with population size n − 1, number of “successes” k, and draws k, giving

k2 n − 1

EH

0 |NX(i) ∩ NY(i)| =

. Substituting into the definition of mKNN,

##### EH

0

mKNN(X,Y) =

n

1 n

##### EH

0

i=1

|NX(i) ∩ NY(i)| k

=

n

1 n

k n − 1

i=1

k n − 1

=

.

| |
|---|

Proposition C.11 (Per-anchor variance and generic bounds for mKNN under the null). Under the assumptions of Theorem C.10, for each anchor i the intersection size Hi := |NX(i) ∩ NY(i)| is hypergeometric with mean k2/(n − 1) and variance

k2(n − 1 − k)2 (n − 1)2(n − 2)

Var[Hi] =

.

Moreover, since mKNN(X,Y) ∈ [0,1] deterministically, we have the fully general bound

1 4

Var[mKNN(X,Y)] ≤

.

If one additionally assumes that the per-anchor terms {|NX(i) ∩ NY(i)|/k}ni=1 are independent (this is a modeling assumption, not a consequence of H0), then Var[mKNN(X,Y)] = O(1/n).

Proof. The hypergeometric variance formula gives

k2(n − 1 − k)2 (n − 1)2(n − 2)

(n − 1) − k (n − 1) − 1

k n − 1

k n − 1 ·

Var[Hi] = k ·

1 −

=

.

The bound Var[mKNN] ≤ 1/4 follows from mKNN ∈ [0,1]. Under the stated additional independence assumption across anchors,

1 n

H1 k

1 nk2

Var[H1], which is O(1/n) for fixed k.

Var mKNN(X,Y) =

Var

=

| |
|---|

### D. Implementation

A key advantage of null calibration is its simplicity: the framework can be applied to any similarity metric with minimal code changes. This section provides pseudocode for the two main calibration procedures described in the paper.

Scalar null calibration. Algorithm 1 shows the complete procedure for calibrating a single similarity comparison. The only requirement is a function similarity(X,Y) that computes the raw metric. The algorithm returns both a permutation p-value and a calibrated score with a principled zero point.

- Algorithm 1 Scalar Null Calibration

Require: Representations X ∈ Rn×d

x, Y ∈ Rn×d

y

Require: Similarity function sim(·,·), permutations K, significance level α Ensure: Calibrated score scal, p-value p

- 1: sobs ← sim(X,Y) {Observed similarity}
- 2: null scores ← []

- 3: for k = 1 to K do
- 4: π ← random permutation(n) {Permute sample indices}

- 5: Yπ ← Y[π,:] {Permute rows of Y}
- 6: null scores[k] ← sim(X,Yπ)

- 7: end for
- 8: combined ← [sobs] ∪ null scores {Combined set}

- 9: τα ← s(⌈(1−α)(K+1)⌉) {Ceiling order statistic of sorted combined (Equation (9)); not an interpolated quantile}
- 10: p ← 1+

K k=1 [null scores[k]≥sobs]

K+1 {Permutation p-value}

- 11: scal ← max s

obs−τα

smax−τα,0 {Calibrated score (use smax = 1 for bounded metrics)}

- 12: return scal,p

Aggregation-aware calibration for layer-wise comparisons. When comparing models with multiple layers and reporting a summary statistic (e.g., maximum similarity across layer pairs), the aggregation step must also be calibrated. Algorithm 2 shows how to extend scalar calibration to this setting. The key insight is that the same sample permutation must be applied consistently across all layers.

Computational cost. Let Csim denote the cost of a single similarity evaluation on n samples. Scalar calibration (Algorithm 1) performs K + 1 evaluations at cost O((K+1)Csim), and aggregation-aware calibration (Algorithm 2) over an LA ×LB layer grid costs O((K+1)LALB Csim). The K null evaluations are independent and run in parallel. The marginal

- Algorithm 2 Aggregation-Aware Null Calibration

′)}L

Require: Layer representations {X(ℓ)}L

ℓ′=1 (all n samples) Require: Similarity function sim(·,·), aggregator T (e.g., max), permutations K, level α Ensure: Calibrated aggregate Tcal, p-value pagg

ℓ=1, {Y(ℓ

A

B

- 1: {Compute observed similarity matrix}
- 2: for ℓ = 1 to LA do
- 3: for ℓ′ = 1 to LB do
- 4: S[ℓ,ℓ′] ← sim(X(ℓ),Y(ℓ

′))

- 5: end for
- 6: end for
- 7: Tobs ← T(S) {e.g., maxℓ,ℓ′ S[ℓ,ℓ′]}
- 8: null aggregates ← []

- 9: for k = 1 to K do
- 10: π ← random permutation(n) {Single permutation for all layers}

- 11: for ℓ = 1 to LA do
- 12: for ℓ′ = 1 to LB do
- 13: S(k)[ℓ,ℓ′] ← sim(X(ℓ),Y(ℓ

′)[π,:]) {Same π for all ℓ′}

- 14: end for
- 15: end for
- 16: null aggregates[k] ← T(S(k)) {Aggregate under null}

- 17: end for
- 18: combined ← [Tobs] ∪ null aggregates {Combined set}

- 19: ταagg ← T(⌈(1−α)(K+1)⌉) {Ceiling order statistic of sorted combined; not an interpolated quantile}
- 20: pagg ← 1+

K k=1 [null aggregates[k]≥Tobs]

K+1

- 21: Tcal ← max Tobs−τ

agg α

smax−ταagg ,0

- 22: return Tcal,pagg

cost per permutation is typically far below Csim, because permuting the rows of Y leaves each model’s own representations unchanged. Metrics that are functionals of the n × n Gram, dissimilarity, or neighbor structures (CKA, the RV coefficient, RSA, CKNNA) and the neighborhood metrics (mKNN, cycle-kNN) build these structures once in O(n2d) and reduce each null draw to a relabeling, at O(n2) or O(nk) and independent of the representation width d. The subspace and shape metrics (CCA, SVCCA, PWCCA, Procrustes) are the exception: each null draw recomputes an eigendecomposition or SVD of a d-dependent operator, at up to O(nd2 + d3) per permutation, which we batch across permutations on the GPU. On a single NVIDIA GTX 1080Ti with K = 200, this overhead is 20ms per layer pair for linear CKA and 52ms for mKNN, so a full layer-wise comparison between two models completes within a few seconds. We use K ∈ {200,...,500} permutations throughout, which we find sufficient for stable thresholds and p-values (Appendix E.6).

- E. Additional Experimental Results This appendix provides additional analyses that support the main text claims.

#### E.1. Phase diagrams across different noise distributions

The theoretical analysis in Section 4 assumes Gaussian entries for tractability, but real neural network activations rarely follow Gaussian distributions. Instead, they often exhibit heavy tails, sparsity, or multimodality. A critical question is whether our calibration, which makes no distributional assumptions, remains effective under such deviations.

- Figure 8 shows phase diagrams under different noise distributions: Gaussian, Student-t (ν = 3), Laplace, and Gaussian mixtures. Each panel shows raw scores (left) and calibrated scores (right) across the (d/n,σ) grid, where σ controls the noise level added to a fixed shared signal. At low σ, the signal dominates and both raw and calibrated scores correctly indicate high similarity. At high σ, noise overwhelms the signal, and similarity should approach zero. The key finding is that raw scores remain elevated (around 0.4–0.6) even at high noise levels where no detectable signal remains, while calibrated scores correctly collapse to near-zero. This pattern holds across all noise distributions tested, confirming that permutation-based calibration adapts to the data-generating process without requiring explicit distributional modeling.

Raw score

Calibrated score

3.0

1.0

1.0

|[Figure 41]|
|---|

| |[Figure 42]|
|---|---|
| | |
| | |
| | |
| | |
| | |

[Figure 43]

[Figure 44]

2.5

0.8

Noiselevelσ

Noiselevelσ

0.8

2.0

0.6

1.5

0.6

0.4

1.0

0.4

0.2

0.5

0.0

0.0

0.25 0.50 1.00

0.25 0.50 1.00

d/n

d/n

(a) Gaussian

Raw score

Calibrated score

3.0

1.0

1.0

| |[Figure 45]|
|---|---|
| | |
| | |
| | |
| | |
| | |

|[Figure 46]|
|---|

[Figure 47]

[Figure 48]

2.5

0.8

Noiselevelσ

d/n Noiselevelσ

0.8

2.0

0.6

0.6

1.5

0.4

1.0

0.4

0.2

0.5

0.2

0.0

0.25 0.50 1.00

0.25 0.50 1.00

d/n

(c) Laplace

Raw score

Calibrated score

3.0

1.0

1.0

| |[Figure 49]| |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| |[Figure 50]| |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

[Figure 51]

[Figure 52]

2.5

0.8

Noiselevelσ

Noiselevelσ

0.8

2.0

0.6

0.6

1.5

0.4

1.0

0.4

0.2

0.5

0.2

0.0

0.25 0.50 1.00

0.25 0.50 1.00

d/n

d/n

(b) Student-t (ν = 3)

Raw score

Calibrated score

3.0

1.0

1.0

| |[Figure 53]|
|---|---|
| | |
| | |
| | |
| | |
| | |

| |[Figure 54]|
|---|---|
| | |
| | |
| | |
| | |
| | |

[Figure 55]

[Figure 56]

2.5

0.8

Noiselevelσ

Noiselevelσ

0.8

2.0

0.6

0.6

1.5

0.4

1.0

0.4

0.2

0.5

0.2

0.0

0.25 0.50 1.00

0.25 0.50 1.00

d/n

d/n

(d) Gaussian mixture

- Figure 8. Phase diagrams under different noise types. Calibrated scores (right) collapse to near-zero at high noise levels across the (d/n, σ) grid, while raw scores (left) exhibit systematic positive bias. Calibration remains effective regardless of tail behavior.

#### E.2. SNR heatmaps

The experiments of the main paper (Figure 4) demonstrated that calibration eliminates false positives under H0 while preserving sensitivity to fixed signals. This section extends the analysis by characterizing how calibrated similarity varies jointly with signal strength, noise level, and dimensionality ratio, thereby delineating the regimes in which similarity

estimation remains reliable.

- Figure 9 presents heatmaps of raw scores (top row) and calibrated scores (bottom row) across the (Noise level, Signal strength) grid for three signal ranks (r ∈ {1,5,10}). The results reveal a clear phase transition structure. Raw scores (top) show uniformly high values across most of the grid, obscuring the true detection boundary. Calibrated scores (bottom) reveal the underlying signal: high scores concentrate in the low-noise, high-signal corner (bottom-left), while scores correctly collapse to zero as noise increases (moving right) or signal weakens (moving down). The detection boundary shifts rightward (tolerating higher noise) as signal rank increases. This phase structure is meaningful: it delineates when similarity measurements carry information about shared structure versus when they reflect only finite-sample artifacts.

Raw score

- 0.25

- 0.5
- 1

- 1.5
- 2
- 3
- 4

1.0

| |[Figure 57]|
|---|---|
| | |
| | |
| | |
| | |
| | |

[Figure 58]

0.8

Signalstrength

0.6

0.4

0.2

0.0

0 1 2 3

Noise level σ

(a) Rank r = 1

Calibrated score

- 0.25

- 0.5
- 1

- 1.5
- 2
- 3
- 4

1.0

| |[Figure 59]|
|---|---|
| | |
| | |
| | |
| | |
| | |

[Figure 60]

0.8

Signalstrength

0.6

0.4

0.2

0.0

0 1 2 3

Noise level σ

(d) Rank r = 1

Raw score

- 0.25

- 0.5
- 1

- 1.5
- 2
- 3
- 4

1.0

| |[Figure 61]|
|---|---|
| | |
| | |
| | |
| | |
| | |

[Figure 62]

0.8

Signalstrength

0.6

0.4

0.2

0.0

0 1 2 3

Noise level σ

(b) Rank r = 5

Calibrated score

- 0.25

- 0.5
- 1

- 1.5
- 2
- 3
- 4

1.0

| |[Figure 63]|
|---|---|
| | |
| | |
| | |
| | |
| | |

[Figure 64]

0.8

Signalstrength

0.6

0.4

0.2

0.0

0 1 2 3

Noise level σ

(e) Rank r = 5

Raw score

- 0.25

- 0.5
- 1

- 1.5
- 2
- 3
- 4

1.0

|[Figure 65]|
|---|

[Figure 66]

0.8

Signalstrength

0.6

0.4

0.2

0.0

0 1 2 3

Noise level σ

###### (c) Rank r = 10

Calibrated score

- 0.25

- 0.5
- 1

- 1.5
- 2
- 3
- 4

1.0

|[Figure 67]|
|---|

[Figure 68]

0.8

Signalstrength

0.6

0.4

0.2

0.0

0 1 2 3

Noise level σ

(f) Rank r = 10

- Figure 9. SNR heatmaps (calibrated scores). Higher-rank signals are detected at higher noise levels. The clear gradient confirms calibration preserves sensitivity to genuine structure.
- Figure 10 provides a complementary view by collapsing the 2D heatmaps into 1D curves, plotting calibrated score against noise level for different signal strengths s. As expected, calibrated scores decrease monotonically with noise level: at low noise, scores are high (reflecting the detectable shared signal), while at high noise, scores collapse to zero (reflecting that the signal is buried). Stronger signals (larger s) maintain elevated scores across a wider range of noise levels before eventually succumbing. Higher-rank signals (r = 5,10) show more gradual decay compared to r = 1, consistent with their greater statistical detectability. All curves converge to zero at high noise, confirming that the null floor is correctly calibrated regardless of signal strength or rank.

#### E.3. Comparing calibration approaches

A natural question is whether the choice of calibration summary affects the correction. We consider several approaches: (i) gated score, which thresholds at a significance level and rescales (α ∈ {0.05,0.1}); (ii) null-centered, subtracting the null mean; (iii) z-score, standardizing by null mean and standard deviation; and (iv) ARI-style, applying the chance-correction formula (s − E[s])/(smax − E[s]). Figure 11 evaluates these variants across metrics as d/n increases.

The results demonstrate that the gated score, null-centered, and ARI-style corrections all successfully collapse to appropriate null baselines across all metrics, regardless of whether the raw metric exhibits severe inflation (CKA, approaching 0.8) or mild inflation (RSA and mKNN, below 0.1). The z-score calibration, while correcting the mean, can exhibit artifacts when the null distribution is skewed, as occurs for bounded metrics like CKA at high d/n, making it less suitable as a universal correction.

r = 1

r = 5

r = 10

1.0

- s = 0.25

- s = 0.5

- s = 1

- s = 1.5

- s = 2

- s = 3

- s = 4

- s = 0.25

- s = 0.5

- s = 1

- s = 1.5

- s = 2

- s = 3

- s = 4

- s = 0.25

- s = 0.5

- s = 1

- s = 1.5

- s = 2

- s = 3

- s = 4

Calibratedscore

0.8

0.6

0.4

0.2

0.0

0 1 2 3

0 1 2 3

0 1 2 3

Noise level σ

Noise level σ

Noise level σ

- Figure 10. Calibrated scores decay with noise level. Each curve shows calibrated score versus noise level for a fixed signal strength s. Stronger signals maintain elevated scores across wider noise ranges; all curves converge to zero at high noise.

#### E.4. Comparison with analytical debiasing

We validate our empirical null calibration by comparing it to existing analytical bias corrections for CKA. Figure 12 shows the difference between our calibrated CKA and two existing estimators: the debiased CKA of Murphy et al. (2024) and the dep-cols CKA of Chun et al. (2025).

Our calibrated CKA closely matches the debiased CKA estimator, indicating that our calibration automatically corrects the dominant width-induced bias without requiring a metric-specific derivation. In contrast, dep-cols CKA is designed to correct column dependence, which is not present in our experimental setup (columns are independent by construction), so it does not address the dominant width-induced inflation. Under genuine signal (H1) it stays near its maximal value across all d/n, far above the closely agreeing debiased and calibrated estimates, while under the null (H0) it fluctuates around zero with high variance.

Our calibration targets the same source of finite-sample bias that the unbiased HSIC estimator of Song et al. (2012) subtracts analytically: the self-similarity (diagonal) terms. This is consistent with the close agreement between the two corrections in Figure 12. Unlike such analytical estimators, our permutation calibration requires no metric-specific derivation and applies to metrics for which no debiasing exists.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | |Cali Cali<br><br>|brated − brated −|Debiased Dep-cols|
| || |
|---|
| | | | |
| | || |
|---|
<br><br>|| |
|---|
|| |
|---|
<br><br>|| |
|---|
|

| | | | | | | |
|---|---|---|---|---|---|---|
| | | || |
|---|
| | | |
| | || |
|---|
| || |
|---|
<br><br>|| |
|---|
|| |
|---|
|
| | | | | | | |
| | | | |Cali Cali<br><br>|brated − brated −|Debiased Dep-cols|

0.0

Differenceundersignal

0.4

Differenceundernull

−0.2

0.2

−0.4

0.0

−0.6

−0.2

−0.8

−0.4

2−2 2−1 20 21 22 23

2−2 2−1 20 21 22 23

d/n

d/n

- Figure 12. Calibration recovers analytical debiasing. Difference between calibrated CKA and existing estimators (n = 1024, d/n swept). (Left) Under signal. (Right) Under null.

#### E.5. Width confounder under genuine signal on real networks

We now verify on real pretrained networks that calibration corrects the width confounder when a genuine signal is present, complementing the analytical result of Proposition C.5. Because the bias scales as O(d/n) (Proposition 4.1), width and sample size can be separated only by varying one while holding the other fixed, which this experiment does.

We extract last-layer features from the DINOv2 and AugReg ViT families. Within a family the models share training objective, data, and architecture and differ only in width d (e.g., ViT-S/B/L/g for DINOv2), so a within-family pair isolates the effect of width. All pairs use the same n = 1024 WIT images as the PRH setting, and for each pair we compute raw and

CKA (lin)

gated q90

gated q95

null-centered

z-score

ARI-adjusted

4096

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

0.25

0.25

0.25

2048

1024

0.5

0.5

0.5

0.00

0.00

0.00

512

n

256

−0.25

−0.25

−0.25

128

0.0

0.0

0.0

12825651210242048

12825651210242048

12825651210242048

12825651210242048

12825651210242048

12825651210242048

d

d

d

d

d

d

(a) CKA linear

CKA (rbf)

gated q90

gated q95

null-centered

z-score

ARI-adjusted

4096

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

0.25

0.25

0.25

2048

1024

0.5

0.5

0.5

0.00

0.00

0.00

512

n

256

−0.25

−0.25

−0.25

128

0.0

0.0

0.0

12825651210242048

12825651210242048

12825651210242048

12825651210242048

12825651210242048

12825651210242048

d

d

d

d

d

d

(b) CKA RBF

###### CCA

gated q90

gated q95

null-centered

z-score

ARI-adjusted

4096

25

25

25

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

2048

1024

0.5

0.5

0.5

0

0

0

512

n

256

−25

−25

−25

128

0.0

0.0

0.0

12825651210242048

12825651210242048

12825651210242048

12825651210242048

12825651210242048

12825651210242048

d

d

d

d

d

d

(c) CCA

###### RSA

gated q90

gated q95

null-centered

z-score

ARI-adjusted

0.01

0.01

0.01

4096

0.25

0.25

0.25

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

2048

1024

0.00

0.00

0.00

512

n

0.00

0.00

0.00

256

−0.25

−0.25

−0.25

128

12825651210242048

12825651210242048

12825651210242048

12825651210242048

12825651210242048

12825651210242048

d

d

d

d

d

d

(d) RSA (Spearman)

mKNN

gated q90

gated q95

null-centered

z-score

ARI-adjusted

4096

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

0.25

0.25

0.25

2048

0.05

0.05

0.05

1024

0.00

0.00

0.00

512

n

256

−0.25

−0.25

−0.25

128

0.00

0.00

0.00

12825651210242048

12825651210242048

12825651210242048

12825651210242048

12825651210242048

12825651210242048

d

d

d

d

d

d

(e) Mutual k-NN

- Figure 11. Comparing calibration approaches across metrics. Each panel shows raw scores alongside four calibration variants (gated score, null-centered, z-score, ARI-style) as d/n increases. Gated score, null-centered, and ARI-style corrections collapse to appropriate baselines; z-score exhibits artifacts for skewed null distributions.

calibrated linear CKA. At fixed n = 1024, the permutation threshold τ (the magnitude of the width correction) increases with the total representation width dX + dY , with Pearson correlation r = 0.88 (Figure 13). Since n is held constant, this dependence is attributable to width alone: calibration adapts its correction to representation width rather than applying a fixed offset. Together with Proposition C.5, this confirms that calibration corrects the width confounder on real networks with genuine similarity.

τ vs width at fixed n = 1024

r = 0.88

0.18

τPermutationthreshold

0.16

0.14

0.12

0.10

AugReg DINOv2

0.08

500 1000 1500 2000 2500 dX + dY

###### Figure 13. The width correction scales with representation width (real networks). For DINOv2 and AugReg ViT pairs on the same

n = 1024 WIT images, the permutation threshold τ (the width correction) grows with total width dX + dY (Pearson r = 0.88). With n fixed, the increasing correction is driven by width alone.

E.6. Permutation budget analysis

Permutation-based calibration introduces a computational-statistical tradeoff: more permutations yield more stable threshold estimates but increase runtime. Practitioners need guidance on the minimum budget required for reliable inference.

We analyze the stability of threshold estimates τα and calibrated scores as a function of the permutation budget K across 50 random seeds. Figure 14 shows two panels: the left panel displays threshold estimates, while the right panel shows calibrated scores under H0. Threshold estimates (left) stabilize rapidly, reaching stable values by approximately K = 50 for all metrics tested. Calibrated scores (right) exhibit more variability at very low budgets (K < 50), with occasional spikes due to unstable threshold estimation, but converge to near-zero by K ≈ 100–200.

Based on these results, we recommend K ≥ 200. The computational cost scales linearly with K, so this recommendation represents a favorable tradeoff between precision and efficiency.

Permutation budget

Permutation budget

0.0150

CKA (linear)

0.0125

0.4

Calibratedscore

CKA (RBF)

Thresholdτ

mKNN

0.0100

CKA (linear)

0.3

RSA

CKA (RBF)

0.0075

mKNN

0.2

0.0050

RSA

0.1

0.0025

0.0000

0.0

0 100 200 300 400 500

0 100 200 300 400 500

Number of Permutations K

Number of Permutations K

###### Figure 14. Permutation budget analysis. Left: threshold τα stabilizes by K ≈ 50. Right: calibrated scores under H0 converge to near-zero by K ≈ 100–200. Shaded regions show variability across random seeds.

E.7. Full null drift results

The main text presents null drift results for a representative subset of metrics under Gaussian noise. Here, we present additional results across all metrics evaluated in this work, including RSA, the RV coefficient, and Procrustes distance, as well as results under heavy-tailed noise distributions.

###### Figure 15 presents results under Gaussian noise for all metrics. The severity and shape of the null baseline vary substantially

across metric families: among the spectral metrics, CKA variants and the RV coefficient show the strongest monotonic inflation with width, whereas the CCA family follows a distinct rank-deficiency pattern (mean CCA peaks near d ≈ n then decays, PWCCA saturates, and SVCCA stays width-insensitive); the neighborhood metrics show the mildest drift. This reflects the structural sensitivity of the metrics to high-dimensional spurious correlations. RSA is the exception: as a self-normalized correlation of within-space dissimilarities, its null stays near zero and shows no width-driven drift. Critically, calibration eliminates drift across all metrics, collapsing scores to zero regardless of the raw bias magnitude.

- Figure 16 extends these results to heavy-tailed noise (Student-t, ν = 3). The qualitative pattern is preserved: the confounded metrics exhibit positive drift under the null, and calibration eliminates this drift. The magnitude of raw bias under heavy-tailed noise is comparable to the Gaussian case (marginally lower for the spectral metrics), and calibration adapts automatically without requiring distributional knowledge.

CKA (lin) CKA (rbf) mKNN RSA CCA SVCCA PWCCA RV Procrustes

4096

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

0.01

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

0.2

2048

0.5

0.05

1024

0.5

0.5

0.5

0.5

0.5

n

0.1

512

0.00

0.0

256

128

0.0

0.0

0.00

0.0

0.0

0.0

0.0

4096

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

0.01

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

0.2

2048

0.5

0.05

1024

0.5

0.5

0.5

0.5

0.5

n

0.1

512

0.00

0.0

256

128

0.0

0.0

0.00

0.0

0.0

0.0

0.0

12825651210242048

12825651210242048

12825651210242048

12825651210242048

12825651210242048

12825651210242048

12825651210242048

12825651210242048

12825651210242048

d

d

d

d

d

d

d

d

d

- Figure 15. Full null drift results (Gaussian). Raw scores (top) exhibit systematic positive bias; calibrated scores (bottom) collapse to zero.

[Figure 165]

128

256

512

1024

2048

4096

n

CKA (lin) CKA (rbf) mKNN RSA CCA SVCCA PWCCA RV Procrustes

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

12825651210242048

d

128

256

512

1024

2048

4096

n

[Figure 175]

12825651210242048

d

[Figure 176]

12825651210242048

d

[Figure 177]

12825651210242048

d

[Figure 178]

12825651210242048

d

[Figure 179]

12825651210242048

d

[Figure 180]

12825651210242048

d

[Figure 181]

12825651210242048

d

[Figure 182]

12825651210242048

d

[Figure 183]

0.25

0.50

0.75

[Figure 184]

0.25

0.50

0.75

[Figure 185]

0.0

0.5

[Figure 186]

0.0

0.5

[Figure 187]

0.00

0.05

[Figure 188]

0.00

0.05

[Figure 189]

0.00

0.02

[Figure 190]

0.00

0.02

[Figure 191]

0.0

0.5

[Figure 192]

0.0

0.5

[Figure 193]

0.0

0.1

0.2

[Figure 194]

0.0

0.1

0.2

[Figure 195]

0.0

0.5

1.0

[Figure 196]

0.0

0.5

1.0

[Figure 197]

0.0

0.5

[Figure 198]

0.0

0.5

[Figure 199]

0.0

0.5

[Figure 200]

0.0

0.5

- Figure 16. Full null drift results (heavy-tailed). Student-t (ν = 3) noise. The pattern is consistent across all metrics: calibration eliminates spurious similarity regardless of noise distribution.

Robustness to the generative process. The experiments above use i.i.d. Gaussian and heavy-tailed noise. To confirm that the width confounder is not tied to these specific generators, we also produce representations as xi = fx(ai) and yi = fy(ai), where ai is a shared Gaussian input and fx,fy are independent random linear maps or MLPs, so the two representations share inputs but no systematic structure. Figure 17 shows that raw CKA still inflates with dimensionality d in every regime, while calibrated CKA stays at zero. Sharing inputs through unrelated networks thus produces width-driven inflation rather than genuine convergence, and calibration removes it regardless of how the representations are generated.

#### E.8. Extended PRH alignment results (image–text)

The main text establishes a divergence between local and global similarity metrics when applied to the Platonic Representation Hypothesis (PRH): neighborhood-based metrics retain significant cross-modal alignment after calibration, while spectral metrics lose their apparent convergence trend. A natural question is whether this finding is robust across model families and metric variants.

Here we present comprehensive results across all five vision model families in the PRH setting (DINOv2, CLIP, ImageNet21K, MAE, and CLIP-finetuned) and a broad range of metrics spanning the local-to-global spectrum (Figures 18 to 20).

The results reinforce and extend the main text findings. Neighborhood metrics (mKNN, cycle-kNN, CKNNA) show a consistent alignment trend across all vision families with a neighborhood size of 10. This pattern holds for both self-

Raw CKA

Calibrated CKA

1.0

|Indepe Shared|ndent a , linear| | || |
|---|
|
|---|---|---|---|---|
|Shared|i<br><br>ai, MLP| || |
|---|
| |
| | | || |
|---|
| |
| | | | | |
|| |
|---|
<br><br>|| |
|---|
<br><br>| |
|---|
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

0.8

0.6

CKA

0.4

0.2

0.0

101 102 103

101 102 103

Dimensionality d

Dimensionality d

- Figure 17. The width confounder is robust to the generative process. Even when X and Y are computed from the same inputs by unrelated random networks (linear or MLP), raw CKA (left) inflates with dimensionality d although no systematic alignment is present; calibrated CKA (right) stays at zero. The inflation is a finite-width artifact, not evidence of convergence.

supervised (DINOv2, MAE) and supervised (ImageNet-21K) pretraining objectives, as well as for both CLIP-aligned and CLIP-finetuned variants. Spectral metrics (CKA linear, CKA RBF, unbiased CKA, RV coefficient, SVCCA) show a different pattern: raw scores suggest increasing alignment with model scale, but calibrated scores show no such scaling trend.

Trend with model scale, before versus after calibration. For each metric separately, we measure the Pearson correlation between language-model capability (the model ranking of Huh et al. (2024)) and the similarity score across all model pairs, before and after calibration (Table 2). Global metrics lose most of their trend with scale after calibration, while local (neighborhood) metrics retain it across neighborhood sizes.

Table 2. Trend of each similarity metric with model scale, before versus after calibration. Pearson correlation between language-model capability and the similarity score in the PRH setting. The costly spectral and geometric metrics (SVCCA, PWCCA, Procrustes, RV) are evaluated on the reduced DINOv2 subset (12 language models × 4 vision models); the remaining metrics use the full grid (204 pairs). RV coincides with linear CKA under centering and is listed only for completeness. Global metrics lose their trend with scale after calibration; local (neighborhood) metrics retain it.

Metric Before After Global metrics (spectral and geometric) CKA (linear) 0.86 0.45 CKA (RBF) 0.83 0.35 Unbiased CKA 0.65 −0.01 Procrustes 0.89 0.39 RV coefficient 0.92 0.31 Local metrics (neighborhood)

mKNN (k=10) 0.85 0.84 mKNN (k=20) 0.84 0.84 mKNN (k=50) 0.83 0.82 mKNN (k=100) 0.79 0.79 CKNNA (k=10) 0.87 0.87 CKNNA (k=20) 0.86 0.86 CKNNA (k=50) 0.86 0.86 CKNNA (k=100) 0.84 0.84 cycle-kNN (k=10) 0.85 0.85 cycle-kNN (k=20) 0.85 0.85 cycle-kNN (k=50) 0.84 0.85 cycle-kNN (k=100) 0.77 0.74

Continuous language-performance axis. The preceding figures place each language model at its performance rank. Because Huh et al. (2024) instead plot cross-modal alignment against a continuous measure of language performance, we reproduce all of the above results on that axis for direct comparison. We quantify language performance by bits-per-byte (BPB) over OpenWebText, as in Huh et al. (2024), oriented so that larger values, max(BPB) − BPB, indicate stronger

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| |bas larg|e e| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |gia|nt| | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

0.175

0.14

tiny small base large

base large huge

small

base large huge

base large huge

0.10

Alignmentscore

0.200

0.150

0.150

0.12

0.175

0.08

0.125

0.125

0.150

0.10

0.100

0.06

0.100

0.125

0.08

0.075

0.04

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(a) mKNN: Neighborhood overlap.

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

0.6

tiny small base large

base large huge

small base large giant

base large huge

base large huge

Alignmentscore

0.35

0.6

0.6

0.6

0.30

0.5

0.4

0.4

0.4

0.25

0.4

0.2

0.2

0.20

0.2

0.3

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(b) CKA RBF: Spectral alignment.

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

0.7

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

0.5

tiny small base large

base large huge

base large huge

base large huge

0.7

0.6

Alignmentscore

0.6

0.6

0.4

0.5

0.6

0.5

small base large giant

0.5

0.3

0.4

0.5

0.4

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(c) cycle-kNN: Bidirectional consistency.

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

0.40

0.35

| | | | | | | | | | | | |ba lar|se ge|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

0.30

0.8

tiny small

base large huge

base large huge

Alignmentscore

0.35

0.6

0.30

0.40

0.25

0.30

0.4

0.25

0.25

0.35

0.20

0.2

small base large giant

base large huge

0.20

0.20

0.0

0.15

0.30

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(d) Unbiased CKA.

- Figure 18. PRH alignment results (all vision families). All five vision model families are shown (DINOv2, CLIP, ImageNet-21K, MAE, CLIP-finetuned). The divergence between local and global metrics is consistent across all families.

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

0.5

0.35

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

0.5

tiny small base large

base large huge

small base large giant

base large huge

base large huge

Alignmentscore

0.5

0.5

0.4

0.4

0.30

0.4

0.3

0.4

0.3

0.25

0.3

0.2

0.2

0.2

0.20

0.3

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(a) CKA linear.

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| |lar gia|ge nt| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |

0.200

tiny small base large

base large huge

small base

base large huge

base large huge

0.175

0.25

Alignmentscore

0.10

0.20

0.175

0.150

0.20

0.150

0.08

0.125

0.15

0.125

0.06

0.15

0.100

0.100

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(b) CKNNA.

- Figure 19. Additional PRH metrics (all vision families). CKA linear (a) shows the same loss of convergence trend as CKA RBF. CKNNA (b) shows consistent local alignment across all vision families.

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

560m1b11b73b7b1 3b7b13b 7b13b30b65b

0.2

0.3

0.4

0.5

Alignmentscore

BLOOM OpenLLaMA LLaMA

DINOv2

small base large giant

calibrated uncalibrated

(a) RV coefficient.

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

560m1b11b73b7b1 3b7b13b 7b13b30b65b

0.35

0.40

0.45

Alignmentscore

BLOOM OpenLLaMA LLaMA

DINOv2

small base large giant

calibrated uncalibrated

(b) SVCCA.

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

560m1b11b73b7b1 3b7b13b 7b13b30b65b

0.0

0.1

0.2

0.3

Alignmentscore

BLOOM OpenLLaMA LLaMA

DINOv2

small base large giant

calibrated uncalibrated

(c) Procrustes.

- Figure 20. Additional geometric and spectral metrics (DINOv2). The RV coefficient (a), SVCCA (b), and Procrustes distance (c) confirm the same pattern as CKA: calibrated scores show no convergence trend with model scale, so the disappearance of global convergence is not specific to CKA but extends to shape- and geometry-based metrics. Due to the computational cost of permutation-based calibration for these metrics, we report results for DINOv2 only.

next-token prediction, and place each model at its measured value with a per-encoder least-squares fit (Figures 21 to 23). The local–global divergence is unchanged on this axis: after calibration, neighborhood metrics retain their alignment trend with language performance while spectral metrics do not, so the finding does not depend on whether capability is expressed as a rank or as a continuous performance value.

Statistical significance. Beyond calibrated scores, we report permutation p-values to quantify statistical evidence against the null hypothesis of no cross-modal alignment (Figure 24). All 204 vision–language model pairs are significant at p < 0.05, with most achieving p ≈ 0.002 (the minimum attainable with K = 500 permutations) for both local and global metrics. This confirms that cross-modal similarity is statistically significant (i.e., has some alignment) across all model pairs. The critical distinction between local and global metrics lies not in statistical significance but in the magnitude and trends of calibrated scores. Local metrics show substantial alignment above the null threshold that persists across scales, whereas global metrics, although significant, show no convergence in calibrated effect sizes.

#### E.9. Extended video–language alignment results

The main text extends the PRH analysis to video–language alignment following Zhu et al. (2026). Here, we provide additional results to verify that the local-vs-global pattern observed for image–language alignment extends to the video modality.

We use 1024 samples from the PVD (Bolya et al., 2025; Cho et al., 2025) test set. We evaluate video-native models (VideoMAE (Tong et al., 2022)) and, as a frame-level baseline, image models (DINOv2 and CLIP) applied to the middle frame of each video, comparing all against the same three language model families used in the image–language experiments (BLOOM, OpenLLaMA, LLaMA) at multiple scales. For VideoMAE we include both the fine-tuned scale series (small/base/large/huge, fine-tuned on Kinetics) used in the main text and the corresponding non-fine-tuned checkpoints. Figure 25 shows results for spectral (CKA RBF) and neighborhood (mKNN, CKNNA) metrics.

The pattern mirrors the image–language findings. For spectral metrics, raw scores suggest alignment, whereas calibrated scores drop significantly, indicating that much of the apparent alignment is attributable to width and depth confounders. In contrast, neighborhood metrics retain significant alignment after calibration, confirming that video and language representations share local topological structure. This local alignment strengthens with the capability of the video encoder: both fine-tuning on Kinetics and increasing scale raise the calibrated neighborhood alignment, in line with the Aristotelian hypothesis that more capable representations converge more in local structure. Calibration removes the spectral inflation throughout.

#### E.10. Characterizing the locality of cross-modal alignment

The main text establishes that local neighborhood metrics retain significant alignment after calibration, while global spectral metrics do not. We next characterize how local this alignment is. Both mKNN and CKA-RBF have hyperparameters that control their sensitivity to local versus global structure. By varying these parameters, we can characterize the scale at which cross-modal alignment emerges.

Experimental setup. We vary two locality parameters: the neighborhood size k in mKNN, testing k ∈ {10,20,50,100} where smaller values focus on immediate neighbors and larger values consider broader local structure, and the RBF kernel bandwidth σ in CKA-RBF, testing σ ∈ {0.1,0.5,2.0,5.0}, which controls the length scale over which the kernel assigns significant weight.

RBF bandwidth. The RBF (radial basis function) kernel is defined as k(x,y) = exp −∥x − y∥2/2σ2 . The bandwidth σ determines the length scale of similarity. When σ is small (e.g., 0.1), the kernel is sharply peaked: only very close points contribute significantly to the Gram matrix, making the similarity measure sensitive to exact pairwise distances in the immediate neighborhood. When σ is large (e.g., 5.0), the kernel is broad: even moderately distant points contribute, and the similarity measure aggregates information over larger neighborhoods, becoming sensitive to coarser geometric structure.

Neighborhood size. For mKNN, the parameter k controls how many nearest neighbors are considered when measuring overlap. Small k (e.g., 10) measures agreement on immediate neighbors, i.e., the closest points to each sample, capturing fine-grained local topology. Large k (e.g., 100) measures agreement on a broader neighborhood. With n = 1000 samples

INet21K

###### MAE

###### DINOv2

###### CLIP

###### CLIP (INet ft)

| | | |
|---|---|---|
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

|base large| | | |
|---|---|---|---|
|huge| | | |
| | | | |
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |
| | | |

0.175

0.14

tiny small base large

base large huge

small base large giant

base large huge

0.10

0.200

Alignmentscore

0.150

0.150

0.12

0.175

0.08

0.125

0.125

0.150

0.10

0.100

0.06

0.100

0.125

0.08

0.075

0.04

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

calibrated uncalibrated

(a) mKNN: Neighborhood overlap.

INet21K

###### MAE

###### DINOv2

###### CLIP

###### CLIP (INet ft)

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| |base large| | | |
|---|---|---|---|---|
| |huge| | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

0.6

tiny small base large

small base large giant

base large huge

base large huge

0.35

0.6

0.6

0.6

Alignmentscore

0.30

0.5

0.4

0.4

0.4

0.25

0.4

0.2

0.2

0.20

0.2

0.3

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

calibrated uncalibrated

(b) CKA RBF: Spectral alignment.

INet21K

###### MAE

###### DINOv2

###### CLIP

CLIP (INet ft)

0.7

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

0.5

tiny small base large

base large huge

small base large giant

base large huge

base large huge

0.7

0.6

Alignmentscore

0.6

0.6

0.4

0.5

0.6

0.5

0.5

0.3

0.4

0.5

0.4

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

calibrated uncalibrated

(c) cycle-kNN: Bidirectional consistency.

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

0.40

0.35

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| |base large<br><br>| | |
| |huge| | |

| | | |
|---|---|---|
| | | |
| | | |
|small base| | |
|large giant| | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

0.30

0.8

tiny small base large

base large huge

base large huge

Alignmentscore

0.35

0.6

0.30

0.40

0.25

0.30

0.4

0.25

0.25

0.35

0.20

0.2

0.20

0.20

0.0

0.15

0.30

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

calibrated uncalibrated

(d) Unbiased CKA.

- Figure 21. PRH alignment versus language performance (all vision families). As Figure 18, but with the horizontal axis given by continuous language performance (max(BPB) − BPB over OpenWebText) and a per-encoder least-squares fit. The local–global divergence is unchanged from the rank-based axis.

INet21K

###### MAE

DINOv2

###### CLIP

###### CLIP (INet ft)

0.5

0.35

| | | |
|---|---|---|
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

0.5

tiny small base large

base large huge

small base large giant

base large huge

base large huge

0.5

Alignmentscore

0.5

0.4

0.4

0.30

0.4

0.3

0.4

0.3

0.25

0.3

0.2

0.2

0.2

0.20

0.3

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

calibrated uncalibrated

(a) CKA linear.

INet21K

###### MAE

###### DINOv2

###### CLIP

###### CLIP (INet ft)

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |

0.200

tiny small base large

base large huge

small base large giant

base large huge

base large huge

0.175

0.25

Alignmentscore

0.10

0.20

0.175

0.150

0.20

0.150

0.08

0.125

0.15

0.125

0.06

0.15

0.100

0.100

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

0.0 0.2 0.4 LANGUAGE performance

calibrated uncalibrated

(b) CKNNA.

- Figure 22. Additional PRH metrics versus language performance (all vision families). As Figure 19, with the continuous languageperformance axis. CKA linear (a) loses its trend after calibration; CKNNA (b) retains local alignment across all vision families.

| | | |
|---|---|---|
| | | |
| | | |

0.0 0.2 0.4 LANGUAGE performance

0.2

0.3

0.4

0.5

Alignmentscore

DINOv2

small base large giant

calibrated uncalibrated

(a) RV coefficient.

| | | | |
|---|---|---|---|
| | | | |
| | | | |

0.0 0.2 0.4 LANGUAGE performance

0.35

0.40

0.45

Alignmentscore

DINOv2

small base large giant

calibrated uncalibrated

(b) SVCCA.

| | | | |
|---|---|---|---|
| | | | |
| | | | |

0.0 0.2 0.4 LANGUAGE performance

0.0

0.1

0.2

0.3

Alignmentscore

DINOv2

small base large giant

calibrated uncalibrated

(c) Procrustes.

- Figure 23. Additional geometric and spectral metrics versus language performance (DINOv2). As Figure 20, with the continuous language-performance axis. The RV coefficient (a), SVCCA (b), and Procrustes (c) show no calibrated convergence trend with language performance, matching the rank-based axis.

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |as arg|e e| | | | | | | | | | | |
|α|ug =|e 0.0|5| | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |as arg|e e| | | | | | | | | | | |
|α|ug =|e 0.0|5| | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |as arg|e e| | | | | | | | | | | |
|α|ug =|e 0.0|5| | | | | | | | | | |
| | | | | | | | | | | | | | |

p-value(logscale)

tiny small base large α = 0.05

small base large giant α = 0.05

10−2

###### 10−2

###### 10−2

###### 10−2

###### 10−2

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

(a) mKNN (k = 10).

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |as arg|e e| | | | | | | | | | | |
|α|ug =|e 0.0|5| | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |as arg|e e| | | | | | | | | | | |
|α|ug =|e 0.0|5| | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |as arg|e e| | | | | | | | | | | |
|α|ug =|e 0.0|5| | | | | | | | | | |
| | | | | | | | | | | | | | |

p-value(logscale)

tiny small base large α = 0.05

small base large giant α = 0.05

10−2

###### 10−2

###### 10−2

###### 10−2

###### 10−2

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

(b) CKA linear.

- Figure 24. Permutation p-values for PRH alignment. All model pairs are significant at p < 0.05, with most achieving p < 0.005 for both local (a) and global (b) metrics. The difference between metric families lies in calibrated effect sizes, not significance.

and k = 100, we ask whether the 10% closest points agree across representations. Crucially, mKNN is a rank-based metric: it asks which points are neighbors (ordinal information), not how close they are (cardinal information).

mKNN across k values. Figure 26 shows the PRH alignment results for mKNN with varying k. A consistent pattern emerges: all k values show significant alignment after calibration, with calibrated scores remaining well above zero even at k = 100. However, the scaling trend is most pronounced at small k. For k = 10, raw scores show a clear upward trend with model capacity that persists after calibration. At large k, this trend flattens even in raw scores. For k = 100, raw scores plateau for larger models, suggesting that broader neighborhood agreement is already saturated across model scales. This pattern indicates that scaling-driven improvement in alignment is concentrated at the finest topological level.

CKA-RBF across bandwidth values. Figure 27, and the accompanying p-values in Figure 28, show results for CKA-RBF with varying bandwidth σ, revealing a different pattern from mKNN. At σ = 0.1 (very local), there is no significant alignment after calibration: raw scores are near 1.0, reflecting the high similarity of any high-dimensional representations under a sharply peaked kernel. However, calibrated scores collapse to approximately zero with p-values exceeding 0.05 for most model pairs, indicating that the observed similarity is indistinguishable from chance. At σ = 0.5, alignment emerges, but with a flattening trend after calibration. Calibrated scores initially rise with model scale, then plateau and slightly decline for the largest models. At σ = 2.0 and σ = 5.0, significant alignment persists, but the calibrated trend also flattens, resembling the pattern observed for large-k mKNN: alignment exists, but scaling-driven improvement disappears after calibration.

Topological versus metric alignment. The contrasting behavior of mKNN and small-σ CKA-RBF reveals a fundamental distinction in what “local alignment” means. On one hand, mKNN measures topological alignment: do the representations agree on which points are neighbors? This captures ordinal information where the ranking of distances matters but not their absolute values. On the other hand, small-σ CKA-RBF measures metric alignment: do the representations agree on how close neighbors are? This captures cardinal information where exact distance values matter.

The fact that mKNN shows alignment at all k values while small-σ CKA-RBF shows no alignment reveals that cross-modal representations agree on neighborhood identity (which points are close) but not on exact local distances (how close they are). This finding is consistent with the observation that different training objectives and architectures induce different distance scales in representation space while preserving the relative ordering of neighbors. The Aristotelian Representation Hypothesis should therefore be understood as convergence to shared topological structure rather than shared metric structure.

VideoMAE

DINOv2

###### CLIP

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |sm bas bas|all e e (F|(FT)<br><br>T)|
| | | | | | | | | | |larg larg hug<br><br>|e e ( e (|FT) FT)|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | |sm<br><br>ba|all se|
| | | | | | | | | | | | |la gi<br><br>|rge ant|
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | |b la|ase rge|
| | | | | | | | | | | | | |h g<br><br>|uge iant|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

0.7

0.6

Alignmentscore

0.6

0.6

0.5

0.4

0.4

0.4

0.2

0.3

0.2

0.2

560M1B11B73B7B1 3B7B13B 7B13B30B65B

560M1B11B73B7B1 3B7B13B 7B13B30B65B

560M1B11B73B7B1 3B7B13B 7B13B30B65B

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(a) CKA RBF (spectral).

VideoMAE

DINOv2

###### CLIP

0.26

0.20

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| |ug ian|e t| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

small base large giant

base large

small (FT) base base (FT) large large (FT) huge (FT)

0.24

0.24

Alignmentscore

0.15

0.22

0.22

0.10

0.20

0.18

0.20

0.05

0.16

560M1B11B73B7B1 3B7B13B 7B13B30B65B

560M1B11B73B7B1 3B7B13B 7B13B30B65B

560M1B11B73B7B1 3B7B13B 7B13B30B65B

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(b) mKNN (k = 10, neighborhood).

VideoMAE

DINOv2

###### CLIP

0.28

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | |sma bas bas|ll<br><br>e e (|(FT)<br><br>FT)| | | | | | | | | |
| | | | | | | | | | | | | | |
| | |larg larg hug|e e ( e (|FT) FT)| | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

small base large giant

base large huge giant

0.20

0.275

0.26

Alignmentscore

0.15

0.250

0.24

0.225

0.10

0.22

0.200

0.05

0.20

0.175

560M1B11B73B7B1 3B7B13B 7B13B30B65B

560M1B11B73B7B1 3B7B13B 7B13B30B65B

560M1B11B73B7B1 3B7B13B 7B13B30B65B

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(c) CKNNA (k = 10, neighborhood).

- Figure 25. Extended video–language alignment. Each panel shows VideoMAE (fine-tuned and non-fine-tuned), DINOv2, and CLIP (frame-level) against the language model families; raw scores are dotted, calibrated solid. (a) Spectral alignment (CKA RBF) drops after calibration. (b,c) Neighborhood alignment is retained and strengthens with video-encoder capability.

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| |bas larg|e e| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |gia|nt| | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

0.175

0.14

tiny small base large

base large huge

small

base large huge

base large huge

0.10

Alignmentscore

0.200

0.150

0.150

0.12

0.175

0.08

0.125

0.125

0.150

0.10

0.100

0.06

0.100

0.125

0.08

0.075

0.04

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(a) mKNN (k = 10)

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

0.200

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

0.200

tiny small base large

base large huge

small base large giant

base large huge

base large huge

0.16

Alignmentscore

0.12

0.175

0.175

0.20

0.14

0.10

0.150

0.150

0.12

0.08

0.125

0.125

0.15

0.10

0.100

0.06

0.100

0.08

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(b) mKNN (k = 20)

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

0.25

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

tiny small base large

base large huge

small base large giant

base large huge

base large huge

Alignmentscore

0.175

0.200

0.25

0.150

0.175

0.20

0.20

0.125

0.150

0.20

0.15

0.100

0.15

0.125

0.15

0.075

0.100

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(c) mKNN (k = 50)

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

0.30

0.25

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | |iny m|all| | | | | | | | | | | |
| | |as arg|e e| | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |as arg|e e| | | | | | | | | | | |
| |ug|e| | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | |sm<br><br>ba<br><br>|all se|
| | | | | | | | | | | |lar gia|ge nt|

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

base large huge

base large huge

Alignmentscore

0.25

0.30

0.25

0.25

0.20

0.20

0.25

0.20

0.20

0.15

0.15

0.20

0.15

0.15

0.10

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(d) mKNN (k = 100)

- Figure 26. PRH alignment with varying neighborhood size k for mKNN. All k values show significant alignment after calibration. The scaling trend is clearest at small k and flattens at large k, suggesting scaling improvements are concentrated at the finest local scale.

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | |tin sm<br><br>|y all|
| | | | | | | | | | | |ba lar<br><br>|se ge|
| | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | |ba lar<br><br>|se ge|
| | | | | | | | | | | |hu|ge|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | |sm<br><br>ba|all se|
| | | | | | | | | | | |lar gia|ge nt|
| | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | |ba lar|se ge|
| | | | | | | | | | | |hu|ge|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | |ba lar|se ge|
| | | | | | | | | | | |hu|ge|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

1.00

1.00

1.00

1.00

1.00

Alignmentscore

0.75

0.75

0.75

0.75

0.75

0.50

0.50

0.50

0.50

0.50

0.25

0.25

0.25

0.25

0.25

0.00

0.00

0.00

0.00

0.00

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(a) CKA-RBF (σ = 0.1)

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

1.0

1.0

1.0

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | |tin sm ba<br><br>|y all<br><br>se|
| | | | | | | | | | | |lar|ge|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | |sm<br><br>ba|all se|
| | | | | | | | | | | |lar gia|ge nt|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | |ba la|se rge|
| | | | | | | | | | | |hu|ge|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

Alignmentscore

0.4

0.8

0.8

0.8

0.8

0.6

0.3

0.6

0.6

base large huge

base large huge

0.6

0.4

0.4

0.4

0.2

0.4

0.2

0.2

0.2

0.1

0.2

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(b) CKA-RBF (σ = 0.5)

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

| | |iny ma as|ll e| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | |arg|e| | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

0.35

0.5

base large huge

small base large giant

base large huge

base large huge

Alignmentscore

0.5

0.5

0.5

0.4

0.30

0.4

0.4

0.3

0.3

0.4

0.25

0.3

0.2

0.2

0.2

0.20

0.3

0.1

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(c) CKA-RBF (σ = 2.0)

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

0.5

0.35

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

0.5

tiny small base large

base large huge

small base large giant

base large huge

base large huge

Alignmentscore

0.5

0.5

0.4

0.4

0.30

0.4

0.3

0.4

0.3

0.25

0.3

0.2

0.2

0.2

0.3

0.20

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(d) CKA-RBF (σ = 5.0)

- Figure 27. PRH alignment with varying bandwidth σ for CKA-RBF. At very small σ (a), no significant alignment remains after calibration. Larger σ values (b–d) show significant alignment, but the scaling trend flattens after calibration.

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

100

100

100

| | | | | |ti<br><br>sm|ny a|ll| | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | |b la α|ase rge<br><br>=|.0|5| | | | | |
| | | | | | | | | | | | | | |

| |ug α =|e 0.0|5| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |sm<br><br>ba lar<br><br>|al se ge|l|
| | | | | | | | | | |gia<br><br>α|nt = 0|.05|

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |as arg|e e| | | | | | | | | | | |
|α|ug =|e 0.0|5| | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| |as|e| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|α|arg<br><br>ug =|e e 0.0|5| | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

p-value(logscale)

base large

10−1

10−1

10−1

###### 10−2

10−2

10−2

10−2

10−1

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

(a) CKA-RBF (σ = 0.1)

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |as arg|e e| | | | | | | | | | | |
|α|ug =|e 0.0|5| | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |as arg|e e| | | | | | | | | | | |
|α|ug =|e 0.0|5| | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |as arg|e e| | | | | | | | | | | |
|α|ug =|e 0.0|5| | | | | | | | | | |
| | | | | | | | | | | | | | |

p-value(logscale)

tiny small base large α = 0.05

small base large giant α = 0.05

10−2

###### 10−2

###### 10−2

###### 10−2

###### 10−2

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

(b) CKA-RBF (σ = 0.5)

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |as arg|e e| | | | | | | | | | | |
|α|ug =|e 0.0|5| | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |as arg|e e| | | | | | | | | | | |
|α|ug =|e 0.0|5| | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |as arg|e e| | | | | | | | | | | |
|α|ug =|e 0.0|5| | | | | | | | | | |
| | | | | | | | | | | | | | |

p-value(logscale)

tiny small base large α = 0.05

small base large giant α = 0.05

10−2

###### 10−2

###### 10−2

###### 10−2

###### 10−2

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

(c) CKA-RBF (σ = 2.0)

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |as arg|e e| | | | | | | | | | | |
|α|ug =|e 0.0|5| | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |as arg|e e| | | | | | | | | | | |
|α|ug =|e 0.0|5| | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |as arg|e e| | | | | | | | | | | |
|α|ug =|e 0.0|5| | | | | | | | | | |
| | | | | | | | | | | | | | |

p-value(logscale)

tiny small base large α = 0.05

small base large giant α = 0.05

10−2

###### 10−2

###### 10−2

###### 10−2

###### 10−2

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

(d) CKA-RBF (σ = 5.0)

- Figure 28. Significance of PRH alignment with varying bandwidth σ for CKA-RBF. Alignment with σ = 0.1 (a) is not significant for multiple models where larger bandwidths have significance (b–d).

#### E.11. Sensitivity to significance level α

The main text uses a significance level of α = 0.05 throughout. To confirm that the PRH conclusions are not sensitive to this choice, we repeat the PRH evaluation from Section 6.3 with α ∈ {0.01,0.05,0.10} for representative global (CKA linear, CKA RBF) and local (mKNN with k = 10) metrics.

Figures 29 to 31 show that the conclusions are entirely invariant to the choice of α. For global metrics, calibrated scores show no convergence trend at any significance level. For local metrics, calibrated scores retain their alignment trend across all three α values. Stricter thresholds (α = 0.01) produce slightly lower calibrated scores, while more permissive thresholds (α = 0.10) produce slightly higher ones, but the qualitative pattern is unchanged. This confirms that our findings are not an artifact of a particular significance level.

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

0.5

0.35

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

0.5

tiny small base large

base large huge

small base large giant

base large huge

base large huge

Alignmentscore

0.5

0.5

0.4

0.4

0.30

0.4

0.3

0.4

0.3

0.25

0.3

0.2

0.2

0.2

0.20

0.3

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

###### (a) α = 0.01

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

0.5

0.35

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

0.5

tiny small base large

base large huge

small base large giant

base large huge

base large huge

Alignmentscore

0.5

0.5

0.4

0.4

0.30

0.4

0.3

0.4

0.3

0.25

0.3

0.2

0.2

0.2

0.20

0.3

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(b) α = 0.05 (default)

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

0.5

0.35

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

0.5

tiny small base large

base large huge

small base large giant

base large huge

base large huge

Alignmentscore

0.5

0.5

0.4

0.4

0.30

0.4

0.3

0.4

0.3

0.25

0.3

0.2

0.2

0.2

0.3

0.20

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(c) α = 0.10 Figure 29. Sensitivity to α for CKA linear. Calibrated scores show no convergence trend regardless of significance level.

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

0.6

tiny small base large

base large huge

small base large giant

base large huge

base large huge

Alignmentscore

0.35

0.6

0.6

0.6

0.30

0.5

0.4

0.4

0.4

0.25

0.4

0.2

0.2

0.20

0.2

0.3

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

###### (a) α = 0.01

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

0.6

tiny small base large

base large huge

small base large giant

base large huge

base large huge

Alignmentscore

0.35

0.6

0.6

0.6

0.30

0.5

0.4

0.4

0.4

0.25

0.4

0.2

0.2

0.20

0.2

0.3

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(b) α = 0.05 (default)

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

0.6

tiny small base large

base large huge

small base large giant

base large huge

base large huge

Alignmentscore

0.35

0.6

0.6

0.6

0.30

0.5

0.4

0.4

0.4

0.25

0.4

0.2

0.2

0.20

0.2

0.3

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(c) α = 0.10 Figure 30. Sensitivity to α for CKA RBF. The same pattern holds: no convergence trend at any significance level.

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| |bas larg|e e| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |gia|nt| | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

0.175

0.14

tiny small base large

base large huge

small

base large huge

base large huge

0.10

Alignmentscore

0.200

0.150

0.150

0.12

0.175

0.08

0.125

0.125

0.150

0.10

0.06

0.100

0.100

0.125

0.08

0.075

0.04

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

###### (a) α = 0.01

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| |bas larg|e e| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |gia|nt| | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

0.175

0.14

tiny small base large

base large huge

small

base large huge

base large huge

0.10

Alignmentscore

0.200

0.150

0.150

0.12

0.175

0.08

0.125

0.125

0.150

0.10

0.100

0.06

0.100

0.125

0.08

0.075

0.04

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(b) α = 0.05 (default)

INet21K

###### MAE

DINOv2

###### CLIP

CLIP (INet ft)

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

0.175

0.14

tiny small base large

base large huge

small base large giant

base large huge

base large huge

0.10

Alignmentscore

0.200

0.150

0.150

0.12

0.175

0.08

0.125

0.125

0.150

0.10

0.100

0.06

0.100

0.125

0.08

0.075

0.04

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

560m1b11b73b7b1 3b7b13b 7b13b30b65b

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

BLOOM OpenLLaMA LLaMA

calibrated uncalibrated

(c) α = 0.10 Figure 31. Sensitivity to α for mKNN (k = 10). Local alignment and its scaling trend persist across all significance levels.

