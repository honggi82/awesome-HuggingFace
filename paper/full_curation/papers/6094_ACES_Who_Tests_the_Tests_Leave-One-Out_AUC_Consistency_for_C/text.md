# arXiv:2604.03922v1[cs.LG]5Apr2026

[Figure 1]

## ACES: WHO TESTS THE TESTS? LEAVE-ONE-OUT AUC CONSISTENCY FOR CODE GENERATION

A PREPRINT

Hui Sun*1,2, Yun-Ji Zhang∗1,2, Zheng Xie1, Ren-Biao Liu1,2, Yali Du1,2, Xin-Ye Li1,2, and Ming Li†1,2 1National Key Laboratory for Novel Software Technology, Nanjing University, China 2School of Artificial Intelligence, Nanjing University, China {sunh, zhangyunji, xiez, liurb, duyl, lixy, lim}@lamda.nju.edu.cn

April 7, 2026

#### ABSTRACT

Selecting LLM-generated code candidates using LLM-generated tests is challenging because the tests themselves may be incorrect. Existing methods either treat all tests equally or rely on ad-hoc heuristics to filter unreliable tests. Yet determining test correctness requires knowing which codes are correct, creating a circular dependency. Our key insight is that we need not determine test correctness at all: test votes should rank, not merely count. What matters is not how many codes pass a test, but whether the test can distinguish correct from incorrect code. We break the circular dependency via leave-one-out evaluation: hold out one test, rank codes by their aggregate scores on all remaining tests, and measure whether the held-out test’s pass/fail pattern agrees with this ranking. We formalize this agreement as the leave-one-out AUC (LOO-AUC) and prove that the expected LOO-AUC is proportional to each test’s ability to separate correct code from incorrect code. Building on this, we propose ACES (AUC ConsistEncy Scoring) with two complementary variants: ACES-C provides closed-form weights that provably approximate the oracle in expectation under a mild assumption on average test quality; ACES-O drops this assumption and iteratively optimizes a differentiable LOO-AUC objective. Both operate solely on the binary pass matrix with negligible overhead, and achieve state-of-the-art Pass@k on multiple code generation benchmarks.

#### 1 Introduction

Large language models (LLMs) have demonstrated strong capabilities in code generation [Chen et al., 2021, Guo et al., 2024, Liu et al., 2025, Du et al., 2025], yet individual generations are not always correct. A promising strategy is to scale test-time computation [Snell et al., 2025, Brown et al., 2024, Wu et al., 2025]: generate many candidate solutions together with test cases, and select the best candidates based on test execution [Li et al., 2022, Chen et al., 2023]. The central challenge is that neither the code nor the tests are guaranteed to be correct: we need reliable tests to judge code quality, and reliable code to judge test quality, but have neither.

Existing methods either treat all tests equally via majority voting or rely on heuristics: CodeT [Chen et al., 2023] scores consensus sets by size without weighting individual tests, while MBR-exec (Minimum Bayes Risk) [Shi et al., 2022] and SRank [To et al., 2024] require pairwise output comparison beyond the binary pass matrix. More heavyweight approaches co-evolve code and tests via reinforcement learning [Wang et al., 2025] or evolutionary search [Li et al., 2025a], requiring substantially more computation. Yet the circular dependency between code and test quality remains: to our knowledge, no existing method offers formal guarantees for identifying reliable tests without knowing which codes are correct.

∗Equal contribution. †Corresponding author.

What should such a criterion measure? We observe that code selection is fundamentally a ranking problem: we must order candidates so that correct solutions appear near the top. Our key insight is that, for this ranking task, a test’s value lies not in its correctness but in its ability to distinguish correct code from incorrect code: a trivially correct test that all codes pass offers no ranking signal, while a demanding test that separates candidates is valuable even if imperfect. Test votes should rank, not merely count. Under uniform counting, however, this distinction is lost: easy tests that most codes pass dilute the ranking signal, while tests that favor incorrect codes actively corrupt it.

How can we measure a test’s ability to distinguish codes without knowing which codes are correct? We break the circular dependency via leave-one-out evaluation among tests themselves. Hold out a single test from the generated test set; the remaining tests, aggregated via any reasonable weighting, induce a ranking of the code candidates. If codes ranked highly by the other tests also tend to pass the held-out test, then the test is informative. If it contradicts the ranking, it is misleading. If it is uncorrelated, it is uninformative. This evaluation requires no knowledge of code correctness; it exploits only the internal structure of the pass matrix.

We formalize this as the leave-one-out AUC (LOO-AUC): the area under the ROC curve with the remaining tests’ aggregate scores as predictions and each held-out test’s pass/fail column as the label. A test contributes to ranking to the extent that correct codes are more likely to pass it than incorrect codes; we call this difference the test’s discriminative power. This quantity is latent, since code correctness is unknown. Yet LOO-AUC requires no knowledge of code correctness. We prove that each test’s expected LOO-AUC is proportional to its discriminative power, with a coefficient that depends on the test’s pass-rate variance and the ranking quality of the remaining tests (Theorem 3). This provides the theoretical basis for principled non-uniform test weighting.

Building on this, we propose ACES (AUC Consistency Scoring) with two complementary variants. ACES-C applies the pass-rate correction in closed form under uniform weighting, provably approximating the oracle-optimal test weights in expectation under a mild assumption on average test quality (Theorem 6). ACES-O instead iteratively optimizes the test weights through a differentiable LOO-AUC objective, without requiring this assumption. Both operate solely on the binary pass matrix with negligible overhead. The two are complementary: the average-quality assumption holds for the majority of tasks, and in this regime ACES-C’s one-shot correction is near-optimal; for more challenging tasks where it fails, ACES-O’s iterative optimization remains effective.

Our main contributions are as follows:

- • Theoretical foundation. We introduce the LOO-AUC identity (Theorem 3), linking each test’s observable consistency with the ranking to its latent discriminative power. To our knowledge, this is the first provable criterion for distinguishing informative from misleading tests using only the binary pass matrix.
- • Algorithms. Building on this identity, we propose two lightweight algorithms. ACES-C provides closedform weights provably approximating the oracle-optimal weights in expectation (Theorem 6); ACES-O iteratively optimizes test weights via a differentiable LOO-AUC objective without requiring the averagequality assumption.
- • Empirical results. We show that ACES advances the state-of-the-art in Pass@k on multiple benchmarks, with ACES-O leading as a standalone method and ACES-C excelling as a plug-and-play execution-only scorer when combined with other pre-filtering mechanisms.

#### 2 Theoretical Foundations

We formalize code ranking as a weighted voting problem over the pass matrix and develop the theoretical tools that motivate ACES.

###### 2.1 Problem Setup

Given a programming problem, an LLM generates n candidate solutions C = {c1,...,cn} and m test cases T = {t1,...,tm}. Executing every candidate on every test yields the pass matrix

B ∈ {0,1}n×m, Bij = 1[ci passes tj]. (1)

Each code has an unknown correctness label yi ∈ {0,1}; we write n+ = |{i : yi = 1}| and n− = n − n+. Test quality is also unknown: some tests may have incorrect expected outputs. Since test reliability varies, we parameterize code

ranking by a weight vector over tests: w ∈ ∆m = {w ∈ Rm+ : j wj = 1} induces scores si(w) = mj=1 wjBij. Codes are ranked by si(w) in descending order (ties broken uniformly at random). We evaluate with Pass@k, the

- Table 1: Vote types for a test tj on a correct-incorrect pair (c+,c−). Each test casts a vote hj = Bc+,j − Bc−,j on their ordering; the score difference sc+ − sc− = j wjhj is the weighted aggregate.

Type Condition hj Effect

Informative c+ pass, c− fail +1 Raises sc+−sc− Uninformative Both same 0 No effect

Misleading c− pass, c+ fail −1 Lowers sc+−sc−

Tests tj

Correct

###### Incorrect

αj = 1

αj < 1

###### Perfect

###### Permissive

###### Constructive

###### Misleading

δj ∈ (0, 1)

δj ∈ (0, 1)

δj = 1

δj < 0

Helps ranking (δj > 0) Hurts ranking (δj < 0)

Figure 1: For ranking, only δj matters. Taxonomy of tests by correctness (αj) and discriminative power (δj).

probability that at least one correct code appears in the top k:

Pass@k(w) = P min

i: yi=1

rank(ci) ≤ k . (2)

The simplest baseline, majority voting, sets wunif = (1/m,...,1/m), ranking codes by total passes.

Pairwise analysis. To understand how w affects ranking, consider a correct code c+ and an incorrect code c−. Each test tj casts a vote hj = Bc+,j − Bc−,j ∈ {+1,0,−1} on their ordering (Table 1), and the score difference sc+(w)−sc−(w) = j wjhj is the weighted sum of these votes: code ranking is a weighted voting problem. Averaging over all such pairs, the probability that w places the correct code above the incorrect one defines the true AUC:

- 1

- 2

P(sc+(w) = sc−(w)). (3)

A(w) = P(sc+(w) > sc−(w)) +

Since A(w) depends on the unknown code labels, it cannot be computed directly and appears only in the analysis. Under uniform weighting, misleading tests contribute as much as informative ones, actively lowering A(w) and corrupting the ranking. Tests on which all codes agree (constant columns of B) produce only uninformative votes and are removed in preprocessing; we assume this throughout.

###### 2.2 Discriminative Power and Pass@k Bound

Codes and tests are typically sampled independently from the LLM. Conditioned on its correctness label yi, each code’s test outcomes are identically distributed and independent across tests. This conditional independence is standard when codes and tests are sampled independently from the LLM, and is shared by classical item response theory.

Definition 1 (Discriminative Power). For each test tj, define the class-conditional pass rates and discriminative power: αj = P(Bij = 1 | yi = 1), βj = P(Bij = 1 | yi = 0), δj = αj − βj.

A test is informative (δj > 0), uninformative (δj = 0), or misleading (δj < 0) according to whether correct codes are more, equally, or less likely to pass it than incorrect ones.

- Figure 1 refines this classification by test correctness: correct tests (αj = 1) are always informative, while incorrect tests (αj < 1) may be constructive (δj > 0) or misleading (δj < 0). Prior approaches that filter tests by correctness discard constructive tests together with misleading ones, losing valuable ranking signal; for ranking, only δj matters. Equivalently, δj = E[hj]: discriminative power is the expected pairwise vote. Pass@k is controlled by a signal-to-noise ratio determined by w and {δj}:

- Theorem 2 (Pass@k Bound). Define the mean signal and signal-to-noise ratio of weights w:

M(w) :=

j

wjδj, R(w) :=

M(w)2 j wj2

.

For any w with M(w) > 0 and any k ≥ 1:

n− k

R(w) 2

Pass@k(w) ≥ 1 −

exp −

, (4)

where n− = n − n+ is the number of incorrect codes. Over non-negative weights, R(w) is maximized by wj∗ ∝ max(0, δj).

- The proof is in Appendix A.2. The bound improves exponentially with R(w), but the oracle-optimal weights w∗ require knowing the unknown discriminative powers δj.

Majority voting baseline. Writing δ¯ := (1/m) mj=1 δj, substituting uniform weights gives R(wunif) = mδ¯2 and hence:

Pass@k(wunif) ≥ 1 −

n− k

exp −

mδ¯2 2

. (5)

Non-uniform weights can improve this bound by increasing R(w) beyond mδ¯2, but the optimal weights w∗ are unknown in practice.

2.3 Estimating Test Quality for Ranking: LOO-AUC Identity

The discriminative powers δj are unknown, but the pass matrix B provides m surrogate label vectors, one per test column. An informative test’s column correlates with the unknown y, while a misleading test’s column anti-correlates.

To distinguish the two, one can hold out test tj, rank codes by their aggregate scores on the remaining tests, and measure whether tj’s pass/fail pattern agrees with that ranking. Formally, the leave-one-out scores and LOO-AUC are:

Si(−j) =

j′̸=j

wj′ Bij′, LOO-AUCj(w) = AUC S(−j), B:,j , (6)

where S(−j) serves as scores and B:,j as binary labels. Both quantities are computed from the pass matrix B alone; no knowledge of y is required. LOO-AUCj measures the consistency between test tj and the remaining tests: the probability that, among codes distinguished by tj, those passing it are ranked higher by the remaining tests. A high value (LOO-AUCj > 1/2) indicates that tj agrees with the consensus ranking; a low value (LOO-AUCj < 1/2) indicates disagreement. The following identity formalizes the connection between this consistency measure and the discriminative power δj:

Theorem 3 (LOO-AUC Identity). Let A(−j)(w) denote the true AUC (Eq. 3) computed from all tests except tj. For any weights w:

E[LOO-AUCj(w)] −

- 1

- 2

= cj(w) · δj, cj(w) :=

π(1 − π) pj(1 − pj)

A(−j)(w) −

- 1

- 2

, (7)

where π = n+/n is the correct-code fraction and pj = P(Bij = 1) the marginal pass rate of tj.

- The proof is in Appendix A.3. The coefficient cj(w) has two test-dependent factors: the inverse pass-rate variance 1/[pj(1 − pj)] and the leave-one-out ranking quality A(−j)(w) − 1/2. In Section 3, we exploit this decomposition to construct test weights that target the discriminative powers δj.

#### 3 ACES: AUC Consistency Scoring

We now exploit the LOO-AUC Identity (Theorem 3) to construct non-uniform test weights that improve Pass@k, developing two complementary approaches: ACES-C, a closed-form weighting with pass-rate correction, and ACES-O, an optimized weighting via differentiable LOO-AUC optimization.

###### 3.1 ACES-C: Closed-Form Weighting

The LOO-AUC Identity (Theorem 3) gives E[LOO-AUCj(w)] − 1/2 = cj(w) · δj. When cj(w) > 0, LOO-AUC excess shares the same sign as δj, so tests scoring above 1/2 are identified as informative and those below as misleading. This holds whenever A(−j)(w) > 1/2, i.e., the remaining tests rank correct codes higher, the natural regime since the complement would mean the aggregate ranking is no better than random. To guarantee this for all j under uniform weights, we introduce:

Assumption 4. The average discriminative power is positive, and the test pool is large enough that leaving out any single test has negligible effect on the aggregate ranking. Quantitatively:

δ¯ :=

m

1 m

δj > 2

j=1

ln2 m

.

This is our only assumption beyond the model in Section 2.2; the threshold is O(1/√m), and unlike the classical weak learner condition [Freund et al., 2003], which requires every ranker to be better than random, ours requires this only on average, permitting arbitrarily many misleading tests.

Proposition 5 (Structure of cj(wunif)). Under Assumption 4, A(−j)(wunif) > 1/2 for all j, so cj(wunif) > 0. Moreover, A(−j)(wunif) is approximately constant across j.

The proof and quantitative bound are in Appendix A.4. Under uniform weights, A(−j)(wunif) − 1/2 is approximately constant across j (Proposition 5). Combined with the constant π(1 − π), the only significant test-dependent factor in

cj(wunif) is 1/[pj(1 − pj)]. Multiplying the LOO-AUC excess by pj(1 − pj) removes this distortion:

- 1

- 2 · pj(1 − pj), (8)

wj = max 0, LOO-AUCj(wunif) −

where pj = n1 ni=1 Bij is the empirical pass rate of test j. The max(0,·) ensures non-negative weights, assigning zero weight to tests with LOO-AUCj ≤ 1/2 and filtering out those identified as misleading. Since only relative magnitudes affect the induced ranking, normalization is unnecessary. The following theorem shows that the pass-rate-corrected

LOO-AUC excess recovers the discriminative power δj exactly in expectation. Theorem 6 (ACES-C Recovers Discriminative Power). Under Assumption 4, define the quality score qj := (LOO-AUCj(wunif) − 21)pj(1−pj). Then:

- 1

- 2 · δj.

E[qj] = π(1 − π) A(−j)(wunif) −

In particular, E[qj] > 0 if and only if δj > 0: the expected quality score’s sign identifies informative versus misleading tests. Since A(−j)(wunif) is nearly constant in j (Proposition 5), E[qj] ∝ δj.

Proof. By Theorem 3, E[LOO-AUCj] − 1/2 = cjδj with cj = pπ(1−π)

j(1−pj)(A(−j) − 1/2) > 0 under Assumption 4. Multiplying by pj(1−pj) cancels the 1/[pj(1−pj)] factor in cj, yielding E[qj] = π(1−π)(A(−j) − 1/2)δj. Positivity when δj > 0 follows from A(−j) > 1/2 (Proposition 5).

| |
|---|

The ACES-C weight (Eq. 8) is wj = max(0, qj): by the sign property above, the clipping assigns positive weight precisely to tests that are informative in expectation (δj > 0), targeting the oracle wj∗ ∝ max(0, δj). Quantitatively, the ACES-C weights achieve near-oracle signal-to-noise ratio: R ≥ ρ2R∗ where ρ → 1 as m grows (Corollary 8 in Appendix A.5).

###### 3.2 ACES-O: Optimized Weighting

ACES-C is efficient and closed-form, with provable guarantees under Assumption 4. To complement it in settings where the assumption may not hold, ACES-O takes an optimization-based approach, jointly optimizing the weights and the induced ranking via the following objective:

max

w∈∆m

J(w) =

m

1 2

wj LOO-AUCj(w) −

j=1

, (9)

where LOO-AUCj(w) depends on w through the leave-one-out scores S(−j)(w) (Eq. 6). The simplex constraint does not affect the induced ranking, but is necessary because unconstrained weights can grow to infinity, making the optimization ill-defined. Unlike in ACES-C, the LOO-AUC values are evaluated under the current weights w rather than the fixed uniform ranking. Tests with LOO-AUCj(w) < 1/2 are softly down-weighted rather than permanently excluded; as the optimization improves the weights, the leave-one-out ranking also improves, potentially raising LOO-AUCj(w) above 1/2 and recovering initially excluded informative tests.

Theoretical motivation. The LOO-AUC identity (Theorem 3) gives

m

wj cj(w)δj. (10)

E[J(w)] =

j=1

When cj(w) > 0, each term has the same sign as δj: informative tests increase E[J(w)] while misleading tests decrease it. Maximizing J(w) thus drives the weights toward informative tests and away from misleading ones, without requiring

##### Easy

(a) Pass matrix (b) ACES-C (c) ACES-O (d) Scores

MV -C -O c1 .60 1.00 .99 c2 .50 .75 .69 c3 .40 .35 .71 c4 .40 .30 .30 c5 .40 .30 .30 c6 .40 .15 .01 c7 .30 .00 .00 c8 .20 .00 .00

- 0

.5

- 1

- 0

.5

- 1

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

wjj

AUC .90 1.00 1.00

δj >0 δj <0

δj >0 δj <0

δj >0 δj <0

##### Hard

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

δj >0 δj <0

- 0

.5

- 1

wjj

δj >0 δj <0

1

.5

0

MV -C -O c1 .60 1.00 1.00 c2 .40 .00 .99 c3 .50 1.00 .70 c4 .60 .90 .01 c5 .50 .00 .29 c6 .50 .10 .01 c7 .40 .00 .00 c8 .30 .00 .00

AUC .60 .77 1.00

δj >0 δj <0

- Figure 2: ACES on two constructed 8 × 10 instances (full data in Appendix C.1). Test colors: green = perfect, blue =

constructive/permissive, red = misleading. (a) Pass matrix. (b),(c) LOO-AUCj (upper) and weights wj (lower, purple). (d) Scores; gold/silver/bronze mark the top-3 codes.

knowledge of δj. Assumption 4 guarantees cj(wunif) > 0 for all j (Proposition 5), providing a valid starting point. As the optimization updates w, the leave-one-out ranking improves, which in turn raises cj(w) for previously misidentified tests, creating a positive feedback loop that can succeed even when the assumption is only weakly satisfied. Since the AUC is non-differentiable, we optimize a logistic surrogate via gradient ascent; pseudocode, pre-filtering, and hyperparameters are in Appendix B.

###### 3.3 Illustrative Example

When Assumption 4 is well satisfied, ACES-C’s closed-form solution is efficient and competitive; ACES-O’s iterative refinement is most beneficial when the assumption is weakly satisfied. Figure 2 illustrates this on two constructed instances. In the Easy case (top, δ¯ = 0.16), the assumption is well satisfied: most informative tests already have LOO-AUCj > 1/2 under uniform weights, and ACES-C achieves perfect ranking. In the Hard case (bottom, δ¯ = 0.04), misleading tests are more prevalent and only 2 of 6 informative tests have LOO-AUCj > 1/2 under uniform weights. ACES-C improves the ranking using these two (AUC: 0.60 → 0.77); ACES-O’s co-evolution iteratively recovers all 6 and achieves AUC = 1.00. A step-by-step walkthrough is in Appendix C.1.

#### 4 Experiments

We evaluate ACES-C and ACES-O on three code generation benchmarks, focusing on: (1) comparison with existing reranking methods, (2) empirical validation of Assumption 4, and (3) analysis of robustness and test quality detection under LOO-AUC-based weighting.

###### 4.1 Setup

Benchmarks and generation. We evaluate on HumanEval [Chen et al., 2021] (164 problems), HumanEval+ [Liu et al., 2023] (164 problems, stricter tests), and MBPP [Austin et al., 2021] (427 problems). We use the candidate solutions and test cases from Huang et al. [2024], generated by GPT-3.5-Turbo, with approximately 200 candidates and 500 tests per problem. We report Pass@k for k ∈ {1,2,5}. For reranking methods, Pass@k follows Eq. 2; for direct inference baselines, Pass@k is the unbiased estimator of Chen et al. [2021], which averages over random k-subsets of candidates.

- Table 2: Pass@k (%) on HumanEval, HumanEval+, and MBPP. Post-hoc methods rerank GPT-3.5-Turbo candidates (n≈200, m≈500). Subscripts denote the change from GPT-3.5-Turbo direct inference. Bold/underline: best/secondbest within each group. Shaded : ours.

HumanEval HumanEval+ MBPP Method Pass@1 Pass@2 Pass@5 Pass@1 Pass@2 Pass@5 Pass@1 Pass@2 Pass@5 Direct inference GPT-3.5-Turbo 68.38 76.24 83.15 58.75 66.58 73.96 66.80 72.34 76.60 GPT-4 81.48 86.31 90.46 70.52 75.48 79.54 71.26 74.27 76.99 DeepSeek-Coder 79.30 – – – – – 70.00 – – WizardCoder 73.20 – – – – – 61.20 – – CodeLlama 62.20 – – – – – 62.20 – – Post-hoc reranking using only code × test execution

MBR-exec 72.96+4.6 76.47+0.2 79.00-4.2 62.12+3.4 67.08+0.5 71.38-2.6 70.79+4.0 73.14+0.8 74.85-1.8 CodeT 78.05+9.7 78.05+1.8 78.30-4.9 67.87+9.1 68.75+2.2 69.65-4.3 71.90+5.1 71.95-0.4 72.02-4.6 Majority Voting 80.49+12.1 82.93+6.7 83.54+0.4 69.51+10.8 73.17+6.6 76.83+2.9 68.62+1.8 70.49-1.9 72.83-3.8 ACES-C 82.93+14.6 82.93+6.7 82.93-0.2 71.34+12.6 71.95+5.4 74.39+0.4 71.19+4.4 71.43-0.9 72.37-4.2 ACES-O 84.15+15.8 85.98+9.7 86.59+3.4 74.39+15.6 75.61+9.0 79.88+5.9 72.37+5.6 73.54+1.2 73.77-2.8

Post-hoc reranking using additional information SC+Spec 73.86+5.5 73.93-2.3 74.10-9.1 63.50+4.8 64.70-1.9 65.67-8.3 71.70+4.9 71.73-0.6 71.82-4.8 Self-collaborate 74.40+6.0 – – – – – 68.20+1.4 – – MPSC 74.17+5.8 77.02+0.8 78.53-4.6 65.05+6.3 69.76+3.2 71.72-2.2 69.34+2.5 70.06-2.3 71.85-4.8 DS3 81.71+13.3 82.32+6.1 82.32-0.8 72.56+13.8 73.78+7.2 75.00+1.0 75.88+9.1 77.28+4.9 78.45+1.9 ACES-C + DS3 85.37+17.0 85.98+9.7 87.20+4.1 77.44+18.7 78.66+12.1 81.10+7.1 76.11+9.3 77.28+4.9 78.69+2.1 ACES-O + DS3 83.54+15.2 83.54+7.3 85.98+2.8 75.00+16.3 76.22+9.6 79.88+5.9 76.58+9.8 77.28+4.9 78.69+2.1

Baselines. We compare against post-hoc methods that use code×test execution: Majority Voting (wunif), CodeT [Chen

- et al., 2023] (consensus-set scoring on the pass matrix), and MBR-exec [Shi et al., 2022] (pairwise output comparison beyond the pass matrix); as well as methods requiring additional information: SC+Spec [Huang et al., 2024] (execution

+ specification consistency voting), Self-collaborate [Dong et al., 2024] (multi-agent LLM calls), MPSC [Huang

- et al., 2024] (multi-perspective consistency voting; Uniform variant, using only inter-consistency), and DS3 [Liu et al., 2026] (static analysis). We also include direct inference from GPT-3.5-Turbo, GPT-4 [Achiam et al., 2023], DeepSeek-Coder [Guo et al., 2024], WizardCoder [Luo et al., 2024], and CodeLlama [Roziere et al., 2023]. All post-hoc methods use the same GPT-3.5-Turbo candidates and test cases.

Implementation details of ACES-C and ACES-O are in Appendix B; results with additional generation models and benchmarks are in Appendix C.

###### 4.2 Main Results

Table 2 summarizes the results. Using only the binary pass matrix, ACES achieves the best Pass@k among all executionbased methods on all three benchmarks. When further combined with complementary static-analysis methods, it yields the best overall results across all benchmarks.

Execution-only methods. Within the execution-only group, ACES-O achieves the best Pass@k on all three benchmarks. On HumanEval, ACES-O reaches 84.15% Pass@1, surpassing even DS3 (81.71%) which leverages additional static-analysis signals beyond the pass matrix. On HumanEval+, whose stricter evaluation criteria increase the fraction of misleading tests among the same generated test suite, ACES-O reaches 74.39% Pass@1, again outperforming DS3 (72.56%) despite using only the pass matrix. The improvement over Majority Voting widens compared to HumanEval (+4.88% vs. +3.66% Pass@1), consistent with our theory: when misleading tests are more prevalent, principled test weighting provides greater benefit (Section 4.3). On MBPP, ACES-O leads all execution-only methods (72.37% Pass@1) but falls behind DS3 (75.88%), whose orthogonal static-analysis signals provide complementary value. Overall, ACES surpasses all baselines on HumanEval and HumanEval+ using only the pass matrix; on MBPP, only DS3 (static analysis) ranks higher.

Combination with static analysis. DS3 operates in two stages: heuristic pre-filtering of candidates, followed by static-analysis scoring (Pylint, AST similarity, cyclomatic complexity). We reuse its first-stage filtering and combine the second-stage static-analysis scores with ACES scores via a weighted sum (details in Appendix B). ACES-C + DS3

Majority Voting (MV) ACES-C ACES-O

Majority Voting (MV) ACES-C ACES-O

80

| |MV 0/14 ACES-C 0/14<br><br>MV ACES-C<br><br>Hard Midd|35/89 46/89<br><br>MV 134/136 ACES-C 134/136<br><br>le Easy| |
|---|---|---|---|
| |ACES-O 1/14 ACES-O|52/89 ACES-O 132/136| |
| | | | |
| | | | |
| | | | |
| | | | |

0.225

0.200

70

60

Numberoftasks

Passrate(%)

AUCPass@1∆∆

0.00

100

50

40

75

0.05

30

50

20

25

0.400

10

0.375

0

0

-1.0-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.10.00.10.20.30.40.50.60.70.80.91.0

0.00

Not satisfied δ¯= 0 Satisfied

0.05

- 8/52 161/187
- 9/52 171/187

MV ACES-C ACES-O

0.10

16/52 169/187

1.00 0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00

40 20 0 20 40 60 80 100

δj (ground truth)

Pass rate (%)

(a) Assumption 4 satisfaction on MBPP at Pass@1. Bottom: pass rate when δ¯ > 0 vs. δ¯ ≤ 0.

(b) Impact of each δj bin on Pass@1 (top) and AUC (bottom). Positive: helpful; negative: harmful.

- Figure 3: Empirical analysis on MBPP at Pass@1. (a) Tasks binned by δ¯; bars show pass/fail counts, lines show pass

rate; bottom panel compares pass rates by assumption status. (b) Performance impact of each δj bin upon removing its tests; full results in Appendix C.4.

improves over DS3 alone on all three benchmarks (+3.66/+4.88/+0.23 Pass@1 on HumanEval/HumanEval+/MBPP); ACES-O + DS3 similarly improves, with the largest gain on MBPP (+0.70), confirming that the two methods capture orthogonal signals. Interestingly, ACES-C + DS3 outperforms ACES-O + DS3 on HumanEval and HumanEval+ (e.g., 85.37% vs. 83.54% Pass@1 on HumanEval), because pre-filtering improves candidate quality and makes Assumption 4 better satisfied, a regime where ACES-C’s closed-form correction is already near-optimal (cf. Section 3.3). In practice, ACES-O extracts more signal for standalone reranking, while ACES-C’s closed-form weights integrate more effectively with high-quality pre-filtering and require no optimization.

Component analysis (ablation). Table 2 shows a clear progression at Pass@1: Majority Voting → ACES-C → ACES-O on all three benchmarks, validating that each component contributes independently. ACES-C already improves over majority voting (e.g., 82.93% vs. 80.49% on HumanEval), confirming that LOO-AUC captures meaningful test quality signal even in closed form; ACES-O’s iterative refinement provides further gains across all k. Additional analyses (Pass@k over finer k grids, selection vs. weighting, hyperparameter sensitivity, computational cost, etc.) are in Appendix C.

###### 4.3 Analysis

Throughout this section, we restrict analysis to non-trivial tasks (0 < n+ < n; details in Table 7) and non-constant tests (0 < pj < 1), since trivial tasks and constant-column tests carry no ranking signal.

Assumption satisfaction. Figure 3a empirically validates Assumption 4 on MBPP, which has the most non-trivial tasks and the lowest assumption satisfaction rate of the three benchmarks. We bin tasks by δ¯ and use the simplified threshold δ¯ > 0 for visual clarity (the formal threshold 2 ln2/m is small for m ≈ 500; see Appendix C.4 for the formal analysis). The assumption is well satisfied on all benchmarks; detailed satisfaction rates are in Table 7. The performance gains concentrate in the Middle region, where informative and misleading tests coexist: ACES-O passes 52/89 tasks versus 46/89 for ACES-C and 35/89 for Majority Voting. In the Easy region, all methods achieve near-perfect pass rates (≥ 97%), leaving little room for improvement; in the Hard region, no method passes more than 1 of 14 tasks, suggesting the pass matrix alone is insufficient when average test quality is very low and complementary signals such as static analysis may be needed (cf. Section 4.2). The bottom panel shows that both ACES variants outperform Majority Voting regardless of whether the assumption holds; when it is not satisfied (δ¯ ≤ 0), ACES-O has a substantial advantage over ACES-C (16/52 vs. 9/52), indicating that iterative optimization is particularly valuable in this regime.

Impact of test quality on performance. To understand why ACES-O outperforms, we measure each δj bin’s impact on performance as the drop incurred by removing its tests (Figure 3b). The key finding is an asymmetric dependence

on test quality. In the misleading-test region (δj < 0), both ACES variants are more robust: the most misleading bin (δj ≈ −0.9) reduces Pass@1 by 0.056 for Majority Voting but by only 0.049 for ACES-C and 0.030 for ACESO (46% less sensitive). This is consistent with the ACES weights having already down-weighted misleading tests, so

###### HumanEval+

###### HumanEval

###### MBPP

FP FN

TP TN

| |
|---|

| |
|---|

| |
|---|

FP 9.5% TP 79.4%

FP 8.9% TP 80.3%

FP 17.3% TP 69.9%

0.10

−·−(LOOAUC0.5)p(1p)_jjj

0.05

0.00

0.05

0.10

TN 9.3% FN 1.9%

TN 8.9% FN 1.9%

TN 9.0% FN 3.8%

1.00 0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00

1.00 0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00

1.00 0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00

δj (ground truth)

δj (ground truth)

δj (ground truth)

- Figure 4: Test quality detection. ACES-C weight (LOO-AUCj − 12)pj(1−pj) vs. ground-truth discriminative power δj for all non-trivial tests across three benchmarks. Green and red points denote informative (δj > 0) and misleading (δj < 0) tests, respectively; top marginals show the per-bin classification breakdown, with errors (FP, FN) in saturated colors and correct classifications (TP, TN) in lighter shades. Quadrant percentages report the fraction of tests classified as TP/FP/FN/TN by the sign of the ACES-C weight (summing to 100%).

that their presence or absence has little effect on the final ranking. Conversely, informative tests (δj > 0) contribute more to ACES-O than to Majority Voting (e.g., 0.030 vs. 0.012 at δj ≈ 0.3), because the optimization has up-weighted these high-quality tests. This asymmetry shows that LOO-AUC-based weighting effectively separates informative from misleading tests in the pass matrix.

Test quality detection. Figure 4 plots the ACES-C weight against ground-truth δj across all three benchmarks. Theorem 3 predicts that E[LOO-AUCj] − 21 shares the sign of δj. Consistent with this, the sign of the ACES-C weight correctly identifies at least 94.8% of informative tests across all three benchmarks. The top marginals further show that misclassified tests (FP and FN) concentrate near δj ≈ 0, where discriminative power is weak; since these borderline tests carry little ranking signal, their misclassification has limited impact on ranking quality. The FP fraction varies across benchmarks (17.3% on MBPP vs. 9.5% on HumanEval), reflecting differences in misleading-test prevalence (FP+TN = 26.3% vs. 18.8%). Accordingly, the additional gain from ACES-O’s iterative refinement is largest on MBPP (Table 2).

- 5 Related Work

Code selection with generated tests. Post-hoc selection methods rerank candidates from code LLMs [Lozhkov et al., 2024, Hui et al., 2024] using execution-based signals. CodeT [Chen et al., 2023] groups candidates into consensus sets by their binary pass profiles and scores each set by the number of passed tests, operating entirely on the pass matrix. Several methods exploit execution outputs beyond the pass matrix: MBR-exec [Shi et al., 2022] applies minimum Bayes risk decoding [Eikema and Aziz, 2022] via pairwise output comparison, SRank [To et al., 2024] clusters candidates by output equivalence, and ALGO [Zhang et al., 2023a] verifies against LLM-synthesized oracle programs. S∗ [Li et al.,

- 2025b] uses LLM-guided tournament-style selection with adaptively generated inputs. Others incorporate non-execution signals: Coder-Reviewer [Zhang et al., 2023b] uses generation log-probabilities, and LEVER [Ni et al., 2023] trains a verifier on code and execution features. MPSC [Huang et al., 2024] adds specification-level consistency voting, and DS3 [Liu et al., 2026] augments execution with static analysis. CURE [Wang et al., 2025] and CoCoEvo [Li et al., 2025a] jointly improve code and test generators via reinforcement learning or evolutionary search, at the cost of model training; self-debugging [Chen et al., 2024] iteratively refines candidates using execution feedback rather than selecting among them. On the test-quality side, ConVerTest [Taherkhani et al., 2026] applies self-consistency filtering [Wang

- et al., 2023a] before voting, and TestCase-Eval [Yang et al., 2025] evaluates LLM-generated test reliability; recent systematic studies [Li et al., 2024, Sun et al., 2024] further examine how execution-based strategies compose. Across these approaches, no formal criterion has been established for identifying test quality from the pass matrix alone; ACES fills this gap with a provable criterion for distinguishing informative from misleading tests using only binary execution outcomes.

Ranking and noisy evaluation. Code selection via weighted voting is a bipartite ranking problem [Clémençon et al., 2008, Agarwal et al., 2005, Liu, 2010]. Pairwise surrogate consistency [Gao and Zhou, 2015] justifies our logistic loss. Recent extensions address partial [Wang et al., 2023b], weakly supervised [Xie et al., 2024], and multilabel [Lukasik et al., 2025] AUC optimization; our setting fits naturally into the last framework, where each test acts as

a noisy annotator [Dawid and Skene, 1979]. Unlike RankBoost [Freund et al., 2003], which requires every ranker to beat random, our Assumption 4 requires this only on average. The noisy-comparisons literature provides minimax rates [Shah and Wainwright, 2018] and noise tolerance results [Haddad, 2022], but does not address how to identify reliable comparators. Nguyen et al. [2024] show that multiple annotators are necessary for such identification; our LOO-AUC mechanism addresses this without external supervision. The verifier paradigm [Cobbe et al., 2021, Lightman

- et al., 2024] and LLM-as-judge evaluation [Zheng et al., 2023] score candidates via trained or prompted models; ACES

requires no additional training. Our δj mirrors the item discrimination index from classical test theory [Lord and Novick, 2008], adapted to noisy, machine-generated tests; the key difference is our self-referential LOO-AUC mechanism, which replaces the external criterion classical theory assumes.

#### 6 Conclusion

We presented ACES (AUC Consistency Scoring), built on the insight that test votes should rank, not merely count. By evaluating tests against each other via leave-one-out AUC, we break the circular dependency between code and test quality assessment without any external supervision. The LOO-AUC identity (Theorem 3) establishes that each test’s observable consistency with the ranking is proportional to its latent discriminative power, providing the first provable criterion for distinguishing informative from misleading tests using only the binary pass matrix. Building on this, ACES-C provides closed-form weights that provably approximate the oracle in expectation, and ACES-O complements it by iteratively optimizing a differentiable LOO-AUC objective without requiring the average-quality assumption. Both operate solely on the binary pass matrix with negligible overhead, and achieve state-of-the-art Pass@k among execution-only methods; when combined with complementary static-analysis signals, they yield the best overall results across all benchmarks.

The LOO-AUC framework opens several research directions. Incorporating correlations among LLM-generated tests could yield tighter bounds and stronger weighting schemes. More broadly, the principle of assessing evaluator quality via internal consistency extends naturally to other noisy-evaluator settings, including LLM-as-judge ensembles, crowdsourced annotation, and verification with process reward models.

#### References

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. Arxiv Preprint Arxiv:2107.03374, 2021.

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yifan Wu, YK Li, et al. DeepSeek-Coder: When the large language model meets programming–the rise of code intelligence. Arxiv Preprint Arxiv:2401.14196, 2024.

Ren-Biao Liu, Anqi Li, Chaoding Yang, Hui Sun, and Ming Li. Revisiting Chain-of-Thought in code generation: Do language models need to learn reasoning before coding? In Proceedings of the 42nd International Conference on Machine Learning, volume 267, pages 38809–38826. PMLR, 13–19 Jul 2025.

Yali Du, Hui Sun, and Ming Li. Post-incorporating code structural knowledge into pretrained models via icl for code translation. IEEE Transactions on Software Engineering, 51(11):3038–3055, 2025.

Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning. In The Thirteenth International Conference on Learning Representations, 2025.

Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher Ré, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. Arxiv Preprint Arxiv:2407.21787, 2024.

Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, and Yiming Yang. Inference scaling laws: An empirical analysis of compute-optimal inference for LLM problem-solving. In The Thirteenth International Conference on Learning Representations, 2025.

Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alphacode. Science, 378

(6624):1092–1097, 2022. Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. CodeT: Code generation with generated tests. In The Eleventh International Conference on Learning Representations, 2023.

Freda Shi, Daniel Fried, Marjan Ghazvininejad, Luke Zettlemoyer, and Sida I. Wang. Natural language to code translation with execution. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3533–3546. Association for Computational Linguistics, 2022.

Hung Quoc To, Minh Huynh Nguyen, and Nghi D. Q. Bui. Functional overlap reranking for neural code generation. In Findings of the Association for Computational Linguistics: Acl 2024, pages 3686–3704. Association for Computational Linguistics, 2024.

Yinjie Wang, Ling Yang, Ye Tian, Ke Shen, and Mengdi Wang. CURE: Co-evolving coders and unit testers via reinforcement learning. In The Thirty-Ninth Annual Conference on Neural Information Processing Systems, 2025.

Kefan Li, Yuan Yuan, Hongyue Yu, Tingyu Guo, and Shijie Cao. CoCoEvo: Co-evolution of programs and test cases to enhance code generation. Ieee Transactions on Evolutionary Computation, pages 1–1, 2025a.

Yoav Freund, Raj Iyer, Robert E Schapire, and Yoram Singer. An efficient boosting algorithm for combining preferences. Journal of Machine Learning Research, 4(Nov):933–969, 2003.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and LINGMING ZHANG. Is your code generated by chatGPT really correct? rigorous evaluation of large language models for code generation. In Thirty-Seventh Conference on Neural Information Processing Systems, 2023.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. Arxiv Preprint Arxiv:2108.07732, 2021.

Baizhou Huang, Shuai Lu, Xiaojun Wan, and Nan Duan. Enhancing large language models in coding through multiperspective self-consistency. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (volume 1: Long Papers), pages 1429–1450. Association for Computational Linguistics, 2024.

Yihong Dong, Xue Jiang, Zhi Jin, and Ge Li. Self-collaboration code generation via ChatGPT. Acm Transactions on Software Engineering and Methodology, 33(7), 2024.

Ren-Biao Liu, Jiang-Tian Xue, Chao-Zeng Ma, Hui Sun, Xin-Ye Li, and Ming Li. Dynamic-Static synergistic selection method for candidate code solutions with generated test cases. In Proceedings of the Aaai Conference on Artificial Intelligence, pages 32096–32104, 2026.

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. Arxiv Preprint Arxiv:2303.08774, 2023.

Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. WizardCoder: Empowering code large language models with evol-instruct. In The Twelfth International Conference on Learning Representations, 2024.

Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, et al. Code llama: Open foundation models for code. Arxiv Preprint Arxiv:2308.12950, 2023.

Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. Starcoder 2 and the stack v2: The next generation. Arxiv Preprint Arxiv:2402.19173, 2024.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. Qwen2. 5-Coder technical report. Arxiv Preprint Arxiv:2409.12186, 2024.

Bryan Eikema and Wilker Aziz. Sampling-based approximations to minimum Bayes risk decoding for neural machine translation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10978–10993. Association for Computational Linguistics, 2022.

Kexun Zhang, Danqing Wang, Jingtao Xia, William Yang Wang, and Lei Li. ALGO: Synthesizing algorithmic programs with generated oracle verifiers. In Thirty-Seventh Conference on Neural Information Processing Systems, 2023a.

Dacheng Li, Shiyi Cao, Chengkun Cao, Xiuyu Li, Shangyin Tan, Kurt Keutzer, Jiarong Xing, Joseph E. Gonzalez, and Ion Stoica. S*: Test time scaling for code generation. In Findings of the Association for Computational Linguistics: Emnlp 2025, pages 15964–15978. Association for Computational Linguistics, 2025b.

Tianyi Zhang, Tao Yu, Tatsunori Hashimoto, Mike Lewis, Wen-tau Yih, Daniel Fried, and Sida Wang. Coder reviewer reranking for code generation. In Proceedings of the 40th International Conference on Machine Learning, pages 41832–41846. PMLR, 2023b.

Ansong Ni, Srini Iyer, Dragomir Radev, Veselin Stoyanov, Wen-tau Yih, Sida Wang, and Xi Victoria Lin. LEVER: Learning to verify language-to-code generation with execution. In Proceedings of the 40th International Conference on Machine Learning, pages 26106–26128. PMLR, 2023.

Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. Teaching large language models to self-debug. In The Twelfth International Conference on Learning Representations, 2024.

Hamed Taherkhani, Alireza DaghighFarsoodeh, Mohammad Chowdhury, Hung Viet Pham, and Hadi Hemmati. Consistency meets verification: Enhancing test generation quality in large language models without ground-truth solutions. Arxiv Preprint Arxiv:2602.10522, 2026.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023a.

Zheyuan Yang, Zexi Kuang, Xue Xia, and Yilun Zhao. Can LLMs generate high-quality test cases for algorithm problems? TestCase-eval: A systematic evaluation of fault coverage and exposure. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (volume 2: Short Papers), pages 1050–1063. Association for Computational Linguistics, 2025.

Haau-Sing Li, Patrick Fernandes, Iryna Gurevych, and André FT Martins. DOCE: Finding the sweet spot for executionbased code generation. Arxiv Preprint Arxiv:2408.13745, 2024.

Zhihong Sun, Yao Wan, Jia Li, Hongyu Zhang, Zhi Jin, Ge Li, and Chen Lyu. Sifting through the chaff: On utilizing execution feedback for ranking the generated code candidates. pages 229–241. Association for Computing Machinery, 2024.

Stéphan Clémençon, Gábor Lugosi, and Nicolas Vayatis. Ranking and empirical minimization of u-statistics. The Annals of Statistics, pages 844–874, 2008.

Shivani Agarwal, Thore Graepel, Ralf Herbrich, Sariel Har-Peled, and Dan Roth. Generalization bounds for the area under the roc curve. Journal of Machine Learning Research, 6(14):393–425, 2005.

Tie-Yan Liu. Learning to rank for information retrieval. In Proceedings of the 33rd International Acm Sigir Conference on Research and Development in Information Retrieval, page 904. Association for Computing Machinery, 2010.

Wei Gao and Zhi-Hua Zhou. On the consistency of AUC pairwise optimization. In International Joint Conference on Artificial Intelligence, pages 939–945, 2015.

Zitai Wang, Qianqian Xu, Zhiyong Yang, Yuan He, Xiaochun Cao, and Qingming Huang. Optimizing partial area under the top-k curve: Theory and practice. Ieee Transactions on Pattern Analysis and Machine Intelligence, 45(4): 5053–5069, 2023b.

Zheng Xie, Yu Liu, Hao-Yuan He, Ming Li, and Zhi-Hua Zhou. Weakly supervised auc optimization: A unified partial auc approach. Ieee Transactions on Pattern Analysis and Machine Intelligence, 46(7):4780–4795, 2024.

Michal Lukasik, Lin Chen, Harikrishna Narasimhan, Aditya Krishna Menon, Wittawat Jitkrittum, Felix X. Yu, Sashank J. Reddi, Gang Fu, Mohammadhossein Bateni, and Sanjiv Kumar. Bipartite ranking from multiple labels: On loss versus label aggregation. In Proceedings of the 42nd International Conference on Machine Learning, volume 267, pages 41074–41102. PMLR, 13–19 Jul 2025.

Alexander Philip Dawid and Allan M Skene. Maximum likelihood estimation of observer error-rates using the em algorithm. Journal of the Royal Statistical Society: Series C (applied Statistics), 28(1):20–28, 1979.

Nihar B. Shah and Martin J. Wainwright. Simple, robust and optimal ranking from pairwise comparisons. Journal of Machine Learning Research, 18(199):1–38, 2018.

Dany Haddad. Noise tolerance of learning to rank under class-conditional label noise. Arxiv Preprint Arxiv:2208.02126, 2022.

Tri Nguyen, Shahana Ibrahim, and Xiao Fu. Noisy label learning with instance-dependent outliers: Identifiability via crowd wisdom. In Advances in Neural Information Processing Systems, 2024.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems, 2021. Arxiv Preprint Arxiv:2110.14168, 2021.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Thirty-Seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023.

Frederic M Lord and Melvin R Novick. Statistical Theories of Mental Test Scores. IAP, 2008. Wassily Hoeffding. Probability inequalities for sums of bounded random variables. Journal of the American Statistical

Association, 58(301):13–30, 1963.

Yunhui Xia, Wei Shen, Yan Wang, Jason Klein Liu, Huifeng Sun, Siyue Wu, Jian Hu, and Xiaolong Xu. LeetCodeDataset: A temporal dataset for robust evaluation and efficient training of code LLMs. arXiv preprint arXiv:2504.14655, 2025.

### Appendix

###### A Proofs and Theoretical Details 15

- A.1 Proof Preliminaries . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- A.2 Proof of Theorem 2 (Pass@k Bound) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- A.3 Proof of Theorem 3 (LOO-AUC Identity) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- A.4 Proof of Proposition 5 (Structure of cj(wunif)) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- A.5 Oracle Approximation Guarantee . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

###### B Algorithm Details 20

- B.1 ACES-C . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- B.2 ACES-O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- B.3 Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- B.4 Combination with DS3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

###### C Additional Experimental Results 22

- C.1 Illustrative Example . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- C.2 Additional Generation Models and Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- C.3 Pass@k vs. k . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- C.4 Assumption Satisfaction Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- C.5 Selection vs. Weighting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- C.6 Sensitivity to Number of Tests . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- C.7 Sensitivity to Number of Candidate Codes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- C.8 Effect of Pre-Filtering Cutoff . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- C.9 ACES-O Convergence . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- C.10 ACES-O Hyperparameter Sensitivity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- C.11 Computational Cost . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- C.12 Pairwise Vote Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- C.13 Generation Prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

#### A Proofs and Theoretical Details

This section provides complete proofs for all theoretical results in the main text: the Pass@k bound (A.2), the LOO-AUC identity (A.3), the structure of cj(wunif) (A.4), and the oracle approximation guarantee (A.5).

- A.1 Proof Preliminaries We summarize the notation and model properties used throughout the proofs. Notation. Throughout the appendix, we use the following notation from the main text:

- • αj = P(Bij = 1 | yi = 1), βj = P(Bij = 1 | yi = 0): class-conditional pass rates of test tj.
- • δj = αj − βj: discriminative power of test tj (Definition 1).
- • C+ = {ci : yi = 1}, C− = {ci : yi = 0}: sets of correct and incorrect codes.
- • n+ = |C+|, n− = |C−|, π = n+/n: counts and fraction of correct codes.
- • pj = P(Bij = 1) = παj + (1 − π)βj: marginal pass rate of test tj.

Model properties. As stated in Section 2.2, codes are sampled independently from the LLM. Conditioned on the correctness label yi, each code’s test outcomes are identically distributed and mutually independent across tests, since the class-conditional pass rates αj and βj (Definition 1) depend only on the test and the label, not on the specific code. It follows that the true AUC A(w) is the same for every pair (c+ ∈ C+, c− ∈ C−) and that two codes from the same class have identically distributed scores.

Hoeffding’s inequality. Several proofs below rely on the following classical concentration inequality, which we state here for reference.

Lemma 7 (Hoeffding’s inequality [Hoeffding, 1963]). If Z1,...,Zm are independent random variables with Zj ∈ [aj,bj], then for any t > 0:

P

 

m

j=1

Zj − E

 

m

j=1

Zj

  ≤ −t

  ≤ exp −

2t2

m j=1(bj − aj)2

.

- A.2 Proof of Theorem 2 (Pass@k Bound)

- Theorem (2, restated). Define the mean signal and signal-to-noise ratio of weights w:

M(w) :=

j

wjδj, R(w) :=

M(w)2 j wj2

.

For any w with M(w) > 0 and any k ≥ 1: Pass@k(w) ≥ 1 −

n− k

R(w) 2

exp −

,

where n− = n − n+ is the number of incorrect codes. Proof. The proof proceeds in two steps: first we reduce Pass@k to a pairwise error probability, then we bound that probability using Hoeffding’s inequality.

- Step I: Reducing Pass@k to pairwise comparisons. Let c∗ = arg maxc∈C+ sc(w) be the highest-scoring correct code, and define

Rc∗ = |{c− ∈ C− : sc−(w) ≥ sc∗(w)}|, i.e., the number of incorrect codes that score at least as high as the best correct code. If Pass@k fails (no correct code in the top k), then all k top-scoring codes are incorrect and each scores ≥ sc∗(w), so Rc∗ ≥ k. We now bound E[Rc∗]. For any fixed correct code c+ (not necessarily c∗) and any incorrect code c−, the event {sc− ≥ sc∗} implies {sc− ≥ sc+} (since c∗ has the highest score among correct codes). Therefore:

P(sc− ≥ sc∗) ≤ P(sc− ≥ sc+).

By exchangeability, the right-hand side is the same for every pair (c+,c−). Summing over all n− incorrect codes:

E[Rc∗] ≤ n− · P(sc− ≥ sc+). Applying Markov’s inequality:

n− · P(sc− ≥ sc+) k

P(Pass@k fails) ≤ P(Rc∗ ≥ k) ≤

. It remains to bound the pairwise error probability P(sc− ≥ sc+).

- Step II: Bounding pairwise error via Hoeffding’s inequality. Fix a pair (c+,c−) and define the per-test vote hj = Bc+,j − Bc−,j ∈ {−1,0,+1}. Then the score difference is:

X = sc+(w) − sc−(w) =

m

j=1

wj hj.

Since tests are independently sampled, conditioned on (yc+,yc−) = (1,0), the test outcomes for c+ and c− are independent across tests. This means {hj}mj=1 are mutually independent random variables with:

E[hj] = αj − βj = δj, E[X] =

j

wjδj = M(w) > 0.

Each wjhj ∈ [−wj,wj]. Applying Lemma 7 with Zj = wjhj (so aj = −wj, bj = wj, bj −aj = 2wj) and t = M(w):

P(sc− ≥ sc+) = P(X ≤ 0) ≤ exp −

2M(w)2 j(2wj)2

= exp −

M(w)2 2 j wj2

= exp −

R(w) 2

.

Combining. Substituting the Hoeffding bound into Step I:

P(Pass@k fails) ≤

n− k

exp −

R(w) 2

, and therefore:

Pass@k(w) = 1 − P(Pass@k fails) ≥ 1 −

n− k

exp −

R(w) 2

.

- Step III: Oracle-optimal weights. Since R(w) = M(w)2/ j wj2 is scale-invariant, we may normalize j wj2 = 1, so that R(w) = ( j wjδj)2. For wj ≥ 0, any test with δj ≤ 0 contributes non-positively to j wjδj, so the optimum requires wj∗ = 0 for all such tests. Among the remaining tests (δj > 0), by Cauchy–Schwarz:

wjδj ≤

j: δj>0

wj2 ·

j: δj>0

δj2,

j: δj>0

with equality iff wj ∝ δj. Under the normalization j wj2 = 1, this gives wj∗ ∝ max(0, δj) and R(w∗) = j:δ

j>0 δj2. Since j:δ

j>0 δj ≥ mδ¯, and by Cauchy–Schwarz with g = |{j : δj > 0}| ≤ m: R∗ =

j≤0 δj ≤ 0, we have j:δ

(mδ¯)2 m

= mδ¯2 = R(wunif).

δj2 ≥

| |
|---|

j: δj>0

###### A.3 Proof of Theorem 3 (LOO-AUC Identity)

- Theorem (3, restated). For any weights w:

π(1 − π) pj(1 − pj)

- 1

- 2

- 1

- 2

A(−j)(w) −

E[LOO-AUCj(w)] −

= cj(w) · δj, cj(w) :=

,

where π = n+/n is the fraction of correct codes and pj = P(Bij = 1) is the marginal pass rate of test j.

Proof. Setup. Recall the true AUC of the leave-one-out ranking:

###### A(−j)(w) = P Sc(−j)

###### (w) > Sc(−j)

(w) +

−

+

1 2

P Sc(−j)

###### (w) = Sc(−j)

###### (w) ,

−

+

and LOO-AUCj(w) = AUC(S(−j),B:,j), where S(−j) are the leave-one-out scores and B:,j are the pass/fail labels of test tj. By the probabilistic interpretation of AUC, LOO-AUCj(w) equals the probability that, for a randomly drawn pair of codes (one from those that pass test tj and one from those that fail it), the passer receives a higher leave-one-out score than the failer (with ties counted as 1/2).

Formally, let cp and cf denote codes drawn uniformly at random from {i : Bij = 1} and {i : Bij = 0}, respectively. We emphasize that cp and cf are defined by the observed pass/fail outcome on test tj, not by the true labels: unlike c+ ∈ C+ and c− ∈ C− (which are truly correct and incorrect), a passer cp may be incorrect and a failer cf may be correct. With this notation:

- 1

- 2

###### E[LOO-AUCj(w)] = P Sc(−j)

> Sc(−j)

P Sc(−j)

###### = Sc(−j)

+

###### .

p

p

f

f

Case decomposition. We condition on the true labels (yc

). Let q10, q01, and qsame denote the probabilities that the pair falls into each case (these sum to 1):

###### ,yc

p

f

ycp ycf Prob. Cond. AUC∗ correct incorrect q10 A(−j)(w)

incorrect correct q01 1 − A(−j)(w) same class same class qsame 1/2

∗The last column is the conditional AUC of the leave-one-out ranking for each case:

P Sc(−j)

> Sc(−j)

###### | yc

###### ,yc

p

f

p

f

+

- 1

- 2

P Sc(−j)

###### = Sc(−j)

###### | yc

###### ,yc

p

f

p

f

###### .

In the first row, cp is correct and cf is incorrect, so this reduces to A(−j)(w) by definition. In the second row the roles are reversed, giving 1 − A(−j)(w). In the third row, both codes belong to the same class; by exchangeability their leave-one-out scores are identically distributed, so the last column equals 1/2. By the law of total probability, and substituting qsame = 1 − q10 − q01:

- 1

- 2

E[LOO-AUCj(w)] = q10 A(−j) + q01(1 − A(−j)) + (1 − q10 − q01) ·

- 1

- 2

- 1

- 2 − q01 A(−j) −

- 1

- 2

+ q10 A(−j) −

=

- 1

- 2

- 1

- 2

+ (q10 − q01) A(−j)(w) −

=

.

Computing q10 and q01. The quantity q10 is the probability that cp is correct and cf is incorrect. Recall that π = n+/n is the fraction of correct codes, αj = P(Bij = 1 | yi = 1) and βj = P(Bij = 1 | yi = 0) are the class-conditional pass rates, and pj = παj + (1 − π)βj is the marginal pass rate. By Bayes’ rule, the probability that a code passing test j is correct is:

P(Bij = 1 | yi = 1)P(yi = 1) P(Bij = 1)

παj pj

P(yi = 1 | Bij = 1) =

, and the probability that a code failing test j is incorrect is:

=

P(Bij = 0 | yi = 0)P(yi = 0) P(Bij = 0)

(1 − π)(1 − βj) 1 − pj

P(yi = 0 | Bij = 0) =

=

.

Since cp and cf are drawn independently, q10 is the product of these two probabilities:

q10 =

(1 − π)(1 − βj) 1 − pj

παj pj ·

.

For q01 (cp is incorrect, cf is correct), the probability that a code passing test j is incorrect is:

P(Bij = 1 | yi = 0)P(yi = 0) P(Bij = 1)

(1 − π)βj pj

P(yi = 0 | Bij = 1) =

, and the probability that a code failing test j is correct is:

=

P(Bij = 0 | yi = 1)P(yi = 1) P(Bij = 0)

π(1 − αj) 1 − pj

P(yi = 1 | Bij = 0) =

. Therefore:

=

(1 − π)βj pj ·

π(1 − αj) 1 − pj

q01 =

.

Computing q10 − q01. Factoring out the common terms:

π(1 − π) pj(1 − pj)

αj(1 − βj) − βj(1 − αj)

q10 − q01 =

π(1 − π) pj(1 − pj)

π(1 − π)δj pj(1 − pj)

αj − βj =

=

. Substituting back into the total probability expression:

π(1 − π) pj(1 − pj) · A(−j)(w) −

- 1

- 2

- 1

- 2

- 1

- 2

= (q10 − q01) A(−j)(w) −

E[LOO-AUCj(w)] −

=

= cj(w)

· δj.

| |
|---|

###### A.4 Proof of Proposition 5 (Structure of cj(wunif))

Proposition (5, restated). Under Assumption 4, A(−j)(wunif) > 1/2 for all j, so cj(wunif) > 0. Moreover, A(−j)(wunif) is approximately constant across j.

Proof. Under uniform weights wk = 1/m, removing test j gives wk(−j) = 1/m for k ̸= j and wj(−j) = 0. The mean signal and sum of squared weights are:

mδ¯− δj m

m − 1 m2

1 m k̸=j

(wk(−j))2 =

M(−j) =

δk =

,

.

k

Therefore:

(mδ¯− δj)2 m−1

M(−j) 2 k(wk(−j))2

R(−j)(wunif) =

.

=

√

Since |δj| ≤ 1 (each δj = αj − βj is a difference of probabilities) and Assumption 4 implies mδ¯ > 2

mln2 > 1, the minimum of R(−j)(wunif) over δj ∈ [−1,1] is attained at δj = 1:

(mδ¯− 1)2 m−1

R(−j)(wunif) ≥

.

We claim this lower bound exceeds 2ln2. Let a = mδ¯. From Assumption 4, mδ¯2 > 4ln2, so m < a2/(4ln2) and 2(m−1)ln2 = 2mln2 − 2ln2 < a2/2 − 2ln2. Therefore:

a2 2

(a − 2)2 2

(a − 1)2 − 2(m−1)ln2 > (a − 1)2 −

+ (2ln2 − 1) > 0,

+ 2ln2 =

where the last step uses 2ln2 > 1. Hence R(−j)(wunif) > 2ln2 for every j. To connect this to the AUC, recall from Section A.2 that for the score difference X = sc+(w) − sc−(w), the Hoeffding bound gives P(X ≤ 0) ≤ exp(−R(w)/2). Since the true AUC satisfies

- 1

- 2

1 − A(w) = P(X < 0) +

P(X = 0) ≤ P(X ≤ 0),

we obtain 1 − A(w) ≤ exp(−R(w)/2) for any w with M(w) > 0. Applying this to the leave-one-out ranking:

R(−j)(wunif) 2

- 1

- 2

1 − A(−j)(wunif) ≤ exp −

< exp(−ln2) =

. Therefore A(−j)(wunif) > 1/2 for every j, which gives cj(wunif) > 0.

###### Approximate constancy of A(−j). The bound above gives

(mδ¯− 1)2 2(m−1)

1 − A(−j)(wunif) ≤ exp −

for every j. Since all A(−j)(wunif) ∈ (1/2, 1], for any a ≥ b > 1/2 we have a − b ≤ 1 − b = max(1 − a, 1 − b), so

(mδ¯− 1)2 2(m−1)

A(−j) − A(−k) ≤ max 1 − A(−j), 1 − A(−k) ≤ exp −

.

The same reasoning applies to A(wunif), since R(wunif) = mδ¯2 > 4ln2 gives 1 − A(wunif) ≤ exp(−mδ¯2/2), which is also exponentially small. Therefore all values A(−j)(wunif) and A(wunif) are exponentially close to each other, and A(−j)(wunif) − 1/2 is approximately constant across j.

| |
|---|

###### A.5 Oracle Approximation Guarantee

Corollary 8. Under Assumption 4, let aj = A(−j)(wunif) − 1/2 and ρ = amin/amax. Define the population ACES-C weights w¯j := max(0, E[qj]), where qj is the quality score from Theorem 6. Since aj > 0, this gives w¯j = π(1−π)aj max(0, δj). Then:

(mδ¯− 1)2 2(m−1)

R( ¯w) ≥ ρ2 R∗, ρ ≥ 1 − 2exp −

,

where R∗ = j:δ

j>0 δj2 is the oracle signal-to-noise ratio (Theorem 2). Since ρ → 1 as m grows, the population weights achieve near-oracle performance. Proof. By definition and Theorem 6:

w¯j = π(1−π)aj max(0, δj),

where aj > 0 by Proposition 5. Only tests with δj > 0 receive positive population weight, so:

M( ¯w) =

aj δj2 · π(1−π),

j: δj>0

j

w¯j2 =

a2j δj2 · π2(1−π)2.

j: δj>0

Since R is scale-invariant, π(1−π) cancels:

aj δj2 2 j:δj>0 a2j δj2 ≥

R( ¯w) = j:δj>0

j>0 δj2 2 a2max j:δ

a2min j:δ

j>0 δj2

= ρ2 R∗.

For the bound on ρ: by Appendix A.4,

(mδ¯− 1)2 2(m−1)

1 − A(−j) ≤ exp −

for all j.

Since amax ≤ 1/2 (because A(−j) ≤ 1) and amin ≥ 1/2 − exp −(mδ¯− 1)2/[2(m−1)] :

1/2 − exp −(mδ¯− 1)2/[2(m−1)] 1/2

(mδ¯− 1)2 2(m−1)

amin amax ≥

= 1 − 2exp −

ρ =

.

| |
|---|

#### B Algorithm Details

This section provides pseudocode for both ACES variants, implementation details, and the combination procedure with DS3 referenced in Section 4.

###### B.1 ACES-C

- Algorithm 1 implements ACES-C (Section 3.1). Under uniform initial weights, the leave-one-out scores reduce to unweighted row sums with one column removed; since the AUC is scale-invariant, normalization is unnecessary. The algorithm has no tunable parameters: it computes the LOO-AUC for each test, applies the pass-rate correction (Eq. 8), and returns the weighted scores in a single pass.

- Algorithm 1 ACES-C (Closed-Form Weighting)

Require: Pass matrix B ∈ {0,1}n×m Ensure: Code scores s1,...,sn

- 1: for j = 1 to m do
- 2: Compute Si(−j) = j′̸=j Bij′ for all i {LOO scores; ∝ Eq. (6) under wunif}
- 3: Compute LOO-AUCj(wunif) = AUC(S(−j),B:,j) {AUC is scale-invariant}
- 4: end for
- 5: pj ← n1 i Bij for all j {empirical pass rate}

- 6: Set wj ← max(0, LOO-AUCj(wunif) − 1/2) · pj(1 − pj) for all j {Eq. (8)}
- 7: return si = j wjBij for all i

B.2 ACES-O

- Algorithm 2 implements ACES-O (Section 3.2), which maximizes the objective J(w) = j wj (LOO-AUCj(w) − 1/2) via gradient ascent. Since the exact LOO-AUC involves indicator functions and is therefore non-differentiable, we replace it with a logistic surrogate. For each test tj, let Pj = {i : Bij = 1} and Nj = {i : Bij = 0} denote the sets of codes that pass and fail tj, respectively. The surrogate LOO-AUC is:

LOO-AUCj(w) =

1 |Pj||Nj| i∈P

j k∈Nj

σ γ Si(−j)(w) − Sk(−j)(w) , (11)

where σ(x) = 1/(1 + e−x) is the logistic function and γ > 0 is a sharpness parameter controlling the surrogate tightness. As γ → ∞, σ(γx) approaches the step function and the surrogate recovers the exact LOO-AUC. This surrogate is smooth and statistically consistent [Gao and Zhou, 2015], enabling gradient computation via automatic differentiation. The weights are parameterized as w = softmax(ℓ) with learnable logits ℓ ∈ Rm; this ensures w ∈ ∆m throughout optimization.

- Algorithm 2 ACES-O (Optimized Weighting)

Require: Pass matrix B ∈ {0,1}n×m, sharpness γ, learning rate η, steps T Ensure: Code scores s1,...,sn

- 1: Initialize logits ℓ ← 0 ∈ Rm
- 2: for t = 1 to T do
- 3: w ← softmax(ℓ)
- 4: for j = 1 to m do
- 5: Si(−j)(w) ← j′̸=j wj′Bij′ for all i {leave-one-out scores}
- 6: Compute LOO-AUCj(w) via logistic surrogate (Eq. 11)
- 7: end for
- 8: ℓ ← ℓ + η ∇ℓ j wj LOO-AUCj(w) {gradient ascent (Adam in practice)}
- 9: end for
- 10: w ← softmax(ℓ)
- 11: return si = j wjBij for all i

The surrogate objective is non-convex because LOO-AUCj(w) depends on w through the leave-one-out scores, so gradient ascent converges to a stationary point rather than a global maximum. Initializing from wunif (i.e., ℓ = 0) provides a favorable starting point, since Proposition 5 guarantees A(−j)(wunif) > 1/2 for all j under Assumption 4: informative tests are up-weighted in early iterates, improving the ranking and further separating informative from misleading tests in subsequent gradient steps. Across all benchmarks, we observe stable convergence with no restarts required.

###### B.3 Implementation Details

For ACES-O, we adopt a two-stage pipeline: first pre-filter to the top-K candidates by majority vote, then optimize weights on this shortlist. Pre-filtering removes obviously low-quality candidates and reduces the per-step cost from O(nm2) to O(Km2). ACES-C requires no pre-filtering and runs on all n candidates directly. ACES-O uses Adam with surrogate sharpness γ=10, learning rate η=0.01, and T=90 steps on HumanEval (η=0.05, T=90 on HumanEval+ and MBPP), with top-K=24 pre-filtering (K=32 on MBPP).

###### B.4 Combination with DS3

As described in Section 4.2, we combine ACES with DS3 [Liu et al., 2026]. We first summarize DS3’s scoring mechanism, then describe the combined formulation.

DS3 scoring. DS3 first applies a pre-filtering stage that sanitizes candidate code (AST-based extraction of the longest valid code block followed by dependency-driven dead-code elimination) and validates test inputs (AST-based argument extraction with signature-conformance checking). This removes malformed candidates and invalid tests, reducing the sets from n to n′ ≤ n and from m to m′ ≤ m. On the filtered set, DS3 computes two signals. A static quality vector q ∈ [0,1]n

′

scores each candidate by combining a normalized Pylint score (code quality) with a normalized cyclomatic-complexity score (lower complexity yields a higher score). A pairwise consensus matrix P ∈ [0,1]n

′×n′

captures behavioral similarity:

Pik =

m′

1 m′

1 exec(ci,tj) = exec(ck,tj) ,

j=1

where exec(ci,tj) denotes the full output of candidate ci on test tj. The final DS3 score is q∗ = P · q: each candidate receives a consensus-weighted sum of static quality across all candidates, favoring solutions that are both well-structured and behaviorally consistent.

Combined formulation. ACES and DS3 capture complementary signals: ACES computes adaptive test weights from the pass matrix to produce per-candidate scores si(w) = j wjBij, while DS3 combines static code quality with pairwise output consensus. We reuse DS3’s pre-filtering to obtain a cleaned candidate set of size n′, then combine both signals on the filtered candidates via

r = α In′ + (1 − α)P · β s(w) + (1 − β)q , (12)

where In′ is the identity matrix and α,β ∈ [0,1]. The coefficient β blends the per-candidate scoring signal: ACES scores s(w) at β = 1 and static quality q at β = 0. The coefficient α controls the aggregation: α = 1 scores each

candidate independently, while α = 0 applies pairwise consensus weighting as in DS3. Setting α = β = 0 recovers the original DS3 score P · q. Both s(w) and q lie in [0,1]n

′

by construction, so no additional normalization is required. All DS3 components (pre-filtering, static scoring, pairwise consensus matrix, and all associated hyperparameters) use the official implementation and default settings of Liu et al. [2026].

#### C Additional Experimental Results

This section provides supplementary experiments: an illustrative walkthrough (C.1), additional models and benchmarks (C.2), extended Pass@k curves (C.3), assumption verification (C.4), selection vs. weighting analysis (C.5), sensitivity analyses (C.6–C.8), convergence (C.9), hyperparameter sensitivity (C.10), runtime (C.11), vote statistics (C.12), and generation prompts (C.13).

###### C.1 Illustrative Example

Tables 3 and 4 provide the full data for the two 8 × 10 instances in Figure 2 (3 correct codes, 5 incorrect, 10 tests sorted by δj descending). Each table reports the binary pass matrix, per-test statistics (αj, βj, δj), and the LOO-AUC and weights for both ACES-C and ACES-O. We walk through each case below, following the same order: MV → ACES-C → ACES-O.

Easy case (Table 3). Eight of ten tests are informative (δj > 0) and only two are misleading, so the signal-to-noise ratio is favorable. MV assigns uniform weight to all tests and achieves AUC = 0.90; however, the lowest correct code c3 ties with three incorrect ones at vote = 4, making the top-1 selection unreliable. ACES-C computes LOO-AUC under uniform weights and finds that 6 of the 8 informative tests have LOO-AUCj > 1/2; it assigns non-zero weight to these

- 6 via Eq. (8), concentrating weight on the most discriminative tests (t1, t2, t4 each receive w = 0.25). This breaks the MV tie and achieves AUC = 1.00: all three correct codes now rank above all five incorrect ones. ACES-O also reaches

AUC = 1.00, further concentrating weight on the single strongest test (t1, w = 0.40). When test quality is high, the closed-form ACES-C already suffices for perfect separation.

- Table 3: Easy case: 8 × 10 pass matrix with 8 informative and 2 misleading tests (δ¯ = 0.16). Tests are sorted by δj descending. MV AUC: 0.90; ACES-C AUC: 1.00; ACES-O AUC: 1.00.

correct (αj =1) incorrect (αj <1) perf. constructive (0 < δj < 1) misleading (δj < 0)

t1 t2 t3 t4 t5 t7 t6 t8 t9 t10 MV -C -O

✓ c1 1 1 1 1 1 0 1 0 0 0 .60 1.00 1.00 ✓ c2 1 1 0 1 0 1 0 1 0 0 .50 .75 .69 ✓ c3 1 0 1 0 1 1 0 0 0 0 .40 .35 .71 × c4 0 1 1 0 0 0 0 0 1 1 .40 .30 .30 × c5 0 0 0 1 1 0 0 0 1 1 .40 .30 .30 × c6 0 0 0 0 0 1 1 0 1 1 .40 .15 .01 × c7 0 0 0 0 0 0 0 1 1 1 .30 .00 .00 × c8 0 0 0 0 0 0 0 0 1 1 .20 .00 .00

αj 1.00 .67 .67 .67 .67 .67 .33 .33 .00 .00 βj .00 .20 .20 .20 .20 .20 .20 .20 1.00 1.00 δj +1.0 +.47 +.47 +.47 +.47 +.47 +.13 +.13 −1.0 −1.0

LOOC .67 .67 .53 .67 .53 .40 .63 .33 .00 .00 wjC .25 .25 .05 .25 .05 .00 .15 .00 .00 .00 LOOO .87 .80 .80 .80 .80 .67 .58 .42 .00 .00 wjO .40 .14 .15 .14 .15 .00 .00 .00 .00 .00

Hard case (Table 4). Six tests are informative and four misleading, with all four misleading tests having strong discriminative power in the wrong direction (|δj| ≥ 0.67). MV achieves only AUC = 0.60: the incorrect code c4 receives vote = 6, tying the best correct code c1. ACES-C faces a challenge: the strong misleading tests corrupt the uniform-weight ranking, causing only 2 of 6 informative tests (t4 and t5) to have LOO-AUCj > 1/2. Since ACES-C computes weights in closed form from these initial estimates, it can only assign weight to these two tests. This improves AUC to 0.77, but the incorrect c4 remains ranked third. ACES-O overcomes this limitation through iterative co-evolution: improved weights produce a better ranking, which raises the LOO-AUC of initially misidentified tests, enabling further weight refinement. At convergence, all six informative tests have LOO-AUCj > 1/2, and ACES-O concentrates weight on the three strongest (t1, t3, t2), achieving AUC = 1.00. This demonstrates the value of iterative optimization when misleading tests are prevalent.

- Table 4: Hard case: 8 × 10 pass matrix with 6 informative and 4 misleading tests (δ¯ = 0.04). Under uniform weights,

only 2/6 informative tests have LOO-AUCj > 1/2; ACES-O’s co-evolution recovers all 6. MV AUC: 0.60; ACES-C AUC: 0.77; ACES-O AUC: 1.00.

correct (αj =1) incorrect (αj <1) perf. perm. constructive misleading (δj < 0)

t1 t2 t3 t4 t5 t6 t8 t9 t7 t10 MV -C -O

✓ c1 1 1 1 1 1 0 1 0 0 0 .60 1.00 1.00 ✓ c2 1 1 1 0 0 1 0 0 0 0 .40 .00 .99 ✓ c3 1 1 0 1 1 0 0 1 0 0 .50 1.00 .70 × c4 0 0 0 1 0 1 1 1 1 1 .60 .90 .01 × c5 0 1 0 0 0 0 1 1 1 1 .50 .00 .29 × c6 0 0 0 0 1 0 1 1 1 1 .50 .10 .01 × c7 0 0 0 0 0 0 1 1 1 1 .40 .00 .00 × c8 0 0 0 0 0 0 1 1 0 1 .30 .00 .00

αj 1.00 1.00 .67 .67 .67 .33 .33 .33 .00 .00 βj .00 .20 .00 .20 .20 .20 1.00 1.00 .80 1.00 δj +1.0 +.80 +.67 +.47 +.47 +.13 −.67 −.67 −.80 −1.0

LOOC .40 .41 .46 .80 .53 .46 .42 .21 .41 .20 wjC .00 .00 .00 .90 .10 .00 .00 .00 .00 .00 LOOO 1.00 .81 .83 .73 .67 .67 .17 .00 .22 .00 wjO .40 .29 .30 .01 .00 .00 .00 .00 .00 .00

C.2 Additional Generation Models and Benchmarks

Tables 5 and 6 evaluate ACES on three additional code generation models beyond GPT-3.5-Turbo: Qwen2.5-Coder7B and 14B [Hui et al., 2024], and DeepSeek-Coder-V2-16B [Guo et al., 2024]. In addition to HumanEval and HumanEval+, we include LeetCodeDataset [Xia et al., 2025] as a harder benchmark. For each model, we generate candidates and test cases following the same protocol as Section 4.1. Table 5 reports results on all tasks; Table 6 reports results on non-trivial tasks (0 < n+ < n), excluding tasks where all or no candidates are correct.

- Table 5: Pass@k (%) with additional generation models on all tasks. Each model generates its own candidates and test cases following the protocol of Section 4.1. Subscripts denote the change from each model’s zero-shot baseline. Bold/underline: best/second-best among reranking methods within each model. Shaded : ours.

HumanEval HumanEval+ LeetCodeDataset Method Pass@1 Pass@2 Pass@5 Pass@1 Pass@2 Pass@5 Pass@1 Pass@2 Pass@5 Qwen2.5-Coder-7B Zero-shot 81.69 87.21 91.27 75.63 81.38 85.62 14.64 18.82 24.50

Majority Voting 89.02+7.3 90.85+3.6 90.85-0.4 78.66+3.0 82.32+0.9 82.93-2.7 13.16-1.5 14.91-3.9 17.98-6.5 ACES-C 89.02+7.3 90.24+3.0 90.24-1.0 79.88+4.3 82.32+0.9 82.93-2.7 16.23+1.6 17.54-1.3 18.86-5.6 ACES-O 90.24+8.6 91.46+4.3 91.46+0.2 81.10+5.5 82.93+1.6 83.54-2.1 14.04-0.6 16.23-2.6 18.86-5.6

Qwen2.5-Coder-14B Zero-shot 87.74 91.45 94.50 79.40 82.97 85.96 20.54 24.94 30.48

Majority Voting 91.46+3.7 92.68+1.2 93.29-1.2 84.15+4.8 84.76+1.8 87.20+1.2 20.61+0.1 24.12-0.8 29.39-1.1 ACES-C 92.68+4.9 92.68+1.2 93.90-0.6 84.15+4.8 84.15+1.2 86.59+0.6 25.44+4.9 26.75+1.8 30.70+0.2 ACES-O 92.68+4.9 93.29+1.8 93.29-1.2 84.76+5.4 84.76+1.8 86.59+0.6 22.37+1.8 25.00+0.1 28.95-1.5

DeepSeek-Coder-V2-16B Zero-shot 77.74 82.26 86.50 71.05 75.30 79.03 20.68 24.25 28.14

Majority Voting 82.93+5.2 84.15+1.9 85.98-0.5 74.39+3.3 75.61+0.3 78.05-1.0 16.74-3.9 18.50-5.8 22.91-5.2 ACES-C 85.37+7.6 85.98+3.7 86.59+0.1 78.66+7.6 78.66+3.4 79.27+0.2 21.15+0.5 22.47-1.8 24.23-3.9 ACES-O 84.15+6.4 84.76+2.5 86.59+0.1 76.22+5.2 76.22+0.9 78.66-0.4 18.50-2.2 20.26-4.0 23.79-4.4

- Table 6: Pass@k (%) with additional generation models on non-trivial tasks (0 < n+ < n), excluding tasks where all or no candidates are correct. Subscripts denote the change from each model’s zero-shot baseline on the same set of tasks. Bold/underline: best/second-best among reranking methods within each model. Shaded : ours.

HumanEval HumanEval+ LeetCodeDataset Method Pass@1 Pass@2 Pass@5 Pass@1 Pass@2 Pass@5 Pass@1 Pass@2 Pass@5 Qwen2.5-Coder-7B Zero-shot 76.98 86.03 92.68 75.03 84.46 91.42 31.02 42.24 57.49

Majority Voting 89.00+12.0 92.00+6.0 92.00-0.7 80.00+5.0 86.00+1.5 87.00-4.4 27.06-4.0 31.76-10.5 40.00-17.5 ACES-C 89.00+12.0 91.00+5.0 91.00-1.7 82.00+7.0 86.00+1.5 87.00-4.4 35.29+4.3 38.82-3.4 42.35-15.1 ACES-O 91.00+14.0 93.00+7.0 93.00+0.3 84.00+9.0 87.00+2.5 88.00-3.4 29.41-1.6 35.29-7.0 42.35-15.1

Qwen2.5-Coder-14B Zero-shot 68.41 80.36 90.16 67.05 77.32 85.91 32.83 42.87 55.50

Majority Voting 80.39+12.0 84.31+4.0 86.27-3.9 80.70+13.7 82.46+5.1 89.47+3.6 33.00+0.2 41.00-1.9 53.00-2.5 ACES-C 84.31+15.9 84.31+4.0 88.24-1.9 80.70+13.7 80.70+3.4 87.72+1.8 44.00+11.2 47.00+4.1 56.00+0.5 ACES-O 84.31+15.9 86.27+5.9 86.27-3.9 82.46+15.4 82.46+5.1 87.72+1.8 37.00+4.2 43.00+0.1 52.00-3.5

DeepSeek-Coder-V2-16B Zero-shot 64.38 75.63 86.15 61.77 72.63 82.19 39.14 49.74 61.25

Majority Voting 77.27+12.9 80.30+4.7 84.85-1.3 70.31+8.5 73.44+0.8 79.69-2.5 27.27-11.9 32.47-17.3 45.45-15.8 ACES-C 83.33+19.0 84.85+9.2 86.36+0.2 81.25+19.5 81.25+8.6 82.81+0.6 40.26+1.1 44.16-5.6 49.35-11.9 ACES-O 80.30+15.9 81.82+6.2 86.36+0.2 75.00+13.2 75.00+2.4 81.25-0.9 32.47-6.7 37.66-12.1 48.05-13.2

Three findings emerge from these results.

ACES improves Pass@1 over Majority Voting across all models and benchmarks. The improvement is most pronounced at k=1, where ranking quality is most critical. On HumanEval Pass@1, ACES achieves +1.2 to +2.4 points over MV across the three models; on HumanEval+ Pass@1, up to +4.3 points. The two variants exhibit complementary strengths: ACES-O leads on Qwen2.5-Coder-7B (90.24% vs. 89.02% on HumanEval), while ACES-C leads on DeepSeek-Coder-V2-16B (85.37% vs. 84.15%).

On LeetCodeDataset, ACES-C recovers from MV’s degradation below zero-shot. LeetCodeDataset is substantially harder than HumanEval, and Majority Voting can degrade below zero-shot performance (e.g., 13.16% vs. 14.64% Pass@1 for Qwen2.5-Coder-7B; 16.74% vs. 20.68% for DeepSeek-Coder-V2-16B), indicating that misleading tests dominate the uniform vote. Both ACES variants improve over MV on all three models, with ACES-C consistently leading: the ranking is ACES-C > ACES-O > MV across all models at Pass@1. At Pass@1, ACES-C surpasses zero-shot on all three models; the gain is largest on Qwen2.5-Coder-14B (25.44%, +4.8 over MV, +4.9 over zero-shot). That ACES-C outperforms ACES-O on LeetCodeDataset is consistent with the main-text observation (Section 4.2): when misleading tests are highly prevalent, the closed-form weighting is more robust than iterative optimization, which may amplify noise in this regime.

Non-trivial tasks amplify the improvements. When trivial tasks are excluded (Table 6), all methods operate on tasks where the ranking actually matters, and the ACES advantage becomes more visible. On LeetCodeDataset non-trivial tasks, ACES-C improves over MV by +8.2 points (Qwen2.5-Coder-7B), +11.0 points (Qwen2.5-Coder-14B), and +13.0 points (DeepSeek-Coder-V2-16B) at Pass@1. On HumanEval non-trivial tasks, ACES-C achieves up to +6.1 points over MV (DeepSeek-Coder-V2-16B, Pass@1).

###### C.3 Pass@k vs. k

Figure 5 extends the main results (combined with DS3 pre-filtering) to k = 1,...,20. Both ACES variants substantially outperform Majority Voting across all benchmarks and k, with the largest gains at small k where ranking quality matters most. On HumanEval and HumanEval+, ACES-C + DS3 leads at small k (e.g., +2.44% Pass@1 on HumanEval+), reflecting its closed-form weighting that concentrates on a few highly discriminative tests for sharp top-1 discrimination; the gap between the two variants narrows as k grows and they converge at larger k. On MBPP, ACES-O + DS3 slightly leads at small k, consistent with MBPP’s higher fraction of misleading tests (Section 4.3) making iterative optimization more valuable; beyond k=2 the two variants are nearly identical. The overall improvement scales with benchmark difficulty: ∼6–8% on MBPP (most misleading tests) vs. ∼3–5% on HumanEval, confirming the greater benefit of principled test weighting when test quality is more heterogeneous.

###### HumanEval

###### HumanEval+

###### MBPP

84

90

85.0

82

82.5

80

88

Pass@(%)k

78

80.0

86

76

77.5

74

84

75.0

72

72.5

MV

MV

MV

82

ACES-C + 3 ACES-O + 3

ACES-C + 3 ACES-O + 3

ACES-C + 3 ACES-O + 3

70

70.0

68

80

1 5 10 15 20

1 5 10 15 20

1 5 10 15 20

k

k

k

- Figure 5: Pass@k vs. k (k = 1,...,20) on all three benchmarks, combined with DS3 pre-filtering. ACES-C + DS3 leads at small k on HumanEval/HumanEval+; the two variants converge at larger k. On MBPP they perform comparably throughout.

###### C.4 Assumption Satisfaction Analysis

We verify Assumption 4 empirically on all three benchmarks. For each task, we remove trivial tasks (all codes correct or all incorrect) and constant-column tests (all codes pass or all fail), then compute δ¯ = (1/m) j δj over the remaining m non-constant tests. Assumption 4 formally requires δ¯ > 2 ln2/m, where the threshold is task-specific since each task has a different number of non-constant tests m. Table 7 reports the formal satisfaction rate. The figures in this section bin tasks by δ¯ using the simplified threshold δ¯ > 0 for visual clarity. Tasks are binned with width 0.1 (left-open right-closed) and further divided into three equal-width regions (Hard / Middle / Easy).

- Table 7: Assumption 4 satisfaction across benchmarks. The threshold 2 ln2/m is evaluated per task using its number of non-constant tests m.

Benchmark Non-trivial tasks δ¯ > 2 ln2/m (%) δ¯ range

HumanEval 118 83.1 [−0.42, 1.00] HumanEval+ 123 82.1 [−0.42, 1.00] MBPP 239 71.1 [−0.90, 1.00]

The assumption is satisfied for the majority of non-trivial tasks on all benchmarks. MBPP has a higher proportion of hard tasks, consistent with its greater fraction of misleading tests.

Figures 6–8 show the full per-benchmark results for Pass@k (k ∈ {1,2,5}), extending the MBPP Pass@1 analysis in Section 4.3 to all benchmarks and k values. The key findings from the main text generalize consistently across all three benchmarks and k values:

- (i) The Middle region is where ACES provides the most value. In the Easy region (high δ¯), all methods achieve near-perfect pass rates since most tests are informative and even uniform weighting works well. The performance gains of ACES concentrate in the Middle region, where informative and misleading tests coexist and principled test weighting becomes critical.
- (ii) ACES-O and ACES-C have complementary strengths across difficulty regimes. When the assumption is not well satisfied (low δ¯), ACES-O’s iterative optimization is more effective at identifying and up-weighting the few reliable tests, giving it a clear advantage over ACES-C. When the assumption is well satisfied (high δ¯), ACES-C’s closed-form solution is already near-optimal, matching or approaching ACES-O without requiring iterative optimization.
- (iii) As k increases, the gap between methods narrows, since selecting more candidates reduces the sensitivity to ranking quality at the top. This pattern is consistent across all benchmarks.

Majority Voting (MV) ACES-C ACES-O

| |MV 2/8 ACES-C 3/8<br><br>MV ACES-C<br><br>Hard Midd|49/67 52/67<br><br>MV 43/43 ACES-C 43/43<br><br>le Easy| |
|---|---|---|---|
| |ACES-O 2/8 ACES-O|55/67 ACES-O 43/43| |
| | | | |

25

Numberoftasks

20

Passrate(%)

100

15

75

10

50

25

5

0

0

-0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

Majority Voting (MV) ACES-C ACES-O

| |MV 3/8 ACES-C 3/8<br><br>MV ACES-C<br><br>Hard Midd|52/67 52/67<br><br>MV 43/43 ACES-C 43/43<br><br>le Easy| |
|---|---|---|---|
| |ACES-O 2/8 ACES-O|58/67 ACES-O 43/43| |
| | | | |

25

Numberoftasks

20

Passrate(%)

100

15

75

10

50

25

5

0

0

-0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

Majority Voting (MV) ACES-C ACES-O

| |MV 3/8 ACES-C 3/8<br><br>MV ACES-C<br><br>Hard Midd|53/67 52/67<br><br>MV 43/43 ACES-C 43/43<br><br>le Easy| |
|---|---|---|---|
| |ACES-O 3/8 ACES-O|58/67 ACES-O 43/43| |
| | | | |

25

Numberoftasks

20

Passrate(%)

100

15

75

10

50

25

5

0

0

-0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

(a) Pass@1

(b) Pass@2

(c) Pass@5

- Figure 6: Assumption 4 analysis on HumanEval (tasks binned by δ¯). ACES-O leads in the Middle region across all k; the gap narrows as k increases.

-0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

0

5

10

15

20

25

30

Numberoftasks

Majority Voting (MV) ACES-C ACES-O

0

25

50

75

100

Passrate(%)

MV 0/8 ACES-C 1/8 ACES-O 0/8

MV 48/76 ACES-C 50/76 ACES-O 56/76

MV 39/39 ACES-C 39/39 ACES-O 39/39

Hard Middle Easy

(a) Pass@1

-0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

0

5

10

15

20

25

30

Numberoftasks

Majority Voting (MV) ACES-C ACES-O

0

25

50

75

100

Passrate(%)

MV 1/8 ACES-C 1/8 ACES-O 0/8

MV 53/76 ACES-C 51/76 ACES-O 58/76

MV 39/39 ACES-C 39/39 ACES-O 39/39

Hard Middle Easy

(b) Pass@2

-0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

0

5

10

15

20

25

30

Numberoftasks

Majority Voting (MV) ACES-C ACES-O

0

25

50

75

100

Passrate(%)

MV 1/8 ACES-C 1/8 ACES-O 2/8

MV 59/76 ACES-C 55/76 ACES-O 63/76

MV 39/39 ACES-C 39/39 ACES-O 39/39

Hard Middle Easy

(c) Pass@5

- Figure 7: Assumption 4 analysis on HumanEval+ (tasks binned by δ¯). ACES-O leads in the Middle region across all k; the gap narrows as k increases.

| |MV 0/14 ACES-C 0/14<br><br>MV ACES-C<br><br>Hard Midd|35/89 46/89<br><br>MV 134/136 ACES-C 134/136<br><br>le Easy| |
|---|---|---|---|
| |ACES-O 1/14 ACES-O|52/89 ACES-O 132/136| |
| | | | |
| | | | |
| | | | |
| | | | |

-1.0-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.10.00.10.20.30.40.50.60.70.80.91.0

0

10

20

30

40

50

60

70

80

Numberoftasks

Majority Voting (MV) ACES-C ACES-O

0

25

50

75

100

Passrate(%)

(a) Pass@1

| |MV 0/14 ACES-C 0/14<br><br>MV ACES-C<br><br>Hard Midd|43/89 47/89<br><br>MV 134/136 ACES-C 134/136<br><br>le Easy| |
|---|---|---|---|
| |ACES-O 1/14 ACES-O|55/89 ACES-O 134/136| |
| | | | |
| | | | |
| | | | |
| | | | |

-1.0-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.10.00.10.20.30.40.50.60.70.80.91.0

0

10

20

30

40

50

60

70

80

Numberoftasks

Majority Voting (MV) ACES-C ACES-O

0

25

50

75

100

Passrate(%)

(b) Pass@2

| |MV 1/14 ACES-C 0/14<br><br>MV ACES-C<br><br>Hard Midd|51/89 51/89<br><br>MV 135/136 ACES-C 134/136<br><br>le Easy| |
|---|---|---|---|
| |ACES-O 1/14 ACES-O|55/89 ACES-O 135/136| |
| | | | |
| | | | |
| | | | |
| | | | |

-1.0-0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2-0.10.00.10.20.30.40.50.60.70.80.91.0

0

10

20

30

40

50

60

70

80

Numberoftasks

Majority Voting (MV) ACES-C ACES-O

0

25

50

75

100

Passrate(%)

(c) Pass@5

- Figure 8: Assumption 4 analysis on MBPP (tasks binned by δ¯). ACES-O leads in the Middle region across all k; the gap narrows as k increases.

###### C.5 Selection vs. Weighting

ACES-C combines two effects: test selection (filtering out misleading tests via the max(0,·) threshold) and non-uniform weighting (assigning higher weight to more discriminative tests). To disentangle their contributions, we first establish a theoretical guarantee for filtering alone, then empirically quantify the additional benefit of weighting.

Theoretical motivation. By Theorem 3, E[LOO-AUCj(w)] − 1/2 = cj(w) · δj, where cj(wunif) > 0 for all j under Assumption 4 (Proposition 5). Since the proportionality constant is positive, the sign is preserved:

sign E[LOO-AUCj(wunif)] − 21 = sign(δj).

That is, tests with LOO-AUCj > 1/2 are informative (δj > 0) in expectation, and those with LOO-AUCj < 1/2 are misleading (δj < 0).

If one could perfectly identify and remove all misleading tests (δj ≤ 0), assigning uniform weights to the remaining m′ informative tests would provably improve the signal-to-noise ratio. Let R(wunif) = mδ¯2 denote the SNR of Majority

Voting (Section 2.2). Uniform weighting over only the m′ tests with δj > 0 gives:

R(wfilter) R(wunif)

2

δj

= j:δj>0

m j=1 δj

≥ 1

m m′ ≥ 1

·

≥ 1,

since removing non-positive δj terms can only increase the sum, and fewer tests reduce j wj2. By Theorem 2, a higher R(w) yields a tighter Pass@k bound.

The sign identity above provides exactly such a criterion from the pass matrix alone: threshold at LOO-AUCj = 1/2. We define ACES-C Filter as:

wjfilter =

1/m′ if LOO-AUCj(wunif) > 1/2, 0 otherwise,

m′ = |{j : LOO-AUCj(wunif) > 1/2}|.

Experimental design. We compare three methods: (1) Majority Voting (uniform weights on all tests), (2) ACES-C Filter (discard tests with LOO-AUCj ≤ 1/2, uniform weights on the rest), and (3) ACES-C (LOO-AUC excess weighting, Eq. (8)).

- Table 8: Selection vs. weighting (Pass@1). Subscripts denote the change from GPT-3.5-Turbo direct inference. ACES-C Filter isolates the contribution of test selection; ACES-C adds non-uniform weighting on top.

Method HumanEval HumanEval+ MBPP GPT-3.5-Turbo 68.38 58.75 66.80

Majority Voting 80.49+12.1 69.51+10.8 68.62+1.8 ACES-C Filter 81.10+12.7 69.51+10.8 69.32+2.5 ACES-C 82.93+14.6 71.34+12.6 71.19+4.4

Results. Consistent with the theoretical guarantee, ACES-C Filter matches or improves over Majority Voting on all three benchmarks (+0.61 on HumanEval, +0.00 on HumanEval+, +0.70 on MBPP). However, the gains are modest because most misleading tests have |δj| close to zero (Figure 4), so removing them changes R(w) only slightly. The additional gain from non-uniform weighting (ACES-C vs. ACES-C Filter) is substantially larger (+1.83 on HumanEval, +1.83 on HumanEval+, +1.87 on MBPP), consistently accounting for over 70% of the total improvement. This demonstrates that the primary value of LOO-AUC lies not in binary test filtering, but in the continuous weight magnitudes that amplify informative tests proportionally to their discriminative power.

###### C.6 Sensitivity to Number of Tests

- Figure 9 shows Pass@1 as a function of the number of available tests m′, where tests are randomly subsampled from the full test suite (averaged over 5 trials). All methods use the same configuration as in the main experiments (Section 4.1). All three methods improve consistently with m′, but their scaling behaviors differ substantially.

MV plateaus around m′ = 50–100. Since uniform weighting treats informative and misleading tests equally, adding more tests of mixed quality provides diminishing marginal benefit once the pool is large enough. Both ACES variants continue improving beyond this plateau: more tests provide better LOO-AUC estimates, enabling finer-grained weight differentiation between informative and misleading tests. At m′ = 100, ACES-C already outperforms MV at full m′ = 500 on all benchmarks, showing that intelligent weighting is more valuable than simply collecting more tests.

The gap between ACES-O and ACES-C generally widens with m′: at small m′ the two variants perform similarly, but with more tests ACES-O’s optimization can exploit the richer signal to further refine the weights. Practically, even a modest test budget (m′ ≈ 100) suffices for the LOO-AUC weighting to capture most of the available signal.

HumanEval

HumanEval+

MBPP

84

74

- 68

- 69

- 70

- 71

- 72

82

72

80

Pass@1(%)

Pass@1(%)

Pass@1(%)

70

78

68

76

66

MV

MV

MV

74

ACES-C ACES-O

ACES-C ACES-O

ACES-C ACES-O

64

1020 50 100 200 300 400 500

1020 50 100 200 300 400 500

1020 50 100 200 300 400 500

m (number of tests)

m (number of tests)

m (number of tests)

- Figure 9: Sensitivity to the number of tests m′. Tests are randomly subsampled; results averaged over 5 trials (shaded: ± std). MV plateaus around m′=50–100; both ACES variants continue improving and surpass MV at m′=500 with only m′=100 tests.

C.7 Sensitivity to Number of Candidate Codes

- Figure 10 shows Pass@1 as a function of the number of candidate codes n′, where codes are stratified-subsampled to preserve the correct/incorrect ratio π (averaged over 10 trials). All m tests are retained; all methods use the same configuration as in the main experiments.

All three methods are stable with respect to n′ (standard deviations ≤ 1.6% throughout), but their scaling behaviors differ.

MV is approximately flat across all n′, and on MBPP it decreases from 70.8% (n′ = 20) to 68.6% (n′ = 200): with more candidates (most of which are incorrect), there are more incorrect codes that may outscore the best correct one under equal weighting. ACES-C maintains performance close to its full-data result even at n′ = 20. This is because ACES-C computes weights in closed form from the pass matrix; even a small candidate pool suffices to produce meaningful LOO-AUC estimates. ACES-O benefits the most from additional candidates. At small n′, the optimization has too few pairwise comparisons between passing and failing codes to learn reliable weights; as n′ grows, the gradient signal becomes richer and ACES-O achieves the best performance at n′ = 200 on all benchmarks.

HumanEval

HumanEval+

MBPP

MV

MV

- 77

- 78

- 79

- 80

- 81

- 82

- 83

- 84

ACES-C ACES-O

ACES-C ACES-O

- 67

- 68

- 69

- 70

- 71

- 72

- 73

- 74

- 69

- 70

- 71

- 72

Pass@1(%)

Pass@1(%)

Pass@1(%)

MV

ACES-C ACES-O

20 50 100 150 200

20 50 100 150 200

20 50 100 150 200

Number of codes n

Number of codes n

Number of codes n

- Figure 10: Sensitivity to the number of candidate codes n′. Codes are stratified-subsampled to preserve the correct/incorrect ratio; results averaged over 10 trials (shaded: ± std). MV is flat or declining; ACES-C is robust even at small n′; ACES-O improves steadily and leads at n′=200.

Summary of data sensitivity (Appendices C.6–C.7). The two sensitivity analyses reveal complementary strengths of the three methods. MV plateaus early with m′ and can even degrade with larger n′, since uniform weighting cannot exploit additional tests or candidates. ACES-C is robust to both m′ and n′, requiring neither many tests nor many candidates to achieve strong performance; it is the safest choice under limited data budgets. ACES-O achieves the best performance when both m′ and n′ are sufficiently large, as the optimization benefits from richer pairwise signal; in the standard setting (n=200, m ≈ 500), it consistently leads on all benchmarks.

###### C.8 Effect of Pre-Filtering Cutoff

- Figure 11 shows the effect of the pre-filtering cutoff K on both ACES variants. For each K, we select the top-K candidates by majority vote, run the reranking method on the resulting K × m submatrix, and score all n = 200 codes with the learned weights.

The two variants exhibit complementary trends, each excelling in a different regime. ACES-C generally improves as K increases, reaching its best performance at K = 200 (no pre-filtering) on all benchmarks. This is consistent with the findings in Appendix C.7: more codes provide better LOO-AUC estimates for the closed-form weighting.

ACES-O achieves its best results at small to moderate K (8–32), where the pre-filtered candidate pool concentrates on high-quality codes and provides a cleaner optimization landscape. Performance is strong across a wide range of K values and declines gradually for larger K, since including many low-quality candidates introduces noise into the pairwise comparisons.

These complementary profiles are consistent with the design of the two variants: ACES-C operates on all n candidates to maximize its closed-form estimates, while ACES-O benefits from a focused candidate pool for optimization.

8 16 32 64 128 200

Pre-filtering cutoff K

74

76

78

80

82

84

Pass@1(%)

HumanEval

MV

ACES-C ACES-O

8 16 32 64 128 200

Pre-filtering cutoff K

64

66

68

70

72

74

HumanEval+

MV

ACES-C ACES-O

8 16 32 64 128 200

Pre-filtering cutoff K

62

64

66

68

70

72

74

MBPP

MV

ACES-C ACES-O

- Figure 11: Sensitivity to the pre-filtering cutoff K. ACES-C (blue) benefits from larger candidate pools; ACES-O (green) benefits from focused pre-filtering. Dashed line: MV baseline (independent of K).

0 50 100 150 200 250 300

Iteration

0.27

0.28

0.29

0.30

0.31

0.32

0.33

SurrogateObjectiveJ(w)

HumanEval

ACES-O

0 50 100 150 200 250 300

Iteration

0.28

0.30

0.32

0.34

HumanEval+

ACES-O

0 50 100 150 200 250 300

Iteration

0.30

0.32

0.34

0.36

MBPP

ACES-O

0 50 100 150 200 250 300

Iteration

- 81

- 82

- 83

- 84

- 85

Pass@1(\%)

HumanEval

ACES-O

ACES-C (82.9\%)

MV (80.5\%)

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | |ACES ACES<br><br>|-O<br>-C (71.3\%<br>|)|
| | | | | | |MV|(69.5\%)| |
| | | | | | | | | |
| | | | | | | | | |

0 50 100 150 200 250 300

Iteration

- 69
- 70
- 71
- 72
- 73
- 74
- 75
- 76

HumanEval+

0 50 100 150 200 250 300

Iteration

- 69

- 70

- 71

- 72

MBPP

ACES-O

ACES-C (71.2\%)

MV (68.6\%)

- Figure 12: ACES-O convergence over 300 iterations, averaged across all tasks. Top: surrogate objective J(w) converges by iteration 80–100. Bottom: Pass@1 (smoothed; raw in light green) surpasses both MV (dashed gray) and ACES-C (dashed blue) and remains stable for T ∈ [50,300].

C.9 ACES-O Convergence

- Figure 12 shows the convergence behavior of ACES-O over T = 300 iterations, averaged across all tasks per benchmark.

The top row plots the surrogate objective J(w) = j wj LOO-AUCj(w) − 12 (Eq. 9), where LOO-AUCj is the logistic surrogate (Eq. 11). The bottom row plots the corresponding Pass@1 (smoothed with a 15-step moving average; raw values shown in light green).

The surrogate objective J(w) increases smoothly and monotonically on all three benchmarks, with the steepest gains in the first 50 iterations and near-complete convergence by iteration 80–100. Pass@1 reflects this stability: once the surrogate converges, the discrete Pass@1 metric remains essentially flat, fluctuating by at most 0.6 percentage points between T = 50 and T = 300 on all benchmarks. ACES-O surpasses both MV and ACES-C early in the optimization and maintains this advantage throughout, confirming that the objective J(w) is well-aligned with downstream Pass@k performance. The results are robust to the choice of T: any value in the range [50,300] yields comparable performance.

###### C.10 ACES-O Hyperparameter Sensitivity

- Figure 13 shows the effect of varying the two key ACES-O hyperparameters independently: the logistic surrogate sharpness γ (top row) and the learning rate η (bottom row). When sweeping one parameter, the other is held at its default value.

Both hyperparameters exhibit broad stable regions. For γ, the only requirement is that the logistic surrogate be sharp enough to approximate the indicator function: γ ≤ 2 is insufficient, but any γ ≥ 5 yields strong performance, with the entire range [5,50] varying by fewer than 2 percentage points on all benchmarks. For η, the stable region is even wider: η ∈ [0.005,0.5] consistently outperforms MV, again varying by at most 2 percentage points. The values used in the main experiments fall well within these stable regions.

HumanEval+

HumanEval

MBPP

- 70

- 71

- 72

- 73

- 74

- 75

ACES-O

ACES-O

ACES-O

- 80

- 81

- 82

- 83

- 84

MV

MV

MV

- 68

- 69

- 70

- 71

- 72

Pass@1(%)

Pass@1(%)

Pass@1(%)

1 2 5 8 10 15 20 30 50

1 2 5 8 10 15 20 30 50

1 2 5 8 10 15 20 30 50

γ

γ

γ

- 70

- 71

- 72

- 73

- 74

- 75

ACES-O

ACES-O

ACES-O

72.5

84.0

MV

MV

MV

72.0

83.5

71.5

83.0

Pass@1(%)

Pass@1(%)

Pass@1(%)

71.0

82.5

70.5

82.0

70.0

81.5

69.5

81.0

69.0

80.5

68.5

0.001 0.005 0.01 0.02 0.05 0.1 0.2 0.5

0.001 0.005 0.01 0.02 0.05 0.1 0.2 0.5

0.001 0.005 0.01 0.02 0.05 0.1 0.2 0.5

η

η

η

- Figure 13: ACES-O hyperparameter sensitivity. Top: Pass@1 vs. surrogate sharpness γ (other parameters at defaults). Bottom: Pass@1 vs. learning rate η. Dashed gray line: MV baseline. For γ ≥ 5 and η ∈ [0.005,0.5], ACES-O consistently outperforms MV on all benchmarks.

C.11 Computational Cost

- Figure 14 compares the reranking time per task for five methods (n = 200, m = 500), measured on a single CPU core of an Apple M3 MacBook Air (16GB RAM) and averaged over all tasks. We time only the reranking step; shared upstream costs (code generation, test generation, code execution) are excluded. To ensure a fair comparison, all execution outputs are pre-hashed before timing, so that string comparisons in output-level methods (MBR-exec) reduce to integer comparisons.

All methods finish within one second per task. MV, CodeT, ACES-C, and ACES-O operate solely on the binary pass matrix; among these, ACES-C (9ms) adds negligible overhead over MV (4ms) thanks to its closed-form computation. ACES-O requires ∼0.85s for iterative optimization, comparable to MBR-exec (∼0.38s), which incurs O(n2m) pairwise output comparisons. In practice, the reranking cost for all methods is negligible relative to upstream code execution, which dominates wall-clock time. Note that all methods compared above operate solely on execution results (the binary pass matrix or output strings). Methods that incorporate static analysis or invoke LLMs for repeated evaluation incur substantially higher costs, often orders of magnitude greater, due to the expense of additional model inference, and are therefore not included in this timing comparison.

Reranking Time per Task (n = 200, m = 500)

846 ms

Timepertask(ms,logscale)

- 100

- 101

- 102

- 103

383 ms

9.0 ms

4.1 ms 4.2 ms

MV CodeT ACES-C MBR-exec ACES-O

- Figure 14: Reranking time per task (log scale, n=200, m=500, single CPU core). Shared upstream costs are excluded; execution outputs are pre-hashed for fair comparison. All methods finish within one second.

###### C.12 Pairwise Vote Statistics

To complement the assumption analysis, we directly measure the empirical distribution of the three pairwise vote types defined in Table 1. For each non-trivial task, we enumerate all (c+,c−,tj) triples and classify each vote hj = Bc+,j − Bc−,j as informative (+1), uninformative (0), or misleading (−1). Table 9 reports the aggregate proportions.

- Table 9: Pairwise vote distribution across benchmarks. Informative votes consistently dominate misleading ones, confirming Assumption 4.

Benchmark Informative (%) Uninformative (%) Misleading (%) Inf/Mis ratio

HumanEval 22.6 72.2 5.3 4.3× HumanEval+ 20.7 74.9 4.4 4.7× MBPP 30.7 59.1 10.2 3.0×

Misleading votes are rare across all benchmarks (≤ 10.2%), confirming that the majority of pairwise test–code interactions provide either correct or neutral ranking signal. Uninformative votes dominate (59–75%): even after removing constant-column tests, most non-constant tests agree on any given (c+,c−) pair, since a test that distinguishes some pairs may still assign the same outcome to others. This highlights the value of concentrating weight on the subset of tests that actively discriminate. MBPP has roughly twice the misleading rate of HumanEval (10.2% vs. 5.3%), consistent with its lower assumption satisfaction rate (71.1% vs. 83.1%; Table 7). The informative-to-misleading ratio remains ≥ 3× on all benchmarks, providing a comfortable margin for Assumption 4.

###### C.13 Generation Prompts

HumanEval, HumanEval+, and MBPP. We use the prompts from the MPSC protocol [Huang et al., 2024], reproduced below:

Prompt for generating candidate solutions (HumanEval / HumanEval+ / MBPP)

I want you to act like a Python programmer. I will give you the declaration of a function and comments about its property. You need to implement the body of the function in the code block. Do not modify any code I provide. Do not provide any explanations.

Here is the question. ```Python {Docstring} ```

###### Prompt for generating test cases (HumanEval / HumanEval+ / MBPP)

```Python # Given a docstring, continue to write the # following code with 10 valid assertion # statements to check the correctness of the # function. Provide diverse test cases. {Docstring}

pass

# check the correctness of with 10 different # valid assertion statements in the form of # "assert {entry point}(...) == ..." assert

###### LeetCodeDataset (Appendix C.2).

###### Prompt for generating candidate solutions (LeetCodeDataset)

You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

### Question: {question}

### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters. ```python {starter_code} ``` ### Answer: (use the provided format with backticks) Prompt for generating test cases (LeetCodeDataset)

You are an expert Python programmer. You will be given a question (problem specification) and starter code, and you will generate diverse and semantically correct Python assertion statements

to test the target function.

### Question: {question}

### Starter Code: ```python {starter_code} ``` ### Target Call: {entry_point} ### Format: Output exactly 10 raw Python assertion statements. Each line must start with `assert `. Use the exact target call that is specified. Do not output markdown, code fences, comments, explanations, helper functions, or any text other than the assertions. ### Answer:

