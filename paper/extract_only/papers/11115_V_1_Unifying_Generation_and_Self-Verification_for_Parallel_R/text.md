# arXiv:2603.04304v1[cs.CL]4Mar2026

## V1: Unifying Generation and Self-Verification for Parallel Reasoners

Harman Singh∗1 Xiuyu Li∗1

Kusha Sareen2 Monishwaran Maheswaran1 Sijun Tan1 Xiaoxia Wu3 Junxiong Wang3 Alpay Ariyak3 Qingyang Wu3 Samir Khaki1 Rishabh Tiwari1 Long Lian1 Yucheng Lu3 Boyi Li1,4

Alane Suhr1 Ben Athiwaratkun3 Kurt Keutzer1 1UC Berkeley 2Mila 3Together AI 4NVIDIA Project Page | Code

Test-time scaling for complex reasoning tasks shows that leveraging inference-time compute, by methods such as independently sampling and aggregating multiple solutions, results in significantly better task outcomes. However, a critical bottleneck is verification: sampling is only effective if correct solutions can be reliably identified among candidates. While existing approaches typically evaluate candidates independently via scalar scoring, we demonstrate that models are substantially stronger at pairwise self-verification. Leveraging this insight, we introduce V1, a framework that unifies generation and verification through efficient pairwise ranking. V1 comprises two components: V1-Infer, an uncertainty-guided algorithm using a tournament-based ranking that dynamically allocates self-verification compute to candidate pairs whose relative correctness is most uncertain; and V1-PairRL, an RL framework that jointly trains a single model as both generator and pairwise self-verifier, ensuring the verifier adapts to the generator’s evolving distribution. On code generation (LiveCodeBench, CodeContests, SWE-Bench) and math reasoning (AIME, HMMT) benchmarks, V1-Infer improves Pass@1 by up to 10% over pointwise verification and outperforms recent test-time scaling methods while being significantly more efficient. Furthermore, V1-PairRL achieves 7–9% test-time scaling gains over standard RL and pointwise joint training, and improves base Pass@1 by up to 8.7% over standard RL in a code-generation setting.

Correspondence to: {harmans, xiuyu, keutzer}@berkeley.edu ∗Equal contribution.

[Figure 1]

### 1 Introduction

Large language models (LLMs) have demonstrated remarkable problem-solving abilities, largely driven by the paradigm of “System 2” thinking [6, 28, 29] of generating extended chains of thought to reflect, refine, and verify answers at inference time. Parallel reasoning, which complements this sequential “deep thinking” by sampling multiple independent chains of thought to explore diverse solution paths, has emerged as a powerful technique for test-time scaling [4, 17, 30, 39, 45]. In this setup, parallel sampling of multiple chainsof-thought is followed by an aggregation step to select the final answer. The simplest form of aggregation is to select the most common solution among the set of candidates (majority voting). While this suffices for domains like math where answers are objective and easily verifiable [45], this can not be used for more general domains which do not admit objective ground truth answers. Instead, the ability to accurately self-verify solutions can support an aggregation method that selects the correct solution from the candidate set, as long as it exists in the set, even if it isn’t the most common solution. Thus, taking full advantage of parallel reasoning fundamentally hinges on accurate self-verification: sampling N solutions is useful if the

model can reliably identify the correct one.

Our experiments identify a critical bottleneck in existing approaches that use models’ intrinsic verification capabilities to take advantage of inference-time compute: without a globally comparable scale of solution quality, existing models are not calibrated to evaluate candidate solutions independent of one another. In addition, existing work suggests that in such settings, models used as verifiers are biased towards positively evaluating their own samples, even if those samples are incorrect [21]. We find that, instead, self-verifying between candidate solutions via pairwise comparison leads to more robust and accurate outcomes. We explore two core questions. First, how can we enable LLMs to more accurately self-verify candidate solutions obtained via parallel reasoning, by leveraging pairwise candidate self-verification? Second, given that self-verification is typically applied after a model has already been trained, can we instead train models to be better at self-verification for parallel reasoning?

While pairwise ranking has been extensively studied in reward modeling for LLM alignment [3, 59], this approach remains underexplored for self-verification in a parallel reasoning setting. Additionally, while reinforcement learning is commonly used to improve the solution-generation capabilities of LLMs on verifiable domains such as code and math [6, 35], no existing methods effectively utilizes the parallel reasoning chains of LLMs at training time to jointly optimize both the generation and self-verification capabilities, resulting in distribution shifts at inference time. We demonstrate that inducing pairwise self-verification capabilities during RL training is an effective technique to improve test-time scaling performance for parallel reasoners. We leverage our observations to develop V1, a unified framework that includes a strong inference time scaling algorithm for parallel reasoning, as well as a reinforcement learning framework that induces self-verification capabilities during RL training of LLMs with verifiable rewards.

Our contributions include the following:

- 1. We show that in parallelized reasoning, independent self-verification of candidate solutions suffers from calibration collapse due to lack of comparative reference. On the other hand, self-aggregation methods like recursive self-aggregation [RSA; 44] may lead to diversity collapse where Pass@N monotonically decreases with aggregation steps. This motivates pairwise verification as a principled alternative for self-verification and an orthogonal methodology for test-time scaling without loss of diversity.
- 2. We develop V1-Infer, an uncertainty-guided pairwise verification algorithm. Rather than scoring solutions in isolation, it pairs candidates by employing a Swiss-system tournament refinement strategy that dynamically allocates verification compute to the most uncertain pairs. This approach provides a significant boost in selection accuracy, effectively improving the performance of the model closer to Pass@N of the original

sampled responses. Notably, we find that V1-Infer outperforms or matches RSA on tasks where aggregation leads to diversity collapse, while requiring significantly fewer verification calls.

- 3. We develop V1-PairRL, an RL framework that co-trains a single model as both generator and pairwise self-verifier. Unlike prior co-training approaches that rely on pointwise rewards [18, 33] or offline data [44],

V1-PairRL uses an online, co-evolving objective where generation and pairwise verification improve together. This ensures that as the generator improves, the verifier trains on in-distribution data from the model’s current capabilities, thereby leading to stronger self-verification capabilities at inference time.

We evaluate our framework on code generation (LiveCodeBench, CodeContests) and math reasoning (AIME, HMMT) benchmarks. V1-Infer improves Pass@1 by up to 10% over pointwise verification and matches or exceeds RSA. V1-PairRL achieves 7–9% test-time scaling gains over pointwise co-training standard RL, and improves base Pass@1 by up to 8.7% over the standard RL. Our results demonstrate that unified training for solution generation and self-verification capability, coupled with our pairwise self-verification technique, enables effective parallel reasoning.

### 2 Related Work

Parallel reasoning and test-time scaling. Test-time scaling typically follows two paradigms: sequential refinement [23] or parallel generation of multiple reasoning paths [4, 30, 34]. While parallel scaling allows

for the exploration of diverse solutions, it necessitates a robust mechanism to verify and select the correct output. Existing approaches often rely on access to ground-truth verification signals during inference. For instance, mathematical reasoning often leverages majority voting based on exact answer matching [39, 45], while code generation methods rely on executable test cases or execution feedback [8, 14]. In contrast, we focus on self-verification, where the model must judge the quality of its own parallel generations without access to external feedback or ground-truth oracles.

Self-verification and self-aggregation. Early work demonstrated that LLMs can verify their own Chain-ofThought (CoT) reasoning [47], though recent studies indicate that pointwise self-verification suffers from a bias toward accepting incorrect solutions [21]. While sequential self-refinement [23, 40] explores verification, it does not address the parallel reasoning setting. Alternatively, self-aggregation methods combine solutions from the same model [24, 44] but may suffer from diversity collapse, leading to the loss of correct solutions. Our work addresses these limitations by adopting pairwise self-verification, which mitigates pointwise bias and preserves solution diversity better than aggregation-based approaches.

Generative verifiers and Co-training. Generative reward models, which produce reasoning before scoring, have been shown to outperform discriminative approaches [25, 54], with pairwise ranking often proving more effective than absolute scoring [10, 43]. In this context, Zhao et al. [56] propose AggLM, which explicitly trains an aggregator via RL to synthesize correct answers for improving majority voting. However, these approaches typically rely on separate verifier or aggregator models, incurring significant overhead in compute, memory, and data curation. Recent co-training methods attempt to unify generation and verification to mitigate this cost [18, 33, 46], but they predominantly rely on pointwise verification rewards. Our V1-PairRL framework advances this by co-training a single model for pairwise self-verification, eliminating the need for external verifiers while enabling efficient online learning.

See Appendix §A for more discussion about prior work and extension to this section.

### 3 Limitations of Current Self-Verification and Aggregation Approaches

[Figure 2]

[Figure 3]

[Figure 4]

- Figure 1: (Left) Pairwise self-verification (using V1-Infer, §4) outperforms pointwise self-verification in self-verification measured on problems which have both correct and incorrect solutions in their parallel generations (Results with GPT-OSS-20B on LiveCodeBench-V6 prompts). (Middle and Right) Recursive self-aggregation on LiveCodeBench benchmarks shows declining Pass@N (diversity collapse) for both GPTOSS-20B and Qwen3-4B-Instruct. See §3 for more details.

From a test-time scaling perspective without external verifiers, parallel reasoning offers two prevalent mechanisms to generate the final solution: a) self-selection (verifying and choosing the best candidate) and

Solutions (Parallel Generation) Coverage + Swiss Reﬁnement

[Figure 5]

[Figure 6]

s2

s3 s4

s1

...

Q LLM

s5 s6

s7

0.0 1.0

Self-Veriﬁcation Score Pairwise Self-Veriﬁcaton Highest Scoring Solution

- Figure 2: Swiss Refinement Overview. Increasing pairwise verifications enables LLMs to better self-verify for selecting the best response among N self-generated solutions. See Section 4

b) self-aggregation (combining solutions to generate a better one). We analyze the limitations of current approaches in both categories to motivate our pairwise framework 1.

Pointwise self-verification suffers from calibration collapse. Standard pointwise verification assigns scalar scores to solutions in isolation. This approach is fundamentally limited by the lack of a comparative reference set. Statistically, latent utilities in choice models (e.g., Bradley–Terry) are identifiable only up to monotonic transformations, meaning absolute scores lack a globally comparable scale [2]. Consequently, pointwise scores exhibit high variance and poor cross-context calibration, often over-scoring plausible but incorrect solutions. Christiano et al. [3] noted that for learning from human preferences, relative scores are much easier to provide for humans compared to absolute scores. Pairwise judgments simplify the task to a well-posed relative comparison. As shown in Figure 1 (left), this shift to relative ranking yields significantly higher top-1 self-ranking accuracy.

Self-aggregation leads to reduction in pass@N and diversity collapse. Self-aggregation-based methods prompt the same LLM to consolidate parallelly generated solutions into one solution. While self-aggregation methods [12, 16, 24, 44] can help consolidate parallel reasoning chains and improve Pass@1, they may lead to diversity collapse. As shown in Figure 1 (Middle and Right), the Pass@N score, representing the probability that at least one correct solution exists in a set of generated N solutions, monotonically decreases as aggregation steps increase for RSA. This indicates that RSA frequently discards or degrades correct outlier solutions during refinement. Specifically, since the refined Pass@1 rarely exceeds the initial Pass@N of the raw samples, the value of self-aggregation is unclear compared with a strong self-verifier that can select the best answer and reach near Pass@N performance. Instead of relying on implicit self-verification capabilities of aggregation based method, explicit and accurate self-verification can provide orthogonal improvements to aggregator-based approaches since each aggregation step (that may induce diversity collapse) can benefit from self-verification (that maintains pass@N), which can help select promising candidate solutions to aggregate.

Takeaway: Key Limitations of Existing Self-Verification and Aggregation Approaches

Pointwise self-verification lacks calibration due to the absence of comparative references, while selfaggregation methods suffer from diversity collapse that discards correct solutions during aggregation. Pairwise self-verification offers a principled orthogonal method that is both better calibrated and diversity-preserving.

1Majority voting is less general and only applicable to scenarios with objective ground-truth answers, such as math.

### 4 Improving the Self-Verification Capability of LLMs using V1-Infer

Pointwise verification is limited in its ability to capture relative nuances and lacks calibration. These limitations, coupled with a desire to maintain solution diversity, motivate us to propose V1-Infer, a pairwise self-verification framework designed for parallel reasoning. While we can use pairwise self-verification to score responses by the model, naively pairing solutions may require quadratic number of C(N,2) selfverification attempts. Our approach, detailed in Algorithm 1, consists of a weighted aggregation mechanism and a two-phase budgeting strategy designed to maximize information gain with each new pair which allows us to efficiently scale test-time self-verification compute while achieving significant improvement in reasoning performance. A high-level overview of our approach is presented in Fig. 2

[Figure 7]

- Figure 3: Performance after self-verification using V1-Infer compared with pointwise self-verification across benchmarks and models at N=16 base generations. Results presented for GPT-OSS-20B and Qwen3-4BInstruct-2507. Results for GPT-OSS-120B and Qwen3-4B-Thinking-2507 are in Fig. 12 and show similar trends.

[Figure 8]

- Figure 4: Accuracy vs. total budget (generation + verification calls). Stars, circles, and squares denote N = 8, N = 16, and N = 32 base generations from the LLM, while the total budget = N + V where V is the number

of verification calls. V1-Infer consistently outperforms pointwise self-verification at equivalent budgets and shows monotonic performance scaling with compute. See Fig. 15 for per-benchmark results.

[Figure 9]

- Figure 5: Comparison with Recursive Self-Aggregation (RSA) [44] on LCB-v6. V1-Infer achieves higher accuracy with fewer model calls.

Uncertainty-Guided Score Aggregation. Standard pairwise voting often treats all “wins” equally, and thus fails to distinguish between a marginal preference (e.g., a 6 vs. 5 rating) and a decisive victory. Instead, we prompt the models to score their solutions in a pairwise setting instead of simply providing "correct" or "incorrect" as the output as in standard pairwise voting, which provides us with fine-grained information about the goodness of solutions. To capture this nuance of relative quality, we adopt a weighted aggregation scheme in which the magnitude of the rating difference serves as a proxy for the judge’s confidence. Given N candidate solutions S generated by an LLM, and a comparison budget B (number of self-verification LLM calls), a pairwise comparison between (si,sj) using the same LLM outputs calibrated ratings (ri,rj) ∈ [1,10] for the two solutions. We define a confidence weight

wij = max |ri − rj| 9

,τ ,

where τ > 0 is a small floor that ensures non-zero weights even for near-ties. Let vij ∈ {0,0.5,1} denote the comparison outcome for si, corresponding to a win, tie, or loss, respectively. The estimated quality score µi is then computed as the uncertainty-weighted win rate

wijvij j∈N(i) wij

, (1)

µi = j∈N(i)

where N(i) denotes the set of opponents compared against si. This formulation ensures that high-confidence judgments dominate the global ranking, while ambiguous comparisons contribute minimally to score variance.

- Phase 1: Topology Coverage. A primary failure mode in low-budget pairwise ranking is path dependence in which solutions become “orphaned” or misranked due to insufficient pairwise comparisons with other solutions. We mitigate this by enforcing a minimum degree constraint to ensure all solutions are pairwise self-verified at least a minimum number of times. We begin with random disjoint pairings to guarantee global connectivity (di ≥ 1), ensuring every solution enters the tournament. Subsequently, we iteratively target under-sampled nodes (di < dmin) to meet a minimum degree threshold. Rather than selecting random opponents, we pair these low-degree nodes with candidates having the closest current mean score µ. This “anchors” solutions against comparable peers early in the process, preventing initial noise from propagating into the refinement phase.
- Phase 2: Swiss Refinement. With the topology anchored, the remaining budget focuses on resolving rank ambiguity through an uncertainty-aware Swiss system. In each round, solutions are sorted by their current score µ, and we pair neighbors within a local window to minimize the score gap |µi −µj| among unseen pairs.

Algorithm 1 Uncertainty-Guided Pairwise Ranking

Require: Problem x, candidates S = {si}Ni=1, pairwise budget B, min-degree dmin, Swiss window size h, weight floor τ Ensure: Ranking π

- 1: Initialize: ∀ i, scores µi ← 0.5 and degrees di ← 0
- 2: history H ← ∅ ▷ (i, j) ∈ H iff (si, sj) has been compared
- 3: state T ← ({µi}Ni=1, {di}Ni=1, H)
- 4: used ← 0

- Phase 1: Topology Coverage

5: while used < B and ∃i : di < dmin do 6: P ← CoveragePairs(T , dmin) 7: O ← PairSelfVerify(x, S, P) ▷ parallel LLM judging 8: UpdateStats(T , O, τ) ▷ update µi, di; record H 9: used ← used +|P|

10: end while

- Phase 2: Swiss Refinement

- 11: while used < B and N > 2 do
- 12: π ← Rank({µi}Ni=1) ▷ descending µ
- 13: P ← SwissPairs(π, T , h) ▷ within window h; prefer unseen and near-ties
- 14: O ← PairSelfVerify(x, S, P)
- 15: UpdateStats(T , O, τ)
- 16: used ← used +|P|
- 17: end while
- 18: return Rank({µi}Ni=1)

This strategy is grounded in active learning principles: under Bradley-Terry models, comparisons between items of similar skill (near-ties) yield the highest marginal information gain. By concentrating the judge budget on these ambiguous decision boundaries, V1-Infer efficiently reduces uncertainty where it matters most, achieving high ranking accuracy even with a sparse comparison graph (K ≪ N). See Appendix §D for the detailed algorithm and full specification of the helper procedures.

#### 4.1 Experimental Settings

Models and Benchmarks. We evaluate four diverse models: GPT-OSS-20B, Qwen3-4B-Instruct, GPT-OSS120B, and Qwen3-4B-Thinking. For code generation, we use LiveCodeBench-v5 (279 problems between date range of 24.08 to 25.02), LiveCodeBench-v6 (131 problems between 25.02-25.05 following official Qwen3-2507 models2) [9], and CodeContests (165 problems) [15]. For real-world software engineering, we evaluate on SWE-bench Lite [11] (300 instances) using Gemini-2.5-Flash [5] as both the generation and verification model. For math, we use AIME’25 and HMMT’25 [1].

Evaluation Protocol and Baselines. For all methods, we first generate N candidate solutions independently from the model, then apply the verification strategy to select the final answer. We use N ∈ {8,16} for most experiments, and additionally N = 32 for pointwise verification in budget-matched comparisons. For comparison with RSA [44] we keep the population size as N = 16 with k = 4 as the aggregation size (number of solutions to aggregate at a time), with T = 10 RSA steps (number of recursive steps that the RSA algorithm is run for). See Venkatraman et al. [44] for a detailed explanation of these parameters. Details about the parameters and settings for inference sampling and swiss-refinement algorithm are provided in Appendix C.

Pointwise Baseline. For fair comparison, pointwise self-verification uses the same 1–10 grading system as our pairwise method. The model evaluates each solution in isolation by assigning a score between 1 and 10, and the solution with the highest score is selected. This ensures that any performance differences stem from the pairwise comparison structure rather than the scoring mechanism. The exact prompts for both pointwise and pairwise verification (for code and math) are provided in Appendix F.

Verification Budget. V1-Infer allows flexible control over the verification budget. We report results for budget multipliers of 1×, 2×, and 3× the number of initially generated solutions (N). For example, with N = 16 and budget 2×, we perform 32 pairwise comparisons.

2https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

#### 4.2 Results

Our goal is to measure the efficacy of V1-Infer compared to standard LLM-as-a-judge (pointwise) verification for parallel reasoners, and analyze how it compares with pointwise and aggregation-based test-time scaling methods in a budget-matched setting. Figures 3, 4, and 5 provide an overview of our results, comparing pointwise verification with pairwise verification (Figure 3), budget-matched evaluation (Figure 4), and budgeted comparison with Recursive Self-Aggregation (Figure 5).

V1-Infer consistently outperforms pointwise self-verification. On CodeContests, GPT-OSS-20B improves from 66.06% to 73.33% (+7.3%), while Qwen3-4B-Instruct improves from 39.4% to 46.1% (+6.7%). On LiveCodeBench-v5, GPT-OSS-20B gains +8.6% and Qwen3-4B-Instruct gains +4.3%. On HMMT, GPTOSS-20B gains +10.0% and Qwen3-4B-Instruct gains +6.7%. See Figure 3 for more results and Appendix Figure 12 for results on Qwen-4b-Thinking-2507 and GPT-OSS-120B. These results demonstrate that pairwise comparisons provide more informative signals than independent pointwise scores. We further observe that as verification budget increases, pairwise verification performance improves or stays consistent across most models and benchmarks (see Appendix Figure 12). While the results above are for the same number of base solution generations (N=16), we find similar trends while comparing methods in a compute-matched setting

- as well, Figure 4, e.g., with a compute budget of 64 model calls, with Qwen3-4B-Instruct-2507, pointwise verification performs approx. 33% while pairwise verification reaches 45% accuracy on average on code gen. benchmarks.

V1-Infer enables improved test-time scaling. We compare against Recursive Self-Aggregation (RSA) [44], a state-of-the-art test-time scaling method that iteratively refines solutions through an evolutionary selfaggregation process. As shown in Figure 5, V1-Infer achieves higher accuracy with significantly fewer LLM calls. On LiveCodeBench-v6 with N=16, our method reaches 76% Pass@1 with only 48 verification calls, higher compared to the maximum accuracy attained by RSA. This efficiency stems from our uncertaintyguided Swiss refinement, which concentrates comparisons on informative pairs rather than exhaustively aggregating or verifying all solution pairs. See Figures. 13, 14, 15 for extended results.

Generalization to real-world software engineering tasks. To evaluate whether pairwise self-verification extends beyond competitive programming and math to open-ended software engineering, we apply V1-Infer to SWEbench Lite [11], a benchmark of 300 real GitHub issues from popular Python repositories. For each instance, we use Gemini2.5-Flash to generate N=8 candidate patches via an agentic coding pipeline (mini-sweagent [51]), then apply self-verification to select the final patch. Crucially, the verification step receives only the issue description and the candidate patch diffs. No repository context, execution feedback, or agent trajectory is provided, making this a pure selfverification setting. As shown in Figure 6, pairwise verification achieves a 33.3% resolve rate compared to 28.3% for pointwise and 26.3% for vanilla (first-candidate) selection, an absolute gain of +5.0% over pointwise and +7.0% over vanilla. This result demonstrates that head-to-head patch comparison enables the verifier to identify subtle correctness differences, such as proper root-cause fixes versus surfacelevel changes, that are difficult to assess in isolation. See Appendix G for detailed examples showing how pairwise and pointwise verification select different patches across diverse bug categories.

[Figure 10]

Figure 6: Pairwise comparison with vanilla solution generation and pointwise self-verification on SWE-bench Lite (300 instances, N=8 candidates, Gemini 2.5 Flash, mini-sweagent [51] pipeline).

#### 4.3 Analysis and Ablations

Performance gains across different difficulty levels. Figure 7 shows how V1-Infer improves accuracy across different problem difficulty levels. On easy problems, Pass@1 is already near-optimal (99.3%). However, on hard problems where Pass@1 is only 40.2%, pairwise verification with budget 3x achieves 63.9%, a gain of +23.7%. Medium difficulty problems show an improvement of +15.4%. This pattern demonstrates that V1-Infer is most valuable precisely where it is needed most: on challenging problems where the model generates a diverse set of candidate solutions and accurate selection among them is critical for bridging the gap between Pass@1 and Pass@N.

[Figure 11]

V1-Infer outperforms random pairwise verification. To validate the effectiveness of our uncertainty-guided refinement algorithm, we compare against a baseline that randomly selects pairs for pairwise comparison. On LCB-v6 with GPT-OSS-20B

Figure 7: Accuracy improvement with increasing verification budget across problem difficulty levels (GPT-OSS-20B on LCB-v6, N=16). V1-Infer provides the largest gains on hard problems (+23.7%).

- at budget 3×, V1-Infer achieves 76.3% accuracy compared to 72.5% for random pairing, a gain of +3.8%. This demonstrates

that the strategic pair selection in V1-Infer, which prioritizes comparisons between solutions with similar quality scores, yields more informative judgments than random sampling.

Complementing aggregation with self-verification for better test-time scaling. Self-verification and aggregation-based approaches such as RSA [44] offer complementary benefits for test-time scaling. RSA is verifier-free, relying on implicit verification through iterative self-aggregation, while evolutionary approaches like AlphaEvolve [27] depend on external evaluators to guide search. By combining aggregation with pairwise self-verification, the model’s own verification capability can serve as an explicit fitness signal within the evolutionary loop. We explore this by running RSA with population size N=16 and aggregation size k=4: every 2 RSA loops, we apply selfverification to the 16 candidates, retain the top 8 by verification score, construct 16 new aggregation sets of 4 from these 8, and continue the RSA loop to maintain the population size. As shown in Figure 8, pairwise self-verification reaches 90% accuracy early in the RSA loop and converges to 93.3% with lower cumulative latency than vanilla RSA requires to reach comparable accuracy. In contrast, pointwise self-verification converges more slowly, lagging behind both pairwise and vanilla RSA at equivalent latency. This demonstrates that pairwise self-verification provides a reliable fitness signal for guiding evolutionary test-time scaling, enabling improved convergence with fewer aggregation steps.

[Figure 12]

Figure 8: Combining self-verification with RSA on AIME 2025.

Qualitative analysis: why pairwise outperforms pointwise on code. We examine representative LCB-v6 problems where pairwise and pointwise verification select different solutions (Qwen3-4B-Instruct, N=16; full examples in Appendix H). A recurring failure mode of pointwise verification is score saturation: the verifier assigns high scores (e.g., 10/10) to most candidates, losing its ability to discriminate. These patterns suggest that pairwise comparison provides a natural calibration mechanism: instead of assigning an absolute quality score, the verifier only needs to determine which of two solutions is better, a judgment that is more robust than absolute rating. We illustrate two representative cases below.

- Example 1: Group Element Assignment (4/16 correct). Pairwise ✓, Pointwise ✗. Assign elements to groups by divisibility (smallest index wins). Pointwise assigns 12 of 16 candidates the maximum score 10/10, selecting an O(|groups|×|elements|) brute-force that passes examples but exceeds time limits on large inputs. Pairwise (µ=0.917) selects a solution using divisor enumeration (O(√g) per group) with a hash map. Head-to-head comparison surfaces the algorithmic difference that independent scoring misses.

- Example 2: Binary String Trade (1/16 correct). Pairwise ✓, Pointwise ✗. Maximize active sections after at most one trade on an augmented binary string. Only 1 of 16 solutions is correct (idx=15). Pointwise gives it 8/10 but gives an incorrect solution (idx=4) the maximum 10/10. Pairwise assigns the correct solution the highest µ score of 1.000 through head-to-head wins, successfully finding the needle in a haystack of 16 candidates. See Appendix H for full code and additional examples.

Takeaway: Improved Self-Verification and Test-Time Scaling with V1-Infer

Pairwise self-verification (V1-Infer) yields more accurate candidate ranking than pointwise scoring and enables improved performance by test-time scaling of verification compute, compared with self-aggregation based test-time scaling. Furthermore, pairwise self-verification is complementary to aggregation-based approaches: when combined with RSA, it improves convergence latency by providing a reliable fitness signal within the evolutionary loop.

### 5 V1-PairRL: Improving Self-Verification via Unified RL Training

The previous section demonstrated that pairwise self-verification significantly outperforms pointwise approaches at inference time. In the remainder of the paper we consider a second question: can we explicitly train models to become stronger self-verifiers? Current RL paradigms for reasoning focus almost exclusively on optimizing the generation of correct solutions, treating verification as either an afterthought or an external process. While recent work has explored co-training generators with verifiers [18, 33], these approaches rely on pointwise rewards and fail to leverage the parallel responses that techniques like GRPO naturally generate during training. Additionally, pointwise verification is uncalibrated, which may make optimization hard. Other methods train for aggregation awareness [44, 56] but use offline data, limiting the model’s ability to adapt to its own evolving generation distribution during RLVR training.

We propose V1-PairRL, a unified RL framework that trains a single LLM to be both a strong reasoner and an accurate pairwise self-verifier. The key insight is that generation and verification should co-evolve: as the generator improves, the distribution of responses changes, and the verifier must learn score increasingly high-quality solutions. This online, co-evolving setup ensures verification training data is always in-distribution for the model’s current capabilities. However, unified training introduces unique challenges: naive implementations suffer from reward hacking, where the generator and verifier collude to maximize reward without improving actual capabilities. We address these challenges through careful reward design and pairing strategies.

#### 5.1 Preliminaries: RL for LLMs

The goal of RL for LLMs is to maximize expected reward r of an LLM policy πθ over a distribution of prompts P(Q). This is commonly done with policy gradient methods. Specifically, Group-Relative Policy Optimization (GRPO) [6] and its variants [52] have shown significant promise and stable optimization dynamics. In each episode, we sample prompts q ∼ P(Q) and a group of G rollouts per prompt {oi} ∼ πold(.|q). We maximize JGen(θ):

  1

  (2)

G,|oi|

min ρi,tAi, clip(ρi,t, 1−ϵ, 1+ϵ)Ai

###### E

G i=1 |oi|

q,{oi}

i=1,t=1

Parallel Generation

Paired Solutions Ratings

Veriﬁcation Reward Veriﬁcation Reward

g1

rc

- s1
- s2

g3

- s1
- s2

rv

s1 s3

LLM

1,v2

- 1

rc

- 2

Q LLM

Correctness Reward

g4 g5

s4 s5

rv

LLM

4,v5

. . .

. . .

. . .

. . .

Veriﬁcation Reward

rc

rv

gM gN

sN

sM sN

sN

[Figure 13]

LLM

M,vN

[Figure 14]

N

Generative Veriﬁcation Policy, LGenVerif

Solution Generation Policy, LGen

- Figure 9: V1-PairRL: Unified RL training for co-evolving generation and pairwise verification. A single LLM is trained with two objectives: JGen optimizes solution generation using correctness rewards, while JPairVerif optimizes pairwise verification accuracy. The generator produces G solutions per problem, which are evaluated for correctness and paired for verification training. The verifier evaluates solutions in a paired setting by providing correctness scores for both solutions. Since we are in a verifiable setting and know from ground truth (such as test case execution during code generation) which of the generator solutions are correct, we can calculate correctness rewards for the verifier as well. See Section 5 for more details on how optimization proceeds and rewards are calculated in this framework.

whereAi = ri−mean(r)is the advantage computed over the group andρi,t = πθ(oi,t|q,oi,<t)/πold(oi,t|q,oi,<t) is the importance sampling ratio.

- 5.2 Co-evolving Solver-Verifier Training Our training objective combines generation and pairwise verification in a unified RL formulation:

J(θ) = JGen(θ) + λJPairVerif(θ) (3)

JGen optimizes the likelihood of correct reasoning paths (using GRPO) and JPairVerif optimizes the accuracy of pairwise ranking judgments. Both objectives use the same rollouts: during each training step, the model generates G solutions per problem, which are used both to compute generation rewards and to form solution pairs for verification training. This ensures the verifier always trains on in-distribution data from the current policy. For verification training, we do one LLM call for each input pair of solutions. Figure 9 shows an overview of V1-PairRL. While our framework is general, we instantiate experiments on RL for code-generation following DeepCoder [22].

Rewards. For the solution generator, we use a standard binary correctness reward rgen ∈ {0,1} based on passing all ground truth test cases. Reward is set to 0 if response formatting is incorrect, i.e., the model does not output the response withing thinking tags as mentioned in the prompt (§E). For the pairwise self-verification objective, the model compares two solutions (sA,sB) and outputs a confidence score between 1 and 10, which is normalized to vi ∈ [0,1] for each solution. We reward the verifier based on how well its scores align with ground truth correctness yi ∈ {0,1}:

- 1

- 2 i∈{A,B}

I(|vi − yi| ≤ 0.2) · (1 − |vi − yi|) (4)

rverif =

The indicator function I(|vi − yi| ≤ 0.2) implements a sparsity threshold: the verifier receives reward only when its score is within 0.2 of the ground truth (i.e., scoring a correct solution ≥ 0.8 or an incorrect solution ≤ 0.2). This design choice is critical for preventing reward hacking, which we discuss below.

Mitigating Reward Hacking. Unified training of generators and verifiers is prone to specific collapse modes. We address two critical forms of reward hacking:

[Figure 15]

- Figure 10: V1-PairRL training results (N=16). (left) Test-time scaling: V1-PairRL outperforms V1-PointRL and improves with increased verification budget. (middle) With V1-Infer as the test-time scaling algorithm (at 2x budget): V1-PairRL outperforms RL baseline even when both use pairwise verification. (right) Base Pass@1 accuracy without any test-time scaling: Co-training with pairwise verification improves generation quality over the RL baseline, even without test time scaling.

- 1. The Safe Bet Collapse: Without the sparsity threshold, the verifier learns to output a safe, middle-ground

score (e.g., vi = 0.5) for every solution. This minimizes the risk of being “very wrong” but yields a meaningless discriminator. Sparsity threshold forces the model to commit to confident judgments: only scores near 0 or 1 receive positive reward.

- 2. The Empty Solution Loop: If the verifier is trained on pairs of two incorrect solutions, the generator may collapse into producing empty or trivially incorrect outputs. The verifier easily identifies these as incorrect (scoring them near 0), receiving high reward. This creates a situation where the generator degrades to maximize the verifier’s ease of judgment. To prevent this, we enforce a strict pairing strategy: we only trigger verification training when we can form pairs containing at least one correct solution (Correct-Incorrect or Correct-Correct pairs).

#### 5.3 Experimental Settings

We instantiate V1 on RLVR for code generation following the DeepCoder recipe [22]. Models are trained on the DeepCoder training set, which comprises 24K verified coding problems, utilizing binary rewards determined by executing generated code against ground-truth test cases (1 if all pass, 0 otherwise). We track solution generation performance on the DeepCoder validation set.

Models and Benchmarks. We train Qwen3-4B-Instruct-2507, an instruction-tuned model, following the DeepCoder experimental protocol. We evaluate trained models on LiveCodeBench V5, V6, and CodeContests. See Appendix §C.1 for evaluation hyperparameters.

Pairwise Verification Training. For JPairVerif, we group multiple verification prompts (problem + solution pairs from the solver) rather than generating multiple rollouts for a single prompt. This strategy allows us to leverage information from all pairwise comparisons without increasing the total rollout budget. The advantage for verification rollouts reduces to a REINFORCE-style estimator with a mean baseline calculated across prompts.

Baselines. We compare our approach against two primary baselines: (1) a standard RL baseline trained solely for generation without a verification objective, and (2) V1-PointRL, a model jointly trained with pointwise verification rewards under the same setup. Training details for V1-PointRL are in App. B.

Training Setup. We adopt the DAPO [52] configuration, removing the KL penalty, using Clip High, and applying token-level loss, and follow Dr. GRPO [19] by removing standard deviation normalization. To ensure a fair comparison, we enforce a fixed compute budget of 8 total rollouts per problem. The baseline allocates all 8 rollouts to the solver, while co-evolving models (V1-PairRL and PointRL) split this budget into

###### 4 solver and 4 verifier rollouts. We verified that training the baseline for a longer duration with fewer rollouts does not improve performance, confirming that gains stem from the co-training objective. All models are

trained for 150 steps, with checkpoints selected based on the validation accuracy (pass@1). Prompts used for code-generation training are in Section E.

#### 5.4 Results

V1-PairRL Offers Superior Test-Time Scaling Capabilities. To evaluate the test-time scaling benefits of co-trained verification, we compare V1-PairRL against V1-PointRL, our pointwise verification baseline trained with the same co-evolving setup (as described in Appendix B). As shown in Figure 10(a), pairwise self-verification consistently outperforms pointwise across all benchmarks at N=16. On LiveCodeBench-v5, V1-PairRL achieves 53.9% (2x budget) compared to 47.4% for V1-PointRL (+6.5%). Similar improvements are observed on LiveCodeBench-v6 (+6.8%) and CodeContests (+7.3%). V1-PairRL exhibits positive scaling with verification budget: increasing budget yields consistent accuracy gains across all benchmarks, demonstrating that the model learns to leverage additional pairwise comparisons effectively.

V1-PairRL outperforms RL Baseline with V1-Infer as the inference algorithm. In a test-time scaling setup with the same algorithm, a key question arises: does V1-PairRL show improved performance compared to the standard RL baseline? To investigate, we apply V1-Infer at inference time to both V1-PairRL and the RL baseline, using identical 2x verification budgets. As shown in Figure 10 (middle), V1-PairRL consistently outperforms the RL baseline even when both leverage pairwise verification: +3.6% on LiveCodeBenchv5, +1.9% on LiveCodeBench-v6, and +8.9% on CodeContests. This shows that co-training with pairwise verification enhances overall performance in an equivalent test-time scaling setup.

V1-PairRL improves RL Baseline. Co-training for pairwise self-verification yields substantial improvements in generation quality. As shown in Figure 10(right), V1-PairRL achieves consistent Pass@1 improvements over the RL baseline across all three code generation benchmarks at N=16: +2.9% on LiveCodeBench-v5, +2.7% on LiveCodeBench-v6, and +8.7% on CodeContests. The gains demonstrate that jointly optimizing for generation and pairwise verification creates a beneficial learning signal that improves the model’s underlying reasoning capabilities (which can also include improving the self-verification capability within long chains of thought), not just its verification accuracy. Our results echo prior findings by Sareen et al. [33], who showed that co-training with pointwise verification improves the base model’s Pass@1 performance. We show that pairwise verification takes this one step further: V1-PairRL also outperforms V1-PointRL in generation quality, particularly on CodeContests (by about +6%), where the gap is most pronounced (see Appendix Figure 16).

#### 5.5 Ablations

Co-evolving training is critical for inducing strong pairwise self-verification capabilities. V1-PairRL trains solution generation and verification capabilities in a co-evolving setup where the model is trained for self-verification using samples generated online at the same iteration thus adapting to its own evolving generation distribution. An alternative way of jointly optimizing these capabilities is to train the model in a multi-task setup where the data for the verification task is generated offline, using the base model, before training. To isolate the effect of co-evolving training, we train a non-co-evolving baseline: the same model trained with RL for generation as well as verification. For this, we first run the Qwen3-4B-Instruct-2507 model on the DeepCoder dataset prompts to generate 8 solutions per prompt, and then use these solutions as part of the verifier prompt during RL training. We apply V1-Infer at inference time with the same 2x budget. As shown in Figure 11, co-evolving training consistently outperforms the non-co-evolving setup across all benchmarks.

[Figure 16]

Figure 11: Co-evolving vs. non-co-evolving pairwise verification. Both use V1-Infer (2x budget).

Takeaway: Co-Training Generation and Pairwise Self-Verification

Unified RL training that jointly optimizes generation and pairwise verification (V1-PairRL) produces stronger reasoning models and improves test-time scaling compared with generation-only baselines or models co-trained with pointwise verification.

### 6 Conclusion

We presented V1, a unified framework for advancing self-verification in parallel reasoning, grounded in the insight that pairwise comparison constitutes a fundamentally more robust primitive for verification than absolute scoring. We introduced V1-Infer, which employs tournament-based refinement to dynamically allocate compute toward ambiguous pairs, significantly outperforming pointwise verification and prior aggregation techniques. We further improved these results by complementing V1 with a post-training approach, V1-PairRL, which demonstrated that jointly training a single model for both generation and pairwise self-verification unlocks superior test-time scaling compared to standard RL and pointwise baselines. By unifying generation with pairwise verification, V1 provides a robust framework for both effective RL training and scalable parallel reasoning.

### Acknowledgements

We acknowledge the gracious support from the Furiosa AI, Apple, NVIDIA, Macronix, Mozilla team, Open Philanthropy / Coefficient Giving, and Amazon Research. Furthermore, we appreciate the support from Google Cloud, the Google TRC team Prof. David Patterson, along with support from Google Gemini team, and Divy Thakkar. We also acknowledge support by the Director, Office of Science, Office of Advanced Scientific Computing Research, of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231. We thank Modal for providing compute credits through the Modal for Academics program. Our conclusions do not necessarily reflect the position or the policy of our sponsors, and no official endorsement should be inferred.

### References

- [1] Mislav Balunović, Jasper Dekoninck, Ivo Petrov, Nikola Jovanović, and Martin Vechev. Matharena: Evaluating llms on uncontaminated math competitions, February 2025. URL https://matharena.ai/. 7
- [2] R Bradley and M Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952. 4
- [3] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30,

2017. 2, 4, 19

- [4] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. 1, 2, 19
- [5] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. URL https://arxiv.org/abs/2507.06261. 7

- [6] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948. 1, 2, 10, 20
- [7] Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=IkmD3fKBPQ. 19
- [8] Arnav Kumar Jain, Gonzalo Gonzalez-Pumariega, Wayne Chen, Alexander M Rush, Wenting Zhao, and Sanjiban Choudhury. Multi-turn code generation through single-step rewards. In International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=aJeLhLcsh0. 3, 19
- [9] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024. 7
- [10] Dongfu Jiang, Xiang Ren, and Bill Yuchen Lin. LLM-blender: Ensembling large language models with pairwise ranking and generative fusion. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14165–14178, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.792. URL https://aclanthology.org/2023.acl-long.792/. 3, 19
- [11] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. SWE-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id= VTF8yNQM66. 7, 8
- [12] Ammar Khairi, Daniel D’souza, Marzieh Fadaee, and Julia Kreutzer. Making, not taking, the best of n,

2025. URL https://arxiv.org/abs/2510.00931. 4

- [13] Dongjun Lee, Changho Hwang, and Kimin Lee. Learning to generate unit test via adversarial reinforcement learning, 2025. URL https://arxiv.org/abs/2508.21107. 19
- [14] Dacheng Li, Shiyi Cao, Chengkun Cao, Xiuyu Li, Shangyin Tan, Kurt Keutzer, Jiarong Xing, Joseph E. Gonzalez, and Ion Stoica. S*: Test time scaling for code generation, 2025. URL https://arxiv.org/ abs/2502.14382. 3, 19
- [15] Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d’Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. Competition-level code generation with alphacode. Science, 378(6624):1092–1097, 2022. doi: 10.1126/science.abq1158. 7
- [16] Zichong Li, Xinyu Feng, Yuheng Cai, Zixuan Zhang, Tianyi Liu, Chen Liang, Weizhu Chen, Haoyu Wang, and Tuo Zhao. Llms can generate a better answer by aggregating their own responses. arXiv preprint arXiv:2503.04104, 2025. 4
- [17] Long Lian, Sida Wang, Felix Juefei-Xu, Tsu-Jui Fu, Xiuyu Li, Adam Yala, Trevor Darrell, Alane Suhr, Yuandong Tian, and Xi Victoria Lin. Threadweaver: Adaptive threading for efficient parallel reasoning in language models, 2025. URL https://arxiv.org/abs/2512.07843. 1, 19
- [18] Xiaoyuan Liu, Tian Liang, Zhiwei He, Jiahao Xu, Wenxuan Wang, Pinjia He, Zhaopeng Tu, Haitao Mi, and Dong Yu. Trust, but verify: A self-verification approach to reinforcement learning with verifiable rewards, 2025. URL https://arxiv.org/abs/2505.13445. 2, 3, 10, 19
- [19] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective, 2025. URL https://arxiv.org/abs/2503.

###### 20783. 12

- [20] Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, and Yu Wu. Inferencetime scaling for generalist reward modeling, 2025. URL https://arxiv.org/abs/2504.02495. 19
- [21] Jack Lu, Ryan Teehan, Jinran Jin, and Mengye Ren. When does verification pay off? a closer look at llms as solution verifiers, 2025. URL https://arxiv.org/abs/2512.02304. 2, 3, 19
- [22] Michael Luo, Sijun Tan, Roy Huang, Ameen Patel, Alpay Ariyak, Qingyang Wu, Xiaoxiang Shi, Rachel Xin, Colin Cai, Maurice Weber, Ce Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepcoder: A fully open-source 14b coder at o3-mini level. Notion Blog, 2025. 11, 12
- [23] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 2023. 2, 3, 19
- [24] Lovish Madaan, Aniket Didolkar, Suchin Gururangan, John Quan, Ruan Silva, Ruslan Salakhutdinov, Manzil Zaheer, Sanjeev Arora, and Anirudh Goyal. Rethinking thinking tokens: Llms as improvement operators, 2025. URL https://arxiv.org/abs/2510.01123. 3, 4, 19
- [25] Dakota Mahan, Duy Van Phung, Rafael Rafailov, Chase Blagden, Nathan Lile, Louis Castricato, Jan-Philipp Fränken, Chelsea Finn, and Alon Albalak. Generative reward models. arXiv preprint arXiv:2410.12832, 2024. 3, 19
- [26] Sadegh Mahdavi, Branislav Kisacanin, Shubham Toshniwal, Wei Du, Ivan Moshkov, George Armstrong, Renjie Liao, Christos Thrampoulidis, and Igor Gitman. Scaling generative verifiers for natural language mathematical proof verification and selection. arXiv preprint arXiv:2511.13027, 2025. 19
- [27] Alexander Novikov, Ngân V˜u, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, and Matej Balog. Alphaevolve: A coding agent for scientific and algorithmic discovery, 2025. 9
- [28] OpenAI. Learning to reason with LLMs, 2024. URL https://openai.com/index/ learning-to-reason-with-llms/. 1
- [29] OpenAI. Learning to reason with llms, 2024. URL https://openai.com/index/ learning-to-reason-with-llms/. 1
- [30] Jiayi Pan, Xiuyu Li, Long Lian, Charlie Snell, Yifei Zhou, Adam Yala, Trevor Darrell, Kurt Keutzer, and Alane Suhr. Learning adaptive parallel reasoning with language models. arXiv preprint arXiv: 2504.15466, 2025. 1, 2, 19
- [31] Chi Ruan, Dongfu Jiang, Yubo Wang, and Wenhu Chen. Critique-coder: Enhancing coder models by critique reinforcement learning, 2025. URL https://arxiv.org/abs/2509.22824. 19
- [32] Swarnadeep Saha, Xian Li, Marjan Ghazvininejad, Jason Weston, and Tianlu Wang. Learning to plan & reason for evaluation with thinking-llm-as-a-judge, 2025. URL https://arxiv.org/abs/2501.18099. 19
- [33] Kusha Sareen, Morgane M Moss, Alessandro Sordoni, Rishabh Agarwal, and Arian Hosseini. Putting the value back in rl: Better test-time scaling by unifying llm reasoners with verifiers, 2025. URL https://arxiv.org/abs/2505.04842. 2, 3, 10, 13, 19, 20
- [34] Amrith Setlur, Nived Rajaraman, Sergey Levine, and Aviral Kumar. Scaling test-time compute without verification or rl is suboptimal, 2025. URL https://arxiv.org/abs/2502.12118. 2, 19
- [35] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300. 2

- [36] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024. 21
- [37] Wenlei Shi and Xing Jin. Heimdall: test-time scaling on the generative verification, 2025. URL https: //arxiv.org/abs/2504.10337. 19
- [38] Nishad Singhi, Hritik Bansal, Arian Hosseini, Aditya Grover, Kai-Wei Chang, Marcus Rohrbach, and Anna Rohrbach. When to solve, when to verify: Compute-optimal problem solving and generative verification for llm reasoning, 2025. URL https://arxiv.org/abs/2504.01005. 19
- [39] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters, 2024. URL https://arxiv.org/abs/2408.03314. 1, 3, 19
- [40] Kaya Stechly, Karthik Valmeekam, and Subbarao Kambhampati. On the self-verification limitations of large language models on reasoning and planning tasks. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=4O0v4s3IzY. 3, 19
- [41] Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F. Christiano. Learning to summarize from human feedback. CoRR, abs/2009.01325,

2020. URL https://arxiv.org/abs/2009.01325. 19

- [42] Sijun Tan, Michael Luo, Colin Cai, Tarun Venkat, Kyle Montgomery, Aaron Hao, Tianhao Wu, Arnav Balyan, Manan Roongta, Chenguang Wang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. rllm: A framework for post-training language agents, 2025. Notion Blog. 21
- [43] Shubham Toshniwal, Ivan Sorokin, Aleksander Ficek, Ivan Moshkov, and Igor Gitman. Genselect: A generative approach to best-of-n. arXiv preprint arXiv:2507.17797, 2025. 3, 19
- [44] Siddarth Venkatraman, Vineet Jain, Sarthak Mittal, Vedant Shah, Johan Obando-Ceron, Yoshua Bengio, Brian R. Bartoldson, Bhavya Kailkhura, Guillaume Lajoie, Glen Berseth, Nikolay Malkin, and Moksh Jain. Recursive self-aggregation unlocks deep thinking in large language models, 2025. URL https: //arxiv.org/abs/2509.26626. 2, 3, 4, 6, 7, 8, 9, 10, 19
- [45] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=1PL1NIMMrw. 1, 3, 19
- [46] Yinjie Wang, Ling Yang, Ye Tian, Ke Shen, and Mengdi Wang. Co-evolving llm coder and unit tester via reinforcement learning, 2025. URL https://arxiv.org/abs/2506.03136. 3, 19
- [47] Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao. Large language models are better reasoners with self-verification. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 2550–2575, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.167. URL https://aclanthology. org/2023.findings-emnlp.167/. 3, 19
- [48] Chenxi Whitehouse, Tianlu Wang, Ping Yu, Xian Li, Jason Weston, Ilia Kulikov, and Swarnadeep Saha. J1: Incentivizing thinking in llm-as-a-judge via reinforcement learning, 2025. URL https: //arxiv.org/abs/2505.10320. 19
- [49] Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, and Yiming Yang. Inference scaling laws: An empirical analysis of compute-optimal inference for problem-solving with language models, 2025. URL https://arxiv.org/abs/2408.00724. 19

- [50] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388. 20
- [51] John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik R Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://arxiv. org/abs/2405.15793. 8
- [52] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https://arxiv.org/abs/2503.14476. 10, 12
- [53] Kaiwen Zha, Zhengqi Gao, Maohao Shen, Zhang-Wei Hong, Duane S. Boning, and Dina Katabi. Rl tango: Reinforcing generator and verifier together for language reasoning, 2025. URL https://arxiv. org/abs/2505.15034. 19
- [54] Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal. Generative verifiers: Reward modeling as next-token prediction, 2025. URL https://arxiv.org/abs/ 2408.15240. 3, 19
- [55] Eric Zhao, Pranjal Awasthi, and Sreenivas Gollapudi. Sample, scrutinize and scale: Effective inferencetime search by scaling verification, 2025. URL https://arxiv.org/abs/2502.01839. 19
- [56] Wenting Zhao, Pranjal Aggarwal, Swarnadeep Saha, Asli Celikyilmaz, Jason Weston, and Ilia Kulikov. The majority is not always right: Rl training for solution aggregation. arXiv preprint arXiv:2509.06870,

2025. 3, 10, 19

- [57] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E Gonzalez, and Ion Stoica. Judging llm-asa-judge with mt-bench and chatbot arena. In Advances in Neural Information Processing Systems, 2023. 19
- [58] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Livia Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Sglang: Efficient execution of structured language model programs. Advances in neural information processing systems, 2024. 20
- [59] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019. 2, 19

### A Extended Related Work

Parallel reasoning and test-time scaling. Test-time scaling approaches fall into two paradigms: sequential methods such as long chain-of-thought that iteratively refine a single solution [23], and parallel methods that generate multiple reasoning paths simultaneously [4, 17, 30, 34, 38, 55]. Parallel scaling helps explore diverse solution paths simultaneously; however, it requires effective mechanisms to combine various solutions. For math, majority voting is effective due to objective answers [39, 45, 49], but this approach is domain-specific and unavailable for domains such as code generation and scientific discovery, where answers are not directly comparable to each other via exact matching. For code generation, prior work [14] has explored test-time scaling but requires executable test cases in the evaluation data or generated by larger models like GPT-4 for clustering and selection, while µCode [8] leverages external, multi-turn execution feedback, both of which are specific to code generation. We focus on self-verification for parallel reasoners, where a model judges its own generations without external verified or feedback.

Self-verification and self-aggregation Early work by Weng et al. [47] demonstrated that LLMs can verify their own chain-of-thought reasoning, improving accuracy on arithmetic and commonsense tasks. Related to this, there have been multiple studies analysing the self-refinement capabilities and limitations of LLMs [7, 23, 40], however, they focus on a sequential reasoning setting, while our focus is on explicit self-verification in parallel reasoning. Recently Lu et al. [21] show that LLMs exhibit a bias toward accepting incorrect solutions during pointwise self-verification. On the other hand, self-aggregation methods like RSA [44] and parallel-distill-refine [24] combine solutions generated by the same model, but suffer from diversity collapse (Section 4), resulting in correct solutions being discarded during inference. Such aggregation-based approaches are orthogonal to self-verification and can be combined with better self-verification for more better performance and efficient test-time scaling. Our work addresses limitations of the above prior work by studying pairwise self-verification in a parallel reasoning setup, showing it significantly outperforms pointwise self-verification while preserving solution diversity.

Generative verifiers and reward models. Reward models have been central to LLM alignment and RLHF [3, 41, 59], and have emerged as a natural mechanism for scoring and selecting LLM generations [4, 57]. Recent work demonstrates that generative reward models, which produce chain-of-thought reasoning before scoring, substantially outperform discriminative approaches [25, 32, 37, 54], with RL-trained verifiers showing improved judgment performance [20, 48]. Recognizing that pairwise comparison is often easier than absolute scoring, several works have explored pairwise reward models: PairRM [10] trains a discriminative pairwise ranker for ensembling LLM outputs, while GenSelect [43] and concurrent work by Mahdavi et al. [26] scale generative verification for mathematical reasoning. Related to this, Zhao et al. [56] propose AggLM, which trains an aggregator via RL to synthesize correct answers for improving majority voting. However, these approaches use separate verifiers or aggregator models. Our work differs in two key aspects: (1) we study self-verification, where the same parallel reasoner model judges its own outputs. This eliminates the need for additional training with curated judge data, saves memory, and reduces the computational overhead associated with external verification. (2) We co-train the model to be both a solution generator and a pairwise self-verifier in a unified framework.

Co-training for generation and verification. Training separate verifiers incurs significant overhead in compute, memory, and data curation. Recent work has explored co-training a single model for both generation and verification. Sareen et al. [33] train models to self-verify online during RL, ensuring the verifier sees in-distribution generations and uses this capability to perform better majority voting at test-time for math reasoning. Liu et al. [18] extends this to unified CoT verifiers, while Zha et al. [53] co-trains a separate process reward model that provides intermediate rewards. Offline approaches train verifiers or aggregators on data sampled from the base model [31, 44]. For code generation specifically, several works co-train generators with unit-test producers [13, 46]. However, existing co-training methods rely on pointwise verification rewards. Our V1-PairRL framework instead co-trains for pairwise self-verification in an online, co-evolving setup,

which we show yields superior test-time scaling performance.

### B V1-PointRL Training Details

To provide a fair comparison for our pairwise verification approach, we train V1-PointRL using the same co-evolving setup as V1-PairRL. The key difference lies in the verification objective: instead of comparing pairs of solutions, V1-PointRL assigns an independent score to each solution.

Verification Reward. For pointwise verification, the model evaluates a single solution s and outputs a confidence score between 1-10 which is normalized to v ∈ [0,1]. The reward is computed as:

rverif = I(|v − y| ≤ 0.2) · (1 − |v − y|) (5)

where y ∈ {0,1} is the ground truth correctness. The sparsity threshold prevents the “safe bet” collapse mode where the model learns to always output v = 0.5 (in practice, not applying the sparsity threshold leads to collapse in solution generation performance and is necessary for stable training).

Training Configuration. All hyperparameters (learning rate, batch size, rollout allocation, etc.) are identical to V1-PairRL as shown in Table 1. The only difference is the verification prompt and reward computation.

Inference. At inference time, V1-PointRL generates N solutions and independently scores each one. The solution with the highest pointwise score is selected as the final answer.

### C Hyperparameters

#### C.1 Inference Sampling Parameters

Inference Framework. We use SGLang [58] for all inference experiments, which provides efficient batched inference for large language models.

Sampling Parameters. We set temperature T = 0.6 for all code generation experiments and T = 1.0 for all math reasoning experiments to encourage better exploration of the solution space. We use top-p sampling with p = 0.95 for all experiments. For experiments with V1-Infer in Section 4 where we test the inference algorithm on widely used open source LLMs like GPT-OSS-20B, GPT-OSS-120B, Qwen3-4B-Instruct-2507, Qwen3-4B-Thinking-2507, we set the max generation length to 32768 for all methods.

For trained model evaluations in Section 5 we find that training all models lead to longer chains of thoughts, as is commonly the case in RLVR [6]. This leads to frequent truncation, especially for the baseline RL model. To resolve this, we adopt the truncate-and-continue-generation method from prior work [33, 50]. When generation reaches the maximum token budget (finish_reason=length), we resume decoding by appending the partially generated assistant output to the message history and issuing a continuation request. If the truncated output does not contain a closing </thinking> tag, we append the following transition sentence before continuing: “Considering the limited time by the user, I have to directly give the required response based on the reasoning till now directly.</thinking>”. This explicitly closes the thinking block and forces the continuation to produce the final answer. If the truncated output already contains </thinking>, we directly continue generation without appending any additional text. We continue the generation for 2K tokens, making the total maximum generation length 34 K for all experiments in Section 5.

Number of Samples. We generate N ∈ {8,16} candidate solutions for most experiments. For budget-matched comparisons between pointwise and pairwise verification, we additionally evaluate pointwise verification with N = 32 samples to match the total LLM calls of pairwise verification with N = 16 and budget 3×. This budget-matched comparison is performed only for GPT-OSS-20B and Qwen3-4B-Instruct models.

Verification Budget. For V1-Infer experiments in Section 4, we evaluate verification budgets of 1×, 2×, and 3× the number of base solutions (N). Full results across all budgets are provided in the appendix figures.

Experimental Runs. All inference experiments for base models (Section 4) are run once. All trained model results (Section 5) are run 3 times with different random seeds, and we report the mean. Due to the computational cost of running multiple seeds, we limit the budget exploration for trained models to 1× and 2×, though we believe performance can further scale with additional budget for the pairwise method (V1-Infer).

Swiss Refinement Algorithm Parameters. For V1-Infer, we use the following settings: Minimum degree dmin = 2 (each solution compared at least twice in coverage phase), Swiss window size h = 8, Confidence floor set to a small value, τ = 0.1 for weighted aggregation. We do not tune parameters to any specific benchmark. We use the same parameters across benchmarks or models across all experiments.

#### C.2 Training Hyperparameters

Table 1 summarizes the key training hyperparameters. We train models using rLLM [42] and verl [36] backend.

Table 1: Training hyperparameters for V1.

##### Hyperparameter Qwen3-4B-Inst

Learning rate 1 × 10−6 Batch size 64 Rollouts per prompt (G) 4 (8 for baseline RL) Judge rollouts 4 (0 for baseline RL) Max prompt length 10240 Max response length 24576 Temperature 0.6 Top-p 0.95 Clip ratio (low) 0.2 Clip ratio (high) 0.28 λ (loss weight) 1.0 KL coefficient 0 Entropy coefficient 0 Std normalization No Loss aggregation token-mean

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

- Figure 12: Self-verification bar plots across benchmarks and models (LiveCodeBench-v5, LiveCodeBench-v6, and CodeContests). Each plot compares Pass@1, pointwise verification, and pairwise verification (budgets 1×/2×/3×) at N ∈ {8,16}. For Qwen3-4B-Thinking (bottom row), we run experiments only on LiveCodeBench at N = 8 to manage compute.

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

###### Figure 13: Comparing Pairwise Verification with RSA. Top row: LiveCodeBench-v6. Bottom row: LiveCodeBench-v5. Columns (left→right): gpt-oss-20b, Qwen3-4B-Instruct-2507. Here number of generations by the base model is 16, followed by verification (pointwise, pairwise) or aggregation (RSA).

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

###### Figure 14: Comparing Pairwise Verification with RSA. Top row: LiveCodeBench-v6. Bottom row: LiveCodeBench-v5. Columns (left→right): gpt-oss-20b, Qwen3-4B-Instruct-2507. Here number of generations by the base model is 8, followed by verification (pointwise, pairwise) or aggregation (RSA).

[Figure 36]

- Figure 15: Budget vs accuracy per benchmark. Accuracy vs. total budget (generation + verification calls) for CodeContests (top row), LiveCodeBench-v5 (middle row), and LiveCodeBench-v6 (bottom row). Left column: GPT-OSS-20B. Right column: Qwen3-4B-Inst. Stars denote N = 8 base generations, circles denote N = 16, squares denote N = 32 (pointwise only).

### D Detailed Algorithm

This section provides the full specification of the helper procedures used in Algorithm 1. The main paper presents a high-level view for clarity; here we give the exact update rules and pairing strategies required for reproducibility.

State. We maintain a state T = ({µi}Ni=1,{di}Ni=1,H), where µi is the current estimated quality score of solution si, di is the number of distinct opponents it has been compared against, and H ⊂ [N] × [N] records previously compared pairs.

CoveragePairs. Ensures minimum degree while anchoring comparisons to similar-quality peers.

- Algorithm 2 CoveragePairs(T ,dmin)

Require: State T = ({µi},{di},H), minimum degree dmin Ensure: Disjoint pair set P

- 1: P ← ∅, U ← ∅ ▷ used indices
- 2: L ← {i : di < dmin}; sort L by increasing di
- 3: for each i ∈ L do
- 4: if i ∈ U then
- 5: continue
- 6: end if
- 7: C ← {j ̸= i : j ∈/ U, (i,j) ∈/ H}
- 8: if C = ∅ then
- 9: continue
- 10: end if
- 11: j ← arg minj∈C |µi − µj|
- 12: P ← P ∪ {(i,j)}
- 13: U ← U ∪ {i,j}
- 14: end for
- 15: return P

UpdateStats. This is where the mathematical aggregation is implemented. Note. In practice, the update

- Algorithm 3 UpdateStats(T ,O,τ)

Require: State T = ({µi},{di},H), outcomes O, floor τ > 0 Ensure: Updated state T

- 1: for each (i,j,w,ri,rj) ∈ O do
- 2: vij ← I[w = A] + 0.5 · I[w = tie]
- 3: wij ← max |ri−9rj|,τ

- 4: update µi,µj using Eq. (1)
- 5: if (i,j) ∈/ H then
- 6: di ← di + 1; dj ← dj + 1
- 7: H ← H ∪ {(i,j),(j,i)}
- 8: end if
- 9: end for

of µi is implemented incrementally using sufficient statistics, but Eq. (1) fully specifies the aggregation semantics.

SwissPairs. Allocates remaining budget to the most informative near-ties.

##### Algorithm 4 SwissPairs(π,T ,h)

Require: Ranking π, state T = ({µi},{di},H), window size h Ensure: Disjoint pair set P

- 1: P ← ∅, U ← ∅
- 2: for k = 1 to N do
- 3: i ← πk
- 4: if i ∈ U then
- 5: continue
- 6: end if
- 7: C ← ∅
- 8: for t = k + 1 to min(k + h,N) do
- 9: j ← πt
- 10: if j ∈ U then
- 11: continue
- 12: end if
- 13: add (|µi − µj|,I[(i,j) ∈ H],j) to C
- 14: end for
- 15: if C ̸= ∅ then
- 16: choose j with lexicographically minimal key in C
- 17: P ← P ∪ {(i,j)}
- 18: U ← U ∪ {i,j}
- 19: end if
- 20: end for
- 21: return P

The above procedures jointly ensure (i) early stabilization of the comparison topology via minimum-degree coverage and (ii) efficient use of remaining budget through uncertainty-focused Swiss refinement, enabling accurate ranking with O(N)–scale pairwise verification.

[Figure 37]

[Figure 38]

[Figure 39]

- Figure 16: V1-PairRL vs RL baseline across benchmarks (N=16). Left: LiveCodeBench-v5. Middle: LiveCodeBench-v6. Right: CodeContests. Co-training with pairwise verification consistently improves Pass@1 accuracy across all benchmarks.

### E Generation Prompts

#### E.1 Code Generation

Code Generation Prompt

**Problem** {problem}

Think and reason step by step before coding the final solution for the problem above. Put your reasoning and any draft coding solutions between <thinking> ... </thinking> tags. After the reasoning (i.e. after the </thinking> tag), use the format provided in the problem above (code-block with backticks) to format your final code solution. Do not include any thinking within the code block.

### Answer:

Note that "Do not include any thinking within the code-block." is added to the prompt, since in the absence of it, training the model (Qwen3-4B-Instruct-2507) leads to it generating large amounts of comments within the code block, effectively leading to truncation of the response (due to reaching max length) and hence, 0 rewards and training collapse.

#### E.2 Math Solution Generation

###### Math Solution Generation Prompt

{problem}. Let’s think step by step and output the final answer within \boxed{}.

### F Verification Prompts

This section provides the exact prompts used for pointwise and pairwise self-verification in our experiments. Both verification methods use a 1–10 rating scale for fair comparison.

#### F.1 Pointwise Verification Prompts

##### F.1.1 Code Generation

Pointwise Code Verification Prompt

You are an expert code reviewer. Rate the correctness of a solution to a programming problem.

**Evaluation Guidelines:**

- - Analyze the problem’s requirements and constraints.
- - Mentally trace the solution with test cases (including edge cases) to verify correctness.
- - Give a higher score if the solution is robust and fault-tolerant.

**Problem** {problem}

**Solution** {code}

**Output Format:** First, provide your step-by-step reasoning. Then, on a new line, provide your final rating using the EXACT tags below. Add no other text after the tags. <rating>INTEGER_1_TO_10</rating>

**Rating Rules:**

- - Rate correctness on a 1-10 scale (10 = correct & robust, 5 = borderline, 1 = incorrect). Please provide your analysis now.

##### F.1.2 Math Reasoning

###### Pointwise Math Verification Prompt

You are an expert math contest grader. Rate the correctness of a submission based solely on the final answer.

**Evaluation Guidelines:**

- - Extract the submission’s final answer. Use any provided reasoning only to help you assess whether the stated final answer is trustworthy. Do not award credit for method quality or rigor.
- - Carefully analyze the problem statement and the submission to assess whether the final answer is correct. Grade only the final answer.

**Problem** {problem}

**Solution** {solution}

**Output Format:** First, provide your reasoning (what checks you performed). Then, on a new line, provide your final rating using the EXACT tag below. Add no other text after the tag. <rating>INTEGER_1_TO_10</rating>

**Rating Rules:**

- - Rate correctness on a 1-10 scale (10 = certainly correct, 8 = very likely correct, 5 = uncertain/borderline, 3 = likely incorrect, 1 = certainly incorrect). Please provide your analysis now.

#### F.2 Pairwise Verification Prompts

##### F.2.1 Code Generation

Pairwise Code Verification Prompt

You are an expert code reviewer. Compare two solutions to a programming problem and rate their correctness.

**Evaluation Guidelines:**

- - Analyze the problem’s requirements and constraints.
- - Mentally trace each solution with test cases (including edge cases) to verify correctness.
- - If both solutions appear correct, prefer the more robust and fault-tolerant one.

**Problem** {problem}

- **Solution A**

- {code_A}

**Solution B**

- {code_B}

**Output Format:** First, provide your step-by-step reasoning. Then, on separate new lines, provide your final ratings using the EXACT tags below. Add no other text after the tags.

- <rating_A>INTEGER_1_TO_10</rating_A>
- <rating_B>INTEGER_1_TO_10</rating_B>

**Rating Rules:**

- - Rate correctness on a 1-10 scale (10 = correct & robust, 5 = borderline, 1 = incorrect).
- - The higher rating wins. Equal ratings imply a tie. Please provide your analysis now.

##### F.2.2 Math Reasoning

###### Pairwise Math Verification Prompt

You are an expert math contest grader. Compare two submissions and rate correctness based solely on the final answer.

**Evaluation Guidelines:**

- - Extract each submission’s final answer. Use any provided reasoning only to help you assess whether the stated final answer is trustworthy. Do not award credit for method quality or rigor.
- - Carefully analyze the problem statement and the submissions to assess whether each final answer is correct. Grade only the final answer.

**Problem** {problem}

- **Solution A**

- {sol_A}

- **Solution B**

- {sol_B}

**Output Format:** First, provide your reasoning (what checks you performed). Then, on separate new lines, give ratings using the EXACT tags below. Add no other text after the tags.

- <rating_A>INTEGER_1_TO_10</rating_A>
- <rating_B>INTEGER_1_TO_10</rating_B>

**Rating Rules:**

- - Rate correctness on a 1-10 scale (10 = certainly correct, 8 = very likely correct, 5 = uncertain/borderline, 3 = likely incorrect, 1 = certainly incorrect).
- - Higher rating wins. Equal ratings imply a tie. Please provide your analysis now.

### G SWE-bench Verification Examples

This section presents representative examples from our SWE-bench Lite evaluation (300 instances, 8 candidate patches each, Gemini 2.5 Flash as both generator and verifier). We show cases where pairwise and pointwise verification select different patches, illustrating how head-to-head comparison can surface correct solutions that independent scoring misses, and vice versa. For each example, we show the issue description, the patches selected by each method, and their rankings.

#### G.1 Example 1: django__django-11049: Pairwise Correct, Pointwise Wrong

###### Issue: Correct expected format in invalid DurationField error message

If you enter a duration “14:00” into a duration field, it translates to “00:14:00” which is 14 minutes. The current error message for invalid DurationField says that this should be the format of durations: [DD] [HH:[MM:]]ss[.uuuuuu]. But according to the actual behaviour, it should be: [DD] [[HH:]MM:]ss[.uuuuuu], because seconds are mandatory, minutes are optional, and hours are optional if minutes are provided. This seems to be a mistake in all Django versions that support the DurationField. Also the duration fields could have a default help_text with the requested format, because the syntax is not self-explanatory.

Setup: 8 candidate patches generated, all non-empty. Pairwise ranking: [6, 3, 5, 1, 7, 0, 4, 2] → selected idx=6 ✓

- Pointwise ranking: [0, 1, 3, 5, 6, 7, 4, 2] → selected idx=0 ✗ Pairwise-Selected Patch (idx=6): Resolves the issue ✓

diff –git a/django/db/models/fields/__init__.py @@ -1587,7 +1587,7 @@ class DurationField(Field): default_error_messages = { ’invalid’: _("’%(value)s’ value has an invalid format. It must be in "

- "[DD] [HH:[MM:]]ss[.uuuuuu] format.") + "[DD] [[HH:]MM:]ss[.uuuuuu] format.") }

diff –git a/django/forms/fields.py @@ -467,6 +467,8 @@ class DurationField(Field):

+ help_text = _(’Accepted format is [DD] [[HH:]MM:]ss[.uuuuuu]’)

###### Pointwise-Selected Patch (idx=0): Does not resolve the issue ✗

diff –git a/django/forms/fields.py @@ -466,12 +466,19 @@ class DurationField(Field):

+ default_help_text = _(’Accepted duration format is [[DD] [[HH:]MM:]SS[.uuuuuu]].’) def __init__(self, *, help_text=None, **kwargs): if help_text is None: help_text = self.default_help_text super().__init__(help_text=help_text, **kwargs)

Analysis: The correct patch (idx=6) fixes both the error message format string in the model field and adds help text to the form field. The pointwise-selected patch (idx=0) only adds help text to the form field but does not fix the incorrect format string in the model’s error message. Pairwise comparison enables the verifier to directly contrast these two approaches, surfacing that fixing the error message is the core requirement.

#### G.2 Example 2: matplotlib__matplotlib-25311: Pairwise Correct, Pointwise Wrong

###### Issue: Unable to pickle figure with draggable legend

Bug summary: Unable to pickle figure with draggable legend. Same error comes for draggable annotations. Code for reproduction: fig = plt.figure() ax = fig.add_subplot(111) ax.plot(time, speed, label="speed") leg = ax.legend() leg.set_draggable(True) # pickling works after removing this line pickle.dumps(fig) # raises TypeError

Actual outcome: TypeError: cannot pickle ’FigureCanvasQTAgg’ object Expected outcome: Pickling successful

Setup: 8 candidate patches generated, 7 non-empty.

- Pairwise ranking: [6, 2, 3, 4, 5, 1, 0] → selected idx=6 ✓

- Pointwise ranking: [0, 1, 2, 3, 4, 6, 5] → selected idx=0 ✗ Pairwise-Selected Patch (idx=6): Resolves the issue ✓

diff –git a/lib/matplotlib/offsetbox.py @@ -1472,6 +1472,23 @@ class DraggableBase:

+ def __getstate__(self):

+ self.disconnect()

+ state = self.__dict__.copy()

+ state["canvas"] = None

+ state["cids"] = []

+ if "_c1" in state: del state["_c1"]

+ return state

+ def __setstate__(self, state):

+ self.__dict__.update(state)

###### Pointwise-Selected Patch (idx=0): Does not resolve the issue ✗

diff –git a/lib/matplotlib/legend.py @@ -1204,6 +1204,22 @@ class Legend(Artist):

+ def __getstate__(self):

+ draggable_state = self._draggable is not None

+ if draggable_state: self.set_draggable(False)

+ state = super().__getstate__()

+ state["_draggable_state"] = draggable_state

+ return state

+ def __setstate__(self, state):

+ draggable_state = state.pop("_draggable_state")

+ super().__setstate__(state)

+ if draggable_state: self.set_draggable(True)

Analysis: The correct patch (idx=6) fixes the problem at its root in DraggableBase, the parent class used by both draggable legends and annotations. It properly handles the unpickleable canvas reference. The pointwise-selected patch (idx=0) attempts the fix only in the Legend subclass, missing the general case for draggable annotations. The pairwise tournament allowed the verifier to compare these approaches side-by-side, recognizing that the root-class fix is more complete.

#### G.3 Example 3: matplotlib__matplotlib-23964: Pairwise Correct, Pointwise Wrong

Issue: PostScript backend crash on empty text with usetex When using the PostScript backend with usetex=True, rendering an empty string raises an error because curr_stream is uninitialized when appended to the stream list after iterating over an empty glyph sequence.

Setup: 8 candidate patches generated, all non-empty.

- Pairwise ranking: [4, 0, 1, 2, 3, 5, 6, 7] → selected idx=4 ✓

Pointwise ranking: [0, 1, 4, 2, 3, 5, 6, 7] → selected idx=0 ✗

Pairwise-Selected Patch (idx=4): Resolves the issue ✓

diff –git a/lib/matplotlib/backends/backend_ps.py @@ -666,7 +666,8 @@ ) # append the last entry

- stream.append(curr_stream)

+ if curr_stream:

+ stream.append(curr_stream)

Pointwise-Selected Patch (idx=0): Does not resolve the issue ✗

diff –git a/lib/matplotlib/backends/backend_ps.py @@ -666,6 +666,7 @@ ) # append the last entry

+ if curr_stream: stream.append(curr_stream)

Analysis: Both patches add the same guard (if curr_stream), but the pointwise-selected patch (idx=0) has an indentation error—the stream.append line is not nested under the new if block. This subtle formatting difference makes the patch syntactically different from the intended fix. Pairwise comparison helped the verifier notice this indentation discrepancy when directly contrasting the two candidates.

G.4 Example 4: django__django-14017: Pointwise Correct, Pairwise Wrong

Issue: Q(...) & Exists(...) raises a TypeError

Exists(...) & Q(...) works, but Q(...) & Exists(...) raises a TypeError. The & (and |) operators should be commutative on Q-Exists pairs, but they are not. The issue is in Q._combine() which checks isinstance(other, Q) and raises TypeError for non-Q operands like Exists.

Setup: 8 candidate patches generated, 7 non-empty.

- Pairwise ranking: [5, 4, 2, 0, 3, 6, 1] → selected idx=5 ✗ Pointwise ranking: [2, 3, 4, 5, 6, 1, 0] → selected idx=2 ✓

###### Pointwise-Selected Patch (idx=2): Resolves the issue ✓

diff –git a/django/db/models/query_utils.py @@ -40,7 +40,10 @@ class Q(tree.Node): def _combine(self, other, conn): - if not isinstance(other, Q):

+ from django.db.models.expressions import Expression

+ if isinstance(other, Expression):

+ other = Q(other)

+ elif not isinstance(other, Q): raise TypeError(other)

###### Pairwise-Selected Patch (idx=5): Does not resolve the issue ✗

diff –git a/django/db/models/expressions.py # Moves Q import to local scope + modifies __and__ in Combinable diff –git a/django/db/models/query_utils.py @@ -40,6 +41,8 @@ class Q(tree.Node): def _combine(self, other, conn):

+ if isinstance(other, expressions.Expression) and getattr(other, "conditional", False):

+ other = Q(other) if not isinstance(other, Q): raise TypeError(other) # Also modifies Q.deconstruct() for Expression children

Analysis: The correct patch (idx=2) is a minimal, focused fix: it wraps any Expression operand in Q() within _combine. The pairwise-selected patch (idx=5) attempts a more complex fix spanning two files and adding a conditional attribute check, but introduces regressions in deconstruct() handling. This example illustrates a failure mode of pairwise verification: a more complex-looking patch can appear “more thorough” in head-to-head comparison, even when the simpler patch is correct.

#### G.5 Example 5: scikit-learn__scikit-learn-13439: Both Correct, Different Indices

Issue: Pipeline should implement __len__ With the new indexing support, pipe[:len(pipe)] raises an error because Pipeline does not implement __len__. from sklearn.pipeline import Pipeline pipe = Pipeline([(’anova’, anova_filter), (’svc’, clf)]) len(pipe) # TypeError: object of type ’Pipeline’ has no len()

Setup: 8 candidate patches generated, all non-empty.

- Pairwise ranking: [7, 4, 3, 1, 2, 5, 0, 6] → selected idx=7 ✓

- Pointwise ranking: [1, 2, 3, 4, 5, 7, 0, 6] → selected idx=1 ✓ Pairwise-Selected Patch (idx=7): Resolves the issue ✓

diff –git a/sklearn/pipeline.py @@ -130,6 +130,12 @@ class Pipeline(_BaseComposition): self.memory = memory

+ def __len__(self):

+ """Returns the length of the Pipeline"""

+ return len(self.steps)

###### Pointwise-Selected Patch (idx=1): Also resolves the issue ✓

diff –git a/sklearn/pipeline.py @@ -27,6 +27,8 @@ class Pipeline(_BaseComposition):

+ def __len__(self):

+ return len(self.steps)

Analysis: Both patches correctly implement __len__ by returning len(self.steps). They differ only in placement within the class (after __init__ vs. at the top of the class) and the presence of a docstring. This example demonstrates that when multiple correct solutions exist among the candidates, both verification methods may successfully identify a correct patch, even if they select different candidates.

### H Code Generation Verification Examples (LiveCodeBench-v6)

This section presents representative examples from our LiveCodeBench-v6 evaluation (131 problems, 16 candidate solutions each, Qwen3-4B-Instruct as both generator and verifier with budget 3×). We show cases where pairwise and pointwise verification select different solutions, illustrating how head-to-head comparison can surface correct solutions that independent scoring misses, and vice versa.

#### H.1 Example 1: Binary String Trade (1/16 correct): Pairwise Correct, Pointwise Wrong

###### Problem: Maximize active sections after at most one trade

You are given a binary string s of length n, where ’1’ represents an active section and ’0’ represents an inactive section. You can perform at most one trade: (1) convert a contiguous block of ’1’s surrounded by ’0’s to all

- ’0’s, then (2) convert a contiguous block of ’0’s surrounded by ’1’s to all ’1’s. The string is augmented with
- ’1’ at both ends: t = ’1’ + s + ’1’. Return the maximum number of active sections after the optimal trade.

Setup: 16 candidate solutions, only 1 correct (idx=15). Pairwise µ ranking: [15, 4, 10, 1, 9, 14, 5, 13, ...] → selected idx=15 (µ=1.000) ✓ Pointwise ranking: [4, 13, 15, 10, 0, ...] → selected idx=4 (score=10/10) ✗ Pointwise scores: Correct idx=15 received 8/10; incorrect idx=4 received 10/10.

Pairwise-Selected Solution (idx=15): Correct ✓

class Solution: def maxActiveSectionsAfterTrade(self, s: str) -> int: t = ’1’ + s + ’1’ original_ones = s.count(’1’) # Split augmented string into blocks of 1s and 0s blocks = []

- i = 0

- while i < len(t):

j = i

- while j < len(t) and t[j] == t[i]: j += 1 blocks.append((t[i], i, j-1, j - i)) i = j # For each 1-block surrounded by 0-blocks, # compute gain from trading it and filling adjacent 0-block max_candidate = original_ones for idx in range(len(blocks)): if blocks[idx][0] == ’1’: # Check if surrounded by 0-blocks on both sides # Trade: sacrifice this 1-block, fill a neighboring 0-block

... # correct enumeration of all valid trades return max_candidate

###### Pointwise-Selected Solution (idx=4): Incorrect ✗

class Solution: def maxActiveSectionsAfterTrade(self, s: str) -> int: t = ’1’ + s + ’1’ original_ones = s.count(’1’) ones = [i for i in range(len(t)) if t[i] == ’1’] # Groups contiguous 1-segments but incorrectly # enumerates valid trades but mishandles augmented boundaries

Analysis: This is a needle-in-a-haystack case: only 1 of 16 solutions passes all test cases. The correct solution (idx=15) correctly enumerates all valid trades by iterating over blocks of the augmented string, while idx=4 mishandles boundary conditions in the augmented string. Pointwise verification gave the correct solution only 8/10 (perhaps penalizing its longer code) while giving the incorrect idx=4 a perfect 10/10. Pairwise verification, through head-to-head comparisons, assigned idx=15 the highest µ score of 1.000, successfully identifying the only correct solution among 16 candidates.

#### H.2 Example 2: Substring Character Check (14/16 correct): Pairwise Correct, Pointwise Wrong

###### Problem: Check for special substring of length k

Given a string s and an integer k, determine if there exists a substring of length exactly k that: consists of only one distinct character, has a different character immediately before it (if one exists), and has a different character immediately after it (if one exists). Return true if such a substring exists, false otherwise.

Setup: 16 candidate solutions, 14 correct, 2 incorrect (idx=0 and idx=12). Pairwise µ ranking: [5, 3, 6, 1, 13, ..., 12, 0] → selected idx=5 ✓ Pointwise ranking: [0, 1, 2, 3, 4, ...] → selected idx=0 ✗ Pointwise scores: 15 of 16 solutions scored 10/10 (including wrong idx=0); only idx=12 scored 6/10.

Pairwise-Selected Solution (idx=5): Correct ✓

# Checks boundary correctly: after_char = s[i+k] if i+k < n if after_char is not None and after_char == substring[0]: continue

###### Pointwise-Selected Solution (idx=0): Incorrect (off-by-one) ✗

# Off-by-one error: uses s[i+k-1] instead of s[i+k] after_char = s[i+k-1] if i + k < n else None

Analysis: With 14 out of 16 correct solutions, the task reduces to avoiding the 2 incorrect ones. The bug in idx=0 is a subtle off-by-one error: it checks the last character inside the substring (s[i+k-1]) rather than the first character after it (s[i+k]). Pointwise verification assigned the wrong idx=0 a score of 10/10, identical to 14 of the 15 correct solutions, making selection essentially random among the tied candidates: with idx=0 first in the tie-broken ordering, it was selected. Only the other wrong solution (idx=12) received a lower score of 6/10. Pairwise comparison detected the subtle bug by directly comparing solutions, ranking both wrong solutions (idx=0 and idx=12) at the bottom of the ranking.

#### H.3 Example 3: Group Element Assignment (4/16 correct): Pairwise Correct, Pointwise Wrong

###### Problem: Assign elements to groups by divisibility

Given an integer array groups (group sizes) and an integer array elements, assign one element to each group such that groups[i] is divisible by elements[j]. Among valid elements, choose the one with the smallest index j. If no element works, assign −1.

Setup: 16 candidates, 4 correct (idx=8,9,11,12). Pairwise µ ranking: [12, 14, 9, 11, 15, ...] → selected idx=12 (µ=0.917) ✓ Pointwise ranking: [0, 1, 2, 3, 4, ...] → selected idx=0 (score=10/10) ✗ Pointwise scores: 12 of 16 candidates scored 10/10, including 9 of the 12 incorrect solutions. Correct idx=8 scored only 3/10.

Pairwise-Selected Solution (idx=12): Correct ✓

class Solution: def assignElements(self, groups, elements): value_to_min_index = {} for j, val in enumerate(elements): if val not in value_to_min_index: value_to_min_index[val] = j assigned = [-1] * len(groups) for i, g in enumerate(groups): # Enumerate divisors of g in O(sqrt(g)) d = 1 while d * d <= g: if g % d == 0: # Check both d and g//d in the map

... d += 1 return assigned

###### Pointwise-Selected Solution (idx=0): Incorrect (TLE) ✗

class Solution: def assignElements(self, groups, elements): assigned = [-1] * len(groups)

- for i in range(len(groups)):
- for j in range(len(elements)): if groups[i] % elements[j] == 0: assigned[i] = j break return assigned

Analysis: The correct solution uses divisor enumeration (O(√g) per group) with a hash map for element lookup, while the incorrect solution uses a brute-force nested loop (O(|groups| × |elements|)) that exceeds time limits on large inputs. Both produce identical outputs on small examples, which is why pointwise verification (which reasons about correctness from code inspection) assigns 10/10 to both. This is a classic score saturation failure: pointwise scoring cannot distinguish algorithmic efficiency, while pairwise comparison, by directly contrasting the two approaches, can recognize the more efficient solution.

#### H.4 Example 4: Almost Missing Integer (1/16 correct): Pointwise Correct, Pairwise Wrong

###### Problem: Find the largest almost missing integer

Given an integer array nums and an integer k, an integer x is almost missing if x appears in exactly one subarray of size k. Return the largest such integer, or −1 if none exists.

Setup: 16 candidates, only 1 correct (idx=0). Pairwise µ ranking: [12, 8, 6, 7, 3, ..., 0] → selected idx=12 ✗ Pointwise ranking: [0, 1, 2, 3, ...] → selected idx=0 ✓ Pointwise scores: All 16 candidates scored 10/10. Pointwise selected idx=0 by tie-breaking (lowest index).

Pointwise-Selected Solution (idx=0): Correct ✓

class Solution: def largestInteger(self, nums, k): max_val = max(nums) for val in range(max_val, -1, -1): count = 0 for start in range(len(nums) - k + 1): if val in nums[start:start + k]: count += 1 if count == 1: return val return -1

###### Pairwise-Selected Solution (idx=12): Incorrect ✗

class Solution: def largestInteger(self, nums, k): count = {} for i in range(len(nums) - k + 1): for num in nums[i:i + k]: count[num] = count.get(num, 0) + 1 # Bug: counts total occurrences across subarrays, # not the number of distinct subarrays containing the element almost = [n for n, f in count.items() if f == 1] return max(almost) if almost else -1

Analysis: The pairwise-selected solution (idx=12) has a subtle semantic bug: it counts the total number of times each element appears across all subarrays, rather than the number of distinct subarrays containing that element. An element appearing twice in one subarray would be counted as 2, not 1. Pointwise verification assigned all 16 solutions the maximum score of 10/10, indicating complete inability to distinguish them. However, pointwise “won” here by coincidence: when all scores are tied, the tie-breaking rule selects idx=0, which happens to be the sole correct solution. This illustrates a failure mode of pairwise verification: when all solutions appear superficially similar, head-to-head comparisons can amplify small stylistic differences into ranking signals that do not correlate with correctness.

#### H.5 Example 5: Three-Subarray Distinct Count (2/16 correct): Pairwise Correct, Pointwise Wrong

###### Problem: Maximize sum of distinct counts in three subarrays

Given an integer sequence of length N, split it at two positions into three non-empty contiguous subarrays to maximize the total count of distinct integers across the three subarrays. Constraints: 3 ≤ N ≤ 3 × 105.

Setup: 16 candidates, 2 correct (idx=2 and idx=3). Pairwise µ ranking: [3, 0, 6, 13, 2, 9, ...] → selected idx=3 (µ=0.946) ✓ Pointwise ranking: [10, 14, 0, 2, 4, 13, 3, ...] → selected idx=10 (score=6/10) ✗ Pointwise scores: Correct idx=2 scored 4/10, correct idx=3 scored 3/10; incorrect idx=10 scored 6/10.

Pairwise-Selected Solution (idx=3): Correct ✓

def solve(): n = int(input()) a = list(map(int, input().split())) # Precompute prefix/suffix distinct counts prefix = [0] * (n + 1); seen = set() for i in range(n): seen.add(a[i]); prefix[i+1] = len(seen) suffix = [0] * (n + 1); seen = set() for i in range(n-1, -1, -1): seen.add(a[i]); suffix[i] = len(seen) # Enumerate all split positions with O(Nˆ2) middle segment max_sum = 0 for i in range(n - 2): for j in range(i+1, n-1): mid_seen = set(a[i+1:j+1]) max_sum = max(max_sum, prefix[i+1] + len(mid_seen) + suffix[j+1]) print(max_sum)

###### Pointwise-Selected Solution (idx=10): Incorrect ✗

def solve(): N = int(input()); A = list(map(int, input().split())) # Similar prefix/suffix precomputation, but the middle # segment enumeration has an off-by-one error in the # sliding window that miscounts distinct elements # when the left boundary advances past a repeated element.

Analysis: This hard algorithmic problem (N up to 3×105) requires careful handling of the middle segment’s distinct count. The pointwise verifier gave the correct solutions lower scores (3–4/10) than the incorrect idx=10 (6/10), likely because the correct solutions use a straightforward O(N2) enumeration that appears less “optimized.” Pairwise comparison, by directly contrasting solutions, ranked both correct solutions in the top 5 (µ = 0.946 and 0.773) while placing idx=10 at rank 10. This demonstrates that pairwise verification is less susceptible to surface-level heuristics about code quality and better at identifying solutions that produce correct outputs.

