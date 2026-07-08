# arXiv:2505.12992v4[cs.LG]12Jun2026

## Fractured Chain-of-Thought Reasoning

Baohao Liao∗⋆◦ Hanze Dong∗‡ Yuhui Xu∗†§ Doyen Sahoo¶ Christof Monz† Junnan Li¶ Caiming Xiong△ ⋆University of Amsterdam ◦eBay ‡Microsoft §Google Research ¶Salesforce △Recursive Superintelligence b.liao@uva.nl hanzedong@microsoft.com

#### Abstract

Inference-time scaling techniques have significantly bolstered the reasoning capabilities of large language models (LLMs) by harnessing additional computational effort at inference without retraining. Similarly, Chainof-Thought (CoT) prompting and its extension, Long CoT, improve accuracy by generating rich intermediate reasoning trajectories, but these approaches incur substantial token costs that impede their deployment in latency-sensitive settings. In this work, we first show that truncated CoT, which stops reasoning before completion and directly generates the final answer, often matches the full CoT sampling while using dramatically fewer tokens. Building on this insight, we introduce Fractured Sampling, a unified inference-time strategy that interpolates between full CoT and solution-only sampling along three orthogonal axes: (1) the number of reasoning trajectories, (2) the number of final solutions per trajectory, and (3) the depth at which reasoning traces are truncated. Through extensive experiments on five diverse reasoning benchmarks and several model scales, we demonstrate that Fractured Sampling consistently achieves superior accuracy-cost trade-offs, yielding steep log-linear scaling gains in Pass@k versus token budget. Our analysis reveals how to allocate computation across these dimensions to maximize performance, paving the way for more efficient and scalable LLM reasoning.

#### 1 Introduction

Recent advances in LLMs have enabled impressive capabilities in complex reasoning and problem solving (Guo et al., 2025; Kojima et al., 2022; Jaech et al., 2024; Brown et al., 2020; Hurst et al., 2024; Anthropic, 2024; Team et al., 2024). While much progress has been driven by scaling model size and training data (Hestness et al., 2017; Kaplan et al., 2020; Hoffmann

- et al., 2022), a complementary direction, inference-time scaling, has gained traction (Wang
- et al., 2023). This approach enhances performance by increasing computational effort at inference, without altering model parameters. Techniques such as self-consistency decoding (majority voting) (Wang et al., 2022), best-of-n sampling (Stiennon et al., 2020; Brown et al., 2024; Cobbe et al., 2021; Dong et al., 2023), and ensemble-style methods (Yao et al., 2023; Zhou et al., 2022; Liao et al., 2025) leverage multiple forward passes to produce more accurate and robust predictions from instructed models.

In parallel with these inference-time scaling methods, another line of work has focused on improving the quality of individual reasoning paths. Chain-of-Thought (CoT) prompting (Wei et al., 2022) has emerged as a particularly effective technique by encouraging models to articulate intermediate reasoning steps before arriving at a final answer. Recently, Long

∗BL, HD, YX contributed equally to this work. Correspondence to HD and YX. †Work done before joining Google.

###### MATH500 L5

###### AIME24

###### AIME25

###### AIMO2

###### GPQA

| |Vanilla Sampling<br><br>Truncated CoT Sampling| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| |DS-R1-Qwen-1.5B| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

70

25

30

25

20

65

25

20

15

20

60

20

15

10

pass@1(%)

55

15

15

10

5

50

50

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| |DS-R1-Qwen-7B| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

85

35

45

80

40

40

30

40

75

25

35

70

35

30

30

20

65

5 10 15 20 25 30

5 10 15 20 25 30

5 10 15 20 25 30

5 10 15 20 25 30

5 10 15 20 25 30

Max tokens (K)

- Figure 1: Pass@1 accuracy versus maximum token budget. Solid blue lines show the original full CoT sampling, while dashed orange lines show our truncated CoT + response approach. Across all benchmarks, truncating the CoT (and generating the final answer) achieves equal or even better accuracy with substantially fewer tokens, demonstrating that full CoT is not always necessary and that truncating CoT can save tokens without sacrificing performance.

Chain-of-Thought (Long-CoT) reasoning (Guo et al., 2025; Jaech et al., 2024) introduces longer and more diverse reasoning trajectories, often incorporating mechanisms like self-reflection and self-correction (Kumar et al., 2024). These extended CoTs explore a broader solution space and aggregate diverse intermediate steps into a single response. It has been shown to significantly improve accuracy and robustness, especially for tasks requiring multi-step or logical reasoning. The downside is their dramatic token usage, resulting in higher costs.

Combining inference-time scaling with Long-CoT methods (e.g., using Long-CoT with self-consistency decoding) further amplifies this computational burden. Each technique alone may require thousands of tokens per input; together, they often push token budgets to impractical levels, making such methods unsuitable for latency-sensitive or resourceconstrained applications. It raises a central question: Can we retain the benefits of Long-CoT reasoning without incurring the full cost?

To address this, we revisit the common assumption that complete Long-CoT traces are essential for accurate reasoning. Surprisingly, we find that incomplete CoT trajectories, i.e., traces truncated before the final answer, can still yield highly accurate results. As shown in Figure 1, simply truncating the CoT prefix and generating the answer (dashed orange) matches or even exceeds the accuracy of full CoT sampling (solid blue) given a max token constraint. This result challenges the notion that “more reasoning” always leads to better outcomes and suggests a new frontier for efficiency: partial reasoning traces.

To systematically trade off between cost and performance, we propose Fractured Sampling, a unified inference-time strategy that interpolates between full CoT and solution-only sampling. As illustrated in Figure 2(a), Fractured Sampling explores three dimensions:

- • Thinking trajectories: the number of distinct CoT prefixes sampled;
- • Solution diversity: the number of final solutions generated per prefix;
- • Thinking prefix length: the depth at which each CoT is truncated.

Figure 2(b) further reveals that the thinking steps dominate the overall token count, while the final solutions contribute minimally, highlighting ample opportunities to optimize the depth and breadth of the reasoning.

Contributions. Our key contributions are as follows: (1) We show that truncated CoT trajectories often achieve performance comparable to or better than full CoT, at a fraction of the inference cost. (2) We propose Fractured Sampling, a unified inference-time framework that jointly controls reasoning depth, diversity, and token efficiency. (3) We provide a comprehensive analysis of the scaling behavior of Fractured Sampling across multiple reasoning benchmarks, offering practical insights into efficient inference strategies.

- Step 1-1

S 1-1

S 1-m S 2-m

S 2-1

Solution 1

- Solution 1

- Solution 2

Step 1-2

- Step 2-1 Step 2-2 Solution 2

Qwen-1.5B Qwen-7B Qwen-14B DeepSeek-R1

Thinking Solution

20

| |
|---|

| |
|---|

Sampling Trajectories

| |
|---|

| |
|---|

TokenCount(K)

15

###### Prompt

Step 1 Step 2

Sampling Solutions

10

S h-1

5

Step 1 Step 2

S h-m

Thinking

0

Fractured Sampling

MATH500 Lv5 AIME24 AIME25 AIMO2 GPQA

Solution

Tasks

(a) Sampling strategies for reasoning LLMs. (b) Token statistics.

- Figure 2: (a) Comparison of sampling strategies for reasoning LLMs. Top: Sampling Trajectories–multiple complete reasoning chains are sampled independently. Middle: Sampling Solutions–a single reasoning chain is used to generate diverse final solutions. Bottom: Fractured Sampling–our proposed method samples across both multiple reasoning trajectories and intermediate reasoning steps, enabling fine-grained control over diversity and computation. (b) Token statistics across tasks and models. The thinking process dominates.

#### 2 Preliminary

Notations. Let x denote the input prompt and ε be a random seed used to introduce stochasticity. The instruct LLM generates an initial response as follows: z = f(x, ε), and a parser g extracts the final answer: y = g(z).

Baseline inference techniques. Before introducing our method, we review common sampling-based inference techniques widely used to enhance output quality from LLMs.

Vanilla Sampling: Generate n independent completions with different random seeds:

Fn(x, ε1:n) = {g ◦ f(x, εi) | i = 1, . . . , n} .

Pass@k: Estimate the probability that at least one of the k samples is correct:

pass@k = P (∃ yi ∈ Fk(x, ε1:k) s.t. yi is correct) .

Best-of-n: Select the most confident response among n candidates. If s(z) denotes a scoring function (e.g., reward model), the best-of-n output is:

ybest = g argmaxz

i∈f(x,ε1:n)s(zi) .

Our approach builds on these inference-time strategie by explicitly leveraging internal reasoning traces to enhance sample efficiency and answer diversity.

Reasoning LLMs and long-CoT thinking process. To better capture intermediate reasoning steps, reasoning-augmented LLMs use a CoT mechanism. Instead of producing a direct answer, the model first generates a reasoning trace:

h = [h1ε, · · · , hεH] = fh(x, ε), where H denotes the total number of reasoning steps. The final solution is then generated conditioned on the full thought process: z = fo(x, h, ε). This CoT formulation provides richer supervision and enables more structured sampling strategies, which our approach builds upon to enhance efficiency and performance.

To better reflect the internal reasoning process, we enhance diversity by sampling m additional random seeds for each of n thinking processes:

Fn,m(x, ε1:n, ε1:m) = g ◦ fo(x, fh(x, εi), εj) i = 1, · · · , n; j = 1, · · · , m .

However, standard sampling methods only operate on the reasoning trajectory or the final solutions, overlooking the model’s intermediate reasoning dynamics. To fully exploit the internal structure of CoT reasoning, we propose sampling not just across independent trajectories, but also across intermediate reasoning steps.

Algorithm 1 3D Sampling Framework (Full Trajectory → Segmentation → Solution Sampling)

- 1: Input: prompt x; trajectories n; depth segments H; solutions per depth m; selector S
- 2: Output: final answer yˆ
- 3: C ← ∅ // All candidate answers
- 4: for i = 1 to n do
- 5: Ti ← MODEL.GenerateFullCoT(x)
- 6: Tokenize Ti into {t1, . . . , tL}
- 7: s ← max(1, ⌊L/H⌋) // Segment size
- 8: for t = 1 to H do
- 9: pi,t ← detokenize({t1, . . . , tts})
- 10: for j = 1 to m do
- 11: y˜i,t,j ← MODEL.GenerateSolution(pi,t, x)
- 12: ai,t,j ← ExtractAnswer(y˜i,t,j)
- 13: C ← C ∪ {ai,t,j}
- 14: end for
- 15: end for
- 16: end for
- 17: yˆ ← S(C) // Majority voting or best-of-N
- 18: return yˆ

- 3 Fractured sampling for long-CoT reasoning To formalize intermediate reasoning, the partial reasoning trace up to step t is denoted as:

h1:ε t = [h1ε, · · · , hεt] = fht(x, ε). Our approach leverages intermediate reasoning traces to aggregate predictions, thereby enhancing both efficiency and diversity. The key idea is to decompose the response generation into multiple stages and perform aggregation not only over independent final responses but also across intermediate reasoning steps.

Fractured sampling. Fractured sampling extends this idea by incorporating intermediate reasoning stages directly into the sampling process. Specifically, we sample solutions at each step of the reasoning chain:

Fn,m,H(x, ε1:n, ε1:m,1:H) = g ◦ fo x, fht(x, εi), εj,t | i = 1, · · · , n; j = 1, · · · , m; t = 1, · · · , H} .

Here, fht(x, εi) denotes the partial reasoning trace up to step t, and εj,t is the random seed used for generating the response at that stage. By aggregating responses across all H intermediate steps, fractured sampling captures the evolving thought process and synthesizes diverse insights into a more robust final answer.

Fractured sampling offers two primary advantages: (1) Granular Aggregation: Integrating intermediate reasoning steps enables early detection of conclusions and avoid overthinking, improving the consistency of final predictions. (2) Enhanced Diversity: The multi-level sampling mechanism encourages a wide range of reasoning trajectories. Aggregating these paths produces a consensus that is more resilient to individual failures.

Three orthogonal dimensions of sampling. As shown in Algorithm 1, Fractured Sampling unifies and extends existing sampling strategies by operating along three orthogonal axes:

- • m: Solution Diversity — sampling multiple final outputs from a single reasoning trace.
- • n: Trajectory Diversity — sampling independent reasoning traces with various seeds (vanilla CoT).
- • H: Reasoning Depth Diversity — sampling at different intermediate stages of a single reasoning trace (unique to fractured sampling).

This tri-dimensional framework enables a fine-grained exploration of the cost–performance landscape. While m and n offer diversity at the output or full-trajectory level, the H

dimension uniquely captures the temporal evolution of reasoning, offering early, diverse, and efficient decision points. Together, they provide a powerful toolkit for scalable and reliable inference-time reasoning.

###### 3.1 Analysis of fractured sampling

Fractured sampling benefits from diverse solutions. By distributing samples across both trajectories and intermediate steps, fractured sampling capitalizes on diverse error modes to boost overall success. The following proposition provides an analysis.

Proposition 3.1 (Diversity Lower Bound, informal). Let Fk be the indicator of failure for branch sample k and qk = P(Fk = 1). Then the fractured-sampling success probability satisfies

pseg = 1 − Pr ∧kK=1Fk = 1 = 1 − E

and by inclusion–exclusion

K

### ∏

Fk ,

k=1

E

K

### ∏

Fk =

k=1

K

### ∏

### qk +∑

Cov(Fi,Fj) + · · · .

i<j

k=1

That is to say, negative covariance Cov(Fi,Fj) ≤ 0 means that failures at two different samples i, j tend not to coincide, i.e. the two sampling locations provide diverse error modes.

If we only consider the second order expansion, we have pseg ≥ 1 − ∏kK=1 qk = 1 − ∏tH=1(1 − pt)m. Fractured sampling spreads samples across intermediate steps to maximize this diversity: because failures are unlikely to all happen together, the probability that every sample fails is strictly less than the na¨ıve product of their marginal failure rates. Consequently, the overall success probability pseg is boosted above the independent-baseline 1 − ∏(1 − pt)m.

To understand the limits of fractured sampling, we examine two extreme correlation regimes among the K = mH branch samples.

Almost perfect correlation. When every sample fails or succeeds in unison (Fi = Fj almost surely), the entire set of K trials collapses to a single Bernoulli event. In this case,

Pr(F1 = · · · = FK = 1) = q, pseg = 1 − q,

so the sampling reduces to plain single-step sampling and yields no extra benefit. Sampling only along the m-axis (multiple outputs per trace) behaves similar to this.

Full independence. If all Fk are mutually independent with Pr(Fk = 1) = qk, then

K

### ∏

Pr(F1 = · · · = FK = 1) =

k=1

qk =

H

(1 − pt)m, pseg = 1 −

### ∏

t=1

H

(1 − pt)m.

### ∏

t=1

The sampling achieves the product-of-marginals bound: diversity arises purely from geometric averaging of each step’s success rate. Standard trajectory sampling (n-axis) behaves similar to this regime with a single successful rate p.

Intermediate regimes. Between these extremes, negative pairwise covariances (Cov(Fi,Fj) < 0) drive the all-fail probability below the independent baseline, delivering gains beyond simple marginal aggregation. By contrast, positive correlations (Cov(Fi,Fj) > 0) erode this advantage, interpolating smoothly between full independence and perfect correlation. Sampling along the depth dimension H exploits these intermediate correlations to maximize diversity and overall success.

As illustrated in Figure 3, the correlation matrices shows how failure events at different reasoning depths H co-occur across five diverse benchmarks. Dark green cells along the diagonal indicate that failures at the same depth are, by definition, almost perfectly

MATH500 L5

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 13

- 14

- 15

[Figure 1]

Hposition

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

H position

###### AIME24

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 13

- 14

- 15

[Figure 2]

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

H position

###### AIME25

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 13

- 14

- 15

[Figure 3]

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

H position

###### AIMO2

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 13

- 14

- 15

[Figure 4]

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

H position

###### GPQA

1.0

[Figure 5]

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 13

- 14

- 15

[Figure 6]

CorrelationCoefficient

0.5

0.0

0.5

1.0

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

H position

- Figure 3: Correlation matrices of binary failure indicators across intermediate reasoning depths (positions H) under fractured Sampling. Each cell shows the Pearson correlation coefficient between failure events at two depth positions; green denotes positively correlated failures (synchronized error modes), while pink denotes negatively correlated failures (diverse error modes) that fractured sampling exploits to boost overall success.

correlated. More interestingly, the off-diagonal pattern varies by task: many entries are light or even pink (negative), signalling that failures at two distinct depths tend not to happen simultaneously. This negative covariance across depths is precisely what fractured sampling exploits, by spreading samples over intermediate stages, it decorrelates error modes and thus markedly reduces the probability that all branch samples fail together. Benchmarks with stronger negative off-diagonal structure (e.g. GPQA) exhibit the largest gains from fractured sampling, confirming our theoretical diversity lower-bound analysis.

###### 3.2 Scaling laws along the trajectory dimension

In fractured sampling, we allocate computation across three orthogonal axes (n, m, H). Here, we hold the branching factor m and fracturing depth H constant, and investigate how increasing the number of independent trajectories n affects performance under a fixed token budget. Denote the total tokens consumed as

B(n, m, H) = n Cthinking + n m H Csolution = n Cthinking + mH Csolution , where Cthinking is the average tokens per trajectory spent on “thinking” (the reasoning prefix), and Csolution is the per-step cost of generating each candidate solution. Log-linear scaling behavior. Empirical studies reveal a remarkably consistent log-linear relationship between computational budget and success rate along each axis:

pass@k Bn ≈ Cn log Bn + cn, Bn = B(n,1,1), pass@k Bm ≈ Cm log Bm + cm, Bm = B(1, m,1), pass@k BH ≈ CH log BH + cH, BH = B(1,1, H).

Here, the constants Cn, Cm, CH measure the marginal gain in log-budget per unit improvement in pass rate, while cn, cm, cH capture dataset-specific offsets.

Depth yields the steepest slope. Across a range of tasks, we consistently find

CH ≥ max{Cn, Cm}, indicating that allocating tokens to deeper intermediate sampling (the H axis) produces the largest incremental improvements per token. Intuitively, early-stage branching captures coarse but high-signal glimpses of the solution space, allowing the model to “course-correct” before committing to full trajectories and thus yielding higher gains for each additional intermediate sample.

Beyond single-axis scaling. While single-axis laws offer valuable intuition, actual performance often improves when (n, m, H) are tuned jointly. Since the n-axis contributes additively and independently, we condition on fixed (m, H) and model

pass@k B ≈ Cm,H log B n | m, H + cm,H,

where the coefficient Cm,H encapsulates the combined effect of branching factor and depth. These cross-terms reveal synergistic gains or trade-offs between axes, guiding more nuanced budget allocations. We explore these interactions in Section 4.

###### MATH500 L5

###### AIME24

###### AIME25

###### AIMO2

###### GPQA

40.0

30

| |n H m<br><br>|
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

| |DS-R1-Qwen-1.5B|
|---|---|
| | |
| | |
| | |
| | |
| | |

80

40

37.5

85

70

25

35.0

35

80

60

32.5

20

50

75

30

30.0

40

27.5

15

70

30

10 20 30 40

20 30 40 60

20 30 40 60

20 30 40 60 80

10 20 30 40 60

90

80

70

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

| |DS-R1-Qwen7B|
|---|---|
| | |
| | |
| | |
| | |
| | |

97.5

65

Pass@k(%)

80

95.0

60

60

70

92.5

70

55

50

90.0

60

60

50

87.5

40

45

50

50

10 20 30 40

20 30 40 60

20 30 40 60

20 30 40 60 80

10 20 30 40 60

90.0

90

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| |DS-R1|
|---|---|
| | |
| | |
| | |

85

- 94
- 95
- 96
- 97

87.5

80

80

80

85.0

70

82.5

70

75

80.0

60

60

6 10 20 30

10 20 30 40

20 30 40 60

10 20 30 40 60

10 20 30

Number of tokens (K, log scale)

- Figure 4: Pass@k performance versus token budget. We compare: m–sampling only the final solution; n–sampling full reasoning trajectories; H–fractured sampling across all intermediate steps. Fractured sampling consistently yields higher pass@k at a given token budget. Refer to Figure B.2 and B.3 for more models with a similar pattern.

#### 4 Empirical results

Settings. All inference experiments are conducted using NVIDIA A100-80GB GPUs, leveraging the vLLM framework (Kwon et al., 2023). Following the sampling configuration recommended by Guo et al. (2025), we set temperature=0.6, top p=0.95, and max tokens=32768. Our primary focus is on models from the DeepSeek-R1 family (Guo et al., 2025), and we further validate our findings using reasoning models from Qwen3 (Team, 2025), SkyworkOR1 (He et al., 2025), DeepScaler (Luo et al., 2025) and GPT-OSS (Agarwal et al., 2025). For clarity, we refer to DeepSeek-R1 as DS-R1, DeepSeek-R1-Distill-Qwen-1.5B as DS-R1-Qwen1.5B, DeepScalerR-1.5B-Preview as DSR-1.5B, and Skywork-OR1-7B as SW-OR1-7B.

We evaluate on five challenging math and scientific reasoning benchmarks: MATH500 Level

##### 5 (Lightman et al., 2023), AIME24, AIME25 (MAA Committees, 2025), AIMO2 reference questions (Frieder et al., 2024), and the GPQA Diamond set (Rein et al., 2024). Unless otherwise specified, we set n = 16, H = 16 and m = 4. H = 16 indicates that the original thinking CoT is divided into 16 equally sized segments based on token count. E.g., the third fractured CoT consists of the first three segments of the full thinking trajectory.

###### 4.1 Scaling law for each dimension

- Figure 4 plots pass@k versus total tokens B for three sampling dimensions. Across all benchmarks and models, fractured sampling exhibits the steepest log-linear gains per token. In particular, we fit

pass@k B∗ ≈ C∗ log B∗ + c∗, ∗ ∈ {n, m, H},

and consistently observe CH ≥ max{Cn, Cm}. This confirms that allocating budget to intermediate-step branching yields higher marginal returns than either sampling more independent traces or more final solutions alone.

Interpreting the gains. Fractured sampling captures rich, underutilized variation in intermediate reasoning states, allowing the model to “course-correct” early and avoid committing to error-prone trajectories. This leads to: (1) Higher early returns: At small budgets, H–sampling yields a much steeper rise in pass rate than n or m, since few intermediate samples can quickly pinpoint correct partial reasoning. (2) Consistent dominance: The gap between the H-curve and the others persists across all budgets, demonstrating robustness to scale. (3) Task-dependent effect: Benchmarks with less positive error correlations across depths (e.g. GPQA) show the largest absolute improvement from fractured sampling, in line with our diversity-bound analysis.

###### AIME24

###### AIME25

###### AIMO2

###### GPQA

60

100

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| |DS-R1-Qwen-1.5B|
|---|---|
| | |
| | |
| | |
| | |

50

60

60

80

40

60

30

40

40

Pass@k(%)

20

40

30 100

30 100

30 100

30 100

90

100

| |H=1, m=1 H=1, m=4<br><br>H=16, m=1 H=16, m=4<br><br>|
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| |GPT-OSS-20B|
|---|---|
| | |
| | |
| | |
| | |

90

90

80

90

80

70

80

70

80

60

60

70

50

70

30 100

30 100

30 4×10 100

2×10 30 100

1 6 × 101

1 4 × 101 6 × 101

Number of tokens (K, log scale)

- Figure 5: Pass@k performance versus total token budget for four sampling schemes. In each subplot, we compare: H=1, m=1–only sampling full reasoning trajectory; H=1, m=4– sampling both full reasoning trajectories and the final solution; H=16, m=1–both full reasoning trajectory sampling and fractured sampling across all intermediate steps; H=16,

- m=4–sampling all three dimensions. n is in [1, 2, 4, 8, 16] for the five points (from left to right) on each line (n ≤ 8 on GPQA). Refer to Figure B.4 for more models.

H=1,m=1H=16,m=1H=-4,m=1H=1,m=4H=16,m=4H=-4,m=4

55.0

57.5

60.0

62.5

65.0

67.5

70.0

72.5

75.0

AverageAccuracy(%)

60.4

61.4

68.0

61.6

66.7

70.8 68.3

Best-of-N (BoN)

DS-R1-Qwen-14B

H=1,m=1 H=16,m=1H=-4,m=1 H=1,m=4 H=-4,m=4

55.0

57.5

60.0

62.5

65.0

67.5

70.0

72.5

75.0

AverageAccuracy(%)

66.7

65.3

69.9

66.8

71.3

68.0

Majority Voting (Maj)

DS-R1-Qwen-14B

Figure 6: BoN and Maj accuracy with different dimensional settings on DS-R1-Qwen-7B.

- n = 16 here for all settings. H = 1, m = 1 denotes the standard sampling setting. H = −4 here means that we take the last 4 solutions among all 16 predictions in the H dimension. Refer to Table B.2 for the detailed accuracy.

These results empirically validate that, under the same compute budget, fractured sampling shifts the inference-time scaling curve upward, achieving higher accuracy at lower cost by leveraging the temporal structure of chain-of-thought.

###### 4.2 Scaling law across dimensions

Thus far we have examined each sampling axis in isolation. Figure 5 extends this analysis by comparing four representative schemes that allocate budget across the solution (m) and depth (H) dimensions simultaneously, with the trajectory axis (n) swept to 16. We have: (1) (H=1, m=1): standard single-path CoT sampling (baseline). (2)(H=1, m=4): augment baseline with 4 final answers per trajectory. (3) (H=16, m=1): fractured sampling across 16 depths, one answer each. (4) (H=16, m=4): full three-axis sampling (both deep fracturing and multiple final answers).

Across all tasks and models, non-baseline schemes (excl. (H=1, m=4)) outperform (H=1, m=1) at fixed budget. More importantly, expanding H is usually more effective than expanding m. These multi-axis scaling laws reveal that, under the same token budget, the most efficient use of compute is to allocate tokens for temporal branches H.

With such a high pass@k from low token budget, our unified sampling across three dimensions potentially paves a new way for reinforcement learning (RL), where both high pass@k and efficient sampling are important for the large-scale RL training, leaving to future work.

Accuracy (%) ↑

Model Method

MATH500 AIME25 AIMO2 GPQA Avg. Avg Tokens / Question (K) ↓ DS-R1-1.5B

Vanilla 70.8 27.5 15.0 34.1 36.9 14.3 Early Stop +1.2 -0.0 +10.6 -0.3 +2.9 -2.9

Vanilla 76.5 41.7 20.0 19.2 39.4 8.0 Early Stop -0.9 -0.0 -0.0 +0.4 -0.1 -1.5

DSR-1.5B

Vanilla 89.0 45.0 47.5 48.6 57.5 10.9 Early Stop -0.4 -0.0 -0.0 +1.1 +0.2 -2.2

SW-OR1-7B

Table 1: The relative performance for early stop compared to vanilla sampling. Early stopping significantly saves inference budget (≈ 20%) while preserving accuracy.

###### 4.3 Accuracy across dimensions

In prior experiments, we reported the metric pass@k, which indicates whether a correct prediction is present among a set of generated samples. In this section, we further examine whether a correct solution can be identified by (1) a reward model from the predictions generated across the three sampling axes; or (2) majority voting. We employ the process reward model (PRM), specifically Qwen2.5-Math-PRM-72B (Zhang et al., 2025b), which has shown strong performance across a range of PRM benchmarks. Due to its limited context window (4K tokens), we score only the final solution, rather than the intermediate reasoning steps. The reward assigned to the final step is used as the overall score for the entire solution.

As shown in Figure 6, sampling with H = 1, m = 4 yields a modest improvement in average accuracy compared to the standard sampling setting of H = 1, m = 1 (61.6% vs. 60.4% for BoN and 66.8% vs 66.7% for Maj). Interestingly, increasing only the H dimension to H = 16, m = 1 leads to a slight improvement for BoN and degradation for Maj, which contrasts with our earlier observation that varying H is typically more effective than varying m in terms of pass@k. We hypothesize that incorporating all H = 16 generated solutions introduces excessive noise, making it challenging for a PRM to correctly identify the optimal solution. This may be due to two factors: (1) the Long-CoT model tends to generate coherent and logically consistent solutions, which are difficult for the PRM to differentiate; (2) the PRM is trained predominantly on simpler and short-CoT data and may struggle to evaluate responses to more complex and long ones. More noise also causes a worse Maj accuracy.

Motivated by the trend observed in Figure B.5—where later reasoning positions (i.e., higher H indices) are associated with improved accuracy—we apply a simple denoising strategy by discarding earlier solutions (H = 1 to H = 11) and retaining only the last four (H = −4). This simple adjustment significantly enhances performance, raising the accuracy from 61.4% (H = 16, m = 1) to 68.0% (H = −4, m = 1) for BoN and 65.3% to 69.9% for Maj. Further combining both dimensions (H = −4, m = 4) yields an accuracy of 70.8% for BoN and 71.3% for Maj, a 10.4% and 4.6% improvement over the baseline setting (H = 1, m = 1). Notably, this configuration even outperforms standard sampling with a larger model that has twice the number of parameters (70.8% vs. 68.3% for BoN, and 71.3% vs 68.0% for Maj).

###### 4.4 Early stopping for efficient generation

Here we further explore whether the consistency of predictions across the H dimension can be leveraged for early stopping. Specifically, if a particular prediction appears with high frequency (i.e., exceeds a predefined threshold) across multiple H positions, we consider this as a signal to terminate the generation early, thereby reducing computational cost.

As illustrated in Figure B.5, prediction accuracy tends to be low at earlier positions. When the reasoning trace is divided into too many intermediate steps (i.e., a larger H), the model must generate a correspondingly large number of partial solutions, each requiring additional tokens. To balance computational efficiency and accuracy, we empirically initialize the first H position at a token index of 6144 and evaluate predictions at every subsequent 2048token interval. For example, given a question, the model first generates 6144 reasoning tokens. Based on these tokens, a solution is generated and a prediction is extracted. Then,

conditioned on the original question and the previously generated 6144 reasoning tokens, the model continues generating another 2048 tokens to produce the next prediction. Generation terminates once the same prediction occurs more than once or when the maximum token limit (max tokens) is reached. In the latter case, we adopt the final prediction, as later predictions tend to benefit from more extensive reasoning.

As shown in Table 1, this early stopping strategy preserves accuracy and, in some cases, improves it—achieving a 2.9% increase for DeepScaleR-1.5B-Preview. In terms of computational efficiency, early stopping reduces the number of generated tokens by approximately 20% compared to standard generation. Notably, this method is simple to implement and requires no additional training.

###### 4.5 More results

Please refer to Appendix B.1 for a new method, linear depth-weighted aggregation, that is designed to handle the noisy prediction problem in a more elegant way; and Appendix B.2 for the scaling behavior w.r.t. the latency instead of number of tokens.

#### 5 Related work

Test-time scaling law. Scaling laws have traditionally described how model performance improves with increased training compute (Hestness et al., 2017; Kaplan et al., 2020; Hoffmann et al., 2022), e.g., through more supervised fine-tuning or reinforcement learning steps. However, a complementary class of test-time scaling laws has emerged (Snell et al., 2024; Jaech et al., 2024), which characterizes performance gains obtained purely by increasing inference-time budget, without modifying model parameters. This includes techniques such as self-consistency decoding (Wang et al., 2022), best-of-n sampling (Brown et al., 2024; Cobbe et al., 2021; Dong et al., 2023). On the other hand, CoT prompting, where performance improves with more samples or longer reasoning traces (Wei et al., 2022). Recent work, including the O1 and R1 series (Jaech et al., 2024; Guo et al., 2025), further demonstrates that extended trajectories (e.g., Long CoT) with multiple rollouts yields predictable improvements under test-time scaling curves.

On the other hand, Process Reward Models (PRMs) (Lightman et al., 2023; Zhang et al., 2024a; Wang et al., 2023) further enable fine-grained control by assigning dense, steplevel rewards, which can guide search methods like Monte Carlo Tree Search (Luo et al., 2024). However, most approaches scale only along coarse dimensions, such as sample count or token length, or require external supervision via PRMs for finer control. In this work, we propose a more fine-grained view through Fractured Sampling without relying on PRMs, which explicitly decomposes generation into multi-stage reasoning traces and enables aggregation at intermediate steps. This design reveals richer scaling behaviors across trajectory depth, diversity, and stage-wise composition, and offers a more nuanced understanding of inference-time compute allocation.

Efficient sampling for LLMs. As large language models grow in size and capability, their inference cost becomes a significant bottleneck (Wan et al., 2023), especially when relying on multi-sample or multi-turn decoding strategies in reinforcement learning (Ouyang et al.,

- 2022; Xiong et al., 2023; Dong et al., 2024; Xiong et al., 2025; Shao et al., 2024) or large-scale serving (Ainslie et al., 2023). This has motivated a line of work on efficient sampling, which aims to reduce compute without sacrificing performance. Approaches such as speculative decoding (Stern et al., 2018; Leviathan et al., 2023; Xia et al., 2024; Chen et al., 2023a; Zhang et al., 2023; Sun et al., 2024; Chen et al., 2023b; Li et al., 2024b; Liao et al., 2025), KV cache pruning (Xu et al., 2024; Xiao et al., 2023; Zhang et al., 2024c; Li et al., 2024a; Ge et al.,
- 2023; Zhang et al., 2024b; Yang et al., 2024; Liu et al., 2024), are widely used in real-world LLM services. While these methods achieve notable efficiency gains, they largely operate within a fixed test-time scaling curve: improving the efficiency of a given point on the curve without fundamentally changing its shape. In contrast, we argue that the most principled path forward lies in reshaping the scaling law itself: by rethinking how inference budget is allocated across reasoning stages and sampling axes, one can unlock qualitatively different

compute-performance tradeoffs. Our proposed Fractured Sampling method embodies this principle, revealing richer scaling dynamics and enabling more cost-effective reasoning through staged aggregation.

Reducing redundancy of CoT. Another line of efficiency for long-CoT LLMs focuses on reducing the redundancy of CoT by using training Xia et al. (2025); Zhang et al. (2025a) or training-free Wang et al. (2026); Fang et al. (2026); Ma et al. (2025) methods. Our early stopping has a similar effect in a training-free way. However, early stopping is only one of the practical implementation of Fractured Sampling. In this paper, we mainly explore the diversity of the H dimension, and show it achieves the best token-accuracy trade-off.

#### 6 Conclusion

In this work, we introduce Fractured Sampling, a new Long-CoT inference paradigm that seamlessly unifies partial-trace and final-answer sampling by jointly controlling reasoning depth, trajectory diversity, and solution diversity. We uncover consistent log–linear scaling trends along each axis and offer theoretical insights into how sampling across intermediate reasoning steps maximizes diversity and per-token gains. Fractured Sampling redefines the cost–performance frontier of chain-of-thought inference, enabling powerful reasoning in LLMs with lower computational overhead.

#### References

Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025.

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebr´on, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.

AI Anthropic. Claude 3.5 sonnet model card addendum. Claude-3.5 Model Card, 3, 2024. Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher

R´e, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023a.

Ziyi Chen, Xiaocong Yang, Jiacheng Lin, Chenkai Sun, Kevin Chen-Chuan Chang, and Jie Huang. Cascade speculative drafting for even faster llm inference. arXiv preprint arXiv:2312.11462, 2023b.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, KaShun SHUM, and Tong Zhang. RAFT: Reward ranked finetuning for generative foundation model alignment. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=m7p5O7zblY.

Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen Sahoo, Caiming Xiong, and Tong Zhang. Rlhf workflow: From reward modeling to online rlhf. arXiv preprint arXiv:2405.07863, 2024.

Gongfan Fang, Xinyin Ma, and Xinchao Wang. Thinkless: Llm learns when to think. Advances in neural information processing systems, 38:151268–151295, 2026.

Simon Frieder, Sam Bealing, Arsenii Nikolaiev, Geoff C. Smith, Kevin Buzzard, Timothy Gowers, Peter J. Liu, Po-Shen Loh, Lester Mackey, Leonardo de Moura, Dan Roberts, D. Sculley, Terence Tao, David Balduzzi, Simon Coyle, Alex Gerko, Ryan Holbrook, Addison Howard, and XTX Markets. Ai mathematical olympiad - progress prize 2, 2024. Kaggle.

Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells you what to discard: Adaptive kv cache compression for llms. arXiv preprint arXiv:2310.01801, 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Yang Liu, and Yahui Zhou. Skywork open reasoner series, 2025. Notion Blog.

Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan Kianinejad, Md Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou. Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409, 2017.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.

Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, et al. Training language models to self-correct via reinforcement learning. arXiv preprint arXiv:2409.12917, 2024.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, pp. 19274–19286. PMLR, 2023.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation. arXiv preprint arXiv:2404.14469, 2024a.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle: Speculative sampling requires rethinking feature uncertainty. arXiv preprint arXiv:2401.15077, 2024b.

Baohao Liao, Yuhui Xu, Hanze Dong, Junnan Li, Christof Monz, Silvio Savarese, Doyen Sahoo, and Caiming Xiong. Reward-guided speculative decoding for efficient llm reasoning. arXiv preprint arXiv:2501.19324, 2025.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. Kivi: A tuning-free asymmetric 2bit quantization for kv cache. arXiv preprint arXiv:2402.02750, 2024.

Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Meiqi Guo, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, et al. Improve mathematical reasoning in language models by automated process supervision. arXiv preprint arXiv:2406.06592, 2024.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025. Notion Blog.

Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia.

Reasoning models can be effective without thinking. arXiv preprint arXiv:2504.09858, 2025. MAA Committees. AIME Problems and Solutions. https://artofproblemsolving.com/

wiki/index.php/AIME Problems and Solutions, 2025.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level googleproof q&a benchmark. In First Conference on Language Modeling, 2024.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. Blockwise parallel decoding for deep autoregressive models. Advances in Neural Information Processing Systems, 31, 2018.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in neural information processing systems, 33:3008–3021, 2020.

Hanshi Sun, Zhuoming Chen, Xinyu Yang, Yuandong Tian, and Beidi Chen. Triforce: Lossless acceleration of long sequence generation with hierarchical speculative decoding. arXiv preprint arXiv:2404.11912, 2024.

Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

Qwen Team. Qwen3, April 2025. URL https://qwenlm.github.io/blog/qwen3/.

Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, Zhongnan Qu, Shen Yan, Yi Zhu, Quanlu Zhang, Mosharaf Chowdhury, et al. Efficient large language models: A survey. arXiv preprint arXiv:2312.03863, 1, 2023.

Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. arXiv preprint arXiv:2312.08935, 2023.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.

Yiming Wang, Pei Zhang, Siyuan Huang, Baosong Yang, Zhuosheng Zhang, Fei Huang, and Rui Wang. Sampling-efficient test-time scaling: Self-estimating the best-of-n sampling in early decoding. Advances in Neural Information Processing Systems, 38:162137–162174, 2026.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, and Zhifang Sui. Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding. arXiv preprint arXiv:2401.07851, 2024.

Heming Xia, Chak Tou Leong, Wenjie Wang, Yongqi Li, and Wenjie Li. Tokenskip: Controllable chain-of-thought compression in llms. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 3351–3363, 2025.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.

Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, and Tong Zhang. Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint. arXiv preprint arXiv:2312.11456, 2023.

Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan Li, Nan Jiang, Tong Zhang, Caiming Xiong, et al. A minimalist approach to llm reasoning: from rejection sampling to reinforce. arXiv preprint arXiv:2504.11343, 2025.

Yuhui Xu, Zhanming Jie, Hanze Dong, Lei Wang, Xudong Lu, Aojun Zhou, Amrita Saha, Caiming Xiong, and Doyen Sahoo. Think: Thinner key cache by query-driven pruning. arXiv preprint arXiv:2407.21018, 2024.

Dongjie Yang, XiaoDong Han, Yan Gao, Yao Hu, Shilin Zhang, and Hai Zhao. Pyramidinfer: Pyramid kv cache compression for high-throughput llm inference. arXiv preprint arXiv:2405.12532, 2024.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.

Hanning Zhang, Pengcheng Wang, Shizhe Diao, Yong Lin, Rui Pan, Hanze Dong, Dylan Zhang, Pavlo Molchanov, and Tong Zhang. Entropy-regularized process reward model. arXiv preprint arXiv:2412.11006, 2024a.

Jintian Zhang, Yuqi Zhu, Mengshu Sun, Yujie Luo, Shuofei Qiao, Lun Du, Da Zheng, Huajun Chen, and Ningyu Zhang. Lightthinker: Thinking step-by-step compression. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 13318–13339, 2025a.

Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, and Sharad Mehrotra. Draft & verify: Lossless large language model acceleration via self-speculative decoding. arXiv preprint arXiv:2309.08168, 2023.

Yichi Zhang, Bofei Gao, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao Chang, Junjie Hu, Wen Xiao, et al. Pyramidkv: Dynamic kv cache compression based on pyramidal information funneling. arXiv preprint arXiv:2406.02069, 2024b.

Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. The lessons of developing process reward models in mathematical reasoning. arXiv preprint arXiv:2501.07301, 2025b.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher R´e, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. Advances in Neural Information Processing Systems, 36, 2024c.

Denny Zhou, Nathanael Sch¨arli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, et al. Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625, 2022.

#### A Proof of the diversity lower bound

Proof. Let qi = Pr(Fi = 1) = E[Fi] and denote

µij = E[FiFj], µijk = E[FiFjFk], µijkl = E[FiFjFkFℓ]. Define the joint cumulants

κij = µij − qiqj, κijk = µijk − µijqk − µikqj − µjkqi + 2qiqjqk.

Using the inclusion-exclusion identity ∏k Fk = ∑I⊆[K](−1)|I| ∏i∈I(1 − Fi) and collecting equal-order terms yields the exact expansion:

E ∏kK=1 Fk = ∏kK=1 qk + ∑i<j κij + ∑i<j<k κijk + ∑i<j<k<ℓ κijkl + · · · + κ1,2,...,K (1)

The dots represent cumulants of order four and higher. (1) can be written compactly as

K

Fk = ∑

∏

κI,

E

k=1

I⊆[K]

where κI is the joint cumulant on the index set I (with κ{i} = qi and κ∅ = 1).

| |
|---|

#### B More results

###### B.1 Linear depth-weighted aggregation

We further design a more elegant approach to weight predictions along the H dimension. As shown in Figure B.5, overall accuracy is largely proportional to reasoning depth. This observation motivates a linear depth-weighted selection mechanism for H-dimension aggregation. For simplicity, we ignore the m dimension here by setting m = 1.

Let the LLM produce an answer a at depth t, where the reasoning depth t ∈ {1, . . . , H}. For each instance of answer a produced at depth t, we obtain a selector score s(a, t) (e.g., a PRM score; for majority voting, we set s(a, t) = 1) and define a depth weight

t ∑kH=1 k

wt =

2t H(H + 1)

.

=

This weighting scheme ensures that:

- • deeper predictions receive higher weight;
- • early, noisier predictions receive substantially lower weight;
- • weights sum to one;
- • the mechanism is selector-agnostic and can be applied to either PRM or majority voting.

We then aggregate weighted scores for each canonical answer a across all depths and trajectories:

Sagg(a) = ∑

wt · s(a, t).

all occurrences t of a

Finally, the selected answer is

aˆ = argmax

Sagg(a).

a

As shown in Table B.1, linear depth-weighted aggregation outperforms the early denoising method (H = −4) and significantly improves upon the original H = 1 setting.

Metric Method H m MATH500 L5 AIME24 AIME25 AIMO2 GPQA Avg.

Original 1 1 90.3 63.3 53.3 40.0 55.1 60.4 Original 16 1 90.3 70.0 53.3 40.0 53.5 61.4 Original -4 1 93.3 73.3 60.0 60.0 53.5 68.0 Linear depth-weighted 16 1 96.3 76.7 60.0 60.0 56.1 69.8

BoN

Original 1 1 95.5 76.7 60.0 50.0 51.5 66.7 Original 16 1 94.0 73.3 60.0 50.0 49.0 65.3 Original -4 1 96.3 76.7 63.3 60.0 53.0 69.9 Linear depth-weighted 16 1 95.9 76.7 66.7 60.0 52.2 70.3

Maj

- Table B.1: Comparison between the orginal method (as Figure 6) and linear depth-weighted aggregation.

###### B.2 Latency

###### In Figure B.1, we include the scaling behavior w.r.t. the wall clock time. The pattern is similar to the token one.

###### MATH500 L5

###### AIME24

###### AIME25

###### AIMO2

###### GPQA

40

30.0

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

42.5

87.5

80

38

27.5

40.0

85.0

70

36

25.0

37.5

82.5

34

35.0

80.0

60

22.5

77.5

32.5

32

20.0

50

75.0

30.0

30

17.5

40

72.5

27.5

28

###### Pass@k(%)

15.0

70.0

30

101 2×101 3×101 4×101

101 2×101 3×101 4×101

2×101 3×101 4×101 6×101 102

2 × 101 3 × 101 4 × 101 6 × 101

2 × 101 3 × 101 4 × 101 6 × 101

Number of tokens (K, log10 scale)

40

30.0

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

42.5

87.5

80

38

27.5

40.0

85.0

70

36

25.0

37.5

82.5

34

35.0

80.0

60

22.5

77.5

32.5

32

20.0

50

75.0

30.0

30

17.5

40

72.5

27.5

28

15.0

70.0

30

101

101

101

101

101

Wall-clock time (second, log10 scale)

- Figure B.1: Pass@k performance versus token budget or wall-clock time. We compare: m–sampling only the final solution; n–sampling full reasoning trajectories; H–fractured sampling across all intermediate steps. DS-R1-Qwen-1.5B is utilized here. Fractured sampling consistently yields higher pass@k at a given token or time budget.

###### MATH500 L5

###### AIME25

###### GPQA

80

| |n H m<br><br>|
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| |DeepScaleR-1.5B-Preview|
|---|---|
| | |
| | |
| | |
| | |
| | |

52.5

90

70

50.0

60

47.5

85

50

45.0

40

80

42.5

Pass@k(%)

30

6 10 20

20 30 40 60

10 20 30 40 60

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| |Qwen3-1.7B|
|---|---|
| | |
| | |
| | |
| | |

45

70

90

60

40

85

50

35

80

40

10 20 30 40

20 30 40 60

10 20 30 40 60

Number of tokens (K, log scale)

- Figure B.2: Pass@k performance versus total token budget for three sampling schemes In each subplot, we compare: m (green dotted)–sampling only the final solution; n (blue solid)–sampling full reasoning trajectories; H (orange dashed)–fractured sampling across all intermediate steps. Rows correspond to DeepScaleR-1.5B-Preview and Qwen3-1.7B models models. Fractured sampling (H) consistently yields higher pass@k at a given token budget.

| |n H m<br><br>|
|---|---|
| | |
| | |
| | |
| | |

2×101 3×101 4×101 6×101 102

70

75

80

85

pass@K

AIME24

| | |
|---|---|
| | |
| | |
| | |

3×101 4×101 6×101 102

60

70

80

AIME25

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

3×101 4×101 6×101 102

50

55

60

65

70

AIMO2

| | |
|---|---|
| | |
| | |
| | |
| | |

2 × 101 3 × 101 4 × 101 6 × 101

70

75

80

85

GPQA

Number of tokens (K, log scale)

- Figure B.3: Pass@k performance versus total token budget for three sampling schemes on GPT-OSS-20B. In each subplot, we compare: m (green dotted)–sampling only the final solution; n (blue solid)–sampling full reasoning trajectories; H (orange dashed)–fractured sampling across all intermediate steps. Fractured sampling (H) consistently yields higher pass@k at a given token budget.

###### MATH500 L5

###### AIME25

###### GPQA

95

| |H=1, m=1 H=1, m=4<br><br>H=16, m=1 H=16, m=4<br><br>|
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| |DeepScaleR-1.5B-Preview|
|---|---|
| | |
| | |
| | |

80

60

90

60

85

50

40

80

Pass@k(%)

40

10 100

10 100

10 100

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| |Qwen3-1.7B|
|---|---|
| | |
| | |
| | |

95

70

80

60

90

50

85

60

40

80

40

30

10 100

20 100

10 60

Number of tokens (K, log scale)

- Figure B.4: Pass@k performance versus total token budget for four sampling schemes on five benchmarks. In each subplot, we compare: H=1, m=1–only sampling full reasoning trajectory; H=1, m=4–sampling both full reasoning trajectories and the final solution; H=16,

- m=1–both full reasoning trajectory sampling and fractured sampling across all intermediate steps; H=16, m=4–sampling all three dimensions. n is in [1, 2, 4, 8, 16] for the five points (from left to right) on each line. Rows correspond to DeepScaleR-1.5B-Preview and Qwen31.7B models. MATH500 L5 is saturated here, resulting in a less efficient gain from dimensions H and m.

Metric H m MATH500 L5 AIME24 AIME25 AIMO2 GPQA Avg.

BoN

DS-R1-Qwen-7B

- 1 1 90.3 63.3 53.3 40.0 55.1 60.4 16 1 90.3 70.0 53.3 40.0 53.5 61.4

- -4 1 93.3 73.3 60.0 60.0 53.5 68.0 1 4 90.3 70.0 53.3 40.0 54.6 61.6

16 4 92.5 73.3 60.0 50.0 57.6 66.7

- -4 4 94.8 73.3 60.0 70.0 56.1 70.8

DS-R1-Qwen-14B

- 1 1 91.8 80.0 60.0 50.0 59.6 68.3

Maj

DS-R1-Qwen-7B

1 1 95.5 76.7 60.0 50.0 51.5 66.7 16 1 94.0 73.3 60.0 50.0 49.0 65.3

- -4 1 96.3 76.7 63.3 60.0 53.0 69.9 1 4 95.5 76.7 60.0 50.0 52.0 66.8
- -4 4 96.3 80.0 66.7 60.0 53.5 71.3

DS-R1-Qwen-14B 1 1 94.0 83.3 53.3 50.0 59.6 68.0

Table B.2: Best-of-N and majority voting accuracy with different dimensional settings.

- n = 16 here for all settings. H = 1, m = 1 denotes the standard sampling setting. H = −4 here means that we take the last 4 solutions among all 16 predictions in the H dimension.

DS-R1-Qwen-1.5B

###### DS-R1-Qwen-7B

###### DS-R1

30

Accuracy(%)

40

60

AIME25

AIMO2

20

GPQA

20

40

10

1 3 5 7 9 11 13 15

1 3 5 7 9 11 13 15

1 3 5 7 9 11 13 15

H position

- Figure B.5: Accuracy vs. the position of Fractured CoT. We split the whole reasoning

trajectory into 16 intermediate steps equally, and observe: (1) Even with a 161 reasoning trajectory, the accuracy is still decent; (2) More reasoning tokens lead to higher accuracy.

