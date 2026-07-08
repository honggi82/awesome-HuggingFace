# arXiv:2510.08457v1[cs.CL]9Oct2025

ARES: MULTIMODAL ADAPTIVE REASONING VIA DIFFICULTY-AWARE TOKEN-LEVEL ENTROPY SHAPING

Shawn Chen1∗, Yue Guo1∗, Yimeng Ye3, Shijue Huang2, Wenbo Hu1, Haoxi Li2, Manyuan Zhang4, Jiayu Chen2, Song Guo2, Nanyun Peng1

1University of California, Los Angeles 2The Hong Kong University of Science and Technology 3Columbia University 4The Chinese University of Hong Kong

Code Page: https://github.com/shawn0728/ARES

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 1: Accuracy comparison across selected open-source reasoning models on nine multimodal and textual benchmarks. Each group represents 3B-scale and 7B-scale models evaluated under the same benchmarks. The rightmost column (“Avg.”) reports the average accuracy over all selected benchmarks, showing the overall advantage of the proposed adaptive reasoning framework. Our ARES-7B achieves superior performance.

ABSTRACT

Recent advances in multimodal large reasoning models (MLRMs) have substantially improved their ability to solve complex textual and visual tasks. However, these models tend to overthink on simple problems, producing unnecessarily lengthy reasoning traces, while under-exploring on challenging ones, leading to missed solutions. To address this imbalance, we propose ARES, a unified opensource framework for adaptive reasoning that dynamically allocates exploration effort based on task difficulty. Our approach is motivated by two key empirical findings: (i) while single-token entropy is noisy, high window-entropy (HWE) tokens (token-level entropies averaged under a sliding window) can reliably capture reasoning-critical moments; and (ii) reducing HWE usage benefits easy problems, while increasing it is essential for solving hard ones. Building on these insights, ARES introduces a two-stage training pipeline. In the Adaptive ColdStart stage, we curate multimodal and textual data paired with reasoning traces of length proportional to problem difficulty, equipping the model with initial difficulty awareness. In the second stage, we develop Adaptive Entropy Policy Optimization (AEPO), which uses HWE tokens as exploration triggers to decide when to explore, and a hierarchical entropy reward with dynamic KL control to decide how much to explore. Extensive experiments demonstrate that ARES achieves superior performance and reasoning efficiency across diverse mathematical, logical,

∗Equal contributions.

and multimodal benchmarks, while closing the gap to leading commercial systems under significantly lower inference costs.

- 1 INTRODUCTION

Leveraging long Chain-of-Thought (CoT) reasoning with reflection, Multimodal Large Reasoning Models (MLRMs) have showcased strong capabilities on both complex textual and visual tasks (Comanici et al., 2025; Guo et al., 2025). While long CoT reasoning substantially improves performance on complex reasoning tasks, it often causes models to generate unnecessarily lengthy reasoning even for easy tasks (Qu et al., 2025). This substantially increases inference costs and latency, limiting MLRMs’ usability in various real-world scenarios.

Recent studies have explored both training-free (Han et al., 2025; Yang et al., 2025b) and trainingbased strategies (Zhang et al., 2025; Arora & Zanette, 2025; Ling et al., 2025; Tu et al., 2025) to mitigate the issue of verbose long-thought outputs. While these methods alleviate overthinking, they often cause model performance degradation (Huang et al., 2025b). To balance model’s efficiency and accuracy, adaptive reasoning mechanisms have become a promising research direction, which dynamically adjust reasoning effort to trade off performance against computational cost. Existing approaches attempt this by either curating cold-start data with varying difficulty (Wang et al., 2025e; Zhang et al., 2025; Yu et al., 2025a) or using difficulty-aware penalties during RL (Huang et al., 2025a; Shen et al., 2025b). However, such methods tend to encourage over-exploration on difficult problems, leading to unnecessarily verbose reasoning traces, and thus still fail to deliver significant performance improvements.

To address these challenges, we seek to answer two foundational questions for adaptive reasoning mechanism: When should a model be encouraged to explore? How much reasoning effort should be allocated during exploration? To gain some insights, the paper presents a detailed preliminary analysis of exploration during the RL stage of MLRMs, specifically investigating how exploration affects the overall model’s performance and efficiency. We meticulously design experiments by constructing an easy, medium and hard set for study the trade off of RL exploration cost against accuracy. As illustrated in Figure 4, our findings can be summarized in two key observations. Firstly, high token-level entropy in MLRMs not only captures tokens relevant to reasoning but also often includes noise such as punctuation, formulas, or other superficial elements. By comparison, window entropy, the mean entropy computed over consecutive tokens, more reliably identifies pivotal decision points that guide the reasoning trajectory across multiple potential pathways. Secondly, for easy problems, reducing the number of high window entropy (HWE) tokens shortens the reasoning trace and improves performance. For complex problems, by contrast, increasing the number of HWE tokens promotes more thorough exploration, thereby enhancing the model’s ability to solve challenging tasks. These findings highlight HWE tokens as an exploration trigger for adaptive reasoning in MLRMs, improving performance while reducing inference costs.

Motivated by the analysis findings, we propose a two-stage adaptive reasoning training approach ARES, Adaptive Reasoning via difficulty-aware token-level Entropy reward Shaping for multimodal models. Specifically, first, during the Adaptive Cold-Start stage, we instill an initial difficulty awareness by fine-tuning the model on data where reasoning length is explicitly correlated with problem complexity. Then in the second stage, we introduce Adaptive-Entropy Policy Optimization (AEPO), which uses high window entropy regions to trigger exploration and a hierarchical reward aware of difficulty to control its depth. This two-stage approach enables ARES to adaptively allocate reasoning effort, rewarding deep exploration on complex problems while penalizing overthinking on simpler ones.

In summary, our main contributions are as follows:

- • We empirically demonstrate that high–window-entropy (HWE) tokens can guide MLRMs in exploration and reveal distinct exploration behaviors between easy and hard tasks.
- • We introduce ARES, with meticulously curated high-quality data for cold-start, and AEPO with token-level entropy reward shaping for RL.

[Figure 1]

- Figure 2: (a) F1 Score vs. threshold percentile across different window sizes. Window-based entropy aggregation consistently outperforms single-token entropy, especially at higher thresholds. (b) A word cloud visualization of semantically filtered high-entropy tokens, where font size reflects relative frequency. These tokens (e.g., explain, assume, constraint, conclude) correspond to reasoning triggers that mark the onset of logical transitions, highlighting the interpretable basis of our entropy-based reward.

[Figure 2]

- Figure 3: Training dynamics of ValLine GRPO on Coldstart Model: (a) average response length, (b) number of high-entropy tokens, and (c) accuracy, all measured across iterations. The trends indicate that the growth in high-entropy tokens is closely aligned with increases in response length and accuracy.

• Extensive experiments demonstrate that ARES achieves superior performance and inference efficiency across a wide range of mathematical, general knowledge, textual, and multimodal reasoning benchmarks.

- 2 RELATIONS BETWEEN TOKEN ENTROPY AND REASONING DIFFICULTY

In this section, we show two intriguing findings involving the exploration and effectiveness of MLRMs, paving the way for the proposed ARES in Section 3.

- 2.1 WINDOW ENTROPY SERVES AS A TRIGGER FOR EXPLORATION IN MLRMS

Recent works (Wang et al., 2025b; Zheng et al., 2025) highlight that high-entropy tokens often mark the starting points of reasoning forks, where the model must branch into alternative continuations. Such points are critical for determining the depth of reasoning, since they open up new solution paths. However, token-level entropy Ht discussed in Appendix B only captures local uncertainty at a single step and is often dominated by short-lived lexical ambiguities (e.g., punctuation, function words). Conversely, discourse markers that indicate major logical transitions (e.g., “but,” “however”) may appear with very low entropy, despite signaling important shifts in reasoning. Hence, relying solely on single-token entropy is noisy and insufficient for reliably identifying when exploration should be triggered.

From a cognitive perspective, humans rarely experience uncertainty as an instantaneous hesitation on a single word. Instead, when encountering a conceptual bifurcation, an entire segment of reasoning

tends to remain unstable. Motivated by this intuition, we aggregate token-level entropies into a sliding-window statistic that captures the persistence of uncertainty across consecutive steps:

t+w−1

1 w

H¯t:w =

Hτ,

τ=t

where w is the window size hyper-parameter controlling how many consecutive tokens are considered.

This measure highlights regions where the model sustains high uncertainty over multiple tokens, thus providing a smoother and more semantically aligned indicator of reasoning-critical moments. Window entropy therefore reduces the noise of single-token signals while better localizing genuine reasoning bifurcations.

Our empirical analysis supports this intuition. As shown in Figure 2(a), window entropy achieves consistently higher F1 scores than single-token entropy for detecting reasoning-critical tokens. Moderate windows (4–8 tokens) offer the best trade-off: they smooth local noise while remaining focused enough to capture sustained uncertainty. In contrast, single-token measures are overly sensitive to lexical artifacts, and long windows (16–32 tokens) dilute the signal with low-entropy tokens. These results validate window entropy as a more robust diagnostic of reasoning uncertainty and motivate its role as a central trigger in our Adaptive Exploration Policy Optimization (AEPO) framework (Section 3.2.1).

- 2.2 ENTROPY–DIFFICULTY INTERACTION

Exploration should not be uniform across all problem difficulties. Intuitively, for easy problems, the model already possesses sufficient knowledge to arrive at the correct answer; excessive highentropy branching in such cases is likely to lead to unnecessary verbosity or even distract from the correct reasoning path. In contrast, hard problems inherently require sustained exploration: reaching the correct answer typically involves probing multiple solution paths, which manifests as longer responses and more high-entropy tokens.

We validate these intuitions by analyzing model generations using window-entropy statistics. As illustrated in Figure 4(b)(iii), exploration interacts strongly with problem difficulty. In the easy set, samples with high-entropy counts below a batch-adaptive threshold attain higher accuracy while producing shorter responses, indicating that suppressing unnecessary exploration is beneficial. By contrast, in the hard set, above-threshold samples achieve higher accuracy but at the cost of longer responses, showing that sustained exploration improves success on challenging problems.

Conditioning on correctness further highlights this pattern. For easy problems, correct cases contain markedly fewer high-entropy tokens and shorter responses compared to incorrect ones. For hard problems, correct cases typically involve longer responses and more high-entropy tokens than incorrect ones, consistent with the need for extended exploratory reasoning. Together, these results establish a clear entropy–difficulty interaction: exploration should be suppressed for easy tasks but encouraged for hard ones.

These results establish a clear entropy–difficulty interaction: on easy problems, suppressing exploration reduces verbosity and improves accuracy, while on hard problems, encouraging additional high-entropy reasoning is crucial for success. This motivates our difficulty-aware exploration strategy, where entropy-based triggers are adaptively modulated across difficulty buckets to balance performance and reasoning efficiency. As further justified in Appendix J, we establish theoretically that response length grows approximately linearly with the number of high-entropy tokens, validating entropy as a principled proxy for reasoning effort.

- 3 METHOD

We aim to endow a multimodal policy with the ability to adapt its reasoning depth: produce short answers for easy instances and longer, exploratory chains for hard ones. Our ARES method consists of two stages: a cold-start curriculum (AdaCS) that preserves length-controllable modes, followed by a KL-regularized RL stage (AEPO) that couples a token-level high-entropy window detector with a difficulty-aware KL budget, which is illustrated in Figure 5.

##### (a) Difficulty modulates the exploratory effort along the reasoning path

##### (b) Quantitative analysis across Easy and Hard

Above threshold Avg. Acc Below threshold Avg. Acc

[Figure 3]

6000

5280.0

1.2

0.991

5000

0.939

ResponseLength

- 0.8
- 1

4000

Accuracy

2871.3

3000

0.6

2000

0.4

0.2

1000

0.098

0.067 0

0

Easy Hard

Easy Hard

(i) Accuracy by entropy threshold (ii) Response length by difficulty

Easy Medium

acc=0 acc=1

acc=0 acc=1

425.0

450

7000

High-EntropyTokenCount

376.2

5878.2

400

6000

5300.7

350

ResponseLength

297.2

4587.2

Hard

5000

300

4000

250

188.0

200

2704.5

3000

150

Easy: Hard:

Medium:

2000

100

Q: Prove that the sum of two odd numbers is always even?

Q: What is 7 + 5 (mod 10)?

1000

50

Q: What is 7 + 5 ? A: Let’s add 7 to 5 ... the answer is 12.

0

0

A: Let’s write an odd number a

A: Firstly, I wi -ll calculate 7+ 5 ... , but we need to take the remainder, so the answer is 2.

Easy Hard

Easy Hard

- -s 𝟐𝒌 + 𝟏. Suppose we add two
- -numbers ... Hmm, the result lo o-ks like 𝟐 𝒌 + 𝒎 + 𝟏 …Wait, let me check again ... Therefore, the result is divisible by 2…

(iii) High-entropy token count by correctness

(iv) Response length by correctness

- Figure 4: Entropy–difficulty interaction in exploration. (a) Conceptual illustration: task difficulty modulates the reasoning trajectory, with easy problems requiring little exploration and hard problems benefiting from deeper branching. (b) Quantitative analysis: (i) for easy tasks, responses below the entropy threshold are both shorter and more accurate; for hard tasks, above-threshold exploration yields higher accuracy; (ii) response length increases significantly with difficulty; (iii) within each difficulty, correct cases use fewer high-entropy tokens for easy problems but more for hard problems; and (iv) correctness further amplifies this trend in response length. Together, these results show that limiting exploration improves efficiency on easy problems, while encouraging additional exploration is crucial for solving difficult ones.

Table 1: Textual and multimodal reasoning datasets source of ARES cold-start data.

Multimodal Text-only

Source Samples Source Samples Source Samples Source Samples FigureQA (Kahou et al., 2017) 100K Super-CLEVR (Li et al., 2023) 30K Big-Math-RL (Albalak et al., 2025) 251K GAIR LIMO (Ye et al., 2025) 0.8K MAVIS (Zhang et al., 2024c) 218K TabMWP (Lu et al., 2022) 38K Big-Math-RL-U (Albalak et al., 2025) 35K s1K-1.1 (Muennighoff et al., 2025) 1K K12-Vista (Li et al., 2025) 33K MMK12 Meng et al. (2025b) 15K AM-Thinking-v1 (Ji et al., 2025) 1,890K MegaScience (Fan et al., 2025) 1,250K GeoQA (Chen et al., 2021) 5K UniGeo (Chen et al., 2022) 16K OpenThoughts (Guha et al., 2025) 114K OpenMathR (Moshkov et al., 2025) 3,200K Geometry3K (Lu et al., 2021a) 2.1K MultiMath (Peng et al., 2024) 300K DeepMath (He et al., 2025) 103K OrcaMath (Mitra et al., 2024) 200K IconQA (Lu et al., 2021b) 107K Grammer (Chen et al., 2025c) 30K OpenR1-220k (Face, 2025) 220K NuminaMath-CoT (Li et al., 2024b) 859K BMMR (Xi et al., 2025) 89K ViRL (Wang et al., 2025a) 39K

- 3.1 ADACS: ADAPTIVE COLDSTART FINE-TUNING

Inspired by Wang et al. (2025e), during the cold-start phase, training with simple problems paired with short chains of thought (CoT) and difficult problems paired with long CoTs effectively enhances the model’s difficulty awareness and facilitates the acquisition of key high-window-entropy (HWE) tokens as well as reflective reasoning capabilities. We first curate a high-quality set of RLVR textual data and multimodal STEM tasks, which serves as the foundation for the cold-start stage. More details regarding dataset sources and preprocessing can be found in Section 4.1. Different from prior work (Wang et al., 2025e) that discards samples with a pass rate of 1 and oversamples those with lower pass rates, our methodology is specifically designed to train the model to differentiate problem difficulty and generate responses of corresponding lengths. To achieve this, for each data source, we establish a target response length for every pass rate bracket, which is determined by the median response lengths of the easiest (pass rate = 1) and hardest (pass rate = 0) problems. We then sample responses whose lengths are proximate to this target, while ensuring uniform sampling across all pass rate brackets. This strategy is motivated by the objective of instilling a strong, explicit correlation between perceived problem difficulty and the verbosity of the generated rationale. Formally, for any intermediate pass rate p ∈ (0,1), the target response length is defined as

Ltarget(p) = (1 − p) · L(0) + p · L(1),

Difficulty-Aware Selective Module

[Figure 4]

[Figure 5]

[Figure 6]

Data Collection

- Sys1

- Sys2

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Text

Reasoning Model

[Figure 11]

Data Curation

[Figure 12]

[Figure 13]

Multimodal

Training Stage 1: Adaptive Coldstart Fine-Tuning

[Figure 14]

Dynamic KL KL Budget Allocator

Reference Model

|[Easy]|
|---|

|[Hard]|
|---|

[Figure 15]

[Figure 16]

[Figure 17]

Online Difficulty Bucketing

Policy Model ……

Accuracy Reward

Overall Reward ……

Group Computation ……

|[Med]|
|---|

|query| |
|---|---|
| | |

|[Med]|
|---|

[Figure 18]

Entropy Reward

|[Hard]|
|---|

High Entropy window work as starting point of the reasoning fork

[Figure 19]

Easy

[Figure 20]

[Figure 21]

Medium

[Figure 22]

Hard

Varying difficulty calls for varying thinking depth

Training Stage 2: Adaptive Entropy Policy Optimization(AEPO)

- Figure 5: Overall training pipeline of our method. Stage 1 (Adaptive Coldstart Fine-Tuning): difficulty-aware selective data curation and adaptive KL-guided fine-tuning establish a strong initialization across text and multimodal inputs. Stage 2 (Adaptive Entropy Policy Optimization, AEPO): online difficulty bucketing and entropy-aware rollout allocate reasoning depth dynamically, with high-entropy windows serving as branching points for exploration. Together, the two stages enable uncertainty-aware, difficulty-adaptive reasoning for large language models.

where L(0) and L(1) denote the median token lengths of responses for problems with pass rates of 0 and 1, respectively, for a given data source. This design maximizes response length diversity across pass rates, thereby amplifying the distinction between easy and hard problems and enabling the model to better learn difficulty-aware reasoning strategies.

- 3.2 AEPO: ADAPTIVE-ENTROPY POLICY OPTIMIZATION

To overcome the imbalance between overthinking easy problems and under-exploring difficult ones (Section 2.2), we introduce Adaptive-Entropy Policy Optimization (AEPO). The central idea is to endow the policy with an adaptive exploration mechanism that decides both when to explore and how much to explore, guided by entropy signals. A detailed flowchart of AEPO is presented in Algorithm 1 in the Appendix.

- 3.2.1 WHEN TO EXPLORE.

To operationalize window entropy as an exploration trigger, we introduce a bucket-wise highentropy threshold. Specifically, for each rollout trajectory, we compute the token-level entropies

{Ht}Tt=1 and extract the 95th percentile value as the high-entropy threshold for that sequence. This choice is motivated by recent empirical findings (Wang et al., 2025b), which show that RLVR pre-

dominantly reshapes the entropy distribution of tokens in the top 5% percentile range, while lowentropy tokens remain comparatively stable. Hence, the 95th percentile serves as a robust indicator of reasoning-critical uncertainty. To reduce noise at the single-sequence level, the thresholds are

then averaged across trajectories in a mini-batch to yield a stable batch-level cutoff τhigh:

1 |D| y∈D

Quantile0.95({Ht(y)}|ty=1| ),

τhigh =

where D denotes the set of trajectories in a batch.

During training, this threshold is updated dynamically on a batch-by-batch basis, allowing AEPO to adaptively align the exploration schedule with the evolving entropy distribution. At rollout time, whenever the windowed entropy H¯t:w of a segment exceeds τhigh, the model branches additional trajectories from step t; otherwise, the rollout proceeds linearly. In this way, exploration is triggered only at sustained high-uncertainty regions, concentrating computational resources on reasoningcritical moments while avoiding unnecessary branching at stable, low-entropy segments.

- 3.2.2 HOW MUCH TO EXPLORE.

Motivated from our preliminary studies, we next investigate the principle of how much exploration should be allocated once a reasoning-critical region is detected. While window entropy serves as a reliable trigger for when to explore, the amount of exploration must be carefully controlled to avoid excessive reasoning on simple tasks and insufficient exploration on complex ones. To achieve this balance, AEPO integrates two complementary mechanisms:

Hierarchical Reward Design. The goal of AEPO’s hierarchical reward is to adaptively regulate the degree of exploration once reasoning-critical regions are detected. Building on the online difficulty buckets to split instances into three difficulty levels (Appendix G), we integrate accuracy and entropy-based shaping into a single constraint-driven formulation that requires no additional hyperparameters.

Formally, for a trajectory y with NHE high-entropy tokens, we define the bucket-dependent target as the batch mean:

NHEtarget(d) = Ebatch[ NHE | d ], d ∈ {easy, medium, hard},

which is updated online at each iteration to track the evolving distribution of exploratory behavior. To regulate deviations from this target, we introduce a closed-form Lagrange multiplier derived from batch statistics:

Ebatch[NHE | d] − NHEtarget(d) Varbatch[NHE | d] + ε

λd = max 0,

.

This term automatically scales the strength of entropy shaping without requiring manually tuned weights or learning rates.

We further define ∆(y;d) = NHE − NHEtarget(d) as the deviation of a response’s high-entropy token count from the target value. This deviation serves as the shaping signal that determines whether

exploration should be encouraged (positive ∆ for easy tasks) or suppressed (negative ∆ for hard tasks). The shaping direction is defined as:

 

max(0, ∆), d = easy, |∆|, d = medium, max(0, −∆), d = hard.

gd(∆) =



The qualitative effect of gd(∆) differs across difficulty levels. For easy problems, positive deviations (∆ > 0) correspond to unnecessary over-exploration and are penalized. For medium problems, both under- and over-exploration are discouraged, resulting in a symmetric shaping curve around the target. For hard problems, negative deviations (∆ < 0) indicate insufficient exploration and are penalized, thereby encouraging longer and more exploratory reasoning chains. We provide a detailed visualization and discussion of these curves in Appendix I (Figure 9).

Finally, the overall hierarchical reward is given by:

R(x, y; d) = Racc(x, y) − 1[acc(x, y) = 0] λd gd ∆(y; d) ,

where hierarchical reward R(x,y;d) unifies the correctness term Racc(x,y) with a difficulty-aware entropy regularization. Specifically, the entropy penalty is applied only when the response y is

incorrect, ensuring that already-correct solutions are not discouraged while encouraging exploration on failed attempts.

This formulation establishes a difficulty-aware intrinsic reward that suppresses unnecessary exploration on easy tasks, stabilizes reasoning depth around a batch-adaptive target on medium tasks, and promotes sustained exploration on hard tasks. Importantly, the entire scheme operates in a closedform manner based solely on batch-level statistics, thereby achieving adaptive exploration control without introducing additional hyperparameters.

Dynamic KL Design. As established in Appendix K, the KL loss provides a valid thinking budget, while Appendix E further shows that a KL penalty inflates variance, motivating our exclusive use of the KL loss. To realize adaptive exploration, AEPO employs only a token–adaptive weight mechanism:

 

ρ (< 1), if t ∈ Wvalid, 1, otherwise,

(1)

βi,t = βd · ρt, ρt =



where βd is the difficulty–dependent baseline and ρt relaxes the KL constraint inside validated high–entropy windows. This simple yet effective design ensures that KL divergence is tightly controlled on easy tokens but adaptively relaxed at reasoning-critical segments, functioning as a tokenwise thinking budget allocator.

Then we present the surrogate objective of AEPO, which is adapted from the GRPO and DAPO formulations discussed in Appendix C. The objective function is defined as:

JAEPO(θ) = E(q,a)∼D,{oi}Gi=1∼πθold(·|q)

|oi|

G

1 G

1 |oi|

min ri,t(θ) A˜i,t,

t=1

i=1

clip ri,t(θ), 1 − ϵℓ, 1 + ϵh A ˜i,t − βd(i),t DKL πθ(· | si,t) ∥ πref(· | si,t) ,

(2)

i t|si,t)

where ri,t(θ) = πθ(o

πref(oit|si,t), si,t = (q,oi<t), and A˜i,t is the task+entropy shaped (token-level) advantage induced by the hierarchical reward (Section 3.2.2).

- 4 EXPERIMENTS

- 4.1 EXPERIMENTAL DETAILS

Datasets The training of ARES follows our proposed methodology, utilizing carefully curated datasets for each stage. The initial adaptive cold-start phase employs a dataset of approximately 224K samples, including both high-quality textual data and multimodal STEM tasks (Table 1). We then employ a strong open-source cold-start model, Revisual-R1 (Chen et al., 2025c), to generate eight responses per sample and compute the pass rate. To establish a robust initial policy, textual problems are answered using the powerful DeepSeek-R1 model (DeepSeek-AI et al., 2025), while multimodal reasoning tasks are addressed with Seed-1.5-VL (Guo et al., 2025). The subsequent RLVR stage with AEPO then utilizes the ViRL39K dataset (Wang et al., 2025a), a collection of verifiable question–answer pairs derived from existing multimodal datasets.

Benchmarks To comprehensively evaluate the capabilities of our model, we selected a diverse suite of benchmarks designed to test a wide range of reasoning skills. In the multimodal domain, we assess multimodal mathematical reasoning using MathVerse (Zhang et al., 2024b), MathVision (Wang et al., 2024a), MathVista (Lu et al., 2023), DynaMath (Zou et al., 2025), and WeMath (Qiao et al., 2024). To evaluate broader logical and general-purpose multimodal capabilities, we incorporated LogicVista (Xiao et al., 2024), MMMU (Yue et al., 2024a), MMMU-Pro (Yue et al.,

- 2024b), CharXIV (Wang et al., 2024c), and MMStar Chen et al. (2024). For text-based reasoning, we measured the model’s ability to solve challenging mathematical problems with AIME24/25 (Li

- et al., 2024a) and MATH-500 (Hendrycks et al., 2021), while its general question-answering and reasoning proficiency were tested on GPQA (Rein et al., 2024), MMLU Pro (Wang et al., 2024b), and BBEH (Kazemi et al., 2025). For all evaluations, we report pass@1 accuracy, with the exception of the AIME24/25 benchmark, for which performance is measured using average@16 accuracy.

- Table 2: Performance comparison of various MLLMs on diverse multimodal reasoning benchmarks. Within each model group (3B and 7B), the best results are highlighted in bold, and the second-best are underlined. Scores in italics indicate that they are not reported in the original work and are obtained using the VLMEvalKit (Duan et al., 2025) for evaluation. MathVerse-V, DynaMath-W and WeMath-S denotes the vision-only, worst, and strict settings, respectively.

Multimodal Reasoning Benchmarks

Model MathVerse-V MathVision MathVista DynaMath-W WeMath LogicVista MMMU MMMU-Pro CharXIV MMStar Avg. Close-Source

GPT-4.1 59.8 51.8 72.0 48.3 55.5 63.8 75.0 65.0 55.8 71.2 61.8 Gemini-2.5-Pro-Thinking 81.2 55.3 83.8 57.1 78.0 75.2 82.0 76.5 69.3 79.7 73.8 Claude-4-Sonnet 66.1 54.6 70.4 46.9 63.0 64.4 74.4 60.7 58.4 66.9 62.6 Doubao-1.5-thinking-vision-pro 80.4 68.7 85.6 60.5 78.0 71.8 77.9 67.6 63.4 78.2 73.2

3B-scale MLLMs

Qwen2.5-VL-3B-Instruct 33.0 21.2 62.3 17.0 17.9 35.8 53.1 31.6 25.3 51.1 34.8 FAST-3B 37.2 26.8 66.2 15.4 23.3 35.4 52.0 34.6 28.3 55.2 37.4 VLAA-Thinker-3B 36.4 24.4 61.0 18.2 33.8 38.5 49.7 33.3 28.2 53.4 37.7 ARES-3B 48.2 44.2 66.8 24.6 40.8 43.5 56.8 45.2 34.4 56.7 46.1 ∆ (Ours–Open 3B SoTA) +11.0 +17.4 +0.6 +6.4 +7.0 +5.0 +3.7 +10.6 +6.1 +1.5 +8.4

7B scale Models

Qwen2.5-VL-7B-Instruct 42.9 25.1 68.2 21.2 36.2 45.0 58.6 38.3 35.5 62.1 43.3 OpenVLThinker-1.2-7B 40.7 25.9 72.3 21.2 37.9 41.4 58.7 42.9 39.3 62.9 44.3 MM-Eureka-Qwen-7B 49.6 26.9 73.0 24.0 34.7 46.8 57.3 43.3 39.5 64.4 46.0 MMR1-Math-v0 45.1 30.2 71.0 25.2 33.2 50.8 57.1 43.2 39.3 63.8 45.9 ThinkLite-7B-VL 45.3 32.9 75.1 22.0 26.5 40.7 55.5 41.3 39.3 63.4 44.2 VLAA-Thinker-7B 48.2 26.4 68.0 22.4 29.2 48.5 54.6 41.6 36.1 64.5 44.0 VL-Rethinker-7B 49.1 32.3 74.9 27.4 27.8 44.5 56.7 41.7 39.8 62.7 45.7 Vision-G1 50.0 31.3 76.1 27.2 29.0 50.2 53.4 41.2 41.0 63.1 46.2 ARES-7B 56.5 51.9 74.6 39.7 47.2 54.1 67.9 54.8 47.0 65.3 55.9 ∆ (Ours–7B Open SoTA) +6.5 +19.0 -1.5 +12.3 +9.3 +3.3 +9.2 +11.5 +6.0 +0.8 +9.7

Baselines We benchmark ARES against three categories of baselines. First, we include leading proprietary systems such as GPT-4.1 (OpenAI), Gemini-2.5-Pro-Thinking (Comanici et al.,

- 2025), Claude-4-Sonnet (Anthropic), and Doubao-1.5-Thinking-Vision-Pro (Guo et al., 2025), which serve as upper-bound references for multimodal reasoning. Second, we consider representative lightweight open-source MLLMs, including Qwen2.5-VL-3B-Instruct (Bai et al., 2025), FAST-

- 3B (Xiao et al., 2025), and VLAA-Thinker-3B (Chen et al., 2025a), which are suitable for efficient deployment. Third, we evaluate competitive 7B-scale open-source MLLMs, such as Qwen2.5VL-7B-Instruct, OpenVLThinker-1.2-7B (Deng et al., 2025), MM-Eureka-Qwen-7B (Meng et al., 2025b), MMR1-Math-v0 (Leng et al., 2025), ThinkLite-7B-VL (Wang et al., 2025c), VLAAThinker-7B, VL-Rethinker-7B (Wang et al., 2025a), and Vision-G1 (Zha et al., 2025). These represent widely adopted open-source alternatives that support research reproducibility. Most of the open-source baselines are fine-tuned from Qwen2.5-VL-3B-Instruct or Qwen2.5-VL-7B-Instruct, ensuring comparability in architecture and training setup. Collectively, this suite provides a comprehensive coverage of both proprietary and open-source models across multiple scales.

- 4.2 MAIN RESULTS

As summarized in Tables 2 and 3, MRLMs consistently outperform chat models across complex reasoning, knowledge-intensive, and general-purpose benchmarks. This confirms that training stages such as cold-start and RLVR effectively enhance self-reflection and reasoning ability. However, we also observe a sharp drop in textual reasoning for many open-source MRLMs likely due to reliance on multimodal verifiable data without cold-start fine-tuning.

In contrast, ARES achieves strong gains at both 3B and 7B scales. On multimodal benchmarks (Table 2), ARES-7B exceeds the best open-source models by +19.0 on MathVision and +11.5 on MMMU-Pro. On textual reasoning (Table 3), it attains 61.7 on AIME25, where most 7B baselines score below 3.3. These results confirm that ARES improves core reasoning ability rather than overfitting to multimodal tasks.

Textual Reasoning Benchmarks Model AIME24 AIME25 MATH500 MMLU Pro BBEH GPQA Avg. Close-Source

GPT-4.1 46.7 23.3 89.6 80.2 32.7 67.7 56.7 Gemini-2.5-Pro-Thinking 66.7 88.0 85.8 - 17.7 82.3 68.1 Claude-4-Sonnet 46.7 33.3 92.0 83.2 35.0 68.7 59.8 Doubao-1.5-Thinking-Vision-Pro 80.0 53.3 96.8 83.5 39.7 71.2 70.8

3B-scale MLLMs

Qwen2.5-VL-3B-Instruct 0.0 0.0 55.8 41.5 7.0 29.3 22.3 FAST-3B 6.7 3.3 55.4 34.6 8.6 25.3 22.3 VLAA-Thinker-3B 3.3 0.0 61.4 33.3 8.1 27.8 22.3 ARES-3B 41.5 34.8 86.2 55.8 15.0 41.4 45.8 ∆ (Ours–Open 3B SoTA) +34.8 +31.5 +24.8 +14.3 +6.4 +12.1 +20.7

7B-scale MLLMs

Qwen2.5-VL-7B-Instruct 6.7 0.0 66.0 50.5 10.8 36.9 28.5 OpenVLThinker-1.2-7B 0.0 0.0 63.2 49.3 13.6 31.3 26.2 MM-Eureka-Qwen-7B 13.3 0.0 65.4 53.4 12.2 41.4 31.0 MMR1-Math-v0 3.3 3.3 66.6 51.8 11.4 32.3 28.1 ThinkLite-7B-VL 3.3 3.3 67.6 51.9 11.3 35.9 28.9 VLAA-Thinker-7B 6.7 3.3 68.6 51.2 11.7 28.8 28.4 VL-Rethinker-7B 3.3 3.3 64.4 52.0 12.9 31.8 28.0 Vision-G1 3.3 0.0 69.0 53.0 13.9 31.3 28.4 ARES-7B 65.0 61.7 95.2 67.0 18.9 49.5 59.6 ∆ (Ours–7B Open SoTA) +51.7 +58.4 +26.2 +13.6 +5.0 +8.1 +27.2

- Table 3: Performance on textual reasoning benchmarks. Results on AIME24/25, MATH500, MMLU Pro, BBEH, and GPQA. ARES-3B and ARES-7B substantially outperform all open-source baselines at their respective scales, achieving large average gains (∆ rows) and narrowing the gap to leading proprietary systems.

6000

ARES

0.80

ARES w/o KL

5750

ARES w/o Entropy

0.75

GRPO DAPO

5500

0.70

5250

###### ResponseLength

0.65

Accuracy

5000

0.60

4750

0.55

4500

0.50

ARES

ARES w/o KL

4250

ARES w/o Entropy

0.45

GRPO DAPO

4000

0.40

0 50 100 150 200 250

0 50 100 150 200 250

Steps

Steps

- Figure 6: Training dynamics of accuracy (left) and response length (right). We observe that ARES achieves higher accuracy compared with its ablations (w/o KL, w/o Entropy) and baseline methods (GRPO, DAPO), while also producing shorter and more stable responses during training. These results indicate that the joint use of KL and entropy shaping contributes to improving both correctness and conciseness.

Finally, Table 4 and Figure 6 demonstrate the adaptivity of our training strategy. AdaCS (ARESCS-7B) modulates response length by task difficulty, while AEPO (ARES-RL-7B) further enhances this effect—extending reasoning for hard tasks (e.g., OlympiadBench, AIME25) and shortening it for easier ones (e.g., GSM8K, MathVista). These results, consistent with our entropy–difficulty analysis (Section 2.2), show that AEPO improves both accuracy and token efficiency by encouraging exploration only when needed.

[Figure 23]

- Figure 7: Accuracy comparison across multimodal and textual benchmarks. Accuracy of three model variants (ARES-CS-Vanilla, ARES-CS-7B, and ARES-RL-7B) on six benchmarks. Results are grouped into three difficulty levels (Easy, Medium, Hard). RL fine-tuning consistently improves accuracy across all benchmarks.

- 5 VISUALIZATION RESULTS

In this subsection, we provide additional visualization results to complement the quantitative findings in the main text. Figures 7 and 8 illustrate the accuracy and response length dynamics across different benchmarks and difficulty levels.

As shown in Figure 7, RL fine-tuning (ARES-RL-7B) consistently improves accuracy over both cold-start variants (ARES-CS-Vanilla and ARES-CS-7B) on all benchmarks. The gains are evident across easy, medium, and hard categories, demonstrating that our proposed reinforcement learning approach enhances reasoning robustness without sacrificing performance on simpler tasks. This confirms the effectiveness of incorporating entropy-aware objectives in stabilizing training and improving generalization.

- Figure 8 further analyzes the average response length. We observe that RL fine-tuning reduces response length on easier datasets (e.g., GSM8K, MathVista), reflecting improved reasoning efficiency by eliminating unnecessary verbose reasoning. On more challenging benchmarks (e.g., OlympiadBench, AIME25), the response length of ARES-RL-7B increases compared to coldstart models, indicating that the model adaptively allocates more reasoning steps when required. This adaptive behavior—shortening trivial reasoning while allowing deeper exploration on difficult tasks—highlights the benefit of our entropy-shaped reward design.

Together, these visualizations provide clear evidence that our RL-based framework not only improves accuracy but also adapts reasoning length to problem difficulty, thereby balancing efficiency and effectiveness in multimodal reasoning tasks.

- 5.1 ABLATION STUDIES

To validate the contributions of the key components within the ARES framework, we conduct a series of ablation studies. We systematically isolate and evaluate the effects of our Hierarchical Reward Design (entropy shaping), and the Dynamic KL Design in our ARES.

- 5.1.1 EFFECTIVENESS OF HIERARCHICAL REWARD DESIGN

To verify the impact of our hierarchical entropy-based reward, we conduct an experiment comparing a model trained with only this component against the GRPO baseline. As shown in Table 5,

- Table 4: Accuracy and response length comparison across multimodal and textual benchmarks. We report both accuracy (Acc) and average response length (Len) for three model variants (ARES-CS-Vanilla, ARES-CS-7B, and ARES-RL-7B) on six benchmarks. The ARES-CS-Vanilla variant is trained using uniform sampling across all pass rates, disregarding the pass rate-to-length sampling strategy detailed in Section 3.1. Visualization of these results is provided in Figure 7 (accuracy) and Figure 8 (response length).

Multimodal Textual

Model OlympiadBench MathVerse-V MathVista AIME25 MATH500 GSM8K Acc Len Acc Len Acc Len Acc Len Acc Len Acc Len

ARES-CS-Vanilla 54.2 9123.2 50.5 3432.1 72.1 1853.3 55.2 16361.6 91.4 2750.1 92.7 358.8 ARES-CS-7B 53.4 10281.5 52.8 3575.4 72.3 1712.3 55.0 17895.0 93.4 3011.8 92.3 332.3 ARES-RL-7B 56.9 12893.4 56.5 3198.3 74.6 1494.5 61.7 22618.8 95.2 2257.7 95.7 278.9

- Table 5: Ablation study of Dynamic KL Loss and Entropy Reward. Building upon our Cold Start stage. Best results per column are bold and second-best are underlined.

Ablation Setting MathVerse-V MathVision MathVista DynaMath-W Avg Cold start only 52.8 49.7 72.9 27.4 50.7 Cold Start + GRPO (baseline) 53.7 50.4 73.3 36.1 53.4 Cold Start + Dynamic KL 55.2 (+2.4) 51.7 (+2.0) 73.7 (+0.8) 38.3 (+10.9) 54.7 (+4.0) Cold Start + Entropy Shaping 55.9 (+3.1) 51.3 (+1.6) 74.4 (+1.5) 39.3 (+11.9) 55.2 (+4.5) ARES-7B 56.5 (+3.7) 51.9 (+2.2) 74.6 (+1.7) 39.7 (+12.3) 55.7 (+5.0)

this component alone yields a significant accuracy improvement of +1.8 points, demonstrating that better depth and valid exploration lead to better performance. Additionally, this adaptive regulation is clearly visible in Figure 6. While the accuracy for this model (labeled “ARES w/o KL”) consistently tracks above the GRPO baseline, its response length shows a marked and steady decrease. This shows the model is learning to curtail unnecessary reasoning (i.e., overthinking) on simpler problems. The strong performance gains on difficult benchmarks like DynaMath-W (+3.2 points) further prove it correctly encourages deeper exploration when necessary. This evidence confirms the hierarchical reward’s primary role as an adaptive regulator of reasoning depth.

- 5.1.2 EFFECTIVENESS OF DYNAMIC KL DESIGN

To verify that the dynamic KL mechanism allocates the exploration budget efficiently, we analyze its isolated contribution. The results in Table 5 show that adding only the dynamic KL component improves average accuracy by +1.3 points over the GRPO baseline. Its role as an efficient budget allocator is best illustrated in Figure 6. The full ARES model’s accuracy curve ultimately converges to a higher final accuracy than all other variants. Concurrently, its response length curve shows the most pronounced reduction, indicating that the token-wise budget allocation pushes the model to find more efficient reasoning paths. This outcome validates that the dynamic KL design successfully allocates the thinking budget, guiding the policy toward greater reasoning efficiency and higher accuracy.

Finally, combining the hierarchical reward with the dynamic KL mechanism results in the full ARES framework. As shown in Table 5, this complete model achieves the highest average accuracy of all configurations (55.7). This result is corroborated by Figure 6, where we see the full ARES model not only converges to the highest final accuracy but also achieves the most significant reduction in response length. This demonstrates the synergistic effect of our two components: by simultaneously regulating reasoning depth and efficiently allocating the exploration budget, ARES achieves both stronger accuracy and more efficient training.

[Figure 24]

Figure 8: Response length comparison across multimodal and textual benchmarks. We report the average number of generated tokens for three model variants (ARES-CS-Vanilla, ARES-CS-7B, and ARES-RL-7B). RL training consistently reduces response length on most benchmarks, indicating improved reasoning efficiency. In contrast, response length increases on the most challenging datasets—AIME25 (textual) and OlympiadBench (multimodal)—highlighting the adaptive behavior of our RL approach: trimming unnecessary reasoning on easy problems while encouraging deeper exploration on difficult ones.

- 6 RELATED WORK

- 6.1 MULTIMODAL LARGE REASONING MODELS Reasoning is widely recognized as a foundational element of intelligent behavior (de Winter et al.,

- 2024; Xu et al., 2025a; Huang et al., 2025c; Bi et al., 2025). To meet the demands of realistic multimodal environments, Multimodal Large Reasoning Models (MLRMs) fuse information from heterogeneous modalities to support deeper thinking. Extending chain-of-thought (CoT) fine-tuning to instill stepwise reasoning, works such as LLaVA-CoT (Xu et al., 2025b) and LlamaV-o1 (Thawakar

et al., 2025) transpose this paradigm to the multimodal setting. Following the success of DeepSeekR1 (DeepSeek-AI et al., 2025), a growing body of research leverages reinforcement learning (RL) to extend long-horizon reasoning in MLRMs: MM-EUREKA (Meng et al., 2025a), R1-V (Chen et al., 2025b), and LMM-R1 (Peng et al., 2025) apply RL to verifiable mathematical geometry and report gains in depth, coherence, and domain robustness via exploration-driven optimization of multi-step solution trajectories. Complementary efforts, including VLM-R1 (Shen et al., 2025a), Visual-RFT (Liu et al., 2025b), and Seg-Zero (Liu et al., 2025a), likewise employ RL to strengthen core visual competencies in MLRMs (e.g., grounding, detection, and understanding). Against this backdrop, our goal is to build an adaptive MLRM that can decide how much reasoning effort to allocate at inference time.

- 6.2 ADAPTIVE REASONING

Current large reasoning models such as OpenAI o1 (OpenAI et al., 2024) and DeepSeek R1 (DeepSeek-AI et al., 2025) show notable reasoning capabilities while often generate excessively lengthy reasoning for trivial questions while providing insufficient exploration for more challenging ones—essentially overthinking simple problems and underthinking complex ones (Zhu & Li,

- 2025; Alomrani et al., 2025; Yue et al., 2025; Wang et al., 2025d; Gao et al., 2024). To address this limitation, numerous studies have sought to establish adaptive reasoning mechanisms in text-based reasoning tasks to enhance efficiency. One line of research develops training-free approaches based on prompt engineering, such as explicitly imposing token budgets (Han et al., 2025) or employing budget-forcing mechanisms that regulate reasoning by either forcefully truncating or artificially

extending the generation process (e.g., by appending repeated “Wait” tokens) (Muennighoff et al., 2025). Instead of fixed budgets, other training-free methods adopt dynamic early-exiting strategies, terminating the reasoning process once the model demonstrates sufficient confidence in its answer (Jiang et al., 2025; Yang et al., 2025a; Yong et al., 2025). Complementary to these approaches, training-based methods aim to instill more strategic adaptive reasoning. Some studies curate variable-length datasets containing both concise and elaborate reasoning chains for supervised fine-tuning (SFT) (Zhang et al., 2025; Yu et al., 2025a; Ma et al., 2025), while others leverage reinforcement learning (RL) to encourage dynamic control over reasoning depth. For example, certain works integrate length penalties into the RL framework (Arora & Zanette, 2025; Ling et al., 2025; Tu et al., 2025), whereas others employ difficulty-awareness to promote succinct responses for simple questions and encourage more detailed reasoning for complex ones, thereby improving both efficiency and accuracy (Huang et al., 2025a; Shen et al., 2025b; Chen et al., 2025d; Xiang et al., 2025). However, much like reasoning models that lack efficiency penalties, these approaches often over-encourage exploration on hard problems, producing unnecessarily verbose traces with limited gains. And most work still focuses on text-only reasoning, adaptive reasoning in multimodal settings remains largely unexplored. In this work, we take an initial step toward a multimodal adaptive reasoning framework.

- 6.3 ENTROPY IN REINFORCEMENT LEARNING

Rooted in information theory, entropy has long served to balance exploration and exploitation in Reinforcement Learning (RL), from early entropy bonuses and maximum-entropy formulations to widespread adoption in deep RL (Ziebart et al., 2008; Toussaint, 2009; Mnih et al., 2015; Schulman et al., 2018). In the context of Large Language Models (LLMs), however, evidence for a global entropy bonus is mixed, and several reports find muted gains (Klein et al., 2024; Shen, 2025), thereby prompting alternatives such as reward reshaping or clipping updates that over-collapse entropy (Zhang et al., 2024a; Cui et al., 2025). In this paper, we treat entropy as a localized signal to encourage concise solutions for easy items and sustained exploration on hard ones, aligning “when” and “how much” to explore with task difficulty.

- 7 CONCLUSION

This paper introduces ARES, a framework designed to cultivate adaptive reasoning in MLRMs, addressing their tendency to overthink simple problems while under-exploring complex ones. ARES uses a two-stage training curriculum: an Adaptive Cold-Start phase instills difficulty awareness, followed by our novel Adaptive-Entropy Policy Optimization (AEPO). AEPO leverages high windowentropy to determine when to explore and a hierarchical reward to control how much reasoning depth is applied. Experiments confirm that ARES achieves superior performance and significantly improves reasoning efficiency, validating our adaptive, entropy-guided approach.

REFERENCES

Alon Albalak, Duy Phung, Nathan Lile, Rafael Rafailov, Kanishk Gandhi, Louis Castricato, Anikait Singh, Chase Blagden, Violet Xiang, Dakota Mahan, et al. Big-math: A large-scale, high-quality math dataset for reinforcement learning in language models. arXiv preprint arXiv:2502.17387, 2025.

Mohammad Ali Alomrani, Yingxue Zhang, Derek Li, Qianyi Sun, Soumyasundar Pal, Zhanguang Zhang, Yaochen Hu, Rohan Deepak Ajwani, Antonios Valkanas, Raika Karimi, Peng Cheng, Yunzhou Wang, Pengyi Liao, Hanrui Huang, Bin Wang, Jianye Hao, and Mark Coates. Reasoning on a budget: A survey of adaptive and controllable test-time compute in llms, 2025. URL https: //arxiv.org/abs/2507.02076.

Anthropic. Introducing claude 4. URL https://www.anthropic.com/news/claude-4. Daman Arora and Andrea Zanette. Training language models to reason efficiently, 2025. URL

https://arxiv.org/abs/2502.04463.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Jing Bi, Susan Liang, Xiaofei Zhou, Pinxin Liu, Junjia Guo, Yunlong Tang, Luchuan Song, Chao Huang, Guangyu Sun, Jinxi He, Jiarui Wu, Shu Yang, Daoan Zhang, Chen Chen, Lianggong Bruce Wen, Zhang Liu, Jiebo Luo, and Chenliang Xu. Why reasoning matters? a survey of advancements in multimodal reasoning (v1), 2025. URL https://arxiv.org/abs/2504. 03151.

Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. Sft or rl? an early investigation into training r1-like reasoning large vision-language models. arXiv preprint arXiv:2504.11468, 2025a.

Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. arXiv preprint arXiv:2105.14517, 2021.

Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, and Xiaodan Liang. Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression. arXiv preprint arXiv:2212.02746, 2022.

Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, and Vinci. R1-v: Reinforcing super generalization ability in vision-language models with less than $3. https://github.com/Deep-Agent/ R1-V, 2025b. Accessed: 2025-02-02.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024.

Shuang Chen, Yue Guo, Zhaochen Su, Yafu Li, Yulun Wu, Jiacheng Chen, Jiayu Chen, Weijie Wang, Xiaoye Qu, and Yu Cheng. Advancing multimodal reasoning: From optimized cold start to staged reinforcement learning. arXiv preprint arXiv:2506.04207, 2025c.

Weize Chen, Jiarui Yuan, Tailin Jin, Ning Ding, Huimin Chen, Zhiyuan Liu, and Maosong Sun. The overthinker’s diet: Cutting token calories with difficulty-aware training, 2025d. URL https: //arxiv.org/abs/2505.19217.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit S. Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris, Sam Petulla, Colin Gaffney, Asaf Aharoni, Nathan Lintz, Tiago Cardal Pais, Henrik Jacobsson, Idan Szpektor, Nan-Jiang Jiang, Krishna Haridasan, Ahmed Omran, Nikunj Saunshi, Dara Bahri, Gaurav Mishra, Eric Chu, Toby Boyd, Brad Hekman, Aaron Parisi, Chaoyi Zhang, Kornraphop Kawintiranon, Tania Bedrax-Weiss, Oliver Wang, Ya Xu, Ollie Purkiss, Uri Mendlovic, Ila¨ı Deutel, Nam Nguyen, Adam Langley, Flip Korn, Lucia Rossazza, Alexandre Ram´e, Sagar Waghmare, Helen Miller, Nathan Byrd, Ashrith Sheshan, Raia Hadsell Sangnie Bhardwaj, Pawel Janus, Tero Rissa, Dan Horgan, Sharon Silver, Ayzaan Wahid, Sergey Brin, Yves Raimond, Klemen Kloboves, Cindy Wang, Nitesh Bharadwaj Gundavarapu, Ilia Shumailov, Bo Wang, Mantas Pajarskas, Joe Heyward, Martin Nikoltchev, Maciej Kula, Hao Zhou, Zachary Garrett, Sushant Kafle, Sercan Arik, Ankita Goel, Mingyao Yang, Jiho Park, Koji Kojima, Parsa Mahmoudieh, Koray Kavukcuoglu, Grace Chen, Doug Fritz, Anton Bulyenov, Sudeshna Roy, Dimitris Paparas, Hadar Shemtov, Bo-Juen Chen, Robin Strudel, David Reitter, Aurko Roy, Andrey Vlasov, Changwan Ryu, Chas Leichner, Haichuan Yang, Zelda Mariet, Denis Vnukov, Tim Sohn, Amy Stuart, Wei Liang, Minmin Chen, Praynaa Rawlani, Christy Koh, JD CoReyes, Guangda Lai, Praseem Banzal, Dimitrios Vytiniotis, Jieru Mei, and Mu Cai. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. CoRR, abs/2507.06261, 2025. doi: 10.48550/ARXIV.2507.06261. URL https://doi.org/10.48550/arXiv.2507.06261.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, and Ning Ding. The entropy mechanism of reinforcement learning for reasoning language models, 2025. URL https://arxiv.org/abs/2505.22617.

Joost C. F. de Winter, Dimitra Dodou, and Yke Bauke Eisma. System 2 thinking in openai’s o1-preview model: Near-perfect performance on a mathematics exam. Computers, 13(11): 278, October 2024. ISSN 2073-431X. doi: 10.3390/computers13110278. URL http: //dx.doi.org/10.3390/computers13110278.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, and S. S. Li. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. CoRR, abs/2501.12948, 2025. doi: 10. 48550/ARXIV.2501.12948. URL https://doi.org/10.48550/arXiv.2501.12948.

Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement, 2025. URL https://arxiv.org/abs/2503.17352.

Haodong Duan, Xinyu Fang, Junming Yang, Xiangyu Zhao, Yuxuan Qiao, Mo Li, Amit Agarwal, Zhe Chen, Lin Chen, Yuan Liu, Yubo Ma, Hailong Sun, Yifan Zhang, Shiyin Lu, Tack Hwa Wong, Weiyun Wang, Peiheng Zhou, Xiaozhe Li, Chaoyou Fu, Junbo Cui, Jixuan Chen, Enxin Song, Song Mao, Shengyuan Ding, Tianhao Liang, Zicheng Zhang, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, Dahua Lin, and Kai Chen. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models, 2025. URL https://arxiv.org/abs/2407.11691.

Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL https: //github.com/huggingface/open-r1.

Run-Ze Fan, Zengzhi Wang, and Pengfei Liu. Megascience: Pushing the frontiers of posttraining datasets for science reasoning. arXiv preprint arXiv:2507.16812, 2025. URL https: //arxiv.org/abs/2507.16812.

Minghe Gao, Shuang Chen, Liang Pang, Yuan Yao, Jisheng Dang, Wenqiao Zhang, Juncheng Li, Siliang Tang, Yueting Zhuang, and Tat-Seng Chua. Fact :teaching mllms with faithful, concise and transferable rationales, 2024. URL https://arxiv.org/abs/2404.11129.

Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, et al. Openthoughts: Data recipes for reasoning models. arXiv preprint arXiv:2506.04178, 2025.

Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, Jingji Chen, Jingjia Huang, Kang Lei, Liping Yuan, Lishu Luo, Pengfei Liu, Qinghao Ye, Rui Qian, Shen Yan, Shixiong Zhao, Shuai Peng, Shuangye Li, Sihang Yuan, Sijin Wu, Tianheng Cheng, Weiwei Liu, Wenqian Wang, Xianhan Zeng, Xiao Liu, Xiaobo Qin, Xiaohan Ding, Xiaojun Xiao, Xiaoying Zhang, Xuanwei Zhang, Xuehan Xiong, Yanghua Peng, Yangrui Chen, Yanwei Li, Yanxu Hu, Yi Lin, Yiyuan Hu, Yiyuan Zhang, Youbin Wu, Yu Li, Yudong Liu, Yue Ling, Yujia Qin, Zanbo Wang, Zhiwu He, Aoxue Zhang, Bairen Yi, Bencheng Liao, Can Huang, Can Zhang, Chaorui Deng, Chaoyi Deng, Cheng Lin, Cheng Yuan, Chenggang Li, Chenhui Gou, Chenwei Lou, Chengzhi Wei, Chundian Liu, Chunyuan Li, Deyao Zhu, Donghong Zhong, Feng Li, Feng Zhang, Gang Wu, Guodong Li, Guohong Xiao, Haibin Lin, Haihua Yang, Haoming Wang, Heng Ji, Hongxiang Hao, Hui Shen, Huixia Li, Jiahao Li, Jialong Wu, Jianhua Zhu, Jianpeng Jiao, Jiashi Feng, Jiaze Chen, Jianhui Duan, Jihao Liu, Jin Zeng, Jingqun Tang, Jingyu Sun, Joya Chen, Jun Long, Junda Feng, Junfeng Zhan,

Junjie Fang, Junting Lu, Kai Hua, Kai Liu, Kai Shen, Kaiyuan Zhang, and Ke Shen. Seed1.5vl technical report. CoRR, abs/2505.07062, 2025. doi: 10.48550/ARXIV.2505.07062. URL https://doi.org/10.48550/arXiv.2505.07062.

Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen. Tokenbudget-aware llm reasoning, 2025. URL https://arxiv.org/abs/2412.18547.

Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, et al. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. arXiv preprint arXiv:2504.11456, 2025.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Shijue Huang, Hongru Wang, Wanjun Zhong, Zhaochen Su, Jiazhan Feng, Bowen Cao, and Yi R. Fung. Adactrl: Towards adaptive and controllable reasoning via difficulty-aware budgeting, 2025a. URL https://arxiv.org/abs/2505.18822.

Shijue Huang, Hongru Wang, Wanjun Zhong, Zhaochen Su, Jiazhan Feng, Bowen Cao, and Yi R. (May) Fung. Adactrl: Towards adaptive and controllable reasoning via difficulty-aware budgeting. CoRR, abs/2505.18822, 2025b. doi: 10.48550/ARXIV.2505.18822. URL https: //doi.org/10.48550/arXiv.2505.18822.

Shijue Huang, Wanjun Zhong, Deng Cai, Fanqi Wan, Chengyi Wang, Mingxuan Wang, Mu Qiao, and Ruifeng Xu. Empowering self-learning of llms: inner knowledge explicitation as a catalyst. In Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence and ThirtySeventh Conference on Innovative Applications of Artificial Intelligence and Fifteenth Symposium on Educational Advances in Artificial Intelligence, AAAI’25/IAAI’25/EAAI’25. AAAI Press, 2025c. ISBN 978-1-57735-897-8. doi: 10.1609/aaai.v39i23.34590. URL https: //doi.org/10.1609/aaai.v39i23.34590.

Yunjie Ji, Xiaoyu Tian, Sitong Zhao, Haotian Wang, Shuaiting Chen, Yiping Peng, Han Zhao, and Xiangang Li. Am-thinking-v1: Advancing the frontier of reasoning at 32b scale. CoRR, abs/2505.08311, 2025. doi: 10.48550/ARXIV.2505.08311. URL https://doi.org/10. 48550/arXiv.2505.08311.

Guochao Jiang, Guofeng Quan, Zepeng Ding, Ziqin Luo, Dixuan Wang, and Zheng Hu. Flashthink: An early exit method for efficient reasoning, 2025. URL https://arxiv.org/abs/2505. 13949.

Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Akos´ K´ad´ar, Adam Trischler, and Yoshua Bengio. Figureqa: An annotated figure dataset for visual reasoning. arXiv preprint arXiv:1710.07300, 2017.

Mehran Kazemi, Bahare Fatemi, Hritik Bansal, John Palowitch, Chrysovalantis Anastasiou, Sanket Vaibhav Mehta, Lalit K Jain, Virginia Aglietti, Disha Jindal, Peter Chen, et al. Big-bench extra hard. arXiv preprint arXiv:2502.19187, 2025.

Sara Klein, Simon Weissmann, and Leif D¨oring. Beyond stationarity: Convergence analysis of stochastic softmax policy gradient methods. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=1VeQ6VBbev.

Sicong Leng, Jing Wang, Jiaxi Li, Hao Zhang, Zhiqiang Hu, Boqiang Zhang, Hang Zhang, Yuming Jiang, Xin Li, Deli Zhao, Fan Wang, Yu Rong, Aixin Sun, and Shijian Lu. Mmr1: Advancing the frontiers of multimodal reasoning. https://github.com/LengSicong/MMR1, 2025.

Chong Li, Chenglin Zhu, Tao Zhang, Mingan Lin, Zenan Zhou, and Jian Xie. K12vista: Exploring the boundaries of mllms in K-12 education. CoRR, abs/2506.01676, 2025. doi: 10.48550/ARXIV. 2506.01676. URL https://doi.org/10.48550/arXiv.2506.01676.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q. Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. https: //huggingface.co/datasets/Numinamath, 2024a. Hugging Face repository, 13:9.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13(9):9, 2024b.

Zhuowan Li, Xingrui Wang, Elias Stengel-Eskin, Adam Kortylewski, Wufei Ma, Benjamin Van Durme, and Alan L Yuille. Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14963–14973, 2023.

Zehui Ling, Deshu Chen, Hongwei Zhang, Yifeng Jiao, Xin Guo, and Yuan Cheng. Fast on the easy, deep on the hard: Efficient reasoning via powered length penalty, 2025. URL https: //arxiv.org/abs/2506.10446.

Yuqi Liu, Bohao Peng, Zhisheng Zhong, Zihao Yue, Fanbin Lu, Bei Yu, and Jiaya Jia. Segzero: Reasoning-chain guided segmentation via cognitive reinforcement, 2025a. URL https: //arxiv.org/abs/2503.06520.

Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning, 2025b. URL https://arxiv.org/ abs/2503.01785.

Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv preprint arXiv:2105.04165, 2021a.

Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. arXiv preprint arXiv:2110.13214, 2021b.

Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. arXiv preprint arXiv:2209.14610, 2022.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

Xinyin Ma, Guangnian Wan, Runpeng Yu, Gongfan Fang, and Xinchao Wang. Cot-valve: Lengthcompressible chain-of-thought tuning, 2025. URL https://arxiv.org/abs/2502. 09601.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, Ping Luo, Yu Qiao, Qiaosheng Zhang, and Wenqi Shao. Mm-eureka: Exploring the frontiers of multimodal reasoning with rulebased reinforcement learning, 2025a. URL https://arxiv.org/abs/2503.07365.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, Ping Luo, Yu Qiao, Qiaosheng Zhang, and Wenqi Shao. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. CoRR, abs/2503.07365, 2025b. doi: 10.48550/ARXIV.2503.07365. URL https://doi.org/10.48550/arXiv.2503.07365.

Arindam Mitra, Hamed Khanpour, Corby Rosset, and Ahmed Awadallah. Orca-math: Unlocking the potential of slms in grade school math. arXiv preprint arXiv:2402.14830, 2024.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, February 2015. ISSN 00280836. URL http://dx.doi.org/10.1038/nature14236.

Ivan Moshkov, Darragh Hanley, Ivan Sorokin, Shubham Toshniwal, Christof Henkel, Benedikt Schifferer, Wei Du, and Igor Gitman. Aimo-2 winning solution: Building state-of-the-art mathematical reasoning models with openmathreasoning dataset. arXiv preprint arXiv:2504.16891, 2025.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Cand`es, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.

OpenAI. Introducing gpt-4.1 in the api. URL https://openai.com/index/gpt-4-1/.

OpenAI, :, Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, Ilge Akkaya, Ilya Kostrikov, Ilya Sutskever, Irina Kofman, Jakub Pachocki, James Lennon, Jason Wei, Jean Harb, Jerry Twore, Jiacheng Feng, Jiahui Yu, Jiayi Weng, Jie Tang, Jieqi Yu, Joaquin Qui˜nonero Candela, Joe Palermo, Joel Parish, Johannes Heidecke, John Hallman, John Rizzo, Jonathan Gordon, Jonathan Uesato, Jonathan Ward, Joost Huizinga, Julie Wang, Kai Chen, Kai Xiao, Karan Singhal, Karina Nguyen, Karl Cobbe, Katy Shi, Kayla Wood, Kendra Rimbach, Keren Gu-Lemberg, Kevin Liu, Kevin Lu, Kevin Stone, Kevin Yu, Lama Ahmad, Lauren Yang, Leo Liu, Leon Maksin, Leyton Ho, Liam Fedus, Lilian Weng, Linden Li, Lindsay McCallum, Lindsey Held, Lorenz Kuhn, Lukas Kondraciuk, Lukasz Kaiser, Luke Metz, Madelaine Boyd, Maja Trebacz, Manas Joglekar, Mark Chen, Marko Tintor, Mason Meyer, Matt Jones, Matt Kaufer, Max Schwarzer, Meghan Shah, Mehmet Yatbaz, Melody Y. Guan, Mengyuan Xu, Mengyuan Yan, Mia Glaese, Mianna Chen, Michael Lampe, Michael Malek, Michele Wang, Michelle Fradin, Mike McClay, Mikhail Pavlov, Miles Wang, Mingxuan Wang, Mira Murati, Mo Bavarian, Mostafa Rohaninejad, Nat McAleese, Neil Chowdhury, Neil Chowdhury, Nick Ryder, Nikolas Tezak, Noam Brown, Ofir Nachum, Oleg Boiko, Oleg Murk, Olivia Watkins, Patrick Chao, Paul Ashbourne, Pavel Izmailov, Peter Zhokhov, Rachel Dias, Rahul Arora, Randall Lin, Rapha Gontijo Lopes, Raz Gaon, Reah Miyara, Reimar Leike, Renny Hwang, Rhythm Garg, Robin Brown, Roshan James, Rui Shu, Ryan Cheu, Ryan Greene, Saachi Jain, Sam Altman, Sam Toizer, Sam Toyer, Samuel Miserendino, Sandhini Agarwal, Santiago Hernandez, Sasha Baker, Scott McKinney, Scottie Yan, Shengjia Zhao, Shengli Hu, Shibani Santurkar, Shraman Ray Chaudhuri, Shuyuan Zhang, Siyuan Fu, Spencer Papay, Steph Lin, Suchir Balaji, Suvansh Sanjeev, Szymon Sidor, Tal Broda, Aidan Clark, Tao Wang, Taylor Gordon, Ted Sanders, Tejal Patwardhan, Thibault Sottiaux, Thomas Degry, Thomas Dimson, Tianhao Zheng, Timur Garipov, Tom Stasi, Trapit Bansal, Trevor Creech, Troy Peterson, Tyna Eloundou, Valerie Qi, Vineet Kosaraju, Vinnie Monaco, Vitchyr Pong, Vlad Fomenko, Weiyi Zheng, Wenda Zhou, Wes McCabe, Wojciech Zaremba, Yann Dubois, Yinghai Lu, Yining Chen, Young Cha, Yu Bai, Yuchen He, Yuchen Zhang, Yunyun Wang, Zheng Shao, and Zhuohan Li. Openai o1 system card, 2024. URL https://arxiv.org/abs/2412.16720.

Shuai Peng, Di Fu, Liangcai Gao, Xiuqin Zhong, Hongguang Fu, and Zhi Tang. Multimath: Bridging visual and mathematical reasoning for large language models. arXiv preprint arXiv:2409.00147, 2024.

Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl, 2025. URL https://arxiv.org/abs/2503. 07536.

Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024.

Xiaoye Qu, Yafu Li, Zhaochen Su, Weigao Sun, Jianhao Yan, Dongrui Liu, Ganqu Cui, Daizong Liu, Shuxian Liang, Junxian He, Peng Li, Wei Wei, Jing Shao, Chaochao Lu, Yue Zhang, XianSheng Hua, Bowen Zhou, and Yu Cheng. A survey of efficient reasoning for large reasoning models: Language, multimodality, and beyond. CoRR, abs/2503.21614, 2025. doi: 10.48550/ ARXIV.2503.21614. URL https://doi.org/10.48550/arXiv.2503.21614.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

John Schulman, Xi Chen, and Pieter Abbeel. Equivalence between policy gradients and soft qlearning, 2018. URL https://arxiv.org/abs/1704.06440.

Han Shen. On entropy control in llm-rl algorithms, 2025. URL https://arxiv.org/abs/ 2509.03493.

Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, Ruochen Xu, and Tiancheng Zhao. Vlm-r1: A stable and generalizable r1-style large vision-language model, 2025a. URL https://arxiv.org/ abs/2504.07615.

Yi Shen, Jian Zhang, Jieyun Huang, Shuming Shi, Wenjing Zhang, Jiangze Yan, Ning Wang, Kai Wang, Zhaoxiang Liu, and Shiguo Lian. Dast: Difficulty-adaptive slow-thinking for large reasoning models, 2025b. URL https://arxiv.org/abs/2503.04472.

Omkar Thawakar, Dinura Dissanayake, Ketan More, Ritesh Thawkar, Ahmed Heakl, Noor Ahsan, Yuhao Li, Mohammed Zumri, Jean Lahoud, Rao Muhammad Anwer, Hisham Cholakkal, Ivan Laptev, Mubarak Shah, Fahad Shahbaz Khan, and Salman Khan. Llamav-o1: Rethinking stepby-step visual reasoning in llms, 2025. URL https://arxiv.org/abs/2501.06186.

Marc Toussaint. Robot trajectory optimization using approximate inference. In Proceedings of the 26th Annual International Conference on Machine Learning, ICML ’09, pp. 1049–1056, New York, NY, USA, 2009. Association for Computing Machinery. ISBN 9781605585161. doi: 10. 1145/1553374.1553508. URL https://doi.org/10.1145/1553374.1553508.

Songjun Tu, Jiahao Lin, Qichao Zhang, Xiangyu Tian, Linjing Li, Xiangyuan Lan, and Dongbin Zhao. Learning when to think: Shaping adaptive reasoning in r1-style models via multi-stage rl,

2025. URL https://arxiv.org/abs/2505.10832.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vlrethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025a.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024a.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for LLM reasoning. CoRR, abs/2506.01939, 2025b. doi: 10. 48550/ARXIV.2506.01939. URL https://doi.org/10.48550/arXiv.2506.01939.

Xiyao Wang, Zhengyuan Yang, Chao Feng, Hongjin Lu, Linjie Li, Chung-Ching Lin, Kevin Lin, Furong Huang, and Lijuan Wang. Sota with less: Mcts-guided sample selection for data-efficient visual reasoning self-improvement. arXiv preprint arXiv:2504.07934, 2025c.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multitask language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024b.

Yue Wang, Qiuzhi Liu, Jiahao Xu, Tian Liang, Xingyu Chen, Zhiwei He, Linfeng Song, Dian Yu, Juntao Li, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. Thoughts are all over the place: On the underthinking of o1-like llms, 2025d. URL https://arxiv.org/ abs/2501.18585.

Yunhao Wang, Yuhao Zhang, Tinghao Yu, Can Xu, Feng Zhang, and Fengzong Lian. Adaptive deep reasoning: Triggering deep thinking when needed. CoRR, abs/2505.20101, 2025e. doi: 10. 48550/ARXIV.2505.20101. URL https://doi.org/10.48550/arXiv.2505.20101.

Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems, 37:113569–113697, 2024c.

Zhiheng Xi, Guanyu Li, Yutao Fan, Honglin Guo, Yufang Liu, Xiaoran Fan, Jiaqi Liu, Jingchao Ding, Wangmeng Zuo, Zhenfei Yin, Lei Bai, Tao Ji, Tao Gui, Qi Zhang, Philip Torr, and Xuanjing Huang. BMMR: A large-scale bilingual multimodal multi-discipline reasoning dataset. CoRR, abs/2507.03483, 2025. doi: 10.48550/ARXIV.2507.03483. URL https://doi.org/10. 48550/arXiv.2507.03483.

Violet Xiang, Chase Blagden, Rafael Rafailov, Nathan Lile, Sang Truong, Chelsea Finn, and Nick Haber. Just enough thinking: Efficient reasoning with adaptive length penalties reinforcement learning, 2025. URL https://arxiv.org/abs/2506.05256.

Wenyi Xiao, Leilei Gan, Weilong Dai, Wanggui He, Ziwei Huang, Haoyuan Li, Fangxun Shu, Zhelun Yu, Peng Zhang, Hao Jiang, et al. Fast-slow thinking for large vision-language model reasoning. arXiv preprint arXiv:2504.18458, 2025.

Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts, 2024. URL https://arxiv.org/abs/2407.04973.

Fengli Xu, Qianyue Hao, Zefang Zong, Jingwei Wang, Yunke Zhang, Jingyi Wang, Xiaochong Lan, Jiahui Gong, Tianjian Ouyang, Fanjin Meng, Chenyang Shao, Yuwei Yan, Qinglong Yang, Yiwen Song, Sijian Ren, Xinyuan Hu, Yu Li, Jie Feng, Chen Gao, and Yong Li. Towards large reasoning models: A survey of reinforced reasoning with large language models, 2025a. URL https://arxiv.org/abs/2501.09686.

Guowei Xu, Peng Jin, Ziang Wu, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason step-by-step, 2025b. URL https://arxiv.org/abs/ 2411.10440.

Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Qiaowei Li, Zheng Lin, Li Cao, and Weiping Wang. Dynamic early exit in reasoning models, 2025a. URL https://arxiv. org/abs/2504.15895.

Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Zheng Lin, Li Cao, and Weiping Wang. Dynamic early exit in reasoning models. CoRR, abs/2504.15895, 2025b. doi: 10. 48550/ARXIV.2504.15895. URL https://doi.org/10.48550/arXiv.2504.15895.

Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning. arXiv preprint arXiv:2502.03387, 2025.

Xixian Yong, Xiao Zhou, Yingying Zhang, Jinlin Li, Yefeng Zheng, and Xian Wu. Think or not? exploring thinking efficiency in large reasoning models via an information-theoretic lens, 2025. URL https://arxiv.org/abs/2505.18237.

Bin Yu, Hang Yuan, Haotian Li, Xueyin Xu, Yuliang Wei, Bailing Wang, Weizhen Qi, and Kai Chen. Long-short chain-of-thought mixture supervised fine-tuning eliciting efficient reasoning in large language models, 2025a. URL https://arxiv.org/abs/2505.03469.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025b. URL https: //arxiv.org/abs/2503.14476.

Linan Yue, Yichao Du, Yizhi Wang, Weibo Gao, Fangzhou Yao, Li Wang, Ye Liu, Ziyu Xu, Qi Liu, Shimin Di, and Min-Ling Zhang. Don’t overthink it: A survey of efficient r1-style large reasoning models, 2025. URL https://arxiv.org/abs/2508.02120.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024a.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024b.

Yuheng Zha, Kun Zhou, Yujia Wu, Yushu Wang, Jie Feng, Zhi Xu, Shibo Hao, Zhengzhong Liu, Eric P Xing, and Zhiting Hu. Vision-g1: Towards general vision language reasoning with multidomain data curation. arXiv preprint arXiv:2508.12680, 2025.

Hanning Zhang, Pengcheng Wang, Shizhe Diao, Yong Lin, Rui Pan, Hanze Dong, Dylan Zhang, Pavlo Molchanov, and Tong Zhang. Entropy-regularized process reward model, 2024a. URL https://arxiv.org/abs/2412.11006.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pp. 169–186. Springer, 2024b.

Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Yichi Zhang, Ziyu Guo, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, Shanghang Zhang, et al. Mavis: Mathematical visual instruction tuning. arXiv e-prints, pp. arXiv–2407, 2024c.

Shengjia Zhang, Junjie Wu, Jiawei Chen, Changwang Zhang, Xingyu Lou, Wangchunshu Zhou, Sheng Zhou, Can Wang, and Jun Wang. Othink-r1: Intrinsic fast/slow thinking mode switching for over-reasoning mitigation, 2025. URL https://arxiv.org/abs/2506.02397.

Tianyu Zheng, Tianshun Xing, Qingshui Gu, Taoran Liang, Xingwei Qu, Xin Zhou, Yizhi Li, Zhoufutu Wen, Chenghua Lin, Wenhao Huang, Qian Liu, Ge Zhang, and Zejun Ma. First return, entropy-eliciting explore, 2025. URL https://arxiv.org/abs/2507.07017.

Jason Zhu and Hongyu Li. Towards concise and adaptive thinking in large reasoning models: A survey, 2025. URL https://arxiv.org/abs/2507.09662.

Brian D. Ziebart, Andrew Maas, J. Andrew Bagnell, and Anind K. Dey. Maximum entropy inverse reinforcement learning. In Proceedings of the 23rd National Conference on Artificial Intelligence

- Volume 3, AAAI’08, pp. 1433–1438. AAAI Press, 2008. ISBN 9781577353683.

#### Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models, 2025. URL https://arxiv.org/abs/2411.00836.

APPENDIX

APPENDIX CONTENTS

- A Implementation Details 24
- B Token-Level Entropy Measurement 25
- C RLVR Algorithms 25

- C.1 Group Relative Policy Optimization (GRPO). . . . . . . . . . . . . . . . . . . . . 25
- C.2 Dynamic sAmpling Policy Optimization (DAPO). . . . . . . . . . . . . . . . . . . 25

- D Window-Entropy Aggregation 26
- E KL Penalty Inflates GRPO Advantage Variance compared to KL Loss 26
- F The Algorithm workflow of AEPO 27
- G On-policy difficulty and buckets 28
- H High-Entropy Tokens 29
- I Visual Analysis of Entropy Reward Design 29
- J Why High–Entropy Tokens Predict Reasoning Response Length 30
- K Why KL Loss is a Valid Thinking Budget 31
- L Fisher–Geometry Justification of AEPO 33
- M Case Study 35

- A IMPLEMENTATION DETAILS

Our models, based on Qwen2.5-VL-7B-Instruct and Qwen2.5-VL-3B-Instruct, are trained using a two-stage strategy. The first stage is a cold-start supervised fine-tuning (SFT), where we independently fine-tune the large language model (LLM) module for two epochs. This stage uses a batch size of 256, a sequence length of 32k, and a learning rate of 2 × 10−5. Following SFT, we further optimize the model using Adaptive Entropy Policy Optimization (AEPO). In each AEPO iteration, we sample 512 prompts from the training set and generate eight rollouts per prompt, yielding 4,096 trajectories. Rollouts are generated with a temperature of 1.0 and a top-p of 0.99, and the maximum sequence length is set to 20k tokens (a 4k token prompt and a 16k token response). For policy updates, we use a global batch size of 128 and the AdamW optimizer with a learning rate of 1 × 10−6 and a weight decay of 1×10−2, without a learning rate warmup. To ensure training stability, we apply online filtering to discard samples whose mean overall reward falls outside the predefined range of [0.01, 0.99]. This process removes outliers that could distort advantage estimation and impede policy updates.

- B TOKEN-LEVEL ENTROPY MEASUREMENT

To quantify the model’s local uncertainty during generation, we compute the token-level entropy at step t as

Ht = −

V

j=1

pt,j log pt,j, pt = πθ(· | q,o<t), (3)

where πθ denotes the policy under parameters θ, q is the input query, and o<t is the prefix of generated tokens up to position t − 1. The probability vector pt ∈ RV is obtained from the normalized logits zt by a temperature-scaled softmax transformation:

pt = Softmax z

t

T ,

with vocabulary size V and decoding temperature T controlling the sharpness of the distribution.

Role in our setting. Unlike sequence-level measures (e.g., log-likelihood of the whole output), Ht captures the fine-grained uncertainty at each generation step. During reinforcement learning, we treat Ht as an intrinsic signal of “reasoning hesitation”: a higher entropy indicates that the model considers multiple plausible continuations, whereas a lower entropy suggests that the model is confident about the next token.

Note that Ht is tied to the distribution pt, not to the realized token ot drawn from pt. Hence two identical tokens generated at different timesteps may correspond to different entropies, depending on the local distributional context. In this sense, token entropy should be interpreted as a property of the model’s decision process rather than of the discrete outcome itself.

- C RLVR ALGORITHMS

In this subsection, we provide additional details of the baseline RLVR algorithms used in our experiments, namely Group Relative Policy Optimization (GRPO) and Dynamic sAmpling Policy Optimization (DAPO). Both methods extend traditional policy optimization frameworks to improve the stability and effectiveness of reinforcement learning for reasoning tasks, yet they differ in how they handle variance reduction, KL regularization, and sampling dynamics.

- C.1 GROUP RELATIVE POLICY OPTIMIZATION (GRPO).

GRPO (DeepSeek-AI et al., 2025) extends conventional policy optimization by introducing groupwise normalization. Specifically, training samples are divided into K groups {G1,...,GK} based on input prompts or task-specific criteria. For each group Gi, both a policy πθ and a frozen reference policy πθ

ref

are maintained. The GRPO objective can be written as:

Ex∼G

i, y∼πθ(y|x) min

πθ(y|x) πθ

ref

(y|x)

Aˆ(x,y), clip

πθ(y|x) πθ

ref

(y|x)

, 1 − ϵ, 1 + ϵ A ˆ(x,y) ,

where ϵ controls the size of the trust region. The group-relative advantage Aˆ(x,y) is defined as:

Aˆ(x,yi) =

r(x,yi) − mean({r(x,y1),...,r(x,yG)}) std({r(x,y1),...,r(x,yG)}) + ϵ

.

This relative advantage stabilizes training by normalizing each sample’s reward against the variance of its group. By leveraging intra-group normalization, GRPO encourages responses that are better than their peers, while maintaining diversity across groups.

- C.2 DYNAMIC SAMPLING POLICY OPTIMIZATION (DAPO).

Building on GRPO (DeepSeek-AI et al., 2025), DAPO (Yu et al., 2025b) removes the explicit KL penalty and instead incorporates several mechanisms for improved sample efficiency. First, it introduces a clip-higher strategy to reduce over-penalization of promising trajectories. Second, it adopts dynamic sampling, where response sampling probabilities are adaptively adjusted based on entropy

signals. Third, DAPO employs token-level policy gradient updates with overlong reward shaping. The DAPO objective is formulated as:

  1

|oi|

G

min rti(θ), clip(rti(θ),1 − ϵlow,1 + ϵhigh)Aˆit

JDAPO(θ) = E(q,a)∼D,{oi}Gi=1∼πθold(·|q)

G i=1 |oi|

t=1

i=1

where rti(θ) is defined as in GRPO, and Aˆit is the token-level relative advantage. Unlike GRPO, which focuses on group-level normalization, DAPO shifts optimization to the token level, enabling finer-grained control and adaptive shaping of the learning signal. This makes DAPO a strong baseline for RLVR without requiring a value network.

Both GRPO and DAPO contribute to reducing variance and stabilizing policy optimization in RLVR. GRPO leverages group-relative normalization, while DAPO refines the approach with token-level updates and dynamic sampling. In our work, we adopt DAPO as the primary baseline due to its strong empirical performance and its widespread use in prior RLVR literature.

 ,

- D WINDOW-ENTROPY AGGREGATION

While token-level entropy Ht provides a fine-grained measure of uncertainty, single-token fluctuations are often dominated by lexical ambiguity or local randomness. To capture more coherent reasoning segments, we aggregate entropy over a sliding window of length w:

H¯t:w =

1 w

t+w−1

τ=t

Hτ. (4)

This windowed statistic smooths out spurious token-level spikes and highlights regions where the model persistently maintains high uncertainty.

A window is flagged as high-entropy if its average entropy exceeds a dynamic threshold θ, which is updated online to match the distributional scale of the current batch, which is shown in Section 2.1. Intuitively, these windows correspond to stretches of the output where the model is genuinely uncertain about its reasoning trajectory, as opposed to momentary ambiguity on function words or symbols.

The aggregated entropy H¯t:w therefore serves two purposes: (i) it localizes “hard thinking” phases within the output, providing a natural signal for allocating additional exploration, and (ii) it stabilizes training by reducing sensitivity to token-level noise. In our AEPO algorithm, only tokens falling inside validated high-entropy windows are granted relaxed KL budgets, thereby ensuring that extra computation is spent precisely where reasoning effort is most needed.

- E KL PENALTY INFLATES GRPO ADVANTAGE VARIANCE COMPARED TO KL LOSS

For a fixed prompt x, GRPO samples a group of N responses {yi}Ni=1 i.i.d. from πθ(· | x). Let the task+entropy shaped return be

Cent(x,yi) Zent

Si ≜ racc(x,yi) − λd

, (5) and the sample-averaged per-token reference KL estimator be

L

1 L

Ki ≜

kldt,i. (6) We compare two GRPO implementations:

t=1

- 1. KL penalty (merge KL into the return):

N

Ri′ = Si − κKi, A′i = Ri′ − R¯′, R¯′ = N1

Rj′ .

j=1

## 2. KL loss (actor-only):

N

Ri = Si, Ai = Ri − R,¯ R¯ = N1

Rj,

j=1

and add the separate actor regularizer κ · L1 t βt kldt,i which does not enter Ai. Assume {(Si,Ki)}Ni=1 are i.i.d. with finite second moments, and denote

σS2 = Var(S), σK2 = Var(K), Cov(S,K) = ρσSσK.

Lemma 1 (Group-baseline advantage variance). For i.i.d. R1,...,RN with finite variance, the GRPO group-baseline advantage Ai = Ri − R¯ satisfies

Var(Ai) = 1 − N1 Var(R). (7)

Proof. Write Ai = (1 − N1 )Ri − N1 j̸=i Rj and use i.i.d. plus independence across indices to expand Var(Ai); cross-terms cancel and equation 7 follows.

| |
|---|

Proposition 1 (KL penalty inflates GRPO advantage variance). Under the assumptions above,

Var(A′i) − Var(Ai) = 1 − N1 κ2σK2 − 2κCov(S,K) . (8)

In particular, if Cov(S,K) ≈ 0 (empirically common), then Var(A′i)−Var(Ai) ≈ 1− N1 κ2σK2 >

- 0 for any κ ̸= 0. More generally, unless κ ∈ 0, 2Cov(S,K)/σK2 (a narrow interval when Cov(S,K) is small), one has Var(A′i) > Var(Ai).

Proof. By Lemma 1, it suffices to compare Var(Ri′) to Var(Ri). We have Var(Ri) = Var(S) = σS2 and

Var(Ri′) = Var(Si − κKi) = σS2 + κ2σK2 − 2κCov(S,K). Subtracting and multiplying by (1 − N1 ) yields equation 8.

| |
|---|

Consequences for gradient variance. Let the score-function term be Gi = Lt=1 ∇θ log πθ(yi,t | si,t) and suppose, as is standard in variance analyses, that A and G are weakly correlated.1 Then

Var A′iGi ≈ Var(A′i)E∥Gi∥2 > Var(Ai)E∥Gi∥2 ≈ Var AiGi ,

whenever Var(A′i) > Var(Ai) by Proposition 1. In the KL-loss implementation, the additional KL gradient κ t βt∇θkldt,i is not multiplied by the group advantage and therefore does not inherit its variance amplification; it also targets the trust-region deviation at a per-token granularity, improving credit assignment.

Merging per-token KL into the return injects a high-frequency, task-agnostic signal (Ki) into the groupwise competition, thereby amplifying the variance of the advantage and the policy-gradient estimator. Using KL as an actor-only regularizer decouples the trust-region control from the task/entropy signals, keeping the advantage low-variance and allowing a dedicated controller to track the KL budget.

- F THE ALGORITHM WORKFLOW OF AEPO

In this section, we provide a detailed flowchart of our AEPO algorithm in the following diagram 1.

1This assumption is used for comparative purposes; the conclusion also holds under more general covariance decompositions.

Algorithm 1: Adaptive-Entropy Policy Optimization (AEPO)

Input : Reference model πref; initial policy πθ; prompts D; group size G; window size w; rollout parameters (temperature, top-p); entropy quantile q = 0.95; relaxation factor ρ < 1; difficulty buckets {easy, medium, hard} with KL targets {δd}, entropy weights {λd}, base KL weights {βd}, and shaping functions Sd(·); GRPO clipping parameter ϵ; controller step size ακ.

Output: Updated policy πθ

- 1 for each training iteration do

- 2 Sample a mini-batch of prompts {xb}Bb=1 ∼ D
- 3 foreach prompt x do

- // (1) Rollout generation and entropy computation

4 Generate G rollouts {oi}Gi=1 ∼ πθ(· | x) 5 Compute token entropies Hi,t and sliding-window means H¯i,t:w

- // (2) Dynamic thresholding

6 θi ← Quantileq({Hi,t}t) for each i; 7 θ ← G1 Gi=1 θi mi,t ← I{H¯i,t:w ≥ θ}; NHE(i) ← t mi,t

- // (3) Difficulty estimation

8 Compute pass@G accuracy px; assign difficulty bucket d(x) based on px

- // (4) Sequence rewards with entropy shaping

9 Ri ← racc(x, oi) + Sd(x)(NHE(i), acci) − λd(x) C

(i) ent

Zi

- // (5) Group-centered advantages with entropy bonus

10 Ri ← Ri − G1 Gj=1 Rj 11 for t = 1 to Li do

12 Agrpi,t ← Ri/Li 13 ψi,t ← λd(x) ϕ([H¯i,t:w − θ]+) mi,t − bi 14 A˜i,t ← Agrpi,t + ψi,t 15 end

- // (6) KL regularization with window-adaptive weights

16 kldi,t ← DKL(πθ(· | si,t)∥πref(· | si,t)) 17 βi,t ← βd(x) · ρmi,t

- // (7) Actor update with clipped GRPO objective

- 18 ri,t(θ) ← πθ(oi,t | si,t)/πθold(oi,t | si,t)
- 19 Lactor ← −G1 i

1

Li t min{ri,t(θ)A˜i,t, clip(ri,t(θ), 1 − ϵ, 1 + ϵ)A˜i,t}

- 20 LKL ← κd(x) · G1 i

1

Li t βi,t kldi,t

- 21 L ← Lactor + LKL
- 22 Update θ ← θ − η∇θL // (8) KL controller update (non-window tokens only)
- 23 KLctrl ← G1 i

1

Li t I{mi,t = 0} kldi,t

- 24 κd(x) ← clip κd(x) 1 + ακ(KLctrl/δd(x) − 1) , κmin, κmax
- 25 end
- 26 end

- G ON-POLICY DIFFICULTY AND BUCKETS

Instance difficulty is estimated online using group rollouts. With K=8 samples per prompt,

and we assign buckets

1 8

pass@8(x) =

8

1{correct(y(k),x)}, (9)

k=1

d(x) =

 

easy, pass@8(x) ≥ 6, medium, 3 ≤ pass@8(x) < 6, hard, pass@8(x) ≤ 2.



(10)

[Figure 25]

Figure 9: Effect of entropy shaping and KL regularization on accuracy reward. We plot the moving-average accuracy reward over training steps under different ablation settings. Baseline methods like GRPO and DAPO either lack stable improvement or plateau early. In contrast, our ARES variants consistently achieve higher accuracy rewards throughout training. Notably, combining both KL regularization and entropy shaping yields the most stable and significant gains, demonstrating the necessity of the two components working together.

Buckets are recomputed each iteration and will parameterize per-bucket budgets/targets in our method.

- H HIGH-ENTROPY TOKENS

In our analysis, we define the number of high-entropy tokens in a trajectory by combining an entropybased thresholding procedure with a semantics-aware filtering step.

Semantic filtering. Rather than counting all tokens that exceed a certain entropy threshold, we first restrict attention to a fixed vocabulary of semantically meaningful reasoning triggers. This vocabulary is constructed using a filtering prompt applied to GPT, which classifies each candidate token as either “reasoning-relevant” (True) or not (False). The retained tokens include transition markers (e.g., but, however), reasoning initiators (e.g., therefore, thus), and other structural markers that signal the onset of a reasoning step. Tokens with little semantic content (e.g., A, B, digits, or domain-specific placeholders) are excluded. The entire resulting vocabulary can be visualized in Figure ??(b), where the word cloud highlights the most frequent reasoning triggers.

Entropy thresholding. Given token-level entropy values {Ht}Tt=1, we compute the 95th percentile of entropies within the current batch, denoted θ0.95. A token ot is labeled as high-entropy if

Ht ≥ θ0.95 and ot ∈ Vsem, where Vsem is the fixed set of semantically filtered tokens. The number of high-entropy tokens in a trajectory y = (o1,...,oT) is thus defined as

NHE(y) =

T

t=1

1[Ht ≥ θ0.95 ∧ ot ∈ Vsem].

This definition ensures that our entropy-based reward focuses only on tokens that both exhibit unusually high uncertainty and carry clear semantic signals of reasoning transitions. As a result, the entropy reward mechanism becomes more interpretable and more closely aligned with reasoningrelevant exploratory behavior.

- I VISUAL ANALYSIS OF ENTROPY REWARD DESIGN

To better understand the effect of our entropy reward shaping, we provide a visual analysis in Figure 9. Each plot illustrates the entropy reward component as a function of the deviation

∆ = NHE − NHEtarget, where NHE is the number of detected high-entropy tokens and NHEtarget is the difficulty-dependent target.

Easy tasks. For easy problems, excessive high-entropy activity indicates unnecessary overthinking. As shown in the left panel, the entropy reward heavily penalizes positive deviations (∆ > 0), while providing only mild encouragement when errors occur (acc = 0). This ensures that correct responses remain short and efficient, while still allowing a small degree of exploratory thinking if the model initially fails.

Medium tasks. For medium-difficulty problems, the design is symmetric around the target. Moderate deviations from the target are tolerated, but excessive deviations in either direction are penalized. Importantly, when the model answers incorrectly, the entropy reward encourages it to increase reasoning length (i.e., generate more high-entropy tokens). This balances efficiency with robustness, preventing both under- and over-thinking.

Hard tasks. For hard problems, high entropy is consistently rewarded regardless of accuracy. As shown in the right panel, both correct and incorrect responses receive positive shaping when ∆ > 0. This reflects the intuition that difficult problems require longer and more exploratory reasoning chains, and the model should be encouraged to sustain high-entropy reasoning segments.

Overall, the entropy reward curves realize a difficulty-aware shaping strategy: they suppress unnecessary verbosity on easy problems, regulate reasoning depth for medium ones, and strongly encourage exploratory thinking on hard problems. This design provides a smooth and interpretable mechanism for adaptive reasoning across the full difficulty spectrum.

- J WHY HIGH–ENTROPY TOKENS PREDICT REASONING RESPONSE LENGTH

Given input x, the policy πθ generates a response o = (o1,...,oL) with random stopping time τ = L. Let ht = (x,o1:t−1) be the history at step t and define the (possibly top-k normalized) token entropy

pθ(v | ht) log pθ(v | ht). (11) For a window size w and threshold θ, define window entropy and a high–entropy indicator

Ht = −

v

t+w−1

1 w

H¯t:w =

Hτ, IHEt = I{H¯t:w ≥ θ}, (12) and let the (validated) high–entropy count be

τ=t

L

IHEt . (13)

NHE =

t=1

Our goal is to relate L to NHE (or the total high–entropy residence time).

A two–state latent process. Assume an interpretable latent state St ∈ {R,V} with: (i) Reasoning (R): exploratory/high–entropy; (ii) Verbatim (V): declarative/low–entropy. There exist constants

HR > HV and a threshold θ ∈ (HV,HR) such that the detector has bounded error

Pr(IHEt = 1 | St = R) ≥ 1 − α, Pr(IHEt = 0 | St = V) ≥ 1 − β, (14) with α,β ∈ [0,1). Let the R → V transition probability be q ∈ (0,1], and let the answer–emitting hazard in V be h ∈ (0,1] (stopping can only occur in V).

- Theorem 1 (Linear relation with reasoning residence). Let TR = τt=1 I{St = R} be the total time spent in the reasoning state. Then there exist a,b > 0 such that

E[L] = a + bE[TR]. (15) Moreover, under equation 14 there exist constants c1,c2 > 0 for which

c1 E[NHE] ≤ E[TR] ≤ c2 E[NHE], (16) hence

|E[L] = a′ + b′ E[NHE]|
|---|

for some a′,b′ > 0. (17)

Proof sketch. Within R, the process cannot stop; each reasoning excursion has geometric length with mean 1/q. Within V, the segment ends with geometric stopping hazard h (mean 1/h). The total length is a renewal sum of i.i.d. excursions plus a terminal V segment, yielding equation 15. Bounded detector errors map TR to NHE, proving equation 17. □

A stopping–time view via entropy–dependent hazard. Assume the probability of emitting the final answer at step t is a non–increasing function of entropy,

Pr(stop at t | ht) = λ(Ht), λ′(H) ≤ 0. (18) Then larger entropy sequences stochastically dominate the stopping time.

- Theorem 2 (High entropy delays stopping). For two trajectories with entropy paths {Ht(1)} and {Ht(2)} such that Ht(1) ≤ Ht(2) for all t, one has E[L(1)] ≤ E[L(2)]. Moreover, for any threshold θ,

τ

I{Ht ≥ θ} λ(θ) ≥

1 λ(θ)

E[NHE]. (19)

E[L] ≥ E

t=1

Proof sketch. By stochastic ordering under equation 18, higher entropy lowers the instantaneous hazard and yields larger stopping times in the convex order, implying the expectation inequality. Since λ(H) ≤ λ(θ) whenever H ≥ θ, each high–entropy step contributes at least 1/λ(θ) units in expectation; summing gives equation 19. □

Information–theoretic lower bound. Suppose answering with error ε over a label set of size M requires mutual information

### I† ≜ log M − h2(ε) − εlog(M − 1) (20)

an (by Fano’s inequality). Let it ≡ I(A;ot | ht−1) be per–step information about the (correct) answer A. By data processing, it ≤ Ht. Empirically, non–reasoning steps convey negligible information: there exists η ≪ 1 such that it ≤ η when Ht < θ. Let ιθ ≡ maxH≥θ i(H). Then

τ

I† − η E[L] ιθ − η

I† ≤

, (21)

it ≤ η (L − NHE) + ιθ NHE ⇒ NHE ≥

t=1

which rearranges to a linear lower bound of E[L] in terms of E[NHE].

Synthesis and testable predictions. The renewal argument (Theorem 1), the hazard view (Theorem 2), and the information budget bound jointly imply a stable, monotone, near–linear relationship between reasoning length and high–entropy activity:

### E[L] ≈ α + β E[NHE], β > 0. (22)

This yields concrete diagnostics: (i) E[L | NHE] is monotone increasing; (ii) the slope β correlates with the mean reasoning–excursion duration 1/q; (iii) the estimated stopping hazard λˆ(H) is decreasing; and (iv) the per–step mutual information it concentrates within high–entropy windows. As shown in Figure 3, the number of high-entropy tokens grows approximately in tandem with response length and accuracy, validating entropy as a principled proxy for reasoning effort. Yet, this relationship is not uniform across all problems: as we show next, the way entropy and reasoning length contribute to accuracy differs substantially between easy and hard instances.

- K WHY KL LOSS IS A VALID Thinking Budget

We show that the KL loss used in AEPO is mathematically equivalent to enforcing a budget on the policy’s deviation from the reference model, and that such a budget provably controls the expected “thinking” cost (e.g., reasoning length L or entropy-based cost Cent).

For a prompt x, let πθ(· | x) be the policy over full responses o ∈ Y, πref(· | x) the frozen reference, and r(x,o) a task reward. Let c(x,o)≥0 be a thinking cost (e.g., response length, or high-entropy

window cost), and define the shaped reward r˜(x,o) = r(x,o) − λc(x,o) with λ ≥ 0. Denote the (pathwise) KL divergence DKL πθ(· | x)∥πref(· | x) . Constrained formulation. We pose the budgeted policy optimization:

θ(·|x)[˜r(x,o)] s.t. Ex∼D DKL πθ(· | x)∥πref(· | x) ≤ δ, (23)

Ex∼D Eo∼π

max

πθ

where δ > 0 is the thinking budget. The feasible set is convex because DKL(·∥πref) is convex in its first argument.

- Lemma 2 (Strong duality and KL-as-budget). Problem equation 23 admits strong duality. Its Lagrangian is

L(π,κ) = Ex Eo∼π(·|x)[˜r(x,o)] − κEx DKL π(· | x)∥πref(· | x) − δ , κ ≥ 0, (24) and the optimal policy for fixed x has the exponential-tilt form

π∗(o | x) ∝ πref(o | x) exp κ 1 r˜(x,o) . (25)

Moreover, at the dual optimum κ∗ the budget is active unless κ∗ hits its boundary: ExDKL π∗(· | x)∥πref(· | x) = δ.

Proof. The objective is linear in π and the constraint set is convex with nonempty interior (Slater’s condition holds by taking π = πref). Thus strong duality holds. Optimizing L over π(· | x) under the simplex constraint yields equation 25 via standard exponential-family calculus. Complementary slackness gives the budget activity.

| |
|---|

Token factorization and actor-only KL loss. For auto-regressive policies, by the chain rule,

DKL πθ(· | x)∥πref(· | x) = Eo∼π

θ

|o|

t=1

DKL πθ(· | st)∥πref(· | st) , (26)

with st = (x,o1:t−1). Hence a per-token KL loss in the actor objective

κ ·

1 |o| t

βt DKL πθ(· | st)∥πref(· | st)

is precisely a discretization of the dual term in equation 23; weights βt∈(0,∞) allow token-adaptive emphasis (e.g., relaxing inside validated high-entropy windows). Because this term does not enter the advantage, it controls deviation magnitude independently of the task/entropy signals that decide when to explore.

Budget tracking via dual updates. Let KL be a moving-average estimate of the left-hand side of equation 23. The multiplicative update

κ ← clip κ 1 + ακ( KL/δ − 1) , κmin,κmax (27)

is a stochastic dual ascent that drives KL → δ. Under standard Robbins–Monro conditions on ακ and bounded variance, κt →κ∗ a.s., and the primal iterates satisfy the budget asymptotically. Thus the actor-only KL loss with controller implements an operational thinking budget.

From KL budget to thinking budget. We now show that bounding KL controls the expected thinking cost (e.g., c = L or c = Cent).

- Lemma 3 (Pinsker-type bound). For any bounded f : Y →R with ∥f∥∞ ≤ M,

### [f] ≤ M 2DKL πθ∥πref . (28)

### Eπ

### [f] − Eπ

θ

ref

Thus if f is the (clipped) reasoning length or normalized entropy cost, a global KL budget δ implies a corresponding bound on the change of its expectation.

Theorem 1 (Donsker–Varadhan control of moment budgets). For any η > 0 and measurable f,

1 η

1 η

eηf +

DKL πθ∥πref . (29) Consequently, under the budget DKL ≤ δ,

### Eπ

log Eπ

[f] ≤

θ

ref

1 η

δ η

[eηf] +

. (30)

### Eπ

log Eπ

[f] ≤ inf

θ

ref

η>0

Proof. equation 29 is the Donsker–Varadhan variational inequality obtained by upper-bounding the log-moment generating function via relative entropy.

| |
|---|

Taking f = c(x,o) (e.g., L or Cent) shows that a KL budget upper-bounds the expected thinking cost relative to the reference through the reference MGF, i.e., the policy cannot arbitrarily increase reasoning length or entropy cost without paying KL.

Mirror-descent / trust-region view. Maximizing E[˜r] under a local KL ball DKL(π∥πold) ≤ δ yields the mirror-descent (natural-gradient) update

1 √

π∗ ∝ πold exp κ 1 r˜ , with κ ≍

, (31)

δ

so δ is precisely the trust-region radius: a smaller (larger) budget forces smaller (larger) policy movement and therefore smaller (larger) allowable growth of c by Lemma 3/Theorem 1. In autoregressive models, equation 26 further identifies the budget with a tokenwise sum of deviations, making the KL loss a time–additive budget meter of exploration.

Token-adaptive windows preserve convexity. Let {wt} be nonnegative weights (e.g., wt ∈[ρ,1] with ρ < 1 inside validated reasoning windows). The weighted budget t wt DKL(πt∥πtref) remains convex in {πt} and admits the same dual treatment; hence the actor-only term κ t wt kldt is still a valid Lagrangian penalty for a window-relaxed KL budget. This realizes the intuition: “relax

- KL where we intend to think” without breaking the budgeting semantics.

(i) Strong duality turns the KL loss into the Lagrange multiplier of the reference-deviation budget; (ii) dual updates equation 27 track the target δ; (iii) information inequalities (Pinsker, Donsker–Varadhan) translate a KL budget into explicit upper bounds on the expected thinking cost. Therefore, the actor-only KL loss in AEPO is not merely a regularizer: it is a principled and operational thinking budget.

L FISHER–GEOMETRY JUSTIFICATION OF AEPO

This subsection formalizes why the proposed Adaptive–Entropy Policy Optimization (AEPO) is principled from the viewpoint of information geometry. We show that AEPO is equivalent to a token–reweighted natural–gradient update under a per–difficulty trust region, and that the entropy–augmented advantage and token–adaptive KL together maximize improvement in the directions that matter for reasoning.

Setup. Given a prompt x, the auto–regressive policy πθ generates a response o = (o1:L) with states st = (x,o<t). Let r(x,o) be the task reward and Cent(x,o) the entropy–window cost (Section ??). For a difficulty bucket d = d(x), AEPO solves the following constrained problem:

max

πθ

Ex∼D Eo∼π

θ(·|x) r(x,o) − λd(x) Cent(x,o)

s.t. Ex∼D Eo∼π

θ(·|x)

L(o)

1 L(o)

βd(x),t DKL πθ(· | st) πref(· | st) ≤ δd(x) .

t=1

(32)

Here L(o) denotes the response length and st = (x,o<t). The token weight is βd,t = βd ρt, where ρt ∈ (0,1) if t lies in a validated high–entropy window and ρt = 1 otherwise.

KL as a Fisher quadratic. Let Ft denote the token–wise Fisher information matrix under the reference policy πref:

ref(·|st) ∇log πref(a | st) ∇log πref(a | st)⊤ . For a small parameter displacement ∆θ, the per–token KL admits the standard second–order approximation

Ft = Ea∼π

### DKL πθ+∆θ(· | st) πref(· | st) = 12 ∆θ⊤Ft ∆θ + o(∥∆θ∥2). (33)

- Lemma 4 (Weighted Fisher trust region). Under the approximation equation 33, the KL con-

straint in equation 32 is equivalent to the quadratic trust region 12 ∆θ⊤Fβ ∆θ ≤ δd, with the token–reweighted Fisher

Fβ ≜

L

1 L

βd,t Ft, βd,t = βd ρt, ρt ∈ (0,1]. (34)

t=1

Proof. Average equation 33 over t with weights βd,t and use linearity. Dual/Lagrangian form and natural gradient. Introducing a Lagrange multiplier κd ≥ 0, the inner (fixed–d) problem becomes

| |
|---|

2 ∆θ⊤Fβ ∆θ, g ≜ ∇θ E r(x,o) − λd Cent(x,o) , (35) whose maximizer is the natural–gradient step

g⊤∆θ − κ

max

d

∆θ

|∆θ⋆ = κ−d 1 Fβ−1 g|
|---|

. (36)

Proposition 2 (One–step improvement bound). Under equation 33, the first–order improvement at ∆θ⋆ satisfies

- 1

- 2κd

2 ∆θ⋆⊤Fβ∆θ⋆ =

### ∆J ≜ g⊤∆θ⋆ − κ

g⊤Fβ−1g. (37)

d

Proof. Plug equation 36 into equation 35.

| |
|---|

Why token–adaptive relaxation helps. Inside validated reasoning windows we choose ρt < 1, which reduces the local curvature contribution in Fβ on those tokens. Simultaneously, the entropy–augmented advantage increases the corresponding components of g only where H¯t:w ≥ ϑ. Because the gain equation 37 increases when (i) g has larger energy and (ii) the metric penalty Fβ is smaller in the same directions, AEPO aligns a boosted gradient with a relaxed Fisher metric on reasoning windows.

Corollary 1 (Directional benefit of window relaxation). Let u be a unit vector supported on window tokens. If ρt is reduced on these tokens, then u⊤Fβu decreases while u⊤g increases under entropy shaping. Hence the directional contribution (u⊤g)2/(2κd u⊤Fβu) to equation 37 strictly increases, improving sample efficiency on reasoning segments.

Why use non–window KL for control. The KL controller adjusts κd to match the per–bucket target δd. If we fed the controller with the already–relaxed (window–weighted) KL, the true natural length 21∆θ⊤(L1 t Ft)∆θ could be under–estimated, risking drift. Using a non–window (or down–weighted–window) control signal KLctrl stabilizes the global trust region while keeping local relaxation intact.

Per–bucket decoupling: shape vs. scale. The token weights βd,t determine the shape of the metric Fβ (where to allow deviation), whereas the multiplier κd determines its scale (how much to deviate) for each difficulty bucket. This yields a well–conditioned, two–time–scale scheme: slow adaptation of κd (budget tracking) and fast ascent in equation 35.

AEPO corresponds to a per–difficulty, token–reweighted natural–gradient update

−1

∆θ⋆ = κ−d 1 L 1

∇θE r − λd Cent

### ,

βd,t Ft

t

entropy–shaped g

with βd,t = βdρt relaxing KL strictly within high–entropy windows and κd controlled by a non–window KL signal. Proposition 2 shows this maximizes g⊤Fβ−1g/(2κd) by (i) amplifying g where the model is genuinely reasoning and (ii) reducing curvature along the same directions, while a stable bucket–wise trust region ensures controlled deviation from πref.

- M CASE STUDY

|QUESTION As shown in the figure, in the right triangle one acute angle is 60◦ and the side opposite this angle has length 6√3. Find the hypotenuse y.<br><br>|y|6|
|---|---|
| | |
<br><br>x<br><br>√3<br><br>60◦ Answer:<br><br>|12|
|---|
|
|---|

## Tokens: 243

OUR ADAPTIVE MODEL SOLUTION FOR EASY QUESTION Use the sine ratio for the 60◦ angle:

6√3 y

opposite hypotenuse

sin60◦ =

=

.

√3

Since sin60◦ =

2 , we have

√3 2

6√3 y

6√3 · 2 √3

=⇒ y =

=

= 12.

|y = 12|
|---|

|QUESTION<br><br>The sum of the circumferences of circles H, J, and K is 56π units. Their radii are marked as rH = x, rJ = 2x, rK = 4x, and the three circles are pairwise tangent externally. Find the distance KJ between the centers of circles K and J.<br><br>[Figure 26]|
|---|

## Tokens: 4173

OUR ADAPTIVE MODEL SOLUTION FOR HARD QUESTION First, recall that the circumference of a circle is 2πr. The problem states that the sum of the circumferences of circles H,J,K is 56π, so

2π(rH + rJ + rK) = 56π =⇒ rH + rJ + rK = 28.

At this point, I examined the diagram and noticed the labels 4x, x, 2x. I hesitated: are these radii, or are they distances between centers? If interpreted as center distances, then

### KH = 4x, HJ = x, JK = 2x,

but this quickly fails because x+2x = 3x < 4x, violating the triangle inequality. Therefore, this interpretation cannot be correct. Next, I considered that these labels instead denote the radii:

rH = x, rJ = 2x, rK = 4x. Substituting into rH + rJ + rK = 28 yields

x + 2x + 4x = 28 =⇒ 7x = 28 =⇒ x = 4.

However, doubts persisted. For instance, if one misreads JK as 2x, then KJ would be 8, which contradicts the tangent-circle rule requiring

### KJ = rK + rJ = 4x + 2x = 6x = 24.

... Here I went back and forth, testing whether the labels might be mixed as radii or distances, but each alternative led to inconsistencies. For example, assuming KH = 4x as a distance forces rK = 3x if tangent to rH = x, yet then HJ should be 3x not x, another contradiction. These cycles of reasoning strongly suggested that the only consistent interpretation is that the labels 4x, x, 2x represent the radii themselves.

Therefore, with x = 4 we compute the actual radii:

### rK = 16, rH = 4, rJ = 8.

Since circles K and J are externally tangent, the distance between their centers is

KJ = rK + rJ = 16 + 8 = 24. Finally, check consistency:

2π(rH + rJ + rK) = 2π(4 + 8 + 16) = 2π(28) = 56π, matching the given condition exactly.

|24|
|---|

|QUESTION In ⊙W, arcs RS and TV are congruent. Find the length of RS.<br><br>[Figure 27]|
|---|

QUESTION WITH APPROPRIATE NUMBER OF HIGH-ENTROPY TOKENS

difficulty: Easy Gen high entropy token: 20 Threshold(easy): 28 Entropy reward: 0.0 acc reward: 1.0 overall reward: 1.0

Okay, let’s see. I need to find the length of RS in circle ⊙W. The problem says the arcs RS and TV are congruent. First, recall that in a circle, congruent arcs correspond to congruent chords. Therefore, if arc RS ∼= arc TV , then chord RS ∼= chord TV . The problem gives:

RS = 5x − 9, TV = 2x + 3. So we set them equal:

5x − 9 = 2x + 3. Solving:

3x − 9 = 3 =⇒ 3x = 12 =⇒ x = 4. Now check:

RS = 5(4) − 9 = 11, TV = 2(4) + 3 = 11.

Wait, let me double-check. Are RS and TV the correct chords? Arc RS corresponds to chord RS, arc TV to chord TV . Since arcs are congruent, chords must be equal. Yes, consistent.

Thus, the required length is

|11|
|---|

.

|QUESTION In △ABC, the side lengths are a = 14, b = 48, and c = 50. Find cosB.<br><br>[Figure 28]|
|---|

QUESTION WITH EXCESS HIGH-ENTROPY TOKENS

difficulty: Easy Gen high entropy token: 51 Threshold(easy): 28 Entropy reward: -0.5 acc reward: 1.0 overall reward: 0.5

Okay, so I need to find cosB given a triangle with sides a = 14, b = 48, and c = 50. Let me recall the Law of Cosines:

c2 = a2 + b2 − 2abcosC.

First, label the triangle properly. Side a is opposite angle A, b opposite B, and c opposite C. The diagram shows ∠C = 90◦. Therefore, c = 50 must be the hypotenuse, and a = 14, b = 48 are the legs.

Now, angle B is at vertex B, between sides BC (a = 14) and AB (c = 50). The opposite side is AC (b = 48). Thus, by definition:

adjacent hypotenuse

a c

14 50

7 25

cosB =

=

=

=

.

But let me confirm with the Law of Cosines. For angle B:

a2 + c2 − b2 2ac

cosB =

## . Substitute carefully:

142 + 502 − 482 2 · 14 · 50

196 + 2500 − 2304 1400

392 1400

7 25

cosB =

=

=

=

.

Wait, let me double-check. If a = 14, b = 48, c = 50, then indeed 142 + 482 = 196 + 2304 = 2500 = 502, confirming it is a right triangle. Hence the direct trigonometric ratio already suffices.

...(several “wait” moments, checking whether sides match standard notation, omitted for brevity) ...

Finally, both direct definition and Law of Cosines give the same result. Therefore, the value is consistent:

7 25 ≈ 0.28.

cosB =

|7 25<br><br>|
|---|

|QUESTION Find x.<br><br>[Figure 29]|
|---|

MEDIUM QUESTION WITH EXCESS HIGH-ENTROPY TOKENS (INCORRECT ANSWER)

difficulty: Medium Gen high entropy token: 84 Threshold(medium): 55 Entropy reward: 0.38 acc reward: 0.0 overall reward: 0.38

Okay, let’s try to figure out how to find x. The problem involves a diagram with a green square and a blue square connected. There is a point F with a segment of length 10, and another segment labeled 4.

Wait, maybe the diagram is a cube net, since dashed lines often indicate hidden edges. If FW = 10 and WW′ = 4, perhaps x is the distance from F to W′.

Alternatively, maybe this is related to vectors or a triangle. If x, 10, and 4 form sides of a right triangle, then

√

√

x2 = 102 + 42 = 116, x =

116 = 2

29.

But hold on, maybe the 10 and 4 are not perpendicular. Then by the Law of Cosines:

x2 = 102 + 42 − 2 · 10 · 4cosθ, but without θ, this is indeterminate. Next, let me try coordinates. Suppose F = (0,0,0). If W = (10,0,0) and W′ = (10,4,0), then

√

FW′ = 102 + 42 =

116. Alternatively, if W′ = (10,0,4), the distance is also √116.

...(several further “wait” moments about whether this is a parallelogram, trapezoid, or 3D projection are omitted here) ...

Another thought: perhaps FW and WW’ are adjacent edges of a cube net, so x is the straightline distance from F to W′. Therefore, again,

√

x = 102 + 42 = 2

29.

Yet, could it be colinear? If so, x = 10 + 4 = 14. But that seems inconsistent with the diagram. Maybe it’s a right triangle hidden in the net. If FW is one leg, WW’ another, then x is the hypotenuse. Finally, after multiple conflicting attempts, the reasoning converges to

√

x = 2

29.

|QUESTION Find y.<br><br>[Figure 30]|
|---|

MEDIUM QUESTION WITH FEW HIGH-ENTROPY TOKENS (INCORRECT ANSWER)

difficulty: Medium Gen high entropy token: 24 Threshold(medium): 55 Entropy reward: 0.19 acc reward: 0.0 overall reward: 0.19

Okay, so I need to find the value of y in this triangle. The diagram shows a right triangle with a 60◦ angle, a side of length 15, and the hypotenuse labeled y. Since it is a right-angled triangle, the other non-right angle must be 30◦.

In a 30-60-90 triangle, the sides are in ratio 1 : √3 : 2. The side opposite 30◦ is the shortest. If the side of length 15 is opposite 30◦, then the hypotenuse should be 2 × 15 = 30. Alternatively, using cos60◦ = 12:

15 y

cos60◦ =

=⇒ y = 30.

But, in the problem statement the side labeled 15 might not match this assumption. If it were opposite 60◦, the calculation would change.

...(further checks on labeling are omitted) ... Finally, the reasoning settled on

y = 30.

