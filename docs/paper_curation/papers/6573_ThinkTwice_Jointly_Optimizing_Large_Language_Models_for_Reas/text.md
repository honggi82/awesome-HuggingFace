# arXiv:2604.01591v2[cs.AI]6Apr2026

## ThinkTwice: Jointly Optimizing Large Language Models for Reasoning and Self-Refinement

Difan Jiao1†, Qianfeng Wen1†, Blair Yang1,2, Zhenwei Tang1 & Ashton Anderson1 1Department of Computer Science, University of Toronto 2Coolwei AI Lab †Equal contribution. Contact: {difanjiao, ashton}@cs.toronto.edu

##### Abstract

We introduce ThinkTwice, a simple two-phase framework that jointly optimizes LLMs to solve reasoning problems and refine the answers, based on Group Relative Policy Optimization (GRPO). In each pair of training steps, ThinkTwice first optimizes the model on solving reasoning problems, then optimizes it on refining its own solutions to the same problems, using the same binary correctness reward in both phases without correctness signals or critique annotations. Across five mathematical reasoning benchmarks and two model families including Qwen3-4B and Olmo3-7B, ThinkTwice substantially improves both reasoning and refinement performance over competitive online policy optimization baselines. Specifically, on Qwen3-4B, ThinkTwice outperforms GRPO on AIME by 5 percentage points before refinement and by 11.5 points after one self-refinement step, measured by pass@4. Analysis of the training dynamics of ThinkTwice reveals an implicit rectify-then-fortify curriculum: refinement predominantly corrects errors early in training and naturally shifts toward preserving already-correct solutions as the model improves, yielding a more rectified reward signal. Our work establishes joint training of reasoning and self-refinement as a principled and effective methodology for RLVR. Our codebase is available at https://github.com/CSSLab/ThinkTwice.

##### 1 Introduction

Reinforcement learning with verifiable rewards (RLVR) has emerged as an effective paradigm for improving the reasoning capabilities of large language models (LLM) (Shao et al., 2024; Guo et al., 2025). Yet even strong reasoners can produce solutions that contain correctable errors, such as incomplete derivations, algebraic mistakes, or unproductive solution paths. A natural strategy from human problem-solving is self-refinement: people routinely revisit initial solutions, identify errors, and revise their reasoning (Polya, 1945). There is growing evidence that self-refinement can similarly benefit LLMs to recover on challenging problems where initial attempts narrowly fail (Gou et al., 2023; Weng et al., 2023).

However, existing approaches to improving self-refinement in LLMs, broadly categorized into training-free and training-based methods, each have notable limitations. First, trainingfree methods (Madaan et al., 2023; Shinn et al., 2023) prompt the model to critique and revise at inference time but do not learn a reusable refinement policy. Moreover, promptonly refinement remains brittle even for frontier models. As illustrated in Figure 1(A), we observe a slight performance decrease from the frontier LLM on AIME24 after self-refinement prompting. Second, although training-based methods learn refinement behavior, existing methodologies typically rely on process supervision (Zhang et al., 2024a; Cui et al., 2025), critique annotations (Kumar et al., 2024; Zhang et al., 2025b), or explicit signals indicating whether the initial answer is correct (Ma et al., 2025). Such supervision is not always available in practice: at the frontier, no stronger model exists to provide reliable critique, and human oversight may be insufficient for challenging problems (Burns et al., 2023). This limits existing training-based approaches to settings with rich external feedback.

Figure 1: (A) Prompt-only reflection can reduce top frontier LLM’s performance on AIME24, indicating brittleness. (B) ThinkTwice compared with existing method families. (C) ThinkTwice addresses these gaps by sequentially training a shared model backbone—first solving, then reflecting—yielding significant gains (+5 points reasoning, +11 points refinement) on AIME with Qwen3-4B.

In this work, we introduce ThinkTwice, a unified RLVR framework that jointly optimizes reasoning and self-refinement without access to external information. In each pair of training steps, ThinkTwice first optimizes the model on solving a batch of reasoning problems, then optimizes it on refining its own solutions to the same problems (“thinking twice”), using a generic review instruction and the same binary correctness reward in both phases. The refinement phase gives the model a second attempt at each problem conditioned on its prior solution, allowing it to recover from correctable errors that the first attempt missed. As shown in Figure 1(B, C), ThinkTwice serves as a simple, effective method that doesn’t require correctness signals, critique annotations, or external teacher models.

We train ThinkTwice with Qwen3-4B and Olmo3-7B and evaluate on a suite of five mathematical reasoning benchmarks. ThinkTwice substantially improves both reasoning and self-refinement capabilities over competitive baselines. Specifically, on Qwen3-4B, ThinkTwice outperforms GRPO on the challenging AIME problems by 5 percentage points before refinement and by 11.5 points after one step of self-refinement, measured by pass@4. Across both models, ThinkTwice achieves the highest average performance for both reasoning and self-refinement, and when used as a refiner applied to other baselines’ solutions, ThinkTwice also achieves the highest score.

To understand how ThinkTwice works, we analyze its training dynamics and find that the refinement phase gives rise to an implicit rectify-then-fortify curriculum: early in training, refinement predominantly corrects errors in failed solutions, and as the model improves, it naturally shifts toward preserving and polishing already-correct ones. This dynamic yields a more rectified reward signal, as the refinement phase recovers useful learning signal from problems where the base attempt alone falls short. Additionally, ThinkTwice adds only 3% training overhead per step, and it reaches its best checkpoint in 16% less wall-clock time than GRPO, as the enriched training signal empirically leads to faster convergence.

##### 2 Related Work

Reinforcement learning with verifiable rewards (RLVR). Recent progress in reasoning post-training has been driven by reinforcement learning with verifiable rewards (RLVR), where models are optimized using automatically checkable outcome signals such as exact answers. DeepSeekMath introduced Group Relative Policy Optimization (GRPO), a practical critic-free PPO-style recipe for this setting, and DeepSeek-R1 further showed that largescale outcome-only RL can elicit strong reasoning behavior without process supervision (Shao et al., 2024; Guo et al., 2025). A large follow-up literature has since expanded the RLVR design space. Open reproductions and training recipes such as Open-Reasoner-Zero, SimpleRL-Zoo, and Skywork-OR1 helped make large-scale RLVR more reproducible, while systems work such as AReaL improved training efficiency at scale (Hu et al., 2025; Zeng et al., 2025b; He et al., 2025b; Fu et al., 2025). On the optimization side, DAPO improves

Phase 1: Reasoning Optimization Phase 2: Refinement Optimization

###### Think Twice

Sampled Problems Policy

GRPO Update

Policy (LLM)

(Eq.1)

(LLM)

Compute Rewards & Advantage

Evaluate Answers:

𝟙

Refinement Attempts

Same Reward System

Candidate Solutions

Refinement Prompt Construction

Sample Refinement Input from Phase 1

User: A right square pyramid with volume 54 ...

[Figure 1]

Ground-Truth Answers

Correctness

[Figure 2]

Compute Rewards & Advantage

[Figure 3]

Assistant: We are given a right square ...

User: Carefully review your previous solution, and ...

...

GRPO Update

[Figure 4]

Randomly pick one base solution from each sample.

(Eq.1)

Assistant: Let's go through the problem again carefully, step by step,...

[Figure 5]

Figure 2: ThinkTwice at a glance.

long-CoT training stability; Dr. GRPO analyzes optimization bias in GRPO; GSPO moves from token-level to sequence-level importance ratios and clipping; and newer variants such as GMPO, GPG, and shrinkage baselines revisit ratio aggregation, simplification, responselength bias, and baseline variance (Yu et al., 2025a; Liu et al., 2025; Zheng et al., 2025; Zhao et al., 2025b; Chu et al., 2025; Zeng et al., 2025a).

Self-refinement. Self-refinement aims to improve an initial response by generating feedback or verification and then revising the answer (Madaan et al., 2023). A first class of methods is training-free and operates purely at inference time, such as Self-Refine, Reflexion, self-verification prompting, CRITIC, and self-consistency decoding (Madaan et al., 2023; Shinn et al., 2023; Weng et al., 2023; Gou et al., 2023; Wang et al., 2022). These approaches show that extra test-time computation, self-feedback, or tool-based critique can improve outputs, but they do not learn a reusable refinement policy; moreover, prompt-only selfcorrection can be unreliable for reasoning without external feedback (Huang et al., 2023). A second class of work trains refinement behavior more directly, including methods based on verifiers (Cobbe et al., 2021; Zhang et al., 2024b), process supervision (Uesato et al., 2022; Lightman et al., 2023; Wang et al., 2024; Yuan et al., 2024; Cui et al., 2025), critique data (Xi et al., 2024; Yu et al., 2025b), synthetic correction traces (Welleck et al., 2022; Qu et al., 2024; Xiong et al., 2025; Zhao et al., 2025a), or multi-turn RL with explicit self-verification or critique objectives (Kumar et al., 2024; Ma et al., 2025; He et al., 2025a; Jiang et al., 2025; Zhang et al., 2025a;b). ThinkTwice is closest to the RL-based self-refinement line, but is deliberately simpler: it uses a shared policy, a generic review instruction, and the same final-answer correctness reward in both phases, without process labels, critique annotations, correctness hints, or an explicit verifier indicating whether the initial answer is correct. A more detailed comparison is provided in Appendix A.

##### 3 Methodology

We introduce ThinkTwice, an RLVR framework that jointly optimizes reasoning and selfrefinement without verification—i.e., without access to external signals indicating whether the intermediate steps or solutions are correct or not. We briefly recap GRPO in Section 3.1, then describe how we formulate the self-refinement problem in Section 3.2, and finally explain how we integrate both reasoning and self-refinement optimization in Section 3.3.

###### 3.1 Preliminaries

We use Group Relative Policy Optimization (GRPO) (Shao et al., 2024; Guo et al., 2025) as our backbone reinforcement learning (RL) algorithm. GRPO eliminates the need for a separate critic model by computing advantages from grouped samples, which offers better stability and efficiency during LLM RL.

Let πθ denote the language model policy parameterized by θ. For a given input x, GRPO samples a group of G responses {y1, . . . , yG} ∼ πθold(·|x), where each generation of yi is often called a rollout, and optimizes:

JGRPO(θ) = Ex,{y

i}iG=1

G

1 G

#### ∑

Li − βDKL(πθ∥πref) , (1)

i=1

where Li = min (ρiAi,clip(ρi,1−ϵ,1+ϵ)Ai) denotes the clipped surrogate objective, ρi = πθ(yi|x)/πθold(yi|x) is the importance ratio, ϵ is the clipping threshold, and β controls the KL penalty against the reference policy πref.

The advantage Ai is computed via group normalization:

ri − mean(r1, . . . ,rG) std(r1, . . . ,rG)

, (2)

Ai =

where ri is the reward for response yi. For mathematical reasoning tasks with verifiable answers, we use outcome-based correctness rewards: ri = [E(y i) = a∗], where E extracts the answer from yi and a∗ is the ground-truth answer.

###### 3.2 Self-Refinement Formulation

In our self-refinement framework, we use base solution to refer to an initial solution attempt that the model will subsequently self-refine into a refined solution. Given a problem x and a base solution ybase generated by the current policy, we construct the refinement prompt as a multi-turn conversation:

xrefine = [User: x ][Ast: ybase ][User: Irefine ], (3)

where Irefine is a task-agnostic instruction that asks the model to review and improve its previous solution. Crucially, Irefine contains no indication of whether ybase is correct or incorrect, distinguishing our approach from methods that rely on external critique signals. We use a general refinement instruction for ThinkTwice as shown in Appendix B.1.

###### 3.3 ThinkTwice

ThinkTwice alternates between two training phases within a unified GRPO framework, as depicted in Figure 2. In each pair of training steps, the model first optimizes on reasoning for a batch of problems, then optimizes on refining its own solutions to the same problems. This exposes the model to each problem twice per training batch under complementary objectives: solving from scratch and improving upon its own prior attempt.

- Phase 1: Reasoning Optimization. For each problem x sampled from the training set D, we generate G candidate solutions {y1, . . . , yG} using the current policy πθ. Each solution is

evaluated with the correctness reward ri(base) = [E(y i) = a∗], and the policy is updated via Eq. (1), yielding πθ′ . From the G rollouts per problem x, we randomly select one base solution for the subsequent refinement phase.

- Phase 2: Refinement Optimization. Using the selected base solutions from Phase 1, we construct refinement prompts following Eq. (3). The model with updated policy πθ′ then

generates G refinement attempts {y1(ref), . . . , y(Gref)} for each refinement prompt. Since πθ′ has already been updated using reward signal from x during Phase 1, it is better equipped

to improve upon the base solutions produced by πθ. Each refined solution is evaluated using the same correctness reward ri(ref) = [E(y i(ref)) = a∗], and the policy is updated via Eq. (1) as well, yielding πθ′′. We now elaborate on two key design choices in the refinement optimization phase.

Base Solution Sampling. Since GRPO relies on multiple rollouts for advantage estimation, we accordingly have multiple base solutions as candidates for ybase. In ThinkTwice, we

Algorithm 1 ThinkTwice: Jointly Optimizing Reasoning and Refinement Require: Training dataset D, current policy πθ, training batch size B, group size G, refine-

ment instruction Irefine

- 1: for each training iteration t = 1,2, . . . , T do
- 2: // Phase 1: Reasoning
- 3: Sample batch of problems {xj}Bj=1 from D
- 4: for each problem xj do
- 5: Generate solutions {yj,1, . . . , yj,G} ∼ πθ(·|xj)
- 6: Compute rewards rj,i = [E(y j,i) = a∗j ] for i = 1, . . . , G
- 7: end for
- 8: Update πθ to πθ′ on reasoning samples
- 9: Sample refinement pairs from: {(xj, yj,i)}j,i
- 10: // Phase 2: Refinement
- 11: for each (x, ybase) in sampled pairs do
- 12: Construct xrefine via Eq. (3)
- 13: Generate refinements {y1(ref), . . . , y(Gref)} ∼ πθ′ (·|xrefine)
- 14: Compute rewards ri(ref) = [E(y i(ref)) = a∗] for i = 1, . . . , G
- 15: end for
- 16: Update πθ′ to πθ′′ on refinement samples
- 17: end for

adopt the simplest sampling strategy: randomly picking one base solution per problem x. This design not only inherently covers the full spectrum of training samples, but also aims to establish an emergent curriculum: early in training, when the model frequently produces incorrect solutions, the refinement phase predominantly trains on error correction. As reasoning accuracy improves, fewer incorrect solutions are available, and the refinement phase naturally shifts toward polishing already-correct solutions. This adaptive curriculum naturally steers training toward appropriately challenging refinement scenarios aligned with the model’s current capability boundaries.

Refined Solution Reward. While Irefine provides high-level guidance for the refinement task, we do not use extra reward signals to enforce adherence to its prescribed structurethrough RL, the model is free to develop its own refinement strategies. ThinkTwice solely employs the binary signal that depends on the correctness of the refined solution. This reward signal encourages the model to: (1) detect and correct errors when the base solution is wrong, or (2) preserve and polish when the base solution is already correct. Because the refinement is formulated as a standard multi-turn conversation, the optimization process is seamlessly handled by native GRPO.

The training procedure is detailed in Algorithm 1 and Appendix B.1.

##### 4 Results

###### 4.1 Experimental Setup

We follow the training and evaluation protocol of Dr. GRPO (Liu et al., 2025). We train on math questions from the MATH training dataset (Hendrycks et al., 2021) and evaluate on a suite of five benchmarks: AIME, AMC, MATH500, Minerva Math (Lewkowycz et al., 2022), and OlympiadBench (He et al., 2024). All methods use the binary correctness reward via Math-Verify,1 with exact-match verification against ground-truth answers.

We experiment with two instruction-tuned models, Qwen3-4B-Instruct-2507 (Yang et al., 2025) and OLMo3-7B-Instruct (Olmo et al., 2025), as the refinement phase requires the ability to follow multi-turn instructions. Our baselines include standard GRPO (Shao et al., 2024), Dr. GRPO (Liu et al., 2025), and DAPO (Yu et al., 2025a). We also include two training-free baselines, Reflexion (Shinn et al., 2023) and Self-Refine (Madaan et al., 2023), for refinement

1https://github.com/huggingface/Math-Verify

Table 1: Reasoning performance (pass@4, ↑) across five mathematical reasoning benchmarks. Bold and underline denote the best and second-best results per dataset within each model.

Method AIME AMC MATH500 Minerva OlympiadBench Average Qwen3-4B

Base Model 29.18 64.87 88.47 39.61 57.90 56.01 GRPO 39.06 75.36 91.86 41.03 63.80 62.22 DrGRPO 35.46 77.65 91.83 42.75 66.51 62.84 DAPO 42.54 80.68 93.55 38.38 67.50 64.53 ThinkTwice 44.11 79.59 93.60 42.94 67.60 65.57

###### OLMo3-7B

Base Model 32.81 68.77 89.87 40.63 61.36 58.69 GRPO 39.38 77.05 92.28 41.13 62.42 62.45 DrGRPO 36.09 74.33 91.65 42.07 65.09 61.85 DAPO 36.72 76.16 91.56 42.39 63.80 62.12 ThinkTwice 39.24 79.89 92.74 43.43 65.81 64.22

evaluation. Our experiments are implemented using VERL (Sheng et al., 2024),2 with implementation details and hyperparameter configurations provided in Appendix B.1.

Our evaluations are two-fold. In Section 4.2, we test the reasoning capability of ThinkTwice and baselines with direct prompting as a single-turn generation. In Section 4.3, we evaluate the refinement capability using the multi-turn chat format, testing models on refining both their own generations and also those produced by others.

###### 4.2 Reasoning

For each benchmark problem, we sample n=32 independent solutions for Qwen3-4B and OLMo3-7B. Following Chen et al. (2021), we report pass@4, the estimated probability that at least one of 4 sampled solutions is correct. The full reasoning pass@k curves are provided in Figure 6 at Appendix C.

- Table 1 reports the reasoning performance on both models. For Qwen3-4B, ThinkTwice achieves the highest average score of 65.57%, outperforming all competitive baselines. The gains are most pronounced on the most challenging AIME benchmark, where ThinkTwice reaches 44.11% compared to 39.06% for GRPO. Across the remaining benchmarks, ThinkTwice is consistently the best or second-best method. For OLMo3-7B, the overall trend mirrors that of Qwen3-4B. with ThinkTwice achieving the highest average score. Notably, these improvements are obtained on direct prompting alone, before any self-refinement step is applied, which indicates that the refinement training phase in ThinkTwice itself strengthens the model’s reasoning capability.

###### 4.3 Refinement

Self-Refinement. We first evaluate the end-to-end reason-then-refine pipeline: for each problem, we sample n=32 base solutions, refine each with greedy decoding (thus yielding n=32 refined solutions), and report pass@4 over the refined solutions. Table 2 reports results on Qwen3-4B and Olmo3-7B. For Qwen3-4B, ThinkTwice achieves the highest average of 71.88%, outperforming DAPO by 2.9 percentage points and GRPO by 4.5 points. On AIME, ThinkTwice reaches 60.43%, a gain of over 11 percentage points above GRPO. For OLMo3-7B, the pattern is consistent: ThinkTwice again achieves the highest average selfrefinement score, outperforming all baselines on every benchmark. Meanwhile, ThinkTwice also significantly outperforms training-free baselines3. The full self-refinement pass@k curves are provided in Figure 7 at Appendix C.

2https://github.com/verl-project/verl 3Even under our favorable implementation (details in Appendix B.2), Reflexion appears less suited

to offline math refinement, because retrying from scratch after reflection can discard useful local corrections.

- Table 2: Self-refinement performance (pass@4, ↑) across five mathematical reasoning benchmarks. Bold and underline denote the best and second-best results per dataset within each model. † denotes training-free methods.

Method AIME AMC MATH500 Minerva OlympiadBench Average Qwen3-4B

Base Model 45.25 78.10 92.82 40.81 63.52 64.10 Reflexion† 38.47 73.17 91.48 40.87 60.89 60.98 Self-Refine† 50.37 82.40 93.86 41.19 66.33 66.83 GRPO 48.91 81.86 93.78 42.90 69.67 67.42 DrGRPO 46.98 82.66 94.46 44.84 71.75 68.14 DAPO 49.86 87.31 94.96 40.09 72.81 69.01 ThinkTwice 60.43 85.54 95.70 43.93 73.78 71.88

###### OLMo3-7B

Base Model 39.31 78.18 91.75 41.58 66.14 63.39 Reflexion† 37.38 72.18 91.29 41.48 62.36 60.94 Self-Refine† 47.34 83.81 93.24 42.35 68.30 67.01 GRPO 46.04 84.48 92.28 41.08 66.53 66.08 DrGRPO 45.24 82.32 93.54 42.75 69.81 66.73 DAPO 44.26 84.55 93.33 42.81 68.51 66.69 ThinkTwice 49.33 87.06 94.66 44.33 71.38 69.35

###### (a) Qwen3-4B

###### (b) OLMo3-7B

###### Refinement Model

###### Refinement Model

Base GRPO DrGRPO DAPO ThinkTwice

Base GRPO DrGRPO DAPO ThinkTwice

[Figure 6]

[Figure 7]

58.63 62.84 64.38 65.04 67.45

- 57.13 60.86 61.03 60.24 61.65
- 58.92 60.98 61.01 60.30 62.19

Base

Base

ReasoningModel

- 58.52 61.24 64.39 64.30 67.39

61.79 64.41 63.23 65.00 67.44

- 59.96 63.50 63.56 62.71 67.07

GRPO

GRPO

56.89 60.25 60.37 59.88 61.96

DrGRPO

DrGRPO

- 58.47 62.41 62.06 61.19 62.96
- 59.99 61.88 61.20 61.32 64.65

DAPO

DAPO

63.39 64.21 64.80 65.83 66.97

ThinkTwice

ThinkTwice

[Figure 8]

[Figure 9]

58 60 62 64 66 Average pass@4

56 58 60 62 64 Average pass@4

- Figure 3: Cross-model refinement evaluation (average pass@4, ↑). Rows denote the backbone reasoning model; columns denote the refinement model.

Cross-Model Refinement. To isolate refinement capability from base solution quality, we evaluate each model as a refinement model applied to every other model’s base solutions. For each problem, we generate a single base solution with greedy decoding from the reasoning model, then sample 4 refinements from the refinement model and report the accuracy after refinement. This protocol fixes the input and attributes all variance to the model’s refinement capabilities. Figure 3 reports the results averaged across all five benchmarks. ThinkTwice as the refinement model (the rightmost column) achieves the highest score regardless of which model produced the base solution, demonstrating that its refinement capability generalizes beyond its own outputs.

Taken together, these results show that ThinkTwice improves both reasoning and selfrefinement within a single training framework, and that these gains persist across two different types and sizes of LLMs.

28

Damage-CorrectRate(%)

26

- 0

- 1

- 2

- 3

- 4

- 5

Fix-WrongRate(%)

24

22

20

18

16

14

12

0 50 100 150 200 250 300

Checkpoint step

ThinkTwice fix-wrong

ThinkTwice damage-correct

GRPO fix-wrong

GRPO damage-correct

100

Rate(%)

80

ThinkTwice boxed

ThinkTwice marker

GRPO boxed

GRPO marker

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

750

Words

500

250

0 50 100 150 200 250 300

Checkpoint step

ThinkTwice GRPO

- Figure 4: Training dynamics of refinement across checkpoints. The vertical dashed lines mark the best checkpoints. Left (a): transition metrics on the training set across checkpoints. Right (b): formatting and length metrics across checkpoints during training. Top: boxedanswer and final-answer marker rates; bottom: average response length for self-refinement on correct-only base solutions.
- 5 Discussion

###### 5.1 Training Dynamics of ThinkTwice

We provide quantitative and qualitative analyses for understanding the learning dynamics of ThinkTwice. Our central observation is that ThinkTwice gives rise to an implicit rectify-thenfortify dynamic: refinement predominantly corrects errors early in training, and gradually shifts toward preserving and polishing already-correct solutions as the base policy improves.

Transition Metrics. To characterize this dynamic, we first define two conditional transition rates measured on the training set throughout the course of training. Let fix-wrong (%) denote the fraction of incorrect base solutions that become correct after refinement, and damage-correct (%) denote the fraction of correct base solutions that become incorrect after refinement. Figure 4a plots both metrics across training checkpoints for ThinkTwice and the GRPO baseline.

Early Training: Rectification via Refinement. In the early training stages, the model can fail on mathematical problems due to typical failure modes, namely including incomplete derivations or unproductive search and solution paths. The refinement phase provides a second attempt on the same problem, allowing the model to accordingly recover from such failures where the base attempt narrowly fails. As shown in Figure 4a, ThinkTwice maintains a consistently higher fix-wrong rate than the baseline throughout training, peaking around the half journey of the training. This means that for problems near the model’s current capability boundary, the refinement phase can still produce correct solutions, yielding a more rectified reward signal than base reasoning alone would provide.

Late Training: Fortification via Refinement. As training progresses and the base policy becomes stronger and closer to model’s capability boundary, fewer base solutions are incorrect, and the refinement phase naturally shifts from error correction to solution preservation. Figure 4a shows that ThinkTwice’s damage-correct rate drops near zero in the second half of training, while the baseline’s rate is consistently more than 5 times higher than ThinkTwice’s best checkpoint. Meanwhile, with ThinkTwice, the model’s outputs of correct solutions become significantly shorter over training (Figure 4b, Bottom), indicating that refinement is not merely acting as a prolonged generation window but rather helping remove exploratory clutter from already-correct solutions. Also, refined solutions exhibit better answer formatting than vanilla GRPO (Figure 4b, Top), with a higher rate of providing both boxed answers and Final Answer markers despite no format reward being applied during training.

Qualitative Evidence. Finally, we provide detailed case studies in Appendix D illustrating three recurring refinement behaviors that emerge during ThinkTwice training: (i) route

(a) Reward

(b) Response Length

(c) Time per Step

1.0

200

| |* *| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

MeanReward

1500

150

0.9

Seconds

Tokens

100

1000

0.8

50

0.7

* * 500

* *

0

0 50 100 150 200 250 300

Checkpoint step

###### (d) Accumulated Training Time

(e) Average Benchmark Accuracy

0.60

GRPO

7.5

0.58

Accuracy

ThinkTwice (Reasoning)

Hours

5.0

0.56

ThinkTwice (Refinement)

2.5

ThinkTwice (Both)

0.54

7.2h 8.6h

0.0

* *

* *

0 50 100 150 200 250 300

0 50 100 150 200 250 300

Checkpoint step

Checkpoint step

- Figure 5: Training-time cost and dynamics of ThinkTwice compared with GRPO. * denoted best checkpoint step of models. (a) Mean reward. (b) Response length. (c) Wall-clock time per update. (d) Accumulated training time. (e) Within-checkpoint macro average benchmark accuracy. Solid orange denotes ThinkTwice base updates, solid blue denotes GRPO, and dashed orange denotes ThinkTwice refinement updates when applicable.

switching, where refinement abandons a bad solution path and finds a better method; (ii) solution completion, where a promising but unfinished base trace is completed by the refinement; and (iii) late-stage fortification, where an already-correct base solution is shortened and cleaned up without altering the answer. These behaviors align with the quantitative transition from correction-dominant to preservation-dominant refinement observed above.

Taken together, the training dynamics suggest that ThinkTwice’s joint optimization naturally yields a more rectified reward signal, as the refinement phase provides useful gradient information from problems where the base attempt alone provides limited learning signal.

###### 5.2 Training Cost of ThinkTwice

A natural concern is whether the two-phase design of ThinkTwice introduces significant overhead or instabilities during training. We summarize core training-time signals from Qwen3-4B comparing ThinkTwice with GRPO in Figure 5.

Panel (a) shows that the mean reward of ThinkTwice remains comparably stable to GRPO throughout training, indicating that the refinement phase does not destabilize policy optimization. Panel (b) shows that the base response lengths of ThinkTwice stay in the same range as GRPO, while the refinement responses become progressively shorter over training, consistent with the shift toward concise and fortified generation observed in Section 5.1.

Panel (c) shows the wall-clock time per update4. Refinement phase updates are more expensive than standard GRPO updates, especially early in training, because the refinement prompt prepends the model’s earlier solution and therefore contains substantially longer context. However, as training progresses, refined responses become progressively shorter (as shown in Panel (b)), narrowing the training cost gap. Thus, Panel (d) shows that the accumulated wall-clock time of ThinkTwice remains close to GRPO: at the same step count up to step 300, ThinkTwice is only 3% slower in total wall-clock time (9.42 hours vs 9.15 hours). Moreover, ThinkTwice reaches its best checkpoint in 16% less wall-clock time than GRPO (7.2h vs 8.6h), as it converges in fewer total steps (220 vs 280). Finally, Panel (e) shows that ThinkTwice improves faster in the early and middle stages of training and maintains higher benchmark accuracy than GRPO for most of the trajectory.

##### 6 Conclusion

We introduced ThinkTwice, a two-phase RLVR framework that jointly optimizes reasoning and self-refinement using the same binary correctness reward in both phases, without correctness signals, critique annotations, or external verifiers. Across five mathematical reasoning benchmarks and two model families, ThinkTwice consistently outperforms competitive online policy optimization baselines on both direct reasoning and self-refinement.

4All timing results are measured on 2×H100 80GB GPUs. Hyperparameter details are in Appendix B.1.

Our quantitative and qualitative analyses of the training dynamics of ThinkTwice reveal an implicit rectify-then-fortify curriculum that yields a more rectified reward signal, while adding only minimal training overhead compared to GRPO which we build upon.

Several directions naturally extend this work. First, while we evaluate on mathematical reasoning with verifiable answers, ThinkTwice’s design is domain-agnostic and could be applied to other tasks with outcome-based rewards, such as code generation. Second, our framework natively supports arbitrary numbers of refinement turns within the multi-turn conversation format; exploring multi-step iterative refinement is a direct extension.

##### Ethics Statement

This work focuses on improving mathematical reasoning capabilities of large language models through reinforcement learning with verifiable rewards. Our method uses publicly available datasets and models, and does not involve human subjects, private data, or dual-use concerns beyond those inherent to general-purpose language model research.

##### Reproducibility Statement

We provide comprehensive implementation details to facilitate reproduction of our results. Appendix B.1 includes the full training workflow with code-level illustrations, the exact refinement instruction, hyperparameter configurations (Table 4), and dataset sources with links. Appendix B.2 details the implementation of all training-free baselines. An anonymous codebase for ThinkTwice is linked in the abstract. All experiments use publicly available base models and datasets.

##### References

Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, et al. Weak-tostrong generalization: Eliciting strong capabilities with weak supervision. arXiv preprint arXiv:2312.09390, 2023.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. Gpg: A simple and strong reinforcement learning baseline for model reasoning. arXiv preprint arXiv:2504.02546, 2025.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Yuchen Zhang, Jiacheng Chen, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025.

Wei Fu, Jiaxuan Gao, Xujie Shen, Chen Zhu, Zhiyu Mei, Chuyi He, Shusheng Xu, Guo Wei, Jun Mei, Jiashu Wang, et al. Areal: A large-scale asynchronous reinforcement learning system for language reasoning. arXiv preprint arXiv:2505.24298, 2025.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Critic: Large language models can self-correct with tool-interactive critiquing. arXiv preprint arXiv:2305.11738, 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Bolei He, Xinran He, Mengke Chen, Xianwei Xue, Ying Zhu, and Zhen-Hua Ling. Rise: reasoning enhancement via iterative self-exploration in multi-hop question answering. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 14925–14948, 2025a.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, 2024.

Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, et al. Skywork open reasoner 1 technical report. arXiv preprint arXiv:2505.22312, 2025b.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. arXiv preprint arXiv:2503.24290, 2025.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. arXiv preprint arXiv:2310.01798, 2023.

Yuhua Jiang, Yuwen Xiong, Yufeng Yuan, Chao Xin, Wenyuan Xu, Yu Yue, Qianchuan Zhao, and Lin Yan. Pag: Multi-turn reinforced llm self-correction with policy as generative verifier. arXiv preprint arXiv:2506.10406, 2025.

Yizhu Jiao, Jiaqi Zeng, Julien Veron Vialard, Oleksii Kuchaiev, Jiawei Han, and Olivier Delalleau. Think twice: Branch-and-rethink reasoning reward model. arXiv preprint arXiv:2510.23596, 2025.

Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, et al. Training language models to self-correct via reinforcement learning. arXiv preprint arXiv:2409.12917, 2024.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in neural information processing systems, 35:3843–3857, 2022.

Moxin Li, Wenjie Wang, Fuli Feng, Fengbin Zhu, Qifan Wang, and Tat-Seng Chua. Think twice before trusting: Self-detection for large language models through comprehensive answer reflection. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 11858–11875, 2024.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The twelfth international conference on learning representations, 2023.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

Ruotian Ma, Peisong Wang, Cheng Liu, Xingyan Liu, Jiaqi Chen, Bang Zhang, Xin Zhou, Nan Du, and Jia Li. S2r: Teaching llms to self-verify and self-correct via reinforcement learning. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 22632–22654, 2025.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in neural information processing systems, 36:46534– 46594, 2023.

Luk´aˇs Mikula, Michal Stef´ˇ anik, Marek Petroviˇc, and Petr Sojka. Think twice: Measuring the efficiency of eliminating prediction shortcuts of question answering models. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2179–2193, 2024.

Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, et al. Olmo 3. arXiv preprint arXiv:2512.13961, 2025.

Hoang Phan, Victor Li, and Qi Lei. Think twice, generate once: Safeguarding by progressive self-reflection. arXiv preprint arXiv:2510.01270, 2025.

George Polya. How to solve it: A new aspect of mathematical method. Princeton university press, 1945.

Yushan Qian, Bo Wang, Shangzhao Ma, Wu Bin, Shuo Zhang, Dongming Zhao, Kun Huang, and Yuexian Hou. Think twice: A human-like two-stage conversational agent for emotional response generation. arXiv preprint arXiv:2301.04907, 2023.

Yuxiao Qu, Tianjun Zhang, Naman Garg, and Aviral Kumar. Recursive introspection: Teaching language model agents how to self-improve. Advances in Neural Information Processing Systems, 37:55249–55285, 2024.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652, 2023.

Xiaoyu Tian, Sitong Zhao, Haotian Wang, Shuaiting Chen, Yunjie Ji, Yiping Peng, Han Zhao, and Xiangang Li. Think twice: Enhancing llm reasoning by scaling multi-round test-time thinking. arXiv preprint arXiv:2503.19855, 2025.

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process-and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.

Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9426–9439, 2024.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models, 2022. URL https://arxiv.org/abs/2203.11171. ICLR 2023.

Sean Welleck, Ximing Lu, Peter West, Faeze Brahman, Tianxiao Shen, Daniel Khashabi, and Yejin Choi. Generating sequences by learning to self-correct. arXiv preprint arXiv:2211.00053, 2022.

Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao. Large language models are better reasoners with self-verification. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 2550–2575, 2023.

Alex Wilf, Sihyun Lee, Paul Pu Liang, and Louis-Philippe Morency. Think twice: Perspectivetaking improves large language models’ theory-of-mind capabilities. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8292–8308, 2024.

Zhiheng Xi, Dingwen Yang, Jixuan Huang, Jiafu Tang, Guanyu Li, Yiwen Ding, Wei He, Boyang Hong, Shihan Do, Wenyu Zhan, et al. Enhancing llm reasoning via critique models with test-time and training-time supervision. arXiv preprint arXiv:2411.16579, 2024.

Wei Xiong, Hanning Zhang, Chenlu Ye, Lichang Chen, Nan Jiang, and Tong Zhang. Selfrewarding correction for mathematical reasoning. arXiv preprint arXiv:2502.19613, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, et al. DAPO: An open-source LLM reinforcement learning system at scale, 2025a. URL https://arxiv.org/abs/2503.14476.

Tianshu Yu, Chao Xiang, Mingchuan Yang, Pei Ke, Bosi Wen, Cunxiang Wang, Jiale Cheng, Li Zhang, Xinyu Mu, Chuxiong Sun, et al. Training language model to critique for better refinement. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 26760–26804, 2025b.

Lifan Yuan, Wendi Li, Huayu Chen, Ganqu Cui, Ning Ding, Kaiyan Zhang, Bowen Zhou, Zhiyuan Liu, and Hao Peng. Free process rewards without process labels. arXiv preprint arXiv:2412.01981, 2024.

Guanning Zeng, Zhaoyi Zhou, Daman Arora, and Andrea Zanette. Shrinking the variance: Shrinkage baselines for reinforcement learning with verifiable rewards. arXiv preprint arXiv:2511.03710, 2025a.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025b.

Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, and Jie Tang. ReSTMCTS*: LLM self-training via process reward guided tree search, 2024a. URL https: //arxiv.org/abs/2406.03816. NeurIPS 2024.

Fuxiang Zhang, Jiacheng Xu, Chaojie Wang, Ce Cui, Yang Liu, and Bo An. Incentivizing llms to self-verify their answers. arXiv preprint arXiv:2506.01369, 2025a.

Xiaoying Zhang, Yipeng Zhang, Hao Sun, Kaituo Feng, Chaochao Lu, Chao Yang, and Helen Meng. Critique-grpo: Advancing llm reasoning with natural language and numerical feedback. arXiv preprint arXiv:2506.03106, 2025b.

Yunxiang Zhang, Muhammad Khalifa, Lajanugen Logeswaran, Jaekyeom Kim, Moontae Lee, Honglak Lee, and Lu Wang. Small language models need strong verifiers to selfcorrect reasoning. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 15637–15653, 2024b.

Xutong Zhao, Tengyu Xu, Xuewei Wang, Zhengxing Chen, Di Jin, Liang Tan, Zishun Yu, Zhuokai Zhao, Yun He, Sinong Wang, et al. Boosting llm reasoning via spontaneous self-correction. arXiv preprint arXiv:2506.06923, 2025a.

Yuzhong Zhao, Yue Liu, Junpeng Liu, Jingye Chen, Xun Wu, Yaru Hao, Tengchao Lv, Shaohan Huang, Lei Cui, Qixiang Ye, et al. Geometric-mean policy optimization. arXiv preprint arXiv:2507.20673, 2025b.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

##### A Additional Related Work Comparison

- Table 3: A compact comparison of ThinkTwice with related work. We compare whether methods use reinforcement learning, require supervision beyond final-answer correctness, and rely on an additional verifier, teacher, or explicit critique channel. Entries summarize the dominant tendency of each family.

Core characteristic

Family Representative methods RL? Extra supervision? Extra verifier / teacher / critique

One-pass RLVR GRPO, DeepSeek-R1, DAPO, Dr. GRPO

Yes No No Trains first-pass reasoning only; no learned revision policy.

Training-free refinement

Self-Refine, Reflexion, self-verification prompting, CRITIC, self-consistency

No No during training No, or optional tools

Uses extra test-time feedback or sampling, but refinement is not learned.

Training-based refinement

Cobbe verifier, Let’s Verify, Math-Shepherd, Free Process Rewards, PRIME, AutoMathCritique, CFT, Recursive Introspection, Self-rewarding Correction, SPOC

Mixed Often yes Often yes or partial

Usually relies on process labels, critique data, synthetic correction traces, or stronger verifiers.

SCoRe, S2R, RISE, PAG, Self-Verify, Critique-GRPO

RL-based refinement

Yes Usually yes Usually explicit

Closest prior line, but typically adds explicit verification or critique objectives.

ThinkTwice Two-phase RLVR with a shared

Yes No No Same correctness reward in solve and revise phases; revision conditions only on the model’s prior answer plus a generic review instruction.

policy

Position of ThinkTwice in the literature. Table 3 provides a broader comparison between ThinkTwice and prior approaches discussed in Section 2. We organize the literature into four families: one-pass RLVR methods, training-free self-refinement methods, training-based self-refinement methods, and RL-based self-refinement methods. The comparison highlights that ThinkTwice is closest to the RL-based self-refinement line, but differs in using a shared policy and the same sparse final-answer correctness reward in both the solve and revise phases, without process labels, critique annotations, correctness hints, or an explicit verifier indicating whether the initial answer is correct.

Other second-pass methods. Prior work has explored second-pass reasoning, reflection, and related reconsideration strategies in a range of settings, including emotional dialogue generation (Qian et al., 2023), Theory-of-Mind prompting via SimToM (Wilf et al., 2024), shortcut analysis in question answering (Mikula et al., 2024), LLM self-detection via T3 (Li

- et al., 2024), Multi-round Thinking for inference-time scaling (Tian et al., 2025), Progressive Self-Reflection for safety (Phan et al., 2025), and branch-and-rethink for reward modeling (Jiao
- et al., 2025). Although these works share the high-level intuition that revisiting an initial response can help, they differ substantially in task, objective, and supervision. In contrast, our ThinkTwice jointly trains a single reasoning policy with RL to perform both first-pass reasoning and second-pass refinement using the same sparse final-answer correctness reward in both phases, without critique annotations, external verifiers, or a separate rewardmodel objective.

##### B Reproducibility

- B.1 ThinkTwice Implementation Details

- B.1.1 Datasets

We use the MATH dataset 5 for training, which consists of 7,500 problems spanning difficulty levels 1–5 across seven subjects. For evaluation, we use a test set of 1,526 problems from five widely used mathematical reasoning benchmarks:

- • MATH500 6: 500 problems from the MATH test set.
- • OlympiadBench 7: 581 problems from olympiad-level competitions.
- • Minerva Math 8: 272 problems from high school math competitions.
- • AIME: 90 problems from AIME 2022–2024 9.
- • AMC: 83 problems from the AMC 10/12 competitions 10.

Each problem is formatted with a task instruction requiring the model to output its final answer in boxed format (\boxed{}). We evaluate accuracy using exact matching on the boxed content with Huggingface Math-Verify.

- B.1.2 A Detailed Illustration of ThinkTwice Training Workflow

We provide a concrete implementation-style walkthrough of ThinkTwice training, bridging the Algorithm 1 pseudocode with the actual code execution flow. The following uses Python-like notation containing the core training logic from ray trainer.py.

The two-phase training. The implementation follows the two-phase structure in Algorithm 1. When we activate ThinkTwice training, the main training loop alternates between reasoning and refinement steps within each iteration:

|for epoch in range(total_epochs): for batch in dataloader:<br><br># Phase 1: Reasoning (with refinement candidate preparation) pending_refinement = run_step(batch , allow_refinement=True)<br><br># Phase 2: Refinement (if candidates selected) if pending_refinement:<br><br><br>run_step(pending_refinement , allow_refinement=False)<br><br>|
|---|

When allow refinement=True, run step internally performs candidate selection after reward computation and policy update:

|num_select = len(batch) // repeat_times # One per prompt refinement_indices = select_refinement_indices(<br><br>uids , responses , is_correct , num_select , selection_mode='random' # for random selection from base solutions<br><br>) selected_base = base_batch.select_idxs(refinement_indices) selected_gen = gen_batch_output.select_idxs(refinement_indices) # yielding (selected_base , selected_gen) for refinement prompt<br><br>|
|---|

The refinement prompts are then constructed from selected gen (containing the generated responses) and selected base (containing the original prompts):

- 5https://huggingface.co/datasets/hendrycks/math
- 6https://huggingface.co/datasets/HuggingFaceH4/MATH-500
- 7https://huggingface.co/datasets/afraamn/olympiadbench math textonly

- 8https://huggingface.co/datasets/math-ai/minervamath
- 9https://huggingface.co/datasets/AI-MO/aimo-validation-aime
- 10https://huggingface.co/datasets/AI-MO/aimo-validation-amc

###### Category Hyperparameter

GRPO Training

Learning rate 1 × 10−6 PPO clip ratio 0.2 Max response length 3000 Train batch size 32 PPO mini batch size 8 Group size (G) 8 Entropy coefficient 0.0 KL penalty in reward Disabled

Refinement Training

Refinement steps 2 (one reasoning + one refinement) Refinement selection mode random

Generation

Temperature 1.0 (train), 0.0 (val) Top-p 1.0 Top-k −1 Max model length 8192

Table 4: Key hyperparameters for ThinkTwice training.

|messages = [ {'role': 'user', 'content': problem}, # raw prompt {'role': 'assistant', 'content': base_solution}, # from Phase 1 {'role': 'user', 'content': refinement_INSTRUCTION}<br><br>] # messages is then packaged into pending_refinement , then passed to<br><br>run_step() for Phase 2<br><br>|
|---|

with the refinement instruction as:

Follow this instruction, carefully review your previous solution:

- 1. Go through each calculation step-by-step. Check if there are any errors in calculations, logic, or problem understanding.
- 2. If you find any mistakes, explicitly point out what was wrong and explain the correct approach.
- 3. If the solution is already correct, verify each step and explain it more clearly.
- 4. Finally, after finishing the review, provide your refined solution and answer.

The refinement instruction is task-agnostic and contains no correctness signals, ensuring the model learns self-refinement without external supervision.

###### B.1.3 Hyperparameter Configuration

- Table 4 summarizes the key hyperparameters for ThinkTwice training.

###### B.2 Implementation of Training-free Refinement Baselines

We compare our trained models against four non-training test-time baselines: base reasoning, one-step refinement, and two iterative methods inspired by SELF-REFINE (Madaan et al., 2023) and REFLEXION (Shinn et al., 2023). All baselines are evaluated in non-thinking mode. To isolate the effect of the inference procedure from prompt engineering, we keep the refinement instruction fixed across all refinement-based baselines.

Prompt formatting. Each evaluation example is rendered with the model’s chat template and an added generation prompt. For OLMO-3-7B-INSTRUCT, the tokenizer template automatically prepends a default system message. In the no-system ablation, we remove only this automatically injected prefix before decoding, so that the model is evaluated without the template-provided system instruction. For one-step refinement, SELF-REFINE, and the memory-generation stage of REFLEXION, we use the same refinement prompt as shown in Appendix B.1.

base reasoning. For each problem xi, we sample n = 32 candidate solutions

yi(,10), yi(,20), . . . , yi(,0n)

from the model using temperature 0.7, top-p = 0.8, top-k = 20, and a maximum generation length of 3000 tokens. All second-stage generations (refinement, refinement, and retry) are decoded greedily. Each initial sample defines one branch. Iterative updates occur only within a branch, and only the final answer of that branch is scored. Consequently, iterative baselines do not receive extra pass@k opportunities beyond the original n branches.

One-step refinement. Given a sampled solution yi(,0j), we construct a second-round prompt containing the original problem, the sampled answer as an assistant turn, and the shared refinement instruction. We then decode a single refined solution yˆi,j greedily. The set

{yˆi,j}nj=1 constitutes the refinement baseline for problem i.

SELF-REFINE. We implement SELF-REFINE as repeated refinement-and-refinement within the same branch, while keeping the refinement instruction identical to the one-step refine-

ment baseline. Starting from yi(,0j), we iteratively apply

yi(,tj+1) = M xi, yi(,tj),r , t = 0, . . . , T − 1, where M denotes the chat model and r denotes the fixed refinement instruction. We score only the final output yi(,Tj ). Under this controlled prompt design, the case T = 1 is identical to one-step refinement; therefore, when reporting SELF-REFINE as a distinct baseline, we use T ≥ 2.

REFLEXION. For REFLEXION, we again begin from a base sample yi(,0j), but use the refine-

ment stage to produce a short verbal memory rather than a final revised answer. Let mi(,1j) denote the first refinement note. The model then retries the original problem from scratch while conditioning on the accumulated memory:

y˜i(,tj+1) = M xi, mi(,≤j t) .

We report the final retry output after the prescribed number of retries. In our default setting, we use one retry so that the test-time cost remains comparable to one-step refinement and two-step SELF-REFINE. Ground-truth answers are never revealed during inference; they are used only for offline evaluation.

Evaluation Protocol. A branch is marked correct if its final answer matches the groundtruth solution under the same boxed-answer grader used throughout evaluation. Let ci be the number of correct branch finals among the n = 32 branches for problem i. Similarly, we estimate pass@k following Chen et al. (2021). This estimator allows all reported values of k to be obtained from a single set of sampled generations, without rerunning inference separately for each k.

Reproducibility note. All baselines use the same random seed and the same underlying evaluation set. The only difference between methods is how each branch is post-processed at test time: no revision for base reasoning, one revision for refinement, repeated revision for SELF-REFINE, and retry conditioned on verbal memory for REFLEXION.

##### C Additional Results

We provide the full pass@k curves for both reasoning and self-refinement evaluation, complementing the pass@4 results reported in Tables 1 and 2. Figure 6 shows reasoning performance and Figure 7 shows self-refinement performance, both across k ∈ {1,2,4,8,16,32} for all five benchmarks and both model families.

Base GRPO DrGRPO DAPO ThinkTwice

###### AIME

###### AMC

###### MATH500

Minerva

OlympiadBench

50

60

95

80

50

70

45

Pass@k(%)

Qwen3-4B

90

40

70

40

60

30

85

60

35

20

50

90

50

95

70

50

80

Pass@k(%)

OLMo3-7B

40

90

70

40

60

85

30

60

30

80

50

20

1 2 4 8 16 32

1 2 4 8 16 32

1 2 4 8 16 32

1 2 4 8 16 32

1 2 4 8 16 32

- Figure 6: Reasoning pass@k curves across five mathematical reasoning benchmarks for Qwen3-4B (top) and OLMo3-7B (bottom).

40

60

80

Qwen3-4B

Pass@k(%)

AIME

70

80

90

AMC

85

90

95

100

MATH500

35

40

45

50

Minerva

60

70

80

OlympiadBench

1 2 4 8 16 32

30

40

50

60

OLMo3-7B

Pass@k(%)

1 2 4 8 16 32

60

70

80

90

1 2 4 8 16 32

85

90

95

1 2 4 8 16 32

30

40

50

1 2 4 8 16 32

60

70

80

Base GRPO DrGRPO DAPO ThinkTwice

- Figure 7: Self-refinement pass@k curves across five mathematical reasoning benchmarks for Qwen3-4B (top) and OLMo3-7B (bottom).

For reasoning (Figure 6), ThinkTwice outperforms across nearly all values of k on both Qwen3-4B and OLMo3-7B than other baselines, with its advantage most pronounced on AIME, where the gap widens steadily as k increases. For self-refinement (Figure 7), the separation is even more striking: ThinkTwice maintains a clear lead on AIME across the entire k range, reaching nearly 77% pass@32 on Qwen3-4B compared to roughly 62% for the next-best baseline. Across both models, the curves confirm that ThinkTwice’s improvements are not artifacts of a particular k value but hold consistently across the pass@k spectrum.

##### D Case Studies

The three recurring mechanisms we wanted to document are: (i) route switching, where refinement abandons a bad search path and finds a better method; (ii) solution completion, where the base trace is promising but unfinished and refinement closes the argument; and (iii) late-stage exploitation, where the base trace is already correct but refinement removes exploratory clutter, normalizes the final answer, and presents a shorter proof. Table 5 summarizes the selected cases.

###### Case Sample Checkpoints Why it matters

- A 481 100, 240 Same problem shows both phases of the story: early refinement discovers the missing telescoping structure; later refinement gives a much shorter version of the now-correct base proof.
- B 497 110, 250 refinement completes a stalled geometric derivation early, then later preserves the method while converting a long decimal-heavy trace into a shorter exact solution.
- C 194 30, 220 Early refinement repairs a concrete derivative mistake; later refinement turns a wandering but correct proof into a short direct argument.
- D 425 100, 220 Geometry case: early refinement recomputes faulty vector algebra; later refinement keeps the invariant but removes most of the symbolic overhead.

Table 5: Case-study map for the detailed qualitative appendix.

- D.1 Case A: sample 481 — early structural discovery, late exploitation on the same problem

Problem and role in the appendix Problem (MATH500, level 5, sample 481). Let n be a positive integer. Simplify

(24 + 14)(44 + 14) · · · [(2n)4 + 41] (14 + 41)(34 + 14) · · · [(2n − 1)4 + 14] Ground truth:

|8n2 + 4n + 1|
|---|

.

Why this case is useful. This is the clearest single-problem narrative we found. At step 100 the ThinkTwice reasoning trace is wrong; the refinement keeps the same factorization but discovers the missing telescoping relation and solves the problem. By step 240 the same problem has become a late-stage exploitation example: the ThinkTwice reasoning is now correct but still very long, while the refinement keeps the proof and compresses it aggressively. The pure-GRPO baseline is informative at both checkpoints: it reaches the right refinement answer at step 100 only after a longer pattern-checking route, and by step 240 its base trace is still wrong while its refinement remains much longer than ThinkTwice’s.

Checkpoint Run Base output refinement output Step 100

ThinkTwice WRONG; 1222 words; last boxed value

CORRECT; 686 words;

|85|
|---|

|8n2 + 4n + 1|
|---|

Pure GRPO WRONG; 1308 words; no final boxed general form

CORRECT; 892 words;

|8n2 + 4n + 1|
|---|

ThinkTwice CORRECT; 1222 words;

CORRECT; 358 words;

Step 240

|8n2 + 4n + 1|
|---|

|8n2 + 4n + 1|
|---|

Pure GRPO WRONG; 1243 words; no final boxed general form

CORRECT; 1067 words;

|8n2 + 4n + 1|
|---|

- Table 6: Case A summary. The same problem exhibits both early correction and late-stage compression.

Early checkpoint (step 100): refinement finds the missing telescoping argument The ThinkTwice reasoning answer already has the right local algebraic move,

x4 + 14 = x2 − x + 21 x2 + x + 21 , but it never turns that factorization into a general product formula. Instead, the trace drifts into pattern-matching on small values of n:

###### ThinkTwice reasoning, step 100 (wrong; 1222 words)

“But maybe we can look for a telescoping pattern? Let’s compute the expression for small values of n and see if we can spot a pattern.” It checks n = 1,2,3 and ends with the last computed value

|85|
|---|

, which is the n = 3 instance rather than a general closed form.

The refinement does not restart from scratch. Instead, it keeps the factorization and adds the structural observation that the base trace missed. After simplifying a single term to

8k2 + 4k + 1 8k2 − 12k + 5

, it makes the crucial recurrence explicit:

###### ThinkTwice refinement, step 100 (correct; 686 words)

“So the denominator of term k is equal to the numerator of term k − 1. Therefore, the entire product telescopes.” It then rewrites the product as

n

bk+1 bk

bn+1 b1

, bk = 8k2 − 12k + 5,

#### ∏

=

k=1

|8n2 + 4n + 1|
|---|

and concludes with

.

The same-step pure-GRPO baseline refinement also reaches the right answer, but by a looser and longer route. It explicitly computes several small cases before inferring the formula:

###### Pure-GRPO refinement, step 100 (correct; 892 words)

“Step 4: Look for telescoping pattern. We now compute this product for small n to detect a pattern.” Later it writes “Perfect! So the entire product simplifies to 8n2 + 4n + 1.”

The important qualitative difference here is not that ThinkTwice is the only system that can solve the problem at this checkpoint; rather, ThinkTwice’s refinement is already more structural. The base gets trapped in empirical pattern-chasing, while the refinement explicitly identifies the recurrence that makes telescoping work.

Late checkpoint (step 240): same proof idea, much shorter refinement By step 240 the ThinkTwice reasoning trace has learned the right proof family: it now factors correctly and even states the recurrence

ak = bk+1

for the telescoping terms. But the base answer is still long and exploratory, with repeated recalculations and explicit small-n checks. A representative moment is:

###### ThinkTwice reasoning, step 240 (correct; 1222 words)

“Wait, let’s do it carefully.” Later, after reaching the closed form, it still spends another long block numerically checking n = 1 and n = 2 from scratch.

The refinement compresses that same proof into a direct bk+1/bk argument: ThinkTwice refinement, step 240 (correct; 358 words) It defines

bk = 8k2 − 12k + 5, computes

bk+1 = 8k2 + 4k + 1, and then writes

n

bk+1 bk

bn+1 b1

= 8n2 + 4n + 1.

#### ∏

=

k=1

|8n2 + 4n + 1|
|---|

The answer is then immediately boxed as

.

The same-step pure-GRPO baseline remains less stable. Its base answer is still wrong, and even its correct refinement is over three times longer than ThinkTwice’s and continues to rely on guess-and-check language:

###### Pure-GRPO refinement, step 240 (correct; 1067 words)

After deriving the same single-term ratio, it still pauses for several empirical checks: “Wait: E1 = 13 ... E2 = 41 ... E3 = 85? Let’s verify.” Only later does it box 8n2 + 4n + 1.

This case is the cleanest qualitative support for the “explore early, exploit later” story. Early on, refinement discovers the missing structure that the base answer cannot yet use. Later on, once the base has learned the right route, refinement becomes the short and decisive version of that same route.

- D.2 Case B: sample 497 — completing a stalled geometric derivation, then normalizing the final answer

Problem and role in the appendix Problem (MATH500, level 5, sample 497). An equiangular octagon has four sides of length 1 and four sides of length

√2

2 , arranged so that no two consecutive sides have the same length. What is the area of the octagon? Ground truth:

|7 2<br><br>|
|---|

.

Why this case is useful. This is our clearest “finish the job” geometry example. At step 110 the ThinkTwice reasoning trace has the right global setup — directions, vectors, cumulative coordinates, and shoelace — but it stalls in the area computation and never produces a final boxed answer. The refinement keeps the same coordinate plan and makes it executable. By step 250 the base trace is correct, but it is still long and reports the area as the decimal 3.5; the refinement keeps the same proof while shortening it

and normalizing the answer to the exact form 72.

Checkpoint Run Base output refinement output

|7 2<br><br>|
|---|

CORRECT; 633 words; Pure GRPO WRONG; 1284 words; no final

ThinkTwice WRONG; 1280 words; no final boxed answer

Step 110

WRONG; 970 words; no final boxed answer

boxed answer

|7 2<br><br>|
|---|
|7 2<br><br>|

|3.5|
|---|

ThinkTwice CORRECT; 1303 words;

CORRECT; 591 words;

Step 250

|7 2<br><br>|
|---|

Pure GRPO CORRECT; 1209 words;

CORRECT; 884 words;

- Table 7: Case B summary. Early refinement completes the coordinate-shoelace derivation; late refinement keeps the method but presents it more cleanly.

###### Early checkpoint (step 110): the base has the right plan but cannot close it

The ThinkTwice reasoning answer chooses a sensible geometry strategy: it sets the side directions to 0◦,45◦, . . . ,315◦, alternates the side lengths, constructs the cumulative coordinates, and then moves to the shoelace formula. The failure is not conceptual; it is procedural. The trace keeps changing how it wants to compute the area:

###### ThinkTwice reasoning, step 110 (wrong; 1280 words)

The trace reaches the area stage and then hesitates: “We can compute the area of a polygon given its vertices using the shoelace formula. Wait — actually, that’s not quite right. Better: use the vector cross-product method.” It never reaches a final boxed answer.

The refinement keeps the same coordinate walk but turns it into a checkable proof. It explicitly verifies the alternating side assignment, the vector directions, the cumulative coordinates, and finally the shoelace sum:

###### ThinkTwice refinement, step 110 (correct; 633 words)

√2

2 , 315◦ → (12, −12).” After listing the vertices, it computes the shoelace terms and concludes with

“Side 1: length 1, 0◦ → (1,0) ... Side 8: length

|7 2<br><br>|
|---|

.

The same-step pure-GRPO baseline is informative because it fails in exactly the place where ThinkTwice’s refinement succeeds. The baseline refinement is still expanding shoelace terms at the output limit and never boxes a final answer:

Pure-GRPO refinement, step 110 (wrong; 970 words) The last available lines are still inside the term-by-term expansion,

√2 2

√2

√2 2

√2 2

1 +

,

x6y7 − x7y6 = −

2 − −

with no final boxed answer.

This is a useful mechanism example because refinement is not inventing a completely different strategy. The base trace already knows the right global method; refinement makes it executable, removes the indecision at the area step, and finishes the derivation.

Late checkpoint (step 250): same geometry, cleaner ending, exact final form By step 250 the ThinkTwice reasoning answer solves the problem, but the trace is still long and numerically heavy. It ends with the decimal result

|3.5|
|---|

, after a lengthy shoelace calculation. The last few lines read like a running calculator tape:

###### ThinkTwice reasoning, step 250 (correct; 1303 words)

“0 + 0.5 + 1.5 + 1.5 + 2.0 + 1.0 + 0.5 + 0 = 7.0. So total sum = 7.0. Area = 12 × |7.0| =

|3.5|
|---|

.”

The refinement keeps the same coordinate proof but tightens the presentation and restores the exact answer format:

###### ThinkTwice refinement, step 250 (correct; 591 words)

It verifies each side vector once, summarizes the shoelace sum compactly, and then states

|7 2<br><br>|
|---|

. The refinement keeps the proof exact instead of finishing in decimal form.

The pure-GRPO baseline is also correct at this late checkpoint, but its refinement is still substantially longer (884 words versus 591) and includes an extra “reasonableness check” section after the exact shoelace computation. This is the kind of late-stage difference that is easy to miss if one looks only at accuracy: both systems are correct, but ThinkTwice’s

refinement is much closer to the polished proof style we would actually want to show a user.

- D.3 Case C: sample 194 — concrete error repair early, concise proof late Problem and role in the appendix

Problem (MATH500, level 5, sample 194). Find the maximum value of

x − y x4 + y4 + 6

over all real numbers x and y. Ground truth:

|1 4<br><br>|
|---|

.

Why this case is useful. This case separates two different refinement behaviors. At step 30 the base trace reaches the right one-variable reduction but makes a concrete derivative arithmetic error; the refinement explicitly identifies and fixes that error. By step 220 the base trace is correct, yet still wanders through multiple abandoned ideas before it settles on the final proof. The refinement turns that wandering derivation into a much shorter argument. The same-step pure-GRPO baseline is already correct at both checkpoints, so this case is not about a baseline failure; it is included because it shows that ThinkTwice learns both concrete bug repair and later proof compression on the same problem.

Checkpoint Run Base output refinement output

|1 4<br><br>|
|---|

ThinkTwice WRONG; 1204 words; no final boxed answer

CORRECT; 719 words;

Step 30

|1 4<br><br>|
|---|

|1 4<br><br>|
|---|

Pure GRPO CORRECT; 1236 words;

CORRECT; 1181 words;

|1 4<br><br>|
|---|

|1 4<br><br>|
|---|
|1 4<br><br>|

ThinkTwice CORRECT; 1373 words;

CORRECT; 374 words;

Step 220

|1 4<br><br>|
|---|

Pure GRPO CORRECT; 906 words;

CORRECT; 755 words;

- Table 8: Case C summary. Early refinement fixes a concrete derivative mistake; late refinement turns a wandering proof into a short direct argument.

Early checkpoint (step 30): explicit derivative bug repair The ThinkTwice reasoning answer makes a strong start. It correctly reduces the problem to maximizing

d

f(d) =

1 8d4 + 6

by minimizing x4 + y4 at fixed difference d = x − y. The failure is a concrete arithmetic error in the derivative step:

ThinkTwice reasoning, step 30 (wrong; 1204 words) After deriving

f′(d) = −83d4 + 6 (18d4 + 6)2

,

it sets the numerator to zero and writes

−38d4 + 6 = 0 ⇒ d4 = 1, which is wrong; the correct value is d4 = 16.

The refinement is unusually explicit about what went wrong:

###### ThinkTwice refinement, step 30 (correct; 719 words)

“Mistake in previous step: I incorrectly computed d4 = 1 instead of d4 = 16.” It then corrects the derivative equation to

−38d4 + 6 = 0 ⇒ d4 = 16 ⇒ d = 2, substitutes x = 1, y = −1, and concludes with

|1 4<br><br>|
|---|

.

The pure-GRPO baseline already solves this problem at the same early checkpoint, so the point of this example is narrower: ThinkTwice’s refinement learns a genuine bug-fixing behavior, not just a generic “rewrite more clearly” behavior.

###### Late checkpoint (step 220): preserving correctness while removing exploratory clutter

At step 220 the ThinkTwice reasoning answer is correct, but it is still visibly exploratory. It tries multiple proof routes, including one that it explicitly rejects:

###### ThinkTwice reasoning, step 220 (correct; 1373 words) The trace first tries an inequality route,

z4 + 1 ≥ 4z for all real z,

then immediately checks it at z = 1 and notes “1 − 4 + 1 = −2 — not non-negative, so that inequality fails.” Only after this detour does it return to the correct calculus proof.

The refinement starts much closer to the final proof. It opens with the extremal point, then proves optimality directly:

ThinkTwice refinement, step 220 (correct; 374 words) “We try a substitution: let x = 1, y = −1.” It then proves

1 4

x − y x4 + y4 + 6 ≤

by minimizing

f(x, y) = x4 + y4 − 4x + 4y + 6, checks the critical point (1, −1), and ends with

|1 4<br><br>|
|---|

.

The same-step pure-GRPO baseline is also correct, but its refinement is more exploratory and almost twice as long. It begins with several test points (e.g., x = 1, y = 0; x = 1, y = −1; x = 2, y = −1) before moving to the formal proof. This makes the late-stage contrast very clear: once the base policy has learned the right answer, ThinkTwice’s refinement behaves like an exploitation layer that strips away the dead ends.

- D.4 Case D: sample 425 — vector recomputation as a backup geometry case Problem and role in the appendix

Problem (MATH500, level 5, sample 425). Let G and H denote the centroid and orthocenter of triangle ABC, respectively. Let F be the midpoint of GH. Express AF2 + BF2 + CF2 in terms of the side lengths a, b, c and circumradius R.

|3R2|
|---|

.

Ground truth:

Why this case is useful. This is a strong backup geometry example. Early on, the ThinkTwice reasoning trace carries out the right vector setup but leaves the derivation in an unresolved mixed term involving ⃗S · O⃗ . The refinement recomputes the vector expansion carefully and substitutes the missing circumcenter identity. By step 220 the base trace has learned the cleaner circumcenter-at-origin derivation, and the refinement compresses it further.

Checkpoint Run Base output refinement output

|3R2|
|---|

ThinkTwice WRONG; 1138 words; no final boxed answer

CORRECT; 795 words;

Step 100

|3R2|
|---|

|3R2|
|---|

Pure GRPO CORRECT; 711 words;

CORRECT; 1046 words;

|3R2|
|---|

|3R2|
|---|
|3R2|

ThinkTwice CORRECT; 746 words;

CORRECT; 410 words; Pure GRPO CORRECT; 1259 words;

Step 220

|3R2|
|---|

CORRECT; 993 words;

- Table 9: Case D summary. A backup geometry example showing early recomputation and late compression.

Early checkpoint (step 100): refinement repairs the vector algebra The ThinkTwice reasoning answer gets close to the goal but stops at an unresolved mixed term:

ThinkTwice reasoning, step 100 (wrong; 1138 words) After expanding the squared norms, the base trace reduces the expression to

AF2 + BF2 +CF2 = −2(⃗S·O⃗) +3∥O⃗∥2 +∑∥⃗X∥2,

but never substitutes the identity needed to eliminate the remaining ⃗S · O⃗ term.

The refinement explicitly decides to recompute instead of trusting the previous algebra:

###### ThinkTwice refinement, step 100 (correct; 795 words)

“Let’s recompute the expression carefully, term by term.” It recomputes ⃗S · H⃗ and ∥H⃗ ∥2, then uses

###### ∥⃗X∥2 − 2⃗X · O⃗ + ∥O⃗ ∥2 = 3R2

#### ∑

X=A,B,C

|3R2|
|---|

to replace the mixed term and conclude with

.

The pure-GRPO baseline is also correct here, but its refinement is substantially longer and contains its own visible false starts (including the line “Still wrong” while checking an auxiliary identity). This makes the ThinkTwice refinement look less like another exploratory search and more like a controlled repair.

Late checkpoint (step 220): concise invariant-based proof By step 220 the ThinkTwice reasoning answer has already adopted the cleaner circumcenterat-origin derivation. The refinement compresses it to a short invariant argument:

###### ThinkTwice refinement, step 220 (correct; 410 words) It places the circumcenter at the origin, uses

H⃗ = A⃗ + ⃗B + C⃗, G⃗ = 31(A⃗ + ⃗B + C⃗), ⃗F = 23(A⃗ + ⃗B + C⃗), and then cancels the cross terms in

### ∑ AF2 = 3∥⃗F∥2 +∑∥A⃗∥2 −2⃗F·(A⃗ +⃗B+C⃗)

|3R2|
|---|

to get

.

The pure-GRPO baseline is also correct at this checkpoint, but both its base and refinement are much longer (1259 and 993 words). This is a good backup example when we want a geometry case that is less about a wrong final answer and more about the shift from symbolic overhead to a clean invariant.

###### D.5 Key takeaway from the case study

Across all four examples, the same qualitative pattern recurs. Early in training, refinement is most useful because it can change what the model is doing: it can discover a missing structural relation (Case A), finish a derivation that the base trace cannot close (Case B), or repair a concrete algebraic error (Case C). Later in training, once the base policy more often lands on the right route, refinement shifts toward exploitation: it preserves correctness while removing numerical detours, dead-end branches, and unnecessary verification blocks (Cases A–D). This is exactly the qualitative behavior we would expect if joint reasoning-and-refinement training is teaching a reusable refinement policy rather than merely adding another pass of unconstrained generation.

