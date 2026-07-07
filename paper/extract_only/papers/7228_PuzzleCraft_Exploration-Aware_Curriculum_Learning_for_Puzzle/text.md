# arXiv:2512.14944v2[cs.CV]13Mar2026

## PuzzleCraft: Exploration-Aware Curriculum Learning for Puzzle-Based RLVR in VLMs

Ahmadreza Jeddi1,2,3⋆, Hakki C. Karaimer1⋆, Hue Nguyen1, Zhongling Wang1⋆⋆, Ke Zhao1⋆⋆, Javad Rajabi1,2,3⋆⋆, Ran Zhang1⋆⋆, Raghav Goyal1, Konstantinos G. Derpanis1,3,4, Babak Taati2,3, and Radek Grzeszczuk1

1 AI Center-Toronto, Samsung Electronics 2 University of Toronto

- 3 Vector Institute
- 4 York University

ajeddi@cs.toronto.edu, hakki.k@samsung.com

Abstract. RL post-training with verifiable rewards (RLVR) has become a practical route to eliciting chain-of-thought reasoning in vision– language models (VLMs), but scaling it in the visual domain remains challenging due to costly or noisy supervision and reliance on external verifiers. Puzzle-based RLVR is a promising alternative, yet existing approaches often treat puzzle rewards as flat or sparse, which weakens group-relative learning signal. Existing curriculum strategies are overly restrictive: they rely mainly on reward statistics and do not account for exploration in the solution space, which can lead to collapsed rollout dynamics. Further, RL post-training can induce reasoning–answer inconsistency as training progresses. To address these shortcomings, we present PuzzleCraft, a supervision-free framework that scales vision-centric RLVR using a set of lightweight puzzle environments with built-in verification. PuzzleCraft instantiates three puzzles inspired by classic visual pretext tasks: PatchFit, Rotation, and Jigsaw. We introduce a curriculum that combines difficulty with an exploration signal derived from solutionspace dispersion, and use it to downweight collapsed prompt groups. In addition, we introduce a new post-training metric, Reasoning-Answer Consistency (RAC), to measure the degree that the chain-of-though supports the answer, and show our exploration-aware curriculum improves RAC and downstream performance. Across a broad suite of visioncentric benchmarks, PuzzleCraft improves robustness and reasoning consistency, yielding consistent downstream gains on both Qwen2.5-VL and Qwen3-VL backbones. Overall, our results suggest that scalable puzzlebased RLVR benefits from curricula that account for both difficulty and solution-space collapse, together with explicit consistency-enhancing schemes. Project page: https://puzzlecraftgrpo.github.io/

Keywords: VLMs · RLVR · Curriculum · Reasoning consistency

⋆ Equal contribution ⋆⋆ Equal contribution

### 1 Introduction

Recent progress in vision–language models (VLMs) has been driven by RL posttraining, which moves beyond supervised instruction tuning and offers a practical way to elicit stepwise reasoning behaviors [6,26,54,68]. Among these approaches, GRPO-style objectives for RL with verifiable rewards (RLVR) have gained traction [20,37,47,48]. Yet scaling GRPO-style RLVR in the visual domain is bottlenecked by a practical constraint: obtaining vision-centric, reliably verifiable reward signals is costly and noisy, and often requires curated supervision or external verifiers [7,62]. This has motivated growing interest in puzzle environments, self-contained visual tasks with automatic verification, as a promising direction for scalable, supervision-free RLVR [40,41,61,62].

Existing puzzle-based RLVR pipelines leave several gaps that limit how far this direction can go. First, many works focus on a single puzzle in isolation, making it unclear whether gains reflect general improvements in multimodal reasoning or narrow specialization; inter-puzzle transfer and generalization remain under-explored. Second, puzzle RLVR often relies on binary success rewards, which are sparse and do not reward partial progress. Third, rewards are frequently treated as flat across difficulties, so easy, medium, and hard instances contribute with nearly the same influence on updates [69, 74]. This can exacerbate vanishing-advantage dynamics when rollouts within a group become homogeneous, driving group-relative advantages toward zero and weakening learning [54]. Moreover, the existing curriculum solutions in VLM GRPO only consider the reward statistics without considering the rollout exploration, which as we observe, limits the gains. Relatedly, existing curriculum strategies for VLM GRPO [21,28] typically rely only on reward statistics and do not account for whether rollouts continue to explore or instead collapse into repetitive, low-diversity behavior. As we observe, this omission can cap achievable gains. Finally, puzzle training exposes a broader RLVR failure mode: as post-training progresses, models may increasingly shortcut or drift into reasoning–answer inconsistency, where the chain-of-thought (CoT) no longer supports the final answer [9,11,25,64]. Together, these gaps suggest that unlocking the full potential of puzzle-based RLVR requires richer reward signals, curricula that account for both difficulty and rollout collapse, and explicit attention to consistency during optimization.

We present PuzzleCraft, a supervision-free RLVR framework that scales vision-centric post-training by using a set of lightweight puzzle environments, promoting exploration through curriculum design, and monitoring and improving reasoning–answer consistency during training. PuzzleCraft instantiates three tasks inspired by classic pretext objectives for visual pretraining: PatchFit [4,23], Rotation [19], and Jigsaw [40,41,61,62]. Our puzzle environments are fully unsupervised, require no external verifiers or teacher models, and provide automatic rewards at scale.

PuzzleCraft targets three bottlenecks in puzzle-based RLVR. (1) Reward sparsity: we use Jigsaw as a core environment with a graded, partial-credit reward (fraction of correctly placed tiles), which rewards intermediate progress

and penalizes localized errors. (2) Flat rewards and vanishing advantages: we introduce a curriculum that accounts for both difficulty and exploration. The difficulty term upweights medium-hard prompts using reward variance, while the exploration term downweights prompt groups whose rollouts collapse to the same solution. For binary puzzles, exploration is measured by class entropy over selected options; for Jigsaw, it is measured by diversity in the induced permutations, reflecting its combinatorial structure. (3) Consistency drift: we introduce a post-training diagnostic metric, Reasoning–Answer Consistency (RAC). We sample rollouts throughout training and measure whether the reasoning trace supports the final answer, then use RAC to study how our curriculum and consistency-aware rewards affect faithfulness over time.

Empirically, our exploration-aware curriculum consistently outperforms difficulty-only curricula. Ablations further show that both the curriculum and consistency-aware reward schemes [11] improve reasoning–answer consistency and translate to higher downstream accuracy. PuzzleCraft yields reliable gains across model families and evaluation settings. On Qwen2.5-VL-7B, our Jigsaw variant improves average performance across nine image benchmarks, outperforming the strongest puzzle-based baseline by 2.2 points. On Qwen2.5-VL-3B, our best variant improves over baselines by up to 1.39 points. The same trend holds for Qwen3-VL across scales: Jigsaw post-training increases Avg. over the corresponding Qwen3-VL-Instruct checkpoints by +2.6 (2B), +2.99 (4B), and +1.87 (8B). We also observe positive transfer to video reasoning benchmarks despite not using any video data during post-training.

#### Contributions:

- – PuzzleCraft: a supervision-free puzzle-based RLVR framework for scalable vision-centric post-training with a set of lightweight, verifiable environments.
- – Exploration-aware curriculum: dynamic weighting that accounts for both difficulty and rollout collapse, emphasizing medium-hard prompts while encouraging solution-space exploration, mitigating flat rewards and vanishing advantages.
- – Consistency analysis: we introduce RAC as a diagnostic for consistency drift, and show that our exploration-aware curriculum and consistency-aware reward schemes improve both faithfulness and downstream performance.

### 2 Related Work

LLM/VLM RL post-training. RL post-training has played a central role in improving instruction-following and alignment in language models [42, 44]. More recently, RL with verifiable rewards (RLVR) has emerged as a scalable way to induce stepwise reasoning by training on outcomes that can be checked automatically [20]. Group-relative policy optimization (GRPO) [47] and related objectives have been studied from multiple angles, including comparisons to alternative RL formulations [36, 66, 73], how RLVR interacts with supervised fine-tuning (SFT) [6, 12], and training efficiency and scaling behavior [3, 33].

Motivated by these results, recent work has adapted RLVR and GRPO-style post-training to vision–language models (VLMs) [13,15,26,48,54]. Many efforts target multimodal reasoning on math- and science-oriented benchmarks, while others extend RL post-training to more vision-centric settings such as grounding and segmentation [2,35]. Despite this progress, scaling RLVR in the visual domain remains difficult: reliable verification is often task-specific and many pipelines still depend on curated annotations or external verifiers, which limits throughput and portability across tasks.

Towards supervision-free post-training. Obtaining clean ground-truth annotations can be costly, impractical at scale, and noisy, motivating post-training methods that reduce or remove dependence on labeled answers. In LLMs, several directions have been explored, including confidence-based heuristics (e.g., entropy-based selection) [43], majority voting and self-consistency across rollouts [7,75], and training strategies that tolerate imperfect reward signals [46]. For VLMs, verifier-based pipelines [58,59] (e.g., critic/judge models that assess captions or textual responses) and gamified or self-play environments [10, 57] improve perception and reasoning but introduce new costs and biases through external evaluators. Closer to our setting, recent work introduces visual puzzle tasks for post-training VLMs [16,61,62,67]. These studies, while promising, typically cover a narrow puzzle set, rely on vanilla GRPO, and offer limited analysis of training dynamics, difficulty, and generalization. PuzzleCraft instead focuses on these dynamics, introducing an exploration-aware curriculum and consistency monitoring to improve stability and transfer.

Curricula for GRPO post-training. A growing body of work examines failure modes of CoT-enabled VLMs under GRPO-style training [27,31]. From an optimization perspective, vanilla GRPO is largely difficulty-agnostic: when rollouts within a group become homogeneous, group-relative advantages shrink toward zero and updates become weak [21, 56, 60]. Sparse rewards can further worsen this behavior by pushing groups toward all-success or all-failure regimes [11, 63]. Recent approaches mitigate these issues through offline or online curricula and by introducing more efficient or stabilized GRPO variants [22,28,73]. However, these curricula typically rely on reward statistics alone and do not capture whether the policy is exploring diverse solutions or collapsing to a repeated guess. PuzzleCraft extends this line of work by measuring solutionspace dispersion within each rollout group and using it to downweight collapsed groups.

Consistency drift in reasoning. A recently highlighted failure mode of CoT is reasoning–answer inconsistency, where the final answer is not supported by the reasoning trace [49,64]. Prior work reports that faithfulness can improve early during GRPO-style training but later degrade, motivating closer monitoring of post-training dynamics [9]. Complementary methods propose consistency-aware objectives or auxiliary checks that better couple the final answer to the reasoning chain [11,25]. In our setting, we observe the same trend in visual puzzle posttraining: vanilla GRPO exhibits worsening consistency over time. We therefore introduce a simple diagnostic metric tracked throughout training, and find that

[Figure 1]

[Figure 2]

Prompt: Your task is to solve a visual puzzle ...

Puzzles

[Figure 3]

Patchfit Rotation

Jigsaw

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

<Answer>1,3,4,2</Answer> <Answer>90</Answer>

<Answer>B</Answer>

PuzzleCraft

[Figure 9]

R

RL Environment

<think>Theimageprovidedisalreadyinthecorrecttop-to-bottomandleft-to-rightorder...noreorderingis necessary.Theanswerisrepresentingthecorrectsequenceoftheindicesintheoriginalundistortedimage</think>

[Figure 10]

1

...

Exploration-aware Curriculum

- <answer>1, 2, 3, 4<\answer> reward = 0.25

0.25

0.5

reward = 1.0

1.0

- <answer>1, 3, 4, 2</answer>

- 1,2,3,4
- 1,3,2,4

R <think>Uponobservingtheimages,Patch1featuresblueskywithsometrees,indicatingatoporleftmostposition...Therefore,thecorrectordershouldswapPatch3andPatch4thenswapPatch2andPatch3</think>

1,3,4,2 ...

2

[Figure 11]

Reasoning Answer Consistency

...

<think>Thefirstpatch1presentspartoftheskywithsometreeindicatingitslefttopposition....Thethirdpatch3 showsmoreskyandcloud,placingitnexttothefirstforhorizontalcontinuity....Thisarrangementslightlymodifies theoriginallandscapesequencebyreplacingpatch1withpatch3.

<think>

R

K

<answer>

reward = 0.5

<answer>1, 3, 2, 4</answer>

- Fig. 1: PuzzleCraft overview. We sample puzzle prompts and weight them by a curriculum. The model generates a group of candidate solutions, receives automatically verifiable rewards, and updates the policy using a curriculum-weighted GRPO objective. We also track reasoning–answer consistency during post-training and study how curricula and consistency-aware optimization affect both RAC and downstream performance.

our exploration-aware curriculum mitigates the decline. When combined with a lightweight consistency-aware GRPO variant (GRPO-CARE [11]), the gains in both consistency and downstream performance are further amplified.

### 3 PuzzleCraft

This section presents PuzzleCraft, a supervision-free RLVR framework that leverages automatically verifiable puzzle rewards to scale vision-centric posttraining. Recent work has shown that puzzles provide a practical source of automatically verifiable rewards, yet existing pipelines typically follow vanilla GRPO recipes and treat puzzles as purely outcome-driven tasks. Here, we reframe puzzle-based RLVR as a curriculum and consistent reasoning problem rather than a pure reward-maximization objective. Specifically, we consider multiple vision-centric puzzle environments and focus on two aspects that become central in this setting: (i) how to weight prompts so that GRPO updates concentrate on informative, medium-hard instances while avoiding rollout collapse, and (ii) how to monitor and improve reasoning–answer consistency, which can degrade during post-training even as puzzle rewards continue to rise.

Our approach has three components. First, we instantiate a set of lightweight puzzle environments with built-in verification. Second, we introduce a curriculum that is both difficulty-aware and exploration-aware: in addition to mean success, it detects when rollouts within a prompt group collapse to the same solution, and it downweights such groups to mitigate vanishing-advantage updates. Third, we track a simple Reasoning–Answer Consistency (RAC) metric during post-training, and optionally incorporate GRPO-CARE [11] as an add-on to explicitly shape consistency. Figure 1 provides an overview of our approach.

#### 3.1 Verifiable rewards via puzzle environments

Motivated by classic self-supervised pretext tasks for pretraining, we instantiate three programmatically verifiable puzzle environments (see Figure 1): Rotation [19] (predict an angle from a fixed set), PatchFit (identify a masked patch among confusable candidates), and Jigsaw [40,41,61,62] (assign tiles to grid positions under a valid permutation). Rotation and PatchFit yield binary rewards, r ∈ {0,1}, via exact checks.

Jigsaw has a property that we explicitly leverage. It naturally supports partial credit enabling graded supervision without process-level labels. We use a reward, r ∈ [0,1], equal to the fraction of tiles placed correctly. This rewards intermediate progress and penalizes localized errors without collapsing the entire rollout to failure.

Each puzzle exposes a controllable difficulty parameter: grid size for Jigsaw, angle-set cardinality for Rotation, and distractor hardness for PatchFit. In our experiments, we keep these settings fixed when studying curricula, and analyze puzzle choice and inter-puzzle transfer separately in §4.

#### 3.2 Difficulty and exploration aware curriculum

For each prompt, x, we sample a group of G rollouts, {oi}Gi=1, with rewards ,{ri}. We assign the group a weight, w(x), so that training focuses on prompts that yield informative GRPO updates.

Why difficulty alone is insufficient. Difficulty-aware curricula in RLVR often rely on reward statistics such as mean success or reward variance, which emphasize medium-difficulty prompts. This is necessary, but in puzzle RL it does not distinguish whether the model is exploring different candidate solutions or repeatedly collapsing to the same guess. As a simple example with a multiple-choice verifier and ground-truth option A and incorrect options B −F, the groups {A,B,B,B,B,B,B,B} and {A,B,B,C,C,D,E,F} have the same mean success, yet the first reflects strong collapse to a single incorrect option. Collapsed groups are particularly problematic for GRPO because homogeneous rollouts cause group-relative advantages to shrink, even when the mean success suggests the prompt is nontrivial. This highlights that reward statistics alone are insufficient to characterize training signal quality in puzzle-based RLVR.

Difficulty signal. We use the group mean reward as the primary difficulty signal,

G

d(x) = r¯ = G1

ri ∈ [0,1],

i=1

where d ≈ 1 indicates easy prompts and d ≈ 0 indicates hard prompts. The term d(1 − d) peaks at medium difficulty and vanishes at the extremes, matching the regime where GRPO provides the most informative learning signal.

Exploration signal. To detect collapse, we measure the diversity in the puzzle solution space within the rollout group. The definition is puzzle-specific but

follows a common principle: extract a discrete solution key, si, from each rollout and compute a normalized dispersion statistic, dπ(x) ∈ [0,1].

Binary puzzles (Rotation, PatchFit). Each rollout selects a discrete option (angle class for Rotation, candidate index for PatchFit). Let si ∈ {1,...,K} denote the selected option and pˆk = G1 Gi=1 1[si = k] be the empirical frequency. We define the normalized solution entropy as

dπ(x) = − Kk=1 pˆk log pˆk log K

,

where, dπ ≈ 0 indicates that rollouts collapse to a single option, while larger values indicate broader exploration across options.

Jigsaw. Jigsaw is a structured puzzle with partial grading, where we score each rollout by the fraction of correctly placed tiles. A useful side effect of this grading is that multiple distinct permutations can yield the same reward, so reward statistics alone can miss whether the model is collapsing to a single arrangement. We therefore track exploration directly in permutation space. Each rollout induces a permutation over grid positions. Let Π(oi) ∈ S denote the induced permutation, and define

M − 1 G − 1

M = {Π(oi)}Gi=1 , dπ(x) =

.

This quantity approaches 0 when rollouts collapse to the same arrangement and increases as the group explores distinct permutations, including cases where rollouts achieve similar partial-credit rewards through different tile placements. Curriculum weight. We combine difficulty and exploration into a single group weight:

w(x) = λd(x) 1 − d(x) dπ(x) γ, (1) where we use fixed values of λ = 4 and γ = 0.5 in all experiments. The term d(x)(1−d(x)) emphasizes medium difficulty and becomes zero when all rollouts succeed or all fail. The term dπ(x) downweights groups whose rollouts collapse to the same solution, which are most prone to weak or uninformative grouprelative updates. For Rotation and PatchFit, dπ is computed from class entropy; for Jigsaw, it is computed from permutation diversity.

Optimization objective. We adopt a token-level GRPO objective with curriculum weighting. We set the KL-to-reference coefficient to zero (β = 0), following recent guidance [11,21] that strong KL anchoring can over-constrain exploration in GRPO-style post-training:

i}Gi=1∼πθold(·|q) (2)

JPuzzleCraft(θ) = E(q,a)∼D,{o

|oi|

G

min ρi,t Aˆi,t, ρ˜i,t Aˆi,t ,

1 |oi|

1 G

w(d(q))

t=1

i=1

curriculum

where given ϵ > 0:

Ai = ri − r,¯ Aˆi,t = Ai,

πθ oi,t | q,oi,<t πθ

, ρ˜i,t = clip ρi,t, 1−ϵ, 1+ϵ .

ρi,t =

oi,t | q,oi,<t

old

#### 3.3 Consistency monitoring and consistency-aware optimization

Recent work highlighted that RL post-training can introduce a gap between the reasoning trace and the final answer, and that this gap may widen over training [9, 11]. For puzzle RL, this is easy to miss if one only monitors puzzle reward, since reward can improve while faithfulness degrades. We therefore monitor Reasoning–Answer Consistency (RAC) as a training-dynamics signal.

RAC metric (diagnostic only). To monitor reasoning–answer alignment during post-training, we periodically sample rollouts and compute RAC as a diagnostic signal. Specifically, we prompt a fixed open-source judge (Qwen2.5-VL72B in inference mode) with each rollout’s rationale and final <answer>, and ask whether the rationale explicitly supports the emitted answer. Each trial is scored in [0,1] and we report a moving average over training steps. Importantly, RAC is not used for reward shaping, sample reweighting, or gradient updates in our training pipeline. Empirically, RAC tends to increase early in training and may drift downward later, consistent with observations in GRPO-style post-training. In §C we provide the full evaluation setup, including the exact prompts and scoring procedure used to compute RAC.

Consistency-aware add-on. Consistency shaping is complementary to curriculum design, so we optionally incorporate GRPO-CARE [11] as an add-on to PuzzleCraft. Empirically, our curriculum mitigates late-stage RAC degradation, and adding CARE further improves both RAC and downstream performance. We use RAC as a diagnostic throughout and study its interaction with reward design and curricula in §4.

### 4 Experiments

We first describe our implementation details and evaluation setup. We then run ablations to study the main design choices in PuzzleCraft. Finally, using the bestperforming configuration, we train models across different scales and backbones and evaluate them on a broad suite of vision-centric benchmarks.

#### 4.1 Experimental Setup

Training data. We train on the COCO 2014 [32] training split (82,783 images). We choose COCO for reproducibility and to avoid introducing new images beyond those already seen during base-model pretraining, which helps isolate the effect of GRPO in our setting. For each image, we synthesize a single puzzle instance (Jigsaw, Rotation, or PatchFit). Unless stated otherwise, each model is trained on 82,783 puzzle prompts. We apply no additional preprocessing or filtering. Examples are shown in Fig. 1 (top), and full puzzle and prompt details are provided in §A.

Training framework and models. We initialize from Qwen2.5-VL-Instruct and Qwen3-VL-Instruct checkpoints [1]. Unless stated otherwise, ablations are conducted with Qwen2.5-VL-7B-Instruct. We build on the public VLM-R1 training code for GRPO post-training. When enabling the consistency-aware variant GRPO-CARE, we use the authors’ released implementation and match their hyperparameters where possible. All training runs are performed on 8×A100 (80GB) GPUs.

Hyperparameters. Unless noted, we follow VLM-R1 defaults with two changes: KL coefficient β=0 and learning rate 5×10−7. We use batch size 16 for VLM-R1 and batch size 8 for GRPO-CARE. All runs are trained for 1 epoch with maximum decoding length 2048. Each prompt uses G=8 rollouts with temperature 0.9 and top-p 0.95, using one GRPO iteration per update. We train in bfloat16 and cap vision-encoder tokens at 1024 during post-training. We set the PPO clipping parameter ϵ=0.2 for VLM-R1. When using GRPO-CARE, we adopt the authors’ defaults (ref_ema_decay 0.995, EMA update every 10 steps, bonus coefficient 0.5, confidence upper bound 0.95, consistency margin 0.01) and set ϵ=0.

Evaluation setup. We evaluate primarily with VLMEvalKit [14]. Unless otherwise noted, all results are reported in thinking mode: we append a standardized prompt to elicit think→answer formatting, and use Qwen-VL-2.5-72B within VLMEvalKit for post-processing and format checking. The base prompt is:

First output the thinking process in <think></think> tags and then output the final answer in <answer></answer> tags.

Qwen3-VL-Instruct models do not support <think></think>, so we instead use <analyze></analyze> for the reasoning trace.

Baselines. In addition to the underlying base models, we compare against a broad set of recent RLVR frameworks with publicly released checkpoints. Our main points of comparison are puzzle-based methods, including ViCrit [59], Vision-Zero [57], Visual Jigsaw [62], Jigsaw-R1 [61], VisualSphinx [16], and Game-RL [51]. We also report results against annotated RLVR approaches such as Vision-R1 [26], VL-Rethinker [54], Video-R1 [15], ViGoRL [45], GRPOCARE [11], and VLM-R1 [48].

Image benchmarks. We evaluate on a wide range of multimodal benchmarks. For images, our suite includes math-oriented reasoning tasks (MathVista [38], MathVision [55], MathVerse [70]) and vision-centric VQA benchmarks (MME [17], MMStar [8], POPE [30], MMT-Bench [65], CVBench-2D [52], and MMVP [53]).

Video benchmarks. For video understanding and reasoning, we evaluate on VideoMME [18], TempCompass [34], Video-TT [71], MVBench [29], Q-BenchVideo [72], and Video-MMMU [24], and CG-Bench [5].

#### 4.2 Ablating design choices

We ablate the main components of PuzzleCraft to understand their individual contributions and identify the most effective configuration for puzzle-based

- Table 1: Ablations of PuzzleCraft design choices on vision-centric benchmarks. consistency variants (CL (curriculum), CARE, and their combination). Avg. denotes the mean across benchmarks; MME is normalized by dividing by 2800.

|Design choice|MME MMStar POPE MMT CV-Bench MMVP<br><br>|Avg.|
|---|---|---|
|Qwen2.5-VL-Instruct<br><br>|2243 64.67 81.77 59.64 73.62 77.67<br><br>|72.91|

Consistency

Jigsaw 2340 64.47 85.17 59.96 72.13 75.33 73.44 Jigsaw w/ CARE 2319 65.80 86.95 61.18 77.76 77.00 75.25 Jigsaw w/ CL 2365 64.27 84.35 62.62 75.37 74.67 74.30 Jigsaw w/ CL + CARE (Main) 2366 64.60 86.52 62.26 77.63 78.67 75.70

RLVR. Once this setup is established, we apply it across different model families and puzzle environments. All the ablations are done with Qwen2.5-VL-7B and on the Jigsaw puzzle environment on six vision-centric image benchmarks.

Reasoning–Answer Consistency. We study how reasoning–answer consistency evolves during post-training using the RAC metric from §3.3. We compare four variants: vanilla GRPO, GRPO with our curriculum, GRPO-CARE, and curriculum+GRPO-CARE. Fig. 2 tracks RAC alongside puzzle reward, reward variance, and response length over training.

Vanilla GRPO shows the clearest consistency drift: RAC increases early, then degrades later even as puzzle reward continues to improve, consistent with observations in prior work on GRPO-style post-training [9]. Adding our curriculum mitigates the late-stage decline, while GRPO-CARE further improves RAC. The combined setup maintains the highest RAC through most of training, suggesting that curriculum weighting and consistency-aware optimization are complementary in puzzle-based RLVR.

Table 1 reports downstream performance across six vision-centric benchmarks. Both the curriculum and consistency-aware optimization improve average accuracy. Notably, curriculum+GRPO-CARE outperforms GRPO-CARE alone, despite GRPO-CARE achieving higher puzzle reward during post-training. This highlights that higher puzzle reward is not, by itself, a reliable indicator of downstream reasoning performance.

Curricula. Most puzzle-based RLVR frameworks treat rewards as flat across difficulty levels. As noted in prior work, this can trigger vanishing-advantage dynamics: rollouts within a group become homogeneous, the group-relative signal collapses, and learning slows or stalls, wasting compute. A common remedy is difficulty-aware reweighting that emphasizes medium-difficulty samples during post-training. However, as discussed in §3.2, these curricula typically rely only on reward statistics and do not reflect whether rollouts are actively exploring the solution space or have collapsed to a repeated guess. PuzzleCraft instead uses an exploration-aware curriculum that couples sample difficulty with a solutionspace dispersion signal.

We compare our curriculum against a representative difficulty-aware baseline, Observe-R1 [21], with results reported in §B. On Jigsaw, the explorationaware curriculum achieves the strongest downstream performance and outper-

PC-GRPO variants

- 0.30

Base Jigsaw

Jigsaw + Curriculum

Jigsaw + GRPO-CARE

Jigsaw + Curriculum + GRPO-CARE

0.25

RewardStd

0.20

0.15

0.10

0.05

1000 2000 3000 4000 5000

Training step

(a) Reward variance

600

500

ResponseLength

400

300

200

100

1000 2000 3000 4000 5000

Training step

(c) Response length (in tokens)

| | |
|---|---|
| | |
| | |

1.0

0.8

0.6

RAC

0.4

0.2

0.0

0 1000 2000 3000 4000 5000

Training step

###### (b) Reasoning-answer consistency

0.6

RewardScore

0.5

0.4

0.3

Random ≈ 0.26

1000 2000 3000 4000 5000

Training step

(d) Reward score

- Fig. 2: Tracking GRPO metrics during post-training across four puzzle environments. All charts report a moving average with window size of 100 over training steps. (a) Variance among the rollout rewards (b) Consistency rate between rollout reasoning and final answer, measured by Qwen2.5-VL-72B model (c) Average numbers of tokens decoded by each trajectory (d) Reward score which is the partially graded Jigsaw solution reward.

forms Observe-R1 under the same training setup. This suggests that curricula which explicitly track rollout exploration, rather than reward statistics alone, can yield more effective puzzle-based post-training.

#### 4.3 Main Results

Guided by the analysis above, we apply our best-performing configuration (curriculum+GRPO-CARE) to Rotation and PatchFit in addition to Jigsaw. Since different puzzles emphasize different skills, we also train a mixed setting to study multi-puzzle post-training. Concretely, we sample 40K training instances (15K Jigsaw, 15K PatchFit, 10K Rotation) and run post-training.

Table 2 reports results on nine image benchmarks spanning math reasoning and vision-centric VQA for Qwen2.5-VL and Qwen3-VL across multiple model sizes. Table 3 further compares our models against baselines on 7 video reasoning benchmarks.

Findings. Table 2 and Table 3 highlight the following trends:

– PuzzleCraft improves puzzle-based RLVR. PuzzleCraft models, especially Jigsaw and Mix, achieve the strongest results among puzzle-based approaches. On image benchmarks, Qwen2.5-VL-7B-Jigsaw reaches 65.18% Avg., improving by more than 2.2% over the closest puzzle baseline (Visual Jigsaw). Similar gains hold for Qwen2.5-VL-3B and across Qwen3-VL sizes.

- Table 2: Results on image reasoning benchmarks for Qwen2.5-VL and Qwen3-VL backbones. We compare supervised RLVR baselines, puzzle-based RLVR baselines, and PuzzleCraft variants (single-puzzle and Mix). Avg. is the mean over all 9 benchmarks, with MME normalized as (MME/2800) × 100 before averaging. PuzzleCraft delivers consistent gains across model sizes and backbones, with Jigsaw and Mix outperforming prior puzzle-based RLVR and remaining competitive with supervised RLVR baselines.

Math Image VQA

Model

MathVista MathVision MathVerse MME MMStar POPE MMT CV-Bench MMVP Avg. Qwen2.5-VL-7B

###### Baselines

|Qwen 2.5 VL (Vanilla) Vision-R1 (ICLR26) VL-Rethinker (NeurIPS25) GRPO-CARE (25.06)<br><br>|66.30 23.68 56.49 68.89 39.47 59.99 72.39 29.90 67.29 68.70 20.39 47.71<br><br>|2243 64.67 81.77 59.64 73.62 77.67 2292 62.33 88.41 59.70 72.17 77.00 2311 63.06 83.62 61.01 76.90 77.33 2352 64.13 88.18 62.62 74.91 80.33|64.88 67.76 68.23 65.66<br><br>|
|---|---|---|---|
|ViCrit (NeurIPS25) Vision-Zero (ICLR26) Visual Jigsaw (ICLR26) VisualSphinx (25.05) Game-RL (ICLR26)|61.40 18.75 38.02<br><br>66.20 21.05 45.86<br><br>67.50 29.27 57.50<br><br><br>67.80 26.31 53.90 67.40 24.01 58.00<br><br>|2167 62.27 80.50 59.83 70.84 75.33 2248 63.47 82.67 60.47 73.53 78.00 2243 62.53 84.78 57.76 74.46 76.33 2296 63.20 83.93 60.63 73.98 77.33 2229 64.60 81.77 61.27 75.24 77.66|60.48 63.50 65.58 65.45 65.51<br><br>|

Our Variants Jigsaw 68.20 30.92 56.70 2366 64.60 86.52 62.26 77.63 78.67 67.78 PatchFit 68.03 26.31 50.89 2316 59.87 85.05 58.94 73.37 78.00 64.80 Rotation 71.70 22.69 57.70 2357 64.60 87.36 61.91 75.08 79.67 67.21 Mix 68.20 24.01 58.10 2359 65.20 85.40 62.65 76.96 78.33 67.01 Qwen2.5-VL-3B

###### Baselines

|Qwen 2.5 VL (Vanilla) VLM-R1 (25.04) ViGoRL (NeurIPS25)<br><br>|57.19 21.38 38.30 57.99 19.73 40.80 56.10 18.42 34.60<br><br>|2180 54.73 77.41 53.31 65.62 63.33 2207 55.20 79.66 52.06 66.65 69.67 1919 50.46 84.75 54.90 79.21 68.66<br><br>|56.57 57.84 57.29|
|---|---|---|---|
|Jigsaw-R1 (TMLR25)<br><br>|58.80 22.69 40.30<br><br>|2184 55.53 78.05 57.53 70.87 69.66|59.05|

Our Variants Jigsaw 58.09 18.42 44.70 2223 55.40 78.68 57.88 71.33 68.67 59.17 Mix 60.60 24.01 49.00 2127 57.53 77.30 57.72 72.88 69.0 60.44 Qwen3-VL

|Qwen3-VL-2B-Instruct Qwen3-VL-2B-Jigsaw (Ours)|40.69 15.13 28.10 42.00 19.07 37.50<br><br>|2072 46.6 82.88 53.53 71.36 70.33 2076 48.86 84.81 56.53 72.10 71.00<br><br>|53.62 56.22<br><br>|
|---|---|---|---|
|Qwen3-VL-4B-Instruct Qwen3-VL-4B-Jigsaw (Ours)|51.63 22.03 44.30 53.40 28.28 51.90<br><br>|2138 53.40 87.57 58.77 73.98 75.66 2207 55.80 87.56 61.24 76.55 76.33|60.41 63.32<br><br>|
|Qwen3-VL-8B-Instruct Qwen3-VL-8B-Jigsaw (Ours)|57.40 23.68 47.90<br><br>58.19 28.28 55.20<br><br><br>|2243 57.46 86.25 59.96 76.84 75.33 2227 57.66 86.58 61.72 76.56 78.00<br><br>|62.77 64.64|

On video benchmarks, Qwen2.5-VL-7B-Jigsaw also outperforms the strongest puzzle baseline (VisualSphinx) by about 1.5%. These results suggest that puzzle-based RLVR still has meaningful headroom when paired with curricula that avoid rollout collapse and consistency-aware optimization.

– Competitive with curated RLVR baselines. Despite using only supervision-free puzzle training, our models are competitive with methods trained using large-scale curated RLVR data. For Qwen2.5-VL-7B, Jigsaw improves over GRPO-CARE by about 2% on Avg. and is comparable to Vision-R1 and VL-Rethinker overall, trailing mainly on math benchmarks while improving on vision-centric ones. For Qwen2.5-VL-3B, Mix outperforms curated RLVR baselines (VLM-R1 and ViGoRL) by more than 2.5%. On video

- Table 3: Results on the seven video reasoning benchmarks. We compare PuzzleCraft models against recent RLVR baseline; Avg. denotes the mean across video benchmarks. PuzzleCraft gains transfer to video reasoning without video post-training, surpassing puzzle-based baselines and approaching video-supervised methods on several benchmarks.

|Method<br><br>|CGBench Video-MMMU MVBench TempCompass Video-MME Video-TT QBench-Video<br><br>|Avg.|
|---|---|---|
|Qwen2.5-VL-Instruct<br><br>|26.71 33.83 55.80 55.00 65.40 34.40 58.50|47.09|

Baselines

VisualJigsaw 31.56 32.33 55.60 66.26 66.20 37.40 57.48 49.54 VisualSphinx 31.17 43.16 57.40 70.00 72.00 36.40 57.14 52.46 Video-R1 (NeurIPS25) 37.00 40.16 63.40 78.20 72.20 39.40 55.44 55.11 ViCrit 35.85 34.00 58.40 54.40 72.20 36.60 56.80 49.75 GRPO-CARE 33.94 34.50 62.00 67.60 62.80 38.80 54.76 50.62 Vision-Zero 27.48 37.50 60.00 51.20 67.80 34.40 58.16 48.07 Our variants

Jigsaw 36.33 38.00 59.00 75.40 71.40 38.60 58.84 53.93 Mix 34.86 42.66 61.00 71.80 71.00 34.59 56.12 53.14

benchmarks, Qwen2.5-VL-7B-Jigsaw is close to Video-R1 despite not using any video data during post-training.

– Puzzle choice and transfer matter. Different puzzles transfer differently to downstream tasks. Jigsaw performs best overall, which we attribute in part to its graded, partial-credit reward. Rotation also transfers well, but its gains are benchmark-dependent and complementary to Jigsaw. PatchFit transfers weakly on both math and vision-centric benchmarks. Mix is a robust choice across tasks..

#### 4.4 Direct Inference (Non-Thinking)

All experiments in §4.3 are reported under a CoT-inducing prompt. Here we also evaluate in direct inference mode, using default benchmark prompts that do not request an explicit reasoning trace. We focus on the Qwen2.5-VL-7B backbone and compare against representative baselines. Results are shown in Table 4.

Consistent with prior observations on Qwen backbones and RLVR-tuned variants, some benchmarks achieve higher accuracy in direct mode than with CoT. Understanding this discrepancy remains an active research topic. Overall, our models are competitive with or outperform baselines across vision-centric tasks. At the same time, direct-mode results are generally close to the base model, suggesting that most RL gains in our setting emerge when the model is prompted to produce an explicit reasoning trace.

#### 4.5 Performance on Puzzles

Motivated by the reported weakness of LLMs/VLMs on puzzle environments [39, 50], we evaluate how well our puzzle training improves model performance and whether the learned skills transfer across puzzle types. To create the test set, we randomly select 1000 samples from the test images of COCO2014 and create Jigsaw, PatchFit, and Rotation puzzles. As shown in Table 5, we clearly observe the challenges of reasoning models with our setup as well. Specially, we can see

- Table 4: Performance of 7B models on vision-centric benchmarks in direct-mode prompting (no explicit CoT). Avg. is the mean across benchmarks. Although directmode results are generally close to the base model, our Mix setup achieves the best overall Avg., suggesting that most RLVR gains emerge when models are prompted to produce an explicit reasoning trace.

|Model<br><br>|MME MMStar POPE MMT CV-Bench MMVP<br><br>|Avg.|
|---|---|---|
|Qwen2.5-VL-Instruct|2308 63.27 86.37 62.74 76.29 78.33|74.90|

Baselines

ViCrit 2178 64.13 85.96 62.58 76.59 76.33 73.90 Vision-Zero 2306 63.53 85.92 63.06 76.00 77.67 74.76 Visual Jigsaw 2313 64.13 86.37 62.20 76.47 79.00 75.13 VisualSphinx 2341 63.73 86.34 62.65 76.76 77.33 75.07 GRPO-CARE 2355 63.93 86.85 63.58 75.81 78.00 75.38 Our variants

Jigsaw 2348 64.33 86.05 63.70 76.12 77.33 75.23 Mix 2371 64.73 86.65 64.09 77.36 78.67 76.03

that post-training on a puzzle environment improves performance on that one, but the gains do not transfer to other puzzle setups, but the performance even degrades compared to the Qwen baseline, a mixture of environments alleviates this problem, however, there is still no reliable way to indicate whether learning one skills transfers to others.

- Table 5: Evaluating inter-puzzle transferability across our three puzzle environments. Training on a specific puzzle consistently improves performance, but gains do not transfer to others. Mixed-puzzle training yields strong gains across all tasks.

|Model|Jigsaw PatchFit Rotation<br><br>|
|---|---|
|Qwen2.5-VL-Instruct<br><br>|25.59 21.2 53.3|
|Jigsaw PatchFit Rotation Mix<br><br>|36.65 21.7 33.9 17.88 62.6 51.8 25.55 20.9 70.8 36.83 48.6 83.2<br><br>|

### 5 Conclusion

We studied puzzle environments as a scalable source of supervision-free RL with verifiable rewards (RLVR) for vision–language model post-training. We introduced PuzzleCraft, which combines lightweight, automatically verifiable puzzles with an exploration-aware curriculum that accounts for both difficulty and rollout collapse, and a simple Reasoning–Answer Consistency (RAC) diagnostic with optional consistency-aware optimization. Across Qwen2.5-VL and Qwen3VL backbones, PuzzleCraft improves robustness and consistency on a broad suite

of image benchmarks, and transfers to video reasoning without using video data during post-training.

Our results suggest that progress in puzzle-based RLVR is limited less by the availability of puzzles and more by training dynamics: flat weighting and collapsed rollouts can waste compute, and reward alone can be a poor proxy for faithful reasoning. Looking forward, we believe puzzle-based RLVR can be strengthened by expanding to richer families of verifiable puzzles, improving exploration signals for structured outputs, and further developing consistency diagnostics and objectives that anticipate downstream generalization.

### References

- 1. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923 (2025) 9
- 2. Bai, S., Li, M., Liu, Y., Tang, J., Zhang, H., Sun, L., Chu, X., Tang, Y.: UniVGR1: Reasoning Guided Universal Visual Grounding with Reinforcement Learning. arXiv preprint arXiv:2505.14231 (2025) 4
- 3. Cai, Y., Cai, S., Shi, Y., Xu, Z., Chen, L., Qin, Y., Tan, X., Li, G., Li, Z., Lin, H., Mao, Y., Li, K., Sun, X.: Training-Free Group Relative Policy Optimization. arXiv preprint arXiv:2510.08191 (2025) 3
- 4. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9650–9660 (2021) 2
- 5. Chen, G., Liu, Y., Huang, Y., He, Y., Pei, B., Xu, J., Wang, Y., Lu, T., Wang, L.: Cg-bench: Clue-grounded question answering benchmark for long video understanding. arXiv preprint arXiv:2412.12075 (2024) 9
- 6. Chen, H., Tu, H., Wang, F., Liu, H., Tang, X., Du, X., Zhou, Y., Xie, C.: SFT or RL? An Early Investigation into Training R1-Like Reasoning Large VisionLanguage Models. arXiv preprint arXiv:2504.11468 (2025) 2, 3
- 7. Chen, L., Prabhudesai, M., Fragkiadaki, K., Liu, H., Pathak, D.: Self-Questioning Language Models. arXiv preprint arXiv:2508.03682 (2025) 2, 4
- 8. Chen, L., Li, J., Dong, X., Zhang, P., Zang, Y., Chen, Z., Duan, H., Wang, J., Qiao, Y., Lin, D., et al.: Are We on the Right Way for Evaluating Large VisionLanguage Models? Advances in Neural Information Processing Systems 37, 27056– 27087 (2024) 9
- 9. Chen, Y., Benton, J., Radhakrishnan, A., Uesato, J., Denison, C., Schulman, J., Somani, A., Hase, P., Wagner, M., Roger, F., Mikulik, V., Bowman, S.R., Leike, J., Kaplan, J., Perez, E.: Reasoning Models Don’t Always Say What They Think. arXiv preprint arXiv:2505.05410 (2025) 2, 4, 8, 10, 3
- 10. Chen, Y., Shen, Y., Huang, W., Zhou, S., Lin, Q., Cai, X., Yu, Z., Bu, J., Shi, B., Qiao, Y.: Learning Only with Images: Visual Reinforcement Learning with Reasoning, Rendering, and Visual Feedback. arXiv preprint arXiv:2507.20766 (2025) 4
- 11. Chen, Y., Ge, Y., Wang, R., Ge, Y., Cheng, J., Shan, Y., Liu, X.: GRPOCARE: Consistency-Aware Reinforcement Learning for Multimodal Reasoning. arXiv preprint arXiv:2506.16141 (2025) 2, 3, 4, 5, 7, 8, 9

- 12. Chu, T., Zhai, Y., Yang, J., Tong, S., Xie, S., Schuurmans, D., Le, Q.V., Levine, S., Ma, Y.: SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training. arXiv preprint arXiv:2501.17161 (2025) 3
- 13. Deng, Y., Bansal, H., Yin, F., Peng, N., Wang, W., Chang, K.W.: OpenVLThinker: An Early Exploration to Complex Vision-Language Reasoning via Iterative SelfImprovement. arXiv preprint arXiv:2503.17352 (2025) 4
- 14. Duan, H., Yang, J., Qiao, Y., Fang, X., Chen, L., Liu, Y., Dong, X., Zang, Y., Zhang, P., Wang, J., et al.: Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In: Proceedings of the 32nd ACM international conference on multimedia. pp. 11198–11201 (2024) 9
- 15. Feng, K., Gong, K., Li, B., Guo, Z., Wang, Y., Peng, T., Wu, J., Zhang, X., Wang, B., Yue, X.: Video-R1: Reinforcing Video Reasoning in MLLMs. arXiv preprint arXiv:2503.21776 (2025) 4, 9
- 16. Feng, Y., Xu, Z., Jiang, F., Li, Y., Ramasubramanian, B., Niu, L., Lin, B.Y., Poovendran, R.: VisualSphinx: Large-Scale Synthetic Vision Logic Puzzles for RL. arXiv preprint arXiv:2505.23977 (2025) 4, 9
- 17. Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al.: MME: A comprehensive evaluation benchmark for multimodal large language models. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track (2025) 9
- 18. Fu, C., Dai, Y., Luo, Y., Li, L., Ren, S., Zhang, R., Wang, Z., Zhou, C., Shen, Y., Zhang, M., et al.: Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 24108–24118 (2025) 9
- 19. Gidaris, S., Singh, P., Komodakis, N.: Unsupervised Representation Learning by Predicting Image Rotations. arXiv preprint arXiv:1803.07728 (2018) 2, 6
- 20. Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al.: Deepseek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv preprint arXiv:2501.12948 (2025) 2, 3
- 21. Guo, Z., Hong, M., Jin, T.: Observe-R1: Unlocking Reasoning Abilities of MLLMs with Dynamic Progressive Reinforcement Learning. arXiv preprint arXiv:2505.12432 (2025) 2, 4, 7, 10
- 22. Hammoud, H.A.A.K., Alhamoud, K., Hammoud, A., Bou-Zeid, E., Ghassemi, M., Ghanem, B.: Train Long, Think Short: Curriculum Learning for Efficient Reasoning. arXiv preprint arXiv:2508.08940 (2025) 4
- 23. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked autoencoders are scalable vision learners. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16000–16009 (2022) 2
- 24. Hu, K., Wu, P., Pu, F., Xiao, W., Zhang, Y., Yue, X., Li, B., Liu, Z.: VideoMMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos. arXiv preprint arXiv:2501.13826 (2025) 9
- 25. Huang, M., Huang, R., Zheng, C., Li, J., Chen, G., Shi, H., Cheng, H.: AnswerConsistent Chain-of-thought Reinforcement Learning For Multi-modal Large Langauge Models. arXiv preprint arXiv:2510.10104 (2025) 2, 4
- 26. Huang, W., Jia, B., Zhai, Z., Cao, S., Ye, Z., Zhao, F., Xu, Z., Hu, Y., Lin, S.: Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models. arXiv preprint arXiv:2503.06749 (2025) 2, 4, 9
- 27. Jiang, D., Zhang, R., Guo, Z., Li, Y., Qi, Y., Chen, X., Wang, L., Jin, J., Guo, C., Yan, S., Zhang, B., Fu, C., Gao, P., Li, H.: MME-CoT: Benchmarking Chainof-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency. arXiv preprint arXiv:2502.09621 (2025) 4

- 28. Jiang, G., Feng, W., Quan, G., Hao, C., Zhang, Y., Liu, G., Wang, H.: VCRL: Variance-based Curriculum Reinforcement Learning for Large Language Models. arXiv preprint arXiv:2509.19803 (2025) 2, 4
- 29. Li, K., Wang, Y., He, Y., Li, Y., Wang, Y., Liu, Y., Wang, Z., Xu, J., Chen, G., Luo, P., et al.: MVBench: A Comprehensive Multi-modal Video Understanding Benchmark. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22195–22206 (2024) 9
- 30. Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W.X., Wen, J.R.: Evaluating Object Hallucination in Large Vision-Language Models. In: The 2023 Conference on Empirical Methods in Natural Language Processing (2023) 9
- 31. Liao, Y.H., Elflein, S., He, L., Leal-Taixé, L., Choi, Y., Fidler, S., Acuna, D.: LongPerceptualThoughts: Distilling System-2 Reasoning for System-1 Perception. arXiv preprint arXiv:2504.15362 (2025) 4
- 32. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference on computer vision. pp. 740–755. Springer (2014) 8
- 33. Lin, Z., Lin, M., Xie, Y., Ji, R.: CPPO: Accelerating the Training of Group Relative Policy Optimization-Based Reasoning Models. arXiv preprint arXiv:2503.22342

(2025) 3

- 34. Liu, Y., Li, S., Liu, Y., Wang, Y., Ren, S., Li, L., Chen, S., Sun, X., Hou, L.: TempCompass: Do Video LLMs Really Understand Videos? arXiv preprint arXiv:2403.00476 (2024) 9
- 35. Liu, Y., Peng, B., Zhong, Z., Yue, Z., Lu, F., Yu, B., Jia, J.: Seg-Zero: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement. arXiv preprint arXiv:2503.06520 (2025) 4
- 36. Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., Lee, W.S., Lin, M.: Understanding R1-Zero-Like Training: A Critical Perspective. arXiv preprint arXiv:2503.20783

(2025) 3

- 37. Liu, Z., Sun, Z., Zang, Y., Dong, X., Cao, Y., Duan, H., Lin, D., Wang, J.: VisualRFT: Visual Reinforcement Fine-Tuning. arXiv preprint arXiv:2503.01785 (2025) 2
- 38. Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.W., Galley, M., Gao, J.: MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts. arXiv preprint arXiv:2310.02255 (2023) 9
- 39. Lyu, Z., Zhang, D., Ye, W., Li, F., Jiang, Z., Yang, Y.: Jigsaw-puzzles: From seeing to understanding to reasoning in vision-language models. arXiv preprint arXiv:2505.20728 (2025) 13
- 40. Misra, I., Maaten, L.v.d.: Self-Supervised Learning of Pretext-Invariant Representations. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6707–6717 (2020) 2, 6
- 41. Noroozi, M., Favaro, P.: Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles. In: European Conference on Computer Vision. pp. 69–84. Springer (2016) 2, 6
- 42. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C.L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., Lowe, R.: Training Language Models to Follow Instructions with Human Feedback. arXiv preprint arXiv:2203.02155 (2022) 3
- 43. Prabhudesai, M., Chen, L., Ippoliti, A., Fragkiadaki, K., Liu, H., Pathak, D.: Maximizing Confidence Alone Improves Reasoning. arXiv preprint arXiv:2505.22660

(2025) 4

- 44. Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C.D., Finn, C.: Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arXiv preprint arXiv:2305.18290 (2024) 3
- 45. Sarch, G., Saha, S., Khandelwal, N., Jain, A., Tarr, M.J., Kumar, A., Fragkiadaki, K.: Grounded reinforcement learning for visual reasoning. arXiv preprint arXiv:2505.23678 (2025) 9
- 46. Shao, R., Li, S.S., Xin, R., Geng, S., Wang, Y., Oh, S., Du, S.S., Lambert, N., Min, S., Krishna, R., Tsvetkov, Y., Hajishirzi, H., Koh, P.W., Zettlemoyer, L.: Spurious Rewards: Rethinking Training Signals in RLVR. arXiv preprint arXiv:2506.10947

(2025) 4

- 47. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y.K., Wu, Y., Guo, D.: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300 (2024) 2, 3
- 48. Shen, H., Liu, P., Li, J., Fang, C., Ma, Y., Liao, J., Shen, Q., Zhang, Z., Zhao, K., Zhang, Q., Xu, R., Zhao, T.: VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model. arXiv preprint arXiv:2504.07615 (2025) 2, 4, 9
- 49. Shen, S., Shen, P., Zhao, W., Zhu, D.: Mitigating Think-Answer Mismatch in LLM Reasoning Through Noise-Aware Advantage Reweighting. arXiv preprint arXiv:2508.05928 (2025) 4
- 50. Shojaee, P., Mirzadeh, I., Alizadeh, K., Horton, M., Bengio, S., Farajtabar, M.: The illusion of thinking: Understanding the strengths and limitations of reasoning models via the lens of problem complexity. arXiv preprint arXiv:2506.06941 (2025) 13
- 51. Tong, J., Tang, J., Li, H., Mou, Y., Zhang, M., Zhao, J., Wen, Y., Song, F., Zhan, J., Lu, Y., et al.: Game-rl: Synthesizing multimodal verifiable game data to boost vlms’ general reasoning. arXiv preprint arXiv:2505.13886 (2025) 9
- 52. Tong, S., Brown, E., Wu, P., Woo, S., Middepogu, M., Akula, S.C., Yang, J., Yang, S., Iyer, A., Pan, X., Wang, A., Fergus, R., LeCun, Y., Xie, S.: Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs. In: Proceedings of the 38th International Conference on Neural Information Processing Systems. NIPS ’24, Curran Associates Inc., Red Hook, NY, USA (2024) 9
- 53. Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., Xie, S.: Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs (2024) 9
- 54. Wang, H., Qu, C., Huang, Z., Chu, W., Lin, F., Chen, W.: VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning. arXiv preprint arXiv:2504.08837 (2025) 2, 4, 9
- 55. Wang, K., Pan, J., Shi, W., Lu, Z., Ren, H., Zhou, A., Zhan, M., Li, H.: Measuring Multimodal Mathematical Reasoning with MATH-Vision Dataset. Advances in Neural Information Processing Systems 37, 95095–95169 (2024) 9
- 56. Wang, Q., Ke, J., Ye, H., Lin, Y., Fu, Y., Zhang, J., Keutzer, K., Xu, C., Chen, Y.: Angles Don’t Lie: Unlocking Training-Efficient RL Through the Model’s Own Signals. arXiv preprint arXiv:2506.02281 (2025) 4
- 57. Wang, Q., Liu, B., Zhou, T., Shi, J., Lin, Y., Chen, Y., Li, H.H., Wan, K., Zhao, W.: Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play. arXiv preprint arXiv:2509.25541 (2025) 4, 9
- 58. Wang, X., Li, C., Yang, J., Zhang, K., Liu, B., Xiong, T., Huang, F.: LLaVACritic-R1: Your Critic Model is Secretly a Strong Policy Model. arXiv preprint arXiv:2509.00676 (2025) 4

- 59. Wang, X., Yang, Z., Feng, C., Liang, Y., Zhou, Y., Liu, X., Zang, Z., Li, M., Lin, C.C., Lin, K., Li, L., Huang, F., Wang, L.: ViCrit: A Verifiable Reinforcement Learning Proxy Task for Visual Perception in VLMs. arXiv preprint arXiv:2506.10128 (2025) 4, 9
- 60. Wang, Z., Cui, G., Li, Y.J., Wan, K., Zhao, W.: DUMP: Automated DistributionLevel Curriculum Learning for RL-based LLM Post-training. arXiv preprint arXiv:2504.09710 (2025) 4
- 61. Wang, Z., Zhu, J., Tang, B., Li, Z., Xiong, F., Yu, J., Blaschko, M.B.: Jigsaw-R1: A Study of Rule-based Visual Reinforcement Learning with Jigsaw Puzzles. arXiv preprint arXiv:2505.23590 (2025) 2, 4, 6, 9
- 62. Wu, P., Zhang, Y., Diao, H., Li, B., Lu, L., Liu, Z.: Visual Jigsaw Post-Training Improves MLLMs. arXiv preprint arXiv:2509.25190 (2025) 2, 4, 6, 9
- 63. Xia, J., Zang, Y., Gao, P., Li, S., Zhou, K.: Visionary-R1: Mitigating Shortcuts in Visual Reasoning with Reinforcement Learning. arXiv preprint arXiv:2505.14677

(2025) 4

- 64. Yao, Z., Liu, Y., Chen, Y., Chen, J., Fang, J., Hou, L., Li, J., Chua, T.S.: Are Reasoning Models More Prone to Hallucination? arXiv preprint arXiv:2505.23646

(2025) 2, 4

- 65. Ying, K., Meng, F., Wang, J., Li, Z., Lin, H., Yang, Y., Zhang, H., Zhang, W., Lin, Y., Liu, S., Lei, J., Lu, Q., Chen, R., Xu, P., Zhang, R., Zhang, H., Gao, P., Wang, Y., Qiao, Y., Luo, P., Zhang, K., Shao, W.: MMT-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask AGI. In: Salakhutdinov, R., Kolter, Z., Heller, K., Weller, A., Oliver, N., Scarlett, J., Berkenkamp, F. (eds.) Proceedings of the 41st International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 235, pp. 57116–57198. PMLR (21–27 Jul 2024) 9
- 66. Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al.: DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv preprint arXiv:2503.14476 (2025) 3
- 67. Zeng, Y., Huang, W., Huang, S., Bao, X., Qi, Y., Zhao, Y., Wang, Q., Chen, L., Chen, Z., Chen, H., et al.: Agentic jigsaw interaction learning for enhancing visual perception and reasoning in vision-language models. arXiv preprint arXiv:2510.01304 (2025) 4
- 68. Zhang, J., Huang, J., Yao, H., Liu, S., Zhang, X., Lu, S., Tao, D.: R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization. arXiv preprint arXiv:2503.12937 (2025) 2
- 69. Zhang, J., Zuo, C.: GRPO-LEAD: A Difficulty-Aware Reinforcement Learning Approach for Concise Mathematical Reasoning in Language Models. arXiv preprint arXiv:2504.09696 (2025) 2
- 70. Zhang, R., Jiang, D., Zhang, Y., Lin, H., Guo, Z., Qiu, P., Zhou, A., Lu, P., Chang, K.W., Qiao, Y., et al.: MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems? In: European Conference on Computer Vision. pp. 169–186. Springer (2024) 9
- 71. Zhang, Y., Chew, Y., Dong, Y., Leo, A., Hu, B., Liu, Z.: Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 20626–20636 (2025) 9
- 72. Zhang, Z., Jia, Z., Wu, H., Li, C., Chen, Z., Zhou, Y., Sun, W., Liu, X., Min, X., Lin, W., et al.: Q-Bench-Video: Benchmark the Video Quality Understanding of LMMs. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 3229–3239 (2025) 9

- 73. Zheng, C., Liu, S., Li, M., Chen, X.H., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., Zhou, J., Lin, J.: Group Sequence Policy Optimization. arXiv preprint arXiv:2507.18071 (2025) 3, 4
- 74. Zhou, J., Ma, L., Liang, H., Shen, C., Cui, B., Zhang, W.: DARO: Difficulty-Aware Reweighting Policy Optimization. arXiv preprint arXiv:2510.09001 (2025) 2
- 75. Zuo, Y., Zhang, K., Sheng, L., Qu, S., Cui, G., Zhu, X., Li, H., Zhang, Y., Long, X., Hua, E., Qi, B., Sun, Y., Ma, Z., Yuan, L., Ding, N., Zhou, B.: TTRL: Test-Time Reinforcement Learning. arXiv preprint arXiv:2504.16084 (2025) 4

[Figure 12]

- Fig. S1: Example Jigsaw puzzle used in PuzzleCraft training. The image taken from the Microsoft COCO dataset by Lin et al. is licensed under CC BY 4.0. Source: https: //cocodataset.org/.

### A Additional Discussion and Details on our Puzzles

Measuring the intrinsic complexity of a visual puzzle is nontrivial. In practice, we expose a single difficulty knob per puzzle type. For Rotation, the knob is the cardinality of the angle set; in our experiments we fix it to {0◦,90◦,180◦,270◦} and consider standard variations such as clockwise vs. counterclockwise phrasing. For PatchFit, the knob is distractor hardness: given a ground-truth patch, we sample D ∈ {3,5,7} decoys drawn from mirror/rotation/color perturbations of the true patch or visually similar patches from other regions. For Jigsaw, difficulty is controlled by grid size: given an M×N grid, we allow any integer pair with 2 ≤ MN ≤ 9 (i.e., up to 3×3). Sampling is uniform over the chosen configurations.

Under random guessing, the success rates are as follows. Rotation: 1/4 = 25%. PatchFit: averaging over D ∈ {3,5,7} decoys yields an expected success of

- 1 4, 16, 18 respectively, i.e., ≈ 18% on average. Jigsaw: with graded reward defined as the fraction of tiles placed correctly, the expected score under a random permutation depends on MN; in our sampling it averages to ≈ 26% (gridsize dependent). Exact puzzle-generation details will be provided in the released code. Figure S1 shows a Jigsaw training sample, Figure S2 a Rotation sample, and Figure S3 a PatchFit sample.

[Figure 13]

- Fig. S2: Example Rotation puzzle used in PuzzleCraft training. The image taken from the Microsoft COCO dataset by Lin et al. is licensed under CC BY 4.0. Source: https: //cocodataset.org/.

### B Additional Ablations

- B.1 Difficulty-Only Curricula

We compare PuzzleCraft’s exploration-aware curriculum against the difficultyonly curriculum of Observe-R1 [21] and VCRL [28]. Results are shown in Table S1. Our curriculum consistently performs better, improving Avg. by more than 2.5 points over both methods. This suggests that reward statistics alone are not sufficient for puzzle-based RLVR: accounting for rollout exploration and collapse leads to stronger learning signal and better downstream performance.

- B.2 Frozen Vision Encoder

We also study a setting where the vision encoder is frozen during post-training, with results reported in Table S1. Even in this constrained setup, PuzzleCraft remains strong and clearly outperforms the base model, indicating that a substantial portion of the gains comes from improving the reasoning policy in the language model. At the same time, the full model still performs better than the frozen-ViT variant, suggesting that allowing the visual encoder to adapt provides additional benefits and that part of the improvement also comes from refining the perception module.

### C RAC Measurement

To measure RAC, we sample rollouts uniformly across the post-training trajectory and, at regular intervals, query a fixed open-source judge (Qwen2.5-VL-72B

[Figure 14]

- Fig. S3: Example PatchFit puzzle used in PuzzleCraft training. The image taken from the Microsoft COCO dataset by Lin et al. is licensed under CC BY 4.0. Source: https: //cocodataset.org/.

Table S1: Additional ablations of PuzzleCraft on image benchmarks. We compare our exploration-aware curriculum against difficulty-only curricula (Observe-R1 and VCRL), and further study a frozen-ViT setting to separate gains from visual adaptation versus improvements in the reasoning policy. Avg. is the mean across all 9 benchmarks, with MME normalized as (MME/2800) × 100 before averaging.

|Design choice|MathVista MathVision MathVerse MME MMStar POPE MMT CV-Bench MMVP<br><br>|Avg.|
|---|---|---|
|Qwen2.5-VL-Instruct Jigsaw w/ CL + CARE (Main)<br><br>|66.30 23.68 56.49 2243 64.67 81.77 59.64 73.62 77.67 68.20 30.92 56.70 2366 64.60 86.52 62.26 77.63 78.67<br><br>|64.88 67.78|

Curricula

Main w/ Observe-R1 67.40 23.02 54.49 2263 64.86 83.59 62.04 75.54 74.33 65.12 Main w/ VCRL 64.00 26.31 55.99 2296 62.20 83.58 60.12 74.49 78.33 65.22 Frozen ViT

Main w/ Frozen ViT 68.00 25.32 54.00 2340 65.40 85.64 61.04 76.69 78.00 66.41

in inference mode) with each rollout’s rationale and final <answer>. The judge determines whether the rationale explicitly supports the answer, and each sample is assigned a binary score in [0,1]. The prompt used for this evaluation is shown in Figure S5. We also provide representative examples from Jigsaw posttraining, together with their corresponding RAC scores, to illustrate the kinds of reasoning–answer patterns captured by this metric (Figure S4).

Observed pattern. Figure 2b summarizes RAC dynamics for vanilla GRPO, GRPO+curriculum, GRPO+CARE, and GRPO+curriculum+CARE on Jigsaw. Consistent with observations in LLM post-training [9], we observe an early increase in faithfulness followed by a later decline for vanilla GRPO. In our setting, the curriculum mitigates this decline, and CARE further improves RAC

- Fig. S4: Examples of RAC evaluation during Jigsaw post-training. Each sample shows a model-generated reasoning trace and final answer, together with the RAC score assigned by the judge. These examples illustrate both faithful and inconsistent cases captured by the metric.

##### throughout training. We use RAC as a diagnostic signal for monitoring consistency dynamics rather than as a strict model-selection criterion.

###### Fig. S5: Prompt used to measure Reasoning–Answer Consistency (RAC).

