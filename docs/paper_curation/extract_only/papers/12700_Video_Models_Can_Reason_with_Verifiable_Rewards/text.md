# arXiv:2605.15458v1[cs.CV]14May2026

## Video Models Can Reason with Verifiable Rewards

Tinghui Zhu♠ Sheng Zhang♣ James Y. Huang♡ Selena Song♢ Xiaofei Wen♠ Yuankai Li♠ Hoifung Poon♣ Muhao Chen♠ ♠University of California, Davis ♣Microsoft Research ♡University of Southern California ♢University of California, Santa Cruz {thuzhu, muhchen}@ucdavis.edu shezhan@microsoft.com

Project Page: https://darthzhu.github.io/VideoRLVR-page/

### Abstract

Video diffusion models have made rapid progress in perceptual realism and temporal coherence, but they remain primarily optimized for plausible generation rather than verifiable reasoning. This limitation is especially pronounced in tasks where generated videos must satisfy explicit spatial, temporal, or logical constraints. Inspired by the role of reinforcement learning with verifiable rewards (RLVR) in reasoning-oriented language models, we introduce VideoRLVR, a practical recipe for optimizing video diffusion models with rule-based feedback. VideoRLVR formulates video reasoning as the generation of verifiable visual trajectories and consists of an SDE-GRPO optimization backbone, dense decomposed rewards, and an Early-Step Focus strategy for efficient training. The Early-Step Focus strategy restricts policy optimization to the early denoising phase, reducing training latency by about 40% while preserving performance. We evaluate VideoRLVR on Maze, FlowFree, and Sokoban, three procedurally generated domains with objective success criteria. Across these tasks, VideoRLVR consistently improves over supervised fine-tuning baselines, with dense decomposed rewards proving especially important in low-success-rate settings. Our RL-optimized model also outperforms the evaluated proprietary and open-source video generation models on these verifiable reasoning benchmarks and out-of-domain benchmarks. These results suggest that verifiable RL can move video models beyond perceptual imitation toward more reliable rule-consistent visual reasoning.

### 1 Introduction

Recent progress in large language models (LLMs) has reshaped the role of generative models from content producers into increasingly capable reasoning systems [11, 33, 6]. A key intuition behind this shift is that the model can externalize the problem-solving process by generating intermediate states rather than only a final answer. This raises a natural question for video generation: if language models can reason through sequences of tokens, can video models reason through sequences of frames? Videos provide an appealing foundation for this idea, where each frame can represent an intermediate visual state in a goal-directed process. In domains such as navigation [7], puzzle solving [14], and embodied planning [28], a generated video can therefore be viewed not merely as motion synthesis, but as a temporally ordered chain of visual states [40] that encodes a visual reasoning trajectory.

Despite this potential, current video diffusion models are still primarily optimized for perceptual quality, temporal coherence, and plausible motion [13, 46, 37]. While large-scale video models have begun to show signs of visual reasoning [40, 12, 38], these abilities remain difficult to elicit reliably and verify under standard training objectives. The core challenge is the mismatch between perceptual plausibility and objective correctness. Supervised fine-tuning (SFT) on ground-truth solution videos can teach the model the visual form of valid trajectories, yet it does not directly optimize the correctness of sampled outputs. As a result, models may imitate solution-like patterns while failing to satisfy the underlying rules that make those solutions valid [9, 29]. This suggests an

Preprint.

#### An RL Recipe for Video Reasoning

##### Conventional Video Generation Video Reasoning Tasks

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Goal-Directed Planning

[Figure 8]

Visual Realism

[Figure 9]

Maze

[Figure 10]

[Figure 11]

[Figure 12]

Long-Horizon Reasoning

[Figure 13]

Temporal Coherence

FlowFree

[Figure 14]

[Figure 15]

[Figure 16]

Veriﬁable Success

[Figure 17]

Plausible Motion

Sokoban

[Figure 18]

Training Paradigm

[Figure 19]

Language Reasoning Models Pre-training

RLVR

Supervised Fine-tuning

[Figure 20]

[Figure 21]

[Figure 22]

Missing Piece

Reinforcement Learning with Veriﬁable Rewards

[Figure 23]

Video Reasoning Models Pre-training

RLVR

Supervised Fine-tuning

[Figure 24]

[Figure 25]

[Figure 26]

Reinforcement Learning with Veriﬁable Rewards

Figure 1: Evolution towards verifiable video reasoning. Top: Comparison between perceptionfocused generation and reasoning-intensive tasks. Bottom: We introduce VideoRLVR, the missing puzzle in the training paradigm for video reasoning models.

analogy to reasoning-oriented LLMs where pre-training provides broad generative competence, SFT teaches the format of reasoning traces, Reinforcement Learning with Verifiable Rewards (RLVR) is the essential third stage required to optimize objective correctness, as illustrated in Figure 1.

In this work, we introduce VideoRLVR, a systematic recipe for applying reinforcement learning with verifiable rewards to video models. Our framework has three main components. First, we adopt an SDE-GRPO optimization backbone [24] for optimizing flow-matching video models. Second, we propose an Early-Step Focus strategy for efficient video RL. Instead of applying stochastic exploration and backpropagation across the entire denoising trajectory, this strategy concentrates optimization on the early denoising phase, where coarse structure and long-range planning are largely determined [39]. Finally, we design dense decomposed rewards that break sparse task success into verifiable structural components, providing informative feedback even when full success is rare. To acquire dense reward signals, we construct verifiable video reasoning data by generating solution trajectories with rule-based planners and aligning each logical transition with the video frame sequence.

We evaluate our RLVR recipe on a multi-task suite designed for rule-based verification, including Maze, FlowFree, and Sokoban. Our experiments show that VideoRLVR improves video reasoning beyond supervised imitation. Across all three domains, the RL-optimized model consistently achieves higher success rates than the SFT checkpoint used to initialize training, with gains of 6.1%, 5.5%, and 3.2% on Maze, FlowFree, and Sokoban, respectively. Compared with continued supervised training, VideoRLVR yields larger gains on harder tasks, suggesting that verifiable rewards provide an optimization signal beyond what can be captured by imitation alone. We further evaluate VideoRLVR on the out-of-domain split of VBVR [38], where VideoRLVR shows improved transfer beyond the training domains. Our ablations further show that dense decomposed rewards are crucial in low-success-rate domains, and that Early-Step Focus reduces training time by about 40% while maintaining nearly the same performance. Finally, VideoRLVR outperforms several proprietary and open-source video generation models on our verifiable reasoning benchmarks, indicating that targeted verifiable RL can substantially improve the logical correctness of generated visual trajectories.

In summary, our contributions are as follows:

- 1. We introduce VideoRLVR, a reinforcement learning framework that optimizes video diffusion models with verifiable rewards, including dense decomposed reward functions to provide informative feedback for rule-verifiable visual trajectories.
- 2. We introduce a scalable training pipeline that combines rule-based trajectory generation, SDEGRPO optimization, and an Early-Step Focus strategy that reduces training time by about 40% while preserving the performance.
- 3. We show that VideoRLVR improves over supervised fine-tuning and competitive proprietary and open-source video generation models on Maze, FlowFree, and Sokoban, while also demonstrating improved out-of-domain transfer on VBVR.

### 2 Related Work

Reinforcement learning for diffusion and flow-matching models. Reinforcement learning has increasingly been used to align diffusion and flow-based generative models with human preferences, perceptual objectives, and task-specific rewards [43]. Prior work formulates denoising as a sequential decision process and applies policy-gradient or preference-optimization methods to improve textto-image and video generation [2, 8, 36]. For flow-matching models, recent methods address the deterministic nature of ODE sampling by introducing stochastic transitions or alternative preference objectives, enabling likelihood-ratio or GRPO-style optimization [24, 44, 5, 27]. Other extensions apply these ideas to video or embodied objectives [1, 25]. However, existing work optimizes perceptual or preference-based criteria such as aesthetics, text rendering, image fidelity, geometric consistency, or motion quality [21, 22]. In contrast, our work studies reinforcement learning for verifiable video reasoning, where rewards are computed from objective task rules and success depends on the logical correctness of the generated visual trajectory.

Reasoning in video generation models. Recent work has begun to investigate whether video generation models can serve as reasoning systems rather than only visual synthesizers. Large-scale video models have shown emerging abilities on visual puzzles and sequential prediction tasks, motivating the view that video generation can be interpreted as a chain of visual states or “chain of frames” [40, 12, 18]. Benchmark efforts [38, 4, 45, 35] further evaluate video models on reasoningoriented tasks that require temporal consistency, spatial planning, or rule satisfaction. Other studies analyze video models as world simulators or physical reasoners, highlighting both their potential and their limitations in capturing causal and physical structure [3, 19, 28, 29, 48, 34]. These works suggest that video models may contain useful visual reasoning priors, but also show that standard generation objectives do not reliably produce rule-correct trajectories [12, 26]. Our work addresses this gap by directly optimizing video models with verifiable rewards, using rule-based success criteria rather than relying solely on supervised imitation or zero-shot generation.

Verifiable reinforcement learning and reasoning models. Reinforcement learning with verifiable rewards has played an important role in recent progress on reasoning-oriented language models [11, 33, 6]. In these settings, the model is rewarded according to objective correctness signals, such as mathematical equivalence, executable code tests, or rule-based verification, instead of only human preference judgments [23, 47, 16, 17]. This paradigm is attractive because it provides scalable supervision when outcomes can be automatically checked, which facilitates the development of emerging behaviors like searching and backtracking [49, 42]. Our work extends this training from language outputs to video trajectories. Whereas text reasoning is often verified by final-answer correctness, video reasoning requires trajectory-level verification over visual, temporal, and process constraints. We study how verifiable RL can optimize video diffusion models under these criteria.

### 3 Problem Formulation

RLVR for Video Reasoning. Following [40], we formulate video reasoning as a conditional generation task where a model generates a temporal sequence of visual states whose transitions and terminal state can be checked against task-specific rules. Given an initial image I0 and a textual instruction T, let c = (I0,T) denote the conditioning input. The model generates a video V = {I0,I1,...,IF−1}, where F is the number of frames. Unlike standard video synthesis, which primarily evaluates perceptual quality and temporal coherence, video reasoning requires the generated sequence to satisfy task-specific correctness criteria. This formulation allows us to treat video generation as a search for a valid visual trajectory conditioned on the initial state and instruction.

Video Generation as a Markov Decision Process. To apply reinforcement learning to flow-matching video generation, we formulate the reverse denoising process as a Markov Decision Process (MDP) over latent variables. This MDP is defined over denoising steps rather than reasoning steps, where the reward is computed after the final video is decoded. At denoising step k ∈ {1,...,K}, the state is the noisy video latent xt

,tk,c), which determines the mean update of the next latent. Under the Ordinary Differential Equation (ODE) solver, the transition is given by xt

at noise level tk, and the action is the model velocity prediction vˆθ(xt

k

k

+ (tk+1 − tk)ak. After the final denoising step, the decoded video V receives a verifier-derived reward R(V,c). A fundamental challenge in this formulation is that standard flow matching employs a deterministic ODE solver, making it a

= xt

k+1

k

deterministic function of the initial noise x1. Under this deterministic solver, the next latent is a deterministic function of (xt

,c), yielding no tractable stochastic transition density πθ(xt

k+1 | xt

,c) for likelihood-ratio policy gradients. In Section 4, we address this by adopting an SDE-based formulation that introduces stochastic transitions compatible with flow-matching generation.

k

k

Tasks. To evaluate VideoRLVR across different reasoning domains, we instantiate our framework on three rule-verifiable visual reasoning domains: Maze, FlowFree, and Sokoban. We choose these tasks because they satisfy three properties: 1) solution correctness can be checked by rule-based verifiers, 2) large-scale training and test instances can be generated, and 3) the tasks span different levels of reasoning complexity. Maze primarily tests spatial connectivity under explicit obstacle constraints, FlowFree requires globally consistent non-overlapping path connectivity and implicit constraints, and Sokoban introduces object interaction, irreversible transitions, and longer-horizon reasoning.

### 4 RLVR Recipe for Video Reasoning Models

We present VideoRLVR, a systematic recipe for optimizing video models with verifiable rewards. The recipe consists of three components: 1) an SDE-GRPO optimization backbone, 2) an Early-Step Focus optimization strategy, and 3) dense decomposed rewards design and acquisition.

###### 4.1 SDE-GRPO for Video Reasoning

GRPO [32] estimates relative advantages from groups of sampled outputs without training a separate critic, making it well suited for verifiable reward settings. However, standard flow-matching models generate samples with a deterministic ODE sampler, which does not provide a tractable stochastic transition density over denoising steps. Following Flow-GRPO [24], we convert the deterministic denoising dynamics into stochastic transitions with Gaussian log-probabilities.

Stochastic denoising transitions. For a discretized denoising schedule {tk}Kk=1, the SDE formulation defines a Gaussian transition:

,tk,c),σk2I , (1) where µθ(xt

k+1 | xt

,c) = N xt

πθ(xt

;µθ(xt

k

k+1

k

,tk,c) is the mean update induced by the model and σk2 is the SDE transition variance. This stochastic transition enables closed-form log-probabilities and likelihood-ratio policy gradients. GRPO objective. Given a group of G sampled videos for each condition, we compute verifierderived rewards and normalize them within the group to obtain advantages Ai. For each sample i and denoising step k, we compute the dimension-normalized log-ratio:

k

πθ(x(ti)

| x(ti)

D

,ci) πold(x(ti)

2 d

2 d

- 1

- 2σk2 ·

1 D

x(ti)

− µ(θi,k)

− x(ti)

− µ(oldi,k)

k+1

k

= −

log ρi,k = log

,

| x(ti)

k+1

k+1

,ci)

d=1

k+1

k

(2) where µ(θi,k) = µθ(x(ti)

,tk,ci), µ(oldi,k) = µold(x(ti)

,tk,ci), and D is the number of latent elements. The policy loss uses PPO-style clipping:

k

k

Lpolicy = −Ei,k [min(ρi,kAi, clip(ρi,k,1 − ε,1 + ε)Ai)]. (3) We additionally regularize the policy against the reference model with a closed-form KL penalty:

∥µθ − µref∥22 2σk2

1 D

. (4)

LKL = Ek

The final objective is LVideoRLVR = Lpolicy + βLKL,where β controls the strength of regularization.

###### 4.2 Early-Step Focus for Efficient Video RL

Video RL is substantially more expensive than text RL because each rollout requires generating and backpropagating through high-dimensional spatio-temporal latents. A full SDE-GRPO update over all K denoising steps therefore incurs large memory and time costs. However, not all denoising steps contribute equally to the reasoning objective. Early high-noise steps are primarily responsible for coarse layout, object placement, and long-range structure, whereas later low-noise steps mainly refine local appearance and consolidate the generation into a specific visual trajectory [39].

Motivated by this observation, we introduce Early-Step Focus. During RL optimization, we sample the full denoising trajectory for generation and reward evaluation, but restrict stochastic perturbation, log-probability computation, and gradient backpropagation to the first L < K denoising steps. This creates an efficient exploration-exploitation trade-off: early denoising steps receive stochastic perturbations and policy-gradient updates for high-level reasoning, while later steps preserve the generative prior and refine visual details. The policy loss becomes:

LESF = −Ei,k≤L [min(ρi,kAi, clip(ρi,k,1 − ε,1 + ε)Ai)] + βLkKL≤L. (5) In our experiments, we use K = 20 denoising steps and L = 10 early steps. This reduces training latency by about 40% while preserving reasoning performance, suggesting that the early denoising phase carries most of the reward-relevant structural signal.

###### 4.3 Verifiable Reward Design and Acquisition

A key requirement for VideoRLVR is that generated videos can be automatically parsed and evaluated. Existing video reasoning datasets [45, 38] often lack the scale, task diversity, or fine-grained difficulty variation required to study RLVR for video reasoning. We synthesize task instances with rule-based planners that sample an initial configuration, solve it with a valid action sequence, and render the resulting state trajectory into a video. Alongside each trajectory, we retain environment metadata, such as grid layouts, endpoint locations, object states, and goal conditions, which is used for automatic verification and reward computation. Each discrete environment action is mapped to a unique frame transition If → If+1, making the generated video directly interpretable as a reasoning trajectory. Task-specific generation details are provided in Section A.

Given the metadata from the data curation process, we now can convert task rules into dense reward signals. Instead of using only a binary success reward signal, we decompose each task into structural components that measure partial progress toward a valid solution. This is especially important in low-success-rate domains, where most sampled videos receive zero reward and therefore provide little variation within a GRPO group.

Task-aware Reward Function. We use a task-aware reward function for joint training across heterogeneous domains. For each conditioning input c, the dispatcher identifies the task T (c) ∈ {Maze,FlowFree,Sokoban} and evaluates the generated video with the corresponding reward:

R(V,c) = RT (c)(V,c). (6) This allows mixed-task RL batches while preserving task-specific verification criteria. Dense Reward Formulations. For each task, we decompose the global objective into measurable rule-based components:

- • Maze. We define the reward as: Rmaze = Rconn · Rwall, where Rconn measures start-to-goal path connectivity and Rwall penalizes wall violations. Compared with an additive formulation, the multiplicative form produces sharper reward separation within a GRPO group by assigning high scores only to trajectories that satisfy both connectivity and wall consistency, yielding more informative relative advantages.
- • FlowFree. We combine four structural metrics: Rff = λvalidRvalid + λpresRpres + λconnRconn + λfillRfill, where Rvalid measures endpoint-to-endpoint path validity, Rpres measures preservation of the given endpoints, Rconn measures 4-connected color regions, and Rfill measures grid coverage by valid path colors. The weights λvalid,λpres,λconn,λfill balance the relative importance of these components. In our experiments, we set them to be 0.15, 0.35, 0.30, and 0.20, respectively.
- • Sokoban. We use a combination of final-state and process-validity rewards: Rsok = λstateRstate + λprocRproc, where Rstate measures box placement on target cells and Rproc measures the fraction of valid transitions under Sokoban movement rules. The weights λstate and λproc balance final-state correctness and process validity. We use λstate = λproc = 0.5 in all experiments.

### 5 Experiments

In this section, we evaluate VideoRLVR from two perspectives. First, we compare against supervised fine-tuning and competitive video generation baselines on three rule-verifiable reasoning domains: Maze, FlowFree, and Sokoban. Then, we test transfer beyond the training domains using the out-ofdomain split of VBVR [38]. Together, these experiments assess whether verifiable RL improves both in-domain rule-based correctness and out-of-domain visual reasoning behavior.

Table 1: Comparison of our method with other baselines. We report Precision (Prec), Recall (Rec), F1, and Success Rate (SR). Bold indicates the best and underlined indicates second best.

Maze FlowFree Sokoban Prec Rec F1 SR Prec Rec F1 SR Prec Rec F1 SR

Model

Proprietary Models Sora 2 15.8 17.2 16.5 3.1 10.8 5.1 5.8 0.0 8.5 4.8 5.4 0.0 Kling V3 24.8 15.7 19.2 23.5 18.8 2.7 4.7 0.0 5.7 2.7 3.7 0.0 Veo 3.1 22.8 18.1 20.2 26.0 23.9 4.7 7.5 4.0 22.2 6.0 9.4 0.0 Open-Source Models CogVideoX1.5 13.3 10.8 11.9 0.0 18.7 2.2 3.9 0.0 3.2 0.3 0.5 0.0 HunyuanVideo 17.3 11.4 13.8 2.2 12.5 2.9 4.8 0.0 8.2 2.7 3.2 0.0 Wan2.2-TI2V-5B 18.3 12.2 14.6 0.0 17.4 2.0 3.4 0.0 4.1 0.7 1.0 0.0 SFT Models Wan-R1 20.9 65.6 31.7 31.9 20.9 3.6 6.1 0.0 7.7 2.1 3.3 0.0 VBVR-Wan2.2 62.7 77.8 69.4 60.8 17.9 5.6 8.5 1.7 16.2 1.7 3.1 0.0 SFT Epoch 5 80.2 83.0 81.6 66.1 42.8 42.2 42.4 2.4 33.6 11.9 17.6 2.9 SFT Epoch 10 80.4 85.1 82.7 69.0 43.1 42.5 42.8 2.5 32.8 11.6 17.1 2.7 RL Model VideoRLVR 82.1 86.9 84.4 72.2 44.3 43.8 44.0 7.9 34.0 12.5 29.4 6.1

###### 5.1 Experimental Setup

Dataset. We train and evaluate on a multi-task suite of three procedurally generated reasoning domains: Maze, FlowFree, and Sokoban. To prevent the model from overfitting to specific visual features, we apply varied color themes across the dataset, encouraging the model to rely on structural invariants. Each sample consists of an input image, a task instruction, and an 81-frame ground-truth video at 480×832 resolution. The total training dataset consists of 30,000 samples (10,000 per task). For the test set, we maintain a held-out set of 3,000 samples (1,000 per task) generated with disjoint random seeds. Dataset construction details are provided in Section B.1.

Base Model and SFT Baseline. We use Wan2.2-TI2V-5B [37], a state-of-the-art video generation model, as our base model. It generates F = 81 frames at 480 × 832 resolution. We first establish an SFT baseline by training the model on ground-truth solution videos using the standard flow matching objective. This SFT checkpoint provides the necessary perceptual and structural prior for the model, serving as both the initial policy and the reference policy πref for RL optimization.

Baselines. To evaluate the effectiveness of our method, we compare our model with competitive proprietary and open-source video generation baselines. For proprietary models, we use Sora 2 [30], Kling V3 [20], and Veo 3.1 [40]. For open-source models, we compare with Wan2.2-TI2V-5B [37], CogVideoX1.5-5B-I2V [13], and HunyuanVideo-I2V [41]. We also compare with specialized SFTbased video reasoning models, including Wan-R1 [45] and VBVR-Wan2.2 [38]. Wan-R1 adopts the same base model as ours and is trained on the Maze and Sokoban domains with LoRA [15]. VBVR-Wan2.2 utilizes Wan2.2-I2V-A14B [37] and is trained on the VBVR dataset with LoRA.

Training Configuration. We train the SFT baseline for 5 epochs with a learning rate of 1×10−5. For VideoRLVR, we initialize from the SFT checkpoint and train on the same training set for 1 epoch using SDE-GRPO as the optimization backbone. We use group size G = 16, T = 20 denoising steps, learning rate 5×10−6, and KL coefficient β = 0.04. Following the Early-Step Focus strategy in Section 4.2, backpropagation and SDE injection are restricted to the first L = 10 steps of the denoising trajectory. All training experiments are conducted on 8 NVIDIA B200 GPUs.

Evaluation. We evaluate the results using two complementary metric families: 1) trajectory alignment metrics, including Precision (Prec), Recall (Rec), and F1, which measure pixel-, cell-, or action-level alignment with the reference solution, and 2) symbolic success rate, which verifies whether the video satisfies the underlying task rules. Evaluation details are listed in Section B.2

###### 5.2 Main Results

Table 1 compares VideoRLVR with supervised baselines and competitive video generation models on our verifiable reasoning benchmarks.

RLVR consistently outperforms supervised baselines. VideoRLVR yields consistent improvements across all three reasoning domains. Compared with the SFT Epoch 5 checkpoint used to initialize

RL training, VideoRLVR improves success rate by 6.1% on Maze, 5.5% on FlowFree, and 3.2% on Sokoban. Notably, VideoRLVR also significantly surpasses the performance of recent state-of-the-art closed-source models on visual reasoning tasks, validating the efficacy of verifiable reinforcement learning in domains where generic video pre-training remains insufficient for complex logical tasks.

Superior scaling on high-complexity tasks. To isolate the specific advantages of RLVR over extended supervised learning, we evaluate a stronger SFT baseline (SFT Epoch 10), representing the result of conducting further supervised training on the same Epoch 5 checkpoint.

As shown in Table 1 and Figure 2, VideoRLVR is more robust than continued SFT as task difficulty increases. Within the Maze domain, RLVR establishes a 3.2% margin over the SFT Epoch 10 checkpoint and shows less degradation when the scale of the maze increases. On FlowFree, VideoRLVR improves over SFT Epoch 10 by

0.90

SFT Epoch 5

SFT Epoch 10

0.85

SDE-GRPO

0.80

SuccessRate

0.75

0.70

- 5.4%, while continued SFT provides little improvement over the Epoch 5 checkpoint. On Sokoban, continued SFT slightly degrades performance, whereas VideoRLVR improves over SFT Epoch 10 by 3.4%. These trends suggest that verifiable rewards provide an optimization signal that is not captured by additional imitation training alone.

0.65

0.60

25 (n=11)

26-49 (n=147)

>121 (n=211) Maze grid size (H × W cells)

50-81 (n=340)

82-121 (n=291)

Figure 2: Success rate of different grid size for maze. n represents the number of samples falling into this range

- 5.3 Comparison with LLMs Table 2: Comparison with LLMs on maze tasks.

Model

Maze Prec Rec F1 SR GPT 4o 11.7 13.0 12.3 0.0 GPT 5.5 Pro 76.0 70.1 72.9 66.0

- Gemini 2.5 Flash 11.2 10.5 10.9 0.0

- Gemini 3.1 Pro 26.8 27.0 26.9 23.0 VideoRLVR 82.1 86.9 84.4 72.2

To determine if our reasoning domains can be solved by language reasoning alone, we benchmark frontier LLMs on the Maze task. Table 2 presents the results of state-of-the-art models, including GPT-5.5 Pro [31] and Gemini 3.1 Pro [10], compared against our RLVR-optimized video model. Despite their sophisticated reasoning capabilities in textual domains, LLMs exhibit a sharp performance decay in maze tasks. This divergence highlights a representation bottleneck: while LLMs must reason over a tokenized rendering of the maze, our video model operates directly on a visual latent space that inherently preserves the visual topological relationships necessary for complex visual reasoning. These results suggest that, for visual reasoning, directly generating and optimizing visual trajectories can be more effective than solving the task through language-token representations alone.

- 5.4 OOD Results

To evaluate if VideoRLVR transfers beyond the training domains, we test our model on the out-of-domain split of VBVR [38]. This benchmark covers multiple reasoning categories and therefore provides a broader test of whether VideoRLVR improves general video reasoning behavior beyond Maze, FlowFree, and Sokoban. As shown in Table 3, VideoRLVR substantially improves over the 5B baseline, increasing the average score from 26.2 to 60.2 with gains across all VBVR-OOD categories. VideoRLVR also performs competitively with the larger 14B VBVR-Wan2.2 model, achieving a similar average score despite using a smaller 5B backbone and much less training data. These results suggest that VideoRLVR learns transferable visual reasoning ability that generalizes beyond the generated training tasks.

Table 3: OOD evaluation on VBVR. We report average performance and category-wise scores on the VBVR-OOD split.

Model Avg. Abst. Know. Perc. Spat. Trans. 5B Models

CogVideoX1.5 26.2 28.1 23.5 25.0 25.4 28.2 VideoRLVR 60.2 65.5 62.0 59.7 58.8 58.2 14B Models

Wan2.2-I2V-A14B 32.9 40.5 30.8 34.3 23.6 30.7 VBVR-Wan2.2 61.0 76.8 57.2 54.7 61.8 61.5

### 6 Analysis

In this section, we analyze the main components of VideoRLVR, including GRPO group size, Early-Step Focus, KL regularization, dense reward design, and qualitative generation behavior.

###### 6.1 Ablation Study

Impact of Group Size. In GRPO, the group size G affects the stability of group-relative advantage estimation. We investigate the impact of this hyperparameter within the Maze reasoning

domain, as shown in Figure 3. Our results indicate that performance scales positively with group size, primarily due to the stabilization of the reward distribution’s statistics. While the expected sample standard deviation of rewards is a property of the policy’s diversity, a small group size (e.g., G ≤ 4) provides a noisy and often biased estimate of this value. This leads to significant fluctuations in the advantage calculation, as the group mean fails to accurately represent the current policy’s performance level. Increasing the group size to G = 16 provides a more stable comparison set for estimating relative advantages, which improves training stability in our experiments. However, we observe diminishing returns beyond this point. Furthermore, because video generation remains a significant computational bottleneck, scaling G entails a linear increase in rollout time and VRAM overhead. We therefore use G = 16 as a practical trade-off between advantage-estimation stability and computational cost.

0.75

0.70

0.65

Maze

SuccessRate

0.60

FlowFree Sokoban

0.10

SFT epoch 5

0.08

0.06

0.04

0.02

0 5 10 15 20 25 Group Size

Figure 3: Scaling results with group size.

Early-Step Focus. To validate the efficacy of our Early-Step Focus strategy, we conduct a controlled experiment within the Maze domain. We fix the total inference budget at T = 20 denoising steps

and compare the reasoning performance when the gradient and noise injection is calculated over the full trajectory (L = 20) versus the first L = 10 steps. As illustrated in Table 4, the success rates and F1 scores remain nearly unchanged, while training time is substantially reduced. This suggests that the early denoising steps carry much of the reward-relevant structural signal for visual reasoning. Because the later denoising steps primarily govern local textural refinement, they contribute less to the verifier-derived reasoning objective in this setting. Restricting backpropagation and noise injection to these early steps thus serves as a computationally efficient optimization path, significantly reducing the training time without degrading the performance.

Table 4: Comparisons of computing over full denoising steps and Early-Step Focus on Maze.

|Gradient Steps<br><br>|F1 SR Time / step|
|---|---|
|20 (Full) 10 (Early 10 steps)<br><br>|84.6 72.3 156 s 84.4 72.2 93.5 s|

KL Constraint. The KL-divergence constraint is essential for maintaining the model’s generative prior. We show a qualitative example in Section C.1. As shown in Figure 5, removing the KL penalty (β = 0) at an early stage of GRPO optimization can lead to reward-hacking behavior. Without this regularization, the model may produce visually implausible patterns that satisfy parts of the verifier while degrading generation quality. Implementing a constant penalty of β = 0.04 successfully anchors the optimization to the original quality, ensuring that improvements in logical success are achieved without sacrificing the model’s inherent visual plausibility.

###### 6.2 Reward Design

To evaluate the necessity of our dense decomposed reward design, we investigate the efficacy of a sparse reward (R ∈ {0,1}) based exclusively on success rate. This ablation aims to determine if binary feedback is sufficient across varying levels of task complexity.

Table 5: Training with a sparse binary success reward.

Steps 0 200 400 600 800 1000 Maze 66.1 67.2 68.9 70.1 71.5 72.9 FlowFree 2.4 2.3 2.5 2.4 2.6 2.5

Our results, shown in Table 5, reveal that sparse success rewards behave differently across domains. In domains like maze, where the baseline model already achieves a decent success rate, the sparse

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Initial SFT Epoch 5 VideoRLVR

[Figure 35]

Maze

[Figure 36]

[Figure 37]

FlowFree

[Figure 38]

[Figure 39]

Sokoban

[Figure 40]

Figure 4: Qualitative case study across three reasoning domains. We compare generations from the SFT baseline (Epoch 5) and the VideoRLVR model.

reward signal is sufficient to provide an informative gradient. The model is able to encounter success frequently enough during group rollouts to differentiate between advantageous and disadvantageous trajectories. Conversely, on high-complexity tasks like FlowFree and Sokoban, where the initial success rate is near-zero, the sparse reward provides little useful signal. In these environments, the model suffers from extreme gradient sparsity. Since success is rarely encountered within the group rollout G, the advantage estimates remain uninformative. This underscores a critical cold-start problem in video RL: while binary success is the ultimate goal, it is an insufficient signal for exploring high-dimensional latent space from a low-performance starting point. Dense decomposed rewards address this issue by providing partial credit for intermediate structural properties, giving the policy useful feedback before full success becomes frequent.

###### 6.3 Case Study

- Figure 4 provides qualitative examples across Maze, FlowFree, and Sokoban. The SFT baseline often captures the visual format of each domain, such as drawing paths, coloring grids, or rendering objects, but it can fail to satisfy the task rules. For example, SFT outputs may contain disconnected paths, inconsistent color connectivity, or invalid object transitions. In the Sokoban example, the SFT model produces a visually plausible but invalid shortcut rather than a valid sequence of box-pushing actions. In contrast, the VideoRLVR-optimized model more consistently satisfies the symbolic constraints checked by our verifiers. Across the shown examples, it produces connected paths, more coherent grid solutions, and more valid object transitions while preserving the overall visual structure of the task. These qualitative results support the quantitative findings: verifiable RL improves rule-based correctness beyond what is achieved by supervised imitation alone.

### 7 Conclusion

This work studies reinforcement learning with verifiable rewards for video reasoning and introduces VideoRLVR, a practical recipe for optimizing video reasoning models. By combining rule-verifiable data generation, an SDE-GRPO optimization backbone, dense decomposed rewards, and Early-Step Focus, VideoRLVR addresses the gap between perceptual video synthesis and task-level logical correctness. Our experiments show that supervised fine-tuning provides an important visual and structural prior, but can plateau or degrade on harder reasoning tasks when optimized only through imitation. In contrast, verifiable RL improves success rates across Maze, FlowFree, and Sokoban, with dense decomposed rewards proving especially useful in low-success-rate domains. We further show that Early-Step Focus reduces training time by about 40% with little observed loss in reasoning performance. Overall, our results suggest that verifiable RL can substantially improve the logical correctness of generated videos, enabling open-source video models to outperform stronger generalpurpose video generation baselines on visual reasoning benchmarks.

### References

- [1] Zhaochong An, Orest Kupyn, Théo Uscidda, Andrea Colaco, Karan Ahuja, Serge Belongie, Mar Gonzalez-Franco, and Marta Tintore Gazulla. Vggrpo: Towards world-consistent video generation with 4d latent reward. arXiv preprint arXiv:2603.26599, 2026.
- [2] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.
- [3] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Leo Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024.
- [4] Zefan Cai, Haoyi Qiu, Tianyi Ma, Haozhe Zhao, Gengze Zhou, Kung-Hsiang Huang, Parisa Kordjamshidi, Minjia Zhang, Wen Xiao, Jiuxiang Gu, et al. Mmgr: Multi-modal generative reasoning. arXiv preprint arXiv:2512.14691, 2025.
- [5] Wentse Chen, Shiyu Huang, Yuan Chiang, Tim Pearce, Wei-Wei Tu, Ting Chen, and Jun Zhu. Dgpo: discovering multiple strategies with diversity-guided policy optimization. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 11390–11398, 2024.
- [6] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [7] Yifei Dong, Fengyi Wu, Yilong Dai, Lingdong Kong, Guangyu Chen, Xu Zhu, Qiyu Hu, Tianyu Wang, Johnalbert Garnica, Feng Liu, et al. Language-conditioned world modeling for visual navigation. arXiv preprint arXiv:2603.26741, 2026.
- [8] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36:79858–79885, 2023.
- [9] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.
- [10] Google DeepMind. Gemini 3.1 Pro Model Card. https://deepmind.google/models/ model-cards/gemini-3-1-pro/, February 2026.
- [11] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [12] Ziyu Guo, Xinyan Chen, Renrui Zhang, Ruichuan An, Yu Qi, Dongzhi Jiang, Xiangtai Li, Manyuan Zhang, Hongsheng Li, and Pheng-Ann Heng. Are video models ready as zero-shot reasoners? an empirical study with the mme-cof benchmark. arXiv preprint arXiv:2510.26802, 2025.
- [13] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.
- [14] Sepidehsadat Sepid Hossieni, Mohammad Amin Shabani, Saghar Irandoust, and Yasutaka Furukawa. Puzzlefusion: Unleashing the power of diffusion models for spatial puzzle solving. Advances in Neural Information Processing Systems, 36:9574–9597, 2023.
- [15] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022.

- [16] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. arXiv preprint arXiv:2503.24290, 2025.
- [17] Yixu Huang, Tinghui Zhu, and Muhao Chen. Learning adaptive reasoning paths for efficient visual reasoning. arXiv preprint arXiv:2604.14568, 2026.
- [18] Ziqi Huang, Ning Yu, Gordon Chen, Haonan Qiu, Paul Debevec, and Ziwei Liu. Vchain: Chain-of-visual-thought for reasoning in video generation. arXiv preprint arXiv:2510.05094, 2025.
- [19] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective. arXiv preprint arXiv:2411.02385, 2024.
- [20] Kling AI. All you need to know about kling video 3.0. https://kling.ai/blog/ kling-video-3-0-ai-director-features-guide, February 2026.
- [21] Rui Li, Yuanzhi Liang, Ziqi Ni, Haibing Huang, Chi Zhang, and Xuelong Li. Growing with the generator: Self-paced grpo for video generation. arXiv preprint arXiv:2511.19356, 2025.
- [22] Yuming Li, Yikai Wang, Yuying Zhu, Zhongyu Zhao, Ming Lu, Qi She, and Shanghang Zhang. Branchgrpo: Stable and efficient grpo with structured branching in diffusion models. arXiv preprint arXiv:2509.06040, 2025.
- [23] Zhong-Zhi Li, Duzhen Zhang, Ming-Liang Zhang, Jiaxin Zhang, Zengyan Liu, Yuxuan Yao, Haotian Xu, Junhao Zheng, Pei-Jie Wang, Xiuyi Chen, et al. From system 1 to system 2: A survey of reasoning large language models. arXiv preprint arXiv:2502.17419, 2025.
- [24] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.
- [25] Xiao Liu, Yifan Zhou, Fabian Weigend, Shubham Sonawani, Shuhei Ikemoto, and Heni Ben Amor. Diff-control: A stateful diffusion-based policy for imitation learning. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7453–7460. IEEE, 2024.
- [26] Yang Luo, Xuanlei Zhao, Baijiong Lin, Lingting Zhu, Liyao Tang, Yuqi Liu, Ying-Cong Chen, Shengju Qian, Xin Wang, and Yang You. V-reasonbench: Toward unified reasoning benchmark suite for video generation models. arXiv preprint arXiv:2511.16668, 2025.
- [27] David McAllister, Songwei Ge, Brent Yi, Chung Min Kim, Ethan Weber, Hongsuk Choi, Haiwen Feng, and Angjoo Kanazawa. Flow matching policy gradients. arXiv preprint arXiv:2507.21053,

- 2025.

[28] Zhiting Mei, Tenny Yin, Ola Shorinwa, Apurva Badithela, Zhonghe Zheng, Joseph Bruno, Madison Bland, Lihan Zha, Asher Hancock, Jaime Fernández Fisac, et al. Video generation models in robotics-applications, research challenges, future directions. arXiv preprint arXiv:2601.07823,

- 2026.

- [29] Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models understand physical principles? In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 948–958, 2026.
- [30] OpenAI. Sora 2 system card. https://openai.com/index/sora-2-system-card/, September 2025.
- [31] OpenAI. GPT-5.5 System Card. https://openai.com/index/gpt-5-5-system-card/, April 2026.
- [32] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [33] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.
- [34] Selena Song, Ziming Xu, Zijun Zhang, Kun Zhou, Jiaxian Guo, Lianhui Qin, and Biwei Huang. Learning plug-and-play memory for guiding video diffusion models. arXiv preprint arXiv:2511.19229, 2025.
- [35] Jingqi Tong, Yurong Mou, Hangcheng Li, Mingzhe Li, Yongzhuo Yang, Ming Zhang, Qiguang Chen, Tianyi Liang, Xiaomeng Hu, Yining Zheng, et al. Thinking with video: Video generation as a promising multimodal reasoning paradigm. arXiv preprint arXiv:2511.04570, 2025.
- [36] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024.
- [37] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [38] Maijunxian Wang, Ruisi Wang, Juyi Lin, Ran Ji, Thaddäus Wiedemer, Qingying Gao, Dezhi Luo, Yaoyao Qian, Lianyu Huang, Zelong Hong, et al. A very big video reasoning suite. arXiv preprint arXiv:2602.20159, 2026.
- [39] Ruisi Wang, Zhongang Cai, Fanyi Pu, Junxiang Xu, Wanqi Yin, Maijunxian Wang, Ran Ji, Chenyang Gu, Bo Li, Ziqi Huang, et al. Demystifing video reasoning. arXiv preprint arXiv:2603.16870, 2026.
- [40] Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.
- [41] Bing Wu, Chang Zou, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Jack Peng, Jianbing Wu, Jiangfeng Xiong, Jie Jiang, et al. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025.
- [42] Siye Wu, Jian Xie, Yikai Zhang, Aili Chen, Kai Zhang, Yu Su, and Yanghua Xiao. Arm: Adaptive reasoning model. arXiv preprint arXiv:2505.20258, 2025.
- [43] Zeyue Xue, Siming Fu, Jie Huang, Shuai Lu, Haoran Li, Yijun Liu, Yuming Li, Xiaoxuan He, Mengzhao Chen, Haoyang Huang, et al. A systematic post-train framework for video generation. arXiv preprint arXiv:2604.25427, 2026.
- [44] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.
- [45] Cheng Yang, Haiyuan Wan, Yiran Peng, Xin Cheng, Zhaoyang Yu, Jiayi Zhang, Junchi Yu, Xinlei Yu, Xiawu Zheng, Dongzhan Zhou, et al. Reasoning via video: The first evaluation of video models’ reasoning abilities through maze-solving tasks. arXiv preprint arXiv:2511.15065, 2025.
- [46] Ruihan Yang, Prakhar Srivastava, and Stephan Mandt. Diffusion probabilistic modeling for video generation. Entropy, 25(10):1469, 2023.
- [47] Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025.
- [48] Chenyu Zhang, Daniil Cherniavskii, Antonios Tragoudaras, Antonios Vozikis, Thijmen Nijdam, Derck WE Prinzhorn, Mark Bodracska, Nicu Sebe, Andrii Zadaianchuk, and Efstratios Gavves. Morpheus: Benchmarking physical reasoning of video generative models with real physical experiments. arXiv preprint arXiv:2504.02918, 2025.

###### [49] Tinghui Zhu, Kai Zhang, Jian Xie, and Yu Su. Deductive beam search: Decoding deducible rationale for chain-of-thought reasoning. arXiv preprint arXiv:2401.17686, 2024.

### A Dataset Generation Details

We generate all training and evaluation instances using rule-based algorithms so that each sample has a known valid trajectory and task metadata for automatic verification. Each generated instance consists of an initial visual state, a textual task instruction, a ground-truth state/action trajectory, and a rendered video. The generation process differs by task, but all domains are designed so that retained samples have verified valid solutions.

###### A.1 Maze Generation

For Maze, we first generate a connected maze graph using standard maze-carving algorithms, including depth-first search, Prim’s algorithm, and Kruskal’s algorithm. Given a generated maze, we set the logical start and goal cells to opposite corners, (0,0) and (n − 1,n − 1), respectively. The ground-truth trajectory is obtained by running shortest-path search on the maze graph from the start to the goal. This produces a sequence of adjacent logical cells that forms a valid path without crossing walls.

To render this logical trajectory as a continuous visual path, we expand logical cell coordinates into pixel-level path coordinates. For every move between adjacent cells, we insert the corresponding corridor cell between them, producing a continuous walkable sequence in the rendered maze. Step directions are then derived by differencing successive coordinates, yielding a sequence of discrete actions in {U,D,L,R}. This action sequence is used to render the ground-truth video and to support verifier-based evaluation of connectivity and wall consistency.

###### A.2 FlowFree Generation

For FlowFree, we generate the solution first and derive the puzzle from it. We construct a Hamiltonian path over the n × n grid using Warnsdorff-style search: at each step, the algorithm prioritizes neighboring cells with the fewest onward unvisited moves, with random tie-breaking and bounded retries. This produces a full-coverage path over the grid.

We then partition the Hamiltonian path into k contiguous segments, where k is the number of colors. The split indices are sampled randomly subject to the constraint that each segment contains at least two cells. For each segment, the two endpoints become the colored endpoint dots shown in the puzzle, while the full segment is retained as the ground-truth flow for that color. Because the puzzle is constructed from the solution trajectory, every generated instance has a known valid solution by construction. The verifier can therefore check endpoint preservation, color connectivity, non-overlap, and grid coverage against the stored segment metadata.

###### A.3 Sokoban Generation

For Sokoban, we first synthesize a candidate level by sampling connected floor cells, player position, box positions, and target positions. During sampling, we apply basic deadlock filtering to avoid trivially unsolvable configurations, such as boxes placed in wall corners where they cannot be pushed out. Candidate levels are then passed to a symbolic Sokoban solver.

The solver performs breadth-first search over the discrete state space. Each state is represented by the player position and the set of box positions, written as (p,B) where p is the player cell and B is a set of occupied box cells. At each expansion, the solver considers the four possible player moves. If the target cell is empty floor, the player moves without changing the box set. If the target cell contains a box, the move is valid only when the cell beyond the box is also empty floor; in that case, the box is pushed one cell forward. The goal condition is satisfied when the set of box positions matches the set of target positions.

Because breadth-first search explores states in increasing action length, the first returned solution is an optimal-move trajectory under this transition model. To control generation cost, we cap the number of expanded states. Candidate levels that cannot be solved within the cap are discarded and regenerated. Thus, every retained Sokoban sample has a verified valid trajectory, along with process-level metadata for checking player motion, box motion, illegal pulls, teleportation, and final target satisfaction.

[Figure 41]

- Figure 5: Qualitative example of reward hacking in the absence of KL-divergence regularization. It preserves the wall constraint, and saturates all paths to connect two endpoints, thereby achieving a maximal reward.

### B Experimental Setup

###### B.1 Dataset

- • Maze: We generate 10,000 samples with grid dimensions ranging from 7×7 to 21×21. Each instance pairs an unsolved layout containing start and goal markers with a ground-truth video that renders the contiguous path between them.
- • FlowFree: We generate 10,000 puzzles with grid sizes between 5×5 and 8×8 by splitting Hamiltonian paths into colored segments, ensuring that a valid solution must occupy all available cells. The initial frame displays only the discrete color-pair endpoints, and the video unrolls the progressive coloring of each path.
- • Sokoban: This domain includes 10,000 puzzles with grid sizes from 6×6 to 10×10 and 1–3 boxes. Solution trajectories are capped at 60 moves. The input frame depicts the initial board configuration, while the video unrolls the solution at a resolution of one agent push per frame, ensuring strict alignment between temporal and logical steps.

###### B.2 Evaluation

Trajectory Alignment Metrics. To measure the alignment with the ground-truth reference, we compute precision, recall, and F1 at the unit most natural to each task’s solution manifold:

- • Maze (Pixel-level): We compute a change mask between the initial and final frames to isolate the generated path from the static background. This ensures the metric captures the model’s intervention rather than background reconstruction.
- • FlowFree (Cell-level): We extract mean colors for each grid cell in the terminal frame. This avoids penalizing anti-aliasing artifacts and focuses on the semantic correctness of the path coloring.
- • Sokoban (Action-level): Since multiple trajectories can yield the same final state, we decode the video into a symbolic action sequence a ∈ {U(Up),D(Down),L(Left),R(Right)}. We report position-aligned F1, which penalizes out-of-order or invalid moves.

Symbolic Success Rate. Alignment with a single GT reference is insufficient to detect valid alternative solutions or visually plausible but rule-violating outputs. We therefore implement binary success detectors that parse the video V into symbolic states to verify task-specific rules:

- • Maze Success: Requires a connected path between markers without violating wall constraints.
- • FlowFree Success: Validates endpoint preservation, color connectivity, and the grid fill-rate.
- • Sokoban Success: Evaluates process validity over all frames, checking that player and box displacements follow physics-based rules and the final state matches the target.

- C Analysis C.1 KL Constraint

To study the role of KL regularization, we compare VideoRLVR training with and without the KL penalty while keeping all other hyperparameters fixed. Both runs start from the same SFT checkpoint, use the same group size, denoising-step budget, reward function, and training prompts, but differ in the KL coefficient β. The regularized setting uses the default value β = 0.04, while the ablated setting sets β = 0, removing the constraint to the SFT reference policy. We inspect generations from an early stage of GRPO training, where policy drift and reward-hacking behavior are most visible. This controlled comparison isolates the effect of the KL term on preserving the model’s visual prior during verifier-driven optimization.

