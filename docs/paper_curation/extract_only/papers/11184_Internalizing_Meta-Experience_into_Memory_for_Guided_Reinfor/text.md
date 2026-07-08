### Internalizing Meta-Experience into Memory for Guided Reinforcement Learning in Large Language Models

# arXiv:2602.10224v1[cs.LG]10Feb2026

##### Shiting Huang1 Zecheng Li1 Yu Zeng1 Qingnan Ren1 Zhen Fang1 Qisheng Su1

Kou Shi1 Lin Chen1 Zehui Chen1 Feng Zhao1 1University of Science and Technology of China

: Corresponding Author

#### Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as an effective approach for enhancing the reasoning capabilities of Large Language Models (LLMs). Despite its efficacy, RLVR faces a meta-learning bottleneck: it lacks mechanisms for error attribution and experience internalization intrinsic to the human learning cycle beyond practice and verification, thereby limiting fine-grained credit assignment and reusable knowledge formation. We term such reusable knowledge representations derived from past errors as meta-experience. Based on this insight, we propose Meta-Experience Learning (MEL), a novel framework that incorporates selfdistilled meta-experience into the model’s parametric memory. Building upon standard RLVR, we introduce an additional design that leverages the LLM’s self-verification capability to conduct contrastive analysis on paired correct and incorrect trajectories, identify the precise bifurcation points where reasoning errors arise, and summarize them into generalizable meta-experience. The meta-experience is further internalized into the LLM’s parametric memory by minimizing the negative log-likelihood, which induces a language-modeled reward signal that bridges correct and incorrect reasoning trajectories and facilitates effective knowledge reuse. Experimental results demonstrate that MEL achieves consistent improvements on benchmarks, yielding 3.92%– 4.73% Pass@1 gains across varying model sizes.

#### 1. Introduction

Reinforcement Learning (RL) has emerged as a pivotal paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs) on complex tasks, such as mathematics, programming, and logic reasoning (Shao et al., 2024; Chen et al., 2025; Zeng et al., 2025a; Wang et al.,

2025; Zeng et al., 2025b; 2026; Huang et al., 2026). By leveraging feedback signals obtained from interaction with the task environment, RL enables LLMs to move beyond passive imitation learning toward goal-directed reasoning and action (Schulman et al., 2017; Ouyang et al., 2022; Wulfmeier et al., 2024). Furthermore, by replacing learned reward models with programmatically verifiable signals, Reinforcement Learning with Verifiable Rewards (RLVR) eliminates the need for expensive human annotations and mitigates reward hacking, thereby enabling models to explore problem-solving strategies more effectively, which has contributed to its growing attention (Lambert et al., 2024).

However, RLVR still faces a fundamental bottleneck regarding the granularity and utilization of learning signals. From a meta-learning perspective, the human learning cycle involves three critical components: practice and verification, error attribution, and experience internalization. While RLVR primarily drives policy updates through practice and verification, it overlooks the critical stages of error attribution and experience internalization, both of which are essential for fine-grained credit assignment and the formation of reusable knowledge (Wu et al., 2025; Zhang et al., 2025a). In other words, RLVR is largely limited to assessing the overall quality of entire trajectories, while struggling to reason about fine-grained knowledge at the level of intermediate steps (Xie et al., 2025). Although RL approaches (Lightman et al., 2023; Khalifa et al., 2025) employing Process Reward Models (PRMs) to provide dense learning signals attempt to mitigate this limitation, their reliance on trained proxies makes them inherently susceptible to reward hacking (Cheng et al., 2025; Guo et al., 2025), and poses a fundamental tension with the RLVR paradigm, which is centered on programmatically verifiable rewards.

Recently, a growing number of studies have explored integrating experience learning within the RLVR framework to address the above challenge. Early attempts, such as StepHint (Zhang et al., 2025b) utilizes experience as hints to elicit superior reasoning paths from the original problems, treating these trajectories as off-policy migration signals.

Policy Model

RLVR, which relies on coarse-grained outcome rewards and treats correct and incorrect trajectories independently, by explicitly connecting them via meta-experiences. Hence, this process can be viewed as a language-modeled processlevel reward signal, providing continuous and fine-grained guidance for improving reasoning capability. To further enhance stability and effectiveness during RLVR training, we propose empirical validation via replay, which uses metaexperiences as auxiliary in-context hints to assess their contribution to output accuracy. Meta-experiences that pass validation are integrated via negative log-likelihood minimization, while those that fail validation are excluded. In summary, our main contributions are as follows:

(a) Standard RLVR (b) MEL (Ours)

###### Q

###### Q

[Figure 1]

###### Meta-Experience

Reward=1 Reward=0 TrajectoryLevel optimization

Reward=1 Reward=0 TrajectoryLevel optimization

(bifurcation point, critique, heuristic)

Encourage

Suppress

Encourage

Suppress

Knowledgelevel optimization

[Figure 2]

[Figure 3]

- • We propose MEL, a novel framework that integrates self-distilled meta-experience with reinforcement learning, addressing the limitations of standard RLVR in error attribution and experience internalization by embedding these meta-experiences directly into the parametric memory of LLMs.
- • We validate the effectiveness of MEL through extensive experiments on five challenging mathematical reasoning benchmarks across multiple LLM scales (4B, 8B, and 14B). Compared with both the vanilla GRPO baseline and the corresponding base models, MEL consistently improves performance across Pass@1, Avg@8, and Pass@8 metrics.
- • Empirical results confirm that MEL seamlessly integrates with diverse paradigms (e.g., RFT, GRPO, REINFORCE++) to reshape reasoning patterns and elevate performance ceilings. Notably, these benefits exhibit strong scalability, becoming increasingly pronounced as model size expands.

- Figure 1. Paradigm comparison between standard RLVR and MEL, where MEL extends RLVR with an explicit knowledge-level learning loop.

However, the resulting off-policy deviation in response distribution can compromise optimization stability, undermining the theoretical benefits of on-policy reinforcement learning. To alleviate such instability, Scaf-GRPO (Zhang et al., 2025d) leverages superior models to generate multi-level knowledge-intensive experience, injecting them as on-policy prefixes for policy updates. Yet, while effective in teaching models to reason within specific experience-augmented distributions, such prefixes are unavailable during inference, inducing a severe distributional mismatch, thereby limiting performance gains. Critically, despite their differences, these approaches utilize retrieved experience primarily as external hints. While these strategies effectively elicit better reasoning paths during training, the resulting learning signals remain predominantly at the trajectory-level, yielding superficial corrections rather than intrinsic cognitive enhancements.

#### 2. Related Work

Reinforcement Learning with Verifiable Rewards. Reinforcement Learning with Verifiable Rewards (RLVR) leverages rule-based validators to provide deterministic feedback on models’ self-generated solutions (Lambert et al., 2024). Extensive research has systematically investigated RLVR, exploring how this paradigm improves the performance of complex reasoning (Jaech et al., 2024; Guo et al., 2025; Liu et al., 2025; Zhang et al., 2025c). The pioneering framework Group Relative Policy Optimization (GRPO) (Shao et al., 2024) estimates advantages via groupwise relative comparisons, eliminating the need for a separate value model. Building on this base method, recent studies have introduced a range of algorithmic variants to improve training stability and efficiency. For instance, REINFORCE++ (Hu, 2025) enhances stability through global advantage normalization; DAPO (Yu et al., 2025) mitigates

Building on this insight, we introduce the concept of metaexperience, elevating experience learning from trajectorylevel instances to knowledge-level representations. Through contrastive analysis on paired correct and incorrect trajectories, we pinpoint the bifurcation points underlying reasoning failures and abstracts them into reusable metaexperiences. Accordingly, we propose Meta-Experience Learning (MEL), a framework explicitly designed to enable knowledge-level internalization and reuse of metaexperiences. During training phase, MEL leverages metaexperiences to inject generalizable insights via a selfdistillation mechanism, and internalizes them by minimizing the negative log-likelihood in the model’s parametric memory. As shown in Figure 1, MEL differs from standard

[Figure 4]

[Figure 5]

Question

Meta-Experience

[Figure 6]

[Figure 7]

[Figure 8]

## + +

Abstraction & Validation

[Figure 9]

Policy Model

Knowledge-Level Optimization

Heuristic ℋ

Bifurcation Point 𝑠∗ Critique 𝒞

###### Reinforcement Learning with Verifiable Rewards

Reward Advantage

[Figure 10]

Contrastive Pair

- y1
- y2 ⋮

[Figure 11]

- r1
- r2 ⋮

- A1
- A2 ⋮

Group Norm

⋯ ⋯

[Figure 12]

⋯ ⋯

yG

rG

AG

Trajectories

Trajectory-Level Optimization

- Figure 2. Overview of Meta-Experience Learning (MEL), which constructs meta-experiences from contrastive pairs via abstraction and validation, thereby introducing an explicit knowledge-level learning loop on top of standard RLVR.

entropy collapse and improves reward utilization via relaxed clipping and dynamic sampling; and GSPO (Zheng et al., 2025) reduces gradient estimation variance with sequencelevel clipping. Despite these algorithmic advancements, a fundamental limitation persists: current RLVR methods predominantly rely on outcome-level rewards. This failure to assign fine-grained credit to specific knowledge points prevents the construction of reusable knowledge formation, fundamentally constraining the development of systematic and generalizable reasoning capabilities.

incorporate expert solution traces as additional experience. Despite improving exploration efficiency, these methods primarily induce trajectory-level imitation. Consequently, models become proficient at following specific patterns but fail to develop the meta-cognitive understanding required for establishing reusable knowledge structures.

#### 3. Meta-Experience Learning

Human learning follows a recurrent cognitive cycle consisting of practice and verification, error attribution, and experience internalization, which in turn informs subsequent practice. Motivated by this cognitive process, we define meta-experience for LLMs as generalizable and reusable knowledge derived from accumulated reasoning trials, capturing both underlying knowledge concepts and common failure modes. Building on this notion, we propose MetaExperience Learning (MEL), a framework operating within the RLVR paradigm and expressly designed to internalize such self-distilled, knowledge-level insights into the model’s parametric memory. As illustrated in Figure 2, we first formalize the model exploration stage in RLVR (§3.1), then present the details of the Meta-Experience construction (§3.2). Finally, we describe the internalization mechanism (§3.3) for consolidating these insights into parametric memory, followed by the joint training objective for policy optimization (§3.4).

Experience Learning. Recent studies have increasingly recognized that leveraging various forms of experience can substantially enhance LLM reasoning capabilities. One prominent line of research lies in test-time scaling methods, which store experience in external memory pools. For example, SpeedupLLM (Pan & Zhao, 2025) appends memories of previously reasoning traces as experience to accelerate inference, while Training Free GRPO (Cai et al., 2025) and ReasoningBank (Ouyang et al., 2025) distill accumulated experience into structured memory entries for retrieval-based augmentation. However, these approaches rely on evergrowing external memory, preventing the experience from being truly internalized and thus failing to substantively enhance intrinsic reasoning capabilities. Complementarily, another stream of research integrates experience directly into RL training as guiding signals. Methods such as Scaf-GRPO (Zhang et al., 2025d) and StepHint (Zhang et al., 2025b) employ external models to generate experiential hints, which are injected as prefixes or migration signals, to guide the policy toward higher-quality trajectories. Similarly, approaches like LUFFY (Yan et al., 2025) and SRFT (Fu et al., 2025)

##### 3.1. Explorative Rollout and Verifiable Feedback

Mirroring the “practice and check” phase in human learning, the RLVR framework engages the model in exploring po-

tential solutions for reasoning tasks, while the environment serves as a deterministic verifier that provides verifiable feedback on the final answers. As mastering complex logic typically requires traversing the solution space through multiple attempts, we simulate this stochastic exploration by adopting the group rollout formulation from Group Relative Policy Optimization (GRPO) (Shao et al., 2024).

Formally, given a query x sampled from the dataset D, the policy model πθ performs stochastic exploration over the solution space and generates a group of G independent reasoning trajectories Y = {y1,y2,...,yG}. A rule-based verifier then evaluates each trajectory using a verification function V (·), which compares the extracted final answer from yi against the ground-truth answer y∗ and assigns a binary outcome reward:

ri = I V (yi,y∗) ∈ {0,1}. (1)

This process partitions the generated group Y into two distinct subsets: the set of correct trajectories Y+ = {yi | ri = 1} and the set of incorrect trajectories Y− = {yi | ri = 0}. The coexistence of Y+ and Y− under the same prompt distribution suggests that the model is capable of solving the task, while producing diverse reasoning trajectories. For our method, such diversity constitutes a beneficial property and serves a dual role. On the one hand, it supplies the variance necessary for effective policy updates in standard RLVR. On the other hand, it enables the extraction of knowledgelevel meta-expression through systematic contrast between correct and incorrect reasoning outcomes.

##### 3.2. Meta-Experience Construction

Prior studies (Xie et al., 2025; Khalifa et al., 2025; Huang et al., 2025) have shown that effective learning does not arise from merely knowing that a final answer is incorrect, but rather from identifying the specific bifurcation point at which the reasoning process deviates from the correct trajectory, a critical cognitive process known as error attribution. Building on this insight, we leverage pairs of correct and incorrect trajectories to localize reasoning errors and distill such bifurcation points into explicit meta-experiences.

Locating the Bifurcation Point. To extract knowledgelevel learning signals from the exploration results, we focus on identifying the bifurcation points where the reasoning logic diverges into an erroneous path. With the exploration results partitioned into Y+ and Y− by the verifier, we construct a set of contrastive pairs Px = {(y+,y−) | y+ ∈ Y+, y− ∈ Y−} for each query x, whose contrast naturally exposes the specific errors in the reasoning process. Such contrastive analysis requires the presence of both positive and negative trajectories; accordingly, we only consider gradient-informative samples with non-empty Y+ and Y−.

For fine-grained comparison within each pair, each trajectory y can be formatted as a reasoning chain y = (s1,s2,...,sL,a), where each st denotes an atomic reasoning step and a indicates the final answer. Since both trajectories originate from the same context, they typically share a correct reasoning path until a critical divergence step s∗ occurs.

Given deterministic verification signals and full access to the reasoning chains, identifying the bifurcation point can be viewed as a discriminative task that is easier than reasoning from scratch (Saunders et al., 2022; Swamy et al., 2025). Motivated by this observation, we task the policy model with analyzing each contrastive pair to identify the reasoning bifurcation point s∗:

s∗ ∼ πθ · | I,x,y+,y− . (2)

Where I denotes a structured instruction guiding introspective analysis.

Deep Diagnosis and Abstraction. Identifying the bifurcation point s∗ localizes where the reasoning fails, serving as the raw material for subsequent learning. Anchored at s∗, the policy model conducts a deep diagnostic analysis to contrast the strategic choices underlying the successful and failed trajectories. Specifically, the model examines the local reasoning context around s∗ to pinpoint the root cause of failure, such as violated assumptions, erroneous sub-goals, overlooked constraints, or the misuse of specific principles. Complementarily, it inspects the successful trajectory to uncover the mechanisms that prevented such pitfalls, including precise knowledge application, explicit constraint verification, coherent knowledge representations, or emergent self-correction behaviors. By jointly synthesizing these perspectives, the model distills the structural divergence between the correct and incorrect logic, crystallizing it into explicit knowledge. Formally, we model this diagnostic process as generating a critique C that encapsulates the error attribution, the comparative strategic gap, and the corresponding corrective principle:

C ∼ πθ · | I,x,y+,y−,s∗ . (3)

To ensure generalization, it is imperative for the model to distill instance-specific critiques into abstract heuristics capable of guiding future reasoning. This abstraction mechanism systematically strips away context-dependent variables, mapping the concrete logic of success and failure onto a generalized space of preconditions and responses. Structurally, such heuristics synthesize abstract problem categorization with the corresponding reasoning principles, encompassing the essential knowledge points, theoretical theorems, and decision criteria. Crucially, they also demarcate error-prone boundaries, explicitly highlighting potential pitfalls or latent constraints associated with the specific

problem class. We define the extraction of this heuristic knowledge H as a generation process conditioned on the full critique context:

H ∼ πθ · | I,x,y+,y−,s∗,C . (4)

Finally, we consolidate these components into a unified Meta-Experience tuple M, which elevates experience learning from trajectory-level instances to knowledge-level representations.

M = s∗,C,H . (5)

This formulation enables meta-experiences to be reused across tasks that share analogous reasoning structures, serving as a fine-grained learning signal. By applying the extraction process across distinct contrastive pairs for a query x, we construct a candidate pool of meta-experiences DM = {(x,yi+,yi−,Mi)}Ni=1, where N denotes the total number of meta-experiences derived from x, and (yi+,yi−) represents the specific contrastive pair used to derive Mi.

Empirical Validation via Replay. Closing the cognitive loop requires re-instantiating theoretical insights derived from past failures in future problem-solving to assess their validity. We recognize that the raw meta-experience M may still suffer from intrinsic hallucinations or causal misalignment. To mitigate this, we conduct empirical verification by incorporating the extracted tuple M as short-term working memory into the prompt, thereby guiding the model to reattempt the original query x. This procedure tests whether the injected meta-experience can effectively steer the model away from the previously identified bifurcation point s∗ and toward a correct reasoning trajectory.

We retain a meta-experience only if the corresponding replay trajectory yval ∼ πθ(· | x,M) satisfies the verifier by producing the correct answer.

DM∗ = (x, y+, y−, M) ∈ DM I V (yval, y∗) = 1 . (6)

Consequently, this empirical validation preserves only highquality meta-experiences for integration into parametric long-term memory, guaranteeing the reliability of the supervision signals used in the subsequent optimization phase.

##### 3.3. Internalization Mechanism

The verified meta-experiences DM∗ constitute a highquality reservoir of reasoning guidance. However, treating these insights solely as retrieval-augmented memory imposes a substantial computational burden during the inference forward pass, as it necessitates processing elongated contexts for every query. To overcome this limitation, we propose to transfer these insights from the transient context window to the model’s parametric memory. Unlike the finite

context buffer, the model parameters offer a virtually unlimited capacity for accumulating diverse meta-experiences, allowing the policy to internalize vast amounts of reasoning patterns without incurring inference-time latency.

We establish this internalization process as a self-distilled paradigm, where the model learns from its own verified experiences. Specifically, we employ fine-tuning based on the token-averaged negative log-likelihood (NLL) objective to compile the meta-experiences into the policy’s weights. Formally, given the retrospective context Cretro = [I,x,y+,y−], the internalization loss is defined as:

LNLL(θ) = −E(x,y+,y−,M∗)∼DM∗ 1 |M∗|

|M∗|

log πθ(M∗t | Cretro, M∗<t)

t=1

= −Ex∼D,{y

i}Gi=1∼πθold(·|x)

E(y+,y−,M∗)∼T (x,{yi}Gi=1)

(7)

|M∗|

1 |M∗|

log πθ(M∗t | Cretro, M∗<t)

t=1

where T (·) represents the stochastic construction process detailed in §3.2.

Based on this formulation, the internalization process can be viewed as a specialized sampling form within the RLVR framework. By inverting the loss, we define the MetaExperience Return RMEL as the expected log-likelihood over the stochastically constructed verification set:

RMEL = E(y+,y−,M∗)∼T (x,{yi}Gi=1)

|M∗|

1 |M∗|

log πθ(M∗t | Cretro, M∗<t) .

t=1

(8)

##### 3.4. Joint Training Objective

To simultaneously encourage solution exploration and consolidate the internalized meta-experiences, achieving dual optimization across trajectory-level behaviors and knowledge-level representations, we train the policy model πθ using a joint optimization objective. To simultaneously encourage solution exploration and consolidate the internalized meta-experiences, achieving dual optimization across trajectory-level behaviors and knowledge-level representations, we train the policy model πθ using a joint optimization objective. This objective synergizes the RLVR signal derived from diverse explorative rollouts with the supervised signal distilled from high-quality meta-experiences:

J (θ) = JRLVR(θ) + JMEL(θ). (9)

We adopt GRPO (Shao et al., 2024) as the RLVR component and compute group-normalized advantages by standardizing

Table 1. Main Results Comparison. Comparison of Pass@1, Avg@8, and Pass@8 accuracy (%) across different model scales. The best results within each model scale are marked in bold.

AIME 2024 AIME 2025 AMC 2023 Method Pass@1 Avg@8 Pass@8 Pass@1 Avg@8 Pass@8 Pass@1 Avg@8 Pass@8 Qwen3-4B-Base

- Baseline 13.33 9.90 30.00 10.00 6.56 23.33 45.00 42.73 72.50 GRPO 13.33 18.33 30.00 6.67 17.50 30.00 57.50 58.13 85.00 MEL 20.00 20.83 33.00 16.67 18.33 33.00 60.00 60.31 87.50

Qwen3-8B-Base

Baseline 6.67 10.00 26.67 13.33 15.00 33.33 65.00 52.50 87.50 GRPO 16.67 24.58 43.33 20.00 20.83 36.67 67.50 69.06 87.50 MEL 30.00 25.42 60.00 23.33 23.33 36.67 70.00 70.31 90.00

Qwen3-14B-Base

- Baseline 13.33 10.83 36.67 6.66 9.58 33.33 60.00 51.25 82.50 GRPO 30.00 35.41 56.67 33.33 24.17 43.33 75.00 75.94 95.00 MEL 33.33 35.83 60.00 36.67 30.00 46.67 82.50 82.81 95.00

MATH 500 OlympiadBench Average Method Pass@1 Avg@8 Pass@8 Pass@1 Avg@8 Pass@8 Pass@1 Avg@8 Pass@8 Qwen3-4B-Base

Baseline 74.20 65.74 89.60 39.17 35.37 60.38 36.34 32.06 55.16 GRPO 81.80 82.20 93.00 48.51 48.46 67.21 41.56 44.92 61.04 MEL 82.20 82.30 93.80 48.51 49.48 69.73 45.48 46.25 63.41

Qwen3-8B-Base Baseline 77.00 73.40 91.40 44.51 39.41 64.09 41.30 38.06 60.60

- GRPO 84.40 86.28 95.40 53.56 54.60 73.74 48.43 51.07 67.33 MEL 86.60 86.70 96.20 54.01 55.60 73.00 52.79 52.27 71.17

Qwen3-14B-Base Baseline 80.80 74.15 93.60 45.25 40.50 65.58 41.21 37.26 62.34

- GRPO 85.00 88.35 96.40 58.16 58.46 74.78 56.30 56.47 73.24 MEL 90.80 90.80 97.20 61.87 60.90 75.82 61.03 60.07 74.94

rewards within the sampled group and broadcast them to each token. Let yi,t denote the t-th token in trajectory yi and yi,<t, the corresponding prefix. Substituting the definition of RMEL from Eq. 8, the joint objective in Eq. 9 is explicitly expanded as:

J (θ) =Ex∼D,{y

i}Gi=1∼πθold(·|x)

|yi|

G

1 G

1 |yi|

min ρi,t(θ)Aˆi,t,

t=1

i=1

clip ρi,t(θ), 1 − ϵ, 1 + ϵ A ˆi,t + RMEL .

(10)

Although derived from a log-likelihood objective, its optimization gradient is mathematically equivalent to a policy gradient update where the reward signal is a constant positive scalar. Consequently, the total objective J (θ) can be unified as maximizing the expected cumulative return of

a hybrid reward function. In this unified view, the metaexperiences function as a dense process reward model.

Unlike the sparse outcome rewards in standard RLVR that only evaluate the final correctness, RMEL provides explicit, step-by-step reinforcement for the reasoning process itself. This ensures that the model not only pursues correct outcomes via broad exploration but is also continuously shaped by the dense supervision of its own successful cognitive patterns, effectively bridging the gap between trajectory-level search and token-level knowledge encoding.

#### 4. Experiments

Datasets. We train our model on the DAPO-Math-17k dataset (Yu et al., 2025) and evaluate it on five challenging mathematical reasoning benchmarks: AIME24, AIME25, AMC23 (Li et al., 2024), MATH500 (Hendrycks et al.,

2021), and OlympiadBench (He et al., 2024).

Setups. All reinforcement learning training is conducted using the VERL framework (Sheng et al., 2024) on 8×H20 GPUs, with Math-Verify providing rule-based outcome verification. During training, we sample 8 responses per prompt at a temperature of 1.0 with a batch size of 128. Optimization uses a learning rate of 1 × 10−6 and a mini-batch size of 128. For evaluation, we report Pass@1 at temperature 0, and Avg@8 and Pass@8 at temperature 0.6.

Models and Baselines. To demonstrate the general applicability of MEL, we conduct experiments across a diverse range of model scales, including Qwen3-4B-Base, Qwen38B-Base, and Qwen3-14B-Base (Yang et al., 2025). We adopt GRPO (Shao et al., 2024) as the base reinforcement learning algorithm for MEL, and thus perform a direct and controlled comparison between the vanilla GRPO and our meta-experience learning approach.

##### 4.1. Experimental Results

As shown in Table 1, MEL achieves consistent and significant improvements over vanilla GRPO and the basemodel across multiple benchmarks and model scales. We report three complementary metrics: Pass@1 reflects one-shot reliability, Avg@8 measures the average performance over 8 samples, and Pass@8 reports the best-of-8 success rate.

First, the gains in Pass@1 demonstrate that MEL substantially enhances the model’s confidence in following correct reasoning paths. Across all model scales, it achieves a consistent improvement of 3.92–4.73% over the strong GRPO baseline. This indicates that MEL effectively internalizes the explored insights into the model’s parametric memory. By consolidating these successful reasoning patterns, the model generates high-confidence solutions, markedly reducing the need for extensive test-time sampling. This reliability is further corroborated by the gains in Avg@8, which reveal that MEL significantly enhances reasoning consistency and output stability. High performance on this metric supports our hypothesis that internalized meta-experiences function as intrinsic process-level guidance, continuously steering the generation toward valid logic and effectively reducing variance across sampled outputs. Finally, the sustained gains in Pass@8 suggest that learning from metaexperience does not harm exploration; instead, it expands the reachable solution space and raises the upper bound of best-of-k performance.

##### 4.2. Training Dynamics and Convergence Analysis

To understand the mechanisms driving the performance gains under MEL, we monitored the training dynamics and validation performance in Figures 3 and 6–8.

0.7

- 0.2

- 0.3

- 0.4

- 0.5

- 0.6 GRPO (8B)

GRPO (4B)

GRPO (14B)

MEL (4B)

MEL (8B)

MEL (14B)

0.6

0.4

TrainingReward

0.5

0.3

0.4

0.3

0.2

0.2

0 20 40 60 80 100 120

0 20 40 60 80 100 120

0 20 40 60 80 100 120

Training Steps

Training Steps

Training Steps

Figure 3. Training curves comparing GRPO and MEL.

Vanilla GRPO methods often struggle to obtain positive reinforcement in the early stages, particularly when initial performance is low, due to the sparsity of outcome-based rewards. As illustrated in the training curve, vanilla GRPO exhibits a relatively slow ascent during the initial phase. In contrast, MEL demonstrates a sharp, rapid trajectory growth immediately from the onset of training. This acceleration is attributed to the internalized meta-experience return, RMEL. By functioning as a dense, language-modeling process reward, RMEL continuously provides informative gradient signals for every reasoning step, even when successful trajectories yielding positive reinforcement are scarce.

Beyond sample efficiency, MEL achieves a consistently higher performance upper bound. The training curves show that the average reward of MEL consistently surpasses that of vanilla GRPO throughout the entire training process. Crucially, the downstream validation trajectories reveal that even as performance growth begins to plateau in the later stages, MEL maintains a distinct and sustained advantage over the baseline. This phenomenon demonstrates that the internalization of meta-experiences empowers the model to effectively navigate and explore more complex, longhorizon solutions that remain inaccessible to the baseline.

##### 4.3. How Meta-Experience Shapes Reasoning Patterns

To investigate how MEL shapes the model’s cognitive processes beyond numerical metrics, we conduct a qualitative analysis comparing the reasoning trajectories of MEL and the baseline GRPO model, as visualized in Figure 4.

A distinct behavioral divergence is observed from the onset of the solution. While the GRPO baseline tends to prioritize immediate execution through direct numerical operations, MEL adopts a structured preparatory strategy by explicitly outlining relevant theorems and formulas. Although the direct approach may appear efficient for simple queries, it increases the susceptibility to errors in complex tasks due to the lack of a holistic view of problem constraints.

Notably, MEL exhibits an emergent cognitive behavior. When applying specific theorems, it spontaneously activates internalized “bitter lessons” as endogenous safeguards to regulate its actions. These active signals effectively reduce reasoning drift by encouraging earlier constraint checking

[Figure 13]

Question

Meta-Experience in Early Stage

1. Failure Resolution Path & Error Pattern Recognition Failure Point: The error occurs in the use of the circumcircle formula. The formula should be BC = 2R sin ∠BAC, but the incorrect solution uses BC = 2R sin(∠BAC / 2).

Triangle ABC has AB = 4, BC = 5, and CA = 6. Points A', B',

C' are such that B'C' is tangent to the circumcircle of triangle ABC at A, C'A' is tangent to the circumcircle at B, and A'B' is tangent to the circumcircle at C. Find the length B'C'.

Latent Cognitive Pattern: The error is due to a conceptual confusion in formula usage.

###### ...

[Figure 14]

4. Subject Heuristics (Internalized Experience) Angle Verification Rule: When using a formula of the form 2Rsin θ in a circumcircle, always ensure that θ is the full geometric angle spanning the chord or tangent segment, not a derived half-angle. Formula–Geometry Consistency Rule: Before applying trigonometric formulas for lengths on the circumcircle, confirm that the chosen angle corresponds exactly to the geometric length being calculated, in order to avoid errors caused by half-angle substitution.

GRPO

Alright. I have this geometry problem about a triangle and some tangents to its circumcircle.

[Figure 15]

... Step 1: Find the circumcircle of triangle ABC. Semi-perimeter:

MEL

Alright, I have this geometry problem about a triangle and some tangent lines.

###### ...

s = (4 + 5 + 6) / 2 = 7.5

This seems like a problem involving tangent lines and the properties of a triangle's circumcircle. I think I can use some properties of tangents and maybe some trigonometry or coordinate geometry to solve it. First, I need to recall some properties:

Area:

K = sqrt(s(s - a)(s - b)(s - c)) = sqrt(7.5 × 3.5 × 2.5 × 1.5)

- 1. Tangent to a Circle: A tangent to a circle is perpendicular to the radius at the point of tangency.
- 2. Circumcircle: The circle that passes through all three vertices of a triangle. Maybe I can find the radius of the circumcircle first, and then use the tangent properties to find B'C'. Step 1: Find the Circumradius R of triangle ABC. I remember there‘s a formula to find the circumradius of a triangle: R = abc / (4K), where a, b, c are the side lengths, and K is the area of the triangle. First, I need to find the area K of triangle ABC. I can use Heron's formula: K = sqrt(s(s - a)(s - b)(s - c)), where s = (a + b + c) / 2 is the semi-perimeter.

Let's compute step by step:

... In a triangle, the tangential triangle is the triangle formed by the points where the tangents to the circumcircle at

the vertices intersect.

There's a formula that relates the side lengths of the tang ential triangle to the sides and angles of the original triangle. The formula is:

... Based on the calculations and the relationships I've explored, I think the length B'C' is related to the circumradius and the angles of the triangle and can be calculated using the formula:

- a' = 2R sin(A / 2)
- b' = 2R sin(B / 2)
- c' = 2R sin(C / 2)

B'C' = 2R sin θ where θ is the angle between the tangent lines. When using this formula, θ must be the full geometric angle that directly governs the

where a', b', c' are the sides of the tangential triangle. But

opening of the segment, not a derived half-angle. In this case, θ is the angle at ∠A.

in our case, the side B'C' corresponds to a', so:

###### ...

... Conclusion: The length B'C’ is \\boxed{2}

Final Answer: \\boxed{5}

Figure 4. Case study comparing GRPO and MEL, with visualization of meta-experience in early stage.

and consistent self-correction when the model enters uncertain regions.

##### 4.4. Generality Across Learning Paradigms

OlympiadBench

- 44

44.5

45.0

- 45.5

MATH500

AIME24

80

15

78.3

9.9

76.6

4.9

75

0

50

0

56.6

7.2

63.2

14.5

70

22

AMC23 AIME25

Qwen3-8B-Base

RFT

RFT w. ME

| |
|---|
| |

| |
|---|
| |

| |
|---|
| |

OlympiadBench

55

51.2

47.6

MATH500

AIME24

86

29

82.2

19.1

44

78.6

9.6

75

0

60

8

64.6

13.6

69.2

19.2

74

25

AMC23 AIME25

Qwen3-8B-Base

REINFORCE++

REINFORCE++ w. ME

| |
|---|
| |

| |
|---|
| |

| |
|---|
| |

Figure 5. Impact of meta-experience across different training methods, including Rejection Sampling Fine-Tuning (RFT) and REINFORCE++. ME denotes Meta-Experience.

##### 4.5. Scalability Analysis

As indicated by the training curves in Figure 3, the method exhibits a distinct positive scaling law: the performance margin between MEL and the baseline widens significantly as the model size increases. This phenomenon consistently extends to downstream validation benchmarks.

We attribute this effect to the quality of self-generated supervision, which is inherently bounded by the model’s intrinsic capability. As shown in Figure 9, the 14B model achieves a significantly higher yield rate of valid meta-experiences than its smaller counterparts. While limited-capacity models introduce noise due to imprecise error attribution, larger models benefit from stronger self-verification, enabling the distillation of high-quality heuristics that provide more accurate gradient signals and fully realize the potential of our framework.

To demonstrate the versatility of meta-experience, we integrated it into RFT and REINFORCE++ using the Qwen-8BBase model as the backbone and the same training set in our experiments. As shown in Figure 5, while vanilla RFT often suffers from rote memorization and tends to overfit to specific samples in this training set, the incorporation of meta-experiences introduces robust reasoning heuristics. This allows the model to internalize the underlying logic rather than merely imitating specific answers, thereby effectively mitigating overfitting and enhancing generalization to unseen test sets. Similarly, applying meta-experiences to REINFORCE++ significantly raises the performance ceiling on benchmarks. This confirms that the benefit of internalized meta-experiences is a universal enhancement, not limited to the GRPO framework.

#### 5. Conclusion

In this paper, we introduced MEL, a novel framework designed to overcome the meta-learning bottleneck in standard RLVR by transforming instance-specific failure patterns into reusable cognitive assets. Unlike traditional methods that rely solely on outcome-oriented rewards, MEL empowers models to perform granular error attribution, distilling specific failure modes into natural language heuristics—termed Meta-Experiences. By internalizing these experiences into parametric memory, our approach bridges the gap between verifying a solution and understanding the underlying reasoning logic. Extensive empirical evaluations confirm that MEL consistently boosts mathematical reasoning across diverse model scales.

#### Impact Statement

This paper presents research aimed at advancing the field of reinforcement learning. While our work may have broader societal implications, we do not identify any specific impacts that require particular attention at this stage.

#### References

Cai, Y., Cai, S., Shi, Y., Xu, Z., Chen, L., Qin, Y., Tan, X., Li, G., Li, Z., Lin, H., et al. Training-free group relative policy optimization. arXiv preprint arXiv:2510.08191, 2025.

Chen, J., He, Q., Yuan, S., Chen, A., Cai, Z., Dai, W., Yu, H., Yu, Q., Li, X., Chen, J., et al. Enigmata: Scaling logical reasoning in large language models with synthetic verifiable puzzles. arXiv preprint arXiv:2505.19914, 2025.

Cheng, J., Xiong, G., Qiao, R., Li, L., Guo, C., Wang, J., Lv, Y., and Wang, F.-Y. Stop summation: Min-form credit assignment is all process reward model needs for reasoning. arXiv preprint arXiv:2504.15275, 2025.

Fu, Y., Chen, T., Chai, J., Wang, X., Tu, S., Yin, G., Lin, W., Zhang, Q., Zhu, Y., and Zhao, D. Srft: A single-stage method with supervised and reinforcement fine-tuning for reasoning. arXiv preprint arXiv:2506.19767, 2025.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

He, C., Luo, R., Bai, Y., Hu, S., Thai, Z., Shen, J., Hu, J., Han, X., Huang, Y., Zhang, Y., et al. Olympiadbench: A challenging benchmark for promoting agi with olympiadlevel bilingual multimodal scientific problems. In Proceedings of the Association for Computational Linguistics, pp. 3828–3850, 2024.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv: 2103.03874, 2021.

Hu, J. Reinforce++: A simple and efficient approach for aligning large language models. arXiv preprint arXiv:2501.03262, 2025.

Huang, S., Fang, Z., Chen, Z., Yuan, S., Ye, J., Zeng, Y., Chen, L., Mao, Q., and Zhao, F. Critictool: Evaluating self-critique capabilities of large language models in toolcalling error scenarios. arXiv preprint arXiv:2506.13977, 2025.

Huang, W., Zeng, Y., Wang, Q., Fang, Z., Cao, S., Chu, Z., Yin, Q., Chen, S., Yin, Z., Chen, L., et al. Visiondeepresearch: Incentivizing deepresearch capability in multimodal large language models. arXiv preprint arXiv:2601.22060, 2026.

Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Khalifa, M., Agarwal, R., Logeswaran, L., Kim, J., Peng, H., Lee, M., Lee, H., and Wang, L. Process reward models that think. arXiv preprint arXiv:2504.16828, 2025.

Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., Brahman, F., Miranda, L. J. V., Liu, A., Dziri, N., Lyu, S., et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Li, J., Beeching, E., Tunstall, L., Lipkin, B., Soletskyi, R., Huang, S., Rasul, K., Yu, L., Jiang, A. Q., Shen, Z., et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13(9):9, 2024.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. In Proceedings of the International Conference on Learning Representations, 2023.

Liu, K., Yang, D., Qian, Z., Yin, W., Wang, Y., Li, H., Liu, J., Zhai, P., Liu, Y., and Zhang, L. Reinforcement learning meets large language models: A survey of advancements and applications across the llm lifecycle. arXiv preprint arXiv:2509.16679, 2025.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

Ouyang, S., Yan, J., Hsu, I., Chen, Y., Jiang, K., Wang, Z., Han, R., Le, L. T., Daruki, S., Tang, X., et al. Reasoningbank: Scaling agent self-evolving with reasoning memory. arXiv preprint arXiv:2509.25140, 2025.

Pan, B. and Zhao, L. Can past experience accelerate llm reasoning? arXiv preprint arXiv:2505.20643, 2025.

Saunders, W., Yeh, C., Wu, J., Bills, S., Ouyang, L., Ward, J., and Leike, J. Self-critiquing models for assisting human evaluators. arXiv preprint arXiv:2206.05802, 2022.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Swamy, G., Choudhury, S., Sun, W., Wu, Z. S., and Bagnell, J. A. All roads lead to likelihood: The value of reinforcement learning in fine-tuning. arXiv preprint arXiv:2503.01067, 2025.

Wang, Q., Ding, R., Zeng, Y., Chen, Z., Chen, L., Wang, S., Xie, P., Huang, F., and Zhao, F. Vrag-rl: Empower vision-perception-based rag for visually rich information understanding via iterative reasoning with reinforcement learning. arXiv preprint arXiv:2505.22019, 2025.

Wu, R., Wang, X., Mei, J., Cai, P., Fu, D., Yang, C., Wen, L., Yang, X., Shen, Y., Wang, Y., et al. Evolver: Self-evolving llm agents through an experience-driven lifecycle. arXiv preprint arXiv:2510.16079, 2025.

Wulfmeier, M., Bloesch, M., Vieillard, N., Ahuja, A., Bornschein, J., Huang, S., Sokolov, A., Barnes, M., Desjardins, G., Bewley, A., et al. Imitating language via scalable inverse reinforcement learning. Advances in Neural Information Processing Systems, 37:90714–90735, 2024.

Xie, G., Shi, Y., Tian, H., Yao, T., and Zhang, X. Capo: Towards enhancing llm reasoning through verifiable generative credit assignment. arXiv e-prints, pp. arXiv–2508, 2025.

Yan, J., Li, Y., Hu, Z., Wang, Z., Cui, G., Qu, X., Cheng, Y., and Zhang, Y. Learning to reason under off-policy guidance. arXiv preprint arXiv:2504.14945, 2025.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Zeng, Y., Huang, W., Huang, S., Bao, X., Qi, Y., Zhao, Y., Wang, Q., Chen, L., Chen, Z., Chen, H., et al. Agentic jigsaw interaction learning for enhancing visual perception and reasoning in vision-language models. arXiv preprint arXiv:2510.01304, 2025a.

Zeng, Y., Qi, Y., Zhao, Y., Bao, X., Chen, L., Chen, Z., Huang, S., Zhao, J., and Zhao, F. Enhancing large visionlanguage models with ultra-detailed image caption generation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 26703–26729, 2025b.

Zeng, Y., Huang, W., Fang, Z., Chen, S., Shen, Y., Cai, Y., Wang, X., Yin, Z., Chen, L., Chen, Z., et al. Visiondeepresearch benchmark: Rethinking visual and textual search for multimodal large language models. arXiv preprint arXiv:2602.02185, 2026.

Zhang, K., Chen, X., Liu, B., Xue, T., Liao, Z., Liu, Z., Wang, X., Ning, Y., Chen, Z., Fu, X., et al. Agent learning via early experience. arXiv preprint arXiv:2510.08558, 2025a.

Zhang, K., Lv, A., Li, J., Wang, Y., Wang, F., Hu, H., and Yan, R. Stephint: Multi-level stepwise hints enhance reinforcement learning to reason. arXiv preprint arXiv:2507.02841, 2025b.

Zhang, K., Zuo, Y., He, B., Sun, Y., Liu, R., Jiang, C., Fan, Y., Tian, K., Jia, G., Li, P., et al. A survey of reinforcement learning for large reasoning models. arXiv preprint arXiv:2509.08827, 2025c.

Zhang, X., Wu, S., Zhu, Y., Tan, H., Yu, S., He, Z., and Jia, J. Scaf-grpo: Scaffolded group relative policy optimization for enhancing llm reasoning. arXiv preprint arXiv:2510.19807, 2025d.

Zheng, C., Liu, S., Li, M., Chen, X.-H., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

#### A. Result of Performance Evolution

As illustrated in Figures 6, 7, and 8, we visualize the performance evolution of models with different scales (Qwen3-4B-Base, Qwen3-8B-Base, and Qwen3-14B-Base) across multiple benchmarks throughout training. It can be observed that MEL consistently outperforms standard GRPO in terms of average performance on all benchmarks.

Benchmark: AIME24

0.225

0.200

ValidationScore

0.175

0.150

0.125

0.100

GRPO

0.075

MEL

0 20 40 60 80 100 120 140

Training Step

Benchmark: MATH500

0.84

0.82

ValidationScore

0.80

0.78

0.76

GRPO

MEL

0.74

0 20 40 60 80 100 120 140

Training Step

Benchmark: AIME25

0.200

0.175

0.150

ValidationScore

0.125

0.100

0.075

GRPO

0.050

MEL

0.025

0 20 40 60 80 100 120 140

Training Step

Benchmark: OlympiadBench

0.50

0.48

ValidationScore

0.46

0.44

0.42

GRPO

0.40

MEL

0 20 40 60 80 100 120 140

Training Step

Benchmark: AMC23

0.60

0.58

0.56

ValidationScore

0.54

0.52

0.50

0.48

GRPO

0.46

MEL

0 20 40 60 80 100 120 140

Training Step

Benchmark: Average

0.46

0.44

ValidationScore

0.42

0.40

0.38

GRPO

MEL

0.36

0 20 40 60 80 100 120 140

Training Step

Figure 6. Performance evolution of GRPO and MEL on Qwen3-4B-Base across training steps on multiple benchmarks.

Benchmark: AIME24

0.30

0.25

ValidationScore

0.20

0.15

0.10

GRPO

MEL

0 20 40 60 80 100 120 140

Training Step

Benchmark: MATH500

0.86

0.84

ValidationScore

0.82

0.80

GRPO

0.78

MEL

0 20 40 60 80 100 120 140

Training Step

Benchmark: AIME25

0.275

0.250

0.225

ValidationScore

0.200

0.175

0.150

0.125

GRPO

MEL

0.100

0 20 40 60 80 100 120 140

Training Step

Benchmark: OlympiadBench

0.54

0.52

ValidationScore

0.50

0.48

0.46

GRPO

MEL

0.44

0 20 40 60 80 100 120 140

Training Step

Benchmark: AMC23

0.725

0.700

0.675

ValidationScore

0.650

0.625

0.600

0.575

GRPO

MEL

0.550

0 20 40 60 80 100 120 140

Training Step

Benchmark: Average

0.52

0.50

ValidationScore

0.48

0.46

0.44

GRPO

0.42

MEL

0 20 40 60 80 100 120 140

Training Step

Figure 7. Performance evolution of GRPO and MEL on Qwen3-8B-Base across training steps on multiple benchmarks.

Benchmark: AIME24

0.45

0.40

ValidationScore

0.35

0.30

0.25

0.20

GRPO

0.15

MEL

0 20 40 60 80 100 120 140

Training Step

Benchmark: MATH500

0.90

0.88

ValidationScore

0.86

0.84

0.82

GRPO

MEL

0.80

0 20 40 60 80 100 120 140

Training Step

Benchmark: AIME25

0.35

0.30

ValidationScore

0.25

0.20

0.15

GRPO

0.10

MEL

0 20 40 60 80 100 120 140

Training Step

Benchmark: OlympiadBench

0.625

0.600

0.575

ValidationScore

0.550

0.525

0.500

0.475

GRPO

MEL

0.450

0 20 40 60 80 100 120 140

Training Step

Benchmark: AMC23

0.80

0.75

ValidationScore

0.70

0.65

0.60

GRPO

0.55

MEL

0 20 40 60 80 100 120 140

Training Step

Benchmark: Average

0.600

0.575

0.550

ValidationScore

0.525

0.500

0.475

0.450

GRPO

0.425

MEL

0 20 40 60 80 100 120 140

Training Step

Figure 8. Performance evolution of GRPO and MEL on Qwen3-14B-Base across training steps on multiple benchmarks.

#### B. Retention Ratio of Meta-Experience

Through empirical validation via replay, MEL is able to collect high-quality meta-experiences. To examine the utilization of collected meta-experiences, Figure 9 reports the retention ratio of meta-experiences after empirical validation throughout training. We observe that the retention ratio consistently increases with model scale, indicating that larger models are more effective at abstracting high-quality knowledge into meta-experiences, thereby achieving higher retention.

| |Qwen3-4B-Base Qwen3-8B-Base Qwen3-14B-Base<br><br>| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.8

0.6

RetentionRatio

0.4

0.2

0.0

20 40 60 80 100 120

Training Steps

Figure 9. Dynamics of the retention ratio of MEL across different model scales.

#### C. Prompt Template

We use the same prompt template for all models. Details of the prompts used for meta-experience construction and for empirical validation via replay are shown below.

##### Meta-Experience Prompt

You are a Meta-Cognitive Reasoning Analyst specializing in self-reflection, error root-cause analysis, and the extraction of generalizable heuristics. You are provided with multiple solution trajectories for the same problem. Note that the labels Correct or Incorrect apply to the final answer, but the reasoning process itself may contain twists and turns. Your task is to conduct a deep comparative autopsy of the thinking processes. You must identify the structural differences in cognition that led to success or failure, and synthesize these into abstract principles for future use. Core Analysis Requirements:

##### 1. Deep Dive into Correct Trajectories (Resilience & Robustness Analysis):

- • Scenario A (Self-Correction): If you find the reasoning contains initial errors or uncertainties, look for moments of self-correction. What triggered the realization? What structural insight allowed the reasoning to pivot back to the right track?
- • Scenario B (Flawless Execution): When every step of the reasoning is correct from the start, identify the Foundational “Immunity”. What specific definition, clear knowledge representation, or disciplined step-by-step verification prevented this Agent from falling into the traps that the Incorrect Agent fell into?
- • Goal: Extract the specific logic validation technique or robust mental representation that saved the solution.

##### 2. Deep Dive into Incorrect Trajectories (Vulnerability Analysis):

- • You must identify not only where the math/logic went wrong, but also why the reasoning drifted.
- • Identify: The “Bifurcation Point” where a correct start turned into a hallucination or logic gap.
- • Analyze: The latent cognitive defect (e.g., concept conflation, rigid mindset, overlooking edge cases, intuitive bias) that caused the error.
- • Identify: What specific knowledge point or constraint was violated?

##### 3. Comparative Synthesis:

- • Contrast the Solutions and Decision Boundaries. Why did the successful trajectory avoid the trap that the failed one fell into?
- • What structural insight did the winner have that the loser missed? (e.g., The winner treated the problem as a geometric issue, while the loser got stuck in algebra).

##### 4. Strict Generalization Constraint:

- • Forbidden: Do NOT mention the specific numbers, variables, or exact answer of the current problem in your “Heuristics” or “Reflective Summary”.
- • Required: Convert specific lessons into abstract heuristics (e.g., instead of “The integral of x2 is...”, use “When integrating polynomial functions...”). Formulate them as conditionally triggered rules (“If...Then...”, “When dealing with [Concept X]...I should...”).

Output Format (Strict Adherence Required)

- 1. Failure Resolution Path & Error Pattern Recognition (Mandatory for incorrect samples)

- • Failure Point: Identify the exact step where logic diverged. Did it start correctly? Where did the drift happen?
- • Latent Cognitive Pattern: Reveal the deep-seated reasoning defect. Was it a bias? A missing prerequisite? A misunderstanding of the prompt’s intent? Do not list surface-level calculation errors.

- 2. Analysis of Success Factors (Mandatory for correct samples)

- • Reasoning Pivot (If applicable): If the path involved self-correction, describe the moment of realization and the strategy used to fix it.
- • Robustness Factor (If flawless): If the path was perfect, explain the fundamental concept or structural approach that effectively “immunized” the reasoning against common errors.
- • Reason for Effectiveness: Why did this perspective work? What fundamental logic did it satisfy?

- 3. First-Person Reflective Summary (Mandatory) Write a meta-cognitive reflection from the first-person perspective (“I”).

- • Review: Briefly review the thinking process differences.
- • Insight: Discuss the specific knowledge point or cognitive habit that was critical.
- • Action: Explain how you will restructure your approach to avoid the identified “Internal Reasoning Defects” in the future.

Focus on the “How” of thinking, not the “What” of the answer.

- 4. Subject Heuristics (Internalized Experience) (Mandatory)

- • [Pattern Name]: If [abstract condition] occurs, then [abstract action]...
- • [Pattern Name]: When dealing with [concept type], I must strictly verify [constraint]...

(Note: These must be applicable to *future* problems of a similar class, completely stripped of this problem’s specifics.) Here the question and the corresponding solutions. <question> {question} </question>

- Solution 1: <answer> {error ans} </answer> <judge> Incorrect </judge>

- Solution 2: <answer> {correct ans} </answer> <judge> Correct </judge>

##### Empirical Validation Prompt

Prior study has provided some internal reference information relevant to this question, including the key approaches, steps, and reasoning needed for a correct solution; the typical reasoning biases, logical flaws, or pitfalls that appear in incorrect solutions; and various heuristic insights on how to complete this problem more effectively.

{experience}

Now, please fully internalize this information as your own experience, then independently think through the problem in detail and produce a complete answer. Note:

• You must perform full, in-depth reasoning internally and arrive at the final answer while making full use of the information above.

Answer the following question: {question}

