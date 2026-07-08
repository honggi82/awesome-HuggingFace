## arXiv:2406.16768v1[cs.LG]24Jun2024

[Figure 1]

# WARP: On the Benefits of Weight Averaged Rewarded Policies

###### Alexandre Ramé, Johan Ferret, Nino Vieillard, Robert Dadashi, Léonard Hussenot, Pierre-Louis Cedoz, Pier Giuseppe Sessa, Sertan Girgin, Arthur Douillard, Olivier Bachem

Google DeepMind

Reinforcement learning from human feedback (RLHF) aligns large language models (LLMs) by encouraging their generations to have high rewards, using a reward model trained on human preferences. To prevent the forgetting of pre-trained knowledge, RLHF usually incorporates a KL regularization; this forces the policy to remain close to its supervised fine-tuned initialization, though it hinders the reward optimization. To tackle the trade-off between KL and reward, in this paper we introduce a novel alignment strategy named Weight Averaged Rewarded Policies (WARP). WARP merges policies in the weight space at three distinct stages. First, it uses the exponential moving average of the policy as a dynamic anchor in the KL regularization. Second, it applies spherical interpolation to merge independently fine-tuned policies into a new enhanced one. Third, it linearly interpolates between this merged model and the initialization, to recover features from pre-training. This procedure is then applied iteratively, with each iteration’s final model used as an advanced initialization for the next, progressively refining the KL-reward Pareto front, achieving superior rewards at fixed KL. Experiments with Gemma policies validate that WARP improves their quality and alignment, outperforming other open-source LLMs.

Keywords: Alignment, RLHF, LLM, Model Merging

#### 1. Introduction

LLM alignment. Conversational agents like Gemini [36, 110] and GPT-4 [93], along with their open-weight counterparts like Gemma [129], have demonstrated remarkable abilities in complex tasks including mathematics, coding, and tool use [13]. These capabilities largely emerge from pretraining on next-token prediction [101, 102], subsequently refined through supervised fine-tuning (SFT) [105, 135]. As these LLMs become more powerful, aligning them with human values becomes increasingly crucial to ensure safe deployment [5, 46]. To this end, reinforcement learning from human feedback (RLHF) has become the prominent strategy [20, 122, 145], first learning a reward model (RM) on human preferences, before optimizing the LLM to maximize predicted rewards.

Challenges in RLHF. However, RLHF introduces several unresolved challenges [16]. First, the limited scope of fine-tuning, often restricted to relatively small datasets, can lead to excessive specialization and catastrophic forgetting [31] of the broad and diverse knowledge acquired during pre-training [38, 66, 67, 79]. Such alignment tax [97] can degrade the LLM’s reasoning capabilities and performance on NLP benchmarks [25, 81]. Second, maximizing an imperfect RM presents several issues on its own, as the LLM can learn to exploit loopholes in the RM [21, 98] when it deviates significantly from its initialization [33]. Such reward hacking [7, 120] can produce outputs that are linguistically flawed [77], excessively verbose [119], or sycophantic [99, 116], thereby raising misalignment [90, 128] and safety [5, 46] concerns. Finally, RLHF can reduce the diversity of generations [65], potentially leading to policy collapse [42, 86]. Such loss of diversity limits use in creative or exploratory tasks and can result in the LLM systematically refusing to answer. Overall, achieving high rewards based on an imperfect RM on a selected distribution of prompts is insufficient due to potential reward misspecification and distribution shifts upon deployment.

Corresponding author: alexandrerame@google.com

[Figure 2]

1. RLHF with EMA anchor in KL

2. Spherical averaging of rewarded policies

[Figure 3]

[Figure 4]

[Figure 5]

3. Linear interpolation towards init

SFT init

New init

[Figure 6]

… and iterate from the new init ♻

(a) WARP with three model merging stages, applicable iteratively.

= 1.0

= 0.8

0.3

| |
|---|

| |
|---|

= 0.5

| |
|---|

0.4

= 0.3

| |
|---|

Reward

0.5

WARP: 5th iteration WARP: 4th iteration WARP: 3rd iteration WARP: 2nd iteration WARP: 1st iteration

- = 0.0

- = 0.1

0.6

| |
|---|

REINFORCE: EMA anchor

0.7

REINFORCE: SFT anchor

0 20 40 60 80 100 120 140 160

KL

(b) KL-reward Pareto front.

Figure 1 | Figure 1(a) illustrates the RLHF alignment process with WARP from a supervised fine-tuned (SFT) LLM. WARP uses model merging by weight averaging at three different stages. First, the exponential moving average (EMA) [55] of the policy serves as the anchor for KL regularization [59]. Second, the independently fine-tuned policies are merged by spherical linear interpolation (SLERP) [118] of their task vectors [53]. Third, we interpolate towards the initialization (LITI) [138], revealing a Pareto front of solutions as we slide the interpolating coefficient 𝜂 from 1 to 0. This results in the “WARP: 1st iteration” curve from Figure 1(b) which improves over the REINFORCE [136] fine-tuning trajectories. Critically, iteratively using a point from this Pareto front as an advanced initialization for the next episode WARP improves performance. Details in Figure 4(c).

RL with KL regularization. To address these issues, previous works constrained the reward optimization by integrating a Kullback-Leibler (KL) regularization [35, 59], using the SFT initialization

- as the anchor. As clarified in Section 2, this KL regularization forces the policy to remain close to its initialization [74, 84], mitigating forgetting and reward hacking [33]. However, employing the SFT model as the anchor may lead to reward underfitting: indeed, there is a fundamental tension between reducing KL and maximizing reward. Thus, different policies should be compared in terms of KL-reward Pareto optimality as in Figure 1(b), where the 𝑥-axis is the KL and the 𝑦-axis is the reward as estimated by the RM, with the optimal policies located in the top-left of the plot.

On model merging by weight averaging. To improve the trade-off between KL and reward during RLHF, we leverage the ability to merge LLMs by weight averaging (WA) [131]. WA relies on the linear mode connectivity [30, 89], an empirical observation revealing linear paths of high performance between models fine-tuned from a shared pre-trained initialization. Model merging was shown to improve robustness under distribution shifts [55, 106, 137] by promoting generalization and reducing memorization [108], to combine models’ abilities [52, 53, 109], to reduce forgetting in continual learning [123], to enable collaborative [104] and distributed [27] learning at scale, without computational overheads at inference time. Model merging is increasingly adopted within the opensource community [37, 72], leading to state-of-the-art models in specialized domains [70] but also significant advancements on general-purpose benchmarks [68, 69]. In particular, while WA was initially mostly used for discriminative tasks [137] such as reward modeling [108], it is now becoming popular for generative tasks [4, 111]; its use in KL-constrained RLHF has already shown preliminary successes in a few recent works [39, 81, 83, 88, 92, 109], further elaborated in Section 5.

WARP. In this paper, we propose Weight Averaged Rewarded Policies (WARP), a simple strategy for aligning LLMs, illustrated in Figure 1(a) and detailed in Section 3. WARP is designed to optimize the KL-reward Pareto front of solutions, as demonstrated in Figure 1(b). WARP uses three variants of WA

- at three different stages of the alignment procedure, for three distinct reasons.

- Stage 1: Exponential Moving Average (EMA). During RL fine-tuning, instead of regularizing the policy towards the SFT initialization, WARP uses the policy’s own exponential moving average [100] as a dynamic updatable anchor in the KL. This stage enables stable exploration with distillation from a mean teacher [127] and annealed constraint.

- Stage 2: Spherical Linear intERPolation of task vectors (SLERP). Considering 𝑀 policies RL fine-tuned independently with their own EMA anchor, we merge them by spherical linear interpolation [118] of their task vectors [53]. This stage creates a merged model with higher reward by combining the strengths of the 𝑀 individual policies.

- Stage 3: Linear Interpolation Towards Initialization (LITI). Considering the merged policy from SLERP, WARP linearly interpolates towards the initialization, akin to WiSE-FT [138]. This stage allows to run through an improved Pareto-front simply by adjusting the interpolating coefficient 𝜂 between 1 (high reward but high KL) and 0 (small KL but small reward). Critically, selecting an intermediate value for 0 < 𝜂 < 1 offers a balanced model that can serve as a new, improved initialization for subsequent iterations of WARP.

Experiments and discussion. In Section 4, we validate the efficacy of WARP for the fine-tuning of Gemma "7B" [129]. Finally, in Section 6, we discuss the connections between WARP, the distributed learning literature [27, 104] and iterated amplification [19], illustrating how WARP embodies their principles to enable scaling post-training, for continuous alignment and improvement of LLMs.

#### 2. Context and notations

RL for LLMs. We consider a transformer [132] LLM 𝑓 (·, 𝜃) parameterized by 𝜃. Following the foundation model paradigm [12] and the principles of transfer learning [94], those weights are trained via a three-stage procedure: pre-training through next token prediction, supervised finetuning resulting in 𝜃sft, and ultimately, RLHF [20, 97] to optimize a reward 𝑟 as determined by a RM trained to reflect human preferences. In this RL stage, 𝜃 defines a policy 𝜋𝜃(· | 𝒙) by auto-regressively generating token sequences 𝒚 from the prompt 𝒙. The primary objective is to find weights maximizing the average reward over a dataset of prompts X: argmax𝜃 𝔼𝒙∈X𝔼𝒚∼𝜋

𝜃(·|𝒙) 𝑟(𝒙, 𝒚) .

KL vs. reward. Optimizing solely for 𝑟 can (i) forget general abilities from pre-training [31] as an alignment tax [81, 97], (ii) hack the reward [7, 120] leading to potential misalignment, or (iii) reduce the diversity of possible generations [65] (confirmed in Appendix F). To mitigate these risks, a KL regularization is usually integrated to balance fidelity to the initialization and high rewards:

argmax

𝜃(·|𝒙)𝑟(𝒙, 𝒚) − 𝛽KL 𝜋𝜃(· | 𝒙)∥𝜋𝜃anchor(· | 𝒙) , (1)

𝔼𝒙∈X 𝔼𝒚∼𝜋

𝜃

where 𝜃anchor ← 𝜃sft and 𝛽 is the regularization strength, with high values leading to low KL though also lower reward. The reward function adjusted with this KL is 𝑟(𝒙, 𝒚) − 𝛽 log 𝜋 𝜋𝜃(𝒚|𝒙)

𝜃anchor(𝒚|𝒙) . Our base RL algorithm is a variant of REINFORCE [136]. This choices follows recent RLHF works [75, 108, 112] and the findings from [2, 80, 126] that, in terms of KL-reward Pareto optimality, REINFORCE performs better than the more complex PPO [114] and also better than various offline algorithms such as DPO [103], IPO [11] or RAFT [26]. Practitioners then typically employ early stopping to select an optimal point on the training trajectory based on their specific use cases.

#### 3. WARP

We introduce a novel alignment strategy named Weight Averaged Rewarded Policies (WARP), illustrated in Figure 1(a) and described in Algorithm 1 below. WARP merges LLMs in the weight space to enhance the KL-reward Pareto front of policies. The following Sections 3.1 to 3.3 describe the motivations behind applying three distinct variants of WA at the three different stages of WARP. In particular, we summarize the key insights as observations, that will be experimentally validated in Section 4 (and in Appendices C and D), and theoretically motivated in Appendix B when possible. Overall, WARP outperforms other RL alignment strategies, without any memory or inference overhead at test time. However, training WARP is costly, requiring multiple RL runs at each iteration: see Section 6 for a detailed discussion on the required compute scaling.

Algorithm 1 WARP for KL-reward Pareto optimal alignment Input: Weights 𝜃sft pre-trained and supervised fine-tuned

Reward model 𝑟, prompt dataset X, optimizer Opt 𝐼 iterations with 𝑀 RL runs each for 𝑇 training steps 𝜇 EMA update rate, 𝜂 LITI update rate

- 1: Define 𝜃init ← 𝜃sft
- 2: for iteration 𝑖 from 1 to 𝐼 do
- 3: for run 𝑚 from 1 to 𝑀 do ⊲ Run in parallel
- 4: Define 𝜃𝑚, 𝜃ema𝑚 ← 𝜃init
- 5: for step 𝑡 from 1 to 𝑇 do
- 6: Generate completion 𝒚 ∼ 𝜋𝜃𝑚(· | 𝒙) for 𝒙 ∈ X
- 7: Compute 𝑟𝛽(𝒚) ← 𝑟(𝒙, 𝒚) − 𝛽 log 𝜋𝜋𝜃𝑚(𝒚|𝒙)

𝜃𝑚

ema(𝒚|𝒙) ⊲ KL regularized reward

- 8: Update 𝜃𝑚 ← Opt 𝜃𝑚, 𝑟𝛽(𝒚)∇𝜃[log𝜋𝜃𝑚(𝒚 | 𝒙)] ⊲ Policy gradient
- 9: Update 𝜃ema𝑚 ← (1 − 𝜇) · 𝜃ema𝑚 + 𝜇 · 𝜃𝑚 ⊲ Equation (EMA): update anchor
- 10: end for
- 11: end for
- 12: Define 𝜃slerp𝑖 ← slerp 𝜃init, {𝜃𝑚}𝑚𝑀=1, 𝜆 = 𝑀1 ⊲ Equation (SLERP): merge 𝑀 weights

- 13: Update 𝜃init ← (1 − 𝜂) · 𝜃init + 𝜂 · 𝜃slerp𝑖 ⊲ Equation (LITI): interpolate towards init
- 14: end for Output: KL-reward Pareto front of weights {(1 − 𝜂) · 𝜃sft + 𝜂 · 𝜃slerp𝐼 | 0 ≤ 𝜂 ≤ 1}

###### 3.1. Stage 1: exponential moving average as a dynamic anchor in KL regularization

EMA anchor. KL-regularized methods typically use the SFT initialization as a static anchor [59, 112], but in RL for control tasks, it is common to regularly update the anchor [1, 113]. In this spirit, WARP uses the policy’s own exponential moving average (EMA) [100], updated throughout the RL fine-tuning process such as, at each training step with 𝜇 = 0.01:

𝜃ema ← (1 − 𝜇) · 𝜃ema + 𝜇 · 𝜃policy. (EMA) Using 𝜃ema as the anchor 𝜃anchor in Equation (1) provides several benefits, outlined below.

- Observation 1 (EMA). Policies trained with an exponential moving average anchor benefit from automatic annealing of the KL regularization and from distillation from a dynamic mean teacher [127]. Empirical evidence in Section 4.1.

Benefits from EMA. Unlike a static SFT anchor, the dynamic nature of an EMA anchor induces a gradual automatic annealing and relaxation of the KL regularization. Specifically, the policy is initially strongly tied to the SFT initialization, and then progressively unleashed, allowing for more aggressive gradient updates later in training, leading to higher rewards. Moreover, by progressively incorporating knowledge from the training, EMA acts as slow weight [76, 123], and thus performing better than the initialization. But, by also maintaining essential information from the initialization, EMA can even perform better than the final policy’s weights; studies [6, 55, 125] (see [87] for a review), and specifically [62] within the context of LLMs, indicate that averaging checkpoints over steps improves internal representations and thus predictions. Then, EMA guides the policy by KL distillation [49] of high-quality target predictions, akin to a mean teacher [127] for self-supervised [15, 40, 44, 95, 121] learning. This also relates to deep RL techniques where EMA stabilizes exploration toward a Nash equilibrium [9, 10, 39, 88], and approximates mirror descent [14, 35, 130].

###### 3.2. Stage 2: spherical linear interpolation of independently rewarded policies

SLERP. While EMA helps for a single RL and a fixed compute budget, it faces limitations due to the similarity of the weights collected along a single fine-tuning [106]. In this second stage, we merge 𝑀 weights RL fine-tuned independently (each with their own EMA anchor). This follows model soups from Wortsman et al. [137] and its variants [106, 107] showing that WA improves generalization, and that task vectors [53] (the difference between fine-tuned weights and their initialization) can be arithmetically manipulated by linear interpolation (LERP) [131]. Yet, this time, we use spherical linear interpolation (SLERP) [118], illustrated in Figure 2 and defined below for 𝑀 = 2:

sin[(1 − 𝜆)Ω] sin Ω · 𝛿1 +

sin[𝜆Ω] sin Ω · 𝛿2, (SLERP)

slerp 𝜃init, 𝜃1, 𝜃2, 𝜆 = 𝜃init +

where Ω is the angle between the two task vectors 𝛿1 = 𝜃1 − 𝜃init and 𝛿2 = 𝜃2 − 𝜃init, and 𝜆 the interpolation coefficient. Critically SLERP is applied layer by layer, each having a different angle. In Appendix B.3 we clarify how SLERP can be used iteratively to merge 𝑀 > 2 models. To enforce diversity across weights, we simply vary the order in which text prompts 𝒙 are given in each run: this was empirically sufficient, though other diversity strategies could help, e.g., varying the hyperparameters or the reward objectives (as explored in Figure 18(c)).

Benefits from SLERP vs. LERP. Merging task vectors, either with SLERP or LERP, combines their abililities [53]. The difference is that SLERP preserves their norms, reaching higher rewards than the base models; this is summarized in Observation 2. In contrast, and as summarized in Observation 3, the more standard LERP has less impact on reward, but has the advantage of reducing KL; indeed, as shown in Appendix B, LERP tends to pull the merged model towards the initialization, especially as the angle Ω between task vectors is near-orthogonal (see Observation 3).

𝜃1

𝜃slerp𝜆

𝛿1 𝜃lerp𝜆

Ω ≈ 90◦

𝛿2

𝜃2

𝜃init

Figure 2 | SLERP vs. LERP.

- Observation 2 (SLERP). Spherical linear interpolation boosts rewards, yet slightly increases KL. Empirical evidence in Section 4.2 and theoretical insights in Lemma 1.
- Observation 3 (LERP). Linear interpolation reduces KL, yet has reduced impact on reward. Empirical evidence in Appendix C.1 and theoretical insights in Lemmas 2 and 3.
- Observation 4 (Task vectors). Task vectors 𝛿 are close to orthogonal with Ω ≈ 90◦, while the full weights 𝜃 are collinear. Empirical evidence in Appendix C.2.

- 3.3. Stage 3: linear interpolation towards initialization

LITI. In the previous stage, SLERP combines multiple policies into one with higher rewards and slightly higher KL. This third stage, inspired by WiSE-FT from Wortsman et al. [138], interpolates from the merged model towards the initialization:

𝜃𝜂 ← (1 − 𝜂) · 𝜃init + 𝜂 · 𝜃slerp. (LITI)

Adjusting the interpolating coefficient 𝜂 ∈ [0, 1] trades off between some newly acquired behaviors leading to high rewards vs. general knowledge from the SFT initialization. Specifically, large values 𝜂 ≈ 1 provide high rewards but also high KL, while smaller values 𝜂 ≈ 0 lean towards smaller rewards and minimal KL. Fortunately, we observe that the reduction in KL is proportionally greater than the reduction in reward when decreasing 𝜂. Then, LITI empirically yields Pareto fronts that are noticeably above the “diagonal”, but also above those revealed during the base RLs.

- Observation 5 (LITI). Interpolating weights towards the initialization reveals a better Pareto front than the one revealed during RL fine-tuning. Empirical evidence in Figure 1(b) and Section 4.3, and theoretical insights in Lemmas 4 and 5.

Benefits from LITI. Previous works tried to understand how weight interpolation can mitigate forgetting while increasing robustness and generalization. [138] argues that WiSE-FT, akin to LITI in supervised learning contexts, recovers generalizable features from pre-training that might be lost during fine-tuning [67], consistently with WA reducing catastrophic forgetting [28, 123] in continual learning. Then in the context of RL, [81] argues that LITI increases feature diversity, efficiently balancing between generality and task specificity. Finally, [58] argues that the geometric projection of the ideal weights is located between the merged model and the initialization.

3.4. Iterative WARP

Iterative training. The model merging strategies previously described not only establish an improved Pareto front of solutions, but also set the stage for iterative improvements. Indeed, if the computational budget is sufficient, we can apply those three stages iteratively, using 𝜃𝜂 from previous Pareto front (usually with 𝜂 = 0.3, choice ablated in Appendix D.3) as the initialization 𝜃init for the next iteration, following the model recycling [24, 107] strategies. Then, the entire training procedure is made of multiple iterations, each consisting of those three stages, where the final weight from a given iteration serves as an improved initialization for the next one.

- Observation 6 (Iterative WARP). Applying WARP iteratively improves results, converging to an optimal Pareto front. Empirical evidence in Sections 4.4 and 4.5.

#### 4. Experiments: on the benefits of WARP

Setup. We consider the Gemma "7B" [129] LLM, which we seek to fine-tune with RLHF into a better conversational agent. We use REINFORCE [136] policy gradient to optimize the KL-regularized reward. The dataset X contains conversation prompts. We generate on-policy samples with temperature 0.9, batch size of 128, Adam [64] optimizer with learning rate 10−6 and warmup of 100 steps. SLERP is applied independently to the 28 layers. Except when stated otherwise, we train for 𝑇 = 9𝑘 steps, with KL strength 𝛽 = 0.1, EMA update rate 𝜇 = 0.01, merging 𝑀 = 2 policies uniformly 𝜆 = 0.5, and LITI update rate 𝜂 = 0.3; we analyze those values in Appendix D. We rely on a high capacity reward model, the largest available, which prevents the use of an oracle control RM as done in [33, 108].

Summary. In our experiments, we analyze the KL to the SFT policy (reflecting the forgetting of pre-trained knowledge) and the reward (evaluating alignment to the RM). In Section 4.1, we first show the benefits of using an EMA anchor; then in Section 4.2, we show that merging policies trained independently helps. Moreover, in Section 4.3, we show that LITI improves the KL-reward Pareto front; critically, repeating those three WARP stages can iteratively improve performances in Section 4.4. A limitation is that our RM accurately approximates true human preferences only in low KL region, though can be hacked away from the SFT [33]. Therefore, we finally report other metrics in Section 4.5, specifically comparing against open-source baselines such as Mixtral [61], and reporting performances on standard benchmarks such as MMLU [47].

###### 4.1. Stage 1: exponential moving average as a dynamic anchor in KL regularization

In Figures 3(a) and 3(b), we compare the training trajectories of different REINFORCE variants, where the changes lie in the choice of the anchor in the KL regularization and of the hyperparameter 𝛽 controlling its strength. Results are computed every 100 training steps. In our proposed version, the anchor is the EMA of the trained policy with 𝛽 = 0.1 and an EMA update rate 𝜇 = 0.1 (other values are ablated in Figure 15). As the Pareto front for our strategy is above and to the left in Figure 3(b), this confirms the superiority of using such an adaptive anchor. The baseline variants all use the SFT as the anchor, with different values of 𝛽. The lack of regularization (𝛽 = 0.0) leads to very fast optimization of the reward in Figure 3(a), but largely through hacking, as visible by the KL exploding in just a few training steps in Figure 3(b). In contrast, higher values such as 𝛽 = 0.1 fail to optimize the reward as regularization is too strong, causing a quick reward saturation around −0.62 in Figure 3(a). Higher values such as 𝛽 = 0.01 can match our EMA anchor in low KL regime, but saturates around a reward of −0.46. In contrast, as argued in Observation 1, the dynamic EMA anchor progressively moves away from the SFT initialization, causing implicit annealing of the regularization. In conclusion, relaxing the anchor with EMA updates allows the efficient learning of KL-reward Pareto-optimal policies, at any given KL level, for a fixed compute budget. We refer the interested reader to additional experiments in Figure 14 from Appendix D.2 where we compare the trained policies with their online EMA version.

0.40

0.45

0.50

| |
|---|

Reward

0.55

0.60

EMA anchor: = 0.1

| |
|---|

- SFT anchor: = 0.0

- SFT anchor: = 0.0001

0.65

SFT anchor: = 0.001

0.70

SFT anchor: = 0.01

SFT anchor: = 0.1

0.75

| |
|---|

0 2000 4000 6000 8000

# steps

(a) Reward vs. steps.

0.40

0.45

0.50

| |
|---|

| |
|---|

Reward

| |
|---|

0.55

0.60

EMA anchor: = 0.1

| |
|---|

- SFT anchor: = 0.0

- SFT anchor: = 0.0001

0.65

SFT anchor: = 0.001

0.70

SFT anchor: = 0.01

SFT anchor: = 0.1

0.75

| |
|---|

0 20 40 60 80 100 120

KL

(b) Reward vs. KL.

0.36

0.38

Reward

0.40

SLERP at T = 9k

0.42

LERP at T = 9k

SLERP at T = 7k

LERP at T = 7k

0.44

SLERP at T = 5k

LERP at T = 5k

SLERP at T = 3k

0.46

LERP at T = 3k

0.0 0.2 0.4 0.6 0.8 1.0

(c) SLERP vs. LERP.

- Figure 3 | EMA and SLERP experiments. We first compare RL runs with different anchors and strengths 𝛽 in the KL regularization. We show their results along training in Figure 3(a), and their KL-reward Pareto fronts in Figure 3(b). We perform evaluation every 100 steps, and train them for 𝑇 = 9𝑘 steps, though we stopped the trainings if they ever reach a KL of 200 (e.g., after 𝑇 = 1𝑘 training steps when 𝛽 = 0.0). Figure 3(c) plots the reward obtained when merging two policies (trained independently after 𝑇 RL steps with their own EMA anchor) with interpolating coefficient 𝜆; highest rewards are with SLERP for 𝜆 = 0.5 and 𝑇 = 9𝑘 steps.

###### 4.2. Stage 2: spherical linear interpolation of independently rewarded policies

- In Figure 3(c), we plot 𝜆 → 𝑟 slerp 𝜃init, 𝜃1, 𝜃2, 𝜆 showing reward convexity when interpolating policies via SLERP, validating Observation 2. This mirrors the linear mode connectivity [30] property across weights fine-tuned from a shared initialization, i.e., the fact that interpolated weights perform better than the initial models (recovered for 𝜆 = 0 or 𝜆 = 1). Moreover, SLERP consistently obtains higher rewards than LERP; yet, this is at slightly higher KL, as further detailed in Appendices B and C.1, where we analyze respectively their theoretical and empirical differences.

4.3. Stage 3: linear interpolation towards initialization

- In Figure 4(a), we merge policies trained for 𝑇 steps, and then apply the LITI procedure. Critically, sliding the interpolating coefficient 𝜂 ∈ {0, 0.1, 0.3, 0.5, 0.8, 1.0} reveals various Pareto fronts, consistently above the training trajectories obtained during the two independent RL fine-tunings. Interestingly, longer fine-tunings improve performances, at high KL, but also at lower KL, simply by using a smaller 𝜂 afterwards. Then in Figure 4(b), we report the Pareto fronts when merging up to 𝑀 = 5 weights. We note that all Pareto fronts revealed when applying LITI are consistently above the ones from RL fine-tunings, validating Observation 5. More precisely, best results are achieved by merging an higher number of policies 𝑀, suggesting a promising scaling direction.

###### 4.4. Iterative WARP

In Figure 4(c), we apply the iterative procedure described in Section 3.4. At each of the 𝐼 = 5 iterations we train 𝑀 = 2 policies for 𝑇 steps, with 𝑇 = 9𝑘 for the first iteration, and 𝑇 = 7𝑘 for iterations 2 and 3, and then 𝑇 = 5𝑘 for computational reasons. The LITI curves interpolate towards their own initialization (while Figure 1(b) interpolated towards the SFT initialization, see Appendix D.4 for a comparison). We systematically observe that LITI curves are above the RL training trajectories used to obtain the inits. Results get better at every iteration, validating Observation 6, although with reduced returns after a few iterations.

= 1.0

= 0.8

= 0.5

0.4

| |
|---|

| |
|---|

= 0.3

| |
|---|

0.5

| |
|---|

Reward

- = 0.0

- = 0.1

0.6

T = 9k T = 7k T = 5k T = 3k

| |
|---|

0.7

REINFORCE v1 REINFORCE v2

0 20 40 60 80 100 120 140

KL

(a) LITI of SLERP after 𝑇 steps.

= 0.8 = 1.0

= 0.5

0.35

| |
|---|

= 0.3

0.40

| |
|---|

0.45

Reward

0.50

0.55

- = 0.0

- = 0.1

M = 5 M = 4 M = 3 M = 2 M = 1 REINFORCE

| |
|---|

0.60

0.65

0.70

0 20 40 60 80 100 120 140 160

KL

(b) LITI of SLERP of 𝑀 weights.

| |
|---|

0.3

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

= 0.8 = 1.0

| |
|---|

| |
|---|

| |
|---|

= 0.5

| |
|---|

0.4

| |
|---|

| |
|---|

= 0.3

| | |
|---|---|

Reward

5th iter: LITI

5th iter: REINFORCE

0.5

4th iter: LITI

4th iter: REINFORCE

3rd iter: LITI

- = 0.0

- = 0.1

0.6

3rd iter: REINFORCE

| |
|---|

2nd iter: LITI

2nd iter: REINFORCE

1st iter: LITI

0.7

| |
|---|

1st iter: REINFORCE

0 20 40 60 80 100 120 140

KL

(c) Iterative WARP.

- Figure 4 | LITI and iterative experiments. Figure 4(a) considers the LITI of the SLERP of 𝑀 = 2 policies after 𝑇 steps with 𝜆 = 0.5, interpolating towards their SFT init as we slide 𝜂, revealing Pareto fronts above the 𝑀 = 2

REINFORCE training trajectories. Then Figure 4(b) plots the LITI of the SLERP of 𝑀 weights with 𝜆 = 𝑀1 after 𝑇 = 9𝑘 steps: light-colored areas show standard deviations across 5 experiments. The iterative WARP procedure

is illustrated in Figure 4(c); we fine-tune 𝑀 = 2 policies with their own EMA as the anchor, merge them with SLERP, interpolate towards their init with LITI, and iteratively leverage the weights obtained with 𝜂 = 0.3 as the new initialization for the next iteration.

###### Table 1 | Side by side comparisons.

Methods Mistral 7B v1 Mistral 7B v2 Mixtral 8x7B

- Gemma "7B" 1.0 0.24 -0.01 -0.08
- Gemma "7B" 1.1 0.37 0.16 0.08 REINFORCE EMA anchor 0.37 0.16 0.07 WARP: 1st iter 0.42 0.23 0.13 WARP: 2nd iter 0.45 0.25 0.16 WARP: 3rd iter 0.45 0.26 0.18

- WARP: 4th iter 0.45 0.25 0.16
- WARP: 5th iter 0.45 0.24 0.17

4.5. Comparisons and benchmarks

Side by side comparisons. To conclude our experiments, we compare our trained policies against Mistral [60] and Mixtral [61] LLMs. Each policy generates a candidate answer on an held-out collection of prompts, as in the Gemma tech report [129]. Then similarly to Gemini 1.5 [110], we compute side by side preference rates [144] with “much better”, “better” and “slightly better” receiving scores of ±1.5, ±1, and ±0.5 respectively (and ties receiving a score of 0). A positive score represents better policies. The results in Table 1 validate the efficiency of WARP, as our policies are preferred over Mistral variants, and also outperform the two previous Gemma "7B" releases. However, we note that the results stagnate after the 3rd iteration.

Benchmarks. Table 2 compares WARP (3rd iter) and the latest Gemma "7B" 1.1 release [129] on popular benchmarks in the zero-shot setup: MBPP [8] and HumanEval [18] benchmarking coding capabilities, MMLU [47] assessing STEM knowledge, the GSM8K [22] and MATH [48] benchmarks targeting reasoning abilities, and the Big Bench Hard (BBH) [124] benchmark evaluating general capabilities through questions that were deemed difficult for frontier LLMs. WARP has particularly strong results on mathematics benchmarks, suggesting higher analytical capabilities.

###### Table 2 | Benchmark results.

Methods MBPP MMLU GSM8K MATH HumanEval BBH Gemma "7B" 1.1 39.0 56.4 55.6 25.6 46.9 53.1 WARP 45.4 57.6 66.8 31.0 50.0 58.8

#### 5. Related work

How to merge models. The question of how best to merge models has recently garnered significant attention, driven by the discoveries that deep models can be merged in the weight space [131] instead of in the prediction space, as traditionally done in ensembling [43, 71]. For clarity, we collectively refer to these methods as weight averaging (WA). The most common is LERP, initially used to average checkpoints collected along a single run, uniformly [55, 125] or with an exponential moving average (EMA) [100], notably as a mean teacher [127] for self-supervision [15, 40, 44, 95, 121]. Following the linear mode connectivity [30] observation, the model soups variants [53, 107, 137] linearly interpolate from different fine-tunings; this relies on the shared pre-training, limiting divergence [89] such as models remain in constrained weight regions [41], which also suggests that pre-training mitigates the need to explicitly enforce trust regions in gradient updates [113, 114]. Subsequent works such as TIES merging [140] and DARE [141] reduce interferences in multi-task setups with sparse task

vectors [53]. In contrast, we use SLERP, introduced in [118], increasingly popular in the open-source community [37] but relatively underexplored in the academic literature, with limited studies such as [63]. Some tried to align weights trained from scratch [3, 29] or with different architectures [133]; yet, the methods are complex, less robust, and usually require additional training.

Benefits of model merging. WA boosts generalization by reducing variance [106, 137], decreasing memorization [82, 108, 142] and flattening the loss landscape [17]. Additionally, merging weights combines their strengths [53], which helps in multi-task setups [52, 109], to tackle catastrophic forgetting [28, 123] or to provide better initializations [24], as explored in [51, 57, 58] for iterative procedures in classification tasks. In particular, we considered using the geometric insights from Eq. 2 in [58]; yet, as our task vectors are nearly orthogonal Ω ≈ 90◦ (see Appendix C.2), using the update rule 𝜂 → 12cos+cosΩΩ failed. WA is now also used in RL setups [34, 73, 91]; for example, WARM [108] merges reward models to boost their efficiency, robustness and reliability. Actually, WARP is conceived as a response to WARM, demonstrating that model merging can tackle two key RLHF challenges; policy learning in WARP and reward design in WARM. The most similar works are the following, which also explore how WA can improve policy learning. [92] proposes an iterative approach with the EMA as a new initialization for subsequent iterations. [39] and [88] uses EMA as the reference, but only for direct preference optimization. [109] employs LERP to improve alignment in multi-objective RLHF when dealing with different objectives; similarly, [139] targets multi-task setups with LERP. Finally, [81] and [32] use model merging to reduce the alignment tax, although without incorporating EMA during training, without merging multiple rewarded policies and not iteratively. Critically, none of these works focus on KL as a measure of forgetting, use EMA as the anchor in KL, apply SLERP or use LITI as the initialization for subsequent RL iterations. In contrast, WARP integrates all those elements, collectively leading to an LLM outperforming Mixtral [61].

#### 6. Discussion

Distributed learning for parallelization and open-source. WARP addresses a crucial challenge: aligning LLMs with human values and societal norms, while preserving the capabilities that emerged from pre-training. To this end, we leverage a (perhaps surprising) ability: policies trained in parallel can combine their strengths within a single policy by weight averaging. Then, the distributed nature of WARP makes it flexible and scalable, as it is easily parallelizable by enabling intermittent weight sharing across workers. Actually, iterative WARP shares similarities with DiLoCo [27]: by analogy, the first stage performs inner optimization on multiple workers independently; the second stage merges gradients from different workers; the third stage performs SGD outer optimization with a learning rate equal to 𝜂. More generally, WARP could facilitate open-source [37] collaborative training of policies [104], optimizing resource and supporting privacy in federated learning [85] scenarios; collaborators could train and share their LLMs, while keeping their data and RMs private. In particular, we show in Appendix E that WARP can handle diverse objectives, similarly to [109].

Iterated amplification. WARP improves LLM alignment by leveraging the principles of iterated amplification [19] and progressive collaboration of multiple agents. By analogy, model merging via WA acts as an effective alternative to debate [54], with agents communicating within the weight space instead of the token space, ensuring that only essential information is retained [108]. Then, WARP refines the training signal by combining insights and exploration from diverse models, iteratively achieving higher rewards through self-distillation [127], surpassing the capabilities of any single agent. If this is the way forward, then an iterative safety assessment would be required to detect and mitigate potential risks early, ensuring that the development remains aligned with safety standards.

Scaling alignment. The WARP procedure increases the compute training cost by performing multiple fine-tunings at each iteration. Yet, this should be viewed as “a feature rather than a bug”. Specifically, by preventing memorization and forgetting, we see WARP as a fine-tuning method that can transform additional compute allocated to alignment into enhanced capabilities and safety. This would allow scaling (the traditionally cheap) post-training alignment, in the same way pre-training has been scaled [50]. Critically for large-scale deployment, the acquired knowledge is within a single (merged) model, thus without inference or memory overhead, in contrast to “more agents” approaches [78, 134]. Finally, although WARP improves policy optimization, it is important to recognize that WARP does not address other critical challenges in RLHF [16]: to mitigate the safety risks [5, 45, 46] from misalignment [90, 128], WARP should be part of a broader responsible AI framework.

#### 7. Conclusion

We introduce Weight Averaged Rewarded Policies (WARP), a novel RLHF strategy to align LLMs with three distinct stages of model merging: exponential moving average as a dynamic anchor during RL, spherical interpolation to combine multiple policies rewarded independantly, and interpolation towards the shared initialization. This iterative application of WARP improves the KL-reward Pareto front, aligning the LLMs while protecting the knowledge from pre-training, and compares favorably against state-of-the-art baselines. We hope WARP could contribute to safe and powerful AI systems by scaling alignment, and spur further exploration of the magic behind model merging.

#### References

- [1] Abbas Abdolmaleki, Jost Tobias Springenberg, Yuval Tassa, Remi Munos, Nicolas Heess, and Martin Riedmiller. Maximum a posteriori policy optimisation. ICLR, 2018. (p. 4)
- [2] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting REINFORCE style optimization for learning from human feedback in LLMs. arXiv preprint, 2024. (p. 3)
- [3] Samuel K. Ainsworth, Jonathan Hayase, and Siddhartha Srinivasa. Git re-basin: Merging models modulo permutation symmetries. In ICLR, 2022. (p. 10)
- [4] Takuya Akiba, Makoto Shing, Yujin Tang, Qi Sun, and David Ha. Evolutionary optimization of model merging recipes. arXiv preprint, 2024. (p. 2)
- [5] Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete problems in AI safety. arXiv preprint, 2016. (pp. 1 and 11)
- [6] Devansh Arpit, Huan Wang, Yingbo Zhou, and Caiming Xiong. Ensemble of averages: Improving model selection and boosting performance in domain generalization. In NeurIPS, 2021. (pp. 5 and 30)
- [7] Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, and Jared Kaplan. A general language assistant as a laboratory for alignment. arXiv preprint, 2021. (pp. 1 and 3)
- [8] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint, 2021. (p. 9)
- [9] Mostafa D Awheda and Howard M Schwartz. Exponential moving average Q-learning algorithm. In ADPRL, 2013. (p. 5)
- [10] Mostafa D Awheda and Howard M Schwartz. Exponential moving average based multiagent reinforcement learning algorithms. Artificial Intelligence Review, 2016. (p. 5)
- [11] Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, and Rémi Munos. A general theoretical paradigm to understand learning from human preferences. arXiv preprint, 2023. (p. 3)
- [12] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint, 2021. (p. 3)
- [13] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint, 2023. (p. 1)
- [14] Sébastien Bubeck et al. Convex optimization: Algorithms and complexity. Foundations and Trends in ML, 2015. (p. 5)

- [15] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. (pp. 5 and 9)
- [16] Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, et al. Open problems and fundamental limitations of reinforcement learning from human feedback. TMLR, 2023. (pp. 1 and 11)
- [17] Junbum Cha, Sanghyuk Chun, Kyungjae Lee, Han-Cheol Cho, Seunghyun Park, Yunsung Lee, and Sungrae Park. SWAD: Domain generalization by seeking flat minima. In NeurIPS, 2021. (p. 10)
- [18] Mark Chen, Jerry Tworek, Heewoo Jun, et al. Evaluating large language models trained on code. arXiv preprint, 2021. (p. 9)
- [19] Paul Christiano, Buck Shlegeris, and Dario Amodei. Supervising strong learners by amplifying weak experts. arXiv preprint, 2018. (pp. 3 and 10)
- [20] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In NeurIPS, 2017. (pp. 1 and 3)
- [21] Jack Clark and Dario Amodei. Faulty Reward Functions in the Wild. https://openai.com /research/faulty-reward-functions, 2016. (p. 1)
- [22] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. (p. 9)
- [23] Imre Csiszár. I-divergence geometry of probability distributions and minimization problems. The annals of probability, pages 146–158, 1975. (p. 25)
- [24] Shachar Don-Yehiya, Elad Venezian, Colin Raffel, Noam Slonim, Yoav Katz, and Leshem Choshen. ColD fusion: Collaborative descent for distributed multitask finetuning. In ACL,

2023. (pp. 6 and 10)

- [25] Guanting Dong, Hongyi Yuan, Keming Lu, Chengpeng Li, Mingfeng Xue, Dayiheng Liu, Wei Wang, Zheng Yuan, Chang Zhou, and Jingren Zhou. How abilities in large language models are affected by supervised fine-tuning data composition. arXiv preprint, 2023. (p. 1)
- [26] Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, KaShun SHUM, and Tong Zhang. RAFT: Reward ranked finetuning for generative foundation model alignment. TMLR, 2023. (p. 3)
- [27] Arthur Douillard, Qixuan Feng, Andrei A Rusu, Rachita Chhaparia, Yani Donchev, Adhiguna Kuncoro, Marc’Aurelio Ranzato, Arthur Szlam, and Jiajun Shen. Diloco: Distributed lowcommunication training of language models. arXiv preprint, 2023. (pp. 2, 3, and 10)
- [28] Steven Vander Eeckt et al. Weight averaging: A simple yet effective method to overcome catastrophic forgetting in automatic speech recognition. arXiv preprint, 2022. (pp. 6 and 10)
- [29] Rahim Entezari, Hanie Sedghi, Olga Saukh, and Behnam Neyshabur. The role of permutation invariance in linear mode connectivity of neural networks. In ICLR, 2022. (p. 10)

- [30] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, and Michael Carbin. Linear mode connectivity and the lottery ticket hypothesis. In ICML, 2020. (pp. 2, 8, 9, and 26)
- [31] Robert M French. Semi-distributed representations and catastrophic forgetting in connectionist networks. Connection Science, 1992. (pp. 1 and 3)
- [32] Tingchen Fu, Deng Cai, Lemao Liu, Shuming Shi, and Rui Yan. Disperse-then-merge: Pushing the limits of instruction tuning via alignment tax reduction. arXiv preprint, 2024. (p. 10)
- [33] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In ICML, 2023. (pp. 1, 2, 6, and 7)
- [34] Jean-Baptiste Gaya, Laure Soulier, and Ludovic Denoyer. Learning a subspace of policies for online adaptation in reinforcement learning. In ICLR, 2022. (p. 10)
- [35] Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. A theory of regularized markov decision processes. In ICML, 2019. (pp. 2 and 5)
- [36] Google Gemini Team. Gemini: A family of highly capable multimodal models. 2023. (p. 1)
- [37] Charles Goddard, Shamane Siriwardhana, Malikeh Ehghaghi, Luke Meyers, Vlad Karpukhin, Brian Benedict, Mark McQuade, and Jacob Solawetz. Arcee’s mergekit: A toolkit for merging large language models. arXiv preprint, 2024. (pp. 2 and 10)
- [38] Ian J Goodfellow, Mehdi Mirza, Da Xiao, Aaron Courville, and Yoshua Bengio. An empirical investigation of catastrophic forgetting in gradient-based neural networks. arXiv preprint,

2013. (p. 1)

- [39] Alexey Gorbatovski, Boris Shaposhnikov, Alexey Malakhov, Nikita Surnachev, Yaroslav Aksenov, Ian Maksimov, Nikita Balagansky, and Daniil Gavrilov. Learn your reference model for real good alignment. arXiv preprint, 2024. (pp. 2, 5, and 10)
- [40] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. NeurIPS, 2020. (pp. 5 and 9)
- [41] Almog Gueta, Elad Venezian, Colin Raffel, Noam Slonim, Yoav Katz, and Leshem Choshen. Knowledge is a region in weight space for fine-tuned language models. In EMNLP, 2023. (p. 9)
- [42] Sil Hamilton. Detecting mode collapse in language models via narration. arXiv preprint, 2024. (pp. 1 and 34)
- [43] Lars Kai Hansen and Peter Salamon. Neural network ensembles. TPAMI, 1990. (p. 9)
- [44] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020. (pp. 5 and 9)
- [45] Dan Hendrycks. Natural selection favors AIs over humans. arXiv preprint, 2023. (p. 11)
- [46] Dan Hendrycks and Mantas Mazeika. X-risk analysis for AI research. arXiv preprint, 2022. (pp. 1 and 11)
- [47] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint, 2020. (pp. 7 and 9)

- [48] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In NeurIPS, 2021. (p. 9)
- [49] Geoffrey Hinton, Oriol Vinyals, and Jeffrey Dean. Distilling the knowledge in a neural network. In NeurIPS, 2015. (p. 5)
- [50] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. In NeurIPS, 2022. (p. 11)
- [51] Zitong Huang, Ze Chen, Bowen Dong, Chaoqi Liang, Erjin Zhou, and Wangmeng Zuo. Imwa: Iterative model weight averaging benefits class-imbalanced learning tasks. arXiv preprint, 2024. (p. 10)
- [52] Gabriel Ilharco, Mitchell Wortsman, Samir Yitzhak Gadre, Shuran Song, Hannaneh Hajishirzi, Simon Kornblith, Ali Farhadi, and Ludwig Schmidt. Patching open-vocabulary models by interpolating weights. In NeurIPS, 2022. (pp. 2 and 10)
- [53] Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In ICLR, 2023. (pp. 2, 3, 5, 9, 10, 22, and 23)
- [54] Geoffrey Irving, Paul Christiano, and Dario Amodei. Ai safety via debate. arXiv preprint, 2018. (p. 10)
- [55] Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization. In UAI, 2018. (pp. 2, 5, 9, 22, and 30)
- [56] Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural Tangent Kernel: Convergence and generalization in neural networks. In NeurIPS, 2018. (p. 24)
- [57] Samyak Jain, Sravanti Addepalli, Pawan Kumar Sahu, Priyam Dey, and R Venkatesh Babu. Dart: Diversify-aggregate-repeat training improves generalization of neural networks. In CVPR,

2023. (p. 10)

- [58] Dong-Hwan Jang, Sangdoo Yun, and Dongyoon Han. Model stock: All we need is just a few fine-tuned models. arXiv preprint, 2024. (pp. 6, 10, 23, 24, and 29)
- [59] Natasha Jaques, Shixiang Gu, Dzmitry Bahdanau, José Miguel Hernández-Lobato, Richard E Turner, and Douglas Eck. Sequence tutor: Conservative fine-tuning of sequence generation models with kl-control. In ICML, 2017. (pp. 2, 4, and 22)
- [60] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint, 2023. (p. 9)
- [61] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024. (pp. 7, 9, and 10)
- [62] Jean Kaddour. Stop wasting my time! saving days of ImageNet and BERT training with latest weight averaging. In NeurIPS Workshop, 2022. (p. 5)

- [63] Minchul Kim, Shangqian Gao, Yen-Chang Hsu, Yilin Shen, and Hongxia Jin. Token fusion: Bridging the gap between token pruning and token merging. In WACV, 2024. (p. 10)
- [64] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR,

2015. (p. 6)

- [65] Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu. Understanding the effects of RLHF on LLM generalisation and diversity. In ICLR, 2024. (pp. 1, 3, and 34)
- [66] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. In PNAS, 2017. (p. 1)
- [67] Ananya Kumar, Aditi Raghunathan, Robbie Matthew Jones, Tengyu Ma, and Percy Liang. Fine-tuning can distort pretrained features and underperform out-of-distribution. In ICLR,

2022. (pp. 1 and 6)

- [68] Maxime Labonne. Merge Large Language Models with mergekit, 2024. URL https://hugg ingface.co/blog/mlabonne/merge-models. (p. 2)
- [69] Maxime Labonne. NeuralBeagle14-7B. https://huggingface.co/mlabonne/NeuralBe agle14-7B-GGUF, 2024. (p. 2)
- [70] Yanis Labrak, Adrien Bazoge, Emmanuel Morin, Pierre-Antoine Gourraud, Mickael Rouvier, and Richard Dufour. Biomistral: A collection of open-source pretrained large language models for medical domains. arXiv preprint, 2024. (p. 2)
- [71] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In NeurIPS, 2017. (p. 9)
- [72] Nathan Lambert and Jacob Morrison. Model merging lessons in The Waifu Research Department, 2024. URL https://www.interconnects.ai/p/model-merging. (p. 2)
- [73] Daniel Lawson and Ahmed H Qureshi. Merging decision transformers: Weight averaging for forming multi-task policies. In ICLR RRL Workshop, 2023. (p. 10)
- [74] Angeliki Lazaridou, Anna Potapenko, and Olivier Tieleman. Multi-agent communication meets natural language: Synergies between functional and structural language learning. In ACL,

2020. (p. 2)

- [75] Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Victor Carbune, and Abhinav Rastogi. RLAIF: Scaling reinforcement learning from human feedback with AI feedback. In ICML, 2024. (p. 3)
- [76] Jaerin Lee, Bong Gyun Kang, Kihoon Kim, and Kyoung Mu Lee. Grokfast: Accelerated grokking by amplifying slow gradients. arXiv preprint, 2024. (p. 5)
- [77] Mike Lewis, Denis Yarats, Yann N Dauphin, Devi Parikh, and Dhruv Batra. Deal or no deal? end-to-end learning for negotiation dialogues. arXiv preprint, 2017. (p. 1)
- [78] Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, and Deheng Ye. More agents is all you need. arXiv preprint, 2024. (p. 11)
- [79] Zhizhong Li and Derek Hoiem. Learning without forgetting. TPAMI, 2017. (p. 1)

- [80] Ziniu Li, Tian Xu, Yushun Zhang, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models. arXiv preprint, 2023. (p. 3)
- [81] Yong Lin, Hangyu Lin, Wei Xiong, Shizhe Diao, Jianmeng Liu, Jipeng Zhang, Rui Pan, Haoxiang Wang, Wenbin Hu, Hanning Zhang, Hanze Dong, Renjie Pi, Han Zhao, Nan Jiang, Heng Ji, Yuan Yao, and Tong Zhang. Mitigating the alignment tax of rlhf. arXiv preprint, 2024. (pp. 1, 2, 3, 6, and 10)
- [82] Yong Lin, Lu Tan, Yifan Hao, Honam Wong, Hanze Dong, Weizhong Zhang, Yujiu Yang, and Tong Zhang. Spurious feature diversification improves out-of-distribution generalization. In ICLR, 2024. (p. 10)
- [83] Tianlin Liu, Shangmin Guo, Leonardo Bianco, Daniele Calandriello, Quentin Berthet, Felipe Llinares, Jessica Hoffmann, Lucas Dixon, Michal Valko, and Mathieu Blondel. Decoding-time realignment of language models. In ICML, 2024. (p. 2)
- [84] Yuchen Lu, Soumye Singhal, Florian Strub, Aaron Courville, and Olivier Pietquin. Countering language drift with seeded iterated learning. In ICML, 2020. (p. 2)
- [85] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In AISTATS, 2017. (p. 10)
- [86] Skander Moalla, Andrea Miele, Razvan Pascanu, and Caglar Gulcehre. No representation, no trust: Connecting representation, collapse, and trust issues in ppo. arXiv preprint, 2024. (pp. 1 and 34)
- [87] Daniel Morales-Brotons, Thijs Vogels, and Hadrien Hendrikx. Exponential moving average of weights in deep learning: Dynamics and benefits. TMLR, 2024. (p. 5)
- [88] Rémi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Zhaohan Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Andrea Michi, et al. Nash learning from human feedback. arXiv preprint, 2023. (pp. 2, 5, and 10)
- [89] Behnam Neyshabur, Hanie Sedghi, and Chiyuan Zhang. What is being transferred in transfer learning? In NeurIPS, 2020. (pp. 2, 9, 24, and 29)
- [90] Richard Ngo, Lawrence Chan, and Soren Mindermann. The alignment problem from a deep learning perspective. arXiv preprint, 2022. (pp. 1 and 11)
- [91] Evgenii Nikishin, Pavel Izmailov, Ben Athiwaratkun, Dmitrii Podoprikhin, Timur Garipov, Pavel Shvechikov, Dmitry Vetrov, and Andrew Gordon Wilson. Improving stability in deep reinforcement learning with weight averaging. In UDL, 2018. (p. 10)
- [92] Michael Noukhovitch, Samuel Lavoie, Florian Strub, and Aaron Courville. Language model alignment with elastic reset. In NeurIPS, 2023. (pp. 2 and 10)
- [93] OpenAI. Gpt-4 technical report. 2023. (p. 1)
- [94] Maxime Oquab, Leon Bottou, Ivan Laptev, and Josef Sivic. Learning and transferring mid-level image representations using convolutional neural networks. In CVPR, 2014. (p. 3)

- [95] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. TMLR, 2024. (pp. 5 and 9)
- [96] Guillermo Ortiz-Jimenez, Alessandro Favero, and Pascal Frossard. Task arithmetic in the tangent space: Improved editing of pre-trained models. In NeurIPS, 2023. (p. 24)
- [97] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. NeurIPS, 2022. (pp. 1 and 3)
- [98] Alexander Pan, Kush Bhatia, and Jacob Steinhardt. The effects of reward misspecification: Mapping and mitigating misaligned models. In ICLR, 2022. (p. 1)
- [99] Ethan Perez, Sam Ringer, Kamil˙e Lukoši¯ut˙e, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, et al. Discovering language model behaviors with model-written evaluations. arXiv preprint, 2022. (p. 1)
- [100] Boris T Polyak and Anatoli B Juditsky. Acceleration of stochastic approximation by averaging. SIAM, 1992. (pp. 3, 4, and 9)
- [101] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018. (p. 1)
- [102] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019. (p. 1)
- [103] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint, 2023. (p. 3)
- [104] Colin Raffel. Building Machine Learning Models Like Open Source Software. ACM, 2023. (pp. 2, 3, and 10)
- [105] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020. (p. 1)
- [106] Alexandre Ramé, Matthieu Kirchmeyer, Thibaud Rahier, Alain Rakotomamonjy, Patrick Gallinari, and Matthieu Cord. Diverse weight averaging for out-of-distribution generalization. In NeurIPS, 2022. (pp. 2, 5, 10, 24, 30, and 33)
- [107] Alexandre Ramé, Kartik Ahuja, Jianyu Zhang, Matthieu Cord, Léon Bottou, and David LopezPaz. Model ratatouille: Recycling diverse models for out-of-distribution generalization. In

- ICML, 2023. (pp. 5, 6, and 9)

[108] Alexandre Ramé, Nino Vieillard, Léonard Hussenot, Robert Dadashi, Geoffrey Cideron, Olivier Bachem, and Johan Ferret. WARM: On the benefits of weight averaged reward models. In

- ICML, 2024. (pp. 2, 3, 6, and 10)

- [109] Alexandre Ramé, Guillaume Couairon, Mustafa Shukor, Corentin Dancette, Jean-Baptiste Gaya, Laure Soulier, and Matthieu Cord. Rewarded soups: towards pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards. In NeurIPS, 2023. (pp. 2, 10, and 33)
- [110] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jeanbaptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint, 2024. (pp. 1 and 9)
- [111] Mark Rofin, Nikita Balagansky, and Daniil Gavrilov. Linear interpolation in parameter space is good enough for fine-tuned language models. arXiv preprint, 2022. (p. 2)
- [112] Paul Roit, Johan Ferret, Lior Shani, Roee Aharoni, Geoffrey Cideron, Robert Dadashi, Matthieu Geist, Sertan Girgin, Léonard Hussenot, Orgad Keller, et al. Factually consistent summarization via reinforcement learning with textual entailment feedback. In ACL, 2023. (pp. 3 and 4)
- [113] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In ICML, 2015. (pp. 4 and 9)
- [114] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint, 2017. (pp. 3 and 9)
- [115] Thibault Sellam, Dipanjan Das, and Ankur Parikh. BLEURT: Learning robust metrics for text generation. In ACL, 2020. (p. 34)
- [116] Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R Johnston, et al. Towards understanding sycophancy in language models. arXiv preprint, 2023. (p. 1)
- [117] Wei Shen, Rui Zheng, Wenyu Zhan, Jun Zhao, Shihan Dou, Tao Gui, Qi Zhang, and Xuanjing Huang. Loose lips sink ships: Mitigating length bias in reinforcement learning from human feedback. In ACL, 2023. (p. 33)
- [118] Ken Shoemake. Animating rotation with quaternion curves. In SIGGRAPH, 1985. (pp. 2, 3, 5, 10, and 22)
- [119] Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. A long way to go: Investigating length correlations in rlhf. arXiv preprint, 2023. (pp. 1 and 33)
- [120] Joar Max Viktor Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward gaming. In NeurIPS, 2022. (pp. 1 and 3)
- [121] Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. In NeurIPS, 2020. (pp. 5 and 9)
- [122] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. NeurIPS, 2020. (p. 1)
- [123] Zafir Stojanovski, Karsten Roth, and Zeynep Akata. Momentum-based weight interpolation of strong zero-shot models for continual learning. In NeurIPS Workshop, 2022. (pp. 2, 5, 6, and 10)
- [124] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint, 2022. (p. 9)

- [125] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016. (pp. 5 and 9)
- [126] Fahim Tajwar, Anikait Singh, Archit Sharma, Rafael Rafailov, Jeff Schneider, Tengyang Xie, Stefano Ermon, Chelsea Finn, and Aviral Kumar. Preference fine-tuning of llms should leverage suboptimal, on-policy data. arXiv preprint, 2024. (p. 3)
- [127] Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. NeurIPS, 2017. (pp. 3, 4, 5, 9, 10, and 31)
- [128] Jessica Taylor, Eliezer Yudkowsky, Patrick LaVictoire, and Andrew Critch. Alignment for advanced machine learning systems. Ethics of AI, 2016. (pp. 1 and 11)
- [129] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint, 2024. (pp. 1, 3, 6, and 9)
- [130] Manan Tomar, Lior Shani, Yonathan Efroni, and Mohammad Ghavamzadeh. Mirror descent policy optimization. arXiv preprint, 2020. (p. 5)
- [131] Joachim Utans. Weight averaging for neural networks and local resampling schemes. In AAAI,

1996. (pp. 2, 5, 9, and 23)

- [132] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. (p. 3)
- [133] Fanqi Wan, Xinting Huang, Deng Cai, Xiaojun Quan, Wei Bi, and Shuming Shi. Knowledge fusion of large language models. arXiv preprint, 2024. (p. 10)
- [134] Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Zou. Mixture-of-agents enhances large language model capabilities. arXiv preprint, 2024. (p. 11)
- [135] Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In ICLR,

2022. (p. 1)

- [136] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Reinforcement learning, 1992. (pp. 2, 3, 6, and 22)
- [137] Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael GontijoLopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In ICML, 2022. (pp. 2, 5, 9, 10, 23, and 24)
- [138] Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Hanna Hajishirzi, Ali Farhadi, Hongseok Namkoong, and Ludwig Schmidt. Robust fine-tuning of zero-shot models. In CVPR,

2022. (pp. 2, 3, 6, 22, and 24)

- [139] Shitao Xiao, Zheng Liu, Peitian Zhang, and Xingrun Xing. LM-cocktail: Resilient tuning of language models via model merging. arXiv preprint, 2023. (p. 10)
- [140] Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. TIES-merging: Resolving interference when merging models. In NeurIPS, 2023. (p. 9)

- [141] Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. Language models are super mario: Absorbing abilities from homologous models as a free lunch. arXiv preprint, 2023. (p. 9)
- [142] Kerem Zaman, Leshem Choshen, and Shashank Srivastava. Fuse to forget: Bias reduction and selective memorization through model fusion. arXiv preprint, 2023. (p. 10)
- [143] Chujie Zheng, Ziqi Wang, Heng Ji, Minlie Huang, and Nanyun Peng. Weak-to-strong extrapolation expedites alignment. arXiv preprint, 2024. (pp. 27 and 28)
- [144] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-bench and chatbot arena. In NeurIPS, 2023. (p. 9)
- [145] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint, 2019. (p. 1)

### WARP: On the Benefits of Weight Averaged Rewarded Policies

##### Supplementary material

This supplementary material is organized as follows:

- • Appendix A provides additional illustration of the WARP procedure.
- • Appendix B details theoretical insights on task vectors, SLERP, LERP and LITI.
- • Appendix C details empirical insights on task vectors, SLERP, LERP and LITI.
- • Appendix D shows the impact of different design choices in WARP.
- • Appendix E investigates a potential length bias in WARP, and how to fix it.
- • Appendix F explores the relationship between KL and diversity in generations.

#### A. Strategy illustration

In Figure 5, we propose an alternative illustration of WARP, where the different stages are more detailed than in Figure 1(a). Then in Figure 6, we also refine our illustration showcasing the similarity and difference between SLERP and LERP.

- 𝜃ema1

- 𝜃ema2

...

- 𝜃rl1

- 𝜃rl2

(1 − 𝜂) · 𝜃init + 𝜂 · 𝜃′

slerp

𝜃slerp 𝜃′

𝜃init 𝜃′

slerp

init

...

REINFORCE [136]

KL [59]

EMA [55]

SLERP [118] of task vectors [53]

LITI [138]

- Figure 5 | Detailed illustration of the WARP strategy. From a (pre-trained and supervised fine-tuned) LLM

𝜃init, we launch 𝑀 = 2 fine-tunings (black arrows ). The innovation of WARP lies in the use of model merging by weight averaging at three different stages. First, the exponential moving averages (EMA, blue dashed arrows ) of the policy (collected at different training steps) serves as the anchor for the KL regularization (black double-headed dotted arrows ). The fine-tuned networks are weight averaged using spherical linear interpolation of task vectors (SLERP, yellow dashed arrows ). Third, we interpolate towards the initialization (LITI, red dashed arrows ). This obtained model 𝜃′

init serves as an updated initialization for the next iteration, progressively refining the model’s capabilities and alignment. Overall, the final model 𝜃′

slerp has high reward but also high KL. Then, by interpolation towards the SFT init, we reveal a KL-reward Pareto front of solutions: {(1 − 𝜂) · 𝜃sft + 𝜂 · 𝜃slerp𝐼 | 0 ≤ 𝜂 ≤ 1}.

𝜃1

𝜃slerp𝜆

slerp

𝛿𝜆

𝜃lerp𝜆

𝛿1

Ω ≈ 90◦

𝛿𝜆lerp

𝜃init

𝛿2

𝜔 ≈ 0◦ 𝜃2

𝜃0

- Figure 6 | Illustration of the difference between the full weights 𝜃𝑚 and their task vectors 𝛿𝑚 = 𝜃𝑚 − 𝜃init, where darker areas are of better performance. We found in Appendix C.2 that Ω ≈ 90◦ where Ω is the angle between

task vectors such as cos Ω = ∥𝛿𝛿11∥∥·𝛿𝛿22∥ , while 𝜔 the angle between the full weights such as cos 𝜔 = ∥𝜃𝜃11∥∥·𝜃𝜃22∥ satisfies 𝜔 ≈ 0◦.

#### B. Theoretical insights on task vectors, SLERP, LERP and LITI

Based on the insights from [53] that task vectors (the differences between a fine-tuned model and its initialization) are semantically manipulable and interpretable units in the weight space, we compare SLERP and LERP merging operations by analyzing their task vectors.

Background. Linear interpolation (LERP) [131] is the simplest merging strategy, notably used in the model soups variants [137], and defined as:

lerp 𝜃1, 𝜃2, 𝜆 = (1 − 𝜆) · 𝜃1 + 𝜆 · 𝜃2. (LERP) Then, as illustrated in Figure 6, the task vector for LERP with interpolating coefficient 𝜆 is given by: 𝛿lerp𝜆 = lerp 𝜃1, 𝜃2, 𝜆 −𝜃init = (1− 𝜆) ·𝛿1 + 𝜆 ·𝛿2. Similarly, we define 𝛿slerp𝜆 = slerp 𝜃init, 𝜃1, 𝜃2, 𝜆 −𝜃init where slerp is defined in Equation (SLERP).

###### B.1. Theoretical insights on the SLERP and LERP task vectors We denote Ω the angle between the the task vectors 𝛿1 and 𝛿2:

𝛿1 · 𝛿2 ∥𝛿1∥∥𝛿2∥

cos Ω =

. (2)

Based on the empirical observations from [58], confirmed in our Figure 11(c), we introduce the following Assumption 1 for simplicity.

- Assumption 1 (Task vectors of equal norm). Independently fine-tuned task vectors have a same norm 𝑙: ∥𝛿1∥ = ∥𝛿2∥ = 𝑙. (3)

- Lemma 1 (SLERP task vector). Under Assumption 1, SLERP preserves the norm of the task vector: ∥𝛿slerp𝜆 ∥ = 𝑙. (4)

Proof. By definition,

sin[𝜆Ω]

sin[(1 − 𝜆)Ω] sin Ω · 𝛿1 +

sin Ω · 𝛿2 (5) Then, as 𝛿1 · 𝛿2 = 𝑙2 cos Ω,

𝛿slerp𝜆 =

∥𝛿slerp𝜆 ∥2 𝑙2

2

2

sin[(1 − 𝜆)Ω] sin Ω

sin[𝜆Ω] sin Ω

sin[𝜆Ω] sin Ω

sin[(1 − 𝜆)Ω] sin Ω

+ 2

cos(Ω) +

(6)

=

sin2[(1 − 𝜆)Ω] + 2sin[(1 − 𝜆)Ω] sin[𝜆Ω] cos(Ω) + sin2[𝜆Ω] sin2 Ω

(7)

=

sin2 Ω sin2 Ω

(8)

=

= 1 (9) using trigonometric identities, proving Lemma 1. □

- Lemma 2 (LERP task vector). Under Assumption 1, LERP reduces the norm of the task vector:

∥𝛿lerp𝜆 ∥ = 𝑙√︃1 − 2(1 − cos Ω)(𝜆 − 𝜆2). (10)

We recover that averaging weights with 𝜆 = 0.5 tends to reduce the norm of the task vectors, as previously highlighted in [58].

Proof. By definition:

𝛿lerp𝜆 = (1 − 𝜆) · 𝛿1 + 𝜆 · 𝛿2. (11) Then, as 𝛿1 · 𝛿2 = 𝑙2 cos Ω,

∥𝛿slerp𝜆 ∥2 𝑙2

= (1 − 𝜆)2 + 2𝜆(1 − 𝜆) cos Ω + 𝜆2 (12) = 1 − 2𝜆(1 − cos Ω) + 2𝜆2(1 − cos Ω) (13) = 1 − 2(1 − cos Ω)(𝜆 − 𝜆2), (14)

proving Lemma 2 when 0 < 𝜆 < 1. □

- B.2. Theoretical insights on the KL

- B.2.1. Linear regime

- Assumption 2 (Linear regime [138]). We assume that the predictions of a model 𝑓, with weights initialized from 𝜃0 and fine-tuned into 𝜃, can be approximated by first-order Taylor expansion: ∀𝒙,

𝑓 (𝒙, 𝜃) ≈ 𝑓 (𝒙, 𝜃0) + (𝜃 − 𝜃0) · ∇𝜃 𝑓 (𝒙, 𝜃0). (15)

- Assumption 2 defines a neural tangent [56] space in which the relationship between weights and functions is linear. As previously argued in [106, 137], this Taylor expansion is reasonable partly because weights remain close during fine-tunings [89], as confirmed in Figure 11 where they have equal norms and a cosine of one. Yet, please note that [96] highlighted some limitations.

###### B.2.2. KL variations for LERP

We consider 𝜃1 and 𝜃2 weights fine-tuned from a shared SFT initialization 𝜃sft. Then in the linear regime from Assumption 2, weight and prediction ensembling behaves similarly:

𝑓 𝒙, (1 − 𝜆) · 𝜃1 + 𝜆 · 𝜃2 ≈ (1 − 𝜆) · 𝑓 (𝒙, 𝜃1) + 𝜆 · 𝑓 (𝒙, 𝜃2). (16) This similarity enables to prove the following Lemma 3.

Lemma 3 (LERP reduces KL). For an interpolating coefficient 0 ≤ 𝜆 ≤ 1, denoting 𝜋𝜆 the LERP policy from weight interpolation (1 − 𝜆) · 𝜃1 + 𝜆 · 𝜃2, and 𝜋ˆ𝜆 the ensembling policy from prediction interpolation (1 − 𝜆) · 𝜋𝜃1 + 𝜆 · 𝜋𝜃2, then under Assumption 2,

KL(𝜋𝜆||𝜋𝜃sft) ≈ KL(𝜋ˆ𝜆||𝜋𝜃sft) ≤ (1 − 𝜆)KL(𝜋𝜃1||𝜋𝜃sft) + 𝜆KL(𝜋𝜃2||𝜋𝜃sft), (17) i.e., the KL for LERP is lower than the interpolated KL.

Proof. The following proof applies the linear assumption and properties of the KL divergence. Approximation of KL. The first approximate equality is a direct application of Assumption 2 to 𝜋𝜆. Precisely, applying Equation (16) to the definition of 𝜋𝜆 = 𝜋(1−𝜆)𝜃1+𝜆𝜃2 yields that 𝜋𝜆 ≈ 𝜋ˆ𝜆. Upper bound of the KL. The KL divergence is convex in both its arguments [23], thus we directly have that

KL((1 − 𝜆) · 𝜋𝜃1 + 𝜆 · 𝜋𝜃2||𝜋𝜃sft) ≤ (1 − 𝜆)KL(𝜋𝜃1||𝜋𝜃sft) + 𝜆KL(𝜋𝜃2||𝜋𝜃sft), (18) which completes the proof.

□

Remark 1. Lemma 3 shows that the LERP 𝜋𝜆 is closer in KL to the original SFT initialization. This relates to Lemma 2, where we show that the linear interpolation reduces the norm to the initialization. As the interpolation brings the weights of the models closer, it is natural that it would also bring the resulting policies closer.

###### B.2.3. KL and reward variation for LITI

We now consider a given weight 𝜃 (in practice either obtained from LERP or SLERP of multiple fine-tuned weights) and its associated task vector 𝛿 = 𝜃−𝜃sft. In the linear regime from Assumption 2, for each 𝜂 ∈ [0, 1], we have the following:

𝑓 (𝒙, 𝜃sft + 𝜂 · 𝛿) − 𝑓 (𝒙, 𝜃sft) ≈ 𝜂 · ( 𝑓 (𝒙, 𝜃sft + 𝛿) − 𝑓 (𝒙, 𝜃sft)). (19) We try to show that:

KL 𝜋𝜃sft+𝜂·𝛿∥𝜋𝜃sft ≤ 𝜂 · KL 𝜋𝜃sft+𝛿∥𝜋𝜃sft . (20)

- Lemma 4 (KL upper bound for interpolated distributions). For an interpolating coefficient 0 ≤ 𝜂 ≤ 1, denoting 𝜋𝜂 the LITI policy from weight interpolation 𝜃sft + 𝜂 · 𝛿, and 𝜋ˆ𝜂 the ensembling policy from prediction interpolation (1 − 𝜂) · 𝜋𝜃sft + 𝜂 · 𝜋𝜃sft+𝛿, then under Assumption 2, For each 𝜂 ∈ [0, 1], we have that

KL 𝜋𝜂∥𝜋𝜃sft ≈ KL(𝜋ˆ𝜂∥𝜋𝜃sft) ≤ 𝜂KL 𝜋𝜃sft+𝛿∥𝜋𝜃sft . (21)

Proof. The following proof uses the same method as the one of Lemma 3. We use Assumption 2 to link the policy with the interpolation of polices, and the inequality is a result of the KL convexity.

Approximation of KL. The first approximate equality is a direct application of Assumption 2 to 𝜋𝜂. Precisely, applying Equation (19) to the definition of 𝜋𝜂 = 𝜋𝜃sft+𝜂·𝛿 yields that 𝜋𝜂 ≈ 𝜋ˆ𝜂. Upper bound of the KL. Using the fact that the KL is convex, we have

KL(𝜂 · 𝜋𝜃sft+𝛿 + (1 − 𝜂) · 𝜋𝜃sft||𝜋𝜃sft) ≤ 𝜂KL 𝜋𝜃sft+𝛿∥𝜋𝜃sft . (22) □

- Assumption 3 (LITI reward is above the expected reward). The rewards for the LITI interpolated weights are above the interpolated rewards:

𝑟(𝜋0 + 𝜂 · (𝜋 − 𝜋𝜃sft)) ≥ 𝜂𝑟(𝜋) + (1 − 𝜂)𝑟(𝜋𝜃sft), (23)

This Assumption 3 is based on observations from Figure 9(b), and extends to a reward maximization setup the notion of linear mode connectivity [30], usually defined w.r.t. the accuracy in supervised learning.

- Lemma 5 (LITI Pareto optimality). Be given the convexity of the KL from Lemma 4 and the concavity of the reward 𝑟 in Assumption 3, then the reward vs. KL front of LITI is above the diagonal. Illustration in Figure 7.

Proof. We obtain a policy 𝜋𝜃 fine-tuned from 𝜋𝜃sft. The LITI policy for 𝜃𝜂 = (1 − 𝜂) · 𝜃sft + 𝜂 · 𝜃 is noted 𝜋𝜂. Combining the approximation from Lemma 4 and Assumption 3, we have that

𝑟(𝜋𝜂) ≥ (1 − 𝜂)𝑟(𝜋𝜃sft) + 𝜂𝑟(𝜋𝜃). (24) And, from Lemma 4, we also have that

KL 𝜋𝜂∥𝜋𝜃sft ≤ 𝜂KL 𝜋𝜃∥𝜋𝜃sft . (25)

This means that for every LITI coefficient 𝜂, the LITI policy has a higher reward than the interpolated reward at a lower KL. Geometrically, this means that each point on the Reward-KL front from LITI is on the top left quadrant of the plane according to the corresponding point on the diagonal. □

###### B.3. Uniformly averaging 𝑀 > 2 weights with SLERP

The SLERP merging formula from Equation (SLERP) is only defined for 𝑀 = 2 weights. We trivially (and certainly suboptimally) generalize this to 𝑀 > 2 weights in the uniform averaging setup, thus giving an equal coefficient to each of them, i.e., 𝜆 = 𝑀1 . In that setup, removing the dependency of 𝜃init that is assumed shared, we generalize SLERP to merge 𝑀 weights uniformly through the iterative procedure defined below:

1

slerpm {𝜃𝑚}𝑚𝑀=1 = slerp slerpm {𝜃𝑚}𝑚𝑀=−11 , 𝜃𝑀, 𝜆 =

𝑀

. (26)

Though these operations are not associative, the standard deviations in performances are small, as indicated by the shaded areas in Figure 4(b).

Equation (24)

𝑟(𝜋)

LITI front

𝜋𝜃

𝜋𝜂

Equation (25)

KL(𝜋||𝜋𝜃sft)

𝜂KL(𝜋||𝜋𝜃sft)

𝜋𝜃sft

- Figure 7 | Illustration of Lemma 5. Based on experimental observation and theoretical insights, we see that the Pareto front of the LITI policy is better than the identity. It highlights how Equations (24) and (25) place LITI policies on the KL-reward plane.

#### C. Empirical insights on task vectors, SLERP, LERP and LITI

###### C.1. Empirical insights on the difference between SLERP and LERP

We now empirically investigate how those theoretical differences between SLERP and LERP affect the performance of the merged policies.

SLERP vs. LERP. In Figure 8 we adjust the interpolating coefficient 𝜆, highlighting distinct behaviors for SLERP and LERP. SLERP consistently enhances rewards more than LERP, as depicted in Figures 3(c) and 8(a). However, a comprehensive evaluation must consider both KL and reward. As shown in Figure 8(b), LERP consistently reduces KL, corroborating with Lemma 2 that LERP reduces the norm of updates (while SLERP preserves it). When plotting these metrics together in Figure 8(c), we observe that SLERP and LERP target different regions on the Pareto front: SLERP achieves higher rewards at the expense of increased KL, while the main impact of LERP is to lower KL. This is consistent with Lemmas 2 and 3, be given the orthogonal angles between task vectors Ω ≈ 90◦ (as shown in Figure 11(a)).

Combining SLERP and LERP with LITI. We also compare the behaviours of SLERP and LERP when we apply LITI, as we adjust the interpolating coefficient 𝜂. Figure 9(a) and Figure 9(b) validate that KL is convexe with regard to 𝜂 while the reward is concave with regard to 𝜂, for different values of 𝑀. This is also highlighted in Figure 10(a), which reproduces the results from Figure 4(b) (and maintaining the same axis limits), replacing SLERP by LERP: this leads to critical changes in the Pareto fronts. Inded, increasing 𝑀 now tends to decrease KL for LERP, while it used to increase reward with SLERP. In Figure 10(b), we explore the extrapolation strategies from [143], using 0 ≤ 𝜂 ≤ 2 to compare the full extrapolated fronts from LERP and SLERP. While both perform similarly on low KL, our results suggest that SLERP perform better in high KL regions.

Conclusion. SLERP demonstrates some key advantages. In particular, it reveals the full Pareto front of solutions, while LERP only exposes a portion; extrapolation Figure 10(b) with 𝜂 > 1 can partially mitigate this but as our experiments suggest, LERP curves consistently lag behind SLERP curves in high-reward regions. Moreover, from a practical perspective, SLERP scales the choice of 𝜂 effectively, where 1 represents full updates and a fixed value of 0.3 always corresponds to the same operational region, optimizing for high reward and KL.

0.35

0.36

0.37

Reward

0.38

0.39

SLERP

LERP

0.0 0.2 0.4 0.6 0.8 1.0

(a) Reward vs. 𝜆.

140

130

120

KL

110

100

SLERP

90

LERP

0.0 0.2 0.4 0.6 0.8 1.0

(b) KL vs. 𝜆.

0.35

0.36

= 0.8

= 0.6

0.37

Reward

= 0.3

= 0.7

= =0.50.4

0.38

= 0.2 = 0.1 = 0.0

- = 0.0

= 0.1

= 0.2

= 0.3

= 0.4

- = 0.5

- = 0.6

= 0.7

- = 0.8

- = 0.9

- = 1.0

= 1.0

0.39

SLERP

LERP

= 0.9

90 100 110 120 130 140

KL

(c) Reward vs. KL.

- Figure 8 | SLERP vs. LERP when sliding the interpolating coefficient 𝜆. Considering 𝑀 = 2 weights after 𝑇 = 9𝑘 RL steps, we merge them using either SLERP or LERP, while sliding the interpolating coefficient 𝜆 between 0 and 1. We then evaluate the merged checkpoints. Figure 8(a) shows that SLERP leads to higher reward than LERP, as previously in Figure 3(c). Figure 8(b) shows that LERP signicantly reduces the KL (consistently with Lemma 3) while SLERP slightly increases it. Figure 8(c) shows how this impact the KL-reward Pareto front, where larger markers/darker colors indicate higher values of 𝜆; while SLERP covers high KL-high reward regions, LERP tends to cover regions of lower KL and thus also lower rewards.

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

0.0 0.2 0.4 0.6 0.8 1.0

0

20

40

60

80

100

120

140

160

KL

SLERP M = 5

LERP M = 5

SLERP M = 4

LERP M = 4

SLERP M = 3

LERP M = 3

SLERP M = 2

LERP M = 2

M = 1

(a) KL for 𝜂.

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>SLERP M = 5<br><br>LERP M = 5<br><br>SLERP M = 4<br><br>LERP M = 4<br><br>SLERP M = 3<br><br>LERP M = 3<br><br>SLERP M = 2<br><br>LERP M = 2<br><br>M = 1| | | | | | |
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

0.0 0.2 0.4 0.6 0.8 1.0

0.75

0.70

0.65

0.60

0.55

0.50

0.45

0.40

0.35

Reward

(b) Reward for 𝜂.

- Figure 9 | SLERP vs. LERP when sliding the interpolating coefficient 𝜂 of LITI. In Figure 9(a) we show that the KL is convex (and almost linear) with regard to 𝜂, consistently with Lemma 4. In contrast, Figure 9(b) shows that the reward is concave, validating Assumption 3.

| |
|---|

| |
|---|

0 20 40 60 80 100 120 140 160

KL

0.70

0.65

0.60

0.55

0.50

0.45

0.40

0.35

Reward

- = 0.0

- = 0.1

= 0.3

= 0.5

= 0.8

= 1.0

M = 5 M = 4 M = 3 M = 2 M = 1 REINFORCE

(a) LITI of LERP of 𝑀 weights.

|| | |
|---|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | |
|---|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>= 0.0<br><br>= 0.1<br><br><br>= 0.3<br><br>= 0.5<br><br>= 0.8<br><br>==1.01.1<br><br>= 1.3<br><br>= 1.5<br><br>= 1.8<br><br>= 2.0<br><br><br>SLERP M = 5<br><br>LERP M = 5<br><br>SLERP M = 4<br><br>LERP M = 4<br><br>SLERP M = 3<br><br>LERP M = 3<br><br>SLERP M = 2<br><br>LERP M = 2<br><br>M = 1<br><br>REINFORCE| | | | |
|---|---|---|---|---|
| | | | | |

0 50 100 150 200 250

KL

0.75

0.70

0.65

0.60

0.55

0.50

0.45

0.40

0.35

Reward

(b) Extrapolation with 0 ≤ 𝜂 ≤ 2.

0 20 40 60 80 100 120 140

KL

0.75

0.70

0.65

0.60

0.55

0.50

0.45

0.40

0.35

Reward

- = 0.0

- = 0.1

= 0.3

= 0.5

= 0.8 = 1.0

SLERP w/ task vectors

LERP

SLERP w/ weights

REINFORCE

(c) SLERP w/o task vectors.

- Figure 10 | SLERP vs. LERP when sliding the interpolating coefficient 𝜂 of LITI. Figure 10(a) merges 𝑀

policies with LERP and 𝜆 = 𝑀1 (the endpoints on the top right of the solid lines), and then interpolates towards their SFT init, where light-colored areas show standard deviations across 5 experiments, and with 0 ≤ 𝜂 ≤ 1.

In contrast, in Figure 10(b) we investigate extrapolation [143], using 0 ≤ 𝜂 ≤ 2 enabling to compare the full fronts of solutions with LERP and SLERP. Finally, Figure 10(c) confirms that applying SLERP on the full weights 𝜃 rather than on the task vectors 𝛿 perform very similarly to LERP.

###### C.2. Empirical insights on the role of task vectors

We now explore the effectiveness of applying SLERP on task vectors 𝛿 vs. full weights 𝜃, as illustrated in Figure 6. To this end, in Figure 11 we draw inspiration from [58] and plot the angles Ω and 𝜔 and norms of 𝛿 and 𝜃.

Angles of task vectors Ω ≈ 90◦. Figure 11(a) shows that the task vectors are typically orthogonal (Ω ≈ 90◦), highlighting the diverse trajectories of the different RL fine-tunings. This is in contrast with [58] for supervised fine-tunings, where Ω typically range between 40◦ and 80◦. We suspect that this is related to the underlying differences between reinforcement and supervised learning; in RL the policies are trained on their own generations, creating more orthogonal task vectors, whereas in supervised learning the LLM try to imitate the groundtruth labels, leading to more similar task vectors. The orthogonality of our task vectors prevents the use of the update rule 𝜂 → 12cos+cosΩΩ suggested from Eq. 2 in [58], as it would lead to 𝜂 ≈ 0, deleting any potential update.

Angles of full weights 𝜔 ≈ 0◦. In contrast, Figure 11(b) show that full weights remain collinear (𝜔 ≈ 0◦). This explains the empirical results from Figure 10(c), where applying SLERP directly to full weights results in behaviors similar to LERP. Indeed, as the angles 𝜔 ≈ 0◦, spherical interpolation effect is minimal because sin(𝑥) ≈ 𝑥 + O(𝑥3), and thus sinsin[𝜆𝜔𝜔 ] ≈ 𝜆𝜔𝜔 ≈ 𝜆.

Norms consistency. Figure 11(c) confirms the consistency in the norms of different task vectors, supporting our Assumption 1. This uniformity is aligned with previous research [58]. As a side note, this consistency extends to full weights 𝜃, confirming that fine-tuning typically results in minimal changes to the overall weight [89].

| | |
|---|---|
| | |
| | |
| | |
| | |

40

30

Density

20

10

0

0.0 0.2 0.4 0.6 0.8 1.0

Cosines of task vectors

(a) ∥𝛿𝛿11∥∥·𝛿𝛿22∥ .

40

30

Density

20

10

0

0.9990 0.9995 1.0000

Cosines of weights

(b) ∥𝜃𝜃11∥∥·𝜃𝜃22∥ .

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

30

25

20

Density

15

10

5

0

0.95 1.00 1.05 1.10 1.15

Ratio of task vectors' norms

1∥ ∥𝛿2∥ .

(c) ∥𝛿

20.0

17.5

15.0

Density

12.5

10.0

7.5

5.0

2.5

0.0

0.999 1.000 1.001

Ratio of weights' norms

1∥ ∥𝜃2∥ .

(d) ∥𝜃

- Figure 11 | Angles and norms of (full) weights 𝜃𝑚 and their task vectors 𝛿𝑚 = 𝜃𝑚 − 𝜃init. The histograms represent data across the 28 layers of the Gemma "7B" architecture. Figure 11(a) plots the histograms of task vector cosines. Figure 11(b) plots the histograms of weights cosines. Figure 11(c) plots the histograms of task vector norms ratio. Figure 11(d) plots the histograms of weights norms ratio.

#### D. Empirical investigation of several design choices

We include several experiments showcasing the robustness of WARP to different design choices, while further demonstrating its superiority in terms of KL-reward Pareto optimality. Specifically, Appendix D.1 analyzes the performances along training at different steps 𝑇; Appendix D.2 provides results with different values for the hyperparameters 𝜇 and 𝛽; Appendix D.3 shows the impact of the update rate 𝜂 to provide an improved initialization for the 2nd iteration of WARP; finally, Appendix D.4 shows that in iterative WARP, interpolating towards the episode initialization or the SFT initialization both perform similarly.

###### D.1. Analyzing the number of training steps

5th iter: EMA anchor 4th iter: EMA anchor 3rd iter: EMA anchor 2nd iter: EMA anchor 1st iter: EMA anchor

0.3

0.4

1st iter: SFT anchor = 0.0

1st iter: SFT anchor = 0.0001

Reward

1st iter: SFT anchor = 0.001

0.5

1st iter: SFT anchor = 0.01

1st iter: SFT anchor = 0.1

0.6

0.7

0 2000 4000 6000 8000

# steps

(a) Reward along training.

200

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

5th iter: EMA anchor 4th iter: EMA anchor 3rd iter: EMA anchor 2nd iter: EMA anchor 1st iter: EMA anchor

175

150

- 1st iter: SFT anchor = 0.0

- 1st iter: SFT anchor = 0.0001

125

1st iter: SFT anchor = 0.001

KL

1st iter: SFT anchor = 0.01

100

1st iter: SFT anchor = 0.1

75

50

25

0

0 2000 4000 6000 8000

# steps

(b) KL along training.

- Figure 12 | Rewards and KL at different number of training steps 𝑇. Figures 12(a) and 12(b) complement Figure 3(b) and Figure 4(c), this time plotting rewards and KL separately as a function of the number of training steps 𝑇. Regarding iterative WARP, we observe that each iteration has higher rewards but also higher KL (by starting at training step 0 from a new initialization). Regarding the baseline (REINFORCE with SFT anchor), we observe that low values of 𝛽 lead to very fast hacking of the reward, as visible by the KL exploding, while high values of 𝛽 slow down the learning procedure.

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

0 20 40 60 80 100 120

KL

0.70

0.65

0.60

0.55

0.50

0.45

0.40

Reward

- = 0.0

- = 0.1

= 0.3

= 0.5

= 0.8

= 1.0

MA

T = 9k T = 7k T = 5k T = 3k REINFORCE

- Figure 13 | LITI with 𝑀 = 1 at different number of training steps 𝑇. The reward gain is significantly reduced compared to Figure 4(a) where we first merged 𝑀 = 2 policies before applying LITI. We also try to perform moving average (MA) [6, 55] before applying LITI, averaging multiple checkpoints collected along a single RL fine-tuning at steps {6𝑘, 7𝑘, 8𝑘, 9𝑘}; this does not improve performances, suggesting the need to merge weights from independent fine-tunings to have enough diversity [106].

###### D.2. Analyzing the values of 𝜇 and 𝛽

| |EMA anchor<br><br>Policy: EMA anchor<br><br>EMA: = 0.0<br><br>Policy: = 0.0<br><br>EMA: = 0.001<br><br>Policy: = 0.001<br><br><br><br><br>EMA: = 0.01<br><br>Policy: = 0.01<br><br>EMA: = 0.1<br><br>Policy: = 0.1<br><br>| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.40

0.45

0.50

Reward

0.55

0.60

0.65

0.70

0.75

0 50 100 150 200

KL

(a) Reward vs. KL.

| |EMA anchor<br><br>Policy: EMA anchor<br><br>EMA: = 0.0<br><br>Policy: = 0.0<br><br>EMA: = 0.001<br><br>Policy: = 0.001<br><br><br><br><br>EMA: = 0.01<br><br>Policy: = 0.01<br><br>EMA: = 0.1<br><br>Policy: = 0.1<br><br>| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.40

0.45

0.50

Reward

0.55

0.60

0.65

0.70

0.75

0 2000 4000 6000 8000

# steps

(b) Reward vs. steps.

EMA anchor

200

Policy: EMA anchor

- EMA: = 0.0

- Policy: = 0.0

EMA: = 0.001

- Policy: = 0.001

150

EMA: = 0.01

Policy: = 0.01

KL

EMA: = 0.1

100

Policy: = 0.1

50

0

0 2000 4000 6000 8000

# steps

(c) KL vs. steps.

- Figure 14 | EMA vs. their base policies, complementing the results from Figures 3(a) and 3(b). We confirm in Figure 14(a) that the EMA of all variants (with SFT anchor) perform similarly or better than their base policies in terms of KL-reward Pareto optimality. As a reminder, we perform evaluation every 100 steps, and train them for 𝑇 = 9𝑘 steps, though we stopped the trainings if the base policy ever reaches a KL of 200. This confirms Observation 1; the benefits of our variant with EMA anchor is partly explained by distillation from an improved mean teacher [127].

| | |
|---|---|

| |
|---|

| |
|---|

| |
|---|

| | |
|---|---|
| | |

0 20 40 60 80 100 120

KL

0.75

0.70

0.65

0.60

0.55

0.50

0.45

0.40

Reward

- = 0.01 = 0.1

= 0.005 = 0.1

- = 0.01 = 0.2

= 0.01 = 0.1 w/ LP

(a) Reward vs. KL.

| | |
|---|---|

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

| |
|---|

| |
|---|

| |
|---|

| | | |
|---|---|---|
| | | |

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0 2000 4000 6000 8000

# steps

0.75

0.70

0.65

0.60

0.55

0.50

0.45

0.40

Reward

- = 0.01 = 0.1

= 0.005 = 0.1

- = 0.01 = 0.2

= 0.01 = 0.1 w/ LP

(b) Reward vs. steps.

- Figure 15 | Experiments ablating the values for the EMA update rate 𝜇 and the KL regularization strength 𝛽. So far we have systematically used 𝜇 = 0.01 and 𝛽 = 0.1 for all EMA-based runs, including in the iterative WARP. These hyperparameters were chosen at the project’s onset and have remained unchanged. In Figures 15(a) and 15(b) we increase regularization with 𝜇 = 0.005 and 𝛽 = 0.2. Our results indicate that reducing 𝜇 or increasing 𝛽 behaves similarly, marginally improving the KL-reward Pareto front but slowing down training. Additionally, we include the training trajectory when using a length penalty (LP), as detailed in Appendix E.

###### D.3. Analyzing the values of 𝜂

0.30

0.32

| |
|---|

0.34

| |
|---|

0.36

| |
|---|

Reward

| |
|---|

0.38

| |
|---|

2nd iter: LITI M = 4 2nd iter: LITI M = 2, {0.3, 0.5}

| |
|---|

| |
|---|
| |

0.40

| | |
|---|---|
| | |

| |
|---|

2nd iter: LITI M = 2, = 0.3 2nd iter: LITI M = 2, = 0.5

| |
|---|

| |
|---|

| |
|---|

0.42

| |
|---|

| |
|---|

2nd iter: REINFORCE = 0.3 2nd iter: REINFORCE = 0.5 1st iter: LITI

| |
|---|

| |
|---|

| | |
|---|---|
| | |

| |
|---|

0.44

1st iter: REINFORCE

0.46

20 40 60 80 100 120 140 160 180

KL

Figure 16 | Experiments ablating the LITI update rate 𝜂. As we initiate the 2nd iteration of WARP, selecting an appropriate value for 𝜂 is key, as it determines the starting point 𝜃𝜂 and functions similarly to an outer learning rate (see Section 6). We usually set 𝜂 = 0.3, but now provide results with an increased 𝜂 = 0.5, starting the 2nd iteration from a more “advanced” position on the previous Pareto front. We run and average 𝑀 = 2 fine-tunings from each of those two initializations for 𝑇 = 7𝑘 steps, before applying LITI. Our results indicate that a higher 𝜂 (0.5) performs better in regions of high KL, whereas a lower 𝜂 (0.3) helps in regions with KL below 65. This suggests that the optimal choice for 𝜂 is compute-dependent; a lower 𝜂 is appropriate if further iterations can explore high KL regions, whereas a limited compute budget might benefit from a higher 𝜂. This resembles the learning rate trade-off in optimization, where lower rates improve results but require more training steps. As a final note, we can also use different 𝜂 for the different fine-tunings; notably, we observe that merging all those 𝑀 = 4 RLs perform better (though it doubles the compute).

###### D.4. Interpolate towards the initialization? or towards the SFT?

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>= 0.0<br><br>= 0.1<br><br><br>= 0.3<br><br>= 0.5<br><br>= 0.8<br><br>= 1.0<br><br>5th iter: Interpolate 4th<br><br>5th iter: Interpolate SFT<br><br>4th iter: Interpolate 3rd<br><br>4th iter: Interpolate SFT<br><br>3rd iter: Interpolate 2nd<br><br>3rd iter: Interpolate SFT<br><br>2nd iter: Interpolate 1st<br><br>2nd iter: Interpolate SFT<br><br>1st iter: Interpolate SFT<br><br>| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.30

0.35

Reward

0.40

0.45

0.50

0.55

20 40 60 80 100 120 140 160

KL

- Figure 17 | Experiments ablating the initialization in LITI. We compare linear interpolating either towards the episode-specific initialization (i.e., the 𝜃𝜂 selected from previous iteration) or towards the SFT, which was the initialization of the 1st episode. The two resulting Pareto fronts are similar. However, in our iterative experiments we interpolate towards the episode-specific initialization as it allows maintaining a constant 𝜂 at each WARP iteration, enabling a smooth progression towards the high KL regions.

#### E. Addressing length bias in WARP

315

| || | |
|---|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | |
|---|---|
| | |
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | |
|---|---|
| | |
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>= 0.0<br><br>= 0.1<br><br><br>= 0.3<br><br>= 0.5<br><br>= 0.8<br><br>= 1.0<br><br>5th iter: LITI<br><br>5th iter: REINFORCE<br><br>4th iter: LITI<br><br>4th iter: REINFORCE<br><br>3rd iter: LITI<br><br>3rd iter: REINFORCE<br><br>2nd iter: LITI<br><br>2nd iter: REINFORCE<br><br>1st iter: LITI<br><br>1st iter: REINFORCE| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

310

305

300

#tokens

295

290

285

280

275

270

0 20 40 60 80 100 120 140

KL

(a) Length bias in iterative WARP.

320

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>= 0.0<br><br>= 0.1<br><br><br>= 0.3 = 0.5<br><br>= 0.8<br><br>= 1.0<br><br>WARP of 1 w/o LP + 1 w/ LP<br><br>WARP of 2 w/o LP<br><br><br>REINFORCE w/ LP<br><br>1st REINFORCE w/o LP<br><br>2nd REINFORCE w/o LP<br>| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

300

280

#tokens

260

240

220

200

0 20 40 60 80 100 120 140

KL

(b) Adding length penalty (LP).

= 1.0

= 0.8

| |
|---|

| |
|---|

= 0.5

| |
|---|

0.4

= 0.3

| |
|---|

0.5

Reward

- = 0.0

- = 0.1

| |
|---|

0.6

- WARP of 1 w/o LP + 1 w/ LP

- WARP of 2 w/o LP

REINFORCE w/ LP

0.7

- 1st REINFORCE w/o LP

- 2nd REINFORCE w/o LP

0 20 40 60 80 100 120 140

KL

(c) Benefits of diversity.

- Figure 18 | Addressing length bias in WARP. Figure 18(a) explores how length and KL change in successive WARP iterations. Figure 18(b) demonstrates the effectiveness of length penalty (LP) in reducing output length, and how such policies can be merged with others trained without LP. Finally, Figure 18(c) suggest an additional benefit of merging policies with different objectives, as it improves the KL-reward Pareto front.

Problem: length bias. We investigate a potential length bias in WARP [117]. LLMs after RLHF tend to be unnecessarily verbose because RMs often prefer longer generations to shorter ones, leading to this form of reward hacking. We confirm such a phenomenon in Figure 18(a), where the length of the generated text increases with higher KL values. This trend is even more pronounced in iterative WARP, where the 3rd iteration generates longer sentences than the 1st iteration at the same KL.

Mitigation strategy: length penalty. To mitigate this length bias, we integrate a length penalty (LP) into the reward: −0.0005 × len(𝑦), following [119]. From the SFT init, we launch one RL fine-tuning run with this LP, highlighted with red stars in Figure 18(b). This LP leads to significantly shorter outputs as training occurs and KL increases, in contrast to policies trained without LP.

SLERP with different configurations. Figure 18(b) also displays the lengths of generations from a SLERP merging of two policies, one trained with the LP and the other without. Critically, merging policies from diverse training configurations not only mitigates the length bias but also improves the Pareto front, as illustrated in Figure 18(c). This improvement is likely due to the increased diversity in weights and predictive mechanisms across policies, which seems beneficial for generalization, as shown in supervised learning [106].

Conclusion. Those experiments highlight the possibility to fix the length bias, and also the benefits of merging policies trained with diverse rewards, supporting previous suggestions from [109].

#### F. Diversity in predictions

0.56

0.54

0.52

Similarity

0.50

0.48

0.46

LITI of M = 2

REINFORCE

0 20 40 60 80 100 120 140

KL

- Figure 19 | Confirming diversity loss in RLHF. The 𝑥-axis is the KL compared to the SFT initialization; the 𝑦-axis is the similarity across two generations from a given policy when decoding with temperature 0.9.

Finally, we investigate the loss in diversity across generations when aligning LLMs, as reported in [65]. This could have negative consequences for creative or exploratory tasks, or even lead to policy collapse [42, 86]. In Figure 19 we plot the BLEURT similarity [115] across generations, during REINFORCE, or in LITI (as we interpolate back towards the SFT initialization). We observe that KL is strongly positively correlated with similarity across generations, confirming that RLHF induces a loss of diversity across generations. This experiment confirms that optimizing the Pareto optimality between reward and KL enables to trade-off between alignment and other benefits from pre-training, such as diversity in generations.

