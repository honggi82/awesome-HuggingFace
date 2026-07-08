## TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics

##### Shirui Chen12 Cole Harrison3 Ying-Chun Lee1 Angela Jin Yang1 Zhongzheng Ren124 Lillian J. Ratliff1 Jiafei Duan†12 Dieter Fox†12 Ranjay Krishna†12

https://topreward.github.io/webpage/

# arXiv:2602.19313v1[cs.RO]22Feb2026

### Abstract

While Vision-Language-Action (VLA) models have seen rapid progress in pretraining, their advancement in Reinforcement Learning (RL) remains hampered by low sample efficiency and sparse rewards in real-world settings. Developing generalizable process reward models is essential for providing the fine-grained feedback necessary to bridge this gap. However, existing temporal value functions often fail to generalize beyond their training domains. We introduce TOPReward, a novel, probabilistically grounded temporal value function that leverages the latent world knowledge of pretrained video Vision-Language Models (VLMs) to estimate robotic task progress. Unlike prior methods that prompt VLMs to directly output progress values—which are prone to numerical misrepresentation—TOPReward extracts task progress directly from the VLM’s internal token logits. In zero-shot evaluations across 130+ distinct real-world tasks and multiple robot platforms (e.g., Franka, YAM, SO-100/101), TOPReward achieves 0.947 mean Value-Order Correlation (VOC) on Qwen3-VL, dramatically outperforming the state-of-the-art GVL baseline which achieves near-zero correlation on the same opensource model. We further demonstrate that TOPReward serves as a versatile tool for downstream applications, including success detection and reward-aligned behavior cloning.

### 1. Introduction

Recent advances in Vision-Language-Action (VLA) models have spurred significant interest in leveraging Reinforcement Learning (RL) to achieve truly generalizable realworld performance Lei et al. (2025); Chen et al. (2025a);

†Co-advised 1University of Washington 2Allen Institute for AI 3Amazon 4University of North Carolina at Chapel Hill. Correspondence to: Shirui Chen <sc256@uw.edu>.

Preprint. February 24, 2026.

Xiao et al. (2025). However, real-world RL remains bottlenecked by the extreme sample inefficiency inherent in sparse reward signals. To bridge this gap, the research community has pivoted toward developing generalizable process reward models that provide fine-grained and dense feedback. Current efforts typically focus on directly fine-tuning visionlanguage models (VLMs) as process reward functions on curated robot datasets Duan et al. (2024); Budzianowski et al. (2025); Rocamonde et al. (2023); Ma et al. (2024); Lin et al. (2025) or training specific networks with customcollected datasets (Zhang et al., 2025; Chen et al., 2025a). For instance, RoboDopamine (Tan et al., 2025) trains a reward model on 3,400+ hours of manipulation data with step-aware multi-view perception, but requires task-specific demonstrations for adaptation. Likewise, Lee et al. (2026) introduces RoboReward, which fine-tunes a VLM on a largescale set of robot trajectories with human-provided success labels and progress scores. However, these approaches rely on non-scalable constraints: RoboDopamine requires additional demonstrations when adapting to each new task, and RoboReward reports clear gaps across different embodiments and views, indicating limited generalization guarantees. Therefore, while these efforts demonstrate promise for narrow-domain reward models, they still rely on extensive data collection and struggle to generalize beyond the training distribution.

To circumvent the high costs of task-specific fine-tuning, we investigate the use of pretrained VLMs as zero-shot reward models. Our goal is to harness the web-scale world knowledge embedded in these models to provide generalizable state-value estimations. Recent literature (Rocamonde et al., 2023; Ma et al., 2024; Baumli et al., 2023) has converged on progress estimation as a primary proxy for this value, as it provides the dense temporal signal necessary for agents to learn and adapt. Formally, progress estimation functions as a temporal value function that monotonically increases as a task nears completion, aligning it with the principles of Universal Value Learning (Schaul et al., 2015). The current state-of-the-art training-free method, GVL (Ma et al., 2024), casts progress estimation as visual question-answering—but performs well only on proprietary VLMs like Gemini and GPT-4 (Budzianowski et al., 2025), collapsing on opensource alternatives. Indeed, contemporary studies (Zhang

et al., 2026) suggest that open-source VLMs are not yet “robotics-ready” for progress estimation.

In this work, we challenge the prevailing assumption that open-source VLMs are unfit for reward modeling by introducing a novel formulation for zero-shot progress estimation. We hypothesize that the failure of current opensource models stems not from a lack of temporal understanding, but from the representation bottleneck of textual output—specifically, the models’ inconsistent instructionfollowing and their notorious bias in representing numerical tokens. To resolve this, we present TOPReward, a probabilistically grounded progress estimator that bypasses autoregressive text generation entirely. Instead of prompting the VLM to output the value of the completion percentage in text space, TOPReward extracts the model’s internal “belief” by analyzing the probabilistic distribution of its token logits. By measuring the shift in confidence toward task-completion tokens over time, we derive a continuous, well-posed progress signal directly from the VLM’s latent world knowledge. This approach requires zero additional training or fine-tuning, revealing that robust reward modeling is an emergent capability already present in pretrained video VLMs—if one looks beyond the text.

To enable rigorous evaluation of progress estimation methods, we introduce ManiRewardBench, a benchmark comprising over 130 distinct real-world manipulation tasks spanning multiple robot platforms (Franka Emika arms, Single-Arm/Bimanual YAM, SO-100/101) with temporal annotations of task progress. We demonstrate that TOPReward can effectively track task progress across this diverse benchmark. The predicted progress signal is wellcalibrated and general, enabling downstream applications such as automatic dataset ranking and filtering by task completion. We show it serves as a reliable success detector, identifying when a task has been achieved without any additional supervised training. Furthermore, we integrate TOPReward into imitation and reinforcement learning pipelines - for example, using it to weight expert examples in offline advantage-weighted behavior cloning. In realworld deployment on six single-arm SO-100 manipulation tasks, advantage-weighted fine-tuning with TOPReward consistently improves success rates over standard behavior cloning, achieving up to 10 out of 10 successes on challenging tasks where baseline behavior cloning reaches only 7 out of 10.

### 2. Related Work

The reward bottleneck for VLA. Large-scale visionlanguage-action policies such as OpenVLA (Kim et al., 2024), π0 (Black et al., 2026), MolmoAct (Lee et al., 2025) and Gemini Robotics (Team et al., 2025) have demonstrated strong language-conditioned manipulation capa-

bilities across diverse embodiments, yet reliably deploying them in real-world settings remains an open problem (Firoozi et al., 2025). A natural path forward is reinforcement learning with online or offline fine-tuning, but RL in practice hinges on the availability of a reward signalone that is traditionally hand-crafted per task in robotics, difficult to scale, and brittle under distribution shift (Kober et al., 2013; Dulac-Arnold et al., 2019). Recent efforts have applied RL to improve generalist robot policies in real-world deployment (Zhang et al., 2024; Nakamoto et al., 2023; Chen et al., 2025b; Hu et al., 2025; Ankile et al., 2025; Wagenmaker et al., 2025; Dong et al., 2025), including RL100 (Lei et al., 2025), which trains diffusion-based visuomotor policies directly on real robots using human-provided success signals, and π0∗.6 (Intelligence et al., 2025), which improves π0 through real-world RL with human-annotated episode outcomes. All of these approaches, however, remain reliant on manual reward specification, underscoring the need for automated, scalable alternatives.

Learned reward models. A long line of work seeks to replace hand-crafted rewards with learned alternatives (Pomerleau, 1991; Ho & Ermon, 2016; Hester et al., 2017). Embedding-based methods such as VIP (Ma et al., 2022), LIV (Ma et al., 2023a), and R3M (Nair et al., 2022) learn visual representations that capture progress toward a goal, but require task-specific fine-tuning and offer limited language grounding. VQA-style approaches such as SuccessVQA (Du et al., 2023) and related frameworks (Stone et al., 2023; Huang et al., 2022) reframe reward as a binary classification problem—asking a VLM whether the task succeeded—but produce signals too coarse for dense reward shaping (Lynch et al., 2020; Andrychowicz et al., 2017). Code-generation strategies like Eureka (Ma et al., 2023b) synthesize reward functions via LLMs, yet depend on access to simulation ground truth. More recently, generalist reward models such as RoboReward (Lee et al., 2026) and RoboDopamine (Tan et al., 2025) are trained on largescale datasets of successes and failures to predict progress scores or distance-to-goal estimates (Wu et al., 2023; Fan et al., 2022). While these models move toward broader coverage, they still require domain-specific training data and can struggle to generalize across embodiments and environments (Kalashnikov et al., 2021). A common thread across all these approaches is the dependence on training or domain-specific resources—a requirement our method eliminates entirely.

Training-free value estimation with VLMs. A separate and more relevant line of work asks whether VLMs can estimate task progress without any additional training. Generative Value Learning (GVL) (Ma et al., 2024) poses progress prediction as a temporal ordering problem: given a batch of shuffled trajectory frames, the VLM is prompted to assign per-frame progress scores, exploiting its semantic ground-

Temporal Visual Progress Estimator

###### TOPReward

Log Probability

VLM

[Figure 1]

[Figure 2]

[Figure 3]

video tokens Prompt

"true" token

ManiRewardBench

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

t

###### t

t

1

1:t

1:T

[Figure 12]

[Figure 13]

[Figure 14]

Success Detection

GVL

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Reward-Aligned Learning

[Figure 25]

[Figure 26]

t

Figure 1. Result highlights. TOPReward enables effective zero-shot estimation of task progress across diverse and challenging realworld manipulation tasks, without task-specific training. By bootstrapping on a range of vision–language model backbones, TOPReward provides a temporally consistent visual reward signal that supports multiple downstream applications, including success detection, policy improvement, and evaluation on our in-house benchmark, ManiRewardBench.

ing to rank frames by task completion. This enables various downstream applications such as dataset filtering and advantage-weighted regression without task-specific reward engineering. The OpenGVL benchmark (Budzianowski et al., 2025) evaluates this paradigm across diverse tasks and model families, revealing that open-source VLMs fall substantially short of their proprietary counterparts on temporal progress prediction. In this paper, we hypothesize that the reason for this is not a lack of visual understanding in open-source VLMs but rather the instability of numeric token generation—LLMs are poorly calibrated when asked to produce precise numerical outputs (Wallace et al., 2019; Yuan et al., 2023). This observation motivates a fundamental shift: rather than asking a VLM to generate a progress value, one can instead probe what the model already knows through its internal representations.

Internal representations as reward signals. A growing body of work in NLP has shown that a model’s internal activations—logits, hidden states, and embeddings—track its certainty and factual accuracy more reliably than its generated text (Kadavath et al., 2022; Tian et al., 2023; Azaria & Mitchell, 2023; Burns et al., 2022; Liu et al., 2023). In robotics, recent methods have begun to lever-

age such representations for reward definition (Rocamonde et al., 2023; Grislain et al., 2025), bypassing the instabilities of text generation. Our work, TOPReward, takes this principle further: instead of prompting a VLM to generate numeric progress estimates, we pose a binary completion query (“does this trajectory complete the task?”) and extract the probability of the affirmative token as a continuous reward signal. This formulation is zero-shot, requires no finetuning or domain-specific data, and yields a well-calibrated progress estimate that scales to over 130 real-world tasks in ManiRewardBench as well as the Open X-Embodiment dataset across multiple robot platforms.

### 3. TOPReward

The stark difference between GVL’s (Ma et al., 2024) performance on Gemini versus open-source models might mislead one into believing that only the most powerful VLM can accurately estimate the progress of a robotic trajectory. Indeed, for a model to output well-formatted and accurate progress estimates for all shuffled frames, it needs strong instructionfollowing capability and an accurate internal representation of numerical values. However, neither of these is necessarily

correlated with the model’s true underlying video understanding capability. Therefore, we propose TOPReward, a method that leverages the internal understanding of the VLM to produce a progress estimator without requiring accurate, well-formatted numerical generation.

Problem setup. We formulate the progress estimation task as follows: given an instruction x and a video trajectory τ1:T = (I1,...,IT) (frames in chronological order), our goal is to produce a scalar progress signal pt ∈ [0,1] for each prefix τ1:t that increases as the task is completed.

##### 3.1. Token probability as the reward

Key idea. We use the VLM’s internal output, i.e., predicted token probabilities as the reward. Concretely, we ask the model to judge whether the observed trajectory completes the instruction and score the probability of an affirmative answer (e.g. the token True).

Let pθ be a VLM defining a next-token distribution with pretrained weights θ. We form a prompt u that grounds the judgment in the video:

“<|video|> The above video shows a robot manipulation trajectory that completes the following task: {INSTRUCTION}. Decide whether the above statement is True or not. The answer is: {a}”

and compute the probability of the answer token sequence a=“True” 1. We choose True because we found boolean tokens to show the clearest success–failure separation, with True exhibiting the largest absolute difference in mean token probability across episodes (see Section B for the tokenprobability comparison). Denoting the (video-conditioned) textual context by c(τ1:t,u), we define the reward for a prefix to be

rt = log pθ(a | c(τ1:t,u)). (1)

In this way, we verbally construct the conditional probability of task completion given the trajectory-instruction pair. This procedure entirely sidesteps the need for the language model to rely on its instruction-following or numerical generation capabilities. When t = T, i.e., when τ1:T is the entire trajectory, this formulation guarantees that any VLM capable of video understanding can give a high reward close to 0. As we will see in Section 5, rt will also increase as t gets larger, and the visual evidence in τ1:t makes the completion statement more plausible.

Chat templates. We do not use any chat template in our experiments. The ablation study in Section 5.4 shows that

1We also explore another variant that evaluates the probability over the entire instruction which consists of multiple tokens, but found it to be less effective. See Section A for details.

[Figure 27]

Figure 2. Qualitative example of “Fold the Towel”: Instructionconditioned progress estimation on a real trajectory. The curve shows TOPReward’s predicted completion value over time, with annotated values at selected frames corresponding to semantic subtasks.

adding a chat template substantially degrades performance. We hypothesize this is because progress estimation is better aligned with the pretraining objective of next-token prediction.

##### 3.2. Progress estimation from trajectory prefixes

Prefix sampling. To obtain a temporal progress curve, we evaluate Eq. (1) on a set of K uniformly spaced prefix lengths {tk}Kk=1 with 1 = t1 < ··· < tK = T. This involves K model forwards and produces rewards {rt

k} summarizing how completion evidence accumulates over time (Figure 2).

Normalization. Usually a progress estimate implies a number that is between 0 and 1, but the range of log probability is (−∞,0]. We therefore use min-max normalization to map rewards to a normalized progress score st

within each episode:

k

st

k

=

k − minj rt

rt

, (2)

j

j − minj rt

maxj rt

+ ε

j

with a small ε for numerical stability. This yields a wellposed progress estimate in [0,1] that is comparable across time within a trajectory.

Dense rewards for downstream use. When a per-step reward is needed (e.g., for reward-aligned behavior cloning), we use the progress increment to construct such a signal:

= clip(τ ·exp(st

k −st

∆t

),min = 0,max = δmax),

k−1

k

(3) where τ is a scaling factor to control how different the weight for good action and bad action should be, and δmax is a maximum allowed reward to prevent the model from only focusing on a subset of actions with excessively large weights.

### 4. ManiRewardBench: a benchmark for reward modeling in robotic manipulation

Existing robotics reward benchmarks remain limited in scope and do not fully stress-test reward models on realworld manipulation trajectories. We introduce Mani RewardBench, a reward-model benchmark designed to evaluate progress sensitivity, completion detection, and cross-embodiment robustness in real-world manipulation. See Section D for more details of this benchmark.

Scope and diversity. The benchmark contains 130 unique manipulation tasks spanning everyday object interactions. Data are collected across four robot platforms (Franka, SO-100/101, single-arm YAM, and bimanual YAM), enabling evaluation under embodiment shifts. We provide subtask-level temporal annotations (start/end seconds) for each episode, which makes it possible to probe fine-grained progress understanding rather than coarse success/failure, and to evaluate reward responses at subtask boundaries.

Stage-aware annotation. For each task in the benchmark, episodes are manually labeled and segmented into a sequence of predefined subtasks. Each task is associated with an ordered list of subtasks that represent stages of execution (e.g., reaching for an object, grasping it, or placing it). This fine-grained annotation allows for better evaluation of whether a reward model produces accurate progress estimates, as we apply a stage-aware evaluation to our pipeline.

Failure trajectories. We additionally include a dataset of 23 tasks with 156 episodes in total, containing both successful and failed attempts. We evaluate our method on this dataset for success detection, as detailed in Section 5.2. This ensures that the method evaluated on our benchmark is consistent across both failed and successful trajectories.

Access. To ensure evaluation integrity and prevent data leakage, the underlying dataset will remain restricted. Access is available exclusively through a controlled evaluation protocol.

### 5. Experiments

We evaluate TOPReward across three main dimensions: (1) zero-shot progress estimation on large-scale robot datasets

- (Section 5.1); (2) success detection on failed trajectories
- (Section 5.2), demonstrating that our probability-based approach overcomes a fundamental limitation of VOC-based methods; and (3) real-world deployment for advantageweighted behavior cloning, showing consistent improvements over standard imitation learning. We conclude with ablation studies examining the effect of chat templates on model performance (Section 5.4).

VLM backbones. We evaluate TOPReward on three video-language models: Qwen3-VL-8B (Bai et al., 2025),

[Figure 28]

Figure 3. VOC comparison across datasets. Mean dataset-level VOC for GVL (0-shot) and TOPReward across two evaluation sets: OXE (39 datasets, 20 episodes each) and ManiReward Bench (4 datasets, 113 tasks, 497 episodes). Error bars denote standard deviation across datasets within each evaluation set.

Molmo2 (Clark et al., 2026), and Gemini-2.5-Pro (et al., 2025). We select Qwen3-VL and Molmo2 as representative open-source models with strong video understanding capabilities, enabling reproducible research without proprietary API dependencies. Gemini-2.5-Pro serves as our proprietary baseline, as it provides access to logit distributions required by our method. This allows us to assess whether open-source models can match or exceed closed-source performance on progress estimation tasks directly from logits.

##### 5.1. Large-scale real-world evaluation

To evaluate the zero-shot progress estimation capability of TOPReward, we evaluate the VOC on two sets of large expert robotic trajectories. We mainly compare against GVL (Budzianowski et al., 2025; Ma et al., 2024), the stateof-the-art training-free progress estimator. GVL predicts the task progress by shuffling the frames and prompting the language model to give structured output that assigns numerical progress from 0 to 1 to each frame.

Metrics. Following standard practice (Ma et al., 2024; Lee et al., 2026; Ma et al., 2023a), we use Value-Order Correlation (VOC) to measure Spearman’s rank correlation between the chronological order of input video frames and the predicted values,

VOC = rank-correlation argsort(st

,··· ,st

), (t1,t2,··· ,tK) .

,st

1

2

K

(4)

VOC ranges from −1 to 1. When VOC equals −1, the predicted order is exactly the opposite of the ground truth, and VOC = 1 indicates perfect alignment.

Results on Open X-Embodiment. The Open XEmbodiment (OXE) dataset (O’Neill et al., 2024) is a collection of 50 academic robot datasets spanning diverse tasks, camera configurations, and robot platforms. We use the LeRobot collection from OXE and select a subset of 39 datasets. For each dataset, we randomly sample 20 episodes and evaluate the performance of TOPReward. Results are

[Figure 29]

- Figure 4. Progress traces for ManiRewardBench. Example progress traces predicted by TOPReward (orange) compared to stageaware ground-truth completion (dashed) from ManiRewardBench, computed from annotated subtask boundaries. We also overlay Gemini-GVL (blue) on the same episodes when available.

- Table 1. Results on the Open X-Embodiment dataset. We report mean VOC over 39 datasets and 20 episodes each. Higher is better.

Method Molmo2-8B Qwen3-VL-8B Gemini-2.5-Pro

GVL -0.016 0.194 0.541 TOPReward 0.417 0.857 0.433

shown in Table 1. On open-source models, TOPReward substantially outperforms GVL: Qwen3-VL-8B achieves 0.857 VOC (vs. 0.194 for GVL, a +0.663 improvement), and Molmo2-8B reaches 0.417 (vs. −0.016 for GVL, a +0.433 improvement). On the proprietary Gemini-2.5-Pro, GVL performs better (0.541), while TOPReward achieves 0.433. This reversal on Gemini reflects an issue with using the chat template, a detail later in Section 5.4.

Results on ManiRewardBench. We evaluate on successful trajectories across 4 robotic platforms, totaling 113 tasks and 497 episodes. Results are shown in Figure 3 and Table 2. On Qwen3-VL-8B, TOPReward achieves remarkably consistent performance across all four datasets (0.942–0.954 VOC), dramatically outperforming GVL which struggles with near-zero or negative VOC on Molmo2-8B and shows inconsistent performance (0.164–0.544) on other models. This demonstrates that TOPReward’s logit-based approach successfully utilizes implicit progress estimation capabilities of open-source VLMs, whereas GVL’s text-generation formulation fails to leverage their pretrained video understanding. The detailed per-task breakdowns and distribution plots are provided in Section C.

Qualitative results. We visualize representative progress traces in Figure 4. The traces demonstrate that TOPReward produces smooth, monotonically increasing progress signals that closely track stage-aware ground-truth task completion (computed from annotated subtask boundaries) across diverse manipulation tasks. In contrast, Gemini-GVL (when available) exhibits noisier predictions with frequent nonmonotonic fluctuations. Notably, TOPReward correctly captures the temporal structure of multi-step tasks, with

progress plateaus corresponding to intermediate subtask completions and accelerations during active manipulation phases.

##### 5.2. Success detection

A failure mode of the VOC metric. GVL instructs the VLM to output a progress score from 0 to 1 for each input frame. Nevertheless, GVL does not rely on the maximum output progress as an indicator of successful trajectories. Rather, they argue that failed trajectories can be difficult for the model to reshuffle, leading to low VOC scores. We observe that failure trajectories in ManiRewardBench exhibit a pattern where the robot makes progress early and subsequently moves randomly in later stages. The groundtruth progress in this case should increase and then plateau at a value less than one, since random movement does not necessarily undo progress. Since VOC measures rank correlation (i.e., the ordering of predictions), not the absolute completion level, a trajectory that plateaus early—for example, stalling at 30% completion—can still achieve a high VOC because predictions remain well-ordered. Figure 5 illustrates this: synthetic trajectories plateauing at 80%, 50%, and 30% completion all achieve VOC ≥ 0.85. Empirically, on the failure split of ManiRewardBench, the mean VOC with our method is virtually identical for failed and successful trajectories (0.946 vs. 0.943). In contrast, TOPReward directly measures the probability of the instruction being satisfied, so failed trajectories naturally receive lower scores.

Results. We evaluate success detection on the failure trajectory split of ManiRewardBench, which contains both successful and failed attempts across 23 tasks. We frame success detection as binary classification and report ROCAUC. For TOPReward, we use the average log probability over the last 3 sampled frames; for GVL, we use the VOC score.

Results are reported in Table 3. We observe that for the open-source Qwen3-VL-8B model, GVL’s ROC-AUC is

- Table 2. Results on ManiRewardBench. We report mean VOC over 113 tasks and 497 episodes. Higher is better. ∗Note that the Gemini API forces a chat template, which negatively affects our method as detailed in Section 5.4.

Molmo2-8B Qwen3-VL-8B Gemini-2.5-Pro∗ Dataset GVL TOPReward GVL TOPReward GVL TOPReward

Lerobot -0.001 0.595 0.332 0.954 0.620 0.578 Franka 0.000 0.662 0.242 0.942 0.695 0.448 Bimanual YAM 0.007 0.565 0.164 0.947 0.566 0.546 Single-arm YAM -0.017 0.642 0.544 0.945 0.752 0.488

[Figure 30]

Figure 5. Illustrative example of the VOC failure mode. Because VOC depends only on the rank order of predicted values (not the absolute completion level), trajectories that rise and then plateau at different final completion levels can all score highly (≥ 0.85). As a result, VOC may not distinguish a well-ordered but incomplete (early-plateau) trajectory from a complete trajectory.

- Table 3. Success detection results. We report ROC-AUC on ManiRewardBench. TOPReward matches or exceeds GVL across both open-source and proprietary models.

Peters & Schaal, 2007). We start from a base policy π0 pretrained on 200 hours of the publicly available single-arm SO-100 dataset (HuggingFace). For each of six real-world tasks, we collect 50 demonstrations (potentially noisy and suboptimal) and use TOPReward to compute the value of each state-action pair. We convert these values to advantages by subtracting the dataset mean, then fine-tune π0 with the AWR objective:

t|a) ∆t · ∥vθ(at,t | o) − (a − ϵ)∥2 ,

LAWR = Ep(a|o),q(a

(5)

where at = (1−t)ϵ+ta, t ∼ U(0,1), ϵ ∼ N(0,I), and ∆t is the advantage computed in Equation (3) (we use τ = 2.0 and δmax = 2.0). We compare AWR performance against a behavior cloning (BC) baseline that directly minimizes the unweighted flow matching loss on the same dataset. Results are shown in Table 4. We measure partial success as the fraction of predefined subtasks completed per trial, summed over 10 trials (maximum score 10). We observe that AWR consistently outperforms BC across all six tasks, demonstrating the effectiveness of TOPReward in real-world robot learning. The configurations of six evaluation tasks are shown in Figure 6, and a qualitative rollout comparison is provided in Figure 7.

###### Method Qwen3-VL-8B Gemini-2.5-Pro

GVL 0.519 0.823 TOPReward 0.654 0.826

Table 4. Real-world experiments. We report partial success score out of 10 trials (fraction of subtasks completed, summed over trials) for advantage-weighted behavior cloning on single-arm SO-100 tasks.

essentially random (0.519) for success detection, while TOPReward achieves 0.654—a +0.135 AUC improvement. On the proprietary Gemini model, both methods perform comparably (0.823 vs. 0.826), suggesting that the VOC failure mode is most pronounced when the underlying VLM already struggles with calibrated progress estimation. Across both model classes, TOPReward matches or exceeds GVL without requiring structured numerical generation.

Task Pretrained BC TOP-AWR (Ours)

Place toy car in box 1 2 3 Stack red cube on green cube 1.33 1 2.33 Put pen into cup 1.67 5.67 6.33 Place doll in box 0 7 10 Pick up cube 4 7 10 Put cube in cup 4 6 9

##### 5.4. Ablation

##### 5.3. Real-world advantage-weighted behavior cloning

To further showcase TOPReward as a signal for policy improvement, we use it to compute advantage weights for advantage-weighted regression (AWR) (Peng et al., 2019;

In this section, we explore why our method performs worse with Gemini. We hypothesize that this is because the Gemini API enforces a chat template on our prompt. To verify this hypothesis, we wrap the prompt in Section 3.1 with the chat template and evaluate the probability of the answer being

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

(e) Stack red cube on green cube

(a) Put the doll in box (b) Put toy car in box (d) Put the cube in cup

(c) Pick up the cube (f) Put pen into the cup

Figure 6. The six real-world single-arm SO-100 manipulation tasks used for advantage-weighted behavior cloning evaluation.

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

|[Figure 40]|
|---|

|[Figure 41]|
|---|

|[Figure 42]|
|---|

Pretrained

|[Figure 43]|
|---|

|[Figure 44]|
|---|

|[Figure 45]|
|---|

|[Figure 46]|
|---|

|[Figure 47]|
|---|

|[Figure 48]|
|---|

BC

|[Figure 49]|
|---|

|[Figure 50]|
|---|

|[Figure 51]|
|---|

|[Figure 52]|
|---|

|[Figure 53]|
|---|

|[Figure 54]|
|---|

###### TOP-AWR (Ours)

Time

- Figure 7. Qualitative comparison on “Place doll in box.” The pretrained policy and behavior cloning (BC) both fail, while TOP-AWR, fine-tuned with advantage weights from TOPReward, succeeds consistently. Frames are uniformly sampled from evaluation rollouts.

- Table 5. Effect of Chat Template on TOPReward VOC. Chat template degrades Qwen3-VL-8B performance by nearly 50% and Molmo2-8B by 20%, demonstrating that the logit-based formulation is sensitive to prompt formatting.

###### Qwen3-VL-8B Molmo2-8B

Dataset Base +Chat Base +Chat Bimanual YAM 0.947 0.269 0.570 0.408 Franka 0.943 0.528 0.696 0.615 Single-arm YAM 0.946 0.703 0.691 0.546 Mean 0.945 0.500 0.652 0.523 ∆ −47.1% −19.8%

True. We summarize the results in Table 5. Indeed, we see a significant drop in VOC score for both Qwen3-VL-8B and Molmo2-8B.

### 6. Conclusion

We presented TOPReward, a zero-shot progress estimator that repurposes the token probabilities of pretrained video VLMs as temporal value functions for robotic manipulation. By querying the model’s internal belief about instruction completion rather than requiring it to generate calibrated numerical outputs, TOPReward sidesteps the

well-known limitations of VLMs in numerical reasoning and instruction following. On the Open X-Embodiment dataset (39 datasets, 780 episodes) and our newly introduced ManiRewardBench benchmark (113 tasks, 497 episodes across four robot platforms), TOPReward with the opensource Qwen3-VL-8B achieves a mean VOC of 0.857 and 0.947 respectively, substantially outperforming GVL. We further showed that TOPReward naturally supports success detection—where VOC-based methods degrade to chancelevel performance on open-source models—and that the progress signal can be used for advantage-weighted behavior cloning, yielding consistent improvements over standard BC across six real-world SO-100 manipulation tasks.

Limitations. Our method inherits the visual perception limitations of the underlying VLM: tasks requiring finegrained spatial reasoning (e.g., precise alignment or small object manipulation) may receive noisy progress estimates when the model cannot visually distinguish intermediate states. The min-max normalization in Equation (2) is performed per-episode, which prevents direct comparison of absolute progress values across different trajectories without additional calibration. Nevertheless, our success detection experiments show that the probability of the entire trajectory enables quality comparison across episodes. Finally, while TOPReward works well with current open-source

video VLMs, its performance is bounded by the video understanding capabilities of the backbone model. Future improvements in video VLMs should directly translate to better progress estimation.

### 7. Impact Statements

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

### References

Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J., Pieter Abbeel, O., and Zaremba, W. Hindsight experience replay. Advances in neural information processing systems, 30, 2017.

Burns, C., Ye, H., Klein, D., and Steinhardt, J. Discovering latent knowledge in language models without supervision. arXiv preprint arXiv:2212.03827, 2022.

Chen, Q., Yu, J., Schwager, M., Abbeel, P., Shentu, Y., and Wu, P. Sarm: Stage-aware reward modeling for long horizon robot manipulation. arXiv preprint arXiv:2509.25358, 2025a.

Chen, Y., Tian, S., Liu, S., Zhou, Y., Li, H., and Zhao, D. Conrft: A reinforced fine-tuning method for vla models via consistency policy. arXiv preprint arXiv:2502.05450, 2025b.

Ankile, L., Simeonov, A., Shenfeld, I., Torne, M., and Agrawal, P. From imitation to refinement-residual rl for precise assembly. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 01–08. IEEE, 2025.

Azaria, A. and Mitchell, T. The internal state of an llm knows when it’s lying. arXiv preprint arXiv:2304.13734, 2023.

Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-vl technical report, 2025. URL https://arxiv.org/abs/2511.21631.

Baumli, K., Baveja, S., Behbahani, F., Chan, H., Comanici, G., Flennerhag, S., Gazeau, M., Holsheimer, K., Horgan, D., Laskin, M., et al. Vision-language models as a source of rewards. arXiv preprint arXiv:2312.09187, 2023.

Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., Jakubczak, S., Jones, T., Ke, L., Levine, S., LiBell, A., Mothukuri, M., Nair, S., Pertsch, K., Shi, L. X., Tanner, J., Vuong, Q., Walling, A., Wang, H., and Zhilinsky, U. π0: A vision-language-action flow model for general robot control, 2026. URL https:

//arxiv.org/abs/2410.24164.

Budzianowski, P., Wi´snios, E., G´oral, G., Kulakov, I., Petrenko, V., and Walas, K. Opengvl–benchmarking visual temporal progress for data curation. arXiv preprint arXiv:2509.17321, 2025.

Clark, C., Zhang, J., Ma, Z., Park, J. S., Salehi, M., Tripathi, R., Lee, S., Ren, Z., Kim, C. D., Yang, Y., Shao, V., Yang, Y., Huang, W., Gao, Z., Anderson, T., Zhang, J., Jain, J., Stoica, G., Han, W., Farhadi, A., and Krishna, R. Molmo2: Open weights and data for vision-language models with video understanding and grounding, 2026. URL https://arxiv.org/abs/2601.10611.

Dong, P., Mirchandani, S., Sadigh, D., and Finn, C. What matters for batch online reinforcement learning in robotics? arXiv preprint arXiv:2505.08078, 2025.

Du, Y., Konyushkova, K., Denil, M., Raju, A., Landon, J., Hill, F., De Freitas, N., and Cabi, S. Vision-language models as success detectors. arXiv preprint arXiv:2303.07280, 2023.

Duan, J., Pumacay, W., Kumar, N., Wang, Y. R., Tian, S., Yuan, W., Krishna, R., Fox, D., Mandlekar, A., and Guo, Y. Aha: A vision-language-model for detecting and reasoning over failures in robotic manipulation. arXiv preprint arXiv:2410.00371, 2024.

Dulac-Arnold, G., Mankowitz, D., and Hester, T. Challenges of real-world reinforcement learning. arXiv preprint arXiv:1904.12901, 2019.

et al., G. C. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025. URL https: //arxiv.org/abs/2507.06261.

Fan, L., Wang, G., Jiang, Y., Mandlekar, A., Yang, Y., Zhu, H., Tang, A., Huang, D.-A., Zhu, Y., and Anandkumar, A. Minedojo: Building open-ended embodied agents with internet-scale knowledge. Advances in Neural Information Processing Systems, 35:18343–18362, 2022.

Firoozi, R., Tucker, J., Tian, S., Majumdar, A., Sun, J., Liu, W., Zhu, Y., Song, S., Kapoor, A., Hausman, K., et al. Foundation models in robotics: Applications, challenges, and the future. The International Journal of Robotics Research, 44(5):701–739, 2025.

Grislain, C., Rahimi, H., Sigaud, O., and Chetouani, M. I-failsense: Towards general robotic failure detection with vision-language models. arXiv preprint arXiv:2509.16072, 2025.

Hester, T., Vecer´ık, M., Pietquin, O., Lanctot, M., Schaul, T., Piot, B., Sendonaris, A., Dulac-Arnold, G., Osband, I., Agapiou, J. P., Leibo, J. Z., and Gruslys, A. Learning from demonstrations for real world reinforcement learning. ArXiv, abs/1704.03732, 2017. URL https://api.

semanticscholar.org/CorpusID:15254659.

Ho, J. and Ermon, S. Generative adversarial imitation learning. Advances in neural information processing systems, 29, 2016.

Hu, J., Hendrix, R., Farhadi, A., Kembhavi, A., Mart´ınMart´ın, R., Stone, P., Zeng, K.-H., and Ehsani, K. Flare: Achieving masterful and adaptive robot policies with large-scale reinforcement learning fine-tuning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 3617–3624. IEEE, 2025.

Huang, W., Xia, F., Xiao, T., Chan, H., Liang, J., Florence, P., Zeng, A., Tompson, J., Mordatch, I., Chebotar, Y., et al. Inner monologue: Embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608, 2022.

Intelligence, P., Amin, A., Aniceto, R., Balakrishna, A., Black, K., Conley, K., Connors, G., Darpinian, J., Dhabalia, K., DiCarlo, J., Driess, D., Equi, M., Esmail, A., Fang, Y., Finn, C., Glossop, C., Godden, T., Goryachev, I., Groom, L., Hancock, H., Hausman, K., Hussein, G., Ichter, B., Jakubczak, S., Jen, R., Jones, T., Katz, B., Ke, L., Kuchi, C., Lamb, M., LeBlanc, D., Levine, S., Li-Bell, A., Lu, Y., Mano, V., Mothukuri, M., Nair, S., Pertsch, K., Ren, A. Z., Sharma, C., Shi, L. X., Smith, L., Springenberg, J. T., Stachowicz, K., Stoeckle, W., Swerdlow, A., Tanner, J., Torne, M., Vuong, Q., Walling, A., Wang, H., Williams, B., Yoo, S., Yu, L., Zhilinsky, U., and Zhou, Z. π0∗.6: a vla that learns from experience, 2025. URL https://arxiv.org/abs/2511.14759.

Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma, N., Tran-Johnson, E., et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

Kalashnikov, D., Varley, J., Chebotar, Y., Swanson, B., Jonschkowski, R., Finn, C., Levine, S., and Hausman, K. Mt-opt: Continuous multi-task robotic reinforcement learning at scale. arXiv preprint arXiv:2104.08212, 2021.

Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G.,

Sanketi, P., Vuong, Q., Kollar, T., Burchfiel, B., Tedrake, R., Sadigh, D., Levine, S., Liang, P., and Finn, C. Openvla: An open-source vision-language-action model, 2024. URL https://arxiv.org/abs/2406.09246.

Kober, J., Bagnell, J. A., and Peters, J. Reinforcement learning in robotics: A survey. The International Journal of Robotics Research, 32(11):1238–1274, 2013.

Lee, J., Duan, J., Fang, H., Deng, Y., Liu, S., Li, B., Fang, B., Zhang, J., Wang, Y. R., Lee, S., et al. Molmoact: Action reasoning models that can reason in space. arXiv preprint arXiv:2508.07917, 2025.

Lee, T., Wagenmaker, A., Pertsch, K., Liang, P., Levine, S., and Finn, C. Roboreward: General-purpose visionlanguage reward models for robotics. arXiv preprint arXiv:2601.00675, 2026.

Lei, K., Li, H., Yu, D., Wei, Z., Guo, L., Jiang, Z., Wang, Z., Liang, S., and Xu, H. Rl-100: Performant robotic manipulation with real-world reinforcement learning, 2025. URL https://arxiv.org/abs/2510.14830.

Lin, Z., Duan, J., Fang, H., Fox, D., Krishna, R., Tan, C., and Wen, B. Failsafe: Reasoning and recovery from failures in vision-language-action models. arXiv preprint arXiv:2510.01642, 2025.

Liu, K., Casper, S., Hadfield-Menell, D., and Andreas, J. Cognitive dissonance: Why do language model outputs disagree with internal representations of truthfulness? arXiv preprint arXiv:2312.03729, 2023.

Lynch, C., Khansari, M., Xiao, T., Kumar, V., Tompson, J., Levine, S., and Sermanet, P. Learning latent plans from play. In Conference on robot learning, pp. 1113–1132. Pmlr, 2020.

Ma, Y. J., Sodhani, S., Jayaraman, D., Bastani, O., Kumar, V., and Zhang, A. Vip: Towards universal visual reward and representation via value-implicit pre-training. arXiv preprint arXiv:2210.00030, 2022.

Ma, Y. J., Kumar, V., Zhang, A., Bastani, O., and Jayaraman, D. Liv: Language-image representations and rewards for robotic control. In International Conference on Machine Learning, pp. 23301–23320. PMLR, 2023a.

Ma, Y. J., Liang, W., Wang, G., Huang, D.-A., Bastani, O., Jayaraman, D., Zhu, Y., Fan, L., and Anandkumar, A. Eureka: Human-level reward design via coding large language models. arXiv preprint arXiv:2310.12931, 2023b.

Ma, Y. J., Hejna, J., Fu, C., Shah, D., Liang, J., Xu, Z., Kirmani, S., Xu, P., Driess, D., Xiao, T., et al. Vision language models are in-context value learners. In The Thirteenth International Conference on Learning Representations, 2024.

Nair, S., Rajeswaran, A., Kumar, V., Finn, C., and Gupta, A. R3m: A universal visual representation for robot manipulation. arXiv preprint arXiv:2203.12601, 2022.

Nakamoto, M., Zhai, S., Singh, A., Sobol Mark, M., Ma, Y., Finn, C., Kumar, A., and Levine, S. Cal-ql: Calibrated offline rl pre-training for efficient online fine-tuning. Advances in Neural Information Processing Systems, 36: 62244–62269, 2023.

O’Neill, A., Rehman, A., Maddukuri, A., Gupta, A., Padalkar, A., Lee, A., Pooley, A., Gupta, A., Mandlekar, A., Jain, A., et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 6892–6903. IEEE, 2024.

Peng, X. B., Kumar, A., Zhang, G., and Levine, S. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.

Peters, J. and Schaal, S. Reinforcement learning by rewardweighted regression for operational space control. In Proceedings of the 24th international conference on Machine learning, pp. 745–750, 2007.

Pomerleau, D. A. Efficient training of artificial neural networks for autonomous navigation. Neural computation, 3(1):88–97, 1991.

Rocamonde, J., Montesinos, V., Nava, E., Perez, E., and Lindner, D. Vision-language models are zero-shot reward models for reinforcement learning. arXiv preprint arXiv:2310.12921, 2023.

Schaul, T., Horgan, D., Gregor, K., and Silver, D. Universal value function approximators. In International conference on machine learning, pp. 1312–1320. PMLR, 2015.

Stone, A., Xiao, T., Lu, Y., Gopalakrishnan, K., Lee, K.H., Vuong, Q., Wohlhart, P., Kirmani, S., Zitkovich, B., Xia, F., et al. Open-world object manipulation using pre-trained vision-language models. arXiv preprint arXiv:2303.00905, 2023.

Tan, H., Chen, S., Xu, Y., Wang, Z., Ji, Y., Chi, C., Lyu, Y., Zhao, Z., Chen, X., Co, P., et al. Robo-dopamine: General process reward modeling for high-precision robotic manipulation. arXiv preprint arXiv:2512.23703, 2025.

Team, G. R., Abeyruwan, S., Ainslie, J., Alayrac, J.-B., Arenas, M. G., Armstrong, T., Balakrishna, A., Baruch, R., Bauza, M., Blokzijl, M., Bohez, S., Bousmalis, K., Brohan, A., Buschmann, T., Byravan, A., Cabi, S., Caluwaerts, K., Casarini, F., Chang, O., Chen, J. E., Chen, X., Chiang, H.-T. L., Choromanski, K., D’Ambrosio, D.,

Dasari, S., Davchev, T., Devin, C., Palo, N. D., Ding, T., Dostmohamed, A., Driess, D., Du, Y., Dwibedi, D., Elabd,

- M., Fantacci, C., Fong, C., Frey, E., Fu, C., Giustina, M., Gopalakrishnan, K., Graesser, L., Hasenclever, L., Heess,
- N., Hernaez, B., Herzog, A., Hofer, R. A., Humplik, J., Iscen, A., Jacob, M. G., Jain, D., Julian, R., Kalashnikov, D., Karagozler, M. E., Karp, S., Kew, C., Kirkland, J., Kirmani, S., Kuang, Y., Lampe, T., Laurens, A., Leal,

I., Lee, A. X., Lee, T.-W. E., Liang, J., Lin, Y., Maddineni, S., Majumdar, A., Michaely, A. H., Moreno, R., Neunert, M., Nori, F., Parada, C., Parisotto, E., Pastor, P., Pooley, A., Rao, K., Reymann, K., Sadigh, D., Saliceti, S., Sanketi, P., Sermanet, P., Shah, D., Sharma, M., Shea, K., Shu, C., Sindhwani, V., Singh, S., Soricut, R., Springenberg, J. T., Sterneck, R., Surdulescu, R., Tan, J., Tompson, J., Vanhoucke, V., Varley, J., Vesom, G., Vezzani, G., Vinyals, O., Wahid, A., Welker, S., Wohlhart, P., Xia, F., Xiao, T., Xie, A., Xie, J., Xu, P., Xu, S., Xu, Y., Xu, Z., Yang, Y., Yao, R., Yaroshenko, S., Yu, W., Yuan, W., Zhang, J., Zhang, T., Zhou, A., and Zhou, Y. Gemini robotics: Bringing ai into the physical world, 2025. URL https://arxiv.org/abs/2503.20020.

Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., and Manning, C. D. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. arXiv preprint arXiv:2305.14975, 2023.

Wagenmaker, A., Nakamoto, M., Zhang, Y., Park, S., Yagoub, W., Nagabandi, A., Gupta, A., and Levine, S. Steering your diffusion policy with latent space reinforcement learning. arXiv preprint arXiv:2506.15799, 2025.

Wallace, E., Wang, Y., Li, S., Singh, S., and Gardner, M. Do nlp models know numbers? probing numeracy in embeddings. arXiv preprint arXiv:1909.07940, 2019.

Wu, H., Jing, Y., Cheang, C., Chen, G., Xu, J., Li, X., Liu, M., Li, H., and Kong, T. Unleashing large-scale video generative pre-training for visual robot manipulation. arXiv preprint arXiv:2312.13139, 2023.

Xiao, W., Lin, H., Peng, A., Xue, H., He, T., Xie, Y., Hu, F., Wu, J., Luo, Z., Fan, L., et al. Self-improving visionlanguage-action models with data generation via residual rl. arXiv preprint arXiv:2511.00091, 2025.

Yuan, Z., Yuan, H., Tan, C., Wang, W., and Huang, S. How well do large language models perform in arithmetic tasks? arXiv preprint arXiv:2304.02015, 2023.

Zhang, J., Heo, M., Liu, Z., Biyik, E., Lim, J. J., Liu, Y., and Fakoor, R. Extract: Efficient policy learning by extracting transferable robot skills from offline data. arXiv preprint arXiv:2406.17768, 2024.

Zhang, J., Luo, Y., Anwar, A., Sontakke, S. A., Lim, J. J., Thomason, J., Biyik, E., and Zhang, J. Rewind: Language-guided rewards teach robot policies without new demonstrations. arXiv preprint arXiv:2505.10911,

- 2025.

Zhang, J., Qian, C., Sun, H., Lu, H., Wang, D., Xue, L., and Liu, H. Progresslm: Towards progress reasoning in visionlanguage models. arXiv preprint arXiv:2601.15224,

- 2026.

### A. Alternative Reward Formulation

In addition to the main formulation of TOPReward presented in Section 3, we also experimented with an alternative reward formulation that evaluates the probability of generating the entire instruction given the video trajectory. Specifically, we construct the following prompt,

“ <|video|> The above video shows a robot manipulation trajectory that completes the following task: {instruction}.”

We then define the reward for a video prefix τ1:t to be

rt =

i

log pθ(insti | c(τ1:t,u,inst<i)), (6)

where insti is the i-th token of the instruction, and u represents the prompt between the video and the instruction. However, we found this alternative formulation to be less effective than the main formulation presented in Section 3. We hypothesize that this is because the model will assign high probability to entities in the instruction if they ever appear in the robot video trajectory. For example, if there is an apple in the video, and the instruction is to peel the apple, then the model will assign high probability to apple in the instruction, defeating the purpose of progress estimation. In contrast, our main formulation only requires the model to judge whether the trajectory completes the instruction, which prevents such distraction in probability evaluation.

### B. Why the True Token?

We choose True as the affirmative completion token rather than alternatives (e.g., Yes) because it is a single token in our evaluated vocabularies and yields the largest, most consistent separation between successful and failed trajectories at the final step. Figure 8 shows the top tokens by absolute difference in mean final-step token probability; True exhibits the largest gap.

[Figure 55]

Figure 8. Top 10 tokens by absolute difference in mean final-step token probability between successful and failed trajectories. The affirmative token True shows the largest separation, motivating its use as the completion token in TOPReward. Left: mean token probability by group; right: absolute difference in mean token probability.

### C. Dataset-level breakdown

This section provides dataset-level details that complement the aggregate results in Figure 3. Table 6 reports dataset-level VOC for GVL (0-shot) and TOPReward (TOPR) for each dataset and model backbone. Figure 10 visualizes per-episode VOC distributions (top) and the per-dataset improvement ∆VOC = VOC(TOPReward) − VOC(GVL) (bottom).

[Figure 56]

###### Figure 9. Per-episode VOC distributions, broken down by evaluation set (ManiRewardBench vs Open X-Embodiment) and model backbone.

[Figure 57]

###### Figure 10. Distribution of dataset-level ∆VOC = VOC(TOPReward) − VOC(GVL), shown separately for each model backbone. Positive values indicate TOPReward outperforms GVL; the dashed line marks the per-model mean.

#### Dataset Qwen3-VL-8B Gemini-2.5-Pro Molmo2-8B

GVL TOPR ∆ GVL TOPR ∆ GVL TOPR ∆ ManiRewardBench

ManiRewardBench bimanual yam 0.164 0.947 0.783 0.566 0.546 -0.021 0.007 0.565 0.558 ManiRewardBench franka 0.242 0.942 0.700 0.695 0.448 -0.247 0.000 0.662 0.662 ManiRewardBench lerobot 0.332 0.954 0.622 0.620 0.578 -0.041 -0.001 0.595 0.597 ManiRewardBench single yam 0.544 0.945 0.401 0.752 0.488 -0.264 -0.017 0.642 0.659

#### Open X-Embodiment

aloha mobile cabinet 0.199 0.822 0.623 0.814 0.422 -0.392 0.000 0.211 0.211 aloha mobile wash pan 0.127 0.977 0.850 0.856 0.734 -0.121 0.012 0.893 0.880 aloha mobile wipe wine 0.243 0.961 0.718 0.699 0.825 0.125 0.000 0.804 0.804 aloha static candy 0.414 0.963 0.549 0.697 0.803 0.105 0.000 0.842 0.842 aloha static coffee 0.101 0.977 0.876 0.877 0.114 -0.764 0.012 0.400 0.388 aloha static cups open 0.271 0.837 0.565 0.524 0.765 0.241 0.000 0.864 0.864 aloha static pro pencil 0.000 0.963 0.963 0.457 -0.029 -0.486 0.000 0.225 0.225 aloha static screw driver 0.212 0.937 0.725 0.833 0.171 -0.662 0.000 0.925 0.925 aloha static vinh cup 0.332 0.916 0.584 0.836 0.723 -0.113 0.000 0.971 0.971 aloha static vinh cup left -0.057 0.903 0.960 0.458 0.898 0.441 -0.050 0.707 0.757 aloha static ziploc slide 0.301 0.978 0.677 0.845 0.394 -0.452 0.000 0.918 0.918 asu table top 0.206 0.949 0.743 0.303 0.301 -0.002 0.000 -0.127 -0.127 austin buds dataset 0.347 0.954 0.607 0.777 -0.055 -0.832 0.000 0.908 0.908 austin sirius dataset 0.242 0.854 0.612 0.704 0.701 -0.003 0.000 0.119 0.119 berkeley fanuc manipulation 0.201 0.866 0.665 0.422 0.333 -0.089 0.000 -0.077 -0.077 berkeley rpt 0.399 0.983 0.584 0.303 0.600 0.297 0.000 0.605 0.605 berkeleymvp 0.240 0.966 0.727 0.744 0.564 -0.180 0.000 0.472 0.472 cmu franka exploration dataset 0.000 0.626 0.626 0.622 0.272 -0.350 0.000 0.325 0.325 cmu play fusion 0.069 0.975 0.906 0.204 0.576 0.372 0.000 -0.496 -0.496 cmustretch 0.119 0.962 0.844 0.478 0.501 0.023 0.000 -0.388 -0.388 columbia cairlab pusht real -0.036 0.243 0.280 0.231 0.632 0.401 0.000 -0.244 -0.244 dlr edan shared control 0.166 0.867 0.700 0.651 0.583 -0.068 -0.600 0.770 1.370 dlr sara grid clamp 0.000 0.740 0.740 0.253 -0.166 -0.419 0.000 0.562 0.562 dlr sara pour 0.251 0.721 0.470 0.351 0.374 0.023 0.000 0.081 0.081 jaco play 0.090 0.907 0.818 0.513 0.508 -0.005 0.000 -0.299 -0.299 kaist nonprehensile 0.158 0.914 0.756 0.491 0.349 -0.142 0.000 0.251 0.251 nyudoor 0.608 0.802 0.194 0.872 0.729 -0.142 0.000 0.650 0.650 nyufranka 0.232 0.875 0.642 0.772 0.494 -0.278 0.059 0.865 0.806 stanford hydra dataset 0.164 0.973 0.809 0.379 0.557 0.178 0.000 0.091 0.091 stanford kuka multimodal dataset 0.122 0.821 0.700 -0.390 -0.055 0.335 0.000 0.493 0.493 stanford robocook 0.035 0.443 0.408 0.329 0.521 0.191 0.000 0.582 0.582 taco play 0.015 0.779 0.764 0.050 0.024 -0.026 0.000 0.342 0.342 tokyo u lsmo 0.215 0.977 0.762 0.690 0.501 -0.188 0.000 0.719 0.719 ucsd kitchen dataset 0.183 0.718 0.536 0.542 -0.006 -0.547 0.000 0.228 0.228 ucsd pick and place dataset 0.090 0.819 0.729 0.683 0.520 -0.163 0.000 0.605 0.605 utaustin mutex -0.217 0.735 0.951 0.666 0.297 -0.369 0.000 0.111 0.111 utokyo pr2 opening fridge 0.485 0.957 0.472 0.372 0.644 0.272 -0.075 0.849 0.924 utokyo pr2 tabletop manipulation 0.784 0.946 0.162 0.696 0.485 -0.211 0.000 0.121 0.121 utokyo xarm bimanual 0.266 0.815 0.549 0.495 0.277 -0.218 0.000 0.386 0.386

- Table 6. All datasets (alphabetical) comparing GVL (0-shot) vs TOPReward (TOPR) and their difference, broken down by model backbone. Bold indicates the higher VOC between the two methods for that dataset/model.

### D. Additional details of ManiRewardBench

The benchmark includes 130 diverse tasks that capture a wide range of everyday manipulation activities, such as stacking objects, sorting items, and interacting with containers. The dataset is manually collected using Franka, SO-100/101, bimanual YAM, and single-arm YAM.

Example challenging tasks in ManiRewardBenchinclude:

##### Multi-step / Reasoning tasks.

- •“Push the puzzles to spell word GO” (LeRobot) — spatial reasoning combined with sequential multi-object manipulation.
- •“Build a pyramid” (Bimanual YAM) — multi-step stacking with precise positioning, requiring four distinct subtasks.
- •“Group the cubes by color” (Bimanual YAM) — requires color perception and categorical reorganization of multiple objects.
- •“Put the cubes of the same colors together” (Franka) — color-conditional sorting across multiple objects.
- •“Remove the block and stack the green cube on the red cube” (LeRobot) — obstacle removal followed by colorconditional stacking.
- •“Pack and close the box” (Franka) — multi-phase task involving packing objects then closing the container.

##### Fine manipulation / Precise control.

- •“Align the cubes horizontally” (Bimanual YAM) — fine spatial alignment, corresponding to the longest execution durations in the dataset.
- •“Rotate the banana by 90 degrees” / “Rotate the marker by 45 degrees” (Franka) — precise rotation control with specified angles.
- •“Make the screw points to the glue” (Bimanual YAM) — precise orientation alignment of two distinct objects.
- •“Pour tea” (Franka) — requires controlled pouring motion and spatial orientation awareness.

##### Deformable object handling.

- •“Fold the towel” / “Fold towel” (Franka / Bimanual YAM) — deformable material manipulation requiring careful grasp and fold planning.
- •“Stack one cloth on top of another” (Single-arm YAM) — soft object stacking with non-rigid geometry.

##### Abstract / Symbolic tasks.

- •“Press enter and then space key” (Franka) — keyboard interaction requiring sequential key presses.
- •“Set table” (Bimanual YAM) — open-ended task requiring understanding of table-setting conventions.

The following table summarizes the statistics of each dataset in ManiRewardBench. We briefly describe each dataset below:

- • Lerobot Bimanual dataset: Successful bimanual LeRobot manipulation demos (push, put, remove, stack tasks), 5–10 episodes per task.
- • Lerobot failure dataset: Mixed failed and successful trajectory examples with the same task types, ∼7 episodes per task.

Table 7. Summary of ManiRewardBench datasets

##### Dataset Episodes Tasks

Lerobot 150 22 Lerobot failed 156 23 Franka 150 51 Bimanual YAM 97 20 Single-arm YAM 100 20

6 tasks appear in both the Lerobot and Lerobot failed splits, giving 130 unique tasks

Total 653 136

across the full benchmark.

- • Franka dataset: Franka robot demos with a diverse set of 51 instructions (rotation, cleaning, packing, pick-and-place), mostly 3 episodes per task.
- • Bimanual YAM dataset: Bimanual YAM manipulation (fold, stack, build, open, etc.), 5 episodes per task.
- • Single-arm YAM dataset: Single-arm YAM manipulation (put, remove, stack), 5 episodes per task.

##### D.1. Subtask Annotation

For each task, episodes are manually labeled and segmented into a sequence of predefined subtasks. Each task is associated with an ordered list of subtasks that represent stages of execution (e.g., reaching for an object, grasping it, or placing it). For every subtask, annotators specify a start second (the time in seconds when the subtask begins) and an end second (the time in seconds when it ends). Subtasks are non-overlapping and strictly ordered in time, with each subtask beginning immediately after the previous one ends.

[Figure 58]

###### Figure 11. Counts of different example tasks in the single-arm YAM dataset.

[Figure 59]

###### Figure 12. Counts of different example tasks in the bimanual YAM dataset.

[Figure 60]

###### Figure 13. Counts of different example tasks in the SO-100 dataset.

[Figure 61]

###### Figure 14. Counts of different example tasks in the Franka dataset.

[Figure 62]

Figure 15. Frequency of verbs

[Figure 63]

[Figure 64]

Figure 16. Screenshot of the Annotation Tool.

[Figure 65]

[Figure 66]

start second: 0.0 end second: 3.9 (a) Grab the can

[Figure 67]

[Figure 68]

start second: 4.0 end second: 6.4 (b) Place the can in the plate

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

start second: 6.5 end second: 9.5 (c) Grab the spoon

start second: 9.6 end second: 11.4 (d) Place the spoon in the plate Figure 17. Expert demonstration with annotation for the task ”Clean the table”.

