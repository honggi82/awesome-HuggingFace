## arXiv:2602.05548v3[cs.LG]30Mar2026

# Unveiling Implicit Advantage Symmetry: Why GRPO Struggles with Exploration and Difficulty Adaptation

Zhiqi Yu1*, Zhangquan Chen2*, Mengting Liu3, Heye Zhang3 and Liangqiong Qu † 1 1University of Hong Kong, 2Tsinghua University, 3Sun Yat-sen University

Reinforcement Learning with Verifiable Rewards (RLVR), particularly GRPO, has become the standard for eliciting LLM reasoning. However, its efficiency in exploration and difficulty adaptation remains an open challenge. In this work, we argue that these bottlenecks stem from an implicit advantage symmetry inherent in Group Relative Advantage Estimation (GRAE). This symmetry induces two critical limitations: (i) at the group level, strict symmetry in weights between correct and incorrect trajectories leaves unsampled action logits unchanged, thereby hindering exploration of novel correct solution. (ii) at the sample level, the algorithm implicitly prioritizes medium-difficulty samples, remaining agnostic to the non-stationary demands of difficulty focus. Through controlled experiments, we reveal that this symmetric property is sub-optimal, yielding two pivotal insights: (i) asymmetrically suppressing the advantages of correct trajectories encourages essential exploration; (ii) learning efficiency is maximized by a curriculum-like transition—prioritizing simpler samples initially before gradually shifting to complex ones. Motivated by these findings, we propose Asymmetric GRAE (A-GRAE), which dynamically modulates exploration incentives and sample-difficulty focus. Experiments across seven benchmarks demonstrate that A-GRAE consistently improves GRPO and its variants across both LLMs and MLLMs.

Keywords: Large Language Models, Reinforcement Learning, LLM Reasoning

Date: February 5, 2026

Code Repository: https://github.com/HKU-HealthAI/A-GRAE

Contact: zhiqiyu777@connect.hku.hk

### 1. Introduction

Reinforcement Learning with Verifiable Rewards (RLVR) Ouyang et al. (2022), Achiam et al. (2023), Shao et al. (2024), Wen et al. (2025), Guo et al. (2025) has become a cornerstone for activating Foundation Models’ capacity to address complex reasoning tasks through Chain-of-Thought (CoT) generation. Among various algorithms, Group Relative Policy Optimization (GRPO) has emerged as the standard implementation in advanced systems such as DeepSeek-R1 Guo et al. (2025) and OpenAI-o1 Jaech et al. (2024). The core innovation of GRPO lies in Group Relative Advantage Estimation (GRAE), which computes relative advantage scores within sampled groups to obviate the need for a value model.

Despite its empirical success, recent studies have identified two fundamental limitations of GRPO. The first is capability boundary shrinkage: A prominent critique posits that GRPO primarily leads to an exploration-exploitation trade-off within existing policy constraints rather than effectively expanding the decision boundary Yue et al. (2025), Bamba et al. (2025), He et al. (2025), Ma et al. (2025), Huang et al. (2025). This claim is supported by empirical evidence showing that GRPO’s Pass@k can even fall below that of the base model at large k. The second limitation is inadequate focus on problem difficulty: GRPO’s

[Figure 1]

- Figure 1: The two-fold implicit advantage symmetry problem of GRAE in GRPO. At the group level, the advantage weights for correct trajectories equal those of incorrect trajectories. This symmetry leads to the logits of low-probability correct paths unchanged within the behavior space, thereby hindering the model’s exploration. At the sample level, samples of medium-difficulty exhibit the largest sum of absolute advantage values, which leads to insufficient training on harder data.

reward mechanism is difficulty-agnostic, treating all tasks uniformly without accounting for their inherent complexity or the model’s current capacity Zhang and Zuo (2025), Zhou et al. (2025), Zhang et al. (2025b), Jeddi et al. (2025), Zhou et al. (2026). This lack of granularity often leads to either catastrophic overfitting on simpler tasks or insufficient learning on more challenging ones.

In this work, we begin by formalizing reinforcement learning as an advantage-led reweighting variant of SFT, and reveal that these two deficiencies stem from a previously overlooked implicit advantage symmetry inherent in GRAE. As illustrated in fig. 1, this symmetry manifests at two levels: At the group level, the advantage weights for correct and incorrect trajectories are strictly equivalent. Through logits analysis in the behavior space, we formally demonstrate that this symmetry restricts the exploration of unsampled, potentially optimal paths. At the sample level, by quantifying the absolute sum of advantages across samples, we reveal that the algorithm implicitly prioritizes medium-difficulty instances. Consequently, the optimization remains agnostic to the non-stationary demands of training dynamics, failing to adapt its focus as the model evolves.

Building on this analysis, we perform controlled interventions that deliberately break GRAE’s intrinsic advantage symmetry at both the group and sample levels to examine its causal effect on learning dynamics. Our experiments reveal that the symmetric property is sub-optimal and yield two key design principles: (i) asymmetrically suppressing the weights of correct trajectories fosters essential exploration; (ii) learning efficiency is substantially boosted by a curriculum-like progression, which prioritizes simpler samples initially before escalating to complex ones. Motivated by these findings, we propose Asymmetric GRAE (A-GRAE), which refines the original GRAE strategy by explicitly instantiating these two principles within the GRPO framework: it introduces asymmetric exploration to push the policy beyond its current solution set, and a curriculum-like progression learning schedule to align the optimization focus with the model’s evolving capability.

To fully validate A-GRAE, we conduct comprehensive experiments across seven diverse benchmarks—spanning both natural language reasoning and vision-language reasoning tasks—using various commonly used LLMs and VLMs. Experimental results demonstrate that A-GRAE consistently enhances the reasoning performance

of GRPO and its representative variants DAPO Yu et al. (2025) and Dr.GRPO Liu et al. (2025b) across all settings, with significant improvements in key metrics (e.g., accuracy, pass@k).

In summary, our contributions are as follows:

- • We identify and define the implicit advantage symmetry property in GRAE. Through systematic investigation, we demonstrate that this symmetry is sub-optimal, prompting a fundamental rethinking of advantage function design in RLVR.
- • We reveal that asymmetrically suppressing the weight of correct trajectories at the group level can enhance reasoning performance. Furthermore, we provide a theoretical analysis of the associated risk of learning collapse, attributing it to blindly intensified exploration.
- • We uncover a dynamic shift in optimal learning efficiency at the sample level: contrary to GRAE’s static focus, prioritizing simple samples in early stages and harder ones later yields superior performance.
- • We propose Asymmetric GRAE (A-GRAE), a novel framework that dynamically encourages trajectory exploration and adapts sample difficulty bias. Extensive evaluations across seven benchmarks validate that it can consistently improve GRPO and its variants.

### 2. Preliminary

Notation. In this work, we frame a foundation model (either an LLM or MLLM), parameterized by θ, as a policy model: πθ and πθold denote the current and prior policies, respectively. For batch processing, we sample a question q from the dataset 𝒬. For each question q, the prior policy πθold generates a set of G candidate responses {oi}iG=1. Each question-response pair (q, oi) is then assigned a scalar reward ri via a rule-based verifier, where i ∈ {1,2, . . . , G} indexes responses per question. By default, we employ a binary accuracy reward: ri = 1 if response oi is correct for question q, and ri = 0 otherwise. For each question q, we further compute a group-relative advantage value Ai for each of its candidate responses oi.

Group Relative Policy Optimization (GRPO). GRPO simplifies the training pipeline of PPO by removing the critic model—typically matching the size of the policy model—and instead estimates baselines via group-level reward scores. Specifically, for each query q, GRPO samples a set of responses {o1, o2, . . . , oG} from the prior policy πθold and optimizes the target policy πθ by maximizing the following objective function:

∣oi∣

G

[min(ρi,tAi,t,clip(ρi,t,1 ± ϵ) Ai,t) − βDKL], (1)

JGRPO(πθ) = Eq∼𝒬,{o

1 G

1 ∣oi∣

∑

∑

i}iG=1∼πθold(⋅∣q)

t=1

i=1

where ρi,t = ππθ(oi,t∣q,oi,<t)

θold(oi,t∣q,oi,<t), ϵ and β are hyperparameters, the KL term is defined as

πref(oi,t∣q, oi,<t) πθ(oi,t∣q, oi,<t)

πref(oi,t∣q, oi,<t) πθ(oi,t∣q, oi,<t)

− 1, (2)

DKL =

− log

and the advantage Ai is computed using a group of rewards {r1,r2,...,rG} with GRAE:

ri −stdmean({r1({,rr21,...,,r2,...,rG})rG})

. (3)

Ai =

### 3. Implicit Advantage Symmetry in GRPO

First, we demonstrate that GRPO’s policy gradient optimization can be formulated as a reweighting variant of Supervised Fine-Tuning (SFT). Specifically, the gradient update can be formulated as follows (proof in section B.1):

⎡

⎤

∣oi∣

G

▽θ𝒥GRPO(θ) = Eq∼𝒬,{o

▽θ log πθ(oi,t ∣ q, oi,<t) SFT Group Level

1 ∣oi∣

1 G

∑

∑

i}iG=1∼πθold(⋅∣q)

.

ρi,tAi,t weight

(4)

t=1

i=1

⎢⎣

⎥⎦

Sample Level

In eq. (4), ρi,t is relatively stable with the clip operation, so the dominant part of reweighting is Ai,t. In general, the advantage of GRPO is shared across the entire sequence, meaning that all tokens within a single

response correspond to the same advantage value, i.e.,Ai,1 = Ai,2 = ⋅⋅⋅ = Ai,t = Ai. We then examine this reweighting mechanism from the group level and the sample level.

Advantage Symmetry in Group Level: In RLVR, a trajectory is considered correct provided that the final answer extracted from the response aligns with the ground truth. Given a query, we partition the sampled within-group responses into correct (positive) and incorrect (negative) trajectories. Crucially, we observe that the weights of policy updates attributed to correct trajectories is strictly equivalent to that of incorrect ones (proof in section B.2). This property can be expressed as:

∣Ai∣ = ∑

∣Ai∣ (5)

∑

i∈𝒢pos

i∈𝒢neg

where 𝒢pos and 𝒢neg represent the sets of positive and negative trajectories, satisfying the conditions 𝒢pos ∩ 𝒢neg = ∅ and 𝒢pos ∪ 𝒢neg = 𝒢 (the sampled response set).

- Theorem 1 (The Logits Update in Behavior Space) Assume that the set of all possible behaviors is ℬ = {bi}iN=1,

consisting of sampled set 𝒢 and unsampled set 𝒰. Given the intragroup advantage sum C = ∑o

i∈𝒢 Ai and negligible importance sampling bias, the produced probability updates of path bi can be expressed as:

▽θJ = η ⋅ [I(bi ∈ 𝒢)Ai − Cπbi], (6)

where η denotes the learning rate and πbi is the model’s current sampling probability in behavior space for path bi. The proof can be found in section B.3.

In the standard GRPO setting, the advantage normalization enforces a zero-sum property, i.e., C = 0. The probability updates can be divided into the following cases.

- Case A. For the sampled positive responses bi ∈ 𝒢pos: ∆hbi = ηApos, (7)

where Apos > 0. The logits of sampled correct trajectories receive a positive update, strictly increasing their probability. This confirms that GRPO effectively exploits known correct solutions.

- Case B. For the sampled negative responses bi ∈ 𝒢neg: ∆hbi = ηAneg, (8)

where Aneg < 0. The logits are directly penalized by the negative advantage. This suppresses the generation of known errors.

- Case C. For the unsampled response bi ∈ 𝒰:

∆hbi = η ⋅ (0 − 0 ⋅ πbi) = 0. (9)

It can be observed that GRPO yields a strictly zero gradient for any unsampled trajectory. Consequently, even if a low probability correct trajectory exists in the behavior space, its logit remains static unless it is stochastically sampled. This mathematical property proves that GRPO lacks an intrinsic active exploration mechanism for unsampled correct trajectories, leading to local optima entrapment.

Advantage Symmetry in Sample Level: At the sample level, the overall learning contribution can be captured by the sum of absolute advantages, which represents the total magnitude of one query. To evaluate the relative contributions across queries of varying complexities, we introduce the sample success rate, denoted as p for a group of size ∣G∣, as a proxy for task difficulty; specifically, a higher p indicates a lower level of difficulty. By leveraging this metric, we can formally quantify the relationship between sample difficulty and the corresponding update magnitude.

- Theorem 2 (Update Magnitude with respect to Sample Difficulty). Consider a group G of trajectories for one

query (sample) with binary rewards {ri}iG=1, the sum of absolute advantages over the group under Group Relative Advantage Estimation (GRAE) can be derived as:

√p(1 − p). (10)

∣Ai∣ = 2∣G∣

∑

i∈G

where p = ∑i∈G ∣ri∣/∣G∣ denotes the empirical success probability of the corresponding sample within group G, and the proof can be found in section B.4.

Theorem 2 reveals an intrinsic difficulty bias of GRAE, where samples of intermediate difficulty (p = 0.5) dominate policy updates, irrespective of the training stage. Furthermore, as shown in fig. 6, due to the symmetry of the term √p(1 − p), simple samples (e.g., p = 0.75) and hard samples (e.g., p = 0.25) exhibiting the same deviation from p = 0.5 are assigned identical importance weights. However, considering the dynamics of model development, there is an inherent shift in the sample distribution: the ratio of relatively simple instances increases over time as model evolves, whereas the frequency of difficult samples steadily declines. This distributional shift predisposes the model to overfit on trivial data while leaving it insufficiently trained on challenging scenarios. Consequently, the intrinsic advantage symmetry at the sample level fails to satisfy the non-stationary demands of the evolving training process.

### 4. Deconstructing the Implicit Advantage Symmetry of GRAE

##### 4.1. Experimental Setup

To systematically investigate the impact of advantage symmetry on reasoning performance, we design two sets of ablation experiments as shown in table 1. The detailed setup is as follows:

- • Control Experiment I (Breaking Intra-Group Symmetry): This experiment examines whether the model benefits more from suppressing the contribution of correct trajectories. To achieve this, we introduce a scaling coefficient β = 10 to disrupt the zero-sum equilibrium (∑ Ai = 0). Denoting the original positive advantage as Apos, we design two variants:

- 1. Positive-Dominant Group: We scale up positive advantages (A∗pos = β ⋅ Apos).
- 2. Negative-Dominant Group: We scale down positive advantages (A∗pos = Apos/β).

The original GRPO serves as the control group, representing the symmetric equilibrium.

- • Control Experiment II (Breaking Sample-Level Symmetry): This experiment investigates the appropriate difficulty focus during training process. We modify the advantage magnitude based on the sampling success rate p. Let Ai be the original advantage of GRPO. We define two curriculum variants:

- 1. Hard-Focused Group: We shift the learning focus toward harder queries by rescaling the advantage with

the factor 1/

√p (A∗i = γ ⋅ Ai/

√p).

- 2. Easy-Focused Group: Conversely, we shift the focus to simple queries by scaling with the factor 1/

√1 − p (A∗i = γ ⋅ Ai/

√1 − p).

Here, γ = 0.5 is a normalization constant to ensure that the theoretical maximum value remains consistent with the control group (standard GRPO). Note that no extra rescaling is conducted when the success rate is 0 or 1, as the GRPO advantage is zero in these cases.

Table 1: Summary of Controlled Experiments. We design two sets of ablation studies to investigate the impact of breaking symmetry at the group level and sample level. α = 10 and γ = 0.5 are scaling constants.

- (a) Experiment I: Breaking Group-Level Symmetry.

Variant Formulation (A∗) ∑iG A∗i

GRPO (Control Group) A∗pos = Apos = 0 Positive-Dominant A∗pos = β ⋅ Apos > 0 Negative-Dominant A∗pos = Apos/β < 0

- (b) Experiment II: Breaking Sample-Level Symmetry.

Variant Formulation (A∗) ∑iG ∣A∗i ∣ GRPO (Control Group) A∗i = Ai 2∣G∣

√p(1 − p) Hard-Focused A∗i = γ ⋅ Ai √p ∣G∣

√1 − p Easy-Focused A∗i = γ ⋅ Ai/

√1 − p ∣G∣

√p

Models and Training Setup: To ensure the model-agnostic generalizability of our investigations across different architectures, we conduct experiments using two distinct LLM backbones: the math-specialized Qwen2.5-Math-7B Yang et al. (2024) and the general-purpose Llama-3.2-3B-Instruct Dubey et al.

- (2024). We conduct training on the MATH dataset Hendrycks et al. (2021) under the verl framework Sheng et al. (2025). In terms of hyperparameters, we set the batch size to 1,024, with G = 8 rollouts generated for each query. The policy is updated with a mini-batch size of 256 and a learning rate of 1 × 10−6. We evaluate the models on three widely recognized mathematical reasoning benchmarks: the test sets of MATH, AMC23, and the most challenging AIME 2025. Following prior works Hochlehnert et al. (2025), Chen (2021), Chen et al. (2025), Zhu et al. (2025), we set our primary evaluation metric as Pass@k with an unbiased estimator. Experimental Details can be found in section C.2.

[Figure 2]

[Figure 3]

[Figure 4]

- Figure 2: Experimental results on breaking group-level symmetry using Qwen2.5-Math-7B. We amplify (PositiveDominant) or suppress (Negative-Dominant) the advantages of correct trajectories to compare their performance with that of GRPO and the base model.The performance is evaluated using Pass@k (k = {1,2,4,8,16,32,64,128,256}).

- 4.2. Rethinking Symmetry in Group Level The results of Control Experiment I are shown in fig. 2. The following is an analysis of each group.

GRPO enhances efficiency without expanding reasoning boundaries. It is evident that GRPO significantly improves Pass@1 accuracy over the base model across all three datasets, demonstrating its effectiveness in enhancing sampling efficiency and reasoning capabilities. However, the performance gains are gradually eroded as k increases. Notably, on the AMC23 and MATH datasets, GRPO’s performance at Pass@256 falls below that of the base model. This suggests that while GRPO improves the sampling probability of correct paths, it fails to discover novel solutions that lie outside the base model’s original sampling support. In other words, GRPO does not fundamentally expand the intrinsic reasoning boundaries of the base model. This observation aligns with findings in recent literature Yue et al. (2025), Yao et al. (2025).

Over-emphasizing correct paths triggers entropy collapse. In contrast to GRPO, amplifying the weight of positive trajectories fails to yield performance improvements. In particular, on the AIME2025 and MATH datasets, the Positive-Dominant group significantly underperforms other methods at larger sampling budgets (k > 16). To investigate the underlying cause, we found that the entropy of the Positive-Dominant group on the test set exhibits the most precipitous decline as illustrated in fig. 3. This phenomenon indicates that overemphasis on correct trajectories reduces sampling diversity by excessively sharpening the output distribution, leading to entropy collapse and thus impairing the model’s reasoning capability.

Suppressing the correct path improves performance but risks instability. Remarkably, suppressing positive trajectories proves more effective than GRPO. On one hand, the Negative-Dominant group consistently outperforms GRPO in Pass@k metrics across all three datasets, with the advantage becoming increasingly pronounced as k grows. On the other hand, even at a high sampling budget (k = 256), this approach maintains parity with the base model and achieves significant gains on the most challenging AIME2025 dataset. This suggests that the strategy effectively mitigates the potential capability boundary shrinkage observed in GRPO. Correlating this with fig. 3, we attribute this gains to the continuous increase in entropy throughout training which indicates model exploration—a trend diametrically opposed to the other groups. However, in practice, we observe that persistent entropy growth can lead to training instability in later stages, which is manifested by a sudden increase in fully unsolved questions (demonstrated in section D.3). This instability stems from the fact that in the Negative-Dominant group, overconfident incorrect trajectories may displace the correct ones during training process. We discuss this situation in section B.5 based on Theorem 1. Consequently, a dynamic adjustment mechanism may be required as training progresses to balance diversity and stability.

- Figure 3: Entropy dynamics across the three groups in Experiment I on the training set. Notably, the NegativeDominant group exhibits a monotonic increase in entropy except at the very beginning, while the other groups show the opposite behavior.

Takeaway 1: As discussed in section 4.2, while suppressing positive advantage incentivizes the model to explore unsampled correct trajectories to improve the reasoning ability, it simultaneously introduces the risk of training collapse in the later stages of reinforcement learning . Consequently, a dynamic adjustment mechanism may be required as training progresses.

- 4.3. Rethinking Symmetry in Sample Level

[Figure 6]

[Figure 7]

[Figure 8]

- Figure 4: Experimental results on breaking sample-level symmetry using Qwen2.5-Math-7B. We rescaling the advantages to shift the learning focus toward harder queries (Hard-Focused) or easier queries (Easy-Focused) to compare their performance with that of GRPO and the base model.The performance is evaluated using Pass@k (k = {1,2,4,8,16,32,64,128,256}).

The results for Experiment II are presented in fig. 4, from which the following observations can be derived. Difficulty-based reweighting offers no universal advantage and depends on the test difficulty. As shown in fig. 4, none of the methods yielded a uniform advantage across all datasets. More precisely, their relative efficacy varies by benchmark: hard-focused methods achieve peak results on the challenging AIME2025, whereas easy-focused approaches marginally outperform others on simpler datasets such as AMC23 and MATH at Pass@1. These findings indicate that difficulty reweighting should be calibrated to test difficulty, challenging the prevailing trend of focusing solely on hard samples throughout training. Zhang and Zuo

- (2025), Pikus et al. (2025), Guan et al. (2025), Ding et al. (2026).

- Figure 5: The within-batch count of correct sampling responses on the training set. Easy-Focused exhibits the most rapid initial convergence during the early stages of training, whereas Hard-Focused maintains a sustained upward trajectory in the later phases, eventually achieving superior performance.

Prioritizing simple samples in early stages and hard samples later yields better learning efficiency gains. As discussed above, direct evaluation of the trained models fails to identify a universally optimal difficulty reweighting strategy. This necessitates a deeper investigation into their respective training dynamics to potentially integrate their complementary strengths. During training, the within-batch count of correct samples serves as a direct indicator of learning gains, the results of which are visualized in Figure 5. Notably, the Easy-Focused strategy exhibits the most rapid initial acceleration. This suggests that during the early phases of training, prioritizing simpler tasks promotes the learning of basic formatting rules and core reasoning patterns. In contrast, the Hard-Focused strategy emerges as the superior performer in the later stages. This shift suggests that once model capability reaches relative saturation, a transition toward difficult samples becomes essential to further elevate the performance ceiling and mitigate the risk of overfitting on simpler data.

Takeaway 2: Our findings in section 4.3 suggest that static difficulty reweighting is insufficient, as optimal sample utility is phase-dependent. While easy samples facilitate more effective learning in the early stage, a strategic transition toward hard samples is useful as performance begins to plateau.

##### 4.4. Connecting RL Methods via Advantage Symmetry

While recent advancements such as Dr.GRPO Liu et al. (2025b) and DAPO Yu et al. (2025) have refined the GRAE paradigm, they overlook the intrinsic property of advantage symmetry—which still holds under their methodologies—and consequently fail to address the persistent issues of boundary shrinkage and difficulty adaptation. Additionally, the concept of advantage symmetry provides an explanatory framework for the efficacy of existing methods. For instance, some approaches encourage exploration through negative learning Zhu et al. (2025), Yao et al. (2025), Nan et al. (2025), Li et al. (2024), Wang et al. (2024b), which implicitly disrupts group-level symmetry by suppressing or removing the advantages of correct trajectories. Others enhance model performance by emphasizing high-entropy tokens Cheng et al. (2025), Hao et al. (2025), Zhang et al. (2025c), Deng et al. (2025b) or hard samples Park et al. (2025), Zhang and Zuo (2025), Shen et al. (2025), Zhang et al. (2025a), Bae et al. (2025), essentially breaking sample-level symmetry. However, these methods only implicitly address symmetry at one of these levels, and fail to simultaneously

improve accuracy and diversity. In summary, advantage symmetry is a critical yet underrecognized property. Integrating it into reinforcement learning frameworks will inspire a rethinking of advantage design strategies1.

### 5. Asymmetric GRAE

##### 5.1. Method

Our prior analysis reveals that the advantage symmetry inherent in GRPO undermines model exploration and difficulty adaptation. To address these limitations, we propose the Asymmetric Group Relative Advantage Estimation (A-GRAE) framework to dynamically modulate exploration incentives and sample-difficulty focus. To implement this, a metric is required to quantify training state; accordingly, we introduce the batch-wise mean reward as a proxy indicator:

∑iB=1 ri B

, (11)

ωs =

where B denotes the total number of trajectories in the batch, ωs denotes the mean reward of the current step, where a higher value implies stronger model proficiency. Then we introduce a dynamic attention shift at the sample level, transitioning from easy to hard samples as training progresses:

ristd−({meanr1,r({2,...,r1,rrG2,...,}) ⋅r√Gp})

ri − mean({r1,r2,...,rG}) std({r1,r2,...,rG}) ⋅ √1 − p

- 1 − ωs

- 2

ωs 2

, (12)

Ai =

+

⋅

⋅

where p denotes the sampling success rate for a given query. Note that rescaling is omitted when p ∈ {0,1}, as the standard advantage equals 0. As the model evolves with increasing sampling success rate, the weight assigned to the hard-focused component (first term) progressively increases, while the corresponding weight for the easy-focused component (second term) diminishes. This mechanism facilitates adaptive trade-offs, dynamically shifting training focus to the hard questions as the model’s proficiency improves.

At the group level, we propose an attenuation suppression strategy for correct-response advantages, which encourages adequate exploration in the early training stage while preserving stability in the later phase:

Ai ∗ min(1, ωαs), if A > 0 Ai, if A ≤ 0

A∗i = {

(13)

Here, α ≤ 1 is a scaling parameter. Once the refined advantages are computed, they can be seamlessly incorporated into the GRPO objective function, as defined in eq. (1), or other GRPO variants for policy optimization.

##### 5.2. Experimental Setup

We validate our method across seven benchmarks, including text-only mathematical tasks (AIME2025, AMC23, MATH Hendrycks et al. (2021)) with Qwen2.5-Math-7B and DeepSeek-R1-7B (section D.2) and multimodal tasks in mathematics (Geo3k Lu et al. (2021), MathVision Wang et al. (2024a), MathVerse Zhang

- et al. (2024)) and medicine (HuatuoGPT-Vision Chen et al. (2024)) with Qwen2.5-VL-3B-Instruct. Our 1Related work is discussed in detail in section A.

Method Pass@k k 1 2 4 8 16 32 64 128 256

MATH Base Model 63.4 74.8 83.2 88.6 91.2 93.4 94.1 95.0 96.3 GRPO 76.5 82.3 86.1 88.8 90.3 92.6 93.5 93.9 95.0 GRPO-LEAD 77.8 83.0 86.5 89.2 90.5 92.3 92.8 93.6 95.0 W-REINFORCE 76.6 82.8 87.1 90.2 92.4 94.1 95.3 96.1 96.7 GRPO + A-GRAE 78.3 85.0 89.2 91.0 92.5 94.6 95.0 95.5 96.5 DAPO 75.0 79.8 85.0 88.4 89.8 92.0 92.8 93.4 94.3 DAPO + A-GRAE 76.9 82.6 86.5 89.2 90.6 92.8 93.6 94.2 95.3 Dr.GRPO 77.2 82.5 87.4 89.6 91.0 92.8 93.6 94.3 95.0 Dr.GRPO + A-GRAE 78.6 86.2 89.8 90.6 92.8 95.0 95.4 96.0 96.9

AIME 2025 Base Model 6.1 9.9 14.4 19.3 24.4 29.1 33.4 39.2 46.7 GRPO 10.3 14.3 18.7 23.1 27.5 31.8 36.1 40.8 46.7 GRPO-LEAD 11.0 14.8 19.2 23.4 27.8 32.0 36.5 41.4 47.3 W-REINFORCE 10.6 15.3 20.0 24.7 29.7 34.6 40.5 47.8 56.7 GRPO + A-GRAE 11.3 15.6 19.8 24.7 28.6 34.1 39.2 47.8 56.7 DAPO 12.0 16.1 21.3 25.2 29.4 33.2 38.5 45.4 53.3 DAPO + A-GRAE 13.3 18.4 23.0 26.3 30.0 35.1 41.1 48.7 60.0 Dr.GRPO 11.0 14.8 19.3 24.3 28.8 33.0 37.1 41.2 46.7 Dr.GRPO + A-GRAE 11.8 16.2 19.8 25.0 29.3 34.8 37.9 48.0 56.7

AMC23 Base Model 40.6 55.3 68.6 78.6 85.0 89.4 93.4 97.3 100.0 GRPO 60.2 66.7 72.1 76.4 80.6 84.8 88.3 90.8 92.5 GRPO-LEAD 62.3 68.0 73.3 77.8 81.5 85.0 88.2 90.3 92.3 W-REINFORCE 62.0 70.0 77.0 83.1 87.8 91.8 95.2 97.1 97.5 GRPO + A-GRAE 62.6 70.0 77.5 83.7 88.2 92.0 95.1 96.8 97.5 DAPO 62.0 70.3 77.2 83.1 87.8 91.4 94.0 96.1 97.5 DAPO + A-GRAE 63.3 72.5 80.5 86.7 90.2 92.9 95.0 97.0 100.0 Dr.GRPO 60.7 69.8 75.6 82.6 87.8 90.9 93.2 94.6 95.0 Dr.GRPO + A-GRAE 62.8 71.6 78.2 84.0 89.6 92.3 95.2 95.9 100.0

experiments cover GRPO variants (GRPO, DAPO, Dr.GRPO) and state-of-the-art methods W-REINFORCE Zhu

- et al. (2025) (addressing GRPO’s insufficient exploration) and GRPO-LEAD Zhang and Zuo (2025) (tackling its difficulty adaptation). We report Pass@k on the text-only datasets and Pass@1 on the multi-modal datasets since most of them are multiple choice questions. For details please refer to section C.2.

##### 5.3. Main Results

Consistent improvements in Pass@1 and Pass@k. As illustrated in table 3, our proposed method yields substantial performance gains when integrated with various GRPO variants, which proves the effectiveness of our proposed method. Notably, in comparison with W-REINFORCE and GRPO-LEAD, our method realizes consistent gains in both accuracy (Pass@1) and diversity (Pass@k). Such results indicate that our method effectively mitigates the issue of capability boundary shrinkage inherent to traditional reinforcement learning paradigms, while enhancing reasoning accuracy at the same time.

Universal applicability to multi-modal domains. To evaluate the versatility of A-GRAE, we extended our

Method Pass@k k 1 2 4 8 16 32 64 128 256

MATH Base Model 63.4±0.4 74.8±0.5 83.2±0.6 88.6±0.8 91.2±0.9 93.4±1.0 94.1±1.1 95.0±1.2 96.3±1.3 GRPO 76.5±0.3 82.3±0.4 86.1±0.5 88.8±0.6 90.3±0.7 92.6±0.8 93.5±1.0 93.9±1.1 95.0±1.2 GRPO-LEAD 77.8±0.3 83.0±0.4 86.5±0.5 89.2±0.6 90.5±0.7 92.3±0.9 92.8±1.0 93.6±1.1 95.0±1.2 W-REINFORCE 76.6±0.5 82.8±0.6 87.1±0.8 90.2±0.9 92.4±1.0 94.1±1.1 95.3±1.2 96.1±1.3 96.7±1.4 GRPO + A-GRAE 78.3±0.2 85.0±0.3 89.2±0.4 91.0±0.5 92.5±0.6 94.6±0.7 95.0±0.8 95.5±0.9 96.5±1.0 DAPO 75.0±0.4 79.8±0.5 85.0±0.6 88.4±0.7 89.8±0.8 92.0±0.9 92.8±1.1 93.4±1.2 94.3±1.3 DAPO + A-GRAE 76.9±0.3 82.6±0.4 86.5±0.5 89.2±0.6 90.6±0.7 92.8±0.8 93.6±0.9 94.2±1.0 95.3±1.1 Dr.GRPO 77.2±0.4 82.5±0.5 87.4±0.6 89.6±0.7 91.0±0.8 92.8±0.9 93.6±1.0 94.3±1.2 95.0±1.3 Dr.GRPO + A-GRAE 78.6±0.3 86.2±0.3 89.8±0.4 90.6±0.5 92.8±0.6 95.0±0.7 95.4±0.8 96.0±0.9 96.9±1.0

AIME 2025 Base Model 6.1±0.7 9.9±0.9 14.4±1.0 19.3±1.2 24.4±1.4 29.1±1.6 33.4±1.8 39.2±1.9 46.7±2.0 GRPO 10.3±0.5 14.3±0.7 18.7±0.8 23.1±1.0 27.5±1.2 31.8±1.4 36.1±1.6 40.8±1.8 46.7±1.9 GRPO-LEAD 11.0±0.5 14.8±0.6 19.2±0.8 23.4±1.0 27.8±1.2 32.0±1.4 36.5±1.5 41.4±1.7 47.3±1.8 W-REINFORCE 10.6±0.7 15.3±0.8 20.0±1.0 24.7±1.2 29.7±1.4 34.6±1.6 40.5±1.8 47.8±1.9 56.7±2.0 GRPO + A-GRAE 11.3±0.4 15.6±0.5 19.8±0.7 24.7±0.8 28.6±1.0 34.1±1.1 39.2±1.3 47.8±1.5 56.7±1.7 DAPO 12.0±0.6 16.1±0.7 21.3±0.9 25.2±1.1 29.4±1.3 33.2±1.5 38.5±1.7 45.4±1.9 53.3±2.0 DAPO + A-GRAE 13.3±0.4 18.4±0.5 23.0±0.7 26.3±0.9 30.0±1.0 35.1±1.2 41.1±1.4 48.7±1.6 60.0±1.8 Dr.GRPO 11.0±0.6 14.8±0.7 19.3±0.9 24.3±1.1 28.8±1.3 33.0±1.5 37.1±1.7 41.2±1.8 46.7±1.9 Dr.GRPO + A-GRAE 11.8±0.4 16.2±0.6 19.8±0.7 25.0±0.9 29.3±1.1 34.8±1.2 37.9±1.4 48.0±1.6 56.7±1.8

AMC23 Base Model 40.6±0.5 55.3±0.6 68.6±0.8 78.6±1.0 85.0±1.1 89.4±1.3 93.4±1.5 97.3±1.7 100.0±1.9 GRPO 60.2±0.4 66.7±0.5 72.1±0.6 76.4±0.8 80.6±0.9 84.8±1.1 88.3±1.2 90.8±1.4 92.5±1.5 GRPO-LEAD 62.3±0.4 68.0±0.5 73.3±0.7 77.8±0.8 81.5±1.0 85.0±1.1 88.2±1.3 90.3±1.4 92.3±1.6 W-REINFORCE 62.0±0.6 70.0±0.7 77.0±0.9 83.1±1.0 87.8±1.2 91.8±1.3 95.2±1.5 97.1±1.6 97.5±1.8 GRPO + A-GRAE 62.6±0.3 70.0±0.4 77.5±0.5 83.7±0.7 88.2±0.8 92.0±0.9 95.1±1.1 96.8±1.2 97.5±1.4 DAPO 62.0±0.4 70.3±0.5 77.2±0.7 83.1±0.8 87.8±1.0 91.4±1.1 94.0±1.3 96.1±1.5 97.5±1.6 DAPO + A-GRAE 63.3±0.3 72.5±0.4 80.5±0.5 86.7±0.6 90.2±0.8 92.9±0.9 95.0±1.0 97.0±1.2 100.0±1.3 Dr.GRPO 60.7±0.5 69.8±0.6 75.6±0.7 82.6±0.9 87.8±1.1 90.9±1.2 93.2±1.4 94.6±1.5 95.0±1.7 Dr.GRPO + A-GRAE 62.8±0.3 71.6±0.4 78.2±0.6 84.0±0.7 89.6±0.8 92.3±1.0 95.2±1.1 95.9±1.3 100.0±1.4

empirical analysis to the multimodal domain, with the results summarized in table 4. The result reveals that A-GRAE not only delivers significant improvements in the in-distribution (ID) domain but also yields substantial gains in out-of-distribution (OOD) scenarios. These findings provide strong evidence that our approach effectively enhances sampling efficiency while simultaneously preserving the model’s generalization capabilities. In summary, the efficacy of A-GRAE across diverse domains and tasks underscores its universal applicability as a robust framework.

##### 5.4. Further Analysis

Ablation studies. To validate the contribution of each component in A-GRAE, we perform a series of ablation studies using and the results are demonstrated in table 6. Firstly, sample-level asymmetry primarily bolsters Pass@1 performance, consistently attaining second-best results across all three benchmarks. Conversely, the group-level asymmetric mechanism exerts a more pronounced impact on enhancing Pass@k metrics, suggesting that this module effectively fosters generative diversity. Ultimately, the full framework yields the most significant gains across all evaluation metrics, demonstrating that the constituent modules of our methodology are mutually complementary. We also compare each module of our method with the control groups in section D.5 to verify their effectiveness.

Table 4: Performance comparison on multi-modal benchmarks with Qwen2.5-VL-3B-Instruct. Bold and underlined numbers indicate the best and second-best results respectively.

ID Domain OOD Domain Method Geo3K MathVision Mathverse

- Task A: General Mathematical Reasoning base model 27.8 20.8 31.6 GRPO 43.5 23.4 35.2 GRPO + A-GRAE 45.7 24.0 36.8 DAPO 44.7 23.8 35.9 DAPO + A-GRAE 45.9 24.3 37.5 Dr.GRPO 44.9 24.2 36.5 Dr.GRPO + A-GRAE 46.8 25.6 38.4

- Task B: Medical Imaging Reasoning MRI300 CT300 Xray300

base model 35.6 42.5 42.0 GRPO 87.2 71.7 63.2 GRPO + A-GRAE 88.2 73.1 71.3 DAPO 84.3 71.6 63.1 DAPO + A-GRAE 87.0 72.3 71.6 Dr.GRPO 87.8 72.4 69.5 Dr.GRPO + A-GRAE 89.0 73.6 72.0

Training dynamics. To capture the evolution of training dynamics, we monitor the entropy and greedy decoding accuracy on the MATH dataset throughout the training process, as shown in fig. 10. On the training set, the entropy of GRPO exhibits a continuous decline throughout the training process. In contrast, our proposed method shows a rapid initial drop followed by a sustained plateau, suggesting that it effectively mitigates the issue of entropy collapse. On the test set, the entropy of A-GRAE follows a trajectory of initial increase followed by a gradual decrease, reflecting a learning paradigm that balances exploration and exploitation. Finally, the greedy decoding accuracy of our approach significantly surpasses that of GRPO in the latter stages of training, further validating the efficacy of our method in facilitating sustained learning.

### 6. Conclusion

In this paper, we rethink the mechanics of Group Relative Advantage Estimation (GRAE) within the GRPO framework. Our theoretical and empirical analyzes reveal a previously overlooked “advantage symmetry" in standard GRPO, and we demonstrate that this symmetry restricts exploration and fails to adapt to the difficulty focus during the learning process. To overcome these limitations, we propose A-GRAE, a novel mechanism that dynamically modulates exploration incentives and prioritizes samples based on their evolving utility. Extensive evaluations across seven benchmarks demonstrate our superiority.

### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Sanghwan Bae, Jiwoo Hong, Min Young Lee, Hanbyul Kim, JeongYeon Nam, and Donghyun Kwak. Online difficulty filtering for reasoning oriented reinforcement learning. arXiv preprint arXiv:2504.03380, 2025.

Udbhav Bamba, Minghao Fang, Yifan Yu, Haizhong Zheng, and Fan Lai. Xrpo: Pushing the limits of grpo with targeted exploration and exploitation. arXiv preprint arXiv:2510.06672, 2025.

Junying Chen, Chi Gui, Ruyi Ouyang, Anningzhe Gao, Shunian Chen, Guiming Hardy Chen, Xidong Wang, Ruifei Zhang, Zhenyang Cai, Ke Ji, et al. Huatuogpt-vision, towards injecting medical visual knowledge into multimodal llms at scale. arXiv preprint arXiv:2406.19280, 2024.

Mark Chen. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. Zhipeng Chen, Xiaobo Qin, Youbin Wu, Yue Ling, Qinghao Ye, Wayne Xin Zhao, and Guang Shi. Pass@ k

training for adaptively balancing exploration and exploitation of large reasoning models. arXiv preprint arXiv:2508.10751, 2025.

Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective. arXiv preprint arXiv:2506.14758, 2025.

Xingyu Dang, Christina Baek, J Zico Kolter, and Aditi Raghunathan. Assessing diversity collapse in reasoning. In Scaling Self-Improving Foundation Models without Human Supervision, 2025.

Huilin Deng, Ding Zou, Rui Ma, Hongchen Luo, Yang Cao, and Yu Kang. Boosting the generalization and reasoning of vision language models with curriculum reinforcement learning. arXiv preprint arXiv:2503.07065,

- 2025a.

Jia Deng, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, and Ji-Rong Wen. Decomposing the entropy-performance exchange: The missing keys to unlocking effective reinforcement learning. arXiv preprint arXiv:2508.02260,

- 2025b.

Ruiyi Ding, Yongxuan Lv, Xianhui Meng, Jiahe Song, Chao Wang, Chen Jiang, and Yuan Cheng. Prpo: Aligning process reward with outcome reward in policy optimization, 2026. URL https://arxiv.org/ abs/2601.07182.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pages arXiv–2407, 2024.

Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, and Noah D Goodman. Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars. arXiv preprint arXiv:2503.01307, 2025.

Wei Guan, Jun Lan, Jian Cao, Hao Tan, Huijia Zhu, and Weiqiang Wang. Emit: Enhancing mllms for industrial anomaly detection via difficulty-aware grpo. arXiv preprint arXiv:2507.21619, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Zhezheng Hao, Hong Wang, Haoyang Liu, Jian Luo, Jiarui Yu, Hande Dong, Qiang Lin, Can Wang, and Jiawei Chen. Rethinking entropy interventions in rlvr: An entropy change perspective. arXiv preprint arXiv:2510.10150, 2025.

Andre Wang He, Daniel Fried, and Sean Welleck. Rewarding the unlikely: Lifting grpo beyond distribution sharpening. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 25559–25571, 2025.

Xuehai He, Yichen Zhang, Luntian Mou, Eric Xing, and Pengtao Xie. Pathvqa: 30000+ questions for medical visual question answering. arXiv preprint arXiv:2003.10286, 2020.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Andreas Hochlehnert, Hardik Bhatnagar, Vishaal Udandarao, Samuel Albanie, Ameya Prabhu, and Matthias Bethge. A sober look at progress in language model reasoning: Pitfalls and paths to reproducibility. arXiv preprint arXiv:2504.07086, 2025.

Jian Hu. Reinforce++: A simple and efficient approach for aligning large language models. arXiv preprint arXiv:2501.03262, 2025.

Yutao Hu, Tianbin Li, Quanfeng Lu, Wenqi Shao, Junjun He, Yu Qiao, and Ping Luo. Omnimedvqa: A new large-scale comprehensive evaluation benchmark for medical lvlm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22170–22183, 2024.

Fanding Huang, Guanbo Huang, Xiao Fan, Yi He, Xiao Liang, Xiao Chen, Qinting Jiang, Faisal Nadeem Khan, Jingyan Jiang, and Zhi Wang. Beyond the exploration-exploitation trade-off: A hidden state approach for llm reasoning in rlvr. arXiv preprint arXiv:2509.23808, 2025.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Ahmadreza Jeddi, Hakki Can Karaimer, Hue Nguyen, Zhongling Wang, Ke Zhao, Javad Rajabi, Ran Zhang, Raghav Goyal, Babak Taati, and Radek Grzeszczuk. Puzzle curriculum grpo for vision-centric reasoning. arXiv preprint arXiv:2512.14944, 2025.

Jason J Lau, Soumya Gayen, Asma Ben Abacha, and Dina Demner-Fushman. A dataset of clinically generated visual questions and answers about radiology images. Scientific data, 5(1):1–10, 2018.

Jiazheng Li, Hongzhou Lin, Hong Lu, Kaiyue Wen, Zaiwen Yang, Jiaxuan Gao, Yi Wu, and Jingzhao Zhang. Questa: Expanding reasoning capacity in llms via question augmentation. arXiv preprint arXiv:2507.13266, 2025a.

Tianjian Li, Yiming Zhang, Ping Yu, Swarnadeep Saha, Daniel Khashabi, Jason Weston, Jack Lanchantin, and Tianlu Wang. Jointly reinforcing diversity and quality in language model generations. arXiv preprint arXiv:2509.02534, 2025b.

Yiwei Li, Peiwen Yuan, Shaoxiong Feng, Boyuan Pan, Bin Sun, Xinglin Wang, Heda Wang, and Kan Li. Turning dust into gold: Distilling complex reasoning capabilities from llms by leveraging negative data. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 18591–18599, 2024.

Bo Liu, Li-Ming Zhan, Li Xu, Lin Ma, Yan Yang, and Xiao-Ming Wu. Slake: A semantically-labeled knowledgeenhanced dataset for medical visual question answering. In 2021 IEEE 18th international symposium on biomedical imaging (ISBI), pages 1650–1654. IEEE, 2021.

Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864, 2025a.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025b.

Miao Lu, Han Zhong, Tong Zhang, and Jose Blanchet. Distributionally robust reinforcement learning with interactive data collection: Fundamental hardness and near-optimal algorithms. Advances in Neural Information Processing Systems, 37:12528–12580, 2024.

Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv preprint arXiv:2105.04165, 2021.

Lu Ma, Hao Liang, Meiyi Qiang, Lexiang Tang, Xiaochen Ma, Zhen Hao Wong, Junbo Niu, Chengyu Shen, Runming He, Yanhao Li, et al. Learning what reinforcement learning can’t: Interleaved online fine-tuning for hardest questions. arXiv preprint arXiv:2506.07527, 2025.

Gongrui Nan, Siye Chen, Jing Huang, Mengyu Lu, Dexun Wang, Chunmei Xie, Weiqi Xiong, Xianzhou Zeng, Qixuan Zhou, Yadong Li, et al. Ngrpo: Negative-enhanced group relative policy optimization. arXiv preprint arXiv:2509.18851, 2025.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Jiazhen Pan, Che Liu, Junde Wu, Fenglin Liu, Jiayuan Zhu, Hongwei Bran Li, Chen Chen, Cheng Ouyang, and Daniel Rueckert. Medvlm-r1: Incentivizing medical reasoning capability of vision-language models (vlms) via reinforcement learning. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 337–347. Springer, 2025.

Jinyoung Park, Jeehye Na, Jinyoung Kim, and Hyunwoo J Kim. Deepvideo-r1: Video reinforcement fine-tuning via difficulty-aware regressive grpo. arXiv preprint arXiv:2506.07464, 2025.

Benjamin Pikus, Pratyush Ranjan Tiwari, and Burton Ye. Hard examples are all you need: Maximizing grpo post-training under annotation budgets. arXiv preprint arXiv:2508.14094, 2025.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Yi Shen, Jian Zhang, Jieyun Huang, Shuming Shi, Wenjing Zhang, Jiangze Yan, Ning Wang, Kai Wang, Zhaoxiang Liu, and Shiguo Lian. Dast: Difficulty-adaptive slow-thinking for large reasoning models. arXiv preprint arXiv:2503.04472, 2025.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pages 1279–1297, 2025.

Yuqiao Tan, Minzheng Wang, Shizhu He, Huanxuan Liao, Chengfeng Zhao, Qiunan Lu, Tian Liang, Jun Zhao, and Kang Liu. Bottom-up policy optimization: Your language model policy secretly contains internal policies. arXiv preprint arXiv:2512.19673, 2025. URL https://arxiv.org/abs/2512.19673.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024a.

Renxi Wang, Haonan Li, Xudong Han, Yixuan Zhang, and Timothy Baldwin. Learning from failure: Integrating negative examples when fine-tuning large language models as agents. arXiv preprint arXiv:2402.11651, 2024b.

Zengzhi Wang, Fan Zhou, Xuefeng Li, and Pengfei Liu. Octothinker: Mid-training incentivizes reinforcement learning scaling. arXiv preprint arXiv:2506.20512, 2025.

Xumeng Wen, Zihan Liu, Shun Zheng, Shengyu Ye, Zhirong Wu, Yang Wang, Zhijian Xu, Xiao Liang, Junjie Li, Ziming Miao, et al. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. arXiv preprint arXiv:2506.14245, 2025.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement, 2024. URL https://arxiv.org/abs/2409.12122.

Xinhao Yao, Lu Yu, Xiaolin Hu, Fengwei Teng, Qing Cui, Jun Zhou, and Yong Liu. The debate on rlvr reasoning capability boundary: Shrinkage, expansion, or both? a two-stage dynamic view. arXiv preprint arXiv:2510.04028, 2025.

Edward Yeo, Yuxuan Tong, Xinyao Niu, Graham Neubig, and Xiang Yue. Demystifying long chain-of-thought reasoning in llms. In ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint

- arXiv:2503.14476, 2025.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint

- arXiv:2504.13837, 2025.

Enci Zhang, Xingang Yan, Wei Lin, Tianxiang Zhang, and Lu Qianchun. Learning like humans: Advancing llm reasoning capabilities via adaptive difficulty curriculum learning and expert-guided self-reformulation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 6630–6644, 2025a.

Jixiao Zhang and Chunsheng Zuo. GRPO-LEAD: A difficulty-aware reinforcement learning approach for concise mathematical reasoning in language models. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 5642–5665. Association for Computational Linguistics, November 2025. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.287. URL https://aclanthology.org/ 2025.emnlp-main.287/.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024.

Shijie Zhang, Guohao Sun, Kevin Zhang, Xiang Guo, and Rujun Guo. Clpo: Curriculum learning meets policy optimization for llm reasoning. arXiv preprint arXiv:2509.25004, 2025b.

Xiaoman Zhang, Chaoyi Wu, Ziheng Zhao, Weixiong Lin, Ya Zhang, Yanfeng Wang, and Weidi Xie. Pmc-vqa:

Visual instruction tuning for medical visual question answering. arXiv preprint arXiv:2305.10415, 2023. Xingjian Zhang, Siwei Wen, Wenjun Wu, and Lei Huang. Edge-grpo: Entropy-driven grpo with guided error

correction for advantage diversity. arXiv preprint arXiv:2507.21848, 2025c.

Yixiao Zhou, Yang Li, Dongzhou Cheng, Hehe Fan, and Yu Cheng. Look inward to explore outward: Learning temperature policy from llm internal states via hierarchical rl. arXiv preprint arXiv:2602.13035, 2026.

Zhanke Zhou, Xiangyu Lu, Chentao Cao, Brando Miranda, Tongliang Liu, Bo Han, and Sanmi Koyejo. Codapo: Confidence and difficulty-adaptive policy optimization for post-training language models. In 2nd AI for Math Workshop@ ICML 2025, 2025.

Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, and Yu Meng. The surprising effectiveness of negative reinforcement in LLM reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

### A. Related Works

### Appendix

Advantage Estimations in Reinforcement Learning. In Proximal Policy Optimization (PPO), advantage estimation relies on the Generalized Advantage Estimation (GAE Schulman et al. (2015)) framework. Nevertheless, the critic model required by GAE introduces non-trivial computational overhead. To mitigate this cost, GRPO Shao et al. (2024) and REINFORCE++ Hu (2025) replace the critic with lightweight baselines (batch-averaged rewards) or group-relative rewards for advantage computation. Building on this, several methods have pointed out and addressed the limitations of GRPO’s group relative advantage estimation. Dr.GRPO Liu et al. (2025b) enhances token efficiency by removing length and standard deviation normalization terms, while DAPO Yu et al. (2025) balances the policy gradient loss at the token level to mitigate the insufficient gradient contribution of long sequences in long CoT scenarios. However, the advantage symmetry phenomenon inherent in GRPO’s advantage estimation remains under-explored, with the existing literature lacking an in-depth analysis of this property.

Exploration-Exploitation Dilemma in Reinforcement learning with Verifiable Rewards (RLVR). Whether RLVR can genuinely expand the reasoning capabilities of LLMs has sparked extensive debate recently. Some studies Yue et al. (2025), Dang et al. (2025), He et al. (2025), Ma et al. (2025), Gandhi et al. (2025) argue that RLVR primarily enhances sampling efficiency at the cost of reduced diversity and exploration capacity, narrowing the model’s capability boundary—evidenced by its failure to improve Pass@k (e.g., pass@256). However, other approaches have effectively mitigated this limitation through prolonged training Liu et al. (2025a), negative learning Zhu et al. (2025), or curriculum learning Deng et al. (2025a), Li et al. (2025a), Tan et al. (2025), demonstrating that reinforcement learning can indeed yield novel reasoning strategies. In this work, we also observe that GRPO exhibits a performance drop at pass@256 compared to the base model, but our improved advantage estimation can partially offset this loss or even enhance the base model’s pass@k. We hope this work can offer new insights into this debate from the perspective of advantage estimation in GRPO.

### B. Supplementary Proof

- B.1. Proof of Equation 4 Proof. We begin by reviewing the objective function of GRPO under the notation used in our formulation.

∣oi∣

G

𝒥GRPO(θ) = Eq∼𝒬,{o

{min(ρi,t(θ)Ai,t,clip(ρi,t(θ),1 − ϵ,1 + ϵ)Ai,t)} − βDKL(πθ∣∣πref)],

i}iG=1∼πθold(⋅∣q)[

1 G

1 ∣oi∣

∑

∑

t=1

i=1

(14)

where ρi,t(θ) = ππθ(oi,t∣q,oi,<t)

θold(oi,t∣q,oi,<t), and β is the coefficient for the KL divergence term.

To better understand the model’s learning dynamics under the verifiable reward setting and to derive the specific reweighting formulation, we omit the regularization components (e.g., KL term & clipping operation) and focus on the core policy optimization objective:

G

𝒥GRPO(θ) = Eq∼𝒬,{o

i}iG=1∼πθold(⋅∣q) [

1 G

∑

i=1

∣oi∣

ρi,t(θ)Ai,t], (15)

1 ∣oi∣

∑

t=1

Taking the gradient with respect to θ:

∣oi∣

G

Ai,t ▽θ ρi,t(θ)]

▽θ𝒥GRPO(θ) = Eq∼𝒬,{o

i}iG=1∼πθold(⋅∣q) [

1 ∣oi∣

1 G

∑

∑

t=1

i=1

∣oi∣

▽θπθ(oi,t ∣ q, oi,<t) πθold(oi,t ∣ q, oi,<t)

G

]

i}iG=1∼πθold(⋅∣q) [

1 ∣oi∣

1 G

∑

∑

= Eq∼𝒬,{o

Ai,t

t=1

i=1

∣oi∣

πθ(oi,t ∣ q, oi,<t) πθold(oi,t ∣ q, oi,<t)

G

▽θ log πθ(oi,t ∣ q, oi,<t)]

i}iG=1∼πθold(⋅∣q) [

1 ∣oi∣

1 G

∑

∑

= Eq∼𝒬,{o

Ai,t

⎤

⎡

t=1

i=1

∣oi∣

G

ρi,t(θ)Ai,t weight

▽θ log πθ(oi,t ∣ q, oi,<t) SFT

1 G

1 ∣oi∣

∑

∑

. (16)

= Eq∼𝒬,{o

i}iG=1∼πθold(⋅∣q)

⎥⎦

⎢⎣

t=1

i=1

We complete the proof of eq. (4). As the weight equal to 1, GRPO degrade to the standard SFT function, thus it can be formulated as a reweighting variant of Supervised Fine-Tuning.

##### B.2. Proof of Equation 5

Given a group of sampled responses indexed by i ∈ {1, . . . , G}, partitioned into a set of correct trajectories 𝒢pos and incorrect trajectories 𝒢neg. Let the advantage Ai be computed via GRAE as:

G

G

(rj − µ)2. (17)

rj, σ = √⎷

ri − µ σ , where µ =

1 G

1 G

∑

∑

Ai =

j=1

j=1

Then, the sum of absolute advantages for correct trajectories equals that of incorrect trajectories: ∑

∣Ai∣ = ∑

∣Ai∣. (18)

i∈𝒢pos

i∈𝒢neg

Proof. First, we examine the sum of the standardized advantages over the entire group. By definition:

G

G

ri − µ σ

∑

∑

Ai =

i=1

i=1

G

G

1 σ (

µ)

∑

∑

=

ri −

i=1

i=1

G

G

1 σ (

rj)

1 G

∑

∑

=

ri − G ⋅

i=1

j=1

G

G

1 σ (

ri) = 0. (19)

∑

∑

=

ri −

i=1

i=1

This confirms the zero-sum property of the standardized advantages. Now, we decompose the summation into the two disjoint sets 𝒢pos and 𝒢neg (assuming the group contains mixed results, i.e., 𝒢pos ∪ 𝒢neg = 𝒢):

G

∑

Ai = ∑

Ai + ∑

Ai = 0. (20)

i=1

i∈𝒢pos

i∈𝒢neg

Rearranging the terms, we obtain:

∑

Ai = − ∑

Ai. (21)

i∈𝒢pos

i∈𝒢neg

In the context of verifiable rewards (e.g., binary outcomes r ∈ {0,1}), correct trajectories receive higher rewards than the group mean (ri > µ), implying Ai > 0 for i ∈ 𝒢pos. Conversely, incorrect trajectories receive lower rewards (ri < µ), implying Ai < 0 for i ∈ 𝒢neg. We have:

∣Ai∣ = ∑

∣Ai∣. (22)

∑

i∈𝒢pos

i∈𝒢neg

Thus, the magnitude of policy updates attributed to correct trajectories is strictly equivalent to that of incorrect ones.

##### B.3. Proof of Equation 6

To ensure a rigorous derivation, we first establish the definitions and notations within the context of the behavior space.

##### 1. Behavior Space and Set Definitions

- • Let ℬ = {b1, b2, . . . , bN} denote the Behavior Space, representing the universal set of all possible trajectories (responses) the model can generate, with size N.
- • Let 𝒢 ⊂ ℬ be the Sampled Group for the current query, consisting of the trajectories actually generated and evaluated in the current step. Let its size be G.
- • Let 𝒰 = ℬ \ 𝒢 be the Unsampled Set, representing the vast majority of potential trajectories not explored in this iteration.
- • Note: In the context of LLM reasoning, the space of possible sequences is combinatorially large (N → ∞) while the sample size is small (e.g., G = 8), implying ∣𝒰∣ ≫ ∣𝒢∣.

##### 2. Model Outputs in Behavior Space

- • Let h ∈ RN be the vector of Logits corresponding to the trajectories in ℬ. Specifically, hbi represents the unnormalized log-probability (score) of trajectory bi.
- • π(h) denotes the probability distribution over the behavior space, modeled via the Softmax function:

ehbi ∑Nj=1 ehbj

. (23)

πbi =

##### 3. Optimization Objective

• The objective function J(h) maximizes the expected return over the sampled group: J(h) = ∑

Aˆbk ln πbk(h). (24)

bk∈𝒢

Derivation of the Gradient Field. We seek to compute the gradient of J with respect to the logit hbi of an arbitrary trajectory bi. Applying the chain rule:

∂ ln πbk ∂hbi

∂J ∂hbi

= ∑

. (25)

Aˆbk

bk∈𝒢

Recall the standard derivative of the log-softmax function: ∂ln∂hπbk

= δik − πbi, where δik is the Kronecker delta (δik = 1 if bi = bk, else 0). Substituting this into Eq. (25):

bi

Aˆbk(δik − πbi). (26)

∂J ∂hbi

= ∑

bk∈𝒢

We decompose the summation into two distinct components:

= ( ∑

Aˆbkδik) − (πbi ∑ bk∈𝒢

Aˆbk). (27)

∂J ∂hbi

bk∈𝒢

The first term simplifies using the indicator function I(bi ∈ 𝒢), which isolates the advantage if bi is a sampled trajectory. The summation in the second term corresponds exactly to our defined constant C. Thus, the gradient is expressed as:

J = I(bi ∈ 𝒢)Aˆbi − Cπbi. (28)

▽hb

i

The Logits Dynamics. Finally, applying a gradient ascent update rule with learning rate η > 0, the update dynamics for the logit of any trajectory bi are governed by:

∆hbi = η [I(bi ∈ 𝒢)Aˆbi − Cπbi]. (29)

##### B.4. Proof of Theorem 2

Theorem 2 (Update Magnitude with respect to Sample Difficulty). Consider a group G of trajectories for one query (sample) with binary rewards {ri}iG=1, the sum of absolute advantages over the group under Group Relative Advantage Estimation (GRAE) can be derived as:

√p(1 − p). (30)

∣Ai∣ = 2∣G∣

∑

i∈G

where p = ∑i∈G ∣ri∣/∣G∣ denotes the empirical success probability of the corresponding sample within group G, and the proof can be found in section B.4.

Proof. Given a group G of trajectories with binary rewards ri ∈ {0,1}, let p = ∑i∈G ∣ri∣/∣G∣ denote the success probability. The group mean µ and standard deviation σ are:

µ = p, σ = √p(1 − p) (31) The advantage in GRAE is defined as Ai = (ri − µ)/σ. Summing the absolute advantages over the group G,

we partition the sum into successful (ri = 1) and failed (ri = 0) subsets:

∣Ai∣ = ∑ ri=1

1 − p √p(1 − p)

−p √p(1 − p)

+ ∑

∑

ri=0

i∈G

= (G ⋅ p)

+ (G ⋅ (1 − p))

1 − p √p(1 − p)

p √p(1 − p)

(32)

2Gp(1 − p) √p(1 − p)

=

√p(1 − p)

= 2∣G∣

This completes the proof, and a visualization of the relationship between update magnitude and sample difficulty is provided below, illustrating the derived curve:

[Figure 10]

- Figure 6: The Visualization of Update Magnitude with respect to Sample Difficulty. The update magnitude reaches its peak at p = 0.5, simple samples (e.g., p = 0.75) and hard samples (e.g., p = 0.25) exhibiting the same deviation from p = 0.5 are assigned identical importance weights.

##### B.5. The Logits Update of Negative-Dominant Group in Behavior Space

Theorem 1 (The Logits Update in Behavior Space) Assume that the set of all possible behaviors is ℬ = {bi}iN=1, consisting of sampled set 𝒢 and unsampled set 𝒰.The produced probability updates of path bi can be expressed as:

▽bi J = η ⋅ [I(bi ∈ 𝒢)Ai − Cπbi], (33)

where η denotes the learning rate and πbi is the model’s current sampling probability in behavior space for path bi.

Suppressing the advantages of correct paths leads to C < 0, and we subsequently examine the dynamics of probability updates across the following cases.

- Case A. For the sampled positive responses bi ∈ 𝒢pos: ∆hbi = η(Apos − Cπbi), (34)

where Apos > 0, (−Cπbi) > 0 thus ∆hbi > 0. Hence, the sampling probability of a correct response will strictly increase.

- Case B. For the sampled negative responses bi ∈ 𝒢neg: ∆hbi = η(Aneg − Cπbi), (35)

where Aneg < 0 but (−Cπbi) > 0. Compared to GRPO(C = 0) which directly penalizes the incorrect responses, the logits of overconfident negative responses can even be increased. As reinforcement learning sharpens

the output distribution He et al. (2025), Li et al. (2025b), Lu et al. (2024), Yue et al. (2025), overconfident erroneous responses occasionally displace correct ones in later stages, leading to a sharp surge in fully unsolved samples.

- Case C. For the unsampled response bi ∈ 𝒰:

∆hbi = η(−Cπbi) (36)

It can be observed that the output probabilities of all unsampled responses exhibit an increase, whereas they remain constant under GRPO. This phenomenon demonstrates that the Negative-Dominant group provides an exploration incentive, potentially facilitating the discovery of correct yet previously unsampled trajectories.

### C. Implementation Details

##### C.1. Dataset Introduction

Mathematical Reasoning We first conduct experiments on three widely recognized math benchmarks designed to evaluate the logical reasoning capabilities of LLM:

- • MATH Hendrycks et al. (2021): A comprehensive dataset containing challenging competition-level mathematics problems across various subjects.
- • AMC23: Problems derived from the 2023 American Mathematics Competitions, representing highschool-level competitive reasoning.
- • AIME 2025: The latest problems from the American Invitational Mathematics Examination, designed to test advanced problem-solving skills and long-chain reasoning.

Multi-modal Mathematical Reasoning To validate the universality of our method in Vision-Language Models (VLMs), we extend our evaluation to three multi-modal mathematical reasoning datasets:

- • Geo3K Lu et al. (2021): A dataset focused on geometry problem-solving that requires synchronized understanding of both visual diagrams and textual descriptions.
- • MathVerse Zhang et al. (2024): A comprehensive benchmark designed to evaluate visual mathematical reasoning across a wide array of subjects and visual patterns.
- • MathVision Wang et al. (2024a): A large-scale challenge consisting of diverse mathematical problems sourced from real-world competitions with complex visual contexts.

Medical Imaging Reasoning Finally, to verify whether the proposed method enhances the model’s ability to perceive and reason over fine-grained image details, we utilize the HuatuoGPT-Vision Chen et al. (2024)

dataset. This benchmark is a curated and combined dataset from several publicly available medical VQA benchmarks, including VQA-RAD Lau et al. (2018), SLAKE Liu et al. (2021), PathVQA He et al. (2020), OmniMedVQA Hu et al. (2024), and PMC-VQA Zhang et al. (2023). It is specifically designed to assess medical domain expertise and visual grounding capabilities in complex clinical scenarios. Specifically, the training set comprises 600 MRI image-question pairs, while the evaluation is conducted on a diverse test suite consisting of 300 MRI, 300 CT, and 300 X-ray pairs following Pan et al. (2025).

##### C.2. Hyperparameter Settings

For all reinforcement learning experiments, responses were generated with a temperature of 1.0 and a maximum completion length of 2048 tokens. The learning rate is 1e-6. During evaluation, we used a generation temperature of 0.6, a top-p value of 0.95, and set the maximum new tokens to 4096. To ensure stable results, we perform 16 runs for AIME2025, and AMC23, and 4 runs for other benchmarks, reporting the average performance across the respective runs. Note that we did not use a dynamic sampling strategy in DAPO to ensure fair overhead. Our experiments are conducted over a single node with 8 NVIDIA H200 GPUs.

For experiments trained on the MATH dataset, we used the following system prompt to guide the model’s reasoning process: “Please reason step by step, and put your final answer within \boxed{}.” The training batch size is set to 1024 with total 20 epochs. For each prompt, we generated 8 responses within the group. The reward was based on binary accuracy, where a correct final answer yielded a reward of 1 and an incorrect one yielded 0.

For experiments trained on the Geo3K dataset, we used the following system prompt to guide the model’s reasoning process: “This is a multiple-choice question. You FIRST think about the reasoning process as an internal monologue and then provide the final choice from the given options. The reasoning process MUST BE enclosed within tags." The training batch size is set to 512 with total 20 epochs. The final answer MUST BE put in \boxed{}.” For each prompt, we generated 5 responses within the group. The reward was based on 0.1*acc-reward + 0.9 * format-reward. As for the medical dataset, the training batch size is set to 32 with total 10 epochs, and the other settings are the same as those on Geo3K.

Our proposed A-GRAE method is highly accessible, requiring only a single hyperparameter: the scaling parameter α in Eq. (13). In our experiments, α is set to 1 for the Math dataset and 0.5 for the Geo3k and Medical datasets. This adjustment is motivated by our observation that multi-modal datasets exhibit a higher propensity for training collapse, thus necessitating a smaller scaling factor to maintain stability. Following prior works Hochlehnert et al. (2025), Chen (2021), Chen et al. (2025), Zhu et al. (2025), we set our primary evaluation metric as Pass@k. Pass@k estimates the probability of generating a correct solution within k attempts. Unlike greedy decoding, Pass@k provides a more reliable assessment of a model’s potential and capability boundaries Hochlehnert et al. (2025), Chen (2021), Chen et al. (2025), Zhu et al. (2025). We employ the standard unbiased estimator, which generates n responses (n ≥ k) for each question q, counts the correct responses c, and computes the metric as follows:

Pass@k = Eq∼𝒬 [1 − (n−kc) (nk)

] (37)

### D. Supplementary Experimental Results

##### D.1. Experimental Results of Control Experiment using Llama-3.2-3B-Instruct

It is observed that the empirical patterns on Llama-3.2-3B-Instruct remain largely consistent with those detailed in the main text. Specifically, suppressing the positive advantages at the group level significantly enhances reasoning performance, particularly in terms of the Pass@k metric. Conversely, no universally optimal difficulty re-weighting strategy exists at the sample level. It is noteworthy, however, that when k reaches 256, all evaluated methods underperform relative to the base model. This suggests that the backbone model itself often determines the efficacy of reinforcement learning, a finding consistent with recent literature Yeo et al. (2025), Zhu et al. (2025), Wang et al. (2025). Nevertheless, the Negative-Dominant group remains the most effective approach for preserving the fundamental capabilities of the base model.

[Figure 11]

[Figure 12]

[Figure 13]

- Figure 7: Experimental results on breaking group-level symmetry using Llama-3.2-3B-Instruct. The performance is evaluated using Pass@k (k = {1,2,4,8,16,32,64,128,256}) to provide a comprehensive analysis of the model’s capabilities.

[Figure 14]

[Figure 15]

[Figure 16]

- Figure 8: Experimental results on breaking sample-level symmetry using Llama-3.2-3B-Instruct. The performance is evaluated using Pass@k (k = {1,2,4,8,16,32,64,128,256}) to provide a comprehensive analysis of the model’s capabilities.

- D.2. Supplementary Experimental Results using DeepSeek-R1-7B
- D.3. Training Collapse of Negative-Dominant Group

To further investigate the reliability of the Negative-Dominant configuration, we conducted multiple training runs under identical hyperparameter settings. As illustrated in fig. 9, we observe a striking stochastic instability within this specific group. While one trial remains relatively stable throughout the training

- Table 5: Pass@k results on MATH, AIME 2025 and AMC23 with DeepSeek-R1-7B. Bold and underlined numbers denote the best and second-best results for each k.

Method Pass@k k 1 2 4 8 16 32 64 128 256

MATH Base Model 84.3 89.6 92.8 94.8 96.0 96.8 97.4 97.8 98.0 GRPO 92.4 95.3 96.5 97.0 97.3 97.7 98.0 98.3 98.6 GRPO + A-GRAE 92.9 95.7 96.9 97.5 97.9 98.3 98.6 99.0 99.3 DAPO 92.6 95.3 96.4 97.2 97.5 97.8 98.0 98.2 98.6 DAPO + A-GRAE 93.0 95.6 97.0 97.5 98.1 98.2 98.4 98.7 99.3 Dr.GRPO 92.8 95.6 96.8 97.3 97.7 98.0 98.3 98.5 98.8 Dr.GRPO + A-GRAE 93.3 96.0 97.3 97.8 98.3 98.6 98.8 99.0 99.4

AIME 2025 Base Model 22.7 26.5 29.7 33.2 37.8 43.2 48.0 51.1 53.3 GRPO 28.5 35.0 41.4 47.7 53.1 57.6 62.1 67.2 73.3 GRPO + A-GRAE 29.6 36.0 42.6 48.9 54.0 58.3 63.0 68.3 74.3 DAPO 29.2 35.5 42.2 48.5 53.3 57.7 62.5 67.6 73.6 DAPO + A-GRAE 30.3 36.8 43.2 49.3 54.6 58.6 63.3 68.4 75.0 Dr.GRPO 29.0 35.2 41.8 48.2 53.0 57.5 62.3 67.3 74.0 Dr.GRPO + A-GRAE 29.8 36.3 43.3 49.8 55.2 58.8 63.5 68.8 74.6

AMC23 Base Model 64.2 73.0 80.1 85.5 89.8 92.8 95.2 97.4 100.0 GRPO 82.8 89.3 92.3 94.0 95.2 96.1 97.2 98.8 100.0 GRPO + A-GRAE 84.2 90.6 93.5 95.4 96.6 97.7 98.0 99.3 100.0 DAPO 83.3 89.8 92.6 94.3 95.8 96.5 97.6 99.2 100.0 DAPO + A-GRAE 84.6 91.6 94.2 96.3 97.0 97.8 98.2 99.5 100.0 Dr.GRPO 83.0 89.6 92.3 93.8 95.6 96.3 97.2 98.2 100.0 Dr.GRPO + A-GRAE 84.5 91.3 93.6 96.0 97.2 98.0 98.5 99.3 100.0

process, the other undergoes a catastrophic training collapse after approximately step 78. This divergence highlights that the Negative-Dominant approach is highly sensitive to initial conditions or sampling noise. In the "collapse" instance, the number of fully unsolved questions exhibits high-amplitude oscillations and a persistent upward trend, eventually peaking at over 110. This suggests that without proper regularization, the Negative-Dominant objective can easily lead the model into a divergent optimization path, where erroneous trajectories reinforce themselves and eventually overwhelm the model’s reasoning capabilities. Such unpredictable behavior underscores the necessity of balancing negative trajectory weights to ensure consistent convergence.

- D.4. Experimental Results in Further Analysis
- D.5. Comparative Analysis with Control Groups

Effectiveness of Attenuation Suppression Strategy (ASS). As shown in table 7, our proposed ASS method is comparable with the Negative-Dominant group in all Pass@k metrics. Furthermore, we conducted 10 independent training runs from scratch for each group to evaluate their empirical stability. Our results

[Figure 17]

[Figure 18]

#### Figure 9: Training collapse of Negative-Dominant group during training with Qwen2.5-Math-7B. Left Part: The unsolved question count suddenly increase in the later stage. Right Part: The collapse leads to the significant performance decrease.

- Table 6: Ablation studies on MATH, AIME 2025 and AMC23 with Qwen2.5-Math-7B.

Method Pass@k k 1 2 4 8 16 32 64 128 256

MATH Base Model 63.4 74.8 83.2 88.6 91.2 93.4 94.1 95.0 96.3 GRPO 76.5 82.3 86.1 88.8 90.3 92.6 93.5 93.9 95.0 GRPO + A-GRAE (sample level) 77.8 84.2 88.4 90.5 92.1 94.0 94.3 94.8 96.0 GRPO + A-GRAE (group level) 77.6 84.0 88.1 91.2 93.0 94.3 95.3 95.6 97.0 GRPO + A-GRAE (full) 78.3 85.0 89.2 91.0 92.5 94.6 95.0 95.5 96.5

AIME 2025 Base Model 6.1 9.9 14.4 19.3 24.4 29.1 33.4 39.2 46.7 GRPO 10.3 14.3 18.7 23.1 27.5 31.8 36.1 40.8 46.7 GRPO + A-GRAE (sample level) 10.9 15.1 19.6 24.1 28.0 33.5 40.0 47.0 52.3 GRPO + A-GRAE (group level) 11.0 15.4 20.3 24.5 28.3 33.9 41.0 49.6 60.0 GRPO + A-GRAE (full) 11.3 15.6 19.8 24.7 28.6 34.1 39.2 47.8 56.7

AMC23 Base Model 40.6 55.3 68.6 78.6 85.0 89.4 93.4 97.3 100.0 GRPO 59.2 66.7 72.1 76.4 80.6 84.8 88.3 90.8 92.5 GRPO + A-GRAE (sample level) 62.3 70.4 76.8 83.0 86.3 90.4 93.2 94.3 95.0 GRPO + A-GRAE (group level) 61.4 69.8 77.0 82.3 87.0 90.8 94.4 97.3 100.0 GRPO + A-GRAE (full) 62.6 70.0 77.5 83.7 88.2 92.0 95.1 96.8 97.5

reveal that the Negative-Dominant group experienced the learning collapse phenomenon—as characterized in section D.3—in three out of the ten trials, whereas our proposed method exhibited no such failure cases. This demonstrates that our approach effectively maintains training stability while simultaneously facilitating model exploration.

##### Effectiveness of Dynamic Difficulty Attention Shift (DDAS). To verify that our dynamic difficulty attention

[Figure 19]

[Figure 20]

[Figure 21]

- Figure 10: Training dynamics using Qwen2.5-Math-7B. Left Part: Model’s entropy on the Math training set. Center Part: Model’s entropy (actor entropy loss) on the Math test set. Right Part: The greedy decoding accuracy on the Math test set.

shift truly improve the reasoning performance in comparison with the fixed difficulty preference in Control Experiment II, we demonstrate the results in table 7. It is observed that DDAS achieves substantial gains in the Pass@1 metric across all datasets. Furthermore, its performance remains competitive with Hard-Focused as k increases. These results demonstrate that our method, by incorporating a dynamic shift in difficulty-aware attention, effectively integrates the advantages of both Easy-Focused and Hard-Focused groups, thereby facilitating consistent performance improvements.

- Table 7: Performance comparison of A-GRAE against control groups on MATH, AIME 2025, and AMC23 datasets using Qwen2.5-Math-7B. Cells shaded in light red and light green denote Control Experiment I and II, respectively.

Method Pass@k k 1 2 4 8 16 32 64 128 256

MATH Base Model 63.4 74.8 83.2 88.6 91.2 93.4 94.1 95.0 96.3 GRPO 76.5 82.3 86.1 88.8 90.3 92.6 93.5 93.9 95.0 Positive-Dominant 75.5 80.1 83.2 85.4 88.5 90.1 91.2 92.0 93.2 Negative-Dominant 77.2 83.8 87.6 90.4 92.8 94.1 95.2 95.8 97.0 GRPO + ASS (group level) 77.6 84.0 88.1 91.2 93.0 94.3 95.3 95.6 97.0 Easy-Focused 77.0 82.8 86.4 89.2 91.0 92.5 93.8 94.2 95.0 Hard-Focused 75.8 81.2 86.0 89.8 92.2 93.6 94.8 95.2 96.0 GRPO + DDAS (sample level) 77.8 84.2 88.4 90.5 92.1 94.0 94.3 94.8 96.0

AIME 2025 Base Model 6.1 9.9 14.4 19.3 24.4 29.1 33.4 39.2 46.7 GRPO 10.3 14.3 18.7 23.1 27.5 31.8 36.1 40.8 46.7 Positive-Dominant 9.7 13.6 17.1 20.3 23.7 27.1 30.9 35.9 40.0 Negative-Dominant 10.8 15.0 20.1 23.8 27.7 34.2 40.9 49.2 60.0 GRPO + ASS (group level) 11.0 15.4 20.3 24.5 28.3 33.9 41.0 49.6 60.0 Easy-Focused 9.7 13.9 18.6 23.7 27.6 31.1 36.8 39.9 46.7 Hard-Focused 11.8 14.6 19.4 24.5 28.0 33.0 36.9 41.4 50.3 GRPO + DDAS (sample level) 10.9 15.1 19.6 24.1 28.0 33.5 40.0 47.0 52.3

AMC23 Base Model 40.6 55.3 68.6 78.6 85.0 89.4 93.4 97.3 100.0 GRPO 59.2 66.7 72.1 76.4 80.6 84.8 88.3 90.8 92.5 Positive-Dominant 59.2 66.7 72.1 76.4 80.6 84.8 88.3 90.8 92.5 Negative-Dominant 60.9 69.5 76.6 82.1 86.5 90.6 94.1 97.3 100.0 GRPO + ASS (group level) 61.4 69.8 77.0 82.3 87.0 90.8 94.4 97.3 100.0 Easy-Focused 61.5 68.8 72.3 76.4 79.4 82.3 88.4 91.8 95.0 Hard-Focused 60.8 69.9 75.6 82.5 86.8 91.3 93.5 96.2 100.0 GRPO + DDAS (sample level) 62.3 70.4 76.8 83.0 86.3 90.4 93.2 94.3 95.0

