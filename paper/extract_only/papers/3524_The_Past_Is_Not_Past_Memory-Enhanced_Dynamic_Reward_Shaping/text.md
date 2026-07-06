# arXiv:2604.11297v1[cs.LG]13Apr2026

OpenMOSS

[Figure 1]

## The Past Is Not Past: Memory-Enhanced Dynamic Reward Shaping

### Yang Liu1,2,∗, Enxi Wang1,∗, Yufei Gao1,∗, Weixin Zhang1,∗, Bo Wang1, Zhiyuan Zeng1, Yikai Zhang1, Yining Zheng1,2,†, Xipeng Qiu1,2,†

1Fudan University 2Shanghai Innovation Institute

### Abstract

Despite the success of reinforcement learning for large language models, a common failure mode is reduced sampling diversity, where the policy repeatedly generates similar erroneous behaviors. Classical entropy regularization encourages randomness under the current policy, but does not explicitly discourage recurrent failure patterns across rollouts. We propose MEDS, a MemoryEnhanced Dynamic reward Shaping framework that incorporates historical behavioral signals into reward design. By storing and leveraging intermediate model representations, we capture features of past rollouts and use density-based clustering to identify frequently recurring error patterns. Rollouts assigned to more prevalent error clusters are penalized more heavily, encouraging broader exploration while reducing repeated mistakes. Across five datasets and three base models, MEDS consistently improves average performance over existing baselines, achieving gains of up to 4.13 pass@1 points and 4.37 pass@128 points. Additional analyses using both LLM-based annotations and quantitative diversity metrics show that MEDS increases behavioral diversity during sampling.

Github: https://github.com/Linxi000/MEDS

[Figure 2]

### 1 Introduction

With the advancement of the fundamental capabilities of Large Language Models (LLMs), reinforcement learning has achieved significant success across various domains [1–3]. By incorporating reward signals, whether derived from rule-based evaluation [1] or proxy models [4], LLMs iteratively alternate between a sampling phase and a gradient-based optimization phase. As the model’s performance is optimized toward maximizing the expected return, designing the reward scoring structure becomes the primary way to guide the model’s behavior [5].

A well-known failure mode of such on-policy optimization is that, as training progresses, the policy often collapse into a narrow and stereotyped set of behaviors [6–8]. This degeneration produces highly repetitive responses that not only weaken effective exploration and waste on-policy samples, but also entrench the model in self-reinforcing erroneous reasoning trajectories[9]. A series of works [10–12] introduce entropy regularization to promote output diversity by injecting randomness at the level of the current policy distribution. However, in LLM reasoning, such distribution-level stochastic exploration is often insufficient

‗These authors contributed equally. †Corresponding authors.

[Figure 3]

Early Exploration Later Error Collapse

Within-Step Diversity

[Figure 4]

[Figure 5]

Early Exploration Later Error Collapse

Early Exploration Later Error Collapse

- Step k1
- Step k2
- Step k3 Step …

Within-Step Diversity

Within-Step Diversity

|StepkMi1|k1<br><br>sreM|adisreadMisreadproblemproblemproblemsetupsetus|etupp| |
|---|---|---|---|---|

Step

Step

|Stepk2<br><br>W|k2<br><br>ro|ngWrongWrongsetupsetupparsingsetupparsingpars|ing| |
|---|---|---|---|---|

Question X

Question X

Question X

Question X

###### Question X

Question X

- Step t1
- Step t2

- Step t1
- Step t2

- Step t1
- Step t2

Step tn

Step tn

Step tn

Step

|Stepk3<br><br>Setu|k3<br><br>Sep|understandingtupSetupunderstandingunderstandingerrorer|rorerr|or|
|---|---|---|---|---|

Policy

Step t1

Policy

Policy

Policy

Step t1

Step t1

Policy

Policy

Step tn

Step tn

Step tn

Step …

Step …

| | |………………………| | |
|---|---|---|---|---|

Step t1

Step t1

Step t1

Policy

Policy

Policy

Step tn+1

Step tn+1

Step tn+1

[Figure 6]

[Figure 7]

[Figure 8]

Step kn

Step

|Stepkn<br><br>Fa|kn<br><br>ul|tyFaultysetupFaultysetupreadingsetupreadingread|ing| |
|---|---|---|---|---|

Across-Step Diversity

Across-Step Diversity

Across-Step Diversity

Step tn+1

Step tn+1

Step tn+1

Step t2

Step t2

Step t2

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Stable Error Basin Distribution

Stable Error Basin Distribution

Stable Error Basin Distribution

High Entropy Policy Distribution

High Entropy Policy Distribution

High Entropy Policy Distribution

|||policy<br><br>Shared logical failure produced repeatedly by the<br><br>Shared logical failure produced repeatedly by the policy<br><br>Shared logical failure produced repeatedly by the policy|
|---|
<br><br>re|
|---|
<br><br>r<br><br>f repe|
|---|

||policy<br><br>Shared logical failure produced repeatedly by the<br><br>Shared logical failure produced repeatedly by the policy<br><br>Shared logical failure produced repeatedly by the policy|
|---|
<br><br>re|
|---|

y

(a)

(b)

###### (c)

- Figure 1 Illustration of error collapse in on-policy optimization and its effect on output diversity. (a) During on-policy optimization, the policy evolves from a high-entropy distribution into a narrow, stable error basin. (b) Different errors share the same failure mode and become progressively reinforced during training. (c) Comparison of output diversity between DAPO and our method, with detailed description provided in Section 4.3.

###### a.

###### a.

###### a.

b.

b.

b.

c.

c.

c.

due to the vast action space. Moreover, as illustrated in Figure 1, once training drifts into a particular error pattern, the policy repeatedly reproduce the same mistake by sampling slightly different reasoning expressions, remaining trapped in this pattern despite continued optimization over many steps.

Inspired by the dynamic reward mechanisms observed in human learning [13], where individuals impose stronger psychological penalties on recurring mistakes to reinforce avoidance, we propose a MemoryEnhanced Dynamic reward Shaping framework (MEDS). Unlike static reward functions, our method dynamically records historical error patterns and imposes incremental penalties on repetitive failure paths. This mechanism not only facilitates the correction of persistent errors but also encourages the model to escape local optima, thereby enhancing exploration in complex solution spaces. We further provide theoretical analysis showing that additional penalties on repeated errors can improve performance.

In practice, by reusing the layer-wise logits already produced during the forward pass, we construct feature vectors of the model’s implicit reasoning trajectory. This reuse introduces almost no additional computational overhead and serves as a lightweight yet effective representation. We then maintain an error memory that is updated online throughout training. By incorporating HDBSCAN-based clustering [14], this memory adaptively captures error types and their density structure, enabling the training objective to adjust penalty strength according to the recurrence of failure patterns.

Empirical results across multiple benchmarks show that our method improves both pass@1 and pass@128 across all three models, with gains of up to 4.13 points in pass@1 and 4.37 points in pass@128. The largest pass@128 improvement is observed on Qwen3-8B, where performance increases from 70.81 to 82.67 on a single dataset, corresponding to a relative gain of 17%. We also analyze changes in the model’s exploration behavior, demonstrating that our method increases the diversity of exploration. In addition, we calculate the consistency between logit-based clustering and LLM-based clustering, showing that logits can serve as a proxy for reasoning structure. The positive correlation between clustering quality and downstream model performance further validates the effectiveness of our approach. Our main contributions are summarized as follows:

- 1. We propose MEDS, a novel memory-enhanced dynamic reward shaping framework for LLM reinforcement learning. To the best of our knowledge, this is the first work to explicitly incorporate historical error patterns into reward modeling, enabling the policy to recognize and avoid recurrent failure behaviors rather than merely increasing randomness under the current policy. This design provides more informative supervision during training and effectively mitigates repeated errors.
- 2. We provide theoretical analysis supporting the effectiveness of memory-based dynamic penalties in improving learning and mitigating recurrent failure modes.

- 3. We introduce a representation-level criterion for response similarity that leverages layer-wise logits as a compact and efficient proxy, with little additional computational overhead.
- 4. We conduct extensiveexperiments across multiple datasets and models, showing that MEDS consistently achieves the best overall performance and substantially improves both pass@1 and pass@128. Further analyses show that our method increases exploration diversity, that logit-based clustering aligns well with LLM-based semantic clustering, and that better recognition of repeated error patterns is strongly correlated with downstream performance gains.

### 2 Related Work

##### 2.1 Reinforcement Learning with Verifiable Rewards

The application of reinforcement learning in LLMs relies heavily on the design of reward signals. And Reinforcement Learning with Verifiable Rewards (RLVR) has become the dominant paradigm due to the simplicity of its reward signals. Representative works adopt diverse verifiable criteria across tasks. OpenAI [15] and Shao et al. [1] assign binary rewards based on exact answer matching. Zhu et al. [16] automatically verifies correctness by executing generated programs in a sandbox. Zhang et al. [17] uses both temporal localization quality and multiple-choice Video QA accuracy as reward signals. Chen et al. [18] employs a multimodal large language model (MLLM) to evaluate similarity between generated and target images. Zhang et al. [19] adopts metrics such as nDCG@10 as the primary reward. In all these works, the verifier operates independently within each epoch and assigns rewards based solely on current samples, overlooking useful information in historical trajectories. Our approach instead leverages trajectories from earlier stages to guide later sampling.

##### 2.2 Reward Shaping for Diversity Exploration

During RLVR training, policy updates tend to over-optimize dominant solution patterns, leading to collapse of reasoning diversity [20, 21]. Traditional approaches mitigate this by constraining policy shift via KL regularization [4, 22] or encouraging randomness via entropy regularization [10].

Beyond implicit regularization, several works explicitly reward rare reasoning behaviors. Hu et al. [23] clusters rollouts by solution strategy and inversely weights advantages by cluster size. Wan et al. [11] promotes diversity among correct trajectories with token-level entropy regularization. However, these methods operate only on current-batch differences and do not track solution patterns over long-term training history.

To leverage historical information, Song et al. [24] introduces UCB-style exploration rewards based on historical outcome frequencies. Yet it still lacks fine-grained cross-step reasoning behavior tracking. Another line of work [25–29] uses external models to adaptively adjust rewards based on historical performance, but the additional computational cost limits scalability. In contrast, our method captures reasoning patterns via layer-wise logits, enabling history-aware reward adjustment with minimal overhead.

##### 2.3 Inner Thoughts of LLMs

Works that attempt to understand and utilize how models reason and arrive at their final conclusions are an important source of inspiration for our approach. Meng et al. [30] localizes factual knowledge in the model by examining logit changes. Subsequent studies [31–34] use Sparse Autoencoders (SAEs) and related logit-intervention techniques to obtain a more fine-grained understanding of LLM reasoning processes. These efforts have further motivated the use of logits for analysis and supervision. For example, Zhao et al. [35] detects erroneous reasoning by verifying the inner attribute graph, while Tan et al. [36] leverages layer-wise logits to provide more fine-grained supervision.

A key component of our work is to identify shared patterns across different reasoning trajectories. Inspired by prior work on logit-based model analysis, we cluster logits to capture such patterns without requiring extra models. We describe this approach in detail in Section 3.3.

[Figure 15]

###### A. Logic Feature Extraction B. Memory-based Clustering C. Reward Shaping

y*

[Figure 16]

[Figure 17]

HDBSCAN Cluster

[Figure 18]

Error Memory

[Figure 19]

Gx

Task Specific Reward Model

[Figure 20]

[Figure 21]

l*(N)

[Figure 22]

[Figure 23]

C1

[Figure 24]

r(x,y˜)

C2

Transformer layer N

l*(N−1)

[Figure 25]

Feature Vector Construction

f(y1) f(y12)

[Figure 26]

[Figure 27]

Transformer layer N-1

previous

[Figure 28]

Cnoise

[Figure 29]

[Figure 30]

l*(N/2)

Ck

[Figure 31]

[Figure 32]

Reward Shaping

[Figure 33]

pθ

Transformer layer N/2

f(ytb−1) f(˜ytb)

f(˜y) ∈ Ck

r˜(x,y˜) = r(x,y˜) − penalty

l*(2) l*(1)

[Figure 34]

[Figure 35]

Transformer layer 2

[Figure 36]

[Figure 37]

[Figure 38]

l*(N/2) l*(N/2+1) ⋮ l*(N)

[Figure 39]

[Figure 40]

[Figure 41]

Transformer layer 1

| |
|---|

penalty =

f(˜y) =

[Figure 42]

min(α log(|Ck| + 1),β)

x ∼ D

{ y1, y2, y3,…, yL−t−1,…, yL } Pold(˜y|x)

Pnew(y|x)

y˜ =

- Figure 2 Overview of MEDS. The process consists of three stages: (A) Logic Feature Extraction: For an input 𝑥, we sample a response 𝑦˜ and extract a logic feature vector 𝑓 (˜𝑦) from the layer-wise logits of the first token 𝑦∗ in the final answer, using the latter half of the Transformer layers to capture reasoning patterns. (B) Memory-based Clustering: We maintain a per-prompt error memory 𝐺𝑥 and apply HDBSCAN to group historical responses with similar logic features.

(C) Reward Shaping: For a response assigned to cluster 𝐶𝑘, we compute a penalty min(𝛼 log(|𝐶𝑘| + 1), 𝛽) proportional to the cluster size, and adjust the reward as 𝑟˜(𝑥, 𝑦˜) = 𝑟(𝑥, 𝑦˜) − penalty, which is then used to update the policy.

- 3 Method

The first token of the final answer.

[Figure 43]

Logic Feature extraction

Reward Shaping

Sample

Error Policy Update

[Figure 44]

response reward

##### 3.1 Preliminaries

Given an input x ∼ 𝐷, the LLM is modeled as a policy 𝑝𝜃 that generates a response sequence y ∼ 𝑝𝜃(y | x), where y = {𝑦1, 𝑦2, . . . , 𝑦𝐿} ∈ 𝒴 represents the complete response and 𝑦𝑡 denotes the token generated at step 𝑡. The response is evaluated and mapped to a scalar reward via a reward function 𝑟 : 𝒳 × 𝒴 → ℝ. In practical optimization, a KL-regularized objective is adopted, and we maximize the expected return within a trust region. The corresponding training objective is

l*(N/2) l*(N/2)

logits

l*(N/2)

l*(n) ∈ ℝ

1 𝜂

KL 𝑝𝜃(· | x)∥ 𝑝ref(· | x)

max

𝔼x∼𝐷 𝔼y∼𝑝𝜃(·|x) 𝑟(x, y) −

𝜃

where 𝑝ref is the reference policy that constrains the update space, and 𝜂 is the regularization coefficient controlling the strength of the constraint.

###### 3.2 Theoretical Benefits of Penalizing Repeated Errors With the mathematical formulation above, we theoretically analyze the benefits of penalizing repeated errors.

Theorem 1. Suppose there are two different reward signals, 𝜇1 = 𝑟(x, y) and 𝜇2 = 𝑟(x, y) − 𝜆𝑐(y), and let 𝑞1 and 𝑞2 denote the corresponding updated policies under these reward signals. Let 𝐽(𝑞) = 𝔼x,y∼𝑞[𝑟(x, y)]. Then it holds that

𝐽(𝑞2) ≥ 𝐽(𝑞1),

where 𝜆 is the repetition penalty coefficient. 𝑐(y) : 𝑋 × 𝑌 → ℕ denotes the indicator of the number of repeated errors, where a value greater than zero indicates that y corresponds to an error that has been repeatedly sampled in the past. The function 𝑟(x, y) is a task-specific scoring function. The function 𝑐(y) increases monotonically with the number of occurrences.

We provide a detailed proof in Appendix B. Intuitively, repeatedly sampled errors indicate higher probability mass under the current policy, and adding an extra penalty helps reallocate this probability. Since the penalty is applied to repeated errors, the expected return is non-decreasing from the perspective of optimization. The

Transformer Layer 3

- Layer 1

Transformer

- Layer 2

1344

3-layer logits

numbers divisors 2022 smallest multiply 80 smaller

1342

| | |
|---|---|
| | |

| | |
|---|---|
| | |

1344

number divisor 2022 small

| | |
|---|---|
| | |

numbers

Figure 3 The logits aggregation process, exemplified by a 3-layer model feature graph. Each layer contributes a scalar logit at the final answer token position, whose value is determined by the activated computational circuits. The left and middle graphs activate nearly identical circuits, resulting in nearly identical 3-layer logit vectors. In contrast, the right graph exhibits a substantially different circuit structure, yielding divergent logits.

indicator function acts as a memory for the reward model and adjusts the score based on the historical sampling behavior of the policy.

- 3.3 Logic Feature Extraction

Transformer

Embeddings

When implementing the indicator function 𝑐(y), the key aspect is to handle the relationships among different responses. To do so efficiently, we directly reuse the layer-wise logits of each response.

Specifically, suppose the final answer tokens are {𝑦𝐿−𝑡, . . . , 𝑦𝐿} ⊂ {𝑦1, . . . , 𝑦𝐿}, and let 𝑦∗ = 𝑦𝐿−𝑡 denote the first token of the final answer. The corresponding logits at each layer 𝑛 are 𝑙𝑦(𝑛𝐿−)𝑡 ∈ ℝ𝑉, where V is the size of vocabulary. We select the logit corresponding to this token position, denoted as 𝑙∗(𝑛) ∈ ℝ, and aggregate them across layers. The evolution of logits for the final answer depends on feature activations across different layers [31], and therefore contains information about the model’s internal reasoning process.

- Figure 3 provides an intuitive illustration of this process. We also present further examples in Section 4.4, showing that this correlation construction captures more fundamental similarities than approaches based solely on surface-level string matching. The logits vector can be processed in several ways. The two most straightforward approaches are direct aggregation and aggregation after differencing. For clarity, we adopt the direct aggregation approach in our description here and compare the performance of several alternatives in Section 4.6. Since earlier layers typically model simpler semantic information [37], we use the logits from the latter half of the layers. The resulting feature representation is

𝑓 (y) = concat 𝑙∗(𝑛) | 𝑛 =

𝑁 2

, . . . , 𝑁 ∈ ℝ𝑁

2 .

##### 3.4 Cluster-based Reward Shaping

Based on the constructed response representations, we can directly compute the indicator function and perform the corresponding reward shaping. For each specific prompt x, we maintain a set of feature representations for all responses sampled across historical rollouts:

𝐺x = { 𝑓 (y11), 𝑓 (y12), . . . , 𝑓 (y𝑡𝑏)},

where 𝑏 denotes the number of responses sampled for each prompt and 𝑡 denotes the training epoch. We then apply HDBSCAN clustering on this set to obtain

𝐺x = {𝐶1, 𝐶2, . . . , 𝐶𝐾, 𝐶noise}, where 𝐶𝑖 denotes the 𝑖-th cluster, and the number of clusters 𝐾 is determined dynamically. For a specific response y˜, let 𝐶𝑘 denote the cluster to which 𝑓 (˜y) is assigned, where |𝐶𝑘| denotes the size

AIME24 AMC23 MATH500 Minerva OlympiadBench Average pass@1 pass@128 pass@1 pass@128 pass@1 pass@128 pass@1 pass@128 pass@1 pass@128 pass@1 pass@128

Method

Qwen3-1.7B 22.58 66.67 58.57 87.50 80.86 95.20 37.82 63.6 41.97 64.15 48.36 75.42 +GRPO 18.20 46.67 51.58 92.50 79.58 95.60 38.62 65.81 40.19 70.37 45.63 74.19 +DAPO 13.46 63.33 57.65 95.00 80.25 96.60 39.45 69.49 42.20 77.04 46.6 80.29 +GRPO w/ Entropy Adv 12.50 50.00 51.80 95.00 78.72 94.00 38.91 62.87 38.47 66.96 44.08 73.77 +MEDS 23.65 70.00 65.51 95.00 84.66 97.40 41.77 68.75 46.86 77.63 52.49 81.76

- Qwen2.5-Math-7B 6.43 56.67 19.82 92.50 30.84 96.00 6.62 63.24 9.27 70.07 14.60 75.70 +GRPO 23.85 66.67 70.08 97.50 84.10 96.20 39.07 73.16 42.57 73.63 51.93 81.43 +DAPO 32.27 70.00 72.71 97.50 85.61 96.00 43.01 75.74 44.43 75.00 55.61 82.85 +GRPO w/ Entropy Adv 29.74 63.33 70.64 97.50 84.50 96.20 40.68 70.96 42.57 73.63 53.63 80.32 +MEDS 34.32 76.67 74.38 97.50 86.33 96.00 42.51 74.26 44.80 75.56 56.47 84.00

- Qwen3-8B 34.51 70.00 65.72 90.00 83.62 95.40 40.90 61.03 44.69 61.63 53.89 75.61 +GRPO 22.45 70.00 58.79 90.00 82.38 95.40 44.07 64.71 44.22 69.78 50.38 77.98 +DAPO 45.42 73.33 81.37 95.00 89.18 96.60 46.82 65.44 52.77 70.81 63.11 80.24 +GRPO w/ Entropy Adv 27.32 63.33 65.49 92.50 85.89 96.00 45.32 65.44 49.92 70.67 54.79 77.59 +MEDS 45.78 76.67 82.62 97.50 92.51 98.20 51.58 68.02 61.12 82.67 66.72 84.61

Table 1 Main results on five mathematical reasoning benchmarks. Best and second-best are marked with bold and underline; MEDS rows are highlighted.

of the cluster. The corresponding indicator is defined as 𝑐(˜y) = log(|𝐶𝑘| + 1), a monotonically increasing transformation of cluster size that preserves the ordering used in the theoretical analysis in Appendix B. The reward is then adjusted as follows:

𝑟˜(x, y˜) = 𝑟(x, y˜) − min 𝛼 log(|𝐶𝑘| + 1), 𝛽 , s.t. 𝑓 (˜y) ∈ 𝐶𝑘. where 𝛼 and 𝛽 are hyperparameters controlling the penalty strength and the upper bound of the penalty.

### 4 Experiments

##### 4.1 Experimental Setup

Training Details We train on three base models: Qwen3-1.7B, Qwen3-8B, and Qwen2.5-Math-7B. Qwen3-8B and Qwen2.5-Math-7B represent models with and without explicit reasoning processes, while Qwen3-1.7B serves as a model with a different scale. The training corpus is constructed by merging the DAPO-Math-17K dataset [6] with the levels 3-5 of the MATH subdataset [38, 39]. All problems are formatted using the Qwen-Math template (Appendix A.1). All reinforcement learning training is implemented based on the verl framework [40].

Building on DAPO, our work further introduces a clustering penalty term: for each prompt, we maintain a memory of the logits representations of incorrect samples, where we concatenate the logits from the last 14 Transformer layers and apply L2 normalization, then perform clustering in the logit space using HDBSCAN. During clustering, we set the minimum cluster size to 2, the minimum number of samples to 1, and use Euclidean distance as the distance metric. As for the penalty coefficients, we set 𝛼 = 0.1 and 𝛽 = 0.2 for Qwen3-1.7B, Qwen2.5-Math-7B, and set 𝛼 = 0.02 and 𝛽 = 0.04 for Qwen3-8B.

In all training runs, we use a prompt-level batch size of 512 and generate 16 rollouts per prompt; the maximum prompt length is 1024 tokens. The maximum response length is 7168 tokens for Qwen3-1.7B, Qwen3-8B, and 3072 tokens for Qwen2.5-Math-7B. All training is conducted on H200 hardware. Other hyperparameters can be found in Appendix A.2.

OlympiadBench

AIME24

###### AMC23

###### MATH500

Minerva

70

75

96

| |
|---|

90

| |
|---|

| |
|---|

| |
|---|

60

| |
|---|

| |
|---|

Qwen3-1.7B

| |
|---|

| |
|---|

| |
|---|

60

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

90

| |
|---|

kpass@

60

| |
|---|

75

| |
|---|

| |
|---|

| |
|---|

| |
|---|

40

| |
|---|

| |
|---|

| |
|---|

| |
|---|

50

| |
|---|

| |
|---|

| |
|---|

| |
|---|

84

| |
|---|

| |
|---|

| |
|---|

60

| |
|---|

| |
|---|

45

20

| |
|---|

40

| |
|---|

| |
|---|

| |
|---|

| |
|---|

78

| |
|---|

1 2 4 8 16 32 64 128

1 2 4 8 16 32 64 128

1 2 4 8 16 32 64 128

1 2 4 8 16 32 64 128

1 2 4 8 16 32 64 128

80

100

| |
|---|

| |
|---|

Qwen2.5-Math-7B

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

| |
|---|
| |

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

60

| |
|---|

60

60

| |
|---|

| |
|---|

80

| |
|---|

75

| |
|---|

| |
|---|

| |
|---|

| |
|---|
| |

| |
|---|

| |
|---|

kpass@

| |
|---|

| |
|---|

| |
|---|

40

40

40

| |
|---|

60

| |
|---|

| |
|---|

50

| |
|---|

| |
|---|

20

20

20

40

25

1 2 4 8 16 32 64 128

1 2 4 8 16 32 64 128

1 2 4 8 16 32 64 128

1 2 4 8 16 32 64 128

1 2 4 8 16 32 64 128

75

64

| |
|---|

| |
|---|

90

75

95

| |
|---|

| |
|---|

| |
|---|

Qwen3-8B

| |
|---|

| |
|---|

60

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

kpass@

| |
|---|

56

| |
|---|

| |
|---|

90

| |
|---|

| |
|---|

| |
|---|

75

45

60

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

48

| |
|---|

| |
|---|

| |
|---|

| |
|---|

85

| |
|---|

30

| |
|---|

| |
|---|

| |
|---|

60

45

40

| |
|---|

| |
|---|

| |
|---|

| |
|---|

1 2 4 8 16 32 64 128

1 2 4 8 16 32 64 128

1 2 4 8 16 32 64 128

1 2 4 8 16 32 64 128

1 2 4 8 16 32 64 128

k

k

k

k

k

Base GRPO DAPO GRPO w/ Entropy Adv Ours

- Figure 4 Pass@𝑘 across 𝑘 ∈ {1, 2, 4, 8, 16, 32, 64, 128} on five math benchmarks for three model scales. Each row corresponds to one model; each column to one dataset. Ours consistently matches or outperforms Base, GRPO, DAPO, and GRPO with entropy advantage across settings.

Baselines We compare the proposed MEDS with two commonly used RL algorithms: GRPO and DAPO. In addition, as discussed in section 2.2, we also compare with a approach that encourages exploration through entropy regularization [41]. The hyperparameters and training objectives configurations for these algorithms are provided in appendix A.2. For reference, we also report the performance of the base model without fine-tuning as a baseline.

Evaluation We evaluate our method on five mathematical reasoning benchmarks covering different difficulty levels and problem types: AIME24 [42], AMC23 [42], MATH500 [38], Minerva [43], and OlympiadBench [44].

For pass@𝑘 evaluation, we set the number of samples to 𝑛 = 128 and report both pass@1 and pass@128. During evaluation, we also use the Qwen-Math template. The temperature is set to 1.0 and top-𝑝 to 1.0 for all models. The random seed is fixed to 0, and the maximum number of tokens is kept consistent with the training configuration.

##### 4.2 Main Results

- Table 1 summarizes the performance across three base models and five mathematical reasoning benchmarks. Our method consistently achieves the best overall results.

Across all three model scales, our method consistently achieves the best overall performance, outperforming both the base model and strong baselines such as DAPO. This advantage is observed on both Qwen3-1.7B and Qwen2.5-Math-7B, and becomes even more pronounced on Qwen3-8B model, suggesting that the benefits of our method scales with the capability of the base model. We also plot the pass@𝑘 curves, where 𝑘 takes values in {1, 2, 4, 8, 16, 32, 64, 128}. The corresponding results are shown in Figure 4, where our method achieves the best performance in most cases.

Training efficiency remains close to DAPO: on Qwen2.5-Math-7B, MEDS takes 8.46 minutes for 50 steps and

Top-1EigenRatio

0.930

DAPO

Ours

0.925

0.920

0.915

0 25 50 75 100

Step

Logit Aggregation Correlation(%) single cluster 45.16 28–diff 52.52 14–diff 54.71 28 53.87 14 61.2

- Figure 5 (Left) Top-1 eigen ratio over training steps for DAPO and Ours. Our method consistently yields lower ratios at later steps, indicating less concentration on a single dominant error direction. (Right) Correlation of different logit aggregation strategies on Qwen2.5-Math-7B. Single cluster assigns all logits to one cluster; 28 and 14 use direct aggregation over all 28 layers or the last 14 layers; 28–diff and 14–diff use aggregation over layer-wise differences between adjacent logits.

9.73 minutes for 100 steps, compared with 8.00 and 8.95 minutes for DAPO. Our method introduces only a small computational overhead compared with DAPO. Considering the consistent performance improvements, this overhead is relatively modest.

##### 4.3 Impact on Exploration Behavior

To better understand how our method influences the model’s exploration behavior during reasoning, we conduct a detailed analysis from both behavioral and representational perspectives. Specifically, we first employ an LLM-based evaluation to measure Within-Step Diversity and Across-Step Diversity, which capture two complementary aspects of rollout diversity. We then further analyze the statistical structure of the logits to provide additional evidence of diversity at the representation level.

Within-Step Diversity measures the diversity of rollouts generated by DAPO and MEDS at the same training step for the same input prompt. Since the first training epoch is mainly used for memory formation and does not involve reward shaping, we restrict this analysis to checkpoints from the second epoch, spanning approximately step 20 to step 100. At each step, we randomly sample 10 groups of 160 rollouts from each method, resulting in 1,600 rollout groups in total, and use Claude-Haiku-4.5 with the prompt in Appendix D to score group-level output diversity.

Across-Step Diversity instead evaluates whether later-stage rollouts exhibit new reasoning patterns relative to earlier ones for the same model and input prompt. For this purpose, we construct 21 pairs of data groups with varying step intervals, each roughly one epoch apart, where each data group contains 10 rollout groups from DAPO and 10 from MEDS. Using the same prompt-based protocol, the LLM compares rollouts from the earlier and later steps and scores the extent to which the later outputs introduce novel reasoning patterns. In both settings, higher scores indicate greater diversity.

Beyond these LLM-based evaluations, we further analyze the covariance structure of the stored logits across all steps and compute the ratio between the largest eigenvalue and the sum of all eigenvalues. We refer to this metric as the Top-1 Eigen Ratio. Its formal definition is given in Appendix E. A smaller value of this ratio indicates a more uniform spread of the logits across different directions in the representation space.

Our method improves exploration across multiple metrics. The LLM-based evaluation results are shown in the bottom-right panel of Figure 1, and the Top-1 Eigen Ratio results are presented in Figure 5 (left). We observe that rollout diversity generally declines as training progresses. However, our method consistently attains higher exploration scores than the baseline under both Within-Step Diversity and Across-Step Diversity. This trend is further supported by the Top-1 Eigen Ratio, where our method yields consistently lower values than the baseline, indicating a more diverse exploration behavior.

Question: Call a positive integer good if it has three distinct positive divisors whose sum equals

Cluster B reasoning pattern

- • Assume structural from p2
- • Derive other quadratic forms under N
- • Fallback to enumeration
- • The most common form of such a number is p2, with factors 1, p, p2.
- • We need to check next possible forms, such as n = p1 · p2 for two different primes.
- • Code a solution to check the values directly.
- • output: 1342

- • The structure of n needs to be such that it has divisors of specific forms. A number n = p2 has exactly three divisors: 1, p, p2.
- • The simplest structure … two different primes, start with factors like n = p1 · p2.
- • Examine these cases systematically. We will write some code to check these from the smallest numbers.
- • output: 1342

- • Identify the structure of divisors: 1, d, n/d. d is a divisor of n other than 1 and n.
- • We need pairs (d, n/d) such that their sum is 2021, with both values integers.
- • Iterate over divisors d and test by code.
- • output: 1349

2022. Find the smallest good number.

Ground truth: 1344

B1

- Cluster A reasoning pattern
- • Lock on to p2 premise
- • Directly turn to attempt systematic numerical enumeration
- • N must have exactly three distinct divisors, so the only possible form is p2. The divisors of p2 are 1, p, and p2.
- • Verify nearby prime values and implement Python code.
- • output: 1342

- • A number with exactly three positive divisors must be the square of a prime number p. Its divisors are 1, p, and p2.
- • Since the answer to the equation is not an integer, we check nearby integer values and verify whether p can be prime.
- • output: 1351

- B2

- B3

A1

- A1

B1

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Layer 20

Layer 20

Layer 20

A2

- A2

- B2
- B3

- Figure 6 A case study. We manually annotate the reasoning structures of different responses. The responses are grouped

into Cluster A containing A1, A2 and Cluster B containing B1, B2, B3 by our clustering method. Although A1, B1 and B2 produce the same incorrect answer 1342, their reasoning structures are different. A similar phenomenon can also be observed for the responses within Cluster B. We further visualize the logits using a heatmap. See Appendix C for the complete case study and the corresponding logit values.

##### 4.4 Logits can Reflect Reasoning Patterns

To demonstrate that logit vectors capture the underlying logical reasoning structure, we first perform a qualitative analysis of the logit vectors collected from rollouts across the MEDS training process. This preliminary analysis suggests that responses with similar reasoning patterns tend to exhibit similar layer-wise logit trajectories across layers.

Logit patterns reflect reasoning structure Figure 6 presents a case study example. The logits show relatively limited variation in the first 20 layers, where they primarily encode semantic information. In contrast, larger changes emerge in the later layers, where responses with similar reasoning patterns show clear correlations in their logit vectors, with layer-wise values tending to fall within similar ranges. Meanwhile, responses following different reasoning paths remain distinguishable. We further find that logit patterns reflect not only the final answer tokens, but also the intermediate reasoning process. Responses leading to the same answer may exhibit different logit patterns, while responses with different answers may still share similar reasoning behaviors.

##### 4.5 Logits Provide Correct Clustering Signals

To further validate this observation at scale, we introduce an automated annotation procedure using ClaudeHaiku-4.5 as a proxy annotator. We randomly sample 800 negative responses from the entire dataset. Based on 11 common error categories distilled through manual inspection, we first ask the proxy model to assign each response to one of these error clusters. We then sample response pairs and compare whether the two responses belong to the same cluster according to both the LLM annotations and the logit-based clustering. If the two judgments agree, i.e., both indicate that the responses belong to the same cluster or both indicate that they belong to different clusters, we regard the logit-based clustering as correct for that pair. The agreement rate is then used as the correlation metric. The manually defined error categories and the prompts used for annotation are provided in Appendix D.

AIME24 AMC MATH500 Minerva OlympiadBench Average pass@1 pass@128 pass@1 pass@128 pass@1 pass@128 pass@1 pass@128 pass@1 pass@128 pass@1 pass@128

Method

DAPO 32.27 70.00 72.71 97.50 85.61 96.00 43.01 75.74 44.43 75.00 55.61 82.85 MEDS-single cluster 29.70 70.00 73.60 100.00 86.60 96.80 41.20 71.69 44.60 74.52 55.14 82.60 MEDS-28-diff 31.88 70.00 72.34 97.50 85.80 96.00 43.38 73.53 45.48 74.37 55.78 82.28 MEDS-14-diff 32.40 73.33 71.64 100.00 86.40 96.60 43.01 72.43 44.15 74.52 55.52 83.38 MEDS-28 35.63 70.00 73.13 97.50 86.20 96.40 41.90 73.16 44.00 73.78 56.17 82.17 MEDS-14 34.32 76.67 74.38 97.50 86.33 96.00 42.51 74.26 44.80 75.56 56.47 84.00

- Table 2 Comparison of different clustering feature constructions on Qwen2.5-Math-7B, including assigning all logits to a single cluster (single cluster); using adjacent logit differences over 28 layers (28-diff) or the last 14 layers (14-diff); using logits from all 28 layers; and using only the last 14 layers. The first row corresponds to the vanilla DAPO baseline.

Logit clustering serves as an acceptable proxy for LLM-based clustering Figure 5 (right) shows the correlation between our logit-based clustering results and the annotations produced by Claude. With a correlation of 61.2%, the logit clustering exhibits a substantial positive relationship with the logical similarity identified by the LLM annotator.

##### 4.6 Clustering Quality Matters

Building on the rich representational signals provided by logits, clustering features can be constructed in multiple ways. We investigate how different feature construction choices affect clustering quality and how these differences further influence downstream evaluation performance.

Specifically, we consider several feature construction strategies that vary in the selected layers and in whether logit differencing is applied. As a worst-case baseline, we also include a degenerate setting in which all responses are assigned to a single cluster, yielding a trivial clustering with virtually no discriminative power. All experiments use Qwen2.5-Math-7B and differ only in the choice of clustering features. Following the Claude-correlation evaluation protocol described in Section 4.4, we report the results of different logit aggregation strategies in Figure 5 (right).

Using the last 14 layers as clustering features yields the best downstream performance. Table 2 reports the results under different feature construction settings. Among all configurations, using the last 14 layers as clustering features yields the best performance. In contrast, excluding logits altogether results in performance worse than DAPO, while every variant that leverages clustering information outperforms the DAPO baseline. This provides further evidence that logits serve as effective historical signals for reward shaping.

Clustering quality strongly correlates with downstream performance. Figure 5 (right) shows that using the last 14 layers as clustering features achieves the highest agreement with the Claude annotations, whereas the degenerate single-cluster setting yields the lowest. More importantly, the ranking of clustering consistency closely mirrors the ranking of downstream performance, suggesting that better clustering yields larger downstream improvements. This provides further evidence for the effectiveness of our approach.

### 5 Conclusion

In this work, we study the recurring error patterns that emerge during reinforcement learning for LLM reasoning. We introduce MEDS, a memory-aware reward shaping framework that dynamically penalizes recurrent failure patterns. We first provide theoretical analysis demonstrating the effectiveness of our method. By reusing layer-wise logits as lightweight representations of reasoning trajectories, our approach clusters historical responses to identify and suppress repeated error modes. Empirical results across multiple benchmarks show that MEDS consistently improves both reasoning performance and exploration diversity. These findings highlight the importance of incorporating historical trajectory information into reward design and open new directions for improving exploration in LLM reinforcement learning.

### 6 Limitations

The main limitation of our study is that the methods explored for utilizing logits are relatively simple and do not incorporate more sophisticated aggregation functions. It remains unclear whether adopting more complex aggregation strategies could further improve performance.

### References

- [1] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300,

2024. doi: 10.48550/ARXIV.2402.03300. URL https://doi.org/10.48550/arXiv.2402.03300.

- [2] Shihan Dou, Yan Liu, Haoxiang Jia, Limao Xiong, Enyu Zhou, Wei Shen, Junjie Shan, Caishuang Huang, Xiao Wang, Xiaoran Fan, Zhiheng Xi, Yuhao Zhou, Tao Ji, Rui Zheng, Qi Zhang, Xuanjing Huang, and Tao Gui. Stepcoder: Improve code generation with reinforcement learning from compiler feedback. CoRR, abs/2402.01391, 2024. doi: 10.48550/ARXIV.2402.01391. URL https://doi.org/10.48550/arXiv.2402.01391.
- [3] Parshin Shojaee, Aneesh Jain, Sindhu Tipirneni, and Chandan K. Reddy. Execution-based code generation using deep reinforcement learning. Trans. Mach. Learn. Res., 2023, 2023. URL https://openreview.net/forum?id=0XBuaxqEcG.
- [4] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips. cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html.
- [5] Gokul Swamy, Sanjiban Choudhury, Wen Sun, Zhiwei Steven Wu, and J. Andrew Bagnell. All roads lead to likelihood: The value of reinforcement learning in fine-tuning. CoRR, abs/2503.01067, 2025. doi: 10.48550/ARXIV.2503.01067. URL https://doi.org/10.48550/arXiv.2503.01067.
- [6] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: an open-source LLM reinforcement learning system at scale. CoRR, abs/2503.14476, 2025. doi: 10.48550/ARXIV.2503.14476. URL https://doi.org/10.48550/arXiv.2503.14476.
- [7] Abhĳeet Sinha, Sundari Elango, and Dianbo Liu. Expected return causes outcome-level mode collapse in reinforcement learning and how to fix it with inverse probability scaling. CoRR, abs/2601.21669, 2026. doi: 10.48550/ARXIV.2601.21669. URL https://doi.org/10.48550/arXiv.2601.21669.
- [8] Liang Chen, Xueting Han, Qizhou Wang ands Bo Han, Jing Bai, Hinrich Schutze, and Kam-Fai Wong. EEPO: exploration-enhanced policy optimization via sample-then-forget. CoRR, abs/2510.05837, 2025. doi: 10.48550/ ARXIV.2510.05837. URL https://doi.org/10.48550/arXiv.2510.05837.
- [9] Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, and Yu Meng. The surprising effectiveness of negative reinforcement in LLM reasoning. CoRR, abs/2506.01347, 2025. doi: 10.48550/ARXIV.2506.01347. URL https://doi.org/10.48550/arXiv.2506.01347.
- [10] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In Jennifer G. Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pages 1856–1865. PMLR, 2018. URL http: //proceedings.mlr.press/v80/haarnoja18b.html.
- [11] Zhongwei Wan, Yun Shen, Zhihao Dou, Donghao Zhou, Yu Zhang, Xin Wang, Hui Shen, Jing Xiong, Chaofan Tao, Zixuan Zhong, et al. Dsdr: Dual-scale diversity regularization for exploration in llm reasoning. arXiv preprint arXiv:2602.19895, 2026.

- [12] Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Maria-Florina Balcan and Kilian Q. Weinberger, editors, Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016, volume 48 of JMLR Workshop and Conference Proceedings, pages 1928–1937. JMLR.org, 2016. URL http://proceedings.mlr.press/v48/mniha16.html.
- [13] Clay Holroyd and Michael Coles. The neural basis of human error processing: Reinforcement learning, dopamine, and the error-related negativity. Psychological Review, 109:679–709, 11 2002. doi: 10.1037/0033-295X.109.4.679.
- [14] Leland McInnes, John Healy, and Steve Astels. hdbscan: Hierarchical density based clustering. J. Open Source Softw., 2(11):205, 2017. doi: 10.21105/JOSS.00205. URL https://doi.org/10.21105/joss.00205.
- [15] OpenAI. Openai o1 system card. CoRR, abs/2412.16720, 2024. doi: 10.48550/ARXIV.2412.16720. URL https: //doi.org/10.48550/arXiv.2412.16720.
- [16] YaoyuZhu, DiHuang, HanqiLyu, XiaoyunZhang, ChongxiaoLi, WenxuanShi, YutongWu, JiananMu, JinghuaWang, Yang Zhao, Pengwei Jin, Shuyao Cheng, Shengwen Liang, Xishan Zhang, Rui Zhang, Zidong Du, Qi Guo, Xing Hu, and Yunji Chen. Codev-r1: Reasoning-enhanced verilog generation, 2025. URL https://arxiv.org/abs/2505.24183.
- [17] Haoji Zhang, Xin Gu, Jiawen Li, Chixiang Ma, Sule Bai, Chubin Zhang, Bowen Zhang, Zhichao Zhou, Dongliang He, and Yansong Tang. Thinking with videos: Multimodal tool-augmented reinforcement learning for long video reasoning. CoRR, abs/2508.04416, 2025. doi: 10.48550/ARXIV.2508.04416. URL https://doi.org/10.48550/arXiv. 2508.04416.
- [18] Yang Chen, Yufan Shen, Wenxuan Huang, Sheng Zhou, Qunshu Lin, Xinyu Cai, Zhi Yu, Jiajun Bu, Botian Shi, and Yu Qiao. Learning only with images: Visual reinforcement learning with reasoning, rendering, and visual feedback. arXiv preprint arXiv:2507.20766, 2025.
- [19] Le Zhang, Bo Wang, Xipeng Qiu, Siva Reddy, and Aishwarya Agrawal. REARANK: reasoning re-ranking agent via reinforcement learning. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, EMNLP 2025, Suzhou, China, November 4-9, 2025, pages 2458–2471. Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025.EMNLP-MAIN.125. URL https://doi.org/10.18653/v1/2025.emnlp-main.125.
- [20] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https: //openreview.net/forum?id=v8L0pN6EOi.
- [21] Jonathan Uesato, Nate Kushman, Ramana Kumar, H. Francis Song, Noah Y. Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process- and outcome-based feedback. CoRR, abs/2211.14275, 2022. doi: 10.48550/ARXIV.2211.14275. URL https://doi.org/10.48550/arXiv.2211.14275.
- [22] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017. URL http://arxiv.org/abs/1707.06347.
- [23] Zhiyuan Hu, Yucheng Wang, Yufei He, Jiaying Wu, Yilun Zhao, See-Kiong Ng, Cynthia Breazeal, Anh Tuan Luu, Hae Won Park, and Bryan Hooi. Rewarding the rare: Uniqueness-aware RL for creative problem solving in llms. CoRR, abs/2601.08763, 2026. doi: 10.48550/ARXIV.2601.08763. URL https://doi.org/10.48550/arXiv.2601.08763.
- [24] Yuda Song, Julia Kempe, and Rémi Munos. Outcome-based exploration for LLM reasoning. CoRR, abs/2509.06941,

2025. doi: 10.48550/ARXIV.2509.06941. URL https://doi.org/10.48550/arXiv.2509.06941.

- [25] Hao Li, Xue Yang, Zhaokai Wang, Xizhou Zhu, Jie Zhou, Yu Qiao, Xiaogang Wang, Hongsheng Li, Lewei Lu, and Jifeng Dai. Auto mc-reward: Automated dense reward design with large language models for minecraft. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 16426–16435. IEEE, 2024. doi: 10.1109/CVPR52733.2024.01554. URL https://doi.org/10.1109/CVPR52733.2024.01554.
- [26] Shunyu Yao, Fei Liu, Xi Lin, Zhichao Lu, Zhenkun Wang, and Qingfu Zhang. Multi-objective evolution of heuristic using large language model. In Toby Walsh, Julie Shah, and Zico Kolter, editors, AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25 - March 4, 2025, Philadelphia, PA, USA, pages 27144–27152. AAAI Press, 2025. doi: 10.1609/AAAI.V39I25.34922. URL https://doi.org/10.1609/aaai.v39i25.34922.

- [27] Yun Qu, Yuhang Jiang, Boyuan Wang, Yixiu Mao, Cheems Wang, Chang Liu, and Xiangyang Ji. Latent reward: Llm-empowered credit assignment in episodic reinforcement learning. In Toby Walsh, Julie Shah, and Zico Kolter, editors, AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25 - March 4, 2025, Philadelphia, PA, USA, pages 20095–20103. AAAI Press, 2025. doi: 10.1609/AAAI.V39I19.34213. URL https://doi.org/10.1609/aaai.v39i19.34213.
- [28] Rishi Hazra, Alkis Sygkounas, Andreas Persson, Amy Loutfi, and Pedro Zuidberg Dos Martires. Revolve: Reward evolution with large language models using human feedback. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum? id=cJPUpL8mOw.
- [29] Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Eureka: Human-level reward design via coding large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net,

2024. URL https://openreview.net/forum?id=IEduRUO55F.

- [30] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/2022/ hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html.
- [31] Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas L Turner, Callum McDougall, Monte MacDiarmid, C. Daniel Freeman, Theodore R. Sumers, Edward Rees, Joshua Batson, Adam Jermyn, Shan Carter, Chris Olah, and Tom Henighan. Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet. Transformer Circuits Thread, 2024. URL https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html.
- [32] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context learning and induction heads. Transformer Circuits Thread, 2022. https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html.
- [33] Zhengfu He, Junxuan Wang, Rui Lin, Xuyang Ge, Wentao Shu, Qiong Tang, Junping Zhang, and Xipeng Qiu. Towards understanding the nature of attention with low-rank sparse decomposition. CoRR, abs/2504.20938, 2025. doi: 10.48550/ARXIV.2504.20938. URL https://doi.org/10.48550/arXiv.2504.20938.
- [34] Zhengfu He, Wentao Shu, Xuyang Ge, Lingjie Chen, Junxuan Wang, Yunhua Zhou, Frances Liu, Qipeng Guo, Xuanjing Huang, Zuxuan Wu, Yu-Gang Jiang, and Xipeng Qiu. Llama scope: Extracting millions of features from llama-3.1-8b with sparse autoencoders. CoRR, abs/2410.20526, 2024. doi: 10.48550/ARXIV.2410.20526. URL https://doi.org/10.48550/arXiv.2410.20526.
- [35] Zheng Zhao, Yeskendir Koishekenov, Xianjun Yang, Naila Murray, and Nicola Cancedda. Verifying chain-of-thought reasoning via its computational graph. CoRR, abs/2510.09312, 2025. doi: 10.48550/ARXIV.2510.09312. URL https://doi.org/10.48550/arXiv.2510.09312.
- [36] Yuqiao Tan, Minzheng Wang, Shizhu He, Huanxuan Liao, Chengfeng Zhao, Qiunan Lu, Tian Liang, Jun Zhao, and Kang Liu. Bottom-up policy optimization: Your language model policy secretly contains internal policies. CoRR, abs/2512.19673, 2025. doi: 10.48550/ARXIV.2512.19673. URL https://doi.org/10.48550/arXiv.2512.19673.
- [37] Xin Cheng, Wangding Zeng, Damai Dai, Qinyu Chen, Bingxuan Wang, Zhenda Xie, Kezhao Huang, Xingkai Yu, Zhewen Hao, Yukun Li, Han Zhang, Huishuai Zhang, Dongyan Zhao, and Wenfeng Liang. Conditional memory via scalable lookup: A new axis of sparsity for large language models. CoRR, abs/2601.07372, 2026. doi: 10.48550/ARXIV.2601.07372. URL https://doi.org/10.48550/arXiv.2601.07372.
- [38] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Joaquin Vanschoren and Sai-Kit Yeung, editors, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, 2021. URL https://datasets-benchmarks-proceedings. neurips.cc/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html.

- [39] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. CoRR, abs/2503.20783, 2025. doi: 10.48550/ARXIV.2503.20783. URL https://doi.org/10.48550/arXiv.2503.20783.
- [40] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient RLHF framework. In Proceedings of the Twentieth European Conference on Computer Systems, EuroSys 2025, Rotterdam, The Netherlands, 30 March 2025 - 3 April 2025, pages 1279–1297. ACM,

2025. doi: 10.1145/3689031.3696075. URL https://doi.org/10.1145/3689031.3696075.

- [41] Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective. ArXiv, abs/2506.14758, 2025. URL https://api.semanticscholar.org/ CorpusID:279410424.
- [42] Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Zĳu Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13(9):9, 2024.
- [43] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay V. Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy GurAri, and Vedant Misra. Solving quantitative reasoning problems with language models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 18abbeef8cfe9203fdf9053c9c4fe191-Abstract-Conference.html.
- [44] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 3828–3850. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.211. URL https://doi.org/10.18653/v1/2024.acl-long.211.
- [45] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/ a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html.

### A Implementation Details

##### A.1 Qwen-Math Template

All problems in our training and evaluation datasets are processed using the Qwen-Math template, which provides a standardized format for mathematical problem-solving prompts.

<|im_start|>system Please reason step by step, and put your final answer within \boxed{}.<|im_end|> <|im_start|>user {problem} <|im_end|> <|im_start|>assistant

##### A.2 Training Hyperparameters and Objectives

All methods are trained with the Adam optimizer at a learning rate of 1 × 10−6, weight decay 0.1, gradient clipping threshold 1.0, and 10 warmup steps, with a PPO mini-batch size of 16.

GRPO follows the official DeepSeekMath configuration [1], with clipping ratio 𝜖 = 0.2 and KL penalty coefficient 𝛽 = 10−3.

DAPO employs decoupled clipping with asymmetric thresholds 𝜀low = 0.2 and 𝜀high = 0.28, and removes the KL penalty term entirely.

GRPO w/ Entropy Adv follows Cheng et al. [41], augmenting the group-relative advantage with a per-token entropy bonus weighted by 𝛼. The training objectives are defined as follows.

ℒGRPO(𝜃) = 𝔼𝑞∼𝒟

𝐾

1 𝐾

min 𝑟𝑖(𝜃)𝐴𝑖, clip(𝑟𝑖(𝜃), 1−𝜖, 1+𝜖)𝐴𝑖 − 𝛽 𝔻KL[𝜋𝜃 ∥𝜋ref] (1)

𝑖=1

ℒDAPO(𝜃) = 𝔼𝑞∼𝒟

𝐾

1 𝐾

min 𝑟𝑖(𝜃)𝐴𝑖, clip(𝑟𝑖(𝜃), 1−𝜖low, 1+𝜖high)𝐴𝑖 (2)

𝑖=1

𝐾

1 𝐾

min 𝑟𝑖(𝜃)𝐴˜𝑖, clip(𝑟𝑖(𝜃), 1−𝜖, 1+𝜖)𝐴˜𝑖 − 𝛽 𝔻KL[𝜋𝜃 ∥𝜋ref] (3)

ℒGRPO+Ent(𝜃) = 𝔼𝑞∼𝒟

𝑖=1

where the probability ratio, group-relative advantage, entropy-augmented advantage, and per-token entropy are respectively defined as:

𝜋𝜃(𝑜𝑖 | 𝑞) 𝜋𝜃old(𝑜𝑖 | 𝑞)

(4)

𝑟𝑖(𝜃) =

𝑅𝑖 − mean({𝑅𝑗}𝐾𝑗=1) std({𝑅𝑗}𝐾𝑗=1)

(5)

𝐴𝑖 =

𝐴˜𝑖 = 𝐴𝑖 + 𝛼 · ℋ𝑡 (6) ℋ𝑡 = −

𝜋𝜃(𝑣 | 𝑞, 𝑜<𝑡)log𝜋𝜃(𝑣 | 𝑞, 𝑜<𝑡) (7)

𝑣∈𝒱

### B Theoratical Proof

Theorem 2. Suppose there are two different reward signals, 𝜇1 = 𝑟(x, y) and 𝜇2 = 𝑟(x, y) − 𝜆𝑐(y), and let 𝑞1 and 𝑞2 denote the corresponding updated policies under these reward signals. Let 𝐽(𝑞) = 𝔼x,y∼𝑞[𝑟(x, y)]. Then it holds that

𝐽(𝑞2) ≥ 𝐽(𝑞1), (8)

where 𝜆 is the repetition penalty coefficient. 𝑐(y) : x × 𝑌 → ℕ denotes the indicator of the number of repeated errors, where a value greater than zero indicates that y corresponds to an error that has been repeatedly sampled in the past. The function 𝑟(x, y) is a task-specific scoring function.

Proof. For clarity we work pointwise in x and assume 𝒴 is finite (the continuous case follows by replacing sums with integrals). According to the design that imposes additional penalties on repeated errors, we have:

𝑐(y1) ≥ 𝑐(y2) =⇒ 𝑟(x, y1) ≤ 𝑟(x, y2). (9) That is, responses with larger repetition count are not higher-scoring under the task reward. For any scoring signal 𝜇(x, y), define the KL-regularized one-step update (for each x) as

1 𝜂

𝑞𝜇(· | x) ∈argmax 𝑞(·|x)

KL(𝑞(· | x)∥ 𝑝ref(· | x)) .

𝔼y∼𝑞(·|x)[𝜇(x, y)] −

Using a Lagrange multiplier for the constraint y 𝑞(y | x) = 1, the optimizer has the Gibbs form[45]

𝑝ref(y | x)exp(𝜂 𝜇(x, y)) 𝑍𝜇(x)

𝑞𝜇(y | x) =

,

(10)

𝑝ref(y | x)exp(𝜂 𝜇(x, y)).

𝑍𝜇(x) =

y

Let 𝜇1(x, y) = 𝑟(x, y) and 𝜇2(x, y) = 𝑟(x, y) − 𝜆𝑐(y) with 𝜆 > 0, and denote 𝑞1 = 𝑞𝜇1, 𝑞2 = 𝑞𝜇2. From (10),

- 𝑞1(y | x) =

𝑝ref(y | x)exp(𝜂𝑟(x, y) 𝑍1(x)

,

- 𝑞2(y | x) =

𝑝ref(y | x)exp(𝜂𝑟(x, y) − 𝜂𝜆𝑐(y)) 𝑍2(x)

. Define 𝑤(y) := exp(−𝜂𝜆𝑐(y)) ∈ (0, 1]. Then for each fixed x,

𝑞1(y | x) 𝑤(y)

𝑞2(y | x) =

y′ 𝑞1(y′ | x) 𝑤(y′)

𝑞1(y | x) 𝑤(y) 𝔼y∼𝑞1(·|x)[𝑤(y)]

=

.

Define the per-x expected reward under policy 𝑞 as

(11)

𝐽x(𝑞) := 𝔼y∼𝑞(·|x)[𝑟(x, y)].

Using (11), we obtain

𝑞2(y | x)𝑟(x, y)

𝐽x(𝑞2) =

y

𝑞1(y | x) 𝑤(y) 𝑟(x, y)

= y

(12)

y 𝑞1(y | x) 𝑤(y)

𝔼y∼𝑞1(·|x)[𝑤(y) 𝑟(x, y)] 𝔼y∼𝑞1(·|x)[𝑤(y)]

=

.

Hence,

𝔼𝑞1[𝑤 𝑟] − 𝔼𝑞1[𝑤]𝔼𝑞1[𝑟] 𝔼𝑞1[𝑤]

𝐽x(𝑞2) − 𝐽x(𝑞1) =

(13)

Cov𝑞1(𝑟(x, y), 𝑤(y)) 𝔼𝑞1[𝑤(y)]

.

=

Since 𝔼𝑞1[𝑤(y)] > 0, it suffices to show that Cov𝑞1(𝑟(x, y), 𝑤(y)) ≥ 0.

Let 𝑌 ∼ 𝑞1(· | x) and 𝐶 = 𝑐(𝑌). Note that 𝑤(𝑌) = exp(−𝜂𝜆𝐶) is non-increasing in 𝐶. By equation (9), 𝑟(x,𝑌) is also non-increasing in 𝐶 (equivalently, the conditional mean 𝑟¯x(𝑘) := 𝔼[𝑟(x,𝑌) | 𝐶 = 𝑘] is non-increasing in 𝑘).

Therefore, by Chebyshev’s rearrangement inequality,

𝔼[¯𝑟x(𝐶)exp(−𝜂𝜆𝐶)] ≥ 𝔼[¯𝑟x(𝐶)] · 𝔼[exp(−𝜂𝜆𝐶)]. (14)

Using the tower property,

𝔼𝑞1[𝑟(x,𝑌)𝑤(𝑌)]

= 𝔼 𝔼[𝑟(x,𝑌) | 𝐶]exp(−𝜂𝜆𝐶)

= 𝔼[¯𝑟x(𝐶)exp(−𝜂𝜆𝐶)].

Similarly,

𝔼𝑞1[𝑟(x,𝑌)] = 𝔼[¯𝑟x(𝐶)], 𝔼𝑞1[𝑤(𝑌)] = 𝔼[exp(−𝜂𝜆𝐶)].

Thus, Cov𝑞1(𝑟(x,𝑌), 𝑤(𝑌)) ≥ 0, and (13) implies 𝐽x(𝑞2) ≥ 𝐽x(𝑞1) for all x. Taking expectation over x ∼ 𝐷, we obtain

𝐽(𝑞2) = 𝔼x∼𝐷[𝐽x(𝑞2)] ≥ 𝔼x∼𝐷[𝐽x(𝑞1)]

= 𝐽(𝑞1).

This completes the proof.

(15)

(16)

(17)

| |
|---|

### C Full Results of Case Study

###### Question

Call a positive integer good if it has three distinct positive divisors whose sum equals 2022. Find the smallest good number. Ground truth: 1344

##### C.1 Logit Values

- Figure 7 is the logit heatmap in Figure 6, which has logit values on it.

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

#### A1 A2 B1 B2 B3

Figure 7 Heatmap in Figure 6 with logit values.

##### C.2 Custer A Responses

###### Response A1 Output: 1342

To solve the problem of finding the smallest good number, we need to understand that a number is good if it has three distinct positive divisors that sum to 2022. A key insight here is that if a number 𝑛 is a good number, then it must have exactly three distinct divisors. The only type of number that has exactly three distinct divisors is a square of a prime number. This is because the number

of divisors of a number 𝑛 = 𝑝1𝑒1𝑝2𝑒2 ··· 𝑝𝑘𝑒𝑘 is given by (𝑒1 + 1)(𝑒2 + 1)···(𝑒𝑘 + 1). To have exactly 3 divisors, the only possible form is 𝑝2 because 3 can only be factored as 3 = 3 × 1. The divisors of 𝑝2 are 1, 𝑝, 𝑝2. We need these to sum to 2022, so we need:

1 + 𝑝 + 𝑝2 = 2022 This simplifies the equation to:

𝑝2 + 𝑝 + 1 = 2022 Rearranging the terms, we get a standard quadratic equation:

𝑝2 + 𝑝 − 2021 = 0

√

𝑏2−4𝑎𝑐 2𝑎 :

We will solve this quadratic equation using the quadratic formula 𝑝 = −𝑏±

√1 + 4 × 2021 2

𝑝 = −1 ±

√8085 2

= −1 ±

We are only interested in the positive value of 𝑝, so we need to check if √8085 is an integer. We will calculate √8085 and check its value. If 8085 is a perfect square, we can proceed further. By calculating, we find that √8085 ≈ 89.9166, which is not an integer. So we need to find the closest prime number. We check the closest perfect square value and determine:

892 = 7921 and 902 = 8100

Since 8085 lies between these two, we check the values for the primes near 89 and verify. We check whether we can substitute 89 into our equation to see if we get a prime number for 𝑝: If we let 𝑝 = 89: 892 + 89 + 1 = 7921 + 89 + 1 = 8011 ≠ 2022. So we will try 𝑝 = 83: 832 + 83 + 1 = 6889 + 83 + 1 = 6973 ≠ 2022. This also does not match. We will check 𝑝 = 83. Let’s use a closest prime number which might satisfy, let’s take 𝑝 = 43: 432 + 43 + 1 = 1849 + 43 + 1 = 1893 ≠ 2022. This does not match either. Let’s check with smaller primes. We will check 𝑝 = 41: 412 + 41 + 1 = 1681 + 41 + 1 = 1723 ≠ 2022. This is still not 2022. Let’s check with 𝑝 = 47: 472 + 47 + 1 = 2209 + 47 + 1 = 2257 ≠ 2022. This also does not work. Now we check 𝑝 = 43: 432 = 1849, 1849 + 43 + 1 = 1893 ≠ 2022. Let’s check with larger prime numbers and verify by code to ensure we get the exact result. We can implement a Python code to check the values and find the smallest 𝑛. We will start by checking 𝑝 = 45 as a possible value: 452 = 2025, which is greater than 2022, but let’s proceed and confirm by code. We need to find 𝑝 such that 𝑝 + 1 + 𝑝2 = 2022. Let’s implement this code to find the exact prime number.

import math def find_smallest_good_number():

# We will start with the smallest possible # number and check upwards num = 1 while True:

divisors = [d for d in range(1, \ num + 1) if num % d == 0]

if len(divisors) >= 3: divisors = sorted(divisors)[:3] if sum(divisors) == 2022:

return num num += 1

smallest_good_number = \

find_smallest_good_number() print(smallest_good_number)

Output: 1342 The smallest good number, which satisfies the condition that three distinct positive divisors sum to 2022, is

.

|1342|
|---|

###### Response A2 Output: 1351

To find the smallest number with exactly three distinct positive divisors that sum up to 2022, we need to analyze the structure of such a number. A number with exactly three positive divisors must be the square of a prime number 𝑝, because its divisors are 1, 𝑝, and 𝑝2. Given the condition, if 𝑛 = 𝑝2, then we need 1 + 𝑝 + 𝑝2 = 2022. We can set up the equation:

𝑝2 + 𝑝 + 1 = 2022. Rearranging gives:

𝑝2 + 𝑝 − 2021 = 0. This is a quadratic equation of the form 𝑝2+𝑝−2021 = 0. We can solve for 𝑝 using the quadratic formula 𝑝 = −𝑏±

√

𝑏2−4𝑎𝑐 2𝑎 , where 𝑎 = 1, 𝑏 = 1, and 𝑐 = −2021. This gives:

𝑝 = −1 ± 12 − 4 · 1 · (−2021) 2 · 1

√1 + 8084 2

= −1 ±

√8085 2

= −1 ±

.

We are only interested in the positive root, so we compute:

√8085 2

𝑝 = −1 +

.

√8085 2 is a prime number.

We need to check if √8085 is an integer, and if −1+

First, let’s approximate √8085: √8085 ≈ 89.9166. So, √8085 is very close to 90. Let’s check if 90 is indeed the square root: 902 = 8100, which is slightly larger than 8085. So, √8085 ≈ 89.9166 ≈ 90 (not exact, so we check 89). Let’s check 89 directly: 892 = 7921, which is smaller than 8085. Let’s check 90: 902 = 8100, which is slightly larger than

8085. Thus, 89 is closer and we need to check it more precisely:

892 = 7921 and 902 = 8100. Since 892 is closer to 8085, we use it: √8085 ≈ 89.9166, and thus:

𝑝 = −1 + 89.9166 2 ≈

88.9166 2 ≈ 44.4583,

which isn’t an integer. Let’s check the closest integer values. We need 𝑝 to be an integer and prime. Let’s check if 89 is a solution by substituting it back:

𝑝2 + 𝑝 − 2021 = 892 + 89 − 2021 = 7921 + 89 − 2021

= 4078 ≠ 0.

Thus, we check the next closest prime number, and we know that 𝑝 must be an integer. So let’s check manually for the factorization and see if we can find a prime 𝑝: If we try solving the equation directly for integer values:

𝑝2 + 𝑝 − 2021 = 0 =⇒ 𝑝(𝑝 + 1) = 2021. We look for pairs of factors of 2021. We factorize 2021:

2021 = 43 × 47.

We see that 43 and 44 don’t work, but checking 43 directly in the equation: 432 = 1849 and 43 + 1849 = 1892 ≠ 2022. Let’s check if there’s a smaller 𝑝 that works by iterating through possible prime values. We use Python to automate checking for the smallest 𝑛. We’ll start by finding potential 𝑝 values that make 1 + 𝑝 + 𝑝2 = 2022 a valid equation and check if 𝑝 is prime.

import math def is_prime(n):

if n <= 1:

return False if n <= 3:

return True if n % 2 == 0 or n % 3 == 0:

return False i = 5 while i * i <= n:

if n % i == 0 or n % (i + 2) == 0:

return False i += 6

return True

def find_smallest_good_number(): target_sum = 2022 p = 1 while True:

p_squared = p * p if 1 + p + p_squared == target_sum:

if is_prime(p):

return p_squared p += 1

smallest_good_number = \

find_smallest_good_number() print(smallest_good_number)

Output: 1351 The smallest number that has exactly three distinct positive divisors summing up to 2022 is

.

|1351|
|---|

##### C.3 Cluster B Responses

###### Response B1 Output: 1342

To find the smallest positive integer 𝑛 that has exactly three distinct positive factors whose sum equals 2022, we need to analyze the structure of the factors. Analysis:

- 1. Understanding the Factors: For a number 𝑛 to have exactly three distinct positive factors, the most common form of such a number is 𝑝2, where 𝑝 is a prime number. The factors of 𝑝2 are 1, 𝑝, and 𝑝2. So, if 𝑛 = 𝑝2, the sum of the factors 1 + 𝑝 + 𝑝2 = 2022. Therefore, we have:

𝑝2 + 𝑝 + 1 = 2022 =⇒ 𝑝2 + 𝑝 − 2021 = 0

√

𝑏2−4𝑎𝑐

- 2. Solving the Quadratic Equation: We solve for 𝑝 using the quadratic formula 𝑝 = −𝑏±

2𝑎 where 𝑎 = 1, 𝑏 = 1, and 𝑐 = −2021:

√1 + 4 · 2021 2

𝑝 = −1 ±

√8085 2

= −1 ±

√8085 2 is a prime number. We find that

Since 𝑝 is a prime number, we need to check if √8085 is an integer and if −1+

√8085 = 89.917. This means the closest integer value for the square root is 89. Checking if 89 works:

𝑝 = −1 + 89 2

88 2

= 44

=

But 44 is not a prime number. We need to check the next possible forms. Another form for having three factors might be when 𝑛 = 𝑝1 · 𝑝2 for two different primes 𝑝1 and 𝑝2. However, since 𝑝1 · 𝑝2 would suggest factors 1, 𝑝1, 𝑝2, and 𝑝1𝑝2, having exactly three factors requires a form without additional factors. Instead, let’s examine the simplest form more closely. Since 44 wasn’t a prime, we incrementally check the next prime values for 𝑝:

Iterating through Prime Numbers: We should check if the factor forms might lead to other structures that fit, or directly check if smaller values fit our sum of three factors form directly. Let’s code a solution to check the values directly: iterate through numbers calculating their factors and check the sum. Python Code: We will write a Python script to find the smallest number whose factors sum to 2022. We’ll start with a function to calculate the sum of the factors and then iterate over potential candidates. Let’s implement the solution in Python.

import sympy def sum_of_factors(n):

# Function to calculate sum of all factors # of n factors = set(sympy.divisors(n)) return sum(factors)

# We need to find the smallest n such that # sum_of_factors(n) == 2022 target_sum = 2022 n = 1

while True:

if sum_of_factors(n) == target_sum: smallest_good_number = n break

n += 1 print(smallest_good_number)

Output: 1342 The smallest positive integer 𝑛 that has exactly three distinct positive factors whose sum equals 2022 is

.

|1342|
|---|

- Response B2 Output: 1342 To solve the problem of finding the smallest positive integer that has three different positive divisors whose sum equals

2022, we need to delve into the properties of divisors. Suppose 𝑛 = 𝑝1𝑒1 · 𝑝2𝑒2 ··· 𝑝𝑘𝑒𝑘 is the prime factorization of 𝑛. The total number of divisors of 𝑛 is given by (𝑒1 + 1)(𝑒2 + 1)···(𝑒𝑘 + 1). If 𝑛 has exactly three positive divisors greater than 1 and these three divisors sum up to 2022, the structure of 𝑛 needs to be such that it has divisors of specific forms.

###### Analysis

- 1. The number 𝑛 cannot be a perfect power of more than one prime because it would have more than three divisors. Hence, we must look for specific structures.

- 2. A number 𝑛 that is a perfect square of a prime, i.e., 𝑛 = 𝑝2, will have exactly three divisors: 1, 𝑝, and 𝑝2. But since we require three different divisors greater than 1, we realize 𝑝 itself should be considered as having divisors 𝑝 and 𝑝2 along with a new third different divisor (other than 𝑝 and 𝑝2). So let’s consider the structure that has more than just those simple prime power forms. Since the simplest structure is a product of two different primes, let’s start with factors like 𝑛 = 𝑝1 · 𝑝2. Such 𝑛 would have divisors 1, 𝑝1, 𝑝2, and 𝑝1𝑝2. But since we want exactly three different positive divisors (excluding 1) that sum to 2022, let’s look into it systematically. We will check numbers of the form 𝑝2 · 𝑞, since it would have divisors 1, 𝑝, 𝑞, 𝑝2, 𝑝𝑞, 𝑝2𝑞. Here we are interested in checking 𝑝2, 𝑝𝑞, 𝑝2𝑞 or similar other combinations. However, another approach is checking sums systematically for candidate numbers directly. We will write some code to check these systematically to find the smallest 𝑛 such that three of its distinct divisors (excluding 1) sum to 2022. The plan is to check each number’s divisors and see if any three of them (excluding 1) sum up to 2022. We’ll start from the smallest numbers. We will write a function to get the divisors of a number and then check the sums.

def get_divisors(n): """ Returns a sorted list of all divisors of n. """ divisors = set() for i in range(1, int(n**0.5) + 1):

if n % i == 0: divisors.add(i) divisors.add(n // i)

return sorted(divisors) def has_three_divisors_summing_to_target(n, \ target):

""" Checks if there are three distinct divisors (excluding 1) that sum to target. """ divisors = get_divisors(n) if 1 in divisors:

divisors.remove(1) for i in range(len(divisors)): for j in range(i + 1, len(divisors)): for k in range(j + 1, \ len(divisors)): if divisors[i] + divisors[j] \

+ divisors[k] == target: return True

return False

def find_smallest_good_number(target): n = 2 while True:

if has_three_divisors_summing_to_target(n, target):

return n n += 1

# We are looking for the smallest number such # that 3 different divisors (excluding 1) sum # to 2022 smallest_good_number = \

find_smallest_good_number(2022) print(smallest_good_number)

Output: 1342 The smallest positive integer 𝑛 such that three of its distinct positive divisors (excluding 1) sum to 2022 is

.

|1342|
|---|

###### Response B3 Output: 1349

To solve this problem, we need to find the smallest positive integer 𝑛 that has three distinct divisors 𝑑1, 𝑑2, and 𝑑3 such that their sum is 2022. Let’s break down the problem step by step.

###### 1. Identify the Structure of Divisors:

- • For a number 𝑛 to have three distinct divisors, they must be in the form 1, 𝑑, 𝑛𝑑 where 𝑑 is a divisor of 𝑛 other than 1 and 𝑛. The sum of these three divisors is 1 + 𝑑 + 𝑛𝑑 = 2022.

- • This equation can be rearranged to: 1 + 𝑑 + 𝑛𝑑 = 2022 or 𝑑 + 𝑛𝑑 = 2021. Thus, 𝑛 = 𝑑(2021 − 𝑑).

###### 2. Find the Smallest 𝑛:

- • We need to find pairs (𝑑, 𝑛𝑑 ) such that their sum is 2021 and 𝑑 ≠ 1, 𝑛𝑑 . Both 𝑑 and 𝑛𝑑 must be integers, so 𝑑 must be a divisor of 𝑛 and 2021 − 𝑑 must be an integer greater than 𝑑 (because 𝑑 ≠ 𝑛𝑑 ).

- • The smallest possible 𝑑 greater than 1 is 2, but let’s check all possible values systematically.

We’ll iterate over possible divisors 𝑑 that are less than √2021 ≈ 44.95. Any divisor 𝑑 must satisfy 𝑑 < 1011 since if 𝑑 ≥ 1011, 2021 − 𝑑 ≤ 1010, and their product would not be 𝑛. Let’s write a Python code to iterate and find the smallest 𝑛.

We’ll check each integer 𝑑 from 2 up to 1010 (since 𝑑 ≠ 1 and 𝑑 cannot be greater than or equal to 1011) to see if 2021 − 𝑑 is an integer and if it’s greater than 𝑑. For each valid 𝑑, we’ll compute 𝑛 = 𝑑 × (2021 − 𝑑) and check if the sum of the

divisors 1 + 𝑑 + 𝑛𝑑 = 2022. We will then keep track of the smallest 𝑛 found. Let’s write the Python code to implement this approach:

def smallest_good_number(target_sum): # Using a large initial value smallest_n = float('inf')

# We start from 2 since 1 is the smallest # possible divisor for d in range(2, target_sum // 2):

if (target_sum - d) > d and d * \

(target_sum - d) > 0: n = d * (target_sum - d) if len(divisors(n)) == 3 and \

sum(divisors(n)) == target_sum: smallest_n = min(smallest_n, n)

return smallest_n

def divisors(n): divs = [] for i in range(1, int(n**0.5) + 1):

if n % i == 0: divs.append(i) if i != n // i:

divs.append(n // i) return sorted(divs)

target_sum = 2022 smallest_n = smallest_good_number(target_sum) print(smallest_n)

Output: 1349 The smallest positive integer that has three distinct positive divisors whose sum is 2022 is

.

|1349|
|---|

### D Prompts

###### Prompt used in Figure 1, Within-Step Diversity

###### System Prompt:

You are an expert evaluator for AI model outputs. Your task is to assess, for each group of answers independently, how diverse or varied the answers WITHIN that group are. Focus on whether answers in the same group share the same reasoning approach, arrive at the same conclusion, and have similar quality.

###### User Prompt:

## Question {q_trunc}

---

- ## Group A — {len(orig_answers)} answer(s) {fmt_answers(orig_answers)}

---

- ## Group B — {len(filt_answers)} answer(s) {fmt_answers(filt_answers)}

--## Task For **each group independently**, rate the INTERNAL DIVERSITY of the answers within that group on a scale from 1 to 5:

- 1 = All answers are nearly identical — same reasoning path, same final answer, just minor wording differences
- 2 = Mostly similar — same general approach but slight variations in steps or details
- 3 = Moderately diverse — some answers take a noticeably different approach or differ in quality
- 4 = Highly diverse — answers use clearly different methods, or some are correct while others are wrong
- 5 = Extremely diverse — answers are completely different in approach, quality, or conclusion Consider for each group:

- - Do all answers reach the same final answer?
- - Do they all use the same mathematical method or strategy?
- - Is there variation in correctness or reasoning quality? Respond with ONLY the following two lines (no explanation):

- Group A Score: <integer from 1 to 5>
- Group B Score: <integer from 1 to 5>

###### Prompt used in Figure 1, Across-Step Diversity

System Prompt: You are comparing two groups of answers generated by the same model at two different training stages for the same math question. User Prompt:

## Question {q} ---

- ## Group A (earlier training stage) — {len(answers_step1)} answer(s)

- {fmt(answers_step1)}

--## Group B (later training stage) — {len(answers_step2)} answer(s)

- {fmt(answers_step2)}

--## Task Compare Group B against Group A and judge whether Group B contains any NEW and DIFFERENT reasoning paths. Definition:

- - A "reasoning path" means the core line of reasoning, solution strategy, or analytical framework used to answer the question.
- - Count a path as "new" only if Group B uses a substantively different reasoning idea that does not appear in Group A.
- - Differences in wording, order, formatting, or level of detail do NOT count as a new reasoning path if the core reasoning is the same. Scoring rule:
- - Score 1: Group B has 0 new reasoning paths compared with Group A
- - Score 2: Group B has 1 new reasoning path compared with Group A
- - Score 3: Group B has 2 or more new reasoning paths compared with Group A Important:
- - Focus only on whether Group B introduces new reasoning paths relative to Group A.
- - Do not score based on correctness, quality, or final answer unless they reflect a genuinely new reasoning path.
- - Each group has at most 3 answers, so cap the score at 3. Respond with ONLY the following line (no explanation): Score: <integer from 1 to 3>

###### Prompt used in 4.5

###### User Prompt:

The following answer is wrong. Analyze the reasoning errors and categorize it into the following categories: <error_category_list>

Completely irrelevant response Misinterpretation of problem setup Incorrect mathematical modeling Unverified assumptions Probability and statistics error Incorrect use of mathematical theorems / formulas / principles Errors in basic calculations Miscounting error Missing cases or edge cases Implementation or algorithmic error Self-contradiction or abandonment of correct intermediate result

</error_category_list> <question> {question} </question> <ground_truth> {ground_truth} </ground_truth> <answer> {answer} </answer> Output format: <result>

<mistake_analysis>

Step-by-step analysis of the reasoning process and where it goes wrong. </mistake_analysis> <error_category>

One of the categories in the error category list.

</error_category> </result> </think>

### E Top-1 Eigen Ratio

We provide a formal definition of the Top-1 Eigen Ratio. Let Σ ∈ ℝ𝑑×𝑑 denote the covariance matrix of the stored logits, and let its eigenvalues be ordered as 𝜆1 ≥ 𝜆2 ≥ ··· ≥ 𝜆𝑑 ≥ 0. The Top-1 Eigen Ratio is defined as

𝜆1

Top-1 Eigen Ratio =

.

𝑑 𝑗=1 𝜆𝑗

Intuitively, 𝜆1 measures how much variance in logit space is explained by the dominant principal direction, while 𝑑𝑗=1 𝜆𝑗 corresponds to the total variance.

