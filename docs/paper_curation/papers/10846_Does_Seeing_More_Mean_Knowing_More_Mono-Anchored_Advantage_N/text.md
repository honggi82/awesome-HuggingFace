# arXiv:2605.25437v1[cs.CV]25May2026

## Does Seeing More Mean Knowing More? Mono-Anchored Advantage Normalization for Multi-Source Visual Reasoning

Fanhu Zeng1 Zhicong Luo2 Zefan Wang1 You Li3 Chi Chen1 Maosong Sun1 1Tsinghua University 2Northwest Polytechnical University 3Beijing Jiaotong University

### Abstract

Visual reasoning through reinforcement learning with verifiable rewards (RLVR) has achieved remarkable progress. However, when dealing with multi-source inputs, existing approaches tend to treat them as a mere accumulation of information, lacking explicit mechanisms to distinguish whether integrating additional sources yields information gain or introduces interference. Therefore, they struggle to effectively model dynamic interaction when integrating multiple sources, particularly when they differ significantly in physical properties and semantics, e.g., infrared and depth, leading to inferior performance to mono-source reasoning when a certain source holds the dominant signal. To address this issue, we propose MARS, a novel mono-anchored multi-source reasoning framework that models each visual modality as an independent information source. Specifically, by treating mono-source rewards as dynamic anchors, our method explicitly incorporates the information gain introduced by multi-source fusion into advantage normalization and adaptively emphasizes mutual promotion between sources while suppressing potential noise or conflicts during RLVR. From theoretical analysis, our method effectively quantifies information gain introduced by multi-source integration in gradient estimation, enabling consistent modality regulation. Empirical results also show impressive 3.2% and 4.9% performance gains on GRPO and DAPO across diverse datasets, confirming effectiveness of our method. Code is available here.

### 1 Introduction

Recent advances in multimodal large language models (MLLMs), which align representations across vision and language modalities [2], have demonstrated strong capabilities in multimodal perception and understanding [15]. More recently, visual reasoning [14, 21] has been introduced to encourage deeper thinking through reinforcement learning with verifiable rewards (RLVR), allowing models to generate structured responses with self-reflection through explicit reasoning rather than direct prediction, thereby fostering the emergence of chain-of-thought (CoT) reasoning [20] and enhancing the ability of complex understanding, multi-step reasoning, and logical consistency.

Despite the progress of visual reasoning, current methods largely optimize for aligned representations, and the complementary strengths of different sources are often assumed and overutilized, i.e., seeing more means knowing more, but potential interference or conflicts are seldom explicitly explored. In particular, existing RLVR frameworks optimize multi-source rewards directly, without explicitly assessing whether integrating additional sources yields positive information gain or instead introduces interference relative to strong mono-source reasoning, especially when their attributes and semantics have significant differences, such as medical imaging [1], autonomous driving [7], remote sensing [28], and so on. In these scenarios, naively integrating multiple sources can even lead to performance inferior to strong mono-source reasoning, when a specific source contains the dominant and

Preprint.

[Figure 1]

[Figure 2]

reliable signal. As shown in Fig. 1, in tasks where inherent physical limitations and degradation are caused by illumination variation, occlusion, and adverse weather conditions, relying solely on RGB imagery or the relationship between sources is often inadequate. In contrast, different sources such as infrared, depth, or multi-view can provide crucial and robust information with more reliable scene understanding, which requires handling multi-source data in a comprehensive manner.

[Figure 3]

[Figure 4]

[Figure 5]

In this paper, we aim to enhance the ability of visual reasoning when dealing with multi-source data. Based on the analysis, we uncover that a core reason for this limitation lies in the way current visual reasoning frameworks handle source integration. Specifically, they fail to explicitly model the performance interactions between specific source and multisource data. From an optimization perspective, these interactions correspond to whether multi-source reasoning improves or degrades performance relative to mono-source baselines, a distinction that remains invisible to advantage estimation in existing RLVR frameworks. This gap motivates the need for a general approach that can dynamically regulate contributions from a certain source.

(a) Naive Reasoning

[Figure 6]

Question

What’s in the scene?

Scene is dark,

[Figure 7]

[Figure 8]

unclear, maybe a person?

[Figure 9]

Naive Multi-Source Fusion

Multi-SourceImages

[Figure 10]

Incorrect Prediction

[Figure 11]

###### RGB

[Figure 12]

Suppress Conflict

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Mono

Infrared

RGB is vague but IR clearly shows two people …

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Depth

Multi

[Figure 22]

Accurate Prediction

Multi-Source Information Gain

Filter

(Multiple sensors, Multiple views, etc.)

Noise

Model Input

(b) Mono-Anchored Reasoning(Ours)

Reasoning Outcome

Figure 1: Illustration of multi-source visual reasoning. (a) Existing methods struggle to model dynamic interaction in multi-source data; (b) Our method explicitly uses mono-source rewards as anchors to measure the information gain from multi-source integration, enhancing reasoning and prediction.

To this end, we propose MARS, a novel multi-source reasoning framework that explicitly incorporates each visual modality as an individual information source and models the information gain introduced by multi-source integration. Concretely, by treating mono-source rewards as the anchors, it computes advantages based on information gains between multi-source and mono-source1 rewards. We theoretically analyze that our method guarantees and enables dynamic optimization that emphasizes promotion while suppressing noisy or conflicting information during training by maximizing multisource information gain. Notably, our algorithm enhances multi-source utilization from inherent capability without architectural redesign, offering a general and effective solution for improving visual reasoning performance.

We conduct experiments on various multi-source tasks, including depth, infrared, multi-view and text-rich understanding. Extensive results with notable 3.2% and 4.9% improvements on GRPO and DAPO and in-depth analyses strongly validate the effectiveness and generalizability of our method. Our contributions are summarized as follows:

- • We reveal that existing multi-source visual reasoning can systematically degrade performance, and identify relative information gain over mono-source reasoning as the key factor for effective multi-source integration from theoretical derivation.
- • We design a novel visual reasoning method that introduces mono-source rewards as anchors to quantitatively measure multi-source information gain from integration in advantage normalization, enabling adaptive regulation of different sources during RLVR training.
- • We conduct extensive experiments on various multi-source visual reasoning tasks, and the consistent and significant performance improvements on different RL algorithms validate the effectiveness and generality of our approach.

### 2 Related Work

Reinforcement Learning with Verifiable Rewards has made substantial progress in recent years, with pioneering systems such as DeepSeek-R1 [10] and Kimi [19] demonstrating that complex reasoning patterns can emerge through optimization with verifiable rewards, where outcome reward signals are used to guide the learning of long reasoning chains. Within this paradigm, some approaches focus on enhanced optimization strategies [31, 30], such as regularization, stabilized policy updates, and refined reward designs, to improve consistency and robustness. Building on these foundations, visual

1They specifically describe features of different visual modalities that differ in physical properties and semantics.

reasoning incorporates images into reasoning by coordinating linguistic reasoning with perceptual states. It achieves strong performance in vision-centric tasks such as grounding [4] and image understanding [23], highlighting it as a promising paradigm for complex multimodal understanding and deduction. In this paper, we focus on the capability of visual reasoning with multi-source data.

Multi-Source Visual Reasoning refers to tasks that require a joint understanding of images from multiple sources, potentially captured from different sensors, times or viewpoints [29]. This is crucial for real-world intelligent systems, where a single source is often insufficient for achieving completeness and reliable decisions in complex environments. Early studies focus on multi-source fusion [6, 26], where extracted features from different cameras are explicitly fused to enhance robustness and geometric consistency. More recently, multimodal large language models have reframed multi-image reasoning as unified and aligned comprehension with implicit correspondences and shared representations. Nevertheless, current attention is paid to general domain enhancement and evaluation [9, 25], which overlooks the complementarity and contradictions of multi-source data in the reasoning process.

### 3 Methodology

#### 3.1 Motivation

Visual reasoning has exhibited strong understanding capabilities under multi-image inputs. However, we observe a consistent and non-trivial phenomenon: as depicted in Fig. 1, when handling images from multiple sources, e.g., infrared, depth and so on, and only one image among multiple sources is truly informative for the task, typical multi-source reasoning often fails to capture and concentrate on the critical visual scene and therefore underperforms the upper bound of mono-source reasoning, even if all available sources are provided jointly. This contradicts the cognition of humans that integrating more information always brings more knowledge, and naturally raises an open question:

Does seeing more mean knowing more in multi-source visual reasoning? If not, how can we solve it? We attribute the issue to potential modality interference in multi-source reasoning. Specifically, a typical visual reasoning model is normally trained under the assumption of complementary data integration, i.e., seeing more images brings more knowledge, and only learns the positive guidance of multi-image fusion with implicitly unified and shared representations. Without explicitly identifying which image is causally responsible for correct decisions, it therefore struggles to capture the dynamic interaction, i.e., promotion or inference between modalities. This is especially severe in multi-source scenarios where images have different properties and semantics, resulting in unstable or noisy learning dynamics. Therefore, advantage estimation becomes unreliable under such conflict, where standard advantage normalization estimates statistics solely from multi-source trajectories, and may be dominated by spurious correlations introduced by non-informative sources.

At this point, specific mono-source reasoning often provides a significant and stable inductive signal in these scenarios. When the key image is present, a specific mono-source rollout tends to produce more consistent reward with semantic information, effectively guiding the optimization direction.

To this end, we propose to incorporate mono-source rollouts into the advantage estimation of multisource rollouts. Intuitively, mono-source reasoning acts as a general dynamic anchor to stabilize and guide multi-source reinforcement learning: (1) if it underperforms with modality conflicts, the algorithm softly regularizes trajectory updates toward the more reliable mono-source behavior; (2) moreover, if multi-source reasoning outperforms mono-source reasoning with modality mutual promotion, the algorithm also encourages exploration beyond mono-source cues.

Subsequently, Sec. 3.2 introduces details of our method, and Sec. 3.3 provides theoretical analysis.

#### 3.2 MARS: Mono-Anchored Advantage Normalization for Multi-Source Reasoning

Preliminary. For each instance consisting of question q and multiple images i in training dataset D, multi-source trajectories (rollouts) are generated through policy πθ parameterized with θ, where multiple images are jointly provided as input:

Gmulti = omultij Nj=1 . (1)

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

###### Update Action

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Both

Verifying & Group Computing

…

…

[Figure 33]

###### Policy Model

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

###### RGB

[Figure 41]

Mono-Source Anchor

|𝒒,𝒊|
|---|

Mono-Source Anchor

[Figure 42]

RGB Infrared

…

| |
|---|

[Figure 43]

Infrared

[Figure 44]

Dataset

Multi-Source Statistics

Multi-source Information Gain

- Figure 2: Structure of the proposed mono-anchored advantage normalization for multi-source visual reasoning. Mono-source rewards serve as dynamic anchors to quantify the influence of source integration with multi-source information gain in on-policy optimization.

[Figure 45]

[Figure 46]

[Figure 47]

The reward r is exploited to measure the output in response to input and each rollout is normalized by group-wise mean and variance to obtain advantage for stability:

r(q,i,omultij ) − mean(Gmulti) std(Gmulti)

. (2)

Aj =

The standard policy gradient algorithm optimizes the expected advantage function J(θ), and its policy gradient estimator [18] has the following form:

θ(q,i)[A · ∇θ log πθ(o|q,i)], (3)

∇θJ(θ) = E{q,i}∼D, o∼π

where {q,i} is the question and image from dataset D, and the policy πθ generate the trajectories for verifiable reward.

Advantage Normalization with Mono-Source Anchor. As illustrated in Fig. 2, in terms of reasoning with multi-source visual tasks, motivated by the function of mono-source rewards in advantage estimation, we additionally generate mono-source rollouts, where each image is individually paired with the textual input to produce rewards with the same policy model:

Gmono = omonoj Mj=1 . (4) In terms of advantage estimation, it is performed for multi-source rollouts only, while leveraging mono-source rewards for gradient estimation to stabilize the normalization:

r(q,i,omultij ) − mean(Gmulti ∪ Gmono) std(Gmulti ∪ Gmono)

Ahyj =

,j = 1,··· ,N. (5)

Specifically, mono-source rollouts are not used to directly update the multi-source policy. Instead, their role is to adjust the normalization statistics as an adaptive reference. Intuitively, as illustrated in the right of Fig. 2, when multi-source reward outperforms mono-source reward, the introduced estimation would lower the mean for multi-source enhancement. Conversely, if a particular modality plays a decisive role, our algorithm will inhibit the model from learning contradictory multi-source rewards and instead drive it toward better modality-specific learning.

Verifiable Rewards. The verifiable reward is a key component in reinforcement learning to align the preferences of models, which may include simple verification functions [17] that check whether predictions match the correct answers in contents and formats. Applying this concept to visual tasks requires adaptation of specific rule-based verifiable reward functions. For grounding tasks, grounding reward is directly formulated by calculating the average Intersection-over-Union (IoU) between predicted and ground truth bounding boxes:

K

1 K

IoUi, (6)

riou(q,i,o) =

i=1

where K is the number of objects in the scene, and the grounding reward consists of the iou reward and the format reward:

rgrounding = riou + rformat. (7)

[Figure 48]

[Figure 49]

[Figure 50]

Current point Standard direction

###### Zero-Shot DAPO Ours

Optimal point Mono-source anchor New direction

100

| | |
|---|---|
| | |
| | |
| | |
| | |

85

mIoU(%)

70

55

40

25

Conflict (78%) Promotion (22%)

(a) Conflict

###### (b) Promotion

- Figure 3: Illustration of mono-source anchor in optimization. It provides a consistent facilitation in both modality conflict and promotion. This results in significant gain across model scales.

Union Multi Ours

[Figure 51]

100

| |
|---|
| |
|whether the output|
| |
| |

86

mIoU(%)

In visual question answering tasks, the accuracy reward is determined by matches the ground truth:

72

58

44

1 ,if ground truth in o, 0 ,otherwise.

(8) The final reward is a combination of accuracy reward and format reward:

racc(q,i,o) =

rvqa = racc + rformat. (9)

Remark. In the case of multi-source visual reasoning, instead of estimating the baseline from multi-source rewards alone, our algorithm computes hybrid statistics over the union of mono-source and multi-source rewards:

R = omono1 ,...,omonoM ,omulti1 ,...,omultiN , (10)

from the same policy model. This can be seen as on-policy optimization with a hybrid distribution. Concretely, by leveraging mono-source rewards as anchors, our method precisely utilizes the difference between trajectories, thereby enhancing performance with exact information gain from mono-source to multi-source rewards shown in Fig. 4b.

#### 3.3 Theoretical Analysis

We present several theoretical analyses of the proposed mono-anchored advantage normalization algorithm from policy optimization perspective to construct key properties for stability and rationality.

- Theorem 3.1 (Unbiasedness). For any measurement of gradient estimation, the expectation of our algorithm for is equivalent to the expectation of on-policy optimization:

Eq,i∼D, o∼π

θ(q,i)[Ahy · ∇θ log πθ(·|q,i)] = ∇θJ(θ). (11)

This provides a theoretical guarantee that the proposed algorithm introduces no bias for gradient estimation under the condition of on-policy optimization from the perspective of expectation, ensuring stability.

- Theorem 3.2 (Gradient Decomposition). The gradient optimization based on mono-anchored advantage normalization is equivalent to maximizing the multi-source information gain while optimizing the standard multi-source reward:

∇θJhy(θ) = ∇θJ(θ) + (1 − α)∆IG∇θJreg(θ), (12)

where conventional advantages have a zero mean. α = N/(M +N) is the proportion of multi-source rollouts representing the strength of guidance. ∆IG = mean(Gmulti)−mean(Gmono) is the expectation of the reward increment of multi-source trajectories compared to mono-source trajectories. It measures the relative information gain from multiple image fusion relative to mono-source reasoning, and a negative value indicates conflict between modalities that perform inferior to mono-source reasoning.

The derivation holds for any α ∈ [0,1] and justifies the practical effectiveness of the advantage estimation scheme combining both sample types through unified standardization. This reveals that our algorithm is theoretically optimizing a weighted multi-source reward with a multi-source information

0.40

1.58

| |
|---|
| |
| |
| |
|Converged Entropy<br><br>|

Multi-Source Information Gain

0.34

1.52

Entropy

Reward

1.46

0.28

1.40

0.22

GRPO

1.34

0.16

Ours

1.28

0.10

0 10 20 30 40

0 10 20 30 40

Step

Step

(a) Entropy during training (b) Reward

- Figure 4: Learning statistics of (a) entropy and (b) reward when training vanilla GRPO and our algorithm. Our method achieves better performance with multi-source information gain from a mono-source anchor while maintaining stability.

###### Reward

1.58

1.52

gain regularization. By leveraging the mono-source reward as the anchor, it dynamically adjusts the standard gradient direction according to multi-source information gain during the optimization procedure, which guides the optimization towards an optimal point with faster convergence and better multi-source performance. Fig. 3 gives an intuitive illustration of mono-source anchor in optimization. In conflict where certain source performs well, it is around the optimal point, and pull the optimization direction close by information gain. This greatly improves the performance, where standard multi-source reasoning undergoes a severe performance drop. The process is similar in promotion that pushes the direction away from mono-source anchor towards the optimal point, providing a consistent facilitation in multi-source visual reasoning. The quantatative results show the commonalities of conflict and the utility of our method in resolving conflict rather than a general effect.

1.46

1.4

GRPO

1.34

Remark. Inherently, we study policy optimization with a mono-anchored advantage for multi-image reasoning. Instead of universally increasing rewards or advantages, it enforces a principled crossmodal comparison: multi-source rollouts receive positive updates if and only if they outperform monosource reasoning. This is consistent with the motivation that multi-source reasoning improves when it provides complementary information, while being regularized otherwise. Therefore, MARS reduces the blind exploration of multi-source policies, accelerates convergence, and improves robustness against possible visual inconsistency, yielding a more stable and interpretable optimization trajectory for multi-source reasoning models as shown in Fig. 4.

Simplicity and Stability. Our algorithm only requires one policy model for optimization and does not introduce additional storage for models or samples, as opposed to experience sampling [27] or off-policy correction [22] methods. In practical implementation, we generate the mono-source samples by modifying image inputs and obtain the normalized statistics without computing gradients, thereby maintaining algorithmic efficiency and stability as shown in Tab. 3.

### 4 Experiments

#### 4.1 Experimental Setup

Datasets. Regarding the datasets, we employ diverse multi-source datasets. For visual modalities, in addition to typical RGB images, we incorporate four different modalities, including depth SpatialQA [8], infrared LLVIP [11], multi-view nuScenes [5] and text-rich OCR-VQA [13].

Baselines. For previous algorithms, we compare with various methods with [23, 16] and without [12,

- 3] reinforcement post-training. In addition, for supervised post-training, SFT and CoT are also incorporated for comprehensive comparison. Regarding reinforcement post-training, we employ GRPO [17] and DAPO [24], which are two typical group-based reinforcement learning algorithms for visual reasoning. Since our method does not rely on a specific training framework, unless otherwise stated, all comparative experimental results are conducted within the same basic structure.

Implementation details. We use Qwen2.5-VL-3B [3] as the base model for supervised and reinforcement post-training. We mainly conduct the experiments on visual question answering (VQA) and grounding, and the evaluation metrics are accuracy and mIoU, respectively. The concrete calculation is similar to that for verifiable rewards. For a comprehensive understanding, in addition to standard multi-source visual reasoning that jointly take all images as inputs (Multi), we furthermore perform mono-source reasoning (Union), i.e. reason with each single source and then obtain the best result as

- Table 1: Overall performance of four different multi-source tasks based on Qwen2.5-VL-3B. We compare with previous methods, supervised and reinforcement post-training methods, respectively. Union denotes the best performance among all mono-source results. The best results within a comparable group are in Bold.

Infrared Depth Multi-View Text-Rich Avg. Union Multi Union Multi Union Multi Union Multi Union Multi

Model

Previous Methods

Qwen2.5-VL-3B [3] 63.0 38.8 86.3 73.8 82.0 74.0 69.0 51.0 75.1 59.4 LLaVA-OV-7B [12] 4.9 2.1 80.0 56.3 88.0 44.0 81.0 67.0 63.5 42.3 R1-Onevision-7B [23] 19.7 11.1 87.5 72.5 90.0 81.0 77.0 58.0 68.5 55.6 VisionReasoner-7B [16] 82.5 75.2 85.0 78.8 85.0 70.0 62.0 45.0 78.6 67.2

Supervised Post-Training

SFT 87.1 84.5 85.0 65.0 90.0 78.0 83.0 78.0 86.2 76.4 Chain-of-Thought 88.2 84.2 90.0 77.5 90.0 75.0 77.0 76.0 86.3 78.2

Reinforcement Post-Training

GRPO [17] 89.7 86.0 80.0 70.0 88.0 72.0 77.0 71.0 83.6 74.7 GRPO+Ours 89.9 87.7 80.0 73.8 90.0 75.0 79.0 75.0 84.7 (+1.1) 77.9 (+3.2)

DAPO [24] 88.8 85.5 76.3 73.8 92.0 74.0 75.0 70.0 83.0 75.8 DAPO+Ours 92.3 89.4 82.5 77.5 88.0 81.0 78.0 75.0 85.2 (+2.2) 80.7 (+4.9)

final performance, as the upper bound to showcase the utility of information gain in performance. Concretely, we separately take every single source as input for visual reasoning during inference, and consider it to be correct if any single source correctly answers. We uniformly generate one trajectory for each visual source, i.e., M = image_num and N = 12.

#### 4.2 Main Results

We mainly conduct the experiment on multi-source datasets. Furthermore, we extend our algorithm to the application scenarios, the reinforcement post-training strategies and the scale of the model to comprehensively validate the effectiveness of the proposed method.

MARS is effective across various multi-source visual reasoning datasets. We evaluate the proposed advantage algorithm on diverse visual reasoning tasks with multi-source datasets. It can be seen in Tab. 1 that: (1) the performance of Union exhibits better performance than Multi by a substantial margin across all tasks, which can be seen as the multi-source information gain, showing great potential to utilize for performance enhancement in multi-source tasks; (2) higher Union does not mean better performance in Multi, especially in supervised post-training methods, including SFT and CoT and naive RLVR methods. This strongly demonstrates the dynamic interaction between different sources is not well modeled by existing methods, leading to performance inferior to monosource upper bound, which is consistent with our topic that seeing more does not mean knowing more in multi-source visual reasoning; (3) compared to supervised post-training, better reinforcement learning strategy does not necessarily achieve better performance for Multi, revealing the difficulty of the task and highlighting the necessity for designed algorithms; (4) in contrast to standard visual reasoning that even achieves no gains in some scenarios, our method consistently outperforms baselines and obtains substantial improvements on multi-source visual reasoning. Specifically, on the most important multi-source reasoning metric, Multi, our approach significantly boosts all tasks, with notable 3.8% and 7.0% improvements on infrared and multi-view datasets, respectively. Also, our method further facilitates the upper bound of Union, bringing in coherent improvements. This confirms that dynamically leveraging mono-source rewards as anchors effectively guides the model to focus on informative sources while suppressing noise from less relevant images, and therefore achieves an impressive performance enhancement, firmly validating the effectiveness of our method.

Different training strategies. Also, we validate the versatility of our approach by integrating it into two distinct reinforcement learning frameworks, i.e., GRPO and DAPO. As indicated in bottom of Tab. 1, our method obtains substantial improvements under both settings. Concretely, our method achieves an average performance gain of 3.2% in GRPO and 4.9% in DAPO on visual reasoning with various vision sources. Moreover, our algorithm also boosts the performance of mono-source

|| |
|---|
| |
| |
| |
| |
| |
<br><br>60<br><br>66<br><br>72<br><br>78<br><br>84<br><br>90<br><br>96<br><br>1 2<br><br>Qwen2.5-VL-3B<br><br>LLVIP OCR<br><br>|60<br><br>66<br><br>72<br><br>78<br><br>84<br><br>90<br><br>96|1 2<br><br>Qwen2.5-VL-7B<br><br>DAPO<br><br>Ours<br><br>LLVIP OCR<br><br>|
|---|---|---|

|84<br>85<br>86<br>87<br>88<br>89<br><br><br>Normal Gaussian Blur Occlusion<br><br>mIoU(%)<br><br>Type of Degradation<br><br>GRPO<br><br>Ours|
|---|

|84<br><br>86<br><br>88<br><br>90<br><br>92<br><br>0 1 2 3 4 5 6<br><br>mIoU(%)<br><br>Number of Mono-Source Samples<br><br>Union<br><br>Multi|
|---|

mIoU(%)

mIoU(%)

(a) (b) (c)

- Figure 5: (a) Performance across different model sizes. (b) Influence of different numbers of monosource samples in grounding. (c) Performance comparison under different visual degradations.

reasoning by 1.1% and 2.2%, respectively. It highlights that our method is independent of the reinforcement learning framework, and only requires additional mono-source reward signals during advantage estimation without modifying the core training objectives, which shows great generalization ability and strong promise to serve as a plug-and-play integration to enhance multi-source performance in different training strategies.

Generalizability across different model sizes. To further assess the scalability, we validate our method with model sizes comprising 3B and 7B parameters, respectively. As shown in Fig. 5a, we can conclude that (1) larger models benefit from larger knowledge capacity and typically achieve a better performance; (2) the proposed advantage algorithm brings consistent and substantial improvement across scales, e.g., a 4.4% gain for small model and 2.8% for large model on average. This shows that the stable performance promotion stems from the inherent adaptability of the method by advantage estimation, making it agnostic to variation of model size, which suggests that the approach is not limited by model capacity and generalizes effectively.

#### 4.3 Ablation Study and Further Analysis

We conduct extensive ablation studies and in-depth analyses to validate the effectiveness of our method and provide insight into the contribution of the core components under different conditions.

- Table 2: Detailed statistics of reward in different algorithms.

Table 3: performance and efficiency balance with different numbers of trajectory generation.

Statistics Qwen2.5-VL GRPO Ours

Number of Trajectory N M + N Ours

max(r(omulti)) (↑) 1.67 1.69 1.68 mean(r(omulti)) (↑) 1.49 1.51 1.62 max(r(omono)) (↑) 1.55 1.56 1.63

GPU Hours 13 ×8 17 ×8 14×8 mIoU (%) 86.0 86.2 87.7

Incorporating more mono-source samples. In the main results, we take one momo-source reward from every single source as the dynamic anchor. We investigate how the number of available monosource samples, i.e., M, influences overall performance as the number of mono-source anchors employed in advantage normalization indicates the strength of the multi-source information gain towards robust visual reasoning. Results are shown in Fig. 5b, where M = 0 represents the vanilla algorithm. It reveals that visual reasoning accuracy saturates as M increases and additional samples yield diminishing returns. It implies that a small set of mono-source trajectories is adequate to provide diverse information gain. Conversely, a larger M does not necessarily bring gains, as mono-source samples may dominate the advantage estimation with unstable direction, leading to higher dynamics against multi-source reasoning in policy updates. Thus, we employ M = 2, which is the number of sources, for stable training. More importantly, guiding multi-source learning without requiring exhaustive sampling brings negligible computational or storage overhead, showing the efficiency.

Impact of reward statistics. We compare different statistics of reward in Tab. 2 to investigate the changes in intuitive indicators. The results reveal that instead of increasing the best trajectory (maximum multi-source reward), MARS significantly improves the average quality of multi-source reasoning (1.49 to 1.62), approaching the upper bound of mono-source reward. This shows that our method indeed exploits multi-source information gain to guide the optimization towards a better direction rather than stochastic exploration with diversity, which demonstrates the effectiveness of mono-source anchor in modeling the dynamic interaction. Also, the simultaneous improvement in mono-source trajectory (1.55 to 1.63) during training uncovers the efficiency of the on-policy generation strategy without introducing additional model overload.

Grounding GRPO VQA

###### GRPO

<think> Color Image depicts an urban street at night... Infrared Image provides better contrast... From the infrared image, we observe one person riding a motorcycle/scooter on the right-hand side of the frame...

<think> The RGB image shows.... The depth image is entirely white, indicating no discernible depth information... The entire analysis relies solely on the RGB image... Based on the RGB image alone, the frisbee is likely

[Figure 52]

[Figure 53]

| |
|---|

The bounding box should encapsulate the full height of the person,

###### still in the man's hands. Without depth data, we cannot confirm

precise spatial positioning… Therefore, the evidence supports the conclusion that the frisbee has not left the man's hands. </think> <answer> No </answer> <think> The scene shows… The depth image is entirely white, indicating no discernible depth information… Our analysis relies solely on the RGB image… Based on the RGB image alone, the frisbee's position indicates it has left the man's hands. Without depth data, we rely

starting just above the handlebars of the motorcycle/scooter and extending down past the lower part of the vehicle. </think> <answer> [998, 127, 1165, 448] </answer>

Ours

###### Ours

| |
|---|

[Figure 54]

###### <think>

[Figure 55]

| |
|---|

Color Image depicts… Infrared Image provides… From the color image,

we see a person riding a motorcycle/scooter on the right side of the frame. Their presence is faintly visible due to the low light... Person 2 is only visible in the infrared image, this person seems to be standing or walking near the motorcycle/scooter. They appear smaller compared to the rider and are located closer to the center-right of the frame.

###### on the visual evidence of its flight path. The absence of depth

information does not affect this interpretation since the relevant details are provided by the RGB image. </think> <answer> Yes </answer>

Q: Detect all person in the scene and return their locations in the

</think>

Q: Has the frisbee left the hands of the man?

<answer> [[835, 1, 976, 154], [1015, 131, 1139, 340]] </answer>

form of coordinates.

- Figure 6: Qualitative results of visual reasoning on multi-source datasets. In grounding with RGB and infrared images and VQA with RGB and depth images, GRPO would excessively rely on the RGB image, resulting in improper predictions. Our method benefits from multi-source information gain and is capable of adaptively focusing on the key images with correct responses.

Performance-efficiency trade-off. Our algorithm requires additional trajectory generation in practical implementation during reinforcement post-training. To certify the key component in effectiveness, we additionally perform GRPO with rollout to be M + N and compare the training efficiency in Tab. 3. The results show that our method, which includes only N samples in advantage normalization, introduces moderate training overhead. By contrast, incorporating more actual trajectory in policy update brings marginal improvement (0.2%) with substantial overhead (30%). This strongly suggests that the key to this approach is the mono-source anchor rather than introducing more rollouts, showcasing both effectiveness and efficiency.

Robustness to visual degradation. As our method dynamically integrates interaction between modalities, and concentrates on a specific source if another displays no information gain, it shows great promise in exhibiting robustness to possible visual degradations of certain sources, i.e., noise, illumination, occlusion and so on. To simulate real-world conditions where visual inputs may be corrupted, we randomly degrade image quality by adding Gaussian noise, motion blur, and occlusion to the input images. As is illustrated in Fig. 5c, our method maintains superior performance under all degradation types, e.g., only 0.5% drop in accuracy under severe Gaussian noise. Moreover, in the case of motion blur and occlusion, our method even obtains 0.8% performance gain, which outperforms the baseline by a substantial 1.3% improvement, showcasing its strong robustness and effectiveness. This can be attributed to our way of advantage normalization, which explicitly reduces the influence from unreliable source and relies more on stable, informative modality, thereby exhibiting adaptability and generalization ability.

Case Study. We visualize the reasoning process on several representative scenarios in Fig. 6 to qualitatively validate the effectiveness of our method. Specifically, in grounding task, our method successfully prioritizes key information in infrared image and detects the specific person that is hard to see in RGB, yielding more precise bounding boxes than GRPO. Moreover, in VQA task with RGB and depth images, while vanilla GRPO is uncertain about the answer without clearly analyzing depth image, our approach is confident about the inference relying solely on the RGB image and generates correct responses accordingly. The intuitive visualizations clearly show how our algorithm guides the model to weigh different sources appropriately, showcasing the effectiveness and superior capability.

### 5 Conclusion

In this paper, we revisit visual reasoning from a multi-source perspective, where visual modalities differing significantly in physical properties and semantics exhibit both promotion and conflict characteristics that are insufficiently modeled by existing methods. To address this issue, we propose a multi-source visual reasoning framework that adaptively emphasizes informative interaction while suppressing conflicting ones using mono-source rewards as dynamic anchors under RLVR training. This theoretically formulates the multi-source information gain of integration to guide the optimization towards stable and advanced trajectories. Comprehensive experiments across diverse datasets and training strategies with significant and consistent performance improvements strongly validate the effectiveness, efficiency and generalizability of the proposed approach.

### References

- [1] Muhammad Adeel Azam, Khan Bahadar Khan, Sana Salahuddin, Eid Rehman, Sajid Ali Khan, Muhammad Attique Khan, Seifedine Kadry, and Amir H Gandomi. A review on multimodal medical image fusion: Compendious analysis of medical modalities, multimodal databases, fusion techniques and quality metrics. Computers in biology and medicine, 144:105253, 2022.
- [2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [4] Sule Bai, Mingxing Li, Yong Liu, Jing Tang, Haoji Zhang, Lei Sun, Xiangxiang Chu, and Yansong Tang. Univg-r1: Reasoning guided universal visual grounding with reinforcement learning. arXiv preprint arXiv:2505.14231, 2025.
- [5] Ankan Bansal, Yuting Zhang, and Rama Chellappa. Visual question answering on image sets. In European Conference on Computer Vision, pages 51–67. Springer, 2020.
- [6] Martin Brenner, Napoleon H Reyes, Teo Susnjak, and Andre LC Barczak. Rgb-d and thermal sensor fusion: A systematic literature review. IEEE Access, 11:82410–82442, 2023.
- [7] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020.
- [8] Wenxiao Cai, Iaroslav Ponomarenko, Jianhao Yuan, Xiaoqi Li, Wankou Yang, Hao Dong, and Bo Zhao. Spatialbot: Precise spatial understanding with vision language models. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 9490–9498. IEEE, 2025.
- [9] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pages 148–166. Springer, 2024.
- [10] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [11] Xinyu Jia, Chuang Zhu, Minzhen Li, Wenqi Tang, and Wenli Zhou. Llvip: A visible-infrared paired dataset for low-light vision. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3496–3504, 2021.
- [12] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [13] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024.
- [14] You Li, Chi Chen, Yanghao Li, Fanhu Zeng, Kaiyu Huang, Jinan Xu, and Maosong Sun. Imagination helps visual reasoning, but not yet in latent space. arXiv preprint arXiv:2602.22766, 2026.

- [15] You Li, Heyu Huang, Chi Chen, Kaiyu Huang, Chao Huang, Zonghao Guo, Zhiyuan Liu, Jinan Xu, Yuhua Li, Ruixuan Li, et al. Migician: Revealing the magic of free-form multi-image grounding in multimodal large language models. arXiv preprint arXiv:2501.05767, 2025.
- [16] Yuqi Liu, Tianyuan Qu, Zhisheng Zhong, Bohao Peng, Shu Liu, Bei Yu, and Jiaya Jia. Visionreasoner: Unified visual perception and reasoning via reinforcement learning. arXiv preprint arXiv:2505.12081, 2025.
- [17] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [18] Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.
- [19] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025.
- [20] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [21] Guowei Xu, Peng Jin, Ziang Wu, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason step-by-step. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2087–2098, 2025.
- [22] Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason under off-policy guidance. arXiv preprint arXiv:2504.14945, 2025.
- [23] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025.
- [24] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- [25] Youngjoon Yu, Sangyun Chung, Byung-Kwan Lee, and Yong Man Ro. Spark: Multi-vision sensor perception and reasoning benchmark for large-scale vision-language models. arXiv preprint arXiv:2408.12114, 2024.
- [26] Maoxun Yuan, Xiaorong Shi, Nan Wang, Yinyan Wang, and Xingxing Wei. Improving rgbinfrared object detection with cascade alignment-guided transformer. Information Fusion, 105:102246, 2024.
- [27] Runzhe Zhan, Yafu Li, Zhi Wang, Xiaoye Qu, Dongrui Liu, Jing Shao, Derek F Wong, and Yu Cheng. Exgrpo: Learning to reason from experience. arXiv preprint arXiv:2510.02245, 2025.
- [28] Jixian Zhang. Multi-source remote sensing data fusion: status and trends. International journal of image and data fusion, 1(1):5–24, 2010.
- [29] Lili Zhang, Yuxiang Xie, Luan Xidao, and Xin Zhang. Multi-source heterogeneous data fusion. In 2018 International conference on artificial intelligence and big data (ICAIBD), pages 47–51. IEEE, 2018.
- [30] Yi-Fan Zhang, Xingyu Lu, Xiao Hu, Chaoyou Fu, Bin Wen, Tianke Zhang, Changyi Liu, Kaiyu Jiang, Kaibing Chen, Kaiyu Tang, et al. R1-reward: Training multimodal reward model through stable reinforcement learning. arXiv preprint arXiv:2505.02835, 2025.
- [31] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

