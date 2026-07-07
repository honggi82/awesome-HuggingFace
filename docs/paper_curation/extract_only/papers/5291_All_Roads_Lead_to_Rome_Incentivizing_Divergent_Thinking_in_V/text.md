## All Roads Lead to Rome: Incentivizing Divergent Thinking in Vision-Language Models

Xinyu Tian1 Shu Zou1,2 Zhaoyuan Yang3 Mengqi He1 Peter Tu3 Jing Zhang1 1Australian National University 2Shanghai AI Lab 3GE Research

1firstname.lastname@anu.edu.au, 3firstname.lastname@ge.com

# arXiv:2604.00479v1[cs.CV]1Apr2026

### Abstract

Recent studies have demonstrated that Reinforcement Learning (RL), notably Group Relative Policy Optimization (GRPO), can intrinsically elicit and enhance the reasoning capabilities of Vision-Language Models (VLMs). However, despite the promise, the underlying mechanisms that drive the effectiveness of RL models as well as their limitations remain underexplored. In this paper, we highlight a fundamental behavioral distinction between RL and base models, where the former engages in deeper yet narrow reasoning, while base models, despite less refined along individual path, exhibit broader and more diverse thinking patterns. Through further analysis of training dynamics, we show that GRPO is prone to diversity collapse, causing models to prematurely converge to a limited subset of reasoning strategies while discarding the majority of potential alternatives, leading to local optima and poor scalability. To address this, we propose Multi-Group Policy Optimization (MUPO), a simple yet effective approach designed to incentivize divergent thinking across multiple solutions, and demonstrate its effectiveness on established benchmarks. Project page: https://xytian1008.github.io/MUPO/.

### 1. Introduction

Since the advent of Large Language Models (LLMs) [27, 30, 37], reasoning has emerged as a critical capability for addressing complex tasks. Early approaches typically rely on prompt engineering [16, 57], which elicit chain-ofthought solutions, or on Supervised Fine-Tuning (SFT) [6, 22], where models are trained on high-quality trajectories to emulate human-like thinking. More recently, the rise of Reinforcement Learning (RL), notably Group Relative Policy Optimization (GRPO) [26, 28], facilitates self-reflection and verification, highlighting remarkable self-improving capabilities. Consequently, researchers have increasingly sought to integrate reasoning into Vision-Language Models (VLMs) [1, 10, 45]. By leveraging RL, VLMs acquire the ability to extract, analyze and reflect over visual information, achieving significant gains on both challenging logical

visual questions [13, 47] and traditional tasks [18, 29].

However, despite the promise, the underlying mechanisms driving the effectiveness of RL models, as well as their limitations, remain underexplored. Early doubts arise from LLMs, where [52] observe that although RL models demonstrate higher accuracy, their performance ceilings remain constrained by the capabilities of base models, lacking the ability to expand novel reasoning patterns. More recently, in the context of VLMs, [17, 43] notice that incorporating reasoning, in some cases, leads RL models to underperform base models. These findings lead us to ask: Do RL models truly outperform their base counterparts?

Motivated by this, we conduct a behavioral comparison between RL and base models on established benchmarks. Our results indicate that, when limited to a single attempt, RL models generally achieve higher accuracy than their base counterparts. However, when multiple samplings are permitted, base models are consistently capable of solving a broader number of problems. Notably, in failure cases where RL models are unable to handle despite multiple attempts, we find that base models often succeed by leveraging through diverse and alternative reasoning pathways that are not captured by RL models. For instance, as shown in Fig. 1, for geometry problems, RL models tend to rely exclusively on equation solving, which is prone to logical errors. In contrast, base models are capable of proposing simpler, verification-based strategies. Similarly, in object counting involving large quantities, RL models consistently adopt tedious sequential enumeration. Base models, however, are often able to employ efficient elimination strategies that reach the correct answer in significantly fewer steps.

These observations reveal a fundamental distinction between RL models and their vanilla base counterparts. During reasoning, RL models, despite demonstrating deeper deliberation, tend to be conservative, often adhering to a dominant strategy. In contrast, base models, although less refined along a single path, display divergent thinking, frequently exploring potential alternative solutions. Intuitively, the latter more closely aligns with human problem-solving, in which individuals, when given multiple attempts, often approach the same problem from varied perspectives, increas-

[Figure 1]

[Figure 2]

###### Question

###### Base Model

###### RL Model

A

To solve it, we first establish necessary equations:

To find AD, we may set up equations first:

1. Let BD = x, so we have CD = BC – BD = 14 – x.

1. We set BD = x, given BC = 14, we have CD = 14 – x.

... (solving equations) Therefore, the solution is x = 9.8, then AD = 11.3.

... (solving equations) We may derive the solution x = 10.8, and AD = 10.4.

15 13

…

…

D

B

14 C

Given AD as height, we can solve this by validation: 1. We can validate the options by assuming AD = 12. ... (validating options)

We start by formulating equations to solve AD:

1. Given BC = 14, we denote CD = x, then BD = 14 – x.

In the triangle, find the height AD corresponding to side BC. A. 12 B. 11 C. 10 D. 9

... (solving equations) Finally, we get CD = 7.4, the height AD = 10.7.

Thus, AD = 12 since it matches the given conditions.

[Figure 3]

[Figure 4]

###### Question

###### RL Model

###### Base Model

[Figure 5]

Let’s count the cylinders in this photo one by one:

Let’s find out the cylinder numbers step-by-step:

1. There is a large deep blue cylinder in the left.

1. A tall yellow cylinder is shown in the top right.

... (counting cylinders) In conclusion, there are 18 cylinders in the photo.

... (counting cylinders) Summing over the above, there are 16 cylinders.

… …

To solve this, we may count them by exclusion:

I will perform sequential counting of the cylinders:

Here is a total of 19 objects. How many cylinders are there? A. 15 B. 16 C. 17 D. 18

1. I can see a small red sphere in the bottom right.

1. A large green cylinder appears at the bottom.

... (counting other objects) By subtraction, we have 19 – 2 = 17 cylinders.

... (counting cylinders) In summary, we have 15 cylinders in this image.

- Figure 1. The failure cases of RL models. We use Vision-R1 [13] as a representative RL model, with its corresponding base model being Qwen2.5-VL-7B [1]. The examples are selected from MathVerse [54] and MathVista [20], respectively. For each question, we set the sampling temperature to 1.0 and generate multiple responses, each of which is displayed in a gray box. Main differences in the proposed reasoning strategies are annotated in blue and pink, while correct and incorrect answers are highlighted in green and red, respectively.

duces diversity reward to promote greater separation among groups. Intuitively, each group serves as a distinct realization of a reasoning strategy, and MUPO aims for the model to not only generate a range of diverse strategies but also to refine each effectively. Our result model, MUPO-Thinker7B, demonstrates the ability to explore diverse reasoning paths in search of globally optimal solutions and achieves average gains of 2 ∼ 7% over strong baselines on established benchmarks, setting a new state of the art. In summary, our contributions are as follows:

ing their chances of success, especially on challenging tasks where common strategies may fail or be error-prone. Our further experiments reveal that models also exhibit similar patterns, where greater diversity in reasoning significantly enhances the probability of reaching correct answers.

The above findings indicate that models trained with RL, i.e., GRPO, appear to forgo the inherent divergent thinking exhibited by their base counterparts. To further investigate the cause of this shift, we examine the evolution of reasoning diversity throughout the training process. We observe that during the early training stage, diversity declines sharply to a negligible level, suggesting the model rapidly converges on a narrow set of strategies while discarding the vast majority of potential paths. As a result, the model concentrates on optimizing this limited subset for most of the training period. This collapse in diversity has two issues: 1) exploitation over exploration, where the model prioritizes a dominant strategy over exploring alternative modes, leading to local optima; 2) poor scalability, where convergent reasoning struggles to cover the broad spectrum of problems, thereby constraining test-time scaling capabilities.

- • We highlight a fundamental difference in reasoning behavior between RL models and base models: the former engage in deeper yet narrowly focused reasoning, whereas the latter, despite being less sophisticated, exhibit broader and more diverse thought patterns.
- • We find that GRPO is prone to diversity collapse, causing the model to search within a narrow set of strategies while disregarding the majority of potential alternatives, leading to local optima and limited scaling capabilities.
- • We propose MUPO, a straightforward yet effective policy algorithm designed to incentivize divergent thinking across multiple solutions, and demonstrate its effectiveness through comprehensive experimental validation.

Based on the above, a natural question is: can we preserve divergent thinking from base models during RL, enabling them to reason deeply about individual solutions while also maintaining a repertoire of diverse strategies? To address this, we propose Multi-Group Policy Optimization (MUPO), a drop-in replacement of GRPO to incentivize divergent reasoning. Inspired by the diversity collapse in GRPO, we partition model responses into multiple groups. Instead of computing advantages globally, MUPO performs localized advantage estimation within each group and intro-

### 2. Related Work

Reasoning in VLMs. Enhancing reasoning capabilities has become a critical objective in the pursuit of general intelligence. In the context of VLMs, this progression evolves from prompt engineering which elicits chain-ofthought reasoning [16, 33, 35, 57], to SFT on high-quality, human-designed trajectories [3, 22]. More recently, the

70

60

54

65

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

Base Model

Base Model

51

61

GRPO Model

GRPO Model

60

50

48

57

Acck(%)@

45

50

53

40

42

49

40

30

39

45

36

30

41

20

0.05 0.25

0.03 0.23

HallBench LogicVista MathVerse WeMath

HallBench LogicVista MathVerse WeMath

Average Reasoning Diversity Average Reasoning Diversity

(A) Acc@k Performance (3B) (B) Acc@k Performance (7B) (C) Acc@4 vs. Diversity (3B) (D) Acc@4 vs. Diversity (7B)

- Figure 2. The impact of reasoning diversity on model performance. In (A) and (B), we report acc@k for both RL and base models on established benchmarks, with color intensity decreasing as k = 1, 2, 4. In (C) and (D), we plot relationship between reasoning diversity and the corresponding acc@4 scores. Each point is based on a set of 4 responses, and a regression line is fitted to capture the overall trend.

advent of RL, particularly GRPO [26, 28], has shifted the paradigm from passive distillation toward active selfimprovement, empowering models to autonomously discover and refine optimal strategies. Consequently, a growing body of research explores reasoning via diverse RL strategies [12, 23, 32], dataset construction [21, 36, 47, 48], and reward shaping [42, 46, 49, 58]. Despite these advances, the mechanisms of RL models and their limitations remain underexplored. In this work, we show that while RL models exhibit higher accuracy, they tend to forgo the divergent thinking of base models, thereby limiting the potential to solve a broader range of problems.

posed method, MUPO, bridges this gap by integrating parallel thinking into RL, yielding significant gains in both accuracy and scalability.

### 3. Exploring Divergent Thinking in VLMs

In this section, we begin with a comparison between existing RL models and their base counterparts, focusing on diversity as a key axis. We examine how variations in reasoning diversity influence model performance. Building on this, we further investigate the dynamics of diversity throughout the GRPO training process, revealing a critical issue where models tend to optimize within a narrow subset of strategies while discarding other potential alternatives.

RL for Reasoning. As researchers notice that sparse yet easily verifiable rewards, e.g., accuracy and format, yield unexpectedly strong performance in RL, subsequent studies have emerged aiming to advance GRPO along multiple dimensions. For instance, [50, 53] adopt a sampling-based perspective, prioritizing informative trajectories to enhance training efficiency. In parallel, [14, 34, 59] focus on visual features, promoting perceptually-aware reasoning through fine-grained reward design. Recently, another line of research [7, 15, 40] explores the integration of uncertainty via entropy mechanisms to encourage exploration. However, these approaches fail to foster reasoning across divergent strategies. In contrast, we propose MUPO, which enables models to reason with both depth and breadth, ultimately guiding them toward the discovery of optimal solutions.

##### 3.1. Divergent vs Convergent

To assess the impact of reasoning diversity to model performance, here we conduct a simple motivational experiment. Experimental Setup. We consider models at two scales: 3B and 7B. For the 3B setting, we select VLAA-Thinker [4] and LMM-R1 [24], while for the 7B setting, we consider Vision-R1 [13] and R1-OneVision [47], with Qwen2.5-VL3B and Qwen2.5-VL-7B being the base models [1]. All models are evaluated on a suite of reasoning-centric benchmarks, including MathVerse [54], LogicVista [44], WeMath [25] and HallusionBench [9]. The sampling temperature is set to 1.0 to enable generation of multiple responses. Metric. Beyond standard accuracy that only evaluates a single path, we introduce a more relaxed metric, acc@k, which is considered positive if at least one of k sampled trajectories leads to the correct answer. We set k = 1, 2 and 4 to assess the model’s ability to reach the correct answer when given multiple attempts. To quantify reasoning diversity, we employ Qwen3-Embedding-0.6B [55] to encode the reasoning segments of generated responses and compute cosine distances to measure differences. The diversity across multiple trajectories is then calculated as the pairwise average.

Test-Time Scaling. Given the substantial expense of training, allocating extra compute at test time is becoming a cost-effective strategy to enhance performance [22, 31]. In the case of VLMs, this manifests as allowing for more thinking budgets. Such scaling law can be categorized into two types: sequential and parallel. The former promotes deeper reasoning through patterns such as step-bystep generation, self-reflection, and verification [11, 18, 38]. The latter emphasizes divergent exploration by sampling multiple candidates and aggregating via self-consistency or verifiers [41, 56]. However, existing RL algorithms predominantly focus on sequential thinking, i.e., refining along a single reasoning path, while neglecting alternative branches, which limits their scaling potentials. Our pro-

Results. Fig. 2 displays the impact of reasoning diversity on models, from which we may derive following key insights. RL models dive depth, base models seek breadth. As shown in Fig. 2 (A) and (B), when k = 1, RL models markedly outperform their base counterparts, reflecting so-

0.20

Reward Policy

[Figure 6]

[Figure 7]

[Figure 8]

Qwen2.5-VL-3B (GRPO) Qwen2.5-VL-7B (GRPO)

[Figure 9]

0.15

Qwen2.5-VL-7B (MUPO)

AverageDiversity

0.10

[Figure 10]

[Figure 11]

[Figure 12]

0.05

0.00

0 20 40 60 80 100

Reasoning Trajectories

Training

Training Steps

(A) Diversity vs. Training Progress (B) Convergent and Divergent Policy Distribution vs. Training Progress

- Figure 3. The diversity collapse of GRPO. In (A), we plot the evolution of reasoning diversity across training steps. In (B), we present an illustration of policy distribution over training to highlight the contrasting dynamics of convergent and divergent thinking. The gray region denotes rewards associated with different reasoning trajectories, while the blue curve indicates corresponding sampling probabilities.

−100 −50 0 50 100

−90

−60

−30

0

30

60

−50 0 50 100

−100

−50

0

50

100

Base Correct Base Incorrect RL Correct RL Incorrect

Let’s solve the equations…

We can estimate the answer…

We may count by exclusion…

Let’s count them one by one…

(A) Successful Example (B) Failure Example

Base Correct Base Incorrect RL Correct RL Incorrect

- Figure 4. The t-SNE projection of reasoning embeddings. We analyze a successful case where RL models produce correct answers, and a failure case where they cannot despite multiple samplings.

beddings are more densely clustered, while those of base models are more sparsely distributed. This structural difference explains higher pass rate of RL models in successful cases due to concentrated distribution. However, in failure cases, their narrow embedding region fails to cover any correct trajectory. In contrast, base models, benefiting from a wider and more diverse reasoning space, are more likely to retrieve correct solutions from alternative regions.

##### 3.2. Diversity Collapse

The above findings indicate that RL, when applied to base models, diminishes inherent divergent thinking, steering them toward a single, specialized reasoning strategy. To gain deeper insights into this behavioral shift, we examine the evolution of diversity throughout the training process.

phisticated reasoning along a single trajectory. However, as k increases, a notable shift occurs: base models succeed in solving substantially more problems, while the gains of RL models remain marginal. This observation suggests that base models exhibit a greater capacity to generate effective alternative solutions in challenging cases, indicating their higher potential. Our qualitative analysis in Fig. 1 further highlights this contrast, where RL models tend to adopt narrowly focused reasoning, yet base models display divergent reasoning, approaching problems from varied perspectives. Divergent thinking increases the odds of success. To further investigate the above distinctions, we perform multiple runs of base models and plot the relationship between reasoning diversity and corresponding acc@4 scores, as shown in Fig. 2 (C) and (D). We observe a strong positive correlation: as the reasoning diversity increases, acc@4 improves substantially. Intuitively, this indicates that tackling a problem through diverse paths rather than adhering to a single strategy facilitates the discovery of correct answers. This observation aligns with real-world problem-solving, where most tasks admit multiple viable solutions, and broader exploration often leads to more effective outcomes. Such findings also offer an explanation for the superior performance of base models over RL under parallel thinking.

Experimental Setup. We adopt GRPO as the typical RL algorithm and consider two base models: Qwen2.5-VL3B and Qwen2.5-VL-7B [1]. The models are trained on ViRL39K [38], a high-quality and comprehensive dataset, for around half an epoch. For each step, we generate 10 responses per example and record diversity on validation set. GRPO narrows the mind before learning begins. As shown in Fig. 3 (A), we observe that reasoning diversity drops sharply during the early training stage, i.e., first 20 steps, where the model has been exposed to only limited training data. This collapse suggests a premature convergence to a narrow set of strategies, resulting in the exclusion of most potential solution paths and dedicating the majority of training to refining only a small subset. Such phenomenon has two key issues. 1) Exploitation over exploration: the model favors unimodal optimization, neglecting alternative modes and thereby becoming susceptible to local optima; 2) Limited scalability: as discussed in Section 3.1, this convergent reasoning fails to generalize across a broad range of questions, constraining the scaling capabilities.

In Fig. 3 (B), we provide a conceptual illustration of the training dynamics for convergent and divergent reasoning. Given that a problem may have multiple valid strategies, each corresponding to a distinct mode as shown in the figure, convergent training, i.e., GRPO, tends to select one

To provide a deeper understanding, we visualize the tSNE projection of reasoning embeddings for both RL and base models in Fig. 4. It reveals that RL reasoning em-

#### MUPO

Policy Optimization

𝑜𝑜11

𝑜𝑜12

- 𝑜𝑜1𝐾𝐾

- 𝑜𝑜2𝐾𝐾 … 𝑜𝑜|𝐺𝐺𝐾𝐾𝐾𝐾|

- 𝒥𝒥1

- 𝒥𝒥2 … 𝒥𝒥𝐾𝐾

- 𝑜𝑜1

- 𝑜𝑜2 … 𝑜𝑜𝑁𝑁

|[Figure 13]|
|---|

𝑜𝑜21 …

𝑜𝑜22 …

###### Partition

𝑞𝑞 Rollout

Policy Model

…

Question

𝑜𝑜|𝐺𝐺1 1|

𝑜𝑜|𝐺𝐺2 2|

Input Image

Responses Multiple Groups

LoadBalance

Verifier

×𝑤𝑤𝑘𝑘

- 𝑜𝑜1𝑚𝑚
- 𝑜𝑜2𝑚𝑚

𝐴𝐴̂1𝑘𝑘

- 𝑜𝑜1𝑘𝑘
- 𝑜𝑜2𝑘𝑘

- 𝑜𝑜1𝑘𝑘

- 𝑜𝑜2𝑘𝑘 … 𝑜𝑜|𝐺𝐺𝑘𝑘 𝑘𝑘|

- 𝑑𝑑1𝑘𝑘

- 𝑑𝑑2𝑘𝑘 … 𝑑𝑑|𝐺𝐺𝑘𝑘 𝑘𝑘|

- 𝑎𝑎1𝑘𝑘

- 𝑎𝑎2𝑘𝑘 … 𝑎𝑎|𝐺𝐺𝑘𝑘 𝑘𝑘|

- 𝜋𝜋1𝑘𝑘

…

- 𝜋𝜋2𝑘𝑘

·λ +

𝐴𝐴̂𝑘𝑘2 …

𝒥𝒥𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺 𝐺𝐺𝑘𝑘

&

… … 𝑜𝑜|𝐺𝐺𝑘𝑘 𝑘𝑘|

𝛽𝛽

N 𝐾𝐾 𝐺𝐺𝑘𝑘

𝑜𝑜|𝐺𝐺𝑚𝑚𝑚𝑚|

𝜋𝜋|𝐺𝐺𝑘𝑘 𝑘𝑘|

𝐴𝐴̂𝑘𝑘|𝐺𝐺𝑘𝑘|

𝑤𝑤𝑘𝑘 =

𝑡𝑡cur 𝑡𝑡max

Accuracy Advantage Reward

Probability

Average Distance Between Groups

kth Group

Diversity Reward

λ ∝ 1 + cos(𝜋𝜋 ⋅

)

- Figure 5. The overview of MUPO. The upper part illustrates the high-level pipeline, where responses are partitioned into multiple groups, and the overall optimization objective is formulated as a composition of multiple GRPO objectives, each corresponding to a group. In the lower part, we present the advantage computation for a group, in which we introduce diversity reward to encourage inter-group separation.

mode early and progressively sharpens the distribution toward it. In contrast, divergent training, i.e., our expectation, encourages the model to explore and refine multiple modes, enabling the discovery of globally optimal solutions.

### 4. The Proposed Method

##### 4.1. Preliminary

To introduce our approach, we first revisit the standard RL algorithm, notably Group Relative Policy Optimization (GRPO). In the multimodal setting, consider a dataset D, where each example comprises a question q, a gold answer y, and corresponding visual input I. GRPO is designed to promote the generation of high-quality responses by contrasting multiple candidate outputs, rewarding superior ones while penalizing less effective alternatives. Given a policy model πθ, the objective is to maximize the following:

JGRPO(θ) = E(q,y,I)∼D,{o  i}|iG=1| ∼πθold(·|q,I)  1

 

|G|

min ri(θ)Aˆi,clip ri(θ),1 − ϵ,1 + ϵ A ˆi

|G|

i=1

πθ(oi | q,I) πθ

, (1)

with ri(θ) =

(oi | q,I)

old

where G represents the set of sequences {o1,o2,...,o|G|} sampled from the old policy πold, ϵ controls the clipping bounds to maintain on-policy training. It is optional to impose a KL constraint toward the reference policy for stable

optimization. The estimated token-level advantage Aˆi is derived by broadcasting the normalized sequence-level reward Ri across all token positions, which is defined as follows:

Ri − mean(R) std(R)

Aˆi =

, i = 1,··· ,|G|, (2)

where R = {R1,R2,...,R|G|} indicates the reward of the sequence group. A common choice for reward design is the verifiable reward, which provides direct feedback based on metrics such as accuracy and format consistency.

##### 4.2. Multi-Group Policy Optimization

Despite the promise of GRPO, our findings in Section 3.2 reveal that it is prone to diversity collapse, where models prematurely converge to a subset of reasoning paths while abandoning alternative strategies, leading to local optima. To address this, we propose Multi-Group Policy Optimization (MUPO), a simple yet effective drop-in replacement of GRPO to incentivize divergent thinking. Inspired by the intra-group diversity collapse, MUPO partitions responses into multiple groups. Unlike GRPO which computes advantages globally, MUPO performs localized advantage estimation within each group, and introduces a diversity reward to promote inter-group separation. Intuitively, each group serves as a distinct realization of a reasoning strategy, and MUPO aims to maintain multiple modes while refining each of them, thereby achieving both breadth and depth. The overview of our approach has been presented in Fig. 5. Multi-Group Objective. Given N sampled responses at each step, we partition them into K groups {Gk}Kk=1 based

on the embedding space as described in Section 3.1. Specifically, we apply constrained clustering [2] to group responses with similar trajectories, while enforcing minimum group size Gmin for reliable advantage estimation. This enables each group to capture a distinct reasoning mode. The objective of MUPO is to maximize the following:

JMUPO(θ) = E(q,y,I)∼D,{o

i}Ni=1∼πθold(·|q,I) K

|Gk|

wk |Gk|

min ri(θ)Aˆki ,clip ri(θ),1 − ϵ,1 + ϵ A ˆki

i=1

k=1

JGRPO with ri(θ) =

β

πθ(oki | q,I) πθ

N K |Gk|

. (3)

, wk =

(oki | q,I)

old

Essentially, MUPO can be regarded as a composition of multiple GRPO objectives, enabling the search for optima from diverse modes. wk is a load-balance scaler controlled by sensitivity exponent β, which modulates the contribution of each group to the overall objective, preventing larger groups from dominating the optimization process. The advantage Aˆki is estimated locally within each group to ensure the refinement of each mode proceeds independently:

Rik − mean(R) std(R)

Aˆki =

, i = 1,··· ,|Gk|. (4)

Diversity Reward. To encourage models to explore various reasoning strategies, we introduce a diversity reward that increases the separation between groups. Specifically, for a given trajectory oki from Gk, we compute its diversity reward as the average distance between its reasoning embedding and those of responses from all other groups:

|Gm|

K

1 N − |Gk|

d(oki ,omj ), (5)

Rdiv =

m=1 m̸=k

j=1

where omj indicates the trajectory from Gm, and d(·,·) denotes the cosine distance between the reasoning embeddings of two responses. This ensures that responses within a group that are more distant from other groups receive a higher advantage. The final reward for oki is computed as:

Rik = Racc + Rfmt + λ · 1[Racc = 1] · Rdiv. (6)

Racc and Rfmt denote accuracy and format reward, respectively. We impose an accuracy condition on the diversity reward to prevent reward hacking, where models pursue diverse outputs at the expense of correctness. λ is the weight of the diversity reward and is annealed over the current training step tcur according to the cosine schedule:

λmax − λmin 2

tcur tmax

, (7)

1 + cos π ·

λ = λmin +

where tmax is the total number of training steps, λmax and λmin specify the desired initial and final values of λ with

a smooth and monotonic decay. This design encourages broad exploration of diverse reasoning strategies in the early training, while gradually shifting the focus toward identifying globally optimal solutions in later stages.

### 5. Experiment and Results

Implementation Details. Similar to settings in Section 3.2, we train all models on ViRL39K [38] for 2 epochs with a learning rate of 1e−6. A random subset of the dataset is used in ablation study for efficiency. We select Qwen2.5-VL [1] at 3B and 7B parameter scales as base models. For group reward computation, we generate N = 15 responses per example with a sampling temperature of 1.0. For MUPO, unless otherwise specified, we partition responses into K = 3 groups with a minimum size of Gmin = 3. We set the loadbalance exponent β = 1, and the initial and final values of diversity reward weight λmax = 0.4 and λmin = 0.1.

Benchmarks. We denote models trained with our method as MUPO-Thinker and evaluate on nine reasoning benchmarks spanning various task types: mathematical benchmarks including MathVerse [54], MathVista [20], MathVision [39], LogicVista [44], WeMath [25], and Geometry3K [19], as well as general-purpose ones encompassing MMStar [5], HallusionBench [9] and MMVet [51].

Baseline Methods. To verify the effectiveness of our approach, we compare MUPO-Thinker against existing strong baselines, including InternVL2.5 [6], R1-OneVision [47], VLAA-Thinker [4], Vision-R1 [13], VLM-R1 [29] and LMM-R1 [24], spanning both 3B and 7B models. We also report scores of proprietary models such as GPT-5Thinking [30] and Gemini-2.5-Pro [8] for reference. The baseline results are primarily cited from the corresponding papers, secondarily from the OpenCompass leaderboard, and reproduced when neither source is available.

Metrics. To evaluate both the depth and breadth of reasoning, we report acc@1 and acc@4 as defined in Section 3.1. For the former, we employ greedy decoding, which yields the most confident responses. For the latter, we generate 4 candidates per example with a sampling temperature of 1.0, thereby assessing the model’s test-time scaling capabilities. Please refer to Appendix A for more configuration details.

##### 5.1. Main Results

MUPO establishes a new state of the art. As shown in Table 1 and 2, MUPO-Thinker-7B achieves an average acc@1 improvement of 2.5% (49.1% → 51.6%) on mathematical benchmarks and 2.3% (63.3% → 65.6%) on generalpurpose benchmarks over previous best results. Similarly, in Table 3, MUPO-Thinker-3B surpasses existing strong baselines of the same scale with notable gains of 2.0% (41.5% → 43.5%) and 2.4% (55.4% → 57.8%) on two types of benchmarks, respectively. This suggests that incorporating divergent thinking enables models to discover

MathVerse [54] MathVista [20] MathVision [39] LogicVista [44] WeMath [25] Geometry3k [19] Average Acc@1 Acc@4 Acc@1 Acc@4 Acc@1 Acc@4 Acc@1 Acc@4 Acc@1 Acc@4 Acc@1 Acc@4 Acc@1 Acc@4 Closed-Source Models

Models

GPT-5-Thinking* [42] 81.2 85.5 81.9 86.1 72.0 79.2 70.0 81.5 71.1 78.4 79.9 84.3 76.1 82.5 Gemini-2.5-Pro* [8] 76.9 79.3 80.9 85.2 69.1 72.5 73.8 76.4 78.0 82.7 77.2 80.1 76.0 79.4

Open-Source Models

Qwen2.5-VL-7B [1] 40.7 55.2 62.3 78.5 23.2 41.6 42.6 59.3 33.1 50.1 38.5 54.4 40.1 56.5 InternVL2.5-8B* [6] 34.5 42.3 68.2 72.8 25.6 29.4 38.3 44.7 38.6 43.5 44.8 48.0 41.7 46.8 R1-OneVision-7B [47] 46.4 49.9 64.1 69.4 29.9 34.1 45.6 52.5 44.6 47.9 46.1 50.4 46.1 50.7 VLAA-Thinker-7B [4] 48.2 51.6 68.0 70.8 26.4 30.3 48.5 54.5 41.5 46.8 50.6 55.2 47.2 51.5 Vision-R1-7B [13] 52.4 56.1 73.5 75.6 28.2 32.9 49.7 53.8 41.6 44.3 49.0 54.1 49.1 52.8

Our Models

MUPO-Thinker-3B 41.3 50.8 64.3 72.5 27.8 35.4 42.8 50.3 36.5 46.8 45.1 52.9 43.0 51.5 MUPO-Thinker-7B 53.9 61.7 77.9 82.4 31.3 39.7 50.6 61.5 44.1 48.6 52.1 59.1 51.6 58.8

- Table 1. The evaluation of our method on mathematical benchmarks. We report scores of both open-source and proprietary VLMs. The best and second best results are bolded and underlined, respectively. * indicates the results sourced from the OpenCompass leaderboard.

Models

MMStar [5] HallBench [9] MMVet [51] Average Acc@1 Acc@4 Acc@1 Acc@4 Acc@1 Acc@4 Acc@1 Acc@4

QwenVL [1] 59.2 74.8 50.0 65.6 64.8 74.1 58.0 71.5 InternVL [6] 63.2 70.6 49.0 58.3 62.8 69.0 58.3 66.0 R1-OV [47] 64.7 67.5 52.5 57.6 65.2 67.3 60.8 64.1 VLAA [4] 66.1 69.1 54.7 56.9 70.0 72.7 63.6 66.2 V-R1 [13] 66.3 68.9 55.4 59.0 68.2 70.5 63.3 66.1 MUPO 68.7 75.4 57.5 63.6 70.6 78.2 65.6 72.4

- Table 2. The results of MUPO on general-purpose benchmarks. All models considered in this evaluation are of the 7B scale.

Mathematics General Average Acc@1 Acc@4 Acc@1 Acc@4 Acc@1 Acc@4

Models Params

QwenVL [1] 3B 33.4 47.0 51.3 62.3 40.0 52.1 InternVL [6] 2B 26.5 34.4 53.1 58.9 35.3 42.6 VLM-R1 [29] 4B 37.8 42.3 53.9 57.7 43.2 47.4 VLAA [4] 3B 39.5 43.3 55.1 59.3 44.7 48.6 LMM-R1 [24] 4B 41.5 45.5 55.4 59.2 46.1 50.1 MUPO 3B 43.5 51.5 57.8 65.2 48.3 56.0

Table 3. The comparison of MUPO with baselines at 3B scale. We report average scores of mathematical and general benchmarks.

100

60

improvement enables MUPO-Thinker-3B to attain performance comparable to several strong 7B baselines. These results demonstrate that our approach successfully integrates the complementary strengths of RL and base models, significantly raising the upper bound of their capabilities.

30

50

0

0

−30

−50

−60

##### 5.2. Further Discussion

GRPO Correct GRPO Incorrect MUPO Correct MUPO Incorrect

GRPO Correct GRPO Incorrect MUPO Correct MUPO Incorrect

−90

−100

MUPO approaches problems with diverse strategies. To verify whether MUPO indeed induces divergent thinking, we visualize the distribution of reasoning embeddings produced by our model, as shown in Fig. 6. In contrast to GRPO, which tends to sample from a narrow region, MUPO exhibits a broad, multimodal structure, with each mode corresponding to a distinct solution strategy. This proves particularly advantageous in the failure case on the right, where GRPO fails to handle the problem, while MUPO successfully learns the correct reasoning from alternative modes, providing a clear explanation for its superior performance.

−50 0 50 100

−100 −50 0 50 100

- Figure 6. The t-SNE projection of our reasoning embeddings. The selected examples here correspond to the ones presented in Fig. 4.

globally superior solutions and demonstrate more sophisticated reasoning along individual path, highlighting the importance of reasoning diversity for effective RL training.

MUPO exhibits stronger test-time scaling capabilities. As shown in Table 1 and 2, MUPO-Thinker-7B significantly outperforms existing strong RL models in acc@4 by 6.0% (52.8% → 58.8%) on mathematical benchmarks and 6.2% (66.2% → 72.4%) on general-purpose benchmarks, while surpassing the base model of the same scale. In Table 3, MUPO-Thinker-3B also shows a substantial performance gap over recent baseline models, achieving an average gain of 5.9% (50.1% → 56.0%). Moreover, this scalability

MUPO is capable of discovering better solutions. To further understand the benefits of divergent training in MUPO, we conduct a qualitative analysis in Fig. 7. In this spatial reasoning example, where the task is to estimate the height of a building, GRPO adopts a rigid layer-by-layer

###### Question GRPO MUPO

[Figure 14]

We may use other objects as reference to estimate:

We can evaluate the height from bottom to top:

- 1. By observing the surrounding objects, we can see a utility telegraph pole on the right side.
- 2.According to safety regulations, the minimum height for such pole is no less than 8 to 10 meters.
- 3.It is evident from the image that the building is at least twice as tall as the telegraph pole.
- 4.Taking into account perspective distortion, the building height should exceed 2 x 9 =18 meters.

- 1. The ground floor resembles the lobby of a hotel, which typically has a height of 3 to 3.5 meters.
- 2.The upper floors appears to be residential units, which are notably lower, likely 2 meters each.
- 3.From the camera’s perspective, the building has five floors, excluding the basement.
- 4.Therefore, the total height of the building is estimated to be 3.5 + 2 x 4 = 11.5 meters high.

Estimate the building height. A. 10m B. 15m C. 20m D. 25m

In conclusion, the height is most likely 20 meters.

Considering floor gaps, the answer is 15 meters.

- Figure 7. The qualitative analysis. We consider Vision-R1 [13] and MUPO-Thinker as typical GRPO and MUPO models and evaluate on an example from MMStar [5] with greedy decoding. We reorganize responses with numbering and omit special tags for better visualization.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

0 70 140 210 280 350

0.3

0.4

0.5

AccuracyReward

DiversityReward

Training Steps Training Steps

0 70 140 210 280 350

0.15

0.18

0.21

0.24

- Figure 8. The learning curves of accuracy and diversity reward. We use exponential moving average (EMA) for smoothing.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

- 52
- 53
- 54

- 53
- 54

AverageAccuracy(%)

0.1 0.2 0.3 0.4 0.5 0.6

0.0 0.1 0.2 0.3 0.4

Initial Weight λmax Final Weight λmin

Figure 9. The ablation study of initial and final values of diversity reward weight λmax and λmin on average benchmark accuracy.

The number of groups K. In Table 4, we investigate the impact of number of groups K on accuracy. Notably, when K = 1, the training process degrades to GRPO. As K increases, the accuracy rises rapidly and peaks at K = 3. Moreover, for mathematical benchmarks, larger K yields better performance, whereas smaller K is more suitable for general problems. This reflects an intrinsic task nature: the former desires more flexible and diverse strategies, while the latter favors more uniform and structured reasoning.

K MathVerse MathVista MathVision MMStar HallBench Average

- 1 46.9 69.1 24.1 64.8 54.7 51.9
- 2 49.4 72.3 28.0 65.2 56.7 54.3
- 3 51.2 74.1 29.3 65.8 56.5 55.4
- 4 50.9 74.6 29.4 65.1 56.3 55.3
- 5 50.6 74.8 29.1 64.5 55.9 55.0 Table 4. The benchmark accuracy varying number of groups K.

estimation, which is prone to cumulative errors and ultimately fails. In contrast, our model leverages surrounding reference objects to precisely locate the building height range. This illustrates that MUPO enables models to learn a broader set of reasoning strategies and to generate smarter solutions when faced with various problem types.

The diversity reward weight λmax and λmin. MUPO adopts an annealing schedule for the diversity reward weight, progressively degrading from the initial value λmax = 0.4 to the final value λmin = 0.1 over the course of training. As shown in Fig. 9, we perform an ablation study by fixing one of the parameters as its default while varying the other. The results show the optimal values align with our default settings. Increasing either parameter causes the diversity reward to dominate at the expense of accuracy, while reducing them weakens exploration, amplifying the risk of convergence to local optima. More results such as ablation of β and limitation analysis are provided in Appendix.

MUPO learns from exploration to exploitation. In Fig. 8, we plot the learning curves of accuracy and diversity reward to further analyze the training dynamics of MUPO. As accuracy steadily improves, the diversity reward exhibits a distinct rise-fall-plateau trend. The initial rise indicates increasing distances between response groups, suggesting the model is actively exploring diverse modes. The slight decline in the middle is attributed to the annealing weight of the diversity reward, since the model begins to exploit the most promising strategy it has discovered. The final plateau reflects stabilization in training, implying the model has situated around effective solutions. We also present pairwise diversity of MUPO on validation set in Fig. 3 (A), which shows a much more gradual decline in contrast to sharp collapse observed in GRPO, indicating MUPO achieves a balanced trade-off between exploration and exploitation.

### 6. Conclusion

In this paper, we identify a fundamental distinction between the behavioral patterns of RL and base models, where the former tends to engage in deeper yet narrow reasoning trajectories, while base models, despite less refined along individual path, exihibit broader and more diverse reasoning strategies. Through further analysis of training dynamics, we find that GRPO is prone to diversity collapse, causing

models to converge to a limited set of strategies while neglecting the majority of potential alternatives, leading to local optima and poor scalability. To address this, we propose MUPO, a simple yet effective policy algorithm designed to incentivizes divergent thinking across multiple solutions and demonstrate its effectivenss on established benchmarks.

###### Acknowledgement

This research was, in part, funded by the U.S. Government – DARPA TIAMAT HR00112490421. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the U.S. Government.

### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1, 2, 3, 4, 6, 7
- [2] Paul S Bradley, Kristin P Bennett, and Ayhan Demiriz. Constrained k-means clustering. Microsoft Research, Redmond, 20(0):0, 2000. 6
- [3] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. Internlm2 technical report. arXiv preprint arXiv:2403.17297, 2024. 2
- [4] Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. Sft or rl? an early investigation into training r1-like reasoning large vision-language models. arXiv preprint arXiv:2504.11468,

2025. 3, 6, 7

- [5] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? NeurIPS, 37:27056–27087,

2024. 6, 7, 8

- [6] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024. 1, 6, 7
- [7] Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective. arXiv preprint arXiv:2506.14758, 2025. 3
- [8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 6, 7
- [9] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang,

- Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In CVPR, pages 14375– 14385, 2024. 3, 6, 7
- [10] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025. 1
- [11] Mengqi He, Xinyu Tian, Xin Shen, Jinhong Ni, Shu Zou, Zhaoyuan Yang, and Jing Zhang. Few tokens matter: Entropy guided attacks on vision-language models. arXiv preprint arXiv:2512.21815, 2025. 3
- [12] Zican Hu, Shilin Zhang, Yafu Li, Jianhao Yan, Xuyang Hu, Leyang Cui, Xiaoye Qu, Chunlin Chen, Yu Cheng, and Zhi Wang. Diversity-incentivized exploration for versatile reasoning. arXiv preprint arXiv:2509.26209, 2025. 3
- [13] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749,

2025. 1, 2, 3, 6, 7, 8

- [14] Pu Jian, Junhong Wu, Wei Sun, Chen Wang, Shuo Ren, and Jiajun Zhang. Look again, think slowly: Enhancing visual reflection in vision-language models. arXiv preprint arXiv:2509.12132, 2025. 3
- [15] Yuhua Jiang, Jiawei Huang, Yufeng Yuan, Xin Mao, Yu Yue, Qianchuan Zhao, and Lin Yan. Risk-sensitive rl for alleviating exploration dilemmas in large language models. arXiv preprint arXiv:2509.24261, 2025. 3
- [16] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. NeurIPS, 35:22199–22213, 2022. 1, 2
- [17] Ming Li, Jike Zhong, Shitian Zhao, Yuxiang Lai, Haoquan Zhang, Wang Bill Zhu, and Kaipeng Zhang. Think or not think: A study of explicit thinking in rule-based visual reinforcement fine-tuning. arXiv preprint arXiv:2503.16188,

2025. 1

- [18] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visualrft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025. 1, 3
- [19] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. ACL, 2021. 6, 7
- [20] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. ICLR,

2023. 2, 6, 7

- [21] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025. 3
- [22] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy

- Liang, Emmanuel Cand`es, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025. 1, 2, 3
- [23] Yi Peng, Peiyu Wang, Xiaokun Wang, Yichen Wei, Jiangbo Pei, Weijie Qiu, Ai Jian, Yunzhuo Hao, Jiachun Pan, Tianyidan Xie, et al. Skywork r1v: Pioneering multimodal reasoning with chain-of-thought. arXiv preprint arXiv:2504.05599,

2025. 3

- [24] Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025. 3, 6, 7
- [25] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? ACL, 2024. 3, 6, 7
- [26] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 1, 3
- [27] ByteDance Seed, Jiaze Chen, Tiantian Fan, Xin Liu, Lingjun Liu, Zhiqi Lin, Mingxuan Wang, Chengyi Wang, Xiangpeng Wei, Wenyuan Xu, et al. Seed1. 5-thinking: Advancing superb reasoning models with reinforcement learning. arXiv preprint arXiv:2504.13914, 2025. 1
- [28] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 1, 3
- [29] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025. 1, 6, 7
- [30] Aaditya Singh, Adam Fry, Adam Perelman, et al. Gpt-5 system card, 2025. 1, 6
- [31] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. ICLR, 2025. 3
- [32] Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Minglan Lin, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang. Reason-rft: Reinforcement fine-tuning for visual reasoning. arXiv preprint arXiv:2503.20752, 2025. 3
- [33] Xinyu Tian, Shu Zou, Zhaoyuan Yang, and Jing Zhang. Argue: Attribute-guided prompt tuning for vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28578– 28587, 2024. 2
- [34] Xinyu Tian, Shu Zou, Zhaoyuan Yang, Mengqi He, Fabian Waschkowski, Lukas Wesemann, Peter Tu, and Jing Zhang. More thought, less accuracy? on the dual nature of reasoning in vision-language models. arXiv preprint arXiv:2509.25848, 2025. 3
- [35] Xinyu Tian, Shu Zou, Zhaoyuan Yang, Mengqi He, and Jing Zhang. Black sheep in the herd: Playing with spuriously

- correlated attributes for vision-language recognition. arXiv preprint arXiv:2502.15809, 2025. 2
- [36] Xinyu Tian, Shu Zou, Zhaoyuan Yang, and Jing Zhang. Identifying and mitigating position bias of multi-image visionlanguage models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10599–10609, 2025. 3
- [37] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1
- [38] Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025. 3, 4, 6
- [39] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. NeurIPS, 37:95095–95169, 2024. 6, 7
- [40] Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939, 2025. 3
- [41] Weiyun Wang, Zhangwei Gao, Lianjie Chen, Zhe Chen, Jinguo Zhu, Xiangyu Zhao, Yangzhou Liu, Yue Cao, Shenglong Ye, Xizhou Zhu, et al. Visualprm: An effective process reward model for multimodal reasoning. arXiv preprint arXiv:2503.10291, 2025. 3
- [42] Zhenhailong Wang, Xuehang Guo, Sofia Stoica, Haiyang Xu, Hongru Wang, Hyeonjeong Ha, Xiusi Chen, Yangyi Chen, Ming Yan, Fei Huang, et al. Perception-aware policy optimization for multimodal reasoning. arXiv preprint arXiv:2507.06448, 2025. 3, 7
- [43] Jiaer Xia, Yuhang Zang, Peng Gao, Yixuan Li, and Kaiyang Zhou. Visionary-r1: Mitigating shortcuts in visual reasoning with reinforcement learning. arXiv preprint arXiv:2505.14677, 2025. 1
- [44] Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024. 3, 6, 7
- [45] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 1
- [46] Shuo Yang, Yuwei Niu, Yuyang Liu, Yang Ye, Bin Lin, and Li Yuan. Look-back: Implicit visual re-focusing in mllm reasoning. arXiv preprint arXiv:2507.03019, 2025. 3
- [47] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025. 1, 3, 6, 7
- [48] Yue Yao, Xinyu Tian, Zheng Tang, Sujit Biswas, Huan Lei, Tom Gedeon, and Liang Zheng. Training with prod-

- uct digital twins for autoretail checkout. arXiv preprint arXiv:2308.09708, 2023. 3
- [49] Yue Yao, Zelin Wen, Yan Tong, Xinyu Tian, Xuqing Li, Xiao Ma, Dongliang Xu, and Tom Gedeon. Simple radiology vllm test-time scaling with thought graph traversal. arXiv preprint arXiv:2506.11989, 2025. 3
- [50] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. 3
- [51] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. ICML, 2023. 6, 7
- [52] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025. 1
- [53] Kaichen Zhang, Yuzhong Hong, Junwei Bao, Hongfei Jiang, Yang Song, Dingqian Hong, and Hui Xiong. Gvpo: Group variance policy optimization for large language model posttraining. arXiv preprint arXiv:2504.19599, 2025. 3
- [54] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In ECCV, pages 169–186. Springer, 2024. 2, 3, 6, 7
- [55] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025. 3
- [56] Tong Zheng, Hongming Zhang, Wenhao Yu, Xiaoyang Wang, Xinyu Yang, Runpeng Dai, Rui Liu, Huiwen Bao, Chengsong Huang, Heng Huang, et al. Parallel-r1: Towards parallel thinking via reinforcement learning. arXiv preprint arXiv:2509.07980, 2025. 3
- [57] Denny Zhou, Nathanael Sch¨arli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, et al. Least-to-most prompting enables complex reasoning in large language models. ICLR, 2022. 1, 2
- [58] Shu Zou, Xinyu Tian, Qinyu Zhao, Zhaoyuan Yang, and Jing Zhang. Simlabel: Consistency-guided ood detection with pretrained vision-language models. In Australasian Joint Conference on Artificial Intelligence, pages 110–121. Springer, 2025. 3
- [59] Shu Zou, Xinyu Tian, Lukas Wesemann, Fabian Waschkowski, Zhaoyuan Yang, and Jing Zhang. Unlocking vision-language models for video anomaly detection via fine-grained prompting. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 4223–4233, 2026. 3

