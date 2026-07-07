# arXiv:2507.22607v2[cs.CV]31Jul2025

[Figure 1]

## VL-Cogito: Progressive Curriculum Reinforcement Learning for Advanced Multimodal Reasoning

Ruifeng Yuan1,2,3,♣, Chenghao Xiao1,♣, Sicong Leng1, Jianyu Wang1, Long Li1, Weiwen Xu1, Hou Pong Chan1, Deli Zhao1,2, Tingyang Xu1,2, Zhongyu Wei3, Hao Zhang1,♠, Yu Rong1,2

1DAMO Academy, Alibaba Group 2Hupan Lab 3Fudan University

♣Equal Contribution, ♠Corresponding Author

Reinforcement learning has proven its effectiveness in enhancing the reasoning capabilities of large language models. Recent research efforts have progressively extended this paradigm to multimodal reasoning tasks. Due to the inherent complexity and diversity of multimodal tasks, especially in semantic content and problem formulations, existing models often exhibit unstable performance across various domains and difficulty levels. To address these limitations, we propose VL-Cogito, an advanced multimodal reasoning model trained via a novel multi-stage Progressive Curriculum Reinforcement Learning (PCuRL) framework. PCuRL systematically guides the model through tasks of gradually increasing difficulty, substantially improving its reasoning abilities across diverse multimodal contexts. The framework introduces two key innovations: (1) an online difficulty soft weighting mechanism, dynamically adjusting training difficulty across successive RL training stages; and (2) a dynamic length reward mechanism, which encourages the model to adaptively regulate its reasoning path length according to task complexity, thus balancing reasoning efficiency with correctness. Experimental evaluations demonstrate that VL-Cogito consistently matches or surpasses existing reasoning-oriented models across mainstream multimodal benchmarks spanning mathematics, science, logic, and general understanding, validating the effectiveness of our approach.

Date: August 1, 2025 Correspondence: Hao Zhang, hz.hhea2e@alibaba-inc.com Homepage: https://github.com/alibaba-damo-academy/VL-Cogito

1 Introduction

Recent advancements of Reinforcement Learning (RL) in Large Language Models (LLMs), such as OpenAI o-series (OpenAI, 2024b), Kimi-K1.5 (Kimi, 2025), and DeepSeek-R1 (DeepSeek-AI, 2025), highlight its promise to incentivize the long-chain reasoning abilities of LLMs, enabling them to effectively tackle complex tasks involving code generation, mathematical problems, and scientific reasoning. In particular, RL with verifiable rewards, such as GRPO (Shao et al., 2024), has emerged as a pivotal paradigm for training a “slow-thinking” system, which directly employs rule-based rewards and promotes the generation and iterative refinement of multiple reasoning paths, using group-based relative advantage estimations. This approach has notably improved the performance of LLMs to solve challenging reasoning tasks, pushing the boundaries of what LLMs can achieve in various domains (Yeo et al., 2025; Chen et al., 2025b).

Following the success of this training paradigm in LLMs, researchers have increasingly explored its application to Multimodal Large Language Models (MLLMs). Initial efforts primarily focus on adapting these techniques to specific multimodal domains such as mathematics and logic (Meng et al., 2025; Huang et al., 2025; Chen et al., 2025a). As MLLMs are not constrained to textual modalities, it has the opportunity to enable reasoning across diverse domains. The spectrum ranges from straightforward chart interpretation (Huang et al., 2024), complex geometry problems (Lu et al., 2024), to intricate scientific analysis (Lu et al., 2022a). With this expanded scope of domains, the heterogeneity among different problem types becomes increasingly apparent, presenting a challenge in effectively learning reasoning skills for tasks of varying complexities and types (Wang et al., 2025e; Bi et al., 2025; Li et al., 2025).

In this work, we introduce VL-Cogito, a reasoning-oriented MLLM trained on an extensive dataset comprising

diverse multimodal task domains. To address the varying levels of data difficulty, we propose a Progressive Curriculum Reinforcement Learning framework (PCuRL). Central to our approach is a novel curriculum learning strategy, which systematically guides the model through progressively complex tasks to build robust reasoning capabilities. Specifically, we introduce an online difficulty soft weighting mechanism to dynamically adjust data selection according to predefined difficulty distributions across successive training stages, allowing the model to incrementally transition from mastering simpler questions to effectively handling intricate problems. Moreover, another notable innovation in PCuRL is the dynamic length reward mechanism. Conventional reasoning models often extend reasoning length indiscriminately, potentially compromising efficiency. In contrast, our dynamic reward explicitly incentivizes the model to adapt its reasoning length based on the demands of individual problems. This adaptive strategy ensures the model not only engages in thorough reasoning for complex tasks but also maintains efficiency and succinctness when simpler problems arise, thus optimizing performance across diverse scenarios.

To thoroughly assess the efficacy of PCuRL and the performance of VL-Cogito, we perform extensive experiments on various multimodal reasoning benchmarks spanning mathematical, scientific, and general domains. It is worth noting that VL-Cogito bypasses a cold-start SFT phase and is instead trained directly from the backbone model via PCuRL using the GRPO approach. The results demonstrate that VL-Cogito achieves state-of-the-art or highly competitive performance across all evaluation sets, underscoring the superiority and effectiveness of our proposed method. Furthermore, comprehensive ablation studies are conducted to analyze the contribution of each module, confirming their respective importance. Visualizations of the training process and detailed case studies validate that the progressive curriculum strategy ensures training stability while enhancing both effectiveness and efficiency.

Our contributions can be summarized as follows:

- • We propose PCuRL, a novel multi-stage progressive curriculum RL framework, incorporating an online difficulty soft weighting mechanism that progressively exposes models to increasingly challenging tasks, enhancing multimodal reasoning capabilities. In addition, we introduce a dynamic length reward mechanism designed to encourage the model to modulate reasoning length based on question-specific complexity, effectively balancing depth and efficiency.
- • VL-Cogito achieves the state-of-the-art or highly competitive performance in multimodal benchmarks across various domains, underscoring the effectiveness and versatility of our proposed framework and training pipeline.
- • Extensive ablation studies confirm that our progressive curriculum learning mechanism effectively and consistently increases reasoning depth, leading to improved performance on complex tasks. The dynamic length reward strategy allows the model to produce concise answers for simple problems while promoting longer, more in-depth reasoning for more difficult ones. Overall, PCuRL delivers substantial and balanced gains in accuracy, training stability, and efficiency across a range of task difficulties.

- 2 Related Work

As RL-based reasoning models have demonstrated success in LLMs, significant research attention has shifted toward adapting these advances to multimodal large language models (MLLMs; Yao et al. (2024); Xu et al. (2025); Wang et al. (2025c); Peng et al. (2025a); Xia et al. (2025)). Some work aims to improve MLLMs’ reasoning abilities directly via RLVR-style optimization (Kimi et al., 2025; Wang et al., 2025b; Guo et al., 2025; Team et al., 2025b). For instance, Vision-R1 (Huang et al., 2025) distills long-chain reasoning data from existing models using modality bridging and introduces progressive thinking suppression to address overthinking. R1-OneVision (Yang et al., 2025) presents a model that converts visual information into formal textual representations for accurate cross-modal reasoning, while OpenVLThinker (Deng et al., 2025) combines supervised fine-tuning with reinforcement learning in an iterative training framework.

To enhance the stability and effectiveness of RLVR training, some research investigates advanced strategies for multimodal reasoning models, such as data selection (Wang et al., 2025a; Meng et al., 2025), reward design (Shen et al., 2025; Tan et al., 2025), and advanced RL recipes (Peng et al., 2025b; Zhang et al., 2025; Liu et al., 2025; Yao et al., 2025; Chen et al., 2025c). MM-Eureka (Meng et al., 2025) demonstrates that

[Figure 2]

[Figure 3]

Base Model

RL Data

[Figure 4]

[Figure 5]

Easy Stage Medium Stage Hard Stage

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Accuracy Reward Accuracy Reward

Dynamic Length Reward

[Figure 12]

[Figure 13]

[Figure 14]

Format Reward Format Reward

VL-COGITO

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Easy

Hard

Training Progress

Online Difficulty Soft Weighting (ODSW)

- Figure 1 An overview of the proposed Progressive Curriculum Reinforcement Learning (PCuRL) framework. It consists of two key components: (1) a multi-stage curriculum RL structure that utilizes online difficulty soft weighting, which partitions the training progress into different stages based on task difficulty; (2) a dynamic length reward mechanism that encourages the model to adapt its reasoning chain length according to task complexity, rather than indiscriminately increasing it. In the Easy stage, the model tends to assign higher weights to relatively easier questions for policy optimization, a pattern that similarly applies to the Medium and Hard stages.

difficulty-based data selection is critical for successful RL training. ThinkLite-VL (Wang et al., 2025d) proposes a data-efficient visual reasoning framework employing Monte Carlo Tree Search for sample difficulty estimation, enabling RL with significantly reduced training data. MMR1 (Leng et al., 2025) highlights the gradient vanishing issue in GRPO (Razin et al., 2025) and introduces a variance-aware sampling strategy to mitigate it. R1-VL (Zhang et al., 2025) presents StepGRPO, a multi-stage online RL framework that incorporates step-wise reasoning rewards to improve the reasoning capabilities of MLLMs. VL-Rethinker (Wang et al., 2025a) introduces selective sample replay to address vanishing advantages and a forced rethinking mechanism to promote slow and deliberate reasoning. Praxis-VLM (Hu et al., 2025) proposes an adaptive R1 reward that targets different skills (formatting and reasoning) of MLLMs at different RL training stages. In contrast, our work proposes a novel RL recipe inspired by curriculum learning. We integrate online difficulty soft weighting to train the model on tasks of increasing difficulty, and introduce a novel dynamic length reward design that adaptively determines the ideal reasoning length for each prompt.

- 3 Preliminaries

Group Relative Policy Optimization (GRPO; Shao et al. (2024)) is a reinforcement learning algorithm designed to improve the efficiency and effectiveness of training large language models. It estimates the advantages of language model generations by comparing responses within a group specific to the same input. Given an input x, the behavior policy πθ

first samples a group of G candidate responses {yi}Gi=1. At time step t, the advantage for the i-th response is calculated based on the following equation:

old

r(x,yi) − mean({r(x,y1),...,r(x,yG)}) std({r(x,y1),...,r(x,yG)})

. (1)

Ai,t =

The GRPO also adopts a clipped surrogate objective like PPO (Schulman et al., 2017):

|yi|

G

πθ(yi,t | x,yi,<t) πθ

πθ(yi,t | x,yi,<t) πθ

1 G

1 |yi|

Ai,t,clip

,1 − ϵ,1 + ϵ Ai,t , (2)

min

(yi,t | x,yi,<t)

(yi,t | x,yi,<t)

old

old

t=1

i=1

where ϵ serves as a hyperparameter that controls the tolerance for policy deviation. The clip function is used to prevent excessively large updates by maintaining the ratio between the current policy and the reference policy within a specified range.

Reinforcement Learning with Verifiable Rewards (RLVR) has become a major paradigm for training reasoning models in domains like mathematical reasoning, exemplified by OpenAI’s o-series (OpenAI, 2024b, 2025) and

DeepSeek-R1 (DeepSeek-AI, 2025). The key idea is to provide the model with clear and objective feedback based on predefined correctness criteria. RLVR utilizes rule-based functions to evaluate the accuracy of a model’s output. A format reward is often combined with the accuracy reward to distinguish between the reasoning process and the final answer:

r(x,yi) = racc(x,yi) + rformat(yi), (3)

where the racc refers to a binary accuracy reward of 1 for correct and 0 for incorrect outputs, and rformat is a strict binary format reward used to constrain the structure of the model’s output.

### 4 Methodology

- 4.1 Data Curation

To promote robust performance and strong generalization across a variety of domains, we curated an extensive collection of open-source multimodal datasets, covering 23 datasets distributed across six distinct task categories. (1) Mathematical Reasoning: Geometry3K training set (hiyouga, 2025), GeoQA+ (Cao and Xiao,

- 2022), Geos (Seo et al., 2015), GeomVerse (Kazemi et al., 2024), Inter-GPS (Lu et al., 2021), MultiMath (Peng et al., 2024); (2) Logical Reasoning: Raven (nimapourjafar, 2024), MM-IQ (Cai et al., 2025), EasyArc (Unsal and Akkus, 2025); (3)Counting: CLEVR-Math (Lindström and Abraham, 2022), Super-CLEVR (Li et al.,
- 2023); (4) Science Reasoning: AI2D training set (Kembhavi et al., 2016), ScienceQA training set (Lu et al., 2022a), TQA (Kembhavi et al., 2017)); (5) Chart Understanding: ChartQA training set (Masry et al., 2022), TabMWP (Lu et al., 2023), DVQA (Kafle et al., 2018), FigureQA (Kahou et al., 2018), ArXivQA (Li et al.,
- 2024b), InfographicVQA (Mathew et al., 2022); and (6) General Image Understanding: OKVQA (Marino et al.,

- 2019), VQA2.0 (Antol et al., 2015), LLaVA-CoT (Xu et al., 2025).

For the data used in reinforcement learning, we selectively sample a subset of the data based on our designed quality and category criteria to construct the training set. During data curation, our primary objective is to enhance the overall difficulty and coverage of the training samples, encouraging the model to perform more in-depth reasoning. To this end, two additional measures are implemented: (1) Open-ended Format: To prevent the reasoning model from relying on superficial cues present in specific answer formats, such as multiple-choice, we reformulate most samples into an open-ended QA format; (2) Difficulty Sampling: To exclude questions that do not necessitate genuine reasoning, we employ difficulty-based sampling approach by removing samples that achieve above 50% accuracy over 8 trails using Qwen2.5-VL-7B-Instruct (Bai et al.,

- 2025). The resultant distribution of RL training data following these filtering procedures is presented in Table 4 of Appendix A.

- 4.2 Progressive Curriculum Reinforcement Learning Framework

Consistent with prior work (Meng et al., 2025; Wang et al., 2025a,d), we leverage the Group Relative Policy Optimization (GRPO)-based RLVR scheme. On this foundation, we propose a Progressive Curriculum Reinforcement Learning (PCuRL) framework, which incrementally trains the model on tasks of increasing difficulty. The goal is to enable the MLLM to gradually acquire more sophisticated reasoning skills and extend its reasoning chain, while still maintaining concise and effective responses on simpler tasks. The PCuRL framework comprises two key components: (1) a multi-stage curriculum RL structure that utilizes online difficulty soft weighting, and (2) a dynamic length reward mechanism that encourages the model to adapt its reasoning chain length according to task complexity, rather than indiscriminately increasing it. The overall architecture of our framework is depicted in Figure 1.

- 4.2.1 Online Difficulty Soft weighting (ODSW)

#### The difficulty soft weighting mechanism is designed to prioritize target prompts with specific difficulty levels during the RL training process. Prior work (Bae et al., 2025; Cui et al., 2025) proposes an online hard-weighted difficulty filtering method that discards prompts outside the appropriate difficulty range from each training batch. In contrast, our mechanism allows for more prompts to be considered during training, assigning each a weight that represents its relative importance. Drawing inspiration from ADORA (Gui and Ren, 2025),

1.0

0.8

0.6

Weight

0.4

0.2

0.0

0.0 0.2 0.4 0.6 0.8 1.0

Accuracy

1.0

0.8

0.6

Weight

0.4

0.2

0.0

0.0 0.2 0.4 0.6 0.8 1.0

Accuracy

1.0

0.8

0.6

Weight

0.4

0.2

0.0

0.0 0.2 0.4 0.6 0.8 1.0

Accuracy

(a) ODSW Easy

(b) ODSW Medium

(c) ODSW Hard

- Figure 2 Three difficulty distributions, i.e., easy, medium, and hard, for the Online Difficulty Soft weighting (ODSW).

we realize Online Difficulty Soft Weighting (ODSW) by adjusting the advantage values at the prompt level according to the weights that reflect the difficulty of each question prompt. Consequently, prompts with higher weights contribute more substantially to the gradient updates during training. Specifically, these weights are calculated based on the rollout accuracy of each prompt using a predefined function F as:

G

1 G

Aˆi,t = F

acc(x,yi) · Ai,t, (4)

i=1

where G denotes the number of rollout responses per prompt, acc indicates whether a rollout is correct or not (i.e., 1 for correct and 0 for incorrect), and the definition of F is based on the theory of learnability (Foster and Foerster, 2025; Rutherford et al., 2024; Tzannetos et al., 2023).

According to the theory of learnability, prompts that achieve a rollout accuracy close to 0.5 are considered optimal for RL training, as they present an appropriate level of challenge for effective model learning. Motivated by this theoretical insight, we construct F as a continuous piecewise function, which integrates both sine and constant components with 0.5 as the threshold. The constant components allow the model to focus on the target difficulty, while the sine component ensures a smooth transition between different difficulty levels. This design emphasizes the prompts with the highest learnability, and guides the gradient update direction to explore a specific difficult level. Additionally, it prevents the model from entirely disregarding prompts of other difficulties. F is allowed to flexibly accommodate varying difficulty distributions, facilitating the targeted weighting of prompts that maximize training efficacy. In particular, we design three distinct versions of the function F (as illustrated in Figure 2), each corresponding to prompts with predominantly easy, medium, or hard difficulty levels. To provide a baseline for comparison, we also implement a binary weighting strategy alongside our proposed difficulty soft weighting approach. The definitions of different difficulty soft weighting variants and binary weighting strategies are computed as follows:

- • ODSW Easy (see Figure 2a):

F(Acc) =

sin(π · Acc), if Acc ∈ [0,0.5) 1, if Acc ∈ [0.5,1]

;

- • ODSW Medium (see Figure 2b): F(Acc) = Sin(π · Acc);
- • ODSW Hard (see Figure 2c):

F(Acc) =

1, if Acc ∈ [0,0.5] sin(π · Acc), if Acc ∈ (0.5,1]

;

- • Binary Weighting:

1, if Acc ∈ [Tmin,Tmax] 0, otherwise

F(Acc) =

.

For simplicity, we denote the average rollout accuracy as Acc = G1 Gi=1 acc(x,yi). The Tmin and Tmax in the binary weighting strategy are predefined ranges of average accuracy.

By integrating explicit difficulty control, i.e., ODSW, with curriculum learning, we facilitate progressive model optimization. Initially, focusing on simpler problems enables the model to efficiently acquire correct reasoning patterns and receive clearer optimization signals, thereby stabilizing the training process. Subsequently, focusing on more challenging problems promotes deeper reasoning and exploration of diverse reasoning paths, further improving performance.

- 4.2.2 Dynamic Length Reward (DyLR)

Extending the reasoning process typically provides reasoning models with greater space for thought and allows them to explore more diverse reasoning patterns to tackle challenging problems. A common practice is to implement a length reward mechanism, e.g., Cosine Reward (Yeo et al., 2025), which incentivizes the model to generate longer reasoning paths. However, such a length reward has notable limitations, as it sets a uniform target length and reward strategy across all prompts, resulting in homogeneous reasoning-length adjustments despite variations among tasks. In multimodal reasoning contexts, the preferred reasoning length often differs significantly across tasks. For instance, chart understanding, which primarily relies on perceptual skills, generally necessitates shorter reasoning lengths compared to more complex mathematical tasks, such as geometry problems. Applying a uniform length reward across all tasks without differentiation can thus lead to excessively verbose reasoning in certain cases, while inadequately addressing length requirements in others. Consequently, this indiscriminate approach may impair both reasoning efficiency and model performance.

To address this issue, we propose a Dynamic Length Reward (DyLR) mechanism to determine the ideal reasoning length for each prompt. The core idea is to estimate an appropriate reasoning length for each prompt by utilizing its associated rollout samples at the prompt level. To be specific, the target reasoning budget for each prompt is defined as the average length of all correct responses within its rollout samples. This strategy encourages the model to generate reasoning paths that closely align with the prompt-specific target length. As training progresses and model accuracy improves, the target lengths adapt dynamically and eventually stabilize. For prompts without any correct responses, the dynamic length reward incentivizes the model to extend reasoning up to a predefined maximum length. The formal definition of the dynamic length reward, rlen, is written as:

rlen(yi,y) =

CosFn(Li,Lavg,rlenmin,rlenmax) if Acc > 0 CosFn(Li,Lmax,rlenmin,rlenmax) if Acc = 0

, (5)

where Li refers the reasoning length of i-th response yi of a prompt x, Lavg denotes the average length of all correct responses of the prompt x, Lmax represents the preset maximum target length, rlenmin and rlenmax are the preset minimum and maximum length rewards, respectively. The function CosFn is defined as:

CosFn(Li,Ltgt,rlenmin,rlenmax) = rlenmin +

- 1

- 2

(rlenmax − rlenmin) 1 − cos

Li · π Ltgt

, (6)

where Ltgt is the target reasoning length, i.e., Lavg or Lmax for Acc > 0 and Acc = 0 respectively. Through combining the correctness, format, and dynamic length rewards, the overall reward of PCuRL is defined as:

r(x,yi) = α · racc(x,yi) + β · rformat(yi) + γ · rlen(yi,y), (7) where α, β, and γ are hyperparameters to control the contributions of each specific reward term.

To mitigate excessive growth in reasoning length that may degrade reasoning quality, we introduce an additional hyperparameter w to the advantages of prompts with zero accuracy, i.e., Acc = 0. Accordingly, when integrating dynamic length reward with difficulty soft weighting, Equation 4 is reformulated as:

 

F G 1 Gi=1 acc(x,yi) · Ai,t if Acc > 0 w · F G 1 Gi=1 acc(x,yi) · Ai,t if Acc = 0

Aˆi,t =

. (8)



#### In general, the incorporation of DyLR enables the model to dynamically adjust its output length rather than conforming to a fixed target, thereby mitigating potential over- or under-thinking issues. For simple tasks, the model generates concise responses through rapid reasoning, while for more complex tasks, it automatically adopts longer reasoning trajectories to derive accurate answers.

- 4.2.3 Progressive Curriculum RL

Based on carefully designed online difficulty soft weighting and dynamic length reward mechanisms, we employ a progressive curriculum learning strategy to systematically guide the model’s learning trajectory during RL training, which facilitates a gradual transition from simple tasks to increasingly complex and challenging ones. Similar approaches are widely applied in LLM training to enhance the stability of learning and enable continuous performance improvement (Team et al., 2025a; Liu et al., 2024; Wang et al., 2025f).

Within the PCuRL framework, the training process is explicitly structured into three consecutive stages, i.e., easy, medium, and hard. The dataset remains consistent across all three stages. However, in each stage, the difficulty soft weighting mechanism for each stage is employed to preferentially focus on data corresponding to the targeted difficulty level, and a dedicated F(Acc) function is used to tailor the training focus (as detailed in Section 4.2.1). Meanwhile, the data order is independently shuffled to promote sufficient generalization and exploration. This structure ensures the model is exposed to varied learning conditions throughout the curriculum. Notably, the dynamic length reward mechanism is introduced exclusively during the hard stage to further strengthen the model’s capacity for complex reasoning. The rationale behind this design is twofold: 1) in the easy and medium stages, the absence of the dynamic length reward allows the model to explore the task space more freely and achieve rapid adaptation and initial performance gains; 2) the challenging questions in the hard stage are the main driving force for model to increases reasoning length and explore the complex reasoning chains. Meanwhile, due to this increased reasoning length brought by the dynamic length reward, the final phase takes more steps for the model to converge than the easy and medium stages.

### 5 Experiments

- 5.1 Experimental Settings

Benchmark datasets. To comprehensively assess the performance of VL-Cogito, we select a diverse set of challenging multimodal benchmarks spanning multiple domains:

- • Mathematics and Logic Reasoning: For mathematical problems, we choose the test set of Geometry@3k (Geo3k; hiyouga (2025)), MathVision (Wang et al., 2024) test set, the testmini set of MathVista (Lu et al., 2024), and MathVerse (Zhang et al., 2024). For logical reasoning, we use LogicVista (Xiao et al., 2024). For chart understanding, we utilize the test set of ChartQA (Masry et al., 2022).
- • Science-related Reasoning: The test sets of ScienceQA (SciQA; Lu et al. (2022b), MMMU (Yue et al., 2024), and EMMA (Hao et al., 2025) are selected in our benchmark suite to evaluate the model’s capability in the scientific domain.
- • General Understanding: we further use a general image understanding benchmark, MMStar (Chen et al., 2024), to measure the model’s fundamental visual understanding capability.

Baselines. We compare VL-Cogito against a diverse set of baseline models, encompassing both generalpurpose and reasoning-oriented MLLMs of similar model size:

- • General-purpose MLLMs: Qwen2.5-VL-Instruct-7B (Bai et al., 2025), InternVL2.5-8B (Chen et al., 2025d), InternVL3-8B (Zhu et al., 2025), and LLaVA-OneVision-7B (LLaVA-OV; Li et al. (2024a)), where are the recent state-of-the-art MLLMs.
- • Reasoning-oriented MLLMs: MM-Eureka-8B (Meng et al., 2025), R1-VL-7B (Zhang et al., 2025), MMR17B (Leng et al., 2025), R1-OneVision-7B (Yang et al., 2025), OpenVLThinker-7B (Deng et al., 2025), Vision-R1-7B (Huang et al., 2025), VL-Rethinker (Wang et al., 2025a), and ThinkLite-VL-7B (Wang et al., 2025d).

###### Table 1 Performance comparison of VL-Cogito with other MLLMs on an extensive set of multimodal reasoning benchmarks that encompass mathematical, scientific, and general-domain tasks. All baseline models are reevaluated under identical experimental conditions to ensure a fair comparison; values shown in parentheses denote the results reported in the corresponding original papers. We exclude results for models where benchmark contamination occurred during training, marked with “-”. The bold and underline indicate the best and the second-best scores, respectively.

Mathematics Science General Geo3K MathVerse MathVista MathVision LogicVista ChartQA SciQA MMMU MMMU-Sci EMMA MMStar

Model Size

General-Purpose Models

Qwen2.5-VL 7B 61.6 50.4 (49.2) 69.3 (68.2) 28.7 (25.1) 44.0 82.4 85.4 50.9 44.6 24.6 62.5 (63.9) InternVL2.5 8B 60.6 40.0 (39.5) 61.4 (64.4) 19.9 (19.7) 37.7 (36.0) 73.4 90.3 43.1 (48.9) 35.0 19.9 62.2 (62.8) InternVL3 8B 63.3 49.4 (39.8) 68.5 (71.6) 30.0 (29.3) 41.3 (44.1) 81.3 89.3 50.8 40.6 14.5 67.5 (68.2) LLaVA-OV 7B 48.5 33.6 (26.2) 56.4 (63.2) 15.9 30.6 65.0 80.5 41.6 34.8 18.3 53.5 (61.7)

Reasoning-Oriented Models

MM-Eureka 8B 67.2 52.3 (50.3) 73.4 (73.0) 29.4 (26.9) 47.1 82.7 86.4 52.3 46.7 27.4 64.7 R1-VL 7B 57.5 41.3 (40.0) 61.5 (63.5) 23.0 (24.7) 36.3 76.3 86.0 38.1 33.4 24.0 55.6 (60.0) MMR1 7B 65.9 52.5 (45.1) 73.6 (71.0) 32.9 (30.2) 46.6 (50.8) 82.8 86.7 53.1 47.3 28.1 66.3 R1-OneVision 7B 57.9 44.0 (46.4) 60.3 (64.1) 22.0 (29.9) 40.0 72.5 85.3 43.4 37.3 22.2 56.2 OpenVLThinker 7B 60.6 48.1 (47.9) 70.6 (70.2) 22.0 (25.3) 41.0 81.0 85.9 50.9 43.9 24.9 62.8 VL-Rethinker 7B 67.7 54.6 (54.2) 73.7 (74.9) 30.1 (32.3) 45.7 83.5 86.7 52.9 46.5 28.6 (29.7) 64.2 Vision-R1 7B 67.0 51.9 (52.4) 72.1 (73.5) - 44.7 82.7 - 26.4 26.3 28.3 65.4 ThinkLite-VL 7B 63.5 51.3 (50.7) 72.5 (75.1) 27.5 44.3 83.1 - 50.9 44.3 26.4 64.6 (65.0)

VL-Cogito 7B 68.7 53.3 74.8 30.7 48.9 83.4 87.6 52.6 47.0 29.1 66.3

Evaluation. We adopt a unified prompt instruction across all evaluations and require models to enclose their final answers within “\box{}”, where the complete prompt is presented in Appendix B. Model inference is conducted using vLLM (Kwon et al., 2023) for accelerating generation. For benchmarks that provide official evaluation protocols such as MathVision and MMMU, we strictly adhere to and utilize their respective original evaluation procedures. For other benchmarks, mathematics-related questions are evaluated using Math-Verify1 and MathRuler2, whereas non-mathematical questions are assessed through strict exact matching. To further enhance the robustness and fairness of the evaluation process, two supplementary measures are implemented. First, for multiple-choice questions where the model’s generated answer does not correspond to any of the provided options, the most semantically similar option to the generated answer is selected as the final answer. Second, in cases involving open-ended questions where answers cannot be exactly matched or parsers fail to accurately extract the answers, GPT-4o (OpenAI, 2024a) is employed as an auxiliary judgment tool.

Implementation Details. We utilize the Qwen2.5-VL-Instruct-7B as the backbone model. We train our model using the AdamW optimizer with a learning rate of 1e-6 under the DeepSpeed-ZeRO3 (Rajbhandari et al.,

- 2020) configuration. For the GRPO configuration, we set the rollout batch size to 512, the global batch size to 128, and the maximum sequence length to 4,096; the KL divergence loss coefficient is configured at 1e-3; and 16 responses are sampled for each prompt with a temperature of 1.0. The system prompt used for the training is presented in Appendix B. For the progressive curriculum RL setting, our empirical experiments across easy, medium, and hard stages indicate that the reward and validation accuracy are plateaued after approximately 100 optimization steps in the easy and medium stages due to their relatively simple data. Conversely, the hard stage requires substantially more training steps to achieve improved performance. Thus, we conduct policy optimization for 100 steps each during both the easy and medium stages, selecting the optimal checkpoint based on the validation performance to serve as the starting point for the subsequent stage. In the hard stage, the model is trained for 1 epoch (approximately 200 steps), as incorporating the length reward necessitates more training steps to achieve convergence due to the increased reasoning complexity. For hyperparameters in reward, we set α = 1, β = 0.5, γ = 1 and w = 0.25. The dynamic length reward

is considered as punishment, so rlenmin is set to -1 and rlenmax it set to 0. The resultant model is denoted as VL-Cogito. For the baseline models, we directly download their open-source versions from HuggingFace3

and deploy them under the same environment as ours.

- 1https://github.com/huggingface/Math-Verify
- 2https://github.com/hiyouga/MathRuler
- 3https://huggingface.co/

###### Table 2 The component-wise performance decomposition of the PCuRL framework, where “+Curriculum” and “+DyLR” represent the addition of progressive curriculum strategy and dynamic length reward to GRPO, respectively. The bold and underline indicate the best and the second-best scores, respectively.

Mathematics Science General

Model

Avg Geo3K MathVerse MathVista MathVision LogicVista ChartQA SciQA MMMU EMMA MMStar

Vanilla GRPO 66.1 52.2 71.4 30.0 44.0 83.8 87.8 51.4 28.0 66.3 58.1 +Curriculum 67.4 51.9 74.0 30.4 47.5 83.9 87.6 52.7 28.3 65.2 58.9 +DyLR 66.2 52.5 73.3 29.4 48.4 83.2 87.1 52.6 28.2 64.9 58.6 VL-Cogito 68.7 53.3 74.8 30.7 48.9 83.4 87.6 52.6 29.1 66.3 59.5

- 5.2 Performance Comparison on Diverse Multimodal Benchmarks

- Table 1 presents a detailed comparison between VL-Cogito and a diverse set of both general-purpose and reasoning-oriented MLLMs across ten different multimodal benchmarks. General-purpose models, such as the InternVL series and Qwen2.5-VL, demonstrate strong performance in general and certain scientific domains. However, they typically underperform compared to reasoning-oriented MLLMs on mathematical benchmarks, which demand advanced analytical and reasoning capabilities to reach correct answers. In comparison to the backbone model Qwen2.5-VL-Instruct, VL-Cogito exhibits substantial performance enhancements across all evaluated benchmarks, encompassing mathematics, science, and general domains. VL-Cogito achieves absolute gains of 7.6%, 5.5%, and 4.9% on Geometry@3K, MathVista, and LogicVista, respectively, as well as improvements of 2.2% on ScienceQA, 4.5% on EMMA, and 3.8% on MMStar. These results clearly demonstrate that our approach yields consistent and robust improvements across a diverse range of domains, rather than being tailored solely to a particular domain.

Among reasoning-oriented models, VL-Cogito attains either the best or highly competitive performance even without necessitating a cold-start warm-up phase. This result underscores the effectiveness of our proposed progressive curriculum learning strategy. Notably, it achieves the highest performance on 6 out of 10 multimodal benchmarks, demonstrating exceptional capability on rigorous mathematics and scientific benchmarks, while also attaining superior results on comparatively less complex tasks such as ScienceQA and MMStar. Compared to VL-Rethinker, VL-Cogito achieves marginally lower performance only on MathVerse, ChartQA, and MMMU, while outperforming it across other benchmarks. VL-Rethinker adopts a forced rethinking strategy, which explicitly directs the model to engage in multiple iterations of reasoning during RL training. In contrast, VL-Cogito exclusively leverages the model’s capacity for autonomous exploration, without relying on additional guidance. Although models such as R1-VL, R1-OneVision, OpenVLThinker, and Vision-R1 utilize cold-start initialization with meticulously curated SFT datasets, VL-Cogito is consistently superior to these approaches. These findings underscore the efficacy of our RL pipeline in enhancing the model’s reasoning capabilities across a diverse range of domains.

5.3 Ablation Study

In this section, we conduct a comprehensive ablation study to analyze the effectiveness of each component within PCuRL, including the multi-stage curriculum strategy, online difficulty soft weighting, and dynamic length reward. Furthermore, we examine the training dynamics by visualizing key aspects of the reinforcement learning process, including reward trajectories and length propagation curves, to provide deeper insights into the model’s learning behaviors.

- 5.3.1 Component-wise Performance Decomposition of PCuRL

- Table 2 presents a systematic ablation that quantifies the marginal contribution of each PCuRL component. Compared to the vanilla GRPO baseline, VL-Cogito enhances the performance on the more demanding benchmarks,i.e., MathVision, LogicVista, MMMU, and EMMA, while preserving comparable results on the easier suites, i.e., ChartQA, ScienceQA, and MMStar, confirming that our pipeline is more beneficial where reasoning depth is critical.

Introducing the multi-stage progressive curriculum alone, i.e., “+Curriculum,” yields consistent gains across nearly every task, underscoring the value of an easy-to-hard progression. Appending the dynamic length

###### Table 3 Ablation study examining the Online Difficulty Soft weighting (ODSW) in our progressive curriculum RL settings. “Binary” denotes the binary weighting strategy, where we assess three difficulty ranges of [Tmin, Tmax]; ODSW “Easy”, “Medium”, and “Hard” represent only utilizing the three ODSW variants during the RL training, respectively. The bold and underline indicate the best and the second-best scores.

Mathematics Science General

Model

Avg Geo3K MathVerse MathVista MathVision LogicVista ChartQA ScienceQA MMMU EMMA MMStar

Binary [0.50,1.00] 59.6 49.3 72.0 27.8 41.5 82.6 87.1 49.1 25.2 64.5 55.9 Binary [0.25,0.75] 67.2 51.9 72.3 30.4 46.7 84.7 87.6 51.0 27.0 64.4 58.3 Binary [0.00,0.50] 65.6 51.4 73.1 29.5 45.3 83.9 86.4 51.8 25.5 65.4 57.8

ODSW Easy only 64.2 51.8 72.4 28.6 48.2 83.4 87.3 51.3 28.2 64.1 58.0 ODSW Medium only 68.2 51.9 74.7 29.9 44.2 83.9 88.3 52.4 27.9 64.8 58.6 ODSW Hard only 68.2 52.3 74.5 29.5 46.2 84.0 88.0 52.7 27.0 64.6 58.7

VL-Cogito 68.7 53.3 74.8 30.7 48.9 83.4 87.6 52.6 29.1 66.3 59.5

reward at the final curriculum stage further elevates scores on the most challenging mathematical datasets, e.g., MathVista and LogicVista, pushing the overall average score to 59.5%. In contrast, when the dynamic length reward is applied directly, i.e., “+DyLR,” to the GRPO baseline—without the stabilizing effect of progressive curriculum learning—the model exhibits unstable and inconsistent performance. We suppose that, during the early phases of RL training, many format-mismatched or ill-structured queries are incorrectly interpreted by the model as inherently “difficult” due to their lack of correct responses. As a result, DyLR prematurely encourages the model to extend its reasoning length excessively in response to these misclassified inputs, which not only leads to inefficient learning but also disrupts the optimization trajectory, ultimately hindering convergence and generalization.

- 5.3.2 Impact of Online Difficulty Soft Weighting on Model Performance

To assess how various difficulty weighting methods influence the effectiveness of VL-Cogito, we conduct a systematic ablation using single-stage RL training with different weighting strategies, and the results are summarized in Table 3.

Soft weighting consistently surpasses binary weighting. The results indicate that online difficulty soft weighting (ODSW) achieves superior performance relative to binary weighting schemes across the majority of benchmarks. Notably, when target difficulty is highly skewed or imbalanced, the advantages of the soft weighting approach become particularly pronounced. This suggests that soft weighting maintains a broader and more balanced optimization direction by dynamically emphasizing appropriate difficulty levels. Conversely, excessively focusing on either solely easy or hard samples tends to degrade the overall reasoning capability. Consequently, we adopt online difficulty soft weighting within our multi-stage progressive curriculum RL framework to sustain comprehensive model development.

Exposure to challenging queries is pivotal for deep reasoning gains. Our analysis further reveals that models receiving disproportionately more training on easier samples generally exhibit suboptimal performance. In contrast, placing greater emphasis on more challenging problems—especially through a soft weighting approach—consistently improves outcomes, underscoring the essential role of difficult questions in advancing reasoning ability. This insight motivates our design choice to progressively prioritize challenging tasks during the final stages of curriculum RL training, thereby maximizing the model’s reasoning proficiency.

- 5.3.3 Impact of Dynamic Length Reward on Reasoning Efficiency and Performance

- Figure 3 illustrates how different length-reward strategies influence the average reasoning accuracy of VLCogito, alongside the corresponding average token lengths of generated responses on representative multimodal benchmarks. Specifically, we compare the performance of our dynamic length reward approach, with varying target lengths, against the fixed-length cosine reward method proposed by Yeo et al. (2025). The fixed-length method uniformly encourages all incorrect outputs to reach a pre-determined target length, regardless of the actual complexity or demands of the questions. We denote our dynamic length reward as Dynamic-N, where N ∈ {500,750,1000}, and the fixed-length cosine reward as Fix-N, where we set N = 750. Dynamic length reward consistently surpasses fixed-length reward. Our findings reveal that models guided

29/07/2025, 14:38 Dual axes, line and column Demo | Highcharts

##### 1

800

60.0

AverageAccuracy(%)

600

59.4

AverageLength

400

58.8

200

58.2

0

57.6

Fix-750 Dynamic-500 Dynamic-750 Dynamic-1000

MathVerse Geo3K MathVision ScienceQA MMStar Average Accuracy

- Figure 3 Performance comparison of models trained with different length reward strategies. “Dynamic-N” denotes models employing our dynamic length reward with a target length of N during the final stage of curriculum RL. “Fix-N” refers to models trained with a fixed-length reward that enforces the fixed target length of N across all responses. We visualize both the average response length and the overall accuracy across selected benchmarks.

0 50 100 150 200 250 300 350

Step

0.4

0.5

0.6

0.7

0.8

0.9

1.0

1.1

AverageReward

Vanilla GRPO

Easy Stage

Medium Stage

Hard Stage

(a) Average Reward Curve

0 50 100 150 200 250 300 350

Step

0.3

0.4

0.5

0.6

0.7

Accuracy

Vanilla GRPO

Easy Stage

Medium Stage

Hard Stage

(b) Validation Accuracy Curve

0 50 100 150 200 250 300 350

Step

250

300

350

400

450

500

AverageResponseTokens

Vanilla GRPO

Easy Stage

Medium Stage

Hard Stage

(c) Average Response Length Curve

- Figure 4 Training curves for PCuRL (with a target response length of 500 tokens) and vanilla GRPO. The average reward curve indicates the mean reward of sampled responses during training. The validation accuracy curve shows model performance on a held-out validation set (around 1, 000 questions, split from the original training set at initialization) as measured by the accuracy reward function. The average length curve displays the mean response length of sampled outputs during training.

Highcharts.com

by dynamic-length rewards achieve noticeably higher average accuracy across benchmarks, coupled with a more adaptive and context-appropriate response length distribution. Dynamic reward mechanisms tend to generate shorter, more concise reasoning paths overall, while exhibiting significant flexibility—longer responses for difficult problems and shorter responses for simpler ones. In contrast, fixed-length rewards indiscriminately inflate response lengths, often imposing unnecessary complexity and redundant reasoning on relatively straightforward tasks. The inferior accuracy scores further confirm that mere increases in reasoning length do not necessarily translate into enhanced performance in multimodal reasoning tasks.

Dynamic reward selectively lengthens responses primarily for challenging tasks. Further investigation into the influence of target lengths highlights a clear trend: as we elevate the target reasoning length (from 500 to 1,000 tokens), response lengths on mathematics-related datasets grow steadily, reflecting the inherent complexity and depth requirements of these tasks. While benchmarks requiring less reasoning, such as ScienceQA and MMStar, remain unaffected and maintain concise responses. This pattern confirms our expectation that dynamic length rewards strategically and beneficially extend response lengths specifically for challenging queries, thereby optimizing reasoning depth precisely where it matters most.

https://www.highcharts.com/demo/highcharts/combo-dual-axes 1/1

- 5.3.4 Comparison of Training Dynamics between PCuRL and Vanilla GRPO

#### Figure 4a demonstrates that, during its initial stages, PCuRL’s reward trajectory closely aligns with that of vanilla GRPO, suggesting similar reward functions focused primarily on correctness. Both the easy and medium stages exhibit steadily increasing rewards. At the onset of the hard stage, PCuRL’s average reward drops slightly below the baseline due to the dynamic length reward, which penalizes responses beneath

||What is the predominant feature in the image? A.Cars B.People C.Buildings D.Skyscrapers|
|---|
<br><br>[Figure 21]<br><br>|Which of the organisms is the decomposer in this food web? A.green algae B.water mold C.golden algae|
|---|
<br><br>|To determine the decomposer in the given food web, let's analyze the roles of the organisms involved: 1. Green algae: These are producers, as they produce their own food through photosynthesis. They are at the base of the food web and are consumed by other organisms. 2. Golden algae: These are also producers, similar to green algae, and are part of the food web but do not fit the role of a decomposer. 3.Water mold: This organism is typically a decomposer in aquatic ecosystems. Decomposers break down dead organic matter and waste products, returning nutrients to the environment. In this food web, the water mold is shown to be connected to the black crappie, which suggests it is breaking down the black crappie's waste or dead body. Given the options and the role of the water mold in the food web, the correct answer is: \boxed{B}.|
|---|
<br><br>[Figure 22]<br><br>[Figure 23]<br><br>|If c=5, find a. A.2.5 B.5 C.7.5 D.10|
|---|
<br><br>|To solve for ( a ) in the given right triangle, we need to use the properties of the 30-60-90 triangle. In a 30-60-90 triangle, the sides are in the ratio ( 1 : sqrt{3} : 2 ). Here, the side opposite the 30-degree angle is half the hypotenuse, and the side opposite the 60-degree angle is ( sqrt{3} ) times the side opposite the 30-degree angle. Given:<br><br>- The hypotenuse ( c = 5 )<br>- The angle at ( A ) is 30 degrees<br>- The angle at ( B ) is 60 degrees<br>- The angle at ( C ) is 90 degrees Since the side opposite the 30-degree angle is half the hypotenuse, we can denote the side opposite the 30-degree angle as ( a ) and the side opposite the 60-degree angle as ( b ). Therefore, we have: a = frac{c}{2} = frac{5}{2} = 2.5 So, the value of ( a ) is ( 2.5 ). The correct answer is \boxed{A}.<br>|
|---|
<br><br>|Circles of radius 2 and 3 are externally tangent and are circumscribed by a third circle. Find the area of the shaded region. A.3pi B.4pi C.6pi D.9pi E.12pi|
|---|
<br><br>|To solve the problem, we need to find the area of the shaded region, which is the area of the largest circle minus the areas of the two smaller circles.<br><br>1. Determine the radius of the largest circle: The two smaller circles are tangent to each other and to the largest circle. The distance between the centers of the two smaller circles is the sum of their radii, which is (3 + 2 = 5). The center of the largest circle is the midpoint of the line segment joining the centers of the two smaller circles. The radius of the largest circle is the distance from the center of the largest circle to the center of one of the smaller circles, which is the hypotenuse of a right triangle with legs of 3 and 2. Using the Pythagorean theorem, the radius (R) of the largest circle is:<br><br>R = sqrt{3^2 + 2^2} = sqrt{9 + 4} = sqrt{13}<br><br>However, we need to re-evaluate the problem because the radius of the largest circle is actually the sum of the radius of the larger circle and the distance from the center of the larger circle to the point of tangency, which is (3 + 2 = 5). So, the radius of the largest circle is (3 + 2 = 5).<br><br>2. Calculate the area of the largest circle: The area of the largest circle is: pi R^2 = pi times 5^2 = 25pi<br>3. Calculate the areas of the two smaller circles: The area of the circle with radius 3 is: pi times 3^2 = 9pi<br><br>The area of the circle with radius 2 is:<br><br>pi times 2^2 = 4pi<br><br>4. Calculate the area of the shaded region: The area of the shaded region is the area of the largest circle minus the areas of the two smaller circles:<br><br><br>25pi - 9pi - 4pi = 12pi Therefore, the area of the shaded region is \boxed{E}.|
|---|
<br><br>[Figure 24]<br><br>|To determine the predominant feature in the image, let's analyze the visual elements step by step: 1.Cars: There are cars in the image, but they are not the most prominent feature. They are part of the scene but are not the main focus. 2.People: There are many people in the image, and they are a significant part of the scene. However, they are not the most dominant feature; they are part of the bustling city life. 3.Buildings: The image shows a cityscape with numerous buildings. These buildings are tall and appear to be skyscrapers, which are a characteristic feature of a busy city like New York. 4.Skyscrapers: The image is dominated by tall buildings, which are the defining characteristic of a city like New York. The skyscrapers are the most prominent feature in the image, as they tower over the street and are the focal point of the scene. Given this analysis, the predominant feature in the image is the skyscrapers. The correct answer is \boxed{D}.|
|---|
<br><br>|
|---|

- Figure 5 Case studies of VL-Cogito, where samples are drawn from multiple benchmarks, including MMStar, ScienceQA, Geometry@3K, and MathVision.

the growing target length. The model initially struggles to meet this expanded requirement, temporarily reducing its average reward. However, as it learns to generate longer responses, the reward recovers, eventually matching or exceeding that of the vanilla model.

- Figure 4b illustrates the corresponding validation accuracy. In the easy and medium stages, PCuRL’s accuracy mirrors that of vanilla GRPO, both rising from 0.3 to around 0.7. As depicted in Figure 4c, the hard stage introduces a significant increase in response length, coinciding with PCuRL’s validation accuracy surpassing the baseline. While vanilla GRPO’s accuracy saturates at approximately 0.7, PCuRL continues to improve, reaching a new peak. This result suggests that the extended reasoning chains produced during the hard stage effectively enhance problem-solving performance, rather than merely increasing output length.
- Figure 4c displays the average response length for both methods. Vanilla GRPO’s response length remains relatively static, fluctuating between 250 and 300 tokens, and even exhibits a slight downward trend, highlighting its limitation in encouraging diverse or extended reasoning. In contrast, PCuRL effectively modulates response length according to its curriculum: during the easy and medium stages, lengths remain similar to the baseline, indicating a focus on correctness. In the hard stage, the introduction of the dynamic length reward drives a marked increase in response length, rising consistently from approximately 280 to

the target of 500 tokens. This demonstrates the curriculum’s capacity to guide the model toward more sophisticated outputs incrementally.

In summary, these findings indicate that the progressive curriculum adopted in PCuRL is highly effective in training MLLMs to generate longer and more intricate reasoning. Unlike standard GRPO, which neglects response length, PCuRL’s use of a dynamic length reward in the hard stage not only achieves the desired target length but also leads to notable gains in validation accuracy. The observed reward and accuracy curves collectively confirm the successful adaptation of the model to evolving objectives throughout the curriculum.

- 5.4 Case Study

- Figure 5 illustrates multimodal reasoning cases of VL-Cogito across diverse domains:

Reasoning Trajectories: For multiple-choice questions in both general and scientific domains, VL-Cogito systematically evaluates each option to identify the correct answer. In contrast, when addressing mathematical problems, the model typically first derives the solution independently and subsequently maps it to the given options. The third and fourth examples further exemplify the model’s capacity to handle math questions of varying complexity. In the comparatively straightforward third case, the model promptly recalls properties of a 30-60-90 triangle and directly provides the solution. Conversely, in the more challenging fourth case, the model adopts a meticulous, stepwise approach, decomposing the problem into four discrete reasoning steps. These examples collectively demonstrate that the model dynamically adjusts its reasoning depth, employing more extensive and detailed reasoning processes when confronted with more complex problems.

Self-Reflective Reasoning: Notably, the model also exhibits self-reflection capabilities, as evidenced in the fourth example (highlighted in red in the figure). Initially, the model incorrectly applied the Pythagorean theorem when calculating the radius of the largest circle, leading to an erroneous result. However, it promptly recognized and addressed this error by invoking the “re-evaluate” mechanism, subsequently correcting its calculation and ensuring the accuracy of the remaining steps. This behavior underscores the effectiveness of our RL-based training pipeline in instilling valuable self-reflective abilities within multimodal models.

- 6 Conclusion

In this work, we introduce VL-Cogito, an advanced multimodal large language model (MLLM) enhanced by a progressive curriculum-based reinforcement learning framework termed PCuRL, designed to systematically improve multimodal reasoning capabilities. The multi-stage curriculum embedded within PCuRL progressively guides the model on tasks of increasing complexity, enhancing its proficiency in addressing reasoning challenges across various domains. A key component of our framework is an online difficulty soft weighting mechanism, which dynamically adjusts task difficulty in response to the model’s evolving abilities, facilitating a balanced transition from simpler to more complex tasks. Furthermore, the dynamic length reward mechanism modulates the length of the model’s responses according to problem-specific demands, optimizing the balance between reasoning depth and efficiency. Experimental results demonstrate that VL-Cogito achieves the state-ofthe-art or highly competitive performance across various benchmarks, underscoring the substantial potential of meticulously crafted curriculum learning strategies to broaden the applicability of multimodal reasoning models.

### References

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), December 2015. https://openaccess.thecvf.com/content_iccv_2015/papers/Antol_VQA_Visual_ Question_ICCV_2015_paper.pdf.

Sanghwan Bae, Jiwoo Hong, Min Young Lee, Hanbyul Kim, JeongYeon Nam, and Donghyun Kwak. Online difficulty filtering for reasoning oriented reinforcement learning. ArXiv, abs/2504.03380, 2025. https://arxiv.org/abs/2504. 03380.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. ArXiv, abs/2502.13923, 2025. https://arxiv.org/abs/2502.13923.

Jing Bi, Susan Liang, Xiaofei Zhou, Pinxin Liu, Junjia Guo, Yunlong Tang, Luchuan Song, Chao Huang, Guangyu Sun, Jinxi He, Jiarui Wu, Shu Yang, Daoan Zhang, Chen Chen, Lianggong Bruce Wen, Zhang Liu, Jiebo Luo, and Chenliang Xu. Why reasoning matters? a survey of advancements in multimodal reasoning (v1). ArXiv, abs/2504.03151, 2025. https://arxiv.org/abs/2504.03151.

Huanqia Cai, Yijun Yang, and Winston Hu. Mm-iq: Benchmarking human-like abstraction and reasoning in multimodal models. ArXiv, abs/2502.00698, 2025. https://arxiv.org/abs/2502.00698.

Jie Cao and Jing Xiao. An augmented benchmark dataset for geometric question answering through dual parallel text encoding. In Proceedings of the 29th International Conference on Computational Linguistics, pages 1511–1520, Gyeongju, Republic of Korea, October 2022. International Committee on Computational Linguistics. https: //aclanthology.org/2022.coling-1.130/.

Felix Chen, Hangjie Yuan, Yunqiu Xu, Tao Feng, Jun Cen, Pengwei Liu, Zeying Huang, and Yi Yang. Mathflow: Enhancing the perceptual flow of mllms for visual mathematical problems. ArXiv, abs/2503.16549, 2025a. https: //arxiv.org/abs/2503.16549.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models? In Advances in Neural Information Processing Systems, volume 37, pages 27056–27087. Curran Associates, Inc., 2024. https://proceedings. neurips.cc/paper_files/paper/2024/file/2f8ee6a3d766b426d2618e555b5aeb39-Paper-Conference.pdf.

Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang Zhou, Te Gao, and Wanxiang Che. Towards reasoning era: A survey of long chain-of-thought for reasoning large language models. ArXiv, abs/2503.09567, 2025b. https://arxiv.org/abs/2503.09567.

Yi Chen, Yuying Ge, Rui Wang, Yixiao Ge, Junhao Cheng, Ying Shan, and Xihui Liu. Grpo-care: Consistency-aware reinforcement learning for multimodal reasoning. ArXiv, abs/2506.16141, 2025c. https://arxiv.org/abs/2506. 16141.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, Conghui He, Botian Shi, Xingcheng Zhang, Han Lv, Yi Wang, Wenqi Shao, Pei Chu, Zhongying Tu, Tong He, Zhiyong Wu, Huipeng Deng, Jiaye Ge, Kai Chen, Kaipeng Zhang, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. ArXiv, abs/2412.05271, 2025d. https://arxiv.org/abs/2412.05271.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025.

DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv, abs/2501.12948,

2025. https://arxiv.org/abs/2501.12948.

Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement. ArXiv, abs/2503.17352, 2025. https://arxiv.org/abs/2503.17352.

Thomas Foster and Jakob Foerster. Learning to reason at the frontier of learnability. ArXiv, abs/2502.12272, 2025. https://arxiv.org/abs/2502.12272.

Lujun Gui and Qingnan Ren. Training reasoning model with dynamic advantage estimation on reinforcement learning. https://github.com/ShadeCloak/ADORA, 2025.

Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, Jingji Chen, Jingjia Huang, Kang Lei, Liping Yuan, Lishu Luo, Pengfei Liu, Qinghao Ye, Rui Qian, Shen Yan, Shixiong Zhao, Shuai Peng, Shuangye Li, Sihang Yuan, Sijin Wu, Tianheng Cheng, Weiwei Liu, Wenqian Wang, Xianhan Zeng, Xiao Liu, Xiaobo Qin, Xiaohan Ding, Xiaojun Xiao, Xiaoying Zhang, Xuanwei Zhang, Xuehan Xiong, Yanghua Peng, Yangrui Chen, Yanwei Li, Yanxu Hu, Yi Lin, Yiyuan Hu, Yiyuan Zhang, Youbin Wu, Yu Li, Yudong Liu, Yue Ling, Yujia Qin, Zanbo Wang, Zhiwu He, Aoxue Zhang, Bairen Yi, Bencheng Liao, Can Huang, Can Zhang, Chaorui Deng, Chaoyi Deng, Cheng Lin, Cheng Yuan, Chenggang Li, Chenhui Gou, Chenwei Lou, Chengzhi

Wei, Chundian Liu, Chunyuan Li, Deyao Zhu, Donghong Zhong, Feng Li, Feng Zhang, Gang Wu, Guodong Li, Guohong Xiao, Haibin Lin, Haihua Yang, Haoming Wang, Heng Ji, Hongxiang Hao, Hui Shen, Huixia Li, Jiahao Li, Jialong Wu, Jianhua Zhu, Jianpeng Jiao, Jiashi Feng, Jiaze Chen, Jianhui Duan, Jihao Liu, Jin Zeng, Jingqun Tang, Jingyu Sun, Joya Chen, Jun Long, Junda Feng, Junfeng Zhan, Junjie Fang, Junting Lu, Kai Hua, Kai Liu, Kai Shen, Kaiyuan Zhang, Ke Shen, Ke Wang, Keyu Pan, Kun Zhang, Kunchang Li, Lanxin Li, Lei Li, Lei Shi, Li Han, Liang Xiang, Liangqiang Chen, Lin Chen, Lin Li, Lin Yan, Liying Chi, Longxiang Liu, Mengfei Du, Mingxuan Wang, Ningxin Pan, Peibin Chen, Pengfei Chen, Pengfei Wu, Qingqing Yuan, Qingyao Shuai, Qiuyan Tao, Renjie Zheng, Renrui Zhang, Ru Zhang, Rui Wang, Rui Yang, Rui Zhao, Shaoqiang Xu, Shihao Liang, Shipeng Yan, Shu Zhong, Shuaishuai Cao, Shuangzhi Wu, Shufan Liu, Shuhan Chang, Songhua Cai, Tenglong Ao, Tianhao Yang, Tingting Zhang, Wanjun Zhong, Wei Jia, Wei Weng, Weihao Yu, Wenhao Huang, Wenjia Zhu, Wenli Yang, Wenzhi Wang, Xiang Long, XiangRui Yin, Xiao Li, Xiaolei Zhu, Xiaoying Jia, Xijin Zhang, Xin Liu, Xinchen Zhang, Xinyu Yang, Xiongcai Luo, Xiuli Chen, Xuantong Zhong, Xuefeng Xiao, Xujing Li, Yan Wu, Yawei Wen, Yifan Du, Yihao Zhang, Yining Ye, Yonghui Wu, Yu Liu, Yu Yue, Yufeng Zhou, Yufeng Yuan, Yuhang Xu, Yuhong Yang, Yun Zhang, Yunhao Fang, Yuntao Li, Yurui Ren, Yuwen Xiong, Zehua Hong, Zehua Wang, Zewei Sun, Zeyu Wang, Zhao Cai, Zhaoyue Zha, Zhecheng An, Zhehui Zhao, Zhengzhuo Xu, Zhipeng Chen, Zhiyong Wu, Zhuofan Zheng, Zihao Wang, Zilong Huang, Ziyu Zhu, and Zuquan Song. Seed1.5-vl technical report. ArXiv, abs/2505.07062, 2025. https://arxiv.org/abs/2505.07062.

Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. Can mllms reason in multimodality? emma: An enhanced multimodal reasoning benchmark. ArXiv, abs/2501.05444, 2025. https://arxiv.org/abs/2501.05444.

hiyouga. Geometry3k dataset. https://huggingface.co/datasets/hiyouga/geometry3k, 2025. Zhe Hu, Jing Li, Zhongzhu Pu, Hou Pong Chan, and Yu Yin. Praxis-vlm: Vision-grounded decision making via

text-driven reinforcement learning. ArXiv, abs/2503.16965, 2025. https://arxiv.org/abs/2503.16965.

Kung-Hsiang Huang, Hou Pong Chan, May Fung, Haoyi Qiu, Mingyang Zhou, Shafiq Joty, Shih-Fu Chang, and Heng Ji. From pixels to insights: A survey on automatic chart understanding in the era of large foundation models. IEEE Transactions on Knowledge and Data Engineering, 37(5):2550–2568, 2024. https://doi.org/10.1109/TKDE.2024. 3513320.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. ArXiv, abs/2503.06749, 2025. https://arxiv.org/abs/2503.06749.

Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. Dvqa: Understanding data visualizations via question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018. https://openaccess.thecvf.com/content_cvpr_2018/papers/Kafle_DVQA_Understanding_ Data_CVPR_2018_paper.pdf.

Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Akos Kadar, Adam Trischler, and Yoshua Bengio. Figureqa: An annotated figure dataset for visual reasoning. ArXiv, abs/1710.07300, 2018. https://arxiv.org/abs/1710.07300.

Mehran Kazemi, Hamidreza Alvari, Ankit Anand, Jialin Wu, Xi Chen, and Radu Soricut. Geomverse: A systematic evaluation of large models for geometric reasoning. In AI for Math Workshop @ ICML 2024, 2024. https: //openreview.net/forum?id=1AUbiBrOF1.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. arXiv, abs/1603.07396, 2016. https://arxiv.org/abs/1603.07396.

Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017. https: //openaccess.thecvf.com/content_cvpr_2017/papers/Kembhavi_Are_You_Smarter_CVPR_2017_paper.pdf.

Kimi. Kimi k1.5: Scaling reinforcement learning with llms. ArXiv, abs/2501.12599, 2025. https://arxiv.org/abs/ 2501.12599.

Kimi, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, Congcong Wang, Dehao Zhang, Dikang Du, Dongliang Wang, Enming Yuan, Enzhe Lu, Fang Li, Flood Sung, Guangda Wei, Guokun Lai, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haoning Wu, Haotian Yao, Haoyu Lu, Heng Wang, Hongcheng Gao, Huabin Zheng, Jiaming Li, Jianlin Su, Jianzhou Wang, Jiaqi Deng, Jiezhong Qiu, Jin Xie, Jinhong Wang, Jingyuan Liu, Junjie Yan, Kun Ouyang, Liang Chen, Lin Sui, Longhui Yu, Mengfan Dong, Mengnan Dong, Nuo Xu, Pengyu Cheng, Qizheng Gu, Runjie Zhou, Shaowei Liu, Sihan Cao, Tao

Yu, Tianhui Song, Tongtong Bai, Wei Song, Weiran He, Weixiao Huang, Weixin Xu, Xiaokun Yuan, Xingcheng Yao, Xingzhe Wu, Xinhao Li, Xinxing Zu, Xinyu Zhou, Xinyuan Wang, Y. Charles, Yan Zhong, Yang Li, Yangyang Hu, Yanru Chen, Yejie Wang, Yibo Liu, Yibo Miao, Yidao Qin, Yimin Chen, Yiping Bao, Yiqin Wang, Yongsheng Kang, Yuanxin Liu, Yuhao Dong, Yulun Du, Yuxin Wu, Yuzhi Wang, Yuzi Yan, Zaida Zhou, Zhaowei Li, Zhejun Jiang, Zheng Zhang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Zijia Zhao, Ziwei Chen, and Zongyu Lin. Kimi-vl technical report. ArXiv, abs/2504.07491, 2025. https://arxiv.org/abs/2504.07491.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, SOSP ’23, page 611–626, New York, NY, USA, 2023. Association for Computing Machinery. https://doi.org/10.1145/3600006.3613165.

Sicong Leng, Jing Wang, Jiaxi Li, Hao Zhang, Zhiqiang Hu, Boqiang Zhang, Hang Zhang, Yuming Jiang, Xin Li, Deli Zhao, Fan Wang, Yu Rong, Aixin Sun, and Shijian Lu. Mmr1: Advancing the frontiers of multimodal reasoning. https://github.com/LengSicong/MMR1, 2025.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. ArXiv, abs/2408.03326, 2024a. https://arxiv.org/ abs/2408.03326.

Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal ArXiv: A dataset for improving scientific comprehension of large vision-language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14369–14387, Bangkok, Thailand, August 2024b. Association for Computational Linguistics. https://aclanthology.org/2024.acl-long.775/.

Yunxin Li, Zhenyu Liu, Zitao Li, Xuanyu Zhang, Zhenran Xu, Xinyu Chen, Haoyuan Shi, Shenyuan Jiang, Xintong Wang, Jifang Wang, Shouzheng Huang, Xinping Zhao, Borui Jiang, Lanqing Hong, Longyue Wang, Zhuotao Tian, Baoxing Huai, Wenhan Luo, Weihua Luo, Zheng Zhang, Baotian Hu, and Min Zhang. Perception, reason, think, and plan: A survey on large multimodal reasoning models. ArXiv, abs/2505.04921, 2025. https://arxiv.org/abs/ 2505.04921.

Zhuowan Li, Xingrui Wang, Elias Stengel-Eskin, Adam Kortylewski, Wufei Ma, Benjamin Van Durme, and Alan L. Yuille. Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14963–14973, June 2023. https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Super-CLEVR_A_Virtual_Benchmark_To_ Diagnose_Domain_Robustness_in_Visual_CVPR_2023_paper.pdf.

Adam Dahlgren Lindström and Savitha Sam Abraham. Clevr-math: A dataset for compositional language, visual and mathematical reasoning. arXiv, abs/2208.05358, 2022. https://arxiv.org/abs/2208.05358.

Yinpeng Liu, Jiawei Liu, Xiang Shi, Qikai Cheng, Yong Huang, and Wei Lu. Let’s learn step by step: Enhancing in-context learning ability with curriculum learning. arXiv preprint arXiv:2402.10738, 2024.

Zhiyuan Liu, Yuting Zhang, Feng Liu, Changwang Zhang, Ying Sun, and Jun Wang. Othink-mr1: Stimulating multimodal generalized reasoning capabilities via dynamic reinforcement learning. ArXiv, abs/2503.16081, 2025. https://arxiv.org/abs/2503.16081.

Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-GPS: Interpretable geometry problem solving with formal language and symbolic reasoning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6774–6786, Online, August 2021. Association for Computational Linguistics. https://aclanthology.org/2021.acl-long.528/.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems, volume 35, pages 2507– 2521. Curran Associates, Inc., 2022a. https://proceedings.neurips.cc/paper_files/paper/2022/file/ 11332b6b6cf4485b84afadb1352d3a9a-Paper-Conference.pdf.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems, volume 35, pages 2507– 2521. Curran Associates, Inc., 2022b. https://proceedings.neurips.cc/paper_files/paper/2022/file/ 11332b6b6cf4485b84afadb1352d3a9a-Paper-Conference.pdf.

Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. In The Eleventh International Conference on Learning Representations, 2023. https://openreview.net/forum?id=DHyHRBwJUTN.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations, 2024. https://openreview.net/ forum?id=KUNzEQMWU7.

Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019. https://openaccess.thecvf.com/content_CVPR_2019/papers/Marino_ OK-VQA_A_Visual_Question_Answering_Benchmark_Requiring_External_Knowledge_CVPR_2019_paper.pdf.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279, Dublin, Ireland, May 2022. Association for Computational Linguistics. https://aclanthology.org/2022.findings-acl.177/.

Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and C.V. Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 1697– 1706, January 2022. https://openaccess.thecvf.com/content/WACV2022/papers/Mathew_InfographicVQA_WACV_ 2022_paper.pdf.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. ArXiv, abs/2503.07365, 2025. https://arxiv.org/abs/2503.07365.

nimapourjafar. Mm-raven dataset. https://huggingface.co/datasets/nimapourjafar/mm_raven, 2024. OpenAI. Gpt-4o system card. ArXiv, abs/2410.21276, 2024a. https://arxiv.org/abs/2410.21276. OpenAI. Openai o1 system card. ArXiv, abs/2412.16720, 2024b. https://arxiv.org/abs/2412.16720. OpenAI. Introducing o3 and o4-mini, Apr 2025. https://openai.com/index/introducing-o3-and-o4-mini/. Shuai Peng, Di Fu, Liangcai Gao, Xiuqin Zhong, Hongguang Fu, and Zhi Tang. Multimath: Bridging visual and

mathematical reasoning for large language models. ArXiv, abs/2409.00147, 2024. https://arxiv.org/abs/2409. 00147.

Yi Peng, Peiyu Wang, Xiaokun Wang, Yichen Wei, Jiangbo Pei, Weijie Qiu, Ai Jian, Yunzhuo Hao, Jiachun Pan, Tianyidan Xie, Li Ge, Rongxian Zhuang, Xuchen Song, Yang Liu, and Yahui Zhou. Skywork r1v: Pioneering multimodal reasoning with chain-of-thought. ArXiv, abs/2504.05599, 2025a. https://arxiv.org/abs/2504.05599.

Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. ArXiv, abs/2503.07536, 2025b. https://arxiv.org/abs/2503.07536.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. ArXiv, abs/1910.02054, 2020. https://arxiv.org/abs/1910.02054.

Noam Razin, Zixuan Wang, Hubert Strauss, Stanley Wei, Jason D. Lee, and Sanjeev Arora. What makes a reward model a good teacher? an optimization perspective. ArXiv, abs/2503.15477, 2025. https://arxiv.org/abs/2503.15477.

Alex Rutherford, Michael Beukman, Timon Willi, Bruno Lacerda, Nick Hawes, and Jakob Foerster. No regrets: Investigating and improving regret approximations for curriculum discovery. In Advances in Neural Information Processing Systems, volume 37, pages 16071–16101. Curran Associates, Inc., 2024. https://proceedings.neurips. cc/paper_files/paper/2024/file/1d0ed12c3fda52f2c241a0cebcf739a6-Paper-Conference.pdf.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. ArXiv, abs/1707.06347, 2017. https://arxiv.org/abs/1707.06347.

Minjoon Seo, Hannaneh Hajishirzi, Ali Farhadi, Oren Etzioni, and Clint Malcolm. Solving geometry problems: Combining text and diagram interpretation. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 1466–1476, Lisbon, Portugal, September 2015. Association for Computational Linguistics. https://aclanthology.org/D15-1171/.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. ArXiv, abs/2402.03300, 2024. https://arxiv.org/abs/2402.03300.

Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, Ruochen Xu, and Tiancheng Zhao. Vlm-r1: A stable and generalizable r1-style large vision-language model. ArXiv, abs/2504.07615, 2025. https://arxiv.org/abs/2504.07615.

Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Minglan Lin, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang. Reason-rft: Reinforcement fine-tuning for visual reasoning. ArXiv, abs/2503.20752, 2025. https://arxiv.org/abs/ 2503.20752.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, C Chen, C Li, C Xiao, C Du, C Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms, 2025. URL https://arxiv. org/abs/2501.12599, 2025a.

Kwai Keye Team, Biao Yang, Bin Wen, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, Fan Yang, Guorui Zhou, Hao Peng, Haojie Ding, Jiaming Huang, Jiangxia Cao, Jiankang Chen, Jingyun Hua, Jin Ouyang, Kaibing Chen, Kaiyu Jiang, Kaiyu Tang, Kun Gai, Shengnan Zhang, Siyang Mao, Sui Huang, Tianke Zhang, Tingting Gao, Wei Chen, Wei Yuan, Xiangyu Wu, Xiao Hu, Xingyu Lu, Yang Zhou, Yi-Fan Zhang, Yiping Yang, Yulong Chen, Zhenhua Wu, Zhenyu Li, Zhixin Ling, Ziming Li, Dehua Ma, Di Xu, Haixuan Gao, Hang Li, Jiawei Guo, Jing Wang, Lejian Ren, Muhao Wei, Qianqian Wang, Qigen Hu, Shiyao Wang, Tao Yu, Xinchen Luo, Yan Li, Yiming Liang, Yuhang Hu, Zeyi Lu, Zhuoran Yang, and Zixing Zhang. Kwai keye-vl technical report. ArXiv, abs/2507.01949, 2025b. https://arxiv.org/abs/2507.01949.

Georgios Tzannetos, Bárbara Gomes Ribeiro, Parameswaran Kamalaruban, and Adish Singla. Proximal curriculum for reinforcement learning agents. Transactions on Machine Learning Research, 2023. https://openreview.net/ forum?id=8WUyeeMxMH.

Mert Unsal and Aylin Akkus. Easyarc: Evaluating vision language models on true visual reasoning. arXiv, abs/2506.11595, 2025. https://arxiv.org/abs/2506.11595.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. ArXiv, abs/2504.08837, 2025a. https: //arxiv.org/abs/2504.08837.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with MATH-vision dataset. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. https://openreview.net/forum? id=QWTCcxMpPA.

Peiyu Wang, Yichen Wei, Yi Peng, Xiaokun Wang, Weijie Qiu, Wei Shen, Tianyidan Xie, Jiangbo Pei, Jianhao Zhang, Yunzhuo Hao, Xuchen Song, Yang Liu, and Yahui Zhou. Skywork r1v2: Multimodal hybrid reinforcement learning for reasoning. ArXiv, abs/2504.16656, 2025b. https://arxiv.org/abs/2504.16656.

Weiyun Wang, Zhangwei Gao, Lianjie Chen, Zhe Chen, Jinguo Zhu, Xiangyu Zhao, Yangzhou Liu, Yue Cao, Shenglong Ye, Xizhou Zhu, Lewei Lu, Haodong Duan, Yu Qiao, Jifeng Dai, and Wenhai Wang. Visualprm: An effective process reward model for multimodal reasoning. ArXiv, abs/2503.10291, 2025c. https://arxiv.org/abs/2503.10291.

Xiyao Wang, Zhengyuan Yang, Chao Feng, Hongjin Lu, Linjie Li, Chung-Ching Lin, Kevin Lin, Furong Huang, and Lijuan Wang. Sota with less: Mcts-guided sample selection for data-efficient visual reasoning self-improvement. ArXiv, abs/2504.07934, 2025d. https://arxiv.org/abs/2504.07934.

Yaoting Wang, Shengqiong Wu, Yuecheng Zhang, Shuicheng Yan, Ziwei Liu, Jiebo Luo, and Hao Fei. Multimodal chain-

of-thought reasoning: A comprehensive survey. ArXiv, abs/2503.12605, 2025e. https://arxiv.org/abs/2503.12605. Zhenting Wang, Guofeng Cui, Yu-Jhe Li, Kun Wan, and Wentian Zhao. Dump: Automated distribution-level

curriculum learning for rl-based llm post-training. arXiv preprint arXiv:2504.09710, 2025f. Jiaer Xia, Yuhang Zang, Peng Gao, Yixuan Li, and Kaiyang Zhou. Visionary-r1: Mitigating shortcuts in visual reasoning with reinforcement learning. ArXiv, abs/2505.14677, 2025. https://arxiv.org/abs/2505.14677. Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. ArXiv, abs/2407.04973, 2024. https://arxiv.org/abs/2407.04973. Guowei Xu, Peng Jin, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason step-by-step. ArXiv, abs/2411.10440, 2025. https://arxiv.org/abs/2411.10440.

Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. ArXiv, abs/2503.10615, 2025. https://arxiv.org/abs/2503.10615.

Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, and Dacheng Tao. Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search. ArXiv, abs/2412.18319, 2024. https://arxiv.org/abs/2412.18319.

Huanjin Yao, Qixiang Yin, Jingyi Zhang, Min Yang, Yibo Wang, Wenhao Wu, Fei Su, Li Shen, Minghui Qiu, Dacheng Tao, and Jiaxing Huang. R1-sharevl: Incentivizing reasoning capability of multimodal large language models via share-grpo. ArXiv, abs/2505.16673, 2025. https://arxiv.org/abs/2505.16673.

Edward Yeo, Yuxuan Tong, Xinyao Niu, Graham Neubig, and Xiang Yue. Demystifying long chain-of-thought reasoning in LLMs. In ICLR 2025 Workshop on Navigating and Addressing Data Problems for Foundation Models, 2025. https://openreview.net/forum?id=AgtQlhMQ0V.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9556–9567, June 2024. https://openaccess.thecvf.com/content/CVPR2024/papers/Yue_MMMU_A_Massive_Multi-discipline_ Multimodal_Understanding_and_Reasoning_Benchmark_for_CVPR_2024_paper.pdf.

Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. ArXiv, abs/2503.12937, 2025. https://arxiv.org/abs/2503.12937.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024. https://www.ecva.net/papers/eccv_ 2024/papers_ECCV/papers/01270.pdf.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Nianchen Deng, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Han Lv, Lijun Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. ArXiv, abs/2504.10479, 2025. https://arxiv.org/abs/2504.10479.

###### Table 4 Data statistics of the training dataset in the RL stage, where the filter rate refers to the percentage of questions removed in the difficulty sampling, and the data size represents the number of samples after sampling.

Category QA Type Dataset Data size Filter rate

Open-ended Geometry3K 2101 39% Open-ended GeoQA+ 7886 50% Open-ended Geos 367 22% Open-ended GeomVerse 8179 12% Open-ended Inter-GPS 946 26% Open-ended MultiMath 13503 35%

Mathematics

Open-ended Raven 6919 65% Open-ended MM-IQ 2492 7% Open-ended EasyArc 600 0%

Logical

Open-ended CLEVR-Math 1000 92% Open-ended Super-CLEVR 3000 20%

Counting

Open-ended AI2D 5034 35% Multi-choice ScienceQA 1098 82% Open-ended TQA 5959 31%

Science

Open-ended ChartQA 5570 72% Open-ended TabMWP 2963 70% Open-ended DVQA 2589 45% Open-ended FigureQA 2000 40% Open-ended ArXivQA 2000 57% Open-ended InfographicVQA 626 70%

Charts

Open-ended OKVQA 1500 12% Open-ended VQA2.0 1500 22% Open-ended LLaVA-CoT 1500 8%

General

## Appendix

- A Data Statistics

- Table 4 summarizes the distribution of RL training data, where we show both the number of samples and the filter rate per dataset.

- B System Prompt

Training Prompt: We use this system prompt in RL training to encourage the model to follow the reasoning format and put the final answer in a “\boxed{}”.

Evaluation Prompt: To ensure fairness during the evaluation, we adopt a simple prompt that can be generalized to most models rather than the system prompt used in the training.

- Table 5 The system prompt used for RL training.

A conversation between User and Assistant. The User provides an image and asks a question. The Assistant first analyzes both the image and the question, then carefully thinks about the reasoning process step by step, and finally provides the User with an accurate answer. The Assistant must carefully checkout the correctness and validity of each reasoning step. If any errors or inconsistencies are found during the reasoning process, the Assistant reflects and corrects them logically. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> detailed reasoning process here, with potential reflections and corrections </think><answer> final answer here, with the key result enclosed within \boxed{} </answer>

- Table 6 The system prompt used for evaluation.

Please solve the problem step by step and put your answer in one “\boxed{}”. If it is a multiple-choice question, only one letter is allowed in the “\boxed{}”.

