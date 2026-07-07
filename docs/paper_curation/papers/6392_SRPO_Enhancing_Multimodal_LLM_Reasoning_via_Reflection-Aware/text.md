# arXiv:2506.01713v3[cs.CL]5Oct2025

[Figure 1]

[Figure 2]

## SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning

### Zhongwei Wan, Zhihao Dou, Che Liu, Yu Zhang, Dongfei Cui, Qinjian Zhao, Hui Shen, Jing Xiong, Yi Xin, Yifan Jiang, Chaofan Tao, Yangfan He, Mi Zhang, Shen Yan

1ByteDance Seed, 2The Ohio State University Full author list in Contributions.

### Abstract

Multimodal large language models (MLLMs) have shown promising capabilities in reasoning tasks, yet still struggle significantly with complex problems requiring explicit self-reflection and self-correction, especially compared to their unimodal text-based counterparts. Existing reflection methods are simplistic and struggle to generate meaningful, instructive feedback, as the reasoning ability and knowledge limits of pre-trained models are largely fixed during initial training. To overcome these challenges, we propose multimodal Self-Reflection enhanced reasoning with Group Relative Policy Optimization SRPO, a two-stage reflection-aware reinforcement learning (RL) framework explicitly designed to enhance multimodal LLM reasoning. In the first stage, we construct a high-quality, reflection-focused dataset under the guidance of an advanced MLLM, which generates reflections based on initial responses to help the policy model to learn both reasoning and self-reflection. In the second stage, we introduce a novel reward mechanism within the GRPO framework that encourages concise and cognitively meaningful reflection while avoiding redundancy. Extensive experiments across multiple multimodal reasoning benchmarks—including MathVista, MathVision, Mathverse, and MMMU-Pro—using Qwen-2.5-VL-7B and Qwen-2.5-VL32B demonstrate that SRPO significantly outperforms state-of-the-art models, achieving notable improvements in both reasoning accuracy and reflection quality.

Correspondence: Zhongwei Wan at wan.512@osu.edu, Shen Yan at sheny@bytedance.com Project Page: https://srpo.pages.dev/

### 1 Introduction

Multimodal reasoning is central to numerous real-world scenarios, such as interpreting scientific figures, geometric reasoning, and integrated complex image-text comprehension tasks [1–3]. Although recent approaches have attempted to transfer effective RL-based reasoning methods [4–7] from textual models to multimodal scenarios [8–11], these methods generally encounter considerable limitations. Specifically, existing MLLMs typically follow a token-level Markov process [12, 13] during generation, which relies on local dependencies. This often leads to redundant, repetitive, or erroneous reasoning steps in their output [14]. Such issues hinder reasoning models from achieving significant improvements over fast-thinking models; in some cases, their performance is even inferior. For instance, GPT-o1, despite its explicitly structured reasoning pathways, achieves slightly lower accuracy (73.9%) on MathVista compared to Qwen2.5-VL-72B (74.8%) [1, 15]. The

[Figure 3]

[Figure 4]

[Figure 5]

All the sloping edges of the solid figure are 12 cm long. Find Y, the size of angle PNM, correct to two decimal

The fencing costs £37 per metre. the lengths of EF and CD are x metre and y metre....how much will it cost... ?

The fencing costs £37 per metre. the lengths of EF and CD are x metre and y metre....how much will it cost... ?

80

SRPO-7B GRPO-7B Qwen-2.5-VL-7B

75.8

75

Images places.

Images

Images

72.3

70

[Figure 6]

[Figure 7]

[Figure 8]

68.2

###### SRPO

###### GRPO

###### SRPO

65

<think>To find the size of angle PNM...Since P, N, and M are midpoints of...</think> <answer>The answer is 69.30. </answer>

<think>To calculate the cost of ....the cost is 555

<think> To determine the cost of building the fence.... \\text{Perimeter} = AB + BG + GE + EF + FD + DC + CE + EB...So, the perimeter is:\n\\[ \\text{Perimeter} = 5 + 1 + 2 + x....the cost is 555 + 37x. </think> <answer> The answer is 555 + 37x. </answer>

60.6

60

###### + 37x. </think> <answer>The answer is 555 + 37x. </answer>

✔

Score

❌

55.8

<reflection>The steps correctly identified...The explanation could be more concise by avoiding repetition and summarizing the perimeter calculation more clearly. </reflection>

<reflection> The initial reasoning misidentified the structure of the perimeter...To correct the reasoning...</reflection>

55

53.5

52.9

50

<think>We need to re-compute the total length... </think> <answer>The answer is 777. </answer>

<think>To verify, we can use the fact that... </think> <answer>The answer is 69.30.</answer>

46.3

45.4

✔

45

✔

❌

42.3

39.9

40

36.9

[Figure 9]

[Figure 10]

[Figure 11]

Reflecting on correct reasoning steps and refining them yields a more concise reasoning process.

The thinking steps lack autonomous reflection and refinement, leading to incorrect answers.

Reflecting on and revising incorrect reasoning steps leads to a correct answer.

35

MathVista MathVerse MMMU-Pro Physics

Benchmark

Figure 1 Left: Illustrative examples of reflection improving reasoning. Right: Quantitative comparison on benchmark datasets.

primary reason lies in the presence of incorrect and redundant steps, which negatively affect final performance. Previous studies have shown that self-reflection [16, 17] is an effective approach to address this issue. By explicitly encouraging the model to review, evaluate, and revise its own reasoning process, self-reflection helps eliminate unnecessary or incorrect steps, enhances logical coherence, and promotes deeper understanding [18]. However, recent empirical studies [18–20] indicate that the upper bounds of reasoning capabilities in pre-trained models are largely established during the initial pre-training phase. Consequently, these studies indicate that reinforcement learning improves reasoning by activating decision-making within fixed structures, rather than enabling the acquisition of new knowledge or behaviors. To effectively surpass these inherent limitations, external interventions such as advanced reflective experiences or cognitively guided techniques are required. While previous approaches [16, 18, 20] have attempted to enhance self-reflective reasoning through direct prompting or reinforcement learning, their effectiveness remains limited by the constraints imposed during pre-training, making them insufficient for substantially improving reflective reasoning and overall reasoning performance.

[Figure 12]

[Figure 13]

[Figure 14]

Therefore, designing effective enhancement strategies to improve the intrinsic reasoning capabilities of MLLMs beyond the constraints established during pre-training remains a challenging problem. To address this question, inspired by cognitive science emphasizes that robust human reasoning involves active self-reflection and iterative self-correction steps [16, 21–23], we integrate explicit reflective methods within both multimodal Supervised Fine-Tuning (SFT) and RL, enabling models to surpass their intrinsic reasoning boundaries established in the pre-training phase. Unlike previous studies [24–26], which focus solely on enhancing reasoning ability by aligning with extended chain-of-thought supervision, our goals are not only strengthens the model’s reasoning performance but also fosters its capacity for self-reflection.

Motivated by these insights, we introduce SRPO (multimodal Self-Reflection enhanced reasoning with Group Relative Policy Optimization), a novel two-stage reflective training framework specifically designed to promote explicit self-reflection and self-correction behaviors within MLLMs. (i) In the first stage, we utilize an advanced MLLM to generate reflection content based on the discrepancies between the policy model’s outputs and the ground truth. In this process, the model autonomously evaluates its multiple generated responses, identifies errors, and iteratively revises them through reflective reasoning. Subsequently, we leverage these high-quality reflection datasets to perform multimodal reflection-based supervised fine-tuning (SFT), providing a cold-start initialization for subsequent reinforcement learning. (ii) In the second stage, we further propose a reflection-aware RL method built upon the Group Relative Policy Optimization (GRPO) algorithm [4]. Our specifically designed reward function actively incentivizes concise, task-oriented reflection steps, explicitly punishing overly verbose or redundant reflections, thus effectively encouraging MLLMs to adopt meaningful reflective behaviors via RL stage. As illustrated in Figure 1, after two-stage training, SRPO enables MLLMs to autonomously generate reflective reasoning, effectively refine intermediate thinking steps, and consequently achieve improved reasoning performance across various benchmarks compared to the GRPO.

We conduct comprehensive experiments across several widely adopted multimodal reasoning benchmarks, including MathVista [1], MathVison [27], and MMMU-Pro [3], utilizing representative multimodal models (e.g., Qwen-2.5-VL-7B and Qwen-2.5-VL-32B [28]). Results demonstrate that SRPO consistently and significantly

outperforms current state-of-the-art models, achieving notable improvements in reasoning accuracy, reflection quality, and cross-task generalization. These empirical findings provide strong evidence that explicit reflectionoriented training can effectively extend multimodal models’ reasoning capabilities beyond the inherent cognitive boundaries set during pre-training. Our core contributions are summarized as follows:

- • Novel reflection-oriented SFT construction. We introduce a novel reflective data generation pipeline that leverages the original model’s responses. By using a large MLLM (e.g., GPT-o4-mini [7]), we generate corresponding reflection processes aligned with the gold-standard answers. This pipeline is designed to teach the policy model both effective reasoning and reflective thinking.
- • Reflection-aware reinforcement learning. We develop a tailored GRPO-based RL method (SRPO) equipped with an explicit reward function to incentivize meaningful reflective reasoning.
- • Empirical validation and insights. Extensive evaluations across various multimodal reasoning benchmarks confirm that SRPO achieves state-of-the-art performance, clearly demonstrating the effectiveness of selfreflection enhancements in multimodal reasoning contexts.

### 2 Related Works

Reinforcement Learning for LLM Reasoning. Recent advancements in large-scale RL, such as DeepSeekR1 [4], have demonstrated substantial progress in enhancing complex, human-like Chain-of-Thought (CoT) reasoning by utilizing result-oriented or formatting-specific reward signals. In parallel, several studies, including Open-Reasoner-Zero [29], SimpleRL-Zoo [30], AlphaMed [31], and Logic-RL [32], have explored directly fine-tuning base language models using RL without any supplementary supervised fine-tuning stages. Additionally, methods such as Light-R1 [33] and DeepScaler [34] introduce specially constructed cold-start datasets designed explicitly to encourage detailed step-wise reasoning during initial training phases. Meanwhile, recent analyses [18–20] have also shed light on intrinsic limitations of purely RL-based reasoning enhancement strategies. Furthermore, complementary approaches such as VAPO [35], DAPO [12], PTA-GRPO [36], and Dr. GRPO [37] have sought to refine the Group Relative Policy Optimization (GRPO) framework by optimizing reward design and enhancing advantage estimation techniques, thus more effectively promoting deeper reasoning behaviors within language models. In contrast, our work specifically targets multimodal complex reasoning, explicitly emphasizing self-reflection or correction to enhance reasoning performance during both multimodal SFT and RL training phases.

Reinforcement Learning for Multimodal LLM Reasoning. State-of-the-art multimodal reasoning capabilities are largely dominated by proprietary models, such as GPT-o3 and o4 [7], Gemini-2.5-Pro-T [5], and Seed1.5VL-T [38]. Recent studies aim to close this gap via reinforcement learning (RL) on open-source multimodal LLMs. LMM-R1 [9] introduces a two-stage, rule-based RL, though mainly benefiting textual scenarios. Reason-RFT [39] leverages supervised fine-tuning (SFT) with Chain-of-Thought (CoT) data to initialize RL. Vision-R1 [26] enhances multimodal CoT datasets using DeepSeek-R1 and employs progressive thinking suppression in GRPO training. MM-Eureka [8] presents the MMK12 dataset alongside a two-stage RL method, while VL-Rethinker [15] utilizes selective sample replay and explicit textual rethinking triggers to refine multimodal reasoning. R1-V [40] explores RL primarily within visual-centric reasoning tasks but has limited generalization to broader multimodal domains. However, none of these approaches explicitly emphasize self-reflection or correction during both SFT and RL training phases, resulting in suboptimal reasoning performance. Furthermore, poorly designed reward functions leave these methods vulnerable to length redundancy.

### 3 Method of SRPO

In the following sections, we present the detailed methodology of our SRPO training framework, emphasizing our two core contributions: (1) Novel reflection-oriented SFT data construction. In this stage, we construct a reflection dataset to inject reflective capabilities into the policy model. Through training on this dataset, we aimed to achieve two goals: first, to enhance the policy model’s ability for self-reflection and self-correction during cold-start initialization; and second, to effectively transfer the reflective knowledge of large-scale MLLMs into the policy model, enabling it to learn how to reflect effectively gradually. (2) Reflection-aware reinforcement learning, where we propose a tailored GRPO-based RL algorithm, SRPO, equipped with a reflection-aware reward function that promotes reflective reasoning.

Prompt for CoT Generation

[Figure 15]

[Figure 16]

###### Multi-model Dataset

<think> … </think> <reflection> … </reflection> <answer> … </answer>

CoT Samples with Correct Answer

<think> … </think> <answer> … <answer>

[Figure 17]

[Figure 18]

[Figure 19]

Image

[Figure 20]

[Figure 21]

Physics Math General

Prompt for Self-Reflection Generation & Ground Truth

Self-Reflection

CoT Samples with Wrong Answer

Sample

Question

[Figure 22]

CoT Candidates

Selective Dataset

Figure 2 Pipeline of Self-Reflection SFT data construction, including CoT and self-reflection generation.

#### 3.1 Reflection-oriented Cold-start Initialization

##### 3.1.1 Self-Reflection SFT data construction

Motivation. To address the limitations of local dependency in MLLM reasoning, often resulting in redundant, incoherent, or incorrect outputs, self-reflection [16] becomes essential for improving reasoning quality. However, the absence of reflection knowledge and skills can lead to low-quality or superficial self-correction. RL methods typically guide a model towards selecting high-reward reasoning paths already represented within its intrinsic knowledge distribution, rather than inducing genuinely novel cognitive capabilities or knowledge [19]. In contrast, by incorporating external knowledge distillation, supervised fine-tuning (SFT) has been shown to effectively expand the cognitive boundaries of reasoning [20]. To this end, we propose a supervised fine-tuning (SFT) approach that explicitly injects reflection knowledge into the policy model. By learning from high-quality reflective examples, the model acquires the ability to identify, diagnose, and revise flawed reasoning, ultimately enhancing its coherence, efficiency, and self-awareness during the reasoning.

Less is More. To support effective reflection learning, we construct a high-quality dataset generated by advanced MLLMs, containing two types of reflective examples: one for refining correct CoTs by removing redundancy, and another for revising incorrect CoTs through error correction. While prior SFT approaches [24– 26] focus on mimicking correct reasoning, our reflection-oriented SFT explicitly injects reflection knowledge, enabling the model to detect flaws and refine its reasoning. These approaches are complementary—ours enhances reasoning correction and self-improvement. As illustrated in Figure 2, we begin by curating a highquality subset of N = 10,000 multimodal reasoning samples from three large-scale datasets: LLaVA-CoT [25] (100K), Mulberry [24] (260K), and MathV360K [41]. These samples cover diverse domains including physics, mathematics, and general knowledge. Based on this subset, we construct our self-reflection dataset through two complementary strategies: (1) Refinement of correct CoTs, and (2) Revision of incorrect CoTs. For each sample, we first obtain the initial response generated by the policy model through CoT prompting. Then, using the ground truth answer as guidance, we employ a larger MLLM (e.g., GPT-o4-mini) to generate a self-reflection that either revises flawed reasoning or streamlines correct but verbose outputs. Each final sample thus contains three components: the initial response, the generated self-reflection, and the ground truth answer. In our curated data, approximately 30% of initial responses are correct, while the remaining 70% contain reasoning errors, highlighting the necessity of self-reflection for both wrong solution correction and right question refinement.

##### 3.1.2 Cold-start Initialization of Policy Model

This phase equips the policy model πθ (initial MLLMs) with fundamental self-reflective reasoning capabilities, ensuring it can generate proper reflection-aware reasoning paths before reinforcement learning:

Lcold-start = −Eτ∼D

T

log (πinital(a1,< reflection > ... < /reflection >,a2 | q)) . (1)

t=1

Here, a1 is the policy model’s initial response, ⟨reflection⟩...⟨/reflection⟩ denotes the reflection generated by a large LLM, and a2 is the ground truth answer. Given the input prompt q and the policy model πinitial, the objective is twofold: (1) to train the model to revise a1 toward a2 using the reflection ⟨reflection⟩, and

(2) to leverage the reasoning and knowledge embedded in a2 to guide future predictions. This reflection-driven learning process equips the policy model with self-correction capabilities and improves alignment with correct reasoning trajectories.

#### 3.2 Reflection-aware Reinforcement Learning

Under the reinforcement learning framework, complex reasoning tasks often leverage Chain of Thought (CoT) steps to improve prediction accuracy and interpretability [4, 6]. However, simply encouraging CoT generation can result in redundant or misleading reasoning [18–20]. To address this, recent work introduces self-reflection, allowing models to revise their reasoning after generation and improve overall quality. Yet without proper control, models may exploit reflection—e.g., by inflating length or complexity for higher rewards without real gains. To counter this, we propose SRPO, a framework that enhances both reasoning and reflection through carefully designed reward signals that discourage superficial behaviors.

##### 3.2.1 Group Relative Policy Optimization (GRPO)

We utilize Group Relative Policy Optimization (GRPO), following recent advances [4], to optimize our RL-based training. Unlike SFT, which uses token-level losses, GRPO leverages policy gradients calculated from reward losses for effective policy optimization. GRPO promotes exploration of richer and more diverse reasoning solutions by comparing generated responses within sampled groups. Formally, let Q be the question set, πθ

for question q. Let πθ

be the policy model, and {o1,o2,...,oG} be a group of responses from πθ

old

old

denote the frozen reference model. The GRPO optimization objective is defined as follows: JGRPO(θ) = Eq∼Q,{o  i}Gi=1∼πθold  1

ref

 

|oi|

(2)

G

πθ(oi,t|q) πθ

πθ(oi,t|q) πθ

Ai, clip

,1 − ϵ,1 + ϵ Ai − βDKL(πθ∥πref)

min

(oi,t|q)

(oi,t|q)

G

old

old

t=1

i=1

Here, ϵ and β are clipping hyper-parameters and KL-divergence penalty coefficients, respectively. The advantage Ai for each response is computed as:

ri − mean({r1,r2,...,rG}) std({r1,r2,...,rG})

, where {ri}Gi=1 are rewards from the group. (3)

Ai =

GRPO thus replaces the critic model traditionally required in PPO with a computationally efficient intra-group advantage estimation.

##### 3.2.2 Enhanced GRPO with Reflection-Aware Rewards (SRPO)

In SRPO, we aim to achieve two main goals: (1) enhance the policy model’s reasoning ability through RL, and (2) strengthen its capacity for self-reflection. To realize these goals, we introduce an enhanced reward function that specifically targets the reflection process within the CoT reasoning framework. The format of this response can be summarized as first solution → reflection → second refined solution. The total reward Rtotal can be shown as:

Rtotal = Rtask + Rreflection. (4)

Task Reward. The task-specific reward Rtask combines a format reward and an accuracy reward. The format reward Rformat encourages the model to enclose its reasoning process within <think>...</think> tags. The accuracy reward Raccuracy verifies whether the predicted answer matches the ground truth, guiding the model to produce logically consistent and correct outputs for the first solution. This repeated supervision helps enhance the model’s reasoning ability. Formally, Raccuracy is defined as:

Rtask = Rformat =

0.5, if format is correct 0, otherwise

+ Raccuracy =

0.5, if first solution matches gt label 0, otherwise

(5)

Reflection Reward. Let Iref ∈ {0,0.25} indicates proper formatting of reflection segments (enclosed with <reflect> tags) , Lresponse is the total response token length, Ttarget is the optimal length for all responses

and Tmax is maximum response lengths respectively. It is noted that the response contains the first solution, reflection and the second solution guided by reflection. The reward can be defined as:

Rreflection = Ieff + Iref + α flen(Lresponse), (6)

where the reflective brevity reward flen(Lresponse) is explicitly defined to encourage appropriate lengths which can achieve exact and brief results:

2

|L − Ttarget| Tmax − Ttarget

. (7)

flen(Lresponse) = exp −

The reward flen peaks at a target length, encouraging concise, informative reasoning, and decays smoothly toward zero as length approaches a defined maximum. This softly constrains output within a desirable range without hard cutoffs. We adopt this exponential form for its simplicity, differentiability, and stable gradient behavior during training.

Additionally, the effectiveness indicator Ieff provides extra rewards if the reflection genuinely improves reasoning outcomes, measured by improvement in the correctness or accuracy of final answers post-reflection:

 

0.25, if reflection keeps a corrected answer, 0.5, if reflection corrects the wrong answer, 0, if reflection fails to correct the wrong answer, −0.25 if reflection misconducts the right into wrong answer.

(8)

Ieff =



The proposed reward function Ieff focuses on the second solution results and assesses the reflection’s impact on answer correctness in four cases: preserving a correct answer yields +0.25, successfully correcting an incorrect answer results in +0.5, failing to fix a wrong answer receives no reward, and misleading a correct answer incurs -0.25. This design encourages the model to treat reflection not as a formality, but as a tool for improving reasoning quality and avoiding redundancy.

Advantages over Standard GRPO. Compared with standard GRPO—which primarily relies on sparse task-level accuracy supervision—our enhanced reflection-aware SRPO framework introduces several critical improvements: (i) By enforcing structured reflection formatting through the Iref indicator, the model is guided to produce consistently well-organized and identifiable reflection segments. (ii) The introduction of a smooth, differentiable length reward flen(Lref) encourages the generation of reflections that are concise yet informative, avoiding hard cutoffs while softly constraining verbosity. (iii) The effectiveness reward Ieff directly aligns reward signals with functional improvement, providing positive incentives only when the reflection corrects errors or preserves correctness, and penalizing harmful reflections. (iv) By explicitly rewarding reflection utility rather than mere presence, our approach discourages reward gaming behaviors such as empty or verbose reflections, leading to more meaningful reasoning supervision. Together, these enhancements enable SRPO to foster deeper self-correction capabilities, improve sample efficiency, and achieve superior performance in complex reasoning tasks compared to standard GRPO.

### 4 Experiment

#### 4.1 Experiment Settings

Training Dataset. (1) SFT: To construct the self-reflection SFT dataset for the cold-start initialization phase, we first curate samples from several established multimodal reasoning sources, including the Mulberry dataset (260K) [24], MathV360K [41], and LLaVA-CoT dataset (100K) [25]. We then apply the data construction procedure detailed in Section 3.1.1, ultimately resulting in a refined SFT dataset comprising approximately 10K samples. (2) RL: For the subsequent reinforcement learning phase, we aggregate a diverse collection of multimodal reasoning samples from multiple datasets, such as ScienceQA [42], Geometric Math QA [43], ChartQA [44], DVQA [45], AI2D [46], MATH [47], Virgo [48], R1-OneVision [11], MMK12 [8], and PhyX [49]. These datasets collectively encompass mathematical reasoning, general scientific reasoning, and general chart comprehension tasks. The RL training dataset consists of diverse, cross-domain reasoning samples. More details about SFT and RL training dataset collection are shown in Appendix B.2.

Table 1 Comparison between our 7B and 32B models, closed-source baselines, and other vision–language models. † reproduced by us. The results of other baselines are obtained from their official reports. Bold indicates the best-performing open-source model.

Model Math-Benchmark General-Benchmark

MathVista MathVerse MathVision OlympiadBench WeMath MMMU-Pro MMMU EMMA

Closed-Source MLLMs

Claude3.7-Sonnet 66.8 52.0 41.3 48.9 72.6 51.5 68.3 35.1 GPT-4o 63.8 50.2 30.4 35.0 68.8 51.9 69.1 32.7 GPT-o1 73.9 57.0 60.3 68.0 98.7 62.4 78.2 45.7 Gemini2-flash 70.4 59.3 41.3 51.0 71.4 51.7 70.7 33.6 Seed1.5-VL-T 85.6 - 68.7 65.0 - 67.6 77.9 -

Open-Source General MLLMs (7B-16B)

InternVL2-8B 58.3 22.8 17.4 †10.1 †47.2 29.0 51.2 19.8 InternVL2.5-8B 64.4 39.5 19.7 12.3 53.5 34.3 56.0 †20.6 QwenVL2-7B 58.2 19.7 16.3 †9.7 †51.6 30.5 54.1 20.2 Llava-OV-7B 63.2 26.2 †18.5 †8.5 †49.9 24.1 48.8 18.3 Kimi-VL-16B 68.7 44.9 21.4 – – – 55.7 – QwenVL2.5-7B 68.2 46.3 25.1 20.2 62.1 36.9 54.3 21.5

Open-Source Reasoning MLLMs (7B)

MM-Eureka-8B1 67.1 40.4 22.2 8.6 †55.7 27.8 49.2 †21.5 R1-VL-7B 63.5 40.0 24.7 †10.8 †53.8 7.8 44.5 8.3 R1-Onevision-7B 64.1 46.4 23.5 17.3 61.8 21.6 – 20.8 OpenVLThinker-7B 70.2 47.9 25.3 20.1 64.3 37.3 52.5 26.6 VL-Rethinker-7B 74.9 54.2 32.3 †20.5 †70.2 41.7 56.7 29.7 Vision-R1-7B 73.5 52.4 †27.2 †19.4 †62.9 †37.7 †54.7 †22.4 MM-Eureka-7B2 73.0 50.3 26.9 20.1 66.1 †37.6 †55.2 †23.5

⋆ (Ours - SRPO-7B) 75.8 55.8 32.9 22.8 71.6 42.3 57.1 29.6

Open-Source General and Reasoning MLLMs (32B) InternVL2.5-VL-38B 71.9 49.4 31.8 32.0 67.5 46.0 57.6 Qwen-2.5-VL-32B 74.7 48.5 38.4 30.0 69.1 49.5 59.4 31.1 InternVL2.5-38B-MPO 73.8 46.5 32.3 25.6 66.2 - – MM-Eureka-32B 74.8 56.5 34.4 35.9 73.4 †50.4 †62.3 †34.5 ⋆ (Ours - SRPO-32B) 78.5 58.9 39.6 38.5 76.4 51.3 66.1 38.2

Baselines and Benchmarks. To comprehensively evaluate SRPO, we compare against three groups of baselines: (1) Closed-source MLLMs: General-purpose models GPT-4o [50], Claude3.7-Sonnet [51], Gemini2-flash [52], and the reasoningoptimized GPT-o1 [53]; (2) Open-source general MLLMs: Instruction-tuned multimodal models InternVL2.5 [54] and Qwen-2.5-VL [28], ranging from 7B to 78B parameters; and (3) Open-source reasoning MLLMs: Explicitly fine-tuned reasoning models, including InternVL2.5-MPO [55], OpenVLThinker7B [56], MM-Eureka-7B [57], VL-Rethinker-7B [15], R1Onevision-7B [11], and R1-VL-7B [40]. We evaluate SRPO across three categories of multimodal reasoning benchmarks: mathematical reasoning (MathVista [1], MathVerse [2], MathVision [27], OlympiadBench [58], WeMath [59]), general reasoning (MMMU-Pro [3], MMMU [60], EMMA [61]), and crossdisciplinary reasoning (MMK12 [8]), covering physics, chemistry, and biology tasks.

Table 2 Performance comparison across different disciplines in MMK12.

Model Math Phys Chem Bio

Closed Models

Claude3.7 57.4 53.4 55.4 55.0 GPT-4o 55.8 41.2 47.0 55.4 o1 81.6 68.8 71.4 74.0 Gemini2 76.8 53.6 64.6 66.0

Open General MLLMs

IntVL2.5-8B 46.8 35.0 50.0 50.8 Qwen-2.5-7B 58.4 45.4 56.4 54.0 IntVL2.5-38B 61.6 49.8 60.4 60.0 Qwen-2.5-32B 71.6 59.4 69.6 66.6 Qwen-2.5-72B 75.6 64.8 69.6 72.0

Open Reasoning MLLMs

IntVL2.5-8B-MPO 26.6 25.0 42.4 44.0 IntVL2.5-38B-MPO 41.4 42.8 55.8 53.2 R1-OneVision 44.8 33.8 39.8 40.8 MM-Eureka-7B 71.2 56.2 65.2 65.2 OpenVLThinker 63.0 53.8 60.6 65.0 MM-Eureka-32B 74.6 62.0 75.4 76.8 SRPO-7B 75.3 60.6 70.3 69.5 SRPO-32B 77.5 64.2 77.5 79.2

Implementation Setup. For self-reflection cold-start SFT and

subsequent RL training, Qwen2.5-VL-7B-Instruct and Qwen2.5-VL-32B-Instruct models are trained on 8 and 32 NVIDIA H100 GPUs, respectively. We adopt 1 epoch for SFT to avoid overfitting. During RL, we adopt the OpenRLHF framework [62], training for 3 epochs on 30K samples with rollout and training batch sizes

###### Table 3 Ablation study of SRPO-7B on RL training data size and self-reflection components.

Model Components RL Data Size MathVista MathVerse MathVision MMMU-Pro Physics Avg. Qwen-2.5-VL-7B - 68.2 46.3 25.1 36.9 45.4 44.4 + GRPO 37K 72.3 52.9 30.3 39.9 53.5 49.8 ⋆ (Ours - SRPO-7B) 37K 75.8 55.8 32.9 42.3 60.6 53.5 SRPO-7B 15K 74.5 54.9 32.2 41.4 60.1 52.6 SRPO-7B 5K 73.7 53.6 31.2 40.3 57.7 51.3 w/o Self-Reflection SFT 37K 74.2 53.3 30.3 39.7 58.6 51.2 w/o Self-Reflection RL 37K 70.3 48.2 27.2 38.7 48.5 46.6

- - no Length Reward (flen(·)) 37K 75.3 56.2 32.4 41.7 60.1 53.1
- - no Effectiveness Reward (Ieff) 37K 73.9 54.7 31.6 40.9 58.8 52.0

set to 128 (8 rollouts per sample), a sampling temperature of 1.0, and Adam optimizer with a learning rate of 1 × 10−6. For the reflection reward parameter α, we set it to 0.1 to ensure training stability. Regarding the reflective brevity reward flen(Lresponse), to discourage excessively verbose outputs, we define Ttarget as 2× the length of the original response (i.e., reflection plus new reasoning equals the first think length), and set Tmax to 2.5× the original length (i.e., reflection plus new reasoning equals 1.5× the first think length). Additional hyper-parameter settings and detailed prompt configurations are provided in Appendix B.3 and Appendix B.4.

#### 4.2 Main Results of Benchmarks

Multimodal General Reasoning. We further evaluate our approach on general multimodal reasoning tasks to assess the effectiveness of our reflection-enhanced training strategy beyond mathematical reasoning. As shown in Table 1, SRPO-7B consistently outperforms existing open-source MLLMs on three general-domain benchmarks: MMMU-Pro, MMMU, and EMMA. Notably, compared to state-of-the-art closed-source reasoning models, SRPO-32B still demonstrates highly competitive performance, exceeding Gemini2-flash by 4.6 on the EMMA benchmark. These results underscore the broader generalizability of reflection-enhanced training in improving multimodal reasoning capabilities.

Multimodal Mathematical Reasoning. As presented in Table 1, SRPO achieves highly competitive performance on multiple mathematical reasoning benchmarks, even when compared to leading closed-source MLLMs. For instance, on the MathVista benchmark, SRPO obtains a score of 78.5%, trailing the widely acknowledged stateof-the-art model, OpenAI GPT-o1, by only 73.9%. Moreover, SRPO consistently outperforms open-source general multimodal baselines by a clear margin. Notably, when compared to state-of-the-art open-source reasoning models such as VL-Rethinker-7B and MM-Eureka-7B, SRPO demonstrates obvious advantages, even on complex, graduate-level reasoning datasets like OlympiadBench. These results strongly validate our claim that explicitly enhancing the model’s self-reflection capabilities during both the SFT and RL stages positively contributes to improved complex reasoning performance.

Cross-disciplinary Reasoning. Beyond evaluating our model on widely-used multimodal mathematical and general reasoning benchmarks, we also investigate its capability for cross-disciplinary generalization to novel tasks not included in the training data, such as physics, chemistry, and biology. Results presented in Table 2 demonstrate that SRPO achieves superior cross-disciplinary reasoning performance, surpassing MM-Eureka-7B (trained solely via RL without self-reflection incentivization) by 5.1 points on Physics and OpenVLThinker7B (SFT-enhanced reasoning) by 9.7 points on Chemistry. These findings highlight that integrating both reflection-enhanced SFT during the cold-start stage and employing a reflection-aware reward function during the RL stage significantly improves the model’s generalization to previously unseen reasoning domains.

#### 4.3 Ablation Study

RL Train-Set Size. We analyze SRPO’s performance sensitivity to the RL training set size by sampling subsets of 15K and 7K from our original 37K dataset. As shown in Table 4.3, SRPO consistently improves with more data. Remarkably, even at 5K samples, SRPO significantly outperforms Qwen-2.5-VL-7B and standard GRPO, exceeding GRPO by 7.1 points on the Physics benchmark. Thus, enhancing self-reflection within RL

efficiently boosts reasoning even under limited data. Effectiveness of Self-Reflection. We further investigate individual self-reflection components within SRPO. Table 4.3 shows that removing Self-Reflection SFT notably reduces performance, yet still maintains a 5.1-point advantage over standard GRPO on Physics. Conversely, eliminating Self-Reflection RL yields minimal improvements over Qwen-2.5-VL-7B, indicating that reflection training solely in the SFT stage is insufficient. Hence, explicitly incentivizing reflection behavior during RL is essential for ehancing multimodal reasoning. Effectiveness of Reflection-Aware RL Components. We also observe that omitting any specific Self-Reflection RL component can degrade performance, especially when the Effectiveness Reward (Ieff) is removed, resulting in a drop in average performance from 53.5 to 52.0. It indicates that the model critically relies on reward signals that explicitly evaluate the quality of reflective responses to achieve optimal reasoning. Similarly, reducing the Length Reward (flen) also leads to a decline in reasoning performance, suggesting that overly redundant thinking steps can interfere with the model’s accurate reasoning.

Table 4 Comparison between SRPO and GRPO with 2-Step Thinking.

Methods MathVista MathVerse MathVision MMMU-Pro Physics Avg. ⋆ (Ours - SRPO-7B) 75.8 55.8 32.9 42.3 60.6 53.5 GRPO 72.3 52.9 30.3 39.9 53.5 49.8 GRPO + 2-Step Thinking 73.5 53.6 30.6 40.3 53.6 50.3

Comparative Study of Reflective and Two-Step Thinking. To assess the effectiveness of our proposed reflection pattern—comprising thinking, reflection, and rethinking—we compare it against a GRPO-based 2-step thinking paradigm, where the model generates two consecutive <think>...</think> steps. Training uses the task reward from Equation 5 and a relational reward inspired by Equation 8 that captures consistency between the two steps. As shown in Table 4, GRPO + 2-Step Thinking offers no significant gains over vanilla GRPO, except on MathVerse. In contrast, SRPO’s explicit reflection on prior thinking substantially improves reasoning, underscoring the importance of combining Self-Reflection SFT with Reflection-aware RL.

Disentangling the Effect of Reflection Format vs. Teacher Distillation To examine whether the SFT improvement stems from teacher distillation or the proposed reflection structure, we conduct controlled experiments on the 7B model. All models share the same teacher (GPT-4-mini), dataset size (10k), optimizer, and number of training epochs. The only differences lie in the inclusion of <reflection> segments and whether reflectionaware RL is applied. The results are shown in Table 5. Adding explicit reflection segments in SFT yields an average gain of +1.3 over plain CoT SFT, indicating that structured reflection rather than teacher distillation contributes to better reasoning performance. When both models undergo RL, SRPO achieves an additional +2.1 gain by aligning rewards with reflection effectiveness, which highlights the benefit of reflection-aware reward design.

Table 5 Ablation of Self-Reflection SFT on the 7B model.

Methods MathVista MathVerse MathVision MMMU-Pro Physics Avg.

Qwen-2.5-VL-7B 68.2 46.3 25.1 36.9 45.4 44.4 Plain-CoT SFT 69.1 47.2 26.4 37.2 47.6 45.3 Reflection-SFT (ours) 70.3 48.2 27.2 38.7 48.5 46.6 Plain-CoT SFT + GRPO 73.6 54.2 30.6 40.6 58.0 51.4 Reflection-SFT + Reflection-RL 75.8 55.8 32.9 42.3 60.6 53.5

#### 4.4 Reasoning Qualitative Analysis

Self-Reflection in RL Training. Figure 3 The left part compares samples generated during RL training with SRPO and standard GRPO using Qwen-2.5-VL-7B, specifically highlighting the intermediate reasoning steps. We observe that SRPO explicitly guides the model to engage in effective self-reflection on its initial reasoning paths and answers. During the reflection process, SRPO corrects wrong reasoning steps and provides concise revisions, leading to refined final answers. In contrast, GRPO-generated samples typically contain

###### Sample generated for real test case

###### Sample generated during RL training Question:

Question: Given that the measure of angle 1 is 123 degrees in the figure, what is the measure of angle 11? Choices: (A) 57 (B) 67 (C) 113 (D) 123

[Figure 23]

[Figure 24]

###### Assume that both boxes are identical in size. Find the surface area of the solid.Round your answer to two decimal places____.

###### Ground truth: 8090.80

Ground truth: (A) 123

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

###### GRPO SRPO

GRPO SRPO

<think>

<think> Since angle 1 and angle 11 are corresponding angles formed by a transversal intersecting......the measure of angle 1. </think> <answer> The answer is (A).57. </answer> <reflection>

<think>: To find the surface area of the solid, we need to consider the dimensions of......First, we calculate the surface area of one box without the hole:\nThe surface area of a box is given by:\n\\[ \\text{Surface Area} = 2(lw + lh + wh) \\]\nwhere \\( l = 25 \\) cm, \\( w = 20 \\) cm, and \\( h = 33 \\) cm.\......The surface area of the solid is:\n\\[ 7940 - 414.56 = 7525.44 \\text{ cm}^2 \\]\n .Rounding to two decimal places, the surface area of the solid is...... </think> <answer> The answer is 7525.44. </answer>

To compute the total surface area of the solid......\text{Surface Area} = 2(lw + lh + wh)......\text{Lateral Surface Area} = 2\pi rh = 2\pi \times 3 \times 11 = 66\pi......7940 - 414.56 = 7525.44\ \text{cm}^2......

<think>:

At the figure, angle 1 and angle 11 appear to be positioned in similar locations relative to their respective intersections with the transversal lines. Assuming the lines involved are parallel, angle 1 and angle 11 would be corresponding angles. According to the properties of parallel lines cut by a transversal, corresponding angles are congruent. Based on this reasoning, angle 11 should have the same measure as angle 1.

</think> <answer> The answer is 7525.44. </answer> <reflection>

The original reasoning has a significant flaw: it incorrectly identifies angle 1 and angle 11 as corresponding angles formed by a transversal intersecting parallel lines......To improve the reasoning, one should: 1)re-evaluate the geometric relationship...

The first reasoning step incorrectly subtracted the lateral surface......(1) Compute the total surface area of the two boxes.......

</think> <answer> The answer is (A).57. </answer>

</reflection> <think> The correct approach here would be to notice that ......Thus, the measure of angle 11 is 123 degrees. <answer> The answer is (D).123. </answer>

Refinement and correction

</reflection> <think> Each box is identical with dimensions: length = 25 cm......Now compute the total surface area of the composite solid: \[SA = 7940 - 56.55 + 207.35 = 8090.80\ \text{cm}^2\] <answer> The answer is 8090.80. </answer>

Refinement and correction

Wrong answer with uncorrect thinking steps.

Wrong answer with uncorrect thinking steps.

Obtain the right answer

Obtain the right answer

Figure 3 Generated samples in RL training (left) and generated samples in real test case (right).

reasoning steps without explicit reflective corrections, rarely revising the initial reasoning paths. We provide the complete version of samples in Appendix B.5.

Self-Reflection in Test Case. Futhermore, figure 3 right part illustrates examples of reasoning outputs during inference. Responses generated by SRPO exhibit clear self-reflection patterns acquired during RL training, actively refining or correcting flawed reasoning steps and answers to improve overall accuracy. Conversely, models trained without self-reflection rarely adjust or rectify incorrect reasoning paths, resulting in persistent reasoning errors and reduced final performance.

[Figure 29]

[Figure 30]

[Figure 31]

Figure 4 Training curves for SRPO and baselines: (a) training reward, (b) response length, and (c) upper clipping ratio.

#### 4.5 Further Analysis

RL Training Dynamics Analysis. We analyze training dynamics to highlight SRPO’s advantages (Figure 4). SRPO and SRPO w/o self-reflection SFT converge faster and outperform standard GRPO, illustrating that reflection-enhanced initialization accelerates reflection skill acquisition and improves reasoning. Moreover, SRPO consistently generates longer responses (Figure 4(b)), indicating effective early-stage reflection training from cold-start initialization. Interestingly, SRPO’s lower, smoother ratio clip upper curve (Figure 4(c)) reflects stable policy updates, avoiding excessively large gradients or step sizes, confirming enhanced training consistency from reflection-based RL. More training visualizations are shown in Appendix B.6.

Combining Self-Reflection with Alternative RL Methods. To validate

DAPO+Self-reflection DAPO SRPO

GRPO PPO+Self-reflection PPO

| |
|---|

62

| |
|---|

| |
|---|

| |
|---|

| |
|---|

60

Accuracy(%)

58

56

54

52

50

Mathverse MMLU Physics

Figure 5 Performance of various RL methods with and without self-reflection.

the generality of our self-reflection strategy, we incorporate it into PPO and DAPO algorithms within the OpenRLHF framework, following identical cold-start SFT initialization and evaluating checkpoints at 500 training steps. Figure 5 shows consistent improvements from self-reflection integration across all RL algorithms. Reflection-enhanced DAPO achieves performance comparable to SRPO, while SRPO slightly surpasses reflection-enhanced PPO. The result highlights the advantage of GRPO’s group-based advantage estimation and reflection-oriented rewards over PPO’s single-trajectory reward signals for effectively incentivizing selfreflection.

### 5 Conclusion

In this paper, we introduced SRPO, a reflection-aware reinforcement learning framework designed to enhance multimodal reasoning capabilities in mutlimodal large language models. By systematically generating highquality reflection-focused training data and employing a novel reward mechanism that explicitly incentivizes concise and effective self-reflection, our method successfully addresses the limitations of previous approaches, including insufficient data quality and lack of self-reflective behavior for refining response. Comprehensive experiments across multiple multimodal reasoning benchmarks demonstrated the significant effectiveness of SRPO, surpassing existing state-of-the-art models in both reasoning accuracy and reflection quality. Our results highlight the critical role of reflection-driven training strategies for robust multimodal reasoning.

### 6 Acknowledgment

We thank Xuehan Xiong, Kunchang Li and Qinghao Ye for their support in technical discussions related to this work. We also thank Faming Wu for his assistance in addressing infrastructure issues.

### 7 Contributions

#### Project Lead 1

Zhongwei Wan2

#### Core Contributors

Zhongwei Wan2, Zhihao Dou3

#### Contributors

Che Liu4, Yu Zhang11, Dongfei Cui5, Qinjian Zhao6, Hui Shen7, Jing Xiong10, Yi Xin12, Yifan Jiang8, Chaofan Tao10, Yangfan He9

#### Supervisors

Mi Zhang2, Shen Yan1

#### Affiliation

1ByteDance Seed 2The Ohio State University

- 3Case Western Reserve University 4Imperial College London 5Duke University 6Kean University

- 7University of Michigan 1Work done during internship at ByteDance Seed.

- 8University of Southern California
- 9University of Minnesota
- 10The University of Hong Kong
- 11Tongji University 12Nanjing University References

- [1] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

- [2] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024.

- [3] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Ming Yin, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. ArXiv, abs/2409.02813, 2024. URL https: //api.semanticscholar.org/CorpusID:272397682.

- [4] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- [5] Google. Gemini 2.5: Our most intelligent ai model. https://blog.google/technology/google-deepmind/ gemini-model-thinking-updates-march-2025/, March 2025. Accessed: 2025-04-18.
- [6] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025.

- [7] OpenAI. Openai o3 and o4-mini system card. https://openai.com/index/o3-o4-mini-system-card/, April

2025. Accessed: 2025-04-18.

- [8] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, Ping Luo, Yu Qiao, Qiaosheng Zhang, and Wenqi Shao. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. 2025.
- [9] Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025.

- [10] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.

- [11] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025.

- [12] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

- [13] Jieyi Long. Large language model guided tree-of-thought. arXiv preprint arXiv:2305.08291, 2023.

- [14] Jinghan Zhang, Xiting Wang, Fengran Mo, Yeyang Zhou, Wanfu Gao, and Kunpeng Liu. Entropy-based exploration conduction for multi-step reasoning. arXiv preprint arXiv:2503.15848, 2025.

- [15] Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025.

- [16] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36:46534–46594, 2023.

- [17] Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, et al. Training language models to self-correct via reinforcement learning. arXiv preprint arXiv:2409.12917, 2024.

- [18] Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, and Noah D Goodman. Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars. arXiv preprint arXiv:2503.01307, 2025.

- [19] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

- [20] Darsh J Shah, Peter Rushton, Somanshu Singla, Mohit Parmar, Kurt Smith, Yash Vanjani, Ashish Vaswani, Adarsh Chaluvaraju, Andrew Hojel, Andrew Ma, et al. Rethinking reflection in pre-training. arXiv preprint arXiv:2504.04022, 2025.

- [21] Simon Herbert et al. The architecture of complexity. Proceedings of the American Philosophical Society, 106(6): 467–482, 1962.

- [22] Philip David Zelazo. Executive function: Reflection, iterative reprocessing, complexity, and the developing brain. Developmental Review, 38:55–68, 2015.

- [23] Linda Flower and John R Hayes. A cognitive process theory of writing. College Composition & Communication, 32(4):365–387, 1981.

- [24] Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, et al. Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search. arXiv preprint arXiv:2412.18319, 2024.

- [25] Guowei Xu, Peng Jin, Li Hao, Yibing Song, Lichao Sun, and Li Yuan. Llava-o1: Let vision language models reason step-by-step. arXiv preprint arXiv:2411.10440, 2024.

- [26] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.

- [27] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024.

- [28] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

- [29] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. arXiv preprint arXiv:2503.24290, 2025.

- [30] Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025.

- [31] Che Liu, Haozhe Wang, Jiazhen Pan, Zhongwei Wan, Yong Dai, Fangzhen Lin, Wenjia Bai, Daniel Rueckert, and Rossella Arcucci. Beyond distillation: Pushing the limits of medical llm reasoning with minimalist rule-based rl. arXiv preprint arXiv:2505.17952, 2025.

- [32] Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo. Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2502.14768, 2025.

- [33] Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang, Xiaowei Lv, et al. Light-r1: Curriculum sft, dpo and rl for long cot from scratch and beyond. arXiv preprint arXiv:2503.10460, 2025.

- [34] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Li Erran Li, et al. Deepscaler: Surpassing o1-preview with a 1.5 b model by scaling rl. Notion Blog, 2025.

- [35] Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, Xiangpeng Wei, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025.

- [36] Zhihao Dou, Qinjian Zhao, Zhongwei Wan, Dinggen Zhang, Weida Wang, Towsif Raiyan, Benteng Chen, Qingtao Pan, Yang Ouyang, Zhiqiang Gao, Shufei Zhang, and Sumon Biswas. Plan then action: High-level planning guidance reinforcement learning for llm reasoning. arXiv preprint arXiv:2510.01833, 2025.

- [37] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

- [38] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1.5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.

- [39] Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Minglan Lin, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang. Reason-rft: Reinforcement fine-tuning for visual reasoning. arXiv preprint arXiv:2503.20752, 2025.

- [40] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. arXiv preprint arXiv:2503.12937, 2025.

- [41] Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models. arXiv preprint arXiv:2406.17294, 2024.

- [42] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.

- [43] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. arXiv preprint arXiv:2105.14517, 2021.

- [44] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.

- [45] Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. Dvqa: Understanding data visualizations via question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5648–5656, 2018.

- [46] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–251. Springer, 2016.

- [47] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

- [48] Yifan Du, Zikang Liu, Yifan Li, Wayne Xin Zhao, Yuqi Huo, Bingning Wang, Weipeng Chen, Zheng Liu, Zhongyuan Wang, and Ji-Rong Wen. Virgo: A preliminary exploration on reproducing o1-like mllm. arXiv preprint arXiv:2501.01904, 2025.

- [49] Hui Shen, Taiqiang Wu, Qi Han, Yunta Hsieh, Jizhou Wang, Yuyue Zhang, Yuxin Cheng, Zijian Hao, Yuansheng Ni, Xin Wang, et al. Phyx: Does your model have the" wits" for physical reasoning? arXiv preprint arXiv:2505.15929, 2025.

- [50] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

- [51] Anthropic. Claude 3.5 Sonnet Model Card Addendum. https://huggingface.co/anthropic/ claude-3-sonnet-20240229, 2024. Accessed: 2025-05-01.

- [52] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

- [53] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

- [54] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024.

- [55] Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, and Jifeng Dai. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. arXiv preprint arXiv:2411.10442, 2024.

- [56] Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement. arXiv preprint arXiv:2503.17352, 2025.

- [57] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, Ping Luo, Yu Qiao, Qiaosheng Zhang, and Wenqi Shao. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning, 2025.
- [58] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.

- [59] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024.

- [60] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9556–9567, 2023.

- [61] Trevor Standley, Ruohan Gao, Dawn Chen, Jiajun Wu, and Silvio Savarese. An extensible multi-modal multi-task object dataset with materials. In International Conference on Learning Representations, 2023.

- [62] Jian Hu, Xibin Wu, Zilin Zhu, Xianyu, Weixun Wang, Dehao Zhang, and Yu Cao. Openrlhf: An easy-to-use, scalable and high-performance rlhf framework. arXiv preprint arXiv:2405.11143, 2024.

- [63] Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. arXiv preprint arXiv:2401.06066, 2024.

- [64] Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, Jiachen Liu, Zhongnan Qu, Shen Yan, Yi Zhu, Quanlu Zhang, et al. Efficient large language models: A survey. arXiv preprint arXiv:2312.03863, 2023.

- [65] Hui Shen, Jingxuan Zhang, Boning Xiong, Rui Hu, Shoufa Chen, Zhongwei Wan, Xin Wang, Yu Zhang, Zixuan Gong, Guangyin Bao, et al. Efficient diffusion models: A survey. arXiv preprint arXiv:2502.06805, 2025.

- [66] Yu Zhang, Jialei Zhou, Xinchen Li, Qi Zhang, Zhongwei Wan, Tianyu Wang, Duoqian Miao, Changwei Wang, and Longbing Cao. Enhancing text-to-image diffusion transformer via split-text conditioning. arXiv preprint arXiv:2505.19261, 2025.

- [67] Jing Xiong, Gongye Liu, Lun Huang, Chengyue Wu, Taiqiang Wu, Yao Mu, Yuan Yao, Hui Shen, Zhongwei Wan, Jinfa Huang, et al. Autoregressive models in vision: A survey. arXiv preprint arXiv:2411.05902, 2024.

- [68] Zhongwei Wan, Ziang Wu, Che Liu, Jinfa Huang, Zhihong Zhu, Peng Jin, Longyue Wang, and Li Yuan. Look-m: Look-once optimization in kv cache for efficient multimodal long-context inference. arXiv preprint arXiv:2406.18139, 2024.

- [69] Zhongwei Wan, Xinjian Wu, Yu Zhang, Yi Xin, Chaofan Tao, Zhihong Zhu, Xin Wang, Siqi Luo, Jing Xiong, Longyue Wang, et al. D2o: Dynamic discriminative operations for efficient long-context inference of large language models. arXiv preprint arXiv:2406.13035, 2024.

- [70] Jing Xiong, Jianghan Shen, Fanghua Ye, Chaofan Tao, Zhongwei Wan, Jianqiao Lu, Xun Wu, Chuanyang Zheng, Zhijiang Guo, Lingpeng Kong, et al. Uncomp: Uncertainty-aware long-context compressor for efficient large language model inference. arXiv preprint arXiv:2410.03090, 2024.

- [71] Jing Xiong, Qiujiang Chen, Fanghua Ye, Zhongwei Wan, Chuanyang Zheng, Chenyang Zhao, Hui Shen, Alexander Hanbo Li, Chaofan Tao, Haochen Tan, et al. A1: Asynchronous test-time scaling via conformal prediction. arXiv preprint arXiv:2509.15148, 2025.

- [72] Jing Xiong, Jianghan Shen, Chuanyang Zheng, Zhongwei Wan, Chenyang Zhao, Chiwun Yang, Fanghua Ye, Hongxia Yang, Lingpeng Kong, and Ngai Wong. Parallelcomp: Parallel long-context compressor for length extrapolation. arXiv preprint arXiv:2502.14317, 2025.

- [73] Zhongwei Wan, Hui Shen, Xin Wang, Che Liu, Zheda Mai, and Mi Zhang. Meda: Dynamic kv cache allocation for efficient multimodal long-context inference. arXiv preprint arXiv:2502.17599, 2025.

- [74] Xin Wang, Yu Zheng, Zhongwei Wan, and Mi Zhang. Svd-llm: Truncation-aware singular value decomposition for large language model compression. arXiv preprint arXiv:2403.07378, 2024.

- [75] Dong Liu, Yanxuan Yu, Yite Wang, Jing Wu, Zhongwei Wan, Sina Alinejad, Benjamin Lengerich, and Ying Nian Wu. Designing large foundation models for efficient training and inference: A survey. arXiv preprint arXiv:2409.01990, 2024.

- [76] Zixuan Li, Jing Xiong, Fanghua Ye, Chuanyang Zheng, Xun Wu, Jianqiao Lu, Zhongwei Wan, Xiaodan Liang, Chengming Li, Zhenan Sun, et al. Uncertaintyrag: Span-level uncertainty enhanced long-context modeling for retrieval-augmented generation. arXiv preprint arXiv:2410.02719, 2024.

- [77] Hui Shen, Zhongwei Wan, Xin Wang, and Mi Zhang. Famba-v: Fast vision mamba with cross-layer token fusion. In European Conference on Computer Vision, pages 268–278. Springer, 2024.

- [78] Yizhi Li, Qingshui Gu, Zhoufutu Wen, Ziniu Li, Tianshun Xing, Shuyue Guo, Tianyu Zheng, Xin Zhou, Xingwei Qu, Wangchunshu Zhou, et al. Treepo: Bridging the gap of policy optimization and efficacy and inference efficiency with heuristic tree-based modeling. arXiv preprint arXiv:2508.17445, 2025.

- [79] Alex ZH Dou, Zhongwei Wan, Dongfei Cui, Xin Wang, Jing Xiong, Haokun Lin, Chaofan Tao, Shen Yan, and Mi Zhang. Enhancing test-time scaling of large language models with hierarchical retrieval-augmented mcts. arXiv preprint arXiv:2507.05557, 2025.

- [80] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

### A Limitation and Social Impacts

#### A.1 Limitation

Our experiments primarily evaluated the effectiveness of SRPO on dense MLLMs at 7B and 32B scales, without conducting scaling experiments on Mixture-of-Experts (MoE) [63, 64] or diffusion LM [65–67] architectures. Additionally, the reinforcement learning training data utilized in our experiments were selected exclusively from publicly available multimodal reasoning datasets, without exploration of larger-scale commercial reasoning datasets. Extending our method to MoE-based models, incorporating larger-scale RL training datasets, and combining with efficient inference [68–77] or other decoding settings [78, 79] remain promising avenues for future work, potentially leading to even broader improvements in multimodal reasoning capabilities.

#### A.2 Social Impacts

Our work offers clear positive contributions by significantly enhancing MLLMs’ capabilities in complex reasoning tasks. These advances can enable more accurate and reliable AI assistance in education, scientific discovery, and decision-making scenarios, ultimately contributing to broader accessibility of high-quality reasoning support. However, improved reasoning capabilities also carry potential risks, such as generating

more convincing yet inaccurate or biased content if models reflect on misleading data. Consequently, careful curation of training datasets and explicit efforts in mitigating potential biases remain essential. Additionally, deploying advanced multimodal reasoning models without adequate safeguards might inadvertently reinforce existing societal inequalities if access to these advanced technologies remains restricted. Overall, responsible and transparent use of these enhanced multimodal reasoning frameworks is crucial to ensure positive societal outcomes.

### B Appendix

In this appendix, we provide supplementary details and extended analyses supporting the main findings presented in our paper. Specifically, we describe the datasets employed for self-reflection SFT and RL training (§B.2), present detailed hyper-parameter settings used throughout our experiments (§B.3), and provide explicit prompt templates for self-reflection SFT, RL training, and evaluation (§B.4). Additionally, we illustrate training dynamics, including convergence trends and key metrics during the reinforcement learning phase (§B.6). Finally, we offer qualitative analyses through representative samples generated during RL training and inference, demonstrating the effectiveness of our reflection-based strategies (§B.5).

#### B.1 Further Experiment

Sensitivity to the Reflection Length Reward Coefficient α. To evaluate the robustness of SRPO to the reflection-length reward coefficient, we additionally trained models with α = 0.05 and α = 0.3, using the same setup as in the main experiments. The results are shown in Table 6. All experiments were performed under identical optimization and data settings, with only the length-reward coefficient varied. Results indicate that α = 0.05 performs similarly to α = 0.1, while α = 0.3 slightly degrades accuracy. This suggests that over-emphasizing length reward weakens the reflection signal, whereas values near 0.1 offer a balanced trade-off, helping to control response length and stabilize training with limited impact on accuracy.

Table 6 Sensitivity of SRPO to the reflection length reward coefficient α.

Ratio MathVerse MMMU-Pro Physics Avg.

α=0.05 55.7 41.9 61.1 52.9 α=0.1 (Ours) 55.8 42.3 60.6 53.5 α=0.3 55.1 42.0 59.7 52.2

Qualitative & quantitative reflection analysis. To quantitatively assess reflection effectiveness, we conducted both human and LLM-based evaluations on SRPO-7B’s generated reflections.

- a. Human Expert Evaluation. We randomly sampled 100 MathVista test questions (answered by SRPO-7B). Two senior PhD students specializing in NLP and LLMs independently rated each reflection on a 0–3 scale: 3

= highly effective, 2 = partially effective, 1 = redundant, and 0 = detrimental. We also measured the Wrong Answer Fix Rate, i.e., how often an initially incorrect answer was corrected after reflection. The results are summarized in Table 7.

Table 7 Human Expert Evaluation of Reflection Quality.

Metric Human Expert 1 Human Expert 2

Effective Reflection Rate (score ≥ 2) 73% 69% Redundancy Rate (score = 1) 9% 11% Detrimental Rate (score = 0) 3% 1% Wrong Answer Fix Rate 39% 39%

###### Seventy percent of reflections were judged to be effective. Among 100 questions, the initial solution was incorrect in 33 cases, of which 13 were corrected after reflection, resulting in a 39% wrong-answer fix rate.

- b. LLM-as-a-Judge (GPT-4o) Evaluation. Each reflection was further scored by GPT-4o on four 0–5 dimensions—logical flaws, missing assumptions, clarity, and actionable suggestions—following the prompt format described in Appendix B.3. The average reflection quality is shown in Table 8.

Table 8 LLM-as-a-Judge Evaluation (GPT-4o).

Model Logic Missing Clarity Suggestions Avg. Quality SRPO-7B (Ours) 4.1 3.9 4.0 3.6 3.9

These results confirm that SRPO produces high-quality reflections from both human and LLM perspectives, validating the effectiveness of its self-reflection design.

- 102
- 103
- 104

|143|33| | | | |
|---|---|---|---|---|---|
| | | | | | |
| |102|27| | | |
| | | | | | |
| | |59|53| | |
| | | | |35|34|
| | | |33|43| |
| | | | | | |

|70|25<br><br>17|09| | | | |
|---|---|---|---|---|---|---|
| | |41|3<br><br>23|2<br><br>9|1| |
| | | | | |4|9|

14000

12000

###### NumberofSamples

###### NumberofSamples

10000

8000

6000

4000

2000

Mathematical Reasoning

Physical Reasoning LogicalPuzzle Reasoning

General Reasoning NaturalScene Reasoning

Chemical Reasoning

0

OtherMathGeometricMathChartsDiagramsNaturalScienceGeneralReasoning

(a) Self-reflection SFT data statistic

(b) RL training data statistic Figure 6 Statistics of reasoning-type distribution in different stages of training.

Training and Inference Efficiency

We report the measured wall-clock time for each SRPO training stage under both 7B and 32B settings using the OpenRLHF framework. All settings share the same data size and optimization schedule as in the main experiments; only the model size and stage differ. Table 9 summarizes training efficiency, and Table 10 summarizes inference efficiency on the MathVista test-mini split ( 1k samples) using a single H100-80G GPU with vLLM. SRPO requires only modest additional wall-time compared with GRPO, while delivering substantially better reasoning quality in the main results.

Table 9 Training efficiency overview. Wall-clock time for each stage under 7B and 32B settings.

Training Stage GPUs Wall-Time Notes

Reflection-SFT (7B) 8×H100 3.5 hours 10k samples, 1 epoch Reflection-RL (7B, SRPO) 8×H100 31.2 hours 37k samples, 500 steps GRPO-RL (7B) 8×H100 25.8 hours 37k samples, 500 steps Reflection-SFT (32B) 4 × 8×H100 4.7 hours 10k samples, 1 epoch Reflection-RL (32B, SRPO) 4 × 8×H100 45.1 hours 37k samples, 500 steps

- B.2 Training Dataset Self-reflection SFT Dataset. We primarily select samples from three established multimodal reasoning datasets:

- • Mulberry Dataset (260K) [24]: A multimodal reasoning dataset enriched through collective Monte Carlo tree search, specifically designed to enhance reflection and reasoning capabilities of multimodal LLMs. It features diverse reasoning problems requiring explicit cognitive processes.

Table 10 Inference efficiency overview. Latency on MathVista test-mini using a single H100-80G with vLLM.

Inference Stage GPUs Wall-Time Notes

GRPO-RL (7B) 1×H100 45.6 min 1k test samples, avg response = 355.4 tokens SRPO (7B) 1×H100 60.5 min 1k test samples, avg response = 502.3 tokens

- • MathV360K [41]: The dataset focuses on mathematical reasoning for multimodal LLMs. It systematically bootstraps multimodal reasoning by constructing high-quality mathematical reasoning prompts paired with visual contexts.
- • LLaVA-CoT Dataset (100K) [25]: A vision-language reasoning dataset explicitly designed for chain-of-thought (CoT) prompting. It consists of multimodal problems that encourage step-by-step logical reasoning aligned with visual inputs.

From these datasets, we randomly sample 100K data points. For each sampled multimodal problem, we apply a specialized CoT-generation template, feeding both the visual inputs and associated questions to two pretrained models: Qwen-2.5-VL-7B-Instruct and Qwen-2.5-VL-32B-Instruct [28], respectively, to generate corresponding reasoning steps. Subsequently, we use the DeepSeek-V3 [80] API to assess the quality of these generated reasoning paths, selecting a high-quality subset of 10K samples containing approximately 30% correctly solved and 70% incorrectly solved reasoning paths.

Next, leveraging these selected CoT samples and their associated ground truths, we utilize GPT-o4-mini [7] with a specifically designed self-reflection generation prompt. This approach yields concise, meaningful self-reflective feedback regarding the generated reasoning steps. Finally, we structure these samples into our self-reflection enhanced SFT dataset following the proposed Self-reflection SFT template.

Self-reflection RL Dataset. We curate our reinforcement learning (RL) dataset by selectively sampling from several multimodal reasoning benchmarks, each featuring distinct reasoning characteristics and data modalities. These datasets include:

- • ScienceQA [42]: Contains 21K multimodal science questions, encouraging explicit reasoning chains through visual contexts, textual explanations, and multiple-choice tasks.
- • Geometric Math QA (GeoQA) [43]: Consists of 5K geometry-focused multimodal problems designed to evaluate numerical reasoning over geometric concepts.
- • ChartQA [44]: Provides 9.6K questions for visual and logical reasoning based on diverse chart types such as bar, line, and pie charts.
- • DVQA [45]: Includes 3.5K questions requiring comprehensive reasoning to interpret data visualizations effectively.
- • AI2D [46]: Features 5K diagram-based science questions aimed at evaluating visual and conceptual understanding through diagrams.
- • MATH [47]: Comprises 12.5K challenging mathematical problems across various difficulty levels, extensively utilized to measure mathematical reasoning capabilities.
- • Virgo [48]: Offers around 10K multimodal reasoning examples intended to emulate the reasoning complexity of state-of-the-art models like OpenAI O1.
- • R1-OneVision [11]: Contains 5K multimodal reasoning instances specifically designed for cross-modal reasoning and generalization.
- • MMK12 [8]: Provides a curated set of around 12K multimodal problems that involve rigorous rule-based reasoning across diverse domains.
- • PhyX [49]: Comprises 3K carefully designed multimodal physics questions spanning thermodynamics, electromagnetism, mechanics, modern physics, optics, and acoustics, aimed at evaluating physical reasoning

capabilities in realistic visual contexts.

By systematically combining and sampling from these datasets, we obtain a comprehensive self-reflection RL dataset containing high-quality multimodal reasoning instances for our experiments.

#### B.3 Hyper-parameters

In the reinforcement learning phase of SRPO, we highlight several critical hyper-parameters: we set both the rollout and training batch sizes to 128, generating 8 samples per prompt to ensure response diversity. Sampling is performed with a temperature of 1.0. The learning rate is fixed at 1 × 10−6 using the Adam optimizer with parameter offloading enabled, and training is conducted using bf16 mixed-precision. We adopt group-normalized advantage estimation (specific to GRPO) to stabilize training and utilize the "k3" KL divergence estimator for controlled policy updates. Additionally, we freeze the visual encoder parameters during training, enable gradient checkpointing and flash attention for memory efficiency, and perform accuracy filtering (retaining samples with accuracy scores between 0.1 and 0.9) to maintain data quality throughout the RL process.

#### B.4 Prompt Template Prompt Template for CoT Generation System: "You are a precise AI assistant and must strictly follow the following rules:

- • First reason step-by-step, and wrap the thought process in <think> tags.
- • The final answer must be wrapped in <answer> tags.
- • Formatting requirements:

- – Choice answers must be uppercase letters (A/B/C/D).
- – Fill-in-the-blank answers should be digits.

- • DO NOT EXPLAIN ANYTHING IN <answer>.
- • You must provide both <think> and <answer>.
- • Please strictly follow the formatting requirements and do not add any extra content!

User: [type: "text", content: question,

type: "image_url", image_url: image_url]

##### Prompt Template for Self-Reflection Generation

System: "You are a helpful math reasoning assistant. Think carefully. Output only JSON." User:

You are an expert visual reasoning assistant. Your task is to reflect on the quality of a chain-of-thought (CoT) reasoning given for a visual question. The goal is to improve the CoT by identifying weaknesses and offering suggestions for refinement.

Please follow this structure strictly:

1. "reflection": Provide a detailed critique of the original CoT, pointing out:

- • Logical flaws or inconsistencies
- • Missing assumptions or information
- • Any correct reasoning that could be made clearer or more robust
- • Suggestions for improving the reasoning process

Only return a valid JSON object with a "reflection" field.

–- Input –Question: {query} Image: {image_url} Original Chain of Thought: {cot} Predicted Answer (Based on CoT): {answer} Correct Answer (ground truth): {ground truth}

##### Prompt Template for Self-reflection SFT

System: You are a reasoning expert. Given an image and a question, please generate two rounds of step-by-step reasoning:

- • First, provide your initial chain of thought and answer.
- • Then reflect on it.
- • Finally, based on your reflection, give your final reasoning and answer.

User:

{

"messages": [ {

"role": "system", "content": "You are a reasoning expert. Given an image and a question, please generate two rounds of step-by-step reasoning: First, provide your initial chain of thought and answer. Then reflect on it, and finally, based on your reflection, give your final reasoning and answer."

}, {

"role": "user", "content": "Question: <question>{query}</question>\nImage: <image>"

}, {

"role": "assistant", "content": "<think>{cot}</think>\n<answer>{answer}</answer> \n<reflection>{reflection}</reflection> \n<answer>{ground_truth}</answer>"

}

], "images": ["{image_url}"]

}

##### Prompt Template for Self-reflection RL

System: Solve the user’s question step by step. First, think about the reasoning process internally and write it inside <think>...</think> tags. Then, provide the first answer in LaTeX format wrapped with $...$, and the final result must be enclosed with \boxed{}. Wrap this answer inside <answer>...</answer> tags. After that, perform a critical self-reflection on the previous reasoning and answer, writing the reflection inside <reflection>...</reflection> tags. Then, based on the reflection, generate a new reasoning process and a new answer:

- • The new reasoning is again placed inside <think>...</think>.
- • The new answer is written inside <answer>...</answer> and uses LaTeX $...$ with \boxed{} for the final output.

Make sure both reasoning steps are clear and detailed. Even if the final answer does not change, the second reasoning must incorporate improvements based on the reflection.

Format Example:

<think> Since $1+1=2$, so the answer is $2$. </think> <answer> The answer is $\boxed{2}$. </answer> <reflection> The reasoning is correct but too brief; I could have explained the addition more explicitly. </reflection> <think> Adding $1$ and $1$ together results in $2$ because $1$ plus $1$ means taking one and adding another one, leading to $2$. </think> <answer> The answer is $\boxed{2}$. </answer>

All reasoning, answer, and reflection steps must be included without omission and in the exact sequence: <think>...</think> <answer>...</answer> <reflection>...</reflection> <think>...</think> <answer>...</answer>.

#### B.5 Generated Samples Analysis

In this section, we present generated samples during RL training (Sample 1 and Sample 2) and from real test cases (Sample 3 and Sample 4). In the training examples (Sample 1 and 2), we observe that Qwen-2.5-VL-7B, under our proposed SRPO framework, exhibits the ability to perform self-reflection and subsequently revise its reasoning steps. Similarly, in downstream evaluation tasks, the model demonstrates reflective adjustments when answering incorrectly, and when answering correctly, it further refines and simplifies its reasoning through self-reflection.

Question: Given that the measure of angle 1 is 123 degrees in the figure, what is the measure of angle 11?

[Figure 32]

Choices: (A) 57 (B) 67 (C) 113 (D) 123 Ground truth: (D) 123

<think> Since angle 1 and angle 11 are corresponding angles formed by a transversal intersecting two parallel lines, they are congruent. Therefore, the measure of angle 11 is the same as the measure of angle 1. </think> <Answer> The answer is

|57|
|---|

(Incorrect due to flawed assumptions)

</Answer> <reflection>

The original reasoning has a significant flaw: it incorrectly identifies angle 1 and angle 11 as corresponding angles formed by a transversal intersecting parallel lines. This assumption is unfounded given the image description, which does not specify that the lines are parallel. Moreover, the reasoning does not consider the possibility that angle 11 might be obtained from a different relationship, such as a vertical angle or an alternate interior angle formed by a non-parallel configuration. To improve the reasoning, one should: 1) analyze the given visual elements more carefully rather than relying on assumptions about parallel lines; 2) re-evaluate the geometric relationships present in the figure, including the possibility of vertical angles or other angle pairs that could be equal; and 3) include intermediate steps or calculations that confirm why the measure of angle 11 should be different from angle 1. Clarifying these points would lead to a more robust and accurate conclusion.

</reflection> <think> (Generate the right answer after effective self-reflection)

The correct approach here would be to notice that if angle 1 and angle 11 were somehow related in a way that the sum of their measures was a specific constant (for example, if they were supplementary or complementary in context), we could deduce the measure of angle 11 by subtracting from 180 degrees (since supplementary angles add up to 180 degrees). But since the problem does not specify this relationship, and considering the answer choices, the most logical deduction is that angle 11 is the same as angle 1 due to a specific geometric configuration not detailed in the problem statement.Thus, the measure of angle 11 is 123◦.

</think> <Answer> The answer is </Answer>

|123|
|---|

Question: Based on the upper image, what task is the robotic arm most likely performing?

[Figure 33]

Options: (A) Sorting playing cards based on a predefined algorithm.

- (B) Performing a magic trick with the cards.
- (C) Playing a card game against a human opponent.
- (D) Conducting an experiment on object recognition. Ground truth: (D) <think> The image shows a robotic arm with a camera mounted on it, which suggests that it is involved in a task that requires visual input and processing. The presence of playing cards on the table and the arm’s position above them indicate that the robot might be interacting with the cards in some way. Given the context of the setup, the most logical task for the robotic arm would be to sort the playing cards based on a predefined algorithm, as this is a common application for robotic arms equipped with cameras for visual recognition. </think>

|A|
|---|

. (Incorrect)

<answer> The answer is

</answer> <reflection>

The original reasoning is overall correct in observing the camera and the cards, but it overlooks deeper experimental context. The red and yellow point clouds and structured laser-like patterns shown in the lower part of the image are indicative of 3D perception or object localization, which are typical in object recognition experiments. Additionally, there’s no evidence of sorting mechanisms or card repositioning. The reflection should thus account for the experimental nature of the setup, making option D more consistent with the evidence provided.

</reflection> <think> (Generate the right answer after effective self-reflection)

Let’s rethink, the image shows a robotic arm with a camera, indicating a task that involves visual data collection or processing. The presence of playing cards on the table suggests an interaction with objects, and the alignment of the dots and lines might represent data points collected for object recognition or spatial analysis. Given these observations, the most likely task for the robotic arm is related to object recognition, where the camera would be used to identify and analyze the cards’ features.

</think> <answer>

|D|
|---|

.

The answer is

</answer>

Question: A farmer wants to build a fence around the entire perimeter of his land, as shown in the diagram. The fencing costs £37 per metre. The lengths of EF and CD are x metres and y metres respectively. At £37 per metre, how much will it cost him to build the fence along the entire perimeter? Ground truth: 777

[Figure 34]

<think>

To determine the cost of building the fence around the entire perimeter of the land, we first need to find the total perimeter of the land. The perimeter is the sum of the lengths of all the sides of the shapes forming the land. Given: - The height of the rectangle ABGF is 5 m. - The width of the rectangle ABGF is 1 m. - The height of the rectangle BCDE is 7 m. - The width of the rectangle BCDE is 2 m. - The length of EF is x m. - The length of CD is y m. The perimeter of the land is the sum of the lengths of all the sides:

Perimeter = AB + BG + GE + EF + FD + DC + CE + EB From the diagram:

Perimeter = 5 + 1 + 2 + x + 1 + 2 + 3 + 1 = 15 + x Fencing cost is:

|555 + 37x|
|---|

</think> <answer>

|555 + 37x|
|---|

. (Incorrect)

The answer is

</answer> <reflection>

The initial reasoning misidentified the structure of the perimeter by incorrectly including internal segments such as BG, GE, and EB. These are auxiliary construction lines used solely to form right triangles for computing the diagonal lengths EF and CD, not actual parts of the outer boundary of the land that require fencing. This led to an overestimation of the total perimeter. To correct the reasoning, one must focus strictly on the visual geometry and trace only the external contour of the land—specifically, the true fencing path comprises segments AB, AF, EF, ED, DC, and CA. The diagonal lengths EF and CD should be computed using the Pythagorean theorem based on right triangle constructions derived from the diagram. This corrected understanding yields the true perimeter and thus the accurate total fencing cost. </reflection> <think> (Generate the right answer after effective self-reflection) Tracing only the external boundary gives segments

AB = 5, AF = 1, FE = x, ED = 2, DC = y, CB = 2. Therefore the total perimeter is

5 + 1 + x + 2 + y + 2 = 10 + (x + y).

From the diagram’s length proportions (and the single “11 m” label spanning the two slanted edges), we see directly that

x + y = 11. Hence the full perimeter is 10 + 11 = 21 m, and the fencing cost is

|777|
|---|

37 × 21 =

.

25

</think> <answer>

|777|
|---|

The answer is

.

</answer>

[Figure 35]

Question: M, W, X, Y, and Z are main-group elements in the same period. The number of valence electrons in X is three times the number of electrons in the penultimate shell of W. The element R is a key player in inorganic non-metal materials. The structure of the compound formed by M, W, X, Y, and Z is shown. Which of the following statements is incorrect?

Choices:

- • A. Elemental M is usually stored under kerosene.
- • B. The oxide of R dissolves in aqueous solution of Y’s hydride.
- • C. Every atom in the anion shown satisfies an octet.
- • D. The boiling points of the simplest hydrides of X, Y, Z follow X > Y > Z.

<think>

Determine the period: all elements are in period 3. W’s penultimate shell contains 8 electrons, so X has 3×8 = 24 valence electrons; therefore, X = Cl. Using known oxidation states: M = Na, W = Mg, Y = Al, Z = P. R is most likely Si. Now evaluate the options: A. Sodium is not stored under kerosene but under oil; therefore, this statement is incorrect.

|A|
|---|

B, C, and D are chemically consistent. Therefore, the incorrect statement is

.

</think>

<answer>

|A|
|---|

The answer is

.

</answer>

<reflection> Reflect on the redundancy of the prior reasoning and generate a simplified version.

The initial reasoning is correct but can be simplified. Instead of step-by-step element identification, we can streamline the logic into three clear steps: (1) From W’s inner shell (8 electrons), deduce that X has 24 valence electrons, hence X = Cl.

(2) Identify M = Na, W = Mg, Y = Al, Z = P from standard main-group patterns. (3) Eliminate each option: only A fails (Na is not stored under kerosene). This improved structure yields the same answer more directly.

</reflection>

<think>

From periodic positions and common valences: M = Na, W = Mg, Y = Al, Z = P. Only statement A is chemically incorrect. Answer:

|A|
|---|

.

</think>

<answer>

|A|
|---|

The answer is

.

</answer>

#### B.6 Training Dynamics

We provide additional visualizations of RL training curves, including total response length, correct and incorrect response lengths, ratio clip lower, policy loss, and accuracy reward. Several key observations can be made: First, due to the explicit emphasis on self-reflection during training, SRPO consistently generates longer total responses and exhibits notably greater growth in response length compared to GRPO and SRPO without self-reflection. This is attributable to the model’s active engagement in self-reflection and subsequent correction of prior reasoning steps. Additionally, SRPO consistently achieves higher accuracy reward values than baselines, confirming that reinforcement of reflective reasoning effectively enhances the model’s reasoning

capabilities. Furthermore, from the ratio clip lower and policy loss curves, we observe that SRPO—whether employing self-reflection in both SFT and RL phases or solely in the RL phase—maintains stable clip lower values consistently below 0.005. This indicates that the integration of self-reflection contributes to stable policy updates with moderate gradient adjustments throughout training.

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Figure 7 More training dynamics of SRPO, GRPO, and SRPO w/o Self-reflection SFT.

