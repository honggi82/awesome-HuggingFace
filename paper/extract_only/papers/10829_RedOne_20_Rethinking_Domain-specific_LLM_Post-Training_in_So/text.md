# arXiv:2511.07070v1[cs.AI]10Nov2025

## RedOne 2.0: Rethinking Domain-specific LLM Post-Training in Social Networking Services

Fei Zhao, Chonggang Lu, Haofu Qian, Fangcheng Shi, Zijie Meng, Jianzhao Huang, Xu Tang, Zheyong Xie, Zheyu Ye, Zhe Xu, Yao Hu, Shaosheng Cao∗

NLP Team, Xiaohongshu Inc. Huangpu District, Shanghai, China caoshaosheng@xiaohongshu.com

##### Abstract

As a key medium for human interaction and information exchange, social networking services (SNS) pose unique challenges for large language models (LLMs): heterogeneous workloads, fast-shifting norms and slang, and multilingual, culturally diverse corpora that induce sharp distribution shift. Supervised fine-tuning (SFT) can specializemodels butoftentriggersa “seesaw” between in-distribution gains and out-of-distribution robustness, especially for smaller models. To address these challenges, we introduce RedOne 2.0, an SNS-oriented LLM trained with a progressive, RL-prioritized post-training paradigm designed for rapid and stable adaptation. The pipeline consist in three stages: (1) Exploratory Learning on curated SNS corpora to establish initial alignment and identify systematic weaknesses; (2) Targeted Fine-Tuning that selectively applies SFT to the diagnosed gaps while mixing a small fraction of general data to mitigate forgetting; and (3) Refinement Learning that re-applies RL with SNS-centric signals to consolidate improvements and harmonize trade-offs across tasks. Across various tasks spanning three categories, our 4B scale model delivers an average improvements about 2.41 over the 7B sub-optimal baseline. Additionally, RedOne 2.0 achieves average performance lift about 8.74 from the base model with less than half the data required by SFT-centric method RedOne, evidencing superior data efficiency and stability at compact scales. Overall, RedOne 2.0 establishes a competitive, cost-effective baseline for domain-specific LLMs in SNS scenario, advancing capability without sacrificing robustness.

##### CCS Concepts

• Human-centered computing → Social networks; • Computing methodologies → Learning paradigms; Multi-task learning.

##### Keywords

Large language model, Post training, Social networking services

∗Corresponding author.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

Conference acronym ’XX, Woodstock, NY © 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/2018/06 https://doi.org/XXXXXXX.XXXXXXX

###### ACM Reference Format:

Fei Zhao, Chonggang Lu, Haofu Qian, Fangcheng Shi, Zijie Meng, Jianzhao Huang,, Xu Tang, Zheyong Xie, Zheyu Ye, Zhe Xu, Yao Hu, Shaosheng Cao. 2018. RedOne 2.0: Rethinking Domain-specific LLM Post-Training in Social Networking Services. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 10 pages. https://doi.org/XXXXXXX.XXXXXXX

##### 1 Introduction

Powered by the rapid advance of electronic devices and network infrastructure, social networking services (SNS) have evolved into core infrastructure for human daily interaction and the spread of information [71]. In parallel, large language models (LLMs) have delivered striking progress in natural language processing (NLP), enabling promising performance across a wide range of downstream tasks [32, 73]. However, deploying a general-purpose LLM into an SNS scenario is not a simple lift-and-shift. The workloads of SNS platform are highly heterogeneous, such as real-time moderation and abuse response, recommendation-shaped dialogue, creator assistance, and community operations, each with distinct latency, safety, and tone requirements. And the environment changes in a very fast speed, where trends, slang, and community norms rise and fade within days. Additionally, because SNS connects vast, diverse audiences, models must handle large, complex corpora across languages and expression styles to bridge users from different cultural backgrounds. Together these factors amplify distribution shift and increase the risk of brittle generalization, where models, optimized on the standard benchmarks, may misread communityspecific rules, over- or under-enforce policies, or drift as conventions change [79, 83]. Therefore, seamlessly integrating LLMs for SNS scenario requires that adapt quickly, remain stable, and maintain competence across languages and communities, especially without compromising safety or user trust.

As a pioneer in adapting LLM in SNS, RedOne [83] established an early domain-specific post-training route applying a classical SFTbased pipeline and showed that SNS performance can be increased without sacrificed general ability. However, this SFT-dominated pipelines exhibit a pronounced “seesaw” effect, which improves on in-distribution (ID) tasks often come at the expense of out-ofdistribution (OOD) generalization. And the problem is especially acute for smaller models, which are more susceptible to catastrophic forgetting as new domain patterns overwrite previously learned skills. General mitigations try to smooth the issue by scaling huger training sets, crafting elaborate data mixture schedules, and designing diverse learning objectives [29, 42, 43, 59]. While these methods can somehow recover models’ robustness on OOD dataset, they lift

[Figure 1]

###### Figure 1: The comparison of various scale models’ performance in the SNS domain.

up the budgets of data and compute to a sky-high level. In contrast, reinforcement learning (RL) offers a more distinctive advantage for domain specific post-training [36]. By optimizing against preference or reward signals, RL directly aligns model behavior with human and downstream objectives, preserving existing competencies while unlocking latent capabilities. As RL has matured in LLM optimization, its practical value has been demonstrated across preference alignment, safety shaping, controllable generation and task-level policy tuning [24, 56, 76]. However, how to structure RL-centric training to balance data efficiency, stability, and domain transfer in the fast-evolving SNS environment remains underexplored.

To address this gap, we introduces RedOne 2.0, a new SNSoriented LLM that adopts a progressive, RL-prioritized post-training paradigm. Our design philosophy is that exploration targeted correction refinement yields better stability generalization trade-offs than SFT-heavy recipes, especially at smaller parameter scales and with limited domain data, as shown in Fig. 1. Concretely, RedOne

- 2.0 is trained in three stages: 1) Exploratory Learning. The model is exposed to curated SNS corpora to establish initial domain alignment and to diagnose the lack of ability for realistic distributions.

- 2) Targeted Fine-Tuning. We apply SFT on tasks where previous stage diagnostics reveal systematic weaknesses, blending a small fraction of general data to explicitly regularize against forgetting and retain broad generalization. 3) Refinement Learning. Building on the corrected model, we reuse RL with SNS-centric signals to consolidate improvements and smooth trade-offs across different tasks, yielding the coherent capability gains.

We also conducted extensive experiments across various tasks spanning three categories to validate our post-training pipeline. The 4B-parameter variant surpasses the 7B counterpart by 2.41 on average, demonstrating that strong performance is attainable at compact scales. Using Qwen3-4B as the base, RedOne 2.0 requires only half of RedOne’s data while achieving a performance lift with

8.74, which demonstrates superior data efficiency and broader capability gains from RL-centric curriculum learning.

Our contribution can be summarized as follows:

- • We present RedOne 2.0, an SNS-domain LLM that achieves higher capability with less data and smaller models.
- • We propose a progressive, RL-prioritized post-training paradigm that delivers consistent improvements in both general and SNS-specific abilities while mitigating the “seesaw” effect of SFT.
- • We provide comprehensive empirical validation showing state-of-the-art results in the SNS scenario and strong robustness under distribution shift, establishing RedOne 2.0 as a competitive and cost-effective baseline for domain LLMs.

2 Related Work

- 2.1 LLM in Social Networking Services

Given the central role of SNS platforms in everyday information exchange, interest in this domain has surged [3, 8, 67]. Advances in LLMs have accelerated integration across sentiment analysis [68, 81], search and question answering over social content [37, 61], personalized content generation [50, 51], content moderation [41, 44], and platform operations [17, 57]. Moving beyond task-specific pipelines thatunderuse modelknowledge and generalization, recent work explores unified solutions. Social-LLM combines localized social interaction signals with text features to scale inductive user detection across seven real-world datasets [34]. [79] organizes LLMfor-SNS applications into knowledge, engagement, and foundation tasks while outlining deployment challenges. RedOne introduces a large-scale SNS dataset and a common domain post-training recipe that yields strong offline gains and measurable online improvements [83]. Yet most approaches remain data-centric, expanding domain competence primarily through large annotated corpora and dependence on strong base models, which drives up cost. RedOne 2.0 revisits post-training in the SNS setting and achieves stronger downstream performance with substantially less data and smaller model scales.

- 2.2 General LLM Post-training

Post-training bridges large-scale pre-training and deployment by enhancing instruction following, safety, and factuality. Typical pipelines first perform SFT on curated instruction–response pairs, then apply preference optimization with human or automated feedback, as exemplified by InstructGPT [56]. Subsequent advances such as RRHF [77] and DPO [58] simplify preference learning and improve training stability, while GRPO [62] and DAPO [76] introduce more efficient, reward-driven reinforcement learning frameworks that better balance exploration and alignment quality. Despite these developments, most approaches remain domain-agnostic and tend to underutilize specialized knowledge crucial for vertical applications.

- 2.3 Domain-specific LLM Post-training

Domain adaptation injects targeted knowledge and preferences to boost in-domain performance. Typical pipelines combine continued pre-training on domain corpora with supervised fine-tuning and

###### Reward Function

### General

SNS Data

Data

Title: Here is an exam… Tag: education

#### Data

### Collection

Exact Match

Mixed Data

Mixed Data

General Data

SNS Data

Sand Box

Generation

Reward Function

Reward Function

judge

Metric-based

[Figure 2]

[Figure 3]

DAPO

SFT DAPO

Pattern

Targeted Fine-Tuning Refinement Learning

### Exploratory Learning

Figure 2: Overview of our RL-based incremental three-stage training pipeline.

preference optimization on domain tasks, yielding strong results in finance [9, 22, 70, 75], law [13, 15, 23, 38], medicine [35, 53, 63, 72], and the sciences [6, 7, 65, 80]. However, heavy reliance on SFT can overfit to benchmarks and weaken generalization, while later reinforcement learning often only partly fixes this drift. RedOne 2.0 addresses these limits with an RL-centric design that emphasizes staged reinforcement and dynamic task sampling to improve both general competence and domain reasoning.

##### 3 Methodology for RedOne 2.0

As illustrated in Fig. 2, the post-training pipeline of RedOne 2.0 includes three stages, and each stage builds on the previous one, progressively aligning the model to the SNS domain while preserving general capabilities and ensuring stability. First, during Exploratory Learning, we initially align the base model to the SNS domain and identify SNS-specific tasks where the model requires targeted reinforcement. Next, in Targeted Fine-Tuning, we address the deficiencies revealed in the previous stage to improve model’s performance in SNS scenario and create exploration space for subsequent optimization. Finally, in Refinement Learning, we apply RL on SNS-domain data to stabilize and smooth model behavior, driving better performance across various tasks.

##### 3.1 Dataset Definition

To ground the three-stage pipeline, we first construct and characterize the training dataset. We curate large-scale data from both the SNS domain and the general domain. The former covers capabilities commonly evaluated in SNS scenarios, including information extraction, semantic matching, content understanding, user behavior modeling, dialogue, and translation, and spans more than 75 carefully defined tasks. This provides sufficient supervision for following training pipeline and establishes a solid data foundation for adapting RedOne 2.0 to a wide range of SNS real-world applications. The latter integrates high-quality open-source datasets that have been widely validated by the community, thereby reducing redundant annotation and processing costs while ensuring robust general capabilities of RedOne 2.0.

We then normalize all collected data into a unified format of question 𝑄 and answer 𝐴, yielding the final dataset

###### D = { DSNS ∪ DGEN | (𝑄,𝐴) } (1)

where DSNS and DGEN denote the SNS-domain and general-domain subsets, respectively.

##### 3.2 Exploratory Learning

The goal of Exploratory Learning is to achieve initial alignment between the base model and SNS-domain characteristics. Instead of committing early to narrowly scoped objectives, this stage immerses the model in diverse SNS data to capture the breadth of task distributions and domain-specific interaction patterns, while preserving general competence and revealing tasks that remain hard due to unfamiliarity from base model.

- 3.2.1 Data Construction. We select about 750K SNS entries DSNS1 from DSNS, covering 75 heterogeneous tasks and all capability types, such as post taxonomy, query classification, machine reading comprehension, post view search and SNS domain translation. A balanced sampling schedule is also conducted to preserve the visibility of long-tail behaviors. Additionally, to maintain reasoning and

general competence, we attach 50K data DGEN1 from DGEN with rationales, which are widely regarded as beneficial for preserving model knowledge and supporting structured reasoning.

- 3.2.2 Reward Function. In RL, the reward function is the most critical supervision signal during training. Unlike prior works [24] that takes a simple approach by defining a single rule-based reward to check whether a response follows a specified format and whether the final answer is correct, we take a different path. Considering that the downstream scenarios aligned with RedOne 2.0 span heterogeneous tasks with substantial variation in both format and content, and that their evaluation criteria are highly diverse, we define task type specific rewards for a sampled pair (𝑄,𝐴) and model’s output 𝑂 as following:

1) Exact Match. For close-ended problems with determinate answers, such as classification or multiple-choice, we focus on

constraining answer consistency with exact match score.

1, 𝑂 = 𝐴, 0, otherwise.

(2)

REM(𝑂,𝐴) =

- 2) Metrics-based. For open-ended tasks such as translation, we

avoid seeking a single binary “correct” standard. Instead, we define rewards using task-specific evaluation metrics Eval.

RMet(𝑂,𝐴) = Eval(𝑂,𝐴) (3)

- 3) Sand Box. For tasks like code generation, traditional exact-

match or metric-based scoring struggles to objectively assess output quality. The most direct approach is sandbox simulation, where we create an execution environment to run the generated solution and evaluate it by the obtained results Exe(𝑂).

RSandBox(𝑂,𝐴) =

1, Exe(𝑂) = 𝐴, 0, otherwise.

(4)

- 4) Pattern. Additionally, given the instability of generative LLM

output formats and the existence of community benchmarks targeting instruction following, we design a pattern-based matching mechanism Match that emphasizes adherence to specified formats rather than the semantic content itself.

RPattern(𝑂,𝐴) = Match(𝑂,𝐴) (5)

Finally, each sample’s reward R(𝑂,𝐴) is mapped to the corresponding reward function based on its task category.

- 3.2.3 Domain Alignment. In this stage, our domain alignment objective is to jointly raise general competence and SNS-specific capability while fully exploiting the model’s latent potential, and to systematically surface sub-tasks that remain difficult due to base

model’s unfamiliarity. In practice, we randomly mix DSNS1 and DGEN1 to form D1, which conducts DAPO-based [76] RL training for this stage. For a specific data entry (𝑄,𝐴), DAPO samples a group of𝐺 individual candidate outputs {𝑂𝑖}𝐺𝑖=1 from the old policy model 𝜋𝜃old. Then, we can optimize the policy 𝜋𝜃 by optimizing the following loss function:

LDAPO(𝜃) =E(𝑄,𝐴)∼D1,{𝑂𝑖}𝐺

𝑖=1∼𝜋𝜃old (·|𝑄)

###### ∑︁𝐺

###### ∑︁|𝑂𝑖|

1

min 𝑟𝑖,𝑡 (𝜃) 𝐴ˆ𝑖,𝑡,

𝐺 𝑖=1 |𝑂𝑖|

𝑖=1

𝑡=1

clip(𝑟𝑖,𝑡 (𝜃), 1 − 𝜀low, 1 + 𝜀high) 𝐴ˆ𝑖,𝑡

s.t. 0 < {𝑂𝑖 | is_equivalent(𝐴,𝑂𝑖)} < 𝐺 (6) where 𝜀low and 𝜀high control the clipping range, and

𝜋𝜃 (𝑂𝑖,𝑡 | 𝑄,𝑂𝑖,<𝑡) 𝜋𝜃old(𝑂𝑖,𝑡 | 𝑄,𝑂𝑖,<𝑡)

(7)

𝑟𝑖,𝑡 (𝜃) =

𝐴ˆ𝑖,𝑡 = R𝑖 − mean({R𝑖}𝐺𝑖=1) std({R𝑖}𝐺𝑖=1)

(8)

Finally, this alignment stage yields broad, stable gains without premature specialization, while producing a fine-grained diagnosis of where the model underperforms. And the resulting capability map guides targeted repair in the next stage.

##### 3.3 Targeted Fine-Tuning

After initial SNS alignment, Targeted Fine-Tuning directly addresses tasks that remain weak. The emphasis is on repairing deficiencies while preserving previous gains, achieved by blending difficult SNS data with carefully filtered general data.

- 3.3.1 Data Preparation. We construct a dataset D2 of 1.8M examples comprising 1.7M SNS instances and 100K general-domain instances. The SNS portion is derived from our pre-training data

corpus DSNS, which refer to the failure tasks bucket in previous stage identified via evaluation result on various benchmarks. We further stratify these examples by capability and upweight rare but impactful cases. For the general portion, we sample them from DSNS and introduce examples with soft labels by generating candidates responses from the previous stage model. Concretely, for a given prompt we generate 8 candidate completions by the resulted model from the first stage, and score them with a composite quality signal from a judge model, and select the best one to form a soft supervisory target. These soft labels not only mitigate catastrophic forgetting of general knowledge during SFT, but also reduce impact on distributional transformation between “ground-truth” labels and the first stage model’s learned distribution, thereby improving learning efficiency for SNS tasks [48]. In fact, this small set of soft-labeled general data functions as a data-level regularizer, preventing SNS-focused SFT from drifting too far from the reference model.

- 3.3.2 Targeted Learning. In this stage, optimization aims to close gaps on underperforming SNS tasks while preserving gains from the previous stage. We use a plain SFT objective on a mixture of hard SNS examples and a small set of general-domain examples with soft labels:

∑︁|𝑂|

log𝜋𝜃 𝑂𝑡 𝑄,𝑂<𝑡 (9)

LSFT = −E(𝑄,𝐴)∼D2,𝑂∼𝜋𝜃 (·|𝑄)

𝑡=1

where 𝜋𝜃 denotes the current policy model, 𝑄 is the question, and 𝑂 is the target output sequence. Finally, this stage yields consistent improvements on previously weak SNS tasks while maintaining gains from the first stage on the most capability types. The combination of SNS-prioritized repair and soft-label regularization from a small general set produces a more balanced and robust model prepared for RL-based refinement in the last stage.

##### 3.4 Refinement Learning

The final stage, Refinement Learning, consolidates prior gains and achieves further performance improvements. This is done by applying RL after the previous SFT-based stage, with the training again centered on SNS data.

3.4.1 Further Refinement. Specifically, we use approximately 400K examples drawn from the SNS and general sources as in the previous stage, with an emphasis on the difficult subsets. We initialize the policy from the prior stage to provide a strong starting base model, and then apply preference-based DAPO [76] as same as the first stage training process for refinement. In this stage, we also increase the proportion of samples with rationale to 57.18%, further preserving the model’s reasoning ability and benefiting a broad range of downstream tasks. After training, the model’s behavior is

###### Table 1: Comparison results across General-Bench, SNS-Bench and SNS-TransBench. Bold entries indicate the best performance, while underlined entries denote the second one in each category.

General-Bench SNS-Bench SNS-TransBench

ZH→EN EN→ZH

Models

Avg. Taxon. Hash. QCorr MRC NER Gender CHLW QGen Avg.

Avg. BLEU chrF++ BLEU chrF++

Proprietary Large Language Models or The Scale of Large Language Models > 100B

GPT-4o-1120 70.72 65.79 84.98 51.79 58.89 54.99 88.08 38.96 47.33 61.35 40.32 63.91 49.15 47.28 50.17 Gemini-2.0-Flash 74.42 68.76 87.36 48.41 52.21 53.58 89.64 37.39 46.27 60.45 32.72 58.84 41.80 40.16 43.38 Claude-3.7-Sonnet 75.10 72.03 88.83 54.10 54.86 56.13 92.23 31.11 45.49 61.85 35.63 61.66 45.79 44.23 46.83 Doubao-1.5-Pro-32k 76.13 30.00 83.21 58.25 61.32 56.60 90.67 30.61 46.55 57.15 33.71 61.85 45.54 44.35 46.36 Qwen-Max 71.86 65.68 84.47 54.36 61.34 55.78 91.19 37.97 46.64 62.18 35.55 60.92 46.08 44.14 46.67 GLM-4-Plus 70.25 65.46 84.31 52.13 55.81 53.16 86.53 30.09 44.68 59.02 41.57 65.95 48.79 47.06 50.84 GPT-OSS-120B 76.71 67.20 86.04 56.83 61.45 55.84 91.19 38.53 45.61 62.84 33.06 59.73 42.67 40.47 43.98 dots.llm1 70.20 62.96 82.45 42.10 40.75 14.93 89.12 31.09 44.63 51.00 30.93 58.66 44.42 42.8 44.20 GLM-4.5 73.66 70.76 86.93 56.22 64.94 57.23 92.75 41.32 45.47 64.45 30.57 56.77 39.55 38.2 41.27 Deepseek-V3-0324 75.22 67.27 86.59 47.71 60.97 56.00 90.16 40.45 46.03 61.90 35.65 61.58 46.86 44.58 47.17 DeepSeek-V3.1 77.02 70.20 88.97 48.67 62.37 55.22 91.19 33.60 46.42 62.08 31.94 58.8 41.64 39.77 43.04

The Scale of Large Language Models < 10B

Qwen3-4B 69.80 60.88 81.90 38.31 34.69 44.50 79.27 28.17 46.75 51.81 26.87 54.26 36.35 35.41 38.22 Qwen2.5-7B 63.01 49.50 73.80 42.37 45.32 45.41 88.08 33.76 44.65 52.86 31.43 55.91 38.36 36.48 40.55 Llama-3.1-8B 51.24 37.74 66.62 33.32 31.27 47.10 74.61 26.88 38.60 44.52 23.07 48.15 29.32 29.13 32.42 Ministral-8B 49.93 42.62 70.58 36.24 30.71 37.79 82.38 28.04 46.27 46.83 25.67 50.91 32.02 31.18 34.95 InternLM3-8B 58.55 51.83 76.98 38.65 25.25 39.41 66.84 44.71 43.46 48.39 24.85 50.44 35.58 34.04 36.23 Qwen3-8B 66.90 58.67 82.44 46.47 48.45 44.68 89.12 27.95 45.89 55.46 33.21 58.81 40.09 38.85 42.74 GLM-4-9B-0414 63.27 56.03 77.67 38.03 45.29 47.01 51.30 27.51 45.52 48.55 32.20 56.90 39.73 37.40 41.57 RedOne-7B 63.83 72.18 88.02 65.09 63.98 51.86 70.47 74.73 48.69 66.88 38.06 62.66 46.88 44.82 48.11 RedOne 2.0 4B 70.80 75.85 89.05 60.92 66.54 43.15 78.76 79.11 47.17 67.57 38.61 62.46 45.78 43.84 47.67

10B < The Scale of Large Language Models < 100B

Phi-4-14B 63.00 57.62 79.56 46.32 53.39 44.99 89.12 29.23 44.76 55.62 31.28 57.23 37.58 36.68 40.69 GPT-OSS-20B 74.76 62.89 83.99 54.58 56.43 54.81 92.23 32.68 45.26 60.36 30.74 57.46 37.83 36.19 40.56 Mistral-Small-24B 65.63 64.88 83.89 48.77 46.51 52.09 91.19 32.10 46.01 58.18 31.29 56.72 39.28 37.32 41.15 Qwen3-30B-A3B 74.46 64.29 85.81 44.75 52.23 45.75 90.16 27.19 45.67 56.98 34.07 58.86 41.19 39.51 37.05 GLM-4-32B-0414 74.39 63.36 85.50 47.33 53.72 50.41 80.31 33.19 46.90 57.59 36.32 61.31 42.53 40.77 45.23 Qwen2.5-32B 71.68 59.90 80.51 46.00 55.04 54.51 90.67 38.84 45.66 58.89 32.56 58.14 42.34 40.71 43.44 Qwen3-32B 72.67 61.52 86.04 49.39 54.56 53.76 91.19 33.48 45.74 59.46 32.15 58.54 40.44 38.85 42.50 Llama-3.3-70B 67.64 62.94 83.28 50.76 27.38 56.09 91.19 33.58 46.41 56.45 34.00 59.18 41.25 39.56 43.50 RedOne-32B 73.72 81.45 90.19 67.07 59.24 51.66 81.87 70.40 50.37 69.03 40.55 64.54 48.20 46.05 49.84 RedOne 2.0 30B-A3B 75.17 77.02 89.99 63.76 62.16 54.15 81.87 74.19 49.15 69.04 40.22 63.88 48.06 45.95 49.54

stabilized and smoothed within the explored solution space, yielding further improvements on both SNS-specific and general tasks. Compared to the previous stage, the RL-based refinement delivers better convergence and more robust domain adaptation.

- 4 Experiments

- 4.1 Implementation Details

During the Exploratory Learning stage, we trained for 500 steps with maximum prompt/response lengths of 10,000/8,192 tokens (18,192 total), plus a 4,096-token overlong buffer with 1.0 penalty factor. We used a prompt batch size of 1,024 with 16 responses per prompt (global batch size 16,384) and mini-batch size covering 256 prompts, yielding 4 gradient updates per rollout. We adopted DAPO with clipping parameters 𝜀low = 0.2 and 𝜀high = 0.28. Optimization employed AdamW with a constant learning rate of 5×10−6, weight decay 0.1, with linear warmup applied for 10 rollout steps. In Targeted Fine-Tuning, we trained for 2 epochs with batch size 64 and maximum sequence length of 16,384 using sequence packing. We optimized cross-entropy loss with AdamW at a learning rate 5 × 10−6, applying a warmup ratio of 0.1 followed by cosine

scheduling. The final Refinement Learning stage mirrored the first stage configuration.

##### 4.2 Experimental Setting

4.2.1 Benchmarks. We perform a comprehensive evaluation of RedOne 2.0 and baselines in both the general and SNS domain capabilities using commonly used benchmarks in the community. Specifically, in general domain, we systematically assess six capabilities, including knowledge reasoning, mathematical reasoning, code generation, machine translation, instruction following, and hallucination detection, as well as CompassBench [14], a comprehensive bench to provide an integrated, multi-dimensional view of model performance. 1) Knowledge Reasoning. We use MMLU [27], CMMLU [45], C-Eval [30], GPQA-Diamond [60], NewsBench [47], MMLU-Pro [69],BBH [64], andGaokaoBench [82] to probe broad and specialized knowledge, reasoning robustness, difficulty-calibrated multiple choice, and exam-style generalization in both English and Chinese. 2) Mathematical Reasoning. We adopt GSM8K [12], MATH500 [28], and the high-stakes AIME 2025 [52] set to measure multi-step arithmetic and competition-level problem solving. 3)

Code Generation. We evaluate program synthesis and correctness with HumanEval [11], MBPP [5], and the temporally refreshed, contamination-aware LiveCodeBench [33], reporting pass@k and execution-based metrics. 4) Machine Translation. We benchmark multilingual translation with the WMT tasks (i.e. WMT-22 [40], WMT-23 [18] and WMT-24 [39]) and FLORES [20], covering diverse language pairs and domains. 5) Instruction Following. We employ IFEval [84], which provides automatically verifiable constraints to quantify compliance under explicit instructions. 6) Hallucination Detection. We use HaluEval [46] to assess the tendency to produce unverifiable or fabricated content across question answering, dialogue, and summarization settings.

In the SNS domain, we validate models on benchmarks built from real SNS scenarios, covering five aspects: post comprehension, information retrieval, sentiment and intent analysis, personalized recommendation, and translation. We use SNS-Bench [25], a large-scale bench with 6,658 questions spanning eight tasks from a social platform with over 300M users, which includes the following tasks: 1) Note-Taxonomy (Taxon.) for content categorization; 2) Note-Hashtag (Hash.) to select suitable tags; 3) NoteQueryCorr (QCorr) to align user queries with note content and topic; 4) Note-MRC (MRC) for simple and complex reading comprehension over long notes; 5) Note-NER (NER) for entity extraction; 6) Note-Gender (Gender) to assess gender-sensitive appeal; 7) Note-CHLW (CHLW) to highlight salient words in comment threads; and 8) Note-QueryGen (QGen) to produce effective search queries. For translation, we adopt SNS-TransBench [26], a curated set of 2,858 English–Chinese cases from posts, comments, and multimedia captions that emphasizes phenomena central to SNS translation, including humor localization, emoji semantics, and meme adaptation. It tests whether models can preserve pragmatics, style, and culture-bound references in short, high-context text typical of social platforms.

- 4.2.2 Baselines. We conduct comparison experiments with various proprietary models, including GPT4o-1120 [32], Gemini-2.0Flash [66], Claude-3.7-Sonnet [4], Doubao-1.5-Pro-32k [16], QwenMax [74], and GLM-4-Plus [19], open-source models, such as Qwen series [73, 74], Llama series [21], Ministral [54], Mistral-Small24B [55], InternLM3-8B [10], Phi-4-14B [1], dots.llm1 [31], gpt-oss series [2], GLM series [78] and DeepSeek series [49], as well as SNS domain specific models RedOne [83].

##### 4.3 Main Results

As shown in Table 1, we conduct a comparison with various models across General-Bench, SNS-Bench, and SNS-TransBench, covering a broad spectrum of capabilities from general reasoning to SNSdomain understanding and multilingual transfer.

Across all benchmarks, RedOne 2.0 averagely achieves strong and balanced results, surpassing both open- and closed-source baselines of comparable scale. Specifically, RedOne 2.0 4B attains the highest average score on General-Bench with 70.8, exceeding even larger open models such as Qwen3-8B and GLM-4-9B, and achieving comparable or superior results to some proprietary LLMs or LLMs with more than 100B parameters. This demonstrates that the proposed three-stage post-training pipeline effectively enhances both general and domain-specific capabilities even at smaller scales.

###### Table 2: Generalization of our training pipeline over different base models. We report the average performance for GeneralBench, SNS-Bench and SNS-TransBench.

###### Models General-Bench SNS-Bench SNS-TransBench

Qwen3-4B 69.80 51.81 38.22 RedOne 2.0 4B 70.80 67.57 47.67

Qwen3-8B 66.90 55.46 42.74 RedOne 2.0 8B 69.27 65.82 46.72

Qwen3-30B-A3B 74.46 56.98 37.05 RedOne 2.0 30B-A3B 75.17 69.04 49.54

Qwen3-32B 72.67 59.46 42.50 RedOne 2.0 32B 73.17 69.76 49.11

On SNS-Bench, which evaluates domain-specific understanding and reasoning across eight tasks, RedOne 2.0 still leads within its scale group. The 4B variant achieves an average score of 67.57, outperforming all sub-10B baselines and exceeding the previous RedOne-7B model by 0.69, despite having fewer parameters. Similarly, the 30B-A3B version achieves 69.04, even matching or surpassing much larger models such as GPT-4o and GLM-4.5. These results validate that RedOne 2.0 not only inherits strong generalization from the base models but also substantially improves SNS-domain competence through progressive alignment. On SNS-TransBench, which measures cross-lingual understanding and translation quality between Chinese and English, RedOne 2.0 maintains competitive results across BLEU and chrF++ metrics. Both the 4B and 30B-A3B variants achieve the top-2 overall averages with 47.67 and 49.54, respectively, outperforming all similarly scaled models. The consistent performance across both translation directions indicates that RedOne 2.0’s alignment pipeline preserves linguistic versatility while improving domain adaptation.

We further observe clear scalability in the RedOne 2.0 family. As model size increases from 4B to 30B, both general and SNS-specific metrics steadily improve, with the average gain on General-Becnh, SNS-Bench and SNS-TransBench exceeding 4.37, 1.47 and 1.87, respectively. This confirms that the proposed three-stage pipeline scales effectively and provides stable improvements without overfitting to the SNS domain. Compared with the RedOne models, RedOne 2.0 shows promising gains across majority evaluation suites. For instance, RedOne 2.0 4B improves by 6.97 on General-Bench and by 0.69 on SNS-Bench relative to RedOne-7B. Additionally, for SNS-TransBench, although RedOne 2.0 series are smaller than RedOne, they still achieve comparable performance. And comparing with RedOne, RedOne 2.0 obtains higher improvement from base model. Overall, these results confirm the effectiveness of RedOne 2.0 in achieving efficient, scalable and stable alignment.

##### 4.4 More Analysis

4.4.1 Generalization Across Different Base-Model Scales. Table 2 demonstrates that our three-stage training pipeline generalizes effectively across base models of different scales. Consistent improvements are observed on all benchmarks, confirming the robustness of our approach in transferring alignment benefits from smaller to larger models. Moreover, scaling up the base model further amplifies the overall performance of RedOne 2.0, indicating that larger

- Table 3: The impact of different training stage on Qwen3-4B’s performance.

Exploratory Learning

Targeted Fine-Tuning

Refinement Learning

General -Bench

SNS -Bench

SNS-Trans Bench

69.80 51.81 38.22 ✓ 63.65 61.10 46.00 ✓ ✓ 69.80 63.03 45.95

✓ 71.25 62.27 43.35 ✓ ✓ 70.04 65.67 47.72 ✓ ✓ ✓ 70.80 67.57 47.67

- Table 4: Comparison with task specific fine-tuning on Qwen34B and RedOne2.0 4B.

Models Hash. QCorr MRC Qwen3-4B 81.90 38.31 34.69

Qwen3-4B (Fine-tuned) 90.12 60.11 57.54 RedOne 2.0 4B 89.05 60.92 66.54

Models CHLW QGen SNS-Trans Qwen3-4B 28.17 46.75 38.22

Qwen3-4B (Fine-tuned) 67.24 49.24 44.25 RedOne 2.0 4B 79.11 47.17 47.67

capacities allow more effective utilization of the staged optimization signals. Notably, the 4B and 30B-A3B variants exhibit superior gains comparing with similar scale models, particularly on SNS-related benchmarks. We think this is because that they are both based on instruction-tuned backbones, and stronger instruction-following capabilities can better absorb and express the multi-stage alignment signals, leading to stronger generalization and domain adaptation.

- 4.4.2 IncrementalPerformanceOver three-Stage Training. As shown in Table 3), we evaluate the incremental impact of each stage in our three-phase training framework. The RL-based Exploratory Learning stage establishes a strong foundation, improving performance to 71.25% on General-Bench, 62.27% on SNS-Bench, and 43.35% on SNS-TransBench, highlighting its effectiveness in consistently enhancing the overall capability of the base model. The SFT-based Targeted Fine-Tuning stage then addresses weaknesses in the SNS domain exhibited in previous stage, raising scores to 65.67% on SNS-Bench and 47.72% on SNS-TransBench, with only a slight drop 1.21% on General-Bench. Finally, the RL-based Refinement Learning stage balances performance across tasks, increasing the average from 61.14% to 62.01% and resulting in final scores of 70.80% on General-Bench, 67.57% on SNS-Bench, and 47.67% on SNS-TransBench.
- 4.4.3 Superiority to the Naive SFT Followed RL Baseline. Then, considering the most notable shift in RedOne 2.0 lies in its departure from the traditional SFT-centric domain-specific post-training paradigm to RL, we conduct experiments to compare it with naive SFT followed RL baseline. This baseline typically began with SFT for domain adaptation, followed by RL to align the model with human preferences or downstream objectives, as shown in the second and

###### Table 5: RedOne2.0’s online application on personalized recreation of posts’ title.

Metrics Relative Change Business Value Advertiser Value (AdvV) ↑ +0.43%

Vague Titles Ratio ↓ -11.9% Practical Titles Ratio ↑ +7.1% Authentic Titles Ratio ↑ +12.9% Interactive Titles Ratio ↑ +25.8%

Content Quality

third rows of Table 3. While SFT can effectively boost performance in SNS domains, it often causes a “seesaw” effect, significantly reducing general capability from 69.80 to 63.65. Although subsequent RL attempts to mitigate this issue, the overall improvements across the three benchmarks remain limited. In contrast, RedOne 2.0 refine the process: starting with RL to establish domain priors, followed by SFT for targeted enhancement, and concluding with RL for final optimization. This paradigm effectively avoids the trade-off between general and domain-specific performance and surpasses the naive baseline by 1.00 on General-Bench, 4.54 on SNS-Bench, and 1.72 on SNS-TransBench.

- 4.4.4 Comparison with Task Specific Fine-tuning. We also compares RedOne2.0 framework, designed for unified optimization across all tasks, against the conventional task specific fine-tuning approach. As detailed in Table 4, this method also yields strong performance on its target objective. For instance, a Qwen3-4B model fine-tuned specifically for QGen achieves 49.24, and another fine-tuned for Hash. reaches 90.12. However, RedOne2.0 4B, trained concurrently on a mixture of all tasks, delivers robust and highly competitive results across the entire spectrum of benchmarks. Especially, it outperforms the task specific fine-tuned Qwen3-4B models on MRC with 9.00, and on CHLW with 11.87. It also maintains strong performance on QCorr at 60.92 and on SNS-Trans at 47.67. These results substantiate that a unified training framework can effectively capture and leverage beneficial inter-task relationships, enabling a single model to achieve comprehensive and better capability.
- 4.5 Online Application

We deployed RedOne2.0 on a large scale social networking platform with more than 3 million users to recommend personalized re-created post titles in real time. Each pre-published title is routed to the service, which performs semantic analysis and produces an enhanced title that preserves the original intent while optimizing for engagement. The suggestion is exhibited to the creator or, in selected traffic buckets, used directly to measure performance against the original. Evaluation covered business impact and content quality. The primary business indicator is Advertiser Value (AdvV), which reflects the value delivered to advertisers through audience quality and engagement. Content quality is measured through human review across four dimensions: vagueness, practicality, authenticity, and interactivity.

As shown in Table 5, the online test was conducted over several weeks and millions of posts, and showed consistent gains.

###### Table 6: Good cases for personalized re-creation of post titles.

Title Source Content

Case 1 Original

Plum rain season, a great helper for dehumidification and mold prevention. Base Model

Dehumidification essential for the plum rain season,a fresh choice for a dry and comfortable life. RedOne 2.0

Say goodbye to “steamy” homes! Rescue your plum rain season.

Case 2 Original

Beijing wedding photo recommendations: 17 lawn wedding photo outdoor spots. Base Model

Dreamy lawn wedding photos, capturing the most beautiful moments. RedOne 2.0

Escape the studio! 17 stunning lawns capture cinematic-level wedding photos.

###### Table 7: Bad case for personalized re-creation of post titles.

###### Title Source Content

Don’t buy the wrong transportation card for Osaka and Kyoto! A lesson learned the hard way! Base Model

Original

A guide to Japanese transportation cards—stop making these mistakes!

Avoid these pitfalls for your Kansai trip, check out the guide now.

RedOne 2.0

Advertiser Value increased by 0.43%, a statistically significant improvement at platform scale. Human evaluation reported an 11.9% reduction in vague titles and increases of 7.1% in practical titles, 12.9% in authentic titles, and 25.8% in interactive titles. The strong rise in interactive titles indicates that the model learns linguistic patterns that encourage responses such as comments and shares. These results also suggest a clear link between quality improvements and business outcomes. Moreover, practical and authentic titles are likely to increase user trust and dwell time, while interactive titles stimulate community activity. Deploying RedOne2.0 therefore improves user experience and yields measurable advertiser value, which demonstrates its effectiveness for real world content optimization.

##### 4.6 Case Study

To qualitatively assess RedOne 2.0, we compare its outputs with a baseline on personalized re-created titles. As shown in Table 6, RedOne 2.0 consistently produces more evocative and engaging phrasing. For the dehumidification example, the baseline remains serviceable yet generic, whereas RedOne 2.0 introduces a vivid word of a steamy home and adds a clear imperative that heightens emotional resonance. A similar pattern holds for the example of the wedding photography. The baseline provides a broad description, while RedOne 2.0 frames the content as an exclusive discovery that

promises cinematic results, which is likely to raise curiosity and strengthen click intent. We also exhibited a bad case in scenarios that require strict preservation of critical facts, as illustrated in Table 7. The original title centers on the risk of choosing the wrong transportation card for Osaka and Kyoto and stresses a hard learned lesson. The baseline remains close to this focus. RedOne 2.0 generates a more interactive sentence but generalizes the topic to Kansai travel and omits the key reference to the transportation card, which weakens informational precision and urgency. These cases indicate that RedOne 2.0 excels at optimizing for engagement and stylistic appeal yet can sometimes over optimize at the expense of essential details. Future work should reinforce faithfulness constraints while preserving expressiveness.

##### 5 Conclusion

In this paper, we present RedOne 2.0, an SNS-specific LLM posttraining framework tailored for SNS, where tasks are highly heterogeneous, dynamic, and culturally diverse. Unlike traditional SFTcentric approaches that risk catastrophic forgetting and unstable trade-offs between in-domain and out-of-domain performance, RedOne 2.0 adopts a progressive, RL-prioritized three-stage pipeline: Exploratory Learning to establish initial domain alignment and surface weaknesses, Targeted Fine-Tuning to selectively repair deficiencies while retaining general competence, and Refinement Learning to consolidate improvements via RL with SNS-centric rewards. Supported by a large, task-diverse dataset spanning more than 75 SNS tasks and high quality general corpus, this paradigm demonstrates strong data efficiency, stable adaptation, and robust generalization even at compact model scales. Overall, RedOne 2.0 provides a competitive, cost-effective, and scalable baseline for LLM deployment in SNS, advancing capability without sacrificing robustness, safety, or general usability.

##### References

- [1] Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. 2024. Phi-4 technical report. arXiv preprint arXiv:2412.08905 (2024).
- [2] Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. 2025. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925 (2025).
- [3] Sacha Altay, Emma Hoes, and Magdalena Wojcieszak. 2025. Following news on social media boosts knowledge, belief accuracy and trust. Nature Human Behaviour (2025), 1–10.
- [4] Claude Anthropic. 2025. 3.7 sonnet and claude code.
- [5] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. 2021. Program synthesis with large language models. arXiv preprint arXiv:2108.07732 (2021).
- [6] Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q Jiang, Jia Deng, Stella Biderman, and Sean Welleck.

2023. Llemma: An open language model for mathematics. arXiv preprint arXiv:2310.10631 (2023).

- [7] Lei Bai, Zhongrui Cai, Maosong Cao, Weihan Cao, Chiyu Chen, Haojiong Chen, Kai Chen, Pengcheng Chen, Ying Chen, Yongkang Chen, et al. 2025. Intern-s1: A scientific multimodal foundation model. arXiv preprint arXiv:2508.15763 (2025).
- [8] Eytan Bakshy, Solomon Messing, and Lada A Adamic. 2015. Exposure to ideologically diverse news and opinion on Facebook. Science 348, 6239 (2015), 1130–1132.
- [9] Gagan Bhatia, El Moatez Billah Nagoudi, Hasan Cavusoglu, and Muhammad Abdul-Mageed. 2024. Fintral: A family of gpt-4 level multimodal financial large language models. arXiv preprint arXiv:2402.10986 (2024).
- [10] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, Xiaoyi Dong, Haodong Duan, Qi Fan, Zhaoye Fei, Yang Gao, Jiaye Ge, Chenya Gu, Yuzhe Gu, Tao Gui, Aijia Guo, Qipeng Guo, Conghui He, Yingfan Hu, Ting Huang, Tao Jiang, Penglong Jiao, Zhenjiang Jin, Zhikai Lei, Jiaxing Li, Jingwen Li, Linyang Li, Shuaibin Li, Wei Li, Yining Li,

- Hongwei Liu, Jiangning Liu, Jiawei Hong, Kaiwen Liu, Kuikun Liu, Xiaoran Liu, Chengqi Lv, Haijun Lv, Kai Lv, Li Ma, Runyuan Ma, Zerun Ma, Wenchang Ning, Linke Ouyang, Jiantao Qiu, Yuan Qu, Fukai Shang, Yunfan Shao, Demin Song, Zifan Song, Zhihao Sui, Peng Sun, Yu Sun, Huanze Tang, Bin Wang, Guoteng Wang, Jiaqi Wang, Jiayu Wang, Rui Wang, Yudong Wang, Ziyi Wang, Xingjian Wei, Qizhen Weng, Fan Wu, Yingtong Xiong, Chao Xu, Ruiliang Xu, Hang Yan, Yirong Yan, Xiaogui Yang, Haochen Ye, Huaiyuan Ying, Jia Yu, Jing Yu, Yuhang Zang, Chuyu Zhang, Li Zhang, Pan Zhang, Peng Zhang, Ruijie Zhang, Shuo Zhang, Songyang Zhang, Wenjian Zhang, Wenwei Zhang, Xingcheng Zhang, Xinyue Zhang, Hui Zhao, Qian Zhao, Xiaomeng Zhao, Fengzhe Zhou, Zaida Zhou, Jingming Zhuo, Yicheng Zou, Xipeng Qiu, Yu Qiao, and Dahua Lin. 2024. InternLM2 Technical Report. arXiv:2403.17297 [cs.CL]
- [11] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374 (2021).
- [12] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168 (2021).
- [13] Pierre Colombo, Telmo Pessoa Pires, Malik Boudiaf, Dominic Culver, Rui Melo, Caio Corro, Andre FT Martins, Fabrizio Esposito, Vera Lúcia Raposo, Sofia Morgado, et al. 2024. Saullm-7b: A pioneering large language model for law. arXiv preprint arXiv:2403.03883 (2024).
- [14] OpenCompass Contributors. 2023. OpenCompass: A Universal Evaluation Platform for Foundation Models. https://github.com/open-compass/opencompass.
- [15] Matthew Dahl, Varun Magesh, Mirac Suzgun, and Daniel E Ho. 2024. Large legal fictions: Profiling legal hallucinations in large language models. Journal of Legal Analysis 16, 1 (2024), 64–93.
- [16] Doubao-Team. 2025. Doubao-1.5-pro: Model release. https://team.doubao.com/ en/special/doubao_1_5_pro.
- [17] Shangbin Feng, Herun Wan, Ningnan Wang, Zhaoxuan Tan, Minnan Luo, and Yulia Tsvetkov. 2024. What does the bot say? opportunities and risks of large language models in social media bot detection. arXiv preprint arXiv:2402.00371

(2024).

- [18] Markus Freitag, Nitika Mathur, Chi-kiu Lo, Eleftherios Avramidis, Ricardo Rei, Brian Thompson, Tom Kocmi, Frederic Blain, Daniel Deutsch, Craig Stewart, et al. 2023. Results of WMT23 metrics shared task: Metrics might be guilty but references are not innocent. In Proceedings of the Eighth Conference on Machine Translation. 578–628.
- [19] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, et al. 2024. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793 (2024).
- [20] Naman Goyal, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc’Aurelio Ranzato, Francisco Guzmán, and Angela Fan. 2022. The flores-101 evaluation benchmark for low-resource and multilingual machine translation. Transactions of the Association for Computational Linguistics 10 (2022), 522–538.
- [21] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783

(2024).

- [22] Thomas S Gruca, Joyce E Berg, and Michael Cipriano. 2008. Incentive and accuracy issues in movie prediction markets. The Journal of Prediction Markets 2, 1 (2008), 29–43.
- [23] Neel Guha, Julian Nyarko, Daniel Ho, Christopher Ré, Adam Chilton, Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel Rockmore, Diego Zambrano, et al. 2023. Legalbench: A collaboratively built benchmark for measuring legal reasoning in large language models. Advances in neural information processing systems 36 (2023), 44123–44279.
- [24] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025).
- [25] Hongcheng Guo, Shaosheng Cao, Boyang Wang, Lei Li, Liang Chen, Xinze Lyu, Zhe Xu, Yao Hu, Zhoujun Li, et al. [n.d.]. SNS-Bench: Defining, Building, and Assessing Capabilities of Large Language Models in Social Networking Services. In Forty-second International Conference on Machine Learning.
- [26] Hongcheng Guo, Fei Zhao, Shaosheng Cao, Xinze Lyu, Ziyan Liu, Yue Wang, Boyang Wang, Zhoujun Li, Chonggang Lu, Zhe Xu, et al. 2025. Redefining Machine Translation on Social Network Services with Large Language Models. arXiv preprint arXiv:2504.07901 (2025).
- [27] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300 (2020).
- [28] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem

- solving with the math dataset. arXiv preprint arXiv:2103.03874 (2021).
- [29] Jianheng Huang, Leyang Cui, Ante Wang, Chengyi Yang, Xinting Liao, Linfeng Song, Junfeng Yao, and Jinsong Su. 2024. Mitigating catastrophic forgetting in large language models with self-synthesized rehearsal. arXiv preprint arXiv:2403.01244 (2024).
- [30] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Yao Fu, et al. 2023. C-eval: A multilevel multi-discipline chinese evaluation suite for foundation models. Advances in Neural Information Processing Systems 36 (2023), 62991–63010.
- [31] Bi Huo, Bin Tu, Cheng Qin, Da Zheng, Debing Zhang, Dongjie Zhang, En Li, Fu Guo, Jian Yao, Jie Lou, et al. 2025. dots. llm1 Technical Report. arXiv preprint arXiv:2506.05767 (2025).
- [32] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024).
- [33] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2024. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974 (2024).
- [34] Julie Jiang and Emilio Ferrara. 2023. Social-llm: Modeling user behavior at scale using language models and social network data. arXiv preprint arXiv:2401.00893

(2023).

- [35] Songtao Jiang, Yuan Wang, Sibo Song, Yan Zhang, Zijie Meng, Bohan Lei, Jian Wu, Jimeng Sun, and Zuozhu Liu. 2025. Omniv-med: Scaling medical vision-language model for universal visual understanding. arXiv preprint arXiv:2504.14692 (2025).
- [36] Hangzhan Jin, Sitao Luan, Sicheng Lyu, Guillaume Rabusseau, Reihaneh Rabbany, Doina Precup, and Mohammad Hamdaqa. 2025. RL Fine-Tuning Heals OOD Forgetting in SFT. arXiv preprint arXiv:2509.12235 (2025).
- [37] Kris-Fillip Kahl, Tolga Buz, Russa Biswas, and Gerard De Melo. 2024. LLMs Cannot (Yet) Match the Specificity and Simplicity of Online Communities in Long Form Question Answering. In Findings of the Association for Computational Linguistics: EMNLP 2024. 2028–2053.
- [38] Daniel Martin Katz, Michael James Bommarito, Shang Gao, and Pablo Arredondo.

2024. Gpt-4 passes the bar exam. Philosophical Transactions of the Royal Society A 382, 2270 (2024), 20230254.

- [39] Tom Kocmi, Eleftherios Avramidis, Rachel Bawden, Ondřej Bojar, Anton Dvorkovich, Christian Federmann, Mark Fishel, Markus Freitag, Thamme Gowda, Roman Grundkiewicz, et al. 2024. Findings of the WMT24 general machine translation shared task: The LLM era is here but MT is not solved yet. In Proceedings of the Ninth Conference on Machine Translation. 1–46.
- [40] Tom Kocmi, Rachel Bawden, Ondřej Bojar, Anton Dvorkovich, Christian Federmann, Mark Fishel, Thamme Gowda, Yvette Graham, Roman Grundkiewicz, Barry Haddow, et al. 2022. Findings of the 2022 conference on machine translation (WMT22). In Proceedings of the Seventh Conference on Machine Translation (WMT). 1–45.
- [41] Mahi Kolla, Siddharth Salunkhe, Eshwar Chandrasekharan, and Koustuv Saha.

2024. Llm-mod: Can large language models assist content moderation?. In Extended Abstracts of the CHI Conference on Human Factors in Computing Systems. 1–8.

- [42] Suhas Kotha, Jacob Mitchell Springer, and Aditi Raghunathan. 2023. Understanding catastrophic forgetting in language models via implicit inference. arXiv preprint arXiv:2309.10105 (2023).
- [43] Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, and Percy Liang.

2022. Fine-tuning can distort pretrained features and underperform out-ofdistribution. arXiv preprint arXiv:2202.10054 (2022).

- [44] Deepak Kumar, Yousef Anees AbuHashem, and Zakir Durumeric. 2024. Watch your language: Investigating content moderation with large language models. In Proceedings of the International AAAI Conference on Web and Social Media, Vol. 18. 865–878.
- [45] Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. 2023. Cmmlu: Measuring massive multitask language understanding in chinese. arXiv preprint arXiv:2306.09212 (2023).
- [46] Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. 2023. Halueval: A large-scale hallucination evaluation benchmark for large language models. arXiv preprint arXiv:2305.11747 (2023).
- [47] Miao Li, Ming-Bin Chen, Bo Tang, Shengbin Hou, Pengyu Wang, Haiying Deng, Zhiyu Li, Feiyu Xiong, Keming Mao, Peng Cheng, et al. 2024. NewsBench: a systematic evaluation framework for assessing editorial capabilities of large language models in chinese journalism. arXiv preprint arXiv:2403.00862 (2024).
- [48] Zhizhong Li and Derek Hoiem. 2017. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence 40, 12 (2017), 2935–2947.
- [49] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 (2024).
- [50] Sebastian Lubos, Thi Ngoc Trang Tran, Alexander Felfernig, Seda Polat Erdeniz, and Viet-Man Le. 2024. LLM-generated explanations for recommender systems. In Adjunct Proceedings of the 32nd ACM Conference on User Modeling, Adaptation and Personalization. 276–285.

- [51] Hanjia Lyu, Song Jiang, Hanqing Zeng, Yinglong Xia, Qifan Wang, Si Zhang, Ren Chen, Christopher Leung, Jiajie Tang, and Jiebo Luo. 2023. Llm-rec: Personalized recommendation via prompting large language models. arXiv preprint arXiv:2307.15780 (2023).
- [52] MAA. 2025. American Invitational Mathematics Examination - AIME. American Invitational Mathematics Examination - AIME 2025 (2025). https://maa.org/mathcompetitions/american-invitational-mathematics-examination-aime
- [53] Zijie Meng, Jin Hao, Xiwei Dai, Yang Feng, Jiaxiang Liu, Bin Feng, Huikai Wu, Xiaotang Gai, Hengchuan Zhu, Tianxiang Hu, et al. 2025. DentVLM: A Multimodal Vision-Language Model for Comprehensive Dental Diagnosis and Enhanced Clinical Practice. arXiv preprint arXiv:2509.23344 (2025).
- [54] Mistral-AI. 2024. Un Ministral, des Ministraux. https://mistral.ai/news/ministraux.

- Accessed: 2024-10-16.

[55] Mistral-AI. 2025. Mistral Small 3.1. https://mistral.ai/news/mistral-small-3-1.

- Accessed: 2025-03-17.

- [56] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems 35 (2022), 27730–27744.
- [57] Boyu Qiao, Kun Li, Wei Zhou, Shilong Li, Qianqian Lu, and Songlin Hu. 2025. BotSim: LLM-Powered Malicious Social Botnet Simulation. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 14377–14385.
- [58] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems 36 (2023), 53728–53741.
- [59] Vinay Venkatesh Ramasesh, Aitor Lewkowycz, and Ethan Dyer. 2021. Effect of scale on catastrophic forgetting in neural networks. In International conference on learning representations.
- [60] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. 2024. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling.
- [61] Nidhish Shah, Zulkuf Genc, and Dogu Araci. 2024. Stackeval: Benchmarking llms in coding assistance. Advances in Neural Information Processing Systems 37

(2024), 36976–36994.

- [62] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024).
- [63] Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed Amin, Le Hou, Kevin Clark, Stephen R Pfohl, Heather Cole-Lewis, et al. 2025. Toward expert-level medical question answering with large language models. Nature Medicine 31, 3 (2025), 943–950.
- [64] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. 2022. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261 (2022).
- [65] Ross Taylor, Marcin Kardas, Guillem Cucurull, Thomas Scialom, Anthony Hartshorn, Elvis Saravia, Andrew Poulton, Viktor Kerkez, and Robert Stojnic. 2022. Galactica: A large language model for science. arXiv preprint arXiv:2211.09085 (2022).
- [66] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 (2023).
- [67] Soroush Vosoughi, Deb Roy, and Sinan Aral. 2018. The spread of true and false news online. science 359, 6380 (2018), 1146–1151.
- [68] Qianlong Wang, Keyang Ding, Bin Liang, Min Yang, and Ruifeng Xu. 2023. Reducing spurious correlations in aspect-based sentiment analysis with explanation from large language models. In Findings of the Association for Computational Linguistics: EMNLP 2023. 2930–2941.
- [69] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. 2024. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems 37 (2024), 95266– 95290.
- [70] Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann.

2023. Bloomberggpt: A large language model for finance. arXiv preprint arXiv:2303.17564 (2023).

- [71] Feng Xia, Li Liu, Jie Li, Jianhua Ma, and Athanasios V Vasilakos. 2013. Socially aware networking: A survey. IEEE Systems Journal 9, 3 (2013), 904–921.
- [72] Weiwen Xu, Hou Pong Chan, Long Li, Mahani Aljunied, Ruifeng Yuan, Jianyu Wang, Chenghao Xiao, Guizhen Chen, Chaoqun Liu, Zhaodonghui Li, et al.

2025. Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning. arXiv preprint arXiv:2506.07044 (2025).

- [73] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025).
- [74] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. 2024. Qwen2. 5 Technical Report. CoRR (2024).
- [75] Yi Yang, Yixuan Tang, and Kar Yan Tam. 2023. Investlm: A large language model for investment using financial domain instruction tuning. arXiv preprint arXiv:2309.13064 (2023).
- [76] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. 2025. Dapo: An opensource llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476

(2025).

- [77] Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. 2023. Rrhf: Rank responses to align language models with human feedback without tears. arXiv preprint arXiv:2304.05302 (2023).
- [78] Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, et al. 2025. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471

(2025).

- [79] Jingying Zeng, Richard Huang, Waleed Malik, Langxuan Yin, Bojan Babic, Danny Shacham, Xiao Yan, Jaewon Yang, and Qi He. 2024. Large language models for social networks: Applications, challenges, and solutions. arXiv preprint arXiv:2401.02575 (2024).
- [80] Dan Zhang, Ziniu Hu, Sining Zhoubian, Zhengxiao Du, Kaiyu Yang, Zihan Wang, Yisong Yue, Yuxiao Dong, and Jie Tang. 2024. Sciglm: Training scientific language models with self-reflective instruction annotation and tuning. arXiv preprint arXiv:2401.07950 (2024).
- [81] Wenxuan Zhang, Yue Deng, Bing Liu, Sinno Jialin Pan, and Lidong Bing. 2023. Sentiment analysis in the era of large language models: A reality check. arXiv preprint arXiv:2305.15005 (2023).
- [82] Xiaotian Zhang, Chunyang Li, Yi Zong, Zhengyu Ying, Liang He, and Xipeng Qiu. 2023. Evaluating the performance of large language models on gaokao benchmark. arXiv preprint arXiv:2305.12474 (2023).
- [83] Fei Zhao, Chonggang Lu, Yue Wang, Zheyong Xie, Ziyan Liu, Haofu Qian, JianZhao Huang, Fangcheng Shi, Zijie Meng, Hongcheng Guo, et al. 2025. RedOne: Revealing Domain-specific LLM Post-Training in Social Networking Services. arXiv preprint arXiv:2507.10605 (2025).
- [84] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911 (2023).

Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009

