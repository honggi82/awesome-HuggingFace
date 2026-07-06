# arXiv:2507.14683v1[cs.CL]19Jul2025

MIROMIND-M1: AN OPEN-SOURCE ADVANCEMENT IN MATHEMATICAL REASONING VIA CONTEXT-AWARE MULTISTAGE POLICY OPTIMIZATION

Xingxuan Li∗, Yao Xiao∗, Dianwen Ng∗, Hai Ye∗, Yue Deng∗, Xiang Lin∗, Bin Wang∗ Zhanfeng Mo, Chong Zhang, Yueyi Zhang, Zonglin Yang, Ruilin Li, Lei Lei Shihao Xu, Han Zhao, Weiling Chen, Feng Ji, Lidong Bing†

MiroMind AI

[Figure 1]

https://github.com/MiroMindAsia/MiroMind-M1 https://huggingface.co/miromind-ai/MiroMind-M1-RL-7B https://huggingface.co/datasets/miromind-ai/MiroMind-M1-RL-62K https://miromind.ai/

[Figure 2]

[Figure 3]

[Figure 4]

ABSTRACT

Large language models have recently evolved from fluent text generation to advanced reasoning across diverse domains, giving rise to reasoning language models (RLMs). Among these domains, mathematical reasoning serves as a representative benchmark as it requires precise multi-step logic and abstract reasoning, which can be generalized to other tasks. While closed-source RLMs such as GPT-o3 and Claude Sonnet 4 demonstrate impressive reasoning capabilities, their proprietary nature limits transparency and reproducibility. Although many open-source projects aim to close this gap, most of them lack sufficient openness by omitting critical resources such as curated datasets and detailed training configurations, which hinders reproducibility. To contribute toward greater transparency in RLM development, we introduce the MiroMind-M1 series, a set of fully open-source RLMs built on the Qwen-2.5 backbone that match or exceed the performance of existing open-source RLMs. Specifically, our models are trained in two stages: supervised fine-tuning (SFT) on a carefully curated corpus of 719K math-reasoning problems with verified chain-of-thought (CoT) trajectories, followed by reinforcement learning with verifiable reward (RLVR) on 62K challenging and verifiable problems. To enhance the robustness and efficiency of the RLVR process, we introduce Context-Aware Multi-Stage Policy Optimization (CAMPO), an algorithm that integrates length-progressive training with an adaptive repetition penalty to encourage context-aware RL training. Our model achieves state-of-the-art or competitive performance and superior token efficiency among Qwen-2.5-based open-source 7B and 32B models on the AIME24, AIME25, and MATH benchmarks. To facilitate reproducibility, we release the complete stack: models (MiroMind-M1-SFT-7B, MiroMind-M1-RL-7B, MiroMind-M1-RL-32B); datasets (MiroMind-M1-SFT-719K, MiroMind-M1-RL-62K); and all training and evaluation configurations. We hope these resources will support further research and foster community advancement.

∗ Equal contribution. † Corresponding author: Lidong Bing <lidong.bing@miromind.ai>

CONTENTS

- 1 Introduction 3
- 2 Related Work 4

- 2.1 Reasoning Language Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.2 Supervised Fine-Tuning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 2.3 Reinforcement Learning with Verfiable Rewards . . . . . . . . . . . . . . . . . . . . . . . . 5

- 3 Enhancing Math Reasoning Capabilities of Language Models via Supervised Fine-Tuning 6

- 3.1 Data Curation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 3.1.1 Data Collection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 3.1.2 Data Processing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 3.2 Supervised Fine-tuning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 3.2.1 Main Experiment Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.2.2 Insights . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

- 4 Boosting Reasoning Performance and Efficiency with Reinforcement Learning 9

- 4.1 Data Collection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 4.2 Preliminaries of Reinforcement Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 4.3 Context-Aware Multi-Stage Policy Optimization (CAMPO) . . . . . . . . . . . . . . . . . . 12

- 4.3.1 Efficiency-Aware Perspective on Multi-Stage Training . . . . . . . . . . . . . . . . 13
- 4.3.2 Stabilizing Training with Repetition Penalty . . . . . . . . . . . . . . . . . . . . . . 14
- 4.3.3 Accurate Verifier for Efficient Reasoning . . . . . . . . . . . . . . . . . . . . . . . 15

- 4.4 MiroMind-M1-RL-32B . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 4.4.1 Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 4.4.2 Experimental Results and Insights . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 4.5 MiroMind-M1-RL-7B . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- 4.5.1 Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 4.5.2 Experimental Results and Insights . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- 5 Conclusions 21

- 1 INTRODUCTION

Recent advances in large language models (LLMs), driven by the Transformer architecture (Vaswani et al., 2017), massive pre-training (Kaplan et al., 2020; Kavukcuoglu, 2025; Dubey et al., 2024), and emergent abilities such as in-context learning (Brown et al., 2020; Dong et al., 2024), have led to strong performance in planning, reasoning, and problem solving (Valmeekam et al., 2023; Feng et al., 2025; Xu et al., 2025a; Yao et al., 2023). Among these capabilities, structured multi-step reasoning has captured significant interest, shifting the focus from fluent generation to cognitively grounded chain-of-though (CoT) reasoning (Wei et al., 2022a;b). Building on this momentum, researchers have developed reasoning language models (RLMs), specialized LLMs explicitly trained to produce multi-step CoT for complex reasoning tasks. This approach has led to significant performance improvements across various domains, including mathematics (Shao et al., 2024; Luo et al., 2025b), question answering (Wei et al., 2024a;b), planning (Wei et al., 2025), and code generation (Luo et al., 2025a; Liu & Zhang, 2025; Xu et al., 2025c). Specifically, many studies (Shao et al.,

- 2024; Luo et al., 2025b; He et al., 2025a; Hu et al., 2025; Wen et al., 2025b) focus on mathematical reasoning as it demands robust multi-step logical thinking and abstract problem-solving skills, which are fundamental to advanced reasoning in language models.

While RLMs have achieved unprecedented performance, recent progress has been dominated by a handful of proprietary, closed-source models (OpenAI, 2024; Anthropic, 2024; Kavukcuoglu, 2025), the training data, methodologies, and evaluation protocols of these commercial systems remain largely unclear, creating significant barriers to transparency, reproducibility, and further scientific innovation. As a result, the research community is urgently seeking open and reproducible pathways for building high-performing RLMs. In response, a series of open-source initiatives have emerged, demonstrating that state-of-the-art reasoning capabilities can be achieved through a two-stage training paradigm: supervised fine-tuning (SFT) on curated reasoning traces, followed by reinforcement learning with verifiable rewards (RLVR) (Guo et al., 2025; Kimi-Team et al., 2025; Team, 2025b; Mistral-AI et al., 2025). In the SFT stage, the pretrained model is fine-tuned on curated datasets of annotated QA pairs and verified CoT traces, enabling it to learn step-by-step reasoning through imitation. In the subsequent RLVR stage, the model is further optimized on reasoning tasks using reinforcement learning to maximize a task-specific reward signal. This process encourages the exploration of more effective and robust CoT patterns. Rewards are typically predefined and computed by an external verifier, with higher scores assigned to correct answers and lower scores to incorrect ones.

However, despite the overall training paradigm being disclosed by these open-source projects, reproducing high-performing reasoning models in practice remains challenging due to missing or insufficiently documented implementation details. In the SFT stage, key components such as data collection and curation strategies, training data composition, curriculum design, and configuration details are often under-specified or ambiguously described. Similarly, in the RLVR stage, key aspects such as sampling strategies, data composition, and reward function implementations are often unclear or not publicly disclosed. The lack of transparency severely limits reproducibility and hinders further progress on improving reasoning capabilities.

To promote greater transparency in RLM development and advance cutting-edge research, we present a comprehensive and well-documented reproduction of reinforcement-incentivized RLMs. We specifically focus on training RLMs for mathematical reasoning, a representative and practical testbed to evaluate reasoning capabilities, where strong performance reliably reflects the ability of an LLM to generate logically consistent multi-step CoTs. As each LLM-generated answer can be rigorously verified to provide a clear reward signal, this domain presents an ideal setting for RLVR. Moreover, the abundance of open-source, well-annotated mathematical reasoning datasets greatly facilitates reproducibility.

This report details our reproduction of RLMs, covering key aspects such as training dataset curation, opensource codebases, and the implementation and evaluation of both SFT and RLVR. We begin by carefully curating a high-quality SFT dataset, drawing from publicly available sources such as OpenR1 (Face, 2025), OpenThoughts (Team, 2025a), Light-R1 (Wen et al., 2025b), and Synthetic-1 (Mattern et al., 2025). To ensure

data quality and enable fair comparisons, we perform rigorous deduplication and decontamination, removing duplicate samples and filtering out any overlap with evaluation benchmarks such as AIME24, AIME25, and MATH500 (Mathematical Association of America, n.d.; Lightman et al., 2023). Through systematic ablation studies on the impact of dataset properties on SFT performance, we find that incorporating longer and more complex reasoning trajectories consistently leads to substantial improvements. This highlights the importance of trajectory depth and semantic richness in CoT reasoning traces for enhancing mathematical reasoning through SFT. Motivated by this, we propose CAMPO, a context-aware RLVR algorithm that promotes progressively longer CoT reasoning traces via a length-based curriculum. Specifically, CAMPO employs a multi-stage training strategy with progressively increasing context lengths and a dynamic repetition penalty to reduce redundancy and improve training stability. Furthermore, we open-source our rigorous data curation pipeline for RLVR and release an improved, more robust math verifier to support a more stable and effective RLVR training process. Leveraging the Qwen2.5 base models (Yang et al., 2024), we develop the MiroMind-M1 series of mathematical RLMs trained on carefully curated datasets through a two-stage process of SFT and RLVR, employing CAMPO during the RLVR stage. Comprehensive evaluations show that our MiroMind-M1-RL achieves state-of-the-art or competitive performance and token efficiency among Qwen2.5-based open-source models on mathematical benchmarks such as AIME24, AIME25, and MATH500. In summary, our main contributions are listed as follows:

- • We release a high-quality dataset combining complicated mathematical reasoning problems (MiroMind-M1-RL-62K) and long-form reasoning trajectories (MiroMind-M1-SFT-719K) constructed based on openly available data sources.
- • We introduce CAMPO, a context-aware reinforcement learning framework with staged context expansion and repetition penalties for training high-performing RLMs.
- • We present the MiroMind series, including MiroMind-SFT-7B, a strong supervised baseline, and MiroMind-M1-RL, which achieves state-of-the-art or competitive performance among fully open-sourced 7B or 32B math reasoning models.
- • We open-source the full stack, datasets, models, code, and an improved robust verifier, to enable reproducible math reasoning research on mathematical RLMs. We share key insights and lessons learned from data curation and the SFT-RLVR process to advance future research in RLMs.

- 2 RELATED WORK

- 2.1 REASONING LANGUAGE MODELS

RLMs have become a promising direction for advancing large language models. Several leading proprietary models, such as OpenAI’s o-series (OpenAI, 2025), Anthropic’s Claude (Anthropic, 2024), Gemini 2.5 Pro (Kavukcuoglu, 2025), Seed (Seed, 2025), and Magistral (Mistral-AI et al., 2025), place a strong emphasis on structured multi step reasoning. Recent models including DeepSeek-R1 (e.g., R1, R1-0528) (Guo et al., 2025), Qwen (QwQ, Qwen3) (Team, 2025c;b), and Gemma (Team et al., 2024) have partially disclosed their training pipelines, providing insight into the emerging paradigm for reasoning enhancement.

Mathematics has become the primary benchmark for evaluating RLMs, due to its objective verifiability and controllable difficulty. High-level benchmarks such as AIME24, AIME25, and MATH500 remain unsolved and have become central to recent progress. To tackle these challenges, most RLMs follow a two-stage training pipeline: SFT on curated datasets with verified CoT traces, followed by reinforcement learning RLVR.

Although methods such as GRPO (Shao et al., 2024) and reward-based optimization have been adopted in models like DeepSeek-R1 and Qwen, many training details (e.g., data composition, reward design, and sampling strategies) remain undisclosed. This has prompted numerous open-source replication efforts (e.g.,

AM (Ji et al., 2025a), Light-R1 (Wen et al., 2025b), Open-R1 (Face, 2025)) to explore and refine best practices. Notably, general models trained with SFT and RLVR often outperform those trained primarily on math (e.g., Qwen2.5-Math), underscoring the value of verifiable feedback and broad reasoning supervision.

- 2.2 SUPERVISED FINE-TUNING

The success of DeepSeek distilled models (Guo et al., 2025) demonstrates that SFT on massive data can equip LLMs with advanced reasoning capabilities. On the one hand, many replication works make efforts to collect large-scale supervised datasets from publicly available resources of various domains for fine-tuning reasoning language models. The OpenThoughts datasets (Guha et al., 2025) curates a synthetic reasoning dataset with 114k examples, covering multiple domains including math, science, coding, and puzzles. The CoTs are generated by DeepSeek-R1 and verified. They further conduct comprehensive ablations to find the best question resources and scale up their dataset to 1.2M with verified CoTs from QwQ-32B. AM-DeepSeekR1-Distilled-1.4M (Zhao et al., 2025) is an another compelling dataset with 1.4M samples covering math, code, science, general chat, instruction following and dialogue. Among these, 500k samples are cleaned from other open-source datasets, and the other 900k samples are distilled from the DeepSeek-R1-671B, filtered based on quality and assigned with difficulty levels. Mixture-of-Thoughts (Face, 2025) collects 350k verified reasoning traces distilled from DeepSeek-R1, spanning tasks in math, code, and science. This dataset is known as designed to teach language models to reason step-by-step. Synthetic-1 (Mattern et al., 2025) constructs 894k samples distilled from DeepSeek-R1, covering math, code and STEM domains. All these works emphasized the importance of data cleaning, deduplication and verification. For verification, besides from rule-based verifiers for math problems and unit-test-based validation for code problems, these works also use LLM-as-judge (Guha et al., 2025; Mattern et al., 2025) and reward model evaluation (Zhao et al., 2025) to verify the samples from general domains.

On the other hand, several open-sourced models have already achieved strong performance on verifiable reasoning tasks, especially math problem solving. OpenR1-Qwen-7B (Face, 2025) leverages 220k math problems with high quality long CoTs to achieve comparable performance towards the corresponding DeepSeek distilled model. Light-R1 (Wen et al., 2025c) constructs a two-stage SFT training pipeline to boost its performance. In the first stage, the model leverages around 76k difficult and diverse samples with verified long-form CoT responses to reproduce a similar training procedure to DeepSeek distilled models. In the second stage, the model leverages a filtered subset of 3k samples which are not completely tackled by DeepSeek-R1-Distill-Qwen-32B and DeepSeek-R1 to further improve model performance. AM-Thinking-v1 (Ji et al., 2025b) takes advantage of larger learning rate and batch size during SFT to get rid of SFT pattern shifts (Tian et al., 2025). These advancements highlight a promising trend that open-source models can excel in complex reasoning tasks by carefully configuring on data curation and training procedure.

- 2.3 REINFORCEMENT LEARNING WITH VERFIABLE REWARDS

RLVR has emerged as the most promising paradigm for advancing the reasoning abilities of language models, distinguished by its comprehensive approach to data quality, algorithmic innovation, reward design, and training methodology (Zhang et al., 2025). At the core of RLVR are high-quality and verifiable datasets, including DeepScaleR (Luo et al., 2025b), Skywork OR1 (He et al., 2025a), Open Reasoner Zero (Hu et al., 2025), DeepMath (He et al., 2025b), among others. These datasets are meticulously curated to ensure a suitable balance of medium difficulty, thorough removal of benchmark leakage, comprehensive crossreferencing of answers, and exclusion of uncertain or ambiguous cases. Such rigorous data preparation guarantees both verifiability and maximal learning effectiveness.

On the algorithmic side, RLVR has progressed beyond foundational methods such as PPO (Schulman et al., 2017) and GRPO (Shao et al., 2024), incorporating specialized variants including DAPO (Yu et al., 2025), Dr. GRPO (Liu et al., 2025), VC-PPO (Yuan et al., 2025), VAPO (Yue et al., 2025), and others. These algorithms

are crafted to address specific challenges like length bias, difficulty imbalance, and value function management for long reasoning chains, while also promoting training stability. In addition, reward engineering plays a critical role by providing clear and verifiable signals, such as correctness, adherence to answer formats, and penalties for inefficient or excessively long responses (Zhang et al., 2025). These mechanisms help to minimize reward hacking and foster robust, stepwise reasoning.

With respect to training strategies, RLVR adopts curriculum learning by gradually increasing the complexity or length of tasks, which supports the systematic acquisition of reasoning skills (Zhang et al., 2025). Techniques such as KL loss regularization—sometimes selectively omitted to encourage exploration—and rollout pruning are also employed to discard low-value trajectories (Xu et al., 2025b), thereby improving both training efficiency and stability.

By systematically integrating advances in data, algorithms, rewards, and training practices, RLVR provides a clear and scalable path toward reliable and high-performing chain of thought generation in large language models.

- 3 ENHANCING MATH REASONING CAPABILITIES OF LANGUAGE MODELS VIA SUPERVISED FINE-TUNING

In this section, we explore how to construct high-quality SFT data capable of replicating or even surpassing the performance of the DeepSeek-R1-distilled models (Guo et al., 2025) for math reasoning. We detail methods for sourcing and processing data. In addition, we share insights and lessons learned throughout the data construction and SFT training process.

- 3.1 DATA CURATION

- 3.1.1 DATA COLLECTION

We focus mainly on enhancing the math reasoning capabilities of LLMs in this work, so we only collect the R1 distillation data from the math domain. We collect data from four public sources which are OpenR1 (Face, 2025), Open-thoughts (Guha et al., 2025), Light-R1 (Wen et al., 2025a), and Synthetic-1 (Mattern et al., 2025). We display the detailed data statistics of these datasets in Table 1.

OpenR1. OpenR1 (Face, 2025) sources approximately 400K math problems from AI-MO/NuminaMath-1.5. Each question includes between 1 and 8 reasoning traces from DeepSeek-R1. We retain all traces with correct final answers, verified by MathVerify (Kydlicek, 2024) and an LLM-based judge.

OpenThoughts. Unlike OpenR1, OpenThoughts (Guha et al., 2025) includes questions from diverse domains such as math, science, coding and puzzles. It contains 114K questions with reasoning traces generated by DeepSeek-R1. We retained only math-related questions, resulting in approximately 56K question-response pairs.

Light-R1. Light-R1 (Wen et al., 2025b) sources math questions from OpenR1-Math-220K, OpenThoughts114K, LIMO, OpenMathInstruct-2, s1K, Omni-MATH, Hendrycks_Math, and AIME (up to 2023). The dataset is decontaminated against common reasoning benchmarks such as AIME2024, AIME2025, MATH500, and GPQA Diamond. Reasoning traces are generated using DeepSeek-R1. The final datasets include 76k and 3k challenging problems for stage 1 and stage 2 training, respectively.

Synthetic-1. Synthetic-1 (Mattern et al., 2025) is a large-scale dataset for supervised fine-tuning, featuring reasoning traces produced by DeepSeek-R1. It covers a range of domains including Math, Algorithmic Coding, Real-World Software Engineering, Open-Ended STEM Q&A, and Synthetic Code Understanding. The math questions are derived from NuminaMath and undergo LLM-driven postprocessing to transform

#### Dataset # Questions # Traces # Traces used in SFT Category

OpenR1 191k 418k 418k math Open-thoughts 56k 56k 56k math Light-R1 75k 76k 76k math Synthetic-1 362k 638k 247k math Ours 412k 719k 719k math

- Table 1: Statistics of math-focused datasets used for SFT, including the number of prompts, total reasoning traces, and traces selected for training.

multiple-choice items into free-form questions and remove those without automatically verifiable answers (e.g., proof-based queries). For our work, we focus only on math-related problems, which results in 247k problems.

- 3.1.2 DATA PROCESSING

Data Deduplication To ensure data quality, we first remove duplicate question-response pairs by computing N-gram overlaps on the concatenated query and response text. Samples with significant overlap are excluded to reduce redundancy in the training data. Note that we allow one question to have multiple correct responses.

Data Decontamination To avoid data contamination and ensure fair evaluation, we perform decontamination for training data with respect to our target evaluation sets. Specifically, we apply an Ngram overlap filter to remove any training samples whose questions match those in Math500, AIME24, or AIME25. This filtering effectively prevents data leakage.

Data Statistics of the Final Training Set We end up with 719K SFT training samples shown in Table 1. We demonstrate the length distribution of thinking tokens in Figure 1 , together with Open-R1, Synthetic-1, Open-Thoughts and Light-R1.

0.0005

Ours OpenThoughts Synthetic OpenR1 LightR1

| |
|---|

0.0004

| |
|---|

| |
|---|

0.0003

Density

| |
|---|

0.0002

0.0001

0.0000

0 5000 10000 15000 20000 25000 30000

Lens

Figure 1: Length distribution for of datasets.

- 3.2 SUPERVISED FINE-TUNING

Experimental Setup. We train our models for 3 epochs using a peak learning rate of 5.0 × 10−5 with a cosine learning rate scheduler. We set the warmup step ratio to 10% and use a batch size of 128. To support long generations of complex reasoning, we increase model’s max_position_embeddings to 32,768 using Linear RoPE scaling. We adopt a no-packing strategy for training because of its empirical advantages. More details on why we employ no-packing can be found in Section 3.2.2. Our implementation is based on LlamaFactory (Zheng et al., 2024), with a cutoff length of 26,000. We follow Deepseek-R1 (Guo et al., 2025) to employ Qwen-2.5-Math-7B as the initial checkpoint.

- 3.2.1 MAIN EXPERIMENT RESULTS

For evaluation, we report avg@k to ensure stable and reliable results, setting k = 64 for AIME24 and AIME25, and k = 5 for MATH-500. For evaluation, we use a maximum generation length of 32,768 tokens, a sampling temperature of 0.6, and a top-p value of 0.95.

We report our results in Table 2. Our model achieves 60.5 on AIME24, 45 on AIME25, and 94.6 on MATH-500. Across all three tasks, our model consistently surpasses other SFT models of the same size, demonstrating the quality of our constructed dataset. In particular, our model also outperforms a recent SFT model, MiMo-7B-SFT (Xiaomi LLM-Core Team, 2025).

Project Initial Checkpoint AIME24 AIME25 MATH500

DeepSeek-R1 (Guo et al., 2025) Qwen2.5-Math-7B 55.5 40.4† 92.8 OpenThoughts (Team, 2025a) Qwen2.5-7-Instruct 31.3 23.3 83.2 Open-R1 (Face, 2025) Qwen2.5-Math-7B-Instruct 36.7 40.0 90.6 Synthetic-1 (Mattern et al., 2025) Qwen-2.5-7B-Instruct 30.0 26.6 85.6 MiMo-7B-SFT (Xiaomi LLM-Core Team, 2025) MiMo-7B-Base 58.7 44.3 93.0

MiroMind-SFT-7B Qwen-2.5-Math-7B 60.4 45.0 94.6

- Table 2: We report results of our model on math reasoning benchmarks. It outperforms the distilled models trained from the same initial checkpoint by Deepseek. † means that the score of DeepSeek-R1 on AIME25 is from our evaluation.

- 3.2.2 INSIGHTS

In this section, we show some experiences obtained in the data curation and training process. We first elaborate on why we use no-packing for training. We also show that trajectory length is a simple but effective metric for sample selection. We finally present a strategy to mix packing and no-packing to improve training efficiency while preserving performance. The training of these ablations follows Section 3.2, unless otherwise specified. And experiments are based on Qwen2.5-Math-7B-Instruct and default split of OpenR1-Math-220k 1.

No-packing outperforms packing. In this part, we first clarify the difference between multiple packing strategies. Instead of training each sequence individually as no-packing, packing concatenates multiple sequences into a single input to maximize the usage of GPU memories, which significantly improves the training efficiency. However, this packing strategy might introduce cross-sample attention pollution, where tokens from different sequences within the same packed input may inadvertently attend to each other. To address this, neat-packing are usually employed during training to ensure that each token can only attend to tokens within its own original sequence. We subsequently compared the performance of these strategies. We report results in Table 3. We observe that no-packing outperforms packing and neat-packing 2.

Strategy AIME24 AIME25 MATH500

Packing 35.41 26.66 89.06 Neat-packing 32.50 26.25 88.80 No-packing 38.12 29.37 90.40

- Table 3: No-Packing usually leads to better performance.

# Data Method AIME24 AIME25 MATH500

Random 29.58 23.12 86.93 Long 35.21 24.79 88.66

30k

Random 31.66 24.58 89.93 Long 35.41 30.21 89.80

50k

Table 4: Training on the data points that have long trajectories obtains better performance.

- 1We select Qwen2.5-Math-7B-Instruct because it can achieve better performance than other models(e.g. Qwen-2.5Math-7B) on Open-R1 dataset.
- 2Note that these results are based on LlamaFactory’s implementation, which uses a knapsack algorithm to minimize padding by packing as many sequences as possible. This approach may violate the i.i.d. assumption in training, potentially contributing to the observed inferior performance of the packing and neat-packing strategies.

Response length is a strong metric for sample selection. In the data preparation process, we find that training on longer trajectories usually leads to better performance. To support the finding, we select a subset of data samples from the full dataset for training here. We primarily consider two selection strategies: random and long. For the random approach, we randomly sample data points from the dataset. For the long approach, we select the longest samples. In both cases, we ensure that the number of selected samples remains identical. We find that long consistently outperforms random with different sample scales. We report results in Table 4. In almost all cases, models training on samples selected by long outperform those of random. The reason may be that longer trajectories are more likely to arise from complex questions that can better satisfy challenging math tasks such as AIME.

Packing followed by no-packing can save time without significant performance drop. Although the no-packing strategy can produce better performance, it is less efficient compared to the other two methods. To balance performance and training efficiency, we attempt to first train our models with packing for the first two epochs, and then switch to no-packing for the final epoch. The performance on math 500 is 90.4 and 89.6, respectively. This approach allows us to leverage the benefits of no-packing while reducing the overall training time.

- 4 BOOSTING REASONING PERFORMANCE AND EFFICIENCY WITH REINFORCEMENT LEARNING

[Figure 5]

Figure 2: Initial composition distribution. Big-Math comprises HARP and reformulated machine outputs (Albalak et al., 2025); Skywork-OR1-RL-Data (He et al., 2025a) contains only maths.

[Figure 6]

Figure 3: Final composition distribution. Data are filtered using the inclusion-exclusion criteria described in Section 4.1 and the illustration shown in Figure 4.

Recent studies have demonstrated that RL can enhance reasoning performance in both base (Yu et al.,

- 2025; Wen et al., 2025c) and R1-distilled LLMs (Luo et al., 2025b; He et al., 2025a). A key observation is that performance gains are often accompanied by increased response length. However, longer output may introduce redundancy and unnecessary repetition (Xie et al., 2025), suggesting that efficiency is an equally important yet under-explored aspect. In this section, we introduce the MiroMind-M1-RL model series, including MiroMind-M1-RL-32B and MiroMind-M1-RL-7B. MiroMind-M1-RL not only improves performance but also enhances efficiency in mathematical reasoning. We begin by describing the data

[Figure 7]

Figure 4: Inclusion-Exclusion Criteria. Overview of the filtering strategy used to construct the final training dataset, consisting of 62,118 problems selected from an initial pool of 1M candidates drawn from four different data sources. The filtering process, which applies criteria such as de-duplication, difficulty-based pruning, and verifiability constraints, results in the exclusion of approximately 94% of the original data.

preparation process in Section 4.1, including the selection criteria. Section 4.2 provides preliminaries on RLbased training. Section 4.3 introduces the context-aware multi-stage policy optimization (CAMPO) algorithm used to train the MiroMind-M1-RL model series, along with an analysis of each of its components. Section 4.4 and 4.5 present the MiroMind-M1-RL-32B and MiroMind-M1-RL-7B models, respectively.

- 4.1 DATA COLLECTION

To enhance mathematical reasoning capabilities, we incorporate data from diverse sources to ensure broader coverage and increased complexity of mathematical topics. Our dataset includes NuminaMath-1.5 (LI et al., 2024) with 896K problems; Skywork-OR1-RL-Data (He et al., 2025a), containing a 105K-size math subset; Big-Math (Albalak et al., 2025), comprising 50K problems from both reformulated content and HARP (Yue et al., 2024); and DAPO-Math-17K (Yu et al., 2025), a carefully curated collection of 17K high-quality math problems. These datasets constitute the initial selection of our training candidates, forming a total pool of approximately 1M examples, as illustrated in Figure 2.

Despite the dataset’s broad coverage of mathematical topics, rigorous curation is necessary to maintain the quality standards needed for successful reinforcement learning. Some problems, such as those requiring long, free-form answers or detailed mathematical proofs, are difficult for rule-based verifiers to evaluate accurately. In these cases, even correct responses may be mistakenly marked as incorrect due to limitations in the verification logic. Other problems involve ambiguous or incomplete answer formats, especially when the final answers are lengthy or contain multiple valid interpretations. These cases may result in inconsistent labeling, where partially correct answers are penalized despite capturing the key idea. Such misjudgments introduce conflicting signals during training. That is, the model may receive negative feedback for answers that are actually valid, or positive feedback for flawed ones. These misleading updates not only hinder the model’s ability to generalize but also make learning unstable and, in some cases, cause the model to collapse. To mitigate these issues, we perform a series of filtering steps to refine the candidate set, as part of the overall

data processing pipeline outlined in Figure 4 and explaining the rationale below, which depicts our selection criteria.

- 1. Filter by style/format: We begin by filtering the collected math problems based on style and format, removing proof-based questions that are non-verifiable. These tend to be open-ended and too complex for consistent evaluation using rule-based methods. To ensure linguistic consistency, we also restrict the dataset to English-language problems, as our current study does not explore the effects of multilingual learning.
- 2. Filter by duplicates: To reduce redundancy and improve the diversity of training signals, we perform deduplication in two stages. First, we remove math problems that are exact duplicates. Second, we eliminate near-duplicates by applying a 10-gram similarity threshold, which helps filter out semantically similar problems that may differ only in surface form. This ensures a cleaner dataset that better supports generalization during training.
- 3. Filter by difficulty: Prior studies (Yu et al., 2025; He et al., 2025a; Wen et al., 2025c) have shown that not all training examples are equally beneficial for learning. This challenge becomes more prominent in the GRPO framework, where batches consisting entirely of correct or incorrect responses result in zero advantage, leading to uninformative gradient updates. To mitigate this, we conduct an offline difficulty-aware filtering strategy that adapts to the model’s current performance. For each math problem, we generate five rollouts using Deepseek-R1-Distill-Qwen-32B (with temperature set to 1.0 and a maximum output length of 32K tokens). We exclude extreme cases, such as those with fully correct or fully incorrect responses, and instead focus on problems that are neither trivial nor unsolvable. This targeted selection increases the likelihood of generating informative learning signals during RL training. Additionally, we find that including relatively simple problems (e.g., those with a pass rate of 0.8) is beneficial. These examples reinforce the model’s existing knowledge and help prevent divergence during training. This stabilizing effect is particularly valuable when KL regularization is not applied, as such problems act as anchors that support steady learning. In contrast, always presenting harder problems with lower success rates may overly emphasize exploration, potentially leading the model away from already-correct reasoning patterns. For datasets such as Skywork-OR1-RL, which include offline difficulty estimates for each problem relative to the performance of Deepseek-R1-Distill-Qwen-32B (measured over 16 independent rollouts), we also exclude both ends of the spectrum, removing trivial (fully correct) and unsolvable (fully incorrect) cases from the training examples.
- 4. Filter by character length of verifiable answer: Similar to proof-based questions, some problems require sentence-form answers that are evaluated using exact string matching. This introduces high variance in rule-based verification, as even minor differences in phrasing can lead to incorrect assessments. Such variance may destabilize reinforcement learning, especially when it dominates the training signal. To mitigate this, we exclude target answers longer than 20 characters, as these are more prone to linguistic variability and harder to verify reliably.

Through the above process, we eliminate 94% of the candidates, yielding a curated training set of 62K math problems that are both challenging and reliably verifiable, as illustrated in Figure 3. In addition, we ensure that the training data is decontaminated from common evaluation benchmarks, including AIME 2024 and 2025, and MATH 500.

- 4.2 PRELIMINARIES OF REINFORCEMENT LEARNING

Let each data pair (q,a) be i.i.d from a distribution D, where q is a query and a is the ground-truth answer. Given an LLM policy πθ(·|·), let o be an LLM-generated response to q, and r(·,·) is a predefined reward function that quantifies whether the response o yields a. RL-based fine-tuning aims to maximize this expected

reward over D, i.e.,

J(πθ) ≜ E(q,a)∼DEo∼π

θ(·|q)[r(o,a)]. (1)

max

θ

Proximal policy optimization (PPO) (Schulman et al., 2017) is one of the most popular actor-critic RL algorithms for LLM policy optimization (Ouyang et al., 2022; Hu et al., 2025). PPO trains both the target LLM policy πθ (actor) and a value model Vϕ (critic), which estimates the quality of responses generated by πθ. The PPO objective is:

JPPO(πθ) ≜ E(q,a)∼D,{o

i}Gi=1∼πθold(·|q)

(2)

|oi|

G

1 G

1 |oi|

min ri,t(θ)Aˆi,t(ϕ),clip(ri,t(θ),1 − ε,1 + ε)Aˆi,t(ϕ) ,

t=1

i=1

## where ri,t(θ) ≜ πθ(oi,t|q,oi,<t)/πθ

(oi,t|q,oi,<t) is the likelihood ratio between current and old policies;

old

Aˆi,t(ϕ) is the gated advantage estimator (GAE) estimated from Vϕ(oi,t|q,oi,<t). Compared to raw reward signals, GAE yields more stable policy updates (Hu et al., 2025).

Group relative policy optimization (GRPO) (Shao et al., 2024) is an efficient PPO variant that eliminates the critic and GAE, reducing memory and computation costs. GRPO normalizes rewards within each rollout group to lower variance and combines likelihood ratio clipping with a KL-divergence penalty to keep πθ close to the initial SFT LLM. The GRPO objective is:

JGRPO(πθ) ≜ E(q,a)∼D,{o

i}Gi=1∼πθold(·|q)

(3)

|oi|

G

1 |oi|

1 G

min ri,t(θ)Aˆi,t, clip (ri,t(θ), 1 − ε, 1 + ε) Aˆi,t − βKL(πθ|πref)i,t ,

t=1

i=1

where Aˆi,t ≜ (r(oi,a) − mean({r(oi,a)}Gi=1))/std({r(oi,a)}Gi=1) is the group-normalized reward, and the KL term uses the K3 estimator (Schulman., 2020). GRPO enables efficient large-scale LLM policy optimization, promoting long-chain CoT patterns (Guo et al., 2025).

Decoupled clip and dynamic sampling policy optimization (DAPO) (Yu et al., 2025) identifies key issues in GRPO, including entropy collapse, training instability, and length bias from sample-level loss. To address these, DAPO proposes a new objective:

JDAPO(πθ) ≜ E(q,a)∼D,{o

i}Gi=1∼πθold(·|q)

|oi|

G

1

(4)

min ri,t(θ)Aˆi,t,clip(ri,t(θ),1 − εlow,1 + εhigh)Aˆi,t subject to 0 < |{oi |is_equivalent(a,oi)}| < G.

G i=1 |oi|

t=1

i=1

DAPO decouples ε into εlow and εhigh to prevent entropy collapse, enforces diverse rollouts for stable gradients, and averages loss over all tokens to remove length bias.

- 4.3 CONTEXT-AWARE MULTI-STAGE POLICY OPTIMIZATION (CAMPO)

We propose the context-aware multi-stage policy optimization (CAMPO) algorithm. CAMPO enhances training by incorporating awareness of both context length and content. Specifically, multi-stage training with gradually increasing context length improves efficiency, while a repetition penalty encourages redundancyaware learning and promotes more effective reasoning. Training progresses to the next stage once the response

Algorithm 1 Context-Aware Multi-Stage Policy Optimization (CAMPO) Require: initial policy model πθ; reward function r(·,·); repetition critic f; clip ratio models ϕlow,ϕhigh;

task prompts D

- 1: for stage = 1,...,M do
- 2: Update the max context length for rollout
- 3: Sample εlow,εhigh from ϕlow,ϕhigh
- 4: for step = 1,...,N do
- 5: Sample a batch Db from D
- 6: Update the old policy model πθ

old ← πθ

- 7: Sample G outputs {oi}Gi=1 ∼ πθ

old

(·|q) for each question q ∈ Db

- 8: Filter questions where their G outputs are neither all correct nor all incorrect (Equation 5)
- 9: if filtered questions size < N then
- 10: continue
- 11: end if
- 12: For each oi of filtered questions, compute repetition penalty using f(oi), compute Aˆi,t for the t-th token of oi (Equation 6)
- 13: for iteration = 1,...,µ do
- 14: Update the policy model πθ by maximizing the CAMPO objective
- 15: end for
- 16: end for
- 17: end for Ensure: πθ

length saturates. The CAMPO training objective is formalized as:

JCAMPO(πθ) ≜ E(q,a)∼D,{o

i}Gi=1∼πθold(·|q)

|oi|

G

1

(5)

min ri,t(θ)Aˆi,t,clip(ri,t(θ),1 − ϕlow(s),1 + ϕhigh(s))Aˆi,t subject to 0 < |{oi |is_equivalent(a,oi)}| < G,

G i=1 |oi|

t=1

i=1

where

r(oi, a) − f(oi) − mean({r(oi, a) − f(oi)}Gi=1) std({r(oi, a) − f(oi)}Gi=1)

πθ(oi,t|q, oi,<t) πθold(oi,t|q, oi,<t)

, Aˆi,t ≜

ri,t(θ) ≜

, (6)

s denotes the training stage, ϕlow(s) and ϕhigh(s) represent the corresponding decoupled clipping distributions for different stages. f(oi) measures repetition in oi and outputs a score between 0 and 1, depending on how early the repetition occurs. Specifically, f(oi) returns the proportion of the whole token sequence that lies inside the detected repeating loop. The full algorithm can be found in Algorithm 1.

- 4.3.1 EFFICIENCY-AWARE PERSPECTIVE ON MULTI-STAGE TRAINING

One of the main bottlenecks in RL training lies in the high computational cost of long rollouts, which can significantly slow down training and reduce sampling efficiency. To address this, inspired by DeepScaleR (Luo et al., 2025b) and Skywork-OR1 (He et al., 2025a), we adopt a multistage RL training strategy designed to enhance training efficiency and model performance. In this framework, the maximum permitted response length is progressively increased across training stages, as illustrated in Algorithm 1. As rollouts exceed the permitted response will be assigned zero reward, it is necessary to increase the lengths to allow more complex reasoning during training.

This staged setup offers several benefits. First, the shorter response limit helps constrain the model’s output space, reducing rollout length and accelerating feedback cycles. Furthermore, responses that exceed the current maximum length are treated as failures, providing a clear training signal that encourages the model to produce more concise and refined outputs. This effect is demonstrated in Figure 5, which shows the mean token length of model outputs over 64 runs on AIME24 and AIME25 benchmarks. Compared to Skywork-OR1-32B-Preview, MiroMind-M1-RL-32B generates significantly shorter responses with minimal impact on accuracy, as shown in Table 5.

As training advances, progressively relaxing the length constraint becomes essential, especially for complex tasks that inherently demand longer reasoning trajectories. This gradual increase allows the model to scale its capacity for deeper reasoning while preserving the efficiency gains from earlier stages.

[Figure 8]

Figure 5: Average token length of model responses computed across all rollout attempts, evaluated over 64 independent runs on the corresponding test sets. This analysis includes both correct and incorrect answers, highlighting overall response efficiency regardless of correctness.

- 4.3.2 STABILIZING TRAINING WITH REPETITION PENALTY

The training process of RL can be unstable and prone to sudden collapse, particularly when the model starts generating a narrow set of low-diversity outputs.

To encourage the emergence of long reasoning patterns, the KL loss is often omitted; however, this can cause the trained model to drift significantly from the original policy model, potentially resulting in suboptimal behaviors. Introducing a repetition penalty helps mitigate this issue by promoting output diversity and discouraging repetitive patterns, thereby preventing the policy from collapsing into a narrow output space. As described in Section 4.3 and Algorithm 1, the repetition score ranges from 0 to 1, depending on how early the repetition occurs. Earlier repetitions incur heavier penalties. This design allows the penalty to dynamically adapt to the model’s repetition behavior. Figure 6 presents the training performance on AIME24 and AIME25 for CAMPO with the 7B model, compared to CAMPO without the repetition penalty. As shown, incorporating the repetition penalty leads to a more stable training process.

MiroMind-RL-7B-ablation

MiroMind-RL-7B-ablation-no-repetition

50

45

avg@8

|AIME24|
|---|

AIME25

40

35

40 60 80 100 120 140 160 180

Steps

- Figure 6: The training process is more stable with repetition penalty.

- 4.3.3 ACCURATE VERIFIER FOR EFFICIENT REASONING

The success of RL in training language models depends heavily on the accuracy of the reward signals, which act as the model’s very few forms of feedback. Unlike supervised learning, where models are guided by explicit ground-truth labels, RL relies on these signals to shape behavior indirectly. If the reward function mistakenly penalizes correct answers, often due to verifier errors, it can lead the model to develop inefficient reasoning patterns.

[Figure 9]

52.5

50.0

47.5

45.0

avg@8

MiroMind-RL-7B-ablation

|AIME24|
|---|

MiroMind-RL-7B-ablation-naive-verifier

42.5

AIME25

40.0

37.5

35.0

32.5

40 60 80 100 120 140 160 180

Steps

- Figure 7: Average token count of model responses conditioned on correct answers, computed over all rollouts and averaged across 64 runs on the respective test sets. Unlike Figure 5, this analysis considers only responses that received correct reward signals, highlighting that correct, rewarded outputs appear more efficient.

Figure 8: RL benefits from a more accurate verifier, highlighting the critical role of verifier quality in enabling robust training. High-quality verification ensures that reward signals align more closely with ground truth correctness, reducing noise and inconsistencies during optimization.

This problem becomes even more important in settings like GRPO, where the model’s internal state is estimated based on the outcomes of a batch of rollouts. Inaccurate reward signals, such as when correct answers are penalized due to verifier failure, can distort the model’s belief about which behaviors are desirable. As a result, it may revise valid solutions or introduce unnecessary reasoning steps that offer little value. Over time, this misalignment can cause the model to adopt overly complicated or even unstable reasoning behaviors (Zeng et al., 2025).

To reduce these issues, we significantly improved the math verifier 3. The original version (v0.7.0) struggled with various edge cases, including answers involving units, constants like π and degrees, percentages (%), and small differences in numerical precision. Our updated verifier introduces a cascade design with multiple verification stages that gradually increase in strictness. We also incorporated a range of human-curated fixes to make the verifier more robust and comprehensive. Together, these changes lead to more reliable feedback during training, helping the model learn to generate concise and logically sound answers more efficiently.

The impact of these improvements is illustrated in Figure 7, which shows the average token count of correct model responses across 64 runs on AIME24 and AIME25. Compared to Skywork-OR1-32B-Preview, MiroMind-M1-RL-32B consistently produces significantly shorter and more efficient reasoning traces for

3https://github.com/huggingface/Math-Verify

Model AIME24 AIME25 MATH500 DeepSeek-R1 79.8 70.0 – DeepSeek-R1-0528 91.4 87.5 – Qwen3-8B 76.0 67.3 – DeepSeek-R1-0528-Qwen3-8B 86.0 76.3 – MiMo-7B-RL 68.2 55.4 95.8 32B Models trained from Qwen2.5 series

DeepSeek-R1-Distill-Qwen-32B 70.8 52.1 95.8 Skywork-OR1-32B-Preview 77.1 68.2 97.5 MiroMind-M1-RL-32B 77.5 65.6 96.4

7B Models trained from Qwen2.5 series

DeepSeek-R1-Distill-Qwen-7B 55.5 39.2 – MiroMind-M1-SFT-7B 60.4 45.0 94.6 Light-R1-7B-DS 59.1 44.3 – Skywork-OR1-7B 72.2 54.6 – MiroMind-M1-RL-7B 73.4 57.8 96.7

Table 5: Comparison of our 7B and 32B model performance across benchmark datasets.

responses that received correct reward signals. As further shown in Figure 8, RL benefits from more accurate verification, underscoring the importance of verifier quality in shaping robust training.

- 4.4 MIROMIND-M1-RL-32B

In this section, we present MiroMind-M1-RL-32B, our direct attempt with CAMPO on SOTA reasoning models.

- 4.4.1 TRAINING DETAILS

To demonstrate the effectiveness of our RL settings with CAMPO on state-of-the-art models, we initialize training from DeepSeek-R1-Distill-Qwen-32B using the dataset described in Section 4.1. For multistage training, we begin with a maximum response length of 16,384 in Stage 1, then progressively increase it to 32,768 and 49,152 in Stages 2 and 3, respectively. All other training configurations remain consistent across stages. Specifically, we omit the KL loss to allow more flexibility in model learning. We use a constant learning rate of 1e-6 and a clipping ratio of 0.2. During rollouts, the temperature is set to 1.0 for more diverse sampling, and each data sample is rolled out 16 times. Both the training and mini-batch sizes are set to 32, ensuring strictly on-policy rollouts that better reflect the model’s current state. All experiments on DeepSeek-R1-Distill-Qwen-32B are conducted using 16*8 A100 GPUs. We follow the same evaluation setting as described in Section 3, except that the temperature is set to 1.0 to match the setting used during RL training.

- 4.4.2 EXPERIMENTAL RESULTS AND INSIGHTS

As shown in Table 5, MiroMind-M1-RL-32B demonstrates that CAMPO can significantly enhance performance on math benchmarks when finetuned from a strong initialization checkpoint (i.e., DeepSeek-R1-Distill-Qwen-32B). Specifically, it achieves notable improvements of 6.7% on

- AIME24 and 13.5% on AIME25, indicating the effectiveness of our RL framework.

| |+2.6<br><br>+7.5<br><br>+8.7<br><br>+0.2<br><br>MiroMind-M1-RL-32B<br><br>Skywork-OR1-32B-Preview| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

80

70

Accuracy(avg@16)

60

50

40

30

20

10

0

4k 8k 16k 32k

Max Response Length

(a) AIME24

| |+3.7<br><br>+4.8<br><br>+5.8<br><br>-3.4<br><br>MiroMind-M1-RL-32B<br><br>Skywork-OR1-32B-Preview| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

80

70

Accuracy(avg@16)

60

50

40

30

20

10

0

4k 8k 16k 32k

Max Response Length

(b) AIME25

Figure 9: Comparison of MiroMind-M1-RL-32B and Skywork-OR1-32B-Preview on AIME24 and

- AIME25 across different maximum response lengths.

Nevertheless, our 32B model still lags behind certain SOTA models that also employ RL for training. For instance, Skywork-OR1-32B-Preview surpasses our model by 2.6% on AIME25. We attribute this performance gap primarily to differences in the composition of training data. Skywork-OR1-32B-Preview benefits from a diverse mixture of math and code data, where the inclusion of code likely helps the model develop stronger symbolic reasoning capabilities. In contrast, our RL training relies solely on math-focused data, which may limit the model’s generalization on certain problem types.

On the other hand, CAMPO leads to more token-efficient reasoning behavior. As illustrated in Figure 9, MiroMind-M1-RL-32B consistently outperforms Skywork-OR1-32B-Preview across all shorter max response lengths, ranging from 4k to 16k. This suggests that our model is able to arrive at correct answers more succinctly. We attribute this token-efficiency to three key components in CAMPO: the repetition penalty, the cascade verifier, and the multi-stage training strategy. Together, these components reduce redundant reasoning and encourage the model towards generating token-efficient yet accurate responses, rather than unnecessarily prolonging outputs.

- 4.5 MIROMIND-M1-RL-7B

In this section, we present MiroMind-M1-RL-7B, our strongest model derived from the Qwen2.5 pretrained base series. This model represents a fully transparent and reproducible workflow for building reasoning-capable models through supervised fine-tuning (SFT) followed by reinforcement learning (RL). Starting from the Qwen2.5-7B-Math-Base checkpoint, the model is first fine-tuned via SFT (as introduced in Section 3) to reach MiroMind-M1-SFT-7B, then further optimized with RL to enhance its mathematical reasoning capabilities. As shown in Table 5, it achieves the best performance among all Qwen2.5-based models across multiple benchmarks. Further implementation details, comparative analysis, and insights from reinforcement learning are presented in the sections below to support reproducibility and highlight directions for future improvement.

- 4.5.1 TRAINING DETAILS

First, we use the same set of training data as the 32B model for RL training. All data points are accompanied by verifiable outputs. While it is commonly argued that the difficulty level of RL training data plays a crucial

role—particularly in early training stages where smoother reasoning progression is preferred. We find that it is sufficient for the dataset to span a range of difficulties for our algorithms. Notably, the 32B training data does not negatively impact the training of our 7B model. We attribute this to two key factors: 1. In our algorithm design, we explicitly exclude rollout samples that are either entirely correct or entirely incorrect, retaining only partially correct samples for training, as described in Algorithm 1; 2. During training, we ensure that each batch contains a sufficient number of valid samples (i.e., samples with partially correct rollouts) before performing gradient updates, maintaining a stable and efficient optimization process.

Second, for initial checkpoint, we use our post-trained reasoning model MiroMind-M1-SFT-7B as the initial checkout which is shown to be a better reasoning model compared to DeepSeek-R1-Distill-Qwen-7B and is fully reproducible with open-source data.

Third, similar to our 32B model variant, we adopt multiple training stages with varying maximum generation lengths. This strategy primarily aims to improve training efficiency, as the rollout phase dominates the computational cost in RL training, and generation length is a critical factor influencing that cost. Further details on the multi-stage setup and comparisons are provided in the following sections.

All experiments are conducted on a distributed setup consisting of 8 compute nodes, each equipped with 8 NVIDIA A800 GPUs, totaling 64 GPUs. The training pipeline is implemented using the VERL4 framework, which provides efficient infrastructure support for large-scale reinforcement learning workflows.

[Figure 10]

Figure 10: The model’s performance steadily improves throughout the training process.

- 4.5.2 EXPERIMENTAL RESULTS AND INSIGHTS

We adopt a two-stage training strategy. In the first stage, we cap the maximum rollout length at 16K tokens, discarding samples that fail to produce a complete answer within this limit. This constraint significantly reduces training cost, as the rollout phase dominates computational overhead and scales with sequence length. After 1200 steps, we update the maximum rollout length per sample to 32K tokens and continue training with the same model checkpoint and optimizers. We summarize notable patterns observed during training and analyze their impact on model performance and optimization dynamics.

Model Performance Over Training Steps. In Figure 10, we show the performance trajectory of our MiroMind-M1-RL-7B model during training. The RL process yields over a 15% accuracy improvement

4https://github.com/volcengine/verl

on both AIME24 and AIME25. As a result, our model achieves state-of-the-art (SOTA) math capabilities among all models derived from the Qwen2.5 series. While some recently introduced models outperform ours, they are built upon newer base models. Our techniques can be similarly applied to those newer models if needed. Additionally, our model outperforms MiMo-7B-RL (Xiaomi LLM-Core Team, 2025), which is trained from scratch using 500k SFT samples, a training size comparable to ours. A newer variant, MiMo-7B-RL-0530, was recently released with 6 million newly curated SFT samples, making direct comparison less appropriate due to the substantially larger training set.

[Figure 11]

Figure 11: Response length trend during two-stage training. Under the 16K generation cap, the model generates relatively shorter responses. Once the limit is increased to 32K, the model begins producing significantly longer outputs.

Response Length. We examine the average generation length across all training samples throughout the training process. As shown in Figure 11, the response length becomes compressed during the initial phase with a 16K token generation cap, stabilizing between 8K and 9K tokens. After transitioning to the 32K generation limit, the average maximum length increases significantly, with responses extending beyond 13K tokens. We hypothesize that the initial constraint pushes the model to operate near its capacity within a limited budget, helping establish a strong foundation for reasoning. This, in turn, may lead to more stable and effective optimization in the longer sequence regime.

Comparison with Single-Stage Training. As a parallel study to our two-stage pipeline, we experiment using a single-stage training process with the 7B model by setting the maximum context length to 32K. All hyperparameters and settings were kept consistent with those used in MiroMind-M1-RL-7B. The full training performance trajectory is shown in Figure 12. While the single-stage approach can achieve competitive performance compared to the two-stage method, the latter offers two key advantages: (1) faster training: since starting with long sequences imposes greater memory and computational overhead, resulting in slower training speeds; and (2) potentially better performance when the output token budget is limited, as same patterns found in our MiroMind-M1-RL-32B models, which also experienced multi-stage training, presented in Figure 9.

Discussion on Evluation Stability. There are growing concerns about the stability of evaluation results, particularly for challenging benchmarks like AIME24 and AIME25, which contain only 30 questions. A difference of just 1–2 correct answers can cause performance fluctuations exceeding 5%, posing significant challenges for consistent benchmarking. A common mitigation strategy is to run the evaluation 64 times and report the average accuracy. However, as shown in Figure 13, the two-sided standard deviation can still

[Figure 12]

- Figure 12: Performance trend of the model trained using a single-stage 32K max context length schema.

[Figure 13]

- Figure 13: Evaluation stability assessment: 64 repeated evaluations of MiroMind-M1-RL-7B model performance on AIME24.

exceed 8%. While we currently lack a better solution, increasing the number of evaluation runs can yield more robust results, but at the cost of a significantly more time-consuming benchmarking process.

Efficiency in RL Training. Another important aspect to consider is the efficiency of reinforcement learning (RL) training. We found that the primary bottleneck lies not in updating model parameters, but in the rollout phase, where the model generates responses to compute rewards. In our current framework, rollouts are executed in a synchronized manner across batches. As a result, the presence of even a few samples with extremely long generations can significantly delay the entire batch, leading to prolonged GPU idle time and reduced overall training efficiency. This inefficiency becomes more pronounced when encountering long-tail generation issues during inference, where certain inputs result in abnormally long outputs. Such cases dominate the runtime and disrupt the parallelism benefits typically expected in batch processing. Similar challenges have been reported in prior work Ji et al. (2025a), where researchers proposed various strategies

such as detached rollout and streaming load balancing architecture. Our two-stage training strategy partially alleviates this issue by bounding the generation length in the early stage, thus containing rollout variability to some extent. However, addressing this bottleneck more comprehensively will require further research and system-level optimizations in future work.

- 5 CONCLUSIONS

In this paper, we open-source a comprehensive system for training reasoning language models, including codes, datasets, and checkpoints. For the SFT phase, we curate a large scale of high-quality data, which results in significantly stronger performance on math reasoning benchmarks compared to DeepSeek’s 7B distillation models. In addition, we propose CAMPO, a context-aware multi-stage policy optimization framework, to further improve the performance of the reinforcement learning stage. We show that our reasoning models of 7B can achieve better performance than Skywork’s counterpart with fewer tokens, indicating greater efficiency. We hope that our efforts can facilitate future research on reasoning language models.

REFERENCES

Alon Albalak, Duy Phung, Nathan Lile, Rafael Rafailov, Kanishk Gandhi, Louis Castricato, Anikait Singh, Chase Blagden, Violet Xiang, Dakota Mahan, and Nick Haber. Big-math: A large-scale, high-quality math dataset for reinforcement learning in language models. arXiv preprint arXiv:2502.17387, 2025.

Anthropic. The claude 3 model family: Opus, sonnet, haiku, 2024.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Proceedings of NeurIPS, 2020.

Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, Xu Sun, Lei Li, and Zhifang Sui. A survey on in-context learning. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of EMNLP, 2024.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. Xiachong Feng, Longxu Dou, and Lingpeng Kong. Reasoning does not necessarily improve role-playing

ability. arXiv preprint arXiv:2502.16940, 2025.

Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, Ashima Suvarna, Benjamin Feuer, Liangyu Chen, Zaid Khan, Eric Frankel, Sachin Grover, Caroline Choi, Niklas Muennighoff, Shiye Su, Wanjia Zhao, John Yang, Shreyas Pimpalgaonkar, Kartik Sharma, Charlie Cheng-Jie Ji, Yichuan Deng, Sarah Pratt, Vivek Ramanujan, Jon Saad-Falcon, Jeffrey Li, Achal Dave, Alon Albalak, Kushal Arora, Blake Wulfe, Chinmay Hegde, Greg Durrett, Sewoong Oh, Mohit Bansal, Saadia Gabriel, Aditya Grover, Kai-Wei Chang, Vaishaal Shankar, Aaron Gokaslan, Mike A. Merrill, Tatsunori Hashimoto, Yejin Choi, Jenia Jitsev, Reinhard Heckel, Maheswaran Sathiamoorthy, Alexandros G. Dimakis, and Ludwig Schmidt. Openthoughts: Data recipes for reasoning models. arXiv preprint arXiv:2506.04178, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Bo An, Yang Liu, and Yahui Zhou. Skywork open reaonser series. Notion Blog, 2025a.

Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. Deepmath103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. arXiv preprint arXiv:2504.11456, 2025b.

Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, and Heung-Yeung Shum Xiangyu Zhang. Openreasoner-zero: An open source approach to scaling reinforcement learning on the base model. GitHub, 2025.

Yunjie Ji, Xiaoyu Tian, Sitong Zhao, Haotian Wang, Shuaiting Chen, Yiping Peng, Han Zhao, and Xiangang Li. Am-thinking-v1: Advancing the frontier of reasoning at 32b scale. arXiv preprint arXiv:2505.08311,

- 2025a.

Yunjie Ji, Xiaoyu Tian, Sitong Zhao, Haotian Wang, Shuaiting Chen, Yiping Peng, Han Zhao, and Xiangang Li. Am-thinking-v1: Advancing the frontier of reasoning at 32b scale. arXiv preprint arXiv:2505.08311,

- 2025b.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Koray Kavukcuoglu. Gemini 2.5: Our most intelligent ai model, 2025.

Kimi-Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, and Zonghan Yang. Kimi k1.5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Hynek Kydlicek. Math-verify: Math verification library, 2024. Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul,

Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath, 2024.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

Jiawei Liu and Lingming Zhang. Code-r1: Reproducing r1 for code with reliable rewards, 2025. Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin.

Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

Michael Luo, Sijun Tan, Roy Huang, Xiaoxiang Shi, Rachel Xin, Colin Cai, Ameen Patel, Alpay Ariyak, Qingyang Wu, Ce Zhang, Li Erran Li, and Ion Stoica Raluca Ada Popa. Deepcoder: A fully open-source 14b coder at o3-mini level. Notion Blog, 2025a.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. Notion Blog, 2025b.

Mathematical Association of America. Aime problems and solutions. AoPS Wiki, Art of Problem Solving, n.d.

Justus Mattern, Sami Jaghouar, Manveer Basra, Jannik Straube, Matthew Di Ferrante, Felix Gabriel, Jack Min Ong, Vincent Weisser, and Johannes Hagemann. Synthetic-1: Two million collaboratively generated reasoning traces from deepseek-r1, 2025.

Mistral-AI, :, Abhinav Rastogi, Albert Q. Jiang, Andy Lo, Gabrielle Berrada, Guillaume Lample, Jason Rute, Joep Barmentlo, Karmesh Yadav, Kartik Khandelwal, Khyathi Raghavi Chandu, Léonard Blier, Lucile Saulnier, Matthieu Dinot, Maxime Darrin, Neha Gupta, Roman Soletskyi, Sagar Vaze, Teven Le Scao, Yihan Wang, Adam Yang, Alexander H. Liu, Alexandre Sablayrolles, Amélie Héliou, Amélie Martin, Andy Ehrenberg, Anmol Agarwal, Antoine Roux, Arthur Darcet, Arthur Mensch, Baptiste Bout, Baptiste Rozière, Baudouin De Monicault, Chris Bamford, Christian Wallenwein, Christophe Renaudin, Clémence Lanfranchi, Darius Dabert, Devon Mizelle, Diego de las Casas, Elliot Chane-Sane, Emilien Fugier, Emma Bou Hanna, Gauthier Delerce, Gauthier Guinet, Georgii Novikov, Guillaume Martin, Himanshu Jaju, Jan Ludziejewski, Jean-Hadrien Chabran, Jean-Malo Delignon, Joachim Studnia, Jonas Amar, Josselin Somerville Roberts, Julien Denize, Karan Saxena, Kush Jain, Lingxiao Zhao, Louis Martin, Luyu Gao, Lélio Renard Lavaud, Marie Pellat, Mathilde Guillaumin, Mathis Felardos, Maximilian Augustin, Mickaël Seznec, Nikhil Raghuraman, Olivier Duchenne, Patricia Wang, Patrick von Platen, Patryk Saffer, Paul Jacob, Paul Wambergue, Paula Kurylowicz, Pavankumar Reddy Muddireddy, Philomène Chagniot, Pierre Stock, Pravesh Agrawal, Romain Sauvestre, Rémi Delacourt, Sanchit Gandhi, Sandeep Subramanian, Shashwat Dalal, Siddharth Gandhi, Soham Ghosh, Srijan Mishra, Sumukh Aithal, Szymon Antoniak, Thibault Schueller, Thibaut Lavril, Thomas Robert, Thomas Wang, Timothée Lacroix, Valeriia Nemychnikova, Victor Paltz, Virgile Richard, Wen-Ding Li, William Marshall, Xuanyu Zhang, and Yunhao Tang. Magistral. arXiv preprint arXiv:2506.10910, 2025.

OpenAI. Gpt-4o system card, 2024. OpenAI. Introducing openai o3 and o4-mini, 2025.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke E. Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Francis Christiano, Jan Leike, and Ryan J. Lowe. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155, 2022.

J. Schulman. Approximating kl divergence, 2020. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimiza-

tion algorithms. arXiv preprint arXiv:1707.06347, 2017.

ByteDance Seed. Seed-thinking-v1.5: Advancing superb reasoning models with reinforcement learning, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.

OpenThoughts Team. Open Thoughts, 2025a. Qwen Team. Qwen3: Think deeper, act faster, 2025b. Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, 2025c. Xiaoyu Tian, Sitong Zhao, Haotian Wang, Shuaiting Chen, Yiping Peng, Yunjie Ji, Han Zhao, and Xiangang

Li. Deepdistill: Enhancing llm reasoning capabilities via large-scale difficulty-graded data training. arXiv preprint arXiv:2504.17565, 2025.

Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, and Subbarao Kambhampati. On the planning abilities of large language models - a critical investigation. In Proceedings of NeurIPS, 2023.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Proceedings of NeurIPS, 2017.

Hui Wei, Zihao Zhang, Shenghua He, Tian Xia, Shijia Pan, and Fei Liu. Plangenllms: A modern survey of llm planning capabilities. arXiv preprint arXiv:2502.11221, 2025.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In Proceedings of ICLR, 2022a.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Proceedings of NeurIPS, 2022b.

Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, and William Fedus. Measuring short-form factuality in large language models. arXiv preprint arXiv:2411.04368, 2024a.

Jerry Wei, Chengrun Yang, Xinying Song, Yifeng Lu, Nathan Zixia Hu, Jie Huang, Dustin Tran, Daiyi Peng, Ruibo Liu, Da Huang, Cosmo Du, and Quoc V Le. Long-form factuality in large language models. In Proceedings of NeurIPS, 2024b.

Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang, Xiaowei Lv, Haosheng Zou, Yongchao Deng, Shousheng Jia, and Xiangzheng Zhang. Light-r1: Curriculum sft, dpo and rl for long cot from scratch and beyond. arXiv preprint arXiv:2503.10460, 2025a.

Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang, Xiaowei Lv, et al. Light-r1: Curriculum sft, dpo and rl for long cot from scratch and beyond. arXiv preprint

- arXiv:2503.10460, 2025b.

Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang, Xiaowei Lv, et al. Light-r1: Curriculum sft, dpo and rl for long cot from scratch and beyond. arXiv preprint

- arXiv:2503.10460, 2025c.

Xiaomi LLM-Core Team. Mimo: Unlocking the reasoning potential of language model – from pretraining to posttraining. arXiv preprint arXiv:2505.07608, 2025.

Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo. Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2502.14768, 2025.

Fengli Xu, Qianyue Hao, Zefang Zong, Jingwei Wang, Yunke Zhang, Jingyi Wang, Xiaochong Lan, Jiahui Gong, Tianjian Ouyang, Fanjin Meng, Chenyang Shao, Yuwei Yan, Qinglong Yang, Yiwen Song, Sijian Ren, Xinyuan Hu, Yu Li, Jie Feng, Chen Gao, and Yong Li. Towards large reasoning models: A survey of reinforced reasoning with large language models. arXiv preprint arXiv:2501.09686, 2025a.

Yixuan Even Xu, Yash Savani, Fei Fang, and Zico Kolter. Not all rollouts are useful: Down-sampling rollouts in llm reinforcement learning. arXiv preprint arXiv:2504.13818, 2025b.

Zhangchen Xu, Yang Liu, Yueqin Yin, Mingyuan Zhou, and Radha Poovendran. Kodcode: A diverse, challenging, and verifiable synthetic dataset for coding. arXiv preprint arXiv:2503.02951, 2025c.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In Proceedings of ICLR, 2023.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Yufeng Yuan, Yu Yue, Ruofei Zhu, Tiantian Fan, and Lin Yan. What’s behind ppo’s collapse in long-cot? value optimization holds the secret. arXiv preprint arXiv:2503.01491, 2025.

Albert S Yue, Lovish Madaan, Ted Moskovitz, DJ Strouse, and Aaditya K Singh. Harp: A challenging human-annotated math reasoning benchmark. arXiv preprint arXiv:2412.08819, 2024.

Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, Xiangpeng Wei, Gaohong Liu, Juncai Liu, Lingjun Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Ru Zhang, Xin Liu, Mingxuan Wang, Yonghui Wu, and Lin Yan. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025.

Chong Zhang, Yue Deng, Xiang Lin, Bin Wang, Dianwen Ng, Hai Ye, Xingxuan Li, Yao Xiao, Zhanfeng Mo, Qi Zhang, and Lidong Bing. 100 days after deepseek-r1: A survey on replication studies and more directions for reasoning language models. arXiv preprint arXiv:2505.00551, 2025.

Han Zhao, Haotian Wang, Yiping Peng, Sitong Zhao, Xiaoyu Tian, Shuaiting Chen, Yunjie Ji, and Xiangang Li. 1.4 million open-source distilled reasoning dataset to empower large language model training. arXiv preprint arXiv:2503.19633, 2025.

### Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of ACL, 2024.

