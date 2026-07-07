[Figure 1]

## CHIMERA: Compact Synthetic Data for Generalizable LLM Reasoning

Xinyu Zhu1 Yihao Feng2 Yanchao Sun2 Xianzhi Du2 Pingzhi Li3 Olli Saarikivi2 Yun Zhu2 Yu Meng1

# arXiv:2603.00889v1[cs.CL]1Mar2026

### Abstract

Large Language Models (LLMs) have recently exhibited remarkable reasoning capabilities, largely enabled by supervised fine-tuning (SFT)- and reinforcement learning (RL)-based post-training on high-quality reasoning data. However, reproducing and extending these capabilities in open and scalable settings is hindered by three fundamental data-centric challenges: (1) the cold-start problem, arising from the lack of seed datasets with detailed, long Chain-of-Thought (CoT) trajectories needed to initialize reasoning policies; (2) limited domain coverage, as most existing open-source reasoning datasets are concentrated in mathematics, with limited coverage of broader scientific disciplines; and (3) the annotation bottleneck, where the difficulty of frontier-level reasoning tasks makes reliable human annotation prohibitively expensive or infeasible. To address these challenges, we introduce CHIMERA, a compact synthetic reasoning dataset comprising 9K samples designed to support generalizable reasoning across domains. CHIMERA is constructed with three key properties: (1) it provides rich, long CoT reasoning trajectories synthesized by state-of-the-art reasoning models; (2) it has broad and structured coverage, spanning 8 major scientific disciplines and over 1K fine-grained topics organized via a model-generated hierarchical taxonomy; and (3) it employs a fully automated, scalable evaluation pipeline that uses strong reasoning models to cross-validate both problem validity and answer correctness, removing the reliance on human annotations. We use CHIMERA

1Department of Computer Science, University of Virginia, Charlottesville, VA, USA 2Apple, Cupertino, CA, USA 3Department of Computer Science, University of North Carolina at Chapel Hill, Chapel Hill, NC, USA. Correspondence to: Xinyu Zhu <xinyuzhu@virginia.edu>, Yun Zhu <gabrielzhuyun@gmail.com>, Yu Meng <yumeng5@virginia.edu>.

Preprint.

to post-train a 4B Qwen3 model using a combination of SFT and RL. Despite the dataset’s modest size, the resulting model achieves strong performance on a suite of challenging reasoning benchmarks, including GPQA-Diamond, AIME 24/25/26, HMMT 25 and Humanity’s Last Exam, approaching or matching the reasoning performance of substantially larger models such as DeepSeek-R1 and Qwen3-235B.1

### 1. Introduction

Large language models (LLMs) have recently demonstrated substantial advances in complex, multi-step reasoning across mathematics, science, and general problemsolving tasks (Guo et al., 2025; Jaech et al., 2024; Lambert et al., 2024). These capabilities are largely enabled by post-training procedures, most notably reinforcement learning (Shao et al., 2024; Chen et al., 2025a; Team et al., 2025), that encourage LLMs to generate explicit intermediate reasoning steps, often in the form of long CoT trajectories. Models trained under such regimes exhibit improved planning, abstraction, and self-correction behaviors (Gandhi et al., 2025; Zeng et al., 2025; Muennighoff et al., 2025). Despite these successes, replicating and extending frontier-level reasoning capabilities in open and resource-constrained settings remains challenging. In particular, progress is increasingly limited not by training techniques, but by the availability and quality of reasoning data. As reasoning-oriented post-training becomes more central to LLM development, the construction of scalable, highquality reasoning datasets has emerged as a key bottleneck.

Notably, three fundamental data-centric challenges hinder the development of open general-purpose reasoning models: (1) Cold-start data scarcity. Effective reasoning-oriented post-training typically requires an initial corpus of examples containing detailed, long CoT trajectories to bootstrap policy learning (Wang et al., 2025; Yeo et al., 2025). However, existing datasets often provide either only ground-

1Dataset is available at https://huggingface.co/ datasets/TianHongZXY/CHIMERA.

###### Stage 1: Subject Expansion Stage 2: Problem Generation Stage 3: Solution Synthesis

[Figure 2]

[Figure 3]

Problem 1

Specific Topic / ∈ ($

Seed Subjects (e.g., Math, Physics) ! = {$1,…,$"}

Drafting Self-Contained Reasoning Problems

Generating Detailed Reasoning Chains

[Figure 4]

[Figure 5]

"

ℛ

###### Expansion & Deduplication

ℰ

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

1,,2, 1-,2- 1.,2.

[Figure 15]

[Figure 16]

[Figure 17]

Long-Form Trajectory 3

[Figure 18]

[Figure 19]

###### Verifying Validity & Correctness

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

#

[Figure 25]

Correctness Labeling

[Figure 26]

[Figure 27]

[Figure 28]

%

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

(# = set ℰ($) Comprehensive Topic Hierarchies

4%&'()*+ = {($,/,1,2,3,5)}

Validated (1,2) Pairs

- Figure 1. Data synthesis pipeline overview: stage 1 expands a small set of seed subjects into thousands of fine-grained topics; stage 2 creates well-defined problems with concise, verifiable answers based on these topics; stage 3 generates detailed reasoning trajectories and labels their correctness.

truth answers (Yu et al., 2025) or brief explanations (Cobbe et al., 2021; Hendrycks et al., 2021b; LI et al., 2024; Yu et al., 2024), which are insufficient for initializing models that must learn to perform long-horizon complex reasoning. This cold-start problem is particularly critical for smaller or mid-sized models, which have more reliance on the quality and structure of supervision. (2) Limited domain coverage. Most publicly available reasoning datasets focus narrowly on mathematics and coding tasks (Yu et al., 2024; He et al., 2025; Yu et al., 2025; Hugging Face, 2025; Guha et al., 2025). While these domains are valuable, they represent only a small fraction of the reasoning demands encountered in real-world problem solving. As a result, models trained on these datasets often struggle to generalize their reasoning strategies to other scientific disciplines or interdisciplinary problems. (3) The annotation bottleneck. As reasoning benchmarks approach or exceed human expert difficulty, reliable manual annotation becomes increasingly impractical (Phan et al., 2025; Wang et al., 2023). Producing correct answers and especially high-quality CoT explanations for frontier-level questions often requires deep domain expertise and substantial time investment. This makes large-scale human annotation costly, slow, and in many cases unreliable.

to domains where human annotation is infeasible. However, naive synthetic data generation risks narrow coverage, error accumulation, and uncontrolled quality, limiting its effectiveness for post-training (Yu et al., 2023; Chen et al., 2024a). In this work, we explore whether a compact yet carefully constructed synthetic dataset can meaningfully support reasoning-oriented post-training. Rather than maximizing dataset size, we focus on three design principles: (1) long intermediate reasoning steps, (2) broad and structured domain coverage, and (3) scalable quality control without human supervision.

Guided by these principles, we introduce CHIMERA, a synthetic reasoning dataset consisting of 9K high-quality samples. Each sample includes a long CoT reasoning trajectory generated by state-of-the-art reasoning models, providing rich supervision for multi-step reasoning behaviors. For broad domain coverage, CHIMERA spans 8 major scientific disciplines and over 1K fine-grained topics. These topics are derived from a structured, model-generated hierarchical taxonomy, enabling systematic coverage of both core concepts and specialized subfields. To ensure quality and scalability, CHIMERA employs a fully automated evaluation protocol. Instead of relying on human annotators, we use strong reasoning models to cross-validate problem validity and answer correctness and filter low-quality or inconsistent samples.

Recent advances in synthetic data generation suggest a promising direction (Lee et al., 2024; Cui et al., 2024; Xu

- et al., 2024b): leveraging strong models themselves to synthesize high-quality training data. When carefully designed, model-generated CoT trajectories can exhibit rich intermediate structure, cover diverse reasoning patterns, and scale

We evaluate the effectiveness of CHIMERA by post-training a Qwen3-4B model (Yang et al., 2025) using a combination

of SFT and RL. Despite the compact size of CHIMERA, the resulting model demonstrates strong reasoning performance across a diverse set of challenging benchmarks, including GPQA-Diamond (Rein et al., 2024), AIME24, AIME25 (AIME, 2025), AIME26, HMMT25 (Balunovi´c

- et al., 2025), and Humanity’s Last Exam (HLE) (Phan et al., 2025). Notably, the fine-tuned 4B model achieves performance competitive with substantially larger models (e.g., DeepSeek-R1 and Qwen3-235B). In summary, our contributions are as follows:

- 1. We identify and formalize key data-centric challenges that limit scalable reasoning post-training for LLMs.
- 2. We introduce CHIMERA, a compact, fully synthetic dataset featuring long CoT trajectories, broad subject coverage, and automated quality control.
- 3. We show that post-training on CHIMERA enables a 4Bparameter model to match or approach the performance of substantially larger models (e.g., DeepSeek-R1 and Qwen3-235B) across a range of challenging reasoning benchmarks.

### 2. Method

#### 2.1. Overview

Our goal is to automatically construct a reasoning dataset that covers as broad topics as possible without human annotation. To this end, we propose a modular, LLM-driven data synthesis pipeline that consists of three decoupled stages, illustrated in Figure 1. The complete procedure is formalized in Algorithm 1. Specifically, the pipeline comprises: (1) subject expansion, given a small set of high-level subjects (e.g., math, physics, etc.), we leverage gpt-5-2025-08-07 (hereafter referred to as gpt-5) to list as many topics under the subject as possible, which leads to a comprehensive topic hierarchy for each subject; (2) problem generation, given a specific topic, we further use gpt-5 to propose a clear, self-contained and easy-toverify problem along with the corresponding answer, this process can be repeated for several times to synthesize multiple examples for one topic; (3) solution synthesis, for each problem, we generate a detailed reasoning trajectory with Qwen3-235B-A22B-Thinking-2507, a state-of-theart open reasoning language model.2

All the stages are separate and intermediate artifacts are saved, enabling subsequent dataset filtering and curation. The pipeline is designed to be simple, scalable, and extensible, allowing adding new subjects or adjusting topic

2Many proprietary LLMs such as gpt-5 do not provide full access to its intermediate thinking trajectory and thus cannot be used to synthesize the detailed solution.

distributions with little additional effort. All the prompts (e.g., subject expansion, problem generation, solution synthesis, problem validator and correctness verifier) can be found in Appendix A.

Algorithm 1 CHIMERA Data Synthesis Pipeline

Require: Seed subjects S = {s1,...,sm}; number of samples per topic n; topic expander E; problem/answer generator G; problem validator V; correctness verifier C; reasoning trajectory generator R

Ensure: CHIMERA dataset DCHIMERA

- 1: Initialize DCHIMERA ← ∅
- 2: Stage 1: Subject expansion §2.2
- 3: for each subject s ∈ S do
- 4: Ts ← set(E(s)) // expand subject
- 5: end for
- 6: Stage 2: Problem generation §2.3
- 7: for each subject s ∈ S do
- 8: for each topic t ∈ Ts do
- 9: for j = 1 to n do
- 10: (q,a) ← G(t) // draft problem and answer
- 11: if not V(q,a) then
- 12: continue // discard ill-posed problems
- 13: end if
- 14: Stage 3: Solution synthesis §2.4
- 15: r ← R(q) // reasoning trajectory
- 16: y ← C(q,a,r) // y=1 if correct else 0
- 17: DCHIMERA ← DCHIMERA ∪ {(s,t,q,a,r,y)}
- 18: end for
- 19: end for
- 20: end for
- 21: return DCHIMERA

#### 2.2. Subject Expansion

We first collect a small set of high-level subjects S that are broad and abstract, such as mathematics, physics, and computer science. These subjects are intentionally coarsegrained to minimize human design choices and to encourage wide domain coverage. For each subject s, we prompt gpt-5 to generate a comprehensive list of fine-grained topics Ts that span the conceptual space of the subject. To cover the most foundational concepts of mathematics, which has too many subfields, we sample multiple times for it. After the expansion, we conduct deduplication to ensure each topic is unique. The resulting topic lists form a hierarchical taxonomy that serves as the backbone of the dataset. By decoupling topic expansion from later stages, the pipeline allows new subjects to be added or existing topic distributions to be modified easily. All the subjects and topics can be found in Appendix B.

- Table 1. Statistics of commonly used reasoning datasets. CHIMERA features substantially longer problem statements and more detailed solutions than existing datasets (lengths are measured in words), enabling complex and long-horizon reasoning for training modern LLMs.

###### Dataset # Problems # Subjects # Topics Prompt Solution Answer Solution

Length Length Format Annotator Human-Curated Reasoning Datasets

GSM8K 7,473 1 – 45.1 51.7 Numeric Human MATH 7,500 1 – 33.0 89.5 Free-form Human NuminaMath-CoT 859,494 1 – 44.0 205.3 Free-form Human

Synthetic Reasoning Datasets

MetaMathQA 395,000 1 – 40.5 101.3 Free-form AI DAPO-Math-17K 17,398 1 – 42.5 1 Numeric AI OpenR1-Math-220K 225,129 1 – 43.6 2,624.6 Free-form AI TULU3-SFT 939,343 – – 148.6 227.6 Free-form AI DeepMath-103K 103,022 1 – 33.7 2,959.2 Free-form AI OpenScience3 315,579 – – 76.1 1,296.8 Multiple-choice AI

Our Dataset

###### CHIMERA 9,225 8 1,179 211.1 11,121.4 Free-form AI

Distribution of Problem Subjects

- Figure 2. Distribution of problem subjects in CHIMERA. The left panel illustrates broad disciplinary coverage, with mathematics accounting for 48.3% of the dataset, followed by computer science, chemistry and physics. The right panel decomposes the mathematics subset into fine-grained subfields. This distribution reflects the CHIMERA’s emphasis on disciplinary breadth and topic diversity.

#### 2.3. Problem Generation

Given the expanded set of topics, we generate reasoning problems by prompting gpt-5 to produce one problem and the corresponding answer (q,a) for each topic t. Each problem is required to satisfy the following criteria:

- • Solvability and difficulty: The problem must be solvable by an expert at a PhD level and is not an open research problem.
- • Self-contained: All necessary information must be included in the problem statement.

3OpenScience has multiple subsets; here it refers to the OSQwen3-235B-4 subset.

• Unambiguous and verifiable answer: The problem should admit a clear and unique answer. The correctness of the answer should be easy to verify.

These constraints are enforced through careful prompt design (details in Appendix A) and subsequent LLM-based filtering, enabling fully automated and scalable data generation.

Cross-model verification of problem validity. To ensure the reliability of synthesized data, we perform cross-model verification after problem generation. Specifically, we employ two independent LLMs, gpt-5 and o4-mini, as verifiers V to assess both problem validity and answer correctness. Each verifier checks whether (i) the problem is

well-posed and unambiguous, and (ii) the provided answer correctly solves the problem. A problem is retained only if it passes verification by both models. This dual-verifier design reduces the risk of systematic model bias or hallucinated solutions from a single model. By requiring agreement across independent models, we obtain a higher-confidence subset of synthesized problems with verifiable correctness. The accepted problems are stored together with their associated subjects and topics.

#### 2.4. Solution Synthesis

Although frontier proprietary LLMs like ChatGPT (OpenAI, 2023), Gemini (Team, 2025) and Claude (Anthropic, 2025) are good at generating high-quality responses, their detailed thinking processes are often inaccessible to the users. Therefore, their generated solutions are only partially shown to the user, which are concise and brief, making it challenging to utilize them for training advanced reasoning models.

To address this issue, we apply a strong open reasoningintensive model R and regenerate a detailed reasoning trajectory r for each problem q created in the previous stage. In our implementation, we employ Qwen3-235B-A22B-Thinking-2507 to produce the step-by-step reasoning trajectories. Then we compare each trajectory r with the original answer a to the problem q and label its correctness y ∈ {0,1}. Reasoning trajectories that lead to correct final answers can be used for supervised finetuning, while the other problems without correct reasoning trajectories are kept as problem-answer-only instances and can be used for reinforcement learning, where only the final answer is required for training.

#### 2.5. Dataset Statistics

- Table 1 compares CHIMERA with representative humancurated and synthetic reasoning datasets along scale, subject coverage, and problem characteristics. Human-annotated datasets such as GSM8K and MATH are high-quality but limited to a single subject domain, with relatively short prompts and solutions. Recent synthetic datasets greatly increase scale, yet most remain focused on a single domain or lack explicit subject and topic organization. Our dataset prioritizes structured diversity over sheer scale, with detailed distribution statistics shown in Figure 2. In total, it contains 9,225 problems with detailed reasoning trajectories. While smaller in total number of problems, it explicitly covers 8 subjects and 1,179 topics, enabling broad and systematic coverage across disciplines. The problems are more complex, resulting in substantially longer prompts. Moreover, the solutions are significantly more detailed than those in prior datasets, reaching 11K words and encouraging rigorous reasoning behavior of modern thinking LLMs. Overall, our dataset complements existing reasoning resources by

emphasizing explicit subject structure and long-form reasoning across diverse subjects.

### 3. Experiments

#### 3.1. Experimental Setup

Training Setting. All experiments use Qwen3-4B-Thinking-2507 as the base model. Unless otherwise stated, all fine-tuned models are initialized from the same checkpoint to ensure fair comparison. We compare models trained on our synthesized dataset and public synthetic baselines.

We first perform supervised fine-tuning on problems whose reasoning trajectories are verified as correct during solution synthesis, using a batch size of 256 and a learning rate of 1e−5. Starting from the SFT model, we further apply reinforcement learning with CISPO (Chen et al., 2025a) for one epoch, using the same batch size, a learning rate of 1e−6, and 8 rollouts per prompt. RL is conducted on a mixture of (i) the SFT training set and (ii) a curated subset of synthesized problems that were unsolved during solution synthesis but can be solved by the SFT model within 8 trials. Since our dataset contains free-form answers rather than multiplechoice outputs, we rely on LLM-based reward evaluation to provide reliable reward signals. We use o4-mini as the reward model to score generated rollouts during RL.

Benchmarks. We evaluate models on a diverse set of challenging reasoning benchmarks spanning scientific reasoning, mathematical problem solving, and knowledgeintensive tasks: GPQA-Diamond (GPQA-D) (Rein et al.,

- 2024), AIME24, AIME25 (AIME, 2025), AIME26, HMMT25 (Balunovi´c et al., 2025) and HLE (Phan et al.,
- 2025), for HLE we only consider text-only problems as the models are not multi-modal.

Evaluation setting. For all evaluations, we use the official suggested decoding configuration: temperature = 0.6, top-p = 0.95, top-k = 20, maximum token number = 102,400. To reduce variance and fairly evaluate reasoning performance, we sample 32 solutions per problem for AIME and HMMT, 10 for GPQA-Diamond, and 8 for Humanity’s Last Exam. For each problem, we report the unbiased pass@1, following common practice in prior works (Zhu et al., 2025).

Baselines. We compare three settings: (1) the base model Qwen3-4B-Thinking-2507 without additional finetuning, (2) the base model fine-tuned on OpenScience, and (3) the base model fine-tuned on CHIMERA.

- Table 2. Main results on reasoning benchmarks. Models are categorized into Standard Scale (≤ 70B) and Large Scale (> 70B). Notably, fine-tuning the Qwen3-4B base model on CHIMERA yields performance competitive with substantially larger models (e.g., DeepSeek-R1, Qwen3-235B-A22B), highlighting the strong data efficiency of our dataset.

Model # Params GPQA-D AIME24 AIME25 AIME26 HMMT Feb 25 HMMT Nov 25 HLE Large Scale (> 70B)

DeepSeek-R1 671B 71.5 79.8 70.0 – 41.7 – 8.5 DeepSeek-R1-0528 671B 81.0 91.4 87.5 – 79.4 – 17.7 Qwen3-235B-A22B 235B 71.1 85.7 81.5 – 62.5 – 11.8 Qwen3-235B-A22B-Thinking-2507 235B 81.1 – 92.3 – 83.9 – 18.2

- o3-mini (medium) – 76.8 79.6 74.8 – – – 10.3
- o4-mini (high) – 81.4 93.4 92.7 – 66.7 – 18.1 gemini-2.5-pro – 86.4 – 88.0 – 82.5 – 18.4

Small to Medium Scale (≤ 70B)

Qwen3-4B-Thinking-2507 4B 65.8 81.6 81.0 80.8 59.2 57.3 7.3 Qwen3-32B 32B 68.4 81.4 72.9 74.3 56.6 50.0 8.9 DeepSeek-R1-0528-Qwen3-8B 8B 61.1 82.2 76.3 78.0 59.2 57.7 6.9 DeepSeek-R1-Distill-Llama-70B 70B 65.2 70.0 55.3 59.4 36.7 40.2 5.2 Qwen3-4B-Thinking-2507 + OpenScience 4B 53.5 61.7 53.3 53.0 40.0 36.9 4.6 Qwen3-4B-Thinking-2507 + CHIMERA 4B 70.1 86.9 80.7 82.7 65.7 67.0 9.0

1 2 4 8

k (Number of Samples)

60

65

70

75

80

85

90

95

Pass@kPerformance(%)

65.8

72.7

77.5

81.5

70.1

78.9

85.7

Base 90.7

Ours

(a) GPQA-Diamond

1 2 4 8

k (Number of Samples)

5

10

15

20

25

Pass@kPerformance(%)

7.3

10.8

14.9

19.5

9.0

13.2

18.5

24.0

Base Ours

(b) HLE

Figure 3. Pass@k results on GPQA-Diamond and HLE. Fine-tuning on CHIMERA consistently improves pass@k, indicating enhanced reasoning coverage and improved solution discovery under increased sampling.

- 3.2. Main Results

is primarily due to its reliance on multiple-choice problem formats. Compared to free-form reasoning tasks, multiplechoice questions typically require less explicit multi-step reasoning and allow models to exploit elimination strategies as a shortcut rather than reasoning from scratch.

- Table 2 summarizes performance across all reasoning benchmarks. Fine-tuning the base model on CHIMERA consistently yields gains on multiple challenging benchmarks, including GPQA-Diamond (+4.3), AIME24 (+5.3), HMMT Feb 25 (+6.5), HMMT Nov 25 (+9.7) and HLE (+1.7). Notably, despite being only a 4B-parameter model, our finetuned model becomes competitive with substantially larger models. For example, it matches or surpasses 8B–70B scale baselines on almost all the benchmarks and approaches the performance of models two orders of magnitude larger (e.g., DeepSeek-R1, Qwen3-235B-A22B). This highlights the strong data efficiency of our data synthesis pipeline: a carefully constructed synthetic reasoning dataset can boost modern LLMs’ reasoning capability effectively.

Despite being substantially smaller than existing public synthetic datasets, our dataset yields stronger and consistent performance gains. This highlights the importance of data quality, broad and structured subject coverage, and detailed reasoning traces for improving reasoning capabilities in modern LLMs.

#### 3.3. Inference-Time Scaling Performance

We further examine whether the gains from training on our synthesized dataset persist under inference-time scaling. Following standard practice in reasoning benchmarks (Zhu et al., 2025), we report pass@k performance for k ∈ {1,2,4,8}.

In contrast, fine-tuning on the OpenScience dataset leads to worse downstream performance than the base model across benchmarks. We hypothesize that this degradation

Figure 3 compares the base model and the trained model on GPQA-Diamond and HLE. Across both benchmarks, the trained model consistently outperforms the base model for all values of k. On GPQA-Diamond, the performance gap widens as k increases, reaching 90.7% versus 81.5% at pass@8. A similar pattern is observed on HLE, where pass@1 improves from 7.3% to 9.0%, and pass@8 from 19.5% to 24.0%.

Importantly, the consistent gains across increasing sampling budgets suggest that the improvements are not confined to better single-shot predictions. Instead, they reflect enhanced reasoning robustness and a broader coverage of valid solution trajectories. This behavior is aligned with the design of CHIMERA, which emphasizes long-horizon, multi-step reasoning and detailed solution supervision. As a result, the trained model not only improves accuracy but also benefits more effectively from inference-time scaling.

#### 3.4. SFT-Only Performance on CHIMERA

We evaluate the effect of supervised fine-tuning on CHIMERA without additional reinforcement learning. Starting from the base model Qwen3-4B-Thinking-2507, we perform SFT on problems whose reasoning trajectories are verified as correct during solution synthesis.

- Table 3. Reasoning benchmark performance of the base model, the SFT model trained on CHIMERA, and the subsequent RL model. SFT alone accounts for the majority of performance gains across benchmarks, with RL providing additional improvements.

### 4. Analysis

#### 4.1. Data Difficulty Analysis

We analyze the difficulty of existing synthetic reasoning datasets and compare them with CHIMERA. An effective reasoning dataset should present sufficient challenge to strong base models; otherwise, it provides limited learning signal and is unlikely to further improve reasoning capability.

To quantify dataset difficulty, we evaluate the base model Qwen3-4B-Thinking-2507 without additional fine-tuning. We randomly sample 30K examples from OpenScience, 20K from OpenR1-Math-220K, 10K from DeepMath-103K, and use the full DAPO-Math-17K dataset, then compute the model’s solution accuracy on each. We then compare these results with the model’s performance on CHIMERA.

88.6

DAPO-Math-17K

88.2

DeepMath-103K

87.2

OpenScience

76.4

OpenR1-Math-220K

37.5

CHIMERA

0 20 40 60 80 100

Accuracy (%)

Figure 4. Accuracy of Qwen3-4B-Thinking-2507 on existing synthetic datasets and CHIMERA. The base model achieves near-saturation performance on prior datasets, whereas CHIMERA remains substantially more challenging.

Benchmark Base SFT SFT + RL GPQA-D 65.8 68.8 70.1

- AIME24 81.6 86.5 86.9
- AIME25 81.0 79.8 80.7
- AIME26 80.8 80.3 82.7 HMMT Feb 25 59.2 63.1 65.7 HMMT Nov 25 57.3 66.3 67.0 HLE 7.3 9.0 9.0

As shown in Table 3, SFT alone already leads to substantial improvements over the base model across multiple reasoning benchmarks, including GPQA-Diamond (+3.0), AIME24 (+4.9), HMMT Feb 25 (+3.9), HMMT Nov 25 (+9.0) and HLE (+1.7).

The consistent improvements across competition-style and long-horizon benchmarks indicate that the synthesized dataset alone is sufficient to significantly strengthen reasoning performance. While reinforcement learning can provide further incremental gains, the majority of improvements are already achieved through SFT, highlighting the quality and difficulty of CHIMERA.

As shown in Figure 4, the base model achieves high accuracy on existing synthetic datasets, reaching approximately 88% on DAPO-Math-17K, DeepMath-103K, and OpenScience, and approximately 76% on OpenR1-Math-220K. These near-saturation results indicate that many problems in these datasets pose limited difficulty for current reasoning models. In contrast, CHIMERA is substantially more challenging: the same model achieves only 37.5% accuracy, leaving considerable headroom for improvement.

Overall, these findings indicate that prior synthetic datasets may not provide sufficient difficulty to meaningfully advance current LLMs. By explicitly constructing harder, multi-step reasoning problems with longer solution trajectories, CHIMERA delivers a stronger training signal, which we find critical for further improving reasoning performance.

#### 4.2. Data Quality Analysis

Beyond empirical performance, we conduct a qualitative study to assess whether the synthesized problems are comparable in clarity and difficulty to human-written problems using an LLM-as-a-Judge protocol (Zheng et al., 2023). We perform a blind scoring experiment comparing problems

generated by our pipeline with human-curated problems from HLE.

Specifically, we randomly sample 100 mathematics and 100 physics problems from HLE, and 100 mathematics and 100 physics problems from our synthesized training set generated by gpt-5. In addition, we regenerate 100 mathematics and 100 physics problems using gemini-3-pro. To control for topic distribution, the regenerated problems are conditioned on the same topics as those used for the gpt-5 generation. This yields three sources of problems: (i) HLE (human-written), (ii) gpt-5-generated, and (iii) gemini-3-pro-generated.

GPT-5 Gemini-3-Pro HLE

2.75

2.67

| |
|---|

2.53

| |
|---|

AverageScore(higherisbetter)

2.50

2.20 2.18

2.25

2.00

1.75

1.50

1.30

1.25

1.14

1.00

Evaluator: o4-mini Evaluator: gemini-2.5-pro

Figure 5. Average problem quality scores across sources, evaluated by o4-mini and gemini-2.5-pro. Under both evaluators, LLM-generated problems receive higher average scores than human-curated problems in this ranking protocol.

For each trial, we construct a set of three problems containing one example from each source, randomly shuffle their order, and ask an LLM to rank them by overall quality. The judge assigns a score of 3 to the best, 2 to the middle, and 1 to the worst, considering clarity, well-posedness, and reasoning depth. We repeat this procedure across all sampled problems and report the average score for each source.

In choosing evaluators, we intentionally decouple generation and judging. We use the strongest available models (gpt-5 and gemini-3-pro) as generators to approximate state-of-the-art problem quality, while employing different models as judges to mitigate potential self-preference effects (Chen et al., 2025b). Since ranking problem statements is typically easier than generating them, we use two independent but sufficiently capable evaluators o4-mini and gemini-2.5-pro.

As shown in Figure 5, the results are grouped by evaluator. Under o4-mini, the average scores for problems from HLE, gpt-5, and gemini-3-pro are 1.14, 2.67, and 2.20, respectively. Under gemini-2.5-pro, the corresponding scores are 1.30, 2.53, and 2.18. While minor differences exist between judges, both exhibit consistent

trends and assign higher average scores to LLM-generated problems. These findings suggest that, under this evaluation protocol, synthetic problems are perceived to be of comparable quality to human-curated problems in terms of clarity, well-posedness, and reasoning depth.

#### 4.3. Data Contamination Analysis

To ensure that performance gains are not attributable to unintended data leakage, we conduct a decontamination analysis between our synthesized training data and two evaluation benchmarks: GPQA-Diamond and HLE. Following prior work (Brown et al., 2020; Touvron et al., 2023), we measure lexical overlap using an n-gram similarity score.

Let T denote the set of synthetic training questions and S denote the set of test questions for a benchmark. For each training question ti ∈ T , we compute its maximum n-gram Jaccard similarity with all test questions and then average these maximum scores over the training set:

|T |

|Gn(ti) ∩ Gn(s)| |Gn(ti) ∪ Gn(s)|

1 |T |

Scoren =

, (1)

max

s∈S

i=1

where Gn(·) denotes the set of distinct n-grams extracted from a question. A lower score indicates weaker lexical overlap and lower risk of contamination.

Table 4. n-gram Jaccard overlap between CHIMERA and test benchmarks.

Benchmark 8-gram 13-gram

GPQA-Diamond 0 0 HLE 7.89 × 10−6 0

We report contamination results using stricter 8-gram and 13-gram overlaps, which are commonly adopted for nearduplicate detection. As shown in Table 4, the average maximum overlap ratios are effectively zero across both benchmarks. In particular, both 8-gram and 13-gram overlaps on GPQA-Diamond are exactly 0, while HLE exhibits only a negligible 8-gram similarity of 7.89 × 10−6 and 0 for 13-grams. These near-zero scores indicate an absence of meaningful lexical overlap or memorized instances between our synthesized training data and the evaluation benchmarks, suggests that the observed performance improvements are unlikely to be driven by data contamination.

#### 4.4. Case Study

Each problem in CHIMERA is designed to be self-contained, verifiable, and reasoning-focused. As illustrated in Table 15, each sample includes the following components: Subject and Topic for hierarchical categorization, a formally stated Problem requiring multi-step deductive reasoning, a concise and verifiable Answer, a step-by-step Detailed Solu-

Table 5. A data sample from CHIMERA under the physics subject.

Subject: Physics Topic: Topological Quantum Field Theory Problem

Consider the 2+1-dimensional untwisted Dijkgraaf–Witten finite gauge TQFT with gauge group G = Zn (the cyclic group of order n). In this theory, the partition function on a closed oriented 3-manifold M is defined by

Z(M) = |Hom(π1(M), Zn)|/n.

Let L(p, q) be a lens space, and use the standard fact that π1(L(p, q)) ∼= Zp (a cyclic group of order p), independent of q. Compute Z(L(p, q)) as a function of n and p. Here gcd(a, b) denotes the greatest common divisor.

###### Answer:

|Z(L(p, q)) =<br><br>gcd(n, p) n<br><br>|
|---|

Detailed Solution Okay, let us try to figure out this problem step by step. First, let us recall what a lens space is. A lens space L(p, q) is a 3manifold constructed by performing surgery on the 3-sphere S3 with a single handle, ..., the computation is straightforward: since π1(L(p, q)) ∼= Zp, the number of homomorphisms is gcd(p, n), and thus Z(L(p, q)) = gcd(np,n). Correctness: True

tion suitable for SFT, and a Correctness label confirming solution validity. More cases can be found in Appendix C.

### 5. Related Work 5.1. Datasets for LLM Reasoning

A growing number of benchmarks have been proposed to evaluate LLM reasoning across mathematics, science, and general knowledge domains. Early datasets such as GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021b) emphasize multi-step mathematical problem solving, while science-oriented benchmarks including ARC (Clark et al., 2018), SciQ (Welbl et al., 2017), and OpenBookQA (Mihaylov et al., 2018) focus on structured question answering across physics, chemistry, and biology. Broader evaluation suites such as BIG-bench (Srivastava et al., 2023), BBH (Suzgun et al., 2023), and MMLU (Hendrycks et al., 2021a) probe compositional and cross-domain reasoning abilities.

More recent efforts introduce increasingly difficult reasoning benchmarks. GPQA (Rein et al., 2024) presents graduate-level, retrieval-resistant problems requiring deep conceptual reasoning. Humanity’s Last Exam (HLE) (Phan et al., 2025) stress-tests frontier models with expertcurated, high-difficulty questions across advanced do-

mains. LiveBench (White et al., 2025) further emphasizes contamination-aware and continuously updated evaluation.

Despite the rapid emergence of challenging reasoning benchmarks, relatively fewer datasets are tailored for training advanced reasoning models. As shown in Section 4.1, many existing training datasets exhibit near-saturation performance for modern LLMs. Our work bridge this gap by constructing a compact yet high-difficulty reasoning dataset specifically designed for complex, long-horizon post-training.

#### 5.2. LLMs for Data Generation

To reduce reliance on costly human annotation, recent work leverages LLMs to automatically synthesize training data for instruction following and reasoning. Self-Instruct (Wang et al., 2023) and subsequent efforts such as Stanford Alpaca (Taori et al., 2023) demonstrate that models can bootstrap instruction–response pairs for fine-tuning, achieving performance competitive with systems trained on proprietary data. Extensions such as Evol-Instruct (Xu et al.,

- 2024a) and InstructZero (Chen et al., 2024b) further improve diversity and difficulty by evolving prompts or optimizing instruction generation. Beyond instruction synthesis, works such as UltraFeedback (Cui et al., 2024) show that LLMs can also generate large-scale feedback and preference data for alignment.

In mathematical reasoning, recent datasets including JiuZhang3.0 (Zhou et al., 2024), SkyworkMath (Zeng et al., 2024), DeepMath-103K (He et al.,

- 2025), OpenThoughts (Guha et al., 2025), and OpenScience (NVIDIA Corporation, 2025) explore scalable synthetic or semi-synthetic pipelines with step-by-step solutions and verifiable answers. These efforts highlight the importance of data size, quality control and contamination-aware construction for effective supervision.

However, despite the promise of large-scale synthetic generation, a key open question remains: can modern LLMs generate reasoning problems that match the quality of expertcurated data? Our experimental results show that LLMgenerated data can substantially improve downstream reasoning performance. Furthermore, as shown in Section 4.2, blind evaluations indicate that LLM-generated problems are rated on par with human-curated benchmarks in terms of clarity and reasoning depth. Together, these findings provide strong empirical evidence that LLM-driven synthetic data is a scalable and cost-effective alternative to manual curation for advancing reasoning capabilities.

### 6. Conclusion

We identify three core data-centric barriers to scalable reasoning post-training—cold-start supervision, limited domain coverage, and costly human annotation—and demon-

strate that these challenges can be mitigated through carefully designed synthetic data. We introduce CHIMERA, a compact dataset featuring long Chain-of-Thought trajectories, broad scientific coverage, and fully automated quality control. Despite its modest size, post-training on CHIMERA enables a 4B model to achieve strong performance across diverse reasoning benchmarks, reaching parity with substantially larger models like DeepSeek-R1 and Qwen3-235B. Overall, our results suggest that structured, high-quality synthetic data, rather than scale alone, plays a central role in enabling effective reasoning capabilities in LLMs.

### Acknowledgements

We would like to thank Yuhong Li for valuable feedback and discussions during the early stage of this work.

### References

AIME. AIME Problems and Solutions. https: //artofproblemsolving.com/wiki/index. php/AIME_Problems_and_Solutions, 2025.

Anthropic. Claude 3.7 sonnet. https://www. anthropic.com/news/claude-3-7-sonnet,

2025. Accessed: 2026-02-11.

Balunovi´c, M., Dekoninck, J., Petrov, I., Jovanovi´c, N., and Vechev, M. Matharena: Evaluating llms on uncontaminated math competitions, February 2025. URL https://matharena.ai/.

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. In NeurIPS, 2020.

Chen, A., Li, A., Gong, B., Jiang, B., Fei, B., Yang, B., Shan, B., Yu, C., Wang, C., Zhu, C., et al. Minimaxm1: Scaling test-time compute efficiently with lightning attention. CoRR, abs/2506.13585, 2025a.

Chen, H., Waheed, A., Li, X., Wang, Y., Wang, J., Raj, B., and Abdin, M. I. On the diversity of synthetic data and its impact on training large language models. CoRR, abs/2410.15226, 2024a.

Chen, L., Chen, J., Goldstein, T., Huang, H., and Zhou, T. Instructzero: Efficient instruction optimization for blackbox large language models. In ICML. OpenReview.net, 2024b.

Chen, W., Wei, Z., Zhu, X., Feng, S., and Meng, Y. Do LLM evaluators prefer themselves for a reason? CoRR, abs/2504.03846, 2025b.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge. ArXiv, abs/1803.05457, 2018. URL https://api.

semanticscholar.org/CorpusID:3922816.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems. ArXiv, abs/2110.14168, 2021.

Cui, G., Yuan, L., Ding, N., Yao, G., He, B., Zhu, W., Ni, Y., Xie, G., Xie, R., Lin, Y., Liu, Z., and Sun, M. ULTRAFEEDBACK: boosting language models with scaled AI feedback. In ICML. OpenReview.net, 2024.

Gandhi, K., Chakravarthy, A., Singh, A., Lile, N., and Goodman, N. D. Cognitive behaviors that enable selfimproving reasoners, or, four habits of highly effective stars. arXiv preprint arXiv:2503.01307, 2025.

Guha, E. K., Marten, R., Keh, S., Raoof, N., Smyrnis, G., Bansal, H., Nezhurina, M., Mercat, J., Vu, T., Sprague, Z., Suvarna, A., Feuer, B., Chen, L., Khan, Z., Frankel, E., Grover, S., Choi, C., Muennighoff, N., Su, S., Zhao, W., Yang, J., Pimpalgaonkar, S., Sharma, K., Ji, C. C., Deng, Y., Pratt, S. M., Ramanujan, V., Saad-Falcon, J., Li, J., Dave, A., Albalak, A., Arora, K., Wulfe, B., Hegde, C., Durrett, G., Oh, S., Bansal, M., Gabriel, S., Grover, A., Chang, K., Shankar, V., Gokaslan, A., Merrill, M. A., Hashimoto, T., Choi, Y., Jitsev, J., Heckel, R., Sathiamoorthy, M., Dimakis, A. G., and Schmidt, L. Openthoughts: Data recipes for reasoning models. CoRR, abs/2506.04178, 2025.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

He, Z., Liang, T., Xu, J., Liu, Q., Chen, X., Wang, Y., Song, L., Yu, D., Liang, Z., Wang, W., Zhang, Z., Wang, R., Tu, Z., Mi, H., and Yu, D. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. CoRR, abs/2504.11456, 2025.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021a.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021b.

Hugging Face. Open R1: A Fully Open Reproduction of DeepSeek-R1. https://github.com/ huggingface/open-r1, January 2025. GitHub repository.

Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al. OpenAI o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., Brahman, F., Miranda, L. J. V., Liu, A., Dziri, N., Lyu, S., et al. T¨ulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Lee, H., Phatale, S., Mansoor, H., Mesnard, T., Ferret, J., Lu, K. R., Bishop, C., Hall, E., Carbune, V., Rastogi, A., et al. Rlaif vs. rlhf: Scaling reinforcement learning from human feedback with ai feedback. In International Conference on Machine Learning, pp. 26874–26901. PMLR, 2024.

LI, J., Beeching, E., Tunstall, L., Lipkin, B., Soletskyi, R., Huang, S. C., Rasul, K., Yu, L., Jiang, A., Shen, Z., Qin, Z., Dong, B., Zhou, L., Fleureau, Y., Lample, G., and Polu, S. Numinamath. [https://huggingface.co/AI-MO/

NuminaMath-CoT](https://github.com/ project-numina/aimo-progress-prize/ blob/main/report/numina_dataset.pdf),

2024.

Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A. Can a suit of armor conduct electricity? a new dataset for open book question answering. pp. 2381–2391. Association for Computational Linguistics, October-November 2018. doi: 10.18653/v1/D18-1260. URL https:

//aclanthology.org/D18-1260/.

Muennighoff, N., Yang, Z., Shi, W., Li, X. L., Fei-Fei, L., Hajishirzi, H., Zettlemoyer, L., Liang, P., Cand`es, E., and Hashimoto, T. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.

NVIDIA Corporation. Openscience. https: //huggingface.co/datasets/nvidia/ OpenScience, 2025.

OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023.

Phan, L., Gatti, A., Han, Z., Li, N., Hu, J., Zhang, H., Zhang, C. B. C., Shaaban, M., Ling, J., Shi, S., Choi, M., et al. Humanity’s last exam, 2025. URL https: //arxiv.org/abs/2501.14249.

Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024. URL https:

//openreview.net/forum?id=Ti67584b98.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A. M., Abid, A., Fisch, A., Brown, A. R., Santoro, A., Gupta, A., Garriga-Alonso, A., et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https:// openreview.net/forum?id=uyTL5Bvosj. Featured Certification.

Suzgun, M., Scales, N., Sch¨arli, N., Gehrmann, S., Tay, Y., Chung, H. W., Chowdhery, A., Le, Q., Chi, E., Zhou, D., and Wei, J. Challenging BIG-bench tasks and whether chain-of-thought can solve them. pp. 13003–13051. Association for Computational Linguistics, July 2023. doi: 10.18653/v1/2023.findings-acl. 824. URL https://aclanthology.org/2023.

findings-acl.824/.

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., and Hashimoto, T. B. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/ stanford_alpaca, 2023.

Team, G. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. CoRR, abs/2507.06261, 2025.

Team, K., Du, A., Gao, B., Xing, B., Jiang, C., Chen, C., Li, C., Xiao, C., Du, C., Liao, C., et al. Kimi k1. 5: Scaling reinforcement learning with LLMs. arXiv preprint arXiv:2501.12599, 2025.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. CoRR, abs/2307.09288, 2023.

- Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., and Hajishirzi, H. Self-instruct: Aligning language models with self-generated instructions. In ACL

(1), pp. 13484–13508. Association for Computational Linguistics, 2023.

- Wang, Z., Zhou, F., Li, X., and Liu, P. OctoThinker: Midtraining Incentivizes Reinforcement Learning Scaling,

June 2025. URL http://arxiv.org/abs/2506.

20512. arXiv:2506.20512 [cs].

Welbl, J., Liu, N. F., and Gardner, M. Crowdsourcing multiple choice science questions. pp. 94–106. Association for Computational Linguistics, September 2017. doi: 10.18653/v1/W17-4413. URL https:

//aclanthology.org/W17-4413/.

White, C., Dooley, S., Roberts, M., Pal, A., Feuer, B., Jain, S., Shwartz-Ziv, R., Jain, N., Saifullah, K., Dey, S., Shubh-Agrawal, et al. Livebench: A challenging, contamination-free LLM benchmark. In The Thirteenth International Conference on Learning Representations, 2025.

Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J., Tao, C., Lin, Q., and Jiang, D. Wizardlm: Empowering large pre-trained language models to follow complex instructions. In ICLR. OpenReview.net, 2024a.

Xu, Z., Jiang, F., Niu, L., Deng, Y., Poovendran, R., Choi, Y., and Lin, B. Y. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. ArXiv, abs/2406.08464, 2024b. URL https://arxiv.org/ pdf/2406.08464.pdf.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Yeo, E., Tong, Y., Niu, M., Neubig, G., and Yue, X. Demystifying long chain-of-thought reasoning in llms. arXiv preprint arXiv:2502.03373, 2025.

Yu, L., Jiang, W., Shi, H., YU, J., Liu, Z., Zhang, Y., Kwok, J., Li, Z., Weller, A., and Liu, W. MetaMath: Bootstrap your own mathematical questions for large language models. In The Twelfth International Conference on Learning Representations, 2024.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., YuYue, Dai, W., Fan, T., Liu, G., Liu, J., Liu, L., Liu, X., Lin, H., Lin, Z., Ma, B., Sheng, G., Tong, Y., Zhang, C., Zhang, M., Zhang, R., Zhang, W., Zhu, H., Zhu, J., Chen, J., Chen, J., Wang, C., Yu, H., Song, Y., Wei, X., Zhou, H., Liu, J., Ma, W.-Y., Zhang, Y.-Q., Yan, L., Wu, Y., and Wang, M. DAPO: An open-source LLM reinforcement learning system at scale. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum? id=2a36EMSSTp.

Yu, Y., Zhuang, Y., Zhang, J., Meng, Y., Ratner, A. J., Krishna, R., Shen, J., and Zhang, C. Large language model as attributed training data generator: A tale of diversity and bias. In NeurIPS, 2023.

Zeng, L., Zhong, L., Zhao, L., Wei, T., Yang, L., He, J., Cheng, C., Hu, R., Liu, Y., Yan, S., Fang, H., and Zhou, Y. Skywork-math: Data scaling laws for mathematical reasoning in large language models - the story goes on. CoRR, abs/2407.08348, 2024.

Zeng, W., Huang, Y., Liu, Q., Liu, W., He, K., Ma, Z., and He, J. SimpleRL-Zoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025.

Zheng, L., Chiang, W., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., and Stoica, I. Judging llm-as-a-judge with mt-bench and chatbot arena. In NeurIPS, 2023.

Zhou, K., Zhang, B., Wang, J., Chen, Z., Zhao, X., Sha, J., Sheng, Z., Wang, S., and Wen, J. Jiuzhang3.0: Efficiently improving mathematical reasoning by training small data synthesis models. In NeurIPS, 2024.

Zhu, X., Xia, M., Wei, Z., Chen, W.-L., Chen, D., and Meng, Y. The surprising effectiveness of negative reinforcement in llm reasoning. arXiv preprint arXiv:2506.01347, 2025.

### A. Prompts

Table 6. Prompts used for the data synthesis pipline. Usage Prompt Template

Subject Expansion I have a broad subject. Generate a list of more specific topics that fall under this subject. Each topic should be unique and different from the other topics. The topics should be an important field of study within this subject. Here is the broad subject: {subject}. Be as specific and diverse as possible when listing topics. For example, subject “Probability Theory” could have topics like “Probability Distributions”, “Random Variable”, “Conditional Probability”, “Bayesian Statistics”, etc. Output should be in the format of a list of topics: <topic1>\n<topic2>\n<topic3>\n...

Problem Generation Given a topic, generate a new problem that falls under this topic. The problem should be at a PhD level or higher. Here is the topic: {topic}. Your answer must strictly follow the format and criteria below, without any markdown formatting: Problem: A clearly stated problem that requires advanced knowledge, techniques, or reasoning related to the topic. The problem must have a single, unambiguous, and objectively verifiable answer (e.g., a specific numeric value, closed-form expression, proof, or well-defined result). Clearly define all variables, parameters, and assumptions to avoid ambiguity. Reasoning: A detailed, logical reasoning process that shows how to solve the problem step by step, ensuring each step is verifiable and reproducible. Solution: A complete and rigorous solution derived from the reasoning steps above, including all necessary calculations or proofs. Answer: The final, concise answer to the problem. It must be clearly checkable and easy to verify as correct.

Solution Synthesis <|im start|>|user\n{problem}<|im end|>\n Problem Validator Given a topic and a problem with the answer, verify whether the problem and answer are valid and

satisfy the requirements. Topic: {topic} Problem: {problem} Answer: {answer} Check the following:

- The problem matches the topic and is at a PhD level or higher. - The statement is clear, selfcontained, and unambiguous. - There is exactly one objectively verifiable answer. - The final answer is correct. Your answer must strictly follow the format and criteria below, without any markdown formatting: Verdict: VALID or INVALID Reason: one short sentence explaining the decision

Correctness Verifier Judge whether the following predicted solution to question is correct or not based on the precise and unambiguous correct answer below. question: {question} correct answer: {correct answer} predicted solution: {predicted solution} Your judgement must be in the format and criteria specified below: extracted final answer: The final exact answer extracted from the predicted solution. Put the extracted answer as ’None’ if there is no exact, final answer to extract from the predicted solution. reasoning: Explain why the extracted final answer is correct or incorrect based on correct answer, focusing only on if there are meaningful differences between correct answer and the extracted final answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than correct answer, focus only on whether the answers match. correct: Answer ’yes’ if extracted final answer matches the correct answer given above, or is within a small margin of error for numerical problems. Answer ’no’ otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect. confidence: The extracted confidence score between 0% and 100% from predicted solution. Put 100 if there is no confidence score available.

### B. Subjects and Topics

Table 7. Subjects and their expanded topic sets used for creating CHIMERA. Subject Topic

Mathmematics ‘3-manifolds and hyperbolic structures’, ‘4-manifolds and smooth structures’, ‘Abelian varieties, Jacobians, and Pryms’, ‘Absoluteness and reflection principles’, ‘Actuarial science and risk theory’, ‘Additive combinatorics’, ‘Additive combinatorics and sum-product phenomena’, ‘Additive number theory and sumset problems’, ‘Adic and perfectoid spaces’, ‘Algebra’, ‘Algebraic coding theory’, ‘Algebraic combinatorics’, ‘Algebraic cycles and Chow groups’, ‘Algebraic geometry and moduli’, ‘Algebraic geometry and tropical combinatorics’, ‘Algebraic number theory’, ‘Algebraic spaces and stacks’, ‘Algebraic topology’, ‘Algebraic topology and homotopy theory’, ‘Algorithmic class field theory’, ‘Algorithms and randomized algorithms’, ‘Analysis’, ‘Analytic combinatorics and singularity analysis’, ‘Analytic number theory’, ‘Applications to analysis and Banach spaces’, ‘Applied combinatorics in networks and data’, ‘Applied probability’, ‘Applied topology and persistent homology’, ‘Approximate counting and correlation decay’, ‘Approximation algorithms for graph problems’, ‘Approximation theory and splines’, ‘Arithmetic combinatorics’, ‘Arithmetic geometry and Arakelov theory’, ‘Arithmetic geometry and Diophantine geometry’, ‘Arithmetic statistics’, ‘Arithmetic statistics and Cohen–Lenstra heuristics’, ‘Association schemes and distance-regular graphs’, ‘Automated reasoning and theorem proving’, ‘Automated theorem proving for mathematics’, ‘Automorphic forms and the Langlands program’, ‘Automorphism groups and Cayley graphs’, ‘Axiomatic frameworks and foundations of mathematics’, ‘Axiomatic set theory (ZFC and alternatives)’, ‘BSD conjecture and L-series of elliptic curves’, ‘Backward SDE and FBSDE’, ‘Bandits and online learning’, ‘Bayesian nonparametrics and Dirichlet processes’, ‘Bayesian statistics and Bayesian nonparametrics’, ‘Bijective combinatorics’, ‘Bioinformatics and phylogenetics’, ‘Biophysics and neuroscience models’, ‘Biostatistics and clinical trials’, ‘Bipartite and multipartite graphs’, ‘Birational geometry and minimal model program’, ‘Blockchain primitives and number-theoretic protocols’, ‘Borel equivalence relations and classification theory’, ‘Branching processes and coalescent theory’, ‘Brauer groups and central simple algebras’, ‘Bridgeland stability conditions’, ‘Brill–Noether theory and syzygies’, ‘Brownian motion and Gaussian processes’, ‘CAT(0) and Alexandrov spaces’, ‘Calabi–Yau manifolds and special holonomy’, ‘Calabi–Yau varieties and K3 surfaces’, ‘Calculus of variations’, ‘Calibrations and special Lagrangians’, ‘Cardinal characteristics of the continuum’, ‘Category theory and topos theory’, ‘Causal inference under uncertainty’, ‘Chabauty–Coleman methods and p-adic integration’, ‘Chaos and bifurcation theory’, ‘Character sheaves and Springer theory’, ‘Cheeger inequalities and isoperimetry on graphs’, ‘Chip-firing games and sandpile groups’, ‘Chordal, interval, and comparability graphs’, ‘Chromatic invariants and Tutte polynomials’, ‘Class field theory’, ‘Classical algebraic geometry (varieties and maps)’, ‘Classifying spaces and characteristic classes’, ‘Clique-width and rank-width’, ‘CoHAs and categorification in topology’, ‘Coarse and controlled topology’, ‘Cobordism theory and Thom spectra’, ‘Coding theory and combinatorial designs’, ‘Coding theory and compressed sensing’, ‘Coding theory and number-theoretic constructions’, ‘Combinatorial commutative algebra’, ‘Combinatorial optimization’, ‘Combinatorial representation theory’, ‘Combinatorial set theory’, ‘Combinatorics for ML and graph neural networks’, ‘Combinatorics in statistical physics (Ising, Potts, dimers)’, ‘Combinatorics of polynomials and real stable polynomials’, ‘Commutative algebra’, ‘Comparison geometry and Ricci curvature’, ‘Complex and K¨ahler geometry’, ‘Computability and recursion theory’, ‘Computational algebra and Gr¨obner bases’, ‘Computational algebraic geometry and Gr¨obner bases’, ‘Computational and algorithmic number theory’, ‘Computational and applied mathematics’, ‘Computational geometry’, ‘Concentration of measure and isoperimetry’, ‘Conformal and CR geometry’, ‘Conformal and topological quantum field theories’, ‘Conformal prediction and uncertainty quantification in ML’, ‘Constructible universe L and fine structure’, ‘Constructive and intuitionistic mathematics’, ‘Contact geometry and Reeb dynamics’, ‘Container method and hypergraph containers’, ‘Continuum theory and Peano continua’, ‘Control and observability on networks’, ‘Control theory and systems theory’, ‘Convex geometry and Brunn–Minkowski theory’, ‘Core model theory and extender models’, ‘Coupling methods and contractivity’, ‘Covering spaces and fundamental group’, ‘Cryptography and algebraic cryptography’, ‘Cryptography and post-quantum cryptography’, ‘Data science and manifold learning’, ‘Decorated character varieties and higher Teichm¨uller theory’, ‘Deep learning theory and generalization’, ‘Deligne–Mumford and Artin stacks’, ‘Derived categories of coherent sheaves’, ‘Descriptive set theory’, ‘Descriptive set theory (Borel and projective hierarchies)’, ‘Descriptive set-theoretic dynamics’, ‘Design theory and block designs’, ‘Design theory and finite geometries’, ‘Determinacy axioms (AD, ADL(R))’, ‘Differential geometry’, ‘Diophantine approximation and transcendence’, ‘Diophantine geometry and arithmetic geometry’, ‘Discrete Morse theory’, ‘Discrete and computational geometry’, ‘Discrete conformal geometry and circle packings’, ‘Discrete geometry and incidence geometry’, ‘Discrete geometry and polyhedral combinatorics’, ‘Discrete mathematics’, ‘Disordered systems and spin glasses’

‘Distance-regular graphs and association schemes’, ‘Distributed and local graph algorithms’, ‘Distribution of rational points and unlikely intersections’, ‘Donaldson and Seiberg–Witten theory’, ‘Donaldson–Thomas and Pandharipande–Thomas invariants’, ‘Econophysics and financial mathematics’, ‘Edge coloring and total coloring’, ‘Edge expansion, vertex expansion, and conductance’, ‘Effective descriptive set theory’, ‘Electrical networks and effective resistance’, ‘Elementary number theory’, ‘Elliptic curve methods and isogeny computations’, ‘Elliptic curves and abelian varieties’, ‘Embeddings and metric distortion’, ‘Emerging directions’, ‘Empirical processes and Donsker theorems’, ‘Empirical processes and concentration bounds’, ‘Enumerative and extremal combinatorics’, ‘Enumerative combinatorics and generating functions’, ‘Enumerative geometry and Hurwitz theory’, ‘Environmental and climate risk modeling’, ‘Epidemic models and branching processes’, ‘Epidemics and contagion on networks’, ‘Epidemiology and infectious disease modeling’, ‘Erdos distinct distances and Szemer´edi–Trotter’, ‘Erdos–Ko–Rado and intersecting families’, ‘Ergodic methods on homogeneous spaces’, ‘Ergodic theory and dynamical systems’, ‘Ergodic theory and mixing’, ‘Euclidean and classical geometry’, ‘Exchangeability and de Finetti theorems’, ‘Expander constructions and pseudorandomness’, ‘Expander constructions via number theory’, ‘Expanders and expander constructions’, ‘Exponential sums and the circle method’, ‘Extremal combinatorics’, ‘Extreme value theory and heavy tails’, ‘Extreme value theory and regular variation’, ‘Face numbers, f-vectors, and h-vectors’, ‘Facility location and scheduling’, ‘Fairness and privacy in algorithms’, ‘Fairness and robustness via probabilistic modeling’, ‘Faltings’ theorem and Mordell-type problems’, ‘Fano varieties and rationality problems’, ‘Federated Bayesian learning and distributed inference’, ‘Fibre bundles and principal bundles’, ‘Field theory and Galois theory’, ‘Filtering and state estimation (Kalman, particle filters)’, ‘Financial mathematics and derivatives pricing’, ‘Fine-grained lower bounds for combinatorial problems’, ‘Finite element and spectral methods’, ‘Finite geometries and incidence structures’, ‘Floer homology and pseudo-holomorphic curves’, ‘Flows, cuts, and circulations’, ‘Flows, cuts, and sparsifiers’, ‘Foliations, laminations, and measured laminations’, ‘Forcing and iterated forcing’, ‘Formal methods in cyber-physical systems’, ‘Fractal geometry’, ‘Fractional coloring and circular coloring’, ‘Free probability and noncommutative probability’, ‘Free probability in applications’, ‘Functional CLTs and invariance principles’, ‘Functional analysis and operator theory’, ‘Functional inequalities (Poincar´e, log-Sobolev, transport)’, ‘Galois representations and deformation theory’, ‘Galois theory of number fields’, ‘Game theory and mechanism design’, ‘Gaussian processes and kernel methods’, ‘General relativity and geometric PDE’, ‘Generalized metric spaces and compactness’, ‘Generators, Dirichlet forms, and potential theory’, ‘Generic absoluteness and forcing axioms (MA, PFA, MM)’, ‘Geometric Langlands program’, ‘Geometric analysis’, ‘Geometric analysis and PDE on manifolds’, ‘Geometric deep learning and graph neural networks’, ‘Geometric graphs and unit disk graphs’, ‘Geometric group theory’, ‘Geometric group theory and quasi-isometries’, ‘Geometric measure theory’, ‘Geometric measure theory and currents’, ‘Geometric set cover and epsilon-nets’, ‘Geometric topology’, ‘Geometric topology interfaces’, ‘Geometrization and 3-manifold geometry’, ‘Geometry and topology’, ‘Geometry of Banach spaces’, ‘Geometry of numbers and lattice point problems’, ‘Gibbs measures, spin systems, and phase transitions’, ‘Graph coloring and list coloring’, ‘Graph compression and summarization’, ‘Graph databases and knowledge graphs’, ‘Graph decomposition and minors structure’, ‘Graph drawing and visualization’, ‘Graph embeddings and metric embeddings’, ‘Graph homomorphisms and constraint satisfaction’, ‘Graph isomorphism and canonical labeling’, ‘Graph limits and graphons’, ‘Graph minors and Robertson–Seymour theory’, ‘Graph polynomials (Tutte, chromatic, interlace)’, ‘Graph products and powers’, ‘Graph representation learning and GNNs’, ‘Graph theory and spectral graph theory’, ‘Graph theory and structural graph theory’, ‘Greedoid theory’, ‘Gromov–Witten theory and quantum cohomology’, ‘Group theory’, ‘Hamiltonian systems and symplectic dynamics’, ‘Hardness of approximation and PCP connections’, ‘Harmonic analysis and Fourier analysis’, ‘Heegaard Floer homology and contact invariants’, ‘Heights, Arakelov geometry, and Northcott property’, ‘High-dimensional probability and concentration inequalities’, ‘High-dimensional probability for data science’, ‘Hilbert schemes and Quot schemes’, ‘Hodge theory’, ‘Hodge theory and period mappings’, ‘Hodge theory and variations of Hodge structure’, ‘Holographic and Holant frameworks’, ‘Homological algebra’, ‘Homological mirror symmetry’, ‘Homology, cohomology, and cohomology operations’, ‘Homotopical algebra and model categories’, ‘Homotopy theory and Postnikov towers’, ‘Homotopy type theory in proof assistants’, ‘Hopf algebras and quantum groups’, ‘Hydrology and extreme events’, ‘Hyperbolic geometry and Kleinian groups’, ‘Hypergraph theory’, ‘Hypergraph theory and set systems’, ‘Hypergraphs and uniformity’, ‘Ideals on omega and forcing with ideals’, ‘Imaging science and tomography’, ‘Index theory and Atiyah–Singer theorem’, ‘Infinity categories and higher topos theory’, ‘Information geometry and statistical manifolds’, ‘Information geometry for learning’, ‘Information theory and entropy methods’, ‘Information-theoretic inequalities and entropy methods’, ‘Inner model theory and mice’, ‘Insurance mathematics and solvency modeling’, ‘Integrable probability and last passage percolation’, ‘Integrable systems’, ‘Integral geometry and geometric probability’, ‘Integral polyhedra and total unimodularity’

‘Interacting particle systems and hydrodynamic limits’, ‘Interacting particle systems and mean-field models’, ‘Interactive proof assistants and formalization of mathematics’, ‘Interdisciplinary and applied areas’, ‘Intersection theory and Poincar´e duality’, ‘Intersection theory and Riemann–Roch’, ‘Invariant theory’, ‘Inverse problems and regularization’, ‘Iwasawa theory’, ‘K-theory (algebraic and topological)’, ‘K-theory (topological and algebraic)’, ‘KPZ universality and integrable probability’, ‘Kazhdan–Lusztig theory and Hecke algebras’, ‘Knot theory and link invariants’, ‘Knot theory and low-dimensional topology’, ‘L-functions and functional equations’, ‘L-functions and zeta functions’, ‘Langlands program (classical and geometric)’, ‘Large cardinal axioms’, ‘Large cardinals and elementary embeddings’, ‘Large deviations and Gartner–Ellis theorem’, ‘Latin squares and orthogonal arrays’, ‘Lattice algorithms and LLL’, ‘Levy processes and jump diffusions’, ‘Lie groups and Lie algebras’, ‘Linear systems and base loci’, ‘Liouville quantum gravity and random geometry’, ‘Local fields and ramification’, ‘Local limit theorems and Berry–Esseen bounds’, ‘Logarithmic geometry’, ‘Low-dimensional topology’, ‘Manifold topology and smoothing theory’, ‘Manin conjecture and Peyre constants’, ‘Mapping class groups and surfaces’, ‘Markov chain Monte Carlo and mixing analysis’, ‘Markov chains and Markov processes’, ‘Markov processes and semigroup theory’, ‘Martingales and optional stopping’, ‘Matching theory and matroid intersection’, ‘Matching, matroid parity, and factors’, ‘Mathematical biology and epidemiology’, ‘Mathematical physics’, ‘Matroid theory and oriented matroids’, ‘Mean curvature flow and Ricci flow’, ‘Mean field games’, ‘Mean-field models and McKean–Vlasov equations’, ‘Measurable cardinals and ultrafilters’, ‘Measure theory and integration’, ‘Measure-theoretic probability and integration’, ‘Measure-theoretic probability and stochastic processes’, ‘Metric geometry and Gromov hyperbolicity’, ‘Microlocal analysis and pseudodifferential operators’, ‘Minimal surfaces and geometric flows’, ‘Minimal surfaces and mean curvature flow’, ‘Mirror symmetry and Fukaya categories’, ‘Mirror symmetry and SYZ’, ‘Mixed Hodge structures and degenerations’, ‘Model theory’, ‘Model theory in number theory and o-minimality’, ‘Moderate deviations and LIL’, ‘Modular forms and automorphic forms’, ‘Moduli of curves and stable maps’, ‘Moduli problems and geometric invariant theory’, ‘Monte Carlo and Markov chain mixing’, ‘Monte Carlo methods and variance reduction’, ‘Motives and motivic cohomology’, ‘Multiplier ideals and singular metrics’, ‘Network design and survivable networks’, ‘Network design, reliability, and survivability’, ‘Network inference and community detection’, ‘Network science and complex systems’, ‘Network science and multilayer networks’, ‘Neural differential equations and neural operators’, ‘Non-Euclidean geometry (hyperbolic and spherical)’, ‘Noncommutative algebra’, ‘Noncommutative algebraic geometry’, ‘Noncommutative topology and C*-algebraic topology’, ‘Nonlinear analysis’, ‘Nonstandard analysis’, ‘Number theory’, ‘Numerical analysis and scientific computing’, ‘Numerical linear algebra’, ‘O-minimality and definable sets in AG’, ‘Obstruction theory’, ‘Operations research and mathematical programming’, ‘Operations research under uncertainty’, ‘Operator algebras (C*algebras, von Neumann algebras)’, ‘Optimal experimental design and active learning’, ‘Optimal transport and Wasserstein geometry’, ‘Optimal transport and probability metrics’, ‘Optimal transport for probabilistic modeling’, ‘Optimization (convex, nonconvex, integer, semidefinite)’, ‘Orbifolds and stacks’, ‘Order dimension and comparability graphs’, ‘PCF theory and singular cardinals’, ‘Parameterized algorithms and kernels on graphs’, ‘Parameterized complexity and kernelization’, ‘Partial differential equations (elliptic, parabolic, hyperbolic)’, ‘Partition calculus and infinite combinatorics’, ‘Percolation and interacting particle systems’, ‘Percolation and phase transitions’, ‘Percolation and phase transitions on networks’, ‘Perfect graphs and the strong perfect graph theorem’, ‘Perfectoid spaces and p-adic Hodge theory’, ‘Permutation patterns and sorting networks’, ‘Perpetuities and fixedpoint equations in distribution’, ‘Perverse sheaves and D-modules’, ‘Perverse sheaves and microlocal sheaf theory’, ‘Philosophy and history of mathematics’, ‘Physics-informed machine learning and scientific ML’, ‘Planar graphs and graphs on surfaces’, ‘Point-set topology and separation axioms’, ‘Poisson geometry’, ‘Polyhedral combinatorics’, ‘Polyhedral geometry and polytopes’, ‘Polytopes, zonotopes, and associahedra’, ‘Posets, lattices, and M¨obius inversion’, ‘Positivity of line bundles and vanishing theorems’, ‘Post-quantum cryptography and isogeny-based crypto’, ‘Primality testing and integer factorization’, ‘Prime number theory and zero-density estimates’, ‘Privacy-preserving analytics and differential privacy’, ‘Probabilistic combinatorics’, ‘Probabilistic combinatorics and random structures’, ‘Probabilistic numerics’, ‘Probabilistic potential theory’, ‘Probabilistic programming and approximate inference’, ‘Probability and statistics’, ‘Projective and affine geometry’, ‘Proof theory’, ‘Propagation of chaos and chaos decompositions’, ‘Property testing and sublinear algorithms’, ‘Property testing for graph properties’, ‘Pseudo-Riemannian and Lorentzian geometry’, ‘Pseudorandomness and extractor constructions’, ‘Quadratic forms and modular curves’, ‘Quantum field theory (constructive, algebraic, axiomatic)’, ‘Quantum gravity and string theory’, ‘Quantum information theory and quantum error correction’, ‘Quantum mechanics and spectral analysis’, ‘Quantum unique ergodicity and arithmetic quantum chaos’, ‘Queueing theory and network queues’, ‘Quiver varieties and symplectic resolutions’, ‘Ramsey theory’, ‘Ramsey theory and van der Waerden-type problems’, ‘Random Schr¨odinger operators’, ‘Random Schr¨odinger operators and Anderson localization’, ‘Random fields and Gaussian free fields’, ‘Random geometry and SLE in applications’

‘Random graphs and complex networks’, ‘Random graphs and hypergraphs’, ‘Random graphs and percolation theory’, ‘Random graphs and stochastic block models’, ‘Random matrices and free probability’, ‘Random matrices and spectral statistics’, ‘Random matrices and universality’, ‘Random matrix theory and L-function statistics’, ‘Random multiplicative functions’, ‘Random multiplicative functions and pretentious methods’, ‘Random permutations and descents’, ‘Random tilings and dimer models’, ‘Random walk in random environment’, ‘Random walks and diffusion on graphs’, ‘Random walks on groups and heat kernel bounds’, ‘Random walks, mixing times, and cover times’, ‘Random, Cohen, and Sacks forcing’, ‘Rational connectedness and Campana program’, ‘Rational homotopy theory’, ‘Rational points on higher-dimensional varieties’, ‘Real and complex analysis’, ‘Reinforcement learning with probabilistic models’, ‘Reliability engineering and maintenance’, ‘Reliability theory and survival analysis’, ‘Representation theory’, ‘Reverse mathematics’, ‘Ricci flow and comparison geometry’, ‘Riemannian and pseudo-Riemannian geometry’, ‘Riemannian geometry’, ‘Rigid analytic and Berkovich spaces’, ‘Ring theory and module theory’, ‘Risk-sensitive and robust decision-making’, ‘Rough paths and regularity structures’, ‘Sato–Tate equidistribution and automorphic Sato–Tate’, ‘Schemes and morphisms of schemes’, ‘Schmidt subspace theorem and Roth’s theorem’, ‘Schramm–Loewner evolution and conformal invariance’, ‘Schubert calculus and Coxeter combinatorics’, ‘Semimartingales and stochastic integration’, ‘Sequential Monte Carlo and particle methods’, ‘Set theory and foundations’, ‘Set-theoretic topology and selection principles’, ‘Shape theory and ANR spaces’, ‘Sheaf theory and cohomology’, ‘Shimura varieties and automorphic sheaves’, ‘Shimura varieties and modularity’, ‘Sieve methods and additive number theory’, ‘Sieve methods and distribution in arithmetic progressions’, ‘Signal processing and compressed sensing’, ‘Simplicial complexes and collapsibility’, ‘Singularity theory and resolutions’, ‘Spatial statistics and geostatistics’, ‘Spatial-temporal modeling and point processes’, ‘Spectral geometry and isoperimetric inequalities’, ‘Spectral graph theory and Laplacians’, ‘Spectral graph theory and expanders’, ‘Spectral hypergraph theory’, ‘Spectral sequences and derived functors’, ‘Spectral sparsification’, ‘Spectral theory’, ‘Sperner theory and lattice of subsets’, ‘Spin geometry and Dirac operators’, ‘Square and diamond principles’, ‘Stable homotopy and spectra’, ‘Stable homotopy category and spectra’, ‘Stable laws and domains of attraction’, ‘Stable pairs and wall crossing’, ‘Stanley–Reisner theory and face rings’, ‘Stationary sets and club guessing’, ‘Statistical decision theory and inference’, ‘Statistical mechanics and phase transitions’, ‘Statistical mechanics and probabilistic combinatorics’, ‘Steiner systems and finite projective planes’, ‘Stein’s method and normal approximation’, ‘Stochastic PDE and random fields’, ‘Stochastic PDE in applications’, ‘Stochastic analysis on manifolds’, ‘Stochastic calculus (Itˆo and Stratonovich)’, ‘Stochastic calculus and SDE’, ‘Stochastic control and optimal stopping’, ‘Stochastic control, HJB, and viscosity solutions’, ‘Stochastic differential equations and flows’, ‘Stochastic networks and communication systems’, ‘Stochastic optimization and stochastic gradient methods’, ‘Stochastic partial differential equations’, ‘Stratified spaces and intersection homology’, ‘Streaming algorithms for combinatorial problems’, ‘Structural graph theory’, ‘Sub-Riemannian and Finsler geometry’, ‘Sublinear and streaming graph algorithms’, ‘Submodular optimization’, ‘Sumset inequalities and Freiman-type results’, ‘Surface classification and higher-dimensional birational geometry’, ‘Suslin trees and Aronszajn trees’, ‘Symplectic and contact geometry’, ‘Symplectic and contact topology’, ‘Symplectic geometry and Hamiltonian dynamics’, ‘Systems biology and network biology’, ‘TQFT and modular tensor categories’, ‘Teichm¨uller theory and moduli of curves’, ‘Teichm¨uller theory and moduli spaces’, ‘Temporal and dynamic graphs’, ‘Theoretical computer science and complexity’, ‘Theta functions and moduli of vector bundles’, ‘Threshold phenomena and sharp thresholds’, ‘Tilings, packings, and sphere packing’, ‘Time series analysis and stochastic forecasting’, ‘Time-frequency analysis and wavelets’, ‘Topological combinatorics’, ‘Topological data analysis’, ‘Topological data analysis and persistent homology’, ‘Topological dynamics and minimal sets’, ‘Topological graph theory and crossing number’, ‘Topological groups and Lie groups’, ‘Topological methods in combinatorics’, ‘Topological methods in data analysis’, ‘Topological quantum computing’, ‘Topological robotics and configuration spaces’, ‘Topology (point-set, geometric, and algebraic)’, ‘Toric and tropical geometry’, ‘Toric varieties and fans’, ‘Tournament and oriented graphs’, ‘Transcendence and Diophantine approximation’, ‘Treewidth, clique-width, and decompositions’, ‘Treewidth, pathwidth, and decompositions’, ‘Tropical geometry and tropical intersection’, ‘Type theory and homotopy type theory’, ‘Uncertainty quantification’, ‘VC theory and uniform laws of large numbers’, ‘Variational inequalities and complementarity problems’, ‘Waring’s problem and Vinogradov methods’, ‘Weak convergence and Skorokhod topology’, ‘Young tableaux and Schur functions’, ‘Zero-free regions and the Riemann hypothesis’, ‘Zeta functions of graphs’, ‘p-adic Hodge theory and Fontaine modules’, ‘p-adic analysis and Hida families’, ‘p-adic number theory and local fields’, ‘q-series, partitions, and symmetric functions’

Computer Science ‘AI for systems and systems for AI’, ‘AR, VR, and XR interfaces’, ‘Accessibility and inclusive design’, ‘Active learning and optimal data selection’, ‘Algorithms and data structures’, ‘Animation and physicsbased simulation’, ‘Approximation algorithms and PTAS’, ‘AutoML and hyperparameter optimization’, ‘Bayesian deep learning and uncertainty estimation’, ‘Bilevel optimization and hyperparameter tuning’, ‘Binary analysis and software hardening’, ‘Blockchain and distributed ledger technologies’, ‘Category theory in CS’, ‘Causal inference and causal discovery in CS’, ‘Causal representation learning’, ‘Causality in ML and causal discovery’, ‘Cloud computing and virtualization’, ‘Computability and automata theory’, ‘Computational biology and bioinformatics’, ‘Computational chemistry and materials informatics’, ‘Computational complexity theory’, ‘Computational economics and market design’, ‘Computational imaging and computational photography’, ‘Computational neuroscience’, ‘Computer architecture and hardware-software co-design’, ‘Computer graphics and rendering’, ‘Computer networks and programmable data planes’, ‘Computer vision and multimodal learning’, ‘Computer vision and pattern recognition’, ‘Computer-supported cooperative work’, ‘Convolutional, recurrent, and attention-based networks’, ‘Cryptography and cryptographic protocols’, ‘Cyber-physical systems and real-time computing’, ‘Data mining and knowledge discovery’, ‘Data-centric AI and data governance’, ‘Data-centric AI, data governance, and quality’, ‘Database systems and data engineering’, ‘Datacenter architectures and disaggregated systems’, ‘Deep learning and representation learning’, ‘Deep learning architectures’, ‘Differential privacy and federated learning’, ‘Differential privacy and secure computation’, ‘Digital currencies and cryptoeconomics’, ‘Digital twins and cyber-physical security’, ‘Distributed systems and consensus’, ‘Domain generalization and distribution shift’, ‘Edge AI and on-device learning’, ‘Edge computing and IoT’, ‘Education technologies and computing education research’, ‘Embedded systems and robotics platforms’, ‘Energy-aware and green computing’, ‘Energybased models and score matching’, ‘Fairness, accountability, and ethics’, ‘Fairness, accountability, and transparency in AI’, ‘Federated and decentralized learning’, ‘Formal languages and parsing’, ‘Formal methods and model checking’, ‘Foundation models and large language models’, ‘GPU and accelerator computing’, ‘Generative chemistry and materials discovery’, ‘Generative modeling’, ‘Geometric and topological data analysis’, ‘Geometry processing and CAD’, ‘Graph algorithms and network algorithms’, ‘Graph and relational learning for science’, ‘Graph neural networks and geometric deep learning’, ‘Hierarchical and multi-agent RL’, ‘Human-AI interaction and interactive ML’, ‘Humancomputer interaction and UX’, ‘Implicit neural representations and neural fields’, ‘In-memory and approximate computing’, ‘Information retrieval and search’, ‘Information-theoretic learning bounds’, ‘Instruction tuning, RLHF, and RLAIF’, ‘Interpretability and explainability’, ‘Inverse RL and imitation learning’, ‘Knowledge representation and reasoning’, ‘Lambda calculus and functional programming theory’, ‘Large language models and foundation models’, ‘Learning PDE solvers and neural operators’, ‘Logic in computer science’, ‘MLOps and ML systems’, ‘MLOps, deployment, and monitoring’, ‘Machine learning and data-driven methods’, ‘Mechanistic interpretability of neural networks’, ‘Metalearning and few-shot learning’, ‘Model compression, pruning, quantization, distillation’, ‘Multi-task and continual learning’, ‘Multiagent systems and game-theoretic AI’, ‘Natural language processing and LLMs’, ‘Natural language processing and computational linguistics’, ‘Networking for AI workloads’, ‘Neural architecture search and AutoML’, ‘Neurosymbolic AI and program synthesis’, ‘Nonconvex optimization and landscape analysis’, ‘Offline and batch RL’, ‘Online algorithms and competitive analysis’, ‘Online learning and bandits’, ‘Operating systems and kernels’, ‘Optimization for ML’, ‘PAC learning and VC theory’, ‘Parallel and high-performance computing’, ‘Parameterized and fine-grained complexity’, ‘Perception, SLAM, and control in robotics’, ‘Physics-informed ML and operator learning’, ‘Planning, search, and constraint programming’, ‘Policy gradients and actor-critic methods’, ‘Post-Moore architectures and neuromorphic computing’, ‘Post-quantum cryptography’, ‘Privacy engineering and compliance’, ‘Probabilistic modeling and graphical models’, ‘Probabilistic programming and inference engines’, ‘Program analysis and synthesis’, ‘Program synthesis and neuro-symbolic systems’, ‘Programming languages and compilers’, ‘Proof assistants and program verification’, ‘Protein structure prediction and generative biology’, ‘Quantum programming languages and compilers’, ‘Randomized algorithms and probabilistic methods’, ‘Recommender systems and personalization’, ‘Reinforcement learning (model-free and model-based)’, ‘Reinforcement learning and decision-making’, ‘Responsible and safe AI’, ‘Retrieval-augmented generation and tool use’, ‘Robotics and autonomous systems’, ‘Robustness and adversarial ML’, ‘Runtime systems and garbage collection’, ‘Safe and risk-sensitive RL’, ‘Scientific computing and numerical methods’, ‘Scientific machine learning’, ‘Second-order and adaptive optimization’, ‘Secure hardware and side-channel defenses’, ‘Security and privacy’, ‘Self-supervised and contrastive learning’, ‘Semi-supervised and weakly supervised learning’, ‘Serverless computing and microservices’, ‘Smart contracts and formal verification’, ‘Social computing and computational social science’, ‘Software engineering and software architecture’, ‘Software testing, fuzzing, and reliability’, ‘Software-defined networking and NFV’, ‘Speech and audio processing’

‘Speech, audio, and music ML’, ‘Stability and compression-based generalization’, ‘Stochastic gradient methods and variance reduction’, ‘Storage systems and filesystems’, ‘Supervised learning (classification and regression)’, ‘Surrogate modeling and uncertainty quantification’, ‘Systems and network security’, ‘Theoretical ML and generalization’, ‘Theoretical ML and statistical learning theory’, ‘Time series forecasting and anomaly detection’, ‘Transaction processing and concurrency control’, ‘Transfer learning and domain adaptation’, ‘Transformers and sequence modeling’, ‘Trustworthy AI and robustness’, ‘Type systems and program semantics’, ‘Ubiquitous and wearable computing’, ‘Unsupervised learning (clustering, density estimation)’, ‘Usable security and human factors’, ‘VAEs, GANs, normalizing flows, and diffusion models’, ‘Visualization and visual analytics’, ‘Web and internet protocols’, ‘Wireless, mobile, and 5G/6G networks’, ‘World models and model-based planning’

Chemistry ‘Allostery and enzyme regulation’, ‘Analytical chemistry and instrumentation’, ‘Astrochemistry and prebiotic chemistry’, ‘Asymmetric synthesis and chiral pool strategies’, ‘Atmospheric chemistry and aerosols’, ‘Autophagy and apoptosis’, ‘Battery chemistry and solid electrolytes’, ‘Biocatalysis and industrial biochemistry’, ‘Biochemistry and chemical biology’, ‘Bioenergetics and oxidative phosphorylation’, ‘Bioinorganic chemistry and metalloenzymes’, ‘Biosynthetic pathway-inspired synthesis’, ‘Carbocation and carbanion chemistry’, ‘Carbohydrate and peptide synthesis’, ‘Carbohydrate chemistry and glycosylation’, ‘Cascade and multicomponent reactions’, ‘Catalysis (heterogeneous, homogeneous, biocatalysis)’, ‘Catalyst design and ligand development’, ‘Chaperones and protein quality control’, ‘Chemical biology and bioorthogonal chemistry’, ‘Chemical genetics and epigenetic probes’, ‘Chemical kinetics and reaction dynamics’, ‘Cheminformatics and QSAR’, ‘Chemoenzymatic synthesis and biocatalysis’, ‘Chromatography and separations’, ‘Click chemistry’, ‘Click chemistry and bioorthogonal reactions’, ‘Coenzymes and cofactors’, ‘Computational biochemistry and molecular dynamics’, ‘Computational reaction mechanisms and QM/MM’, ‘Coordination chemistry and ligand design’, ‘Covalent inhibitors and PROTACs’, ‘Covalent inhibitors and targeted degradation (PROTACs)’, ‘Cross-coupling (Suzuki, Heck, Negishi, Buchwald–Hartwig)’, ‘Crystal engineering and MOFs/COFs’, ‘Crystallization and polymorphism’, ‘Cytokine signaling and immunometabolism’, ‘C–H activation and functionalization’, ‘C–H activation and functionalization strategies’, ‘DNA repair mechanisms and genome stability’, ‘Density functional theory and ab initio methods’, ‘Directed evolution and enzyme engineering’, ‘Docking and structure-based drug design’, ‘Electrochemistry and electrocatalysis’, ‘Electroorganic synthesis’, ‘Energy catalysis and CO2 utilization’, ‘Environmental chemistry and toxicology’, ‘Enzyme mechanisms and catalysis’, ‘Enzymology and enzyme kinetics’, ‘Epigenetics and chromatin remodeling’, ‘Flow chemistry and microfluidic synthesis’, ‘Flow chemistry and microreactors’, ‘Flux analysis and metabolic control analysis’, ‘Forensic chemistry and analytical forensics’, ‘Fragment-based and structure-based design’, ‘Fragment-based drug discovery’, ‘Geochemistry and cosmochemistry’, ‘Glycobiology and glycoproteins’, ‘Glycolipids and glycoconjugates’, ‘Green chemistry and sustainable processes’, ‘Green solvents and sustainable synthesis’, ‘Heterocyclic chemistry’, ‘Heterocyclic chemistry (N, O, S heterocycles)’, ‘Hydrogenation and asymmetric catalysis’, ‘Hypervalent iodine chemistry’, ‘Immunochemistry and antigen–antibody interactions’, ‘Industrial catalysis and reactor design’, ‘Inorganic chemistry’, ‘Ionic liquids and deep eutectic solvents’, ‘Lipid nanoparticles and drug delivery’, ‘Machine learning in chemistry and autonomous labs’, ‘Macrocyclization and supramolecular synthesis’, ‘Main group chemistry and hypervalent compounds’, ‘Mass spectrometry and proteomics’, ‘Medicinal chemistry and SAR’, ‘Medicinal chemistry and drug design’, ‘Membrane structure and lipidomics’, ‘Membrane transporters and ion channels’, ‘Metabolic engineering and synthetic pathways’, ‘Metabolism and metabolic pathways’, ‘Mitochondrial dynamics and mitophagy’, ‘NMR, EPR, IR, Raman, UV-Vis spectroscopy’, ‘Nanochemistry and colloids’, ‘Nitrogen metabolism and assimilation’, ‘Nuclear chemistry and radiochemistry’, ‘Nucleic acids biochemistry (DNA and RNA)’, ‘Nucleoside and nucleotide chemistry’, ‘Olefin metathesis and ring-closing metathesis’, ‘Organic chemistry and synthesis’, ‘Organic electronics (OLED, OPV, OFET)’, ‘Organic materials and conjugated polymers’, ‘Organocatalysis (enamine, iminium, NHC catalysis)’, ‘Organocatalysis and enamine/iminium catalysis’, ‘Organofluorine and perfluoroalkylation’, ‘Organometallic chemistry and cross-coupling’, ‘Organometallics in organic synthesis’, ‘Peptide synthesis and solid-phase methods’, ‘Pericyclic reactions and orbital symmetry’, ‘Perovskites and photovoltaic materials’, ‘Pharmaceutical biotechnology and biologics’, ‘Photochemistry and photophysics’, ‘Photopharmacology and light-controlled drugs’, ‘Photoredox and electroorganic synthesis’, ‘Photoredox catalysis and energy transfer catalysis’, ‘Photosynthesis biochemistry’, ‘Physical chemistry (thermodynamics, kinetics, spectroscopy)’, ‘Physical organic chemistry and reaction mechanisms’, ‘Polymer chemistry and soft materials’, ‘Post-translational modifications and proteostasis’, ‘Prions, amyloids, and protein aggregation’, ‘Process chemistry and scale-up’, ‘Protecting group strategies and chemoselectivity’, ‘Protein structure, dynamics, and folding’

‘Proteomics and mass spectrometry’, ‘QM/MM and free energy calculations’, ‘Quantum chemistry and electronic structure theory’, ‘RNA processing, splicing, and RNA-based regulation’, ‘Radical chemistry and persistent radicals’, ‘Reaction discovery with ML and autonomous labs’, ‘Reaction mechanisms and physical organic chemistry’, ‘Redox biology and ROS signaling’, ‘Replication, transcription, and translation’, ‘Retrosynthetic analysis and strategy’, ‘Safety, regulation, and chemical education’, ‘Second messengers (cAMP, IP3, Ca2+)’, ‘Self-assembly and host–guest chemistry’, ‘Signal transduction (GPCRs, kinases, phosphatases)’, ‘Solid-state chemistry and materials’, ‘Statistical mechanics and molecular simulation’, ‘Stereochemistry and asymmetric catalysis’, ‘Stereochemistry and conformational analysis’, ‘Structural methods (X-ray, NMR, cryo-EM)’, ‘Structure, bonding, and stereoelectronics’, ‘Supramolecular chemistry and host–guest systems’, ‘Surface science and catalysis characterization’, ‘Systems biochemistry and metabolic regulation’, ‘Theoretical and computational chemistry’, ‘Total synthesis of natural products’, ‘X-ray crystallography and diffraction’

Physics ‘2D materials and van der Waals heterostructures’, ‘Ab initio and many-body methods’, ‘Accelerator physics and beam dynamics’, ‘Acoustics and aeroacoustics’, ‘Amorphous solids and glasses’, ‘Astrophysical and space plasmas’, ‘Astrophysics and cosmology’, ‘Atomic, molecular, and optical physics’, ‘Battery physics and solid-state electrolytes’, ‘Biophysics and physical biology’, ‘Chemical physics and physical chemistry’, ‘Classical mechanics and analytical dynamics’, ‘Cold atoms, BoseEinstein condensates, and quantum gases’, ‘Colloids, emulsions, and granular matter’, ‘Computational physics and scientific computing’, ‘Condensed matter physics (theory and experiment)’, ‘Continuum mechanics and elasticity’, ‘Cosmology (inflation, CMB, dark matter, dark energy)’, ‘Data-driven physics and physics-informed ML’, ‘Digital twins for plasmas and fusion’, ‘Econophysics and sociophysics’, ‘Electromagnetism and classical electrodynamics’, ‘Electronic structure theory and density functional theory’, ‘Energy materials and photovoltaics’, ‘Equilibrium and nonequilibrium statistical physics’, ‘Floquet engineering and time crystals’, ‘Fluid dynamics and turbulence’, ‘Galactic dynamics and black holes’, ‘Gauge theories and the Standard Model’, ‘Geophysics and seismology’, ‘Gravitation and gravitational waves’, ‘Gravitational lensing and weak lensing’, ‘Hadron physics and effective field theories’, ‘Heavy-ion collisions and quark-gluon plasma’, ‘High-energy astrophysics and cosmic rays’, ‘High-intensity lasers and free-electron lasers’, ‘Higher-order topological phases’, ‘Hybrid quantum devices and atomtronics’, ‘Information thermodynamics and Maxwell’s demon’, ‘Instrumentation and detector physics’, ‘Inverse problems in physics’, ‘Large-scale structure and galaxy formation’, ‘Lattice gauge theory’, ‘Machine learning for materials discovery’, ‘Magnetism and spintronics’, ‘Majorana modes and non-Abelian anyons’, ‘Materials physics and nanoscience’, ‘Medical physics and imaging’, ‘Metamaterials and metasurfaces’, ‘Microfluidics and lab-on-chip physics’, ‘Moir´e materials and twisted bilayers’, ‘Monte Carlo and molecular dynamics’, ‘Neutrino physics and oscillations’, ‘Non-Hermitian and PT-symmetric physics’, ‘Nonequilibrium physics and fluctuation theorems’, ‘Nonlinear and ultrafast optics (attosecond physics)’, ‘Nonlinear dynamics and chaos’, ‘Nuclear physics and nuclear structure’, ‘Numerical relativity’, ‘Open quantum systems and decoherence’, ‘Particle physics (theory and experiment)’, ‘Photonics and nanophotonics’, ‘Physical oceanography and climate physics’, ‘Physics education research and outreach’, ‘Planetary science and heliophysics’, ‘Plasma physics and magnetohydrodynamics’, ‘Polaritonic chemistry and cavity QED materials’, ‘Polymer physics and biophysics of macromolecules’, ‘Quantum chromodynamics and QCD phenomenology’, ‘Quantum computing and quantum algorithms’, ‘Quantum control and quantum thermodynamics’, ‘Quantum error correction and fault tolerance’, ‘Quantum field theory’, ‘Quantum information science’, ‘Quantum mechanics and quantum foundations’, ‘Quantum networks and quantum internet’, ‘Quantum optics and quantum metrology’, ‘Quantum sensing and precision metrology’, ‘Quantum simulators with atoms, ions, and photons’, ‘Quantum transport and mesoscopic physics’, ‘Quantum-enhanced imaging and spectroscopy’, ‘Relativity (special and general)’, ‘Rydberg atoms and quantum simulation’, ‘Soft matter physics and complex fluids’, ‘Spectroscopy across the EM spectrum’, ‘Stellar evolution and nucleosynthesis’, ‘Strongly correlated electrons’, ‘Superconductivity and superfluidity’, ‘Surface physics and scanning probe microscopy’, ‘Synchrotron and neutron scattering’, ‘Thermodynamics and statistical mechanics’, ‘Thermoelectrics and heat transport’, ‘Topological insulators and semimetals’, ‘Topological phases of matter’, ‘Topological photonics and phononics’, ‘Topological quantum matter out of equilibrium’

Literature ‘Adaptation studies and intermediality’, ‘African, Caribbean, and diasporic literatures’, ‘American and European literatures’, ‘Ancient Near Eastern literatures’, ‘Archive studies and critical bibliography’, ‘Book history and print culture’, ‘Children’s and young adult literature’, ‘Classical literatures (Greek, Latin)’, ‘Cognitive literary studies and neuro-literary approaches’, ‘Contemporary and global anglophone literature’, ‘Corpus approaches to literature’, ‘Critical race and ethnic studies’, ‘Cultural studies and semiotics’, ‘Detective and crime fiction’, ‘Digital humanities and computational literary studies’, ‘Disability studies and medical humanities’, ‘East Asian literatures (Chinese, Japanese, Korean)’, ‘Ecocriticism and environmental humanities’, ‘Enlightenment literature’, ‘Evolutionary literary studies’, ‘Feminist, gender, and queer theory’, ‘Film, theatre, and performance studies’, ‘Game narratives and interactive fiction’, ‘Genre studies (poetry, drama, novel, short story)’, ‘Graphic narratives and comics studies’, ‘Hermeneutics and interpretation’, ‘Holocaust and genocide literatures’, ‘Horror and Gothic literatures’, ‘Indigenous and First Nations literatures’, ‘Latin American and Iberian literatures’, ‘Law and literature’, ‘Life writing, autobiography, and memoir’, ‘Literary theory and criticism’, ‘Manuscript studies, codicology, and paleography’, ‘Medieval literature’, ‘Middle Eastern and Islamic literatures’, ‘Modernist and postmodernist literature’, ‘Narratology and discourse analysis’, ‘New formalism and historical poetics’, ‘New historicism and cultural materialism’, ‘Open scholarship and digital editions’, ‘Orality, folklore, and myth’, ‘Philology and historical linguistics interface’, ‘Platform studies and e-literature’, ‘Poetics and rhetoric’, ‘Poetry and prosody’, ‘Postcolonial studies and decolonial theory’, ‘Reader-response and affect theory’, ‘Renaissance and early modern literature’, ‘Romanticism and Victorian literature’, ‘Science and literature’, ‘Science fiction, fantasy, and speculative fiction’, ‘Shakespeare and early modern drama’, ‘Social media narratives and fan fiction’, ‘South Asian and Southeast Asian literatures’, ‘Structuralism, poststructuralism, and deconstruction’, ‘Stylometry and authorship attribution’, ‘Text mining and distant reading’, ‘Textual criticism, stemmatics, and editorial theory’, ‘Translation studies and comparative poetics’, ‘Trauma and memory studies’, ‘Travel writing and nature writing’, ‘World literature and comparative literature’, ‘World-systems and global literary circulation’

History ‘African history (regional and thematic)’, ‘Ancient Near Eastern history’, ‘Archival studies and paleography’, ‘Big history and deep history’, ‘Business history and financial history’, ‘Classical Greek and Roman history’, ‘Cliometrics and quantitative history’, ‘Contemporary history’, ‘Cultural heritage, archives, and public history’, ‘Cultural history and history of the senses’, ‘Demographic history and family history’, ‘Digital history and computational humanities’, ‘Diplomatic history and international relations’, ‘Early modern history’, ‘East Asian history (China, Japan, Korea)’, ‘Economic history and the history of capitalism’, ‘Environmental and climate history’, ‘Epigraphy and numismatics’, ‘European history (Western, Central, Eastern)’, ‘Food history and culinary cultures’, ‘Gender history and women’s history’, ‘Genocide studies and Holocaust history’, ‘Geographic information systems in history’, ‘Historiography and theory of history’, ‘History of medicine and public health’, ‘History of science and technology’, ‘Indigenous and First Nations histories’, ‘Intellectual history and history of ideas’, ‘Labor history and working-class studies’, ‘Late antiquity and early medieval history’, ‘Latin American and Caribbean history’, ‘Legal and constitutional history’, ‘Material culture and consumption’, ‘Medieval history’, ‘Memory studies and memorialization’, ‘Methods in historical research and source criticism’, ‘Microhistory and local history’, ‘Middle Eastern and Islamic history’, ‘Military history and war studies’, ‘Modern history’, ‘North American history’, ‘Oral history and life histories’, ‘Political history and state formation’, ‘Prehistory and prehistoric archaeology’, ‘Queer histories and sexuality’, ‘Religious history and church history’, ‘Remote sensing and landscape archaeology’, ‘Russian and Soviet history’, ‘Social history and everyday life’, ‘South Asian and Southeast Asian history’, ‘Transnational and diaspora studies’, ‘Urban and rural history’, ‘World history and global history’

Biology ‘Aging, senescence, and geroscience’, ‘Agricultural biology and crop science’, ‘Astrobiology and origins of life’, ‘Behavioral ecology and ethology’, ‘Bioethics and policy in biotechnology’, ‘Biomaterials and tissue engineering’, ‘Biomechanics and mechanobiology’, ‘Biophysics and single-molecule biophysics’, ‘CRISPR technologies and gene editing’, ‘Cancer biology and tumor microenvironment’, ‘Cell biology and organelle dynamics’, ‘Circadian biology and chronobiology’, ‘Cognitive neuroscience and connectomics’, ‘Computational neuroscience and brain-inspired models’, ‘Conservation biology and biodiversity’, ‘Developmental biology and morphogenesis’, ‘Ecology (population, community, ecosystem)’, ‘Endocrinology and metabolic disease’, ‘Epidemiology and disease ecology’, ‘Epigenetics and chromatin biology’, ‘Evolutionary biology and evo-devo’, ‘Gene drives and population engineering’, ‘Genetics and genomics’, ‘High-throughput screening and functional genomics’, ‘Immunology and immuno-oncology’, ‘Interactomics and network biology’, ‘Marine and freshwater biology’, ‘Metabolomics and systems metabolism’, ‘Microbiology (bacteriology, virology, mycology, parasitology)’, ‘Microbiome and host–microbe interactions’, ‘Molecular biology and gene regulation’, ‘Neurobiology and systems neuroscience’, ‘Paleobiology and evolutionary paleontology’, ‘Pharmacology and toxicology’, ‘Photosynthesis and plant metabolism’, ‘Phylogenetics and comparative genomics’, ‘Plant biology and plant immunity’, ‘Population genetics and quantitative genetics’, ‘Proteomics and protein science’, ‘Quantitative and single-cell biology’, ‘Spatial omics and imaging genomics’, ‘Stem cell biology and regenerative medicine’, ‘Structural biology (X-ray, NMR, cryo-EM)’, ‘Synthetic biology and genome engineering’, ‘Systems biology and dynamical modeling’, ‘Systems immunology and immunometabolism’, ‘Transcriptomics and RNA biology’, ‘Virology and viral evolution’

Linguistics ‘Anthropological linguistics and ethnography of speaking’, ‘Bilingualism and multilingualism’, ‘Cognitive linguistics and conceptual metaphor’, ‘Comparative and contrastive linguistics’, ‘Computational linguistics and NLP’, ‘Corpus linguistics and big data’, ‘Dialectology and microvariation’, ‘Discourse analysis and conversation analysis’, ‘Distributional semantics and embeddings’, ‘Educational linguistics and literacy’, ‘Endangered languages and revitalization’, ‘Experimental methods (eye-tracking, EEG/MEG, fMRI)’, ‘Field linguistics and language documentation’, ‘Forensic linguistics and language policy’, ‘Gesture and multimodal communication’, ‘Grammaticalization and contact-induced change’, ‘Historical linguistics and language change’, ‘Information structure and focus’, ‘Language acquisition (first and second)’, ‘Language typology and universals’, ‘Lexicography and lexicology’, ‘Morphology (inflectional and derivational)’, ‘Neurolinguistics and language in the brain’, ‘Parsing, tagging, and morphological analysis’, ‘Philosophy of language and formal pragmatics’, ‘Phonetics (articulatory, acoustic, auditory)’, ‘Phonology (theory and phonology-phonetics interface)’, ‘Pragmatics (speech acts, implicature, relevance)’, ‘Pragmatics in conversational AI’, ‘Prosody, stress, and intonation’, ‘Psycholinguistics and sentence processing’, ‘Semantic change and diachronic semantics’, ‘Semantics (formal, lexical, and dynamic)’, ‘Sign language linguistics’, ‘Sociolinguistics and language variation’, ‘Sociophonetics and indexical meaning’, ‘Speech recognition and text-to-speech’, ‘Syntax (generative, dependency, construction grammar)’, ‘Word formation and lexical morphology’, ‘Writing systems, orthography, and graphemics’

### C. Examples of CHIMERA

Table 8. A data sample from CHIMERA (Subject: Math).

Example from CHIMERA Subject: Math Topic: Sieve methods and distribution in arithmetic progressions Problem: Let q ≥ 1 be fixed, and let a be an integer coprime to q. For x ≥ ee, fix u ∈ [2, 3], and set

x1/u (log x)A

, A ≥ 5.

z =

Let P denote the set of primes not dividing q. Define the sifted count in the arithmetic progression

S(x; q, a; z) = # n ≤ x : n ≡ a (mod q), and every prime divisor p | n satisfies p ≥ z .

Prove that, as x → ∞, uniformly in a with (a, q) = 1 (and for fixed q), the following two-sided sieve bounds hold:

x ϕ(q)

V (z)f(u) + O

V (z) log z − O(zu) ≤ S(x; q, a; z) ≤

x ϕ(q)

x ϕ(q)

V (z)F(u) + O

V (z) log z

x ϕ(q)

+ O(zu),

where

•

V (z) =

p<z p∤q

1 p

1 −

,

- • for u ∈ [2, 3],

f(u) =

2eγ log(u − 1) u

, F(u) =

2eγ 1 + log(u − 1) u

,

- • and γ is the Euler–Mascheroni constant.

Moreover, deduce the asymptotic form

 

  Φ(u) + O

−1

e−γ log z

1 log z

x ϕ(q)

1 p

+ O(zu),

1 −

S(x; q, a; z) =

p|q

with Φ(u) satisfying

f(u) ≤ Φ(u) ≤ F(u), and in particular

 

  e−γΦ(u) + o(1) + O(zu).

−1

x ϕ(q) log z

1 p

1 −

S(x; q, a; z) =

p|q

Answer: For fixed q ≥ 1, with (a, q) = 1, u ∈ [2, 3], and

x1/u (log x)A

, A ≥ 5,

z =

as x → ∞ one has

x ϕ(q)

V (z)f(u) + O

V (z) log z − O(zu) ≤ S(x; q, a; z) ≤

x ϕ(q)

x ϕ(q)

V (z)F(u) + O

V (z) log z

x ϕ(q)

+ O(zu).

Continued on next page

Here

1 p

1 −

V (z) =

,

p<z p∤q

2eγ log(u − 1) u

,

f(u) =

2eγ 1 + log(u − 1) u

. Equivalently,

F(u) =

  e−γΦ(u) + O

 

−1

x ϕ(q) log z

1 p

1 −

S(x; q, a; z) =

p|q

where

###### f(u) ≤ Φ(u) ≤ F(u).

1 log z

+ O(zu),

Detailed Solution: Okay, let’s tackle this problem step by step. First, part 1 asks to show the empirical measure converges weakly to a Gaussian law, find m(t) and v(t), and discuss the invariant measure. Starting with the McKean–Vlasov SDE:

dXt = −aXt dt − b E[Xt] dt + σ dWt. Let’s denote m(t) = E[Xt], so the SDE becomes

dXt = −aXt dt − bm(t) dt + σ dWt. Taking expectation on both sides:

dm(t) dt

= −am(t) − bm(t) = −λm(t),

where λ = a + b > 0. That’s a simple ODE:

###### m(t) = m0e−λt.

Good, that’s straightforward. Now for the variance v(t) = Var(Xt) = E (Xt − m(t))2 . Let’s set Yt = Xt − m(t), so

dYt = dXt − dm(t). Then

dYt = −aXt − bm(t) dt + σ dWt + λm(t) dt,

since ddmt = −λm. Substitute Xt = Yt + m(t):

dYt = −a(Yt + m(t)) − bm(t) + λm(t) dt + σ dWt. But λ = a + b, so

−am(t) − bm(t) + λm(t) = 0. Thus

dYt = −aYt dt + σ dWt. That’s an Ornstein–Uhlenbeck process!

... To analyze the sifted count

###### S(x; q, a; z) = # n ≤ x : n ≡ a (mod q), and every prime divisor p | n satisfies p ≥ z ,

Continued on next page

we apply sieve theory, specifically the linear sieve (Rosser–Iwaniec sieve), which is well-suited for counting integers with all prime factors at least z, especially when z ≍ x1/u with fixed u ∈ [2, 3].

- 1. Setup and key observations Let P be the set of primes not dividing q. Since (a, q) = 1, any n ≡ a (mod q) satisfies (n, q) = 1, hence all prime divisors of such n are coprime to q. Therefore, the condition that every prime divisor of n is ≥ z is equivalent to saying that n is coprime to all primes p < z with p ∤ q. Define

Pz = { p < z : p ∤ q }, V (z) =

p<z p∤q

1 −

1 p

,

and the set

A = { n ≤ x : n ≡ a (mod q) }.

We aim to estimate the number of elements of A that avoid all primes in Pz, namely

S(x; q, a; z) = #{ n ∈ A : (n, P(z)) = 1 }, P(z) :=

p∈Pz

p.

- 2. Sieve density and linear sieve input For each prime p, define

g(p) =

 



1 p

, p ∤ q, 0, p | q.

This is a dimension-1 sieve (since g(p) ∼ 1/p). Extend g multiplicatively to squarefree integers d by

g(d) =

p|d

g(p).

Then

p<z

(1 − g(p)) =

p<z p∤q

1 −

1 p

= V (z),

matching the definition in the problem. Let u ∈ [2, 3], take

z =

x1/u (log x)A

, A ≥ 5, and D = zu. The Rosser–Iwaniec linear sieve yields two-sided bounds of the form

S(x; q, a; z) ≥

x ϕ(q)

V (z) f(u) + O

x ϕ(q)

V (z) log z − O(zu),

and

S(x; q, a; z) ≤

x ϕ(q)

V (z) F(u) + O

x ϕ(q)

V (z) log z

+ O(zu),

where, for u ∈ [2, 3],

f(u) =

2eγ log(u − 1) u

, F(u) =

2eγ 1 + log(u − 1) u

, and γ is the Euler–Mascheroni constant.

- 3. Two-sided bounds Combining the linear sieve bounds with the above choices of parameters gives exactly the stated inequalities:

x ϕ(q)

V (z)f(u) + O

V (z) log z − O(zu) ≤ S(x; q, a; z) ≤

x ϕ(q)

x ϕ(q)

V (z)F(u) + O

V (z) log z

x ϕ(q)

+ O(zu).

Continued on next page

- 4. Asymptotic form By Mertens’ theorem,

e−γ log z

1 p

1 log z

1 −

=

1 + O

.

p<z

Hence

 

  =

 

  1 + O

−1

−1

e−γ log z

1 p

1 p

1 p

1 p

1 −

1 −

1 −

1 −

V (z) =

=

p<z p∤q

p<z

p|q

p|q

Writing the main term as ϕ(xq)V (z)Φ(u) with f(u) ≤ Φ(u) ≤ F(u), we obtain

  Φ(u) + O

 

−1

e−γ log z

1 log z

x ϕ(q)

1 p

+ O(zu).

1 −

S(x; q, a; z) =

p|q

In particular,

  e−γΦ(u) + o(1) + O(zu).

 

−1

x ϕ(q) log z

1 p

1 −

S(x; q, a; z) =

p|q

Conclusion We have shown that

1 log z

.

x ϕ(q)

S(x; q, a; z) =

V (z)Φ(u) + O

V (z) log z

x ϕ(q)

+ O(zu), f(u) ≤ Φ(u) ≤ F(u),

and equivalently

 

  Φ(u) + O

−1

e−γ log z

1 log z

x ϕ(q)

1 p

+ O(zu).

1 −

S(x; q, a; z) =

p|q

###### Correctness: True

Table 9. A data sample from CHIMERA (Subject: Computer Science).

Example from CHIMERA Subject: Computer Science Topic: Algorithms and randomized algorithms Problem: Consider the following idealized version of the Karger–Stein randomized contraction algorithm applied to an arbitrary simple undirected graph on n ≥ 2 vertices that has a unique minimum cut. Let t(n) = √n2. Define the per-run cost T(n) and per-run success probability p(n) by the recurrences

- 1. T(2) = 0 and T(n) = 2T(t(n)) + cn2 for all real n ≥ 2,

where c > 0 is a fixed constant. Assume independence between the two recursive calls and across all runs.

- 2. p(2) = 1 and p(n) = 1 − 1 − s(n)p(t(n)) 2 for all real n ≥ 2,

where

t(n) t(n) − 1 n(n − 1)

s(n) =

is the exact probability that the unique minimum cut survives the contraction from n to t(n) vertices. Let r(n, δ) be the minimum number of independent repetitions needed so that the overall failure probability is at most δ ∈ (0, 1/2), i.e.,

r(n, δ) = min {r ∈ N : (1 − p(n))r ≤ δ} . Define the minimal expected total work

W(n, δ) = r(n, δ) T(n). Determine the following, with exact leading constants:

- (a) An exact closed form for T(n).
- (b) The limit

L = lim

n→∞

p(n) ln n.

- (c) The asymptotic behavior of r(n, δ) as n → ∞ with δ fixed, including the exact leading constant: r(n, δ) ∼ K ln n ln(1/δ),

and determine K.

- (d) The asymptotic behavior of W(n, δ) as n → ∞ with δ fixed, including the exact leading constant C(c) such that W(n, δ) ∼ C(c) n2(ln n)2 ln(1/δ).

###### Answer:

- (a)

T(n) =

2c ln 2

n2 ln n − 2c n2.

- (b)

L = lim

n→∞

p(n) ln n = 2 ln 2.

- (c)

ln n 2 ln 2

r(n, δ) ∼

ln(1/δ), so

- 1

- 2 ln 2

K =

.

Continued on next page

- (d)

c (ln 2)2

n2(ln n)2 ln(1/δ), so

W(n, δ) ∼

c (ln 2)2

C(c) =

.

Detailed Solution: Okay, let’s tackle part (a) first. We need to find a closed form for T(n) given the recurrence T(2) = 0 and

n √2

T(n) = 2T(t(n)) + cn2, t(n) =

.

First, let’s consider how many times we can apply the recurrence before we get down to n = 2. Let’s define a sequence nk where n0 = n, nk+1 = t(nk) = nk/√2. So

n (√2)k

n 2k/2

nk =

=

.

We want to find the smallest k such that nk ≤ 2. Let’s solve for k when nk = 2:

k 2

n 2k/2 ⇒ 2k/2 =

n 2 ⇒

= log2(n/2) ⇒ k = 2 log2(n/2) = 2 log2 n − 2.

2 =

But maybe it’s better to use exponents with base √2. Let’s let m be the number of recursive steps until we reach 2. So

√

n 2 ⇒ m = log√2(n/2) = 2 log2(n/2) = 2 log2 n − 2,

2)m =

(

same as before. But since the recurrence is T(n) = 2T(n/√2) + cn2, this is a divide-and-conquer recurrence. Let’s try to unroll it. Let’s compute T(n):

√

2) + cn2

T(n) = 2T(n/

√

2)2 + cn2 = 4T(n/2) + cn2 + cn2 = 4T(n/2) + 2cn2. Wait, hold on:

= 2 2T(n/2) + c(n/

√

n2 2

2)2 =

(n/

,

...

- (a) Exact Closed Form for T(n) We are given the recurrence:

n √2

+ cn2 for n ≥ 2.

T(2) = 0, T(n) = 2T

This is a divide-and-conquer recurrence where the problem size reduces by a factor of √2 at each step. Let us unroll the recurrence:

- At level 0:

T(n) = 2T(n/

√

2) + cn2

- At level 1:

√

√

2)2 Substituting back:

T(n/

2) = 2T(n/2) + c(n/

√

2)2] + cn2 = 4T(n/2) + 2cn2.

T(n) = 2[2T(n/2) + c(n/

Continued on next page

Continuing this process, we observe that each recursive call contributes cn2 to the total cost, and the number of such levels is determined by how many times we can divide n by √2 until it reaches 2. Let

m = log√2(n/2). Then,

√

2)m = 2 ⇒ m = log√2(n/2) = 2 log2(n/2). At each level i, the contribution is cn2, and there are m such levels. Thus,

n/(

T(n) = cn2 · m = cn2 · 2 log2(n/2). Hence, the exact closed form is:

|T(n) = 2cn2 log2<br><br>n 2<br><br>|
|---|

- (b) Limit L = limn→∞ p(n) ln n Given:

2

n √2

p(2) = 1, p(n) = 1 − 1 − s(n)p

, where

t(n)(t(n) − 1) n(n − 1)

n √2

s(n) =

, t(n) =

.

For large n, s(n) ≈ 12, so the recurrence becomes approximately:

p(n) ≈ 2s(n)p

n √2 ≈ p

n √2

.

This suggests that p(n) decreases slowly as n increases. To analyze the asymptotic behavior, define

- m = log√2 n,

so

- n = (

√

2)m, and consider the recurrence in terms of m:

√

q(m) = p((

2)m), q(m) ≈ 1 − 1 −

- 1

- 2

q(m − 1)

2

###### .

Expanding for small q(m − 1), we get:

1 4

q(m − 1)2. This leads to the differential equation approximation:

q(m) ≈ q(m − 1) −

dq dm ≈ −

1 4

4 m

q2 ⇒ q(m) ∼

.

Since

m = log√2 n = 2 log2 n, we have:

4 2 log2 n

2 log2 n

2 ln 2 ln n

p(n) ∼

. Thus,

=

=

p(n) ln n ∼ 2 ln 2. Therefore, the limit is:

|L = 2 ln 2|
|---|

Continued on next page

###### (c) Asymptotic Behavior of r(n, δ) We are given:

r(n, δ) = min{r ∈ N : (1 − p(n))r ≤ δ}. Using the approximation (1 − p(n))r ≈ e−rp(n), we require:

ln(1/δ) p(n)

e−rp(n) ≤ δ ⇒ r ≥

.

From part (b),

2 ln 2 ln n

p(n) ∼

, so:

ln(1/δ) p(n) ∼

ln(1/δ) ln n 2 ln 2

r(n, δ) ∼

.

Thus, the asymptotic behavior is:

and the constant K is:

|r(n, δ) ∼<br><br>1<br><br>2 ln 2<br><br><br>ln n ln<br><br>1 δ<br><br>|
|---|

|K =<br><br>1<br><br>2 ln 2<br>|
|---|

###### (d) Asymptotic Behavior of W(n, δ) = r(n, δ)T(n) From parts (a) and (c), we have:

2cn2 ln n ln 2

T(n) ∼ 2cn2 log2 n =

,

ln(1/δ) ln n 2 ln 2

r(n, δ) ∼

. Multiplying these:

cn2(ln n)2 ln(1/δ) (ln 2)2

2cn2 ln n ln 2

ln(1/δ) ln n 2 ln 2

W(n, δ) ∼

=

.

Thus, the asymptotic behavior is:

|W(n, δ) ∼<br><br>c (ln 2)2<br><br>n2(ln n)2 ln<br><br>1 δ<br><br>|
|---|

and the constant C(c) is:

|C(c) =<br><br>c (ln 2)2<br><br>|
|---|

###### Correctness: True

Table 10. A data sample from CHIMERA (Subject: Chemistry).

Example from CHIMERA Subject: Chemistry Topic: Green solvents and sustainable synthesis Problem: Consider a unimolecular, homogeneous reaction

###### A → P

at 298 K performed in a binary, ideal liquid mixture of two green solvents:

S1 = gamma-valerolactone (GVL), S2 = 2-methyltetrahydrofuran (2-MeTHF).

Let x ∈ [0, 1] denote the mole fraction of S1 (GVL) in the solvent mixture. Assume the following models and parameters hold exactly:

Kinetics. The first-order rate constant depends on solvent polarity and viscosity via

k(x) = k0 · exp a(ET(x) − ETref) · [η(x)]−β, with

k0 = 1.0 × 10−4 s−1, a = 12, ETref = 0.30, β = 0.5. The Kamlet–Taft polarity parameter mixes linearly:

ET(x) = xET1 + (1 − x)ET2, with

ET1 = 0.60, ET2 = 0.20. The viscosity follows log-linear mixing:

ln η(x) = x ln η1 + (1 − x) ln η2, with

η1 = 1.8 mPa · s, η2 = 0.6 mPa · s. All logarithms are natural logs, and viscosities in mPa · s are used consistently in the model. Solubility and solvent requirement. The solubility of A in the mixed solvent is given by a Hildebrand-type correlation:

S(x) = S0 · exp − K(δ(x) − δA)2 , with

S0 = 5.0 mol · L−1, K = 0.10 MPa−1, δA = 22 MPa0.5. The mixture solubility parameter obeys

δ(x)2 = xδ12 + (1 − x)δ22, with

δ1 = 25 MPa0.5, δ2 = 17 MPa0.5. The minimum solvent volume needed to dissolve 1 mol of A is

1 S(x)

L.

Vmin(x) =

The mixture density is linear in x:

ρ(x) = xρ1 + (1 − x)ρ2, with

ρ1 = 1.20 kg · L−1, ρ2 = 0.86 kg · L−1.

Continued on next page

Environmental intensity. The cradle-to-gate GWP of the mixed solvent, per unit solvent mass, is linear in x:

gmix(x) = xg1 + (1 − x)g2, with

g1 = 1.2 kg CO2e per kg, g2 = 3.0 kg CO2e per kg.

Process objective and constraint. Per mole of product P (assuming 1:1 stoichiometry A → P and quantitative isolation after reaction), the GWP attributable to solvent usage is

G(x) = msolv(x) · gmix(x), where

ρ(x) S(x)

(kg · mol−1). The process must achieve conversion

msolv(x) = ρ(x) Vmin(x) =

###### X∗ = 0.99 within

Tmax = 3600 s. For first-order kinetics, this requires

ln(1 − X∗) Tmax

k(x) ≥ kreq, kreq = −

.

Task. Determine the unique

###### x∗ ∈ [0, 1]

that minimizes G(x) subject to the kinetic constraint k(x) ≥ kreq. Report:

- (1) A proof of uniqueness of the optimizer in the feasible set.
- (2) An explicit expression for x∗ in terms of the model parameters.
- (3) The numerical values of x∗, S(x∗), ρ(x∗), msolv(x∗), gmix(x∗), and the minimal GWP per mole G(x∗) in

###### kg CO2e · mol−1. Answer:

- • Unique optimal composition: x∗ = 0.82123

(mole fraction of GVL in the solvent mixture).

- • Minimal cradle-to-gate solvent GWP per mole product: G(x∗) ≈ 0.474 kg CO2e · mol−1.
- • Verification values at x∗: S(x∗) ≈ 3.66 mol · L−1, ρ(x∗) ≈ 1.139 kg · L−1,

msolv(x∗) ≈ 0.311 kg · mol−1, gmix(x∗) ≈ 1.522 kg CO2e · kg−1, and

k(x∗) = kreq ensuring 99% conversion in 3600 s.

Continued on next page

Detailed Solution: Okay, let’s tackle this problem step by step. First, I need to understand all the components involved. The goal is to find the mole fraction x of GVL (S1) that minimizes the GWP from solvent usage, G(x), while satisfying the kinetic constraint k(x) ≥ kreq.

First, let’s handle the kinetic constraint because that will define the feasible region for x. The reaction is first-order, so to get 99% conversion in 3600 seconds, we need

ln(1 − 0.99) 3600

kreq = −

. Let’s calculate that first. 1 − X∗ = 0.01, so ln(0.01) = −4.60517, so

4.60517

3600 ≈ 0.001279 s−1. Now, let’s write out k(x). The rate constant is

kreq =

k(x) = k0 exp a(ET(x) − ETref) [η(x)]−β. ET(x) is linear:

ET(x) = x · 0.6 + (1 − x) · 0.2 = 0.2 + 0.4x. Viscosity η(x) follows log-linear mixing:

ln η(x) = x ln η1 + (1 − x) ln η2. So

η(x) = η1x η21−x. Let’s compute η1 = 1.8, η2 = 0.6. So

ln η1 = ln(1.8) ≈ 0.5878, ln η2 = ln(0.6) ≈ −0.5108.

...

###### 1) Proof of Uniqueness of the Optimizer in the Feasible Set The feasible region is defined by the kinetic constraint

k(x) ≥ kreq, where

k(x) = k0 · exp[a(ET(x) − ET,ref)] · [η(x)]−β, ET(x) = xET1 + (1 − x)ET2 is linear in x, and

ln η(x) = x ln η1 + (1 − x) ln η2

implies η(x) is log-linear in x. Taking the natural logarithm of k(x), we obtain:

ln k(x) = ln k0 + a(ET(x) − ET,ref) − β ln η(x). Substituting the expressions for ET(x) and ln η(x), this becomes:

ln k(x) = [a(ET1 − ET2) − β ln(η1/η2)]

x + [ln k0 + a(ET2 − ET,ref) − β ln η2]

###### .

mk

bk

Since

mk = 4.2507 > 0, ln k(x) is strictly increasing in x. Therefore, the equation

k(x) = kreq has a unique solution xmin ∈ [0, 1], and the feasible set is [xmin, 1].

Continued on next page

For the objective function

G(x) = msolv(x) · gmix(x), we analyze its derivative

d dx

f′(x) =

ln G(x). The unconstrained minimum of G(x) occurs at

x0 ≈ 0.635, which is less than

xmin ≈ 0.822.

Since G(x) is decreasing on [0, x0] and increasing on [x0, 1], it is strictly increasing on [xmin, 1]. Thus, the minimum of G(x) over the feasible region occurs at x = xmin, and this is the unique minimizer.

- 2) Explicit Expression for x∗ The optimal x∗ satisfies the active constraint

k(x∗) = kreq. Solving for x∗ using the linearized form of ln k(x):

x∗ =

ln kreq − ln k0 − a(ET2 − ET,ref) + β ln η2 a(ET1 − ET2) − β ln(η1/η2)

.

Substituting known parameters:

kreq = −

ln(1 − X∗) Tmax

=

ln(100) 3600

,

a = 12, ET1 = 0.60, ET2 = 0.20, ET,ref = 0.30,

β = 0.5, η1 = 1.8, η2 = 0.6. This yields:

x∗ =

ln ln(100)3600 − ln(10−4) − 12(0.2 − 0.3) + 0.5 ln(0.6) 12(0.6 − 0.2) − 0.5 ln(3)

.

- 3) Numerical Values

###### Step 1: Compute x∗

ln(100) 3600 ≈ 0.0012792 s−1,

kreq =

ln kreq ≈ −6.6615, ln k0 = ln(10−4) = −9.2103. Numerator:

−6.6615 + 9.2103 + 1.2 − 0.2554 = 3.4934. Denominator:

4.8 − 0.5 ln(3) ≈ 4.2507.

x∗ = 3.4934/4.2507 ≈ 0.8218.

###### Step 2: Compute S(x∗) δ(x∗)2 = x∗δ12 + (1 − x∗)δ22 = 0.8218 · 625 + 0.1782 · 289 = 565.14.

√

δ(x∗) =

565.14 ≈ 23.7727.

(δ(x∗) − δA)2 = (23.7727 − 22)2 = 3.1424.

34 Continued on next page

S(x∗) = 5.0 · exp(−0.1 · 3.1424) = 5.0 · 0.7300 = 3.650 mol · L−1.

- Step 3: Compute ρ(x∗) ρ(x∗) = x∗ρ1 + (1 − x∗)ρ2 = 0.8218 · 1.20 + 0.1782 · 0.86 = 1.1394 kg · L−1.
- Step 4: Compute msolv(x∗)

msolv(x∗) =

ρ(x∗) S(x∗)

=

1.1394 3.650

= 0.3122 kg · mol−1.

- Step 5: Compute gmix(x∗) gmix(x∗) = x∗g1 + (1 − x∗)g2 = 0.8218 · 1.2 + 0.1782 · 3.0 = 1.5207 kg CO2e · kg−1.
- Step 6: Compute G(x∗) G(x∗) = msolv(x∗) · gmix(x∗) = 0.3122 · 1.5207 = 0.4748 kg CO2e · mol−1.

###### Final Results

|x∗ = 0.822, S(x∗) = 3.65 mol · L−1, ρ(x∗) = 1.14 kg · L−1, msolv(x∗) = 0.312 kg · mol−1, gmix(x∗) = 1.52 kg CO2e · kg−1, G(x∗) = 0.475 kg CO2e · mol−1.|
|---|

###### Correctness: True

Table 11. A data sample from CHIMERA (Subject: Physics).

Example from CHIMERA Subject: Physics Topic: Rydberg atoms and quantum simulation Problem: Consider an infinite one-dimensional chain of identical neutral atoms separated by a fixed lattice spacing

###### a = 5 µm.

Each atom has a ground state |g⟩ and two Rydberg states |r1⟩ and |r2⟩ with van der Waals interactions

C6,1 r6

C6,2 r6

V1(r) =

, V2(r) =

,

between atoms both in |r1⟩ or both in |r2⟩, respectively. Two far-off-resonant dressing lasers (i = 1, 2) couple |g⟩ to |ri⟩ with Rabi frequencies Ωi and detunings ∆i, and the single-atom rotating-frame Hamiltonian term is

Ωi 2

(|g⟩⟨ri| + h.c.) . Assume weak dressing, i.e.,

∆i|ri⟩⟨ri| +

Ωi ∆i ≪ 1,

and neglect many-body (three- or more-body) terms beyond pairwise additivity in the effective ground-manifold Hamiltonian. Given numerical parameters:

Ω1 2π

=

Ω2 2π

= 3.0 MHz,

∆1 2π

= +40.0 MHz,

∆2 2π

40.0 √2

= −

MHz = −28.2842712474619 MHz,

- C6,1

h

= +1250 GHz µm6,

- C6,2

= −312.5 GHz µm6.

h

- (a) For a single laser i and two atoms separated by r, derive to fourth order in Ωi/∆i the effective two-body interaction Ui(r) that shifts the |gg⟩ energy in the ground manifold, and show that

Ω4i 8∆3i

Vi(r) 2∆i + Vi(r)

Ui(r) =

,

where

C6,i r6

. State all approximations used.

Vi(r) =

Continued on next page

- (b) Show that as r → ∞,

Ui(r) ≈

Ω4iC6,i 16∆4ir6

,

and, using the numbers above, prove that the total far-tail 1/r6 coefficient

U1(r) + U2(r) vanishes exactly.

- (c) Using the exact expression in (a) and the parameters above, compute in Hz units the total effective interaction U(r) = U1(r) + U2(r)

for

r = a and r = 2a.

Provide exact closed-form expressions (in radicals and rational numbers) and numerical values to at least 10−3 Hz accuracy.

- (d) Report the ratio J2

, where

J1

J1 = U(a), J2 = U(2a), as a decimal number with at least four significant digits. All quantities must be computed in frequency units (Hz). Assume pairwise additivity is valid at the stated parameters and that higher-order corrections

Ω6 ∆5

O

are negligible. Answer:

###### (a)

Ω4i 8∆3i

Vi(r) 2∆i + Vi(r)

, where

Ui(r) =

C6,i r6

, valid to fourth order in Ωi/∆i under

Vi(r) =

Ωi ∆i ≪ 1.

###### (b)

- As r → ∞,

Ω4iC6,i 16∆4ir6

Ui(r) ≈

. With

C6,1 4

∆1 √2

Ω1 = Ω2, C6,2 = −

, ∆2 = −

,

the total 1/r6 tail cancels exactly:

αtotal = 0.

Continued on next page

###### (c)

10,125 (4√2 − 9) 896

Hz ≈ −37.778292219 Hz,

U(a) =

10,125 (12,800 − 50√2) 52,427,200

10,125 4,160 −

Hz ≈ −0.024449061 Hz.

U(2a) =

###### (d)

###### J2 J1 ≈ 0.0006471.

Detailed Solution: Okay, let’s tackle part (a) first. The problem is about deriving the effective two-body interaction for two atoms in the ground state when they’re dressed by a single laser. We need to do this to fourth order in Ωi/∆i. First, recall that for a single atom, the rotating-frame Hamiltonian is

H = ∆i|ri⟩⟨ri| +

Ωi 2

(|g⟩⟨ri| + h.c.) .

For two atoms, the Hamiltonian will have single-atom terms and the van der Waals interaction Vi(r) between the Rydberg states. So the two-atom Hamiltonian in the basis

{|g1g2⟩, |g1r2⟩, |r1g2⟩, |r1r2⟩} is





0 Ω2i Ω2i 0 Ωi

2 ∆i 0 Ω2i

.

H =

 

 

Ωi 2 0 ∆i Ω2i 0 Ω2i Ω2i 2∆i + Vi(r)

Wait, actually, the Rydberg–Rydberg interaction for both in |ri⟩ is Vi(r), so the |r1r2⟩ state has energy

###### 2∆i + Vi(r),

because each |ri⟩ has energy ∆i from the detuning, and then the interaction adds Vi(r). The single Rydberg states (|g1r2⟩ and |r1g2⟩) have energy ∆i each, since only one atom is in the Rydberg state. ...(a) Derivation of Effective Two-Body Interaction Ui(r)

To derive the effective two-body interaction Ui(r) for two atoms in the ground state |gg⟩ under weak dressing, we consider the Hamiltonian of the system. The single-atom rotating-frame Hamiltonian is:

Ωi 2

(|g⟩⟨ri| + h.c.) . For two atoms, the total Hamiltonian includes the van der Waals interaction

H = ∆i|ri⟩⟨ri| +

C6,i r6 between Rydberg states |riri⟩. The full Hamiltonian in the basis

Vi(r) =

{|gg⟩, |gr⟩, |rg⟩, |rr⟩} is:

Continued on next page





0 Ω2i Ω2i 0 Ωi

2 ∆i 0 Ω2i

H =

.

 

 

Ωi 2 0 ∆i Ω2i 0 Ω2i Ω2i 2∆i + Vi(r)

Using perturbation theory to fourth order in Ωi/∆i, we compute the energy shift of the ground state |gg⟩. The second-order contribution gives the standard light shift

Ω2i 2∆i

−

,

but the fourth-order correction arises from paths involving the double Rydberg state |rr⟩, which introduces the interaction-dependent term. The effective interaction Ui(r) is the difference between the energy shift with and without the interaction Vi(r). After expanding the perturbation series and simplifying, we obtain:

Ω4i 8∆3i ·

Vi(r) 2∆i + Vi(r)

###### . Approximations used:

Ui(r) =

- • Weak dressing: ∆ Ωi

i

≪ 1.

- • Pairwise additivity: Only two-body interactions are considered.
- • Higher-order terms beyond O(Ω6/∆5) are neglected.
- • The Rydberg–Rydberg interaction Vi(r) is treated as a small perturbation relative to the detuning ∆i.

###### (b) Far-Tail Coefficient Vanishing

- As r → ∞, Vi(r) → 0, so we expand Ui(r) to leading order in Vi(r):

Ω4iC6,i 16∆4ir6

Ω4i 8∆3i ·

Vi(r) 2∆i

Ui(r) ≈

=

.

Thus, the far-tail coefficient of 1/r6 in

U(r) = U1(r) + U2(r) is:

Ω4 16

C6,1 ∆41

C6,2 ∆42

. Given:

+

Ω1 = Ω2 = Ω = 3.0 × 106 Hz,

40.0

∆1 = 40.0 × 106 Hz, ∆2 = −

√2 × 106 Hz, C6,1 h

C6,2 h

= −312.5 GHz · µm6, we compute:

= 1250 GHz · µm6,

1250 × 109 (40 × 106)4

C6,1 ∆41

=

,

= −312.5 × 109 · 4 404 × 1024

1250 × 109 404 × 1024

= −312.5 × 109 (40/√2)4 × 1024

C6,2 ∆42

= −

.

Continued on next page

Therefore:

C6,1 ∆41

C6,2 ∆42

= 0. Hence, the total far-tail coefficient vanishes exactly.

+

- (c) Total Effective Interaction at r = a and r = 2a Given parameters (in Hz):

- • Ω = 3.0 × 106 Hz
- • ∆1 = 40.0 × 106 Hz
- • ∆2 = −40√.20 × 106 Hz

- • Ch6,1 = 1250 × 109 Hz · µm6

- • Ch6,2 = −312.5 × 109 Hz · µm6

- • a = 5 µm

- At r = a = 5 µm: r6 = 56 = 15625 µm6,

V1(a) =

1250 × 109 15625

= 80 × 106 Hz,

V2(a) = −312.5 × 109 15625

= −20 × 106 Hz.

- U(a) = 78984.6826443 Hz.

At r = 2a = 10 µm:

r6 = 106 = 1,000,000 µm6,

V1(2a) = 1.25 × 106 Hz,

- V2(2a) = −0.3125 × 106 Hz.

U(2a) = 2431.43613937 Hz.

- (d) Ratio Let

J1 = U(a), J2 = U(2a).

Rounded to four significant digits:

2431.43613937 78984.6826443 ≈ 0.0307836.

J2 J1

=

|0.03078|
|---|

###### . Correctness: True

Table 12. A data sample from CHIMERA (Subject: Literature).

Example from CHIMERA Subject: Literature Topic: Archive studies and critical bibliography Problem:

Consider an authority-control deduplication problem in archive studies and critical bibliography. Let R be a finite set of bibliographic records, each record r ∈ R having fields author (a free-text personal-name string in either “Family, Given ...” form), title (a free-text string), and year (an integer). Define the following normalization and similarity functions:

###### 1) Unicode and token normalization For any string x, let normalize(x) be the result of:

- (a) Unicode NFKD decomposition and removal of all combining diacritical marks.
- (b) Lowercasing.
- (c) Replacing all non-alphanumeric characters (including punctuation and hyphens) by a single ASCII space.
- (d) Collapsing multiple spaces to one space, then trimming leading/trailing spaces.

###### 2) Author key AKey(r) Given r.author in personal-name form “Family, Given ...” (assume every record here has exactly one comma, indicating this form), let:

- (a) Let L = normalize(left-of-comma(r.author)), then remove all spaces from L to get L∗.
- (b) Let G = normalize(right-of-comma(r.author)), and let g be the first ASCII letter in G (if none exists, define g as the empty string; in the dataset below every record has one).
- (c) Define AKey(r) = L∗ ∥ g,

where ∥ denotes string concatenation. Define

1 if AKey(i) = AKey(j), 0 otherwise.

A(i, j) =

###### 3) Title tokenization and Jaccard Let stopword set

S = {a, an, the, of, and, for, to, into, in, on, with, by, from}.

Tokenize a title t as follows: apply normalize(t), split on spaces to tokens, discard any token in S, and discard any empty token. Let T(r) be the resulting set of unique tokens for record r. Define

J(i, j) = |T(i) ∩ T(j)| |T(i) ∪ T(j)|

.

###### 4) Year similarity Define

Y (i, j) =

1 if i.year = j.year, 0 otherwise.

###### 5) Weighted similarity Let weights be

13 25

19 50

1 10

. For records i, j define

w1 =

, w2 =

, w3 =

s(i, j) = w1A(i, j) + w2J(i, j) + w3Y (i, j).

Continued on next page

###### 6) Mergeability at threshold τ Fix

41 50

τ0 =

. Define an undirected graph Gτ on vertex set R with an edge i—j iff

s(i, j) ≥ τ. Let Pτ be the partition of R into the connected components of Gτ. Dataset R (9 records)

- • A1: author = “Smith, John”; title = “Public archive origins and governance”; year = 1999.
- • A2: author = “Smith, J.”; title = “Public archive origins and governance: an enquiry”; year = 1999.
- • A3: author = “Smith, John”; title = “Governance of the public archive: its origins”; year = 1999.
- • B1: author = “Garc´ıa M´arquez, Ana-Luisa”; title = “Metadata for colonial archives: a critical bibliography”; year = 2015.
- • B2: author = “Garcia Marquez, Ana Luisa”; title = “A critical bibliography of colonial archives metadata”; year = 2015.
- • B3: author = “Garc´ıa-M´arquez, Ana L.”; title = “Colonial archives metadata: toward a critical bibliography”; year = 2015.
- • C1: author = “Lef`evre, Paul”; title = “Archival silences and the politics of records retention”; year = 2008.
- • C2: author = “Lefevre, P.”; title = “Archival silence and politics of records retention”; year = 2008.
- • C3: author = “Lef`evre, Paul”; title = “Politics of records retention and archival silences”; year = 2008.

###### Tasks

- (a) Prove that for any finite R and any τ, the partition Pτ is unique and independent of the order in which pairwise merges are performed, i.e., clustering by repeatedly merging any pair with s ≥ τ until no merges remain yields the same partition as the connected components of Gτ.
- (b) For the dataset above, compute AKey for each record, then compute J(i, j) and s(i, j) for all pairs with A(i, j) = 1, determine all edges in Gτ0, and output the partition Pτ0 explicitly as a set of disjoint record-id blocks.
- (c) Let τ∗ be the supremum of all τ ≥ τ0 such that Pτ′ = Pτ0 for every τ′ ∈ [τ0, τ].

Prove that for any weighted undirected graph, for a fixed component H of Gτ0, the largest τ at which H remains connected equals the bottleneck value of a maximum spanning tree (MaxST) of H, i.e., the maximum over all spanning trees of the minimum edge weight. Then compute τ∗ for the dataset.

Answer: Partition at τ0 = 4150

Pτ0 = {{A1, A2, A3}} {{B1, B2, B3}} {{C1, C2, C3}}.

###### Critical threshold

131 150

τ∗ =

.

Detailed Solution: Okay, let me tackle this problem step by step. It’s about authority control deduplication in bibliographic records, and there are three parts to it. I’ll start with part (a) because it’s a proof about the uniqueness of the partition.

Continued on next page

Part (a): Proving uniqueness of Pτ The question is asking to prove that the partition Pτ, defined as the connected components of the graph Gτ (where edges exist if s(i, j) ≥ τ), is unique and doesn’t depend on the order of merging pairs. This sounds familiar from graph theory and clustering. First, recall that in graph theory, connected components are well-defined. For any undirected graph, the connected components partition the vertex set uniquely. The graph Gτ is defined with edges exactly where similarity meets or exceeds τ. So by definition, the connected components are fixed once the graph is fixed. But the problem mentions “clustering by repeatedly merging any pair with s ≥ τ until no merges remain.”

...

- (a) Proof of Uniqueness of Partition Pτ The partition Pτ is defined as the connected components of the undirected graph Gτ, where an edge exists between records i and j if s(i, j) ≥ τ. In graph theory, the connected components of a graph are uniquely determined by the structure of the graph itself, independent of the order in which edges are processed or merged. Clustering via repeated pairwise merges (as described) corresponds to single-linkage hierarchical clustering, where two clusters are merged if any pair of elements between them has similarity ≥ τ. This method is equivalent to computing the connected components

of Gτ, since two nodes belong to the same cluster if and only if there exists a path between them where each consecutive pair satisfies s(i, j) ≥ τ. Connected components are invariant under edge addition order: regardless of the sequence of merges, the final partition depends solely on the existence of paths in Gτ. Thus, Pτ is unique and independent of merge order.

- (b) Computation for Dataset R

###### Step 1: Compute AKey for Each Record

|Record|Author (Normalized)<br><br>|L∗|g<br><br>|AKey|
|---|---|---|---|---|
|A1<br>A2<br>A3<br><br><br>B1<br>B2<br>B3 C1 C2 C3<br>|Smith, John Smith, J. Smith, John Garc´ıa M´arquez, Ana-Luisa Garcia Marquez, Ana Luisa Garc´ıa-M´arquez, Ana L. Lef`evre, Paul Lefevre, P. Lef`evre, Paul<br><br>|smith smith smith garciamarquez garciamarquez garciamarquez lefevre lefevre lefevre<br><br>|j j j a a a p p p<br><br>|smithj smithj smithj garciamarqueza garciamarqueza garciamarqueza lefevrep lefevrep lefevrep<br><br>|

Thus, A(i, j) = 1 only within groups A, B, and C.

###### Step 2: Compute J(i, j) and s(i, j) for pairs with A(i, j) = 1 Weights:

13 25

, w2 =

w1 =

###### Group A (year = 1999)

19 50

1 10

, w3 =

.

###### Group B (year = 2015)

###### Group C (year = 2008)

|Pair<br><br>|J(i, j)<br><br>|s(i, j)|
|---|---|---|
|A1–A2<br>A1–A3 A2–A3<br>|4<br><br>5<br><br><br>4<br><br>5<br><br>4<br><br>6<br>|0.924 0.924<br><br>131 150<br><br>|

|Pair|J(i, j)<br><br>|s(i, j)|
|---|---|---|
|B1–B2<br>B1–B3 B2–B3<br>|1<br><br>5<br><br>6<br><br><br>5<br><br>6<br><br><br>|1<br><br>281 300 281 300<br><br>|

|Pair|J(i, j)<br><br>|s(i, j)|
|---|---|---|
|C1–C2 C1–C3 C2–C3<br><br>|4 16 4 6<br><br>|131 1501 131 150<br><br>|

Continued on next page

- Step 3: Edges in Gτ0 with τ0 = 5041 All intra-group similarities satisfy

41 50

s(i, j) ≥

= 0.82,

and all inter-group similarities are < 0.48 < τ0. Thus the connected components are:

{A1, A2, A3}, {B1, B2, B3}, {C1, C2, C3}.

|Pτ0 = {A1, A2, A3}, {B1, B2, B3}, {C1, C2, C3} .|
|---|

###### (c) Computation of τ∗

For a connected component H of Gτ0, the largest τ at which H remains connected equals the bottleneck value of a maximum spanning tree (MaxST), i.e.,

w(e). Bottleneck values:

###### max

min

T spanning tree of H

e∈T

Thus,

- Group A:

231 250

= 0.924,

- Group B:

281 300 ≈ 0.9367,

- Group C:

131 150 ≈ 0.8733.

τ∗ = min

231 250

,

281 300

,

131 150

|131 150<br><br>|
|---|

=

.

###### Final Answers

|Pτ0 = {A1, A2, A3}, {B1, B2, B3}, {C1, C2, C3}|
|---|

|τ∗ =<br><br>131 150<br><br>.|
|---|

###### Correctness: True

Table 13. A data sample from CHIMERA (Subject: History).

Example from CHIMERA Subject: History Topic: Business history and financial history Problem: Consider two nineteenth-century bimetallic countries i ∈ {A, B} with legal gold/silver mint ratios Mi > 0, defined as the number of units of domestic silver coin that constitute the legal-tender face value of one unit of domestic gold coin. Let the unit of account be the domestic silver coin in each country, so one unit of silver coin has face value 1. For each country i and metal m ∈ {g, s} (gold or silver), coining (minting) one unit of bullion into coin incurs an ad valorem cost κim ∈ [0, 1) and a deterministic delay Tcoin(m, i) ≥ 0 before coin is delivered; melting one unit of coin into bullion incurs an ad valorem loss µim ∈ [0, 1) and a deterministic delay Tmelt(m, i) ≥ 0 before bullion is delivered. Shipping one unit of bullion of metal m from country i to country j incurs an ad valorem loss τmi→j ∈ [0, 1), with τmi→i = 0. The risk-free interest rate is r ≥ 0 (continuous compounding). The world bullion relative price R > 0 is defined as the number of units of silver bullion obtainable per unit of gold bullion via the bullion market, and is assumed constant over the time scales involved (or locked in via forward contracts), with zero bid–ask spread. Arbitrageurs can freely access bullion markets and mints subject to the costs and delays above, and can repatriate coin value between A and B at par in silver units via bills of exchange (so there is no return shipping cost of coin; only bullion shipping is relevant).

Define, for each ordered pair (i, j) ∈ {A, B} × {A, B}, the total processing delays:

TGS(i, j) = Tmelt(g, i) + Tcoin(s, j), TSG(i, j) = Tmelt(s, i) + Tcoin(g, j), and the effective one-leg shipping retention factor

si→j = max{1 − τgi→j, 1 − τsi→j} ∈ (0, 1],

reflecting that an arbitrageur can choose to ship bullion either before or after exchanging metals, selecting the route with the higher retention.

An arbitrageur can execute gold→silver loops (starting with 1 unit of gold coin in i, ending in silver coin units) or silver→gold loops (starting with 1 unit of silver coin in i, ending in silver coin units), choosing any origin i and destination mint j, incurring the corresponding costs, delays, and shipping. Working in present-value terms at t = 0, derive the necessary and sufficient no-arbitrage interval [LBglobal, UBglobal] for the world bullion ratio R such that no riskless profit can be earned by any such loop. Express LBglobal and UBglobal in closed form as functions of the parameters.

Then, specialize to the single-country case (ignore shipping and set i = j and drop country indices) to simplify the band and provide ln(UB/LB) in the symmetric-parameters case where κg = κs = κ, µg = µs = µ, and Tmelt(g) = Tmelt(s) = Tm, Tcoin(g) = Tcoin(s) = Tc.

Answer: The global no-arbitrage interval for the world bullion price ratio R is

where

e−r[Tmelt(s,i)+Tcoin(g,j)] (1 − µis) si→j (1 − κjg) Mj,

LBglobal = max

i,j∈{A,B}

Mi er[Tmelt(g,i)+Tcoin(s,j)] (1 − µig) si→j (1 − κjs)

UBglobal = min

,

i,j∈{A,B}

si→j = max{1 − τgi→j, 1 − τsi→j}.

###### Special case (single country, no shipping).

M er[Tmelt(g)+Tcoin(s)] (1 − µg) (1 − κs)

LB = M (1 − µs) (1 − κg) e−r[Tmelt(s)+Tcoin(g)], UB =

.

Continued on next page

Symmetric-parameters specialization. If κg = κs = κ, µg = µs = µ, and Tmelt(g) = Tmelt(s) = Tm, Tcoin(g) = Tcoin(s) = Tc, then

M er(Tm+Tc) (1 − µ)(1 − κ)

LB = M(1 − µ)(1 − κ) e−r(Tm+Tc), UB =

, and

UB LB

= 2r(Tm + Tc) − 2 ln(1 − µ) + ln(1 − κ) .

ln

Detailed Solution: Okay, let’s try to tackle this problem step by step. First, I need to understand what the question is asking. We have two countries, A and B, each with their own mint ratios for gold and silver. The goal is to find the no-arbitrage bounds for the world bullion ratio R, which is the silver bullion per gold bullion. The problem mentions “loops” that arbitrageurs can execute: gold→silver loops and silver→gold loops, starting in one country and possibly ending in another. Each loop involves melting coin to get bullion, shipping bullion, coining bullion into coin in another country, etc., with associated costs and delays. We need to consider the present value of these operations to ensure no riskless profit. First, let’s recall that in bimetallic systems, the mint ratio is the legal ratio of silver to gold in coin. If the market bullion ratio (R) deviates from the mint ratio, there might be arbitrage opportunities through melting or coining.

###### ... Global No-Arbitrage Bounds for the World Bullion Ratio R

To determine the necessary and sufficient no-arbitrage interval [LBglobal, UBglobal] for the world bullion ratio R, we analyze all possible arbitrage loops between two bimetallic countries A and B. These loops involve converting gold coins into silver coins (gold→silver) or vice versa (silver→gold), with associated costs, delays, and shipping losses. The key insight is that arbitrageurs will always choose the most profitable route (i.e., the one with the highest retention factor), which is captured by the effective shipping retention factor

si→j = max{1 − τgi→j, 1 − τsi→j}.

Closed-Form Expressions for LBglobal and UBglobal Lower Bound (LBglobal). This arises from silver→gold loops, where an arbitrageur starts with 1 unit of silver coin in country i, converts it to gold bullion, ships it to country j, and coins it into gold coin. The present value of the resulting gold coin must not exceed the initial silver coin’s value to prevent riskless profit.

(1 − µis) · si→j · Mj · (1 − κjg) · e−r·TSG(i,j) .

LBglobal = max

i,j∈{A,B}

- • (1 − µis): Retention after melting silver coin in country i.
- • si→j: Effective shipping retention from i to j.
- • Mj: Legal mint ratio in country j.
- • (1 − κjg): Retention after coining gold bullion in country j.
- • e−r·TSG(i,j): Discounting for total delay TSG(i, j) = Tmelt(s, i) + Tcoin(g, j).

Upper Bound (UBglobal). This arises from gold→silver loops, where an arbitrageur starts with 1 unit of gold coin in country i, melts it, ships the bullion to country j, and coins it into silver coin. The present value of the resulting silver coins must not exceed the initial gold coin’s value.

Mi · er·TGS(i,j) (1 − µig) · si→j · (1 − κjs)

UBglobal = min

.

i,j∈{A,B}

- • Mi: Legal mint ratio in country i.
- • er·TGS(i,j): Compounding for total delay TGS(i, j) = Tmelt(g, i) + Tcoin(s, j).
- • (1 − µig): Retention after melting gold coin in country i.
- • si→j: Effective shipping retention from i to j.
- • (1 − κjs): Retention after coining silver bullion in country j. Continued on next page

Special Case: Single-Country Bimetallic System When ignoring shipping (i.e., i = j, τg = τs = 0, so si→i = 1), the global bounds simplify to:

LB = (1 − µs) · M · (1 − κg) · e−r·TSG, UB =

M · er·TGS (1 − µg) · (1 − κs)

. Where:

TGS = Tmelt(g) + Tcoin(s), TSG = Tmelt(s) + Tcoin(g).

###### Symmetric Parameter Case Assume symmetry:

κg = κs = κ, µg = µs = µ, Tmelt(g) = Tmelt(s) = Tm, Tcoin(g) = Tcoin(s) = Tc. Then:

MerT (1 − µ)(1 − κ)

TGS = TSG = T = Tm + Tc, LB = M(1 − µ)(1 − κ)e−rT, UB =

. Compute the logarithmic width of the band:

ln

UB LB

= ln

MerT/[(1 − µ)(1 − κ)] M(1 − µ)(1 − κ)e−rT = ln

e2rT (1 − µ)(1 − κ) 2

|ln<br><br>UB LB<br><br>= 2 r(Tm + Tc) − ln (1 − µ)(1 − κ)|
|---|

.

###### Final Answer

- • Global No-Arbitrage Interval:

|LBglobal = max<br><br>i,j<br><br>(1 − µis) · si→j · Mj · (1 − κjg) · e−r·TSG(i,j) , UBglobal = min<br><br>i,j<br><br>Mi · er·TGS(i,j) (1 − µig) · si→j · (1 − κjs)<br><br>|
|---|

- • Logarithmic Width in Symmetric Single-Country Case:

|ln<br><br>UB LB<br><br>= 2 r(Tm + Tc) − ln (1 − µ)(1 − κ)|
|---|

###### Correctness: True

Table 14. A data sample from CHIMERA (Subject: Biology).

Example from CHIMERA Subject: Biology Topic: Structural biology (X-ray, NMR, cryo-EM) Problem: In single-particle cryo-EM, let S(s) denote the true complex Fourier coefficient of the 3D signal at spatial frequency shell s. Two independently reconstructed half-maps F1(s) and F2(s) are modeled as

###### F1(s) = S(s) + N1(s), F2(s) = S(s) + N2(s),

where N1(s), N2(s) are zero-mean, circular complex Gaussian noise terms, independent of S(s). Assume stationarity across each shell:

ΦS(s) = E |S(s)|2 , ΦN(s) = E |N1(s)|2 = E |N2(s)|2 , and a known real-valued inter-half noise correlation coefficient

E[N1(s)N2(s)∗] ΦN(s)

ρ(s) =

with − 1 < ρ(s) < 1.

Define the half-map Fourier shell correlation

E[F1(s)F2(s)∗] E[|F1(s)|2] E[|F2(s)|2]

c(s) =

.

Let

F1(s) + F2(s) 2

Favg(s) =

be the simple average (the “full map” in the two-half convention). Consider the linear MMSE estimator

Sˆ(s) = H(s) Favg(s). Derive a closed-form expression for the optimal filter H(s) that minimizes

E |S(s) − H(s)Favg(s)|2

over complex scalar H(s), expressed solely in terms of c(s) and ρ(s). Your result must hold for each shell independently. State all assumptions and show all steps. The final expression for H(s) must be a closed-form function of c(s) and ρ(s) only and must reduce correctly to the uncorrelated-noise special case.

###### Answer:

and for ρ(s) = 0,

2 c(s) − ρ(s) 1 − ρ(s) 1 + c(s)

H(s) =

,

2 c(s) 1 + c(s)

.

H(s) =

Detailed Solution: Okay, let’s try to tackle this problem step by step. First, let’s recall what we need: the linear MMSE estimator Sˆ(s) = H(s)Favg(s) that minimizes E[|S − HFavg|2]. We need to find H(s) in terms of c(s) and ρ(s). First, let’s write down Favg(s). Since Favg = (F1 + F2)/2, and F1 = S + N1, F2 = S + N2, then

N1 + N2 2

Favg = S +

.

Let’s denote the noise in the average as Navg = (N1 + N2)/2, so Favg = S + Navg.

Continued on next page

The MMSE estimator for a scalar case (since we’re dealing with each shell independently, it’s a scalar complex random variable problem) is given by

E[SFavg∗ ] E[|Favg|2]

.

H =

This is because for the mean squared error, the optimal linear estimator (since we’re restricting to linear estimators, and MMSE for Gaussian is linear) is the Wiener filter, which in the scalar case is the cross-correlation over the auto-correlation. So let’s compute E[SFavg∗ ]. We have Favg∗ = S∗ + (N1∗ + N2∗)/2, so:

E[SN1∗] + E[SN2∗] 2

N1∗ + N2∗ 2

E[SFavg∗ ] = E S S∗ +

= E[|S|2] +

.

... To derive the optimal linear MMSE estimator Sˆ(s) = H(s)Favg(s) that minimizes the mean squared error E[|S(s) − H(s)Favg(s)|2], we proceed as follows:

- Step 1: Define Key Quantities Given:

- • F1(s) = S(s) + N1(s)
- • F2(s) = S(s) + N2(s)
- • Favg(s) = F1(s)+2F2(s) = S(s) + N1(s)+2N2(s)

Let:

- • ΦS(s) = E[|S(s)|2]
- • ΦN(s) = E[|N1(s)|2] = E[|N2(s)|2]
- • ρ(s) = E[N1(s)N2(s)

∗]

ΦN(s) (real-valued, −1 < ρ < 1) We are given the Fourier Shell Correlation (FSC):

c(s) =

E[F1(s)F2(s)∗] E[|F1(s)|2]E[|F2(s)|2]

.

From stationarity and independence assumptions:

- • E[F1(s)F2(s)∗] = ΦS + ρΦN
- • E[|F1(s)|2] = E[|F2(s)|2] = ΦS + ΦN

Thus,

c(s) =

ΦS + ρΦN ΦS + ΦN

.

- Step 2: Express MMSE Filter in Terms of Signal-to-Noise Ratio The MMSE filter for a scalar complex random variable is:

E[S(s)Favg(s)∗] E[|Favg(s)|2]

H(s) =

.

Compute numerator and denominator: Numerator:

E[S(s)Favg(s)∗] = E S(s) S(s)∗ + 12(N1(s)∗ + N2(s)∗) = ΦS

(since S is independent of N1, N2, and all noise terms have zero mean) Denominator:

1 4

E[|Favg(s)|2] = E S(s) + 12(N1(s) + N2(s)) 2 = ΦS +

E[|N1 + N2|2].

Continued on next page

Now compute E[|N1 + N2|2]:

E[|N1 + N2|2] = E[|N1|2] + E[|N2|2] + 2ℜ(E[N1N2∗]) = 2ΦN + 2ρΦN = 2ΦN(1 + ρ). So:

ΦN(1 + ρ) 2

1 4 · 2ΦN(1 + ρ) = ΦS +

E[|Favg(s)|2] = ΦS +

.

###### Step 3: Express H(s) in Terms of c(s) and ρ(s)

Let γ = ΦΦS

. Then:

N

c − ρ 1 − c

γ + ρ γ + 1 ⇒ γ =

. Substitute into the expression for H(s):

c(s) =

ΦS ΦS + ΦN(1+2 ρ)

γ γ + 1+2ρ

H(s) =

=

.

Plug in γ = c1−−ρc:

c−ρ 1−c

c − ρ (c − ρ) + (1−c)(1+2 ρ)

H(s) =

=

.

c−ρ 1−c + 1+2ρ

Simplify the denominator:

2(c − ρ) + (1 − c)(1 + ρ) 2

(1 − c)(1 + ρ) 2

(c − ρ) +

. Expand the numerator:

=

2(c − ρ) + (1 − c)(1 + ρ) = 2c − 2ρ + 1 + ρ − c − cρ = c − ρ + 1 − cρ = (1 + c)(1 − ρ). Thus:

2(c − ρ) (1 + c)(1 − ρ)

c − ρ

.

H(s) =

=

(1+c)(1−ρ) 2

Final Expression

|H(s) =<br><br>2(c(s) − ρ(s)) (1 + c(s))(1 − ρ(s))<br><br>|
|---|

Assumptions Used

- • Zero Mean: All signals and noise terms are zero-mean.
- • Independence: S(s) is independent of N1(s), N2(s).
- • Stationarity: ΦS(s) = E[|S(s)|2], ΦN(s) = E[|N1(s)|2] = E[|N2(s)|2].
- • Real-Valued Noise Correlation: ρ(s) is real, so E[N1N2∗] = ρΦN.
- • Scalar Shell Independence: Each spatial frequency shell s is treated independently.
- • Circular Complex Gaussian Noise: Ensures that the MMSE estimator is linear and optimal.

Special Case: Uncorrelated Noise (ρ = 0) When ρ = 0, the formula becomes:

2c(s) 1 + c(s)

.

H(s) =

This matches the standard Wiener filter used in cryo-EM for uncorrelated half-maps, confirming correctness. Continued on next page

Conclusion The closed-form expression for the optimal linear MMSE filter H(s), expressed solely in terms of c(s) and ρ(s), is:

|H(s) =<br><br>2(c(s) − ρ(s)) (1 + c(s))(1 − ρ(s))<br><br>|
|---|

###### .

This result holds for each spatial frequency shell independently and reduces correctly to the known uncorrelated-noise case when ρ = 0.

###### Correctness: True

Table 15. A data sample from CHIMERA (Subject: Linguistics).

Example from CHIMERA Subject: Linguistics Topic: Syntax (generative, dependency, construction grammar) Problem:

Let n ≥ 1 and let [n] = {1, 2, . . . , n} be a fixed linear order of word positions. A dependency tree D on [n] is a directed tree (V = [n], E) with a designated root r ∈ [n], where each node other than r has exactly one incoming arc, and arcs are oriented from heads to dependents. For i ∈ [n], let subD(i) denote the set of nodes reachable from i in D (including i). Define the span of i, spD(i), as the set

{j ∈ [n] : j ∈ subD(i)}. The dependency tree D is projective if, for every arc (h, d) ∈ E, every index k strictly between h and d (i.e., min(h, d) < k < max(h, d)) belongs to spD(h).

An ordered binary head-annotated phrase-structure tree (PS tree) P on [n] is a full binary ordered rooted tree whose leaves, read left-to-right, are exactly 1, 2, . . . , n in that order, and where each node N (including leaves) has a head index hd(N) ∈ [n] such that:

- • For a leaf labeled i, hd(leaf i) = i.
- • For an internal node N with left child L and right child R, hd(N) ∈ {hd(L), hd(R)}.

Define the head child H(N) as the unique child of N whose head equals hd(N); the other child is the nonhead child.

Define the canonical class HCB(n) of head-annotated PS trees to be those P that satisfy, for each head index i ∈ [n], the following canonical spine condition. Let Si be the maximal path from leaf i upward through consecutive nodes M with hd(M) = i. Along Si from bottom to top:

- • Every internal node on Si with a nonhead child whose head d satisfies d < i (a left-dependent) appears before any internal node on Si with a nonhead child whose head d satisfies d > i (a right-dependent).
- • Among left-dependent attachments along Si, the heads d appear in strictly increasing order of d.
- • Among right-dependent attachments along Si, the heads d appear in strictly increasing order of d.

Additionally, if at a node on Si the nonhead child head is d < i, then that nonhead child is the left child; if d > i, it is the right child.

- Define the mapping F from the set ProjDep(n) of projective dependency trees on [n] to HCB(n) recursively as follows. For

D ∈ ProjDep(n) and i ∈ [n], let

Li = {d ∈ [n] : d < i and (i, d) ∈ E}, Ri = {d ∈ [n] : d > i and (i, d) ∈ E}, sorted increasingly as l1 < · · · < lk and r1 < · · · < rm. Define T(i) by recursion on the size of subD(i):

- • If subD(i) = {i}, then T(i) is the single leaf i with hd = i.
- • Otherwise, after constructing T(l) and T(r) for all l ∈ Li and r ∈ Ri, build a sequence of binary nodes with head i by first attaching left dependents in increasing order on the left and then right dependents in increasing order on the right:

S0 := leaf i with hd = i.

For t = 1, . . . , k, set St to be a node with left child T(lt), right child St−1, hd = i. For s = 1, . . . , m, set Sk+s to be a node with left child Sk+s−1, right child T(rs), hd = i. Define T(i) := Sk+m.

Finally, if r is the root of D, define F(D) := T(r).

- Define the mapping G from HCB(n) to ProjDep(n) as follows. Given P ∈ HCB(n), let r := hd(root(P)). Define EP by adding, for each internal node N with children L and R:

- • If hd(N) = hd(L) and hd(R) = d ̸= hd(N), add the arc (hd(N), d).
- • If hd(N) = hd(R) and hd(L) = d ̸= hd(N), add the arc (hd(N), d). Continued on next page

Define G(P) := (V = [n], EP, root r). Prove that F is a bijection between ProjDep(n) and HCB(n), and that G is its inverse, i.e.,

###### G ◦ F = IdProjDep(n) and F ◦ G = IdHCB(n).

Answer: F is a bijection from ProjDep(n) (projective dependency trees on [n]) to HCB(n) (canonical head-annotated binary PS trees on [n]) and G is its inverse. Concretely, for all D ∈ ProjDep(n), G(F(D)) = D, and for all P ∈ HCB(n), F(G(P)) = P.

Detailed Solution: Okay, so I need to prove that the mappings F and G are inverses of each other, establishing a bijection between projective dependency trees and the canonical head-annotated phrase-structure trees in HCB(n). Let me start by recalling the definitions to make sure I understand them correctly. First, a projective dependency tree D on [n] is a directed tree with root r, where each node except r has exactly one incoming arc (so it’s a rooted tree with edges directed away from the root, I think?), and for every arc (h, d), all positions between h and d are in the subtree of h. That’s the projectivity condition. So if h is the head of d, then the span of h must include all the words between h and d. That makes sense for projectivity in dependency parsing, where dependencies can’t cross over other words without including them. Then, HCB(n) is the set of ordered binary head-annotated phrase-structure trees that satisfy the canonical spine condition.

... To establish that the mappings F and G form a bijection between the set of projective dependency trees ProjDep(n) and the set of canonical head-annotated binary phrase-structure trees HCB(n), we must prove two key properties:

- 1. G ◦ F = idProjDep(n): For every projective dependency tree D ∈ ProjDep(n), applying F to D yields a tree P = F(D) ∈ HCB(n), and then applying G to P recovers the original dependency tree D.
- 2. F ◦ G = idHCB(n): For every canonical head-annotated phrase-structure tree P ∈ HCB(n), applying G to P yields a dependency tree D = G(P) ∈ ProjDep(n), and then applying F to D recovers the original phrase-structure tree P.

- 1. Proof that G ◦ F = idProjDep(n) Let D ∈ ProjDep(n). The mapping F constructs a binary phrase-structure tree P = F(D) by recursively building subtrees for each node i ∈ [n], attaching left dependents (those d < i with (i, d) ∈ E) in increasing order on the left, and right dependents (those d > i with (i, d) ∈ E) in increasing order on the right. Key observations:

- • The construction ensures that the leaves of T(i) are exactly the elements of the span spD(i), which is a contiguous interval due to the projectivity condition.
- • The in-order traversal of the resulting tree P gives the correct linear order 1, 2, . . . , n, as required for a valid phrase-structure tree.
- • Each internal node in P corresponds to an arc in D: if a node has head i and a nonhead child with head d, then G adds the arc (i, d).
- • Since F builds the tree by attaching all direct dependents of i in the exact order they appear in D, and G recovers these arcs via the head annotations, the resulting dependency tree from G(P) matches D exactly.

Formal justification:

- • Let (h, d) ∈ E be an arc in D. Then d ∈ Lh ∪ Rh, and F attaches d as a nonhead child along the spine of h in P.
- • When G processes this node in P, it adds the arc (h, d), since the head of the node is h, and the nonhead child has head d.
- • All arcs in D are thus recovered by G(F(D)), and the root of G(F(D)) is r = hd(root(P)), which is the same as the root of D.

Therefore, G(F(D)) = D.

- 2. Proof that F ◦ G = idHCB(n) Let P ∈ HCB(n). The mapping G constructs a dependency tree D = G(P) by adding an arc (h, d) for each internal node where the head is h and the nonhead child has head d.

Continued on next page

###### Key observations:

- • The canonical spine condition in HCB(n) ensures that:

- – Left-dependent attachments (where d < i) occur before right-dependent ones along the spine of i.
- – Among left-dependent attachments, the heads d increase as we move up the spine.
- – Similarly for right-dependent attachments.

- • These conditions match the attachment order used in F: left dependents are attached in increasing order on the left, and right dependents in increasing order on the right.
- • Thus, when F is applied to D = G(P), it reconstructs the same structure as P, because the order of attachment in F aligns precisely with the canonical spine structure of P.

###### Formal justification:

- • Consider any node i ∈ [n] in P. The canonical spine condition dictates the order in which its dependents are attached.
- • G(P) produces a dependency tree D where the direct dependents of i are exactly those heads d that appear as nonhead children in the spine of i in P.
- • F(D) then rebuilds the subtree rooted at i by attaching these dependents in increasing order (left first, then right), which matches the canonical spine condition.
- • Since the attachment order in F mirrors the structure enforced by the canonical spine condition in P, the resulting tree F(G(P)) is identical to P.

Conclusion We have shown that:

- • G(F(D)) = D for all D ∈ ProjDep(n),
- • F(G(P)) = P for all P ∈ HCB(n).

This confirms that the mappings F and G are mutual inverses, establishing a bijection between ProjDep(n) and HCB(n).

|G ◦ F = idProjDep(n) and F ◦ G = idHCB(n)|
|---|

###### Correctness: True

