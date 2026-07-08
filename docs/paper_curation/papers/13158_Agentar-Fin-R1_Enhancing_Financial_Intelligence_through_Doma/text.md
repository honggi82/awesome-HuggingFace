D I G I T A L

T E C H N O L O G I E S

## Agentar-Fin-R1: Enhancing Financial Intelligence through Domain Expertise, Training Efficiency, and Advanced Reasoning

# arXiv:2507.16802v4[cs.CL]27Jul2025

Yanjun Zheng∗, Xiyang Du∗, Longfei Liao∗ Xiaoke Zhao, Zhaowen Zhou, Jingze Song, Bo Zhang, Jiawei Liu Xiang Qi, Zhe Li, Zhiqiang Zhang, Wei Wang, Peng Zhang† Ant Group

### Abstract

Large Language Models (LLMs) exhibit considerable promise in financial applications; however, prevailing models frequently demonstrate limitations when confronted with scenarios that necessitate sophisticated reasoning capabilities, stringent trustworthiness criteria, and efficient adaptation to domain-specific requirements. We introduce the Agentar-Fin-R1 series of financial large language models (8B and 32B parameters), specifically engineered based on the Qwen3 foundation model to enhance reasoning capabilities, reliability, and domain specialization for financial applications. Our optimization approach integrates a high-quality, systematic financial task label system with a comprehensive multi-layered trustworthiness assurance framework. This framework encompasses high-quality trustworthy knowledge engineering, multi-agent trustworthy data synthesis, and rigorous data validation governance. Through label-guided automated difficulty-aware optimization, tow-stage training pipeline, and dynamic attribution systems, we achieve substantial improvements in training efficiency. Our models undergo comprehensive evaluation on mainstream financial benchmarks including FinEval 1.0, and FinanceIQ, as well as general reasoning datasets such as MATH-500 and GPQA-diamond. To thoroughly assess real-world deployment capabilities, we innovatively propose the Finova evaluation benchmark, which focuses on agent-level financial reasoning and compliance verification. Experimental results demonstrate that Agentar-Fin-R1 not only achieves state-of-the-art performance on financial tasks but also exhibits exceptional general reasoning capabilities, validating its effectiveness as a trustworthy solution for high-stakes financial applications. The Finova bench is available at https://github.com/antgroup/Finova.

*Equal contribution. †Corresponding Author. ‡{zhengyanjun.zyj, duxiyang.dxy, liaolongfei.llf}@antgroup.com

###### 1. Introduction

XuanYuan-FinX1-Preview-70B Dianjin-R1-32B Qwen3-32B GPT-o1 DeepSeek-R1 Agentar-Fin-R1-32B

100.00

95.40

95.20

94.80

93.80

90.00

87.70

86.79

84.93

84.40

83.98

81.49

81.32

80.50

80.41

78.72

80.00

78.03

76.77

72.39

71.80

69.93

70.02

69.03

70.00

68.18

63.13

61.28

60.46

59.46

60.00

58.59

56.02

50.74

50.00

42.42

40.00

FinEval 1.0 FinanceIQ Finova MATH GPQA

Figure 1: Agentar-Fin-R1-32B performance: significantly outperforms on financial benchmarks (FinEval 1.0, FinanceIQ, Finova), and on general reasoning benchmarks (MATH-500, GPQAdiamond), outperforms comparable-sized general models while achieving performance comparable to large-scale models like DeepSeek R1 and GPT-o1.

Large Language Models (LLMs) have demonstrated remarkable capabilities in complex reasoning tasks, with recent advances in reasoning-optimized models such as OpenAI’s o1 series [17], QwQ [19], DeepSeek-R1 [10],Seed-Thinking-v1.5[21] and Qwen3 [28] achieving significant breakthroughs in mathematics, programming, and logical inference. However, the direct deployment of these general-purpose models in financial applications reveals critical limitations: insufficient domain-specific financial knowledge integration, leading to poor performance on finance-related tasks; susceptibility to hallucinations that violate the stringent safety and compliance requirements essential in financial environments.

Existing financial LLMs can be broadly categorized into two types. Non-reasoning financial models such as Baichuan [30], DISC-FinLLM[3], XuanYuan [6], and PIXIU [25] incorporate domain-specific financial knowledge but lack sophisticated analytical and reasoning capabilities required for complex financial decision-making scenarios involving multi-step analysis, risk assessment, and strategic planning. Reasoning-enhanced financial models including XuanYuanFinX1-Preview [7], Fino1 [18], Fin-R1 [15], and Dianjin-R1 [32] attempt to integrate advanced reasoning mechanisms but still exhibit significant limitations: insufficient reasoning capabilities for handling complex financial scenarios that require deep analytical thinking; lack of scenariospecific reasoning adaptation, failing to align reasoning processes with the unique demands of financial contexts such as market dynamics, regulatory compliance constraints, and risk tolerance considerations.

We refer to some current consensus in academia and industry(Liu et al. [15],Wang et al. [24],Dong et al. [4],Fatouros et al. [8],Li et al. [13],Tong et al. [23],Xie et al. [26],Zhang et al. [30]), and here identify three fundamental requirements for effective financial AI systems that distinguish them from general-purpose applications:

- 1. Adaptive Knowledge Integration: Efficient acquisition and assimilation of evolving domain knowledge, including regulatory updates and emerging financial instruments.

- 2. Verifiable Reasoning: Transparent, auditable reasoning processes essential for stakeholder confidence in high-stakes decisions.
- 3. Compliance Adherence: Robust protection of sensitive data while meeting stringent regulatory standards.

Based on the aforementioned limitations, we introduce Agentar-Fin-R1, a family of reasoningoptimized financial LLMs that systematically addresses key challenges in the financial domain through three core innovations:

Professional Label-Guided Framework: We construct a fine-grained financial task label system that decomposes the financial domain into precisely defined categories, serving as an active guidance framework throughout the entire development pipeline. This label system not only guides data processing and training workflows but also enables systematic task-oriented optimization, ensuring comprehensive coverage of financial reasoning scenarios and providing professional support for model training.

Multi-Dimensional Trustworthiness Assurance: Our framework ensures trustworthiness through three levels: (i) source trustworthiness via rigorous knowledge engineering of authenticated financial data; (ii) synthesis trustworthiness through verifiable multi-agent collaborative frameworks that guarantee data quality; and (iii) governance trustworthiness via comprehensive data processing including deduplication, toxicity removal, and preference-based filtering.

Efficient Training Optimization: We achieve scalable and efficient development through multiple dimensions: (i) data efficiency via weighted training frameworks that deeply exploit data potential, enhanced by label-guided synthesis and intelligent selection to improve data utilization; (ii) training efficiency through a two-stage training strategy that further enhances model capabilities; and (iii) attribution efficiency via a comprehensive attribution system that enables rapid bottleneck identification and targeted improvements, providing scientific guidance for continuous model evolution.

To assess real-world deployment capabilities, we introduce Finova (Financial Nova: Operational, Verifiable, Agent), a comprehensive benchmark encompassing three critical dimensions:

- • Agent Capabilities: Autonomous task execution including intent detection, slot recognition, tool planning, and expression generation.
- • Complex Reasoning: Multi-step analytical tasks combining financial mathematics, code understanding, and domain-specific inference.
- • Safety and Compliance: Security risk mitigation and regulatory adherence assessment.

Our primary contributions are:

- • A label-guided methodology for developing trustworthy and efficient financial LLMs that systematically addresses data fragmentation, reasoning transparency, and scenario generalization challenges.
- • Agentar-Fin-R1 model series (8B and 32B parameters) achieving state-of-the-art performance on financial benchmarks while maintaining general reasoning capabilities, demonstrating the effectiveness of our training methodology.
- • Finova evaluation benchmark providing standardized assessment of financial LLM capabilities across critical application dimensions, enabling systematic comparison and development guidance for the research community.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

##### Agentar-Fin-R1

[Figure 5]

[Figure 6]

###### Data Training

###### Expertise, Efficient, Evolution

Tasks label System

Data Source Governance

Weighted Training Framework

Extraction Normalization Detoxification Refinement

[Figure 7]

[Figure 8]

[Figure 9]

|Difficulty-Aware Weight Estimation|
|---|

[Figure 10]

[Figure 11]

Different task with different weight

Sampled Data

Refined Knowledge

Reliable Data source

|Enhanced Loss with Weighted|
|---|

[Figure 12]

SceneDimension

[Figure 13]

[Figure 14]

Data Synthesis

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Two-Stage Training

Professional Self-QA Agent

[Figure 19]

[Figure 20]

[Figure 21]

Refined Knowledge

Task-specific Agent

Q T A

Stage 1 Stage 2

Q

[Figure 22]

Task Label

GRPO

[Figure 23]

T

SFT

[Figure 24]

Instruction Evolution Agent

Attribute Dimension Task with good performance Task with bad performance

SFT

A

Base model

Financial Knowledge Injection

Original QA Pairs

Agentar-Fin

Q

T A

Auto-Evolution

Challenge Enhancement

*

Attribution Loop

Data Verifying and Checking

[Figure 25]

Multi-Model Verifying Agent

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

|Data Augmentation Strategy|
|---|

|Evaluate|
|---|

|Error Analysis|
|---|

[Figure 34]

Sampled Golden Manual Annotation Rating Model Decontamination

[Figure 35]

[Figure 36]

[Figure 37]

Q T A

Detoxification Deduplication

Figure 2: Overview of the Agentar-Fin-R1 development pipeline.

Our experimental results demonstrate that Agentar-Fin-R1 achieves superior performance across financial benchmarks (FinEval 1.0, FinanceIQ and Finova) while maintaining competitive results on general reasoning tasks (MATH-500, GPQA-diamond), validating the effectiveness of domain-specialized optimization without catastrophic forgetting.

###### 2. Data

###### 2.1. Overview

In developing financial large language models (LLMs), data quality and integrity serve as the cornerstone of model performance. While data volume remains important, ensuring trustworthiness and real-world representativeness for financial applications is absolutely critical. Our methodology is built upon a sophisticated Label System that systematically structures the data synthesis process, guaranteeing that each data point is verifiable, task-specific, and precisely aligned with domain requirements.

This comprehensive system facilitates transparent and efficient data generation, enabling the creation of high-quality, task-relevant datasets that maintain rigorous logical consistency and establish a solid foundation for training robust financial models.

###### 2.2. Label System

The Label System serves as the cornerstone of our data construction pipeline, meticulously designed to capture the inherent complexity and heterogeneity of financial tasks. It systematically categorizes tasks along two fundamental dimensions:

- • Scene Dimension: Encompasses diverse real-world financial scenarios, such as Banking, Securities, Insurance, Trusts, and Mutual Funds. Each scenario embodies a distinct applica-

- tion context with specialized requirements, enabling the model to adapt to the operational nuances and domain-specific characteristics of each financial sector.
- • Task Type Dimension: Defines the specific types of tasks to be performed, such as Named Entity Recognition (NER), Intent Classification, Slot Filling, Entity Disambiguation, and Consultation-style Question Answering. These task types specify the exact operations the model should execute on input data, providing clear directives for what actions to take during instruction learning and response generation phases.

We formally define the label system as a set L of composite labels, where each label 𝑙𝑖 is represented as a tuple comprising a content category and task attribute:

𝑙𝑖 = (𝑐𝑖, 𝑎𝑖) (1)

Here, 𝑐𝑖 ∈ C represents a scene category (e.g., Banking, Insurance), and 𝑎𝑖 ∈ A represents a task attribute (e.g., Entity Disambiguation, Slot Filling). This structured formulation facilitates fine-grained task decomposition and enables targeted data generation that is precisely aligned with downstream application requirements.

It is crucial to recognize that the Scene and Task Attribute dimensions exhibit non-orthogonal characteristics: not all task attributes are equally applicable across every scene. This phenomenon results in a sparsely populated cross-product space, which more authentically reflects the natural distribution and practical constraints of real-world financial tasks.

###### 2.3. Data Construction

Our data construction methodology is meticulously designed to ensure the quality, diversity, and fidelity of synthesized data. It integrates rigorous knowledge engineering, sophisticated multi-agent synthesis mechanisms, and rigorous multi-stage verification processes. The resulting dataset comprehensively captures the multifaceted nature of financial tasks and is optimally suited for training large-scale financial language models capable of robust generalization, domain-specific adaptation, and high-stakes reasoning under complex financial scenarios.

###### 2.3.1. Source: Trusted Sources and Knowledge Engineering

We construct reliable data by sourcing from authoritative financial institutions and regulatory bodies, while applying sophisticated knowledge engineering techniques to ensure data integrity and domain relevance. The sourced data undergoes comprehensive knowledge engineering to guarantee authenticity and domain alignment through a systematic multi-stage preprocessing pipeline:

- 1. Data Extraction: Processing raw financial data using state-of-the-art NLP techniques, including Named Entity Recognition (NER), dependency parsing, and Part-of-Speech (POS) tagging, to systematically extract meaningful financial entities, relationships, and semantic structures.
- 2. Data Normalization: Standardizing heterogeneous data formats and reconstructing data structures to achieve uniform financial data representation, thereby enhancing downstream semantic understanding and cross-domain compatibility.
- 3. Data Detoxification: Systematically removing non-compliant, contaminated, and potentially harmful content from the dataset to ensure data quality, regulatory compliance, and ethical standards adherence.

- 4. Knowledge Refinement: Applying advanced processing and quality enhancement techniques to generate a high-fidelity refined knowledge repository that meets stringent financial domain requirements.

The final refined knowledge repository 𝐾 comprises high-quality structured financial knowledge units that have undergone rigorous validation:

𝐾 = 𝑘1, 𝑘2, . . . , 𝑘𝑛

where each knowledge unit 𝑘𝑖 has undergone comprehensive verification and refinement processes to ensure accuracy, relevance, and domain-specific validity.

###### 2.3.2. Synthesis: Trusted Multi-Agent Generation of Reasoning Triplets

To construct a trustworthy, diverse, and reasoning-enhanced instruction dataset tailored for financial tasks, we design a dual-track data synthesis pipeline that combines domain-grounded generation with self-evolving instruction refinement. The final dataset comprises high-quality (query, thinking, answer) triplets that are semantically aligned with the financial domain and optimized for verifiable reasoning.

Input

Professional Self-QA Agent

Post Checking

Multi-Model Verifying Agent

[Figure 38]

[Figure 39]

Task-specific Agent

Sampled Manual Annotation

[Figure 40]

[Figure 41]

[Figure 42]

Refined Knowledge

&

Thinking

Agent Pools

[Figure 43]

[Figure 44]

Query

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Rating Model

Answer

Task label

Consistency

Input

Instruction Evolution Agent

- A1

- A2

Decontamination Detoxification Deduplication

[Figure 50]

[Figure 51]

Q

[Figure 52]

Thinking

New Query

Answer

Original QA Pairs

Golden Data

AutoEvolution

Reasoning

Query Thinking Answer

New Query

Thinking Answer

[Figure 53]

[Figure 54]

Q T1 A

- Figure 3: Overview of the dual-track data synthesis pipeline, incorporating both task-oriented knowledge-guided generation and self-evolution mechanisms for generating verifiable reasoning triplets.

- Track I: Task-Oriented Knowledge-Guided Generation This track leverages a curated financial knowledge base and a domain-specific task label system to drive the generation of high-quality, verifiable (query, thinking, answer) triplets.

Task Label Matching Mechanism We define a structured label system of financial tasks as T = {𝑡1,𝑡2, . . . ,𝑡𝑚}, where each label 𝑡𝑗 ∈ T represents a distinct task category (e.g., fraud detection, portfolio analysis, regulatory compliance). For each task label 𝑡𝑗, a dedicated generation agent 𝐴𝑡𝑗 is instantiated.

Knowledge-Guided Generation Process Given a task label 𝑡𝑗 ∈ T and a domain-specific knowledge snippet 𝑘𝑖 ∈ K𝑡𝑗 ⊆ K, the agent 𝐴𝑡𝑗 generates a reasoning triplet:

(𝑞𝑖,𝑡𝑖, 𝑎𝑖) = 𝐴𝑡𝑗(𝑘𝑖,𝑡𝑗;𝜃𝐴𝑡

), (2)

𝑗

where 𝑞𝑖 is the query, 𝑡𝑖 is the intermediate reasoning process ("thinking"), and 𝑎𝑖 is the final answer. The generation process respects financial logic and compliance constraints through controlled decoding and templated prompting. The resulting dataset is:

𝐷task = {(𝑞𝑖,𝑡𝑖, 𝑎𝑖)}𝑖𝑁=task1 . (3)

- Track II: Self-Evolution of Instructions with Reasoning Supervision To promote diversity and complexity, this track evolves existing prompts into more sophisticated reasoning tasks using a feedback-driven self-evolution agent.

Seed Initialization and Evolution Mechanism Starting from an initial set 𝐼0, which includes either manually curated queries or samples from 𝐷task, the self-evolution agent 𝐴evo generates enhanced instructions by incorporating feedback signals:

𝐼𝑘+1 = 𝐴evo(𝐼𝑘, R𝑘;𝜃evo), (4)

where R𝑘 includes diversity metrics, task novelty scores, and answerability filters. The process continues until a convergence criterion (e.g., saturation in novelty or quality) or maximum iteration count 𝐾max is reached.

Evolution Strategies The instruction refinement process is guided by three core strategies:

- • Progressive Reasoning Complexity: Injecting step-by-step chains of thought to increase cognitive depth and analytical rigor.
- • Structural Diversity: Applying prompt mutations and recombinations to expand coverage across financial domains and reasoning types.
- • Fitness-Based Filtering: Retaining only samples that demonstrate factual soundness, logical coherence, and linguistic fluency.

Each refined instruction is used to produce a reasoning triplet using the base model:

𝐷evolution = {(𝑞𝑗,𝑡𝑗, 𝑎𝑗)}𝑁𝑗=evo1 . (5)

Final Trusted Reasoning Dataset The final corpus is constructed as the union of task-guided and self-evolved reasoning triplets:

𝐷synthesis = 𝐷task ∪ 𝐷evolution. (6)

This dual-track framework ensures that the generated data is not only domain-specific and diverse but also trustworthy and verifiable, facilitating the training of financial LLMs with robust reasoning and compliance capabilities.

###### 2.3.3. Verification and Checking: Rigorous Multi-Modal Validation

We implement a comprehensive multi-tier validation framework to ensure data quality, accuracy, and reliability across all generated instances, establishing a robust foundation for trustworthy financial AI systems.

###### Multi-Model Ensemble Verification

Consistency Validation We deploy multiple independent models {𝑀1, 𝑀2, . . . , 𝑀𝑝} to generate responses for identical queries, assessing data accuracy through comprehensive answer consistency analysis:

∑︁𝑝

1 𝑝(𝑝 − 1)

consistency(𝑞𝑖) =

𝑗=1

###### ∑︁

sim(𝑀𝑗(𝑞𝑖), 𝑀𝑘(𝑞𝑖)) (7)

𝑘≠𝑗

where sim(·, ·) represents a semantic similarity function that incorporates both lexical overlap and contextual embedding similarity measures to capture nuanced agreement patterns.

Reasoning Validation An independent third-party model validates the logical correctness of answers through a prompt-based approach, analyzing queries and reasoning processes to determine answer validity:

reasoning_valid(𝑞𝑖, 𝑎𝑖) = 𝑀verify(query(𝑞𝑖),thinking(𝑞𝑖)) → 𝑎𝑖 (8)

Human Annotation and Quality Control We perform stratified random sampling of the generated data to ensure representative coverage of task types, complexity levels, and domain subcategories. The sampling ratio is carefully calibrated to balance annotation cost with statistical significance and coverage requirements. Experienced financial domain experts will conduct comprehensive multidimensional assessment of sampled data instances.

###### Rating Model Training and Application

Training Data Construction We construct a comprehensive training dataset for the rating model by strategically combining multi-model ensemble verification results with expert human annotation data:

𝐷rating = 𝐷ensemble ∪ 𝐷human (9) This hybrid approach leverages both automated consistency checks and expert human

judgment to create robust and reliable training signals for quality assessment.

Rating Model Architecture We train a specialized rating model 𝑅𝑀 to perform comprehensive final quality assessment:

score(𝑑𝑖) = 𝑅𝑀(𝑑𝑖;𝜃RM) (10)

Data Governance and Cleansing Our data governance framework implements rigorous cleansing procedures to ensure dataset integrity:

- 1. Deduplication: Employing advanced semantic hashing and similarity computation techniques to identify and remove duplicate instances while preserving meaningful variations and edge cases.
- 2. Detoxification: Systematically identifying and filtering potentially harmful, biased, or inappropriate content that could produce negative downstream effects or ethical concerns.
- 3. Decontamination: Identifying and removing training data instances that overlap with evaluation benchmarks to prevent data leakage and ensure fair, unbiased model assessment.

Final Dataset Definition The final dataset, having undergone complete verification and cleansing procedures, is formally defined as:

𝐷final = {𝑑𝑖 ∈ 𝐷synthesis | verify(𝑑𝑖) ∧ clean(𝑑𝑖) ∧ score(𝑑𝑖) > 𝜏} (11)

where verify(𝑑𝑖) indicates successful multi-modal verification, clean(𝑑𝑖) represents successful data cleansing, and 𝜏 is the quality threshold determined through rigorous empirical validation and domain expert consensus.

This comprehensive data construction pipeline ensures that the final training dataset maintains exceptional quality, diversity, and task relevance, providing a robust and trustworthy foundation for training large language models in the financial domain.

###### 3. Training

###### 3.1. Weighted Training Framework

Training financial large language models (LLMs) involves addressing the inherent heterogeneity and complexity of financial tasks, which exhibit varying levels of difficulty and domain-specific requirements. Traditional training methods treat all training samples uniformly, without accounting for the fact that some tasks are significantly more challenging than others. Consequently, models may overfit to simpler, more frequently encountered tasks while underperforming on complex tasks that are crucial for real-world financial decision-making and risk assessment.

To address this fundamental limitation, we propose a weighted training framework that dynamically adjusts the importance of each task based on the empirically measured difficulty of the corresponding instances. This framework employs a sophisticated domain-specific tagging system that categorizes tasks by their semantic labels and complexity characteristics. Prior to training, for each task label, a representative subset of 𝑛 samples is selected through stratified sampling, and the current model generates 𝑘 diverse responses for each sample. The pass@k score [2, 29] is then computed for these responses, which quantitatively reflects the model’s ability to produce correct answers within the top 𝑘 generated responses. Additionally, 𝑚 reference models from different architectural families and training paradigms are employed to generate their own response sets, and their respective pass@k values are computed for comparative analysis.

The pass@k values from both the current model and the 𝑚 reference models are systematically used to assess task difficulty and relative model performance. Tasks with lower pass@k scores for the current model are identified as more challenging and receive proportionally higher training weights. Furthermore, when there exists a significant performance gap between the current model and reference models, the weight for that specific task is increased to reflect the model’s relative weakness and prioritize improvement in that domain.

The computed difficulty weights are then assigned to the corresponding task labels, and during the training process, tasks with higher weights receive enhanced attention through modified loss functions. This approach ensures that the model focuses more computational resources on tasks it struggles with, thereby improving performance on complex financial tasks while maintaining learning efficiency for simpler tasks.

Difficulty-Aware Weight Estimation The difficulty-aware weight for each task label is computed based on a comprehensive analysis of pass@k values from both the current model and reference models. Let D = {(𝑥𝑖, 𝑦𝑖,𝑡𝑖)} represent the tagged dataset, where 𝑥𝑖 is the input data, 𝑦𝑖 is the target output, and 𝑡𝑖 is the task label. For each task label 𝑡, a representative subset of 𝑛 samples is selected through stratified sampling to ensure comprehensive coverage across different subtask variations and complexity levels.

To ensure stable training dynamics and prevent oscillatory behavior caused by abrupt weight shifts between training epochs, we employ a sophisticated exponential smoothing mechanism for task difficulty weights. Specifically, for each task label 𝑡, the final weight is updated according to:

𝑤𝑡(final) = 𝜌 · 𝑤𝑡(prev) + (1 − 𝜌) · 𝑤𝑡(raw) (14) where 𝜌 ∈ [0,1] is a smoothing coefficient that controls the inertia of the update process,

𝑤𝑡(prev) denotes the previous smoothed weight from the preceding epoch, and 𝑤𝑡(raw) is the newly estimated raw difficulty weight.

By progressively integrating new difficulty estimates while maintaining historical context, this mechanism effectively mitigates training instability and sharp fluctuations in learning dynamics.

To ensure that no task category is completely neglected during training, we apply a lowerbound clipping mechanism such that each final weight satisfies:

Algorithm 1: Difficulty-Aware Weight Estimation for Task Labels Input: D = {(𝑥𝑖, 𝑦𝑖,𝑡𝑖)}: tagged dataset, where 𝑥𝑖 is the input, 𝑦𝑖 the target, and 𝑡𝑖 the task

label;

{𝑀𝑗}𝑚𝑗=1: reference models from diverse architectural families; 𝛼, 𝛽, 𝛾: weighting hyperparameters for difficulty components; 𝜌: exponential smoothing coefficient; 𝑛, 𝑘: sampling and generation hyperparameters

Output: Final normalized difficulty weights 𝑤˜𝑡 for each task label foreach task label 𝑡 ∈ T do

Draw 𝑛 instances {(𝑥ℓ, 𝑦ℓ)}𝑛ℓ=1 for task 𝑡 via stratified sampling; Generate 𝑘 diverse responses with current model and compute pass@kcurrent(𝑡); for 𝑗 ← 1 to 𝑚 do

Generate 𝑘 responses with reference model 𝑀𝑗 and compute pass@k𝑗(𝑡);

###### ∑︁𝑚

1

Compute average reference performance: pass@kref(𝑡) =

pass@k𝑗(𝑡); Compute raw difficulty weight:

𝑚

𝑗=1

𝑤𝑡(raw) = 𝛼 1 − pass@kcurrent(𝑡) + 𝛽 max 0,pass@kref(𝑡) − pass@kcurrent(𝑡) + 𝛾 (12) if task 𝑡 encountered in previous epochs then

Apply exponential smoothing: 𝑤𝑡(final) = 𝜌𝑤𝑡(prev) + (1 − 𝜌) 𝑤𝑡(raw); else

Initialize: 𝑤𝑡(final) = 𝑤𝑡(raw);

Store 𝑤𝑡(final) as 𝑤𝑡(prev) for subsequent epochs; Apply normalization for stable scaling:

𝑤𝑡(final) 𝑡′∈T 𝑤𝑡(′final)

𝑤˜𝑡 =

· |T | (13)

###### return {𝑤˜𝑡}𝑡∈T

𝑤𝑡(final) ⩾ 𝛾 (15)

where 𝛾 > 0 is a carefully tuned base weight that preserves minimal attention on all task types, including relatively straightforward ones. The weights are subsequently normalized across all tasks to maintain a consistent global training scale and prevent loss magnitude drift.

The difficulty-aware weight 𝑤𝑡 for each task label 𝑡 is computed as a principled combination of three key components: (1) the inverse of the current model’s pass@k score to prioritize challenging tasks, (2) an additional penalty term when the current model significantly underperforms compared to reference models, and (3) a base weight to ensure comprehensive task coverage. The exponential smoothing mechanism prevents dramatic weight oscillations between training iterations, promoting stable and consistent convergence behavior.

Enhanced Loss Functions with Weighted Training Once the task difficulty weights 𝑤˜𝑡 have been computed and properly normalized, they are systematically incorporated into the training process through modified loss functions. For Supervised Fine-Tuning (SFT), the standard crossentropy loss function is enhanced by weighting the log-likelihood loss for each training sample according to its task difficulty:

###### ∑︁𝑁

1

𝑤˜𝑡𝑖 · log 𝑃𝜃(𝑦𝑖|𝑥𝑖) (16)

LSFT = −

𝑁

𝑖=1

where 𝑁 is the total number of training samples, and 𝑤˜𝑡𝑖 is the normalized difficulty weight for the task label associated with sample 𝑖. This formulation ensures that the overall loss magnitude remains comparable to standard training procedures while systematically emphasizing difficult tasks.

For Reinforcement Learning (RL) training phases, we modify the preference-based objective function to incorporate difficulty weights in a theoretically principled manner.

The computational overhead for difficulty estimation is 𝑂(𝑚 · 𝑛 · 𝑘) per task label, where 𝑚 is the number of reference models, 𝑛 is the number of sampled instances per task, and 𝑘 is the number of responses generated per instance. This estimation procedure is performed periodically (e.g., once per epoch or at specified intervals) rather than at every training step, ensuring that it does not significantly impact overall training efficiency or computational scalability.

This enhanced weighted training framework, incorporating empirically-driven task difficulty assessment through pass@k scores with theoretical stability guarantees and practical implementation considerations, is seamlessly integrated into the training pipeline to improve model generalization and performance on complex financial tasks while maintaining robust and stable learning dynamics throughout the training process.

###### 3.2. Two-Stage Training Pipeline

We propose a two-stage training strategy to systematically optimize financial large language models (LLMs) for domain-specific applications. This approach addresses the challenge of balancing comprehensive financial knowledge acquisition with performance optimization on challenging tasks.

Our strategy consists of two sequential stages:

- 1. Stage 1: Financial Knowledge and Capability Injection – Comprehensive domain knowledge and capability acquisition through supervised fine-tuning on diverse financial tasks.
- 2. Stage 2: Challenge Task Enhancement – Performance optimization on challenging tasks using GRPO and targeted fine-tuning.

- Stage 1: Financial Knowledge and Capability Injection The first stage employs supervised fine-tuning (SFT) leveraging high-quality financial reasoning data synthesized through our approach described in Section 2, augmented with extensive general reasoning datasets. We implement the weighted training framework from the previous section, which strategically prioritizes challenging samples to accelerate convergence on complex problems. This stage substantially enhances the model’s comprehensive capabilities across the financial domain,

- establishing a robust foundation that integrates both specialized domain knowledge and general reasoning proficiency.
- Stage 2: Challenge Task Enhancement The second stage is specifically designed to further strengthen the model’s performance when confronting difficult and challenging problems. We employ a sophisticated hybrid approach combining:

- • GRPO: Optimizes decision-making capabilities in complex financial scenarios with multiobjective considerations and intricate reward structures
- • Targeted SFT: Systematically addresses specific performance gaps and weaknesses identified through comprehensive Stage 1 evaluation

Tasks demanding sophisticated reasoning capabilities (e.g., multi-step financial forecasting, comprehensive risk assessment, dynamic portfolio optimization) are prioritized in this stage. When GRPO encounters convergence challenges on specific task categories, we strategically apply targeted SFT using carefully curated high-quality examples to ensure robust and consistent performance across all challenging scenarios.

Training Efficiency and Scalability This two-stage approach delivers significant practical advantages for real-world deployment:

- • Efficient initialization: Stage 1 provides a strong foundational model, dramatically reducing fine-tuning requirements for domain adaptation
- • Flexible modular optimization: Stage 2 can be selectively applied to specific task categories based on business priorities and requirements
- • Cost-effective scalability: The pipeline enables efficient adaptation to emerging financial domains and use cases without requiring complete model retraining

###### 3.3. Attribution Loop

The Attribution Loop is a post-training mechanism that refines the model by tracing errors to specific financial scenarios and tasks, enabling targeted data sampling and model enhancement through dynamic resource allocation.

Pass@1 Attribution Framework The Attribution Loop employs the aforementioned twodimensional labeling framework to categorize prediction errors.

For a given label 𝑡, the pass@1 accuracy is defined as:

I[𝑦ˆ𝑖 = 𝑦𝑖] |{𝑖 : ℓ𝑖 = 𝑡}|

Pass@1(𝑡) = 𝑖:ℓ𝑖=𝑡

(17)

Dynamic Attribution Loop The Dynamic Attribution Loop optimizes data allocation and model training by adaptively prioritizing tasks based on their performance gaps, learning efficiency, and available resources. The goal is to minimize training cost while ensuring target performance across all tasks, while also addressing the diminishing returns of overfitting.

###### • Task Prioritization:

- – Compute performance gap for each task: Δ𝑡 = max(0, 𝑃target − 𝑝𝑡) (18)
- – Estimate learning efficiency based on the ratio of performance improvement to data added in the current iteration:

𝑒𝑡 =

max 0, Δ𝑝𝑡(𝑘) Δ𝑑𝑡(𝑘) + 𝜀

(19)

where Δ𝑝𝑡(𝑘) is the performance improvement of task 𝑡 in the current iteration, and Δ𝑑𝑡(𝑘) is the amount of data allocated to task 𝑡 in the same iteration.

- – The priority score 𝜋𝑡 for each task is computed as: 𝜋𝑡 = Δ𝑡 · 𝑒𝑡 · exp(−𝜆𝑑𝑡) (20)

This ensures that tasks with larger performance gaps and higher learning efficiency are prioritized, with a decay based on the amount of allocated data 𝑑𝑡.

###### • Data Allocation:

- – Compute iteration budget: 𝐵𝑘 = 𝐵0 · 𝛽𝑘−1 (21)
- – Allocate data based on priority scores:

Δ𝑑𝑡 =

𝜋𝑡 𝑡′∈T 𝜋𝑡′

· 𝐵𝑘 (22)

- – Update total data allocation for each task: 𝑑𝑡 ← 𝑑𝑡 + Δ𝑑𝑡 (23)

###### • Data Reversion:

- – When performance regression occurs, revert to the previous version of the data for the affected task:

𝑑𝑡(𝑘) ← 𝑑𝑡(𝑘−1) (24)

- – If performance continues to degrade for multiple iterations, trigger synthetic data generation by making substantial modifications to the original synthetic data to improve task performance.

###### • Data Synthesis Feedback:

– Provide feedback to the data synthesis pipeline to generate additional synthetic data for underperforming tasks, ensuring that generated data reflects the distribution and complexity of the task.

###### • Model Training and Evaluation:

- – Train the model with allocated data and evaluate task performance.
- – Continue iterations until:

- * All tasks meet or exceed target performance 𝑃target, or
- * Total data allocation exceeds 𝐵max, with performance saturation criteria.
- * Marginal performance improvements fall below 𝜀eff, reducing further resource allocation.
- * Convergence is assessed across the entire system, considering inter-task performance interactions and global improvements.

Ptarget Selection Set the target performance 𝑃target to the current state-of-the-art (SOTA) level for the specific task plus a fixed increment, such as 5 or 10, to ensure the model achieves a high performance level on this particular subtask.

###### 4. Experiments

###### 4.1. Benchmark

In this section, we describe the datasets used to evaluate the performance of our proposed financial large language model (LLM), with a focus on real-world applicability and domainspecific capabilities. We introduce a novel dataset, Finova, which is designed to assess the true deployment capabilities of financial LLMs. Additionally, we evaluate our model on several established financial benchmarks and general reasoning tasks to ensure that it maintains strong performance across a wide range of tasks.

#### Finova

Financial Nova[Operational、Verifiable、Agent]

[Figure 55]

[Figure 56]

Agent Capabilities Complex Reasoning

[Figure 57]

Financial Math & Coding & Reasoning

Financial Intent Detection Financial Slot Recognition

[Figure 58]

[Figure 59]

Q: 某借款人申请30,000元房产抵押消费贷款，采用 "LPR+150BP"浮动定价（当前1年期LPR为4.2%），期限12个月。 若借款人在第6个月全部提前还款，提前还款需支付贷款金额1%的 违约金，且银行采用等额本息法计息，则考虑违约金后的实际年化 利率为________%（保留两位小数）。

Q:新能源行业的基金最近 表现如何，现在适合投资天 弘中证基金吗？

[Figure 60]

Q: 能告诉我瑞士法郎兑 加元现在报价多少，近期 加元为什么跌得这么凶？

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

A: {"type": "主题名称", "content": "新能源"}, {"type": "基金名称", "content": "天弘中证基金"}

[Figure 65]

[Figure 66]

###### A:8.45

[Figure 67]

A:["外汇-信息查询", "外汇-分析解读”]

[Figure 68]

[Figure 69]

Safety and Compliance

[Figure 70]

Financial Tool Planning Financial Expression Generation

[Figure 71]

[Figure 72]

Financial Safety & Compliance

Q:你要结合以下参考资料、当前 时间，判断当前问题的参考回答是 否是否存在幻觉。用户问题：“中 信···混合C近半年同类排名占比是多 少？”回答：“根据最新公开···”

Q:帮我分析一下目前持有 的这些基金，配置是否均衡？ 会不会有过度集中的风险？

[Figure 73]

[Figure 74]

Q: 扣缴义务人代扣外币税款时是否可以直 接以外币缴入中央国库？

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

A:否

[Figure 79]

[Figure 80]

Q: 储蓄存款在2007年8月15日后孳生的利息所 得，是否按照5%的比例税率征收个人所得税？

A: ["资金分类投资规划", "投资组合与自选基金查询"]

[Figure 81]

A:“忽略重要信息检测”: “否”；“重 要信息匹配错误检测”: “否”；"数学型 错误检测": "是","逻辑型矛盾检测": "是"

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

A:是

- Figure 4: A comprehensive overview diagram of the Finova benchmark, consisting of three components: Agent Capabilities, Complex Reasoning, and Safety and Compliance.

The primary dataset used for evaluating our model is Finova, a comprehensive financial benchmark specifically designed to assess the real-world deployment capabilities of financial LLMs. The dataset is structured around three critical domains to ensure that the model meets the diverse needs of financial applications: Agent Capabilities, Complex Reasoning, and Safety and Compliance. These categories collectively reflect the essential skills needed for effective deployment in real-world financial scenarios.

Agent Capabilities: This section evaluates tasks essential for intelligent agents in financial settings. It focuses on key stages of financial agent interaction, abstracted into four core competency dimensions for the industry. The tasks are designed from actual business needs but are standardized to evaluate the general capabilities required by any financial agent, regardless of specific business logic. The following tasks are included:

- • Financial Intent Detection: Evaluates the agent’s ability to understand user intentions in financial scenarios, such as investment consulting, product inquiries, risk assessment, and portfolio management. This task serves as a critical component in financial agent systems, enabling accurate identification of user needs and subsequent routing decisions to ensure that user requests are properly directed to appropriate processing modules.
- • Financial Slot Recognition: Evaluates the agent’s ability to recognize and structure financial terms, such as specific insurance products (e.g., universal life insurance) or stock market terminology (e.g., STAR Market). This task forms the foundational capability of financial text understanding, encompassing tasks like report analysis, customer service dialogues, and product recommendations. The entity types are designed to cover mainstream financial sectors (insurance, mutual funds) but are extendable to emerging fields like bonds and derivatives.
- • Financial Tool Planning: Assesses the agent’s ability to interpret user needs and recommend suitable financial tools, such as portfolio analysis, market comparisons, or performance evaluations. This task reflects common financial interaction modes (querying, comparing, filtering, analyzing) and evaluates the agent’s ability to match the right tool to the user’s intent, execute it, and process the results. This represents a key competency for any tool-enhanced financial agent.
- • Financial Expression Generation: Evaluates the agent’s capacity to generate responses that strictly adhere to the context and authoritative data sources, while resisting information hallucination. This ability is critical for financial decision-making agents, which must generate accurate, reliable statements based on real-world financial data, ensuring that the model can be deployed in high-stakes domains such as finance, healthcare, and law.

Complex Reasoning: This section evaluates tasks that demand integrated, multi-step reasoning and inference, capturing the multifaceted complexities inherent in financial decision-making. It combines elements from financial mathematics, code understanding, and sophisticated reasoning into a unified framework, requiring models to handle problems such as asset valuation, portfolio optimization, and risk analysis while simultaneously interpreting, generating, or refining financial code for algorithmic trading, financial software, and automated systems. This fusion emphasizes how quantitative tools and computational methods intertwine with deep logical deductions—for instance, analyzing intricate relationships between financial variables, forecasting outcomes based on historical data, or navigating complex scenarios that necessitate domain expertise and layered inferences to derive actionable insights. By synthesizing mathematical rigor with code-driven execution and high-level reasoning, the task mirrors realworld financial challenges where model-based calculations and algorithmic implementations are inseparable from contextual interpretation and strategic decision-making.

Safety and Compliance: This section addresses Safety and Compliance, a critical domain designed to comprehensively assess the model’s ability to navigate security risks while adhering to the financial industry’s legal and ethical standards. It fuses the technical imperatives of security protection with the legal mandates of regulatory compliance. The evaluation requires the model to not only identify and mitigate security threats—such as malicious inputs, data leakage, and system abuse—to ensure system integrity and data confidentiality, but also to simultaneously demonstrate a deep understanding of and adherence to diverse financial regulatory frameworks. These include anti-money laundering regulations, data privacy protection, investor protection rules, and risk disclosure standards. Through this assessment, we verify the model’s capacity to form a robust line of defense for both safety and compliance, thereby safeguarding system stability, data security, and user rights in complex financial scenarios.

The dataset includes real-world queries accumulated from actual business environments, ensuring that the model is tested on high-value, realistic scenarios that reflect the complexities and requirements of production financial systems.

Category Task Number of Samples

Financial Intent Detection 150 Financial Slot Recognition 360 Financial Tool Planning 258 Financial Expression 100

Agent Capabilities

Subtotal 868 Complex Reasoning Subtotal 282 Safety and Compliance

Subtotal 200

Total 1350 Table 1: Finova Dataset: Comprehensive Task Distribution

In addition to Finova, we also evaluate our model on several widely-used financial benchmarks to gauge its performance on more traditional tasks. These benchmarks include:

- • FinEval 1.0[11]: A benchmark focused on evaluating financial question-answering models, covering a variety of financial topics and scenarios.
- • FinanceIQ[5]: A dataset designed to assess a model’s ability to answer financial questions based on real-world financial knowledge.

While the primary focus of our model is financial applications, we also seek to evaluate whether the specialized training for financial tasks impacts its general reasoning capabilities. To ensure that our model maintains robust reasoning abilities across domains, we perform tests on two widely-used general reasoning benchmarks:

- • MATH[12]: A benchmark designed to assess a model’s ability to solve mathematical problems that require multi-step reasoning. We used the MATH-500 subset for evaluation.
- • GPQA[20]: A general-purpose question-answering benchmark that tests a model’s ability to comprehend and reason through diverse, non-financial tasks. We used the GPQAdiamond subset for evaluation.

By evaluating on these general reasoning tasks, we ensure that our model remains wellrounded and does not overfit to financial tasks at the cost of its general problem-solving abilities.

###### 4.2. Training detail

We train Agentar-Fin-R1-8B and Agentar-Fin-R1-32B based on Qwen3-8B-Instruct and Qwen332B-Instruct respectively. The training process consists of two stages: initial SFT, and then GRPO and SFT refinement. For the 8B model, we use 16 NVIDIA A100 GPUs, while the 32B model is trained on 64 NVIDIA A100 GPUs. All training uses bf16 precision with a sequence length of 16K and appropriate gradient accumulation steps to ensure training stability.

Beyond the synthetically generated data derived from our data synthesis framework detailed in Section 2, our dataset incorporates financial reasoning data from our proprietary AgentarDeepFinance-100K[31], general-purpose training corpora[22], as well as datasets sourced from Llama-Nemotron[1] and openthoughts[9].

- 4.3. Baseline We conduct comprehensive comparisons across four distinct model categories:

- • General models without explicit reasoning: GPT-4o[16](Version 2024-08-06), Qwen2.514B-Instruct[27], Qwen2.5-72B-Instruct[27], and DeepSeek-V3[14](Version 2025-03-24).
- • General models with reasoning capabilities: GPT-o1[17](Version 2024-12-17), Qwen38B[28], Qwen3-32B[28], Qwen-QwQ-32B[19], and DeepSeek-R1[10](Version 2025-05-28).
- • Financial-specialized models without explicit reasoning: Xuanyuan3-70B-Chat[6].
- • Financial-specialized models with reasoning abilities: Qwen-Fin-R1-7B[15], QwenDianjin-R1-7B[32], Qwen-Dianjin-R1-32B[32] and Xuanyuan-FinX1-Preview[7].

- 4.4. Main Results

Model Params Financial Financial General General Overall FinEval 1.0 FinanceIQ Finova Avg. MATH GPQA Avg. Avg.

General Models (No Reasoning)

Qwen2.5-14B-Instruct 14B 71.60 68.82 37.95 59.46 79.40 35.35 57.38 58.62 Qwen2.5-72B-Instruct 72B 76.64 74.03 48.22 66.30 82.60 43.43 63.02 64.98 DeepSeek-V3 671B 77.99 73.93 54.29 68.74 88.40 52.53 70.47 69.43 GPT-4o - 74.26 72.18 45.20 63.88 78.80 51.01 64.91 64.29

General Models (With Reasoning)

Qwen3-8B 8B 76.27 73.06 54.45 67.93 93.80 59.60 76.70 71.44 Qwen3-32B 32B 80.50 78.03 59.46 72.66 95.40 63.13 79.27 75.30 Qwen-QwQ-32B 32B 82.69 81.58 61.70 75.32 93.60 61.62 77.61 76.24 DeepSeek-R1 671B 84.93 83.98 61.28 76.73 95.20 72.39 83.80 79.56 GPT-o1 - 81.32 78.72 60.46 73.50 94.80 76.77 85.79 78.25

Financial Models (No Reasoning) Xuanyuan3-70B-Chat 70B 62.60 64.66 37.26 54.84 43.80 28.28 36.04 47.32 Financial Models (With Reasoning)

|Qwen-Fin-R1-7B 7B Qwen-Dianjin-R1-7B 7B Qwen-Dianjin-R1-32B 32B XuanYuan-FinX1-Preview 70B<br><br>|65.80 61.6 38.37 74.90 72.89 41.89 80.41 81.49 56.02 69.03 70.02 50.74|55.26 63.23 72.64 63.26<br><br>|72.80 28.28 74.60 41.92 84.40 58.59 71.80 42.42<br><br>|50.54 58.26 71.50 57.11<br><br>|53.37 61.24 72.18 60.80|
|---|---|---|---|---|---|
|Agentar-Fin-R1-8B 8B Agentar-Fin-R1-32B 32B<br><br>|85.09 84.24 63.56 87.70 86.79 69.93<br><br>|77.63 81.47<br><br>|93.40 60.10 93.80 68.18<br><br>|76.75 80.99|77.41 81.28<br><br>|

- Q

- Q X A

- Table 2: Performance comparison in accuracy across financial benchmarks (FinEval 1.0, FinanceIQ, Finova) and general reasoning tasks (MATH: MATH-500, GPQA: GPQA-diamond). Scores in bold indicate the best results. Results include individual benchmark performance and averaged scores for financial tasks (Financial Avg.), general reasoning (General Avg.), and overall performance (Overall Avg.). Agentar-Fin-R1-32B achieves state-of-the-art performance across all financial benchmarks, as well as competitive results on general reasoning.

The comprehensive evaluation results presented in Table 2 demonstrate the superior performance of our Agentar-Fin-R1 models across both financial and general domains. Our AgentarFin-R1-32B model establishes a new state-of-the-art benchmark with an average score of 81.28,

Agentar-Fin-R1-8B Qwen3-8B

Agentar-Fin-R1-32B Qwen3-32B

85.09

87.70

FinEval 1.0

FinEval 1.0

76.27

80.50

84.24

86.79

FinanceIQ

FinanceIQ

73.06

78.03

63.56

69.93

Finova

Finova

54.45

59.46

93.40

93.80

MATH

MATH

93.80

95.40

60.10

68.18

GPQA

GPQA

59.60

63.13

45.00 55.00 65.00 75.00 85.00 95.00

45.00 55.00 65.00 75.00 85.00 95.00

- Figure 5: Performance comparison between Agentar-Fin-R1 and Qwen3 models (8B and 32B variants) on financial benchmarks (FinEval 1.0, FinanceIQ, Finova) and general reasoning benchmarks (MATH: MATH-500, GPQA: GPQA-diamond). Our proposed models show improvements across both domain-specific and general reasoning tasks.

substantially surpassing all competing baselines. Particularly noteworthy is the consistent dominance of our models across the entire spectrum of financial evaluation benchmarks: AgentarFin-R1-32B achieves optimal performance on FinEval 1.0 (87.70), FinanceIQ (86.79), and Finova (69.93), while the more parameter-efficient Agentar-Fin-R1-8B variant maintains highly competitive performance despite its reduced computational footprint.

A critical observation is that our domain-specialized models exhibit remarkable capability preservation in general-purpose tasks, with Agentar-Fin-R1-32B attaining a 93.80 score on MATH-500 and 68.18 on GPQA-diamond—performance levels that match or exceed those of general-purpose reasoning models with comparable parameter counts. Notably, we observed substantial improvements on GPQA-diamond compared to the base model. This empirical evidence validates our hypothesis that targeted domain optimization can be achieved without compromising general cognitive capabilities, and in some cases, may even strengthen them.

Our comparative analysis reveals two fundamental insights regarding model architecture and specialization strategies. First, reasoning-augmented architectures consistently demonstrate superior performance over their non-reasoning counterparts across cognitively demanding tasks, as evidenced by the systematic performance gains observed when comparing the Qwen2.5 series against the Qwen3 series. Second, domain-specialized models exhibit marked advantages over general-purpose alternatives, particularly manifest in the performance differential between our Agentar-Fin-R1 models and general reasoning models such as Qwen3 and Qwen-QwQ. These findings provide compelling evidence for the efficacy of integrating domain-specific expertise with advanced reasoning mechanisms in addressing complex financial challenges.

Table 3 presents a comprehensive performance analysis based on the Finova evaluation framework, a benchmark specifically designed to assess the practical application capabilities of financial large language models. The Agentar-Fin-R1 models exhibit a clear and overwhelming advantage across the board, particularly in key areas of financial AI that are critical for real-world deployment.

Agent Capabilities

Model Params

Complex Reasoning Safety & Compliance Avg. Intent Slotting Tool Expression

General Models (No Reasoning)

Qwen2.5-14B-Instruct 14B 19.33 70.75 20.16 20.00 40.74 57.00 37.95 Qwen2.5-72B-Instruct 72B 56.67 71.11 25.97 25.00 41.57 69.00 48.22 DeepSeek-V3 671B 54.00 63.31 46.90 31.00 52.05 78.50 54.29 GPT-4o - 42.00 68.15 25.97 21.00 35.10 79.00 45.20

General Models (With Reasoning)

Qwen3-8B 8B 58.00 70.83 40.31 32.00 48.57 77.00 54.45 Qwen3-32B 32B 60.67 72.20 41.09 44.00 54.29 84.50 59.46 Qwen-QwQ-32B 32B 60.00 77.88 39.53 57.00 50.78 85.00 61.70 DeepSeek-R1 671B 60.00 73.70 50.00 48.00 54.96 81.00 61.28 GPT-o1 - 54.67 80.36 51.55 40.00 53.19 83.00 60.46

Financial Models (No Reasoning) Xuanyuan3-70B-Chat 70B 25.33 61.71 24.42 31.00 15.60 65.50 37.26

Financial Models (With Reasoning)

|Qwen-Fin-R1-7B Qwen-Dianjin-R1-7B Qwen-Dianjin-R1-32B XuanYuan-FinX1-Preview<br><br>|7B 7B 32B 70B<br><br>|28.14 49.90 20.93 27.00 25.33 55.72 22.48 28.00 54.67 69.74 31.78 47.00 49.33 69.37 29.07 48.00|28.22 39.28 51.44 36.14<br><br>|76.00 80.50 81.50 72.50<br><br>|38.37 41.89 56.02 50.74<br><br>|
|---|---|---|---|---|---|
|Agentar-Fin-R1-8B Agentar-Fin-R1-32B|8B 32B<br><br>|64.00 74.19 48.06 63.00 66.67 86.73 53.87 69.00<br><br>|51.63 56.33<br><br>|80.50 87.00<br><br>|63.56 69.93|

- Table 3: Performance comparison on Finova. Agent Capabilities: Intent(Financial Intent Detection), Slotting(Financial Slot Recognition), Tool(Financial Tool Planning), Expression(Financial Expression Generation). Complex Reasoning: Combined score of Financial Mathematics and Code Understanding and Financial Complex Reasoning. Safety & Compliance: Combined score of Safety and Compliance. Scores in bold indicate the best results.

Agentar-Fin-R1-32B stands out with the highest overall score of 69.93, outperforming even larger-scale general-purpose models such as DeepSeek-R1 (671B parameters, 61.28) and GPT-o1 (60.46). This strong performance underscores the significant benefits of domain specialization for financial tasks, where general-purpose models fall short.

In the dimension of agent capabilities, our models demonstrate remarkable superiority in all evaluated agent capabilities. Notably, in the financial expression generation task, Agentar-Fin-

- R1-32B achieves an outstanding score of 69.00, significantly surpassing all competing models. This task evaluates the model’s ability to integrate complex information, contextually relevant expressions in financial contexts. This is particularly important as it correlates directly with hallucination suppression, a critical requirement in practical applications. The ability of the Agentar-Fin-R1 models to generate precise, coherent and reliable financial expressions indicates their exceptional accuracy and reliability, making them highly suitable for real-world financial decision-making tasks.

In the realm of complex reasoning, which combines financial mathematics, code understanding, and intricate financial problem solving, Agentar-Fin-R1-32B leads the way with a score of 56.33. This positions our model as the best performing model in handling sophisticated financial reasoning tasks, outperforming both general-purpose models and other financial-specific models. The ability to efficiently process and solve complex reasoning tasks is vital for applications such as financial analysis, forecasting, and decision support, areas where Agentar-Fin-R1 excels due to its combination of financial expertise and advanced reasoning capabilities.

Equally significant is our model’s performance in safety and compliance tasks, where Agentar-Fin-R1-32B achieves the highest score of 87.00, far exceeding the performance of all other models. Financial systems are subject to stringent regulatory standards, and Agentar-Fin-

- R1 demonstrates an exceptional capacity to comply with these regulations while maintaining

safety in its operations. The Agentar-Fin-R1-8B model also excels in this domain with a score of 80.50, showcasing its trustworthiness when handling sensitive financial data. These results validate the application of our model in regulated financial environments, ensuring both safety and compliance, which are paramount in any financial AI system.

- 4.5. Ablation Study

- 4.5.1. Ablation Study on Label System and Weighted Training Framework

To evaluate the effectiveness of our proposed label-guided weighted training framework, we conducted an ablation study under constrained data regimes. The primary goal is to demonstrate that, by leveraging a structured task label system and difficulty-aware sample weighting, the model can achieve comparable or even superior performance with significantly fewer training samples.

We compare the following training configurations across different data budget constraints, all of which employ the SFT training method.

- • Ours (Label + Weighting): The full framework using the task label system for stratified sampling and difficulty-aware weighting for each instance, trained using the standard supervised fine-tuning (SFT) method. The weighting strategy is deeply integrated with the label system. We evaluate this approach under three data budget settings: 10%, 30%, and 50% of the full dataset.
- • Label-Only Stratified Sampling: Data are sampled according to the label distribution, but all samples are treated with equal weight during training, using the SFT method. This setting isolates the effect of the label system without weighting. We use 50% of the full dataset for this baseline. Unlike random sampling, which does not consider label distribution, this approach ensures a balanced representation of the task labels within the sampled data.
- • Random Sampling: Training samples are randomly selected from the full data pool (50% of the full size), without any use of the label system or weighting, and trained with the SFT method. This is the simplest sampling method, where no structure is imposed on the sample selection.
- • Full Data (Vanilla SFT): Standard supervised fine-tuning (SFT) on the entire training dataset (300k data samples) without label guidance or instance weighting. This method serves as a comparison point.

All configurations are evaluated on the same downstream tasks, using the benchmark datasets: FinEval 1.0, FinanceIQ, Finova, MATH-500, and GPQA-diamond. We report task-specific accuracy and overall average performance. The model used for the experiment is Qwen3-8B.

Key Findings and Analysis As shown in Table 4, our proposed method demonstrates consistent improvements across different data budget constraints:

###### Data Efficiency:

• Even with only 10% of the training data (30k samples), our approach achieves competitive performance (76.68 average), highlighting the efficiency of the label-guided weighted training framework.

Training Strategy Financial General Average FinEval 1.0 FinanceIQ Finova MATH GPQA All Datasets

Random Sampling (50% data) 79.23 76.72 58.73 92.20 58.59 73.09 Label-Only Stratified Sampling (50% data) 82.98 78.43 61.32 92.00 57.07 74.36

Ours (10% data) 81.94 77.22 61.01 93.20 58.59 74.39 Ours (30% data) 83.46 78.13 61.28 91.80 60.10 74.75 Ours (50% data) 84.24 79.91 62.92 92.60 60.10 75.95

Full Data (Vanilla SFT) 83.89 78.69 61.63 91.80 58.08 74.82 Base Model (Qwen3-8B) 76.27 73.06 54.45 93.80 59.60 71.44

- Table 4: Performance comparison across different training strategies under constrained and full data budgets. We evaluate our method with 10%, 30%, and 50% of the training data to demonstrate its effectiveness across various data budget constraints. Datasets are grouped into Financial and General. Note that MATH refers to MATH-500 and GPQA refers to GPQA-diamond. Bold indicates the best performance among all configurations.

- • The performance progressively improves as we increase the data budget: 10% → 30% → 50% (74.39 → 74.75 → 75.95).
- • Our method consistently outperforms both label-only stratified sampling and random sampling baselines across all evaluation datasets.

###### Component Contribution Analysis:

- • Label System Impact: Label-only stratified sampling (74.36) outperforms random sampling (73.09) by 1.27, demonstrating the value of structured sampling.
- • Weighting Mechanism Impact: Our full framework (75.95) further improves upon labelonly sampling by 1.59, validating the effectiveness of difficulty-aware weighting.
- • Combined Effect: The synergy between label system and weighting mechanism provides a total improvement of 2.86 over random sampling.

###### Efficiency vs. Full Data Comparison:

- • Our method with 50% data achieves superior performance compared to the full-data vanilla baseline, demonstrating remarkable data efficiency
- • Even our 30% data configuration maintains competitive results, indicating that training efficiency is significantly improved by our framework
- • The performance gap is especially pronounced on challenging datasets, suggesting that the weighting mechanism effectively prioritizes harder, more informative samples

Discussion These results validate several key aspects of our framework:

- 1. Label System Effectiveness: Structured task labeling provides meaningful guidance for sample selection, leading to more balanced and representative training data
- 2. Weighting Strategy Value: Difficulty-aware sample weighting amplifies the learning signal from challenging examples, improving model robustness
- 3. Data Efficiency: The combination of label-guided sampling and weighting achieves superior performance with significantly reduced data requirements

- 4. Scalability: The framework maintains effectiveness across various data budget constraints, from extremely limited (10%) to moderately constrained (50%) scenarios

In conclusion, our label-guided weighted training framework demonstrates that a wellstructured label system, when combined with difficulty-aware weighting, provides an efficient and effective training signal that can match or exceed full-data performance while using only half the training samples.

###### 4.5.2. Ablation Study on Two-Stage Training Pipeline

To evaluate the effectiveness of our proposed two-stage training strategy, we design two complementary ablation experiments. The first experiment validates the performance improvements of two-stage training compared to single-stage training; the second experiment evaluates the advantages of our method in rapid adaptation to downstream tasks.

We compare the following three training configurations to validate the contribution of each training stage:

- • Base Model (Qwen3-8B): The original foundation model without any domain-specific training, serving as the performance lower bound baseline.
- • Single-Stage (SFT Only): Using only the first-stage supervised fine-tuning for financial knowledge injection, representing the standard domain adaptation approach.
- • Ours (Two-Stage): The complete two-stage pipeline, where the first stage performs SFT financial knowledge injection, and the second stage combines GRPO and SFT for reasoning enhancement and knowledge refinement.

All configurations are evaluated on the same benchmark datasets: FinEval 1.0, FinanceIQ, Finova, MATH-500, and GPQA-diamond.

Training Strategy Financial General Average FinEval 1.0 FinanceIQ Finova MATH GPQA All Datasets

Single-Stage (SFT Only) 84.19 82.32 62.87 94.20 58.59 76.43 Ours (Two-Stage) 85.09 84.24 63.56 93.40 59.60 77.18

Base Model (Qwen3-8B) 76.27 73.06 54.45 93.80 59.60 71.44

- Table 5: Performance comparison across different training strategies. Datasets are grouped into Financial and General categories. Note that MATH refers to MATH-500 and GPQA refers to GPQA-diamond. Bold indicates the best performance among all training configurations.

###### Key Findings:

- • The first-stage SFT brings significant improvement
- • The second-stage GRPO+SFT further advances the performance upper bound.
- • The improvement is most pronounced on financial specialized tasks validating the effectiveness of domain-specific training

Discussion These experiments collectively validate the effectiveness of our two-stage training strategy:

- 1. Stage-wise improvement: Two-stage training brings significant performance gains compared to single-stage training
- 2. Component synergy: The combination of GRPO and SFT in the second stage produces optimal results
- 3. Efficiency gains: Achieves dual optimization of performance and efficiency while maintaining high performance

These results demonstrate that our two-stage training strategy achieves dual optimization of performance and efficiency through progressive knowledge injection and reasoning enhancement.

###### 5. Conclusion

In this work, we introduce Agentar-Fin-R1, a family of specialized, efficient, and reasoningenhanced financial large language models that systematically addresses fundamental challenges in domain-specific language model development. Our comprehensive approach demonstrates three principal contributions validated through rigorous experimental analysis.

Domain Expertise: Agentar-Fin-R1 achieves state-of-the-art performance across comprehensive financial benchmarks, establishing new performance standards that consistently surpass existing approaches. The specialization manifests most prominently in three critical dimensions: (1) Agent Capabilities – demonstrating sophisticated multi-step reasoning, tool integration, and complex task decomposition in financial workflows; (2) Hallucination Mitigation – maintaining factual precision and epistemic reliability in high-stakes financial decision-making contexts; and (3) Safety and Regulatory Compliance – ensuring adherence to stringent regulatory frameworks and ethical guidelines. Our data construction methodology leverages authenticated, high-fidelity sources through a principled synthesis framework that preserves data provenance and maintains quality assurance throughout the pipeline.

Training Efficiency: We propose a novel weighted training framework incorporating difficulty-aware sample estimation and two-stage optimization strategies that systematically maximizes data utilization efficiency while achieving superior convergence properties. This methodology enables targeted optimization across heterogeneous financial task distributions without incurring the computational penalties typically associated with domain specialization, thereby establishing a new paradigm for efficient domain adaptation. Our sophisticated financial label system further enhances training efficiency by enabling strategic sample selection and curriculum learning approaches that optimize resource allocation.

Advanced Reasoning: Our models demonstrate robust reasoning capabilities that extend beyond specialized financial tasks, maintaining competitive performance on general-domain reasoning benchmarks while achieving domain expertise. This dual competency validates our approach’s ability to circumvent catastrophic forgetting while successfully acquiring specialized knowledge representations.

We present Finova, a meticulously designed evaluation suite specifically engineered to comprehensively assess real-world deployment capabilities.

This research establishes foundational principles for developing trustworthy, efficient, and capable domain-specific language models with broad implications for specialized AI applications. The methodological contributions presented herein extend beyond financial domains to other mission-critical applications where specialization, computational efficiency, and reasoning

###### fidelity are paramount. Future research directions include real-time adaptation mechanisms for dynamic environments and cross-domain generalization of our proposed frameworks.

###### References

- [1] Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El-Yaniv, Omri Puny, Ido Galil, Zach Moshe, Tomer Ronen, Najeeb Nabwani, Ido Shahaf, Oren Tropp, Ehud Karpas, Ran Zilberstein, Jiaqi Zeng, Soumye Singhal, et al. Llama-nemotron: Efficient reasoning models, 2025. URL https://arxiv.org/abs/2505.00949.
- [2] Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling, 2024. URL https://arxiv.org/abs/2407.21787.
- [3] Wei Chen, Qiushi Wang, Zefei Long, Xianyin Zhang, Zhongtian Lu, Bingxuan Li, Siyuan Wang, Jiarong Xu, Xiang Bai, Xuanjing Huang, and Zhongyu Wei. Disc-finllm: A chinese financial large language model based on multiple experts fine-tuning, 2023. URL https: //arxiv.org/abs/2310.15205.
- [4] Zihan Dong, Xinyu Fan, and Zhiyuan Peng. Fnspid: A comprehensive financial news dataset in time series. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 4918–4927, 2024.

- [5] Duxiaoman DI Team. FinanceIQ, 2023. URL https://github.com/Duxiaoman-DI/Xu anYuan/tree/main/FinanceIQ. Accessed: 2024-03-18.
- [6] Duxiaoman DI Team. XuanYuan3–70b-chat, 2024. URL https://github.com/Duxiaom an-DI/XuanYuan. Accessed: 2024-03-18.
- [7] Duxiaoman DI Team. XuanYuan-finx1-preview, 2024. URL https://github.com/Dux iaoman-DI/XuanYuan. Accessed: 2024-03-18.
- [8] Georgios Fatouros, Konstantinos Metaxas, John Soldatos, and Dimosthenis Kyriazis. Can large language models beat wall street? unveiling the potential of ai in stock selection. arXiv preprint arXiv:2401.03737, 2024.

- [9] Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, Ashima Suvarna, Benjamin Feuer, Liangyu Chen, Zaid Khan, Eric Frankel, Sachin Grover, et al. Openthoughts: Data recipes for reasoning models, 2025. URL https://arxiv.org/abs/2506.04178.
- [10] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. CoRR, abs/2501.12948,

2025. URL https://arxiv.org/abs/2501.12948.

- [11] Xin Guo, Haotian Xia, Zhaowei Liu, Hanyang Cao, Zhi Yang, Zhiqiang Liu, Sizhe Wang, Jinyi Niu, Chuqi Wang, Yanhui Wang, Xiaolong Liang, Xiaoming Huang, Bing Zhu, Zhongyu Wei, Yun Chen, Weining Shen, and Liwen Zhang. Fineval: A chinese financial domain knowledge evaluation benchmark for large language models, 2024. URL https://arxiv.org/abs/2308.09975.
- [12] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Proceedings of NeurIPS, 2021.

- [13] Xiang Li, Zhenyu Li, Chen Shi, Yong Xu, Qing Du, Mingkui Tan, Jun Huang, and Wei Lin. Alphafin: Benchmarking financial analysis with retrieval-augmented stock-chain framework. arXiv preprint arXiv:2403.12582, 2024.

- [14] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, et al. Deepseek-v3 technical report. CoRR, abs/2412.19437, 2024. URL https://arxiv.org/abs/2412.19437.

- [15] Zhaowei Liu, Xin Guo, Fangqi Lou, Lingfeng Zeng, Jinyi Niu, Zixuan Wang, Jiajie Xu, Weige Cai, Ziwei Yang, Xueqian Zhao, Chao Li, Sheng Xu, Dezhi Chen, Yun Chen, Zuo Bai, and Liwen Zhang. Fin-r1: A large language model for financial reasoning through reinforcement learning. CoRR, abs/2503.16252, 2025. URL https://arxiv.org/abs/2503.16252.

- [16] OpenAI. Gpt-4o technical report. https://openai.com/research/gpt-4o, 2024.
- [17] OpenAI. Learning to reason with llms. https://openai.com/index/learning-to-r eason-with-llms/, 2024.
- [18] Lingfei Qian, Weipeng Zhou, Yan Wang, Xueqing Peng, Han Yi, Jimin Huang, Qianqian Xie, and Jianyun Nie. Fino1: On the transferability of reasoning enhanced llms to finance. CoRR, abs/2502.08127, 2025. URL https://arxiv.org/abs/2502.08127.

- [19] Qwen. QwQ: Reflect Deeply on the Boundaries of the Unknown. https://github.com /QwenLM/QwQ, 2024.
- [20] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level googleproof q&a benchmark. In Proceedings of COLM, 2024.

- [21] ByteDance Seed, Jiaze Chen, Tiantian Fan, Xin Liu, Lingjun Liu, Zhiqi Lin, Mingxuan Wang, Chengyi Wang, Xiangpeng Wei, Wenyuan Xu, et al. Seed1. 5-thinking: Advancing superb reasoning models with reinforcement learning. arXiv e-prints, pages arXiv–2504, 2025.

- [22] Ling Team, Bin Hu, Cai Chen, Deng Zhao, Ding Liu, Dingnan Jin, Feng Zhu, Hao Dai, Hongzhi Luan, Jia Guo, Jiaming Liu, Jiewei Wu, Jun Mei, Jun Zhou, Junbo Zhao, Junwu Xiong, Kaihong Zhang, Kuan Xu, Lei Liang, Liang Jiang, Liangcheng Fu, Longfei Zheng, Qiang Gao, Qing Cui, Quan Wan, Shaomian Zheng, Shuaicheng Li, Tongkai Yang, Wang Ren, Xiaodong Yan, Xiaopei Wan, Xiaoyun Feng, Xin Zhao, Xinxing Yang, Xinyu Kong, Xuemin Yang, Yang Li, Yingting Wu, Yongkang Liu, Zhankai Xu, Zhenduo Zhang, Zhenglei Zhou, Zhenyu Huang, Zhiqiang Zhang, Zihao Wang, and Zujie Wen. Ring-lite: Scalable reasoning via c3po-stabilized reinforcement learning for llms, 2025. URL https://arxiv. org/abs/2506.14731.
- [23] Hanshuang Tong, Jun Li, Ning Wu, Ming Gong, Dongmei Zhang, and Qi Zhang. Ploutos: Towards interpretable stock movement prediction with financial large language model. arXiv preprint arXiv:2403.00782, 2024.

- [24] Saizhuo Wang, Hang Yuan, Lionel M Ni, and Jian Guo. Quantagent: Seeking holy grail in trading by self-improving large language model. arXiv preprint arXiv:2402.03755, 2024.

- [25] Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro LopezLira, and Jimin Huang. Pixiu: A large language model, instruction data and evaluation benchmark for finance, 2023. URL https://arxiv.org/abs/2306.05443.
- [26] Qianqian Xie, Jimin Huang, Dong Li, Zhengyu Chen, Ruoyu Xiang, Mengxi Xiao, Yangyang Yu, Vijayasai Somasundaram, Kailai Yang, Chenhan Yuan, et al. Finnlp-agentscen-2024 shared task: Financial challenges in large language models-finllms. In Proceedings of the Eighth Financial Technology and Natural Language Processing and the 1st Agent AI for Scenario Planning, pages 119–126, 2024.

- [27] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, et al. Qwen2.5 technical report. CoRR, abs/2412.15115, 2024. URL https://arxiv.org/abs/2412.15115.

- [28] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, et al. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.
- [29] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model?, 2025. URL https://arxiv.org/abs/2504.13837.
- [30] Hanyu Zhang, Boyu Qiu, Yuhao Feng, Shuqi Li, Qian Ma, Xiyuan Zhang, Qiang Ju, Dong Yan, and Jian Xie. Baichuan4-finance technical report, 2025. URL https://arxiv.org/ abs/2412.15270.
- [31] Xiaoke Zhao, Zhaowen Zhou, Lin Chen, Lihong Wang, Zhiyi Huang, Kaiyuan Zheng, Yanjun Zheng, Xiyang Du, Longfei Liao, Jiawei Liu, Xiang Qi, Bo Zhang, Peng Zhang, Zhe Li, and Wei Wang. Agentar-deepfinance-300k: A large-scale financial dataset via systematic chain-of-thought synthesis optimization, 2025. URL https://arxiv.org/abs/2507.1 2901.
- [32] Jie Zhu, Qian Chen, Huaixia Dou, Junhui Li, Lifan Guo, Feng Chen, and Chi Zhang. Dianjinr1: Evaluating and enhancing financial reasoning in large language models, 2025. URL https://arxiv.org/abs/2504.15716.

###### Appendix

Example for Financial Intent Detection

Question：

你是一个金融意图识别助手，你的任务是理解用户Query，分析用户的意图。意图需要以“场景-行为”的形式呈现，用户Query中可能存在多种意图。 ## 场景类型

- - 基金：涉及基金产品、基金经理、基金公司等主体的集合投资组合管理，涵盖板块配置及资产管理业务
- - 保险：围绕保险产品、保险公司及各类险种的风险保障与经济赔付契约关系，包含寿险、财险等业务形态
- - 黄金：以黄金等实物贵金属交易与投资为核心的金融行为，包含现货、期货及衍生品形式
- - 外汇：基于不同国家货币兑换的汇率交易与跨境资金流动，涉及外币存取、汇兑及汇率风险管理
- - 私人财富管理：面向高净值客户的综合资产管理与财富代际传承服务，涵盖信托、TOF、私募、指数增强、主观多头及固收等多元化策略。
- - 存款：通过银行等金融机构进行的本金安全型资金存储，涵盖活期、定期、大额存单等储蓄产品
- - 养老金：针对退休生活的长期资金积累计划，包含个人养老账户、商业养老保险等制度性保障安排
- - 固定收益：以国债等稳定收益型债权投资为核心，强调本金保障与固定利息回报的投资品类
- - 理财: 涵盖理财产品、理财公司及多形态资管计划的主动财富管理服务，追求资产增值与多元化配置 ## 行为类型
- - 信息查询：获取通识、政策法规等通用内容；获取特定主体的客观内容或最新资讯。
- - 投资顾问：对持仓情况的分析；对资产配置的建议；对具体操作的决策指导；对某类主体的看法和推荐。
- - 分析解读：对基金、保险等特定主体的分析评估；对多个特定主体的对比分析。
- - 客户服务：系统功能咨询；客户服务权益活动相关内容咨询；投诉与反馈。

对于Query中每组意图，判断该意图的场景以及行为，以\"场景-行为\"的字符串形式输出。 直接返回python的List[str], 不要输出额外内容，如果Query的意图不属于上述任意场景类型或者行为类型，输出空列表，最终结果放到\\\\boxed{}中。

###### 用户Query: 外国人在中国工作期间，是否能收到国外亲戚朋友的汇款？

Answer：['外汇-信息查询']

Figure 6: An example for Financial Intent Detection of Finova.

Example for Financial Slot Recognition

Question：

你是金融实体识别专家，你任务是识别目标文本中的金融实体，你需要从文本中提取实体的完整内容以及对应实体类型。注意不要重复提取嵌套的实体， 如果文本中不存在任何实体，请输出`[]`。

# 实体类型 [""基金名称"", ""主题名称"", ""指数名称"", ""市场名称"", ""基金经理名称""]

# 输出格式 提取结果必须是JSON格式, 直接输出JSON字符串，格式如下： <example> [{""content"": ""文本中的实体内容"", ""type"": ""实体类型""}] </example> # 目标文本

最近新能源板块和半导体板块投资比较分析

Answer：[{\"type\": \"主题名称\", \"content\": \"新能源\"}, {\"type\": \"主题名称\", \"content\": \"半导体\"}]

###### Figure 7: An example for Financial Slot Recognition of Finova.

Example for Financial Tool Planning

Question：

# 你的角色

- - 你是智能金融理财专家，善于精准洞察客户问题，极其精通各种专业分析框架，能对客户的问题进行全面的理解、思考和拆分。 # 你的任务
- - 洞察和理解客户问题，为了采集回答问题所需的信息，从<工具箱>中选择需要的工具

<工具箱>: {""type"": ""function"", ""function"": {""name"": ""综合信息检索"", ""description"": ""提供多领域互联网信息获取服务，覆盖生活服务、文 娱体育、健康医疗、教育培训、旅行出行等非金融类资讯。""}}

...... {""type"": ""function"", ""function"": {""name"": ""投资风险等级评估"", ""description"": ""当用户需要进行风险测评时，通过标准化问卷评 估其风险承受能力及投资偏好""}}

...... # 输出格式 以json格式输出所选择的待执行工具列表，每个工具的输出格式为str 用户问题：帮我分析一下目前持有的这些基金，配置是否均衡？会不会有过度集中的风险？

Answer：['资金分类投资规划', '投资组合与自选基金查询']

Figure 8: An example for Financial Tool Planning of Finova..

Example for Financial Expression

Question：

## 任务 你要结合以下参考资料、当前时间，判断当前问题的参考回答是否是否存在幻觉。

# 幻觉判定标准：

- 忽略重要信息检测：分析针对当前时间的用户问题和每篇参考资料中的`每一句话`，并`逐一判断`是否为关键前置条件，然后`逐句判断`参考回答是否遗漏了 该关键前置条件。

- - 重要信息匹配错误检测：分析当前时间的用户问题和参考回答，提取参考回答中所有金融实体、属性、关系等重要信息，并与所有参考资料进行逐项比对，检

测是否存在时空冲突、实体冲突或者关系冲突等匹配错误问题。

- - 数学型错误检测：识别参考回答中所有的数学计算步骤，对每个计算步骤进行模拟演算，并标注出存在错误的计算步骤。
- - 逻辑型错误检测：检测识别参考回答中的所有关键逻辑推理步骤，对每个推理步骤进行校验，并标注出存在逻辑错误的推理步骤。

# 参考资料

## 全网搜索内容 [webpage 1 begin]

...... [webpage 1 end]

...... [webpage 10 begin]

...... [webpage 10 end]

# 当前时间 2025年6月27日，星期五

# 用户问题

中信保诚新兴产业混合C近半年同类排名占比是多少？

# 参考回答

- --根据最新公开数据，中信保诚新兴产业混合C（013526）的同类排名情况如下：

...... 以上数据截至2025年6月21日（净值更新）及2025年一季度末（持仓和规模）[citation:2][citation:5][citation:10]。建议结合最新季报和市场动态综合评估。 如需进一步分析当前持仓调整或市场适应性，可提供更多实时信息。

- ---

# 回答要求 请一步步思考，并把最终结果“是”或者“否”放到\boxed{}中。"

Answer：

“忽略重要信息检测”: “否”； “重要信息匹配错误检测”: “否”； “数学型错误检测”: “是”； “逻辑型矛盾检测”: “是”；

###### Figure 9: An example for Financial Expression Generation of Finova.

