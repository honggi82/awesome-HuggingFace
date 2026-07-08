# arXiv:2508.20374v1[cs.AI]28Aug2025

## TCIA: A Task-Centric Instruction Augmentation Method for Instruction Finetuning

### Simin Ma 1,*,†, Shujian Liu 1,*, Jun Tan 1,*, Yebowen Hu 1, Song Wang 1, Sathish Reddy Indurthi 1, Sanqiang Zhao 1, Liwei Wu 1, Jianbing Han 1, Kaiqiang Song 1

1Zoom Communications Inc. *Equal contribution. †Corresponding author: simin.ma@zoom.us

##### Abstract

Diverse instruction data is vital for effective instruction tuning of large language models, as it enables the model to generalize across different types of inputs . Building such diversified instruction dataset is an essential step in this process. Existing approaches often leverage large language models to automatically explore and generate diverse instructions, ensuring both data diversity and quality. However, they tend to overlook an important factor in real-world applications: on-task relevance. In practice, only a few real-world applications require a truly general-purpose model; most benefit from task-specific knowledge tailored to their particular use case. Therefore, it is vital to develop instruction augmentation methods that not only maintain diversity but are also optimized for specific, real-world scenarios.

We thus introduce Task Centric Instruction Augmentation (TCIA), a framework that systematically expands instructions while preserving both diversity and task alignment. By representing instructions in a discrete query-constraints space, TCIA creates a rich set of task-relevant instructions and enables models to generalize to these task-specific instructions without sacrificing overall performance. Experiments show that TCIA improves open-source LLMs’ performance by an average of 8.7% across four real-world, task-specific applications, and in some cases outperforming leading closedsource models. These improvements do not compromise general instruction-following ability, making TCIA a scalable and efficient solution for adapting LLMs to real-world, taskfocused applications.

### Introduction

LLMs have transformed NLP, but closed-source models are costly and lack task-specific expertise (Brown et al. 2020; Ouyang et al. 2022; Achiam et al. 2023). Fine-tuning opensource models using user-defined instructions offers a practical and cost-effective alternative. However, reliably steering these models toward specific, task-centric instructions remains a challenge (Xu et al. 2024; Bao et al. 2023).

Early supervised fine-tuning (SFT) relies on manually crafted task-specific instructions, but this approach is resource-intensive and typically yields instruction datasets with limited diversity (Wei et al. 2021; Sanh et al. 2021; K¨opf et al. 2023). Consequently, LLMs trained on such data

Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

TCIA

Frequency

WizardLM

0.5 0.6 0.7 0.8 0.9

Diversity

(a) Hop 1

TCIA

Frequency

WizardLM

0.5 0.6 0.7 0.8 0.9

Diversity

(c) Hop 3

TCIA

Frequency

WizardLM

0.5 0.6 0.7 0.8 0.9

Diversity

(b) Hop 2

- 0.5
- 1

| | | | | |
|---|---|---|---|---|
| | | | | |
| ||TCIA<br><br>WizardLM|
|---|
| | | |
| | | | | |

Ratio

0

1 2 3

Hop Number

(d) On-Task Ratios

Figure 1: Comparison of average diversity (sub-figure (a)(c)) and average on-task ratio (sub-figure (d)) across our four in-house real-world tasks between TCIA and WizardLM for hops 1-3. For sub-figure (a)-(c), Diversity = 1 − cosine-similarity between embeddings, and means are marked by the dotted lines.

often struggle to generalize, particularly in novel or challenging scenarios (Chiang et al. 2023; Wang et al. 2022a; Honovich et al. 2022; Taori et al. 2023a; Chung et al. 2024).

Recent research has shifted towards automatic or humanin-the-loop instruction generation, using LLMs to rapidly expand training data and reduce reliance on manual annotation (Chung et al. 2024; Wang et al. 2022b; Xu et al. 2024). These methods typically begin with a small set of humanauthored “seed” instructions and use large models to iteratively generate additional instructions, often across multiple “hops”, significantly increasing the dataset size. However, this automation has two key drawbacks: the generated instructions often become repetitive and lack diversity, and they frequently suffer from task drift. Task drift occurs when instructions become less relevant to the target task and introduce irrelevant or artificial requirements. As a result, finetuned models may underperform on nuanced or specialized tasks, and risk amplifying initial data biases (Huang et al.

[Figure 1]

[Figure 2]

Figure 2: Demonstration of TCIA framework, which is composed of six key steps: (1) Instruction State Decomposition, (2) Instruction Database Construction & Retrieval, (3) Breadth-First Search (BFS) for Instruction Augmentation, (4) Conversion Back to Natural Language Instruction, (5) Instruction Validation & Context Integration, (6) Response Generation & Data Quality Filtering (Sampling). The final generated high-quality <instruction, response> pairs are used for downstream SFT.

2022; Perez et al. 2022).

To address these shortcomings, we introduce TCIA (TaskCentric Instruction Augmentation), a new framework that systematically augments instructions while explicitly maintaining both diversity and task relevance. TCIA expands the instruction set by leveraging both retrieved, task-aligned instructions and high-quality, automatically generated variants, thereby maintaining diversity while ensuring close alignment with the target task. This combination broadens the instruction space and enables the generation process to remain substantially grounded and on-task, even through multiple augmentation hops.

Empirical analysis demonstrates that TCIA addresses the key shortcomings of recent automatic instruction generation frameworks. First, in terms of diversity, Figure 1a–1c shows that TCIA sustains a high degree of instruction variety across multiple augmentation steps, while recent frameworks such as like WizardLM (Xu et al. 2024), quickly converge to repetitive templates. Second, in terms of constraint fidelity, Figure 1d and Table 2 demonstrate that TCIA consistently produces instructions that remain task-relevant even as complexity and the number of hops increase. In contrast, previous frameworks often produce repetitive or off-task outputs, making them less suitable for nuanced or evolving instructions. This strong balance of diversity and constraint fidelity improves downstream performance, allowing TCIA-trained models to adapt flexibly to novel or complex instructions—crucial for real-world applications.

We evaluate the performance of TCIA across both opensource benchmarks and challenging, task-specific in-house scenarios related to online meeting. TCIA consistently out-

performs existing instruction-tuning baselines and even surpasses advanced closed-source models such as GPT-4o on several specialized tasks. Notably, TCIA also retains competitive scores on general-purpose benchmarks, highlighting its ability to balance task-specific adaptation with broad instruction-following ability. Below are our main contributions:

- • We introduce the first task-centric instruction augmentation framework, which unifies task-centric retrieval, systematic state exploration, and advanced constraint augmentation, enabling models to flexibly adapt to diverse real-world tasks and scenarios.
- • The proposed framework (aka TCIA) boosts the instruction diversity, ensuring that outputs remain task-relevant, even after multiple hops, enabling models to flexibly follow complex constraints.
- • The model trained on TCIA delivers state-of-the-art results on several task-specific benchmarks (8.7% average gain), surpassing closed-source models, while maintaining strong general performance.

Collectively, these findings establish TCIA as a powerful, efficient, and general-purpose tool to maximize the real-world utility of open-source language models.

### Related Works

Early work on instruction tuning aligned LLMs to NLP tasks through supervised fine-tuning (SFT) with manually crafted instructions, which required costly annotation and limited generalizability to complex instructions or novel tasks (Raffel et al. 2020; Khashabi et al. 2020; Brown et al. 2020; Wei

###### Original Instruction I Base Query Q and Constraints {Ci}

Write a travel ad for a trip to the Adirondack mountains. Focus on activities there, the scenery, and keep it concise and under 200 words.

Q: Write a travel ad for a trip to the Adirondack mountains C1: Must include activities and scenery of the Adirondack mountains C2: Must be concise C3: Must be under 200 words

For each following Query, return a series of clear, factual and concise Summary Bullet Points of the Query using only information available in the Query. Query: {placeholder} Summary:

Q: Return a summary of the given query C1: Must be presented in bullet points C2: Must only use information available in the query C3: Must be clear, factual and concise

In this task, you are given two strings A, B. Find the longest common substring in the strings A and B.

Q: Find the longest common substring in the given strings A and B.

Table 1: Examples of instruction decomposition via decomposition prompt in Table 5 (in the Appendix).

et al. 2021; K¨opf et al. 2023). To address scalability, recent methods such as Self-Instruct and Alpaca employ LLMs to generate synthetic instruction, enabling rapid data expansion but often resulting in repetitive, less nuanced, or off-task instructions over multiple generations (Wang et al. 2022a; Taori et al. 2023b; Chen et al. 2024; Perez et al. 2022; Tan et al. 2024). In pursuit of more relevant and realistic instructions, newer approaches integrate structured templates, curriculum learning, and retrieval-augmented generation, yet maintaining both diversity and task-relevance over many augmentation steps remains challenging, especially for specialized tasks (Xu et al. 2020, 2024; K¨opf et al. 2023; Chen et al. 2024; Peng et al. 2023). More recently, instruction augmentation has been framed as structured state space exploration, leveraging techniques such as breadth-first search, discrete mutation, and incremental constraint addition to improve coverage, though these do not always guarantee sustained diversity and task-centricity at scale (Chai, Xie, and Qin 2025; Ye et al. 2023; Sun et al. 2023).

### Method

This section detailly outlines the TCIA framework in six key steps as illustrated in Figure 2.

#### Instruction State Decomposition

We begin by representing each instruction in a structured format, decomposing a natural language instruction I into a base query Q and a set of n constraints C = {Ci}ni=1, with both components kept in natural language (Xu et al. 2024; Sun et al. 2024; Qin et al. 2024; An et al. 2025). For example, as demonstrated in Table 1, a complex instruction is split into its core directive and distinct, explicit constraints. This decomposition provides several key advantages: (1) it enhances human and machine interpretability, as the breakdown clarifies which specifications must be met and facilitates both review and debugging; (2) it makes it possible to quantitatively compare instructions, supporting precise diversity measurement by treating instructions as points in a discrete space; and (3) it increases controllability during augmentation, enabling targeted, principled modifications and systematic exploration via BFS. This state-based representation is foundational for building a meaningful, diverse

set of prompts later in the pipeline.

To enable this decomposition, we employ a single LLM prompt that simultaneously extracts the task type T, the base query Q, and a set of categorized constraints C for each instruction (see Table 5 in the Appendix for the detailed prompt). Extracting the task type is crucial for our diversification procedure: it allows us to identify related tasks and sample compatible constraints for the “Add” and “Replace” steps in our BFS, resulting in more context-aware augmentation. By leveraging a generative LLM approach, we achieve greater flexibility and precision in task type identification than is possible with rigid classification methods, accommodating a wide range of instruction types. The explicit categorization of constraints further supports structured clustering, sampling, and targeted manipulation. Overall, this unified decomposition yields an interpretable and controllable state representation, forming a robust foundation for systematic instruction augmentation in our pipeline.

#### Instruction Database (DB) Construction

Before BFS-based augmentation, we construct a large and diverse instruction constraint pool to enable efficient augmentation operations for any instruction state. We leverage Tulu-3 (Lambert et al. 2024), a broad-coverage, carefully curated public dataset that includes instructions spanning knowledge recall, reasoning, mathematics, coding, general chat, and more. For each datapoint, we apply our unified task identification and decomposition prompt (as described above), which may yield multiple triplets (Tj,Qj,Cj) if several distinct tasks are present in the same instruction, where Tj,Qj,Cj are the task types, base queries, and the sets of explicit constraints. This produces a task-organized, richly diversified instruction database optimized for retrieval and constraint transfer.

To retrieve similar task types or constraints during augmentation, we employ the “sentence-transformers/allmpnet-base-v2” embedding model1 for semantic retrieval, ensuring sampled constraints are both contextually appropriate and restricted to matching task types. Organizing instructions by task type and incorporating semantic retrieval

1https://huggingface.co/sentence-transformers/all-mpnet-base-v2

offers several benefits: (1) Contextual Relevance: constraints are always sampled within the same task domain, preserving task consistency and specialized instruction fidelity; (2) Improved Adaptability: constraint selection enables adaptive augmentation within domains (e.g., professional writing), while preventing irrelevant mixing from unrelated tasks; and (3) Enhanced Specialization: enabling nuanced transfer and generalization across related tasks, facilitating specialized behavior learning, and maximizing diversity using a semantically structured constraint set.

We further conduct statistical analysis on our instruction database constructed from the Tulu-3 Dataset. Table 18 presents an example datapoint decomposed into its base query, task type, and associated constraints. Table 19 summarizes the top 20 extracted task types (clustered via DBSCAN (Ester et al. 1996)); while mathematical analysis and programming tasks predominate, a significant number of relevant tasks to our in-house tasks (e.g., Q&A, explanation, text generation) are present. Table 6 illustrates the distribution of query-to-constraint ratios and Table 20 shows the ten most common constraints for the “mathematical probability and statistical analysis” task type, indicating sufficient constraint diversity per query. Overall, this approach enables robust, context-aware augmentation for instruction rewriting and expansion within the BFS pipeline.

#### Breadth-First Search (BFS) for Instruction Augmentation

Algorithm 1: Diversified Constraint Generation via BFS (TCIA)

Data: Q: Query, C: Initial constraints, T: Task type, D: instruction DB containing {task, constraints} pairs, K: # output constraints limit, m: # operation repeat, k: sample size

Result: k diversified constraint sets

- 1 Initialize queue with (Q, C); Call ← [ ];
- 2 while queue not empty and |queue| < K do

- 3 Dequeue (Q, C);
- 4 (T,˜ C˜) ← most similar in D to T;
- 5 foreach op ∈ {Add, Remove, Replace} do

- 6 for i = 1 to m do

- 7 Cj ← rand in C;
- 8 if op=Remove then

- 9 C′ ← C \ {Cj};
- 10 else if op=Add then

- 11 Ci′ ← rand in C˜;
- 12 C′ ← C ∪ {Ci′};
- 13 end
- 14 else

- 15 C˜j ← most similar in C˜ to Cj;
- 16 C′ ← (C \ {Cj}) ∪ {C˜j};
- 17 end
- 18 Enqueue (Q, C′), append C′ to Call;
- 19 end
- 20 end
- 21 end
- 22 Return k random samples from Call

With each instruction decomposed and a task-organized instruction database constructed, we diversify prompts using a BFS algorithm (shown in Algorithm 1) that systematically explores the space of constraint combinations while preserving task context. Each BFS state consists of a base query Q and a set of constraints C for a given task type T. Starting from the original decomposed instruction, the BFS algorithm iteratively generates new candidate states via three operations: (1) Add: Randomly add a constraint sampled from a similar task type in the instruction DB; (2) Remove: Randomly delete a constraint; 3) Replace: Substitute a randomly chosen constraint with a similar constraint from another instruction in the instruction DB. For both Add and Replace, the retrieval of similar task types and constraints uses the embedding model described in the previous subsection. Each operation is performed m times per state. All candidate states are enqueued for further exploration, and their constraint sets are collected in Call. The search continues until the queue is empty or a limit of K unique constraint sets is reached. After BFS completes, k diverse constraint sets are randomly sampled from Call as output. By guiding augmentation with embedding-based retrieval from semantically similar instructions and constraints, our approach balances high diversity with strict task fidelity and contextual grounding throughout.

#### Convert Back to Natural Language Instructions

After structured augmentation, each instruction state is converted back to a complete natural language prompt using an LLM, with iterative critique and refinement to guarantee all constraints are included and correctly translated (or if max number of retries is reached). Specifically, a critique prompt checks for missing or mistranslated constraints, and if necessary, a refinement prompt guides regeneration. This ensures the completeness and integrity in the natural language instructions. All prompts are in Table 6 - 7 in the Appendix.

#### Instruction Validation

To ensure task feasibility and data quality, synthesized instructions undergo thorough LLM-based validation, scored on two aspects (rated 1-5): (1) validity: verifies relevance and absence of constraint violations, and (2) selfconsistency: ensuring no logical contradictions or ambiguities. Instructions not meeting thresholds are discarded, leaving only robust prompts. Detailed scoring prompts are in Tables 9 and 10 in the Appendix. Validated instructions are then paired with real-world context (e.g., document excerpts or conversation snippets).

#### Data Quality Filtering

Lastly, we employ state-of-the-art LLMs to generate diverse and high-quality responses for each instruction-context pair. Then, each instruction and candidate response pair will undergo a rigorous quality control stage using an LLM-as-ajudge. Each response is evaluated across five key dimensions inspired by (Lambert et al. 2024): general quality, helpfulness, instruction following, uncertainty, and truthfulness, each rated on a 1–5 scale. This multi-dimensional assessment ensures that the selected data is not only accurate and

informative, but is also reliable and well-aligned with the prompt’s intent. For each instruction, only the response with the highest average score across all criteria is retained, guaranteeing that only the most useful and robust examples are included for SFT, leading to a high-quality task-optimized SFT dataset ready for training robust real-world models. Detailed prompts are in Table 11 - 15 in the Appendix.

### Experiments

We rigorously evaluate the effectiveness of TCIA at (1) the instruction prompt level, where we diagnose diversity and task relevance of generated instructions, and (2) the model level, where we measure downstream model robustness, task adaptation, and generalization after supervised fine-tuning (SFT). Collectively, these experiments demonstrate both the direct impact of TCIA’s instruction generation method and its practical downstream value.

#### Instruction Diversity and Task Fidelity

|Hop|TCIA|WizardLM<br><br>|
|---|---|---|
|1|Must include topics and descriptions.<br><br>|Highlight Potential Risks:<br><br>- Identify and briefly explain any potential risks or challenges mentioned.|
|1<br><br>|Must focus on data engineering and analyst roles.<br><br>|Identify Key Metrics:<br><br>- Extract and highlight any quantitative data or performance indicators mentioned during the meeting.|
|2|Must include at least two direct quotations from the interviewee under ‘key quotes’.<br><br>|Add a “Key Metrics” section highlighting quantitative data or performance indicators, ensuring all numerical values are accurately represented.|
|2<br><br>|Construct the summary in an innovative fashion.<br><br>|Key Metrics:<br><br>- Highlight and summarize any quantitative information or metrics.<br>- Present data points in a clear, concise manner to support decision-making.<br>|
|3<br><br>|Must maintain a professional tone.|KPIs:<br><br>- Identify and explain any mentioned KPIs or relevant metrics.|
|3|Output should be structured as a summary of user’s criteria and information requests.|Identify Key Metrics and Performance Indicators:<br><br>- Extract and highlight any mentioned KPIs, targets, or quantitative goals.<br>- Include these metrics to provide context for decision-making.<br>|

- Table 2: Example instruction constraints generated by TCIA vs WizardLM at Hops 1, 2, and 3, for in-house Task B. The generation processes’ seed prompt is in Table 16 in the Appendix, which is a meeting summarization task prompt similar to our in-house Task B.

We begin by analyzing the effectiveness of TCIA at the instruction prompt level by comparing with WizardLM, showcasing TCIA’s ability to simultaneously address two major pitfalls of prior methods, such as WizardLM: (1) rapid diversity collapse and (2) Task drift. Note that these are the

same instruction pools later used for SFT, and the generation details are explained in subsection below.

First, WizardLM and similar baselines tend to generate increasingly repetitive and formulaic instruction patterns after a few augmentation hops, diminishing both instructional diversity and model robustness. Diversity is computed as one minus the pairwise cosine-similarity between instruction prompt embeddings (obtained using “text-embedding3-large” (OpenAI 2025b)). In contrast to WizardLM, TCIA’s explicit task-type conditioning and constraint retrieval sustain diverse, meaningful instruction variants across multiple hops, as shown in Figure 1a - 1c (average across all four inhouse tasks). A closer look at individual tasks in Figure 4 (in Appendix) reveals that while diversity distributions of WizardLM and TCIA overlap closely at hop 1, TCIA already achieves a slightly higher diversity density (TCIA’s mean ≈ 0.85 versus WizardLM’s mean ≈ 0.79). As augmentation progresses, TCIA largely retains its diversity, shifting only slightly to a mean of 0.8 at hop 3. In contrast, WizardLM’s diversity distribution progressively collapses, with the mean dropping sharply (falling below 0.65 for tasks B and D at hop 3), indicating substantial loss of variability and increased template repetition.

Second, WizardLM often suffers from task drift, producing instructions that stray from the intended task or lose alignment with key requirements. This effect is quantitatively measured in Figure 1d (average on-task ratio over tasks) and is further detailed by task in Figure 5 (in Appendix). TCIA consistently preserves task fidelity through all augmentation hops, maintaining an on-task ratio close to 100% even at later hops. In contrast, WizardLM’s task alignment erodes rapidly: it starts at around 80% at hop 1, drops to roughly 60% by hop 2, and falls below 60% at hop 3, with the lowest on-task ratio dipping to 40% for taskC. Concrete examples in Table 2 further illustrate this drift. After three hops, TCIA generates precise and targeted constraints, such as “Must include at least two direct quotations from the interviewee under key quotes”, ensuring outputs remain directly relevant to the original meeting summarization objectives. On the other hand, WizardLM instructions often shift focus almost exclusively to “Key Metrics” or “KPIs”, for instance “Key Metrics: Highlight and summarize any quantitative information or metrics ...”, which not only signals a repetitive pattern but also demonstrates clear drift from the specialized requirements of the seed task.

#### Supervised Fine-Tuning (SFT) Setup and Baselines

We next assess how the different instruction generation methods propagate into downstream model performance. All SFT experiments are conducted using Llama-3.1-8B as the base model, fine-tuned with identical hyperparameters (5 × 10−6 learning rate, batch size 2, 1 node of 8×H100 GPUs, 1 epoch), and evaluated on identical data splits for fair comparison. We use Nemo-Aligner for all SFT training (Shen et al. 2024).

We consider three SFT variants: (1) TCIA, which systematically generates diverse, constraint-aware instructions for each task from one seed prompt per task by BFS exploration over 3 hops (2k unique instructions per task), augments each

with 5 task-specific inputs (10k instruction–input pairs per task), generates LLM outputs from a diverse model pool (“claude-3-5-sonnet-2024102” (Anthropic 2024), “claude3-5-sonnet-20240620” (Anthropic 2024), “gpt-4o-2024-0806” (OpenAI 2024), and “gpt-4.1-2025-04-14” (OpenAI 2025a)), filters with our data filtering system (see Section “Data Quality Filtering”), and yields 10k high-quality examples per task (more generation details are in Appendix Table 17, as well as a toy example in Figure 7); (2) Fixed Instruction (FI), a standard low-resource baseline using just one fixed instruction prompt (same as the seed prompt for TCIA) per task and otherwise identical generation and filtering; and (3) WizardLM (Xu et al. 2024), which performs automated instruction augmentation from a single seed prompt, also exploring 3 hops to yield 2k instructions per task, substituting 5 context inputs per instruction, and applying the same generation and filtering process for 10k examples per task. All SFT datasets are further augmented with Tulu-3 data.

To contextualize SFT performance, we compare to strong open- and closed-source instruction-following LLMs: (1) GPT-4o (OpenAI 2024), OpenAI’s one of the latest flagship; (2) Llama-3.1-8B-Instruct (Llama-8B) (Grattafiori et al. 2024), Meta’s RLHF-aligned for broad utility; and (3) Llama-3.1-Tulu-3-8B-SFT (Tulu-8B-SFT) (Lambert et al. 2024), AllenAI’s fine-tuned on varied instructions. All evaluations focus on four proprietary meeting AI tasks (Task AD)2, which represent challenging, real-world summarization and information extraction scenarios within workplace environments.

#### Robustness to Evolving Instruction Constraints

|Instruction Constraint|FI-8B WizardLM-8B TCIA-8B<br><br>|
|---|---|
|Output in a numbered list<br><br>|0.0% 98.4% 99.2%|
|Output no more than 5 bullet points|29.4% 61.2% 87.6%<br><br>|
|Sort the output by entity-specific groupings|42.6% 64.9% 82.7%|

- Table 3: Strict constraint adherence (“pass rate”) of outputs from FI-8B, WizardLM-8B, and TCIA-8B on Task A. “Pass rate” is the proportion of outputs that fully satisfy the constraint, which is tested one at a time (no multiple constraints). “Pass rate” is checked solely for compliance not content quality.

Before reporting overall model accuracy, we first evaluate the models’ robustness to evolving and unseen user constraints, focusing on strict instruction adherence for a representative in-house task (Task A). In real-world production settings, constraints frequently change, i.e. users may require formats to be switched, output lengths limited, or information reorganized on demand. Models trained on fixed or insufficiently diverse instructions often fail to generalize, resulting in unreliable outputs and increased manual intervention. To directly assess adaptability, we evaluate the

2We cannot disclose data or prompts related to the in-house task A, B, C, and D due to company policy.

strict constraint adherence (“pass rate”) of FI, WizardLM, and TCIA models under new constraints not seen in training, such as switching from bullet points to numbered lists, enforcing maximum output lengths, or requiring alternative groupings. Table 3 presents the detailed results, where TCIA-trained models achieve consistently higher pass rates than both WizardLM and the single-instruct baseline across all constraint types. TCIA excels at adapting to new formats (e.g., numbered lists) and stricter requirements (e.g., response limits, different orderings). In contrast, FI models almost always fail to generalize, and WizardLM frequently ignores new constraints or strays off-task. These results confirm that TCIA’s instruction-level diversity significantly enhances robust generalization, providing practical benefits for real-world AI deployments requiring rapid task adaptation.

#### End-to-End In-house Task Evaluations

We then benchmark SFT model performance on the four proprietary, production-oriented tasks to assess overall endto-end effectiveness in real-world applications. These tasks represent challenging, high-impact applications in realworld deployment, requiring nuanced comprehension and the ability to adapt to varying requirements and edge cases. We employ both reference-based and reference-free evaluation frameworks, tailored to each task. For reference-based evaluation, ground-truth labels are generated by aggregation and majority voting from 3–4 independent human annotations per sample, ensuring a rigorous reference-based gold standard. For reference-free evaluation, which are less amenable to precise labeling, model outputs are evaluated using variants of the LLM-as-a-Judge protocol. All evaluation scores are normalized to a 0-100 scale for comparability.

Figure 3 shows that TCIA achieves decisive gains across all four in-house tasks. On average, TCIA delivers an 8.7% improvement over FI (Fixed Instruct) and a 3% improvement over WizardLM. Specifically, TCIA surpasses FI by 9.2% on Task A, 10.9% on Task B, 8.1% on Task C, and 3.1% on Task D. Compared to WizardLM, TCIA provides a further 2.9%, 3.9%, 4%, and 1.9% gain on Tasks A, B, C, and D, respectively. Notably, TCIA not only outperforms FI and WizardLM but also exceeds the strong GPT-4o baseline on all tasks, outperforming it by 2.7% on average, with especially substantial improvements on Task A (+2.67%) and Task B (+3.00%). Furthermore, all on-task fine-tuned models (FI-8B, WizardLM-8B, TCIA-8B) significantly outperform generic open-source instruction-tuned models such as Llama-8B and Tulu-8B-SFT on every in-house task (with average gains over Llama-8B ranging from 1.8% to 7.9% and over Tulu-8B-SFT from 5.5% to 11.6%). This underscores the high level of task specificity and challenge present in the in-house tasks, which general-purpose instructiontuned or RLHF models cannot address adequately.

WizardLM, while improving over FI, still falls short of TCIA, showing that inadequate diversity and frequent task drift in its instruction generation ultimately limit its task performance. In contrast, TCIA’s systematic, diverse, and taskspecific augmentation consistently enables models to excel on real-world, nuanced, and edge-case scenarios, where pre-

Task A

Task B

Task C

Task D

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| |58.5<br><br>60.2| | | | | | |
| |57.5<br><br>50.5 49.7<br><br>55.1| | | | | | |
| | | | | | | | |
| | | | | | | | |

| |80.5| | | | | | |
|---|---|---|---|---|---|---|---|
| |77.5<br><br>74.6<br><br>69.4<br><br>72.6<br><br>77.4| | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| |82.4<br><br>78.4<br><br>82.4| | | | | | |
|---|---|---|---|---|---|---|---|
| |72.4<br><br>67.8<br><br>74.3| | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| |78.0 79.3| | | | | | |
|---|---|---|---|---|---|---|---|
| |73.2<br><br>69.0<br><br>76.1 77.4| | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

80

70

60

50

40

|GPT-4o Llama-8B Tulu-8B-SFT FI-8B WizardLM-8B TCIA-8B|
|---|

Figure 3: Performance of models on our four internal tasks (scores are average of 3 runs).

cise task understanding is vital. Moreover TCIA enables LLMs to acquire a deeper, more generalized understanding of tasks that transfers well even as requirements or phrasings shift. This adaptability is crucial for practical deployment, where rapid iteration and evolving user needs are the norm.

as, and in several cases surpasses, FI and WizardLM (e.g., achieving a +1.5 gain on Info-Bench and +2.0 on MMLUP compared to WizardLM). TCIA also matches or exceeds the performance of Tulu-8B-SFT on almost all public benchmarks (e.g., 81.26 vs. 80.28 on Info-Bench and 34.99 vs. 34.57 on MMLU-Pro). These results indicate that the skills reinforced through TCIA’s task-centric instruction augmentation are transferable, and the high degree of specialization does not come at the expense of broader generalization.

#### Generalization on Public LLM Benchmarks

Model IFEval InFoB. GPQA BBH MMLUP. Avg. GPT-4o 82.44 92.83 46.70 81.40 74.66 75.61 Llama-8B 76.34 89.38 30.40 41.12 36.10 54.67 Tulu-8B-SFT 72.80 80.28 29.85 42.44 34.57 51.99 FI-8B 67.84 79.91 27.47 41.68 33.98 50.17 WizardLM-8B 72.46 79.54 32.23 41.39 34.33 51.99 TCIA-8B 68.95 81.26 29.49 41.98 34.99 51.33

RLHF-aligned models like Llama-8B and GPT-4o achieve higher absolute scores on public benchmarks, which is expected given their large-scale optimization for generalpurpose use. Despite this, TCIA distinguishes itself among SFT approaches by delivering substantial improvements on specialized tasks while preserving strong general capabilities. Training with TCIA’s structured, task-oriented data not only boosts real-world, task-specific performance but also ensures competitiveness on open-domain benchmarks. This balance makes TCIA a highly practical and effective solution for organizations seeking both advanced task specialization and reliable general-purpose language modeling.

- Table 4: Performance on different general benchmarks, including IFEval (2023), Info-Bench (InFoB.) (2024), GPQA

(2024), BBH (2022), and MMLU-Pro (MMLUP.) (2024).

Finally, we examine whether extensive task adaptation impacts model generality by evaluating all models on diverse, standard public LLM benchmarks. To assess whether TCIA’s gains in task-specific performance come at the expense of general instruction-following ability, we evaluate all models on widely adopted public LLM benchmark suites. These benchmarks include academic, linguistic, and reasoning tasks, providing a rigorous test of generalization and broad language capability. We use the OpenLLM Leaderboard from HuggingFace(Beeching et al. 2023), which features MMLU-Pro (academic/postgraduate knowledge and reasoning, denote as MMLUP)(Wang et al. 2024), IF-Eval (robustness to diverse instruction following)(Zhou et al. 2023), GPQA (general-purpose QA across knowledge domains)(Rein et al. 2024), and BBH (complex linguistic and logical reasoning)(Suzgun et al. 2022). We supplement with Info-Bench (Qin et al. 2024), which offers a comprehensive assessment of linguistic and logical reasoning.

### Conclusion

In this paper, we introduced TCIA, a well-structured framework for instruction augmentation that breaks down natural language instructions into understandable base queries and clearly defined, categorized constraints. By utilizing a semantically organized instruction database and a BFSdriven diversification process guided by contextual similarity, TCIA effectively generates diverse, high-quality instruction variations while maintaining task relevance and intent. Our approach has demonstrated strong performance across a range of in-house tasks focused on instruction-following and constraint adherence, showcasing both its flexibility and generalizability in real-world applications.

Looking ahead, there are several exciting avenues for exploration. Enhancing TCIA to incorporate richer contexts from multi-turn interactions or dialogue-based tasks could expand its applicability. Additionally, automating prompt refinement, especially for ambiguous or poorly specified instructions, could enhance robustness. Further opportunities lie in extending TCIA to multimodal scenarios or exploring more advanced retrieval strategies, paving the way for future research and development.

Table 4 summarizes the benchmark performance of all evaluated models. TCIA matches the average benchmark score of FI (50.17 vs. 51.33), providing clear evidence that extensive task-specific adaptation in TCIA does not degrade general instruction-following capability. Examining individual benchmarks, TCIA consistently performs at least as well

### References

Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023. GPT-4 technical report (2023). arXiv preprint arXiv:2303.08774.

An, K.; Sheng, L.; Cui, G.; Si, S.; Ding, N.; Cheng, Y.; and Chang, B. 2025. UltraIF: Advancing Instruction Following from the Wild. arXiv preprint arXiv:2502.04153.

Anthropic. 2024. Introducing Claude 3.5 Sonnet. https: //www.anthropic.com/news/claude-3-5-sonnet. Accessed: 2024-06-20.

Bao, K.; Zhang, J.; Zhang, Y.; Wang, W.; Feng, F.; and He,

- X. 2023. Tallrec: An effective and efficient tuning framework to align large language model with recommendation. In Proceedings of the 17th ACM Conference on Recommender Systems, 1007–1014.

Beeching, E.; Fourrier, C.; Habib, N.; Han, S.; Lambert, N.; Rajani, N.; Sanseviero, O.; Tunstall, L.; and Wolf, T. 2023. Open llm leaderboard.

Brown, T.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J. D.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877– 1901.

Chai, Y.; Xie, H.; and Qin, J. S. 2025. Text Data Augmentation for Large Language Models: A Comprehensive Survey of Methods, Challenges, and Opportunities. arXiv preprint arXiv:2501.18845.

Chen, L.; Li, J.; Dong, X.; Zhang, P.; He, C.; Wang, J.; Zhao, F.; and Lin, D. 2024. Sharegpt4v: Improving large multimodal models with better captions. In European Conference on Computer Vision, 370–387. Springer.

Chiang, W.-L.; Li, Z.; Lin, Z.; Sheng, Y.; Wu, Z.; Zhang, H.; Zheng, L.; Zhuang, S.; Zhuang, Y.; Gonzalez, J. E.; et al.

- 2023. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3): 6.

Chung, H. W.; Hou, L.; Longpre, S.; Zoph, B.; Tay, Y.; Fedus, W.; Li, Y.; Wang, X.; Dehghani, M.; Brahma, S.; et al.

- 2024. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70): 1–53.

Ester, M.; Kriegel, H.-P.; Sander, J.; Xu, X.; et al. 1996. A density-based algorithm for discovering clusters in large spatial databases with noise. In kdd, volume 96, 226–231.

Grattafiori, A.; Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.; Vaughan, A.; et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Honovich, O.; Shaham, U.; Bowman, S. R.; and Levy, O. 2022. Instruction induction: From few examples to natural language task descriptions. arXiv preprint arXiv:2205.10782.

Huang, J.; Gu, S. S.; Hou, L.; Wu, Y.; Wang, X.; Yu, H.; and Han, J. 2022. Large language models can self-improve. arXiv preprint arXiv:2210.11610.

Khashabi, D.; Min, S.; Khot, T.; Sabharwal, A.; Tafjord, O.; Clark, P.; and Hajishirzi, H. 2020. Unifiedqa: Crossing format boundaries with a single qa system. arXiv preprint arXiv:2005.00700.

K¨opf, A.; Kilcher, Y.; Von R¨utte, D.; Anagnostidis, S.; Tam, Z. R.; Stevens, K.; Barhoum, A.; Nguyen, D.; Stanley, O.; Nagyfi, R.; et al. 2023. Openassistant conversationsdemocratizing large language model alignment. Advances in Neural Information Processing Systems, 36: 47669–47681.

Lambert, N.; Morrison, J.; Pyatkin, V.; Huang, S.; Ivison, H.; Brahman, F.; Miranda, L. J. V.; Liu, A.; Dziri, N.; Lyu, S.; et al. 2024. T\” ulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124.

- OpenAI. 2024. Hello GPT-4o. https://openai.com/index/ hello-gpt-4o/. Accessed: 2024-05-31.
- OpenAI. 2025a. Introducing GPT-4.1 in the API. https: //openai.com/index/gpt-4-1/. Accessed: 2025-04-14. OpenAI. 2025b. New embedding models and API updates. https://openai.com/index/new-embedding-modelsand-api-updates/. Accessed: 2025-05-31. Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright, C.; Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray, A.; et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35: 27730–27744. Peng, B.; Li, C.; He, P.; Galley, M.; and Gao, J.

- 2023. Instruction tuning with gpt-4. arXiv preprint arXiv:2304.03277. Perez, E.; Huang, S.; Song, F.; Cai, T.; Ring, R.; Aslanides, J.; Glaese, A.; McAleese, N.; and Irving, G. 2022. Red teaming language models with language models. arXiv preprint arXiv:2202.03286. Qin, Y.; Song, K.; Hu, Y.; Yao, W.; Cho, S.; Wang, X.; Wu, X.; Liu, F.; Liu, P.; and Yu, D. 2024. Infobench: Evaluating instruction following ability in large language models. arXiv preprint arXiv:2401.03601. Raffel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.; Matena, M.; Zhou, Y.; Li, W.; and Liu, P. J. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140): 1–67. Rein, D.; Hou, B. L.; Stickland, A. C.; Petty, J.; Pang, R. Y.; Dirani, J.; Michael, J.; and Bowman, S. R. 2024. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling. Sanh, V.; Webson, A.; Raffel, C.; Bach, S. H.; Sutawika, L.; Alyafeai, Z.; Chaffin, A.; Stiegler, A.; Scao, T. L.; Raja, A.; et al. 2021. Multitask prompted training enables zero-shot task generalization. arXiv preprint arXiv:2110.08207. Shen, G.; Wang, Z.; Delalleau, O.; Zeng, J.; Dong, Y.; Egert, D.; Sun, S.; Zhang, J.; Jain, S.; Taghibakhshi, A.; et al.
- 2024. Nemo-aligner: Scalable toolkit for efficient model alignment. arXiv preprint arXiv:2405.01481. Sun, H.; Liu, L.; Li, J.; Wang, F.; Dong, B.; Lin, R.; and Huang, R. 2024. Conifer: Improving complex constrained instruction-following ability of large language models. arXiv preprint arXiv:2404.02823.

Sun, Z.; Shen, Y.; Zhou, Q.; Zhang, H.; Chen, Z.; Cox, D.; Yang, Y.; and Gan, C. 2023. Principle-driven self-alignment of language models from scratch with minimal human supervision. Advances in Neural Information Processing Systems, 36: 2511–2565.

Suzgun, M.; Scales, N.; Sch¨arli, N.; Gehrmann, S.; Tay,

- Y.; Chung, H. W.; Chowdhery, A.; Le, Q. V.; Chi, E. H.; Zhou, D.; et al. 2022. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261.

- Tan, Z.; Li, D.; Wang, S.; Beigi, A.; Jiang, B.; Bhattacharjee, A.; Karami, M.; Li, J.; Cheng, L.; and Liu, H. 2024. Large language models for data annotation and synthesis: A survey. arXiv preprint arXiv:2402.13446. Taori, R.; Gulrajani, I.; Zhang, T.; Dubois, Y.; Li, X.;

- Guestrin, C.; Liang, P.; and Hashimoto, T. B. 2023a. Alpaca: A strong, replicable instruction-following model. Stanford Center for Research on Foundation Models. https://crfm. stanford. edu/2023/03/13/alpaca. html, 3(6): 7. Taori, R.; Gulrajani, I.; Zhang, T.; Dubois, Y.; Li, X.;
- Guestrin, C.; Liang, P.; and Hashimoto, T. B. 2023b. Stanford alpaca: An instruction-following llama model.

Wang, Y.; Kordi, Y.; Mishra, S.; Liu, A.; Smith, N. A.; Khashabi, D.; and Hajishirzi, H. 2022a. Self-instruct: Aligning language models with self-generated instructions. arXiv preprint arXiv:2212.10560.

Wang, Y.; Ma, X.; Zhang, G.; Ni, Y.; Chandra, A.; Guo, S.; Ren, W.; Arulraj, A.; He, X.; Jiang, Z.; et al. 2024. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Wang, Y.; Mishra, S.; Alipoormolabashi, P.; Kordi, Y.; Mirzaei, A.; Arunkumar, A.; Ashok, A.; Dhanasekaran,

- A. S.; Naik, A.; Stap, D.; et al. 2022b. Supernaturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks. arXiv preprint arXiv:2204.07705. Wei, J.; Bosma, M.; Zhao, V. Y.; Guu, K.; Yu, A. W.; Lester,
- B.; Du, N.; Dai, A. M.; and Le, Q. V. 2021. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652.

- Xu, B.; Zhang, L.; Mao, Z.; Wang, Q.; Xie, H.; and Zhang, Y. 2020. Curriculum learning for natural language understanding. In Proceedings of the 58th annual meeting of the association for computational linguistics, 6095–6104.
- Xu, C.; Sun, Q.; Zheng, K.; Geng, X.; Zhao, P.; Feng, J.;

- Tao, C.; Lin, Q.; and Jiang, D. 2024. WizardLM: Empowering large pre-trained language models to follow complex instructions. In The Twelfth International Conference on Learning Representations.

Ye, S.; Kim, D.; Kim, S.; Hwang, H.; Kim, S.; Jo, Y.; Thorne, J.; Kim, J.; and Seo, M. 2023. Flask: Fine-grained language model evaluation based on alignment skill sets. arXiv preprint arXiv:2307.10928.

Zhou, J.; Lu, T.; Mishra, S.; Brahma, S.; Basu, S.; Luan, Y.; Zhou, D.; and Hou, L. 2023. Instruction-following

evaluation for large language models. arXiv preprint arXiv:2311.07911.

### Prompts

#### Instruction Decomposition Prompt

We use the following prompt to decompose an original natural language instruction (I) into a base queries (Q), a set of constraints (C), and identify its task types T, i.e. decomposed into {T(i),Q(i),C(i)}ni=1, where T(i),Q(i) and is the ith task type and base query, and C(i) = {Cj(i)}n in-house tasks, a given seed prompt will only decompose into {T,Q,C}, i.e. one task, one query, and a constraints set triplet, since our in-house task’s seed prompt is clearly defined as only one task type. However, for Tulu-3 dataset, multiple triples (i.e. multiple task types) are allowed.

- i
- j=1 is the ith set of ni constraints. Note that when generating the SFT training dataset for our

|Decompose Prompt<br><br>You are a precise assistant for parsing and analyzing user instructions. Given any input query, your task is to extract all structured information as follows:<br><br>Definitions:<br><br>- A "constraint" is any explicit restriction, condition, or requirement imposed by the query (about format, style, content, length, etc.).<br>- The "Basic Query" is the core task or request with all such constraints removed|stating only the essential goal.<br>- A "Task Type" is the general category or nature of the task being requested (e.g., summarization, translation, classification, creative writing, data extraction, formatting, etc.).<br><br><br>Process:<br><br>1. **Language Detection:** Detect the main language(s) present in the query (ISO 639-1, e.g., "en", "es").<br>2. **Simple Queries:** If the query is simple, direct, and contains no explicit requirements, respond only as: {<br><br>"Complex": "False" }<br><br>3. **Complex Queries** (with constraints):<br><br><br>a. Identify all task types implied in the query (e.g., "summarization", "creative writing", "math problem", etc.).<br>b. For each task type found:<br><br>- Extract the **Basic Query** (state the central goal, without any constraints).<br>- Extract all explicit constraints. For each constraint:<br><br><br>* Assign a concise category from: Content, Numerical, Style/Tone, Format, Language, Input Placeholder.<br>* Give the simplified query that results from removing this constraint but keeping all others.<br><br><br>c. Return results in JSON format as follows (must be valid JSON in English):<br><br><br>{<br><br>"Complex": "True", "Language": ["en"], "Tasks": [<br><br>{<br><br>"Task Type": "...", "Basic Query": "...", "Constraints": [<br><br>{"category": "...", "constraint": "...", "simplified query": "..."} ]<br><br>} // ... (one object per detected task type)<br><br>] }<br><br>Guidelines:<br><br>- If there are multiple task types, repeat the extraction for each.<br>- Each constraint should appear only once under its relevant task type.<br>- Category keywords: Content, Numerical, Style/Tone, Format, Language, Input Placeholder.<br>- Only include explicit "Constraints" (not just descriptive or illustrative details unless they restrict the response).<br>- Do not add extra commentary, and do not include sample queries or other meta-text outside the JSON.<br>|
|---|

- Table 5: Decompose Prompt

#### Compose Prompt, Compose Verify Prompt, and Compose Refine Prompt

|Compose Prompt<br><br>You are an expert in generating synthetic instruction from given base user query and constraints. Definition of Constraint: The smallest unit of restriction or requirement that the instruction imposes on the task.<br><br><base_query> {query} </base_query><br><br><constraints> {constraints} </constraints><br><br>Your task is the generate a compact instruction but contains all the information from base user query and constraints. For the placeholder, please add line breaks around them, e.g. ‘\n\n [SECTION SUMMARIES]\n{placeholder} \n[END OF SECTION SUMMARIES] \n\n‘. Avoid adding any extra information beyond the base query, constraints and placeholder. The output should be in English with JSON format. Here is an output template: {{"Created Prompt": "created prompt"}}|
|---|

- Table 6: Compose Prompt

|Compose Verify Prompt<br><br><constraints> {constraints} </constraints><br><br><generated_prompt> {prompt} </generated_prompt><br><br>Given constraints and a generated prompt, your task is to check if each constraint is covered in the generated prompt. Reply "yes" or "no" for each constraint. When answering "no", include a reason to explain why the constraint is missing. The final answer should be in JSON format. Here is an output template: {{<br><br>"1": {{ "reason": "Explain why this constraint is missing", "result": "yes or no",<br><br>}},<br><br>"2": {{ "reason": "Explain why this constraint is missing", "result": "yes or no",<br><br><br>}},<br><br>}} Let’s think step by step and output the final answer at the last step.|
|---|

- Table 7: Compose Verify Prompt

|Compose Refine Prompt<br><br>You are an expert in generating synthetic instruction from given base user query and constraints. Definition of Constraint: The smallest unit of restriction or requirement that the instruction imposes on the task.<br><br><base_query> {query} </base_query><br><br><constraints> {constraints} </constraints><br><br><critique> {critique} </critique><br><br>Your task is the generate a compact instruction but contains all the information from base user query and constraints. Pay extra attention to the critique so that the generated instruction contains all the constraint information. For the placeholder, please add line breaks around them, e.g. ‘\n\n[SECTION SUMMARIES]\n{{placeholder}}\n[END OF SECTION SUMMARIES]\n\n‘. Avoid adding any extra information beyond the base query, constraints.<br><br>The output should be in English with JSON format. Here is an output template: {{"Created Prompt": "created prompt"}}|
|---|

###### Table 8: Compose Refine Prompt

#### Prompt Scoring

|Prompt Scoring - Validity<br><br>Please evaluate the following AI task prompt for relevance, assumption correctness, and alignment with intent:<br><br>1. Determine how relevant each part of the prompt is to the main goal or question.<br>2. Identify any incorrect assumptions or factual inaccuracies.<br>3. Evaluate how well the prompt reflects its intended outcome or request.<br>4. Offer suggestions to improve relevance and clarity while maintaining the core intent.<br>5. Rate the overall validity of the task prompt (1-5, where 5 means perfectly valid).<br><br><br>Feel free to use specific examples from the prompt to illustrate your points.<br><br>Here is the AI task prompt: <AI_task_prompt> {task_prompt} </AI_task_prompt><br><br>Please think step-by-step and output your final judgment in the following JSON format. {{"reason": "Your reason", "score": "Score from 1 to 5"}}|
|---|

- Table 9: Prompt Scoring - Validity

|Prompt Scoring - Self-consistency<br><br>Please analyze the given task prompt for logical contradictions and inconsistencies in using the following requirements:<br><br>1. List any direct contradictions (where one requirement directly conflicts with another)<br>2. Identify any implicit contradictions (where requirements indirectly conflict)<br>3. Point out any ambiguous requirements that could lead to conflicts<br>4. Suggest ways to resolve these contradictions while maintaining the core intent<br>5. Rate the overall logical consistency of the prompt (1-5, where 5 is perfectly consistent) Feel free to use specific examples from the prompt to illustrate any contradictions you find.<br><br><br>Here is the AI task prompt: <AI_task_prompt> {task_prompt} </AI_task_prompt><br><br>Please think step-by-step and output your final judgment in the following JSON format. {{"reason": "Your reason", "score": "Score from 1 to 5"}}|
|---|

Table 10: Prompt Scoring - Self-consistency

#### Data Scoring

|Data Scoring - General<br><br>You are a quality evaluator to judge the output quality given by an AI assistant model. Given the AI assistant’s system_message (if any) and user_query, you need to evaluate whether the assistant_output satisfies the requirements in the AI assistant’s system_message (if any) and user_query, and whether the assistant_output can be directly shown to human users. You need to first produce a rationale of the reasoning process, followed by a score value. The score value is an integer from 1 to 5:<br><br>* 1 means the output quality is unacceptable for the task and cannot be shown to the users;<br>* 2 means the output quality is low, and needs significant modification to satisfy the requirement in the task prompt and presented to users;<br>* 3 means the output quality is acceptable, and needs some modest modification to address the requirement in the task prompt to show to users;<br>* 4 means the output quality is quite good, although the output can be made a bit more precise and concise to show to users;<br>* 5 means the output quality is prefect for the given task and can be shown to the users.<br><br><br>Here are the system message (if any), user query and assistant output: {system_message}<br><br><user_query> {user_query} </user_query><br><br><assistant_output> {assistant_output} </assistant_output><br><br>Please think step-by-step and output your final judgment in the following JSON object: {{"reason": "Your reason", "score": "Score from 1 to 5"}}|
|---|

- Table 11: Data Scoring - General

|Data Scoring - Helpfulness<br><br>You are a quality evaluator to judge the output quality given by an AI assistant model, focusing on informativeness and helpfulness. Given the AI assistant’s system_message (if any) and user_query, you need to evaluate if the assistant_output fulfill the task objective and provide high-quality, correct, and, informative content.<br><br>Please obey the following guidelines for correctness, informativeness, and helpfulness.<br><br><correctness> Accurate computation, reasoning steps, and outputs without misunderstandings or fabrication. </correctness><br><br><informativeness> Assign a numeric identifier (or \None") from 1 to 3 for each type of informativeness.<br><br>1. Clarity and Relevance: Ensure the response relates to the task and seek clarifications if needed.<br>2. Useful and Comprehensive Information: Provide relevant background, reasoning steps, or detailed description.<br>3. Not Lengthy, No Repetition: Avoid verbosity or recycling content. </informativeness><br><br><br><helpfulness> Helpfulness assessment emphasizes Overall Quality regarding correctness and informativeness.<br><br>Score 1 to 5 based on the extent of helpfulness, regarding both informativeness and correctness:<br><br>1. Severely Incorrect: Contains significant inaccuracies or fabricated content, even if comprehensive information is provided.<br>2. Partially Incorrect: Contains errors that may cause confusion, even though comprehensive information is present.<br>3. Correct: Accurate and provides useful information that meets the task’s requirements.<br>4. Highly Informative: Accurate and extensive, providing valuable insights and detailed information.<br>5. Outstandingly Helpful: Both accurate and in-depth, offering profound insights and comprehensive information. </helpfulness><br><br><br>Here are the system message (if any), user query and assistant output: {system_message}<br><br><user_query> {user_query} </user_query><br><br><assistant_output> {assistant_output} </assistant_output><br><br>Please think step-by-step and output your final judgment in the following JSON object: {{"reason": "Your reason", "score": "Score from 1 to 5"}}|
|---|

###### Table 12: Data Scoring - Helpfulness

|Data Scoring - Instruction Following<br><br>You are a quality evaluator to judge the output quality given by an AI assistant model, focusing on instruction following. Given the AI assistant’s system_message (if any), user_query, and assistant_output, you need to evaluate the alignment between the assistant_output, and AI assistant’s system_message (if any) and user_query as a whole.<br><br>Instruction Components: Task Goal (intended outcome), Restrictions (text styles, formats, or designated methods, etc). Please first produce a rationale of your reasoning process, followed by a score value from 1 to 5, where:<br><br>* 1 means Irrelevant: No alignment.<br>* 2 means Partial Focus: Addresses one aspect poorly.<br>* 3 means Partial Compliance: Meets goal or restrictions, neglecting other. And acknowledges both but slight deviations.<br>* 4 means Almost There: Near alignment, minor deviations.<br>* 5 means Comprehensive Compliance: Fully aligns, meets all requirements.<br><br><br>Here are the system message (if any), user query and assistant output: {system_message}<br><br><user_query> {user_query} </user_query><br><br><assistant_output> {assistant_output} </assistant_output><br><br>Please think step-by-step and output your final judgment in the following JSON object: {{"reason": "Your reason", "score": "Score from 1 to 5"}}|
|---|

###### Table 13: Data Scoring - Instruction Following

|Data Scoring - Uncertainty<br><br>You are a quality evaluator to judge the output quality given by an AI assistant model, focusing on honesty and uncertainty expression assessment.<br><br>Given the AI assistant’s system_message (if any), user_query, and assistant_output, you need to access how well the model conveys honesty and uncertainty. Please evaluate if the model’s confidence aligns with assistant_output correctness.<br><br>- Indicators of Uncertainty:<br><br>1. Weakeners: e.g., :"I guess," "probably."<br>2. Refusal to answer.<br>3. Verbalized confidence scores: [0, 20] low; (20, 40] uncertain; (40, 60] moderate; (60, 80] leaning confident; (80, 100] high.<br><br><br>- No uncertainty expression indicate confidence.<br>- Response Correctness: Align with ground truth, or provide accurate content without fabrication.<br><br><br>Please first produce a rationale of your reasoning process, followed by a score value from 1 to 5, where:<br><br>* 1 means Confidently Incorrect: Confident but entirely wrong.<br>* 2 means Confident with Significant Mistakes/Unconfident Incorrect:<br><br>- Confident but contains major errors.<br>- Unconfident and entirely wrong.<br><br><br>* 3 means Uncertain/"I Don’t Know"/Subtle Mistakes:<br><br>- "I don’t know" or declines.<br>- Confident but contains minor errors.<br>- Unconfident and contains significant mistakes.<br><br><br>* 4 means Correct but Uncertain/Expressed Subtle Mistakes:<br><br>- Correct but unconfident.<br>- Makes subtle mistakes but expresses uncertainty without specifying the exact area of doubt.<br><br><br>* 5 means Correct and Confident/Precisely Express Uncertainty:<br><br><br>- Correct and confident.<br>- Makes mistakes, but precisely acknowledges minor errors and indicates uncertainty on potential mistakes.<br><br><br>Here are the system message (if any), user query and assistant output: {system_message}<br><br><user_query> {user_query} </user_query><br><br><assistant_output> {assistant_output} </assistant_output><br><br>Please think step-by-step and output your final judgment in the following JSON object: {{"reason": "Your reason", "score": "Score from 1 to 5"}}|
|---|

###### Table 14: Data Scoring - Uncertainty

|Data Scoring - Truthfulness<br><br>You are a quality evaluator to judge the output quality given by an AI assistant model, focusing on truthfulness and hallucination. Given the AI assistant’s system_message (if any), user_query, and assistant_output, you need to evaluate the model’s accuracy in providing information in assistant_output without introducing misleading or fabricated details. Please obey the following guidelines for hallucination. <hallucination> Assign numeric identifier (or \None") from 1 to 3 for each type of hallucination below. Please output all the applicable types.<br><br>1. Contradictory with the World (Factual Error): Entities, locations, concepts, events that conflict with established knowledge.<br>2. Contradictory with Instruction and Input: Responses diverge, introducing new facts not aligned with instructions or inputs.<br>3. Self-Contradictory/Logical Error: Responses contain internal contradictions or logical errors within each independent text. Scoring: Rate outputs 1 to 5 based on extent of hallucination:<br><br><br>1. Completely Hallucinated: Entirely unreliable due to hallucinations.<br>2. Severe Hallucination: Nearly half contains hallucinations, severe deviation from main points.<br>3. Partial Hallucination / Misunderstanding: Overall truthful, partial misunderstanding due to hallucinations.<br>4. Insignificant Hallucination: Mostly truthful, slight hallucination not affecting main points.<br>5. No Hallucination: Free of hallucinations. </hallucination> Here are the system message (if any), user query and assistant output: {system_message}<br><br><br><user_query> {user_query} </user_query><br><br><assistant_output> {assistant_output} </assistant_output><br><br>Please think step-by-step and output your final judgment in the following JSON object: {{"reason": "Your reason", "score": "Score from 1 to 5"}}|
|---|

###### Table 15: Data Scoring - Truthfulness

#### GPT-4.1 Generated Meeting Summary Instruction Prompt

|Task Prompt for Meeting Summerization<br><br>Here is the meeting transcript for your analysis: <transcript> {INSERT_TRANSCRIPT_HERE} </transcript> You are an expert meeting summarizer with advanced skills. Your task is to thoroughly analyze the transcript and deliver a precise summary.<br><br>Please follow these guidelines to ensure clarity and high quality:<br><br>Understand Context and Participants: Identify each attendee’s role and area of expertise. Infer the overall purpose and context of the meeting (e.g., strategy, project update, decision-making).<br><br>Extract Main Points: Bring out significant decisions, concerns, proposals, and consensus areas. Condense extended dialogue into concise statements or bullet points without losing essential meaning.<br><br>Identify Follow-Up Steps: Gather all clear responsibilities discussed (both explicit and strongly implied). For each: Specify the responsible person/role, the nature of the commitment, relevant timelines or deadlines if mentioned, and the underlying reason or objective. Present these in a structured format for easy reference. Use active language and precise phrasing.<br><br>Maintain Relevance and Organization: Exclude side conversations or unrelated information. Avoid unnecessary repetition. Organize related points logically under subheadings if beneficial.<br><br>Output Structure: Start with a one-paragraph executive summary that captures the meeting’s outcomes. Follow with a section titled "Assignments" clearly listing all follow-up activities and responsibilities. Do not provide content beyond the summary and assignments.<br><br>Tone and Expression: Maintain a neutral, objective, and professional tone. Use clear and direct language. Ensure the findings are unambiguous and concise.<br><br>Do not include speculative content or infer steps not supported by the transcript.|
|---|

Table 16: Meeting summarization task prompt generated by GPT-4.1. This is used as a seed instruction prompt for TCIA and WizardLM to generate instructions for table 2.

### Additional Experimental Results

###### Task A - Diversity Density Curves by Hop

###### Hop 1

###### Hop 2

###### Hop 3

12.5

TCIA

TCIA

TCIA

10

8

TCIA Mean: 0.836

TCIA Mean: 0.788

TCIA Mean: 0.756

10.0

8

Frequency

WizardLM

WizardLM

WizardLM

6

7.5

6

WizardLM Mean: 0.844

WizardLM Mean: 0.735

WizardLM Mean: 0.710

4

5.0

4

2

2.5

2

0.0

0

0

0.5 0.6 0.7 0.8 0.9

0.5 0.6 0.7 0.8 0.9

0.5 0.6 0.7 0.8 0.9

Diversity

Diversity

Diversity

###### Task B - Diversity Density Curves by Hop

###### Hop 1

###### Hop 2

###### Hop 3

12.5

12.5

10

TCIA

TCIA

TCIA

10.0

10.0

TCIA Mean: 0.828

TCIA Mean: 0.810

TCIA Mean: 0.794

8

Frequency

WizardLM

WizardLM

WizardLM

7.5

7.5

6

WizardLM Mean: 0.798

WizardLM Mean: 0.722

WizardLM Mean: 0.647

5.0

5.0

4

2.5

2.5

2

0.0

0.0

0

0.5 0.6 0.7 0.8 0.9

0.5 0.6 0.7 0.8 0.9

0.5 0.6 0.7 0.8 0.9

Diversity

Diversity

Diversity

###### Task C - Diversity Density Curves by Hop

###### Hop 1

###### Hop 2

###### Hop 3

12.5

15

TCIA

TCIA

TCIA

10.0

TCIA Mean: 0.856

TCIA Mean: 0.811

TCIA Mean: 0.783

10.0

Frequency

WizardLM

WizardLM

WizardLM

7.5

10

7.5

WizardLM Mean: 0.775

WizardLM Mean: 0.746

WizardLM Mean: 0.709

5.0

5.0

5

2.5

2.5

0

0.0

0.0

0.5 0.6 0.7 0.8 0.9

0.5 0.6 0.7 0.8 0.9

0.5 0.6 0.7 0.8 0.9

Diversity

Diversity

Diversity

Task D - Diversity Density Curves by Hop

###### Hop 1

###### Hop 2

###### Hop 3

12.5

TCIA

TCIA

TCIA

15

TCIA Mean: 0.854

TCIA Mean: 0.851

TCIA Mean: 0.835

10.0

10

Frequency

WizardLM

WizardLM

WizardLM

7.5

10

WizardLM Mean: 0.774

WizardLM Mean: 0.695

WizardLM Mean: 0.653

5.0

5

5

2.5

0

0

0.0

0.5 0.6 0.7 0.8 0.9

0.5 0.6 0.7 0.8 0.9

0.5 0.6 0.7 0.8 0.9

Diversity

Diversity

Diversity

Figure 4: The diversity density plot of TCIA and WizardLM on in-house task A, B, C and D, after 1-3 hops.

Task A - On Topic Ratios by Hop

Task B - On Topic Ratios by Hop

100%

100%

On-TopicRatio

On-TopicRatio

80%

80%

60%

60%

40%

40%

TCIA

TCIA

20%

20%

WizardLM

WizardLM

0%

0%

Hop 1 Hop 2 Hop 3

Hop 1 Hop 2 Hop 3

Hop Number

Hop Number

Task C - On Topic Ratios by Hop

Task D - On Topic Ratios by Hop

100%

100%

On-TopicRatio

On-TopicRatio

80%

80%

60%

60%

40%

40%

TCIA

TCIA

20%

20%

WizardLM

WizardLM

0%

0%

Hop 1 Hop 2 Hop 3

Hop 1 Hop 2 Hop 3

Hop Number

Hop Number

- Figure 5: On-task ratio of the generated instructions by TCIA and WizardLM after 1-3 hops, for in-house task A, B, C and D.

|Details of TCIA’s setup for the Experiment section<br><br>BFS Hyperparameters: K = 2,700 m = 10 k = 2,000<br><br>Instruction Validation LLM: claude-3-5-sonnet-20241024<br><br>Output Generation LLMs:<br><br>(1) claude-3-5-sonnet-2024102<br>(2) claude-3-5-sonnet-20240620<br>(3) gpt-4o-2024-08-06<br>(4) gpt-4.1-2025-04-14<br><br><br>Data Quality Filtering LLM: gpt-4.1-2025-04-14|
|---|

Table 17: Details of TCIA’s setup for the Experiment section.

### Additional Statistical Analysis on the Instruction DB created from Tulu-3 Dataset

|Examples of decomposed instructions and constraint sets of a given user prompt in Tulu-3 Dataset<br><br>{<br><br>"Original User Query": "Create a snippet of Terraform HCL code that create an AWS autoscaling group, and an ALB in front to expose an application to internet.", "Complex": "True", "Language": "en", "Tasks": [<br><br>{<br><br>"Task Type": "Code Generation", "Basic Query": "Create code that sets up cloud infrastructure", "Constraints": [<br><br>{<br><br>"category": "Format Constraints", "constraint": "Must be written in Terraform HCL format", "simplified query": "Create code that sets up cloud infrastructure for AWS autoscaling group and ALB"<br><br>}, {<br><br>"category": "Content Constraints", "constraint": "Must include AWS autoscaling group configuration", "simplified query": "Create Terraform code that sets up an ALB to expose an application to internet"<br><br>}, {<br><br>"category": "Content Constraints", "constraint": "Must include AWS Application Load Balancer (ALB) configuration", "simplified query": "Create Terraform code that sets up an autoscaling group"<br><br>}, {<br><br>"category": "Content Constraints", "constraint": "Must configure ALB to expose application to internet", "simplified query": "Create Terraform code that sets up an autoscaling group and ALB"<br><br>} ]<br><br>} ]<br><br>}|
|---|

- Table 18: Examples of decomposed instructions and constraint sets of a given user prompt in Tulu-3 Dataset. The decomposition is conducted using the decompose prompt in Table 5 via GPT-4.1.

|Task Type<br><br>|Total number of queries<br><br>|# unique constraints/# of unique base query|
|---|---|---|
|Mathematical Problem Solving|144244<br><br>|2.84|
|Mathematical Word Problem Solving|89221|3.44<br><br>|
|Mathematical and Engineering Calculations|51785|2.47<br><br>|
|Technical and Creative Problem-Solving Across Multiple Domains|30030|2.52<br><br>|
|Mathematical Optimization and Linear Programming Problem Solving|28348|2.43<br><br>|
|Mathematical Function Creation and Analysis<br><br>|28316<br><br>|3.31|
|Mathematical Analysis|27827|2.44|
|Code Generation and Implementation<br><br>|21111|2.77|
|Translation|17944|3.00<br><br>|
|Mathematical Probability and Statistical Analysis<br><br>|17820|2.43|
|Creative Writing and Composition<br><br>|16181|4.61|
|Information Retrieval<br><br>|15144|1.82<br><br>|
|Code Implementation Questions|13677<br><br>|2.76|
|Natural Language Inference|12510|14.59|
|Multiple Choice Question Answering|9157|1.86|
|Mathematical Derivation|7858<br><br>|2.25|
|Explanation|6696|2.03<br><br>|
|Question Answering<br><br>|6309<br><br>|1.26|
|Text and Document Generation|6153<br><br>|2.68|
|Information Request|5644<br><br>|2.02|
|Statistical Analysis|5467<br><br>|2.27|

- Table 19: The top 20 task types of Tulu-3 Dataset, extracted using the decompose prompt in Table 5 (shown in the “Task Type” field of the LLM output, e.g. Table 18). We use all-mpnet-base-v2 embedding and DBSCAN clustering algorithm to cluster them, resulting in total of 9728 task types.

[Figure 3]

[Figure 4]

- Figure 6: Histograms of (1) Left: ratio of unique constraints and unique base queries ratio, (2) Right: ratio of unique constraints and unique raw queries ratio. This shows the distribution of query-constraint ratios in Tulu-3 Dataset. Note that the outliers mainly comes from multiple duplicated queries with different responses, for example “Write a restaurant review”.

|Constraint Type|Example constraints<br><br>|Occurrences|
|---|---|---|
|Probability and Statistical Calculations with Specific Parameters<br><br>|Must use softmax function σ(zi) = exp(zi)/ exp(zj), The context involves random selection of voters regarding mayor approval, Must consider six specific lines: AB, AC, AD, BC, BD, and CD|34989|
|Poisson Distribution Applications with Various Lambda Parameters<br><br>|Must use Poisson distribution with λ from previous calculation, Must use Poisson distribution with λ = 1200, Must use Poisson distribution with λ = 15|1113|
|Normal Distribution Calculations with Specified Parameters<br><br>|Must use normal distribution N(µ, σ2), Must use normal distribution with mean score of 75 points and standard deviation of 8 points, Must use normal distribution assumption for calculations|1062<br><br>|
|Binomial Distribution Probability Calculations<br><br>|Must use binomial distribution, Must use binomial distribution for probability calculation, Must use binomial distribution for calculation|301|
|Time-Based Normal Distribution Parameter Specifications<br><br>|Normal distribution with mean of 10 hours and standard deviation of 2 hours, Must use normal distribution with mean of 2 hours and standard deviation of 0.5 hours, Must use normal distribution with known standard deviation σ = 2 seconds<br><br>|281|
|Advanced Probability Theory and Mathematical Notation Requirements|Must use advanced statistical techniques, Must use mathematical notation and probability theory formalism, Must express the solution in mathematical/statistical notation<br><br>|170|
|Independent Events Requirement in Probability Problems|Must assume events are independent, All probabilities must be treated as independent events, Probabilities must be treated as independent events<br><br>|168|
|Statistical Confidence Level and Interval Requirements|Must achieve at least 90% confidence level, Must use 95% confidence interval, 95% confidence interval required<br><br>|122|
|Poisson Distribution Modeling for Sports Goal Statistics<br><br>|Must use Poisson distribution model for goal-scoring rates, Goals must follow a Poisson distribution with mean of 3 goals per game, Must use Poisson distribution with mean (λ) of 3.5 goals per game|113<br><br>|
|Use Given Transition Matrix for Markov Chain Calculations|Must use specific transition matrix P with given values, Must use the given 3x3 Markov transition matrix P with specific values, Must use a specific 3x3 transition probability matrix with given values<br><br>|90|

###### Table 20: Top 10 occurred example constraints in the task type “Mathematical Probability and Statistical Analysis” (the top 10th task in Table 19).

[Figure 5]

###### Figure 7: Toy example of TCIA on our in-house task B. Here, we use the seed prompt in Table 16 to first decompose into a base query Q and constraints C. Then the BFS algorithm is applied with only one operation: replace constraint 4. Lastly, {Q,C} is converted back to natural language prompt, which will subsequently go through instruction validation.

