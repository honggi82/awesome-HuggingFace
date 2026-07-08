# arXiv:2410.01999v4[cs.SE]9Apr2025

CODEMMLU: A MULTI-TASK BENCHMARK FOR ASSESSING CODE UNDERSTANDING & REASONING CAPABILITIES OF CODELLMS

Dung Nguyen Manh†∗, Thang Phan Chau†, Nam Le Hai‡, Thong T. Doan†, Nam V. Nguyen†, Quang Pham♢†, Nghi D. Q. Bui†∗ †FPT Software AI Center, Viet Nam, ‡Hanoi University of Science and Technology,

♢Independent Researcher

ABSTRACT

Recent advances in Code Large Language Models (CodeLLMs) have primarily focused on open-ended code generation, often overlooking the crucial aspect of code understanding & reasoning. To bridge this gap, we introduce CodeMMLU, a comprehensive multiple-choice benchmark designed to evaluate the depth of software and code comprehension in LLMs. CodeMMLU includes nearly 20,000 questions spanning diverse domains, including code analysis, defect detection, and software engineering principles across multiple programming languages. Unlike traditional benchmarks that emphasize code generation, CodeMMLU assesses a model’s ability to reason about programs across a wide-range of tasks such as code repair, execution reasoning, and fill-in-the-blank challenges. Our extensive evaluation reveals that even state-of-the-art models struggle with CodeMMLU, highlighting significant gaps in comprehension beyond generation. By emphasizing the essential connection between code understanding and effective AI-assisted development, CodeMMLU provides a critical resource for advancing more reliable and capable coding assistants. CodeMMLU is publicly available at: CodeMMLU

1 INTRODUCTION

Recent advancements in Code Large Language Models (CodeLLMs) (Wang et al., 2021; 2023c; Feng et al., 2020; Allal et al., 2023; Li et al., 2023; Lozhkov et al., 2024b; Guo et al., 2024b; Pinnaparaju et al., 2024; Zheng et al., 2024b; Roziere et al., 2023; Nijkamp et al., 2022; Luo et al., 2023; Xu et al., 2022; Bui et al., 2023; Hui et al., 2024; Bui et al., 2022; Dau et al., 2024b;a) have demonstrated impressive capabilities across various software engineering (SE) tasks (Bui et al., 2022; Dau et al., 2024a; To et al., 2023; White et al., 2024; Sobania et al., 2023; Phan et al., 2024; Sun et al., 2023; Nguyen et al., 2022; Bui, 2019; Zhang et al., 2022; Wang et al., 2023b). However, existing benchmarks often fall short to provide rigorous and reliable evaluations, largely due to outdated methodologies and the risk of data leakage (Matton et al., 2024). Moreover, practical applications of CodeLLMs reveal limitations such as bias and hallucination (Rahman & Kundu, 2024; Liu et al., 2024a) that current benchmarks fail to adequately address.

The predominant focus of coding-related benchmarks has been on open-ended, free-form generation tasks, such as code generation/code completion (Iyer et al., 2018; Lu et al., 2021; Chen et al., 2021; Austin et al., 2021; Lai et al., 2023; Hendrycks et al., 2021; Ding et al., 2023; Zhuo et al., 2024) and other SE tasks like program repair Ouyang et al. (2024); Xia et al. (2023) (Table 1). While appealing, these benchmarks struggle to discern whether CodeLLMs truly understand code or merely reproduce memorized training data (Carlini et al., 2022; Nasr et al., 2023). Additionally, the reliance on test cases and executability for evaluation limits the quantity and diversity of these benchmarks across domains, potentially leading to biased and limited generalizations. Recent efforts to improve

∗Corresponding authors: Dung Nguyen Manh (dungnm31@fpt.com) and Nghi D. Q. Bui (bdqnghi@gmail.com).

†The author contributed to this work while working with FPT Software AI Center.

GPT 4o

Llama3.3 70B Inst

QwenCoder2.5 32B Inst

Claude3.7 Sonnet

GPT o3-mini

Phi4 14B

DeekSeek R1

Programming syntax

API & Frameworks

Execution prediction

80

60

40

Software principles

Fill in the blank

20

DBMS & SQL

Code repair

Code completion

Others

- Figure 1: Summary performance of LLMs on the CodeMMLU benchmark. This radar chart presents the evaluation results (accuracy %) of different models across various CodeMMLU tasks.

evaluation through free-form question answering (Liu & Wan, 2021; Li et al., 2024) have introduced new challenges, often requiring less rigorous metrics or LLM-as-a-judge approaches (Zheng et al., 2023). However, LLMs-as-a-judge methods are susceptible to adversarial attacks (Raina et al., 2024), raising concerns about the reliability of such evaluation pipelines for coding tasks.

To address the aforementioned shortcomings, we introduce CodeMMLU, a novel benchmark designed to evaluate CodeLLMs’ ability to comprehend and reason about code through multiple-choice question answering (MCQ). This approach enables a deeper assessment of how CodeLLMs grasp coding concepts, moving beyond the mere generation capabilities. Inspired by the MMLU dataset (Hendrycks et al., 2020) from natural language understanding, CodeMMLU offers a robust and easy evaluation with the following key features. CodeMMLU comprises nearly 20,000 questions, facilitating a robust and comprehensive evaluation. Its large-scale data curation process mitigates potential biases and improves statistical reliability in measuring CodeLLMs’ performance across many capabilities. CodeMMLU covers over 50 software engineering disciplines and more than 10 programming languages, providing a holistic evaluation of CodeLLMs. The MCQ format is highly scalable, allowing for an accurate and straightforward assessment using precision-based metrics. Moreover, by incorporating permutations of answer choices, CodeMMLU emphasizes the model’s code understanding capabilities rather than memorizing the training datasets. Consequently, CodeMMLU provides a robust and accurate assessment of the models capabilities to understand software tasks.

CodeMMLU evaluates LLMs’ abilities in coding and software problem-solving from a fresh perspective, extending beyond conventional code generation and completion tasks. Our analysis uncovers several key insights: (1) previously unidentified bias issues in CodeLLMs, consistent with those observed in natural language MCQA tasks; (2) proprietary models (GPT-4o and Claude 3.5 Sonnet) consistently deliver the highest average performance; (3) among open-source models, the DeepSeek and Meta-Llama families achieve the greatest accuracy; (4) scaling laws tied to model size hold partially within the same model family but not across families, highlighting the critical roles of pre-training, post-training, and model architecture; (5) advanced prompting strategies, such as Chain-of-Thought (CoT), consistently impair performance, casting doubt on CodeLLMs’ reasoning capabilities for complex, multi-step tasks; and (6) when code completion benchmarks (e.g., HumanEval) are reframed from open-ended generation to MCQA format, LLMs exhibit reduced performance, questioning their true understanding of code. Notably, even strong reasoning models like DeepSeek-R1 underperform on CodeMMLU, suggesting significant room for improvement in

future iterations. These findings highlight the shortcomings of CodeLLMs in truly understanding code and reveal several promising future research avenues.

In summary, this work makes the following contributions:

- 1. We present the first MCQ benchmark for software and coding-related tasks, addressing the need for a comprehensive and large-scale evaluation in the code domain. CodeMMLU enables the evaluation of LLMs’ alignment with human inference in the software knowledge domain, similar to advances in the NLP field.
- 2. CodeMMLU provides a thorough assessment of LLM capabilities, ensuring a substantial number of samples and the diversity across tasks, domains, and languages. This enables a more nuanced understanding of an LLM’s strengths and weaknesses, facilitating the development of models better aligned with the complexities and demands of the software domain.
- 3. Our experiments offer critical insights into LLM performance, highlighting the impact of factors such as model size, model family, and prompting techniques. Notably, CodeMMLU unveils a performance gap between LLMs’ code generation and comprehension abilities. Additionally, we identify LLMs’ sensitivity to the selection biases introduced by the MCQ format. These findings provide valuable guidance for the community to enhance the robustness, adaptability, reliability, and domain-specific capabilities of LLMs in real-world software engineering.

- 2 RELATED WORK

Code Intelligence Benchmarks. The rapid development of Large Language Models (LLMs) for code-related tasks necessitates the development of diverse benchmarks to evaluate their performance. Algorithm-focused benchmarks, such as HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), and their extended versions (e.g., HumanEval+, MultiPL, MBPP+) (Liu et al., 2024b), focus on small-scale code generation tasks but lack the depth needed to assess broader comprehension. More challenging tasks, such as those in CodeContests (Li et al., 2022) and LiveCodeBench (Jain et al., 2024), provide competitive programming problems but remain primarily generative (Table 1). Comprehensive evaluation frameworks, such as CodeXGLUE (Lu et al., 2021), XLCoST (Zhu et al., 2022), and XCodeEval (Khan et al., 2023), provide versatility through multi-task assessments. However, these benchmarks are either dependent on metrics like BLEU and ROUGE, or testcase execution, which limits their reliability and scalability for a large-scale, comprehensive evaluation of LLM.

In contrast, multiple-choice question (MCQ) benchmarks offer a more standardized, scalable, and reliable evaluation method, as demonstrated in popular general-purpose benchmarks such as MMLU (Hendrycks et al., 2020) and TruthfulQA (Lin et al., 2022). Although MCQs facilitate large-scale assessments, recent studies highlight their susceptibility to biases, such as sensitivity to the answer choice orders (Wang et al., 2023a; Robinson et al., 2023). Existing MCQ benchmarks also lack focus on software engineering, limiting their applicability to code-related evaluations. In contrast, we curated data from a wide range of tasks and applied various filtering and debiasing techniques to improve CodeMMLU’s comprehensiveness, while minimizing data leakage and biases.

Understanding & Reasoning on Code There is a large body of research leveraging AI models for reasoning about code Gu et al. (2024a); Chen et al. (2024); Liu & Jabbarvand (2025); Dehghan (2024); Le et al. (2024b;a); Bieber et al. (2020); Shi et al. (2019); Li et al. (2021). Reasoning about code requires AI models to comprehend both its syntactic and semantic aspects. In the early days, static analysis was the primary approach to reasoning about code. With the rise of deep learning, methods that model code structures Mou et al. (2016); Bui et al. (2021b;a) emerged as promising alternatives. Subsequently, graph-based representations of code have gained traction, enabling reasoning about program properties and execution using graph neural networks (GNNs) Le et al. (2024a); Bieber et al. (2020); Shi et al. (2019). With the advent of large language models (LLMs), these models have been directly applied to reasoning about program behavior in downstream tasks Gu et al. (2024a); Chen et al. (2024); Liu & Jabbarvand (2025); Dehghan (2024); Le et al. (2024b). CodeMMLU provides a comprehensive benchmark that reflects real-world reasoning requirements by offering ground-truth answers to programming-related questions, allowing LLMs to derive final solutions through various reasoning paths.

## Table 1: Comparison between common code understanding benchmarks for LLMs in terms of coverage of foundation tasks of programming comprehension model.

Question Programming Task

Benchmark

Open-end MCQ SWE Test size knowledge

Code composition

Code comprehension

Code debugging

APPS Hendrycks et al. (2021) ✓ ✓ 5000 MBPP Austin et al. (2021) ✓ ✓ 974 HumanEval Chen et al. (2021) ✓ ✓ 164 CRUXEval Gu et al. (2024b) ✓ ✓ 800 LiveCodeBench Jain et al. (2024) ✓ ✓ ✓ ✓ 8801 CodeApex Fu et al. (2023) ✓ ✓ ✓ ✓ ✓ 2.056

CodeMMLU ✓ ✓ ✓ ✓ ✓ 19.912

- 3 CODEMMLU: DATA CURATION

The CodeMMLU benchmark is structured into two primary categories: (i) knowledge-based tests, designed to evaluate programming knowledge through questions addressing both syntactic and semantic aspects, and (ii) fundamental coding-skill tests, created by transforming high-quality codebase seeds into task-specific challenges. CodeMMLU includes nearly 20,000 questions spanning 52 diverse topics (Table 2). We design the knowledge-based tests to probe multi-level cognitive structures, assessing an LLM’s understanding of software knowledge at both semantic and syntactic levels. In contrast, the fundamental coding-skill test sets align with the cognitive process model of (Shneiderman & Mayer, 1979), focusing on core programming tasks that mimic real-world problem-solving scenarios.

- 3.1 KNOWLEDGE-BASED TASK CREATION

The knowledge-based test sets are designed to cover a wide range of topics and follow the multi-level cognitive structures model (Shneiderman & Mayer, 1979) which combines syntactic and semantic knowledge. The subset target is to measure the LLM’s coding capability and comprehensibleness of programming concepts. We collected raw programming-related questions and their corresponding multiple-choices answer from W3School (W3Schools, 2024) and Common Crawl project2 (See more license detail in Appendix A.3). The knowledge-based test set include:

- • Syntactic subset. Focused on programming language grammar and structural correctness, such as condition statement, format of iteration, common library usage.
- • Semantic subset. Targeted more abstract programming concepts, such as algorithms, data structures, object-oriented principles.

We maintain a high-quality evaluation set by filtering the raw data that undergoes a rigorous formatting and deep-learning-based filter in which we remove any instances that do not meet our quality criteria (see in section 3.3 and Appendix A.1). Resulting in an evaluation set (Table 2) that contains more than 11,000 instances, lying in 52 topics classified to 5 main subjects (categorized by source tag).

- 3.2 FUNDAMENTAL TEST CONSTRUCTION

Our benchmark encompasses four distinct MCQ programming tasks designed to assess the foundational capabilities outlined in the cognitive process model of programmer comprehension, namely: composition, comprehension, debugging, and modification.

Code Completion evaluates a model’s composition ability by requiring it to complete partially written code based on provided requirements. We adapted HumanEval (Chen et al., 2021), originally designed for code generation, into an MCQ format. From its 164 unique programming problems, we employed Large Language Models (LLMs) to generate plausible but incorrect solutions as distractors. All options, including correct solutions migrated from HumanEval and generated incorrect ones, were tested for executability. Some incorrect solutions were designed to pass certain test cases but fail

- 1https://github.com/LiveCodeBench/LiveCodeBench
- 2https://commoncrawl.org/

## Table 2: Summary of CodeMMLU Subject Categories and Task Distribution.

##### Subject Topic Source Testsize

Jquery, Django, Pandas, Numpy, Scipy,

knowledge

Syntactic

API & Frameworks usage

Azure, Git, AWS, svg, xml, 740 Bootstrap, NodeJS, AngularJS, React, Vue.

W3Schools,Geeksforgeeks,

C, C#, C++, Java, Javascript, PHP, Python, R, Ruby, MatLab, HTML, CSS, TypeScript.

Programming language syntax

6,220

CommonCrawl

DBMS & SQL DBMS, MySQL, PostgreSQL, SQL. 393 Data structure & Algorithm,

Semanticknowledge

Object-oriented programming, Compiler design, Computer organization and Architecture, Software Development & Engineering,

Software principles

3,246

System Design. Others

Program accessibility, Computer networks, Computer science, Cybersecurity, Linux, Web technologies, AWS.

1,308

Code completion HumanEval 163

Fundamen

-taltask

Fill in the blank LeetCode 2,129 Code repair QuixBugs 76

Execution Prediction IBM CodeNet 6,006

others, adding complexity and challenging models to distinguish between correct and nearly-correct solutions based on semantic and syntactic understanding.

Code Repair assesses a model’s debugging capability by requiring it to identify and fix errors in provided code snippets. We built this task upon QuixBugs (Lin et al., 2017), which was originally designed for debugging algorithmic programs. We used a "diff" operation on buggy and corrected versions in QuixBugs (Python and Java) to identify specific fixes, which served as correct solutions. To create plausible distractors, we targeted components frequently involved in bugs (e.g., return statements, loop conditions, if/else/switch expressions) and guided LLMs to generate alternative fixes. These alternatives were designed to seem plausible but not fully resolve the bug. Each distractor was verified for incorrectness, and all options were made executable to ensure that models needed a deep understanding of the code to identify and apply the correct fix.

Execution Prediction evaluates a model’s ability to identify and understand defects within code snippets, focusing on both logical and syntactical errors. This task measures the comprehension and debugging capabilities of LLMs by requiring them to predict the execution outcome of given code. It includes two sub-tasks: detecting any defects/flaws in the provided code and comprehending the output of a certain test sample. We derived this task set from IBM CodeNet (Puri et al., 2021), a large-scale benchmark for algorithmic coding tasks. We focused on Python and Java subsets, collecting both accepted and buggy versions of code. After filtering out duplicates, we created a diverse set of code samples. For each snippet, we provide the correct execution result (golden answer) and three distracting options, which could be one of several possible outcomes: (i) Compile Error, (ii) Time Limit Exceeded, (iii) Memory Limit Exceeded, (iv) Runtime Error, or (v) No abnormally found.

Fill in the Blank evaluates a model’s code comprehension ability by requiring it to complete missing parts of a code snippet, given documentation and an incomplete code sample. This task assesses not only the model’s ability to fill gaps but also its understanding of both high-level programming concepts and low-level grammatical structures. We collected approximately 2,000 coding problems from LeetCode 3, covering solutions in three widely-used programming languages (Python, Java, C++). From each problem’s solution, we parsed and randomly selected key components (i.e. crucial elements of the program’s logic and flow like loop conditions, expression statements, conditional statements) to be blanked out. To create plausible but incorrect options for the multiple-choice question (MCQ) format, we employed LLM to generate alternative solutions for the blanked-out components. These distractors were designed to be contextually relevant but incorrect, adding complexity to the task. We executed all generated options to verify their incorrectness, ensuring they do not solve the problem as intended.

3https://leetcode.com/

| |CodeMMLU| |
|---|---|---|
| | | |

###### Raw MCQs

[Figure 1]

(1) Deep-learning Filters: LLMs judgement follow defined categories.

Crawling Filtering

(2) Filters by difficulty: Test on set of high quality LLMs to filter out low difficulty question.

GeeksForGeeks, W3school, etc

Prompt configs (fewshot, CoT, etc)

###### Question & Golden answer

evaluate

Filtering

Extracting

Code

Testcases

Synthesize distractor

IBM CodeNet, QuixBugs, LeetCode, etc

Execution-based Filter

Language models

Language models

- Figure 2: Overview of CodeMMLU data creation pipeline. The blue diagram describe the process of collecting raw multiple-choice questions (MCQs) from open source internet for a knowledge testset. Otherwise, the pipeline of real-world problem indicated in orange area.

- 3.3 DATA CLEANING

The preprocessing process (described in Figure 2) includes a deep learning-based filtering and execution-based filtering to ensure that each question met the desired quality standards, including clarity, lack of ambiguity, and difficulty.

LLM-based Filtering To begin, we employed an LLM-based filter to assess the instances in the crawled knowledge test set. Each instance was evaluated based on three criteria: Completeness, Coherence and clarity, and Coding relevance. The models utilized for this evaluation included GPT-3.5, Llama3.1-8B Instruct, and Mixtral-8×7B Instruct. We averaged the scores and used them to select a filtering threshold for each criteria (see discussion in Appendix A.1). To detect and handle duplications, we applied the MinHash LSH algorithm ((Zhu et al., 2023)), configured with 256 permutations, to cluster near-duplicate questions. We remove all false positive instances in each cluster with 0.8 as the similarity threshold. To verify the efficacy of the LLM-based filter, we randomly selected 100 instances from each subject area for manual verification against the three criteria.

Execution-based Filtering To ensure the question correctness, we apply an execution-based filtering in the fundamental test sets. We merge the distractor of (i) code completion, (ii) fill-inthe-blank, and (iii) code repair with their codebase and execute with their corresponding test cases. The distractor is designed to bring challenge since it requires LLM to comprehend their correctness without executing it, we select distractors that are executable with 0 to few (less than 50%) test cases passed in their execution result. In the other hand, the task Execution Prediction’s groundtruth are collected from executing process, the distractor are randomly pick from common executing scenarios.

- 4 EXPERIMENTAL RESULTS

- 4.1 SETUP

Model selection. We evaluate CodeMMLU on 40 popular open-source LLMs, covering a wide range of parameter sizes and architectures. The models were selected from 13 different families, with parameters ranging from 1 billion to over 70 billion. Each family included base and instructed/chat versions. In addition to open-source models, we also included several proprietary models from OpenAI and Anthropic to ensure a comprehensive coverage of the state-of-the-art in language modeling. All model information can be found at C.

Answer extraction. CodeMMLU leverages the MCQ format for scalability and ease of evaluation. In order to maintain this advantage, we only apply simple regex methods to extract the selection answer (i.e., extract by directly answering (A|B|C|D) or containing the pattern “answer is A|B|C|D”). The model response is required to be parsable; otherwise, it will be marked as unanswered.

Random score

80

Zero

Few CoT

60

40

20

0

PLsyntaxAPI&FrameworksDBMS&SQLSoftwareprinciples OthersCodeCompletionFillinthemiddleCodeRepairExecutionPrediction

- Figure 3: Comparison of prompt configuration on GPT-4o. The experiment exposes the drawback of Chain-of-Thought prompting technique in term of boosting performance on task that not require logic or reasoning.

In the following, we present key findings of CodeMMLU on (i) knowledge and fundamental test correlation; (ii) MCQ bias evidance; (iii) Disagreement between code-generation alike benchmark and MCQ format. Due to space constraints, we provide detail experimental results, additional discussions and analyses in the appendix, including assessing data leakage (Appendix A.2); MCQs analysis (Appendix B.1); Chain-of-thought technique analysis B.2 and full 43 LLMs results B.3.

4.2 KEY INSIGHTS

Overall performance CodeMMLU revealed significant performance differences across models, as shown in Table 3. OpenAI’s GPT-4o outperformed all models on CodeMMLU, demonstrating its quality across diverse tasks (Figure 1). Notably, despite not being the latest model, the instructed version of Llama3.1 70B from Meta achieved the highest score among open-source models from 13 families. While LLMs perform well on knowledge-based tasks, they struggle with real-world problems, particularly in execution prediction tasks (see Appendix B.3 for all the experiment details.)

- Figure 4 illustrates CodeMMLU’s capability to measure LLMs’ coding knowledge and skills across a wide range of subjects. Our benchmark provides clear, distinct rankings that establish a higher hierarchy of models compared to other benchmarks (see the result in Table 7). Interestingly, the results do not strictly adhere to scaling laws (Kaplan et al., 2020), where larger parameter sizes typically outperform smaller ones. This highlights the impact of data quality in the LLM pretraining process, as recently released models often achieve comparable performance to larger models from previous versions. CodeMMLU also indicates the importance of instruction tuning in improving model performance on complex tasks. Models with instruction tuning substantially outperform their non-instructed counterparts, for example, DeepSeek-Coder-33b surpasses its base model by approximately 29%.

The impact of model reasoning and Chain-of-Thought (CoT) prompting Although CoT prompting (Wei et al., 2023) is often expected to enhance performance by eliciting deeper reasoning, our experiments reveal that CoT and reasoning models may not always offer improvements in CodeMMLU. Table 3 shows that DeepSeek R1 performs significantly worse than its base model, DeepSeek V3, despite being designed for reasoning tasks. Meanwhile, GPT o3-mini, a native reasoning model, achieves the best results. Our analysis suggests that, apart from GPT o3-mini, other reasoning models (e.g, GPT o1, DeepSeek R1) tend to overreason across all CodeMMLU tasks. In contrast, o3-mini demonstrates the ability to decide when to apply reasoning versus directly answering knowledgeseeking tasks, such as in syntactic and semantic evaluations (Figure 11 12 ). These findings align with (Chen et al., 2025), which observes that current reasoning models often overreason even on simple questions. This suggests the need for a more effective decision-making mechanism to determine when reasoning is beneficial. Table 6 further highlights that while R1 generates the longest responses (in tokens), it underperforms compared to the latest top-tier LLMs.

We further investigate the impact of different prompting techniques and report the results in Figure 3. The results show a significant decline in GPT-4o’s performance with CoT, suggesting that the additional complexity introduced by step-by-step reasoning does not align well with knowledgeseeking tasks (see Appendix B.2 for more discussions). In contrast, few-shot prompting consistently emerges as the most reliable and effective strategy across various tasks, offering a balanced approach without overwhelming the models. Overall, with the exception of GPT o3-mini, we found no improvements from reasoning models or CoT prompting, suggesting that CodeMMLU presents a challenging benchmark to test the models reasoning capabilities in code domains.

- Table 3: Summary of LLM Family Performance on CodeMMLU. The evaluation results (accuracy %) of different language models across the CodeMMLU task (CodeMMLU column represents the accuracy average among all subject).

Knowledge test

Syntactic Semantic Fundamental test CodeMMLU Closed-source models

Family Model name Size (B)

Claude3.7 Sonnet - 52.78 76.26 60.92 61.65 Claude3.5 Sonnet - 52.23 73.45 58.56 59.81 Claude3.5 Haiku - 49.24 68.20 57.83 57.25 Claude3 Sonnet - 67.22 66.08 38.26 53.97

Claude

GPT o3-mini - 53.08 75.50 62.77 62.36 GPT 4o - 50.63 69.61 53.89 56.40 GPT 4o-mini - 48.66 55.90 20.33 38.43 GPT-3.5-turbo - 61.68 53.65 45.26 51.70

GPT

Open-source models

Llama3.3 70B Inst 70 44.31 52.76 30.96 40.66 Llama3.1 405B Inst 405 50.82 71.41 57.10 58.23 Llama3 70B Inst 70 46.94 62.64 53.15 53.19 CodeLlama34B Inst 34 56.81 46.93 23.55 38.73

Llama

DeepSeek R1 671 42.39 56.77 38.08 43.91 DeepSeek V3 685 48.30 57.53 45.06 49.08 DeepSeekCoder 33B Inst 33 53.65 45.43 21.46 36.60 DeepSeekMoE 16B Chat 16.4 31.74 35.42 27.32 31.01

DeepSeek

Mistral7B Inst (v0.3) 7 54.42 51.25 31.85 43.33 Mixtral 8×7B Inst 46.7 61.17 54.89 24.09 42.96 Codestral 22B 22 60.34 52.10 37.85 47.60

Mistral

Phi Phi4 14 45.34 57.46 47.82 49.19 Phi4 Mini Inst 12 41.94 51.59 19.75 34.85

Qwen2.5 14B Inst 14 46.38 58.70 51.49 51.38 QwQ 32B Preview 57 61.34 57.48 30.48 46.34 QwenCoder2.5 32B Inst 32 50.63 69.61 53.89 56.40

Qwen

| | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |

Programming Language Syntax

API & Frameworks

Software Principles

Database management & SQL

Others

Code Completion

Code Repair

Fill in the blank

Execution Prediction

Starcoder215BInst

Yi1.534BChat

DeepSeekMoE16BChat

DeepSeekMoE16B

DeekSeekR1DeepSeekV3

QwenCoder2.532BInstQwen257BA14BInst

QwQ32B Phi3MediumInst

Phi4

Codestral22Bv0.1Mixtral8x7BInstMistral7BInstv0.3CodeLlama34BInstCodeLlama13BInstLlama3.1405BInstLlama3.170BInst

Llama3.170B Llama3.370BInst GPT3.5-TurboGPT4o-mini

GPT4o GPTo3-mini Claude3SonnetClaude3.7SonnetClaude3.5HaikuClaude3.5Sonnet

- Figure 4: CodeMMLU accuracy by task on LLMs. While knowledge tasks are following the scaling law, real-world tasks offer more challenges to LLMs which indicate the performance of instruction tuning and data quality when evaluating on CodeMMLU.

Correlation Between Software Knowledge and Real-World Performance Our experiments revealed a strong correlation between performance on knowledge-based tasks and real-world coding challenges. Specifically, the Pearson correlation score r = 0.61 between model rankings on the knowledge test set and their performance on real-world problems, derived from the accuracy of 43 LLMs across 15 model families, indicates a moderate alignment (Figure 5). This suggests that models demonstrating a deeper understanding of software principles consistently excel in real-world coding tasks, highlighting the importance of foundational knowledge for practical coding performance.

80

60

Real-Task

40

20

0

20 40 60

Knowledge-Task

- Figure 5: Correlation between knowledge tests and fundamental skill tests. Experiments on 10 LLM families show a clear alignment between models with a strong understanding of software knowledge and their performance on diverse problem-solving tasks in the CodeMMLU fundamental skill tests.

Model comparison with CodeMMLU score and Response length

| |Syntatic<br><br>Semantic<br><br>Fundamental<br><br>56.40<br><br>62.36<br><br>43.91<br><br>49.08<br><br>CodeMMLU| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

62.5

1750

60.0

1500

57.5

Modelresponselength(tokens)

1250

CodeMMLUScore

55.0

1000

52.5

750

50.0

500

47.5

250

45.0

0

GPT4o GPTo3-mini DeepSeekR1 DeepSeekV3

Figure 6: Comparison Between LLM Response Length and CodeMMLU Score. While reasoning models (e.g., DeepSeek-R1) generate longer, reasoningheavy responses, this does not necessarily correlate with higher accuracy on CodeMMLU.

Selection bias in MCQs format We experimented with multiple answer order permutations (follow (Zheng et al., 2024a)), the result displayed significant inconsistent behavior exhibited by LLMs when swapping golden answer positions. As presented in Table 7, the model’s performance changes dramatically in each answer order configuration, which is based on the correct answer’s position. The LLM’s accuracy fluctuates between different permutations (i.e. DeepSeek-Coder-34B ∆σ = 36.66), demonstrating how sensitive it can be to the structure and order of answers (Figure 9). However, Table 8 indicate the different of MCQ bias between strong models (e.g GPT-4o, Claude3-orpus) and others, which highlight the consistency and robustness among them (see discussion in B.1).

Disagreement between Open-ended generation benchmark and MCQ Code completion A notable finding from our experiments is the discrepancy in model performance between open-ended benchmarks and multiple-choice formats. Specifically, when comparing the original HumanEval questions with their multiple-choice equivalents in our CodeMMLU code completion set, we found that models performing well on HumanEval do not consistently replicate their success in CodeMMLU. For instance, when evaluating identical questions across the formats, the number of cases where models answered both correctly or incorrectly was unexpectedly low. The correlation scores in Figure 7 further illustrate the weak alignment of success between these two benchmarks, revealing that performance in open-ended tasks does not reliably predict performance in multiple-choice coding tasks. This lack of alignment suggests that traditional benchmarks might overestimate a model’s understanding by focusing too narrowly on code generation, which is highly susceptible to data leakages. In contrast, CodeMMLU requires the models to engage in complex reasoning to understand code and solve software engineering problems.

- 5 CONCLUSIONS

In this work, we introduced CodeMMLU, a comprehensive and scalable benchmark designed to evaluate large language models’ (LLMs) capabilities across a wide range of software knowledge

- Table 4: Performance Comparison between HumanEval and MCQ Code Completion Tasks. The performance fluctuation highlights the selection biases observed when the correct (golden) answer is moved to positions A, B, C, or D.

Code Completion MCQ A B C D

Models HumanEval

0.00 90.24 14.02 0.61

CodeLlama-7B-Python 40.48

- (-40.48) (+49.76) (-26.46) (-39.87)

CodeLlama-7B-Instruct 45.65

3.66 1.22 93.90 15.85

- (-41.99) (-44.43) (+48.25) (-29.80)

CodeLlama-13B-Python 42.89

0.61 54.88 70.12 12.20

- (-42.28) (+11.99) (+27.23) (-30.69)

2.44 68.29 72.56 29.88 (-48.16) (+17.69) (+21.96) (-20.72) CodeLlama-34B-Python 45.11

CodeLlama-13B-Instruct 50.6

0.61 77.44 70.73 49.39

(-44.50) (+32.33) (+25.62) 4.28 CodeLlama-34B-Instruct 50.79

9.15 84.76 65.24 46.34

(-41.64) (+33.97) (+14.45) (-4.45) Deepseek-Coder-7B-base-v1.5 43.2

40.85 74.39 64.02 39.02 (-2.35) (+31.19) (+20.82) (-4.18) DeepSeek-Coder-33B-base 56.1

1.22 82.32 75.00 56.10

(-54.88) (+26.22) (+18.90) (0.00) Phind-CodeLLama-34B-v2 71.95

6.10 90.85 75.00 46.34 (-65.85) (+18.90) (+3.05) (-25.61) Mixtral-8x7B-Instruct-v0.1 40.2

22.56 74.39 71.95 63.41 (-17.64) (+34.19) (+31.75) (+23.21)

Meta-Llama-3.1-8B-Instruct (corrcoef=0.0986)

Meta-Llama-3-8B-Instruct (corrcoef=0.0971)

CodeQwen1.5-7B (corrcoef=0.1227)

False True

False True

False True

[Figure 2]

|[Figure 3]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

[Figure 4]

[Figure 5]

80

60 5

- 32 39
- 33 60

52 26

FalseTrue

FalseTrue

FalseTrue

HumanEval

HumanEval

HumanEval

70

60

47 39

85 14

50

Code completion

Code completion

Code completion

Qwen2-7B-Instruct (corrcoef=0.1083)

starcoder2-7b (corrcoef=0.1335)

False True

False True

40

[Figure 6]

[Figure 7]

33 62

74 28

FalseTrue

FalseTrue

30

HumanEval

HumanEval

20

17 52

37 25

10

Code completion

Code completion

Figure 7: Comparison of CodeMMLU’s code completion task and HumanEval. Many LLMs show a performance discrepancy between the two tasks, where models that successfully passed the HumanEval code generation test often failed to select the correct answer in the multiple-choice (MCQ) format, or vice versa, for the same question.

and real-world programming tasks. Our experiments highlighted the benchmark’s key advantages, including its cost-effectiveness, scalability, and extensive task coverage. The insights gained revealed a strong correlation between software knowledge and real-world task performance, demonstrating that models with deeper comprehension outperform those relying purely on probabilistic generation.

Additionally, CodeMMLU provides more accurate and detailed rankings of LLMs, particularly in open-source models, where significant reordering of performance was observed. The benchmark also revealed inconsistencies in model comprehension when compared to traditional evaluations like HumanEval, emphasizing the need for more robust benchmarks that go beyond simple code generation.

REFERENCES

01. AI, :, Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, Kaidong Yu, Peng Liu, Qiang Liu, Shawn Yue, Senbin Yang, Shiming Yang, Tao Yu, Wen Xie, Wenhao Huang, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Pengcheng Nie, Yuchi Xu, Yudong Liu, Yue Wang, Yuxuan Cai, Zhenyu Gu, Zhiyuan Liu, and Zonghong Dai. Yi: Open foundation models by 01.ai, 2024. URL https://arxiv.org/ abs/2403.04652.

Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, et al. Santacoder: don’t reach for the stars! arXiv preprint arXiv:2301.03988, 2023.

Anthropic. The claude 3 model family: Opus, sonnet, haiku. 2024. URL https://api. semanticscholar.org/CorpusID:268232499.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

David Bieber, Charles Sutton, Hugo Larochelle, and Daniel Tarlow. Learning to execute programs with instruction pointer attention graph neural networks. Advances in Neural Information Processing Systems, 33:8626–8637, 2020.

Nghi Bui. Towards zero knowledge learning for cross language api mappings. In 2019 IEEE/ACM 41st International Conference on Software Engineering: Companion Proceedings (ICSE-Companion), pp. 123–125. IEEE, 2019.

Nghi DQ Bui, Yijun Yu, and Lingxiao Jiang. Infercode: Self-supervised learning of code representations by predicting subtrees. In 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE), pp. 1186–1197. IEEE, 2021a.

Nghi DQ Bui, Yijun Yu, and Lingxiao Jiang. Treecaps: Tree-based capsule networks for source code processing. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 30–38, 2021b.

Nghi DQ Bui, Yue Wang, and Steven Hoi. Detect-localize-repair: A unified framework for learning to debug with codet5. arXiv preprint arXiv:2211.14875, 2022.

Nghi DQ Bui, Hung Le, Yue Wang, Junnan Li, Akhilesh Deepak Gotmare, and Steven CH Hoi. Codetf: One-stop transformer library for state-of-the-art code llm. arXiv preprint arXiv:2306.00029, 2023.

Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, Xiaoyi Dong, Haodong Duan, Qi Fan, Zhaoye Fei, Yang Gao, Jiaye Ge, Chenya Gu, Yuzhe Gu, Tao Gui, Aijia Guo, Qipeng Guo, Conghui He, Yingfan Hu, Ting Huang, Tao Jiang, Penglong Jiao, Zhenjiang Jin, Zhikai Lei, Jiaxing Li, Jingwen Li, Linyang Li, Shuaibin Li, Wei Li, Yining Li, Hongwei Liu, Jiangning Liu, Jiawei Hong, Kaiwen Liu, Kuikun Liu, Xiaoran Liu, Chengqi Lv, Haijun Lv, Kai Lv, Li Ma, Runyuan Ma, Zerun Ma, Wenchang Ning, Linke Ouyang, Jiantao Qiu, Yuan Qu, Fukai Shang, Yunfan Shao, Demin Song, Zifan Song, Zhihao Sui, Peng Sun, Yu Sun, Huanze Tang, Bin Wang, Guoteng Wang, Jiaqi Wang, Jiayu Wang, Rui Wang, Yudong Wang, Ziyi Wang, Xingjian Wei, Qizhen Weng, Fan Wu, Yingtong Xiong, Chao Xu, Ruiliang Xu, Hang Yan, Yirong Yan, Xiaogui Yang, Haochen Ye, Huaiyuan Ying, Jia Yu, Jing Yu, Yuhang Zang, Chuyu Zhang, Li Zhang, Pan Zhang, Peng Zhang, Ruijie

Zhang, Shuo Zhang, Songyang Zhang, Wenjian Zhang, Wenwei Zhang, Xingcheng Zhang, Xinyue Zhang, Hui Zhao, Qian Zhao, Xiaomeng Zhao, Fengzhe Zhou, Zaida Zhou, Jingming Zhuo, Yicheng Zou, Xipeng Qiu, Yu Qiao, and Dahua Lin. Internlm2 technical report, 2024. URL https://arxiv.org/abs/2403.17297.

Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, and Chiyuan Zhang. Quantifying memorization across neural language models. arXiv preprint arXiv:2202.07646, 2022.

Junkai Chen, Zhiyuan Pan, Xing Hu, Zhenhao Li, Ge Li, and Xin Xia. Reasoning runtime behavior of a program with llm: How far are we? arXiv preprint cs.SE/2403.16437, 2024.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. Do not think that much for 2+3=? on the overthinking of o1-like llms, 2025. URL https: //arxiv.org/abs/2412.21187.

Anh Dau, Jin LC Guo, and Nghi Bui. Docchecker: Bootstrapping code large language model for detecting and resolving code-comment inconsistencies. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pp. 187–194, 2024a.

Anh TV Dau, Hieu Trung Dao, Anh Tuan Nguyen, Hieu Trung Tran, Phong X Nguyen, and Nghi DQ Bui. Xmainframe: A large language model for mainframe modernization. arXiv preprint arXiv:2408.04660, 2024b.

DeepSeek-AI, Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu, Y. Wu, Yukun Li, Huazuo Gao, Shirong Ma, Wangding Zeng, Xiao Bi, Zihui Gu, Hanwei Xu, Damai Dai, Kai Dong, Liyue Zhang, Yishi Piao, Zhibin Gou, Zhenda Xie, Zhewen Hao, Bingxuan Wang, Junxiao Song, Deli Chen, Xin Xie, Kang Guan, Yuxiang You, Aixin Liu, Qiushi Du, Wenjun Gao, Xuan Lu, Qinyu Chen, Yaohui Wang, Chengqi Deng, Jiashi Li, Chenggang Zhao, Chong Ruan, Fuli Luo, and Wenfeng Liang. Deepseek-coder-v2: Breaking the barrier of closed-source models in code intelligence, 2024. URL https://arxiv.org/abs/2406.11931.

Saman Dehghan. Assessing code reasoning in large language models: A literature review of benchmarks and future directions. 2024.

Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Hantian Ding, Ming Tan, Nihal Jain, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, and Bing Xiang. Crosscodeeval: A diverse and multilingual benchmark for cross-file code completion. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu

Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaoqing Ellen Tan, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aaron Grattafiori, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alex Vaughan, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Franco, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, Danny Wyatt, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Firat Ozgenel, Francesco Caggioni, Francisco Guzmán, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Govind Thattai, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Igor Molybog, Igor Tufanov, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Karthik Prasad, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kun Huang, Kunal Chawla, Kushal Lakhotia, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Maria Tsimpoukelli, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikolay Pavlovich Laptev, Ning Dong, Ning Zhang, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Raymond Li,

Rebekkah Hogan, Robin Battey, Rocky Wang, Rohan Maheswari, Russ Howes, Ruty Rinott, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Kohler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaofang Wang, Xiaojian Wu, Xiaolan Wang, Xide Xia, Xilun Wu, Xinbo Gao, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yuchen Hao, Yundi Qian, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, and Zhiwei Zhao. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, et al. Codebert: A pre-trained model for programming and natural languages. arXiv preprint arXiv:2002.08155, 2020.

Lingyue Fu, Huacan Chai, Shuang Luo, Kounianhua Du, Weiming Zhang, Longteng Fan, Jiayi Lei, Renting Rui, Jianghao Lin, Yuchen Fang, Yifan Liu, Jingkuan Wang, Siyuan Qi, Kangning Zhang, Weinan Zhang, and Yong Yu. Codeapex: A bilingual programming evaluation benchmark for large language models. ArXiv, abs/2309.01940, 2023. URL https://api.semanticscholar.

org/CorpusID:261530384.

Alex Gu, Baptiste Rozière, Hugh Leather, Armando Solar-Lezama, Gabriel Synnaeve, and Sida I Wang. Cruxeval: A benchmark for code reasoning, understanding and execution. arXiv preprint arXiv:2401.03065, 2024a.

Alex Gu, Baptiste Rozière, Hugh Leather, Armando Solar-Lezama, Gabriel Synnaeve, and Sida I. Wang. Cruxeval: A benchmark for code reasoning, understanding and execution, 2024b. URL https://arxiv.org/abs/2401.03065.

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, and Wenfeng Liang. Deepseek-coder: When the large language model meets programming – the rise of code intelligence, 2024a. URL https://arxiv.org/abs/2401.14196.

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y Wu, YK Li, et al. Deepseek-coder: When the large language model meets programming–the rise of code intelligence. arXiv preprint arXiv:2401.14196, 2024b.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, and Jacob Steinhardt. Measuring coding challenge competence with apps. NeurIPS, 2021.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186, 2024.

Srinivasan Iyer, Ioannis Konstas, Alvin Cheung, and Luke Zettlemoyer. Mapping language to code in programmatic context. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 1643–1652, 2018.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.

Fred Jelinek, Robert L Mercer, Lalit R Bahl, and James K Baker. Perplexity—a measure of the difficulty of speech recognition tasks. The Journal of the Acoustical Society of America, 62(S1): S63–S63, 1977.

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, MarieAnne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts, 2024. URL https://arxiv.org/abs/2401.04088.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models,

2020. URL https://arxiv.org/abs/2001.08361.

Mohammad Abdullah Matin Khan, M Saiful Bari, Xuan Long Do, Weishi Wang, Md Rizwan Parvez, and Shafiq Joty. xcodeeval: A large scale multilingual multitask benchmark for code understanding, generation, translation and retrieval. arXiv preprint arXiv:2303.03004, 2023.

Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, Luke Zettlemoyer, Wen-tau Yih, Daniel Fried, Sida Wang, and Tao Yu. Ds-1000: A natural and reliable benchmark for data science code generation. In International Conference on Machine Learning, pp. 18319–18345. PMLR, 2023.

Cuong Chi Le, Hoang Nhat Phan, Huy Nhat Phan, Tien N Nguyen, and Nghi DQ Bui. Learning to predict program execution by modeling dynamic dependency on code graphs. arXiv preprint arXiv:2408.02816, 2024a.

Cuong Chi Le, Hoang-Chau Truong-Vinh, Huy Nhat Phan, Dung Duy Le, Tien N Nguyen, and Nghi DQ Bui. Visualcoder: Guiding large language models in code execution with fine-grained multimodal chain-of-thought reasoning. arXiv preprint arXiv:2410.23402, 2024b.

Linyi Li, Shijie Geng, Zhenwen Li, Yibo He, Hao Yu, Ziyue Hua, Guanghan Ning, Siwei Wang, Tao Xie, and Hongxia Yang. Inficoder-eval: Systematically evaluating the question-answering capabilities of code large language models. arXiv preprint arXiv:2404.07940, 2024.

Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. Starcoder: may the source be with you! arXiv preprint arXiv:2305.06161, 2023.

Xuechen Li, Chris J Maddison, and Daniel Tarlow. Learning to extend program graphs to work-inprogress code. arXiv preprint arXiv:2105.14038, 2021.

Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alphacode. Science, 378(6624):1092–1097, 2022.

Derrick Lin, James Koppel, Angela Chen, and Armando Solar-Lezama. Quixbugs: a multi-lingual program repair benchmark set based on the quixey challenge. Proceedings Companion of the 2017 ACM SIGPLAN International Conference on Systems, Programming, Languages, and Applications: Software for Humanity, 2017. URL https://api.semanticscholar.org/CorpusID: 7158771.

Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods, 2022. URL https://arxiv.org/abs/2109.07958.

Changshu Liu and Reyhaneh Jabbarvand. A tool for in-depth analysis of code execution reasoning of large language models. arXiv preprint arXiv:2501.18482, 2025.

Chenxiao Liu and Xiaojun Wan. Codeqa: A question answering dataset for source code comprehension. arXiv preprint arXiv:2109.08365, 2021.

Fang Liu, Yang Liu, Lin Shi, Houkun Huang, Ruifeng Wang, Zhen Yang, and Li Zhang. Exploring and evaluating hallucinations in llm-powered code generation. arXiv preprint arXiv:2404.00971, 2024a.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. Advances in Neural Information Processing Systems, 36, 2024b.

Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, Tianyang Liu, Max Tian, Denis Kocetkov, Arthur Zucker, Younes Belkada, Zijian Wang, Qian Liu, Dmitry Abulkhanov, Indraneil Paul, Zhuang Li, Wen-Ding Li, Megan Risdal, Jia Li, Jian Zhu, Terry Yue Zhuo, Evgenii Zheltonozhskii, Nii Osae Osae Dade, Wenhao Yu, Lucas Krauß, Naman Jain, Yixuan Su, Xuanli He, Manan Dey, Edoardo Abati, Yekun Chai, Niklas Muennighoff, Xiangru Tang, Muhtasham Oblokulov, Christopher Akiki, Marc Marone, Chenghao Mou, Mayank Mishra, Alex Gu, Binyuan Hui, Tri Dao, Armel Zebaze, Olivier Dehaene, Nicolas Patry, Canwen Xu, Julian McAuley, Han Hu, Torsten Scholak, Sebastien Paquet, Jennifer Robinson, Carolyn Jane Anderson, Nicolas Chapados, Mostofa Patwary, Nima Tajbakhsh, Yacine Jernite, Carlos Muñoz Ferrandis, Lingming Zhang, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. Starcoder 2 and the stack v2: The next generation, 2024a. URL https://arxiv.org/abs/2402.19173.

Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. Starcoder 2 and the stack v2: The next generation. arXiv preprint arXiv:2402.19173, 2024b.

Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio Blanco, Colin Clement, Dawn Drain, Daxin Jiang, Duyu Tang, et al. Codexglue: A machine learning benchmark dataset for code understanding and generation. arXiv preprint arXiv:2102.04664, 2021.

Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. Wizardcoder: Empowering code large language models with evol-instruct. arXiv preprint arXiv:2306.08568, 2023.

Dung Nguyen Manh, Nam Le Hai, Anh T. V. Dau, Anh Minh Nguyen, Khanh Nghiem, Jin Guo, and Nghi D. Q. Bui. The vault: A comprehensive multilingual dataset for advancing code understanding and generation, 2023. URL https://arxiv.org/abs/2305.06156.

Alexandre Matton, Tom Sherborne, Dennis Aumiller, Elena Tommasone, Milad Alizadeh, Jingyi He, Raymond Ma, Maxime Voisin, Ellen Gilsenan-McMahon, and Matthias Gallé. On leakage of code generation evaluation datasets. arXiv preprint arXiv:2407.07565, 2024.

Lili Mou, Ge Li, Lu Zhang, Tao Wang, and Zhi Jin. Convolutional neural networks over tree structures for programming language processing. In Proceedings of the AAAI conference on artificial intelligence, volume 30, 2016.

Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A Feder Cooper, Daphne Ippolito, Christopher A Choquette-Choo, Eric Wallace, Florian Tramèr, and Katherine Lee. Scalable extraction of training data from (production) language models. arXiv preprint arXiv:2311.17035, 2023.

Minh Huynh Nguyen, Nghi DQ Bui, Truong Son Hy, Long Tran-Thanh, and Tien N Nguyen. Hierarchynet: Learning to summarize source code with heterogeneous representations. arXiv preprint arXiv:2205.15479, 2022.

Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large language model for code with multi-turn program synthesis. arXiv preprint arXiv:2203.13474, 2022.

OpenAI. GPT-3.5-Turbo-16K-0613, 6 2023. URL https://chat.openai.com.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian,

Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774.

Yicheng Ouyang, Jun Yang, and Lingming Zhang. Benchmarking automated program repair: An extensive study on both real-world and artificial bugs. In Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis, pp. 440–452, 2024.

Huy Nhat Phan, Tien N Nguyen, Phong X Nguyen, and Nghi DQ Bui. Hyperagent: Generalist software engineering agents to solve coding tasks at scale. arXiv preprint arXiv:2409.16299, 2024.

Phind. Beating GPT-4 on HumanEval with a Fine-Tuned CodeLlama-34B, 8 2023. URL https: //www.phind.com/blog/code-llama-beats-gpt4.

Nikhil Pinnaparaju, Reshinth Adithyan, Duy Phung, Jonathan Tow, James Baicoianu, Ashish Datta, Maksym Zhuravinskyi, Dakota Mahan, Marco Bellagente, Carlos Riquelme, et al. Stable code technical report. arXiv preprint arXiv:2404.01226, 2024.

Ruchir Puri, David Kung, Geert Janssen, Wei Zhang, Giacomo Domeniconi, Vladimir Zolotov, Julian T Dolby, Jie Chen, Mihir Choudhury, Lindsey Decker, Veronika Thost, Veronika

Thost, Luca Buratti, Saurabh Pujar, Shyam Ramji, Ulrich Finkler, Susan Malaika, and Frederick Reiss. Codenet: A large-scale ai for code dataset for learning a diversity of coding tasks. In J. Vanschoren and S. Yeung (eds.), Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021. URL https:

//datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/ 2021/file/a5bfc9e07964f8dddeb95fc584cd965d-Paper-round2.pdf.

Mirza Masfiqur Rahman and Ashish Kundu. Code hallucination. arXiv preprint arXiv:2407.04831, 2024.

Vyas Raina, Adian Liusie, and Mark Gales. Is llm-as-a-judge robust? investigating universal adversarial attacks on zero-shot llm assessment. arXiv preprint arXiv:2402.14016, 2024.

Joshua Robinson, Christopher Michael Rytting, and David Wingate. Leveraging large language models for multiple choice question answering, 2023. URL https://arxiv.org/abs/ 2210.12353.

Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.

Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. Code llama: Open foundation models for code, 2024. URL https://arxiv.org/abs/2308.12950.

Zhan Shi, Kevin Swersky, Daniel Tarlow, Parthasarathy Ranganathan, and Milad Hashemi. Learning execution through neural code fusion. arXiv preprint arXiv:1906.07181, 2019.

Ben Shneiderman and Richard Mayer. Syntactic/semantic interactions in programmer behavior: A model and experimental results. International Journal of Parallel Programming, 8:219–238, 06

1979. doi: 10.1007/BF00977789.

Dominik Sobania, Martin Briesch, Carol Hanna, and Justyna Petke. An analysis of the automatic bug fixing performance of chatgpt. In 2023 IEEE/ACM International Workshop on Automated Program Repair (APR), pp. 23–30. IEEE, 2023.

Zayne Sprague, Fangcong Yin, Juan Diego Rodriguez, Dongwei Jiang, Manya Wadhwa, Prasann Singhal, Xinyu Zhao, Xi Ye, Kyle Mahowald, and Greg Durrett. To cot or not to cot? chain-ofthought helps mainly on math and symbolic reasoning, 2024. URL https://arxiv.org/ abs/2409.12183.

Weisong Sun, Chunrong Fang, Yudu You, Yun Miao, Yi Liu, Yuekang Li, Gelei Deng, Shenghan Huang, Yuchen Chen, Quanjun Zhang, et al. Automatic code summarization via chatgpt: How far are we? arXiv preprint arXiv:2305.12865, 2023.

Hung Quoc To, Nghi DQ Bui, Jin Guo, and Tien N Nguyen. Better language models of code through

self-improvement. arXiv preprint arXiv:2304.01228, 2023. W3Schools. W3Schools.com, 2024. URL https://www.w3schools.com/quiztest/. Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu

Liu, and Zhifang Sui. Large language models are not fair evaluators, 2023a. URL https: //arxiv.org/abs/2305.17926.

Wenbo Wang, Tien N Nguyen, Shaohua Wang, Yi Li, Jiyuan Zhang, and Aashish Yadavally. Deepvd: Toward class-separation features for neural network vulnerability detection. In 2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE), pp. 2249–2261. IEEE, 2023b.

Wenhan Wang, Chenyuan Yang, Zhijie Wang, Yuheng Huang, Zhaoyang Chu, Da Song, Lingming Zhang, An Ran Chen, and Lei Ma. Testeval: Benchmarking large language models for test case generation, 2024. URL https://arxiv.org/abs/2406.04531.

Yue Wang, Weishi Wang, Shafiq Joty, and Steven CH Hoi. Codet5: Identifier-aware unified pre-trained encoder-decoder models for code understanding and generation. arXiv preprint arXiv:2109.00859, 2021.

Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi DQ Bui, Junnan Li, and Steven CH Hoi. Codet5+: Open code large language models for code understanding and generation. arXiv preprint arXiv:2305.07922, 2023c.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. URL https://arxiv.org/abs/2201.11903.

Jules White, Sam Hays, Quchen Fu, Jesse Spencer-Smith, and Douglas C Schmidt. Chatgpt prompt patterns for improving code quality, refactoring, requirements elicitation, and software design. In Generative AI for Effective Software Development, pp. 71–108. Springer, 2024.

Chunqiu Steven Xia, Yuxiang Wei, and Lingming Zhang. Automated program repair in the era of large pre-trained language models. In 2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE), pp. 1482–1494. IEEE, 2023.

Frank F Xu, Uri Alon, Graham Neubig, and Vincent Josua Hellendoorn. A systematic evaluation of large language models of code. In Proceedings of the 6th ACM SIGPLAN International Symposium on Machine Programming, pp. 1–10, 2022.

Ruijie Xu, Zengzhi Wang, Run-Ze Fan, and Pengfei Liu. Benchmarking benchmark leakage in large language models, 2024. URL https://arxiv.org/abs/2404.18824.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024. URL https://arxiv.org/abs/2407.10671.

Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. Repocoder: Repository-level code completion through iterative retrieval and generation, 2023. URL https://arxiv.org/abs/2303.12570.

Yuhao Zhang, Yasharth Bajpai, Priyanshu Gupta, Ameya Ketkar, Miltiadis Allamanis, Titus Barik, Sumit Gulwani, Arjun Radhakrishna, Mohammad Raza, Gustavo Soares, et al. Overwatch: Learning patterns in code edit sequences. Proceedings of the ACM on Programming Languages, 6 (OOPSLA2):395–423, 2022.

Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. Large language models are not robust multiple choice selectors. In International Conference on Learning Representations, 2024a. URL https://openreview.net/forum?id=shr9PXz7T0.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623, 2023.

Tianyu Zheng, Ge Zhang, Tianhao Shen, Xueling Liu, Bill Yuchen Lin, Jie Fu, Wenhu Chen, and Xiang Yue. Opencodeinterpreter: Integrating code generation with execution and refinement. arXiv preprint arXiv:2402.14658, 2024b.

Eric Zhu, Vadim Markovtsev, Aleksey Astafiev, Chris Ha, Wojciech Łukasiewicz, Adam Foster, Sinusoidal36, Andrii Oriekhov, Joe Halliwell, JonR, Kevin Mann, Keyur Joshi, Michael Joseph Rosenthal, Qin TianHuan, Senad Ibraimoski, Spandan Thakur, Stefano Ortolani, Titusz, Vojtech Letal, Zac Bentley, fpug, hguhlich, long2ice, oisincar, and Ron Assa. ekzhu/datasketch: v1.6.4, October 2023.

Ming Zhu, Aneesh Jain, Karthik Suresh, Roshan Ravindran, Sindhu Tipirneni, and Chandan K Reddy. Xlcost: A benchmark dataset for cross-lingual code intelligence. arXiv preprint arXiv:2206.08474,

- 2022.

Terry Yue Zhuo, Minh Chien Vu, Jenny Chim, Han Hu, Wenhao Yu, Ratnadira Widyasari, Imam Nur Bani Yusuf, Haolan Zhan, Junda He, Indraneil Paul, et al. Bigcodebench: Benchmarking code generation with diverse function calls and complex instructions. arXiv preprint arXiv:2406.15877, 2024.

- A DATASET

- A.1 DATA CLEANING

Rule-based filtering We prefer questions that contain code when collecting data; therefore, MCQs often contain noisy patterns and low-quality questions. In the cleaning process, we defined a heuristic rule-based filter to eliminate incomplete data and non-textual content. First, we detect and eliminate non-textual questions by filtering questions that contain hrefs, image URLs, links to other questions or media. We also applied BeautifulSoup to remove unwanted HTML tags.

Deep learning-based filtering To ensure the CodeMMLU is fully targeted on coding and softwarerelated task, we employed models from OpenAI (GPT-3.5-turbo), Mistral (Mixtral 8×7B Instruct), and Llama (Llama3.1 8B) as our annotators to judge the triple criteria: Completeness; coherence and clarity; and coding relevance (check appendix C.1 to see the prompt). We averaged LLM ratings by category and selected a threshold of 4 in 3 aspects. Result of removing ≈ 25.6% of raw data. On the other hand, we simultaneously sampled a subset of 100 instances in each subject to update our filter rule. (Figure 8

0 1 2 3 4 5 Avg. Rating score

- 100

- 101

- 102

- 103

Logof#Rating

Completeness Coherence Relevance

| |
|---|

| |
|---|

Figure 8: LLM-based filter score distribution.

Execution-based filtering After synthesizing the fundamental task’s distractor (i.e., false answer), we concatenated and executed them as a complete function in an isolated environment. The code completion and fill-in-the-blank tasks have their original test cases, while code repair needs to synthesize new test cases. Therefore, we extracted the method signature (using the code-text parser toolkit from (Manh et al., 2023)) and synthesized the function input, which later was executed to create test cases for the corresponding function. We ran in parallel the distractor executing on the testcase and synthesized a new distractor, ensuring the distractor collection is executable and able to pass 0-50% test cases.

- A.2 DATA CONTAMINATION

The development of large language models (LLMs) often involves crawling data from diverse sources across the internet, with limited transparency regarding their preprocessing. Given the vast and often proprietary nature of these training datasets, it is widely acknowledged that creating a fully leakage-free benchmark is virtually impossible. While recent benchmarks have recognized this issue and generally accept that avoiding data leakage entirely is extremely difficult, one common mitigation strategy involves filtering data based on its timeline (Jain et al., 2024; Wang et al., 2024; Zhang et al.,

- 2023).

In our efforts to address this challenge, we acknowledge the complexity of completely eliminating data leakage. To enhance the reliability of CodeMMLU, we adopt proactive measures during the data creation process. Specifically, we transform seed data into multiple-choice question formats and introduce synthetic distractors. For tasks like code repair and execution prediction, the test sets were extracted from codebase seeds and modified to align with specific task requirements.

To further assess and quantify potential data leakage, we employ the methodology outlined in (Xu et al., 2024). This includes calculating perplexity and conducting n-gram analysis on several wellknown models from diverse families (e.g., Mistral, DeepSeek, Llama). The results, presented in Tables 5 and 6, highlight a significant margin between CodeMMLU and other coding benchmarks, reinforcing the reliability of CodeMMLU as a robust evaluation tool.

Perplexity measures the uncertainty of a language model when predicting the next token in a sequence (Jelinek et al., 1977). Therefore, as low as the perplexity score indicates, the model is confident in predicting the evaluating sequence and the more likely that the model was encountered during the training process. Perplexity is expressed as the exponentiated average negative loglikelihood of a sequence:

1 t

PPL(X) = exp −

t

logpθ(xi|x<i) (1)

t=0

where X = [x0,x1,...,xt] denotes a tokenized sequence.

N-gram Accuracy (Xu et al., 2024) is a metric designed to detect fine-grained data leakage at the instance level by combining the question and answer into a single text (X), uniformly sampling starting points, and predicting the next n-grams based on the given prompts. If most n-grams are accurately predicted, it suggests the model may have encountered the data during training. The N-gram accuracy can be expressed as:

S

1 S · K

N-gram Accuracy(X) =

i=0

K

j:startj+n,Xˆstart

j:startj+n), (2)

I(Xstart

j=0

where S is the dataset size, K is the number of sampled starting points, Xstart

j:startj+n is the actual n-gram, Xˆstart

j:startj+n is the predicted n-gram, and I checks for exact matches. Author add ROUGE-L and edit distance similarity to provide robustness for augmented datasets. A high accuracy for each n-gram in a prediction indicates a strong likelihood that the sample was seen during the training process. (Xu et al., 2024)

- Table 5: Perplexity score comparison between coding benchmark. (higher is better) Models CodeScope CodeApex CodeMMLU

Mistral7B-v0.3 9.32 16.08 16.32 DeepSeekCoder7B-v1.5 5.26 9.39 57.36

DeepSeekV2-Lite 6.89 11.99 1419.48 Llama-3.1-8B 10.05 123.20 197.31

- Table 6: 5-gram accuracy comparison between coding benchmark. (lower is better)

Models CodeScope CodeApex CodeMMLU Mistral7B-v0.3 0.2510 0.1702 0.1365

DeepSeekCoder7B-v1.5 0.2818 0.1680 0.1416 DeepSeekV2-Lite 0.2492 0.1587 0.0687 Llama-3.1-8B 0.2219 0.1309 0.0652

- A.3 LICENSE

In the construction of CodeMMLU, we collect only the multiple-choice questions, problem descriptions, code solutions, and test cases from the publicly visible parts of W3School and Geeksforgeeks quizzes/puzzles and LeetCode. We avoid any data collection that requires login or interaction with these websites. On one hand, most of our knowledge test set (≈ 61%) are collected from Common

Crawl (from portion tagged CC-MAIN-2021-41 to CC-MAIN-2024-46). On the other hand, the fundamental tasks were created on a permissively licensed codebase, namely IBM Project CodeNet (Apache 2.0), HumanEval, QuixBugs (MIT). For data crawled from websites such as W3Schools (fair use for research purposes) and GeeksforGeeks (under the Copyright Act 1957), we fully complied with their copyrights or sought their permission to use such data for this project. CodeMMLU will be published and distributed under the MIT license.

- B ALL EXPERIMENTAL RESULTS

- Table 7: CodeMMLU and other coding benchmarks comparison. The ranking reorder comparison between CodeMMLU (CM) and other benchmarks (namely HumanEval (HE)).

Family Model Size (B) MMLU GSM8k HumanEval MBPP CodeMMLU HE→CM

Closed-source models Anthropic Claude-3 Sonnet - 88.70 96.40 92.00 76.6 55.48 1→4 OpenAI

GPT-4o - 88.70 95.80 90.20 81.4 64.96 2→1

GPT-3.5-turbo - 61.90 73.80 61.40 78.5 51.59 10→6 Open-source models

Llama3.1 70B Instruct 70 83.60 95.10 80.50 75.4 59.68 6→3

Llama3.1 70B 70 79.30 83.70 58.50 66.2 40.45 11→20

Llama3 70B 70 79.50 83.00 48.20 70.4 49.7 14→8

MetaLlama

Llama3 70B Instruct 70 82.00 93.00 81.70 82.3 61.79 4→2 CodeLlama 34B Instruct 34 - - 41.50 57 39.27 17→21

Mistral 7B Instruct (v0.3) 7 62.50 50.00 26.20 50.2 44.14 21→17

Mixtral 8x7B Instruct 46.7 70.60 74.40 40.20 60.7 42.74 18

Mistral

Codestral 22B 22 - - 81.10 78.2 47.61 5→13

Phi3 Medium 128k Instruct 14 78.00 91.00 62.20 75.2 48.65 9

Phi

Phi3 Mini 128k Instruct 3.8 68.80 82.50 58.50 70 39.22 11→22

Qwen2 7B Instruct 7 70.50 82.30 79.90 - 51.86 7→5

Qwen2 57B-A14B Instruct 57 76.50 80.70 53.00 71.9 47.34 12→14

Qwen

CodeQwen1.5 7B Chat 7 - - 83.50 77.7 47.71 3→12

Yi1.5 34B Chat 34 67.62 71.70 23.20 41 50.03 22→7

Yi

Yi1.5 9B Chat 9 68.40 52.30 39.00 54.4 48.15 19→10

DeepSeek Coder 7B Instruct (v1.5) 7 49.20 41.00 42.10 60.7 41.59 16→19

DeepSeek Coder 33B Instruct 33 - 60.70 79.30 70 37.45 8→23

DeepSeek

DeepSeek Moe 16B Chat 16.4 45.00 18.80 26.80 39.2 31.45 20→24

DeepSeek CoderV2 Lite Instruct 16 60.10 86.40 81.10 - 47.12 5→15

InternLM InternLM2.5 20B Chat 20 66.50 79.60 48.80 63 46.15 13→16 StarCoder StarCoder2 15B Instruct 15 - - 46.3 66.2 47.76 15→11

- B.1 SELECTION BIAS IN MCQS FORMAT

Building on the findings from (Zheng et al., 2024a), which investigated the effects of reordering answer options in multiple-choice questions (MCQs), we observe inconsistent behavior among large language models (LLMs) when performing the same code completion task. Table 8 highlights the sensitivity of LLMs to the order of answers, even for models renowned for their high performance (e.g., GPT, Claude, MetaLlama). Specifically, the results reveal that most models experience significant performance degradation when the correct answer is positioned as “A”, with an average performance drop of 25%. In contrast, placing the correct answer in position “B” leads to a marked performance improvement, with an average increase of 15.49%.

The standard deviation (STD) further illustrates how differently models respond to answer reordering. For instance, models such as CodeLlama-7B/13B/34B and DeepSeekCoder-33B exhibit substantial dependency on the arrangement of options, whereas models like GPT-4o/3.5, Claude-3, and Claude3.5 show greater resilience to such selection bias. Interestingly, instruction-tuned models, which are generally expected to demonstrate increased robustness, show minimal to no improvement over their base versions in this regard.

These findings suggest that higher-quality models are more resistant to MCQ biases, reflecting a human-like ability to maintain performance irrespective of answer order. We believe that introducing this MCQ bias into the CodeMMLU benchmark adds an extra layer of difficulty for LLMs, encouraging the research community to prioritize enhancing the consistency and robustness of LLMs.

[Figure 8]

Figure 9: Task-Specific Accuracy and Performance Fluctuations Across Answer Options Models exhibit marked fluctuations in accuracy depending on the position of the correct answer in Code Completion in CodeMMLU. Revealing the bias and inconsistencies in related coding multiple-choice question (MCQ) task and how sensitive LLMs are to answer ordering.

- Table 8: Selection bias effect comparison on LLMs. The performance fluctuation trends show a significant margin of model with high quality and the other. STD stands for standard deviation.

Models Instructed A B C D STD GPT-4o ✓ 80.49 78.05 71.34 70.12 4.38

GPT-3.5-turbo ✓ 51.22 43.29 47.56 54.88 4.30

Claude3.5 Sonnet ✓ 90.24 81.1 85.37 79.27 4.23 Claude3.5 Haiku ✓ 86.59 69.51 72.56 68.29 7.30

Claude3 Opus ✓ 79.27 77.44 82.32 84.76 2.81 Claude3 Sonnet ✓ 62.8 64.02 73.17 73.78 5.06 Claude3 Haiku ✓ 56.1 75 73.78 76.83 8.34

Mixtral 8x7B ✓ 22.56 74.39 71.95 63.41 20.91 DSCoder 33B - 1.22 82.32 75.00 56.10 31.75

DSCoder 7B - 40.85 74.39 64.02 39.02 15.10 Phind-CL 34B ✓ 6.10 90.85 75.00 46.34 32.21

CL 34B Python - 0.61 77.44 70.73 49.39 30.09 CL 34B Instruct ✓ 9.15 84.76 65.24 46.34 27.91 CL 13B Python - 0.61 54.88 70.12 12.20 28.85 CL 13B Instruct ✓ 2.44 68.29 72.56 29.88 28.85

CL 7B Python - 0.00 90.24 14.02 0.61 37.39 CL 7B Instruct ✓ 3.66 1.22 93.90 15.85 38.07

### B.2 COT MIGHT NOT BE ALL YOU NEED

In our experiments with models from over 15 families, we evaluated CodeMMLU under two different prompt settings: standard zero-shot and few-shot, as well as Chain-of-Thought (CoT) with short and long prompts. The detailed results, provided in B.3, reveal a consistent trend of decreased performance in the CoT setting compared to zero-shot and few-shot configurations. Even strong models like GPT-4o and Llama3 70B, known for their robust reasoning and comprehensive capabilities, exhibited this pattern, as illustrated in Figure 13.

Comparison between prompts (GPT-4o)

Distinct corrects compared to zeroshot

Chain-of-Thought

(28/64 correct)

Fewshot

(31/64 correct)

Zeroshot

(34/64 correct)

#correct

overlap

10 20 30

Figure 10: Comparison experiment of different prompts by GPT-4o in OOP subject.

A closer analysis of results for zero-shot, few-shot, and CoT prompts, particularly in the GPT-4o experiment on the object-oriented programming (OOP) subject (Figure 10), highlights the effects of an additional “thinking” step introduced by CoT. While this step intended to enhance reasoning, the thought often diverged from the original focus or brought up another challenge for the task. For instance, example 1 in Figure 14 showcases a case where the CoT approach incorporated an unnecessary definitional analysis but failed to arrive at the correct answer. Similarly, examples 2 and 3 (Figure 14) demonstrate misinterpretations during the reasoning process. This additional “thinking” step often introduced new challenges that might lead to incorrect conclusions, resulting in a significant performance gap between zero-shot and CoT prompts.

Interestingly, when visualizing the newly addressed questions by CoT settings (compared to 0-shot), we do not observe a clear pattern of these new corrects. Questions that CoT addressed were neither inherently tricky nor required deeper interpretive reasoning. For example, example 4 (Figure 14) asks for a prediction of program outcomes within the OOP domain, comparable in scope to example 3, yet the CoT approach showed inconsistent performance.

As concluded by the findings in (Sprague et al., 2024), the CoT strategy introduces an additional reasoning step that is effective primarily for tasks involving mathematical or logical problem-solving. However, for benchmarks like MMLU, the results with and without CoT prompting are nearly identical. The observed performance decline in the CodeLLM knowledge test set aligns with (Sprague et al., 2024), further underscoring the inefficiency of CoT for non-reasoning tasks.

### B.3 EXPERIMENT RESULTS We provide the full experiment results of 43 LLMs from 15 families in Table 9, 10, 11.

|Question|
|---|
|Question: Which of the following is another name for class components?<br><br>(A) Stateful components<br>(B) Stateless Components<br>(C) Props<br>(D) State<br>|

|DeepSeek R1|
|---|
|<think> Okay, let me see... The question is about the correct use of useLayoutEffect in React. I remember that both useEffect and useLayoutEffect are hooks for handling side effects in React components, but there's a key difference between them. From what I recall, useEffect runs after the component has rendered and the screen has been updated. That's the usual case. On the other hand, useLayoutEffect is designed to run synchronously after all DOM mutations but before the browser paints the screen. So the main point is about when they fire. UseLayoutEffect's purpose is to make changes that need to be visible to the user immediately, without causing a flicker. Looking at the options:<br><br>(A) Optimize for all devices \u2013 Hmm, optimization might be a side effect of correctly handling layout, but I don't think that's the direct purpose of useLayoutEffect.<br>(B) Complete all the update \u2013 Not sure. Maybe refers to batching updates? Or ensuring updates are finished?<br>(C) Change the layout \u2013 That sounds more like a possible use case. If you need to change the layout, and want to do it before the browser paints, useLayoutEffect would be appropriate here.<br>(D) Paint before the effect runs \u2013 Wait, useLayoutEffect runs before paint. So this option seems incorrect. The normal useEffect runs after paint. If D says \"Paint before the effect runs\" then in the case of useLayoutEffect, the effect runs before painting, so the painting happens after. So D's description might fit useEffect instead. Wait, the answer options are for what useLayoutEffect is correctly used for The documentation says useLayoutEffect fires synchronously after all DOM mutations. That is, after React has updated the DOM, but before the browser paints those changes. So the use case for useLayoutEffect is when you need to make DOM changes (like measuring layout or modifying the DOM) and you don't want the user to see a flicker between the initial paint and the updated state. So one of the correct reasons to use useLayoutEffect is to read layout from the DOM and then make changes to it synchronously. So options that relate to changing the layout (C) or ensuring that effects run before paint to prevent flickering. Looking back at the options: Option D is \"Paint before the effect runs\". No, because useLayoutEffect's effect runs before the paint. So D's statement is the opposite. Option B, complete all the update, maybe that refers to completing updates before the paint. But not sure. Option C is \"Change the layout\", which could be correct if you need to make a layout change that's done before the browser paints. So maybe the correct answer is C or D, but I need to think. Wait the answer options: <<CONTINUE REASONING ABOUT ~5400 TOKENS>><br><br><br>...Therefore, option (C) is correct. </think> The correct use of `useLayoutEffect` is to perform DOM mutations or read layout properties that need to be reflected synchronously before the browser paints the screen. This prevents users from seeing intermediate states or flickering.<br><br>**Analysis of Options:**<br><br>- **(A) Optimize for all devices**: Incorrect. Optimization is not directly tied to `useLayoutEffect`.<br>- **(B) Complete all the update**: Vague and not specific to the purpose of `useLayoutEffect`.<br>- **(C) Change the layout**: **Correct**. `useLayoutEffect` is used to modify the DOM (e.g., adjust styles, element positions) before the browser paints, ensuring changes are immediate.<br>- **(D) Paint before the effect runs**: Incorrect. `useLayoutEffect` runs *before* painting, not after.<br><br><br>**Answer:** (C) Change the layout<br><br>[Figure 9]|

|GPT o1|
|---|
|The correct answer is: (C) Change the layout Explanation: In React, the `useLayoutEffect` Hook is used when you need to perform side effects that read or write layout information (like getting the position or size of DOM elements) synchronously after all DOM mutations but before the browser has a chance to paint. This can be crucial when you need to make visual changes that must happen before the browser updates the screen to prevent visual discrepancies or flickering. Here's how the options relate to `useLayoutEffect`:<br><br>- (A) Optimize for all devices: This is too vague and not specifically related to the purpose of `useLayoutEffect`.<br>- (B) Complete all the update: While `useLayoutEffect` runs after DOM updates, it's more about reading and synchronizing layout changes before painting rather than just completing updates. \u2705 (C) Change the layout: This is the best choice. `useLayoutEffect` is intended for operations that need to read or alter the DOM layout before the browser paints the screen.<br>- (D) Paint before the effect runs: This is the opposite of what actually happens. `useLayoutEffect` runs **before** the browser paints, not after. Note: It's generally recommended to use `useEffect` for most side effects to avoid blocking visual updates. Use `useLayoutEffect` sparingly for cases where you need to perform measurements or mutations that affect layout and need to happen before painting. Answer: (C) Change the layout<br><br><br>[Figure 10]|

|GPT o3-mini|
|---|
|The correct answer is D. Explanation: React 2019s useLayoutEffect hook is designed to run synchronously after all DOM mutations but before the browser actually repaints the screen. This means it can be used to perform tasks like measuring the layout or synchronously updating the DOM to avoid visual inconsistencies (flicker) during the rendering process. In other words, any changes that need to happen before the browser paints should be handled in useLayoutEffect, which is why option D ("Paint before the effect runs") best captures that behavior.<br><br>[Figure 11]|

### Figure 11: Comparison reasoning process of GPT o1, GPT o3-mini and DeepSeek R1 on knowledgeseeking task.

|Question|
|---|
| |
|The following are multiple choice questions (with answers) about programming problem.<br><br>Question: Given a code snippet below, which behavior most likely to occur when execute it? ```import java.util.*;<br><br>public class Main {<br><br>public static void main(String[] args) { Scanner sc = new Scanner(System.in); int num = Integer.parseInt(sc.next());<br><br>long[] inputs = new long[num]; long sum = 0; for (int i = 0; i < num; i++) {<br><br>inputs[i] = Long.parseLong(sc.next()); sum ^= inputs[i];<br><br>} // only interested in 1's bit for (int i = 0; i < num; i++) {<br><br>inputs[i] &= ~sum;<br><br>} System.out.println(sum + (maxSubarrayXOR(inputs) << 1));<br><br>}<br><br>public static long maxSubarrayXOR(long A[]) { int idx = 0; for (int bit = 60; bit >= 0; bit--) {<br><br>int find = idx;<br><br><<QUESTION ID rt08365>><br><br>return ret; }<br><br>```<br><br>Answer:<br><br>A. Time Limit Exceeded<br>B. Compile Error<br>C. Memory Limit Exceeded<br>D. No abnormally found<br>|

|DeepSeek R1|
|---|
|<think> Okay, let's see. So I've got this Java code here, and the question is about what happens when you execute it. The options are about various runtime or compile issues, including no error. Hmm. First, I should read through the code and see if there's anything that jumps out as an error. Let's start with the main method. The code reads some input using Scanner. Then it parses an integer 'num', which is the size of an array 'inputs'. Then it reads 'num' numbers into the array, calculating the sum as the XOR of all elements. Then, there's a loop that iterates over each element in 'inputs' and performs a bitwise AND with the bitwise NOT of sum. So each inputs[i] is being set to inputs[i] & ~sum. That leaves only the bits where sum had a 0, since ~sum flips all the bits. But since sum is the XOR of all elements, that has something to do with their combined parity. But maybe that's part of an algorithm, like maybe finding maximum XOR subarrays? The code then calls maxSubarrayXOR on the modified array, shifts the result left by one, adds it to sum, and prints. Now looking at the maxSubarrayXOR method. Oh right, this looks like it's trying to implement Gaussian elimination for the basis of XOR vectors. Let's think. The method starts with an index of 0, then loops from the highest bit (60) down. For each bit, it's trying to find a number with that bit set starting from the current index. If it finds one, it swaps with the current index (using XOR swaps maybe?), then eliminates that bit from all other numbers. That's standard for Gaussian elimination over GF(2) to find the basis. Then after all bits are processed, it XORs all the elements in the basis to get the maximum XOR. But wait, since they might have done the elimination, maybe that correctly gives the maximum subset XOR. Wait, but in the code, when they swap elements, are they actually using XOR swaps correctly here? Let's look: the code swaps A[idx] and A[find]. The way they do it is: A[idx] ^= A[find]; A[find] ^= A[idx]; A[idx] ^= A[find]; But wait, the typical XOR swap between two variables x and y is done as:<br><br>x ^= y;<br>y ^= x; x ^= y; Which swaps x and y. But the code here does the same using the initial and new values. Wait, but in the code, after the first line, A[idx] becomes A[idx] ^ A[find]. Then the second line is A[find] ^= A[idx], which is now the new A[idx] (so original a[idx] XOR a[find] XOR that would be what?), but perhaps during the swap between two array elements, unless they are the same element, this works. Though if idx == find, which they check before doing the swap, so they skip that case. So if idx != find, the swap code is correct. Wait, but the way that code works, after the swap, find is set to idx and then the loop continues. Maybe that's correct. Now, the possible bugs here: When dealing with Gaussian elimination, after selecting the pivot element (the one with the highest bit set), we want to eliminate that bit from all other vectors. The code checks for each i not equal to find (but here find is updated to idx?), maybe there's an off-by-one error. Let's look: after the swap, find becomes idx. Wait, the code says: if (idx != find) {<br><br><br>// perform the swaps as mentioned, then sets find = idx; ??? Wait in the code after swapping, they have: A[idx] ^= A[find]; A[find] ^= A[idx]; A[idx] ^= A[find]; find = idx; Wait no, the swap is in three steps. Let's see: Original A[idx] is 'x', A[find] is 'y'. First line: x becomes x ^ y. Second line: y becomes y ^ (x^y) = x. Third line: new x (x^y) becomes (x^y) ^ x = y. So after all, A[idx] is y, A[find] is x. Then, after swapping, they set find = idx. So after swapping, find is equal to idx. Then they proceed to eliminate the bit from all other elements. Wait, but in the code after swapping, the code's for loop iterates through all values of i, and checks if i != find, then if the bit is set in A[i], it XORs A[i] with A[find] (which is now the element at idx). So that part seems correct since the current pivot is at idx. Hmm. Another possible mistake is in the initial loop conditions. The code for the bits runs from 60 down to 0. For each bit, in the while loop, starting from 'find = idx', but is that correct? Because maybe the maximum subarray algorithm requires processing the elements in order, and once an element is used as a pivot, it's left alone. So it keeps increasing idx each time a pivot is found. This seems correct. But what's the scenario here? The code in the maxSubarrayXOR function is trying to compute the maximum subset XOR, perhaps, and then returns it, which is shifted left by one and added to sum. Another part that might be problematic: after modifying the array A, which is done in-place, the final loop is just XORing all the elements. Wait, but in some cases, after Gaussian elimination, some elements could be zero. The maximum XOR is the XOR of the basis vectors, which in this procedure might be stored in the first elements up to idx. So XORing all the elements may not give the correct result. However, maybe with how the elimination is done, the non-zero elements contribute. For example, all the non-zero elements in the array after elimination form a basis, but the code XORs all of them to get the maximum. But I remember that in some versions of this, you build the basis and then greedily choose to include elements to maximize the result, maybe. If they're processed from highest bit to lowest, maybe XORing all gives the maximum sum. Not entirely sure. But if the Gaussian elimination is correctly implemented in the code, that's probably okay. Alternatively, maybe there's a bug in how the for loop variables are used. But the problem here is the possible options: which of the four outputs (TLE, Compile Error, MLE, or no issue) happens when this code is executed. Looking through the code for syntax errors. In Java, the syntax seems okay. Let me scan. The code starts with import java.util.*; which is valid. The Main class has a public static main method, and the code is in a class. So no problem there. Inside maxSubarrayXOR, the code uses a for loop, variables. The code here uses the method parameter long A[], which is okay. The code uses bit manipulations like 1L << bit (since bit can be up to 60, 1L is a long, so that's correct. Since in Java shifting 1 (as int) might not get higher than 31 bits, but 1L << 60 is a long, which is correct. The code looks like it's doing the Gaussian elimination correctly in terms of syntax. No obvious syntax errors here. The only possibility for a compile error might be if, for instance, some variable is used out of scope. Wait, in the main method, in the loop: for (int i = 0; i < num; i++) {<br><br>inputs[i] &= ~sum;<br><br>} But 'sum' is of type long. The bitwise complement (~sum) is a long, which when ANDed with inputs[i] (a long) is okay. So that's valid Java. So the code would compile. So option B (Compile Error) is out. Now, runtime issues: Could it be TLE (timeout), MLE, or something? Well, for the maxSubarrayXOR function, the algorithm seems to have a time complexity. Let's see: Outer loop runs from bit 60 downto 0: 61 bits. For each bit, it performs a loop that could scan up to A.length elements (in the 'find' loop), then in the worst case for each bit, and then another loop over all elements to eliminate the bit. So the total time is O(number_of_bits * n). Given that number_of_bits is 61 (since 0 to 60), and n is the length of the array, which is up to the input's num. If num is large, 1e5 for example, then 61*1e5 is 6.1e6 operations\u2014manageable. If the input is large, like 1e5, this would be acceptable. However, if in Java that is acceptable for the time constraints (assuming the programming problem might have tight time limits), perhaps. But the code as presented seems like it's performing a Gaussian elimination which is O(n * bits), which is feasible in Java for n up to 1e5 and bits 60. But wait: the code reads inputs as all lines. How many elements are being passed in depends on the input. But perhaps in a scenario where num is up to, say, 1e4 or 1e5, the code might not run into TLE. But without knowing the test cases, perhaps it's more about possible logical errors leading to infinite loops" <<UNABLE TO CONCLUDE THE FINAL ANSWER>><br><br>[Figure 12]|

|GPT o1|
|---|
|The correct answer is (B) Compile Error.<br><br>The code snippet contains an incorrect implementation of the XOR swap algorithm. The XOR swap is intended to swap two variables without using a temporary variable. However, it can lead to issues in<br><br>high-level languages like Java because it assumes that variables are stored in distinct memory locations and that there are no side effects during operations.<br><br>In the code, the swapping is done using: ```java A[idx] ^= A[find]; A[find] ^= A[idx]; A[idx] ^= A[find]; find = idx;```<br><br>The problem with this swapping method in Java is that it may not work as intended due to Java's abstraction of memory management and potential optimizations performed by the JVM. Additionally, using bitwise operations on `long` types can cause unexpected behavior if not handled carefully.<br><br>To fix the compile error, you should use a temporary variable to swap the elements: ```java long temp = A[idx]; A[idx] = A[find]; A[find] = temp;``` This method is clear, efficient, and less error-prone. Answer: (B) Compile Error<br><br>[Figure 13]|

|GPT o3-mini|
|---|
|The answer is (D) No abnormally found. Explanation: The provided code snippet is a Java implementation that reads input, performs bitwise XOR operations, and computes the maximum subarray XOR using a Gauss-Jordan like elimination approach on the<br><br>bit representation of numbers. There is no inherent flaw such as causing time limit, compile, or memory limit errors under normal circumstances. Hence, the code executes as expected without<br><br>triggering any abnormal behaviors.<br><br>[Figure 14]|

### Figure 12: Comparison reasoning process of GPT o1, GPT o3-mini and DeepSeek R1 on problem solving task

PL syntax

API & Frameworks

###### DBMS & SQL

90

42

65

80

40

60

38

55

70

Performance

Performance

Performance

36

50

60

34

45

32

40

50

30

35

GPT-4o

GPT-4o

GPT-4o

40

Meta-Llama-3-70B

Meta-Llama-3-70B

Meta-Llama-3-70B

28

30

Zero Few CoT Type

Zero Few CoT Type

Zero Few CoT Type

Software principles

Others

Code Completion

GPT-4o

60

60

Meta-Llama-3-70B

80

55

55

70

Performance

Performance

Performance

50

60

50

50

45

45

40

40

40

GPT-4o

GPT-4o

30

35

Meta-Llama-3-70B

Meta-Llama-3-70B

35

Zero Few CoT Type

Zero Few CoT Type

Zero Few CoT Type

Fill in the middle

Code Repair

Defect Detection

90

GPT-4o

60

Meta-Llama-3-70B

80

50

80

70

40

Performance

Performance

Performance

70

GPT-4o

30

Meta-Llama-3-70B

60

60

20

50

10

50

GPT-4o

40

Meta-Llama-3-70B

0

Zero Few CoT Type

Zero Few CoT Type

Zero Few CoT Type

- Figure 13: Comparison between GPT4o and Meta Llama-3 70B on various prompt settings. We experiment with zero-shot, 1-shot, and CoT prompt configuration, where the result indicates the ineffectiveness of CoT in boosting the models’ performance. Comparing to zeroshot config, 1-shot prompt slightly increase the performance in knowledge tasks but falls shorter in real tasks.

|Question|
|---|
|Question: Converting a primitive type data into its corresponding wrapper class object instance is called<br><br>(A) Boxing<br>(B) Wrapping<br>(C) Instantiation<br>(D) Autoboxing<br>|

|Zeroshot|
|---|
|(D) Autoboxing<br><br>[Figure 15]|

|Chain-of-Though|
|---|
|To determine the correct answer, let's break down each option and understand what they mean in the context of software development:<br><br>(A) Boxing: This term is used to describe the process of converting a primitive type into its corresponding wrapper class object. For example, converting an `int` to an `Integer`.<br>(B) Wrapping: While this term might sound relevant, it is not commonly used in this specific context. It generally refers to enclosing or encapsulating something within another layer.<br>(C) Instantiation: This term refers to creating an instance of a class. It does not specifically relate to converting primitive types into their wrapper classes.<br>(D) Autoboxing: This is a more specific term that describes the automatic conversion that the Java compiler makes between primitive types and their corresponding object wrapper classes. For example, automatically converting an `int` to an `Integer`. Given these definitions:<br><br><br>- "Boxing" (A) correctly describes manually converting a primitive type into its corresponding wrapper class.<br>- "Autoboxing" (D), however, specifies that this conversion happens automatically by the compiler. Since we are looking for just "converting," without specifying whether it's manual or automatic:<br><br><br>Answer: (A) Boxing|

|Question|
|---|
|Question: Which among the following is correct for the following code? class A {<br><br>public : class B {<br><br>public : B(int i): data(i) { } int data;<br><br>}<br><br>}; class C: public A {<br><br>class D:public A::B{ }; };<br><br>(A) Multi-level inheritance is used, with nested classes<br>(B) Multiple inheritance is used, with nested classes<br>(C) Single level inheritance is used, with enclosing classes<br>(D) Single level inheritance is used, with both enclosing and nested classes<br>|

|Chain-of-Though|
|---|
|Let's analyze the code step by step to determine which type of inheritance and class structure is being used.<br><br>1. Class A:<br><br>- Contains a nested public class `B`.<br>- Class `B` has a constructor that initializes an integer member variable `data`. 2. Class C:<br>- Inherits publicly from class `A`.<br>- Contains a nested public class `D` that inherits publicly from `A::B`.<br><br>Now, let's break down the inheritance and nesting:<br><br>- Nested Classes:<br>- Class B is nested within Class A.<br>- Class D is nested within Class C.<br>- Inheritance Relationships:<br>- Class C inherits from A (single level).<br>- Nested class D in C inherits from B in A (multi-level).<br><br>Given these points, we can evaluate each option:<br><br>(A) Multi-level inheritance is used, with nested classes<br><br>- This seems correct because there are multiple levels of inheritance involved: C -> A (single level) D -> B (nested multi-level)<br><br>(B) Multiple inheritance is used, with nested classes<br><br>- This isn't correct because no single class directly inherits from more than one base class.<br><br>(C) Single level inheritance is used, with enclosing classes<br><br>- This isn't entirely accurate since there are multiple levels of nesting and some multi-level relationships.<br><br>(D) Single level inheritance is used, with both enclosing and nested classes<br><br><br><br><br>- While it mentions both enclosing and nesting correctly, it incorrectly states "Single level" when there's also multi-level involved due to the relationship between D and B through their respective parent classes.<br><br>Therefore,<br><br>The correct answer should be:<br><br>(A) Multi-level inheritance is used, with nested classes<br><br>[Figure 16]|

|Zeroshot|
|---|
|(D) Single level inheritance is used, with both enclosing and nested classes<br><br>Explanation:<br><br>- The code defines a class `A` which contains a nested class `B`.<br>- Class `C` inherits from class `A`, making it single-level inheritance.<br>- Within class `C`, there is another nested class `D` that inherits from the nested class `B` of the base class.<br><br><br>Therefore, this scenario involves single-level inheritance (class C inheriting from A) and includes both enclosing (class A containing B, and C containing D) and nested classes.<br><br>[Figure 17]|

|Question|
|---|
|Question: What is the output of the following program?<br><br>class A {<br><br>protected: int a,b; public: void disp() {<br><br>cout&lt;&lt;a&lt;&lt;b; }<br><br>};<br><br>class B:public A {<br><br><br>int x,y; };<br><br>(A) Garbage value<br>(B) Compile time error<br>(C) Runtime error<br>(D) Runs but gives random values as output<br>|

|Chain-of-Though|
|---|
|To determine the output of the given program, let's analyze it step by step:<br><br>1. Class Definition:<br><br>- `class A` has two protected integer members `a` and `b`.<br>- It also has a public member function `disp()` that prints the values of `a` and `b`.<br><br><br>2. Inheritance:<br><br>- `class B` publicly inherits from class A.<br>- Class B introduces two additional private integer members, `x` and `y`.<br><br><br>3. Functionality:<br><br>- The function in class A (`disp()`) is designed to print the values of its protected members (`a`, and `b`).<br><br>4. Output Analysis:<br><br>- Since there are no constructors or any other functions initializing or setting values for variables 'a' and 'b', they will contain whatever data happens to be at their memory locations when an object of class B (or even class A) is created.<br><br>5. Compilation Check:<br><br>- There are no syntax errors in this code snippet; hence it will compile successfully.<br><br>6. Runtime Behavior:<br><br>- When you call the disp() method on an instance of either class A or B without<br><br>initializing 'a' and 'b', they will hold garbage values (uninitialized local variables).<br><br>Given these points, we can conclude:<br><br>- The program compiles successfully.<br>- At runtime, since 'a' and 'b' are not initialized before being printed, they will display garbage/random values.<br><br><br>Therefore, considering all steps above: The correct answer is: (D) Runs but gives random values as output<br><br>[Figure 18]|

|Zeroshot|
|---|
|The correct answer is: (B) Compile time error Explanation: The program will result in a compile-time error because the `cout` statement inside the `disp()` method of class `A` is using the variables `a` and `b`, which are protected members. However, these variables have not been initialized or assigned any values before being used in the output statement. Additionally, there might be an issue with including necessary headers (like `<iostream>`) and using the appropriate namespace (`std::`).<br><br>[Figure 19]|

|Question|
|---|
|Question: What is the output of the following code?<br><br>package pack1;<br><br>class A {<br><br>public A() {<br><br>System.out.print(“object created”); }<br><br>} package pack2; import pack1.*;<br><br>class B {<br><br><br><br><br>A a=new A(); }<br><br>(A) Output is: object created<br>(B) Output is: object createdobject created<br>(C) Compile time error<br>(D) Run time error<br>|

|Chain-of-Though|
|---|
|Sure, let's break it down step by step.<br><br>1. Understanding the Code Structure:<br><br>- There are two packages: `pack1` and `pack2`.<br>- In `pack1`, there is a class `A` with a constructor that prints "object created".<br>- In `pack2`, there is a class `B` which creates an instance of class `A`.<br><br><br>2. Code Analysis: ```java<br><br>package pack1;<br><br>class A { public A() {<br><br>System.out.print("object created"); }<br><br>} package pack2; import pack1.*;<br><br>class B { A a = new A();<br><br><br><br><br>} ```<br><br>3. Compilation and Execution Steps:<br><br>- When you compile the code, both packages (`pack1` and `pack2`) need to be compiled.<br>- The import statement in package 2 allows access to classes in package 1.<br><br><br>4. **Expected Output or Errors:**<br><br>- If everything compiles correctly, creating an instance of class B will result in creating<br><br>an instance of class A due to the line: ```java A a = new A(); ``` This will invoke the constructor of Class A which prints "object created".<br><br>5. Possible Issues Leading to Errors:<br><br>- If there's any issue with accessing classes across packages (e.g., if they are not properly compiled or if there's some visibility issue), it could lead to either compile-time or runtime errors.<br><br>6. Checking for Compile-Time Errors: Since all necessary imports are done correctly and assuming no other issues like file<br><br>structure problems exist, there should be no compile-time error.<br><br>7. Checking for Runtime Errors: Assuming proper setup (correct directory structure, correct compilation), there should<br><br>be no runtime errors as well.<br><br>8.Output Verification: When you create an object of Class B: ```java B b = new B(); ``` It will internally create an object of Class A due to its member variable initialization: ```java A a = new A(); // This line triggers the print statement inside Class 'A' constructor. ``` So this results in printing "object created".<br><br>Therefore: Answer: (A) Output is: object created<br><br>[Figure 20]|

|Zeroshot|
|---|
|The correct answer is:<br><br>(C) Compile time error<br><br>Explanation: In Java, each class must be in its own file with the same name as the class. The provided code snippet does not follow this rule, and it also has issues with package declarations and imports.<br><br>1. Each class should be in a separate file named after the class (e.g., `A.java` for class A and `B.java` for class B).<br>2. The packages need to be properly structured in directories that match their names.<br>3. The import statement should correctly reference classes from other packages.<br><br><br>Given these issues, attempting to compile this code will result in a compile-time error due to incorrect packaging and file structure.<br><br>[Figure 21]|

## Figure 14: Examples of Chain-of-Thought and zeroshot in Object-oriented programming (OOP) subset.

CodeMMLUzeroshotexperimentalresults.Table9:

Repair Execution

Predic-

Claude3.7Sonnet88.4548.7677.8583.7270.8279.2767.8773.6857.79

Claude3.5Sonnet89.4448.0474.5578.1269.8087.8062.2461.8456.41

Claude3.5Haiku84.5945.2669.2370.9965.2875.0069.9450.0053.16

GPTo3-mini88.4549.1077.4681.6869.6687.2073.2767.1158.33

GPT4o87.0246.0869.9370.9965.2167.6854.3959.2151.27

GPT4o-mini88.4544.1850.1270.2363.8265.8539.5046.0511.97

GPT3.5-turbo84.8838.4751.0658.5251.3576.8329.3138.1536.74

Llama3.1405BInstruct91.0146.2972.3474.5568.5688.4154.5859.2157.11

Llama3.170BInstruct87.3042.4062.7167.1861.2076.8354.4946.0552.11

Llama3.370BInstruct87.5939.4450.7262.0954.3481.7163.2738.1618.03

89.0245.4262.2772.5263.464.8887.9856.863.32Claude3Sonnet

CodeLlama7BInstruct72.0428.2336.8047.8439.020.006.957.904.26

CodeLlama7bPython54.2123.1835.2843.7731.5842.6817.9010.746.11

CodeLlama13BInstruct72.0429.1239.0950.8939.020.0039.971.350.79

CodeLlama13BPython0.0025.7914.650.000.0051.2270.60-7.38

CodeLlama13B72.0429.1239.0850.8938.881.2220.430.004.96

CodeLlama34BInstruct79.1734.4441.7052.9346.1737.8024.8027.823.77

CodeLlama34BPython0.0031.1416.020.000.0012.208.3114.662.30

Llama370B86.0240.7452.9263.1056.8921.9592.0224.912.28

Llama370BInstruct87.5942.2058.8367.6862.3678.0576.0082.646.66

Llama38B75.8932.3842.7153.9446.7548.1794.6566.705.85

Llama38BInstruct81.3134.8933.4360.3150.9160.3755.5234.303.33

Llama3.170B87.0241.1653.4465.6557.913.054.9322.422.48

Llama3.18B75.0432.8641.9454.7147.6335.3733.2652.605.11

Llama3.18BInstruct78.3234.7647.1753.4450.4711.5971.5455.793.88

- Mistral7BInstruct(v1)74.7530.7441.8753.4443.1859.7679.6623.632.90

- Mistral7BInstruct(v2)71.9032.3842.9254.9646.0243.9010.3847.612.15

- Mistral7BInstruct(v3)75.4633.3847.7757.7648.2156.1016.3052.742.25

tion

blank Code

tion Fillinthe

Comple-

&SQL OthersCode

ple DBMS

tax Software

princi-

works PLSyn-

API&Model

Frame-

Phi-487.0240.6456.0867.4357.4875.6158.4360.5343.14

Phi-4MiniInstruct87.0236.8648.8661.8354.3423.1723.5335.5318.12

Qwen2.514BInstruct86.5941.8557.8366.9258.1380.4976.6638.1641.94

QwQ38B78.3239.0552.9359.2951.7135.3740.8636.8442.79

QwenCoder2.532BInstruct86.8846.5471.1573.2865.3581.7168.5838.1648.12

QwenCoder2.514BInstruct87.4541.5858.0863.1055.5182.9373.4155.2631.35

DeepSeek-R171.4739.1256.1263.8756.0956.7133.2642.1139.23

DeepSeek-V387.0243.9454.8566.9260.3956.1041.3351.3246.00

Mixtral8x7BInstruct85.0237.3249.9661.8352.883.0533.094.32

Codestral22B82.1738.5047.6258.5250.1874.3925.6948.822.53

Phi3MediumInstruct(128k)79.8937.1951.4458.7853.4667.6840.8239.573.48

Phi3MediumInstruct(4k)80.0336.8252.5362.0954.4172.5644.7651.663.68

Phi3MiniInstruct(128k)74.1831.8344.9954.2046.7542.0731.3814.451.55

Phi3MiniInstruct(4k)76.3233.1440.7053.9445.3053.6624.0030.532.35

Phi3SmallInstruct(8k)77.8937.3254.4064.1252.2262.8021.4221.071.95

PhindCL34Bv279.8935.2644.6351.9145.8853.0527.4318.503.06

- Qwen20.5BInstruct52.9224.2336.4543.2632.9028.6613.7274.741.07

- Qwen21.5BInstruct73.0430.0343.5254.2044.789.7621.2827.202.49

Qwen257B-A14BInstruct84.8837.8051.8765.1455.4358.5424.5235.902.96

Qwen27B82.8833.7348.1262.8554.7075.0056.9360.644.63

Qwen27BInstruct82.7437.0654.4162.8553.9869.5150.7846.054.31

CodeQwen1.57B74.4730.5638.8951.6540.4139.635.40-5.92

CodeQwen1.57BChat67.6231.7038.2546.0639.2489.0237.06-3.38

Yi1.534BChat79.6037.0352.9361.3252.5267.6835.0955.303.00

Yi-1.56BChat75.7534.4545.1958.0249.5353.0541.9036.662.67

Yi1.59BChat75.4635.8252.3260.0552.8160.9832.2749.905.43

DSCoder33B0.000.0015.980.000.006.1013.6218.645.83

DSCoder33BInstruct74.8932.4041.2452.1642.898.5431.9440.065.32

70.1928.7136.9649.3639.1016.460.092.630.02DSCoder6.7B

- DSCoder6.7BInstruct71.7529.8438.0152.6738.1526.833.660.003.69

- DSCoder7B(v1.5)81.6035.9847.2157.5146.320.0027.6241.130.00

DSCoder7BInstruct(v1.5)79.7433.6043.2955.4744.9313.4131.0965.493.85

DSMoE16B56.4923.4731.8143.0034.870.6126.2146.051.25

DSMoE16BChat41.9421.5433.4441.4831.3618.9050.5437.212.65

DSCoderV2Lite83.8835.0047.3459.2948.5040.8519.9632.880.17

DSCoderV2LiteInstruct81.7438.0749.8564.3850.0446.3445.0439.573.53

InternLM2.520BChat78.0337.6751.8560.0554.6357.9324.4738.221.12

79.3235.3250.9356.7451.7159.1518.6926.475.41InternLM2.57BChat

StarCoder215BInstruct78.7434.4146.9452.4247.8573.1742.0452.533.40

StarCoder27B63.7727.9734.1948.8536.2530.4967.786.624.81

CodeMMLUfew-shotexperimentalresults.Table10:

Repair Execution

Predic-

GPT-4o42.1179.3267.6856.8356.0286.5979.0563.7638.28

GPT-3.5-turbo38.4784.8858.5252.9350.0470.1254.0635.3834.45

Claude3Sonnet44.5886.3170.9964.7363.6811.5940.1152.673.11

CodeLlama13BInstruct29.1272.0450.8939.0939.0262.8017.6125.713.03

CodeLlama13BPython25.7962.6244.0235.1633.410.610.001.280.03

29.1272.0450.8942.0438.880.610.190.000.00CodeLlama13B

CodeLlama34BInstruct34.4479.1752.9344.5146.1710.370.0027.030.07

CodeLlama34BPython31.1474.1850.1339.9442.521.220.001.280.69

28.2372.0447.8439.5639.021.2244.206.690.02CodeLlama7BInstruct

CodeLlama7BPython23.4954.0744.2737.2331.800.000.000.000.12

Llama370B40.7486.0263.1055.4756.8967.0725.5012.020.80

Llama370BInstruct42.2087.5967.6861.0462.3650.0060.4047.611.93

Llama38B32.3875.8953.9445.5246.7593.9040.4492.313.62

Llama38BInstruct34.7781.7460.8150.3050.6227.4487.6917.360.28

Llama3.170B41.1687.0265.6556.4957.9148.782.0718.160.75

Llama3.170BInstruct41.3887.4569.2160.9859.306.1030.3980.154.31

Llama3.18B32.8675.0454.7145.1347.631.220.282.630.03

Llama3.18BInstruct34.7678.3253.4448.7350.4750.0067.3657.523.53

- Mistral7BInstruct(v1)30.7474.7553.4444.7643.1814.027.8444.532.13

- Mistral7BInstruct(v2)32.3871.9054.9645.9346.0223.785.317.970.75

- Mistral7BInstruct(v3)33.3875.4657.7650.2748.2119.512.359.320.84

Mixtral8x7BInstruct37.3384.7461.0753.2453.3225.007.239.321.83

Codestral22B38.5082.1758.5250.3550.1831.7114.6121.000.97

Phi3MediumInstruct(128k)37.1979.8958.7853.2753.4610.374.042.700.22

Phi3MediumInstruct(4k)36.8280.0362.0954.9254.417.326.3429.660.50

Phi3MiniInstruct(128k)31.8374.1854.2047.2946.751.220.382.560.07

Phi3MiniInstruct(4k)33.1476.3253.9444.0145.300.000.190.000.05

tion

blank Code

tion Fillinthe

Comple-

ples OthersCode

&SQL Software

princi-

works DBMS

syntax API&

Frame-

ModelPL

Phi3SmallInstruct(8k)37.3278.6064.1256.8352.2256.1056.6538.771.90

Qwen257B-A14BInstruct37.8084.8865.1455.1955.4342.0719.2614.591.14

Qwen27B36.3882.8861.5853.0654.497.322.541.280.00

Qwen27BInstruct37.0182.7462.8556.3554.2735.9843.6810.602.91

CodeQwen1.57B30.5674.0451.1542.4640.340.000.140.000.07

CodeQwen1.57BChat31.7067.6246.0640.1739.101.833.058.041.58

Yi1.534BChat37.0380.1762.3454.9352.8121.9513.0121.280.25

Yi-1.56BChat34.3475.7558.0248.5149.530.615.2615.731.13

35.7475.4660.0554.1952.810.6124.993.980.07Yi1.59BChat

DSCoder33B29.2674.7548.6040.9340.410.610.001.280.00

32.3575.3251.9143.7343.0314.639.022.561.22DSCoder33BInstruct

DSCoder6.7B28.7170.1949.3640.0639.101.220.281.280.00

- DSCoder6.7BInstruct29.8471.7552.6741.6838.155.490.891.280.02

- DSCoder7B(v1.5)36.0881.3157.5149.7946.4637.8063.7423.222.60

DSCoder7BInstruct(v1.5)33.4679.6054.7145.8644.930.000.000.000.02

DSMoE16B23.8956.4943.5135.5934.720.000.090.000.00

DSMoE16BChat21.5443.3741.7335.8731.150.610.050.000.08

35.0083.8859.2950.3348.500.610.001.280.00DSCoderV2Lite

DSCoderV2LiteInstruct38.0781.7464.3853.4850.047.9314.092.630.17

InternLM2.520BChat37.6778.0360.0553.9054.6325.611.031.350.02

StarCoder215BInstruct34.4178.8952.9347.8547.704.274.232.630.17

CodeMMLUChain-of-Thoughwithzeroshotexperimentalresults.Table11:

Repair Execution

Predic-

GPT-4o27.8337.0930.0334.6934.0624.3944.7638.4657.44

GPT-3.5-turbo0.7985.7360.8149.0050.4769.5139.3648.8241.95

Claude3Sonnet42.6041.9471.7672.8161.200.6161.3448.613.26

CodeLlama7BInstruct27.3069.4745.5538.4337.9337.2035.2329.182.70

CodeLlama7BPython24.4959.4942.2430.0330.4956.710.0051.350.00

30.0676.8950.6437.6540.0420.1240.0265.702.45CodeLlama13BInstruct

CodeLlama13BPython27.1769.3346.0630.0336.110.000.00100.000.00

CodeLlama13B30.0576.7550.8937.6040.040.610.523.920.03

34.0077.3254.4542.7645.3779.2741.9418.162.32CodeLlama34BInstruct

CodeLlama34BPython23.8774.7551.9132.0640.340.001.6930.460.77

Llama370B30.8072.9050.6438.0340.7039.6319.160.000.53

Llama370BInstruct30.8568.1945.0438.1236.1189.6346.9798.722.84

Llama38B39.8485.7363.1052.2355.7319.5192.02100.003.53

Llama38BInstruct41.6486.1667.1855.3660.2573.1776.0085.763.98

Llama3.170B31.4876.6056.7441.0546.3211.5994.6579.872.65

Llama3.170BInstruct33.5478.6055.9846.0649.0259.7613.8133.022.15

Llama3.18B40.5086.1664.1252.2155.8719.514.9345.151.62

Llama3.18BInstruct40.1078.8965.1453.5057.5585.9884.6482.993.70

- Mistral7BInstruct(v1)31.9875.0455.9843.9946.2423.1717.5738.150.00

- Mistral7BInstruct(v2)33.4178.7454.2046.0246.5472.5671.5444.702.81

- Mistral7BInstruct(v3)26.5363.9150.6435.3233.8475.0062.6174.954.53

Mixtral8x7BInstruct31.4569.9051.6541.0743.8437.8023.4447.681.92

Codestral22B32.1170.4752.6739.9744.2757.3226.3055.372.13

Phi3MediumInstruct(128k)35.1679.6055.7346.0147.7833.5460.4081.703.21

Phi3MediumInstruct(4k)38.2279.3258.5249.1548.9480.4946.1760.782.45

Phi3MiniInstruct(128k)37.3079.3257.7650.2151.7985.9847.1173.873.28

Phi3MiniInstruct(4k)35.6978.0357.7647.4349.6085.3745.8960.642.46

tion

middle Code

tion Fillinthe

Comple-

ples OthersCode

&SQL Software

princi-

works DBMS

syntax API&

Frame-

ModelPL

Phi3SmallInstruct(8k)30.2165.0550.6440.4340.9267.0730.6260.572.97

PhindCL34Bv230.7462.7747.3337.8240.0457.9329.9231.602.50

- Qwen20.5BInstruct36.7478.6064.8956.4153.1043.2941.9943.833.03

- Qwen21.5BInstruct35.2380.6055.2245.8245.510.6136.5927.891.42

Qwen257B-A14BInstruct22.1448.5040.9731.3729.6137.2013.7221.281.27

28.1568.7653.6941.3741.5027.4421.2836.663.53Qwen27B

Qwen27BInstruct31.9372.3356.7441.3545.1572.5640.1657.733.70

CodeQwen1.57B36.9682.8864.6351.4554.1275.0077.6458.144.18

CodeQwen1.57BChat32.3277.6057.2547.2248.2952.4476.0060.645.31

Yi1.534BChat35.6977.4654.4545.6147.9287.2076.3368.754.66

Yi-1.56BChat32.1970.0458.5242.1745.0865.8568.1128.763.25

Yi1.59BChat34.4573.4756.4952.4249.4571.9575.9519.583.08

DSCoder6.7B26.5160.4944.7830.9831.580.000.050.000.00

- DSCoder6.7BInstruct29.4468.9049.1134.3736.4785.3747.3977.513.10

- DSCoder7B(v1.5)35.2479.3254.9644.7842.300.0027.3855.722.08

DSCoder7BInstruct(v1.5)33.2279.3252.4243.8245.0850.0045.6629.113.29

28.5071.3347.8438.3638.070.000.003.920.44DSCoder33B

DSCoder33BInstruct31.5475.7549.1138.8441.5845.1228.5131.462.48

DSMoE16B21.2549.5038.9333.4829.980.000.000.000.13

DSMoE16BChat23.8160.7748.0937.3736.8329.2736.1776.652.20

DSCoderV2Lite33.7883.5957.5147.1946.1029.880.8064.620.00

DSCoderV2LiteInstruct22.2238.6628.2425.6420.5091.4655.4762.992.63

InternLM2.520BChat35.0075.7557.7645.7050.0481.1075.9064.483.58

InternLM2.57BChat31.3067.7650.3842.0342.0179.2749.9860.643.75

StarCoder215BInstruct34.4483.3156.2346.9946.6175.0053.3124.912.30

StarCoder27B27.7763.7747.3336.9534.721.220.051.350.08

- B.4 CODEMMLU EXAMPLE General knowledge MCQ example:

The following are multiple-choice questions (with answers) about debugging a programming problem.

Question: Suppose we have an O(n) time algorithm that finds the median of an unsorted array. Now consider a QuickSort implementation where we first find the median using the above algorithm, then use the median as a pivot. What will be the worst-case time complexity of this modified QuickSort?

- (A) O(n2 log n)
- (B) O(n2)
- (C) O(nlog nlog n)
- (D) O(nlog n)

## Code Completion example:

The following are multiple-choice questions (with answers) about programming problems. Question: Which solution below is the most likely to complete the following code to achieve the desired goal?

from typing import List def has_close_elements(numbers: List[float], threshold: float) −> bool:

""" Check if in given list of numbers , are any two numbers closer to each

→ other than given threshold. >>> has_close_elements ([1.0, 2.0, 3.0], 0.5) False >>> has_close_elements ([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True """

### (A)

for i in range(len(numbers)): # Change range to len(numbers) for j in range(i + 1, len(numbers)): if abs(numbers[i] − numbers[j]) < threshold:

return True return False

### (B)

return any(abs(a − b) < threshold for a, b \ in zip(numbers , numbers [1:]))

### (C)

for i in range(len(numbers) − 1): for j in range(i + 1, len(numbers)): if abs(numbers[i] − numbers[j]) > threshold:

return False return True

### (D)

for idx , elem in enumerate(numbers): for idx2 , elem2 in enumerate(numbers):

if idx != idx2: distance = abs(elem − elem2) if distance < threshold:

return True return False

## Answer: Fill in the blank example:

The following are multiple-choice questions (with answers) about a programming problem with incomplete solution.

Problem statement: You are given an array of intervals, where intervals[i] = [starti, endi] and each starti is unique. The right interval for an interval i is an interval j such that startj >= endi and startj is minimized. Note that i may equal j. Return an array of right interval indices for each interval i. If no right interval exists for interval i, then put -1 at index i.

## Incomplete Solution:

def find_right_interval(intervals): n = len(intervals) res = [−1] ∗ n for i in range(n):

intervals[i].append(i)

def binary_search(ele): left , right = 0, n−1 ans = float(’inf’) while left <= right:

mid = (left + right) // 2 if intervals[mid][0] >= ele:

ans = min(ans , mid) right = mid − 1

else:

left = mid + 1 return ans

intervals.sort() for i in intervals:

_________________ if val != float(’inf’):

res[i[2]] = intervals[val][2] return res

Question: The provided solution is missing a part, which option below is the most likely to complete the solution and achieve the desired goal?

- (A) val = binary_search(i[1])
- (B) if val != float(’inf’):
- (C) val = binary_search(i[1])
- (D)

if val != float(’inf’): res[i[2]] = intervals[val][2]

Answer:

Code Repair example:

The following are multiple-choice questions (with answers) about debugging a programming problem. Question: The following code snippet is producing incorrect results; Which solution below correctly identifies the bug and repairs it to achieve the desired goal?

- 1 import java.util.∗;
- 2 public class DETECT_CYCLE {
- 3 public static boolean detect_cycle(Node node) {
- 4 Node hare = node;
- 5 Node tortoise = node;
- 6 while (true) {
- 7 if (hare.getSuccessor() == null)
- 8 return false;
- 9 tortoise = tortoise.getSuccessor();
- 10 hare = hare.getSuccessor().getSuccessor();
- 11 if (hare == tortoise)
- 12 return true;

- 13 }
- 14 }
- 15 }

- (A) Modify line 6: for (; ; ) {
- (B) Modify line 7: if (null==hare ||hare.getSuccessor() == null)
- (C) Modify line 12: return hare.getSuccessor() != null && hare == tortoise;
- (D) Modify line 11: if (Objects.equals(hare , tortoise))

## Execution Prediction example:

The following are multiple-choice questions (with answers) about programming problem. Question: Given a code snippet below, which behavior most likely to occur when running the solution?

import java.util.∗; public class Main {

public static void main(string[] args) { Scanner sc = new Scanner(System.in);

- int A = sc.nextInt();
- int B = sc.nextInt(); int T = sc.nextInt(); int S = T/A System.out.println(s∗b);

} }

- (A) Memory Limit Exceeded
- (B) Runtime Error
- (C) Compile Error
- (D) No abnormally found

C MODELS SETUP

In our experiment and study, we consider GPT-4o (OpenAI et al., 2024), GPT-3.5 (OpenAI, 2023), Claude-3.5, Claude-3 (Anthropic, 2024), MetaLlama 3.1 (Dubey et al., 2024), MetaLlama3 (Dubey et al., 2024), CodeLLaMA (Rozière et al., 2024), DeepSeek AI, DeepSeek Coder, DeepSeek Coder V2 (Guo et al., 2024b; DeepSeek-AI et al., 2024; Guo et al., 2024a), MistralAI, Codetral (Jiang et al., 2024), Qwen2 (Yang et al., 2024), CodeQwen1.5 (Bai et al., 2023), Yi (AI et al., 2024), StarCoder2 (Lozhkov et al., 2024a), InternLM (Cai et al., 2024), Phind (Phind, 2023).

Table 12: Language Models Description

## Model ID Short Name Link

GPT-o1-2024-12-17 GPT o1 GPT-o3-mini-2025-01-31 GPT o3-mini GPT-4o-2024-05-13 GPT 4o GPT-4o-mini-2024-07-18 GPT 4o mini GPT-3.5-turbo-16k-0613 GPT-3.5-turbo -

OpenAI

Claude-3.5-sonnet-20241022 Claude3.5 Sonnet Claude-3.7-sonnet-20250219 Claude3.7 Sonnet Claude-3.5-haiku-20241022 Claude3.5 Haiku Claude-3-haiku-20240307 Claude3 Haiku -

Anthropic

Claude-3-sonnet-202402029 Claude3 Sonnet Claude-3-opus-20240229 Claude3 Opus -

codellama/CodeLlama-13bInstruct-hf

CodeLlama 13B Instruct

codellama/CodeLlama13b-Instruct-hf

codellama/CodeLlama-13bPython-hf

CodeLlama 13B Python

codellama/CodeLlama13b-Python-hf

CodeLlama

codellama/CodeLlama-13b-hf CodeLlama 13B codellama/CodeLlama-

13b-hf

codellama/CodeLlama-34bInstruct-hf

CodeLlama 34B Instruct

codellama/CodeLlama34b-Instruct-hf

codellama/CodeLlama-34bPython-hf

CodeLlama 34B Python

codellama/CodeLlama34b-Python-hf

codellama/CodeLlama-7bInstruct-hf

CodeLlama 7B Instruct

codellama/CodeLlama7b-Instruct-hf

codellama/CodeLlama-7bPython-hf

CodeLlama 7B Python

codellama/CodeLlama7b-Python-hf

meta-llama/Meta-Llama-370B

Llama3 70B meta-llama/Meta-

Llama-3-70B

meta-llama/Meta-Llama-370B-Instruct

Llama3 70B Instruct

meta-llama/MetaLlama-3-70B-Instruct

MetaLlama

meta-llama/Meta-Llama-3-8B Llama3 8B meta-llama/Meta-

Llama-3-8B

meta-llama/Meta-Llama-38B-Instruct

Llama3 8B Instruct

meta-llama/MetaLlama-3-8B-Instruct

meta-llama/Meta-Llama-3.170B

Llama3.1 70B meta-llama/Meta-

Llama-3.1-70B

meta-llama/Meta-Llama-3.170B-Instruct

Llama3.1 70B Instruct

meta-llama/MetaLlama-3.1-70BInstruct

meta-llama/Meta-Llama-3.18B

Llama3.1 8B meta-llama/Meta-

Llama-3.1-8B

meta-llama/Meta-Llama-3.18B-Instruct

Llama3.1 8B Instruct

meta-llama/MetaLlama-3.1-8B-Instruct

meta-llama/Meta-Llama-3.1405B-Instruct

Llama3.1 405B Instruct

meta-llama/MetaLlama-3.1-405BInstruct

meta-llama/Meta-Llama-3.370B-Instruct

Llama3.3 70B Instruct

meta-llama/MetaLlama-3.3-70BInstruct

mistralai/Mistral-7B-Instruct-

- v0.1

Mistral 7B Instruct (v1)

mistralai/Mistral-7BInstruct-v0.1

mistralai/Mistral-7B-Instruct-

- v0.2

Mistral 7B Instruct (v2)

mistralai/Mistral-7BInstruct-v0.2

mistralai/Mistral-7B-Instruct-

- v0.3

Mistral

Mistral 7B Instruct (v3)

mistralai/Mistral-7BInstruct-v0.3

mistralai/Mixtral-8x7BInstruct-v0.1

Mixtral 8x7B Instruct

mistralai/Mixtral8x7B-Instruct-v0.1

mistralai/Codestral-22B-v0.1 Codestral 22B mistralai/Codestral-

22B-v0.1

microsoft/phi-4 Phi-4 microsoft/phi-4

microsoft/Phi-4-mini-instruct Phi-4-mini-

microsoft/Phi-4-miniinstruct

Phi

instruct

microsoft/Phi-3-medium128k-instruct

Phi3 Medium Instruct (128k)

microsoft/Phi-3medium-128k-instruct

microsoft/Phi-3-medium-4kinstruct

Phi3 Medium Instruct (4k)

microsoft/Phi-3medium-4k-instruct

microsoft/Phi-3-mini-128kinstruct

Phi3 Mini Instruct (128k)

microsoft/Phi-3-mini128k-instruct

microsoft/Phi-3-mini-4kinstruct

Phi3 Mini Instruct (4k)

microsoft/Phi-3-mini4k-instruct

microsoft/Phi-3-small-8kinstruct

Phi3 Small Instruct (8k)

microsoft/Phi-3-small8k-instruct

PhinD Phind/Phind-CodeLlama-

PhindCL 34B v2 Phind/Phind-

CodeLlama-34B-v2 CodeQwen

34B-v2

Qwen/CodeQwen1.5-7B CodeQwen1.5 7B Qwen/CodeQwen1.5-

7B

Qwen/CodeQwen1.5-7BChat

CodeQwen1.5 7B Chat

Qwen/CodeQwen1.57B-Chat

Qwen/Qwen2.5-Coder-32BInstruct

QwenCoder2.5 32B Inst

Qwen/Qwen2.5-Coder32B-Instruct

Qwen

Qwen/Qwen2.5-Coder-14BInstruct

QwenCoder2.5 14B Inst

Qwen/Qwen2.5-Coder14B-Instruct

Qwen/QwQ-32B-Preview QwQ 32B Qwen/Qwen2-0.5B-

Instruct

- Qwen/Qwen2-0.5B-Instruct Qwen2 0.5B In-

struct

Qwen/Qwen2-0.5BInstruct

- Qwen/Qwen2-1.5B-Instruct Qwen2 1.5B In-

Qwen/Qwen2-1.5BInstruct

struct

Qwen/Qwen2-57B-A14BInstruct

Qwen2 57BA14B Instruct

Qwen/Qwen2-57BA14B-Instruct

Qwen/Qwen2-7B Qwen2 7B Qwen/Qwen2-7B

Qwen/Qwen2-7B-Instruct Qwen2 7B

Qwen/Qwen2-7BInstruct

Instruct

01-ai/Yi-1.5-34B-Chat Yi1.5 34B Chat 01-ai/Yi-1.5-34B-Chat

Yi

01-ai/Yi-1.5-6B-Chat Yi-1.5 6B Chat 01-ai/Yi-1.5-6B-Chat 01-ai/Yi-1.5-9B-Chat Yi1.5 9B Chat 01-ai/Yi-1.5-9B-Chat

deepseek-ai/DeepSeek-R1 DeepSeek R1 deepseek-ai/R1

DeepSeek

deepseek-ai/deepseek-V3 DeepSeek V3 deepseek-ai/DeepSeek-

V3

deepseek-ai/deepseek-coder33b-base

DSCoder 33B deepseek-ai/deepseek-

coder-33b-base

deepseek-ai/deepseek-coder33b-instruct

DSCoder 33B Instruct

deepseek-ai/deepseekcoder-33b-instruct

DeepSeek Coder

deepseek-ai/deepseek-coder6.7b-base

DSCoder 6.7B deepseek-ai/deepseek-

coder-6.7b-base

deepseek-ai/deepseek-coder6.7b-instruct

DSCoder 6.7B Instruct

deepseek-ai/deepseekcoder-6.7b-instruct

deepseek-ai/deepseek-coder7b-base-v1.5

DSCoder 7B (v1.5)

deepseek-ai/deepseekcoder-7b-base-v1.5

deepseek-ai/deepseek-coder7b-instruct-v1.5

DSCoder 7B Instruct (v1.5)

deepseek-ai/deepseekcoder-7b-instruct-v1.5

deepseek-ai/DeepSeek-CoderV2-Lite-Base

DSCoderV2 Lite deepseek-ai/DeepSeek-

Coder-V2-Lite-Base

deepseek-ai/DeepSeek-CoderV2-Lite-Instruct

DSCoderV2 Lite Instruct

deepseek-ai/DeepSeekCoder-V2-LiteInstruct

DeepSeek MoE

deepseek-ai/deepseek-moe16b-base

DSMoE 16B deepseek-ai/deepseek-

moe-16b-base

deepseek-ai/deepseek-moe16b-chat

DSMoE 16B Chat deepseek-ai/deepseek-

moe-16b-chat

internlm/internlm2_5-20bchat

InternLM2.5 20B Chat

internlm/internlm2_520b-chat

InternLM

internlm/internlm2_5-7b-chat InternLM2.5 7B

internlm/internlm2_57b-chat

Chat

bigcode/starcoder2-15binstruct-v0.1

StarCoder2 15B Instruct

bigcode/starcoder215b-instruct-v0.1

StarCoder2

bigcode/starcoder2-7b StarCoder2 7B bigcode/starcoder2-7b

C.1 PROMPT LIBARARY Filtering prompts: LLM-based filtering for ranking questions’ completeness, coherence, and clarity.

Quality filtering prompt:

Please rate the following question based on three criteria , with a score from 1 to 5 for

→ each criterion (where 1 is the lowest and 5 is the highest). No explanation needed:

- 1. Completeness: − Does the question stand alone and provide enough information independently? − Avoid including any images , links , or external references.
- 2. Coherence and Clarity: − Is the question phrased clearly , with proper grammar? − Is there any ambiguity or confusion in the wording?
- 3. Relevance: − Is the question directly related to software development or programming issues? − Does it involve technical challenges , concepts , or tools commonly used in software

→ or programming? Question: """{}"""

Data creation prompts: Prompt used for synthesis distractor for real-world task:

Code Repair distractor creation prompts:

After extracting statement from buggy version, we use LLMs to rewrite a new version of that statement. We command LLMs to assume the bug is located in the assigned line and their target is correct that line. Here is the prompt:

Given a buggy Python code snippet , you will be asked to debugging the code. ‘‘‘ def truncate_number(number: float) −> float:

return number ∗ (number % 1)

‘‘‘ Let assume the bug is located in this line: ‘‘‘ return number ∗ (number % 1)‘‘‘ Adjust this line in order to solve the bug. The re−written line must be syntactic correct , executable and wrapped in ‘‘‘ ‘‘‘

→ brace. Don’t give any details. ### Rewritten line: ‘‘‘ return number % 1.0‘‘‘

Given a buggy Java code snippet , you will be asked to debugging the code. ‘‘‘{code}‘‘‘ Let assume the bug is located in this line: ‘‘‘{line}‘‘‘ Adjust this line in order to solve the bug. The re−written line must be syntactic correct , executable and wrapped in ‘‘‘ ‘‘‘

→ brace. Don’t give any details. ### Rewritten line:

We executing the problem with given test cases. Our target is to create reasonable false answer that would require deep interpretation. Follow by an LLMs based filter to pick from pool of negative answer the most likely able to solve the buggy problem. This result a set of confusing negative answer. Those reasonable false sample with executable (and if they can pass through few testcases) is golden negative answer.

#### Fill in the blank distractor creation prompt:

From correct solution from leetcode, we randomly mask a line/a block of code and generate false answer (for multiple choice) from LLMs:

Following this code: {code} I prepare some multiple choice questions answering so i want to make small change on this line but it still look true of this line : {line} help me generate 3 version change in this code and each output should in ‘‘‘ ‘‘‘

→ brace and code only. Don’t give any details

Experimental prompts: Prompt used in CodeMMLU evaluation.

Zero-shot prompts General knowledge MCQ test set:

The following are multiple −choice questions (with answers) about software

→ development.

Question: {question} {multiple_choices}

Answer:

#### Code completion:

The following are multiple −choice questions (with answers) about software

→ development.

Question: {question} {multiple_choices}

Answer:

#### Fill in the blank:

The following are multiple −choice questions (with answers) about a programming

→ problem with an incomplete solution. Problem statement: {question} Incomplete Solution: {codebase} Question: The provided solution is missing a part , Which option below is the

→ most likely to complete the solution and achieve the desired goal? {multiple_choices} Answer:

#### Code Repair:

The following are multiple −choice questions (with answers) about debugging a

→ programming problem. Question: The implementation below is producing incorrect results. Which

→ solution below correctly identifies the bug and repairs it to achieve the desired

→ goal?

{question} {multiple_choices} Answer:

#### Defect Detection:

The following are multiple −choice questions (with answers) about programming

→ problems. Question: Given a code snippet below , which behavior most likely to occur when

→ execute it? {question} {multiple_choices} Answer:

#### Few-shot prompt General knowledge MCQ test set:

The following are multiple choice questions (with answers) about software

→ development.

Question: If a sorted array of integers is guaranteed to not contain duplicate → values , in order to search a for a specific value which of the following algorithms

→ is the most efficient for this task? (A) Bubble Sort (B) Linear Search (C) Insertion Sort (D) Binary Search Answer: The answer is (D). Question: {question} {multiple_choices} Answer:

#### Code completion:

The following are multiple −choice questions (with answers) about programming

→ problems. Question: Which solution below is the most likely completion the following code

→ snippet to achieve the desired goal? ‘‘‘python from typing import List def two_sum(nums: List[int], target: int) −> List[int]:

""" Given an array of integers nums and an integer target , return indices of the

→ two numbers such that they add up to target. You may assume that each input would have exactly one solution , and you may

→ not use the same element twice.

- >>> two_sum([2,7,11,15], 9)

- [0,1]

>>> two_sum([3,2,4], 6)

- [1,2] >>> two_sum([3,3], 6) [0,1] """

‘‘‘

- (A) ‘‘‘python n = len(nums) for i in range(n − 1):

for j in range(i + 1, n): if nums[i] + nums[j] == target:

return [i, j] return []

‘‘‘

- (B) ‘‘‘python for num in nums:

if target − num in nums:

return [nums.index(num), nums.index(target − num)] return []

‘‘‘

- (C) ‘‘‘python for i in range(len(nums)):

if nums[i] ∗ 2 == target:

return [i, i] return []

‘‘‘

- (D) ‘‘‘python num_dict = {} for i, num in enumerate(nums):

if target − num in num_dict:

return [num_dict[target − num], i] num_dict[i] = num

return []

‘‘‘ Answer: The answer is A.

Question: Which solution below is the most likely completion the following code

→ snippet to achieve the desired goal? ‘‘‘python {question} ‘‘‘

{multiple_choices} Answer:’’’

#### Fill in the blank:

The following are multiple −choice questions (with answers) about a programming

→ problem with incomplete solution. Problem statement: You are given an array of intervals , where intervals[i] =

→ [starti , endi] and each starti is unique. The right interval for an interval i is an interval j such that startj >= endi

→ and startj is minimized. Note that i may equal j. Return an array of right interval indices for each

→ interval i. If no right interval exists for interval i, then put −1 at index i. Incomplete Solution: python ‘‘‘ def find_right_interval(intervals):

n = len(intervals) res = [−1] ∗ n for i in range(n):

intervals[i].append(i)

def binary_search(ele): left , right = 0, n−1 ans = float(’inf’) while left <= right:

mid = (left + right) // 2 if intervals[mid][0] >= ele:

ans = min(ans , mid) right = mid − 1

else:

left = mid + 1 return ans

intervals.sort() for i in intervals:

_________________ return res

‘‘‘ Question: The provided solution is missing a part , Which option below is the

→ most likely to complete the solution and achieve the desired goal?

- (A) ‘‘‘python val = binary_search(i[1]) if val != float(’inf’):

res[i[2]] = intervals[val][2] ‘‘‘

- (B) ‘‘‘python if val != float(’inf’):

res[i[2]] = intervals[val][2] else:

continue ‘‘‘

- (C) ‘‘‘python val = binary_search(i[1]) if val != float(’inf’): res[i[2] + 1] = intervals[val][2]

‘‘‘

- (D) ‘‘‘python if val != float(’inf’):

res[i[2]] = intervals[val][2] else:

continue

‘‘‘ Answer: The answer is (A).

Problem statement: {question} Incomplete Solution: {codebase} Question: The provided solution is missing a part , Which option below is the

→ most likely to complete the solution and achieve the desired goal? {multiple_choices} Answer:

#### Code Repair:

The following are multiple −choice questions (with answers) about debugging a

→ programming problem.

Question: The implementation below is producing incorrect results. Which solution below correctly identifies the bug and repairs it to achieve the

→ desired goal?

- 1 def two_sum(nums , target):
- 2 complement_map = {{}}
- 3 for i, num in enumerate(nums):
- 4 complement = target − num
- 5 complement_map[num] = i
- 6 if complement in complement_map:
- 7 return [complement_map[complement], i]
- 8 return None

- (A) Remove line 5.
- (B) Remove line 5. Add at line 7: ‘‘‘ complement_map[num] = i‘‘‘
- (C) Modify line 7: ‘‘‘ return [i, complement_map[complement]]‘‘‘
- (D) Remove line 5. Add at line 7: ‘‘‘ if i == len(nums) − 1:

return None

complement_map[num] = i‘‘‘ Answer: The answer is (B). Question: The implementation below is producing incorrect results. Which solution below correctly identifies the bug and repairs it to achieve the

→ desired goal? {question} {choices} Answer:

#### Defect Detection:

The following are multiple choice questions (with answers) about programming

→ problem. Question: Given a code snippet below , which behavior most likely to occurr when

→ execute it? ‘‘‘python def chkPair(A, size , x):

for i in range(0, size − 1): for j in range(i + 1, size): if (A[i] + A[j] == x):

return 1 return 0

‘‘‘

- (A). The code contain no issue.
- (B). Memory Limit Exceeded
- (C). Compile error
- (D). Runtime Error

Answer: The answer is (A). Question: Given a code snippet below , which behavior most likely to occurr when

→ execute it? {question} {multiple_choices} Answer:

#### Chain-of-Thought zero-shot prompts General knowledge MCQ test set:

The following are multiple choice questions (with answers) about software

→ devopment.

Question: {question} {multiple_choices}

Answer: Let’s think step by step.

#### Code completion:

The following are multiple choice questions (with answers) about programming

→ problems. Question: Which solution below is the most likely completion the following code

→ snippet to achieve the desired goal? ‘‘‘python {question} ‘‘‘ {multiple_choices}

Answer: Let’s think step by step.

#### Fill in the blank:

The following are multiple −choice questions (with answers) about a programming

→ problem with uncomplete solution. Problem statement: {question} Incomplete Solution: {codebase} Question: The provided solution is missing a part , Which option below is the

→ most likely to complete the solution and achieve the desired goal? {multiple_choices} Answer: Let’s think step by step.

#### Code Repair:

The following are multiple −choice questions (with answers) about debugging a

→ programming problem.

Question: The implementation below is producing incorrect results. Which solution below correctly identifies the bug and repairs it to achieve the

→ desired goal? {question} {multiple_choices} Answer: Let’s think step by step.

#### Defect Detection:

The following are multiple −choice questions (with answers) about debugging a

→ programming problem.

The algorithm implementation below is producing incorrect results; Which solution below correctly identifies the bug and repairs it to achieve the

→ desired goal? {question} {multiple_choices} Answer: Let’s think step by step.

#### Chain-of-Thought few-shot prompts General knowledge MCQ test set:

The following are multiple choice questions (with answers) about software

→ devopment.

Question: If a sorted array of integers is guaranteed to not contain duplicate → values , in order to search a for a specific value which of the following algorithms

→ is the most efficient for this task? (A) Bubble Sort (B) Linear Search (C) Insertion Sort (D) Binary Search Answer: Let’s think step by step. Binary Search is a divide −and −conquer

→ algorithm that works by repeatedly dividing the search interval in half and searching → for the value in the appropriate half. Since the array is already sorted and does not → contain any duplicate value , this algorithm is optimal to find the desired value. The → answer is (D).

Question: {question} {multiple_choices}

Answer: Let’s think step by step.

#### Code completion:

The following are multiple choice questions (with answers) about programming

→ problem. Question: Which solution below is the most likely completion the following code

→ snippet to achieve the desired goal? ‘‘‘python def is_vowel(char: str) −> bool:

""" Checks if the input character is a vowel. """

‘‘‘

- (A) ‘‘‘python return char.lower().is_vowel()

‘‘‘

- (B) ‘‘‘python vowels = set("aeiou") return char.lower() in vowels

‘‘‘

- (C) ‘‘‘python vowels = set("aeiou") return char.upper() in vowels

‘‘‘

- (D) ‘‘‘python vowels = "aeiou" return char.count(vowels) > 0

‘‘‘ Answer: Let’s think step by step. The goal is to write a function is_vowel(char:

→ str) −> bool that checks if the input character char is a vowel. The solution B

→ correctly converts the input character to lowercase and checks if it is in the set of

→ vowels. The answer is (B). Question: Which solution below is the most likely completion the following code

→ snippet to achieve the desired goal? ‘‘‘python {question} ‘‘‘ {multiple_choices}

Answer: Let’s think step by step.

#### Fill in the blank:

The following are multiple −choice questions (with answers) about a programming

→ problem with uncomplete solution. Problem statement: You are given an array of intervals , where intervals[i] =

→ [starti , endi] and each starti is unique. The right interval for an interval i is an interval j such that startj >= endi

→ and startj is minimized. Note that i may equal j. Return an array of right interval indices for each

→ interval i. If no right interval exists for interval i, then put −1 at index i. Incomplete Solution: python ‘‘‘ def find_right_interval(intervals):

n = len(intervals) res = [−1] ∗ n for i in range(n):

intervals[i].append(i)

def binary_search(ele): left , right = 0, n−1 ans = float(’inf’) while left <= right:

mid = (left + right) // 2 if intervals[mid][0] >= ele:

ans = min(ans , mid) right = mid − 1

else:

left = mid + 1 return ans

‘‘‘

intervals.sort() for i in intervals:

_________________ return res

Question: The provided solution is missing a part , Which option below is the

→ most likely to complete the solution and achieve the desired goal?

- (A) ‘‘‘python val = binary_search(i[1]) if val != float(’inf’):

res[i[2]] = intervals[val][2] ‘‘‘

- (B) ‘‘‘python if val != float(’inf’):

res[i[2]] = intervals[val][2] else:

continue ‘‘‘

- (C) ‘‘‘python val = binary_search(i[1]) if val != float(’inf’): res[i[2] + 1] = intervals[val][2]

‘‘‘

- (D) ‘‘‘python if val != float(’inf’):

res[i[2]] = intervals[val][2] else:

continue ‘‘‘

Answer: Let’s think step by step. The incomplete solution first sorts the

→ intervals and then iterates over the sorted intervals. For each interval , it finds

→ the right interval using a binary search. This option (A) finds the right interval index using the binary search and

→ updates the result array accordingly.

- The option (B) is similar to (A), but it does not increment the index when

→ finding the right interval index. This could lead to incorrect results.

- The option (C) increments the index when finding the right interval index.

→ However , this is incorrect because the problem statement asks for the index of the

→ right interval , not the offset from the original index.

- The option (D) uses the same index for both the original interval and the right

→ interval , which could lead to incorrect results. The answer is (A). Problem statement: {question} Incomplete Solution: {codebase} Question: The provided solution is missing a part , Which option below is the

→ most likely to complete the solution and achieve the desired goal? {multiple_choices} Answer: Let’s think step by step.

#### Code Repair:

The following are multiple −choice questions (with answers) about debugging a

→ programming problem.

Question: The implementation below is producing incorrect results. Which solution below correctly identifies the bug and repairs it to achieve the

→ desired goal?

1 def two_sum(nums , target): 2 complement_map = {{}} 3 for i, num in enumerate(nums): 4 complement = target − num 5 complement_map[num] = i 6 if complement in complement_map: 7 return [complement_map[complement], i] 8 return None

(A) Remove line 5. (B) Remove line 5. Add at line 7: ‘‘‘ complement_map[num] = i‘‘‘ (C) Modify line 7: ‘‘‘ return [i, complement_map[complement]]‘‘‘ (D) Remove line 5. Add at line 7: ‘‘‘ if i == len(nums) − 1:

return None complement_map[num] = i‘‘‘

Answer: Let’s think step by step. The bug in the code occurs because the current

→ number is added to the complement_map before checking if its complement already

→ exists , which can lead to incorrectly matching a number with itself. To fix this , the

→ number should only be added to the map after checking for its complement. Solution

→ (B) does exactly this by moving the line that adds the current number to the map

→ after the complement check , ensuring the logic works as intended without

→ self −matching errors. The answer is (B). Question: The implementation below is producing incorrect results. Which solution below correctly identifies the bug and repairs it to achieve the

→ desired goal? {question} {choices} Answer: Let’s think step by step.

#### Defect Detection:

The following are multiple choice questions (with answers) about programming

→ problem. Question: Given a code snippet below , which behavior most likely to occurr when

→ execute it? ‘‘‘python def chkPair(A, size , x):

for i in range(0, size − 1): for j in range(i + 1, size): if (A[i] + A[j] == x):

return 1 return 0

‘‘‘

- (A). The code contain no issue.
- (B). Memory Limit Exceeded
- (C). Compile error
- (D). Runtime Error

Answer: Let’s think step by step. The code appears to have no issues with

→ typical valid inputs and will function as expected. It correctly checks for pairs of

→ elements whose sum is x. The answer is (A). Question: Given a code snippet below , which behavior most likely to occurr when

→ execute it? {question} {multiple_choices}

Answer: Let’s think step by step.

