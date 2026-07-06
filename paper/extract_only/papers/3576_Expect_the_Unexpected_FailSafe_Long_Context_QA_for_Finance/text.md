# arXiv:2502.06329v1[cs.CL]10Feb2025

## Expect the Unexpected: FailSafe Long Context QA for Finance

Kiran Kamble†, Melisa Russak†, Dmytro Mozolevskyi, Muayad Ali, Mateusz Russak Waseem AlShikh Writer, Inc {kiran, melisa, ... waseem}@writer.com

Abstract

We propose a new long-context financial benchmark, FailSafeQA, designed to test the robustness and context-awareness of LLMs against six variations in human-interface interactions in LLM-based query-answer systems within finance. We concentrate on two case studies: Query Failure and Context Failure. In the Query Failure scenario, we perturb the original query to vary in domain expertise, completeness, and linguistic accuracy. In the Context Failure case, we simulate the uploads of degraded, irrelevant, and empty documents. We employ the LLM-as-a-Judge methodology with Qwen2.5-72B-Instruct and use fine-grained rating criteria to define and calculate Robustness, Context Grounding, and Compliance scores for 24 off-the-shelf models. The results suggest that although some models excel at mitigating input perturbations, they must balance robust answering with the ability to refrain from hallucinating. Notably, Palmyra-Fin-128k-Instruct, recognized as the most compliant model, maintained strong baseline performance but encountered challenges in sustaining robust predictions in 17% of test cases. On the other hand, the most robust model, OpenAI o3-mini, fabricated information in 41% of tested cases. The results demonstrate that even high-performing models have significant room for improvement and highlight the role of FailSafeQA as a tool for developing LLMs optimized for dependability in financial applications. The dataset is available at: https://huggingface.co/ datasets/Writer/FailSafeQA

### 1 Introduction

As the domains of financial services and Large Language Models (LLMs) evolve at a rapid pace, it comes as no surprise that finance, with its growing need for new tools to uncover insights from data, is increasingly adopting newly emerged LLMs for

† These authors contributed equally. The order is determined by dice rolling.

this purpose (Li et al., 2023; Zhao et al., 2024; Maple et al., 2023). These tools are later used with significant impact in critical areas such as risk analysis, customer service, and operational decisions. Despite warnings against over-reliance on LLMbased systems in financial domains, people increasingly depend on fully automated processes, driven by trust in automation and the complexity of managing vast amounts of data, such as long context window financial reports (The Alan Turing Institute, 2023; Lee and See, 2004). Adding to these concerns, recent research has shown, that models are very sensitive to subtle changes in prompt formatting (Lu et al., 2022; Sclar et al., 2024). These findings highlight the need for robust measures to evaluate the risks associated with LLM dependence and establish criteria for differentiating between safe and unsafe models. Our approach extends beyond traditional financial benchmarks that focus solely on LLMs performance under ideal conditions (Xie et al., 2024; Liu et al., 2024b; Islam et al., 2023; Guo et al., 2023; Xie et al., 2023). We have developed testing scenarios that more accurately mirror real-world interactions between users and query-answer systems (QA systems), ensuring the tool maintains reliability even when queries deviate from typical patterns or involve topics beyond the scope of the document. This approach is particularly crucial when the user has limited domain expertise or knowledge of the document contents.

In response to these identified issues, we introduce a new long context benchmark, FailSafeQA, which evaluates LLM resilience against variations in human input within the financial sector caused by varying domain expertise, query incompleteness, source irrelevance, and linguistic inaccuracies. Research has demonstrated that LLMs tend to overlook details or fabricate responses when processing long-context texts (Hsieh et al., 2024; Liu et al., 2024a). Consequently, we have chosen long 10-K annual reports as our primary text

| | |
|---|---|
| | |
| | |
| | |
| | |

| |
|---|

| |
|---|

| |
|---|

- Figure 1: FailSafeQA: Robustness and Context Grounding Evaluation We evaluate the resilience of an LLMbased QA system in two case studies: Query Failure and Context Failure. In the Query Failure scenario, we perturb the original query into three variants: containing spelling errors (Misspelled Query), query-term form (Incomplete Query), rephrased to exclude in-domain terminology (Out-of-Domain Query). In the Context Failure case, we assume users can either fail to upload the document (Missing Context) , use degraged quality documents due to OCR (OCRed Context) or upload a document irrelevant to the query (Irrelevant Context). Robustness involves maintaining consistent model performance across perturbations (A)-(C) and (E), which preserve the intended meaning, while Context Grounding involves preventing hallucinations in scenarios (D) and (F).

source. To simplify the judging process, we base it on the ground truth and supporting citations, ensuring that all answers can be sourced from a short, relevant citation from the document. This approach reduces the context length required during the judging phase, leading to quicker and more precise evaluations of accuracy and comprehensiveness.

### 2 FailSafeQA Dataset

We have used publicly available annual reports of U.S. publicly traded companies that filed with the SEC’s EDGAR system during the years 1998, 1999, 2017, and 20182. We utilized 10-K filings, which have been truncated to maintain complete paragraphs while adhering to a context window that does not exceed 25k tokens.

We employed the Meta Llama 3.1 405B model (Dubey et al., 2024) for synthetic task generation and postprocessing steps (generate, rewrite, filter) and LongCite-llama3.1-8b (Zhang et al., 2024) for citation extraction (extract). Our semiautomated data generation pipeline consisted of

2https://github.com/fengbinzhu/fintech-nlp

three phases: query generation, query perturbation and context perturbation. 2.1 Query Generation

This phase focuses on producing and refining queries generated from historical financial documents.

- • (generate) Generate multi-turn query and answer pairs based on the truncated 10-K filings.
- • (filter) Identify the best standalone query from each interaction.
- • (rewrite) Standardize the queries to make each a clear, standalone question, intentionally removing courteous expressions which have been shown to affect results (Yin et al., 2024).
- • (extract) Extract and sanitize supporting citations from the full context for each queryanswer pair (refer to: Appendix A).
- • (filter) Retain only those data points for which the provided citations adequately support the query response.

[Figure 1]

- Figure 2: The Dataset Analysis of root verbs and their direct objects from the first sentence of each normalized query shows the top 20 verbs and their top five direct objects1. This distribution can be used as a proxy measure for the diversity of tasks in the dataset, with 83.0% related to question answering (QA) and 17.0% involving text generation (TG).

Prompts used for generating and filtering question-answer pairs can be found in Appendix C. 2.2 Query Perturbation

We used the Meta Llama 3.1 405B model to generate three types of query perturbations: Misspelled, Incomplete, and Out-of-Domain. These cases represent three key failure factors: language accuracy, search engine-style query phrasing, and lack of domain expertise.

Misspellings We introduced controlled spelling errors into financial queries using a rule-based approach. We generated four types of spelling errors: Split Errors where combined words were separated (e.g., "newspaper" to "news paper"), Segment Errors involving incorrect splitting or merging of words (e.g., "cat" to "c at" or "a cat" to "acat"), Realword Errors substituting words with similarlooking ones from a confusable list, and Common Typos sourced from Wikipedia’s List of Common Misspellings3 by reversing correct to incorrect spellings. These errors were distributed across the dataset with split errors making up 31.7%, segment errors 25.5%, realword substitutions 23.2%, and common typos 19.6%.

3https://en.wikipedia.org/wiki/Wikipedia: Lists_of_common_misspellings

Incomplete Queries In this perturbation type, our focus is on query incompleteness, drawing inspiration from the key-term-based queries typical of search engines. Some of these queries resemble the original by appearing as if words have been intentionally omitted or rearranged. For instance, the query "What are the details of the capital conservation buffer mentioned in the K-10 filings?" is transformed into "Details on the capital conservation buffer mentioned?" We created these incomplete queries using Meta Llama 3.1 405B and manually chose the most effective transformations.

Out-of-Domain The last category of query perturbation is inspired by the varying levels of expertise that users bring to a QA system. Ideally, whether a query is created by an in-domain expert or someone out-of-domain, it should lead to the same answer if the query is clear and targets the same information. The specific wording used should not impact the LLMs’ performance, as the model should possess the necessary expertise to interpret user intent accurately. For example, "What is the primary reason for the revenue increase in 2017?" should be equivalent to "Why did the company make more money in 2017?"

##### 2.3 Context Perturbation

After exploring query perturbations, we now shift our focus to transforming another part of the input - the context, which in this case is the 10-K filing.

Missing Context We have simply omitted the context from the final prompt while maintaining the original prompt structure intended to introduce the context. The expected LLM response is to refrain from addressing the query directly and to notify the user that the context is unavailable, possibly due to reasons such as a file upload failure.

OCR Errors Simulation We have simulated Optical Character Recognition (OCR) errors to reflect the typical contract execution process where a clean, digital version of a contract is converted into a paper document for signatures. This paper-based version, necessitated by the legal requirement for wet ink signatures (United States Bankruptcy Court for the Southern District of Florida, 2020), must then be converted back into digital form through scanning and OCR processing. This process introduces various inaccuracies into the text.

For OCR error generation, we used Mathieu Timmerman (2023), which manipulates characters

through deletions, replacements, and insertions based on probabilities derived from a normal distribution and a customizable character set. We have capped the upper limit on the character error probability at 10%. This value was empirically chosen to reflect a balance between preserving readability and mimicking realistic error occurrences. An example of OCR-corrupted text is shown in Appendix B.

Irrelevant Context We have randomly paired queries with irrelevant contexts and manually verified that these pairs were irrelevant to each other. An ideal LLM should acknowledge when the context is insufficient to answer the query, avoid fabricating responses or using general knowledge, and inform the user of the mismatch while suggesting the need for relevant documentation.

See Figure 1 for a visual summary of the different query and context perturbation scenarios discussed above.

##### 2.4 Dataset Statistics

The final dataset consists of 220 examples, each originally containing between 4.1k and 27k tokens

- as processed by the GPT-4 tiktoken tokenizer4. Notably, a large proportion (93.64%) of examples feature a long context window, exceeding 16k tokens. This count is based on the original context before the injection of OCR errors, which can affect tokenization and increase the token count by approximately 1.3 times.

Each data point includes a context paired with five questions (the original query, three perturbed variants, and an irrelevant query), an OCRed context, the ground truth answer, and supporting citations from the full context.

- Figure 2 shows the root verb and direct object of

the normalized query sentence for each data point, which we interpret as a proxy for the variety of instructions in the dataset5. The data generation prompt specified an 80/20 split between question answering (QA) and text generation (TG) tasks. After filtering and postprocessing, the final distribution showed proportions of 83.0% and 17.0%, respectively, indicating the influence of the initial data generation requirements.

4https://github.com/openai/tiktoken 5We utilized SpaCy 3.7.6 with en_core_web_sm 3.7.1

model for verb-dobj analysis.

### 3 Metrics

Answer Relevance Following Xu et al. (2024b), we assign each answer a label from the set {1,2,3,4,5,6}. We have designed the relevance labeling criteria such that the values {4,5,6} denote answers that are relevant to the ground truth and free from hallucinations, varying only in their comprehensiveness. Answers with ratings {1,2,3} either fail in terms of information accuracy or contain irrelevant content. We will use the cutoff property of the rating to define an auxiliary binary mapping that will determine a compliant answer in the next section. Detailed rating criteria are presented in subsection C.3.

Answer Compliance In our approach, a fail-safe QA system should never mislead a user by providing fabricated or irrelevant information. We will say that the answer is compliant if the relevance score is at least 4. We define an auxiliary metric c≥4 that maps the relevance score into its binary compliance score:

c≥4 =

1 r ≥ 4 0 otherwise

(1)

where r ∈ {1,2,3,4,5,6} is the original relevance label.

We note that the mapping has another useful mathematical property: the Answer Relevance is a categorical value, so one cannot directly take an average of relevances. The Answer Compliance maps categorical data into a binary classification, where the average is well-defined. In subsequent sections, whenever aggregate scores are mentioned without specifying the metric, we will always refer to an average of Answer Compliances, i.e., the ratio of cases when the rating was at least 4.

3.1 LLM Robustness Following HELM (Liang et al., 2023) we define LLM’s Robustness (R) as:

n

1 n

R =

c≥4(model(Tj(xi)),yi) (2)

min

j

i=1

where c≥4 is the compliance mapping. In our case input transformations T1,...Tk include identity (our baseline), query perturbations producing Misspelled, Incomplete and Out-of-Domain Queries, and OCR context perturbation. Robust QA systems are those that can provide a good answer despite perturbations of query and context.

- Figure 3: Answer Relevance Classes We evaluate two scenarios in our benchmark: when models should provide an answer (ANSWER QUERY) and when they must decline to answer (REFUSE QUERY) due to lack of relevant context. Our findings reveal that all the tested models are more adept at offering suitable answers than providing a justified refusal in situations where the context lacks sufficient information. Among all models evaluated, Palmyra-Fin-128k-Instruct demonstrates the most effective balance between these capabilities.

##### 3.2 LLM Context Grounding

We define LLM’s Context Grounding (G) as an average:

2

n

1 nj

c≥4(model(Tj(xi),Y )) (3)

G =

j=1

i=1

where c≥4 is defined as above, and two input transformations Tj are Missing Context and Irrelevant Context. Intuitively, the QA system with a high G score is able to detect cases where the problem is unanswerable and refrain from producing potentially misleading hallucinations.

In subsection C.3, we present the criteria used for rating queries affected by Missing Context and Irrelevant Context. We applied the same rules as for other input perturbations: a rating of 4−6 indicates that the model met fail-safe requirements, mean-

ing it refuses to answer while providing varying degrees of feedback.

##### 3.2.1 The Trade-off: LLM Compliance Score

Based on the observations detailed in our results, we decided to introduce a new metric, LLM Compliance Score (LLMCβ), that quantifies the tradeoff we identified between Robustness and Context Grounding. This metric is inspired by the classic precision-recall trade-off.

RG (β2 × G + R)

LLMCβ = (1 + β2) ·

(4)

where β is a positive real factor. Intuitively, Context Grounding and Robustness measure the ability of an LLM to refuse and answer the query, respectively. For β < 1, the compliance metric prioritizes refusal to reduce the hallucination ratio.

| | |
|---|---|
| | |

- Figure 4: Robustness and Compliance (Left) All models lose with respect to the baseline when input perturbations are applied. The biggest drop is observed for Out-Of-Domain and OCR context perturbations. Among the 24 tested models, OpenAI o3-mini is the most robust. (Right) Reasoning models like OpenAI-o1/o3-mini and the DeepSeek-R1 series reach scores up to 0.59, while Qwen models consistently surpass 0.60. Palmyra-Fin-128k-Instruct excels with the highest Context Grounding score of 0.80.

### 4 Evaluation

##### 4.1 Models

We have evaluated a range of both open-source LLMs and proprietary solutions that support context length minimum 128k.

For open-sourced models we have chosen: DeepSeek-R1 and its four distilled models (8B, 14B, 32B, 70B) (DeepSeek-AI et al., 2025), three Llama instruct models 3.x from Meta (3.1 8B, 3.1 70B, 3.3-70B) (Dubey et al., 2024), six Qwen 2.5 models including two 1M context window variants (7B, 14B, 32B, 72B, 7B-1M, 14B-1M) (Yang et al., 2024), Nvidia’s Nemotron-70B-Instruct-HF (Wang et al., 2024), three Phi 3 series models (mini, small, medium) (Abdin et al., 2024) and Writer’s Palmyra-Fin-128k-Instruct.

For proprietary APIs we have selected five: GPT4o, OpenAI o1, OpenAI o3-mini (OpenAI, 2024; OpenAI et al., 2024) Gemini 2.0 Flash Exp and Gemini 1.5 Pro 002 (Team et al., 2024).

Whenever possible we used the same temperature 0 and max new tokens (or max completion tokens) 2048. All models ran locally used halfprecision inference with 8x NVIDIA H100 GPUs.

##### 4.2 Judging

We used the LLM-as-a-Judge method (Zheng et al., 2023) with the generalist LLM, Qwen2.5-72BInstruct. In the evaluation stage, we provided the judge LLM with the rating criteria, reference so-

lution, relevant context citations, and the candidate answer. We used a temperature setting of 0, a maximum of 256 new tokens, and half-precision inference.

We note that due to the positive correlation between increasing task context length and performance drop (Hsieh et al., 2024), the judging task is seen as much simpler than the predictions made during the evaluation phase. We can think of citationbased judging as an escape from the performance degradation associated with long contexts and a justification for utilizing a potentially weaker LLM judge than the LLM being tested. Judging prompts are shown in subsection C.4.

### 5 Results

In FailSafe QA benchmark, we assessed models in two scenarios: providing a robust answer (Baseline Query, Misspelled Query, Incomplete Query, Outof-Domain Query, OCRed Context) and declining to answer when justifiable (Missing Context, Irrelevant Context). Figure 3 shows normalized Answer Relevance classes for both cases. It appears all models are better at delivering appropriate answers than at justifiably refusing to answer when the context is insufficient. Among the models, PalmyraFin-128k-Instruct strikes the best balance in these scenarios.

Robustness All models exhibited decreased performance when measuring Robustness, which as-

[Figure 2]

- Figure 5: Robustness vs. Query Type. (Left) Across all models, the decrease in robustness is more prominent in text generation (TG) than in question-answering (QA) tasks. (Right) Similar statement also holds true for Context Grounding - when a model is asked to generate text (e.g., a blog post), it is more likely to ignore the lack of relevant information and fabricate details. For almost all models, it is easier to refuse to answer based on a wrong document (irrelevant context) than to deal with empty context (e.g., due to a failed document upload).

[Figure 3]

- Figure 6: Context Grounding vs. Query Type. (Left) Across all models, the decrease in robustness is more prominent in text generation (TG) than in question-answering (QA) tasks. (Right) Similar statement also holds true for Context Grounding - when a model is asked to generate text (e.g., a blog post), it is more likely to ignore the lack of relevant information and fabricate details. For all models, it is easier to refuse to answer based on a wrong document (irrelevant context) than to deal with empty context (e.g., due to a failed document upload).

sesses the minimum effectiveness under both the original query (baseline) and perturbed conditions (a 0.07 to 0.28 decrease). Average Answer Compliance for all perturbations is presented in Figure 4. The most notable declines were observed in OCR and Out-of-Domain Query scenarios, with the smallest tested model, Phi-3-mini-128k-Instruct, experiencing declines reaching up to 0.17. The most robust model is OpenAI-o3-mini, with a score of 0.90 compared to the baseline score of 0.98.

Context Grounding Models showed considerable variation in Context Grounding, a key metric for assessing how well responses align with provided information or its absence. Missing context posed the biggest challenge for almost all tested

models (0.21–0.68), whereas irrelevant context appeared to be easier, showing consistent improvement by up to 0.30 across all models. Notably, reasoning models (OpenAI-o1/o3-mini and DeepSeekR1 series), which lead the robustness race, achieve at most a 0.59 score, whereas Qwen models easily achieve scores above 0.60, with the best one scoring 0.79. The best Context Grounding score of 0.80 is achieved by Palmyra-Fin-128k-Instruct.

Figure 6 shows how Compliance and Robustness varied across query types—question answering (QA) and text generation (TG). Content generation tasks (e.g., writing a blog post) were especially vulnerable to context alterations. When tasked with these queries, models showed a greater tendency to

Model Name Baseline Mispelled (∆) Incomplete (∆) Out-of-Domain (∆) OCR Context (∆) Robustness (∆) Gemini 2.0 Flash Exp 0.95 0.95 (0.0) 0.95 (0.0) 0.88 (↓0.07) 0.91 (↓0.04) 0.83 (↓0.12) Gemini 1.5 Pro 002 0.96 0.96 (0.0) 0.94 (↓0.02) 0.92 (↓0.04) 0.92 (↓0.04) 0.84 (↓0.12) OpenAI GPT-4o 0.95 0.94 (↓0.01) 0.94 (↓0.01) 0.92 (↓0.03) 0.95 (0.0) 0.85 (↓0.1) OpenAI o1 0.97 0.95 (↓0.02) 0.94 (↓0.03) 0.89 (↓0.08) 0.94 (↓0.03) 0.81 (↓0.16) OpenAI o3-mini 0.98 0.96 (↓0.02) 0.96 (↓0.02) 0.95 (↓0.03) 0.90 (↓0.08) 0.90 (↓0.08) DeepSeek-R1-Distill-Llama-8B 0.83 0.85 (↑0.02) 0.82 (↓0.01) 0.87 (↑0.04) 0.72 (↓0.11) 0.64 (↓0.19) DeepSeek-R1-Distill-Qwen-14B 0.95 0.90 (↓0.05) 0.92 (↓0.03) 0.93 (↓0.02) 0.86 (↓0.09) 0.82 (↓0.13) DeepSeek-R1-Distill-Qwen-32B 0.95 0.97 (↑0.02) 0.95 (0.0) 0.92 (↓0.03) 0.89 (↓0.06) 0.86 (↓0.09) DeepSeek-R1-Distill-Llama-70B 0.96 0.97 (↑0.01) 0.95 (↓0.01) 0.94 (↓0.02) 0.93 (↓0.03) 0.89 (↓0.07) DeepSeek-R1 0.94 0.94 (0.0) 0.93 (↓0.01) 0.91 (↓0.03) 0.88 (↓0.06) 0.80 (↓0.14) Meta-Llama-3.1-8B-Instruct 0.91 0.90 (↓0.01) 0.86 (↓0.05) 0.82 (↓0.09) 0.80 (↓0.11) 0.70 (↓0.21) Meta-Llama-3.1-70B-Instruct 0.94 0.92 (↓0.02) 0.94 (0.0) 0.87 (↓0.07) 0.88 (↓0.06) 0.80 (↓0.14) Meta-Llama-3.3-70B-Instruct 0.95 0.92 (↓0.03) 0.93 (↓0.02) 0.90 (↓0.05) 0.89 (↓0.06) 0.82 (↓0.13) Qwen2.5-7B-Instruct 0.92 0.91 (↓0.01) 0.90 (↓0.02) 0.85 (↓0.07) 0.80 (↓0.12) 0.75 (↓0.17) Qwen2.5-14B-Instruct 0.95 0.94 (↓0.01) 0.94 (↓0.01) 0.94 (↓0.01) 0.88 (↓0.07) 0.86 (↓0.09) Qwen2.5-32B-Instruct 0.95 0.94 (↓0.01) 0.93 (↓0.02) 0.92 (↓0.03) 0.92 (↓0.03) 0.85 (↓0.1) Qwen2.5-72B-Instruct 0.94 0.94 (0.0) 0.94 (0.0) 0.92 (↓0.02) 0.91 (↓0.03) 0.84 (↓0.1) Qwen2.5-7B-Instruct-1M 0.91 0.91 (0.0) 0.91 (0.0) 0.86 (↓0.05) 0.77 (↓0.14) 0.74 (↓0.17) Qwen2.5-14B-Instruct-1M 0.95 0.92 (↓0.03) 0.91 (↓0.04) 0.91 (↓0.04) 0.89 (↓0.06) 0.80 (↓0.15) Nemotron-70B-Instruct-HF 0.94 0.94 (0.0) 0.93 (↓0.01) 0.90 (↓0.04) 0.91 (↓0.03) 0.82 (↓0.12) Phi-3-mini-128k-Instruct 0.86 0.85 (↓0.01) 0.78 (↓0.08) 0.79 (↓0.07) 0.69 (↓0.17) 0.58 (↓0.28) Phi-3-small-128k-Instruct 0.88 0.84 (↓0.04) 0.88 (0.0) 0.83 (↓0.05) 0.78 (↓0.1) 0.70 (↓0.18) Phi-3-medium-128k-Instruct 0.89 0.84 (↓0.05) 0.84 (↓0.05) 0.81 (↓0.08) 0.72 (↓0.17) 0.63 (↓0.26) Palmyra-Fin-128k-Instruct 0.96 0.93 (↓0.03) 0.92 (↓0.04) 0.90 (↓0.06) 0.89 (↓0.07) 0.83 (↓0.13)

- Table 1: Robustness Results. Misspelled and incomplete queries seem manageable for all models; however, the most significant drop in performance, reaching up to 0.17 for Phi-3-mini-128k-Instruct and Phi-3-medium-128k-Instruct, is observed in cases involving OCRed queries. While the baseline performance appears relatively straightforward for all models, with scores ranging from 0.98 to 0.81, the point-wise minimum across all perturbations—indicative of robustness—reveals that models face challenges in consistently adapting to various input types. Even the most robust model, OpenAI o3-mini, experiences a decrease of 0.08 relative to the baseline. The best results in each category are in bold and second best are underlined.

disregard missing context and produce fabricated responses.

Compliance The trade-off between Context Grounding and Robustness is captured by the LLM Compliance score: while some models, like Palmyra-Fin-128k-Instruct, managed moderate Robustness (0.83) and satisfactory Context Grounding (0.80), achieving an optimal balance in Compliance scores (0.81), the biggest imbalance between Robustness and Context Grounding is seen in the second-best model in the Robustness category, DeepSeek-R1-Distill-Llama-70B (0.89 vs. 0.38). In our calculation of LLM Compliance, we used β = 0.5.

### 6 Related work

##### 6.1 LLMs Robustness Evaluation

A significant line of research examines the robustness of LLMs when challenged to directly handle and interpret raw user inputs. A crucial study in this area is the Holistic Evaluation of Language Models (HELM) by Liang et al. (2023). HELM investigates how LLMs manage both invariance and equivariance under varying conditions. The robustness of LLMs to invariance is assessed by

evaluating the consistency of their outputs under minor, semantics-preserving transformations, such as typographical errors or changes in capitalization. Regarding equivariance, the study examined the models’ responses to semantically altering modifications, to see if LLMs can appropriately adjust their outputs when the meaning of the input changes. This aspect was evaluated using Contrast Sets, which provide counterfactually augmented data for a limited set of datasets, like the BoolQ question answering dataset and the IMDB sentiment analysis scenario.

##### 6.2 Financial Benchmarks

FinBen (Xie et al., 2024) is an open-source evaluation framework, consisting of 36 datasets across 24 tasks, including areas like risk management and text generation, and introduces tasks like stock trading using the Cattell-Horn-Carroll theory. FinDABench (Liu et al., 2024b) assesses foundational, reasoning, and technical skills of LLMs in financial data analysis, aimed at providing a robust analysis of LLM capabilities. FinanceBench (Islam et al., 2023), created by AI researchers and financial experts, tests LLMs against the top 100 questions from SEC filings and earnings reports. FinLMEval

Model Name Irrelevant Ctx No Ctx Ctx Grounding QA Ctx Grounding TG Ctx Grounding Robustness Compliance Gemini 2.0 Flash Exp 0.81 0.66 0.77 0.46 0.74 0.83 0.76 Gemini 1.5 Pro 002 0.74 0.64 0.72 0.53 0.69 0.84 0.72 OpenAI GPT-4o 0.52 0.43 0.50 0.25 0.47 0.85 0.52 OpenAI o1 0.56 0.55 0.57 0.45 0.55 0.81 0.59 OpenAI o3-mini 0.67 0.51 0.63 0.27 0.59 0.90 0.63 DeepSeek-R1-Distill-Llama-8B 0.32 0.27 0.30 0.25 0.30 0.64 0.34 DeepSeek-R1-Distill-Qwen-14B 0.49 0.21 0.36 0.27 0.35 0.82 0.40 DeepSeek-R1-Distill-Qwen-32B 0.54 0.24 0.40 0.35 0.39 0.86 0.44 DeepSeek-R1-Distill-Llama-70B 0.50 0.27 0.41 0.22 0.38 0.89 0.43 DeepSeek-R1 0.51 0.22 0.39 0.20 0.37 0.80 0.41 Meta-Llama-3.1-8B-Instruct 0.67 0.63 0.70 0.27 0.65 0.70 0.66 Meta-Llama-3.1-70B-Instruct 0.46 0.47 0.48 0.37 0.47 0.80 0.51 Meta-Llama-3.3-70B-Instruct 0.50 0.40 0.47 0.31 0.45 0.82 0.49 Qwen2.5-7B-Instruct 0.75 0.64 0.75 0.31 0.70 0.75 0.71 Qwen2.5-14B-Instruct 0.75 0.61 0.70 0.55 0.68 0.86 0.71 Qwen2.5-32B-Instruct 0.89 0.68 0.82 0.55 0.79 0.85 0.80 Qwen2.5-72B-Instruct 0.69 0.60 0.68 0.39 0.64 0.84 0.67 Qwen2.5-7B-Instruct-1M 0.63 0.58 0.65 0.29 0.60 0.74 0.62 Qwen2.5-14B-Instruct-1M 0.78 0.53 0.69 0.37 0.65 0.80 0.68 Nemotron-70B-Instruct-HF 0.52 0.48 0.52 0.39 0.50 0.82 0.54 Phi-3-mini-128k-Instruct 0.54 0.34 0.47 0.24 0.44 0.58 0.46 Phi-3-small-128k-Instruct 0.37 0.26 0.34 0.10 0.31 0.70 0.35 Phi-3-medium-128k-Instruct 0.36 0.25 0.33 0.14 0.30 0.63 0.34 Palmyra-Fin-128k-Instruct 0.95 0.66 0.83 0.65 0.80 0.83 0.81

- Table 2: Context Grounding Results. Missing context poses the biggest challenge for almost all tested models, except for Qwen2.5-32B-Instruct and Palmyra-Fin-128k-Instruct, which are also the most compliant models in our tests. The most robust model, OpenAI o3-mini, achieves 0.59 Context Grounding, leading to comparably low Compliance score of 0.63. Text generation queries (TG) achieve much lower Context Grounding results than question answering (QA) requests. In our calculation of LLM Compliance, we used β = 0.5. The best results in each category are in bold and second best are underlined.

(Guo et al., 2023) compares various LLMs, including specialized models in financial NLP tasks, highlighting the gap between general and specialized model performance. FLUE focuses on language understanding within finance, while PIXIU (Xie et al.,

- 2023) evaluates instruction-tuned LLMs for tasks like investment strategies and market predictions.

6.3 Hallucination Detection and Measurement Researchers have developed multiple methods to detect hallucinations in large language models (LLMs). For instance, Valentin et al. (2024) analyzed LLM inference dynamics, achieving an 88% success rate in detecting hallucinatory predictions. Additionally, a comprehensive overview of these techniques was surveyed by Du et al. (2023), while Xu et al. (2024a) introduced a detection method via the HaluEval 2.0 benchmark. Studies have also aimed to quantify hallucination occurrences. For example, Chelli et al. (2024) noted varying hallucination rates across different LLM versions, and another paper (Hong et al., 2024) introduced a Hallucinations Leaderboard to compare models. Theoretical approaches include Li et al. (2024)’s learning theory analysis indicating that LLMs inherently hallucinate, and a study in Nature proposing new statistical methods to identify specific types of

hallucinations (Jiang et al., 2024). Further, Ji et al. (2024) offered a detailed framework for evaluating LLM hallucinations.

### 7 Conclusions

We have introduced a new benchmark, FailSafeQA, where we redefine robustness with a new approach, emphasizing a user interface-oriented perspective—identifying what the user has access to and what could potentially go wrong. We defined a new metric inspired by the precision-recall tradeoff—LLM Compliance—that balances model factuality (Context Grounding) with model resilience against semantically equivalent input perturbations (Robustness).

Indeed, the model with the best Context Grounding score, Palmyra-Fin-128k-Instruct, despite its strong baseline, failed to maintain robust predictions in 17% of test cases. On the other hand, the most robust model, OpenAI o3-mini, fabricated information in 41% of tested cases. The hypothesis of negative correlation between model’s Robustness and Context Grounding requires further research.

The results for the Context Grounding issue are particularly troubling — thinking style models like

DeepSeek-R1 and DeepSeek-R1-Distill or OpenAI-

- o1/o3-mini models, which are generally praised for reasoning tasks like inference outputs, fabricate information in 41% to 70% of test cases. While some problems, such as failed document uploads
- or degraded OCRed documents, can be detected

- at the software level before querying, irrelevant context cannot. This requires that users already possess some knowledge about the document they are querying. We find that models like Qwen and Palmyra-Fin are more suitable for tasks requiring precise information.

We have found that text generation queries (e.g., generating a blog post) requesting information retrieval are the most susceptible to hallucinations. We hypothesize that a more effective approach might involve, for example, first extracting information through a QA query before integrating it into a blog post.

The benchmark primarily focuses on the longcontext financial domain. However, our methodology can be easily adapted to other domains and varied context lengths; the primary adjustment involves the definition of out-of-domain query mapping.

The top-performing model, Palmyra-Fin-128kInstruct, achieved a Compliance score of 0.81, highlighting significant potential for improvement. We hope FailSafeQA will differentiate fail-safe compliant models from stochastic parrots (Bender et al., 2021).

Reproducibility Statement To ensure the reproducibility of our results, we have made our benchmark publicly available, including all prompts and inference parameters used during the evaluation. The data can be accessed on the HuggingFace Hub at https://huggingface.co/ datasets/Writer/FailSafeQA. For our evaluations, we utilized Qwen2.5-72B-Instruct as the LLM-as-a-Judge, which is available under a permissive license.

Limitations Our benchmark effectively identifies robustness and context grounding issues in models across various long-context financial tasks. As LLMs expand into new financial applications, we aim to continually update the dataset with new skills. The benchmark’s design prioritizes computational affordability and accuracy in judging, currently limiting queries to specific citations. Future work will broaden the dataset’s scope to include

tasks requiring information aggregation across multiple sources.

### References

Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck, Sébastien Bubeck, Martin Cai, Qin Cai, Vishrav Chaudhary, Dong Chen, Dongdong Chen, Weizhu Chen, Yen-Chun Chen, Yi-Ling Chen, Hao Cheng, Parul Chopra, Xiyang Dai, Matthew Dixon, Ronen Eldan, Victor Fragoso, Jianfeng Gao, Mei Gao, Min Gao, Amit Garg, Allie Del Giorno, Abhishek Goswami, Suriya Gunasekar, Emman Haider, Junheng Hao, Russell J. Hewett, Wenxiang Hu, Jamie Huynh, Dan Iter, Sam Ade Jacobs, Mojan Javaheripi, Xin Jin, Nikos Karampatziakis, Piero Kauffmann, Mahoud Khademi, Dongwoo Kim, Young Jin Kim, Lev Kurilenko, James R. Lee, Yin Tat Lee, Yuanzhi Li, Yunsheng Li, Chen Liang, Lars Liden, Xihui Lin, Zeqi Lin, Ce Liu, Liyuan Liu, Mengchen Liu, Weishung Liu, Xiaodong Liu, Chong Luo, Piyush Madan, Ali Mahmoudzadeh, David Majercak, Matt Mazzola, Caio César Teodoro Mendes, Arindam Mitra, Hardik Modi, Anh Nguyen, Brandon Norick, Barun Patra, Daniel Perez-Becker, Thomas Portet, Reid Pryzant, Heyang Qin, Marko Radmilac, Liliang Ren, Gustavo de Rosa, Corby Rosset, Sambudha Roy, Olatunji Ruwase, Olli Saarikivi, Amin Saied, Adil Salim, Michael Santacroce, Shital Shah, Ning Shang, Hiteshi Sharma, Yelong Shen, Swadheen Shukla, Xia Song, Masahiro Tanaka, Andrea Tupini, Praneetha Vaddamanu, Chunyu Wang, Guanhua Wang, Lijuan Wang, Shuohang Wang, Xin Wang, Yu Wang, Rachel Ward, Wen Wen, Philipp Witte, Haiping Wu, Xiaoxia Wu, Michael Wyatt, Bin Xiao, Can Xu, Jiahang Xu, Weijian Xu, Jilong Xue, Sonali Yadav, Fan Yang, Jianwei Yang, Yifan Yang, Ziyi Yang, Donghan Yu, Lu Yuan, Chenruidong Zhang, Cyril Zhang, Jianwen Zhang, Li Lyna Zhang, Yi Zhang, Yue Zhang, Yunan Zhang, and Xiren Zhou. 2024. Phi-3 technical report: A highly capable language model locally on your phone. Preprint, arXiv:2404.14219.

Emily M. Bender, Timnit Gebru, Angelina McMillanMajor, and Shmargaret Shmitchell. 2021. On the dangers of stochastic parrots: Can language models be too big? In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’21, page 610–623, New York, NY, USA. Association for Computing Machinery.

Steven Bird, Ewan Klein, and Edward Loper. 2009. Natural language processing with Python: analyzing text with the natural language toolkit. " O’Reilly Media, Inc.".

Mikaël Chelli, Jules Descamps, Vincent Lavoué, Christophe Trojani, Michel Azar, Marcel Deckert, Jean-Luc Raynier, Gilles Clowez, Pascal Boileau, and Caroline Ruetsch-Chelli. 2024. Hallucination

rates and reference accuracy of chatgpt and bard for systematic reviews: Comparative analysis. Journal of Medical Internet Research, 26(1):e53164.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. Preprint, arXiv:2501.12948.

Li Du, Zhuoye Ding, Zhuoye Ding, Xuanjing Huang, and Zhongyu Wei. 2023. Quantifying and attributing the hallucination of large language models via association analysis. arXiv preprint arXiv:2309.05217.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Let-

man, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Yue Guo, Zian Xu, and Yi Yang. 2023. Is ChatGPT a financial expert? evaluating language models on financial natural language processing. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 815–821, Singapore. Association for Computational Linguistics.

Giwon Hong, Eunsol Jang, Jeonghyeon Kim, Jamin Shin, Soyoung Yoon, Sungdong Yoon, Jongwon Shin, and Minjoon Seo. 2024. The hallucinations leaderboard – an open effort to measure hallucinations in large language models. arXiv preprint arXiv:2404.05904.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, and Boris Ginsburg. 2024. RULER: What’s the real context size of your long-context language models? In First Conference on Language Modeling.

Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, and Bertie Vidgen. 2023. Financebench: A new benchmark for financial question answering. arXiv preprint arXiv:2311.11944.

Ziwei Ji, Yuzhe Gu, Wenwei Zhang, Chengqi Lyu, Dahua Lin, and Kai Chen. 2024. ANAH: Analytical annotation of hallucinations in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8135–8158, Bangkok, Thailand. Association for Computational Linguistics.

Che Jiang, Biqing Qi, Xiangyu Hong, and Dayuan Fu. 2024. On large language models’ hallucination with regard to known facts. arXiv preprint arXiv:2403.20009.

John D. Lee and Katrina A. See. 2004. Trust in automation: Designing for appropriate reliance. Human Factors, 46(1):50–80.

Junyi Li, Jie Chen, Ruiyang Ren, Xiaoxue Cheng, Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. 2024. The dawn after the dark: An empirical study on factuality hallucination in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10879–10899, Bangkok, Thailand. Association for Computational Linguistics.

Yinheng Li, Shaofei Wang, Han Ding, and Hang Chen. 2023. Large language models in finance: A survey. In Proceedings of the Fourth ACM International Conference on AI in Finance, ICAIF ’23, page 374–382, New York, NY, USA. Association for Computing Machinery.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan,

Ce Zhang, Christian Alexander Cosgrove, Christopher D Manning, Christopher Re, Diana AcostaNavas, Drew Arad Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue WANG, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri S. Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Andrew Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta Koreeda. 2023. Holistic evaluation of language models. Transactions on Machine Learning Research. Featured Certification, Expert Certification.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024a. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173.

Shu Liu, Shangqing Zhao, Chenghao Jia, Xinlin Zhuang, Zhaoguang Long, Jie Zhou, Aimin Zhou, Man Lan, Qingquan Wu, and Chong Yang. 2024b. Findabench: Benchmarking financial data analysis ability of large language models. arXiv preprint arXiv:2401.02982.

Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2022. Fantastically ordered prompts and where to find them: Overcoming fewshot prompt order sensitivity. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8086–8098, Dublin, Ireland. Association for Computational Linguistics.

Carsten Maple, Lukasz Szpruch, Gregory Epiphaniou, Kalina Staykova, Simran Singh, William Penwarden, Yisi Wen, Zijian Wang, Jagdish Hariharan, and Pavle Avramovic. 2023. The ai revolution: Opportunities and challenges for the finance sector. arXiv preprint arXiv:2308.16538.

Filip Ginter Mathieu Timmerman. 2023. Ocr errors simulator.

OpenAI, :, Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David

Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, Ilge Akkaya, Ilya Kostrikov, Ilya Sutskever, Irina Kofman, Jakub Pachocki, James Lennon, Jason Wei, Jean Harb, Jerry Twore, Jiacheng Feng, Jiahui Yu, Jiayi Weng, Jie Tang, Jieqi Yu, Joaquin Quiñonero Candela, Joe Palermo, Joel Parish, Johannes Heidecke, John Hallman, John Rizzo, Jonathan Gordon, Jonathan Uesato, Jonathan Ward, Joost Huizinga, Julie Wang, Kai Chen, Kai Xiao, Karan Singhal, Karina Nguyen, Karl Cobbe, Katy Shi, Kayla Wood, Kendra Rimbach, Keren Gu-Lemberg, Kevin Liu, Kevin Lu, Kevin Stone, Kevin Yu, Lama Ahmad, Lauren Yang, Leo Liu, Leon Maksin, Leyton Ho, Liam Fedus, Lilian Weng, Linden Li, Lindsay McCallum, Lindsey Held, Lorenz Kuhn, Lukas Kondraciuk, Lukasz Kaiser, Luke Metz, Madelaine Boyd, Maja Trebacz, Manas Joglekar, Mark Chen, Marko Tintor, Mason Meyer, Matt Jones, Matt Kaufer, Max Schwarzer, Meghan Shah, Mehmet Yatbaz, Melody Y. Guan, Mengyuan Xu, Mengyuan Yan, Mia Glaese, Mianna Chen, Michael Lampe, Michael Malek, Michele Wang, Michelle Fradin, Mike McClay, Mikhail Pavlov, Miles Wang, Mingxuan Wang, Mira Murati, Mo Bavarian, Mostafa Rohaninejad, Nat McAleese, Neil Chowdhury, Neil Chowdhury, Nick Ryder, Nikolas Tezak, Noam Brown, Ofir Nachum, Oleg Boiko, Oleg Murk, Olivia Watkins, Patrick Chao, Paul Ashbourne, Pavel Izmailov, Peter Zhokhov, Rachel Dias, Rahul Arora, Randall Lin, Rapha Gontijo Lopes, Raz Gaon, Reah Miyara, Reimar Leike, Renny Hwang, Rhythm Garg, Robin Brown, Roshan James, Rui Shu, Ryan Cheu, Ryan Greene, Saachi Jain, Sam Altman, Sam Toizer, Sam Toyer, Samuel Miserendino, Sandhini Agarwal, Santiago Hernandez, Sasha Baker, Scott McKinney, Scottie Yan, Shengjia Zhao, Shengli Hu, Shibani Santurkar, Shraman Ray Chaudhuri, Shuyuan Zhang, Siyuan Fu, Spencer Papay, Steph Lin, Suchir Balaji, Suvansh Sanjeev, Szymon Sidor, Tal Broda, Aidan Clark, Tao Wang, Taylor Gordon, Ted Sanders, Tejal Patwardhan, Thibault Sottiaux, Thomas Degry, Thomas Dimson, Tianhao Zheng, Timur Garipov, Tom Stasi, Trapit Bansal, Trevor Creech, Troy Peterson, Tyna Eloundou, Valerie Qi, Vineet Kosaraju, Vinnie Monaco, Vitchyr Pong, Vlad Fomenko, Weiyi Zheng, Wenda Zhou, Wes McCabe, Wojciech Zaremba, Yann Dubois, Yinghai Lu, Yining Chen, Young Cha, Yu Bai, Yuchen He, Yuchen Zhang, Yunyun Wang, Zheng Shao, and Zhuohan Li. 2024. Openai o1 system card. Preprint, arXiv:2412.16720.

OpenAI. 2024. Hello gpt-4o.

Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. 2024. Quantifying language models’ sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting. In The Twelfth International Conference on Learning Representations.

Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, Soroosh Mariooryad, Yifan Ding, et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. Preprint, arXiv:2403.05530.

The Alan Turing Institute. 2023. The impact of large language models in finance: Towards trustworthy adoption. Technical report, The Alan Turing Institute. Accessed on October 14, 2024.

United States Bankruptcy Court for the Southern District of Florida. 2020. Signatures and document retention. https: //www.flsb.uscourts.gov/local-rule/ signatures-and-document-retention. Accessed on October 10, 2024.

Simon Valentin, Jinmiao Fu, Gianluca Detommaso, Shaoyuan Xu, Giovanni Zappella, and Bryan Wang. 2024. Cost-effective hallucination detection for llms. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’24, New York, NY, USA. Association for Computing Machinery.

Zhilin Wang, Alexander Bukharin, Olivier Delalleau, Daniel Egert, Gerald Shen, Jiaqi Zeng, Oleksii Kuchaiev, and Yi Dong. 2024. Helpsteer2preference: Complementing ratings with preferences. Preprint, arXiv:2410.01257.

Qianqian Xie, Weiguang Han, Zhengyu Chen, Ruoyu Xiang, Xiao Zhang, Yueru He, Mengxi Xiao, Dong Li, Yongfu Dai, Duanyu Feng, Yijing Xu, Haoqiang Kang, Ziyan Kuang, Chenhan Yuan, Kailai Yang, Zheheng Luo, Tianlin Zhang, Zhiwei Liu, Guojun Xiong, Zhiyang Deng, Yuechen Jiang, Zhiyuan Yao, Haohang Li, Yangyang Yu, Gang Hu, Jiajia Huang, Xiao-Yang Liu, Alejandro Lopez-Lira, Benyou Wang, Yanzhao Lai, Hao Wang, Min Peng, Sophia Ananiadou, and Jimin Huang. 2024. Finben: A holistic financial benchmark for large language models. arXiv preprint arXiv:2402.12659.

Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, and Jimin Huang. 2023. PIXIU: A comprehensive benchmark, instruction dataset and large language model for finance. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Jingfeng Xu, Yikang Shen, Yanan Ou, Xiang Liang, Xing Xie, and Meng Jiang. 2024a. Measuring and reducing llm hallucination without gold-standard answers. arXiv preprint arXiv:2402.10412.

Ziyue Xu, Peilin Zhou, Xinyu Shi, Jiageng Wu, Yikang Jiang, Bin Ke, and Jie Yang. 2024b. Fintruthqa: A benchmark dataset for evaluating the quality of financial information disclosure. Preprint, arXiv:2406.12009.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. 2024. Qwen2 technical report. Preprint, arXiv:2407.10671.

Ziqi Yin, Hao Wang, Kaito Horio, Daisuke Kawahara, and Satoshi Sekine. 2024. Should we respect llms? a cross-lingual study on the influence of prompt politeness on llm performance. Preprint, arXiv:2402.14531.

Jiajie Zhang, Yushi Bai, Xin Lv, Wanjun Gu, Danqing Liu, Minhao Zou, Shulin Cao, Lei Hou, Yuxiao Dong, Ling Feng, and Juanzi Li. 2024. Longcite: Enabling llms to generate fine-grained citations in long-context qa. Preprint, arXiv:2409.02897.

Huaqin Zhao, Zhengliang Liu, Zihao Wu, Yiwei Li, Tianze Yang, Peng Shu, Shaochen Xu, Haixing Dai, Lin Zhao, Gengchen Mai, Ninghao Liu, and Tianming Liu. 2024. Revolutionizing finance with llms: An overview of applications and insights. arXiv preprint arXiv:2401.11641.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

### A Citation Sanitization

This postprosessing step aimed to improve citations extracted by LongCite-llama3.1-8b (Zhang et al.,

- 2024) model by extending them to include full sentences and providing more context.

- • Sentence Tokenization: Split context into sentences using NLTK (Bird et al., 2009), associating each with its character indices.
- • Citation Extension: Extend citations to include full sentences containing the citation boundaries, plus one sentence before and after for context.
- • Consolidation and Gluing: Combine extended citations, removing duplicates. Glue adjacent sentences, marking omissions with "[...]".
- • Final Formatting: Join citation chunks with separators, wrapping the entire citation to indicate it’s an excerpt.

We additionally prepended the first 10 sentences of every report to the citation. This method ensures that citations are more coherent and provide fuller context, while still indicating where text has been omitted. An average citation length for every sample was: 1.3k tokens, whereas the average original context had 24.6k tokens. We present an example of before and after below:

[{'start_sentence_idx': 169, 'end_sentence_idx': 169, 'start_char_idx': 30378, 'end_char_idx': 30677, → 'cite': 'Results of Operations\n\nYear Ended September 30, 2018, compared to Year Ended

→ September 30, 2017 (all references are to fiscal years)\n\nConsolidated\n\nConsolidated sales

→ decreased $1.3 million before the impact of intersegment sales, or 3%, to $47.4 million for

→ 2018 from $48.7 million for\n\n 2017.\xa0 '}, {'start_sentence_idx': 170, 'end_sentence_idx':

→ 170, 'start_char_idx': 30677, 'end_char_idx': 30833, 'cite': 'The decrease in sales was due

→ to a decrease in the Cable TV segment of $2.9 million, partially offset by an increase in the

→ Telco segment of $1.5 million.\n\n'}]

10-K 1 form10-k.htm ADDVANTAGE TECHNOLOGIES 10-K Securities registered under Section 12(g) of the Act

→ : None ADDVANTAGE TECHNOLOGIES GROUP, INC. FORM 10-K YEAR ENDED SEPTEMBER 30, 2018 INDEX 2

→ PART I Item 1.Business. Forward-Looking Statements Certain matters discussed in this report

→ constitute forward-looking statements, within the meaning of the Private Securities

→ Litigation Reform Act of 1995, including statements which relate to, among other things,

→ expectations of the business environment in which ADDvantage Technologies Group, Inc. (the "

→ Company", "We" or "ADDvantage") operates, projections of future performance, perceived

→ opportunities in the market and statements regarding our goals and objectives and other

→ similar matters. The words "estimates", "projects", "intends", "expects", "anticipates", " → believes", "plans", "goals", "strategy", "likely", "may", "should" and similar expressions → often identify forward-looking statements. These forward-looking statements are found at

→ various places throughout this report and the documents incorporated into it by reference.

→ These and other statements, which are not historical facts, are hereby identified as "forward-

→ looking statements" for purposes of the safe harbor provided by Section 21E of the Securities → Exchange Act of 1934, as amended, and Section 27A of the Securities Act of 1933, as amended. → These statements are subject to a number of risks, uncertainties and developments beyond our

→ control or foresight, including changes in the cable television and telecommunications → industries, changes in customer and supplier relationships, technological developments, → changes in the economic environment generally, the growth or formation of competitors,

→ changes in governmental regulation or taxation, changes in our personnel, our ability to

→ identify, complete and integrate acquisitions on favorable terms and other such factors. Our

→ actual results, performance or achievements may differ significantly from the results,

→ performance or achievements expressed or implied in the forward-looking statements. We do not

→ undertake any obligation to publicly release any revisions to these forward-looking

→ statements to reflect events or circumstances after the date of this report or to reflect the

→ occurrence of unanticipated events. Background The Company was incorporated under the laws

→ of Oklahoma in September 1989 as "ADDvantage Media Group, Inc." In December 1999, its name

→ was changed to ADDvantage Technologies Group, Inc. Our headquarters are located in Broken

→ Arrow, Oklahoma. We (through our subsidiaries) distribute and service a comprehensive line of → electronics and hardware for the cable television and telecommunications industries. We also → provide equipment repair services to cable operators. [...]

----[...] A deposit of $500,000 was paid on December 27, 2018 in connection with signing the purchase → agreement. Results of Operations Year Ended September 30, 2018, compared to Year Ended

→ September 30, 2017 (all references are to fiscal years) Consolidated Consolidated sales

→ decreased $1.3 million before the impact of intersegment sales, or 3%, to $47.4 million for

→ 2018 from $48.7 million for 2017. The decrease in sales was due to a decrease in the Cable TV

→ segment of $2.9 million, partially offset by an increase in the Telco segment of $1.5

→ million. Consolidated gross profit decreased $3.6 million, or 24%, to $11.2 million for 2018 → from $14.8 million for 2017. The decrease in gross profit was due to a decrease in the Cable → TV segment of $3.9 million, partially offset by an increase in the Telco segment of $0.3

→ million. [...]

### B Example of OCR Errors Injection into Context

- B.1 Original Context

In December 2009, we formed and organized a new wholly-owned subsidiary, Athena Minerals, Inc. (" → Athena Minerals") to take an assignment of a Sale and Purchase Agreement and Joint Escrow

→ Instructions dated December 4, 2009 (the "Purchase Agreement"). The Purchase Agreement

→ granted us an option to purchase a 413 acre group of patented mining claims (the "cLangtry

→ Property") located in the Calico Mining District at the base of the Calico Mountains

→ northeast of Barstow, California.

In March 2010, we entered into a Mining Lease with Option to Purchase (the "cLangtry Lease" or the "

→ cLease") which superseded the Purchase Agreement and granted us a 20 year lease to develop

→ and conduct mining operations on the Langtry Property, also with an option to purchase.

- B.2 OCR Corrupted Context

In December 2009, we formed and organized a new whorly-owned sUbsidiary, Athera Mineralsi Inci. ("

→ Athena Minerals") to take an assignment ofa Sale, and Purchae Agreement and Jont Escrow

→ Instructianf.dated December 4, 2009i(the "Purchale Agrement"). The P rahaje Agreement

→ grantedus ancoption to purchase a 413 cre, group of patfnted mi-ning claims (the "GcLngtry

→ Propety") lfcated in the 'Caclico minig Diseict at the base of th' Calico Muntains nortbast

→ of Barstow, California.

In Mrch 201o, we entered into a Mining lease iith Option to- Purc'bs (the "cLangtr' Lase" or th "

→ cLcafe") which Ruperseded the Purchase Agremet and granted us a 20 year lease to develop

→ andconducty mning operations n the Langtry Prodperty, also with n option to purchase.

#### C Prompts C.1 Conversation Generation Based on the provided context, generate a very long dialogue between an AI assistant and a user. The user is interested in:

- A. Question answering

- - questions about the context,
- - fact checking,
- - claim verification,

- B. Content generation

- - summarising all or selected files from the context,
- - comparing or combining facts from different documents in the context,
- - extracting useful information from the text,
- - creating a long or short form text based on the extracted information (ex. blog, LinkedIn post,

→ email, short note),

- - creating html tables with numbers mentioned in the text,
- - formatting the output (like adding or removing headlines, style correction, adding more details etc

###### → ).

Keep 80/20 ratio between question answering (A) and content generation (B) types of queries. This → means the user uses AI assistant as a search engine four times more often than asks it to

→ generate the content. Assume that users work for the company most often mentioned in the context. If asked about formatting or a style change, always apply suggested corrections right away.

When the user mentions files by names, always refer to the context files. Sometimes the user might mention different file formats when referring to files (like ppt), but it is

→ ok. You have access to the raw text of files.

Keep in mind that users use the AI assistant as a tool only, so they are very straightforward:

- - users generally don't bother being very polite
- - users often don't make requests in full sentences nor are they usually so wordy with "filler"

→ content like "I want to start by...", "Brilliant!", and "Just one last thing"

- - users tend to be more lazy and make more unclear references e.g., they will just be like "turn

→ these into a comprehensive concise paragraph", "summarize this"

- - the typical user will be more likely to "command" the model, rather than say something like "can

→ you assist with that"

- - users can ask AI whether it can read/process/access files. AI always answers it can.

Here are examples of questions and answers related to a different context: {examples}

Provide the format as json lines with keys: USER, ASSISTANT.

##### C.2 Question-Answer Pair Filtering

You are a quality assurance expert for question-answering datasets. Your task is to evaluate the

→ quality of question-answer pairs based on the given context. Please assess the following: Context: {context} Question: {question} Answer: {answer} Evaluate the quality of the question-answer pair using the following criteria:

- 1. Relevance: Is the question directly related to the context?
- 2. Clarity: Is the question clear and unambiguous?
- 3. Complexity: Does the question require reasoning or simple fact retrieval?
- 4. Accuracy: Is the answer correct based on the given context?
- 5. Completeness: Does the answer fully address the question?

For each criterion, provide a score from 1-5 (1 being lowest, 5 being highest) and a brief

→ explanation.

Then, give an overall quality assessment:

- - High quality: Meets all criteria well (4-5 in all categories)
- - Medium quality: Meets most criteria adequately (3-4 in most categories)
- - Low quality: Falls short on multiple criteria (1-2 in multiple categories) Provide your final verdict on whether this question-answer pair should be included in a high-quality

→ dataset, strictly following this format: "[[rating]]" for example: "Rating: [[5]]"

C.3 Judging Criteria

- C.3.1 Answerable Query

CRITERIA = """CRITERIA: The answer is completely accurate and comprehensive, extending the ground

→ truth with relevant and factual information from the context. SCORE: 6

--CRITERIA: The answer is completely accurate and contains the same information as the ground truth.

- SCORE: 5

--CRITERIA: The answer has minor omissions, but all presented facts are accurate and relevant to the

→ ground truth.

- SCORE: 4

--CRITERIA: The answer has moderate relevance to the ground truth but contains inaccuracies.

- SCORE: 3

---

CRITERIA: The answer has minor relevance to the ground truth but contains inaccuracies.

- SCORE: 2

--CRITERIA: The answer is completely unrelated to the ground truth.

- SCORE: 1"""

C.3.2 Unanswerable Query: Missing Context

CRITERIA: The model correctly identifies that no context is provided, refrains from answering,

→ explicitly states the need for context to provide a relevant response, and suggests uploading

→ the necessary documents or refining the question. SCORE: 6

--CRITERIA: The model appropriately recognizes that no context is available, refrains from answering,

→ and advises the user to provide context relevant to their inquiry. SCORE: 5

--CRITERIA: The model identifies the absence of context and refrains from providing an answer, but does

→ not offer guidance on how to rectify the issue. SCORE: 4

--CRITERIA: The model acknowledges the lack of provided context yet still attempts to answer based on

→ general knowledge or assumptions. SCORE: 3

--CRITERIA: The model attempts to answer the question while noting uncertainty or the absence of needed

→ context.

- SCORE: 2

--CRITERIA: The model provides an answer without recognizing or acknowledging the lack of provided

→ context.

- SCORE: 1

C.3.3 Unanswerable Query: Irrelevant Context

CRITERIA: The model correctly identifies the question as out of scope, refrains from answering,

→ explains its limitations clearly, and suggests either rephrasing the question or asking about

→ topics within the given context. SCORE: 6

--CRITERIA: The model correctly identifies the question as out of scope, refrains from answering, and

→ suggests asking about topics related to the given context. SCORE: 5

--CRITERIA: The model correctly identifies the question as out of scope and refrains from answering,

→ but doesn't suggest alternatives or clarify its limitations. SCORE: 4

--CRITERIA: The model admits the context doesn't contain relevant information but still provides a

→ speculative or general answer. SCORE: 3

--CRITERIA: The model attempts to answer the question but expresses uncertainty or mentions the lack of

→ context.

- SCORE: 2

--CRITERIA: The model answers the question without acknowledging it's out of scope. SCORE: 1

##### C.4 Judging Prompt

[Instruction] Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant

→ based on the context, ground truth and the user question displayed below.

Begin your evaluation by providing a short explanation. Be as objective as possible. After providing → your explanation, you must rate the response on a scale of 1 to 6 by strictly following this

→ format: "[[rating]]", for example: "Rating: [[5]]". {criteria}

[The Start of Context] {context} [The End of Context]

[Question] {question}

[The Start of Ground Truth] {reference} [The End of Ground Truth]

[The Start of Assistant's Answer] {prediction} [The End of Assistant's Answer]

