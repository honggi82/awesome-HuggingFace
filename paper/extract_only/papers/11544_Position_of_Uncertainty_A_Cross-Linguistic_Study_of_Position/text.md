# arXiv:2505.16134v3[cs.CL]11Dec2025

## Beyond Early-Token Bias: Model-Specific and Language-Specific Position Effects in Multilingual LLMs

Menschikov Mikhail* ITMO Applied AI

Alexander Kharitonov* SberAI

Maiia Kotyga Applied AI

Vadim Porvatov Sber

Anna Zhukovskaya Lomonosov MSU

David Kagramanyan HSE University

Egor Shvetsov† Applied AI

Evgeny Burnaev Applied AI AIRI

Abstract

Large Language Models (LLMs) exhibit position bias systematically underweighting information based on its location in the context but how this bias varies across languages and models remains unclear. We conduct a multilingual study across five typologically diverse languages (English, Russian, German, Hindi, Vietnamese) and five model architectures, analyzing how position bias interacts with prompting strategies and affects output entropy. Our key findings are: (1) Position bias is primarily model-driven but shows language-specific nuances. Notably, Qwen2.57B-Instruct, DeepSeek 7B Chat and Mistral 7B consistently favor late positions challenging the common assumption of universal earlytoken preference. (2) Explicitly instructing the model, in the presence of irrelevant distractors, that “the most relevant context to the query is marked as 1” unexpectedly reduces accuracy across all languages, questioning standard prompt-engineering practices. (3) Accuracy consistently drops most when relevant information appears in the middle of the context, yet this is not reflected in a corresponding increase in output entropy, suggesting the model remains confident even when it fails to use midcontext cues.

### 1 Introduction

Many recent applications based on large language models (LLMs) require support for long-context processing1. However, developing new training strategies to accommodate longer contexts is insufficient, as new challenges continue to arise. One notable issue is position bias – the systematic neglect of information located at specific positions, typically in the middle of the context (Baker et al., 2024)

1Such applications may include Retrieval-Augmented Generation, autonomous agents, scientific research, customer support, among others.

While position bias has been well-documented in English-centric studies (Zhang et al., 2024b; Baker et al., 2024), its manifestation in multilingual contexts and across-various architectures remains underexplored. Furthermore, existing bias mitigation strategies (Peysakhovich and Lerer, 2023; Zhao et al., 2021; Zhang et al., 2024a; Guo and Vosoughi, 2024) have predominantly been evaluated on English-language datasets. As discussed in Section 2, lexical and morphological variations across languages necessitate careful consideration in multilingual modeling. As a step toward adapting multilingual large language models for realworld long-context applications, this work seeks to address the following research questions:

- Q1: Is position bias primarily a model-driven phenomenon, or do language-specific patterns emerge due to lexical, morphological, and syntactic differences (Ghosh et al., 2024a)?
- Q2: Do prompt-based strategies, e.g., explicit position guidance (Zhang et al., 2024a), effectively mitigate bias across languages? Can we focus model attention via prompting, and would it improve model performance?
- Q3: Finally, we are interested in a deeper understanding of how position bias affects model generation, and thus we perform formal and empirical analysis of how position bias affects entropy of the output distribution.

Contributions. We present a multilingual analysis of position bias across five typologically diverse languages (English, German, Russian, Hindi, and Vietnamese) and five model architectures. Our analysis is based on 2,000 sampled examples per language, where each model was evaluated under nine distinct experimental conditions (3 context positions × 3 scoring strategies), yielding 18,000 model generations and evaluations per language. Across all five languages and five models, this totals 450,000 evaluated question–answer pairs.

Our key contributions are:

- • We show that position bias is predominantly model-driven, yet exhibits significant language-specific variations. For instance, models like Qwen2.5-7B-Instruct (Qwen et al., 2025), DeepSeek-7B-Chat (DeepSeekAI, 2024) and Mistral 7B display a strong lateposition bias, contradicting prior claims of an inherent early-token preference in LLMs (Wu et al., 2025; Barbero et al., 2025). In contrast, Llama3.1-8B-Instruct (Touvron et al., 2023) prioritizes early positions. We speculate these differences stem from variations in training data and model architecture.
- • We find that explicitly instructing models about correct context placement (e.g., “the correct context has label 1”) consistently degrades accuracy across all languages. This result holds when the incorrect context is sampled randomly, a setup that differs from Zhang et al. (2024a), who report that instructions can mitigate position bias. The key distinction is that Zhang et al. (2024a) use semantically relevant distractors, whereas our distractors are random.
- • Through an empirical analysis of output entropy, we reveal a counterintuitive dynamic: while model accuracy is lowest when relevant context is in the middle, this performance drop is not accompanied by a corresponding peak in entropy. This indicates a disconnect between positional disadvantage and the model’s expressed uncertainty.

These findings yield the following practical implications:

- 1. Chain-of-Thought and other reasoning strategies that rely on explicit positional guidance demand careful reconsideration, as our results show they can cause consistent performance degradation.
- 2. Retrieval-augmented generation (RAG) systems should account for specific model and language characteristics. Strategies that reorder context based on the assumption that models prioritize recent or initial tokens (see Section 2) may be ineffective or detrimental for models with a late-position bias.
- 3. The observed entropy dynamics complicate uncertainty-based bias mitigation techniques (Duan et al., 2024) and are crucial

for developing effective uncertainty quantification strategies in RAG pipelines.

The code for our experiments is available in the GitHub repository.

### 2 Related Work

What causes position bias? Prior work identifies multiple contributing factors. (Zhang et al., 2024b) attribute position bias to U-shaped attention patterns in transformers, which prioritize extremal positions and degrade performance for mid-context evidence. Theoretical and empirical studies further demonstrate that transformers attention is inherently biased toward earlier tokens (Wu et al., 2025; Barbero et al., 2025). (Wu et al., 2025) explains position bias due to "causal masking inherently biases attention toward earlier positions, as tokens in deeper layers attend to increasingly contextualized representations of earlier tokens". Our results reveal exceptions where later positions are desired, highlighting the complexity of the problem. Training data biases, such as serial-position effects in corpora, shape how models prioritize sequence positions (Wu et al., 2025; Guo and Vosoughi, 2024).

Interplay of Culture, Language, and Model Design. The way we perceive the world is influenced not only by our culture (Oyserman and Lee, 2008) but also by the language we speak (Boroditsky et al., 2003). The latter point is particularly relevant for LLMs, since they are trained on specific languages. Recent studies have shown that multilingual LLMs often initiate their "thinking process" in English, pivoting to the prompt’s original language in the middle layers (Zhang et al., 2024b; Peysakhovich and Lerer, 2023). These models exhibit lower lexical naturalness in nonEnglish languages, with the naturalness gap being more pronounced for languages structurally distant from English (Guo et al., 2024). While the volume of training data plays a crucial role (Arnett and Bergen, 2025), linguistic complexity — including lexical and morphological variations across languages — must also be considered (Ghosh et al., 2024a; Dang et al., 2024; Ismayilzada et al., 2025). Additionally, architectural design choices affect languages in different ways; for instance, removing positional encoding from language models would most degrade performance in languages with limited morphological systems (Ghosh et al., 2024a).

At the same time, most of bias mitigation approaches evaluate their performance in En-

glish (Zhang et al., 2024b; Peysakhovich and Lerer,

- 2023; Zhang et al., 2024a; Yu et al., 2024b; Wang

- et al., 2024b). These approaches fall into two categories: prompt-based techniques and architectural interventions. Architectural methodssuch as positional encodings, alternative masking schemes, and calibration mechanisms—address root causes but often require retraining and introduce computational overhead (Zhang et al., 2024b; Wu et al., 2025; Zhao et al., 2021). Prompt-based strategies, including query-aware contextualization and recency prompting, aim to redirect attention dynamically (Peysakhovich and Lerer, 2023; Wang et al., 2024b; Yu et al., 2024b). We focus on prompt-based strategies, using as a starting point the work done by (Zhang et al., 2024a). The authors studied whether a model can improve its performance when given explicit placement of the correct answer. They used two types of instructions — relative and absolute — and found that models lack relative awareness and that implicit information about absolute placement of the correct prompt marginally improves model performance. The main difference with this approach and ours is that authors in (Zhang et al., 2024a) use relevant distractors, while in our work we use random distractors.

Practical considerations. Position bias affects Chain of Thought Strategies (CoT). CoT struggles with position bias even when reasoning steps are correct, models often fail to retrieve evidence from middle positions (Zhang et al., 2024b). In (Zhang et al., 2024b; Yu et al., 2024b), authors analyzed error propagation in multi-hop reasoning. In RAG systems, one of the mitigating strategies is context ordering (Wang et al., 2024a; Alessio et al., 2024; Jin et al., 2024) (we discuss these approaches in Appendix A). While conventional approaches often assume a monotonic relationship between document position and attention (e.g., prioritizing the first/last positions), our analysis reveals that position bias may exhibit language-specific patterns and is not always maximized at early tokens. This observation challenges assumptions in methods like Long-Context RAG, which rely on fixed position prioritization, and highlights the need for language-adaptive reordering.

While predictive entropy is widely used to quantify model uncertainty (Huang et al., 2024; Sychev

- et al., 2025), its relationship with position bias remains unexplored. (Duan et al., 2024) note that uncertainty estimates can be token-biased, but how

position bias interacts with uncertainty dynamics is unclear.

### 3 Methods

#### 3.1 Position bias formalization

To evaluate the effect of position bias, we consider a question-answering task. For each question Q, we assume the existence of a ground truth answer A and a set of N contexts ctx1,...,ctxN. In our experimental setup, exactly one context is relevant to each question, while the remaining N − 1 contexts are randomly sampled from the dataset. The response of the model, denoted as Model, using a prompt function Prompt, to the question Q when the relevant context is placed at position i is given by: Ri = Model(Prompt(Q,ctx1,...,ctxN,i)). We define that the model exhibits position bias on dataset D toward position i over position j if the expected Accuracy when the relevant context is at position i exceeds that at position j:

[Acc(A,Ri)] > E

[Acc(A,Rj)]

E

(Q,A)∼D

(Q,A)∼D

#### 3.2 Context Placement Strategies

This experimental series examines whether explicit information regarding the relevance of contexts to a given query can improve performance. We investigate a practical scenario in which context relevance can be quantified (e.g., through cosine similarity between context and question embeddings) and assess the impact of integrating relevance scores into prompts on context selection. The experimental framework is illustrated in Figure 1.

Scoring Strategies. Three distinct scoring strategies are evaluated. The Aligned strategy assigns a relevance score of rs = 1 to contextually relevant information and rs = 0 to all other contexts. The All Zero strategy assigns rs = 0 to all contexts, including those that are relevant, to test the hypothesis that intentional mislabeling will degrade model performance if relevance scores influence context selection. The No Scores strategy omits relevance scores entirely from the prompts to assess whether explicit scoring contributes to model efficacy. In the experimental setup, relevance scores rs1,...,rsN were incorporated into the prompt where applicable. As illustrated in Figure 1, three positions for relevant contexts are considered: TOP (first), MIDDLE (N/2), and BOTTOM (last).

Example question: who wrote the book The Story of Tibet?

| | | |
|---|---|---|
| | |[Figure 1]|
| | | |
| | |[Figure 2]|

Example prompt: Answer the question using the available information from the texts in the list below. Each text has a corresponding real-value score of its relevance to the question in square brackets at the beginning. Scores are ranged from 0.0 (the text is not suitable for generating an answer based on it) to 1.0 (the text is suitable for generating an answer based on it). Use this information. Choose texts with high enough relevance scores. If, based on the specified scores, there are no texts in the list that are relevant enough to generate answer based on them, then generate the following answer: "I do not have an answer to your question". Generate answer only in English. Do not duplicate the question in the answer. Generate only the answer to the specified question. Answer need to be short. Do not generate anything extra.

Example answer: Thomas Laird Ground truth: Journalist and author Thomas Laird

Available information: <[score1] ctx1>, <[score2] ctx2>, <[score3] ctx3>, <[score4] ctx4>, <[score5] ctx5> Question: <question>

Figure 1: Experiment Structure. For each question, the relevant context is placed in one of three positions: top, middle, or bottom of the context list. Each context is assigned a binary score (0 or 1), indicating its relevance to the question. Three scoring strategies are evaluated: Aligned, where the relevant context receives a score of 1 and all others are assigned 0; All Zero, in which every context, including the relevant one, is scored 0; No Scores, where relevance scores are entirely omitted. This experimental design assesses the influence of scoring mechanisms and answer positioning on model performance under varying degrees of contextual guidance.

Context Volume. Prior research has shown that longer contexts can exacerbate position bias (Baker et al., 2024; Peysakhovich and Lerer, 2023). In our work, we default to five contexts based on preliminary experiments. We vary the number of contexts, setting N ∈ {5,10,15} , and find that while increasing N does degrade performance for some models, the effect is relatively modest. Given limited computational resources, we therefore opted to use five contexts in our main experiments. Preliminary results are provided in Appendix G.

#### 3.3 Average Predictive Entropy

We adopt the token-wise entropy framework introduced by (Lu et al., 2022), normalized by the total number of tokens, to quantify uncertainty in model responses (Lu et al., 2022; Wang et al., 2025). Let x represent the input prompt and s = {z1,z2,...,zn} denote a generated completion sequence of n tokens. For a given large language model (LLM), the conditional probability of generating the i-th token zi, given the preceding tokens s<i = {z1,...,zi−1} and the prompt x, is denoted as p(zi | s<i,x) for 1 ≤ i ≤ n. The average predictive entropy1 (denoted as PEavg) is defined as:

n

1 n

1 n

PEavg(s, x) = −

log p(s | x) =

− log p(zi | s<i, x).

i=1

(1)

This formulation computes the average uncer-

tainty per token by decomposing the joint probability p(s | x) into a product of conditional probabilities using the chain rule. The normalization by n ensures comparability across sequences of varying lengths, consistent with the interpretation of entropy as a measure of "average uncertainty".

### 4 Experiment Set-Up

#### 4.1 Datasets

In this study, we employed three open-ended question-answering datasets encompassing five languages characterized by divergent syntactic structures and semantic distributions. The statistics for these datasets are summarized in Table 1, with a comprehensive description provided in Appendix B. These datasets were selected for two principal reasons: (1) they are well-established within the research community, ensuring familiarity and reproducibility; (2) the context lengths for question-answer pairs are sufficiently concise (less than 4096 characters), allowing multiple instances to be included in a single prompt without exceeding the maximum sequence length constraints of the language model.

Preprocessing. To accommodate computational constraints, the analysis was limited to 2,000 question-answer (QA) pairs per language. A twostage preprocessing pipeline was applied prior to sampling to ensure data quality and consistency: (1) duplicate removal, in which all redundant QA

Language Source Size English SQuAD2.0 (Rajpurkar et al., 2018) 150k Russian MTS-SQuAD (link) 60k German MLQA (Lewis et al., 2020) 5k Hindi MLQA (Lewis et al., 2020) 5k Vietnamese MLQA (Lewis et al., 2020) 5k

Table 1: Summary of datasets utilized in this study, categorized by language, with corresponding sources and associated question-answer pairs sizes

pairs were excluded; (2) answer validation, where pairs lacking valid responses, such as those with missing or ambiguous answers, were discarded.

#### 4.2 Models

To investigate whether position bias in large language models (LLMs) arises from model-specific design and training choices or from languagespecific characteristics, we evaluate five popular open-source multilingual models: Qwen2.57B-Instruct(Qwen et al., 2025), Llama3-8BInstruct(Touvron et al., 2023), DeepSeek-7BChat(DeepSeek-AI, 2024), Gemma-7B-it(Team et al., 2024), and Mistral-7B-Instruct-v0.3(Jiang et al., 2023). All models support English, German, Russian, Vietnamese, and Hindi, yet differ in architecture and training paradigms. This selection enables us to disentangle model driven factors of position bias such as attention mechanisms and training data composition from language-related influences. It is important to note that DeepSeek7B-Chat performed poorly on Hindi, therefore, we excluded its Hindi results from our analysis.

5 Evaluation

#### 5.1 LLM as a Judge

Traditional statistical evaluation metrics of openended generations, such as BLEU (Papineni et al., 2002), ROUGE (Lin, 2004), and Meteor Universal (Denkowski and Lavie, 2014), are limited in their ability to differentiate between syntactically similar yet semantically distinct texts. Although semantic evaluation methods like BERTScore (Zhang et al., 2019) were developed to overcome these shortcomings, our experimental findings indicate that BERTScore exhibits insufficient discriminative power, frequently failing to capture subtle distinctions between correct and incorrect responses. Consequently, we employ the LLM as a judge framework (Zheng et al., 2023) and select Mistral

Large2 as the evaluator for the following reasons: (1) prior research demonstrates its strong alignment with human judgments and generalizability across diverse tasks (Bavaresco et al., 2024; Kim et al., 2024); (2) it provides a freely accessible API for research, facilitating large-scale evaluation; (3) its architectural design differs from the majority of models used for response generation, thereby reducing potential bias toward self-generated outputs. The evaluator assesses question-answer pairs using a structured prompt that includes the question, ground truth, and model-generated answer. It assigns a label of 1 for correct answers and 0 for incorrect ones; accuracy is adopted as the primary metric. Further details regarding the prompts and evaluation methodology are provided in Appendix F.

#### 5.2 Human Evaluation

To validate the reliability of the large language model (LLM) as an evaluative judge, human annotation was conducted on a set of 150 questions in both English and Russian. The responses generated by the Llama model were annotated using the Overlap-3 metric, with domain experts adhering to the same evaluation criteria as the automated judge. Inter-annotator agreement was quantified using Krippendorff’s α (Krippendorff, 2011), yielding a mean α = 0.755, which indicates a high degree of assessment reliability. Further evaluation of alignment between the automated judge and human annotators is conducted by computing the Pearson correlation coefficient r between the judge’s scores and the majority vote derived from human annotations. A strong mean correlation of r = 0.716 was observed, indicating substantial agreement. Additional details regarding the annotation procedure are provided in Appendix F.

6 Experiments and Results

#### 6.1 Sanity Check for evaluation procedure

Before proceeding with the remaining experiments, we first perform a simple sanity check to demonstrate that without any relevant context model performance drops significantly on two benchmarks: SQuAD2.0 (English) and MTS-SQuAD (Russian) with Llama3.1 8B model. In these experiment we wanted to answer a question if the model can generate accurate responses based solely on its internal knowledge. Additionally, we measure predictive

2https://mistral.ai/news/mistral-large-2407

entropy and show that it increases when no relevant context is provided. While these results are neither novel nor surprising, they serve to validate our overall evaluation procedure. The results are presented in Figure 2.

Accuracy Entropy

contextonlynocontexts

|0.951 0.906<br><br>0.248 0.220|
|---|

|0.194 0.169<br><br>0.820 0.861|
|---|

[Figure 3]

0.8

0.6

0.4

0.2

English Russian English Russian

Figure 2: Results in the presence and absence of contextual information for Llama3.1 8B model on SQuAD2.0 (en) and MTS-SQuAD (ru). We use a single relevant context to evaluate whether the provision of such information enhances model performance or if the model can generate accurate responses based solely on its internal knowledge.

Dataset Acc SQuAD2.0 (en) 89.6

MTS-SQuAD (ru) 86.5 MLQA (de) 62.8 MLQA (hi) 47.1 MLQA (vi) 47.5

Model Acc

DeepSeek 7B 71.1 Gemma 7B 48.6 Mistral 7B 74.3

Llama3.1 8B 69.0 Qwen2.5 7B 75.4

Table 2: Performance on specific language averaged by models, positions and strategies

Table 3: Performance of specific model averaged by datasets, positions, and strategies

Strategy Position Aligned All

No Scores

TOP MID BOT

Zero

0.695 0.610 0.721 0.684 0.666 0.676

Table 4: Accuracy score for different Strategies of position-bias elimination(Aligned, All Zero, No Scores) and Positions of relevant context(TOP, MIDDLE, BOTTOM). The results in each column are averaged across all other setups. The worst performance is highlighted in blue.

#### 6.2 Overall Results Across Languages and Models

The second step of our evaluation assesses overall model performance and dataset difficulty. Tables 2 and 3 show performance averaged across all positional configurations and prompting strategies (full results are provided in Appendix H). Language performance is highest for English (SQuAD2.0: 0.896) and Russian (MT-SQuAD: 0.865), but drops for German (0.628) and especially Hindi and Vietnamese (≈0.47), indicating greater difficulty in lower-resource or morphologically complex languages. For model performance, Qwen2.5-7B leads (0.754), followed by Mistral-7B (0.743), while DeepSeek-7B (0.711) and Llama3.18B (0.690) perform moderately. Gemma-7B lags substantially (0.486), suggesting weaker multilingual capabilities or architectural limitations.

#### 6.3 Position bias is mostly driven by models

While the “Lost in the Middle” phenomenon holds true, all models perform worst in the middle, as shown in Table 4, preferences for the beginning versus the end vary across datasets and models. We observe both model-specific bias (Table 7) and language-specific bias (Table 6). However, the aggregated results in Table 5 suggest that positional bias is primarily model-driven, as discrepancies across positions are more pronounced

between models than across languages. Specifically, DeepSeek-7B, Mistral-7B, and Qwen2.5-7B exhibit a preference for later positions, whereas Llama3.1-8B and Gemma-7B favor earlier ones. The strongest positional effect is observed for DeepSeek-7B and Llama3.1-8B, with a performance difference of approximately 0.1 points between positions.

#### 6.4 Positional guidance

Sensitivity to prompt guidance. For all considered languages incorrect relevance scoring leads to significant performance decrease. It varies from 4.3% drop for English (0.918 Aligned vs 0.874 All Zero) to 15.6% for Hindi (0.519 Aligned vs 0.363 All Zero). All models except Gemma exhibit pronounced sensitivity to positional cues when contextual scoring is perturbed. The introduction of misleading scores results in a marked decline in accuracy ranging from 1.4% drop for DeepSeek performance (from 0.683 Aligned to 0.669 All-Zero), to 25.4% drop for Llama performance (from 0.735 Aligned to 0.481 All-Zero). The absence of fluctuations in accuracy for Gemma is probably observed, because of poor quality of this model for all setups and general inability to handle positional guidance. Overall, the majority of setups are sensitive to positional guidance, so such techniques could be utilized to handle position bias.

|Position|Dataset Acc<br><br>|Model Acc|
|---|---|---|
|TOP MID BOT|en<br><br>0.904 0.894 0.891<br><br>|DeepSeek 7B<br><br>0.683 0.697 0.753<br><br>|
|TOP MID BOT|ru<br><br>0.861 0.861 0.873<br><br>|Gemma 7B<br><br>0.497 0.492 0.470<br><br>|
|TOP MID BOT<br><br>|de<br><br>0.646 0.615 0.624|Mistral 7B<br><br>0.748 0.729 0.751<br><br>|
|TOP MID BOT|hi<br><br>0.493 0.452 0.469<br><br>|Llama3.1 8B<br><br>0.743 0.671 0.655|
|TOP MID BOT|vi<br><br>0.480 0.463 0.483<br><br>|Qwen2.5-7B<br><br>0.752 0.744 0.767<br><br>|

Table 5: Model-wise and Language-wise position bias, highest accuracies reflect bias and are highlighted in bold-green. Results represent accuracy averaged across all strategies Aligned, All Zero and No Scores presented in Table 6 and Table 7. Position indicates correct context position among all contexts.

Score Omission Enhances Robustness. Notably, the No Scores consistently outperforms other strategies. For languages there is a little improvement for Russian in BOTTOM position( 0.889 Aligned vs 0.888 No Scores). Similarly, only for Qwen2.5 7B( 0.780 Aligned vs 0.777 No Scores) and DeepSeek 7B (0.761 Aligned vs 0.756 No Scores) in BOTTOM position the minor accuracy improvements are observed. For all other setups the performance is deteriorate, ranging from 0.1% to 10.2% accuracy drop for languages and from 0.2% to 7.1% for models (Aligned vs No Scores). This performance degradation is particularly pronounced in low-resource languages such as Vietnamese. These findings challenge previous works (Zhang et al., 2024a) and force to thorough validation of guiding strategies.

Language-Specific Sensitivity. High-resource languages, such as English and Russian, demonstrated minimal performance variation across scenarios (∆ < 2.5% in English), whereas languages on MLQA benchmark exhibited more pronounced differences. A phenomenon potentially attributable to orthographic or syntactic properties that may mediate position bias.

6.5 Highest Entropy is not associated with the lowest performance.

While we observe that models perform most poorly in the middle, the highest predictive entropy is not always associated with the middle position (see Table 8). We speculate that this phenomenon may

Dataset Position Aligned All Zero No Scores

TOP 0.918 0.874 0.921 MID 0.919 0.844 0.920 BOT 0.916 0.835 0.922

SQuAD2.0 (en)

TOP 0.877 0.823 0.883 MID 0.873 0.825 0.883 BOT 0.889 0.841 0.888

MTS-SQuAD (ru)

TOP 0.674 0.579 0.684 MID 0.658 0.525 0.661 BOT 0.664 0.542 0.666

MLQA (de)

TOP 0.518 0.415 0.544 MID 0.483 0.353 0.520 BOT 0.519 0.363 0.525

MLQA (hi)

TOP 0.466 0.410 0.565 MID 0.452 0.379 0.558 BOT 0.486 0.397 0.566

MLQA (vi)

- Table 6: Language specific bias with resulting accuracy averaged by models.

Model Position Aligned All Zero No Scores

DeepSeek 7B

TOP 0.683 0.669 0.696 MID 0.702 0.684 0.704 BOT 0.761 0.743 0.756

Gemma 7B

TOP 0.507 0.507 0.573 MID 0.515 0.515 0.571 BOT 0.485 0.485 0.556

Mistral 7B

TOP 0.755 0.709 0.779 MID 0.743 0.684 0.761 BOT 0.762 0.716 0.776

Llama3.1 8B

TOP 0.772 0.648 0.808 MID 0.720 0.517 0.777 BOT 0.735 0.481 0.748

Qwen2.5 7B

TOP 0.768 0.715 0.772 MID 0.750 0.715 0.767 BOT 0.780 0.745 0.777

- Table 7: Model specific bias with resulting accuracy averaged by datasets.

arise due to token homogenization and provide a formal analysis in Appendix D.

#### 6.6 Word Order Analysis

We additionally perform investigation if there is the relationship between the position of relevant context, model behavior, and the dominant word order of a language. Since we find no evidence to suggest that position bias influences models to favor specific word orders we discuss our methodology and detailed results in Appendix E.

|Model Position<br><br>|Aligned All Zero<br><br>No Scores<br><br>Mean|
|---|---|
|DeepSeek 7B<br><br>TOP MID BOT<br><br>|0.257 0.258 0.233 0.250 0.250 0.254 0.227 0.243 0.233 0.238 0.213 0.228<br><br>|
|Gemma 7B<br><br>TOP MID BOT<br><br>|0.189 0.189 0.194 0.191 0.187 0.187 0.196 0.190 0.189 0.189 0.200 0.193<br><br>|
|Mistral 7B<br><br>TOP MID BOT<br><br>|0.194 0.194 0.217 0.202 0.202 0.205 0.221 0.209 0.199 0.201 0.215 0.205<br><br>|
|Llama3.1 8B<br><br>TOP MID BOT|0.248 0.258 0.217 0.241 0.251 0.241 0.232 0.241 0.254 0.238 0.241 0.244<br><br>|
|Qwen2.5 7B<br><br>TOP MID BOT|0.101 0.097 0.105 0.101 0.106 0.104 0.112 0.107 0.106 0.101 0.108 0.105<br><br>|

Table 8: Predictive Entropy - PE across positions, models and strategies. Highest mean values across strategies are highlighted in purple.

### 7 Conclusion

This study reveals that position bias in multilingual LLMs is primarily model-driven, contradicting the assumed universal early-token preference—with architectures like Qwen2.5-7B and DeepSeek-7B favoring late positions, while Llama-3.1-8B prefers early ones. Language-specific effects exist but are secondary. Surprisingly, explicitly signaling context relevance via prompt-based relevance scores consistently harms performance, especially in lowresource languages. Moreover, the worst accuracy (when relevant context is in the middle) does not correspond to higher output entropy, indicating models are confidently wrong under positional disadvantage. These findings challenge common RAG and prompting practices and underscore the need for model- and language-aware context handling.

### 8 Limitations

Since LLM-as-a-Judge was utilized to assess the correctness of open-ended question-answering task, our methodology depends on its performance critically.

Our evaluation used 2,000 question–answer pairs per language. Across nine experimental conditions and five models, this amounts to 9 × 5 × 2000 = 70000 model evaluations per language, a computationally intensive effort. Given this scale, we took care to ensure statistical rigor. Specifically, we performed pairwise t-tests with Holm–Bonferroni correction for the three positional comparisons: (1) top vs. middle, (2) top vs. bottom, and (3) middle vs. bottom. For every dataset and model, at

least one of these (two on average) comparisons yielded statistically significant differences in accuracy (p < 0.05). However, when aggregating results across models, we found no significant differences between languages. The same pattern held for predictive entropy values.

Acknowledgment on LLM assisted writing: This paper used open access Qwen3-Max, in some parts of the paper, for proofreading and text rephrasing in accordance with formal style.

### References

Marco Alessio, Guglielmo Faggioli, Nicola Ferro, Franco Maria Nardini, Raffaele Perego, and 1 others. 2024. Improving rag systems via sentence clustering and reordering. In CEUR WORKSHOP PROCEEDINGS, volume 3784, pages 34–43.

Catherine Arnett and Benjamin Bergen. 2025. Why do language models perform worse for morphologically complex languages? In Proceedings of the 31st International Conference on Computational Linguistics, pages 6607–6623, Abu Dhabi, UAE. Association for Computational Linguistics.

George Arthur Baker, Ankush Raut, Sagi Shaier, Lawrence E Hunter, and Katharina von der Wense. 2024. Lost in the middle, and in-between: Enhancing language models’ ability to reason over long contexts in multi-hop qa. ArXiv, abs/2412.10079.

Federico Barbero, ’Alvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Michael M. Bronstein, Petar Velivckovi ’c, and Razvan Pascanu. 2025. Why do llms attend to the first token? ArXiv, abs/2504.02732.

Anna Bavaresco, Raffaella Bernardi, Leonardo Bertolazzi, Desmond Elliott, Raquel Fern’andez, Albert Gatt, E. Ghaleb, Mario Giulianelli, Michael Hanna, Alexander Koller, Andr’e F. T. Martins, Philipp Mondorf, Vera Neplenbroek, Sandro Pezzelle, Barbara Plank, David Schlangen, Alessandro Suglia, Aditya K Surikuchi, Ece Takmaz, and Alberto Testoni. 2024. Llms instead of human judges? a large scale empirical study across 20 nlp evaluation tasks. ArXiv, abs/2406.18403.

Lera Boroditsky, Lauren A Schmidt, and Webb Phillips. 2003. Sex, syntax, and semantics. Language in mind: Advances in the study of language and thought, 22(61-79):3.

Hee-Soo Choi, Bruno Guillaume, Karën Fort, and Guy Perrier. 2021. Investigating dominant word order on Universal Dependencies with graph rewriting. In Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021), pages 281–290, Held Online. INCOMA Ltd.

Anh Dang, Limor Raviv, and Lukas Galke. 2024. Morphology matters: Probing the cross-linguistic morphological generalization abilities of large language models through a wug test. In 13th edition of the Workshop on Cognitive Modeling and Computational Linguistics (CMCL 2024), pages 177–188. Association for Computational Linguistics (ACL).

DeepSeek-AI. 2024. Deepseek llm: Scaling opensource language models with longtermism. arXiv preprint arXiv:2401.02954.

Michael Denkowski and Alon Lavie. 2014. Meteor universal: Language specific translation evaluation for any target language. In Proceedings of the ninth workshop on statistical machine translation, pages 376–380.

Matthew S. Dryer. 2013. Order of subject, object and verb (v2020.4). In Matthew S. Dryer and Martin Haspelmath, editors, The World Atlas of Language Structures Online. Zenodo.

Jinhao Duan, Hao Cheng, Shiqi Wang, Alex Zavalny, Chenan Wang, Renjing Xu, Bhavya Kailkhura, and Kaidi Xu. 2024. Shifting attention to relevance: Towards the predictive uncertainty quantification of free-form large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5050–5063, Bangkok, Thailand. Association for Computational Linguistics.

Pavel Efimov, Andrey Chertok, Leonid Boytsov, and Pavel Braslavski. 2020. Sberquad–russian reading comprehension dataset: Description and analysis. In Experimental IR Meets Multilinguality, Multimodality, and Interaction: 11th International Conference of the CLEF Association, CLEF 2020, Thessaloniki, Greece, September 22–25, 2020, Proceedings 11, pages 3–15. Springer.

Poulami Ghosh, Shikhar Vashishth, Raj Dabre, and Pushpak Bhattacharyya. 2024a. A morphologybased investigation of positional encodings. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 21035– 21045, Miami, Florida, USA. Association for Computational Linguistics.

Poulami Ghosh, Shikhar Vashishth, Raj Dabre, and Pushpak Bhattacharyya. 2024b. A morphologybased investigation of positional encodings. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 21035– 21045, Miami, Florida, USA. Association for Computational Linguistics.

Xiaobo Guo and Soroush Vosoughi. 2024. Serial position effects of large language models. ArXiv, abs/2406.15981.

Yanzhu Guo, Simone Conia, Zelin Zhou, Min Li, Saloni Potdar, and Henry Xiao. 2024. Do large language models have an english accent? evaluating and improving the naturalness of multilingual llms. ArXiv, abs/2410.15956.

Hsiu-Yuan Huang, Yutong Yang, Zhaoxi Zhang, Sanwoo Lee, and Yunfang Wu. 2024. A survey of uncertainty estimation in llms: Theory meets practice. ArXiv, abs/2410.15326.

Mete Ismayilzada, Defne Circi, Jonne Sälevä, Hale Sirin, Abdullatif Köksal, Bhuwan Dhingra, Antoine Bosselut, Duygu Ataman, and Lonneke Van Der Plas. 2025. Evaluating morphological compositional generalization in large language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1270–1305, Albuquerque, New Mexico. Association for Computational Linguistics.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825.

Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan Ö. Arik. 2024. Long-context llms meet rag: Overcoming challenges for long inputs in rag. ArXiv, abs/2410.05983.

Seungone Kim, Juyoung Suk, Shayne Longpre, Bill Yuchen Lin, Jamin Shin, Sean Welleck, Graham Neubig, Moontae Lee, Kyungjae Lee, and Minjoon Seo. 2024. Prometheus 2: An open source language model specialized in evaluating other language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 4334–4353, Miami, Florida, USA. Association for Computational Linguistics.

Klaus Krippendorff. 2011. Computing krippendorff’s alpha-reliability.

Patrick Lewis, Barlas Oguz, Ruty Rinott, Sebastian Riedel, and Holger Schwenk. 2020. Mlqa: Evaluating cross-lingual extractive question answering. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics.

Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.

Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2022. Fantastically ordered prompts and where to find them: Overcoming fewshot prompt order sensitivity. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8086–8098, Dublin, Ireland. Association for Computational Linguistics.

Daphna Oyserman and Spike WS Lee. 2008. Does culture influence what and how we think? effects of priming individualism and collectivism. Psychological bulletin, 134(2):311.

Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318.

Alexander Peysakhovich and Adam Lerer. 2023. Attention sorting combats recency bias in long context language models. ArXiv, abs/2310.01427.

Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton, and Christopher D Manning. 2020. Stanza: A python natural language processing toolkit for many human languages. arXiv preprint arXiv:2003.07082.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, and 25 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018. Know what you don’t know: Unanswerable questions for squad. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 784–789.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392.

Lisa Schut, Yarin Gal, and Sebastian Farquhar. 2025. Do multilingual llms think in english? ArXiv, abs/2502.15603.

Petr Sychev, Andrey Goncharov, Daniil Vyazhev, Edvard Khalafyan, and Alexey Zaytsev. 2025. When an llm is apprehensive about its answers - and when its uncertainty is justified. ArXiv, abs/2503.01688.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, Léonard Hussenot, Pier Giuseppe Sessa, Aakanksha Chowdhery, Adam Roberts, Aditya Barua, Alex Botev, Alex Castro-Ros, Ambrose Slone, and 89 others. 2024. Gemma: Open models based on gemini research and technology. Preprint, arXiv:2403.08295.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. Preprint, arXiv:2302.13971.

Yuhao Wang, Ruiyang Ren, Junyi Li, Xin Zhao, Jing Liu, and Ji-Rong Wen. 2024a. REAR: A relevance-aware retrieval-augmented framework for open-domain question answering. In Proceedings

of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5613–5626, Miami, Florida, USA. Association for Computational Linguistics.

Zhiyuan Wang, Jinhao Duan, Chenxi Yuan, Qingyu Chen, Tianlong Chen, Yue Zhang, Ren Wang, Xiaoshuang Shi, and Kaidi Xu. 2025. Word-sequence entropy: Towards uncertainty estimation in free-form medical question answering applications and beyond. Engineering Applications of Artificial Intelligence, 139:109553.

Ziqi Wang, Hanlin Zhang, Xiner Li, Kuan-Hao Huang, Chi Han, Shuiwang Ji, Sham M. Kakade, Hao Peng, and Heng Ji. 2024b. Eliminating position bias of language models: A mechanistic approach. ArXiv, abs/2407.01100.

Xinyi Wu, Yifei Wang, Stefanie Jegelka, and Ali Jadbabaie. 2025. On the emergence of position bias in transformers. ArXiv, abs/2502.01951.

Tan Yu, Anbang Xu, and Rama Akkiraju. 2024a. In defense of rag in the era of long-context language models. ArXiv, abs/2409.01666.

Yijiong Yu, Huiqiang Jiang, Xufang Luo, Qianhui Wu, Chin-Yew Lin, Dongsheng Li, Yuqing Yang, Yongfeng Huang, and Lili Qiu. 2024b. Mitigate position bias in large language models via scaling a single dimension. ArXiv, abs/2406.02536.

Meiru Zhang, Zaiqiao Meng, and Nigel Collier. 2024a. Can we instruct llms to compensate for position bias? In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 12545–12556.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. ArXiv, abs/1904.09675.

Zhenyu (Allen) Zhang, Runjin Chen, Shiwei Liu, Zhewei Yao, Olatunji Ruwase, Beidi Chen, Xiaoxia Wu, and Zhangyang Wang. 2024b. Found in the middle: How language models use long contexts better via plug-and-play positional encoding. ArXiv, abs/2403.04797.

Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In International conference on machine learning, pages 12697–12706. PMLR.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, and 1 others. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623.

Chengzhi Zhong, Fei Cheng, Qianying Liu, Junfeng Jiang, Zhen Wan, Chenhui Chu, Yugo Murawaki, and Sadao Kurohashi. 2024. Beyond english-centric llms: What language do multilingual language models think in? ArXiv, abs/2408.10811.

### Appendix

### A RAG systems with context reordering

REAR (Wang et al., 2024a) – integrates document relevance scores into LLMs via embeddings, guiding generation to use internal knowledge (low relevance) or external evidence (high relevance).

Long-Context LLMs Meet RAG (Jin et al., 2024) – addresses the "lost-in-the-middle" problem by reordering retrieved documents, placing highestscoring ones at sequence boundaries to optimize LLM attention.

OP-RAG (Yu et al., 2024a) – order preserving RAG preserves original document order (vs. sorting chunks), demonstrating improved answer quality through position-aware context organization. However, authors do not mention multidocument scenario.

Clustering & Reordering RAG (Alessio et al., 2024) - cluster sentences by query similarity and sort clusters in descending similarity order for improved retrieval quality.

### B Datasets

SQuAD2.03 (Rajpurkar et al., 2018) is an English reading-comprehension benchmark built on Wikipedia passages. It combines 100 000 spananswerable questions from SQuAD 1.1 (Rajpurkar et al., 2016) with more than 50 000 adversarial questions whose answers are deliberately absent.

MTS-SQuAD4 is an extension of the SberQuAD dataset (Efimov et al., 2020) which is a Russian counterpart of SQuAD 2.0. It includes more than 60 000 quesion-answer pairs with improved readability and consistency.

MLQA5 (Lewis et al., 2020) is a multilingual benchmark built from aligned Wikipedia passages including 12 000 question-answer pairs in English and about 5 000 in each of the other six languages: Arabic, German, Spanish, Hindi, Vietnamese, and Simplified Chinese. Among these languages, we utilized German, Hindi and Vietnamese.

### C Technical Details

Models inference. To achieve reproducibility of the obtained results, the LLM-inference was

- 3https://huggingface.co/datasets/rajpurkar/squad_v2
- 4https://huggingface.co/datasets/MTS-AI-

SearchSkill/MTSBerquad

- 5https://github.com/facebookresearch/MLQA

performed using a deterministic generation strategy. The following hyperparameter were used/set: "max_new_tokens" – 1024, "do_sample" – False, "num_beams" – 1.

Computational Resources. The experiments were run in a Docker container on a dedicated server with the following hardware: CPU: AMD Ryzen 9 7900X 12-Core Processor, GPU: NVIDIA GeForce RTX 3090 24GB, RAM: Kingston FURY Beast Black 32GB, SSD: M.2 NVMe Samsung 990 PRO 1T.

Required GPU-time for experiments. In total it is required approximately 50 GPU-hours to reproduce the experiments.

### D Effect of position bias on Entropy

This section formalizes the propagation of position bias toward initial tokens across layers in transformer-based large language models (LLMs) and examines its effect on attention entropy. Under the assumption of a standard multi-head selfattention architecture, we derive conditions under which such bias leads to homogenization of token representations, consequently increasing entropy in the final layer.

Notation. Let X(0) = [x1,x2,...,xn] ∈ Rd×n denote the input token embeddings at layer 0, where x1 is the first token and d is the embedding dimension. At each layer l ≥ 1, the self-attention operation computes:

A(t) = softmax X(l)WQ(l)(X(l)WK(t))⊤ (2)

X(t+1) = WO(l)A(l)X(l)WV(l) (3)

where WQ(l),WK(l),WV(l),WO(l) ∈ Rd×d are learnable projection matrices, A(l) ∈ Rn×n contains the attention weights and dQK = 1 for simplicity.

Assumptions. To isolate the effect of position bias, we make simplifying assumptions: (1) dominant first token attention, (2) position bias does not change over layers and (3) attention A can be represented as a linear combination of contextual attention Acont and positional attention Apos. For all layers l ≥ 1, the positional attention weights Aposi,j (l) are sharply concentrated on the first token:

Aposi,j (l) ≈

1 if j = 1, 0 otherwise.

Or in vector form Apos(l) ≈ 1⊤i e1. Linear combination of attention: A = λ1 Acont+λ2 Apos, λ1+

λ2 = 1;λ1,λ2 ∈ [0;1] where λ are normalizing weights for each attention type.

Token Homogenization. Under these assumptions, the hidden state of token i at layer l becomes:

n

λ1Aposi,j(l) + λ2Aconti,j (l) WV(l)x(jl−1)

x(il) ≈ WO(l) ·

j=1

= λ2WO(l)WV(l) · 1⊤i e1x(il−1) + λ1xconi (l)

= λ2P(l)x(il−1) + λ1xconi (l), ∀i.

(4)

Where P(l) = WO(l)WV(l). If λ2 > λ1 recursively applying this across L layers yields:

x(iL) ≈ x(0)1 ∀i, (5) implying all tokens collapse to a copy of the

initial first token embedding x(0)1 (up to projection transformations). Although token collapse would not happen in a real scenario, for example, due to residual connections, tokens may become more similar to tokens under position bias.

Entropy Dynamics. Let HA(l) denote the general attention entropy at layer l. As tokens homogenize (x(il) ≈ x(0)1 ), queries and keys become indistinguishable, causing contextual attention weights to approach uniformity: Aconi,j (l) ≈ n1 ∀i,j. This results in the maximization of entropy: HA(l) → log n.

Aligned positional and contextual attention. This section establishes a theoretical connection between entropy, attention mechanisms, and position bias under several modeling assumptions. We demonstrate that alignment between contextual attention Acon and positional attention Apos increases the likelihood of homogenization. This leads to a counterintuitive outcome: when the relevant context coincides with the model’s inherent positional bias, the model allocates greater attention to the corresponding tokens, ultimately elevating entropy. Further empirical support for these findings is provided in Section D, where we observe that minimal entropy does not consistently coincide with alignment between contextual relevance and positional bias.

Predictive and Attention Entropy. Attention mechanisms are designed to prioritize relevant tokens within a sequence. When attention is uniformly distributed (indicating high entropy), the model is unable to effectively leverage contextual cues. This lack of focused attention results in diminished predictive signals, thereby increasing un-

certainty in the model’s output predictions (manifested as higher predictive entropy).6

Multilingual Caveat. In multilingual large language models (LLMs), position bias may vary across different layers; for instance, early layers often prioritize English tokens, whereas later layers tend to align more closely with the language of the input prompt (Zhong et al., 2024; Schut et al., 2025). This shift may introduce language-specific positional preferences, thereby challenging the assumption of static position bias. Nevertheless, if homogenization occurs, the overall entropy dynamics remain consistent.

### E Word Order Analysis

In this section, we perform additional investigation if there is the relationship between the position of relevant context, model behavior, and the dominant word order of a language. Specifically, we examine whether position bias amplifies or diminishes the influence of a model’s dominant language word order. Our analysis focuses on the No Scores configuration with five contexts, using Hindi and German as representative languages due to their non-SVO dominant word orders, as documented in The World Atlas of Language Structures. Sentences are parsed using Stanza (Qi et al., 2020), following a methodology similar to that of (Choi et al., 2021). For each verb, we identify its dependents; if a verb has both a subject (indicated by a "subj" relation, or such a relation for the nearest preceding verb connected via a "conj" dependency) and an object (indicated by an "obj" relation), we record the word order of these triplets using the abbreviations "S", "V", and "O". The distribution of "SVO" and "SOV" patterns is then analyzed relative to all extracted triplets.

We find no evidence to suggest that position bias influences models to favor specific word orders. For instance, in Hindi a predominantly subject–object–verb (SOV) language—one might expect subject–verb–object (SVO) rates to be lowest when relevant context appears last for Qwen and first for Llama, however, this pattern is not observed (Table 9). In German, where dominant word order varies by clause type (SVO in main clauses

6For example, in the sentence "The capital of France is ***," the token "France" is critical for accurate prediction. If attention is uniform, the model may assign equal weight to less relevant tokens (e.g., "The" or "of"), failing to emphasize "France." This ambiguity elevates uncertainty in predicting the subsequent token, such as "Paris."

and SOV in subordinate clauses; (Dryer, 2013)), we examine the prevalence of SVO (quantified as the SVO–SOV difference) in Table 10. Here, also, no association with position bias is detected, as the observed prevalence appears to arise naturally from the proportion of complex sentences.

Model Word order TOP MID BOT Llama3.1 8B

SVO 5.39 5.12 4.75 SOV 88.52 88.49 88.43

SVO 2.34 4.15 2.96 SOV 93.98 91 91.78

Qwen2.5 7B

- Table 9: Percentage of word orders for Hindi, cells where context placements align with position bias are highlighted in green. Bolded results indicate stronger alignment with expected word order.

Model Position Difference SVO - SOV

% of Complex Sentences

Llama3.1 8B

TOP 26.88 13.56 MIDDLE 32.13 12.13 BOTTOM 40.04 10.88

Qwen2.5 7B

TOP 34.17 8.59 MIDDLE 29.56 9.54 BOTTOM 29.78 8.96

- Table 10: Prevalence of SVO over SOV word order and the proportion of complex sentences in German

For model-level analysis, the following distinctions are observed: (1) Llama generates a higher proportion of subject–verb–object (SVO) sentences than Qwen across Hindi, German, and Russian (Table 11)—note that Russian is included despite its typological preference for SVO order due to its rich morphological system, which permits considerable word-order flexibility (Ghosh et al., 2024b); (2) Llama produces syntactically more complex sentences than Qwen across all three contextual positions (Table 12); (3) both models exhibit a tendency to generate more complex sentences when relevant context is provided in the initial position (Table 13).

Model Hindi German Russian Llama3.1 8B 4.8 - 5.4% 60.3 - 67.5% 95.9 - 96.2% Qwen2.5 7B 2.3 - 4.2% 57.9 - 61.4% 93.6 - 94.5%

- Table 11: Percentage of SVO structures in generated responses across all three positions of relevant context

Model Hindi German Russian Llama3.1 8B 9.94 % 12.19 % 7.19% Qwen2.5 7B 6.88% 9.03 % 7%

- Table 12: Mean percentage of complex sentences among all sentences containing at least one triplet of a subject, verb, and object

|Language<br><br>|Llama3.1 8B| | |Qwen2.5 7B| | |
|---|---|---|---|---|---|---|
| |TOP|MID<br><br>|BOT|TOP|MID<br><br>|BOT|
|Hindi|✓| | |✓| | |
|German|✓| | | |✓| |
|Russian| |✓| |✓| | |

- Table 13: The context position in which the proportion of complex sentences is highest, given a specific model and language

### F Evaluation Details

- F.1 LLM-as-a-Judge Verification

Krippendorff’s alpha and Pearson correlation coefficients, calculated for each experimental setup can be seen in Tables 14 and 15 correspondingly. Comparison of human and Llama evaluation can be seen in Figure 3.

- F.2 Human and LLM instructions

We prompt Mistral-Large to judge whether the LLM responses are correctly answering questions. For each dataset we create an evaluation prompt on the language of this dataset and add 4 shots as examples of judgments. The resulting prompt consists of system prompt "You are an AI assistant who speaks English.", which we translate to other languages and user prompt.

For human annotation we consider only English and Russian languages, since our annotators speaks these languages. We use the same instructions as for LLM-as-a-Judge settings, omitting shots.

- F.3 Human Annotators Information

Annotation was conducted by the authors of the work, so no additional recruitment or payment are required on this stage. All assessors held bachelor’s degree and had prior experience in the evaluation of LLM responses.

### G Context Volume Analysis

From Figure 4 we can observe that our Aligned strategy does not have an effect on position bias with increasing of information load. With context quantity N = 15 for Llama3.1-8B we can see

#### Language Position Aligned All Zero No Scores Mean

TOP 0.783 0.663 0.595 MIDDLE 0.611 0.861 0.685 0.726 BOTTOM 0.704 0.916 0.718

English

TOP 0.814 0.825 0.674 MIDDLE 0.742 0.865 0.695 0.783 BOTTOM 0.801 0.855 0.776

Russian

Table 14: Krippendorff’s alpha coefficient, calculated for each experimental setup

#### Language Position Aligned All Zero No Scores Mean

TOP 0.612 0.632 0.604 MIDDLE 0.739 0.83 0.791 0.727 BOTTOM 0.738 0.908 0.685

English

TOP 0.488 0.704 0.518 MIDDLE 0.709 0.87 0.769 0.705 BOTTOM 0.669 0.888 0.732

Russian

Table 15: Pearson correlation coefficient, calculated for each experimental setup

- (a)
- (b)

Correct Score All Zero Scores No Scores

1.0

1.0

1.0

0.955

0.95

0.95

0.95

Accuracy

Accuracy

Accuracy

0.90

0.90

0.90

0.85

0.85

0.85

0.80

0.80

0.80

TOP MIDDLE BOTTOM TOP MIDDLE BOTTOM TOP MIDDLE BOTTOM

Position Position Position

Correct Score All Zero Scores No Scores

1.0

1.0

1.0

0.928

0.9

0.9

0.9

Accuracy

Accuracy

Accuracy

0.8

0.8

0.8

0.7

0.7

0.7

0.6

0.6

0.6

TOP MIDDLE BOTTOM TOP MIDDLE BOTTOM TOP MIDDLE BOTTOM

Position Position Position

- Figure 3: Human evaluation and LLM as a Judge for Correct Scores(Aligned), All Zero, and No Scores strategies at three postions (TOP, MIDDLE, BOTTOM) for two languages: (a) English and (b) Russian. Bars in green represent human evaluations, while the blue bars represent the Llama model.

### H Non Aggregated results for main experiments

significant accuracy decrease, compared to other quantities. From the other hand, for Qwen2.5-7B position bias does not correlate with passed number of contexts. This result can be explained by the fact that for Qwen2.5-7B training larger dataset with long contexts was used, compared to Llama3.1-8B. This feature increase for Qwen2.5-7B the size of its attention window and allowed to conditioning on a larger amount of input knowledge during response generation.

Our non aggregated results for LLM as an accuracy and entropy are presented in Tables 16,17 and 18, 19 correspondingly.

(a) (b)

1.0

1.0

0.95

0.95

Accuracy

Accuracy

0.90

0.90

0.85 0.80

0.85 0.80

Top

Middle Bottom

0.75

0.75

5 10 15 5 10 15

contexts number contexts number

- Figure 4: Accuracy dependence on the number of contexts, added to the user-prompt, and position of the relevant context in a list with Aligned placement strategy: (a) Llama3.1-8B; (b) Qwen2.5-7B

DeepSeek 7B Gemma 7B Mistral 7B Dataset Position

Aligned All Zero No Scores Aligned All Zero No Scores Aligned All Zero No Scores TOP 0.907 0.877 0.913 0.815 0.683 0.831 0.970 0.956 0.961

Squadv2 (en) MIDDLE 0.922 0.887 0.921 0.849 0.614 0.832 0.958 0.929 0.957 BOTTOM 0.953 0.929 0.950 0.782 0.534 0.815 0.963 0.939 0.958

TOP 0.804 0.791 0.841 0.775 0.655 0.797 0.939 0.926 0.936 MTS-SQuAD (ru) MIDDLE 0.834 0.829 0.843 0.777 0.701 0.819 0.920 0.911 0.924

BOTTOM 0.884 0.875 0.881 0.782 0.707 0.824 0.932 0.928 0.930

TOP 0.737 0.732 0.720 0.477 0.324 0.524 0.829 0.799 0.813 MLQA (de) MIDDLE 0.735 0.708 0.724 0.508 0.289 0.486 0.815 0.750 0.801

BOTTOM 0.799 0.772 0.777 0.432 0.244 0.441 0.823 0.787 0.819

TOP 0.431 0.372 0.229 0.520 0.439 0.601 MLQA (hi) MIDDLE 0.402 0.334 0.237 0.505 0.425 0.558

-

BOTTOM

0.393 0.349 0.228 0.546 0.471 0.585 TOP 0.284 0.277 0.309 0.039 0.021 0.484 0.520 0.428 0.584

MLQA (vi) MIDDLE 0.317 0.314 0.330 0.039 0.020 0.483 0.517 0.406 0.566 BOTTOM 0.407 0.397 0.415 0.035 0.014 0.471 0.545 0.457 0.588

Table 16: Accuracy for three models: DeepSeek 7B, Gemma 7B and Mistral 7B, DeepSeek consistently failed in Hindi

|Dataset Position<br><br>Llama3.1 8B Aligned All Zero No Scores<br><br>|Qwen2.5 7B Aligned All Zero No Scores|
|---|---|
|SQuAD2.0 (en)<br><br>TOP 0.955 0.929 0.951 MIDDLE 0.926 0.858 0.946 BOTTOM 0.932 0.826 0.930<br><br>|0.943 0.927 0.952 0.941 0.931 0.945 0.949 0.948 0.956<br><br>|
|MTS-SQuAD (ru)<br><br>TOP 0.928 0.837 0.906 MIDDLE 0.908 0.768 0.888 BOTTOM 0.911 0.761 0.866<br><br>|0.938 0.909 0.934 0.927 0.917 0.944 0.937 0.935 0.939<br><br>|
|MLQA (de)<br><br>TOP 0.680 0.485 0.719 MIDDLE 0.602 0.312 0.668 BOTTOM 0.613 0.307 0.646<br><br>|0.648 0.553 0.644 0.629 0.566 0.628 0.655 0.602 0.649<br><br>|
|MLQA (hi)<br><br>TOP 0.532 0.309 0.729 MIDDLE 0.448 0.112 0.676 BOTTOM 0.494 0.038 0.644<br><br>|0.591 0.540 0.619 0.579 0.543 0.609 0.642 0.592 0.643<br><br>|
|MLQA (vi)<br><br>TOP 0.764 0.679 0.737 MIDDLE 0.714 0.536 0.707 BOTTOM 0.726 0.473 0.656<br><br>|0.722 0.644 0.713 0.676 0.619 0.707 0.718 0.647 0.700<br><br>|

Table 17: Accuracy for two models: Llama3.1 8B and Qwen2.5 7B

|DeepSeek 7B Chat Dataset Position<br><br>Aligned All Zero No Scores<br><br>|Gemma 7B Aligned All Zero No Scores|Mistral 7B Aligned All Zero No Scores<br><br>|
|---|---|---|
|TOP 0.140 0.148 0.121 SQuAD2.0 (en) MIDDLE 0.132 0.137 0.116<br><br>BOTTOM 0.128 0.129 0.114<br><br>|0.131 0.160 0.116 0.141 0.166 0.110 0.146 0.175 0.113<br><br>|0.122 0.124 0.126 0.125 0.130 0.131 0.117 0.122 0.125<br><br>|
|TOP 0.225 0.222 0.240 MTS-SQuAD (ru) MIDDLE 0.216 0.222 0.232<br><br>BOTTOM 0.208 0.212 0.231<br><br>|0.183 0.229 0.155 0.164 0.204 0.149 0.167 0.199 0.145<br><br>|0.185 0.191 0.204 0.193 0.200 0.210 0.190 0.197 0.207<br><br>|
|TOP 0.237 0.241 0.240 MLQA (de) MIDDLE 0.238 0.249 0.238<br><br>BOTTOM 0.222 0.233 0.228<br><br>|0.234 0.228 0.219 0.218 0.209 0.225 0.215 0.197 0.235<br><br>|0.276 0.288 0.292 0.286 0.321 0.295 0.274 0.300 0.287<br><br>|
|TOP<br><br>MLQA (hi) MIDDLE BOTTOM<br><br>-|0.279 0.325 0.366 0.302 0.349 0.382 0.312 0.343 0.391<br><br>|0.213 0.202 0.238 0.221 0.204 0.236 0.227 0.214 0.237<br><br>|
|TOP 0.427 0.423 0.332 MLQA (vi) MIDDLE 0.412 0.409 0.322<br><br>BOTTOM 0.374 0.379 0.277<br><br>|0.120 0.112 0.115 0.110 0.106 0.113 0.104 0.104 0.114<br><br>|0.175 0.166 0.225 0.185 0.171 0.232 0.185 0.171 0.218<br><br>|

Table 18: Predictive entropy values for three models: DeepSeek 7B Chat, Gemma 7B and Mistral 7B, Deep Seek consistently failed in Hindi

|Dataset Position<br><br>Llama3.1 8B Aligned All Zero No Scores<br><br>|Qwen2.5 7B Aligned All Zero No Scores|
|---|---|
|SQuAD2.0 (en)<br><br>TOP 0.232 0.248 0.194 MIDDLE 0.237 0.239 0.201 BOTTOM 0.241 0.240 0.206<br><br>|0.092 0.093 0.100 0.094 0.102 0.100<br><br>0.093 0.099 0.097<br><br><br>|
|MTS-SQuAD (ru)<br><br>TOP 0.203 0.224 0.169 MIDDLE 0.217 0.219 0.182 BOTTOM 0.214 0.222 0.191<br><br>|0.105 0.111 0.129 0.112 0.118 0.126 0.112 0.109 0.126<br><br>|
|MLQA (de)<br><br>TOP 0.231 0.204 0.220 MIDDLE 0.220 0.158 0.225 BOTTOM 0.225 0.152 0.220<br><br>|0.117 0.106 0.115 0.126 0.118 0.130 0.128 0.118 0.125<br><br>|
|MLQA (hi)<br><br>TOP 0.303 0.320 0.237 MIDDLE 0.294 0.290 0.257 BOTTOM 0.289 0.277 0.275<br><br>|0.078 0.075 0.076<br><br>0.082 0.077 0.082<br><br>0.083 0.082 0.080<br><br><br>|
|MLQA (vi)<br><br>TOP 0.272 0.296 0.267 MIDDLE 0.288 0.299 0.295 BOTTOM 0.298 0.298 0.313<br><br>|0.114 0.101 0.107 0.117 0.103 0.120 0.111 0.100 0.112<br><br>|

Table 19: Predictive entropy values for two models Llama3.1 8B and Qwen2.5 7B.

### I Prompts

Our user–prompts for LLM-inference in terms of context placement strategies can be seen in Tables 20, 21. For Aligned and All Zero strategies items in a contexts–list has a following format: "- [{score}] {document}". For No Scores strategy items has the following format: "- {document}". As a system– prompt the same instruction was used for all languages (translated correspondingly): "You are an AI assistant who helps solve user issues.".

##### Language User Prompt

English

Russian

German

Hindi

Vietnamese

Table 20: User–prompts in five languages for LLM–inference in the No Scores context placement strategy

##### Language User Prompt

English

Russian

German

Hindi

Vietnamese

Table 21: User–prompts in five languages for LLM–inference in the Aligned and All Zero context placement strategies

