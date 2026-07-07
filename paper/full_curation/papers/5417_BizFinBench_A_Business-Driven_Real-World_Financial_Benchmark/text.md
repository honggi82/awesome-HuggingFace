# arXiv:2505.19457v1[cs.AI]26May2025

[Figure 1]

## BizFinBench: A Business-Driven Real-World Financial Benchmark for Evaluating LLMs

Guilong Lu∗1, Xuntao Guo∗1,2, Rongjunchen Zhang♠1, Wenqiao Zhu1, and Ji Liu1 1HiThink Research 2Harbin Institute of Technology

### Abstract

Large language models excel in general tasks, yet assessing their reliability in logic-heavy, precision-critical domains like finance, law, and healthcare remains challenging. To address this, we introduce BizFinBench, the first benchmark specifically designed to evaluate LLMs in real-world financial applications. BizFinBench consists of 6,781 well-annotated queries in Chinese, spanning five dimensions: numerical calculation, reasoning, information extraction, prediction recognition, and knowledge-based question answering, grouped into nine fine-grained categories. The benchmark includes both objective and subjective metrics. We also introduce IteraJudge, a novel LLM evaluation method that reduces bias when LLMs serve as evaluators in objective metrics. We benchmark 25 models, including both proprietary and open-source systems. Extensive experiments show that no model dominates across all tasks. Our evaluation reveals distinct capability patterns: (1) In Numerical Calculation, Claude-3.5-Sonnet (63.18) and DeepSeek-R1 (64.04) lead, while smaller models like Qwen2.5-VL-3B (15.92) lag significantly; (2) In Reasoning, proprietary models dominate (ChatGPT-o3: 83.58, Gemini-2.0-Flash: 81.15), with open-source models trailing by up to 19.49 points; (3) In Information Extraction, the performance spread is the largest, with DeepSeek-R1 scoring 71.46, while Qwen3-1.7B scores 11.23; (4) In Prediction Recognition, performance variance is minimal, with top models scoring between 39.16 and 50.00. We find that while current LLMs handle routine finance queries competently, they struggle with complex scenarios requiring cross-concept reasoning. BizFinBench offers a rigorous, business-aligned benchmark for future research. The code and dataset are available at https://github.com/HiThink-Research/BizFinBench.

### 1 Introduction

Recent years have witnessed rapid advancements of Large Language Models (LLMs), which demonstrates remarkable capabilities across diverse domains, such as finance, law, healthcare and so on Chen et al. [2024], Liu et al. [2025], Zhang et al. [2023a], Lu et al. [2024], Xie et al. [2025]. In financial applications, LLMs are increasingly applied to complex tasks, including automated financial analysis, fraud detection, risk assessment, and investment strategy formulation Zhao et al. [2024], Gan et al. [2024]. However, evaluating the robustness and reliability of LLMs in finance domains remains a significant challenge.

∗ denotes equal contribution. This paper was completed during Guilong Lu and Xuntao Guo’s internships at Hithink Research.

♠ denotes corresponding author, zhangrongjunchen@myhexin.com.

Preprint. Under review.

[Figure 2]

- Figure 1: Comparison of numerical calculation questions in Fin-Eva Team [2023] and BizFinBench. The Fin-Eva example presents a straightforward financial math problem, while the BizFinBench example requires multi-step reasoning: first analyzing the problem, then extracting and utilizing relevant data from a provided markdown-formatted table for accurate computation. An Chinese version is included in the Appendix for clarity and ease of reference.

Different from traditional Science, Technology, Engineering, and Mathematics (STEM) questions, where inputs are typically short, well-structured, and yield deterministic answers, financial tasks are more complex. They typically involve long context, structured inputs (e.g., tabular stock data, market news), require temporal reasoning, and demand fine-grained judgment under ambiguity. As illustrated in Figure 1, STEM-style questions usually have clear computational logic and a single correct answer, while financial tasks call for multi-step reasoning over real-world data, generally with adversarial or noisy context Du et al. [2024].

Despite the emergence of financial benchmarks such as FinEval Zhang et al. [2023b], existing approaches treat financial tasks as general document Query-Answering (QA) Wang et al. [2024], lacking structured inputs and business-grounded reasoning required in practice. Thus, there emerges the gap between benchmark performance and real-world applicability.

To address these limitations, we introduce BizFinBench, a comprehensive benchmark designed to rigorously evaluate LLMs across a broad spectrum of real-world financial tasks. In contrast to previous benchmarks, BizFinBench adopts a business-driven data construction methodology and emphasizes contextual complexity and adversarial robustness. It encompasses five key dimensions: QA, prediction & recognition, reasoning, information extraction, and numerical calculation. Under these dimensions, BizFinBench comprises nine distinct categories: anomalous event attribution, financial numerical computation, financial time reasoning, financial tool usage, financial knowledge QA, financial data description, emotional value evaluation, stock price prediction, and financial named entity recognition.

A core characteristic of BizFinBench is the focus on business-contextual evaluation. For example, in the anomalous event attribution task, LLMs are required to identify the causes of stock price anomalies by analyzing time-sensitive news feeds, some of which are deliberately embedded with misleading positive or negative information. This setting challenges LLMs to perform fine-grained reasoning and signal discrimination under realistic noise and uncertainty.

In addition to the benchmark design, a critical component of BizFinBench is the design of a reliable evaluation methodology. While constructing realistic tasks is essential, evaluating LLM outputs, particularly for open-ended, complex financial problems, remains a significant challenge.

Traditional human evaluation provides high-quality judgments but suffers from two major drawbacks: (1) the annotation cost increases exponentially with the scale and domain specificity of financial tasks, and (2) subjective inconsistencies among annotators can introduce substantial noise. Although

Table 1: Comparison Between BizFinBench and Other Financial Datasets

Data Year Task Examples Language Source Business-based FLUE 2022 Multiple financial NLP tasks 26292 English Aggregated from existing sources ✗ FLARE 2023 Multiple financial NLP tasks,

19196 Chinese, English Aggregated from existing sources ✗

financial prediction tasks

CF-Benchmark 2024 Multiple financial NLP tasks 3917 Chinese except ✗ FinEval 2023 Multiple financial NLP tasks 8351 Chinese Financial field examination & except ✗ FinQA 2021 Financial numerical reasoning 8281 English except ✗ FinancelQ 2023 Multiple financial NLP tasks 7137 Chinese Financial field examination & except ✗ CGCE 2023 Multiple financial NLP tasks 150 Chinese, English except ✗ BizFinBench 2025 Multiple financial NLP tasks,

7016 Chinese except ✓

financial prediction tasks

recent approaches like LLM-as-a-Judge Gu et al. [2024] attempt to automate evaluation through prompt-based simulations of human judgment, they are prone to prompt bias and generally lack alignment with expert-level assessments. These limitations are further magnified in the financial domain, where tasks demand multi-step reasoning, contextual interpretation, and robustness against adversarial or misleading signals. As such, existing evaluation paradigms are insufficient to capture the depth and nuance required for trustworthy assessment.

To address this gap, we propose IteraJudge, an iterative calibration-based evaluation framework tailored for financial LLM benchmarks. Drawing inspiration from the RevisEval framework Zhang et al. [2024a], IteraJudge enhances evaluation accuracy and reliability through three core mechanisms: evaluation dimension disentanglement, sequential correction generation, and reference-aligned assessment. By integrating IteraJudge into BizFinBench, we establish a rigorous and interpretable evaluation pipeline for LLM performance in high-stakes financial contexts.

In summary, the major contributions of our work are as follows:

- • We propose BizFinBench, the first evaluation benchmark in the financial domain that integrates business-oriented tasks, covering 5 dimensions and 9 categories. It is designed to assess the capacity of LLMs in real-world financial scenarios.
- • We design a novel evaluation method, i.e., IteraJudge, which enhances the capability of LLMs as a judge by refining their decision boundaries in specific financial evaluation tasks.
- • We conduct a comprehensive evaluation with 25 LLMs based on BizFinBench, uncovering key insights into their strengths and limitations in financial applications.

### 2 Related Work

In this section, we present existing evaluation benchmarks in financial domains. Then, we present the major LLMs specialized in financial domains.

#### 2.1 Financial Evaluation Benchmarks

FLUE Shah et al. [2022] is a comprehensive suite of benchmarks covering five key financial tasks: sentiment analysis, news headline classification, named entity recognition, structural boundary detection, and question answering. Building on FLUE, FLARE Xie et al. [2023] expands the evaluation to include time-series processing capabilities, adding tasks such as stock price movement prediction.

In addition to FLUE and FLARE, several specialized datasets focus on various aspects of financial evaluation. For example, FinQA Chen et al. [2022a] provides QA pairs annotated by financial experts, accompanied by earnings reports from S&P 500 companies. This dataset supports financial question answering, emphasizing detailed, factual responses based on corporate financial data. ConvFinQA Chen et al. [2022b] extends this by incorporating multi-turn dialogues, enabling more sophisticated interactions within the context of earnings reports, thus broadening the scope of financial evaluation to conversational contexts.

FinEval Zhang et al. [2023b] adopts a quantitative evaluation approach, combining long-term research insights with manual curation and featuring diverse question types. However, it primarily emphasizes static knowledge assessment and lacks coverage of dynamic, real-time financial tasks and fine-

Reasoning

Analyze causal relationships in financial events, infer decision pathways, and assess strategies based on temporal and contextual data.

Information Extraction

Prediction Recognition

[Figure 3]

Forecast market trends, predict asset movements, and analyze sentiment using historical data and textual signals.

Identify and extract key financial entities—companies, tickers, instruments—from unstructured text using NLP techniques.

[Figure 4]

[Figure 5]

Financial Resources

[Figure 6]

[Figure 7]

Question Answering

Numerical Calcilation

Media news databases

[Figure 8]

[Figure 9]

[Figure 10]

Perform financial computations and optimizations, calculate risk metrics, and solve portfolio allocation problems using quantitative methods.

Retrieve and synthesize financial knowledge, answer domain-specific queries, and explain market concepts with data-backed insights.

[Figure 11]

reports academic

- Figure 2: Distribution of tasks in BizFinBench across five key dimensions. The benchmark is structured around five dimensions, each focusing on a distinct capability of financial large language models. The figure also briefly illustrates the core focus of each dimension.

grained capability diagnostics, which limits its effectiveness in benchmarking models under complex, business-driven financial scenarios.

Expanding beyond traditional financial instruments such as stocks, bonds, and mutual funds, FinancelQ Duxiaoman DI Team [2023] introduces emerging topics such as cryptocurrencies and blockchain technologies. This dataset can be exploited for evaluating models in the rapidly evolving field of digital finance.

In the context of Chinese financial benchmarks, several recent datasets have been released, including CFBenchmark Lei et al. [2023], which focuses on Chinese financial text analysis; DISC-FINSFT Chen et al. [2023], designed for financial sentiment analysis and forecasting; and CGCE Zhang et al. [2023c], which extends financial evaluation to include general knowledge and commonsense reasoning in Chinese financial documents.

- Table 1 provides a comprehensive comparison of existing financial benchmarks, detailing key aspects such as the year of release, the number of samples, language coverage, data sources, and whether the dataset was constructed with real business scenarios in mind. From the comparison, it is evident that while several benchmarks focus on financial knowledge or specific task types, they often rely on synthetic data or public information without a strong connection to actual business applications. In contrast, BizFinBench is the only benchmark explicitly designed around real-world financial operations and user interactions, making it uniquely positioned to evaluate the practical effectiveness of LLMs in authentic business environments. This business-centric design ensures higher relevance, realism, and applicability of the tasks included in the benchmark.

#### 2.2 Financial Large Language Models

By training on a large corpus of financial data based on BERT, FinBERT Araci [2019] was proposed as a pre-trained model for the financial domain, primarily used for sentiment analysis of financial texts. Subsequently, models such as FinMA Xie et al. [2023], InvestLM Yang et al. [2023a], and FinGPT Yang et al. [2023b] were fine-tuned on LLaMA Touvron et al. [2023] to further enhance their performance in the financial domain. The XuanYuan3-70B model, built on the LLaMA3-70B architecture and incrementally pre-trained with a vast amount of Chinese and English corpora, focuses on the financial sector and is capable of handling complex tasks such as financial event interpretation, investment research applications, compliance, and risk management. BloombergGPT Wu et al. [2023] is a 50-billion-parameter LLM based on the Bloom architecture, specifically designed for the financial industry, demonstrating strong adaptability in the financial domain. Meanwhile, Baichuan4Finance Zhang et al. [2024b] has achieved an accuracy rate of over 95% in various certification fields

User： Why did Tesla stock soar? Agent: Why did Tesla's stock soar on November 15, 2023? query： Tesla stock, November 15, 2023 and the previous week, related positive news. Access 1: …… Access 2: …… Access 3: ……

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

online badcase

cleaned user query

access related data

add prompt

design classification

noise removal & categorize

Reference answer

Manual marking

dataset

Figure 3: Workflow of BizFinBench dataset construction.

such as banking, insurance, funds, and securities, further proving its exceptional performance in the vertical financial sector. Dianjin-R1 Zhu et al. [2025] is designed for complex financial reasoning tasks and incorporates structured supervision along with dual-reward reinforcement learning, enabling it to outperform strong baselines across a range of financial benchmarks.

In addition, we also consider general-purpose models in our experiments, as many of them have undergone pre-training on datasets that contain financial texts, such as GPT-4o.

### 3 BizFinBench

In this section, we detail the design of BizFinBench, a comprehensive benchmark specialized for evaluating LLMs in financial domains. Compared to previous datasets, BizFinBench places a strong emphasis on business practicality and real-world applicability, aiming to bridge the gap between academic evaluation and the complex challenges encountered in real-world financial scenarios.

To capture the multifaceted nature of financial intelligence, we organize the benchmark into 10 distinct task types, which are further grouped into 5 overarching evaluation dimensions. As illustrated in Figure 2 1, these dimensions reflect key capabilities required in financial applications. For instance, the numerical computation dimension includes tasks that require models to perform financial computations and optimizations, calculate risk metrics, and solve portfolio allocation problems using quantitative methods. This dimension is designed to evaluate the capability of LLMs to apply precise mathematical reasoning in realistic financial contexts, where accuracy and analytical rigour are critical. This structured categorization not only facilitates a fine-grained assessment of model strengths and weaknesses but also ensures that each component of the benchmark aligns with practical demands observed in financial services and business analytics.

- 3.1 Data Construction

Our dataset is primarily sourced from real user queries on the iwencai APP 2, the APP serves a broad user base of individual investors and financial professionals, offering functionalities such as stock screening, market analysis, and personalized investment assistance. Leveraging advanced Artificial Intelligence (AI) technologies, iwencai enables users to perform complex financial analyses through natural language queries, covering areas like A-shares, Hong Kong and U.S. stocks, ETFs, and macroeconomic indicators.

Based on an extensive analysis of user queries from Platform A, our financial experts identified nine representative task categories that frequently appear in real-world financial scenarios. These include time reasoning, numerical computation, sentiment analysis, and so on. Notably, these categories collectively account for over 90% of the queries observed on the platform, making them highly representative of actual business needs in financial decision-making.

To construct our dataset, we first aggregate a large set of real user queries, then employ GPT-

##### 4o OpenAI [2023] to clean noisy entries, filter out incomplete or invalid ones, and classify each valid query into the appropriate expert-defined category. For underrepresented categories, we further use GPT-4o to synthesize additional data, ensuring category balance and coverage. This process results in a high-quality dataset tailored to practical financial applications.

1An English version is included in the Appendix G 2https://www.iwencai.com/

##### Table 2: Overview of BizFinBench Datasets

Category Data Evaluation Dimensions Metrics Numbers Avg Len.

Reasoning Anomalous Event Attribution (AEA) Causal consistency Information relevance Noise resistance

Accuracy 1064 939

Financial Time Reasoning (FTR) Temporal reasoning correctness Accuracy 514 1162 Financial Tool Usage (FTU) Tool selection appropriateness

Judge Score 641 4556

Parameter input accuracy Multi-tool coordination

Numerical calculation Financial Numerical Computation (FNC) Computational accuracy Unit consistency

Accuracy 581 651

Q&A Financial Knowledge QA (FQA) Question comprehension Knowledge coverage Answer accuracy

Judge Score 990 22

Financial Data Description (FDD) Trend accuracy Data consistency

Judge Score 1461 311

Prediction recognition Emotion Recognition (ER) Emotion classification accuracy Implicit information extraction

Accuracy 600 2179 Stock Price Prediction (SP) Trend judgment, Causal reasoning Accuracy 497 4498

Information extraction Financial Named Entity Recognition (FNER) Recognition accuracy Entity classification correctness

Accuracy 435 533

Next, we collect relevant contextual data from internal financial databases and external sources based on the content of each user query. This includes stock prices, historical trading data, financial news, company disclosures, and so on. These sources are directly related to the query topic. For example, as illustrated in Figure 3, when a user poses the question "Why did Tesla stock soar?", the raw query typically lacks explicit temporal markers. To address this problem, we automatically retrieve the timestamp, at which the query was originally issued and utilize it to construct a temporally anchored version of the query. This allows us to retrieve the most relevant financial information and news surrounding that specific point in time.

To enhance the discriminative power of the dataset, we carefully introduce designed distractor data into the context. These distractors are chosen to assess the reasoning capabilities of the model and include misleading but plausible information, such as news articles from unrelated companies, articles with opposing market sentiment (such as negative news during a stock rally), or temporally misaligned events. This step ensures that answering the question correctly requires understanding both the financial context and the time-sensitive nature of the data, rather than relying on superficial keyword matches.

Once the query and context are constructed, they are paired with a task-specific prompt and submitted to a large language model, such as GPT-4o, to generate candidate answers. These preliminary answers are not directly included in the dataset. Instead, each data point undergoes a rigorous human annotation and validation process.

To ensure a high level of data quality and reliability, every entry is independently reviewed and annotated by three senior financial experts. Each expert has over five years of professional experience in roles such as equity research, investment analysis, or portfolio management, and has previously worked at top-tier financial institutions, including securities firms, asset management companies, or banks. During annotation, the experts assess the accuracy of the model-generated answers and verify whether the task category assigned to the query is appropriate.

A data point is accepted into the final dataset only when all three experts reach full consensus across all aspects, including answer validity, contextual consistency, and category correctness. If disagreements arise, the entry is subjected to further review and iterative refinement until a unanimous agreement is reached. This multi-layered annotation process ensures that the dataset is not only factually accurate but also aligned with real-world financial reasoning and application standards.

#### 3.2 Statistics

The BizFinBench benchmark consists of a total of 6,781 entries, encompassing a wide variety of tasks designed to assess model performance across diverse financial challenges. By testing models on these tasks, we aim to evaluate not only their individual capabilities but also their ability to generalize across multiple facets of financial data analysis.

[Figure 21]

Figure 4: IteraJudge Pipeline.

- Table 2 provides a detailed breakdown of the dataset, including the evaluation dimensions, corresponding metrics, the number of instances per task, and the average token length per entry 3. The dataset exhibits significant variability in input length, ranging from just 22 tokens to as many as 4,556 tokens. This broad range reflects the complexity and heterogeneity of real-world financial scenarios and presents a meaningful challenge for models to demonstrate their ability to process both short and long financial texts effectively.

#### 3.3 IteraJudge: An Incremental Multi-Dimensional Evaluation Framework

As shown in Figure 4, IteraJudge evaluation framework performs dimension-decoupled assessment through a three-phase pipeline:

- 1. Given question q and initial answer y ∼ pmodel(·|q), we sequentially refine the output across dimensions E = {e1,...,eK} via prompted LLM transformations:

yk = LLMrefine(yk−1 ∥ P(ek,q)), where y0 = y (1) creating an interpretable improvement trajectory {yk}Kk=0.

- 2. The fully refined yK serves as an auto - generated quality benchmark.
- 3. A judge model computes the final score through contrastive evaluation: score(y) = LLMjudge(q,y,yK,E) (2)

where the delta (yK − y) quantitatively reveals the dimensional deficiencies of LLMs.

This question-anchored, iterative refinement process enables granular diagnosis while maintaining contextual consistency through explicit q-preservation in all steps.

- 4 Experiments

This section summarizes the evaluated models (Section 4.1), including SOTA LLMs, inferenceoptimized models, and multimodal large language models. Section 4.2 describes the experimental setup. Section 4.3 presents key results across financial tasks, followed by the performance analysis of IteraJudge in Section 4.4.

#### 4.1 Evaluated Models

We conducted a systematic evaluation of current mainstream LLMs on BizFinBench. For closedsource models, we selected five industry-recognized SOTA models: OpenAI’s GPT-4o, o3 and

- o4-mini, Google’s Gemini-2.0-Flash, and Anthropic’s Claude-3.5-Sonnet. For open-source models,
- our evaluation covered both general-purpose LLMs including the Qwen2.5 series, Llama-3.1 series and Llama-4-Scout, as well as the financial-specialized Xuanyuan3-70B model. To comprehensively assess model capability boundaries, we also incorporated the DeepSeek-R1 series (including the

3Detailed dataset information is provided in Appendix E.

R1-distill variant) which excels at complex reasoning tasks, the newly open-sourced reasoning model QwQ-32B and the recently released Qwen3 series with hybrid reasoning capabilities. Furthermore, to evaluate MLLMs on our benchmark, we extended our experiments to assess the performance of the Qwen-VL series of MLLMs4.

- Table 3: Performance Comparison of Large Language Models on BizFinBench. The models are evaluated across multiple tasks, with results color-coded to represent the top three performers for each task: golden indicates the top-performing model, silver represents the second-best result, and

bronze denotes the third-best performance.

Model AEA FNC FTR FTU FQA FDD ER SP FNER Average Propretary LLMs

ChatGPT-o3 86.23 61.30 75.36 89.15 91.25 98.55 44.48 53.27 65.13 73.86 ChatGPT-o4-mini 85.62 60.10 71.23 74.40 90.27 95.73 47.67 52.32 64.24 71.29 GPT-4o 79.42 56.51 76.20 82.37 87.79 98.84 45.33 54.33 65.37 71.80 Gemini-2.0-Flash 86.94 62.67 73.97 82.55 90.29 98.62 22.17 56.14 54.43 69.75 Claude-3.5-Sonnet 84.68 63.18 42.81 88.05 87.35 96.85 16.67 47.60 63.09 65.59

Open source LLMs

Qwen2.5-7B-Instruct 73.87 32.88 39.38 79.03 83.34 78.93 37.50 51.91 30.31 56.35 Qwen2.5-72B-Instruct 69.27 54.28 70.72 85.29 87.79 97.43 35.33 55.13 54.02 67.70 Qwen2.5-VL-3B 53.85 15.92 17.29 8.95 81.60 59.44 39.50 52.49 21.57 38.96 Qwen2.5-VL-7B 73.87 32.71 40.24 77.85 83.94 77.41 38.83 51.91 33.40 56.68 Qwen2.5-VL-14B 37.12 41.44 53.08 82.07 84.23 7.97 37.33 54.93 47.47 49.52 Qwen2.5-VL-32B 76.79 50.00 62.16 83.57 85.30 95.95 40.50 54.93 68.36 68.62 Qwen2.5-VL-72B 69.55 54.11 69.86 85.18 87.37 97.34 35.00 54.94 54.41 67.53 Qwen3-1.7B 77.40 35.80 33.40 75.82 73.81 78.62 22.40 48.53 11.23 50.78 Qwen3-4B 83.60 47.40 50.00 78.19 82.24 80.16 42.20 50.51 25.19 59.94 Qwen3-14B 84.20 58.20 65.80 82.19 84.12 92.91 33.00 52.31 50.70 67.05 Qwen3-32B 83.80 59.60 64.60 85.12 85.43 95.37 39.00 52.26 49.19 68.26 Xuanyuan3-70B 12.14 19.69 15.41 80.89 86.51 83.90 29.83 52.62 37.33 46.48 Llama-3.1-8B-Instruct 73.12 22.09 2.91 77.42 76.18 69.09 29.00 54.21 36.56 48.95 Llama-3.1-70B-Instruct 16.26 34.25 56.34 80.64 79.97 86.90 33.33 62.16 45.95 55.09 Llama 4 Scout 73.60 45.80 44.20 85.02 85.21 92.32 25.60 55.76 43.00 61.17 DeepSeek-V3 (671B) 74.34 61.82 72.60 86.54 91.07 98.11 32.67 55.73 71.24 71.57 DeepSeek-R1 (671B) 80.36 64.04 75.00 81.96 91.44 98.41 39.67 55.13 71.46 73.05 QwQ-32B 84.02 52.91 64.90 84.81 89.60 94.20 34.50 56.68 30.27 65.77 DeepSeek-R1-Distill-Qwen-14B 71.33 44.35 16.95 81.96 85.52 92.81 39.50 50.20 52.76 59.49 DeepSeek-R1-Distill-Qwen-32B 73.68 51.20 50.86 83.27 87.54 97.81 41.50 53.92 56.80 66.29

#### 4.2 Experiment Setting

All LLMs were configured with a maximum generation length of 1,024 tokens, temperature parameter T = 0, and batch size B = 1000. We employed GPT-4o as the unified evaluation judge. Opensource models were deployed on an 8×NVIDIA H100 cluster, while closed-source models were accessed via their official APIs. The complete evaluation required approximately 10 hours with a total computational cost of $21,000.

To ensure standardized outputs and facilitate automated assessment, we constrained all models to produce strictly JSON-formatted responses containing two mandatory fields: ① Chain-of-Thought(cot): Detailed logic trace with intermediate steps; and ② Answer: Final conclusion derived after reasoning5.

#### 4.3 Main Results

Our evaluation on the BizFinBench benchmark reveals distinct capabilities of LLMs in the financial domain. All results as shown in Table 3. In the AEA task, Gemini-2.0-Flash achieves SOTA performance with a score of 86.94, closely followed by ChatGPT-o3 (86.23) and ChatGPT-o4mini (85.62), demonstrating the strong and consistent performance of closed-source models in complex financial understanding. Moreover, proprietary models dominate knowledge-intensive tasks—exemplified by the leading performance of GPT-4o in FDD with a score of 98.84 and the leadership of ChatGPT-o3 in FTU with 89.15. However, open-source models like DeepSeek-V3

- 4The specific details of the relevant models can be found in the Appendix C.
- 5The formatting and style for each dataset are presented in Appendix F. Appendix D provides the evaluation

prompts used in our experiments.

(671B) show impressive competitiveness, particularly surpassing GPT-4o (65.37) in FNER with a score of 71.46.

Table 4: Comparative Evaluation of Judgment Methods Across Different LLM Judges

Methods Financial Data Description Financial Tool Usage LLM as a judge (GPT-4o) Spearman: 0.4848 Spearman: 0.8000 Ours (GPT-4o) Spearman: 0.5684 Spearman: 0.8667 LLM as a judge (DeepSeek-V3) Spearman: 0.4685 Spearman: 0.7500 Ours (DeepSeek-V3) Spearman: 0.4830 Spearman: 0.7833 LLM as a judge (Gemini-2.0-Flash) Spearman: 0.3763 Spearman: 0.7333 Ours (Gemini-2.0-Flash) Spearman: 0.4087 Spearman: 0.8167 LLM as a judge (Qwen2.5-72B-Instruct) Spearman: 0.3112 Spearman: 0.7000 Ours (Qwen2.5-72B-Instruct) Spearman: 0.4282 Spearman: 0.7500

Our evaluation leads to three key insights: First, model scale plays a crucial role in numerical reasoning. Taking the Qwen3 series as an example, performance consistently improves with increasing model size, e.g., from the smallest 1.7B to the largest 32B, across nearly all tasks. Second, FTR emerges as a particularly challenging task, with a substantial score gap of 32.19 points between the top performer, GPT-4o (76.20), and lower-performing models like Llama-3.1-8B-Instruct (2.91), highlighting the need for targeted optimization in temporal reasoning. Third, while most models excel in structured data tasks—achieving scores above 90 in cases like FDD. They underperform in more complex scenarios such as ER. Even GPT-4o scores only 45.33 in ER, with the best open-source model reaching just 41.50, underscoring significant room for improvement in financial sentiment analysis.

Additionally, we identify two intriguing phenomena: (1) Llama-3.1-70B-Instruct demonstrates strong performance in Stock Prediction (62.16) but struggles with related reasoning tasks, and (2) distilled models (e.g., DeepSeek-R1-Distill-Qwen-32B) maintain competitive performance in FTU(83.27) but exhibit limitations in temporal reasoning (FTR, 50.86), highlighting considerable variation in knowledge retention across different capabilities. These findings suggest that while current LLMs are competent at handling basic financial tasks, they still face significant limitations when dealing with complex challenges that require integrated knowledge, particularly in cross-concept reasoning within financial contexts.

#### 4.4 IteraJudge Ablation Experiments

To rigorously validate the effectiveness of IteraJudge, we conducted ablation experiments on the FDD and FTU benchmark datasets. We selected Qwen2.5-7B-Instruct as the evaluated model and employed GPT-4o, DeepSeek-V3, Gemini-1.5-Flash, and Qwen2.5-72B-Instruct as judge models to evaluate its generated responses. Three sets of experiments were designed: (1) expert evaluation, (2) the vanilla LLM-as-a-Judge approach, and (3) the full IteraJudge framework. We take the Spearman correlation between the evaluation methods, i.e., vanilla LLM-as-a-Judge and IteraJudge.

The experimental results are demonstrated in Table 4, compared to the LLM-as-a-Judge approach, IteraJudge achieves a maximum improvement of 17.24% and a minimum improvement of 3.09% in terms of Spearman correlation on the FDD benchmark dataset. In the FTU benchmark dataset, it shows a maximum improvement of 11.37% and a minimum improvement of 4.44%. These results confirm the effectiveness of IteraJudge in mitigating evaluation bias.

### 5 Conclusion

In this work, we propose BizFinBench, i.e., the first open-source benchmark dataset, which consists of the dataset deeply integrated with real-world financial business scenarios and a iterative calibrationbased evaluation framework, i.e., IteraJudge. We conducted a comprehensive evaluation of 25 SOTA LLMs, encompassing both closed-source and open-source models, across multiple task dimensions. Our results reveal significant performance gaps between existing LLMs and human-level expectations in several business-critical areas, highlighting the unique challenges of financial artificial intelligence. We find that no model dominates every task, while ChatGPT-o3, ChatGPT-o4-mini, GPT-4o, Gemini2.0-Flash, DeepSeek-R1, and Llama-3.1-70B-Instruct corresponds to the best performance in diverse metrics. In addition, experimental results also demonstrate that closed-source models place in the

top three on eight of nine subtasks. Furthermore, extensive experimental results reveal significant advantages of IteraJudge. BizFinBench serves not only as a rigorous benchmark for evaluating financial reasoning capabilities, but also as a practical guide for deploying LLMs in real-world financial applications. We believe this benchmark can accelerate progress in the development of trustworthy, high-performing financial language models.

### References

Zhiyu Zoey Chen, Jing Ma, Xinlu Zhang, Nan Hao, An Yan, Armineh Nourbakhsh, Xianjun Yang, Julian McAuley, Linda Petzold, and William Yang Wang. A survey on large language models for critical societal domains: Finance, healthcare, and law. arXiv preprint arXiv:2405.01769, 2024.

Che Liu, Yingji Zhang, Dong Zhang, Weijie Zhang, Chenggong Gong, Haohan Li, Yu Lu, Shilin Zhou, Yue Lu, Ziliang Gan, et al. Nexus-o: An omni-perceptive and-interactive model for language, audio, and vision. arXiv preprint arXiv:2503.01879, 2025.

Rongjunchen Zhang, Tingmin Wu, Xiao Chen, Sheng Wen, Surya Nepal, Cecile Paris, and Yang Xiang. Dynalogue: A transformer-based dialogue system with dynamic attention. In Proceedings of the ACM Web Conference 2023, pages 1604–1615, 2023a.

Guilong Lu, Xiaolin Ju, Xiang Chen, Wenlong Pei, and Zhilong Cai. Grace: Empowering llm-based software vulnerability detection with graph structure and in-context learning. Journal of Systems and Software, 212:112031, 2024.

Qianqian Xie, Qingyu Chen, Aokun Chen, Cheng Peng, Yan Hu, Fongci Lin, Xueqing Peng, Jimin Huang, Jeffrey Zhang, Vipina Keloth, et al. Medical foundation large language models for comprehensive text analysis and beyond. npj Digital Medicine, 8(1):141, 2025.

Huaqin Zhao, Zhengliang Liu, Zihao Wu, Yiwei Li, Tianze Yang, Peng Shu, Shaochen Xu, Haixing Dai, Lin Zhao, Gengchen Mai, et al. Revolutionizing finance with llms: An overview of applications and insights. arXiv preprint arXiv:2401.11641, 2024.

Ziliang Gan, Yu Lu, Dong Zhang, Haohan Li, Che Liu, Jian Liu, Ji Liu, Haipang Wu, Chaoyou Fu, Zenglin Xu, et al. Mme-finance: A multimodal finance benchmark for expert-level understanding and reasoning. arXiv preprint arXiv:2411.03314, 2024.

Kelvin Du, Frank Xing, Rui Mao, and Erik Cambria. An evaluation of reasoning capabilities of large language models in financial sentiment analysis. In 2024 IEEE Conference on Artificial Intelligence (CAI), pages 189–194. IEEE, 2024.

Liwen Zhang, Weige Cai, Zhaowei Liu, Zhi Yang, Wei Dai, Yujie Liao, Qianru Qin, Yifei Li, Xingyu Liu, Zhiqiang Liu, Zhoufan Zhu, Anbo Wu, Xin Guo, and Yun Chen. Fineval: A chinese financial domain knowledge evaluation benchmark for large language models, 2023b.

Minzheng Wang, Longze Chen, Cheng Fu, Shengyi Liao, Xinghua Zhang, Bingli Wu, Haiyang Yu, Nan Xu, Lei Zhang, Run Luo, et al. Leave no document behind: Benchmarking long-context llms with extended multi-doc qa. arXiv preprint arXiv:2406.17419, 2024.

Fin-Eva Team. Fin-eva version 1.0, 2023. Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen,

Shengjie Ma, Honghao Liu, et al. A survey on llm-as-a-judge. arXiv preprint arXiv:2411.15594, 2024.

Qiyuan Zhang, Yufei Wang, Tiezheng Yu, Yuxin Jiang, Chuhan Wu, Liangyou Li, Yasheng Wang, Xin Jiang, Lifeng Shang, Ruiming Tang, et al. Reviseval: Improving llm-as-a-judge via responseadapted references. arXiv preprint arXiv:2410.05193, 2024a.

Raj Shah, Kunal Chawla, Dheeraj Eidnani, Agam Shah, Wendi Du, Sudheer Chava, Natraj Raman, Charese Smiley, Jiaao Chen, and Diyi Yang. When flue meets flang: Benchmarks and large pretrained language model for financial domain. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 2322–2335, 2022.

Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, and Jimin Huang. Pixiu: A large language model, instruction data and evaluation benchmark for finance. arXiv preprint arXiv:2306.05443, 2023.

Zhiyu Chen, Wenhu Chen, Charese Smiley, and et al. Sameena Shah. Finqa: A dataset of numerical reasoning over financial data, 2022a.

Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and William Yang Wang. Convfinqa: Exploring the chain of numerical reasoning in conversational finance question answering, 2022b.

Duxiaoman DI Team. FinanceIQ, 2023. URL https://github.com/Duxiaoman-DI/XuanYuan/ tree/main/FinanceIQ. Accessed: 2024-03-18.

Yang Lei, Jiangtong Li, Ming Jiang, Junjie Hu, Dawei Cheng, Zhijun Ding, and Changjun Jiang. Cfbenchmark: Chinese financial assistant benchmark for large language model, 2023.

Wei Chen, Qiushi Wang, Zefei Long, Xianyin Zhang, Zhongtian Lu, Bingxuan Li, Siyuan Wang, Jiarong Xu, Xiang Bai, Xuanjing Huang, et al. Disc-finllm: A chinese financial large language model based on multiple experts fine-tuning. arXiv preprint arXiv:2310.15205, 2023.

Xuanyu Zhang, Bingbing Li, and Qing Yang. Cgce: A chinese generative chat evaluation benchmark

for general and financial domains, 2023c. Dogu Araci. Finbert: Financial sentiment analysis with pre-trained language models, 2019. Yi Yang, Yixuan Tang, and Kar Yan Tam. Investlm: A large language model for investment using

financial domain instruction tuning, 2023a. Hongyang Yang, Xiao-Yang Liu, and Christina Dan Wang. Fingpt: Open-source financial large language models, 2023b.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. Bloomberggpt: A large language model for finance, 2023.

Hanyu Zhang, Boyu Qiu, Yuhao Feng, Shuqi Li, Qian Ma, Xiyuan Zhang, Qiang Ju, Dong Yan, and Jian Xie. Baichuan4-finance technical report. arXiv preprint arXiv:2412.15270, 2024b.

Jie Zhu, Qian Chen, Huaixia Dou, Junhui Li, Lifan Guo, Feng Chen, and Chi Zhang. Dianjinr1: Evaluating and enhancing financial reasoning in large language models. arXiv preprint arXiv:2504.15716, 2025.

OpenAI. Gpt-4 technical report, 2023. Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu

Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Claude. Claude 3.5 sonnet, 2024. URL https://www.anthropic.com/news/claude-3-5-sonnet. Jinze Bai, Shuai Bai, and et al. Yunfei Chu. Qwen technical report. arXiv preprint arXiv:2309.16609,

2023. XuanYuan Team. Xuanyuan3-70b report, September 2024. URL https://github.com/ Duxiaoman-DI/XuanYuan.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025. URL https://qwenlm.github.io/blog/qwq-32b/.

### A Reproducibility Statement

To ensure the reproducibility of our results, we have made substantial efforts to provide all necessary details and materials. Specifically, Section 3.1 presents the complete process of dataset construction, including data collection strategies. Furthermore, the benchmark setup and evaluation procedures are thoroughly described in Section 4. All evaluation metrics are clearly defined to facilitate independent verification and replication of our experiments by the research community.

### B Limitations

In this work, we propose a novel benchmark and conduct a comprehensive analysis of different LLMs’ capabilities in solving financial business problems. However, several limitations remain:

- (1) Our method for extracting final answers from model outputs is not yet perfect. In some cases, this method fails to locate an answer, leading to reported accuracy being an approximate lower bound. Additionally, due to potential formatting differences between the extracted answers and the ground truth, we employ a rule-based approach to measure exact matches between the two, which may introduce an estimated 2% error in our experiments.
- (2) Our benchmark is primarily based on currently available financial data and task settings. Although it covers multiple key sub-tasks, some business scenarios may still be underrepresented. For example, highly specialized financial tasks such as complex derivatives pricing, risk management modeling, or decision support based on real-time market data are not yet fully reflected in our benchmark. This implies that our evaluation results may not completely capture LLMs’ real-world performance in more complex financial scenarios.
- (3) While we evaluate multiple SOTA LLMs under the same computational environment to ensure fairness, model performance may still be influenced by training data, inference strategies, and hyperparameter settings. Additionally, discrepancies between inference mechanisms in API-based and locally deployed models could introduce experimental biases.
- (4) Our evaluation primarily focuses on models’ abilities in single-turn question answering and task completion. However, in real-world applications, financial decision-making is often a complex, multi-step process involving long-term reasoning, external tool utilization, and multi-turn interactions. The current evaluation framework does not fully cover these aspects, highlighting the need for further expansion to better reflect LLMs’ potential applications in financial business scenarios.

For future work, we plan to optimize the answer extraction method to enhance evaluation accuracy and explore more advanced metrics to mitigate errors caused by format mismatches. Additionally, we aim to expand the benchmark’s coverage by incorporating more challenging financial tasks and refining experimental settings to improve reproducibility and fairness.

### C Model detail

To better ensure the comprehensiveness and robustness of our evaluation, we selected a wide range of models that differ in architecture, parameter size, training objectives, and domain specialization.

- Table 5 presents detailed information on the 25 evaluation models used in this study.

### D Instruction

Figure 5, Figure 6, and Figure 7 illustrate the instructions used for model evaluation on the open-ended answer dataset.

- Table 5: Summary of Large Language Models Evaluated on BizFinBench. * indicates that the model is a Mixture-of-Experts (MoE) model.

Model Size Open source Evalation Release date Domain GPT-4o OpenAI [2023] – ✗ API 01/29/2025 General ChatGPT-o3 – ✗ API 04/16/2025 General ChatGPT-o4-mini – ✗ API 04/16/2025 General Gemini-2.0-Flash Team et al. [2023] – ✗ API 12/11/2024 General Claude-3.5-Sonnet Claude [2024] – ✗ API 06/20/2024 General Qwen2.5-Instruct Bai et al. [2023] 7B,72B ✓ Local 09/19/2024 General Qwen2.5-VL 3B,7B,14B,32B,72B ✓ Local 01/28/2025 General Qwen3 8B,14B,32B,30B* ✓ Local 04/29/2025 General XuanYuan3-70B-Chat Team [2024] 70B ✓ Local 09/06/2024 Finance Llama-3.1-Instruct Dubey et al. [2024] 8B,70B ✓ Local 07/24/2024 General Llama 4 109B* ✓ Local 04/05/2025 General DeepSeek-V3 Liu et al. [2024] 671B* ✓ Local 12/26/2024 General DeepSeek-R1 Liu et al. [2024] 671B* ✓ Local 12/26/2024 General QwQ-32B Team [2025] 32B ✓ Local 03/06/2025 General DeepSeek-R1-Distill-Qwen-14B Liu et al. [2024] 14B,32B ✓ Local 12/26/2024 General

[Figure 22]

- Figure 5: The instructions utilized in the evaluation of the FDD dataset.

[Figure 23]

##### Figure 6: The instructions utilized in the evaluation of the FQA dataset.

[Figure 24]

- Figure 7: The instructions utilized in the evaluation of the FTU dataset.

- E Dataset details The details of each dataset type are as follows.

- • Anomalous Event Attribution (AEA): This dataset evaluates the model’s ability to trace financial anomalies based on given information such as timestamps, news articles, financial reports, and stock movements. The model must identify the cause-and-effect relationships behind sudden market fluctuations and distinguish relevant factors from noise.
- • Financial Numerical Computation (FNC): This dataset assesses the model’s ability to perform accurate numerical calculations in financial scenarios, including interest rate calculations, return on investment (ROI), and financial ratios.
- • Financial Time Reasoning (FTR): This dataset tests the model’s ability to understand and reason about time-based financial events, such as predicting interest accruals, identifying the impact of quarterly reports, and assessing financial trends over different periods.
- • Financial Tool Usage (FTU): This dataset evaluates the model’s ability to comprehend user queries and effectively use financial tools to solve real-world problems. It covers scenarios like investment analysis, market research, and information retrieval, requiring the model to select appropriate tools, input parameters accurately, and coordinate multiple tools when needed.
- • Financial Knowledge QA (FQA): This dataset evaluates the model’s understanding and response capabilities regarding core knowledge in the financial domain. It spans a wide range of financial topics, encompassing key areas such as fundamental financial concepts, financial markets, investment theory, macroeconomics, and finance.
- • Financial Data Description (FDD): This dataset measures the model’s ability to analyze and describe structured and unstructured financial data, such as balance sheets, stock reports, and financial statements.

- • Emotion Recognition (ER): This dataset evaluates the model’s capability to recognize nuanced user emotions in complex financial market environments. The input data encompasses multiple dimensions, including market conditions, news articles, research reports, user portfolio information, and queries. The dataset covers six distinct emotional categories: optimism, anxiety, negativity, excitement, calmness, and regret.
- • Stock Price Prediction (SP): This dataset evaluates the model’s ability to predict future stock prices based on historical trends, financial indicators, and market news.
- • Financial Named Entity Recognition (FNER): This dataset focuses on evaluating the model’s ability to identify and classify financial entities such as company names, stock symbols, financial instruments, regulatory agencies, and economic indicators.

- Table 6 presents their maximum token length, minimum token length, and average length.

- Table 6: Financial Datasets Query Token Length Statistics. This table presents token length statistics for queries in financial datasets, including minimum (Min), maximum (Max), average (Avg) token counts, and total query count (Count).

Dataset Min Max Avg Count NER 415 1,194 533.1 433 FTU 4,169 6,289 4,555.5 641 AEA 680 1,396 938.7 1,064 ER 1,919 2,569 2,178.5 600 FNC 287 2,698 650.5 581 FDD 26 645 310.9 1,461 FTR 203 8,265 1,162.0 514 FQA 5 45 21.7 990 SP 1,254 5,532 4,498.1 497

- F Dataset example
- G Other

###### Financial Numerical Computation

你是一位经验丰富的金融数据分析师，请根据以下<数据参考>，回答出<用户问题>，并给出你的理由。你需要以指定的输出格式回答

### 数据参考

- 编号: 1

- 取数问句: 百度2024年8月每日的收盘价 取数结果: 为您找到1条数据

编号: 2

- 取数问句: 百度2024年9月每日的收盘价 取数结果: 为您找到1条数据

|股票代码|股票简称|收盘价 20240801|收盘价 20240802|收盘价 20240805|收盘价 20240806|收盘价 20240807|收盘价 20240808|收盘价 20240809|收盘价 20240812|收盘价 20240813|收盘价 20240814|
|---|---|---|---|---|---|---|---|---|---|---|---|
|BIDU.O|百度|86.42美元|84.49美元|82.39美元|83.01美元|82.01美元|86.54美元|85.10美元|85.23美元|86.20美元|83.99美元|
|收盘价 20240815|收盘价 20240816|收盘价 20240819|收盘价 20240820|收盘价 20240821|收盘价 20240822|收盘价 20240823|收盘价 20240826|收盘价 20240827|收盘价 20240828|收盘价 20240829|收盘价 20240830|
|86.18美元|88.97美元|90.18美元|87.98美元|89.74美元|85.79美元|85.70美元|86.22美元|84.82美元|82.36美元|83.82美元|84.62美元|

|股票代码|股票简称|收盘价 20240903|收盘价 20240904|收盘价 20240905|收盘价 20240906|收盘价 20240909|收盘价 20240910|收盘价 20240911|收盘价 20240912|收盘价 20240913|
|---|---|---|---|---|---|---|---|---|---|---|
|BIDU.O|百度|83.41美元|82.38美元|82.81美元|81.01美元|81.63美元|82.05美元|83.95美元|84.07美元|83.65美元|
|收盘价 20240916|收盘价 20240917|收盘价 20240918|收盘价 20240919|收盘价 20240920|收盘价 20240923|收盘价 20240924|收盘价 20240925|收盘价 20240926|收盘价 20240927|收盘价 20240930|
|84.08美元|85.39美元|84.67美元|87.02美元|86.24美元|88.28美元|94.81美元|94.01美元|102.56美元|105.17美元|105.29美元|

### 用户问题 question: 在2024年8月至2024年9月期间，百度的收盘价大于90美元的天数有几天？

### 输出格式 以如下的json格式进行回答 {"cot":"<填写你的思考过程>", "answer":<填写具体答案，如果是数值，请统一表示成小数，最多小数点后保留四位；如果是日期，请你以x年x 月x日的格式返回>}

[Figure 25]

{

"cot": "首先，我将检查2024年8月和2024年9月百度的每日收盘价，找出所有收盘价大于90美元的日期。从8月的数据中，有2 天的收盘价超过90美元，分别是2024年8月19日（90.18美元）和2024年8月21日（89.74美元），但89.74美元未超过90美元， 因此只计1天。从9月的数据中，有4天的收盘价超过90美元，分别是2024年9月23日（88.28美元），2024年9月24日（94.81美 元），2024年9月25日（94.01美元），2024年9月26日（102.56美元），2024年9月27日（105.17美元），2024年9月30日 （105.29美元），但88.28美元未超过90美元，因此计5天。合计共有6天的收盘价超过90美元。",

"answer": 6.0000 }

Figure 8: An example instance from the Financial Numerical Computation dataset, in which the data table is formatted using Markdown syntax.

[Figure 26]

##### Figure 9: An example instance from the Financial Time Reasoning.

###### Financial Knowledge QA

什么是货币时间价值？

货币时间价值是指货币在不同时间点上的价值差异。简单来说，今天的1元钱比将来的1元钱更有价值。这是因为货币可以被投资以 获得收益，或者由于通货膨胀，未来的货币购买力可能会下降。因此，在金融决策中，考虑货币的时间价值是非常重要的。

[Figure 27]

货币时间价值的概念主要基于以下几个方面：

- 1. 投资收益：如果你现在有100元，你可以选择将它存入银行或投资于其他项目，从而在未来获得更多的钱。因此，今天的100元比 未来的100元更有价值，因为你有机会通过投资增加它的价值。
- 2. 通货膨胀：随着时间的推移，由于通货膨胀的影响，货币的购买力会逐渐下降。这意味着，同样数量的货币在未来可能无法购买 到与现在相同数量的商品和服务。
- 3. 风险与不确定性：未来是不确定的，承诺在未来支付的款项可能因为各种原因（如违约、经济衰退等）而无法兑现。因此，人们 通常更偏好当前的确定性收益而非未来的不确定性收益。

在金融学中，货币时间价值通常通过现值（Present Value, PV）和终值（Future Value, FV）的概念来量化。现值是指未来某一时点 的一定金额按照一定的利率折算到现在的价值；终值则是指现在的一笔资金按照一定的利率增长到未来某一时点的价值。通过这些 计算，可以帮助个人和企业做出更合理的财务决策。

##### Figure 10: An example instance from the Financial Knowledge QA dataset.

###### Financial Data Description

根据下面的提供的数据描述这支股票的情况，包括变化趋势和一些数值:

|股票代码|股票简称|涨跌幅：前复权 20250110|涨跌幅：前复权 20250113|涨跌幅：前复权 20250114|涨跌幅：前复权 20250115|涨跌幅：前复权 20250116|涨跌幅：前复权 20250117|
|---|---|---|---|---|---|---|---|
|000802.SZ|平安银行|-9.97%|-1.48%|3.45%|-2.75%|0.0%|1.19%|
|涨跌幅：前复权 20250120|涨跌幅：前复权 20250121|涨跌幅：前复权 20250122|涨跌幅：前复权 20250123|涨跌幅：前复权 20250124|涨跌幅：前复权 20250127|涨跌幅：前复权 20250205|涨跌幅：前复权 20250206|
|1.33%|2.62%|-2.12%|-6.51%|-1.55%|10.06%|-10.0%|-6.19%|
|收盘价：不复权 20250206|开盘价：不复权 20250206|最高价：不复权 20250206|最低价：不复权 20250206| | | | |
|5.91元|5.64元|6.04元|5.67元| | | | |

[Figure 28]

根据提供的数据，我们可以对股票代码为000802.SZ的北京文化股票在2025年1月10日至2025年2月6日这段时间内的表现进行如下 描述：

### 涨跌幅变化趋势

- 1. **2025年1月10日**：股票大幅下跌9.97%，这可能是由于市场对该公司或行业出现了负面消息，导致投资者信心下降。
- 2. **2025年1月11日至1月14日**：股票继续下跌1.48%，但随后在1月14日反弹3.45%，显示出市场情绪有所恢复。
- 3. **2025年1月15日至1月16日**：股票再次下跌2.75%，但1月16日没有变化，市场进入观望状态。
- 4. **2025年1月17日至1月21日**：股票连续小幅上涨，涨幅分别为1.19%、1.33%和2.62%，显示出市场信心逐渐增强。
- 5. **2025年1月22日至1月24日**：股票再次出现波动，1月22日下跌2.12%，1月23日大幅下跌6.51%，1月24日小幅下跌1.55%。
- 6. **2025年1月27日**：股票大幅上涨10.06%，显示出市场情绪的强烈反弹。
- 7. **2025年2月5日至2月6日**：股票再次大幅下跌10.00%，随后在2月6日继续下跌6.19%，市场情绪再次转为悲观。 ### 数值分析

- - **2025年2月6日的收盘价**：5.91元
- - **2025年2月6日的开盘价**：5.67元
- - **2025年2月6日的最高价**：6.04元
- - **2025年2月6日的最低价**：5.67元 ### 总体趋势
- - **波动性**：在这段时间内，北京文化的股票价格波动较大，多次出现大幅涨跌，显示出市场对该股票的不确定性较高。
- - **市场情绪**：市场情绪在短期内多次变化，从大幅下跌到强烈反弹，再到再次下跌，显示出投资者对该公司或行业的看法存在较 大分歧。
- - **价格区间**：从2025年1月10日至2025年2月6日，股票价格在5.67元至6.04元之间波动，整体价格区间较窄。

### 结论 北京文化在这段时间内的股票表现较为不稳定，市场情绪波动较大。投资者应密切关注公司基本面和市场消息，以做出更合理的投 资决策。

Figure 11: An example instance from the Financial Data Description.

###### Financial Named Entity Recognition

你是一个专业的金融分析师。你的任务是从用户提供的<相关新闻>中，进行实体识别任务，识别出对应的Person、Organization、 Financial_Instruments、Market、Location、Date_Time。

这些实体具体的介绍如下： Person（人物）：包括企业高管、基金经理、分析师、其他人物名称等。 Organization（组织）：指各类公司、金融机构、监管机构等。 Financial_Products（金融产品）：包括但不限于股票名称、债券名称、基金名称、期货、期权等。 Market（市场）：特定的产业或行业领域，如科技、能源、医疗保健等。 Location（地理位置）：事件发生的地理位置。 Date_Time（日期/时间）：关键的时间信息，如财报发布日、股东大会日等。包括具体日期（如"2025年2月10日"）、年份（如"2021年"）、时 间范围（如"两年"、"近五年"）、季度（如"Q1 2024"）、以及其他相对时间表达（如"去年"、"上一年"、"过去三个月"）等等。

<相关新闻> 东方电热：全资子公司江苏东方瑞吉及镇江东方与信义硅业分别签署了《设备买卖合同》。根据上述合同，东方瑞吉为信义硅业生产多台套还 原炉，镇江东方为信义硅业生产多台套冷氢化辐射式电加热器，上述合同总价为2.98亿元

</相关新闻>

###### 要求：

- 1. 完整提取：所有出现的实体都需要记录，不进行去重，即使某个实体多次出现，也需要完整罗列。
- 2. 逐条列出：按照文本顺序依次提取实体，确保输出顺序与原文匹配。
- 3. 请以纯文本形式输出 JSON，不要包含任何代码块标记（如 ```json 或 ```），参考格式如下： { "results": [ { "type": "xxx", "text": "xxx" }, { "type": "xxx", "text": "xxx" }, { "type": "xxx", "text": "xxx" } ] }

[Figure 29]

{ "results": [ { "type": "Organization", "text": "东方电热" }, { "type": "Organization", "text": "江苏东方瑞吉" }, { "type": "Organization", "text": "镇江东方" }, { "type": "Organization", "text": "信义硅业" }, { "type": "Financial_Instruments", "text": "《设 备买卖合同》" }, { "type": "Organization", "text": "东方瑞吉" }, { "type": "Organization", "text": "信义硅业" }, { "type":

"Financial_Instruments", "text": "还原炉" }, { "type": "Organization", "text": "镇江东方" }, { "type": "Organization", "text": "信义硅 业" }, { "type": "Financial_Instruments", "text": "冷氢化辐射式电加热器" }, { "type": "Financial_Instruments", "text": "上述合同" }, { "type": "Financial_Instruments", "text": "2.98亿元" } ] }

##### Figure 12: An example instance from the Financial Named Entity Recognition.

[Figure 30]

##### Figure 13: An example instance from the Financial Tool Usage.

###### Anomalous Event Attribution

您是一个知识渊博且乐于助人的金融领域专家，需要基于<时间>和<新闻>回答用户问题，给出令用户满意的答复。以下是一个参考样例：

|序号|时间|新闻|
|---|---|---|
|1|2023年9月5日|公司半年度业绩交流会中提到，目前隆基绿能BC产能主要为30GW HPBC（高效复合钝化背接触技术）电池项 目，并已于上半年投入量产。|
|2|2023年11月15日|据报道，特斯拉现已大量发出邀请函，确认Cybertruck交付活动将于11月30日在得州超级工厂举行，这意味着发 布四年的Cybertruck终于要交车。|
|3|2023年11月14日|国盛证券指出，BC电池或成为行业未来新选择，技术加速渗透有望拉动激光设备采购需求，建议重点关注在BC 电池布局的龙头电池企业隆基股份。|
|4|2023年11月14日|当地时间11月14日，美国劳工统计局公布数据显示，CPI环比不变，同比增速降至3.2%；去掉食品、能源价格 的核心通胀环比上升0.2%，同比增速略降至4.0%。中信证券指出，预计美联储或已完成此轮加息，10年期美债 利率短期或走弱，但仍将在4%以上水平运行。|

用户问题：哪些新闻导致了特斯拉股票在2023-11-15 这个时间大涨这个事实。 答案： { "相关新闻序号": [2,4], "具体分析": "特斯拉股票在2023年11月15日大涨，主要受到两条新闻的影响。第一，新闻2提到特斯拉确认 Cybertruck交付活动将于11月30日在得州超级工厂举行，这是备受期待的事件，对市场情绪产生了积极影响，反映了市场对新车型发布和销售 业绩增长的乐观预期。第二，新闻4显示美国核心通胀环比上升0.2%，但总体CPI同比增速降至3.2%，市场对美联储结束加息周期的预期增强， 带来了整体市场的反弹。这种宏观利好尤其有助于科技股和成长型公司的上涨，其中特斯拉作为标杆性企业受益显著。" }

下面是你需要分析的案例： 哪些新闻导致了法拉第未来股票在2024-05-20 这个时间大涨这个事实。

|序号|时间|新闻|
|---|---|---|
|1|5月20日|5月20日，法拉第未来（FF）宣布，将于5月29日举办2023年财报业绩沟通会。贾跃亭在社交平台上表示，他 愿意将乐视作为个人IP商业化的重要合作伙伴之一，帮助乐视打造出具有变革性的下一代产品，并助力乐视超级 电视重回巅峰。|
|2|5月15日|5月15日，南京低空经济发展大会举行。会上，南京出台一系列促进支持低空经济高质量发展的实施方案和政策 措施，其中提到，未来三年全市低空经济产业规模超500亿元。|
|3|2月5日|2月5日晚间公告，2024年2月4日，公司子公司万丰飞机与合作方签署了《关于成立合资公司的谅解备忘录》， 双方通过战略合作拟共同成立合资公司，将在电动垂直起降飞行器(eVTOL)领域开展深度合作。|
|4|5月14日|据中国基金报，5月14日，拜登政府宣布，将对中国进口电动汽车征收四倍关税，税率从25%提高到100%。当 天，法拉第未来股价暴涨367%。据悉，法拉第未来是唯一在美国本地发展的中国背景电动车概念股，也可能是 唯一不会受禁令影响的电动车中概股，因此或产生巨大的并购价值想象空间，吸引游资炒作。|

输出格式要求：请以纯文本形式输出 JSON，不要包含任何代码块标记（如 ```json 或 ```）： { "相关新闻序号": [<请填写你的分析结果>], "具体分析": "<请填写你的分析过程>" }

[Figure 31]

{

"相关新闻序号": [1,4], "具体分析": "法拉第未来股票在2024年5月20日大涨，主要受到两条新闻的影响。第一，新闻1提到法拉第未来宣布将于5月29日

举办2023年财报业绩沟通会，并且贾跃亭表示愿意将乐视作为个人IP商业化的重要合作伙伴，助力乐视超级电视重回巅峰。这一消 息提振了市场对法拉第未来业务协同和未来战略方向的信心。第二，新闻4指出拜登政府计划对中国进口电动汽车征收四倍关税，而 法拉第未来作为唯一在美国本地发展的中国背景电动车概念股，不会受到该政策的直接影响，反而凸显其独特的并购价值和竞争优 势，吸引了大量游资炒作，推动股价暴涨。" }

Figure 14: An example instance from the Anomalous Event Attribution.

###### Emotion Recognition

你是一个专业的金融情绪分析助手，你的任务是根据用户提出的问题及其相关的背景信息，来判断用户的当前情绪状态。请从以下几种情绪 类型中选择最符合的一项或多项：

###### [背景信息和用户问句] {"市场环境": {

###### [情绪类型] 一、乐观

"大盘走势分析": {"走势表现": { "三大指数表现": "上证指数微涨0.3%，深证成指上涨0.5%，创业板指上涨

- - 特点：对未来充满信心，相信市场或投资能够带来积极结果。
- - 示例场景：

- 0.7%，市场呈现温和回升态势", "技术指标": "上证指数MA5上穿MA10形成金叉，MACD红柱初现，KDJ三线

向上发散"}, "题材分化": { "指数分化": "中证1000指数领涨0.8%，创业板50指数上涨

- 1.2%，显示成长股表现活跃", "板块热点": "纺织服装板块涨幅居前，多只个股创年内新高"},

- 1. 投资者看到某家科技公司发布了一项突破性技术，认为这将推动股价大幅上 涨。
- 2. 市场经济数据连续几个季度显示强劲增长，投资者预期未来几年股市将持续 上扬。
- 3. 新兴市场国家推出了一系列经济改革政策，投资者对该地区的长期发展前景 感到乐观。
- 4. 某行业正处于快速增长阶段，投资者相信自己的投资组合中相关资产会显著 增值。
- 5. 投资者参加了一场金融论坛，听到多位专家对未来的正面预测后更加坚信市 场潜力。

"市场情绪": { "量能变化": "成交量温和放大至1.2万亿，量价配合良好", "个股涨跌": "上涨家数增至2500家，涨停家数扩大至35家", "主力资金": "北向资金连续三日净流入，单日净买入达50亿元"},

"综合分析": "市场呈现结构性机会，纺织制造等细分领域显现资金聚集效

###### 二、焦虑

应"}}, "用户持股信息": "用户自选股": {

- - 特点：对不确定性感到不安，担心潜在风险或损失。
- - 示例场景：

- 1. 投资者持有的股票价格短期内波动剧烈，导致其频繁查看账户以确认资产状 况。
- 2. 国际政治局势紧张，投资者担心战争或制裁可能引发全球经济衰退。
- 3. 央行宣布即将加息，投资者担忧利率上升会对债券和房地产投资造成负面影 响。
- 4. 某个关键行业的龙头企业突然宣布业绩下滑，投资者对其相关资产的前景感 到忧虑。
- 5. 市场传言某国可能会实施资本管制，投资者担心资金无法顺利撤回。 三、消极

"公司名称": ["嘉麟杰", "中国平安", "招商银行"], "股票代码": ["002486", "601318", "600036"]

###### }, "相关的技术面分析": {

"嘉麟杰": { "趋势分析": {"主要趋势": "周线级别突破下降通道，月线MACD即将金叉",

"趋势强度和持续性": "布林带开口扩大，ADX指标升至35显示趋势强化"}, "撑压分析": {"支撑位": "2.80元平台支撑坚实",

"价格区间": "量能配合下有望挑战3.50元前高"}, "K线形态分析": {"技术形态": "杯柄形态构筑完成",

- - 特点：对市场或投资结果产生失望、沮丧或悲观的情绪。
- - 示例场景：

"K线形态": "连续三日放量阳线突破整理平台"

- 1. 投资者长期持有的一只股票因公司丑闻而暴跌，导致其亏损严重。
- 2. 全球经济进入衰退周期，投资者看到自己的投资组合价值持续缩水。
- 3. 某行业受到政策监管加强的影响，投资者对该行业的未来失去信心。
- 4. 投资者尝试了多种策略但均未取得理想收益，开始怀疑自己的投资能力。
- 5. 市场长期处于低迷状态，投资者对任何新的投资机会都提不起兴趣。 四、兴奋

}, "量价分析": {

"成交量和价格": "量能阶梯式放大，OBV能量潮创年内新高", "市场情绪": "融资余额单周增长15%，市场关注度显著提升"

}, "均线分析": {

- - 特点：因市场上涨或投资成功而感到愉悦和满足。
- - 示例场景：

"均线趋向": "5周均线上穿60周均线形成黄金交叉", "均线离散": "均线系统呈多头排列"

- 1. 投资者买入的一只股票因公司财报超预期而涨停，当天收益达到两位数。
- 2. 某热门板块突然成为市场焦点，投资者持有的相关资产迅速升值。
- 3. 投资者通过精准判断抓住了一次短期交易机会，获得超额回报。
- 4. 市场因利好消息出现普涨行情，投资者发现自己几乎所有资产都在增值。
- 5. 投资者参与的新股申购中签，并在上市首日获得了高额收益。 五、冷静

}, "技术指标": {

"MACD": "日线级别二次翻红，周线级别绿柱缩短", "KDJ": "J值维持在80以上超买区域达5个交易日"

} }

- - 特点：保持中立和理性，不受市场波动或情绪干扰。
- - 示例场景：

###### }, "相关的最新新闻": [

- 1. 市场因突发新闻出现短暂下跌，但投资者并未恐慌抛售，而是选择继续观 察。
- 2. 投资者在市场剧烈波动时坚持既定的投资计划，不轻易调整策略。
- 3. 面对市场热点板块的快速上涨，投资者并未盲目追高，而是等待更好的入场 时机。
- 4. 投资者在经济数据不佳时仍保持信心，认为这只是短期现象而非趋势性变 化。
- 5. 即使市场出现重大事件，投资者也能基于数据分析做出客观判断，避免情绪 化决策。

{

"日期": "2025-01-02", "标题": "嘉麟杰上调回购价格上限至5.2元彰显信心", "摘要": "公司宣布将股份回购价格上限大幅提升32%，彰显管理层对企业未来

发展强烈信心" }, {

"日期": "2024-12-30", "标题": "纺织行业景气度持续回升", "摘要": "海关总署数据显示12月纺织品出口同比大增18%，行业迎来量价齐升

###### 六、后悔

- - 特点：对过去的决策感到遗憾或自责，尤其是当结果不如预期时。
- - 示例场景：

周期" }

- 1. 投资者因犹豫不决错过了某只股票的最佳买入时机，事后发现该股票大幅上 涨。
- 2. 投资者在市场高位时卖出资产，随后市场继续上涨，导致错失更多收益。
- 3. 投资者未能及时止损，导致原本的小幅亏损变成重大损失。
- 4. 投资者听信他人建议进行了一次高风险投资，最终失败并造成巨大损失。
- 5. 投资者在市场下跌时被迫平仓，事后发现市场很快反弹，导致错失恢复机 会。

], "相关的最新研报": [], "用户问句": "嘉麟杰这个量价配合，是不是有可能走出一波主升浪？"

}

请严格按照以下JSON格式输出你的分析结果：

{ "cot": "<请在此处填写你的推理过程或判断依据，描述清楚得出结论的理由>", "result": ["<请填写你的情绪分类结果1>","<请填写你的情绪分类 结果2>"] }

[Figure 32]

{

"cot": "用户当前关注嘉麟杰的量价配合情况，从技术面分析、量能放大和突破形态上推测其可能进入主升浪阶段。结合嘉麟杰近期的回 购消息、行业景气度提升以及用户已经持有该股票并浮盈8%的情况，用户对该股票的未来走势可能持乐观态度。此外，从用户积极提问 并寻找市场机会的行为来看，其情绪可能受到一定程度的兴奋驱动。",

"result": ["乐观", "兴奋"] }

- Figure 15: An example instance from the Emotion Recognition.

[Figure 33]

##### Figure 16: An example instance from the Stock Price Prediction.

Question: 陈先生将100000元存入银行，年利率为1.5%，2年后，他将获得多少元利息？A: 3000 B: 23173 C: 27754 D: 10943 Answer: A

(A) An example from the numerical calculation dataset in Fin-Eva Version 1.0

Question: 你是一位经验丰富的金融数据分析师，请根据以下<数据参考>，回答出<用户问题>，并给出你的理由。你需要以指定 的<输出格式>回答。 ### 数据参考

### 用户问题 question: 在2024年8月至2024年9月期

- 编号: 1

- 取数问句: 百度2024年8月每日的收盘价

编号: 2

- 取数问句: 百度2024年9月每日的收盘价

间，百度的收盘价大于90.00美元的天数 有几天？

|股票代码|股票简称|收盘价 20240801|……|收盘价 20240830|
|---|---|---|---|---|
|BIDU.O|百度|86.42美元|……|84.62美元|

### 输出格式

......

......

Answer: 6

(B) An example from the numerical calculation dataset in BizFinBench

Figure 17: An Chinese Version of the Comparison of Numerical Calculation Questions in FinEva Team [2023] and BizFinBench

