# arXiv:2601.06401v1[cs.AI]10Jan2026

[Figure 1]

## BizFinBench.v2: A Unified Dual-Mode Bilingual Benchmark for Expert-Level Financial Capability Alignment

Xin Guo1,2,∗, Rongjunchen Zhang1,∗,♠, Guilong Lu1, Xuntao Guo1, Shuai Jia1, Zhi Yang2, Liwen Zhang2,♠ 1HiThink Research, 2Shanghai University of Finance and Economics zhangrongjunchen@myhexin.com, zhang.liwen@shufe.edu.cn

### Abstract

[Figure 2]

Large language models have undergone rapid evolution, emerging as a pivotal technology for intelligence in financial operations. However, existing benchmarks are often constrained by pitfalls such as reliance on simulated or generalpurpose samples and a focus on singular, offline static scenarios. Consequently, they fail to align with the requirements for authenticity and real-time responsiveness in financial services, leading to a significant discrepancy between benchmark performance and actual operational efficacy. To address this, we introduce BizFinBench.v2, the first large-scale evaluation benchmark grounded in authentic business data from both Chinese and U.S. equity markets, integrating online assessment. We performed clustering analysis on authentic user queries from financial platforms, resulting in eight fundamental tasks and two online tasks across four core business scenarios, totaling 29,578 expert-level Q&A pairs. Experimental results demonstrate that ChatGPT-5 achieves a prominent 61.5% accuracy in main tasks, though a substantial gap relative to financial experts persists; in online tasks, DeepSeek-R1 outperforms all other commercial LLMs. Error analysis further identifies the specific capability deficiencies of existing models within practical financial business contexts. BizFinBench.v2 transcends the limitations of current benchmarks, achieving a business-level deconstruction of LLM financial capabilities and providing a precise basis for evaluating efficacy in the widespread deployment of LLMs within the financial domain. The data and code are available at https://github.com/ HiThink-Research/BizFinBench.v2.

Figure 1: BizFinBench.v2 comprises eight foundational tasks and two online tasks distributed across four major scenarios. The top-right corner displays a real-time screenshot of the Portfolio Asset Allocation task.

aries in the financial sector continue to expand, making them an important technological direction for promoting the intelligent upgrade of financial services (Chen et al., 2024; Liu et al., 2025a; Zhang et al., 2023b; Lu et al., 2024). However, as the scope of LLMs’ application in the financial sector becomes increasingly broad, the discrepancy between model performance under existing evaluation paradigms and their actual performance in real financial business scenarios has become increasingly prominent, directly leading to a conflict between real financial business needs and the limitations of various evaluation benchmarks.

### 1 Introduction

The core characteristics of financial scenarios lie in authenticity and online capability, yet the vast majority of current financial benchmarks are undermined by two fundamental limitations:

Large Language Models (LLMs) have developed rapidly in recent years, and their application bound-

*These authors contributed equally to this work.

(1) Disconnected from Real-world Business

♠Corresponding authors.

[Figure 3]

- Figure 2: We have ranked the performance of the LLMs participating in the evaluation under the zero-shot setting, and these results reflect their authentic practical business capabilities.

Scenarios: Most existing studies rely on simulated samples or general datasets that are disconnected from actual business operations. These datasets generally exhibit low difficulty and are decoupled from the requirements of real-world financial business, as they lack the core logic of business workflows and fail to reflect the practical challenges inherent in the financial domain. (Zhu et al., 2024; Jiang et al., 2025).

(2) Focusing on Purely Static Offline Tasks: Existing benchmarks universally focus on purely offline static tasks (Xie et al., 2024), lacking coverage of online tasks in financial scenarios. This prevents the support for evaluating model performance in online business scenarios such as real-time market data push and dynamic risk monitoring, thus being disconnected from the core requirement of online capability in financial services (Tang et al., 2025). These two problems result in a disconnect between existing benchmarks and real-world applications, leading to a significant gap between LLM benchmark test performance and actual business performance.

To bridge the gap between current LLM evaluation performance and actual operational efficacy, we introduce BizFinBench.v2, the first large-scale financial business evaluation benchmark that integrates authentic business data from both Chinese and U.S. equity markets and features a dual-track evaluation of "Core Business Capabilities + Online Performance." BizFinBench.v2 comprises 29,578 QA pairs based on real business data. Through

the clustering of massive user queries from authentic business platforms and rigorous quality control, we have categorized four core business scenarios: Business Information Provenance, Financial Logic Reasoning, Stakeholder Feature Perception, and Real-time Market Discernment. These encompass eight fundamental tasks, including Anomaly Information Tracing, Financial Multi-turn Perception, Financial Data Description, Financial Quantitative Computation, Event Logic Reasoning, Counterfactual Inference, User Sentiment Analysis and Financial Report Analysis, along with two online tasks: Stock Price Prediction and Portfolio Asset Allocation. The detailed architecture is illustrated in Figure 1.

Unlike existing benchmarks, BizFinBench.v2 not only enables a precise and objective quantitative evaluation of the actual efficacy of LLMs in authentic financial scenarios but also identifies core capability shortcomings in financial business applications. It achieves a comprehensive deconstruction of LLM capabilities from a business perspective, thereby providing a reliable performance reference for the deployment of LLMs in the financial domain.

Our contributions are threefold:

(1) Authentic Business Data Driven: We propose the first large-scale financial evaluation benchmark driven by authentic data. Developed upon a core foundation of real-world business data from Chinese and U.S. equity markets, BizFinBench.v2 transcends the limitations inherent in existing eval-

uation frameworks, specifically their disconnetion from practical business scenarios and their reliance on low-complexity, general-purpose, or synthetic data.

- (2) Innovative Dual-Track Evaluation Framework: We propose a dual-track evaluation framework encompassing "Core Business Capabilities + Online Performance," thereby addressing the deficiencies in application scenario coverage inherent in traditional offline static assessments.
- (3) Expert-Informed Business Error Analysis: We provide an error analysis of the actual performance of LLMs from the business perspective of financial experts, offering targeted optimization directions for broader business adaptation.

The organization of this paper is as follows: Section 2 deliberates on the current status of various operations and benchmarks in the financial domain, identifying the corresponding requirements and the limitations of existing research. Section 3 delineates the construction, quality control, and detailed statistics of BizFinBench.v2. Sections 4 and Section 5 primarily describe the experimental setup, as well as the main experimental results and analysis. Section 6 presents an error analysis conducted from an expert business perspective. Finally, Section 7 concludes the study and proposes directions for future research.

### 2 Related Work

#### 2.1 Financial Business Analysis

In financial business scenarios, LLM-based tasks such as text understanding (Wilson et al., 2024), sentiment analysis (Delgadillo et al., 2024; Zhang et al., 2023a), time-series forecasting (Li et al., 2024a,b; Wang et al., 2024; Mai, 2024), and wealth management & investment (Yu et al., 2024; Wang et al., 2025; Yu et al., 2025; Yang et al., 2023) serve as foundational components for building universal financial business capabilities. They provide crucial technical support for judging market trends and mitigating potential risks. However, the financial sector’s demand for timeliness means standalone offline tasks cannot fully evaluate LLMs’ performance in real-world business contexts—especially scenarios closely tied to dynamic financial markets, such as stock price fluctuation prediction (Yu and Yan, 2020), real-time response to market hot topics and public opinion (Mao et al., 2025), intraday trading signal interpretation (Cervelló-Royo et al., 2015), and instant investment decisions (Suresh,

2024). Nevertheless, most current research on LLMs in finance focuses on enhancing offline question-answering quality and optimizing reasoning accuracy, verifying model performance through static datasets (Liu et al., 2025b; Zhu et al., 2025). Few studies test LLMs’ capabilities in processing real-time market data and generating instantaneous decisions from the perspective of financial businesses’ online and real-time needs. This gap results in a mismatch between LLM performance and the core demands of financial markets, leaving significant room for future exploration.

#### 2.2 Financial Benchmark Analysis

Mainstream evaluation benchmarks in the current financial domain have achieved multi-dimensional coverage for assessing the foundational capabilities of LLMs. Benchmarks such as CFLUE (Zhu et al., 2024), FinEval (Guo et al., 2025b), and CGCE (Zhang et al., 2023c) focus on basic financial knowledge and text comprehension, encompassing tasks like text classification and policy interpretation. Others including FinQA (Chen et al., 2021), ConvFinQA (Chen et al., 2022), FinanceReasoning (Tang et al., 2025), and FinBen (Xie et al., 2024) emphasize financial numerical reasoning and complex decision-making, simulating quantitative analysis scenarios such as financial report interpretation. Meanwhile, benchmarks like DISC-FINSFT (Chen et al., 2023), PIXIU (Xie et al., 2023), FinMaster (Jiang et al., 2025) and CFBenchmark (Lei et al., 2023) prioritize instruction following and tool adaptation. However, these benchmarks are generally focused on verifying performance in single, offline, and static tasks, failing to accommodate the real-time demands inherent in financial services. Furthermore, the construction of the relevant data is mostly derived from simulated synthesis or manual rewriting, making it difficult to align with real-world financial business requirements. Simultaneously, there is a lack of in-depth discussion on the consistency between evaluated performance and the LLM’s actual financial application capability, leading to a divergence between the practical guidance provided by the benchmarks and the core demands of the financial market. Detailed comparison with existing benchmarks can be found in Table 1.

- Table 1: In-depth multi-dimensional comparison between BizFinBench.v2 and existing benchmarks from a real-business perspective. The abbreviations and their core meanings are as follows: Cross-Lingual (CL), Online Testing (OT), Real Business Data (RBD).

Benchmarks CL OT RBD

FinEval (Guo et al., 2025b) × × × CFLUE (Zhu et al., 2024) ✓ × × FinQA (Chen et al., 2021) × × × ConvFinQA (Chen et al., 2022) × × × FinMaster (Jiang et al., 2025) × × × FinBen (Xie et al., 2024) ✓ × × FinanceMath (Zhao et al., 2024) × × × FinanceReasoning (Tang et al., 2025) × × ×

BizFinBench.v2 ✓ ✓ ✓

### 3 BizFinBench.v2 Benchmark

#### 3.1 Overview

As the first large-scale financial evaluation benchmark to cover authentic business data from both Chinese and U.S. equity markets while integrating offline evaluation and online testing, BizFinBench.v2 aims to bridge the gap between the performance of LLMs in the financial domain and their actual business capabilities. All data within the benchmark is derived from real financial business platforms and has undergone rigorous compliance review and quality control; the dataset will be released as open-source for research purposes in the future.

[Figure 4]

- Figure 3: The platform’s user structure is primarily analyzed through user type, device distribution, and national distribution; here, Inst. serves as an abbreviation for institutional users.

#### 3.2 Task Construction

In the process of task construction, BizFinBench.v2 anchors itself in the core functions of authentic financial business platforms and is guided by the

actual demands of key market participants. It covers the diverse needs of individual users, including novices, short-term investors, and high-networth individuals, while accounting for the critical requirements of mainstream financial institutions such as banks and securities firms. The platform user distribution is shown in Figure 3.

Business Information Provenance (BIP) focuses on frontline scenarios, requiring LLMs to resolve various business inquiries through direct interaction. This includes real-world perturbations such as erroneous input, irrelevant information, or typos, necessitating robust discernment capabilities from the models. Specifically, the Anomaly Information Tracing (AIT) task addresses user inquiries regarding the root causes of market anomalies, such as significant fluctuations in watchlist stocks; it requires LLMs to filter key clues from massive multidimensional information, exclude interference, and precisely identify the core events driving stock or market volatility. The Financial Multi-turn Perception (FMP) task requires LLMs to respond to continuous inquiries within long-context interactions, integrating historical user queries to provide sustained service. The Financial Data Description (FDD) task demands that LLMs judge the authenticity and accuracy of specific data within various information sources to provide precise conclusions.

The Financial Logical Reasoning (FLR) scenario serves as both a foundational element of Business Information Provenance, requiring LLMs to perform rigorous financial reasoning based on market data returned by various interfaces. Within this category, the Financial Quantitative Computation (FQC) task requires LLMs complete accurate indicator calculations based on complex financial data. The Event Logical Reasoning (ELR) task requires LLMs to sort macro or micro market events based on chronological order or causal logic. The Counterfactual Inference (CI) task requires LLMs to simulate financial experts by conducting reasonable logical analysis and judgment based on user-proposed hypotheses, outputting reliable and valid conclusions.

The Stakeholder Feature Perception (SFP) scenario aims to provide users with in-depth analysis and summaries of the market or industry. The User Sentiment Analysis (SA) task requires LLMs to classify and evaluate overall user sentiment through the quantitative analysis of various user information and behaviors, providing support for customized services and highly relevant product

recommendations. The Financial Report Analysis (FRA) task requires LLMs to conduct quantitative analysis by combining technical, fundamental, and news-based periodic report information to determine enterprise industry rankings, helping users intuitively grasp corporate qualifications and future prospects. This task is limited to the Chinese market.

Finally, the Real-time Market Discernment (RMD) scenario covers two types of online tasks of high interest to users: Stock Price Prediction (SPP) and Portfolio Asset Allocation (PAA). It should be noted that the real-time nature of online tasks requires all relevant data to be derived from the actual equity market, and theoretically, the number of questions is unlimited. However, due to the static nature of a research paper, the cutoff time for all tasks is set to December 24, 2025. The Stock Price Prediction task addresses user needs for predicting stocks in their watchlists, requiring LLMs to predict daily closing prices by integrating historical and real-time stock prices, technical indicators, and relevant news. The Portfolio Asset Allocation task presents a higher degree of difficulty. We have constructed an LLM investment simulation system for this task that restores real-world trading rules, including transaction fees, latency, and slippage, on top of real-time data access. Through systematic prompt engineering and various data and tool interfaces, LLMs can perform asset allocation in a real market environment using simulated funds. Decisions are generated hourly during the day, allowing for either trading or skipping, with each decision cycle covering the full process from reasoning and analysis to order submission and trade execution. We will subsequently open-source this LLM investment system to allow all researchers interested in the financial field to integrate different open-source or proprietary LLMs for in-depth study.

For a more detailed introduction to the above content, please refer to Appendix A.

#### 3.3 Quality Control

For tasks in offline tasks, the quality control process strictly adopts a three-level progressive mechanism of "Platform Clustering and Desensitization - Frontline Staff Review - Expert Team Cross-Validation" to ensure the validity and compliance of the evaluation data. First, all raw data undergo clustering through the platform’s multi-dimensional algorithms to automatically aggregate and categorize different task types. Simultaneously, data desensiti-

zation is performed (including user PII or sensitive corporate data), and a preliminary review is conducted in alignment with financial industry compliance requirements to eliminate content that clearly violates regulatory standards. Second, ten frontline business professionals with over five years of experience manually screen the data samples one by one to filter out invalid Q&A pairs, duplicate records, and samples with abnormal expressions, while retaining specific real-world perturbations to ensure each Q&A pair fully reflects authentic business scenarios. Finally, we invited six senior financial experts, each with over a decade of industry experience from three different professional teams within the Financial Research Institute, to conduct crossvalidation in three groups. Ultimately, through this rigorous three-level screening mechanism, highquality Q&A pairs are formed that balance authenticity, business alignment, and information security.

As for online tasks, the financial experts mentioned above strictly defined the structured data required for the Stock Price Prediction task, the specific equity market configurations for the Portfolio Asset Allocation task, and the system prompts, ensuring full compliance with the requirements of authentic business scenarios.

It should be noted that for offline tasks, due to the protection of financial business privacy, the specific evaluation rubrics and quantitative standards employed during the quality review process cannot be disclosed to the public. In contrast, for online tasks, as real-time public equity market data is utilized and the configurations are designed to replicate the actual market environment, the details of these settings and the system prompts will be made open-source.

#### 3.4 Statistics

BizFinBench.v2 covers the two major equity markets (China and the U.S.), with all data drawn from authentic user demands within financial business scenarios. This has resulted in four core scenarios: BIP, FLR, SFP and RMD. These comprise eight offline tasks and two online tasks, totaling 29,578 questions derived from real-world financial service requests. Specifically, the AIT, ELR, SA, and CI tasks consist of 4,000 questions each, with average input token counts of 8,679, 437, 3,326, and 5,510, respectively. FQC and FRA each contain 2,000 questions, with average input token counts of 1,984 and 19,681. FMP and FDD consist of 3,741 and 3,837 questions, with average input token counts

of 10,361 and 3,577, respectively. The SPP task, based on 100 popular stocks, accumulated a total of 4,000 questions with an average input token count of 5,510. Regarding the PAA task, due to its unique nature as an online task, we do not include the number of transactions in the dataset; instead, we evaluate the actual performance of the LLMs by comparing final metrics based on cumulative results. More intuitive statistical results can be found in Table 5 within Appendix A.

### 4 Experiment Setting

#### 4.1 Evaluated Methods

All questions in BizFinBench.v2 are verifiable open-ended questions. Except for the Portfolio Asset Allocation task, all other tasks utilize Accuracy as the primary metric. We provide additional clarifications here: (1) For the User Sentiment Analysis and Stock Price Prediction tasks, we draw upon the concept of conformal prediction algorithms (Shafer and Vovk, 2008) used in machine learning. Models are required to construct prediction intervals based on preset business tolerance levels; a prediction is judged valid by checking whether the standard answer falls within the model-provided interval. Relevant results are labeled in the interval_boxed{} format for verification. Here, the business tolerance for the sentiment analysis task is set at 10%, and for the stock price prediction task, it is set at 1%. (2) Metrics for the Portfolio Asset Allocation task primarily consist of practical business indicators such as cumulative return, Sharpe ratio, and maximum drawdown; therefore, the models evaluated for this task (mainly six proprietary commercial models) and their results are independent of other tasks. In the experimental results section, we present the average ranking of models across different metrics as the final result. Given the variations in input length across different tasks and the constraints of model context windows, the primary evaluation method is standardized as zero-shot. For further details regarding experimental settings, please refer to Appendix B.1.

#### 4.2 Evaluated Models

We evaluated 21 LLMs, where proprietary models were accessed via their respective APIs and opensource models were deployed locally. All inference tasks were executed on an 8×NVIDIA H100 GPU cluster. For additional details regarding the models, refer to Appendix B.2. We additionally invited

two financial experts who were not involved in the data construction process, to participate in the task evaluation of BizFinBench.v2 to provide a performance comparison with the LLMs.

In the realm of proprietary LLMs, we evaluated 6 leading, high- performance models, including ChatGPT-5 (OpenAI, 2023), Gemini-3 (Team et al., 2023), Claude-Sonnet-4 (Anthropic, 2025), Grok-

- 4 (xAI, 2025), Doubao-Seed-1.6 (Bytedance, 2025) and Kimi-k2 (Moonshot, 2025). For open-source models, we evaluated 11 models from several mainstream LLMs, including Qwen2.5-7B/72BInstruct, Qwen3-32B and Qwen3-235B-A22BThinking-2507 (Bai et al., 2023), InternLM2.57B/20B (Team, 2023), GLM-Z1-9B/32B (GLM et al., 2024), DeepSeek-V3.2 and Deepseek-R1Distill-Qwen-7B/32B (Guo et al., 2025a). In financial domain, we evaluated 4 representative LLMs tailored for the financial tasks, including Fin-R1 (Liu et al., 2025b), Dianjin-R1 (Zhu et al., 2025), FinX1 (Team, 2024) and Fino1 (Qian et al., 2025). In the PAA task, we have included Qwen3Max as an additional model for evaluation.
- 5 Main Results

The complete model ranking results are shown in Figure 2. Table 2 presents the quantitative performance of all evaluated models across various tasks. From an overall performance perspective, ChatGPT-5 ranks first among all participating models with an average accuracy of 61.5%, highlighting its comprehensive competitive advantage in financial scenarios. Furthermore, proprietary models such as Gemini-3 and Doubao-Seed-1.6 also demonstrate excellent performance, ranking within the top three across multiple tasks. Within the category of open-source models, Qwen3-235B-A22B-Thinking-2507 emerges as the top-performing model with an average accuracy of 53.3%. In contrast, the leading financial model, Dianjin-R1, achieves an average accuracy of only 35.7%, trailing Qwen3-32B by a margin of 5.6%. We analyze this discrepancy from two perspectives: first, the training data for financial-specific models is primarily derived from open-source financial datasets centered on financial knowledge and simulated scenarios, which fail to map the complex characteristics of real-world financial business environments; second, although Dianjin-R1 incorporates customer service Q&A data, its business coverage remains narrow, mak-

- Table 2: LLM performance in zero-shot setting on BizFinBench.v2 (%). The results are color-coded to indicate the top three performers in each task: gold represents the best-performing model, silver represents the second-best result, and bronze represents the third-best performance.. The meanings of the abbreviations are as follows: AIT(Anomaly Information Tracing), FMP(Financial Multi-turn Perception), FDD(Financial Data Description), FQC(Financial Quantitative Computation), ELR(Event Logic Reasoning), CI(Counterfactual Inference), SA(User Sentiment Analysis), FRA(Financial Report Analysis), SPP(Stock Price Prediction).

Model Size AIT FMP FDD FQC ELR CI SA FRA SPP Average Propretary LLMs

ChatGPT-5 unknown 54.2 90.8 68.3 89.2 62.0 83.9 18.8 54.1 32.1 61.5 Gemini-3 unknown 64.8 87.0 69.7 85.8 69.5 82.2 7.4 50.8 34.9 61.3 Doubao-Seed-1.6 unknown 62.6 90.2 63.8 78.2 61.2 78.7 22.7 46.3 31.1 59.4 Kimi-k2 unknown 55.2 80.9 22.4 62.2 44.6 15.4 20.1 45.0 30.5 41.8 Claude-Sonnet-4 unknown 54.8 79.9 28.4 29.8 44.8 23.4 17.3 47.7 29.1 39.5 Grok-4 unknown 61.8 86.6 37.4 9.3 42.1 4.7 17.8 45.2 30.3 37.2

Open-source General LLMs

Qwen3-235B-A22B-Thinking-2507 235B 49.3 87.9 68.0 76.0 50.6 72.2 16.4 22.5 36.9 53.3 DeepSeek-V3.2 685B 60.2 87.4 42.3 22.0 49.4 8.3 23.5 50.2 32.8 41.7 Qwen3-32B 32B 54.3 80.9 40.0 48.4 42.0 47.2 13.4 40.5 5.1 41.3 Qwen2.5-72B-Instruct 72B 61.0 78.2 26.5 19.8 39.9 20.7 21.6 47.0 8.2 35.9 GLM-Z1-32B 32B 49.6 66.9 45.6 34.4 31.6 31.3 1.2 36.8 6.8 33.8 GLM-Z1-9B 9B 46.5 69.2 40.4 26.9 35.2 25.2 0.4 36.7 3.4 31.5 DeepSeek-R1-Distill-Qwen-32B 32B 52.1 75.4 20.5 21.0 35.9 24.2 18.4 27.3 6.8 31.3 Qwen2.5-7B-Instruct 7B 35.2 43.2 33.8 0.8 20.9 1.0 23.3 28.3 2.8 21.0 InternLM2.5-20B 20B 32.1 41.1 32.0 0.2 27.8 0.7 3.7 32.5 0.6 19.0 InternLM2.5-7B 7B 28.3 14.8 30.6 0.5 18.9 0.3 6.7 28.5 0.0 14.3 DeepSeek-R1-Distill-Qwen-7B 7B 16.6 30.9 17.1 3.5 10.3 6.2 19.1 13.0 1.0 13.1

Open-source Financial LLMs

Dianjin-R1 32B 54.2 70.7 40.9 25.3 45.8 22.0 6.7 46.0 9.9 35.7 FinX1 70B 47.9 73.0 29.5 14.3 31.5 11.6 5.3 34.7 3.6 27.9 Fino1 14B 27.5 38.6 24.3 14.1 14.6 22.1 11.1 35.3 8.6 21.8 Fin-R1 7B 21.2 29.1 0.5 1.5 9.8 3.2 10.9 24.5 2.9 11.5

Financial Experts – 92.6 98.0 94.5 100 91.7 100 57.9 96.0 32.3 84.8

- Table 3: Comparison of key metrics in Portfolio Asset Allocation tasks. The primary indicators include Total Return (TR), Profit Factor (PF), Sharpe Ratio (SR), Maximum Drawdown (MD), and Total Assets (TA).

SA task reflect the difficulty of models in executing subjective analysis tasks; even the top-performing DeepSeek-V3.2 achieved an average accuracy of only 23.5%, which is far below that of financial experts. Furthermore, in the SPP task, the bestperforming Qwen3-235B-A22B-2507 reached an accuracy of only 36.9%, confirming that current LLMs are still unable to meet the high demands of online tasks for precise stock movement prediction based on multi-dimensional information. Table 3 presents the key metrics of six mainstream proprietary models in the asset allocation task, revealing significant differences in their real-world investment capabilities. DeepSeek-R1 demonstrates the most outstanding performance, leading in metrics such as total return, profit-loss ratio, and Sharpe ratio, while maintaining a maximum drawdown of -8%, reflecting its superior risk-reward balancing capability. In contrast, ChatGPT-5 and Claude Sonnet-4 failed to outperform the market benchmark (SPY); specifically, ChatGPT-5 experienced a slight loss, with both its profit-loss ratio and Sharpe ratio falling below 1, suggesting that its strategy ad-

Model TR PF SR MD TA

DeepSeek-R1 +13.46% 2.3 1.8 -8% 11346 Qwen3-Max +8.43% 1.8 1.4 -10% 10843 Grok-4 +7.97% 1.4 1.1 -1% 10797 Gemini-3 +5.88% 1.4 1.1 -6% 10588 Claude-Sonnet-4 +0.10% 1.1 0.8 -12% 10010 ChatGPT-5 -0.92% 0.9 0.5 -6% 9908 SPY – – – – 10474

ing it difficult to adapt to more volatile and longcontext practical business scenarios.

From a task perspective, models with larger parameter scales perform better on tasks that demand high data precision, such as FDD, FQC, and CI. Notably, the proprietary models Claude-Sonnet-4 and Grok-4 show a significant gap compared to other proprietary models in these tasks, indicating that their computational capabilities require further enhancement to adapt to the application requirements of the financial domain. The results of the

justment capabilities need improvement. Notably, although Grok-4’s return is moderate, its maximum drawdown is only -1%, demonstrating exceptional risk control.

Viewed comprehensively, in foundational capability testing, LLMs remain far below the level of financial experts (84.8%), with a significant gap existing between the two; however, in online tasks, mainstream commercial proprietary models have demonstrated a certain degree of potential. This implies that there is still substantial room for improvement regarding the adaptation capabilities and practical potential of current LLMs in real-world financial business scenarios. Subsequent optimization efforts should focus on dimensions such as training data quality and scenario adaptability.

[Figure 5]

- Figure 4: We selected representative LLMs for error analysis and summarized five typical dilemmas faced by LLMs in actual business scenarios

### 6 Error Analysis

To further investigate the performance deficiencies of LLMs in financial business scenarios, we conducted a targeted analysis on a random sample of 20% of the incorrect responses from all evaluated models. By deconstructing model issues through the lens of business practice, we have identified five typical business dilemmas prevalent across current LLMs in financial tasks: Financial Semantic Deviation, Long-term Business Logic Discontinuity, Multivariate Integrated Analysis Deviation (MIAD), High-precision Computational Distortion, and Financial Time-Series Logical Disorder. According to the distribution of error types in Figure 4, the proportion of error types for the vast majority of models exhibits similar characteristics, reflecting the common challenges currently faced by LLMs in financial business adaptation; how-

ever, certain models also show significant differences. Specifically, ChatGPT, Gemini-3, DoubaoSeed-1.6, and Qwen3-235B-A22B-Thinking have the highest proportion of errors in MIAD, indicating that these models possess distinct performance bottlenecks in information integration and analysis that require targeted optimization. Furthermore, higher error proportions suggest that Grok-4 and Claude-Sonnet-4 need to improve their highprecision computational capabilities, while other models with relatively smaller parameter scales require full-process business optimization to better adapt to relevant financial scenarios. For specific error examples and detailed descriptions of each business dilemma, please refer to Appendix D.

### 7 Conclusion

In this study, we innovatively present BizFinBench.v2, the first evaluation benchmark constructed based on authentic user data from realworld platforms. By encompassing offline and online tasks across Chinese and U.S. equity markets, it effectively addresses the dual limitations of existing financial LLM research and provides a scientific evaluation framework tailored to practical business applications. We conducted a comprehensive multi-dimensional assessment and quantitative analysis of 21 mainstream open-source and propretary LLMs, with primary results indicating that ChatGPT-5 achieved the best overall performance, demonstrating superior practical capabilities across various financial tasks. Qwen3235B-A22B-Thinking-2507 ranked among the top of open-source models, though an 8.2% performance gap remains compared to ChatGPT-5, while Dianjin-R1 led among domain-specific financial models despite a more pronounced gap behind the top two. In the Portfolio Asset Allocation tasks, DeepSeek-R1 stood out among all evaluated commercial models by achieving a 13.46% total return while maintaining a relatively favorable maximum drawdown of -8%, showcasing an excellent balance between profitability and risk control. Furthermore, through deep error analysis of model outputs, we distilled five typical dilemmas faced by LLMs in financial business scenarios, which reflect the unique challenges and value of BizFinBench.v2 in assessing business adaptability. As a financial benchmark born from real user data, BizFinBench.v2 provides an objective and reproducible quantitative standard for measuring the real-world business performance

of LLMs, and we expect it to drive the targeted optimization and iterative upgrading of LLMs in the financial domain, providing critical technical support for the large-scale and reliable deployment of LLMs in financial settings.

### Limitations

While BizFinBench.v2 has achieved significant progress in the field of LLM evaluation driven by real-world business data, its application still has three limitations that subsequent research can further improve: first, in terms of data coverage, although the Q&A pairs in BizFinBench.v2 are primarily derived from user data of real business platforms, the core acquisition method relies on platform data clustering, which may result in the exclusion of niche user query types with low proportions, necessitating a further expansion of data source scope in the future; second, regarding the dimension of task design, the current online evaluation tasks only focus on the two scenarios with the highest user demand—stock price prediction and asset allocation tasks directly related to investment—meaning subsequent research could extend the evaluation scenarios to a broader range of online financial business links, such as personalized stock recommendations and account opening guidance; and finally, at the level of evaluation paradigms, since the Q&A data of BizFinBench.v2 are all derived from real platforms and the input text lengths are generally long, the current evaluation mainly concentrates on two paradigms, ZeroShot evaluation and Chain-of-Thought evaluation, requiring the addition of Few-Shot evaluation experiments in the future.

### Acknowledgments

Special thanks to Ning Zhang, Siqi Wei, Kai Xiong, Kun Chen and colleagues at HiThink Research’s data team for their support in building BizFinBench.v2.

### References

Anthropic. 2025. Claude4.5. https://www. anthropic.com/.

Jinze Bai, Shuai Bai, and et al. Yunfei Chu. 2023. Qwen

technical report. arXiv preprint arXiv:2309.16609. Bytedance. 2025. doubao. https://seed.bytedance.

com.

Roberto Cervelló-Royo, Francisco Guijarro, and Karolina Michniuk. 2015. Stock market trading rule based on pattern recognition and technical analysis: Forecasting the djia index with intraday data. Expert systems with Applications, 42(14):5963–5975.

Wei Chen, Qiushi Wang, Zefei Long, Xianyin Zhang, Zhongtian Lu, Bingxuan Li, Siyuan Wang, Jiarong Xu, Xiang Bai, Xuanjing Huang, and 1 others. 2023. Disc-finllm: A chinese financial large language model based on multiple experts fine-tuning. arXiv preprint arXiv:2310.15205.

Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan R Routledge, and 1 others. 2021. Finqa: A dataset of numerical reasoning over financial data. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3697–3711.

Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and William Yang Wang. 2022. Convfinqa: Exploring the chain of numerical reasoning in conversational finance question answering. arXiv preprint arXiv:2210.03849.

Zhiyu Zoey Chen, Jing Ma, Xinlu Zhang, Nan Hao, An Yan, Armineh Nourbakhsh, Xianjun Yang, Julian McAuley, Linda Petzold, and William Yang Wang. 2024. A survey on large language models for critical societal domains: Finance, healthcare, and law. arXiv preprint arXiv:2405.01769.

Josiel Delgadillo, Johnson Kinyua, and Charles Mutigwe. 2024. Finsosent: Advancing financial market sentiment analysis through pretrained large language models. Big Data and Cognitive Computing, 8(8):87.

Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, Juanzi Li, and 37 others. 2024. Chatglm: A family of large language models from glm-130b to glm-4 all tools. Preprint, arXiv:2406.12793.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025a. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Xin Guo, Haotian Xia, Zhaowei Liu, Hanyang Cao, Zhi Yang, Zhiqiang Liu, Sizhe Wang, Jinyi Niu, Chuqi Wang, Yanhui Wang, and 1 others. 2025b. Fineval: A chinese financial domain knowledge evaluation benchmark for large language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6258–6292.

Junzhe Jiang, Chang Yang, Aixin Cui, Sihan Jin, Ruiyu Wang, Bo Li, Xiao Huang, Dongning Sun, and Xinrun Wang. 2025. Finmaster: A holistic benchmark for mastering full-pipeline financial workflows with llms. arXiv preprint arXiv:2505.13533.

Yang Lei, Jiangtong Li, Dawei Cheng, Zhijun Ding, and Changjun Jiang. 2023. Cfbenchmark: Chinese financial assistant benchmark for large language model. arXiv preprint arXiv:2311.05812.

Xiang Li, Zhenyu Li, Chen Shi, Yong Xu, Qing Du, Mingkui Tan, and Jun Huang. 2024a. Alphafin: Benchmarking financial analysis with retrievalaugmented stock-chain framework. In Proceedings of the 2024 joint international conference on computational linguistics, language resources and evaluation (LREC-COLING 2024), pages 773–783.

Xiangyu Li, Xinjie Shen, Yawen Zeng, Xiaofen Xing, and Jin Xu. 2024b. Finreport: Explainable stock earnings forecasting via news factor analyzing model. In Companion Proceedings of the ACM Web Conference 2024, pages 319–327.

Che Liu, Yingji Zhang, Dong Zhang, Weijie Zhang, Chenggong Gong, Haohan Li, Yu Lu, Shilin Zhou, Yue Lu, Ziliang Gan, and 1 others. 2025a. Nexus-o: An omni-perceptive and-interactive model for language, audio, and vision. arXiv preprint arXiv:2503.01879.

Zhaowei Liu, Xin Guo, Fangqi Lou, Lingfeng Zeng, Jinyi Niu, Zixuan Wang, Jiajie Xu, Weige Cai, Ziwei Yang, Xueqian Zhao, and 1 others. 2025b. Finr1: A large language model for financial reasoning through reinforcement learning. arXiv preprint arXiv:2503.16252.

Guilong Lu, Xiaolin Ju, Xiang Chen, Wenlong Pei, and Zhilong Cai. 2024. Grace: Empowering llm-based software vulnerability detection with graph structure and in-context learning. Journal of Systems and Software, 212:112031.

Dat Mai. 2024. Stockgpt: A genai model for stock prediction and trading. arXiv preprint arXiv:2404.05101.

Hangyin Mao, Ying Jiang, Xiaohan Lai, Yehua Zhang, and Xiaoxiao Huang. 2025. Enhancing public opinion monitoring for social hot events with a time series neural network-based logic map. Measurement and Control, 58(7):870–880.

Moonshot. 2025. kimi-k2. https://www.moonshot. cn//.

OpenAI. 2023. Gpt-4 technical report. Preprint, arXiv:2303.08774.

Lingfei Qian, Weipeng Zhou, Yan Wang, Xueqing Peng, Han Yi, Yilun Zhao, Jimin Huang, Qianqian Xie, and Jian-yun Nie. 2025. Fino1: On the transferability of reasoning-enhanced llms and reinforcement learning to finance. arXiv preprint arXiv:2502.08127.

Glenn Shafer and Vladimir Vovk. 2008. A tutorial on conformal prediction. Journal of Machine Learning Research, 9(3).

G Suresh. 2024. Impact of financial literacy and behavioural biases on investment decision-making. FIIB Business Review, 13(1):72–86.

Zichen Tang, E Haihong, Ziyan Ma, Haoyang He, Jiacheng Liu, Zhongjun Yang, Zihua Rong, Rongjin Li, Kun Ji, Qing Huang, and 1 others. 2025. Financereasoning: Benchmarking financial numerical reasoning more credible, comprehensive and challenging. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15721–15749.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, and 1 others. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

InternLM Team. 2023. Internlm: A multilingual language model with progressively enhanced capabilities.

XuanYuan Team. 2024. Xuanyuan3-70b report.

Saizhuo Wang, Hang Yuan, Lionel M Ni, and Jian Guo. 2024. Quantagent: Seeking holy grail in trading by self-improving large language model. arXiv preprint arXiv:2402.03755.

Saizhuo Wang, Hang Yuan, Leon Zhou, Lionel Ni, Heung Yeung Shum, and Jian Guo. 2025. Alpha-gpt: Human-ai interactive alpha mining for quantitative investment. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 196–206.

Ezhilan Wilson, Anshul Saxena, Jayant Mahajan, Lekha Panikulangara, Shruti Kulkarni, and Pritty Jain. 2024. Fin2sum: advancing ai-driven financial text summarization with llms. In 2024 International Conference on Trends in Quantum Computing and Emerging Business Technologies, pages 1–5. IEEE.

xAI. 2025. grok. https://grok.com.

Qianqian Xie, Weiguang Han, Zhengyu Chen, Ruoyu Xiang, Xiao Zhang, Yueru He, Mengxi Xiao, Dong Li, Yongfu Dai, Duanyu Feng, and 1 others. 2024. Finben: A holistic financial benchmark for large language models. Advances in Neural Information Processing Systems, 37:95716–95743.

Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, and Jimin Huang. 2023. Pixiu: A large language model, instruction data and evaluation benchmark for finance. arXiv preprint arXiv:2306.05443.

Yi Yang, Yixuan Tang, and Kar Yan Tam. 2023. Investlm: A large language model for investment using financial domain instruction tuning. arXiv preprint arXiv:2309.13064.

Pengfei Yu and Xuesong Yan. 2020. Stock price prediction based on deep neural networks. Neural Computing and Applications, 32(6):1609–1628.

Yangyang Yu, Haohang Li, Zhi Chen, Yuechen Jiang, Yang Li, Jordan W Suchow, Denghui Zhang, and Khaldoun Khashanah. 2025. Finmem: A performance-enhanced llm trading agent with layered memory and character design. IEEE Transactions on Big Data.

Yangyang Yu, Zhiyuan Yao, Haohang Li, Zhiyang Deng, Yuechen Jiang, Yupeng Cao, Zhi Chen, Jordan Suchow, Zhenyu Cui, Rong Liu, and 1 others. 2024. Fincon: A synthesized llm multi-agent system with conceptual verbal reinforcement for enhanced financial decision making. Advances in Neural Information Processing Systems, 37:137010–137045.

Boyu Zhang, Hongyang Yang, and Xiao-Yang Liu. 2023a. Instruct-fingpt: Financial sentiment analysis by instruction tuning of general-purpose large language models. arXiv preprint arXiv:2306.12659.

Rongjunchen Zhang, Tingmin Wu, Xiao Chen, Sheng Wen, Surya Nepal, Cecile Paris, and Yang Xiang. 2023b. Dynalogue: A transformer-based dialogue system with dynamic attention. In Proceedings of the ACM Web Conference 2023, pages 1604–1615.

Xuanyu Zhang, Bingbing Li, and Qing Yang. 2023c. Cgce: A chinese generative chat evaluation benchmark for general and financial domains. arXiv preprint arXiv:2305.14471.

Yilun Zhao, Hongjun Liu, Yitao Long, Rui Zhang, Chen Zhao, and Arman Cohan. 2024. Financemath: Knowledge-intensive math reasoning in finance domains. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12841–12858.

Jie Zhu, Qian Chen, Huaixia Dou, Junhui Li, Lifan Guo, Feng Chen, and Chi Zhang. 2025. Dianjin-r1: Evaluating and enhancing financial reasoning in large language models. arXiv preprint arXiv:2504.15716.

Jie Zhu, Junhui Li, Yalong Wen, and Lifan Guo. 2024. Benchmarking large language models on cflue–a chinese financial language understanding evaluation dataset. arXiv preprint arXiv:2405.10542.

### A More Details of BizFinBench.v2

In this section, we mainly provide supplementary explanations and example demonstrations for the ten core tasks. The following text contains detailed descriptions and corresponding examples for each task. Given that most questions involve a large volume of real-world data, leading to excessively long

input content that is difficult to present in full, we have adopted an ellipsis approach for partial data in all specific examples (including error analysis examples). Additionally, all example demonstrations are uniformly presented in English.

For the tasks of Abnormal Information Tracing, Financial Multi-turn Perception, and Financial Data Description, we have summarized two main types of questions for each. The first type directly requires the LLM to search for relevant (correct) information or data based on user queries. In contrast, the second type requires the LLM to identify irrelevant (incorrect) information or data based on user queries. Given the overall similarity in the structure of these questions, we only present one type of question herein. For specific examples of the three tasks, please refer to Figure 5, Figure 6 and Figure 7.

In the Financial Quantitative Computation task, we provide the model with either relevant or irrelevant formulas in the input. The model can perform multi-step calculations based on one or more formulas combined with its own capabilities to generate the final result. For detailed examples, please refer to Figure 8.

The Event Logical Reasoning task is based on various financial events such as macro policies and industry markets, requiring the LLM to conduct temporal or causal reasoning on the impact paths and transmission logic of various events on enterprises, industries, and markets. We mainly set the number of logical events to 4, 5, 6, 7, and 8, covering both macro and micro events, with the corresponding difficulty increasing in line with the number of events. For detailed examples, please refer to Figure 9.

The Counterfactual Inference task is based on various assumptions about policies or outcomes in the financial industry, requiring the LLM to perform reasoning or calculations on counterfactual results. For detailed examples, please refer to Figure 10.

The User Sentiment Analysis task requires the LLM to conduct a quantitative analysis of the user’s current emotional state based on the provided userrelated information, market and broader market conditions, as well as relevant news and information, and output the corresponding sentiment score for the user. For detailed examples, please refer to Figure 11.

The Financial Report Analysis task requires the LLM to perform a comprehensive analysis of the

complete financial statements of several enterprises and conduct industry rankings. The number of corporate financial reports used for comparison is primarily set at 2, 3, 4, and 5, with the difficulty increasing accordingly as the number of reports grows. A detailed example is provided in

- Figure 12.

For the Stock Price Prediction task, we primarily utilized the closing prices of 100 popular stocks from both Chinese and U.S. equity markets between November 5, 2025, and December 24, 2025, as the ground truth. The model is provided with relevant information from the preceding month (including individual opening and closing prices, turnover rates, relevant news, and trending forum posts) as input, requiring it to predict the closing prices of these individual stocks. Additionally, a 1% tolerance is established, based on which the LLM outputs a specific prediction interval for the stock’s closing price. By verifying whether the prediction interval encompasses the ground truth, we evaluate the LLM’s capability to analyze the impact of various heterogeneous data sources on individual stocks. A detailed example is shown in

- Figure 13.

The Portfolio Asset Allocation task provides the LLM with specific background requirements and capital constraints, requiring it to make investment decisions based on real-time market conditions to maximize profits. This task spans the time frame from November 19 to December 24, 2025, mandating the LLM to execute one operation per hour during the trading hours of each business day to determine whether to make an investment (with non-investment also recorded as one operation). Ultimately, the LLM’s investment capability in real financial markets is evaluated using relevant metrics derived from the capital changes resulting from all its operations, including the cumulative return rate, maximum drawdown, and Sharpe ratio. Figure 14 displays the detailed page of the real-time investment website, presenting information such as the real-time asset changes and operation summaries of various commercial models. Figure 15 showcases the detailed performance of a representative model, DeepSeek-R1, including its current holdings, trading outputs, and buy-sell curves. Meanwhile, Figure 16 presents the system prompt for the entire online task.

### B Evaluation Details

- B.1 Additional Evaluation Settings

In addition to the zero-shot setting, we also incorporated evaluations under the Chain-of-Thought (CoT) setting, which requires the model to conduct step-by-step reasoning first before generating the final answer. This setup is designed to assess the model’s capabilities in multi-step reasoning and explanation. For specific results, please refer to Appendix C.

- B.2 Model details

Table 6 presents detailed information on the 21 evaluation models used in this study.

### C Evaluation Results and Analysis Under the CoT Setting

The results under the Chain-of-Thought (CoT) setting are presented in Table 4. In stark contrast to the zero-shot setting, the application of the CoT paradigm not only failed to boost model performance but also led to a decline in the overall performance of the vast majority of models, with some models suffering particularly significant performance degradation. Taking Claude-Sonnet-4 as an example, its average accuracy plummeted sharply from the original 37.2% to 13.7%. This abnormal fluctuation fully highlights the model’s obvious shortcomings in reasoning-oriented tasks such as FQC and CI. In comparison, DeepSeek-V3.2 achieved counter-trend performance activation under this setting: its average accuracy increased by nearly 10% compared with the zero-shot setting, and the maximum performance gap between it and various high-performing closed-source models narrowed significantly from nearly 20% originally to within 6%. This result directly confirms that DeepSeek-V3.2 has stronger adaptability to CoT tasks. Overall, the impact of the CoT paradigm on the performance of different large language models in the financial domain shows distinct differentiation characteristics. It does not have a universally applicable optimization effect on all models; instead, it amplifies the inherent flaws of most models in reasoning logic construction, while fully unleashing the potential performance of a small number of models with strong adaptability. This phenomenon can provide an important reference for the targeted optimization of subsequent models and the scenario-based application of reasoning paradigms.

### D More Details of Error Analysis

This section mainly elaborates on the error types from a business perspective in the error analysis, and further identifies the unresolved business problems exhibited by the model during the evaluation process by presenting specific error examples.

Financial Semantic Deviation refers to the phenomenon where, when analyzing real-world financial information, a model fails to accurately grasp the specific implications and impacts of key terms, numerical relationships, or dynamic changes within actual business scenarios. Its outputs are confined to surface-level data, lacking the ability to capture the underlying business connotations, dynamic correlations, and decision-making weights. Such errors do not stem from factual inaccuracies but rather from the model’s insufficient capacity to perceive and interpret the complex business logic, market conventions, and decision-making intentions behind the data. As a result, the model’s outputs often contain directional misguidance or substantive flaws from a professional perspective. Figure 17 presents the model outputs corresponding to this type of error and provides specific explanations for the underlying causes of the errors.

Long-term Business Logic Discontinuity refers to the phenomenon where large language models (LLMs) struggle to maintain a complete, coherent, and business rule-compliant logical chain when handling complex business analyses that require long-cycle and multi-step continuous deduction, such as cross-financial-report comparisons and macro-policy transmission analysis. Its typical manifestations include causal inversion, missing key variables, or broken correlations during the reasoning process, which ultimately lead to the final conclusion deviating from the trajectory of correct business logic. For detailed error examples, please refer to Figure 18.

Multivariate Integrated Analysis Deviation refers to the phenomenon where, in tasks that require the integration of multi-dimensional and multi-source information for comprehensive judgment, the model struggles to effectively identify, weigh, and integrate the complex correlations and weak signals among different information sources. Its flaws are mainly reflected in the insensitivity to critical synergistic or divergent patterns, which results in one-sided analysis conclusions or conclusions inconsistent with the comprehensive context. Figure 19 presents specific examples.

High-precision Computational Distortion refers to the phenomenon where, when confronted with the high-precision numerical calculation requirements unique to the financial field, the model fails to stably and reliably perform complex operations or quantitative deductions. This deviation is not only reflected in errors in numerical results but, more critically, in the failure of its calculation processes and logic to meet the standards of financialgrade rigor and reliability, thereby undermining the feasibility of subsequent analysis and decisionmaking. The examples in Figure 20 provide more specific illustrations.

Financial Time-Series Logical Disorder refers to the phenomenon where, when analyzing information involving time series or event sequences, the model fails to accurately identify and follow the critical chronological order and causal correlations embedded within them. It manifests as confusion in temporal relationships or disorganization in causal chains, leading to conclusions derived from temporal or logical deduction that deviate significantly from the development context and inherent laws of actual business operations. This error category is explained in greater specificity through the examples featured in Figure 21.

### E Additional Case Study

To further corroborate the performance shortcomings of some proprietary models in specific task scenarios, we conducted a dedicated case study on counterfactual inference tasks under the zeroshot setting, presenting the response results of three models, Grok-4, DeepSeek-V3.2 and Qwen3-32B, to the same question respectively. See Figure 22 for specific examples and analyses.

- Table 4: LLM performance in CoT settings on BizFinBench.v2 (%). The results are color-coded to indicate the top three performers in each task: gold represents the best-performing model, silver represents the second-best result, and bronze represents the third-best performance.. The meanings of the abbreviations are as follows: AIT(Anomaly Information Tracing), FMP(Financial Multi-turn Perception), FDD(Financial Data Description), FQC(Financial Quantitative Computation), ELR(Event Logic Reasoning), CI(Counterfactual Inference), SA(User Sentiment Analysis), FRA(Financial Report Analysis), SPP(Stock Price Prediction).

Model Size AIT FMP FDD FQC ELR CI SA FRA SPP Average Propretary LLMs

Doubao-Seed-1.6 unknown 64.3 88.1 61.2 79.2 60.0 75.7 20.3 47.8 6.7 56.9 Gemini-3 unknown 67.8 91.8 54.9 84.6 67.9 80.6 3.7 51.0 0.9 56.5 ChatGPT-5 unknown 54.1 88.6 49.3 86.7 59.5 83.6 8.8 55.7 6.0 54.6 Kimi-k2 unknown 63.7 81.4 27.2 40.8 45.0 42.4 17.8 48.3 2.6 40.1 Grok-4 unknown 63.2 88.7 39.2 20.5 43.5 20.2 18.3 44.7 7.7 37.7 Claude-Sonnet-4 unknown 21.8 39.3 13.4 9.0 13.1 8.1 2.9 48.3 1.6 13.7

Open-source General LLMs

Qwen3-235B-A22B-Thinking-2507 235B 48.4 88.5 63.6 75.3 58.4 70.6 12.8 21.7 4.2 52.7 DeepSeek-V3.2 685B 62.7 88.2 41.2 54.7 50.4 56.7 21.3 45.5 33.8 51.1 Qwen3-32B 32B 55.1 79.7 12.9 44.7 41.4 43.9 9.3 39.8 3.8 36.4 Qwen2.5-72B-Instruct 72B 60.9 82.0 31.0 28.8 39.6 27.2 13.0 45.2 7.6 36.3 DeepSeek-R1-Distill-Qwen-32B 32B 54.4 77.3 22.1 19.7 40.6 20.2 17.8 27.3 6.7 32.4 GLM-Z1-32B 32B 50.9 62.4 42.8 32.7 28.8 30.6 0.8 35.7 6.8 32.0 GLM-Z1-9B 9B 45.9 72.9 41.3 24.8 32.7 25.2 0.6 36.2 3.4 30.9 Qwen2.5-7B-Instruct 7B 26.1 49.4 31.9 4.9 21.1 5.0 16.3 28.5 2.5 19.6 InternLM2.5-20B 20B 33.0 58.1 17.4 1.7 29.4 2.0 3.0 33.7 0.4 18.1 DeepSeek-R1-Distill-Qwen-7B 7B 14.4 30.3 16.0 3.0 10.6 3.8 6.5 13.0 0.9 10.7 InternLM2.5-7B 7B 26.0 23.8 4.8 1.7 19.3 0.7 5.8 27.0 0.1 10.2

Open-source Financial LLMs

Dianjin-R1 32B 53.0 76.6 39.7 18.0 41.3 20.0 5.6 30.0 5.8 32.5 FinX1 70B 45.8 74.4 25.3 8.5 32.0 7.1 2.5 34.7 3.0 24.8 Fino1 14B 36.5 58.4 0.0 13.5 28.2 8.9 12.8 28.5 8.8 20.9 Fin-R1 7B 22.7 30.8 35.7 1.8 10.7 3.6 7.5 24.2 0.6 14.2

- Table 5: Detailed statistics of BizFinBench.v2 data. For the specific meanings corresponding to the abbreviations, please refer to Section 3.2.

#### Scenarios Tasks Avg. Input Tokens #Questions

Business Information Provenance Anomaly Information Tracing 8679 4000 Financial Multi-turn Perception 10361 3741 Financial Data Description 3577 3837

Financial Logic Reasoning Financial Quantitative Computation 1984 2000 Event Logic Reasoning 437 4000 Counterfactual Inference 2267 2000

Stakeholder Feature Perception User Sentiment Analysis 3326 4000

Financial Report Analysis 19681 2000 Real-time Market Discernment Stock Price Prediction 5510 4000

Portfolio Asset Allocation – –

#### All – – 29578

[Figure 6]

- Figure 5: An Example of Abnormal Information Tracing. This task requires the LLM to identify relevant information from various given heterogeneous sources that caused the stock price fluctuation of the corresponding company and to provide the associated information index. It evaluates the LLM’s information analysis and summarization capabilities. Due to the extensive length of the data, we have truncated the content here and primarily present a sample of the input data.

- Table 6: Models evaluated in this paper. The "Access" column shows whether we have full access to the model weights or we can access through API. The “Version Date” column shows the release date of the corresponding version of the model we evaluated.

Category Model Creator Parameter Access Version Date Domain Proprietary ChatGPT-5 OpenAI Undisclosed API 2025.11 General

Gemini-3 Google Undisclosed API 2025.11 General Kimi-k2 MoonshotAI Undisclosed API 2025.11 General Claude-Sonnet-4 Anthropic Undisclosed API 2025.9 General Doubao-Seed-1.6 ByteDance Undisclosed API 2025.6 General Grok-4 X AI Undisclosed API 2025.7 General

Open-Source Qwen2.5-7B-Instruct Alibaba Cloud 7B Weights 2024.9 General Qwen2.5-72B-Instruct Alibaba Cloud 72B Weights 2024.9 General Qwen3-32B Alibaba Cloud 32B Weights 2025.4 General Qwen3-235B-A22B-Thinking-2507 Alibaba Cloud 235B Weights 2025.4 General InternLM2.5-7B Shanghai AI Laboratory 7B Weights 2025.3 General InternLM2.5-20B Shanghai AI Laboratory 20B Weights 2025.3 General GLM-Z1-9B ZhipuAI 9B Weights 2025.4 General GLM-Z1-32B ZhipuAI 32B Weights 2025.4 General DeepSeek-R1-Distill-Qwen-7B DeepSeek AI 7B Weights 2025.2 General DeepSeek-R1-Distill-Qwen-32B DeepSeek AI 32B Weights 2025.2 General DeepSeek-V3.2 DeepSeek AI 685B Weights 2025.12 General Fin-R1 Shanghai University of Finance and Economics 7B Weights 2025.3 Financial FinX1 Duxiaoman-DI 70B Weights 2024.12 Financial Dianjin-R1 Alibaba Cloud 32B Weights 2025.4 Financial Fino1 The Fin AI 14B Weights 2025.2 Financial

[Figure 7]

- Figure 6: Example of the Financial Multi-turn Perception task. This task requires the LLM to identify the descriptions corresponding to specific dialogues based on three consecutive user questions and the platform AI assistant’s responses; it primarily examines the LLM’s capacity for memory and summarization regarding user queries; similarly, we have omitted portions of the data here. Within the ground truth, the "Description Number" field corresponds to the index of the descriptive text, while the "answer" field corresponds to the dialogue index.

[Figure 8]

- Figure 7: Example of the Financial Data Description task. This task requires the LLM to evaluate the logic and numerical values within a data description list based on an existing data list. We primarily demonstrate a reverse-lookup question here, which requires the LLM to identify texts within the data description list that contain obvious errors and provide a formatted output.

[Figure 9]

- Figure 8: Example of the Financial Quantitative Computation task. This task requires the model to retrieve relevant formulas from a formula list and extract pertinent data from a data list to perform calculations that answer user queries. It is important to note that the provided formulas and data may be either relevant or irrelevant; consequently, some questions may be unanswerable. The model must exercise careful judgment and possess a sufficiently deep understanding of the financial domain to provide correct answers. Furthermore, as most of the numerical values are derived from actual financial statements or research reports and involve a high volume of data points, the requirements for the precision of the model’s computational capabilities are extremely high.

[Figure 10]

###### Figure 9: Example of the Event Logical Reasoning task. This task requires the LLM to arrange various financial events in their logical chronological order based on the user’s query. It primarily assesses whether the LLM possesses the capacity for logical reasoning regarding diverse events and whether it has relevant domain expertise.

[Figure 11]

- Figure 10: Example of the Counterfactual Inference task. This task requires the LLM to perform counterfactual reasoning based on existing industry or policy-oriented data by introducing counterfactual hypotheses. It aims to prompt the LLM to simulate the cognitive patterns of financial experts, conducting deliberation and reasoning grounded in established knowledge and industry experience.

[Figure 12]

- Figure 11: Example of the User Sentiment Analysis task. This task requires the LLM to evaluate a user’s emotional state by utilizing input regarding user-specific information (user profile, user queries, user holdings, and transaction status), macro market information, as well as relevant news and research reports. It assesses the LLM’s capability to grasp the holding attitudes and investment sentiments of individual stock users.

[Figure 13]

###### Figure 12: Example of the Financial Report Analysis task. This task requires the LLM to rank multiple companies within the same industry based on financial statement data. It primarily evaluates the LLM’s capability to integrate and analyze various data points from complete financial reports.

[Figure 14]

- Figure 13: The Stock Price Prediction task primarily requires the LLM to perform online predictions of individual stock closing prices based on multidimensional information, including fundamentals, technical indicators, and market sentiment. It evaluates the LLM’s capability to analyze and summarize the directional impact of multidimensional data on individual stocks.

[Figure 15]

- Figure 14: Display of the Portfolio Asset Allocation task webpage. The Portfolio Asset Allocation task primarily involves accessing online market data interfaces, allowing the LLM to autonomously determine investment strategies based on dynamic real-world industry data. It mainly evaluates the LLM’s dynamic market perception and decisionmaking capabilities.

[Figure 16]

###### Figure 15: Display of Investment Details for DeepSeek-R1

[Figure 17]

###### Figure 16: System Prompt for Portfolio Asset Allocation

[Figure 18]

- Figure 17: Example of Financial Semantic Deviation. On the one hand, the model erroneously forced a strong correlation between the industry focus on semiconductor equipment and materials sectors highlighted in Content 3, and the consumer electronics/smart hardware track to which Insta360’s core businesses—panoramic cameras and action cameras—belong. Relying solely on the generalized logic of "both falling under the technology sector," the model overlooked the fundamental differences between the two in terms of business chains, core products, and profit models. On the other hand, the model failed to recognize the financial value directly driving stock prices embedded in Content 6 (core competitiveness data showing Insta360’s global No.1 market share in panoramic cameras), Content 12 (synergistic benefits from a high overseas revenue ratio and RMB appreciation), and Content 13 (strategic growth logic behind the construction of the Shenzhen R&D center). Instead, it excluded these critical pieces of information based on one-sided justifications such as "early news release time," "non-independent driving factors," and "no evidence of new progress." Ultimately, while completely ignoring the company’s own key positive factors, the model arbitrarily identified the short-term market trend of the semiconductor sector— which has extremely low relevance to the company’s business—as the core catalyst for the sharp rise in Insta360’s stock price on July 1, 2025.

[Figure 19]

- Figure 18: Example of Long-term Business Logic Discontinuity. In its rating of Company C, the model delivered a suboptimal conclusion. The core issue lay in its excessive overemphasis on the weight of two positive indicators—"small-scale positive profitability" and "active dividend distribution"—while failing to attach sufficient importance to the critical risk factors inherent to the company. In the case of its medium rating for Company D, the model, on the one hand, placed undue weight on neutral indicators reflecting corporate stability, such as "medium scale" and "low asset-liability ratio." On the other hand, it irrationally overstated the negative effects of "sustained revenue decline," a short-term volatility factor, yet overlooked the company’s core competitive advantages.

[Figure 20]

- Figure 19: Example of Multivariate Integrated Analysis Deviation. The model’s output yielded an overly pessimistic sentiment score due to issues such as misjudgment of core behavioral motivations, misjudgment of information relevance, and imbalance in scoring logic. Specifically, it misinterpreted the strategic questions posed by trend investors regarding position adjustment as concerns about the safety of their holdings, ignoring their core attribute of "prioritizing strategic judgment over short-term panic." The model excessively amplified the negative impact of historical financial information about Ruichen Environmental Protection’s impairment provision and forced an unwarranted correlation between this information and the current stock price trend. Meanwhile, it assigned excessively high weights to negative factors such as minor floating losses on holdings and short-term declines in individual stocks, while severely underestimating the positive buffer effect of the broader market breaking through the 4,000-point mark. Both the baseline setting and the calculation of adjustment scores deviated significantly from the actual sentiment logic of trend investors.

[Figure 21]

- Figure 20: Example of High-precision Computational Distortion. While the model correctly selected the compound growth rate formula and made no errors in the unit conversion of the 2022 and 2024 data, its core mistakes lay in data judgment and the execution of calculation logic. Specifically, the model erroneously classified the net profit attributable to parent company shareholders of Canadian Solar Inc. in 2023 (as presented in Reference Data 3) as "abnormal." In reality, this data could be standardized through unit conversion and was not invalid. Meanwhile, when calculating the compound growth rate, the model incorrectly treated the 3-year time span from 2022 to 2024 as a 2-year period. Furthermore, it failed to perform the calculation by combining the correct initial and final period data with the complete time cycle. These errors ultimately led to a severe deviation in the resulting value.

[Figure 22]

- Figure 21: Example of Financial Time-Series Logical Disorder. The model misjudged the market environment event (Event 4) as the cause of technological innovation, reversing the causal relationship. In reality, Event 3 (algorithm breakthrough) serves as the technological root cause, with Event 1 (application platform) and Event 2 (performance improvement) being its direct outcomes. Event 4, by contrast, represents a macro-level manifestation or background condition rather than a driving force. This error stems from the model’s introduction of external assumptions not mentioned in the task requirements.

[Figure 23]

- Figure 22: Answers from Grok-4, DeepSeek-V3.2 and Qwen3-32B to the same question in CI tasks. As evidenced by the response from Grok-4, the model correctly extracted the relevant figures from Reference Data 4 and performed accurate reasoning and calculations. However, it suffered from severe hallucinations when adhering to the required output format, leading to a final answer that was entirely inconsistent with its own logical derivation. Simultaneously, the response from DeepSeek-V3.2 indicates that it made no attempt to leverage its capabilities for comprehension or reasoning, opting instead to output the default "unable to answer" response. The outputs of these two models represent two of the most typical categories of model errors: correct reasoning paired with an incorrect final answer, and concluding a problem is unanswerable without prior deliberation. Conversely, Qwen3-32B demonstrated comprehensive thinking and reasoning, accurately retrieving data from the input, performing the correct calculations, and delivering the output perfectly.

